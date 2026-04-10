"""
finetune.py — Contrastive fine-tuning of MPNet using triplet loss.

Uses the 640 training games to construct triplets:
  Anchor   : a word from a group
  Positive : another word from the same group
  Negative : hardest negative — the word from a different group that is
             most similar to the anchor under the current MPNet embeddings

Fine-tunes all-mpnet-base-v2 with TripletLoss (cosine distance, margin=0.2)
and saves the result to models/mpnet_finetuned/.

After running this script, ConnectionsEmbedder will automatically detect and
load the fine-tuned model instead of the off-the-shelf one.

Usage:
    python finetune.py
    python finetune.py --epochs 5 --batch-size 32 --margin 0.2
"""

import os
import argparse
import random
import numpy as np
from tqdm import tqdm

from sentence_transformers import SentenceTransformer, losses
from torch.utils.data import DataLoader

try:
    from sentence_transformers import InputExample
except ImportError:
    from sentence_transformers.readers import InputExample

from data_loader import load_games

# ─────────────────────────────────────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────────────────────────────────────
TRAIN_END        = 640
FINETUNED_PATH   = "models/mpnet_finetuned"
SEED             = 42


# ─────────────────────────────────────────────────────────────────────────────
# Triplet construction
# ─────────────────────────────────────────────────────────────────────────────

def build_triplets(games: list, model: SentenceTransformer,
                   hard_negative_ratio: float = 0.8) -> list:
    """
    Build (anchor, positive, negative) triplets from training games.

    For every word in every group:
      - Positives  : the other 3 words in the same group
      - Hard neg   : word from a different group with highest cosine similarity
                     to the anchor under the current (un-finetuned) model
      - Random neg : any word from a different group (for variety)

    hard_negative_ratio controls the fraction of triplets that use a hard
    negative vs. a random one.

    Returns a list of InputExample objects ready for DataLoader.
    """
    random.seed(SEED)
    triplets = []

    for game in tqdm(games, desc="Building triplets"):
        words  = game["words"]            # 16 words
        groups = game["groups"]           # {0: [w,w,w,w], 1: [...], ...}

        # Encode all 16 words once per game
        embeddings  = model.encode(words, normalize_embeddings=True,
                                   show_progress_bar=False)
        word_to_idx = {w: i for i, w in enumerate(words)}

        # Map every word to its group id
        word_to_gid = {}
        for gid, members in groups.items():
            for w in members:
                word_to_gid[w] = gid

        for gid, members in groups.items():
            # Words that do NOT belong to this group
            outside = [w for w in words if word_to_gid[w] != gid]
            outside_idxs = [word_to_idx[w] for w in outside]
            outside_embs = embeddings[outside_idxs]   # (12, 768)

            for anchor in members:
                anchor_emb = embeddings[word_to_idx[anchor]]   # (768,)

                # Hard negative: highest cosine sim among the 12 outsiders
                sims = outside_embs @ anchor_emb               # (12,)
                hard_neg = outside[int(np.argmax(sims))]

                # Positives: the other 3 group-mates
                positives = [w for w in members if w != anchor]

                for positive in positives:
                    if random.random() < hard_negative_ratio:
                        negative = hard_neg
                    else:
                        negative = random.choice(outside)

                    triplets.append(
                        InputExample(texts=[anchor, positive, negative])
                    )

    return triplets


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Contrastive fine-tuning of MPNet for NYT Connections"
    )
    parser.add_argument("--csv",          default="Connections_Data.csv",
                        help="Path to the Connections dataset CSV")
    parser.add_argument("--model-name",   default="all-mpnet-base-v2",
                        help="Base sentence-transformer model to fine-tune")
    parser.add_argument("--out-dir",      default=FINETUNED_PATH,
                        help="Directory to save the fine-tuned model")
    parser.add_argument("--epochs",       type=int,   default=5)
    parser.add_argument("--batch-size",   type=int,   default=32)
    parser.add_argument("--lr",           type=float, default=2e-5)
    parser.add_argument("--warmup-steps", type=int,   default=100)
    parser.add_argument("--margin",       type=float, default=0.2,
                        help="Triplet margin in cosine space")
    parser.add_argument("--hard-ratio",   type=float, default=0.8,
                        help="Fraction of triplets using hard negatives")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    # ── Load training games ───────────────────────────────────────────────────
    print("Loading games...")
    all_games    = load_games(args.csv)
    train_games  = all_games[:TRAIN_END]
    print(f"Training games: {len(train_games)}")

    # ── Load base model ───────────────────────────────────────────────────────
    print(f"Loading base model: {args.model_name} ...")
    model = SentenceTransformer(args.model_name)

    # ── Build triplets ────────────────────────────────────────────────────────
    print("Building triplets (mining hard negatives with base embeddings)...")
    triplets = build_triplets(train_games, model, hard_negative_ratio=args.hard_ratio)
    print(f"Total triplets: {len(triplets):,}")

    # ── DataLoader ────────────────────────────────────────────────────────────
    train_dataloader = DataLoader(triplets, shuffle=True, batch_size=args.batch_size)

    # ── Loss ──────────────────────────────────────────────────────────────────
    train_loss = losses.TripletLoss(
        model=model,
        distance_metric=losses.TripletDistanceMetric.COSINE,
        triplet_margin=args.margin,
    )

    total_steps = len(train_dataloader) * args.epochs
    print(f"\nFine-tuning: {args.epochs} epochs | "
          f"{total_steps} steps | "
          f"warmup {args.warmup_steps} steps | "
          f"lr {args.lr} | margin {args.margin}\n")

    # ── Fine-tune ─────────────────────────────────────────────────────────────
    model.fit(
        train_objectives=[(train_dataloader, train_loss)],
        epochs=args.epochs,
        warmup_steps=args.warmup_steps,
        optimizer_params={"lr": args.lr},
        show_progress_bar=True,
        output_path=args.out_dir,
    )

    print(f"\nFine-tuned model saved to: {args.out_dir}")
    print("ConnectionsEmbedder will automatically load this model on next run.")


if __name__ == "__main__":
    main()
