"""
eval_only.py — Load saved models and run evaluation on VAL + TEST.

Use this after train.py has completed to re-run evaluation without retraining.

Usage:
    python eval_only.py
    python eval_only.py --csv Connections_Data.csv --models-dir models
"""
import os, sys, argparse, pickle

from data_loader import load_games
from embedder import ConnectionsEmbedder, WikidataForbiddenError
from train import evaluate_split, TRAIN_END, VAL_END

def main(csv_path: str, models_dir: str):
    # ── Load data ─────────────────────────────────────────────────────────
    print("Loading data...")
    all_games  = load_games(csv_path)
    val_games  = all_games[TRAIN_END:VAL_END]
    test_games = all_games[VAL_END:]
    print(f"  Val : {len(val_games)} games  |  Test : {len(test_games)} games")

    # ── Load models ───────────────────────────────────────────────────────
    print("\nLoading saved models...")
    paths = {
        "rf":       os.path.join(models_dir, "clf_rf.pkl"),
        "gbm":      os.path.join(models_dir, "clf_gbm.pkl"),
        "ranker":   os.path.join(models_dir, "clf_lgbm_ranker.pkl"),
        "lr_fusion":os.path.join(models_dir, "clf_lr_fusion.pkl"),
    }
    for name, path in paths.items():
        if not os.path.exists(path):
            print(f"  ERROR: {path} not found. Run train.py first.")
            sys.exit(1)

    with open(paths["rf"],        "rb") as f: clf_rf        = pickle.load(f)
    with open(paths["gbm"],       "rb") as f: clf_gbm       = pickle.load(f)
    with open(paths["ranker"],    "rb") as f: clf_ranker    = pickle.load(f)
    with open(paths["lr_fusion"], "rb") as f: clf_lr_fusion = pickle.load(f)
    print("  All models loaded.")

    # ── Embedder ──────────────────────────────────────────────────────────
    print("\nInitialising embedder...")
    embedder = ConnectionsEmbedder()

    # ── Evaluate ──────────────────────────────────────────────────────────
    print("\n" + "="*60)
    print("  EVALUATION  (models are frozen — no training data used)")
    print("="*60)

    try:
        val_game_acc, val_group_acc = evaluate_split(
            val_games, embedder, clf_rf, clf_gbm,
            split_name="VAL",
            clf_ranker=clf_ranker,
            clf_lr_fusion=clf_lr_fusion,
        )
        test_game_acc, test_group_acc = evaluate_split(
            test_games, embedder, clf_rf, clf_gbm,
            split_name="TEST",
            clf_ranker=clf_ranker,
            clf_lr_fusion=clf_lr_fusion,
        )
    except WikidataForbiddenError as e:
        print(f"\nCRITICAL ERROR: {e}")
        print("Stopping run to avoid further Wikidata API block. Run again later.")
        sys.exit(1)

    # ── Summary ───────────────────────────────────────────────────────────
    print("\n" + "="*60)
    print("  FINAL SUMMARY")
    print("="*60)
    print(f"{'Split':<8}  {'Games':>6}  {'Game Acc':>9}  {'Group Acc':>10}")
    print("-"*45)
    print(f"{'Val':<8}  {len(val_games):>6}  {val_game_acc*100:>8.2f}%  {val_group_acc*100:>9.2f}%")
    print(f"{'Test':<8}  {len(test_games):>6}  {test_game_acc*100:>8.2f}%  {test_group_acc*100:>9.2f}%")
    print("="*60)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv",        type=str, default="Connections_Data.csv")
    parser.add_argument("--models-dir", type=str, default="models")
    args = parser.parse_args()

    if not os.path.exists(args.csv):
        print(f"Error: {args.csv} not found")
        sys.exit(1)

    main(args.csv, args.models_dir)
