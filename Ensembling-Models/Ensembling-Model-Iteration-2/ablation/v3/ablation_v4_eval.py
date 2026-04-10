"""
v4_eval_only.py — Load saved V4 models and evaluate on TEST split (last 100 games).

Saves per-game results to a checkpoint file so crashes can be resumed.

Usage:
    python v4_eval_only.py
    python v4_eval_only.py --models-dir ablation_study_results/v4_original_v1 --mode original
"""

import os, sys, argparse, pickle, json

# Must be set before sklearn/joblib import to prevent macOS fork segfault
os.environ['LOKY_MAX_CPU_COUNT'] = '1'
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'

import numpy as np
from tqdm import tqdm

from data_loader import load_games
from embedder import ConnectionsEmbedder, WikidataForbiddenError
from evaluate import build_matrices, extract_features, simulate_game_ml

TEST_SIZE = 100
VAL_SIZE  = 137


def run_eval(games, embedder, clf_rf, clf_gbm, clf_ranker, clf_lr, split_name, checkpoint_path):
    # Load checkpoint if exists
    results = {}
    if os.path.exists(checkpoint_path):
        with open(checkpoint_path) as f:
            results = json.load(f)
        print(f"  Resuming from checkpoint: {len(results)}/{len(games)} games done")

    for i, game in enumerate(tqdm(games, desc=f"[{split_name}]")):
        key = str(i)
        if key in results:
            continue  # already done

        words = game["words"]
        gt_parts = [{words.index(w) for w in gw} for gw in game["groups"].values()]
        try:
            mat_names, mats = build_matrices(
                embedder, words,
                use_numberbatch=True,
                use_multisense=True,
                use_pairwise_context=True,
                use_lexical=True,
                use_knowledge_graphs=True,
            )
            matched, _, _ = simulate_game_ml(
                words, mats, mat_names,
                (clf_rf, clf_gbm), gt_parts,
                clf_ranker=clf_ranker,
                clf_lr_fusion=clf_lr,
            )
        except WikidataForbiddenError as e:
            print(f"\n  Game {i}: Wikidata blocked — {e}. Skipping.")
            matched = 0
        except Exception as e:
            print(f"\n  Game {i}: ERROR — {e}. Skipping.")
            matched = 0

        results[key] = int(matched)
        with open(checkpoint_path, "w") as f:
            json.dump(results, f)

    n = len(games)
    exact = sum(1 for v in results.values() if v == 4)
    total_groups = sum(results.values())
    game_acc  = exact / n
    group_acc = total_groups / (n * 4)

    print(f"\n  [{split_name}] Games solved completely: {exact}/{n}  ({game_acc*100:.2f}%)")
    print(f"  [{split_name}] Groups solved: {total_groups}/{n*4}  ({group_acc*100:.2f}%)")

    # Distribution
    dist = {k: 0 for k in range(5)}
    for v in results.values():
        dist[v] += 1
    print(f"\n  Group match distribution:")
    for k in range(5):
        print(f"    {k}/4 groups correct: {dist[k]} games")

    return game_acc, group_acc


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv",        type=str, default="Connections_Data.csv")
    parser.add_argument("--models-dir", type=str, default="ablation_study_results/v4_original_v1")
    parser.add_argument("--mode",       type=str, default="original")
    args = parser.parse_args()

    if not os.path.exists(args.csv):
        print(f"Error: {args.csv} not found"); sys.exit(1)

    # ── Data ──────────────────────────────────────────────────────────────────
    print(f"Loading data from {args.csv} ...")
    all_games  = load_games(args.csv)
    n_total    = len(all_games)
    test_start = n_total - TEST_SIZE
    test_games = all_games[test_start:]
    print(f"  Test: {len(test_games)} games (indices {test_start}–{n_total-1})")

    # ── Models ────────────────────────────────────────────────────────────────
    m = args.mode
    paths = {
        "rf":        os.path.join(args.models_dir, f"{m}_clf_rf.pkl"),
        "gbm":       os.path.join(args.models_dir, f"{m}_clf_gbm.pkl"),
        "ranker":    os.path.join(args.models_dir, f"{m}_clf_lgbm_ranker.pkl"),
        "lr_fusion": os.path.join(args.models_dir, f"{m}_clf_lr_fusion.pkl"),
    }
    for name, path in paths.items():
        if not os.path.exists(path):
            print(f"  ERROR: {path} not found."); sys.exit(1)

    print(f"\nLoading models from {args.models_dir} ...")
    with open(paths["rf"],        "rb") as f: clf_rf        = pickle.load(f)
    with open(paths["gbm"],       "rb") as f: clf_gbm       = pickle.load(f)
    with open(paths["ranker"],    "rb") as f: clf_ranker    = pickle.load(f)
    with open(paths["lr_fusion"], "rb") as f: clf_lr_fusion = pickle.load(f)
    clf_rf.n_jobs = 1  # prevent macOS fork/parallel segfault
    print("  All models loaded.")

    # ── Embedder ──────────────────────────────────────────────────────────────
    print("\nInitialising embedder ...")
    embedder = ConnectionsEmbedder()

    # ── Evaluate ──────────────────────────────────────────────────────────────
    checkpoint = os.path.join(args.models_dir, f"{m}_eval_checkpoint.json")
    print(f"\n{'='*60}")
    print("  V4 TEST EVALUATION  (models frozen)")
    print(f"{'='*60}")

    test_game_acc, test_group_acc = run_eval(
        test_games, embedder, clf_rf, clf_gbm, clf_ranker, clf_lr_fusion,
        "TEST", checkpoint
    )

    print(f"\n{'='*60}")
    print("  FINAL SUMMARY")
    print(f"{'='*60}")
    print(f"  Test Game  Accuracy : {test_game_acc*100:.2f}%  ({int(test_game_acc*100)} / 100 games fully solved)")
    print(f"  Test Group Accuracy : {test_group_acc*100:.2f}%  ({int(test_group_acc*400)} / 400 groups solved)")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
