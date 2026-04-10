import argparse
import os

# Stabilize macOS threaded libraries
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import pickle
from typing import List, Set

from tqdm import tqdm

from data_loader import load_games
from embedder import ConnectionsEmbedder, WikidataForbiddenError
from evaluate import build_matrices, simulate_game_ml


def load_models(models_dir: str):
    rf_path = os.path.join(models_dir, "clf_rf.pkl")
    gbm_path = os.path.join(models_dir, "clf_gbm.pkl")
    ranker_path = os.path.join(models_dir, "clf_lgbm_ranker.pkl")
    lr_fusion_path = os.path.join(models_dir, "clf_lr_fusion.pkl")

    if not os.path.exists(rf_path):
        raise FileNotFoundError(f"Missing RF model: {rf_path}")
    if not os.path.exists(gbm_path):
        raise FileNotFoundError(f"Missing GBM model: {gbm_path}")

    with open(rf_path, "rb") as f:
        clf_rf = pickle.load(f)
    with open(gbm_path, "rb") as f:
        clf_gbm = pickle.load(f)

    clf_ranker = None
    if os.path.exists(ranker_path):
        with open(ranker_path, "rb") as f:
            clf_ranker = pickle.load(f)

    clf_lr_fusion = None
    if os.path.exists(lr_fusion_path):
        with open(lr_fusion_path, "rb") as f:
            clf_lr_fusion = pickle.load(f)

    return clf_rf, clf_gbm, clf_ranker, clf_lr_fusion


def game_has_two_solved_and_two_one_away(
    matched: int,
    preds: List[Set[int]],
    gt_partitions: List[Set[int]],
) -> bool:
    if matched != 2:
        return False

    solved_gt_indices = set()
    for i, gt in enumerate(gt_partitions):
        if any(pred == gt for pred in preds):
            solved_gt_indices.add(i)

    one_away_unsolved_gt_indices = set()
    for pred in preds:
        if pred in gt_partitions:
            continue

        overlaps = [len(pred & gt) for gt in gt_partitions]
        best_overlap = max(overlaps)
        if best_overlap == 3:
            best_idx = overlaps.index(best_overlap)
            if best_idx not in solved_gt_indices:
                one_away_unsolved_gt_indices.add(best_idx)

    return len(one_away_unsolved_gt_indices) == 2


def format_report(total_groups_solved: int, one_away_groups: int, one_away_games: int, perfect_games: int) -> str:
    return (
        "===== FINAL EVALUATION METRICS =====\n"
        "1) Grouping Accuracy :\n"
        f"   1.1) total no. of group solved: {total_groups_solved}\n"
        f"   1.2) no. of '3 words correct out of 4 in a group': {one_away_groups}\n"
        "____\n"
        "2) no. of games in which (2 groups solved completely and remaining 2 groups are '3 words correct out of 4 in a group'): "
        f"{one_away_games}\n"
        "____\n"
        f"3) no. of games solved completely (out of 100 test games): {perfect_games}\n"
        "_____\n"
    )


def evaluate(
    csv_path: str,
    models_dir: str,
    feature_weights_path: str,
    output_path: str,
):
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"CSV not found: {csv_path}")
    if not os.path.exists(feature_weights_path):
        raise FileNotFoundError(
            f"Feature weights file not found: {feature_weights_path}")

    all_games = load_games(csv_path)
    if len(all_games) < 100:
        raise ValueError(
            f"Expected at least 100 games, found {len(all_games)}")

    test_games = all_games[-100:]

    clf_rf, clf_gbm, clf_ranker, clf_lr_fusion = load_models(models_dir)
    embedder = ConnectionsEmbedder()

    total_groups_solved = 0
    one_away_groups = 0
    one_away_games = 0
    perfect_games = 0

    for game in tqdm(test_games, desc="Evaluating last 100 test games"):
        words = game["words"]
        gt_partitions = [{words.index(w) for w in gw}
                         for gw in game["groups"].values()]

        mat_names, mats = build_matrices(
            embedder,
            words,
            use_numberbatch=True,
            use_multisense=True,
            use_pairwise_context=True,
        )

        matched, preds, pmatch = simulate_game_ml(
            words,
            mats,
            mat_names,
            (clf_rf, clf_gbm),
            gt_partitions,
            clf_ranker=clf_ranker,
            clf_lr_fusion=clf_lr_fusion,
            use_candidates=True,
        )

        total_groups_solved += matched
        one_away_groups += pmatch.get(3, 0)

        if matched == 4:
            perfect_games += 1

        if game_has_two_solved_and_two_one_away(matched, preds, gt_partitions):
            one_away_games += 1

    report = format_report(
        total_groups_solved=total_groups_solved,
        one_away_groups=one_away_groups,
        one_away_games=one_away_games,
        perfect_games=perfect_games,
    )

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(report)

    print(report)
    print(f"Saved report to: {output_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", type=str, default="Connections_Data.csv")
    parser.add_argument("--models-dir", type=str, default="aug_models")
    parser.add_argument("--feature-weights", type=str,
                        default="aug_feature_weights/feature_weights.txt")
    parser.add_argument("--output", type=str,
                        default="aug_result/one_away_metrics.txt")
    args = parser.parse_args()

    try:
        evaluate(
            csv_path=args.csv,
            models_dir=args.models_dir,
            feature_weights_path=args.feature_weights,
            output_path=args.output,
        )
    except WikidataForbiddenError as e:
        print(f"CRITICAL ERROR: {e}")
        print("Stopping run to avoid further Wikidata API block. Run again later.")
        raise SystemExit(1)


if __name__ == "__main__":
    main()
