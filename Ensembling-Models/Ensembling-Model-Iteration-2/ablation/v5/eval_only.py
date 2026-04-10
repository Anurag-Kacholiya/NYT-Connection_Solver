"""
eval_only.py — Load saved models and run evaluation on VAL + TEST.

Use this after train.py has completed to re-run evaluation without retraining.

V5 defaults:
  - Full V4 features ON (via train.extract_train_features / evaluate_split)
  - Ranker OFF
  - LR fusion OFF
  - Auto-train RF+GBM if missing
  - V5 output folders/files

Usage:
    python eval_only.py
    python eval_only.py --csv Connections_Data.csv --models-dir models_v5
"""
import os
import sys
import argparse
import pickle
import json

from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier

from data_loader import load_games
from embedder import ConnectionsEmbedder, WikidataForbiddenError
from evaluate import build_feature_names, save_feature_weights
from train import (
    evaluate_split,
    extract_train_features,
)


def _build_split(all_games, val_size: int, test_size: int):
    """
    Chronological split from the tail:
      - test = last test_size
      - val  = previous val_size from remaining
      - train = all earlier games
    """
    n = len(all_games)
    if test_size <= 0 or val_size <= 0:
        raise ValueError("val_size and test_size must be positive integers")
    if val_size + test_size >= n:
        raise ValueError(
            f"Invalid split sizes: val({val_size}) + test({test_size}) must be < total games ({n})"
        )

    test_start = n - test_size
    val_start = test_start - val_size

    train_games = all_games[:val_start]
    val_games = all_games[val_start:test_start]
    test_games = all_games[test_start:]
    return train_games, val_games, test_games, val_start, test_start


def _train_rf_gbm_if_needed(train_games, embedder, models_dir, output_dir):
    """
    Train V5 RF+GBM models on train split using full V4 features and save them.
    """
    os.makedirs(models_dir, exist_ok=True)
    os.makedirs(output_dir, exist_ok=True)

    print("\nRF/GBM models missing. Training V5 models from scratch...")
    print("  Features: V4 full feature set ON (Numberbatch, Lexical, MultiSense, PairContext, KGs)")

    X_train, y_train, mat_names = extract_train_features(train_games, embedder)
    feature_names = build_feature_names(mat_names)

    clf_rf = RandomForestClassifier(
        n_estimators=200, max_depth=10, random_state=42, n_jobs=-1)
    clf_rf.fit(X_train, y_train)

    clf_gbm = GradientBoostingClassifier(
        n_estimators=200, max_depth=3, learning_rate=0.1,
        subsample=0.8, random_state=42)
    clf_gbm.fit(X_train, y_train)

    rf_path = os.path.join(models_dir, "clf_rf.pkl")
    gbm_path = os.path.join(models_dir, "clf_gbm.pkl")
    with open(rf_path, "wb") as f:
        pickle.dump(clf_rf, f)
    with open(gbm_path, "wb") as f:
        pickle.dump(clf_gbm, f)

    weights_path = os.path.join(output_dir, "v5_feature_weights.txt")
    save_feature_weights(clf_rf, clf_gbm, feature_names,
                         output_path=weights_path)
    print(f"  Saved RF model      -> {rf_path}")
    print(f"  Saved GBM model     -> {gbm_path}")
    print(f"  Saved feature ranks -> {weights_path}")


def _save_outputs(output_dir, run_name, val_games, test_games,
                  val_game_acc, val_group_acc, test_game_acc, test_group_acc,
                  use_ranker, use_lr_fusion):
    os.makedirs(output_dir, exist_ok=True)

    summary_path = os.path.join(output_dir, f"{run_name}_summary.txt")
    metrics_path = os.path.join(output_dir, f"{run_name}_metrics.json")

    summary_lines = [
        "=" * 60,
        "  FINAL SUMMARY",
        "=" * 60,
        f"Run Name       : {run_name}",
        f"Use Ranker     : {use_ranker}",
        f"Use LR Fusion  : {use_lr_fusion}",
        "-" * 60,
        f"{'Split':<8}  {'Games':>6}  {'Game Acc':>9}  {'Group Acc':>10}",
        "-" * 45,
        f"{'Val':<8}  {len(val_games):>6}  {val_game_acc*100:>8.2f}%  {val_group_acc*100:>9.2f}%",
        f"{'Test':<8}  {len(test_games):>6}  {test_game_acc*100:>8.2f}%  {test_group_acc*100:>9.2f}%",
        "=" * 60,
    ]
    with open(summary_path, "w", encoding="utf-8") as f:
        f.write("\n".join(summary_lines) + "\n")

    metrics = {
        "run_name": run_name,
        "use_ranker": use_ranker,
        "use_lr_fusion": use_lr_fusion,
        "val": {
            "games": len(val_games),
            "game_accuracy": val_game_acc,
            "group_accuracy": val_group_acc,
        },
        "test": {
            "games": len(test_games),
            "game_accuracy": test_game_acc,
            "group_accuracy": test_group_acc,
        },
    }
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    print(f"Summary saved -> {summary_path}")
    print(f"Metrics saved -> {metrics_path}")


def main(csv_path: str, models_dir: str | None, output_dir: str, run_name: str,
         use_ranker: bool, use_lr_fusion: bool, train_if_missing: bool,
         val_size: int, test_size: int):
    # ── Load data ─────────────────────────────────────────────────────────
    print("Loading data...")
    all_games = load_games(csv_path)
    train_games, val_games, test_games, val_start, test_start = _build_split(
        all_games, val_size=val_size, test_size=test_size
    )
    print(f"  Total: {len(all_games)} games")
    print(f"  Train: {len(train_games)} games  [0:{val_start}]")
    print(f"  Val  : {len(val_games)} games  [{val_start}:{test_start}]")
    print(f"  Test : {len(test_games)} games  [{test_start}:{len(all_games)}]")

    os.makedirs(output_dir, exist_ok=True)

    # Keep all generated artifacts in one folder by default.
    # If models_dir is omitted, models are saved/loaded from output_dir.
    if not models_dir:
        models_dir = output_dir
    os.makedirs(models_dir, exist_ok=True)

    # ── Embedder ──────────────────────────────────────────────────────────
    print("\nInitialising embedder...")
    embedder = ConnectionsEmbedder()

    # ── Load models ───────────────────────────────────────────────────────
    print("\nLoading saved models...")
    paths = {
        "rf": os.path.join(models_dir, "clf_rf.pkl"),
        "gbm": os.path.join(models_dir, "clf_gbm.pkl"),
        "ranker": os.path.join(models_dir, "clf_lgbm_ranker.pkl"),
        "lr_fusion": os.path.join(models_dir, "clf_lr_fusion.pkl"),
    }

    rf_missing = not os.path.exists(paths["rf"])
    gbm_missing = not os.path.exists(paths["gbm"])
    if rf_missing or gbm_missing:
        if not train_if_missing:
            print("  ERROR: RF/GBM models missing and --no-train-if-missing was set.")
            print(f"    Missing RF : {rf_missing}")
            print(f"    Missing GBM: {gbm_missing}")
            sys.exit(1)
        _train_rf_gbm_if_needed(train_games, embedder, models_dir, output_dir)

    with open(paths["rf"],        "rb") as f:
        clf_rf = pickle.load(f)
    with open(paths["gbm"],       "rb") as f:
        clf_gbm = pickle.load(f)

    clf_ranker = None
    if use_ranker:
        if not os.path.exists(paths["ranker"]):
            print(f"  ERROR: ranker requested but missing: {paths['ranker']}")
            sys.exit(1)
        with open(paths["ranker"], "rb") as f:
            clf_ranker = pickle.load(f)

    clf_lr_fusion = None
    if use_lr_fusion:
        if not os.path.exists(paths["lr_fusion"]):
            print(
                f"  ERROR: lr_fusion requested but missing: {paths['lr_fusion']}")
            sys.exit(1)
        with open(paths["lr_fusion"], "rb") as f:
            clf_lr_fusion = pickle.load(f)

    print("  Required models loaded.")
    print(f"  Ranker enabled   : {use_ranker}")
    print(f"  LR fusion enabled: {use_lr_fusion}")

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

    _save_outputs(
        output_dir=output_dir,
        run_name=run_name,
        val_games=val_games,
        test_games=test_games,
        val_game_acc=val_game_acc,
        val_group_acc=val_group_acc,
        test_game_acc=test_game_acc,
        test_group_acc=test_group_acc,
        use_ranker=use_ranker,
        use_lr_fusion=use_lr_fusion,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", type=str, default="Connections_Data.csv")
    parser.add_argument(
        "--models-dir",
        type=str,
        default=None,
        help="Optional. If omitted, models are saved in --output-dir",
    )
    parser.add_argument("--output-dir", type=str, default="results/v5")
    parser.add_argument("--run-name", type=str, default="v5_rf_gbm_only")
    parser.add_argument("--val-size", type=int, default=173)
    parser.add_argument("--test-size", type=int, default=100)

    # V5 defaults: ranker OFF, lr-fusion OFF
    parser.add_argument("--with-ranker", dest="use_ranker",
                        action="store_true")
    parser.add_argument("--with-lr-fusion",
                        dest="use_lr_fusion", action="store_true")
    parser.set_defaults(use_ranker=False, use_lr_fusion=False)

    # V5 convenience: train RF+GBM automatically when missing
    parser.add_argument(
        "--no-train-if-missing",
        action="store_true",
        help="Do not auto-train RF+GBM if missing in models-dir",
    )

    args = parser.parse_args()

    if not os.path.exists(args.csv):
        print(f"Error: {args.csv} not found")
        sys.exit(1)

    main(
        csv_path=args.csv,
        models_dir=args.models_dir,
        output_dir=args.output_dir,
        run_name=args.run_name,
        use_ranker=args.use_ranker,
        use_lr_fusion=args.use_lr_fusion,
        train_if_missing=not args.no_train_if_missing,
        val_size=args.val_size,
        test_size=args.test_size,
    )
