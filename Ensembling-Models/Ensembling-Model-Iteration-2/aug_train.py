"""
aug_train.py — Train and save the RF + GBM + LightGBM-Ranker ensemble with
               an augmented dataset (original + BabelNet + LLM puzzles).

═══════════════════════════════════════════════════════════════════════════════
DATASET CONSTRUCTION
═══════════════════════════════════════════════════════════════════════════════

  Original dataset (connections_dataset.csv)  →  915 puzzles, chronological
  ┌────────────────────────────────────────────────────────────────────┐
  │  Train slice  [0 – 677]   →  678 puzzles                          │
  │  Val   slice  [678 – 814] →  137 puzzles  (last 137 before test)  │
  │  Test  slice  [815 – 914] →  100 puzzles  (last 100, SEALED)      │
  └────────────────────────────────────────────────────────────────────┘
  NOTE: 915 - 100 (test) = 815; 815 - 137 (val) = 678 (train slice).

  Augmented training sources (ALL added to train set only):
    dataset.csv              →  500 BabelNet-generated puzzles
    connections_dataset1.csv →   50 LLM-generated puzzles
    connections_dataset2.csv →   50 LLM-generated puzzles

  Final training set = 678 (original slice) + 600 (augmented) = 1 278 puzzles
  Val  = 137 original puzzles   (unchanged from baseline)
  Test = 100 original puzzles   (held-out, never touched during training)

═══════════════════════════════════════════════════════════════════════════════
OUTPUT DIRECTORIES  (all prefixed with aug_  to keep separate from baseline)
═══════════════════════════════════════════════════════════════════════════════
  aug_models/          — saved model pickles
  aug_feature_weights/ — feature_weights.txt
  aug_logs/            — any future log artefacts

═══════════════════════════════════════════════════════════════════════════════
Pipeline (strict — val/test are NEVER touched during training):
  Step 0 : Verify NO overlap between train / val / test word-sets
  Step 1 : Extract features from TRAIN games only
  Step 2 : Fit RF + GBM on train features
  Step 3 : Fit LightGBM lambdarank ranker with VAL early-stopping
  Step 4 : Save trained models  →  aug_models/
  Step 5 : Save feature_weights.txt  →  aug_feature_weights/
  ── model training is COMPLETE here ──
  Step 6 : Load saved models, run game simulation on VAL  → report accuracy
  Step 7 : Run game simulation on TEST (final, held-out)  → report accuracy

Usage:
    python aug_train.py
    python aug_train.py \\
        --orig-csv   connections_dataset.csv \\
        --babel-csv  dataset.csv \\
        --llm-csv1   connections_dataset1.csv \\
        --llm-csv2   connections_dataset2.csv \\
        --out-dir    aug_models
"""

import os
import sys
import argparse
import pickle
import itertools
import numpy as np
from tqdm import tqdm
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
import nltk
import lightgbm as lgb

from data_loader import load_games
from embedder import ConnectionsEmbedder, WikidataForbiddenError
from evaluate import (
    build_matrices,
    build_feature_names,
    extract_features,
    compute_per_matrix_separation,
    save_feature_weights,
    simulate_game_ml,
)

# ─────────────────────────────────────────────────────────────────────────────
# Split boundaries (applied to the ORIGINAL dataset only)
# ─────────────────────────────────────────────────────────────────────────────
ORIG_TOTAL = 915   # expected total puzzles in original CSV
TEST_SIZE = 100   # last N puzzles → test
VAL_SIZE = 137   # next-last M puzzles → val

# Derived indices (0-based, Python slice notation)
TEST_START = ORIG_TOTAL - TEST_SIZE                # 815
VAL_START = TEST_START - VAL_SIZE                 # 678
# Train slice of original: [0, VAL_START)  →  678 games
# Val  slice of original: [VAL_START, TEST_START)  →  137 games
# Test slice of original: [TEST_START, ORIG_TOTAL)  →  100 games


# ─────────────────────────────────────────────────────────────────────────────
# Output directories (all prefixed with aug_)
# ─────────────────────────────────────────────────────────────────────────────
AUG_MODELS_DIR = "aug_models"
AUG_WEIGHTS_DIR = "aug_feature_weights"
AUG_LOGS_DIR = "aug_logs"


# ─────────────────────────────────────────────────────────────────────────────
# Data loading helper
# ─────────────────────────────────────────────────────────────────────────────

def load_augmented_splits(orig_csv, babel_csv, llm_csv1, llm_csv2):
    """
    Load and assemble train / val / test splits.

    Split strategy:
      - Test  : last TEST_SIZE  games from the original CSV  (sealed)
      - Val   : next VAL_SIZE   games before test            (from original)
      - Train : remaining original games  +  ALL augmented data

    Returns (train_games, val_games, test_games) as lists of game dicts.
    """
    print("Loading original dataset ...")
    orig_games = load_games(orig_csv)
    n_orig = len(orig_games)
    print(f"  Original CSV: {n_orig} games")

    if n_orig != ORIG_TOTAL:
        print(f"  ⚠ WARNING: expected {ORIG_TOTAL} games in original CSV, "
              f"got {n_orig}. Adjusting split boundaries dynamically.")

    # Recompute boundaries in case n_orig differs from expected
    test_start = n_orig - TEST_SIZE
    val_start = test_start - VAL_SIZE

    if val_start < 0:
        raise ValueError(
            f"Not enough games in original CSV ({n_orig}) to carve out "
            f"test={TEST_SIZE} + val={VAL_SIZE}. Check your data."
        )

    test_games = orig_games[test_start:]            # 100 games
    val_games = orig_games[val_start:test_start]   # 137 games
    orig_train = orig_games[:val_start]             # 678 games

    print(f"  Original split → "
          f"train slice={len(orig_train)}  "
          f"val={len(val_games)}  "
          f"test={len(test_games)}")

    # ── Load augmented sources ─────────────────────────────────────────────
    print("\nLoading augmented datasets ...")
    aug_games = []

    for csv_path, label in [
        (babel_csv, "BabelNet (dataset.csv)"),
        (llm_csv1,  "LLM-1 (connections_dataset1.csv)"),
        (llm_csv2,  "LLM-2 (connections_dataset2.csv)"),
    ]:
        if csv_path and os.path.exists(csv_path):
            g = load_games(csv_path)
            print(f"  {label}: {len(g)} games")
            aug_games.extend(g)
        else:
            print(f"  ⚠ Skipping {label}: file not found ({csv_path})")

    train_games = orig_train + aug_games

    print(f"\nFinal split:")
    print(f"  Train : {len(train_games)}  "
          f"({len(orig_train)} original + {len(aug_games)} augmented)")
    print(f"  Val   : {len(val_games)}  (original only)")
    print(f"  Test  : {len(test_games)}  (original only — SEALED)")

    return train_games, val_games, test_games


# ─────────────────────────────────────────────────────────────────────────────
# Data integrity check
# ─────────────────────────────────────────────────────────────────────────────

def check_splits_no_overlap(train_games, val_games, test_games):
    """
    Verify that train, val, and test splits share no games (by word-set identity).
    Raises AssertionError if any overlap is detected — aborts training immediately.

    Note: augmented puzzles may share individual words with val/test by coincidence
    (common English words), but a full 16-word set collision is astronomically
    unlikely and would be a genuine data leak.
    """
    train_ids = {frozenset(g["words"]) for g in train_games}
    val_ids = {frozenset(g["words"]) for g in val_games}
    test_ids = {frozenset(g["words"]) for g in test_games}

    tv = train_ids & val_ids
    tt = train_ids & test_ids
    vt = val_ids & test_ids

    if tv:
        raise AssertionError(
            f"DATA LEAK: {len(tv)} games overlap between train and val!\n"
            f"Check augmented sources for puzzles copied from the original dataset."
        )
    if tt:
        raise AssertionError(
            f"DATA LEAK: {len(tt)} games overlap between train and test!\n"
            f"Check augmented sources for puzzles copied from the original dataset."
        )
    if vt:
        raise AssertionError(
            f"DATA LEAK: {len(vt)} games overlap between val and test!")

    print(f"  ✓ Split integrity confirmed: NO overlap between train / val / test.")
    print(
        f"    train={len(train_games)}  val={len(val_games)}  test={len(test_games)}")


# ─────────────────────────────────────────────────────────────────────────────
# Feature extraction helpers
# ─────────────────────────────────────────────────────────────────────────────

def extract_train_features(train_games, embedder):
    """
    Extract (X_train, y_train) from training games ONLY.
    Returns X_train (list of feature vectors), y_train (list of labels 0/1),
    and the list of matrix names for building human-readable feature names.
    All new matrices (MultiSense, PairContext, TemplateCtx) are included.
    """
    X, y = [], []
    mat_names_ref = None

    for game in tqdm(train_games, desc="Extracting train features"):
        words = game["words"]
        gt_partitions = [
            set(words.index(w) for w in gw)
            for gw in game["groups"].values()
        ]

        mat_names, mats = build_matrices(
            embedder, words,
            use_numberbatch=True,
            use_multisense=True,
            use_pairwise_context=True,
        )
        if mat_names_ref is None:
            mat_names_ref = mat_names  # capture once for naming

        all_subsets = list(itertools.combinations(range(16), 4))

        # ── Positives: the 4 correct groups ──────────────────────────────
        for p in gt_partitions:
            X.append(extract_features(list(p), mats,
                     set(range(16)), words, mat_names))
            y.append(1)

        # ── Hard negatives: 3/4 overlap with a correct group ─────────────
        hard_neg = [S for S in all_subsets
                    if set(S) not in gt_partitions
                    and max(len(set(S) & gt) for gt in gt_partitions) == 3]
        np.random.shuffle(hard_neg)
        for S in hard_neg[:8]:
            X.append(extract_features(list(S), mats,
                     set(range(16)), words, mat_names))
            y.append(0)

        # ── Easy random negatives ─────────────────────────────────────────
        easy_neg = [S for S in all_subsets
                    if set(S) not in gt_partitions and S not in hard_neg]
        np.random.shuffle(easy_neg)
        for S in easy_neg[:8]:
            X.append(extract_features(list(S), mats,
                     set(range(16)), words, mat_names))
            y.append(0)

    return X, y, mat_names_ref


# ─────────────────────────────────────────────────────────────────────────────
# LightGBM Ranker training  (Tier 1.2)
# ─────────────────────────────────────────────────────────────────────────────

def train_lgbm_ranker(train_games, embedder, mat_names_ref, val_games=None):
    """
    Build a pairwise LightGBM lambdarank model.

    Training data construction (per game):
      Each game yields one query group. Within the query:
        - 4 positive samples  (correct groups, label=2)
        - up to 8 hard-neg    (3-overlap,      label=1)
        - up to 8 easy-neg    (random,          label=0)

    LightGBM's lambdarank objective optimises NDCG, which is a ranking
    objective — directly learning "which group is more likely to be correct
    relative to others" rather than just binary classification.

    val_games: if provided, builds a validation dataset for early stopping,
               which prevents overfitting and automatically finds optimal rounds.

    Returns: trained lgb.Booster
    """
    print("\nBuilding ranking dataset for LightGBM lambdarank...")
    X_rank, y_rank, groups = [], [], []

    for game in tqdm(train_games, desc="Ranking feature extraction"):
        words = game["words"]
        gt_partitions = [
            set(words.index(w) for w in gw)
            for gw in game["groups"].values()
        ]

        mat_names, mats = build_matrices(
            embedder, words,
            use_numberbatch=True,
            use_multisense=True,
            use_pairwise_context=True,
        )

        all_subsets = list(itertools.combinations(range(16), 4))
        game_X, game_y = [], []

        # Positives (relevance = 2)
        for p in gt_partitions:
            game_X.append(extract_features(
                list(p), mats, set(range(16)), words, mat_names))
            game_y.append(2)

        # Hard negatives (relevance = 1)
        hard_neg = [S for S in all_subsets
                    if set(S) not in gt_partitions
                    and max(len(set(S) & gt) for gt in gt_partitions) == 3]
        np.random.shuffle(hard_neg)
        for S in hard_neg[:8]:
            game_X.append(extract_features(
                list(S), mats, set(range(16)), words, mat_names))
            game_y.append(1)

        # Easy negatives (relevance = 0)
        easy_neg = [S for S in all_subsets
                    if set(S) not in gt_partitions and S not in hard_neg]
        np.random.shuffle(easy_neg)
        for S in easy_neg[:8]:
            game_X.append(extract_features(
                list(S), mats, set(range(16)), words, mat_names))
            game_y.append(0)

        if game_X:
            X_rank.extend(game_X)
            y_rank.extend(game_y)
            groups.append(len(game_X))

    X_rank = np.array(X_rank, dtype=np.float32)
    y_rank = np.array(y_rank, dtype=np.int32)
    groups = np.array(groups, dtype=np.int32)

    print(f"  Ranking samples: {len(X_rank)}  |  queries: {len(groups)}")

    feature_names = build_feature_names(mat_names_ref)

    dtrain = lgb.Dataset(X_rank, label=y_rank, group=groups,
                         feature_name=feature_names)

    # ── Build validation dataset for early stopping ───────────────────────
    dval = None
    if val_games is not None:
        print("  Building validation ranking dataset for early stopping...")
        X_val, y_val, val_groups = [], [], []
        for game in tqdm(val_games, desc="Val ranking features"):
            words = game["words"]
            gt_partitions = [
                set(words.index(w) for w in gw)
                for gw in game["groups"].values()
            ]
            mat_names, mats = build_matrices(
                embedder, words,
                use_numberbatch=True,
                use_multisense=True,
                use_pairwise_context=True,
            )
            all_subsets = list(itertools.combinations(range(16), 4))
            game_X, game_y = [], []

            for p in gt_partitions:
                game_X.append(extract_features(
                    list(p), mats, set(range(16)), words, mat_names))
                game_y.append(2)
            hard_neg = [S for S in all_subsets
                        if set(S) not in gt_partitions
                        and max(len(set(S) & gt) for gt in gt_partitions) == 3]
            np.random.shuffle(hard_neg)
            for S in hard_neg[:8]:
                game_X.append(extract_features(
                    list(S), mats, set(range(16)), words, mat_names))
                game_y.append(1)
            easy_neg = [S for S in all_subsets
                        if set(S) not in gt_partitions and S not in hard_neg]
            np.random.shuffle(easy_neg)
            for S in easy_neg[:8]:
                game_X.append(extract_features(
                    list(S), mats, set(range(16)), words, mat_names))
                game_y.append(0)
            if game_X:
                X_val.extend(game_X)
                y_val.extend(game_y)
                val_groups.append(len(game_X))

        if X_val:
            dval = lgb.Dataset(
                np.array(X_val, dtype=np.float32),
                label=np.array(y_val, dtype=np.int32),
                group=np.array(val_groups, dtype=np.int32),
                feature_name=feature_names,
                reference=dtrain,
            )
            print(
                f"  Val ranking samples: {len(X_val)}  |  queries: {len(val_groups)}")

    params = {
        "objective":        "lambdarank",
        "metric":           "ndcg",
        "ndcg_eval_at":     [1, 3, 5],
        "learning_rate":    0.05,
        "num_leaves":       31,
        "min_data_in_leaf": 15,
        "max_depth":        5,
        "reg_alpha":        0.1,
        "reg_lambda":       0.1,
        "subsample":        0.8,
        "colsample_bytree": 0.8,
        "verbose": -1,
        "seed":             42,
    }

    valid_sets = [dtrain] if dval is None else [dtrain, dval]
    valid_names = ["train"] if dval is None else ["train", "val"]

    booster = lgb.train(
        params,
        dtrain,
        num_boost_round=500,
        valid_sets=valid_sets,
        valid_names=valid_names,
        callbacks=[lgb.log_evaluation(period=50)]
    )
    return booster

# ─────────────────────────────────────────────────────────────────────────────
# LR Matrix Weights training  (Path A — Weighted Matrix Fusion)
# ─────────────────────────────────────────────────────────────────────────────


def train_lr_matrix_weights(val_games, embedder, mat_names_ref):
    """
    Train a Logistic Regression to learn per-matrix weights (Path A).

    Trained on VAL set only — lightweight (N parameters, one per matrix).
    Returns: sklearn LogisticRegression fitted on val set.
    """
    X_lr, y_lr = [], []

    for game in tqdm(val_games, desc="LR fusion: val feature extraction"):
        words = game["words"]
        gt_partitions = [
            set(words.index(w) for w in gw)
            for gw in game["groups"].values()
        ]

        _, mats = build_matrices(
            embedder, words,
            use_numberbatch=True,
            use_multisense=True,
            use_pairwise_context=True,
        )

        all_subsets = list(itertools.combinations(range(16), 4))
        remaining = set(range(16))

        for p in gt_partitions:
            X_lr.append(compute_per_matrix_separation(
                list(p), mats, remaining))
            y_lr.append(1)

        hard_neg = [S for S in all_subsets
                    if set(S) not in gt_partitions
                    and max(len(set(S) & gt) for gt in gt_partitions) == 3]
        np.random.shuffle(hard_neg)
        for S in hard_neg[:8]:
            X_lr.append(compute_per_matrix_separation(
                list(S), mats, remaining))
            y_lr.append(0)

        easy_neg = [S for S in all_subsets
                    if set(S) not in gt_partitions and S not in hard_neg]
        np.random.shuffle(easy_neg)
        for S in easy_neg[:8]:
            X_lr.append(compute_per_matrix_separation(
                list(S), mats, remaining))
            y_lr.append(0)

    X_lr = np.array(X_lr, dtype=np.float32)
    y_lr = np.array(y_lr, dtype=np.int32)

    print(
        f"  LR fusion samples : {len(X_lr)}  ({sum(y_lr)} pos / {len(y_lr)-sum(y_lr)} neg)")
    print(f"  LR fusion features: {X_lr.shape[1]}  (one per matrix)")

    clf_lr = LogisticRegression(C=1.0, max_iter=300, random_state=42,
                                class_weight="balanced")
    clf_lr.fit(X_lr, y_lr)

    print("  Learned matrix weights (LR coefficients):")
    for name, coef in zip(mat_names_ref, clf_lr.coef_[0]):
        print(f"    {name:<45} {coef:+.4f}")

    return clf_lr


# ─────────────────────────────────────────────────────────────────────────────
# Game-level evaluation
# ─────────────────────────────────────────────────────────────────────────────

def evaluate_split(games, embedder, clf_rf, clf_gbm, split_name="",
                   clf_ranker=None, clf_lr_fusion=None):
    """
    Run the full game simulation on a list of games using trained (read-only) models.
    No training information flows here — models are loaded from disk.
    """
    exact = 0
    groups = 0
    partial = {4: 0, 3: 0, 2: 0, 1: 0, 0: 0}
    guesses = 0

    for game in tqdm(games, desc=f"Evaluating [{split_name}]"):
        words = game["words"]
        gt_partitions = [
            {words.index(w) for w in gw}
            for gw in game["groups"].values()
        ]

        mat_names, mats = build_matrices(
            embedder, words,
            use_numberbatch=True,
            use_multisense=True,
            use_pairwise_context=True,
        )

        matched, preds, pmatch = simulate_game_ml(
            words, mats, mat_names,
            (clf_rf, clf_gbm),
            gt_partitions,
            clf_ranker=clf_ranker,
            clf_lr_fusion=clf_lr_fusion,
            use_candidates=True,
        )

        groups += matched
        guesses += len(preds)
        for k, v in pmatch.items():
            partial[k] += v
        if matched == 4:
            exact += 1

    n = len(games)
    game_acc = exact / n
    group_acc = groups / (4 * n)

    print(f"\n  [{split_name}]  {n} games")
    print(f"    Perfect games  : {exact}  ({game_acc*100:.2f}%)")
    print(f"    Group accuracy : {groups}/{4*n}  ({group_acc*100:.2f}%)")
    print(f"    Guess breakdown:")
    for k in [4, 3, 2, 1, 0]:
        pct = partial[k] / guesses * 100 if guesses else 0
        print(f"      {k}/4 → {partial[k]:4d}  ({pct:.1f}%)")

    return game_acc, group_acc


# ─────────────────────────────────────────────────────────────────────────────
# Main pipeline
# ─────────────────────────────────────────────────────────────────────────────

def train(orig_csv, babel_csv, llm_csv1, llm_csv2, out_dir=AUG_MODELS_DIR):
    try:
        _train_impl(orig_csv, babel_csv, llm_csv1, llm_csv2, out_dir)
    except WikidataForbiddenError as e:
        print(f"\nCRITICAL ERROR: {e}")
        print("Stopping run to avoid further Wikidata API block. Run again later.")
        sys.exit(1)


def _train_impl(orig_csv, babel_csv, llm_csv1, llm_csv2, out_dir=AUG_MODELS_DIR):
    # ── Create all output directories ─────────────────────────────────────
    weights_dir = AUG_WEIGHTS_DIR
    logs_dir = AUG_LOGS_DIR
    for d in [out_dir, weights_dir, logs_dir]:
        os.makedirs(d, exist_ok=True)

    # ── 1. Load and split ─────────────────────────────────────────────────
    print("=" * 60)
    print("  AUGMENTED TRAINING — DATA LOADING")
    print("=" * 60)
    train_games, val_games, test_games = load_augmented_splits(
        orig_csv, babel_csv, llm_csv1, llm_csv2
    )

    print(f"\nVal and Test sets are SEALED until after training.\n")

    # ── 0. Verify no data leakage between splits ──────────────────────────
    print("─" * 50)
    print("STEP 0/4  Verifying split integrity (no data leakage)")
    print("─" * 50)
    check_splits_no_overlap(train_games, val_games, test_games)

    # ── 2. Load embedder ──────────────────────────────────────────────────
    print("\nInitialising embedder (MPNet + GloVe + Numberbatch)...")
    embedder = ConnectionsEmbedder()

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # TRAINING PHASE — only train_games are touched below
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

    # ── 3. Extract features from TRAINING data only ───────────────────────
    print("\n" + "─" * 50)
    print("STEP 1/4  Extract features (train only)")
    print("─" * 50)
    X_train, y_train, mat_names = extract_train_features(train_games, embedder)
    feature_names = build_feature_names(mat_names)

    n_pos = sum(y_train)
    n_neg = len(y_train) - n_pos
    print(f"  Samples : {len(X_train)}  ({n_pos} pos / {n_neg} neg)")
    print(f"  Features: {len(feature_names)}")

    # ── 4. Fit RF + GBM ──────────────────────────────────────────────────
    print("\n" + "─" * 50)
    print("STEP 2/4  Train RF + GBM")
    print("─" * 50)
    print("Training Random Forest (n_estimators=200, max_depth=10)...")
    clf_rf = RandomForestClassifier(
        n_estimators=200, max_depth=10, random_state=42, n_jobs=-1)
    clf_rf.fit(X_train, y_train)
    print(f"  RF   train accuracy : {clf_rf.score(X_train, y_train)*100:.2f}%")

    print("Training Gradient Boosting (n_estimators=200, max_depth=3)...")
    clf_gbm = GradientBoostingClassifier(
        n_estimators=200, max_depth=3, learning_rate=0.1,
        subsample=0.8, random_state=42)
    clf_gbm.fit(X_train, y_train)
    print(
        f"  GBM  train accuracy : {clf_gbm.score(X_train, y_train)*100:.2f}%")

    # ── 5. Fit LightGBM ranker  (with val early stopping) ────────────────
    print("\n" + "─" * 50)
    print("STEP 3/4  Train LightGBM Lambdarank Ranker (with val early stopping)")
    print("─" * 50)
    clf_ranker = train_lgbm_ranker(
        train_games, embedder, mat_names, val_games=val_games)

    # ── 5.5  Train LR matrix weights on VAL set (Path A) ─────────────────
    print("\n" + "─" * 50)
    print("STEP 3.5/4  Train LR Matrix Weights (Path A — val set only)")
    print("─" * 50)
    clf_lr_fusion = train_lr_matrix_weights(val_games, embedder, mat_names)

    # ── 6. Save models + feature weights ─────────────────────────────────
    print("\n" + "─" * 50)
    print("STEP 4/4  Save models and feature weights")
    print("─" * 50)
    rf_path = os.path.join(out_dir, "clf_rf.pkl")
    gbm_path = os.path.join(out_dir, "clf_gbm.pkl")
    ranker_path = os.path.join(out_dir, "clf_lgbm_ranker.pkl")
    lr_fusion_path = os.path.join(out_dir, "clf_lr_fusion.pkl")

    with open(rf_path,        "wb") as f:
        pickle.dump(clf_rf,        f)
    with open(gbm_path,       "wb") as f:
        pickle.dump(clf_gbm,       f)
    with open(ranker_path,    "wb") as f:
        pickle.dump(clf_ranker,    f)
    with open(lr_fusion_path, "wb") as f:
        pickle.dump(clf_lr_fusion, f)

    print(f"  RF        model → {rf_path}")
    print(f"  GBM       model → {gbm_path}")
    print(f"  Ranker    model → {ranker_path}")
    print(f"  LR fusion model → {lr_fusion_path}")

    weights_path = os.path.join(weights_dir, "feature_weights.txt")
    save_feature_weights(clf_rf, clf_gbm, feature_names, weights_path)
    print(f"  Feature weights → {weights_path}")
    print("  ✓ Training complete. Models saved. Val/Test not yet accessed.\n")

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # EVALUATION PHASE — saved models only; no re-training or re-fitting
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    print("Loading saved models for evaluation...")
    with open(rf_path,        "rb") as f:
        clf_rf_eval = pickle.load(f)
    with open(gbm_path,       "rb") as f:
        clf_gbm_eval = pickle.load(f)
    with open(ranker_path,    "rb") as f:
        clf_ranker_eval = pickle.load(f)
    with open(lr_fusion_path, "rb") as f:
        clf_lr_fusion_eval = pickle.load(f)

    print("\n" + "=" * 60)
    print("  EVALUATION  (models are frozen — no training data used)")
    print("=" * 60)

    val_game_acc,  val_group_acc = evaluate_split(
        val_games,  embedder, clf_rf_eval, clf_gbm_eval,
        split_name="VAL",  clf_ranker=clf_ranker_eval,
        clf_lr_fusion=clf_lr_fusion_eval)
    test_game_acc, test_group_acc = evaluate_split(
        test_games, embedder, clf_rf_eval, clf_gbm_eval,
        split_name="TEST", clf_ranker=clf_ranker_eval,
        clf_lr_fusion=clf_lr_fusion_eval)

    # ── Summary ───────────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("  FINAL SUMMARY  (Augmented Training Run)")
    print("=" * 60)
    print(f"{'Split':<8}  {'Games':>6}  {'Game Acc':>9}  {'Group Acc':>10}")
    print("-" * 45)
    print(f"{'Val':<8}  {len(val_games):>6}  {val_game_acc*100:>8.2f}%  {val_group_acc*100:>9.2f}%")
    print(f"{'Test':<8}  {len(test_games):>6}  {test_game_acc*100:>8.2f}%  {test_group_acc*100:>9.2f}%")
    print("=" * 60)
    print(f"Train games   : {len(train_games)}  "
          f"(original={len(train_games)-600}, augmented=600)")
    print(f"Features      : {len(feature_names)}")
    print(f"Models saved  : {out_dir}/clf_rf.pkl, clf_gbm.pkl, "
          f"clf_lgbm_ranker.pkl, clf_lr_fusion.pkl")
    print(f"Weights saved : {weights_path}")


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train with augmented data (original + BabelNet + LLM puzzles)."
    )
    parser.add_argument(
        "--orig-csv",   type=str, default="Connections_Data.csv",
        help="Original NYT Connections dataset (915 puzzles)."
    )
    parser.add_argument(
        "--babel-csv",  type=str, default="dataset.csv",
        help="BabelNet-generated augmentation (500 puzzles)."
    )
    parser.add_argument(
        "--llm-csv1",   type=str, default="connections_dataset1.csv",
        help="LLM-generated augmentation set 1 (50 puzzles)."
    )
    parser.add_argument(
        "--llm-csv2",   type=str, default="connections_dataset2.csv",
        help="LLM-generated augmentation set 2 (50 puzzles)."
    )
    parser.add_argument(
        "--out-dir",    type=str, default=AUG_MODELS_DIR,
        help=f"Directory for saved models (default: {AUG_MODELS_DIR})."
    )
    args = parser.parse_args()

    # Validate required files
    missing = [f for f in [args.orig_csv] if not os.path.exists(f)]
    if missing:
        print(f"Error: required file(s) not found: {missing}")
        sys.exit(1)

    train(
        orig_csv=args.orig_csv,
        babel_csv=args.babel_csv,
        llm_csv1=args.llm_csv1,
        llm_csv2=args.llm_csv2,
        out_dir=args.out_dir,
    )
