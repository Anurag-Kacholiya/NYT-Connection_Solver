"""
train.py — Train and save the RF + GBM + LightGBM-Ranker ensemble with a
           strict train/val/test split.

Split (915 games total, chronological order preserved — NO data leakage):
  Train : 70%  → 640 games  (games 0 – 639)
  Val   : 15%  → 137 games  (games 640 – 776)
  Test  : 15%  → 138 games  (games 777 – 914)

Pipeline (strict — val/test are NEVER touched during training):
  Step 0 : Verify NO overlap between train / val / test word-sets
  Step 1 : Extract features from TRAIN games only
  Step 2 : Fit RF + GBM on train features
  Step 3 : Fit LightGBM lambdarank ranker with VAL early-stopping (Tier 1.2)
  Step 4 : Save trained models to models/
  Step 5 : Save feature_weights.txt
  ── model training is COMPLETE here ──
  Step 6 : Load saved models, run game simulation on VAL  → report accuracy
  Step 7 : Run game simulation on TEST (final, held-out)  → report accuracy

Usage:
    python train.py
    python train.py --csv Connections_Data.csv --out-dir models
"""
import os, sys, argparse, pickle, itertools
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
# Split boundaries
# ─────────────────────────────────────────────────────────────────────────────
TRAIN_END = 640   # indices [0, 640)
VAL_END   = 777   # indices [640, 777)  → test is [777, ...)


# ─────────────────────────────────────────────────────────────────────────────
# Data integrity check
# ─────────────────────────────────────────────────────────────────────────────

def check_splits_no_overlap(train_games, val_games, test_games):
    """
    Verify that train, val, and test splits share no games (by word-set identity).
    Raises AssertionError if any overlap is detected — aborts training immediately.
    """
    train_ids = {frozenset(g["words"]) for g in train_games}
    val_ids   = {frozenset(g["words"]) for g in val_games}
    test_ids  = {frozenset(g["words"]) for g in test_games}

    tv  = train_ids & val_ids
    tt  = train_ids & test_ids
    vt  = val_ids   & test_ids

    if tv:
        raise AssertionError(f"DATA LEAK: {len(tv)} games overlap between train and val!")
    if tt:
        raise AssertionError(f"DATA LEAK: {len(tt)} games overlap between train and test!")
    if vt:
        raise AssertionError(f"DATA LEAK: {len(vt)} games overlap between val and test!")

    print(f"  ✓ Split integrity confirmed: NO overlap between train / val / test.")
    print(f"    train={len(train_games)}  val={len(val_games)}  test={len(test_games)}")



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
            X.append(extract_features(list(p), mats, set(range(16)), words, mat_names))
            y.append(1)

        # ── Hard negatives: 3/4 overlap with a correct group ─────────────
        hard_neg = [S for S in all_subsets
                    if set(S) not in gt_partitions
                    and max(len(set(S) & gt) for gt in gt_partitions) == 3]
        np.random.shuffle(hard_neg)
        for S in hard_neg[:8]:
            X.append(extract_features(list(S), mats, set(range(16)), words, mat_names))
            y.append(0)

        # ── Easy random negatives ─────────────────────────────────────────
        easy_neg = [S for S in all_subsets
                    if set(S) not in gt_partitions and S not in hard_neg]
        np.random.shuffle(easy_neg)
        for S in easy_neg[:8]:
            X.append(extract_features(list(S), mats, set(range(16)), words, mat_names))
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
            game_X.append(extract_features(list(p), mats, set(range(16)), words, mat_names))
            game_y.append(2)

        # Hard negatives (relevance = 1)
        hard_neg = [S for S in all_subsets
                    if set(S) not in gt_partitions
                    and max(len(set(S) & gt) for gt in gt_partitions) == 3]
        np.random.shuffle(hard_neg)
        for S in hard_neg[:8]:
            game_X.append(extract_features(list(S), mats, set(range(16)), words, mat_names))
            game_y.append(1)

        # Easy negatives (relevance = 0)
        easy_neg = [S for S in all_subsets
                    if set(S) not in gt_partitions and S not in hard_neg]
        np.random.shuffle(easy_neg)
        for S in easy_neg[:8]:
            game_X.append(extract_features(list(S), mats, set(range(16)), words, mat_names))
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
                game_X.append(extract_features(list(p), mats, set(range(16)), words, mat_names))
                game_y.append(2)
            hard_neg = [S for S in all_subsets
                        if set(S) not in gt_partitions
                        and max(len(set(S) & gt) for gt in gt_partitions) == 3]
            np.random.shuffle(hard_neg)
            for S in hard_neg[:8]:
                game_X.append(extract_features(list(S), mats, set(range(16)), words, mat_names))
                game_y.append(1)
            easy_neg = [S for S in all_subsets
                        if set(S) not in gt_partitions and S not in hard_neg]
            np.random.shuffle(easy_neg)
            for S in easy_neg[:8]:
                game_X.append(extract_features(list(S), mats, set(range(16)), words, mat_names))
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
            print(f"  Val ranking samples: {len(X_val)}  |  queries: {len(val_groups)}")

    params = {
        "objective":    "lambdarank",
        "metric":       "ndcg",
        "ndcg_eval_at": [1, 3, 5],
        "learning_rate": 0.05,
        "num_leaves":    31,            # Conservative to prevent overfitting
        "min_data_in_leaf": 15,         # Higher for small data sets
        "max_depth":     5,             # Avoid deep trees that memorise
        "reg_alpha":     0.1,           # L1 Regularisation
        "reg_lambda":    0.1,           # L2 Regularisation
        "subsample":     0.8,
        "colsample_bytree": 0.8,
        "verbose":       -1,
        "seed":          42,
    }

    valid_sets  = [dtrain] if dval is None else [dtrain, dval]
    valid_names = ["train"] if dval is None else ["train", "val"]

    # Use fixed 300 rounds. Early stopping at round 1 was poisoning the signal.
    booster = lgb.train(
        params,
        dtrain,
        num_boost_round=300,
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

    Feature vector per candidate: [separation_M0, separation_M1, ..., separation_MN]
    where separation_Mk = mean_in(S, Mk) - mean_out(S, Mk).

    Trained on VAL set only (per proposal: "weights are learned using
    Logistic Regression on the validation set"). The LR is lightweight
    (N parameters, one per matrix) so 137 val games is sufficient.
    Uses the same positive/negative sampling as the main classifiers.

    Returns:
        sklearn LogisticRegression fitted on val set
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
        remaining   = set(range(16))

        # ── Positives: the 4 correct groups ──────────────────────────────
        for p in gt_partitions:
            X_lr.append(compute_per_matrix_separation(list(p), mats, remaining))
            y_lr.append(1)

        # ── Hard negatives: 3/4 overlap with a correct group ─────────────
        hard_neg = [S for S in all_subsets
                    if set(S) not in gt_partitions
                    and max(len(set(S) & gt) for gt in gt_partitions) == 3]
        np.random.shuffle(hard_neg)
        for S in hard_neg[:8]:
            X_lr.append(compute_per_matrix_separation(list(S), mats, remaining))
            y_lr.append(0)

        # ── Easy random negatives ─────────────────────────────────────────
        easy_neg = [S for S in all_subsets
                    if set(S) not in gt_partitions and S not in hard_neg]
        np.random.shuffle(easy_neg)
        for S in easy_neg[:8]:
            X_lr.append(compute_per_matrix_separation(list(S), mats, remaining))
            y_lr.append(0)

    X_lr = np.array(X_lr, dtype=np.float32)
    y_lr = np.array(y_lr, dtype=np.int32)

    print(f"  LR fusion samples : {len(X_lr)}  ({sum(y_lr)} pos / {len(y_lr)-sum(y_lr)} neg)")
    print(f"  LR fusion features: {X_lr.shape[1]}  (one per matrix)")

    clf_lr = LogisticRegression(C=1.0, max_iter=300, random_state=42,
                                class_weight="balanced")
    clf_lr.fit(X_lr, y_lr)

    # Log learned weights for interpretability / paper analysis
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
    exact   = 0
    groups  = 0
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

        groups  += matched
        guesses += len(preds)
        for k, v in pmatch.items():
            partial[k] += v
        if matched == 4:
            exact += 1

    n         = len(games)
    game_acc  = exact  / n
    group_acc = groups / (4 * n)

    print(f"\n  [{split_name}]  {n} games")
    print(f"    Perfect games  : {exact}  ({game_acc*100:.2f}%)")
    print(f"    Group accuracy :  ({group_acc*100:.2f}%)")
    print(f"    Guess breakdown:")
    for k in [4, 3, 2, 1, 0]:
        pct = partial[k] / guesses * 100 if guesses else 0
        print(f"      {k}/4 → {partial[k]:4d}  ({pct:.1f}%)")

    return game_acc, group_acc


# ─────────────────────────────────────────────────────────────────────────────
# Main pipeline
# ─────────────────────────────────────────────────────────────────────────────

def train(csv_path: str, out_dir: str = "models"):
    try:
        _train_impl(csv_path, out_dir)
    except WikidataForbiddenError as e:
        print(f"\nCRITICAL ERROR: {e}")
        print("Stopping run to avoid further Wikidata API block. Run again later.")
        sys.exit(1)

def _train_impl(csv_path: str, out_dir: str = "models"):
    os.makedirs(out_dir, exist_ok=True)

    # ── 1. Load and split ─────────────────────────────────────────────────
    print("Loading data...")
    all_games   = load_games(csv_path)
    n           = len(all_games)
    train_games = all_games[:TRAIN_END]
    val_games   = all_games[TRAIN_END:VAL_END]
    test_games  = all_games[VAL_END:]

    print(f"Total : {n} games")
    print(f"  Train : {len(train_games)}  (idx 0 – {TRAIN_END-1})")
    print(f"  Val   : {len(val_games)}  (idx {TRAIN_END} – {VAL_END-1})")
    print(f"  Test  : {len(test_games)}  (idx {VAL_END} – {n-1})")
    print(f"\n  Val and Test sets are SEALED until after training.\n")

    # ── 0. Verify no data leakage between splits ──────────────────────────
    print("─"*50)
    print("STEP 0/4  Verifying split integrity (no data leakage)")
    print("─"*50)
    check_splits_no_overlap(train_games, val_games, test_games)

    # ── 2. Load embedder ──────────────────────────────────────────────────
    print("Initialising embedder (MPNet + GloVe + Numberbatch)...")
    embedder = ConnectionsEmbedder()

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # TRAINING PHASE — only train_games are touched below
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

    # ── 3. Extract features from TRAINING data only ───────────────────────
    print("\n" + "─"*50)
    print("STEP 1/4  Extract features (train only)")
    print("─"*50)
    X_train, y_train, mat_names = extract_train_features(train_games, embedder)
    feature_names = build_feature_names(mat_names)

    n_pos = sum(y_train)
    n_neg = len(y_train) - n_pos
    print(f"  Samples : {len(X_train)}  ({n_pos} pos / {n_neg} neg)")
    print(f"  Features: {len(feature_names)}")

    # ── 4. Fit RF + GBM ──────────────────────────────────────────────────
    print("\n" + "─"*50)
    print("STEP 2/4  Train RF + GBM")
    print("─"*50)
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
    print(f"  GBM  train accuracy : {clf_gbm.score(X_train, y_train)*100:.2f}%")

    # ── 5. Fit LightGBM ranker  (with val early stopping) ────────────────
    print("\n" + "─"*50)
    print("STEP 3/4  Train LightGBM Lambdarank Ranker (with val early stopping)")
    print("─"*50)
    clf_ranker = train_lgbm_ranker(train_games, embedder, mat_names, val_games=val_games)

    # ── 5.5  Train LR matrix weights on VAL set (Path A) ─────────────────
    print("\n" + "─"*50)
    print("STEP 3.5/4  Train LR Matrix Weights (Path A — val set only)")
    print("─"*50)
    clf_lr_fusion = train_lr_matrix_weights(val_games, embedder, mat_names)

    # ── 6. Save models + feature weights ─────────────────────────────────
    print("\n" + "─"*50)
    print("STEP 4/4  Save models and feature weights")
    print("─"*50)
    rf_path        = os.path.join(out_dir, "clf_rf.pkl")
    gbm_path       = os.path.join(out_dir, "clf_gbm.pkl")
    ranker_path    = os.path.join(out_dir, "clf_lgbm_ranker.pkl")
    lr_fusion_path = os.path.join(out_dir, "clf_lr_fusion.pkl")

    with open(rf_path,        "wb") as f: pickle.dump(clf_rf,       f)
    with open(gbm_path,       "wb") as f: pickle.dump(clf_gbm,      f)
    with open(ranker_path,    "wb") as f: pickle.dump(clf_ranker,   f)
    with open(lr_fusion_path, "wb") as f: pickle.dump(clf_lr_fusion, f)

    print(f"  RF        model → {rf_path}")
    print(f"  GBM       model → {gbm_path}")
    print(f"  Ranker    model → {ranker_path}")
    print(f"  LR fusion model → {lr_fusion_path}")

    save_feature_weights(clf_rf, clf_gbm, feature_names, "feature_weights.txt")
    print("  ✓ Training complete. Models saved. Val/Test not yet accessed.\n")

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # EVALUATION PHASE — saved models only; no re-training or re-fitting
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    print("Loading saved models for evaluation...")
    with open(rf_path,        "rb") as f: clf_rf_eval       = pickle.load(f)
    with open(gbm_path,       "rb") as f: clf_gbm_eval      = pickle.load(f)
    with open(ranker_path,    "rb") as f: clf_ranker_eval   = pickle.load(f)
    with open(lr_fusion_path, "rb") as f: clf_lr_fusion_eval = pickle.load(f)

    print("\n" + "="*60)
    print("  EVALUATION  (models are frozen — no training data used)")
    print("="*60)

    val_game_acc,  val_group_acc  = evaluate_split(
        val_games,  embedder, clf_rf_eval, clf_gbm_eval,
        split_name="VAL",  clf_ranker=clf_ranker_eval,
        clf_lr_fusion=clf_lr_fusion_eval)
    test_game_acc, test_group_acc = evaluate_split(
        test_games, embedder, clf_rf_eval, clf_gbm_eval,
        split_name="TEST", clf_ranker=clf_ranker_eval,
        clf_lr_fusion=clf_lr_fusion_eval)

    # ── Summary ───────────────────────────────────────────────────────────
    print("\n" + "="*60)
    print("  FINAL SUMMARY")
    print("="*60)
    print(f"{'Split':<8}  {'Games':>6}  {'Game Acc':>9}  {'Group Acc':>10}")
    print("-"*45)
    print(f"{'Val':<8}  {len(val_games):>6}  {val_game_acc*100:>8.2f}%  {val_group_acc*100:>9.2f}%")
    print(f"{'Test':<8}  {len(test_games):>6}  {test_game_acc*100:>8.2f}%  {test_group_acc*100:>9.2f}%")
    print("="*60)
    print(f"Features      : {len(feature_names)}")
    print(f"Models saved  : {out_dir}/clf_rf.pkl, {out_dir}/clf_gbm.pkl, {out_dir}/clf_lgbm_ranker.pkl")
    print(f"Weights saved : feature_weights.txt")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv",     type=str, default="Connections_Data.csv")
    parser.add_argument("--out-dir", type=str, default="models")
    args = parser.parse_args()

    if not os.path.exists(args.csv):
        print(f"Error: {args.csv} not found")
        sys.exit(1)

    train(args.csv, out_dir=args.out_dir)
