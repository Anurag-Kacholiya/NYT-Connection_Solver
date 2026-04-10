"""
ablation_v2_train.py  —  Ablation Variant 2: + Dense Knowledge Embeddings
========================================================================

WHAT THIS VARIANT KEEPS:
  ✓ MPNet_sim          (sentence-transformer cosine similarity)
  ✓ WordNet_sim        (Wu-Palmer similarity via NLTK)
  ✓ GloVe_sim          (GloVe-100 cosine similarity)
  ✓ ConceptNet Numberbatch embeddings (Dense Knowledge)
  ✓ Lexical similarity   (character n-grams)
  ✓ Phonetic similarity  (CMU phoneme edit distance)
  ✓ Morphological similarity (lemma/affix Jaccard)
  ✓ Full ML ensemble   (RF + GBM + LightGBM Ranker + LR Fusion)
  ✓ Original training data (Connections_Data.csv)
  ✓ Joint solver (solve_ml)

WHAT THIS VARIANT REMOVES:
  ✗ MultiSense similarity    (Tier 1.1 – sense-aware embeddings)
  ✗ PairContext similarity   (Tier 2.2 – "A and B are both")
  ✗ TemplateCtx similarity   (Tier 2.2 – template-based context)
  ✗ WikiData similarity      (external KG)
  ✗ Datamuse similarity      (external KG)
  ✗ ConceptNet similarity    (external KG / SQLite)
  ✗ WikipediaCategories_sim  (external KG)
  ✗ GridPropertyUniqueness   (external KG, grid-aware)
  ✗ GridCategoryUniqueness   (external KG, grid-aware)
  ✗ GridDatamuseUniqueness   (external KG, grid-aware)

FEATURE SPACE:
  Matrices    :  7  (MPNet, WordNet, GloVe, MPNet+Numberbatch, Lexical, Phonetic, Morphological)
  Total features: 7 × 9 = 63

RESEARCH QUESTION:
  "How much do static, offline-computed features (dense vectors + word form) 
   add over plain semantics?"

OUTPUT:
  models_v2/
  results_v2/

Split (matching aug_train.py logic: 100 test / 137 val):
  Test : last 100 games
  Val  : next 137 games before test
  Train: the rest
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
# V2 CONFIGURATION
# ─────────────────────────────────────────────────────────────────────────────

V2_FLAGS = dict(
    use_numberbatch=True,        # ← ON: ConceptNet Numberbatch
    use_multisense=False,        # ← OFF
    use_pairwise_context=False,  # ← OFF
    use_lexical=True,           # ← ON: includes Lexical, Phonetic, Morphological
    use_knowledge_graphs=False,  # ← OFF
)

VARIANT_NAME  = "V2 — + Dense Knowledge Embeddings"
OUT_MODELS    = "models_v2"
OUT_RESULTS   = "results_v2"

# Split boundaries (matching aug_train.py: 100 test / 137 val)
TEST_SIZE = 100
VAL_SIZE  = 137


# ─────────────────────────────────────────────────────────────────────────────
# Helper: thin wrapper so every build_matrices() call uses V2_FLAGS
# ─────────────────────────────────────────────────────────────────────────────

def _build(embedder, words):
    return build_matrices(embedder, words, **V2_FLAGS)


# ─────────────────────────────────────────────────────────────────────────────
# Data integrity check
# ─────────────────────────────────────────────────────────────────────────────

def check_splits_no_overlap(train_games, val_games, test_games):
    train_ids = {frozenset(g["words"]) for g in train_games}
    val_ids   = {frozenset(g["words"]) for g in val_games}
    test_ids  = {frozenset(g["words"]) for g in test_games}

    tv = train_ids & val_ids
    tt = train_ids & test_ids
    vt = val_ids   & test_ids

    if tv:
        raise AssertionError(f"DATA LEAK: {len(tv)} games overlap between train and val!")
    if tt:
        raise AssertionError(f"DATA LEAK: {len(tt)} games overlap between train and test!")
    if vt:
        raise AssertionError(f"DATA LEAK: {len(vt)} games overlap between val and test!")

    print(f"  ✓ Split integrity confirmed: NO overlap.")
    print(f"    train={len(train_games)}  val={len(val_games)}  test={len(test_games)}")


# ─────────────────────────────────────────────────────────────────────────────
# Feature extraction
# ─────────────────────────────────────────────────────────────────────────────

def extract_train_features(train_games, embedder):
    X, y = [], []
    mat_names_ref = None

    for game in tqdm(train_games, desc="[V2] Extracting train features"):
        words = game["words"]
        gt_partitions = [
            set(words.index(w) for w in gw)
            for gw in game["groups"].values()
        ]

        mat_names, mats = _build(embedder, words)
        if mat_names_ref is None:
            mat_names_ref = mat_names

        all_subsets = list(itertools.combinations(range(16), 4))

        for p in gt_partitions:
            X.append(extract_features(list(p), mats, set(range(16)), words, mat_names))
            y.append(1)

        hard_neg = [S for S in all_subsets
                    if set(S) not in gt_partitions
                    and max(len(set(S) & gt) for gt in gt_partitions) == 3]
        np.random.shuffle(hard_neg)
        for S in hard_neg[:8]:
            X.append(extract_features(list(S), mats, set(range(16)), words, mat_names))
            y.append(0)

        easy_neg = [S for S in all_subsets
                    if set(S) not in gt_partitions and S not in hard_neg]
        np.random.shuffle(easy_neg)
        for S in easy_neg[:8]:
            X.append(extract_features(list(S), mats, set(range(16)), words, mat_names))
            y.append(0)

    return X, y, mat_names_ref


def train_lgbm_ranker(train_games, embedder, mat_names_ref, val_games=None):
    print("\n[V2] Building ranking dataset for LightGBM lambdarank...")
    X_rank, y_rank, groups = [], [], []

    for game in tqdm(train_games, desc="[V2] Ranking feature extraction"):
        words = game["words"]
        gt_partitions = [
            set(words.index(w) for w in gw)
            for gw in game["groups"].values()
        ]

        mat_names, mats = _build(embedder, words)
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
            X_rank.extend(game_X)
            y_rank.extend(game_y)
            groups.append(len(game_X))

    X_rank = np.array(X_rank, dtype=np.float32)
    y_rank = np.array(y_rank, dtype=np.int32)
    groups = np.array(groups, dtype=np.int32)
    feature_names = build_feature_names(mat_names_ref)
    dtrain = lgb.Dataset(X_rank, label=y_rank, group=groups, feature_name=feature_names)

    dval = None
    if val_games is not None:
        print("  Building validation ranking dataset...")
        X_val, y_val, val_groups = [], [], []
        for game in tqdm(val_games, desc="[V2] Val ranking features"):
            words = game["words"]
            gt_partitions = [
                set(words.index(w) for w in gw)
                for gw in game["groups"].values()
            ]
            mat_names, mats = _build(embedder, words)
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
            dval = lgb.Dataset(np.array(X_val, dtype=np.float32), label=np.array(y_val, dtype=np.int32), 
                                group=np.array(val_groups, dtype=np.int32), feature_name=feature_names, reference=dtrain)

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
        "verbose":          -1,
        "seed":             42,
    }

    valid_sets  = [dtrain] if dval is None else [dtrain, dval]
    valid_names = ["train"] if dval is None else ["train", "val"]

    booster = lgb.train(params, dtrain, num_boost_round=300, valid_sets=valid_sets, valid_names=valid_names,
                        callbacks=[lgb.log_evaluation(period=50)])
    return booster


def train_lr_matrix_weights(val_games, embedder, mat_names_ref):
    X_lr, y_lr = [], []
    for game in tqdm(val_games, desc="[V2] LR fusion: val extraction"):
        words = game["words"]
        gt_partitions = [set(words.index(w) for w in gw) for gw in game["groups"].values()]
        _, mats = _build(embedder, words)
        all_subsets = list(itertools.combinations(range(16), 4))
        remaining = set(range(16))
        for p in gt_partitions:
            X_lr.append(compute_per_matrix_separation(list(p), mats, remaining))
            y_lr.append(1)
        hard_neg = [S for S in all_subsets if set(S) not in gt_partitions and max(len(set(S) & gt) for gt in gt_partitions) == 3]
        np.random.shuffle(hard_neg)
        for S in hard_neg[:8]:
            X_lr.append(compute_per_matrix_separation(list(S), mats, remaining))
            y_lr.append(0)
        easy_neg = [S for S in all_subsets if set(S) not in gt_partitions and S not in hard_neg]
        np.random.shuffle(easy_neg)
        for S in easy_neg[:8]:
            X_lr.append(compute_per_matrix_separation(list(S), mats, remaining))
            y_lr.append(0)
    X_lr = np.array(X_lr, dtype=np.float32); y_lr = np.array(y_lr, dtype=np.int32)
    clf_lr = LogisticRegression(C=1.0, max_iter=300, random_state=42, class_weight="balanced")
    clf_lr.fit(X_lr, y_lr)
    return clf_lr


def evaluate_split(games, embedder, clf_rf, clf_gbm, split_name="", clf_ranker=None, clf_lr_fusion=None):
    exact = 0
    groups = 0
    partial = {4: 0, 3: 0, 2: 0, 1: 0, 0: 0}
    guesses = 0
    for game in tqdm(games, desc=f"[V2] Evaluating [{split_name}]"):
        words = game["words"]
        gt_partitions = [{words.index(w) for w in gw} for gw in game["groups"].values()]
        mat_names, mats = _build(embedder, words)
        matched, preds, pmatch = simulate_game_ml(
            words, mats, mat_names, (clf_rf, clf_gbm), gt_partitions,
            clf_ranker=clf_ranker, clf_lr_fusion=clf_lr_fusion, use_candidates=True)
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
    return game_acc, group_acc, exact, groups, partial, guesses


def train_v2(csv_path: str):
    try:
        _train_v2_impl(csv_path)
    except WikidataForbiddenError as e:
        # Shouldn't happen in V2 (no KG calls), but guard just in case
        print(f"\nCRITICAL ERROR: {e}")
        sys.exit(1)


def _train_v2_impl(csv_path: str):
    os.makedirs(OUT_MODELS,  exist_ok=True)
    os.makedirs(OUT_RESULTS, exist_ok=True)

    print("=" * 60)
    print(f"  ABLATION {VARIANT_NAME}")
    print("=" * 60)
    print(f"  Active matrices : MPNet+Numberbatch, MPNet, WordNet, GloVe,")
    print(f"                    Lexical, Phonetic, Morphological")
    print(f"  Feature vector  : 7 matrices × 9 features = 63-d")
    print(f"  Training data   : {csv_path}  (original — no augmentation)")
    print(f"  Classifiers     : RF + GBM + LightGBM Ranker + LR Fusion")
    print("=" * 60)

    print("\nLoading data...")
    all_games = load_games(csv_path)
    n = len(all_games)
    test_start = n - TEST_SIZE
    val_start = test_start - VAL_SIZE
    train_games = all_games[:val_start]
    val_games = all_games[val_start:test_start]
    test_games = all_games[test_start:]

    print(f"Total : {n} games")
    print(f"  Train : {len(train_games)}  (idx 0 – {val_start-1})")
    print(f"  Val   : {len(val_games)}  (idx {val_start} – {test_start-1})")
    print(f"  Test  : {len(test_games)}  (idx {test_start} – {n-1})")
    check_splits_no_overlap(train_games, val_games, test_games)

    print("\nInitialising embedder (MPNet + GloVe + Numberbatch)...")
    embedder = ConnectionsEmbedder()

    # ── 1. Extract features (TRAIN only) ────────────────────────────────
    print("\n" + "─"*50)
    print("STEP 1/4  Extract features (train only)")
    print("─"*50)
    X_train, y_train, mat_names = extract_train_features(train_games, embedder)
    feature_names = build_feature_names(mat_names)
    n_pos = sum(y_train); n_neg = len(y_train) - n_pos
    print(f"  Samples : {len(X_train)}  ({n_pos} pos / {n_neg} neg)")
    print(f"  Features: {len(feature_names)}  (active matrices: {mat_names})")

    # ── 2. Fit RF + GBM ─────────────────────────────────────────────
    print("\n" + "─"*50)
    print("STEP 2/4  Train RF + GBM")
    print("─"*50)
    print("Training Random Forest (n_estimators=200, max_depth=10)...")
    clf_rf = RandomForestClassifier(n_estimators=200, max_depth=10, random_state=42, n_jobs=-1)
    clf_rf.fit(X_train, y_train)
    print(f"  RF  train accuracy : {clf_rf.score(X_train, y_train)*100:.2f}%")

    print("Training Gradient Boosting (n_estimators=200, max_depth=3)...")
    clf_gbm = GradientBoostingClassifier(n_estimators=200, max_depth=3, learning_rate=0.1, subsample=0.8, random_state=42)
    clf_gbm.fit(X_train, y_train)
    print(f"  GBM train accuracy : {clf_gbm.score(X_train, y_train)*100:.2f}%")

    # ── 3. LightGBM Ranker ────────────────────────────────────────
    print("\n" + "─"*50)
    print("STEP 3/4  Train LightGBM Lambdarank Ranker")
    print("─"*50)
    clf_ranker = train_lgbm_ranker(train_games, embedder, mat_names, val_games=val_games)

    # ── 3.5 LR Matrix Weights ─────────────────────────────────
    print("\n" + "─"*50)
    print("STEP 3.5/4  Train LR Matrix Weights (Path A)")
    print("─"*50)
    clf_lr_fusion = train_lr_matrix_weights(val_games, embedder, mat_names)

    # ── 4. Save models ──────────────────────────────────────────
    print("\n" + "─"*50)
    print("STEP 4/4  Save models and feature weights")
    print("─"*50)
    rf_path        = os.path.join(OUT_MODELS, "clf_rf.pkl")
    gbm_path       = os.path.join(OUT_MODELS, "clf_gbm.pkl")
    ranker_path    = os.path.join(OUT_MODELS, "clf_lgbm_ranker.pkl")
    lr_fusion_path = os.path.join(OUT_MODELS, "clf_lr_fusion.pkl")

    with open(rf_path,        "wb") as f: pickle.dump(clf_rf,        f)
    with open(gbm_path,       "wb") as f: pickle.dump(clf_gbm,       f)
    with open(ranker_path,    "wb") as f: pickle.dump(clf_ranker,    f)
    with open(lr_fusion_path, "wb") as f: pickle.dump(clf_lr_fusion, f)

    fw_path = os.path.join(OUT_RESULTS, "feature_weights.txt")
    save_feature_weights(clf_rf, clf_gbm, feature_names, fw_path)
    print(f"  RF        model → {rf_path}")
    print(f"  GBM       model → {gbm_path}")
    print(f"  Ranker    model → {ranker_path}")
    print(f"  LR fusion model → {lr_fusion_path}")
    print(f"  Feature weights → {fw_path}")
    print("  ✓ Training complete.\n")

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # EVALUATION PHASE — reload saved models (frozen), never re-train
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    print("Loading saved models for evaluation...")
    with open(rf_path,        "rb") as f: clf_rf_eval        = pickle.load(f)
    with open(gbm_path,       "rb") as f: clf_gbm_eval       = pickle.load(f)
    with open(ranker_path,    "rb") as f: clf_ranker_eval    = pickle.load(f)
    with open(lr_fusion_path, "rb") as f: clf_lr_fusion_eval = pickle.load(f)

    print("\n" + "="*60)
    print("  EVALUATION  (frozen models — no training data used)")
    print("="*60)

    val_game_acc,  val_group_acc,  _, _,        _,          _          = evaluate_split(
        val_games,  embedder, clf_rf_eval, clf_gbm_eval, "VAL",
        clf_ranker=clf_ranker_eval, clf_lr_fusion=clf_lr_fusion_eval)

    test_game_acc, test_group_acc, _, test_grp, test_partial, test_guess = evaluate_split(
        test_games, embedder, clf_rf_eval, clf_gbm_eval, "TEST",
        clf_ranker=clf_ranker_eval, clf_lr_fusion=clf_lr_fusion_eval)

    # ── Print + save summary ──────────────────────────────────────
    summary_lines = [
        "=" * 60,
        f"  {VARIANT_NAME}",
        "=" * 60,
        f"  Active matrices : {mat_names}",
        f"  Feature vector  : {len(feature_names)}-d",
        f"  Training games  : {len(train_games)}  (original only)",
        "",
        f"{'Split':<8}  {'Games':>6}  {'Game Acc':>9}  {'Group Acc':>10}",
        "-" * 45,
        f"{'Val':<8}  {len(val_games):>6}  {val_game_acc*100:>8.2f}%  {val_group_acc*100:>9.2f}%",
        f"{'Test':<8}  {len(test_games):>6}  {test_game_acc*100:>8.2f}%  {test_group_acc*100:>9.2f}%",
        "=" * 60,
        "",
        "TEST Guess breakdown:",
    ]
    for k in [4, 3, 2, 1, 0]:
        pct = test_partial[k] / test_guess * 100 if test_guess else 0
        summary_lines.append(f"  {k}/4 → {test_partial[k]:4d}  ({pct:.1f}%)")
    summary_lines += [
        "",
        f"Models saved  : {OUT_MODELS}/",
        f"Weights saved : {fw_path}",
        "=" * 60,
    ]

    summary_text = "\n".join(summary_lines)
    print("\n" + summary_text)

    summary_path = os.path.join(OUT_RESULTS, "summary.txt")
    with open(summary_path, "w") as f:
        f.write(summary_text + "\n")
    print(f"\n  Summary saved → {summary_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Ablation V2 — + Dense Knowledge Embeddings (Numberbatch + Lexical/Phonetic/Morphological)."
    )
    parser.add_argument("--csv", type=str, default="Connections_Data.csv",
                        help="Path to the original NYT Connections dataset CSV.")
    args = parser.parse_args()

    if not os.path.exists(args.csv):
        print(f"Error: {args.csv} not found.")
        sys.exit(1)

    train_v2(args.csv)
