# python v4_eval_only.py --models-dir ablation_study_results/v4_kg_only_original --mode original
# python v4_train.py

"""
v4_train.py — Train the V4 KG-Only variant on either original or augmented data.
V4 Features: WikiData + Datamuse + ConceptNet + Wikipedia + Grid Uniqueness ONLY.
(No MPNet, GloVe, WordNet, Numberbatch, Lexical, MultiSense, or PairwiseContext.)

Fixed Test Set: Last 100 games of Connections_Data.csv for both modes.
"""
# Set 1: Original Data
# python v4_train.py

# Set 2: Augmented Data
# python v4_train.py --use-aug


import os, sys, argparse, pickle, itertools
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
# Split boundaries (Last 100 for Test, 137 for Val, 678 for Train)
# ─────────────────────────────────────────────────────────────────────────────
TEST_SIZE = 100
VAL_SIZE  = 137
# Total = 915. Test mapping indices: [815:915]
# Val mapping indices : [678:815]
# Train mapping indices: [0:678]

def load_data(orig_csv, use_aug=False):
    print(f"Loading original dataset: {orig_csv}")
    orig_games = load_games(orig_csv)
    n_total = len(orig_games)
    
    test_start = n_total - TEST_SIZE
    val_start  = test_start - VAL_SIZE
    
    test_games = orig_games[test_start:]
    val_games  = orig_games[val_start:test_start]
    orig_train = orig_games[:val_start]
    
    if not use_aug:
        print(f"Mode: ORIGINAL DATA ONLY")
        return orig_train, val_games, test_games
    
    print(f"Mode: AUGMENTED + ORIGINAL DATA")
    aug_games = []
    for f in ["dataset.csv", "connections_dataset1.csv", "connections_dataset2.csv"]:
        if os.path.exists(f):
            g = load_games(f)
            print(f"  Added {len(g)} games from {f}")
            aug_games.extend(g)
    
    train_games = orig_train + aug_games
    return train_games, val_games, test_games

def extract_features_v4(games, embedder, desc=""):
    X, y = [], []
    mat_names_ref = None
    for game in tqdm(games, desc=desc):
        words = game["words"]
        gt_partitions = [set(words.index(w) for w in gw) for gw in game["groups"].values()]
        
        # V4 Feature Set (All flags ON)
        mat_names, mats = build_matrices(
            embedder, words,
            use_base_embeddings=False,
            use_numberbatch=False,
            use_multisense=False,
            use_pairwise_context=False,
            use_lexical=False,
            use_knowledge_graphs=True
        )
        if mat_names_ref is None: mat_names_ref = mat_names

        all_subsets = list(itertools.combinations(range(16), 4))
        for p in gt_partitions:
            X.append(extract_features(list(p), mats, set(range(16)), words, mat_names))
            y.append(1)

        # Sampling negatives
        hard_neg = [S for S in all_subsets if set(S) not in gt_partitions and max(len(set(S) & gt) for gt in gt_partitions) == 3]
        np.random.shuffle(hard_neg)
        for S in hard_neg[:8]:
            X.append(extract_features(list(S), mats, set(range(16)), words, mat_names))
            y.append(0)

        easy_neg = [S for S in all_subsets if set(S) not in gt_partitions and S not in hard_neg]
        np.random.shuffle(easy_neg)
        for S in easy_neg[:8]:
            X.append(extract_features(list(S), mats, set(range(16)), words, mat_names))
            y.append(0)
            
    return X, y, mat_names_ref

def train_lgbm_v4(train_games, embedder, mat_names_ref, num_boost_round=300, val_games=None):
    """
    Trains a LightGBM Ranker with optimized settings to break the 30% plateau.
    - Low learning rate (0.01) for granular gradient updates.
    - Balanced validation sampling (Positives, Hard Negs, and 1% sampled Easy Negs)
      to prevent immediate overfitting/early-stopping.
    """
    X_rank, y_rank, groups = [], [], []
    for game in tqdm(train_games, desc="Ranker training data"):
        words = game["words"]
        gt_partitions = [set(words.index(w) for w in gw) for gw in game["groups"].values()]
        mat_names, mats = build_matrices(embedder, words, use_base_embeddings=False, use_numberbatch=False, use_multisense=False, use_pairwise_context=False, use_lexical=False, use_knowledge_graphs=True)
        all_subsets = list(itertools.combinations(range(16), 4))
        game_X, game_y = [], []
        for p in gt_partitions:
            game_X.append(extract_features(list(p), mats, set(range(16)), words, mat_names))
            game_y.append(2)
        hard_neg = [S for S in all_subsets if set(S) not in gt_partitions and max(len(set(S) & gt) for gt in gt_partitions) == 3]
        np.random.shuffle(hard_neg)
        for S in hard_neg[:8]:
            game_X.append(extract_features(list(S), mats, set(range(16)), words, mat_names))
            game_y.append(1)
        easy_neg = [S for S in all_subsets if set(S) not in gt_partitions and S not in hard_neg]
        np.random.shuffle(easy_neg)
        for S in easy_neg[:8]:
            game_X.append(extract_features(list(S), mats, set(range(16)), words, mat_names))
            game_y.append(0)
        if game_X:
            X_rank.extend(game_X)
            y_rank.extend(game_y)
            groups.append(len(game_X))

    feature_names = build_feature_names(mat_names_ref)
    dtrain = lgb.Dataset(np.array(X_rank, dtype=np.float32), label=y_rank, group=groups, feature_name=feature_names)
    
    # Val set for early stopping
    dval = None
    if val_games:
        X_val, y_val, val_groups = [], [], []
        for game in tqdm(val_games, desc="Ranker val data"):
            words = game["words"]
            gt_pts = [set(words.index(w) for w in gw) for gw in game["groups"].values()]
            _, mats = build_matrices(embedder, words, use_base_embeddings=False, use_numberbatch=False, use_multisense=False, use_pairwise_context=False, use_lexical=False, use_knowledge_graphs=True)
            subset_list = list(itertools.combinations(range(16), 4))
            g_X, g_y = [], []
            for p in gt_pts:
                g_X.append(extract_features(list(p), mats, set(range(16)), words, mat_names_ref))
                g_y.append(2)
            for S in subset_list:
                overlap = max(len(set(S) & gt) for gt in gt_pts)
                if set(S) not in gt_pts:
                    if overlap == 3:
                        # Hard negative
                        g_X.append(extract_features(list(S), mats, set(range(16)), words, mat_names_ref))
                        g_y.append(1)
                    elif np.random.rand() < 0.01: # Sample 1% of easy negatives to keep val set efficient
                        # Easy negative
                        g_X.append(extract_features(list(S), mats, set(range(16)), words, mat_names_ref))
                        g_y.append(0)
            if g_X:
                X_val.extend(g_X)
                y_val.extend(g_y)
                val_groups.append(len(g_X))
        dval = lgb.Dataset(np.array(X_val, dtype=np.float32), label=y_val, group=val_groups, feature_name=feature_names, reference=dtrain)

    params = {"objective": "lambdarank", "metric": "ndcg", "learning_rate": 0.01, "num_leaves": 31, "verbose": -1, "seed": 42}
    callbacks = [
        lgb.log_evaluation(period=50),
        lgb.early_stopping(stopping_rounds=50) # Stop if no improvement for 50 rounds
    ]
    return lgb.train(params, dtrain, num_boost_round=num_boost_round, valid_sets=[dtrain, dval] if dval else [dtrain], callbacks=callbacks)

def train_lr_fusion_v4(val_games, embedder):
    X_lr, y_lr = [], []
    for game in tqdm(val_games, desc="LR fusion features"):
        words = game["words"]
        gt_partitions = [set(words.index(w) for w in gw) for gw in game["groups"].values()]
        _, mats = build_matrices(embedder, words, use_base_embeddings=False, use_numberbatch=False, use_multisense=False, use_pairwise_context=False, use_lexical=False, use_knowledge_graphs=True)
        all_subsets = list(itertools.combinations(range(16), 4))
        for p in gt_partitions:
            X_lr.append(compute_per_matrix_separation(list(p), mats, set(range(16))))
            y_lr.append(1)
        hard_neg = [S for S in all_subsets if set(S) not in gt_partitions and max(len(set(S) & gt) for gt in gt_partitions) == 3]
        np.random.shuffle(hard_neg)
        for S in hard_neg[:8]:
            X_lr.append(compute_per_matrix_separation(list(S), mats, set(range(16))))
            y_lr.append(0)
    clf_lr = LogisticRegression(C=1.0, random_state=42, class_weight="balanced")
    clf_lr.fit(X_lr, y_lr)
    return clf_lr

def evaluate_v4(games, embedder, clf_rf, clf_gbm, clf_ranker, clf_lr, split_name=""):
    exact, total_groups = 0, 0
    for game in tqdm(games, desc=f"Evaluating [{split_name}]"):
        words = game["words"]
        gt_parts = [{words.index(w) for w in gw} for gw in game["groups"].values()]
        mat_names, mats = build_matrices(embedder, words, use_base_embeddings=False, use_numberbatch=False, use_multisense=False, use_pairwise_context=False, use_lexical=False, use_knowledge_graphs=True)
        matched, _, _ = simulate_game_ml(words, mats, mat_names, (clf_rf, clf_gbm), gt_parts, clf_ranker=clf_ranker, clf_lr_fusion=clf_lr)
        total_groups += matched
        if matched == 4: exact += 1
    n = len(games)
    game_acc  = exact / n
    group_acc = total_groups / (n * 4)
    print(f"  {split_name} Game Accuracy: {game_acc*100:.2f}%  |  Group Accuracy: {group_acc*100:.2f}%")
    return game_acc, group_acc

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--use-aug", action="store_true", help="Include augmented data in training")
    parser.add_argument("--num-boost-round", type=int, default=600, help="Number of boosting rounds for LightGBM Ranker")
    parser.add_argument("--out-dir", type=str, default=None)
    args = parser.parse_args()
    
    # ─────────────────────────────────────────────────────────────────────────────
    # Directory setup: ablation_study_results/v4_...
    # ─────────────────────────────────────────────────────────────────────────────
    base_dir = "ablation_study_results"
    mode_prefix = "augmented" if args.use_aug else "original"
    sub_dir  = f"v4_kg_only_{mode_prefix}"
    out_dir  = args.out_dir if args.out_dir else os.path.join(base_dir, sub_dir)
    os.makedirs(out_dir, exist_ok=True)
    
    # Start logging console output to a file inside the out_dir (with mode prefix)
    log_file = os.path.join(out_dir, f"{mode_prefix}_console_output.log")
    print(f"Logging console output to {log_file}")
    
    class Logger(object):
        def __init__(self, filename):
            self.terminal = sys.stdout
            self.log = open(filename, "a", buffering=1)
        def write(self, message):
            self.terminal.write(message)
            self.log.write(message)
            self.log.flush() # Force write to disk
        def flush(self):
            self.terminal.flush()
            self.log.flush()

    sys.stdout = Logger(log_file)
    sys.stderr = sys.stdout # Capture progress bars too

    train_games, val_games, test_games = load_data("Connections_Data.csv", use_aug=args.use_aug)
    
    embedder = ConnectionsEmbedder(load_embeddings=False)
    try:
        X_train, y_train, mat_names = extract_features_v4(train_games, embedder, desc="V4 Train Features")
        feature_names = build_feature_names(mat_names)
        
        print("Training RF + GBM...")
        clf_rf = RandomForestClassifier(n_estimators=200, max_depth=10, random_state=42, n_jobs=-1)
        clf_rf.fit(X_train, y_train)
        clf_gbm = GradientBoostingClassifier(n_estimators=200, max_depth=3, random_state=42)
        clf_gbm.fit(X_train, y_train)
        
        clf_ranker = train_lgbm_v4(train_games, embedder, mat_names, num_boost_round=args.num_boost_round, val_games=val_games)
        clf_lr = train_lr_fusion_v4(val_games, embedder)
        
        # ── Save Models ──────────────────────────────────────────────────
        print(f"Saving models to {out_dir} ...")
        with open(os.path.join(out_dir, f"{mode_prefix}_clf_rf.pkl"), "wb") as f: pickle.dump(clf_rf, f)
        with open(os.path.join(out_dir, f"{mode_prefix}_clf_gbm.pkl"), "wb") as f: pickle.dump(clf_gbm, f)
        with open(os.path.join(out_dir, f"{mode_prefix}_clf_lgbm_ranker.pkl"), "wb") as f: pickle.dump(clf_ranker, f)
        with open(os.path.join(out_dir, f"{mode_prefix}_clf_lr_fusion.pkl"), "wb") as f: pickle.dump(clf_lr, f)
        
        # ── Save Feature Weights (RF + GBM) ──────────────────────────────
        weights_name = f"{mode_prefix}_feature_weights.txt"
        save_feature_weights(clf_rf, clf_gbm, feature_names, os.path.join(out_dir, weights_name))
        
        # ── Save Ranker Weights / Importance ─────────────────────────────
        ranker_weights_name = f"{mode_prefix}_ranker_weights_of_models.txt"
        ranker_weights_path = os.path.join(out_dir, ranker_weights_name)
        print(f"Saving ranker feature importance to {ranker_weights_path} ...")
        ranker_imp = clf_ranker.feature_importance(importance_type='gain')
        ranker_order = np.argsort(ranker_imp)[::-1]
        with open(ranker_weights_path, "w") as f:
            f.write(f"Ranker Feature Importance (Gain) - {mode_prefix.upper()}\n")
            f.write("="*40 + "\n")
            for idx in ranker_order:
                f.write(f"{feature_names[idx]:<45}: {ranker_imp[idx]:.4f}\n")
        
        val_game_acc,  val_group_acc  = evaluate_v4(val_games,  embedder, clf_rf, clf_gbm, clf_ranker, clf_lr, "VAL")
        test_game_acc, test_group_acc = evaluate_v4(test_games, embedder, clf_rf, clf_gbm, clf_ranker, clf_lr, "TEST")
        print(f"\n{'='*55}")
        print(f"  KG-ONLY SUMMARY")
        print(f"{'='*55}")
        print(f"{'Split':<8}  {'Games':>6}  {'Game Acc':>9}  {'Group Acc':>10}")
        print(f"{'-'*45}")
        print(f"{'Val':<8}  {len(val_games):>6}  {val_game_acc*100:>8.2f}%  {val_group_acc*100:>9.2f}%")
        print(f"{'Test':<8}  {len(test_games):>6}  {test_game_acc*100:>8.2f}%  {test_group_acc*100:>9.2f}%")
        print(f"{'='*55}")
        print(f"\nAll results saved in: {out_dir}")
    except WikidataForbiddenError as e:
        print(f"Wikidata API Blocked: {e}")

if __name__ == "__main__":
    main()
