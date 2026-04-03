"""
evaluate.py — Feature extraction, matrix construction, game simulation.

Key design:
  Matrices (7 total when all flags on):
    MPNet+Numberbatch_sim, MPNet_sim, WordNet_sim, GloVe_sim   ← original
    MultiSense_sim                                              ← Tier 1.1 (sense-aware)
    PairContext_sim                                             ← Tier 2.2 (relational context)
    TemplateCtx_sim                                             ← Tier 2.2 (template context)

  Features per matrix (9):
    mean_in, min_in, var_in, max_out, separation
    second_max_out, avg_top3_out, group_ambiguity              ← Tier 2.1 (distractor penalty)
    triangle_score                                             ← Tier 2.3 (coherence)

  Total features: up to 7 * 9 = 63 (filtered by active matrices)

  Game simulation (simulate_game_ml)  — 3-STAGE PIPELINE:
    Stage 1  — candidate generation via agglomerative clustering (top_k=120)  ← Tier 1.3
    Stage 2  — ML scoring: RF (35%) + GBM (35%) + LightGBM ranker (30%)      ← Tier 1.2
    Stage 2b — competition-aware solver adjustments:
                 adjusted_score[g] = base_score[g]
                                   + α * (1/rank(g))   ← rank bonus
                                   + β * gap(g)         ← gap-to-next bonus
               (α=0.05, β=0.05 — kept small so ML score dominates)
    Stage 3  — solve_ml(candidates, score_lookup):
                 global joint optimisation — finds 4-partition maximising Σ adjusted_score
                 preventing any single high-scoring group from blocking others

  Greedy fallback retained when solve_ml returns None (sparse candidates).
"""
import os, sys, itertools, pickle
import numpy as np
from tqdm import tqdm
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.spatial.distance import squareform
import nltk

from data_loader import load_games
from embedder import ConnectionsEmbedder
from solver import solve_ml


# ── Utilities ────────────────────────────────────────────────────────────────

def double_center(mat: np.ndarray) -> np.ndarray:
    r_mu = np.mean(mat, axis=1, keepdims=True)
    c_mu = np.mean(mat, axis=0, keepdims=True)
    g_mu = np.mean(mat)
    return mat - r_mu - c_mu + g_mu


# ── Matrix construction ───────────────────────────────────────────────────────

def build_matrices(embedder, words,
                   use_numberbatch=True,
                   use_multisense=True,
                   use_pairwise_context=True):
    """
    Build double-centred similarity matrices for a game.

    Returns:
        names    - list[str]  human-readable matrix names
        matrices - list[np.ndarray]  corresponding (16×16) matrices

    Flags:
        use_numberbatch      → include MPNet+Numberbatch joint embedding
        use_multisense       → include multi-sense WordNet embedding (Tier 1.1)
        use_pairwise_context → include pairwise-context + template-context (Tier 2.2)
    """
    embeddings = embedder.get_embeddings(words)
    mpnet_sim  = double_center(np.dot(embeddings, embeddings.T))
    glove_sim  = double_center(embedder.get_glove_similarity(words))
    wordnet_sim = double_center(embedder.get_wordnet_similarity(words))

    named = []

    if use_numberbatch:
        nb_emb = embedder.get_concatenated_mpnet_numberbatch_embedding(words)
        nb_sim = double_center(np.dot(nb_emb, nb_emb.T))
        named.append(("MPNet+Numberbatch_sim", nb_sim))

    named += [
        ("MPNet_sim",   mpnet_sim),
        ("WordNet_sim", wordnet_sim),
        ("GloVe_sim",   glove_sim),
    ]

    if use_multisense:
        ms_sim = double_center(embedder.get_multisense_sim_matrix(words))
        named.append(("MultiSense_sim", ms_sim))

    if use_pairwise_context:
        pc_sim  = double_center(embedder.get_pairwise_context_sim_matrix(words))
        tc_sim  = double_center(embedder.get_template_context_sim_matrix(words))
        named.append(("PairContext_sim",  pc_sim))
        named.append(("TemplateCtx_sim",  tc_sim))

    names    = [n for n, _ in named]
    matrices = [m for _, m in named]
    return names, matrices


# ── Feature extraction ────────────────────────────────────────────────────────

# Feature suffix order per matrix  (9 features × n_matrices)
_PER_MAT_SUFFIXES = [
    "mean_in",
    "min_in",
    "var_in",
    "max_out",
    "separation",
    "second_max_out",   # Tier 2.1
    "avg_top3_out",     # Tier 2.1
    "group_ambiguity",  # Tier 2.1
    "triangle_score",   # Tier 2.3
]

# Full candidate feature list — the union over all 7 matrices
_ALL_MATRICES = [
    "MPNet+Numberbatch_sim",
    "MPNet_sim",
    "WordNet_sim",
    "GloVe_sim",
    "MultiSense_sim",
    "PairContext_sim",
    "TemplateCtx_sim",
]

ALL_FEATURE_NAMES = [
    f"{m}_{s}"
    for m in _ALL_MATRICES
    for s in _PER_MAT_SUFFIXES
]


def build_feature_names(matrix_names):
    """
    Return the ordered feature names that are active given the supplied matrices.
    """
    active = set(matrix_names)
    return [f for f in ALL_FEATURE_NAMES if any(f.startswith(m) for m in active)]


def extract_features(S, matrices, remaining_items, words, matrix_names):
    """
    Feature vector for a candidate group S (list of 4 word indices).

    For each matrix:
      mean_in, min_in, var_in          — in-group cohesion
      max_out, separation              — original distractor signals
      second_max_out, avg_top3_out     — stronger distractor signals  (Tier 2.1)
      group_ambiguity                  — how much each word "fits" other groups (Tier 2.1)
      triangle_score                   — fraction of consistent triplets  (Tier 2.3)
    """
    feats = {}
    S_set = set(S)
    n_total = len(words)
    outsiders = [i for i in remaining_items if i not in S_set]

    for mname, mat in zip(matrix_names, matrices):
        # ── In-group edges  C(4,2)=6 ──────────────────────────────────────
        edges = [mat[S[a]][S[b]] for a in range(4) for b in range(a + 1, 4)]
        mean_in = float(np.mean(edges))
        min_in  = float(np.min(edges))
        var_in  = float(np.var(edges))

        # ── Out-group similarities ─────────────────────────────────────────
        out_sims = [mat[item][other]
                    for item in S for other in outsiders]
        if out_sims:
            sorted_out = sorted(out_sims, reverse=True)
            max_out        = float(sorted_out[0])
            second_max_out = float(sorted_out[1]) if len(sorted_out) > 1 else max_out
            avg_top3_out   = float(np.mean(sorted_out[:3]))
            mean_out       = float(np.mean(out_sims))
        else:
            max_out = second_max_out = avg_top3_out = mean_out = 0.0

        separation = mean_in - mean_out

        # ── Group ambiguity: each word's average sim to ALL other 15 words ─
        # then sum over the 4 group members
        all_others_range = [i for i in range(n_total) if i not in S_set]
        if all_others_range:
            group_ambiguity = float(sum(
                np.mean([mat[item][o] for o in all_others_range])
                for item in S
            ))
        else:
            group_ambiguity = 0.0

        # ── Triangle consistency ───────────────────────────────────────────
        # threshold = mean of in-group edges
        thresh = mean_in
        consistent = 0
        total_triplets = 0
        for a, b, c in itertools.combinations(range(4), 3):
            ia, ib, ic = S[a], S[b], S[c]
            total_triplets += 1
            ab = mat[ia][ib] >= thresh
            bc = mat[ib][ic] >= thresh
            ac = mat[ia][ic] >= thresh
            # if A~B and B~C then A~C should hold
            if (ab and bc and ac) or (not (ab and bc)):
                consistent += 1
        triangle_score = consistent / total_triplets if total_triplets else 1.0

        feats[f"{mname}_mean_in"]       = mean_in
        feats[f"{mname}_min_in"]        = min_in
        feats[f"{mname}_var_in"]        = var_in
        feats[f"{mname}_max_out"]       = max_out
        feats[f"{mname}_separation"]    = separation
        feats[f"{mname}_second_max_out"]= second_max_out
        feats[f"{mname}_avg_top3_out"]  = avg_top3_out
        feats[f"{mname}_group_ambiguity"] = group_ambiguity
        feats[f"{mname}_triangle_score"]  = triangle_score

    valid_names = build_feature_names(matrix_names)
    return [feats[name] for name in valid_names]


# ── Two-stage candidate filtering  (Tier 1.3) ────────────────────────────────

def generate_candidates(words, matrices, top_k=120):
    num_words = len(words)
    """
    Candidate generation for joint optimisation.
    Returns ALL 1,820 possible combinations of 16-choose-4.
    """
    all_combinations = list(itertools.combinations(range(num_words), 4))
    candidate_set = set(all_combinations)

    return list(candidate_set)


# ── Game simulation ───────────────────────────────────────────────────────────

# Competition-aware solver adjustment hyper-parameters
_RANK_ALPHA = 0.05   # weight for rank bonus: α / rank(g)
_GAP_BETA   = 0.05   # weight for gap bonus:  β * gap_to_next


def _compute_adjusted_scores(subsets, base_probs):
    """
    Stage 2b — competition-aware solver-stage adjustments.

    Given base ML scores for all candidate groups, compute:
        adjusted_score[g] = base_score[g]
                          + α * (1 / rank(g))    ← rank bonus (1-indexed)
                          + β * gap_to_next(g)   ← gap to next best candidate

    These are baked directly into score_lookup passed to solve_ml.
    The ML models are NOT re-run — avoids distribution shift.

    Returns: dict {subset_tuple -> adjusted_score}
    """
    # Sort descending by base ML score to compute ranks
    order = np.argsort(base_probs)[::-1]   # indices sorted best-first
    n = len(subsets)

    # Gap: score[i] - score[i+1] (last item has gap=0)
    sorted_probs = base_probs[order]
    gaps = np.zeros(n)
    gaps[:-1] = sorted_probs[:-1] - sorted_probs[1:]

    score_lookup = {}
    for rank_0, idx in enumerate(order):
        rank_1 = rank_0 + 1                         # 1-indexed
        base   = float(base_probs[idx])
        bonus  = _RANK_ALPHA / rank_1 + _GAP_BETA * float(gaps[rank_0])
        score_lookup[subsets[idx]] = base + bonus

    return score_lookup


def simulate_game_ml(words, matrices, matrix_names, clf, gt_partitions_list,
                     clf_ranker=None, use_candidates=True):
    """
    3-STAGE PIPELINE for group prediction.

    Stage 1  — candidate generation (agglomerative clustering, top_k=120)
    Stage 2  — ML ensemble scoring (RF 35% + GBM 35% + LightGBM 30%)
    Stage 2b — competition-aware solver adjustments (rank bonus + gap bonus)
    Stage 3  — solve_ml(): global joint optimisation to find the PARTITION of
                16 words into 4 groups of 4 that maximises total adjusted score
                → prevents one dominant group from blocking others

    Greedy fallback: if solve_ml returns None (sparse candidates), falls back
    to the original greedy pick-top-scoring-group loop.

    Args:
        words              : list of 16 word strings
        matrices           : list of similarity matrices
        matrix_names       : list of matrix name strings (same order)
        clf                : (clf_rf, clf_gbm) tuple
        gt_partitions_list : list of ground-truth group sets (for evaluation)
        clf_ranker         : optional LightGBM ranker (Tier 1.2)
        use_candidates     : if True, pre-filter to top-120 candidates

    Returns: (matched_groups, pred_partitions, partial_matches)
    """
    remaining_items = set(range(16))
    lives           = 4
    guessed         = set()
    matched_groups  = 0
    pred_partitions = []
    partial_matches = {4: 0, 3: 0, 2: 0, 1: 0, 0: 0}
    clf_rf, clf_gbm = clf

    while remaining_items and lives > 0:
        remaining_list = sorted(remaining_items)

        # ── Stage 1: Candidate generation ────────────────────────────────
        if use_candidates and len(remaining_items) == 16:
            subsets = generate_candidates(words, matrices, top_k=120)
            subsets = [s for s in subsets if all(i in remaining_items for i in s)]
        else:
            subsets = list(itertools.combinations(remaining_list, 4))

        if not subsets:
            break

        # ── Stage 2: ML ensemble scoring ─────────────────────────────────
        feats = [extract_features(list(S), matrices, remaining_items, words, matrix_names)
                 for S in subsets]
        feats_arr = np.array(feats, dtype=np.float32)

        probs_rf  = clf_rf.predict_proba(feats_arr)[:, 1]
        probs_gbm = clf_gbm.predict_proba(feats_arr)[:, 1]

        if clf_ranker is not None:
            probs_ranker = clf_ranker.predict(feats_arr)
            r_min, r_max = probs_ranker.min(), probs_ranker.max()
            if r_max > r_min:
                probs_ranker = (probs_ranker - r_min) / (r_max - r_min)
            # 60% Ranker + 20% RF + 20% GBM
            base_probs = 0.20 * probs_rf + 0.20 * probs_gbm + 0.60 * probs_ranker
        else:
            base_probs = (probs_rf + probs_gbm) / 2.0

        # ── Stage 2b: Competition-aware solver-stage adjustments ──────────
        score_lookup = _compute_adjusted_scores(subsets, base_probs)
        top_base_score = max(base_probs) if len(base_probs) > 0 else 0.0

        # ── Stage 3: Global joint optimisation ───────────────────────────
        # Only solve jointly if (a) 16 words remain AND (b) confidence in
        # the top individual candidate is low. If top_score >= 0.70, greedily 
        # commit to the model's first choice as it's likely correct.
        CONFIDENCE_THRESHOLD = 0.70

        if len(remaining_items) == 16 and not guessed and top_base_score < CONFIDENCE_THRESHOLD:
            # Filter to top 250 for the solver to keep search efficient
            solver_subsets = sorted(subsets, key=lambda s: score_lookup.get(s, 0.0), reverse=True)[:250]
            best_partition, solver_score = solve_ml(solver_subsets, score_lookup)
            if best_partition:
                # Submit groups in decreasing score order
                ordered_groups = sorted(best_partition, key=lambda g: score_lookup.get(g, 0.0), reverse=True)
                for group_tuple in ordered_groups:
                    if group_tuple in guessed: continue
                    guessed.add(group_tuple)
                    guess_set = set(group_tuple)
                    pred_partitions.append(guess_set)
                    max_overlap = max(len(guess_set & gt) for gt in gt_partitions_list)
                    partial_matches[max_overlap] += 1
                    if guess_set in gt_partitions_list:
                        matched_groups += 1
                        remaining_items -= guess_set
                    else:
                        lives -= 1
                        if lives == 0: break
                continue

        # ── Greedy fallback (used for sub-puzzles after first solve) ──────
        scored = sorted(zip(subsets, [score_lookup.get(s, 0.0) for s in subsets]),
                        key=lambda x: x[1], reverse=True)

        made_guess = False
        for subset, prob in scored:
            if subset in guessed:
                continue
            made_guess = True
            guessed.add(subset)
            guess_set = set(subset)
            pred_partitions.append(guess_set)

            max_overlap = max(len(guess_set & gt) for gt in gt_partitions_list)
            partial_matches[max_overlap] += 1

            if guess_set in gt_partitions_list:
                matched_groups += 1
                remaining_items -= guess_set
                break
            else:
                lives -= 1
                if lives == 0:
                    break

        if not made_guess:
            break

    return matched_groups, pred_partitions, partial_matches


# ── Feature importance output ─────────────────────────────────────────────────

def save_feature_weights(clf_rf, clf_gbm, feature_names,
                         output_path="feature_weights.txt"):
    """
    Write a ranked feature-importance table (Gini impurity decrease) to a file.
    """
    rf_imp  = clf_rf.feature_importances_
    gbm_imp = clf_gbm.feature_importances_
    avg_imp = (rf_imp + gbm_imp) / 2.0
    order   = np.argsort(avg_imp)[::-1]

    lines = [
        "=" * 72,
        "  FEATURE IMPORTANCES  (Gini impurity decrease, normalised to sum=1)",
        "=" * 72,
        f"{'Rank':<5} {'Feature':<42} {'RF':>7} {'GBM':>7} {'Avg':>7}",
        "-" * 72,
    ]
    for rank, idx in enumerate(order, 1):
        fname = feature_names[idx] if idx < len(feature_names) else f"feature_{idx}"
        lines.append(
            f"{rank:<5} {fname:<42} {rf_imp[idx]:>7.4f} {gbm_imp[idx]:>7.4f} {avg_imp[idx]:>7.4f}"
        )
    lines += [
        "=" * 72,
        f"RF  importances sum  : {rf_imp.sum():.6f}",
        f"GBM importances sum  : {gbm_imp.sum():.6f}",
    ]

    with open(output_path, "w") as f:
        f.write("\n".join(lines) + "\n")

    print(f"Feature importances saved → {output_path}")


# ── Core train + evaluate loop ────────────────────────────────────────────────

def train_and_evaluate(all_games_for_train, eval_games, embedder,
                       use_numberbatch=True, label=""):
    """
    Train RF+GBM on all_games_for_train, evaluate on eval_games.
    Returns (game_acc, group_acc, clf_rf, clf_gbm, feature_names).
    """
    print(f"\n{'='*60}")
    print(f"  Mode: {label}")
    print(f"{'='*60}")

    X_train, y_train = [], []
    mat_names_ref = None

    for game in tqdm(all_games_for_train, desc="Feature extraction (train)"):
        words = game["words"]
        gt_partitions = [
            set(words.index(w) for w in gw)
            for gw in game["groups"].values()
        ]

        mat_names, mats = build_matrices(embedder, words,
                                         use_numberbatch=use_numberbatch)
        if mat_names_ref is None:
            mat_names_ref = mat_names

        all_subsets = list(itertools.combinations(range(16), 4))

        # Positives
        for p in gt_partitions:
            X_train.append(extract_features(list(p), mats, set(range(16)), words, mat_names))
            y_train.append(1)

        # Hard negatives (3-overlap with a ground-truth group)
        hard_neg = [S for S in all_subsets
                    if set(S) not in gt_partitions
                    and max(len(set(S) & gt) for gt in gt_partitions) == 3]
        np.random.shuffle(hard_neg)
        for S in hard_neg[:8]:
            X_train.append(extract_features(list(S), mats, set(range(16)), words, mat_names))
            y_train.append(0)

        # Easy random negatives
        easy_neg = [S for S in all_subsets
                    if set(S) not in gt_partitions and S not in hard_neg]
        np.random.shuffle(easy_neg)
        for S in easy_neg[:8]:
            X_train.append(extract_features(list(S), mats, set(range(16)), words, mat_names))
            y_train.append(0)

    feature_names = build_feature_names(mat_names_ref)
    n_pos, n_neg  = sum(y_train), len(y_train) - sum(y_train)
    print(f"Training set: {len(X_train)} samples  ({n_pos} pos / {n_neg} neg)")

    print("Training Random Forest...")
    clf_rf = RandomForestClassifier(n_estimators=200, max_depth=10,
                                    random_state=42, n_jobs=-1)
    clf_rf.fit(X_train, y_train)

    print("Training Gradient Boosting...")
    clf_gbm = GradientBoostingClassifier(n_estimators=200, max_depth=3,
                                         learning_rate=0.1, subsample=0.8,
                                         random_state=42)
    clf_gbm.fit(X_train, y_train)

    clf = (clf_rf, clf_gbm)

    # Evaluate
    exact_matches, total_groups = 0, 0
    total_games = len(eval_games)
    all_partial = {4:0, 3:0, 2:0, 1:0, 0:0}
    total_guesses = 0

    for idx, game in enumerate(tqdm(eval_games, desc="Evaluating")):
        words       = game["words"]
        group_names = game["group_names"]
        gt_partitions = []
        gt_names_list = []
        for level, gw in game["groups"].items():
            gt_partitions.append({words.index(w) for w in gw})
            gt_names_list.append(group_names[level])

        mat_names, mats = build_matrices(embedder, words,
                                          use_numberbatch=use_numberbatch)

        matched, preds, partial = simulate_game_ml(
            words, mats, mat_names, clf, gt_partitions)

        total_groups += matched
        for k, v in partial.items():
            all_partial[k] += v
        total_guesses += len(preds)
        if matched == 4:
            exact_matches += 1

        if idx < 2:
            print(f"\nGame {idx+1}")
            for p, name in zip(gt_partitions, gt_names_list):
                print(f"  GT : {[words[i] for i in p]}  →  {name}")
            for p in preds:
                status = "✓" if p in gt_partitions else "✗"
                print(f"  {status}   {[words[i] for i in p]}")
            print(f"  Matched: {matched}/4")

    game_acc  = exact_matches / total_games
    group_acc = total_groups  / (4 * total_games)

    print(f"\n--- Results ({label}) ---")
    print(f"Games          : {total_games}")
    print(f"Perfect solves : {exact_matches}  ({game_acc*100:.2f}%)")
    print(f"Group accuracy : {group_acc*100:.2f}%")
    print(f"Guess overlap  :")
    for k in [4, 3, 2, 1, 0]:
        pct = all_partial[k]/total_guesses*100 if total_guesses else 0
        print(f"  {k}/4 overlap : {all_partial[k]}  ({pct:.1f}%)")

    return game_acc, group_acc, clf_rf, clf_gbm, feature_names


# ── Entry point ───────────────────────────────────────────────────────────────

def evaluate_games(csv_path: str, limit: int = None, ablation: bool = False):
    print("Loading data...")
    all_games = load_games(csv_path)

    train_games = all_games[:640]
    eval_games  = all_games[777:]
    if limit is not None:
        eval_games = eval_games[:limit]
    print(f"{len(eval_games)} unseen test games loaded.")

    print("Initialising embedder...")
    embedder = ConnectionsEmbedder()

    # Run with Numberbatch
    ga_with, gr_with, clf_rf, clf_gbm, feat_names = train_and_evaluate(
        train_games, eval_games, embedder,
        use_numberbatch=True,
        label="MPNet + ConceptNet Numberbatch + MultiSense + PairContext"
    )

    # Save feature importances
    save_feature_weights(clf_rf, clf_gbm, feat_names, "feature_weights.txt")

    # Ablation: without Numberbatch
    if ablation:
        ga_no, gr_no, _, _, _ = train_and_evaluate(
            train_games, eval_games, embedder,
            use_numberbatch=False,
            label="MPNet only (no Numberbatch)"
        )

        print("\n" + "="*60)
        print("  ABLATION SUMMARY")
        print("="*60)
        print(f"{'Condition':<38} {'GameAcc':>8} {'GroupAcc':>9}")
        print("-"*60)
        print(f"{'With Numberbatch':<38} {ga_with*100:>7.2f}% {gr_with*100:>8.2f}%")
        print(f"{'Without Numberbatch':<38} {ga_no*100:>7.2f}% {gr_no*100:>8.2f}%")
        dg, dgr = ga_with - ga_no, gr_with - gr_no
        print(f"{'Delta':<38} {dg*100:>+7.2f}% {dgr*100:>+8.2f}%")
        print("="*60)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv",      type=str, default="Connections_Data.csv")
    parser.add_argument("--limit",    type=int, default=None,
                        help="Limit eval games (for quick tests)")
    parser.add_argument("--ablation", action="store_true",
                        help="Also run without Numberbatch and compare accuracy")
    args = parser.parse_args()

    if not os.path.exists(args.csv):
        print(f"Error: {args.csv} not found"); sys.exit(1)

    evaluate_games(args.csv, limit=args.limit, ablation=args.ablation)
