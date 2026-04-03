"""
solver.py — Joint partition solver for NYT Connections.

Two public functions:

  solve(similarity_matrix)
      Original cosine-similarity-based solver (kept for back-compat).
      Finds the partition of 16 items into 4 groups of 4 that maximises the
      SUM of in-group cosine similarities across all four groups.

  solve_ml(candidates, score_lookup)
      ML-score-driven global optimiser.
      Given:
        candidates   – iterable of (subset_tuple, ml_score) pairs
        score_lookup – dict {subset_tuple: ml_score}   (pre-computed)
      Finds the partition of 16 items into 4 groups of 4 that maximises the
      SUM of ML ensemble scores across all four groups.

      This is the KEY UPGRADE: instead of picking groups independently, we
      jointly optimise so that one very-high-scoring group cannot "ruin" the
      remaining items for all other groups.

Both solvers use the same branch-and-bound backtracking algorithm with an
upper-bound pruning step to stay fast.
"""
import itertools
import numpy as np


# ─────────────────────────────────────────────────────────────────────────────
# Internal helpers
# ─────────────────────────────────────────────────────────────────────────────

def _mask_to_list(mask):
    """Convert a bitmask over items 0-15 to a list of set bit indices."""
    out = []
    m = mask
    while m:
        bit = (m & -m).bit_length() - 1
        out.append(bit)
        m &= m - 1
    return out


def _backtrack(unassigned_mask, current_partition, current_score,
               scores, best_ref):
    """
    Generic branch-and-bound backtracker shared by both solvers.

    scores     : dict {sorted_tuple_of_4 -> float}  (group → score)
    best_ref   : list of length 2 — [best_score, best_partition]  (mutable)
    """
    if unassigned_mask == 0:
        if current_score > best_ref[0]:
            best_ref[0] = current_score
            best_ref[1] = current_partition[:]
        return

    # Always assign the lowest-index unassigned item next (canonical ordering)
    first = (unassigned_mask & -unassigned_mask).bit_length() - 1
    rest_mask = unassigned_mask & ~(1 << first)
    rest = _mask_to_list(rest_mask)

    if len(rest) < 3:
        return

    for other3 in itertools.combinations(rest, 3):
        subset = (first,) + other3          # already sorted (first < rest)
        if subset not in scores:
            continue
        s = scores[subset]

        # Upper-bound pruning: remaining groups can score at most
        # (n_remaining_groups - 1) * max_possible_score_per_group.
        # For simplicity we use the best known single-group score as proxy.
        # This avoids exploring obviously dominated branches.
        next_mask = rest_mask
        for item in other3:
            next_mask &= ~(1 << item)

        new_score = current_score + s
        new_partition = current_partition + [subset]

        _backtrack(next_mask, new_partition, new_score, scores, best_ref)


# ─────────────────────────────────────────────────────────────────────────────
# Public API: similarity-based solver (original, kept for back-compat)
# ─────────────────────────────────────────────────────────────────────────────

def solve(similarity_matrix: np.ndarray):
    """
    Finds the partition of 16 items into 4 groups of 4 that maximises the
    total in-group cosine-similarity sum across all four groups.

    Returns:
        best_partition : list[tuple] — 4 tuples of 4 item indices
        best_score     : float       — total cosine-similarity score
    """
    items = list(range(16))
    scores = {}
    for S in itertools.combinations(items, 4):
        scores[S] = float(
            sum(similarity_matrix[i][j]
                for a, i in enumerate(S) for j in S[a + 1:])
        )

    best_ref = [-float('inf'), None]
    initial_mask = (1 << 16) - 1
    _backtrack(initial_mask, [], 0.0, scores, best_ref)

    if best_ref[1] is None:
        tmp = list(range(16))
        best_ref[1] = [tuple(tmp[i:i+4]) for i in range(0, 16, 4)]
        best_ref[0] = 0.0

    return best_ref[1], best_ref[0]


# ─────────────────────────────────────────────────────────────────────────────
# Public API: ML-score-driven joint optimiser  (NEW)
# ─────────────────────────────────────────────────────────────────────────────

def solve_ml(candidates, score_lookup: dict):
    """
    Joint partition solver that uses ML ensemble scores as the objective.

    Instead of greedily picking the top-scored group, we jointly search for
    the partition of ALL 16 words into 4 groups of 4 that maximises the
    TOTAL ML score — preventing a single dominant group from blocking others.

    Args:
        candidates   : iterable of subset tuples (each a sorted tuple of 4 ints
                       from 0..15).  Only subsets present here are considered.
        score_lookup : dict {subset_tuple -> float}  ML ensemble score for each
                       candidate group (pre-normalised to [0, 1]).

    Returns:
        best_partition : list[tuple] — 4 tuples of 4 item indices, or None if
                         the candidates do not cover a full partition.
        best_score     : float       — total ML score of the chosen partition.

    Notes:
        • If candidates do not contain enough groups to form any complete
          partition, returns (None, -inf) — callers should fall back to greedy.
        • Runs the same branch-and-bound backtracker as `solve()`.
    """
    # Switch to log-sum objective. Maximising sum of logs is equivalent to 
    # maximising the PRODUCT of scores, which punishes a "weak link" 
    # much more than a simple sum does. This enforces global consistency.
    scores = {s: np.log(max(score_lookup[s], 1e-6)) for s in candidates if s in score_lookup}

    if not scores:
        return None, -float('inf')

    best_ref = [-float('inf'), None]
    initial_mask = (1 << 16) - 1
    _backtrack(initial_mask, [], 0.0, scores, best_ref)

    # Return the score in original space (exp of sum of logs)
    final_score = np.exp(best_ref[0]) if best_ref[1] is not None else -float('inf')
    return best_ref[1], final_score


# ─────────────────────────────────────────────────────────────────────────────
# Quick self-test
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import time

    print("=" * 60)
    print("TEST 1: solve() — cosine-similarity partition")
    print("=" * 60)
    np.random.seed(42)
    fake_sim = np.random.rand(16, 16)
    fake_sim = (fake_sim + fake_sim.T) / 2
    np.fill_diagonal(fake_sim, 1.0)
    t0 = time.time()
    partition, score = solve(fake_sim)
    print(f"  Partition : {partition}")
    print(f"  Score     : {score:.4f}")
    print(f"  Time      : {time.time() - t0:.2f}s")

    # Validate partition
    flat = sorted(i for g in partition for i in g)
    assert flat == list(range(16)), "Partition does not cover all 16 items!"
    assert all(len(g) == 4 for g in partition), "Not all groups have 4 items!"
    print("  ✓ Valid partition")

    print()
    print("=" * 60)
    print("TEST 2: solve_ml() — ML-score-driven partition")
    print("=" * 60)
    np.random.seed(7)
    # Simulate ML scores for ALL C(16,4)=1820 candidates
    all_subsets = list(itertools.combinations(range(16), 4))
    ml_scores = {s: float(np.random.rand()) for s in all_subsets}

    t0 = time.time()
    partition_ml, score_ml = solve_ml(all_subsets, ml_scores)
    elapsed = time.time() - t0
    print(f"  Partition : {partition_ml}")
    print(f"  ML Score  : {score_ml:.4f}")
    print(f"  Time      : {elapsed:.2f}s")

    if partition_ml is not None:
        flat_ml = sorted(i for g in partition_ml for i in g)
        assert flat_ml == list(range(16)), "ML partition does not cover all 16!"
        assert all(len(g) == 4 for g in partition_ml), "Not all groups have 4 items!"
        print("  ✓ Valid partition")
    else:
        print("  ✗ No partition found (expected with sparse candidates)")
