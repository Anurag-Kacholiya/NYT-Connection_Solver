import itertools
import numpy as np
from typing import List, Set, Tuple, Dict, Any

# ─────────────────────────────────────────────────────────────────────────────
# Public API: ML-score-driven joint optimiser  (NEW)
# ─────────────────────────────────────────────────────────────────────────────

def solve_ml(score_lookup: Dict[Tuple[int, ...], float], items: Set[int]):
    """
    Given a lookup of group-of-4 scores, find the partition of 16 items 
    that maximizes the TOTAL score.
    """
    sorted_items = sorted(list(items))
    if len(sorted_items) == 0:
        return [], 0.0
    if len(sorted_items) == 4:
        combo = tuple(sorted_items)
        return [combo], score_lookup.get(combo, -1e9)

    # Recursive search (DP approach)
    memo = {}

    def get_best(current_items):
        if not current_items:
            return [], 0.0
        
        state = tuple(sorted(list(current_items)))
        if state in memo:
            return memo[state]
            
        best_p, best_s = [], -1e12
        
        # Take the first item and find all possible groups of 4 containing it
        first = sorted_items[sorted_items.index(min(current_items))]
        other_items = [i for i in current_items if i != first]
        
        for others in itertools.combinations(other_items, 3):
            group = tuple(sorted((first,) + others))
            group_score = score_lookup.get(group, -1e9)
            
            # Recursive step
            rem = set(current_items) - set(group)
            sub_p, sub_s = get_best(rem)
            
            total_s = group_score + sub_s
            if total_s > best_s:
                best_s = total_s
                best_p = [group] + sub_p
        
        memo[state] = (best_p, best_s)
        return best_p, best_s

    return get_best(set(sorted_items))

class InteractiveSolver:
    """
    State-of-the-art Interactive Solver (Path A Strategy).
    Maintains a list of constraints (Correct, One Away, Wrong) and 
    suggests the next optimal move using Global Constrained Optimization.
    """
    def __init__(self, words: List[str], clf_ranker):
        self.words = [w.upper() for w in words]
        self.clf_ranker = clf_ranker
        self.remaining_indices = set(range(16))
        
        self.correct_groups = [] # List of sets
        self.blacklisted_groups = [] # List of sets (Incorrect guesses)
        self.one_away_hints = [] # List of sets (Guesses that were 'One Away')
        
    def add_feedback(self, indices: Tuple[int, ...], result: str):
        guess_set = set(indices)
        if result == 'correct':
            self.correct_groups.append(guess_set)
            self.remaining_indices -= guess_set
        elif result == 'one_away':
            self.one_away_hints.append(guess_set)
            self.blacklisted_groups.append(guess_set)
        elif result == 'incorrect':
            self.blacklisted_groups.append(guess_set)

    def get_next_suggestions(self, mats: Dict[str, np.ndarray], mat_names: List[str], top_k: int = 5):
        """
        Runs the Global Optimizer under current constraints.
        """
        from evaluate import extract_features
        
        # 1. Build score lookup for all possible groups of remaining words
        candidates = list(itertools.combinations(sorted(list(self.remaining_indices)), 4))
        if not candidates: return []
        
        X = []
        for cand in candidates:
            feat = extract_features(list(cand), mats, self.remaining_indices, self.words, mat_names)
            X.append(feat)
        X = np.array(X)
        scores = self.clf_ranker.predict(X)
        
        score_lookup = {}
        for i, cand in enumerate(candidates):
            cand_set = set(cand)
            if cand_set in self.blacklisted_groups:
                score_lookup[cand] = -1e6
            else:
                s = scores[i]
                # Apply 'One Away' boosts
                for oa_set in self.one_away_hints:
                    if len(cand_set & oa_set) == 3:
                        s += 0.5 
                score_lookup[cand] = s
        
        # 2. Global solve
        best_partition, _ = solve_ml(score_lookup, self.remaining_indices)
        
        if not best_partition:
            return []
            
        suggestions = []
        for group in best_partition:
            suggestions.append({
                'indices': group,
                'words': [self.words[i] for i in group],
                'score': score_lookup.get(tuple(sorted(group)), 0.0)
            })
            
        return sorted(suggestions, key=lambda x: x['score'], reverse=True)

# ─── Legacy Solvers (for compatibility with train.py) ───
def solve(sim_matrix):
    # This was the old 31% solver logic — retained for comparison if needed
    from solver_old import solve_legacy
    return solve_legacy(sim_matrix)
