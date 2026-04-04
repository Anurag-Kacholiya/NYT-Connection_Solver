import os
import torch
import itertools
import numpy as np
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
from config import DEVICE, EMBEDDING_DIM, HIDDEN_DIM, NUM_RELATIONS, SAVE_PATH
from data import load_embeddings, get_train_test_splits
from graph import prepare_pyg_data
from model import RGCNSolver

def get_best_partition(pairwise_probs):
    """Greedy heuristic to find 4 groups of 4 words maximizing internal validity."""
    words_idx = set(range(16))
    groups = []
    
    for _ in range(4):
        best_clique = None
        best_score = -1
        
        for clique in itertools.combinations(words_idx, 4):
            score = sum(pairwise_probs.get((min(i,j), max(i,j)), 0) 
                        for i, j in itertools.combinations(clique, 2))
            if score > best_score:
                best_score = score
                best_clique = clique
                
        groups.append(best_clique)
        words_idx -= set(best_clique)
        
    preds = np.zeros(16, dtype=int)
    for group_idx, group in enumerate(groups):
        for word_idx in group:
            preds[word_idx] = group_idx
            
    total_score = 0
    for group in groups:
        total_score += sum(pairwise_probs.get((min(i,j), max(i,j)), 0) 
                           for i, j in itertools.combinations(group, 2))
    confidence = total_score / 24.0
    return preds, confidence

def solve_progressive(puzzle, model, conceptnet_emb):
    """Progressive tier activation based on model confidence."""
    model.eval()
    
    for tier in [1, 2, 3]:
        x, edge_index, edge_type, pair_indices, _ = prepare_pyg_data(puzzle, tier, conceptnet_emb)
        
        with torch.no_grad():
            logits = model(x, edge_index, edge_type, pair_indices)
            probs = torch.sigmoid(logits).cpu().numpy()
            
        pairs_list = pair_indices.cpu().numpy().T
        pairwise_probs = {(p[0], p[1]): prob for p, prob in zip(pairs_list, probs)}
        
        preds, confidence = get_best_partition(pairwise_probs)
        print(f"  Tier {tier} Confidence: {confidence:.2f}")
        
        if confidence > 0.70:
            return preds, tier
            
    return preds, 3

def get_group_overlaps(true_labels, preds):
    best_overlaps = []
    max_matches = -1
    
    for perm in itertools.permutations([0, 1, 2, 3]):
        mapped_preds = np.array([perm[p] for p in preds])
        overlaps = []
        
        for g in range(4):
            pred_idx = np.where(mapped_preds == g)[0]
            true_idx = np.where(np.array(true_labels) == g)[0]
            overlap = len(set(pred_idx).intersection(set(true_idx)))
            overlaps.append(overlap)
            
        total_matches = sum(overlaps)
        if total_matches > max_matches:
            max_matches = total_matches
            best_overlaps = overlaps
            
    return best_overlaps

def evaluate():
    conceptnet_emb = load_embeddings()
    _, test_puzzles = get_train_test_splits()
    
    model = RGCNSolver(in_channels=EMBEDDING_DIM, hidden=HIDDEN_DIM, num_relations=NUM_RELATIONS).to(DEVICE)
    
    # In Jupyter you used two different paths ('_skipping_newForward' and '_newForward'). 
    # Standardizing to your SAVE_PATH from config.
    if os.path.exists(SAVE_PATH):
        model.load_state_dict(torch.load(SAVE_PATH, map_location=DEVICE))
        model.eval() 
        print("Successfully loaded saved model weights!")
    else:
        print("No saved weights found. You are using an untrained model.")

    print("\nEvaluating Progressive Solver on Unseen Test Set...")

    total_groups_solved = 0
    total_groups_3_correct = 0
    games_2_perfect_2_almost = 0
    games_solved_completely = 0

    total_test_games = len(test_puzzles)

    for idx, puzzle in enumerate(test_puzzles):
        preds, final_tier = solve_progressive(puzzle, model, conceptnet_emb)
        true_labels = puzzle["labels"]
        
        ari = adjusted_rand_score(true_labels, preds)
        nmi = normalized_mutual_info_score(true_labels, preds)
        overlaps = get_group_overlaps(true_labels, preds)
        
        groups_solved = overlaps.count(4)
        groups_3_correct = overlaps.count(3)
        
        total_groups_solved += groups_solved
        total_groups_3_correct += groups_3_correct
        
        if groups_solved == 4:
            games_solved_completely += 1
        elif groups_solved == 2 and groups_3_correct == 2:
            games_2_perfect_2_almost += 1

        print(f"\nEvaluating Test Puzzle {idx+1}...")
        print(f"  Stopped at Tier: {final_tier}")
        print(f"  Adjusted Rand Index: {ari:.2f}")
        print(f"  Normalized Mutual Info: {nmi:.2f}")
        print(f"  Group Matches: {overlaps} (Total Words: {sum(overlaps)}/16)")

    total_possible_groups = total_test_games * 4

    print("\n===== FINAL EVALUATION METRICS =====")
    print("1) Grouping Accuracy:")
    print(f"   1.1) Total no. of groups solved: {total_groups_solved} (out of {total_possible_groups})")
    print(f"   1.2) No. of '3 words correct out of 4 in a group': {total_groups_3_correct}")
    print("____")
    print(f"2) No. of games in which (2 groups solved completely and remaining 2 groups are '3 words correct'): {games_2_perfect_2_almost}")
    print("____")
    print(f"3) No. of games solved completely: {games_solved_completely} (out of {total_test_games} test games)")
    print("_____")

if __name__ == "__main__":
    evaluate()