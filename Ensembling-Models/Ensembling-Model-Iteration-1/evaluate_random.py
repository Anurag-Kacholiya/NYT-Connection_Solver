import os
import sys
import numpy as np
import pandas as pd
from tqdm import tqdm
import random

from data_loader import load_games
from embedder import ConnectionsEmbedder
import itertools

def simulate_game_greedy(sim_matrix, gt_partitions_list):
    remaining_items = set(range(16))
    lives = 4
    guessed = set()
    matched_groups = 0
    pred_partitions = []
    
    while remaining_items and lives > 0:
        subsets = list(itertools.combinations(sorted(list(remaining_items)), 4))
        
        def score_subset(S):
            return sum(sim_matrix[S[i]][S[j]] for i in range(4) for j in range(i+1, 4))
            
        subsets.sort(key=score_subset, reverse=True)
        
        made_valid_guess = False
        for subset in subsets:
            if subset in guessed:
                continue
                
            made_valid_guess = True
            guessed.add(subset)
            guess_set = set(subset)
            
            # Check if correct
            is_correct = guess_set in gt_partitions_list
            is_one_away = False
            if not is_correct:
                for gt in gt_partitions_list:
                    if len(guess_set.intersection(gt)) == 3:
                        is_one_away = True
                        break
                        
            pred_partitions.append((guess_set, is_correct, is_one_away))
            
            if is_correct:
                matched_groups += 1
                remaining_items -= guess_set
                break
            else:
                lives -= 1
                if lives == 0:
                    break
                    
        if not made_valid_guess:
            break
            
    return matched_groups, pred_partitions

def evaluate_games(csv_path: str, limit: int = None, seed: int = 42):
    print("Loading data...")
    games = load_games(csv_path)
    
    if limit is not None and limit < len(games):
        print(f"Randomly selecting {limit} games...")
        random.seed(seed)
        games = random.sample(games, limit)
    
    print(f"Loaded {len(games)} valid games to evaluate.")
    
    print("Initializing embedder...")
    embedder = ConnectionsEmbedder()
    
    exact_matches = 0
    total_groups_matched = 0
    total_games = len(games)
    
    print("Evaluating...")
    for idx, game in enumerate(tqdm(games, desc="Solving Games")):
        words = game["words"]
        
        gt_partitions = []
        for level, group_words in game["groups"].items():
            indices = {words.index(w) for w in group_words}
            gt_partitions.append(indices)
            
        embeddings = embedder.get_embeddings(words)
        
        def double_center(mat):
            r_mu = np.mean(mat, axis=1, keepdims=True)
            c_mu = np.mean(mat, axis=0, keepdims=True)
            g_mu = np.mean(mat)
            return mat - r_mu - c_mu + g_mu
            
        semantic_sim = np.dot(embeddings, embeddings.T)
        semantic_sim = double_center(semantic_sim)
        
        lexical_sim = embedder.get_lexical_similarity(words)
        lexical_sim = double_center(lexical_sim)
        
        wordnet_sim = embedder.get_wordnet_similarity(words)
        wordnet_sim = double_center(wordnet_sim)
        
        wikidata_sim = embedder.get_wikidata_similarity(words)
        wikidata_sim = double_center(wikidata_sim)
        
        sim_matrix = 0.55 * semantic_sim + 0.15 * lexical_sim + 0.15 * wordnet_sim + 0.15 * wikidata_sim
        np.fill_diagonal(sim_matrix, 0.0)
        
        matched_groups, pred_partitions = simulate_game_greedy(sim_matrix, gt_partitions)
        
        total_groups_matched += matched_groups
        if matched_groups == 4:
            exact_matches += 1
            
        print(f"\nGame {idx+1}")
        print("Ground Truth Groups:")
        for p in gt_partitions:
            print(f"  {[words[i] for i in p]}")
        print("Predictions sequence (lives used when wrong):")
        for (guess_set, is_correct, is_one_away) in pred_partitions:
            guess_words = [words[i] for i in guess_set]
            if is_correct:
                print(f"  Correct: {guess_words}")
            elif is_one_away:
                print(f"  One-Away: {guess_words}")
            else:
                print(f"  Wrong: {guess_words}")
        print(f"Matched Groups: {matched_groups}/4")
            
    game_accuracy = exact_matches / total_games
    group_accuracy = total_groups_matched / (4 * total_games)
    
    print(f"\n--- Results ---")
    print(f"Total Games: {total_games}")
    print(f"Perfectly Solved Games: {exact_matches}")
    print(f"Game Accuracy: {game_accuracy:.4f} ({game_accuracy*100:.2f}%)")
    print(f"Group Match Accuracy: {group_accuracy:.4f} ({group_accuracy*100:.2f}%)")
    
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", type=str, default="Connections_Data.csv", help="Path to Connections_Data.csv")
    parser.add_argument("--limit", type=int, default=100, help="Limit number of games to evaluate")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    args = parser.parse_args()
    
    if not os.path.exists(args.csv):
        print(f"Error: Could not find {args.csv}")
        sys.exit(1)
        
    evaluate_games(args.csv, limit=args.limit, seed=args.seed)
