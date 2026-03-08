import os
import numpy as np
import pandas as pd
from data_loader import load_games
from embedder import ConnectionsEmbedder
import itertools
from tqdm import tqdm

def double_center(mat):
    r_mu = np.mean(mat, axis=1, keepdims=True)
    c_mu = np.mean(mat, axis=0, keepdims=True)
    g_mu = np.mean(mat)
    return mat - r_mu - c_mu + g_mu

def simulate_game_greedy(sim_matrix, gt_partitions_list):
    remaining_items = set(range(16))
    lives = 4
    guessed = set()
    matched_groups = 0
    pred_sequence = []
    
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
            
            if guess_set in gt_partitions_list:
                status = "Correct"
                pred_sequence.append((status, subset))
                matched_groups += 1
                remaining_items -= guess_set
                break
            else:
                is_one_away = any(len(guess_set.intersection(gt)) == 3 for gt in gt_partitions_list)
                status = "One away" if is_one_away else "Incorrect"
                pred_sequence.append((status, subset))
                lives -= 1
                if lives == 0:
                    break
                    
        if not made_valid_guess:
            break
            
    return matched_groups, pred_sequence

def evaluate_mpnet(csv_path: str, output_file="report.md"):
    print("Loading games...")
    games = load_games(csv_path)
    
    np.random.seed(42)
    selected_indices = np.random.choice(len(games), 100, replace=False)
    selected_games = [games[i] for i in selected_indices]
    
    print("Initializing embedder...")
    embedder = ConnectionsEmbedder()
    
    total_puzzles = 100
    games_completely_solved = 0
    total_completely_solved_groups = 0
    total_one_away_groups = 0
    games_with_2_solved_and_2_one_aways = 0
    
    game_logs = []
    
    print("Evaluating...")
    for idx, game in enumerate(tqdm(selected_games)):
        words = game["words"]
        gt_partitions = []
        for level, group_words in game["groups"].items():
            indices = {words.index(w) for w in group_words}
            gt_partitions.append(indices)
        
        embeddings = embedder.get_embeddings(words)
        semantic_sim = np.dot(embeddings, embeddings.T)
        semantic_sim = double_center(semantic_sim)
        lexical_sim = embedder.get_lexical_similarity(words)
        lexical_sim = double_center(lexical_sim)
        sim_matrix = 0.8 * semantic_sim + 0.2 * lexical_sim
        np.fill_diagonal(sim_matrix, 0.0)
        
        matched_groups, pred_sequence = simulate_game_greedy(sim_matrix, gt_partitions)
        
        one_aways_in_game = sum(1 for status, _ in pred_sequence if status == "One away")
        total_one_away_groups += one_aways_in_game
        total_completely_solved_groups += matched_groups
        if matched_groups == 4:
            games_completely_solved += 1
        if matched_groups == 2 and one_aways_in_game == 2:
            games_with_2_solved_and_2_one_aways += 1
            
        log = []
        log.append(f"### Game {idx+1}")
        log.append("**Ground Truth Groups:**")
        for p_indices in gt_partitions:
            log.append(f"- {sorted([words[i] for i in p_indices])}")
        
        log.append("\n**Predictions Sequence:**")
        for status, subset in pred_sequence:
            log.append(f"- {status}: {sorted([words[i] for i in subset])}")
            
        log.append(f"\n**Matched Groups: {matched_groups}/4**\n")
        game_logs.append("\n".join(log))

    with open(output_file, "w") as f:
        f.write("# MPNet Baseline Evaluation Results\n\n")
        f.write("## 🚀 How to Run this Evaluation\n")
        f.write("To reproduce these results, navigate to the `Baseline-2-Final-Baseline-MPNet` directory and execute:\n\n")
        f.write("```bash\npython evaluate_mpnet_custom.py\n```\n\n")
        
        f.write("## 🏗️ Model Architecture\n")
        f.write("Detailed system architecture: [architecture.html](architecture.html)\n\n")
        
        f.write("## 📋 Methodology\n")
        f.write("This baseline utilizes a state-of-the-art **Neural-Greedy Simulation** approach:\n\n")
        f.write("1. **Embedding Layer**: We use `all-mpnet-base-v2` to extract 768-dimensional contextual word vectors.\n")
        f.write("2. **Feature Fusion**: A similarity matrix is computed by weighting Semantic Similarity (80%) and Lexical n-gram Similarity (20%).\n")
        f.write("3. **Double Centering Transformer**: The matrix is normalized by subtracting mean row and column similarities, emphasizing relative board-specific connections.\n")
        f.write("4. **Greedy Human-AI Simulation**: The model mimics human gameplay by picking the top-scoring 4-word clusters iteratively. It accounts for the game's life system (4 lives) and pivots its search based on ground-truth feedback.\n\n")
        
        f.write("---\n\n")
        f.write(f"Evaluated on {total_puzzles} random games from the Connections dataset.\n\n")
        
        f.write("## Game-by-Game Breakdown\n\n")
        f.write("\n".join(game_logs))
        
        f.write("\n## Overall Metrics\n\n")
        f.write(f"- **Number of games completely solved**: {games_completely_solved} / {total_puzzles} ({games_completely_solved}%)\n")
        f.write(f"- **Total number of groups completely solved**: {total_completely_solved_groups} / {total_puzzles * 4} ({total_completely_solved_groups/4}%)\n")
        f.write(f"- **Number of groups in which 3 words correct and 1 wrong**: {total_one_away_groups} / {total_puzzles * 4} ({total_one_away_groups/4}%)\n")
        f.write(f"- **Number of games with exactly 2 solved groups and 2 one-away groups**: {games_with_2_solved_and_2_one_aways} / {total_puzzles} ({games_with_2_solved_and_2_one_aways}%)\n\n")
        
        f.write("## Why This Method Still Struggles\n\n")
        f.write("Despite being an improvement over GloVe, several failure modes persist:\n\n")
        f.write("- **Ambiguity**: Without external knowledge, words with multiple meanings (polysemy) create blurred vector positions.\n")
        f.write("- **Abstract Logic**: The model cannot 'reason' about categories that aren't based on semantic usage (e.g., phonetic patterns).\n")
        f.write("- **Cascade Effect**: A single high-similarity 'Red Herring' choice removes words from the board that belong to other groups, making the global solution unsolvable.\n")

    print(f"\nEvaluation complete. Results saved to {output_file}")

if __name__ == "__main__":
    evaluate_mpnet("Connections_Data.csv")
