import os
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
import zipfile
import urllib.request
from scipy.optimize import linear_sum_assignment
from scipy.spatial.distance import cdist

def download_and_extract_glove():
    zip_path = "glove.6B.zip"
    txt_path = "glove.6B.300d.txt"
    
    if not os.path.exists(txt_path):
        if not os.path.exists(zip_path):
            print("Downloading GloVe 6B zip file (this might take a while)...")
            url = "https://huggingface.co/stanfordnlp/glove/resolve/main/glove.6B.zip"
            urllib.request.urlretrieve(url, zip_path)
            print("Download complete.")
        
        print(f"Extracting {txt_path}...")
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extract(txt_path)
        print("Extraction complete.")
    return txt_path

def load_glove(path):
    print("Loading GloVe embeddings...")
    embeddings = {}
    with open(path, 'r', encoding='utf8') as f:
        for line in f:
            values = line.strip().split()
            word = values[0]
            vector = np.array(values[1:], dtype='float32')
            embeddings[word] = vector
    print("GloVe embeddings loaded.")
    return embeddings

glove_path = download_and_extract_glove()
glove = load_glove(glove_path)
embedding_dim = 300

def get_word_vector(word):
    if not isinstance(word, str):
        return np.zeros(embedding_dim)
    word = word.lower()
    if word in glove:
        return glove[word]
    if " " in word:
        parts = word.split()
        vectors = [glove[p] for p in parts if p in glove]
        if vectors:
            return np.mean(vectors, axis=0)
    return np.zeros(embedding_dim)

def glove_kmeans_solver(words):
    vectors = np.array([get_word_vector(w) for w in words])
    
    # 1. Run standard K-Means to find initial centroids
    kmeans = KMeans(n_clusters=4, random_state=42, n_init=10)
    kmeans.fit(vectors)
    centroids = kmeans.cluster_centers_
    
    # 2. Duplicate each centroid 4 times to create 16 slots (4 per cluster)
    # Each slot represents a 'seat' in a specific cluster
    expanded_centroids = np.repeat(centroids, 4, axis=0) # Shape: (16, 300)
    
    # 3. Compute cost matrix (Euclidean distance between 16 words and 16 slots)
    cost_matrix = cdist(vectors, expanded_centroids, metric='euclidean')
    
    # 4. Use Linear Sum Assignment (Hungarian Algorithm) to find the optimal 1-to-1 matching
    # This minimizes the total distance while ensuring each word gets 1 slot
    # and each cluster gets exactly 4 words.
    row_ind, col_ind = linear_sum_assignment(cost_matrix)
    
    groups = [[] for _ in range(4)]
    for word_idx, slot_idx in zip(row_ind, col_ind):
        cluster_idx = slot_idx // 4 # Map slot back to cluster 0-3
        groups[cluster_idx].append(words[word_idx])
        
    return groups

def evaluate_solver(df, solver_function, output_file="results.md"):
    # Filter valid games (must have exactly 16 words)
    valid_games = []
    for date, puzzle_data in df.groupby('Puzzle Date'):
        if len(puzzle_data) == 16:
            valid_games.append(date)
    
    # Randomly select 100 games
    np.random.seed(42)  # For reproducibility
    selected_games = np.random.choice(valid_games, 100, replace=False)
    
    # Initialize metric counters
    total_puzzles = 100
    games_completely_solved = 0
    total_completely_solved_groups = 0
    total_one_away_groups = 0
    games_with_2_solved_and_2_one_aways = 0
    
    with open(output_file, "w") as f:
        f.write("# GloVe 300d + KMeans Baseline Evaluation Results\n\n")
        f.write(f"Evaluated on {total_puzzles} randomly selected games.\n\n")
        f.write("## Game-by-Game Breakdown\n\n")
        
        for date in selected_games:
            puzzle_data = df[df['Puzzle Date'] == date]
            words = puzzle_data['Word'].tolist()
            
            # Extract ground truth sets
            ground_truth_sets = []
            for category, group in puzzle_data.groupby('Group Name'):
                ground_truth_sets.append(set(group['Word']))
                
            # Run model
            predicted_groups = solver_function(words)
            predicted_sets = [set(g) for g in predicted_groups]
            
            completely_solved_groups_in_game = 0
            one_aways_in_game = 0
            
            # Analyze each predicted group against ground truth
            for p_set in predicted_sets:
                max_intersection = 0
                for gt_set in ground_truth_sets:
                    intersect = len(p_set.intersection(gt_set))
                    if intersect > max_intersection:
                        max_intersection = intersect
                
                if max_intersection == 4:
                    completely_solved_groups_in_game += 1
                    total_completely_solved_groups += 1
                elif max_intersection == 3:
                    one_aways_in_game += 1
                    total_one_away_groups += 1
                    
            # Check game-level conditions
            if completely_solved_groups_in_game == 4:
                games_completely_solved += 1
                
            if completely_solved_groups_in_game == 2 and one_aways_in_game == 2:
                games_with_2_solved_and_2_one_aways += 1
                
            # Log individual game results
            f.write(f"### Puzzle Date: {date}\n")
            f.write(f"- Completely solved groups: {completely_solved_groups_in_game}\n")
            f.write(f"- One-away groups: {one_aways_in_game}\n\n")

        # Summary
        f.write("## Overall Metrics\n\n")
        f.write(f"- **Number of games completely solved**: {games_completely_solved} / {total_puzzles}\n")
        f.write(f"- **Total number of groups completely solved**: {total_completely_solved_groups} / {total_puzzles * 4}\n")
        f.write(f"- **Number of groups in which 3 words correct and 1 wrong**: {total_one_away_groups} / {total_puzzles * 4}\n")
        f.write(f"- **Number of games with exactly 2 solved groups and 2 one-away groups**: {games_with_2_solved_and_2_one_aways} / {total_puzzles}\n")
        
    print(f"\\nEvaluation complete. Results saved to {output_file}")


if __name__ == '__main__':
    # Locate data file
    data_path = "../Final-Baseline-MPNet/Connections_Data.csv"
    if not os.path.exists(data_path):
        data_path = "../../Ensembling-Models/Ensembling-Model-Iteration-1/Connections_Data.csv"
        
    if not os.path.exists(data_path):
        print("Could not find Connections_Data.csv")
    else:
        df = pd.read_csv(data_path)
        df = df.dropna(subset=['Word'])
        df['Word'] = df['Word'].astype(str)
        evaluate_solver(df, glove_kmeans_solver, output_file="results.md")
