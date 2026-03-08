import os
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
import zipfile
import urllib.request
import itertools
from tqdm import tqdm

def download_and_extract_glove():
    zip_path = "glove.6B.zip"
    txt_path = "glove.6B.300d.txt"
    if not os.path.exists(txt_path):
        if not os.path.exists(zip_path):
            print("Downloading GloVe 6B zip file...")
            url = "https://huggingface.co/stanfordnlp/glove/resolve/main/glove.6B.zip"
            urllib.request.urlretrieve(url, zip_path)
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extract(txt_path)
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
    return embeddings

# Global loading
glove_path = download_and_extract_glove()
glove = load_glove(glove_path)
embedding_dim = 300

def get_word_vector(word):
    word = str(word).lower()
    if word in glove:
        return glove[word]
    if " " in word:
        parts = word.split()
        vectors = [glove[p] for p in parts if p in glove]
        if vectors: return np.mean(vectors, axis=0)
    return np.zeros(embedding_dim)

def simulate_game_greedy(words, gt_sets):
    vectors = np.array([get_word_vector(w) for w in words])
    
    # Simulating lives and iterative guesses
    remaining_indices = set(range(16))
    lives = 4
    guessed = set()
    matched_groups = 0
    pred_sequence = []
    
    # Precompute pairwise similarities (cosine similarity or negative euclidean)
    # Using negative euclidean for consistency with K-Means logic
    from scipy.spatial.distance import pdist, squareform
    dist_matrix = squareform(pdist(vectors, metric='euclidean'))
    sim_matrix = -dist_matrix # Higher is better
    
    while remaining_indices and lives > 0:
        current_indices = sorted(list(remaining_indices))
        if len(current_indices) < 4: break
        
        subsets = list(itertools.combinations(current_indices, 4))
        
        # Score each subset by internal similarity
        def score_subset(S):
            return sum(sim_matrix[S[i]][S[j]] for i in range(4) for j in range(i+1, 4))
            
        subsets.sort(key=score_subset, reverse=True)
        
        made_valid_guess = False
        for subset in subsets:
            if subset in guessed: continue
            
            made_valid_guess = True
            guessed.add(subset)
            guess_words = {words[i] for i in subset}
            
            # Check against Ground Truth
            is_correct = False
            for gt in gt_sets:
                if guess_words == gt:
                    is_correct = True
                    break
            
            if is_correct:
                pred_sequence.append(("Correct", subset))
                matched_groups += 1
                remaining_indices -= set(subset)
                break
            else:
                is_one_away = any(len(guess_words.intersection(gt)) == 3 for gt in gt_sets)
                status = "One away" if is_one_away else "Incorrect"
                pred_sequence.append((status, subset))
                lives -= 1
                if lives == 0: break
                
        if not made_valid_guess: break
        
    return matched_groups, pred_sequence

def evaluate_solver(df, output_file="results.md"):
    valid_games = [date for date, data in df.groupby('Puzzle Date') if len(data) == 16]
    np.random.seed(42)
    selected_dates = np.random.choice(valid_games, 100, replace=False)
    
    total_puzzles = 100
    games_solved = 0
    total_groups = 0
    total_one_away = 0
    dual_solved_count = 0
    
    game_logs = []
    
    print("Evaluating...")
    for idx, date in enumerate(tqdm(selected_dates)):
        puzzle_data = df[df['Puzzle Date'] == date]
        words = puzzle_data['Word'].tolist()
        
        gt_sets = [set(group['Word']) for _, group in puzzle_data.groupby('Group Name')]
        
        matched, sequence = simulate_game_greedy(words, gt_sets)
        
        # Metrics
        one_aways = sum(1 for status, _ in sequence if status == "One away")
        total_one_away += one_aways
        total_groups += matched
        if matched == 4: games_solved += 1
        if matched == 2 and one_aways == 2: dual_solved_count += 1
        
        # Log
        log = [f"### Game {idx+1}"]
        log.append("**Ground Truth Groups:**")
        for gt in gt_sets:
            log.append(f"- {sorted(list(gt))}")
            
        log.append("\n**Predictions Sequence:**")
        for status, subset in sequence:
            log.append(f"- {status}: {sorted([words[i] for i in subset])}")
            
        log.append(f"\n**Matched Groups: {matched}/4**\n")
        game_logs.append("\n".join(log))

    # Writing final results.md
    with open(output_file, "w") as f:
        f.write("# GloVe 300d + KMeans Baseline Evaluation Results\n\n")
        f.write("## How to Run this Evaluation\n")
        f.write("To reproduce these results, navigate to the `Baseline-0-Glove-KMeans` directory and execute:\n\n")
        f.write("```bash\npython Glove_Kmeans.py\n```\n\n")
        
        f.write("## Model Architecture\n")
        f.write("You can view the detailed system architecture for this GloVe-based baseline here: [architecture.html](architecture.html)\n\n")
        
        f.write("## Methodology\n")
        f.write("The evaluation of this GloVe-based baseline follows a structured 4-stage pipeline:\n\n")
        f.write("1. **Static Embedding Retrieval**: Words are mapped to 300-dimensional vectors using the pre-trained GloVe 6B dataset.\n")
        f.write("2. **Greedy Simulation Environment**: To mimic human gameplay, the model iteratively selects the highest-similarity word sets.\n")
        f.write("3. **Iterative Error Handling**: The simulation accounts for 4 lives and 'one-away' feedback, allowing for a realistic assessment of iterative performance.\n")
        f.write("4. **Metric Calculation**: Performance is measured across perfect solves, group intersections, and red-herring susceptibility.\n\n")
        
        f.write("---\n\n")
        f.write(f"Evaluated on {total_puzzles} randomly selected games.\n\n")
        f.write("## Game-by-Game Breakdown\n\n")
        f.write("\n".join(game_logs))
        
        f.write("\n## Overall Metrics\n\n")
        f.write(f"- **Number of games completely solved**: {games_solved} / {total_puzzles} ({games_solved}%)\n")
        f.write(f"- **Total number of groups completely solved**: {total_groups} / {total_puzzles * 4} ({total_groups/4}%)\n")
        f.write(f"- **Number of groups in which 3 words correct and 1 wrong**: {total_one_away} / {total_puzzles * 4} ({total_one_away/4}%)\n")
        f.write(f"- **Number of games with exactly 2 solved groups and 2 one-away groups**: {dual_solved_count} / {total_puzzles} ({dual_solved_count}%)\n\n")
        
        f.write("## Why This Method Still Struggles\n\n")
        f.write("1. **Static Embeddings**: GloVe cannot distinguish between different senses of a word (polysemy), making it vulnerable to ambiguous Connections categories.\n")
        f.write("2. **Semantic Mismatch**: The model relies on co-occurrence data, which often misses the clever, lateral, or phonetic relationships that characterize NYT puzzles.\n")

    print(f"\nEvaluation complete. Results saved to {output_file}")

if __name__ == '__main__':
    data_path = "../Baseline-2-Final-Baseline-MPNet/Connections_Data.csv"
    if not os.path.exists(data_path):
        data_path = "../../Ensembling-Models/Ensembling-Model-Iteration-1/Connections_Data.csv"
    
    if os.path.exists(data_path):
        df = pd.read_csv(data_path).dropna(subset=['Word'])
        evaluate_solver(df, output_file="results.md")
    else:
        print("Data file not found.")
