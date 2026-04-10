import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
from collections import defaultdict
from tqdm import tqdm
import itertools
from scipy.optimize import linear_sum_assignment
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split

# Install dependencies if needed (uncomment to run)
# !pip install -q sentence-transformers

def clustering_accuracy(true_labels, pred_labels):
    cm = confusion_matrix(true_labels, pred_labels)
    row_ind, col_ind = linear_sum_assignment(-cm)
    return cm[row_ind, col_ind].sum() / len(true_labels)

def get_group_overlaps(true_labels, preds):
    """
    Finds the best permutation of labels and returns the number of 
    correctly matched words for each of the 4 individual groups.
    """
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

def main():
    # Load dataset
    # Note: Update path to your local data source if not using the Kaggle path
    try:
        df = pd.read_csv("/kaggle/input/datasets/anuragkacholiya/connections-raw-data/Connections_Data.csv")
    except FileNotFoundError:
        print("Dataset not found at /kaggle/input/datasets/anuragkacholiya/connections-raw-data/Connections_Data.csv")
        print("Please update the path to point to your local Connections_Data.csv file.")
        return

    df["Word"] = df["Word"].str.upper().str.strip()

    puzzles = []

    for game_id, group in df.groupby("Game ID"):
        group = group.sort_values(["Starting Row", "Starting Column"])
        
        if len(group) == 16:
            words = group["Word"].tolist()
            labels = group["Group Level"].tolist()
            
            puzzles.append({
                "game_id": game_id,
                "words": words,
                "labels": labels
            })

    print("Total puzzles:", len(puzzles))

    # Initialize model
    model = SentenceTransformer("all-MiniLM-L6-v2")

    # Baseline 2 Calculation
    print("\nCalculating Baseline 2 Results...")
    all_ari = []
    all_nmi = []
    all_acc = []

    for puzzle in tqdm(puzzles):
        words = puzzle["words"]
        true_labels = puzzle["labels"]
        
        embeddings = model.encode(words)
        
        clustering = AgglomerativeClustering(
            n_clusters=4,
            metric="cosine",
            linkage="average"
        )
        
        pred_labels = clustering.fit_predict(embeddings)
        
        ari = adjusted_rand_score(true_labels, pred_labels)
        nmi = normalized_mutual_info_score(true_labels, pred_labels)
        acc = clustering_accuracy(true_labels, pred_labels)
        
        all_ari.append(ari)
        all_nmi.append(nmi)
        all_acc.append(acc)

    print("\n===== BASELINE 2 RESULTS =====")
    print("Mean ARI :", np.mean(all_ari))
    print("Mean NMI :", np.mean(all_nmi))
    print("Mean Accuracy :", np.mean(all_acc))

    # Test on a single puzzle
    if puzzles:
        p = puzzles[0]
        words = p["words"]
        true = p["labels"]

        embeddings = model.encode(words)
        clustering = AgglomerativeClustering(n_clusters=4, metric="cosine", linkage="average")
        pred = clustering.fit_predict(embeddings)

        print("\nSingle Puzzle Test Case:")
        print("Words:", words)
        print("True :", true)
        print("Pred :", pred)

    # Final Evaluation Metrics
    total_groups_solved = 0
    total_groups_3_correct = 0
    games_2_perfect_2_almost = 0
    games_solved_completely = 0

    # For testing, we split the puzzles but in the original script it used test_size=914 of 915? 
    # Let's replicate original logic if possible, which seemed to use a test set.
    _, test_puzzles = train_test_split(puzzles, test_size=min(914, len(puzzles)-1), random_state=42)

    eval_puzzles = test_puzzles 
    total_test_games = len(eval_puzzles)

    print(f"\nEvaluating Baseline 1 on {total_test_games} puzzles...")

    for puzzle in tqdm(eval_puzzles):
        words = puzzle["words"]
        true_labels = puzzle["labels"]
        
        embeddings = model.encode(words)
        clustering = AgglomerativeClustering(
            n_clusters=4,
            metric="cosine",
            linkage="average"
        )
        pred_labels = clustering.fit_predict(embeddings)
        
        overlaps = get_group_overlaps(true_labels, pred_labels)
        
        groups_solved = overlaps.count(4)
        groups_3_correct = overlaps.count(3)
        
        total_groups_solved += groups_solved
        total_groups_3_correct += groups_3_correct
        
        if groups_solved == 4:
            games_solved_completely += 1
        elif groups_solved == 2 and groups_3_correct == 2:
            games_2_perfect_2_almost += 1

    total_possible_groups = total_test_games * 4

    print("\n===== FINAL EVALUATION METRICS =====")
    print("1) Grouping Accuracy :")
    print(f"   1.1) total no. of group solved: {total_groups_solved} (out of {total_possible_groups})")
    print(f"   1.2) no. of '3 words correct out of 4 in a group': {total_groups_3_correct}")
    print("____")
    print(f"2) no. of games in which (2 groups solved completely and remaining 2 groups are '3 words correct out of 4 in a group'): {games_2_perfect_2_almost}")
    print("____")
    print(f"3) no. of games solved completely (out of {total_test_games} test games): {games_solved_completely}")
    print("_____")

if __name__ == "__main__":
    main()
