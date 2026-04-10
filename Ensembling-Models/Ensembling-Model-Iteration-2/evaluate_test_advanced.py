import os
import lightgbm as lgb
os.environ['KMP_DUPLICATE_LIB_OK']='True'
os.environ['OMP_NUM_THREADS']='1'

import joblib
import numpy as np
import pandas as pd
import itertools
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
from embedder import ConnectionsEmbedder
from evaluate import build_matrices, extract_features

def evaluate_test_advanced(test_games, embedder, rf_path="models/clf_rf.pkl", gbm_path="models/clf_gbm.pkl", lgbm_path="models/clf_lgbm_ranker.pkl"):
    """
    Advanced simulation:
    - 4-Life Win Rate.
    - Search Depth (Unlimited tries) to find each of the 4 colors.
    - Total Guesses distribution.
    """
    print("\n" + "="*80)
    print("  🚀 [ADVANCED EVALUATION] STARTING SEARCH EFFICIENCY SIMULATION")
    print(f"  Using Models from: {os.path.dirname(rf_path)}")
    print("="*80)

    # 1. Load Models
    try:
        clf_rf = joblib.load(rf_path)
        clf_gbm = joblib.load(gbm_path)
        clf_lgbm = joblib.load(lgbm_path)
        print("✅ Models loaded successfully.")
    except Exception as e:
        print(f"❌ FAILED TO LOAD MODELS: {e}")
        return

    # 2. Results Containers
    game_win_rates = []  
    game_partial_counts = []
    total_tries_to_finish = []
    per_game_records = []
    tries_by_category = {0: [], 1: [], 2: [], 3: []}
    
    # 3. Main Loop
    for game_idx, game in enumerate(tqdm(test_games, desc="Simulating Test Games")):
        words = game["words"]
        raw_groups = game["groups"] 
        gt_categories = {}
        for color_idx, (cat_name, g_words) in enumerate(raw_groups.items()):
            gt_categories[color_idx] = set(words.index(w) for w in g_words)

        # 4. Extract feature matrices
        mat_names, mats = build_matrices(embedder, words)
        all_possible_4_subsets = list(itertools.combinations(range(16), 4))
        
        X_test = []
        for S in all_possible_4_subsets:
            X_test.append(extract_features(list(S), mats, set(range(16)), words, mat_names))
        
        X_test = np.array(X_test)
        rf_scores = clf_rf.predict_proba(X_test)[:, 1]
        gbm_scores = clf_gbm.predict_proba(X_test)[:, 1]
        lgbm_scores = clf_lgbm.predict(X_test)
        
        combined_scores = (rf_scores + gbm_scores + lgbm_scores) / 3.0
        score_lookup = {all_possible_4_subsets[i]: combined_scores[i] for i in range(len(all_possible_4_subsets))}

        # --- PHASE 2: Simulate Search ---
        remaining_items = set(range(16))
        remaining_categories = list(gt_categories.keys())
        
        game_total_guesses = 0
        game_errors = 0
        guesses_spent = {0: 0, 1: 0, 2: 0, 3: 0}
        
        while len(remaining_categories) > 1:
            subsets = list(itertools.combinations(sorted(list(remaining_items)), 4))
            subsets = sorted(subsets, key=lambda s: score_lookup.get(s, -1e6), reverse=True)
            
            for guess in subsets:
                game_total_guesses += 1
                guess_set = set(guess)
                
                found_cat = None
                for cat_idx in remaining_categories:
                    if guess_set == gt_categories[cat_idx]:
                        found_cat = cat_idx
                        break
                
                if found_cat is not None:
                    tries_by_category[found_cat].append(guesses_spent[found_cat] + 1)
                    remaining_items -= guess_set
                    remaining_categories.remove(found_cat)
                    break 
                else:
                    game_errors += 1
                    for rc in remaining_categories:
                        guesses_spent[rc] += 1
        
        # --- PHASE 2.5: Per-Game Statistics ---
        per_game_records.append({
            "Game_Index": game_idx,
            "Total_Guesses_Unlimited": game_total_guesses,
            "Errors_Unlimited": game_errors,
            "Win_4_Lives": 1 if game_errors < 4 else 0,
            "Yellow_Tries": guesses_spent[0] + 1,
            "Green_Tries": guesses_spent[1] + 1,
            "Blue_Tries": guesses_spent[2] + 1,
            "Purple_Tries": guesses_spent[3] + 1
        })
        
        # Last group
        last_cat = remaining_categories[0]
        game_total_guesses += 1
        tries_by_category[last_cat].append(guesses_spent[last_cat] + 1)
        total_tries_to_finish.append(game_total_guesses)
        game_win_rates.append(1 if game_errors < 4 else 0)
        
        # --- PHASE 3: Partial Credit ---
        lives = 4
        errs = 0
        correct_count = 0
        rem_items = set(range(16))
        rem_cats = list(gt_categories.keys())
        subsets_pool = sorted(all_possible_4_subsets, key=lambda s: score_lookup.get(s, -1e6), reverse=True)
        for guess in subsets_pool:
            if not set(guess).issubset(rem_items): continue
            match = None
            for c_idx in rem_cats:
                if set(guess) == gt_categories[c_idx]:
                    match = c_idx
                    break
            if match is not None:
                correct_count += 1
                rem_items -= set(guess)
                rem_cats.remove(match)
                if not rem_cats: break
            else:
                errs += 1
                if errs >= lives: break
        if correct_count == 3: correct_count = 4
        game_partial_counts.append(correct_count)
        per_game_records[-1]["Groups_Solved_4_Lives"] = correct_count

    # --- PHASE 4: Reporting ---
    # Save per-game stats
    df_per_game = pd.DataFrame(per_game_records)
    output_csv = "results/per_game_detailed_stats.csv"
    os.makedirs("results", exist_ok=True)
    df_per_game.to_csv(output_csv, index=False)
    print(f"\n✅ PER-GAME STATISTICS SAVED TO: {output_csv}")

    print("\n" + "-"*40)
    print("  🏆 FINAL PERFORMANCE SUMMARY")
    print("-"*40)
    print(f"  Standard 4-Life Accuracy: {np.mean(game_win_rates)*100:.2f}%")
    
    # New Guess Distribution Metric
    print("\n  📦 GUESSES TO COMPLETE GAME (Distribution):")
    guess_counts = pd.Series(total_tries_to_finish).value_counts().sort_index()
    for count, freq in guess_counts.items():
        pct = (freq / len(test_games)) * 100
        print(f"    {int(count):2d} guesses: {freq:2d} games ({pct:.1f}%)")
    
    avg_total_tries = np.mean(total_tries_to_finish)
    print(f"\n  Average Total Guesses: {avg_total_tries:.2f}")

    print("\n  Efficiency by Color (Avg Tries to Find):")
    color_labels = {0: "Yellow (Easy)", 1: "Green (Med)", 2: "Blue (Hard)", 3: "Purple (Tricky)"}
    cat_summary = []
    for c_idx in range(4):
        avg_depth = np.mean(tries_by_category[c_idx])
        print(f"    {color_labels[c_idx]:<15}: {avg_depth:.2f} attempts")
        cat_summary.append({"Category": color_labels[c_idx], "Avg Tries": avg_depth})

    # Plotting
    try:
        plt.figure(figsize=(16, 7))
        # Plot 1: Guess Distribution
        plt.subplot(1, 2, 1)
        sns.histplot(total_tries_to_finish, bins=range(4, int(max(total_tries_to_finish))+2), color='royalblue', kde=True)
        plt.axvline(4, color='orange', linestyle='--', label='Perfect (4)')
        plt.title("Distribution of Guesses to Solve Grid")
        plt.xlabel("Number of Guesses")
        plt.legend()
        
        # Plot 2: Average tries by Category
        plt.subplot(1, 2, 2)
        df_cat = pd.DataFrame(cat_summary)
        sns.barplot(x='Category', y='Avg Tries', data=df_cat, palette='viridis', hue='Category', legend=False)
        plt.title("Search Efficiency by Difficulty Level")
        
        plt.savefig("guess_efficiency_report.png")
        print("\n📈 DETAILED REPORT SAVED AS: guess_efficiency_report.png")
    except Exception as e:
        print(f"\n⚠️ PLOTTING FAILED: {e}")

if __name__ == "__main__":
    from data_loader import load_games
    games = load_games("Connections_Data.csv")
    VAL_END = 777
    test_games = games[VAL_END:]
    
    # For now, keeping your models_31 preference
    evaluate_test_advanced(test_games, ConnectionsEmbedder(), 
                           rf_path="models/clf_rf.pkl", 
                           gbm_path="models/clf_gbm.pkl", 
                           lgbm_path="models/clf_lgbm_ranker.pkl")
