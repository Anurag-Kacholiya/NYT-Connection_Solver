"""
evaluate_test.py — Load saved models and evaluate on the held-out TEST split.
                   Supports the new LightGBM ranker (Tier 1.2) if present.

Test split: games 777+ (games 0-639 = train, 640-776 = val)
"""
import os, sys, pickle
from tqdm import tqdm
from data_loader import load_games
from embedder import ConnectionsEmbedder
from evaluate import build_matrices, simulate_game_ml


def evaluate_split(games_list, embedder, clf_rf, clf_gbm,
                   split_name="TEST", clf_ranker=None):
    total_games  = len(games_list)
    perfect_games = 0
    total_groups  = 0
    total_guesses = 0
    partial_match_counts = {4: 0, 3: 0, 2: 0, 1: 0, 0: 0}

    print(f"\nEvaluating [{split_name}]:")
    for game in tqdm(games_list):
        words = game["words"]
        gt_partitions = [
            {words.index(w) for w in gw}
            for gw in game["groups"].values()
        ]

        mat_names, mats = build_matrices(
            embedder, words,
            use_numberbatch=True,
            use_multisense=True,
            use_pairwise_context=True,
        )
        matched, preds, pmatch = simulate_game_ml(
            words, mats, mat_names,
            (clf_rf, clf_gbm),
            gt_partitions,
            clf_ranker=clf_ranker,
            use_candidates=True,
        )

        total_groups  += matched
        total_guesses += len(preds)
        if matched == 4:
            perfect_games += 1
        for k, v in pmatch.items():
            partial_match_counts[k] += v

    game_acc  = perfect_games / total_games
    group_acc = total_groups  / (total_games * 4)
    print(f"\n  [{split_name}]  {total_games} games")
    print(f"    Perfect games  : {perfect_games}  ({game_acc:.2%})")
    print(f"    Group accuracy :  ({group_acc:.2%})")
    print(f"    Guess breakdown:")
    for k in [4, 3, 2, 1, 0]:
        pct = partial_match_counts[k] / total_guesses * 100 if total_guesses else 0
        print(f"      {k}/4 → {partial_match_counts[k]:4d}  ({pct:.1f}%)")
    return game_acc, group_acc


if __name__ == "__main__":
    print("Loading data...")
    all_games  = load_games("Connections_Data.csv")
    test_games = all_games[777:]  # 15% test split

    print("Loading embedder...")
    embedder = ConnectionsEmbedder()

    print("Loading models...")
    with open("models/clf_rf.pkl",  "rb") as f:
        clf_rf  = pickle.load(f)
    with open("models/clf_gbm.pkl", "rb") as f:
        clf_gbm = pickle.load(f)

    # Load LightGBM ranker if available
    clf_ranker = None
    ranker_path = "models/clf_lgbm_ranker.pkl"
    if os.path.exists(ranker_path):
        with open(ranker_path, "rb") as f:
            clf_ranker = pickle.load(f)
        print("LightGBM ranker loaded.")
    else:
        print("No ranker found — using RF+GBM only.")

    evaluate_split(test_games, embedder, clf_rf, clf_gbm,
                   split_name="TEST", clf_ranker=clf_ranker)
