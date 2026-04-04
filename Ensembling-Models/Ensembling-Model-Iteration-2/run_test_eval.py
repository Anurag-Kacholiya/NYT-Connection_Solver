import urllib.request
import os, sys, pickle
from tqdm import tqdm
from data_loader import load_games
from embedder import ConnectionsEmbedder
from train import evaluate_split
from evaluate import simulate_game_ml

if __name__ == "__main__":
    print("Loading data...")
    all_games = load_games("Connections_Data.csv")
    train_games = all_games[:640]
    val_games   = all_games[640:776]
    test_games  = all_games[777:914] 
    # Ah! In train.py it was test_games = all_games[777:915], all_games[777:] wait.

    print("Loading embedder...")
    embedder = ConnectionsEmbedder()
    
    print("Loading models...")
    with open("models/clf_rf.pkl", "rb") as f:
        clf_rf = pickle.load(f)
    with open("models/clf_gbm.pkl", "rb") as f:
        clf_gbm = pickle.load(f)

    evaluate_split(test_games, embedder, clf_rf, clf_gbm, "TEST")
