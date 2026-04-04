import pandas as pd
import numpy as np
from gensim.models import KeyedVectors
from sklearn.model_selection import train_test_split
from config import EMBEDDINGS_PATH, DATA_PATH, EMBEDDING_DIM

def load_embeddings():
    print("Loading ConceptNet Numberbatch 19.08 from local dataset...")
    conceptnet_emb = KeyedVectors.load_word2vec_format(EMBEDDINGS_PATH, binary=False)
    print("Embeddings loaded successfully!")
    return conceptnet_emb

def get_embedding(word, conceptnet_emb):
    """Fetches the ConceptNet embedding."""
    formatted_word = f"/c/en/{word.strip().replace(' ', '_')}"
    if formatted_word in conceptnet_emb:
        return conceptnet_emb[formatted_word]
    return np.random.normal(scale=0.5, size=(EMBEDDING_DIM,))

def load_puzzles():
    df = pd.read_csv(DATA_PATH)
    df = df.dropna(subset=["Word"])
    df["Word"] = df["Word"].astype(str).str.lower()

    puzzles = []
    for game_id, group in df.groupby("Game ID"):
        words = group["Word"].tolist()
        if len(words) != 16:
            continue
        
        unique_groups = group["Group Name"].unique()
        group_map = {name: idx for idx, name in enumerate(unique_groups)}
        labels = [group_map[name] for name in group["Group Name"]]
        
        puzzles.append({"words": words, "labels": labels})

    print(f"Loaded {len(puzzles)} clean puzzles.")
    return puzzles

def get_train_test_splits():
    puzzles = load_puzzles()
    train_puzzles, test_puzzles = train_test_split(
        puzzles, test_size=100, random_state=42
    )
    print(f"Training set: {len(train_puzzles)} puzzles")
    print(f"Testing set: {len(test_puzzles)} puzzles")
    return train_puzzles, test_puzzles