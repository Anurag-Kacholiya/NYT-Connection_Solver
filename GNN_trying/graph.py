import json
import torch
import networkx as nx
import itertools
import numpy as np
import nltk
from nltk.corpus import wordnet as wn
from config import WIKIDATA_PATH, DEVICE
from data import get_embedding

# Ensure nltk resources are downloaded
try:
    wn.synsets('dog')
except LookupError:
    nltk.download('wordnet')
    nltk.download('omw-1.4')

def load_wikidata_cache():
    try:
        with open(WIKIDATA_PATH, 'r') as f:
            wikidata_cache = json.load(f)
        print("Successfully loaded local Wikidata cache.")
    except FileNotFoundError:
        print("Warning: wikidata_cache.json not found. Tier 2 entity links will be empty.")
        wikidata_cache = {}
    return wikidata_cache

WIKIDATA_CACHE = load_wikidata_cache()

def build_tiered_graph(words, tier):
    """Builds a semantic graph using ONLY local data."""
    G = nx.Graph()
    for i, w in enumerate(words):
        G.add_node(w, type='word', target_idx=i)

    # TIER 1: (WordNet)
    if tier >= 1:
        for w1, w2 in itertools.combinations(words, 2):
            syn1 = set(s for s in wn.synsets(w1))
            syn2 = set(s for s in wn.synsets(w2))
            if syn1.intersection(syn2):
                G.add_edge(w1, w2, relation=0)

    # TIER 2: (Offline Wikidata Cache only)
    if tier >= 2:
        for w in words:
            related_entities = WIKIDATA_CACHE.get(w, [])
            for r in related_entities:
                if r in words and r != w:
                    G.add_edge(w, r, relation=1)

    # TIER 3: Pattern-Based Augmentation (Wordplay)
    if tier >= 3:
        for w1, w2 in itertools.combinations(words, 2):
            if w1[:3] == w2[:3] and len(w1) >= 3: # Prefix match
                G.add_edge(w1, w2, relation=2)
            if len(w1) == len(w2): # Same length match
                G.add_edge(w1, w2, relation=3)

    return G

def prepare_pyg_data(puzzle, tier, conceptnet_emb):
    """Converts NetworkX graph into PyTorch Geometric Data format."""
    words = puzzle["words"]
    true_labels = puzzle["labels"]
    
    G = build_tiered_graph(words, tier)
    node_list = list(G.nodes())
    node_map = {n: i for i, n in enumerate(node_list)}
    
    edges, edge_types = [], []
    for u, v, data in G.edges(data=True):
        edges.append([node_map[u], node_map[v]])
        edges.append([node_map[v], node_map[u]]) # undirected
        edge_types.extend([data["relation"], data["relation"]])
        
    edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous() if edges else torch.empty((2,0), dtype=torch.long)
    edge_type = torch.tensor(edge_types, dtype=torch.long) if edge_types else torch.empty((0,), dtype=torch.long)
    
    x_list = [get_embedding(w, conceptnet_emb) for w in node_list]
    x = torch.tensor(np.array(x_list), dtype=torch.float)
    
    pairs = list(itertools.combinations(range(16), 2))
    pair_indices = torch.tensor(pairs, dtype=torch.long).t()
    
    pair_labels = []
    for i, j in pairs:
        pair_labels.append(1.0 if true_labels[i] == true_labels[j] else 0.0)
    pair_labels = torch.tensor(pair_labels, dtype=torch.float)
    
    return x.to(DEVICE), edge_index.to(DEVICE), edge_type.to(DEVICE), pair_indices.to(DEVICE), pair_labels.to(DEVICE)