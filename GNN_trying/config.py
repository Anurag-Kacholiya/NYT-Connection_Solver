import torch

# Device configuration
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# File paths
EMBEDDINGS_PATH = '/kaggle/input/datasets/anuragkacholiya/numberbatch-embedding-english/numberbatch-en.txt'
DATA_PATH = "/kaggle/input/datasets/anuragkacholiya/connections-raw-data/Connections_Data.csv"
WIKIDATA_PATH = '/kaggle/input/datasets/anuragkacholiya/wikidata-cache-connections/wikidata_cache.json'
SAVE_PATH = '/kaggle/working/correct_best_rgcn_solver_skipping_newForward.pth'

# Model hyperparameters
EMBEDDING_DIM = 300
HIDDEN_DIM = 64
NUM_RELATIONS = 4