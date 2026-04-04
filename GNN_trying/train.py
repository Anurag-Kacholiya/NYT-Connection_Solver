import os
import torch
import torch.nn as nn
from config import DEVICE, EMBEDDING_DIM, HIDDEN_DIM, NUM_RELATIONS, SAVE_PATH
from data import load_embeddings, get_train_test_splits
from graph import prepare_pyg_data
from model import RGCNSolver

def train():
    conceptnet_emb = load_embeddings()
    train_puzzles, _ = get_train_test_splits()

    model = RGCNSolver(in_channels=EMBEDDING_DIM, hidden=HIDDEN_DIM, num_relations=NUM_RELATIONS).to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    weight = torch.tensor([4.0]).to(DEVICE)
    criterion = nn.BCEWithLogitsLoss(pos_weight=weight)

    if os.path.exists(SAVE_PATH):
        print(f"Found saved model at {SAVE_PATH}!")
        print("Skipping training and loading weights directly...")
        model.load_state_dict(torch.load(SAVE_PATH, map_location=DEVICE))
        model.eval() 
        return model, conceptnet_emb

    print("No saved model found. Starting training from scratch...")
    best_loss = float('inf')
    
    for epoch in range(50): 
        model.train()
        total_loss = 0
        for puzzle in train_puzzles:
            x, edge_index, edge_type, pair_indices, pair_labels = prepare_pyg_data(puzzle, 3, conceptnet_emb)
            
            optimizer.zero_grad()
            logits = model(x, edge_index, edge_type, pair_indices)
            loss = criterion(logits, pair_labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            
        epoch_loss = total_loss / len(train_puzzles)

        if (epoch + 1) % 5 == 0:
            print(f"Epoch {epoch+1:02d} | Loss: {epoch_loss:.4f}")
            
        if epoch_loss < best_loss:
            best_loss = epoch_loss
            torch.save(model.state_dict(), SAVE_PATH)
            
    print(f"Training complete! Best model saved to: {SAVE_PATH}")
    model.eval()
    return model, conceptnet_emb

if __name__ == "__main__":
    train()