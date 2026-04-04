import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import RGCNConv

class RGCNSolver(nn.Module):
    def __init__(self, in_channels, hidden, num_relations):
        super().__init__()
        self.conv1 = RGCNConv(in_channels, hidden, num_relations)
        self.conv2 = RGCNConv(hidden, hidden, num_relations)
        
        self.scorer = nn.Sequential(
            nn.Linear((in_channels + hidden) * 2, hidden),
            nn.LeakyReLU(0.1), 
            nn.Dropout(0.2),   
            nn.Linear(hidden, 1)
        )

    def forward(self, x, edge_index, edge_type, pair_indices):
        x_orig = x 
        
        x_graph = F.relu(self.conv1(x, edge_index, edge_type))
        x_graph = self.conv2(x_graph, edge_index, edge_type)
        x_combined = torch.cat([x_orig, x_graph], dim=1)
        
        emb_i = x_combined[pair_indices[0]]
        emb_j = x_combined[pair_indices[1]]
        
        diff = torch.abs(emb_i - emb_j)
        mult = emb_i * emb_j
        
        pair_features = torch.cat([diff, mult], dim=1)
        logits = self.scorer(pair_features).squeeze(-1)
        return logits