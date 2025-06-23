import torch.nn as nn
import torch

# Local Feature Encoder
class LocalEncoder(nn.Module):
    def __init__(self):
        super(LocalEncoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(2, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 16),
            nn.ReLU()
        )

    def forward(self, xy):
        return self.encoder(xy)

# GCN Layer
class GCNLayer(nn.Module):
    def __init__(self, in_dim, out_dim):
        super(GCNLayer, self).__init__()
        self.linear = nn.Linear(in_dim, out_dim)
        self.act = nn.LeakyReLU()

    def forward(self, X, A):
        AXW = torch.matmul(A, self.linear(X))
        return self.act(AXW)

# Global Graph Encoder
class GlobalGraphEncoder(nn.Module):
    def __init__(self):
        super(GlobalGraphEncoder, self).__init__()
        self.gcn0 = GCNLayer(4, 16)
        self.gcn_layers = nn.ModuleList([
            GCNLayer(16, 16),
            GCNLayer(16, 16),
            GCNLayer(16, 16),
            GCNLayer(16, 16)
        ])
        self.final_proj = nn.Linear(16, 128)  # pool across nodes

    def forward(self, X, A):
        X = self.gcn0(X, A)
        for layer in self.gcn_layers:
            X = layer(X, A)
        X_pool = X.mean(dim=0)
        return self.final_proj(X_pool)