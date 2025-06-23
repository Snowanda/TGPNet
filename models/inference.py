import torch.nn as nn
import torch

# Residual Block
class ResidualBlock(nn.Module):
    def __init__(self, dim):
        super(ResidualBlock, self).__init__()
        self.linear = nn.Linear(dim, dim)
        self.act = nn.LeakyReLU()

    def forward(self, x):
        return x + self.act(self.linear(x))

# Depth Inference Module
class DepthInferenceModule(nn.Module):
    def __init__(self):
        super(DepthInferenceModule, self).__init__()
        self.input_proj = nn.Linear(144, 128)
        self.res_layers = nn.Sequential(*[ResidualBlock(128) for _ in range(6)])
        self.mlp = nn.Sequential(
            nn.Linear(128, 64),
            nn.LeakyReLU(),
            nn.Linear(64, 16),
            nn.LeakyReLU(),
            nn.Linear(16, 1)
        )

    def forward(self, local_feat, global_feat):
        x = torch.cat([local_feat, global_feat], dim=-1)
        x = self.input_proj(x)
        x = self.res_layers(x)
        return self.mlp(x).squeeze(-1)
    