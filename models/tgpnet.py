import torch.nn as nn
from .encoders import LocalEncoder, GlobalGraphEncoder
from .inference import DepthInferenceModule

# Model wrapper
class TGPNet(nn.Module):
    def __init__(self):
        super(TGPNet, self).__init__()
        self.local_encoder = LocalEncoder()
        self.global_encoder = None  # Will initialize dynamically with node count
        self.depth_infer = DepthInferenceModule()

    def forward(self, local_input, X, A, current_idx):
        if self.global_encoder is None or self.global_encoder.final_proj.in_features != X.shape[1]:
            self.global_encoder = GlobalGraphEncoder().to(X.device)

        local_feat = self.local_encoder(local_input.unsqueeze(0))
        global_feat = self.global_encoder(X, A).unsqueeze(0)
        return self.depth_infer(local_feat, global_feat)