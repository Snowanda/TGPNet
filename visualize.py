from models.tgpnet import TGPNet
import torch
model = TGPNet()

state_dict = torch.load("checkpoints/tgpnet_test.pth")
print(state_dict.keys())
print(model.state_dict().keys())
