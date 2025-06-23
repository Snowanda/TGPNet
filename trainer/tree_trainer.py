import os
import torch
import torch.nn as nn
from utils.graph_utils import parse_tree_data, build_adjacency_matrix, update_influence_weights

class TreeTrainer:
    def __init__(self, model, tree_folder, device="cuda"):
        device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = model.to(device)
        self.tree_folder = tree_folder
        self.device = device
        self.loss_fn = nn.L1Loss()
        self.optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        self.scheduler = torch.optim.lr_scheduler.ExponentialLR(self.optimizer, gamma=0.8)

    def train(self, filenames, epochs=10):
        for epoch in range(epochs):
            total_loss = 0
            print(f"\n🟢 Epoch {epoch+1}/{epochs} started")

            for f_idx, fname in enumerate(filenames):
                # print(f"\n📄 Processing file {f_idx+1}/{len(filenames)}: {fname}")
                
                with open(os.path.join(self.tree_folder, fname)) as f:
                    lines = f.readlines()
                nodes, edges = parse_tree_data(lines)
                # print(f"🔍 Parsed {len(nodes)} nodes, {len(edges)} edges")

                X, A, idx_map = build_adjacency_matrix(nodes, edges)
                A = A.to(self.device)
                X = X.to(self.device)
                X[0, 3] = 1.0  # Set root node weight w to 1.0
                w_cache = {0 : torch.tensor([1.0], dtype=torch.float32)}  # Initialize weight cache with root node

                node_ids = list(nodes.keys())
                for i, nid in enumerate(node_ids[1:], start=1):
                    current_idx = idx_map[nid]
                    gt_z = nodes[nid]['pos'][2]
                    # print(f"🔍 Processing node {nid} at index {current_idx} with ground truth z: {gt_z:.5f}")

                    X, w_cache = update_influence_weights(X, nodes, idx_map, current_idx, w_cache)
                    # print(X[:current_idx+1])

                    local_input = X[current_idx][:2].to(self.device)
                    gt_z_tensor = torch.tensor([gt_z], dtype=torch.float32, device=self.device)

                    self.optimizer.zero_grad()
                    z_pred = self.model(local_input, X, A, current_idx)
                    loss = self.loss_fn(z_pred, gt_z_tensor)
                    loss.backward()
                    self.optimizer.step()

                    X[current_idx][2] = z_pred.detach()
                    total_loss += loss.item()

                    progress = (i + 1) / len(nodes) * 100
                    print(f"\r⏳ Epoch {epoch+1} | Progress: {progress:.1f}%", end="")

                    #if (i + 1) % 10 == 0 or i == len(nodes.keys()) - 1:
                    #   print(f"    🧠 Step {i+1}/{len(nodes.keys())} | Node {nid} | Predicted z: {z_pred.item():.5f} | GT z: {gt_z:.5f} | Loss: {loss.item():.5f}")

            self.scheduler.step()
            print(f"\n✅ Epoch {epoch+1} finished | Total Loss: {total_loss:.4f}")

        torch.save(self.model.state_dict(), "checkpoints/tgpnet_final.pth")
        print("💾 Model saved to checkpoints/tgpnet_final.pth")
