import os
import re
import ast
import csv
import torch
import torch.nn as nn
from utils.graph_utils import parse_tree_data, build_adjacency_matrix, update_influence_weights

class TreeTrainer:
    def __init__(self, model, tree_folder, log_path="debug_logs.csv", device="cuda"):
        device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = model.to(device)
        self.tree_folder = tree_folder
        self.device = device
        self.loss_fn = nn.L1Loss()
        self.optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        self.scheduler = torch.optim.lr_scheduler.ExponentialLR(self.optimizer, gamma=0.8)
        self.curriculum_filenames = []
        self.full_filenames = []
        self.log_path = log_path

        # Prepare log file
        with open(self.log_path, 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(["Epoch", "Filename", "NodeID", "GT_z", "Pred_z", "AbsError", "PercentError"])

    def prepare_curriculum(self, all_filenames):
        def parse_header(fname):
            with open(os.path.join(self.tree_folder, fname)) as f:
                first_line = f.readline().strip(',\n ')
                first_line = re.sub(r'(\w+)\s*[:=]', r'"\1":', first_line)
                try:
                    entry = ast.literal_eval(first_line)
                    return int(entry.get("species_id", -1)), int(entry.get("treenode_num", float("inf")))
                except:
                    return -1, float("inf")

        species_to_smallest = {}
        for fname in all_filenames:
            species_id, node_count = parse_header(fname)
            if species_id == -1:
                continue
            if species_id not in species_to_smallest or node_count < species_to_smallest[species_id][1]:
                species_to_smallest[species_id] = (fname, node_count)

        self.curriculum_filenames = [v[0] for v in sorted(species_to_smallest.values(), key=lambda x: x[1])]
        self.full_filenames = sorted(all_filenames, key=lambda f: parse_header(f)[1])

    def train(self, epochs=10, curriculum_epochs=10):
        for epoch in range(epochs):
            filenames = self.curriculum_filenames if epoch < curriculum_epochs else self.full_filenames
            print(f"\n📘 Epoch {epoch + 1}/{epochs} | Samples: {len(filenames)}")
            total_loss, total_nodes = 0, 0

            for f_idx, fname in enumerate(filenames):
                print(fname)
                with open(os.path.join(self.tree_folder, fname)) as f:
                    lines = f.readlines()
                nodes, edges = parse_tree_data(lines)
                X, A, idx_map = build_adjacency_matrix(nodes, edges)
                X = X.to(self.device)
                A = A.to(self.device)
                X[0, 3] = 1.0
                w_cache = {0: torch.tensor([1.0], dtype=torch.float32)}
                
                node_ids = list(nodes.keys())[1:]
                self.optimizer.zero_grad()
                batch_loss = 0
                with open(self.log_path, 'a', newline='') as csvfile:
                    writer = csv.writer(csvfile)

                    for i, nid in enumerate(node_ids):
                        current_idx = idx_map[nid]
                        gt_z = nodes[nid]['pos'][2]
                        X, w_cache = update_influence_weights(X, nodes, idx_map, current_idx, w_cache)

                        local_input = X[current_idx][:2]
                        gt_z_tensor = torch.tensor([gt_z], dtype=torch.float32, device=self.device)

                        z_pred = self.model(local_input, X, A, current_idx)
                        loss = self.loss_fn(z_pred, gt_z_tensor)
                        loss.backward(retain_graph=True)

                        abs_err = abs(z_pred.item() - gt_z)
                        percent_err = abs_err / max(abs(gt_z), 1e-6) * 100

                        writer.writerow([epoch + 1, fname, nid, gt_z, z_pred.item(), abs_err, percent_err])
                        batch_loss += loss.item()

                        print(f"\r⏳ Epoch {epoch + 1} | Sample {f_idx + 1}/{len(filenames)} | Node {nid} | z GT: {gt_z:.4f} | z Pred: {z_pred.item():.4f} | %Err: {percent_err:.2f}%", end="")

                self.optimizer.step()
                total_loss += batch_loss
                total_nodes += len(node_ids)

            self.scheduler.step()
            print(f"\n✅ Epoch {epoch + 1} done | Avg Loss: {total_loss / total_nodes:.6f}")

        torch.save(self.model.state_dict(), "checkpoints/tgpnet_debug.pth")
        print("💾 Debug model saved to checkpoints/tgpnet_debug.pth")