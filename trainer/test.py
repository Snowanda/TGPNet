import os
import re
import ast
import csv
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
        self.curriculum_filenames = []  # will be set later
        self.full_filenames = []        # training filenames after curriculum

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

        # Group files by species and find the one with the smallest node count for each
        species_to_smallest = {}
        other_filenames = []

        for fname in all_filenames:
            species_id, node_count = parse_header(fname)
            if species_id == -1:
                continue
            if species_id not in species_to_smallest or node_count < species_to_smallest[species_id][1]:
                if species_id in species_to_smallest:
                    other_filenames.append(species_to_smallest[species_id][0])  # replace previous
                species_to_smallest[species_id] = (fname, node_count)
            else:
                other_filenames.append(fname)

        # Extract curriculum filenames
        self.curriculum_filenames = [v[0] for v in sorted(species_to_smallest.values(), key=lambda x: x[1])]

        # ✅ Include them in full training too
        full_list = self.curriculum_filenames + other_filenames
        self.full_filenames = sorted(full_list, key=lambda f: parse_header(f)[1])


    def train(self, epochs=10, curriculum_epochs=10):
        for epoch in range(epochs):
            if epoch < curriculum_epochs:
                filenames = self.curriculum_filenames
                print(f"\n📘 Curriculum Phase (Epoch {epoch+1}) - Using {len(filenames)} samples")
            else:
                filenames = self.full_filenames
                print(f"\n🟢 Main Training Phase (Epoch {epoch+1}) - Using {len(filenames)} samples")
            total_loss = 0
            total_nodes = 0 
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
                    total_nodes += 1

                    progress = (i + 1) / len(nodes) * 100
                    print(f"\r⏳ Epoch {epoch+1} | Sample {f_idx + 1}/{len(filenames)} | Progress: {progress:.1f}%", end="")

                    debug_file = "debug_logs.csv"
                    with open(debug_file, mode="a", newline="") as f:
                        writer = csv.writer(f)
                        abs_error = abs(z_pred.item() - gt_z)
                        pct_error = abs_error / (abs(gt_z) + 1e-8) * 100
                        writer.writerow([
                            epoch + 1,
                            fname,
                            nid,
                            f"{gt_z:.6f}",
                            f"{z_pred.item():.6f}",
                            f"{abs_error:.6f}",
                            f"{pct_error:.2f}"
                        ])

            self.scheduler.step()
            avg_loss = total_loss / total_nodes if total_nodes > 0 else 0
            print(f"\n✅ Epoch {epoch+1} finished | Avg Loss per Node: {avg_loss:.6f} | Total Loss: {total_loss:.6f}")


        torch.save(self.model.state_dict(), "checkpoints/tgpnet_debug.pth")
        print("💾 Model saved to checkpoints/tgpnet_final.pth")
