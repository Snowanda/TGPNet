import os
import torch
from torch.utils.data import Dataset
from utils.graph_utils import parse_tree_data, build_adjacency_matrix, compute_feature_matrix_with_w

class TreeDataset(Dataset):
    def __init__(self, folder_path):
        self.folder_path = folder_path
        self.file_list = sorted([f for f in os.listdir(folder_path) if f.endswith('.skel')])
        self.samples = []
        for fname in self.file_list:
            with open(os.path.join(folder_path, fname)) as f:
                lines = f.readlines()
            nodes, edges = parse_tree_data(lines)
            for node_id, node in nodes.items():
                self.samples.append((fname, node_id))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        fname, node_id = self.samples[idx]
        with open(os.path.join(self.folder_path, fname)) as f:
            lines = f.readlines()
        nodes, edges = parse_tree_data(lines)
        X, A, idx_map = build_adjacency_matrix(nodes, edges)
        current_idx = idx_map[node_id]
        gt_z = nodes[node_id][2]

        # Compute weight w based on current node index
        X = compute_feature_matrix_with_w(X, A, current_idx)

        # Local input is only (x, y)
        local_input = X[current_idx][:2]
        return local_input, X, A, current_idx, torch.tensor(gt_z, dtype=torch.float32)