import torch
from models.tgpnet import TGPNet  # Replace with your model class
from utils.graph_utils import parse_tree_data, build_adjacency_matrix, update_influence_weights
import os
import re
import ast

device = "cuda" if torch.cuda.is_available() else "cpu"
model = TGPNet().to(device)
model.load_state_dict(torch.load("checkpoints/tgpnet_test.pth"))
model.eval()

tree_folder = "data/datasetFull"
test_files = [
    "tree_0001.skel",
    "tree_0002.skel",
    "tree_0004.skel",
    "tree_0006.skel"
]

def parse_header(fname):
    with open(os.path.join(tree_folder, fname)) as f:
        first_line = f.readline().strip(',\n ')
        first_line = re.sub(r'(\w+)\s*[:=]', r'"\1":', first_line)
        return ast.literal_eval(first_line)

for fname in test_files:
    print(f"\n🌲 Rebuilding: {fname}")
    with open(os.path.join(tree_folder, fname)) as f:
        lines = f.readlines()
    nodes, edges = parse_tree_data(lines)

    X, A, idx_map = build_adjacency_matrix(nodes, edges)
    X = X.to(device)
    A = A.to(device)
    X[0, 3] = 1.0
    w_cache = {0: torch.tensor([1.0], dtype=torch.float32)}

    node_ids = list(nodes.keys())
    diffs = []

    for i, nid in enumerate(node_ids[1:], start=1):
        current_idx = idx_map[nid]
        gt_z = nodes[nid]['pos'][2]

        X, w_cache = update_influence_weights(X, nodes, idx_map, current_idx, w_cache)
        local_input = X[current_idx][:2]
        with torch.no_grad():
            z_pred = model(local_input, X, A, current_idx).item()
        pred_z = z_pred
        diff_pct = abs(pred_z - gt_z) / abs(gt_z + 1e-8) * 100
        diffs.append(diff_pct)

    avg_diff = sum(diffs) / len(diffs)
    print(f"📊 Average % z-error for {fname}: {avg_diff:.2f}%")
