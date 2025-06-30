import torch
from models.tgpnet import TGPNet
from utils.graph_utils import parse_tree_data, build_adjacency_matrix, update_influence_weights
import os
import re
import ast

device = "cuda" if torch.cuda.is_available() else "cpu"

# Instantiate and warm-up the model to initialize dynamic submodules
model = TGPNet().to(device)

# 🔧 Force global_encoder creation
with torch.no_grad():
    dummy_X = torch.randn(10, 4).to(device)
    dummy_X[0, 3] = 1.0  # root node w
    dummy_A = torch.eye(10).to(device)
    dummy_local = torch.randn(2).to(device)
    _ = model(dummy_local, dummy_X, dummy_A, current_idx=1)

# ✅ Now load weights
model.load_state_dict(torch.load("checkpoints/tgpnet_test.pth"))
model.eval()


tree_folder = "data/datasetFull"
test_files = [
    "tree_0643.skel"
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

        # 絕對百分比誤差
        if abs(gt_z) < 1e-6:
            diff_pct = float('inf')
        else:
            diff_pct = abs(pred_z - gt_z) / abs(gt_z) * 100
        diffs.append(diff_pct)

        # 印出前 10 筆
        if i <= 10:
            print(f"  🧪 Node {nid}: GT z = {gt_z:.6f} | Pred z = {pred_z:.6f} | % Error = {diff_pct:.2f}%")

    avg_diff = sum(d for d in diffs if d != float('inf')) / max(len([d for d in diffs if d != float('inf')]), 1)
    print(f"📊 Average % z-error for {fname}: {avg_diff:.2f}%")
