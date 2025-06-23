import os
import torch
from utils.graph_utils import parse_tree_data, build_adjacency_matrix, compute_feature_matrix_with_w
from models.tgpnet import TGPNet
from trainer.tree_trainer import TreeTrainer

def save_predicted_skel(fname, nodes, X, idx_map, out_path):
    with open(os.path.join(out_path, fname), 'w') as f:
        for nid, i in idx_map.items():
            x, y = X[i][0].item(), X[i][1].item()
            z_pred = X[i][2].item()
            node = nodes[nid]
            children = node['children']
            f.write(f'{{"type": "treenode", "node_id": {nid}, "pos": [{x:.6f}, {y:.6f}, {z_pred:.6f}], '
                    f'"parent_id": {node["parent"]}, "children_ids": {children}, '
                    f'"radius": {node["radius"]}, "branch_id": {node["branch_id"]} }},\n')

def predict_single_file(model_path, skel_file, output_path):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = TGPNet().to(device)
    model.load_state_dict(torch.load(model_path))
    model.eval()

    with open(skel_file) as f:
        lines = f.readlines()
    nodes, edges = parse_tree_data(lines)
    X, A, idx_map = build_adjacency_matrix(nodes, edges)
    A = A.to(device)
    X = X.to(device)

    trainer = TreeTrainer(model, os.path.dirname(skel_file), device)
    bfs_order = trainer.get_bfs_order(nodes)

    for nid in bfs_order:
        current_idx = idx_map[nid]
        X = compute_feature_matrix_with_w(X, A, current_idx)
        local_input = X[current_idx][:2].to(device)
        with torch.no_grad():
            z_pred = model(local_input, X, A, current_idx)
            X[current_idx][2] = z_pred

    os.makedirs(output_path, exist_ok=True)
    save_predicted_skel(os.path.basename(skel_file), nodes, X, idx_map, output_path)