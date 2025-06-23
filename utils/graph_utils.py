import math
import numpy as np
import torch
import ast
import re
import torch
from collections import deque

def fix_line_to_valid_dict(line):
    line = line.strip(',\n')
    # Add quotes around keys using regex
    line = re.sub(r'(\w+):', r'"\1":', line)
    return ast.literal_eval(line)

# --- Utility functions ---

def parse_tree_data(tree_lines):
    """Parses a list of lines from a tree file and returns nodes and edges."""
    nodes = {}
    edges = []
    for line in tree_lines:
        if '"treenode"' in line or "'treenode'" in line:
            entry = fix_line_to_valid_dict(line)  # safe here assuming input is controlled
            # print(f"Parsed entry: {entry}")  # Debugging line
            if entry['node_id'] > 1200:
                break
            node_id = entry['node_id']
            x, y, z = entry['pos']
            nodes[node_id] = {'pos': (x, y, z), 'parent': entry['parent_id'], 'children': entry['children_ids']}
            # for child in entry['children_ids']:
            #    edges.append((node_id, child))
            edges.append((entry['parent_id'], node_id))
        elif '"leaf"' in line or "'leaf'" in line:
            break
        # pop the first edge (parent id is -1)
        
    edges.pop(0)
    return nodes, edges

def build_adjacency_matrix(nodes, edges):
    """Creates an NxN adjacency matrix and node feature matrix (x, y, z, w)."""
    N = len(nodes)
    idx_map = {nid: i for i, nid in enumerate(nodes)}
    A = np.zeros((N, N))
    X = np.zeros((N, 4))  # x, y, z, w (w will be computed later)

    for (src, dst) in edges:
        i, j = idx_map[src], idx_map[dst]
        xi, yi, zi = nodes[src]['pos']
        xj, yj, zj = nodes[dst]['pos']
        dist = math.sqrt((xi - xj) ** 2 + (yi - yj) ** 2 + (zi - zj) ** 2)
        A[i][j] = A[j][i] = dist

    for nid, i in idx_map.items():
        x, y, z = nodes[nid]['pos']
        X[i][:3] = [x, y, 0]  # set z=0 initially

    return torch.tensor(X, dtype=torch.float32), torch.tensor(A, dtype=torch.float32), idx_map

def update_influence_weights(X, nodes, idx_map, current_idx, w_cache):
    nid = list(idx_map.keys())[list(idx_map.values()).index(current_idx)]
    parent_id = nodes[nid]["parent"]
    parent_idx = idx_map[parent_id]

    # Update weights for previous nodes
    prev_w = w_cache[parent_id]
    updated_w = 1 / (1 / prev_w + 1)
    X[:parent_idx+1, 3] = updated_w
    X[current_idx, 3] = 1.0  # Set current node's w to 1.0

    # Adjust same-depth siblings
    same_depth_i = current_idx - 1
    while same_depth_i >= 0 and w_cache.get(same_depth_i, torch.tensor([-1.0]))[0] == updated_w[0]:
        distance = 2
        find_parent_i = nodes[same_depth_i]["parent"]
        find_parent_current = parent_id
        while find_parent_i != find_parent_current:
            distance += 2
            find_parent_i = nodes[find_parent_i]["parent"]
            find_parent_current = nodes[find_parent_current]["parent"]
        X[same_depth_i, 3] = 1 / (distance + 1)
        same_depth_i -= 1

    # Update weight cache
    w_cache[nid] = X[:current_idx+1, 3].clone()

    # If this is the last child, remove parent's weight
    if nid == nodes[parent_id]['children'][-1]:
        w_cache.pop(parent_id, None)

    return X, w_cache

'''
def bfs_shortest_paths(adj_matrix, start_idx):
    """Computes shortest path lengths from start_idx to all other nodes using BFS (tree structure assumed)."""
    N = adj_matrix.size(0)
    visited = [False] * N
    distance = [float('inf')] * N
    queue = deque([start_idx])
    visited[start_idx] = True
    distance[start_idx] = 0

    while queue:
        u = queue.popleft()
        for v in range(N):
            if adj_matrix[u][v] != 0 and not visited[v]:
                visited[v] = True
                distance[v] = distance[u] + 1
                queue.append(v)

    return torch.tensor(distance, dtype=torch.float32)

# Influence weight computation
def compute_feature_matrix_with_w(X, A, current_idx):
    hop_adj = (A != 0)[:current_idx+1, :current_idx+1]  # Convert to unweighted
    D_row = bfs_shortest_paths(hop_adj, current_idx)
    w = 1 / (1 + D_row)
    X[:current_idx+1, 3] = w
    return X
'''
