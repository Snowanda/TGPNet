import os
import re
import ast
import pandas as pd

# Utility functions
def fix_line_to_valid_dict(line):
    line = line.strip(',\n')
    line = re.sub(r'(\w+):', r'"\1":', line)
    return ast.literal_eval(line)

def parse_tree_data(tree_lines):
    nodes = {}
    for line in tree_lines:
        if '"treenode"' in line or "'treenode'" in line:
            entry = fix_line_to_valid_dict(line)
            if entry['node_id'] > 1200:
                break
            node_id = entry['node_id']
            x, y, z = entry['pos']
            nodes[node_id] = (x, y, z)
        elif '"leaf"' in line or "'leaf'" in line:
            break
    return nodes

# Scan dataset directory for .skel files
tree_folder = "data/datasetFull"
all_values = []
y_negative_nodes = []

for fname in sorted(os.listdir(tree_folder)):
    if not fname.endswith(".skel"):
        continue
    with open(os.path.join(tree_folder, fname)) as f:
        lines = f.readlines()
    nodes = parse_tree_data(lines)
    all_values.extend(nodes.values())
    for node_id, (x, y, z) in nodes.items():
        if y < 0:
            y_negative_nodes.append((fname, node_id, y))

# Extract boundaries
df = pd.DataFrame(all_values, columns=["x", "y", "z"])
bounds = {
    "x": (df["x"].min(), df["x"].max()),
    "y": (df["y"].min(), df["y"].max()),
    "z": (df["z"].min(), df["z"].max())
}
bounds_df = pd.DataFrame.from_dict(bounds, orient="index", columns=["min", "max"])

# Output results
print("🟨 Coordinate boundaries:")
print(bounds_df)

print("\n📌 Nodes with y < 0:")
for fname, node_id, y in y_negative_nodes:
    print(f"File: {fname}, Node ID: {node_id}, y = {y:.6f}")
