import os
import re
import ast
import random
from collections import defaultdict
import shutil

# Define paths
dataset_dir = "data/datasetFull"
output_dir = "data/test"
os.makedirs(output_dir, exist_ok=True)


def fix_line_to_valid_dict(line):
    line = line.strip(',\n ')
    # Normalize both ":" and "=" to JSON-compatible colon format
    line = re.sub(r'(\w+)\s*[:=]', r'"\1":', line)
    return ast.literal_eval(line)

# Helper to extract species_id and file name
def parse_header_info(file_path):
    with open(file_path, 'r') as f:
        first_line = f.readline().strip(',\n ')
      # quote keys
    try:
        entry = fix_line_to_valid_dict(first_line)
        print(entry.get("species_id"))
        return entry.get("species_id"), entry.get("name")
    except:
        return None, None

# Group files by species
species_map = defaultdict(list)
for fname in os.listdir(dataset_dir):
    if fname.endswith(".skel"):
        full_path = os.path.join(dataset_dir, fname)
        species_id, name = parse_header_info(full_path)
        if species_id is not None:
            species_map[species_id].append(fname)

# Sample 10 files per species and copy to output folder
for species_id, file_list in species_map.items():
    sampled = random.sample(file_list, min(10, len(file_list)))
    for fname in sampled:
        src = os.path.join(dataset_dir, fname)
        dst = os.path.join(output_dir, fname)
        shutil.move(src, dst)

species_summary = {sid: len(files) for sid, files in species_map.items()}
species_summary
