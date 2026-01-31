path = "VisFly/datasets/visfly-beta/configs/scenes/box30_mix2"

# search all the json files in the path and its subdirectories
# for example "VisFly/visfly-beta/configs/scenes/box30_mix2/box30_high_0.07tree/box30_high_0.07tree_1.json"
all_file_paths = []

# replace attributes "uniform_scale" with "non_uniform_scale: [randx, randy, 1]", randx = randy are random float numbers between 0.2 and 0.8.
import os
import json
import random
for root, dirs, files in os.walk(path):
    for file in files:
        if file.endswith(".json"):
            all_file_paths.append(os.path.join(root, file))

print("Found {} files".format(len(all_file_paths)))
for file_path in all_file_paths:
    with open(file_path, "r") as f:
        data = json.load(f)
    if "object_instances" in data:
        modified = False
        parent_folder = os.path.basename(os.path.dirname(file_path))
        is_tree = "tree" in parent_folder

        for obj in data["object_instances"]:
            if "uniform_scale" in obj or "non_uniform_scale" in obj:
                randx = round(random.uniform(0.5, 0.8), 2)
                if is_tree:
                    scale = [randx, randx, 1]
                else:
                    scale = [randx, randx, randx]
                
                obj["non_uniform_scale"] = scale
                if "uniform_scale" in obj:
                    del obj["uniform_scale"]
                modified = True
        
        if modified:
            with open(file_path, "w") as f:
                json.dump(data, f, indent=4)