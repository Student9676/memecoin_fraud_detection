import os
import json

def count_json_files_with_values(base_dir, from_value, to_value):
    count = 0

    for root, _, files in os.walk(base_dir):
        for file in files:
            if file.endswith(".json"):
                file_path = os.path.join(root, file)
                try:
                    with open(file_path, 'r') as f:
                        data = json.load(f)
                        if data.get('from') == from_value and data.get('to') == to_value:
                            count += 1
                except (json.JSONDecodeError, KeyError):
                    print(f"Skipping invalid or unreadable JSON file: {file_path}")

    return count

for file in os.listdir("data/dev_nodes"):
    if file.endswith(".json"):
        with open(os.path.join("data/dev_nodes", file), "r") as f:
            data = json.load(f)
            dev = data.get("dev_address")
            
            base_directory = "data/wallet_wallet_edges"
            from_value = dev
            to_value = from_value

            result = count_json_files_with_values(base_directory, from_value, to_value)
            print(f"Number of JSON files with from and to='{from_value}': {result}")