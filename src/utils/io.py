# src/utils/io.py
import os, json
import yaml

def save_yaml(d, path):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w") as f:
        yaml.safe_dump(d, f)

def load_yaml(path):
    with open(path, "r") as f:
        return yaml.safe_load(f)

def save_json(d, path):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w") as f:
        json.dump(d, f, indent=2)
