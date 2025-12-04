
import pandas as pd
import os
import yaml

def load_config(path="config.yaml"):
    with open(path) as f:
        return yaml.safe_load(f)

def load_sample_csv(path):
    df = pd.read_csv(path)
    return df

def ingest(config_path="config.yaml"):
    cfg = load_config(config_path)
    path = cfg["dataset"]["sample_csv"]
    if not os.path.exists(path):
        raise FileNotFoundError(f"Sample CSV not found at {path}. Please add sample data or point to real datasets.")
    df = load_sample_csv(path)
    print(f"[data_ingest] Loaded {len(df)} rows from {path}")
    return df

if __name__ == "__main__":
    df = ingest()
    print(df.head())
