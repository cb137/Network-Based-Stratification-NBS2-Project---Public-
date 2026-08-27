import os
import argparse
import json
from datetime import datetime

import pandas as pd
import torch
import yaml

from SRW import (
    load_network,
    load_samples,
    load_grouplabels,
    train_srw_torch_dense,
)
from utils.plot_training import plot_training_curves


def load_config(path):
    ext = path.split(".")[-1].lower()
    if ext in ("yaml", "yml"):
        with open(path) as f:
            return yaml.safe_load(f)
    elif ext == "json":
        with open(path) as f:
            return json.load(f)
    else:
        raise ValueError(f"Unsupported config format: {ext}")



# Argument parser
parser = argparse.ArgumentParser()
parser.add_argument(
    "--config",
    type=str,
    required=True,
    help="Path to YAML config file",
)
args = parser.parse_args()



# Load config and dataset info
config = load_config(args.config)

dataset_name = config["dataset_name"]
data_root = config["paths"]["data_root"].rstrip("/")
files = config["files"]
params = config["hyperparams"]



# Build timestamped output directory
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
save_dir = f"outputs/{dataset_name}/{timestamp}"
os.makedirs(save_dir, exist_ok=True)

print(f"\n=== Saving results to: {save_dir} ===\n")



# Build full file paths
edge_file         = f"{data_root}/{files['edge']}"
train_data_file   = f"{data_root}/{files['train_data']}"
train_label_file  = f"{data_root}/{files['train_labels']}"
val_data_file     = f"{data_root}/{files['val_data']}"
val_label_file    = f"{data_root}/{files['val_labels']}"
feature_name_file = f"{data_root}/{files['feature_names']}"



# Load Data
edges, features, node_names = load_network(edge_file)
P_init_train, sample_names_train = load_samples(train_data_file, node_names)
P_init_val, sample_names_val     = load_samples(val_data_file, node_names)

group_labels_train = load_grouplabels(train_label_file)
group_labels_val   = load_grouplabels(val_label_file)

# Feature names
feature_names = []
with open(feature_name_file) as f:
    for line in f.read().strip().splitlines():
        feature_names.append(line)
feature_names.append("selfloop")
feature_names.append("intercept")



# Save config to output folder
with open(f"{save_dir}/config_used.json", "w") as f:
    json.dump(config, f, indent=4)


# Train SRW Model
model, history = train_srw_torch_dense(
    edges=edges,
    edge_features=features,
    P_init_train_csr=P_init_train,
    group_labels_train=group_labels_train,
    rst_prob=params["rst_prob"],
    lam=params["lam"],
    norm_type=params["norm_type"],
    WMW_b=params["WMW_b"],
    P_init_val_csr=P_init_val,
    group_labels_val=group_labels_val,
    lr=params["lr"],
    max_epochs=params["max_epochs"],
    early_stop=params["early_stop"],
)

print("\n=== Training Finished ===\n")


# Save checkpoint
with torch.no_grad():
    Q = model.compute_Q().cpu()
    P_train, P_val = model.compute_P_train_and_val(model.compute_Q())
    P_train = P_train.cpu()
    if P_val is not None:
        P_val = P_val.cpu()

w = model.w.detach().cpu()

torch.save(
    {
        "weights": w,
        "Q": Q,
        "P_train": P_train,
        "P_val": P_val,
        "node_names": node_names,
        "sample_names_train": sample_names_train,
        "sample_names_val": sample_names_val,
        "feature_names": feature_names,
        "hyperparams": params,
        "dataset": dataset_name,
        "timestamp": timestamp,
    },
    f"{save_dir}/checkpoint.pt",
)



# Save logs and weights
pd.DataFrame({"feature": feature_names, "weight": w.numpy()}) \
    .to_csv(f"{save_dir}/edge_feature_weights.tsv", sep="\t", index=False)

pd.DataFrame(history) \
    .to_csv(f"{save_dir}/training_log.tsv", sep="\t", index=False)

print("\n=== All Results Saved Successfully ===\n")

M_train = len(group_labels_train)
M_val = len(group_labels_val)

plot_training_curves(
    history,
    M_train=M_train,
    M_val=M_val,
    save_path=f"{save_dir}/training_plot.png",
    title_prefix=config["dataset_name"]
)


    