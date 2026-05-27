#!/usr/bin/env python3
"""Run AlphaRank inference from pair/protein LMDB files.

Input LMDBs are produced by pipeline/run_pipeline.py:
- pair LMDB: named protein-ligand embeddings
- protein LMDB: protein-only embeddings extracted from pair LMDB

Output is one CSV per target with columns: ligand,score.
"""

import argparse
import os
from collections import defaultdict

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from protenix.model.rank_model import HYPMLPPairRanker
from protenix.utils.lmdb import LMDBDataset


class RankerDataset(Dataset):
    def __init__(self, entries):
        self.entries = entries

    def __getitem__(self, idx):
        return self.entries[idx]

    def __len__(self):
        return len(self.entries)


def collate_fn(batch):
    return {
        "target": [item["target"] for item in batch],
        "ligand": [item["ligand"] for item in batch],
        "s_p": torch.stack([item["s_p"] for item in batch]),
        "s_m": torch.stack([item["s_m"] for item in batch]),
        "z_pm": torch.stack([item["z_pm"] for item in batch]),
        "pro_s": torch.stack([item["pro_s"] for item in batch]),
    }


def average_if_recycled(value, axes=None):
    array = np.asarray(value)
    if axes is None:
        return np.mean(array, axis=0) if array.ndim > 1 else array
    return np.mean(array, axis=axes) if array.ndim > len(axes) else array


def load_entries(data_paths, protein_path):
    protein_data = LMDBDataset(protein_path)
    entries = []

    for data_path in data_paths:
        pair_data = LMDBDataset(data_path)
        with pair_data.env.begin(db=pair_data.db[pair_data.DATA_DB], write=False) as txn:
            keys = [key.decode() for key in txn.cursor().iternext(values=False)]

        for key in tqdm(keys, desc=f"Loading {data_path}"):
            item = pair_data[key]
            name = item["name"]
            if "flip" in name:
                continue

            protein_key = key.split("/")[0]
            s_inputs_p, pro_s = protein_data[protein_key]
            prot_inputs, lig_inputs, prot_s, lig_s, pair_emb = item["emb"]

            target, ligand = name.split(",", 1)
            if np.asarray(prot_inputs).ndim > 1:
                prot_s = np.mean(prot_s, axis=0)
                lig_s = np.mean(lig_s, axis=0)
                pair_emb = np.mean(pair_emb, axis=(0, 1))
                pro_s = np.mean(pro_s, axis=0)

            entries.append(
                {
                    "target": target,
                    "ligand": ligand,
                    "s_p": torch.tensor(prot_s, dtype=torch.float32),
                    "s_m": torch.tensor(lig_s, dtype=torch.float32),
                    "z_pm": torch.tensor(pair_emb, dtype=torch.float32),
                    "pro_s": torch.tensor(pro_s, dtype=torch.float32),
                }
            )

    return entries


def load_model(ckpt_path, device):
    model = HYPMLPPairRanker().to(device)
    print(f"Loading checkpoint from {ckpt_path}")
    checkpoint = torch.load(ckpt_path, map_location=device)

    if "state_dict" in checkpoint:
        state_dict = {
            key.replace("model.", ""): value
            for key, value in checkpoint["state_dict"].items()
            if key.startswith("model.")
        }
    else:
        state_dict = checkpoint

    model.load_state_dict(state_dict)
    model.eval()
    return model


def run_inference(entries, ckpt_path, batch_size):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_model(ckpt_path, device)
    loader = DataLoader(
        RankerDataset(entries),
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_fn,
    )

    results = defaultdict(dict)
    with torch.no_grad():
        for batch in tqdm(loader, desc="Forwarding"):
            outputs = model.single_forward(
                s_p=batch["s_p"].to(device),
                s_m=batch["s_m"].to(device),
                z_pm=batch["z_pm"].to(device),
                prot_s=batch["pro_s"].to(device),
            )
            scores = outputs["pred"].detach().cpu().numpy()

            for target, ligand, score in zip(batch["target"], batch["ligand"], scores):
                results[target.lower()][str(ligand).replace("&", "/")] = float(score)

    return results


def save_results(results, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    for target, ligand_scores in results.items():
        # score is negative hyperbolic geodesic distance: larger is stronger.
        sorted_items = sorted(ligand_scores.items(), key=lambda item: item[1], reverse=True)
        csv_path = os.path.join(output_dir, f"{target}.csv")
        pd.DataFrame(sorted_items, columns=["ligand", "score"]).to_csv(csv_path, index=False)
        print(f"Saved {csv_path}")


def parse_args():
    parser = argparse.ArgumentParser(description="AlphaRank affinity inference")
    parser.add_argument("--data_paths", nargs="+", required=True, help="One or more named protein-ligand LMDB paths")
    parser.add_argument("--protein_path", required=True, help="Protein-only LMDB path")
    parser.add_argument("--ckpt_path", required=True, help="AlphaRank checkpoint path")
    parser.add_argument("--output_dir", default="./results_openbind", help="Directory for per-target CSV files")
    parser.add_argument("--batch_size", type=int, default=64, help="Inference batch size")
    return parser.parse_args()


def main():
    if torch.__version__ >= "1.12":
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    args = parse_args()
    entries = load_entries(args.data_paths, args.protein_path)
    print(f"Loaded {len(entries)} entries.")
    results = run_inference(entries, args.ckpt_path, args.batch_size)
    save_results(results, args.output_dir)


if __name__ == "__main__":
    main()
