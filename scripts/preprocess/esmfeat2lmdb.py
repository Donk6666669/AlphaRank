import json
from pathlib import Path

import torch
import pandas as pd
from ml_collections.config_dict import ConfigDict
from tqdm import tqdm

from protenix.data.esm_featurizer import ESMFeaturizer
from protenix.utils.lmdb import LMDBDataset
from protenix.utils.logger import get_logger

logger = get_logger(__name__)


def main(
    json_path: str,
    output_lmdb_path: str,
    embedding_dir: str,
    mapping_csv_path: str,
):
    cfg_esm = {
        "enable": True,
        "model_name": "esm2-3b",
        "embedding_dim": 2560,
        "embedding_dir": embedding_dir,
        "sequence_fpath": mapping_csv_path,
    }
    cfg_esm = ConfigDict(cfg_esm)

    with open(json_path, "r") as f:
        data = json.load(f)
    unique_uniprot_samples = {}
    for sample in tqdm(data, ncols=80):
        uniprot_id = sample["name"].split("_")[0]
        if uniprot_id not in unique_uniprot_samples:
            unique_uniprot_samples[uniprot_id] = sample

    Path(embedding_dir).mkdir(parents=True, exist_ok=True)

    ESMFeaturizer.precompute_esm_embedding(
        list(unique_uniprot_samples.values()),
        cfg_esm.model_name,
        cfg_esm.embedding_dir,
        cfg_esm.sequence_fpath,
    )

    esm_data = pd.read_csv(cfg_esm.sequence_fpath)
    batch_data = {}
    for _, row in esm_data.iterrows():
        seq_label = row["seq_label"]
        uniprot_id = seq_label.split("_")[0]
        esm_key = f"esm_{uniprot_id}"
        embedding_path = (
            Path(cfg_esm.embedding_dir) / str(row["part_id"]) / f"{seq_label}.pt"
        )
        embedding = torch.load(embedding_path, map_location="cpu")
        batch_data[esm_key] = embedding

    lmdb = LMDBDataset(output_lmdb_path, readonly=False)
    if len(batch_data) > 0:
        lmdb.write_data(
            batch_data
        )
        lmdb.set_split("esm", sorted(batch_data.keys()))
        logger.info(
            f"Saved {len(batch_data)} samples to LMDB at {output_lmdb_path}"
        )


if __name__ == "__main__":
    json_path = "/data/rerank/protenix/chembl_bdb/chembl_bdb_protenix.json"
    output_lmdb_path = "/data/rerank/protenix/chembl_bdb/test.lmdb"
    embedding_dir = "/data/rerank/protenix_release/esm_cache/protenix_esm2_3b"
    sequence_fpath = (
        "/data/rerank/protenix_release/esm_cache/protenix_labels.csv"
    )
    main(json_path, output_lmdb_path, embedding_dir, sequence_fpath)
