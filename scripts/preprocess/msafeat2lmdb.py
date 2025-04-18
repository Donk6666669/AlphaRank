import pickle
import json
import time

import torch
from tqdm import tqdm

from protenix.data.data_pipeline import DataPipeline
from protenix.data.json_to_feature import SampleDictToFeatures
from protenix.data.msa_featurizer import InferenceMSAFeaturizer
from protenix.utils.lmdb import LMDBDataset
from protenix.utils.logger import get_logger

logger = get_logger(__name__)


def process_sample(sample_dict):
    """Process a single sample and return features and atom array"""
    # 1. Basic features
    sample2feat = SampleDictToFeatures(sample_dict)
    features_dict, atom_array, token_array = sample2feat.get_feature_dict()
    features_dict["distogram_rep_atom_mask"] = torch.Tensor(
        atom_array.distogram_rep_atom_mask
    ).long()
    return features_dict, atom_array, token_array


def main(
    json_path: str,
    output_lmdb_path: str,
    replace_msa_root_from: str = None,
    replace_msa_root_to: str = None,
    prefix="msa",
):

    with open(json_path, "r") as f:
        data = json.load(f)
    unique_uniprot_samples = {}
    for sample in tqdm(data, ncols=80):
        uniprot_id = sample["name"].split("_")[0]
        if uniprot_id not in unique_uniprot_samples:
            unique_uniprot_samples[uniprot_id] = sample

    lmdb = LMDBDataset(output_lmdb_path, readonly=False)
    batch_data = {}
    for uniprot_id, sample in tqdm(unique_uniprot_samples.items(), ncols=80):
        # features_dict, atom_array, token_array = process_sample(sample)
        # entity_to_asym_id = DataPipeline.get_label_entity_id_to_asym_id_int(atom_array)
        entity_to_asym_id = {"1": {0}, "2": {1}}
        sequences = sample["sequences"]
        if replace_msa_root_from and replace_msa_root_to:
            sequences[0]["proteinChain"]["msa"][
                "precomputed_msa_dir"
            ] = sequences[0]["proteinChain"]["msa"][
                "precomputed_msa_dir"
            ].replace(
                replace_msa_root_from, replace_msa_root_to
            )
        msa_feats = (
            InferenceMSAFeaturizer.get_inference_prot_msa_features_for_assembly(
                bioassembly=sequences,
                entity_to_asym_id=entity_to_asym_id,
            )
        )
        batch_data[f"{prefix}_{uniprot_id}"] = msa_feats

        write_batch_size = 100
        if len(batch_data) >= write_batch_size:
            lmdb.write_data(batch_data)
            lmdb.set_split(
                prefix, sorted(batch_data.keys()), append=True, deduplicate=False
            )
            logger.info(
                f"Saved {len(batch_data)} samples to LMDB at {output_lmdb_path}"
            )
            batch_data = {}

    if len(batch_data) > 0:
        lmdb.write_data(batch_data)
        lmdb.set_split(
            prefix, sorted(batch_data.keys()), append=True, deduplicate=False
        )
        logger.info(
            f"Saved {len(batch_data)} samples to LMDB at {output_lmdb_path}"
        )


if __name__ == "__main__":
    json_path = "/data/rerank/protenix/chembl_bdb/chembl_bdb_protenix.json"
    output_lmdb_path = "/data/rerank/protenix/chembl_bdb/msa_feat.lmdb"
    main(json_path, output_lmdb_path, prefix="msa_filtered")

    # output_lmdb_path = "/data/rerank/protenix/chembl_bdb/msa_feat_full.lmdb"
    # replace_from = "/data/rerank/protenix/chembl_bdb/precomputed_msa"
    # replace_to = "/data/rerank/protenix/chembl_bdb/precomputed_msa_full"
    # main(json_path, output_lmdb_path, replace_from, replace_to)
