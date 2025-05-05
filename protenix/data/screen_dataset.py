import json
import os
import random
import traceback
from copy import deepcopy
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Optional, Union, Tuple

import numpy as np
import pandas as pd
import torch
from biotite.structure.atoms import AtomArray
from ml_collections.config_dict import ConfigDict
from torch.utils.data import Dataset

from protenix.data.constants import EvaluationChainInterface
from protenix.data.data_pipeline import DataPipeline
from protenix.data.esm_featurizer import ESMFeaturizer
from protenix.data.featurizer import Featurizer
from protenix.data.msa_featurizer import MSAFeaturizer, tokenize_msa
from protenix.data.tokenizer import AtomArrayTokenizer, TokenArray
from protenix.data.utils import data_type_transform, make_dummy_feature
from protenix.utils.cropping import CropData
from protenix.utils.file_io import read_indices_csv
from protenix.utils.logger import get_logger
from protenix.utils.torch_utils import dict_to_tensor
from protenix.utils.lmdb import LMDBDataset

logger = get_logger(__name__)


class FeatureCompressor:

    TYPE = Enum("TYPE", "ALL_ZERO SPARSE")

    def is_all_zero(self, tensor: torch.Tensor) -> bool:
        return torch.all(tensor == 0)

    def is_sparse(self, tensor: torch.Tensor) -> bool:
        return torch.count_nonzero(tensor) < tensor.numel() / 2

    def replace_zeros(self, tensor: torch.Tensor) -> Tuple[torch.Tensor, dict]:
        return None, {
            "type": self.TYPE.ALL_ZERO.name,
            "size": tensor.size(),
            "dtype": tensor.dtype,
        }

    def replace_sparse(self, tensor: torch.Tensor) -> Tuple[torch.Tensor, dict]:
        return tensor.to_sparse(), {"type": self.TYPE.SPARSE.name}

    def restore_zeros(self, tensor: torch.Tensor, record: dict) -> torch.Tensor:
        return torch.zeros(record["size"], dtype=record["dtype"])

    def restore_sparse(
        self, tensor: torch.Tensor, record: dict
    ) -> torch.Tensor:
        return tensor.to_dense()

    def _get_tensor(self, features_dict, key):
        sub_keys = key.split(".")
        tensor = features_dict
        for key in sub_keys:
            tensor = tensor[key]
        return tensor

    def _set_tensor(self, features_dict, key, value):
        sub_keys = key.split(".")
        tensor = features_dict
        for key in sub_keys[:-1]:
            tensor = tensor[key]
        tensor[sub_keys[-1]] = value

    def compress(self, features_dict: dict, replace: bool = False) -> dict:
        if not replace:
            target = deepcopy(features_dict)
        else:
            target = features_dict
        keys = list(target.keys())
        compress_records = {}
        while len(keys) > 0:
            key = keys.pop(0)
            tensor = self._get_tensor(target, key)
            if isinstance(tensor, torch.Tensor):
                if self.is_all_zero(tensor):
                    replace_tensor, record = self.replace_zeros(tensor)
                    self._set_tensor(target, key, replace_tensor)
                    compress_records[key] = record
                elif self.is_sparse(tensor):
                    replace_tensor, record = self.replace_sparse(tensor)
                    self._set_tensor(target, key, replace_tensor)
                    compress_records[key] = record
            elif isinstance(tensor, dict):
                for sub_key in tensor.keys():
                    keys.append(f"{key}.{sub_key}")
        target["__compress_records"] = compress_records
        return target

    def decompress(self, features_dict: dict, replace: bool = False) -> dict:
        if "__compress_records" not in features_dict:
            return features_dict
        if not replace:
            target = deepcopy(features_dict)
        else:
            target = features_dict
        compress_records = target.pop("__compress_records")
        for key, record in compress_records.items():
            tensor = self._get_tensor(target, key)
            compress_type = record["type"]
            if type(compress_type) == str:
                compress_type = self.TYPE[compress_type]
            if compress_type == self.TYPE.ALL_ZERO:
                self._set_tensor(
                    target, key, self.restore_zeros(tensor, record)
                )
            elif compress_type == self.TYPE.SPARSE:
                self._set_tensor(
                    target, key, self.restore_sparse(tensor, record)
                )
        return target

        
def standardize_activity_value(value):
    return - torch.log10(
        torch.tensor(value * 1e-9, dtype=torch.float32)
    ) # Convert nM to M and calculate - log10(value_in_M)


class ComplexFeatureDataset(Dataset):
    def __init__(
        self,
        lmdb_path: str,
        defatult_split: str = "chembl_bdb",
        use_msa: bool = True,
        msa_split: str = "msa",
        use_esm: bool = False,
        esm_split: str = "esm",
    ):
        self.lmdb_path = lmdb_path
        self.default_split = defatult_split
        self.use_msa = use_msa
        self.msa_split = msa_split
        self.use_esm = use_esm
        self.esm_split = esm_split

        self.lmdb = LMDBDataset(lmdb_path, enable_cache=False)
        self.lmdb.set_default_split(defatult_split)
        self.compressor = FeatureCompressor()

    def update_msa_feature(self, meta, feature_dict, atom_array, token_array):
        uniprot_id = meta["uniprot_id"]
        msa_feats = self.lmdb[f"{self.msa_split}_{uniprot_id}"]

        msa_feats = tokenize_msa(
            msa_feats=msa_feats,
            token_array=token_array,
            atom_array=atom_array,
        )
        msa_features = {
            k: v
            for (k, v) in msa_feats.items()
            if k
            in [
                "msa",
                "has_deletion",
                "deletion_value",
                "deletion_mean",
                "profile",
            ]
        }
        feature_dict.update(dict_to_tensor(msa_features))
        return feature_dict

    def update_esm_feature(self, meta, feature_dict, atom_array, token_array):
        uniprot_id = meta["uniprot_id"]
        esm_feats = self.lmdb[f"{self.esm_split}_{uniprot_id}"]
        N_token, embedding_dim = esm_feats.shape

        esm_embedding = torch.zeros([len(token_array), embedding_dim])
        esm_embedding[:N_token] = esm_feats
        feature_dict["esm_token_embedding"] = esm_embedding
        return feature_dict

    def get_feature_dict(self, key, meta):
        feature_key = f"p_{key}"
        features = self.lmdb[feature_key]
        feature_dict = self.compressor.decompress(features["feature_dict"])
        feature_dict["token_bonds"] = feature_dict["token_bonds"].float()

        atom_array = features["atom_array"]
        aa_tokenizer = AtomArrayTokenizer(atom_array)
        token_array = aa_tokenizer.get_token_array()

        if self.use_msa:
            feature_dict = self.update_msa_feature(
                meta, feature_dict, atom_array, token_array
            )

        if self.use_esm:
            feature_dict = self.update_esm_feature(
                meta, feature_dict, atom_array, token_array
            )

        return feature_dict, atom_array, token_array

    def __len__(self):
        return len(self.lmdb)

    def __getitem__(self, index):
        sample = {}
        key = self.lmdb.get_split(self.default_split)[index]
        meta = self.lmdb[key]
        feature_dict, _, _ = self.get_feature_dict(key, meta)
        sample["input_feature_dict"] = feature_dict
        sample["activity"] = standardize_activity_value(
            meta["activity"]["value"]
        )
        return sample


def get_discrepant_pairs(assay_records, threshold=1.0):
    # Extract data to numpy arrays
    assay_records = sorted(assay_records, key=lambda x: x["activity"]["value"])
    names = [r["name"] for r in assay_records]
    relations = np.array([r["activity"]["relation"] for r in assay_records])
    values = np.array([r["activity"]["value"] for r in assay_records])
    n = len(names)

    # Convert nM to M and calculate 9 - log10(value_in_M)
    value_M = values * 1e-9  # All units are nM per user confirmation
    vals = -np.log10(value_M)

    # Calculate interval bounds for each record
    lower = np.full(n, -np.inf)
    upper = np.full(n, np.inf)

    # Handle equality cases
    eq_mask = relations == "="
    lower[eq_mask] = upper[eq_mask] = value_M[eq_mask]

    # Handle greater-than cases
    gt_mask = relations == ">"
    lower[gt_mask] = value_M[gt_mask]

    # Handle less-than cases
    lt_mask = relations == "<"
    upper[lt_mask] = value_M[lt_mask]

    # Vectorized calculations for all pairs
    i, j = np.triu_indices(n, 1)  # Get upper triangle indices (i < j)

    # Calculate absolute differences between all pairs
    delta = np.abs(vals[i] - vals[j])

    # Check interval overlaps using broadcasting
    no_overlap = (upper[i] < lower[j]) | (upper[j] < lower[i])

    # Combine conditions
    valid_mask = (delta >= threshold) & no_overlap

    # Generate sorted pairs
    pairs = [tuple([names[i[k]], names[j[k]]]) for k in np.where(valid_mask)[0]]

    return pairs


class ComplexFeatureRandomPairDataset(ComplexFeatureDataset):
    """samples per epoch = one random pair per assay"""

    def __init__(
        self,
        lmdb_path: str,
        pair_split: str = "assay_pairs",
        use_msa: bool = True,
        msa_split: str = "msa",
        use_esm: bool = False,
        esm_split: str = "esm",
        random_seed: int = 0,
    ):
        super().__init__(
            lmdb_path=lmdb_path,
            use_msa=use_msa,
            msa_split=msa_split,
            use_esm=use_esm,
            esm_split=esm_split,
        )
        self.assay_pairs = self.lmdb[pair_split]
        self.assay_pair_keys = sorted(self.assay_pairs.keys())
        self.random_seed = random_seed

    def __len__(self):
        return len(self.assay_pairs)

    def __getitem__(self, index):
        pairs = self.assay_pairs[self.assay_pair_keys[index]]
        key_a, key_b = random.choice(pairs)
        meta_a, meta_b = self.lmdb[key_a], self.lmdb[key_b]
        feature_dict_a, atom_array_a, token_array_a = self.get_feature_dict(
            key_a, meta_a
        )
        feature_dict_b, atom_array_b, token_array_b = self.get_feature_dict(
            key_b, meta_b
        )
        pair1_label = standardize_activity_value(meta_a["activity"]["value"])
        pair2_label = standardize_activity_value(meta_b["activity"]["value"])
        result = {
            "input_feature_dicts": [
                feature_dict_a,
                feature_dict_b,
            ],
            "activity": torch.tensor([pair1_label, pair2_label]),
            "pair_label": torch.tensor([pair2_label - pair1_label]),
        }
        return result
