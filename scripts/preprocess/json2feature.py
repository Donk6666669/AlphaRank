import os
import argparse
import json
from enum import Enum
from typing import Mapping, Any, Tuple
from pathlib import Path
from copy import deepcopy
from concurrent.futures import (
    ThreadPoolExecutor,
    as_completed,
    ProcessPoolExecutor,
)

import torch
from biotite.structure import AtomArray
from ml_collections.config_dict import ConfigDict
from tqdm import tqdm

from protenix.data.data_pipeline import DataPipeline
from protenix.data.json_to_feature import SampleDictToFeatures
from protenix.data.esm_featurizer import ESMFeaturizer
from protenix.data.msa_featurizer import InferenceMSAFeaturizer
from protenix.data.utils import data_type_transform, make_dummy_feature
from protenix.utils.torch_utils import dict_to_tensor
from protenix.utils.lmdb import LMDBDataset
from protenix.utils.logger import get_logger

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

    def decompress(self, features_dict: dict, replace: bool = True) -> dict:
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


def process_sample(sample_dict: Mapping[str, Any]) -> tuple[dict, AtomArray]:
    try:
        """Process a single sample and return features and atom array"""
        # 1. Basic features
        sample2feat = SampleDictToFeatures(sample_dict)
        features_dict, atom_array, token_array = sample2feat.get_feature_dict()
        features_dict["distogram_rep_atom_mask"] = torch.Tensor(
            atom_array.distogram_rep_atom_mask
        ).long()
        entity_poly_type = sample2feat.entity_poly_type

        # 2. MSA features
        msa_features = {}

        # 3. ESM features (using precomputed)
        esm_embedding_dim = 2560
        features_dict["esm_token_embedding"] = torch.zeros(
            [len(token_array), esm_embedding_dim]
        )

        # 4. Add dummy features
        dummy_feats = ["template"]
        if len(msa_features) == 0:
            dummy_feats.append("msa")
        features_dict = make_dummy_feature(
            features_dict=features_dict,
            dummy_feats=dummy_feats,
        )

        # 5. Data type conversion
        features_dict = data_type_transform(features_dict)

        # 6. Add dimension info
        N_token = features_dict["token_index"].shape[0]
        N_atom = features_dict["atom_to_token_idx"].shape[0]
        N_msa = features_dict["msa"].shape[0] if "msa" in features_dict else 0
        N_asym = len(torch.unique(features_dict["asym_id"]))

        features_dict.update(
            {
                "N_asym": torch.tensor([N_asym]),
                "N_token": torch.tensor([N_token]),
                "N_atom": torch.tensor([N_atom]),
                "N_msa": torch.tensor([N_msa]),
                "entity_poly_type": entity_poly_type,
            }
        )

        compressor = FeatureCompressor()
        features_dict = compressor.compress(features_dict, replace=True)
        return sample_dict["name"], features_dict, atom_array
    except Exception:
        logger.exception(f"Failed to process: {sample_dict}")
        raise


class ProtenixFeatureizer:
    def __init__(
        self,
        json_path: str,
        lmdb_path: str,
        n_parallel: int = None,
        write_batch_size: int = 1000,
        start: int = 0,
        end: int = None,
    ):
        self.json_path = json_path
        self.lmdb_dataset = LMDBDataset(lmdb_path, readonly=False)
        if n_parallel is None:
            self.n_parallel = os.cpu_count()
        else:
            self.n_parallel = n_parallel
        self.write_batch_size = write_batch_size

        # Load all samples from JSON
        print(f"Loading samples from {json_path}...")
        with open(json_path, "r") as f:
            self.samples = json.load(f)

        if end is None:
            self.end = len(self.samples)
        else:
            self.end = end
        logger.info(
            f"Processing samples from {start} to {self.end} (total {len(self.samples)})"
        )
        self.samples = self.samples[start : self.end]
        print(f"Loaded {len(self.samples)} samples.")

    def run(self):
        batch_data = {}
        for sample_dict in tqdm(self.samples, ncols=80):
            sample_name = sample_dict["name"]
            try:
                _, features, atom_array = process_sample(sample_dict)

                batch_data[f"p_{sample_name}"] = {
                    "feature_dict": features,
                    "atom_array": atom_array,
                }
                if len(batch_data) >= self.write_batch_size:
                    self.lmdb_dataset.write_data(batch_data)
                    batch_data = {}

            except Exception:
                logger.exception(f"failed to process {sample_name}")

        # Write any remaining data to LMDB
        if len(batch_data) > 0:
            self.lmdb_dataset.write_data(batch_data)

    def parallel_run(self):
        batch_data = {}
        total_samples = len(self.samples)
        pbar = tqdm(total=total_samples, ncols=80, desc="Processing samples")

        with ProcessPoolExecutor(max_workers=self.n_parallel) as executor:
            while len(self.samples) > 0:
                futures = []
                for _ in range(self.write_batch_size):
                    if len(self.samples) == 0:
                        break
                    sample = self.samples.pop(0)
                    futures.append(executor.submit(process_sample, sample))
                for future in as_completed(futures):
                    try:
                        sample_name, features, atom_array = future.result()

                        batch_data[f"p_{sample_name}"] = {
                            "feature_dict": features,
                            "atom_array": atom_array,
                        }

                    except Exception:
                        logger.exception("parallel failed")
                        continue
                    finally:
                        pbar.update(1)
                if len(batch_data) > 0:
                    self.lmdb_dataset.write_data(batch_data)
                    self.lmdb_dataset.set_split(
                        "feature",
                        sorted(batch_data.keys()),
                        append=True,
                        deduplicate=False,
                    )
                    batch_data = {}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--json_path",
        type=str,
        required=True,
        help="Path to the input JSON file containing sample data.",
    )
    parser.add_argument(
        "--lmdb_path",
        type=str,
        required=True,
        help="Path to the LMDB database where features will be stored.",
    )
    parser.add_argument(
        "--n_parallel",
        type=int,
        default=4,
        help="Number of parallel threads for processing.",
    )
    parser.add_argument(
        "--write_batch_size",
        type=int,
        default=100,
        help="Batch size for writing to LMDB.",
    )
    parser.add_argument(
        "--start",
        type=int,
        default=0,
        help="Start index for processing samples.",
    )
    parser.add_argument(
        "--end",
        type=int,
        default=None,
        help="End index for processing samples.",
    )

    args = parser.parse_args()
    featureizer = ProtenixFeatureizer(
        json_path=args.json_path,
        lmdb_path=args.lmdb_path,
        n_parallel=args.n_parallel,
        write_batch_size=args.write_batch_size,
        start=args.start,
        end=args.end,
    )
    if args.n_parallel == 1:
        featureizer.run()
    else:
        featureizer.parallel_run()
    print("Feature extraction complete.")


if __name__ == "__main__":
    main()
