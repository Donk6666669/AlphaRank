import random
from typing import List, Union, Dict, Any, Optional
from functools import lru_cache

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from hydra.utils import instantiate
from lightning import LightningDataModule
from torch.utils.data.dataloader import default_collate
from torch.nn.utils.rnn import pad_sequence

from protenix.utils.lmdb import LMDBDataset
from protenix.utils.logger import get_logger

logger = get_logger(__name__)


class DatasetBase(Dataset):

    def __init__(
        self,
        meta_path: str,
        meta_split: str = "train_3w",
        feat_lmdb_path: Union[str, List[str]] = None,
        feat_split: Union[str, List[str]] = None,
        include_screen: bool = True,
        max_screen: int = None,
        include_rerank: bool = True,
        max_rerank: int = None,
    ):
        if isinstance(feat_lmdb_path, str):
            feat_lmdb_path = [feat_lmdb_path]
        if isinstance(feat_split, str):
            feat_split = [feat_split] * len(feat_lmdb_path)
        assert len(feat_lmdb_path) == len(
            feat_split
        ), "lmdb_path and split must have the same length"

        self.meta_path = meta_path
        self.meta_split = meta_split
        self.lmdb_path = feat_lmdb_path
        self.feat_split = feat_split
        self.include_screen = include_screen
        self.max_screen = max_screen
        self.include_rerank = include_rerank
        self.max_rerank = max_rerank

        self.meta_dataset = LMDBDataset(meta_path)
        self.meta_keys = self.meta_dataset[self.meta_split]

        self.lmdb_list = []
        self.keys_list = []
        self.keys2lmdbidx = {}
        for i, (path, sp) in enumerate(zip(self.lmdb_path, self.feat_split)):
            lmdb_dataset = LMDBDataset(path)
            keys = lmdb_dataset.get_split(sp)
            assert len(keys) > 0, f"split {sp} is empty for {path}"

            self.keys2lmdbidx.update({key: i for key in keys})
            self.lmdb_list.append(lmdb_dataset)

        self.filter_keys()
        self.check_keys()

    def filter_keys(self):
        new_keys = []
        n_screen = 0
        n_rerank = 0
        for key in self.meta_keys:
            uniprot_id, positive_key, negative_key = key.split("/")
            if uniprot_id in negative_key:
                if self.include_rerank:
                    if self.max_rerank is None or n_rerank < self.max_rerank:
                        new_keys.append(key)
                        n_rerank += 1
            else:
                if self.include_screen:
                    if self.max_screen is None or n_screen < self.max_screen:
                        new_keys.append(key)
                        n_screen += 1
        self.meta_keys = new_keys
        logger.info(
            f"total samples: {len(self.meta_keys)}, "
            f"screen samples: {n_screen}, "
            f"rerank samples: {n_rerank}"
        )

    def check_keys(self):
        pass

    def __len__(self):
        return len(self.meta_keys)

    def __getitem__(self, idx):
        raise NotImplementedError("getitem method not implemented")


class PairDataset(DatasetBase):
    def check_keys(self):
        missing_samples = set()
        missing_keys = set()
        for key in self.meta_keys:
            uniprot_id, positive_key, negative_key = key.split("/")
            positive_feat_key = f"{uniprot_id}/{positive_key}"
            negative_feat_key = f"{uniprot_id}/{negative_key}"

            miss_positive_feat_key = positive_feat_key not in self.keys2lmdbidx
            miss_negative_feat_key = negative_feat_key not in self.keys2lmdbidx
            if miss_positive_feat_key or miss_negative_feat_key:
                missing_samples.add(key)
                if miss_positive_feat_key:
                    missing_keys.add(positive_feat_key)
                if miss_negative_feat_key:
                    missing_keys.add(negative_feat_key)

        logger.info(f"total samples: {len(self.meta_keys)}")
        logger.info(f"total keys: {len(self.keys2lmdbidx)}")
        logger.info(f"missing samples: {len(missing_samples)}")
        logger.info(f"missing keys: {len(missing_keys)}")

    def parse_feat(self, feat_key):
        # Implement this method in subclasses
        raise NotImplementedError("parse method not implemented")

    def __getitem__(self, idx):
        try:
            pair_key = self.meta_keys[idx]
            uniprot_id, positive_key, negative_key = pair_key.split("/")
            positive_feat_key = f"{uniprot_id}/{positive_key}"
            negative_feat_key = f"{uniprot_id}/{negative_key}"

            # whether weak-strong active pairs or active inactive pairs
            hard = uniprot_id in negative_key

            # positive_feat = self.parse_feat(self.lmdb_list[
            #     self.keys2lmdbidx[positive_feat_key]
            # ][positive_feat_key])
            # negative_feat = self.parse_feat(self.lmdb_list[
            #     self.keys2lmdbidx[negative_feat_key]
            # ][negative_feat_key])
            positive_feat = self.parse_feat(positive_feat_key)
            negative_feat = self.parse_feat(negative_feat_key)

            positive_label = 1.0
            negative_label = 0.0

            reverse = random.random() < 0.5
            if reverse:
                res = {
                    "pm1": negative_feat,
                    "pm2": positive_feat,
                    "label": positive_label
                    > negative_label,  # label: pm2 > pm1
                }
            else:
                res = {
                    "pm1": positive_feat,
                    "pm2": negative_feat,
                    "label": negative_label > positive_label,
                }
            res["hard"] = hard
            res["idx"] = idx
            return res
        except Exception:
            return None


class FullPairDataset(PairDataset):
    def parse_feat(self, feat_key):
        data = self.lmdb_list[self.keys2lmdbidx[feat_key]][feat_key]
        s_inputs, s, (z_pm, _) = data
        len_p, len_m = z_pm.shape[0], z_pm.shape[1]
        # s_inputs_p = s_inputs[:len_p]
        # s_inputs_m = s_inputs[len_p:]
        # s_p = s[:len_p]
        # s_m = s[len_p:]
        res = {
            # "s_inputs_p": s_inputs_p.float(),
            # "s_inputs_m": s_inputs_m.float(),
            # "s_p": s_p.float(),
            # "s_m": s_m.float(),
            "z_pm": z_pm.float(),
            "len_p": len_p,
            "len_m": len_m,
            # "mask_p": torch.ones(len_p, dtype=torch.bool),
            # "mask_m": torch.ones(len_m, dtype=torch.bool),
            "mask_pm": torch.ones(len_p, len_m, dtype=torch.bool),
        }
        return res

    def collate_fn(self, batch):
        valid_batch = [b for b in batch if b is not None]

        def process_pm(pm_list):
            # s_inputs_p = [pm["s_inputs_p"] for pm in pm_list]
            # s_inputs_m = [pm["s_inputs_m"] for pm in pm_list]
            # s_p = [pm["s_p"] for pm in pm_list]
            # s_m = [pm["s_m"] for pm in pm_list]
            z_pm = [pm["z_pm"] for pm in pm_list]
            lens_p = [pm["len_p"] for pm in pm_list]
            lens_m = [pm["len_m"] for pm in pm_list]
            # masks_p = [pm["mask_p"] for pm in pm_list]
            # masks_m = [pm["mask_m"] for pm in pm_list]
            masks_pm = [pm["mask_pm"] for pm in pm_list]

            max_p = max(lens_p)
            max_m = max(lens_m)

            # padded_s_inputs_p = pad_sequence(s_inputs_p, batch_first=True)
            # padded_s_inputs_m = pad_sequence(s_inputs_m, batch_first=True)
            # padded_s_p = pad_sequence(s_p, batch_first=True)
            # padded_s_m = pad_sequence(s_m, batch_first=True)

            padded_z_pm = torch.stack(
                [
                    F.pad(
                        mat,
                        (0, 0, 0, max_m - mat.size(1), 0, max_p - mat.size(0)),
                        value=0,
                    )
                    for mat in z_pm
                ]
            )  # (batch, max_p, max_m)

            # padded_masks_p = pad_sequence(
            #     masks_p, batch_first=True, padding_value=False
            # )  # (batch, max_p)
            # padded_masks_m = pad_sequence(
            #     masks_m, batch_first=True, padding_value=False
            # )  # (batch, max_m)
            padded_masks_pm = torch.stack(
                [
                    F.pad(
                        mat,
                        (0, max_m - mat.size(1), 0, max_p - mat.size(0)),
                        value=False,
                    )
                    for mat in masks_pm
                ]
            )  # (batch, max_p, max_m)

            return {
                # "s_inputs_p": padded_s_inputs_p,
                # "s_inputs_m": padded_s_inputs_m,
                # "s_p": padded_s_p,
                # "s_m": padded_s_m,
                "z_pm": padded_z_pm,
                # "mask_p": padded_masks_p,
                # "mask_m": padded_masks_m,
                "mask_pm": padded_masks_pm,
                "len_p": torch.tensor(lens_p),
                "len_m": torch.tensor(lens_m),
            }

        pm1_list = [item["pm1"] for item in valid_batch]
        pm2_list = [item["pm2"] for item in valid_batch]

        collated_pm1 = process_pm(pm1_list)
        collated_pm2 = process_pm(pm2_list)

        labels = torch.tensor(
            [item["label"] for item in valid_batch], dtype=torch.float
        )
        hards = torch.tensor(
            [item["hard"] for item in valid_batch], dtype=torch.bool
        )
        idxs = torch.tensor(
            [item["idx"] for item in valid_batch], dtype=torch.long
        )

        return {
            "pm1": collated_pm1,
            "pm2": collated_pm2,
            "label": labels,
            "hard": hards,
            "idx": idxs,
        }


class FullReducePairDataset(PairDataset):

    @lru_cache(maxsize=None)
    def parse_feat(self, feat_key):
        data = self.lmdb_list[self.keys2lmdbidx[feat_key]][feat_key]
        s_inputs, s, (z_pm, z_mp) = data
        len_p = z_pm.shape[0]
        s_inputs_p = s_inputs[:len_p].mean(dim=0)
        s_inputs_m = s_inputs[len_p:].mean(dim=0)
        s_p = s[:len_p].mean(dim=0)
        s_m = s[len_p:].mean(dim=0)
        z_pm = z_pm.mean(dim=(0, 1))
        z_mp = z_mp.mean(dim=(0, 1))
        res = {
            "s_inputs_p": s_inputs_p.float(),
            "s_inputs_m": s_inputs_m.float(),
            "s_p": s_p.float(),
            "s_m": s_m.float(),
            "z_pm": z_pm.float(),
            "z_mp": z_mp.float(),
        }
        return res

    def collate_fn(self, batch):
        batch = [item for item in batch if item is not None]
        return default_collate(batch)


class ReducePairDataset(PairDataset):

    @lru_cache(maxsize=None)
    def parse_feat(self, feat_key):
        data = self.lmdb_list[self.keys2lmdbidx[feat_key]][feat_key]
        s_inputs_p, s_inputs_m, s_p, s_m, z_pm = data
        res = {
            "s_inputs_p": s_inputs_p.float(),
            "s_inputs_m": s_inputs_m.float(),
            "s_p": s_p.float(),
            "s_m": s_m.float(),
            "z_pm": z_pm.float(),
        }
        return res

    def collate_fn(self, batch):
        batch = [item for item in batch if item is not None]
        return default_collate(batch)


class TripletDataset(DatasetBase):
    def check_keys(self):
        missing_keys = set()
        for key in self.meta_keys:
            miss_key = key not in self.keys2lmdbidx
            if miss_key:
                missing_keys.add(key)

        logger.info(f"total samples: {len(self.meta_keys)}")
        logger.info(f"missing samples: {len(missing_keys)}")

    def parse_feat(self, feat_key):
        # Implement this method in subclasses
        raise NotImplementedError("parse_feat method not implemented")

    def __getitem__(self, idx):
        try:
            triplet_key = self.meta_keys[idx]
            uniprot_id, positive_key, negative_key = triplet_key.split("/")
            hard = uniprot_id in negative_key

            positive_label = 1.0
            negative_label = 0.0
            reverse_m1m2 = random.random() < 0.5
            feat = self.parse_feat(triplet_key, reverse_m1m2=reverse_m1m2)
            if reverse_m1m2:
                label = positive_label > negative_label  # label: pm2 > pm1
            else:
                label = negative_label > positive_label

            res = {
                "label": label,
                "hard": hard,
                "idx": idx,
                **feat,
            }
            return res
        except Exception:
            return None


class FullReduceTripletDataset(TripletDataset):
    def parse_feat(self, feat_key, reverse_m1m2=False):
        data = self.lmdb_list[self.keys2lmdbidx[feat_key]][feat_key]
        s_inputs, s, (z_pm1, z_pm2, z_m1m2, z_m1p, z_m2p, z_m2m1) = data

        len_p, len_m1, len_m2 = z_pm1.shape[0], z_pm1.shape[1], z_pm2.shape[1]
        s_inputs_p = s_inputs[:len_p].mean(dim=0)
        s_inputs_m1 = s_inputs[len_p : len_p + len_m1].mean(dim=0)
        s_inputs_m2 = s_inputs[len_p + len_m1 :].mean(dim=0)
        s_p = s[:len_p].mean(dim=0)
        s_m1 = s[len_p : len_p + len_m1].mean(dim=0)
        s_m2 = s[len_p + len_m1 :].mean(dim=0)
        z_pm1 = z_pm1.mean(dim=(0, 1))
        z_pm2 = z_pm2.mean(dim=(0, 1))
        z_m1m2 = z_m1m2.mean(dim=(0, 1))

        if reverse_m1m2:
            res = {
                "s_inputs_p": s_inputs_p.float(),
                "s_inputs_m1": s_inputs_m2.float(),
                "s_inputs_m2": s_inputs_m1.float(),
                "s_p": s_p.float(),
                "s_m1": s_m2.float(),
                "s_m2": s_m1.float(),
                "z_pm1": z_pm2.float(),
                "z_pm2": z_pm1.float(),
                "z_m1m2": z_m1m2.float(),
            }
        else:
            res = {
                "s_inputs_p": s_inputs_p.float(),
                "s_inputs_m1": s_inputs_m1.float(),
                "s_inputs_m2": s_inputs_m2.float(),
                "s_p": s_p.float(),
                "s_m1": s_m1.float(),
                "s_m2": s_m2.float(),
                "z_pm1": z_pm1.float(),
                "z_pm2": z_pm2.float(),
                "z_m1m2": z_m1m2.float(),
            }
        return res

    def collate_fn(self, batch):
        batch = [item for item in batch if item is not None]
        return default_collate(batch)


class ReduceTripletDataset(TripletDataset):
    @lru_cache(maxsize=None)
    def parse_feat(self, feat_key, reverse_m1m2=False):
        data = self.lmdb_list[self.keys2lmdbidx[feat_key]][feat_key]
        (
            s_inputs_p,
            s_inputs_m1,
            s_inputs_m2,
            s_p,
            s_m1,
            s_m2,
            z_pm1,
            z_pm2,
            z_m1m2,
        ) = data
        if reverse_m1m2:
            return {
                "s_inputs_p": s_inputs_p.float(),
                "s_inputs_m1": s_inputs_m2.float(),
                "s_inputs_m2": s_inputs_m1.float(),
                "s_p": s_p.float(),
                "s_m1": s_m2.float(),
                "s_m2": s_m1.float(),
                "z_pm1": z_pm2.float(),
                "z_pm2": z_pm1.float(),
                "z_m1m2": z_m1m2.float(),
            }
        else:
            return {
                "s_inputs_p": s_inputs_p.float(),
                "s_inputs_m1": s_inputs_m1.float(),
                "s_inputs_m2": s_inputs_m2.float(),
                "s_p": s_p.float(),
                "s_m1": s_m1.float(),
                "s_m2": s_m2.float(),
                "z_pm1": z_pm1.float(),
                "z_pm2": z_pm2.float(),
                "z_m1m2": z_m1m2.float(),
            }

    def collate_fn(self, batch):
        batch = [item for item in batch if item is not None]
        return default_collate(batch)


class DataModule(LightningDataModule):
    def __init__(
        self,
        num_workers: int = 0,
        pin_memory: bool = False,
        batch_size: int = 1,
        prefetch_factor: int = 4,
        dataset_args: Optional[Dict[str, Any]] = None,
        **kwargs,
    ):
        super().__init__()
        self.save_hyperparameters(logger=False)

    def _dataloader(self, split):
        args = self.hparams.dataset_args[split]
        dataset = instantiate(args)
        dataloader = DataLoader(
            dataset,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            collate_fn=dataset.collate_fn,
            shuffle=split == "train",
            persistent_workers=self.hparams.num_workers > 0,
            prefetch_factor=self.hparams.prefetch_factor,
        )
        return dataloader

    def train_dataloader(self):
        return self._dataloader("train")

    def val_dataloader(self):
        return self._dataloader("val")
