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


# class DatasetBase(Dataset):

#     def __init__(
#         self,
#         meta_path: str,
#         meta_split: str = "train_3w",
#         feat_lmdb_path: Union[str, List[str]] = None,
#         feat_split: Union[str, List[str]] = None,
#         include_screen: bool = True,
#         max_screen: int = None,
#         include_rerank: bool = True,
#         max_rerank: int = None,
#     ):
#         if isinstance(feat_lmdb_path, str):
#             feat_lmdb_path = [feat_lmdb_path]
#         if isinstance(feat_split, str):
#             feat_split = [feat_split] * len(feat_lmdb_path)

#         print("DEBUG feat_lmdb_path:", feat_lmdb_path)  # 输出路径内容
#         print("DEBUG len(feat_lmdb_path):", len(feat_lmdb_path))  
#         print("DEBUG feat_split:", feat_split)          # 输出split内容
#         print("DEBUG len(feat_split):", len(feat_split))
#         assert len(feat_lmdb_path) == len(
#             feat_split
#         ), "lmdb_path and split must have the same length"

#         self.meta_path = meta_path
#         self.meta_split = meta_split
#         self.lmdb_path = feat_lmdb_path
#         self.feat_split = feat_split
#         self.include_screen = include_screen
#         self.max_screen = max_screen
#         self.include_rerank = include_rerank
#         self.max_rerank = max_rerank

#         self.meta_dataset = LMDBDataset(meta_path)
#         self.meta_keys = self.meta_dataset[self.meta_split]

#         self.lmdb_list = []
#         self.keys_list = []
#         self.keys2lmdbidx = {}
#         for i, (path, sp) in enumerate(zip(self.lmdb_path, self.feat_split)):
#             lmdb_dataset = LMDBDataset(path)
#             keys = lmdb_dataset.get_split(sp)
#             assert len(keys) > 0, f"split {sp} is empty for {path}"

#             self.keys2lmdbidx.update({key: i for key in keys})
#             self.lmdb_list.append(lmdb_dataset)

#         self.filter_keys()
#         self.check_keys()

#     def filter_keys(self):
#         new_keys = []
#         n_screen = 0
#         n_rerank = 0
#         for key in self.meta_keys:
#             uniprot_id, positive_key, negative_key = key.split("/")
#             if uniprot_id in negative_key:
#                 if self.include_rerank:
#                     if self.max_rerank is None or n_rerank < self.max_rerank:
#                         new_keys.append(key)
#                         n_rerank += 1
#             else:
#                 if self.include_screen:
#                     if self.max_screen is None or n_screen < self.max_screen:
#                         new_keys.append(key)
#                         n_screen += 1
#         self.meta_keys = new_keys
#         logger.info(
#             f"total samples: {len(self.meta_keys)}, "
#             f"screen samples: {n_screen}, "
#             f"rerank samples: {n_rerank}"
#         )

#     def check_keys(self):
#         pass

#     def __len__(self):
#         return len(self.meta_keys)

#     def __getitem__(self, idx):
#         raise NotImplementedError("getitem method not implemented")

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

        print("DEBUG feat_lmdb_path:", feat_lmdb_path)  # 输出路径内容
        print("DEBUG len(feat_lmdb_path):", len(feat_lmdb_path))  
        print("DEBUG feat_split:", feat_split)          # 输出split内容
        print("DEBUG len(feat_split):", len(feat_split))
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
        # ===== 在这里手动追加 =====
        extra_path = "/data/rerank/protenix/chembl_bdb/training_protein/pair/training_protein.lmdb"
        extra_split = "train"   # 你需要确认要用哪个 split
        extra_dataset = LMDBDataset(extra_path)
        extra_keys = extra_dataset.get_split(extra_split)
        assert len(extra_keys) > 0, f"split {extra_split} is empty for {extra_path}"

        extra_idx = len(self.lmdb_list)  # 新的 lmdb 在列表里的 index
        self.keys2lmdbidx.update({key: extra_idx for key in extra_keys})
        self.lmdb_list.append(extra_dataset)
        
        self.filter_keys()
        self.check_keys()

    def filter_keys(self):
        new_keys = []
        n_screen = 0
        n_rerank = 0

        for key in self.meta_keys:
            parts = key.split("/")

            # Pair 格式 (3 段)
            if len(parts) == 3:
                uniprot_id, positive_key, negative_key = parts
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

            # List 格式 (>=4 段)
            elif len(parts) > 3:
                uniprot_id = parts[0]
                ligand_keys = parts[1:]  # 从第二段开始全是 ligand
                # listwise 数据处理逻辑（可按你的排序要求等条件筛选）
                # 这里我假设全部保留
                new_keys.append(key)
                n_rerank += 1  # listwise 一般属于 rerank 样本

            else:
                logger.warning(f"Unexpected key format: {key}")

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
            # if positive_feat_key not in self.keys2lmdbidx:
            #         logger.warning(f"SB!PairDataset missing feat: {positive_feat_key}")
            #         #feats.append(None)
            #         #missing_any = True
            # else:
            #         # parse_feat 由子类实现（与 PairDataset 保持一致）
            #         #feats.append(self.parse_feat(feat_key))
            #         logger.warning(f"GOOD!PairDataset has feat: {positive_feat_key}")

            # if negative_feat_key not in self.keys2lmdbidx:
            #         logger.warning(f"SB!PairDataset missing feat: {negative_feat_key}")
            #         #feats.append(None)
            #         #missing_any = True
            # else:
            #         # parse_feat 由子类实现（与 PairDataset 保持一致）
            #         #feats.append(self.parse_feat(feat_key))
            #         logger.warning(f"GOOD!PairDataset has feat: {negative_feat_key}")


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
            logger.exception(f"Failed to parse pair {idx}")
            return None


# class ListDataset(DatasetBase):
#     """
#     专门处理 listwise keys: uniprot/lig1/lig2/...
#     依赖子类实现 parse_feat(feat_key) 来从 lmdb 中取到对应的特征。
#     __getitem__ 返回的字段示例:
#         {
#           "pm_list": [feat1, feat2, ...],   # feat 为 parse_feat 返回的结构
#           "labels": [rank1, rank2, ...],    # 你 listwise loss 需要的标签形式
#           "idx": idx,
#           "uniprot_id": uniprot_id,
#           "ligand_keys": ["lig1", "lig2", ...]
#         }
#     """
#     def check_keys(self):
#         missing_samples = set()
#         missing_keys = set()
#         for key in self.meta_keys:
#             parts = key.split("/")
#             if len(parts) < 2:
#                 logger.warning(f"ListDataset.check_keys unexpected key: {key}")
#                 missing_samples.add(key)
#                 continue
#             uniprot_id = parts[0]
#             ligand_keys = parts[1:]
#             sample_missing = False
#             for ligand in ligand_keys:
#                 feat_key = f"{uniprot_id}/{ligand}"
#                 if feat_key not in self.keys2lmdbidx:
#                     missing_keys.add(feat_key)
#                     sample_missing = True
#             if sample_missing:
#                 missing_samples.add(key)

#         logger.info(f"total samples: {len(self.meta_keys)}")
#         logger.info(f"total keys: {len(self.keys2lmdbidx)}")
#         logger.info(f"missing samples: {len(missing_samples)}")
#         logger.info(f"missing keys: {len(missing_keys)}")

#     def __getitem__(self, idx):
#         try:
#             key = self.meta_keys[idx]
#             parts = key.split("/")
#             if len(parts) < 2:
#                 logger.warning(f"ListDataset.__getitem__ unexpected key: {key}")
#                 return None
#             uniprot_id = parts[0]
#             ligand_keys = parts[1:]

#             feats = []
#             missing_any = False
#             for ligand in ligand_keys:
#                 feat_key = f"{uniprot_id}/{ligand}"
#                 if feat_key not in self.keys2lmdbidx:
#                     #logger.warning(f"SB!ListDataset missing feat: {feat_key}")
#                     feats.append(None)
#                     missing_any = True
#                 else:
#                     # parse_feat 由子类实现（与 PairDataset 保持一致）
#                     feats.append(self.parse_feat(feat_key))
#                     #logger.warning(f"GOOD!ListDataset has feat: {feat_key}")

#             # labels: 这里给一个常见的默认格式（按 affinity 从大到小给出 rank）
#             # 注意：你可以按你的 listwise loss 需求修改 labels 的形式
#             labels = list(range(len(feats), 0, -1))  # e.g. [N, N-1, ..., 1]

#             return {
#                 "pm_list": feats,
#                 "labels": labels,
#                 "idx": idx,
#                 "uniprot_id": uniprot_id,
#                 "ligand_keys": ligand_keys,
#             }
#         except Exception:
#             logger.exception(f"Failed to parse list sample {idx}")
#             return None

class ListDataset(DatasetBase):
    """
    专门处理 listwise keys: uniprot/lig1/lig2/...
    依赖子类实现 parse_feat(feat_key) 来从 lmdb 中取到对应的特征。
    __getitem__ 返回的字段示例:
        {
          "pm1": feat1,
          "pm2": feat2,
          "pm3": feat3,
          "idx": idx
        }
    """

    def check_keys(self):
        missing_samples = set()
        missing_keys = set()
        for key in self.meta_keys:
            parts = key.split("/")
            if len(parts) < 2:
                logger.warning(f"ListDataset.check_keys unexpected key: {key}")
                missing_samples.add(key)
                continue
            uniprot_id = parts[0]
            ligand_keys = parts[1:]
            sample_missing = False
            for ligand in ligand_keys:
                feat_key = f"{uniprot_id}/{ligand}"
                if feat_key not in self.keys2lmdbidx:
                    missing_keys.add(feat_key)
                    sample_missing = True
            if sample_missing:
                missing_samples.add(key)

        logger.info(f"total samples: {len(self.meta_keys)}")
        logger.info(f"total keys: {len(self.keys2lmdbidx)}")
        logger.info(f"missing samples: {len(missing_samples)}")
        logger.info(f"missing keys: {len(missing_keys)}")

    def __getitem__(self, idx):
        try:
            key = self.meta_keys[idx]
            parts = key.split("/")
            if len(parts) < 2:
                return None

            uniprot_id = parts[0]
            ligand_keys = parts[1:4]

            feats = []
            for ligand in ligand_keys:
                feat_key = f"{uniprot_id}/{ligand}"
                if feat_key not in self.keys2lmdbidx:
                    return None  # 直接丢弃
                feats.append(self.parse_feat(feat_key))

            while len(feats) < 3:
                feats.append(feats[-1])  # 或者复制最后一个填充

            return {
                "pm1": feats[0],
                "pm2": feats[1],
                "pm3": feats[2],
                "idx": idx
            }
        except Exception:
            logger.exception(f"Failed to parse list sample {idx}")
            return None


class FullPairDataset(PairDataset):
    def parse_feat(self, feat_key):
        data = self.lmdb_list[self.keys2lmdbidx[feat_key]][feat_key]
        s_inputs, s, (z_pm, _) = data
        len_p, len_m = z_pm.shape[0], z_pm.shape[1]
        # s_inputs_p = s_inputs[:len_p]
        # s_inputs_m = s_inputs[len_p:]
        s_p = s[:len_p]
        s_m = s[len_p:]
        assert (
            len_m == s_m.shape[0]
        ), f"len_m: {len_m} != s_m.shape[0]: {s_m.shape[0]} for {feat_key}"

        res = {
            # "s_inputs_p": s_inputs_p.float(),
            # "s_inputs_m": s_inputs_m.float(),
            "s_p": s_p.float(),
            "s_m": s_m.float(),
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
            bz = len(pm_list)
            # s_inputs_p = [pm["s_inputs_p"] for pm in pm_list]
            # s_inputs_m = [pm["s_inputs_m"] for pm in pm_list]
            s_p = [pm["s_p"] for pm in pm_list]
            s_m = [pm["s_m"] for pm in pm_list]
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
            padded_s_p = torch.zeros((bz, max_p, s_p[0].size(1))).to(s_p[0])
            padded_s_m = torch.zeros((bz, max_m, s_m[0].size(1))).to(s_m[0])
            for i in range(bz):
                padded_s_p[i, : lens_p[i], :] = s_p[i]
                padded_s_m[i, : lens_m[i], :] = s_m[i]

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
            # padded_mask_pm = torch.bmm(
            #     padded_masks_p.unsqueeze(2).float(),
            #     padded_masks_m.unsqueeze(1).float(),
            # )
            # padded_masks_pm = (padded_mask_pm > 0)  # (batch, max_p, max_m)
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
                "s_p": padded_s_p,
                "s_m": padded_s_m,
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

        #print(f"DEBUG: feat_key={feat_key}, data={data}, len={len(data)}")
        #print(f"DEBUG parse_feat: feat_key={feat_key}, data type={type(data)}, len={len(data) if hasattr(data, '__len__') else 'NA'}, data keys/type if dict: {list(data.keys()) if isinstance(data, dict) else 'NA'}")
        # 或者直接打印前几元素：
        #print(f"DEBUG data sample: {data if len(str(data)) < 500 else str(data)[:500]}")


        s_inputs, s, (z_pm, z_mp) = data
        len_p, len_m, _ = z_pm.shape
        assert (
            len_p + len_m == s.shape[0]
        ), f"length mismatch: {len_p} + {len_m} != {s.shape[0]} for {feat_key}"
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
        # for i, item in enumerate(batch):
        #     print(f"[Sample {i}]")
        #     for k, v in item.items():
        #         if hasattr(v, "shape"):
        #             print(f"  {k}: {tuple(v.shape)}")
        #         else:
        #             print(f"  {k}: type={type(v)}")
        return default_collate(batch)

class ReduceListDataset(ListDataset):

    @lru_cache(maxsize=None)
    def parse_feat(self, feat_key):
        """
        解析并返回特定特征的内容。假设它的数据存储结构与 PairDataset 相似。
        """
        data = self.lmdb_list[self.keys2lmdbidx[feat_key]][feat_key]
        # 假设数据结构为类似 [s_inputs_p, s_inputs_m, s_p, s_m, z_pm] 
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
        # for i, item in enumerate(batch):
        #     print(f"[Sample {i}]")
        #     for k, v in item.items():
        #         if hasattr(v, "shape"):
        #             print(f"  {k}: {tuple(v.shape)}")
        #         else:
        #             print(f"  {k}: type={type(v)}")
        return default_collate(batch)

class HYPReduceListDataset(ListDataset):

    @lru_cache(maxsize=None)
    def parse_feat(self, feat_key):
        """
        解析并返回特定特征的内容。假设它的数据存储结构与 PairDataset 相似。
        """
        data1 = self.lmdb_list[self.keys2lmdbidx[feat_key]][feat_key]
        protein_key = feat_key.split("_")[0]
        data2 = self.lmdb_list[-1][protein_key]
        # 假设数据结构为类似 [s_inputs_p, s_inputs_m, s_p, s_m, z_pm] 
        s_inputs_p, s_inputs_m, s_p, s_m, z_pm = data1
        s_inputs_p1,prot_s = data2
        res = {
            "s_inputs_p": s_inputs_p.float(),
            "s_inputs_m": s_inputs_m.float(),
            "s_p": s_p.float(),
            "s_m": s_m.float(),
            "z_pm": z_pm.float(),
            "prot_s": prot_s.float(),
        }
        return res

    def collate_fn(self, batch):
        batch = [item for item in batch if item is not None]
        # for i, item in enumerate(batch):
        #     print(f"[Sample {i}]")
        #     for k, v in item.items():
        #         if hasattr(v, "shape"):
        #             print(f"  {k}: {tuple(v.shape)}")
        #         else:
        #             print(f"  {k}: type={type(v)}")
        return default_collate(batch)

        # Debug: 打印每个样本的 shape
        # for item in batch:
        #     print(f"Item size: {item.size()}")

        # for i, item in enumerate(batch):
        #     print(f"[Sample {i}]")
        #     for k, v in item.items():
        #         if hasattr(v, "shape"):
        #             print(f"  {k}: {tuple(v.shape)}")
        #         else:
        #             print(f"  {k}: type={type(v)}")

        return default_collate(batch)

    # def collate_fn(self, batch):
    #     """
    #     自定义批量处理函数，移除批量中的 None 项，并使用默认的 collate_fn
    #     """
    #     batch = [item for item in batch if item is not None]
    #     return default_collate(batch)
    # def collate_fn(self,batch):
    #     out = {}
    #     for key in batch[0].keys():
    #         if key == "pm_list":
    #             # 针对 pm_list 做递归 collate
    #             # pm_list 是一个 list（长度不固定），里面每个元素可能是 dict
    #             out[key] = [default_collate(sub_batch) 
    #                         for sub_batch in zip(*[b[key] for b in batch])]
    #         else:
    #             # 其他字段直接收集成 list，不做 collate
    #             out[key] = [b[key] for b in batch]
    #     return out
    
class TripletDataset(DatasetBase):
    def check_keys(self):
        missing_keys = set()
        for key in self.meta_keys:
            miss_key = key not in self.keys2lmdbidx
            if miss_key:
                missing_keys.add(key)

        logger.info(f"total samples: {len(self.meta_keys)}")
        logger.info(f"missing samples: {len(missing_keys)}")

    def parse_feat(self, feat_key, reverse_m1m2=False):
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


class FullTripletDataset(TripletDataset):
    def parse_feat(self, feat_key, reverse_m1m2=False):
        data = self.lmdb_list[self.keys2lmdbidx[feat_key]][feat_key]
        s_inputs, s, (z_pm1, z_pm2, z_m1m2, z_m1p, z_m2p, z_m2m1) = data

        len_p, len_m1, len_m2 = z_pm1.shape[0], z_pm1.shape[1], z_pm2.shape[1]
        assert (
            len_p + len_m1 + len_m2 == s.shape[0]
        ), f"length mismatch: {len_p} + {len_m1} + {len_m2} != {s.shape[0]} for {feat_key}"

        # s_inputs_p = s_inputs[:len_p]
        # s_inputs_m1 = s_inputs[len_p : len_p + len_m1]
        # s_inputs_m2 = s_inputs[len_p + len_m1 :]
        # s_p = s[:len_p]
        # s_m1 = s[len_p : len_p + len_m1]
        # s_m2 = s[len_p + len_m1 :]

        if reverse_m1m2:
            res = {
                # "s_inputs_p": s_inputs_p.float(),
                # "s_inputs_m1": s_inputs_m2.float(),
                # "s_inputs_m2": s_inputs_m1.float(),
                # "s_p": s_p.float(),
                # "s_m1": s_m2.float(),
                # "s_m2": s_m1.float(),
                "z_pm1": z_pm2.float(),
                "z_pm2": z_pm1.float(),
                # "z_m1m2": z_m1m2.float(),
                "mask_pm1": torch.ones(len_p, len_m2, dtype=torch.bool),
                "mask_pm2": torch.ones(len_p, len_m1, dtype=torch.bool),
                "len_p": len_p,
                "len_m1": len_m2,
                "len_m2": len_m1,
            }
        else:
            res = {
                # "s_inputs_p": s_inputs_p.float(),
                # "s_inputs_m1": s_inputs_m1.float(),
                # "s_inputs_m2": s_inputs_m2.float(),
                # "s_p": s_p.float(),
                # "s_m1": s_m1.float(),
                # "s_m2": s_m2.float(),
                "z_pm1": z_pm1.float(),
                "z_pm2": z_pm2.float(),
                # "z_m1m2": z_m1m2.float(),
                "mask_pm1": torch.ones(len_p, len_m1, dtype=torch.bool),
                "mask_pm2": torch.ones(len_p, len_m2, dtype=torch.bool),
                "len_p": len_p,
                "len_m1": len_m1,
                "len_m2": len_m2,
            }
        return res

    def collate_fn(self, batch):
        valid_batch = [b for b in batch if b is not None]

        def process_pm(pm_list):
            bz = len(pm_list)

            lens_p = [pm["len_p"] for pm in pm_list]
            lens_m1 = [pm["len_m1"] for pm in pm_list]
            lens_m2 = [pm["len_m2"] for pm in pm_list]
            z_pm1 = [pm["z_pm1"] for pm in pm_list]
            z_pm2 = [pm["z_pm2"] for pm in pm_list]
            masks_pm1 = [pm["mask_pm1"] for pm in pm_list]
            masks_pm2 = [pm["mask_pm2"] for pm in pm_list]

            max_p = max(lens_p)
            max_m1 = max(lens_m1)
            max_m2 = max(lens_m2)

            padded_z_pm1 = torch.stack(
                [
                    F.pad(
                        mat,
                        (0, 0, 0, max_m1 - mat.size(1), 0, max_p - mat.size(0)),
                        value=0,
                    )
                    for mat in z_pm1
                ]
            )
            padded_z_pm2 = torch.stack(
                [
                    F.pad(
                        mat,
                        (0, 0, 0, max_m2 - mat.size(1), 0, max_p - mat.size(0)),
                        value=0,
                    )
                    for mat in z_pm2
                ]
            )
            padded_masks_pm1 = torch.stack(
                [
                    F.pad(
                        mat,
                        (0, max_m1 - mat.size(1), 0, max_p - mat.size(0)),
                        value=False,
                    )
                    for mat in masks_pm1
                ]
            )
            padded_masks_pm2 = torch.stack(
                [
                    F.pad(
                        mat,
                        (0, max_m2 - mat.size(1), 0, max_p - mat.size(0)),
                        value=False,
                    )
                    for mat in masks_pm2
                ]
            )

            return {
                "z_pm1": padded_z_pm1,
                "z_pm2": padded_z_pm2,
                "mask_pm1": padded_masks_pm1,
                "mask_pm2": padded_masks_pm2,
                "len_p": torch.tensor(lens_p),
                "len_m1": torch.tensor(lens_m1),
                "len_m2": torch.tensor(lens_m2),
            }

        res = process_pm(valid_batch)
        labels = torch.tensor(
            [item["label"] for item in valid_batch], dtype=torch.float
        )
        hards = torch.tensor(
            [item["hard"] for item in valid_batch], dtype=torch.bool
        )
        idxs = torch.tensor(
            [item["idx"] for item in valid_batch], dtype=torch.long
        )

        res.update({
            "label": labels,
            "hard": hards,
            "idx": idxs,
        })
        return res
    

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

        assert (
            len_p + len_m1 + len_m2 == z_pm1.shape[0],
            f"len sum: {len_p + len_m1 + len_m2} != z_pm1.shape[0]: "
            f"{z_pm1.shape[0]} for {feat_key}",
        )

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
        #args = self.hparams.dataset_args[split]

        # 加载 pairwise 和 listwise 数据集
        #pairwise_dataset = instantiate(args["pairwise"])
        listwise_dataset = instantiate(args["listwise"])

        dataset = instantiate(args)
        dataloader = DataLoader(
            listwise_dataset,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            #collate_fn=dataset.collate_fn,
            collate_fn=listwise_dataset.collate_fn,
            shuffle=split == "train",
            persistent_workers=self.hparams.num_workers > 0,
            prefetch_factor=self.hparams.prefetch_factor,
        )
        return dataloader

    def train_dataloader(self):
        return self._dataloader("train")

    def val_dataloader(self):
       return self._dataloader("val")

# class FullReduceListDataset(ListDataset):
    
#     @lru_cache(maxsize=None)
#     def parse_feat(self, feat_key):
#         # 从 LMDB 中提取多个配体的数据
#         data = self.lmdb_list[self.keys2lmdbidx[feat_key]][feat_key]
#         s_inputs, s, (z_pm, z_mp) = data
#         len_p, len_m, _ = z_pm.shape
#         assert len_p + len_m == s.shape[0], f"length mismatch: {len_p} + {len_m} != {s.shape[0]} for {feat_key}"
        
#         # 计算每个配体的特征（可能是一个列表）
#         s_inputs_p = s_inputs[:len_p].mean(dim=0)  # 假设 s_inputs 对应的是配体的特征
#         s_inputs_m = s_inputs[len_p:].mean(dim=0)
        
#         s_p = s[:len_p].mean(dim=0)  # 这里 s 是配体的相关特征，可能需要进行处理
#         s_m = s[len_p:].mean(dim=0)
        
#         z_pm = z_pm.mean(dim=(0, 1))  # 假设这里是交互特征的均值
#         z_mp = z_mp.mean(dim=(0, 1))
        
#         res = {
#             "s_inputs_p": s_inputs_p.float(),
#             "s_inputs_m": s_inputs_m.float(),
#             "s_p": s_p.float(),
#             "s_m": s_m.float(),
#             "z_pm": z_pm.float(),
#             "z_mp": z_mp.float(),
#         }
#         return res

#     def collate_fn(self, batch):
#         """
#         处理 batch 数据，使其可以直接输入模型。
#         假设每个 batch 里的数据都是多个配体的列表，进行合并和处理。
#         """
#         batch = [item for item in batch if item is not None]
        
#         # 合并和整理 batch 数据
#         listwise_data = {
#             "s_inputs_p": torch.stack([item["s_inputs_p"] for item in batch]),
#             "s_inputs_m": torch.stack([item["s_inputs_m"] for item in batch]),
#             "s_p": torch.stack([item["s_p"] for item in batch]),
#             "s_m": torch.stack([item["s_m"] for item in batch]),
#             "z_pm": torch.stack([item["z_pm"] for item in batch]),
#             "z_mp": torch.stack([item["z_mp"] for item in batch]),
#         }
        
#         return default_collate([listwise_data])

#     def __getitem__(self, idx):
#         """
#         返回一个样本数据
#         """
#         feat_key = self.keys[idx]  # 获取当前样本的 feat_key
#         return self.parse_feat(feat_key)
    
#     def __len__(self):
#         return len(self.keys)

class FullReduceListDataset(ListDataset):
    @lru_cache(maxsize=None)
    def parse_feat(self, feat_key):
        # 保持你原来的实现
        data = self.lmdb_list[self.keys2lmdbidx[feat_key]][feat_key]
        #print(f"DEBUG parse_feat: feat_key={feat_key}, data type={type(data)}, len={len(data) if hasattr(data, '__len__') else 'NA'}, data keys/type if dict: {list(data.keys()) if isinstance(data, dict) else 'NA'}")
        s_inputs, s, (z_pm, z_mp) = data
        len_p, len_m, _ = z_pm.shape
        assert len_p + len_m == s.shape[0], f"length mismatch: {len_p} + {len_m} != {s.shape[0]} for {feat_key}"

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

    def __len__(self):
        # 一定要用 meta_keys（DatasetBase 在 init 中设置）
        return len(self.meta_keys)

    def __getitem__(self, idx):
        """
        处理 listwise meta key: uniprot/lig1/lig2/...
        对每个 ligand 调用 parse_feat(uniprot/lig_i)，然后把所有 ligand 的特征合并成一个样本级 tensor dict。
        """
        meta_key = self.meta_keys[idx]
        parts = meta_key.split("/")
        if len(parts) < 2:
            logger.warning(f"FullReduceListDataset.__getitem__ unexpected key: {meta_key}")
            return None

        uniprot_id = parts[0]
        ligand_ids = parts[1:]

        per_lig_feats = []
        for lig in ligand_ids:
            feat_key = f"{uniprot_id}/{lig}"
            if feat_key not in self.keys2lmdbidx:
                logger.warning(f"FullReduceListDataset missing feat_key: {feat_key} (meta={meta_key})")
                continue
            try:
                per_lig_feats.append(self.parse_feat(feat_key))
            except Exception:
                logger.exception(f"parse_feat failed for {feat_key}")
                # 跳过这个 ligand 或者 append None；这里选择跳过
                continue

        if len(per_lig_feats) == 0:
            logger.warning(f"All ligands missing or failed for meta {meta_key}")
            return None

        # 合并：对每个 tensor key 做 stack -> mean (得到固定形状)
        combined = {}
        keys0 = per_lig_feats[0].keys()
        for k in keys0:
            vals = [f[k] for f in per_lig_feats if f is not None]
            # 若 vals 为空逻辑已在上面处理
            combined[k] = torch.stack(vals, dim=0).mean(dim=0)

        combined["idx"] = idx
        combined["meta_key"] = meta_key
        combined["ligand_count"] = len(per_lig_feats)
        return combined

    def collate_fn(self, batch):
        # batch: list of items returned by __getitem__
        batch = [item for item in batch if item is not None]
        if len(batch) == 0:
            return None

        # keys except some meta fields
        tensor_keys = [k for k in batch[0].keys() if k not in ("idx", "meta_key", "ligand_count")]
        listwise_data = {}
        for k in tensor_keys:
            listwise_data[k] = torch.stack([item[k] for item in batch], dim=0)  # batch dim at 0

        # 收集元信息
        listwise_data["idx"] = torch.tensor([item["idx"] for item in batch], dtype=torch.long)
        listwise_data["meta_key"] = [item["meta_key"] for item in batch]
        listwise_data["ligand_count"] = torch.tensor([item["ligand_count"] for item in batch], dtype=torch.long)

        return listwise_data

# class NewDataModule(LightningDataModule):
#     def __init__(
#         self,
#         num_workers: int = 0,
#         pin_memory: bool = False,
#         batch_size: int = 1,
#         prefetch_factor: int = 4,
#         dataset_args: Optional[Dict[str, Any]] = None,
#         **kwargs,
#     ):
#         super().__init__()
#         self.save_hyperparameters(logger=False)

#     def _dataloader(self, split, dataset_type):
#         """
#         根据 split 和 dataset_type 加载对应的数据集。
#         :param split: "train" 或 "val"
#         :param dataset_type: "pairwise" 或 "listwise"
#         """
#         args = self.hparams.dataset_args[split]
#         dataset = instantiate(args[dataset_type])
        
#         dataloader = DataLoader(
#             dataset,
#             batch_size=self.hparams.batch_size,
#             num_workers=self.hparams.num_workers,
#             pin_memory=self.hparams.pin_memory,
#             collate_fn=dataset.collate_fn,
#             shuffle=split == "train",
#             persistent_workers=self.hparams.num_workers > 0,
#             prefetch_factor=self.hparams.prefetch_factor,
#         )
#         return dataloader

#     def train_dataloader(self):
#         # 同时加载 pairwise 和 listwise 数据集
#         pairwise_dataloader = self._dataloader("train", "pairwise")
#         listwise_dataloader = self._dataloader("train", "listwise")
        
#         # 使用 zip() 来同时迭代两个数据加载器
#         return zip(pairwise_dataloader, listwise_dataloader)

#     def val_dataloader(self):
#         # 同时加载 pairwise 和 listwise 数据集
#         pairwise_dataloader = self._dataloader("val", "pairwise")
#         listwise_dataloader = self._dataloader("val", "listwise")
        
#         # 使用 zip() 来同时迭代两个数据加载器
#         return zip(pairwise_dataloader, listwise_dataloader)

from itertools import zip_longest

class NewDataModule(LightningDataModule):
    def __init__(self, num_workers: int = 0, pin_memory: bool = False, batch_size: int = 1, prefetch_factor: int = 4, dataset_args: Optional[Dict[str, Any]] = None, **kwargs):
        super().__init__()
        self.save_hyperparameters(logger=False)

    def _dataloader(self, split):
        args = self.hparams.dataset_args[split]

        # 加载 pairwise 和 listwise 数据集
        pairwise_dataset = instantiate(args["pairwise"])
        listwise_dataset = instantiate(args["listwise"])

        pairwise_dataloader = DataLoader(
            pairwise_dataset,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            collate_fn=pairwise_dataset.collate_fn,
            shuffle=split == "train",
            persistent_workers=self.hparams.num_workers > 0,
            prefetch_factor=self.hparams.prefetch_factor,
        )
        
        listwise_dataloader = DataLoader(
            listwise_dataset,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            collate_fn=listwise_dataset.collate_fn,
            shuffle=split == "train",
            persistent_workers=self.hparams.num_workers > 0,
            prefetch_factor=self.hparams.prefetch_factor,
        )

        return pairwise_dataloader, listwise_dataloader

    def train_dataloader(self):
        pairwise_loader, listwise_loader = self._dataloader("train")
        return zip_longest(pairwise_loader, listwise_loader)

    # def val_dataloader(self):
    #     pairwise_loader, listwise_loader = self._dataloader("val")
    #     return zip_longest(pairwise_loader, listwise_loader)

    def val_dataloader(self):
    # 复用训练数据加载器，但不做验证
        pairwise_loader, listwise_loader = self._dataloader("train")
        return zip_longest(pairwise_loader, listwise_loader)



import logging
from functools import lru_cache
from torch.utils.data.dataloader import default_collate
import torch

# 设置日志记录
logging.basicConfig(level=logging.DEBUG)  # 使用 DEBUG 级别输出更多日志
logger = logging.getLogger(__name__)

class FullReducePairDatasettest(PairDataset):
    @lru_cache(maxsize=None)
    def parse_feat(self, feat_key):
        # 记录方法入口
        logger.debug(f"Entering parse_feat for feature: {feat_key}")

        if feat_key not in self.keys2lmdbidx:
            logger.error(f"feat_key '{feat_key}' not found in keys2lmdbidx.")
            raise KeyError(f"feat_key '{feat_key}' not found in keys2lmdbidx")

        # 尝试加载数据
        try:
            data = self.lmdb_list[self.keys2lmdbidx[feat_key]][feat_key]
        except KeyError as e:
            logger.error(f"Error retrieving data for feat_key '{feat_key}': {e}")
            raise e
        
        if data is None:
            logger.error(f"Data for feature key '{feat_key}' is None.")
            raise ValueError(f"Data for feature key '{feat_key}' is None.")
        
        # 记录数据的形状
        s_inputs, s, (z_pm, z_mp) = data
        logger.debug(f"Data for {feat_key}: s_inputs shape: {s_inputs.shape}, s shape: {s.shape}, z_pm shape: {z_pm.shape}, z_mp shape: {z_mp.shape}")
        
        len_p, len_m, _ = z_pm.shape
        assert len_p + len_m == s.shape[0], f"length mismatch: {len_p} + {len_m} != {s.shape[0]} for {feat_key}"
        
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

        # 记录方法返回
        logger.debug(f"Returning parsed feature: {feat_key}")
        return res

    def collate_fn(self, batch):
        logger.debug(f"Collating batch with {len(batch)} items.")

        # 过滤掉 None 类型的数据
        batch = [item for item in batch if item is not None]

        # 如果过滤后为空，发出警告
        if len(batch) == 0:
            logger.warning("Collated batch is empty after filtering None values.")
        
        return default_collate(batch)
