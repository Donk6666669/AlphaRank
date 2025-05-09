import time
import random
import os
import json
from copy import deepcopy

import torch
import pandas as pd
from torch.optim import AdamW
from protenix.config import parse_configs
from protenix.data.msa_featurizer import tokenize_msa
from protenix.data.tokenizer import AtomArrayTokenizer, TokenArray
from protenix.utils.torch_utils import dict_to_tensor
from protenix.utils.lmdb import LMDBDataset
from protenix.utils.torch_utils import to_device
from tqdm import tqdm
from gpustat import new_query

from protenix.data.screen_dataset import (
    FeatureCompressor,
    ComplexFeatureDataset,
    ComplexFeatureRandomPairDataset,
)
from protenix.model.protenis import ProtenisP, ProtenisPCrossIndependentRanker
from protenix.model.rank_model import IRSimpleClassifier
from protenix.criterion.loss import RankNetLoss

from configs.configs_base import configs as configs_base
from configs.configs_data import data_configs
from configs.configs_inference import inference_configs


def train():
    configs_base["use_deepspeed_evo_attention"] = (
        os.environ.get("USE_DEEPSPEED_EVO_ATTTENTION", False) == "true"
    )
    configs_base["model"]["N_cycle"] = 10
    configs_base["sample_diffusion"]["N_sample"] = 5
    configs_base["sample_diffusion"]["N_step"] = 200
    configs = {**configs_base, **{"data": data_configs}, **inference_configs}
    configs = parse_configs(
        configs=configs,
        fill_required_with_null=True,
    )

    torch.cuda.set_device("cuda:3")
    device = torch.device("cuda:3")

    # dataset = ComplexFeatureDataset(
    #     "/data/rerank/protenix/chembl_bdb/chembl_bdb.lmdb"
    # )
    dataset = ComplexFeatureRandomPairDataset(
        "/data/rerank/protenix/chembl_bdb/chembl_bdb.lmdb"
    )

    cache_data = {}
    for i in tqdm(range(len(dataset)), ncols=80):
        pairs = dataset.assay_pairs[dataset.assay_pair_keys[i]]
        key_a, key_b = random.choice(pairs)
        meta_a, meta_b = dataset.lmdb[key_a], dataset.lmdb[key_b]
        pocket_len = len(meta_a["sequences"][0]["proteinChain"]["sequence"])
        total_len = pocket_len
        key = str(int(int(total_len // 25) * 25))
        if key not in cache_data:
            cache_data[key] = i

    with open(
        "/data/rerank/protenix/chembl_bdb/chembl_bdb_cache_len.json", "w"
    ) as f:
        json.dump(cache_data, f)

    with open(
        "/data/rerank/protenix/chembl_bdb/chembl_bdb_cache_len.json", "r"
    ) as f:
        cache_data = json.load(f)

    # model = ProtenisP(configs).cuda()
    model = ProtenisPCrossIndependentRanker(
        configs, IRSimpleClassifier()
    ).cuda()
    loss_func = RankNetLoss()
    optimizer = AdamW(model.parameters(), lr=1e-4, weight_decay=1e-2)

    statistics = {}
    for key, idx in tqdm(
        sorted(cache_data.items(), key=lambda x: int(x[0])), ncols=80
    ):

        sample = dataset[idx]

        input_feature_dict = to_device(sample["input_feature_dicts"][0], device)
        input_feature_dict_ = to_device(
            sample["input_feature_dicts"][1], device
        )

        pocket_len = (input_feature_dict["entity_id"] == 0).sum().item()
        mol_len = (input_feature_dict["entity_id"] == 1).sum().item()

        pocket_len_ = (input_feature_dict_["entity_id"] == 0).sum().item()
        mol_len_ = (input_feature_dict_["entity_id"] == 1).sum().item()

        start_time = time.time()
        optimizer.zero_grad()
        pred_dict, log_dict = model([input_feature_dict, input_feature_dict_])
        loss = loss_func(
            pred_dict["logits"][0],
            pred_dict["logits"][1],
            to_device(sample["pair_label"], device),
        )
        forward_end_time = time.time()
        loss["loss"].backward()
        optimizer.step()
        end_time = time.time()
        forward_use_time = forward_end_time - start_time
        backward_use_time = end_time - forward_end_time
        total_use_time = end_time - start_time

        gpu_list = new_query().gpus
        gpu_memory = gpu_list[3].entry["memory.used"]

        data = {
            "len_a": (pocket_len, mol_len),
            "len_b": (pocket_len_, mol_len_),
            "forward_use_time": forward_use_time,
            "backward_use_time": backward_use_time,
            "total_use_time": total_use_time,
            "gpu_memory": gpu_memory,
        }
        statistics[key] = data

        with open(
            "/data/rerank/protenix/chembl_bdb/protenis_statistics.json", "w"
        ) as f:
            json.dump(statistics, f)

        del pred_dict, log_dict, input_feature_dict, input_feature_dict_
        torch.cuda.empty_cache()

    # model = ProtenisPCrossIndependentRanker(
    #     configs, IRSimpleClassifier()
    # ).cuda()
    # dataset = ComplexFeatureRandomPairDataset(
    #     "/data/rerank/protenix/chembl_bdb/chembl_bdb.lmdb"
    # )
    # batch = []
    # batch_size = 16
    # loss_func = RankNetLoss()

    # # optimizer = torch.optim.AdamW(
    # #     model.parameters(), lr=1e-4, weight_decay=1e-2
    # # )

    # for i in tqdm(range(len(dataset)), ncols=80):
    #     # optimizer.zero_grad()

    #     print("sample")
    #     sample = dataset[i]
    #     print("cuda")
    #     input_feature_dicts = to_device(sample["input_feature_dicts"], device)

    #     print("model")
    #     pred_dict, log_dict = model(input_feature_dicts)
    #     # print("loss")
    #     # loss = loss_func(
    #     #     pred_dict["logits"][0],
    #     #     pred_dict["logits"][1],
    #     #     to_device(sample["pair_label"], device),
    #     # )
    #     # print("loss complete")
    #     # loss["loss"].backward()

    #     # optimizer.step()


if __name__ == "__main__":
    train()
