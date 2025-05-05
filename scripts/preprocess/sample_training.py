import torch
import json
import numpy as np
import pandas as pd
from tqdm import tqdm
from protenix.utils.lmdb import LMDBDataset
from protenix.utils.plottool import init_plot_settings

import matplotlib.pyplot as plt



lmdb = LMDBDataset("/data/rerank/protenix/chembl_bdb/chembl_bdb.lmdb")
assay_groups = lmdb["assay_groups"]
assay_pairs = lmdb["assay_pairs"]
len(assay_pairs)


print("assay whose protein len < 1024")
assay_below_1024 = {}
for name, assay in assay_groups.items():
    if len(assay) > 0:
        data = lmdb[assay[0]['name']]
        if data["len"]["protein"] < 1024:
            assay_below_1024[name] = assay
print(f"before filter: {len(assay_groups)}")
print(f"after filter: {len(assay_below_1024)}")


print("30 identity filter")
with open("../scripts/split/splits/chembl_bdb.dude_litpcba_fep.30.json") as f:
    full_data_split_30 = json.load(f)
assay_30_train = {}
assay_30_valid = {}
for name, assay in assay_below_1024.items():
    uniprot_id = name.split("_")[-2]
    if uniprot_id in full_data_split_30["train"]:
        assay_30_train[name] = assay
    elif uniprot_id in full_data_split_30["valid"]:
        assay_30_valid[name] = assay
    else:
        print(f"uniprot id {uniprot_id} not in train or valid")
print(f"train: {len(assay_30_train)}")
print(f"valid: {len(assay_30_valid)}")

assay_pairs_30_train = {}
assay_pairs_30_valid = {}
assay_pairs_filtered = {}
for key in assay_30_train:
    if key in assay_pairs:
        assay_pairs_30_train[key] = assay_pairs[key]
        assay_pairs_filtered[key] = assay_pairs[key]
for key in assay_30_valid:
    if key in assay_pairs:
        assay_pairs_30_valid[key] = assay_pairs[key]
        assay_pairs_filtered[key] = assay_pairs[key]