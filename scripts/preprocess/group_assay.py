import numpy as np
from tqdm import tqdm
from protenix.utils.lmdb import LMDBDataset
from protenix.data.screen_dataset import get_discrepant_pairs


def records2assay(lmdb_path):
    lmdb = LMDBDataset(lmdb_path, readonly=False)
    record_keys = lmdb.get_split("clean_keys_has_feature")
    assay = {}
    for key in tqdm(record_keys, ncols=80):
        record = lmdb[key]
        assay_group = record["assay_group"]
        if assay_group not in assay:
            assay[assay_group] = []
        assay[assay_group].append(
            {"name": record["name"], "activity": record["activity"]}
        )
    lmdb["assay_groups"] = assay


def assay2pairs(lmdb_path):
    lmdb = LMDBDataset(lmdb_path, readonly=False)
    assay_groups = lmdb["assay_groups"]
    pairs = {}
    for group, records in tqdm(assay_groups.items(), ncols=80):
        p = get_discrepant_pairs(records)
        if len(p) > 0:
            pairs[group] = p
    lmdb["assay_pairs"] = pairs
    lmdb.close()


def assay2_small_gap_pairs(lmdb_path, min_threshold=0.5, max_threshold=1.0):
    lmdb = LMDBDataset(lmdb_path, readonly=False)
    assay_groups = lmdb["assay_groups"]
    pairs = {}
    for group, records in tqdm(assay_groups.items(), ncols=80):
        p = get_discrepant_pairs(
            records, min_threshold=min_threshold, max_threshold=max_threshold
        )
        if len(p) > 0:
            pairs[group] = p
    lmdb[f"assay_pairs_{min_threshold}_{max_threshold}"] = pairs
    lmdb.close()


if __name__ == "__main__":
    # lmdb_path = "/data/rerank/protenix/chembl_bdb/chembl_bdb.lmdb"
    # # records2assay(lmdb_path)
    # assay2pairs(lmdb_path)

    lmdb_path = "/data/rerank/protenix/chembl_bdb/chembl_bdb.lmdb"
    assay2_small_gap_pairs(lmdb_path, min_threshold=0.5, max_threshold=1.0)
    assay2_small_gap_pairs(lmdb_path, min_threshold=0.25, max_threshold=0.5)
