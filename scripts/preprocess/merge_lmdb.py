from tqdm import tqdm

from protenix.utils.lmdb import LMDBDataset
from protenix.utils.logger import get_logger

logger = get_logger(__name__)


def merge_lmdbs(
    lmdb_mapping: dict[list[tuple]],
    target_lmdb_path: str,
    write_batch_size=1000,
):
    unique_splits = set()
    for lmdb_path, split_data in tqdm(lmdb_mapping.items(), ncols=80):
        logger.info(f"Processing LMDB: {lmdb_path}")
        source_lmdb = LMDBDataset(lmdb_path, readonly=True)
        target_lmdb = LMDBDataset(target_lmdb_path, readonly=False)
        for source_split, target_split in tqdm(split_data, ncols=80):
            logger.info(
                f"Processing split: {source_split} -> {target_split}"
            )
            unique_splits.add(target_split)
            source_split_keys = source_lmdb.get_split(source_split)
            pbar = tqdm(
                total=len(source_split_keys),
                ncols=80,
                leave=False,
            )
            if source_split_keys is not None or len(source_split_keys) > 0:
                while len(source_split_keys) > 0:
                    batch_keys = source_split_keys[:write_batch_size]
                    batch_data = source_lmdb.get_data(batch_keys, ori=True)
                    target_lmdb.write_data(batch_data, ori=True)
                    target_lmdb.set_split(
                        target_split, batch_keys, append=True
                    )
                    pbar.update(len(batch_keys))

                    source_split_keys = source_split_keys[write_batch_size:]

        source_lmdb.close()
        target_lmdb.close()


def dedup_lmdb(lmdb_mapping: dict[list[tuple]], target_lmdb_path: str):
    unique_splits = set()
    for _, split_data in lmdb_mapping.items():
        for _, target_split in split_data:
            unique_splits.add(target_split)

    target_lmdb = LMDBDataset(target_lmdb_path, readonly=False)
    for split in sorted(unique_splits):
        keys = target_lmdb.get_split(split)
        target_lmdb.set_split(split, sorted(set(keys)))
        logger.info(
            f"- {split}: {len(keys)}"
        )
    target_lmdb.close()


if __name__ == "__main__":
    root = "/data/rerank/protenix/chembl_bdb/feat_parts"
    lmdb_mapping = {
        f"{root}/chembl_bdb_20250410.lmdb": [
            ("chembl_bdb", "chembl_bdb"),
        ],
        f"{root}/chembl_bdb_13.lmdb": [
            ("feature", "feature"),
        ],
        f"{root}/chembl_bdb_12.lmdb": [
            ("feature", "feature"),
        ],
        f"{root}/chembl_bdb_19.lmdb": [
            ("feature", "feature"),
        ],
        f"{root}/chembl_bdb_46.lmdb": [
            ("feature", "feature"),
        ],
        f"{root}/chembl_bdb_47.lmdb": [
            ("feature", "feature"),
        ],
        f"{root}/chembl_bdb_48.lmdb": [
            ("feature", "feature"),
        ],
        f"{root}/msa_feat_filtered.lmdb": [
            ("msa_filtered", "msa_filtered"),
        ],
        f"{root}/msa_feat_full.lmdb": [
            ("msa", "msa_full"),
        ],
        f"{root}/esm_feat.lmdb": [
            ("esm", "esm"),
        ],
    }
    target_lmdb_path = "/data/rerank/protenix/chembl_bdb/chembl_bdb.lmdb"
    # merge_lmdbs(lmdb_mapping, target_lmdb_path)
    dedup_lmdb(lmdb_mapping, target_lmdb_path)