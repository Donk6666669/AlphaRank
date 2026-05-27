import os
import lmdb
import pickle
import json
import argparse
import numpy as np
from tqdm import tqdm
from protenix.utils.lmdb import LMDBDataset
import zstandard as zstd
import torch  # 需要用来判断 dtype

# ======== 命令行参数（支持直接运行或脚本调用）========
parser = argparse.ArgumentParser(description="给LMDB添加名字")
parser.add_argument('--old_lmdb', default="/data/rerank/protenix/chembl_bdb/openbind/pair/openbind.lmdb")
parser.add_argument('--json_path', default="/data/rerank/protenix/chembl_bdb/openbind/openbind.json")
parser.add_argument('--new_lmdb', default="/data/rerank/protenix/chembl_bdb/openbind/pair/openbind_111.lmdb")
parser.add_argument('--protein_lmdb', default=None, help='可选：从pair embedding中抽取蛋白embedding并写入该LMDB')
parser.add_argument('--map_size', type=int, default=20_000_000_000)
args = parser.parse_args()

OLD_LMDB = args.old_lmdb
JSON_PATH = args.json_path
NEW_LMDB = args.new_lmdb
PROTEIN_LMDB = args.protein_lmdb
MAP_SIZE = args.map_size
# ==============================

def is_zstd_frame(b: bytes) -> bool:
    # Zstandard 帧魔数：0x28 B5 2F FD
    return len(b) >= 4 and b[:4] == b"\x28\xb5\x2f\xfd"

def to_numpy(x):
    """兼容 torch.Tensor(bfloat16/float16/float32...) 转 numpy"""
    try:
        if isinstance(x, torch.Tensor):
            # 如果是 bfloat16 或 float16，转成 float32
            if x.dtype in (torch.bfloat16, torch.float16):
                x = x.to(torch.float32)
            return x.detach().cpu().numpy()
    except Exception as e:
        print(f"[WARN] tensor 转 numpy 出错：{e}")
        return None

    if isinstance(x, np.ndarray):
        return x

    try:
        return np.asarray(x)
    except Exception:
        return None

# 1) 读 JSON，建立 name 映射（full_name -> (protein, ligand)）
with open(JSON_PATH, "r") as f:
    json_data = json.load(f)

name_map = {}
for entry in json_data:
    full_name = entry["name"]
    p, l = full_name.rsplit("/", 1)  # 只从右拆一次
    name_map[full_name] = (p, l)

# 2) 用 LMDBDataset 打开旧库（只读）
old_ds = LMDBDataset(OLD_LMDB)  # 默认 readonly=True

# 3) 打开旧库的 data/split 子库
old_env = old_ds.env
with old_env.begin(write=False) as txn_old:
    OLD_DATA_DB = old_env.open_db(b"data")
    OLD_SPLIT_DB = old_env.open_db(b"split")

# 4) 创建新库
if os.path.exists(NEW_LMDB):
    print(f"{NEW_LMDB} 已存在，删除后重建...")
    os.system(f"rm -rf {NEW_LMDB}")

new_env = lmdb.open(NEW_LMDB, map_size=MAP_SIZE, max_dbs=2)
NEW_DATA_DB = new_env.open_db(b"data", create=True)
NEW_SPLIT_DB = new_env.open_db(b"split", create=True)

protein_env = None
PROTEIN_DATA_DB = None
PROTEIN_SPLIT_DB = None
if PROTEIN_LMDB:
    if os.path.exists(PROTEIN_LMDB):
        print(f"{PROTEIN_LMDB} 已存在，删除后重建...")
        os.system(f"rm -rf {PROTEIN_LMDB}")
    protein_env = lmdb.open(PROTEIN_LMDB, map_size=MAP_SIZE, max_dbs=2)
    PROTEIN_DATA_DB = protein_env.open_db(b"data", create=True)
    PROTEIN_SPLIT_DB = protein_env.open_db(b"split", create=True)

# 5) 复制 split 子库
with old_env.begin(write=False, db=OLD_SPLIT_DB) as txn_split_old, \
     new_env.begin(write=True, db=NEW_SPLIT_DB) as txn_split_new:

    cur = txn_split_old.cursor()
    cnt = 0
    for k, v in cur:
        txn_split_new.put(k, v)
        cnt += 1
    print(f"已原样复制 split 子库条目数：{cnt}")

# 6) 遍历 data 子库 —— 只处理 reduced_pair split 中的 key
# （同一个 LMDB 可能同时包含 json2feature 的原始特征和 protenix predict 的 embedding，
#  只有 reduced_pair split 列出的才是 embedding，其他跳过）
cctx = zstd.ZstdCompressor(level=3)

with old_env.begin(write=False, db=OLD_SPLIT_DB) as txn_split_ro:
    split_raw = txn_split_ro.get(b"reduced_pair")
    if split_raw is None:
        # 兼容旧格式：直接用 data DB 的全部 key
        print("[INFO] reduced_pair split 不存在，使用全部 data key")
        with old_env.begin(write=False, db=OLD_DATA_DB) as txn_data_old_tmp:
            valid_keys = set(k.decode() for k in txn_data_old_tmp.cursor().iternext(values=False))
    else:
        # split 值格式：zstd压缩的逗号分隔字符串（见 LMDBDataset._smart_encode_list）
        dctx = zstd.ZstdDecompressor()
        split_decompressed = dctx.decompress(split_raw)
        valid_keys = set(split_decompressed.decode().split(","))
        print(f"[INFO] reduced_pair split 包含 {len(valid_keys)} 个 key")

with old_env.begin(write=False, db=OLD_DATA_DB) as txn_data_old:
    all_data_keys = [k.decode() for k in txn_data_old.cursor().iternext(values=False)]
    keys = [k for k in all_data_keys if k in valid_keys]
    print(f"[INFO] data DB 共 {len(all_data_keys)} 个 key，其中 {len(keys)} 个在 reduced_pair split 中")

    # 检测压缩方式
    sample_raw = None
    for k_b in txn_data_old.cursor().iternext(values=False):
        sample_raw = txn_data_old.get(k_b, db=OLD_DATA_DB)
        if sample_raw:
            break
    use_zstd = is_zstd_frame(sample_raw) if sample_raw else True
    print(f"检测到原 data 压缩方式：{'zstd' if use_zstd else 'none'}")

    protein_keys = []
    with new_env.begin(write=True, db=NEW_DATA_DB) as txn_data_new:
        protein_txn = protein_env.begin(write=True, db=PROTEIN_DATA_DB) if protein_env else None
        for key in tqdm(keys, desc="Rewriting data"):
            item = old_ds[key]
            if item is None:
                print(f"[WARN] {key} -> None，跳过")
                continue

            # 转 numpy
            try:
                item_array = tuple(to_numpy(t) for t in item)
                if any(arr is None for arr in item_array):
                    print(f"[WARN] {key} 有元素转 numpy 失败，跳过")
                    continue
            except Exception as e:
                print(f"[WARN] {key} 转 numpy 失败：{e}，跳过")
                continue

            # name 映射（json2feature 会给 key 加 p_ 前缀，需要去掉后再查）
            lookup_key = key[2:] if key.startswith("p_") else key
            if lookup_key not in name_map:
                print(f"[WARN] {key} 不在 JSON，跳过")
                continue
            protein_name, ligand_name = name_map[lookup_key]

            new_item = {
                "name": f"{protein_name},{ligand_name}",
                "emb": item_array,
            }

            payload = pickle.dumps(new_item, protocol=pickle.HIGHEST_PROTOCOL)
            if use_zstd:
                payload = cctx.compress(payload)

            txn_data_new.put(key.encode(), payload)

            if protein_txn is not None and protein_name not in protein_keys:
                if len(item_array) < 3:
                    print(f"[WARN] {key} embedding长度不足，无法抽取protein embedding")
                else:
                    protein_item = (item_array[0], item_array[2])
                    protein_payload = pickle.dumps(protein_item, protocol=pickle.HIGHEST_PROTOCOL)
                    if use_zstd:
                        protein_payload = cctx.compress(protein_payload)
                    protein_txn.put(protein_name.encode(), protein_payload)
                    protein_keys.append(protein_name)
        if protein_txn is not None:
            protein_txn.commit()

    if protein_env is not None:
        split_payload = ",".join(protein_keys).encode()
        if use_zstd:
            split_payload = cctx.compress(split_payload)
        with protein_env.begin(write=True, db=PROTEIN_SPLIT_DB) as txn_protein_split:
            txn_protein_split.put(b"all", split_payload)
        print(f"完成：protein LMDB 写入 -> {PROTEIN_LMDB}（{len(protein_keys)} 个蛋白）")

# 7) 收尾
new_env.sync()
new_env.close()
if protein_env is not None:
    protein_env.sync()
    protein_env.close()
print("完成：新 LMDB 写入 ->", NEW_LMDB)
