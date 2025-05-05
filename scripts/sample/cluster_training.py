import json
from pathlib import Path

from protenix.utils.proteintool import (
    get_fasta_by_uniprotids,
    mmseqs2_cluster_fasta,
)


root = Path(__file__).parent
cluster_dir = root / "cluster30"
cluster_dir.mkdir(parents=True, exist_ok=True)


split_file = (
    root.parent / "split" / "splits" / "chembl_bdb.dude_litpcba_fep.30.json"
)
with open(split_file) as f:
    full_data_split_30 = json.load(f)
chembl_bdb_fasta = root.parent / "split" / "raw_data" / "chembl_bdb.fasta"

train_uniprotids = full_data_split_30["train"]
get_fasta_by_uniprotids(
    train_uniprotids, cluster_dir / "train30.fasta", ref_file=chembl_bdb_fasta
)
mmseqs2_cluster_fasta(cluster_dir / "train30.fasta", identity=30)

val_uniprotids = full_data_split_30["valid"]
get_fasta_by_uniprotids(
    val_uniprotids, cluster_dir / "valid30.fasta", ref_file=chembl_bdb_fasta
)
mmseqs2_cluster_fasta(cluster_dir / "valid30.fasta", identity=30)

print(f"train uniprotids: {len(train_uniprotids)}")
print(f"valid uniprotids: {len(val_uniprotids)}")