import hashlib
import json
import pandas as pd
from tqdm import tqdm
from pathlib import Path

from tqdm import tqdm
from rdkit import Chem
from protenix.utils.lmdb import LMDBDataset


dataset = LMDBDataset(
    "/data/rerank/protenix/chembl_bdb/chembl_bdb.lmdb", readonly=False
)
keys = dataset.get_split("chembl_bdb")
batch_size = 1000

pbar = tqdm(total=len(keys), ncols=80)
while len(keys) > 0:
    batch_keys = keys[:batch_size]
    batch_data = dataset.get_data(batch_keys)
    for key, data in batch_data.items():
        prot_seq = data["sequences"][0]["proteinChain"]["sequence"]
        prot_len = len(prot_seq)

        mol_smi = data["sequences"][1]["ligand"]["ligand"]
        mol_len = 0
        try:
            mol = Chem.MolFromSmiles(mol_smi)
            if mol is not None:
                mol_len = mol.GetNumAtoms()
        except Exception:
            pass
        data["len"] = {
            "protein": prot_len,
            "ligand": mol_len,
        }
    dataset.write_data(batch_data)
    keys = keys[batch_size:]
    pbar.update(len(batch_keys))
dataset.close()
