import hashlib
import json
import pandas as pd
from tqdm import tqdm
from pathlib import Path

from protenix.utils.lmdb import LMDBDataset
from rdkit import Chem


class FastaParser:
    """Class to parse UniProt FASTA files and provide sequence lookup"""

    def __init__(self, fasta_path):
        self.sequences = {}
        self._parse_fasta(fasta_path)

    def _parse_fasta(self, path):
        """Parse FASTA file and build sequence dictionary"""
        current_id = None
        current_seq = []

        with open(path, "r") as f:
            for line in tqdm(f, desc="Parsing FASTA", ncols=80):
                line = line.strip()
                if line.startswith(">"):
                    # Save previous entry
                    if current_id:
                        self.sequences[current_id] = "".join(current_seq)

                    # Parse header line: >sp|P21266|GSTM3_HUMAN...
                    parts = line[1:].split("|")  # Remove '>' and split
                    if len(parts) >= 2:
                        current_id = parts[1]
                        current_seq = []
                    else:
                        current_id = None
                elif current_id is not None:
                    current_seq.append(line)

            # Add last sequence
            if current_id and current_seq:
                self.sequences[current_id] = "".join(current_seq)

    def get_sequence(self, uniprot_id):
        """Retrieve sequence by UniProt ID"""
        if uniprot_id in self.sequences:
            return self.sequences[uniprot_id]
        else:
            return None


def safe_str(value):
    """Convert values to strings safely"""
    if pd.isna(value):
        return ""
    try:
        return (
            str(float(value)) if isinstance(value, (float, int)) else str(value)
        )
    except:
        return str(value)


def main_process(
    csv_path,
    fasta_path,
    output_protenix_json_path,
    output_lmdb_path,
    msa_dir,
):
    msa_dir = Path(msa_dir)
    fasta_parser = FastaParser(fasta_path)
    lmdb = LMDBDataset(output_lmdb_path, readonly=False)

    # Load data with pandas
    df = pd.read_csv(csv_path, low_memory=False)

    # Initialize counters for statistics
    total_samples = 0
    unique_uniprot_ids = set()
    source_journals = set()
    unique_units = set()

    column_mapping = {
        "Database_Assay_ID": "assay_id",
        "Uniprot_ID": "uniprot_id",
        "Database_Source": "source",
        "Journal": "journal",
        "Patent_ID": "patent",
    }
    df = df.rename(columns=column_mapping)
    df = df.astype(
        {
            "assay_id": "string",
            "uniprot_id": "string",
            "source": "string",
            "Standard_Value": "float",
            "Standard_Units": "string",
            "Standard_Type": "string",
            "Standard_Relation": "string",
            "SMILES": "string",
            "journal": "string",
            "patent": "string",
        }
    )

    # normalize column values
    df["assay_id"] = df["assay_id"].str.strip().str.upper()
    df["uniprot_id"] = df["uniprot_id"].str.strip().str.upper()
    df["Standard_Type"] = df["Standard_Type"].str.strip().str.upper()
    df["uniprot_id"] = df["uniprot_id"].fillna("")
    df["journal"] = df["journal"].fillna("")
    df["patent"] = df["patent"].fillna("")

    # Process data with progress visualization
    assay_groups = {}
    processed_data = {}
    keys = []
    protenix_data = []
    print("\nProcessing data samples:")
    with tqdm(total=len(df), desc="Progress", ncols=80) as pbar:
        for _, row in df.iterrows():
            # Generate unique identifier
            hash_fields = [
                row["assay_id"],
                row["uniprot_id"],
                row["SMILES"],
                row["Standard_Type"],
                row["Standard_Relation"],
                row["Standard_Value"],
                row["Standard_Units"],
                row["patent"],
            ]
            hash_str = "".join(safe_str(f) for f in hash_fields)
            entry_hash = hashlib.md5(hash_str.encode()).hexdigest()
            entry_name = f"{row['uniprot_id']}_{entry_hash}"

            # Create assay group identifier
            group_key = (
                row["assay_id"],
                row["uniprot_id"],
                row["Standard_Type"],
            )
            group_name = f"group_{group_key[0]}_{group_key[1]}_{group_key[2]}"

            # Update assay groups
            if group_name not in assay_groups:
                assay_groups[group_name] = []
            assay_groups[group_name].append(entry_name)

            if row["uniprot_id"] == "":
                continue
            sequence = fasta_parser.get_sequence(row["uniprot_id"])
            if sequence is None:
                continue

            # Build entry structure
            entry = {
                "name": entry_name,
                "assay_id": row["assay_id"],
                "uniprot_id": row["uniprot_id"],
                "source": row["source"],
                "journal": row["journal"],
                "patent": row["patent"],
                "sequences": [
                    {
                        "proteinChain": {
                            "sequence": sequence,
                            "count": 1,
                            "msa": {
                                "precomputed_msa_dir": str(
                                    msa_dir / row["uniprot_id"]
                                ),
                                "pairing_db": "uniref100",
                            },
                        }
                    },
                    {"ligand": {"ligand": row["SMILES"], "count": 1}},
                ],
                "activity": {
                    "type": row["Standard_Type"],
                    "relation": row["Standard_Relation"],
                    "value": float(row["Standard_Value"]),
                    "units": row["Standard_Units"],
                },
                "assay_group": group_name,
                "len": {
                    "protein": len(sequence),
                    "ligand": 0,
                }
            }
            try:
                mol_len = 0
                mol = Chem.MolFromSmiles(row["SMILES"])
                if mol is not None:
                    mol_len = mol.GetNumAtoms()
                entry["len"]["ligand"] = mol_len
            except Exception:
                pass
            processed_data[entry_name] = entry
            keys.append(entry_name)
            if len(processed_data) % 10000 == 0:
                lmdb.write_data(processed_data)
                processed_data = {}

            protenix_data.append(
                {
                    "sequences": entry["sequences"],
                    "name": entry["name"],
                }
            )

            total_samples += 1
            unique_uniprot_ids.add(row["uniprot_id"])
            source_journals.add((row["source"], row["journal"]))
            unique_units.add(row["Standard_Units"])
            pbar.update(1)

    # Write remaining data to LMDB
    if len(processed_data) > 0:
        lmdb.write_data(processed_data)
        processed_data = {}
    lmdb.set_split("chembl_bdb", keys)
    lmdb.set_split("full", keys)
    lmdb["assay_groups"] = assay_groups
    lmdb.close()

    # Generate JSON output
    print("\nGenerating JSON output...")
    json_output = json.dumps(protenix_data, indent=2)

    # Display statistics
    print("\nProcessing Statistics:")
    print(f"- Total samples processed: {total_samples}")
    print(f"- Unique Uniprot IDs found: {len(unique_uniprot_ids)}")
    print(f"- Unique source/journal combinations: {len(source_journals)}")
    print(f"- Unique units found: {len(unique_units)}")
    print(
        f"- Value range: {df['Standard_Value'].min():.2f} to {df['Standard_Value'].max():.2f} {df['Standard_Units'].iloc[0]}"
    )

    # Save or display results
    with open(output_protenix_json_path, "w") as f:
        f.write(json_output)

    print("\nProcessing completed successfully!")
    print(
        f"Output saved to {output_protenix_json_path} ({total_samples} entries)"
    )


def filter_clean_keys():
    ori_data = pd.read_csv(
        "/data/rerank/protenix/chembl_bdb/chembl_bdb_merged_mask_updated.csv"
    )
    fasta_parser = FastaParser(
        "/data/rerank/protenix/chembl_bdb/chembl_bdb_merged_sequences_checked.fasta"
    )
    filtered_ori_data = []
    ori_data["Uniprot_ID"] = ori_data["Uniprot_ID"].fillna("")
    for _, sample in tqdm(ori_data.iterrows(), ncols=80, total=len(ori_data)):
        if (
            sample["Uniprot_ID"] == ""
            or fasta_parser.get_sequence(sample["Uniprot_ID"]) is None
        ):
            continue
        filtered_ori_data.append(sample)
    with open("/data/rerank/protenix/chembl_bdb/chembl_bdb_protenix.json") as f:
        protenix_data = json.load(f)
    assert len(filtered_ori_data) == len(protenix_data)
    clean_keys = []
    for idx, sample in enumerate(filtered_ori_data):
        if sample["mask"] == 1:
            clean_keys.append(protenix_data[idx]["name"])
    lmdb = LMDBDataset(
        "/data/rerank/protenix/chembl_bdb/chembl_bdb.lmdb", readonly=False
    )
    lmdb.set_split("chembl_bdb_clean", clean_keys)


if __name__ == "__main__":
    # Example usage
    csv_path = "/data/rerank/protenix/chembl_bdb/chembl_bdb_merged.csv"
    fasta_path = "/data/rerank/protenix/chembl_bdb/chembl_bdb_merged_sequences_checked.fasta"
    output_protenix_json_path = (
        "/data/rerank/protenix/chembl_bdb/chembl_bdb_protenix.json"
    )
    output_lmdb = "/data/rerank/protenix/chembl_bdb/chembl_bdb_20250410.lmdb"
    msa_dir = "/data/rerank/protenix/chembl_bdb/precomputed_msa"

    main_process(
        csv_path, fasta_path, output_protenix_json_path, output_lmdb, msa_dir
    )
    filter_clean_keys()
