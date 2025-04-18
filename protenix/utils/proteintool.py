import os
import json
from pathlib import Path
from tempfile import TemporaryDirectory

import requests
from tqdm import trange


def get_fasta_by_uniprotids_online(uniprotids, output_file):
    """Fetches FASTA sequences for a list of UniProt IDs in batch and saves to a
    file."""
    url = "https://rest.uniprot.org/uniprotkb/stream"
    headers = {"accept": "text/plain;format=fasta"}

    batchsize = 100

    def get_fast_batch(uniprotids):
        # Join IDs with a space (batch request)
        query = " OR ".join([f"accession:{uid}" for uid in uniprotids])
        params = {
            "query": query,  # UniProt IDs query
            "fields": ["accession"],
        }

        response = requests.get(url, headers=headers, params=params)

        if response.status_code == 200:
            return response.text
        else:
            print(
                f"Failed to fetch sequences: {response.status_code}, {response.text}"
            )
            return ""

    with open(output_file, "w") as f:
        for i in trange(0, len(uniprotids), batchsize, ncols=80):
            batch = uniprotids[i : i + batchsize]
            f.write(get_fast_batch(batch))


def extract_fasta_sequences(uniprot_ids, ref_file, output_file):
    """Extracts FASTA sequences for given UniProt IDs from the input file and
    writes them to the output file.

    Args:
        uniprot_ids (list): List of UniProt IDs to extract.
        input_file (str): Path to the input FASTA file.
        output_file (str): Path to the output FASTA file.
    """
    # Convert UniProt IDs to a set for faster lookup
    uniprot_ids_set = set(uniprot_ids)

    with open(ref_file, "r") as infile:
        lines = infile.readlines()

    with open(output_file, "w") as outfile:
        write_sequence = (
            False  # Flag to track if the current sequence should be written
        )
        for line in lines:
            if line.startswith(">"):  # This is a header line
                # Check if the header contains any of the UniProt IDs
                write_sequence = any(
                    uniprot_id in line for uniprot_id in uniprot_ids_set
                )
            if write_sequence:
                # Write the header or sequence line to the output file
                outfile.write(line)


def get_fasta_by_uniprotids(uniprotids, output_file, ref_file=None):
    if ref_file is None or not Path(ref_file).exists():
        print("Fetching sequences from remote UniProt database...")
        return get_fasta_by_uniprotids_online(uniprotids, output_file)
    else:
        return extract_fasta_sequences(uniprotids, ref_file, output_file)


def get_uniprotids_from_fasta(fasta_file):
    """Extracts UniProt IDs from a FASTA file.

    Args:
        fasta_file (str): Path to the input FASTA file.

    Returns:
        list: List of UniProt IDs extracted from the FASTA file.
    """
    uniprot_ids = set()
    with open(fasta_file, "r") as f:
        for line in f:
            if line.startswith(">"):
                # Extract the UniProt ID from the header line
                if "|" in line:
                    # e.g. >sp|O60674|JAK2_HUMAN Tyrosine-protein 
                    uniprot_id = line.split("|")[1]
                else:
                    # e.g. >O60674
                    uniprot_id = line[1:].split()[0]
                uniprot_ids.add(uniprot_id)
    return sorted(uniprot_ids)


def pdbids_to_uniprotids(pdb_ids):
    """Fetches UniProt IDs for a list of PDB IDs in the same order as the input.

    Args:
        pdb_ids (list): List of PDB IDs.

    Returns:
        dict: A dictionary mapping PDB IDs to UniProt IDs.
    """
    url = "https://rest.uniprot.org/uniprotkb/search"
    headers = {"accept": "application/json"}

    # Store results in a dictionary to maintain mapping
    pdb_to_uniprot = {}

    for pdb_id in pdb_ids:
        # Query for each PDB ID individually
        query = f"xref:pdb-{pdb_id}"
        params = {
            "query": query,
            "fields": "accession",
        }

        response = requests.get(url, headers=headers, params=params)

        if response.status_code == 200:
            data = response.json()
            # Extract UniProt IDs for the current PDB ID
            if len(data["results"]) > 0:
                pdb_to_uniprot[pdb_id] = data["results"][0]["primaryAccession"]
        else:
            print(
                f"Failed to fetch UniProt IDs for {pdb_id}: {response.status_code}, {response.text}"
            )

    return pdb_to_uniprot


def mmseqs2_cluster_fasta(fasta_file, identity=30, output_prefix=None):
    """Clusters a FASTA file using MMseqs2.

    Args:
        fasta_file (str): Path to the input FASTA file.
        identity (int): Sequence identity threshold for clustering.

    Returns:
        str: Path to the output clustered FASTA file.
    """
    # Cluster sequences using MMseqs2

    with TemporaryDirectory() as tmpdir:
        if output_prefix is None:
            output_dir = f"{fasta_file.replace('.fasta', '')}.cluster/"
            Path(output_dir).mkdir(parents=True, exist_ok=True)
            output_prefix = f"{output_dir}/id{identity}"
        cmd = (
            f"mmseqs easy-cluster {fasta_file} {output_prefix} "
            f"{tmpdir} --min-seq-id {identity / 100} "
            f" --cov-mode 1"
        )
        os.system(cmd)


def mmseqs2_search_fasta(
    query_fasta, target_fasta, identity=30, output_file=None
):
    """Searches a FASTA file against a target FASTA file using MMseqs2.

    Args:
        query_fasta (str): Path to the query FASTA file.
        target_fasta (str): Path to the target FASTA file.
        identity (int): Sequence identity threshold for clustering.

    Returns:
        str: Path to the output search results.
    """
    query_fasta = str(query_fasta)
    target_fasta = str(target_fasta)
    with TemporaryDirectory() as tmpdir:
        if output_file is None:
            output_dir = (
                f"{query_fasta.replace('.fasta', '')}.search."
                f"{target_fasta.split('/')[-1].replace('.fasta', '')}"
            )
            Path(output_dir).mkdir(parents=True, exist_ok=True)
            output_file = f"{output_dir}/{identity}.m8"
        cmd = (
            f"mmseqs easy-search {query_fasta} {target_fasta} "
            f"{output_file} {tmpdir} "
            f"--min-seq-id {identity / 100} "
            "-s 9 -k 7 --num-iterations 3"
        )
        os.system(cmd)


def split_train_val(uniprotids, test_uniprotids, output_file, ref_file=None):
    res = {"train": [], "valid": []}
    if ref_file is None:
        res["train"] = [
            uniprotid
            for uniprotid in uniprotids
            if uniprotid not in test_uniprotids
        ]
        res["valid"] = [
            uniprotid
            for uniprotid in uniprotids
            if uniprotid in test_uniprotids
        ]
    else:
        with open(ref_file, "r") as f:
            ref_data = f.readlines()

        clusters = {}
        for line in ref_data:
            query, target = line.split()[:2]
            if query not in clusters:
                clusters[query] = [query]
            clusters[query].append(target)

        for cluster in clusters.values():
            if len(set(cluster) & set(test_uniprotids)) > 0:
                res["valid"].extend(list(set(cluster) & set(uniprotids)))
                uniprotids = list(set(uniprotids) - set(cluster))
        res["train"] = uniprotids

    with open(output_file, "w") as f:
        json.dump(res, f)

