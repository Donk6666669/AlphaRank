from pathlib import Path

from protenix.utils.proteintool import (
    get_fasta_by_uniprotids,
    mmseqs2_search_fasta,
    get_uniprotids_from_fasta,
    split_train_val,
)


current_root = Path(__file__).resolve().parent


def fecth_fasta_from_uniprotids(
    uniprotids_file: str,
    output_file: str,
):
    """Fetches FASTA sequences for a list of UniProt IDs in batch and saves to a
    file."""
    uniprotids = set()
    with open(uniprotids_file, "r") as f:
        for line in f:
            if line.strip():
                uniprotids.add(line.strip())
    get_fasta_by_uniprotids(sorted(uniprotids), output_file)


if __name__ == "__main__":
    fecth_fasta_from_uniprotids(
        current_root / "raw_data" / "dude_uniprotids.txt",
        current_root / "raw_data" / "dude.fasta",
    )
    fecth_fasta_from_uniprotids(
        current_root / "raw_data" / "litpcba_uniprotids.txt",
        current_root / "raw_data" / "litpcba.fasta",
    )

    mmseqs2_search_fasta(
        current_root / "raw_data" / "dude.fasta",
        current_root / "raw_data" / "chembl_bdb.fasta",
        identity=30,
    )
    mmseqs2_search_fasta(
        current_root / "raw_data" / "dude.fasta",
        current_root / "raw_data" / "chembl_bdb.fasta",
        identity=60,
    )

    mmseqs2_search_fasta(
        current_root / "raw_data" / "litpcba.fasta",
        current_root / "raw_data" / "chembl_bdb.fasta",
        identity=30,
    )
    mmseqs2_search_fasta(
        current_root / "raw_data" / "litpcba.fasta",
        current_root / "raw_data" / "chembl_bdb.fasta",
        identity=60,
    )

    mmseqs2_search_fasta(
        current_root / "raw_data" / "fep.fasta",
        current_root / "raw_data" / "chembl_bdb.fasta",
        identity=30,
    )
    mmseqs2_search_fasta(
        current_root / "raw_data" / "fep.fasta",
        current_root / "raw_data" / "chembl_bdb.fasta",
        identity=60,
    )

    mmseqs2_search_fasta(
        current_root / "raw_data" / "dude_litpcba_fep.fasta",
        current_root / "raw_data" / "chembl_bdb.fasta",
        identity=30,
    )
    mmseqs2_search_fasta(
        current_root / "raw_data" / "dude_litpcba_fep.fasta",
        current_root / "raw_data" / "chembl_bdb.fasta",
        identity=60,
    )

    dude_uniprotids = get_uniprotids_from_fasta(
        current_root / "raw_data" / "dude.fasta"
    )
    litpcba_uniprotids = get_uniprotids_from_fasta(
        current_root / "raw_data" / "litpcba.fasta"
    )
    fep_uniprotids = get_uniprotids_from_fasta(
        current_root / "raw_data" / "fep.fasta"
    )
    dude_litpcba_fep_uniprotids = get_uniprotids_from_fasta(
        current_root / "raw_data" / "dude_litpcba_fep.fasta"
    )
    chembl_bdb_uniprotids = get_uniprotids_from_fasta(
        current_root / "raw_data" / "chembl_bdb.fasta"
    )

    split_train_val(
        chembl_bdb_uniprotids,
        dude_uniprotids,
        current_root / "splits" / "chembl_bdb.dude.30.json",
        ref_file=current_root / "raw_data" / "dude.search.chembl_bdb" / "30.m8",
    )
    split_train_val(
        chembl_bdb_uniprotids,
        dude_uniprotids,
        current_root / "splits" / "chembl_bdb.dude.60.json",
        ref_file=current_root / "raw_data" / "dude.search.chembl_bdb" / "60.m8",
    )

    split_train_val(
        chembl_bdb_uniprotids,
        litpcba_uniprotids,
        current_root / "splits" / "chembl_bdb.litpcba.30.json",
        ref_file=current_root / "raw_data" / "litpcba.search.chembl_bdb" / "30.m8",
    )
    split_train_val(
        chembl_bdb_uniprotids,
        litpcba_uniprotids,
        current_root / "splits" / "chembl_bdb.litpcba.60.json",
        ref_file=current_root / "raw_data" / "litpcba.search.chembl_bdb" / "60.m8",
    )

    split_train_val(
        chembl_bdb_uniprotids,
        fep_uniprotids,
        current_root / "splits" / "chembl_bdb.fep.30.json",
        ref_file=current_root / "raw_data" / "fep.search.chembl_bdb" / "30.m8",
    )
    split_train_val(
        chembl_bdb_uniprotids,
        fep_uniprotids,
        current_root / "splits" / "chembl_bdb.fep.60.json",
        ref_file=current_root / "raw_data" / "fep.search.chembl_bdb" / "60.m8",
    )

    split_train_val(
        chembl_bdb_uniprotids,
        dude_litpcba_fep_uniprotids,
        current_root / "splits" / "chembl_bdb.dude_litpcba_fep.30.json",
        ref_file=current_root
        / "raw_data"
        / "dude_litpcba_fep.search.chembl_bdb"
        / "30.m8",
    )
    split_train_val(
        chembl_bdb_uniprotids,
        dude_litpcba_fep_uniprotids,
        current_root / "splits" / "chembl_bdb.dude_litpcba_fep.60.json",
        ref_file=current_root
        / "raw_data"
        / "dude_litpcba_fep.search.chembl_bdb"
        / "60.m8",
    )
