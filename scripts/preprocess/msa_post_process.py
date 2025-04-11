from pathlib import Path
from typing import Dict, List, Tuple

from tqdm import tqdm

from scripts.preprocess.seq_entropy import seq_entropy_filter


class A3MProcessor:
    """Processor for A3M file format."""

    def __init__(self, a3m_file: str, out_dir: str):
        self.out_dir = out_dir
        self.a3m_file = Path(a3m_file)
        self.a3m_content = self._read_a3m_file()
        self.chain_info = self._parse_header()

    def _read_a3m_file(self) -> str:
        """Read A3M file content."""
        return self.a3m_file.read_text()

    def _parse_header(self) -> Tuple[List[str], Dict[str, Tuple[int, int]]]:
        """Parse A3M header to get chain information."""
        first_line = self.a3m_content.split("\n")[0]
        if first_line[0] == "#":
            lengths, oligomeric_state = first_line.split("\t")

            chain_lengths = [int(x) for x in lengths[1:].split(",")]
            chain_names = [
                f"10{x+1}" for x in range(len(oligomeric_state.split(",")))
            ]

            # Calculate sequence ranges for each chain
            seq_ranges = {}
            for i, name in enumerate(chain_names):
                start = sum(chain_lengths[:i])
                end = sum(chain_lengths[: i + 1])
                seq_ranges[name] = (start, end)

            return chain_names, seq_ranges
        else:
            non_pairing = ">query\n" + "\n".join(
                self.a3m_content.split("\n")[1:]
            )
            query_seq = self.a3m_content.split("\n")[1]
            pairing = f">query\n{query_seq}"
            msa_path = Path(self.out_dir)
            msa_path.mkdir(exist_ok=True, parents=True)
            with open(msa_path / "non_pairing.a3m", "w") as f:
                f.write(non_pairing)

            with open(msa_path / "pairing.a3m", "w") as f:
                f.write(pairing)

            return [None]

    def _extract_sequence(self, line: str, range_tuple: Tuple[int, int]) -> str:
        """Extract sequence for specific range."""
        seq = []
        no_insert_count = 0
        start, end = range_tuple

        for char in line:
            if char.isupper() or char == "-":
                no_insert_count += 1
            # we keep insertions
            if start < no_insert_count <= end:
                seq.append(char)
            elif no_insert_count > end:
                break

        return "".join(seq)

    def split_sequences(self) -> None:
        """Split A3M file into pairing and non-pairing sequences."""
        out_dir = Path(self.out_dir) / "msa"
        chain_names, seq_ranges = self.chain_info

        pairing_a3ms = {name: [] for name in chain_names}
        nonpairing_a3ms = {name: [] for name in chain_names}

        current_query = None
        for line in self.a3m_content.split("\n"):
            if line.startswith("#"):
                continue

            if line.startswith(">"):
                name = line[1:]
                if name in chain_names:
                    current_query = chain_names[chain_names.index(name)]
                elif name == "\t".join(chain_names):
                    current_query = None

                # Add header line to appropriate dictionary
                if current_query:
                    nonpairing_a3ms[current_query].append(line)
                else:
                    for name in chain_names:
                        pairing_a3ms[name].append(line)
                continue

            # Process sequence line
            if not line:
                continue

            if current_query:
                seq = self._extract_sequence(line, seq_ranges[current_query])
                nonpairing_a3ms[current_query].append(seq)
            else:
                for name in chain_names:
                    seq = self._extract_sequence(line, seq_ranges[name])
                    pairing_a3ms[name].append(seq)

        self._write_output_files(out_dir, nonpairing_a3ms, pairing_a3ms)

    def _write_output_files(
        self,
        out_dir: Path,
        nonpairing_a3ms: Dict[str, List[str]],
        pairing_a3ms: Dict[str, List[str]],
    ) -> None:
        """Write split sequences to output files."""
        out_dir.mkdir(exist_ok=True)

        # Write non-pairing sequences
        for i, (name, lines) in enumerate(nonpairing_a3ms.items()):
            chain_dir = out_dir / str(i)
            chain_dir.mkdir(exist_ok=True)

            with open(chain_dir / "non_pairing.a3m", "w") as f:
                query_seq = lines[1]
                f.write(">query\n")
                f.write(f"{query_seq}\n")
                f.write("\n".join(lines[2:]))

        # Write pairing sequences
        for i, (name, lines) in enumerate(pairing_a3ms.items()):
            chain_dir = out_dir / str(i)
            chain_dir.mkdir(exist_ok=True)

            with open(chain_dir / "pairing.a3m", "w") as f:
                query_seq = lines[1]
                f.write(">query\n")
                f.write(f"{query_seq}\n")

                # Process remaining sequences
                sequences = {}
                for j, line in enumerate(lines[2:]):
                    if line.startswith(">"):
                        current_name = f"UniRef100_{line[1:].split()[i]}_{j}"
                        sequences[current_name] = ""
                    elif line and "DUMMY" not in current_name:
                        sequences[current_name] = line

                # Write processed sequences
                for seq_name, seq in sequences.items():
                    if seq:  # Only write non-empty sequences
                        f.write(f">{seq_name}\n{seq}\n")


def post_process(
    fasta_path,
    msa_dir,
    result_dir,
    filter=False,
    filter_dir="filter",
):
    uniprot_ids = []
    with open(fasta_path, "r") as f:
        for line in f:
            if line.startswith(">"):
                uniprot_ids.append(line[1:].strip())

    msa_dir = Path(msa_dir)
    msa_files = list(msa_dir.glob("*.a3m"))
    assert len(msa_files) == len(
        uniprot_ids
    ), "Number of MSA files does not match number of sequences in FASTA file."
    for i in range(len(msa_files)):
        assert (
            msa_dir / f"{i}.a3m"
        ).exists(), f"{msa_dir}/{i}.a3m does not exist."

    source_dir = msa_dir
    if filter:
        filter_dir = Path(filter_dir)
        filter_dir.mkdir(exist_ok=True, parents=True)
        for i in range(len(msa_files)):
            seq_entropy_filter(msa_dir / f"{i}.a3m", filter_dir / f"{i}.a3m")
        source_dir = filter_dir

    result_dir = Path(result_dir)
    for i, uniprot_id in enumerate(tqdm(uniprot_ids, ncols=80)):
        msa_file = source_dir / f"{i}.a3m"
        processor = A3MProcessor(msa_file, result_dir / uniprot_id)
        if len(processor.chain_info) == 2:
            processor.split_sequences()


def main():
    fasta_path = (
        "/data/rerank/protenix/chembl_bdb/chembl_bdb_unique_sequences.fasta"
    )
    msa_dir = "/data/rerank/protenix/chembl_bdb/msa/data"
    result_dir = "/data/rerank/protenix/chembl_bdb/precomputed_msa"
    # post_process(fasta_path, msa_dir, result_dir)

    filter_dir = "/data/rerank/protenix/chembl_bdb/msa_filtered"
    post_process(
        fasta_path, msa_dir, result_dir, filter=True, filter_dir=filter_dir
    )


if __name__ == "__main__":
    main()
