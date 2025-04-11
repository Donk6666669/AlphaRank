import time
import argparse
from collections import Counter, defaultdict
import dataclasses
import numpy as np
from multiprocessing import Pool
from functools import partial
import multiprocessing
import os
from pathlib import Path
from typing import List

from protenix.utils.logger import get_logger

logger = get_logger(__name__)

cpu_num = multiprocessing.cpu_count()


@dataclasses.dataclass(frozen=True)
class A3Mentry:
    description: str
    sequence: str
    fasta_sequence: str


def a3m_sequence_to_fasta(sequence):
    res = sequence.strip()
    res = "".join([ch for ch in res if not ch.islower()])
    return res


def read_a3m_file(file_path: str):
    with open(file_path) as f:
        lines = f.readlines()
    a3m_entries = []
    sequence = None
    desc = None
    for l in lines:
        if l.startswith(">"):
            if sequence:
                a3m_entries.append(
                    A3Mentry(
                        desc,
                        sequence,
                        a3m_sequence_to_fasta(sequence),
                    )
                )
            desc = l.strip()
            sequence = ""
        else:
            sequence += l.strip()
    if desc:
        a3m_entries.append(
            A3Mentry(desc, sequence, a3m_sequence_to_fasta(sequence))
        )
    return a3m_entries


def msa_to_stat_matrix(sequences_int, num_unique_chars):
    """
    Compute position-specific character frequency matrix using vectorized operations.

    Args:
        sequences_int: Integer-encoded sequences matrix of shape (num_seqs, seq_length)
        num_unique_chars: Number of unique characters in the original sequences

    Returns:
        stat_matrix: Matrix where each element shows the count of its character
                     at that position across all sequences (shape same as input)
    """
    num_seqs, seq_length = sequences_int.shape

    # Initialize count matrix: rows=positions, columns=unique characters
    position_counts = np.zeros((seq_length, num_unique_chars), dtype=np.int64)

    # Create position indices matrix matching sequences_int shape
    # Example for seq_length=3, num_seqs=2:
    # [[0 1 2],
    #  [0 1 2]]
    position_indices = np.tile(np.arange(seq_length), (num_seqs, 1))

    # Flatten both indices matrices for vectorized counting
    flat_positions = position_indices.ravel()  # Shape: (num_seqs*seq_length,)
    flat_characters = sequences_int.ravel()  # Shape: (num_seqs*seq_length,)

    # Aggregate counts using NumPy's unbuffered addition
    np.add.at(position_counts, (flat_positions, flat_characters), 1)

    # Create statistical matrix using advanced indexing
    stat_matrix = position_counts[
        np.arange(seq_length)[:, None], sequences_int.T
    ].T

    return stat_matrix


def new_process(in_a3m, out_a3m, reduce_ratio, least_seqs):
    """
    Main processing pipeline for sequence selection based on positional entropy.

    Args:
        in_a3m: Input file path for A3M format sequences
        out_a3m: Output file path for filtered sequences
        reduce_ratio: Fraction of sequences to remove in each iteration
        least_seqs: Minimum number of sequences to retain
    """
    # Read and validate input
    a3m_entries = read_a3m_file(in_a3m)
    if not a3m_entries:
        with open(out_a3m, "w") as fd:
            fd.write("")
        return

    # Preprocess: Convert sequences to integer encoding
    # ------------------------------------------------------------------
    # Get all unique characters across sequences
    unique_chars = {
        char for entry in a3m_entries for char in entry.fasta_sequence
    }
    char_to_int = {char: idx for idx, char in enumerate(unique_chars)}
    num_unique_chars = len(unique_chars)

    # Create full integer-encoded matrix once (memory efficient)
    sequences_int_array = np.array(
        [
            [char_to_int[c] for c in entry.fasta_sequence]
            for entry in a3m_entries
        ],
        dtype=np.int64,
    )

    # Initialize index tracking for retained sequences
    retained_indices = np.arange(len(a3m_entries))

    # Iterative sequence removal loop
    # ------------------------------------------------------------------
    while retained_indices.size > least_seqs:
        # Get current subset of sequences
        current_sequences = sequences_int_array[retained_indices]

        # Compute statistical matrix
        stat_matrix = msa_to_stat_matrix(current_sequences, num_unique_chars)

        # Calculate sequence entropy
        log_N = np.log(stat_matrix.shape[0])  # N = number of sequences
        entropy = np.sum(log_N - np.log(stat_matrix), axis=1)

        # Determine sequences to remove
        # ------------------------------------------------------------------
        sorted_indices = np.argsort(entropy)
        num_remove = int(retained_indices.size * reduce_ratio)

        # Identify removable candidates (always keep first sequence)
        removal_candidates = sorted_indices[:num_remove]
        removal_candidates = removal_candidates[removal_candidates != 0]

        # Update retained indices using boolean masking
        retention_mask = np.ones_like(retained_indices, dtype=bool)
        retention_mask[removal_candidates] = False
        retained_indices = retained_indices[retention_mask]

        logger.info(f"{retained_indices.size} sequences remaining")

    # Write final output
    # ------------------------------------------------------------------
    final_entries = [a3m_entries[idx] for idx in retained_indices]
    output_content = "".join(
        f"{entry.description}\n{entry.sequence}\n" for entry in final_entries
    )

    with open(out_a3m, "w") as fd:
        fd.write(output_content)


def seq_entropy_filter(in_a3m, out_a3m, reduce_ratio=0.1, least_seqs=5000):
    # strategy_dir = base_dir
    os.makedirs(Path(out_a3m).parent, exist_ok=True)
    if not os.path.exists(out_a3m):
        start = time.time()
        new_process(in_a3m, out_a3m, reduce_ratio, least_seqs)
        end = time.time()
        logger.info(f"Time taken: {end - start:.2f} seconds")
    else:
        # since the strategy changed, msa selection must reprocess every time
        logger.info("File already exists, skipping...")
        # logger.info("File already exists, but since msa change, selection reprocessing...")
        # process(sfn, tfn, reduce_ratio, least_seqs)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # a3m_dir,strategy_dir,seq_id,cov_id,sample,rm_tmp_files=False
    parser.add_argument("-i", "--input_a3m_path", required=True, type=str)
    parser.add_argument("-o", "--output_a3m_path", required=True, type=str)
    parser.add_argument("-r", "--reduce_ratio", default=0.1, type=float)
    parser.add_argument("-l", "--least_seqs", default=5000, type=int)
    args = parser.parse_args()
    seq_entropy_filter(
        args.input_a3m_path,
        args.output_a3m_path,
        args.reduce_ratio,
        args.least_seqs,
    )
