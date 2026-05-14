import argparse
from pathlib import Path

import pandas as pd

parser = argparse.ArgumentParser(
    description="Sample a specific number of reads from multi RNA/DNA mapped parts"
)

parser.add_argument(
    "-i",
    "--input",
    type=str,
    required=True,
    help="Path to the input data directory",
)
parser.add_argument(
    "-o",
    "--output",
    type=str,
    required=True,
    help="Path to the output directory",
)
parser.add_argument(
    "-n",
    "--n_samples",
    type=int,
    required=True,
    help="Number of unique reads to sample per dataset.",
)
parser.add_argument(
    "--seed",
    type=int,
    default=42,
    help="Random seed for reproducibility.",
)

args = parser.parse_args()

input_dir = Path(args.input)
output_dir = Path(args.output)
n_samples = args.n_samples
seed = args.seed

output_dir.mkdir(parents=True, exist_ok=True)

rna_part_cols = [
    "read_id_short",
    "rna_chr",
    "rna_start",
    "rna_end",
    "rna_strand",
]
dna_part_cols = ["read_id_short", "dna_chr", "dna_start", "dna_end"]

rna_part_dtypes = {
    "read_id_short": "int32",
    "rna_chr": "category",
    "rna_start": "int32",
    "rna_end": "int32",
    "rna_strand": "category",
}
dna_part_dtypes = {
    "read_id_short": "int32",
    "dna_chr": "category",
    "dna_start": "int32",
    "dna_end": "int32",
}

FILTER_CHRS = ("chrY", "chrM")


def sample_and_filter(unique_df, multi_df, n, random_state):
    """
    Sample multimappers reads.
    """
    unique_reads = unique_df["read_id_short"].drop_duplicates()

    n_to_sample = min(n, len(unique_reads))

    sampled_reads = unique_reads.sample(n=n_to_sample, random_state=random_state)

    unique_filtered = unique_df[
        unique_df["read_id_short"].isin(sampled_reads)
    ].reset_index(drop=True)
    multi_filtered = multi_df[multi_df["read_id_short"].isin(sampled_reads)].reset_index(
        drop=True
    )

    return unique_filtered, multi_filtered


print("Processing multi DNA datasets...")
rna_parts_multi_dna = pd.read_csv(
    input_dir / "rna_parts_multi_dna_all.tsv",
    sep="\t",
    usecols=rna_part_cols,
    dtype=rna_part_dtypes,  # type: ignore
)

rna_parts_multi_dna = rna_parts_multi_dna[
    ~rna_parts_multi_dna["rna_chr"].isin(FILTER_CHRS)
].reset_index(drop=True)

dna_parts_multi_dna = pd.read_csv(
    input_dir / "dna_parts_multi_dna_all.tsv",
    sep="\t",
    usecols=dna_part_cols,
    dtype=dna_part_dtypes,  # type: ignore
)

dna_parts_multi_dna = dna_parts_multi_dna[
    ~dna_parts_multi_dna["dna_chr"].isin(FILTER_CHRS)
].reset_index(drop=True)

rna_parts_multi_dna_sampled, dna_parts_multi_dna_sampled = sample_and_filter(
    rna_parts_multi_dna, dna_parts_multi_dna, n_samples, seed
)

rna_parts_multi_dna_sampled.to_csv(
    output_dir / "rna_parts_multi_dna_sampled.tsv", sep="\t", index=False
)
dna_parts_multi_dna_sampled.to_csv(
    output_dir / "dna_parts_multi_dna_sampled.tsv", sep="\t", index=False
)

del rna_parts_multi_dna, dna_parts_multi_dna
del rna_parts_multi_dna_sampled, dna_parts_multi_dna_sampled


print("Processing multi RNA datasets...")

dna_parts_multi_rna = pd.read_csv(
    input_dir / "dna_parts_multi_rna_all.tsv",
    sep="\t",
    usecols=dna_part_cols,
    dtype=dna_part_dtypes,  # type: ignore
)
dna_parts_multi_rna = dna_parts_multi_rna[
    ~dna_parts_multi_rna["dna_chr"].isin(FILTER_CHRS)
].reset_index(drop=True)

rna_parts_multi_rna = pd.read_csv(
    input_dir / "rna_parts_multi_rna_all.tsv",
    sep="\t",
    usecols=rna_part_cols,
    dtype=rna_part_dtypes,  # type: ignore
)
rna_parts_multi_rna = rna_parts_multi_rna[
    ~rna_parts_multi_rna["rna_chr"].isin(FILTER_CHRS)
].reset_index(drop=True)

dna_parts_multi_rna_sampled, rna_parts_multi_rna_sampled = sample_and_filter(
    dna_parts_multi_rna, rna_parts_multi_rna, n_samples, seed
)

dna_parts_multi_rna_sampled.to_csv(
    output_dir / "dna_parts_multi_rna_sampled.tsv", sep="\t", index=False
)
rna_parts_multi_rna_sampled.to_csv(
    output_dir / "rna_parts_multi_rna_sampled.tsv", sep="\t", index=False
)

print("Done!")
