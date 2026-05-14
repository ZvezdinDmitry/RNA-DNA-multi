import argparse
from pathlib import Path

import pandas as pd

from multiK2.simulataions_positions import simulate_multi_positions

parser = argparse.ArgumentParser(
    description="Simulate multimappers based on real data statistics."
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
    "-u",
    "--unique",
    type=str,
    required=True,
    help="Path to the unique contacts",
)

parser.add_argument(
    "-s",
    "--bin_size",
    type=int,
    required=True,
    help="Bin size to calc coverage.",
)

parser.add_argument(
    "-n",
    "--n_unique",
    type=int,
    required=False,
    default=0,
    help="Number of unique reads to take into simulation.",
)

parser.add_argument(
    "-p",
    "--multi_prop",
    type=float,
    required=True,
    help="Percent of unique reads to generate multi from.",
)
parser.add_argument(
    "-m",
    "--mode",
    type=str,
    required=False,
    default="coverage",
    help="Coverage mode - simulate from coverage distribution or uniform mode.",
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
unique_path = Path(args.unique)
bin_size = args.bin_size
n_unique = args.n_unique
multi_prop = args.multi_prop
random_state = args.seed
output_dir.mkdir(parents=True, exist_ok=True)
mode = args.mode

FILTER_CHRS = ("chrY", "chrM")
unique_cols = [
    "rna_chr",
    "rna_start",
    "rna_end",
    "rna_strand",
    "dna_chr",
    "dna_start",
    "dna_end",
    "dna_strand",
]
unique_dtypes = {
    "rna_chr": "category",
    "rna_start": "int32",
    "rna_end": "int32",
    "rna_strand": "category",
    "dna_chr": "category",
    "dna_start": "int32",
    "dna_end": "int32",
    "dna_strand": "category",
}
unique = pd.read_csv(
    unique_path,
    sep="\t",
    usecols=unique_cols,
    dtype=unique_dtypes,  # type: ignore
)
unique = unique[
    (~unique["dna_chr"].isin(FILTER_CHRS))
    & (~unique["rna_chr"].isin(FILTER_CHRS))
].reset_index(drop=True)

if n_unique > 0 and n_unique < len(unique):
    sampled_unique = unique.sample(
        n=n_unique, random_state=random_state * 9999
    )
    unique = sampled_unique

rna_multi_cov = pd.read_csv(
    input_dir / f"{mode}_rna_multi.tsv",
    sep="\t",
    dtype={
        "rna_chr": "category",
        "rna_bin": "int32",
        "rna_strand": "category",
        "proportion": "float64",
    },
)

dna_multi_cov = pd.read_csv(
    input_dir / f"{mode}_dna_multi.tsv",
    sep="\t",
    dtype={"dna_chr": "category", "dna_bin": "int32", "proportion": "float64"},
)

rna_multi_cov = pd.read_csv(
    input_dir / f"{mode}_rna_multi.tsv",
    sep="\t",
    dtype={
        "rna_chr": "category",
        "rna_bin": "int32",
        "rna_strand": "category",
        "proportion": "float64",
    },
)

dna_multi_cov = pd.read_csv(
    input_dir / f"{mode}_dna_multi.tsv",
    sep="\t",
    dtype={"dna_chr": "category", "dna_bin": "int32", "proportion": "float64"},
)

rna_multi_per_read = pd.read_csv(
    input_dir / "rna_multi_per_read.tsv",
    sep="\t",
    dtype={"count": "int32", "proportion": "float64"},
)

dna_multi_per_read = pd.read_csv(
    input_dir / "dna_multi_per_read.tsv",
    sep="\t",
    dtype={"count": "int32", "proportion": "float64"},
)

multi_stats_df = pd.read_csv(
    input_dir / "stats.tsv",
    sep="\t",
)

(
    sim_rna_parts_multi_rna,
    sim_dna_parts_multi_rna,
    sim_dna_parts_multi_dna,
    sim_rna_parts_multi_dna,
    sim_unique,
) = simulate_multi_positions(
    unique_contacts=unique,
    rna_multi_cov=rna_multi_cov,
    dna_multi_cov=dna_multi_cov,
    rna_multi_per_read=rna_multi_per_read,
    dna_multi_per_read=dna_multi_per_read,
    multi_stats_df=multi_stats_df,
    bin_size=bin_size,
    multi_prop=multi_prop,
    random_state=random_state,
)


sim_rna_parts_multi_rna.to_csv(
    output_dir / "rna_parts_multi_rna_all.tsv", sep="\t", index=False
)
sim_dna_parts_multi_rna.to_csv(
    output_dir / "dna_parts_multi_rna_all.tsv", sep="\t", index=False
)

sim_dna_parts_multi_dna.to_csv(
    output_dir / "dna_parts_multi_dna_all.tsv", sep="\t", index=False
)
sim_rna_parts_multi_dna.to_csv(
    output_dir / "rna_parts_multi_dna_all.tsv", sep="\t", index=False
)

sim_unique.to_csv(
    output_dir / "contacts_multi_unique_all.tsv", sep="\t", index=False
)
