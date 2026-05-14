import argparse
from pathlib import Path

import pandas as pd

from multimapping.simulataions_positions import get_multi_coverage_positions

parser = argparse.ArgumentParser(
    description="Calculate multimappers stats for simulations."
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
    "-s",
    "--bin_size",
    type=int,
    required=True,
    help="Bin size to calc coverage.",
)
parser.add_argument(
    "-n",
    "--min_maps",
    type=int,
    default=10,
    required=False,
    help="Min map positions for read to be considered.",
)
parser.add_argument(
    "-d",
    "--dist_threshold",
    type=int,
    default=5_000_000,
    help="Threshhold to filter out close cis contacts for DNA parts cov.",
)
parser.add_argument(
    "-m",
    "--mode",
    type=str,
    required=False,
    default="coverage",
    help="Coverage mode - simulate from coverage distribution or uniform mode.",
)

args = parser.parse_args()

input_dir = Path(args.input)
output_dir = Path(args.output)
bin_size = args.bin_size
min_multi_cnt = args.min_maps
dist_threshold = args.dist_threshold
mode = args.mode
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


rna_parts_multi_rna = pd.read_csv(
    input_dir / "rna_parts_multi_rna_all.tsv",
    sep="\t",
    usecols=rna_part_cols,
    dtype=rna_part_dtypes,  # type: ignore
)
rna_parts_multi_rna = rna_parts_multi_rna[
    ~rna_parts_multi_rna["rna_chr"].isin(FILTER_CHRS)
].reset_index(drop=True)

mean_multi_rna_len = int(
    (rna_parts_multi_rna["rna_end"] - rna_parts_multi_rna["rna_start"])
    .abs()
    .mean()
)
mean_multi_dna_len = int(
    (dna_parts_multi_dna["dna_end"] - dna_parts_multi_dna["dna_start"])
    .abs()
    .mean()
)

multi_dna = rna_parts_multi_dna.merge(
    dna_parts_multi_dna, how="right", on="read_id_short"
)
assert multi_dna.isna().sum().sum() == 0

del dna_parts_multi_dna, rna_parts_multi_dna


(
    dna_multi_cov,
    rna_multi_cov,
    dna_multi_per_read,
    rna_multi_per_read,
    prop_multi_rna,
) = get_multi_coverage_positions(
    rna_parts_multi_rna,
    multi_dna,
    mode=mode,
    bin_size=bin_size,
    min_multi_cnt=min_multi_cnt,
    dist_threshold=dist_threshold,
)

stats_file = output_dir / "stats.tsv"

stats = {
    "mean_multi_rna_len": mean_multi_rna_len,
    "mean_multi_dna_len": mean_multi_dna_len,
    "prop_multi_rna": prop_multi_rna,
}

pd.DataFrame(stats, index=[0]).to_csv(stats_file, sep="\t", index=False)

dna_multi_cov.to_csv(
    output_dir / f"{mode}_dna_multi.tsv", sep="\t", index=False
)
rna_multi_cov.to_csv(
    output_dir / f"{mode}_rna_multi.tsv", sep="\t", index=False
)

dna_multi_per_read.to_csv(
    output_dir / "dna_multi_per_read.tsv", sep="\t", index=False
)
rna_multi_per_read.to_csv(
    output_dir / "rna_multi_per_read.tsv", sep="\t", index=False
)
