import argparse
from pathlib import Path

import pandas as pd

from multimapping.prior import (
    calculate_cov_prior,
    calculate_scaling_prior,
    calculate_trans_prior,
    impute_prior,
)

parser = argparse.ArgumentParser(
    description="Calculate prior probailities for scaling and coverage."
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
    "-g",
    "--genes",
    type=str,
    required=True,
    help="Path to the genes bedrc file.",
)
parser.add_argument(
    "-c",
    "--chrom_sizes",
    type=str,
    required=True,
    help="Path to the chromosomes sizes file.",
)
parser.add_argument(
    "-s",
    "--bin_size",
    type=int,
    required=True,
    help="Bin size.",
)
parser.add_argument(
    "-d",
    "--dist_threshold",
    type=int,
    default=5_000_000,
    help="Threshhold to filter out close cis contacts for DNA parts cov.",
)

args = parser.parse_args()

input_dir = Path(args.input)
output_dir = Path(args.output)
genes_path = Path(args.genes)
chrom_sizes_file = Path(args.chrom_sizes)
bin_size = args.bin_size
dist_threshold = args.dist_threshold
output_dir.mkdir(parents=True, exist_ok=True)

contacts_file = input_dir / "unique_bin2bin.tsv"

prior_probs, _ = calculate_scaling_prior(
    contacts_file, chrom_sizes_file, bin_size
)
prior_imputed = impute_prior(prior_probs, chrom_sizes_file, bin_size)

prior_imputed.to_csv(
    output_dir / f"prior_imputed_{bin_size}.tsv",
    sep="\t",
    index=False,
)

# trans prior
trans_prior = calculate_trans_prior(contacts_file, chrom_sizes_file, bin_size)

trans_prior.to_csv(
    output_dir / f"prior_trans_{bin_size}.tsv",
    sep="\t",
    index=False,
)

# coverage prior
genes_dtypes = {
    "gene_chr": "category",
    "gene_start": "int32",
    "gene_end": "int32",
    "gene_strand": "category",
}
genes = pd.read_csv(genes_path, sep="\t", dtype=genes_dtypes)  # type: ignore
genes = genes.rename(
    {
        "name": "gene_name",
        "chrom": "gene_chr",
        "start": "gene_start",
        "end": "gene_end",
    },
    axis=1,
)
genes["gene_ind"] = genes.index.astype("int32")
# use not binned interactions
unique_cols = [
    "rna_chr",
    "rna_start",
    "rna_end",
    "gene_ind",
    "dna_chr",
    "dna_start",
    "dna_end",
]
unique_dtypes = {
    "rna_chr": "category",
    "rna_start": "int64",
    "rna_end": "int64",
    "dna_chr": "category",
    "dna_start": "int64",
    "dna_end": "int64",
    "gene_ind": "int64",
}
contacts_df = pd.read_csv(
    input_dir / "voted_unique.tsv",
    sep="\t",
    usecols=unique_cols,
    dtype=unique_dtypes,  # type: ignore
)

bins_cov, genes_cov = calculate_cov_prior(
    contacts_df, genes, bin_size, dist_threshold
)

bins_cov.to_csv(
    output_dir / f"prior_bins_cov_{bin_size}.tsv",
    sep="\t",
    index=False,
)

genes_cov.to_csv(
    output_dir / f"prior_genes_cov_{bin_size}.tsv",
    sep="\t",
    index=False,
)
