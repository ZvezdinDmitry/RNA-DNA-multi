import argparse
from pathlib import Path

import pandas as pd

parser = argparse.ArgumentParser(
    description="Unique contacts binning into 2 formats: bin2bin, gene2bin"
)

parser.add_argument(
    "-i",
    "--input",
    type=str,
    required=True,
    help="Path to the voted unique contacts in tsv format",
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
    type=str,
    required=True,
    help="Bin size.",
)

args = parser.parse_args()

input_file = Path(args.input)
output_dir = Path(args.output)
bin_size = int(args.bin_size)

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
    "rna_start": "int32",
    "rna_end": "int32",
    "dna_chr": "category",
    "dna_start": "int32",
    "dna_end": "int32",
    "gene_ind": "int32",
}

voted = pd.read_csv(
    input_file, sep="\t", usecols=unique_cols, dtype=unique_dtypes
)

# bin to bin
voted["dna_bin"] = ((voted["dna_start"] + voted["dna_end"]) // 2) // bin_size
voted["rna_bin"] = ((voted["rna_start"] + voted["rna_end"]) // 2) // bin_size

bin_cnts = (
    voted[["rna_chr", "rna_bin", "dna_chr", "dna_bin"]]
    .value_counts()
    .reset_index()
    .sort_values(by=["rna_chr", "rna_bin", "dna_chr", "dna_bin"])
    .reset_index(drop=True)
)


bin_cnts["rna_bin"] = (bin_cnts["rna_bin"] * bin_size).astype(int)
bin_cnts["dna_bin"] = (bin_cnts["dna_bin"] * bin_size).astype(int)
bin_cnts.to_csv(
    output_dir / "unique_bin2bin.tsv",
    sep="\t",
    index=False,
)


# bin to genes unique cnts
bin_genes_cnts = (
    voted[["rna_chr", "gene_ind", "dna_chr", "dna_bin"]]
    .value_counts()
    .reset_index()
    .sort_values(by=["rna_chr", "gene_ind", "dna_chr", "dna_bin"])
    .reset_index(drop=True)
)
bin_genes_cnts["dna_bin"] = (bin_genes_cnts["dna_bin"] * bin_size).astype(int)
bin_genes_cnts = bin_genes_cnts[
    ["gene_ind", "rna_chr", "dna_chr", "dna_bin", "count"]
]

bin_genes_cnts.to_csv(
    output_dir / "unique_gene2bin.tsv",
    sep="\t",
    index=False,
)
