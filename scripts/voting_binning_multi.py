import argparse
from pathlib import Path

import pandas as pd

from multimapping.voting_eff import bin_multi_dna, bin_multi_rna, voting

parser = argparse.ArgumentParser(
    description="RNA parts voting & binnning of multi RNA and DNA parts"
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
    "-s",
    "--bin_size",
    type=int,
    required=True,
    help="Bin size.",
)
parser.add_argument(
    "-f",
    "--simulated",
    type=bool,
    required=False,
    default=False,
    help="Is multimappers simulated and label columns hsould be parsed.",
)

args = parser.parse_args()

input_dir = Path(args.input)
output_dir = Path(args.output)
genes_path = Path(args.genes)
bin_size = args.bin_size
sim = args.simulated

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

genes_dtypes = {
    "chrom": "category",
    "start": "int32",
    "end": "int32",
    "strand": "category",
}
print("start reading")

rna_parts_multi_dna = pd.read_csv(
    input_dir / "rna_parts_multi_dna_all.tsv",
    sep="\t",
    usecols=rna_part_cols,
    dtype=rna_part_dtypes,  # type: ignore
)

FILTER_CHRS = ("chrY", "chrM")
rna_parts_multi_dna = rna_parts_multi_dna[
    ~rna_parts_multi_dna["rna_chr"].isin(FILTER_CHRS)
].reset_index(drop=True)

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
# genes = genes[~genes["gene_chr"].isin(FILTER_CHRS)].reset_index(drop=True)

# genes = genes.rename({"gene_strand": "strand"}, axis=1)
genes["gene_ind"] = genes.index.astype("int32")

unique = pd.read_csv(
    input_dir / "contacts_multi_unique_all.tsv",
    sep="\t",
    usecols=unique_cols,
    dtype=unique_dtypes,  # type: ignore
)
unique = unique[
    (~unique["dna_chr"].isin(FILTER_CHRS))
    & (~unique["rna_chr"].isin(FILTER_CHRS))
].reset_index(drop=True)

print("voting")
unique_voted, genes = voting(unique, rna_parts_multi_dna, genes)

unique_voted.to_csv(output_dir / "voted_unique.tsv", sep="\t", index=False)

del unique_voted


# if label_column:
#     pass

# multi dna processing

dna_parts_multi_dna = pd.read_csv(
    input_dir / "dna_parts_multi_dna_all.tsv",
    sep="\t",
    usecols=dna_part_cols + ["label"] if sim else [],
    dtype=dna_part_dtypes,  # type: ignore
)
dna_parts_multi_dna = dna_parts_multi_dna[
    ~dna_parts_multi_dna["dna_chr"].isin(FILTER_CHRS)
].reset_index(drop=True)
print("binning multi dna")
(
    dna_parts_multi_dna_binned,
    rna_parts_multi_dna_binned,
    unique_after_binning_multi_dna,
) = bin_multi_dna(
    rna_parts_multi_dna,
    dna_parts_multi_dna,
    genes,
    bin_size,
    drop_positions=True,
)

del rna_parts_multi_dna, dna_parts_multi_dna
print("finished binning")
rna_parts_multi_dna_binned.to_csv(
    output_dir / "rna_parts_multi_dna_binned.tsv", sep="\t", index=False
)
dna_parts_multi_dna_binned.to_csv(
    output_dir / "dna_parts_multi_dna_binned.tsv", sep="\t", index=False
)
unique_after_binning_multi_dna.to_csv(
    output_dir / "unique_after_binning_multi_dna.tsv", sep="\t", index=False
)

del (
    rna_parts_multi_dna_binned,
    dna_parts_multi_dna_binned,
    unique_after_binning_multi_dna,
)

# multi rna processing
rna_parts_multi_rna = pd.read_csv(
    input_dir / "rna_parts_multi_rna_all.tsv",
    sep="\t",
    usecols=rna_part_cols + ["label"] if sim else [],
    dtype=rna_part_dtypes,  # type: ignore
)
dna_parts_multi_rna = pd.read_csv(
    input_dir / "dna_parts_multi_rna_all.tsv",
    sep="\t",
    usecols=dna_part_cols,
    dtype=dna_part_dtypes,  # type: ignore
)

rna_parts_multi_rna = rna_parts_multi_rna[
    ~rna_parts_multi_rna["rna_chr"].isin(FILTER_CHRS)
].reset_index(drop=True)

dna_parts_multi_rna = dna_parts_multi_rna[
    ~dna_parts_multi_rna["dna_chr"].isin(FILTER_CHRS)
].reset_index(drop=True)

print("binning multi rna")
(
    rna_parts_multi_rna_binned,
    dna_parts_multi_rna_binned,
    unique_after_binning_multi_rna,
) = bin_multi_rna(
    rna_parts_multi_rna,
    dna_parts_multi_rna,
    genes,
    bin_size,
    drop_positions=True,
)

del rna_parts_multi_rna, dna_parts_multi_rna
print("finished binning")

rna_parts_multi_rna_binned.to_csv(
    output_dir / "rna_parts_multi_rna_binned.tsv", sep="\t", index=False
)
dna_parts_multi_rna_binned.to_csv(
    output_dir / "dna_parts_multi_rna_binned.tsv", sep="\t", index=False
)
unique_after_binning_multi_rna.to_csv(
    output_dir / "unique_after_binning_multi_rna.tsv", sep="\t", index=False
)
