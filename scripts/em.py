import argparse
from pathlib import Path

import pandas as pd

from multimapping.em import (
    multi_em_numpy,
    prepare_multi_pairs,
)

parser = argparse.ArgumentParser(description="Runs EM.")

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
    "-p",
    "--prior",
    type=str,
    required=True,
    help="Path to the prior files dirs.",
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
parser.add_argument(
    "-n",
    "--num_iter",
    type=int,
    default=50,
    help="Number of iteration of the main pairs EM.",
)
parser.add_argument(
    "-u",
    "--cov_num_iter",
    type=int,
    default=0,
    help="Number of iteration of the RNA and DNA coverage EMs.",
)
parser.add_argument(
    "-w",
    "--scaling_weight",
    type=float,
    default=1,
    help="Weight of scaling prior. If 1, coverage prior is not considered.",
)

parser.add_argument(
    "-t",
    "--change_threshold",
    type=int,
    default=1,
    help="When the number of changing positions <= than this param, EM stops.",
)

THRESHOLD = 0.5  # here used only for changes detection

args = parser.parse_args()

input_dir = Path(args.input)
output_dir = Path(args.output)
prior_path = Path(args.prior)
chrom_sizes_file = Path(args.chrom_sizes)
genes_file = Path(args.genes)
bin_size = args.bin_size
dist_threshold = args.dist_threshold
num_iter = args.num_iter
cov_num_iter = args.cov_num_iter
scaling_weight = args.scaling_weight
change_threshold = args.change_threshold

output_dir.mkdir(parents=True, exist_ok=True)


contacts_file = input_dir / "unique_gene2bin.tsv"
contacts_file_bins = input_dir / "unique_bin2bin.tsv"

scaling_prior_file = prior_path / f"prior_imputed_{bin_size}.tsv"
trans_prior_file = prior_path / f"prior_trans_{bin_size}.tsv"
bins_prior_file = prior_path / f"prior_bins_cov_{bin_size}.tsv"
genes_prior_file = prior_path / f"prior_genes_cov_{bin_size}.tsv"

genes_dtypes = {
    "gene_chr": "category",
    "gene_start": "int32",
    "gene_end": "int32",
    "gene_strand": "category",
}

unique_bin2bin_dtypes = {
    "rna_chr": "category",
    "rna_bin": "int32",
    "dna_chr": "category",
    "dna_bin": "int32",
    "uni_cnt": "int32",
}
unique_gene2bin_dtypes = {
    "rna_chr": "category",
    "gene_ind": "int32",
    "dna_chr": "category",
    "dna_bin": "int32",
    "uni_cnt": "int32",
}
rna_part_dtypes = {
    "read_id_short": "int32",
    "rna_chr": "category",
    "rna_bin": "int32",
    "gene_ind": "int32",
}
dna_part_dtypes = {
    "read_id_short": "int32",
    "dna_chr": "category",
    "dna_bin": "int32",
}
rna_parts_multi_rna_file = input_dir / "rna_parts_multi_rna_binned.tsv"
dna_parts_multi_rna_file = input_dir / "dna_parts_multi_rna_binned.tsv"
dna_parts_multi_dna_file = input_dir / "dna_parts_multi_dna_binned.tsv"
rna_parts_multi_dna_file = input_dir / "rna_parts_multi_dna_binned.tsv"

scaling_prior_dtypes = {"distance": "int32", "prob": "float32"}
trans_prior_dtypes = {
    "rna_chr": "category",
    "dna_chr": "category",
    "prob": "float32",
}
bins_prior_dtypes = {
    "dna_chr": "category",
    "dna_bin": "int32",
    "prob": "float32",
}
genes_prior_dtypes = {"gene_ind": "int32", "prob": "float32"}
print("start reading")
scaling_prior = pd.read_csv(
    scaling_prior_file,
    sep="\t",
    dtype=scaling_prior_dtypes,  # type: ignore
)
prior_trans = pd.read_csv(
    trans_prior_file,
    sep="\t",
    usecols=[0, 1, 3],
    dtype=trans_prior_dtypes,  # type: ignore
)

bins_prior = pd.read_csv(bins_prior_file, sep="\t", dtype=bins_prior_dtypes)  # type: ignore

genes_prior = pd.read_csv(genes_prior_file, sep="\t", dtype=genes_prior_dtypes)  # type: ignore


unique = pd.read_csv(
    contacts_file,
    sep="\t",
    dtype=unique_gene2bin_dtypes,  # type: ignore
).rename({"count": "uni_cnt"}, axis=1)

unique_bins = pd.read_csv(
    contacts_file_bins,
    sep="\t",
    dtype=unique_bin2bin_dtypes,  # type: ignore
).rename({"count": "uni_cnt"}, axis=1)

# multi RNA preprocess
rna_parts_multi_rna = pd.read_csv(
    rna_parts_multi_rna_file,
    dtype=rna_part_dtypes,  # type: ignore
    sep="\t",
)
dna_parts_multi_rna = pd.read_csv(
    dna_parts_multi_rna_file,
    dtype=dna_part_dtypes,  # type: ignore
    sep="\t",
)
dna_parts_multi_rna.drop_duplicates(
    subset="read_id_short", inplace=True, ignore_index=True
)

rna_parts_multi_dna = pd.read_csv(
    rna_parts_multi_dna_file,
    dtype=rna_part_dtypes,  # type: ignore
    sep="\t",
)
rna_parts_multi_dna.drop_duplicates(
    subset="read_id_short", inplace=True, ignore_index=True
)


# if cov_num_iter > 0:
#     print("start multi rna EM")
#     genes = pd.read_csv(
#         "data/genome/genes_inds.tsv",
#         sep="\t",
#         dtype=genes_dtypes,  # type: ignore
#     )
#     multi_rna = em_multi_genes(
#         rna_parts_multi_rna, rna_parts_multi_dna, unique, genes, cov_num_iter
#     )

dna_parts_multi_dna = pd.read_csv(
    dna_parts_multi_dna_file,
    dtype=dna_part_dtypes,  # type: ignore
    sep="\t",
)

multi_rna = rna_parts_multi_rna.merge(
    dna_parts_multi_rna, how="left", on="read_id_short"
)
multi_rna["read_ind"] = pd.factorize(multi_rna["read_id_short"])[0]
multi_rna["read_ind"] = multi_rna["read_ind"].astype("int32")
assert multi_rna.isna().sum().sum() == 0

multi_reads_num = len(dna_parts_multi_rna)
max_rna_read_num = multi_rna["read_ind"].max()
del rna_parts_multi_rna, dna_parts_multi_rna

# if cov_num_iter > 0:
#     print("start multi dna EM")
#     bins = create_fragments(chrom_sizes_file, bin_size)
#     multi_dna = em_multi_bins(
#         multi_rna, dna_parts_multi_dna, unique_bins, bins, cov_num_iter
#     )

multi_dna = rna_parts_multi_dna.merge(
    dna_parts_multi_dna, how="right", on="read_id_short"
)
multi_dna["read_ind"] = pd.factorize(multi_dna["read_id_short"])[0]
multi_dna["read_ind"] = multi_dna["read_ind"].astype("int32")
assert multi_dna.isna().sum().sum() == 0

multi_dna["read_ind"] += max_rna_read_num + 1
multi_reads_num += len(rna_parts_multi_dna)

del dna_parts_multi_dna, rna_parts_multi_dna
print("start preparing dato to EM")

pairs, multi = prepare_multi_pairs(
    multi_rna,
    multi_dna,
    unique,
    scaling_prior,
    prior_trans,
    bins_prior,
    genes_prior,
)
print("start EM")
print(multi_reads_num)
multi, pairs, changes = multi_em_numpy(
    multi, pairs, multi_reads_num, num_iter, scaling_weight
)
print("finished EM")

gene_map = pairs["gene_ind"].values
rna_chr_map = pairs["rna_chr"].values
rna_bin_map = pairs["rna_bin"].values
dna_chr_map = pairs["dna_chr"].values
dna_bin_map = pairs["dna_bin"].values

indices = multi["pair_ind"].values

multi["gene_ind"] = gene_map[indices].astype("int32")
multi["rna_chr"] = rna_chr_map[indices].astype("category")
multi["rna_bin"] = rna_bin_map[indices].astype("int32")
multi["dna_chr"] = dna_chr_map[indices].astype("category")
multi["dna_bin"] = dna_bin_map[indices].astype("int32")

multi.to_csv(
    output_dir
    / f"em_results_{num_iter}_cov{cov_num_iter}_{scaling_weight}.tsv",
    sep="\t",
    index=False,
)
pd.DataFrame({"changes": changes}).to_csv(
    output_dir / f"changes_{num_iter}_cov{cov_num_iter}_{scaling_weight}.tsv",
    sep="\t",
    index=False,
)
print("finished")
