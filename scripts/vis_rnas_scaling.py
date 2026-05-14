import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

parser = argparse.ArgumentParser(description="Plot metrics vs confidence.")


parser.add_argument(
    "-m",
    "--multi_path",
    type=str,
    required=True,
    help="Path to the input multi results.",
)

parser.add_argument(
    "-g",
    "--genes_path",
    type=str,
    required=True,
    help="Path to genes table.",
)
parser.add_argument(
    "-t",
    "--prob_threshold",
    type=float,
    default=0.5,
    help="Prediction threshold.",
)
parser.add_argument(
    "-u",
    "--unique_path",
    type=str,
    required=True,
    help="Path to the unique file.",
)
parser.add_argument(
    "-s",
    "--bin_size",
    type=int,
    required=True,
    help="Bin size.",
)
parser.add_argument(
    "-o",
    "--output",
    type=str,
    required=True,
    help="Output path.",
)

args = parser.parse_args()

multi_path = Path(args.multi_path)
genes_path = Path(args.genes_path)
unique_path = Path(args.unique_path)
bin_size = args.bin_size
save_folder = Path(args.output)
prob_threshold = args.prob_threshold
save_folder.mkdir(parents=True, exist_ok=True)

multi_dtypes = {
    # "read_ind": "int32",
    "multi_rna": "bool",
    "Z": "float32",
    "gene_ind": "int32",
    "rna_chr": "category",
    "rna_bin": "int32",
    "dna_chr": "category",
    "dna_bin": "int32",
}
genes_dtypes = {
    "chrom": "category",
    "start": "int32",
    "end": "int32",
    "strand": "category",
    "gene_type": "category",
}
cols = [
    # "read_ind",
    "multi_rna",
    "Z",
    "gene_ind",
    "rna_chr",
    "dna_chr",
    "rna_bin",
    "dna_bin",
]

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

multi = pd.read_csv(multi_path, sep="\t", dtype=multi_dtypes, usecols=cols)  # type: ignore

multi = multi[multi["rna_chr"] == multi["dna_chr"]]
multi = multi[multi["Z"] > prob_threshold]
multi.reset_index(drop=True, inplace=True)
multi["distance"] = (multi["dna_bin"] - multi["rna_bin"]).abs()

unique = pd.read_csv(
    unique_path,
    sep="\t",
    usecols=unique_cols,
    dtype=unique_dtypes,  # type: ignore
)

unique = unique[unique["rna_chr"] == unique["dna_chr"]]
unique.reset_index(drop=True, inplace=True)
unique["dna_bin"] = (
    (unique["dna_start"] + unique["dna_end"]) // 2 // bin_size
) * bin_size
unique["rna_bin"] = (
    (unique["rna_start"] + unique["rna_end"]) // 2 // bin_size
) * bin_size
unique.drop(
    ["rna_start", "dna_start", "rna_end", "dna_end"], axis=1, inplace=True
)
unique["distance"] = (unique["dna_bin"] - unique["rna_bin"]).abs()
genes = pd.read_csv(
    genes_path,
    sep="\t",
    dtype=genes_dtypes,  # type: ignore
)
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
gene_inds = multi["gene_ind"].value_counts(sort=True, ascending=False).index

# i_start = 18
print("start plotting")
plt.rcParams.update({"font.size": 12})

for i_start in range(0, 9 * 5, 9):
    fig, axes = plt.subplots(3, 3, figsize=(18, 18), sharex=True, sharey=True)
    axes = axes.flatten()
    for i, gene_ind in enumerate(gene_inds[i_start : i_start + 9]):
        ax = axes[i]

        gene_name = genes.loc[
            genes["gene_ind"] == gene_ind, "gene_name"
        ].item()
        gene_type = genes.loc[
            genes["gene_ind"] == gene_ind, "gene_type"
        ].item()

        mapped_cis_selected = multi[multi["gene_ind"] == gene_ind]

        # multi RNA
        mapped_cis_selected_rna = mapped_cis_selected[
            mapped_cis_selected["multi_rna"]
        ]
        selected_scaling_rna = (
            mapped_cis_selected_rna["distance"].value_counts().reset_index()
        )
        selected_scaling_rna["count"] = (
            selected_scaling_rna["count"] / selected_scaling_rna["count"].sum()
        )

        # multi DNA
        mapped_cis_selected_dna = mapped_cis_selected[
            ~mapped_cis_selected["multi_rna"]
        ]

        selected_scaling_dna = (
            mapped_cis_selected_dna["distance"].value_counts().reset_index()
        )
        selected_scaling_dna["count"] = (
            selected_scaling_dna["count"] / selected_scaling_dna["count"].sum()
        )

        # unique
        unique_selected = unique[unique["gene_ind"] == gene_ind]

        unique_selected_scaling = (
            unique_selected["distance"].value_counts().reset_index()
        )
        unique_selected_scaling["count"] = (
            unique_selected_scaling["count"]
            / unique_selected_scaling["count"].sum()
        )

        ax.scatter(
            selected_scaling_rna["distance"],
            selected_scaling_rna["count"],
            s=40,
            c="darkorange",
            label="multiRNA",
            alpha=0.6,
        )
        ax.scatter(
            selected_scaling_dna["distance"],
            selected_scaling_dna["count"],
            s=40,
            c="dodgerblue",
            label="multiDNA",
            alpha=0.6,
        )
        ax.scatter(
            unique_selected_scaling["distance"],
            unique_selected_scaling["count"],
            s=40,
            c="limegreen",
            label="Unique",
            alpha=0.6,
        )

        ax.set_title(
            f"{gene_name} - {gene_type}", fontsize=16, fontweight="bold"
        )
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.grid(True, which="major", linestyle="--", alpha=0.7)
        ax.set_xlabel("Distance", fontsize=16)
        ax.set_ylabel("Normalized Count", fontsize=16)

    axes[2].legend(fontsize=20)
    plt.tight_layout()
    plt.savefig(
        save_folder / f"rnas_scaling_{i_start}.png",
        dpi=300,
    )
    print(f"finished {i_start}")
