import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from multimapping.selection import (
    calculate_top1_confidence,
    sample_with_weights,
)

parser = argparse.ArgumentParser(
    description="Calculate prior probailities for scaling and coverage."
)

parser.add_argument(
    "-m",
    "--multi_path",
    type=str,
    required=True,
    help="Path to the input multi results.",
)
parser.add_argument(
    "-u",
    "--unique_path",
    type=str,
    required=True,
    help="Path to the unique file.",
)

parser.add_argument(
    "-t",
    "--prob_threshold",
    type=float,
    default=0.5,
    help="Prediction threshold.",
)
parser.add_argument(
    "-c",
    "--confidence_threshold",
    type=float,
    default=1,
    help="Prediction confidence threshold.",
)
parser.add_argument(
    "-n",
    "--n_sample",
    type=int,
    default=3_000_000,
    help="Sample reads number.",
)
parser.add_argument(
    "-g",
    "--greater_maps_than",
    type=int,
    default=0,
    help="Min number of maps to consider a read.",
)
parser.add_argument(
    "-o",
    "--output",
    type=str,
    required=True,
    help="Output path.",
)
parser.add_argument(
    "-d",
    "--mode",
    type=str,
    default="threshold",
    required=False,
    help="Use random, simple thresholding or 1 to 2 ratio.",
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
multi_path = Path(args.multi_path)
unique_path = Path(args.unique_path)
save_folder = Path(args.output)
prob_threshold = args.prob_threshold
confidence_threshold = args.confidence_threshold
n_sample = args.n_sample
greater_maps_than = args.greater_maps_than
mode = args.mode
sim = args.simulated

save_folder.mkdir(parents=True, exist_ok=True)


multi_dtypes = {
    "read_ind": "int32",
    "multi_rna": "bool",
    "pair_ind": "int32",
    "gene_ind": "int32",
    "Z": "float32",
    "rna_chr": "category",
    "rna_bin": "int32",
    "dna_chr": "category",
    "dna_bin": "int32",
}

multi = pd.read_csv(multi_path, sep="\t", dtype=multi_dtypes)  # type: ignore

# take only high  maps reads
if greater_maps_than != 0:
    multi["map_cnt"] = multi.groupby("read_ind")["rna_bin"].transform("count")
    multi = multi[multi["map_cnt"] >= greater_maps_than]
    print(len(multi), "positions remain")
    multi.reset_index(drop=True, inplace=True)

if mode == "threshold":
    multi["pred"] = multi["Z"] > prob_threshold
elif mode == "confidence":
    multi = calculate_top1_confidence(multi)
    multi["pred"] = multi["confidence"] >= confidence_threshold
elif mode == "sampling":
    multi = sample_with_weights(multi)
else:
    multi["pred"] = multi["Z"] > prob_threshold

if sim:
    multi["pred"] = multi["label"]

# sampled_index = multi.groupby("read_ind").sample(1, random_state=424242).index
# multi["pred"] = False
# multi.loc[sampled_index, "pred"] = True
num_selected = multi["pred"].sum()
num_all = len(multi)
print(f"Mode: {mode}")
print(f"All: {num_all}, selected: {num_selected}, {num_selected / num_all}")

print("multiRNA and multiDNA reads and selections")
print(
    multi.loc[multi["multi_rna"], "read_ind"].max(),
    multi.loc[multi["multi_rna"], "pred"].sum(),
)
print(
    multi.loc[~multi["multi_rna"], "read_ind"].max(),
    multi.loc[~multi["multi_rna"], "pred"].sum(),
)
unique_reads_inds = multi["read_ind"].unique()
if n_sample < len(unique_reads_inds):
    sampled_ids = np.random.choice(
        unique_reads_inds, size=n_sample, replace=False
    )
    mask = np.isin(multi["read_ind"].values, sampled_ids)
    multi = multi[mask].reset_index(drop=True)

if mode == "random":
    sampled_index = (
        multi.groupby("read_ind").sample(1, random_state=424242).index
    )
    print("RANDOM SELECTION MODE")
    print("sampled num", len(sampled_index))
    multi["pred"] = False
    multi.loc[sampled_index, "pred"] = True


multi["distance"] = (multi["dna_bin"] - multi["rna_bin"]).abs()
mapped = multi[multi["pred"]].reset_index(drop=True)

all_cis_ratio = (multi["rna_chr"] == multi["dna_chr"]).mean()
mapped_cis_ratio = (mapped["rna_chr"] == mapped["dna_chr"]).mean()
print(f"Cis ratio, all: {all_cis_ratio}, mapped: {mapped_cis_ratio}")

multi_cis = multi[multi["rna_chr"] == multi["dna_chr"]].reset_index(drop=True)
mapped_cis = mapped[mapped["rna_chr"] == mapped["dna_chr"]].reset_index(
    drop=True
)

unique_bin2bin_dtypes = {
    "rna_chr": "category",
    "rna_bin": "int32",
    "dna_chr": "category",
    "dna_bin": "int32",
    "uni_cnt": "int32",
}

unique = pd.read_csv(
    unique_path,
    sep="\t",
    dtype=unique_bin2bin_dtypes,  # type: ignore
).rename({"uni_cnt": "count"}, axis=1)

unique_num = unique["count"].sum()
unique_cis_ratio = (
    unique.loc[(unique["rna_chr"] == unique["dna_chr"]), "count"].sum()
    / unique_num
)

print(f"Unique num: {unique_num}, unique cis/trans {unique_cis_ratio}")

unique["distance"] = (unique["dna_bin"] - unique["rna_bin"]).abs()

scaling_unique = (
    unique.loc[unique["rna_chr"] == unique["dna_chr"], :]
    .groupby("distance")["count"]
    .sum()
    .reset_index()
)
scaling_unique["count"] = (
    scaling_unique["count"] / scaling_unique["count"].sum()
)

masks = [
    np.full(len(multi_cis), True),
    multi_cis["multi_rna"],
    ~multi_cis["multi_rna"],
]
names = ["all", "multiRNA", "multiDNA"]
for mask, name in zip(masks, names):
    multi_cis_selected = multi_cis[mask]
    if mode == "fractional":
        scaling = (
            multi_cis_selected.groupby("distance")["Z"].sum().reset_index()
        )
        scaling.rename({"Z": "count"}, axis=1, inplace=True)
    else:
        scaling = (
            multi_cis_selected.loc[multi_cis_selected["pred"], "distance"]
            .value_counts()
            .reset_index()
        )
        scaling_unmapped = (
            multi_cis_selected.loc[~multi_cis_selected["pred"], "distance"]
            .value_counts()
            .reset_index()
        )
        scaling_unmapped["count"] = (
            scaling_unmapped["count"] / scaling_unmapped["count"].sum()
        )

    scaling_all = multi_cis_selected["distance"].value_counts().reset_index()

    scaling["count"] = scaling["count"] / scaling["count"].sum()

    scaling_all["count"] = scaling_all["count"] / scaling_all["count"].sum()

    plt.figure(figsize=(8, 6))
    plt.scatter(
        scaling["distance"],
        scaling["count"],
        c="violet",
        label="Fractional maps"
        if mode == "fractional"
        else "Selected mappers",
        zorder=2,
    )
    if mode != "fractional":
        plt.scatter(
            scaling_unmapped["distance"],
            scaling_unmapped["count"],
            label="Unmapped",
        )
    plt.scatter(
        scaling_all["distance"],
        scaling_all["count"],
        label="All multimappers",
        c="dodgerblue",
        zorder=2,
    )
    plt.scatter(
        scaling_unique["distance"],
        scaling_unique["count"],
        c="limegreen",
        label="Unique",
        zorder=2,
    )
    plt.grid(True, alpha=0.7, zorder=1)
    plt.xlabel("Distance", size=22)
    plt.ylabel("Probability", size=22)
    plt.title(f"Multimappers scaling {name}", size=24)
    plt.ylim(1e-5, 2e-1)
    plt.xscale("log")
    plt.yscale("log")
    plt.xticks(size=16)
    plt.yticks(size=16)

    plt.legend(fontsize=18)
    plt.tight_layout()

    # plt.savefig(
    #     save_folder / f"scaling_threshold{CONFIDENCE_RATIO}_{prob_threshold}.png",
    #     dpi=300,
    # )
    if sim:
        name = name + "_labeled"

    if mode == "threshold":
        plt.savefig(
            save_folder
            / f"scaling_{prob_threshold}_{name}_{greater_maps_than}.png",
            dpi=300,
        )
    elif mode == "random":
        plt.savefig(
            save_folder / f"scaling_random_{name}_{greater_maps_than}.png",
            dpi=300,
        )
    elif mode == "confidence":
        plt.savefig(
            save_folder
            / f"scaling_confidence{confidence_threshold}_{name}_{greater_maps_than}.png",
            dpi=300,
        )
    elif mode == "sampling":
        plt.savefig(
            save_folder
            / f"scaling_weights_sampling_{name}_{greater_maps_than}.png",
            dpi=300,
        )
    elif mode == "fractional":
        plt.savefig(
            save_folder / f"scaling_fractional_{name}{greater_maps_than}.png",
            dpi=300,
        )
