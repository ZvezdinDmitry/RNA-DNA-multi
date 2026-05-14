import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from multimapping.selection import (
    calculate_top1_confidence,
    sample_with_weights,
)
from multimapping.visualizations import plot_contact_grid

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
    "-s",
    "--bin_size",
    type=int,
    required=True,
    help="Bin size.",
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
    "-o",
    "--output",
    type=str,
    required=True,
    help="Output path.",
)

args = parser.parse_args()
multi_path = Path(args.multi_path)
unique_path = Path(args.unique_path)
save_folder = Path(args.output)
bin_size = args.bin_size
prob_threshold = args.prob_threshold
confidence_threshold = args.confidence_threshold
mode = args.mode
save_folder.mkdir(parents=True, exist_ok=True)

multi_dtypes = {
    "read_ind": "int32",
    "multi_rna": "bool",
    "pair_ind": "int32",
    "Z": "float32",
    # "gene_ind": "int32",
    "rna_chr": "category",
    "rna_bin": "int32",
    "dna_chr": "category",
    "dna_bin": "int32",
}

multi = pd.read_csv(multi_path, sep="\t", dtype=multi_dtypes)  # type: ignore

SELECTED_CHRS = ("chr19", "chr20")

if mode == "threshold":
    multi["pred"] = multi["Z"] > prob_threshold
elif mode == "confidence":
    multi = calculate_top1_confidence(multi)
    multi["pred"] = multi["confidence"] >= confidence_threshold
elif mode == "sampling":
    multi = sample_with_weights(multi)
else:
    multi["pred"] = multi["Z"] > prob_threshold

# sampled_index = multi.groupby("read_ind").sample(1, random_state=424242).index
# multi["pred"] = False
# multi.loc[sampled_index, "pred"] = True
multi = multi[
    (multi["rna_chr"].isin(SELECTED_CHRS))
    & (multi["dna_chr"].isin(SELECTED_CHRS))
].reset_index(drop=True)
multi["rna_chr"] = multi["rna_chr"].cat.remove_unused_categories()
multi["dna_chr"] = multi["dna_chr"].cat.remove_unused_categories()
if mode == "random":
    print("RANDOM SELECTION MODE")
    sampled_index = (
        multi.groupby("read_ind").sample(1, random_state=424242).index
    )
    print("sampled num", len(sampled_index))
    multi["pred"] = False
    multi.loc[sampled_index, "pred"] = True

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

unique = unique[
    (unique["rna_chr"].isin(SELECTED_CHRS))
    & (unique["dna_chr"].isin(SELECTED_CHRS))
].reset_index(drop=True)
unique["rna_chr"] = unique["rna_chr"].cat.remove_unused_categories()
unique["dna_chr"] = unique["dna_chr"].cat.remove_unused_categories()

if mode == "fractional":
    mapped_binned = (
        multi.groupby(["rna_chr", "rna_bin", "dna_chr", "dna_bin"])["Z"]
        .sum()
        .reset_index()
    )
    mapped_binned.rename({"Z": "count"}, axis=1, inplace=True)

else:
    mapped = multi[multi["pred"]].reset_index(drop=True)
    mapped_binned = (
        mapped[["rna_chr", "rna_bin", "dna_chr", "dna_bin"]]
        .value_counts()
        .reset_index()
    )

multi_binned = (
    multi[["rna_chr", "rna_bin", "dna_chr", "dna_bin"]]
    .value_counts()
    .reset_index()
)


unique_multi_merged = unique.merge(
    mapped_binned,
    how="outer",
    on=["rna_chr", "rna_bin", "dna_chr", "dna_bin"],
    suffixes=["_uni", "_multi"],
)
unique_multi_merged["count_uni"] = unique_multi_merged["count_uni"].fillna(0)
unique_multi_merged["count_multi"] = unique_multi_merged["count_multi"].fillna(
    0
)
unique_multi_merged["count"] = (
    unique_multi_merged["count_multi"] + unique_multi_merged["count_uni"]
)

chrs_list = [
    ("chr20", "chr20"),
    ("chr19", "chr19"),
    ("chr19", "chr20"),
    ("chr20", "chr19"),
    ("chr19", "chr19"),
]
coords_list = [*[(35_000_000, 50_000_000)] * 4, (10_000_000, 25_000_000)]
titles = [
    "Unique",
    "Unique & selected",
    "Selected mappers",
    "All multimappers",
]
# titles = [
#     "Уникальные",
#     "Уникальные & Разрешенные",
#     "Разрешенные множественные",
#     "Все множественные позиции",
# ]
df_list = [unique, unique_multi_merged, mapped_binned, multi_binned]


for chrs, coords in zip(chrs_list, coords_list):
    chr_1, chr_2 = chrs
    start_bin, end_bin = coords
    fig, axes = plot_contact_grid(
        df_list=df_list,
        rna_chr=chr_1,
        dna_chr=chr_2,
        titles=titles,
        start_bin=start_bin,
        end_bin=end_bin,
        bin_size=bin_size,
        vmax=5,
    )
    # axes[2].set_xlabel("Позиция ДНК", size=22)
    # axes[3].set_xlabel("Позиция ДНК", size=22)

    # axes[0].set_ylabel("Позиция РНК", size=22)
    # axes[2].set_ylabel("Позиция РНК", size=22)
    # fig.suptitle(
    #     #"Хромосома 19: 35 - 50 Мб, бин 50 Кб",
    #     size=22,
    #     y=1,
    # )
    if mode == "threshold":
        plt.savefig(
            save_folder
            / f"matricies_{prob_threshold}_{chr_1}_{chr_2}_{start_bin}_{end_bin}.png",
            dpi=300,
        )
    elif mode == "confidence":
        plt.savefig(
            save_folder
            / f"matricies_confidence{confidence_threshold}_{chr_1}_{chr_2}_{start_bin}_{end_bin}.png",
            dpi=300,
        )
    elif mode == "sampling":
        plt.savefig(
            save_folder
            / f"matricies_weighted_sampling_{chr_1}_{chr_2}_{start_bin}_{end_bin}.png",
            dpi=300,
        )
    elif mode == "random":
        plt.savefig(
            save_folder
            / f"matricies_random_{chr_1}_{chr_2}_{start_bin}_{end_bin}.png",
            dpi=300,
        )
    elif mode == "fractional":
        plt.savefig(
            save_folder
            / f"matricies_fractional_{chr_1}_{chr_2}_{start_bin}_{end_bin}.png",
            dpi=300,
        )
