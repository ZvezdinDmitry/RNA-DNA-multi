import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from multimapping.metrics import get_metrics, visualize_metrics
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
    "-e",
    "--temperature",
    type=float,
    default=1,
    help="Temperature for sampling.",
)
parser.add_argument(
    "-l",
    "--lower_maps_than",
    type=int,
    default=0,
    help="Max number of maps to consider a read.",
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

args = parser.parse_args()

multi_path = Path(args.multi_path)
save_folder = Path(args.output)
prob_threshold = args.prob_threshold
confidence_threshold = args.confidence_threshold
temperature = args.temperature
mode = args.mode
lower_maps_than = args.lower_maps_than
greater_maps_than = args.greater_maps_than
save_folder.mkdir(parents=True, exist_ok=True)

multi_dtypes = {
    "read_ind": "int32",
    "multi_rna": "bool",
    "pair_ind": "int32",
    "Z": "float32",
    "gene_ind": "int32",
    "rna_chr": "category",
    "rna_bin": "int32",
    "dna_chr": "category",
    "dna_bin": "int32",
    "label": "bool",
}

multi = pd.read_csv(multi_path, sep="\t", dtype=multi_dtypes)  # type: ignore

print(len(multi))
reads_list = multi.loc[multi["label"], "read_ind"]
multi = multi[multi["read_ind"].isin(reads_list)]
multi.reset_index(drop=True, inplace=True)
print(len(multi))

print(multi.loc[multi["multi_rna"], "label"].mean())
print(multi.loc[~multi["multi_rna"], "label"].mean())
print(multi.loc[multi["dna_chr"] == multi["rna_chr"], "label"].mean())
print(multi.loc[multi["dna_chr"] != multi["rna_chr"], "label"].mean())

# take only high or low maps reads
if greater_maps_than != 0 or lower_maps_than != 0:
    multi["map_cnt"] = multi.groupby("read_ind")["rna_bin"].transform("count")

if greater_maps_than != 0:
    multi = multi[multi["map_cnt"] >= greater_maps_than]
    print(len(multi), "positions remain")
    multi.reset_index(drop=True, inplace=True)

if lower_maps_than != 0:
    multi = multi[multi["map_cnt"] <= lower_maps_than]
    print(len(multi), "positions remain")
    multi.reset_index(drop=True, inplace=True)


if mode == "threshold":
    multi["pred"] = multi["Z"] > prob_threshold
elif mode == "confidence":
    multi = calculate_top1_confidence(multi)
    multi["pred"] = multi["confidence"] >= confidence_threshold
elif mode == "sampling":
    multi = sample_with_weights(multi, temperature)
else:
    sampled_index = (
        multi.groupby("read_ind").sample(1, random_state=424242).index
    )
    print("sampled num", len(sampled_index))
    multi["pred"] = False
    multi.loc[sampled_index, "pred"] = True

print("pred num", multi["pred"].sum())

metrics_df = get_metrics(multi)

if mode == "threshold":
    visualize_metrics(
        metrics_df, f"Metrics with prob threshold {prob_threshold}"
    )
    plt.savefig(
        save_folder
        / f"metrics_threshold{prob_threshold}_{lower_maps_than}_{greater_maps_than}.png",
        dpi=300,
    )
elif mode == "confidence":
    visualize_metrics(
        metrics_df, f"Metrics with confidence threshold {confidence_threshold}"
    )
    plt.savefig(
        save_folder
        / f"metrics_confidence{confidence_threshold}_{lower_maps_than}_{greater_maps_than}.png",
        dpi=300,
    )
elif mode == "sampling":
    visualize_metrics(metrics_df, "Metrics with weighted sampling")
    plt.savefig(
        save_folder
        / f"metrics_sampling{temperature}_{lower_maps_than}_{greater_maps_than}.png",
        dpi=300,
    )
else:
    visualize_metrics(metrics_df, "Metrics with random selection")
    plt.savefig(
        save_folder
        / f"metrics_random_choice_{lower_maps_than}_{greater_maps_than}.png",
        dpi=300,
    )
