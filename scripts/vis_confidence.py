import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, precision_score, recall_score
from tqdm import tqdm

from multimapping.selection import calculate_top1_confidence

parser = argparse.ArgumentParser(description="Plot metrics vs confidence.")


parser.add_argument(
    "-m",
    "--multi_path",
    type=str,
    required=True,
    help="Path to the input multi results.",
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
save_folder = Path(args.output)
save_folder.mkdir(parents=True, exist_ok=True)

multi_dtypes = {
    "read_ind": "int32",
    "multi_rna": "bool",
    # "pair_ind": "int32",
    "Z": "float32",
    # "gene_ind": "int32",
    # "rna_chr": "category",
    # "rna_bin": "int32",
    # "dna_chr": "category",
    # "dna_bin": "int32",
    "label": bool,
}
cols = ["read_ind", "multi_rna", "Z", "label"]

multi = pd.read_csv(multi_path, sep="\t", dtype=multi_dtypes, usecols=cols)  # type: ignore

multi = calculate_top1_confidence(multi)
thresholds = np.arange(1, 5, 0.1)

masks = [
    np.full(len(multi), True),
    multi["multi_rna"],
    ~multi["multi_rna"],
]
names = ["all", "multiRNA", "multiDNA"]

# full
for mask, name in zip(masks, names):
    recall_list = []
    precision_list = []
    f1_list = []
    labels = multi.loc[mask, "label"]
    for ratio_threshold in tqdm(thresholds):
        predict = multi.loc[mask, "confidence"] >= ratio_threshold
        recall = recall_score(labels, predict)
        precision = precision_score(labels, predict)
        f1 = f1_score(labels, predict)
        recall_list.append(recall)
        precision_list.append(precision)
        f1_list.append(f1)

    best_threshold = list(thresholds)[np.argmax(f1_list)]
    print(f"Best {name} threshold: {best_threshold:.03f}")

    plt.figure(figsize=(10, 6))

    plt.plot(thresholds, recall_list, "o-", color="coral", label="Recall")
    plt.plot(
        thresholds, precision_list, "s-", color="violet", label="Precision"
    )
    plt.plot(thresholds, f1_list, "^-", color="lightcoral", label="F1-score")

    plt.xlabel("Threshold", size=16)
    plt.ylabel("Score", size=16)
    plt.title(
        f"Metrics vs Threshold {name}. The best: {best_threshold:.03f}",
        size=18,
    )
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.ylim((0, 1))
    plt.savefig(save_folder / f"sim_metrics_v_threshold_{name}.png", dpi=300)
