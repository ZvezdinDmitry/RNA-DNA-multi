import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.metrics import f1_score, precision_score, recall_score


def compute_mterics(multi_preds, mask):
    recall = recall_score(
        multi_preds.loc[mask, "label"], multi_preds.loc[mask, "pred"]
    )
    precision = precision_score(
        multi_preds.loc[mask, "label"], multi_preds.loc[mask, "pred"]
    )
    f1 = f1_score(
        multi_preds.loc[mask, "label"], multi_preds.loc[mask, "pred"]
    )

    return {"Recall": recall, "Precision": precision, "F1": f1}


def get_metrics(multi_preds_all):
    multi_cis_mask = multi_preds_all["dna_chr"] == multi_preds_all["rna_chr"]

    multi_trans_mask = multi_preds_all["dna_chr"] != multi_preds_all["rna_chr"]

    multi_rna_mask = multi_preds_all["multi_rna"]
    multi_dna_mask = ~multi_preds_all["multi_rna"]
    full_mask = multi_cis_mask | multi_trans_mask
    multi_masks = [
        full_mask,
        multi_cis_mask,
        multi_trans_mask,
        multi_rna_mask,
        multi_dna_mask,
    ]
    multi_labels = ["All", "Cis", "Trans", "Multi-RNA", "Multi-DNA"]
    # multi_labels = ["Все", "Внутрихр.", "Межхр.", "Мульти-РНК", "Мульти-ДНК"]
    metrics_list = []
    for multi_mask in multi_masks:
        metrics_list.append(compute_mterics(multi_preds_all, multi_mask))

    metrics_df = pd.DataFrame(metrics_list)
    metrics_df["label"] = multi_labels
    return metrics_df


def visualize_metrics(metrics_df, title):
    # Transform data to "long" format
    df_melted = metrics_df.melt(
        id_vars="label", var_name="Metric", value_name="Value"
    )
    my_palette = ["#9b59b6", "#3498db", "#e74c3c"]
    # Style setup
    plt.figure(figsize=(12, 7))
    sns.set_style("whitegrid")

    # Create the plot
    ax = sns.barplot(
        data=df_melted, x="label", y="Value", hue="Metric", palette=my_palette
    )

    # Add value labels on top of bars rounded to 3 decimal places
    for container in ax.containers:
        ax.bar_label(container, fmt="%.3f", padding=3, size=15)

    plt.title(title, fontsize=20, pad=20)
    plt.ylabel("Metrics value", size=18)
    plt.xticks(size=16)
    plt.xlabel("")
    plt.yticks(size=16)
    plt.ylim(0, 1.0)
    plt.legend(loc="upper left", fontsize=16)
    plt.tight_layout()
