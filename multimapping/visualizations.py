import matplotlib.pyplot as plt
import numpy as np


def plot_single_contact_matrix(
    ax,
    df,
    rna_chr,
    dna_chr,
    title,
    start_bin=None,
    end_bin=None,
    bin_size=50000,
    vmax=7,
):
    filtered = df[(df["rna_chr"] == rna_chr) & (df["dna_chr"] == dna_chr)]

    if filtered.empty:
        print(f"No contacts for {rna_chr}-{dna_chr}")
        return

    if start_bin is not None and end_bin is not None:
        filtered = filtered[
            (filtered["rna_bin"] >= start_bin)
            & (filtered["rna_bin"] <= end_bin)
            & (filtered["dna_bin"] >= start_bin)
            & (filtered["dna_bin"] <= end_bin)
        ]

    if filtered.empty:
        print("No data in this range")
        return

    min_bin = start_bin
    max_bin = end_bin

    n_bins = (max_bin - min_bin) // bin_size + 1
    matrix = np.zeros((n_bins, n_bins), dtype=int)

    for _, row in filtered.iterrows():
        i = (row["rna_bin"] - min_bin) // bin_size
        j = (row["dna_bin"] - min_bin) // bin_size
        matrix[int(i), int(j)] = row["count"]

    im = ax.imshow(
        matrix,
        cmap="viridis",
        interpolation="nearest",
        aspect="auto",
        extent=[min_bin, max_bin, max_bin, min_bin],
        vmax=vmax,
    )
    ax.set_title(title, size=22)
    ax.set_xticks([])
    ax.set_yticks([])
    # ax.set_ylabel('RNA position', size=16)
    # ax.set_xlabel('DNA position', rotation='vertical', size=16)

    return im


def plot_contact_grid(
    df_list,
    rna_chr,
    dna_chr,
    titles,
    start_bin=None,
    end_bin=None,
    bin_size=50000,
    vmax=7,
):
    fig, axes = plt.subplots(2, 2, figsize=(12, 12), sharex=True, sharey=True)
    axes = axes.flatten()

    for i, (df, title) in enumerate(zip(df_list, titles)):
        plot_single_contact_matrix(
            axes[i],
            df,
            rna_chr,
            dna_chr,
            title,
            start_bin,
            end_bin,
            bin_size,
            vmax,
        )

    axes[2].set_xlabel("DNA position", size=22)
    axes[3].set_xlabel("DNA position", size=22)

    axes[0].set_ylabel("RNA position", size=22)
    axes[2].set_ylabel("RNA position", size=22)

    rna_chrom_num = rna_chr[-2:]
    dna_chrom_num = dna_chr[-2:]
    start_mb = start_bin // 1_000_000
    end_mb = end_bin // 1_000_000
    fig.suptitle(
        f"Chromosome {rna_chrom_num} - {dna_chrom_num}: {start_mb} - {end_mb} Mb",
        size=22,
        y=1,
    )

    plt.tight_layout()
    return fig, axes
