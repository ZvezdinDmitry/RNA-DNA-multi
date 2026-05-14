from pathlib import Path

import numpy as np
import pandas as pd
import scipy.special as scsp
from scipy.interpolate import UnivariateSpline
from sklearn.isotonic import IsotonicRegression

from .intervals_utils import create_fragments


def read_interactions(
    contacts_file: str | Path, bin_size: int, outliers_df=None
):
    unique_dtypes = {
        "rna_chr": "category",
        "rna_bin": "int64",
        "dna_chr": "category",
        "dna_bin": "int64",
        "count": "int64",
    }
    contacts_df = pd.read_csv(
        contacts_file,
        sep="\t",
        dtype=unique_dtypes,
    )
    # for mHi-C format
    contacts_df["rna_bin"] = contacts_df["rna_bin"] + bin_size // 2
    contacts_df["dna_bin"] = contacts_df["dna_bin"] + bin_size // 2

    contacts_df = contacts_df[
        contacts_df["rna_chr"] == contacts_df["dna_chr"]
    ].reset_index(drop=True)

    if outliers_df is not None:
        outliers_df["outlier"] = True
        contacts_df = contacts_df.merge(
            outliers_df,
            how="left",
            on=["rna_chr", "rna_bin", "dna_chr", "dna_bin"],
        )
        contacts_df["outlier"] = contacts_df["outlier"].fillna(False)
        contacts_df = (
            contacts_df[~contacts_df["outlier"]]
            .reset_index(drop=True)
            .drop("outlier", axis=1)
        )

    contacts_df["distance"] = (
        contacts_df["rna_bin"] - contacts_df["dna_bin"]
    ).abs()
    contacts_df = (
        contacts_df[["distance", "count"]]
        .groupby("distance")["count"]
        .sum()
        .reset_index()
    )
    contacts_df = contacts_df.sort_values("distance")
    return contacts_df


def make_bins(contacts_df, resolution, bin_num=50):
    observed = contacts_df["count"].sum()
    per_bin = observed / bin_num
    bins = []
    bins_cnts = []
    interactions_so_far = 0
    interaction_to_full_bin = 0
    dists_to_bin = []
    cnts_to_bin = []
    bin_full = False
    bin_cnt_so_far = 0
    for i, (
        distance,
        cnt,
    ) in contacts_df.iterrows():  # everything here is inrange by definition
        interactions_so_far += cnt
        # if one distance has more than necessary counts to fill a bin
        if (
            interaction_to_full_bin + cnt >= per_bin
        ):  # case cnt >= per_bin included automatically
            interaction_to_full_bin = 0
            bin_full = True
        # if adding the next bin will not fill the bin
        else:
            interaction_to_full_bin += cnt
        # if bin is already full
        dists_to_bin.append(distance)
        cnts_to_bin.append(cnt)
        if bin_full:
            # dynamically update the desiredPerBin after each bin is full
            bin_cnt_so_far += 1
            if bin_cnt_so_far < bin_num:
                per_bin = (observed - interactions_so_far) / (
                    bin_num - bin_cnt_so_far
                )
            bins.append(dists_to_bin)
            bins_cnts.append(cnts_to_bin)
            interaction_to_full_bin = 0
            bin_full = False
            dists_to_bin = []
            cnts_to_bin = []

    bins_cnts = [sum(cnt) for cnt in bins_cnts]
    left = [min(starts) for starts in bins]
    right = [min(starts) for starts in bins]
    right = right[1:] + [max(bins[-1]) + resolution]
    bins_df = pd.DataFrame({"left": left, "right": right})
    bins_df["count"] = bins_cnts
    return bins_df


# dist_scaling just hardcoded from mHi-C
def generate_frag_pairs(
    bins,
    chrom_sizes_file,
    bin_size,
    chrom_list=None,
    outliers_df=None,
    dist_scaling=1_000_000,
):
    fragments = create_fragments(chrom_sizes_file, bin_size, chrom_list)
    fragments = fragments[["chrom", "start"]]
    fragments_max_per_chrom = (
        fragments.groupby("chrom")["start"].max().reset_index()
    )
    chromosomes = fragments_max_per_chrom["chrom"].tolist()
    all_intra_pairs = 0
    fragments_to_bin_all = []
    for chrom in chromosomes:
        # max_frag = fragments_max_per_chrom.loc[
        #     fragments_max_per_chrom["chrom"] == chrom, "start"
        # ].values[0]
        fragments_chr = fragments[fragments["chrom"] == chrom].reset_index(
            drop=True
        )
        fragments_chr = fragments_chr.reset_index().rename(
            {"index": "diag"}, axis=1
        )
        fragments_chr["npairs"] = (
            (fragments_chr.shape[0] - fragments_chr["diag"])
            * (
                1 + (fragments_chr["diag"] > 0).astype(int)
            )  # due to upper and lower triangles of matrix
        )
        fragments_chr["start"] = fragments_chr["start"] + 1
        fragments_to_bin = pd.merge_asof(
            fragments_chr, bins, left_on="start", right_on="left"
        )
        fragments_to_bin["start"] = fragments_chr["start"] - 1
        fragments_to_bin = fragments_to_bin.drop(["diag", "chrom"], axis=1)
        fragments_to_bin["dist_scaling"] = (
            (fragments_to_bin["start"]) * fragments_to_bin["npairs"]
        )
        fragments_to_bin["dist_scaling"] = (
            fragments_to_bin["dist_scaling"] / dist_scaling
        )
        fragments_to_bin = (
            fragments_to_bin.groupby(["left", "right", "count"])[
                ["npairs", "dist_scaling"]
            ]
            .sum()
            .reset_index()
        )
        # all_intra_pairs += (
        #     fragments_chr.shape[0] * (fragments_chr.shape[0] + 1) / 2
        # )
        all_intra_pairs += fragments_chr.shape[0] * (fragments_chr.shape[0])
        baseline_intra_prob = 1.0 / all_intra_pairs
        if outliers_df is not None:
            outliers_df_chr = outliers_df[
                outliers_df["rna_chr"] == chrom
            ].reset_index(drop=True)
            outliers_df_chr["dist"] = (
                np.abs(outliers_df_chr["rna_bin"] - outliers_df_chr["dna_bin"])
                + 1
            )
            outliers_df_chr = outliers_df_chr.sort_values(by="dist")
            outliers_df_chr = pd.merge_asof(
                outliers_df_chr,
                bins[["left", "right"]],
                left_on="dist",
                right_on="left",
            )
            outliers_per_bin_cnt = (
                outliers_df_chr[["left", "right"]].value_counts().reset_index()
            )
            outliers_per_bin_cnt = outliers_per_bin_cnt.rename(
                {"count": "outliers_count"}, axis=1
            )
            fragments_to_bin = fragments_to_bin.merge(
                outliers_per_bin_cnt, how="left", on=["left", "right"]
            )
            fragments_to_bin["outliers_count"] = fragments_to_bin[
                "outliers_count"
            ].fillna(0)
            fragments_to_bin["npairs"] = (
                fragments_to_bin["npairs"] - fragments_to_bin["outliers_count"]
            )
            fragments_to_bin = fragments_to_bin.drop("outliers_count", axis=1)

        fragments_to_bin_all.append(fragments_to_bin)
    fragments_to_bin = fragments_to_bin_all[0]
    for fragments_to_bin_next in fragments_to_bin_all[1:]:
        fragments_to_bin["npairs"] = (
            fragments_to_bin["npairs"] + fragments_to_bin_next["npairs"]
        )
        fragments_to_bin["dist_scaling"] = (
            fragments_to_bin["dist_scaling"]
            + fragments_to_bin_next["dist_scaling"]
        )

    return fragments_to_bin, baseline_intra_prob, all_intra_pairs


def calc_probabilities(bins, dist_scaling=1_000_000, w_file=None):
    observed = bins["count"].sum()
    bins["avg_cc"] = (bins["count"] / bins["npairs"]) / observed
    bins["avg_dist"] = dist_scaling * (bins["dist_scaling"] / bins["npairs"])
    bins["se"] = 0  # just from mHi-C
    bins = bins[["avg_dist", "avg_cc", "se", "npairs", "count"]]
    bins = bins.rename(
        {
            "avg_dist": "avgGenomicDist",
            "avg_cc": "contactProbability",
            "se": "standardError",
            "npairs": "noOfLocusPairs",
            "count": "totalOfContactCounts",
        },
        axis=1,
    )
    if w_file is not None:
        bins.to_csv(w_file, sep="\t", index=False)

    return (
        bins["avgGenomicDist"].values,
        bins["contactProbability"].values,
        bins["standardError"].values,
    )


def fit_spline(
    distance_df,
    x,
    y,
    contacts_file,
    all_intra_pairs,
):
    assert np.all(x[:-1] <= x[1:])
    spline_error = y.min() ** 2  # hardcoded from mHi-C
    ius = UnivariateSpline(x, y, s=spline_error)
    max_x = np.max(x)
    min_x = np.min(x)
    distance_df = distance_df.sort_values(by="distance", ascending=True)
    distance_df = distance_df[
        (distance_df["distance"] >= min_x) & (distance_df["distance"] <= max_x)
    ]
    spline_x = distance_df["distance"].values
    spline_y = ius(spline_x)
    ir = IsotonicRegression(increasing=False)
    calibrated_spline_y = ir.fit_transform(spline_x, spline_y)
    distance_df["calibrated_prob"] = calibrated_spline_y
    distance_prob_df = distance_df
    # residual = ((y - ius(x)) ** 2).sum()
    unique_dtypes = {
        "rna_chr": "category",
        "rna_bin": "int64",
        "dna_chr": "category",
        "dna_bin": "int64",
        "count": "int64",
    }
    contacts_df = pd.read_csv(
        contacts_file,
        sep="\t",
        dtype=unique_dtypes,
    )

    distance_df = distance_df.rename({"count": "count_distance"}, axis=1)
    # pvals for outliers filtering
    contacts_df["distance"] = np.abs(
        contacts_df["dna_bin"] - contacts_df["rna_bin"]
    ).astype(int)
    contacts_df.loc[contacts_df["distance"] > int(max_x), "distance"] = int(
        max_x
    )
    contacts_df.loc[contacts_df["distance"] < int(min_x), "distance"] = int(
        min_x
    )
    contacts_df = contacts_df.merge(distance_df, how="left", on="distance")
    observed = contacts_df["count"].sum()
    p_vals = np.ones(len(contacts_df))
    for i, row in contacts_df[["count", "calibrated_prob"]].iterrows():
        count, prior_p = row
        p_val = scsp.bdtrc(count - 1, observed, prior_p)
        p_vals[i] = p_val

    contacts_df["p_val"] = p_vals
    outlier_threshold = 1 / all_intra_pairs
    outliers_df = contacts_df[
        ["rna_chr", "rna_bin", "dna_chr", "dna_bin", "p_val"]
    ]
    outliers_df = (
        outliers_df[outliers_df["p_val"] < outlier_threshold]
        .reset_index(drop=True)
        .drop("p_val", axis=1)
    )
    return distance_prob_df, outliers_df


def calculate_scaling_prior(
    contacts_file: str | Path,
    chrom_sizes_file: str | Path,
    bin_size: int,
    chrom_list: None | list = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    # 1st run
    distance_df = read_interactions(contacts_file, bin_size)
    bins = make_bins(distance_df, bin_size)

    bins, prob, all_intra_pairs = generate_frag_pairs(
        bins, chrom_sizes_file, bin_size, chrom_list
    )
    x, y, _ = calc_probabilities(bins)
    distance_df_prob, outliers_df = fit_spline(
        distance_df, x, y, contacts_file, all_intra_pairs
    )

    # 2nd run outliers filtering
    distance_df_ = read_interactions(contacts_file, bin_size, outliers_df)
    bins_ = make_bins(distance_df_, bin_size)

    bins_, prob_, all_intra_pairs_ = generate_frag_pairs(
        bins_, chrom_sizes_file, bin_size, chrom_list, outliers_df
    )
    x_, y_, _ = calc_probabilities(bins_)
    distance_df_prob_, outliers_df_ = fit_spline(
        distance_df_, x_, y_, contacts_file, all_intra_pairs_
    )
    prior_probs = distance_df_prob_

    # return outliers filtered prior and not filtered
    return prior_probs, distance_df_prob


def impute_prior(
    prior_probs,
    chrom_sizes_file: str | Path,
    bin_size: int,
    chrom_list: None | list = None,
):
    fragments = create_fragments(
        chrom_sizes_file, bin_size, chrom_list
    ).rename({"start": "distance"}, axis=1)[["chrom", "distance"]]
    fragments = (
        fragments[["distance"]].drop_duplicates().sort_values(by="distance")
    )
    prior_imputed = pd.merge_asof(
        fragments,
        prior_probs[["distance", "calibrated_prob"]],
        direction="nearest",
    )

    prior_imputed = prior_imputed.rename({"calibrated_prob": "prob"}, axis=1)
    return prior_imputed


def calculate_trans_prior(
    contacts_file: str | Path,
    chrom_sizes_file: str | Path,
    bin_size: int,
    chrom_list: None | list = None,
):
    unique_dtypes = {
        "rna_chr": "category",
        "rna_bin": "int64",
        "dna_chr": "category",
        "dna_bin": "int64",
        "count": "int64",
    }
    contacts_df = pd.read_csv(
        contacts_file,
        sep="\t",
        dtype=unique_dtypes,
    )

    contacts_df = contacts_df[
        contacts_df["rna_chr"] != contacts_df["dna_chr"]
    ].reset_index(drop=True)

    trans_prior = (
        contacts_df.groupby(["rna_chr", "dna_chr"])["count"]
        .sum()
        .reset_index()
    )
    trans_prior["prob"] = trans_prior["count"] / trans_prior["count"].sum()
    fragments = create_fragments(
        chrom_sizes_file, bin_size, chrom_list
    ).rename({"start": "distance"}, axis=1)[["chrom", "distance"]]

    trans_pairs_cnt = fragments.groupby("chrom").max().reset_index()
    trans_pairs_cnt["bin_cnt"] = trans_pairs_cnt["distance"] // bin_size
    trans_pairs_cnt = trans_pairs_cnt[["chrom", "bin_cnt"]].merge(
        trans_pairs_cnt[["chrom", "bin_cnt"]],
        how="cross",
        suffixes=["_rna", "_dna"],
    )

    trans_pairs_cnt = trans_pairs_cnt[
        trans_pairs_cnt["chrom_rna"] != trans_pairs_cnt["chrom_dna"]
    ]
    trans_pairs_cnt["npairs"] = (
        trans_pairs_cnt["bin_cnt_rna"] * trans_pairs_cnt["bin_cnt_dna"]
    )

    trans_prior = trans_prior.merge(
        trans_pairs_cnt[["chrom_rna", "chrom_dna", "npairs"]].rename(
            {"chrom_rna": "rna_chr", "chrom_dna": "dna_chr"}, axis=1
        ),
        how="left",
        on=["rna_chr", "dna_chr"],
    )
    trans_prior["prob"] = trans_prior["prob"] / trans_prior["npairs"]

    return trans_prior


def calculate_cov_prior(
    contacts_df, genes, bin_size: int, dist_threshold: int = 5_000_000
):
    contacts_df["dna_bin"] = (
        ((contacts_df["dna_start"] + contacts_df["dna_end"]) // 2) // bin_size
    ) * bin_size
    # contacts_df = (
    #     contacts_df.merge(
    #         genes[["gene_name", "gene_ind"]], how="left", on="gene_name"
    #     )
    #     .reset_index(drop=True)
    #     .dropna(subset="gene_ind")
    # )
    # contacts_df["gene_ind"] = contacts_df["gene_ind"].astype("int")

    condition = (contacts_df["rna_chr"] != contacts_df["dna_chr"]) | (
        (contacts_df["rna_start"] - contacts_df["dna_start"]).abs()
        > dist_threshold
    )

    # contacts_df = contacts_df.drop(
    #     ["rna_end", "dna_end", "rna_start", "dna_start"], axis=1
    # ).reset_index(drop=True)
    # contacts_df_filt = contacts_df_filt.drop(
    #     ["rna_end", "dna_end", "rna_start", "dna_start"], axis=1
    # ).reset_index(drop=True)

    bins_cov = (
        contacts_df.loc[condition, ["dna_chr", "dna_bin"]]
        .value_counts()
        .reset_index()
        .sort_values(["dna_chr", "dna_bin"])
    )
    genes_cov = (
        contacts_df["gene_ind"]
        .value_counts()
        .reset_index()
        .sort_values(["gene_ind"])
    )
    bins_cov = bins_cov.rename({"count": "dna_cov"}, axis=1)
    genes_cov = genes_cov.rename({"count": "gene_cov"}, axis=1)

    genes["length"] = (genes["gene_end"] - genes["gene_start"]).abs()
    genes_cov = genes_cov.merge(genes[["gene_ind", "length"]])

    genes_cov["gene_cov"] = genes_cov["gene_cov"] / genes_cov["length"]
    genes_cov = genes_cov.drop("length", axis=1)
    bins_cov["dna_cov"] = bins_cov["dna_cov"] / bins_cov["dna_cov"].sum()
    genes_cov["gene_cov"] = genes_cov["gene_cov"] / genes_cov["gene_cov"].sum()

    return bins_cov, genes_cov
