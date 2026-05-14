import numpy as np
import pandas as pd
from tqdm import tqdm


def process_multi(multi):
    multi.drop(
        [
            "gene_ind",
            "rna_chr",
            "rna_bin",
            "dna_chr",
            "dna_bin",
        ],
        axis=1,
        inplace=True,
    )
    multi["Z"] = np.float32(0.0)

    return multi


def prepare_multi_pairs(
    multi_rna,
    multi_dna,
    unique,
    scaling_prior,
    prior_trans,
    bins_prior,
    genes_prior,
):
    multi_rna["multi_rna"] = True
    multi_dna["multi_rna"] = False
    multi = pd.concat([multi_rna, multi_dna], ignore_index=True)
    del multi_rna, multi_dna
    # renorm scaling and trans prior
    # trans_prop = (
    #     unique.loc[unique["rna_chr"] != unique["dna_chr"], "uni_cnt"].sum()
    #     / unique["uni_cnt"].sum()
    # )
    # prior_trans["prob"] = prior_trans["prob"] * trans_prop
    # scaling_prior["prob"] = scaling_prior["prob"] * (1 - trans_prop)
    multi["pair_ind"] = multi.groupby(
        ["gene_ind", "rna_chr", "rna_bin", "dna_chr", "dna_bin"]
    ).ngroup()
    # generate all possible mapping pairs
    pairs = multi[
        ["pair_ind", "gene_ind", "rna_chr", "rna_bin", "dna_chr", "dna_bin"]
    ].copy()
    pairs.drop_duplicates(inplace=True)
    pairs.sort_values(by=["pair_ind"], inplace=True)
    pairs.reset_index(drop=True, inplace=True)

    print("pairs built")
    # merge priors and pairs
    pairs["distance"] = np.abs(pairs["dna_bin"] - pairs["rna_bin"])

    # trans pairs
    pairs.loc[pairs["dna_chr"] != pairs["rna_chr"], "distance"] = -1

    pairs = pairs.merge(scaling_prior, on="distance", how="left")
    pairs = pairs.merge(
        prior_trans,
        on=["rna_chr", "dna_chr"],
        how="left",
        suffixes=["", "_trans"],
    )
    # fill trans probs
    pairs.loc[pairs["prob"].isna(), "prob"] = pairs["prob_trans"]
    pairs.drop(["prob_trans", "distance"], axis=1, inplace=True)
    print("scaling prior merged")
    # merge cov prior
    pairs = pairs.merge(genes_prior, how="left", on="gene_ind").merge(
        bins_prior, how="left", on=["dna_chr", "dna_bin"]
    )
    min_gene_prob = genes_prior["gene_cov"].min()
    min_bin_prob = bins_prior["dna_cov"].min()
    print("cov prior merged")
    # NA filling and pseudocnts
    pairs["dna_cov"] = pairs["dna_cov"].fillna(min_bin_prob)
    # pairs["dna_cov"] = pairs["dna_cov"] + 1
    pairs["gene_cov"] = pairs["gene_cov"].fillna(min_gene_prob)
    # pairs["gene_cov"] = pairs["gene_cov"] + 1

    # multiplication and normalization
    # pairs["dna_cov"] = pairs["dna_cov"] / pairs["dna_cov"].sum()
    # pairs["gene_cov"] = pairs["gene_cov"] / pairs["gene_cov"].sum()

    pairs["cov_prior"] = pairs["gene_cov"] * pairs["dna_cov"]
    pairs = pairs.drop(["gene_cov", "dna_cov"], axis=1)
    pairs["cov_prior"] = pairs["cov_prior"] / pairs["cov_prior"].sum()
    print("cov prior aggregated")
    # adding unique contacts
    pairs = pairs.merge(
        unique[["gene_ind", "dna_chr", "dna_bin", "uni_cnt"]],
        how="left",
        on=["gene_ind", "dna_chr", "dna_bin"],
    )
    pairs["uni_cnt"].fillna(0, inplace=True)
    print("unique merged")
    pairs["prior"] = pairs["prob"].copy()

    # unique pairs DF
    # pairs["pair_ind"] = pairs.index.astype("int32")

    pairs["prob"] = pairs["prob"] / pairs["prob"].sum()
    pairs["prior"] = pairs["prior"] / pairs["prior"].sum()
    pairs["cov_prior"] = pairs["cov_prior"] / pairs["cov_prior"].sum()

    multi = process_multi(multi)

    return pairs, multi


def multi_em_numpy(
    multi,
    pairs,
    multi_reads_num,
    num_iter,
    scaling_weight: float = 0.5,
    prior_weight: float = 1,
    prob_threshold: float = 0.5,
    change_threshold: float = 1,
):
    changes = []
    # cnt reads
    num_unique = pairs["uni_cnt"].sum()
    num_all_reads = num_unique + multi_reads_num

    # aggregate prior
    if scaling_weight < 0 or scaling_weight > 1:
        raise ValueError("Weight must be 0-1")
    prior = np.exp(
        np.log(pairs["prior"].values) * scaling_weight
        + (1 - scaling_weight) * np.log(pairs["cov_prior"].values)
    )
    prior = prior / prior.sum()
    pairs["prob"] = prior.copy()

    # get arrays
    m_pair_ind = multi["pair_ind"].values
    m_read_ind = multi["read_ind"].values
    m_Z = multi["Z"].values

    prob = pairs["prob"].values
    uni_cnt = pairs["uni_cnt"].values
    num_pairs = len(prob)
    for iteration in tqdm(range(num_iter)):
        # update Z
        m_prob = prob[m_pair_ind]
        m_read_sum_prob = np.bincount(m_read_ind, weights=m_prob)
        m_read_sum_prob = m_read_sum_prob[m_read_ind]
        new_Z = m_prob / m_read_sum_prob
        select_change = np.sum(
            (new_Z - prob_threshold) * (m_Z - prob_threshold) < 0
        )
        m_Z = new_Z
        changes.append(select_change)

        # update parameteres
        m_read_sum_prob_rev = np.float32(1.0) / m_read_sum_prob
        pairs_rev_pi_sum = np.bincount(
            m_pair_ind, weights=m_read_sum_prob_rev, minlength=num_pairs
        )
        prob_new = (
            pairs_rev_pi_sum * prob
            + uni_cnt
            + num_all_reads * prior * prior_weight
        )
        prob = prob_new
        if select_change <= change_threshold:
            print(f"Finished at {iteration+1}")
            break

    multi["Z"] = m_Z
    pairs["prob"] = prob

    return multi, pairs, changes


def em_genes_iteration(possible_genes, multi_rna_genes, prob_threshold):
    multi_rna_genes = multi_rna_genes.merge(
        possible_genes[["gene_ind", "prob"]], how="left", on="gene_ind"
    )
    multi_rna_genes["read_sum_prob"] = multi_rna_genes.groupby("read_ind")[
        "prob"
    ].transform("sum")
    new_Z = multi_rna_genes["prob"] / multi_rna_genes["read_sum_prob"]
    select_change = (
        (new_Z - prob_threshold) * (multi_rna_genes["Z_cov"] - prob_threshold)
        < 0
    ).sum()
    multi_rna_genes["Z_cov"] = new_Z

    # update gene prob
    multi_rna_genes["read_sum_prob_rev"] = 1 / multi_rna_genes["read_sum_prob"]
    pairs_rev_pi_sum = (
        multi_rna_genes.groupby("gene_ind")["read_sum_prob_rev"]
        .sum()
        .reset_index()
    )
    possible_genes = possible_genes.merge(
        pairs_rev_pi_sum, how="left", on="gene_ind"
    )
    prob_new = (
        possible_genes["read_sum_prob_rev"] * possible_genes["prob"]
        + possible_genes["count"]
    ) / possible_genes["length"]
    diff = ((prob_new - possible_genes["prob"]) ** 2).sum()
    possible_genes["prob"] = prob_new
    possible_genes = possible_genes.drop("read_sum_prob_rev", axis=1)
    multi_rna_genes = multi_rna_genes.drop(
        ["prob", "read_sum_prob", "read_sum_prob_rev"], axis=1
    )
    return possible_genes, multi_rna_genes, diff, select_change


def em_multi_genes(
    multi_rna,
    multi_dna,
    unique,
    genes,
    n_iter,
    prob_threshold=0.5,
    verbose=True,
):
    uni_genes_cnt = (
        unique.groupby("gene_ind")["uni_cnt"]
        .sum()
        .reset_index()
        .rename({"count": "count_uni"}, axis=1)
    )
    multi_dna_genes_cnt = (
        multi_dna["gene_ind"]
        .value_counts()
        .reset_index()
        .rename({"count": "count_multi_dna"}, axis=1)
    )

    genes = (
        genes[["gene_ind", "gene_start", "gene_end"]]
        .merge(uni_genes_cnt, how="left", on="gene_ind")
        .merge(multi_dna_genes_cnt, how="left", on="gene_ind")
    )

    genes["uni_cnt"] = genes["uni_cnt"].fillna(0)
    genes["count_multi_dna"] = genes["count_multi_dna"].fillna(0)

    genes["count"] = genes["uni_cnt"] + genes["count_multi_dna"]
    genes["length"] = genes["gene_end"] - genes["gene_start"]
    possible_genes = (
        multi_rna[["gene_ind"]]
        .value_counts()
        .reset_index()
        .drop(["count"], axis=1)
    )
    possible_genes = possible_genes.merge(
        genes[["gene_ind", "count", "length"]], how="left", on="gene_ind"
    )
    possible_genes["count"] += 1
    possible_genes["prob"] = possible_genes["count"] / possible_genes["length"]

    multi_rna_genes = multi_rna[["read_id_short", "gene_ind"]]
    multi_rna_reads = (
        multi_rna[["read_id_short"]]
        .drop_duplicates()
        .reset_index(drop=True)
        .reset_index()
        .rename({"index": "read_ind"}, axis=1)
    )
    multi_rna_genes["Z_cov"] = 0
    multi_rna_genes = multi_rna_genes.merge(
        multi_rna_reads, how="left", on="read_id_short"
    ).drop("read_id_short", axis=1)

    for iter in range(n_iter):
        possible_genes, multi_rna_genes, diff, select_change = (
            em_genes_iteration(possible_genes, multi_rna_genes, prob_threshold)
        )
        if verbose:
            print(diff, select_change)

    multi_rna["Z_cov"] = multi_rna_genes["Z_cov"]
    return multi_rna


def em_bins_iteration(possible_bins, multi_dna_bins, prob_threshold):
    multi_dna_bins = multi_dna_bins.merge(
        possible_bins[["dna_chr", "dna_bin", "prob"]],
        how="left",
        on=["dna_chr", "dna_bin"],
    )
    print()
    multi_dna_bins["read_sum_prob"] = multi_dna_bins.groupby("read_ind")[
        "prob"
    ].transform("sum")
    new_Z = multi_dna_bins["prob"] / multi_dna_bins["read_sum_prob"]
    select_change = (
        (new_Z - prob_threshold) * (multi_dna_bins["Z_cov"] - prob_threshold)
        < 0
    ).sum()
    multi_dna_bins["Z_cov"] = new_Z

    # update gene prob
    multi_dna_bins["read_sum_prob_rev"] = 1 / multi_dna_bins["read_sum_prob"]
    pairs_rev_pi_sum = (
        multi_dna_bins.groupby(["dna_chr", "dna_bin"])["read_sum_prob_rev"]
        .sum()
        .reset_index()
    )
    possible_bins = possible_bins.merge(
        pairs_rev_pi_sum, how="left", on=["dna_chr", "dna_bin"]
    )
    prob_new = (
        possible_bins["read_sum_prob_rev"] * possible_bins["prob"]
        + possible_bins["count"]
    )

    diff = ((prob_new - possible_bins["prob"]) ** 2).sum()
    possible_bins["prob"] = prob_new
    possible_bins = possible_bins.drop("read_sum_prob_rev", axis=1)
    multi_dna_bins = multi_dna_bins.drop(
        ["prob", "read_sum_prob", "read_sum_prob_rev"], axis=1
    )
    return possible_bins, multi_dna_bins, diff, select_change


def em_multi_bins(
    multi_rna,
    multi_dna,
    unique,
    bins,
    n_iter,
    prob_threshold=0.5,
    verbose=True,
    dist_threshold=5_000_000,
):
    unique_far = unique[
        (unique["rna_chr"] != unique["dna_chr"])
        | ((unique["rna_bin"] - unique["dna_bin"]).abs() > dist_threshold)
    ]
    multi_rna_far = multi_rna[
        (multi_rna["rna_chr"] != multi_rna["dna_chr"])
        | (
            (multi_rna["rna_bin"] - multi_rna["dna_bin"]).abs()
            > dist_threshold
        )
    ]
    uni_bins_cnt = (
        unique_far.groupby(["dna_chr", "dna_bin"])["uni_cnt"]
        .sum()
        .reset_index()
        .rename({"count": "count_uni"}, axis=1)
    )
    multi_rna_bins_cnt = (
        multi_rna_far[["read_id_short", "dna_chr", "dna_bin"]]
        .drop_duplicates()[["dna_chr", "dna_bin"]]
        .value_counts()
        .reset_index()
        .rename({"count": "count_multi_dna"}, axis=1)
    )
    bins = bins.rename({"chrom": "dna_chr", "start": "dna_bin"}, axis=1)
    bins = (
        bins[["dna_chr", "dna_bin"]]
        .merge(uni_bins_cnt, how="left", on=["dna_chr", "dna_bin"])
        .merge(multi_rna_bins_cnt, how="left", on=["dna_chr", "dna_bin"])
    )

    bins["uni_cnt"] = bins["uni_cnt"].fillna(0)
    bins["count_multi_dna"] = bins["count_multi_dna"].fillna(0)
    bins["count"] = bins["uni_cnt"] + bins["count_multi_dna"]
    possible_bins = (
        multi_dna[["dna_chr", "dna_bin"]]
        .value_counts()
        .reset_index()
        .drop(["count"], axis=1)
    )
    possible_bins = possible_bins.merge(
        bins[["dna_chr", "dna_bin", "count"]],
        how="left",
        on=["dna_chr", "dna_bin"],
    )
    possible_bins["count"] += 1
    possible_bins["prob"] = possible_bins["count"]
    multi_dna_bins = multi_dna[["read_id_short", "dna_chr", "dna_bin"]]
    multi_dna_reads = (
        multi_dna[["read_id_short"]]
        .drop_duplicates()
        .reset_index(drop=True)
        .reset_index()
        .rename({"index": "read_ind"}, axis=1)
    )
    multi_dna_bins["Z_cov"] = 0
    multi_dna_bins = multi_dna_bins.merge(
        multi_dna_reads, how="left", on="read_id_short"
    ).drop("read_id_short", axis=1)

    for iter in range(n_iter):
        possible_bins, multi_dna_bins, diff, select_change = em_bins_iteration(
            possible_bins, multi_dna_bins, prob_threshold
        )
        if verbose:
            print(diff, select_change)

    multi_dna["Z_cov"] = multi_dna_bins["Z_cov"]
    return multi_dna
