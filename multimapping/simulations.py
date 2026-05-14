from typing import Literal

import numpy as np
import pandas as pd


def get_multi_coverage(
    multi_rna: pd.DataFrame,
    multi_dna: pd.DataFrame,
    mode: Literal["uniform", "coverage"],
    min_multi_cnt=10,
    dist_threshold=5_000_000,
):
    # calc proportion of multi RNA reads
    prop_multi_rna = len(multi_rna["read_id"].unique()) / (
        len(multi_rna["read_id"].unique()) + len(multi_dna["read_id"].unique())
    )

    # calc distr of map cnts per read
    dna_multi_per_read = (
        multi_dna["read_id"]
        .value_counts()
        .value_counts(normalize=True)
        .reset_index()
    )
    rna_multi_per_read = (
        multi_rna["read_id"]
        .value_counts()
        .value_counts(normalize=True)
        .reset_index()
    )

    # -1 for real position which we add later
    dna_multi_per_read["count"] = dna_multi_per_read["count"] - 1
    rna_multi_per_read["count"] = rna_multi_per_read["count"] - 1

    # calc genome coverage or set to uniform cov
    if mode == "coverage":
        far_multi_rna = multi_rna[
            (multi_rna["dna_chr"] != multi_rna["rna_chr"])
            | (
                np.abs(multi_rna["rna_bin"] - multi_rna["dna_bin"])
                > dist_threshold
            )
        ]

        far_multi_dna = multi_dna[
            (multi_dna["dna_chr"] != multi_dna["rna_chr"])
            | (
                np.abs(multi_dna["rna_bin"] - multi_dna["dna_bin"])
                > dist_threshold
            )
        ]

        far_multi_rna_gt_reads = (
            far_multi_rna["read_id"]
            .value_counts()[
                far_multi_rna["read_id"].value_counts() >= min_multi_cnt
            ]
            .index
        )

        far_multi_dna_gt_reads = (
            far_multi_dna["read_id"]
            .value_counts()[
                far_multi_dna["read_id"].value_counts() >= min_multi_cnt
            ]
            .index
        )

        filtered_multi_rna = far_multi_rna[
            far_multi_rna["read_id"].isin(far_multi_rna_gt_reads)
        ].reset_index(drop=True)

        filtered_multi_dna = far_multi_dna[
            far_multi_dna["read_id"].isin(far_multi_dna_gt_reads)
        ].reset_index(drop=True)

        dna_multi_cov = (
            filtered_multi_dna[["dna_chr", "dna_bin"]]
            .value_counts(normalize=True)
            .reset_index()
        )
        rna_multi_cov = (
            filtered_multi_rna[["rna_chr", "rna_bin", "gene_ind"]]
            .value_counts(normalize=True)
            .reset_index()
        )

    else:
        dna_multi_cov = (
            multi_dna[["dna_chr", "dna_bin"]].drop_duplicates().reset_index()
        )
        rna_multi_cov = (
            multi_rna[["rna_chr", "rna_bin", "gene_ind"]]
            .drop_duplicates()
            .reset_index()
        )
        dna_multi_cov["proportion"] = 1 / dna_multi_cov.shape[0]
        rna_multi_cov["proportion"] = 1 / rna_multi_cov.shape[0]

    return (
        dna_multi_cov,
        rna_multi_cov,
        dna_multi_per_read,
        rna_multi_per_read,
        prop_multi_rna,
    )


def add_fake_mappers(
    contacts_multi_sim, dna_multi_cov, rna_multi_cov, fake_prop, random_state
):
    np.random.seed(random_state * 42)
    fake_num = int(contacts_multi_sim.shape[0] * fake_prop)
    fake_idxs = np.random.choice(
        contacts_multi_sim.shape[0],
        size=fake_num,
        replace=False,
    )
    contacts_multi_sim.loc[fake_idxs, "label"] = False
    fake_rna_parts_num = (
        ~contacts_multi_sim["label"] & contacts_multi_sim["multi_rna"]
    ).sum()

    fake_dna_parts_num = (
        (~contacts_multi_sim["label"]) & (~contacts_multi_sim["multi_rna"])
    ).sum()

    # sampling index
    dna_fake_sample_index = np.random.choice(
        dna_multi_cov.index,
        size=fake_dna_parts_num,
        p=dna_multi_cov["proportion"],
        replace=True,
    )
    rna_fake_sample_index = np.random.choice(
        rna_multi_cov.index,
        size=fake_rna_parts_num,
        p=rna_multi_cov["proportion"],
        replace=True,
    )

    # sampling dna and rna multi parts
    dna_fake_sample = (
        dna_multi_cov[["dna_chr", "dna_bin"]]
        .loc[dna_fake_sample_index, :]
        .reset_index(drop=True)
    )
    rna_fake_sample = (
        rna_multi_cov[["rna_chr", "rna_bin", "gene_ind"]]
        .loc[rna_fake_sample_index, :]
        .reset_index(drop=True)
    )

    # adding fake coordinates for the fake_prop proporion

    contacts_multi_sim.loc[
        ~contacts_multi_sim["label"] & contacts_multi_sim["multi_rna"],
        ["rna_chr", "rna_bin", "gene_ind"],
    ] = rna_fake_sample[["rna_chr", "rna_bin", "gene_ind"]].values

    contacts_multi_sim.loc[
        ~contacts_multi_sim["label"] & ~contacts_multi_sim["multi_rna"],
        ["dna_chr", "dna_bin"],
    ] = dna_fake_sample[["dna_chr", "dna_bin"]].values

    return contacts_multi_sim


def simulate_multi(
    unique_contacts,
    multi_rna,
    multi_dna,
    mode: Literal["uniform", "coverage"],
    bin_size,
    multi_prop: float = 0.75,
    fake_prop: float = 0,
    min_multi_cnt: int = 10,
    dist_threshold: int = 5000000,
    random_state=42,
):
    np.random.seed(random_state)

    # split contacts to multi and uni
    n_sample = int(len(unique_contacts) * multi_prop)
    contacts_multi_sim = unique_contacts.sample(
        n=n_sample, random_state=random_state, axis=0
    )

    unique_contacts = unique_contacts.drop(
        contacts_multi_sim.index
    ).reset_index(drop=True)
    contacts_multi_sim = contacts_multi_sim.reset_index(drop=True)
    contacts_multi_sim["label"] = True

    # get multi mappers statistics
    (
        dna_multi_cov,
        rna_multi_cov,
        dna_multi_per_read,
        rna_multi_per_read,
        prop_multi_rna,
    ) = get_multi_coverage(
        multi_rna, multi_dna, mode, min_multi_cnt, dist_threshold
    )

    # splitting multi to multi-rna and multi-dna
    size_multi_rna = int(prop_multi_rna * contacts_multi_sim.shape[0])
    multi_rna_idxs = np.random.choice(
        contacts_multi_sim.shape[0], size=size_multi_rna, replace=False
    )
    contacts_multi_sim["multi_rna"] = False
    contacts_multi_sim.loc[multi_rna_idxs, "multi_rna"] = True

    contacts_multi_sim["n_multi_sample"] = 0

    # binning
    contacts_multi_sim["rna_bin"] = (
        (
            (contacts_multi_sim["rna_start"] + contacts_multi_sim["rna_end"])
            // 2
        )
        // bin_size
    ) * bin_size
    contacts_multi_sim["dna_bin"] = (
        (
            (contacts_multi_sim["dna_start"] + contacts_multi_sim["dna_end"])
            // 2
        )
        // bin_size
    ) * bin_size

    contacts_multi_sim = contacts_multi_sim.drop(
        [
            "rna_start",
            "rna_end",
            "rna_strand",
            "dna_start",
            "dna_end",
            "gene_name",
        ],
        axis=1,
    )

    if fake_prop != 0:
        contacts_multi_sim = add_fake_mappers(
            contacts_multi_sim,
            dna_multi_cov,
            rna_multi_cov,
            fake_prop,
            random_state,
        )

    # choose map cnts for each read
    dna_sim_map_cnts = np.random.choice(
        dna_multi_per_read["count"],
        size=contacts_multi_sim.shape[0] - size_multi_rna,
        p=dna_multi_per_read["proportion"],
        replace=True,
    )
    rna_sim_map_cnts = np.random.choice(
        rna_multi_per_read["count"],
        size=size_multi_rna,
        p=rna_multi_per_read["proportion"],
        replace=True,
    )
    contacts_multi_sim.loc[
        ~contacts_multi_sim["multi_rna"], "n_multi_sample"
    ] = dna_sim_map_cnts
    contacts_multi_sim.loc[
        contacts_multi_sim["multi_rna"], "n_multi_sample"
    ] = rna_sim_map_cnts

    # calculation full sim multi mappers size
    rna_multi_sample_size = contacts_multi_sim.loc[
        contacts_multi_sim["multi_rna"], "n_multi_sample"
    ].sum()

    dna_multi_sample_size = contacts_multi_sim.loc[
        ~contacts_multi_sim["multi_rna"], "n_multi_sample"
    ].sum()

    # simple read-id as index
    contacts_multi_sim["read_id"] = contacts_multi_sim.index

    # separate rna and dna parts
    contacts_multi_sim_rna = contacts_multi_sim.loc[
        contacts_multi_sim["multi_rna"],
        ["read_id", "dna_chr", "dna_bin", "n_multi_sample"],
    ].reset_index(drop=True)

    contacts_multi_sim_dna = contacts_multi_sim.loc[
        ~contacts_multi_sim["multi_rna"],
        ["read_id", "rna_chr", "rna_bin", "gene_ind", "n_multi_sample"],
    ].reset_index(drop=True)

    # repeating rows sampled times
    contacts_multi_sim_rna_sampled = contacts_multi_sim_rna.loc[
        contacts_multi_sim_rna.index.repeat(
            contacts_multi_sim_rna["n_multi_sample"]
        ),
        ["read_id", "dna_chr", "dna_bin"],
    ].reset_index(drop=True)

    contacts_multi_sim_dna_sampled = contacts_multi_sim_dna.loc[
        contacts_multi_sim_dna.index.repeat(
            contacts_multi_sim_dna["n_multi_sample"]
        ),
        ["read_id", "rna_chr", "rna_bin", "gene_ind"],
    ].reset_index(drop=True)

    # sampling index
    np.random.seed(42)
    dna_multi_sample_index = np.random.choice(
        dna_multi_cov.index,
        size=dna_multi_sample_size,
        p=dna_multi_cov["proportion"],
        replace=True,
    )
    rna_multi_sample_index = np.random.choice(
        rna_multi_cov.index,
        size=rna_multi_sample_size,
        p=rna_multi_cov["proportion"],
        replace=True,
    )

    # sampling dna and rna multi parts
    dna_multi_sample = (
        dna_multi_cov[["dna_chr", "dna_bin"]]
        .loc[dna_multi_sample_index, :]
        .reset_index(drop=True)
    )
    rna_multi_sample = (
        rna_multi_cov[["rna_chr", "rna_bin", "gene_ind"]]
        .loc[rna_multi_sample_index, :]
        .reset_index(drop=True)
    )

    # adding uni part and simulated
    contacts_multi_sim_dna_sampled = pd.concat(
        [contacts_multi_sim_dna_sampled, dna_multi_sample], axis=1
    )

    contacts_multi_sim_rna_sampled = pd.concat(
        [contacts_multi_sim_rna_sampled, rna_multi_sample], axis=1
    )

    # union of GT and simulated
    contacts_multi_sim_rna_sampled["label"] = False
    contacts_multi_sim_dna_sampled["label"] = False

    contacts_multi_sim_rna_sampled = pd.concat(
        [
            contacts_multi_sim_rna_sampled[
                [
                    "read_id",
                    "gene_ind",
                    "rna_chr",
                    "rna_bin",
                    "dna_chr",
                    "dna_bin",
                    "label",
                ]
            ],
            contacts_multi_sim.loc[
                contacts_multi_sim["multi_rna"],
                [
                    "read_id",
                    "gene_ind",
                    "rna_chr",
                    "rna_bin",
                    "dna_chr",
                    "dna_bin",
                    "label",
                ],
            ],
        ]
    )

    contacts_multi_sim_dna_sampled = pd.concat(
        [
            contacts_multi_sim_dna_sampled[
                [
                    "read_id",
                    "gene_ind",
                    "rna_chr",
                    "rna_bin",
                    "dna_chr",
                    "dna_bin",
                    "label",
                ]
            ],
            contacts_multi_sim.loc[
                ~contacts_multi_sim["multi_rna"],
                [
                    "read_id",
                    "gene_ind",
                    "rna_chr",
                    "rna_bin",
                    "dna_chr",
                    "dna_bin",
                    "label",
                ],
            ],
        ]
    )

    # sort read-id
    contacts_multi_sim_dna_sampled = (
        contacts_multi_sim_dna_sampled.sort_values(by="read_id").reset_index(
            drop=True
        )
    )
    contacts_multi_sim_rna_sampled = (
        contacts_multi_sim_rna_sampled.sort_values(by="read_id").reset_index(
            drop=True
        )
    )

    return (
        contacts_multi_sim_rna_sampled,
        contacts_multi_sim_dna_sampled,
        unique_contacts,
    )


def bin_unique(unique_contacts, bin_size):
    # binning rna bin - dna bin
    unique_contacts["rna_bin"] = (
        ((unique_contacts["rna_start"] + unique_contacts["rna_end"]) // 2)
        // bin_size
    ) * bin_size
    unique_contacts["dna_bin"] = (
        ((unique_contacts["dna_start"] + unique_contacts["dna_end"]) // 2)
        // bin_size
    ) * bin_size

    bin_cnts = (
        unique_contacts[["rna_chr", "dna_chr", "rna_bin", "dna_bin"]]
        .value_counts()
        .reset_index()
        .sort_values(by=["rna_chr", "dna_chr", "rna_bin", "dna_bin"])
        .reset_index(drop=True)
    )

    bin_cnts = bin_cnts[["rna_chr", "rna_bin", "dna_chr", "dna_bin", "count"]]

    # binning rna gene - dna bin
    gene_cnts = (
        unique_contacts[["gene_ind", "rna_chr", "dna_chr", "dna_bin"]]
        .value_counts()
        .reset_index()
        .sort_values(by=["rna_chr", "dna_chr", "gene_ind", "dna_bin"])
        .reset_index(drop=True)
    )
    gene_cnts = gene_cnts[
        ["gene_ind", "rna_chr", "dna_chr", "dna_bin", "count"]
    ]

    return bin_cnts, gene_cnts
