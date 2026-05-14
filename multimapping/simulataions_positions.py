from typing import Literal

import numpy as np
import pandas as pd
import scipy


def get_multi_coverage_positions(
    multi_rna: pd.DataFrame,
    multi_dna: pd.DataFrame,
    mode: Literal["uniform", "coverage"],
    bin_size,
    min_multi_cnt=10,
    dist_threshold=5_000_000,
):
    # calc proportion of multi RNA reads
    prop_multi_rna = len(multi_rna["read_id_short"].unique()) / (
        len(multi_rna["read_id_short"].unique())
        + len(multi_dna["read_id_short"].unique())
    )

    # calc distr of map cnts per read
    dna_multi_per_read = (
        multi_dna["read_id_short"]
        .value_counts()
        .value_counts(normalize=True)
        .reset_index()
    )
    rna_multi_per_read = (
        multi_rna["read_id_short"]
        .value_counts()
        .value_counts(normalize=True)
        .reset_index()
    )

    # -1 for real position which we add later
    dna_multi_per_read["count"] = dna_multi_per_read["count"] - 1
    rna_multi_per_read["count"] = rna_multi_per_read["count"] - 1
    # multi_rna["dna_bin"] = (
    #     (((multi_rna["dna_start"] + multi_rna["dna_end"]) // 2) // bin_size) * bin_size
    # ).astype("int32")
    multi_rna["rna_bin"] = (
        (((multi_rna["rna_start"] + multi_rna["rna_end"]) // 2) // bin_size)
        * bin_size
    ).astype("int32")
    multi_dna["dna_bin"] = (
        (((multi_dna["dna_start"] + multi_dna["dna_end"]) // 2) // bin_size)
        * bin_size
    ).astype("int32")
    multi_dna["rna_bin"] = (
        (((multi_dna["rna_start"] + multi_dna["rna_end"]) // 2) // bin_size)
        * bin_size
    ).astype("int32")
    # calc genome coverage or set to uniform cov
    dna_multi_cov = multi_dna[["dna_chr", "dna_bin"]].drop_duplicates()
    dna_multi_cov.reset_index(drop=True, inplace=True)
    rna_multi_cov = multi_rna[
        ["rna_chr", "rna_bin", "rna_strand"]
    ].drop_duplicates()
    rna_multi_cov.reset_index(drop=True, inplace=True)
    if mode == "coverage":
        # far_multi_rna = multi_rna[
        #     (multi_rna["dna_chr"] != multi_rna["rna_chr"])
        #     | (
        #         np.abs(multi_rna["rna_bin"] - multi_rna["dna_bin"])
        #         > dist_threshold
        #     )
        # ]
        far_multi_rna = multi_rna

        far_multi_dna = multi_dna[
            (multi_dna["dna_chr"] != multi_dna["rna_chr"])
            | (
                np.abs(multi_dna["rna_bin"] - multi_dna["dna_bin"])
                > dist_threshold
            )
        ]

        far_multi_rna_gt_reads = (
            far_multi_rna["read_id_short"]
            .value_counts()[
                far_multi_rna["read_id_short"].value_counts() >= min_multi_cnt
            ]
            .index
        )

        far_multi_dna_gt_reads = (
            far_multi_dna["read_id_short"]
            .value_counts()[
                far_multi_dna["read_id_short"].value_counts() >= min_multi_cnt
            ]
            .index
        )

        filtered_multi_rna = far_multi_rna[
            far_multi_rna["read_id_short"].isin(far_multi_rna_gt_reads)
        ]
        filtered_multi_rna.reset_index(drop=True, inplace=True)

        filtered_multi_dna = far_multi_dna[
            far_multi_dna["read_id_short"].isin(far_multi_dna_gt_reads)
        ]
        filtered_multi_dna.reset_index(drop=True, inplace=True)

        dna_multi_cov_filtered = (
            filtered_multi_dna[["dna_chr", "dna_bin"]]
            .value_counts(normalize=False)
            .reset_index()
        ).rename({"count": "proportion"}, axis=1)
        rna_multi_cov_filtered = (
            filtered_multi_rna[["rna_chr", "rna_bin", "rna_strand"]]
            .value_counts(normalize=False)
            .reset_index()
        ).rename({"count": "proportion"}, axis=1)

        dna_multi_cov = dna_multi_cov.merge(
            dna_multi_cov_filtered, on=["dna_chr", "dna_bin"], how="left"
        )

        rna_multi_cov = rna_multi_cov.merge(
            rna_multi_cov_filtered,
            on=["rna_chr", "rna_bin", "rna_strand"],
            how="left",
        )
        dna_multi_cov["proportion"] = dna_multi_cov["proportion"].fillna(1)
        rna_multi_cov["proportion"] = rna_multi_cov["proportion"].fillna(1)
        dna_multi_cov["proportion"] /= dna_multi_cov["proportion"].sum()
        rna_multi_cov["proportion"] /= rna_multi_cov["proportion"].sum()
    else:
        dna_multi_cov["proportion"] = 1 / dna_multi_cov.shape[0]
        rna_multi_cov["proportion"] = 1 / rna_multi_cov.shape[0]

    return (
        dna_multi_cov,
        rna_multi_cov,
        dna_multi_per_read,
        rna_multi_per_read,
        prop_multi_rna,
    )


# def add_fake_mappers_positions(
#     contacts_multi_sim,
#     dna_multi_cov,
#     rna_multi_cov,
#     mean_multi_rna_len,
#     mean_multi_dna_len,
#     fake_prop,
#     bin_size,
#     random_state,
# ):
#     np.random.seed(random_state * 42)
#     fake_num = int(contacts_multi_sim.shape[0] * fake_prop)
#     fake_idxs = np.random.choice(
#         contacts_multi_sim.shape[0],
#         size=fake_num,
#         replace=False,
#     )
#     contacts_multi_sim.loc[fake_idxs, "label"] = False
#     fake_rna_parts_num = (
#         ~contacts_multi_sim["label"] & contacts_multi_sim["multi_rna"]
#     ).sum()

#     fake_dna_parts_num = (
#         (~contacts_multi_sim["label"]) & (~contacts_multi_sim["multi_rna"])
#     ).sum()

#     # sampling index
#     dna_fake_sample_index = np.random.choice(
#         dna_multi_cov.index,
#         size=fake_dna_parts_num,
#         p=dna_multi_cov["proportion"],
#         replace=True,
#     )
#     rna_fake_sample_index = np.random.choice(
#         rna_multi_cov.index,
#         size=fake_rna_parts_num,
#         p=rna_multi_cov["proportion"],
#         replace=True,
#     )

#     # sampling dna and rna multi parts
#     dna_fake_sample = (
#         dna_multi_cov[["dna_chr", "dna_bin"]]
#         .loc[dna_fake_sample_index, :]
#         .reset_index(drop=True)
#     )
#     rna_fake_sample = (
#         rna_multi_cov[["rna_chr", "rna_bin", "rna_strand"]]
#         .loc[rna_fake_sample_index, :]
#         .reset_index(drop=True)
#     )

#     # adding fake coordinates for the fake_prop proporion

#     fake_multi_rna_mask = ~contacts_multi_sim["label"] & contacts_multi_sim["multi_rna"]
#     fake_multi_dna_mask = ~contacts_multi_sim["label"] & ~contacts_multi_sim["multi_rna"]
#     contacts_multi_sim.loc[
#         fake_multi_rna_mask,
#         ["rna_chr", "rna_bin", "rna_strand"],
#     ] = rna_fake_sample[["rna_chr", "rna_bin", "rna_strand"]].values

#     contacts_multi_sim.loc[fake_multi_rna_mask, "gene_ind"] = -1

#     contacts_multi_sim.loc[
#         fake_multi_dna_mask,
#         ["dna_chr", "dna_bin"],
#     ] = dna_fake_sample[["dna_chr", "dna_bin"]].values

#     # sample positions
#     multi_dna_starts = scipy.stats.randint.rvs(
#         0, bin_size, size=fake_dna_parts_num, random_state=random_state * 422
#     )
#     multi_rna_starts = scipy.stats.randint.rvs(
#         0,
#         bin_size,
#         size=fake_rna_parts_num,
#         random_state=random_state * 42,
#     )

#     contacts_multi_sim.loc[fake_multi_dna_mask, "dna_start"] = (
#         contacts_multi_sim.loc[fake_multi_dna_mask, "dna_bin"] + multi_dna_starts
#     )
#     contacts_multi_sim.loc[fake_multi_dna_mask, "dna_end"] = (
#         contacts_multi_sim.loc[fake_multi_dna_mask, "dna_start"] + mean_multi_dna_len
#     )

#     contacts_multi_sim.loc[fake_multi_rna_mask, "rna_start"] = (
#         contacts_multi_sim.loc[fake_multi_rna_mask, "rna_bin"] + multi_rna_starts
#     )
#     contacts_multi_sim.loc[fake_multi_rna_mask, "rna_end"] = (
#         contacts_multi_sim.loc[fake_multi_rna_mask, "rna_start"] + mean_multi_rna_len
#     )

#     return contacts_multi_sim


def simulate_multi_positions(
    unique_contacts,
    rna_multi_cov,
    dna_multi_cov,
    rna_multi_per_read,
    dna_multi_per_read,
    multi_stats_df,
    bin_size,
    multi_prop: float = 0.75,
    random_state=42,
):
    # np.random.seed(random_state)
    prop_multi_rna = multi_stats_df.loc[0, "prop_multi_rna"].item()
    mean_multi_rna_len = multi_stats_df.loc[0, "mean_multi_rna_len"].item()
    mean_multi_dna_len = multi_stats_df.loc[0, "mean_multi_dna_len"].item()
    # split contacts to multi and uni
    n_sample = int(len(unique_contacts) * multi_prop)
    contacts_multi_sim = unique_contacts.sample(
        n=n_sample, random_state=random_state * 8888, axis=0
    )

    unique_contacts.drop(contacts_multi_sim.index, inplace=True)
    unique_contacts.reset_index(drop=True, inplace=True)
    contacts_multi_sim.reset_index(drop=True, inplace=True)
    contacts_multi_sim["label"] = True

    # splitting multi to multi-rna and multi-dna
    np.random.seed(random_state)
    size_multi_rna = int(prop_multi_rna * contacts_multi_sim.shape[0])
    multi_rna_idxs = np.random.choice(
        contacts_multi_sim.shape[0],
        size=size_multi_rna,
        replace=False,
    )
    contacts_multi_sim["multi_rna"] = False
    contacts_multi_sim.loc[multi_rna_idxs, "multi_rna"] = True

    contacts_multi_sim["n_multi_sample"] = 0

    # binning
    # contacts_multi_sim["rna_bin"] = (
    #     ((contacts_multi_sim["rna_start"] + contacts_multi_sim["rna_end"]) // 2)
    #     // bin_size
    # ) * bin_size
    # contacts_multi_sim["dna_bin"] = (
    #     ((contacts_multi_sim["dna_start"] + contacts_multi_sim["dna_end"]) // 2)
    #     // bin_size
    # ) * bin_size

    # contacts_multi_sim = contacts_multi_sim.drop(
    #     [
    #         "gene_name",
    #     ],
    #     axis=1,
    # )

    # if fake_prop != 0:
    #     contacts_multi_sim = add_fake_mappers_positions(
    #         contacts_multi_sim,
    #         dna_multi_cov,
    #         rna_multi_cov,
    #         mean_multi_rna_len,
    #         mean_multi_dna_len,
    #         fake_prop,
    #         bin_size,
    #         random_state,
    #     )

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
    contacts_multi_sim["read_id_short"] = (contacts_multi_sim.index).astype(
        "int32"
    )

    # separate rna and dna parts
    sim_dna_parts_multi_rna = contacts_multi_sim.loc[
        contacts_multi_sim["multi_rna"],
        [
            "read_id_short",
            "dna_chr",
            "dna_start",
            "dna_end",
            "n_multi_sample",
        ],
    ]
    sim_dna_parts_multi_rna.reset_index(drop=True, inplace=True)

    sim_rna_parts_multi_dna = contacts_multi_sim.loc[
        ~contacts_multi_sim["multi_rna"],
        [
            "read_id_short",
            "rna_chr",
            "rna_start",
            "rna_end",
            "rna_strand",
            "n_multi_sample",
        ],
    ]
    sim_rna_parts_multi_dna.reset_index(drop=True, inplace=True)

    gt_sim_rna_parts_multi_rna = contacts_multi_sim.loc[
        contacts_multi_sim["multi_rna"],
        [
            "read_id_short",
            "rna_chr",
            "rna_start",
            "rna_end",
            "rna_strand",
            "label",
        ],
    ]
    gt_sim_rna_parts_multi_rna.reset_index(drop=True, inplace=True)

    gt_sim_dna_parts_multi_dna = contacts_multi_sim.loc[
        ~contacts_multi_sim["multi_rna"],
        [
            "read_id_short",
            "dna_chr",
            "dna_start",
            "dna_end",
            "label",
        ],
    ]
    gt_sim_dna_parts_multi_dna.reset_index(drop=True, inplace=True)

    # repeating rows sampled times
    sim_rna_parts_multi_rna = sim_dna_parts_multi_rna.loc[
        sim_dna_parts_multi_rna.index.repeat(
            sim_dna_parts_multi_rna["n_multi_sample"]
        ),
        ["read_id_short"],
    ]
    sim_rna_parts_multi_rna.reset_index(drop=True, inplace=True)

    sim_dna_parts_multi_dna = sim_rna_parts_multi_dna.loc[
        sim_rna_parts_multi_dna.index.repeat(
            sim_rna_parts_multi_dna["n_multi_sample"]
        ),
        [
            "read_id_short",
        ],
    ]
    sim_dna_parts_multi_dna.reset_index(drop=True, inplace=True)

    # sampling index
    np.random.seed(random_state)
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
    dna_multi_sample = dna_multi_cov[["dna_chr", "dna_bin"]].loc[
        dna_multi_sample_index, :
    ]
    dna_multi_sample.reset_index(drop=True, inplace=True)

    rna_multi_sample = rna_multi_cov[["rna_chr", "rna_bin", "rna_strand"]].loc[
        rna_multi_sample_index, :
    ]
    rna_multi_sample.reset_index(drop=True, inplace=True)

    # adding cols to futher positions sampling
    dna_multi_sample["dna_start"] = dna_multi_sample["dna_bin"]
    dna_multi_sample["dna_end"] = dna_multi_sample["dna_bin"]
    rna_multi_sample["rna_start"] = rna_multi_sample["rna_bin"]
    rna_multi_sample["rna_end"] = rna_multi_sample["rna_bin"]

    sim_rna_parts_multi_rna = pd.concat(
        [sim_rna_parts_multi_rna, rna_multi_sample], axis=1
    )
    # adding uni part and simulated
    sim_dna_parts_multi_dna = pd.concat(
        [sim_dna_parts_multi_dna, dna_multi_sample], axis=1
    )

    sim_rna_parts_multi_rna["label"] = False
    sim_dna_parts_multi_dna["label"] = False

    # adding col for ground truth gene_ind
    # contacts_multi_sim = contacts_multi_sim.rename({"gene_ind": "gene_ind_gt"}, axis=1)
    # contacts_multi_sim_rna_sampled["gene_ind_gt"] = -1
    # contacts_multi_sim_dna_sampled["gene_ind_gt"] = -1

    # sample positions
    multi_dna_starts = scipy.stats.randint.rvs(
        0, bin_size, size=dna_multi_sample_size, random_state=random_state * 8
    )
    multi_rna_starts = scipy.stats.randint.rvs(
        0,
        bin_size,
        size=rna_multi_sample_size,
        random_state=random_state * 777,
    )

    sim_rna_parts_multi_rna["rna_start"] = (
        sim_rna_parts_multi_rna["rna_start"] + multi_rna_starts
    )
    sim_rna_parts_multi_rna["rna_end"] = (
        sim_rna_parts_multi_rna["rna_start"] + mean_multi_rna_len
    )

    sim_dna_parts_multi_dna["dna_start"] = (
        sim_dna_parts_multi_dna["dna_start"] + multi_dna_starts
    )
    sim_dna_parts_multi_dna["dna_end"] = (
        sim_dna_parts_multi_dna["dna_start"] + mean_multi_dna_len
    )

    # union  of gt and simulated
    rna_parts_cols_order = [
        "read_id_short",
        "rna_chr",
        "rna_start",
        "rna_end",
        "rna_strand",
        "label",
    ]

    dna_parts_cols_order = [
        "read_id_short",
        "dna_chr",
        "dna_start",
        "dna_end",
        "label",
    ]

    sim_rna_parts_multi_rna = pd.concat(
        [
            sim_rna_parts_multi_rna[rna_parts_cols_order],
            gt_sim_rna_parts_multi_rna[rna_parts_cols_order],
        ]
    )

    sim_dna_parts_multi_dna = pd.concat(
        [
            sim_dna_parts_multi_dna[dna_parts_cols_order],
            gt_sim_dna_parts_multi_dna[dna_parts_cols_order],
        ]
    )

    # sort read-id
    sim_rna_parts_multi_rna = sim_rna_parts_multi_rna.sort_values(
        by="read_id_short"
    )
    sim_rna_parts_multi_rna.reset_index(drop=True, inplace=True)

    sim_dna_parts_multi_dna = sim_dna_parts_multi_dna.sort_values(
        by="read_id_short"
    )
    sim_dna_parts_multi_dna.reset_index(drop=True, inplace=True)

    return (
        sim_rna_parts_multi_rna,
        sim_dna_parts_multi_rna,
        sim_dna_parts_multi_dna,
        sim_rna_parts_multi_dna,
        unique_contacts,
    )
