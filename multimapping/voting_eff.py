import bioframe as bf
import pandas as pd


def voting(unique, rna_parts_multi_dna, genes):
    # calc coverage
    unique["map_id"] = unique.index.astype("int32")
    merged = pd.concat(
        [
            unique[["rna_chr", "rna_start", "rna_end", "rna_strand"]],
            rna_parts_multi_dna[
                ["rna_chr", "rna_start", "rna_end", "rna_strand"]
            ],
        ],
        ignore_index=True,
    )

    merged = merged[["rna_chr", "rna_start", "rna_end", "rna_strand"]].rename(
        {"rna_strand": "strand"}, axis=1
    )
    merged_cov = bf.overlap(
        merged,
        genes[
            [
                "strand",
                "gene_chr",
                "gene_start",
                "gene_end",
                "gene_ind",
            ]
        ],
        how="inner",
        return_input=False,
        return_index=True,
        cols1=["rna_chr", "rna_start", "rna_end"],
        cols2=["gene_chr", "gene_start", "gene_end"],
        on=["strand"],
    )
    merged_cov = pd.concat(
        [
            merged.loc[merged_cov["index"], :].reset_index(drop=True),
            genes.loc[merged_cov["index_"], "gene_ind"].reset_index(drop=True),
        ],
        axis=1,
    )
    merged_cov = merged_cov["gene_ind"].value_counts().reset_index()
    genes = genes.merge(merged_cov, how="left", on="gene_ind")
    genes["count"] = genes["count"].fillna(0)
    genes["density"] = genes["count"] / (
        genes["gene_end"] - genes["gene_start"]
    )

    # voting unique
    unique_voted = bf.overlap(
        unique.rename({"rna_strand": "strand"}, axis=1),
        genes[
            [
                "strand",
                "gene_chr",
                "gene_start",
                "gene_end",
                "gene_ind",
            ]
        ],
        how="inner",
        return_input=False,
        return_index=True,
        cols1=["rna_chr", "rna_start", "rna_end"],
        cols2=["gene_chr", "gene_start", "gene_end"],
        on=["strand"],
    )

    unique_voted = pd.concat(
        [
            unique.loc[unique_voted["index"], :].reset_index(drop=True),
            genes.loc[
                unique_voted["index_"], ["gene_ind", "density"]
            ].reset_index(drop=True),
        ],
        axis=1,
    )

    unique_voted["max_density"] = unique_voted.groupby("map_id")[
        "density"
    ].transform("max")
    unique_voted = unique_voted[
        unique_voted["density"] >= unique_voted["max_density"]
    ].reset_index(drop=True)
    unique_voted = unique_voted.drop_duplicates(
        subset="map_id", ignore_index=True
    )
    unique_voted = unique_voted.drop(
        ["map_id", "density", "max_density"], axis=1
    ).rename({"strand": "rna_strand"}, axis=1)
    unique_voted["gene_ind"] = unique_voted["gene_ind"].astype("int")

    return unique_voted, genes


def bin_multi_rna(
    rna_parts_multi_rna,
    dna_parts_multi_rna,
    genes,
    bin_size,
    drop_positions=True,
):
    rna_parts_multi_rna["map_id"] = rna_parts_multi_rna.index.astype("int32")
    rna_parts_multi_rna = rna_parts_multi_rna.rename(
        {"rna_strand": "strand"}, axis=1
    )
    genes = genes[
        [
            "strand",
            "gene_chr",
            "gene_start",
            "gene_end",
            "gene_ind",
            "density",
        ]
    ]
    # voting multi RNA
    rna_parts_multi_rna_genes = bf.overlap(
        rna_parts_multi_rna,
        genes,
        how="inner",
        return_input=False,
        return_index=True,
        cols1=["rna_chr", "rna_start", "rna_end"],
        cols2=["gene_chr", "gene_start", "gene_end"],
        on=["strand"],
    )

    rna_parts_multi_rna_genes = pd.concat(
        [
            rna_parts_multi_rna.loc[
                rna_parts_multi_rna_genes["index"], :
            ].reset_index(drop=True),
            genes.loc[
                rna_parts_multi_rna_genes["index_"], ["gene_ind", "density"]
            ].reset_index(drop=True),
        ],
        axis=1,
    )

    rna_parts_multi_rna_genes["max_density"] = (
        rna_parts_multi_rna_genes.groupby("map_id")["density"].transform("max")
    )
    rna_parts_multi_rna_genes = rna_parts_multi_rna_genes[
        rna_parts_multi_rna_genes["density"]
        >= rna_parts_multi_rna_genes["max_density"]
    ].reset_index(drop=True)

    # binning
    rna_parts_multi_rna_genes["rna_bin"] = (
        (
            rna_parts_multi_rna_genes["rna_start"]
            + rna_parts_multi_rna_genes["rna_end"]
        )
        // 2
    ) // bin_size

    dna_parts_multi_rna["dna_bin"] = (
        (dna_parts_multi_rna["dna_start"] + dna_parts_multi_rna["dna_end"])
        // 2
    ) // bin_size

    rna_parts_multi_rna_genes["rna_bin"] = (
        rna_parts_multi_rna_genes["rna_bin"] * bin_size
    ).astype(int)
    dna_parts_multi_rna["dna_bin"] = (
        dna_parts_multi_rna["dna_bin"] * bin_size
    ).astype(int)

    if drop_positions:
        drop_cols_rna = [
            "rna_start",
            "rna_end",
            "map_id",
            "density",
            "max_density",
            "strand",
        ]
        drop_cols_dna = [
            "dna_start",
            "dna_end",
        ]

        rna_parts_multi_rna_genes = rna_parts_multi_rna_genes.drop(
            drop_cols_rna,
            axis=1,
        )
        dna_parts_multi_rna = dna_parts_multi_rna.drop(
            drop_cols_dna,
            axis=1,
        )
        # remove bin level "unique"
        drop_duplicates_cols = [
            "read_id_short",
            "gene_ind",
            "rna_chr",
            "rna_bin",
        ]

        rna_parts_multi_rna_genes = rna_parts_multi_rna_genes.drop_duplicates(
            subset=drop_duplicates_cols
        ).reset_index(drop=True)

        read_cnts = rna_parts_multi_rna_genes["read_id_short"].value_counts()
        unique_after_binning = rna_parts_multi_rna_genes[
            rna_parts_multi_rna_genes["read_id_short"].isin(
                read_cnts[read_cnts == 1].index
            )
        ].reset_index(drop=True)
        rna_parts_multi_rna_genes = rna_parts_multi_rna_genes[
            rna_parts_multi_rna_genes["read_id_short"].isin(
                read_cnts[read_cnts != 1].index
            )
        ].reset_index(drop=True)

        # merge each unique after binning to other half of coords
        unique_after_binning = unique_after_binning.merge(
            dna_parts_multi_rna[["read_id_short", "dna_chr", "dna_bin"]],
            how="left",
            on="read_id_short",
        )
        columns_order = [
            "read_id_short",
            "rna_chr",
            "rna_bin",
            "gene_ind",
            "dna_chr",
            "dna_bin",
        ]
        unique_after_binning = unique_after_binning[columns_order].dropna()

        # remove unique after binning reads
        dna_parts_multi_rna = dna_parts_multi_rna[
            dna_parts_multi_rna["read_id_short"].isin(
                set(rna_parts_multi_rna_genes["read_id_short"])
            )
        ].reset_index(drop=True)
    print(
        f"multi RNA: positions: {len(rna_parts_multi_rna_genes)}, reads: {len(dna_parts_multi_rna)}, unique after bin: {len(unique_after_binning)}"
    )
    return (
        rna_parts_multi_rna_genes,
        dna_parts_multi_rna,
        unique_after_binning,
    )


def bin_multi_dna(
    rna_parts_multi_dna,
    dna_parts_multi_dna,
    genes,
    bin_size,
    drop_positions=True,
):
    rna_parts_multi_dna["map_id"] = rna_parts_multi_dna.index.astype("int32")
    rna_parts_multi_dna = rna_parts_multi_dna.rename(
        {"rna_strand": "strand"}, axis=1
    )
    genes = genes[
        [
            "strand",
            "gene_chr",
            "gene_start",
            "gene_end",
            "gene_ind",
            "density",
        ]
    ]
    # voting multi RNA
    rna_parts_multi_dna_genes = bf.overlap(
        rna_parts_multi_dna,
        genes,
        how="inner",
        return_input=False,
        return_index=True,
        cols1=["rna_chr", "rna_start", "rna_end"],
        cols2=["gene_chr", "gene_start", "gene_end"],
        on=["strand"],
    )

    rna_parts_multi_dna_genes = pd.concat(
        [
            rna_parts_multi_dna.loc[
                rna_parts_multi_dna_genes["index"], :
            ].reset_index(drop=True),
            genes.loc[
                rna_parts_multi_dna_genes["index_"], ["gene_ind", "density"]
            ].reset_index(drop=True),
        ],
        axis=1,
    )

    rna_parts_multi_dna_genes["max_density"] = (
        rna_parts_multi_dna_genes.groupby("map_id")["density"].transform("max")
    )
    rna_parts_multi_dna_genes = rna_parts_multi_dna_genes[
        rna_parts_multi_dna_genes["density"]
        >= rna_parts_multi_dna_genes["max_density"]
    ].reset_index(drop=True)

    # binning
    rna_parts_multi_dna_genes["rna_bin"] = (
        (
            rna_parts_multi_dna_genes["rna_start"]
            + rna_parts_multi_dna_genes["rna_end"]
        )
        // 2
    ) // bin_size

    dna_parts_multi_dna["dna_bin"] = (
        (dna_parts_multi_dna["dna_start"] + dna_parts_multi_dna["dna_end"])
        // 2
    ) // bin_size

    rna_parts_multi_dna_genes["rna_bin"] = (
        rna_parts_multi_dna_genes["rna_bin"] * bin_size
    ).astype(int)
    dna_parts_multi_dna["dna_bin"] = (
        dna_parts_multi_dna["dna_bin"] * bin_size
    ).astype(int)
    if drop_positions:
        drop_cols_rna = [
            "rna_start",
            "rna_end",
            "map_id",
            "density",
            "max_density",
            "strand",
        ]
        drop_cols_dna = [
            "dna_start",
            "dna_end",
        ]

        rna_parts_multi_dna_genes = rna_parts_multi_dna_genes.drop(
            drop_cols_rna,
            axis=1,
        )
        dna_parts_multi_dna = dna_parts_multi_dna.drop(
            drop_cols_dna,
            axis=1,
        )
        # remove bin level "unique"
        drop_duplicates_cols = [
            "read_id_short",
            "dna_chr",
            "dna_bin",
        ]

        dna_parts_multi_dna = dna_parts_multi_dna.drop_duplicates(
            subset=drop_duplicates_cols
        ).reset_index(drop=True)

        read_cnts = dna_parts_multi_dna["read_id_short"].value_counts()
        unique_after_binning = dna_parts_multi_dna[
            dna_parts_multi_dna["read_id_short"].isin(
                read_cnts[read_cnts == 1].index
            )
        ].reset_index(drop=True)
        dna_parts_multi_dna = dna_parts_multi_dna[
            dna_parts_multi_dna["read_id_short"].isin(
                read_cnts[read_cnts != 1].index
            )
        ].reset_index(drop=True)

        # remove parts which does not intersect genes
        dna_parts_multi_dna = dna_parts_multi_dna[
            dna_parts_multi_dna["read_id_short"].isin(
                rna_parts_multi_dna_genes["read_id_short"]
            )
        ].reset_index(drop=True)
        # merge each unique after binning to other half of coords
        unique_after_binning = unique_after_binning.merge(
            rna_parts_multi_dna_genes[
                ["read_id_short", "rna_chr", "rna_bin", "gene_ind"]
            ],
            how="left",
            on="read_id_short",
        )
        columns_order = [
            "read_id_short",
            "rna_chr",
            "rna_bin",
            "gene_ind",
            "dna_chr",
            "dna_bin",
        ]
        unique_after_binning = unique_after_binning[columns_order].dropna()

        # remove unique after binning reads
        rna_parts_multi_dna_genes = rna_parts_multi_dna_genes[
            rna_parts_multi_dna_genes["read_id_short"].isin(
                set(dna_parts_multi_dna["read_id_short"])
            )
        ].reset_index(drop=True)
        print(
            f"multi DNA: positions: {len(dna_parts_multi_dna)}, reads: {len(rna_parts_multi_dna_genes)}, unique after bin: {len(unique_after_binning)}"
        )

    return (
        dna_parts_multi_dna,
        rna_parts_multi_dna_genes,
        unique_after_binning,
    )
