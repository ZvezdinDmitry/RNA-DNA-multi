from pathlib import Path

import numpy as np
import pandas as pd


def create_fragments(
    chrom_sizes_file: str | Path, bin_size: int, chrom_list: None | list = None
) -> pd.DataFrame:
    """Returns bins DF with desired bin size.

    Args:
        chrom_sizes_file (str | Path): _description_
        bin_size (int): _description_
        chrom_list (None | list, optional): _description_. Defaults to None.

    Returns:
        pd.DataFrame: _description_
    """
    dtypes = {
        "chrom": "category",
        "size": "int64",
    }
    df_chromosomes = pd.read_csv(
        chrom_sizes_file,
        sep="\t",
        header=None,
        names=["chrom", "size"],
        dtype=dtypes,
    )
    fragments = []
    if chrom_list is None:
        chrom_list = df_chromosomes["chrom"].to_list()

    for chrom in chrom_list:
        size = df_chromosomes.loc[
            df_chromosomes["chrom"] == chrom, "size"
        ].values[0]

        bins_start = np.arange(start=0, stop=size, step=bin_size)
        bins_end = bins_start + bin_size

        fragments_chr = pd.DataFrame(
            {
                "chrom": chrom,
                "start": bins_start,
                "end": bins_end,
            }
        )
        fragments.append(fragments_chr)

    fragments = pd.concat(fragments, ignore_index=True)
    return fragments
