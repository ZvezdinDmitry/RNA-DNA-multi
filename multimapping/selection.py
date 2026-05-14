import numpy as np
import pandas as pd


def calculate_top1_confidence(df: pd.DataFrame) -> pd.DataFrame:
    df.sort_values(
        by=["read_ind", "Z"],
        ascending=[True, False],
        inplace=True,
        ignore_index=True,
    )

    group_ids = df["read_ind"].values
    z_values = df["Z"].values

    is_first = np.empty(len(group_ids), dtype=bool)
    is_first[0] = True
    is_first[1:] = group_ids[1:] != group_ids[:-1]

    top1_idx = np.where(is_first)[0]

    top2_idx = top1_idx + 1

    confidence = z_values[top1_idx] / z_values[top2_idx]
    df["confidence"] = 0
    df.loc[top1_idx, "confidence"] = confidence

    return df


def sample_with_weights(df, temp=1):
    np.random.seed(424242424)
    noise = np.random.exponential(size=len(df)).astype(np.float32)
    df["temp_score"] = df["Z"].values ** (1 / temp)
    df["temp_score"] = df["temp_score"] / (noise + 1e-10)
    df.sort_values(["read_ind", "temp_score"], inplace=True)

    group_ids = df["read_ind"].values
    is_last_in_group = np.empty(len(df), dtype=bool)
    is_last_in_group[:-1] = group_ids[:-1] != group_ids[1:]
    is_last_in_group[-1] = True

    df["pred"] = is_last_in_group
    df.drop(columns=["temp_score"], inplace=True)

    return df
