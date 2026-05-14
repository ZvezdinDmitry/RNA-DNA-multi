import re

import pandas as pd


def filter_multi_df(df: pd.DataFrame, filter_chr=None) -> pd.DataFrame:
    """Remove unmapped and filter chrs subset, if provided.

    Args:
        df (pd.DataFrame): Contacts
        filter_chr (_type_, optional): Chrs to filter. Defaults to None.

    Returns:
        pd.DataFrame: Filtered
    """
    df = df[~df["ATA_pairtype"].str.contains("N")].reset_index(drop=True)
    if filter_chr is not None:
        df = df[
            (df["dna_chr"].isin(filter_chr)) & (df["rna_chr"].isin(filter_chr))
        ].reset_index(drop=True)

    return df


def process_secondary(row: list):
    """Process single row secondary alignment

    Args:
        row (list): _description_

    Returns:
        _type_: _description_
    """
    cigar: str

    processed = []
    pattern = r"[MDI]+"
    for alignment in row:
        chrom, start, cigar, _ = alignment[1:-1].split(",")
        strand = cigar[0]
        try:
            length = sum(
                map(
                    lambda x: int(x) if x != "" else 0,
                    re.split(pattern, cigar[1:]),
                )
            )
        except ValueError:
            length = 0

        if strand == "+":
            start = int(start)
            end = start + length
        elif strand == "-":
            end = int(start)
            start = end - length
        # else:
        #     print(f"{strand=}")

        processed.append((chrom, start, end, cigar, strand))
    return processed


def process_multi(contacts_multi: pd.DataFrame, parts="dna") -> pd.DataFrame:
    """Process multi reads: multi RNA, DNA or both.

    Args:
        contacts_multi (pd.DataFrame): _description_
        parts (str, optional): _description_. Defaults to "dna".

    Returns:
        pd.DataFrame: _description_
    """
    if parts == "both":
        parts = ["rna", "dna"]
    else:
        parts = [parts]

    for part in parts:
        contacts_multi[f"{part}_secondary_alignments"] = (
            contacts_multi[f"{part}_secondary_alignments"]
            .str.split(";")
            .apply(process_secondary)
        )
        contacts_multi = contacts_multi.explode(f"{part}_secondary_alignments")
        names = [
            f"sec_{part}_chr",
            f"sec_{part}_start",
            f"sec_{part}_end",
            f"sec_{part}_cigar",
            f"sec_{part}_strand",
        ]

        #
        splitted_columns = pd.DataFrame(
            contacts_multi[f"{part}_secondary_alignments"].tolist(),
            index=contacts_multi.index,
            columns=names,
        )
    contacts_multi = contacts_multi.drop(
        ["dna_secondary_alignments", "rna_secondary_alignments"], axis=1
    )
    contacts_multi = pd.concat([contacts_multi, splitted_columns], axis=1)
    return contacts_multi


def convert_to_long(contacts: pd.DataFrame, filter_chr=None, parts="dna"):
    """Convert to long DF: each mapper in a separate row.

    Args:
        contacts (pd.DataFrame): _description_
        filter_chr (_type_, optional): _description_. Defaults to None.
        parts (str, optional): _description_. Defaults to "dna".

    Returns:
        _type_: _description_
    """
    primary = contacts.drop_duplicates(
        subset=[
            "read_id",
            "ATA_pairtype",
            "rna_chr",
            "rna_start",
            "rna_end",
            "rna_strand",
            "dna_chr",
            "dna_start",
            "dna_end",
        ]
    ).reset_index(drop=True)
    primary = primary[
        [
            "read_id",
            "read_id_short",
            "ATA_pairtype",
            "rna_chr",
            "rna_start",
            "rna_end",
            "rna_strand",
            "rna_cigar",
            "dna_chr",
            "dna_start",
            "dna_end",
            "dna_strand",
            "dna_cigar",
        ]
    ]
    primary["primary"] = 1
    if parts == "both":
        parts = ["rna", "dna"]
    else:
        parts = [parts]
    # contacts = contacts.copy()
    for part in parts:
        contacts[f"{part}_chr"] = contacts[f"sec_{part}_chr"]
        contacts[f"{part}_start"] = contacts[f"sec_{part}_start"]
        contacts[f"{part}_end"] = contacts[f"sec_{part}_end"]
        contacts[f"{part}_cigar"] = contacts[f"sec_{part}_cigar"]
        contacts[f"{part}_strand"] = contacts[f"sec_{part}_strand"]
    contacts["primary"] = 0

    contacts = contacts.drop(
        list(filter(lambda x: str(x).startswith("sec_"), contacts.columns)),
        axis=1,
    )
    contacts = pd.concat([primary, contacts], ignore_index=True)
    if filter_chr is not None:
        contacts = contacts[
            (contacts["dna_chr"].isin(filter_chr))
            & (contacts["rna_chr"].isin(filter_chr))
        ].reset_index(drop=True)
    return contacts


def fix_flipped_parts_(contacts_multi_rna, contacts_multi_dna):
    # fixing flipped rna dna parts
    contacts_multi_dna = contacts_multi_dna.rename(
        {
            "rna_chr": "dna_chr",
            "dna_chr": "rna_chr",
            "rna_start": "dna_start",
            "dna_start": "rna_start",
            "rna_end": "dna_end",
            "dna_end": "rna_end",
            "rna_strand": "dna_strand",
            "dna_strand": "rna_strand",
        },
        axis=1,
    )

    contacts_multi_rna = contacts_multi_rna.rename(
        {
            "rna_chr": "dna_chr",
            "dna_chr": "rna_chr",
            "rna_start": "dna_start",
            "dna_start": "rna_start",
            "rna_end": "dna_end",
            "dna_end": "rna_end",
            "rna_strand": "dna_strand",
            "dna_strand": "rna_strand",
        },
        axis=1,
    )

    contacts_multi_dna, contacts_multi_rna = (
        contacts_multi_rna,
        contacts_multi_dna,
    )

    # columns reordering
    contacts_multi_dna = contacts_multi_dna[
        [
            "read_id",
            "rna_chr",
            "rna_start",
            "rna_end",
            "rna_strand",
            "dna_chr",
            "dna_start",
            "dna_end",
            "dna_strand",
            "map_type",
        ]
    ]
    contacts_multi_rna = contacts_multi_rna[
        [
            "read_id",
            "rna_chr",
            "rna_start",
            "rna_end",
            "rna_strand",
            "dna_chr",
            "dna_start",
            "dna_end",
            "dna_strand",
            "map_type",
        ]
    ]

    return contacts_multi_rna, contacts_multi_dna


def filter_bad_reads_(contacts_multi_rna, contacts_multi_dna):
    contacts_multi_dna = contacts_multi_dna[
        (contacts_multi_dna["dna_end"] - contacts_multi_dna["dna_start"] < 150)
        & (
            contacts_multi_dna["rna_end"] - contacts_multi_dna["rna_start"]
            < 150
        )
    ].reset_index(drop=True)

    contacts_multi_rna = contacts_multi_rna[
        (contacts_multi_rna["dna_end"] - contacts_multi_rna["dna_start"] < 150)
        & (
            contacts_multi_rna["rna_end"] - contacts_multi_rna["rna_start"]
            < 150
        )
    ].reset_index(drop=True)

    multi_dna_cnts = contacts_multi_dna["read_id"].value_counts()
    multi_dna_1_cnt_set = set(
        multi_dna_cnts[multi_dna_cnts == 1].index.to_list()
    )

    multi_rna_cnts = contacts_multi_rna["read_id"].value_counts()
    multi_rna_1_cnt_set = set(
        multi_rna_cnts[multi_rna_cnts == 1].index.to_list()
    )

    contacts_multi_rna = contacts_multi_rna[
        ~contacts_multi_rna["read_id"].isin(multi_rna_1_cnt_set)
    ].reset_index(drop=True)
    contacts_multi_dna = contacts_multi_dna[
        ~contacts_multi_dna["read_id"].isin(multi_dna_1_cnt_set)
    ].reset_index(drop=True)

    return contacts_multi_rna, contacts_multi_dna


def preprocess_multi_(contacts_multi_rna, contacts_multi_dna):
    contacts_multi_rna, contacts_multi_dna = fix_flipped_parts_(
        contacts_multi_rna, contacts_multi_dna
    )
    contacts_multi_rna, contacts_multi_dna = filter_bad_reads_(
        contacts_multi_rna, contacts_multi_dna
    )

    return contacts_multi_rna, contacts_multi_dna
