import argparse
import os
from pathlib import Path

import pandas as pd

from multiK2.preprocess_utils import convert_to_long, process_multi

parser = argparse.ArgumentParser(
    description="Preprocess nf-rnachrom output. Separate UU, UM and MU"
)

parser.add_argument(
    "-i",
    "--input",
    type=str,
    required=True,
    help="Path to the input data directory",
)
parser.add_argument(
    "-o",
    "--output",
    type=str,
    required=True,
    help="Path to the output directory",
)

args = parser.parse_args()

input_dir = Path(args.input)
output_dir = Path(args.output)
output_dir.mkdir(parents=True, exist_ok=True)
multi_dna_read_cnt = 0
multi_rna_read_cnt = 0
for i, path_to_maps in enumerate(Path(input_dir).iterdir()):
    print(f"{path_to_maps.name} start")
    contacts_iter = pd.read_csv(
        path_to_maps,
        sep="\t",
        usecols=[
            "read_id",
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
            "rna_secondary_alignments",
            "dna_secondary_alignments",
        ],
        chunksize=1_000_000,
    )

    for contacts in contacts_iter:
        contacts_multi_dna = contacts[
            contacts["ATA_pairtype"] == "UM"
        ].reset_index(drop=True)
        contacts_multi_rna = contacts[
            contacts["ATA_pairtype"] == "MU"
        ].reset_index(drop=True)

        # now without MM
        # contacts_multi_both = contacts[contacts["ATA_pairtype"] == "MM"].reset_index(
        #     drop=True
        # )

        contacts_multi_unique = contacts[
            contacts["ATA_pairtype"] == "UU"
        ].reset_index(drop=True)

        if len(contacts_multi_dna) != 0:
            contacts_multi_dna["read_cnt"] = (
                contacts_multi_dna["dna_secondary_alignments"].str.count("chr")
                + 1
            )
            contacts_multi_dna = contacts_multi_dna[
                contacts_multi_dna["read_cnt"] != 100
            ].reset_index(drop=True)
            contacts_multi_dna["read_id_short"] = (
                contacts_multi_dna.index + multi_dna_read_cnt
            )
            multi_dna_read_cnt += len(contacts_multi_dna)
            contacts_multi_dna = process_multi(contacts_multi_dna, parts="dna")
            contacts_multi_dna = convert_to_long(
                contacts_multi_dna, parts="dna"
            )
            rna_parts_multi_dna = contacts_multi_dna[
                [
                    "read_id",
                    "read_id_short",
                    "rna_chr",
                    "rna_start",
                    "rna_end",
                    "rna_strand",
                    "rna_cigar",
                ]
            ].drop_duplicates(ignore_index=True)
            dna_parts_multi_dna = contacts_multi_dna[
                [
                    "read_id_short",
                    "dna_chr",
                    "dna_start",
                    "dna_end",
                    "dna_strand",
                    "dna_cigar",
                    "primary",
                ]
            ]

            file = output_dir / "rna_parts_multi_dna_all.tsv"
            rna_parts_multi_dna.to_csv(
                file,
                mode="a",
                sep="\t",
                header=not os.path.exists(file),
                index=False,
            )

            file = output_dir / "dna_parts_multi_dna_all.tsv"
            dna_parts_multi_dna.to_csv(
                file,
                mode="a",
                sep="\t",
                header=not os.path.exists(file),
                index=False,
            )

        if len(contacts_multi_rna) != 0:
            contacts_multi_rna["read_cnt"] = (
                contacts_multi_rna["rna_secondary_alignments"].str.count("chr")
                + 1
            )
            contacts_multi_rna = contacts_multi_rna[
                contacts_multi_rna["read_cnt"] != 100
            ].reset_index(drop=True)
            contacts_multi_rna["read_id_short"] = (
                contacts_multi_rna.index + multi_rna_read_cnt
            )
            multi_rna_read_cnt += len(contacts_multi_rna)
            contacts_multi_rna = process_multi(contacts_multi_rna, parts="rna")
            contacts_multi_rna = convert_to_long(
                contacts_multi_rna, parts="rna"
            )
            dna_parts_multi_rna = contacts_multi_rna[
                [
                    "read_id",
                    "read_id_short",
                    "dna_chr",
                    "dna_start",
                    "dna_end",
                    "dna_strand",
                    "dna_cigar",
                ]
            ].drop_duplicates(ignore_index=True)
            rna_parts_multi_rna = contacts_multi_rna[
                [
                    "read_id_short",
                    "rna_chr",
                    "rna_start",
                    "rna_end",
                    "rna_strand",
                    "rna_cigar",
                    "primary",
                ]
            ]

            file = output_dir / "dna_parts_multi_rna_all.tsv"
            dna_parts_multi_rna.to_csv(
                file,
                mode="a",
                sep="\t",
                header=not os.path.exists(file),
                index=False,
            )

            file = output_dir / "rna_parts_multi_rna_all.tsv"
            rna_parts_multi_rna.to_csv(
                file,
                mode="a",
                sep="\t",
                header=not os.path.exists(file),
                index=False,
            )

        if len(contacts_multi_unique) != 0:
            contacts_multi_unique = contacts_multi_unique.drop(
                [
                    "ATA_pairtype",
                    "rna_secondary_alignments",
                    "dna_secondary_alignments",
                ],
                axis=1,
            )
            contacts_multi_unique["primary"] = 1
            file = output_dir / "contacts_multi_unique_all.tsv"
            contacts_multi_unique.to_csv(
                file,
                mode="a",
                sep="\t",
                header=not os.path.exists(file),
                index=False,
            )
    print(f"{path_to_maps.name} finished")
