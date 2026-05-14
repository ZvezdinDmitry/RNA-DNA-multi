import argparse
from pathlib import Path

import pandas as pd

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
rna_maps_cnts = []
dna_maps_cnts = []
for i, path_to_maps in enumerate(Path(input_dir).iterdir()):
    print(f"{path_to_maps.name} start")
    contacts_iter = pd.read_csv(
        path_to_maps,
        sep="\t",
        usecols=[
            "ATA_pairtype",
            "rna_secondary_alignments",
            "dna_secondary_alignments",
        ],
        dtype={"ATA_pairtype": "category"},
        chunksize=5_000_000,
    )
    for contacts in contacts_iter:
        multi_rna = contacts.loc[
            contacts["ATA_pairtype"] == "MU", "rna_secondary_alignments"
        ]
        if len(multi_rna) > 0:
            maps_cnts = (
                (multi_rna.str.count("chr") + 1).value_counts().reset_index()
            )
            rna_maps_cnts.append(maps_cnts)

        multi_dna = contacts.loc[
            contacts["ATA_pairtype"] == "UM", "dna_secondary_alignments"
        ]
        if len(multi_dna) > 0:
            maps_cnts = (
                (multi_dna.str.count("chr") + 1).value_counts().reset_index()
            )
            dna_maps_cnts.append(maps_cnts)

        break

    print(f"{path_to_maps.name} finished")

rna_maps_cnts = pd.concat(rna_maps_cnts, ignore_index=True)
rna_maps_cnts = (
    rna_maps_cnts.groupby("rna_secondary_alignments")
    .sum("count")
    .reset_index()
)

dna_maps_cnts = pd.concat(dna_maps_cnts, ignore_index=True)
dna_maps_cnts = (
    dna_maps_cnts.groupby("dna_secondary_alignments")
    .sum("count")
    .reset_index()
)

rna_maps_cnts.to_csv(
    output_dir / "multi_rna_maps_cnts.tsv", index=False, sep="\t"
)
dna_maps_cnts.to_csv(
    output_dir / "multi_dna_maps_cnts.tsv", index=False, sep="\t"
)
print("finished")
