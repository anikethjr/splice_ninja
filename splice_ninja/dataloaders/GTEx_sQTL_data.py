# Processes GTEx sQTL data published by Linder et al. (2024)
# The data is provided as a group of VCF files, one pair for each GTEx tissue
# The positive VCF file contains sQTLs while the negative VCF file contains allele frequency-matched non-sQTLs
# Following steps need to be performed:
# 1. Load the VCF files
# 2. Extract gene IDs from attributes of the VCF records
# 3. Load VastDB event information file (created by VastDB_data.py). This file contains information about various splicing events.
# 4. Match every sQTL to the nearest splicing event
# 5. Extract the sequence of the splicing event
# 6. Extract the splicing factor expression values for that tissue. Processed file has already been created by TrASPr_GTEx_benchmark_data.py

import numpy as np
import pandas as pd
import os
import pdb
import json
import urllib
from tqdm import tqdm
import zipfile

import genomepy

import torch
import torch.distributed as dist
from torch.utils.data import Dataset, DataLoader

import lightning as L
from lightning.pytorch import LightningDataModule

from splice_ninja.utils import get_ensembl_gene_id_hgnc_with_alias, one_hot_encode_dna
from splice_ninja.dataloaders.samplers import (
    UniformPSIDistributionDistributedSampler,
    NEventsPerBatchDistributedSampler,
)

np.random.seed(0)
torch.manual_seed(0)


class GTExsQTLData(LightningDataModule):
    def prepare_data(self):
        # download the genome if it does not exist
        if not os.path.exists(os.path.join(self.cache_dir, "genomes", "GRCh38.p14")):
            print("Downloading the genome")
            os.makedirs(os.path.join(self.cache_dir, "genomes"), exist_ok=True)
            # download ENSEMBL GRChg38 v112 GTF since that was the version used for the gene counts
            genomepy.install_genome(
                name="GRCh38.p14",
                provider="Ensembl",
                annotation=True,
                version=112,
                genomes_dir=os.path.join(self.cache_dir, "genomes"),
            )
            print("Genome downloaded")

        # download the sQTL data if it does not exist
        if not os.path.exists(
            os.path.join(
                self.cache_dir, "GTEx_sQTL_data", "borzoi_curated_GTEx_sQTLs.zip"
            )
        ):
            print("Downloading the sQTL data")
            os.makedirs(os.path.join(self.cache_dir, "GTEx_sQTL_data"), exist_ok=True)
            urllib.request.urlretrieve(
                "https://github.com/anikethjr/splice_ninja/raw/refs/heads/main/borzoi_curated_GTEx_sQTLs.zip",
                os.path.join(
                    self.cache_dir, "GTEx_sQTL_data", "borzoi_curated_GTEx_sQTLs.zip"
                ),
            )
            print("sQTL data downloaded")

        # unzip the sQTL data if it does not exist
        if (
            not os.path.exists(
                os.path.join(
                    self.cache_dir, "GTEx_sQTL_data", "borzoi_curated_GTEx_sQTLs"
                )
            )
        ) or (
            len(
                os.listdir(
                    os.path.join(
                        self.cache_dir, "GTEx_sQTL_data", "borzoi_curated_GTEx_sQTLs"
                    )
                )
            )
            == 0
        ):
            print("Unzipping the sQTL data")
            with zipfile.ZipFile(
                os.path.join(
                    self.cache_dir, "GTEx_sQTL_data", "borzoi_curated_GTEx_sQTLs.zip"
                ),
                "r",
            ) as zip_ref:
                zip_ref.extractall(os.path.join(self.cache_dir, "GTEx_sQTL_data"))
            print("sQTL data unzipped")

        # download event information, gene information, and event ID to gene ID mapping from VastDB
        os.makedirs(os.path.join(self.cache_dir, "VastDB", "hg38"), exist_ok=True)
        if not os.path.exists(
            os.path.join(self.cache_dir, "VastDB", "hg38", "EVENT_INFO-hg38.tab.gz")
        ):
            print("Downloading event information from VastDB")
            url = "https://vastdb.crg.eu/downloads/hg38/EVENT_INFO-hg38.tab.gz"
            urllib.request.urlretrieve(
                url,
                os.path.join(
                    self.cache_dir, "VastDB", "hg38", "EVENT_INFO-hg38.tab.gz"
                ),
            )
            print("Event information downloaded")
        if not os.path.exists(
            os.path.join(self.cache_dir, "VastDB", "hg38", "GENE_INFO-hg38.tab.gz")
        ):
            print("Downloading gene information from VastDB")
            url = "https://vastdb.crg.eu/downloads/hg38/GENE_INFO-hg38.tab.gz"
            urllib.request.urlretrieve(
                url,
                os.path.join(self.cache_dir, "VastDB", "hg38", "GENE_INFO-hg38.tab.gz"),
            )
            print("Gene information downloaded")
        if not os.path.exists(
            os.path.join(
                self.cache_dir, "VastDB", "hg38", "EVENTID_to_GENEID-hg38.tab.gz"
            )
        ):
            print("Downloading event ID to gene ID mapping from VastDB")
            url = "https://vastdb.crg.eu/downloads/hg38/EVENTID_to_GENEID-hg38.tab.gz"
            urllib.request.urlretrieve(
                url,
                os.path.join(
                    self.cache_dir,
                    "VastDB",
                    "hg38",
                    "EVENTID_to_GENEID-hg38.tab.gz",
                ),
            )
            print("Event ID to gene ID mapping downloaded")

        # prepare the GTEx sQTL data
        if not os.path.exists(
            os.path.join(self.cache_dir, "GTEx_sQTL_data", "final.csv")
        ):
            print("Preparing the GTEx sQTL data")

            # first load all positive and negative variants from the merged VCF files
            positive_sQTLs_df = pd.read_csv(
                os.path.join(self.cache_dir, "GTEx_sQTL_data", "pos_merge.vcf.gz"),
                sep="\t",
                skiprows=4,
            )
            positive_sQTLs_df["label"] = 1
            negative_sQTLs_df = pd.read_csv(
                os.path.join(self.cache_dir, "GTEx_sQTL_data", "neg_merge.vcf.gz"),
                sep="\t",
                skiprows=4,
            )
            negative_sQTLs_df["label"] = 0
            all_variants_df = pd.concat(
                [positive_sQTLs_df, negative_sQTLs_df], ignore_index=True
            )
            print(
                f"Total number of variants: {len(all_variants_df)}, number of positive sQTLs: {len(positive_sQTLs_df)}, number of negative sQTLs: {len(negative_sQTLs_df)}"
            )
            # extract gene_id from INFO
            all_variants_df["gene_id"] = (
                all_variants_df["INFO"]
                .str.split("MT=")
                .str.get(1)
                .str.split(".")
                .str.get(0)
            )

            # load VAST-DB event information to get information needed for input construction
            event_info_from_vastdb = pd.read_csv(
                os.path.join(
                    self.cache_dir, "VastDB", "hg38", "EVENT_INFO-hg38.tab.gz"
                ),
                sep="\t",
            )
            # keep only EX, INT, ALTD, or ALTA events
            event_types = []
            for i in range(len(event_info_from_vastdb)):
                event_id = event_info_from_vastdb.iloc[i]["EVENT"]
                if "EX" in event_id:
                    event_types.append("EX")
                elif "INT" in event_id:
                    event_types.append("INT")
                elif "ALTD" in event_id:
                    event_types.append("ALTD")
                elif "ALTA" in event_id:
                    event_types.append("ALTA")
                else:
                    event_types.append("Other")
            event_info_from_vastdb["EVENT_TYPE"] = event_types
            event_info_from_vastdb = event_info_from_vastdb[
                event_info_from_vastdb["EVENT_TYPE"] != "Other"
            ]
            event_info_from_vastdb.reset_index(drop=True, inplace=True)

            # load VAST-DB event ID to gene ID mapping
            event_id_to_gene_id_from_vastdb = pd.read_csv(
                os.path.join(
                    self.cache_dir, "VastDB", "hg38", "EVENTID_to_GENEID-hg38.tab.gz"
                ),
                sep="\t",
            )

            # merge to get everything in one df
            event_info_from_vastdb = event_info_from_vastdb.merge(
                event_id_to_gene_id_from_vastdb,
                left_on="EVENT",
                right_on="EventID",
                how="inner",
            )
            event_info_from_vastdb.drop(
                columns=["EventID"], inplace=True
            )  # drop redundant column

            # remove events from genes that are not in represented in all_variants_df
            event_info_from_vastdb["filter"] = event_info_from_vastdb["GeneID"].isin(
                all_variants_df["gene_id"]
            )
            event_info_from_vastdb = event_info_from_vastdb[
                event_info_from_vastdb["filter"]
            ]
            event_info_from_vastdb.drop(columns=["filter"], inplace=True)
            event_info_from_vastdb.reset_index(drop=True, inplace=True)

            # for every variant, find the nearest event in that gene
            all_variants_df["closest_EX_EVENT"] = None
            all_variants_df["closest_EX_EVENT_distance"] = None
            all_variants_df["closest_INT_EVENT"] = None
            all_variants_df["closest_INT_EVENT_distance"] = None
            all_variants_df["closest_ALTD_EVENT"] = None
            all_variants_df["closest_ALTD_EVENT_distance"] = None
            all_variants_df["closest_ALTA_EVENT"] = None
            all_variants_df["closest_ALTA_EVENT_distance"] = None
            all_variants_df["closest_EVENT"] = None
            all_variants_df["closest_EVENT_distance"] = None
            for i in tqdm(range(len(all_variants_df))):
                variant_row = all_variants_df.iloc[i]
                current_gene_id = variant_row["gene_id"]
                current_chrom = variant_row["#CHROM"]
                current_pos = variant_row["POS"]

                gene_events = event_info_from_vastdb[
                    (event_info_from_vastdb["GeneID"] == current_gene_id)
                ]

                if not gene_events.empty:
                    gene_strand = gene_events.iloc[0]["REF_CO"].split(":")[-1]
                    # all coordinates are in the format chr:start-end
                    # we consider the CO_A column that indicates the coordinates of the alternative exon
                    # for each event type, we find the nearest event to the variant while considering both the start and end of the alternative exon

                    # function to get exon coordinates
                    def get_exon_coordinates(co_a, strand):
                        if strand == "+":
                            exon_start = int(co_a.split(":")[1].split("-")[0])
                            exon_end = int(co_a.split(":")[1].split("-")[1])
                        else:
                            exon_start = int(co_a.split(":")[1].split("-")[1])
                            exon_end = int(co_a.split(":")[1].split("-")[0])
                        return exon_start, exon_end

                    for event_type in ["EX", "INT", "ALTD", "ALTA"]:
                        filtered_events = gene_events[
                            gene_events["EVENT_TYPE"] == event_type
                        ]
                        filtered_events = filtered_events.dropna(subset="CO_A")
                        if not filtered_events.empty:
                            min_distance = float("inf")
                            closest_event_id = None

                            for _, event_row in filtered_events.iterrows():
                                exon_start, exon_end = get_exon_coordinates(
                                    event_row["CO_A"], gene_strand
                                )
                                distance = min(
                                    abs(current_pos - exon_start),
                                    abs(current_pos - exon_end),
                                )
                                if distance < min_distance:
                                    min_distance = distance
                                    closest_event_id = event_row["EVENT"]

                            all_variants_df.loc[
                                i, f"closest_{event_type}_EVENT"
                            ] = closest_event_id
                            all_variants_df.loc[
                                i, f"closest_{event_type}_EVENT_distance"
                            ] = min_distance

                    # now get closest event of any type
                    min_distance = float("inf")
                    closest_event_id = None
                    for event_type in ["EX", "INT", "ALTD", "ALTA"]:
                        if (
                            all_variants_df.loc[
                                i, f"closest_{event_type}_EVENT_distance"
                            ]
                            is not None
                        ) and (
                            all_variants_df.loc[
                                i, f"closest_{event_type}_EVENT_distance"
                            ]
                            < min_distance
                        ):
                            min_distance = all_variants_df.loc[
                                i, f"closest_{event_type}_EVENT_distance"
                            ]
                            closest_event_id = all_variants_df.loc[
                                i, f"closest_{event_type}_EVENT"
                            ]
                    all_variants_df.loc[i, "closest_EVENT"] = closest_event_id
                    all_variants_df.loc[i, "closest_EVENT_distance"] = min_distance
