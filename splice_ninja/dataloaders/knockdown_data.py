# Processes data published by Rogalska et al. (2024) - Transcriptome-wide splicing network reveals specialized regulatory functions of the core spliceosome

import numpy as np
import pandas as pd
import os
import pdb
import json
from tqdm import tqdm
from statsmodels.stats.multitest import multipletests

import genomepy

import torch
from torch.utils.data import Dataset, DataLoader

import lightning as L
from lightning.pytorch import LightningDataModule

from splice_ninja.utils import get_ensembl_gene_id_hgnc_with_alias

np.random.seed(0)
torch.manual_seed(0)


# Datast for the knockdown data
class KnockdownDataset(Dataset):
    def __init__(self, data_module, split="train"):
        self.data_module = data_module
        self.split = split

        if self.split == "train":
            self.chromosomes = self.data_module.train_chromosomes
        elif self.split == "val":
            self.chromosomes = self.data_module.val_chromosomes
        elif self.split == "test":
            self.chromosomes = self.data_module.test_chromosomes

        self.data = self.data_module.inclusion_levels_full[
            self.data_module.inclusion_levels_full["CHR"].isin(self.chromosomes)
        ].reset_index(drop=True)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        splicing_event = self.data.iloc[idx]

        gene_name = splicing_event["GENE"]

        # VAST-DB event ID. Formed by:
        # Species identifier: Hsa (Human), Mmu (Mouse), or Gga (Chicken);
        # Type of alternative splicing event:
        #   alternative exon skipping (EX),
        #   retained intron (INT),
        #   alternative splice site donor choice (ALTD), or alternative splice site acceptor choice (ALTA).
        #       In the case of ALTD/ALTA, each splice site within the event is indicated (from exonic internal to external) over the
        #       total number of alternative splice sites in the event (e.g. HsaALTA0000011-1/2).
        # Numerical identifier.
        event_id = splicing_event["EVENT"]

        length_of_event = splicing_event["LENGTH"]
        coordinates = splicing_event["FullCO"].split(",")

        # construct full input sequence and a mask which indicates the positions of the event
        # input sequence is the concatenation of the event + upstream and downstream sequences of the event
        # the mask is used to condition the model on the event, therefore model predicts P(inclusion | event) = PSI

        return {
            "gene_name": gene_name,
            "event_id": event_id,
            "length_of_event": length_of_event,
            "coordinates": coordinates,
        }


# DataModule for the knockdown data
class KnockdownData(LightningDataModule):
    def __init__(self, config: dict | str):
        super().__init__()
        if isinstance(config, str):
            with open(config, "r") as f:
                self.config = json.load(f)
        else:
            self.config = config
        self.input_size = self.config["train_config"]["input_size"]
        self.train_chromosomes = self.config["train_config"]["train_chromosomes"]
        self.test_chromosomes = self.config["train_config"]["test_chromosomes"]
        self.val_chromosomes = self.config["train_config"]["val_chromosomes"]

        # cache directory
        self.cache_dir = self.config["data_config"]["cache_dir"]
        if not os.path.exists(self.cache_dir):
            os.makedirs(self.cache_dir, exist_ok=True)

        # load/create the filtered splicing data
        if (
            not os.path.exists(
                os.path.join(self.cache_dir, "inclusion_levels_full_filtered.csv")
            )
            or not os.path.exists(
                os.path.join(self.cache_dir, "gene_counts_filtered.csv")
            )
            or not os.path.exists(
                os.path.join(self.cache_dir, "gene_name_to_ensembl_id.json")
            )
            or not os.path.exists(
                os.path.join(self.cache_dir, "ensembl_id_to_gene_name.json")
            )
        ):
            print("Filtering data")

            if os.path.exists(
                os.path.join(self.cache_dir, "gene_name_to_ensembl_id.json")
            ):
                with open(
                    os.path.join(self.cache_dir, "gene_name_to_ensembl_id.json"), "r"
                ) as f:
                    self.gene_name_to_ensembl_id = json.load(f)
            else:
                self.gene_name_to_ensembl_id = {}

            if os.path.exists(
                os.path.join(self.cache_dir, "ensembl_id_to_gene_name.json")
            ):
                with open(
                    os.path.join(self.cache_dir, "ensembl_id_to_gene_name.json"), "r"
                ) as f:
                    self.ensembl_id_to_gene_name = json.load(f)
            else:
                self.ensembl_id_to_gene_name = {}

            # load gene counts
            self.gene_counts = pd.read_csv(
                os.path.join(self.config["data_config"]["data_dir"], "geneCounts.tab"),
                sep="\t",
                index_col=0,
            )
            self.gene_counts = self.gene_counts.rename({"X": "gene_id"}, axis=1)
            print(
                "Read gene counts data, number of samples: {}".format(
                    self.gene_counts.shape[1] - 2
                )
            )

            # average counts from biological replicates - these are samples with suffixes "_r1" and "_r2"
            self.gene_counts_samples = self.gene_counts.columns[2:]
            rename_dict = {}
            for col in self.gene_counts_samples:
                if col.endswith("_r1"):
                    self.gene_counts[col] = (
                        self.gene_counts[col] + self.gene_counts[col[:-3] + "_r2"]
                    ) / 2
                    rename_dict[col] = col[:-3]
                if col.endswith(
                    ".1"
                ):  # these correspond to replicates with the "_b" or "con" suffix in the splicing data
                    if col[:-2] in [
                        "C1orf55",
                        "CCDC12",
                        "CDC5L",
                        "CWC22",
                        "HFM1",
                        "LENG1",
                        "SRPK2",
                        "XAB2",
                    ]:
                        rename_dict[col] = col[:-2] + "_b"
                    elif col[:-2] in ["IK", "PRPF8", "RBM17", "SF3B1", "SMU1"]:
                        rename_dict[col] = col[:-2] + "con"
                    else:
                        raise ValueError(
                            f"Could not determine replicate suffix for {col}"
                        )
            self.gene_counts = self.gene_counts.drop(
                columns=[i for i in self.gene_counts.columns if i.endswith("_r2")]
            )
            self.gene_counts = self.gene_counts.rename(columns=rename_dict)
            print(
                "Averaged gene counts from biological replicates, number of samples: {}".format(
                    self.gene_counts.shape[1] - 2
                )
            )

            # start building a dictionary to map gene names to Ensembl gene IDs
            drop_columns = []
            for sf in tqdm(self.gene_counts.columns[2:]):
                if sf not in gene_name_to_ensembl_id:
                    if sf.endswith("_b") or sf.endswith("con"):
                        ensembl_id = get_ensembl_gene_id_hgnc_with_alias(
                            sf[:-2] if sf.endswith("_b") else sf[:-3]
                        )  # return can be str | list[str], list is for multiple IDs
                    else:
                        ensembl_id = get_ensembl_gene_id_hgnc_with_alias(
                            sf
                        )  # return can be str | list[str], list is for multiple IDs
                    if ensembl_id is not None:
                        if isinstance(ensembl_id, list):
                            gene_name_to_ensembl_id[sf] = ensembl_id
                        else:
                            gene_name_to_ensembl_id[sf] = [ensembl_id]
                        for ensembl_id in gene_name_to_ensembl_id[sf]:
                            if ensembl_id in ensembl_id_to_gene_name:
                                ensembl_id_to_gene_name[ensembl_id].append(sf)
                            else:
                                ensembl_id_to_gene_name[ensembl_id] = [sf]
                    else:
                        drop_columns.append(sf)
            # if any of the columns to be dropped have a row with the same name in the gene count data, we use the gene ID present in the gene count data and add it to the mapping and don't drop the column
            for sf in drop_columns:
                if (self.gene_counts["alias"] == sf).sum() > 0:
                    ensembl_id = self.gene_counts.loc[
                        self.gene_counts["alias"] == sf, "gene_id"
                    ].iloc[0]
                    gene_name_to_ensembl_id[sf] = [ensembl_id]
                    if ensembl_id in ensembl_id_to_gene_name:
                        ensembl_id_to_gene_name[ensembl_id].append(sf)
                    else:
                        ensembl_id_to_gene_name[ensembl_id] = [sf]
            drop_columns = [i for i in drop_columns if i not in gene_name_to_ensembl_id]

            # drop columns for which we could not find the Ensembl gene ID
            self.gene_counts = self.gene_counts.drop(columns=drop_columns)
            print(
                "Dropping gene count data from {} splicing factors for which the Ensembl ID could not be found".format(
                    len(drop_columns)
                )
            )
            # also drop columns for which there is no corresponding gene count data in the rows
            drop_columns = []
            for sf in self.gene_counts.columns[2:]:
                check = False
                for ensembl_id in gene_name_to_ensembl_id[sf]:
                    if (self.gene_counts["gene_id"] == ensembl_id).sum() > 0:
                        check = True
                        break
                if not check:
                    alias = (
                        sf
                        if not sf.endswith("_b") and not sf.endswith("con")
                        else sf[:-2]
                        if sf.endswith("_b")
                        else sf[:-3]
                    )
                    if (self.gene_counts["alias"] == alias).sum() == 0:
                        drop_columns.append(sf)
                    else:
                        print(
                            "Weird case where the gene ID is not found but the alias is found for {}, using gene ID present in the gene count data".format(
                                sf
                            )
                        )
                        ensembl_id = self.gene_counts.loc[
                            self.gene_counts["alias"] == alias, "gene_id"
                        ].iloc[0]
                        gene_name_to_ensembl_id[sf] = [ensembl_id]
                        if ensembl_id in ensembl_id_to_gene_name:
                            ensembl_id_to_gene_name[ensembl_id].append(sf)
                        else:
                            ensembl_id_to_gene_name[ensembl_id] = [sf]
            self.gene_counts = self.gene_counts.drop(columns=drop_columns)
            print(
                "Dropping gene count data from {} splicing factors for which the gene ID could not be found in the gene count data".format(
                    len(drop_columns)
                )
            )

            # load psi values
            # data was provided in the VastTools output format, more details on the data format are here - https://github.com/vastgroup/vast-tools?tab=readme-ov-file#combine-output-format
            self.inclusion_levels_full = pd.read_csv(
                os.path.join(
                    self.config["data_config"]["data_dir"],
                    "INCLUSION_LEVELS_FULL-Hs2331.tab",
                ),
                sep="\t",
            )
            print(
                "Read PSI values data, number of samples: {}".format(
                    (self.inclusion_levels_full.shape[1] - 6) / 2
                )
            )

            # rename all the PSI and quality columns to remove the trailing "_1" at the end
            rename_dict = {}
            for col in self.inclusion_levels_full.columns[6:]:
                if col.endswith("_1"):
                    rename_dict[col] = col[:-2]
                elif col.endswith("_1-Q"):
                    rename_dict[col] = col[:-4] + "-Q"
            self.inclusion_levels_full = self.inclusion_levels_full.rename(
                columns=rename_dict
            )

            # get the columns for PSI values and quality - every PSI column is followed by a quality column
            # each PSI value is measured after knockdown of a specific splicing factor indicated by the column name
            self.psi_vals_columns = [
                i
                for i in self.inclusion_levels_full.columns[6:]
                if not i.endswith("-Q")
            ]
            self.quality_columns = [
                i for i in self.inclusion_levels_full.columns[6:] if i.endswith("-Q")
            ]
            assert len(self.psi_vals_columns) == len(self.quality_columns)
            print("Every PSI value is followed by a quality column in the raw data")

            # get ensembl gene IDs for the splicing factors being knocked down in the splicing data
            drop_columns = []
            for sf in tqdm(self.psi_vals_columns):
                if sf not in gene_name_to_ensembl_id:
                    if sf.endswith("_b") or sf.endswith("con"):
                        ensembl_id = get_ensembl_gene_id_hgnc_with_alias(
                            sf[:-2] if sf.endswith("_b") else sf[:-3]
                        )
                    else:
                        ensembl_id = get_ensembl_gene_id_hgnc_with_alias(sf)
                    if ensembl_id is not None:
                        if isinstance(ensembl_id, list):
                            gene_name_to_ensembl_id[sf] = ensembl_id
                        else:
                            gene_name_to_ensembl_id[sf] = [ensembl_id]
                        for ensembl_id in gene_name_to_ensembl_id[sf]:
                            if ensembl_id in ensembl_id_to_gene_name:
                                ensembl_id_to_gene_name[ensembl_id].append(sf)
                            else:
                                ensembl_id_to_gene_name[ensembl_id] = [sf]
                    else:
                        drop_columns.append(sf)
                        drop_columns.append(sf + "-Q")
            self.inclusion_levels_full = self.inclusion_levels_full.drop(
                columns=drop_columns
            )
            print(
                "Dropping PSI values data from {} samples for which the Ensembl ID could not be found".format(
                    len(drop_columns) // 2
                )
            )
            self.psi_vals_columns = [
                i
                for i in self.inclusion_levels_full.columns[6:]
                if not i.endswith("-Q")
            ]
            self.quality_columns = [
                i for i in self.inclusion_levels_full.columns[6:] if i.endswith("-Q")
            ]
            assert len(self.psi_vals_columns) == len(self.quality_columns)
            print(
                "Every PSI value is followed by a quality column in the data after removing samples for which the Ensembl ID could not be found"
            )

            # cache the gene name to Ensembl gene ID mapping
            with open(
                os.path.join(self.cache_dir, "gene_name_to_ensembl_id.json"), "w+"
            ) as f:
                json.dump(gene_name_to_ensembl_id, f)

            # cache the Ensembl gene ID to gene name mapping
            with open(
                os.path.join(self.cache_dir, "ensembl_id_to_gene_name.json"), "w+"
            ) as f:
                json.dump(ensembl_id_to_gene_name, f)

            # discard columns corresponding to the following samples due to poor quality of the data
            # AA2, AA1, CCDC12, C1orf55, C1orf55_b, CDC5L, HFM1, LENG1, RBM17, PPIL1, SRRM4, SRRT
            drop_columns = [
                "AA2",
                "AA1",
                "CCDC12",
                "C1orf55",
                "C1orf55_b",
                "CDC5L",
                "HFM1",
                "LENG1",
                "RBM17",
                "PPIL1",
                "SRRM4",
                "SRRT",
            ] + [
                i + "-Q"
                for i in [
                    "AA2",
                    "AA1",
                    "CCDC12",
                    "C1orf55",
                    "C1orf55_b",
                    "CDC5L",
                    "HFM1",
                    "LENG1",
                    "RBM17",
                    "PPIL1",
                    "SRRM4",
                    "SRRT",
                ]
            ]
            self.inclusion_levels_full = self.inclusion_levels_full.drop(
                columns=drop_columns, errors="ignore"
            )
            print(
                "Discarded columns with poor quality data, number of samples: {}".format(
                    (self.inclusion_levels_full.shape[1] - 6) / 2
                )
            )
            self.psi_vals_columns = [
                i
                for i in self.inclusion_levels_full.columns[6:]
                if not i.endswith("-Q")
            ]
            self.quality_columns = [
                i for i in self.inclusion_levels_full.columns[6:] if i.endswith("-Q")
            ]
            assert len(self.psi_vals_columns) == len(self.quality_columns)
            print(
                "Every PSI value is followed by a quality column in the data after discarding poor quality data"
            )
            # drop these columns from the gene counts as well, if they exist + some specific rules to account for data weirdness
            drop_columns = []
            rename_dict = {}
            for col in drop_columns:
                if col in self.gene_counts.columns:
                    if col == "LENG1":
                        if "LENG1_b" in self.gene_counts.columns:
                            drop_columns.append("LENG1")
                            rename_dict["LENG1_b"] = "LENG1"
                        else:
                            print(
                                "We needed LENG1_b gene count data but it was not found, so using LENG1 data"
                            )
                    elif col == "RBM17":
                        if "RBM17con" in self.gene_counts.columns:
                            drop_columns.append("RBM17")
                            rename_dict["RBM17con"] = "RBM17"
                        else:
                            print(
                                "We needed RBM17con gene count data but it was not found, so using RBM17 data"
                            )
                    elif col == "HFM1":
                        if "HFM1_b" in self.gene_counts.columns:
                            drop_columns.append("HFM1")
                            rename_dict["HFM1_b"] = "HFM1"
                        else:
                            print(
                                "We needed HFM1_b gene count data but it was not found, so using HFM1 data"
                            )
                    elif col == "CCDC12":
                        if "CCDC12_b" in self.gene_counts.columns:
                            drop_columns.append("CCDC12")
                            rename_dict["CCDC12_b"] = "CCDC12"
                        else:
                            print(
                                "We needed CCDC12_b gene count data but it was not found, so using CCDC12 data"
                            )
                    elif col == "CDC5L":
                        if "CDC5L_b" in self.gene_counts.columns:
                            drop_columns.append("CDC5L")
                            rename_dict["CDC5L_b"] = "CDC5L"
                        else:
                            print(
                                "We needed CDC5L_b gene count data but it was not found, so using CDC5L data"
                            )
                    else:
                        drop_columns.append(col)
            self.gene_counts = self.gene_counts.drop(columns=drop_columns)
            self.gene_counts = self.gene_counts.rename(columns=rename_dict)
            print(
                "Dropped gene count data from samples for which the PSI value data was dropped due to poor quality, left with num samples = {}".format(
                    self.gene_counts.shape[1] - 2
                )
            )

            # authors of original work use data from second replicate for the following splicing factors:
            # LENG1 (called LENG1_b), RBM17 (called RBM17con), HFM1 (called HFM1_b), CCDC12 (called CCDC12_b), CDC5L (called CDC5L_b)
            # (from https://github.com/estepi/SpliceNet/blob/main/prepareALLTable.R#L20-L29)
            # thus, we drop the columns for the first replicate of these splicing factors (dropping was already done in the previous step) and rename the columns for the second replicate
            rename_dict = {
                "LENG1_b": "LENG1",
                "RBM17con": "RBM17",
                "HFM1_b": "HFM1",
                "CCDC12_b": "CCDC12",
                "CDC5L_b": "CDC5L",
                "LENG1_b-Q": "LENG1-Q",
                "RBM17con-Q": "RBM17-Q",
                "HFM1_b-Q": "HFM1-Q",
                "CCDC12_b-Q": "CCDC12-Q",
                "CDC5L_b-Q": "CDC5L-Q",
            }
            self.inclusion_levels_full = self.inclusion_levels_full.rename(
                columns=rename_dict, errors="ignore"
            )

            # we also drop all other replicate columns (endswith "_b" or "con") since we only use the data from the first replicate
            drop_columns = [
                i
                for i in self.inclusion_levels_full.columns
                if i.endswith("_b") or i.endswith("con")
            ]
            drop_columns += [i + "-Q" for i in drop_columns]
            self.inclusion_levels_full = self.inclusion_levels_full.drop(
                columns=drop_columns
            )
            self.psi_vals_columns = [
                i
                for i in self.inclusion_levels_full.columns[6:]
                if not i.endswith("-Q")
            ]
            self.quality_columns = [
                i for i in self.inclusion_levels_full.columns[6:] if i.endswith("-Q")
            ]
            for i in self.psi_vals_columns:
                assert (
                    f"{i}-Q" in self.quality_columns
                ), f"Quality column for {i} not found"
            assert len(self.psi_vals_columns) == len(self.quality_columns)
            print(
                "Dropped unused replicate columns from PSI values data, number of samples: {}".format(
                    (self.inclusion_levels_full.shape[1] - 6) / 2
                )
            )

            # drop replicate columns from gene counts
            drop_columns = [
                i
                for i in self.gene_counts.columns
                if i.endswith("_b") or i.endswith("con")
            ]
            self.gene_counts = self.gene_counts.drop(columns=drop_columns)
            print(
                "Dropped unused replicate columns from gene count data, number of samples: {}".format(
                    self.gene_counts.shape[1] - 2
                )
            )

            # now rename all the columns to use Ensembl gene IDs instead of gene names since the two files don't always use the same gene names
            # if a gene has multiple Ensembl IDs, we use the one for which the gene id is found in the gene count data and has the cumulative highest expression
            rename_dict = {}
            for i in self.gene_counts.columns[2:]:
                max_count = 0
                best_ensembl_id = None
                for ensembl_id in gene_name_to_ensembl_id[i]:
                    if (self.gene_counts["gene_id"] == ensembl_id).sum() > 0:
                        exp_count = (
                            self.gene_counts[self.gene_counts["gene_id"] == ensembl_id]
                            .iloc[0][self.gene_counts.columns[2:]]
                            .sum()
                        )
                        if exp_count > max_count:
                            max_count = exp_count
                            best_ensembl_id = ensembl_id
                if best_ensembl_id is None:
                    raise Exception(f"Could not find the gene count data for {i}")

                rename_dict[i] = best_ensembl_id
            self.gene_counts = self.gene_counts.rename(columns=rename_dict)

            # now rename the columns in the PSI values data
            # we also rename the quality columns
            # if a gene has multiple Ensembl IDs, we use the one which was used for the gene count data
            rename_dict = {}
            for i in self.psi_vals_columns:
                max_count = 0
                best_ensembl_id = None
                for ensembl_id in gene_name_to_ensembl_id[i]:
                    if ensembl_id in self.gene_counts.columns:
                        exp_count = (
                            self.gene_counts[self.gene_counts["gene_id"] == ensembl_id]
                            .iloc[0][self.gene_counts.columns[2:]]
                            .sum()
                        )
                        if exp_count > max_count:
                            max_count = exp_count
                            best_ensembl_id = ensembl_id
                if best_ensembl_id is None:
                    raise Exception(f"Could not find the gene count data for {i}")

                rename_dict[i] = best_ensembl_id
                rename_dict[i + "-Q"] = best_ensembl_id + "-Q"
            self.inclusion_levels_full = self.inclusion_levels_full.rename(
                columns=rename_dict
            )

            # now drop all splicing data for which the gene count data is not available and vice versa
            drop_columns = []
            self.psi_vals_columns = [
                i
                for i in self.inclusion_levels_full.columns[6:]
                if not i.endswith("-Q")
            ]
            for sf in self.psi_vals_columns:
                if sf not in self.gene_counts.columns:
                    drop_columns.append(sf)
                    drop_columns.append(sf + "-Q")
            self.inclusion_levels_full = self.inclusion_levels_full.drop(
                columns=drop_columns
            )
            print(
                "Dropping PSI values data from {} samples for which the gene count data could not be found".format(
                    len(drop_columns) // 2
                )
            )
            self.psi_vals_columns = [
                i
                for i in self.inclusion_levels_full.columns[6:]
                if not i.endswith("-Q")
            ]

            drop_columns = []
            for sf in self.gene_counts.columns[2:]:
                if sf not in self.psi_vals_columns:
                    drop_columns.append(sf)
            self.gene_counts = self.gene_counts.drop(columns=drop_columns)
            print(
                "Dropping gene count data from {} samples for which the PSI value data could not be found".format(
                    len(drop_columns)
                )
            )

            self.psi_vals_columns = [
                i
                for i in self.inclusion_levels_full.columns[6:]
                if not i.endswith("-Q")
            ]
            self.quality_columns = [
                i for i in self.inclusion_levels_full.columns[6:] if i.endswith("-Q")
            ]
            assert set(self.gene_counts.columns[2:]) == set(
                self.psi_vals_columns
            )  # check that the columns match
            for i in self.psi_vals_columns:
                assert (
                    f"{i}-Q" in self.quality_columns
                ), f"Quality column for {i} not found"
            assert len(self.psi_vals_columns) == len(self.quality_columns)
            assert len(self.psi_vals_columns) == (self.gene_counts.shape[1] - 2)
            print(
                "After dropping splicing data samples for which the gene count data could not be found and vice versa, number of samples: {}".format(
                    (self.inclusion_levels_full.shape[1] - 6) / 2
                )
            )

            # print some statistics about the data
            print(f"Number of samples: {len(self.psi_vals_columns)}")
            initial_num_PSI_vals = (
                (~np.isnan(self.inclusion_levels_full[self.psi_vals_columns]))
                .sum()
                .sum()
            )
            print(f"Initial number of PSI values: {initial_num_PSI_vals}")
            number_of_PSI_vals_of_each_type = {}
            for event_type in self.inclusion_levels_full["COMPLEX"].unique():
                number_of_PSI_vals_of_each_type[event_type] = (
                    (
                        ~np.isnan(
                            self.inclusion_levels_full.loc[
                                self.inclusion_levels_full["COMPLEX"] == event_type,
                                self.psi_vals_columns,
                            ]
                        )
                    )
                    .sum()
                    .sum()
                )
            print(
                f"Number of PSI values of each type: {number_of_PSI_vals_of_each_type}"
            )
            # remove events with NaN PSI values in all samples
            filter_out_events_with_all_PSI_values_NaN = np.all(
                self.inclusion_levels_full[self.psi_vals_columns].isna(), axis=1
            )
            self.inclusion_levels_full = self.inclusion_levels_full.loc[
                ~filter_out_events_with_all_PSI_values_NaN
            ].reset_index(drop=True)
            print(
                f"Number of events of each type after dropping events with no valid measurements: {self.inclusion_levels_full['COMPLEX'].value_counts()}"
            )

            # filter out PSI values which did not pass the quality control
            # first value of the first comma-separated list of values in the quality column is the quality control flag
            # (from https://github.com/estepi/SpliceNet/blob/main/replaceNAI.R#L23-L31)
            for i in self.quality_columns:
                self.inclusion_levels_full.loc[
                    self.inclusion_levels_full[i].apply(
                        lambda x: "OK" not in x.split(",")[0]
                    ),
                    i[:-2],
                ] = np.nan
            num_PSI_vals_after_quality_control_filtering = (
                (~np.isnan(self.inclusion_levels_full[self.psi_vals_columns]))
                .sum()
                .sum()
            )
            percent_events_filtered = (
                100
                * (initial_num_PSI_vals - num_PSI_vals_after_quality_control_filtering)
                / initial_num_PSI_vals
            )
            print(
                f"Number of PSI values after filtering events which did not pass the quality control: {num_PSI_vals_after_quality_control_filtering} ({percent_events_filtered:.2f}% of initial values filtered)"
            )
            num_PSI_vals_of_each_type_after_quality_control_filtering = {}
            for event_type in self.inclusion_levels_full["COMPLEX"].unique():
                num_PSI_vals_of_each_type_after_quality_control_filtering[
                    event_type
                ] = (
                    (
                        ~np.isnan(
                            self.inclusion_levels_full.loc[
                                self.inclusion_levels_full["COMPLEX"] == event_type,
                                self.psi_vals_columns,
                            ]
                        )
                    )
                    .sum()
                    .sum()
                )
            print(
                f"Number of PSI values of each type after filtering events which did not pass the quality control: {num_PSI_vals_of_each_type_after_quality_control_filtering}"
            )
            # remove events with NaN PSI values in all samples
            filter_out_events_with_all_PSI_values_NaN = np.all(
                self.inclusion_levels_full[self.psi_vals_columns].isna(), axis=1
            )
            self.inclusion_levels_full = self.inclusion_levels_full.loc[
                ~filter_out_events_with_all_PSI_values_NaN
            ].reset_index(drop=True)
            print(
                f"Number of events of each type after dropping events with no valid measurements: {self.inclusion_levels_full['COMPLEX'].value_counts()}"
            )

            # filter out intron retention (IR) PSI values where the corrected p-value of a binomial test of balance between reads mapping to the upstream and downstream exon-intron junctions is less than 0.05,
            # indicating a significant imbalance in the reads mapping to the two junctions and therefore a false positive IR event
            # the raw p-value is stored in the quality column for each PSI value - last number before the @ symbol in the string (@ separates two comma-separated lists of values)
            # correction is performed using the Holm method by pooling p-values across all events observed in the same sample
            # (from https://github.com/estepi/SpliceNet/blob/main/replaceNAI.R#L71-L78)
            is_IR = self.inclusion_levels_full["COMPLEX"] == "IR"
            for i in self.quality_columns:
                IR_events_raw_pvals = self.inclusion_levels_full.loc[is_IR, i].apply(
                    lambda x: float(x.split("@")[0].split(",")[-1])
                    if x.split("@")[0].split(",")[-1] != "NA"
                    else 1
                )
                IR_events_corrected_pvals = multipletests(
                    IR_events_raw_pvals, alpha=0.05, method="holm"
                )[1]
                # replace the PSI values of IR events with NaN if the corrected p-value is less than 0.05
                filter_out = np.zeros(self.inclusion_levels_full.shape[0], dtype=bool)
                filter_out[is_IR] = IR_events_corrected_pvals < 0.05
                self.inclusion_levels_full.loc[filter_out, i[:-2]] = np.nan
            num_PSI_vals_after_IR_filtering = (
                (~np.isnan(self.inclusion_levels_full[self.psi_vals_columns]))
                .sum()
                .sum()
            )
            percent_events_filtered = (
                100
                * (initial_num_PSI_vals - num_PSI_vals_after_IR_filtering)
                / initial_num_PSI_vals
            )
            print(
                f"Number of PSI values after filtering IR events: {num_PSI_vals_after_IR_filtering} ({percent_events_filtered:.2f}% of initial values filtered)"
            )
            num_PSI_vals_of_each_type_after_IR_filtering = {}
            for event_type in self.inclusion_levels_full["COMPLEX"].unique():
                num_PSI_vals_of_each_type_after_IR_filtering[event_type] = (
                    (
                        ~np.isnan(
                            self.inclusion_levels_full.loc[
                                self.inclusion_levels_full["COMPLEX"] == event_type,
                                self.psi_vals_columns,
                            ]
                        )
                    )
                    .sum()
                    .sum()
                )
            print(
                f"Number of PSI values of each type after filtering IR events: {num_PSI_vals_of_each_type_after_IR_filtering}"
            )
            # remove events with NaN PSI values in all samples
            filter_out_events_with_all_PSI_values_NaN = np.all(
                self.inclusion_levels_full[self.psi_vals_columns].isna(), axis=1
            )
            self.inclusion_levels_full = self.inclusion_levels_full.loc[
                ~filter_out_events_with_all_PSI_values_NaN
            ].reset_index(drop=True)
            print(
                f"Number of events of each type after dropping events with no valid measurements: {self.inclusion_levels_full['COMPLEX'].value_counts()}"
            )

            # print number of events of each type
            print(
                f"Final number of events of each type: {self.inclusion_levels_full['COMPLEX'].value_counts()}"
            )

            # cache the filtered data
            self.inclusion_levels_full.to_csv(
                os.path.join(self.cache_dir, "inclusion_levels_full_filtered.csv"),
                index=False,
            )

            # cache the gene counts
            self.gene_counts.to_csv(
                os.path.join(self.cache_dir, "gene_counts_filtered.csv"), index=False
            )

            print("Filtered data cached")

        else:
            print("Loading filtered data from cache")
            self.inclusion_levels_full = pd.read_csv(
                os.path.join(self.cache_dir, "inclusion_levels_full_filtered.csv")
            )
            self.gene_counts = pd.read_csv(
                os.path.join(self.cache_dir, "gene_counts_filtered.csv")
            )
            with open(
                os.path.join(self.cache_dir, "gene_name_to_ensembl_id.json"), "r"
            ) as f:
                gene_name_to_ensembl_id = json.load(f)
            with open(
                os.path.join(self.cache_dir, "ensembl_id_to_gene_name.json"), "r"
            ) as f:
                ensembl_id_to_gene_name = json.load(f)

            self.psi_vals_columns = [
                i
                for i in self.inclusion_levels_full.columns[6:]
                if not i.endswith("-Q")
            ]
            self.quality_columns = [
                i for i in self.inclusion_levels_full.columns[6:] if i.endswith("-Q")
            ]

        # load the genome
        os.makedirs(os.path.join(self.cache_dir, "genomes"), exist_ok=True)
        self.genome = genomepy.Genome(
            "hg38", genomes_dir=os.path.join(self.cache_dir, "genomes")
        )  # only need hg38 since the data is from human cell lines

        # add a column for chromosome number
        self.inclusion_levels_full["CHR"] = self.inclusion_levels_full["COORD"].apply(
            lambda x: x.split(":")[0]
        )
