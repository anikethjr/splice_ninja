# Processes data published by Rogalska et al. (2024) - Transcriptome-wide splicing network reveals specialized regulatory functions of the core spliceosome

import numpy as np
import pandas as pd
import os
import pdb
import json
from statsmodels.stats.multitest import multipletests

import genomepy

import torch
from torch.utils.data import Dataset, DataLoader

import lightning as L
from lightning.pytorch import LightningDataModule

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
    def __init__(self, config):
        self.config = config
        self.input_size = self.config["train_config"]["input_size"]
        self.train_chromosomes = self.config["data_config"]["train_chromosomes"]
        self.test_chromosomes = self.config["data_config"]["test_chromosomes"]
        self.val_chromosomes = self.config["data_config"]["val_chromosomes"]

        # load gene counts
        self.gene_counts = pd.read_csv(
            os.path.join(self.config["data_config"]["data_dir"], "geneCounts.tab"),
            sep="\t",
            index_col=0,
        )
        self.gene_counts = self.gene_counts.rename({"X": "gene_id"}, axis=1)

        # cache directory
        self.cache_dir = self.config["data_config"]["cache_dir"]
        if not os.path.exists(self.cache_dir):
            os.makedirs(self.cache_dir, exist_ok=True)

        # load/create the filtered splicing data
        if not os.path.exists(
            os.path.join(self.cache_dir, "inclusion_levels_full_filtered.csv")
        ):
            # load psi values
            # data was provided in the VastTools output format, more details on the data format are here - https://github.com/vastgroup/vast-tools?tab=readme-ov-file#combine-output-format
            self.inclusion_levels_full = pd.read_csv(
                os.path.join(
                    self.config["data_config"]["data_dir"],
                    "INCLUSION_LEVELS_FULL-Hs2331.tab",
                ),
                sep="\t",
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

            # discard columns corresponding to the following samples due to poor quality of the data
            # AA2, AA1, CCDC12, C1orf55, C1orf55_b, CDC5L, HFM1, LENG1, RBM17, PPIL1, SRRM4, SRRT
            self.psi_vals_columns = [
                i
                for i in self.psi_vals_columns
                if i
                not in [
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
            self.quality_columns = [
                i
                for i in self.quality_columns
                if i[: -len("-Q")]
                not in [
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
            assert len(self.psi_vals_columns) == len(self.quality_columns)

            # authors of original work use data from second replicate for the following splicing factors:
            # LENG1 (called LENG1_b), RBM17 (called RBM17con), HFM1 (called HFM1_b), CCDC12 (called CCDC12_b), CDC5L (called CDC5L_b)
            # (from https://github.com/estepi/SpliceNet/blob/main/prepareALLTable.R#L20-L29)
            # thus, we drop the columns for the first replicate of these splicing factors and rename the columns for the second replicate
            drop_columns = ["LENG1", "RBM17", "HFM1", "CCDC12", "CDC5L"] + [
                i + "-Q" for i in ["LENG1", "RBM17", "HFM1", "CCDC12", "CDC5L"]
            ]
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
            self.inclusion_levels_full = self.inclusion_levels_full.drop(
                columns=drop_columns
            )
            self.inclusion_levels_full = self.inclusion_levels_full.rename(
                columns=rename_dict
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
                assert f"{i}-Q" in self.quality_columns
            # assert len(self.psi_vals_columns) == len(self.quality_columns) == 305 # there are 319 for some reason - need to check, maybe there are additional filters that need to be applied

            # print some statistics about the data
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
            print("Filtered data cached")

        else:
            print("Loading filtered data from cache")
            self.inclusion_levels_full = pd.read_csv(
                os.path.join(self.cache_dir, "inclusion_levels_full_filtered.csv")
            )
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
