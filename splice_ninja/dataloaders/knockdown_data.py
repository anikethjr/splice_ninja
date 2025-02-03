# Processes data published by Rogalska et al. (2024) - Transcriptome-wide splicing network reveals specialized regulatory functions of the core spliceosome

import numpy as np
import pandas as pd
import os
import pdb
import json
import urllib
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
        self.input_size = self.data_module.config["train_config"]["input_size"]
        self.intron_keep_ratio = self.data_module.config["train_config"][
            "intron_keep_ratio"
        ]
        self.genome: genomepy.Genome = self.data_module.genome

        if self.split == "train":
            self.chromosomes = self.data_module.train_chromosomes
            self.flattened_inclusion_levels = (
                self.data_module.train_flattened_inclusion_levels_full
            )
            self.event_info = self.data_module.train_event_info
            self.introns_around_splicing_events = (
                self.data_module.train_introns_around_splicing_events
            )

        elif self.split == "val":
            self.chromosomes = self.data_module.val_chromosomes
            self.flattened_inclusion_levels = (
                self.data_module.val_flattened_inclusion_levels_full
            )
            self.event_info = self.data_module.val_event_info
            self.introns_around_splicing_events = (
                self.data_module.val_introns_around_splicing_events
            )

        elif self.split == "test":
            self.chromosomes = self.data_module.test_chromosomes
            self.flattened_inclusion_levels = (
                self.data_module.test_flattened_inclusion_levels_full
            )
            self.event_info = self.data_module.test_event_info
            self.introns_around_splicing_events = (
                self.data_module.test_introns_around_splicing_events
            )

        # remove events that are larger than the input size
        original_num_events = len(self.event_info)
        original_num_flattened_inclusion_levels = len(self.flattened_inclusion_levels)
        self.event_info = self.event_info[
            self.event_info["LENGTH"] <= self.input_size
        ].reset_index(drop=True)
        self.flattened_inclusion_levels = self.flattened_inclusion_levels[
            self.flattened_inclusion_levels["EVENT"].isin(self.event_info["EVENT"])
        ].reset_index(drop=True)
        print(
            f"{split} split - Removed {original_num_events - len(self.event_info)} events that are larger than the input size ({(original_num_events - len(self.event_info)) / original_num_events:.2f}%)"
        )
        print(
            f"{split} split - Removed {original_num_flattened_inclusion_levels - len(self.flattened_inclusion_levels)} rows from the flattened inclusion levels dataframe that correspond to events that are larger than the input size ({(original_num_flattened_inclusion_levels - len(self.flattened_inclusion_levels)) / original_num_flattened_inclusion_levels:.2f}%)"
        )

        # for introns that are larger than the input size, we keep the beginning and end of the intron and the middle part is randomly sampled
        self.introns_around_splicing_events[
            "LENGTH"
        ] = self.introns_around_splicing_events["COORD"].apply(
            lambda x: int(x.split(":")[-1].split("-")[1])
            - int(x.split(":")[-1].split("-")[0])
            + 1
        )
        num_introns_longer_than_input_size = (
            self.introns_around_splicing_events["LENGTH"] > self.input_size
        ).sum()
        print(
            f"{split} split - Number of introns longer than the input size: {num_introns_longer_than_input_size} ({num_introns_longer_than_input_size / len(self.introns_around_splicing_events):.2f}%)"
        )

    def __len__(self):
        return len(self.flattened_inclusion_levels) + np.ceil(
            len(self.introns_around_splicing_events) * self.intron_keep_ratio
        ).astype(int)

    def get_psi_val(self, idx):
        # get the PSI value for the idx-th row in the flattened_inclusion_levels dataframe
        psi_row = self.flattened_inclusion_levels.iloc[idx]
        event_id = psi_row["EVENT"]
        event_type = psi_row["EVENT_TYPE"]
        sample = psi_row["SAMPLE"]
        psi_val = psi_row["PSI"]

        # get the event information for sequence construction
        event_row = self.event_info[
            (self.event_info["EVENT"] == event_id)
            & (self.event_info["EVENT_TYPE"] == event_type)
        ]
        assert len(event_row) == 1
        event_row = event_row.iloc[0]
        gene_id = event_row["GENE_ID"]
        has_gene_exp_values = event_row["HAS_GENE_EXP_VALUES"]
        chrom = event_row["CHR"][3:]  # remove "chr" prefix
        strand = event_row["STRAND"]
        extraction_coordinates = event_row["EVENT_EXTRACTION_COORD"]
        extraction_start = int(extraction_coordinates.split(":")[-1].split("-")[0])
        extraction_end = int(extraction_coordinates.split(":")[-1].split("-")[1])

        # construct sequence
        # now compute input start and end coordinates based on the input size
        # we want to have the event in the middle of the input sequence
        background_sequence_length = self.input_size - (
            extraction_end - extraction_start + 1
        )
        # we have to pad the sequence with background sequence
        input_start = max(
            1,
            extraction_start - np.ceil(background_sequence_length / 2).astype(int),
        )
        input_end = min(self.genome.sizes[chrom], input_start + self.input_size - 1)
        if (input_end - input_start + 1) < self.input_size:
            input_start = max(
                1, input_end - self.input_size + 1
            )  # make sure the input size is exactly self.input_size
        assert (input_end - input_start + 1) == self.input_size
        # both idxs below are inclusive
        spliced_in_sequence_start_idx = extraction_start - input_start
        spliced_in_sequence_end_idx = spliced_in_sequence_start_idx + (
            extraction_end - extraction_start
        )
        if strand == "-":
            # need to account for the reverse complement
            spliced_in_sequence_start_idx, spliced_in_sequence_end_idx = (
                self.input_size - 1 - spliced_in_sequence_end_idx,
                self.input_size - 1 - spliced_in_sequence_start_idx,
            )

        sequence = self.genome.get_seq(
            chrom, input_start, input_end, rc=(strand == "-")
        ).seq
        sequence = sequence.upper()
        assert (
            len(sequence) == self.input_size
        ), f"Sequence is not of the correct length {len(sequence)} vs {self.input_size}"
        event_sequence = self.genome.get_seq(
            chrom, extraction_start, extraction_end, rc=(strand == "-")
        ).seq
        event_sequence = event_sequence.upper()
        assert len(event_sequence) == (
            extraction_end - extraction_start + 1
        ), f"Event sequence is not of the correct length: {len(event_sequence)} vs {(extraction_end - extraction_start + 1)}"
        assert (
            sequence[spliced_in_sequence_start_idx : spliced_in_sequence_end_idx + 1]
            == event_sequence
        ), "Event sequence is not at the correct position in the input sequence"

        # construct one-hot encoding
        one_hot_encoding = np.zeros((self.input_size, 4))
        base_to_idx = {"A": 0, "C": 1, "G": 2, "T": 3}
        for i, base in enumerate(sequence):
            if base in base_to_idx:
                one_hot_encoding[i, base_to_idx[base]] = 1

        # construct mask that indicates the positions of the event
        mask = np.zeros(self.input_size)
        mask[spliced_in_sequence_start_idx : spliced_in_sequence_end_idx + 1] = 1

        # get the gene expression values if they are available
        gene_exp_values = -1.0
        if has_gene_exp_values:
            gene_exp_values = self.data_module.normalized_gene_expression.loc[
                self.data_module.normalized_gene_expression["gene_id"] == gene_id,
                sample + "_log2TPM",
            ]
            assert (
                len(gene_exp_values) == 1
            ), f"Gene expression values not found, length of dataframe is {len(gene_exp_values)}: {gene_exp_values}"
            gene_exp_values = gene_exp_values.iloc[0]

        # get splicing factor expression values
        splicing_factor_exp_values = self.data_module.splicing_factor_expression_levels[
            sample + "_log2TPM"
        ].values.reshape(-1)

        return {
            "sequence": one_hot_encoding,
            "mask": mask,
            "psi_val": np.array([psi_val]),
            "gene_exp_values": gene_exp_values,
            "splicing_factor_exp_values": splicing_factor_exp_values,
            "is_intron": np.array([False]),
        }

    def get_intron(self, idx):
        # return a random intron
        intron_row = self.introns_around_splicing_events.sample().iloc[0]
        sample = intron_row["SAMPLE"]
        event_id = intron_row["EVENT"]
        event_type = intron_row["EVENT_TYPE"]
        intron_coordinates = intron_row["COORD"]
        chrom = intron_coordinates.split(":")[0][3:]  # remove "chr" prefix
        start = int(intron_coordinates.split(":")[-1].split("-")[0])
        end = int(intron_coordinates.split(":")[-1].split("-")[1])
        length = end - start + 1
        strand = intron_row["STRAND"]
        location = intron_row["LOCATION"]

        # get the event information for extracting expression values
        event_row = self.event_info[self.event_info["EVENT"] == event_id]
        assert len(event_row) == 1
        event_row = event_row.iloc[0]
        gene_id = event_row["GENE_ID"]
        has_gene_exp_values = event_row["HAS_GENE_EXP_VALUES"]

        # construct sequence
        # now compute input start and end coordinates based on the input size
        # we want to have the event in the middle of the input sequence
        # however, if the intron is larger than the input size, we keep the beginning and end of the intron and randomly sample the middle part
        if length > self.input_size:
            background_sequence_length = 0
            spliced_in_sequence_start_idx = 0
            spliced_in_sequence_end_idx = self.input_size - 1

            # one-third of the intron is kept on each side and the middle part is randomly sampled
            first_segment_start = start
            first_segment_end = start + np.ceil(self.input_size / 3).astype(int) - 1
            second_segment_start = end - np.ceil(self.input_size / 3).astype(int) + 1
            second_segment_end = end
            middle_segment_length = (
                self.input_size
                - (first_segment_end - first_segment_start + 1)
                - (second_segment_end - second_segment_start + 1)
            )
            middle_segment_start = np.random.randint(
                first_segment_end + 1, second_segment_start - middle_segment_length
            )
            middle_segment_end = middle_segment_start + middle_segment_length - 1

            if strand == ".":
                sequence = (
                    self.genome.get_seq(
                        chrom, first_segment_start, first_segment_end
                    ).seq
                    + self.genome.get_seq(
                        chrom, middle_segment_start, middle_segment_end
                    ).seq
                    + self.genome.get_seq(
                        chrom, second_segment_start, second_segment_end
                    ).seq
                )
            else:
                sequence = (
                    self.genome.get_seq(
                        chrom, second_segment_start, second_segment_end, rc=True
                    ).seq
                    + self.genome.get_seq(
                        chrom, middle_segment_start, middle_segment_end, rc=True
                    ).seq
                    + self.genome.get_seq(
                        chrom, first_segment_start, first_segment_end, rc=True
                    ).seq
                )

            sequence = sequence.upper()
            assert (
                len(sequence) == self.input_size
            ), f"Length of sequence is {len(sequence)} instead of {self.input_size}"

        else:
            background_sequence_length = self.input_size - length
            input_start = max(
                1,
                start - np.ceil(background_sequence_length / 2).astype(int),
            )
            input_end = min(self.genome.sizes[chrom], input_start + self.input_size - 1)
            if (input_end - input_start + 1) < self.input_size:
                input_start = max(1, input_end - self.input_size + 1)
            assert (input_end - input_start + 1) == self.input_size
            # both idxs below are inclusive
            spliced_in_sequence_start_idx = start - input_start
            spliced_in_sequence_end_idx = spliced_in_sequence_start_idx + length - 1
            if strand == "-":
                # need to account for reverse complement
                spliced_in_sequence_start_idx, spliced_in_sequence_end_idx = (
                    self.input_size - spliced_in_sequence_end_idx - 1,
                    self.input_size - spliced_in_sequence_start_idx - 1,
                )

            sequence = self.genome.get_seq(
                chrom, input_start, input_end, rc=(strand == "-")
            ).seq
            sequence = sequence.upper()
            assert (
                len(sequence) == self.input_size
            ), f"Length of sequence is {len(sequence)} instead of {self.input_size}"
            intron_sequence = self.genome.get_seq(
                chrom, start, end, rc=(strand == "-")
            ).seq
            intron_sequence = intron_sequence.upper()
            assert (
                len(intron_sequence) == length
            ), f"Length of intron sequence is {len(intron_sequence)} instead of {length}"
            assert (
                sequence[
                    spliced_in_sequence_start_idx : spliced_in_sequence_end_idx + 1
                ]
                == intron_sequence
            ), "Intron sequence is not at the correct position in the input sequence"

        # construct one-hot encoding
        one_hot_encoding = np.zeros((self.input_size, 4))
        base_to_idx = {"A": 0, "C": 1, "G": 2, "T": 3}
        for i, base in enumerate(sequence):
            if base in base_to_idx:
                one_hot_encoding[i, base_to_idx[base]] = 1

        # construct mask that indicates the positions of the event
        mask = np.zeros(self.input_size)
        mask[spliced_in_sequence_start_idx : spliced_in_sequence_end_idx + 1] = 1

        # get the gene expression values if they are available
        gene_exp_values = -1.0
        if has_gene_exp_values:
            gene_exp_values = self.data_module.normalized_gene_expression.loc[
                self.data_module.normalized_gene_expression["gene_id"] == gene_id,
                sample + "_log2TPM",
            ].iloc[0]

        # get splicing factor expression values
        splicing_factor_exp_values = self.data_module.splicing_factor_expression_levels[
            sample + "_log2TPM"
        ].values.reshape(-1)

        return {
            "sequence": one_hot_encoding,
            "mask": mask,
            "psi_val": np.array([0.0]),
            "gene_exp_values": gene_exp_values,
            "splicing_factor_exp_values": splicing_factor_exp_values,
            "is_intron": np.array([True]),
        }

    def __getitem__(self, idx):
        if idx < len(self.flattened_inclusion_levels):
            return self.get_psi_val(idx)
        else:
            return self.get_intron(idx - len(self.flattened_inclusion_levels))


# DataModule for the knockdown data
class KnockdownData(LightningDataModule):
    def prepare_data(self):
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
                    gene_name_to_ensembl_id = json.load(f)
            else:
                gene_name_to_ensembl_id = {}

            if os.path.exists(
                os.path.join(self.cache_dir, "ensembl_id_to_gene_name.json")
            ):
                with open(
                    os.path.join(self.cache_dir, "ensembl_id_to_gene_name.json"), "r"
                ) as f:
                    ensembl_id_to_gene_name = json.load(f)
            else:
                ensembl_id_to_gene_name = {}

            # load gene counts
            gene_counts = pd.read_csv(
                os.path.join(self.config["data_config"]["data_dir"], "geneCounts.tab"),
                sep="\t",
                index_col=0,
            )
            gene_counts = gene_counts.rename({"X": "gene_id"}, axis=1)
            print(
                "Read gene counts data, number of samples: {}".format(
                    gene_counts.shape[1] - 2
                )
            )

            # average counts from biological replicates - these are samples with suffixes "_r1" and "_r2"
            gene_counts_samples = gene_counts.columns[2:]
            rename_dict = {}
            for col in gene_counts_samples:
                if col.endswith("_r1"):
                    gene_counts[col] = (
                        gene_counts[col] + gene_counts[col[:-3] + "_r2"]
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
            gene_counts = gene_counts.drop(
                columns=[i for i in gene_counts.columns if i.endswith("_r2")]
            )
            gene_counts = gene_counts.rename(columns=rename_dict)
            print(
                "Averaged gene counts from biological replicates, number of samples: {}".format(
                    gene_counts.shape[1] - 2
                )
            )

            # start building a dictionary to map gene names to Ensembl gene IDs
            drop_columns = []
            for sf in tqdm(gene_counts.columns[2:]):
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
                if (gene_counts["alias"] == sf).sum() > 0:
                    ensembl_id = gene_counts.loc[
                        gene_counts["alias"] == sf, "gene_id"
                    ].iloc[0]
                    gene_name_to_ensembl_id[sf] = [ensembl_id]
                    if ensembl_id in ensembl_id_to_gene_name:
                        ensembl_id_to_gene_name[ensembl_id].append(sf)
                    else:
                        ensembl_id_to_gene_name[ensembl_id] = [sf]
            drop_columns = [i for i in drop_columns if i not in gene_name_to_ensembl_id]

            # drop columns for which we could not find the Ensembl gene ID
            gene_counts = gene_counts.drop(columns=drop_columns)
            print(
                "Dropping gene count data from {} splicing factors for which the Ensembl ID could not be found".format(
                    len(drop_columns)
                )
            )
            # also drop columns for which there is no corresponding gene count data in the rows
            drop_columns = []
            for sf in gene_counts.columns[2:]:
                check = False
                for ensembl_id in gene_name_to_ensembl_id[sf]:
                    if (gene_counts["gene_id"] == ensembl_id).sum() > 0:
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
                    if (gene_counts["alias"] == alias).sum() == 0:
                        drop_columns.append(sf)
                    else:
                        print(
                            "Weird case where the gene ID is not found but the alias is found for {}, using gene ID present in the gene count data".format(
                                sf
                            )
                        )
                        ensembl_id = gene_counts.loc[
                            gene_counts["alias"] == alias, "gene_id"
                        ].iloc[0]
                        gene_name_to_ensembl_id[sf] = [ensembl_id]
                        if ensembl_id in ensembl_id_to_gene_name:
                            ensembl_id_to_gene_name[ensembl_id].append(sf)
                        else:
                            ensembl_id_to_gene_name[ensembl_id] = [sf]
            gene_counts = gene_counts.drop(columns=drop_columns)
            print(
                "Dropping gene count data from {} splicing factors for which the gene ID could not be found in the gene count data".format(
                    len(drop_columns)
                )
            )

            # load psi values
            # data was provided in the VastTools output format, more details on the data format are here - https://github.com/vastgroup/vast-tools?tab=readme-ov-file#combine-output-format
            inclusion_levels_full = pd.read_csv(
                os.path.join(
                    self.config["data_config"]["data_dir"],
                    "INCLUSION_LEVELS_FULL-Hs2331.tab",
                ),
                sep="\t",
            )
            print(
                "Read PSI values data, number of samples: {}".format(
                    (inclusion_levels_full.shape[1] - 6) / 2
                )
            )

            # rename all the PSI and quality columns to remove the trailing "_1" at the end
            rename_dict = {}
            for col in inclusion_levels_full.columns[6:]:
                if col.endswith("_1"):
                    rename_dict[col] = col[:-2]
                elif col.endswith("_1-Q"):
                    rename_dict[col] = col[:-4] + "-Q"
            inclusion_levels_full = inclusion_levels_full.rename(columns=rename_dict)

            # get the columns for PSI values and quality - every PSI column is followed by a quality column
            # each PSI value is measured after knockdown of a specific splicing factor indicated by the column name
            psi_vals_columns = [
                i for i in inclusion_levels_full.columns[6:] if not i.endswith("-Q")
            ]
            quality_columns = [
                i for i in inclusion_levels_full.columns[6:] if i.endswith("-Q")
            ]
            assert len(psi_vals_columns) == len(quality_columns)
            print("Every PSI value is followed by a quality column in the raw data")

            # get ensembl gene IDs for the splicing factors being knocked down in the splicing data
            drop_columns = []
            for sf in tqdm(psi_vals_columns):
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
            inclusion_levels_full = inclusion_levels_full.drop(columns=drop_columns)
            print(
                "Dropping PSI values data from {} samples for which the Ensembl ID could not be found".format(
                    len(drop_columns) // 2
                )
            )
            psi_vals_columns = [
                i for i in inclusion_levels_full.columns[6:] if not i.endswith("-Q")
            ]
            quality_columns = [
                i for i in inclusion_levels_full.columns[6:] if i.endswith("-Q")
            ]
            assert len(psi_vals_columns) == len(quality_columns)
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
            inclusion_levels_full = inclusion_levels_full.drop(
                columns=drop_columns, errors="ignore"
            )
            print(
                "Discarded columns with poor quality data, number of samples: {}".format(
                    (inclusion_levels_full.shape[1] - 6) / 2
                )
            )
            psi_vals_columns = [
                i for i in inclusion_levels_full.columns[6:] if not i.endswith("-Q")
            ]
            quality_columns = [
                i for i in inclusion_levels_full.columns[6:] if i.endswith("-Q")
            ]
            assert len(psi_vals_columns) == len(quality_columns)
            print(
                "Every PSI value is followed by a quality column in the data after discarding poor quality data"
            )
            # drop these columns from the gene counts as well, if they exist + some specific rules to account for data weirdness
            drop_columns = []
            rename_dict = {}
            for col in [
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
            ]:
                if col in gene_counts.columns:
                    if col == "LENG1":
                        if "LENG1_b" in gene_counts.columns:
                            drop_columns.append("LENG1")
                            rename_dict["LENG1_b"] = "LENG1"
                        else:
                            print(
                                "We needed LENG1_b gene count data but it was not found, so using LENG1 data"
                            )
                    elif col == "RBM17":
                        if "RBM17con" in gene_counts.columns:
                            drop_columns.append("RBM17")
                            rename_dict["RBM17con"] = "RBM17"
                        else:
                            print(
                                "We needed RBM17con gene count data but it was not found, so using RBM17 data"
                            )
                    elif col == "HFM1":
                        if "HFM1_b" in gene_counts.columns:
                            drop_columns.append("HFM1")
                            rename_dict["HFM1_b"] = "HFM1"
                        else:
                            print(
                                "We needed HFM1_b gene count data but it was not found, so using HFM1 data"
                            )
                    elif col == "CCDC12":
                        if "CCDC12_b" in gene_counts.columns:
                            drop_columns.append("CCDC12")
                            rename_dict["CCDC12_b"] = "CCDC12"
                        else:
                            print(
                                "We needed CCDC12_b gene count data but it was not found, so using CCDC12 data"
                            )
                    elif col == "CDC5L":
                        if "CDC5L_b" in gene_counts.columns:
                            drop_columns.append("CDC5L")
                            rename_dict["CDC5L_b"] = "CDC5L"
                        else:
                            print(
                                "We needed CDC5L_b gene count data but it was not found, so using CDC5L data"
                            )
                    else:
                        drop_columns.append(col)
            gene_counts = gene_counts.drop(columns=drop_columns)
            gene_counts = gene_counts.rename(columns=rename_dict)
            print(
                "Dropped gene count data from samples for which the PSI value data was dropped due to poor quality, left with num samples = {}".format(
                    gene_counts.shape[1] - 2
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
            inclusion_levels_full = inclusion_levels_full.rename(
                columns=rename_dict, errors="ignore"
            )

            # we also drop all other replicate columns (endswith "_b" or "con") since we only use the data from the first replicate
            drop_columns = [
                i
                for i in inclusion_levels_full.columns
                if i.endswith("_b") or i.endswith("con")
            ]
            drop_columns += [i + "-Q" for i in drop_columns]
            inclusion_levels_full = inclusion_levels_full.drop(columns=drop_columns)
            psi_vals_columns = [
                i for i in inclusion_levels_full.columns[6:] if not i.endswith("-Q")
            ]
            quality_columns = [
                i for i in inclusion_levels_full.columns[6:] if i.endswith("-Q")
            ]
            for i in psi_vals_columns:
                assert f"{i}-Q" in quality_columns, f"Quality column for {i} not found"
            assert len(psi_vals_columns) == len(quality_columns)
            print(
                "Dropped unused replicate columns from PSI values data, number of samples: {}".format(
                    (inclusion_levels_full.shape[1] - 6) / 2
                )
            )

            # drop replicate columns from gene counts
            drop_columns = [
                i for i in gene_counts.columns if i.endswith("_b") or i.endswith("con")
            ]
            gene_counts = gene_counts.drop(columns=drop_columns)
            print(
                "Dropped unused replicate columns from gene count data, number of samples: {}".format(
                    gene_counts.shape[1] - 2
                )
            )

            # now rename all the columns to use Ensembl gene IDs instead of gene names since the two files don't always use the same gene names
            # if a gene has multiple Ensembl IDs, we use the one for which the gene id is found in the gene count data and has the cumulative highest expression
            rename_dict = {}
            for i in gene_counts.columns[2:]:
                max_count = 0
                best_ensembl_id = None
                for ensembl_id in gene_name_to_ensembl_id[i]:
                    if (gene_counts["gene_id"] == ensembl_id).sum() > 0:
                        exp_count = (
                            gene_counts[gene_counts["gene_id"] == ensembl_id]
                            .iloc[0][gene_counts.columns[2:]]
                            .sum()
                        )
                        if exp_count > max_count:
                            max_count = exp_count
                            best_ensembl_id = ensembl_id
                if best_ensembl_id is None:
                    raise Exception(f"Could not find the gene count data for {i}")

                rename_dict[i] = best_ensembl_id
            gene_counts = gene_counts.rename(columns=rename_dict)

            # now rename the columns in the PSI values data
            # we also rename the quality columns
            # if a gene has multiple Ensembl IDs, we use the one which was used for the gene count data
            rename_dict = {}
            for i in psi_vals_columns:
                max_count = 0
                best_ensembl_id = gene_name_to_ensembl_id[i][0]
                for ensembl_id in gene_name_to_ensembl_id[i]:
                    if ensembl_id in gene_counts.columns:
                        exp_count = (
                            gene_counts[gene_counts["gene_id"] == ensembl_id]
                            .iloc[0][gene_counts.columns[2:]]
                            .sum()
                        )
                        if exp_count > max_count:
                            max_count = exp_count
                            best_ensembl_id = ensembl_id

                rename_dict[i] = best_ensembl_id
                rename_dict[i + "-Q"] = best_ensembl_id + "-Q"
            inclusion_levels_full = inclusion_levels_full.rename(columns=rename_dict)

            # now drop all splicing data for which the gene count data is not available and vice versa
            drop_columns = []
            psi_vals_columns = [
                i for i in inclusion_levels_full.columns[6:] if not i.endswith("-Q")
            ]
            for sf in psi_vals_columns:
                if sf not in gene_counts.columns:
                    drop_columns.append(sf)
                    drop_columns.append(sf + "-Q")
            inclusion_levels_full = inclusion_levels_full.drop(columns=drop_columns)
            print(
                "Dropping PSI values data from {} samples for which the gene count data could not be found".format(
                    len(drop_columns) // 2
                )
            )
            psi_vals_columns = [
                i for i in inclusion_levels_full.columns[6:] if not i.endswith("-Q")
            ]

            drop_columns = []
            for sf in gene_counts.columns[2:]:
                if sf not in psi_vals_columns:
                    drop_columns.append(sf)
            gene_counts = gene_counts.drop(columns=drop_columns)
            print(
                "Dropping gene count data from {} samples for which the PSI value data could not be found".format(
                    len(drop_columns)
                )
            )

            psi_vals_columns = [
                i for i in inclusion_levels_full.columns[6:] if not i.endswith("-Q")
            ]
            quality_columns = [
                i for i in inclusion_levels_full.columns[6:] if i.endswith("-Q")
            ]
            assert set(gene_counts.columns[2:]) == set(
                psi_vals_columns
            )  # check that the columns match
            assert len(psi_vals_columns) == len(
                set(psi_vals_columns)
            ), "Duplicate columns found in PSI values data"
            assert len(quality_columns) == len(
                set(quality_columns)
            ), "Duplicate columns found in quality data"
            assert len(set(gene_counts.columns[2:])) == (
                gene_counts.shape[1] - 2
            ), "Duplicate columns found in gene count data"
            for i in psi_vals_columns:
                assert f"{i}-Q" in quality_columns, f"Quality column for {i} not found"
            assert len(psi_vals_columns) == len(quality_columns)
            assert len(psi_vals_columns) == (gene_counts.shape[1] - 2)
            print(
                "After dropping splicing data samples for which the gene count data could not be found and vice versa, number of samples: {}".format(
                    (inclusion_levels_full.shape[1] - 6) / 2
                )
            )

            # print some statistics about the data
            print(f"Number of samples: {len(psi_vals_columns)}")
            initial_num_PSI_vals = (
                (~np.isnan(inclusion_levels_full[psi_vals_columns])).sum().sum()
            )
            print(f"Initial number of PSI values: {initial_num_PSI_vals}")
            number_of_PSI_vals_of_each_type = {}
            for event_type in inclusion_levels_full["COMPLEX"].unique():
                number_of_PSI_vals_of_each_type[event_type] = (
                    (
                        ~np.isnan(
                            inclusion_levels_full.loc[
                                inclusion_levels_full["COMPLEX"] == event_type,
                                psi_vals_columns,
                            ]
                        )
                    )
                    .sum()
                    .sum()
                )
            print(
                f"Number of PSI values of each type:\n{number_of_PSI_vals_of_each_type}"
            )
            # remove events with NaN PSI values in all samples
            filter_out_events_with_all_PSI_values_NaN = np.all(
                inclusion_levels_full[psi_vals_columns].isna(), axis=1
            )
            inclusion_levels_full = inclusion_levels_full.loc[
                ~filter_out_events_with_all_PSI_values_NaN
            ].reset_index(drop=True)
            print(
                f"Number of events of each type after dropping events with no valid measurements:\n{inclusion_levels_full['COMPLEX'].value_counts()}"
            )

            # filter out PSI values which did not pass the quality control
            # first value of the first comma-separated list of values in the quality column is the quality control flag
            # (from https://github.com/estepi/SpliceNet/blob/main/replaceNAI.R#L23-L31)
            for i in quality_columns:
                inclusion_levels_full.loc[
                    inclusion_levels_full[i].apply(
                        lambda x: "OK" not in x.split(",")[0]
                    ),
                    i[:-2],
                ] = np.nan
            num_PSI_vals_after_quality_control_filtering = (
                (~np.isnan(inclusion_levels_full[psi_vals_columns])).sum().sum()
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
            for event_type in inclusion_levels_full["COMPLEX"].unique():
                num_PSI_vals_of_each_type_after_quality_control_filtering[
                    event_type
                ] = (
                    (
                        ~np.isnan(
                            inclusion_levels_full.loc[
                                inclusion_levels_full["COMPLEX"] == event_type,
                                psi_vals_columns,
                            ]
                        )
                    )
                    .sum()
                    .sum()
                )
            print(
                f"Number of PSI values of each type after filtering events which did not pass the quality control:\n{num_PSI_vals_of_each_type_after_quality_control_filtering}"
            )
            # remove events with NaN PSI values in all samples
            filter_out_events_with_all_PSI_values_NaN = np.all(
                inclusion_levels_full[psi_vals_columns].isna(), axis=1
            )
            inclusion_levels_full = inclusion_levels_full.loc[
                ~filter_out_events_with_all_PSI_values_NaN
            ].reset_index(drop=True)
            print(
                f"Number of events of each type after dropping events with no valid measurements:\n{inclusion_levels_full['COMPLEX'].value_counts()}"
            )

            # filter out intron retention (IR) PSI values where the corrected p-value of a binomial test of balance between reads mapping to the upstream and downstream exon-intron junctions is less than 0.05,
            # indicating a significant imbalance in the reads mapping to the two junctions and therefore a false positive IR event
            # the raw p-value is stored in the quality column for each PSI value - last number before the @ symbol in the string (@ separates two comma-separated lists of values)
            # correction is performed using the Holm method by pooling p-values across all events observed in the same sample
            # (from https://github.com/estepi/SpliceNet/blob/main/replaceNAI.R#L71-L78)
            is_IR = inclusion_levels_full["COMPLEX"] == "IR"
            for i in quality_columns:
                IR_events_raw_pvals = inclusion_levels_full.loc[is_IR, i].apply(
                    lambda x: float(x.split("@")[0].split(",")[-1])
                    if x.split("@")[0].split(",")[-1] != "NA"
                    else 1
                )
                IR_events_corrected_pvals = multipletests(
                    IR_events_raw_pvals, alpha=0.05, method="holm"
                )[1]
                # replace the PSI values of IR events with NaN if the corrected p-value is less than 0.05
                filter_out = np.zeros(inclusion_levels_full.shape[0], dtype=bool)
                filter_out[is_IR] = IR_events_corrected_pvals < 0.05
                inclusion_levels_full.loc[filter_out, i[:-2]] = np.nan
            num_PSI_vals_after_IR_filtering = (
                (~np.isnan(inclusion_levels_full[psi_vals_columns])).sum().sum()
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
            for event_type in inclusion_levels_full["COMPLEX"].unique():
                num_PSI_vals_of_each_type_after_IR_filtering[event_type] = (
                    (
                        ~np.isnan(
                            inclusion_levels_full.loc[
                                inclusion_levels_full["COMPLEX"] == event_type,
                                psi_vals_columns,
                            ]
                        )
                    )
                    .sum()
                    .sum()
                )
            print(
                f"Number of PSI values of each type after filtering IR events:\n{num_PSI_vals_of_each_type_after_IR_filtering}"
            )
            # remove events with NaN PSI values in all samples
            filter_out_events_with_all_PSI_values_NaN = np.all(
                inclusion_levels_full[psi_vals_columns].isna(), axis=1
            )
            inclusion_levels_full = inclusion_levels_full.loc[
                ~filter_out_events_with_all_PSI_values_NaN
            ].reset_index(drop=True)
            print(
                f"Number of events of each type after dropping events with no valid measurements:\n{inclusion_levels_full['COMPLEX'].value_counts()}"
            )

            # filter out ALTA/ALTD events where either the start or end coordinates of the event are missing -- need to figure out why this happens, but for now we filter them out
            filter_out_ALTA_ALTD_events_with_missing_coordinates = np.zeros(
                inclusion_levels_full.shape[0], dtype=bool
            )
            for i, row in inclusion_levels_full.iterrows():
                if "ALTA" in row["EVENT"] or "ALTD" in row["EVENT"]:
                    coord = row["COORD"]  # format is chr:start-end
                    start, end = coord.strip().split(":")[-1].split("-")
                    if start == "" or end == "":
                        filter_out_ALTA_ALTD_events_with_missing_coordinates[i] = True
                    else:
                        assert int(start) < int(end)
            inclusion_levels_full = inclusion_levels_full.loc[
                ~filter_out_ALTA_ALTD_events_with_missing_coordinates
            ].reset_index(drop=True)
            num_PSI_vals_after_ALTA_ALTD_events_with_missing_coordinates_filtering = (
                (~np.isnan(inclusion_levels_full[psi_vals_columns])).sum().sum()
            )
            percent_events_filtered = (
                100
                * (
                    initial_num_PSI_vals
                    - num_PSI_vals_after_ALTA_ALTD_events_with_missing_coordinates_filtering
                )
                / initial_num_PSI_vals
            )
            print(
                f"Number of PSI values after filtering ALTA/ALTD events with missing start/end coordinates: {num_PSI_vals_after_ALTA_ALTD_events_with_missing_coordinates_filtering} ({percent_events_filtered:.2f}% of initial values filtered)"
            )
            num_PSI_vals_of_each_type_after_ALTA_ALTD_events_with_missing_coordinates_filtering = (
                {}
            )
            for event_type in inclusion_levels_full["COMPLEX"].unique():
                num_PSI_vals_of_each_type_after_ALTA_ALTD_events_with_missing_coordinates_filtering[
                    event_type
                ] = (
                    (
                        ~np.isnan(
                            inclusion_levels_full.loc[
                                inclusion_levels_full["COMPLEX"] == event_type,
                                psi_vals_columns,
                            ]
                        )
                    )
                    .sum()
                    .sum()
                )
            print(
                f"Number of PSI values of each type after filtering ALTA/ALTD events with missing start/end coordinates:\n{num_PSI_vals_of_each_type_after_ALTA_ALTD_events_with_missing_coordinates_filtering}"
            )
            # remove events with NaN PSI values in all samples
            filter_out_events_with_all_PSI_values_NaN = np.all(
                inclusion_levels_full[psi_vals_columns].isna(), axis=1
            )
            inclusion_levels_full = inclusion_levels_full.loc[
                ~filter_out_events_with_all_PSI_values_NaN
            ].reset_index(drop=True)
            print(
                f"Number of events of each type after dropping events with no valid measurements:\n{inclusion_levels_full['COMPLEX'].value_counts()}"
            )

            # print number of events of each type
            print(
                f"Final number of events of each type:\n{inclusion_levels_full['COMPLEX'].value_counts()}"
            )

            # cache the filtered data
            inclusion_levels_full.to_csv(
                os.path.join(self.cache_dir, "inclusion_levels_full_filtered.csv"),
                index=False,
            )

            # cache the gene counts
            gene_counts.to_csv(
                os.path.join(self.cache_dir, "gene_counts_filtered.csv"), index=False
            )

            print("Filtered data cached")

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

        if (
            not os.path.exists(
                os.path.join(
                    self.cache_dir, "flattened_inclusion_levels_full_filtered.csv"
                )
            )
            or not os.path.exists(
                os.path.join(self.cache_dir, "event_info_filtered.csv")
            )
            or not os.path.exists(
                os.path.join(self.cache_dir, "intron_around_splicing_events.csv")
            )
        ):
            # flatten the filtered data - make a row for each sample for each event and remove NaN values
            # this makes it to create a dataset for training
            # to avoid duplicating event information, we only keep the columns for the event and the PSI value
            # another file is created with the event information like event type, coordinates, etc.
            # we also create a file with the intron sequence around the splicing events - this can be used for dataset augmentation

            print("Flattening the filtered data")

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
                    os.path.join(
                        self.cache_dir, "VastDB", "hg38", "GENE_INFO-hg38.tab.gz"
                    ),
                )
                print("Gene information downloaded")
            if not os.path.exists(
                os.path.join(
                    self.cache_dir, "VastDB", "hg38", "EVENTID_to_GENEID-hg38.tab.gz"
                )
            ):
                print("Downloading event ID to gene ID mapping from VastDB")
                url = (
                    "https://vastdb.crg.eu/downloads/hg38/EVENTID_to_GENEID-hg38.tab.gz"
                )
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

            # load VastDB data
            event_info_from_vastdb = pd.read_csv(
                os.path.join(
                    self.cache_dir, "VastDB", "hg38", "EVENT_INFO-hg38.tab.gz"
                ),
                sep="\t",
            )
            gene_info_from_vastdb = pd.read_csv(
                os.path.join(self.cache_dir, "VastDB", "hg38", "GENE_INFO-hg38.tab.gz"),
                sep="\t",
            )
            event_id_to_gene_id_from_vastdb = pd.read_csv(
                os.path.join(
                    self.cache_dir, "VastDB", "hg38", "EVENTID_to_GENEID-hg38.tab.gz"
                ),
                sep="\t",
            )

            # load the filtered data
            gene_counts = pd.read_csv(
                os.path.join(self.cache_dir, "gene_counts_filtered.csv")
            )
            inclusion_levels_full = pd.read_csv(
                os.path.join(self.cache_dir, "inclusion_levels_full_filtered.csv")
            )

            psi_vals_columns = [
                i for i in inclusion_levels_full.columns[6:] if not i.endswith("-Q")
            ]

            # create a column for the chromosome
            inclusion_levels_full["CHR"] = inclusion_levels_full["COORD"].apply(
                lambda x: x.split(":")[0]
            )

            # join inclusion levels data with event info to get reference exon coordinates
            inclusion_levels_full = inclusion_levels_full.merge(
                event_info_from_vastdb[["EVENT", "COORD_o", "CO_C1", "CO_A", "CO_C2"]],
                left_on=["EVENT", "COORD"],
                right_on=["EVENT", "COORD_o"],
                how="left",
            ).reset_index(drop=True)
            event_found_mask = inclusion_levels_full["COORD_o"].notnull()
            assert (
                inclusion_levels_full[event_found_mask]["COORD"]
                == inclusion_levels_full[event_found_mask]["COORD_o"]
            ).all(), "Coordinates do not match between inclusion levels data and event info data from VastDB"
            print(
                "Number of events found in VastDB: {} ({}%)".format(
                    event_found_mask.sum(), 100 * event_found_mask.mean()
                )
            )
            print(
                "Number of events not found in VastDB: {} ({}%)".format(
                    (~event_found_mask).sum(), 100 * (~event_found_mask).mean()
                )
            )
            print("Some event IDs not found in VastDB:")
            print(
                inclusion_levels_full.loc[~event_found_mask, "EVENT"].head()
            )  # print some event IDs not found in VastDB
            print("Per event type:")
            for event_type in inclusion_levels_full["COMPLEX"].unique():
                event_found_mask = inclusion_levels_full.loc[
                    inclusion_levels_full["COMPLEX"] == event_type, "COORD_o"
                ].notnull()
                print(
                    f"{event_type}: {event_found_mask.sum()} ({100 * event_found_mask.mean():.2f}%), not found: {(~event_found_mask).sum()} ({100 * (~event_found_mask).mean():.2f}%)"
                )
                print(
                    "Some event IDs not found in VastDB for {}: {}".format(
                        event_type,
                        inclusion_levels_full.loc[
                            (inclusion_levels_full["COMPLEX"] == event_type)
                            & (~event_found_mask),
                            "EVENT",
                        ].head(),
                    )
                )
            inclusion_levels_full = inclusion_levels_full.drop(columns=["COORD_o"])

            # join inclusion levels data with gene info to get gene ID
            inclusion_levels_full = inclusion_levels_full.merge(
                gene_info_from_vastdb[["GeneID", "Gene_name"]],
                left_on="GENE",
                right_on="Gene_name",
                how="left",
            )
            inclusion_levels_full = inclusion_levels_full.drop(columns=["Gene_name"])
            inclusion_levels_full = inclusion_levels_full.rename(
                columns={"GeneID": "GENE_ID"}
            )

            temp = inclusion_levels_full[["GENE", "GENE_ID"]].drop_duplicates()
            print(
                "Number of genes for which gene ID was found in VastDB: {} ({}%)".format(
                    temp[temp["GENE_ID"].notnull()].shape[0],
                    100 * temp[temp["GENE_ID"].notnull()].shape[0] / temp.shape[0],
                )
            )
            print(
                "Number of genes for which gene ID was not found in VastDB: {} ({}%)".format(
                    temp[temp["GENE_ID"].isnull()].shape[0],
                    100 * temp[temp["GENE_ID"].isnull()].shape[0] / temp.shape[0],
                )
            )
            print("Some genes for which gene ID was not found in VastDB:")
            print(temp[temp["GENE_ID"].isnull()].head())

            # create schemas for the flattened data and the event information
            flattened_inclusion_levels_full = {}
            flattened_inclusion_levels_full["EVENT"] = []  # event ID
            flattened_inclusion_levels_full["EVENT_TYPE"] = []  # general event type
            flattened_inclusion_levels_full[
                "SAMPLE"
            ] = []  # knocked down splicing factor i.e. sample name
            flattened_inclusion_levels_full["PSI"] = []  # PSI value

            event_info = {}
            event_info["EVENT"] = []  # event ID
            event_info["EVENT_TYPE"] = []  # general event type
            event_info["GENE"] = []  # gene name
            event_info["GENE_ID"] = []  # Ensembl gene ID
            event_info[
                "HAS_GENE_EXP_VALUES"
            ] = (
                []
            )  # whether gene expression values are available for the gene in the gene count data
            event_info["COORD"] = []  # coordinates encompassing the event
            event_info["LENGTH"] = []  # length of the event
            event_info["FullCO"] = []  # full coordinates of the event
            event_info["COMPLEX"] = []  # fine-grained event type
            event_info["CHR"] = []  # chromosome
            event_info["STRAND"] = []  # strand
            event_info[
                "EVENT_EXTRACTION_COORD"
            ] = (
                []
            )  # these are the coordinates for the alternative splicing event extracted from the VastDB output, and the inclusion levels are measured for this genome segment

            introns_around_splicing_events = {}
            introns_around_splicing_events[
                "EVENT"
            ] = []  # event ID corresponding to the splicing event
            introns_around_splicing_events["EVENT_TYPE"] = []  # general event type
            introns_around_splicing_events[
                "SAMPLE"
            ] = (
                []
            )  # knocked down splicing factor, only introns with low reads mapped to it in the sample are included - determined based on complexity score of the event in the sample (S is the only acceptable complexity score meaning that less than 5% of the reads mapped to non-reference splice junctions)
            introns_around_splicing_events[
                "LOCATION"
            ] = []  # whether the intron is upstream or downstream of the splicing event
            introns_around_splicing_events[
                "COORD"
            ] = []  # coordinates of the intron around the splicing event
            introns_around_splicing_events["STRAND"] = []  # strand of the intron

            all_gene_ids_with_expression_values = set(gene_counts["gene_id"])

            # iterate over each row in the data and populate the flattened data and event information
            for i, row in tqdm(
                inclusion_levels_full.iterrows(), total=inclusion_levels_full.shape[0]
            ):
                # VAST-DB event ID. Formed by:
                # - Species identifier: Hsa (Human), Mmu (Mouse), or Gga (Chicken);
                # - Type of alternative splicing event:
                #    alternative exon skipping (EX),
                #    retained intron (INT),
                #    alternative splice site donor choice (ALTD), or alternative splice site acceptor choice (ALTA).
                #       In the case of ALTD/ALTA, each splice site within the event is indicated (from exonic internal to external) over the
                #       total number of alternative splice sites in the event (e.g. HsaALTA0000011-1/2).
                # - Numerical identifier.
                event_info["EVENT"].append(row["EVENT"])
                event_type = None
                if "EX" in row["EVENT"]:
                    event_type = "EX"
                elif "INT" in row["EVENT"]:
                    event_type = "INT"
                elif "ALTD" in row["EVENT"]:
                    event_type = "ALTD"
                elif "ALTA" in row["EVENT"]:
                    event_type = "ALTA"
                else:
                    raise Exception(
                        f"Unknown event type for event with ID: {row['EVENT']}"
                    )
                event_info["EVENT_TYPE"].append(event_type)

                for psi_col in psi_vals_columns:
                    if not np.isnan(row[psi_col]):
                        flattened_inclusion_levels_full["EVENT"].append(row["EVENT"])
                        flattened_inclusion_levels_full["EVENT_TYPE"].append(event_type)
                        flattened_inclusion_levels_full["SAMPLE"].append(psi_col)
                        flattened_inclusion_levels_full["PSI"].append(row[psi_col])

                event_info["GENE"].append(row["GENE"])
                event_info["GENE_ID"].append(row["GENE_ID"])
                event_info["HAS_GENE_EXP_VALUES"].append(
                    (not pd.isna(row["GENE_ID"]))
                    and row["GENE_ID"] in all_gene_ids_with_expression_values
                )

                event_info["COORD"].append(row["COORD"])
                event_info["LENGTH"].append(row["LENGTH"])
                event_info["FullCO"].append(row["FullCO"])
                event_info["COMPLEX"].append(row["COMPLEX"])

                # the FullCO format is as follows:
                # - For EX: chromosome:C1donor,Aexon,C2acceptor. Where C1donor is the "reference" upstream exon's donor, C2acceptor the "reference" downstream exon's acceptor, and A the alternative exon.
                # Strand is "+" if C1donor < C2acceptor. If multiple acceptor/donors exist in any of the exons, they are shown separated by "+".
                # NOTE: The "reference" upstream and downstream C1/C2 coordinates are not necessarily the closest upstream and downstream C1/C2 exons, but the most external ones with sufficient support (to facilitate primer design, etc).                #
                # - For ALTD: chromosome:Aexon,C2acceptor. Multiple donors of the event are separated by "+".
                # - For ALTA: chromosome:C1donor,Aexon. Multiple acceptors of the event are separated by "+".
                # - For INT: chromosome:C1exon=C2exon:strand.
                event_info["CHR"].append(row["FullCO"].split(":")[0])
                if event_type == "EX":
                    C1donor, Aexon, C2acceptor = row["FullCO"].split(":")[1].split(",")
                    C1donor = [int(i) for i in C1donor.split("+") if i != ""]
                    C2acceptor = [int(i) for i in C2acceptor.split("+") if i != ""]
                    Aexon_5p_ends = [int(i) for i in Aexon.split("-")[0].split("+")]
                    Aexon_3p_ends = [int(i) for i in Aexon.split("-")[1].split("+")]

                    strand = None
                    if C1donor[0] < C2acceptor[0]:
                        event_info["STRAND"].append(".")
                        strand = "."
                    else:
                        event_info["STRAND"].append("-")
                        strand = "-"

                    extraction_start = int(row["COORD"].split(":")[1].split("-")[0])
                    extraction_end = int(row["COORD"].split(":")[1].split("-")[1])
                    assert (
                        extraction_start < extraction_end
                    ), f"Invalid extraction coordinates: {extraction_start}-{extraction_end}"
                    event_info["EVENT_EXTRACTION_COORD"].append(
                        f"{row['CHR']}:{extraction_start}-{extraction_end}"
                    )

                    if not pd.isna(row["CO_C1"]) and not pd.isna(
                        row["CO_C2"]
                    ):  # only add the introns if the reference coordinates are available
                        # add the introns around the splicing events
                        for psi_col in psi_vals_columns:
                            if not np.isnan(row[psi_col]):
                                quality_col = psi_col + "-Q"
                                complexity_score = (
                                    row[quality_col].split("@")[0].split(",")[-1]
                                )
                                assert complexity_score in ["S", "C1", "C2", "C3", "na"]

                                if complexity_score == "S":
                                    # reference intron upstream of the alternative exon
                                    # add/remove 1 to make sure the coordinates are within the intron
                                    if strand == ".":
                                        intron_start = (
                                            int(
                                                row["CO_C1"].split(":")[1].split("-")[1]
                                            )
                                            + 1
                                        )  # the end of the upstream exon
                                        intron_end = extraction_start - 1
                                    else:
                                        intron_start = extraction_end + 1
                                        intron_end = (
                                            int(
                                                row["CO_C1"].split(":")[1].split("-")[0]
                                            )
                                            - 1
                                        )  # the end of the upstream exon
                                    if intron_start < intron_end:
                                        introns_around_splicing_events["EVENT"].append(
                                            row["EVENT"]
                                        )
                                        introns_around_splicing_events[
                                            "EVENT_TYPE"
                                        ].append(event_type)
                                        introns_around_splicing_events["SAMPLE"].append(
                                            psi_col
                                        )
                                        introns_around_splicing_events[
                                            "LOCATION"
                                        ].append("upstream")
                                        introns_around_splicing_events["COORD"].append(
                                            f"{row['CHR']}:{intron_start}-{intron_end}"
                                        )
                                        introns_around_splicing_events["STRAND"].append(
                                            strand
                                        )

                                    # reference intron downstream of the alternative exon
                                    # add/remove 1 to make sure the coordinates are within the intron
                                    if strand == ".":
                                        intron_start = extraction_end + 1
                                        intron_end = (
                                            int(
                                                row["CO_C2"].split(":")[1].split("-")[0]
                                            )
                                            - 1
                                        )  # the start of the downstream exon
                                    else:
                                        intron_start = (
                                            int(
                                                row["CO_C2"].split(":")[1].split("-")[1]
                                            )
                                            + 1
                                        )  # the start of the downstream exon
                                        intron_end = extraction_start - 1
                                    if intron_start < intron_end:
                                        introns_around_splicing_events["EVENT"].append(
                                            row["EVENT"]
                                        )
                                        introns_around_splicing_events[
                                            "EVENT_TYPE"
                                        ].append(event_type)
                                        introns_around_splicing_events["SAMPLE"].append(
                                            psi_col
                                        )
                                        introns_around_splicing_events[
                                            "LOCATION"
                                        ].append("downstream")
                                        introns_around_splicing_events["COORD"].append(
                                            f"{row['CHR']}:{intron_start}-{intron_end}"
                                        )
                                        introns_around_splicing_events["STRAND"].append(
                                            strand
                                        )

                elif event_type == "INT":
                    strand = row["FullCO"].split(":")[-1]
                    strand = "." if strand == "+" else "-"
                    event_info["STRAND"].append(strand)
                    event_info["EVENT_EXTRACTION_COORD"].append(
                        row["COORD"]
                    )  # the coordinates of the intron are the same as the coordinates of the event

                elif event_type == "ALTD":
                    Aexon, C2acceptor = row["FullCO"].split(":")[1].split(",")
                    C2acceptor = [int(i) for i in C2acceptor.split("+")]
                    Aexon_start, Aexon_end = Aexon.split("-")

                    this_Aexon_start, this_Aexon_end = (
                        row["COORD"].split(":")[-1].split("-")
                    )
                    this_Aexon_start = int(this_Aexon_start)
                    this_Aexon_end = int(this_Aexon_end)
                    assert (
                        this_Aexon_start < this_Aexon_end
                    ), f"Invalid coordinates: {this_Aexon_start}-{this_Aexon_end}"
                    event_info["EVENT_EXTRACTION_COORD"].append(
                        f"{row['CHR']}:{this_Aexon_start}-{this_Aexon_end}"
                    )

                    strand = None
                    if this_Aexon_start < C2acceptor[0]:
                        event_info["STRAND"].append(".")
                        strand = "."
                    else:
                        event_info["STRAND"].append("-")
                        strand = "-"

                    # sometimes the fixed coordinates of the alternative exon are not available
                    # in this case, use the coordinates from the event
                    if len(Aexon_start.strip()) == 0:
                        Aexon_start = this_Aexon_start
                        assert (
                            strand == "."
                        ), "Unfixed coordinates are unavailable for the alternative exon"
                    else:
                        Aexon_start = [int(i) for i in Aexon_start.split("+")]
                    if len(Aexon_end.strip()) == 0:
                        Aexon_end = this_Aexon_end
                        assert (
                            strand == "-"
                        ), "Unfixed coordinates are unavailable for the alternative exon"
                    else:
                        Aexon_end = [int(i) for i in Aexon_end.split("+")]

                    if not pd.isna(
                        row["CO_C2"]
                    ):  # only add the introns if the reference coordinates are available
                        for psi_col in psi_vals_columns:
                            if not np.isnan(row[psi_col]):
                                quality_col = psi_col + "-Q"
                                complexity_score = (
                                    row[quality_col].split("@")[0].split(",")[-1]
                                )
                                assert complexity_score in ["S", "C1", "C2", "C3", "na"]

                                if complexity_score == "S":
                                    # reference intron downstream of the alternative exon
                                    # add/remove 1 to make sure the coordinates are within the intron
                                    if strand == ".":
                                        intron_start = max(Aexon_end) + 1
                                        intron_end = (
                                            int(
                                                row["CO_C2"].split(":")[1].split("-")[0]
                                            )
                                            - 1
                                        )
                                    else:
                                        intron_start = (
                                            int(
                                                row["CO_C2"].split(":")[1].split("-")[1]
                                            )
                                            + 1
                                        )
                                        intron_end = min(Aexon_start) - 1
                                    if intron_start < intron_end:
                                        introns_around_splicing_events["EVENT"].append(
                                            row["EVENT"]
                                        )
                                        introns_around_splicing_events[
                                            "EVENT_TYPE"
                                        ].append(event_type)
                                        introns_around_splicing_events["SAMPLE"].append(
                                            psi_col
                                        )
                                        introns_around_splicing_events[
                                            "LOCATION"
                                        ].append("downstream")
                                        introns_around_splicing_events["COORD"].append(
                                            f"{row['CHR']}:{intron_start}-{intron_end}"
                                        )
                                        introns_around_splicing_events["STRAND"].append(
                                            strand
                                        )

                elif event_type == "ALTA":
                    C1donor, Aexon = row["FullCO"].split(":")[1].split(",")
                    C1donor = [int(i) for i in C1donor.split("+")]
                    Aexon_start, Aexon_end = Aexon.split("-")

                    this_Aexon_start, this_Aexon_end = (
                        row["COORD"].split(":")[-1].split("-")
                    )
                    this_Aexon_start = int(this_Aexon_start)
                    this_Aexon_end = int(this_Aexon_end)
                    assert (
                        this_Aexon_start < this_Aexon_end
                    ), f"Invalid coordinates: {this_Aexon_start}-{this_Aexon_end}"
                    event_info["EVENT_EXTRACTION_COORD"].append(
                        f"{row['CHR']}:{this_Aexon_start}-{this_Aexon_end}"
                    )

                    strand = None
                    if C1donor[0] < this_Aexon_start:
                        event_info["STRAND"].append(".")
                        strand = "."
                    else:
                        event_info["STRAND"].append("-")
                        strand = "-"

                    # sometimes the fixed coordinates of the alternative exon are not available
                    # in this case, use the coordinates from the event
                    if len(Aexon_start.strip()) == 0:
                        Aexon_start = this_Aexon_start
                        assert (
                            strand == "-"
                        ), "Unfixed coordinates are unavailable for the alternative exon"
                    else:
                        Aexon_start = [int(i) for i in Aexon_start.split("+")]
                    if len(Aexon_end.strip()) == 0:
                        Aexon_end = this_Aexon_end
                        assert (
                            strand == "."
                        ), "Unfixed coordinates are unavailable for the alternative exon"
                    else:
                        Aexon_end = [int(i) for i in Aexon_end.split("+")]

                    if not pd.isna(
                        row["CO_C1"]
                    ):  # only add the introns if the reference coordinates are available
                        for psi_col in psi_vals_columns:
                            if not np.isnan(row[psi_col]):
                                quality_col = psi_col + "-Q"
                                complexity_score = (
                                    row[quality_col].split("@")[0].split(",")[-1]
                                )
                                assert complexity_score in ["S", "C1", "C2", "C3", "na"]

                                if complexity_score == "S":
                                    # intron upstream of the alternative exon
                                    # add/remove 1 to make sure the coordinates are within the intron
                                    if strand == ".":
                                        intron_start = (
                                            int(
                                                row["CO_C1"].split(":")[1].split("-")[1]
                                            )
                                            + 1
                                        )
                                        intron_end = min(Aexon_start) - 1
                                    else:
                                        intron_start = max(Aexon_end) + 1
                                        intron_end = (
                                            int(
                                                row["CO_C1"].split(":")[1].split("-")[0]
                                            )
                                            - 1
                                        )
                                    if intron_start < intron_end:
                                        introns_around_splicing_events["EVENT"].append(
                                            row["EVENT"]
                                        )
                                        introns_around_splicing_events[
                                            "EVENT_TYPE"
                                        ].append(event_type)
                                        introns_around_splicing_events["SAMPLE"].append(
                                            psi_col
                                        )
                                        introns_around_splicing_events[
                                            "LOCATION"
                                        ].append("upstream")
                                        introns_around_splicing_events["COORD"].append(
                                            f"{row['CHR']}:{intron_start}-{intron_end}"
                                        )
                                        introns_around_splicing_events["STRAND"].append(
                                            strand
                                        )

            flattened_inclusion_levels_full = pd.DataFrame(
                flattened_inclusion_levels_full
            )
            event_info = pd.DataFrame(event_info)
            introns_around_splicing_events = pd.DataFrame(
                introns_around_splicing_events
            )

            flattened_inclusion_levels_full.to_csv(
                os.path.join(
                    self.cache_dir, "flattened_inclusion_levels_full_filtered.csv"
                ),
                index=False,
            )
            event_info.to_csv(
                os.path.join(self.cache_dir, "event_info_filtered.csv"), index=False
            )
            introns_around_splicing_events.to_csv(
                os.path.join(self.cache_dir, "intron_around_splicing_events.csv"),
                index=False,
            )

            print("Total number of PSI values:", len(flattened_inclusion_levels_full))
            print("Total number of events:", len(event_info))
            print("Total number of introns:", len(introns_around_splicing_events))

            print("Number of PSI values of each event type:")
            print(flattened_inclusion_levels_full["EVENT_TYPE"].value_counts())

            print("Number of events of each event type:")
            print(event_info["EVENT_TYPE"].value_counts())

            print("Number of introns of each event type:")
            print(introns_around_splicing_events["EVENT_TYPE"].value_counts())

            print("Flattened data cached")

        if not os.path.exists(
            os.path.join(self.cache_dir, "normalized_gene_expression.csv")
        ):
            # from the gene counts data, calculate the normalized gene expression values - TPM and RPKM
            print(
                "Calculating normalized gene expression values from gene counts (TPM and RPKM)"
            )

            gene_counts = pd.read_csv(
                os.path.join(self.cache_dir, "gene_counts_filtered.csv")
            )

            genome_annotation = genomepy.Annotation(
                name="GRCh38.p14", genomes_dir=os.path.join(self.cache_dir, "genomes")
            )

            # calculate the length of each gene
            # returns a pandas Series with gene IDs as the index and gene lengths as the values
            gene_lengths: pd.Series = genome_annotation.lengths(attribute="gene_id")
            # remove the version number from the gene IDs
            gene_lengths.index = gene_lengths.index.str.split(".").str[0]
            assert gene_lengths.index.is_unique, "Gene IDs are not unique"

            # convert the gene lengths to a DataFrame
            gene_lengths = gene_lengths.to_frame(name="length")
            gene_lengths["gene_id"] = gene_lengths.index
            gene_lengths = gene_lengths.reset_index(drop=True)

            missing_genes = gene_counts[
                ~gene_counts["gene_id"].isin(gene_lengths["gene_id"])
            ]
            if not missing_genes.empty:
                print(
                    f"Warning: {len(missing_genes)} genes are missing from gene length data, removing them"
                )

            normalized_gene_expression = gene_counts.merge(
                gene_lengths, on="gene_id", how="inner", validate="1:1"
            )
            for sample in gene_counts.columns[2:]:
                # calculate the TPM values
                normalized_gene_expression[sample + "_TPM"] = (
                    normalized_gene_expression[sample]
                    / normalized_gene_expression["length"]
                )  # reads per base pair
                rpk_sum = normalized_gene_expression[sample + "_TPM"].sum()
                normalized_gene_expression[sample + "_TPM"] = (
                    normalized_gene_expression[sample + "_TPM"] / rpk_sum
                ) * 1e6  # normalize to TPM

                # calculate log2(TPM + 1) values
                normalized_gene_expression[sample + "_log2TPM"] = np.log2(
                    normalized_gene_expression[sample + "_TPM"] + 1
                )

                # calculate the RPKM values
                normalized_gene_expression[
                    sample + "_RPKM"
                ] = normalized_gene_expression[sample] / (
                    normalized_gene_expression["length"] / 1e3
                )  # reads per kilobase pair
                normalized_gene_expression[sample + "_RPKM"] = (
                    normalized_gene_expression[sample + "_RPKM"] * 1e6
                )  # normalize to RPKM

                # calculate log2(RPKM + 1) values
                normalized_gene_expression[sample + "_log2RPKM"] = np.log2(
                    normalized_gene_expression[sample + "_RPKM"] + 1
                )

            normalized_gene_expression.to_csv(
                os.path.join(self.cache_dir, "normalized_gene_expression.csv"),
                index=False,
            )

        if not os.path.exists(
            os.path.join(self.cache_dir, "splicing_factor_expression_levels.csv")
        ):
            # from the normalized gene expression data, extract the expression levels of the splicing factors in each sample
            # the splicing factor gene IDs are the same as the sample names
            print("Extracting splicing factor expression levels")

            normalized_gene_expression = pd.read_csv(
                os.path.join(self.cache_dir, "normalized_gene_expression.csv")
            )

            all_gene_ids = normalized_gene_expression["gene_id"].values
            splicing_factor_gene_ids = [
                i for i in normalized_gene_expression.columns if i in all_gene_ids
            ]
            assert (len(splicing_factor_gene_ids) * 5) == len(
                normalized_gene_expression.columns
            ) - 3, "Could not find all splicing factor gene IDs in the normalized gene expression data"

            splicing_factor_expression_levels = normalized_gene_expression.loc[
                normalized_gene_expression["gene_id"].isin(splicing_factor_gene_ids)
            ]
            splicing_factor_expression_levels.to_csv(
                os.path.join(self.cache_dir, "splicing_factor_expression_levels.csv"),
                index=False,
            )
            print(
                "Splicing factor expression levels extracted - dataframe shape:",
                splicing_factor_expression_levels.shape,
            )

    def setup(self, stage: str = None):
        print("Loading filtered and flattened data from cache")
        self.normalized_gene_expression = pd.read_csv(
            os.path.join(self.cache_dir, "normalized_gene_expression.csv")
        )
        self.flattened_inclusion_levels_full = pd.read_csv(
            os.path.join(self.cache_dir, "flattened_inclusion_levels_full_filtered.csv")
        )
        self.event_info = pd.read_csv(
            os.path.join(self.cache_dir, "event_info_filtered.csv")
        )
        self.introns_around_splicing_events = pd.read_csv(
            os.path.join(self.cache_dir, "intron_around_splicing_events.csv")
        )
        self.splicing_factor_expression_levels = pd.read_csv(
            os.path.join(self.cache_dir, "splicing_factor_expression_levels.csv")
        )

        print("Total number of PSI values:", len(self.flattened_inclusion_levels_full))
        print("Total number of events:", len(self.event_info))
        print("Total number of introns:", len(self.introns_around_splicing_events))

        print("Number of PSI values of each event type:")
        print(self.flattened_inclusion_levels_full["EVENT_TYPE"].value_counts())

        print("Number of events of each event type:")
        print(self.event_info["EVENT_TYPE"].value_counts())

        print("Number of introns of each event type:")
        print(self.introns_around_splicing_events["EVENT_TYPE"].value_counts())

        # load the genome
        self.genome = genomepy.Genome(
            "GRCh38.p14", genomes_dir=os.path.join(self.cache_dir, "genomes")
        )  # only need "GRCh38.p14" since the data is from human cell lines

        # create datasets for training, validation, and testing
        # train dataset
        self.train_event_info = self.event_info[
            self.event_info["CHR"].isin(self.train_chromosomes)
        ]
        self.train_flattened_inclusion_levels_full = (
            self.flattened_inclusion_levels_full[
                self.flattened_inclusion_levels_full["EVENT"].isin(
                    self.train_event_info["EVENT"]
                )
            ]
        )
        self.train_introns_around_splicing_events = self.introns_around_splicing_events[
            self.introns_around_splicing_events["EVENT"].isin(
                self.train_event_info["EVENT"]
            )
        ]

        print("Train dataset:")
        print(
            "Number of PSI values: {} ({}%)".format(
                len(self.train_flattened_inclusion_levels_full),
                100
                * len(self.train_flattened_inclusion_levels_full)
                / len(self.flattened_inclusion_levels_full),
            )
        )
        print(
            "Number of events: {} ({}%)".format(
                len(self.train_event_info),
                100 * len(self.train_event_info) / len(self.event_info),
            )
        )
        print(
            "Number of introns: {} ({}%)".format(
                len(self.train_introns_around_splicing_events),
                100
                * len(self.train_introns_around_splicing_events)
                / len(self.introns_around_splicing_events),
            )
        )

        print("Number of PSI values of each event type:")
        full_value_counts = self.flattened_inclusion_levels_full[
            "EVENT_TYPE"
        ].value_counts()
        train_value_counts = self.train_flattened_inclusion_levels_full[
            "EVENT_TYPE"
        ].value_counts()
        for event_type in self.train_flattened_inclusion_levels_full[
            "EVENT_TYPE"
        ].unique():
            print(
                f"{event_type}: {train_value_counts[event_type]} ({train_value_counts[event_type] / full_value_counts[event_type] * 100:.2f}%)"
            )

        print("Number of events of each event type:")
        full_value_counts = self.event_info["EVENT_TYPE"].value_counts()
        train_value_counts = self.train_event_info["EVENT_TYPE"].value_counts()
        for event_type in self.train_event_info["EVENT_TYPE"].unique():
            print(
                f"{event_type}: {train_value_counts[event_type]} ({train_value_counts[event_type] / full_value_counts[event_type] * 100:.2f}%)"
            )

        print("Number of introns of each event type:")
        full_value_counts = self.introns_around_splicing_events[
            "EVENT_TYPE"
        ].value_counts()
        train_value_counts = self.train_introns_around_splicing_events[
            "EVENT_TYPE"
        ].value_counts()
        for event_type in self.train_introns_around_splicing_events[
            "EVENT_TYPE"
        ].unique():
            print(
                f"{event_type}: {train_value_counts[event_type]} ({train_value_counts[event_type] / full_value_counts[event_type] * 100:.2f}%)"
            )

        # val dataset
        self.val_event_info = self.event_info[
            self.event_info["CHR"].isin(self.val_chromosomes)
        ]
        self.val_flattened_inclusion_levels_full = self.flattened_inclusion_levels_full[
            self.flattened_inclusion_levels_full["EVENT"].isin(
                self.val_event_info["EVENT"]
            )
        ]
        self.val_introns_around_splicing_events = self.introns_around_splicing_events[
            self.introns_around_splicing_events["EVENT"].isin(
                self.val_event_info["EVENT"]
            )
        ]

        print("Val dataset:")
        print(
            "Number of PSI values: {} ({}%)".format(
                len(self.val_flattened_inclusion_levels_full),
                100
                * len(self.val_flattened_inclusion_levels_full)
                / len(self.flattened_inclusion_levels_full),
            )
        )
        print(
            "Number of events: {} ({}%)".format(
                len(self.val_event_info),
                100 * len(self.val_event_info) / len(self.event_info),
            )
        )
        print(
            "Number of introns: {} ({}%)".format(
                len(self.val_introns_around_splicing_events),
                100
                * len(self.val_introns_around_splicing_events)
                / len(self.introns_around_splicing_events),
            )
        )

        print("Number of PSI values of each event type:")
        full_value_counts = self.flattened_inclusion_levels_full[
            "EVENT_TYPE"
        ].value_counts()
        val_value_counts = self.val_flattened_inclusion_levels_full[
            "EVENT_TYPE"
        ].value_counts()
        for event_type in self.val_flattened_inclusion_levels_full[
            "EVENT_TYPE"
        ].unique():
            print(
                f"{event_type}: {val_value_counts[event_type]} ({val_value_counts[event_type] / full_value_counts[event_type] * 100:.2f}%)"
            )

        print("Number of events of each event type:")
        full_value_counts = self.event_info["EVENT_TYPE"].value_counts()
        val_value_counts = self.val_event_info["EVENT_TYPE"].value_counts()
        for event_type in self.val_event_info["EVENT_TYPE"].unique():
            print(
                f"{event_type}: {val_value_counts[event_type]} ({val_value_counts[event_type] / full_value_counts[event_type] * 100:.2f}%)"
            )

        print("Number of introns of each event type:")
        full_value_counts = self.introns_around_splicing_events[
            "EVENT_TYPE"
        ].value_counts()
        val_value_counts = self.val_introns_around_splicing_events[
            "EVENT_TYPE"
        ].value_counts()
        for event_type in self.val_introns_around_splicing_events[
            "EVENT_TYPE"
        ].unique():
            print(
                f"{event_type}: {val_value_counts[event_type]} ({val_value_counts[event_type] / full_value_counts[event_type] * 100:.2f}%)"
            )

        # test dataset
        self.test_event_info = self.event_info[
            self.event_info["CHR"].isin(self.test_chromosomes)
        ]
        self.test_flattened_inclusion_levels_full = (
            self.flattened_inclusion_levels_full[
                self.flattened_inclusion_levels_full["EVENT"].isin(
                    self.test_event_info["EVENT"]
                )
            ]
        )
        self.test_introns_around_splicing_events = self.introns_around_splicing_events[
            self.introns_around_splicing_events["EVENT"].isin(
                self.test_event_info["EVENT"]
            )
        ]

        print("Test dataset:")
        print(
            "Number of PSI values: {} ({}%)".format(
                len(self.test_flattened_inclusion_levels_full),
                100
                * len(self.test_flattened_inclusion_levels_full)
                / len(self.flattened_inclusion_levels_full),
            )
        )
        print(
            "Number of events: {} ({}%)".format(
                len(self.test_event_info),
                100 * len(self.test_event_info) / len(self.event_info),
            )
        )
        print(
            "Number of introns: {} ({}%)".format(
                len(self.test_introns_around_splicing_events),
                100
                * len(self.test_introns_around_splicing_events)
                / len(self.introns_around_splicing_events),
            )
        )

        print("Number of PSI values of each event type:")
        full_value_counts = self.flattened_inclusion_levels_full[
            "EVENT_TYPE"
        ].value_counts()
        test_value_counts = self.test_flattened_inclusion_levels_full[
            "EVENT_TYPE"
        ].value_counts()
        for event_type in self.test_flattened_inclusion_levels_full[
            "EVENT_TYPE"
        ].unique():
            print(
                f"{event_type}: {test_value_counts[event_type]} ({test_value_counts[event_type] / full_value_counts[event_type] * 100:.2f}%)"
            )

        print("Number of events of each event type:")
        full_value_counts = self.event_info["EVENT_TYPE"].value_counts()
        test_value_counts = self.test_event_info["EVENT_TYPE"].value_counts()
        for event_type in self.test_event_info["EVENT_TYPE"].unique():
            print(
                f"{event_type}: {test_value_counts[event_type]} ({test_value_counts[event_type] / full_value_counts[event_type] * 100:.2f}%)"
            )

        print("Number of introns of each event type:")
        full_value_counts = self.introns_around_splicing_events[
            "EVENT_TYPE"
        ].value_counts()
        test_value_counts = self.test_introns_around_splicing_events[
            "EVENT_TYPE"
        ].value_counts()
        for event_type in self.test_introns_around_splicing_events[
            "EVENT_TYPE"
        ].unique():
            print(
                f"{event_type}: {test_value_counts[event_type]} ({test_value_counts[event_type] / full_value_counts[event_type] * 100:.2f}%)"
            )

        # create datasets
        self.train_dataset = KnockdownDataset(self, split="train")
        self.val_dataset = KnockdownDataset(self, split="val")
        self.test_dataset = KnockdownDataset(self, split="test")

    def __init__(self, config: dict | str):
        super().__init__()
        if isinstance(config, str):
            with open(config, "r") as f:
                self.config = json.load(f)
        else:
            self.config = config

        # seed for reproducibility
        self.seed = self.config["train_config"]["seed"]
        L.seed_everything(self.seed)

        self.input_size = self.config["train_config"]["input_size"]

        # default config chromosome split so that train-val-test split is 70-10-20 roughly amoung filtered splicing events
        # train proportion = 70.61341911926058%
        # val proportion = 9.178465157513145%
        # test proportion = 20.20811572322628%
        # split was computed using the utils.chromosome_split function
        self.train_chromosomes = self.config["train_config"]["train_chromosomes"]
        self.test_chromosomes = self.config["train_config"]["test_chromosomes"]
        self.val_chromosomes = self.config["train_config"]["val_chromosomes"]

        # cache directory
        self.cache_dir = self.config["data_config"]["cache_dir"]
        if not os.path.exists(self.cache_dir):
            os.makedirs(self.cache_dir, exist_ok=True)

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.config["train_config"]["batch_size"],
            shuffle=True,
            pin_memory=True,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.config["train_config"]["batch_size"],
            shuffle=False,
            pin_memory=True,
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_dataset,
            batch_size=self.config["train_config"]["batch_size"],
            shuffle=False,
            pin_memory=True,
        )
