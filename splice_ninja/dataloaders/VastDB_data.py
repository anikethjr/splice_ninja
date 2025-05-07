# Processes data published on VastDB

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


def worker_init_fn(worker_id):
    worker_info = torch.utils.data.get_worker_info()
    worker_info.dataset.genome = genomepy.Genome(
        worker_info.dataset.genome_name, genomes_dir=worker_info.dataset.genomes_dir
    )


# Datast for the combined data
class VastDBDataset(Dataset):
    def __init__(self, data_module, split="train"):
        self.data_module = data_module
        self.split = split
        self.input_size = self.data_module.config["train_config"]["input_size"]
        self.genome_name = "GRCh38.p14"
        self.genomes_dir = os.path.join(self.data_module.cache_dir, "genomes")
        self.genome = None  # Will be initialized per worker
        self.use_shifts_during_training = self.data_module.use_shifts_during_training
        self.shift_max = self.data_module.shift_max

        if self.split == "train":
            self.data = self.data_module.train_data

        elif self.split == "val":
            self.data = self.data_module.val_data

        elif self.split == "test":
            self.data = self.data_module.test_data

    def __len__(self):
        return len(self.data)

    def get_psi_val(self, idx):
        # get the PSI value for the idx-th row
        row = self.data.iloc[idx]
        event_id = row["EVENT"]
        event_type = row["EVENT_TYPE"]
        sample = row["SAMPLE"]
        psi_val = row["PSI"]
        example_type = "train" if self.split == "train" else row["example_type"]

        # get expression of host gene
        has_gene_exp_values = row["HAS_GENE_EXP_VALUES"]
        gene_exp = row["expression"]

        # get splicing factor expression values
        splicing_factor_exp_values = self.data_module.splicing_factor_expression_levels[
            sample + "_log2TPM_rel_norm"
        ].values

        # get the event information for sequence construction
        gene_id = row["GENE_ID"]
        chrom = row["CHR"][3:]  # remove "chr" prefix
        strand = row["STRAND"]

        # construct sequences
        spliced_in_event_segments = row["SPLICED_IN_EVENT_SEGMENTS"].split(
            ","
        )  # comma-separated list of exon coordinates in the format chr:start-end
        spliced_out_event_segments = row["SPLICED_OUT_EVENT_SEGMENTS"].split(
            ","
        )  # comma-separated list of exon coordinates in the format chr:start-end
        full_event_coord = row["FULL_EVENT_COORD"]  # chr:start-end format
        full_event_length = row["FULL_EVENT_LENGTH"]

        extraction_start = int(full_event_coord.split(":")[-1].split("-")[0])
        extraction_end = int(full_event_coord.split(":")[-1].split("-")[1])

        if (extraction_end - extraction_start + 1) < self.input_size:
            # need to pad the sequence with background sequence
            background_sequence_length = self.input_size - (
                extraction_end - extraction_start + 1
            )
            extraction_start = max(
                1, extraction_start - background_sequence_length // 2
            )
            background_sequence_length_still_needed = self.input_size - (
                extraction_end - extraction_start + 1
            )
            extraction_end = min(
                self.genome.sizes[chrom],
                extraction_end + background_sequence_length_still_needed,
            )
            assert (
                extraction_end - extraction_start + 1
            ) == self.input_size, f"Sequence length is {(extraction_end - extraction_start + 1)} but expected length is {self.input_size}, idx: {idx}"

            # get the sequence
            sequence = self.genome.get_seq(
                chrom, extraction_start, extraction_end
            ).seq.upper()

            # compute the spliced-in and spliced-out masks. mask is 1 for an exon, -1 for an intron, and 0 for background sequence
            # first, we need to compute the indices of the exonic and intronic regions
            spliced_in_exonic_sequences_inds = []
            spliced_in_intronic_sequences_inds = []
            for i, segment in enumerate(spliced_in_event_segments):
                start, end = segment.split(":")[-1].split("-")
                start, end = int(start), int(end)

                relative_start = start - extraction_start
                relative_end = end - extraction_start

                # # check if sequence matches the event sequence - uncomment only during debugging to avoid performance hit
                # exon_seq = self.genome.get_seq(chrom, start, end).seq.upper()
                # assert (
                #     exon_seq == sequence[relative_start : relative_end + 1]
                # ), f"Exon sequence does not match the event sequence: {exon_seq} vs {sequence[relative_start:relative_end + 1]}, idx: {idx}"

                spliced_in_exonic_sequences_inds.append((relative_start, relative_end))
                if i < len(spliced_in_event_segments) - 1:
                    next_segment_start = int(
                        spliced_in_event_segments[i + 1].split(":")[-1].split("-")[0]
                    )
                    relative_next_segment_start = next_segment_start - extraction_start
                    if (relative_end + 1) <= (
                        relative_next_segment_start - 1
                    ):  # check if there is an intronic region between the exons
                        spliced_in_intronic_sequences_inds.append(
                            (relative_end + 1, relative_next_segment_start - 1)
                        )

            spliced_out_exonic_sequences_inds = []
            spliced_out_intronic_sequences_inds = []
            for i, segment in enumerate(spliced_out_event_segments):
                start, end = segment.split(":")[-1].split("-")
                start, end = int(start), int(end)

                relative_start = start - extraction_start
                relative_end = end - extraction_start

                # # check if sequence matches the event sequence - uncomment only during debugging to avoid performance hit
                # exon_seq = self.genome.get_seq(chrom, start, end).seq.upper()
                # assert (
                #     exon_seq == sequence[relative_start : relative_end + 1]
                # ), f"Exon sequence does not match the event sequence: {exon_seq} vs {sequence[relative_start:relative_end + 1]}, idx: {idx}"

                spliced_out_exonic_sequences_inds.append((relative_start, relative_end))
                if i < len(spliced_out_event_segments) - 1:
                    next_segment_start = int(
                        spliced_out_event_segments[i + 1].split(":")[-1].split("-")[0]
                    )
                    relative_next_segment_start = next_segment_start - extraction_start
                    if (relative_end + 1) <= (
                        relative_next_segment_start - 1
                    ):  # check if there is an intronic region between the exons
                        spliced_out_intronic_sequences_inds.append(
                            (relative_end + 1, relative_next_segment_start - 1)
                        )

            # now we can construct the masks
            spliced_in_mask = np.zeros(self.input_size)
            for start, end in spliced_in_exonic_sequences_inds:
                spliced_in_mask[start : end + 1] = 1
            for start, end in spliced_in_intronic_sequences_inds:
                spliced_in_mask[start : end + 1] = -1

            spliced_out_mask = np.zeros(self.input_size)
            for start, end in spliced_out_exonic_sequences_inds:
                spliced_out_mask[start : end + 1] = 1
            for start, end in spliced_out_intronic_sequences_inds:
                spliced_out_mask[start : end + 1] = -1

        else:
            # need to crop the sequence, this is done by removing the middle portions of introns and keeping the ends (segment_length_to_crop_to_if_needed//2 bp on each side) when possible (i.e. when event type not intron retention - IR)
            # if more of the sequence needs to be removed due to long exons, we remove the middle portions of the upstream and downstream exons and keep the ends (while maintaining the length as a multiple of 3 and having segment_length_to_crop_to_if_needed//2 bp on each side)
            # if we still need to remove more sequence, we remove the middle portions of the alternative exons and keep the ends (while maintaining the length as a multiple of 3 and having segment_length_to_crop_to_if_needed//2 bp on each side)

            # get current segments
            cur_seq_len = extraction_end - extraction_start + 1
            length_to_remove = cur_seq_len - self.input_size
            cur_genome_segments = []
            for i, segment in enumerate(
                spliced_in_event_segments
            ):  # spliced in segments will always be a superset of spliced out segments
                start, end = segment.split(":")[-1].split("-")
                start, end = int(start), int(end)
                assert start <= end, f"Start is greater than end, idx: {idx}"

                if i == 1:  # this is the alternative exon
                    cur_genome_segments.append((start, end, "alt_exon"))
                else:
                    cur_genome_segments.append((start, end, "exon"))

                if i < len(spliced_in_event_segments) - 1:
                    next_segment_start = int(
                        spliced_in_event_segments[i + 1].split(":")[-1].split("-")[0]
                    )
                    if (end + 1) <= (
                        next_segment_start - 1
                    ):  # check if there is an intronic region between the exons
                        cur_genome_segments.append(
                            (end + 1, next_segment_start - 1, "intron")
                        )
            assert cur_seq_len == sum(
                [end - start + 1 for start, end, _ in cur_genome_segments]
            ), f"Sequence length is {cur_seq_len} but sum of segment lengths is {sum([end - start + 1 for start, end, _ in cur_genome_segments])}, idx: {idx}"

            # start off by removing the middle portions of the introns
            if (
                event_type != "IR"
            ):  # having as much of the intron as possible is important for IR events
                updated_genome_segments = []
                for i, segment in enumerate(cur_genome_segments):
                    if segment[2] == "intron":
                        start, end = segment[0], segment[1]
                        segment_len = end - start + 1
                        if (
                            segment_len
                            <= self.data_module.segment_length_to_crop_to_if_needed
                        ):
                            updated_genome_segments.append(segment)
                        else:
                            # remove the middle portion of the intron
                            updated_genome_segments.append(
                                (start, start + 500 - 1, "intron")
                            )
                            updated_genome_segments.append(
                                (end - 500 + 1, end, "intron")
                            )
                    else:
                        updated_genome_segments.append(segment)
                cur_genome_segments = updated_genome_segments
                cur_seq_len = sum(
                    [end - start + 1 for start, end, _ in cur_genome_segments]
                )

            # remove the middle portions of the exons if needed
            if cur_seq_len > self.input_size:
                updated_genome_segments = []
                for i, segment in enumerate(cur_genome_segments):
                    if segment[2] == "exon":
                        start, end = segment[0], segment[1]
                        segment_len = end - start + 1
                        if (
                            segment_len
                            <= self.data_module.segment_length_to_crop_to_if_needed
                        ):
                            updated_genome_segments.append(segment)
                        else:
                            # remove the middle portion of the exon
                            updated_genome_segments.append(
                                (start, start + 500 - 1, "exon")
                            )
                            updated_genome_segments.append((end - 500 + 1, end, "exon"))
                    else:
                        updated_genome_segments.append(segment)
                cur_genome_segments = updated_genome_segments
                cur_seq_len = sum(
                    [end - start + 1 for start, end, _ in cur_genome_segments]
                )

            # remove the middle portions of the alternative exons if needed
            if cur_seq_len > self.input_size:
                updated_genome_segments = []
                for i, segment in enumerate(cur_genome_segments):
                    if segment[2] == "alt_exon":
                        start, end = segment[0], segment[1]
                        segment_len = end - start + 1
                        if (
                            segment_len
                            <= self.data_module.segment_length_to_crop_to_if_needed
                        ):
                            updated_genome_segments.append(segment)
                        else:
                            # remove the middle portion of the alternative exon
                            updated_genome_segments.append(
                                (start, start + 500 - 1, "alt_exon")
                            )
                            updated_genome_segments.append(
                                (end - 500 + 1, end, "alt_exon")
                            )
                    else:
                        updated_genome_segments.append(segment)
                cur_genome_segments = updated_genome_segments
                cur_seq_len = sum(
                    [end - start + 1 for start, end, _ in cur_genome_segments]
                )

            assert (
                cur_seq_len <= self.input_size
            ), f"Sequence length is {cur_seq_len} which is still greater than the input size {self.input_size}, idx: {idx}"

            # extract the segments
            sequence = ""
            spliced_in_mask = np.zeros(cur_seq_len)
            spliced_out_mask = np.zeros(cur_seq_len)
            for start, end, segment_type in cur_genome_segments:
                segment_seq = self.genome.get_seq(chrom, start, end).seq.upper()
                assert len(segment_seq) == (
                    end - start + 1
                ), f"Segment length mismatch, idx: {idx}, length: {len(segment_seq)}, expected length: {(end - start + 1)}, segment: {segment_seq}, start: {start}, end: {end}, chrom: {chrom}"
                if segment_type == "exon":
                    spliced_in_mask[
                        len(sequence) : len(sequence) + len(segment_seq)
                    ] = 1
                    spliced_out_mask[
                        len(sequence) : len(sequence) + len(segment_seq)
                    ] = 1
                elif segment_type == "alt_exon":
                    spliced_in_mask[
                        len(sequence) : len(sequence) + len(segment_seq)
                    ] = 1
                    spliced_out_mask[
                        len(sequence) : len(sequence) + len(segment_seq)
                    ] = -1
                elif segment_type == "intron":
                    spliced_in_mask[
                        len(sequence) : len(sequence) + len(segment_seq)
                    ] = -1
                    spliced_out_mask[
                        len(sequence) : len(sequence) + len(segment_seq)
                    ] = -1
                sequence = sequence + segment_seq
            assert (
                len(sequence) == cur_seq_len
            ), f"Sequence length is {len(sequence)} but expected length is {cur_seq_len}, idx: {idx}, genome segments: {cur_genome_segments}"

            # pad the sequence with background sequence
            background_sequence_length = self.input_size - len(sequence)
            sequence_start = max(1, extraction_start - background_sequence_length // 2)
            padding_before_length = (
                extraction_start - sequence_start
            )  # length of the padding before the sequence
            if padding_before_length > 0:
                sequence = (
                    self.genome.get_seq(
                        chrom, sequence_start, extraction_start - 1
                    ).seq.upper()
                    + sequence
                )
                spliced_in_mask = np.concatenate(
                    [np.zeros(padding_before_length), spliced_in_mask]
                )
                spliced_out_mask = np.concatenate(
                    [np.zeros(padding_before_length), spliced_out_mask]
                )
            padding_after_length = self.input_size - len(
                sequence
            )  # length of the padding after the sequence
            if padding_after_length > 0:
                sequence = (
                    sequence
                    + self.genome.get_seq(
                        chrom, extraction_end + 1, extraction_end + padding_after_length
                    ).seq.upper()
                )
                spliced_in_mask = np.concatenate(
                    [spliced_in_mask, np.zeros(padding_after_length)]
                )
                spliced_out_mask = np.concatenate(
                    [spliced_out_mask, np.zeros(padding_after_length)]
                )
            assert (
                len(sequence) == self.input_size
            ), f"Sequence length is {len(sequence)} but expected length is {self.input_size}, idx: {idx}"
            assert (
                len(spliced_in_mask) == self.input_size
            ), f"Spliced-in mask length is {len(spliced_in_mask)} but expected length is {self.input_size}, idx: {idx}"
            assert (
                len(spliced_out_mask) == self.input_size
            ), f"Spliced-out mask length is {len(spliced_out_mask)} but expected length is {self.input_size}, idx: {idx}"

        # convert sequence to indices
        sequence_inds = np.array(
            [self.data_module.base_to_ind[base] for base in sequence]
        )

        # account for genes on the negative strand
        if strand == "-":
            rc_sequence_inds = sequence_inds.copy()  # first copy the array
            A_ind = self.data_module.base_to_ind["A"]
            C_ind = self.data_module.base_to_ind["C"]
            G_ind = self.data_module.base_to_ind["G"]
            T_ind = self.data_module.base_to_ind["T"]
            rc_sequence_inds[sequence_inds == A_ind] = T_ind  # replace A with T
            rc_sequence_inds[sequence_inds == C_ind] = G_ind  # replace C with G
            rc_sequence_inds[sequence_inds == G_ind] = C_ind  # replace G with C
            rc_sequence_inds[sequence_inds == T_ind] = A_ind  # replace T with A
            sequence_inds = rc_sequence_inds[::-1].copy()  # reverse

            spliced_in_mask = spliced_in_mask[::-1].copy()  # reverse
            spliced_out_mask = spliced_out_mask[::-1].copy()  # reverse

        if self.split == "train" and self.use_shifts_during_training:
            shift = np.random.randint(-self.shift_max, self.shift_max + 1)
            # shift sequence/masks and pad with zeros
            if shift > 0:
                sequence_inds = np.concatenate(
                    [np.zeros(shift), sequence_inds[:-shift]]
                )
                spliced_in_mask = np.concatenate(
                    [np.zeros(shift), spliced_in_mask[:-shift]]
                )
                spliced_out_mask = np.concatenate(
                    [np.zeros(shift), spliced_out_mask[:-shift]]
                )
            elif shift < 0:
                sequence_inds = np.concatenate(
                    [sequence_inds[-shift:], np.zeros(-shift)]
                )
                spliced_in_mask = np.concatenate(
                    [spliced_in_mask[-shift:], np.zeros(-shift)]
                )
                spliced_out_mask = np.concatenate(
                    [spliced_out_mask[-shift:], np.zeros(-shift)]
                )
            assert (
                len(sequence_inds) == self.input_size
            ), f"Sequence length is {len(sequence_inds)} but expected length is {self.input_size}, idx: {idx}"
            assert (
                len(spliced_in_mask) == self.input_size
            ), f"Spliced-in mask length is {len(spliced_in_mask)} but expected length is {self.input_size}, idx: {idx}"
            assert (
                len(spliced_out_mask) == self.input_size
            ), f"Spliced-out mask length is {len(spliced_out_mask)} but expected length is {self.input_size}, idx: {idx}"

        return {
            "sequence": sequence_inds.astype(np.int8),
            "spliced_in_mask": spliced_in_mask.astype(np.int8),
            "spliced_out_mask": spliced_out_mask.astype(np.int8),
            "psi_val": (psi_val / 100.0).astype(np.float32),
            "gene_exp": gene_exp if has_gene_exp_values else -1.0,
            "splicing_factor_exp_values": splicing_factor_exp_values.astype(np.float32),
            "event_type": self.data_module.event_type_to_ind[event_type],
            "event_id": self.data_module.event_id_to_ind[event_id],
            "sample": self.data_module.sample_to_ind[sample],
            "event_num_samples_observed": row["NUM_SAMPLES_OBSERVED"],
            "event_mean_psi": (row["MEAN_PSI"] / 100.0).astype(np.float32),
            "event_std_psi": (row["STD_PSI"] / 100.0).astype(
                np.float32
            ),  # if you convert PSI from 0-100 to 0-1, the variance will be divided by 100^2 and the standard deviation will be divided by 100
            "event_min_psi": (row["MIN_PSI"] / 100.0).astype(np.float32),
            "event_max_psi": (row["MAX_PSI"] / 100.0).astype(np.float32),
            "event_controls_avg_psi": np.nan,
            "event_num_controls": np.nan,
            "example_type": self.data_module.example_type_to_ind[example_type],
        }

    def __getitem__(self, idx):
        return self.get_psi_val(idx)


# DataModule for the combined data
class VastDBData(LightningDataModule):
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

        # now download and process the VastDB data
        if (
            not os.path.exists(
                os.path.join(
                    self.cache_dir,
                    "VastDB",
                    "hg38",
                    "inclusion_levels_full_filtered.parquet",
                )
            )
        ) or (
            not os.path.exists(
                os.path.join(
                    self.cache_dir,
                    "VastDB",
                    "hg38",
                    "normalized_gene_expression.parquet",
                )
            )
        ):
            print("Downloading and processing VastDB PSI and expression data")

            # download PSI and expression data from VastDB
            os.makedirs(os.path.join(self.cache_dir, "VastDB", "hg38"), exist_ok=True)
            if not os.path.exists(
                os.path.join(
                    self.cache_dir, "VastDB", "hg38", "EXPRESSION_TABLE-hg38.tab.gz"
                )
            ):
                print("Downloading expression data from VastDB")
                url = (
                    "https://vastdb.crg.eu/downloads/hg38/EXPRESSION_TABLE-hg38.tab.gz"
                )
                urllib.request.urlretrieve(
                    url,
                    os.path.join(
                        self.cache_dir, "VastDB", "hg38", "EXPRESSION_TABLE-hg38.tab.gz"
                    ),
                )
                print("Expression data downloaded")
            if not os.path.exists(
                os.path.join(self.cache_dir, "VastDB", "hg38", "PSI_TABLE-hg38.tab.gz")
            ):
                print("Downloading PSI data from VastDB")
                url = "https://vastdb.crg.eu/downloads/hg38/PSI_TABLE-hg38.tab.gz"
                urllib.request.urlretrieve(
                    url,
                    os.path.join(
                        self.cache_dir, "VastDB", "hg38", "PSI_TABLE-hg38.tab.gz"
                    ),
                )
                print("PSI data downloaded")

            # load gene expression data
            gene_counts = pd.read_csv(
                os.path.join(
                    self.cache_dir, "VastDB", "hg38", "EXPRESSION_TABLE-hg38.tab.gz"
                ),
                sep="\t",
            )
            gene_counts = gene_counts.rename(
                columns={"ID": "gene_id"}
            )  # rename the ID column to gene_id

            # load PSI data
            inclusion_levels_full = pd.read_csv(
                os.path.join(self.cache_dir, "VastDB", "hg38", "PSI_TABLE-hg38.tab.gz"),
                sep="\t",
            )

            psi_vals_columns = [
                i for i in inclusion_levels_full.columns[6:] if (not i.endswith("-Q"))
            ]
            quality_columns = [
                i for i in inclusion_levels_full.columns[6:] if i.endswith("-Q")
            ]

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

            # drop all quality columns, we don't need them anymore
            inclusion_levels_full = inclusion_levels_full.drop(
                columns=[i for i in inclusion_levels_full.columns if i.endswith("-Q")]
            )

            # print number of events of each type
            print(
                f"Final number of events of each type:\n{inclusion_levels_full['COMPLEX'].value_counts()}"
            )

            # cache the filtered PSI data
            inclusion_levels_full.to_parquet(
                os.path.join(
                    self.cache_dir,
                    "VastDB",
                    "hg38",
                    "inclusion_levels_full_filtered.parquet",
                ),
                index=False,
            )

            # from the gene counts data, calculate the normalized gene expression values - TPM and RPKM
            print(
                "Calculating normalized gene expression values from gene counts (TPM and RPKM)"
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
                if not sample.endswith("-Counts"):
                    continue

                sample = sample[: -len("-Counts")]

                # there are some NaN values in the gene counts data, we need to fill them with 0
                normalized_gene_expression[
                    sample + "-Counts"
                ] = normalized_gene_expression[sample + "-Counts"].fillna(0)

                # calculate the TPM values
                normalized_gene_expression[sample + "_TPM"] = (
                    normalized_gene_expression[sample + "-Counts"]
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
                ] = normalized_gene_expression[sample + "-Counts"] / (
                    normalized_gene_expression["length"] / 1e3
                )  # reads per kilobase pair
                normalized_gene_expression[sample + "_RPKM"] = (
                    normalized_gene_expression[sample + "_RPKM"] * 1e6
                )  # normalize to RPKM

                # calculate log2(RPKM + 1) values
                normalized_gene_expression[sample + "_log2RPKM"] = np.log2(
                    normalized_gene_expression[sample + "_RPKM"] + 1
                )

                normalized_gene_expression = normalized_gene_expression.copy()

            # drop all count, TPM, and RPKM columns, we don't support using them downstream
            drop_columns = []
            for col in gene_counts.columns[2:]:
                if col.endswith("-Counts"):
                    drop_columns.append(col)
                    drop_columns.append(col[:-7] + "-cRPKM")
                    drop_columns.append(col[:-7] + "_TPM")
                    drop_columns.append(col[:-7] + "_RPKM")
            normalized_gene_expression = normalized_gene_expression.drop(
                columns=drop_columns
            )

            # now just verify that every sample has both expression and PSI values
            psi_vals_columns = [i for i in inclusion_levels_full.columns[6:]]
            for sample in psi_vals_columns:
                assert (sample + "_log2TPM" in normalized_gene_expression.columns) and (
                    sample + "_log2RPKM" in normalized_gene_expression.columns
                ), f"Sample {sample} does not have expression values"
            expression_columns = [
                i
                for i in normalized_gene_expression.columns[2:]
                if i.endswith("_log2TPM") or i.endswith("_log2RPKM")
            ]
            for col in expression_columns:
                if col.endswith("_log2TPM"):
                    sample = col[: -len("_log2TPM")]
                elif col.endswith("_log2RPKM"):
                    sample = col[: -len("_log2RPKM")]

                assert (
                    sample in psi_vals_columns
                ), f"Sample {sample} does not have PSI values"

            # minor renaming for consistency
            normalized_gene_expression = normalized_gene_expression.rename(
                columns={"NAME": "alias"}
            )

            normalized_gene_expression.to_parquet(
                os.path.join(
                    self.cache_dir,
                    "VastDB",
                    "hg38",
                    "normalized_gene_expression.parquet",
                ),
                index=False,
            )

        if not os.path.exists(
            os.path.join(
                self.cache_dir,
                "VastDB",
                "hg38",
                "splicing_factor_expression_levels.parquet",
            )
        ):
            # from the normalized gene expression data, extract the expression levels of the splicing factors in each sample
            # we will use the same splicing factors as in the knockdown data
            print("Extracting splicing factor expression levels")

            normalized_gene_expression = pd.read_parquet(
                os.path.join(
                    self.cache_dir,
                    "VastDB",
                    "hg38",
                    "normalized_gene_expression.parquet",
                )
            )
            splicing_factor_expression_levels_knockdown = pd.read_parquet(
                os.path.join(
                    self.cache_dir, "splicing_factor_expression_levels.parquet"
                )
            )

            splicing_factor_expression_levels = (
                splicing_factor_expression_levels_knockdown[["gene_id"]].merge(
                    normalized_gene_expression,
                    on="gene_id",
                    how="left",
                )
            )

            # count number of splicing factors for which there is no expression data in VastDB
            for i in range(len(splicing_factor_expression_levels)):
                row = splicing_factor_expression_levels.iloc[i]
                expr_values = row[splicing_factor_expression_levels.columns[2:]]
                if expr_values.isna().any():
                    num_nans = expr_values.isna().sum()
                    print(
                        f"Warning: Splicing factor {row['gene_id']} has {num_nans} NaNs in the expression data from VastDB, will be set to 0 log2TPM and 0 log2RPKM"
                    )

            # set the expression values to 0 for splicing factors with no expression data in VastDB
            splicing_factor_expression_levels[
                splicing_factor_expression_levels.columns[2:]
            ] = splicing_factor_expression_levels[
                splicing_factor_expression_levels.columns[2:]
            ].fillna(
                0
            )

            # compute normalized splicing factor expression levels so that they sum to 1 across all splicing factors in each sample
            for sample in splicing_factor_expression_levels.columns:
                if sample.endswith("_log2TPM"):
                    sample = sample[: -len("_log2TPM")]
                else:
                    continue

                splicing_factor_expression_levels[sample + "_log2TPM_rel_norm"] = (
                    splicing_factor_expression_levels[sample + "_log2TPM"]
                    / splicing_factor_expression_levels[sample + "_log2TPM"].sum()
                )

                splicing_factor_expression_levels = (
                    splicing_factor_expression_levels.copy()
                )

            splicing_factor_expression_levels.to_parquet(
                os.path.join(
                    self.cache_dir,
                    "VastDB",
                    "hg38",
                    "splicing_factor_expression_levels.parquet",
                ),
                index=False,
            )
            print(
                "Splicing factor expression levels extracted - dataframe shape:",
                splicing_factor_expression_levels.shape,
            )

        if not os.path.exists(
            os.path.join(
                self.cache_dir,
                "flattened_inclusion_levels_full_filtered_VastDB.parquet",
            )
        ) or not os.path.exists(
            os.path.join(self.cache_dir, "event_info_filtered_VastDB.parquet")
        ):
            # flatten the filtered data - make a row for each sample for each event and remove NaN values
            # this makes it to create a dataset for training
            # to avoid duplicating event information, we only keep the columns for the event and the PSI value
            # another file is created with the event information like event type, coordinates, etc.

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
            ).drop_duplicates("Gene_name")
            event_id_to_gene_id_from_vastdb = pd.read_csv(
                os.path.join(
                    self.cache_dir, "VastDB", "hg38", "EVENTID_to_GENEID-hg38.tab.gz"
                ),
                sep="\t",
            )

            # load the filtered data
            normalized_gene_expression_VastDB = pd.read_parquet(
                os.path.join(
                    self.cache_dir,
                    "VastDB",
                    "hg38",
                    "normalized_gene_expression.parquet",
                )
            )
            inclusion_levels_full_VastDB = pd.read_parquet(
                os.path.join(
                    self.cache_dir,
                    "VastDB",
                    "hg38",
                    "inclusion_levels_full_filtered.parquet",
                )
            )

            # get event info for VastDB data
            print("Getting event info for VastDB data")

            # define samples
            vastdb_psi_vals_columns = [
                i for i in inclusion_levels_full_VastDB.columns[6:]
            ]

            # create a column for the chromosome
            inclusion_levels_full_VastDB["CHR"] = inclusion_levels_full_VastDB[
                "COORD"
            ].apply(lambda x: x.split(":")[0])
            # join inclusion levels data with event info to get reference exon coordinates
            inclusion_levels_full_VastDB = inclusion_levels_full_VastDB.merge(
                event_info_from_vastdb[["EVENT", "REF_CO", "CO_C1", "CO_A", "CO_C2"]],
                on="EVENT",
                how="left",
            ).reset_index(drop=True)
            event_found_mask = inclusion_levels_full_VastDB["REF_CO"].notnull()
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
                inclusion_levels_full_VastDB.loc[~event_found_mask, "EVENT"].head()
            )  # print some event IDs not found in VastDB
            print("Per event type:")
            for event_type in inclusion_levels_full_VastDB["COMPLEX"].unique():
                event_found_mask = inclusion_levels_full_VastDB.loc[
                    inclusion_levels_full_VastDB["COMPLEX"] == event_type, "CO_C1"
                ].notnull()
                print(
                    f"{event_type}: {event_found_mask.sum()} ({100 * event_found_mask.mean():.2f}%), not found: {(~event_found_mask).sum()} ({100 * (~event_found_mask).mean():.2f}%)"
                )
                print(
                    "Some event IDs not found in VastDB for {}: {}".format(
                        event_type,
                        inclusion_levels_full_VastDB.loc[
                            (inclusion_levels_full_VastDB["COMPLEX"] == event_type)
                            & (~event_found_mask),
                            "EVENT",
                        ].head(),
                    )
                )
            # drop events not found in VastDB
            inclusion_levels_full_VastDB = inclusion_levels_full_VastDB.loc[
                inclusion_levels_full_VastDB["REF_CO"].notnull()
            ].reset_index(drop=True)

            # join inclusion levels data with gene info to get gene ID
            inclusion_levels_full_VastDB = inclusion_levels_full_VastDB.merge(
                gene_info_from_vastdb[["GeneID", "Gene_name"]],
                left_on="GENE",
                right_on="Gene_name",
                how="left",
            )
            inclusion_levels_full_VastDB = inclusion_levels_full_VastDB.drop(
                columns=["Gene_name"]
            )
            inclusion_levels_full_VastDB = inclusion_levels_full_VastDB.rename(
                columns={"GeneID": "GENE_ID"}
            )

            temp = inclusion_levels_full_VastDB[["GENE", "GENE_ID"]].drop_duplicates()
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

            # create flattened data for VastDB
            print("Creating flattened data for VastDB")
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

            event_info[
                "NUM_SAMPLES_OBSERVED"
            ] = []  # number of samples in which the event was observed

            event_info["MEAN_PSI"] = []  # mean of the PSI values across samples
            event_info[
                "STD_PSI"
            ] = []  # standard deviation of the PSI values across samples
            event_info["MIN_PSI"] = []  # minimum of the PSI values across samples
            event_info["MAX_PSI"] = []  # maximum of the PSI values across samples

            event_info["FullCO"] = []  # full coordinates of the event
            event_info["COMPLEX"] = []  # fine-grained event type
            event_info["CHR"] = []  # chromosome
            event_info["STRAND"] = []  # strand
            # all segments below are in the 5' to 3' direction and are separated by a comma. they are extracted from VastDB
            event_info[
                "FULL_EVENT_LENGTH"
            ] = (
                []
            )  # length of the genomic segment from the start to the end of the event
            event_info[
                "FULL_EVENT_COORD"
            ] = (
                []
            )  # genomic coordinates of the genomic segment from the start to the end of the event
            event_info[
                "SPLICED_IN_EVENT_SEGMENTS"
            ] = (
                []
            )  # genomic segments for the spliced in the event: for exon skipping, this is the upstream exon, the alternate exon, and the downstream exon; for intron retention, this is the upstream exon, the intron being retained, and the downstream exon; for alternative splice site choice, this is the alternate exon and the upstream/downstream exon
            event_info[
                "SPLICED_OUT_EVENT_SEGMENTS"
            ] = (
                []
            )  # genomic segments for the spliced out the event: for exon skipping, this is the upstream exon and the downstream exon; for intron retention, this is the upstream exon and the downstream exon; for alternative splice site choice, this is the reference exon and the upstream/downstream exon

            all_gene_ids_with_expression_values = set(
                normalized_gene_expression_VastDB["gene_id"]
            )  # all gene IDs with expression values

            # iterate over each row in the data and populate the flattened data and event information
            for i, row in tqdm(
                inclusion_levels_full_VastDB.iterrows(),
                total=inclusion_levels_full_VastDB.shape[0],
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

                num_samples_vastdb = 0
                for psi_col in vastdb_psi_vals_columns:
                    if not np.isnan(row[psi_col]):
                        flattened_inclusion_levels_full["EVENT"].append(row["EVENT"])
                        flattened_inclusion_levels_full["EVENT_TYPE"].append(event_type)
                        flattened_inclusion_levels_full["SAMPLE"].append(psi_col)
                        flattened_inclusion_levels_full["PSI"].append(row[psi_col])
                        num_samples_vastdb += 1

                event_info["GENE"].append(row["GENE"])
                event_info["GENE_ID"].append(row["GENE_ID"])
                event_info["HAS_GENE_EXP_VALUES"].append(
                    (not pd.isna(row["GENE_ID"]))
                    and row["GENE_ID"] in all_gene_ids_with_expression_values
                )

                # stats about the PSI values
                psi_vals = row[vastdb_psi_vals_columns].values.reshape(-1).astype(float)
                psi_vals = psi_vals[~np.isnan(psi_vals)]
                event_info["NUM_SAMPLES_OBSERVED"].append(len(psi_vals))
                assert len(psi_vals) == num_samples_vastdb
                event_info["MEAN_PSI"].append(psi_vals.mean())
                event_info["STD_PSI"].append(psi_vals.std())
                event_info["MIN_PSI"].append(psi_vals.min())
                event_info["MAX_PSI"].append(psi_vals.max())

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
                # REF_CO column contains reference exon coordinates + the strand
                strand = row["REF_CO"].split(":")[-1]
                strand = (
                    "." if strand == "+" else "-"
                )  # convert "+" to "." for consistency
                event_info["STRAND"].append(strand)

                # extract the genomic segments for the spliced in and spliced out events
                spliced_in_event_segments = []
                spliced_out_event_segments = []
                if event_type == "EX":
                    if strand == ".":
                        upstream_exon_coordinates = row["CO_C1"]
                        alternate_exon_coordinates = row["CO_A"]
                        downstream_exon_coordinates = row["CO_C2"]

                        spliced_in_event_segments = [
                            upstream_exon_coordinates,
                            alternate_exon_coordinates,
                            downstream_exon_coordinates,
                        ]
                        spliced_out_event_segments = [
                            upstream_exon_coordinates,
                            downstream_exon_coordinates,
                        ]
                    else:
                        upstream_exon_coordinates = row["CO_C1"]
                        alternate_exon_coordinates = row["CO_A"]
                        downstream_exon_coordinates = row["CO_C2"]

                        # the order of the exons is reversed for the "-" strand to maintain the 5' to 3' direction
                        spliced_in_event_segments = [
                            downstream_exon_coordinates,
                            alternate_exon_coordinates,
                            upstream_exon_coordinates,
                        ]
                        spliced_out_event_segments = [
                            downstream_exon_coordinates,
                            upstream_exon_coordinates,
                        ]

                elif event_type == "INT":
                    if strand == ".":
                        upstream_exon_coordinates = row["CO_C1"]
                        retained_intron_coordinates = row["CO_A"]
                        downstream_exon_coordinates = row["CO_C2"]

                        spliced_in_event_segments = [
                            upstream_exon_coordinates,
                            retained_intron_coordinates,
                            downstream_exon_coordinates,
                        ]
                        spliced_out_event_segments = [
                            upstream_exon_coordinates,
                            downstream_exon_coordinates,
                        ]
                    else:
                        upstream_exon_coordinates = row["CO_C1"]
                        retained_intron_coordinates = row["CO_A"]
                        downstream_exon_coordinates = row["CO_C2"]

                        # the order of the exons is reversed for the "-" strand to maintain the 5' to 3' direction
                        spliced_in_event_segments = [
                            downstream_exon_coordinates,
                            retained_intron_coordinates,
                            upstream_exon_coordinates,
                        ]
                        spliced_out_event_segments = [
                            downstream_exon_coordinates,
                            upstream_exon_coordinates,
                        ]

                elif event_type == "ALTD":
                    # for ALTD events, the CO_C1 is the reference exon, CO_A is the extra segment added in the event, and CO_C2 is the downstream exon
                    if strand == ".":
                        reference_exon_coordinates = row["CO_C1"]
                        alternate_exon_added_segment_coordinates = row["CO_A"]
                        downstream_exon_coordinates = row["CO_C2"]

                        if pd.isna(alternate_exon_added_segment_coordinates):
                            assert (
                                "-1/" in row["EVENT"]
                            ), "Only the first event in an ALTD event should have a missing CO_A"

                            spliced_in_event_segments = [
                                reference_exon_coordinates,
                                downstream_exon_coordinates,
                            ]
                            spliced_out_event_segments = [
                                reference_exon_coordinates,
                                downstream_exon_coordinates,
                            ]
                        else:
                            spliced_in_event_segments = [
                                reference_exon_coordinates,
                                alternate_exon_added_segment_coordinates,
                                downstream_exon_coordinates,
                            ]
                            spliced_out_event_segments = [
                                reference_exon_coordinates,
                                downstream_exon_coordinates,
                            ]
                    else:
                        reference_exon_coordinates = row["CO_C1"]
                        alternate_exon_added_segment_coordinates = row["CO_A"]
                        downstream_exon_coordinates = row["CO_C2"]

                        if pd.isna(alternate_exon_added_segment_coordinates):
                            assert (
                                "-1/" in row["EVENT"]
                            ), "Only the first event in an ALTD event should have a missing CO_A"

                            spliced_in_event_segments = [
                                downstream_exon_coordinates,
                                reference_exon_coordinates,
                            ]
                            spliced_out_event_segments = [
                                downstream_exon_coordinates,
                                reference_exon_coordinates,
                            ]
                        else:
                            spliced_in_event_segments = [
                                downstream_exon_coordinates,
                                alternate_exon_added_segment_coordinates,
                                reference_exon_coordinates,
                            ]
                            spliced_out_event_segments = [
                                downstream_exon_coordinates,
                                reference_exon_coordinates,
                            ]

                elif event_type == "ALTA":
                    # for ALTA events, the CO_C1 is the upstream exon, CO_A is the extra segment added in the event, and CO_C2 is the reference exon
                    if strand == ".":
                        upstream_exon_coordinates = row["CO_C1"]
                        alternate_exon_added_segment_coordinates = row["CO_A"]
                        reference_exon_coordinates = row["CO_C2"]

                        if pd.isna(alternate_exon_added_segment_coordinates):
                            assert (
                                "-1/" in row["EVENT"]
                            ), "Only the first event in an ALTA event should have a missing CO_A"

                            spliced_in_event_segments = [
                                upstream_exon_coordinates,
                                reference_exon_coordinates,
                            ]
                            spliced_out_event_segments = [
                                upstream_exon_coordinates,
                                reference_exon_coordinates,
                            ]
                        else:
                            spliced_in_event_segments = [
                                upstream_exon_coordinates,
                                alternate_exon_added_segment_coordinates,
                                reference_exon_coordinates,
                            ]
                            spliced_out_event_segments = [
                                upstream_exon_coordinates,
                                reference_exon_coordinates,
                            ]
                    else:
                        upstream_exon_coordinates = row["CO_C1"]
                        alternate_exon_added_segment_coordinates = row["CO_A"]
                        reference_exon_coordinates = row["CO_C2"]

                        if pd.isna(alternate_exon_added_segment_coordinates):
                            assert (
                                "-1/" in row["EVENT"]
                            ), "Only the first event in an ALTA event should have a missing CO_A"

                            spliced_in_event_segments = [
                                reference_exon_coordinates,
                                upstream_exon_coordinates,
                            ]
                            spliced_out_event_segments = [
                                reference_exon_coordinates,
                                upstream_exon_coordinates,
                            ]
                        else:
                            spliced_in_event_segments = [
                                reference_exon_coordinates,
                                alternate_exon_added_segment_coordinates,
                                upstream_exon_coordinates,
                            ]
                            spliced_out_event_segments = [
                                reference_exon_coordinates,
                                upstream_exon_coordinates,
                            ]

                event_info["SPLICED_IN_EVENT_SEGMENTS"].append(
                    ",".join(spliced_in_event_segments)
                )
                spliced_in_event_segments_min_coord = min(
                    [
                        int(x.strip().split(":")[1].split("-")[0])
                        for x in spliced_in_event_segments
                    ]
                )
                spliced_in_event_segments_max_coord = max(
                    [
                        int(x.strip().split(":")[1].split("-")[1])
                        for x in spliced_in_event_segments
                    ]
                )
                full_event_length = (
                    spliced_in_event_segments_max_coord
                    - spliced_in_event_segments_min_coord
                    + 1
                )
                full_event_coord = f"{row['CHR']}:{spliced_in_event_segments_min_coord}-{spliced_in_event_segments_max_coord}"
                event_info["FULL_EVENT_LENGTH"].append(full_event_length)
                event_info["FULL_EVENT_COORD"].append(full_event_coord)

                event_info["SPLICED_OUT_EVENT_SEGMENTS"].append(
                    ",".join(spliced_out_event_segments)
                )

            flattened_inclusion_levels_full = pd.DataFrame(
                flattened_inclusion_levels_full
            ).drop_duplicates()
            event_info = pd.DataFrame(event_info).drop_duplicates("EVENT", keep="first")

            flattened_inclusion_levels_full.to_parquet(
                os.path.join(
                    self.cache_dir,
                    "flattened_inclusion_levels_full_filtered_VastDB.parquet",
                ),
                index=False,
            )
            event_info.to_parquet(
                os.path.join(
                    self.cache_dir,
                    "event_info_filtered_VastDB.parquet",
                ),
                index=False,
            )

            print("Total number of PSI values:", len(flattened_inclusion_levels_full))
            print("Total number of events:", len(event_info))

            print("Number of PSI values of each event type:")
            print(flattened_inclusion_levels_full["EVENT_TYPE"].value_counts())

            print("Number of events of each event type:")
            print(event_info["EVENT_TYPE"].value_counts())

            print("Flattened data cached")

        if "min_samples_for_event_to_be_considered" in self.config["data_config"]:
            min_samples_for_event_to_be_considered = self.config["data_config"][
                "min_samples_for_event_to_be_considered"
            ]
            if not os.path.exists(
                os.path.join(
                    self.cache_dir,
                    f"flattened_inclusion_levels_events_observed_in_min_{min_samples_for_event_to_be_considered}_samples_VastDB.parquet",
                )
            ) or not os.path.exists(
                os.path.join(
                    self.cache_dir,
                    f"event_info_events_observed_in_min_{min_samples_for_event_to_be_considered}_samples_VastDB.parquet",
                )
            ):
                # filter out events that are not observed in the minimum number of samples
                print(
                    f"Filtering out events that are not observed in at least {min_samples_for_event_to_be_considered} samples"
                )
                flattened_inclusion_levels_full = pd.read_parquet(
                    os.path.join(
                        self.cache_dir,
                        "flattened_inclusion_levels_full_filtered_VastDB.parquet",
                    )
                )
                event_info = pd.read_parquet(
                    os.path.join(
                        self.cache_dir,
                        "event_info_filtered_VastDB.parquet",
                    )
                )

                print("Number of events of each type before filtering:")
                print(event_info["EVENT_TYPE"].value_counts())

                print("Number of PSI values of each type before filtering:")
                print(flattened_inclusion_levels_full["EVENT_TYPE"].value_counts())

                event_info = event_info.loc[
                    event_info["NUM_SAMPLES_OBSERVED"]
                    >= min_samples_for_event_to_be_considered
                ].reset_index(drop=True)
                flattened_inclusion_levels_full = flattened_inclusion_levels_full.loc[
                    flattened_inclusion_levels_full["EVENT"].isin(event_info["EVENT"])
                ].reset_index(drop=True)

                print("Number of events of each type after filtering:")
                print(event_info["EVENT_TYPE"].value_counts())

                print("Number of PSI values of each type after filtering:")
                print(flattened_inclusion_levels_full["EVENT_TYPE"].value_counts())

                flattened_inclusion_levels_full.to_parquet(
                    os.path.join(
                        self.cache_dir,
                        f"flattened_inclusion_levels_events_observed_in_min_{min_samples_for_event_to_be_considered}_samples_VastDB.parquet",
                    ),
                    index=False,
                )
                event_info.to_parquet(
                    os.path.join(
                        self.cache_dir,
                        f"event_info_events_observed_in_min_{min_samples_for_event_to_be_considered}_samples_VastDB.parquet",
                    ),
                    index=False,
                )

                print("Filtered data cached")

    def setup(self, stage: str = None):
        print("Loading filtered and flattened data from cache")
        # load the normalized gene expression data and merge
        self.normalized_gene_expression = pd.read_parquet(
            os.path.join(
                self.cache_dir, "VastDB", "hg38", "normalized_gene_expression.parquet"
            )
        )
        assert (
            self.normalized_gene_expression.shape[0]
            == self.normalized_gene_expression["gene_id"].nunique()
        )
        # fill missing values with 0
        self.normalized_gene_expression = self.normalized_gene_expression.fillna(0)

        if "min_samples_for_event_to_be_considered" in self.config["data_config"]:
            min_samples_for_event_to_be_considered = self.config["data_config"][
                "min_samples_for_event_to_be_considered"
            ]
            self.flattened_inclusion_levels_full = pd.read_parquet(
                os.path.join(
                    self.cache_dir,
                    f"flattened_inclusion_levels_events_observed_in_min_{min_samples_for_event_to_be_considered}_samples_VastDB.parquet",
                )
            )
            self.event_info = pd.read_parquet(
                os.path.join(
                    self.cache_dir,
                    f"event_info_events_observed_in_min_{min_samples_for_event_to_be_considered}_samples_VastDB.parquet",
                )
            )

        else:
            self.flattened_inclusion_levels_full = pd.read_parquet(
                os.path.join(
                    self.cache_dir,
                    "flattened_inclusion_levels_full_filtered_VastDB.parquet",
                )
            )
            self.event_info = pd.read_parquet(
                os.path.join(
                    self.cache_dir,
                    "event_info_filtered_VastDB.parquet",
                )
            )

        # filter out events that are not in the event types to model
        if self.event_types_to_model != "ALL":
            original_flattened_inclusion_levels_full_len = len(
                self.flattened_inclusion_levels_full
            )
            self.flattened_inclusion_levels_full = self.flattened_inclusion_levels_full[
                self.flattened_inclusion_levels_full["EVENT_TYPE"].isin(
                    self.event_types_to_model
                )
            ].reset_index(drop=True)
            print(
                f"Filtered flattened inclusion levels data to only include event types {self.event_types_to_model} ({100 * len(self.flattened_inclusion_levels_full) / original_flattened_inclusion_levels_full_len:.2f}% of the original data)"
            )

            original_event_info_len = len(self.event_info)
            self.event_info = self.event_info[
                self.event_info["EVENT_TYPE"].isin(self.event_types_to_model)
            ].reset_index(drop=True)
            print(
                f"Filtered event info data to only include event types {self.event_types_to_model} ({100 * len(self.event_info) / original_event_info_len:.2f}% of the original data)"
            )

        # load the splicing factor expression levels and merge
        self.splicing_factor_expression_levels = pd.read_parquet(
            os.path.join(
                self.cache_dir,
                "VastDB",
                "hg38",
                "splicing_factor_expression_levels.parquet",
            )
        ).drop(columns=["alias", "length"])
        self.num_splicing_factors = self.splicing_factor_expression_levels.shape[0]
        self.has_gene_exp_values = True

        print("Total number of PSI values:", len(self.flattened_inclusion_levels_full))
        print("Total number of events:", len(self.event_info))

        print("Number of PSI values of each event type:")
        print(self.flattened_inclusion_levels_full["EVENT_TYPE"].value_counts())

        print("Number of events of each event type:")
        print(self.event_info["EVENT_TYPE"].value_counts())

        # load the genome
        self.genome = genomepy.Genome(
            "GRCh38.p14", genomes_dir=os.path.join(self.cache_dir, "genomes")
        )  # only need "GRCh38.p14" since the data is from human cell lines

        # create a flattened version of the gene expression data to make it easier to merge with the PSI values
        gene_expression_metric_cols = []
        for sample in self.normalized_gene_expression.columns:
            if sample.endswith(f"_{self.gene_expression_metric}"):
                gene_expression_metric_cols.append(sample)

        self.normalized_gene_expression = self.normalized_gene_expression[
            self.normalized_gene_expression.columns[:2].to_list()
            + gene_expression_metric_cols
        ]
        self.normalized_gene_expression = self.normalized_gene_expression.rename(
            columns={
                i: "_".join(i.split("_")[:-1]) for i in gene_expression_metric_cols
            }
        )  # remove the metric suffix from the column names
        print(
            f"Kept {len(gene_expression_metric_cols)} columns with the gene expression metric '{self.gene_expression_metric}'"
        )
        self.normalized_gene_expression_flattened = (
            self.normalized_gene_expression.melt(
                id_vars=["gene_id"], var_name="sample", value_name="expression"
            )
        )

        # create unified dataframe containing PSI values, event information, and host-gene expression values
        print("Creating unified dataframe")
        # first filter out events without gene expression values if required
        if self.remove_events_without_gene_expression_data:
            original_event_info_len = len(self.event_info)
            self.event_info = self.event_info[
                self.event_info["HAS_GENE_EXP_VALUES"]
            ].reset_index(drop=True)
            print(
                f"Removed {original_event_info_len - len(self.event_info)} events without host-gene expression data ({100 * (original_event_info_len - len(self.event_info)) / original_event_info_len:.2f}%)"
            )
            original_flattened_inclusion_levels_full_len = len(
                self.flattened_inclusion_levels_full
            )
            self.flattened_inclusion_levels_full = self.flattened_inclusion_levels_full[
                self.flattened_inclusion_levels_full["EVENT"].isin(
                    self.event_info["EVENT"]
                )
            ].reset_index(drop=True)
            print(
                f"Removed {original_flattened_inclusion_levels_full_len - len(self.flattened_inclusion_levels_full)} PSI values of events without host-gene expression data ({100 * (original_flattened_inclusion_levels_full_len - len(self.flattened_inclusion_levels_full)) / original_flattened_inclusion_levels_full_len:.2f}%)"
            )
        # add event information to the flattened inclusion levels data
        original_flattened_inclusion_levels_full_len = len(
            self.flattened_inclusion_levels_full
        )
        self.unified_data = self.flattened_inclusion_levels_full.merge(
            self.event_info, on=["EVENT", "EVENT_TYPE"], how="inner"
        )
        assert (
            len(self.unified_data) == original_flattened_inclusion_levels_full_len
        ), f"Number of rows in the unified data is not the same as the flattened inclusion levels data, {len(self.unified_data)} != {original_flattened_inclusion_levels_full_len}"
        print("Merged event information with flattened inclusion levels data")
        # add gene expression values to the unified data
        original_unified_data_len = len(self.unified_data)
        if (
            "do_not_use_gene_expression_data" in self.config["train_config"]
            and self.config["train_config"]["do_not_use_gene_expression_data"]
        ):
            self.unified_data["expression"] = np.nan
        else:
            self.unified_data = self.unified_data.merge(
                self.normalized_gene_expression_flattened,
                left_on=["GENE_ID", "SAMPLE"],
                right_on=["gene_id", "sample"],
                how="left",
            )
            self.unified_data = self.unified_data.drop(columns=["gene_id", "sample"])
            if self.remove_events_without_gene_expression_data:
                assert (
                    self.unified_data["expression"].notnull().all()
                ), "Some events without gene expression data are present in the unified data although they should have been removed"
            print(
                f"Merged gene expression values with unified data ({100 * (original_unified_data_len - len(self.unified_data)) / original_unified_data_len:.2f}% of events without gene expression data)"
            )

        # create datasets for training, validation, and testing
        if self.split_type == "chromosome":
            self.train_data = self.unified_data[
                self.unified_data["CHR"].isin(self.train_chromosomes)
            ].reset_index(drop=True)
            self.val_data = self.unified_data[
                self.unified_data["CHR"].isin(self.val_chromosomes)
            ].reset_index(drop=True)
            self.val_data["example_type"] = "heldout_chromosome"
            self.test_data = self.unified_data[
                self.unified_data["CHR"].isin(self.test_chromosomes)
            ].reset_index(drop=True)
            self.test_data["example_type"] = "heldout_chromosome"

            self.example_types_in_this_split_type = ["train", "heldout_chromosome"]
        elif self.split_type == "sample":
            self.train_data = self.unified_data[
                self.unified_data["SAMPLE"].isin(self.train_samples)
            ].reset_index(drop=True)
            self.val_data = self.unified_data[
                self.unified_data["SAMPLE"].isin(self.val_samples)
            ].reset_index(drop=True)
            self.val_data["example_type"] = "heldout_sample"
            self.test_data = self.unified_data[
                self.unified_data["SAMPLE"].isin(self.test_samples)
            ].reset_index(drop=True)
            self.test_data["example_type"] = "heldout_sample"

            self.example_types_in_this_split_type = ["train", "heldout_sample"]
        elif self.split_type == "chromosome_and_sample":
            self.train_data = self.unified_data[
                self.unified_data["CHR"].isin(self.train_chromosomes)
                & self.unified_data["SAMPLE"].isin(self.train_samples)
            ].reset_index(drop=True)

            # there are four types of val/test examples:
            # 1. examples from the val/test chromosomes and val/test samples
            # 2. examples from the val/test chromosomes and train samples
            # 3. examples from the train chromosomes and val/test samples
            # 4. examples from the val chromosomes and test samples (opposite for test) - these are not used to avoid data leakage
            # we will use the first three types and quantify performance on them separately as well as combined
            self.val_data = []
            t1 = self.unified_data[
                self.unified_data["CHR"].isin(self.val_chromosomes)
                & self.unified_data["SAMPLE"].isin(self.val_samples)
            ].reset_index(drop=True)
            t1["example_type"] = "heldout_chromosome_and_sample"
            self.val_data.append(t1)
            t2 = self.unified_data[
                self.unified_data["CHR"].isin(self.val_chromosomes)
                & self.unified_data["SAMPLE"].isin(self.train_samples)
            ].reset_index(drop=True)
            t2["example_type"] = "heldout_chromosome_and_train_sample"
            self.val_data.append(t2)
            t3 = self.unified_data[
                self.unified_data["CHR"].isin(self.train_chromosomes)
                & self.unified_data["SAMPLE"].isin(self.val_samples)
            ].reset_index(drop=True)
            t3["example_type"] = "heldout_sample_and_train_chromosome"
            self.val_data.append(t3)
            self.val_data = pd.concat(self.val_data, ignore_index=True)

            self.test_data = []
            t1 = self.unified_data[
                self.unified_data["CHR"].isin(self.test_chromosomes)
                & self.unified_data["SAMPLE"].isin(self.test_samples)
            ].reset_index(drop=True)
            t1["example_type"] = "heldout_chromosome_and_sample"
            self.test_data.append(t1)
            t2 = self.unified_data[
                self.unified_data["CHR"].isin(self.test_chromosomes)
                & self.unified_data["SAMPLE"].isin(self.train_samples)
            ].reset_index(drop=True)
            t2["example_type"] = "heldout_chromosome_and_train_sample"
            self.test_data.append(t2)
            t3 = self.unified_data[
                self.unified_data["CHR"].isin(self.train_chromosomes)
                & self.unified_data["SAMPLE"].isin(self.test_samples)
            ].reset_index(drop=True)
            t3["example_type"] = "heldout_sample_and_train_chromosome"
            self.test_data.append(t3)
            self.test_data = pd.concat(self.test_data, ignore_index=True)

            self.example_types_in_this_split_type = [
                "train",
                "heldout_chromosome_and_sample",
                "heldout_chromosome_and_train_sample",
                "heldout_sample_and_train_chromosome",
            ]
        else:
            raise Exception(
                f"Invalid split type specified in config: {self.split_type}"
            )

        # train dataset stats
        print("Train dataset:")
        print(
            "Number of PSI values: {} ({}%)".format(
                len(self.train_data),
                100 * len(self.train_data) / len(self.unified_data),
            )
        )
        print(
            "Number of events: {} ({}%)".format(
                len(self.train_data["EVENT"].unique()),
                100
                * len(self.train_data["EVENT"].unique())
                / len(self.unified_data["EVENT"].unique()),
            )
        )
        print(
            "Number of chromosomes: {} ({}%)".format(
                len(self.train_data["CHR"].unique()),
                100
                * len(self.train_data["CHR"].unique())
                / len(self.unified_data["CHR"].unique()),
            )
        )
        print(
            "Number of samples: {} ({}%)".format(
                len(self.train_data["SAMPLE"].unique()),
                100
                * len(self.train_data["SAMPLE"].unique())
                / len(self.unified_data["SAMPLE"].unique()),
            )
        )

        print("Number of PSI values of each event type:")
        full_value_counts = self.unified_data["EVENT_TYPE"].value_counts()
        train_value_counts = self.train_data["EVENT_TYPE"].value_counts()
        for event_type in self.train_data["EVENT_TYPE"].unique():
            print(
                f"{event_type}: {train_value_counts[event_type]} ({train_value_counts[event_type] / full_value_counts[event_type] * 100:.2f}%)"
            )

        # val dataset stats
        print("Val dataset:")
        print(
            "Number of PSI values: {} ({}%)".format(
                len(self.val_data),
                100 * len(self.val_data) / len(self.unified_data),
            )
        )
        print(
            "Number of events: {} ({}%)".format(
                len(self.val_data["EVENT"].unique()),
                100
                * len(self.val_data["EVENT"].unique())
                / len(self.unified_data["EVENT"].unique()),
            )
        )
        print(
            "Number of chromosomes: {} ({}%)".format(
                len(self.val_data["CHR"].unique()),
                100
                * len(self.val_data["CHR"].unique())
                / len(self.unified_data["CHR"].unique()),
            )
        )
        print(
            "Number of samples: {} ({}%)".format(
                len(self.val_data["SAMPLE"].unique()),
                100
                * len(self.val_data["SAMPLE"].unique())
                / len(self.unified_data["SAMPLE"].unique()),
            )
        )

        for example_type in self.val_data["example_type"].unique():
            print(
                f"Number of PSI values for example type {example_type}: {len(self.val_data[self.val_data['example_type'] == example_type])} ({100 * len(self.val_data[self.val_data['example_type'] == example_type]) / len(self.val_data):.2f}%)"
            )

        print("Number of PSI values of each event type:")
        val_value_counts = self.val_data["EVENT_TYPE"].value_counts()
        for event_type in self.val_data["EVENT_TYPE"].unique():
            print(
                f"{event_type}: {val_value_counts[event_type]} ({val_value_counts[event_type] / full_value_counts[event_type] * 100:.2f}%)"
            )

        # test dataset stats
        print("Test dataset:")
        print(
            "Number of PSI values: {} ({}%)".format(
                len(self.test_data),
                100 * len(self.test_data) / len(self.unified_data),
            )
        )
        print(
            "Number of events: {} ({}%)".format(
                len(self.test_data["EVENT"].unique()),
                100
                * len(self.test_data["EVENT"].unique())
                / len(self.unified_data["EVENT"].unique()),
            )
        )
        print(
            "Number of chromosomes: {} ({}%)".format(
                len(self.test_data["CHR"].unique()),
                100
                * len(self.test_data["CHR"].unique())
                / len(self.unified_data["CHR"].unique()),
            )
        )
        print(
            "Number of samples: {} ({}%)".format(
                len(self.test_data["SAMPLE"].unique()),
                100
                * len(self.test_data["SAMPLE"].unique())
                / len(self.unified_data["SAMPLE"].unique()),
            )
        )

        for example_type in self.test_data["example_type"].unique():
            print(
                f"Number of PSI values for example type {example_type}: {len(self.test_data[self.test_data['example_type'] == example_type])} ({100 * len(self.test_data[self.test_data['example_type'] == example_type]) / len(self.test_data):.2f}%)"
            )

        print("Number of PSI values of each event type:")
        test_value_counts = self.test_data["EVENT_TYPE"].value_counts()
        for event_type in self.test_data["EVENT_TYPE"].unique():
            print(
                f"{event_type}: {test_value_counts[event_type]} ({test_value_counts[event_type] / full_value_counts[event_type] * 100:.2f}%)"
            )

        # get event ID to index mapping
        self.event_id_to_ind = {
            event_id: i for i, event_id in enumerate(self.event_info["EVENT"])
        }

        # get sample ID to index mapping
        all_samples = self.unified_data["SAMPLE"].unique().tolist()
        all_samples.sort()
        all_samples = [sample for sample in all_samples]
        self.sample_to_ind = {sample: i for i, sample in enumerate(all_samples)}
        print("All samples:", all_samples)
        print("Sample ID to index mapping:", self.sample_to_ind)

    def __init__(self, config: dict | str):
        super().__init__()
        if isinstance(config, str):
            with open(config, "r") as f:
                self.config = json.load(f)
        else:
            self.config = config

        # data config
        # cache directory
        self.cache_dir = self.config["data_config"]["cache_dir"]
        if not os.path.exists(self.cache_dir):
            os.makedirs(self.cache_dir, exist_ok=True)

        # training config
        # seed for reproducibility
        self.seed = self.config["train_config"]["seed"]
        L.seed_everything(self.seed)

        self.input_size = self.config["train_config"]["input_size"]
        if "use_shifts_during_training" in self.config["train_config"]:
            self.use_shifts_during_training = self.config["train_config"][
                "use_shifts_during_training"
            ]
            self.shift_max = self.config["train_config"]["shift_max"]
        else:
            self.use_shifts_during_training = False
            self.shift_max = 0
        self.remove_events_without_gene_expression_data = self.config["train_config"][
            "remove_events_without_gene_expression_data"
        ]
        self.gene_expression_metric = self.config["train_config"][
            "gene_expression_metric"
        ]
        assert self.gene_expression_metric in [
            "log2RPKM",
            "log2TPM",
        ], "Invalid gene expression metric specified in config, must be one of 'log2RPKM', 'log2TPM'"
        self.event_types_to_model = self.config["train_config"][
            "event_types_to_model"
        ]  # list of event types to model (comma separated), "ALL" to model all event types
        if self.event_types_to_model != "ALL":
            self.event_types_to_model = self.event_types_to_model.split(",")
            assert all(
                i in ["EX", "INT", "ALTD", "ALTA"] for i in self.event_types_to_model
            ), "Invalid event types specified in config, must be in {'EX', 'INT', 'ALTD', 'ALTA'}"
        self.segment_length_to_crop_to_if_needed = self.config["train_config"][
            "segment_length_to_crop_to_if_needed"
        ]

        self.base_to_ind = {
            "A": 0,
            "C": 1,
            "G": 2,
            "T": 3,
            "N": 4,
        }

        self.event_type_to_ind = {
            "EX": 0,
            "INT": 1,
            "ALTD": 2,
            "ALTA": 3,
        }

        # identify split type to prepare parameters accordingly
        if "split_type" in self.config["train_config"]:
            self.split_type = self.config["train_config"]["split_type"]
        else:
            self.split_type = (
                "chromosome"
                if "train_chromosomes" in self.config["train_config"]
                else "sample"
                if "train_samples" in self.config["train_config"]
                else None
            )
            if self.split_type is None:
                raise Exception(
                    "Either 'train_chromosomes', 'test_chromosomes', and 'val_chromosomes', or 'train_samples', 'test_samples', and 'val_samples' must be specified in the train config if 'split_type' is not specified"
                )
        assert self.split_type in [
            "chromosome",
            "sample",
            "chromosome_and_sample",
        ], "Invalid split type specified in config, must be one of 'chromosome', 'sample', 'chromosome_and_sample'"

        if self.split_type == "chromosome":
            assert (
                "train_chromosomes" in self.config["train_config"]
                and "test_chromosomes" in self.config["train_config"]
                and "val_chromosomes" in self.config["train_config"]
            ), "If split type is 'chromosome', 'train_chromosomes', 'test_chromosomes', and 'val_chromosomes' must be specified in the train config"
        elif self.split_type == "sample":
            assert (
                "train_samples" in self.config["train_config"]
                and "test_samples" in self.config["train_config"]
                and "val_samples" in self.config["train_config"]
            ), "If split type is 'sample', 'train_samples', 'test_samples', and 'val_samples' must be specified in the train config"
        elif self.split_type == "chromosome_and_sample":
            assert (
                "train_chromosomes" in self.config["train_config"]
                and "test_chromosomes" in self.config["train_config"]
                and "val_chromosomes" in self.config["train_config"]
                and "train_samples" in self.config["train_config"]
                and "test_samples" in self.config["train_config"]
                and "val_samples" in self.config["train_config"]
            ), "If split type is 'chromosome_and_sample', 'train_chromosomes', 'test_chromosomes', 'val_chromosomes', 'train_samples', 'test_samples', and 'val_samples' must be specified in the train config"

        if "chromosome" in self.split_type:
            self.train_chromosomes = self.config["train_config"]["train_chromosomes"]
            self.test_chromosomes = self.config["train_config"]["test_chromosomes"]
            self.val_chromosomes = self.config["train_config"]["val_chromosomes"]
        if "sample" in self.split_type:
            self.train_samples = self.config["train_config"]["train_samples"]
            self.test_samples = self.config["train_config"]["test_samples"]
            self.val_samples = self.config["train_config"]["val_samples"]

        self.example_type_to_ind = {
            "train": 0,
            "heldout_chromosome": 1,
            "heldout_sample": 2,
            "heldout_chromosome_and_sample": 3,
            "heldout_chromosome_and_train_sample": 4,
            "heldout_sample_and_train_chromosome": 5,
        }

        # hyperparams that affect when the ranking loss is applied
        if "num_epochs_after_which_to_use_ranking_loss" in self.config["train_config"]:
            self.num_epochs_after_which_to_use_ranking_loss = self.config[
                "train_config"
            ]["num_epochs_after_which_to_use_ranking_loss"]
        else:
            self.num_epochs_after_which_to_use_ranking_loss = 0

        # hyperparams that affect how we define significant events
        if "dPSI_threshold_for_significance" in self.config["train_config"]:
            self.dPSI_threshold_for_significance = self.config["train_config"][
                "dPSI_threshold_for_significance"
            ]
        else:
            self.dPSI_threshold_for_significance = 0.0
        if "upsample_significant_events" in self.config["train_config"]:
            self.upsample_significant_events = self.config["train_config"][
                "upsample_significant_events"
            ]
        else:
            self.upsample_significant_events = False
        if "create_uniform_train_PSI_distribution" in self.config["train_config"]:
            self.create_uniform_train_PSI_distribution = self.config["train_config"][
                "create_uniform_train_PSI_distribution"
            ]
            if self.create_uniform_train_PSI_distribution:
                assert (
                    self.upsample_significant_events is False
                ), "Cannot use both upsampling and uniform distribution at the same time"
                assert (
                    "N_events_per_batch" not in self.config["train_config"]
                ), "Cannot use both N_events_per_batch and uniform distribution at the same time"
        else:
            self.create_uniform_train_PSI_distribution = False

    def train_dataloader(self):
        if self.trainer.current_epoch < self.num_epochs_after_which_to_use_ranking_loss:
            print(
                f"Not using ranking loss in epoch {self.trainer.current_epoch} as it is before the specified epoch {self.num_epochs_after_which_to_use_ranking_loss}, a random sampler will be used"
            )
            return DataLoader(
                VastDBDataset(self, split="train"),
                batch_size=self.config["train_config"]["batch_size"],
                shuffle=True,
                pin_memory=True,
                num_workers=self.config["train_config"]["num_workers"],
                worker_init_fn=worker_init_fn,
            )
        else:
            print(
                f"Current epoch: {self.trainer.current_epoch}, using fully-fledged train dataloader"
            )
            if self.create_uniform_train_PSI_distribution:
                print(
                    f"Creating uniform distribution of PSI values in epoch {self.trainer.current_epoch}"
                )
                dataset = VastDBDataset(self, split="train")
                return DataLoader(
                    dataset,
                    batch_size=self.config["train_config"]["batch_size"],
                    shuffle=None,
                    pin_memory=True,
                    num_workers=self.config["train_config"]["num_workers"],
                    worker_init_fn=worker_init_fn,
                    sampler=UniformPSIDistributionDistributedSampler(
                        self, dataset=dataset
                    ),
                )
            else:
                return DataLoader(
                    VastDBDataset(self, split="train"),
                    batch_size=self.config["train_config"]["batch_size"],
                    shuffle=None
                    if ("N_events_per_batch" in self.config["train_config"])
                    else True,
                    pin_memory=True,
                    num_workers=self.config["train_config"]["num_workers"],
                    worker_init_fn=worker_init_fn,
                    sampler=NEventsPerBatchDistributedSampler(self)
                    if ("N_events_per_batch" in self.config["train_config"])
                    else None,
                )

    def val_dataloader(self):
        return DataLoader(
            VastDBDataset(self, split="val"),
            batch_size=self.config["train_config"]["batch_size"],
            shuffle=False,
            pin_memory=True,
            num_workers=self.config["train_config"]["num_workers"],
            worker_init_fn=worker_init_fn,
        )

    def test_dataloader(self):
        return DataLoader(
            VastDBDataset(self, split="test"),
            batch_size=self.config["train_config"]["batch_size"],
            shuffle=False,
            pin_memory=True,
            num_workers=self.config["train_config"]["num_workers"],
            worker_init_fn=worker_init_fn,
        )
