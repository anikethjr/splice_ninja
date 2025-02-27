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

from splice_ninja.utils import get_ensembl_gene_id_hgnc_with_alias, one_hot_encode_dna

np.random.seed(0)
torch.manual_seed(0)


def worker_init_fn(worker_id):
    worker_info = torch.utils.data.get_worker_info()
    worker_info.dataset.genome = genomepy.Genome(
        worker_info.dataset.genome_name, genomes_dir=worker_info.dataset.genomes_dir
    )


# Datast for the knockdown data
class KnockdownDataset(Dataset):
    def __init__(self, data_module, split="train"):
        self.data_module = data_module
        self.split = split
        self.input_size = self.data_module.config["train_config"]["input_size"]
        self.genome_name = "GRCh38.p14"
        self.genomes_dir = os.path.join(self.data_module.cache_dir, "genomes")
        self.genome = None  # Will be initialized per worker

        if self.split == "train":
            self.chromosomes = self.data_module.train_chromosomes
            self.data = self.data_module.train_data

        elif self.split == "val":
            self.chromosomes = self.data_module.val_chromosomes
            self.data = self.data_module.val_data

        elif self.split == "test":
            self.chromosomes = self.data_module.test_chromosomes
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

        return {
            "sequence": sequence_inds.astype(np.int8),
            "spliced_in_mask": spliced_in_mask.astype(np.float32),
            "spliced_out_mask": spliced_out_mask.astype(np.float32),
            "psi_val": (psi_val / 100.0).astype(np.float32),
            "gene_exp": gene_exp.astype(np.float32)
            if has_gene_exp_values
            else (-1.0).astype(np.float32),
            "splicing_factor_exp_values": splicing_factor_exp_values.astype(np.float32),
            "event_type": self.data_module.event_type_to_ind[event_type],
        }

    def __getitem__(self, idx):
        return self.get_psi_val(idx)


# DataModule for the knockdown data
class KnockdownData(LightningDataModule):
    def prepare_data(self):
        # load/create the filtered splicing data
        if (
            not os.path.exists(
                os.path.join(self.cache_dir, "inclusion_levels_full_filtered.parquet")
            )
            or not os.path.exists(
                os.path.join(self.cache_dir, "gene_counts_filtered.parquet")
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
            inclusion_levels_full.to_parquet(
                os.path.join(self.cache_dir, "inclusion_levels_full_filtered.parquet"),
                index=False,
            )

            # cache the gene counts
            gene_counts.to_parquet(
                os.path.join(self.cache_dir, "gene_counts_filtered.parquet"),
                index=False,
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

        if not os.path.exists(
            os.path.join(self.cache_dir, "normalized_gene_expression.parquet")
        ):
            # from the gene counts data, calculate the normalized gene expression values - TPM and RPKM
            print(
                "Calculating normalized gene expression values from gene counts (TPM and RPKM)"
            )

            gene_counts = pd.read_parquet(
                os.path.join(self.cache_dir, "gene_counts_filtered.parquet")
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

            normalized_gene_expression.to_parquet(
                os.path.join(self.cache_dir, "normalized_gene_expression.parquet"),
                index=False,
            )

        if not os.path.exists(
            os.path.join(self.cache_dir, "splicing_factor_expression_levels.parquet")
        ):
            # from the normalized gene expression data, extract the expression levels of the splicing factors in each sample
            # the splicing factor gene IDs are the same as the sample names
            print("Extracting splicing factor expression levels")

            normalized_gene_expression = pd.read_parquet(
                os.path.join(self.cache_dir, "normalized_gene_expression.parquet")
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

            # compute normalized splicing factor expression levels so that they sum to 1 across all splicing factors in each sample
            for sample in splicing_factor_gene_ids:
                splicing_factor_expression_levels[sample + "_log2TPM_rel_norm"] = (
                    splicing_factor_expression_levels[sample + "_log2TPM"]
                    / splicing_factor_expression_levels[sample + "_log2TPM"].sum()
                )

            splicing_factor_expression_levels.to_parquet(
                os.path.join(
                    self.cache_dir, "splicing_factor_expression_levels.parquet"
                ),
                index=False,
            )
            print(
                "Splicing factor expression levels extracted - dataframe shape:",
                splicing_factor_expression_levels.shape,
            )

        if not os.path.exists(
            os.path.join(
                self.cache_dir, "flattened_inclusion_levels_full_filtered.parquet"
            )
        ) or not os.path.exists(
            os.path.join(self.cache_dir, "event_info_filtered.parquet")
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
            ).drop_duplicates("Gene_name")
            event_id_to_gene_id_from_vastdb = pd.read_csv(
                os.path.join(
                    self.cache_dir, "VastDB", "hg38", "EVENTID_to_GENEID-hg38.tab.gz"
                ),
                sep="\t",
            )

            # load the filtered data
            normalized_gene_expression = pd.read_parquet(
                os.path.join(self.cache_dir, "normalized_gene_expression.parquet")
            )
            inclusion_levels_full = pd.read_parquet(
                os.path.join(self.cache_dir, "inclusion_levels_full_filtered.parquet")
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
                event_info_from_vastdb[["EVENT", "REF_CO", "CO_C1", "CO_A", "CO_C2"]],
                on="EVENT",
                how="left",
            ).reset_index(drop=True)
            event_found_mask = inclusion_levels_full["REF_CO"].notnull()
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
                    inclusion_levels_full["COMPLEX"] == event_type, "CO_C1"
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
            # drop events not found in VastDB
            inclusion_levels_full = inclusion_levels_full.loc[
                inclusion_levels_full["REF_CO"].notnull()
            ].reset_index(drop=True)

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
                normalized_gene_expression["gene_id"]
            )

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
            event_info = pd.DataFrame(event_info).drop_duplicates()

            flattened_inclusion_levels_full.to_parquet(
                os.path.join(
                    self.cache_dir, "flattened_inclusion_levels_full_filtered.parquet"
                ),
                index=False,
            )
            event_info.to_parquet(
                os.path.join(self.cache_dir, "event_info_filtered.parquet"), index=False
            )

            print("Total number of PSI values:", len(flattened_inclusion_levels_full))
            print("Total number of events:", len(event_info))

            print("Number of PSI values of each event type:")
            print(flattened_inclusion_levels_full["EVENT_TYPE"].value_counts())

            print("Number of events of each event type:")
            print(event_info["EVENT_TYPE"].value_counts())

            print("Flattened data cached")

    def setup(self, stage: str = None):
        print("Loading filtered and flattened data from cache")
        self.normalized_gene_expression = pd.read_parquet(
            os.path.join(self.cache_dir, "normalized_gene_expression.parquet")
        )
        self.flattened_inclusion_levels_full = pd.read_parquet(
            os.path.join(
                self.cache_dir, "flattened_inclusion_levels_full_filtered.parquet"
            )
        )
        self.event_info = pd.read_parquet(
            os.path.join(self.cache_dir, "event_info_filtered.parquet")
        )
        self.splicing_factor_expression_levels = pd.read_parquet(
            os.path.join(self.cache_dir, "splicing_factor_expression_levels.parquet")
        )
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
        if self.gene_expression_metric == "count":
            gene_expression_metric_cols = [
                i for i in self.normalized_gene_expression.columns[2:] if (not "_" in i)
            ]
        else:
            gene_expression_metric_cols = [
                i
                for i in self.normalized_gene_expression.columns[2:]
                if i.endswith(self.gene_expression_metric)
            ]
        self.normalized_gene_expression = self.normalized_gene_expression[
            self.normalized_gene_expression.columns[:2].to_list()
            + gene_expression_metric_cols
        ]
        self.normalized_gene_expression = self.normalized_gene_expression.rename(
            columns={i: i.split("_")[0] for i in gene_expression_metric_cols}
        )  # remove the metric suffix from the column names
        print(
            f"Kept {len(gene_expression_metric_cols)} columns with the gene expression metric '{self.gene_expression_metric}'"
        )
        self.normalized_gene_expression_flattened = (
            self.normalized_gene_expression.melt(
                id_vars=["gene_id", "alias"], var_name="sample", value_name="expression"
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
        self.unified_data = self.unified_data.merge(
            self.normalized_gene_expression_flattened,
            left_on=["GENE_ID", "SAMPLE"],
            right_on=["gene_id", "sample"],
            how="left",
        )
        self.unified_data = self.unified_data.drop(
            columns=["gene_id", "sample", "alias"]
        )
        if self.remove_events_without_gene_expression_data:
            assert (
                self.unified_data["expression"].notnull().all()
            ), "Some events without gene expression data are present in the unified data although they should have been removed"
        print(
            f"Merged gene expression values with unified data ({100 * (original_unified_data_len - len(self.unified_data)) / original_unified_data_len:.2f}% of events without gene expression data)"
        )
        if self.event_types_to_model != "ALL":
            original_unified_data_len = len(self.unified_data)
            self.unified_data = self.unified_data[
                self.unified_data["EVENT_TYPE"].isin(self.event_types_to_model)
            ].reset_index(drop=True)
            print(
                f"Filtered unified data to only include event types {self.event_types_to_model} ({100 * len(self.unified_data) / original_unified_data_len:.2f}% of the original data)"
            )

        # create datasets for training, validation, and testing
        # train dataset
        self.train_data = self.unified_data[
            self.unified_data["CHR"].isin(self.train_chromosomes)
        ].reset_index(drop=True)

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

        print("Number of PSI values of each event type:")
        full_value_counts = self.unified_data["EVENT_TYPE"].value_counts()
        train_value_counts = self.train_data["EVENT_TYPE"].value_counts()
        for event_type in self.train_data["EVENT_TYPE"].unique():
            print(
                f"{event_type}: {train_value_counts[event_type]} ({train_value_counts[event_type] / full_value_counts[event_type] * 100:.2f}%)"
            )

        # val dataset
        self.val_data = self.unified_data[
            self.unified_data["CHR"].isin(self.val_chromosomes)
        ].reset_index(drop=True)

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

        print("Number of PSI values of each event type:")
        val_value_counts = self.val_data["EVENT_TYPE"].value_counts()
        for event_type in self.val_data["EVENT_TYPE"].unique():
            print(
                f"{event_type}: {val_value_counts[event_type]} ({val_value_counts[event_type] / full_value_counts[event_type] * 100:.2f}%)"
            )

        # test dataset
        self.test_data = self.unified_data[
            self.unified_data["CHR"].isin(self.test_chromosomes)
        ].reset_index(drop=True)

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

        print("Number of PSI values of each event type:")
        test_value_counts = self.test_data["EVENT_TYPE"].value_counts()
        for event_type in self.test_data["EVENT_TYPE"].unique():
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
        self.remove_events_without_gene_expression_data = self.config["train_config"][
            "remove_events_without_gene_expression_data"
        ]
        self.gene_expression_metric = self.config["train_config"][
            "gene_expression_metric"
        ]
        assert self.gene_expression_metric in [
            "count",
            "RPKM",
            "log2RPKM",
            "TPM",
            "log2TPM",
        ], "Invalid gene expression metric specified in config, must be one of 'count', 'RPKM', 'log2RPKM', 'TPM', 'log2TPM'"
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

        # default config chromosome split so that train-val-test split is 70-10-20 roughly amoung filtered splicing events
        # train proportion = 70.61341911926058%
        # val proportion = 9.178465157513145%
        # test proportion = 20.20811572322628%
        # split was computed using the utils.chromosome_split function
        self.train_chromosomes = self.config["train_config"]["train_chromosomes"]
        self.test_chromosomes = self.config["train_config"]["test_chromosomes"]
        self.val_chromosomes = self.config["train_config"]["val_chromosomes"]

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.config["train_config"]["batch_size"],
            shuffle=True,
            pin_memory=True,
            num_workers=self.config["train_config"]["num_workers"],
            worker_init_fn=worker_init_fn,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.config["train_config"]["batch_size"],
            shuffle=False,
            pin_memory=True,
            num_workers=self.config["train_config"]["num_workers"],
            worker_init_fn=worker_init_fn,
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_dataset,
            batch_size=self.config["train_config"]["batch_size"],
            shuffle=False,
            pin_memory=True,
            num_workers=self.config["train_config"]["num_workers"],
            worker_init_fn=worker_init_fn,
        )
