from lightning.pytorch.utilities.types import EVAL_DATALOADERS, TRAIN_DATALOADERS
import numpy as np
import pandas as pd
import os
import pdb
import json
import urllib
from tqdm import tqdm
from scipy import stats
from statsmodels.stats.multitest import multipletests

import genomepy

import torch
import torch.distributed as dist
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


class TrASPrGTExBenchmarkDataset(Dataset):
    def __init__(self, data_module):
        self.data_module = data_module
        self.input_size = self.data_module.config["train_config"]["input_size"]
        self.genome_name = "GRCh38.p14"
        self.genomes_dir = os.path.join(self.data_module.cache_dir, "genomes")
        self.genome = None  # Will be initialized per worker

        self.data = self.data_module.core_benchmark_data

    def __len__(self):
        return len(self.data)

    def get_psi_val(self, idx):
        # get the PSI value for the idx-th row
        row = self.data.iloc[idx]
        event_id = row["EVENT"]
        event_type = row["EVENT_TYPE"]
        change_case = row["CHANGE_CASE"]
        tissue = row["TISSUE"]
        psi_val = row["PSI"]

        # get splicing factor expression values
        splicing_factor_exp_values = self.data_module.splicing_factor_expression_levels[
            tissue + "_log2TPM_rel_norm"
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
            "psi_val": psi_val.astype(np.float32),
            "gene_exp": (-1.0).astype(np.float32),
            "splicing_factor_exp_values": splicing_factor_exp_values.astype(np.float32),
            "event_type": self.data_module.event_type_to_ind[event_type],
            "event_id": self.data_module.event_id_to_ind[event_id],
            "tissue": self.data_module.tissue_to_ind[tissue],
        }

    def __getitem__(self, idx):
        return self.get_psi_val(idx)


class TrASPrGTExBenchmarkData(LightningDataModule):
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

        # download the core benchmark data if it does not exist
        if not os.path.exists(
            os.path.join(
                self.cache_dir,
                "TrASPr_GTEx_benchmark_data",
                "TrASPr_GTEx_PSI_values.tsv",
            )
        ):
            print("Downloading the core benchmark data")
            os.makedirs(
                os.path.join(self.cache_dir, "TrASPr_GTEx_benchmark_data"),
                exist_ok=True,
            )
            url = "https://raw.githubusercontent.com/AstroSign/TrASPr_model/refs/heads/main/plot_script/data/GTEx_data.tsv"
            urllib.request.urlretrieve(
                url,
                os.path.join(
                    self.cache_dir,
                    "TrASPr_GTEx_benchmark_data",
                    "TrASPr_GTEx_PSI_values.tsv",
                ),
            )
            print("Core benchmark data downloaded")

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

        # prepare the final benchmark data if it does not exist
        if not os.path.exists(
            os.path.join(self.cache_dir, "TrASPr_GTEx_benchmark_data", "final.csv")
        ):
            core_benchmark_data = pd.read_csv(
                os.path.join(
                    self.cache_dir,
                    "TrASPr_GTEx_benchmark_data",
                    "TrASPr_GTEx_PSI_values.tsv",
                ),
                sep="\t",
            )
            print(
                "Core benchmark data loaded, number of examples: ",
                core_benchmark_data.shape[0],
            )

            # join with VAST-DB event information to get information needed for input construction
            event_info = pd.read_csv(
                os.path.join(
                    self.cache_dir, "VastDB", "hg38", "EVENT_INFO-hg38.tab.gz"
                ),
                sep="\t",
            )
            core_benchmark_data["CO_A"] = core_benchmark_data.apply(
                lambda x: f"{x['Chr']}:{x['Exon_start']}-{x['Exon_end']}", axis=1
            )
            core_benchmark_data = core_benchmark_data.merge(
                event_info, how="inner", on="CO_A"
            )
            print(
                "Core benchmark data joined with event information, number of examples: ",
                core_benchmark_data.shape[0],
            )

            # create spliced in and out segments
            core_benchmark_data_final = {}
            core_benchmark_data_final["EVENT"] = []
            core_benchmark_data_final["EVENT_TYPE"] = []
            core_benchmark_data_final["CHANGE_CASE"] = []
            core_benchmark_data_final["TISSUE"] = []
            core_benchmark_data_final["PSI"] = []
            core_benchmark_data_final["GENE_ID"] = []
            core_benchmark_data_final["CHR"] = []
            core_benchmark_data_final["STRAND"] = []
            core_benchmark_data_final["SPLICED_IN_EVENT_SEGMENTS"] = []
            core_benchmark_data_final["SPLICED_OUT_EVENT_SEGMENTS"] = []
            core_benchmark_data_final["FULL_EVENT_COORD"] = []

            for i in range(core_benchmark_data.shape[0]):
                row = core_benchmark_data.iloc[i]
                core_benchmark_data_final["EVENT"].append(row["ID"])
                core_benchmark_data_final["EVENT_TYPE"].append("EX")
                core_benchmark_data_final["CHANGE_CASE"].append(row["Change_case"])
                core_benchmark_data_final["TISSUE"].append(row["Tissue"])
                core_benchmark_data_final["PSI"].append(row["Label"])
                core_benchmark_data_final["GENE_ID"].append(row["gene_id"])
                core_benchmark_data_final["CHR"].append(row["Chr"])
                core_benchmark_data_final["STRAND"].append(
                    "." if row["Strand"] == "+" else "-"
                )
                core_benchmark_data_final["FULL_EVENT_COORD"].append(row["FullCO"])

                # get the spliced in and out segments
                if row["Strand"] == "+":
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

                core_benchmark_data_final["SPLICED_IN_EVENT_SEGMENTS"].append(
                    ",".join(spliced_in_event_segments)
                )
                core_benchmark_data_final["SPLICED_OUT_EVENT_SEGMENTS"].append(
                    ",".join(spliced_out_event_segments)
                )

            core_benchmark_data_final = pd.DataFrame(core_benchmark_data_final)

            core_benchmark_data_final.to_csv(
                os.path.join(self.cache_dir, "TrASPr_GTEx_benchmark_data", "final.csv"),
                index=False,
            )
            print(
                "Final benchmark data prepared, number of examples: ",
                core_benchmark_data_final.shape[0],
            )

        # now get splicing factor expression values in GTEx tissues
        if not os.path.exists(
            os.path.join(
                self.cache_dir,
                "TrASPr_GTEx_benchmark_data",
                "splicing_factor_expression_levels.parquet",
            )
        ):
            assert os.path.exists(
                os.path.join(self.cache_dir, "normalized_gene_expression.parquet")
            ), "Normalized gene expression file from knockdown data not found. Please run the knockdown data preparation first."
            assert os.path.exists(
                os.path.join(
                    self.cache_dir, "splicing_factor_expression_levels.parquet"
                )
            ), "Splicing factor expression levels file from knockdown data not found. Please run the knockdown data preparation first."

            # load processed expression data from knockdown data
            knockdown_data_normalized_gene_expression = pd.read_parquet(
                os.path.join(self.cache_dir, "normalized_gene_expression.parquet")
            )
            knockdown_data_splicing_factor_expression_levels = pd.read_parquet(
                os.path.join(
                    self.cache_dir, "splicing_factor_expression_levels.parquet"
                )
            )

            # load the core benchmark data
            core_benchmark_data = pd.read_csv(
                os.path.join(self.cache_dir, "TrASPr_GTEx_benchmark_data", "final.csv"),
            )
            unique_tissues = core_benchmark_data["Tissue"].unique()

            # download gene counts for each tissue
            for tissue in unique_tissues:
                if not os.path.exists(
                    os.path.join(
                        self.cache_dir,
                        "TrASPr_GTEx_benchmark_data",
                        f"gene_reads_v10_{tissue.lower()}.gct.gz",
                    )
                ):
                    print(f"Downloading gene counts for {tissue}")
                    url = f"https://storage.googleapis.com/adult-gtex/bulk-gex/v10/rna-seq/counts-by-tissue/gene_reads_v10_{tissue.lower()}.gct.gz"
                    urllib.request.urlretrieve(
                        url,
                        os.path.join(
                            self.cache_dir,
                            "TrASPr_GTEx_benchmark_data",
                            f"gene_reads_v10_{tissue.lower()}.gct.gz",
                        ),
                    )
                    print(f"Gene counts for {tissue} downloaded")

            # now load the gene counts for each tissue and get the expression values for the splicing factors
            all_tissues_median_expression_from_counts = pd.DataFrame()
            all_tissues_median_expression_from_counts[
                "gene_id"
            ] = knockdown_data_normalized_gene_expression["gene_id"]
            all_tissues_median_expression_from_counts[
                "length"
            ] = knockdown_data_normalized_gene_expression["length"]

            for tissue in unique_tissues:
                print(f"Loading gene counts for {tissue}")
                tissue_gene_counts = pd.read_csv(
                    os.path.join(
                        self.cache_dir,
                        "TrASPr_GTEx_benchmark_data",
                        f"gene_reads_v10_{tissue.lower()}.gct.gz",
                    ),
                    sep="\t",
                    skiprows=2,
                )
                tissue_gene_counts["gene_id"] = (
                    tissue_gene_counts["Name"].str.split(".").str[0]
                )
                tissue_gene_counts = tissue_gene_counts.drop_duplicates(
                    subset=["gene_id"]
                )
                tissue_gene_counts = tissue_gene_counts.merge(
                    knockdown_data_normalized_gene_expression[["gene_id", "length"]],
                    how="inner",
                    on="gene_id",
                )

                tissue_samples = []
                for col in tissue_gene_counts.columns[2:-2]:
                    assert col.startswith("GTEX-")
                    tissue_gene_counts[col + "_TPM"] = (
                        tissue_gene_counts[col] / tissue_gene_counts["length"]
                    )
                    rpk_sum = tissue_gene_counts[col + "_TPM"].sum()
                    tissue_gene_counts[col + "_TPM"] = (
                        tissue_gene_counts[col + "_TPM"] / rpk_sum
                    ) * 1e6
                    tissue_gene_counts[col + "_log2TPM"] = np.log2(
                        tissue_gene_counts[col + "_TPM"] + 1
                    )
                    tissue_gene_counts = tissue_gene_counts.copy()
                    tissue_samples.append(col + "_log2TPM")
                tissue_gene_counts[f"{tissue}_log2TPM"] = np.median(
                    tissue_gene_counts[tissue_samples].values, axis=1
                )

                all_tissues_median_expression_from_counts = (
                    all_tissues_median_expression_from_counts.merge(
                        tissue_gene_counts[["gene_id", f"{tissue}_log2TPM"]],
                        how="inner",
                        on="gene_id",
                    )
                )

            # now compute the splicing factor relative normalized expression levels
            all_tissues_splicing_factor_expression_levels = (
                knockdown_data_splicing_factor_expression_levels[
                    ["gene_id", "AV_Controls_log2TPM_rel_norm"]
                ].copy()
            )
            all_tissues_splicing_factor_expression_levels = (
                all_tissues_splicing_factor_expression_levels.merge(
                    all_tissues_median_expression_from_counts[
                        ["gene_id"] + [f"{tissue}_log2TPM" for tissue in unique_tissues]
                    ],
                    how="left",
                    on="gene_id",
                )
            )
            for tissue in unique_tissues:
                all_tissues_splicing_factor_expression_levels[
                    f"{tissue}_log2TPM"
                ] = all_tissues_splicing_factor_expression_levels[
                    f"{tissue}_log2TPM"
                ].fillna(
                    0
                )
                all_tissues_splicing_factor_expression_levels[
                    f"{tissue}_log2TPM_rel_norm"
                ] = (
                    all_tissues_splicing_factor_expression_levels[f"{tissue}_log2TPM"]
                    / all_tissues_splicing_factor_expression_levels[
                        f"{tissue}_log2TPM"
                    ].sum()
                )
            all_tissues_splicing_factor_expression_levels = (
                all_tissues_splicing_factor_expression_levels.copy()
            )

            for tissue in unique_tissues:
                spr = stats.spearmanr(
                    all_tissues_splicing_factor_expression_levels[
                        f"{tissue}_log2TPM_rel_norm"
                    ],
                    all_tissues_splicing_factor_expression_levels[
                        "AV_Controls_log2TPM_rel_norm"
                    ],
                )[0]
                pr = stats.pearsonr(
                    all_tissues_splicing_factor_expression_levels[
                        f"{tissue}_log2TPM_rel_norm"
                    ],
                    all_tissues_splicing_factor_expression_levels[
                        "AV_Controls_log2TPM_rel_norm"
                    ],
                )[0]
                print(
                    f"Correlation between {tissue} and AV_Controls SF log2TPM_rel_norm: {spr:.4f} (Spearman), {pr:.4f} (Pearson)"
                )

            all_tissues_splicing_factor_expression_levels.to_parquet(
                os.path.join(
                    self.cache_dir,
                    "TrASPr_GTEx_benchmark_data",
                    "splicing_factor_expression_levels.parquet",
                ),
                index=False,
            )

    def setup(self, stage: str = None):
        # load the core benchmark data
        self.core_benchmark_data = pd.read_csv(
            os.path.join(self.cache_dir, "TrASPr_GTEx_benchmark_data", "final.csv"),
        )
        print(
            "Core benchmark data loaded, number of examples: ",
            self.core_benchmark_data.shape[0],
        )

        # load the splicing factor expression levels
        self.splicing_factor_expression_levels = pd.read_parquet(
            os.path.join(
                self.cache_dir,
                "TrASPr_GTEx_benchmark_data",
                "splicing_factor_expression_levels.parquet",
            ),
        )

        print(
            "Splicing factor expression values loaded, number of SF: ",
            self.splicing_factor_expression_levels.shape[0],
        )

        self.unique_tissues = self.core_benchmark_data["TISSUE"].unique()
        self.unique_tissues = sorted(self.unique_tissues.tolist())
        self.tissue_to_ind = {tissue: i for i, tissue in enumerate(self.unique_tissues)}
        print(
            f"Unique tissues: {self.unique_tissues}, tissue to ind: {self.tissue_to_ind}"
        )

        event_ids = self.core_benchmark_data["EVENT"].unique()
        event_ids = sorted(event_ids.tolist())
        self.event_id_to_ind = {event_id: i for i, event_id in enumerate(event_ids)}
        print(f"Number of unique events: {len(event_ids)}")

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

    def train_dataloader(self):
        raise NotImplementedError("This dataset does not support training.")

    def val_dataloader(self):
        raise NotImplementedError("This dataset does not support validation.")

    def test_dataloader(self):
        dataset = TrASPrGTExBenchmarkDataset(self)
        return DataLoader(
            dataset,
            batch_size=self.config["test_config"]["batch_size"],
            num_workers=self.config["test_config"]["num_workers"],
            shuffle=False,
            pin_memory=True,
        )

    def predict_dataloader(self):
        dataset = TrASPrGTExBenchmarkDataset(self)
        return DataLoader(
            dataset,
            batch_size=self.config["test_config"]["batch_size"],
            num_workers=self.config["test_config"]["num_workers"],
            shuffle=False,
            pin_memory=True,
        )
