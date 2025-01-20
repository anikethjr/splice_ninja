# Processes data published by Rogalska et al. (2024) - Transcriptome-wide splicing network reveals specialized regulatory functions of the core spliceosome

import numpy as np
import pandas as pd
import os
import pdb
import json

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
        self.cache_dir = self.config["data_config"]["cache_dir"]
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

        # load psi values
        # data was provided in the VastTools output format, more details on the data format are here - https://github.com/vastgroup/vast-tools?tab=readme-ov-file#combine-output-format
        self.inclusion_levels_full = pd.read_csv(
            os.path.join(
                self.config["data_config"]["data_dir"],
                "INCLUSION_LEVELS_FULL-Hs2331.tab",
            ),
            sep="\t",
        )

        # keep only the events that are in the chromosomes in the split
        self.inclusion_levels_full["CHR"] = self.inclusion_levels_full["COORD"].apply(
            lambda x: x.split(":")[0]
        )

        # load the genome
        os.makedirs(os.path.join(self.cache_dir, "genomes"), exist_ok=True)
        self.genome = genomepy.Genome(
            "hg38", genomes_dir=os.path.join(self.cache_dir, "genomes")
        )  # only need hg38 since the data is from human cell lines
