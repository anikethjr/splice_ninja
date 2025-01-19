# Processes data published by Rogalska et al. (2024) - Transcriptome-wide splicing network reveals specialized regulatory functions of the core spliceosome

import numpy as np
import pandas as pd
import os
import pdb
import json

import torch
from torch.utils.data import Dataset, DataLoader

np.random.seed(0)
torch.manual_seed(0)


class KnockdownData(Dataset):
    def __init__(self, config, split):
        self.config = config
        self.split = split
        self.cache_dir = self.config["data_config"]["cache_dir"]
        self.chromosomes_in_split = self.config["training_config"][
            f"{self.split}_chromosomes"
        ]

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
        self.inclusion_levels_full = self.inclusion_levels_full[
            self.inclusion_levels_full["CHR"].isin(self.chromosomes_in_split)
        ]
        self.inclusion_levels_full = self.inclusion_levels_full.reset_index(drop=True)

    def __len__(self):
        return len(self.inclusion_levels_full)

    def __getitem__(self, idx):
        splicing_event = self.inclusion_levels_full.iloc[idx]

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
