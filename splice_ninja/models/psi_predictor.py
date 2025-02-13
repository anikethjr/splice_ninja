import numpy as np
import pandas as pd
import os
import pdb
import json
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F

import lightning as L
from lightning.pytorch import LightningModule
import torchmetrics

from splice_ninja.models.SpliceAICNN import SpliceAI10k

np.random.seed(0)
torch.manual_seed(0)


class PSIPredictor(LightningModule):
    def __init__(
        self, config: dict | str, num_splicing_factors: int, has_gene_exp_values: bool
    ):
        super().__init__()
        if isinstance(config, str):
            with open(config, "r") as f:
                config = json.load(f)
        self.config = config

        name_to_model = {"SpliceAI-10k": SpliceAI10k}
        assert (
            config["train_config"]["model_name"] in name_to_model
        ), f"Model {config['train_config']['model_name']} not found. Available models: {name_to_model.keys()}"
        self.model = name_to_model[config["train_config"]["model_name"]](
            config, num_splicing_factors, has_gene_exp_values
        )

        self.loss_fn = nn.MSELoss()
