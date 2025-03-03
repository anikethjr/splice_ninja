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
from pl_bolts.optimizers.lr_scheduler import LinearWarmupCosineAnnealingLR

from splice_ninja.models.SpliceAICNN import SpliceAI10k

np.random.seed(0)
torch.manual_seed(0)


class PSIPredictor(LightningModule):
    def __init__(
        self, config: dict | str, num_splicing_factors: int, has_gene_exp_values: bool
    ):
        super().__init__()
        self.save_hyperparameters()

        if isinstance(config, str):
            with open(config, "r") as f:
                config = json.load(f)
        self.config = config

        # define model
        self.name_to_model = {"SpliceAI10k": SpliceAI10k}
        assert (
            config["train_config"]["model_name"] in self.name_to_model
        ), f"Model {config['train_config']['model_name']} not found. Available models: {self.name_to_model.keys()}"
        self.model = self.name_to_model[config["train_config"]["model_name"]](
            config, num_splicing_factors, has_gene_exp_values
        )

        # define loss function
        self.loss_fn = nn.MSELoss()

        # define metrics
        # no spearmanR for train metrics to avoid memory issues
        self.train_metrics_dict = torchmetrics.MetricCollection(
            {
                "mse": torchmetrics.MeanSquaredError(),
                "mae": torchmetrics.MeanAbsoluteError(),
                "r2": torchmetrics.R2Score(),
                "pearsonR": torchmetrics.PearsonCorrCoef(),
            }
        )
        self.eval_metrics_dict = torchmetrics.MetricCollection(
            {
                "mse": torchmetrics.MeanSquaredError(),
                "mae": torchmetrics.MeanAbsoluteError(),
                "r2": torchmetrics.R2Score(),
                "pearsonR": torchmetrics.PearsonCorrCoef(),
                "spearmanR": torchmetrics.SpearmanCorrCoef(),
            }
        )
        self.train_metrics = self.train_metrics_dict.clone(prefix="train/")
        self.val_metrics = self.eval_metrics_dict.clone(prefix="val/")

        # optimizer params
        self.optimizer_name = config["train_config"]["optimizer"]
        self.name_to_optimizer = {"Adam": torch.optim.Adam, "AdamW": torch.optim.AdamW}
        assert (
            self.optimizer_name in self.name_to_optimizer
        ), f"Optimizer {self.optimizer_name} not found. Available optimizers: {self.name_to_optimizer.keys()}"
        self.learning_rate = config["train_config"]["learning_rate"]
        self.weight_decay = config["train_config"]["weight_decay"]
        self.use_scheduler = config["train_config"]["use_scheduler"]
        self.scheduler_name = (
            "" if not self.use_scheduler else config["train_config"]["scheduler"]
        )
        self.name_to_scheduler = {
            "LinearWarmupCosineAnnealingLR": LinearWarmupCosineAnnealingLR
        }
        if self.use_scheduler:
            assert (
                self.scheduler_name in self.name_to_scheduler
            ), f"Scheduler {self.scheduler_name} not found. Available schedulers: {self.name_to_scheduler.keys()}"

    def configure_optimizers(self):
        optimizer = self.name_to_optimizer[self.optimizer_name](
            filter(lambda p: p.requires_grad, self.parameters()),
            lr=self.learning_rate,
            weight_decay=self.weight_decay,
        )
        if self.use_scheduler:
            if self.scheduler_name == "LinearWarmupCosineAnnealingLR":
                scheduler = self.name_to_scheduler[self.scheduler_name](
                    optimizer,
                    warmup_epochs=self.config["train_config"]["scheduler_params"][
                        "warmup_epochs"
                    ],
                    max_epochs=self.config["train_config"]["scheduler_params"][
                        "max_epochs"
                    ],
                )
                scheduler_config = {
                    "scheduler": scheduler,
                    "interval": "step",
                    "frequency": 1,
                }
                return [optimizer], [scheduler_config]

        return optimizer

    def forward(self, batch):
        pred_psi_val = self.model(batch)
        return pred_psi_val

    def training_step(self, batch, batch_idx):
        pred_psi_val = self(batch)
        loss = self.loss_fn(pred_psi_val, batch["psi_val"])
        self.log("train/loss", loss, on_step=True, on_epoch=True)
        self.log_dict(
            self.train_metrics(pred_psi_val, batch["psi_val"]),
            on_step=False,
            on_epoch=True,
        )
        return loss

    def validation_step(self, batch, batch_idx, dataloader_idx=0):
        pred_psi_val = self(batch)
        loss = self.loss_fn(pred_psi_val, batch["psi_val"])
        self.log("val/loss", loss, on_step=False, on_epoch=True, sync_dist=True)
        self.log_dict(
            self.val_metrics(pred_psi_val, batch["psi_val"]),
            on_step=False,
            on_epoch=True,
            sync_dist=True,
        )
        return loss

    def predict_step(self, batch, batch_idx, dataloader_idx=0):
        pred_psi_val = self(batch)
        return {
            "pred_psi_val": pred_psi_val,
            "psi_val": batch["psi_val"],
        }
