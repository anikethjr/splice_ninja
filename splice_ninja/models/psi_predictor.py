import numpy as np
import pandas as pd
import os
import pdb
import json
from tqdm import tqdm
from scipy.stats import spearmanr, pearsonr

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

        # store val set predictions to compute more complex metrics
        self.val_event_ids = []
        self.val_event_types = []
        self.val_samples = []
        self.val_psi_vals = []
        self.val_pred_psi_vals = []

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

        # store predictions for more complex metrics
        self.val_event_ids.extend(batch["event_id"].detach().cpu())
        self.val_event_types.extend(batch["event_type"].detach().cpu())
        self.val_samples.extend(batch["sample"].detach().cpu())
        self.val_psi_vals.extend(batch["psi_val"].detach().cpu())
        self.val_pred_psi_vals.extend(pred_psi_val.detach().cpu())

        return loss

    def on_validation_epoch_end(self):
        """
        Gather predictions across all GPUs and compute the final correlation metrics.
        """
        print("Computing final validation metrics...")
        # convert lists to torch tensors
        val_event_ids = torch.tensor(self.val_event_ids)
        val_event_types = torch.tensor(self.val_event_types)
        val_samples = torch.tensor(self.val_samples)
        val_psi_vals = torch.tensor(self.val_psi_vals)
        val_pred_psi_vals = torch.tensor(self.val_pred_psi_vals)

        # determine the max length across all processes
        local_size = val_pred_psi_vals.shape[0]
        all_sizes = self.all_gather(torch.tensor([local_size], device=self.device))
        max_size = all_sizes.max().item()
        print(
            f"On process {self.global_rank}, local size: {local_size}, max size: {max_size}"
        )

        # pad all tensors to the same length
        pad_value = float("nan")  # Use NaN so we can ignore padding later
        val_event_ids = F.pad(
            val_event_ids, (0, max_size - local_size), value=pad_value
        )
        val_event_types = F.pad(
            val_event_types, (0, max_size - local_size), value=pad_value
        )
        val_samples = F.pad(val_samples, (0, max_size - local_size), value=pad_value)
        val_psi_vals = F.pad(val_psi_vals, (0, max_size - local_size), value=pad_value)
        val_pred_psi_vals = F.pad(
            val_pred_psi_vals, (0, max_size - local_size), value=pad_value
        )

        # gather all predictions across all processes
        val_event_ids = self.all_gather(val_event_ids)
        val_event_types = self.all_gather(val_event_types)
        val_samples = self.all_gather(val_samples)
        val_psi_vals = self.all_gather(val_psi_vals)
        val_pred_psi_vals = self.all_gather(val_pred_psi_vals)

        # Only compute metrics on rank 0
        if self.global_rank == 0:
            # flatten all tensors
            val_event_ids = val_event_ids.view(-1)
            val_event_types = val_event_types.view(-1)
            val_samples = val_samples.view(-1)
            val_psi_vals = val_psi_vals.view(-1)
            val_pred_psi_vals = val_pred_psi_vals.view(-1)

            # remove padding and convert to numpy
            val_event_ids = val_event_ids[~torch.isnan(val_event_ids)].cpu().numpy()
            val_event_types = (
                val_event_types[~torch.isnan(val_event_types)].cpu().numpy()
            )
            val_samples = val_samples[~torch.isnan(val_samples)].cpu().numpy()
            val_psi_vals = val_psi_vals[~torch.isnan(val_psi_vals)].cpu().numpy()
            val_pred_psi_vals = (
                val_pred_psi_vals[~torch.isnan(val_pred_psi_vals)].cpu().numpy()
            )

            # create a dataframe to store all predictions
            preds_df = pd.DataFrame(
                {
                    "event_id": val_event_ids,
                    "event_type": val_event_types,
                    "sample": val_samples,
                    "psi_val": val_psi_vals,
                    "pred_psi_val": val_pred_psi_vals,
                }
            )
            preds_df.to_csv(
                os.path.join(
                    self.config["train_config"]["saved_models_dir"],
                    "psi_predictor_test"
                    if "run_name" not in self.config["train_config"]
                    else self.config["train_config"]["run_name"],
                    "latest_val_preds.csv",
                ),
                index=False,
            )
            print(
                f"Gathered predictions across all processes. Total number of predictions: {preds_df.shape[0]}"
            )

            # compute correlation metrics
            # first compute average PSI prediction per event and compare with the ground truth
            avg_per_event = (
                preds_df[["event_id", "psi_val", "pred_psi_val"]]
                .groupby("event_id")
                .mean()
            )
            avg_per_event = avg_per_event.reset_index()
            avg_per_event_spearmanR = spearmanr(
                avg_per_event["psi_val"], avg_per_event["pred_psi_val"]
            )[0]
            avg_per_event_pearsonR = pearsonr(
                avg_per_event["psi_val"], avg_per_event["pred_psi_val"]
            )[0]
            self.log(
                "val/avg_per_event_spearmanR",
                avg_per_event_spearmanR,
                on_step=False,
                on_epoch=True,
            )
            self.log(
                "val/avg_per_event_pearsonR",
                avg_per_event_pearsonR,
                on_step=False,
                on_epoch=True,
            )
            print(
                f"SpearmanR between avg predicted PSI of an event across samples and the average ground truth: {avg_per_event_spearmanR}"
            )
            print(
                f"PearsonR between avg predicted PSI of an event across samples and the average ground truth: {avg_per_event_pearsonR}"
            )

            # now for every event that is observed in at least 10 samples, compute the correlation metrics across samples
            # if an event is observed in less than 10 samples, we skip it
            # then log the average of these metrics
            sample_counts = preds_df["event_id"].value_counts()
            sample_counts = sample_counts[sample_counts >= 10]
            sample_counts = sample_counts.index
            sample_wise_spearmanR = []
            sample_wise_pearsonR = []
            for event_id in tqdm(sample_counts):
                event_df = preds_df[preds_df["event_id"] == event_id]
                spearmanR = spearmanr(event_df["psi_val"], event_df["pred_psi_val"])[0]
                pearsonR = pearsonr(event_df["psi_val"], event_df["pred_psi_val"])[0]
                sample_wise_spearmanR.append(spearmanR)
                sample_wise_pearsonR.append(pearsonR)
            avg_sample_wise_spearmanR = np.mean(sample_wise_spearmanR)
            avg_sample_wise_pearsonR = np.mean(sample_wise_pearsonR)
            self.log(
                "val/avg_sample_wise_spearmanR",
                avg_sample_wise_spearmanR,
                on_step=False,
                on_epoch=True,
            )
            self.log(
                "val/avg_sample_wise_pearsonR",
                avg_sample_wise_pearsonR,
                on_step=False,
                on_epoch=True,
            )
            print(
                f"Number of events observed in at least 10 samples: {len(sample_counts)}"
            )
            print(
                f"Average SpearmanR across events between predicted PSI and ground truth in different conditions (min 10 conditions): {avg_sample_wise_spearmanR}"
            )
            print(
                f"Average PearsonR across events between predicted PSI and ground truth in different conditions (min 10 conditions): {avg_sample_wise_pearsonR}"
            )

        # clear the stored predictions
        self.val_event_ids.clear()
        self.val_event_types.clear()
        self.val_samples.clear()
        self.val_psi_vals.clear()
        self.val_pred_psi_vals.clear()
        self.val_event_ids = []
        self.val_event_types = []
        self.val_samples = []
        self.val_psi_vals = []
        self.val_pred_psi_vals = []

    def predict_step(self, batch, batch_idx, dataloader_idx=0):
        pred_psi_val = self(batch)
        return {
            "pred_psi_val": pred_psi_val,
            "psi_val": batch["psi_val"],
        }
