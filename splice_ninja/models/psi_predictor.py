import numpy as np
import pandas as pd
import os
import pdb
import json
from tqdm import tqdm
from sklearn.metrics import r2_score
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


class MSELoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, pred_psi_val, psi_val, **kwargs):
        psi_val = psi_val.view(-1, 1)
        pred_psi_val = pred_psi_val.view(-1, 1)
        loss = F.mse_loss(pred_psi_val, psi_val)
        return loss


class BiasedMSELoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, pred_psi_val, psi_val, **kwargs):
        psi_val = psi_val.view(-1, 1)
        pred_psi_val = pred_psi_val.view(-1, 1)
        loss = (
            psi_val - pred_psi_val
        ) ** 2  # compute the mean squared error per sample
        # bias the loss towards intermediate PSI values
        deviation_from_half = torch.abs(psi_val - 0.5)
        # we want to increase the loss for PSI values that are closer to 0.5
        # so we divide by the deviation from 0.5 + 1 to make the loss larger
        # for values closer to 0.5
        loss = loss / (deviation_from_half + 1)
        loss = loss.mean()
        return loss


class BiasedMSELossBasedOnEventStd(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, pred_psi_val, psi_val, **kwargs):
        event_std_psi = kwargs["event_std_psi"] * 100.0
        psi_val = psi_val.view(-1, 1)
        pred_psi_val = pred_psi_val.view(-1, 1)
        event_std_psi = event_std_psi.view(-1, 1)
        loss = (
            psi_val - pred_psi_val
        ) ** 2  # compute the mean squared error per sample
        # bias the loss towards events with high standard deviation across samples
        # we want to increase the loss for events with high standard deviation
        # so we multiply the loss by the standard deviation + 1 to make the loss larger
        # for events with high standard deviation
        loss = loss * (event_std_psi + 1)
        loss = loss.mean()
        return loss


class BiasedMSELossBasedOnNumSamplesEventObserved(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, pred_psi_val, psi_val, **kwargs):
        num_samples_event_observed = kwargs["event_num_samples_observed"]
        psi_val = psi_val.view(-1, 1)
        pred_psi_val = pred_psi_val.view(-1, 1)
        num_samples_event_observed = num_samples_event_observed.view(-1, 1)
        loss = (
            psi_val - pred_psi_val
        ) ** 2  # compute the mean squared error per sample
        # bias the loss towards events that are observed in more samples
        # we want to increase the loss for events that are observed in more samples
        # so we multiply the loss by the log of the number of samples + 1 to make the loss larger
        # for events that are observed in more samples
        loss = loss * torch.log2(num_samples_event_observed + 1)
        loss = loss.mean()
        return loss


class RankingAndMSELoss(nn.Module):
    def __init__(self, margin=0, ranking_loss_weight=1e4):
        super().__init__()
        self.margin = margin
        self.ranking_loss_weight = ranking_loss_weight  # needed to balance the ranking loss with the MSE loss, ranking loss is about 1e4 times smaller than MSE loss from experiments

    def forward(self, pred_psi_val, psi_val, **kwargs):
        # Create all pairwise differences
        pred_diff = pred_psi_val.unsqueeze(1) - pred_psi_val.unsqueeze(0)  # (N, N)
        true_diff = psi_val.unsqueeze(1) - psi_val.unsqueeze(0)  # (N, N)

        # Get ranking labels: 1 if psi_val_i > psi_val_j, -1 if vice versa, 0 if equal
        ranking_labels = torch.sign(true_diff)

        # Flatten for loss computation
        pred_diff = pred_diff.flatten()
        ranking_labels = ranking_labels.flatten()

        # Apply margin ranking loss, masking out zero-label (equal) pairs
        valid_pairs = ranking_labels != 0
        if valid_pairs.sum() != 0:
            loss = (
                F.margin_ranking_loss(
                    pred_diff[valid_pairs],
                    torch.zeros_like(pred_diff[valid_pairs]),  # Target is 0 margin
                    ranking_labels[valid_pairs],
                    margin=self.margin,
                )
                * self.ranking_loss_weight
            )

            # Add MSE loss
            loss += F.mse_loss(pred_psi_val.view(-1, 1), psi_val.view(-1, 1))
        else:
            # only MSE loss if no valid pairs
            loss = F.mse_loss(pred_psi_val.view(-1, 1), psi_val.view(-1, 1))

        return loss


class BCEWithLogitsLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, pred_psi_val, psi_val, **kwargs):
        psi_val = psi_val.view(-1, 1)
        pred_psi_val = pred_psi_val.view(-1, 1)
        loss = F.binary_cross_entropy_with_logits(pred_psi_val, psi_val)
        return loss


class BiasedBCEWithLogitsLossBasedOnEventStd(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, pred_psi_val, psi_val, **kwargs):
        event_std_psi = kwargs["event_std_psi"] * 100.0
        psi_val = psi_val.view(-1, 1)
        pred_psi_val = pred_psi_val.view(-1, 1)
        event_std_psi = event_std_psi.view(-1, 1)
        loss = F.binary_cross_entropy_with_logits(
            pred_psi_val, psi_val, reduction="none"
        )
        # bias the loss towards events with high standard deviation across samples
        # we want to increase the loss for events with high standard deviation
        # so we multiply the loss by the standard deviation + 1 to make the loss larger
        # for events with high standard deviation
        loss = loss * (event_std_psi + 1)
        loss = loss.mean()
        return loss


class RankingAndBCEWithLogitsLoss(nn.Module):
    def __init__(self, margin=0, ranking_loss_weight=1):
        super().__init__()
        self.margin = margin
        self.ranking_loss_weight = ranking_loss_weight

    def forward(self, pred_psi_val, psi_val, **kwargs):
        # Create all pairwise differences
        pred_diff = pred_psi_val.unsqueeze(1) - pred_psi_val.unsqueeze(0)
        true_diff = psi_val.unsqueeze(1) - psi_val.unsqueeze(0)

        # Get ranking labels: 1 if psi_val_i > psi_val_j, -1 if vice versa, 0 if equal
        ranking_labels = torch.sign(true_diff)

        # Flatten for loss computation
        pred_diff = pred_diff.flatten()
        ranking_labels = ranking_labels.flatten()

        # Apply margin ranking loss, masking out zero-label (equal) pairs
        valid_pairs = ranking_labels != 0
        if valid_pairs.sum() != 0:
            loss = (
                F.margin_ranking_loss(
                    pred_diff[valid_pairs],
                    torch.zeros_like(pred_diff[valid_pairs]),  # Target is 0 margin
                    ranking_labels[valid_pairs],
                    margin=self.margin,
                )
                * self.ranking_loss_weight
            )

            # Add BCEWithLogits loss
            loss += F.binary_cross_entropy_with_logits(
                pred_psi_val.view(-1, 1), psi_val.view(-1, 1)
            )
        else:
            # only BCEWithLogits loss if no valid pairs
            loss = F.binary_cross_entropy_with_logits(
                pred_psi_val.view(-1, 1), psi_val.view(-1, 1)
            )

        return loss


class PSIPredictor(LightningModule):
    def __init__(
        self,
        config: dict | str,
        num_splicing_factors: int,
        has_gene_exp_values: bool,
        event_type_to_ind: dict,
        example_type_to_ind: dict,
        example_types_in_this_split_type: list,
    ):
        super().__init__()
        self.save_hyperparameters()

        if isinstance(config, str):
            with open(config, "r") as f:
                config = json.load(f)
        self.config = config

        self.num_splicing_factors = num_splicing_factors
        if (
            "do_not_use_splicing_factor_expression_data" in self.config["train_config"]
        ) and (
            self.config["train_config"]["do_not_use_splicing_factor_expression_data"]
        ):
            self.num_splicing_factors = 0
            print("Ignoring splicing factor expression data.")

        self.has_gene_exp_values = has_gene_exp_values
        if ("do_not_use_gene_expression_data" in self.config["train_config"]) and (
            self.config["train_config"]["do_not_use_gene_expression_data"]
        ):
            self.has_gene_exp_values = False
            print("Ignoring gene expression data.")

        self.event_type_to_ind = event_type_to_ind
        self.event_ind_to_type = {v: k for k, v in event_type_to_ind.items()}
        print(f"Event type to index mapping: {self.event_type_to_ind}")
        print(f"Event index to type mapping: {self.event_ind_to_type}")

        self.example_type_to_ind = example_type_to_ind
        self.example_ind_to_type = {v: k for k, v in example_type_to_ind.items()}
        print(f"Example type to index mapping: {self.example_type_to_ind}")
        print(f"Example index to type mapping: {self.example_ind_to_type}")
        self.example_types_in_this_split_type = example_types_in_this_split_type
        print(
            f"Example types in this split type: {self.example_types_in_this_split_type}"
        )

        # define model
        self.name_to_model = {"SpliceAI10k": SpliceAI10k}
        assert (
            self.config["train_config"]["model_name"] in self.name_to_model
        ), f"Model {self.config['train_config']['model_name']} not found. Available models: {self.name_to_model.keys()}"
        if "predict_mean_std_psi_and_delta" not in self.config["train_config"]:
            self.config["train_config"]["predict_mean_std_psi_and_delta"] = False
        self.predict_mean_std_psi_and_delta = self.config["train_config"][
            "predict_mean_std_psi_and_delta"
        ]
        self.model = self.name_to_model[self.config["train_config"]["model_name"]](
            self.config, self.num_splicing_factors, self.has_gene_exp_values
        )

        # define loss function
        if self.config["train_config"]["loss_fn"] == "MSELoss":
            self.loss_fn = MSELoss()
        elif (
            self.config["train_config"]["loss_fn"] == "BiasedMSELoss"
        ):  # BiasedMSELoss is a custom loss function that makes the model concentrate more on events with intermediate PSI values i.e. 0.2 < PSI < 0.8
            self.loss_fn = BiasedMSELoss()
        elif self.config["train_config"]["loss_fn"] == "BiasedMSELossBasedOnEventStd":
            self.loss_fn = BiasedMSELossBasedOnEventStd()
        elif (
            self.config["train_config"]["loss_fn"]
            == "BiasedMSELossBasedOnNumSamplesEventObserved"
        ):
            self.loss_fn = BiasedMSELossBasedOnNumSamplesEventObserved()
        elif self.config["train_config"]["loss_fn"] == "RankingAndMSELoss":
            self.loss_fn = RankingAndMSELoss()
        elif self.config["train_config"]["loss_fn"] == "BCEWithLogitsLoss":
            self.loss_fn = BCEWithLogitsLoss()
        elif (
            self.config["train_config"]["loss_fn"]
            == "BiasedBCEWithLogitsLossBasedOnEventStd"
        ):
            self.loss_fn = BiasedBCEWithLogitsLossBasedOnEventStd()
        elif self.config["train_config"]["loss_fn"] == "RankingAndBCEWithLogitsLoss":
            self.loss_fn = RankingAndBCEWithLogitsLoss()
        else:
            raise ValueError(
                f"Loss function {self.config['train_config']['loss_fn']} not found. Available loss functions: MSELoss, BiasedMSELoss, BiasedMSELossBasedOnEventStd, BiasedMSELossBasedOnNumSamplesEventObserved, RankingAndMSELoss, BCEWithLogitsLoss, BiasedBCEWithLogitsLossBasedOnEventStd, RankingAndBCEWithLogitsLoss."
            )

        if self.predict_mean_std_psi_and_delta:
            if "Logits" not in self.config["train_config"]["loss_fn"]:
                self.mean_delta_psi_loss_fn = MSELoss()
            else:
                self.mean_delta_psi_loss_fn = BCEWithLogitsLoss()

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
        self.val_example_types = []
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
        preds = self.model(batch)
        return preds

    def training_step(self, batch, batch_idx):
        if self.predict_mean_std_psi_and_delta:
            preds = self(batch)
            pred_delta_psi_val = preds[:, 0]
            pred_mean_psi_val = preds[:, 1]
            pred_std_psi_val = preds[:, 2]
            pred_psi_val = pred_mean_psi_val + (pred_delta_psi_val * pred_std_psi_val)
        else:
            pred_psi_val = self(batch)

        loss = self.loss_fn(
            pred_psi_val,
            batch["psi_val"],
            event_num_samples_observed=batch["event_num_samples_observed"],
            event_mean_psi=batch["event_mean_psi"],
            event_std_psi=batch["event_std_psi"],
            event_min_psi=batch["event_min_psi"],
            event_max_psi=batch["event_max_psi"],
        )
        if self.predict_mean_std_psi_and_delta:
            self.log(
                "train/psi_val_loss", loss, on_step=True, on_epoch=True, sync_dist=True
            )
            mean_psi_loss = self.mean_delta_psi_loss_fn(
                pred_mean_psi_val, batch["event_mean_psi"]
            )
            std_psi_loss = self.mean_delta_psi_loss_fn(
                pred_std_psi_val, batch["event_std_psi"]
            )
            loss = loss + mean_psi_loss + std_psi_loss
            self.log("train/mean_psi_loss", mean_psi_loss, on_step=True, on_epoch=True)
            self.log("train/std_psi_loss", std_psi_loss, on_step=True, on_epoch=True)
        self.log("train/loss", loss, on_step=True, on_epoch=True)

        if "Logits" in self.config["train_config"]["loss_fn"]:
            pred_psi_val = torch.sigmoid(pred_psi_val)
        self.log_dict(
            self.train_metrics(pred_psi_val, batch["psi_val"]),
            on_step=False,
            on_epoch=True,
        )
        self.log(
            "train/lr", self.trainer.optimizers[0].param_groups[0]["lr"], on_step=True
        )
        if "weight_decay" in self.trainer.optimizers[0].param_groups[0]:
            self.log(
                "train/weight_decay",
                self.trainer.optimizers[0].param_groups[0]["weight_decay"],
                on_step=True,
            )
        return loss

    def validation_step(self, batch, batch_idx, dataloader_idx=0):
        if self.predict_mean_std_psi_and_delta:
            preds = self(batch)
            pred_delta_psi_val = preds[:, 0]
            pred_mean_psi_val = preds[:, 1]
            pred_std_psi_val = preds[:, 2]
            pred_psi_val = pred_mean_psi_val + (pred_delta_psi_val * pred_std_psi_val)
        else:
            pred_psi_val = self(batch)

        loss = self.loss_fn(
            pred_psi_val,
            batch["psi_val"],
            event_num_samples_observed=batch["event_num_samples_observed"],
            event_mean_psi=batch["event_mean_psi"],
            event_std_psi=batch["event_std_psi"],
            event_min_psi=batch["event_min_psi"],
            event_max_psi=batch["event_max_psi"],
        )
        if self.predict_mean_std_psi_and_delta:
            self.log(
                "val/psi_val_loss", loss, on_step=False, on_epoch=True, sync_dist=True
            )
            mean_psi_loss = self.mean_delta_psi_loss_fn(
                pred_mean_psi_val, batch["event_mean_psi"]
            )
            std_psi_loss = self.mean_delta_psi_loss_fn(
                pred_std_psi_val, batch["event_std_psi"]
            )
            loss = loss + mean_psi_loss + std_psi_loss
            self.log(
                "val/mean_psi_loss",
                mean_psi_loss,
                on_step=False,
                on_epoch=True,
                sync_dist=True,
            )
            self.log(
                "val/std_psi_loss",
                std_psi_loss,
                on_step=False,
                on_epoch=True,
                sync_dist=True,
            )
        self.log("val/loss", loss, on_step=False, on_epoch=True, sync_dist=True)

        if "Logits" in self.config["train_config"]["loss_fn"]:
            pred_psi_val = torch.sigmoid(pred_psi_val)
        self.log_dict(
            self.val_metrics(pred_psi_val, batch["psi_val"]),
            on_step=False,
            on_epoch=True,
            sync_dist=True,
        )

        # store predictions for more complex metrics
        self.val_event_ids.extend(batch["event_id"].detach().cpu())
        self.val_event_types.extend(batch["event_type"].detach().cpu())
        self.val_example_types.extend(batch["example_type"].detach().cpu())
        self.val_samples.extend(batch["sample"].detach().cpu())
        self.val_psi_vals.extend(batch["psi_val"].detach().cpu())
        self.val_pred_psi_vals.extend(pred_psi_val.detach().cpu())

        return loss

    def on_validation_epoch_end(self):
        """
        Gather predictions across all GPUs and compute the final metrics.
        """
        print("Computing final validation metrics...")
        # convert lists to torch tensors
        val_event_ids = torch.tensor(self.val_event_ids)
        val_event_types = torch.tensor(self.val_event_types)
        val_example_types = torch.tensor(self.val_example_types)
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
        val_example_types = F.pad(
            val_example_types, (0, max_size - local_size), value=pad_value
        )
        val_samples = F.pad(val_samples, (0, max_size - local_size), value=pad_value)
        val_psi_vals = F.pad(val_psi_vals, (0, max_size - local_size), value=pad_value)
        val_pred_psi_vals = F.pad(
            val_pred_psi_vals, (0, max_size - local_size), value=pad_value
        )

        # gather all predictions across all processes
        val_event_ids = self.all_gather(val_event_ids)
        val_event_types = self.all_gather(val_event_types)
        val_example_types = self.all_gather(val_example_types)
        val_samples = self.all_gather(val_samples)
        val_psi_vals = self.all_gather(val_psi_vals)
        val_pred_psi_vals = self.all_gather(val_pred_psi_vals)

        # Only compute metrics on rank 0
        if self.global_rank == 0:
            # flatten all tensors
            val_event_ids = val_event_ids.view(-1)
            val_event_types = val_event_types.view(-1)
            val_example_types = val_example_types.view(-1)
            val_samples = val_samples.view(-1)
            val_psi_vals = val_psi_vals.view(-1)
            val_pred_psi_vals = val_pred_psi_vals.view(-1)

            # remove padding and convert to numpy
            val_event_ids = val_event_ids[~torch.isnan(val_event_ids)].cpu().numpy()
            val_event_types = (
                val_event_types[~torch.isnan(val_event_types)].cpu().numpy()
            )
            val_example_types = (
                val_example_types[~torch.isnan(val_example_types)].cpu().numpy()
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
                    "example_type": val_example_types,
                    "sample": val_samples,
                    "psi_val": val_psi_vals,
                    "pred_psi_val": val_pred_psi_vals,
                }
            )
            # drop duplicates that might have been created to have the same number of samples across all processes
            preds_df = preds_df.drop_duplicates().reset_index(drop=True)
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

            # compute final metrics
            unique_event_types = preds_df["event_type"].unique().tolist()
            if len(unique_event_types) > 1:
                unique_event_types.append("ALL")
            unique_example_types = preds_df["example_type"].unique().tolist()
            if len(unique_example_types) > 1:
                unique_example_types.append("ALL")
            for event_type in unique_event_types:
                if event_type == "ALL":
                    event_type_df = preds_df
                    event_type_name = "ALL"
                else:
                    event_type_df = preds_df[
                        preds_df["event_type"] == event_type
                    ].reset_index(drop=True)
                    event_type_name = self.event_ind_to_type[event_type]
                for example_type in unique_example_types:
                    print(
                        f"Computing metrics for event type: {event_type}, example type: {example_type}"
                    )
                    if example_type == "ALL":
                        subset_df = event_type_df
                        example_type_name = "ALL"
                    else:
                        subset_df = event_type_df[
                            preds_df["example_type"] == example_type
                        ].reset_index(drop=True)
                        example_type_name = self.example_ind_to_type[example_type]

                    # first compute average PSI prediction per event and compare with the ground truth
                    avg_per_event = (
                        subset_df[["event_id", "psi_val", "pred_psi_val"]]
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
                        f"val/avg_per_{event_type_name}_event_in_{example_type_name}_examples_spearmanR",
                        avg_per_event_spearmanR,
                        on_step=False,
                        on_epoch=True,
                    )
                    self.log(
                        f"val/avg_per_{event_type_name}_event_in_{example_type_name}_examples_pearsonR",
                        avg_per_event_pearsonR,
                        on_step=False,
                        on_epoch=True,
                    )
                    print(
                        f"SpearmanR between avg predicted PSI of an event across samples and the average ground truth in {example_type_name} examples: {avg_per_event_spearmanR}"
                    )
                    print(
                        f"PearsonR between avg predicted PSI of an event across samples and the average ground truth in {example_type_name} examples: {avg_per_event_pearsonR}"
                    )

                    # now for every event that is observed in at least 10 samples, compute the correlation metrics across samples
                    # if an event is observed in less than 10 samples, we skip it
                    # then log the average of these metrics
                    sample_counts = subset_df["event_id"].value_counts()
                    sample_counts = sample_counts[sample_counts >= 10]
                    sample_counts = sample_counts.index
                    std_across_samples = []
                    sample_wise_spearmanR = []
                    sample_wise_pearsonR = []
                    sample_wise_r2 = []
                    for event_id in tqdm(sample_counts):
                        event_df = event_type_df[event_type_df["event_id"] == event_id]
                        std_across_samples.append(np.std(event_df["psi_val"].values))
                        spearmanR = np.nan_to_num(
                            spearmanr(event_df["psi_val"], event_df["pred_psi_val"])[0]
                        )
                        pearsonR = np.nan_to_num(
                            pearsonr(event_df["psi_val"], event_df["pred_psi_val"])[0]
                        )
                        r2 = np.nan_to_num(
                            r2_score(event_df["psi_val"], event_df["pred_psi_val"])
                        )
                        sample_wise_spearmanR.append(spearmanR)
                        sample_wise_pearsonR.append(pearsonR)
                        sample_wise_r2.append(r2)
                    sample_wise_spearmanR = np.array(sample_wise_spearmanR)
                    sample_wise_pearsonR = np.array(sample_wise_pearsonR)
                    sample_wise_r2 = np.array(sample_wise_r2)
                    avg_sample_wise_spearmanR = np.mean(sample_wise_spearmanR)
                    avg_sample_wise_pearsonR = np.mean(sample_wise_pearsonR)
                    avg_sample_wise_r2 = np.mean(sample_wise_r2)
                    self.log(
                        f"val/avg_{event_type_name}_{example_type_name}_examples_sample_wise_spearmanR",
                        avg_sample_wise_spearmanR,
                        on_step=False,
                        on_epoch=True,
                    )
                    self.log(
                        f"val/avg_{event_type_name}_{example_type_name}_examples_sample_wise_pearsonR",
                        avg_sample_wise_pearsonR,
                        on_step=False,
                        on_epoch=True,
                    )
                    self.log(
                        f"val/avg_{event_type_name}_{example_type_name}_examples_sample_wise_r2",
                        avg_sample_wise_r2,
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
                    print(
                        f"Average R2 score across events between predicted PSI and ground truth in different conditions (min 10 conditions): {avg_sample_wise_r2}"
                    )

                    # compute metrics for events with high standard deviation across samples (std > 0.2)
                    high_std_events_mask = np.array(std_across_samples) > 0.2
                    high_std_events_avg_spearmanR = np.mean(
                        np.array(sample_wise_spearmanR)[high_std_events_mask]
                    )
                    high_std_events_avg_pearsonR = np.mean(
                        np.array(sample_wise_pearsonR)[high_std_events_mask]
                    )
                    high_std_events_avg_r2 = np.mean(
                        np.array(sample_wise_r2)[high_std_events_mask]
                    )
                    self.log(
                        f"val/avg_{event_type_name}_{example_type_name}_examples_high_std_events_spearmanR",
                        high_std_events_avg_spearmanR,
                        on_step=False,
                        on_epoch=True,
                    )
                    self.log(
                        f"val/avg_{event_type_name}_{example_type_name}_examples_high_std_events_pearsonR",
                        high_std_events_avg_pearsonR,
                        on_step=False,
                        on_epoch=True,
                    )
                    self.log(
                        f"val/avg_{event_type_name}_{example_type_name}_examples_high_std_events_r2",
                        high_std_events_avg_r2,
                        on_step=False,
                        on_epoch=True,
                    )
                    print(
                        f"Number of events with high standard deviation across samples (std > 0.2): {np.sum(high_std_events_mask)}"
                    )
                    print(
                        f"Average SpearmanR across events with high std between predicted PSI and ground truth in different conditions: {high_std_events_avg_spearmanR}"
                    )
                    print(
                        f"Average PearsonR across events with high std between predicted PSI and ground truth in different conditions: {high_std_events_avg_pearsonR}"
                    )
                    print(
                        f"Average R2 score across events with high std between predicted PSI and ground truth in different conditions: {high_std_events_avg_r2}"
                    )

                    # compute metrics for events with most variance (top 25%)
                    most_var_events = np.argsort(std_across_samples)[
                        -int(0.25 * len(std_across_samples)) :
                    ]
                    most_var_events_avg_spearmanR = np.mean(
                        np.array(sample_wise_spearmanR)[most_var_events]
                    )
                    most_var_events_avg_pearsonR = np.mean(
                        np.array(sample_wise_pearsonR)[most_var_events]
                    )
                    most_var_events_avg_r2 = np.mean(
                        np.array(sample_wise_r2)[most_var_events]
                    )
                    self.log(
                        f"val/avg_{event_type_name}_{example_type_name}_examples_most_var_events_spearmanR",
                        most_var_events_avg_spearmanR,
                        on_step=False,
                        on_epoch=True,
                    )
                    self.log(
                        f"val/avg_{event_type_name}_{example_type_name}_examples_most_var_events_pearsonR",
                        most_var_events_avg_pearsonR,
                        on_step=False,
                        on_epoch=True,
                    )
                    self.log(
                        f"val/avg_{event_type_name}_{example_type_name}_examples_most_var_events_r2",
                        most_var_events_avg_r2,
                        on_step=False,
                        on_epoch=True,
                    )
                    print(
                        f"Number of events with most variance (top 25%): {len(most_var_events)}, avg std: {np.mean(np.array(std_across_samples)[most_var_events])}"
                    )
                    print(
                        f"Average SpearmanR across events with most variance between predicted PSI and ground truth in different conditions: {most_var_events_avg_spearmanR}"
                    )
                    print(
                        f"Average PearsonR across events with most variance between predicted PSI and ground truth in different conditions: {most_var_events_avg_pearsonR}"
                    )
                    print(
                        f"Average R2 score across events with most variance between predicted PSI and ground truth in different conditions: {most_var_events_avg_r2}"
                    )

                    # classification metrics to determine whether the model can predict when PSI deviates significantly from the mean in some samples
                    # we also want to see whether the model detect which events are unaffected by perturbations

                    # classification task #1 - for events with relatively low standard deviation across samples (std < 0.01)
                    # can the model detect samples for which the PSI is significantly different from the mean?
                    # we define significant difference as 0.1

                    # first get low std events
                    print(
                        "Classification task #1 - identifying significant deviations from the mean in low std events"
                    )
                    low_std_events_mask = np.array(std_across_samples) < 0.01
                    low_std_events = [
                        event_id
                        for i, event_id in enumerate(sample_counts)
                        if low_std_events_mask[i]
                    ]
                    print(f"Number of low std events: {len(low_std_events)}")
                    low_std_events_df = subset_df[
                        subset_df["event_id"].isin(low_std_events)
                    ].reset_index(drop=True)

                    # now add the mean and std PSI for these events to the dataframe
                    low_std_events_mean_psi = low_std_events_df.groupby("event_id")[
                        "psi_val"
                    ].mean()
                    low_std_events_std_psi = low_std_events_df.groupby("event_id")[
                        "psi_val"
                    ].std()
                    low_std_events_mean_psi = low_std_events_mean_psi.reset_index()
                    low_std_events_std_psi = low_std_events_std_psi.reset_index()
                    low_std_events_mean_psi.columns = ["event_id", "mean_psi_val"]
                    low_std_events_std_psi.columns = ["event_id", "std_psi_val"]
                    low_std_events_df = low_std_events_df.merge(
                        low_std_events_mean_psi, on="event_id", how="inner"
                    )
                    low_std_events_df = low_std_events_df.merge(
                        low_std_events_std_psi, on="event_id", how="inner"
                    )
                    assert np.all(
                        low_std_events_df["std_psi_val"] < 0.01
                    ), "Some events have std > 0.01, should not happen"

                    # find samples where psi_val > (mean_psi_val + 0.1) or psi_val < (mean_psi_val - 0.1)
                    low_std_events_df["sample_has_sig_lower_PSI"] = low_std_events_df[
                        "psi_val"
                    ] < (low_std_events_df["mean_psi_val"] - 0.1)
                    low_std_events_df["sample_has_sig_higher_PSI"] = low_std_events_df[
                        "psi_val"
                    ] > (low_std_events_df["mean_psi_val"] + 0.1)

                    # we check for 4 things:
                    # 1. does the model predict the mean PSI correctly? (should be within 0.01 of the ground truth)
                    # 2. does the model predict the std PSI correctly? (should be within 0.01 of the ground truth)
                    # 3. does the model correctly identify samples with significantly different PSI values? track the average percentile of the samples that are predicted to be significantly different from the mean - ranking should be based on the absolute difference between the predicted PSI and the mean predicted PSI.
                    # 4. does the model correctly predict the direction of the deviation? track the average percentile of the samples that are predicted to be significantly different from the mean - ranking should be based on the signed difference between the predicted PSI and the mean predicted PSI.

                    # check if the model predicts the mean PSI correctly
                    mean_predicted_psi = low_std_events_df.groupby("event_id")[
                        "pred_psi_val"
                    ].mean()
                    mean_predicted_psi = mean_predicted_psi.reset_index()
                    mean_predicted_psi.columns = ["event_id", "mean_predicted_psi"]
                    low_std_events_df = low_std_events_df.merge(
                        mean_predicted_psi, on="event_id", how="inner"
                    )
                    low_std_events_df["|mean_predicted_psi - mean_psi_val|"] = (
                        low_std_events_df["mean_predicted_psi"]
                        - low_std_events_df["mean_psi_val"]
                    ).abs()
                    temp = (
                        low_std_events_df[
                            ["event_id", "|mean_predicted_psi - mean_psi_val|"]
                        ]
                        .drop_duplicates()
                        .reset_index(drop=True)
                    )
                    accuracy = (
                        temp["|mean_predicted_psi - mean_psi_val|"] < 0.01
                    ).sum() / len(temp)
                    self.log(
                        f"val/{event_type_name}_{example_type_name}_low_std_events_mean_psi_accuracy",
                        accuracy,
                        on_step=False,
                        on_epoch=True,
                    )
                    print(
                        f"Mean PSI prediction accuracy for low std events: {accuracy}"
                    )
                    # check if the model predicts the std PSI correctly
                    std_predicted_psi = low_std_events_df.groupby("event_id")[
                        "pred_psi_val"
                    ].std()
                    std_predicted_psi = std_predicted_psi.reset_index()
                    std_predicted_psi.columns = ["event_id", "std_predicted_psi"]
                    low_std_events_df = low_std_events_df.merge(
                        std_predicted_psi, on="event_id", how="inner"
                    )
                    low_std_events_df["|std_predicted_psi - std_psi_val|"] = (
                        low_std_events_df["std_predicted_psi"]
                        - low_std_events_df["std_psi_val"]
                    ).abs()
                    temp = (
                        low_std_events_df[
                            ["event_id", "|std_predicted_psi - std_psi_val|"]
                        ]
                        .drop_duplicates()
                        .reset_index(drop=True)
                    )
                    accuracy = (
                        temp["|std_predicted_psi - std_psi_val|"] < 0.01
                    ).sum() / len(temp)
                    self.log(
                        f"val/{event_type_name}_{example_type_name}_low_std_events_std_psi_accuracy",
                        accuracy,
                        on_step=False,
                        on_epoch=True,
                    )
                    print(f"Std PSI prediction accuracy for low std events: {accuracy}")
                    # check if the model correctly identifies samples with significantly different PSI values
                    # track the average percentile of the samples that are predicted to be significantly different from the mean
                    low_std_events_df["|predicted_psi_val - mean_predicted_psi|"] = (
                        low_std_events_df["pred_psi_val"]
                        - low_std_events_df["mean_predicted_psi"]
                    ).abs()
                    avg_percentile = 0.0
                    total_num_events = 0.0
                    for event_id in low_std_events_df["event_id"].unique():
                        event_df = low_std_events_df[
                            low_std_events_df["event_id"] == event_id
                        ]
                        if (event_df["sample_has_sig_lower_PSI"] | event_df["sample_has_sig_higher_PSI"]).sum() == 0:
                            continue
                        total_num_events += 1
                        event_df = event_df.sort_values(
                            "|predicted_psi_val - mean_predicted_psi|",
                            ascending=False,
                        )
                        event_df = event_df.reset_index(drop=True)
                        event_df["percentile"] = (event_df.index / len(event_df)) * 100
                        avg_percentile += event_df[
                            event_df["sample_has_sig_lower_PSI"] | event_df["sample_has_sig_higher_PSI"]
                        ]["percentile"].mean()
                    if total_num_events > 0:
                        avg_percentile /= total_num_events
                    else:
                        avg_percentile = -1.0
                    self.log(
                        f"val/{event_type_name}_{example_type_name}_low_std_events_avg_percentile_of_significant_abs_deviations",
                        avg_percentile,
                        on_step=False,
                        on_epoch=True,
                    )
                    print(
                        f"Average percentile of samples with significant deviations from the mean in low std events: {avg_percentile}"
                    )
                    # check if the model correctly predicts the direction of the deviation
                    # track the average percentile of the samples that are predicted to be significantly different from the mean
                    low_std_events_df["predicted_psi_val - mean_predicted_psi"] = (
                        low_std_events_df["pred_psi_val"] - low_std_events_df["mean_predicted_psi"]
                    )
                    avg_percentile_lower = 0.0
                    avg_percentile_higher = 0.0
                    total_num_events = 0.0
                    for event_id in low_std_events_df["event_id"].unique():
                        event_df = low_std_events_df[
                            low_std_events_df["event_id"] == event_id
                        ]
                        if (event_df["sample_has_sig_lower_PSI"] | event_df["sample_has_sig_higher_PSI"]).sum() == 0:
                            continue
                        total_num_events += 1
                        event_df = event_df.sort_values(
                            "predicted_psi_val - mean_predicted_psi",
                            ascending=False,
                        )
                        event_df = event_df.reset_index(drop=True)
                        event_df["percentile"] = (event_df.index / len(event_df)) * 100

                        avg_percentile_lower += event_df[
                            event_df["sample_has_sig_lower_PSI"]
                        ]["percentile"].mean()
                        avg_percentile_higher += event_df[
                            event_df["sample_has_sig_higher_PSI"]
                        ]["percentile"].mean()
                    if total_num_events > 0:
                        avg_percentile_lower /= total_num_events
                        avg_percentile_higher /= total_num_events
                    else:
                        avg_percentile_lower = -1.0
                        avg_percentile_higher = -1.0
                    self.log(
                        f"val/{event_type_name}_{example_type_name}_low_std_events_avg_percentile_of_significant_lower_deviations",
                        avg_percentile_lower,
                        on_step=False,
                        on_epoch=True,
                    )
                    self.log(
                        f"val/{event_type_name}_{example_type_name}_low_std_events_avg_percentile_of_significant_higher_deviations",
                        avg_percentile_higher,
                        on_step=False,
                        on_epoch=True,
                    )
                    print(
                        f"Average percentile of samples with significant lower deviations from the mean in low std events: {avg_percentile_lower}"
                    )
                    print(
                        f"Average percentile of samples with significant higher deviations from the mean in low std events: {avg_percentile_higher}"
                    )
                        

        # clear the stored predictions
        self.val_event_ids.clear()
        self.val_event_types.clear()
        self.val_example_types.clear()
        self.val_samples.clear()
        self.val_psi_vals.clear()
        self.val_pred_psi_vals.clear()
        self.val_event_ids = []
        self.val_event_types = []
        self.val_example_types = []
        self.val_samples = []
        self.val_psi_vals = []
        self.val_pred_psi_vals = []

    def predict_step(self, batch, batch_idx, dataloader_idx=0):
        if self.predict_mean_std_psi_and_delta:
            preds = self(batch)
            pred_delta_psi_val = preds[:, 0]
            pred_mean_psi_val = preds[:, 1]
            pred_std_psi_val = preds[:, 2]
            pred_psi_val = pred_mean_psi_val + (pred_delta_psi_val * pred_std_psi_val)
        else:
            pred_psi_val = self(batch)

        if "Logits" in self.config["train_config"]["loss_fn"]:
            pred_psi_val = torch.sigmoid(pred_psi_val)

        if "psi_val" in batch:
            return {
                "pred_psi_val": pred_psi_val,
                "psi_val": batch["psi_val"],
            }
        else:
            return {
                "pred_psi_val": pred_psi_val,
            }
