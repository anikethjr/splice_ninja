import numpy as np
import pandas as pd
import os
import pdb
import json
from tqdm import tqdm
from sklearn.metrics import (
    r2_score,
    accuracy_score,
    balanced_accuracy_score,
    precision_score,
    recall_score,
    f1_score,
)
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


class RankingAndBCEWithLogitsLossUsingControlDataAndWeightedLoss(nn.Module):
    def __init__(self, dPSI_threshold, margin=0, ranking_loss_weight_multiplier=10):
        super().__init__()
        self.dPSI_threshold = dPSI_threshold
        self.margin = margin
        self.ranking_loss_weight_multiplier = ranking_loss_weight_multiplier

    def forward(self, pred_psi_val, psi_val, **kwargs):
        # Compute BCEWithLogits loss
        loss = F.binary_cross_entropy_with_logits(
            pred_psi_val.view(-1, 1), psi_val.view(-1, 1)
        )

        if kwargs["use_BCE_loss_only"]:
            return loss

        event_id, sample_id = kwargs["event_id"], kwargs["sample"]

        # Compute control-based ranking loss efficiently
        unique_events, event_indices = torch.unique_consecutive(
            event_id, return_inverse=True
        )

        # Extract control PSI values for each event (assumes control sample is always sample_id == 0)
        control_mask = sample_id == 0
        control_psi = torch.zeros_like(unique_events, dtype=psi_val.dtype)
        control_pred_psi = torch.zeros_like(unique_events, dtype=pred_psi_val.dtype)

        control_psi[event_indices[control_mask]] = psi_val[control_mask]
        control_pred_psi[event_indices[control_mask]] = pred_psi_val[control_mask]

        # Compute difference from control values
        psi_diff = psi_val - control_psi[event_indices]
        pred_psi_diff = pred_psi_val - control_pred_psi[event_indices]

        # Determine valid ranking pairs
        ranking_labels = torch.sign(psi_diff)
        valid_pairs = torch.abs(psi_diff) >= self.dPSI_threshold

        # Apply margin ranking loss
        if valid_pairs.any():
            control_ranking_loss = F.margin_ranking_loss(
                pred_psi_diff[valid_pairs],
                torch.zeros_like(pred_psi_diff[valid_pairs]),
                ranking_labels[valid_pairs],
                margin=self.margin,
                reduction="none",
            )
            loss += (
                control_ranking_loss
                * torch.abs(psi_diff[valid_pairs])
                * self.ranking_loss_weight_multiplier
            ).mean()

        # Compute sample-based ranking loss efficiently
        pred_diff = pred_psi_val.unsqueeze(1) - pred_psi_val.unsqueeze(0)
        true_diff = psi_val.unsqueeze(1) - psi_val.unsqueeze(0)

        ranking_labels = torch.sign(true_diff)
        valid_pairs = torch.abs(true_diff) >= self.dPSI_threshold

        if valid_pairs.any():
            sample_ranking_loss = F.margin_ranking_loss(
                pred_diff[valid_pairs],
                torch.zeros_like(pred_diff[valid_pairs]),
                ranking_labels[valid_pairs],
                margin=self.margin,
                reduction="none",
            )
            loss += (
                sample_ranking_loss
                * torch.abs(true_diff[valid_pairs])
                * self.ranking_loss_weight_multiplier
            ).mean()

        return loss


class RankingAndBCEWithLogitsLossEventLevelRankingOnly(nn.Module):
    def __init__(self, dPSI_threshold, margin=0, ranking_loss_weight_multiplier=1):
        super().__init__()
        self.dPSI_threshold = dPSI_threshold
        self.margin = margin
        self.ranking_loss_weight_multiplier = ranking_loss_weight_multiplier

    def forward(self, pred_psi_val, psi_val, **kwargs):
        # Compute BCEWithLogits loss
        loss = F.binary_cross_entropy_with_logits(
            pred_psi_val.view(-1, 1), psi_val.view(-1, 1)
        )

        if kwargs["use_BCE_loss_only"]:
            return loss

        event_id = kwargs["event_id"]

        # Compute ranking loss efficiently
        # Compute pairwise differences for samples from the same event
        pred_diff = pred_psi_val.unsqueeze(1) - pred_psi_val.unsqueeze(0)
        true_diff = psi_val.unsqueeze(1) - psi_val.unsqueeze(0)
        pred_diff = pred_diff[event_id.unsqueeze(1) == event_id.unsqueeze(0)]
        true_diff = true_diff[event_id.unsqueeze(1) == event_id.unsqueeze(0)]
        ranking_labels = torch.sign(true_diff)
        valid_pairs = torch.abs(true_diff) >= self.dPSI_threshold
        pred_diff = pred_diff[valid_pairs]
        ranking_labels = ranking_labels[valid_pairs]

        # Apply margin ranking loss
        if valid_pairs.any():
            event_ranking_loss = F.margin_ranking_loss(
                pred_diff,
                torch.zeros_like(pred_diff),
                ranking_labels,
                margin=self.margin,
                reduction="none",
            )
            loss += (
                event_ranking_loss
                * torch.abs(true_diff[valid_pairs])
                * self.ranking_loss_weight_multiplier
            ).mean()

        return loss


class PairwiseMSELossAndBiasedBCEWithLogitsLoss(nn.Module):
    def __init__(
        self, dPSI_threshold, mse_loss_weight_multiplier=10, scaling_for_bias=1e-2
    ):
        super().__init__()
        self.dPSI_threshold = dPSI_threshold
        self.mse_loss_weight_multiplier = mse_loss_weight_multiplier
        self.scaling_for_bias = scaling_for_bias

    def forward(self, pred_psi_val, psi_val, **kwargs):
        # Compute BCEWithLogits loss
        loss = F.binary_cross_entropy_with_logits(
            pred_psi_val.view(-1, 1), psi_val.view(-1, 1), reduction="none"
        )
        # bias the loss towards intermediate PSI values
        # we want to increase the loss for PSI values that are closer to 0.5
        deviation_from_half = torch.abs(psi_val.view(-1, 1) - 0.5)
        # so we divide by ((the deviation from 0.5 + 1) * self.scaling_for_bias) to make the loss larger
        # for values closer to 0.5
        loss = loss / ((deviation_from_half + 1) * self.scaling_for_bias)
        loss = loss.mean()

        if kwargs["use_BCE_loss_only"]:
            return loss

        event_id = kwargs["event_id"]

        # Compute pairwise MSE loss efficiently
        # first, the ground truth PSI values need to be converted to logits
        # as the model outputs logits
        ori_psi_val = psi_val
        psi_val = torch.special.logit(psi_val, eps=1e-7)
        # Compute pairwise differences for samples from the same event
        pred_diff = pred_psi_val.unsqueeze(1) - pred_psi_val.unsqueeze(0)
        true_diff = psi_val.unsqueeze(1) - psi_val.unsqueeze(0)
        ori_true_diff = ori_psi_val.unsqueeze(1) - ori_psi_val.unsqueeze(0)
        mask = event_id.unsqueeze(1) == event_id.unsqueeze(0)
        pred_diff = pred_diff[mask]
        true_diff = true_diff[mask]
        ori_true_diff = ori_true_diff[mask]
        valid_pairs = torch.abs(ori_true_diff) >= self.dPSI_threshold

        pred_diff = pred_diff[valid_pairs]
        true_diff = true_diff[valid_pairs]
        # Apply MSE loss
        if valid_pairs.any():
            pairwise_mse_loss = (
                F.mse_loss(pred_diff, true_diff) * self.mse_loss_weight_multiplier
            )
            loss += pairwise_mse_loss

        return loss


class PairwiseMSELossAndBCEWithLogitsLoss(nn.Module):
    def __init__(self, dPSI_threshold, mse_loss_weight_multiplier=10):
        super().__init__()
        self.dPSI_threshold = dPSI_threshold
        self.mse_loss_weight_multiplier = mse_loss_weight_multiplier

    def forward(self, pred_psi_val, psi_val, **kwargs):
        # Compute BCEWithLogits loss
        loss = F.binary_cross_entropy_with_logits(
            pred_psi_val.view(-1, 1), psi_val.view(-1, 1)
        )

        if kwargs["use_BCE_loss_only"]:
            return loss

        event_id = kwargs["event_id"]

        # Compute pairwise MSE loss efficiently
        # first, the ground truth PSI values need to be converted to logits
        # as the model outputs logits
        ori_psi_val = psi_val
        psi_val = torch.special.logit(psi_val, eps=1e-7)
        # Compute pairwise differences for samples from the same event
        pred_diff = pred_psi_val.unsqueeze(1) - pred_psi_val.unsqueeze(0)
        true_diff = psi_val.unsqueeze(1) - psi_val.unsqueeze(0)
        ori_true_diff = ori_psi_val.unsqueeze(1) - ori_psi_val.unsqueeze(0)
        mask = event_id.unsqueeze(1) == event_id.unsqueeze(0)
        pred_diff = pred_diff[mask]
        true_diff = true_diff[mask]
        ori_true_diff = ori_true_diff[mask]
        valid_pairs = torch.abs(ori_true_diff) >= self.dPSI_threshold

        pred_diff = pred_diff[valid_pairs]
        true_diff = true_diff[valid_pairs]
        # Apply MSE loss
        if valid_pairs.any():
            pairwise_mse_loss = (
                F.mse_loss(pred_diff, true_diff) * self.mse_loss_weight_multiplier
            )
            loss += pairwise_mse_loss

        return loss


class AllExamplesPairwiseMSELossAndBCEWithLogitsLoss(nn.Module):
    def __init__(self, dPSI_threshold, mse_loss_weight_multiplier=10):
        super().__init__()
        self.dPSI_threshold = dPSI_threshold
        self.mse_loss_weight_multiplier = mse_loss_weight_multiplier

    def forward(self, pred_psi_val, psi_val, **kwargs):
        # Compute BCEWithLogits loss
        loss = F.binary_cross_entropy_with_logits(
            pred_psi_val.view(-1, 1), psi_val.view(-1, 1)
        )

        if kwargs["use_BCE_loss_only"]:
            return loss

        # Compute pairwise MSE loss efficiently
        # first, the ground truth PSI values need to be converted to logits
        # as the model outputs logits
        ori_psi_val = psi_val
        psi_val = torch.special.logit(psi_val, eps=1e-7)
        pred_diff = pred_psi_val.unsqueeze(1) - pred_psi_val.unsqueeze(0)
        true_diff = psi_val.unsqueeze(1) - psi_val.unsqueeze(0)
        ori_true_diff = ori_psi_val.unsqueeze(1) - ori_psi_val.unsqueeze(0)

        # Keep significant pairs
        pred_diff = pred_diff.flatten()
        true_diff = true_diff.flatten()
        ori_true_diff = ori_true_diff.flatten()
        valid_pairs = torch.abs(ori_true_diff) >= self.dPSI_threshold
        pred_diff = pred_diff[valid_pairs]
        true_diff = true_diff[valid_pairs]

        # Apply MSE loss
        if valid_pairs.any():
            pairwise_mse_loss = (
                F.mse_loss(pred_diff, true_diff) * self.mse_loss_weight_multiplier
            )
            loss += pairwise_mse_loss

        return loss


class AllExamplesPairwiseMSELossEventLevelRankingLossAndBCEWithLogitsLoss(nn.Module):
    def __init__(
        self,
        dPSI_threshold,
        mse_loss_weight_multiplier=1,
        margin=0,
        ranking_loss_weight_multiplier=1,
    ):
        super().__init__()
        self.dPSI_threshold = dPSI_threshold
        self.mse_loss_weight_multiplier = mse_loss_weight_multiplier
        self.margin = margin
        self.ranking_loss_weight_multiplier = ranking_loss_weight_multiplier

    def forward(self, pred_psi_val, psi_val, **kwargs):
        # Compute BCEWithLogits loss
        loss = F.binary_cross_entropy_with_logits(
            pred_psi_val.view(-1, 1), psi_val.view(-1, 1)
        )

        if kwargs["use_BCE_loss_only"]:
            return loss

        event_id = kwargs["event_id"]

        # Compute pairwise MSE loss efficiently
        # first, the ground truth PSI values need to be converted to logits
        # as the model outputs logits
        ori_psi_val = psi_val
        psi_val = torch.special.logit(psi_val, eps=1e-7)
        pred_diff = pred_psi_val.unsqueeze(1) - pred_psi_val.unsqueeze(0)
        true_diff = psi_val.unsqueeze(1) - psi_val.unsqueeze(0)
        ori_true_diff = ori_psi_val.unsqueeze(1) - ori_psi_val.unsqueeze(0)

        # Keep significant pairs
        valid_pairs = torch.abs(ori_true_diff) >= self.dPSI_threshold

        # Apply MSE loss
        if valid_pairs.any():
            pairwise_mse_loss = (
                F.mse_loss(pred_diff[valid_pairs], true_diff[valid_pairs])
                * self.mse_loss_weight_multiplier
            )
            loss += pairwise_mse_loss

        # Compute ranking loss efficiently
        # Compute pairwise differences for samples from the same event
        mask = event_id.unsqueeze(1) == event_id.unsqueeze(0)
        pred_diff = pred_diff[mask]
        true_diff = true_diff[mask]
        ori_true_diff = ori_true_diff[mask]
        ranking_labels = torch.sign(ori_true_diff)
        valid_pairs = torch.abs(ori_true_diff) >= self.dPSI_threshold
        pred_diff = pred_diff[valid_pairs]
        ranking_labels = ranking_labels[valid_pairs]

        # Apply margin ranking loss
        if valid_pairs.any():
            event_ranking_loss = F.margin_ranking_loss(
                pred_diff,
                torch.zeros_like(pred_diff),
                ranking_labels,
                margin=self.margin,
                reduction="none",
            )
            loss += (
                event_ranking_loss
                * torch.abs(ori_true_diff[valid_pairs])
                * self.ranking_loss_weight_multiplier
            ).mean()

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

        # hyperparams that affect when the ranking loss is applied
        if "num_epochs_after_which_to_use_ranking_loss" in self.config["train_config"]:
            self.num_epochs_after_which_to_use_ranking_loss = self.config[
                "train_config"
            ]["num_epochs_after_which_to_use_ranking_loss"]
        else:
            self.num_epochs_after_which_to_use_ranking_loss = 0

        # hyperparams that affect how we use the controls data + how we define significant events
        if (
            "num_epochs_for_training_on_control_data_only"
            in self.config["train_config"]
        ):
            self.num_epochs_for_training_on_control_data_only = self.config[
                "train_config"
            ]["num_epochs_for_training_on_control_data_only"]
        else:
            self.num_epochs_for_training_on_control_data_only = 0
        if "dPSI_threshold_for_significance" in self.config["train_config"]:
            self.dPSI_threshold_for_significance = self.config["train_config"][
                "dPSI_threshold_for_significance"
            ]
        else:
            self.dPSI_threshold_for_significance = 0.0

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
        if "predict_mean_psi_and_delta" not in self.config["train_config"]:
            self.config["train_config"]["predict_mean_psi_and_delta"] = False
        self.predict_mean_psi_and_delta = self.config["train_config"][
            "predict_mean_psi_and_delta"
        ]
        if "predict_controls_avg_psi_and_delta" not in self.config["train_config"]:
            self.config["train_config"]["predict_controls_avg_psi_and_delta"] = False
        self.predict_controls_avg_psi_and_delta = self.config["train_config"][
            "predict_controls_avg_psi_and_delta"
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
        elif (
            self.config["train_config"]["loss_fn"]
            == "RankingAndBCEWithLogitsLossUsingControlDataAndWeightedLoss"
        ):
            self.loss_fn = RankingAndBCEWithLogitsLossUsingControlDataAndWeightedLoss(
                self.dPSI_threshold_for_significance
            )
        elif (
            self.config["train_config"]["loss_fn"]
            == "RankingAndBCEWithLogitsLossEventLevelRankingOnly"
        ):
            self.loss_fn = RankingAndBCEWithLogitsLossEventLevelRankingOnly(
                self.dPSI_threshold_for_significance
            )
        elif (
            self.config["train_config"]["loss_fn"]
            == "PairwiseMSELossAndBiasedBCEWithLogitsLoss"
        ):
            self.loss_fn = PairwiseMSELossAndBiasedBCEWithLogitsLoss(
                self.dPSI_threshold_for_significance
            )
        elif (
            self.config["train_config"]["loss_fn"]
            == "PairwiseMSELossAndBCEWithLogitsLoss"
        ):
            self.loss_fn = PairwiseMSELossAndBCEWithLogitsLoss(
                self.dPSI_threshold_for_significance
            )
        elif (
            self.config["train_config"]["loss_fn"]
            == "AllExamplesPairwiseMSELossAndBCEWithLogitsLoss"
        ):
            self.loss_fn = AllExamplesPairwiseMSELossAndBCEWithLogitsLoss(
                self.dPSI_threshold_for_significance
            )
        elif (
            self.config["train_config"]["loss_fn"]
            == "AllExamplesPairwiseMSELossEventLevelRankingLossAndBCEWithLogitsLoss"
        ):
            self.loss_fn = (
                AllExamplesPairwiseMSELossEventLevelRankingLossAndBCEWithLogitsLoss(
                    self.dPSI_threshold_for_significance
                )
            )
        else:
            raise ValueError(
                f"Loss function {self.config['train_config']['loss_fn']} not found."
            )

        if self.predict_mean_std_psi_and_delta:
            if "Logits" not in self.config["train_config"]["loss_fn"]:
                self.mean_delta_psi_loss_fn = MSELoss()
            else:
                self.mean_delta_psi_loss_fn = BCEWithLogitsLoss()
        if self.predict_mean_psi_and_delta:
            if "Logits" not in self.config["train_config"]["loss_fn"]:
                self.mean_psi_loss_fn = MSELoss()
            else:
                self.mean_psi_loss_fn = BCEWithLogitsLoss()
        if self.predict_controls_avg_psi_and_delta:
            if "Logits" not in self.config["train_config"]["loss_fn"]:
                self.controls_avg_psi_loss_fn = MSELoss()
            else:
                self.controls_avg_psi_loss_fn = BCEWithLogitsLoss()

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
        self.val_controls_avg_psi = []
        self.val_num_controls = []

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

        # create a directory to save the model metrics
        self.current_val_metrics = {}

        # needed to control metric computation
        if "dataset_name" not in self.config["data_config"]:
            self.reliant_on_controls = True
        elif self.config["data_config"]["dataset_name"] == "KD":
            self.reliant_on_controls = True
        elif self.config["data_config"]["dataset_name"] == "VastDB+KD":
            self.reliant_on_controls = False

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
        elif self.predict_mean_psi_and_delta:
            preds = self(batch)
            pred_delta_psi_val = preds[:, 0]
            pred_mean_psi_val = preds[:, 1]
            pred_psi_val = pred_mean_psi_val + pred_delta_psi_val
        elif self.predict_controls_avg_psi_and_delta:
            preds = self(batch)
            pred_delta_psi_val = preds[:, 0]
            pred_controls_avg_psi_val = preds[:, 1]
            pred_psi_val = pred_controls_avg_psi_val + pred_delta_psi_val
        else:
            pred_psi_val = self(batch)

        loss = self.loss_fn(
            pred_psi_val,
            batch["psi_val"],
            use_BCE_loss_only=(
                self.current_epoch < self.num_epochs_for_training_on_control_data_only
            )
            or (self.current_epoch < self.num_epochs_after_which_to_use_ranking_loss),
            event_num_samples_observed=batch["event_num_samples_observed"],
            event_mean_psi=batch["event_mean_psi"],
            event_std_psi=batch["event_std_psi"],
            event_min_psi=batch["event_min_psi"],
            event_max_psi=batch["event_max_psi"],
            event_id=batch["event_id"],
            sample=batch["sample"],
            event_controls_avg_psi=batch["event_controls_avg_psi"],
            event_num_controls=batch["event_num_controls"],
        )
        if self.predict_mean_std_psi_and_delta:
            self.log("train/psi_val_loss", loss, on_step=True, on_epoch=True)
            mean_psi_loss = self.mean_delta_psi_loss_fn(
                pred_mean_psi_val, batch["event_mean_psi"]
            )
            std_psi_loss = self.mean_delta_psi_loss_fn(
                pred_std_psi_val, batch["event_std_psi"]
            )
            loss = loss + mean_psi_loss + std_psi_loss
            self.log("train/mean_psi_loss", mean_psi_loss, on_step=True, on_epoch=True)
            self.log("train/std_psi_loss", std_psi_loss, on_step=True, on_epoch=True)
        if self.predict_mean_psi_and_delta:
            self.log("train/psi_val_loss", loss, on_step=True, on_epoch=True)
            mean_psi_loss = self.mean_psi_loss_fn(
                pred_mean_psi_val, batch["event_mean_psi"]
            )
            loss = loss + mean_psi_loss
            self.log("train/mean_psi_loss", mean_psi_loss, on_step=True, on_epoch=True)
        if self.predict_controls_avg_psi_and_delta:
            self.log("train/psi_val_loss", loss, on_step=True, on_epoch=True)
            controls_avg_psi_loss = self.controls_avg_psi_loss_fn(
                pred_controls_avg_psi_val, batch["event_controls_avg_psi"]
            )
            loss = loss + controls_avg_psi_loss
            self.log(
                "train/controls_avg_psi_loss",
                controls_avg_psi_loss,
                on_step=True,
                on_epoch=True,
            )

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
        elif self.predict_mean_psi_and_delta:
            preds = self(batch)
            pred_delta_psi_val = preds[:, 0]
            pred_mean_psi_val = preds[:, 1]
            pred_psi_val = pred_mean_psi_val + pred_delta_psi_val
        elif self.predict_controls_avg_psi_and_delta:
            preds = self(batch)
            pred_delta_psi_val = preds[:, 0]
            pred_controls_avg_psi_val = preds[:, 1]
            pred_psi_val = pred_controls_avg_psi_val + pred_delta_psi_val
        else:
            pred_psi_val = self(batch)

        loss = self.loss_fn(
            pred_psi_val,
            batch["psi_val"],
            use_BCE_loss_only=True,
            event_num_samples_observed=batch["event_num_samples_observed"],
            event_mean_psi=batch["event_mean_psi"],
            event_std_psi=batch["event_std_psi"],
            event_min_psi=batch["event_min_psi"],
            event_max_psi=batch["event_max_psi"],
            event_id=batch["event_id"],
            sample=batch["sample"],
            event_controls_avg_psi=batch["event_controls_avg_psi"],
            event_num_controls=batch["event_num_controls"],
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
        if self.predict_mean_psi_and_delta:
            self.log(
                "val/psi_val_loss", loss, on_step=False, on_epoch=True, sync_dist=True
            )
            mean_psi_loss = self.mean_psi_loss_fn(
                pred_mean_psi_val, batch["event_mean_psi"]
            )
            loss = loss + mean_psi_loss
            self.log(
                "val/mean_psi_loss",
                mean_psi_loss,
                on_step=False,
                on_epoch=True,
                sync_dist=True,
            )
        if self.predict_controls_avg_psi_and_delta:
            self.log(
                "val/psi_val_loss", loss, on_step=False, on_epoch=True, sync_dist=True
            )
            controls_avg_psi_loss = self.controls_avg_psi_loss_fn(
                pred_controls_avg_psi_val, batch["event_controls_avg_psi"]
            )
            loss = loss + controls_avg_psi_loss
            self.log(
                "val/controls_avg_psi_loss",
                controls_avg_psi_loss,
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
        self.val_controls_avg_psi.extend(batch["event_controls_avg_psi"].detach().cpu())
        self.val_num_controls.extend(batch["event_num_controls"].detach().cpu())

        return loss

    def on_validation_epoch_end(self):
        """
        Gather predictions across all GPUs and compute the final metrics.
        """
        print("Computing final validation metrics...")
        # convert lists to torch tensors
        val_event_ids = torch.tensor(self.val_event_ids, dtype=torch.float32)
        val_event_types = torch.tensor(self.val_event_types, dtype=torch.float32)
        val_example_types = torch.tensor(self.val_example_types, dtype=torch.float32)
        val_samples = torch.tensor(self.val_samples, dtype=torch.float32)
        val_psi_vals = torch.tensor(self.val_psi_vals, dtype=torch.float32)
        val_pred_psi_vals = torch.tensor(self.val_pred_psi_vals, dtype=torch.float32)
        val_controls_avg_psi = torch.tensor(
            self.val_controls_avg_psi, dtype=torch.float32
        )
        val_num_controls = torch.tensor(self.val_num_controls, dtype=torch.float32)

        # convert all nans to -1
        val_controls_avg_psi[val_controls_avg_psi.isnan()] = -1
        val_num_controls[val_num_controls.isnan()] = -1

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
        val_controls_avg_psi = F.pad(
            val_controls_avg_psi, (0, max_size - local_size), value=pad_value
        )
        val_num_controls = F.pad(
            val_num_controls, (0, max_size - local_size), value=pad_value
        )

        # gather all predictions across all processes
        val_event_ids = self.all_gather(val_event_ids)
        val_event_types = self.all_gather(val_event_types)
        val_example_types = self.all_gather(val_example_types)
        val_samples = self.all_gather(val_samples)
        val_psi_vals = self.all_gather(val_psi_vals)
        val_pred_psi_vals = self.all_gather(val_pred_psi_vals)
        val_controls_avg_psi = self.all_gather(val_controls_avg_psi)
        val_num_controls = self.all_gather(val_num_controls)

        # Only compute metrics on rank 0
        if self.global_rank == 0:
            # flatten all tensors
            val_event_ids = val_event_ids.view(-1)
            val_event_types = val_event_types.view(-1)
            val_example_types = val_example_types.view(-1)
            val_samples = val_samples.view(-1)
            val_psi_vals = val_psi_vals.view(-1)
            val_pred_psi_vals = val_pred_psi_vals.view(-1)
            val_controls_avg_psi = val_controls_avg_psi.view(-1)
            val_num_controls = val_num_controls.view(-1)

            # remove padding and convert to numpy
            val_event_ids = (
                val_event_ids[~torch.isnan(val_event_ids)]
                .cpu()
                .numpy()
                .astype(np.int32)
            )
            val_event_types = (
                val_event_types[~torch.isnan(val_event_types)].cpu().numpy()
            ).astype(np.int32)
            val_example_types = (
                val_example_types[~torch.isnan(val_example_types)].cpu().numpy()
            ).astype(np.int32)
            val_samples = (
                val_samples[~torch.isnan(val_samples)].cpu().numpy().astype(np.int32)
            )
            val_psi_vals = val_psi_vals[~torch.isnan(val_psi_vals)].cpu().numpy()
            val_pred_psi_vals = (
                val_pred_psi_vals[~torch.isnan(val_pred_psi_vals)].cpu().numpy()
            )
            val_controls_avg_psi = (
                val_controls_avg_psi[~torch.isnan(val_controls_avg_psi)].cpu().numpy()
            )
            val_num_controls = (
                val_num_controls[~torch.isnan(val_num_controls)].cpu().numpy()
            ).astype(np.int32)

            # create a dataframe to store all predictions
            preds_df = pd.DataFrame(
                {
                    "event_id": val_event_ids,
                    "event_type": val_event_types,
                    "example_type": val_example_types,
                    "sample": val_samples,
                    "psi_val": val_psi_vals,
                    "pred_psi_val": val_pred_psi_vals,
                    "event_controls_avg_psi": val_controls_avg_psi,
                    "event_num_controls": val_num_controls,
                }
            )

            # drop duplicates that might have been created to have the same number of samples across all processes
            preds_df = preds_df.drop_duplicates().reset_index(drop=True)
            if self.reliant_on_controls:
                # add the predicted control PSI values - the control sample is always sample 0
                control_preds = preds_df[preds_df["sample"] == 0].reset_index(drop=True)
                control_preds = control_preds.rename(
                    columns={"pred_psi_val": "pred_event_controls_avg_psi"}
                )
                assert np.all(
                    control_preds["psi_val"] == control_preds["event_controls_avg_psi"]
                ), "Control PSI values do not match the average control PSI values."
                ori_num_examples = preds_df.shape[0]
                preds_df = preds_df.merge(
                    control_preds[["event_id", "pred_event_controls_avg_psi"]],
                    on="event_id",
                    how="inner",
                ).reset_index(drop=True)
                preds_df = preds_df.sort_values(
                    by=["event_id", "event_type", "example_type", "sample"]
                ).reset_index(drop=True)

                if preds_df.shape[0] != ori_num_examples:
                    print(
                        "Likely that this is a sanity check, not saving predictions and skipping metrics computation."
                    )
                    # clear the stored predictions
                    self.val_event_ids.clear()
                    self.val_event_types.clear()
                    self.val_example_types.clear()
                    self.val_samples.clear()
                    self.val_psi_vals.clear()
                    self.val_pred_psi_vals.clear()
                    self.val_controls_avg_psi.clear()
                    self.val_num_controls.clear()
                    self.val_event_ids = []
                    self.val_event_types = []
                    self.val_example_types = []
                    self.val_samples = []
                    self.val_psi_vals = []
                    self.val_pred_psi_vals = []
                    self.val_controls_avg_psi = []
                    self.val_num_controls = []
                    return

                else:
                    # save the predictions to a csv file
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
            else:
                preds_df = preds_df.sort_values(
                    by=["event_id", "event_type", "example_type", "sample"]
                ).reset_index(drop=True)
                # save the predictions to a csv file
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

                    # compute overall correlation metrics
                    this_spearmanR = spearmanr(
                        subset_df["psi_val"], subset_df["pred_psi_val"]
                    )[0]
                    this_pearsonR = pearsonr(
                        subset_df["psi_val"], subset_df["pred_psi_val"]
                    )[0]
                    this_r2 = r2_score(subset_df["psi_val"], subset_df["pred_psi_val"])
                    self.log(
                        f"val/{event_type_name}_{example_type_name}_examples_spearmanR",
                        this_spearmanR,
                        on_step=False,
                        on_epoch=True,
                    )
                    self.log(
                        f"val/{event_type_name}_{example_type_name}_examples_pearsonR",
                        this_pearsonR,
                        on_step=False,
                        on_epoch=True,
                    )
                    self.log(
                        f"val/{event_type_name}_{example_type_name}_examples_r2",
                        this_r2,
                        on_step=False,
                        on_epoch=True,
                    )
                    print(
                        f"Overall SpearmanR between predicted PSI and ground truth: {this_spearmanR}"
                    )
                    print(
                        f"Overall PearsonR between predicted PSI and ground truth: {this_pearsonR}"
                    )
                    print(
                        f"Overall R2 score between predicted PSI and ground truth: {this_r2}"
                    )
                    print(
                        f"Number of examples: {subset_df.shape[0]}, Number of events: {subset_df['event_id'].nunique()}"
                    )

                    if self.reliant_on_controls:
                        # first compute the correlation between predicted and ground truth PSI values in the average control
                        # sample 0 is the control sample
                        control_df = subset_df[subset_df["sample"] == 0].reset_index(
                            drop=True
                        )
                        control_spearmanR = spearmanr(
                            control_df["psi_val"], control_df["pred_psi_val"]
                        )[0]
                        control_pearsonR = pearsonr(
                            control_df["psi_val"], control_df["pred_psi_val"]
                        )[0]
                        control_r2 = r2_score(
                            control_df["psi_val"], control_df["pred_psi_val"]
                        )
                        self.log(
                            f"val/control_{event_type_name}_{example_type_name}_examples_spearmanR",
                            control_spearmanR,
                            on_step=False,
                            on_epoch=True,
                        )
                        self.log(
                            f"val/control_{event_type_name}_{example_type_name}_examples_pearsonR",
                            control_pearsonR,
                            on_step=False,
                            on_epoch=True,
                        )
                        self.log(
                            f"val/control_{event_type_name}_{example_type_name}_examples_r2",
                            control_r2,
                            on_step=False,
                            on_epoch=True,
                        )

                        # if we only have control samples, we skip the rest of the metrics
                        if (subset_df["sample"] != 0).sum() == 0:
                            print(
                                f"Only control samples for event type: {event_type}, example type: {example_type}. Skipping rest of the metrics."
                            )
                            continue

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
                        event_df = subset_df[subset_df["event_id"] == event_id]
                        std_across_samples.append(
                            np.std(event_df["psi_val"].values, ddof=1)
                        )
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

                    if not self.reliant_on_controls:
                        print("Skipping the rest of the metrics for VastDB+KD dataset")
                        continue

                    # next, we compute a similar correlation as above but only using samples that deviate from the control sample by at least 0.15
                    # we want at least 10 samples to compute the correlation
                    sample_wise_spearmanR = []
                    sample_wise_pearsonR = []
                    sample_wise_r2 = []
                    for event_id in tqdm(subset_df["event_id"].unique()):
                        event_df = subset_df[subset_df["event_id"] == event_id]
                        # get samples that deviate from the control sample by at least 0.15
                        event_df = event_df[
                            (
                                (
                                    event_df["psi_val"]
                                    - event_df["event_controls_avg_psi"]
                                ).abs()
                            )
                            >= 0.15
                        ].reset_index(drop=True)
                        if len(event_df) < 10:
                            continue
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
                        f"val/num_events_with_at_least_10_significant_perturbations_{event_type_name}_{example_type_name}_examples",
                        len(sample_wise_spearmanR),
                        on_step=False,
                        on_epoch=True,
                    )
                    self.log(
                        f"val/avg_{event_type_name}_{example_type_name}_examples_significant_perturbations_sample_wise_spearmanR",
                        avg_sample_wise_spearmanR,
                        on_step=False,
                        on_epoch=True,
                    )
                    self.log(
                        f"val/avg_{event_type_name}_{example_type_name}_examples_significant_perturbations_sample_wise_pearsonR",
                        avg_sample_wise_pearsonR,
                        on_step=False,
                        on_epoch=True,
                    )
                    self.log(
                        f"val/avg_{event_type_name}_{example_type_name}_examples_significant_perturbations_sample_wise_r2",
                        avg_sample_wise_r2,
                        on_step=False,
                        on_epoch=True,
                    )
                    print(
                        f"Number of events with at least 10 samples deviating from the control sample by at least 0.15: {len(sample_wise_spearmanR)}"
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

                    # classification metrics to determine whether the model can predict when PSI deviates significantly from the controls in some samples
                    # we also want to see whether the model can detect which events are unaffected by perturbations

                    subset_df["sample_has_sig_lower_PSI_than_control"] = (
                        subset_df["psi_val"] - subset_df["event_controls_avg_psi"]
                    ) < -0.15
                    subset_df["sample_has_sig_higher_PSI_than_control"] = (
                        subset_df["psi_val"] - subset_df["event_controls_avg_psi"]
                    ) > 0.15
                    subset_df["sample_has_sig_different_PSI_than_control"] = (
                        subset_df["psi_val"] - subset_df["event_controls_avg_psi"]
                    ).abs() > 0.15
                    subset_df["sample_has_sig_lower_predicted_PSI_than_control"] = (
                        subset_df["pred_psi_val"]
                        - subset_df["pred_event_controls_avg_psi"]
                    ) < -0.15
                    subset_df["sample_has_sig_higher_predicted_PSI_than_control"] = (
                        subset_df["pred_psi_val"]
                        - subset_df["pred_event_controls_avg_psi"]
                    ) > 0.15
                    subset_df["sample_has_sig_different_predicted_PSI_than_control"] = (
                        subset_df["pred_psi_val"]
                        - subset_df["pred_event_controls_avg_psi"]
                    ).abs() > 0.15

                    # compute metrics
                    # 1. predicting samples with significantly different PSI than the control sample
                    subset_df["label_sig_different_PSI_than_control"] = 0
                    subset_df["predicted_label_sig_different_PSI_than_control"] = 0
                    subset_df.loc[
                        subset_df["sample_has_sig_different_PSI_than_control"],
                        "label_sig_different_PSI_than_control",
                    ] = 1
                    subset_df.loc[
                        subset_df[
                            "sample_has_sig_different_predicted_PSI_than_control"
                        ],
                        "predicted_label_sig_different_PSI_than_control",
                    ] = 1
                    accuracy = accuracy_score(
                        y_true=subset_df["label_sig_different_PSI_than_control"],
                        y_pred=subset_df[
                            "predicted_label_sig_different_PSI_than_control"
                        ],
                    )
                    adjusted_balanced_accuracy = balanced_accuracy_score(
                        y_true=subset_df["label_sig_different_PSI_than_control"],
                        y_pred=subset_df[
                            "predicted_label_sig_different_PSI_than_control"
                        ],
                        adjusted=True,
                    )
                    precision = precision_score(
                        y_true=subset_df["label_sig_different_PSI_than_control"],
                        y_pred=subset_df[
                            "predicted_label_sig_different_PSI_than_control"
                        ],
                    )
                    recall = recall_score(
                        y_true=subset_df["label_sig_different_PSI_than_control"],
                        y_pred=subset_df[
                            "predicted_label_sig_different_PSI_than_control"
                        ],
                    )
                    f1 = f1_score(
                        y_true=subset_df["label_sig_different_PSI_than_control"],
                        y_pred=subset_df[
                            "predicted_label_sig_different_PSI_than_control"
                        ],
                    )
                    self.log(
                        f"val/{event_type_name}_{example_type_name}_examples_sig_different_PSI_accuracy",
                        accuracy,
                        on_step=False,
                        on_epoch=True,
                    )
                    self.log(
                        f"val/{event_type_name}_{example_type_name}_examples_sig_different_PSI_adjusted_balanced_accuracy",
                        adjusted_balanced_accuracy,
                        on_step=False,
                        on_epoch=True,
                    )
                    self.log(
                        f"val/{event_type_name}_{example_type_name}_examples_sig_different_PSI_precision",
                        precision,
                        on_step=False,
                        on_epoch=True,
                    )
                    self.log(
                        f"val/{event_type_name}_{example_type_name}_examples_sig_different_PSI_recall",
                        recall,
                        on_step=False,
                        on_epoch=True,
                    )
                    self.log(
                        f"val/{event_type_name}_{example_type_name}_examples_sig_different_PSI_f1",
                        f1,
                        on_step=False,
                        on_epoch=True,
                    )
                    print(
                        f"Accuracy of predicting samples with significantly different PSI than the control sample: {accuracy}"
                    )
                    print(
                        f"Adjusted balanced accuracy of predicting samples with significantly different PSI than the control sample: {adjusted_balanced_accuracy}"
                    )
                    print(
                        f"Precision of predicting samples with significantly different PSI than the control sample: {precision}"
                    )
                    print(
                        f"Recall of predicting samples with significantly different PSI than the control sample: {recall}"
                    )
                    print(
                        f"F1 score of predicting samples with significantly different PSI than the control sample: {f1}"
                    )

                    # 2. predicting direction of deviation
                    subset_df["label_sig_different_PSI_direction"] = 0
                    subset_df["predicted_label_sig_different_PSI_direction"] = 0
                    subset_df.loc[
                        subset_df["sample_has_sig_lower_PSI_than_control"],
                        "label_sig_different_PSI_direction",
                    ] = 1
                    subset_df.loc[
                        subset_df["sample_has_sig_higher_PSI_than_control"],
                        "label_sig_different_PSI_direction",
                    ] = 2
                    subset_df.loc[
                        subset_df["sample_has_sig_lower_predicted_PSI_than_control"],
                        "predicted_label_sig_different_PSI_direction",
                    ] = 1
                    subset_df.loc[
                        subset_df["sample_has_sig_higher_predicted_PSI_than_control"],
                        "predicted_label_sig_different_PSI_direction",
                    ] = 2
                    accuracy = accuracy_score(
                        y_true=subset_df["label_sig_different_PSI_direction"],
                        y_pred=subset_df["predicted_label_sig_different_PSI_direction"],
                    )
                    adjusted_balanced_accuracy = balanced_accuracy_score(
                        y_true=subset_df["label_sig_different_PSI_direction"],
                        y_pred=subset_df["predicted_label_sig_different_PSI_direction"],
                        adjusted=True,
                    )
                    self.log(
                        f"val/{event_type_name}_{example_type_name}_examples_sig_different_PSI_direction_accuracy",
                        accuracy,
                        on_step=False,
                        on_epoch=True,
                    )
                    self.log(
                        f"val/{event_type_name}_{example_type_name}_examples_sig_different_PSI_direction_adjusted_balanced_accuracy",
                        adjusted_balanced_accuracy,
                        on_step=False,
                        on_epoch=True,
                    )
                    print(
                        f"Accuracy of predicting direction of deviation from the control sample: {accuracy}"
                    )
                    print(
                        f"Adjusted balanced accuracy of predicting direction of deviation from the control sample: {adjusted_balanced_accuracy}"
                    )

                    # finally, compute average percentiles of samples that have significantly lower or higher PSI than the control sample
                    # percentiles are computed using the predicted PSI values
                    # percentiles are computed across all samples for each event
                    average_percentile_of_samples_with_sig_lower_PSI = []
                    average_percentile_of_samples_with_sig_higher_PSI = []
                    for event_id in subset_df["event_id"].unique():
                        event_df = subset_df[subset_df["event_id"] == event_id]
                        # remove control sample
                        event_df = event_df[(event_df["sample"] != 0)].reset_index(
                            drop=True
                        )
                        # only compute percentiles if there are samples with significantly lower or higher PSI than the control sample
                        if (
                            event_df["sample_has_sig_lower_PSI_than_control"].sum() > 0
                        ) or (
                            event_df["sample_has_sig_higher_PSI_than_control"].sum() > 0
                        ):
                            # compute percentiles using the predicted PSI values
                            # rank the predicted PSI values
                            event_df["percentile"] = event_df["pred_psi_val"].rank(
                                pct=True
                            )
                            if (
                                event_df["sample_has_sig_lower_PSI_than_control"].sum()
                                > 0
                            ):
                                average_percentile_of_samples_with_sig_lower_PSI.append(
                                    event_df[
                                        event_df[
                                            "sample_has_sig_lower_PSI_than_control"
                                        ]
                                    ]["percentile"].mean()
                                )
                            if (
                                event_df["sample_has_sig_higher_PSI_than_control"].sum()
                                > 0
                            ):
                                average_percentile_of_samples_with_sig_higher_PSI.append(
                                    event_df[
                                        event_df[
                                            "sample_has_sig_higher_PSI_than_control"
                                        ]
                                    ]["percentile"].mean()
                                )

                    average_percentile_of_samples_with_sig_lower_PSI = np.array(
                        average_percentile_of_samples_with_sig_lower_PSI
                    )
                    average_percentile_of_samples_with_sig_higher_PSI = np.array(
                        average_percentile_of_samples_with_sig_higher_PSI
                    )
                    avg_percentile_of_samples_with_sig_lower_PSI = np.mean(
                        average_percentile_of_samples_with_sig_lower_PSI
                    )
                    avg_percentile_of_samples_with_sig_higher_PSI = np.mean(
                        average_percentile_of_samples_with_sig_higher_PSI
                    )
                    self.log(
                        f"val/avg_percentile_of_samples_with_sig_lower_PSI_{event_type_name}_{example_type_name}_examples",
                        avg_percentile_of_samples_with_sig_lower_PSI,
                        on_step=False,
                        on_epoch=True,
                    )
                    self.log(
                        f"val/avg_percentile_of_samples_with_sig_higher_PSI_{event_type_name}_{example_type_name}_examples",
                        avg_percentile_of_samples_with_sig_higher_PSI,
                        on_step=False,
                        on_epoch=True,
                    )
                    print(
                        f"Average percentile of samples with significantly lower PSI than the control sample: {avg_percentile_of_samples_with_sig_lower_PSI}"
                    )
                    print(
                        f"Average percentile of samples with significantly higher PSI than the control sample: {avg_percentile_of_samples_with_sig_higher_PSI}"
                    )

        # clear the stored predictions
        self.val_event_ids.clear()
        self.val_event_types.clear()
        self.val_example_types.clear()
        self.val_samples.clear()
        self.val_psi_vals.clear()
        self.val_pred_psi_vals.clear()
        self.val_controls_avg_psi.clear()
        self.val_num_controls.clear()
        self.val_event_ids = []
        self.val_event_types = []
        self.val_example_types = []
        self.val_samples = []
        self.val_psi_vals = []
        self.val_pred_psi_vals = []
        self.val_controls_avg_psi = []
        self.val_num_controls = []

        # store all current epoch validation metrics
        self.current_val_metrics = {
            "epoch": self.current_epoch,
        }
        for key, value in self.trainer.callback_metrics.items():
            self.current_val_metrics[key] = value.cpu().numpy()
        print("All current val metrics:", self.current_val_metrics)

    def on_save_checkpoint(self, checkpoint):
        checkpoint["val_metrics"] = self.current_val_metrics

    def predict_step(self, batch, batch_idx, dataloader_idx=0):
        if self.predict_mean_std_psi_and_delta:
            preds = self(batch)
            pred_delta_psi_val = preds[:, 0]
            pred_mean_psi_val = preds[:, 1]
            pred_std_psi_val = preds[:, 2]
            pred_psi_val = pred_mean_psi_val + (pred_delta_psi_val * pred_std_psi_val)
        elif self.predict_mean_psi_and_delta:
            preds = self(batch)
            pred_delta_psi_val = preds[:, 0]
            pred_mean_psi_val = preds[:, 1]
            pred_psi_val = pred_mean_psi_val + pred_delta_psi_val
        elif self.predict_controls_avg_psi_and_delta:
            preds = self(batch)
            pred_delta_psi_val = preds[:, 0]
            pred_controls_avg_psi_val = preds[:, 1]
            pred_psi_val = pred_controls_avg_psi_val + pred_delta_psi_val
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
