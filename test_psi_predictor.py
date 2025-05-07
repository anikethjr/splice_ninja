import numpy as np
import pandas as pd
import os
import pdb
import json
from argparse import ArgumentParser, BooleanOptionalAction
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
from lightning import Trainer
from lightning.pytorch.callbacks import BasePredictionWriter

from splice_ninja.dataloaders.knockdown_data import KnockdownData
from splice_ninja.models.psi_predictor import PSIPredictor

np.random.seed(0)
torch.manual_seed(0)
torch.set_float32_matmul_precision("medium")


class CustomWriter(BasePredictionWriter):
    def __init__(self, output_dir, write_interval):
        super().__init__(write_interval)
        self.output_dir = output_dir

    def write_on_epoch_end(self, trainer, pl_module, predictions, batch_indices):
        # this will create N (num processes) files in `output_dir` each containing
        # the predictions of it's respective rank
        torch.save(
            predictions,
            os.path.join(self.output_dir, f"predictions_{trainer.global_rank}.pt"),
        )

        # optionally, you can also save `batch_indices` to get the information about the data index
        # from your prediction data
        torch.save(
            batch_indices,
            os.path.join(self.output_dir, f"batch_indices_{trainer.global_rank}.pt"),
        )


def find_best_checkpoint(
    checkpoint_dir,
    metric_name,
    optimal_direction="max",
    create_best_ckpt_copy=False,
):
    """
    Find the best checkpoint in the directory.
    Args:
        checkpoint_dir: Directory containing the checkpoints.
        metric_name: Name of the metric to use for finding the best checkpoint.
        optimal_direction: Direction of the metric to use for finding the best checkpoint. Can be "max" or "min".
        create_best_ckpt_copy: If True, the function will create a copy of the best checkpoint in the same directory with the name "best.ckpt".
    """
    assert optimal_direction in [
        "max",
        "min",
    ], f"Invalid optimal direction: {optimal_direction}. Must be 'max' or 'min'."

    # find the best checkpoint
    file_name_to_metric = {}
    file_name_to_epoch = {}
    for f in os.listdir(checkpoint_dir):
        if not f.endswith(".ckpt"):
            continue

        if f == "best.ckpt":
            print(
                "WARNING: Found a file named 'best.ckpt' in the directory. Skipping it. It will be overwritten if create_best_ckpt_copy is set to True."
            )
            continue

        # load the ckpt file to get the epoch number and the metric
        ckpt = torch.load(os.path.join(checkpoint_dir, f), map_location="cpu")

        if metric_name == "epoch":
            # if the metric is epoch, just use the epoch number
            file_name_to_metric[f] = ckpt["epoch"]
            file_name_to_epoch[f] = ckpt["epoch"]
            continue

        val_metrics = ckpt["val_metrics"]
        if metric_name not in val_metrics.keys():
            print(
                f"WARNING: No metric named {metric_name} found in the checkpoint file {f}. Skipping it."
            )
            continue
        file_name_to_epoch[f] = ckpt["epoch"]
        file_name_to_metric[f] = val_metrics[metric_name]

    if len(file_name_to_epoch) is None:
        raise ValueError("No valid checkpoints found in the directory.")

    best_ckpt_path = None
    best_metric = None
    best_epoch = None
    for f, metric in file_name_to_metric.items():
        epoch = file_name_to_epoch[f]
        if best_metric is None:
            best_metric = metric
            best_ckpt_path = f
            best_epoch = epoch
        else:
            if optimal_direction == "max":
                if metric > best_metric:
                    best_metric = metric
                    best_ckpt_path = f
                    best_epoch = epoch
                if (
                    metric == best_metric
                ):  # if the metric is the same, prefer the one with the higher epoch number
                    if epoch > best_epoch:
                        best_ckpt_path = f
                        best_epoch = epoch
            else:
                if metric < best_metric:
                    best_metric = metric
                    best_ckpt_path = f
                    best_epoch = epoch
                if (
                    metric == best_metric
                ):  # if the metric is the same, prefer the one with the higher epoch number
                    if epoch > best_epoch:
                        best_ckpt_path = f
                        best_epoch = epoch

    best_ckpt_path = os.path.join(checkpoint_dir, best_ckpt_path)

    # create a copy of the best checkpoint
    if create_best_ckpt_copy:
        best_ckpt_copy_path = os.path.join(checkpoint_dir, "best.ckpt")
        os.system(f"cp {best_ckpt_path} {best_ckpt_copy_path}")
        print(f"Created a copy of the best checkpoint at {best_ckpt_copy_path}")

    return os.path.join(checkpoint_dir, best_ckpt_path)


def compute_and_save_validation_metrics(preds_df, summary_save_path):
    """
    Compute validation metrics and save them to a results file.
    """
    metrics = {}

    # compute metrics for all event types and example types
    unique_event_types = preds_df["EVENT_TYPE"].unique().tolist()
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
            event_type_df = preds_df[preds_df["EVENT_TYPE"] == event_type].reset_index(
                drop=True
            )
            event_type_name = f"event_type_{event_type}"

        for example_type in unique_example_types:
            if example_type == "ALL":
                subset_df = event_type_df
                example_type_name = "ALL"
            else:
                subset_df = event_type_df[
                    event_type_df["example_type"] == example_type
                ].reset_index(drop=True)
                example_type_name = f"example_type_{example_type}"

            # compute correlation metrics for control samples
            control_df = subset_df[subset_df["SAMPLE"] == 0].reset_index(drop=True)
            control_spearmanR = spearmanr(control_df["PSI"], control_df["PSI_PREDS"])[0]
            control_pearsonR = pearsonr(control_df["PSI"], control_df["PSI_PREDS"])[0]
            control_r2 = r2_score(control_df["PSI"], control_df["PSI_PREDS"])

            metrics[
                f"control_{event_type_name}_{example_type_name}_spearmanR"
            ] = control_spearmanR
            metrics[
                f"control_{event_type_name}_{example_type_name}_pearsonR"
            ] = control_pearsonR
            metrics[f"control_{event_type_name}_{example_type_name}_r2"] = control_r2

            # if we only have control samples, skip the rest of the metrics
            if (subset_df["SAMPLE"] != 0).sum() == 0:
                continue

            # compute sample-wise correlation metrics for events observed in at least 10 samples
            sample_counts = subset_df["EVENT"].value_counts()
            sample_counts = sample_counts[sample_counts >= 10]
            sample_counts = sample_counts.index

            sample_wise_spearmanR = []
            sample_wise_pearsonR = []
            sample_wise_r2 = []

            for event_id in tqdm(sample_counts):
                event_df = subset_df[subset_df["EVENT"] == event_id]
                spearmanR = np.nan_to_num(
                    spearmanr(event_df["PSI"], event_df["PSI_PREDS"])[0]
                )
                pearsonR = np.nan_to_num(
                    pearsonr(event_df["PSI"], event_df["PSI_PREDS"])[0]
                )
                r2 = np.nan_to_num(r2_score(event_df["PSI"], event_df["PSI_PREDS"]))

                sample_wise_spearmanR.append(spearmanR)
                sample_wise_pearsonR.append(pearsonR)
                sample_wise_r2.append(r2)

            sample_wise_spearmanR = np.array(sample_wise_spearmanR)
            sample_wise_pearsonR = np.array(sample_wise_pearsonR)
            sample_wise_r2 = np.array(sample_wise_r2)

            metrics[
                f"avg_{event_type_name}_{example_type_name}_sample_wise_spearmanR"
            ] = np.mean(sample_wise_spearmanR)
            metrics[
                f"avg_{event_type_name}_{example_type_name}_sample_wise_pearsonR"
            ] = np.mean(sample_wise_pearsonR)
            metrics[
                f"avg_{event_type_name}_{example_type_name}_sample_wise_r2"
            ] = np.mean(sample_wise_r2)

            # compute correlation metrics for significant perturbations
            sample_wise_spearmanR = []
            sample_wise_pearsonR = []
            sample_wise_r2 = []

            for event_id in tqdm(subset_df["EVENT"].unique()):
                event_df = subset_df[subset_df["EVENT"] == event_id]
                event_df = event_df[
                    (event_df["PSI"] - event_df["CONTROLS_AVG_PSI"]).abs() >= 15.0
                ].reset_index(drop=True)
                if len(event_df) < 10:
                    continue

                spearmanR = np.nan_to_num(
                    spearmanr(event_df["PSI"], event_df["PSI_PREDS"])[0]
                )
                pearsonR = np.nan_to_num(
                    pearsonr(event_df["PSI"], event_df["PSI_PREDS"])[0]
                )
                r2 = np.nan_to_num(r2_score(event_df["PSI"], event_df["PSI_PREDS"]))

                sample_wise_spearmanR.append(spearmanR)
                sample_wise_pearsonR.append(pearsonR)
                sample_wise_r2.append(r2)

            sample_wise_spearmanR = np.array(sample_wise_spearmanR)
            sample_wise_pearsonR = np.array(sample_wise_pearsonR)
            sample_wise_r2 = np.array(sample_wise_r2)

            metrics[
                f"num_events_with_at_least_10_significant_perturbations_{event_type_name}_{example_type_name}"
            ] = len(sample_wise_spearmanR)
            metrics[
                f"avg_{event_type_name}_{example_type_name}_significant_perturbations_sample_wise_spearmanR"
            ] = np.mean(sample_wise_spearmanR)
            metrics[
                f"avg_{event_type_name}_{example_type_name}_significant_perturbations_sample_wise_pearsonR"
            ] = np.mean(sample_wise_pearsonR)
            metrics[
                f"avg_{event_type_name}_{example_type_name}_significant_perturbations_sample_wise_r2"
            ] = np.mean(sample_wise_r2)

            # compute classification metrics
            subset_df["sample_has_sig_different_PSI_than_control"] = (
                subset_df["PSI"] - subset_df["CONTROLS_AVG_PSI"]
            ).abs() > 15.0
            subset_df["sample_has_sig_different_predicted_PSI_than_control"] = (
                subset_df["PSI_PREDS"] - subset_df["PRED_CONTROLS_AVG_PSI"]
            ).abs() > 15.0

            accuracy = accuracy_score(
                y_true=subset_df["sample_has_sig_different_PSI_than_control"],
                y_pred=subset_df["sample_has_sig_different_predicted_PSI_than_control"],
            )
            adjusted_balanced_accuracy = balanced_accuracy_score(
                y_true=subset_df["sample_has_sig_different_PSI_than_control"],
                y_pred=subset_df["sample_has_sig_different_predicted_PSI_than_control"],
                adjusted=True,
            )
            precision = precision_score(
                y_true=subset_df["sample_has_sig_different_PSI_than_control"],
                y_pred=subset_df["sample_has_sig_different_predicted_PSI_than_control"],
            )
            recall = recall_score(
                y_true=subset_df["sample_has_sig_different_PSI_than_control"],
                y_pred=subset_df["sample_has_sig_different_predicted_PSI_than_control"],
            )
            f1 = f1_score(
                y_true=subset_df["sample_has_sig_different_PSI_than_control"],
                y_pred=subset_df["sample_has_sig_different_predicted_PSI_than_control"],
            )

            metrics[
                f"{event_type_name}_{example_type_name}_sig_different_PSI_accuracy"
            ] = accuracy
            metrics[
                f"{event_type_name}_{example_type_name}_sig_different_PSI_adjusted_balanced_accuracy"
            ] = adjusted_balanced_accuracy
            metrics[
                f"{event_type_name}_{example_type_name}_sig_different_PSI_precision"
            ] = precision
            metrics[
                f"{event_type_name}_{example_type_name}_sig_different_PSI_recall"
            ] = recall
            metrics[f"{event_type_name}_{example_type_name}_sig_different_PSI_f1"] = f1

    # save metrics to a JSON file
    with open(summary_save_path, "w") as f:
        json.dump(metrics, f, indent=4)

    # print metrics
    print("\nValidation Metrics:")
    for metric_name, value in metrics.items():
        print(f"{metric_name}: {value}")

    return metrics


def parse_args():
    parser = ArgumentParser()
    parser.add_argument(
        "--config", type=str, required=True, help="Path to the config file."
    )
    parser.add_argument(
        "--split_name",
        type=str,
        required=False,
        default="test",
        help="Name of the split to evaluate on.",
    )
    parser.add_argument(
        "--predictions_dir",
        type=str,
        required=False,
        default=None,
        help="Directory to save predictions.",
    )
    parser.add_argument("--create_best_ckpt_copy", action="store_true", default=False)
    parser.add_argument("--overwrite_predictions", action="store_true", default=False)
    parser.add_argument("--metric_name", type=str, default=None)
    parser.add_argument("--optimal_direction", type=str, default="max")
    return parser.parse_args()


def main():
    args = parse_args()
    config = args.config

    # load config
    with open(config, "r") as f:
        config = json.load(f)

    # setup file storage
    run_name = (
        "psi_predictor_test"
        if "run_name" not in config["train_config"]
        else config["train_config"]["run_name"]
    )
    run_save_dir = os.path.join(
        config["train_config"]["saved_models_dir"],
        run_name,
    )
    ckpts_dir = os.path.join(run_save_dir, "checkpoints")
    assert os.path.exists(
        ckpts_dir
    ), f"Checkpoints directory {ckpts_dir} does not exist."

    predictions_dir = args.predictions_dir
    if predictions_dir is None:
        predictions_dir = os.path.join(run_save_dir, "predictions", args.split_name)
    os.makedirs(predictions_dir, exist_ok=True)

    if (
        os.path.exists(os.path.join(predictions_dir, "preds.csv"))
        and not args.overwrite_predictions
    ):
        print("Final predictions already exist, ending execution.")
        return

    # setup data module
    data_module = KnockdownData(config)
    data_module.prepare_data()
    data_module.setup()
    if args.split_name == "test":
        dataloader = data_module.test_dataloader()
    elif args.split_name == "val":
        dataloader = data_module.val_dataloader()
    elif args.split_name == "train":
        dataloader = data_module.train_dataloader()
    else:
        raise ValueError(
            f"Invalid split name: {args.split_name}, must be one of 'train', 'val', 'test'."
        )

    # get number of gpus
    n_gpus = torch.cuda.device_count()
    os.environ["SLURM_JOB_NAME"] = "interactive"
    print(f"Number of GPUs: {n_gpus}")
    pred_writer = CustomWriter(output_dir=predictions_dir, write_interval="epoch")
    trainer = Trainer(
        accelerator="gpu",
        devices="auto",
        precision="32-true",
        strategy="ddp",
        callbacks=[pred_writer],
    )

    # if all predictions exist, skip the prediction step
    if (
        all(
            [
                os.path.exists(os.path.join(predictions_dir, f"predictions_{i}.pt"))
                for i in range(n_gpus)
            ]
        )
        and not args.overwrite_predictions
    ):
        print("Predictions already exist, skipping prediction step.")
    else:
        # find the best checkpoint
        metric_name = args.metric_name
        if metric_name is None:
            if "early_stopping_metric" in config["train_config"]:
                metric_name = config["train_config"]["early_stopping_metric"]
            else:
                metric_name = "epoch"
        optimal_direction = args.optimal_direction
        if optimal_direction is None:
            if "early_stopping_mode" in config["train_config"]:
                optimal_direction = config["train_config"]["early_stopping_mode"]
            else:
                optimal_direction = "max"

        best_ckpt_path = find_best_checkpoint(
            ckpts_dir,
            metric_name=metric_name,
            optimal_direction=optimal_direction,
            create_best_ckpt_copy=args.create_best_ckpt_copy,
        )

        # setup model
        num_splicing_factors = data_module.num_splicing_factors
        has_gene_exp_values = data_module.has_gene_exp_values
        event_type_to_ind = data_module.event_type_to_ind
        example_type_to_ind = data_module.example_type_to_ind
        example_types_in_this_split_type = data_module.example_types_in_this_split_type
        model = PSIPredictor(
            config,
            num_splicing_factors,
            has_gene_exp_values,
            event_type_to_ind,
            example_type_to_ind,
            example_types_in_this_split_type,
        )

        # make predictions using the best checkpoint
        trainer.predict(
            model,
            dataloader,
            ckpt_path=best_ckpt_path,
            return_predictions=False,
        )
        print("Done predicting.")

    # read predictions from the files and concatenate them
    # only the first rank process will read the predictions and concatenate them
    if trainer.global_rank == 0:
        print("Concatenating predictions and creating final CSV on rank 0.")
        psi_vals = []
        pred_psi_vals = []
        batch_indices = []
        for i in range(n_gpus):
            p = torch.load(os.path.join(predictions_dir, f"predictions_{i}.pt"))
            psi_val = np.concatenate([batch["psi_val"] for batch in p])
            pred_psi_val = np.concatenate([batch["pred_psi_val"] for batch in p])
            psi_vals.append(psi_val)
            pred_psi_vals.append(pred_psi_val)

            bi = torch.load(os.path.join(predictions_dir, f"batch_indices_{i}.pt"))[0]
            bi = np.concatenate([inds for inds in bi])
            batch_indices.append(bi)

        psi_vals = np.concatenate(psi_vals, axis=0)
        pred_psi_vals = np.concatenate(pred_psi_vals, axis=0)
        batch_indices = np.concatenate(batch_indices, axis=0)

        # sort the ground truth and predicted PSI vals based on the original order using the batch indices
        sorted_idxs = np.argsort(batch_indices)
        psi_vals = psi_vals[sorted_idxs]
        pred_psi_vals = pred_psi_vals[sorted_idxs]

        # create a dataframe with the predictions
        df = None
        if args.split_name == "test":
            df = data_module.test_data
        elif args.split_name == "val":
            df = data_module.val_data
        elif args.split_name == "train":
            df = data_module.train_data

        assert len(df) == len(
            psi_vals
        ), f"Length of dataframe ({len(df)}) and PSI values ({len(psi_vals)}) do not match."
        df["PSI_PREDS"] = pred_psi_vals * 100
        assert np.allclose(
            df["PSI"].values, psi_vals * 100
        ), "Ground truth PSI values do not match."

        # add the predicted avg control PSI values to the dataframe
        # the control is the sample with SAMPLE == "AV_Controls"
        control_df = df[df["SAMPLE"] == "AV_Controls"][["EVENT", "PSI_PREDS"]]
        control_df.columns = ["EVENT", "PRED_CONTROLS_AVG_PSI"]
        ori_len = len(df)
        df = df.merge(control_df, on="EVENT", how="inner")
        assert (
            len(df) == ori_len
        ), f"Length of dataframe ({len(df)}) does not match the original length ({ori_len})."
        df.to_csv(os.path.join(predictions_dir, "preds.csv"), index=False)
        print(f"Predictions saved to {predictions_dir}/preds.csv")

        # compute and save validation metrics
        compute_and_save_validation_metrics(
            df, os.path.join(predictions_dir, "validation_metrics.json")
        )


if __name__ == "__main__":
    main()
