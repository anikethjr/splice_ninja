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


def parse_args():
    parser = ArgumentParser()
    parser.add_argument(
        "--config", type=str, required=True, help="Path to the config file."
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
        predictions_dir = os.path.join(run_save_dir, "predictions", "all_KD")
    os.makedirs(predictions_dir, exist_ok=True)

    if (
        os.path.exists(os.path.join(predictions_dir, "preds.csv"))
        and not args.overwrite_predictions
    ):
        print("Final predictions already exist, ending execution.")
        return

    # setup data module
    kd_config = config.copy()
    kd_config["data_config"]["dataset_name"] = "KD"
    kd_config["data_config"]["min_samples_for_event_to_be_considered"] = 100
    kd_config["train_config"]["split_type"] = "chromosome"
    kd_config["train_config"]["use_VastDB_splicing_factors"] = True
    data_module = KnockdownData(kd_config)
    data_module.prepare_data()
    data_module.setup()
    dataloader = data_module.all_data_dataloader()

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
        df = data_module.unified_data

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
        df = df.merge(control_df, on="EVENT", how="inner", validate="one_to_one")
        df.to_csv(os.path.join(predictions_dir, "preds.csv"), index=False)
        print(f"Predictions saved to {predictions_dir}/preds.csv")


if __name__ == "__main__":
    main()
