import numpy as np
import pandas as pd
import os
import pdb
import json
from argparse import ArgumentParser, BooleanOptionalAction
from tqdm import tqdm

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


def find_best_checkpoint_and_verify_that_training_is_complete(
    checkpoint_dir,
    early_stopping_mode,
    patience=10,
    max_train_epochs=100,
    proceed_even_if_training_incomplete=False,
    create_best_ckpt_copy=False,
):
    """
    Find the best checkpoint in the directory and verify that the training is complete.
    Verfication is done by checking if there are at least `patience` number of checkpoints with worse metrics than the best checkpoint.
    Args:
        checkpoint_dir: Directory containing the checkpoints.
        early_stopping_mode: Mode of the early stopping metric. One of "min" or "max".
        patience: Patience for checking if the training is complete.
        max_train_epochs: Maximum number of training epochs
        proceed_even_if_training_incomplete: If True, the function will not raise an error if the training is not complete.
        create_best_ckpt_copy: If True, the function will create a copy of the best checkpoint in the same directory with the name "best.ckpt".
    """
    # find the best checkpoint
    best_checkpoint = None
    best_metric = None
    best_metric_epoch = None
    max_epoch = -1  # epoch number of the last checkpoint
    for f in os.listdir(checkpoint_dir):
        if not f.endswith(".ckpt"):
            continue

        if f == "best.ckpt":
            print(
                "WARNING: Found a file named 'best.ckpt' in the directory. Skipping it. It will be overwritten if create_best_ckpt_copy is set to True."
            )
            continue

        max_epoch = max(max_epoch, int(f.split("epoch=")[1].split("-")[0]))
        ckpt_metric = None

        # names are of the form "epoch={epoch}-step={step}-metric={early_stopping_metric:.6f}.ckpt"
        # split on "-" if there are version numbers in the file name (they are added at the end)
        ckpt_metric = f.split("metric=")[1].split(".ckpt")[0].split("-")[0]
        ckpt_metric = float(ckpt_metric)

        ckpt_is_better_than_best = False
        if best_metric is None:
            ckpt_is_better_than_best = True
        elif early_stopping_mode == "min" and ckpt_metric <= best_metric:
            ckpt_is_better_than_best = True
        elif early_stopping_mode == "max" and ckpt_metric >= best_metric:
            ckpt_is_better_than_best = True

        if ckpt_is_better_than_best:
            if best_metric is not None and ckpt_metric == best_metric:
                # open the ckpt files to compare the exact metric values
                best_ckpt_so_far = torch.load(
                    os.path.join(checkpoint_dir, best_checkpoint),
                    map_location="cpu",
                )
                ckpt = torch.load(os.path.join(checkpoint_dir, f), map_location="cpu")

                check = False
                for key in ckpt["callbacks"].keys():
                    if (
                        key.startswith("ModelCheckpoint")
                        and ckpt["callbacks"][key]["current_score"] is not None
                    ):
                        print(
                            f"Using scores from ckpts to compare the following ckpt files: {best_checkpoint} and {f}"
                        )
                        ckpt_metric = ckpt["callbacks"][key]["current_score"]
                        best_metric = best_ckpt_so_far["callbacks"][key][
                            "current_score"
                        ]
                        if (
                            best_metric is None
                        ):  # if the best ckpt metric is None, then the current ckpt is better because checkpoints with metrics are logged only if they are better than the best checkpoint
                            check = True
                            break
                        if (
                            early_stopping_mode == "min" and ckpt_metric < best_metric
                        ) or (
                            early_stopping_mode == "max" and ckpt_metric > best_metric
                        ):
                            check = True
                        break

                if not check:
                    continue

            best_metric = ckpt_metric
            best_checkpoint = f
            best_metric_epoch = int(f.split("epoch=")[1].split("-")[0])

    # check if the training is complete
    if best_checkpoint is None:
        raise ValueError("No checkpoint found in the directory.")
    if max_epoch - best_metric_epoch < patience:
        if max_epoch == (max_train_epochs - 1):
            print("WARNING: Max training epochs completed, so patience ignored")
        else:
            if not proceed_even_if_training_incomplete:
                raise ValueError(
                    f"Training may not be complete. Current best checkpoint is from epoch {best_metric_epoch} and the last checkpoint is from epoch {max_epoch}."
                )
            else:
                print(
                    "WARNING: Training may not be complete. Current best checkpoint is from epoch",
                    best_metric_epoch,
                    "and the last checkpoint is from epoch",
                    max_epoch,
                )

    # create a copy of the best checkpoint
    if create_best_ckpt_copy:
        best_ckpt_path = os.path.join(checkpoint_dir, best_checkpoint)
        best_ckpt_copy_path = os.path.join(checkpoint_dir, "best.ckpt")
        os.system(f"cp {best_ckpt_path} {best_ckpt_copy_path}")
        print(f"Created a copy of the best checkpoint at {best_ckpt_copy_path}")

    return os.path.join(checkpoint_dir, best_checkpoint)


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
    parser.add_argument(
        "--proceed_even_if_training_incomplete", action="store_true", default=False
    )
    parser.add_argument("--create_best_ckpt_copy", action="store_true", default=False)
    parser.add_argument("--overwrite_predictions", action="store_true", default=False)
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
    if all(
        [
            os.path.exists(os.path.join(predictions_dir, f"predictions_{i}.pt"))
            for i in range(n_gpus)
        ]
    ):
        print("Predictions already exist, skipping prediction step.")
    else:
        # find the best checkpoint and verify that the training is complete
        best_ckpt_path = find_best_checkpoint_and_verify_that_training_is_complete(
            ckpts_dir,
            config["train_config"]["early_stopping_mode"],
            patience=config["train_config"]["patience"],
            max_train_epochs=config["train_config"]["max_epochs"],
            proceed_even_if_training_incomplete=args.proceed_even_if_training_incomplete,
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
        df["PSI_PREDS"] = pred_psi_vals
        assert np.allclose(
            df["PSI"].values, psi_vals * 100
        ), "Ground truth PSI values do not match."
        df.to_csv(os.path.join(predictions_dir, "preds.csv"), index=False)
        print(f"Predictions saved to {predictions_dir}/preds.csv")


if __name__ == "__main__":
    main()
