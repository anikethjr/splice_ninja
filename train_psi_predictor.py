import numpy as np
import pandas as pd
import os
import pdb
import json
import wandb
from argparse import ArgumentParser, BooleanOptionalAction
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F

import lightning as L
from lightning import Trainer
from lightning.pytorch.callbacks.early_stopping import EarlyStopping
from lightning.pytorch.callbacks.model_checkpoint import ModelCheckpoint
from lightning.pytorch.loggers import WandbLogger

from splice_ninja.dataloaders.knockdown_data import KnockdownData
from splice_ninja.models.psi_predictor import PSIPredictor

np.random.seed(0)
torch.manual_seed(0)
torch.set_float32_matmul_precision("medium")


def get_latest_wandb_run_id_from_run_name(team, project, run_name):
    """Retrieve the most recent wandb run ID given a run name."""
    api = wandb.Api()
    runs = api.runs(f"{team}/{project}")

    # Filter runs by name and sort by created timestamp (latest first)
    matching_runs = sorted(
        [run for run in runs if run.name == run_name],
        key=lambda r: r.created_at,  # Sort by creation time
        reverse=True,  # Latest run first
    )

    return matching_runs[0].id if matching_runs else None


def parse_args():
    parser = ArgumentParser()
    parser.add_argument(
        "--config", type=str, required=True, help="Path to the config file."
    )
    parser.add_argument(
        "--resume_from_checkpoint",
        action=BooleanOptionalAction,
        default=False,
        help="Resume training from a checkpoint.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    config = args.config

    # load config
    with open(config, "r") as f:
        config = json.load(f)

    # setup data module
    data_module = KnockdownData(config)
    data_module.prepare_data()
    data_module.setup()

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
    os.makedirs(run_save_dir, exist_ok=True)

    logs_dir = os.path.join(run_save_dir, "logs")
    os.makedirs(logs_dir, exist_ok=True)

    ckpts_dir = os.path.join(run_save_dir, "checkpoints")
    os.makedirs(ckpts_dir, exist_ok=True)

    # setup callbacks + trainer
    team_name = config["train_config"]["wandb_team"]
    project_name = config["train_config"]["wandb_project"]
    existing_run_id = get_latest_wandb_run_id_from_run_name(
        team_name, project_name, run_name
    )
    if existing_run_id is None:
        print("No existing run found.")
    else:
        print(f"Resuming wandb logging from run ID: {existing_run_id}")
    logger = WandbLogger(
        project=project_name,
        name=run_name,
        save_dir=logs_dir,
        id=existing_run_id,
        resume="must" if existing_run_id is not None else "allow",
    )

    early_stopping_metric = config["train_config"]["early_stopping_metric"]
    early_stopping_mode = config["train_config"]["early_stopping_mode"]
    patience = config["train_config"]["patience"]
    max_epochs = config["train_config"]["max_epochs"]
    # # checkpoint based on metric
    # checkpointing_cb = ModelCheckpoint(
    #     dirpath=ckpts_dir,
    #     filename="epoch={epoch}-step={step}-metric={"
    #     + f"{early_stopping_metric}"
    #     + ":.6f}",
    #     monitor=early_stopping_metric,
    #     mode=early_stopping_mode,
    #     save_top_k=1,
    #     auto_insert_metric_name=False,
    # )
    # checkpoint based on epoch so that training can be resumed easily
    checkpointing_cb_based_on_epoch = ModelCheckpoint(
        dirpath=ckpts_dir,
        filename="epoch={epoch}-step={step}-metric={"
        + f"{early_stopping_metric}"
        + ":.6f}",
        monitor=None,
        save_top_k=-1,
        auto_insert_metric_name=False,
    )
    # # early stopping
    # early_stopping_cb = EarlyStopping(
    #     monitor=early_stopping_metric,
    #     mode=early_stopping_mode,
    #     patience=patience,
    # )

    os.environ["SLURM_JOB_NAME"] = "interactive"
    # get number of gpus
    n_gpus = torch.cuda.device_count()
    print(f"Number of GPUs: {n_gpus}")

    trainer = Trainer(
        accelerator="gpu",
        devices="auto",
        log_every_n_steps=10,
        max_epochs=max_epochs,
        gradient_clip_val=0.2,
        logger=logger,
        default_root_dir=os.path.join(
            config["train_config"]["saved_models_dir"], run_name
        ),
        callbacks=[
            checkpointing_cb_based_on_epoch,
            # checkpointing_cb,
            # early_stopping_cb,
        ],
        precision="32-true",
        strategy="ddp",
        reload_dataloaders_every_n_epochs=1,
    )

    # train model
    resume_flag = args.resume_from_checkpoint
    if resume_flag:
        if len(os.listdir(ckpts_dir)) == 0:
            print("No checkpoint found to resume from. Training from scratch.")
            resume_flag = False
        else:
            previous_ckpts = os.listdir(ckpts_dir)
            previous_ckpts = [
                x for x in previous_ckpts if x.endswith(".ckpt")
            ]  # filter out other files
            previous_ckpts = [
                x for x in previous_ckpts if "epoch" in x
            ]  # filter out other files like "best.ckpt"
            print("Previous checkpoints found: ", previous_ckpts)

            # sort by epoch number
            print(
                "Epoch numbers: ",
                [int(x.split("-")[0].split("=")[1]) for x in previous_ckpts],
            )
            previous_ckpts.sort(key=lambda x: int(x.split("-")[0].split("=")[1]))
            previous_ckpt_path = previous_ckpts[-1]

            previous_ckpt_path = os.path.join(ckpts_dir, previous_ckpt_path)
            print(f"Resuming from checkpoint: {previous_ckpt_path}")
            # trainer.validate(
            #     model,
            #     datamodule=data_module,
            #     ckpt_path=previous_ckpt_path,
            #     verbose=True,
            # )
            trainer.fit(model, datamodule=data_module, ckpt_path=previous_ckpt_path)

    if not resume_flag:
        print("Training from scratch.")
        trainer.fit(model, datamodule=data_module)


if __name__ == "__main__":
    main()
