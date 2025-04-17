import numpy as np
import pandas as pd
import os
import pdb
import json
import urllib
from tqdm import tqdm
from statsmodels.stats.multitest import multipletests

import genomepy

import torch
import torch.distributed as dist
from torch.utils.data import Dataset, DataLoader

import lightning as L
from lightning.pytorch import LightningDataModule

from splice_ninja.utils import get_ensembl_gene_id_hgnc_with_alias, one_hot_encode_dna

np.random.seed(0)
torch.manual_seed(0)


# DistributedSampler if we want to have a uniform distribution of PSI values in an epoch
class UniformPSIDistributionDistributedSampler(
    torch.utils.data.distributed.DistributedSampler
):
    def __init__(
        self, data_module: LightningDataModule, dataset=None, num_replicas=None
    ):
        self.data_module = data_module
        self.dataset = dataset
        self.epoch = self.data_module.trainer.current_epoch
        self.set_epoch(self.epoch)

        # get the number of replicas and rank if not provided
        if num_replicas is None:
            if not dist.is_available():
                raise RuntimeError("Requires distributed package to be available")
            num_replicas = dist.get_world_size()
        rank = dist.get_rank()
        if rank >= num_replicas or rank < 0:
            raise ValueError(
                f"Invalid rank {rank}, rank should be in the interval [0, {num_replicas - 1}]"
            )
        self.num_replicas = num_replicas
        self.rank = rank

        # get batch size
        self.batch_size = self.data_module.config["train_config"]["batch_size"]

        # get full data
        if self.dataset is not None:
            self.data = self.dataset.data
        elif self.split == "train":
            self.data = self.data_module.train_data
        else:
            raise ValueError(
                f"UniformPSIDistributionDistributedSampler should only be used with the train split, not {self.split}"
            )

        # compute length
        self.length = len(self.data) // self.num_replicas
        self.length = self.length - (
            self.length % self.batch_size
        )  # make sure the length is a multiple of the batch size
        assert (
            self.length % self.batch_size == 0
        ), f"Length is not a multiple of the batch size, length: {self.length}, batch size: {self.batch_size}"
        print(
            "Rank: {}, Seed: {}, Length: {}, Data length: {}".format(
                self.rank, self.seed, self.length, len(self.data)
            )
        )

    def set_epoch(self, epoch):
        self.epoch = epoch

        # set seed - add epoch to seed to shuffle differently each epoch
        self.seed = self.data_module.config["train_config"]["seed"] + self.epoch
        np.random.seed(self.seed)

    def __len__(self):
        return self.length

    def __iter__(self):
        # resample events uniformly in the [0, 1]
        print("Resampling PSI values so that they are uniformly distributed")
        psi_values = self.data["PSI"].values / 100.0
        n_total = self.length

        # Define bins and bin edges
        n_bins = 10
        bin_edges = np.linspace(0, 1, n_bins + 1)
        print(f"Bin edges: {bin_edges}")
        bin_indices = np.digitize(psi_values, bin_edges, right=True)

        # Create mapping from bin to values
        bin_to_values = {i: psi_values[bin_indices == i] for i in range(1, n_bins + 1)}
        bin_to_df_indices = {
            i: self.data[bin_indices == i].index for i in range(1, n_bins + 1)
        }
        bin_to_edges = {
            k: [bin_edges[k - 1], bin_edges[k]] for k in bin_to_values.keys()
        }

        # Remove empty bins
        bin_to_values = {k: v for k, v in bin_to_values.items() if len(v) > 0}
        bin_to_df_indices = {k: v for k, v in bin_to_df_indices.items() if len(v) > 0}
        final_n_bins = len(bin_to_values)
        print(f"Number of nonzero bins: {final_n_bins} out of {n_bins}")

        # Determine how many samples to draw from each bin
        samples_per_bin = n_total // final_n_bins
        extra = n_total % final_n_bins  # in case n_total not divisible by final_n_bins

        resampled_value_indices = []
        for i in bin_to_values:
            values = bin_to_values[i]
            df_indices = bin_to_df_indices[i]
            count = samples_per_bin + (1 if extra > 0 else 0)
            extra -= 1 if extra > 0 else 0

            sampled = np.random.choice(df_indices, count, replace=True)

            # assert that all sampled values are in the bin
            sampled_values = self.data.loc[sampled, "PSI"].values / 100.0
            this_bin_edges = bin_to_edges[i]
            assert np.all(sampled_values >= this_bin_edges[0]) and np.all(
                sampled_values <= this_bin_edges[1]
            ), f"Sampled values are not in the bin, bin edges: {this_bin_edges}"

            resampled_value_indices.extend(sampled)

        # shuffle the resampled value indices
        np.random.shuffle(resampled_value_indices)
        assert (
            len(resampled_value_indices) == n_total
        ), f"Resampled value indices length is {len(resampled_value_indices)} but expected length is {n_total}"

        number_of_samples_from_bins = []
        sampled_values = self.data.loc[resampled_value_indices, "PSI"].values / 100.0
        for i in range(1, n_bins + 1):
            number_of_samples_from_bins.append(
                np.sum(
                    (sampled_values >= bin_edges[i - 1])
                    & (sampled_values <= bin_edges[i])
                )
            )
        print(f"Number of samples from bins: {number_of_samples_from_bins}")

        return iter(resampled_value_indices)


# DistributedSampler if we want to have N events per batch
# Note that the corresponding control data will always be included in the batch as the first sample from an event.
# This is important for the model to learn the difference between control and non-control data.
class NEventsPerBatchDistributedSampler(
    torch.utils.data.distributed.DistributedSampler
):
    def split_data_among_ranks(self):
        # function to split data among ranks

        # set seed - add epoch to seed to shuffle differently each epoch
        self.seed = self.data_module.config["train_config"]["seed"] + self.epoch
        np.random.seed(self.seed)
        # get full data
        if self.dataset is not None:
            data = self.dataset.data
        elif self.split == "train":
            data = self.data_module.train_data
        else:
            raise ValueError(
                f"NEventsPerBatchDistributedSampler should only be used with the train split, not {self.split}"
            )

        # get unique event IDs
        self.event_ids = data["EVENT"].unique()
        # shuffle the event IDs
        np.random.shuffle(self.event_ids)

        # split the event IDs among the ranks
        self.event_ids = np.array_split(self.event_ids, self.num_replicas)[self.rank]

        # subset the data to only include the events in the event IDs
        # we only need the row idx, the event ID, and the sample name
        self.this_rank_data = data.loc[data["EVENT"].isin(self.event_ids)]

        # compute length
        self.length = len(data) // self.num_replicas
        self.length = self.length - (
            self.length % self.batch_size
        )  # make sure the length is a multiple of the batch size
        assert (
            self.length % self.batch_size == 0
        ), f"Length is not a multiple of the batch size, length: {self.length}, batch size: {self.batch_size}"
        print(
            "Rank: {}, Seed: {}, Length: {}, Num events: {}, Num examples: {}".format(
                self.rank,
                self.seed,
                self.length,
                len(self.event_ids),
                len(self.this_rank_data),
            )
        )

    def __init__(
        self,
        data_module: LightningDataModule,
        dataset=None,
        num_replicas=None,
        rank=None,
        split="train",
    ):
        self.data_module = data_module
        self.dataset = dataset
        self.split = split
        self.epoch = self.data_module.trainer.current_epoch

        # hyperparam that affects when we use the ranking loss and this determines when
        # we return a specific number of events per batch
        if (
            "num_epochs_after_which_to_use_ranking_loss"
            in self.data_module.config["train_config"]
        ):
            self.num_epochs_after_which_to_use_ranking_loss = self.data_module.config[
                "train_config"
            ]["num_epochs_after_which_to_use_ranking_loss"]
        else:
            self.num_epochs_after_which_to_use_ranking_loss = 0

        # hyperparams that affect how we use the controls data + how we define significant events
        if (
            "num_epochs_for_training_on_control_data_only"
            in self.data_module.config["train_config"]
        ):
            self.num_epochs_for_training_on_control_data_only = self.data_module.config[
                "train_config"
            ]["num_epochs_for_training_on_control_data_only"]
        else:
            self.num_epochs_for_training_on_control_data_only = 0
        if "dPSI_threshold_for_significance" in self.data_module.config["train_config"]:
            self.dPSI_threshold_for_significance = self.data_module.config[
                "train_config"
            ]["dPSI_threshold_for_significance"]
        else:
            self.dPSI_threshold_for_significance = 0.0
        if "upsample_significant_events" in self.data_module.config["train_config"]:
            self.upsample_significant_events = self.data_module.config["train_config"][
                "upsample_significant_events"
            ]
        else:
            self.upsample_significant_events = False

        # get the number of replicas and rank if not provided
        if num_replicas is None:
            if not dist.is_available():
                raise RuntimeError("Requires distributed package to be available")
            num_replicas = dist.get_world_size()
        if rank is None:
            if not dist.is_available():
                raise RuntimeError("Requires distributed package to be available")
            rank = dist.get_rank()
        if rank >= num_replicas or rank < 0:
            raise ValueError(
                f"Invalid rank {rank}, rank should be in the interval [0, {num_replicas - 1}]"
            )
        self.num_replicas = num_replicas
        self.rank = rank

        # get batch size
        self.batch_size = self.data_module.config["train_config"]["batch_size"]

        # get N
        if (self.epoch < self.num_epochs_for_training_on_control_data_only) or (
            self.epoch < self.num_epochs_after_which_to_use_ranking_loss
        ):
            assert (
                self.upsample_significant_events
            ), "upsample_significant_events should be True to use NEventsPerBatchDistributedSampler with control data or when ranking loss is not used"
            self.N_events_per_batch = self.batch_size
        else:
            assert (
                "N_events_per_batch" in self.data_module.config["train_config"]
            ), "N_events_per_batch should be in the config to use NEventsPerBatchDistributedSampler"
            self.N_events_per_batch = self.data_module.config["train_config"][
                "N_events_per_batch"
            ]

        # compute examples per event
        if self.batch_size % self.N_events_per_batch != 0:
            print(
                "WARNING: Batch size is not a multiple of N_events_per_batch, all events might not have the same number of examples in a batch"
            )
        self.examples_per_event = np.array_split(
            np.arange(self.batch_size), self.N_events_per_batch
        )
        print(f"Examples per event: {self.examples_per_event}")

        # split data among ranks
        self.split_data_among_ranks()

    def set_epoch(self, epoch):
        self.epoch = epoch
        self.split_data_among_ranks()

    def __len__(self):
        return self.length

    def __iter__(self):
        num_batches_so_far = 0
        total_num_batches = self.length // self.batch_size

        # resample events uniformly in the [0, 1]
        if (self.epoch < self.num_epochs_for_training_on_control_data_only) or (
            self.epoch < self.num_epochs_after_which_to_use_ranking_loss
        ):
            assert (
                self.upsample_significant_events
            ), "upsample_significant_events should be True to use NEventsPerBatchDistributedSampler with control data or when ranking loss is not used"
            # resample PSI values uniformly in the [0, 1]
            print(
                "Resampling control data so that PSI values are uniformly distributed"
            )
            psi_values = self.this_rank_data["PSI"].values / 100.0
            n_total = self.length

            # Define bins and bin edges
            n_bins = 20
            bin_edges = np.linspace(0, 1, n_bins + 1)
            print(f"Bin edges: {bin_edges}")
            bin_indices = np.digitize(psi_values, bin_edges, right=True)

            # Create mapping from bin to values
            bin_to_values = {
                i: psi_values[bin_indices == i] for i in range(1, n_bins + 1)
            }
            bin_to_df_indices = {
                i: self.this_rank_data[bin_indices == i].index
                for i in range(1, n_bins + 1)
            }
            bin_to_edges = {
                k: [bin_edges[k - 1], bin_edges[k]] for k in bin_to_values.keys()
            }

            # Remove empty bins
            bin_to_values = {k: v for k, v in bin_to_values.items() if len(v) > 0}
            bin_to_df_indices = {
                k: v for k, v in bin_to_df_indices.items() if len(v) > 0
            }
            final_n_bins = len(bin_to_values)
            print(f"Final number of non-zero bins: {final_n_bins} out of {n_bins}")

            # Determine how many samples to draw from each bin
            samples_per_bin = n_total // final_n_bins
            extra = (
                n_total % final_n_bins
            )  # in case n_total not divisible by final_n_bins

            resampled_value_indices = []
            for i in bin_to_values:
                values = bin_to_values[i]
                df_indices = bin_to_df_indices[i]
                count = samples_per_bin + (1 if extra > 0 else 0)
                extra -= 1 if extra > 0 else 0

                sampled = np.random.choice(df_indices, count, replace=True)

                # assert that all sampled values are in the bin
                sampled_values = self.this_rank_data.loc[sampled, "PSI"].values / 100.0
                this_bin_edges = bin_to_edges[i]
                assert np.all(sampled_values >= this_bin_edges[0]) and np.all(
                    sampled_values <= this_bin_edges[1]
                ), f"Sampled values are not in the bin, bin edges: {this_bin_edges}"

                resampled_value_indices.extend(sampled)

            # shuffle the resampled value indices
            np.random.shuffle(resampled_value_indices)
            assert (
                len(resampled_value_indices) == n_total
            ), f"Resampled value indices length is {len(resampled_value_indices)} but expected length is {n_total}"

            number_of_samples_from_bins = []
            sampled_values = (
                self.this_rank_data.loc[resampled_value_indices, "PSI"].values / 100.0
            )
            for i in range(1, n_bins + 1):
                number_of_samples_from_bins.append(
                    np.sum(
                        (sampled_values >= bin_edges[i - 1])
                        & (sampled_values <= bin_edges[i])
                    )
                )
            print(f"Number of samples from bins: {number_of_samples_from_bins}")

            return iter(resampled_value_indices)

        self.grouped_rank_data = self.this_rank_data.groupby("EVENT", sort=False)
        assert len(self.grouped_rank_data) == len(
            self.event_ids
        ), f"Expected number of events: {len(self.event_ids)}, actual number of events: {len(self.grouped_rank_data)}"
        indices = []

        cur_event_idx = 0
        current_batch_idxs = np.zeros(self.batch_size, dtype=int)
        while num_batches_so_far < total_num_batches:
            for event_id, event_data in self.grouped_rank_data:
                # stop yielding if we have enough batches
                if num_batches_so_far >= total_num_batches:
                    break

                # sample examples for the event
                # first example is always the control data
                current_batch_idxs[
                    self.examples_per_event[cur_event_idx][0]
                ] = event_data[event_data["SAMPLE"] == "AV_Controls"].index[0]
                examples_needed_from_event = (
                    len(self.examples_per_event[cur_event_idx]) - 1
                )
                if examples_needed_from_event > 0:
                    non_control_indices = event_data[
                        event_data["SAMPLE"] != "AV_Controls"
                    ].index
                    # if we are upsampling significant events, we sample 2/3 of the examples from significant events and 1/3 from non-significant events
                    if self.upsample_significant_events:
                        significant_event_indices = event_data[
                            (
                                (
                                    np.abs(
                                        event_data["PSI"]
                                        - event_data["CONTROLS_AVG_PSI"]
                                    )
                                    / 100.0
                                )
                                >= self.dPSI_threshold_for_significance
                            )
                            & (event_data["SAMPLE"] != "AV_Controls")
                        ].index
                        num_significant_events = len(significant_event_indices)
                        if num_significant_events > 0:
                            num_significant_events_to_sample = int(
                                (examples_needed_from_event * 2.0) / 3.0
                            )
                            current_batch_idxs[
                                self.examples_per_event[cur_event_idx][
                                    1 : (1 + num_significant_events_to_sample)
                                ]
                            ] = np.random.choice(
                                significant_event_indices,
                                size=num_significant_events_to_sample,
                                replace=False
                                if len(significant_event_indices)
                                >= num_significant_events_to_sample
                                else True,
                            )
                            examples_needed_from_event -= (
                                num_significant_events_to_sample
                            )

                        # sample the rest of the examples from non-significant events
                        nonsignificant_event_indices = event_data[
                            (
                                (
                                    np.abs(
                                        event_data["PSI"]
                                        - event_data["CONTROLS_AVG_PSI"]
                                    )
                                    / 100.0
                                )
                                < self.dPSI_threshold_for_significance
                            )
                            & (event_data["SAMPLE"] != "AV_Controls")
                        ].index
                        if examples_needed_from_event > 0:
                            if len(nonsignificant_event_indices) > 0:
                                current_batch_idxs[
                                    self.examples_per_event[cur_event_idx][
                                        -examples_needed_from_event:
                                    ]
                                ] = np.random.choice(
                                    nonsignificant_event_indices,
                                    size=examples_needed_from_event,
                                    replace=False
                                    if len(nonsignificant_event_indices)
                                    >= examples_needed_from_event
                                    else True,
                                )
                            else:  # probably never happens, but just in case, sample from the non-control indices
                                current_batch_idxs[
                                    self.examples_per_event[cur_event_idx][
                                        -examples_needed_from_event:
                                    ]
                                ] = np.random.choice(
                                    non_control_indices,
                                    size=examples_needed_from_event,
                                    replace=False
                                    if len(event_data) >= examples_needed_from_event
                                    else True,
                                )
                    else:
                        current_batch_idxs[
                            self.examples_per_event[cur_event_idx][1:]
                        ] = np.random.choice(
                            non_control_indices,
                            size=examples_needed_from_event,
                            replace=False
                            if len(event_data) >= examples_needed_from_event
                            else True,
                        )

                cur_event_idx += 1

                # if N events have been sampled, yield the indices
                if cur_event_idx == self.N_events_per_batch:
                    indices.extend(current_batch_idxs.tolist())
                    num_batches_so_far += 1
                    cur_event_idx = 0

            # shuffle the events
            self.this_rank_data = self.this_rank_data.sample(frac=1)
            self.grouped_rank_data = self.this_rank_data.groupby("EVENT", sort=False)

        num_unique_indices = len(set(indices))

        print(
            "Rank: {}, Seed: {}, Length: {}, Indices len: {}, Num unique indices: {}".format(
                self.rank, self.seed, self.length, len(indices), num_unique_indices
            )
        )

        return iter(indices)
