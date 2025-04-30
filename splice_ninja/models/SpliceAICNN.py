import numpy as np
import pandas as pd
import os
import pdb
import json

import torch
import torch.nn as nn
import torch.nn.functional as F

np.random.seed(0)
torch.manual_seed(0)


class FiLM(nn.Module):
    """
    Layer that applies feature-wise linear modulation to the input tensor. Used for conditioning the network on the splicing factor expression levels and gene expression values.
    """

    def __init__(self, conditioning_dim, num_features, dropout=0.1):
        """
        conditioning_dim: Dimensionality of the conditioning vector
        num_features: Number of channels in the feature map (C)
        dropout: Dropout probability
        """
        super().__init__()
        self.scale = nn.Sequential(
            nn.Linear(conditioning_dim, conditioning_dim * 2),
            nn.ReLU(),
            nn.Linear(conditioning_dim * 2, num_features),
        )
        self.shift = nn.Sequential(
            nn.Linear(conditioning_dim, conditioning_dim * 2),
            nn.ReLU(),
            nn.Linear(conditioning_dim * 2, num_features),
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, conditioning):
        """
        x: Input feature map (B, C, L)
        conditioning: Conditioning vector (B, conditioning_dim)
        """
        gamma = self.dropout(self.scale(conditioning)).unsqueeze(-1)  # (B, C, 1)
        beta = self.dropout(self.shift(conditioning)).unsqueeze(-1)  # (B, C, 1)
        return (x * gamma) + beta


class ResidualBlock(nn.Module):
    """
    Residual block with two convolutional layers and a residual connection.
    """

    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        dilation=1,
        gn_num_groups=None,
        gn_group_size=16,
        use_film=False,
        conditioning_dim=None,
    ):
        super().__init__()

        stride_for_conv1_and_shortcut = 1

        if in_channels != out_channels:
            stride_for_conv1_and_shortcut = 2

        if gn_num_groups is None:
            gn_num_groups = out_channels // gn_group_size

        # modules for processing the input
        self.gn1 = nn.GroupNorm(gn_num_groups, out_channels)
        self.relu1 = nn.ReLU()
        self.conv1 = nn.Conv1d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride_for_conv1_and_shortcut,
            padding="same",
            bias=False,
            dilation=dilation,
        )

        self.gn2 = nn.GroupNorm(gn_num_groups, out_channels)
        self.relu2 = nn.ReLU()
        self.conv2 = nn.Conv1d(
            in_channels=out_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=1,
            padding="same",
            bias=False,
            dilation=dilation,
        )

        # short cut connections
        self.shortcut = nn.Identity()
        if in_channels != out_channels:
            self.shortcut = nn.Conv1d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=1,
                stride=stride_for_conv1_and_shortcut,
                bias=False,
            )

        self.use_film = use_film
        self.conditioning_dim = conditioning_dim
        if use_film:
            assert (
                conditioning_dim is not None and conditioning_dim > 0
            ), "Conditioning dimension must be provided when using FiLM."
            self.film = FiLM(conditioning_dim, out_channels)

    def forward(self, xl, conditioning=None):
        input = self.shortcut(xl)

        xl = self.conv1(self.relu1(self.gn1(xl)))
        xl = self.conv2(self.relu2(self.gn2(xl)))

        # Apply FiLM conditioning
        if self.use_film:
            xl = self.film(xl, conditioning)

        xlp1 = input + xl

        return xlp1


class SpliceAI10k(nn.Module):
    """
    Takes in a one-hot encoded DNA sequence, two masks denoting the exons, introns and background regions of the spliced-in (alternate segment retained) and spliced-out (alternate segment removed) events,
    a splicing factor expression levels vector, a host gene expression value. The model predicts the percent spliced-in (PSI) value of the event.

    The model architecture is based on SpliceAI-10k and uses a stack of dilated convolutional layers with residual connections. The final prediction is made using a linear layer.
    """

    def __init__(self, config: dict | str, num_splicing_factors, has_gene_exp_values):
        super().__init__()
        if isinstance(config, str):
            with open(config, "r") as f:
                config = json.load(f)
        self.config = config
        self.input_size = config["train_config"]["input_size"]
        assert (
            self.input_size == 10000
        ), "The input size should be 10000 for SpliceAI-10k model."
        self.predict_mean_std_psi_and_delta = self.config["train_config"][
            "predict_mean_std_psi_and_delta"
        ]
        self.predict_mean_psi_and_delta = self.config["train_config"][
            "predict_mean_psi_and_delta"
        ]
        self.predict_controls_avg_psi_and_delta = self.config["train_config"][
            "predict_controls_avg_psi_and_delta"
        ]
        self.predict_logits = "Logits" in self.config["train_config"]["loss_fn"]
        if "use_features_from_alt_sequence" not in self.config["train_config"]:
            self.use_features_from_alt_sequence = False
        else:
            self.use_features_from_alt_sequence = self.config["train_config"][
                "use_features_from_alt_sequence"
            ]
            print(
                f"Using features from alternate sequence: {self.use_features_from_alt_sequence}"
            )

        self.num_splicing_factors = num_splicing_factors
        self.has_gene_exp_values = has_gene_exp_values
        self.conditioning_dim = (
            num_splicing_factors + (1 if has_gene_exp_values else 0) + 4
        )  # +4 for event type one-hot encoding

        self.condition_dropout = nn.Dropout(0.1)
        self.conv1 = nn.Conv1d(
            in_channels=6,
            out_channels=32,
            kernel_size=1,
            stride=1,
            padding="same",
            bias=True,
            dilation=1,
        )  # 4 for one-hot encoding of DNA sequence, 2 for masks
        self.film1 = FiLM(self.conditioning_dim, 32)
        self.side_conv1 = nn.Conv1d(
            in_channels=32,
            out_channels=32,
            kernel_size=1,
            stride=1,
            padding="same",
            bias=True,
            dilation=1,
        )
        self.resblocks1 = nn.ModuleList()
        for i in range(4):
            self.resblocks1.append(
                ResidualBlock(
                    in_channels=32,
                    out_channels=32,
                    kernel_size=11,
                    dilation=1,
                    use_film=self.conditioning_dim > 0,
                    conditioning_dim=self.conditioning_dim,
                )
            )

        self.side_conv2 = nn.Conv1d(
            in_channels=32,
            out_channels=32,
            kernel_size=1,
            stride=1,
            padding="same",
            bias=True,
            dilation=1,
        )
        self.resblocks2 = nn.ModuleList()
        for i in range(4):
            self.resblocks2.append(
                ResidualBlock(
                    in_channels=32,
                    out_channels=32,
                    kernel_size=11,
                    dilation=4,
                    use_film=self.conditioning_dim > 0,
                    conditioning_dim=self.conditioning_dim,
                )
            )

        self.side_conv3 = nn.Conv1d(
            in_channels=32,
            out_channels=32,
            kernel_size=1,
            stride=1,
            padding="same",
            bias=True,
            dilation=1,
        )
        self.resblocks3 = nn.ModuleList()
        for i in range(4):
            self.resblocks3.append(
                ResidualBlock(
                    in_channels=32,
                    out_channels=32,
                    kernel_size=21,
                    dilation=10,
                    use_film=self.conditioning_dim > 0,
                    conditioning_dim=self.conditioning_dim,
                )
            )

        self.side_conv4 = nn.Conv1d(
            in_channels=32,
            out_channels=32,
            kernel_size=1,
            stride=1,
            padding="same",
            bias=True,
            dilation=1,
        )
        self.resblocks4 = nn.ModuleList()
        for i in range(4):
            self.resblocks4.append(
                ResidualBlock(
                    in_channels=32,
                    out_channels=32,
                    kernel_size=41,
                    dilation=25,
                    use_film=self.conditioning_dim > 0,
                    conditioning_dim=self.conditioning_dim,
                )
            )

        self.conv2 = nn.Conv1d(
            in_channels=32,
            out_channels=32,
            kernel_size=1,
            stride=1,
            padding="same",
            bias=True,
            dilation=1,
        )

        self.global_avg_pool = nn.AdaptiveAvgPool1d(1)
        if self.predict_mean_std_psi_and_delta:
            self.mean_std_output_layer = nn.Linear(32, 2)
        if self.predict_mean_psi_and_delta:
            self.mean_output_layer = nn.Linear(32, 1)
        if self.predict_controls_avg_psi_and_delta:
            self.controls_avg_output_layer = nn.Linear(32, 1)
        self.output_layer = nn.Linear(32 + self.conditioning_dim, 1)

    def forward(self, batch):
        sequence = F.one_hot(batch["sequence"].long(), 5)  # (B, 10000, 5)
        sequence = sequence[:, :, :4].float()  # (B, 10000, 4) - remove N
        spliced_in_mask = batch["spliced_in_mask"].float()  # (B, 10000)
        spliced_out_mask = batch["spliced_out_mask"].float()  # (B, 10000)
        gene_exp = batch["gene_exp"]  # (B,)
        splicing_factor_exp_values = batch[
            "splicing_factor_exp_values"
        ]  # (B, num_splicing_factors)
        event_type = batch["event_type"]
        event_type_one_hot = F.one_hot(event_type.long(), 4)  # (B, 4)

        spliced_in_mask = spliced_in_mask.unsqueeze(-1)  # (B, 10000, 1)
        spliced_out_mask = spliced_out_mask.unsqueeze(-1)  # (B, 10000, 1)
        x = torch.cat(
            [sequence, spliced_in_mask, spliced_out_mask], dim=2
        )  # (B, 10000, 6)
        x = x.permute(0, 2, 1)  # (B, 6, 10000)

        gene_exp = gene_exp.unsqueeze(-1)  # (B, 1)
        conditioning = []
        if self.num_splicing_factors > 0:
            conditioning.append(self.condition_dropout(splicing_factor_exp_values))
        if self.has_gene_exp_values:
            conditioning.append(self.condition_dropout(gene_exp))
        conditioning.append(event_type_one_hot)
        if len(conditioning) > 1:  # (B, conditioning_dim)
            conditioning = torch.cat(conditioning, dim=1)
        else:
            conditioning = conditioning[0].float()

        x = self.conv1(x)
        x = self.film1(x, conditioning)
        side = self.side_conv1(x)

        for resblock in self.resblocks1:
            x = resblock(x, conditioning)
        side = side + self.side_conv2(x)

        for resblock in self.resblocks2:
            x = resblock(x, conditioning)
        side = side + self.side_conv3(x)

        for resblock in self.resblocks3:
            x = resblock(x, conditioning)
        side = side + self.side_conv4(x)

        for resblock in self.resblocks4:
            x = resblock(x, conditioning)

        x = self.conv2(x)
        x = x + side  # (B, 32, 10000)

        # if self.use_features_from_alt_sequence is True, we only keep the features from the alternate sequence positions and average over those positions
        # the alternate sequence is where the spliced-in mask is 1 and the spliced-out mask is -1
        if self.use_features_from_alt_sequence:
            alt_sequence_mask = (
                spliced_in_mask - spliced_out_mask
            ).float()  # (B, 10000, 1)
            alt_sequence_mask = alt_sequence_mask.permute(0, 2, 1)  # (B, 1, 10000)
            x = x * alt_sequence_mask  # (B, 32, 10000)
            x = x.sum(dim=2)  # (B, 32)
            counts = alt_sequence_mask.sum(dim=2)  # (B, 1)
            counts = counts + 1  # to avoid division by zero
            x = x / counts  # (B, 32) - average over the alternate sequence positions
        else:
            x = self.global_avg_pool(x).reshape(x.shape[0], -1)

        if self.predict_mean_std_psi_and_delta:
            x_mean_std = self.mean_std_output_layer(x)
            if not self.predict_logits:
                x_mean_std = F.sigmoid(
                    x_mean_std
                )  # (B, 2) - first value is mean, second value is std
        if self.predict_mean_psi_and_delta:
            x_mean = self.mean_output_layer(x)
            if not self.predict_logits:
                x_mean = F.sigmoid(x_mean)
        if self.predict_controls_avg_psi_and_delta:
            x_controls_avg = self.controls_avg_output_layer(x)
            if not self.predict_logits:
                x_controls_avg = F.sigmoid(x_controls_avg)
        x = torch.cat([x, conditioning], dim=1)
        x = self.output_layer(x)
        if not self.predict_logits:
            x = F.sigmoid(x).reshape(-1)
        else:
            x = x.reshape(-1)

        if self.predict_mean_std_psi_and_delta:
            x = torch.cat(
                [x.unsqueeze(1), x_mean_std], dim=1
            )  # (B, 3) - first value is delta psi, second value is mean, third value is std
        if self.predict_mean_psi_and_delta:
            x = torch.cat(
                [x.unsqueeze(1), x_mean], dim=1
            )  # (B, 2) - first value is delta psi, second value is mean
        if self.predict_controls_avg_psi_and_delta:
            x = torch.cat([x.unsqueeze(1), x_controls_avg], dim=1)
            # (B, 2) - first value is delta psi, second value is controls avg psi
        return x


class LargeSpliceAI10k(nn.Module):
    """
    Takes in a one-hot encoded DNA sequence, two masks denoting the exons, introns and background regions of the spliced-in (alternate segment retained) and spliced-out (alternate segment removed) events,
    a splicing factor expression levels vector, a host gene expression value. The model predicts the percent spliced-in (PSI) value of the event.

    The model architecture is based on SpliceAI-10k and uses a stack of dilated convolutional layers with residual connections. The final prediction is made using a linear layer.
    """

    def __init__(self, config: dict | str, num_splicing_factors, has_gene_exp_values):
        super().__init__()
        if isinstance(config, str):
            with open(config, "r") as f:
                config = json.load(f)
        self.config = config
        self.input_size = config["train_config"]["input_size"]
        assert (
            self.input_size == 10000
        ), "The input size should be 10000 for SpliceAI-10k model."
        self.predict_mean_std_psi_and_delta = self.config["train_config"][
            "predict_mean_std_psi_and_delta"
        ]
        self.predict_mean_psi_and_delta = self.config["train_config"][
            "predict_mean_psi_and_delta"
        ]
        self.predict_controls_avg_psi_and_delta = self.config["train_config"][
            "predict_controls_avg_psi_and_delta"
        ]
        self.predict_logits = "Logits" in self.config["train_config"]["loss_fn"]
        if "use_features_from_alt_sequence" not in self.config["train_config"]:
            self.use_features_from_alt_sequence = False
        else:
            self.use_features_from_alt_sequence = self.config["train_config"][
                "use_features_from_alt_sequence"
            ]
            print(
                f"Using features from alternate sequence: {self.use_features_from_alt_sequence}"
            )

        self.num_splicing_factors = num_splicing_factors
        self.has_gene_exp_values = has_gene_exp_values
        self.conditioning_dim = (
            num_splicing_factors + (1 if has_gene_exp_values else 0) + 4
        )  # +4 for event type one-hot encoding

        self.condition_dropout = nn.Dropout(0.1)
        self.conv1 = nn.Conv1d(
            in_channels=6,
            out_channels=32,
            kernel_size=1,
            stride=1,
            padding="same",
            bias=True,
            dilation=1,
        )  # 4 for one-hot encoding of DNA sequence, 2 for masks
        self.film1 = FiLM(self.conditioning_dim, 32)
        self.resblocks1 = nn.ModuleList()
        for i in range(4):
            self.resblocks1.append(
                ResidualBlock(
                    in_channels=32,
                    out_channels=32,
                    kernel_size=11,
                    dilation=1,
                    use_film=self.conditioning_dim > 0,
                    conditioning_dim=self.conditioning_dim,
                )
            )

        self.resblocks2 = nn.ModuleList()
        self.resblocks2.append(
            ResidualBlock(
                in_channels=32,
                out_channels=64,
                kernel_size=11,
                dilation=4,
                use_film=self.conditioning_dim > 0,
                conditioning_dim=self.conditioning_dim,
            )
        )
        for i in range(3):
            self.resblocks2.append(
                ResidualBlock(
                    in_channels=64,
                    out_channels=64,
                    kernel_size=11,
                    dilation=4,
                    use_film=self.conditioning_dim > 0,
                    conditioning_dim=self.conditioning_dim,
                )
            )

        self.resblocks3 = nn.ModuleList()
        self.resblocks3.append(
            ResidualBlock(
                in_channels=64,
                out_channels=128,
                kernel_size=21,
                dilation=10,
                use_film=self.conditioning_dim > 0,
                conditioning_dim=self.conditioning_dim,
            )
        )
        for i in range(3):
            self.resblocks3.append(
                ResidualBlock(
                    in_channels=128,
                    out_channels=128,
                    kernel_size=21,
                    dilation=10,
                    use_film=self.conditioning_dim > 0,
                    conditioning_dim=self.conditioning_dim,
                )
            )

        self.resblocks4 = nn.ModuleList()
        self.resblocks4.append(
            ResidualBlock(
                in_channels=128,
                out_channels=256,
                kernel_size=41,
                dilation=25,
                use_film=self.conditioning_dim > 0,
                conditioning_dim=self.conditioning_dim,
            )
        )
        for i in range(3):
            self.resblocks4.append(
                ResidualBlock(
                    in_channels=256,
                    out_channels=256,
                    kernel_size=41,
                    dilation=25,
                    use_film=self.conditioning_dim > 0,
                    conditioning_dim=self.conditioning_dim,
                )
            )

        self.global_avg_pool = nn.AdaptiveAvgPool1d(1)
        if self.predict_mean_std_psi_and_delta:
            self.mean_std_output_layer = nn.Sequential(
                nn.Linear(256, 512),
                nn.ReLU(),
                nn.Linear(512, 2),
            )
        if self.predict_mean_psi_and_delta:
            self.mean_output_layer = nn.Sequential(
                nn.Linear(256, 512),
                nn.ReLU(),
                nn.Linear(512, 1),
            )
        if self.predict_controls_avg_psi_and_delta:
            self.controls_avg_output_layer = nn.Sequential(
                nn.Linear(256, 512),
                nn.ReLU(),
                nn.Linear(512, 1),
            )
        self.output_layer = nn.Sequential(
            nn.Linear(256 + self.conditioning_dim, 512),
            nn.ReLU(),
            nn.Linear(512, 1),
        )

    def forward(self, batch):
        sequence = F.one_hot(batch["sequence"].long(), 5)  # (B, 10000, 5)
        sequence = sequence[:, :, :4].float()  # (B, 10000, 4) - remove N
        spliced_in_mask = batch["spliced_in_mask"].float()  # (B, 10000)
        spliced_out_mask = batch["spliced_out_mask"].float()  # (B, 10000)
        gene_exp = batch["gene_exp"]  # (B,)
        splicing_factor_exp_values = batch[
            "splicing_factor_exp_values"
        ]  # (B, num_splicing_factors)
        event_type = batch["event_type"]
        event_type_one_hot = F.one_hot(event_type.long(), 4)  # (B, 4)

        spliced_in_mask = spliced_in_mask.unsqueeze(-1)  # (B, 10000, 1)
        spliced_out_mask = spliced_out_mask.unsqueeze(-1)  # (B, 10000, 1)
        x = torch.cat(
            [sequence, spliced_in_mask, spliced_out_mask], dim=2
        )  # (B, 10000, 6)
        x = x.permute(0, 2, 1)  # (B, 6, 10000)

        gene_exp = gene_exp.unsqueeze(-1)  # (B, 1)
        conditioning = []
        if self.num_splicing_factors > 0:
            conditioning.append(self.condition_dropout(splicing_factor_exp_values))
        if self.has_gene_exp_values:
            conditioning.append(self.condition_dropout(gene_exp))
        conditioning.append(event_type_one_hot)
        if len(conditioning) > 1:  # (B, conditioning_dim)
            conditioning = torch.cat(conditioning, dim=1)
        else:
            conditioning = conditioning[0].float()

        x = self.conv1(x)
        x = self.film1(x, conditioning)

        for resblock in self.resblocks1:
            x = resblock(x, conditioning)

        for resblock in self.resblocks2:
            x = resblock(x, conditioning)

        for resblock in self.resblocks3:
            x = resblock(x, conditioning)

        for resblock in self.resblocks4:
            x = resblock(x, conditioning)

        # if self.use_features_from_alt_sequence is True, we only keep the features from the alternate sequence positions and average over those positions
        # the alternate sequence is where the spliced-in mask is 1 and the spliced-out mask is -1
        if self.use_features_from_alt_sequence:
            alt_sequence_mask = (
                spliced_in_mask - spliced_out_mask
            ).float()  # (B, 10000, 1)
            alt_sequence_mask = alt_sequence_mask.permute(0, 2, 1)  # (B, 1, 10000)
            alt_sequence_mask = F.avg_pool1d(
                alt_sequence_mask, kernel_size=11, stride=2, padding="same"
            )
            alt_sequence_mask = F.avg_pool1d(
                alt_sequence_mask, kernel_size=21, stride=2, padding="same"
            )
            alt_sequence_mask = F.avg_pool1d(
                alt_sequence_mask, kernel_size=41, stride=2, padding="same"
            )
            x = x * alt_sequence_mask  # (B, 32, 10000)
            x = x.sum(dim=2)  # (B, 32)
            counts = alt_sequence_mask.sum(dim=2)  # (B, 1)
            counts = counts + 1  # to avoid division by zero
            x = x / counts  # (B, 32) - average over the alternate sequence positions
        else:
            x = self.global_avg_pool(x).reshape(x.shape[0], -1)

        if self.predict_mean_std_psi_and_delta:
            x_mean_std = self.mean_std_output_layer(x)
            if not self.predict_logits:
                x_mean_std = F.sigmoid(
                    x_mean_std
                )  # (B, 2) - first value is mean, second value is std
        if self.predict_mean_psi_and_delta:
            x_mean = self.mean_output_layer(x)
            if not self.predict_logits:
                x_mean = F.sigmoid(x_mean)
        if self.predict_controls_avg_psi_and_delta:
            x_controls_avg = self.controls_avg_output_layer(x)
            if not self.predict_logits:
                x_controls_avg = F.sigmoid(x_controls_avg)
        x = torch.cat([x, conditioning], dim=1)
        x = self.output_layer(x)
        if not self.predict_logits:
            x = F.sigmoid(x).reshape(-1)
        else:
            x = x.reshape(-1)

        if self.predict_mean_std_psi_and_delta:
            x = torch.cat(
                [x.unsqueeze(1), x_mean_std], dim=1
            )  # (B, 3) - first value is delta psi, second value is mean, third value is std
        if self.predict_mean_psi_and_delta:
            x = torch.cat(
                [x.unsqueeze(1), x_mean], dim=1
            )  # (B, 2) - first value is delta psi, second value is mean
        if self.predict_controls_avg_psi_and_delta:
            x = torch.cat([x.unsqueeze(1), x_controls_avg], dim=1)
            # (B, 2) - first value is delta psi, second value is controls avg psi
        return x
