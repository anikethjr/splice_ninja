import numpy as np
import pandas as pd
import os
import pdb
import json

import torch
import torch.nn as nn
import torch.nn.functional as F

import einops
from rotary_embedding_torch import RotaryEmbedding

np.random.seed(0)
torch.manual_seed(0)


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
    ):
        super().__init__()

        stride_for_conv1_and_shortcut = 1

        if in_channels != out_channels:
            stride_for_conv1_and_shortcut = 2

        if gn_num_groups is None:
            gn_num_groups = out_channels // gn_group_size

        # modules for processing the input
        self.gn1 = nn.GroupNorm(gn_num_groups, out_channels)
        self.relu1 = nn.GELU()
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
        self.relu2 = nn.GELU()
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

    def forward(self, xl):
        input = self.shortcut(xl)

        xl = self.conv1(self.relu1(self.gn1(xl)))
        xl = self.conv2(self.relu2(self.gn2(xl)))

        xlp1 = input + xl

        return xlp1


class TransformerBlock(nn.Module):
    def __init__(
        self, d_model, nhead, mlp_dim, dropout=0.1, use_position_embedding=True
    ):
        assert d_model % nhead == 0
        super().__init__()
        embedding_dim = d_model
        self.embedding_dim = embedding_dim
        self.num_heads = nhead
        self.mlp_dim = mlp_dim
        self.dropout = dropout
        self.use_position_embedding = use_position_embedding

        self.layer_norm1 = nn.LayerNorm(self.embedding_dim)
        self.xk = nn.Linear(embedding_dim, embedding_dim, bias=False)
        self.xq = nn.Linear(embedding_dim, embedding_dim, bias=False)
        self.xv = nn.Linear(embedding_dim, embedding_dim, bias=False)

        if self.use_position_embedding:
            self.rotary_emb = RotaryEmbedding(dim=embedding_dim // self.num_heads)

        self.dropout1 = nn.Dropout(dropout)
        self.fc1 = nn.Linear(embedding_dim, embedding_dim, bias=False)
        self.dropout2 = nn.Dropout(dropout)
        self.layer_norm2 = nn.LayerNorm(self.embedding_dim)
        self.fc2 = nn.Linear(embedding_dim, mlp_dim)
        self.fc3 = nn.Linear(mlp_dim, embedding_dim)
        self.dropout3 = nn.Dropout(dropout)

    def forward(self, inputs):
        x = self.layer_norm1(inputs)
        xk = self.xk(x)
        xq = self.xq(x)
        xv = self.xv(x)

        xk = xk.reshape(
            xk.shape[0],
            xk.shape[1],
            self.num_heads,
            self.embedding_dim // self.num_heads,
        )
        xq = xq.reshape(
            xq.shape[0],
            xq.shape[1],
            self.num_heads,
            self.embedding_dim // self.num_heads,
        )
        xv = xv.reshape(
            xv.shape[0],
            xv.shape[1],
            self.num_heads,
            self.embedding_dim // self.num_heads,
        )

        if self.use_position_embedding:
            # make xq and xk have shape (batch_size, num_heads, seq_len, embedding_dim // num_heads)
            xq = xq.permute(0, 2, 1, 3)
            xk = xk.permute(0, 2, 1, 3)
            xq = self.rotary_emb.rotate_queries_or_keys(xq, seq_dim=2)
            xk = self.rotary_emb.rotate_queries_or_keys(xk, seq_dim=2)
            # make xq and xk have shape (batch_size, seq_len, num_heads, embedding_dim // num_heads)
            xq = xq.permute(0, 2, 1, 3)
            xk = xk.permute(0, 2, 1, 3)

        attention_weights = einops.einsum(xq, xk, "... q h d, ... k h d -> ... h q k")

        attention_weights = attention_weights / np.sqrt(
            self.embedding_dim // self.num_heads
        )
        attention_weights = F.softmax(attention_weights, dim=-1)
        attention_weights = self.dropout1(attention_weights)
        attention_output = einops.einsum(
            attention_weights, xv, "... h q k, ... k h d -> ... q h d"
        )
        attention_output = einops.rearrange(attention_output, "... h d -> ... (h d)")
        attention_output = self.fc1(attention_output)
        attention_output = self.dropout2(attention_output)

        mlp_inputs = attention_output + inputs
        x = self.layer_norm2(mlp_inputs)
        x = self.fc2(x)
        x = F.gelu(x)
        x = self.fc3(x)
        x = self.dropout3(x)
        x = x + mlp_inputs

        return x


class Shuriken(nn.Module):
    """
    Takes in a one-hot encoded DNA sequence, two masks denoting the exons, introns and background regions of the spliced-in (alternate segment retained) and spliced-out (alternate segment removed) events,
    a splicing factor expression levels vector, a host gene expression value. The model predicts the percent spliced-in (PSI) value of the event.

    This model uses CNN layers to extract sequence features before passing them to a Transformer that produces the final embedding. The final prediction is made using a linear layer.
    """

    def __init__(self, config: dict | str, num_splicing_factors, has_gene_exp_values):
        super().__init__()
        if isinstance(config, str):
            with open(config, "r") as f:
                config = json.load(f)
        self.config = config
        self.input_size = config["train_config"]["input_size"]
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

        self.num_splicing_factors = num_splicing_factors
        self.has_gene_exp_values = has_gene_exp_values
        self.conditioning_dim = (
            num_splicing_factors + (1 if has_gene_exp_values else 0) + 4
        )  # +4 for event type one-hot encoding

        self.num_splicing_factors = num_splicing_factors
        self.has_gene_exp_values = has_gene_exp_values
        self.conditioning_dim = (
            num_splicing_factors + (1 if has_gene_exp_values else 0) + 4
        )  # +4 for event type one-hot encoding

        self.condition_dropout = nn.Dropout(0.1)

        self.condition_expansion1 = nn.Linear(self.conditioning_dim, 16)
        self.transformer_blocks1 = nn.ModuleList()
        for i in range(6):
            self.transformer_blocks1.append(
                TransformerBlock(
                    d_model=16,
                    nhead=8,
                    mlp_dim=64,
                    dropout=0.1,
                    use_position_embedding=True,
                )
            )

        self.strided_conv1 = nn.Conv1d(
            in_channels=16,
            out_channels=128,
            kernel_size=128,
            stride=128,
        )

        self.condition_expansion2 = nn.Linear(self.conditioning_dim, 128)
        self.transformer_blocks2 = nn.ModuleList()
        for i in range(6):
            self.transformer_blocks2.append(
                TransformerBlock(
                    d_model=128,
                    nhead=8,
                    mlp_dim=512,
                    dropout=0.1,
                    use_position_embedding=True,
                )
            )

        # Output layers
        if self.predict_mean_std_psi_and_delta:
            self.mean_std_output_layer = nn.Linear(128, 2)
        if self.predict_mean_psi_and_delta:
            self.mean_output_layer = nn.Linear(128, 1)
        if self.predict_controls_avg_psi_and_delta:
            self.controls_avg_output_layer = nn.Linear(128, 1)
        self.output_layer = nn.Linear(128, 1)

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

        # expand conditioning to match the input size to the transformer
        conditioning = self.condition_expansion1(conditioning)  # (B, 16)
        conditioning = conditioning.unsqueeze(1)  # (B, 1, 16)

        # add condition as first token to the sequence
        x = torch.cat([conditioning, x], dim=1)  # (B, 10001, 6)

        for transformer_block in self.transformer_blocks1:
            x = transformer_block(x)

        # strided conv
        x = x[:, 1:, :]  # (B, 10000, 16) - remove the condition token
        x = einops.rearrange(x, "b s d -> b d s")  # (B, 16, 10000)
        x = self.strided_conv1(x)  # (B, 128, 78)
        x = einops.rearrange(x, "b d s -> b s d")  # (B, 128, 78)

        # add condition as first token to the sequence
        conditioning = self.condition_expansion2(conditioning)  # (B, 128)
        conditioning = conditioning.unsqueeze(1)  # (B, 1, 128)
        x = torch.cat([conditioning, x], dim=1)  # (B, 79, 128)

        for transformer_block in self.transformer_blocks2:
            x = transformer_block(x)

        x = x[:, 0]  # (B, 128) - take the first token as the output

        # make the final prediction
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
