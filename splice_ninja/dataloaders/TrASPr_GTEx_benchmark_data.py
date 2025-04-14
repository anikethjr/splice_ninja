import numpy as np
import pandas as pd
import os
import pdb
import json
import urllib
from tqdm import tqdm
from scipy import stats
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


def worker_init_fn(worker_id):
    worker_info = torch.utils.data.get_worker_info()
    worker_info.dataset.genome = genomepy.Genome(
        worker_info.dataset.genome_name, genomes_dir=worker_info.dataset.genomes_dir
    )


class TrASPrGTExBenchmarkData(LightningDataModule):
    def prepare_data(self):
        # download the genome if it does not exist
        if not os.path.exists(os.path.join(self.cache_dir, "genomes", "GRCh38.p14")):
            print("Downloading the genome")
            os.makedirs(os.path.join(self.cache_dir, "genomes"), exist_ok=True)
            # download ENSEMBL GRChg38 v112 GTF since that was the version used for the gene counts
            genomepy.install_genome(
                name="GRCh38.p14",
                provider="Ensembl",
                annotation=True,
                version=112,
                genomes_dir=os.path.join(self.cache_dir, "genomes"),
            )
            print("Genome downloaded")

        # download the core benchmark data if it does not exist
        if not os.path.exists(
            os.path.join(
                self.cache_dir,
                "TrASPr_GTEx_benchmark_data",
                "TrASPr_GTEx_PSI_values.tsv",
            )
        ):
            print("Downloading the core benchmark data")
            os.makedirs(
                os.path.join(self.cache_dir, "TrASPr_GTEx_benchmark_data"),
                exist_ok=True,
            )
            url = "https://raw.githubusercontent.com/AstroSign/TrASPr_model/refs/heads/main/plot_script/data/GTEx_data.tsv"
            urllib.request.urlretrieve(
                url,
                os.path.join(
                    self.cache_dir,
                    "TrASPr_GTEx_benchmark_data",
                    "TrASPr_GTEx_PSI_values.tsv",
                ),
            )
            print("Core benchmark data downloaded")

        # download event information, gene information, and event ID to gene ID mapping from VastDB
        os.makedirs(os.path.join(self.cache_dir, "VastDB", "hg38"), exist_ok=True)
        if not os.path.exists(
            os.path.join(self.cache_dir, "VastDB", "hg38", "EVENT_INFO-hg38.tab.gz")
        ):
            print("Downloading event information from VastDB")
            url = "https://vastdb.crg.eu/downloads/hg38/EVENT_INFO-hg38.tab.gz"
            urllib.request.urlretrieve(
                url,
                os.path.join(
                    self.cache_dir, "VastDB", "hg38", "EVENT_INFO-hg38.tab.gz"
                ),
            )
            print("Event information downloaded")

        # prepare the final benchmark data if it does not exist
        if not os.path.exists(
            os.path.join(self.cache_dir, "TrASPr_GTEx_benchmark_data", "final.csv")
        ):
            core_benchmark_data = pd.read_csv(
                os.path.join(
                    self.cache_dir,
                    "TrASPr_GTEx_benchmark_data",
                    "TrASPr_GTEx_PSI_values.tsv",
                ),
                sep="\t",
            )
            print(
                "Core benchmark data loaded, number of examples: ",
                core_benchmark_data.shape[0],
            )

            # join with VAST-DB event information to get information needed for input construction
            event_info = pd.read_csv(
                os.path.join(
                    self.cache_dir, "VastDB", "hg38", "EVENT_INFO-hg38.tab.gz"
                ),
                sep="\t",
            )
            core_benchmark_data["CO_A"] = core_benchmark_data.apply(
                lambda x: f"{x['Chr']}:{x['Exon_start']}-{x['Exon_end']}", axis=1
            )
            core_benchmark_data = core_benchmark_data.merge(
                event_info, how="inner", on="CO_A"
            )
            print(
                "Core benchmark data joined with event information, number of examples: ",
                core_benchmark_data.shape[0],
            )

            # create spliced in and out segments
            core_benchmark_data_final = {}
            core_benchmark_data_final["EVENT"] = []
            core_benchmark_data_final["CHANGE_CASE"] = []
            core_benchmark_data_final["TISSUE"] = []
            core_benchmark_data_final["PSI"] = []
            core_benchmark_data_final["GENE_ID"] = []
            core_benchmark_data_final["CHR"] = []
            core_benchmark_data_final["STRAND"] = []
            core_benchmark_data_final["SPLICED_IN_EVENT_SEGMENTS"] = []
            core_benchmark_data_final["SPLICED_OUT_EVENT_SEGMENTS"] = []
            core_benchmark_data_final["FULL_EVENT_COORD"] = []

            for i in range(core_benchmark_data.shape[0]):
                row = core_benchmark_data.iloc[i]
                core_benchmark_data_final["EVENT"].append(row["ID"])
                core_benchmark_data_final["CHANGE_CASE"].append(row["Change_case"])
                core_benchmark_data_final["TISSUE"].append(row["Tissue"])
                core_benchmark_data_final["PSI"].append(row["Label"])
                core_benchmark_data_final["GENE_ID"].append(row["gene_id"])
                core_benchmark_data_final["CHR"].append(row["Chr"])
                core_benchmark_data_final["STRAND"].append(
                    "." if row["Strand"] == "+" else "-"
                )
                core_benchmark_data_final["FULL_EVENT_COORD"].append(row["FullCO"])

                # get the spliced in and out segments
                if row["Strand"] == "+":
                    upstream_exon_coordinates = row["CO_C1"]
                    alternate_exon_coordinates = row["CO_A"]
                    downstream_exon_coordinates = row["CO_C2"]

                    spliced_in_event_segments = [
                        upstream_exon_coordinates,
                        alternate_exon_coordinates,
                        downstream_exon_coordinates,
                    ]
                    spliced_out_event_segments = [
                        upstream_exon_coordinates,
                        downstream_exon_coordinates,
                    ]
                else:
                    upstream_exon_coordinates = row["CO_C1"]
                    alternate_exon_coordinates = row["CO_A"]
                    downstream_exon_coordinates = row["CO_C2"]

                    # the order of the exons is reversed for the "-" strand to maintain the 5' to 3' direction
                    spliced_in_event_segments = [
                        downstream_exon_coordinates,
                        alternate_exon_coordinates,
                        upstream_exon_coordinates,
                    ]
                    spliced_out_event_segments = [
                        downstream_exon_coordinates,
                        upstream_exon_coordinates,
                    ]

                core_benchmark_data_final["SPLICED_IN_EVENT_SEGMENTS"].append(
                    ",".join(spliced_in_event_segments)
                )
                core_benchmark_data_final["SPLICED_OUT_EVENT_SEGMENTS"].append(
                    ",".join(spliced_out_event_segments)
                )

            core_benchmark_data_final.to_csv(
                os.path.join(self.cache_dir, "TrASPr_GTEx_benchmark_data", "final.csv"),
                index=False,
            )
            print(
                "Final benchmark data prepared, number of examples: ",
                core_benchmark_data_final.shape[0],
            )

        # now get splicing factor expression values in GTEx tissues
        if not os.path.exists(
            os.path.join(
                self.cache_dir,
                "TrASPr_GTEx_benchmark_data",
                "splicing_factor_expression_values.parquet",
            )
        ):
            assert os.path.exists(
                os.path.join(self.cache_dir, "normalized_gene_expression.parquet")
            ), "Normalized gene expression file from knockdown data not found. Please run the knockdown data preparation first."
            assert os.path.exists(
                os.path.join(
                    self.cache_dir, "splicing_factor_expression_levels.parquet"
                )
            ), "Splicing factor expression levels file from knockdown data not found. Please run the knockdown data preparation first."

            # load processed expression data from knockdown data
            knockdown_data_normalized_gene_expression = pd.read_parquet(
                os.path.join(self.cache_dir, "normalized_gene_expression.parquet")
            )
            knockdown_data_splicing_factor_expression_levels = pd.read_parquet(
                os.path.join(
                    self.cache_dir, "splicing_factor_expression_levels.parquet"
                )
            )

            # load the core benchmark data
            core_benchmark_data = pd.read_csv(
                os.path.join(self.cache_dir, "TrASPr_GTEx_benchmark_data", "final.csv"),
            )
            unique_tissues = core_benchmark_data["Tissue"].unique()

            # download gene counts for each tissue
            for tissue in unique_tissues:
                if not os.path.exists(
                    os.path.join(
                        self.cache_dir,
                        "TrASPr_GTEx_benchmark_data",
                        f"gene_reads_v10_{tissue.lower()}.gct.gz",
                    )
                ):
                    print(f"Downloading gene counts for {tissue}")
                    url = f"https://storage.googleapis.com/adult-gtex/bulk-gex/v10/rna-seq/counts-by-tissue/gene_reads_v10_{tissue.lower()}.gct.gz"
                    urllib.request.urlretrieve(
                        url,
                        os.path.join(
                            self.cache_dir,
                            "TrASPr_GTEx_benchmark_data",
                            f"gene_reads_v10_{tissue.lower()}.gct.gz",
                        ),
                    )
                    print(f"Gene counts for {tissue} downloaded")

            # now load the gene counts for each tissue and get the expression values for the splicing factors
            all_tissues_median_expression_from_counts = pd.DataFrame()
            all_tissues_median_expression_from_counts[
                "gene_id"
            ] = knockdown_data_normalized_gene_expression["gene_id"]
            all_tissues_median_expression_from_counts[
                "length"
            ] = knockdown_data_normalized_gene_expression["length"]

            for tissue in unique_tissues:
                print(f"Loading gene counts for {tissue}")
                tissue_gene_counts = pd.read_csv(
                    os.path.join(
                        self.cache_dir,
                        "TrASPr_GTEx_benchmark_data",
                        f"gene_reads_v10_{tissue.lower()}.gct.gz",
                    ),
                    sep="\t",
                    skiprows=2,
                )
                tissue_gene_counts["gene_id"] = (
                    tissue_gene_counts["Name"].str.split(".").str[0]
                )
                tissue_gene_counts = tissue_gene_counts.drop_duplicates(
                    subset=["gene_id"]
                )
                tissue_gene_counts = tissue_gene_counts.merge(
                    knockdown_data_normalized_gene_expression[["gene_id", "length"]],
                    how="inner",
                    on="gene_id",
                )

                tissue_samples = []
                for col in tissue_gene_counts.columns[2:-2]:
                    assert col.startswith("GTEX-")
                    tissue_gene_counts[col + "_TPM"] = (
                        tissue_gene_counts[col] / tissue_gene_counts["length"]
                    )
                    rpk_sum = tissue_gene_counts[col + "_TPM"].sum()
                    tissue_gene_counts[col + "_TPM"] = (
                        tissue_gene_counts[col + "_TPM"] / rpk_sum
                    ) * 1e6
                    tissue_gene_counts[col + "_log2TPM"] = np.log2(
                        tissue_gene_counts[col + "_TPM"] + 1
                    )
                    tissue_gene_counts = tissue_gene_counts.copy()
                    tissue_samples.append(col + "_log2TPM")
                tissue_gene_counts[f"{tissue}_log2TPM"] = np.median(
                    tissue_gene_counts[tissue_samples].values, axis=1
                )

                all_tissues_median_expression_from_counts = (
                    all_tissues_median_expression_from_counts.merge(
                        tissue_gene_counts[["gene_id", f"{tissue}_log2TPM"]],
                        how="inner",
                        on="gene_id",
                    )
                )

            # now compute the splicing factor relative normalized expression levels
            all_tissues_splicing_factor_expression_levels = (
                knockdown_data_splicing_factor_expression_levels[
                    ["gene_id", "AV_Controls_log2TPM_rel_norm"]
                ].copy()
            )
            all_tissues_splicing_factor_expression_levels = (
                all_tissues_splicing_factor_expression_levels.merge(
                    all_tissues_median_expression_from_counts[
                        ["gene_id"] + [f"{tissue}_log2TPM" for tissue in unique_tissues]
                    ],
                    how="left",
                    on="gene_id",
                )
            )
            for tissue in unique_tissues:
                all_tissues_splicing_factor_expression_levels[
                    f"{tissue}_log2TPM"
                ] = all_tissues_splicing_factor_expression_levels[
                    f"{tissue}_log2TPM"
                ].fillna(
                    0
                )
                all_tissues_splicing_factor_expression_levels[
                    f"{tissue}_log2TPM_rel_norm"
                ] = (
                    all_tissues_splicing_factor_expression_levels[f"{tissue}_log2TPM"]
                    / all_tissues_splicing_factor_expression_levels[
                        f"{tissue}_log2TPM"
                    ].sum()
                )
            all_tissues_splicing_factor_expression_levels = (
                all_tissues_splicing_factor_expression_levels.copy()
            )

            for tissue in unique_tissues:
                spr = stats.spearmanr(
                    all_tissues_splicing_factor_expression_levels[
                        f"{tissue}_log2TPM_rel_norm"
                    ],
                    all_tissues_splicing_factor_expression_levels[
                        "AV_Controls_log2TPM_rel_norm"
                    ],
                )[0]
                pr = stats.pearsonr(
                    all_tissues_splicing_factor_expression_levels[
                        f"{tissue}_log2TPM_rel_norm"
                    ],
                    all_tissues_splicing_factor_expression_levels[
                        "AV_Controls_log2TPM_rel_norm"
                    ],
                )[0]
                print(
                    f"Correlation between {tissue} and AV_Controls SF log2TPM_rel_norm: {spr:.4f} (Spearman), {pr:.4f} (Pearson)"
                )

            all_tissues_splicing_factor_expression_levels.to_parquet(
                os.path.join(
                    self.cache_dir,
                    "TrASPr_GTEx_benchmark_data",
                    "splicing_factor_expression_values.parquet",
                ),
                index=False,
            )

    def setup(self, stage: str = None):
        # load the core benchmark data
        self.core_benchmark_data = pd.read_csv(
            os.path.join(self.cache_dir, "TrASPr_GTEx_benchmark_data", "final.csv"),
        )
        print(
            "Core benchmark data loaded, number of examples: ",
            self.core_benchmark_data.shape[0],
        )

        # load the splicing factor expression values
        self.splicing_factor_expression_values = pd.read_parquet(
            os.path.join(
                self.cache_dir,
                "TrASPr_GTEx_benchmark_data",
                "splicing_factor_expression_values.parquet",
            ),
        )
