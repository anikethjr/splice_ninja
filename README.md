# Sequence models conditioned on splicing factor expression predict splicing in unseen tissues

The code base for our work on building sequence-based splicing models that are conditioned on splicing factor expression levels.

This repository is still under construction.

Our best pretrained model and its predictions are available here - https://huggingface.co/anikethjr/splice_ninja.

## Installation instructions

```bash
git clone https://github.com/anikethjr/splice_ninja.git
cd splice_ninja
mamba env create -f environment.yml
mamba install -c conda-forge -c bioconda 'genomepy>=0.15'
pip install -e .
```
