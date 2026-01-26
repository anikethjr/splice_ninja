# Sequence models conditioned on splicing factor expression predict splicing in unseen tissues

The code base for our work on building sequence-based splicing models that are conditioned on splicing factor expression levels. Please cite the following paper if you use our code or models:
```bibtex
@article {Reddy2026.01.20.700496,
	author = {Reddy, Aniketh Janardhan and Sudmant, Peter H. and Ioannidis, Nilah M.},
	title = {Sequence models conditioned on splicing factor expression predict splicing in unseen tissues},
	elocation-id = {2026.01.20.700496},
	year = {2026},
	doi = {10.64898/2026.01.20.700496},
	publisher = {Cold Spring Harbor Laboratory},
	abstract = {Predicting how RNA splicing varies across tissues is important for understanding the impact of genetic variation and identifying splicing-based disease mechanisms. Although many sequence-based deep learning models have been developed to predict splicing, most predict splice sites rather than full splicing events, are restricted to tissues seen during training, or do not account for trans-regulatory variation such as differences in splicing factor expression. Here, we present Splice Ninja, a sequence-based deep learning model that predicts percent spliced-in (PSI) values for individual splicing events across tissues by conditioning on the expression levels of 301 splicing factors. Trained on PSI measurements from many different human tissues and cell types, Splice Ninja is evaluated on three entirely held-out tissues. Despite not seeing these tissues during training, it makes accurate PSI predictions and can identify a substantial fraction of splicing events with high tissue-specificity. Its performance is comparable to Pangolin [1], which is trained directly on the test tissues, but falls short of TrASPr [2], a substantially larger model also trained on the test tissues. Splice Ninja demonstrates that integrating trans-regulatory context into sequence-based splicing models enables generalization to new cellular environments. This framework offers a promising direction for building robust, context-aware predictors of alternative splicing. Our code is available at https://github.com/anikethjr/splice_ninja.Competing Interest StatementThe authors have declared no competing interest.},
	URL = {https://www.biorxiv.org/content/early/2026/01/21/2026.01.20.700496},
	eprint = {https://www.biorxiv.org/content/early/2026/01/21/2026.01.20.700496.full.pdf},
	journal = {bioRxiv}
}
```

This repository is still under construction.

Our best pretrained model and its predictions are available here - https://huggingface.co/anikethjr/splice_ninja.

## Installation instructions

```bash
git clone https://github.com/anikethjr/splice_ninja.git
cd splice_ninja
mamba env create -f environment.yaml
pip install -e .
```
