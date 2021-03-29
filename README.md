# Transitive Learning: Exploring the Transitivity of Degradations for Blind Super-Resolution (TLSR)
This repository is for TLSR introduced in the following paper

Yuanfei Huang, Jie Li, Yanting Hu, Xinbo Gao* and Wen Lu, "Interpretable Detail-Fidelity Attention Network for Single Image Super-Resolution", arXiv, 2021.
[arXiv]()
## Dependenices
* python 3.7
* pytorch >= 1.5
* NVIDIA GPU + CUDA

## Models

## Data preparing
Download [DIV2K](https://data.vision.ee.ethz.ch/cvl/DIV2K/) datasets into the path "data/Datasets/Train/DIV2K". 

For convolutive degradations:
* '-degrad_train' == {'type': 'B', 'min_sigma': 0.2, 'max_sigma': 2.6}
* '-degrad_test' == [{'type': 'B', 'sigma': 1.3}] # for evaluation.

For additive degradations:
* '-degrad_train' == {'type': 'N', 'min_sigma': 0, 'max_sigma': 30}
* '-degrad_test' == [{'type': 'N', 'sigma': 15}] # for evaluation.

## Train
```bash
python main.py --train 'Train'
```
## Test
```bash
python main.py --train 'Test'
```
