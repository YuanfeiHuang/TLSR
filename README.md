This is a reference code of the proposed TLSR method

## Dependenices
* python 3.7
* pytorch >= 1.5
* NVIDIA GPU + CUDA

## Models
As the 100MB limit on the size of the supplementary materials but the DoTNet is based on the ResNet50 (~100MB), we only provide the reference codes in this file and will release the trained models publicly in the future.

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