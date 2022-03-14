# Transitional Learning: Exploring the Transition States of Degradation for Blind Super-resolution (TLSR)
This repository is for TLSR introduced in the following paper

Yuanfei Huang, Jie Li, Yanting Hu, Xinbo Gao and Hua Huang, "Transitional Learning: Exploring the Transition States of Degradation for Blind Super-resolution", arXiv preprint arXiv:2103.15290(2021).
[arXiv](https://arxiv.org/abs/2103.15290)
## Dependenices
* python 3.7
* pytorch >= 1.5
* NVIDIA GPU + CUDA

## Models

## Data preparing
Download [DIV2K](https://data.vision.ee.ethz.ch/cvl/DIV2K/) and [Flickr2K](http://cv.snu.ac.kr/research/EDSR/Flickr2K.tar) datasets into the path "data/Datasets/Train/DF2K". 

For convolutive degradations:
* '-degrad_train' == {'type': 'B', 'min_sigma': 0.2, 'max_sigma': 4.0}
* '-degrad_test' == [{'type': 'B', 'sigma': 2.0}] # for evaluation.

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
## Citation
```
@misc{huang2021transitional,
      title={Transitional Learning: Exploring the Transition States of Degradation for Blind Super-resolution}, 
      author={Yuanfei Huang and Jie Li and Yanting Hu and Xinbo Gao and Hua Huang},
      year={2021},
      eprint={2103.15290},
      archivePrefix={arXiv},
      primaryClass={cs.CV}
}
```
