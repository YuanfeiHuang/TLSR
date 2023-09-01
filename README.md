# Transitional Learning: Exploring the Transition States of Degradation for Blind Super-resolution (TLSR)
This repository is for TLSR introduced in the following paper

Yuanfei Huang, Jie Li, Yanting Hu, Xinbo Gao and Hua Huang*, "Transitional Learning: Exploring the Transition States of Degradation for Blind Super-resolution", IEEE TPAMI, 2023, 45(5): 6495-6510.
[paper](https://ieeexplore.ieee.org/document/9893392)
## Dependenices
* python 3.7
* pytorch >= 1.5
* NVIDIA GPU + CUDA

## Models
Download the pre-trained models from [Google Drive](https://drive.google.com/drive/folders/1UpN0Zf6mqYrj6YU9jwB5XnkGNqYjIaTI?usp=sharing) 
or [百度网盘](https://pan.baidu.com/s/1m3maDvSBRufs6rsVwhL1mQ?pwd=ohwt) (提取码: ohwt)

## Data preparing
Download [DIV2K](https://data.vision.ee.ethz.ch/cvl/DIV2K/) and [Flickr2K](https://cv.snu.ac.kr/research/EDSR/Flickr2K.tar) datasets into the path "data/Datasets/Train/DF2K". 

## Settings (option.py)
For convolutive degradations (isotropic Gaussian):
* '-scale' == 2
* '-degrad_train' == {'type': 'B', 'min_sigma': 0.2, 'max_sigma': 2.0} # for Training.
* '-degrad_test' == [{'type': 'B', 'sigma': 1.0}] # for Testing.

    OR
* '-scale' == 4
* '-degrad_train' == {'type': 'B', 'min_sigma': 0.2, 'max_sigma': 4.0} # for Training.
* '-degrad_test' == [{'type': 'B', 'sigma': 2.0}] # for Testing.

For convolutive degradations (anisotropic Gaussian):
* '-scale' == 4
* '-degrad_train' == {'type': 'B_aniso', 'min_sigma': 0, 'max_sigma': 0.5}
* '-degrad_test' == [{'type': 'B_aniso', 'sigma': 0.25}] # for evaluation.

For additive degradations:
* '-scale' == 1 OR 2 OR 4
* '-degrad_train' == {'type': 'N', 'min_sigma': 0, 'max_sigma': 30}
* '-degrad_test' == [{'type': 'N', 'sigma': 15}] # for evaluation.

For other degradations:
* '-scale' == 1
* '-degrad_train' == {'type': 'JPEG', 'min_sigma': 10, 'max_sigma': 30}
* '-degrad_test' == [{'type': 'JPEG', 'sigma': 20}] # for evaluation.

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
@ARTICLE{TLSR2022TPAMI,
  author={Huang, Yuanfei and Li, Jie and Hu, Yanting and Gao, Xinbo and Huang, Hua},
  journal={IEEE Transactions on Pattern Analysis and Machine Intelligence}, 
  title={Transitional Learning: Exploring the Transition States of Degradation for Blind Super-Resolution}, 
  year={2023},
  volume={45},
  number={5},
  pages={6495-6510},
  doi={10.1109/TPAMI.2022.3206870}}
```
