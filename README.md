# ImFixMatch: A Simple and Effective Scheme for Imbalanced Semi-Supervised Learning

This repository contains code for the paper
**"ImFixMatch: A Simple and Effective Scheme for Imbalanced Semi-Supervised Learning"** 

## Dependencies

* `python3`
* `pytorch == 1.1.0`
* `torchvision`
* `scipy`
* `randAugment (Pytorch re-implementation: https://github.com/ildoonet/pytorch-randaugment)`

## Training procedure 
```
python train.py --dataset cifar10 --num_max 1500 --label_ratio 2 --imb_ratio_l 100 --imb_ratio_u 100 --gpu-id 0
```

## Evaluate Pre-trainined Model
```
python evaluate_model.py --dataset cifar10 --resume pretrained_model/IFM_1500_2_100_100.pth.tar
```
