# Class-Imbalanced Semi-Supervised Learning with Adaptive Thresholding

This repository contains code for the paper
**"Class-Imbalanced Semi-Supervised Learning with Adaptive Thresholding"** 

## Dependencies

* `python3`
* `pytorch == 1.1.0`
* `torchvision`
* `scipy`
* `randAugment (Pytorch re-implementation: https://github.com/ildoonet/pytorch-randaugment)`

## Training procedure 
```
python train.py --alg adsh --dataset cifar10 --num_max 1500 --label_ratio 2 --imb_ratio_l 100 --imb_ratio_u 100
```

## Evaluate Pre-trained Model
```
python evaluate_model.py --dataset cifar10 --resume pretrained_model/ADSH_1500_2_100_100.pth.tar
```
