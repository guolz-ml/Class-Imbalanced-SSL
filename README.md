## Class-Imbalanced Semi-Supervised Learning with Adaptive Thresholding

This repository contains PyTorch implementation for Adsh [Class-Imbalanced Semi-Supervised Learning with Adaptive Thresholding](https://proceedings.mlr.press/v162/guo22e.html) published in ICML 2022.

Adsh is a semi-supervised learning method that can select pseudo-labels based on an adaptive confidence threshold.

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

## Citation

If you find our work useful in your research, please consider citing:

```
@inproceedings{guo2022adsh
  title = 	 {Class-Imbalanced Semi-Supervised Learning with Adaptive Thresholding},
  author =       {Lan-Zhe Guo and Yu-Feng Li},
  booktitle = 	 {Proceedings of the 39th International Conference on Machine Learning},
  pages = 	 {8082--8094},
  year = 	 {2022}
}

```
