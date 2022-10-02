import argparse
import time
import os
import logging
from tqdm import tqdm

import torch
import numpy as np
import torch.nn.functional as F
from torch.utils.data import DataLoader, SequentialSampler
from sklearn.metrics import precision_score, recall_score

from models.wideresnet import WideResNet
from datasets.load_imb_data import DATASET_GETTERS
from utils.misc import *

logger = logging.getLogger(__name__)

parser = argparse.ArgumentParser(description='Pytorch SSL Library')
parser.add_argument('--gpu-id', default='0', type=int,
                    help='id(s) for CUDA_VISIBE_DEVICES')
parser.add_argument('--num-workers', type=int, default=1,
                    help='number of workers')
parser.add_argument('--dataset', default='cifar10', type=str,
                    choices=['cifar10', 'cifar100'], help='dataset name')
parser.add_argument('--num_classes', type=int, default=10,
                    help='number of classes')
parser.add_argument('--num_max', type=int, default=1500,
                    help='Number of samples in the maximal class')
parser.add_argument('--label_ratio', type=int, default=2,
                    help='Relative size between labeled and unlabeled')
parser.add_argument('--imb_ratio_l', type=int, default=100,
                    help='Imbalance ratio for labeled data')
parser.add_argument('--imb_ratio_u', type=int, default=100,
                    help='Imbalance ratio for unlabeled data')
parser.add_argument('--mu', default=2, type=int,
                    help='coefficient of unlabeled batch size')
parser.add_argument('--arch', default='wideresnet', type=str,
                    help='network architecture')
parser.add_argument('--batch_size', default=64, type=int,
                    help='train batchsize')
parser.add_argument('--use_ema',  default=True, type=bool,
                    help='use EMA model')
parser.add_argument('--resume', default='', type=str,
                    help='path to latest checkpoint (default: none)')

args = parser.parse_args()

np.random.seed(0)
torch.backends.cudnn.benchmark = True


def build_model(args, ema=False):
    if args.arch == 'wideresnet':
        logger.info(f"Model: WideResNet {args.model_depth}x{args.model_width}")
        model = WideResNet(width=args.model_width,
                           num_classes=args.num_classes).to(args.device)
    elif args.arch == 'resnext':
        import models.resnext as models
        model = models.build_resnext(cardinality=args.model_cardinality,
                                     depth=args.model_depth,
                                     width=args.model_width,
                                     num_classes=args.num_classes).to(args.device)
    logger.info("Total params: {:.2f}M".format(
        sum(p.numel() for p in model.parameters()) / 1e6))

    if ema:
        for param in model.parameters():
            param.detach_()
    return model


def test(args, test_loader, model):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()
    end = time.time()

    test_loader = tqdm(test_loader)

    y_pred = []
    y_true = []

    lists = [[] for _ in range(10)]

    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(test_loader):
            data_time.update(time.time() - end)
            model.eval()

            inputs = inputs.to(args.device)
            targets = targets.to(args.device).long()
            outputs = model(inputs)[0]

            pred_label = outputs.max(1)[1]
            y_pred += [pred_label.cpu().numpy()]
            y_true += [targets.cpu().numpy()]

            logits = torch.softmax(outputs.detach(), dim=1)
            max_probs, targets_u = torch.max(logits, dim=1)

            for i in range(targets_u.shape[0]):
                lists[targets_u[i]].append(max_probs[i].detach().cpu().numpy())

            loss = F.cross_entropy(outputs, targets)

            prec1, prec5 = accuracy(outputs, targets, topk=(1, 5))
            losses.update(loss.item(), inputs.shape[0])
            top1.update(prec1.item(), inputs.shape[0])
            top5.update(prec5.item(), inputs.shape[0])
            batch_time.update(time.time() - end)
            end = time.time()
            test_loader.set_description(
                "Test Iter: {batch:4}/{iter:4}. Data: {data:.3f}s. Batch: {bt:.3f}s. Loss: {loss:.4f}. top1: {top1:.2f}. top5: {top5:.2f}. ".format(
                    batch=batch_idx + 1,
                    iter=len(test_loader),
                    data=data_time.avg,
                    bt=batch_time.avg,
                    loss=losses.avg,
                    top1=top1.avg,
                    top5=top5.avg,
                ))
        test_loader.close()

    y_pred = np.array(np.concatenate(y_pred, 0))
    y_true = np.array(np.concatenate(y_true, 0))
    lists = np.array(lists, dtype=object)
    for i in range(10):
        print("class: {}, mean: {}, std: {}".format(i, np.mean(lists[i]), np.std(lists[i])))
    print("test_loss: {}, test_acc: {}".format(losses.avg, top1.avg))
    print("Precision: {}, Recall: {}".format(precision_score(y_true, y_pred, average=None),
                                             recall_score(y_true, y_pred, average=None)))



def main():
    device = torch.device('cuda', args.gpu_id)
    args.world_size = 1
    args.device = device

    if args.dataset == 'cifar10':
        if args.arch == 'wideresnet':
            args.model_depth = 28
            args.model_width = 2
        elif args.arch == 'resnext':
            args.model_cardinality = 4
            args.model_depth = 28
            args.model_width = 4

    cls_num_list, labeled_dataset, unlabeled_dataset, test_dataset = DATASET_GETTERS[args.dataset](
        args, './data')

    test_loader = DataLoader(
        test_dataset, sampler=SequentialSampler(test_dataset),
        batch_size=args.batch_size)

    model = build_model(args)
    ema_model = build_model(args, ema=True)
    logger.info("==> Resuming from checkpoint..")
    assert os.path.isfile(args.resume), "Error: no checkpoint directory found!"
    checkpoint = torch.load(args.resume, map_location='cuda:0')
    model.load_state_dict(checkpoint['state_dict'])
    ema_model.load_state_dict(checkpoint['ema_state_dict'])

    test(args, test_loader, ema_model)


if __name__ == '__main__':
    main()
