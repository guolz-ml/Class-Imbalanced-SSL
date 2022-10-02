import argparse
import math
import time
import os
import logging
import shutil
from tqdm import tqdm

import torch
import numpy as np
import torch.nn.functional as F
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler

from sklearn.metrics import precision_score, recall_score
from models.ema import WeightEMA
from models.wideresnet import WideResNet
from algorithms.fixmatch import FixMatch, ADSH
from datasets.load_imb_data import *
from utils.misc import *
from config import config

logger = logging.getLogger(__name__)

parser = argparse.ArgumentParser(description='Pytorch SSL Library')
parser.add_argument('--gpu-id', default='0', type=int,
                    help='id(s) for CUDA_VISIBE_DEVICES')
parser.add_argument('--alg', default='FM', type=str,
                    choices=['supervised', 'FM', 'adsh'],help='algorithms')
parser.add_argument('--num-workers', type=int, default=4,
                    help='number of workers')
parser.add_argument('--dataset', default='cifar10', type=str,
                    choices=['cifar10', 'cifar100', 'stl10'], help='dataset name')
parser.add_argument('--num_classes', type=int, default=10,
                    help='number of classes')
parser.add_argument('--num_max', type=int, default=1500,
                    help='Number of samples in the maximal class')
parser.add_argument('--imb_ratio_l', type=int, default=100,
                    help='Imbalance ratio for labeled data')
parser.add_argument('--imb_ratio_u', type=int, default=100,
                    help='Imbalance ratio for unlabeled data')
parser.add_argument('--label_ratio', type=float, default=2.0,
                    help='Relative size between labeled and unlabeled')
parser.add_argument('--arch', default='wideresnet', type=str,
                    help='network architecture')
parser.add_argument('--model_depth', default='28', type=int,
                    help='depth of wideresnet')
parser.add_argument('--model_width', default='2', type=int,
                    help='width of wideresnet')
parser.add_argument('--total_steps', default=2**18, type=int,
                    help='number of total steps to run')
parser.add_argument('--eval_steps', default=512, type=int,
                    help='number of eval steps to run per epoch')
parser.add_argument('--batch_size', default=64, type=int,
                    help='train batchsize')
parser.add_argument('--lambda_u', default=1, type=float,
                    help='coefficient of unlabeled loss')
parser.add_argument('--mu', default=2, type=int,
                    help='coefficient of unlabeled batch size')
parser.add_argument('--threshold', default=0.95, type=float,
                    help='threshold for pseduo-label confidence')
parser.add_argument('--optim', default='ADAM', type=str,
                    choices=['SGD', 'ADAM'])
parser.add_argument('--lr', default=0.002, type=float,
                    metavar='LR', help='2e-3 for ADAM and 3e-2 for SGD')
parser.add_argument('--weight_decay', default=0.0005, type=float,
                    help='weight decay of SGD')
parser.add_argument('--nesterov', default=True, type=bool,
                    help='use nesterov')
parser.add_argument('--warmup', default=0, type=float,
                    help='warmup epochs (unlabeled data based)')
parser.add_argument('--use_ema',  default=True, type=bool,
                    help='use EMA model')
parser.add_argument('--ema_decay', default=0.999, type=float,
                    help='EMA decay rate')
parser.add_argument('--out', default='result_imb',
                    help='directory to output the result')
parser.add_argument('--resume', default='', type=str,
                    help='path to latest checkpoint (default: none)')

args = parser.parse_args()

np.random.seed(0)
torch.backends.cudnn.benchmark = True


def save_checkpoint(state, is_best, checkpoint, filename='checkpoint_test.pth.tar'):
    filename = args.alg + '_' + str(args.num_max) + '_' + str(args.label_ratio) + '_' + \
               str(args.imb_ratio_l) + '_' + str(args.imb_ratio_u) + filename
    best_filename = args.alg + '_' + str(args.num_max) + '_' + str(args.label_ratio) + '_' + \
                    str(args.imb_ratio_l) + '_' + str(args.imb_ratio_u) + 'model_best.pth.tar'
    filepath = os.path.join(checkpoint, filename)
    torch.save(state, filepath)
    if is_best:
        shutil.copyfile(filepath, os.path.join(checkpoint,
                                               best_filename))


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


def update_s(args, score, unlabel_loader, model):
    with torch.no_grad():
        lists = [[] for _ in range(args.num_classes)]
        for batch_idx, ((inputs_uw, inputs_us), _) in enumerate(unlabel_loader):
            model.eval()
            inputs_uw = inputs_uw.to(args.device)
            outputs = model(inputs_uw)[0]
            logits = torch.softmax(outputs.detach(), dim=1)
            max_probs, targets_u = torch.max(logits, dim=1)

            for i in range(targets_u.shape[0]):
                lists[targets_u[i]].append(max_probs[i].detach().cpu().numpy())

        lists = np.array(lists, dtype=object)
        rho = 1.0
        for i in range(lists.shape[0]):
            lists[i] = np.sort(np.array(lists[i]))[::-1]
        for i in range(lists[0].shape[0]):
            if lists[0][i] < args.threshold:
                break
            rho = (i + 1) / lists[0].shape[0]
        for i in range(1, lists.shape[0]):
            if lists[i].shape[0] != 0:
                idx = max(0, np.round(lists[i].shape[0] * rho - 1).astype(int))
                score[i] = min(args.threshold, lists[i][idx])
        return score


def train_ssl(label_loader, unlabel_loader, test_loader, ssl_obj):
    args.epochs = math.ceil(args.total_steps / args.eval_steps)

    model = build_model(args)
    ema_model = build_model(args, ema=True)
    ema_optimizer = WeightEMA(model, ema_model, ema_decay=args.ema_decay, lr=args.lr)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    args.start_epoch = 0

    conf = [[] for _ in range(args.num_classes)]
    conf_std = [[] for _ in range(args.num_classes)]
    if args.resume:
        logger.info("==> Resuming from checkpoint..")
        assert os.path.isfile(
            args.resume), "Error: no checkpoint directory found!"
        args.out = os.path.dirname(args.resume)
        checkpoint = torch.load(args.resume)
        args.start_epoch = checkpoint['epoch']
        ema_model.load_state_dict(checkpoint['ema_state_dict'])
        ema_model.ema.load_state_dict(checkpoint['ema_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer'])

    logger.info("***** Running training *****")
    logger.info(f"  Task = {args.dataset}@{args.label_ratio}@{args.imb_ratio_l}")
    logger.info(f"  Num Epochs = {args.epochs}")
    logger.info(f"  Batch size per GPU = {args.batch_size}")
    logger.info(f"  Total optimization steps = {args.total_steps}")

    best_acc = 0.0
    test_accs = []
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    losses_x = AverageMeter()
    losses_u = AverageMeter()
    end = time.time()

    labeled_iter = iter(label_loader)
    unlabeled_iter = iter(unlabel_loader)
    score = np.zeros(args.num_classes) + args.threshold

    for epoch in range(args.start_epoch, args.epochs):
        if args.alg == 'adsh' and epoch > 1:
            score = update_s(args, score, unlabel_loader, model)

        model.train()
        p_bar = tqdm(range(args.eval_steps))
        for batch_idx in p_bar:
            try:
                inputs_l, targets = labeled_iter.next()
            except:
                labeled_iter = iter(label_loader)
                inputs_l, targets = labeled_iter.next()

            try:
                (inputs_u, _) = unlabeled_iter.next()
            except:
                unlabeled_iter = iter(unlabel_loader)
                (inputs_u, _) = unlabeled_iter.next()

            data_time.update(time.time() - end)

            inputs_l = inputs_l.to(args.device)
            targets = targets.to(args.device).long()

            logits = model(inputs_l)[0]
            cls_loss = F.cross_entropy(logits, targets)
            if args.alg == 'supervised':
                ssl_loss = torch.zeros(1).to(args.device)
            elif args.alg == 'adsh':
                ssl_loss = ssl_obj(inputs_u[0], inputs_u[1], model, score)
            elif args.alg == 'FM':
                ssl_loss = ssl_obj(inputs_u[0], inputs_u[1], model)

            loss = cls_loss + args.lambda_u * ssl_loss

            losses.update(loss.item(), inputs_l.size(0))
            losses_x.update(cls_loss.item(), inputs_l.size(0))
            losses_u.update(ssl_loss.item(), inputs_l.size(0))

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            ema_optimizer.step()

            batch_time.update(time.time() - end)
            end = time.time()
            p_bar.set_description(
                "Train Epoch: {epoch}/{epochs:}. Iter: {batch:}/{iter:}. Data: {data:.3f}s. Batch: {bt:.3f}s. Loss: {loss:.4f}. Loss_x: {loss_x:.4f}. Loss_u: {loss_u:.4f}.".format(
                    epoch=epoch + 1,
                    epochs=args.epochs,
                    batch=batch_idx + 1,
                    iter=args.eval_steps,
                    data=data_time.avg,
                    bt=batch_time.avg,
                    loss=losses.avg,
                    loss_x=losses_x.avg,
                    loss_u=losses_u.avg))
            p_bar.update()

        p_bar.close()

        test_loss, test_acc = test(args, test_loader, ema_model)
        is_best = test_acc > best_acc
        best_acc = max(test_acc, best_acc)

        save_checkpoint({
            'epoch': epoch + 1,
            'state_dict': model.state_dict(),
            'ema_state_dict': ema_model.state_dict(),
            'acc': test_acc,
            'best_acc': best_acc,
            'optimizer': optimizer.state_dict(),
        }, is_best, args.out)

        test_accs.append(test_acc)
        logger.info('Best top-1 acc: {:.2f}'.format(best_acc))
        logger.info('Mean top-1 acc: {:.2f}\n'.format(
            np.mean(test_accs[-20:])))


def test(args, test_loader, model):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()
    end = time.time()

    test_loader = tqdm(test_loader)

    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(test_loader):
            data_time.update(time.time() - end)
            model.eval()

            inputs = inputs.to(args.device)
            targets = targets.to(args.device).long()
            outputs = model(inputs)[0]
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

    logger.info("top-1 acc: {:.2f}".format(top1.avg))
    logger.info("top-5 acc: {:.2f}".format(top5.avg))
    return losses.avg, top1.avg


def main():
    device = torch.device('cuda', args.gpu_id)
    args.world_size = 1
    args.n_gpu = torch.cuda.device_count()
    args.device = device

    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO)

    logger.warning(
        f"device: {args.device}, "
        f"n_gpu: {args.n_gpu}")

    logger.info(dict(args._get_kwargs()))


    cls_num_list, labeled_dataset, unlabeled_dataset, test_dataset = DATASET_GETTERS[args.dataset](
        args, './data')

    label_loader = DataLoader(
        labeled_dataset, sampler=RandomSampler(labeled_dataset),
        batch_size=args.batch_size, num_workers=args.num_workers, drop_last=True)

    unlabel_loader = DataLoader(
        unlabeled_dataset, sampler=RandomSampler(unlabeled_dataset),
        batch_size=args.batch_size * args.mu, num_workers=args.num_workers,
        drop_last=True)

    test_loader = DataLoader(
        test_dataset, sampler=SequentialSampler(test_dataset),
        batch_size=args.batch_size, num_workers=args.num_workers)

    if args.alg == 'supervised':
        train_ssl(label_loader, unlabel_loader, test_loader, None)
    elif args.alg == 'FM':
        ssl_obj = FixMatch(args, 1, args.threshold)
        train_ssl(label_loader, unlabel_loader, test_loader, ssl_obj)
    elif args.alg == 'adsh':
        ssl_obj = ADSH(args, 1, args.threshold)
        train_ssl(label_loader, unlabel_loader, test_loader, ssl_obj)


if __name__ == '__main__':
    main()
