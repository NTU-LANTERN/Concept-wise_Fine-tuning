import random
import time
import warnings
import sys
import argparse
import shutil
import torch
import torch.backends.cudnn as cudnn
from torch.optim import SGD
from torch.utils.data import DataLoader
import torchvision.transforms as T
import torch.nn.functional as F
import torch.nn as nn
import vision.datasets as datasets
import vision.models as models
from utils.data import MultipleApply
from utils.metric import accuracy
from utils.meter import AverageMeter, ProgressMeter
from utils.data import ForeverDataIterator
from utils.logger import CompleteLogger
from concept_tuning import Classifier, Contuning

import numpy as np
sys.path.append('.')
import utils

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def main(args: argparse.Namespace):
    logger = CompleteLogger(args.log, args.phase)
    print(args)

    if args.seed is not None:
        np.random.seed(args.seed)
        random.seed(args.seed)
        torch.manual_seed(args.seed)
        torch.cuda.manual_seed(args.seed)
        cudnn.deterministic = True
    cudnn.benchmark = False #True

    # Data loading
    image_augmentation = utils.get_train_transform(args.train_resizing, not args.no_hflip, args.color_jitter)
    patch_augmentation = T.Compose([
            T.RandomResizedCrop(224, scale=(0.5, 1.0)),
            T.RandomHorizontalFlip(),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
    val_transform = utils.get_val_transform(args.val_resizing)
    train_transform = MultipleApply([image_augmentation for _ in range(2)] + [patch_augmentation for _ in range(2)])

    train_dataset, val_dataset, num_classes = utils.get_dataset(args.data, args.root, train_transform,
                                                val_transform, args.sample_rate, args.num_samples_per_classes)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True,
                              num_workers=args.workers, drop_last=True)
    train_iter = ForeverDataIterator(train_loader)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.workers)
    print("training dataset size: {} test dataset size: {}".format(len(train_dataset), len(val_dataset)))

    # create model
    print("=> using pre-trained model '{}'".format(args.arch))
    backbone_q = models.__dict__[args.arch](pretrained=True)
    if args.pretrained:
        print("=> loading pre-trained backbone from '{}'".format(args.pretrained))
        pretrained_dict = torch.load(args.pretrained)
        backbone_q.load_state_dict(pretrained_dict, strict=False)
    classifier_q = Classifier(backbone_q, num_classes, ib_dim=args.ib_dim).to(device)
    backbone_k = models.__dict__[args.arch](pretrained=True)
    classifier_k = Classifier(backbone_k, num_classes, ib_dim=args.ib_dim).to(device)
    contuning = Contuning(classifier_q, classifier_k, num_classes, K=args.K, m=args.m, T=args.T, confusing=args.confusing)

    # define optimizer and lr scheduler
    if not args.cos:
        optimizer = SGD(classifier_q.get_parameters(args.lr), lr=args.lr, momentum=args.momentum, weight_decay=args.wd, nesterov=True)
        lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, args.lr_decay_epochs, gamma=args.lr_gamma)
    else:
        optimizer = SGD(classifier_q.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.wd, nesterov=True)
        lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, args.epochs) 
        
    # resume from the best checkpoint
    if args.phase == 'test':
        checkpoint = torch.load(logger.get_checkpoint_path('best'), map_location='cpu')
        classifier_q.load_state_dict(checkpoint)
        acc1 = validate(val_loader, classifier_q, args)
        return

    # start training
    best_acc1 = 0.0
    for epoch in range(args.epochs):
        print(lr_scheduler.get_lr())
        # train for one epoch
        train(train_iter, contuning, optimizer, epoch, args, num_classes)
        lr_scheduler.step()

        # evaluate on validation set
        acc1 = validate(val_loader, classifier_q, args)

        # remember best acc@1 and save checkpoint
        torch.save(classifier_q.state_dict(), logger.get_checkpoint_path('latest'))
        if acc1 > best_acc1:
            shutil.copy(logger.get_checkpoint_path('latest'), logger.get_checkpoint_path('best'))
        best_acc1 = max(acc1, best_acc1)

    print("best_acc1 = {:3.2f}".format(best_acc1))
    logger.close()

def train(train_iter: ForeverDataIterator, contuning, optimizer: SGD, epoch:int, args:argparse.Namespace, num_classes:int):
    batch_time = AverageMeter('Time', ':4.2f')
    data_time = AverageMeter('Data', ':3.1f')
    losses = AverageMeter('Loss', ':3.2f')
    cls_losses = AverageMeter('Cls Loss', ':3.2f')
    contrastive_losses = AverageMeter('Contrastive Loss', ':3.2f')
    concept_contrastive_losses = AverageMeter('Conncept_contrastive_loss', ':3.2f')
    cls_accs = AverageMeter('Cls Acc', ':3.2f')
    IB_cls_accs = AverageMeter('IB Acc', ':3.2f')

    progress = ProgressMeter(
        args.iters_per_epoch,
        [batch_time, data_time, losses, cls_losses, contrastive_losses, concept_contrastive_losses, cls_accs, IB_cls_accs],
        prefix="Epoch: [{}]".format(epoch))

    classifier_criterion = torch.nn.CrossEntropyLoss(label_smoothing=args.smoothing).to(device)

    # switch to train mode
    contuning.train()
    beta_anneal = args.beta_anneal if epoch>args.start else 0.0
    end = time.time()
    for i in range(args.iters_per_epoch):
        x, labels = next(train_iter)

        imgs = [element.to(device) for element in x]
        labels = labels.to(device)

        # measure data loading time
        data_time.update(time.time() - end)

        # compute output
        y, contrastive_loss, conncept_contrastive_loss, logits, IB_info_loss = contuning(imgs, labels)
        cls_loss = classifier_criterion(y, labels)
        IB_class_loss = classifier_criterion(logits, labels)

        loss = cls_loss + contrastive_loss * args.trade_off  + \
            conncept_contrastive_loss * beta_anneal +  IB_class_loss + args.infow * IB_info_loss

        # measure accuracy and record loss
        losses.update(loss.item(), x[0].size(0))
        cls_losses.update(cls_loss.item(), x[0].size(0))
        contrastive_losses.update(contrastive_loss.item(), x[0].size(0))
        concept_contrastive_losses.update(conncept_contrastive_loss.item(), x[0].size(0))
        IB_cls_accs.update(accuracy(logits, labels)[0].item(), x[0].size(0))
        cls_accs.update(accuracy(y, labels)[0].item(), x[0].size(0))

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            progress.display(i)


def validate(val_loader: DataLoader, model: Classifier, args: argparse.Namespace) -> float:
    batch_time = AverageMeter('Time', ':6.3f')
    top1 = AverageMeter('Acc@1', ':6.2f')
    IB_top1 = AverageMeter('IB_Acc@1', ':6.2f')
    progress = ProgressMeter(len(val_loader), [batch_time,  top1, IB_top1], prefix='Test: ')
    model.eval()

    with torch.no_grad():
        end = time.time()
        for i, (images, target) in enumerate(val_loader):
            images = images.to(device)
            target = target.to(device)

            # compute output
            output,_,_,logits = model(images)
            loss = F.cross_entropy(output, target)
            # measure accuracy and record loss
            acc1, _ = accuracy(output, target, topk=(1, 5))
            IB_acc1, _ = accuracy(logits, target, topk=(1, 5))
            IB_top1.update(IB_acc1.item(), images.size(0))
            top1.update(acc1.item(), images.size(0))
            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if i % args.print_freq == 0:
                progress.display(i)

        print(' * Acc@1 {top1.avg:.3f} IB_Acc@1 {IB_top1.avg:.3f}'
              .format(top1=top1, IB_top1=IB_top1))

    return max(top1.avg, IB_top1.avg)


if __name__ == '__main__':
    architecture_names = sorted(
        name for name in models.__dict__
        if name.islower() and not name.startswith("__")
        and callable(models.__dict__[name])
    )
    dataset_names = sorted(
        name for name in datasets.__dict__
        if not name.startswith("__") and callable(datasets.__dict__[name])
    )
    parser = argparse.ArgumentParser(description='Concept-level Fine-tuning')
    # dataset parameters
    parser.add_argument('root', metavar='DIR',
                        help='root path of dataset')
    parser.add_argument('-d', '--data', metavar='DATA')
    parser.add_argument('-sr', '--sample-rate', default=100, type=int,
                        metavar='N',
                        help='sample rate of training dataset (default: 100)')
    parser.add_argument('-sc', '--num-samples-per-classes', default=None, type=int,
                        help='number of samples per classes.')
    parser.add_argument('--train-resizing', type=str, default='default', help='resize mode during training')
    parser.add_argument('--val-resizing', type=str, default='default', help='resize mode during validation')
    parser.add_argument('--no-hflip', action='store_true', help='no random horizontal flipping during training')
    parser.add_argument('--color-jitter', action='store_true', help='no color jitter during training')
    # model parameters
    parser.add_argument('-a', '--arch', metavar='ARCH', default='resnet50',
                        choices=architecture_names,
                        help='backbone architecture: ' +
                             ' | '.join(architecture_names) +
                             ' (default: resnet50)')
    parser.add_argument('--pretrained', default=None,
                        help="pretrained checkpoint of the backbone. "
                             "(default: None, use the ImageNet supervised pretrained backbone)")
    parser.add_argument('--T', default=0.07, type=float, help="temperature. (default: 0.07)")
    parser.add_argument('--K', type=int, default=40, help="queue size. (default: 40)")
    parser.add_argument('--m', type=float, default=0.999, help="momentum coefficient. (default: 0.999)")
    parser.add_argument('--ib-dim', type=int, default=512,
                        help="dimension of the ib head. (default: 512)")
    parser.add_argument('--trade-off', type=float, default=1.0, help="trade-off parameters. (default: 1.0)")
    parser.add_argument('--beta-anneal', type=float, default=1.0, help="trade-off parameters. (default: 1.0)")
    parser.add_argument('--confusing', type=int, default=100, help="number of negative classes for contrastive learning")
    parser.add_argument('--smoothing', type=float, default=0.0)
    parser.add_argument('--cos', action='store_true', help="learning rate schedule")
    parser.add_argument('--infow', type=float, default=5e-3)
    parser.add_argument('--start', type=int, default=-1)
    parser.add_argument('-b', '--batch-size', default=48, type=int,
                        metavar='N',
                        help='mini-batch size (default: 48)')
    parser.add_argument('--lr', '--learning-rate', default=0.01, type=float,
                        metavar='LR', help='initial learning rate', dest='lr')
    parser.add_argument('--lr-gamma', default=0.1, type=float, help='parameter for lr scheduler')
    parser.add_argument('--lr-decay-epochs', type=int, default=(12,24, ), nargs='+', help='epochs to decay lr')
    parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                        help='momentum')
    parser.add_argument('--wd', '--weight-decay', default=0.0005, type=float,
                        metavar='W', help='weight decay (default: 5e-4)')
    parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                        help='number of data loading workers (default: 4)')
    parser.add_argument('--epochs', default=30, type=int, metavar='N',
                        help='number of total epochs to run')
    parser.add_argument('-i', '--iters-per-epoch', default=500, type=int,
                        help='Number of iterations per epoch')
    parser.add_argument('-p', '--print-freq', default=100, type=int,
                        metavar='N', help='print frequency (default: 100)')
    parser.add_argument('--seed', default=None, type=int,
                        help='seed for initializing training. ')
    parser.add_argument("--log", type=str, default='logs/concept_tuning',
                        help="Where to save logs, checkpoints and debugging images.")
    parser.add_argument("--phase", type=str, default='train', choices=['train', 'test'],
                        help="When phase is 'test', only test the model.")
    args = parser.parse_args()
    main(args)