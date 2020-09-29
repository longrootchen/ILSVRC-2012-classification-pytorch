import os
import yaml
import datetime
import warnings
from collections import OrderedDict
from argparse import ArgumentParser

from easydict import EasyDict
from tqdm import tqdm
import torch
from torch.nn import CrossEntropyLoss
from torch.optim import SGD
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from models import *
from data import ILSVRC2012Dataset, get_train_transforms, get_valid_transforms
from utils import errors, AverageMeter

warnings.filterwarnings('ignore')


class Trainer:

    def __init__(self, cfgs, model):
        """
        Args:
            cfgs (class): a class holding configs for training
            model (torch.nn.Module): the model architecture to be trained
        """
        self.cfgs = cfgs
        self.model = model
        self.start_epoch = 1
        self.best_err1 = 1.1

        self.device = torch.device(cfgs.gpu if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)

        self.criterion = CrossEntropyLoss().to(self.device)
        self.optimizer = SGD(self.model.parameters(), lr=cfgs.lr, weight_decay=cfgs.weight_decay,
                             momentum=cfgs.momentum)
        self.scheduler = ReduceLROnPlateau(self.optimizer, factor=cfgs.factor, mode='min', patience=1,
                                           verbose=False, threshold=1e-4, threshold_mode='abs',
                                           cooldown=0, min_lr=1e-8, eps=1e-08)

        # optionally resume from a checkpoint
        if cfgs.resume:
            if os.path.isfile(cfgs.resume):
                print("=> loading checkpoint '{}'".format(cfgs.resume))
                checkpoint = torch.load(cfgs.resume, map_location=self.device)
                self.load(cfgs.resume)
                print("=> loaded checkpoint '{}' (epoch {})".format(cfgs.resume, checkpoint['epoch']))
            else:
                print("=> no checkpoint found at '{}'".format(cfgs.resume))

        # create directory to checkpoint if necessary
        if not os.path.exists(cfgs.save_dir):
            os.makedirs(cfgs.save_dir)
        # create directory to training logs if necessary
        if not os.path.exists(cfgs.log_dir):
            os.makedirs(cfgs.log_dir)
        self.writer = SummaryWriter(log_dir=cfgs.log_dir)

        self.log('Trainer prepared in device: {}'.format(torch.cuda.get_device_name(self.device)))

    def fit(self, train_loader, valid_loader):
        """
        train and validate the model

        Args:
            train_loader (torch.utils.data.DataLoader): DataLoader instance object for training set
            valid_loader (torch.utils.data.DataLoader): DataLoader instance object for validation set
        """
        for epoch in range(self.start_epoch, self.cfgs.epochs + 1):
            if self.cfgs.verbose:
                lr = self.optimizer.param_groups[0]['lr']
                timestamp = datetime.datetime.now().isoformat()
                self.log('{}\tEpoch: {}\tLR: {}'.format(timestamp, epoch, lr))

            # train for an epoch
            err1, err5 = self.train(epoch, train_loader)

            self.save(epoch, f'{self.cfgs.save_dir}/last_checkpoint.pth')
            self.log(f'[RESULT]: Train Epoch: {epoch}\t top-1 err: {err1:6.4f}\t top-5 err: {err5:6.4f}')
            self.writer.add_scalars('top-1 Error Rate', {'train': err1}, epoch)
            self.writer.add_scalars('top-5 Error Rate', {'train': err5}, epoch)

            # validate
            err1, err5 = self.validate(epoch, valid_loader)

            if err1 < self.best_err1:
                self.best_err1 = err1
                self.save(epoch, f'{self.cfgs.save_dir}/best_checkpoint_{str(epoch).zfill(3)}epoch.pth')
            self.log(f'[RESULT]: Valid Epoch: {epoch}\t top-1 err: {err1:6.4f}\t top-5 err: {err5:6.4f}')
            self.writer.add_scalars('top-1 Error Rate', {'valid': err1}, epoch)
            self.writer.add_scalars('top-5 Error Rate', {'valid': err5}, epoch)

            # adjust learning rate if necessary
            self.scheduler.step(metrics=err1)

    def train(self, epoch, train_loader):
        """
        train the model for an epoch

        Args:
            epoch (int): current epoch
            train_loader (torch.utils.data.DataLoader): train data loader
        Returns:
            top1.avg (float): top-1 error rate
            top5.avg (float): top-5 error rate
        """

        losses = AverageMeter()  # ':.4e'
        top1 = AverageMeter()  # ':6.4f'
        top5 = AverageMeter()  # ':6.4f'

        self.model.train()
        with tqdm(train_loader) as pbar:
            pbar.set_description('Train Epoch {}'.format(epoch))

            for step, (input_, target) in enumerate(train_loader):
                # move data to device
                input_ = torch.tensor(input_, device=self.device, dtype=torch.float32)
                target = torch.tensor(target, device=self.device, dtype=torch.long)

                # forward and compute loss
                output = self.model(input_)
                loss = self.criterion(output, target)

                # backward and update params
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                # record loss and compute accuracies
                err1, err5 = errors(output, target, topk=(1, 5))
                losses.update(loss.item(), input_.size(0))
                top1.update(err1[0], input_.size(0))
                top5.update(err5[0], input_.size(0))
                # show info in pbar
                postfix = OrderedDict({
                    'batch_loss': f'{losses.val:6.4f}', 'running_loss': f'{losses.avg:6.4f}',
                    'batch_err@1': f'{top1.val:6.4f}', 'batch_err@5': f'{top5.val:6.4f}'
                })
                pbar.set_postfix(ordered_dict=postfix)
                pbar.update()

                # visualization with TensorBoard
                total_iter = (epoch - 1) * len(train_loader) + step + 1
                self.writer.add_scalar('training_loss', losses.val, total_iter)

        return top1.avg, top5.avg

    def validate(self, epoch, valid_loader):
        """
        validate the model on validation set using ten-crop strategy

        Args:
            epoch (int): current epoch
            valid_loader (torch.utils.data.DataLoader): validatioin data loader
        Returns:
            top1.avg: (float) top-1 error rate
            top5.avg: (float) top-5 error rate
        """

        losses = AverageMeter()  # ':.4e'
        top1 = AverageMeter()  # ':6.4f'
        top5 = AverageMeter()  # ':6.4f'

        self.model.eval()
        with tqdm(valid_loader) as pbar:
            pbar.set_description('Valid Epoch {}'.format(epoch))

            for i, (input_, target) in enumerate(valid_loader):
                # convert 5-d multi-crop format to 4-d for input
                bs, num_crops, c, h, w = input_.size()
                input_ = input_.view(-1, c, h, w)

                # move data to GPU
                input_ = torch.tensor(input_, device=self.device, dtype=torch.float32)
                target = torch.tensor(target, device=self.device, dtype=torch.long)

                with torch.no_grad():
                    # compute output and loss
                    output = self.model(input_)
                    output = output.view(bs, num_crops, -1).mean(1)  # average the multi-crop result
                    loss = self.criterion(output, target)

                # record loss and compute accuracies
                err1, err5 = errors(output, target, topk=(1, 5))
                losses.update(loss.item(), input_.size(0))
                top1.update(err1[0], input_.size(0))
                top5.update(err5[0], input_.size(0))
                # show info in pbar
                postfix = OrderedDict({
                    'batch_loss': f'{losses.val:6.4f}', 'running_loss': f'{losses.avg:6.4f}',
                    'batch_err@1': f'{top1.val:6.4f}', 'batch_err@5': f'{top5.val:6.4f}'
                })
                pbar.set_postfix(ordered_dict=postfix)
                pbar.update()

        return top1.avg, top5.avg

    def save(self, epoch, path):
        self.model.eval()
        torch.save({
            'model': self.model.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'scheduler': self.scheduler.state_dict(),
            'best_err1': self.best_err1,
            'epoch': epoch
        }, path)

    def load(self, path):
        checkpoint = torch.load(path, map_location=self.device)

        # whether the checkpoint contains other training info
        if isinstance(checkpoint, dict):
            self.model.load_state_dict(checkpoint['model'])
            self.optimizer.load_state_dict(checkpoint['optimizer'])
            self.scheduler.load_state_dict(checkpoint['scheduler'])
            self.best_err1 = checkpoint['best_err1']
            self.start_epoch = checkpoint['epoch'] + 1
        else:
            self.model.load_state_dict(checkpoint)

    def log(self, msg):
        if self.cfgs.verbose:
            print(msg)

        log_path = os.path.join(self.cfgs.log_dir, 'log.txt')
        with open(log_path, 'a+') as logger:
            logger.write(f'{msg}\n')


if __name__ == '__main__':
    # for training alexnet:
    # $ python -u train.py --work-dir ./experiments/alexnet
    parser = ArgumentParser(description='Train ConvNets on ILSVRC 2012 in PyTorch')
    parser.add_argument('--work-dir', required=True, type=str)
    args = parser.parse_args()

    # get experiment settings
    with open(os.path.join(args.work_dir, 'config.yaml')) as f:
        cfgs = yaml.load(f, Loader=yaml.FullLoader)
    cfgs = EasyDict(cfgs)

    # set paths
    cfgs.save_dir = os.path.join(args.work_dir, cfgs.save_dir)
    cfgs.log_dir = os.path.join(args.work_dir, cfgs.log_dir)
    cfgs.resume = os.path.join(args.work_dir, cfgs.resume)

    # get model
    model = get_model(cfgs)

    # get data
    train_set = ILSVRC2012Dataset(cfgs.train_root, transform=get_train_transforms(cfgs.scale_size, cfgs.crop_size))
    valid_set = ILSVRC2012Dataset(cfgs.valid_root, transform=get_valid_transforms(cfgs.scale_size, cfgs.crop_size))

    train_loader = DataLoader(train_set, batch_size=cfgs.batch_size, shuffle=True, num_workers=cfgs.workers)
    valid_loader = DataLoader(valid_set, batch_size=cfgs.batch_size, shuffle=False, num_workers=cfgs.workers)

    # train
    trainer = Trainer(cfgs, model)
    trainer.fit(train_loader, valid_loader)
