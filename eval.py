import os
import yaml
import warnings
from collections import OrderedDict
from argparse import ArgumentParser

from easydict import EasyDict
from tqdm import tqdm
import torch
from torch.utils.data import DataLoader

from models import *
from data import ILSVRC2012Dataset, get_valid_transforms
from utils import errors, AverageMeter

warnings.filterwarnings('ignore')


def eval(device, model, test_loader):
    top1 = AverageMeter()  # ':6.4f'
    top5 = AverageMeter()  # ':6.4f'

    model.eval()
    with tqdm(test_loader) as pbar:
        pbar.set_description('Valid Epoch in the val set')

        for i, (input_, target) in enumerate(test_loader):
            # convert 5-d multi-crop format to 4-d for input
            bs, num_crops, c, h, w = input_.size()
            input_ = input_.view(-1, c, h, w)

            # move data to GPU
            input_ = torch.tensor(input_, device=device, dtype=torch.float32)
            target = torch.tensor(target, device=device, dtype=torch.long)

            with torch.no_grad():
                # compute output and loss
                output = model(input_)
                output = output.view(bs, num_crops, -1).mean(1)  # average the multi-crop result

            # record loss and compute accuracies
            err1, err5 = errors(output, target, topk=(1, 5))
            top1.update(err1[0], input_.size(0))
            top5.update(err5[0], input_.size(0))
            # show info in pbar
            postfix = OrderedDict({
                'batch_err@1': f'{top1.val:6.4f}', 'batch_err@5': f'{top5.val:6.4f}'
            })
            pbar.set_postfix(ordered_dict=postfix)
            pbar.update()

        return top1.avg, top5.avg


if __name__ == '__main__':
    # for evaluating alexnet on ILSVRC 2012 val set:
    # $ python -u eval.py --work-dir ./experiments/alexnet --ckpt-name last_checkpoint.pth --test-root ./datasets/val
    parser = ArgumentParser(description='Train ConvNets on CIFAR-100 in PyTorch')
    parser.add_argument('--work-dir', required=True, type=str)
    parser.add_argument('--ckpt-name', required=True, type=str)
    parser.add_argument('--test-root', required=True, type=str)
    args = parser.parse_args()

    # get experiment settings
    with open(os.path.join(args.work_dir, 'config.yaml')) as f:
        cfgs = yaml.load(f, Loader=yaml.FullLoader)
    cfgs = EasyDict(cfgs)

    # hardware
    device = torch.device(cfgs.gpu if torch.cuda.is_available() else 'cpu')

    # get model
    model = get_model(cfgs)
    ckpt_path = os.path.join(args.work_dir, 'checkpoints', args.ckpt_name)
    ckpt = torch.load(ckpt_path)
    if isinstance(ckpt, dict):
        model.load_state_dict(ckpt['model'])
    else:
        model.load_state_dict(ckpt)
    model.to(device)

    # get data
    test_set = ILSVRC2012Dataset(args.test_root, transform=get_valid_transforms(cfgs.scale_size, cfgs.crop_size))
    test_loader = DataLoader(test_set, batch_size=cfgs.batch_size, shuffle=False, num_workers=cfgs.workers)

    err1, err5 = eval(device, model, test_loader)
    print('Top-1 Error: {}\nTop-5 Error: {}'.format(err1, err5))
