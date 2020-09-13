import os
import torch
from torch.nn import CrossEntropyLoss
from torch.optim import SGD
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torchvision import transforms


class Config:
    """Hyper-parameters for training"""

    workers = 4
    gpu = 'cuda:0'

    train_root = os.path.join(os.pardir, 'ILSVRC2012', 'train')  # set directory to training images
    valid_root = os.path.join(os.pardir, 'ILSVRC2012', 'val')  # set directory to validation images

    num_classes = 1000
    epochs = 90
    save_dir = os.path.join(os.curdir, 'checkpoints')
    log_dir = os.path.join(os.curdir, 'logs')  # dir to save training logs and tensorboard event files
    verbose = True  # whether to print logs when training

    arch = 'alexnet'  # model architecture
    resume = os.path.join(os.curdir, 'checkpoints', 'last_checkpoint.pth')  # checkpoint path for further train
    batch_size = 128
    scale_size = 256  # image size after scaling
    crop_size = 227  # image size for model input

    CriterionClass = CrossEntropyLoss  # loss function class
    OptimizerClass = SGD  # optimizer class
    optimizer_params = dict(lr=0.01, momentum=0.9, weight_decay=0.0005)  # hyper-parameters for optimizer
    SchedulerClass = ReduceLROnPlateau  # learning rate scheduler class
    scheduler_params = dict(
        mode='min', factor=0.1, patience=1, verbose=False, threshold=1e-4, threshold_mode='abs',
        cooldown=0, min_lr=1e-8, eps=1e-08
    )  # hyper-parameters for learning rate scheduler


__imagenet_pca = {
    'eigval': torch.tensor([0.2175, 0.0188, 0.0045]),
    'eigvec': torch.tensor([
        [-0.5675, 0.7192, 0.4009],
        [-0.5808, -0.0045, -0.8140],
        [-0.5836, -0.6948, 0.4203]
    ])
}


class Lighting:
    """Lighting noise (AlexNet - style PCA - based noise)"""

    def __init__(self, alphastd, eigval, eigvec):
        self.alphastd = alphastd
        self.eigval = eigval
        self.eigvec = eigvec

    def __call__(self, img):
        if self.alphastd == 0:
            return img

        alpha = img.new().resize_(3).normal_(0, self.alphastd)
        rgb = self.eigvec.type_as(img).clone().mul(alpha.view(1, 3).expand(3, 3)).mul(
            self.eigval.view(1, 3).expand(3, 3)).sum(1).squeeze()
        return img.add(rgb.view(3, 1, 1).expand_as(img))


def get_train_transforms():
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    return transforms.Compose([
        transforms.Resize(Config.scale_size),
        transforms.RandomCrop(Config.crop_size),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        Lighting(0.1, __imagenet_pca['eigval'], __imagenet_pca['eigvec']),
        normalize
    ])


def get_valid_transforms():
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    return transforms.Compose([
        transforms.Resize(Config.scale_size),
        transforms.TenCrop(Config.crop_size),
        transforms.Lambda(lambda crops: torch.stack([normalize(transforms.ToTensor()(crop)) for crop in crops]))
    ])
