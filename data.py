import torch
from torchvision.datasets import ImageFolder
from torchvision import transforms


class ILSVRC2012Dataset(ImageFolder):
    """ILSVRC 2012 Dataset"""

    def __init__(self, root, transform=None):
        super(ILSVRC2012Dataset, self).__init__(root, transform)


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


def get_train_transforms(scale_size, crop_size):
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    return transforms.Compose([
        transforms.Resize(scale_size),
        transforms.RandomCrop(crop_size),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        Lighting(0.1, __imagenet_pca['eigval'], __imagenet_pca['eigvec']),
        normalize
    ])


def get_valid_transforms(scale_size, crop_size):
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    return transforms.Compose([
        transforms.Resize(scale_size),
        transforms.TenCrop(crop_size),
        transforms.Lambda(lambda crops: torch.stack([normalize(transforms.ToTensor()(crop)) for crop in crops]))
    ])


if __name__ == '__main__':
    import os
    import numpy as np
    from matplotlib import pyplot as plt

    train_root = os.path.join(os.pardir, 'ILSVRC2012', 'train')
    valid_root = os.path.join(os.pardir, 'ILSVRC2012', 'val')

    train_set = ILSVRC2012Dataset(train_root)
    valid_set = ILSVRC2012Dataset(valid_root)

    print(len(train_set), len(valid_set))

    print(train_set.class_to_idx)

    image, label = train_set[0]
    print(label)
    fig = plt.figure()
    plt.imshow(np.array(image))
    plt.show()
