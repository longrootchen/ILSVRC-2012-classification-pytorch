from torchvision.datasets import ImageFolder


class ILSVRC2012Dataset(ImageFolder):
    """ILSVRC 2012 Dataset"""

    def __init__(self, root, transform=None):
        super(ILSVRC2012Dataset, self).__init__(root, transform)


if __name__ == '__main__':
    import os
    import numpy as np
    from matplotlib import pyplot as plt

    train_root = os.path.join(os.pardir, 'ILSVRC2012', 'train')
    valid_root = os.path.join(os.pardir, 'ILSVRC2012', 'val')

    # test Dataset class
    train_set = ILSVRC2012Dataset(train_root)
    valid_set = ILSVRC2012Dataset(valid_root)

    # check the length of dataset
    print(len(train_set), len(valid_set))  # (1281167, 50000)

    # see the class to index
    print(train_set.class_to_idx)

    # visulize an image
    image, label = train_set[0]
    print(label)
    fig = plt.figure()
    plt.imshow(np.array(image))
    plt.show()
