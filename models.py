import torch
from torch import nn

__all__ = ['AlexNet', 'alexnet']


class AlexNet(nn.Module):
    def __init__(self, num_classes=1000):
        super().__init__()
        self.features = nn.Sequential(
            # block 1
            nn.Conv2d(in_channels=3, out_channels=96, kernel_size=11, stride=4, padding=0),  # 3x227x227 -> 96x55x55
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2),  # 96x55x55 -> 96x27x27

            # block 2
            nn.Conv2d(96, 256, kernel_size=5, stride=1, padding=2),  # 96x27x27 -> 256x27x27
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2),  # 256x27x27 -> 256x13x13

            # block 3
            nn.Conv2d(256, 384, kernel_size=3, stride=1, padding=1),  # 256x13x13 -> 384x13x13
            nn.ReLU(),
            nn.Conv2d(384, 384, kernel_size=3, stride=1, padding=1),  # 384x13x13 -> 384x13x13
            nn.ReLU(),
            nn.Conv2d(384, 256, kernel_size=3, stride=1, padding=1),  # 384x13x13 -> 256x13x13
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2)  # 256x13x13 -> 256x6x6
        )

        self.classifier = nn.Sequential(
            nn.Linear(in_features=256 * 6 * 6, out_features=4096),  # 9216 -> 4096
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(4096, 4096),  # 4096 -> 4096
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(4096, num_classes)  # 4096 -> 1000
        )

        self._init_weights()

    def _init_weights(self):
        """initialize weights and bias according to the paper"""
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.Linear)):
                nn.init.normal_(m.weight, mean=0, std=0.01)

        for m in self.modules():
            nn.init.constant_(m.bias, 0)

        nn.init.constant_(self.features[3].bias, 1)
        nn.init.constant_(self.features[8].bias, 1)
        nn.init.constant_(self.features[10].bias, 1)
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.constant_(m.bias, 1)

    def forward(self, x):
        x = self.features(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)

        return x


def alexnet(num_classes=1000):
    return AlexNet(num_classes)


if __name__ == '__main__':
    # test
    model = alexnet(num_classes=1000)
    print(model)
