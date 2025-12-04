"""resnet in pytorch



[1] Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun.

    Deep Residual Learning for Image Recognition
    https://arxiv.org/abs/1512.03385v1
"""

import torch
import torch.nn as nn


class BasicBlock(nn.Module):
    """Basic Block for resnet 18 and resnet 34

    """

    def __init__(self, in_channels, out_channels, kernel_size, padding, dropout, stride=1):
        super().__init__()

        # residual function
        self.residual_function = nn.Sequential(
            nn.BatchNorm1d(in_channels),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Conv1d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding,
                      bias=False),
            nn.BatchNorm1d(out_channels),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Conv1d(out_channels, out_channels, kernel_size=kernel_size, padding=padding + 1, bias=False),
        )

        # shortcut
        self.shortcut = nn.Sequential()

        # the shortcut output dimension is not the same with residual function
        # use 1*1 convolution to match the dimension
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv1d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
            )

    def forward(self, x):
        return nn.MaxPool1d(2, 2)(self.residual_function(x) + self.shortcut(x))


class ResNet(nn.Module):

    def __init__(self):
        super().__init__()

        self.conv1 = nn.Sequential(
            nn.Conv1d(1, 64, kernel_size=32, padding=16, bias=False),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True))

        self.block1 = BasicBlock(64, 64, 32, 15, 0.5)
        self.block2 = BasicBlock(64, 128, 16, 7, 0.5)
        self.block3 = BasicBlock(128, 256, 8, 3, 0.5)

        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        # self.fc = nn.Linear(512 * block.expansion, num_classes)

    def forward(self, x):
        output = self.conv1(x)
        output = self.block1(output)
        output = self.block2(output)
        output = self.block3(output)
        output = self.avg_pool(output)
        output = output.view(output.size(0), -1)
        # output = self.fc(output)

        return output


if __name__ == '__main__':
    resnet = ResNet()
    in_p = torch.rand((5, 1, 120))
    out = resnet(in_p)
    print(out.shape)
