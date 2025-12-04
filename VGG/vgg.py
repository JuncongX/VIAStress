"""vgg in pytorch


[1] Karen Simonyan, Andrew Zisserman

    Very Deep Convolutional Networks for Large-Scale Image Recognition.
    https://arxiv.org/abs/1409.1556v6

[2] Multimodal brain–computer interface for in-vehicle driver cognitive load measurement: Dataset and baselines
"""

import torch
import torch.nn as nn


class Conv_Block(nn.Module):
    def __init__(self, input_channel, output_channel, kernel_size, padding):
        super(Conv_Block, self).__init__()
        self.cnn = nn.Sequential(
            nn.Conv1d(input_channel, output_channel, kernel_size=kernel_size, padding=padding),
            nn.BatchNorm1d(output_channel),
            nn.ReLU(inplace=True),
            nn.Conv1d(output_channel, output_channel, kernel_size=kernel_size, padding=padding),
            nn.BatchNorm1d(output_channel),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=2, stride=2)
        )

    def forward(self, x):
        return self.cnn(x)


class VGG(nn.Module):

    def __init__(self):
        super().__init__()
        self.conv_1 = Conv_Block(1, 64, 32, 16)
        self.conv_2 = Conv_Block(64, 128, 16, 8)
        self.conv_3 = Conv_Block(128, 256, 8, 4)
        self.global_avg = nn.AdaptiveAvgPool1d(1)

    def forward(self, x):
        output = self.conv_1(x)
        output = self.conv_2(output)
        output = self.conv_3(output)
        output = self.global_avg(output)
        output = output.view(output.size()[0], -1)

        return output


if __name__ == '__main__':
    vgg = VGG()
    in_p = torch.rand((5, 1, 120))
    out = vgg(in_p)
    print(out.shape)
