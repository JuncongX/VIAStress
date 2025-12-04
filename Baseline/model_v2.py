import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F
from torch.autograd import Variable
from typing import Optional, List
import math
from torch.distributions import Normal
from Baseline.model import Classifier


class BVPEmbed(nn.Module):
    def __init__(self, x_dim):
        super(BVPEmbed, self).__init__()
        self.pooling = 4
        self.conv_1 = nn.Sequential(
            nn.Conv1d(1, 8, 3, 1, 1),
            nn.BatchNorm1d(8),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(self.pooling, self.pooling)
        )

        self.conv_2 = nn.Sequential(
            nn.Conv1d(8, 16, 3, 1, 1),
            nn.BatchNorm1d(16),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(self.pooling, self.pooling)
        )

        self.conv_3 = nn.Sequential(
            nn.Conv1d(16, 32, 3, 1, 1),
            nn.BatchNorm1d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(self.pooling, self.pooling)
        )

        self.fc1 = nn.Linear(32 * 30, x_dim)
        self.fc2 = nn.Linear(x_dim, x_dim)

    def forward(self, ppg):
        x = self.conv_1(ppg)

        x = self.conv_2(x)

        x = self.conv_3(x)

        bz, _, _ = x.shape
        x = x.view(bz, -1)

        x = self.fc1(x)
        x = F.relu(x, inplace=True)
        x = self.fc2(x)
        return x


class EDAEmbed(nn.Module):
    def __init__(self, x_dim):
        super(EDAEmbed, self).__init__()
        self.pooling = 2
        self.conv_1 = nn.Sequential(
            nn.Conv1d(1, 8, 3, 1, 1),
            nn.BatchNorm1d(8),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(self.pooling, self.pooling)
        )

        self.conv_2 = nn.Sequential(
            nn.Conv1d(8, 16, 3, 1, 1),
            nn.BatchNorm1d(16),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(self.pooling, self.pooling)
        )

        self.conv_3 = nn.Sequential(
            nn.Conv1d(16, 32, 3, 1, 1),
            nn.BatchNorm1d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(self.pooling, self.pooling)
        )

        self.fc1 = nn.Linear(32 * 15, x_dim)
        self.fc2 = nn.Linear(x_dim, x_dim)

    def forward(self, eda):
        x = self.conv_1(eda)

        x = self.conv_2(x)

        x = self.conv_3(x)

        bz, _, _ = x.shape
        x = x.view(bz, -1)

        x = self.fc1(x)
        x = F.relu(x, inplace=True)
        x = self.fc2(x)

        return x


class FeatureEmbed(nn.Module):
    def __init__(self, x_dim):
        super(FeatureEmbed, self).__init__()
        self.bvp_embedding = BVPEmbed(x_dim)
        self.eda_embedding = EDAEmbed(x_dim)

    def forward(self, bvp, eda):
        bvp_f = self.bvp_embedding(bvp)
        eda_f = self.eda_embedding(eda)
        x_f = torch.cat((bvp_f, eda_f), dim=-1)
        return x_f


class Model(nn.Module):
    def __init__(self, x_dim, z_dim, h_dim, y_dim):
        super(Model, self).__init__()
        self.feature_cnn = FeatureEmbed(x_dim)
        self.classifier = Classifier(2 * x_dim, h_dim, y_dim)

    def forward(self, bvp, eda):
        z = self.feature_cnn(bvp, eda)
        out = self.classifier(z)

        return out


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='NPStress')
    parser.add_argument('--k', type=int, default=5, help="KFold")
    # parser.add_argument('--epoch', type=int, default=150)
    parser.add_argument('--epoch', type=int, default=150)
    parser.add_argument('--batch_size', type=int, default=120)
    parser.add_argument('--LR', type=float, default=0.0001)
    parser.add_argument('--save_path', type=str, default="./checkpoints")
    parser.add_argument('--x_dim', type=int, default=128)
    parser.add_argument('--y_dim', type=int, default=3)
    parser.add_argument('--h_dim', type=int, default=256)
    parser.add_argument('--r_dim', type=int, default=256)
    parser.add_argument('--context_num', type=int, default=10)
    parser.add_argument('--dataset_name', type=str, default='wesad')

    args = parser.parse_args()

    bvp = torch.rand((8, 1, 1920))
    eda = torch.rand((8, 1, 120))
    y = torch.zeros((8)).long()

    # eda_emb = EDAEmbed(128)
    # print(eda_emb(eda).shape)

    model = Model(args.x_dim, args.r_dim, args.h_dim, args.y_dim)
    out = model(bvp, eda)
    print(out.shape)
