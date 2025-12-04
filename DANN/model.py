import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F
from torch.autograd import Variable
from typing import Optional, List
import math
from torch.distributions import Normal
from VIAStress_wo_MMDG.model_pre import FeatureEmbed
from DANN.functions import ReverseLayerF


class Classifier(nn.Module):
    def __init__(self, z_dim, h_dim, y_dim):
        super(Classifier, self).__init__()

        self.fc = nn.Sequential(
            nn.Linear(z_dim, h_dim),
            nn.ReLU(inplace=True),
            nn.Linear(h_dim, h_dim),
            nn.ReLU(inplace=True),
            nn.Linear(h_dim, y_dim)
        )

        self.domain_classifier = nn.Sequential(
            nn.Linear(z_dim, h_dim),
            # nn.BatchNorm1d(h_dim),
            nn.ReLU(inplace=True),
            nn.Linear(h_dim, h_dim),
            nn.ReLU(inplace=True),
            nn.Linear(h_dim, 2)
        )

    def forward(self, z, alpha):
        reverse_feature = ReverseLayerF.apply(z, alpha)
        domain_output = self.domain_classifier(reverse_feature)

        return self.fc(z), domain_output


class Model(nn.Module):
    def __init__(self, x_dim, z_dim, h_dim, y_dim):
        super(Model, self).__init__()
        self.feature_cnn = FeatureEmbed(x_dim)
        self.classifier = Classifier(2 * x_dim, h_dim, y_dim)

    def forward(self, bvp, eda, alpha):
        z = self.feature_cnn(bvp, eda)
        class_output, domain_output = self.classifier(z, alpha)

        return class_output, domain_output


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
