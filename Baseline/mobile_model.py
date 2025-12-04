import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F
from torch.autograd import Variable
from typing import Optional, List
import math
from torch.distributions import Normal
from VIAStress_wo_MMDG.model_pre import FeatureEmbed


class Classifier(nn.Module):
    def __init__(self, z_dim, h_dim, y_dim):
        super(Classifier, self).__init__()

        self.fc = nn.Sequential(
            nn.Linear(z_dim, h_dim),
            nn.ReLU(inplace=True),
            nn.Linear(h_dim, h_dim),
            nn.ReLU(inplace=True),
            nn.Linear(h_dim, y_dim),
        )

    def forward(self, z):
        return self.fc(z)


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

    parser = argparse.ArgumentParser(description='VIAStress')
    parser.add_argument('--x_dim', type=int, default=256)
    parser.add_argument('--y_dim', type=int, default=2)
    parser.add_argument('--h_dim', type=int, default=256)
    parser.add_argument('--r_dim', type=int, default=256)

    args = parser.parse_args()

    bvp = torch.rand((1, 1, 1920))
    eda = torch.rand((1, 1, 120))

    # eda_emb = EDAEmbed(128)
    # print(eda_emb(eda).shape)

    model = Model(args.x_dim, args.r_dim, args.h_dim, args.y_dim)
    torch.save(model.state_dict(), "baseline_mobile.pt")
    out = model(bvp, eda)
    print(out.shape)

    example_inputs = (bvp, eda)
    traced_model = torch.jit.trace(model, example_inputs)
    traced_model.save("baseline_mobile_traced.pt")