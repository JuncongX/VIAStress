import torch
import torch.nn as nn

from VIAStress_wo_MMDG.model_pre import BVPEmbed, EDAEmbed
import copy


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


class WholeFish(nn.Module):
    def __init__(self, x_dim, h_dim, y_dim, weights=None):
        super(WholeFish, self).__init__()
        self.featurizer = FeatureEmbed(x_dim)
        self.classifier = Classifier(2 * x_dim, h_dim, y_dim)
        if weights is not None:
            self.load_state_dict(copy.deepcopy(weights))

    def reset_weights(self, weights):
        self.load_state_dict(copy.deepcopy(weights))

    def forward(self, ppg, eda):
        z = self.featurizer(ppg, eda)
        return self.classifier(z)
