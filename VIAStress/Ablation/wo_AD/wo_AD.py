import torch
import torch.nn as nn
import torch.nn.functional as F
from VIAStress.model import FeatureEmbed, ProjectHead, EncoderTrans


class Classifier(nn.Module):
    def __init__(self, z_dim, r_dim, h_dim, y_dim):
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
    def __init__(self, x_dim, r_dim, h_dim, y_dim):
        super(Model, self).__init__()
        self.feature_cnn = FeatureEmbed(x_dim)

        self.classifier = Classifier(2 * x_dim, r_dim, h_dim, y_dim)

    def forward(self, bvp, eda):
        z, bvp_z, eda_z = self.feature_cnn(bvp, eda)
        out = self.classifier(z)

        return out, bvp_z, eda_z
