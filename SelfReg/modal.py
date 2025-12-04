import torch
import torch.nn as nn
from VIAStress_wo_MMDG.model_pre import FeatureEmbed
from Baseline.model import Classifier


class Featurizer(nn.Module):
    def __init__(self, x_dim):
        super(Featurizer, self).__init__()
        self.feature_cnn = FeatureEmbed(x_dim)

    def forward(self, bvp, eda):
        z = self.feature_cnn(bvp, eda)
        # z = self.feature_fc(z)
        return z
