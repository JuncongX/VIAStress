from VGG.vgg import VGG
import torch
import torch.nn as nn


class Classifier(nn.Module):
    def __init__(self, z_dim, h_dim, y_dim):
        super(Classifier, self).__init__()

        self.fc = nn.Sequential(
            nn.Linear(z_dim, h_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(0.25),
            nn.Linear(h_dim, y_dim),
        )

    def forward(self, z):
        return self.fc(z)


class FeatureEmbed(nn.Module):
    def __init__(self):
        super(FeatureEmbed, self).__init__()
        self.bvp_embedding = VGG()
        self.eda_embedding = VGG()

    def forward(self, bvp, eda):
        bvp_f = self.bvp_embedding(bvp)
        eda_f = self.eda_embedding(eda)
        x_f = torch.cat((bvp_f, eda_f), dim=-1)
        return x_f


class Model(nn.Module):
    def __init__(self, h_dim, y_dim):
        super(Model, self).__init__()
        self.feature_cnn = FeatureEmbed()
        self.classifier = Classifier(512, h_dim, y_dim)

    def forward(self, bvp, eda):
        z = self.feature_cnn(bvp, eda)
        out = self.classifier(z)

        return out
