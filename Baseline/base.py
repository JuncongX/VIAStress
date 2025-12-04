import torch
import torch.nn as nn
import torch.nn.functional as F


class BVPEmbed(nn.Module):
    def __init__(self, x_dim):
        super(BVPEmbed, self).__init__()
        self.pooling = 4
        self.conv_1 = nn.Sequential(
            nn.Conv1d(1, 8, 9, 1, 4),
            nn.BatchNorm1d(8),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(self.pooling, self.pooling)
        )

        self.conv_2 = nn.Sequential(
            nn.Conv1d(8, 16, 7, 1, 3),
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
            nn.Conv1d(1, 8, 9, 1, 4),
            nn.BatchNorm1d(8),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(self.pooling, self.pooling)
        )

        self.conv_2 = nn.Sequential(
            nn.Conv1d(8, 16, 7, 1, 3),
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


class BaseBaseline(nn.Module):
    def __init__(self, x_dim, z_dim, h_dim, y_dim):
        super(BaseBaseline, self).__init__()
        self.bvp_embedding = BVPEmbed(x_dim)
        self.eda_embedding = EDAEmbed(x_dim)
        self.classifier = Classifier(2 * x_dim, h_dim, y_dim)

    def feature_cnn(self, bvp, eda):
        bvp_f = self.bvp_embedding(bvp)
        eda_f = self.eda_embedding(eda)
        x_f = torch.cat((bvp_f, eda_f), dim=-1)
        return x_f

    def forward(self, bvp, eda):
        z = self.feature_cnn(bvp, eda)
        # z = self.feature_fc(z)
        out = self.classifier(z)

        return out
