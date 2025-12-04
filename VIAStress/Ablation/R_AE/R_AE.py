import torch
import torch.nn as nn
import torch.nn.functional as F

from VIAStress_wo_MMDG.model_pre import ADClassifier, BVPDecoder, EDADecoder
from VIAStress.model import FeatureEmbed, EncoderTrans, ProjectHead


class BVPEncoder(nn.Module):
    def __init__(self, x_dim, r_dim):
        super(BVPEncoder, self).__init__()
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

        self.fc = nn.Linear(32 * 30, x_dim)
        self.fc_ = nn.Linear(x_dim, r_dim)

    def forward(self, x):
        x = self.conv_1(x)
        x = self.conv_2(x)
        x = self.conv_3(x)
        x = x.view(-1, 32 * 30)
        x = self.fc(x)
        x = F.relu(x, inplace=True)
        z = self.fc_(x)
        return z


class EDAEncoder(nn.Module):
    def __init__(self, x_dim, r_dim):
        super(EDAEncoder, self).__init__()
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

        self.fc = nn.Linear(32 * 15, x_dim)
        self.fc_ = nn.Linear(x_dim, r_dim)

    def forward(self, x):
        x = self.conv_1(x)
        x = self.conv_2(x)
        x = self.conv_3(x)
        x = x.view(-1, 32 * 15)
        x = self.fc(x)
        x = F.relu(x, inplace=True)
        z = self.fc_(x)
        return z


class Model(nn.Module):
    def __init__(self, x_dim, r_dim, h_dim, y_dim, bvp_encoder_ae=None, bvp_decoder_ae=None, eda_encoder_ae=None,
                 eda_decoder_ae=None):
        super(Model, self).__init__()
        self.feature_cnn = FeatureEmbed(x_dim)
        self.bvp_encoder_vae = BVPEncoder(128, r_dim) if bvp_encoder_ae is None else bvp_encoder_ae
        self.bvp_decoder_vae = BVPDecoder(r_dim) if bvp_decoder_ae is None else bvp_decoder_ae
        self.eda_encoder_vae = EDAEncoder(128, r_dim) if eda_encoder_ae is None else eda_encoder_ae
        self.eda_decoder_vae = EDADecoder(r_dim) if eda_decoder_ae is None else eda_decoder_ae
        self.classifier = ADClassifier(2 * x_dim, r_dim, h_dim, y_dim)

    def forward(self, bvp, eda):
        z_bvp = self.bvp_encoder_vae(bvp)
        recon_bvp = self.bvp_decoder_vae(z_bvp)

        z_eda = self.eda_encoder_vae(eda)
        recon_eda = self.eda_decoder_vae(z_eda)

        z, bvp_z, eda_z = self.feature_cnn(bvp, eda)
        out = self.classifier(z, z_bvp + z_eda)

        return out, bvp_z, eda_z, recon_bvp, recon_eda
