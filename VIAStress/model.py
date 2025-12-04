import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F
from torch.autograd import Variable
from typing import Optional, List
import math
from VIAStress_wo_MMDG.model_pre import BVPEncoder, BVPDecoder, EDAEncoder, EDADecoder, BVPEmbed, EDAEmbed, ADClassifier, \
    Model as Model_pre


class FeatureEmbed(nn.Module):
    def __init__(self, x_dim):
        super(FeatureEmbed, self).__init__()
        self.bvp_embedding = BVPEmbed(x_dim)
        self.eda_embedding = EDAEmbed(x_dim)

    def forward(self, bvp, eda):
        bvp_f = self.bvp_embedding(bvp)
        eda_f = self.eda_embedding(eda)
        x_f = torch.cat((bvp_f, eda_f), dim=-1)
        return x_f, bvp_f, eda_f


class EncoderTrans(nn.Module):
    def __init__(self, input_dim=128, hidden=128, out_dim=128):
        super(EncoderTrans, self).__init__()
        self.enc_net = nn.Sequential(
            nn.Linear(input_dim, hidden),
            nn.ReLU(inplace=True),
            # nn.Linear(hidden, hidden),
            # nn.ReLU(inplace=True),
            nn.Linear(hidden, out_dim)
        )

    def forward(self, feat):
        feat = self.enc_net(feat)
        return feat


class ProjectHead(nn.Module):
    def __init__(self, input_dim=128, hidden_dim=128, out_dim=64):
        super(ProjectHead, self).__init__()
        self.head = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            # nn.BatchNorm1d(hidden_dim),
            # nn.ReLU(inplace=True),
            # nn.Linear(hidden_dim, hidden_dim),
            # nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, out_dim)
        )

    def forward(self, feat):
        feat = F.normalize(self.head(feat), dim=1)
        return feat


class Model(Model_pre):
    def __init__(self, x_dim, r_dim, h_dim, y_dim, bvp_encoder_vae=None, bvp_decoder_vae=None, eda_encoder_vae=None,
                 eda_decoder_vae=None):
        super(Model, self).__init__(x_dim, r_dim, h_dim, y_dim, bvp_encoder_vae, bvp_decoder_vae, eda_encoder_vae,
                                    eda_decoder_vae)
        self.feature_cnn = FeatureEmbed(x_dim)

    def forward(self, bvp, eda):
        # BVP VAE
        mu_bvp, logvar_bvp = self.bvp_encoder_vae(bvp)
        z_bvp_samples = self.reparameterize(mu_bvp, logvar_bvp)  # shape: (5, batch, r_dim)
        recon_bvp_list = [self.bvp_decoder_vae(z) for z in z_bvp_samples]

        # EDA VAE
        min_vals = eda.min(dim=2, keepdim=True).values  # shape: (3, 1, 1)
        max_vals = eda.max(dim=2, keepdim=True).values  # shape: (3, 1, 1)
        denom = max_vals - min_vals
        denom[denom == 0] = 1
        normalized_eda = 2 * (eda - min_vals) / denom - 1

        mu_eda, logvar_eda = self.eda_encoder_vae(normalized_eda)
        z_eda_samples = self.reparameterize(mu_eda, logvar_eda)
        recon_eda_list = [self.eda_decoder_vae(z) for z in z_eda_samples]

        z_bvp = z_bvp_samples.mean(dim=0)
        z_eda = z_eda_samples.mean(dim=0)

        # CNN + classifier
        z, bvp_z, eda_z = self.feature_cnn(bvp, eda)
        # z = self.feature_fc(z)
        out = self.classifier(z, z_bvp + z_eda)

        recon_bvp = torch.stack(recon_bvp_list, dim=0)
        recon_eda = torch.stack(recon_eda_list, dim=0)

        # bvp_z = F.relu(bvp_z)
        # eda_z = F.relu(eda_z)

        return out, bvp_z, eda_z, recon_bvp, mu_bvp, logvar_bvp, recon_eda, mu_eda, logvar_eda


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='NPStress')
    parser.add_argument('--k', type=int, default=5, help="KFold")
    # parser.add_argument('--epoch', type=int, default=150)
    parser.add_argument('--epoch', type=int, default=150)
    parser.add_argument('--batch_size', type=int, default=120)
    parser.add_argument('--LR', type=float, default=0.0001)
    parser.add_argument('--save_path', type=str, default="./checkpoints")
    parser.add_argument('--x_dim', type=int, default=256)
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
    out, bvp_z, eda_z, recon_bvp, mu_bvp, logvar_bvp, recon_eda, mu_eda, logvar_eda = model(bvp, eda)
    print(out.shape)
    print(recon_bvp.shape)
    print(recon_eda.shape)
