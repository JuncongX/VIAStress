import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F
from torch.autograd import Variable
from typing import Optional, List
import math
from torch.distributions import Normal

from VIAStress_wo_MMDG.model_pre import BVPEncoder, BVPDecoder, EDAEncoder, EDADecoder, ADClassifier


class Model(nn.Module):
    def __init__(self, x_dim, r_dim, h_dim, y_dim, bvp_encoder_vae=None, bvp_decoder_vae=None, eda_encoder_vae=None,
                 eda_decoder_vae=None):
        super(Model, self).__init__()

        self.bvp_encoder_fc = nn.Sequential(
            nn.Linear(32 * 30, x_dim),
            nn.ReLU(inplace=True),
            nn.Linear(x_dim, x_dim),
        )

        self.eda_encoder_fc = nn.Sequential(
            nn.Linear(32 * 15, x_dim),
            nn.ReLU(inplace=True),
            nn.Linear(x_dim, x_dim),
        )

        self.bvp_encoder_vae = BVPEncoder(128, r_dim) if bvp_encoder_vae is None else bvp_encoder_vae
        self.bvp_decoder_vae = BVPDecoder(r_dim) if bvp_decoder_vae is None else bvp_decoder_vae
        self.eda_encoder_vae = EDAEncoder(128, r_dim) if eda_encoder_vae is None else eda_encoder_vae
        self.eda_decoder_vae = EDADecoder(r_dim) if eda_decoder_vae is None else eda_decoder_vae
        self.classifier = ADClassifier(2 * x_dim, r_dim, h_dim, y_dim)

    def feature_cnn(self, bvp, eda):
        x_bvp = self.bvp_encoder_vae.conv_1(bvp)
        x_bvp = self.bvp_encoder_vae.conv_2(x_bvp)
        x_bvp = self.bvp_encoder_vae.conv_3(x_bvp)
        bz_bvp, _, _ = x_bvp.shape
        x_bvp = x_bvp.view(bz_bvp, -1)
        x_bvp = self.bvp_encoder_fc(x_bvp)

        x_eda = self.eda_encoder_vae.conv_1(eda)
        x_eda = self.eda_encoder_vae.conv_2(x_eda)
        x_eda = self.eda_encoder_vae.conv_3(x_eda)
        bz_eda, _, _ = x_eda.shape
        x_eda = x_eda.view(bz_eda, -1)
        x_eda = self.eda_encoder_fc(x_eda)

        x_f = torch.cat((x_bvp, x_eda), dim=-1)

        return x_f, x_bvp, x_eda

    def reparameterize(self, mu, logvar, num_samples=1):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn((num_samples, *std.shape), device=std.device)
        z_samples = mu.unsqueeze(0) + eps * std.unsqueeze(0)
        return z_samples  # shape: (num_samples, batch, latent_dim)

    def forward(self, bvp, eda):
        # BVP VAE
        mu_bvp, logvar_bvp = self.bvp_encoder_vae(bvp)
        z_bvp_samples = self.reparameterize(mu_bvp, logvar_bvp)  # shape: (5, batch, r_dim)
        recon_bvp_list = [self.bvp_decoder_vae(z) for z in z_bvp_samples]

        # EDA VAE
        mu_eda, logvar_eda = self.eda_encoder_vae(eda)
        z_eda_samples = self.reparameterize(mu_eda, logvar_eda)
        recon_eda_list = [self.eda_decoder_vae(z) for z in z_eda_samples]

        z_bvp = z_bvp_samples.mean(dim=0)
        z_eda = z_eda_samples.mean(dim=0)

        # CNN + classifier
        z, bvp_z, eda_z = self.feature_cnn(bvp, eda)
        out = self.classifier(z, z_bvp + z_eda)

        recon_bvp = torch.stack(recon_bvp_list, dim=0)
        recon_eda = torch.stack(recon_eda_list, dim=0)

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
    parser.add_argument('--x_dim', type=int, default=128)
    parser.add_argument('--y_dim', type=int, default=3)
    parser.add_argument('--h_dim', type=int, default=256)
    parser.add_argument('--r_dim', type=int, default=64)
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

    # print(eda.unsqueeze(0).expand_as(recon_eda))
