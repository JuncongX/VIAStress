# (R) VAE
import torch
import torch.nn as nn
import copy
from VIAStress_wo_MMDG.model_pre import BVPEncoder, BVPDecoder, EDAEncoder, EDADecoder, ADClassifier

class Classifier(nn.Module):
    def __init__(self, z_dim, h_dim, y_dim):
        super(Classifier, self).__init__()

        self.fc = nn.Sequential(
            nn.Linear(z_dim, h_dim),
            nn.LeakyReLU(inplace=True),
            nn.Linear(h_dim, h_dim),
            nn.LeakyReLU(inplace=True),
            nn.Linear(h_dim, y_dim),
        )

    def forward(self, z):
        return self.fc(z)


class Model(nn.Module):
    def __init__(self, x_dim, r_dim, h_dim, y_dim, bvp_encoder_vae=None, bvp_decoder_vae=None, eda_encoder_vae=None,
                 eda_decoder_vae=None):
        super(Model, self).__init__()
        self.bvp_encoder_vae = BVPEncoder(x_dim, r_dim) if bvp_encoder_vae is None else bvp_encoder_vae
        self.bvp_decoder_vae = BVPDecoder(r_dim) if bvp_decoder_vae is None else bvp_decoder_vae
        self.eda_encoder_vae = EDAEncoder(x_dim, r_dim) if eda_encoder_vae is None else eda_encoder_vae
        self.eda_decoder_vae = EDADecoder(r_dim) if eda_decoder_vae is None else eda_decoder_vae
        self.feature_fc = nn.Sequential(
            nn.Linear(2 * r_dim, 2 * r_dim),
            nn.LeakyReLU(inplace=True),
            nn.Linear(2 * r_dim, 2 * r_dim)
        )
        self.classifier = Classifier(2 * r_dim, h_dim, y_dim)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, bvp, eda):
        mu_bvp, logvar_bvp = self.bvp_encoder_vae(bvp)
        z_bvp = self.reparameterize(mu_bvp, logvar_bvp)
        recon_bvp = self.bvp_decoder_vae(z_bvp)

        mu_eda, logvar_eda = self.eda_encoder_vae(eda)
        z_eda = self.reparameterize(mu_eda, logvar_eda)
        recon_eda = self.eda_decoder_vae(z_eda)

        z = torch.cat((z_bvp, z_eda), dim=-1)
        z = self.feature_fc(z)
        out = self.classifier(z)

        return out, recon_bvp, mu_bvp, logvar_bvp, recon_eda, mu_eda, logvar_eda


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
    out, recon_bvp, mu_bvp, logvar_bvp, recon_eda, mu_eda, logvar_eda = model(bvp, eda)
    print(out.shape)
    print(recon_bvp.shape)
    print(recon_eda.shape)
