import torch
import torch.nn as nn
from VIAStress.model import BVPEncoder, BVPDecoder, EDAEncoder, EDADecoder, ProjectHead, EncoderTrans, FeatureEmbed

import torch
import torch.nn as nn
import torch.nn.functional as F


class ADClassifier(nn.Module):
    def __init__(self, z_dim, r_dim, h_dim, y_dim):
        super(ADClassifier, self).__init__()
        self.z_dim = z_dim
        self.y_dim = y_dim
        self.h_dim = h_dim
        self.r_dim = r_dim

        self.hidden_layer = nn.Sequential(
            nn.Linear(self.z_dim, self.h_dim),
            nn.ReLU(),
            nn.Linear(self.h_dim, self.h_dim),
            nn.ReLU()
        )
        self.r_enc = nn.Sequential(
            nn.ReLU(),
            nn.Linear(self.r_dim, self.r_dim),
            nn.ReLU(),
            nn.Linear(self.r_dim, self.r_dim),
            nn.ReLU(),
            nn.Linear(self.r_dim, self.r_dim * self.y_dim + self.y_dim)
        )

    def forward(self, z, q):
        hidden = self.hidden_layer(z)
        q_avg = q.mean(dim=0, keepdim=True)
        w_flatten = self.r_enc(q_avg)
        W = w_flatten[..., :self.r_dim * self.y_dim].reshape(self.r_dim, self.y_dim)  # 权重矩阵
        b = w_flatten[..., self.r_dim * self.y_dim:]  # 偏置项

        out = torch.matmul(hidden, W) + b  # 自适应全连接层
        return out


class Model(nn.Module):
    def __init__(self, x_dim, r_dim, h_dim, y_dim, bvp_encoder_vae=None, bvp_decoder_vae=None, eda_encoder_vae=None,
                 eda_decoder_vae=None):
        super(Model, self).__init__()
        self.feature_cnn = FeatureEmbed(x_dim)

        self.bvp_encoder_vae = BVPEncoder(128, r_dim) if bvp_encoder_vae is None else bvp_encoder_vae
        self.bvp_decoder_vae = BVPDecoder(r_dim) if bvp_decoder_vae is None else bvp_decoder_vae
        self.eda_encoder_vae = EDAEncoder(128, r_dim) if eda_encoder_vae is None else eda_encoder_vae
        self.eda_decoder_vae = EDADecoder(r_dim) if eda_decoder_vae is None else eda_decoder_vae
        self.classifier = ADClassifier(2 * x_dim, r_dim, h_dim, y_dim)

    def reparameterize(self, mu, logvar, num_samples=1):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn((num_samples, *std.shape), device=std.device)
        z_samples = mu.unsqueeze(0) + eps * std.unsqueeze(0)
        return z_samples

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
