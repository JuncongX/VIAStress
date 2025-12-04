import torch
import torch.nn as nn
from VIAStress.model import BVPEncoder, BVPDecoder, EDAEncoder, EDADecoder, ProjectHead, EncoderTrans, FeatureEmbed

import torch
import torch.nn as nn
import torch.nn.functional as F


class ADClassifier(nn.Module):
    def __init__(self, z_dim, r_dim, h_dim, y_dim, MLP_dim):
        super(ADClassifier, self).__init__()
        self.z_dim = z_dim
        self.y_dim = y_dim
        self.h_dim = h_dim
        self.r_dim = r_dim

        self.hidden_layer = nn.Sequential(
            nn.Linear(self.z_dim, self.h_dim),
            nn.LeakyReLU(),
            nn.Linear(self.h_dim, self.h_dim),
            nn.LeakyReLU()
        )
        self.r_enc_1 = nn.Sequential(
            nn.LeakyReLU(),
            nn.Linear(self.r_dim, self.r_dim),
            nn.LeakyReLU(),
            nn.Linear(self.r_dim, self.r_dim),
            nn.LeakyReLU(),
        )

        self.last_fc_w = nn.Linear(self.h_dim, self.y_dim)

        self.last_fc = nn.Parameter(self.last_fc_w.weight)

        MLP_trans = []
        for i in range(len(MLP_dim) - 1):
            # MLP_trans.append(nn.BatchNorm1d(MLP_dim[i]))
            MLP_trans.append(nn.LeakyReLU())
            MLP_trans.append(nn.Linear(in_features=MLP_dim[i], out_features=MLP_dim[i + 1]))

        self.MLP_trans = nn.Sequential(*MLP_trans)

    def mix_w_q(self, q):
        q_encoded = self.r_enc_1(q)

        w_expand = self.last_fc.expand(
            [q_encoded.shape[0], self.last_fc.shape[0], self.last_fc.shape[1]])  # bn, y_dim, h_dim
        q_expand = q_encoded.unsqueeze(1).expand(
            [q_encoded.shape[0], w_expand.shape[1], q_encoded.shape[1]])  # bn, y_dim, r_dim

        w_q_cat = torch.cat([q_expand, w_expand], dim=2)  # bn, y_dim, h_dim + r_dim
        # print("w_q_cat:", w_q_cat.shape)
        w_q_cat_view = w_q_cat.view(-1, w_q_cat.shape[-1])  # bn * y_dim, h_dim + r_dim
        # print("w_q_cat_view:", w_q_cat_view.shape)

        new_w = self.MLP_trans(w_q_cat_view)  # 计算新的权重
        # print("new_w:", new_w.shape)
        new_w = new_w.view(w_expand.shape)  # 变回 (bn, y_dim, h_dim)
        new_w = new_w + w_expand  # 残差连接

        return new_w

    def forward(self, z, q):
        hidden = self.hidden_layer(z)
        new_w = self.mix_w_q(q)
        hidden_us = hidden.unsqueeze(2)  # bn, h_dim, 1
        new_w_x = torch.bmm(new_w, hidden_us)  # bn, y_dim, 1
        new_w_x = new_w_x.squeeze(2)  # bn, y_dim

        return new_w_x


class Model(nn.Module):
    def __init__(self, x_dim, r_dim, h_dim, y_dim, bvp_encoder_vae=None, bvp_decoder_vae=None, eda_encoder_vae=None,
                 eda_decoder_vae=None, mlp_dim=None):
        super(Model, self).__init__()
        self.feature_cnn = FeatureEmbed(x_dim)
        self.feature_fc = nn.Sequential(
            nn.LeakyReLU(inplace=True),
            nn.Linear(2 * x_dim, 2 * x_dim),
            nn.LeakyReLU(inplace=True),
            nn.Linear(2 * x_dim, 2 * x_dim),
            nn.LeakyReLU(inplace=True),
        )
        self.bvp_encoder_vae = BVPEncoder(128, r_dim) if bvp_encoder_vae is None else bvp_encoder_vae
        self.bvp_decoder_vae = BVPDecoder(r_dim) if bvp_decoder_vae is None else bvp_decoder_vae
        self.eda_encoder_vae = EDAEncoder(128, r_dim) if eda_encoder_vae is None else eda_encoder_vae
        self.eda_decoder_vae = EDADecoder(r_dim) if eda_decoder_vae is None else eda_decoder_vae
        self.classifier = ADClassifier(2 * x_dim, r_dim, h_dim, y_dim, mlp_dim)

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
        z = self.feature_fc(z)
        out = self.classifier(z, z_bvp + z_eda)

        recon_bvp = torch.stack(recon_bvp_list, dim=0)
        recon_eda = torch.stack(recon_eda_list, dim=0)

        bvp_z = F.leaky_relu(bvp_z)
        eda_z = F.leaky_relu(eda_z)

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

    model = Model(args.x_dim, args.r_dim, args.h_dim, args.y_dim, mlp_dim=[(args.h_dim + args.r_dim), 2 * args.h_dim, args.h_dim])
    out, bvp_z, eda_z, recon_bvp, mu_bvp, logvar_bvp, recon_eda, mu_eda, logvar_eda = model(bvp, eda)
    print(out.shape)
    print(recon_bvp.shape)
    print(recon_eda.shape)
