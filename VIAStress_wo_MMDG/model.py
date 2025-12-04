import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F
from torch.autograd import Variable
from typing import Optional, List
import math
from torch.distributions import Normal


class BVPEmbed(nn.Module):
    def __init__(self, x_dim):
        super(BVPEmbed, self).__init__()
        self.pooling = 4
        self.conv_1 = nn.Sequential(
            nn.Conv1d(1, 8, 9, 1, 4),
            nn.BatchNorm1d(8),
            nn.LeakyReLU(inplace=True),
            nn.MaxPool1d(self.pooling, self.pooling)
        )

        self.conv_2 = nn.Sequential(
            nn.Conv1d(8, 16, 7, 1, 3),
            nn.BatchNorm1d(16),
            nn.LeakyReLU(inplace=True),
            nn.MaxPool1d(self.pooling, self.pooling)
        )

        self.conv_3 = nn.Sequential(
            nn.Conv1d(16, 32, 3, 1, 1),
            nn.BatchNorm1d(32),
            nn.LeakyReLU(inplace=True),
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
        x = F.leaky_relu(x, inplace=True)
        x = self.fc2(x)
        return x


class EDAEmbed(nn.Module):
    def __init__(self, x_dim):
        super(EDAEmbed, self).__init__()
        self.pooling = 2
        self.conv_1 = nn.Sequential(
            nn.Conv1d(1, 8, 9, 1, 4),
            nn.BatchNorm1d(8),
            nn.LeakyReLU(inplace=True),
            nn.MaxPool1d(self.pooling, self.pooling)
        )

        self.conv_2 = nn.Sequential(
            nn.Conv1d(8, 16, 7, 1, 3),
            nn.BatchNorm1d(16),
            nn.LeakyReLU(inplace=True),
            nn.MaxPool1d(self.pooling, self.pooling)
        )

        self.conv_3 = nn.Sequential(
            nn.Conv1d(16, 32, 3, 1, 1),
            nn.BatchNorm1d(32),
            nn.LeakyReLU(inplace=True),
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
        x = F.leaky_relu(x, inplace=True)
        x = self.fc2(x)
        return x


class FeatureEmbed(nn.Module):
    def __init__(self, x_dim):
        super(FeatureEmbed, self).__init__()
        self.bvp_embedding = BVPEmbed(x_dim)
        self.eda_embedding = EDAEmbed(x_dim)

    def forward(self, bvp, eda):
        bvp_f = self.bvp_embedding(bvp)
        eda_f = self.eda_embedding(eda)
        x_f = torch.cat((bvp_f, eda_f), dim=-1)
        return x_f


class BVPEncoder(nn.Module):
    def __init__(self, x_dim, r_dim):
        super(BVPEncoder, self).__init__()
        self.pooling = 4
        self.conv_1 = nn.Sequential(
            nn.Conv1d(1, 8, 9, 1, 4),
            nn.BatchNorm1d(8),
            nn.LeakyReLU(inplace=True),
            nn.MaxPool1d(self.pooling, self.pooling)
        )

        self.conv_2 = nn.Sequential(
            nn.Conv1d(8, 16, 7, 1, 3),
            nn.BatchNorm1d(16),
            nn.LeakyReLU(inplace=True),
            nn.MaxPool1d(self.pooling, self.pooling)
        )

        self.conv_3 = nn.Sequential(
            nn.Conv1d(16, 32, 3, 1, 1),
            nn.BatchNorm1d(32),
            nn.LeakyReLU(inplace=True),
            nn.MaxPool1d(self.pooling, self.pooling)
        )

        self.fc = nn.Linear(32 * 30, x_dim)
        self.fc_mu = nn.Linear(x_dim, r_dim)
        self.fc_logvar = nn.Linear(x_dim, r_dim)

    def forward(self, x):
        x = self.conv_1(x)
        x = self.conv_2(x)
        x = self.conv_3(x)
        x = x.view(-1, 32 * 30)
        x = self.fc(x)
        mu = self.fc_mu(x)
        logvar = self.fc_logvar(x)
        return mu, logvar


class BVPDecoder(nn.Module):
    def __init__(self, r_dim):
        super(BVPDecoder, self).__init__()
        self.fc = nn.Linear(r_dim, 32 * 30)

        self.deconv_1 = nn.Sequential(
            nn.ConvTranspose1d(32, 16, kernel_size=5, stride=2, padding=2, output_padding=1),
            nn.BatchNorm1d(16),
            nn.LeakyReLU(inplace=True)
        )

        self.deconv_2 = nn.Sequential(
            nn.ConvTranspose1d(16, 8, kernel_size=7, stride=2, padding=3, output_padding=1),
            nn.BatchNorm1d(8),
            nn.LeakyReLU(inplace=True)
        )

        self.deconv_3 = nn.Sequential(
            nn.ConvTranspose1d(8, 4, kernel_size=9, stride=4, padding=4, output_padding=3),
            nn.BatchNorm1d(4),
            nn.LeakyReLU(inplace=True)
        )

        self.deconv_4 = nn.Sequential(
            nn.ConvTranspose1d(4, 1, kernel_size=9, stride=4, padding=4, output_padding=3),
            # nn.Tanh()
            nn.Sigmoid()
        )

    def forward(self, z):
        x = self.fc(z)
        x = x.view(-1, 32, 30)

        x = self.deconv_1(x)
        x = self.deconv_2(x)
        x = self.deconv_3(x)
        x = self.deconv_4(x)
        return x


class EDAEncoder(nn.Module):
    def __init__(self, x_dim, r_dim):
        super(EDAEncoder, self).__init__()
        self.pooling = 2
        self.conv_1 = nn.Sequential(
            nn.Conv1d(1, 8, 9, 1, 4),
            nn.BatchNorm1d(8),
            nn.LeakyReLU(inplace=True),
            nn.MaxPool1d(self.pooling, self.pooling)
        )

        self.conv_2 = nn.Sequential(
            nn.Conv1d(8, 16, 7, 1, 3),
            nn.BatchNorm1d(16),
            nn.LeakyReLU(inplace=True),
            nn.MaxPool1d(self.pooling, self.pooling)
        )

        self.conv_3 = nn.Sequential(
            nn.Conv1d(16, 32, 3, 1, 1),
            nn.BatchNorm1d(32),
            nn.LeakyReLU(inplace=True),
            nn.MaxPool1d(self.pooling, self.pooling)
        )

        self.fc = nn.Linear(32 * 15, x_dim)
        self.fc_mu = nn.Linear(x_dim, r_dim)
        self.fc_logvar = nn.Linear(x_dim, r_dim)

    def forward(self, x):
        x = self.conv_1(x)
        x = self.conv_2(x)
        x = self.conv_3(x)
        x = x.view(-1, 32 * 15)
        x = self.fc(x)
        mu = self.fc_mu(x)
        logvar = self.fc_logvar(x)
        return mu, logvar


class EDADecoder(nn.Module):
    def __init__(self, r_dim):
        super(EDADecoder, self).__init__()
        self.fc = nn.Linear(r_dim, 32 * 15)

        self.deconv_1 = nn.Sequential(
            nn.ConvTranspose1d(32, 16, 3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm1d(16),
            nn.LeakyReLU(inplace=True)
        )

        self.deconv_2 = nn.Sequential(
            nn.ConvTranspose1d(16, 8, 7, stride=2, padding=3, output_padding=1),
            nn.BatchNorm1d(8),
            nn.LeakyReLU(inplace=True)
        )

        self.deconv_3 = nn.Sequential(
            nn.ConvTranspose1d(8, 1, 9, stride=2, padding=4, output_padding=1),
            nn.Tanh()
        )

    def forward(self, z):
        x = self.fc(z)
        x = x.view(-1, 32, 15)

        x = self.deconv_1(x)
        x = self.deconv_2(x)
        x = self.deconv_3(x)
        return x


class ADClassifier(nn.Module):
    def __init__(self, z_dim, r_dim, h_dim, y_dim):
        super(ADClassifier, self).__init__()
        self.z_dim = z_dim
        self.y_dim = y_dim
        self.h_dim = h_dim
        self.r_dim = r_dim



        self.hidden_layer_1 = nn.Sequential(
            nn.Linear(self.z_dim, self.h_dim),
            # nn.LayerNorm(self.h_dim),
            nn.LeakyReLU(),
        )
        self.r_enc_1 = nn.Sequential(
            nn.Linear(self.r_dim, self.r_dim),
            nn.LeakyReLU()
        )
        self.film_layer_1_beta = nn.Linear(self.r_dim, self.h_dim, bias=False)
        self.film_layer_1_gamma = nn.Linear(self.r_dim, self.h_dim, bias=False)
        self.film_layer_1_eta = nn.Linear(self.r_dim, self.h_dim, bias=False)
        self.film_layer_1_delta = nn.Linear(self.r_dim, self.h_dim, bias=False)

        self.hidden_layer_2 = nn.Sequential(
            # nn.LayerNorm(self.h_dim),
            nn.LeakyReLU(),
            # nn.Linear(self.h_dim, self.h_dim),
            nn.Linear(self.h_dim, self.y_dim),
            # nn.LayerNorm(self.h_dim),
            # nn.LeakyReLU(),
        )

        # self.film_layer_2_beta = nn.Linear(self.r_dim, self.h_dim, bias=False)
        # self.film_layer_2_gamma = nn.Linear(self.r_dim, self.h_dim, bias=False)
        # self.film_layer_2_eta = nn.Linear(self.r_dim, self.h_dim, bias=False)
        # self.film_layer_2_delta = nn.Linear(self.r_dim, self.h_dim, bias=False)
        #
        # self.final_projection = nn.Sequential(
        #     # nn.LayerNorm(self.h_dim),
        #     nn.LeakyReLU(),
        #     nn.Linear(self.h_dim, self.y_dim),
        #     # nn.LeakyReLU(),
        # )

    def forward(self, z, q):
        hidden_1 = self.hidden_layer_1(z)
        q_1 = self.r_enc_1(q)
        beta_1 = torch.tanh(self.film_layer_1_beta(q_1))
        gamma_1 = torch.tanh(self.film_layer_1_gamma(q_1))
        eta_1 = torch.tanh(self.film_layer_1_eta(q_1))
        delta_1 = torch.sigmoid(self.film_layer_1_delta(q_1))

        gamma_1 = gamma_1 * delta_1 + eta_1 * (1 - delta_1)
        beta_1 = beta_1 * delta_1 + eta_1 * (1 - delta_1)
        # print("gamma_1", gamma_1.shape)

        hidden_2 = torch.mul(hidden_1, gamma_1) + beta_1

        hidden_2 = self.hidden_layer_2(hidden_2)
        # beta_2 = torch.tanh(self.film_layer_2_beta(q))
        # gamma_2 = torch.tanh(self.film_layer_2_gamma(q))
        # eta_2 = torch.tanh(self.film_layer_2_eta(q))
        # delta_2 = torch.sigmoid(self.film_layer_2_delta(q))
        #
        # gamma_2 = gamma_2 * delta_2 + eta_2 * (1 - delta_2)
        # beta_2 = beta_2 * delta_2 + eta_2 * (1 - delta_2)
        #
        # hidden_3 = torch.mul(hidden_2, gamma_2) + beta_2

        return hidden_2


class Model(nn.Module):
    def __init__(self, x_dim, r_dim, h_dim, y_dim):
        super(Model, self).__init__()
        self.feature_cnn = FeatureEmbed(x_dim)
        self.feature_fc = nn.Sequential(
            nn.Linear(2 * x_dim, 2 * x_dim),
            nn.LeakyReLU(inplace=True),
            nn.Linear(2 * x_dim, 2 * x_dim)
        )
        self.bvp_encoder_vae = BVPEncoder(x_dim, r_dim)
        self.bvp_decoder_vae = BVPDecoder(r_dim)
        self.eda_encoder_vae = EDAEncoder(x_dim, r_dim)
        self.eda_decoder_vae = EDADecoder(r_dim)
        self.classifier = ADClassifier(2 * x_dim, r_dim, h_dim, y_dim)

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

        z = self.feature_cnn(bvp, eda)
        z = self.feature_fc(z)
        out = self.classifier(z, z_bvp + z_eda)

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
    out, recon_bvp, mu_bvp, logvar_bvp, recon_eda, mu_eda, logvar_eda = model(bvp, eda)
    print(out.shape)
    print(recon_bvp.shape)
    print(recon_eda.shape)