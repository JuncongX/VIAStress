from VIAStress_wo_MMDG.model_pre import EDAEncoder, EDADecoder, BVPEncoder, FeatureEmbed, ADClassifier
import torch
import torch.nn as nn


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
            nn.Tanh()
        )

    def forward(self, z):
        x = self.fc(z)
        x = x.view(-1, 32, 30)

        x = self.deconv_1(x)
        x = self.deconv_2(x)
        x = self.deconv_3(x)
        x = self.deconv_4(x)
        return x


class Model(nn.Module):
    def __init__(self, x_dim, r_dim, h_dim, y_dim, bvp_encoder_vae=None, bvp_decoder_vae=None, eda_encoder_vae=None,
                 eda_decoder_vae=None):
        super(Model, self).__init__()
        self.feature_cnn = FeatureEmbed(x_dim)
        self.feature_fc = nn.Sequential(
            nn.Linear(2 * x_dim, 2 * x_dim),
            nn.LeakyReLU(inplace=True),
            nn.Linear(2 * x_dim, 2 * x_dim)
        )
        self.bvp_encoder_vae = BVPEncoder(x_dim, r_dim) if bvp_encoder_vae is None else bvp_encoder_vae
        self.bvp_decoder_vae = BVPDecoder(r_dim) if bvp_decoder_vae is None else bvp_decoder_vae
        self.eda_encoder_vae = EDAEncoder(x_dim, r_dim) if eda_encoder_vae is None else eda_encoder_vae
        self.eda_decoder_vae = EDADecoder(r_dim) if eda_decoder_vae is None else eda_decoder_vae
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
