from VIAStress.model import EDAEncoder, EDADecoder, BVPEncoder, FeatureEmbed, ADClassifier, ProjectHead, EncoderTrans
import torch
import torch.nn as nn
import torch.nn.functional as F


class BVPDecoder(nn.Module):
    def __init__(self, r_dim):
        super(BVPDecoder, self).__init__()
        self.fc = nn.Linear(r_dim, 32 * 30)

        self.deconv_1 = nn.Sequential(
            nn.ConvTranspose1d(32, 16, kernel_size=5, stride=2, padding=2, output_padding=1),
            nn.BatchNorm1d(16),
            nn.ReLU(inplace=True)
        )

        self.deconv_2 = nn.Sequential(
            nn.ConvTranspose1d(16, 8, kernel_size=7, stride=2, padding=3, output_padding=1),
            nn.BatchNorm1d(8),
            nn.ReLU(inplace=True)
        )

        self.deconv_3 = nn.Sequential(
            nn.ConvTranspose1d(8, 4, kernel_size=9, stride=4, padding=4, output_padding=3),
            nn.BatchNorm1d(4),
            nn.ReLU(inplace=True)
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
