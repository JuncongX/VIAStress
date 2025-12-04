import torch
import torch.nn as nn
from VIAStress.model import EDAEmbed, EDAEncoder, EDADecoder, ADClassifier


class FeatureEmbed(nn.Module):
    def __init__(self, x_dim):
        super(FeatureEmbed, self).__init__()
        self.eda_embedding = EDAEmbed(x_dim)

    def forward(self, eda):
        eda_f = self.eda_embedding(eda)
        return eda_f


class Model(nn.Module):
    def __init__(self, x_dim, r_dim, h_dim, y_dim, eda_encoder_vae=None, eda_decoder_vae=None):
        super(Model, self).__init__()
        self.feature_cnn = FeatureEmbed(x_dim)
        self.eda_encoder_vae = EDAEncoder(128, r_dim) if eda_encoder_vae is None else eda_encoder_vae
        self.eda_decoder_vae = EDADecoder(r_dim) if eda_decoder_vae is None else eda_decoder_vae
        self.classifier = ADClassifier(x_dim, r_dim, h_dim, y_dim)

    def reparameterize(self, mu, logvar, num_samples=1):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn((num_samples, *std.shape), device=std.device)
        z_samples = mu.unsqueeze(0) + eps * std.unsqueeze(0)
        return z_samples  # shape: (num_samples, batch, latent_dim)

    def forward(self, eda):
        mu_eda, logvar_eda = self.eda_encoder_vae(eda)
        z_eda_samples = self.reparameterize(mu_eda, logvar_eda)
        recon_eda_list = [self.eda_decoder_vae(z) for z in z_eda_samples]

        z_eda = z_eda_samples.mean(dim=0)

        z = self.feature_cnn(eda)
        out = self.classifier(z, z_eda)

        recon_eda = torch.stack(recon_eda_list, dim=0)

        return out, recon_eda, mu_eda, logvar_eda
