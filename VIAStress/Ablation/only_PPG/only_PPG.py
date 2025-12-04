import torch
import torch.nn as nn
from VIAStress.model import BVPEmbed, BVPEncoder, BVPDecoder, ADClassifier


class FeatureEmbed(nn.Module):
    def __init__(self, x_dim):
        super(FeatureEmbed, self).__init__()
        self.bvp_embedding = BVPEmbed(x_dim)

    def forward(self, bvp):
        bvp_f = self.bvp_embedding(bvp)
        return bvp_f


class Model(nn.Module):
    def __init__(self, x_dim, r_dim, h_dim, y_dim, bvp_encoder_vae=None, bvp_decoder_vae=None):
        super(Model, self).__init__()
        self.feature_cnn = FeatureEmbed(x_dim)
        self.bvp_encoder_vae = BVPEncoder(128, r_dim) if bvp_encoder_vae is None else bvp_encoder_vae
        self.bvp_decoder_vae = BVPDecoder(r_dim) if bvp_decoder_vae is None else bvp_decoder_vae
        self.classifier = ADClassifier(x_dim, r_dim, h_dim, y_dim)

    def reparameterize(self, mu, logvar, num_samples=1):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn((num_samples, *std.shape), device=std.device)
        z_samples = mu.unsqueeze(0) + eps * std.unsqueeze(0)
        return z_samples

    def forward(self, bvp):
        mu_bvp, logvar_bvp = self.bvp_encoder_vae(bvp)
        z_bvp_samples = self.reparameterize(mu_bvp, logvar_bvp)  # shape: (5, batch, r_dim)
        recon_bvp_list = [self.bvp_decoder_vae(z) for z in z_bvp_samples]

        z_bvp = z_bvp_samples.mean(dim=0)

        recon_bvp = torch.stack(recon_bvp_list, dim=0)

        z = self.feature_cnn(bvp)
        out = self.classifier(z, z_bvp)

        return out, recon_bvp, mu_bvp, logvar_bvp
