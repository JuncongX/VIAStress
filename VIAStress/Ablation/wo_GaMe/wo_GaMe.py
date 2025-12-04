import torch
import torch.nn as nn
import torch.nn.functional as F

from VIAStress.model import BVPEncoder, BVPDecoder, EDAEncoder, EDADecoder, ProjectHead, EncoderTrans, FeatureEmbed
from VIAStress_wo_MMDG.model_pre import Model as Model_pre


class ADClassifier(nn.Module):
    def __init__(self, z_dim, r_dim, h_dim, y_dim):
        super(ADClassifier, self).__init__()
        self.z_dim = z_dim
        self.y_dim = y_dim
        self.h_dim = h_dim
        self.r_dim = r_dim

        self.hidden_layer_1 = nn.Sequential(
            nn.Linear(self.z_dim, self.h_dim),
            nn.ReLU(),
        )

        self.film_layer_1_beta = nn.Linear(self.r_dim, self.h_dim, bias=False)
        self.film_layer_1_gamma = nn.Linear(self.r_dim, self.h_dim, bias=False)
        self.film_layer_1_eta = nn.Linear(self.r_dim, self.h_dim, bias=False)
        self.film_layer_1_delta = nn.Linear(self.r_dim, self.h_dim, bias=False)

        self.hidden_layer_2 = nn.Sequential(
            nn.ReLU(),
            nn.Linear(self.h_dim, self.y_dim)
        )

    def forward(self, z, q):
        hidden_1 = self.hidden_layer_1(z)

        beta_1 = torch.tanh(self.film_layer_1_beta(q))
        gamma_1 = torch.tanh(self.film_layer_1_gamma(q))
        eta_1 = torch.tanh(self.film_layer_1_eta(q))
        delta_1 = torch.sigmoid(self.film_layer_1_delta(q))
        gamma_1 = gamma_1 * delta_1 + eta_1 * (1 - delta_1)
        beta_1 = beta_1 * delta_1 + eta_1 * (1 - delta_1)

        hidden_2 = torch.mul(hidden_1, gamma_1) + beta_1

        hidden_2 = self.hidden_layer_2(hidden_2)

        return hidden_2


class Model(Model_pre):
    def __init__(self, x_dim, r_dim, h_dim, y_dim, bvp_encoder_vae=None, bvp_decoder_vae=None, eda_encoder_vae=None,
                 eda_decoder_vae=None):
        super(Model, self).__init__(x_dim, r_dim, h_dim, y_dim, bvp_encoder_vae, bvp_decoder_vae, eda_encoder_vae,
                                    eda_decoder_vae)
        self.feature_cnn = FeatureEmbed(x_dim)
        self.classifier = ADClassifier(2 * x_dim, r_dim, h_dim, y_dim)

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

        return out, bvp_z, eda_z, recon_bvp, mu_bvp, logvar_bvp, recon_eda, mu_eda, logvar_eda
