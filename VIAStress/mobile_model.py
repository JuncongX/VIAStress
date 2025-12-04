import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F
from torch.autograd import Variable
from typing import Optional, List
import math
from VIAStress_wo_MMDG.model_pre import BVPEncoder, EDAEncoder, BVPEmbed, EDAEmbed, ADClassifier


class FeatureEmbed(nn.Module):
    def __init__(self, x_dim):
        super(FeatureEmbed, self).__init__()
        self.bvp_embedding = BVPEmbed(x_dim)
        self.eda_embedding = EDAEmbed(x_dim)

    def forward(self, bvp, eda):
        bvp_f = self.bvp_embedding(bvp)
        eda_f = self.eda_embedding(eda)
        x_f = torch.cat((bvp_f, eda_f), dim=-1)
        return x_f, bvp_f, eda_f


class Model(nn.Module):
    def __init__(self, x_dim, r_dim, h_dim, y_dim):
        super(Model, self).__init__()
        self.feature_cnn = FeatureEmbed(x_dim)
        self.bvp_encoder_vae = BVPEncoder(128, r_dim)
        self.eda_encoder_vae = EDAEncoder(128, r_dim)
        self.classifier = ADClassifier(2 * x_dim, r_dim, h_dim, y_dim)

    def reparameterize(self, mu, logvar, num_samples=1):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn((num_samples, *std.shape), device=std.device)
        z_samples = mu.unsqueeze(0) + eps * std.unsqueeze(0)
        return z_samples  # shape: (num_samples, batch, latent_dim)

    def forward(self, bvp, eda):
        # BVP VAE
        mu_bvp, logvar_bvp = self.bvp_encoder_vae(bvp)
        z_bvp_samples = self.reparameterize(mu_bvp, logvar_bvp)  # shape: (5, batch, r_dim)

        # EDA VAE
        min_vals = eda.min(dim=2, keepdim=True).values  # shape: (3, 1, 1)
        max_vals = eda.max(dim=2, keepdim=True).values  # shape: (3, 1, 1)
        denom = max_vals - min_vals
        denom[denom == 0] = 1
        normalized_eda = 2 * (eda - min_vals) / denom - 1

        mu_eda, logvar_eda = self.eda_encoder_vae(normalized_eda)
        z_eda_samples = self.reparameterize(mu_eda, logvar_eda)

        z_bvp = z_bvp_samples.mean(dim=0)
        z_eda = z_eda_samples.mean(dim=0)

        # CNN + classifier
        z, bvp_z, eda_z = self.feature_cnn(bvp, eda)
        # z = self.feature_fc(z)
        out = self.classifier(z, z_bvp + z_eda)


        return out


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='VIAStress')
    parser.add_argument('--x_dim', type=int, default=256)
    parser.add_argument('--y_dim', type=int, default=2)
    parser.add_argument('--h_dim', type=int, default=256)
    parser.add_argument('--r_dim', type=int, default=256)

    args = parser.parse_args()

    bvp = torch.rand((1, 1, 1920))
    eda = torch.rand((1, 1, 120))

    # eda_emb = EDAEmbed(128)
    # print(eda_emb(eda).shape)

    model = Model(args.x_dim, args.r_dim, args.h_dim, args.y_dim)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total Parameters: {total_params}")
    torch.save(model.state_dict(), "viastress_mobile.pt")
    out = model(bvp, eda)
    print(out.shape)

    example_inputs = (bvp, eda)
    traced_model = torch.jit.trace(model, example_inputs)
    traced_model.save("viastress_mobile_traced.pt")