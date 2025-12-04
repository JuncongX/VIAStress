import torch
import torch.nn as nn

from VIAStress.model import FeatureEmbed, BVPEncoder, BVPDecoder, EDAEncoder, EDADecoder, ProjectHead, EncoderTrans


class MemoryEncoder(nn.Module):
    def __init__(self, r_dim, num_memory_slots=16):
        super().__init__()
        self.r_dim = r_dim
        self.num_memory_slots = num_memory_slots

        # Memory bank: [num_slots, r_dim]
        self.memory = nn.Parameter(torch.randn(num_memory_slots, r_dim))

        # Attention projection
        self.q_proj = nn.Linear(r_dim, r_dim)
        self.m_proj = nn.Linear(r_dim, r_dim)

        # Optional MLP for refinement
        self.output_fc = nn.Sequential(
            nn.Linear(r_dim, r_dim),
            nn.LayerNorm(r_dim),
            nn.ReLU()
        )

    def forward(self, q):
        # q: [batch_size, r_dim]
        B = q.size(0)

        # Project q and memory
        q_proj = self.q_proj(q)  # [B, r_dim]
        m_proj = self.m_proj(self.memory)  # [M, r_dim]

        # Compute attention weights
        attn_logits = torch.matmul(q_proj, m_proj.t())  # [B, M]
        attn_weights = torch.softmax(attn_logits, dim=-1)

        # Memory readout: weighted sum
        memory_read = torch.matmul(attn_weights, self.memory)  # [B, r_dim]

        # Gate between q and memory_read
        # gate_input = torch.cat([q, memory_read], dim=-1)  # [B, 2*r_dim]
        out = q + memory_read  # gated fusion

        return self.output_fc(out)


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
            nn.ReLU(),
        )

        self.r_enc_1 = MemoryEncoder(self.r_dim)
        self.film_layer_1_beta = nn.Linear(self.r_dim, self.h_dim, bias=False)
        self.film_layer_1_gamma = nn.Linear(self.r_dim, self.h_dim, bias=False)
        self.film_layer_1_eta = nn.Linear(self.r_dim, self.h_dim, bias=False)
        self.film_layer_1_delta = nn.Linear(self.r_dim, self.h_dim, bias=False)

        self.hidden_layer_2 = nn.Sequential(
            # nn.LayerNorm(self.h_dim),
            nn.ReLU(),
            # nn.Linear(self.h_dim, self.h_dim),
            nn.Linear(self.h_dim, self.y_dim),
            # nn.LayerNorm(self.h_dim),
            # nn.ReLU(),
        )

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

        return hidden_2


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
    parser.add_argument('--x_dim', type=int, default=128)
    parser.add_argument('--y_dim', type=int, default=3)
    parser.add_argument('--h_dim', type=int, default=256)
    parser.add_argument('--r_dim', type=int, default=64)
    parser.add_argument('--context_num', type=int, default=10)
    parser.add_argument('--dataset_name', type=str, default='wesad')

    args = parser.parse_args()

    bvp = torch.rand((16, 1, 1920))
    eda = torch.rand((16, 1, 120))
    y = torch.zeros((16)).long()

    # eda_emb = EDAEmbed(128)
    # print(eda_emb(eda).shape)

    model = Model(args.x_dim, args.r_dim, args.h_dim, args.y_dim)
    out, bvp_z, eda_z, recon_bvp, mu_bvp, logvar_bvp, recon_eda, mu_eda, logvar_eda = model(bvp, eda)
    print(out.shape)
    print(recon_bvp.shape)
    print(recon_eda.shape)

    # print(eda.unsqueeze(0).expand_as(recon_eda))
