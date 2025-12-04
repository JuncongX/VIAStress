import torch
import torch.nn as nn
import torch.nn.functional as F


# class EDA_loss(nn.Module):
#     def __init__(self, total_steps=50, kl_max=1.0):
#         super(EDA_loss, self).__init__()
#         self.l2 = nn.MSELoss(reduction="mean")
#         self.total_steps = total_steps
#         self.kl_max = kl_max
#         self.current_step = 0
#
#     def forward(self,
#                 recon_signal, signal,
#                 mu_signal, logvar_signal
#                 ):
#         # l1 = self.L1(recon_signal, signal)
#         l2 = self.l2(recon_signal, signal)
#         # kl = -0.5 * (logvar_signal + 1 - mu_signal ** 2 - torch.exp(logvar_signal))
#         beta = min(self.kl_max, self.current_step / self.total_steps)
#         kl = torch.mean(-0.5 * torch.sum(1 + logvar_signal - mu_signal ** 2 - logvar_signal.exp(), dim=1), dim=0)
#         # kl = kl.mean()
#         return l2 + beta * kl
#
#
# class PPG_loss(nn.Module):
#     def __init__(self, total_steps=50, kl_max=1.0):
#         super(PPG_loss, self).__init__()
#         self.total_steps = total_steps
#         self.kl_max = kl_max
#         self.current_step = 0
#
#     def forward(self,
#                 recon_signal, signal,
#                 mu_signal, logvar_signal
#                 ):
#         # l1 = self.L1(recon_signal, signal)
#         ce = F.binary_cross_entropy(recon_signal, signal, reduction="mean")
#         # kl = -0.5 * (logvar_signal + 1 - mu_signal ** 2 - torch.exp(logvar_signal))
#         beta = min(self.kl_max, self.current_step / self.total_steps)
#         kl = torch.mean(-0.5 * torch.sum(1 + logvar_signal - mu_signal ** 2 - logvar_signal.exp(), dim=1), dim=0)
#         # kl = kl.mean()
#         return ce + beta * kl
#
#
# class Loss(nn.Module):
#     def __init__(self):
#         super(Loss, self).__init__()
#         self.ce_loss = nn.CrossEntropyLoss(reduction="mean")
#         self.ppg_loss = PPG_loss()
#         self.eda_loss = EDA_loss()
#
#     def forward(self,
#                 pred_logits, labels,
#                 recon_bvp, ppg,
#                 recon_eda, eda,
#                 mu_bvp, logvar_bvp,
#                 mu_eda, logvar_eda
#                 ):
#         ce_loss = self.ce_loss(pred_logits, labels)
#
#         ppg_l = self.ppg_loss(recon_bvp, ppg, mu_bvp, logvar_bvp)
#         eda_l = self.eda_loss(recon_eda, eda, mu_eda, logvar_eda)
#
#         return ce_loss + ppg_l + eda_l

# class Loss(nn.Module):
#     def __init__(self):
#         super(Loss, self).__init__()
#         self.ce_loss = nn.CrossEntropyLoss(reduction="mean")
#         self.eda_L1 = nn.L1Loss(reduction="mean")
#
#     def forward(self,
#                 pred_logits, labels,
#                 recon_bvp, peak,
#                 recon_eda, eda,
#                 mu_bvp, logvar_bvp,
#                 mu_eda, logvar_eda
#                 ):
#         ce_loss = self.ce_loss(pred_logits, labels)
#         bvp_peak_ce = F.binary_cross_entropy(recon_bvp, peak, reduction="mean")
#         eda_l1 = self.eda_L1(recon_eda, eda)
#         # torch.mean(-0.5 * torch.sum(1 + log_var - mu ** 2 - log_var.exp(), dim=1), dim=0)
#         # bvp_kl = -0.5 * (logvar_bvp + 1 - mu_bvp ** 2 - torch.exp(logvar_bvp))
#         bvp_kl = torch.mean(-0.5 * torch.sum(1 + logvar_bvp - mu_bvp ** 2 - logvar_bvp.exp(), dim=1), dim=0)
#         # eda_kl = -0.5 * (logvar_eda + 1 - mu_eda ** 2 - torch.exp(logvar_eda))
#         eda_kl = torch.mean(-0.5 * torch.sum(1 + logvar_eda - mu_eda ** 2 - logvar_eda.exp(), dim=1), dim=0)
#         # bvp_kl = bvp_kl.mean()
#         # eda_kl = eda_kl.mean()
#         return ce_loss + bvp_peak_ce + bvp_kl + eda_l1 + eda_kl

class Loss(nn.Module):
    def __init__(self, args):
        super(Loss, self).__init__()
        self.ce_loss = nn.CrossEntropyLoss(reduction="mean")
        self.eda_L1 = nn.MSELoss(reduction="none")
        self.dataset_name = args.dataset_name

    def forward(self,
                pred_logits, labels,
                recon_bvp, peak,
                recon_eda, eda,
                mu_bvp, logvar_bvp,
                mu_eda, logvar_eda):
        # Cross-entropy loss
        ce_loss = self.ce_loss(pred_logits, labels)

        # BVP Binary Cross Entropy Loss
        # recon_bvp: (5, B, 1, 1920), peak: (B, 1, 1920)
        peak_expanded = peak.unsqueeze(0).expand_as(recon_bvp)  # (5, B, 1, 1920)
        bvp_loss = F.binary_cross_entropy(recon_bvp, peak_expanded, reduction="none")
        bvp_peak_ce = bvp_loss.mean()  # average over all dims

        # EDA L1 Loss
        # recon_eda: (5, B, 1, 120), eda: (B, 1, 120)
        eda_expanded = eda.unsqueeze(0).expand_as(recon_eda)  # (5, B, 1, 120)
        eda_loss = self.eda_L1(recon_eda, eda_expanded)
        eda_l1 = eda_loss.mean()

        # KL divergence for BVP and EDA
        bvp_kl = torch.mean(-0.5 * torch.sum(1 + logvar_bvp - mu_bvp**2 - logvar_bvp.exp(), dim=1))
        eda_kl = torch.mean(-0.5 * torch.sum(1 + logvar_eda - mu_eda**2 - logvar_eda.exp(), dim=1))

        # Total Loss
        # if self.dataset_name == "ubfc_phys":
        #     total_loss = ce_loss + bvp_peak_ce + bvp_kl + eda_l1 + 0.1 * eda_kl
        # elif self.dataset_name == "wesad":
        #     total_loss = ce_loss + bvp_peak_ce + bvp_kl + eda_l1 + eda_kl
        total_loss = ce_loss + bvp_peak_ce + bvp_kl + eda_l1 + eda_kl
        return total_loss

# class Loss(nn.Module):
#     def __init__(self):
#         super(Loss, self).__init__()
#         self.ce_loss = nn.CrossEntropyLoss()
#         self.bvp_L2 = nn.MSELoss(reduction="mean")
#         self.eda_L2 = nn.MSELoss(reduction="mean")
#
#     def forward(self,
#                 pred_logits, labels,
#                 recon_bvp, ppg,
#                 recon_eda, eda,
#                 mu_bvp, logvar_bvp,
#                 mu_eda, logvar_eda
#                 ):
#         ce_loss = self.ce_loss(pred_logits, labels)
#         bvp_l2 = self.bvp_L2(recon_bvp, ppg)
#         eda_l2 = self.eda_L2(recon_eda, eda)
#         bvp_kl = -0.5 * (logvar_bvp + 1 - mu_bvp ** 2 - torch.exp(logvar_bvp))
#         eda_kl = -0.5 * (logvar_eda + 1 - mu_eda ** 2 - torch.exp(logvar_eda))
#         bvp_kl = bvp_kl.mean()
#         eda_kl = eda_kl.mean()
#         return ce_loss + bvp_l2 + bvp_kl + eda_l2 + eda_kl