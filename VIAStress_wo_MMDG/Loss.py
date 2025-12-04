import torch
import torch.nn as nn


class Loss(nn.Module):
    def __init__(self):
        super(Loss, self).__init__()
        self.ce_loss = nn.CrossEntropyLoss()
        self.bvp_L1 = nn.L1Loss(reduction="mean")
        self.eda_Ll = nn.L1Loss(reduction="mean")

    def forward(self,
                pred_logits, labels,
                recon_bvp, ppg,
                recon_eda, eda,
                mu_bvp, logvar_bvp,
                mu_eda, logvar_eda
                ):
        ce_loss = self.ce_loss(pred_logits, labels)
        bvp_l1 = self.bvp_L1(recon_bvp, ppg)
        eda_l1 = self.eda_Ll(recon_eda, eda)
        bvp_kl = -0.5 * (logvar_bvp + 1 - mu_bvp ** 2 - torch.exp(logvar_bvp))
        eda_kl = -0.5 * (logvar_eda + 1 - mu_eda ** 2 - torch.exp(logvar_eda))
        bvp_kl = bvp_kl.mean()
        eda_kl = eda_kl.mean()
        return ce_loss + bvp_l1 + bvp_kl + eda_l1 + eda_kl
