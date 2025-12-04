import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F
from torch.autograd import Variable
from typing import Optional, List
import math
from torch.distributions import Normal
from BCSA.modules import SelfAttentionEncoder, CrossAttentionEncoder, PositionalEncoding


# BVP CNN Net Reference: CorNET: Deep learning framework for PPG-based heart rate estimation and biometric identification in ambulant environment
class BVPEmbed(nn.Module):
    def __init__(self):
        super(BVPEmbed, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(1, 32, 32, 1, 1),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.MaxPool1d(4, 4),
            nn.Conv1d(32, 32, 32, 1, 1),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.MaxPool1d(4, 4)
        )

        self.to_L = nn.Linear(320, 128)

    def forward(self, ppg):
        bs, _, _ = ppg.shape
        slices = []
        for slice in [ppg[:, :, i * 320:(i + 1) * 320] for i in range(6)]:
            ppg_x = self.conv(slice)
            ppg_x = ppg_x.view(bs, -1)
            slices.append(ppg_x)
        f = torch.stack(slices, dim=1)
        f = self.to_L(f)

        return f


class EDAEmbed(nn.Module):
    def __init__(self):
        super(EDAEmbed, self).__init__()
        self.lstm = nn.LSTM(20, 72, 2, batch_first=True, bidirectional=False)
        self.to_L = nn.Linear(72, 128)

    def forward(self, eda):
        self.lstm.flatten_parameters()
        bs, _, _ = eda.shape

        eda = eda.transpose(1, 2)
        slices = []
        for slice in [eda[:, i * 20:(i + 1) * 20, :] for i in range(6)]:
            slice = slice.contiguous().view(bs, -1)
            slices.append(slice)
        slices = torch.stack(slices, dim=1)
        out, (h_1, c_1) = self.lstm(slices)
        out = self.to_L(out)
        return out


class BCSA(nn.Module):
    def __init__(self, heads, d_model):
        super(BCSA, self).__init__()
        # self.cross_atten = nn.MultiheadAttention(num_heads=heads, embed_dim=d_model)
        # self.self_atten = nn.MultiheadAttention(num_heads=heads, embed_dim=d_model)
        self.cross_atten = CrossAttentionEncoder(d_model, d_model, heads, 0.1)
        self.self_atten = SelfAttentionEncoder(d_model, d_model, heads, 0.1)

    def forward(self, x1, x2):
        x_c = self.cross_atten(x1, x2)
        x_s = self.self_atten(x_c)
        return x_s


class FeatureEmbed(nn.Module):
    def __init__(self, x_dim, heads, n_bcsa):
        super(FeatureEmbed, self).__init__()
        self.n_bcsa = n_bcsa

        self.bvp_embedding = BVPEmbed()
        self.eda_embedding = EDAEmbed()

        self.pos_enc_bvp = PositionalEncoding(128)
        self.pos_enc_eda = PositionalEncoding(128)

        self.bcsa_bvp = nn.ModuleList([BCSA(heads, x_dim) for _ in range(n_bcsa)])
        self.bcsa_eda = nn.ModuleList([BCSA(heads, x_dim) for _ in range(n_bcsa)])

    def forward(self, bvp, eda):
        bvp_f = self.bvp_embedding(bvp)
        bvp_f = self.pos_enc_bvp(bvp_f)
        eda_f = self.eda_embedding(eda)
        eda_f = self.pos_enc_eda(eda_f)
        for i in range(self.n_bcsa):
            bvp_f = self.bcsa_bvp[i](bvp_f, eda_f)
            eda_f = self.bcsa_eda[i](eda_f, bvp_f)

        return bvp_f, eda_f


class Classifier(nn.Module):
    def __init__(self, y_dim):
        super(Classifier, self).__init__()
        self.bvp_c = nn.Sequential(
            nn.Linear(128, 256),
            nn.Dropout(0.1),
            nn.Linear(256, y_dim)
        )

        self.eda_c = nn.Sequential(
            nn.Linear(128, 256),
            nn.Dropout(0.1),
            nn.Linear(256, y_dim)
        )

    def forward(self, bvp_f, eda_f):
        y_bvp = self.bvp_c(F.adaptive_avg_pool1d(bvp_f.transpose(1, 2), 1).view(-1, 128))
        y_eda = self.eda_c(F.adaptive_avg_pool1d(eda_f.transpose(1, 2), 1).view(-1, 128))
        return y_bvp + y_eda, y_bvp, y_eda


class DAFMPPSR(nn.Module):
    def __init__(self, x_dim, heads, n_bcsa, y_dim):
        super(DAFMPPSR, self).__init__()
        self.f_e = FeatureEmbed(x_dim, heads, n_bcsa)
        self.c = Classifier(y_dim)

    def forward(self, bvp, eda):
        bvp_f, eda_f = self.f_e(bvp, eda)
        y, y_bvp, y_eda = self.c(bvp_f, eda_f)
        return y, y_bvp, y_eda


class Loss(nn.Module):
    def __init__(self):
        super(Loss, self).__init__()
        self.CE_bvp = nn.CrossEntropyLoss()
        self.CE_eda = nn.CrossEntropyLoss()
        self.MSE = nn.MSELoss()

    def forward(self, y_bvp, y_eda, y_true):
        return self.CE_bvp(y_bvp, y_true) + self.CE_eda(y_eda, y_true) + self.MSE(y_bvp, y_eda)


if __name__ == '__main__':
    bvp_target = torch.rand((32, 1, 1920))
    eda_target = torch.rand((32, 1, 120))
    y_target = torch.rand((32)).to(torch.long)

    model = DAFMPPSR(x_dim=128, heads=4, n_bcsa=2, y_dim=3)
    y, y_bvp, y_eda = model(bvp_target, eda_target)
    print(y_target.shape, y_bvp.shape, y_eda.shape)
    loss = Loss()
    print(loss(y_bvp, y_eda, y_target))
