import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.autograd as autograd

import copy
import numpy as np
import itertools
from SelfReg.modal import Featurizer, Classifier
from sklearn.metrics import confusion_matrix


class SelfReg(nn.Module):
    def __init__(self, args, device):
        super(SelfReg, self).__init__()
        x_dim = args.x_dim
        h_dim = args.h_dim
        self.y_dim = args.y_dim
        self.featurizer = Featurizer(x_dim)
        self.classifier = Classifier(2 * x_dim, h_dim, self.y_dim)
        self.optimizer = torch.optim.Adam(
            itertools.chain(self.featurizer.parameters(), self.classifier.parameters()),
            lr=args.lr,
            weight_decay=args.weight_decay
        )
        self.MSEloss = nn.MSELoss()
        input_feat_size = 2 * x_dim

        self.cdpl = nn.Sequential(
            nn.Linear(input_feat_size, 2048),
            nn.BatchNorm1d(2048),
            nn.ReLU(inplace=True),
            nn.Linear(2048, 2048),
            nn.BatchNorm1d(2048),
            nn.ReLU(inplace=True),
            nn.Linear(2048, input_feat_size),
            nn.BatchNorm1d(input_feat_size)
        )

    def update(self, ppg, eda, labels):
        lam = np.random.beta(0.5, 0.5)

        batch_size = labels.size()[0]

        # cluster and order features into same-class group
        with torch.no_grad():
            sorted_y, indices = torch.sort(labels)
            sorted_ppg = torch.zeros_like(ppg)
            sorted_eda = torch.zeros_like(eda)
            for idx, order in enumerate(indices):
                sorted_ppg[idx] = ppg[order]
                sorted_eda[idx] = eda[order]
            intervals = []
            ex = 0
            for idx, val in enumerate(sorted_y):
                if ex == val:
                    continue
                intervals.append(idx)
                ex = val
            intervals.append(batch_size)

            ppg = sorted_ppg
            eda = sorted_eda
            labels = sorted_y

        feat = self.featurizer(ppg, eda)
        proj = self.cdpl(feat)

        output = self.classifier(feat)

        # shuffle
        output_2 = torch.zeros_like(output)
        feat_2 = torch.zeros_like(proj)
        output_3 = torch.zeros_like(output)
        feat_3 = torch.zeros_like(proj)
        ex = 0
        for end in intervals:
            shuffle_indices = torch.randperm(end - ex) + ex
            shuffle_indices2 = torch.randperm(end - ex) + ex
            for idx in range(end - ex):
                output_2[idx + ex] = output[shuffle_indices[idx]]
                feat_2[idx + ex] = proj[shuffle_indices[idx]]
                output_3[idx + ex] = output[shuffle_indices2[idx]]
                feat_3[idx + ex] = proj[shuffle_indices2[idx]]
            ex = end

        # mixup
        output_3 = lam * output_2 + (1 - lam) * output_3
        feat_3 = lam * feat_2 + (1 - lam) * feat_3

        # regularization
        L_ind_logit = self.MSEloss(output, output_2)
        L_hdl_logit = self.MSEloss(output, output_3)
        L_ind_feat = 0.3 * self.MSEloss(feat, feat_2)
        L_hdl_feat = 0.3 * self.MSEloss(feat, feat_3)

        cl_loss = F.cross_entropy(output, labels)
        C_scale = min(cl_loss.item(), 1.)
        loss = cl_loss + C_scale * (lam * (L_ind_logit + L_ind_feat) + (1 - lam) * (L_hdl_logit + L_hdl_feat))

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        preds = output.argmax(dim=1)

        cm = confusion_matrix(labels.cpu().numpy(), preds.cpu().numpy(), labels=[i for i in range(self.y_dim)])

        return loss.item(), cm

    def predict(self, ppg, eda):
        z = self.featurizer(ppg, eda)
        return self.classifier(z)