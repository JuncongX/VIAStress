import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import OrderedDict
from Han.autoencoder import model_conv1d_autoencoder


class model_conv1d(nn.Module):
    def __init__(self, num_classes=2):
        super(model_conv1d, self).__init__()

        self.ppg_encoder = model_conv1d_autoencoder(1, modality="ppg").encoder
        self.eda_encoder = model_conv1d_autoencoder(1, modality="eda").encoder

        self.avgpool_ppg = nn.AdaptiveAvgPool1d(1)
        self.avgpool_eda = nn.AdaptiveAvgPool1d(1)

        self.fc1 = nn.Linear(64, 512)
        self.fc2 = nn.Linear(512, 256)
        self.out = nn.Linear(256, num_classes)

    def forward(self, ppg, eda):
        ppg_embedding = self.ppg_encoder(ppg)
        ppg_embedding = self.avgpool_ppg(ppg_embedding)
        eda_embedding = self.eda_encoder(eda)
        eda_embedding = self.avgpool_eda(eda_embedding)
        embedding = torch.cat([ppg_embedding.squeeze(-1), eda_embedding.squeeze(-1)], dim=1)
        embedding = F.dropout(embedding, 0.3)
        output = F.relu(self.fc1(embedding))
        output = F.dropout(output, 0.3)
        output = F.relu(self.fc2(output))
        output = self.out(output)
        return output

if __name__ == '__main__':
    bvp_target = torch.rand((1, 1, 1920))
    eda_target = torch.rand((1, 1, 120))

    model = model_conv1d()

    example_inputs = (bvp_target, eda_target)
    traced_model = torch.jit.trace(model, example_inputs)
    traced_model.save("han_mobile_traced.pt")