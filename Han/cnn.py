import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import OrderedDict
from Han.autoencoder import model_conv1d_autoencoder


class model_conv1d(nn.Module):
    def __init__(self, configs, num_classes=2, load_pretrained=True):
        super(model_conv1d, self).__init__()

        self.ppg_encoder = model_conv1d_autoencoder(1, modality="ppg").encoder
        self.eda_encoder = model_conv1d_autoencoder(1, modality="eda").encoder

        if load_pretrained:
            state_dict = torch.load(configs["save_model_path_ppg"])
            model_state = self.ppg_encoder.state_dict()
            pretrained_dict = {k[8:]: v for k, v in state_dict.items() if k[8:] in model_state and "encoder" in k}
            print(pretrained_dict.keys())
            model_state.update(pretrained_dict)
            self.ppg_encoder.load_state_dict(model_state)

            state_dict = torch.load(configs["save_model_path_eda"])
            model_state = self.eda_encoder.state_dict()
            pretrained_dict = {k[8:]: v for k, v in state_dict.items() if k[8:] in model_state and "encoder" in k}
            print(pretrained_dict.keys())
            model_state.update(pretrained_dict)
            self.eda_encoder.load_state_dict(model_state)

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

