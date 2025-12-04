import torch
import torch.nn as nn
from collections import OrderedDict


class model_conv1d_autoencoder(nn.Module):
    def __init__(self, n_features, embedding_dim=64, modality="eda"):
        super(model_conv1d_autoencoder, self).__init__()
        self.n_features, self.embedding_dim = n_features, embedding_dim

        self.avgpool = nn.AdaptiveAvgPool1d(1)  # size to be setted
        self.upsample = nn.Upsample(scale_factor=2)  # size to be setted.

        kernel_size = 5
        if modality == "eda":
            channel_list = [8, 16, 32]
            norm = nn.BatchNorm1d
            scale = 2
            output_padding = 1

        elif modality == "ppg":
            channel_list = [8, 16, 32]
            norm = nn.BatchNorm1d
            scale = 4
            output_padding = 3
            # conv1d require the input size: [N,  n_features, seq_len]


        self.encoder = nn.Sequential(OrderedDict([
            ('conv1', nn.Conv1d(self.n_features, channel_list[0], kernel_size, padding=kernel_size // 2)),
            ('bn1', norm(channel_list[0])),
            ('relu1', nn.ReLU()),
            ('pool1', nn.MaxPool1d(scale)),

            ('conv2', nn.Conv1d(channel_list[0], channel_list[1], kernel_size, padding=kernel_size // 2)),
            ('bn2', norm(channel_list[1])),
            ('relu2', nn.ReLU()),
            ('pool2', nn.MaxPool1d(scale)),

            ('conv3', nn.Conv1d(channel_list[1], channel_list[2], kernel_size, padding=kernel_size // 2)),
            ('bn3', norm(channel_list[2])),
            ('relu3', nn.ReLU()),
            ('pool3', nn.MaxPool1d(scale)),
        ]))

        self.decoder = nn.Sequential(OrderedDict([
            ('deconv1',
             nn.ConvTranspose1d(channel_list[2], channel_list[1], kernel_size, stride=scale, padding=kernel_size // 2,
                                output_padding=output_padding)),
            ('bn6', norm(channel_list[1])),
            ('relu6', nn.ReLU()),
            # ('pool5', nn.Upsample(scale_factor=4)),

            ('deconv2',
             nn.ConvTranspose1d(channel_list[1], channel_list[0], kernel_size, stride=scale, padding=kernel_size // 2,
                                output_padding=output_padding)),
            ('bn7', norm(channel_list[0])),
            ('relu7', nn.ReLU()),
            # ('pool5', nn.Upsample(scale_factor=4)),

            ('deconv3',
             nn.ConvTranspose1d(channel_list[0], self.n_features, kernel_size, stride=scale, padding=kernel_size // 2,
                                output_padding=output_padding)),
        ]))

    def forward(self, x):
        # x: [B, n_features, seq_len]
        # x = torch.transpose(x, 1, 2)   # [B, seg_len, n_features] => [B, n_features, seq_len]
        x = self.encoder(x)  # [B, self.embedding_dim]
        # print(x.shape)
        feats = self.avgpool(x)
        # print(feats.shape)
        # x = self.upsample(x)
        # print(x.shape)

        x = self.decoder(x)
        # print(x.shape)
        # x = torch.transpose(x, 1, 2)
        return x, feats


if __name__ == '__main__':
    ppg = torch.randn((8, 1, 1920))
    eda = torch.randn((8, 1, 120))

    ppg_model_conv1d_autoencoder = model_conv1d_autoencoder(1, modality='ppg')
    eda_model_conv1d_autoencoder = model_conv1d_autoencoder(1, modality='eda')

    ppg_x, ppg_feats = ppg_model_conv1d_autoencoder(ppg)
    eda_x, eda_feats = eda_model_conv1d_autoencoder(eda)

    # print(ppg_x.shape, eda_x.shpe)
    print(ppg_x.shape, ppg_feats.shape)
    print(eda_x.shape, eda_feats.shape)