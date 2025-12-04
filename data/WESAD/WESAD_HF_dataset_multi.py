import torch
import torch.nn as nn
from torch.utils.data import Dataset
from scipy import signal
import numpy as np
import pandas as pd
import math
import os
from utils.filter import butter_bandpass_filter, detrend
from data.general_setting import CUTOFF_LOW, CUTOFF_HIGH, ORDER, PPG_FPS
from sklearn.preprocessing import StandardScaler


def z_score(signal):
    mean = np.mean(signal)
    signal = signal - mean
    std = np.std(signal)

    return signal / std

data = np.load('WESAD_HF_clip30s.npy', allow_pickle=True)
# data = np.load('data/WESAD/WESAD_HF_clip30s.npy', allow_pickle=True)

class WESAD_dataset(Dataset):
    def __init__(self, person_list, binary=True):
        # data = np.load('data/WESAD/WESAD_clip30s_multi_ae.npy', allow_pickle=True)
        self.scaler = StandardScaler()
        needed_data = None
        for person in person_list:
            for label in [0, 1, 2]:
                selected_data = data[(data[:, 0] == person) & (data[:, 1] == label)]
                if needed_data is None:
                    needed_data = selected_data
                else:
                    needed_data = np.concatenate((needed_data, selected_data))
        if binary:
            self.labels = torch.from_numpy(np.array([(l if l == 1 else 0) for l in needed_data[:, 1]]).astype(np.uint8))
        else:
            self.labels = torch.from_numpy(np.array([l for l in needed_data[:, 1]]).astype(np.uint8))
        # self.PPG = torch.Tensor(
        #     [butter_bandpass_filter(z_score(ppg), PPG_FPS, CUTOFF_LOW, CUTOFF_HIGH, ORDER).astype(np.float32) for ppg in
        #      needed_data[:, 2]])

        standardized_features = []
        for feature in needed_data[:, 2]:  # Assuming needed_data[:, 2] gives you a list of features
            # Reshape to (n_samples, 1) for scaling
            feature = np.array(feature).reshape(-1, 1).astype(np.float32)
            scaled_feature = self.scaler.fit_transform(feature)
            standardized_features.append(scaled_feature.flatten())  # Flatten back to 1D if needed

        # Convert to PyTorch tensor
        self.features = torch.from_numpy(np.array(standardized_features))

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, item):
        return self.labels[item].to(torch.long), \
               self.features[item].to(torch.float)


if __name__ == '__main__':
    from torch.utils.data import DataLoader
    import neurokit2 as nk
    import matplotlib.pyplot as plt

    all_persons = []
    for s_i in [2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 13, 14, 15, 16, 17]:
        all_persons.append(r"S{0}".format(s_i))
    dataset = WESAD_dataset(all_persons)
    loader = DataLoader(dataset=dataset, batch_size=16, num_workers=0)
    stress_num = 0
    rest_num = 0
    for i, data in enumerate(loader):
        label, features = data
        # eda = torch.cat((scr, scl), dim=1)
        print("======{0}======".format(i))
        print(label)
        print(features)
        stress_num += np.sum(np.array(label) == 1)
        rest_num += np.sum(np.array(label) == 0)
        # break
    print(stress_num)
    print(rest_num)
