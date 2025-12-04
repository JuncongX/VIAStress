import torch
import torch.nn as nn
from torch.utils.data import Dataset
from scipy import signal
import numpy as np
import pandas as pd
import math
import os
from utils.filter import butter_bandpass_filter, detrend


def z_score(signal):
    mean = np.mean(signal)
    signal = signal - mean
    std = np.std(signal)

    return signal / std


# data = np.load('WESAD_clip30s_multi_peak808_.npy', allow_pickle=True)
data = np.load('data/WESAD/WESAD_clip30s_multi_peak808_.npy', allow_pickle=True)


class WESAD_dataset(Dataset):
    def __init__(self, person_list, binary=True):
        # data = np.load('data/WESAD/WESAD_clip30s_multi_ae.npy', allow_pickle=True)

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

        self.PPG = torch.Tensor([z_score(signal).astype(np.float32) for signal in needed_data[:, 2]])
        self.SCL = torch.Tensor([z_score(signal).astype(np.float32) for signal in needed_data[:, 3]])
        self.SCR = torch.Tensor([z_score(signal).astype(np.float32) for signal in needed_data[:, 4]])
        self.EDA = torch.Tensor([z_score(signal).astype(np.float32) for signal in needed_data[:, 5]])

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, item):
        return self.labels[item].to(torch.long), \
               self.PPG[item].unsqueeze(dim=0).to(torch.float), \
               self.SCL[item].unsqueeze(dim=0).to(torch.float), \
               self.SCR[item].unsqueeze(dim=0).to(torch.float), \
               self.EDA[item].unsqueeze(dim=0).to(torch.float)


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
        label, ppg, scr, scl, eda = data
        # eda = torch.cat((scr, scl), dim=1)
        print("======{0}======".format(i))
        print(label)
        # print(ppg)
        # print(eda)
        ppg, ppg_info = nk.ppg_process(ppg.squeeze().cpu().numpy()[0], sampling_rate=64)
        nk.ppg_plot(ppg, ppg_info)

        eda, eda_info = nk.eda_process(eda.squeeze().cpu().numpy()[0], sampling_rate=4)
        nk.eda_plot(eda, eda_info)
        plt.show()
        stress_num += np.sum(np.array(label) == 1)
        rest_num += np.sum(np.array(label) == 0)
        # break
    print(stress_num)
    print(rest_num)
