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


# data = np.load('VerBIO_clip30s_multi.npy', allow_pickle=True)
data = np.load('data/VerBIO/VerBIO_clip30s_multi.npy', allow_pickle=True)


class VerBIO_dataset(Dataset):
    def __init__(self, person_list):
        needed_data = None
        for person in person_list:
            for label in [0,1]:
                selected_data = data[(data[:, 0] == person) & (data[:, 1] == label)]
                if needed_data is None:
                    needed_data = selected_data
                else:
                    needed_data = np.concatenate((needed_data, selected_data))

        self.labels = torch.from_numpy(np.array([l for l in needed_data[:, 1]]).astype(np.uint8))
        # self.PPG = torch.Tensor(
        #     [butter_bandpass_filter(z_score(ppg), PPG_FPS, CUTOFF_LOW, CUTOFF_HIGH, ORDER).astype(np.float32) for ppg in
        #      needed_data[:, 2]])

        self.PPG = torch.Tensor([z_score(signal).astype(np.float32) for signal in needed_data[:, 2]])
        self.SCL = torch.Tensor([z_score(signal).astype(np.float32) for signal in needed_data[:, 3]])
        self.SCR = torch.Tensor([z_score(signal).astype(np.float32) for signal in needed_data[:, 4]])
        self.EDA = torch.Tensor([z_score(signal).astype(np.float32) for signal in needed_data[:, 5]])
        self.peak = torch.Tensor([signal.astype(np.float32) for signal in needed_data[:, 6]])

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, item):
        return self.labels[item].to(torch.long), \
               self.PPG[item].unsqueeze(dim=0).to(torch.float), \
               self.SCL[item].unsqueeze(dim=0).to(torch.float), \
               self.SCR[item].unsqueeze(dim=0).to(torch.float), \
               self.EDA[item].unsqueeze(dim=0).to(torch.float), \
               self.peak[item].unsqueeze(dim=0).to(torch.float)


if __name__ == '__main__':
    from torch.utils.data import DataLoader

    # all_persons =  ['s1', 's2', 's3', 's4', 's5', 's6', 's7', 's8', 's9', 's10', 's11', 's13', 's14', 's15',
    #                's16', 's18', 's19', 's20', 's21', 's22', 's23', 's24', 's25', 's26', 's27', 's28', 's29',
    #                's30', 's31', 's32', 's33', 's34', 's35', 's36', 's37', 's38', 's39', 's40', 's41', 's42', 's43',
    #                's44', 's45', 's46', 's48', 's49', 's50', 's51', 's52', 's53', 's54', 's55', 's56']
    all_persons = ['P005']
    dataset = VerBIO_dataset(all_persons, binary=True, task=2)
    loader = DataLoader(dataset=dataset, batch_size=128, num_workers=0)
    stress_num = 0
    rest_num = 0
    for i, data in enumerate(loader):
        label, ppg, scr, scl, eda = data
        eda = torch.cat((scr, scl), dim=1)
        print("======{0}======".format(i))
        print(label)
        print(ppg.shape)
        print(eda.shape)
        stress_num += np.sum(np.array(label) != 0)
        rest_num += np.sum(np.array(label) == 0)
        # break
    print(stress_num)
    print(rest_num)
