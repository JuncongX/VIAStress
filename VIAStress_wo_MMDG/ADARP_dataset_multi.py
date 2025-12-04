import torch
import torch.nn as nn
from torch.utils.data import Dataset
from scipy import signal
import numpy as np
import pandas as pd
import math
import os
from utils.filter import butter_bandpass_filter, detrend


# from data.general_setting import CUTOFF_LOW, CUTOFF_HIGH, ORDER, PPG_FPS


def z_score(signal, eps=1e-8):
    signal = np.array(signal, dtype=np.float32)
    mean = np.mean(signal)
    std = np.std(signal)
    if std < eps:
        std = eps
    return (signal - mean) / std


# data = np.load('UBFC_Phys_clip30s_multi_peak808_.npy', allow_pickle=True)
# data = np.load('/home/som/8T/DataSets/ADARP_clip30s_multi_undersampling.npy', allow_pickle=True)
data = np.load('data/ADARP/ADARP_clip30s_multi_undersampling.npy', allow_pickle=True)


class ADARP_dataset(Dataset):
    def __init__(self, person_list):
        needed_data = None
        w_label = [0, 1]
        for person in person_list:
            for label in w_label:
                selected_data = data[(data[:, 0] == person) & (data[:, 1] == label)]
                if needed_data is None:
                    needed_data = selected_data
                else:
                    needed_data = np.concatenate((needed_data, selected_data))
        self.labels = torch.from_numpy(np.array(needed_data[:, 1]).astype(np.uint8))
        # self.PPG = torch.Tensor(
        #     [butter_bandpass_filter(z_score(ppg), PPG_FPS, CUTOFF_LOW, CUTOFF_HIGH, ORDER).astype(np.float32) for ppg in
        #      needed_data[:, 2]])

        ppg_array = np.array([z_score(np.array(signal, dtype=np.float32)) for signal in needed_data[:, 2]],
                             dtype=np.float32)
        self.PPG = torch.from_numpy(ppg_array)
        scl_array = np.array([z_score(np.array(signal, dtype=np.float32)) for signal in needed_data[:, 3]],
                             dtype=np.float32)
        self.SCL = torch.from_numpy(scl_array)
        scr_array = np.array([z_score(np.array(signal, dtype=np.float32)) for signal in needed_data[:, 4]],
                             dtype=np.float32)
        self.SCR = torch.from_numpy(scr_array)
        eda_array = np.array([z_score(np.array(signal, dtype=np.float32)) for signal in needed_data[:, 5]],
                             dtype=np.float32)
        self.EDA = torch.from_numpy(eda_array)
        peak_array = np.array([np.array(signal, dtype=np.float32) for signal in needed_data[:, 6]],
                             dtype=np.float32)
        self.peak = torch.from_numpy(peak_array)

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
    all_persons = ['Part 101C']
    dataset = ADARP_dataset(all_persons)
    loader = DataLoader(dataset=dataset, batch_size=128, num_workers=0)
    stress_num = 0
    rest_num = 0
    for i, data in enumerate(loader):
        label, ppg, scr, scl, eda, peak = data
        eda = torch.cat((scr, scl), dim=1)
        print("======{0}======".format(i))
        print(sum(peak))
        print(ppg.shape)
        print(eda.shape)
        stress_num += np.sum(np.array(label) != 0)
        rest_num += np.sum(np.array(label) == 0)
        # break
    print(stress_num)
    print(rest_num)
