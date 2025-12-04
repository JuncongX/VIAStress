import torch
import torch.nn as nn
from torch.utils.data import Dataset
from scipy import signal
import numpy as np
import pandas as pd
import math
import os
import pywt
from utils.filter import butter_bandpass_filter, detrend
# from data.general_setting import CUTOFF_LOW, CUTOFF_HIGH, ORDER, PPG_FPS


def z_score(signal):
    mean = np.mean(signal)
    signal = signal - mean
    std = np.std(signal)

    return signal / std


class WESAD_dataset(Dataset):
    def __init__(self, person_list, binary=True):
        # data = np.load('data/WESAD/WESAD_clip1920.npy', allow_pickle=True)
        data = np.load('WESAD_clip1920.npy', allow_pickle=True)
        self.wavename = 'morl'
        totalscal = 64
        fc = pywt.central_frequency(self.wavename)
        cparam = 2 * fc * totalscal
        self.scales = cparam / np.arange(totalscal, 0, -1)
        self.sampling_rate = 64

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
        self.PPG = [ppg for ppg in needed_data[:, 2]]

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, item):
        [cwtmatr, frequencies] = pywt.cwt(self.PPG[item], self.scales, self.wavename, 1.0 / self.sampling_rate)
        return self.labels[item].to(torch.long), \
               torch.from_numpy(cwtmatr).to(torch.float)


if __name__ == '__main__':
    from torch.utils.data import DataLoader

    all_persons = []
    for s_i in [2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 13, 14, 15, 16, 17]:
        all_persons.append(r"S{0}".format(s_i))
        # all_persons = [r"S{0}".format(s_i)]
    dataset = WESAD_dataset(all_persons)
    loader = DataLoader(dataset=dataset, batch_size=64, num_workers=0, shuffle=True)
    stress_num = 0
    rest_num = 0
    for i, data in enumerate(loader):
        label, cwtmatr = data
        print("======{0}======".format(i))
        print(label)
        print(cwtmatr.shape)
        stress_num += np.sum(np.array(label) == 1)
        rest_num += np.sum(np.array(label) == 0)
        labels_baseline_index = np.arange(len(label))[label == 0]
        labels_baseline_index_choice = np.random.choice(labels_baseline_index, 5, replace=False)

        labels_others_index_choice = [i for i in range(label.shape[0]) if i not in labels_baseline_index_choice]

        cwtmatr_context = cwtmatr[labels_baseline_index_choice]
        label_context = label[labels_baseline_index_choice]

        label = label[labels_others_index_choice]
        cwtmatr = cwtmatr[labels_others_index_choice]
        # print(cwtmatr_context)
        # print(label_context)

    print(stress_num)
    print(rest_num)
