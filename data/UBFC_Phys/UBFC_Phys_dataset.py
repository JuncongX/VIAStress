import os
import torch
import numpy as np
import pandas as pd
import math
from scipy import signal
from torch.utils.data import Dataset
from utils.filter import butter_bandpass_filter
from data.general_setting import CUTOFF_LOW, CUTOFF_HIGH, ORDER, PPG_FPS

T1_s = ['s1', 's2', 's4', 's5', 's6', 's7', 's10', 's11', 's12', 's13', 's14', 's15', 's16', 's17', 's18', 's19', 's20', 's21',
        's22', 's23', 's24', 's25', 's27', 's29', 's34', 's35', 's36', 's37', 's38', 's39', 's41', 's42', 's43', 's44',
        's45', 's46', 's47', 's48', 's49', 's50', 's51', 's55']
T2_s = ['s2', 's3', 's5', 's7', 's10', 's15', 's16', 's17', 's18', 's20', 's23', 's24', 's29', 's30', 's34', 's36', 's37',
        's40', 's43', 's44', 's46', 's50', 's51', 's54', 's56']
T3_s = ['s1', 's2', 's3', 's4', 's6', 's7', 's11', 's12', 's15', 's16', 's18', 's19', 's20', 's21', 's23', 's24', 's27',
        's29', 's31', 's34', 's36', 's38', 's39', 's41', 's42', 's43', 's44', 's45', 's46', 's51', 's54', 's55', 's56']
T1_selected, T2_selected, T3_selected = set(T1_s), set(T2_s), set(T3_s)


def data_selected():
    stress_person = T2_selected | T3_selected  # 并集
    rest_person = T1_selected
    both_person = stress_person & rest_person  # 交集
    stress_person_T2 = both_person - (T2_selected & T3_selected)
    stress_person_T3 = both_person - stress_person_T2

    person_list, tasks = [], []
    for t2_p in stress_person_T2:
        person_list.append(t2_p)
        tasks.append(2)
    for t3_p in stress_person_T3:
        person_list.append(t3_p)
        tasks.append(3)

    return person_list, tasks


def z_score(signal):
    mean = np.mean(signal)
    signal = signal - mean
    std = np.std(signal)

    return signal / std


class rPPG_Dataset(Dataset):
    def __init__(self, person_list, task_s):
        super(rPPG_Dataset, self).__init__()

        data = np.load(r"ubfc_phys_clip1920.npy", allow_pickle=True)
        # data = np.load(r"data/UBFC_Phys/ubfc_phys_clip1920.npy", allow_pickle=True)
        needed_data = None
        for _, (person, task) in enumerate(zip(person_list, task_s)):
            for label in [0, 1]:
                task_trans = task if label == 1 else 1
                selected_data = data[(data[:, 0] == person) & (data[:, 1] == task_trans) & (data[:, 3] == label)]
                if needed_data is None:
                    needed_data = selected_data
                else:
                    needed_data = np.concatenate((needed_data, selected_data))
        self.labels = torch.from_numpy(needed_data[:, 3].astype(np.uint8))
        self.tasks = torch.from_numpy(needed_data[:, 1].astype(np.uint8))
        self.level = torch.from_numpy(needed_data[:, 2].astype(np.uint8))
        self.rPPG = torch.Tensor(
            [butter_bandpass_filter(z_score(rppg), PPG_FPS, CUTOFF_LOW, CUTOFF_HIGH, ORDER).astype(np.float32) for rppg
             in needed_data[:, 4]])
        self.BVP = torch.Tensor(
            [butter_bandpass_filter(z_score(bvp), PPG_FPS, CUTOFF_LOW, CUTOFF_HIGH, ORDER).astype(np.float32) for bvp in
             needed_data[:, 5]])

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, item):
        return self.labels[item].to(torch.long), \
               self.tasks[item].to(torch.long), \
               self.level[item].to(torch.long), \
               self.rPPG[item].unsqueeze(dim=0).to(torch.float), \
               self.BVP[item].unsqueeze(dim=0).to(torch.float)


if __name__ == '__main__':
    import yaml
    from torch.utils.data import DataLoader
    import matplotlib.pyplot as plt

    from sklearn.model_selection import KFold, train_test_split

    person_list, tasks_list = data_selected()
    person_list, tasks_list = np.array(person_list), np.array(tasks_list)
    # print(person_list)
    seed = 123
    splits = KFold(n_splits=10, shuffle=True, random_state=seed)
    for fold, (train_idx, val_idx) in enumerate(splits.split(np.arange(len(person_list)))):
        train_p, train_t = person_list[train_idx], tasks_list[train_idx]
        val_p, val_t = person_list[val_idx], tasks_list[val_idx]
        valid_dataset = rPPG_Dataset(val_p, val_t)
        valid_loader = DataLoader(dataset=valid_dataset, batch_size=16, num_workers=0, shuffle=False)
        print(len(valid_loader.sampler))
        print(len(valid_dataset))
        print("=================XXX==================")
        train_l = [1 for i in range(len(train_p))]
        train_p_t, test_p_t, _, _ = train_test_split(np.vstack((train_p, train_t)).transpose(1, 0), train_l,
                                                     test_size=1 / (10 - 1), random_state=seed)
        train_p, train_t = train_p_t[:, 0], [int(t) for t in train_p_t[:, 1]]
        test_p, test_t = test_p_t[:, 0], [int(t) for t in test_p_t[:, 1]]

        print(train_p)
        print(test_p)
        print(val_p)
        for i, data in enumerate(valid_loader):
            labels, tasks, level, rppg, bvp = data
            print(labels)
        break
        # train_p_t, test_p_t, train_l, test_l = train_test_split(np.vstack((train_p, train_t)).transpose(1, 0), train_l,test_size=1 / (10 - 1), random_state=123)
        # train_p, train_t = train_p_t[:, 0], [int(t) for t in train_p_t[:, 1]]
        # test_p, test_t = test_p_t[:, 0], [int(t) for t in test_p_t[:, 1]]
