import neurokit2 as nk
from scipy import signal
from scipy import interpolate
from scipy.signal import savgol_filter
import pandas as pd
import numpy as np
import math
import os
import yaml
import csv
import matplotlib.pyplot as plt
import pickle

# data = np.load('data/WESAD/WESAD_clip30s_multi_ae.npy', allow_pickle=True)
# data = np.load('WESAD_clip30s_multi_ae.npy', allow_pickle=True)
all_persons = []
for s_i in [2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 13, 14, 15, 16, 17]:
    all_persons.append(r"S{0}".format(s_i))

# for person in all_persons:
#     for label in [0, 1, 2]:
#         pl_quality = []
#         selected_data = data[(data[:, 0] == person) & (data[:, 1] == label)]
#         for ppg in selected_data[:, 2]:
#             ppg_cleaned = nk.ppg_clean(ppg, sampling_rate=64)
#             quality = nk.ppg_quality(ppg_cleaned, sampling_rate=64, method="templatematch")
#             pl_quality.append(quality)
#         print(f"Person:{person} Label:{label} Quality:{np.average(pl_quality)}")
ecg_sample_rate = 700
ppg_sample_rate = 64


def quality(ppg, person, label, quality_all, quality_label):
    ppg_cleaned = nk.ppg_clean(ppg, sampling_rate=64)
    quality = nk.ppg_quality(ppg_cleaned, sampling_rate=64, method="disimilarity")
    # quality = nk.ppg_quality(ppg_cleaned, sampling_rate=64, method="templatematch")
    # nk.signal_plot([ppg_cleaned, quality], standardize=True)
    # plt.show()
    # plt.close()
    print(f"Person:{person} Label:{label} Quality:{np.mean(quality)}")
    quality_all.append(np.mean(quality))
    quality_label[label].append(np.mean(quality))

quality_all = []
quality_label = [[], [], []]
for person in all_persons:
    s_path = r"E:\dataset\WESAD\{0}\{0}.pkl".format(person)
    with open(s_path, 'rb') as file:
        s_data = pickle.load(file, encoding='latin1')
    w_bvp = s_data['signal']['wrist']['BVP'][:, 0]  # 64Hz
    w_label = s_data['label']

    total_num_ecg = len(w_label)
    baseline_i_ecg = w_label == 1
    stress_i_ecg = w_label == 2
    amusement_i_ecg = w_label == 3

    baseline_index_ecg = []
    stress_index_ecg = []
    amusement_index_ecg = []

    baseline_bool_ecg = False
    stress_bool_ecg = False
    amusement_bool_ecg = False

    for i in range(total_num_ecg):
        if baseline_i_ecg[i] == 1 and not baseline_bool_ecg:
            baseline_bool_ecg = True
            baseline_index_ecg.append(i)
        elif baseline_i_ecg[i] == 0 and baseline_bool_ecg:
            baseline_bool_ecg = False
            baseline_index_ecg.append(i)

        if stress_i_ecg[i] == 1 and not stress_bool_ecg:
            stress_bool_ecg = True
            stress_index_ecg.append(i)
        elif stress_i_ecg[i] == 0 and stress_bool_ecg:
            stress_bool_ecg = False
            stress_index_ecg.append(i)

        if amusement_i_ecg[i] == 1 and not amusement_bool_ecg:
            amusement_bool_ecg = True
            amusement_index_ecg.append(i)
        elif amusement_i_ecg[i] == 0 and amusement_bool_ecg:
            amusement_bool_ecg = False
            amusement_index_ecg.append(i)

    baseline_index_bvp = [int(i / ecg_sample_rate * ppg_sample_rate) for i in baseline_index_ecg]
    stress_index_bvp = [int(i / ecg_sample_rate * ppg_sample_rate) for i in stress_index_ecg]
    amusement_index_bvp = [int(i / ecg_sample_rate * ppg_sample_rate) for i in amusement_index_ecg]

    bvp_baseline = w_bvp[baseline_index_bvp[0]: baseline_index_bvp[1]]
    bvp_stress = w_bvp[stress_index_bvp[0]: stress_index_bvp[1]]
    bvp_amusement = w_bvp[amusement_index_bvp[0]: amusement_index_bvp[1]]

    quality(bvp_baseline, person, 0, quality_all, quality_label)
    quality(bvp_stress, person, 1, quality_all, quality_label)
    quality(bvp_amusement, person, 2, quality_all, quality_label)
print(f"Mean Quality:{np.mean(quality_all)}")
print(f"Label 0 Quality:{np.mean(quality_label[0])}")
print(f"Label 1 Quality:{np.mean(quality_label[1])}")
print(f"Label 2 Quality:{np.mean(quality_label[2])}")
# Mean Quality:0.7352799216351386
# Label 0 Quality:0.7375230291768116
# Label 1 Quality:0.7048472167603718
# Label 2 Quality:0.7634695189682327