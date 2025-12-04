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

# data = np.load('data/UBFC_Phys/UBFC_Phys_clip30s_multi.npy', allow_pickle=True)
# data = np.load('UBFC_Phys_clip30s_multi.npy', allow_pickle=True)
all_persons = ['s1', 's2', 's3', 's4', 's5', 's6', 's7', 's8', 's9', 's10', 's11', 's13', 's14', 's15',
               's16', 's18', 's19', 's20', 's21', 's22', 's23', 's24', 's25', 's26', 's27', 's28', 's29',
               's30', 's31', 's32', 's33', 's34', 's35', 's36', 's37', 's38', 's39', 's40', 's41', 's42', 's43',
               's44', 's45', 's46', 's48', 's49', 's50', 's51', 's52', 's53', 's54', 's55', 's56']

# for person in all_persons:
#     for label in [1, 2, 3]:
#         pl_quality = []
#         selected_data = data[(data[:, 0] == person) & (data[:, 1] == label)]
#         for ppg in selected_data[:, 2]:
#             ppg_cleaned = nk.ppg_clean(ppg, sampling_rate=64)
#             quality = nk.ppg_quality(ppg_cleaned, sampling_rate=64, method="templatematch")
#             pl_quality.append(quality)
#         print(f"Person:{person} Label:{label} Quality:{np.average(pl_quality)}")

data_root = r"/home/som/8T/DataSets/ubfc_phys/video/"
quality_all = []
quality_label = [[], [], []]
for person in all_persons:
    for label in [1, 2, 3]:
        person_path = os.path.join(data_root, person)

        bvp_signal = pd.read_csv(os.path.join(person_path, r"bvp_{1}_T{0}.csv").format(label, person),
                                 header=None)
        bvp_signal = bvp_signal.to_numpy().squeeze(-1)
        # print(bvp_signal.shape)
        ppg_cleaned = nk.ppg_clean(bvp_signal, sampling_rate=64)
        quality = nk.ppg_quality(ppg_cleaned, sampling_rate=64, method="templatematch")
        # nk.signal_plot([ppg_cleaned, quality], standardize=True)
        # plt.show()
        # plt.close()
        print(f"Person:{person} Label:{label} Quality:{np.mean(quality)}")
        quality_all.append(np.mean(quality))
        quality_label[label - 1].append(np.mean(quality))
print(f"Mean Quality:{np.mean(quality_all)}")
print(f"Label 1 Quality:{np.mean(quality_label[0])}")
print(f"Label 2 Quality:{np.mean(quality_label[1])}")
print(f"Label 3 Quality:{np.mean(quality_label[2])}")
# Mean Quality:0.7710726346665775
# Label 1 Quality:0.8266936222038955
# Label 2 Quality:0.7289637128316736
# Label 3 Quality:0.7575605689641631

