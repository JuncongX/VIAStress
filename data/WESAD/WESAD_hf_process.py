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

from utils.HRV import HRV
from utils.filter import butter_bandpass_filter, detrend
import pickle
import neurokit2 as nk

from comparation.ClassicML.feature_extraction import get_plf
import matplotlib
matplotlib.use('Agg')

# Dynamic Alignment and Fusion of Multimodal Physiological Patterns for Stress Recognition
step = 10
clip_len = 30

ecg_sample_rate = 700
ppg_sample_rate = 64
eda_sample_rate = 4

# Stress Detection Using Context-Aware Sensor Fusion From Wearable Devices
# clip_len = 3840
# step = 320
# order = 3
# low = 0.7
# high = 3.7

# clip_len = 3840
# step = 320
order = 3
low = 0.5
high = 8



# def norm(ppg_signal):
#     ppg_mean = np.mean(ppg_signal)
#     ppg_max = np.max(ppg_signal)
#     ppg_min = np.min(ppg_signal)
#     ppg_signal = (ppg_signal - ppg_mean) / (ppg_max - ppg_min)
#     return ppg_signal

def cut_signal(dir_, ppg_signal, eda_signal, label, save_datas):
    print(dir_)
    print(label)
    # ppg_signal = signal.resample(ppg_signal, int(len(ppg_signal) / 64 * resample_rate))
    # ppg_signal = detrend(ppg_signal)
    eda, eda_info = nk.eda_process(eda_signal, sampling_rate=eda_sample_rate, show=False)
    eda_c = eda["EDA_Clean"]

    # ppg_signal = nk.signal_detrend(ppg_signal, sampling_rate=ppg_sample_rate)
    ppg_signal = nk.ppg_clean(ppg_signal, ppg_sample_rate)
    ppg_signal = nk.signal_filter(ppg_signal, ppg_sample_rate, low, high, method="butterworth", order=3)

    # ppg_signal = butter_bandpass_filter(ppg_signal, ppg_sample_rate, low, high)
    total_times = (len(ppg_signal) - clip_len * ppg_sample_rate) / (step * ppg_sample_rate)
    for times in range(math.floor(total_times) - 1):
        # ppg_start_index = img_index + times_index - (self.clip_len - 1) * self.jump_num
        ppg_start_index = int(times * step * ppg_sample_rate)
        ppg_end_index = int(ppg_start_index + clip_len * ppg_sample_rate)
        ppg_signal_ = ppg_signal[ppg_start_index: ppg_end_index]

        eda_start_index = int(times * step * eda_sample_rate)
        eda_end_index = int(eda_start_index + clip_len * eda_sample_rate)
        # eda_signal_ = eda_signal[eda_start_index: eda_end_index]

        eda_signal_ = eda_c[eda_start_index: eda_end_index]

        feature = get_plf(eda_signal_, ppg_signal_)

        datas = [dir_, label, feature]

        save_datas.append(datas)


def get_data(number, save_datas):
    dir_ = "S{0}".format(number)
    s_path = r"/home/som/8T/DataSets/WESAD/{0}/{0}.pkl".format(dir_)
    with open(s_path, 'rb') as file:
        s_data = pickle.load(file, encoding='latin1')
    w_bvp = s_data['signal']['wrist']['BVP'][:, 0]  # 64Hz
    w_eda = s_data['signal']['wrist']['EDA'][:, 0]  # 4Hz
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

    baseline_index_eda = [int(i / ecg_sample_rate * eda_sample_rate) for i in baseline_index_ecg]
    stress_index_eda = [int(i / ecg_sample_rate * eda_sample_rate) for i in stress_index_ecg]
    amusement_index_eda = [int(i / ecg_sample_rate * eda_sample_rate) for i in amusement_index_ecg]

    bvp_baseline = w_bvp[baseline_index_bvp[0]: baseline_index_bvp[1]]
    bvp_stress = w_bvp[stress_index_bvp[0]: stress_index_bvp[1]]
    bvp_amusement = w_bvp[amusement_index_bvp[0]: amusement_index_bvp[1]]

    eda_baseline = w_eda[baseline_index_eda[0]: baseline_index_eda[1]]
    eda_stress = w_eda[stress_index_eda[0]: stress_index_eda[1]]
    eda_amusement = w_eda[amusement_index_eda[0]: amusement_index_eda[1]]

    cut_signal(dir_, bvp_baseline, eda_baseline, 0, save_datas)
    cut_signal(dir_, bvp_stress, eda_stress, 1, save_datas)
    cut_signal(dir_, bvp_amusement, eda_amusement, 2, save_datas)


if __name__ == '__main__':
    havent_done = [2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 13, 14, 15, 16, 17]
    save_datas = []
    for number in havent_done:
        get_data(number, save_datas)
    np.save("WESAD_HF_clip{0}s.npy".format(clip_len, order, low, high), save_datas)
