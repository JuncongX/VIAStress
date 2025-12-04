import neurokit2 as nk
import pandas as pd
import numpy as np
import math
import os
import csv
# from utils.filter import butter_bandpass_filter, detrend

data_root = r"/home/som/8T/DataSets/ubfc_phys/video/"

ppg_sample_rate = 64
eda_sample_rate = 4

step = 10
clip_len = 30

order = 3
low = 0.5
high = 8


def z_score(signal):
    signal_mean = np.mean(signal)
    signal = signal - signal_mean
    signal_std = np.std(signal)

    return signal / signal_std


def cut_signal(dir_, ppg_signal, eda_signal, label, save_datas):
    print(dir_)
    print(label)
    # ppg_signal = signal.resample(ppg_signal, int(len(ppg_signal) / 64 * resample_rate))
    # ppg_signal = detrend(ppg_signal)
    eda, eda_info = nk.eda_process(eda_signal, sampling_rate=eda_sample_rate)
    scr = eda["EDA_Phasic"]
    scl = eda["EDA_Tonic"]
    eda_c = eda["EDA_Clean"]
    eda_c = nk.signal_filter(eda_c, sampling_rate=eda_sample_rate, lowcut=None, highcut=1, method='butterworth',
                             order=4)

    ppg_signal = nk.ppg_clean(ppg_signal, ppg_sample_rate)
    # ppg_signal = nk.signal_filter(ppg_signal, ppg_sample_rate, low, high, method="butterworth", order=3)

    # ppg_signal = nk.signal_detrend(ppg_signal, sampling_rate=ppg_sample_rate)
    # ppg_signal = nk.signal_filter(ppg_signal, ppg_sample_rate, 0.7, 2.5, method="butterworth", order=3)

    total_times = (len(ppg_signal) - clip_len * ppg_sample_rate) / (step * ppg_sample_rate)
    for times in range(math.floor(total_times) - 1):
        # ppg_start_index = img_index + times_index - (self.clip_len - 1) * self.jump_num
        ppg_start_index = int(times * step * ppg_sample_rate)
        ppg_end_index = int(ppg_start_index + clip_len * ppg_sample_rate)
        ppg_signal_ = ppg_signal[ppg_start_index: ppg_end_index]

        eda_start_index = int(times * step * eda_sample_rate)
        eda_end_index = int(eda_start_index + clip_len * eda_sample_rate)
        eda_signal_ = eda_c[eda_start_index: eda_end_index]
        scr_signal_ = scr[eda_start_index: eda_end_index]
        scl_signal_ = scl[eda_start_index: eda_end_index]

        # ppg_signal_ = outlier(ppg_signal_)
        # ppg_signal_ = butter_bandpass_filter(ppg_signal_, 35)
        # ppg_signal_ = detrend(ppg_signal_)
        # ppg_signal_ = norm(ppg_signal_)
        # ppg_signal_ = z_score(ppg_signal_)
        # eda_signal_ = z_score(eda_signal_)
        # scr_signal_ = z_score(scr_signal_)
        # scl_signal_ = z_score(scl_signal_)
        print(ppg_signal_.shape)
        print(scr_signal_.shape)

        # print("cwtmatr_", cwtmatr_.shape)
        # print("movement_info_", movement_info_.shape)
        datas = [dir_, label, ppg_signal_, scl_signal_, scr_signal_, eda_signal_]

        # mi_list = []
        # mistd_list = []
        # for i in range(movement_info_.shape[1]):
        #     mi = missing_fix(movement_info_[:, i])
        #     mi_list.append(np.gradient(mi))
        #     mistd_list.append(np.std(mi, ddof=1))
        # datas = datas + mi_list + mistd_list
        save_datas.append(datas)


def get_data(task, dir_, save_datas):
    person_path = os.path.join(data_root, dir_)

    bvp_signal = pd.read_csv(os.path.join(person_path, r"bvp_{1}_T{0}.csv").format(task, dir_),
                             header=None)
    bvp_signal = bvp_signal.to_numpy().squeeze(-1)

    eda_signal = pd.read_csv(os.path.join(person_path, r"eda_{1}_T{0}.csv").format(task, dir_),
                             header=None)
    eda_signal = eda_signal.to_numpy().squeeze(-1)

    df_info = pd.read_csv(os.path.join(person_path, r'info_{0}.txt'.format(dir_)), header=None)
    if task == 1:
        level = 0
    else:
        if df_info.values[2][0] == "test":
            level = 1
        else:
            level = 2
    cut_signal(dir_, bvp_signal, eda_signal, task, save_datas)


if __name__ == '__main__':
    havent_done = ['s1', 's2', 's3', 's4', 's5', 's6', 's7', 's8', 's9', 's10', 's11', 's13', 's14', 's15', 's16',
                   's18', 's19', 's20', 's21', 's22', 's23', 's24', 's25', 's26', 's27', 's28', 's29', 's30', 's31',
                   's32', 's33', 's34', 's35', 's36', 's37', 's38', 's39', 's40', 's41', 's42', 's43', 's44', 's45',
                   's46', 's48', 's49', 's50', 's51', 's52', 's53', 's54', 's55', 's56', 's12', 's17', 's47']
    save_datas = []
    for dir_ in havent_done:
        person_path = os.path.join(data_root, dir_)
        for task in [1, 2, 3]:
            print(dir_, task)
            get_data(task, dir_, save_datas)
    np.save("UBFC_Phys_clip{0}s_multi.npy".format(clip_len), save_datas)
