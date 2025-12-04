# Reference: Nasseri M, Nurse E, Glasstetter M, et al. Signal quality and patient experience with wearable devices for epilepsy management[J]. Epilepsia, 2020, 61: S25-S35.

import numpy as np
from scipy import signal
import pandas as pd


def compute_sqi_rac(eda_signal, fs):
    """
    计算 EDA 信号的质量指标 SQI_RAC。

    参数:
        eda_signal (np.ndarray): 输入的 EDA 信号，单位 µS。
        fs (int): 采样率，例如 E4 为 4 Hz。

    返回:
        sqi_flags (np.ndarray): 每秒一个质量标志，1 表示正常，0 表示存在伪影。
        minute_sqi (list): 每分钟一个 SQI 值，0 表示质量差，1 表示正常。
    """
    eda_signal = np.asarray(eda_signal)
    n_samples = len(eda_signal)
    n_seconds = n_samples // fs
    sqi_flags = []

    # 每秒计算幅度变化百分比
    for sec in range(n_seconds):
        start = sec * fs
        end = start + fs
        window = eda_signal[start:end]
        if len(window) < fs:
            break
        max_val = np.max(window)
        max_index = np.argmax(window)
        min_val = np.min(window)
        min_index = np.argmin(window)
        if max_index > min_index:
            change_percent = (max_val - min_val) / min_val
        else:
            change_percent = (min_val - max_val) / max_val

        # 判断伪影
        if change_percent > 0.2 or change_percent < -0.1:
            sqi_flags.append(0)  # 伪影
        else:
            sqi_flags.append(1)  # 正常

    return np.array(sqi_flags)

dataset = "ubfc_phys"
# dataset = "wesad"
task = 3

if dataset == "ubfc_phys":
    data = np.load('UBFC_Phys_clip30s_multi_peak808.npy', allow_pickle=True)
    w_label = [1, task]
else:
    data = np.load('WESAD_clip30s_multi_peak808.npy', allow_pickle=True)
    w_label = [0, 1, 2]

needed_data = None
for label in w_label:
    selected_data = data[(data[:, 1] == label)]
    if needed_data is None:
        needed_data = selected_data
    else:
        needed_data = np.concatenate((needed_data, selected_data))

total_n_artifacts = 0
total_seconds = 0
signals = [s for s in needed_data[:, 5]]
for s in signals:
    sqi_flags = compute_sqi_rac(s, 4)
    n_artifacts = np.sum(sqi_flags == 0)  # 统计伪影数量
    total_n_artifacts += n_artifacts
    total_seconds += len(sqi_flags)
print(total_n_artifacts / total_seconds)
print(f"信号中伪影秒数（SQI=0）: {total_n_artifacts}")
print(f"平均每段信号的伪影比例: {total_n_artifacts / total_seconds:.3%}")

