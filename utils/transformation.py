import numpy as np
import os
import scipy.io as scio
import random
from random import choice
from scipy.signal import savgol_filter, medfilt
from sklearn import preprocessing

def inter_data(hr, window=11):
    N = window
    time3 = savgol_filter(hr, window_length=N, polyorder=2)
    return time3


# Noise
def noised(signal):
    SNR = 5
    noise = np.random.randn(signal.shape[0])
    noise = noise - np.mean(noise)
    signal_power = np.linalg.norm(signal) ** 2 / signal.size
    noise_variance = signal_power / np.power(10, (SNR / 10))
    noise = (np.sqrt(noise_variance) / np.std(noise)) * noise
    signal_noise = noise + signal
    return signal_noise


# Negate
def negated(signal):
    return signal * -1


# Reverse
def opposite_time(signal):
    return signal[::-1]


# Permute
def permuted(signal):
    listA = [0, 1, 2, 3, 4]
    unit_len = int(1050 / len(listA))
    random.shuffle(listA)
    sig = signal[listA[0] * unit_len:listA[0] * unit_len + unit_len]
    for i in range(1, len(listA)):
        sig = np.hstack((sig, signal[listA[i] * unit_len:listA[i] * unit_len + unit_len]))
    return sig


# Scale
def scale(signal):
    sc = [0.5, 2, 1.5, 0.8]
    s = choice(sc)
    return signal * s


# Smooth
def time_warp(signal):
    signal = inter_data(signal, 11)
    return signal


def regular_mm(data):
    # min_max_scaler = preprocessing.MinMaxScaler()
    data = data.reshape(-1, data.shape[0])
    # data = min_max_scaler.fit_transform(data)
    data_mean = np.mean(data)
    data = data - data_mean
    data_std = np.std(data)

    return data / data_std


def transformation(dataX):
    r_data = np.zeros((7, dataX.shape[0]))

    data_no = noised(dataX.copy())
    data_ne = negated(dataX.copy())
    data_op = opposite_time(dataX.copy())
    data_pe = permuted(dataX.copy())
    data_sc = scale(dataX.copy())
    data_ti = time_warp(dataX.copy())

    data_raw = regular_mm(dataX)
    data_no = regular_mm(data_no)
    data_ne = regular_mm(data_ne)
    data_op = regular_mm(data_op)
    data_pe = regular_mm(data_pe)
    data_sc = regular_mm(data_sc)
    data_ti = regular_mm(data_ti)

    r_data[0] = data_raw
    r_data[1] = data_no
    r_data[2] = data_ne
    r_data[3] = data_op
    r_data[4] = data_pe
    r_data[5] = data_sc
    r_data[6] = data_ti

    return r_data