import numpy as np
from scipy.signal import butter, filtfilt


def bandpass_filter(signal, lowcut, highcut, fs, order):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    return filtfilt(b, a, signal)

def compute_ppg_noise_ratio(ppg_signal, fs=64, lowcut=0.5, highcut=8.0, order=3):
    """
    计算PPG信号的噪声占比（频域方法）

    参数:
        ppg_signal : ndarray
            一维PPG信号
        fs : float
            采样率 (Hz)
        lowcut : float
            心率信号下限频率 (Hz)
        highcut : float
            心率信号上限频率 (Hz)
        order : int
            带通滤波器阶数

    返回:
        noise_ratio : float
            噪声占比 (0~1)
    """

    # FFT计算功率谱
    N = len(ppg_signal)
    freqs = np.fft.rfftfreq(N, d=1 / fs)
    fft_vals = np.fft.rfft(ppg_signal)
    power_spectrum = np.abs(fft_vals) ** 2

    # 4. 信号能量（心跳频段）
    signal_band = (freqs >= lowcut) & (freqs <= highcut)
    E_signal = np.sum(power_spectrum[signal_band])

    # 5. 噪声能量（其他频段）
    E_noise = np.sum(power_spectrum) - E_signal

    # 6. 噪声占比
    noise_ratio = E_noise / (E_signal + E_noise)

    return noise_ratio


def compute_eda_noise_ratio(eda_signal, fs=4, lowcut=0.01, highcut=1):
    """
    计算EDA信号噪声占比（频域方法）

    参数:
        eda_signal : ndarray
            一维EDA信号
        fs : float
            采样率 Hz
        signal_band : tuple
            信号频率范围 (Hz)
        order : int
            滤波器阶数

    返回:
        noise_ratio : float
            噪声占比
    """
    # 1. 带通滤波（去掉低频漂移和高频噪声）
    # 2. FFT计算功率谱
    N = len(eda_signal)
    freqs = np.fft.rfftfreq(N, d=1 / fs)
    fft_vals = np.fft.rfft(eda_signal)
    power_spectrum = np.abs(fft_vals) ** 2

    # 3. 信号能量
    signal_mask = freqs <= highcut
    E_signal = np.sum(power_spectrum[signal_mask])

    # 4. 噪声能量
    E_noise = np.sum(power_spectrum) - E_signal

    # 5. 噪声占比
    noise_ratio = E_noise / (E_signal + E_noise)

    return noise_ratio