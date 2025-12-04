import numpy as np
from scipy.signal import welch
from data.cvxEDA import cvxEDA
from scipy.signal import periodogram
from scipy.signal import find_peaks


def compute_eda_snr(eda_signal, fs=4.0, component='phasic+tonic'):
    """
    Reference: A Deep Convolutional Autoencoder for Automatic Motion Artifact Removal in Electrodermal Activity
    使用 cvxEDA 分解 EDA 信号并计算 SNR（dB）和残差

    参数:
    -----------
    eda_signal : numpy array
        原始 EDA 信号
    fs : float
        采样率 (Hz)，默认 4.0
    component : str
        用于计算 SNR 的分解信号，可选：
        'phasic' -> 仅 phasic 成分
        'tonic' -> 仅 tonic 成分
        'phasic+tonic' -> phasic + tonic（默认）

    返回:
    -----------
    result : dict
        {
            'SNR_dB': float,  # 信噪比，单位 dB
            'residual': np.array,  # 残差噪声 = 原始信号 - 分解信号
            'signal': np.array   # 用于计算 SNR 的分解信号
        }
    """

    # 1. 使用 cvxEDA 分解信号
    phasic, p, tonic, l, d, e, obj = cvxEDA(eda_signal, 1 / fs)

    # 2. 根据选择的 component 获取信号
    if component == 'phasic':
        x = phasic
    elif component == 'tonic':
        x = tonic
    elif component == 'phasic+tonic':
        x = phasic + tonic
    else:
        raise ValueError("component 参数必须是 'phasic', 'tonic' 或 'phasic+tonic'")

    # 3. 残差 = 原始信号 - 分解信号
    residual = eda_signal - x

    # 4. 按公式计算 SNR(dB)
    numerator = np.sum(x ** 2)
    denominator = np.sum(residual ** 2)

    if denominator == 0:
        snr_db = np.inf
    else:
        snr_db = 10 * np.log10(numerator / denominator)

    # return {'SNR_dB': snr_db, 'residual': residual, 'signal': x}
    return snr_db


def compute_ppg_snr(BVP, FS):
    """
    De Haan G, Jeanne V. Robust pulse rate from chrominance-based rPPG[J]. IEEE transactions on biomedical engineering, 2013, 60(10): 2878-2886.
    Estimate the signal-to-noise ratio (SNR) of a blood volume pulse (BVP) signal.
    Adapted from G. de Haan, TBME, 2013 and Daniel McDuff's MATLAB implementation.

    Parameters
    ----------
    BVP : array-like
        The BVP time series signal.
    FS : float
        Sampling frequency in Hz.

    Returns
    -------
    SNR : float
        Blood Volume Pulse Signal-to-Noise Ratio (in dB).
    HR_F : float
        The dominant heart rate frequency (Hz) detected in the 0.5–4 Hz range.
    """

    NyquistF = FS / 2
    FResBPM = 0.5  # resolution (bpm) of bins
    N = int((60 * 2 * NyquistF) / FResBPM)  # number of bins

    # Compute power spectrum
    f, Pxx = periodogram(BVP, window=np.hamming(len(BVP)), nfft=N, fs=FS)

    # Focus on physiological band (0.5–4 Hz)
    valid_mask = (f >= 0.5) & (f <= 4)
    f_band = f[valid_mask]
    Pxx_band = Pxx[valid_mask]

    # Find main peak frequency in the 0.5–4 Hz band
    peaks, _ = find_peaks(Pxx_band)
    if len(peaks) == 0:
        return np.nan, np.nan  # no detectable HR peak

    main_peak_idx = peaks[np.argmax(Pxx_band[peaks])]
    HR_F = f_band[main_peak_idx]  # dominant HR frequency (Hz)

    # Signal power: HR ±0.1 Hz and harmonic (2*HR ±0.2 Hz)
    GTMask1 = (f >= HR_F - 0.1) & (f <= HR_F + 0.1)
    GTMask2 = (f >= 2 * HR_F - 0.2) & (f <= 2 * HR_F + 0.2)
    SPower = np.sum(Pxx[GTMask1 | GTMask2])

    # Total power in 0.5–4 Hz band
    FMask2 = (f >= 0.5) & (f <= 4)
    AllPower = np.sum(Pxx[FMask2])

    # Compute SNR in dB
    SNR = 10 * np.log10(SPower / (AllPower - SPower + np.finfo(float).eps))

    return SNR


def evaluate_eda_quality(eda_signal, fs=4.0, method='tbme'):
    """
    Reference: Short-Term Detection of Dynamic Stress Levels in Exergaming with Wearables
    evaluate eda quality
    :param eda_signal: numpy array
    :param fs: sampling rate (Hz)
    :return: dict {'SNR_dB': float, 'variance': float, 'quality': 0/1 (good/bad)}
    """

    # 计算信号方差
    var = np.var(eda_signal)

    if method == 'sensor':
        # 计算功率谱密度
        f, Pxx = welch(eda_signal, fs=fs, nperseg=min(256, len(eda_signal)))

        # 计算 0-0.5 Hz 功率（信号）和 0.5-2 Hz 功率（噪声）
        signal_power = np.trapz(Pxx[(f >= 0) & (f <= 0.5)], f[(f >= 0) & (f <= 0.5)])
        noise_power = np.trapz(Pxx[(f > 0.5) & (f <= 2)], f[(f > 0.5) & (f <= 2)])

        # 避免除以零
        if noise_power == 0:
            snr_db = np.inf
        else:
            snr_db = 10 * np.log10(signal_power / noise_power)
    elif method == 'tbme':
        snr_db = compute_eda_snr(eda_signal, fs)
    else:
        raise Exception("No such method")
    # 判断信号质量
    if snr_db < 20 or var < 0.001:
        quality = 1
    else:
        quality = 0

    return {'SNR_dB': snr_db, 'variance': var, 'quality': quality}


def evaluate_ppg_quality(ppg_signal, fs=64.0):
    """
    Evaluate PPG (BVP) signal quality.
    Yang P, Liu N, Liu X, et al. A multimodal dataset for mixed emotion recognition[J]. Scientific Data, 2024, 11(1): 847.
    PulseGAN: Learning to Generate Realistic Pulse Waveforms in Remote Photoplethysmography

    Parameters
    ----------
    ppg_signal : numpy array
        PPG (BVP) signal.
    fs : float
        Sampling frequency in Hz.
    method : str
        Method to compute SNR ('tbme' -> uses compute_ppg_snr).

    Returns
    -------
    dict
        {
            'SNR_dB': float,
            'quality': int (0=good, 1=bad)
        }
    """

    # 计算 SNR
    snr_db = compute_ppg_snr(ppg_signal, fs)
    # 根据信噪比和方差判定质量
    # SNR < -5 dB → 差 (quality=1)
    if snr_db < -5:
        quality = 1  # 差
    else:
        quality = 0  # 好

    return {
        'SNR_dB': snr_db,
        'quality': quality
    }
