import neurokit2 as nk
import biosppy
import pyhrv
import pyhrv.tools as tools
import pyhrv.time_domain as td
import pyhrv.frequency_domain as fd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import pearsonr
import scipy
import matplotlib
import math

matplotlib.use("Agg")


def analysis_ppg(ppg, sample_rate=64):
    ppg = nk.ppg_clean(ppg, sampling_rate=sample_rate, method='elgendi')
    peaks, info = nk.ppg_peaks(ppg, sampling_rate=sample_rate, method="elgendi", show=False)
    nni_samples = np.diff(info["PPG_Peaks"])
    nni_ms = nni_samples * (1000 / sample_rate)

    HR_info = td.hr_parameters(nni=nni_ms)
    HR_mean = HR_info["hr_mean"]
    HR_std = HR_info["hr_std"]

    NN50_info = td.nn50(nni=nni_ms)
    pNN50 = NN50_info["pnn50"]
    NN50 = NN50_info["nn50"]

    nk_td = nk.hrv_time(peaks, sampling_rate=sample_rate, show=False)
    TINN = nk_td["HRV_TINN"].tolist()[0]
    HRV_mean = nk_td["HRV_MeanNN"].tolist()[0]
    HRV_std = nk_td["HRV_SDNN"].tolist()[0]
    RMS = np.sqrt(np.mean(np.square(nni_ms)))

    fbands = {'ulf': (0.01, 0.04), 'vlf': (0.04, 0.15), 'lf': (0.15, 0.4), 'hf': (0.4, 1.0)}
    f_result = fd.welch_psd(nni=nni_ms, show=False, fbands=fbands)
    f_result_ = fd.welch_psd(nni=nni_ms, show=False)

    LF_norm = f_result_["fft_norm"][0]
    HF_norm = f_result_["fft_norm"][1]

    ULF = f_result["fft_abs"][0]
    LF = f_result["fft_abs"][1]
    HF = f_result["fft_abs"][2]
    VHF = f_result["fft_abs"][3]
    ULF_rel = f_result["fft_rel"][0]
    LF_rel = f_result["fft_rel"][1]
    HF_rel = f_result["fft_rel"][2]
    VHF_rel = f_result["fft_rel"][3]
    LFHF_ratio = LF / HF
    Frequency_SUM = ULF + LF + HF + VHF
    return [HR_mean, HR_std, HRV_mean, HRV_std, NN50, pNN50, TINN, RMS, ULF, LF, HF, VHF, LFHF_ratio, Frequency_SUM,
            ULF_rel, LF_rel, HF_rel, VHF_rel, LF_norm, HF_norm]


def analysis_eda(eda_raw, sample_rate=4):
    eda = nk.eda_clean(eda_raw, sampling_rate=sample_rate)

    scr_ = nk.eda_phasic(eda)
    scr = scr_["EDA_Phasic"].values.tolist()
    scl = scr_["EDA_Tonic"].values.tolist()
    scr_info = nk.eda_findpeaks(scr_["EDA_Phasic"].values)
    scr_info = nk.eda_fixpeaks(scr_info)

    eda = nk.signal_filter(eda, sampling_rate=sample_rate, lowcut=None, highcut=1, method='butterworth', order=4)
    eda_std = np.std(eda)
    eda_mean = np.average(eda)
    eda_min = np.min(eda)
    eda_max = np.max(eda)
    eda_range = eda_max - eda_min
    # slope
    # hanning_window = np.hanning(100)
    # smoothed_eda = np.convolve(eda, hanning_window, mode='same') / hanning_window.sum()
    eda_slope = (eda[-1] - eda[0]) / (len(eda) - 1)

    scl_std = np.std(scl)
    scl_mean = np.average(scl)
    scr_std = np.std(scr)

    scl_time = np.arange(len(scl))
    scl_corr, _ = pearsonr(scl, scl_time)

    scr_onsets_index = scr_info["SCR_Onsets"]
    scr_peaks_index = scr_info["SCR_Peaks"]

    if math.isnan(scr_onsets_index[0]):
        scr_onsets_index[0] = 0
    if math.isnan(scr_peaks_index[-1]):
        scr_peaks_index[-1] = len(scr) - 1

    scr_num = len(scr_peaks_index)

    scr_startle_response_durations = (scr_peaks_index - scr_onsets_index) * (1 / sample_rate)
    scr = np.array(scr)
    scr_startle_magnitudes = scr[scr_peaks_index.astype(int)] - scr[scr_onsets_index.astype(int)]

    scr_startle_response_durations_sum = np.sum(scr_startle_response_durations)
    scr_startle_magnitudes_sum = np.sum(scr_startle_magnitudes)

    scr_area = 0
    if scr_startle_response_durations_sum != 0:
        for (o_d, o_m) in zip(scr_startle_response_durations, scr_startle_magnitudes):
            scr_area += o_d * o_m / 2

    return [eda_mean, eda_std, eda_min, eda_max, eda_slope, eda_range, scl_mean, scl_std, scr_std, scl_corr,
            scr_num, scr_startle_response_durations_sum, scr_startle_magnitudes_sum, scr_area]


def get_plf(eda, ppg):
    ppg_f = analysis_ppg(ppg)
    eda_f = analysis_eda(eda)

    return ppg_f + eda_f


if __name__ == '__main__':
    ppg = nk.ppg_simulate(duration=30, sampling_rate=64)
    ppg_f = analysis_ppg(ppg)
    print(ppg_f)

    eda = nk.eda_simulate(duration=30, sampling_rate=4, scr_number=5, drift=0.1, noise=0)
    eda_f = analysis_eda(eda, 4)
    print(eda_f)

    print(get_plf(eda, ppg))
