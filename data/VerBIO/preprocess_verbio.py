import os
import numpy as np
import pandas as pd
import neurokit2 as nk
import math

eda_sample_rate = 4
ppg_sample_rate = 64

clip_len = 30
step = 10

data_root = r"/home/xjc/data/VerBIO_v2/"

pre_path = os.path.join(data_root, "PRE/E4")
post_path = os.path.join(data_root, "POST/E4")

# subjects = [
#     'P005', 'P008', 'P023',
#     'P032', 'P035', 'P037',
#     'P038', 'P041', 'P043',
#     'P044', 'P046', 'P047',
#     'P049', 'P058', 'P062',
#     'P065', 'P071', 'P073'
# ]

subjects = [
    'P001', 'P003', 'P004', 'P005', 'P006', 'P007', 'P008', 'P009', 'P011', 'P012', 'P013',
    'P014', 'P016', 'P017', 'P018', 'P020', 'P021', 'P023', 'P026', 'P027', 'P031', 'P032',
    'P035', 'P037', 'P038', 'P039', 'P040', 'P041', 'P042', 'P043', 'P044', 'P045', 'P046',
    'P047', 'P048', 'P049', 'P050', 'P051', 'P052', 'P053', 'P056', 'P057', 'P058', 'P060',
    'P061', 'P062', 'P063', 'P064', 'P065', 'P066', 'P067', 'P068', 'P071', 'P072', 'P073'
]


def extract_labeled_data(subj_path):
    """
    提取每个受试者的 RELAX (0) 与 PPT (1) 数据，返回带标签的 DataFrame
    输出字段: ["label", "EDA", "PPG"]
    """

    if not os.path.isdir(subj_path):
        return

    eda_relax_file = os.path.join(subj_path, "EDA_RELAX.csv")
    bvp_relax_file = os.path.join(subj_path, "BVP_RELAX.csv")
    if os.path.exists(eda_relax_file) and os.path.exists(bvp_relax_file):
        eda_relax_df = pd.read_csv(eda_relax_file)
        bvp_relax_df = pd.read_csv(bvp_relax_file)
        eda_relax = eda_relax_df.to_numpy().squeeze(1)
        bvp_relax = bvp_relax_df.to_numpy().squeeze(1)
    else:
        print(subj_path, "miss data relax")
        return

    eda_ppt_file = os.path.join(subj_path, "EDA_PPT.csv")
    bvp_ppt_file = os.path.join(subj_path, "BVP_PPT.csv")
    if os.path.exists(eda_ppt_file) and os.path.exists(bvp_ppt_file):
        eda_ppt_df = pd.read_csv(eda_ppt_file)["EDA"]
        bvp_ppt_df = pd.read_csv(bvp_ppt_file)["BVP"]
        eda_ppt = eda_ppt_df.to_numpy()
        bvp_ppt = bvp_ppt_df.to_numpy()
    else:
        print(subj_path, "miss data ppt")
        return

    return eda_relax, bvp_relax, eda_ppt, bvp_ppt


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
    # peak_ppg_signal = nk.signal_filter(ppg_signal, ppg_sample_rate, 0.7, 2.5, method="butterworth", order=3)
    # ppg_signal = nk.signal_filter(ppg_signal, ppg_sample_rate, low, high, method="butterworth", order=3)
    ppg_peak_index = nk.ppg_findpeaks(ppg_signal, sampling_rate=ppg_sample_rate)["PPG_Peaks"]
    ppg_peak = np.zeros_like(ppg_signal)
    ppg_peak[ppg_peak_index] = 1

    # ppg_signal = nk.signal_detrend(ppg_signal, sampling_rate=ppg_sample_rate)
    # ppg_signal = nk.signal_filter(ppg_signal, ppg_sample_rate, 0.7, 2.5, method="butterworth", order=3)

    total_times = (len(ppg_signal) - clip_len * ppg_sample_rate) / (step * ppg_sample_rate) + 1
    for times in range(math.floor(total_times)):
        # ppg_start_index = img_index + times_index - (self.clip_len - 1) * self.jump_num
        ppg_start_index = int(times * step * ppg_sample_rate)
        ppg_end_index = int(ppg_start_index + clip_len * ppg_sample_rate)
        ppg_signal_ = ppg_signal[ppg_start_index: ppg_end_index]
        ppg_peak_ = ppg_peak[ppg_start_index: ppg_end_index]

        eda_start_index = int(times * step * eda_sample_rate)
        eda_end_index = int(eda_start_index + clip_len * eda_sample_rate)
        eda_signal_ = eda_c[eda_start_index: eda_end_index]
        scr_signal_ = scr[eda_start_index: eda_end_index]
        scl_signal_ = scl[eda_start_index: eda_end_index]

        print(ppg_signal_.shape)
        print(scr_signal_.shape)

        # print("cwtmatr_", cwtmatr_.shape)
        # print("movement_info_", movement_info_.shape)
        datas = [dir_, label, ppg_signal_, scl_signal_, scr_signal_, eda_signal_, ppg_peak_]

        save_datas.append(datas)


if __name__ == '__main__':
    save_datas = []
    for subj in subjects:
        pre_sub_path = os.path.join(pre_path, subj)
        post_sub_path = os.path.join(post_path, subj)
        for sub_path in [pre_sub_path, post_sub_path]:
            try:
                eda_relax, bvp_relax, eda_ppt, bvp_ppt = extract_labeled_data(sub_path)
                cut_signal(subj, bvp_relax, eda_relax, 0, save_datas)
                cut_signal(subj, bvp_ppt, eda_ppt, 1, save_datas)
            except Exception as e:
                print(e)
                continue
    np.save("VerBIO_clip{0}s_multi.npy".format(clip_len), np.array(save_datas, dtype=object))
