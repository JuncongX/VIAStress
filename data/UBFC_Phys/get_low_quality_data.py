import numpy as np
from tqdm import tqdm
from data.data_evaluate import evaluate_ppg_quality, evaluate_eda_quality

# === 加载原始数据 ===
file_in = "UBFC_Phys_clip30s_multi_peak808_.npy"
file_out = "UBFC_Phys_clip30s_multi_low_quality.npy"

print(f"Loading {file_in} ...")
data = np.load(file_in, allow_pickle=True)

low_quality_data = []

print(f"Total samples: {len(data)}")
for sample in tqdm(data):
    # 数据结构：
    # [dir_, label, ppg_signal_, scl_signal_, scr_signal_, eda_signal_, ppg_peak_]
    dir_ = sample[0]
    label = sample[1]
    ppg_signal = sample[2]
    scl_signal = sample[3]
    scr_signal = sample[4]
    eda_signal = sample[5]
    ppg_peak = sample[6]

    if not isinstance(ppg_signal, np.ndarray):
        sample[2] = np.asarray(ppg_signal, dtype=float)
    if not isinstance(scl_signal, np.ndarray):
        # 关键修改：把 Series 转成纯 numpy 数组
        sample[3] = np.asarray(eda_signal, dtype=float)
    if not isinstance(scr_signal, np.ndarray):
        # 关键修改：把 Series 转成纯 numpy 数组
        sample[4] = np.asarray(eda_signal, dtype=float)
    if not isinstance(eda_signal, np.ndarray):
        # 关键修改：把 Series 转成纯 numpy 数组
        sample[5] = np.asarray(eda_signal, dtype=float)

    # === 评估 PPG 信号质量 ===
    try:
        ppg_quality_result = evaluate_ppg_quality(ppg_signal, fs=64.0)
        print(ppg_quality_result)
        ppg_quality = ppg_quality_result['quality']
    except Exception as e:
        print(f"[PPG ERROR] {dir_}: {e}")
        ppg_quality = 1  # 出错视为低质量

    # === 评估 EDA 信号质量 ===
    try:
        eda_quality_result = evaluate_eda_quality(eda_signal, fs=4.0)
        print(eda_quality_result)
        eda_quality = eda_quality_result['quality']
    except Exception as e:
        print(f"[EDA ERROR] {dir_}: {e}")
        eda_quality = 1  # 出错视为低质量

    # === 判断是否任一信号为低质量 ===
    if ppg_quality == 1 or eda_quality == 1:
        low_quality_data.append(sample)

print(f"\nLow-quality samples: {len(low_quality_data)} / {len(data)}")

# === 保存结果 ===
np.save(file_out, low_quality_data, allow_pickle=True)
print(f"Saved low-quality data to {file_out}")