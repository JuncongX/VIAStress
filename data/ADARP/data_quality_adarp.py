import neurokit2 as nk
import numpy as np
import glob

import matplotlib.pyplot as plt

# 存储每个参与者的质量得分
quality_scores = {}
sampling_rate = 64  # BVP信号采样率

file_list = sorted(glob.glob("ADARP_clip30s_multi_*C.npy"))
print(f"检测到 {len(file_list)} 个文件")


# file_path = "ADARP/ADARP_clip30s_multi_undersampling.npy"


def ppg_quality(ppg, ppg_sampling_rate):
    return nk.ppg_quality(ppg, sampling_rate=ppg_sampling_rate, method='disimilarity')


for file in file_list:
    # 普通加载（Python对象类型，不适用 mmap）
    data = np.load(file, allow_pickle=True)

    for entry in data:
        participant_ids, label, bvp, scr, scl, eda, peak = entry
        q_s = ppg_quality(bvp, sampling_rate)
        # 保存每个参与者的质量分数
        if participant_ids not in quality_scores:
            quality_scores[participant_ids] = []
        quality_scores[participant_ids].append(q_s)

# 将所有参与者的得分转换为便于绘图的格式
participants = list(quality_scores.keys())
scores = [np.ravel(quality_scores[p]) for p in participants]

# 绘制箱型图
plt.figure(figsize=(12, 6))
plt.boxplot(scores, labels=participants, patch_artist=True)
plt.title("PPG Signal Quality Scores per Participant", fontsize=14)
plt.xlabel("Participant ID", fontsize=12)
plt.ylabel("PPG Quality Score (Dissimilarity)", fontsize=12)
plt.xticks(rotation=45)
plt.grid(True, linestyle="--", alpha=0.6)
plt.tight_layout()
plt.show()
