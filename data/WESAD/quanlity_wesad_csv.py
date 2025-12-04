import neurokit2 as nk
import numpy as np
import pandas as pd
import pickle
import os

# ===============================
# 参数配置
# ===============================
ecg_sample_rate = 700
ppg_sample_rate = 64
all_persons = [f"S{s_i}" for s_i in [2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 13, 14, 15, 16, 17]]
output_csv = r"wesad_ppg_quality_per_person.csv"

# ===============================
# 定义安全质量计算函数
# ===============================
def safe_ppg_quality(ppg, person, label):
    """计算 PPG 质量分数序列，带错误保护"""
    try:
        if len(ppg) < ppg_sample_rate * 2:
            print(f"[Skip] {person}-{label}: signal too short")
            return np.array([])
        ppg = np.nan_to_num(ppg)
        ppg_cleaned = nk.ppg_clean(ppg, sampling_rate=ppg_sample_rate)
        q = nk.ppg_quality(ppg_cleaned, sampling_rate=ppg_sample_rate, method="templatematch")
        print(f"Person:{person} Label:{label} -> Quality length:{len(q)}")
        return np.asarray(q)
    except Exception as e:
        print(f"[Error] {person}-{label}: {e}")
        return np.array([])

# ===============================
# 主循环
# ===============================
results = []

for person in all_persons:
    s_path = f"E:\dataset\WESAD\{person}\{person}.pkl"
    if not os.path.exists(s_path):
        print(f"[Missing] {s_path}")
        continue

    with open(s_path, "rb") as file:
        s_data = pickle.load(file, encoding="latin1")

    w_bvp = s_data["signal"]["wrist"]["BVP"][:, 0]
    w_label = s_data["label"]
    total_num = len(w_label)

    # 三种状态对应的ECG标签
    label_map = {"baseline": 1, "stress": 2, "amusement": 3}
    segment_indices = {}

    for label_name, label_id in label_map.items():
        label_mask = w_label == label_id
        start_end = []
        flag = False
        for i in range(total_num):
            if label_mask[i] and not flag:
                start = i
                flag = True
            elif not label_mask[i] and flag:
                end = i
                flag = False
                start_end.append((start, end))
        segment_indices[label_name] = start_end

    # 分别提取三个状态对应的BVP段
    person_scores = []  # 保存该被试的所有质量分数

    for label_name, pairs in segment_indices.items():
        for (start, end) in pairs:
            start_bvp = int(start / ecg_sample_rate * ppg_sample_rate)
            end_bvp = int(end / ecg_sample_rate * ppg_sample_rate)
            bvp_segment = w_bvp[start_bvp:end_bvp]

            q_values = safe_ppg_quality(bvp_segment, person, label_name)
            if len(q_values) > 0:
                person_scores.extend(q_values.tolist())

    # 保存结果
    if len(person_scores) > 0:
        results.append({
            "Person": person,
            "Num_Samples": len(person_scores),
            "Scores": person_scores
        })

# ===============================
# 保存到 CSV 文件
# ===============================
df = pd.DataFrame(results)
df.to_csv(output_csv, index=False, encoding="utf-8-sig")

print(f"\n✅ 已保存至: {output_csv}")
print(df.head())
