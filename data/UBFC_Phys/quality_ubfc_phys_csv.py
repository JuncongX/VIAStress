import neurokit2 as nk
import pandas as pd
import numpy as np
import os
import pickle
import csv

# ===============================
# 参数配置
# ===============================
ppg_sample_rate = 64
all_persons = ['s1', 's2', 's3', 's4', 's5', 's6', 's7', 's8', 's9', 's10',
               's11', 's13', 's14', 's15', 's16', 's18', 's19', 's20', 's21',
               's22', 's23', 's24', 's25', 's26', 's27', 's28', 's29', 's30',
               's31', 's32', 's33', 's34', 's35', 's36', 's37', 's38', 's39',
               's40', 's41', 's42', 's43', 's44', 's45', 's46', 's48', 's49',
               's50', 's51', 's52', 's53', 's54', 's55', 's56']
data_root = r"/home/som/8T/DataSets/ubfc_phys/video/"
output_csv = r"ubfc_phys_ppg_quality_per_person.csv"

# ===============================
# 安全计算 PPG 质量函数
# ===============================
def safe_ppg_quality(ppg, person, label):
    """计算 PPG 质量分数序列，带异常保护"""
    try:
        if len(ppg) < ppg_sample_rate * 2:
            print(f"[Skip] {person}-{label}: 信号过短")
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
# 主循环：按被试和标签处理
# ===============================
results = []

for person in all_persons:
    person_path = os.path.join(data_root, person)
    if not os.path.exists(person_path):
        print(f"[Missing] {person_path}")
        continue

    person_scores = []

    for label in [1, 2, 3]:  # 三种状态
        bvp_file = os.path.join(person_path, f"bvp_{person}_T{label}.csv")
        if not os.path.exists(bvp_file):
            print(f"[Missing] {bvp_file}")
            continue

        bvp_signal = pd.read_csv(bvp_file, header=None).to_numpy().squeeze(-1)

        q_values = safe_ppg_quality(bvp_signal, person, label)
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
