import numpy as np
from collections import defaultdict, Counter
import csv

# ===== Step 1. 读取 NPY 文件 =====
npy_path = "CAN_STRESS_clip30s_multi.npy"
data = np.load(npy_path, allow_pickle=True)

# ===== Step 2. 统计每个 participant 的样本数量和标签 =====
participant_samples = defaultdict(list)

for entry in data:
    participant_id = entry[0]
    stress_label = entry[1]
    participant_samples[participant_id].append(stress_label)

# ===== Step 3. 统计 0-1 和 2-9 的数量 =====
stats_list = []
for pid, labels in participant_samples.items():
    count_0_1 = sum(1 for l in labels if l in [0, 1, 2, 3, 4])
    count_2_9 = sum(1 for l in labels if 5 <= l <= 9)
    stats_list.append({
        "Participant_ID": pid,
        "Count_0_1": count_0_1,
        "Count_2_9": count_2_9,
        "Total": len(labels)
    })

# ===== Step 4. 保存为 CSV =====
csv_path = "participant_stress_stats.csv"
with open(csv_path, mode="w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=["Participant_ID", "Count_0_1", "Count_2_9", "Total"])
    writer.writeheader()
    writer.writerows(stats_list)

print(f"统计结果已保存到 {csv_path}")