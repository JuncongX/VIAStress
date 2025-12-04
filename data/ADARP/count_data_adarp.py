import numpy as np
from collections import defaultdict

file_path = "ADARP_clip30s_multi_undersampling.npy"

# 普通加载（Python对象类型，不适用 mmap）
data = np.load(file_path, allow_pickle=True)

# 初始化字典来统计每个参与者的标签
participant_label_count = defaultdict(lambda: {"stress": 0, "non_stress": 0})

# 遍历每个数据条目
for entry in data:
    participant_ids, label, _, _, _, _, _ = entry
    # participant_ids 可能是字符串或列表，统一处理为字符串
    if isinstance(participant_ids, list):
        participant_ids = participant_ids[0]

    if label == 1:
        participant_label_count[participant_ids]["stress"] += 1
    else:
        participant_label_count[participant_ids]["non_stress"] += 1

# 输出统计结果
for participant, counts in participant_label_count.items():
    print(f"{participant}: Stress={counts['stress']}, Non-Stress={counts['non_stress']}")