import numpy as np
import pandas as pd

# 加载数据
data = np.load("UBFC_Phys_clip30s_multi_peak808_.npy", allow_pickle=True)

# 数据结构: [subject, label, ppg_signal, scl_signal, scr_signal, eda_signal]
df = pd.DataFrame(data, columns=[
    "subject", "label", "ppg_signal", "scl_signal", "scr_signal", "eda_signal", "peak"
])

# 统计每个被试的任务(label)数量分布
distribution = (
    df.groupby(["subject", "label"])
    .size()
    .unstack(fill_value=0)
)

# 为列加上任务名称
distribution = distribution.rename(columns={1: "Task1_count", 2: "Task2_count", 3: "Task3_count"})

# 添加总数列
distribution["total"] = distribution.sum(axis=1)

# 按总数排序
distribution_sorted = distribution.sort_values(by="total", ascending=False)

print("每个受试者各任务样本数量分布：\n")
print(distribution_sorted)

print("\n所有样本总计：")
print(distribution_sorted[["Task1_count", "Task2_count", "Task3_count", "total"]].sum())