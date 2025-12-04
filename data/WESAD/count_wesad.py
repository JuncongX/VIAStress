import numpy as np
import pandas as pd

# 加载数据
data = np.load("WESAD_clip30s_multi_peak808_.npy", allow_pickle=True)

# 数据结构: [subject, label, ppg_signal, scl_signal, scr_signal, eda_signal, ppg_peak]
df = pd.DataFrame(data, columns=[
    "subject", "label", "ppg_signal", "scl_signal", "scr_signal", "eda_signal", "ppg_peak"
])

# 统计每个受试者每类任务样本数量
distribution = (
    df.groupby(["subject", "label"])
    .size()
    .unstack(fill_value=0)
)

# 为列加上任务名称
distribution = distribution.rename(columns={0: "Baseline_count", 1: "Stress_count", 2: "Amusement_count"})

# 添加总样本数列
distribution["total"] = distribution.sum(axis=1)

# 按总数排序
distribution_sorted = distribution.sort_values(by="total", ascending=False)

print("每个受试者各任务样本数量分布：\n")
print(distribution_sorted)

# 统计总体样本数
total_counts = distribution_sorted[["Baseline_count", "Stress_count", "Amusement_count", "total"]].sum()
print("\n所有样本总计：")
print(total_counts)

# 平均每个受试者的样本数
average_samples_per_subject = distribution_sorted["total"].mean()
print(f"\n平均每个受试者的样本数: {average_samples_per_subject:.2f}")
