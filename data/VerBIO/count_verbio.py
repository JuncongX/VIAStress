import numpy as np
import pandas as pd

# 加载数据
data = np.load("VerBIO_clip30s_multi.npy", allow_pickle=True)

# 转换为DataFrame方便统计
df = pd.DataFrame(data, columns=[
    "subject", "label", "ppg_signal", "scl_signal", "scr_signal", "eda_signal", "ppg_peak"
])

# 统计每个被试的标签数量分布
distribution = (
    df.groupby(["subject", "label"])
    .size()
    .unstack(fill_value=0)
    .rename(columns={0: "RELAX_count", 1: "PPT_count"})
)

# 添加总数列
distribution["total"] = distribution["RELAX_count"] + distribution["PPT_count"]

print(distribution)
print("\n总计：")
print(distribution.sum())

# 计算平均每个人的样本数
average_samples_per_subject = distribution["total"].mean()
print(f"\n平均每个受试者的样本数: {average_samples_per_subject:.2f}")