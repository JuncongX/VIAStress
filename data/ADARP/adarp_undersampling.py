import numpy as np
import glob
from imblearn.under_sampling import RandomUnderSampler
from collections import Counter

# -----------------------
# 1. 找到所有匹配文件
# -----------------------
file_list = sorted(glob.glob("ADARP_clip30s_multi_*C.npy"))
print(f"检测到 {len(file_list)} 个文件")

all_resampled = []

for file in file_list:
    print(f"\n正在处理文件: {file}")

    # -----------------------
    # 2. 读取数据
    # -----------------------
    data = np.load(file, allow_pickle=True)

    # 确认数据形状
    print("数据形状:", data.shape)

    # 第2列是标签 (label)
    y = data[:, 1].astype(int)
    X = data  # 整个样本，包括所有列

    print("原始类别分布:", Counter(y))

    # -----------------------
    # 3. 多数类下采样
    # -----------------------
    rus = RandomUnderSampler(random_state=42)
    X_res, y_res = rus.fit_resample(X, y)

    print("下采样后类别分布:", Counter(y_res))

    # -----------------------
    # 4. 累积结果
    # -----------------------
    all_resampled.append(X_res)

# -----------------------
# 5. 合并所有结果
# -----------------------
final_data = np.vstack(all_resampled)
print("\n✅ 合并完成，总样本数:", final_data.shape[0])
print("总体类别分布:", Counter(final_data[:, 1]))

# -----------------------
# 6. 保存结果
# -----------------------
np.save("ADARP_clip30s_multi_undersampling.npy", final_data)
print("✅ 已保存为 ADARP_clip30s_multi_undersampling.npy")
