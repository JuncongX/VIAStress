import pandas as pd
import glob
import os
import neurokit2 as nk
import numpy as np

# 设置根目录路径
base_path = "/home/xjc/data/CAN_Stress/"

# 搜索所有子文件夹中的 BVP.csv 文件
csv_files = glob.glob(os.path.join(base_path, "**/BVP.csv"), recursive=True)

# 打印找到的文件路径
print(f"找到 {len(csv_files)} 个 BVP.csv 文件：")

results = []

for file in csv_files:
    print(file)
    df = pd.read_csv(file)
    bvp = df.to_numpy().squeeze(1)[2:]
    q_s = nk.ppg_quality(bvp, sampling_rate=64, method='templatematch')
    q_s_mean = np.nanmean(q_s)
    print(q_s_mean)

    # 保存结果
    results.append({
        "file_path": file,
        "quality_score": q_s_mean
    })

# 转为 DataFrame
quality_df = pd.DataFrame(results)

# 保存为 CSV 文件
output_path = os.path.join(base_path, "CAN_Stress_quality.csv")
quality_df.to_csv(output_path, index=False)

print(f"质量评分结果已保存至：{output_path}")