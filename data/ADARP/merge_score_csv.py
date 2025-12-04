import pandas as pd
import glob
import os

# 匹配所有符合命名规则的文件
file_pattern = "ppg_quality_scores_1*C.csv"  # 例如 ppg_quality_scores_101C.csv, ppg_quality_scores_102C.csv
file_list = glob.glob(file_pattern)

# 读取并合并所有 CSV 文件
df_list = []
for file in file_list:
    temp_df = pd.read_csv(file)
    temp_df["Source_File"] = os.path.basename(file)  # 可选：添加来源文件名列
    df_list.append(temp_df)

# 合并为一个大 DataFrame
merged_df = pd.concat(df_list, ignore_index=True)

# 输出合并后的文件
merged_df.to_csv("ppg_quality_scores_merged.csv", index=False)

print(f"已合并 {len(file_list)} 个文件，输出为 ppg_quality_scores_merged.csv")
