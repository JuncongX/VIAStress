import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import ast
import os

# 读取CSV
data_folder = r"/home/xjc/data/ADARP/Sensor Data/"
csv_path = os.path.join(data_folder, "ppg_quality_scores.csv")

df = pd.read_csv(csv_path)

# 将字符串形式的列表转为真正的Python列表
df["Scores"] = df["Scores"].apply(lambda x: ast.literal_eval(x) if isinstance(x, str) else [])

# 提取参与者与得分
participants = df["Participant_ID"].tolist()
scores = [np.ravel(s) for s in df["Scores"].tolist()]

# 合并所有参与者的得分，作为整体统计
all_scores = np.concatenate(scores)

# -------------------------
# 绘制每个个体的箱型图
# -------------------------
plt.figure(figsize=(12, 6))
plt.boxplot(scores, labels=participants, patch_artist=True)
plt.title("PPG Quality Scores per Participant", fontsize=14)
plt.xlabel("Participant ID", fontsize=12)
plt.ylabel("PPG Quality Score (Dissimilarity)", fontsize=12)
plt.xticks(rotation=45)
plt.grid(True, linestyle="--", alpha=0.6)
plt.tight_layout()
plt.show()

# -------------------------
# 绘制整体箱型图
# -------------------------
plt.figure(figsize=(5, 6))
plt.boxplot(all_scores, patch_artist=True)
plt.title("Overall PPG Quality Distribution", fontsize=14)
plt.ylabel("PPG Quality Score (Dissimilarity)", fontsize=12)
plt.grid(True, linestyle="--", alpha=0.6)
plt.tight_layout()
plt.show()
