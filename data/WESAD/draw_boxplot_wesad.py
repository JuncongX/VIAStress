import pandas as pd
import ast
import matplotlib.pyplot as plt
import seaborn as sns

# ===============================
# 读取 CSV
# ===============================
output_csv = r"wesad_ppg_quality_per_person.csv"
df = pd.read_csv(output_csv)

# 将字符串形式的分数列表解析为真正的list
df["Scores"] = df["Scores"].apply(ast.literal_eval)

# ===============================
# 计算每个个体的平均分数
# ===============================
df["Mean_Score"] = df["Scores"].apply(lambda x: sum(x)/len(x) if len(x) > 0 else None)

# 只保留有平均分的行
mean_scores = df["Mean_Score"].dropna()

print("每个被试的平均PPG质量分数：")
print(df[["Person", "Mean_Score"]])

# ===============================
# 绘制整体箱型图
# ===============================
plt.figure(figsize=(6, 6))
sns.boxplot(y=mean_scores, color="skyblue")
sns.stripplot(y=mean_scores, color="red", size=6, jitter=True)
plt.title("Distribution of Mean PPG Quality Scores Across All Subjects", fontsize=14)
plt.ylabel("Mean PPG Quality Score")
plt.xticks([])  # 不显示x轴刻度，因为只有一个箱型
plt.tight_layout()
plt.show()


#    Person  Mean_Score
# 0      S2    0.710355
# 1      S3    0.663676
# 2      S4    0.758324
# 3      S5    0.705434
# 4      S6    0.699457
# 5      S7    0.760940
# 6      S8    0.772080
# 7      S9    0.761285
# 8     S10    0.790332
# 9     S11    0.709090
# 10    S13    0.744973
# 11    S14    0.798481
# 12    S15    0.653105
# 13    S16    0.697722
# 14    S17    0.757107
