import pandas as pd
import ast
import matplotlib.pyplot as plt
import seaborn as sns

# ===============================
# 读取 CSV 路径
# ===============================
wesad_csv = r"WESAD/wesad_ppg_quality_per_person.csv"
ubfc_csv = r"UBFC_Phys/ubfc_phys_ppg_quality_per_person.csv"
verbio_csv = r"VerBIO/VerBIO_ppg_quality_per_person.csv"
merged_csv = r"ADARP/ppg_quality_scores_merged.csv"
can_stress_csv = r"CAN_Stress/CAN_Stress_quality.csv"

# ===============================
# 读取数据
# ===============================
df_wesad = pd.read_csv(wesad_csv, encoding="utf-8-sig")
df_ubfc = pd.read_csv(ubfc_csv, encoding="utf-8-sig")
df_verbio = pd.read_csv(verbio_csv, encoding="utf-8-sig")
df_merged = pd.read_csv(merged_csv, encoding="utf-8-sig")
df_can = pd.read_csv(can_stress_csv, encoding="utf-8-sig")

# ===============================
# 将 Scores 字符串解析为列表
# ===============================
df_ubfc["Scores"] = df_ubfc["Scores"].apply(ast.literal_eval)
df_wesad["Scores"] = df_wesad["Scores"].apply(ast.literal_eval)
df_verbio["Scores"] = df_verbio["Scores"].apply(ast.literal_eval)

# ===============================
# 计算每个被试平均质量分
# ===============================
df_ubfc["Mean_Score"] = df_ubfc["Scores"].apply(lambda x: sum(x) / len(x) if len(x) else None)
df_wesad["Mean_Score"] = df_wesad["Scores"].apply(lambda x: sum(x) / len(x) if len(x) else None)
df_verbio["Mean_Score"] = df_verbio["Scores"].apply(lambda x: sum(x) / len(x) if len(x) else None)

# 其他数据集已经是单分数列
ubfc_scores = df_ubfc["Mean_Score"].dropna()
wesad_scores = df_wesad["Mean_Score"].dropna()
verbio_scores = df_verbio["Mean_Score"].dropna()
merged_scores = df_merged["Mean_Quality"].dropna()
can_scores = df_can["quality_score"].dropna()

# ===============================
# 统一 DataFrame 用于绘图
# ===============================
df_plot = pd.DataFrame({
    "Dataset": (
            ["UBFC-Phys"] * len(ubfc_scores) +
            ["WESAD"] * len(wesad_scores) +
            ["VerBIO"] * len(verbio_scores) +
            ["CAN-Stress"] * len(can_scores) +
            ["ADARP"] * len(merged_scores)
    ),
    "Mean_Score": (
            list(ubfc_scores) +
            list(wesad_scores) +
            list(verbio_scores) +
            list(can_scores) +
            list(merged_scores)
    )
})

# ===============================
# 绘制箱型图 + 散点
# ===============================
plt.figure(figsize=(12, 6))
sns.boxplot(
    x="Dataset",
    y="Mean_Score",
    data=df_plot,
    palette=["skyblue", "lightgreen", "orange", "lightcoral", "violet"]
)
sns.stripplot(
    x="Dataset",
    y="Mean_Score",
    data=df_plot,
    color="black",
    size=5,
    jitter=True,
    alpha=0.6
)

plt.title("Comparison of Mean PPG Quality Scores Across Datasets", fontsize=14)
plt.ylabel("Mean PPG Quality Score")
plt.xlabel("")
plt.tight_layout()
plt.show()
