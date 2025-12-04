import neurokit2 as nk
import pandas as pd
import numpy as np
import os

# ===============================
# 参数配置
# ===============================
ppg_sample_rate = 64

data_root = r"/home/xjc/data/VerBIO_v2/"
pre_path = os.path.join(data_root, "PRE/E4")
post_path = os.path.join(data_root, "POST/E4")

subjects = [
    'P001', 'P003', 'P004', 'P005', 'P006', 'P007', 'P008', 'P009', 'P011', 'P012', 'P013',
    'P014', 'P016', 'P017', 'P018', 'P020', 'P021', 'P023', 'P026', 'P027', 'P031', 'P032',
    'P035', 'P037', 'P038', 'P039', 'P040', 'P041', 'P042', 'P043', 'P044', 'P045', 'P046',
    'P047', 'P048', 'P049', 'P050', 'P051', 'P052', 'P053', 'P056', 'P057', 'P058', 'P060',
    'P061', 'P062', 'P063', 'P064', 'P065', 'P066', 'P067', 'P068', 'P071', 'P072', 'P073'
]

output_csv = "VerBIO_ppg_quality_per_person.csv"


# ===============================
# 读取单个受试者所有原始 PPG
# ===============================
def load_ppg_for_subject(subj_path):
    """返回 relax_ppg, ppt_ppg，若缺失则返回 None"""
    relax_file = os.path.join(subj_path, "BVP_RELAX.csv")
    ppt_file = os.path.join(subj_path, "BVP_PPT.csv")

    if not (os.path.exists(relax_file) and os.path.exists(ppt_file)):
        print(f"[Missing] {subj_path}")
        return None, None

    try:
        # RELAX 格式：一列无列名
        relax_ppg = pd.read_csv(relax_file).to_numpy().squeeze()

        # PPT 格式：有列名 BVP
        ppt_ppg = pd.read_csv(ppt_file)["BVP"].to_numpy()

        return relax_ppg, ppt_ppg
    except Exception as e:
        print(f"[Error loading] {subj_path}: {e}")
        return None, None


# ===============================
# 质量计算（安全版）
# ===============================
def safe_ppg_quality(ppg, person, label):
    """计算 PPG 质量分数序列，带异常保护"""
    try:
        if ppg is None or len(ppg) < ppg_sample_rate * 2:
            print(f"[Skip] {person}-{label}: 信号过短或缺失")
            return np.array([])

        ppg = np.nan_to_num(ppg)

        ppg_cleaned = nk.ppg_clean(ppg, sampling_rate=ppg_sample_rate)
        q = nk.ppg_quality(ppg_cleaned, sampling_rate=ppg_sample_rate, method="templatematch")

        print(f"Person:{person} Label:{label} -> Quality length:{len(q)}")
        return np.asarray(q)

    except Exception as e:
        print(f"[Error] {person}-{label}: {e}")
        return np.array([])


# ===============================
# 主循环
# ===============================
results = []

for subj in subjects:
    person_scores = []

    for set_name, base_path in [("PRE", pre_path), ("POST", post_path)]:
        subj_path = os.path.join(base_path, subj)

        relax_ppg, ppt_ppg = load_ppg_for_subject(subj_path)
        if relax_ppg is None:
            continue

        # 0 = relax, 1 = PPT
        for label, raw_ppg in [(0, relax_ppg), (1, ppt_ppg)]:
            q_values = safe_ppg_quality(raw_ppg, subj, f"{set_name}_{label}")
            if len(q_values) > 0:
                person_scores.extend(q_values.tolist())

    # 保存每个受试者统计
    if len(person_scores) > 0:
        results.append({
            "Person": subj,
            "Num_Samples": len(person_scores),
            "Scores": person_scores
        })

# ===============================
# 保存到 CSV 文件
# ===============================
df = pd.DataFrame(results)
df.to_csv(output_csv, index=False, encoding="utf-8-sig")
print(f"\n✅ 已保存至: {output_csv}")
print(df.head())
