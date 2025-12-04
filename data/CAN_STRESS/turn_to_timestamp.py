import os
import re
import pandas as pd
import datetime
import json

# ===== Step 1. 获取文件路径 =====
excel_path = "logbook.xlsx"  # Excel 文件名
base_dir = os.path.dirname(os.path.abspath(excel_path))  # ✅ 改进点

# ===== Step 2. 扫描文件夹以建立 E4_Sessions -> 日期映射 =====
session_date_map = {}
folder_pattern = re.compile(r"^[A-Za-z0-9]+_(\d{6})-\d{6}__([0-9]+)")

for name in os.listdir(base_dir):
    match = folder_pattern.match(name)
    if match:
        yymmdd, session_id = match.groups()
        date = datetime.datetime.strptime("20" + yymmdd, "%Y%m%d").date()
        session_date_map[str(int(session_id))] = date  # ✅ 强制转字符串整数

print("文件夹中可用的 session_id:", list(session_date_map.keys()))

# ===== Step 3. 读取Excel =====
df = pd.read_excel(excel_path, usecols=["ID", "E4_Sessions", "Time", "Stress_Rating", "Type"])

# 清洗空值与"No Recording"
df = df.dropna(subset=["ID", "E4_Sessions", "Time", "Type"], how="any")
df = df[df["Type"].astype(str).str.lower() != "no recording"]

# ===== Step 4. 生成UTC时间戳 =====
def make_timestamp(row):
    try:
        session_id = str(int(row["E4_Sessions"])).strip()
        if session_id not in session_date_map:
            print(f"⚠️ 未找到匹配日期: {session_id}")
            return None

        date = session_date_map[session_id]
        t = row["Time"]

        # 支持 Timestamp / datetime / time / 字符串
        if isinstance(t, (pd.Timestamp, datetime.datetime)):
            dt = datetime.datetime.combine(date, t.time())
        elif isinstance(t, datetime.time):
            dt = datetime.datetime.combine(date, t)
        else:
            t_str = str(t).strip()
            match = re.match(r"^(\d{1,2}):(\d{2})(?::\d{2})?$", t_str)
            if not match:
                print(f"⚠️ 无法识别时间格式: {t_str}")
                return None
            dt = datetime.datetime.strptime(f"{date} {match.group(1)}:{match.group(2)}", "%Y-%m-%d %H:%M")

        dt_utc = dt.replace(tzinfo=datetime.timezone.utc)
        return int(dt_utc.timestamp())
    except Exception as e:
        print("❌ 转换失败:", e, row)
        return None

df["timestamp"] = df.apply(make_timestamp, axis=1)
df = df.dropna(subset=["timestamp"])

# ===== Step 5. 按ID组织为dict =====
result = {}
for _, row in df.iterrows():
    session_id = int(row["E4_Sessions"])
    record = {
        "ID": int(row["ID"]),
        "timestamp": int(row["timestamp"]),
        "Stress_Rating": int(row["Stress_Rating"]) if not pd.isna(row["Stress_Rating"]) else None,
        "Type": str(row["Type"])
    }
    result.setdefault(session_id, []).append(record)

# ===== Step 6. 导出JSON =====
output_path = os.path.join(base_dir, "logbook_parsed_by_session.json")
with open(output_path, "w", encoding="utf-8") as f:
    json.dump(result, f, ensure_ascii=False, indent=2)

print("✅ 数据整理完成，结果已保存到:", output_path)
