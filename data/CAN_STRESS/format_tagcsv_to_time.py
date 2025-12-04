import pandas as pd
from datetime import datetime
import pytz
# CSV 文件路径
file_path = r'F:\science\secondpaper\TBME大修\审稿人提出的参考文章与数据集\CAN-STEESS\CAN-STRESS\CAN-STRESS\A02160_220304-160223__1463075\tags.csv'

# 读取 CSV
try:
    df = pd.read_csv(file_path)
except FileNotFoundError:
    print(f"文件未找到，请检查路径: {file_path}")
    exit()
except Exception as e:
    print(f"读取 CSV 时出错: {e}")
    exit()

# 给时间戳列命名（假设 CSV 只有一列）
df.columns = ['timestamp']

# 转换时间戳为美国东部时间
try:
    df['ET_time'] = (
        pd.to_datetime(df['timestamp'], unit='s')  # 转为 datetime
        .dt.tz_localize('UTC')                     # 先设为 UTC
        .dt.tz_convert('US/Eastern')              # 转为美国东部时间
    )
except Exception as e:
    print(f"转换时间戳时出错: {e}")
    exit()

# # 打印结果
# for ts, et in zip(df['timestamp'], df['ET_time']):
#     print(f"{ts} -> {et.strftime('%Y-%m-%d %H:%M:%S %Z')}")

# 计算时间差（相邻时间戳的差值，单位为秒/分钟/小时可选）
df['time_diff'] = df['ET_time'].diff()  # 得到 timedelta 对象

# 打印结果
for i, row in df.iterrows():
    ts = row['timestamp']
    et = row['ET_time'].strftime('%Y-%m-%d %H:%M:%S %Z')
    diff = row['time_diff']
    if pd.isna(diff):
        print(f"{ts} -> {et} | 时间差: N/A")
    else:
        # 可以选择打印秒数、分钟或小时
        print(f"{ts} -> {et} | 时间差: {diff}")