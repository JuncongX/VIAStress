from datetime import datetime
import pytz

timestamp = 1666817287

local_time = datetime.fromtimestamp(timestamp)
print("local time:", local_time)

utc_time = datetime.utcfromtimestamp(timestamp)
print("UTC:", utc_time)

# 美国东部时间
eastern = pytz.timezone('US/Eastern')
et_time = datetime.fromtimestamp(timestamp, tz=eastern)
print("US/Eastern time:", et_time)
