import numpy as np

# 加载原始文件
data = np.load("ADARP_clip30s_multi_undersampling.npy", allow_pickle=True)

# 使用 np.savez_compressed 保存，避免 pickle
np.savez_compressed("ADARP_clip30s_multi_undersampling_fixed.npz", data=data)

print("✅ 已保存为 ADARP_clip30s_multi_undersampling_fixed.npz")