import numpy as np

data = np.load("CAN_STRESS_CLM_clip30s.npy", allow_pickle=True)
np.save("CAN_STRESS_CLM_clip30s_v1.npy", data, allow_pickle=True, fix_imports=True)
