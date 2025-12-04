import numpy as np
from ClassicML.feature_extraction import get_plf


# data = np.load('UBFC_Phys_clip30s_multi_peak808_.npy', allow_pickle=True)
data = np.load('data/CAN_STRESS/CAN_STRESS_clip30s_multi.npy', allow_pickle=True)

BINARY_LABEL_DICT = {
    0: 0,
    1: 0,
    2: 0,
    3: 0,
    4: 0,
    5: 1,
    6: 1,
    7: 1,
    8: 1,
    9: 1
}

save_datas = []

for participant_id, stress_rate, ppg_signal, scl_signal, scr_signal, eda_signal, ppg_peak in data:
    print(participant_id, stress_rate)
    feature = get_plf(eda_signal, ppg_signal)
    label = BINARY_LABEL_DICT[stress_rate]
    datas = [participant_id, label, feature]
    save_datas.append(datas)

np.save("CAN_STRESS_CLM_clip30s.npy", np.array(save_datas, dtype=object), allow_pickle=True)
