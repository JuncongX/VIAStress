import numpy as np
from ClassicML.feature_extraction import get_plf


# data = np.load('UBFC_Phys_clip30s_multi_peak808_.npy', allow_pickle=True)
data = np.load('../data/VerBIO/VerBIO_clip30s_multi.npy', allow_pickle=True)

save_datas = []

for participant_id, label, ppg_signal, scl_signal, scr_signal, eda_signal, ppg_peak in data:
    print(participant_id, label)
    try:
        feature = get_plf(eda_signal, ppg_signal)
    except Exception as e:
        print(e)
        continue
    datas = [participant_id, label, feature]
    save_datas.append(datas)

np.save("VerBIO_CLM_clip30s.npy", save_datas, allow_pickle=True, fix_imports=True)
