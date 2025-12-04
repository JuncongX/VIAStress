import pickle
import neurokit2 as nk
s_path = r"E:\dataset\WESAD\{0}\{0}.pkl".format("S2")
with open(s_path, 'rb') as file:
    s_data = pickle.load(file, encoding='latin1')
w_eda = s_data['signal']['wrist']['EDA'][:, 0]  # 4Hz
eda, eda_info = nk.eda_process(w_eda, sampling_rate=4)

print(eda.keys())

# nk.eda_plot(eda, eda_info)
scr = eda["EDA_Phasic"]  # 皮肤电反应
scl = eda["EDA_Tonic"]  # 皮肤电水平
eda_s = eda["EDA_Clean"]


import matplotlib.pyplot as pyplot
pyplot.plot(range(len(eda_s)), eda_s)
pyplot.plot(range(len(scr)), scr)
pyplot.plot(range(len(scl)), scl)
pyplot.show()