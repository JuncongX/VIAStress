# Orphanidou C, Bonnici T, Charlton P, Clifton D, Vallance D, Tarassenko L. Signal-quality indices for the electrocardiogram and photoplethysmogram: derivation and applications to wireless monitoring. IEEE J Biomed Health Inform. 2015 May;19(3):832-8. doi: 10.1109/JBHI.2014.2338351. Epub 2014 Jul 23. PMID: 25069129.
import neurokit2 as nk
import numpy as np
from scipy import signal
import pandas as pd


# dataset = "ubfc_phys"
dataset = "ubfc_phys"
task = 3

if dataset == "ubfc_phys":
    data = np.load('UBFC_Phys_clip30s_multi_peak808.npy', allow_pickle=True)
    w_label = [1, task]
else:
    data = np.load('WESAD_clip30s_multi_peak808.npy', allow_pickle=True)
    w_label = [0, 1, 2]

needed_data = None
for label in w_label:
    selected_data = data[(data[:, 1] == label)]
    if needed_data is None:
        needed_data = selected_data
    else:
        needed_data = np.concatenate((needed_data, selected_data))

total_q = []

signals = [s for s in needed_data[:, 2]]
for s in signals:
    sqi_flags = nk.ppg_quality(s, sampling_rate=64, method="disimilarity")
    total_q.append(np.abs(sqi_flags).mean())
print(np.mean(total_q))

# wesad 0.005308848542414984
# ubfc t2 0.005167349911340246
# ubfc t3 0.0053644650077761275