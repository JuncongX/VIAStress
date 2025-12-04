import numpy as np

file_path = "ADARP_clip30s_multi_part1.npy"
data = np.load(file_path, allow_pickle=True)

person = "Part 102C"

# data[(data[:, 0] == person) & (data[:, 1] == label)]
s_data = data[(data[:, 0] == person)]
print(len(s_data))
