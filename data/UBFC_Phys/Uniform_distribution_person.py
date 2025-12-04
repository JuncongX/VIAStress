# from UBFC_Phys_Dataset_npy_2023_limit_ import rPPG_Dataset, data_selected
# from sklearn.model_selection import KFold, train_test_split
# import numpy as np
#
# train_subject = []
# valid_subject = []
# test_subject = []
# train_task = []
# valid_task = []
# test_task = []
#
# splits = KFold(n_splits=10, shuffle=True, random_state=123)
# foldperf = {}
# person_list, tasks_list = data_selected()
# person_list, tasks_list = np.array(person_list), np.array(tasks_list)
# fine_tune_final_list = []
# for fold, (train_idx, val_idx) in enumerate(splits.split(np.arange(len(person_list)))):
#     train_p, train_t = person_list[train_idx], tasks_list[train_idx]
#     val_p, val_t = person_list[val_idx], tasks_list[val_idx]
#
#     train_p_t, test_p_t, _, _ = train_test_split(np.vstack((train_p, train_t)).transpose(1, 0),
#                                                  [1 for i in range(len(train_p))],
#                                                  test_size=1 / (10 - 1), random_state=123)
#     train_p, train_t = train_p_t[:, 0], [int(t) for t in train_p_t[:, 1]]
#     test_p, test_t = test_p_t[:, 0], [int(t) for t in test_p_t[:, 1]]
#
#     train_subject.append(train_p.tolist())
#     valid_subject.append(val_p.tolist())
#     test_subject.append(test_p.tolist())
#
#     train_task.append(train_t)
#     valid_task.append(val_t.tolist())
#     test_task.append(test_t)
#
# print(train_subject)
# print(valid_subject)
# print(test_subject)
# print(train_task)
# print(valid_task)
# print(test_task)

train_subject = [
    ['s24', 's10', 's18', 's36', 's1', 's43', 's37', 's42', 's51', 's20', 's27', 's29', 's39', 's23', 's44', 's6',
     's19', 's55', 's34', 's16', 's41', 's15', 's21', 's5'],
    ['s15', 's10', 's36', 's29', 's1', 's46', 's37', 's42', 's51', 's20', 's27', 's44', 's4', 's23', 's2', 's6', 's19',
     's39', 's18', 's16', 's50', 's43', 's21', 's5'],
    ['s41', 's15', 's10', 's29', 's55', 's44', 's20', 's37', 's42', 's2', 's23', 's5', 's18', 's45', 's27', 's7', 's6',
     's19', 's4', 's34', 's16', 's50', 's43', 's21', 's1'],
    ['s4', 's15', 's1', 's36', 's10', 's44', 's20', 's50', 's11', 's2', 's51', 's27', 's18', 's39', 's23', 's7', 's6',
     's19', 's55', 's34', 's16', 's41', 's43', 's21', 's5'],
    ['s41', 's15', 's55', 's29', 's39', 's44', 's34', 's37', 's42', 's2', 's51', 's5', 's36', 's45', 's23', 's7', 's6',
     's19', 's4', 's18', 's16', 's50', 's43', 's21', 's10'],
    ['s41', 's15', 's10', 's36', 's55', 's29', 's20', 's37', 's42', 's2', 's51', 's27', 's18', 's4', 's23', 's7', 's6',
     's19', 's39', 's34', 's16', 's50', 's43', 's21', 's1'],
    ['s4', 's15', 's1', 's29', 's10', 's44', 's20', 's50', 's11', 's2', 's51', 's27', 's36', 's39', 's23', 's7', 's21',
     's42', 's55', 's34', 's16', 's41', 's43', 's37', 's5'],
    ['s41', 's15', 's10', 's29', 's55', 's44', 's34', 's37', 's42', 's2', 's51', 's5', 's36', 's45', 's27', 's7', 's6',
     's19', 's39', 's18', 's20', 's50', 's43', 's21', 's1'],
    ['s45', 's7', 's10', 's18', 's55', 's36', 's16', 's37', 's42', 's29', 's23', 's5', 's34', 's4', 's27', 's44', 's6',
     's19', 's39', 's20', 's51', 's38', 's43', 's21', 's1'],
    ['s45', 's24', 's1', 's36', 's55', 's29', 's20', 's11', 's37', 's44', 's51', 's27', 's18', 's4', 's23', 's2', 's6',
     's19', 's39', 's34', 's16', 's38', 's15', 's42', 's5']
]

valid_subject = [['s11', 's38', 's2', 's46'], ['s41', 's55', 's34', 's7'], ['s39', 's51', 's36'], ['s42', 's37', 's29'],
                 ['s1', 's27', 's20'], ['s45', 's5', 's44'], ['s6', 's19', 's18'], ['s4', 's23', 's16'],
                 ['s50', 's24', 's15'], ['s21', 's10', 's43']]

test_subject = [['s7', 's50', 's45', 's4'], ['s24', 's11', 's38', 's45'], ['s38', 's46', 's11', 's24'],
                ['s45', 's46', 's38', 's24'], ['s38', 's46', 's11', 's24'], ['s38', 's46', 's11', 's24'],
                ['s45', 's46', 's38', 's24'], ['s38', 's46', 's11', 's24'], ['s41', 's46', 's11', 's2'],
                ['s41', 's46', 's50', 's7']]

train_task = [[3, 2, 3, 3, 2, 3, 2, 2, 3, 3, 2, 3, 2, 3, 3, 2, 2, 2, 3, 3, 2, 3, 2, 2],
              [3, 2, 3, 3, 2, 3, 2, 2, 3, 3, 2, 3, 2, 3, 3, 2, 2, 2, 3, 3, 2, 3, 2, 2],
              [2, 3, 2, 3, 2, 3, 3, 2, 2, 3, 3, 2, 3, 2, 2, 3, 2, 2, 2, 3, 3, 2, 3, 2, 2],
              [2, 3, 2, 3, 2, 3, 3, 2, 2, 3, 3, 2, 3, 2, 3, 3, 2, 2, 2, 3, 3, 2, 3, 2, 2],
              [2, 3, 2, 3, 2, 3, 3, 2, 2, 3, 3, 2, 3, 2, 3, 3, 2, 2, 2, 3, 3, 2, 3, 2, 2],
              [2, 3, 2, 3, 2, 3, 3, 2, 2, 3, 3, 2, 3, 2, 3, 3, 2, 2, 2, 3, 3, 2, 3, 2, 2],
              [2, 3, 2, 3, 2, 3, 3, 2, 2, 3, 3, 2, 3, 2, 3, 3, 2, 2, 2, 3, 3, 2, 3, 2, 2],
              [2, 3, 2, 3, 2, 3, 3, 2, 2, 3, 3, 2, 3, 2, 2, 3, 2, 2, 2, 3, 3, 2, 3, 2, 2],
              [2, 3, 2, 3, 2, 3, 3, 2, 2, 3, 3, 2, 3, 2, 2, 3, 2, 2, 2, 3, 3, 2, 3, 2, 2],
              [2, 3, 2, 3, 2, 3, 3, 2, 2, 3, 3, 2, 3, 2, 3, 3, 2, 2, 2, 3, 3, 2, 3, 2, 2]]

valid_task = [[2, 2, 3, 3], [2, 2, 3, 3], [2, 3, 3], [2, 2, 3], [2, 2, 3], [2, 2, 3], [2, 2, 3], [2, 3, 3], [2, 3, 3],
              [2, 2, 3]]

test_task = [[3, 2, 2, 2], [3, 2, 2, 2], [2, 3, 2, 3], [2, 3, 2, 3], [2, 3, 2, 3], [2, 3, 2, 3], [2, 3, 2, 3],
             [2, 3, 2, 3], [2, 3, 2, 3], [2, 3, 2, 3]]
