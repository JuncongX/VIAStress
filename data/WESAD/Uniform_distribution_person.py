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
# splits = KFold(n_splits=5, shuffle=True, random_state=123)
# foldperf = {}
# all_persons = []
# for s_i in [2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 13, 14, 15, 16, 17]:
#     all_persons.append(r"S{0}".format(s_i))
# person_list = np.array(all_persons)
# fine_tune_final_list = []
# for fold, (train_idx, val_idx) in enumerate(splits.split(np.arange(len(person_list)))):
#     train_p = person_list[train_idx]
#     val_p = person_list[val_idx]
#
#     train_p, test_p, _, _ = train_test_split(train_p, [1 for i in range(len(train_p))], test_size=1 / (10 - 1))
#
#     train_subject.append(train_p.tolist())
#     valid_subject.append(val_p.tolist())
#     test_subject.append(test_p.tolist())
#
# print(train_subject)
# print(valid_subject)
# print(test_subject)

train_subject = [
    ['S6', 'S2', 'S7', 'S14', 'S11', 'S16', 'S5', 'S3', 'S8', 'S17', 'S4'],
    ['S8', 'S3', 'S9', 'S14', 'S13', 'S16', 'S7', 'S4', 'S10', 'S17', 'S5'],
    ['S6', 'S2', 'S8', 'S14', 'S13', 'S16', 'S5', 'S3', 'S9', 'S17', 'S4'],
    ['S6', 'S2', 'S7', 'S13', 'S11', 'S16', 'S5', 'S3', 'S8', 'S17', 'S4'],
    ['S8', 'S2', 'S9', 'S14', 'S13', 'S16', 'S7', 'S4', 'S10', 'S17', 'S6'],
    ['S6', 'S2', 'S7', 'S13', 'S11', 'S15', 'S5', 'S3', 'S9', 'S16', 'S4', 'S17'],
    ['S6', 'S2', 'S7', 'S11', 'S10', 'S14', 'S5', 'S3', 'S8', 'S16', 'S4', 'S17'],
    ['S7', 'S2', 'S8', 'S13', 'S11', 'S15', 'S6', 'S3', 'S9', 'S16', 'S5', 'S17'],
    ['S6', 'S2', 'S7', 'S11', 'S10', 'S14', 'S5', 'S3', 'S8', 'S15', 'S4', 'S17'],
    ['S6', 'S2', 'S7', 'S11', 'S10', 'S14', 'S5', 'S3', 'S8', 'S15', 'S4', 'S16']
]

valid_subject = [['S9', 'S13'], ['S2', 'S6'], ['S7', 'S11'], ['S10', 'S14'], ['S3', 'S5'], ['S8'], ['S15'], ['S4'],
                 ['S16'], ['S17']]

test_subject = [['S10', 'S15'], ['S11', 'S15'], ['S10', 'S15'], ['S9', 'S15'], ['S11', 'S15'], ['S10', 'S14'],
                ['S9', 'S13'], ['S10', 'S14'], ['S9', 'S13'], ['S9', 'S13']]
