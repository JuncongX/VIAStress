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
# splits = KFold(n_splits=7, shuffle=True, random_state=123)
# foldperf = {}
all_persons = ['s1', 's2', 's3', 's4', 's5', 's6', 's7', 's8', 's9', 's10', 's11', 's13', 's14', 's15',
               's16', 's18', 's19', 's20', 's21', 's22', 's23', 's24', 's25', 's26', 's27', 's28', 's29',
               's30', 's31', 's32', 's33', 's34', 's35', 's36', 's37', 's38', 's39', 's40', 's41', 's42', 's43',
               's44', 's45', 's46', 's48', 's49', 's50', 's51', 's52', 's53', 's54', 's55', 's56', 's12', 's17', 's47']
# person_list = np.array(all_persons)
# # 初始化 KFold 对象，设定折数为 7
# kf = KFold(n_splits=7, shuffle=True, random_state=42)
# 
# # 将参与者转化为 numpy 数组
# all_persons_array = np.array(all_persons)
# 
# # 存储结果
# train_val_test_splits = []
# 
# for train_val_index, test_index in kf.split(all_persons_array):
#     # 从训练集和验证集中再进行 6:1 的划分
#     train_val_split = KFold(n_splits=7 - 1, shuffle=True, random_state=42)
#     train_index, val_index = next(train_val_split.split(all_persons_array[train_val_index]))
# 
#     # 获取各个数据集的名称
#     train_set = all_persons_array[train_val_index][train_index]
#     val_set = all_persons_array[train_val_index][val_index]
#     test_set = all_persons_array[test_index]
# 
#     train_subject.append(train_set.tolist())
#     valid_subject.append(val_set.tolist())
#     test_subject.append(test_set.tolist())
# 
# print(train_subject)
# print(valid_subject)
# print(test_subject)

train_subject = [
    [
        's1', 's2', 's3', 's4', 's5', 's6', 's7', 's8',
        's9', 's10', 's11', 's12', 's13', 's14', 's15', 's16',
        's17', 's18', 's19', 's20', 's21', 's22', 's23', 's24',
        's25', 's26', 's27', 's28', 's29', 's30', 's31', 's32',
        's33', 's34', 's35', 's36', 's37', 's38', 's39', 's40',
    ], [
        's1', 's2', 's3', 's4', 's5', 's6', 's7', 's8',
        's9', 's10', 's11', 's12', 's13', 's14', 's15', 's16',
        's17', 's18', 's19', 's20', 's21', 's22', 's23', 's24',
        's25', 's26', 's27', 's28', 's29', 's30', 's31', 's32',
        's49', 's50', 's51', 's52', 's53', 's54', 's55', 's56'
    ], [
        's1', 's2', 's3', 's4', 's5', 's6', 's7', 's8',
        's9', 's10', 's11', 's12', 's13', 's14', 's15', 's16',
        's17', 's18', 's19', 's20', 's21', 's22', 's23', 's24',
        's41', 's42', 's43', 's44', 's45', 's46', 's47', 's48',
        's49', 's50', 's51', 's52', 's53', 's54', 's55', 's56'
    ], [
        's1', 's2', 's3', 's4', 's5', 's6', 's7', 's8',
        's9', 's10', 's11', 's12', 's13', 's14', 's15', 's16',
        's33', 's34', 's35', 's36', 's37', 's38', 's39', 's40',
        's41', 's42', 's43', 's44', 's45', 's46', 's47', 's48',
        's49', 's50', 's51', 's52', 's53', 's54', 's55', 's56'
    ], [
        's1', 's2', 's3', 's4', 's5', 's6', 's7', 's8',
        's25', 's26', 's27', 's28', 's29', 's30', 's31', 's32',
        's33', 's34', 's35', 's36', 's37', 's38', 's39', 's40',
        's41', 's42', 's43', 's44', 's45', 's46', 's47', 's48',
        's49', 's50', 's51', 's52', 's53', 's54', 's55', 's56'
    ], [
        's17', 's18', 's19', 's20', 's21', 's22', 's23', 's24',
        's25', 's26', 's27', 's28', 's29', 's30', 's31', 's32',
        's33', 's34', 's35', 's36', 's37', 's38', 's39', 's40',
        's41', 's42', 's43', 's44', 's45', 's46', 's47', 's48',
        's49', 's50', 's51', 's52', 's53', 's54', 's55', 's56'
    ], [
        's9', 's10', 's11', 's12', 's13', 's14', 's15', 's16',
        's17', 's18', 's19', 's20', 's21', 's22', 's23', 's24',
        's25', 's26', 's27', 's28', 's29', 's30', 's31', 's32',
        's33', 's34', 's35', 's36', 's37', 's38', 's39', 's40',
        's41', 's42', 's43', 's44', 's45', 's46', 's47', 's48',
    ],

]
valid_subject = [
    ['s41', 's42', 's43', 's44', 's45', 's46', 's47', 's48'],
    ['s33', 's34', 's35', 's36', 's37', 's38', 's39', 's40'],
    ['s25', 's26', 's27', 's28', 's29', 's30', 's31', 's32'],
    ['s17', 's18', 's19', 's20', 's21', 's22', 's23', 's24'],
    ['s9', 's10', 's11', 's12', 's13', 's14', 's15', 's16'],
    ['s1', 's2', 's3', 's4', 's5', 's6', 's7', 's8'],
    ['s49', 's50', 's51', 's52', 's53', 's54', 's55', 's56']
]
test_subject = [
    ['s49', 's50', 's51', 's52', 's53', 's54', 's55', 's56'],
    ['s41', 's42', 's43', 's44', 's45', 's46', 's47', 's48'],
    ['s33', 's34', 's35', 's36', 's37', 's38', 's39', 's40'],
    ['s25', 's26', 's27', 's28', 's29', 's30', 's31', 's32'],
    ['s17', 's18', 's19', 's20', 's21', 's22', 's23', 's24'],
    ['s9', 's10', 's11', 's12', 's13', 's14', 's15', 's16'],
    ['s1', 's2', 's3', 's4', 's5', 's6', 's7', 's8', ]
]
