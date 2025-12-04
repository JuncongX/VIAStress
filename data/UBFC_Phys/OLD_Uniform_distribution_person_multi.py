from sklearn.model_selection import KFold, train_test_split
import numpy as np

train_subject = []
valid_subject = []
test_subject = []
train_task = []
valid_task = []
test_task = []

splits = KFold(n_splits=7, shuffle=True, random_state=123)
foldperf = {}
all_persons = ['s1', 's2', 's3', 's4', 's5', 's6', 's7', 's8', 's9', 's10', 's11', 's13', 's14', 's15',
               's16', 's18', 's19', 's20', 's21', 's22', 's23', 's24', 's25', 's26', 's27', 's28', 's29',
               's30', 's31', 's32', 's33', 's34', 's35', 's36', 's37', 's38', 's39', 's40', 's41', 's42', 's43',
               's44', 's45', 's46', 's48', 's49', 's50', 's51', 's52', 's53', 's54', 's55', 's56']
person_list = np.array(all_persons)
fine_tune_final_list = []
for fold, (train_idx, val_idx) in enumerate(splits.split(np.arange(len(person_list)))):
    train_p = person_list[train_idx]
    val_p = person_list[val_idx]

    train_p, test_p, _, _ = train_test_split(train_p, [1 for i in range(len(train_p))], test_size=1 / (10 - 1))

    train_subject.append(train_p.tolist())
    valid_subject.append(val_p.tolist())
    test_subject.append(test_p.tolist())

print(train_subject)
print(valid_subject)
print(test_subject)

train_subject = [
    ['s8', 's49', 's44', 's7', 's24', 's1', 's10', 's4', 's25', 's28', 's51', 's32', 's2', 's19', 's54', 's14', 's16',
     's18', 's35', 's55', 's37', 's42', 's5', 's45', 's38', 's43', 's11', 's56', 's21', 's22', 's31', 's40', 's20',
     's53', 's36', 's34', 's6', 's30', 's23', 's27'],
    ['s10', 's2', 's44', 's23', 's21', 's50', 's38', 's26', 's33', 's13', 's49', 's55', 's42', 's34', 's3', 's32',
     's31', 's48', 's54', 's29', 's25', 's5', 's52', 's19', 's22', 's35', 's8', 's45', 's20', 's41', 's15', 's46',
     's27', 's16', 's28', 's4', 's18', 's36', 's53', 's56'],
    ['s34', 's50', 's37', 's10', 's5', 's42', 's35', 's45', 's49', 's52', 's51', 's8', 's48', 's46', 's9', 's22', 's31',
     's27', 's25', 's39', 's28', 's36', 's13', 's24', 's20', 's40', 's6', 's11', 's4', 's33', 's41', 's14', 's56', 's3',
     's18', 's16', 's19', 's23', 's53', 's15'],
    ['s25', 's44', 's4', 's24', 's56', 's16', 's19', 's32', 's1', 's45', 's21', 's49', 's6', 's54', 's40', 's52', 's26',
     's5', 's31', 's50', 's43', 's42', 's41', 's2', 's51', 's36', 's11', 's9', 's48', 's3', 's22', 's7', 's38', 's14',
     's46', 's29', 's39', 's20', 's10', 's37'],
    ['s10', 's6', 's52', 's53', 's39', 's35', 's7', 's38', 's56', 's37', 's27', 's45', 's40', 's43', 's29', 's31',
     's26', 's33', 's32', 's1', 's21', 's18', 's24', 's41', 's30', 's2', 's48', 's3', 's15', 's11', 's20', 's55', 's14',
     's46', 's49', 's51', 's36', 's23', 's34', 's8'],
    ['s18', 's29', 's30', 's13', 's20', 's38', 's21', 's7', 's4', 's37', 's33', 's3', 's49', 's51', 's50', 's52', 's14',
     's55', 's34', 's42', 's22', 's44', 's5', 's48', 's23', 's32', 's11', 's15', 's28', 's6', 's8', 's2', 's53', 's1',
     's19', 's9', 's54', 's16', 's41', 's46'],
    ['s28', 's23', 's55', 's25', 's45', 's18', 's33', 's50', 's24', 's11', 's6', 's26', 's48', 's42', 's30', 's5',
     's36', 's8', 's52', 's2', 's27', 's29', 's53', 's9', 's44', 's46', 's51', 's38', 's54', 's10', 's34', 's16', 's13',
     's19', 's40', 's43', 's35', 's4', 's56', 's7']]
valid_subject = [['s13', 's15', 's26', 's29', 's39', 's46', 's48', 's52'],
                 ['s1', 's6', 's9', 's11', 's14', 's24', 's40', 's51'],
                 ['s2', 's7', 's21', 's30', 's32', 's38', 's44', 's54'],
                 ['s8', 's18', 's23', 's27', 's33', 's34', 's53', 's55'],
                 ['s4', 's5', 's16', 's19', 's28', 's42', 's50'], ['s10', 's25', 's35', 's36', 's43', 's45', 's56'],
                 ['s3', 's20', 's22', 's31', 's37', 's41', 's49']]
test_subject = [['s3', 's41', 's9', 's50', 's33'], ['s43', 's7', 's37', 's39', 's30'],
                ['s55', 's26', 's43', 's1', 's29'], ['s35', 's15', 's30', 's13', 's28'],
                ['s22', 's25', 's13', 's44', 's9', 's54'], ['s26', 's24', 's39', 's40', 's31', 's27'],
                ['s32', 's21', 's14', 's15', 's1', 's39']]
