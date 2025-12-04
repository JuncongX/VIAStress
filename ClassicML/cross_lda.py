import sklearn
import argparse
from sklearn.preprocessing import QuantileTransformer, StandardScaler
from sklearn.metrics import confusion_matrix, f1_score, accuracy_score
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC
import numpy as np
import joblib

from ClassicML.load_data import DataLoder


def dataset_name(cd, position):
    if cd[position] == 'w':
        return 'wesad'
    elif cd[position] == 'u':
        return 'ubfc_phys'
    elif cd[position] == 'v':
        return 'verbio'


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--cd', type=str, default='u2v', choices=['w2u', 'u2w', 'w2v', 'v2w', 'u2v', 'v2u'])
    parser.add_argument('--k', type=int, default=7, help="KFold", choices=[5, 7])
    parser.add_argument('--ubfc_phys_task', type=int, default=2, choices=[2, 3])
    parser.add_argument('--y_dim', type=int, default=2)
    parser.add_argument('--context_num', type=int, default=0)
    args = parser.parse_args()

    data_loader = DataLoder(dataset_name(args.cd, -1), binary=True if args.y_dim == 2 else False,
                            task=args.ubfc_phys_task)
    acc_list = []
    f1_list = []
    for fold in range(args.k):

        model = joblib.load('save/LDA_{0}{1}_{2}_{3}.pkl'.format(
            dataset_name(args.cd, 0),
            "_" + str(args.ubfc_phys_task) if (args.cd[0] == 'u' and args.y_dim == 2) else "",
            args.y_dim,
            fold + 1
        ))
        persons, features, labels = data_loader.load(phase='cross', fold=0)

        del_index_choice = None

        for person in np.unique(persons):
            baseline_index = np.arange(len(labels))[(labels == 0) & (persons == person)]
            baseline_index_choice = np.random.choice(baseline_index, args.context_num, replace=False)
            if del_index_choice is None:
                del_index_choice = baseline_index_choice
            else:
                del_index_choice = np.concatenate((del_index_choice, baseline_index_choice))
        remaining_indices = np.delete(np.arange(len(labels)), del_index_choice)
        features = features[remaining_indices]
        labels = labels[remaining_indices]

        res = model.predict(features)
        macro_f1 = f1_score(labels, res, average='macro')
        accuracy = accuracy_score(labels, res)
        cm = confusion_matrix(labels, res)
        print(cm)
        print(100 * accuracy, macro_f1)
        acc_list.append(100 * accuracy)
        f1_list.append(macro_f1)
    print("Performance:")
    print("{0} {1}".format(np.average(acc_list), np.std(acc_list)))
    print("{0} {1}".format(np.average(f1_list), np.std(f1_list)))
