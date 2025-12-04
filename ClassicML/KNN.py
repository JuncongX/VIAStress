from ClassicML.load_data import DataLoder
import sklearn
import argparse
from sklearn.preprocessing import QuantileTransformer, StandardScaler
from sklearn.metrics import confusion_matrix, f1_score, accuracy_score
from sklearn.pipeline import Pipeline
from sklearn.neighbors import KNeighborsClassifier
import joblib

import numpy as np

def KNN():
    return Pipeline([
        ('std_scaler', StandardScaler()),
        ('KNN', KNeighborsClassifier(n_neighbors=9))
    ])


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_name', type=str, default='verbio', choices=['wesad', 'ubfc_phys', 'can_stress', 'verbio'])
    parser.add_argument('--k', type=int, default=5, help="KFold", choices=[5, 7])
    parser.add_argument('--y_dim', type=int, default=2, choices=[2, 3])
    parser.add_argument('--ubfc_phys_task', type=int, default=2, choices=[2, 3])
    parser.add_argument('--context_num', type=int, default=0)
    args = parser.parse_args()
    scaler = StandardScaler()
    data_loader = DataLoder(args.dataset_name, binary=True if args.y_dim == 2 else False, task=args.ubfc_phys_task)
    acc_list = []
    f1_list = []
    for fold in range(args.k):
        _, features_train, labels_train = data_loader.load(phase='train', fold=fold)
        _, features_valid, labels_valid = data_loader.load(phase='valid', fold=fold)
        persons_test, features_test, labels_test = data_loader.load(phase='test', fold=fold)
        del_index_choice = None

        for person in np.unique(persons_test):
            baseline_index = np.arange(len(labels_test))[(labels_test == 0) & (persons_test == person)]
            baseline_index_choice = np.random.choice(baseline_index, args.context_num, replace=False)
            if del_index_choice is None:
                del_index_choice = baseline_index_choice
            else:
                del_index_choice = np.concatenate((del_index_choice, baseline_index_choice))
        remaining_indices = np.delete(np.arange(len(labels_test)), del_index_choice)
        features_test = features_test[remaining_indices]
        labels_test = labels_test[remaining_indices]

        knn = KNN()
        knn.fit(features_train, labels_train)
        res = knn.predict(features_test)
        macro_f1 = f1_score(labels_test, res, average='macro')
        accuracy = accuracy_score(labels_test, res)
        cm = confusion_matrix(labels_test, res)
        print(cm)
        print(100 * accuracy, macro_f1)
        joblib.dump(knn, 'save/KNN_{0}{1}_{2}_{3}.pkl'.format(
            args.dataset_name,
            "_" + str(args.ubfc_phys_task) if (args.dataset_name == 'ubfc_phys' and args.y_dim == 2) else "",
            args.y_dim,
            fold + 1
        ))
        acc_list.append(100 * accuracy)
        f1_list.append(macro_f1)
    print("Performance:")
    print("{0} {1}".format(np.average(acc_list), np.std(acc_list)))
    print("{0} {1}".format(np.average(f1_list), np.std(f1_list)))
