import sklearn
import argparse
from sklearn.preprocessing import QuantileTransformer, StandardScaler
from sklearn.metrics import confusion_matrix, f1_score, accuracy_score
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC
import numpy as np
import joblib

from ClassicML.load_data import DataLoder

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--cd', type=str, default='u2w', choices=['w2u', 'u2w'])
    parser.add_argument('--k', type=int, default=7, help="KFold", choices=[5, 7])
    parser.add_argument('--ubfc_phys_task', type=int, default=3, choices=[2, 3])
    parser.add_argument('--y_dim', type=int, default=2)
    args = parser.parse_args()

    data_loader = DataLoder("wesad" if args.cd == "u2w" else "ubfc_phys", binary=True if args.y_dim == 2 else False, task=args.ubfc_phys_task)
    acc_list = []
    f1_list = []
    for fold in range(args.k):

        model = joblib.load('save/SVM_{0}{1}_{2}_{3}.pkl'.format(
            "ubfc_phys" if args.cd == "u2w" else "wesad",
            "_" + str(args.ubfc_phys_task) if (args.cd == 'u2w' and args.y_dim == 2) else "",
            args.y_dim,
            fold + 1
        ))
        persons, features, labels = data_loader.load(phase='cross', fold=0)
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