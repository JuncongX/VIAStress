# python roc.py --dataset_name wesad --k 5 --y_dim 2
# python roc.py --dataset_name wesad --k 5 --y_dim 3
# python roc.py --dataset_name ubfc_phys --k 7 --y_dim 2 --ubfc_phys_task 2
# python roc.py --dataset_name ubfc_phys --k 7 --y_dim 2 --ubfc_phys_task 3

import argparse
import os
import random
import numpy as np
from ClassicML.load_data_lowq import DataLoder
from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_auc_score, auc, roc_curve, f1_score, accuracy_score
import matplotlib.pyplot as plt

device_list = [0]
os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(str(x) for x in device_list)
import joblib


def calculate_macro_auc(all_probs, all_labels, num_classes):
    aucs = []
    fpr_dict = {}
    tpr_dict = {}

    for i in range(num_classes):
        fpr, tpr, _ = roc_curve(all_labels == i, all_probs[:, i])
        auc_score = roc_auc_score(all_labels == i, all_probs[:, i])  # Calculate AUC for each class
        aucs.append(auc_score)
        fpr_dict[i] = fpr
        tpr_dict[i] = tpr

    # Compute the macro-AUC as the mean of individual AUCs
    macro_auc = np.mean(aucs)

    return macro_auc, aucs, fpr_dict, tpr_dict



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Domain generalization')

    parser.add_argument('--model', type=str, default='Random_Forest',
                        choices=['LDA', 'KNN', 'AdaBoost', 'Decision_Tree', 'Random_Forest'])
    parser.add_argument('--k', type=int, default=6)
    parser.add_argument('--model_name', type=str, default='verbio', choices=['wesad', 'ubfc_phys', 'verbio'])
    parser.add_argument('--ubfc_phys_task', type=int, default=2, choices=[2, 3])
    parser.add_argument('--y_dim', type=int, default=2)

    parser.add_argument('--data_dir', default='./data/', type=str)
    parser.add_argument('--seed', type=int, default=123, help='Seed for everything else')
    parser.add_argument('--save_path', type=str, default="./checkpoints")

    args = parser.parse_args()

    from VIAStress.model import Model

    # If we ever want to implement checkpointing, just persist these values
    # every once in a while, and then load them from disk here.

    algorithm_dict = None

    random.seed(args.seed)
    np.random.seed(args.seed)

    if args.model_name != "wesad":
        data_loader_wesad = DataLoder("wesad")
    if args.model_name != "ubfc_phys":
        data_loader_ubfc_phys = DataLoder("ubfc_phys")
    if args.model_name != "verbio":
        data_loader_verbio = DataLoder("verbio")

    tprs = [0 for i in range(args.k)]
    aucs = [0 for i in range(args.k)]
    accs = [0 for i in range(args.k)]
    f1s = [0 for i in range(args.k)]
    mean_fpr = np.linspace(0, 1, 100)

    fig, ax = plt.subplots()

    tprs_all = []

    for fold in range(args.k):
        print(f"Fold:{fold}")
        model = joblib.load('save/{4}_{0}{1}_{2}_{3}.pkl'.format(
            args.model_name,
            "_" + str(args.ubfc_phys_task) if (args.model_name == 'ubfc_phys' and args.y_dim == 2) else "",
            args.y_dim,
            fold + 1, args.model
        ))
        features_list = []
        labels_list = []
        if args.model_name != "wesad":
            _, feat, label = data_loader_wesad.load(phase='cross', fold=0)
            features_list.append(feat)
            labels_list.append(label)
        if args.model_name != "ubfc_phys":
            _, feat, label = data_loader_ubfc_phys.load(phase='cross', fold=0)
            features_list.append(feat)
            labels_list.append(label)
        if args.model_name != "verbio":
            _, feat, label = data_loader_verbio.load(phase='cross', fold=0)
            features_list.append(feat)
            labels_list.append(label)
        features = np.concatenate(features_list, axis=0)
        labels = np.concatenate(labels_list, axis=0)
        # probs_list = []  # 存储预测得分
        # label_list = []  # 存储真实标签
        res = model.predict_proba(features)
        preds = np.argmax(res, axis=1)  # for multi-class (y_dim >= 2)

        # probs_array = np.array(res)
        probs_array = res
        # print(res)
        # labels_array = np.array(labels)
        labels_array = labels

        # 计算macro-AUC
        macro_auc, aucs_folds, fpr_dict, tpr_dict = calculate_macro_auc(probs_array, labels_array, args.y_dim)
        macro_f1 = f1_score(labels, preds, average='macro')
        accuracy = accuracy_score(labels, preds)
        print(f"Fold {fold} ACC: {accuracy} F1-score: {macro_f1} Macro AUC: {macro_auc}")

        aucs[fold] = macro_auc
        accs[fold] = accuracy * 100
        f1s[fold] = macro_f1

        # Store the tpr values for plotting the mean tpr
        for i in range(args.y_dim):
            tprs_all.append(np.interp(mean_fpr, fpr_dict[i], tpr_dict[i]))

    # 绘制 ROC 曲线
    # ax.plot([0, 1], [0, 1], linestyle='--', lw=2, color='r', label='Chance', alpha=.8)

    mean_auc = np.mean(aucs)
    std_auc = np.std(aucs)

    print(np.mean(accs), np.std(accs))
    print(np.mean(f1s), np.std(f1s))
    print(mean_auc, std_auc)
