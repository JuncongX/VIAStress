# python roc.py --dataset_name wesad --k 5 --y_dim 2
# python roc.py --dataset_name wesad --k 5 --y_dim 3
# python roc.py --dataset_name ubfc_phys --k 7 --y_dim 2 --ubfc_phys_task 2
# python roc.py --dataset_name ubfc_phys --k 7 --y_dim 2 --ubfc_phys_task 3
# python roc_cml.py --dataset_name can_stress --k 5 --y_dim 2 --model LDA

import argparse
import os
import random
import numpy as np
from ClassicML.load_data import DataLoder
from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_auc_score, auc, roc_curve
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


def model_name_tool(args):
    if args.dataset_name == "wesad":
        return ""
    elif args.dataset_name == "ubfc_phys":
        if args.y_dim == 3:
            return ""
        else:
            return "_UP{0}_".format(args.ubfc_phys_task)
    elif args.dataset_name == "universe":
        return "_{0}_".format(args.universe_task)
    elif args.dataset_name == "road":
        return ""


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Domain generalization')

    parser.add_argument('--model', type=str, default='AdaBoost', choices=['LDA', 'KNN', 'AdaBoost', 'Decision_Tree', 'Random_Forest'])
    parser.add_argument('--k', type=int, default=5)
    parser.add_argument('--dataset_name', type=str, default='wesad', choices=['wesad', 'ubfc_phys', 'can_stress', 'verbio'])
    parser.add_argument('--y_dim', type=int, default=3)
    parser.add_argument('--ubfc_phys_task', type=int, default=3, choices=[2, 3])

    parser.add_argument('--data_dir', default='./data/', type=str)
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--seed', type=int, default=0, help='Seed for everything else')
    parser.add_argument('--save_path', type=str, default="./checkpoints")

    args = parser.parse_args()

    from VIAStress.model import Model

    # If we ever want to implement checkpointing, just persist these values
    # every once in a while, and then load them from disk here.

    algorithm_dict = None

    random.seed(args.seed)
    np.random.seed(args.seed)

    data_loader = DataLoder(args.dataset_name, binary=True if args.y_dim == 2 else False, task=args.ubfc_phys_task)

    tprs = [0 for i in range(args.k)]
    aucs = [0 for i in range(args.k)]
    mean_fpr = np.linspace(0, 1, 100)

    fig, ax = plt.subplots()

    tprs_all = []

    for fold in range(args.k):
        print(f"Fold:{fold}")
        model = joblib.load('save/{4}_{0}{1}_{2}_{3}.pkl'.format(
            args.dataset_name,
            "_" + str(args.ubfc_phys_task) if (args.dataset_name == 'ubfc_phys' and args.y_dim == 2) else "",
            args.y_dim,
            fold + 1, args.model
        ))

        persons, features, labels = data_loader.load(phase='test', fold=fold)

        # probs_list = []  # 存储预测得分
        # label_list = []  # 存储真实标签
        res = model.predict_proba(features)

        # probs_array = np.array(res)
        probs_array = res
        # print(res)
        # labels_array = np.array(labels)
        labels_array = labels

        # 计算macro-AUC
        macro_auc, aucs_folds, fpr_dict, tpr_dict = calculate_macro_auc(probs_array, labels_array, args.y_dim)
        print(f"Fold {fold} Macro AUC: {macro_auc}")
        aucs[fold] = macro_auc

        # Store the tpr values for plotting the mean tpr
        for i in range(args.y_dim):
            tprs_all.append(np.interp(mean_fpr, fpr_dict[i], tpr_dict[i]))

    # 绘制 ROC 曲线
    # ax.plot([0, 1], [0, 1], linestyle='--', lw=2, color='r', label='Chance', alpha=.8)

    mean_auc = np.mean(aucs)
    std_auc = np.std(aucs)
    mean_tpr = np.mean(tprs_all, axis=0)
    mean_tpr[-1] = 1.0  # Ensure the last point is 1.0 for all ROC curves
    mean_tpr[0] = 0

    print(f"Mean Macro AUC: {mean_auc}")

    print(",".join(str(n) for n in mean_fpr))
    print(",".join(str(n) for n in mean_tpr))
    print(mean_auc, std_auc)

    # ax.plot(mean_fpr, mean_tpr, color='b',
    #         label=r'Mean ROC (AUC = %0.3f)' % (mean_auc),
    #         lw=2, alpha=.8)
    #
    # ax.set(xlim=[-0.05, 1.05], ylim=[-0.05, 1.05],
    #        title="Stress Detection Macro-AUC curve")
    # ax.legend(loc="lower right")
    #
    # plt.show()
    # plt.close()
