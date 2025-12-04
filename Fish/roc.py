# python roc.py --dataset_name wesad --k 5 --y_dim 2
# python roc.py --dataset_name ubfc_phys --k 7 --y_dim 2 --ubfc_phys_task 2
# python roc.py --dataset_name can_stress --k 5 --y_dim 2
# python roc.py --dataset_name verbio --k 6 --y_dim 2

import argparse
import collections
import json
import os
import random
import sys
import time
import uuid
import numpy as np
import torch
import torchvision
import torch.utils.data
from torch.utils.data import DataLoader
from torch.autograd import Variable
from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_auc_score, auc, roc_curve
import matplotlib.pyplot as plt
from AUC.roc_me import calculate_macro_auc, model_name_tool

device_list = [0]
os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(str(x) for x in device_list)

def auc_score(network, loader, num_classes, device):
    network.eval()
    all_probs = []
    all_labels = []

    with torch.no_grad():
        for labels, ppg, scr, scl, eda, peak in loader:
            labels = labels.to(device)
            ppg = ppg.to(device)
            eda = eda.to(device)
            logits = network(ppg, eda)
            probs = torch.softmax(logits, dim=1)  # 对 logits 进行 softmax 转换为概率
            all_probs.append(probs.cpu().numpy())
            all_labels.append(labels.cpu().numpy())

    all_probs = np.concatenate(all_probs)
    all_labels = np.concatenate(all_labels)

    return all_probs, all_labels


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Domain generalization')

    parser.add_argument('--k', type=int, default=5)
    parser.add_argument('--data_dir', default='./data/', type=str)
    parser.add_argument('--dataset_name', type=str, default='wesad', choices=['wesad', 'ubfc_phys', 'can_stress', 'verbio'])
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--seed', type=int, default=0, help='Seed for everything else')
    parser.add_argument('--lr', type=float, default=1e-3, help='Seed for everything else')
    parser.add_argument('--weight_decay', type=float, default=5e-4, help='Seed for everything else')

    parser.add_argument('--save_path', type=str, default="./checkpoints")

    parser.add_argument('--x_dim', type=int, default=256)
    parser.add_argument('--y_dim', type=int, default=2)
    parser.add_argument('--z_dim', type=int, default=256)
    parser.add_argument('--h_dim', type=int, default=256)
    parser.add_argument('--r_dim', type=int, default=256)

    parser.add_argument('--ubfc_phys_task', type=int, default=3, choices=[2, 3])

    args = parser.parse_args()

    from Fish.model import WholeFish

    # If we ever want to implement checkpointing, just persist these values
    # every once in a while, and then load them from disk here.

    algorithm_dict = None

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    if torch.cuda.is_available():
        device = "cuda"
        print('device count:', torch.cuda.device_count())
    else:
        device = "cpu"

    if args.dataset_name == 'wesad':
        from VIAStress_wo_MMDG.WESAD_dataset_multi import WESAD_dataset
        from data.WESAD.Uniform_distribution_person_multi import train_subject, valid_subject, test_subject
    elif args.dataset_name == 'ubfc_phys':
        from VIAStress_wo_MMDG.UBFC_Phys_dataset_multi import UBFC_Phys_dataset
        from data.UBFC_Phys.Uniform_distribution_person_multi import train_subject, valid_subject, test_subject
    elif args.dataset_name == 'can_stress':
        from VIAStress_wo_MMDG.CAN_STRESS_dataset_multi import CAN_STRESS_dataset
        from data.CAN_STRESS.Uniform_distribution_person_multi_0_4 import train_subject, valid_subject, test_subject
    elif args.dataset_name == 'verbio':
        from VIAStress_wo_MMDG.VerBIO_dataset_multi import VerBIO_dataset
        from data.VerBIO.Uniform_distribution_person_multi import train_subject, valid_subject, test_subject

    tprs = [0 for i in range(args.k)]
    aucs = [0 for i in range(args.k)]
    mean_fpr = np.linspace(0, 1, 100)

    fig, ax = plt.subplots()

    tprs_all = []

    for fold in range(args.k):
        print(f"Fold:{fold}")
        model = WholeFish(args.x_dim, args.h_dim, args.y_dim)
        model.load_state_dict(torch.load(os.path.join(args.save_path,
                                                      'Fish_{0}_{1}_{2}{3}.pth'.format(
                                                          args.dataset_name,
                                                          args.y_dim,
                                                          model_name_tool(args),
                                                          fold+1))))
        model.to(device)

        test_p = test_subject[fold]

        probs_list = []  # 存储预测得分
        label_list = []  # 存储真实标签
        for i, test_p_one in enumerate(test_p):
            if args.dataset_name == 'wesad':
                test_dataset = WESAD_dataset([test_p_one], binary=True if args.y_dim == 2 else False)
            elif args.dataset_name == 'ubfc_phys':
                if args.y_dim == 2:
                    test_dataset = UBFC_Phys_dataset([test_p_one], binary=True, task=args.ubfc_phys_task)
                else:
                    test_dataset = UBFC_Phys_dataset([test_p_one], binary=False)
            elif args.dataset_name == 'can_stress':
                test_dataset = CAN_STRESS_dataset([test_p_one])
            elif args.dataset_name == 'verbio':
                test_dataset = VerBIO_dataset([test_p_one])
            test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=True)
            probs, labels = auc_score(model, test_loader, args.y_dim, device)
            probs_list.extend(probs)
            label_list.extend(labels)

        probs_array = np.array(probs_list)
        labels_array = np.array(label_list)
        # 将label转换成onehot形式
        # label_tensor = torch.tensor(labels_array)
        # label_tensor = label_tensor.reshape((label_tensor.shape[0], 1))
        # label_onehot = torch.zeros(label_tensor.shape[0], args.y_dim)
        # label_onehot.scatter_(dim=1, index=label_tensor, value=1)
        # label_onehot = np.array(label_onehot)

        # fpr, tpr, _ = roc_curve(labels_array.ravel(), probs_array.ravel())
        # roc_auc = auc(fpr, tpr)
        # print(f"Fold {fold} AUC: {roc_auc}")
        # interp_tpr = np.interp(mean_fpr, fpr, tpr)
        # interp_tpr[0] = 0.0
        # tprs[fold] = interp_tpr
        # aucs[fold] = roc_auc

        # 计算macro-AUC
        macro_auc, aucs_folds, fpr_dict, tpr_dict = calculate_macro_auc(probs_array, labels_array, args.y_dim)
        print(f"Fold {fold} Macro AUC: {macro_auc}")
        aucs[fold] = macro_auc

        # Store the tpr values for plotting the mean tpr
        for i in range(args.y_dim):
            tprs_all.append(np.interp(mean_fpr, fpr_dict[i], tpr_dict[i]))


    # 绘制 ROC 曲线
    ax.plot([0, 1], [0, 1], linestyle='--', lw=2, color='r', label='Chance', alpha=.8)

    mean_auc = np.mean(aucs)
    std_auc = np.std(aucs)
    mean_tpr = np.mean(tprs_all, axis=0)
    mean_tpr[-1] = 1.0  # Ensure the last point is 1.0 for all ROC curves
    mean_tpr[0] = 0

    print(f"Mean Macro AUC: {mean_auc}")

    # print(",".join(str(n) for n in mean_fpr))
    print(",".join(str(n) for n in mean_tpr))
    print(mean_auc, std_auc)

    ax.plot(mean_fpr, mean_tpr, color='b',
            label=r'Mean ROC (AUC = %0.3f)' % (mean_auc),
            lw=2, alpha=.8)

    ax.set(xlim=[-0.05, 1.05], ylim=[-0.05, 1.05],
           title="Stress Detection Macro-AUC curve")
    ax.legend(loc="lower right")

    plt.show()
    plt.close()
