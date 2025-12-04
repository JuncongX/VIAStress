# python cross_dataset_bv_vx.py --model_name wesad --dataset_name ubfc_phys --k 5 --y_dim 2 --vx 2 --ubfc_phys_task 2
# python cross_dataset_bv_vx.py --model_name ubfc_phys --dataset_name wesad --k 7 --y_dim 2 --vx 2 --ubfc_phys_task 2
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
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

device_list = [0]
os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(str(x) for x in device_list)


def f1_score_from_confusion_matrix(cm, num_classes):
    TP = np.diag(cm)
    FP = np.sum(cm, axis=0) - TP
    FN = np.sum(cm, axis=1) - TP

    denominator = (2 * TP + FP + FN)

    f1_scores = np.zeros_like(TP, dtype=float)

    valid = denominator != 0
    f1_scores[valid] = 2 * TP[valid] / denominator[valid]

    f1_macro = np.mean(f1_scores)

    total_predictions = np.sum(cm)
    accuracy = np.sum(TP) / total_predictions * 100

    # Recall calculation
    recall = np.zeros(num_classes)
    for label in range(num_classes):
        TP_label = cm[label, label]
        FN_label = np.sum(cm[label, :]) - TP_label
        recall[label] = TP_label / (TP_label + FN_label) if (TP_label + FN_label) != 0 else 0

    return accuracy, f1_macro, recall[1]


def f1_accuracy(network, loader, class_num, device):
    cm = None

    network.eval()
    with torch.no_grad():
        for labels, ppg, scr, scl, eda in loader:
            labels, ppg, eda = Variable(labels).to(device), Variable(ppg).to(device), Variable(eda).to(device)
            p = network(ppg, eda)
            scores, predictions = torch.max(p.data, 1)
            if cm is None:
                cm = confusion_matrix(labels.cpu().numpy(), predictions.cpu().numpy(),
                                      labels=[i for i in range(class_num)])
            else:
                cm += confusion_matrix(labels.cpu().numpy(), predictions.cpu().numpy(),
                                       labels=[i for i in range(class_num)])
    network.train()

    return f1_score_from_confusion_matrix(cm, class_num)


def model_name_tool(args):
    if args.model_name == "wesad":
        return ""
    elif args.model_name == "ubfc_phys":
        if args.y_dim == 3:
            return ""
        else:
            return "_UP{0}_".format(args.ubfc_phys_task)
    elif args.model_name == "universe":
        return "_{0}_".format(args.universe_task)
    elif args.model_name == "road":
        return ""



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Domain generalization')

    parser.add_argument('--k', type=int, default=5)
    parser.add_argument('--data_dir', default='./data/', type=str)
    parser.add_argument('--dataset_name', type=str, default='ubfc_phys', choices=['wesad', 'ubfc_phys'])
    parser.add_argument('--model_name', type=str, default='wesad', choices=['wesad', 'ubfc_phys'])
    parser.add_argument('--batch_size', type=int, default=120)
    parser.add_argument('--seed', type=int, default=0, help='Seed for everything else')
    parser.add_argument('--steps', type=int, default=200,
                        help='Number of steps. Default is dataset-dependent.')
    parser.add_argument('--checkpoint_freq', type=int, default=None,
                        help='Checkpoint every N steps. Default is dataset-dependent.')
    parser.add_argument('--save_path', type=str, default="./checkpoints")
    parser.add_argument('--output_dir', type=str, default="./train_output")
    parser.add_argument('--holdout_fraction', type=float, default=0.2)
    parser.add_argument('--uda_holdout_fraction', type=float, default=0,
                        help="For domain adaptation, % of test to use unlabeled for training.")
    parser.add_argument('--skip_model_save', action='store_true')
    parser.add_argument('--save_model_every_checkpoint', action='store_true')

    parser.add_argument('--lr', type=float, default=0.0001)
    parser.add_argument('--ada_lr', type=float, default=0.0001)
    parser.add_argument('--weight_decay', type=float, default=0.)
    parser.add_argument('--x_dim', type=int, default=256)
    parser.add_argument('--y_dim', type=int, default=2)
    parser.add_argument('--vx', type=int, default=1)
    parser.add_argument('--z_dim', type=int, default=256)
    parser.add_argument('--h_dim', type=int, default=256)
    parser.add_argument('--r_dim', type=int, default=256)

    parser.add_argument('--ubfc_phys_task', type=int, default=3, choices=[2, 3])

    args = parser.parse_args()

    # If we ever want to implement checkpointing, just persist these values
    # every once in a while, and then load them from disk here.

    algorithm_dict = None

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    if args.vx == 1:
        from Baseline.model_v1 import Model
    elif args.vx == 2:
        from Baseline.model_v2 import Model
    elif args.vx == 3:
        from Baseline.model_v3 import Model
    elif args.vx == 4:
        from Baseline.model_v4 import Model
    elif args.vx == 5:
        from Baseline.model_v5 import Model

    if torch.cuda.is_available():
        device = "cuda"
        print('device count:', torch.cuda.device_count())
    else:
        device = "cpu"

    if args.dataset_name == 'wesad':
        from data.WESAD.WESAD_dataset_multi_ae import WESAD_dataset
        from data.WESAD.Uniform_distribution_person_multi import train_subject, valid_subject, test_subject
    elif args.dataset_name == 'ubfc_phys':
        from data.UBFC_Phys.UBFC_Phys_dataset_multi import UBFC_Phys_dataset
        from data.UBFC_Phys.Uniform_distribution_person_multi import train_subject, valid_subject, test_subject

    all_subject = test_subject[0] + valid_subject[0] + train_subject[0]

    if args.dataset_name == 'wesad':
        test_dataset = WESAD_dataset(all_subject, binary=True if args.y_dim == 2 else False)
    elif args.dataset_name == 'ubfc_phys':
        if args.y_dim == 2:
            test_dataset = UBFC_Phys_dataset(all_subject, binary=True, task=args.ubfc_phys_task)
        else:
            test_dataset = UBFC_Phys_dataset(all_subject, binary=False)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=True)

    f1s = [0 for i in range(args.k)]
    ACCs = [0 for i in range(args.k)]
    Recalls = [0 for i in range(args.k)]

    for fold in range(args.k):
        print(f"Fold:{fold}")

        model = Model(y_dim=args.y_dim, z_dim=args.r_dim, x_dim=args.x_dim, h_dim=args.h_dim)
        model.load_state_dict(torch.load(os.path.join(args.save_path,
                                                      'BV_v{4}_{0}_{1}_{2}{3}.pth'.format(
                                                          args.model_name,
                                                          args.y_dim,
                                                          model_name_tool(args),
                                                          fold+1, args.vx))))
        model.to(device)

        acc, f1, recall = f1_accuracy(model, test_loader, args.y_dim, device)
        print(f"Avg_valid_acc: {acc}%, Avg_valid_f1: {f1}, Avg_valid_recall: {recall}")

        f1s[fold] = f1
        ACCs[fold] = acc
        Recalls[fold] = recall

    print(np.average(ACCs), np.std(ACCs))
    print(np.average(f1s), np.std(f1s))
    print(np.average(Recalls), np.std(Recalls))

