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
from Baseline.model import Model
from sklearn.metrics import confusion_matrix
from sklearn.metrics import f1_score
import copy

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

    # Precision calculation
    precision = np.zeros(num_classes)
    for label in range(num_classes):
        TP_label = cm[label, label]
        FP_label = np.sum(cm[:, label]) - TP_label
        precision[label] = TP_label / (TP_label + FP_label) if (TP_label + FP_label) != 0 else 0

    return accuracy, f1_macro, recall[1], precision[1]


def f1_accuracy(model, loader, class_num, device, fine_tune_ratio=0.1, epochs=100):
    model.eval()

    all_data = list(loader.dataset)
    n_total = len(all_data)
    # n_finetune = int(n_total * fine_tune_ratio)
    n_finetune = 10

    indices = list(range(n_total))
    random.shuffle(indices)
    finetune_indices = indices[:n_finetune]
    eval_indices = indices[n_finetune:]

    fine_tune_subset = torch.utils.data.Subset(loader.dataset, finetune_indices)
    eval_subset = torch.utils.data.Subset(loader.dataset, eval_indices)

    fine_tune_loader = torch.utils.data.DataLoader(fine_tune_subset, batch_size=loader.batch_size, shuffle=True)
    eval_loader = torch.utils.data.DataLoader(eval_subset, batch_size=loader.batch_size, shuffle=False)

    for param in model.feature_cnn.parameters():
        param.requires_grad = False

    optimizer = torch.optim.Adam(model.classifier.parameters(), lr=args.lr)
    criterion = torch.nn.CrossEntropyLoss()

    best_acc = 0.0
    best_state_dict = None
    model_c = copy.deepcopy(model)

    for epoch in range(epochs):
        model_c.train()
        running_loss, total = 0.0, 0
        all_preds, all_labels = [], []

        for data in fine_tune_loader:
            labels, ppg, scr, scl, eda = data
            labels, ppg, eda = labels.to(device), ppg.to(device), eda.to(device)

            optimizer.zero_grad()
            pred_logits = model_c(ppg, eda)
            loss = criterion(pred_logits, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * labels.size(0)
            _, preds = torch.max(pred_logits, 1)
            total += labels.size(0)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

        # --- 计算F1分数 ---
        f1 = f1_score(all_labels, all_preds, average='macro')
        if epoch % 10 == 0:
            print(f"[Fine-tune Epoch {epoch + 1}/{epochs}] Loss: {running_loss / total:.4f}, F1: {f1:.4f}")

        if f1 > best_acc:  # 用F1替代acc作为最佳标准
            best_acc = f1
            best_state_dict = model_c.state_dict()

    print(f"Best fine-tune macro-F1 on 10%: {best_acc:.4f}")

    if best_state_dict is not None:
        model_c.load_state_dict(best_state_dict)
    else:
        model_c = model
    model_c.eval()

    valid_cm = None

    for data in eval_loader:
    # for data in loader:
        labels, ppg, scr, scl, eda = data
        labels, ppg, eda = labels.to(device), ppg.to(device), eda.to(device)
        batch_size = labels.size(0)

        with torch.no_grad():
            pred_logits = model_c(ppg, eda)

            _, predictions = torch.max(pred_logits, 1)

            cm = confusion_matrix(labels.cpu().numpy(), predictions.cpu().numpy(),
                                  labels=[i for i in range(args.y_dim)])
            if valid_cm is None:
                valid_cm = cm
            else:
                valid_cm += cm

    return valid_cm


def model_name_tool(args):
    if args.model_name == "wesad":
        return ""
    elif args.model_name == "ubfc_phys":
        if args.y_dim == 3:
            return ""
        else:
            return "_UP{0}_".format(args.ubfc_phys_task)
    elif args.model_name == "verbio":
        return ""
    elif args.model_name == "can_stress":
        return ""



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Domain generalization')

    parser.add_argument('--k', type=int, default=5)
    parser.add_argument('--data_dir', default='./data/', type=str)
    parser.add_argument('--dataset_name', type=str, default='wesad', choices=['wesad', 'ubfc_phys', 'verbio'])
    parser.add_argument('--model_name', type=str, default='wesad', choices=['wesad', 'ubfc_phys', 'verbio'])
    parser.add_argument('--batch_size', type=int, default=128)
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

    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--ada_lr', type=float, default=0.0001)
    parser.add_argument('--weight_decay', type=float, default=0.)
    parser.add_argument('--x_dim', type=int, default=256)
    parser.add_argument('--y_dim', type=int, default=2)
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
    elif args.dataset_name == 'can_stress':
        from data.CAN_STRESS.CAN_STRESS_dataset_multi import CAN_STRESS_dataset
        from data.CAN_STRESS.Uniform_distribution_person_multi_0_4 import train_subject, valid_subject, test_subject
    elif args.dataset_name == 'verbio':
        from data.VerBIO.VerBIO_dataset_multi import VerBIO_dataset
        from data.VerBIO.Uniform_distribution_person_multi import train_subject, valid_subject, test_subject

    f1s = [0 for i in range(args.k)]
    ACCs = [0 for i in range(args.k)]
    Recalls = [0 for i in range(args.k)]
    Precisions = [0 for i in range(args.k)]

    for fold in range(args.k):
        print(f"Fold:{fold}")
        test_p = test_subject[0] + valid_subject[0] + train_subject[0]
        test_loader_list = []
        test_cm = None

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
            test_loader_list.append(test_loader)

        model = Model(y_dim=args.y_dim, z_dim=args.r_dim, x_dim=args.x_dim, h_dim=args.h_dim)
        model.load_state_dict(torch.load(os.path.join(args.save_path,
                                                      'BTL_{0}_{1}_{2}{3}.pth'.format(
                                                          args.model_name,
                                                          args.y_dim,
                                                          model_name_tool(args),
                                                          fold+1))))
        model.to(device)

        for i, test_p_one in enumerate(test_p):
            test_loader = test_loader_list[i]
            temp_cm = f1_accuracy(model, test_loader, args.y_dim, device)
            if test_cm is None:
                test_cm = temp_cm
            else:
                test_cm += temp_cm

        acc, f1, recall, precision = f1_score_from_confusion_matrix(test_cm, args.y_dim)


        print(
            f"Avg_valid_acc: {acc}%, Avg_valid_f1: {f1}, Avg_valid_recall: {recall}, Avg_valid_precision: {precision}")

        f1s[fold] = f1
        ACCs[fold] = acc
        Recalls[fold] = recall
        Precisions[fold] = precision

    print(np.average(ACCs), np.std(ACCs))
    print(np.average(f1s), np.std(f1s))
    print(np.average(Recalls), np.std(Recalls))
    print(np.average(Precisions), np.std(Precisions))