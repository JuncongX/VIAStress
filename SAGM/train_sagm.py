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
from SAGM.sagm import SAGM, LinearScheduler
from sklearn.metrics import confusion_matrix
import torch.nn.functional as F

device_list = [1]
os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(str(x) for x in device_list)

def f1_score_from_confusion_matrix(cm):
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

    return accuracy, f1_macro


def f1_accuracy(network, loader, class_num, device):
    cm = None

    network.eval()
    with torch.no_grad():
        for labels, ppg, scr, scl, eda in loader:
            labels, ppg, eda = Variable(labels).to(device), Variable(ppg).to(device), Variable(eda).to(device)
            p = network(ppg, eda)
            scores, predictions = torch.max(p.data, 1)
            if cm is None:
                cm = confusion_matrix(labels.cpu().numpy(), predictions.cpu().numpy(), labels=[i for i in range(class_num)])
            else:
                cm += confusion_matrix(labels.cpu().numpy(), predictions.cpu().numpy(), labels=[i for i in range(class_num)])
    network.train()

    return cm


def model_name_tool(args):
    if args.dataset_name == "wesad":
        return ""
    elif args.dataset_name == "ubfc_phys":
        if args.class_num == 3:
            return ""
        else:
            return "_UP{0}_".format(args.ubfc_phys_task)
    elif args.dataset_name == "verbio":
        return ""
    elif args.dataset_name == "can_stress":
        return ""


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Domain generalization')

    parser.add_argument('--k', type=int, default=5)
    parser.add_argument('--data_dir', default='./data/', type=str)
    parser.add_argument('--dataset_name', type=str, default='wesad', choices=['wesad', 'ubfc_phys', 'can_stress', 'verbio'])
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--algorithm', type=str, default="SAGM", choices=["SAGM", "BL"])
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
    parser.add_argument('--ada_lr', type=float, default=1e-3)
    parser.add_argument('--weight_decay', type=float, default=5e-4)
    parser.add_argument('--x_dim', type=int, default=256)
    parser.add_argument('--class_num', type=int, default=2)
    parser.add_argument('--z_dim', type=int, default=256)
    parser.add_argument('--h_dim', type=int, default=256)
    parser.add_argument('--r_dim', type=int, default=256)

    parser.add_argument('--ubfc_phys_task', type=int, default=3, choices=[2, 3])
    parser.add_argument('--universe_task', type=str, default="arithmetix",
                        choices=["arithmetix", "n_back", "stroop", "sudoku"])

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


    def loss_fn(predictions, targets):
        return F.cross_entropy(predictions, targets)

    n_steps = args.steps
    best_f1 = [0 for i in range(args.k)]
    best_ACC = [0 for i in range(args.k)]
    for fold in range(args.k):
        print(f"Fold:{fold}")

        train_p = train_subject[fold]
        valid_p = valid_subject[fold]
        test_p = test_subject[fold]

        train_loader_list = []
        valid_loader_list = []

        for i, train_p_one in enumerate(train_p):
            if args.dataset_name == 'wesad':
                train_dataset = WESAD_dataset([train_p_one], binary=True if args.class_num == 2 else False)
            elif args.dataset_name == 'ubfc_phys':
                if args.class_num == 2:
                    train_dataset = UBFC_Phys_dataset([train_p_one], binary=True, task=args.ubfc_phys_task)
                else:
                    train_dataset = UBFC_Phys_dataset([train_p_one], binary=False)
            elif args.dataset_name == 'can_stress':
                train_dataset = CAN_STRESS_dataset([train_p_one])
            elif args.dataset_name == 'verbio':
                train_dataset = VerBIO_dataset([train_p_one])
            train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
            train_loader_list.append(train_loader)

        for i, valid_p_one in enumerate(valid_p):
            if args.dataset_name == 'wesad':
                valid_dataset = WESAD_dataset([valid_p_one], binary=True if args.class_num == 2 else False)
            elif args.dataset_name == 'ubfc_phys':
                if args.class_num == 2:
                    valid_dataset = UBFC_Phys_dataset([valid_p_one], binary=True, task=args.ubfc_phys_task)
                else:
                    valid_dataset = UBFC_Phys_dataset([valid_p_one], binary=False)
            elif args.dataset_name == 'can_stress':
                valid_dataset = CAN_STRESS_dataset([valid_p_one])
            elif args.dataset_name == 'verbio':
                valid_dataset = VerBIO_dataset([valid_p_one])
            valid_loader = DataLoader(valid_dataset, batch_size=args.batch_size, shuffle=True)
            valid_loader_list.append(valid_loader)

        model = Model(args.x_dim, args.r_dim, args.h_dim, args.class_num)
        model.to(device)

        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=args.lr,
            weight_decay=args.weight_decay,
        )

        rho_scheduler = LinearScheduler(T_max=5000, max_value=0.05,
                                             min_value=0.05)

        lr_scheduler = LinearScheduler(T_max=5000, max_value=args.lr,
                                            min_value=args.lr, optimizer=optimizer)

        SAGM_optimizer = SAGM(params=model.parameters(), base_optimizer=optimizer, model=model,
                               alpha=0.5, rho_scheduler=rho_scheduler, adaptive=False)

        start_step = 0
        bestf1_ever = 0.0
        for step in range(start_step, n_steps):
            for i, train_p_one in enumerate(train_p):
                train_loader = train_loader_list[i]
                for j, data in enumerate(train_loader):
                    labels, ppg, scr, scl, eda = data
                    # eda = torch.cat((scr, scl), dim=1)
                    labels, ppg, eda = Variable(labels).to(device), Variable(ppg).to(device), Variable(eda).to(device)

                    SAGM_optimizer.set_closure(loss_fn, ppg, eda, labels)
                    predictions, loss = SAGM_optimizer.step()
                    lr_scheduler.step()

            valid_cm = None
            for i, valid_p_one in enumerate(valid_p):

                valid_loader = valid_loader_list[i]

                cm = f1_accuracy(model, valid_loader, args.class_num, device)
                if valid_cm is None:
                    valid_cm = cm
                else:
                    valid_cm += cm

            acc, f1 = f1_score_from_confusion_matrix(valid_cm)

            print(f"Epoch: {step}, Avg_valid_acc: {acc}%, Avg_valid_f1: {f1}")

            algorithm_dict = model.state_dict()

            if f1 >= bestf1_ever:
                bestf1_ever = f1
                best_f1[fold] = f1
                best_ACC[fold] = acc
                torch.save(algorithm_dict, os.path.join(args.save_path,
                                                        f"{args.algorithm}_{args.dataset_name}_{args.class_num}_{model_name_tool(args)}{fold}"))

    print(np.average(best_ACC), np.std(best_ACC))
    print(np.average(best_f1), np.std(best_f1))