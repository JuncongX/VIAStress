# python eval_diff.py --model_name wesad --dataset_name wesad --k 5 --y_dim 2 --exp 1
# python eval_diff.py --model_name wesad --dataset_name wesad --k 5 --y_dim 3 --exp 0.5
# python eval_diff.py --model_name wesad --dataset_name ubfc_phys --k 5 --y_dim 2 --exp 1 --ubfc_phys_task 2
# python eval_diff.py --model_name wesad --dataset_name verbio --k 5 --y_dim 2 --exp 1

# python eval_diff.py --model_name verbio --dataset_name verbio --k 5 --y_dim 2 --exp 1.5
# python eval_diff.py --model_name verbio --dataset_name wesad --k 5 --y_dim 2 --exp 1.5
# python eval_diff.py --model_name verbio --dataset_name ubfc_phys --k 5 --y_dim 2 --exp 1.5 --ubfc_phys_task 2

# python eval_diff.py --model_name ubfc_phys --dataset_name ubfc_phys --k 7 --y_dim 2 --exp 1 --ubfc_phys_task 2
# python eval_diff.py --model_name ubfc_phys --dataset_name wesad --k 7 --y_dim 2 --exp 1 --ubfc_phys_task 2
# python eval_diff.py --model_name ubfc_phys --dataset_name verbio --k 7 --y_dim 2 --exp 1 --ubfc_phys_task 2

# python eval_diff.py --model_name ubfc_phys --dataset_name ubfc_phys --k 7 --y_dim 2 --exp 0.5 --ubfc_phys_task 3
# python eval_diff.py --model_name ubfc_phys --dataset_name wesad --k 7 --y_dim 2 --exp 0.5 --ubfc_phys_task 3
# python eval_diff.py --model_name ubfc_phys --dataset_name verbio --k 7 --y_dim 2 --exp 0.5 --ubfc_phys_task 3

# python eval_diff.py --model_name can_stress --dataset_name can_stress --k 5 --y_dim 2 --exp 1

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
from VIAStress.model import Model
from sklearn.metrics import confusion_matrix
from data.MMD_permutation_test import permutation_test_mmd
from scipy.stats import combine_pvalues, norm

device_list = [0]
os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(str(x) for x in device_list)


def eval_mmd_p(network, train_loader, test_loader, device, dim_=128):
    network.eval()
    feature_s_train, feature_s_test = [], []

    with torch.no_grad():
        # 训练集特征提取
        for _, ppg, _, _, eda in train_loader:
            ppg, eda = ppg.to(device), eda.to(device)
            pred_logits, bvp_z, eda_z, _, _, _, _, _, _ = network(ppg, eda)
            bvp_S = bvp_z[:, :dim_]
            eda_S = eda_z[:, :dim_]
            feature_s_train.append(torch.cat([bvp_S, eda_S], dim=1).cpu())

        # 测试集特征提取
        for _, ppg, _, _, eda in test_loader:
            ppg, eda = ppg.to(device), eda.to(device)
            pred_logits, bvp_z, eda_z, _, _, _, _, _, _ = network(ppg, eda)
            bvp_S = bvp_z[:, :dim_]
            eda_S = eda_z[:, :dim_]
            feature_s_test.append(torch.cat([bvp_S, eda_S], dim=1).cpu())

    # 拼接所有 batch
    feature_s_train = torch.cat(feature_s_train, dim=0).numpy()
    feature_s_test = torch.cat(feature_s_test, dim=0).numpy()

    # 计算 MMD 统计量
    return permutation_test_mmd(feature_s_train, feature_s_test)


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
    parser.add_argument('--dataset_name', type=str, default='ubfc_phys', choices=['wesad', 'ubfc_phys', 'verbio', 'can_stress'])
    parser.add_argument('--model_name', type=str, default='wesad', choices=['wesad', 'ubfc_phys', 'verbio', 'can_stress'])
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

    parser.add_argument('--lr', type=float, default=0.0001)
    parser.add_argument('--ada_lr', type=float, default=0.0001)
    parser.add_argument('--weight_decay', type=float, default=0.)
    parser.add_argument('--x_dim', type=int, default=256)
    parser.add_argument('--y_dim', type=int, default=2)
    parser.add_argument('--z_dim', type=int, default=256)
    parser.add_argument('--h_dim', type=int, default=256)
    parser.add_argument('--r_dim', type=int, default=256)
    parser.add_argument('--exp', type=str, default=0.5)
    parser.add_argument('--od', type=int, default=64)

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

    from data.WESAD.WESAD_dataset_multi_ae import WESAD_dataset
    from data.UBFC_Phys.UBFC_Phys_dataset_multi import UBFC_Phys_dataset
    from data.VerBIO.VerBIO_dataset_multi import VerBIO_dataset
    from data.CAN_STRESS.CAN_STRESS_dataset_multi import CAN_STRESS_dataset

    if args.dataset_name == 'wesad':
        from data.WESAD.Uniform_distribution_person_multi import train_subject as train_subject_tt, valid_subject as valid_subject_tt, test_subject as test_subject_tt
    elif args.dataset_name == 'ubfc_phys':
        from data.UBFC_Phys.Uniform_distribution_person_multi import train_subject as train_subject_tt, valid_subject as valid_subject_tt, test_subject as test_subject_tt
    elif args.dataset_name == 'verbio':
        from data.VerBIO.Uniform_distribution_person_multi import train_subject as train_subject_tt, valid_subject as valid_subject_tt, test_subject as test_subject_tt
    elif args.dataset_name == 'can_stress':
        from data.CAN_STRESS.Uniform_distribution_person_multi_0_4 import train_subject as train_subject_tt, valid_subject as valid_subject_tt, test_subject as test_subject_tt

    if args.model_name == 'wesad':
        from data.WESAD.Uniform_distribution_person_multi import train_subject as train_subject_tn
    elif args.model_name == 'ubfc_phys':
        from data.UBFC_Phys.Uniform_distribution_person_multi import train_subject as train_subject_tn
    elif args.model_name == 'verbio':
        from data.VerBIO.Uniform_distribution_person_multi import train_subject as train_subject_tn
    elif args.model_name == 'can_stress':
        from data.CAN_STRESS.Uniform_distribution_person_multi_0_4 import train_subject as train_subject_tn

    MMDs = [0 for i in range(args.k)]
    ps = [0 for i in range(args.k)]

    for fold in range(args.k):
        print(f"Fold:{fold}")

        if args.dataset_name == args.model_name:
            tt_subject = test_subject_tt[fold]
        else:
            tt_subject = test_subject_tt[0] + valid_subject_tt[0] + train_subject_tt[0]

        if args.dataset_name == 'wesad':
            test_dataset = WESAD_dataset(tt_subject, binary=True if args.y_dim == 2 else False)
        elif args.dataset_name == 'ubfc_phys':
            if args.y_dim == 2:
                test_dataset = UBFC_Phys_dataset(tt_subject, binary=True, task=args.ubfc_phys_task)
            else:
                test_dataset = UBFC_Phys_dataset(tt_subject, binary=False)
        elif args.dataset_name == 'verbio':
            test_dataset = VerBIO_dataset(tt_subject)
        elif args.dataset_name == 'can_stress':
            test_dataset = CAN_STRESS_dataset(tt_subject)
        test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=True)

        if args.model_name == 'wesad':
            train_dataset = WESAD_dataset(train_subject_tn[fold], binary=True if args.y_dim == 2 else False)
        elif args.model_name == 'ubfc_phys':
            if args.y_dim == 2:
                train_dataset = UBFC_Phys_dataset(train_subject_tn[fold], binary=True, task=args.ubfc_phys_task)
            else:
                train_dataset = UBFC_Phys_dataset(train_subject_tn[fold], binary=False)
        elif args.model_name == 'verbio':
            train_dataset = VerBIO_dataset(train_subject_tn[fold])
        elif args.model_name == 'can_stress':
            train_dataset = CAN_STRESS_dataset(train_subject_tn[fold])
        train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)

        model = Model(y_dim=args.y_dim, r_dim=args.r_dim, x_dim=args.x_dim, h_dim=args.h_dim)
        model.load_state_dict(torch.load(os.path.join(args.save_path,
                                                      'VAEAD_me_{}_{}_{}_{}_{}{}.pth'.format(
                                                          # int(args.exp) if args.exp == int(args.exp) else args.exp,
                                                          args.exp,
                                                          args.od,
                                                          args.model_name,
                                                          args.y_dim,
                                                          model_name_tool(args),
                                                          fold + 1))))
        model.to(device)

        T_obs, p_value, T_perm = eval_mmd_p(model, train_loader, test_loader, device)
        print("T_obs=", T_obs, "p=", p_value)

        MMDs[fold] = T_obs
        ps[fold] = p_value

    print(' '.join([f"{x:.3f}" for x in MMDs]))
    print(' '.join([f"{x:.3f}" for x in ps]))

