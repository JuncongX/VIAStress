# python eval_diff_hf.py --model_name wesad --dataset_name wesad --k 5 --y_dim 2
# python eval_diff_hf.py --model_name wesad --dataset_name ubfc_phys --k 5 --y_dim 2 --ubfc_phys_task 2
# python eval_diff_hf.py --model_name wesad --dataset_name verbio --k 5 --y_dim 2

# python eval_diff_hf.py --model_name verbio --dataset_name verbio --k 5 --y_dim 2
# python eval_diff_hf.py --model_name verbio --dataset_name wesad --k 5 --y_dim 2
# python eval_diff_hf.py --model_name verbio --dataset_name ubfc_phys --k 5 --y_dim 2 --ubfc_phys_task 2

# python eval_diff_hf.py --model_name ubfc_phys --dataset_name ubfc_phys --k 7 --y_dim 2 --ubfc_phys_task 2
# python eval_diff_hf.py --model_name ubfc_phys --dataset_name wesad --k 7 --y_dim 2 --ubfc_phys_task 2
# python eval_diff_hf.py --model_name ubfc_phys --dataset_name verbio --k 7 --y_dim 2 --ubfc_phys_task 2

# python eval_diff_hf.py --model_name ubfc_phys --dataset_name ubfc_phys --k 7 --y_dim 2 --ubfc_phys_task 3
# python eval_diff_hf.py --model_name ubfc_phys --dataset_name wesad --k 7 --y_dim 2 --ubfc_phys_task 3
# python eval_diff_hf.py --model_name ubfc_phys --dataset_name verbio --k 7 --y_dim 2 --ubfc_phys_task 3

# python eval_diff_hf.py --model_name can_stress --dataset_name can_stress --k 5 --y_dim 2

# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
import argparse
import os
import random
import numpy as np

from sklearn.metrics import confusion_matrix
from data.MMD_permutation_test import permutation_test_mmd

device_list = [0]
os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(str(x) for x in device_list)


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
    parser.add_argument('--dataset_name', type=str, default='verbio', choices=['wesad', 'ubfc_phys', 'verbio', 'can_stress'])
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

    parser.add_argument('--ubfc_phys_task', type=int, default=3, choices=[2, 3])
    parser.add_argument('--y_dim', type=int, default=2, choices=[2, 3])

    args = parser.parse_args()

    # If we ever want to implement checkpointing, just persist these values
    # every once in a while, and then load them from disk here.

    algorithm_dict = None

    random.seed(args.seed)
    np.random.seed(args.seed)

    from ClassicML.load_data import DataLoder

    MMDs = [0 for i in range(args.k)]
    ps = [0 for i in range(args.k)]

    for fold in range(args.k):
        print(f"Fold:{fold}")

        train_dataset = DataLoder(args.model_name, binary=True if args.y_dim == 2 else False, task=args.ubfc_phys_task)
        test_loader = DataLoder(args.dataset_name, binary=True if args.y_dim == 2 else False, task=args.ubfc_phys_task)

        _, feature_train, _ = train_dataset.load(phase='train', fold=fold)
        if args.dataset_name == args.model_name:
            _, feature_test, _ = test_loader.load(phase='test', fold=fold)
        else:
            _, feature_test, _ = test_loader.load(phase='cross', fold=0)


        T_obs, p_value, T_perm = permutation_test_mmd(feature_train, feature_test)
        print("T_obs=", T_obs, "p=", p_value)

        MMDs[fold] = T_obs
        ps[fold] = p_value

    print(' '.join([f"{x:.3f}" for x in MMDs]))
    print(' '.join([f"{x:.3f}" for x in ps]))

