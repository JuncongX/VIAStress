# python confusion_matrix_vastress_ab.py --dataset_name wesad --k 5 --y_dim 2
# python confusion_matrix_vastress_ab.py --dataset_name ubfc_phys --k 7 --y_dim 2 --ubfc_phys_task 2
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
import argparse
import os
import random
import numpy as np
import torch
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


def f1_accuracy(args, network, loader, class_num, device):
    cm = None

    network.eval()
    with torch.no_grad():
        for labels, ppg, scr, scl, eda, peak in loader:
            labels, ppg, eda = Variable(labels).to(device), Variable(ppg).to(device), Variable(eda).to(device)
            if args.ab == "only_ppg":
                out = network(ppg)
            elif args.ab == "only_eda":
                out = network(eda)
            else:
                out = network(ppg, eda)
            p = out[0]
            scores, predictions = torch.max(p.data, 1)
            if cm is None:
                cm = confusion_matrix(labels.cpu().numpy(), predictions.cpu().numpy(),
                                      labels=[i for i in range(class_num)])
            else:
                cm += confusion_matrix(labels.cpu().numpy(), predictions.cpu().numpy(),
                                       labels=[i for i in range(class_num)])
    network.train()

    return cm


def model_name_tool(args):
    if args.dataset_name == "wesad":
        return ""
    elif args.dataset_name == "ubfc_phys":
        if args.y_dim == 3:
            return ""
        else:
            return "_UP{0}_".format(args.ubfc_phys_task)
    elif args.dataset_name == "verbio":
        return ""


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Domain generalization')

    parser.add_argument('--k', type=int, default=5)
    parser.add_argument('--data_dir', default='./data/', type=str)
    parser.add_argument('--dataset_name', type=str, default='wesad', choices=['wesad', 'ubfc_phys', 'verbio'])
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
    parser.add_argument('--z_dim', type=int, default=256)
    parser.add_argument('--h_dim', type=int, default=256)
    parser.add_argument('--r_dim', type=int, default=256)
    parser.add_argument('--ab', type=str, default="pre")

    parser.add_argument('--ubfc_phys_task', type=int, default=3, choices=[2, 3])

    args = parser.parse_args()

    if args.ab in ["pre", "wo_pre"]:
        from VIAStress_wo_MMDG.model_pre import Model
    elif args.ab == "r_ae":
        from VIAStress.Ablation.R_AE.R_AE import Model
    elif args.ab == "r_rp":
        from VIAStress.Ablation.R_RP.R_RP import Model
    elif args.ab in ["wo_Lsc", "wo_Ldl", "wo_Lct", "wo_dec", "wo_Pre", "r_SimMMDG", "r_l2"]:
        from VIAStress.model import Model
    elif args.ab == "me":
        from VIAStress.model import Model
    elif args.ab == "r_metaage":
        from VIAStress.bin.R_MetaAge.R_MetaAge import Model
    elif args.ab == "r_wcn":
        from VIAStress.Ablation.R_WCN.R_WCN import Model
    elif args.ab == "wo_ad":
        from VIAStress.Ablation.wo_AD.wo_AD import Model
    elif args.ab == "only_eda":
        from VIAStress.Ablation.only_EDA.only_EDA import Model
    elif args.ab == "only_ppg":
        from VIAStress.Ablation.only_PPG.only_PPG import Model
    elif args.ab == "she":
        from VIAStress.Ablation.share_encoder.SHE import Model
    elif args.ab == "wo_game":
        from VIAStress.Ablation.wo_GaMe.wo_GaMe import Model
    elif args.ab == "wo_gate":
        from VIAStress.Ablation.memory.wo_gate import Model
    elif args.ab == "wo_memory":
        from VIAStress.Ablation.memory.wo_memory import Model
    # elif args.ab == "1s":
    #     from StressMeta_me.Ablation.sample_1.model import Model
    # elif args.ab == "10s":
    #     from StressMeta_me.Ablation.sample_10.model import Model

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
        from data.WESAD.Uniform_distribution_person_multi import test_subject
    elif args.dataset_name == 'ubfc_phys':
        from VIAStress_wo_MMDG.UBFC_Phys_dataset_multi import UBFC_Phys_dataset
        from data.UBFC_Phys.Uniform_distribution_person_multi import test_subject
    elif args.dataset_name == 'verbio':
        from VIAStress_wo_MMDG.VerBIO_dataset_multi import VerBIO_dataset
        from data.VerBIO.Uniform_distribution_person_multi import test_subject

    f1s = [0 for i in range(args.k)]
    ACCs = [0 for i in range(args.k)]
    Recalls = [0 for i in range(args.k)]

    for fold in range(args.k):
        print(f"Fold:{fold}")
        if args.ab == "r_metaage":
            model = Model(y_dim=args.y_dim, r_dim=args.r_dim, x_dim=args.x_dim, h_dim=args.h_dim, mlp_dim=[(args.h_dim + args.r_dim), 2 * args.h_dim, args.h_dim])
        else:
            model = Model(y_dim=args.y_dim, r_dim=args.r_dim, x_dim=args.x_dim, h_dim=args.h_dim)
        model.load_state_dict(torch.load(os.path.join(args.save_path,
                                                      'VAStress_{4}_{0}_{1}_{2}{3}.pth'.format(
                                                          args.dataset_name,
                                                          args.y_dim,
                                                          model_name_tool(args),
                                                          fold + 1, args.ab))))
        model.to(device)

        test_p = test_subject[fold]

        confusion_mats = None
        for i, test_p_one in enumerate(test_p):

            if args.dataset_name == 'wesad':
                test_dataset = WESAD_dataset([test_p_one], binary=True if args.y_dim == 2 else False)
            elif args.dataset_name == 'ubfc_phys':
                if args.y_dim == 2:
                    test_dataset = UBFC_Phys_dataset([test_p_one], binary=True, task=args.ubfc_phys_task)
                else:
                    test_dataset = UBFC_Phys_dataset([test_p_one], binary=False)
            elif args.dataset_name == 'verbio':
                test_dataset = VerBIO_dataset([test_p_one])
            test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=True)
            cm = f1_accuracy(args, model, test_loader, args.y_dim, device)
            if confusion_mats is None:
                confusion_mats = cm
            else:
                confusion_mats += cm
        acc, f1, recall = f1_score_from_confusion_matrix(confusion_mats, args.y_dim)
        print(f"Avg_valid_acc: {acc}%, Avg_valid_f1: {f1}, Avg_valid_recall: {recall}")

        f1s[fold] = f1
        ACCs[fold] = acc
        Recalls[fold] = recall

    print(np.average(ACCs), np.std(ACCs))
    print(np.average(f1s), np.std(f1s))
    print(np.average(Recalls), np.std(Recalls))
