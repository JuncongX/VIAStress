# nohup python -u train_bv_v1.py --y_dim 2 > BV_v1_2.out &
# nohup python -u train_bv_v1.py --y_dim 3 > BV_v1_3.out &
# nohup python -u train_bv_v1.py --y_dim 2 --dataset_name ubfc_phys --ubfc_phys_task 2 --k 7 > BV_v1_2_UP2.out &
# nohup python -u train_bv_v1.py --y_dim 2 --dataset_name ubfc_phys --ubfc_phys_task 3 --k 7 > BV_v1_2_UP3.out &
# nohup python -u train_bv_v1.py --y_dim 3 --dataset_name ubfc_phys --k 7 > BV_v1_3_UP.out &
import numpy as np
import os

device_list = [0]
os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(str(x) for x in device_list)

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.autograd import Variable
import random
import itertools
import argparse

from Baseline.model_v1 import Model
from sklearn.metrics import confusion_matrix

seed = 123
np.random.seed(seed)
torch.manual_seed(seed)  # CPU随机种子确定
torch.cuda.manual_seed(seed)  # GPU随机种子确定
torch.cuda.manual_seed_all(seed)  # 所有的GPU设置种子
torch.backends.cudnn.benchmark = False  # 模型卷积层预先优化关闭
torch.backends.cudnn.deterministic = True  # 确定为默认卷积算法
random.seed(seed)
np.random.seed(seed)
os.environ['PYTHONHASHSEED'] = str(seed)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print("device: ", device)


def train(args, model, data_loader, criterion, optimizer):
    model.train()
    train_loss, train_correct, train_cm = 0.0, 0.0, None
    # ppg_context, eda_context, y_context_one_hot = None, None, None
    for j, data in enumerate(data_loader):
        labels, ppg, scr, scl, eda = data
        # eda = torch.cat((scr, scl), dim=1)
        labels, ppg, eda = Variable(labels).to(device), Variable(ppg).to(device), Variable(eda).to(device)

        batch_size, _, _ = eda.shape

        ppg_target = ppg
        eda_target = eda
        y_target = labels

        pred_logits = model(ppg_target, eda_target)
        optimizer.zero_grad()
        loss = criterion(pred_logits, y_target)
        loss.backward()
        optimizer.step()
        train_loss += loss.item() * batch_size
        scores, predictions = torch.max(pred_logits.data, 1)
        train_correct += (predictions == labels).sum().item()
        if train_cm is None:
            train_cm = confusion_matrix(labels.cpu().numpy(), predictions.cpu().numpy(),
                                        labels=[i for i in range(args.y_dim)])
        else:
            train_cm += confusion_matrix(labels.cpu().numpy(), predictions.cpu().numpy(),
                                         labels=[i for i in range(args.y_dim)])

    return train_loss, train_correct, train_cm


def valid(args, model, data_loader):
    model.eval()
    valid_loss, val_correct, valid_cm = 0.0, 0.0, None
    ppg_context, eda_context, y_context_one_hot = None, None, None
    for j, data in enumerate(data_loader):
        labels, ppg, scr, scl, eda = data
        # eda = torch.cat((scr, scl), dim=1)
        labels, ppg, eda = Variable(labels).to(device), Variable(ppg).to(device), Variable(eda).to(device)

        batch_size, _, _ = eda.shape

        ppg_target = ppg
        eda_target = eda
        y_target = labels
        # y_target_one_hot = nn.functional.one_hot(y_target, num_classes=args.y_dim)

        pred_logits = model(ppg_target, eda_target)
        loss = nn.CrossEntropyLoss()(pred_logits, y_target)

        valid_loss += loss.item() * batch_size
        scores, predictions = torch.max(pred_logits.data, 1)
        val_correct += (predictions == labels).sum().item()
        if valid_cm is None:
            valid_cm = confusion_matrix(labels.cpu().numpy(), predictions.cpu().numpy(),
                                        labels=[i for i in range(args.y_dim)])
        else:
            valid_cm += confusion_matrix(labels.cpu().numpy(), predictions.cpu().numpy(),
                                         labels=[i for i in range(args.y_dim)])

    return valid_loss, val_correct, valid_cm


def f1_score_from_confusion_matrix(cm):
    # 计算每个类别的TP, FP, FN
    TP = np.diag(cm)
    FP = np.sum(cm, axis=0) - TP
    FN = np.sum(cm, axis=1) - TP

    # 计算微平均F1得分
    f1_micro = 2 * np.sum(TP) / (2 * np.sum(TP) + np.sum(FP) + np.sum(FN))

    # 计算宏平均F1得分
    f1_scores = 2 * TP / (2 * TP + FP + FN)
    f1_macro = np.mean(f1_scores)

    return f1_micro, f1_macro


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


if __name__ == '__main__':
    def str2bool(v):
        if isinstance(v, bool):
            return v
        if v.lower() in ('yes', 'true', 't', 'y', '1'):
            return True
        elif v.lower() in ('no', 'false', 'f', 'n', '0'):
            return False
        else:
            raise argparse.ArgumentTypeError('Boolean value expected.')


    parser = argparse.ArgumentParser(description='NPStress')
    parser.add_argument('--k', type=int, default=5, help="KFold")
    # parser.add_argument('--epoch', type=int, default=150)
    parser.add_argument('--epoch', type=int, default=200)
    parser.add_argument('--fps', type=int, default=64)
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--LR', type=float, default=1e-3)
    parser.add_argument('--weight_decay', type=float, default=5e-4)
    parser.add_argument('--save_path', type=str, default="./checkpoints")
    parser.add_argument('--y_dim', type=int, default=2)
    parser.add_argument('--r_dim', type=int, default=256)
    parser.add_argument('--h_dim', type=int, default=256)
    parser.add_argument('--x_dim', type=int, default=256)
    parser.add_argument('--context_num', type=int, default=10)
    parser.add_argument('--dataset_name', type=str, default='wesad', choices=['wesad', 'ubfc_phys', 'road'])
    parser.add_argument('--ubfc_phys_task', type=int, default=3, choices=[2, 3])
    args = parser.parse_args()

    if args.dataset_name == 'wesad':
        from data.WESAD.WESAD_dataset_multi_ae import WESAD_dataset
        from data.WESAD.Uniform_distribution_person_multi import train_subject, valid_subject, test_subject
    elif args.dataset_name == 'ubfc_phys':
        from data.UBFC_Phys.UBFC_Phys_dataset_multi import UBFC_Phys_dataset
        from data.UBFC_Phys.Uniform_distribution_person_multi import train_subject, valid_subject, test_subject
    elif args.dataset_name == 'road':
        from data.AffectiveROAD.AffectiveROAD_dataset_multi import AffectiveROAD_dataset
        from data.AffectiveROAD.Uniform_distribution_person_multi import train_subject, valid_subject, test_subject

    best_f1 = [0 for i in range(args.k)]
    best_ACC = [0 for i in range(args.k)]

    for fold in range(args.k):
        print("Fold {0}".format(fold + 1))
        train_p = train_subject[fold]
        valid_p = valid_subject[fold]
        test_p = test_subject[fold]

        model = Model(y_dim=args.y_dim, z_dim=args.r_dim, h_dim=args.h_dim, x_dim=args.x_dim)
        model = torch.nn.DataParallel(model)
        model = model.to(device)

        # Loss
        criterion = nn.CrossEntropyLoss()
        # 优化参数
        optimizer = optim.Adam(
            model.parameters(),
            lr=args.LR,
            weight_decay=args.weight_decay
        )

        bestf1_ever = 0.0

        if args.dataset_name == 'wesad':
            train_dataset = WESAD_dataset(train_p, binary=True if args.y_dim == 2 else False)
        elif args.dataset_name == 'ubfc_phys':
            if args.y_dim == 2:
                train_dataset = UBFC_Phys_dataset(train_p, binary=True, task=args.ubfc_phys_task)
            else:
                train_dataset = UBFC_Phys_dataset(train_p, binary=False)
        elif args.dataset_name == 'road':
            train_dataset = AffectiveROAD_dataset(train_p, binary=True if args.y_dim == 2 else False)
        train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)

        if args.dataset_name == 'wesad':
            valid_dataset = WESAD_dataset(valid_p, binary=True if args.y_dim == 2 else False)
        elif args.dataset_name == 'ubfc_phys':
            if args.y_dim == 2:
                valid_dataset = UBFC_Phys_dataset(valid_p, binary=True, task=args.ubfc_phys_task)
            else:
                valid_dataset = UBFC_Phys_dataset(valid_p, binary=False)
        elif args.dataset_name == 'road':
            valid_dataset = AffectiveROAD_dataset(valid_p, binary=True if args.y_dim == 2 else False)
        valid_loader = DataLoader(valid_dataset, batch_size=args.batch_size, shuffle=True)

        for epoch in range(args.epoch):
            epoch_train_loss_sum, epoch_train_acc_sum, epoch_train_sampler_sum, epoch_valid_loss_sum, epoch_valid_acc_sum, epoch_valid_sampler_sum = 0, 0, 0, 0, 0, 0
            epoch_train_cm_sum, epoch_valid_cm_sum = 0.0, 0.0

            train_loss, train_correct, train_cm = train(args, model, train_loader, criterion, optimizer)
            valid_loss, valid_correct, valid_cm = valid(args, model, valid_loader)

            epoch_train_acc = train_correct / len(train_loader.sampler)
            epoch_valid_acc = valid_correct / len(valid_loader.sampler)
            _, epoch_train_f1 = f1_score_from_confusion_matrix(train_cm)
            _, epoch_valid_f1 = f1_score_from_confusion_matrix(valid_cm)
            epoch_train_loss = train_loss / len(train_loader.sampler)
            epoch_valid_loss = valid_loss / len(valid_loader.sampler)

            print(
                "Epoch: {}, Avg_train_loss: {}, Avg_train_acc: {}%, Avg_train_f1: {}, Avg_valid_loss: {}, Avg_valid_acc: {}%, Avg_valid_f1: {}".format(
                    epoch, epoch_train_loss, epoch_train_acc, epoch_train_f1, epoch_valid_loss, epoch_valid_acc,
                    epoch_valid_f1))

            if epoch_valid_f1 >= bestf1_ever:
                bestf1_ever = epoch_valid_f1
                best_f1[fold] = epoch_valid_f1
                best_ACC[fold] = epoch_valid_acc
                torch.save(model.module.state_dict(), os.path.join(args.save_path,
                                                                   'BV_v1_{0}_{1}_{2}{3}.pth'.format(
                                                                       args.dataset_name,
                                                                       args.y_dim,
                                                                       model_name_tool(args),
                                                                       fold + 1)))
    print(np.average(best_ACC), np.std(best_ACC))
    print(np.average(best_f1), np.std(best_f1))
