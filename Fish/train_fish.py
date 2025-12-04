import numpy as np
import os

device_list = [2]
os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(str(x) for x in device_list)

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.autograd import Variable
import random
import itertools
import argparse

from Fish.algorithm import Fish
from Fish.fast_data_loader import InfiniteDataLoader
from sklearn.metrics import confusion_matrix
import warnings

warnings.filterwarnings("ignore", category=UserWarning, message="Using a non-full backward hook.*")

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


def pre_train(args, algorithm: Fish, dataloaders):
    algorithm.train()
    print(algorithm.pre_update(dataloaders, device))


def train(args, algorithm: Fish, dataloaders):
    algorithm.train()
    print(algorithm.update(dataloaders, device))


def valid(args, algorithm: Fish, data_loader):
    algorithm.eval()
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

        pred_logits = algorithm.predict(ppg_target, eda_target)
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
    elif args.dataset_name == "verbio":
        return ""
    elif args.dataset_name == "can_stress":
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
    parser.add_argument('--total_step', type=int, default=300)
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--LR', type=float, default=1e-3)
    parser.add_argument('--weight_decay', type=float, default=5e-4)
    parser.add_argument('--save_path', type=str, default="./checkpoints")
    parser.add_argument('--y_dim', type=int, default=2)
    parser.add_argument('--h_dim', type=int, default=256)
    parser.add_argument('--x_dim', type=int, default=256)
    parser.add_argument('--meta_lr', type=float, default=0.1)
    parser.add_argument('--dataset_name', type=str, default='wesad', choices=['wesad', 'ubfc_phys', 'can_stress', 'verbio'])
    parser.add_argument('--ubfc_phys_task', type=int, default=3, choices=[2, 3])
    args = parser.parse_args()

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

    best_f1 = [0 for i in range(args.k)]
    best_ACC = [0 for i in range(args.k)]

    for fold in range(args.k):
        print("Fold {0}".format(fold + 1))
        train_p = train_subject[fold]
        valid_p = valid_subject[fold]
        test_p = test_subject[fold]

        algorithm = Fish(args.x_dim, args.h_dim, args.y_dim, args)
        algorithm = algorithm.to(device)

        # Loss
        criterion = nn.CrossEntropyLoss()
        # 优化参数

        bestf1_ever = 0.0

        if args.dataset_name == 'wesad':
            domain_datasets = [WESAD_dataset([person]) for person in train_p]
        elif args.dataset_name == 'ubfc_phys':
            if args.y_dim == 2:
                domain_datasets = [UBFC_Phys_dataset([person], binary=True, task=args.ubfc_phys_task) for person in
                                   train_p]


            else:
                domain_datasets = [UBFC_Phys_dataset([person], binary=False) for person in train_p]
        elif args.dataset_name == 'can_stress':
            domain_datasets = [CAN_STRESS_dataset([person]) for person in train_p]
        elif args.dataset_name == 'verbio':
            domain_datasets = [VerBIO_dataset([person]) for person in train_p]

        domain_loaders = []
        for dataset in domain_datasets:
            loader = DataLoader(
                dataset=dataset,
                # weights=None,
                batch_size=args.batch_size,
                # num_workers=1,
            )
            domain_loaders.append(loader)

        valid_loader_list = []
        for i, valid_p_one in enumerate(valid_p):
            if args.dataset_name == 'wesad':
                valid_dataset = WESAD_dataset([valid_p_one], binary=True if args.y_dim == 2 else False)
            elif args.dataset_name == 'ubfc_phys':
                if args.y_dim == 2:
                    valid_dataset = UBFC_Phys_dataset([valid_p_one], binary=True, task=args.ubfc_phys_task)
                else:
                    valid_dataset = UBFC_Phys_dataset([valid_p_one], binary=False)
            elif args.dataset_name == 'can_stress':
                valid_dataset = CAN_STRESS_dataset([valid_p_one])
            elif args.dataset_name == 'verbio':
                valid_dataset = VerBIO_dataset([valid_p_one])
            valid_loader = DataLoader(valid_dataset, batch_size=args.batch_size, shuffle=True)
            valid_loader_list.append(valid_loader)

        for p_epoch in range(args.total_step - args.epoch):
            # minibatches = []
            # for loader in domain_loaders:
            #     batch = next(loader)
            #     x, y = batch
            #     minibatches.append(((x[0].to(device), x[1].to(device)), y.to(device)))
            pre_train(args, algorithm, domain_loaders)

        for epoch in range(args.epoch):
            epoch_train_loss_sum, epoch_train_acc_sum, epoch_train_sampler_sum, epoch_valid_loss_sum, epoch_valid_acc_sum, epoch_valid_sampler_sum = 0, 0, 0, 0, 0, 0
            epoch_train_cm_sum, epoch_valid_cm_sum = 0.0, 0.0

            # minibatches = []
            # for loader in domain_loaders:
            #     batch = next(loader)
            #     x, y = batch
            #     minibatches.append(((x[0].to(device), x[1].to(device)), y.to(device)))

            train(args, algorithm, domain_loaders)
            # valid_loss, valid_correct, valid_cm = valid(args, algorithm, valid_loader)
            for i, valid_p_one in enumerate(valid_p):
                valid_loader = valid_loader_list[i]
                valid_loss, valid_correct, valid_cm = valid(args, algorithm, valid_loader)

                epoch_valid_loss_sum += valid_loss
                epoch_valid_acc_sum += valid_correct
                epoch_valid_cm_sum += valid_cm
                epoch_valid_sampler_sum += len(valid_loader.sampler)

            # epoch_train_acc = epoch_train_acc_sum / epoch_train_sampler_sum * 100
            epoch_valid_acc = epoch_valid_acc_sum / epoch_valid_sampler_sum * 100
            # _, epoch_train_f1 = f1_score_from_confusion_matrix(epoch_train_cm_sum)
            _, epoch_valid_f1 = f1_score_from_confusion_matrix(epoch_valid_cm_sum)
            # epoch_train_loss = epoch_train_loss_sum / epoch_train_sampler_sum
            epoch_valid_loss = epoch_valid_loss_sum / epoch_valid_sampler_sum

            print(
                "Epoch: {}, Avg_valid_loss: {}, Avg_valid_acc: {}%, Avg_valid_f1: {}".format(
                    epoch, epoch_valid_loss, epoch_valid_acc,
                    epoch_valid_f1))

            if epoch_valid_f1 >= bestf1_ever:
                bestf1_ever = epoch_valid_f1
                best_f1[fold] = epoch_valid_f1
                best_ACC[fold] = epoch_valid_acc
                torch.save(algorithm.network.state_dict(), os.path.join(args.save_path,
                                                                        'Fish_{0}_{1}_{2}{3}.pth'.format(
                                                                            args.dataset_name,
                                                                            args.y_dim,
                                                                            model_name_tool(args),
                                                                            fold + 1)))
    print(np.average(best_ACC), np.std(best_ACC))
    print(np.average(best_f1), np.std(best_f1))
