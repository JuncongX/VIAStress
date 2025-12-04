import numpy as np
import os

device_list = [3]
os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(str(x) for x in device_list)

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.autograd import Variable
import torch.nn.functional as F
import random
import itertools
import argparse

from DANN.model import Model
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


def valid(args, model, data_loader, alpha):
    model.eval()
    valid_loss, val_correct, valid_cm = 0.0, 0.0, None
    ppg_context, eda_context, y_context_one_hot = None, None, None
    for j, data in enumerate(data_loader):
        labels, ppg, scr, scl, eda = data
        # eda = torch.cat((scr, scl), dim=1)
        labels, ppg, eda = Variable(labels).to(device), Variable(ppg).to(device), Variable(eda).to(device)

        batch_size, _, _ = eda.shape

        pred_logits, _ = model(ppg, eda, alpha)
        loss = nn.CrossEntropyLoss()(pred_logits, labels)

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
    f1_scores = 2 * TP / (2 * TP + FP + FN + 1e-8)
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
    parser.add_argument('--fps', type=int, default=64)
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--LR', type=float, default=1e-3)
    parser.add_argument('--weight_decay', type=float, default=5e-4)
    # parser.add_argument('--weight_decay', type=float, default=0.0)
    parser.add_argument('--save_path', type=str, default="./DANN_model")
    parser.add_argument('--y_dim', type=int, default=2)
    parser.add_argument('--r_dim', type=int, default=256)
    parser.add_argument('--h_dim', type=int, default=256)
    parser.add_argument('--x_dim', type=int, default=256)
    parser.add_argument('--dataset_name', type=str, default='wesad',
                        choices=['wesad', 'ubfc_phys', 'can_stress', 'verbio'])
    parser.add_argument('--cross_dataset_name', type=str, default='ubfc_phys',
                        choices=['wesad', 'ubfc_phys', 'can_stress', 'verbio'])
    parser.add_argument('--ubfc_phys_task', type=int, default=3, choices=[2, 3])
    args = parser.parse_args()

    if args.dataset_name == 'wesad':
        from data.WESAD.WESAD_dataset_multi_ae import WESAD_dataset
        from data.WESAD.Uniform_distribution_person_multi import train_subject, valid_subject
    elif args.dataset_name == 'ubfc_phys':
        from data.UBFC_Phys.UBFC_Phys_dataset_multi import UBFC_Phys_dataset
        from data.UBFC_Phys.Uniform_distribution_person_multi import train_subject, valid_subject
    elif args.dataset_name == 'can_stress':
        from data.CAN_STRESS.CAN_STRESS_dataset_multi import CAN_STRESS_dataset
        from data.CAN_STRESS.Uniform_distribution_person_multi_0_4 import train_subject, valid_subject
    elif args.dataset_name == 'verbio':
        from data.VerBIO.VerBIO_dataset_multi import VerBIO_dataset
        from data.VerBIO.Uniform_distribution_person_multi import train_subject, valid_subject

    if args.cross_dataset_name == 'wesad':
        from data.WESAD.WESAD_dataset_multi_ae import WESAD_dataset
        from data.WESAD.Uniform_distribution_person_multi import train_subject as cross_train_subject, \
            valid_subject as cross_valid_subject, test_subject as cross_test_subject
    elif args.cross_dataset_name == 'ubfc_phys':
        from data.UBFC_Phys.UBFC_Phys_dataset_multi import UBFC_Phys_dataset
        from data.UBFC_Phys.Uniform_distribution_person_multi import train_subject as cross_train_subject, \
            valid_subject as cross_valid_subject, test_subject as cross_test_subject
    elif args.cross_dataset_name == 'can_stress':
        from data.CAN_STRESS.CAN_STRESS_dataset_multi import CAN_STRESS_dataset
        from data.CAN_STRESS.Uniform_distribution_person_multi_0_4 import train_subject as cross_train_subject, \
            valid_subject as cross_valid_subject, test_subject as cross_test_subject
    elif args.cross_dataset_name == 'verbio':
        from data.VerBIO.VerBIO_dataset_multi import VerBIO_dataset
        from data.VerBIO.Uniform_distribution_person_multi import train_subject as cross_train_subject, \
            valid_subject as cross_valid_subject, test_subject as cross_test_subject

    test_all_f1 = [0 for j in range(args.k)]
    test_all_ACC = [0 for j in range(args.k)]

    for fold in range(args.k):
        print("Fold {0}".format(fold + 1))
        train_p = train_subject[fold]
        # valid_p = valid_subject[fold]
        test_p = cross_test_subject[0] + cross_valid_subject[0] + cross_train_subject[0]

        temp_f1 = [0 for i in range(len(test_p))]
        temp_ACC = [0 for i in range(len(test_p))]

        valid_domain_loader_list = []
        valid_loader_list = []

        if args.dataset_name == 'wesad':
            train_dataset = WESAD_dataset(train_p, binary=True if args.y_dim == 2 else False)
        elif args.dataset_name == 'ubfc_phys':
            if args.y_dim == 2:
                train_dataset = UBFC_Phys_dataset(train_p, binary=True, task=args.ubfc_phys_task)
            else:
                train_dataset = UBFC_Phys_dataset(train_p, binary=False)
        elif args.dataset_name == 'can_stress':
            train_dataset = CAN_STRESS_dataset(train_p)
        elif args.dataset_name == 'verbio':
            train_dataset = VerBIO_dataset(train_p)
        train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)

        for i, test_p_one in enumerate(test_p):
            if args.cross_dataset_name == 'wesad':
                valid_dataset = WESAD_dataset([test_p_one], binary=True if args.y_dim == 2 else False)
            elif args.cross_dataset_name == 'ubfc_phys':
                if args.y_dim == 2:
                    valid_dataset = UBFC_Phys_dataset([test_p_one], binary=True, task=args.ubfc_phys_task)
                else:
                    valid_dataset = UBFC_Phys_dataset([test_p_one], binary=False)
            elif args.cross_dataset_name == 'can_stress':
                valid_dataset = CAN_STRESS_dataset([test_p_one])
            elif args.cross_dataset_name == 'verbio':
                valid_dataset = VerBIO_dataset([test_p_one])

            valid_size = len(valid_dataset)
            # domain_size = int(0.1 * valid_size)
            domain_size = 10
            eval_size = valid_size - domain_size

            generator = torch.Generator().manual_seed(123)
            valid_domain_dataset, valid_eval_dataset = torch.utils.data.random_split(
                valid_dataset,
                [domain_size, eval_size],
                generator=generator
            )
            valid_domain_loader = DataLoader(valid_domain_dataset, batch_size=args.batch_size, shuffle=True)
            valid_loader = DataLoader(valid_dataset, batch_size=args.batch_size, shuffle=True)

            valid_domain_loader_list.append(valid_domain_loader)
            valid_loader_list.append(valid_loader)

        for i_vp, test_p_one in enumerate(test_p):
            valid_loader = valid_loader_list[i_vp]
            valid_domain_loader = valid_domain_loader_list[i_vp]
            bestf1_ever = 0.0

            # Loss
            loss_class = nn.CrossEntropyLoss()
            loss_domain = nn.CrossEntropyLoss()

            model = Model(y_dim=args.y_dim, z_dim=args.r_dim, x_dim=args.x_dim, h_dim=args.h_dim)
            model = model.to(device)
            # 优化参数
            optimizer = optim.Adam(
                model.parameters(),
                lr=args.LR,
                weight_decay=args.weight_decay
            )

            for epoch in range(args.epoch):
                len_dataloader = min(len(train_loader), len(valid_domain_loader))
                data_source_iter = iter(train_loader)
                data_target_iter = iter(valid_domain_loader)

                p = float(i_vp + epoch * len_dataloader) / args.epoch / len_dataloader
                alpha = 2. / (1. + np.exp(-10 * p)) - 1

                # training model using source data
                data_source = next(data_source_iter)
                s_labels, s_ppg, _, _, s_eda = data_source

                model.zero_grad()
                batch_size = len(s_labels)

                domain_label = torch.zeros(batch_size).long().to(device)
                # eda = torch.cat((scr, scl), dim=1)
                s_labels, s_ppg, s_eda = Variable(s_labels).to(device), Variable(s_ppg).to(device), Variable(s_eda).to(
                    device)

                class_output, domain_output = model(s_ppg, s_eda, alpha)
                err_s_label = loss_class(class_output, s_labels)
                err_s_domain = loss_domain(domain_output, domain_label)

                # training model using target data
                data_target = next(data_target_iter)
                t_labels, t_ppg, _, _, t_eda = data_target

                batch_size = len(t_ppg)
                domain_label = torch.ones(batch_size).long().to(device)

                t_ppg, t_eda = Variable(t_ppg).to(device), Variable(t_eda).to(device)

                _, domain_output = model(t_ppg, t_eda, alpha)
                err_t_domain = loss_domain(domain_output, domain_label)
                err = err_t_domain + err_s_domain + err_s_label
                err.backward()
                optimizer.step()

                valid_loss, valid_correct, valid_cm = valid(args, model, valid_domain_loader, alpha)
                epoch_valid_sampler_sum = len(valid_domain_loader.sampler)
                epoch_valid_acc = valid_correct / epoch_valid_sampler_sum * 100
                _, epoch_valid_f1 = f1_score_from_confusion_matrix(valid_cm)
                epoch_valid_loss = valid_loss / epoch_valid_sampler_sum

                print(
                    "Epoch: {}, Avg_valid_loss: {}, Avg_valid_acc: {}%, Avg_valid_f1: {}".format(
                        epoch, epoch_valid_loss, epoch_valid_acc, epoch_valid_f1))

                if epoch_valid_f1 >= bestf1_ever:
                    bestf1_ever = epoch_valid_f1
                    temp_f1[i_vp] = epoch_valid_f1
                    temp_ACC[i_vp] = epoch_valid_acc
                    torch.save(model.state_dict(), os.path.join(args.save_path,
                                                                'DANN_{4}_{0}_{1}_{2}{3}.pth'.format(
                                                                    args.dataset_name,
                                                                    args.y_dim,
                                                                    model_name_tool(args),
                                                                    fold + 1, test_p_one)))
            model_t = Model(y_dim=args.y_dim, z_dim=args.r_dim, x_dim=args.x_dim, h_dim=args.h_dim)
            model_t.load_state_dict(torch.load(os.path.join(args.save_path,
                                                            'DANN_{4}_{0}_{1}_{2}{3}.pth'.format(
                                                                args.dataset_name,
                                                                args.y_dim,
                                                                model_name_tool(args),
                                                                fold + 1, test_p_one))))
            model_t = model_t.to(device)
            test_loss, test_correct, test_cm = valid(args, model_t, valid_loader, 1)
            test_sampler_sum = len(valid_loader.sampler)
            test_acc = test_correct / test_sampler_sum * 100
            _, test_f1 = f1_score_from_confusion_matrix(test_cm)

            test_all_ACC[fold] = np.mean(test_acc)
            test_all_f1[fold] = np.mean(test_f1)

    print(np.average(test_all_ACC), np.std(test_all_ACC))
    print(np.average(test_all_f1), np.std(test_all_f1))
