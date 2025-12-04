# nohup python -u train_btl.py --y_dim 2 > BTL_2.out &
# nohup python -u train_btl.py --y_dim 3 > BTL_3.out &
# nohup python -u train_btl.py --dataset_name ubfc_phys --ubfc_phys_task 2 --y_dim 2 --k 7 > BTL_2_UP2.out &
# nohup python -u train_btl.py --dataset_name ubfc_phys --ubfc_phys_task 3 --y_dim 2 --k 7 > BTL_2_UP3.out &
# nohup python -u train_btl.py --dataset_name can_stress --y_dim 2 > BTL_2_CS.out &
# nohup python -u train_btl.py --dataset_name verbio --y_dim 2 > BTL_2_VB.out &
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

from Baseline.model import Model
from sklearn.metrics import confusion_matrix
from sklearn.metrics import f1_score

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

        pred_logits = model(ppg, eda)
        optimizer.zero_grad()
        loss = criterion(pred_logits, labels)
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


def fine_tune_and_validate(args, model, data_loader, fine_tune_ratio=0.1, epochs=100):
    model.eval()

    all_data = list(data_loader.dataset)
    n_total = len(all_data)
    # n_finetune = int(n_total * fine_tune_ratio)
    n_finetune = 10

    indices = list(range(n_total))
    random.shuffle(indices)
    finetune_indices = indices[:n_finetune]
    eval_indices = indices[n_finetune:]

    fine_tune_subset = torch.utils.data.Subset(data_loader.dataset, finetune_indices)
    eval_subset = torch.utils.data.Subset(data_loader.dataset, eval_indices)

    fine_tune_loader = torch.utils.data.DataLoader(fine_tune_subset, batch_size=data_loader.batch_size, shuffle=True)
    eval_loader = torch.utils.data.DataLoader(eval_subset, batch_size=data_loader.batch_size, shuffle=False)

    for param in model.feature_cnn.parameters():
        param.requires_grad = False

    optimizer = torch.optim.Adam(model.classifier.parameters(), lr=args.LR)
    criterion = nn.CrossEntropyLoss()

    best_acc = 0.0
    best_state_dict = None

    for epoch in range(epochs):
        model.train()
        running_loss, total = 0.0, 0
        all_preds, all_labels = [], []

        for data in fine_tune_loader:
            labels, ppg, scr, scl, eda = data
            labels, ppg, eda = labels.to(device), ppg.to(device), eda.to(device)

            optimizer.zero_grad()
            pred_logits = model(ppg, eda)
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
        # if epoch % 10 == 0:
        #     print(f"[Fine-tune Epoch {epoch + 1}/{epochs}] Loss: {running_loss / total:.4f}, F1: {f1:.4f}")

        if f1 > best_acc:  # 用F1替代acc作为最佳标准
            best_acc = f1
            best_state_dict = model.state_dict()

    print(f"Best fine-tune macro-F1 on 10%: {best_acc:.4f}")

    model.load_state_dict(best_state_dict)
    model.eval()

    valid_loss, val_correct, valid_cm = 0.0, 0.0, None
    criterion = nn.CrossEntropyLoss()

    # for data in eval_loader:
    for data in data_loader:
        labels, ppg, scr, scl, eda = data
        labels, ppg, eda = labels.to(device), ppg.to(device), eda.to(device)
        batch_size = labels.size(0)

        with torch.no_grad():
            pred_logits = model(ppg, eda)
            loss = criterion(pred_logits, labels)
            valid_loss += loss.item() * batch_size

            _, predictions = torch.max(pred_logits, 1)
            val_correct += (predictions == labels).sum().item()

            cm = confusion_matrix(labels.cpu().numpy(), predictions.cpu().numpy(),
                                  labels=[i for i in range(args.y_dim)])
            if valid_cm is None:
                valid_cm = cm
            else:
                valid_cm += cm

    valid_acc = val_correct / len(eval_subset)
    # print(f"Validation Accuracy on 90%: {valid_acc:.4f}")
    print(f"Validation Accuracy on 100% Data: {valid_acc:.4f}")

    return best_acc, val_correct, valid_cm


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
    parser.add_argument('--save_path', type=str, default="./checkpoints")
    parser.add_argument('--y_dim', type=int, default=2)
    parser.add_argument('--r_dim', type=int, default=256)
    parser.add_argument('--h_dim', type=int, default=256)
    parser.add_argument('--x_dim', type=int, default=256)
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

        model = Model(y_dim=args.y_dim, z_dim=args.r_dim, x_dim=args.x_dim, h_dim=args.h_dim)
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

        train_loader_list = []
        valid_loader_list = []

        for i, train_p_one in enumerate(train_p):
            if args.dataset_name == 'wesad':
                train_dataset = WESAD_dataset([train_p_one], binary=True if args.y_dim == 2 else False)
            elif args.dataset_name == 'ubfc_phys':
                if args.y_dim == 2:
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

        for epoch in range(args.epoch):
            epoch_train_loss_sum, epoch_train_acc_sum, epoch_train_sampler_sum, epoch_valid_loss_sum, epoch_valid_acc_sum, epoch_valid_sampler_sum = 0, 0, 0, 0, 0, 0
            epoch_train_cm_sum, epoch_valid_cm_sum = 0.0, 0.0
            for i, train_p_one in enumerate(train_p):
                train_loader = train_loader_list[i]
                train_loss, train_correct, train_cm = train(args, model, train_loader, criterion, optimizer)
                # train_loss = train_loss / len(train_loader.sampler)
                # train_acc = train_correct / len(train_loader.sampler) * 100
                epoch_train_loss_sum += train_loss
                epoch_train_acc_sum += train_correct
                epoch_train_cm_sum += train_cm
                epoch_train_sampler_sum += len(train_loader.sampler)

            for i, valid_p_one in enumerate(valid_p):
                valid_loader = valid_loader_list[i]
                valid_loss, valid_correct, valid_cm = fine_tune_and_validate(args, model, valid_loader)

                epoch_valid_loss_sum += valid_loss
                epoch_valid_acc_sum += valid_correct
                epoch_valid_cm_sum += valid_cm
                epoch_valid_sampler_sum += len(valid_loader.sampler)

            epoch_train_acc = epoch_train_acc_sum / epoch_train_sampler_sum * 100
            epoch_valid_acc = epoch_valid_acc_sum / epoch_valid_sampler_sum * 100
            _, epoch_train_f1 = f1_score_from_confusion_matrix(epoch_train_cm_sum)
            _, epoch_valid_f1 = f1_score_from_confusion_matrix(epoch_valid_cm_sum)
            epoch_train_loss = epoch_train_loss_sum / epoch_train_sampler_sum
            epoch_valid_loss = epoch_valid_loss_sum / epoch_valid_sampler_sum

            print(
                "Epoch: {}, Avg_train_loss: {}, Avg_train_acc: {}%, Avg_train_f1: {}, Avg_valid_loss: {}, Avg_valid_acc: {}%, Avg_valid_f1: {}".format(
                    epoch, epoch_train_loss, epoch_train_acc, epoch_train_f1, epoch_valid_loss, epoch_valid_acc,
                    epoch_valid_f1))

            if epoch_valid_f1 >= bestf1_ever:
                bestf1_ever = epoch_valid_f1
                best_f1[fold] = epoch_valid_f1
                best_ACC[fold] = epoch_valid_acc
                torch.save(model.state_dict(), os.path.join(args.save_path,
                                                                   'BTL_{0}_{1}_{2}{3}.pth'.format(
                                                                       args.dataset_name,
                                                                       args.y_dim,
                                                                       model_name_tool(args),
                                                                       fold + 1)))
    print(np.average(best_ACC), np.std(best_ACC))
    print(np.average(best_f1), np.std(best_f1))
