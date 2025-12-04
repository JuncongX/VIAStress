# nohup python -u pretrain_edaae.py > PREAEAD_EDA_w.out &
# nohup python -u pretrain_edaae.py --dataset_name ubfc_phys --k 7 --ubfc_phys_task 2 > PREAEAD_EDA_u2.out &
# nohup python -u pretrain_edaae.py --dataset_name ubfc_phys --k 7 --ubfc_phys_task 3 > PREAEAD_EDA_u3.out &

import numpy as np
import os

device_list = [3]
os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(str(x) for x in device_list)

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
from torch.autograd import Variable
import random
import itertools
import argparse
import torch.nn.functional as F

from VIAStress_wo_MMDG.Ablation.R_AE.R_AE import EDAEncoder, EDADecoder
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


class Loss(nn.Module):
    def __init__(self):
        super(Loss, self).__init__()
        self.l2 = nn.MSELoss(reduction="mean")

    def forward(self,
                recon_signal, signal
                ):
        # l1 = self.L1(recon_signal, signal)
        l2 = self.l2(recon_signal, signal)

        return l2



def train(args, encoder, decoder, data_loader, criterion, optimizer):
    encoder.train()
    decoder.train()
    train_loss = 0.0
    # ppg_context, eda_context, y_context_one_hot = None, None, None
    for j, data in enumerate(data_loader):
        labels, _, scr, scl, eda, _ = data
        # eda = torch.cat((scr, scl), dim=1)
        signal = Variable(eda).to(device)

        batch_size, _, _ = eda.shape

        z = encoder(signal)
        recon_signal = decoder(z)
        optimizer.zero_grad()
        loss = criterion(recon_signal, signal)
        loss.backward()
        optimizer.step()
        train_loss += loss.item() * batch_size

    return train_loss


def pearson_corr(pred, target):
    for i in range(pred.shape[0]):
        a = pred[i, :]
        b = target[i, :]

        sum_x = torch.sum(a)  # x
        sum_y = torch.sum(b)  # y
        sum_xy = torch.sum(torch.mul(a, b))  # xy
        sum_x2 = torch.sum(torch.mul(a, a))  # x^2
        sum_y2 = torch.sum(torch.mul(b, b))  # y^2
        N = pred.shape[-1]
        pearson = (N * sum_xy - sum_x * sum_y) / (
            torch.sqrt((N * sum_x2 - sum_x * sum_x) * (N * sum_y2 - sum_y * sum_y)))
    return pearson.mean()  # 取 batch 维度的均值


def valid(args, encoder, decoder, data_loader, criterion):
    encoder.eval()
    decoder.eval()
    all_preds, all_targets = [], []

    for j, data in enumerate(data_loader):
        labels, _, scr, scl, eda, _ = data
        # eda = torch.cat((scr, scl), dim=1)

        signal = Variable(eda).to(device)

        batch_size, _, _ = eda.shape

        z = encoder(signal)
        recon_signal = decoder(z)
        all_preds.append(recon_signal)
        all_targets.append(signal)

    # 拼接所有 batch
    all_preds = torch.cat(all_preds, dim=0)
    all_targets = torch.cat(all_targets, dim=0)

    valid_loss = criterion(all_preds, all_targets).item()

    return valid_loss, pearson_corr(all_preds, all_targets).item()


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
    # parser.add_argument('--epoch', type=int, default=150)
    parser.add_argument('--k', type=int, default=5)
    parser.add_argument('--epoch', type=int, default=200)
    parser.add_argument('--fps', type=int, default=64)
    parser.add_argument('--batch_size', type=int, default=120)
    parser.add_argument('--LR', type=float, default=0.0001)
    parser.add_argument('--weight_decay', type=float, default=0.0005)
    parser.add_argument('--save_path', type=str, default="./checkpoints_vaepre")
    parser.add_argument('--r_dim', type=int, default=256)
    parser.add_argument('--h_dim', type=int, default=256)
    parser.add_argument('--x_dim', type=int, default=128)
    parser.add_argument('--y_dim', type=int, default=2)
    parser.add_argument('--dataset_name', type=str, default='wesad', choices=['wesad', 'ubfc_phys', 'road'])
    parser.add_argument('--ubfc_phys_task', type=int, default=2, choices=[2, 3])

    args = parser.parse_args()

    # from data.KEmoCon.KEmoCon_dataset_multi import KEmoCon_dataset
    if args.dataset_name == 'wesad':
        from VIAStress_wo_MMDG.WESAD_dataset_multi import WESAD_dataset
        from data.WESAD.Uniform_distribution_person_multi import train_subject, valid_subject, test_subject
    elif args.dataset_name == 'ubfc_phys':
        from VIAStress_wo_MMDG.UBFC_Phys_dataset_multi import UBFC_Phys_dataset
        from data.UBFC_Phys.Uniform_distribution_person_multi import train_subject, valid_subject, test_subject

    for fold in range(args.k):

        encoder = EDAEncoder(r_dim=args.r_dim, x_dim=args.x_dim)
        decoder = EDADecoder(r_dim=args.r_dim)
        encoder = encoder.to(device)
        decoder = decoder.to(device)

        # Loss
        criterion = Loss()
        # 优化参数

        optimizer = optim.Adam([
            {'params': encoder.parameters(), 'lr': args.LR, 'weight_decay': args.weight_decay},  # net1 的学习率
            {'params': decoder.parameters(), 'lr': args.LR, 'weight_decay': args.weight_decay}  # net2 的学习率
        ])

        best_ever = np.inf
        best_pc = 0.0
        if args.dataset_name == 'wesad':
            train_dataset = WESAD_dataset(train_subject[fold], binary=True if args.y_dim == 2 else False)
            test_dataset = WESAD_dataset(valid_subject[fold], binary=True if args.y_dim == 2 else False)
        elif args.dataset_name == 'ubfc_phys':
            if args.y_dim == 2:
                train_dataset = UBFC_Phys_dataset(train_subject[fold], binary=True, task=args.ubfc_phys_task)
                test_dataset = UBFC_Phys_dataset(valid_subject[fold], binary=True, task=args.ubfc_phys_task)
            else:
                train_dataset = UBFC_Phys_dataset(train_subject[fold], binary=False)
                test_dataset = UBFC_Phys_dataset(valid_subject[fold], binary=False)

        # 创建 DataLoader
        train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)

        for epoch in range(args.epoch):

            train_loss = train(args, encoder, decoder, train_loader, criterion, optimizer)
            train_loss = train_loss / len(train_loader.sampler)

            valid_loss, valid_person = valid(args, encoder, decoder, test_loader, criterion)

            print(
                "Epoch: {}, Avg_train_loss: {}, Avg_valid_loss: {}, Avg_person_corr: {}".format(
                    epoch, train_loss, valid_loss, valid_person))

            if valid_loss <= best_ever and epoch > 100:
                best_ever = valid_loss
                best_pc = valid_person
                torch.save(encoder.state_dict(),
                           os.path.join(args.save_path, 'Pre_AEAD_encoder_EDA_{0}_{1}_{2}{3}.pth'.format(
                               args.dataset_name,
                               args.y_dim,
                               model_name_tool(args),
                               fold + 1)))
                torch.save(decoder.state_dict(),
                           os.path.join(args.save_path, 'Pre_AEAD_decoder_EDA_{0}_{1}_{2}{3}.pth'.format(
                               args.dataset_name,
                               args.y_dim,
                               model_name_tool(args),
                               fold + 1)))
        print(best_ever, best_pc)
