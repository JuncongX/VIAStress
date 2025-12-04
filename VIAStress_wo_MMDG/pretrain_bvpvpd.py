# nohup python -u pretrain_bvpvpd.py --y_dim 2 --epoch 100 > PREVPDAD_PPG_w.out &
# nohup python -u pretrain_bvpvpd.py --y_dim 3 --epoch 100 > PREVPDAD_PPG_w3.out &
# nohup python -u pretrain_bvpvpd.py --dataset_name ubfc_phys --k 7 --ubfc_phys_task 2 --epoch 100 > PREVPDAD_PPG_u2.out &
# nohup python -u pretrain_bvpvpd.py --dataset_name ubfc_phys --k 7 --ubfc_phys_task 3 --epoch 100 > PREVPDAD_PPG_u3.out &
# nohup python -u pretrain_bvpvpd.py --dataset_name can_stress --y_dim 2 --epoch 400 > PREVPDAD_PPG_can_stress.out &
# nohup python -u pretrain_bvpvpd.py --dataset_name verbio --k 5 --epoch 100 > PREVPDAD_PPG_verbio.out &

import numpy as np
import os

device_list = [1]
os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(str(x) for x in device_list)

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subsetza
from torch.autograd import Variable
import random
import itertools
import argparse
import torch.nn.functional as F

from VIAStress_wo_MMDG.model_pre import BVPEncoder, BVPDecoder
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


# class Loss(nn.Module):
#     def __init__(self):
#         super(Loss, self).__init__()
#
#     def forward(self,
#                 recon_signal, signal,
#                 mu_signal, logvar_signal
#                 ):
#         # l1 = self.L1(recon_signal, signal)
#         ce = F.binary_cross_entropy(recon_signal, signal, reduction="mean")
#         # kl = -0.5 * (logvar_signal + 1 - mu_signal ** 2 - torch.exp(logvar_signal))
#         kl = torch.mean(-0.5 * torch.sum(1 + logvar_signal - mu_signal ** 2 - logvar_signal.exp(), dim=1), dim=0)
#         # kl = kl.mean()
#         return ce + kl

class Loss(nn.Module):
    def __init__(self, total_steps=50, kl_max=1.0):
        super(Loss, self).__init__()
        self.l1 = nn.L1Loss(reduction="mean")
        self.total_steps = total_steps
        self.kl_max = kl_max
        self.current_step = 0

    def forward(self,
                recon_signal, signal,
                mu_signal, logvar_signal
                ):
        ce = F.binary_cross_entropy(recon_signal, signal, reduction="mean")

        beta = min(self.kl_max, self.current_step / self.total_steps)
        kl = torch.mean(-0.5 * torch.sum(1 + logvar_signal - mu_signal ** 2 - logvar_signal.exp(), dim=1), dim=0)
        # kl = kl.mean()
        return ce + beta * kl


def reparameterize(mu, logvar):
    std = torch.exp(0.5 * logvar)
    eps = torch.randn_like(std)
    return mu + eps * std


# def pearson_corr(pred, target):
#     for i in range(pred.shape[0]):
#         a = pred[i, :]
#         b = target[i, :]
#
#         sum_x = torch.sum(a)  # x
#         sum_y = torch.sum(b)  # y
#         sum_xy = torch.sum(torch.mul(a, b))  # xy
#         sum_x2 = torch.sum(torch.mul(a, a))  # x^2
#         sum_y2 = torch.sum(torch.mul(b, b))  # y^2
#         N = pred.shape[-1]
#         pearson = (N * sum_xy - sum_x * sum_y) / (
#             torch.sqrt((N * sum_x2 - sum_x * sum_x) * (N * sum_y2 - sum_y * sum_y)))
#     return pearson.mean()  # 取 batch 维度的均值


def train(args, encoder, decoder, data_loader, criterion, optimizer):
    encoder.train()
    decoder.train()
    train_loss = 0.0
    # ppg_context, eda_context, y_context_one_hot = None, None, None
    for j, data in enumerate(data_loader):
        labels, ppg, scr, scl, eda, peak = data
        # eda = torch.cat((scr, scl), dim=1)
        signal = Variable(ppg).to(device)
        peak = Variable(peak).to(device)

        batch_size, _, _ = eda.shape

        mu, logvar = encoder(signal)
        z = reparameterize(mu, logvar)
        recon_signal = decoder(z)
        optimizer.zero_grad()
        loss = criterion(recon_signal, peak, mu, logvar)
        criterion.current_step += 1
        # loss = criterion(recon_signal, signal, mu, logvar)
        loss.backward()
        optimizer.step()
        train_loss += loss.item() * batch_size

    return train_loss


def valid(args, encoder, decoder, data_loader, criterion):
    encoder.eval()
    decoder.eval()
    all_preds, all_targets, all_mu, all_logvar = [], [], [], []

    for j, data in enumerate(data_loader):
        labels, ppg, scr, scl, eda, peak = data
        # eda = torch.cat((scr, scl), dim=1)

        signal = Variable(ppg).to(device)
        peak = Variable(peak).to(device)

        batch_size, _, _ = eda.shape

        mu, logvar = encoder(signal)
        z = reparameterize(mu, logvar)
        recon_signal = decoder(z)
        all_preds.append(recon_signal)
        all_targets.append(peak)
        # all_targets.append(signal)
        all_mu.append(mu)
        all_logvar.append(logvar)

    # 拼接所有 batch
    all_preds = torch.cat(all_preds, dim=0)
    all_targets = torch.cat(all_targets, dim=0)
    all_mu = torch.cat(all_mu, dim=0)
    all_logvar = torch.cat(all_logvar, dim=0)

    valid_loss = criterion(all_preds, all_targets, all_mu, all_logvar).item()

    return valid_loss, F.binary_cross_entropy(all_preds, all_targets, reduction="mean").item()
    # return valid_loss, pearson_corr(all_preds, all_targets).item()


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
    # parser.add_argument('--epoch', type=int, default=150)
    parser.add_argument('--k', type=int, default=5)
    parser.add_argument('--epoch', type=int, default=100)
    parser.add_argument('--fps', type=int, default=64)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--LR', type=float, default=0.0001)
    parser.add_argument('--weight_decay', type=float, default=0.0005)
    parser.add_argument('--save_path', type=str, default="./checkpoints_vaepre")
    parser.add_argument('--r_dim', type=int, default=256)
    parser.add_argument('--h_dim', type=int, default=256)
    parser.add_argument('--x_dim', type=int, default=128)
    parser.add_argument('--y_dim', type=int, default=2)
    parser.add_argument('--dataset_name', type=str, default='wesad', choices=['wesad', 'ubfc_phys', 'can_stress', 'verbio'])
    parser.add_argument('--ubfc_phys_task', type=int, default=2, choices=[2, 3])

    args = parser.parse_args()

    # from data.KEmoCon.KEmoCon_dataset_multi import KEmoCon_dataset
    if args.dataset_name == 'wesad':
        from VIAStress_wo_MMDG.WESAD_dataset_multi import WESAD_dataset
        from data.WESAD.Uniform_distribution_person_multi import train_subject, valid_subject, test_subject
    elif args.dataset_name == 'ubfc_phys':
        from VIAStress_wo_MMDG.UBFC_Phys_dataset_multi import UBFC_Phys_dataset
        from data.UBFC_Phys.Uniform_distribution_person_multi import train_subject, valid_subject, test_subject
    elif args.dataset_name == 'can_stress':
        from VIAStress_wo_MMDG.CAN_STRESS_dataset_multi import CAN_STRESS_dataset
        from data.CAN_STRESS.Uniform_distribution_person_multi_0_4 import train_subject, valid_subject, test_subject
    elif args.dataset_name == 'verbio':
        from VIAStress_wo_MMDG.VerBIO_dataset_multi import VerBIO_dataset
        from data.VerBIO.Uniform_distribution_person_multi import train_subject, valid_subject, test_subject

    for fold in range(args.k):

        encoder = BVPEncoder(r_dim=args.r_dim, x_dim=args.x_dim)
        decoder = BVPDecoder(r_dim=args.r_dim)
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
        best_ce = np.inf
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
        elif args.dataset_name == 'can_stress':
            train_dataset = CAN_STRESS_dataset(train_subject[fold])
            test_dataset = CAN_STRESS_dataset(valid_subject[fold])
        elif args.dataset_name == 'verbio':
            train_dataset = VerBIO_dataset(train_subject[fold])
            test_dataset = VerBIO_dataset(valid_subject[fold])

        # 创建 DataLoader
        train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)

        for epoch in range(args.epoch):

            train_loss = train(args, encoder, decoder, train_loader, criterion, optimizer)
            train_loss = train_loss / len(train_loader.sampler)

            valid_loss, valid_ce = valid(args, encoder, decoder, test_loader, criterion)

            print(
                "Epoch: {}, Avg_train_loss: {}, Avg_valid_loss: {}, Avg_ce: {}".format(
                    epoch, train_loss, valid_loss, valid_ce))
            # if args.dataset_name == "ubfc_phys":
            if valid_loss <= best_ever and epoch > 50:
                best_ever = valid_loss
                best_ce = valid_ce
                torch.save(encoder.state_dict(),
                           os.path.join(args.save_path, 'Pre_VPDAD_encoder_PPG_{0}_{1}_{2}{3}.pth'.format(
                               args.dataset_name,
                               args.y_dim,
                               model_name_tool(args),
                               fold + 1)))
                torch.save(decoder.state_dict(),
                           os.path.join(args.save_path, 'Pre_VPDAD_decoder_PPG_{0}_{1}_{2}{3}.pth'.format(
                               args.dataset_name,
                               args.y_dim,
                               model_name_tool(args),
                               fold + 1)))
            # elif args.dataset_name == "wesad":
            #     if valid_loss <= best_ever and epoch > 50:
            #         best_ever = valid_loss
            #         best_ce = valid_ce
            #         torch.save(encoder.state_dict(),
            #                    os.path.join(args.save_path, 'Pre_VPDAD_encoder_PPG_{0}_{1}_{2}{3}.pth'.format(
            #                        args.dataset_name,
            #                        args.y_dim,
            #                        model_name_tool(args),
            #                        fold + 1)))
            #         torch.save(decoder.state_dict(),
            #                    os.path.join(args.save_path, 'Pre_VPDAD_decoder_PPG_{0}_{1}_{2}{3}.pth'.format(
            #                        args.dataset_name,
            #                        args.y_dim,
            #                        model_name_tool(args),
            #                        fold + 1)))
        print(best_ever, best_ce)
