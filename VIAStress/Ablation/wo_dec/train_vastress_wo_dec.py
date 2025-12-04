import numpy as np
import os

device_list = [3]
os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(str(x) for x in device_list)

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.autograd import Variable
import random
import itertools
import argparse

from VIAStress.model import Model, BVPEncoder, BVPDecoder, EDAEncoder, EDADecoder, ProjectHead, EncoderTrans
from VIAStress.loss import SupConLoss
from sklearn.metrics import confusion_matrix
from VIAStress.param_tool import out_dim_tool, exp_param_tool, distance_p

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


def train(args, model, bvp_proj, eda_proj, mlp_b2e, mlp_e2b, data_loader, criterion, criterion_contrast, optimizer):
    model.train()
    train_loss, train_correct, train_cm = 0.0, 0.0, None
    # ppg_context, eda_context, y_context_one_hot = None, None, None
    for j, data in enumerate(data_loader):
        labels, ppg, scr, scl, eda, peak = data
        # eda = torch.cat((scr, scl), dim=1)
        labels, ppg, eda, peak = Variable(labels).to(device), Variable(ppg).to(device), Variable(eda).to(
            device), Variable(peak).to(device)

        batch_size, _, _ = eda.shape

        pred_logits, bvp_z, eda_z, recon_bvp, mu_bvp, logvar_bvp, recon_eda, mu_eda, logvar_eda = model(ppg, eda)

        loss = criterion(pred_logits, labels)

        dim_ = int(args.x_dim / 2)
        bvp_emd_proj = bvp_proj(bvp_z[:, :dim_])
        # bvp_emd_proj = bvp_proj(bvp_z)
        eda_emd_proj = eda_proj(eda_z[:, :dim_])
        # eda_emd_proj = eda_proj(eda_z)
        emd_proj = torch.stack([bvp_emd_proj, eda_emd_proj], dim=1)
        loss_contrast = criterion_contrast(emd_proj, labels)
        loss += loss_contrast * 0.1

        # Feature Splitting with Distance
        loss_e = 0
        if distance_p(args) == 2:
            loss_e += torch.exp(-exp_param_tool(args) * F.mse_loss(bvp_z[:, :dim_], bvp_z[:, dim_:]))
            loss_e += torch.exp(-exp_param_tool(args) * F.mse_loss(eda_z[:, :dim_], eda_z[:, dim_:]))
        else:
            loss_e += torch.exp(-exp_param_tool(args) * F.l1_loss(bvp_z[:, :dim_], bvp_z[:, dim_:]))
            loss_e += torch.exp(-exp_param_tool(args) * F.l1_loss(eda_z[:, :dim_], eda_z[:, dim_:]))
        loss_e = loss_e / 2
        loss += loss_e * 0.1

        # Cross-modal Translation
        bvp_emd_t = mlp_e2b(eda_z[:, :dim_])
        eda_emd_t = mlp_b2e(bvp_z[:, :dim_])
        e2b_loss = torch.mean(
            torch.norm(bvp_emd_t - bvp_z[:, :dim_] / torch.norm(bvp_z[:, :dim_], dim=1, keepdim=True), dim=1))
        b2e_loss = torch.mean(
            torch.norm(eda_emd_t - eda_z[:, :dim_] / torch.norm(eda_z[:, :dim_], dim=1, keepdim=True), dim=1))
        loss += (e2b_loss + b2e_loss) / 2 * 0.1

        optimizer.zero_grad()
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
        labels, ppg, scr, scl, eda, peak = data
        # eda = torch.cat((scr, scl), dim=1)
        labels, ppg, eda, peak = Variable(labels).to(device), Variable(ppg).to(device), Variable(eda).to(
            device), Variable(peak).to(device)

        batch_size, _, _ = eda.shape

        pred_logits, bvp_z, eda_z, recon_bvp, mu_bvp, logvar_bvp, recon_eda, mu_eda, logvar_eda = model(ppg, eda)
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
    parser.add_argument('--LR', type=float, default=0.0001)
    parser.add_argument('--weight_decay', type=float, default=0.0005)
    parser.add_argument('--save_path', type=str, default="./checkpoints")
    parser.add_argument('--pre_save_path', type=str, default="./checkpoints_vaepre")
    parser.add_argument('--temp', type=float, default=0.1)
    parser.add_argument('--y_dim', type=int, default=2)
    parser.add_argument('--r_dim', type=int, default=256)
    parser.add_argument('--h_dim', type=int, default=256)
    parser.add_argument('--x_dim', type=int, default=256)
    parser.add_argument('--dataset_name', type=str, default='wesad', choices=['wesad', 'ubfc_phys', 'verbio'])
    parser.add_argument('--ubfc_phys_task', type=int, default=2, choices=[2, 3])
    parser.add_argument('--pre_train', type=str2bool, default=True)
    args = parser.parse_args()

    if args.dataset_name == 'wesad':
        from VIAStress_wo_MMDG.WESAD_dataset_multi import WESAD_dataset
        from data.WESAD.Uniform_distribution_person_multi import train_subject, valid_subject, test_subject
    elif args.dataset_name == 'ubfc_phys':
        from VIAStress_wo_MMDG.UBFC_Phys_dataset_multi import UBFC_Phys_dataset
        from data.UBFC_Phys.Uniform_distribution_person_multi import train_subject, valid_subject, test_subject
    elif args.dataset_name == 'verbio':
        from VIAStress_wo_MMDG.VerBIO_dataset_multi import VerBIO_dataset
        from data.VerBIO.Uniform_distribution_person_multi import train_subject, valid_subject, test_subject

    best_f1 = [0 for i in range(args.k)]
    best_ACC = [0 for i in range(args.k)]

    for fold in range(args.k):
        print("Fold {0}".format(fold + 1))
        # if fold == 0:
        #     fold = 1
        # if fold == 1:
        #     fold = 0
        train_p = train_subject[fold]
        valid_p = valid_subject[fold]
        test_p = test_subject[fold]
        if args.pre_train:
            bvp_encoder = BVPEncoder(r_dim=args.r_dim, x_dim=128)
            bvp_encoder.load_state_dict(torch.load(os.path.join(args.pre_save_path,
                                                                'Pre_VPDAD_encoder_PPG_{0}_{1}_{2}{3}.pth'.format(
                                                                    args.dataset_name,
                                                                    args.y_dim,
                                                                    model_name_tool(args),
                                                                    fold + 1))))

            bvp_decoder = BVPDecoder(r_dim=args.r_dim)
            bvp_decoder.load_state_dict(torch.load(os.path.join(args.pre_save_path,
                                                                'Pre_VPDAD_decoder_PPG_{0}_{1}_{2}{3}.pth'.format(
                                                                    args.dataset_name,
                                                                    args.y_dim,
                                                                    model_name_tool(args),
                                                                    fold + 1))))

            eda_encoder = EDAEncoder(r_dim=args.r_dim, x_dim=128)
            eda_encoder.load_state_dict(torch.load(os.path.join(args.pre_save_path,
                                                                'Pre_VAEAD_encoder_EDA_{0}_{1}_{2}{3}.pth'.format(
                                                                    args.dataset_name,
                                                                    args.y_dim,
                                                                    model_name_tool(args),
                                                                    fold + 1))))

            eda_decoder = EDADecoder(r_dim=args.r_dim)
            eda_decoder.load_state_dict(torch.load(os.path.join(args.pre_save_path,
                                                                'Pre_VAEAD_decoder_EDA_{0}_{1}_{2}{3}.pth'.format(
                                                                    args.dataset_name,
                                                                    args.y_dim,
                                                                    model_name_tool(args),
                                                                    fold + 1))))

            model = Model(y_dim=args.y_dim, r_dim=args.r_dim, x_dim=args.x_dim, h_dim=args.h_dim,
                          bvp_encoder_vae=bvp_encoder, bvp_decoder_vae=bvp_decoder,
                          eda_encoder_vae=eda_encoder, eda_decoder_vae=eda_decoder)
        else:
            model = Model(y_dim=args.y_dim, r_dim=args.r_dim, x_dim=args.x_dim, h_dim=args.h_dim)
        # model = torch.nn.DataParallel(model)
        model = model.to(device)
        # in_dim = int(args.x_dim / 2)
        bvp_proj = ProjectHead(128, 128, out_dim_tool(args)).to(device)
        eda_proj = ProjectHead(128, 128, out_dim_tool(args)).to(device)
        mlp_b2e = EncoderTrans(128, 128, 128).to(device)
        mlp_e2b = EncoderTrans(128, 128, 128).to(device)

        # Loss
        criterion = nn.CrossEntropyLoss()
        criterion_contrast = SupConLoss(temperature=args.temp)
        # 优化参数
        # optimizer = optim.Adam(
        #     model.parameters(),
        #     lr=args.LR,
        #     weight_decay=args.weight_decay
        # )
        optimizer = torch.optim.Adam([
            {'params': model.feature_cnn.parameters(), 'lr': 1e-3},  # 设置较高的学习率
            {'params': model.bvp_encoder_vae.parameters(), 'lr': 1e-4},  # 设置较低的学习率
            {'params': model.bvp_decoder_vae.parameters(), 'lr': 1e-4},
            {'params': model.eda_encoder_vae.parameters(), 'lr': 1e-4},
            {'params': model.eda_decoder_vae.parameters(), 'lr': 1e-4},
            {'params': model.classifier.parameters(), 'lr': 1e-3},
            {'params': bvp_proj.parameters(), 'lr': 1e-3},
            {'params': eda_proj.parameters(), 'lr': 1e-3},
            {'params': mlp_b2e.parameters(), 'lr': 1e-3},
            {'params': mlp_e2b.parameters(), 'lr': 1e-3},
        ], weight_decay=args.weight_decay)

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
            elif args.dataset_name == 'verbio':
                valid_dataset = VerBIO_dataset([valid_p_one])
            valid_loader = DataLoader(valid_dataset, batch_size=args.batch_size, shuffle=True)
            valid_loader_list.append(valid_loader)

        for epoch in range(args.epoch):
            epoch_train_loss_sum, epoch_train_acc_sum, epoch_train_sampler_sum, epoch_valid_loss_sum, epoch_valid_acc_sum, epoch_valid_sampler_sum = 0, 0, 0, 0, 0, 0
            epoch_train_cm_sum, epoch_valid_cm_sum = 0.0, 0.0
            for i, train_p_one in enumerate(train_p):
                train_loader = train_loader_list[i]
                train_loss, train_correct, train_cm = train(args, model, bvp_proj, eda_proj, mlp_b2e, mlp_e2b,
                                                            train_loader, criterion, criterion_contrast, optimizer)
                # train_loss = train_loss / len(train_loader.sampler)
                # train_acc = train_correct / len(train_loader.sampler) * 100
                epoch_train_loss_sum += train_loss
                epoch_train_acc_sum += train_correct
                epoch_train_cm_sum += train_cm
                epoch_train_sampler_sum += len(train_loader.sampler)

            for i, valid_p_one in enumerate(valid_p):
                valid_loader = valid_loader_list[i]
                valid_loss, valid_correct, valid_cm = valid(args, model, valid_loader)

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
                                                            'VAStress_wo_dec_{0}_{1}_{2}{3}.pth'.format(
                                                                args.dataset_name,
                                                                args.y_dim,
                                                                model_name_tool(args),
                                                                fold + 1)))
                # bvp_proj = ProjectHead().to(device)
                # eda_proj = ProjectHead().to(device)
                # mlp_b2e = EncoderTrans().to(device)
                # mlp_e2b = EncoderTrans().to(device)
                torch.save({"bvp_proj": bvp_proj.state_dict(), "eda_proj": eda_proj.state_dict(),
                            "mlp_b2e": mlp_b2e.state_dict(), "mlp_e2b": mlp_e2b.state_dict()},
                           os.path.join(args.save_path,
                                        'VAStress_wo_dec_pt_{0}_{1}_{2}{3}.pth'.format(
                                            args.dataset_name,
                                            args.y_dim,
                                            model_name_tool(args),
                                            fold + 1)))
    print(np.average(best_ACC), np.std(best_ACC))
    print(np.average(best_f1), np.std(best_f1))
