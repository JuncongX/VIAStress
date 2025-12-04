import torch.nn.functional as F
import matplotlib.pyplot as plt
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
from sklearn.metrics import confusion_matrix

device_list = [0]
os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(str(x) for x in device_list)

def evaluate_vae_reconstruction(model, dataloader, device):
    model.eval()
    total_mse_bvp = 0.0
    total_mse_eda = 0.0
    total_samples_bvp = 0
    total_samples_eda = 0

    with torch.no_grad():

        for labels, bvp, scr, scl, eda, peak in dataloader:
            bvp, eda = bvp.to(device), eda.to(device)

            _, _, _, recon_bvp, _, _, recon_eda, _, _ = model(bvp, eda)

            mse_bvp = F.mse_loss(recon_bvp, bvp, reduction='sum').item()
            mse_eda = F.mse_loss(recon_eda, eda, reduction='sum').item()

            total_mse_bvp += mse_bvp
            total_mse_eda += mse_eda
            total_samples_bvp += bvp.size(0) * bvp.size(2)
            total_samples_eda += eda.size(0) * eda.size(2)

    avg_mse_bvp = total_mse_bvp / total_samples_bvp
    avg_mse_eda = total_mse_eda / total_samples_eda

    print(f"Average MSE for BVP reconstruction: {avg_mse_bvp:.6f}")
    print(f"Average MSE for EDA reconstruction: {avg_mse_eda:.6f}")

    return avg_mse_bvp, avg_mse_eda


def plot_reconstruction(original, reconstructed, signal_type="BVP", sample_idx=0):
    plt.figure(figsize=(10, 4))
    plt.plot(original[sample_idx].cpu().numpy(), label="Original", linestyle='dashed')
    plt.plot(reconstructed[sample_idx].cpu().numpy(), label="Reconstructed", alpha=0.7)
    plt.title(f"{signal_type} Signal Reconstruction")
    plt.xlabel("Time")
    plt.ylabel("Amplitude")
    plt.legend()
    plt.show()


def visualize_reconstructions(model, dataloader, device):
    model.eval()
    with torch.no_grad():
        for labels, ppg, scr, scl, eda, peak in dataloader:
            bvp, eda = ppg.to(device), eda.to(device)
            _, _, _, recon_bvp, _, _, recon_eda, _, _ = model(bvp, eda)

            plot_reconstruction(bvp, recon_bvp, signal_type="BVP")
            plot_reconstruction(eda, recon_eda, signal_type="EDA")
            break  # Only visualize the first batch



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


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Domain generalization')

    parser.add_argument('--k', type=int, default=5)
    parser.add_argument('--data_dir', default='./data/', type=str)
    parser.add_argument('--dataset_name', type=str, default='wesad', choices=['wesad', 'ubfc_phys'])
    parser.add_argument('--batch_size', type=int, default=150)
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
    parser.add_argument('--ab', type=str, default="me")

    parser.add_argument('--ubfc_phys_task', type=int, default=3, choices=[2, 3])

    args = parser.parse_args()

    if args.ab in ["pre", "wo_pre"]:
        from VIAStress_wo_MMDG.model_pre import Model
    elif args.ab == "r_ae":
        from VIAStress_wo_MMDG.Ablation.R_AE.R_AE import Model
    elif args.ab == "r_rp":
        from VIAStress_wo_MMDG.Ablation.R_RP.R_RP import Model
    elif args.ab == "me":
        from VIAStress.model import Model

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
        from data.WESAD.Uniform_distribution_person_multi import train_subject, valid_subject, test_subject
    elif args.dataset_name == 'ubfc_phys':
        from VIAStress_wo_MMDG.UBFC_Phys_dataset_multi import UBFC_Phys_dataset
        from data.UBFC_Phys.Uniform_distribution_person_multi import train_subject, valid_subject, test_subject

    f1s = [0 for i in range(args.k)]
    ACCs = [0 for i in range(args.k)]
    Recalls = [0 for i in range(args.k)]

    for fold in range(args.k):
        print(f"Fold:{fold}")

        if args.dataset_name == 'wesad':
            test_dataset = WESAD_dataset(test_subject[fold], binary=True if args.y_dim == 2 else False)
        elif args.dataset_name == 'ubfc_phys':
            if args.y_dim == 2:
                test_dataset = UBFC_Phys_dataset(test_subject[fold], binary=True, task=args.ubfc_phys_task)
            else:
                test_dataset = UBFC_Phys_dataset(test_subject[fold], binary=False)
        test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=True)

        model = Model(y_dim=args.y_dim, r_dim=args.r_dim, x_dim=args.x_dim, h_dim=args.h_dim)
        model.load_state_dict(torch.load(os.path.join(args.save_path,
                                                      'VAEAD_{4}_{0}_{1}_{2}{3}.pth'.format(
                                                          args.dataset_name,
                                                          args.y_dim,
                                                          model_name_tool(args),
                                                          fold + 1, args.ab))))
        model.to(device)

        evaluate_vae_reconstruction(model, test_loader, device)
        # visualize_reconstructions(model, test_loader, device)
        # print(f"Avg_valid_acc: {acc}%, Avg_valid_f1: {f1}, Avg_valid_recall: {recall}")

    #     f1s[fold] = f1
    #     ACCs[fold] = acc
    #     Recalls[fold] = recall
    #
    # print(np.average(ACCs), np.std(ACCs))
    # print(np.average(f1s), np.std(f1s))
    # print(np.average(Recalls), np.std(Recalls))
