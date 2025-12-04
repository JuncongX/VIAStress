# python draw_tsne_me.py --dataset_name wesad --k 5 --od 64 --exp 1 --y_dim 2
# python draw_tsne_me.py --dataset_name wesad --k 5 --od 64 --exp 0.5 --y_dim 3
# python draw_tsne_me.py --dataset_name ubfc_phys --k 7 --od 64 --y_dim 2 --exp 1 --ubfc_phys_task 2
# python draw_tsne_me.py --dataset_name ubfc_phys --k 7 --od 64 --y_dim 2 --exp 0.5 --ubfc_phys_task 3
# python draw_tsne_me.py --dataset_name verbio --k 5 --od 64 --exp 1.5 --y_dim 2

import torch
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from tqdm import tqdm
import argparse
import os
from torch.utils.data import DataLoader
from torch.autograd import Variable
import numpy as np

parser = argparse.ArgumentParser(description='Domain generalization')

parser.add_argument('--k', type=int, default=5)
parser.add_argument('--dataset_name', type=str, default='wesad', choices=['wesad', 'ubfc_phys', 'verbio'])
parser.add_argument('--batch_size', type=int, default=128)
parser.add_argument('--save_path', type=str, default="./checkpoints")

parser.add_argument('--x_dim', type=int, default=256)
parser.add_argument('--y_dim', type=int, default=2)
parser.add_argument('--z_dim', type=int, default=256)
parser.add_argument('--h_dim', type=int, default=256)
parser.add_argument('--r_dim', type=int, default=256)

parser.add_argument('--exp', type=str, default=0.5)
parser.add_argument('--od', type=int, default=128)

parser.add_argument('--ubfc_phys_task', type=int, default=3, choices=[2, 3])

args = parser.parse_args()

from VIAStress.model import Model

# 假设你已经有 model 和 data_loader，并将模型设置为评估模式

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

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
    elif args.dataset_name == "verbio":
        return ""


if args.dataset_name == 'wesad':
    from VIAStress_wo_MMDG.WESAD_dataset_multi import WESAD_dataset
    from data.WESAD.Uniform_distribution_person_multi import train_subject, valid_subject, test_subject
elif args.dataset_name == 'ubfc_phys':
    from VIAStress_wo_MMDG.UBFC_Phys_dataset_multi import UBFC_Phys_dataset
    from data.UBFC_Phys.Uniform_distribution_person_multi import train_subject, valid_subject, test_subject
elif args.dataset_name == 'verbio':
    from VIAStress_wo_MMDG.VerBIO_dataset_multi import VerBIO_dataset
    from data.VerBIO.Uniform_distribution_person_multi import train_subject, valid_subject, test_subject

marker_dict = {
    0: 'o',  # non-stress
    1: '^',  # stress
    2: 's'   # amusement
}

class_name_dict = {
    0: "non-stress",
    1: "stress",
    2: "amusement"
}

for fold in range(args.k):
    print(f"Fold:{fold}")
    model = Model(y_dim=args.y_dim, r_dim=args.r_dim, x_dim=args.x_dim, h_dim=args.h_dim)
    model.load_state_dict(torch.load(os.path.join(args.save_path,
                                                  'VAEAD_me_{}_{}_{}_{}_{}{}.pth'.format(
                                                      args.exp,
                                                      args.od,
                                                      args.dataset_name,
                                                      args.y_dim,
                                                      model_name_tool(args),
                                                      fold + 1))))
    model.to(device)
    model.eval()

    test_p = test_subject[fold]
    features = []
    labels = []

    with torch.no_grad():
        for i, test_p_one in enumerate(test_p):
            if args.dataset_name == 'wesad':
                test_dataset = WESAD_dataset([test_p_one], binary=(args.y_dim == 2))
            elif args.dataset_name == 'ubfc_phys':
                test_dataset = UBFC_Phys_dataset([test_p_one], binary=(args.y_dim == 2), task=args.ubfc_phys_task)
            elif args.dataset_name == 'verbio':
                test_dataset = VerBIO_dataset([test_p_one])

            test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=True)

            for y, ppg, scr, scl, eda, peak in test_loader:
                y, ppg, eda = Variable(y).to(device), Variable(ppg).to(device), Variable(eda).to(device)
                z, _, _ = model.feature_cnn(ppg, eda)
                features.append(z.cpu())
                labels.append(y)

    # 合并
    features = torch.cat(features, dim=0).cpu().numpy()
    labels = torch.cat(labels, dim=0).cpu().numpy()

    # t-SNE
    tsne = TSNE(n_components=2, perplexity=30, random_state=42)
    features_2d = tsne.fit_transform(features)

    # 可视化
    plt.figure(figsize=(10, 8))
    classes = np.unique(labels)
    colors = plt.cm.tab10(np.linspace(0, 0.5, len(classes)))
    # colors = ['m', 'b', '#FF5733']

    for i, cls in enumerate(classes):
        idx = labels == cls
        plt.scatter(features_2d[idx, 0], features_2d[idx, 1],
                    label=class_name_dict.get(cls, f'Class {cls}'),
                    marker=marker_dict.get(cls, 'x'),
                    color=colors[i],
                    alpha=0.7)

    # plt.title(f'Using MMDG')
    plt.xticks([])  # 去除 x 轴刻度
    plt.yticks([])  # 去除 y 轴刻度
    # plt.xlabel("Dimension 1")
    # plt.ylabel("Dimension 2")
    # plt.legend(title="Classes")
    plt.grid(True)
    plt.tight_layout()

    # 保存图像
    save_path = os.path.join("tsne_plots", f'tsne_fold_{fold + 1}.png')
    plt.savefig(save_path)
    plt.close()
    print(f"Saved t-SNE plot to: {save_path}")
