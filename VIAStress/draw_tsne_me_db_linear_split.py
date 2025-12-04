# python draw_tsne_me_db_linear_split.py --dataset_name wesad --k 5 --od 128 --exp 1 --y_dim 2
# python draw_tsne_me_db_linear_split.py --dataset_name wesad --k 5 --od 128 --exp 0.5 --y_dim 3
# python draw_tsne_me_db_linear_split.py --dataset_name ubfc_phys --k 7 --od 128 --y_dim 2 --exp 1.5 --ubfc_phys_task 2
# python draw_tsne_me_db_linear_split.py --dataset_name ubfc_phys --k 7 --od 128 --y_dim 2 --exp 0.5 --ubfc_phys_task 3

import torch
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import matplotlib.patches as mpatches
from tqdm import tqdm
import argparse
import os
from torch.utils.data import DataLoader
from torch.autograd import Variable
import numpy as np
from sklearn.linear_model import LogisticRegression
from matplotlib.colors import ListedColormap

parser = argparse.ArgumentParser(description='Domain generalization')

parser.add_argument('--k', type=int, default=5)
parser.add_argument('--dataset_name', type=str, default='wesad', choices=['wesad', 'ubfc_phys'])
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


if args.dataset_name == 'wesad':
    from VIAStress_wo_MMDG.WESAD_dataset_multi import WESAD_dataset
    from data.WESAD.Uniform_distribution_person_multi import train_subject, valid_subject, test_subject
elif args.dataset_name == 'ubfc_phys':
    from VIAStress_wo_MMDG.UBFC_Phys_dataset_multi import UBFC_Phys_dataset
    from data.UBFC_Phys.Uniform_distribution_person_multi import train_subject, valid_subject, test_subject

for fold in range(args.k):
    print(f"Fold:{fold}")
    model = Model(y_dim=args.y_dim, r_dim=args.r_dim, x_dim=args.x_dim, h_dim=args.h_dim)
    model.load_state_dict(torch.load(os.path.join(args.save_path,
                                                  'VAEAD_me_{}_{}_{}_{}_{}{}.pth'.format(
                                                      # int(args.exp) if args.exp == int(args.exp) else args.exp,
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
    subject_ids = []  # 记录每条数据来自哪个 subject

    with torch.no_grad():
        for i, test_p_one in enumerate(test_p):

            if args.dataset_name == 'wesad':
                test_dataset = WESAD_dataset([test_p_one], binary=True if args.y_dim == 2 else False)
            elif args.dataset_name == 'ubfc_phys':
                if args.y_dim == 2:
                    test_dataset = UBFC_Phys_dataset([test_p_one], binary=True, task=args.ubfc_phys_task)
                else:
                    test_dataset = UBFC_Phys_dataset([test_p_one], binary=False)
            test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=True)

            for y, ppg, scr, scl, eda, peak in test_loader:
                y, ppg, eda = Variable(y).to(device), Variable(ppg).to(device), Variable(eda).to(device)

                z, _, _ = model.feature_cnn(ppg, eda)  # shape: (batch_size, feature_dim)
                features.append(z.cpu())
                labels.append(y)
                subject_ids.extend([i] * y.shape[0])
    # 合并特征和标签
    features = torch.cat(features, dim=0).cpu().numpy()
    labels = torch.cat(labels, dim=0).cpu().numpy()
    subject_ids = np.array(subject_ids)  # 注意：转换放在 labels 合并之后
    assert len(subject_ids) == len(labels), f"Mismatch: subject_ids={len(subject_ids)}, labels={len(labels)}"

    # t-SNE 降维
    tsne = TSNE(n_components=2, perplexity=30, random_state=42)
    features_2d = tsne.fit_transform(features)

    # 可视化
    # 颜色映射：不同类别不同颜色
    num_classes = len(np.unique(labels))
    # colors = plt.cm.get_cmap('tab10', num_classes)
    cmap = plt.cm.get_cmap('tab10')
    colors = cmap(np.linspace(0, 0.5, num_classes))

    x_min, x_max = features_2d[:, 0].min() - 1, features_2d[:, 0].max() + 1
    y_min, y_max = features_2d[:, 1].min() - 1, features_2d[:, 1].max() + 1
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 500),
                         np.linspace(y_min, y_max, 500))

    for subj in np.unique(subject_ids):
        plt.figure(figsize=(10, 8))

        idx_subj = (subject_ids == subj)
        X = features_2d[idx_subj]
        y = labels[idx_subj]

        # 训练线性分类器
        clf = LogisticRegression(solver='liblinear')
        clf.fit(X, y)

        # 创建网格用于绘制边界

        Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
        Z = Z.reshape(xx.shape)

        # 决策边界背景色
        cmap_light = ListedColormap([colors[i] for i in range(len(np.unique(labels)))])
        plt.contourf(xx, yy, Z, cmap=cmap_light, alpha=0.2)

        # 绘制数据点
        for cls in np.unique(labels):
            idx = (y == cls)
            plt.scatter(X[idx, 0], X[idx, 1],
                        color=colors[cls],
                        marker=marker_dict.get(cls, 'x'),
                        alpha=0.8,
                        label=class_name_dict.get(cls, f'{cls}'))

        # plt.title(f't-SNE (Fold {fold + 1}) - Subject {subj} (Linear Boundary)')
        # plt.title(f'Subject {subj} Linear Boundary')
        plt.xticks([])  # 去除 x 轴刻度
        plt.yticks([])  # 去除 y 轴刻度
        # plt.xlabel('Dimension 1')
        # plt.ylabel('Dimension 2')
        plt.legend(title="Classes", loc='best')
        # plt.grid(True)
        plt.tight_layout()

        os.makedirs("tsne_plots", exist_ok=True)
        plt.savefig(f"tsne_plots/tsne_fold{fold + 1}_subject{subj}_linear.png")
        plt.close()
