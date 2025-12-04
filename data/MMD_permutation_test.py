# https://zhuanlan.zhihu.com/p/163839117
# https://publish.illinois.edu/xshao/files/2020/09/twosample.pdf

import numpy as np
from scipy.spatial.distance import cdist


def pairwise_kernel(Z, kernel='rbf', gamma=None):
    """
    计算合并样本 Z 的核矩阵 K (len(Z) x len(Z))
    kernel: 'rbf' (Gaussian), 'laplace', 'l2', 'l1'
    gamma: 带宽 (for rbf/laplace). 如果 None，对 rbf/laplace 使用 median heuristic。
    """
    n_all = Z.shape[0]
    if kernel in ('rbf', 'laplace'):
        D2 = cdist(Z, Z, metric='sqeuclidean')
        if gamma is None:
            triu_idx = np.triu_indices(n_all, k=1)
            dists = np.sqrt(D2[triu_idx])
            med = np.median(dists)
            if med <= 0:
                med = 1.0
            gamma = med
        if kernel == 'rbf':
            K = np.exp(-D2 / (2 * (gamma ** 2)))
        else:  # laplace
            D = np.sqrt(D2)
            K = np.exp(-D / gamma)
    elif kernel == 'l2':
        # 用欧氏距离做核（非典型）
        K = -cdist(Z, Z, metric='euclidean')
    elif kernel == 'l1':
        K = -cdist(Z, Z, metric='cityblock')
    else:
        raise ValueError("Unsupported kernel")
    return K


def mmd_from_K(K, n, m):
    """
    从合并的核矩阵 K 计算无偏 MMD (开平方后的)
    K is (n+m) x (n+m), first n rows/cols = X, last m = Y
    """
    idxX = slice(0, n)
    idxY = slice(n, n + m)
    K_xx = K[idxX, idxX]
    K_yy = K[idxY, idxY]
    K_xy = K[idxX, idxY]

    sum_xx = np.sum(K_xx) - np.trace(K_xx)
    sum_yy = np.sum(K_yy) - np.trace(K_yy)

    term_xx = sum_xx / (n * (n - 1))
    term_yy = sum_yy / (m * (m - 1))
    term_xy = (2.0 / (n * m)) * np.sum(K_xy)

    mmd2 = term_xx + term_yy - term_xy
    return np.sqrt(max(mmd2, 0.0))  # 避免浮点数负数开方


def mmd_from_K_with_labels(K, labels):
    """
    给定标签 (0 for X, 1 for Y)，从 K 计算 MMD
    """
    idxX = np.where(labels == 0)[0]
    idxY = np.where(labels == 1)[0]
    n = len(idxX)
    m = len(idxY)
    K_xx = K[np.ix_(idxX, idxX)]
    K_yy = K[np.ix_(idxY, idxY)]
    K_xy = K[np.ix_(idxX, idxY)]
    sum_xx = np.sum(K_xx) - np.trace(K_xx)
    sum_yy = np.sum(K_yy) - np.trace(K_yy)
    term_xx = sum_xx / (n * (n - 1)) if n > 1 else 0.0
    term_yy = sum_yy / (m * (m - 1)) if m > 1 else 0.0
    term_xy = (2.0 / (n * m)) * np.sum(K_xy)
    mmd2 = term_xx + term_yy - term_xy
    return np.sqrt(max(mmd2, 0.0))


def permutation_test_mmd(X, Y, kernel='rbf', gamma=None, n_permutations=1000, seed=123):
    """
    两样本 MMD^2 置换检验
    返回: T_obs (MMD^2 值), p_value, permutation values
    H0: MMD2(P,Q)=0 P=Q
    H1: MMD2(P,Q)≠0 P≠Q
    """
    rng = np.random.default_rng(seed)
    n, m = X.shape[0], Y.shape[0]
    Z = np.vstack([X, Y])
    K = pairwise_kernel(Z, kernel=kernel, gamma=gamma)
    labels = np.array([0] * n + [1] * m)

    # 原始统计量
    T_obs = mmd_from_K_with_labels(K, labels)

    # 置换分布
    T_perm = np.empty(n_permutations)
    for s in range(n_permutations):
        perm = rng.permutation(n + m)
        new_labels = np.full(n + m, 1, dtype=int)
        new_labels[perm[:n]] = 0
        T_perm[s] = mmd_from_K_with_labels(K, new_labels)

    # 单侧检验 (越大越偏离 H0)
    p_value = (1.0 + np.sum(T_perm >= T_obs)) / (1.0 + n_permutations)
    return T_obs, p_value, T_perm


# T_obs, pval, Tperm = permutation_test_mmd(X, Y, kernel='rbf', gamma=None, n_permutations=1000, seed=42)
# print(f"MMD² = {T_obs:.6f},  p = {pval:.4f}")
