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
        D2 = cdist(Z, Z, metric='sqeuclidean')  # squared distances
        if gamma is None:
            # median heuristic: median of pairwise distances (not squared)
            # take upper triangle distances
            triu_idx = np.triu_indices(n_all, k=1)
            dists = np.sqrt(D2[triu_idx])
            med = np.median(dists)
            if med <= 0:
                med = 1.0
            gamma = med  # you can tune factor
        if kernel == 'rbf':
            K = np.exp(-D2 / (2 * (gamma ** 2)))
        else:  # laplace
            D = np.sqrt(D2)
            K = np.exp(-D / gamma)
    elif kernel == 'l2':
        # use euclidean distance as k(x,y) = ||x-y||_2
        K = cdist(Z, Z, metric='euclidean')
    elif kernel == 'l1':
        K = cdist(Z, Z, metric='cityblock')
    else:
        raise ValueError("Unsupported kernel")
    return K


def edk_stat_from_K(K, n, m):
    """
    根据论文给出的无偏估计公式，从合并的核矩阵 K 计算 ED_k.
    K is (n+m) x (n+m), first n rows/cols = X, last m rows/cols = Y
    """
    n_all = n + m
    # indices
    idxX = slice(0, n)
    idxY = slice(n, n_all)
    # cross term
    K_xy = K[idxX, idxY]
    term_xy = (2.0 / (n * m)) * np.sum(K_xy)
    # within X (exclude diagonal)
    K_xx = K[idxX, idxX]
    term_xx = (2.0 / (n * (n - 1))) * (np.sum(K_xx) - np.trace(K_xx)) / 2.0 * 2.0  # careful but simplified below
    # better compute sum_{i<j} k(Xi,Xj):
    sum_xx_upper = np.sum(np.triu(K_xx, k=1))
    term_xx = (2.0 / (n * (n - 1))) * sum_xx_upper
    # within Y
    K_yy = K[idxY, idxY]
    sum_yy_upper = np.sum(np.triu(K_yy, k=1))
    term_yy = (2.0 / (m * (m - 1))) * sum_yy_upper
    # EDk = 2/(nm) sum k(X,Y) - 2/(n(n-1)) sum_{i<j} k(Xi,Xj) - 2/(m(m-1)) sum_{i<j} k(Yi,Yj)
    edk = term_xy - term_xx - term_yy
    return edk


def edk_stat_fast_from_K_with_labels(K, labels):
    """
    更通用的方式：给定 labels (0 for X, 1 for Y)，从 K 计算 EDk。
    labels 长度 = n+m
    """
    n_all = len(labels)
    idxX = np.where(labels == 0)[0]
    idxY = np.where(labels == 1)[0]
    n = len(idxX)
    m = len(idxY)
    K_xy = K[np.ix_(idxX, idxY)]
    term_xy = (2.0 / (n * m)) * np.sum(K_xy)
    K_xx = K[np.ix_(idxX, idxX)]
    sum_xx_upper = np.sum(np.triu(K_xx, k=1))
    term_xx = (2.0 / (n * (n - 1))) * sum_xx_upper if n > 1 else 0.0
    K_yy = K[np.ix_(idxY, idxY)]
    sum_yy_upper = np.sum(np.triu(K_yy, k=1))
    term_yy = (2.0 / (m * (m - 1))) * sum_yy_upper if m > 1 else 0.0
    return term_xy - term_xx - term_yy


def permutation_test_edk(X, Y, kernel='rbf', gamma=None, n_permutations=1000, seed=None):
    """
    主函数：对样本 X (n x p) 和 Y (m x p) 做 ED_k / MMD + 置换检验。
    返回：T_obs, p_value, T_permutations (array length n_permutations)
    """
    rng = np.random.default_rng(seed)
    n = X.shape[0]
    m = Y.shape[0]
    Z = np.vstack([X, Y])
    K = pairwise_kernel(Z, kernel=kernel, gamma=gamma)
    labels = np.array([0] * n + [1] * m)
    T_obs = edk_stat_fast_from_K_with_labels(K, labels)
    T_perm = np.empty(n_permutations)
    for s in range(n_permutations):
        perm = rng.permutation(n + m)
        # 把 permuted indices 的前 n 当作组0，后 m 当作组1
        new_labels = np.full(n + m, 1, dtype=int)
        new_labels[perm[:n]] = 0
        T_perm[s] = edk_stat_fast_from_K_with_labels(K, new_labels)
    # p-value (one-sided: large values give evidence against H0)
    p_value = (1.0 + np.sum(T_perm >= T_obs)) / (1.0 + n_permutations)
    return T_obs, p_value, T_perm

# T_obs, pval, Tperm = permutation_test_edk(X, Y, kernel='rbf', gamma=None, n_permutations=1000, seed=42)
# print("T_obs=", T_obs, "p=", pval)
