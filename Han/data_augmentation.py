import math
import random
import numpy as np
from copy import deepcopy
from typing import Tuple, List


def jitter(x: np.ndarray, sigma: float = 0.03):
    """Add Gaussian noise (jitter). Works for (batch, channel, L)."""
    return x + np.random.normal(loc=0.0, scale=sigma, size=x.shape)


def scaling(x: np.ndarray, sigma: float = 0.05):
    """Scale signal by a factor drawn from N(1, sigma)."""
    if x.ndim == 3:
        factors = np.random.normal(1.0, sigma, size=(x.shape[0], 1, 1))
        return x * factors
    elif x.ndim == 2:
        factors = np.random.normal(1.0, sigma, size=(x.shape[0], 1))
        return x * factors
    else:
        factor = np.random.normal(1.0, sigma)
        return x * factor


def _smooth_curve(noise: np.ndarray, kernel_size: int = 31) -> np.ndarray:
    sigma = kernel_size / 6.0
    half = kernel_size // 2
    xs = np.arange(-half, half + 1)
    kernel = np.exp(-0.5 * (xs / sigma) ** 2)
    kernel /= kernel.sum()
    padded = np.pad(noise, (half, half), mode='reflect')
    smooth = np.convolve(padded, kernel, mode='valid')
    return smooth


def magnitude_warp(x: np.ndarray, sigma: float = 0.05, knot=4):
    """Multiply series by a smooth curve (around 1)."""
    def _apply_1d(sig):
        L = sig.shape[-1]
        random_noise = np.random.normal(loc=1.0, scale=sigma, size=(knot + 2,))
        xp = np.linspace(0, L - 1, num=knot + 2)
        x_new = np.interp(np.arange(L), xp, random_noise)
        x_new = _smooth_curve(x_new, kernel_size=max(7, L // 20))
        return sig * x_new

    if x.ndim == 3:
        return np.array([[_apply_1d(sig[0])] for sig in x])
    elif x.ndim == 2:
        return np.array([_apply_1d(sig) for sig in x])
    else:
        return _apply_1d(x)


def time_warp(x: np.ndarray, sigma: float = 0.2, knot=4):
    """Time warp: smoothly distort time axis."""
    def _apply_1d(sig):
        L = sig.shape[-1]
        random_noise = np.random.normal(loc=1.0, scale=sigma, size=(knot + 2,))
        xp = np.linspace(0, L - 1, num=knot + 2)
        tt = np.interp(np.arange(L), xp, random_noise)
        tt = _smooth_curve(tt, kernel_size=max(7, L // 20))
        cumsum = np.cumsum(tt)
        cumsum = (cumsum - cumsum[0]) / (cumsum[-1] - cumsum[0]) * (L - 1)
        x_new = np.interp(cumsum, np.arange(L), sig)
        return x_new

    if x.ndim == 3:
        return np.array([[_apply_1d(sig[0])] for sig in x])
    elif x.ndim == 2:
        return np.array([_apply_1d(sig) for sig in x])
    else:
        return _apply_1d(x)

if __name__ == "__main__":
    import neurokit2 as nk
    import matplotlib.pyplot as plt

    ppg_1 = nk.ppg_simulate(duration=30, sampling_rate=64)
    ppg_2 = nk.ppg_simulate(duration=30, sampling_rate=64, heart_rate=80)

    ppg_1 = np.array(ppg_1).reshape(1, 1, -1)
    ppg_2 = np.array(ppg_2).reshape(1, 1, -1)

    # 拼接成 batch (2, 1, 1920)
    x = np.concatenate([ppg_1, ppg_2], axis=0)

    print("Input shape:", x.shape)  # (2, 1, 1920)

    # 四种增强
    x_jitter = jitter(x)
    x_scaling = scaling(x)
    x_magwarp = magnitude_warp(x)
    x_timewarp = time_warp(x)

    # 绘图 — 分别绘制两个样本
    fig, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=True)

    for i in range(2):
        axes[i].plot(x[i, 0], label="Original", linewidth=2)
        axes[i].plot(x_jitter[i, 0], label="Jitter", alpha=0.8)
        axes[i].plot(x_scaling[i, 0], label="Scaling", alpha=0.8)
        axes[i].plot(x_magwarp[i, 0], label="Magnitude Warp", alpha=0.8)
        axes[i].plot(x_timewarp[i, 0], label="Time Warp", alpha=0.8)
        axes[i].set_title(f"PPG Sample {i + 1}", fontsize=12)
        axes[i].grid(True)
        if i == 1:
            axes[i].set_xlabel("Time (samples)")
        axes[i].set_ylabel("Amplitude")

    axes[0].legend()
    plt.tight_layout()
    plt.show()
