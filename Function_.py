from __future__ import annotations

import pickle
import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
import pandas as pd
import matplotlib.ticker as mtick
import os
import json
from types import SimpleNamespace

# import warnings
from pyparsing import Any
import torch.nn as nn
import torch
from scipy import stats
from scipy.stats import chi2
from scipy.stats import norm
from pandas import DataFrame
from pandas import concat
import io

from datetime import datetime
from typing import Mapping, Sequence
from dataclasses import asdict, is_dataclass
import platform
import sys
import ast
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler


def get_convergence_start_index(data, threshold=0.1, stable_window_size=20):
    """
    Finds the index where the data stabilizes.
    Assumes data is 2D (Time, Features) and checks the first column (index 0).
    Returns the first index `i` + 1 such that the absolute difference between steps
    in the window [i : i + stable_window_size] is less than the threshold.
    """
    if isinstance(data, torch.Tensor):
        # We need CPU numpy for checking
        check_data = data[:, 0].detach().cpu().numpy()
    else:
        check_data = np.asarray(data)[:, 0]

    diffs = np.abs(np.diff(check_data))

    # Iterate through the differences
    for i in range(len(diffs) - stable_window_size):
        # Check if the window is stable
        if np.all(diffs[i : i + stable_window_size] < threshold):
            # The difference at index i represents change between data[i] and data[i+1].
            # If diff[i] is small, stability effectively starts at i+1.
            return i + 1

    return 0  # If no stability found, return 0 (no cut)


# Function to safely load a PyTorch model (serialized by GPU) or any pickled object to CPU
def _to_cpu(obj: Any, *, verbose: bool = True):
    def log(message: str) -> None:
        if verbose:
            print(message)

    if torch.is_tensor(obj):
        log("Its a tensor, moved to CPU")
        return obj.detach().cpu()
    if isinstance(obj, (list, tuple)):
        log("Its a list or tuple, moved to CPU")
        return type(obj)(_to_cpu(x, verbose=verbose) for x in obj)
    if isinstance(obj, dict):
        log("Its a dict, moved to CPU")
        return {k: _to_cpu(v, verbose=verbose) for k, v in obj.items()}
    return obj


def safe_load(path: str, *, verbose: bool = True):
    """Load a file based on its extension, with optional verbose logging.

    - .pkl/.pickle: pickle (CPU-safe, including torch storages)
    - .pt/.pth/.ckpt: torch.load (CPU)
    - .npy/.npz: numpy.load

    For unknown extensions, tries pickle first then torch.
    """

    def log(message: str) -> None:
        if verbose:
            print(message)

    ext = os.path.splitext(path)[1].lower()

    class CPUUnpickler(pickle.Unpickler):
        def find_class(self, module, name):
            if module == "torch.storage" and name == "_load_from_bytes":
                return lambda b: torch.load(
                    io.BytesIO(b), map_location=torch.device("cpu"), weights_only=False
                )
            return super().find_class(module, name)

    def load_pickle():
        try:
            log("Loading with CPUUnpickler...")
            with open(path, "rb") as f:
                obj = CPUUnpickler(f).load()
            return _to_cpu(obj, verbose=verbose)
        except Exception as e:
            log(
                f"CPUUnpickler failed ({type(e).__name__}). Falling back to pickle.load..."
            )
            with open(path, "rb") as f:
                obj = pickle.load(f)
            return _to_cpu(obj, verbose=verbose)

    def load_torch():
        log("Loading with torch.load...")
        return _to_cpu(
            torch.load(path, map_location=torch.device("cpu"), weights_only=False),
            verbose=verbose,
        )

    def load_numpy():
        log("Loading with numpy.load...")
        return np.load(path, allow_pickle=True)

    if ext in (".pkl", ".pickle"):
        return load_pickle()
    if ext in (".pt", ".pth", ".ckpt"):
        try:
            return load_torch()
        except Exception as e:
            log(f"torch.load failed ({type(e).__name__}). Falling back to pickle...")
            return load_pickle()
    if ext in (".npy", ".npz"):
        return load_numpy()

    log(f"Unknown extension '{ext}'. Trying pickle then torch...")
    try:
        return load_pickle()
    except Exception as e:
        log(f"pickle failed ({type(e).__name__}). Trying torch.load...")
        return load_torch()


def calculate_volt_modepi(volt_all):
    """
    Simplified function to calculate volt_modepi and volt_di
    """
    volt_mode = volt_all.mean(axis=1)
    volt_std = volt_all.std(axis=1)
    volt_lamda = 1 / volt_std

    volt_pi1 = (1 / (2 * np.pi * volt_all.pow(3))).mul(volt_lamda, axis=0).pow(0.5)
    volt_pi2 = (
        ((-1) * ((volt_all.sub(volt_mode, axis=0))).pow(2).mul(volt_lamda, axis=0))
        / (2 * volt_all.mul(volt_mode.pow(2), axis=0))
    ).apply(np.exp)
    volt_pi = volt_pi1 * volt_pi2

    volt_modepi = ((volt_pi * volt_all).sum(axis=1)) / (volt_pi.sum(axis=1))
    volt_di = volt_all.sub(volt_modepi, axis=0)

    return volt_modepi, volt_di


def solvers(volt_modepi, volt_di, volt_all, soc, b, current, temp_avg):
    num = volt_all.shape[1]
    P = np.eye(2)
    Pi = np.zeros((volt_all.shape[0], num))
    Pi[0, 0:num] = 1000 * np.ones((1, num))
    Q = np.array([[1e-5, 0], [0, 1e-5]])
    R = 0.1
    Qi = 1e-4
    Ri = 2

    RO = 1e-3
    RP = 5.2203e-4
    CP = 5e3
    # DRi = [3.41688e-6, -6.19014e-6, -1.11437e-5, -3.64918e-06]
    DRi = 3 * 10 ** (-6) * np.ones(volt_all.shape[1])
    t = RP * CP
    SOC = np.zeros(volt_all.shape[0])
    smin = np.zeros(volt_all.shape[0])
    RH = np.zeros(volt_all.shape[0])

    SOC[0] = 0.01 * soc[0]
    smin[0] = 0.01 * soc[0] - 0.001
    RH[0] = 0.01 * soc[0]

    x = np.zeros((2, volt_all.shape[0]))
    xpre = np.zeros((2, volt_all.shape[0] + 1))
    x[:, 0] = np.array([0, soc[0]]).T
    xi = np.zeros((volt_all.shape[0], num))
    Xi = np.zeros((volt_all.shape[0], num))
    Xi[0, 0:num] = SOC[0] * np.ones((1, num))
    Xipre = np.zeros((volt_all.shape[0] + 1, num))

    A = np.array([[np.exp(-1 / t), 0], [0, 1]])
    B = np.array([(1 - np.exp(-1 / t)) * RP, -1 / (23 * 3600)])
    SOCi = np.zeros((volt_all.shape[0], num))
    SOCi = SOC[0] * np.ones((1, num))
    I = current
    temp = temp_avg
    OCV = [0] * volt_all.shape[0]
    OCVi = np.zeros((volt_all.shape[0], num))
    DUi = np.zeros((volt_all.shape[0], num))
    Ui = np.zeros((volt_all.shape[0], num))
    Uipre = np.zeros((volt_all.shape[0] + 1, num))
    ei = np.zeros((volt_all.shape[0], num))
    DUt = np.array(volt_di)
    C = np.zeros((volt_all.shape[0], 2))
    U = [0] * volt_all.shape[0]
    Upre = [0] * (volt_all.shape[0] + 1)
    Spre = [0] * (volt_all.shape[0] + 1)
    # Ut = volt_mode
    Ut = volt_modepi
    V = [0] * volt_all.shape[0]
    e = [0] * volt_all.shape[0]

    for k in range(1, volt_all.shape[0] - 1):
        # for k in range(1, 112):
        x[:, k] = np.dot(A, x[:, k - 1]) + B * I.iloc[k - 1]
        # print(temp.iloc[k])
        if x[1, k] > 1:
            x[1, k] = 1
        OCV[k] = (
            b[17] * x[1, k] ** 17
            + b[16] * x[1, k] ** 16
            + b[15] * x[1, k] ** 15
            + b[14] * x[1, k] ** 14
            + b[13] * x[1, k] ** 13
            + b[12] * x[1, k] ** 12
            + b[11] * x[1, k] ** 11
            + b[10] * x[1, k] ** 10
            + b[9] * x[1, k] ** 9
            + b[8] * x[1, k] ** 8
            + b[7] * x[1, k] ** 7
            + b[6] * x[1, k] ** 6
            + b[5] * x[1, k] ** 5
            + b[4] * x[1, k] ** 4
            + b[3] * x[1, k] ** 3
            + b[2] * x[1, k] ** 2
            + b[1] * x[1, k]
            + b[0]
            + b[18] * temp.iloc[k]
            + b[19] * temp.iloc[k] ** 2
            + b[20] * temp.iloc[k] ** 3
        )
        C[k, :] = [
            -1,
            b[17] * x[1, k] ** 16 * 17
            + b[16] * x[1, k] ** 15 * 16
            + b[15] * x[1, k] ** 14 * 15
            + b[14] * x[1, k] ** 13 * 14
            + b[13] * x[1, k] ** 12 * 13
            + b[12] * x[1, k] ** 11 * 12
            + b[11] * x[1, k] ** 10 * 11
            + b[10] * x[1, k] ** 9 * 10
            + b[9] * x[1, k] ** 8 * 9
            + b[8] * x[1, k] ** 7 * 8
            + b[7] * 7 * x[1, k] ** 6
            + b[6] * 6 * x[1, k] ** 5
            + b[5] * 5 * x[1, k] ** 4
            + b[4] * 4 * x[1, k] ** 3
            + 3 * b[3] * x[1, k] ** 2
            + 2 * b[2] * x[1, k]
            + b[1],
        ]
        U[k] = OCV[k] - x[0, k] - RO * I.iloc[k]
        e[k] = Ut.iloc[k] - U[k]
        rou = 1.2
        beta = 0.15
        gamma = 1.5
        if k == 2:
            V[k] = e[k] * e[k].T
        else:
            V[k] = (rou * V[k - 1] + e[k] * e[k].T) / (1 + rou)
        N = V[k] - beta * R - np.dot(np.dot(C[k, :], Q), (np.dot(C[k, :], Q)).T)
        M = np.dot(np.dot(C[k, :], A), np.dot(np.dot(P, A.T), np.dot(C[k, :], A).T))
        Ek = N / M
        if gamma * Ek > 1 and gamma * Ek < 1.5:
            lambda_k = gamma * Ek
        elif gamma * Ek >= 1.5:
            lambda_k = 1.5
        else:
            lambda_k = 1
        P = lambda_k * (A.dot(P)).dot(A.T) + Q
        K = np.dot(P, C[k, :].T) / (np.dot(C[k, :], np.dot(P, C[k, :].T)) + R)
        x[:, k] = x[:, k] + np.dot(K, e[k])
        P = P - np.dot(np.dot(K, C[k, :]), P)
        xpre[:, k + 1] = np.dot(A, x[:, k]) + B * I.iloc[k]
        Spre[k + 1] = soc[k] + B[1] * I.iloc[k]
        Upre[k + 1] = OCV[k] - xpre[0, k + 1] - RO * I.iloc[k + 1]

        for j in range(volt_all.shape[1]):
            xi[k, j] = xi[k - 1, j]
            Pi[k, j] = Pi[k - 1, j] + Qi
            OCVi[k, j] = (
                b[17] * (xi[k, j] + x[1, k]) ** 17
                + b[16] * (xi[k, j] + x[1, k]) ** 16
                + b[15] * (xi[k, j] + x[1, k]) ** 15
                + b[14] * (xi[k, j] + x[1, k]) ** 14
                + b[13] * (xi[k, j] + x[1, k]) ** 13
                + b[12] * (xi[k, j] + x[1, k]) ** 12
                + b[11] * (xi[k, j] + x[1, k]) ** 11
                + b[10] * (xi[k, j] + x[1, k]) ** 10
                + b[9] * (xi[k, j] + x[1, k]) ** 9
                + b[8] * (xi[k, j] + x[1, k]) ** 8
                + b[7] * (xi[k, j] + x[1, k]) ** 7
                + b[6] * (xi[k, j] + x[1, k]) ** 6
                + b[5] * (xi[k, j] + x[1, k]) ** 5
                + b[4] * (xi[k, j] + x[1, k]) ** 4
                + b[3] * (xi[k, j] + x[1, k]) ** 3
                + b[2] * (xi[k, j] + x[1, k]) ** 2
                + b[1] * (xi[k, j] + x[1, k])
                + b[0]
                + b[18] * temp[k]
                + b[19] * temp[k] ** 2
                + b[20] * temp[k] ** 3
            )
            if OCVi[k, j] > OCV[k] + 0.1:
                OCVi[k, j] = OCV[k] + 0.1
            Ci = (
                b[17] * (xi[k, j] + x[1, k]) ** 16 * 17
                + b[16] * (xi[k, j] + x[1, k]) ** 15 * 16
                + b[15] * (xi[k, j] + x[1, k]) ** 14 * 15
                + b[14] * (xi[k, j] + x[1, k]) ** 13 * 14
                + b[13] * (xi[k, j] + x[1, k]) ** 12 * 13
                + b[12] * (xi[k, j] + x[1, k]) ** 11 * 12
                + b[11] * (xi[k, j] + x[1, k]) ** 10 * 11
                + b[10] * (xi[k, j] + x[1, k]) ** 9 * 10
                + b[9] * (xi[k, j] + x[1, k]) ** 8 * 9
                + b[8] * (xi[k, j] + x[1, k]) ** 7 * 8
                + b[7] * 7 * (xi[k, j] + x[1, k]) ** 6
                + b[6] * 6 * (xi[k, j] + x[1, k]) ** 5
                + b[5] * 5 * (xi[k, j] + x[1, k]) ** 4
                + b[4] * 4 * (xi[k, j] + x[1, k]) ** 3
                + 3 * b[3] * (xi[k, j] + x[1, k]) ** 2
                + 2 * b[2] * (xi[k, j] + x[1, k])
                + b[1]
            )
            DUi[k, j] = OCVi[k, j] - OCV[k] - I[k] * DRi[j]
            Ui[k, j] = U[k] + DUi[k, j]
            Uipre[k + 1, j] = Upre[k + 1] + DUi[k, j]
            ei[k, j] = DUt[k, j] - DUi[k, j]
            Ki = (Pi[k, j] * Ci.T) / ((Ci * Pi[k, j] * Ci.T + Ri))
            xi[k, j] = xi[k, j] + Ki * ei[k, j]
            # Xi[k,j]=x[1,k].T+xi[k,j]
            Xi[k, j] = soc[k].T + xi[k, j]
            Xipre[k + 1, j] = xpre[1, k + 1].T + xi[k, j]
            Pi[k, j] = Pi[k, j] - Ki * Ci * Pi[k, j]
            # deta1[k,j]=1000*(Ui[k,j]-Utt[k,j])
            # deta4[k,j]=100*(Xi[k,j]-SOCi[k,j])
    xipre = np.zeros((xi.shape[0], xi.shape[1]))
    xipre[1:, :] = xi[: xi.shape[0] - 1, :]
    return xipre, xpre, Spre, Upre, Xipre, x, DUi, Xi


class CustomSigmoidFunc:
    # The authors use scale default = 1.8, shift default = 2.5
    # We set scale = 1, shift = 0 as default values for general use.
    def __init__(self, scale=1, shift=0):
        self.scale = scale
        self.shift = shift

    def forward(self, x):
        return self.shift + self.scale * torch.sigmoid(x)


def Custom_PCA(data, l1, l2):
    # Data standardization
    data_mean = np.mean(data, 0)
    data_std = np.std(data, 0)
    data_nor = (data - data_mean) / data_std
    # Calculate covariance matrix for standardized data
    X = np.cov(data_nor.T)
    # Calculate singular values for covariance matrix
    P, v, P_t = np.linalg.svd(X)  # This function returns three values u s v
    v_ratio = np.cumsum(v) / np.sum(
        v
    )  # Cumulative contribution rate of eigenvalues -> 이 중에서 상위 95%의 기여도를 차지하는 벡터들 선택.
    # Find the index of eigenvalues with a cumulative ratio greater than 0.95
    k = np.where(v_ratio > 0.95)[0]
    # New principal components
    p_k = P[:, : k[0]]
    v_I = np.diag(1 / v[: k[0]])
    # T2 statistic threshold calculation
    coe = (
        k[0]
        * (np.shape(data)[0] - 1)
        * (np.shape(data)[0] + 1)
        / ((np.shape(data)[0] - k[0]) * np.shape(data)[0])
    )
    T_95_limit = coe * stats.f.ppf(0.95, k[0], (np.shape(data)[0] - k[0]))
    T_99_limit = coe * stats.f.ppf(l1, k[0], (np.shape(data)[0] - k[0]))
    # SPE statistic threshold calculation
    O1 = np.sum((v[k[0] :]) ** 1)
    O2 = np.sum((v[k[0] :]) ** 2)
    O3 = np.sum((v[k[0] :]) ** 3)
    h0 = 1 - (2 * O1 * O3) / (3 * (O2**2))
    c_95 = norm.ppf(0.95)
    c_99 = norm.ppf(l2)
    SPE_95_limit = O1 * (
        (h0 * c_95 * ((2 * O2) ** 0.5) / O1 + 1 + O2 * h0 * (h0 - 1) / (O1**2))
        ** (1 / h0)
    )
    SPE_99_limit = O1 * (
        (h0 * c_99 * ((2 * O2) ** 0.5) / O1 + 1 + O2 * h0 * (h0 - 1) / (O1**2))
        ** (1 / h0)
    )
    return (
        v_I,
        v,
        v_ratio,
        p_k,
        data_mean,
        data_std,
        T_95_limit,
        T_99_limit,
        SPE_95_limit,
        SPE_99_limit,
        P,
        k,
        P_t,
        X,
        data_nor,
    )


def T2(data_in, data_mean, data_std, p_k, v_I):
    data_nor = np.array((data_in - data_mean) / data_std)
    D = p_k @ v_I @ p_k.T
    t2 = np.dot(np.dot((data_nor).T, D), (data_nor))
    return t2  # T2 statistic


def T2_array(data_in, data_mean, data_std, p_k, v_I):
    D = p_k @ v_I @ p_k.T  # (F×F)
    X = np.asarray((data_in - data_mean) / data_std, dtype=float)  # (N×F)
    contrib_array = (X @ D) * X
    return np.sum(contrib_array, axis=1), contrib_array


def SPE_array(data_in, data_mean, data_std, p_k):
    """
    Vectorized SPE (Squared Prediction Error) for all samples.

    SPE_i = ||(I - p_k p_k^T) x_i||^2, where x_i is standardized.

    Args:
        data_in: array-like or DataFrame with shape (N, F) or (F,) for single sample.
        data_mean: feature-wise mean (F,).
        data_std: feature-wise std (F,).
        p_k: principal component matrix (F, k).

    Returns:
        ndarray of shape (N,), SPE per sample.
    """
    X = np.asarray((data_in - data_mean) / data_std, dtype=float)
    if X.ndim == 1:
        X = X[None, :]  # (1, F)
    F_dim = p_k.shape[0]
    C = p_k @ p_k.T  # (F, F)
    M = np.eye(F_dim) - C  # residual projection
    R = X @ M  # (N, F)
    return np.einsum("ij,ij->i", R, R), R**2  # row-wise squared norm


def SPE(data_in, data_mean, data_std, p_k):
    # test_data_nor = ((data_in - data_mean) / data_std).reshape(len(data_in), 1)
    test_data_nor = np.array(((data_in - data_mean) / data_std)).reshape(6, 1)
    I = np.eye(len(data_in))
    Q_count = np.dot(
        np.dot((I - np.dot(p_k, p_k.T)), test_data_nor).T,
        np.dot((I - np.dot(p_k, p_k.T)), test_data_nor),
    )
    return Q_count  # Squared prediction error


def chi_square_dist_components(p_k, v_I, X, SPE_limit, T_limit):
    Pi = ((p_k @ v_I @ p_k.T) / T_limit) + (
        (np.eye(p_k.shape[0]) - p_k @ p_k.T) / SPE_limit
    )
    M = np.dot(X, Pi)
    tr_M = np.trace(M)
    tr_M2 = np.trace(np.dot(M, M))
    g = tr_M2 / tr_M
    h = (tr_M**2) / tr_M2
    return Pi, g, h


def diagnosis_thresholds(g, h, sigma_levels=[2, 3, 4.5, 6], show_plot=True):
    mean = h * g
    sigma = g * np.sqrt(2 * h)

    x = np.linspace(0, 30 * g, 10000)
    pdf = (1 / g) * chi2.pdf(x / g, h)

    thresholds = list()
    for k in sigma_levels:
        thresholds.append(mean + k * sigma)

    # -----------------------------
    # Numerical output
    # -----------------------------
    print(f"Chi-square distribution (df = {h})")
    print(f"Mean = {mean:.2f}, SD = {sigma:.2f}\n")

    print(f'{"k(SD)":>6} {"Threshold":>12} {"Area":>12}')
    print("-" * 50)

    for i in range(len(sigma_levels)):
        lower = max(0, mean - sigma_levels[i] * sigma)
        upper = mean + sigma_levels[i] * sigma
        area = chi2.cdf(upper / g, h) - chi2.cdf(lower / g, h)
        print(f"{sigma_levels[i]:6.1f} {thresholds[i]:12.4f} {area:12.6f}")

    if not show_plot:
        return thresholds

    # -----------------------------
    # Plot PDF
    # -----------------------------
    plt.figure()
    plt.plot(x, pdf, linewidth=2)
    plt.grid(True)

    # Plot mean
    plt.axvline(mean, color="k", linewidth=2, label="Mean")

    # Plot SD ranges
    for i in range(len(sigma_levels)):
        plt.axvline(thresholds[i], linestyle="--", linewidth=1.2)
        plt.text(
            thresholds[i],
            max(pdf) * 0.9,
            f"{sigma_levels[i]} SD",
            rotation=90,
            verticalalignment="bottom",
        )

    # Labels and title
    plt.xlabel("x")
    plt.ylabel("Probability Density")
    plt.title(f"Chi-square Distribution (df = {h}) with SD Ranges")
    plt.legend()
    plt.show()


def save_pca_results(output_dir, pca_outputs):
    """
    Save PCA function outputs ensuring types and shapes are preserved.

    Args:
        output_dir: Directory to store artifacts.
        pca_outputs: Tuple returned by `PCA(...)` in the exact order:
            (v_I, v, v_ratio, p_k, data_mean, data_std,
             T_95_limit, T_99_limit, SPE_95_limit, SPE_99_limit,
             P, k, P_t, X, data_nor)
    """
    (
        v_I,
        v,
        v_ratio,
        p_k,
        data_mean,
        data_std,
        T_95_limit,
        T_99_limit,
        SPE_95_limit,
        SPE_99_limit,
        P,
        k,
        P_t,
        X,
        data_nor,
    ) = pca_outputs

    os.makedirs(output_dir, exist_ok=True)

    arrays_path = os.path.join(output_dir, "pca_arrays.npz")
    manifest_path = os.path.join(output_dir, "pca_manifest.json")
    df_path = os.path.join(output_dir, "pca_data_nor.csv")

    # Store arrays and scalars in a single NPZ
    np.savez_compressed(
        arrays_path,
        v_I=np.asarray(v_I),
        v=np.asarray(v),
        v_ratio=np.asarray(v_ratio),
        p_k=np.asarray(p_k),
        data_mean=np.asarray(data_mean),
        data_std=np.asarray(data_std),
        T_95_limit=np.asarray(T_95_limit),
        T_99_limit=np.asarray(T_99_limit),
        SPE_95_limit=np.asarray(SPE_95_limit),
        SPE_99_limit=np.asarray(SPE_99_limit),
        P=np.asarray(P),
        k=np.asarray(k),
        P_t=np.asarray(P_t),
        X=np.asarray(X),
    )

    # Save data_nor: keep original type
    data_nor_type = "DataFrame" if isinstance(data_nor, pd.DataFrame) else "ndarray"
    if data_nor_type == "DataFrame":
        # No extra engines required; use CSV
        data_nor.to_csv(df_path, index=True)
        df_shape = list(data_nor.shape)
    else:
        # Store ndarray into separate NPZ for clarity
        np.savez_compressed(
            os.path.join(output_dir, "pca_data_nor.npz"), data_nor=np.asarray(data_nor)
        )
        df_path = os.path.join(output_dir, "pca_data_nor.npz")
        df_shape = list(np.asarray(data_nor).shape)

    manifest = {
        "types": {
            "v_I": "ndarray",
            "v": "ndarray",
            "v_ratio": "ndarray",
            "p_k": "ndarray",
            "data_mean": "ndarray",
            "data_std": "ndarray",
            "T_95_limit": "scalar",
            "T_99_limit": "scalar",
            "SPE_95_limit": "scalar",
            "SPE_99_limit": "scalar",
            "P": "ndarray",
            "k": "ndarray",
            "P_t": "ndarray",
            "X": "ndarray",
            "data_nor": data_nor_type,
        },
        "shapes": {
            "v_I": list(np.asarray(v_I).shape),
            "v": list(np.asarray(v).shape),
            "v_ratio": list(np.asarray(v_ratio).shape),
            "p_k": list(np.asarray(p_k).shape),
            "data_mean": list(np.asarray(data_mean).shape),
            "data_std": list(np.asarray(data_std).shape),
            "P": list(np.asarray(P).shape),
            "k": list(np.asarray(k).shape),
            "P_t": list(np.asarray(P_t).shape),
            "X": list(np.asarray(X).shape),
            "data_nor": df_shape,
        },
        "paths": {"arrays": arrays_path, "data_nor": df_path},
    }
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)


def load_pca_results(
    output_dir, load_data_nor: bool = True, validate_shapes: bool = True
):
    """
    Load PCA function outputs and validate type/shape consistency.

    Returns:
        Tuple in the same order as `PCA(...)` returns:
        (v_I, v, v_ratio, p_k, data_mean, data_std,
         T_95_limit, T_99_limit, SPE_95_limit, SPE_99_limit,
         P, k, P_t, X, data_nor)
    """
    manifest_path = os.path.join(output_dir, "pca_manifest.json")
    if not os.path.exists(manifest_path):
        raise FileNotFoundError("pca_manifest.json not found in output_dir")

    with open(manifest_path, "r", encoding="utf-8") as f:
        manifest = json.load(f)
    arrays_path = os.path.join(output_dir, "pca_arrays.npz")
    arrays = np.load(arrays_path, allow_pickle=False)

    # Extract arrays and scalars
    v_I = arrays["v_I"]
    v = arrays["v"]
    v_ratio = arrays["v_ratio"]
    p_k = arrays["p_k"]
    data_mean = arrays["data_mean"]
    data_std = arrays["data_std"]
    # Preserve numpy scalar type
    T_95_limit = arrays["T_95_limit"][()]
    T_99_limit = arrays["T_99_limit"][()]
    SPE_95_limit = arrays["SPE_95_limit"][()]
    SPE_99_limit = arrays["SPE_99_limit"][()]
    P = arrays["P"]
    k = arrays["k"]
    P_t = arrays["P_t"]
    X = arrays["X"]

    data_nor = None
    if load_data_nor:
        # Load data_nor according to stored type
        dn_type = manifest["types"]["data_nor"]

        # NOTE: 기존 코드에서는 CSV를 고정 경로로 읽고 있었음(호환 유지)
        dn_path = os.path.join(output_dir, "pca_data_nor.csv")
        # dn_path = manifest["paths"]["data_nor"]
        if dn_type == "DataFrame":
            # Read back CSV with index preserved
            data_nor = pd.read_csv(dn_path, index_col=0)
        elif dn_type == "ndarray":
            dn = np.load(dn_path, allow_pickle=False)
            data_nor = dn["data_nor"]
        else:
            raise ValueError("Unknown data_nor type in manifest: " + str(dn_type))

    if validate_shapes:
        # Validate shapes
        expected = manifest["shapes"]

        def _chk(name, obj):
            shp = list(np.asarray(obj).shape) if name != "data_nor" else list(obj.shape)
            if shp != expected[name]:
                raise ValueError(
                    f"Shape mismatch for {name}: loaded {shp}, expected {expected[name]}"
                )

        _chk("v_I", v_I)
        _chk("v", v)
        _chk("v_ratio", v_ratio)
        _chk("p_k", p_k)
        _chk("data_mean", data_mean)
        _chk("data_std", data_std)
        _chk("P", P)
        _chk("k", k)
        _chk("P_t", P_t)
        _chk("X", X)
        if load_data_nor:
            _chk("data_nor", data_nor)

    return (
        v_I,
        v,
        v_ratio,
        p_k,
        data_mean,
        data_std,
        T_95_limit,
        T_99_limit,
        SPE_95_limit,
        SPE_99_limit,
        P,
        k,
        P_t,
        X,
        data_nor,
    )


def save_lstm_model(model, models_dir="./models", filename="lstm.pth"):
    """
    LSTM 전체 모델 객체를 CPU로 이동해 저장합니다.

    Args:
        model: 학습된 LSTM 모델 객체
        models_dir: 저장 디렉터리
        filename: 파일명 (기본: lstm.pth)
    """
    os.makedirs(models_dir, exist_ok=True)
    path = os.path.join(models_dir, filename)
    torch.save(model.to(torch.device("cpu")), path)


def load_lstm_model(models_dir="./models", filename="lstm.pth", device=None):
    """
    CPU에서 안전하게 로드 후 지정한 디바이스로 이동합니다.

    Args:
        models_dir: 로드 디렉터리
        filename: 파일명
        device: torch.device 또는 None(그대로 사용)

    Returns:
        로드된 LSTM 모델 객체
    """
    path = os.path.join(models_dir, filename)
    mdl = torch.load(path, map_location=torch.device("cpu"), weights_only=False)
    if device is not None:
        mdl = mdl.to(device)
    return mdl


def save_net_state(model, models_dir="./models", filename="net_model.pth"):
    """
    net(CombinedAE)의 state_dict를 CPU 텐서로 저장합니다.
    """
    os.makedirs(models_dir, exist_ok=True)
    path = os.path.join(models_dir, filename)
    state = {k: v.detach().cpu() for k, v in model.state_dict().items()}
    torch.save(state, path)


def load_net_state_dict(models_dir="./models", filename="net_model.pth"):
    """
    net(CombinedAE)의 state_dict를 CPU에서 로드해 반환합니다.
    """
    path = os.path.join(models_dir, filename)
    return torch.load(path, map_location=torch.device("cpu"), weights_only=False)


def load_net_into(model, models_dir="./models", filename="net_model.pth", device=None):
    """
    제공된 모델 인스턴스에 state_dict를 로드하고 선택적 디바이스로 이동합니다.
    """
    state = load_net_state_dict(models_dir, filename)
    model.load_state_dict(state)
    if device is not None:
        model = model.to(device)
    return model


def plot_array(
    arr,
    title=None,
    x_label=None,
    y_label=None,
    figsize=(12, 4),
    save_path=None,
    show=True,
):

    # x축 생성: 제공된 x가 없으면 DataFrame index 또는 0..N-1 사용
    x = np.arange(len(arr))

    # 플롯
    plt.figure(figsize=figsize)
    plt.plot(x, arr, color="tab:blue", lw=1.5)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.title(title)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    if save_path is not None:
        plt.savefig(save_path, dpi=150)

    if show:
        plt.show()
    else:
        plt.close()


def load_loss_csv(loss_csv_path, *, delimiter=",", skiprows: int = 1):
    """Train_.py에서 저장한 loss CSV(epoch,avg_loss)를 로드.

    Returns:
        (epochs, losses): np.ndarray, np.ndarray
    """
    data = np.loadtxt(loss_csv_path, delimiter=delimiter, skiprows=skiprows)
    if data.ndim == 1:
        data = data.reshape(1, -1)
    if data.shape[1] < 2:
        raise ValueError(
            f"loss CSV는 최소 2개 컬럼(epoch,loss)이 필요합니다: {loss_csv_path}"
        )
    epochs = data[:, 0].astype(np.int64, copy=False)
    losses = data[:, 1].astype(np.float64, copy=False)
    return epochs, losses


def plot_loss_curve(
    *,
    loss_csv_path=None,
    epochs=None,
    losses=None,
    title: str = "Loss Curve",
    x_label: str = "Epoch",
    y_label: str = "Average Loss",
    figsize=(10, 4),
    save_path=None,
    show: bool = True,
    dpi: int = 150,
    tight_layout: bool = False,
    downsample: int = 1,
    yscale=None,
    png_compress_level=1,
):
    """Loss curve를 플로팅.

    입력은 (1) loss_csv_path 또는 (2) epochs/losses 배열 둘 중 하나.
    속도를 위해 downsample, tight_layout 옵션을 제공.

    Returns:
        (epochs, losses): 플롯에 사용된 배열
    """
    if loss_csv_path is not None:
        epochs, losses = load_loss_csv(loss_csv_path)
    else:
        if epochs is None or losses is None:
            raise ValueError(
                "loss_csv_path 또는 (epochs, losses) 중 하나는 반드시 제공해야 합니다."
            )
        epochs = np.asarray(epochs)
        losses = np.asarray(losses, dtype=np.float64)

    if epochs.shape[0] != losses.shape[0]:
        raise ValueError(
            f"epochs/losses 길이가 다릅니다: {epochs.shape[0]} vs {losses.shape[0]}"
        )

    ds = int(downsample) if downsample is not None else 1
    if ds < 1:
        ds = 1
    if ds > 1:
        epochs_p = epochs[::ds]
        losses_p = losses[::ds]
    else:
        epochs_p = epochs
        losses_p = losses

    if save_path is not None:
        os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)

    fig = plt.figure(figsize=figsize)
    ax = fig.gca()
    ax.plot(epochs_p, losses_p, color="tab:blue", lw=1.5)
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    if yscale is not None:
        ax.set_yscale(yscale)

    if tight_layout:
        fig.tight_layout()
    else:
        # tight_layout은 비용이 커서 기본은 빠른 여백으로
        fig.subplots_adjust(left=0.08, right=0.98, top=0.88, bottom=0.15)

    if save_path is not None:
        if str(save_path).lower().endswith(".png") and png_compress_level is not None:
            fig.savefig(
                save_path,
                dpi=dpi,
                pil_kwargs={"compress_level": int(png_compress_level)},
            )
        else:
            fig.savefig(save_path, dpi=dpi)

    if show:
        plt.show()
    else:
        plt.close(fig)

    return epochs, losses


def plot_loss_curve_all(path):
    save_path = f"{path}/loss_net.png"
    if not os.path.exists(save_path):
        plot_loss_curve(
            loss_csv_path=f"{path}/loss_net.csv",
            title="Voltage train loss",
            save_path=save_path,
            show=False,  # 저장만 하고 창은 안 띄움
            dpi=200,
            downsample=1,  # 1이면 전체 epoch 표시
            tight_layout=False,
            png_compress_level=9,
            # yscale="log",
        )

    save_path = f"{path}/loss_netx.png"
    if not os.path.exists(save_path):
        plot_loss_curve(
            loss_csv_path=f"{path}/loss_netx.csv",
            title="SoC train loss",
            save_path=save_path,
            show=False,  # 저장만 하고 창은 안 띄움
            dpi=200,
            downsample=1,  # 1이면 전체 epoch 표시
            tight_layout=False,
            png_compress_level=9,
            # yscale="log",
        )

    save_path = f"{path}/loss_net_val.png"
    if not os.path.exists(save_path):
        plot_loss_curve(
            loss_csv_path=f"{path}/loss_net_val.csv",
            title="Voltage validation loss (every 10 epochs)",
            save_path=save_path,
            show=False,
            downsample=1,
        )

    save_path = f"{path}/loss_netx_val.png"
    if not os.path.exists(save_path):
        plot_loss_curve(
            loss_csv_path=f"{path}/loss_netx_val.csv",
            title="SoC validation loss (every 10 epochs)",
            save_path=save_path,
            show=False,
            downsample=1,
        )


def plot_auc_roc_curve_from_threshold_matrix(
    y_true,
    predict_results,
    predict_thresholds,
    *,
    save_path=None,
    title: str = "AUC-ROC Curve",
    figsize=(6, 6),
    show: bool = True,
):
    """(호환용) threshold grid 기반 ROC/AUC 계산 + 최적점 계산 + 플로팅을 한 번에 수행."""

    roc = compute_roc_auc_from_threshold_matrix(
        y_true, predict_results, predict_thresholds
    )
    opt = compute_optimal_thresholds_from_roc(
        roc["fpr"], roc["tpr"], roc["thresholds"], candidate_idx=roc["candidate_idx"]
    )

    print(f"AUC = {roc['auc']:.4f}")
    print(
        f"Best threshold (Youden J) ≈ {opt['best']['threshold']:.6g} @ (FPR={opt['best']['fpr']:.4f}, TPR={opt['best']['tpr']:.4f})"
    )
    print(
        f"Closest to (0,1) ≈ {opt['closest_to_01']['threshold']:.6g} @ (FPR={opt['closest_to_01']['fpr']:.4f}, TPR={opt['closest_to_01']['tpr']:.4f}), dist={opt['closest_to_01']['distance']:.4f}"
    )

    plot_roc_curve(
        roc["fpr"],
        roc["tpr"],
        roc["auc"],
        best_point=opt["best"],
        closest_to_01_point=opt["closest_to_01"],
        save_path=save_path,
        title=title,
        figsize=figsize,
        show=show,
    )

    if save_path is not None:
        print(f"ROC curve saved to: {save_path}")

    return {**roc, **opt}


def compute_roc_auc_from_threshold_matrix(y_true, predict_results, predict_thresholds):
    """threshold grid 기반 ROC point들과 AUC를 계산.

    Returns:
        dict: fpr, tpr, thresholds(thr_arr), auc, candidate_idx(finite threshold index), pos_count, neg_count
    """
    from sklearn import metrics as sk_metrics

    y_true = np.asarray(y_true)
    if y_true.ndim != 1:
        y_true = y_true.reshape(-1)

    pos_count = int(np.sum(y_true == 1))
    neg_count = int(np.sum(y_true == 0))
    if pos_count == 0 or neg_count == 0:
        raise ValueError(
            f"ROC 계산을 위해 정상/고장 샘플이 모두 필요합니다. pos={pos_count}, neg={neg_count}"
        )

    predict_results = np.asarray(predict_results)
    predict_thresholds = np.asarray(predict_thresholds, dtype=float)
    if predict_results.ndim != 2:
        raise ValueError(
            f"predict_results는 2D여야 합니다. got ndim={predict_results.ndim}"
        )
    if predict_results.shape[1] != predict_thresholds.shape[0]:
        raise ValueError(
            f"predict_results.shape[1]({predict_results.shape[1]}) != len(predict_thresholds)({predict_thresholds.shape[0]})"
        )

    # threshold를 큰 값 -> 작은 값으로 낮추면, 예측 Positive가 늘어나며 ROC가 (0,0) -> (1,1)로 진행
    fpr_list = [0.0]
    tpr_list = [0.0]
    thr_list = [float("inf")]

    for j in range(len(predict_thresholds) - 1, -1, -1):
        y_pred = predict_results[:, j].astype(bool)
        tp = int(np.sum(y_pred & (y_true == 1)))
        fp = int(np.sum(y_pred & (y_true == 0)))
        tpr_list.append(tp / pos_count)
        fpr_list.append(fp / neg_count)
        thr_list.append(float(predict_thresholds[j]))

    fpr_list.append(1.0)
    tpr_list.append(1.0)
    thr_list.append(float("-inf"))

    fpr = np.asarray(fpr_list, dtype=float)
    tpr = np.asarray(tpr_list, dtype=float)
    thr_arr = np.asarray(thr_list, dtype=float)
    auc_val = float(sk_metrics.auc(fpr, tpr))

    finite_mask = np.isfinite(thr_arr)
    candidate_idx = np.where(finite_mask)[0]

    return {
        "fpr": fpr,
        "tpr": tpr,
        "thresholds": thr_arr,
        "auc": auc_val,
        "candidate_idx": candidate_idx,
        "pos_count": pos_count,
        "neg_count": neg_count,
    }


def compute_optimal_thresholds_from_roc(fpr, tpr, thresholds, *, candidate_idx=None):
    """ROC curve 상에서 최적 임계값(Youden J, (0,1) 최소거리)을 계산."""
    fpr = np.asarray(fpr, dtype=float)
    tpr = np.asarray(tpr, dtype=float)
    thr_arr = np.asarray(thresholds, dtype=float)
    if fpr.shape != tpr.shape or fpr.shape != thr_arr.shape:
        raise ValueError("fpr/tpr/thresholds는 같은 shape 여야 합니다.")

    if candidate_idx is None:
        candidate_idx = np.where(np.isfinite(thr_arr))[0]
    else:
        candidate_idx = np.asarray(candidate_idx, dtype=int)

    j_scores = tpr - fpr
    if candidate_idx.size > 0:
        best_idx = int(candidate_idx[np.argmax(j_scores[candidate_idx])])
    else:
        best_idx = int(np.argmax(j_scores))

    dist2 = (fpr**2) + ((1.0 - tpr) ** 2)
    if candidate_idx.size > 0:
        closest_idx = int(candidate_idx[np.argmin(dist2[candidate_idx])])
    else:
        closest_idx = int(np.argmin(dist2))

    ## ================ Youden J 기준 최적점 ================
    best = {
        "threshold": float(thr_arr[best_idx]),
        "fpr": float(fpr[best_idx]),
        "tpr": float(tpr[best_idx]),
        "index": best_idx,
        "youden_j": float(j_scores[best_idx]),
    }

    ## ================ (0,1) 최소거리 기준 최적점 ================
    closest_to_01 = {
        "threshold": float(thr_arr[closest_idx]),
        "fpr": float(fpr[closest_idx]),
        "tpr": float(tpr[closest_idx]),
        "index": closest_idx,
        "distance": float(np.sqrt(dist2[closest_idx])),
    }

    return {"best": best, "closest_to_01": closest_to_01}


def plot_roc_curve(
    fpr,
    tpr,
    auc_val,
    *,
    best_point=None,
    closest_to_01_point=None,
    save_path=None,
    title: str = "AUC-ROC Curve",
    figsize=(6, 6),
    show: bool = True,
    dpi: int = 200,
):
    """ROC 커브를 플로팅하고(옵션으로) 최적점들을 표시."""

    fpr = np.asarray(fpr, dtype=float)
    tpr = np.asarray(tpr, dtype=float)
    auc_val = float(auc_val)

    if save_path is not None:
        os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)

    plt.figure(figsize=figsize)
    plt.plot(fpr, tpr, label=f"ROC (AUC={auc_val:.3f})")
    plt.plot([0, 1], [0, 1], linestyle="--", color="gray", label="Random")

    if best_point is not None:
        best_fpr = float(best_point.get("fpr"))
        best_tpr = float(best_point.get("tpr"))
        best_thr = float(best_point.get("threshold"))
        dx = 0.05 if best_fpr <= 0.7 else -0.25
        dy = 0.05 if best_tpr <= 0.7 else -0.25
        plt.scatter(
            [best_fpr],
            [best_tpr],
            color="red",
            s=60,
            zorder=5,
            label=f"Best thr={best_thr:.3g}",
        )
        plt.annotate(
            f"thr={best_thr:.3g}\nFPR={best_fpr:.2f}, TPR={best_tpr:.2f}",
            xy=(best_fpr, best_tpr),
            xytext=(
                min(max(best_fpr + dx, 0.0), 1.0),
                min(max(best_tpr + dy, 0.0), 1.0),
            ),
            textcoords="data",
            arrowprops=dict(arrowstyle="->", color="red", lw=1.0),
            fontsize=9,
            color="red",
        )

    if closest_to_01_point is not None:
        closest_fpr = float(closest_to_01_point.get("fpr"))
        closest_tpr = float(closest_to_01_point.get("tpr"))
        closest_thr = float(closest_to_01_point.get("threshold"))
        dx2 = 0.05 if closest_fpr <= 0.7 else -0.25
        dy2 = -0.12 if closest_tpr >= 0.5 else 0.08
        plt.scatter(
            [closest_fpr],
            [closest_tpr],
            color="blue",
            s=60,
            zorder=5,
            label=f"Closest(0,1) thr={closest_thr:.3g}",
        )
        plt.annotate(
            f"thr={closest_thr:.3g}\nFPR={closest_fpr:.2f}, TPR={closest_tpr:.2f}",
            xy=(closest_fpr, closest_tpr),
            xytext=(
                min(max(closest_fpr + dx2, 0.0), 1.0),
                min(max(closest_tpr + dy2, 0.0), 1.0),
            ),
            textcoords="data",
            arrowprops=dict(arrowstyle="->", color="blue", lw=1.0),
            fontsize=9,
            color="blue",
        )

    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(title)
    plt.grid(True, alpha=0.3)
    plt.legend(loc="lower right")
    plt.tight_layout()

    if save_path is not None:
        plt.savefig(save_path, dpi=dpi)

    if show:
        plt.show()
    else:
        plt.close()


def cont(X_col, data_mean, data_std, X_test, P, num_pc, lamda, T2UCL1):
    X_test = (X_test - data_mean) / data_std
    S = np.dot(X_test, P[:, :num_pc])
    r = []
    ee = T2UCL1 / num_pc
    for i in range(num_pc):
        aa = S[i] * S[i]
        a = aa / lamda[i, i]
        if a > ee:
            r = np.append(r, i)
    cont = np.zeros((len(r), X_col))
    for i in range(len(r)):
        for j in range(X_col):
            cont[i, j] = np.abs(S[i] / lamda[i, i] * P[j, i] * X_test[j])
    contT = np.zeros(X_col)
    for j in range(X_col):
        contT[j] = np.sum(cont[:, j])
    I = np.eye((np.dot(P, P.T)).shape[0], (np.dot(P, P.T)).shape[1])
    e = np.dot(X_test, (I - np.dot(P, P.T)))
    contQ = np.square(e)
    return contT, contQ


def Ratio_cu(x):
    sum = np.sum(x)
    for i in range(x.shape[0]):
        x[i] = x[i] / sum
    return x


def SlidingAverage_list(s, n):
    mean = []
    if len(s) > n:
        for m in range(n):
            mean.append(np.mean(s[:n]))
        # mean = s[:n].tolist()
        for i in range(n, len(s)):
            select_s = s[i - n : i]
            mean_s = np.mean(select_s)
            mean.append(mean_s)
    else:
        mean = s.tolist()
    return mean


def SlidingAverage(s, n):
    mean = []
    if len(s) > n:
        for m in range(n):
            mean.append(np.mean(s[:n]))
        for i in range(n, len(s)):
            select_s = s[i - n : i]
            mean_s = np.mean(select_s)
            mean.append(mean_s)
    else:
        mean = s.tolist()
    return mean


def DiagnosisFeature(ERRORU, ERRORX, get_true_feature=False):
    # 성능 최적화 버전:
    # - pandas DataFrame 반복 생성 제거
    # - np.apply_along_axis(파이썬 루프) 제거
    # - SlidingAverage_list(파이썬 루프) 대신 numpy 누적합 기반 계산

    U = np.asarray(ERRORU, dtype=float)
    X = np.asarray(ERRORX, dtype=float)
    if U.ndim != 2 or X.ndim != 2:
        raise ValueError(
            f"DiagnosisFeature expects 2D arrays, got {U.ndim}D and {X.ndim}D"
        )

    eps = 1e-12

    U_max = U.max(axis=1)
    X_max = X.max(axis=1)
    U_mean = U.mean(axis=1)
    X_mean = X.mean(axis=1)
    U_std = np.maximum(U.std(axis=1), eps)
    X_std = np.maximum(X.std(axis=1), eps)

    Z_U = ((U - U_mean[:, None]) / U_std[:, None]).max(axis=1)
    Z_X = ((X - X_mean[:, None]) / X_std[:, None]).max(axis=1)

    # 두 번째로 큰 값(행별): np.partition을 axis=1로 벡터화
    # (열 수가 1인 경우를 대비)
    if U.shape[1] >= 2:
        U_second = np.partition(U, -2, axis=1)[:, -2]
    else:
        U_second = U[:, 0]

    if X.shape[1] >= 2:
        X_second = np.partition(X, -2, axis=1)[:, -2]
    else:
        X_second = X[:, 0]

    max_diff_ERRORU = (U_max - U_second) / U_std
    max_diff_ERRORX = (X_max - X_second) / X_std

    # pandas EWM(지수이동평균)은 그대로 사용(여기서의 비용은 작음)
    alpha = 0.2
    ERRORUm = pd.Series(U_max)
    ERRORXm = pd.Series(X_max)
    Z_U_smoothed = ERRORUm.ewm(alpha=alpha).mean()
    Z_X_smoothed = ERRORXm.ewm(alpha=alpha).mean()

    def _sliding_average_prev_window_np(s, n: int):
        a = np.asarray(s, dtype=float).reshape(-1)
        L = int(a.shape[0])
        n = int(n)
        if L == 0:
            return a
        if L <= n or n <= 0:
            return a
        out = np.empty(L, dtype=float)
        first_mean = float(np.mean(a[:n]))
        out[:n] = first_mean
        c = np.cumsum(a, dtype=float)
        c = np.concatenate(([0.0], c))
        window_sums = c[n:] - c[:-n]  # ends at n..L
        out[n:] = window_sums[:-1] / n  # ends at n..L-1
        return out

    # 원래 로직 유지: X 관련 3개만 sliding average 적용
    if get_true_feature:
        original_max_diff_ERRORX = pd.Series(max_diff_ERRORX.copy())
        origin_Z_X = pd.Series(Z_X.copy())
        origin_Z_X_smoothed = pd.Series(Z_X_smoothed.copy())
    max_diff_ERRORX = pd.Series(_sliding_average_prev_window_np(max_diff_ERRORX, 100))
    Z_X = pd.Series(_sliding_average_prev_window_np(Z_X, 100))
    Z_X_smoothed = pd.Series(
        _sliding_average_prev_window_np(Z_X_smoothed.to_numpy(), 100)
    )

    max_diff_ERRORU = pd.Series(max_diff_ERRORU)
    Z_U = pd.Series(Z_U)

    df_data = pd.concat(
        [max_diff_ERRORU, max_diff_ERRORX, Z_U, Z_X, Z_U_smoothed, Z_X_smoothed],
        axis=1,
    )

    df_data2 = None
    if get_true_feature:
        df_data2 = pd.concat(
            [
                max_diff_ERRORU,
                original_max_diff_ERRORX,
                Z_U,
                origin_Z_X,
                Z_U_smoothed,
                origin_Z_X_smoothed,
            ],
            axis=1,
        )

    return df_data, df_data2


def ClassifyFeature(temp_max, temp_avg, CONTN, insulation_resistance, threshold1, fai):
    reversed_fai = np.flip(fai[:f_time])
    index_array = np.where(reversed_fai < threshold1)[0]
    index = index_array[0] if index_array.size > 0 else None
    original_index = f_time - index - 1
    temp_dif = temp_max - temp_avg
    features_array = np.empty((0, CONTN.shape[1] + 5))
    for ttime in range(original_index + 1, f_time + 1, 1):
        Feature1 = CONTN[ttime, :]
        f1 = max(fai)
        max_erroru_column = np.argmax(ERRORU[ttime, :])
        max_count_U = np.sum(
            np.argmax(ERRORU[ttime - 3000 : ttime, :], axis=1) == max_erroru_column
        )
        f2 = max_count_U / 3000
        f3 = temp_dif[f_time - 50 : f_time].max()
        f4 = insulation_resistance.iloc[ttime - 1000 : ttime].min()
        f5 = volt_all.iloc[ttime - 100 : ttime, :].min().min()
        new_features = np.concatenate((Feature1, np.array([f1, f2, f3, f4, f5])))
        features_array = np.vstack((features_array, new_features))
    vin_feature = pd.DataFrame(features_array)
    return vin_feature


# Refer to https://machinelearningmastery.com/convert-time-series-supervised-learning-problem-python/
def series_to_supervised(data, n_in=1, n_out=1, dropnan=True):
    """
    Frame a time series as a supervised learning dataset.
    Arguments:
            data: Sequence of observations as a list or NumPy array.
            n_in: Number of lag observations as input (X).
            n_out: Number of observations as output (y).
            dropnan: Boolean whether or not to drop rows with NaN values.
    Returns:
            Pandas DataFrame of series framed for supervised learning.
    """
    n_vars = 1 if type(data) is list else data.shape[1]
    df = DataFrame(data)
    cols, names = list(), list()
    # input sequence (t-n, ... t-1)
    for i in range(n_in, 0, -1):
        cols.append(df.shift(i))
        names += [("var%d(t-%d)" % (j + 1, i)) for j in range(n_vars)]
    # forecast sequence (t, t+1, ... t+n)
    for i in range(0, n_out):
        cols.append(df.shift(-i))
        if i == 0:
            names += [("var%d(t)" % (j + 1)) for j in range(n_vars)]
        else:
            names += [("var%d(t+%d)" % (j + 1, i)) for j in range(n_vars)]
    # put it all together
    agg = concat(cols, axis=1)
    agg.columns = names
    # drop rows with NaN values
    if dropnan:
        agg.dropna(inplace=True)
    return agg


def prepare_training_data(test_X, INPUT_SIZE, TIME_STEP, device):
    test_X_df = pd.DataFrame(test_X.cpu().detach().numpy()[:, 0, :])
    reframed = series_to_supervised(test_X_df, 1, 1)
    reframed.drop(
        reframed.columns[INPUT_SIZE : INPUT_SIZE * 2 - 2], axis=1, inplace=True
    )
    train = reframed.values
    train_X, train_y = (
        train[:, :-2],
        train[:, -2:],
    )  # Last two columns are the targets (volt_modepi and soc)
    train_y = train_y.reshape(-1, 2)
    batch_train = int(reframed.shape[0] / TIME_STEP)

    train_X = torch.tensor(train_X)
    train_X = train_X.reshape(batch_train, TIME_STEP, INPUT_SIZE).to(device)
    train_y = torch.tensor(train_y)
    train_y = train_y.reshape(batch_train, TIME_STEP, 2).to(device)

    return train_X, train_y


def plot_testX_timeseries(
    input,
    feature_names=None,
    title=None,
    figsize=(12, 6),
    save_path=None,
    show=True,
    seperate=False,
    _range: list = [-1],
    start_idx=0,
    ax=None,
    close=None,
):
    """
    test_X 시계열 데이터(형상: [T, 1, 7] 또는 [T, 7])를 시간(x축) 대비 다중 라인(y축)으로 그립니다.

    Args:
        test_X: torch.Tensor | np.ndarray | pd.DataFrame
            - 권장 형상: [T, 1, 7] (tensor) 또는 [T, 7]
        feature_names: 특성 이름 리스트(길이 = 특성 수). None이면 기본 이름 사용.
        title: 그래프 제목. None이면 기본 제목 사용.
        figsize: matplotlib figure size.
        save_path: 저장 경로(확장자 포함). None이면 저장하지 않음.
        show: True면 화면에 표시, False면 닫음(서버/배치 환경용).
    """

    if not (len(_range) == 1 and _range[0] == -1) and len(_range) != 2:
        raise ValueError(
            "Expected _range to be a 1D array with a single element -1 or a 2D array."
        )

    """ sperate가 True이면 for 문을 돌고 false 이면 한번에 그린다."""
    plot_all = True if _range == [-1] else False
    if seperate:
        _range = [0, input.shape[-1] - 1] if plot_all else _range
        for i in range(_range[0], _range[1] + 1):
            print(f"Plotting feature index: {i}")
            print(f"Input shape: {input.shape}")
            if plot_all:  # 모든 특성 그리기
                if input.shape[-1] > 10:
                    raise Exception(
                        "특성 수가 너무 많아 개별 플롯으로 그릴 수 없습니다. 'seperate'를 False로 설정하세요."
                    )

            print(f"{i}번째 특성을 개별 플롯으로 그립니다.")
            if input.ndim == 2:
                test_X = input[start_idx:, i].unsqueeze(1)
            elif input.ndim == 3:
                test_X = input[start_idx:, :, i].unsqueeze(1)
            else:
                raise ValueError(f"Expected 2D or 3D array, got shape {input.shape}")

            title_input = title + f"-{i}st"
            if save_path is not None:
                save_path_input = save_path + f"-{i}st.png"

            print(f"test_X shape for plotting: {test_X.shape}")
            # seperate=True는 개별 figure를 전제로 하므로, 외부 ax가 있어도 전달하지 않음
            plot_timeseries(
                test_X,
                feature_names,
                title_input,
                figsize,
                save_path_input,
                show,
                ax=None,
                close=close,
            )
    else:
        if plot_all:  # 모든 특성 그리기
            print("모든 특성을 한 번에 플롯으로 그립니다.")
            test_X = input[start_idx:]
            title += f"-all({test_X.shape[-1]})"
            if save_path is not None:
                save_path += f"-all.png"
        else:  # 특정 범위의 특성 그리기
            print(
                f"{_range[0]}부터 {_range[1]}까지의 특성을 한 번에 플롯으로 그립니다."
            )
            if input.ndim == 2:
                test_X = input[start_idx:, _range[0] : _range[1] + 1]
            elif input.ndim == 3:
                test_X = input[start_idx:, :, _range[0] : _range[1] + 1]
            else:
                raise ValueError(f"Expected 2D or 3D array, got shape {input.shape}")
            title += f"-range({_range[0]}-{_range[1]})"
            if save_path is not None:
                save_path += f"-range({_range[0]}-{_range[1]}).png"

        plot_timeseries(
            test_X, feature_names, title, figsize, save_path, show, ax=ax, close=close
        )


def plot_timeseries(
    test_X,
    feature_names=None,
    title=None,
    figsize=(12, 6),
    save_path=None,
    show=True,
    ax=None,
    close=None,
):
    # 입력을 numpy 2D [T, F]로 변환
    if isinstance(test_X, torch.Tensor):
        arr = test_X.detach().cpu().numpy()
    elif isinstance(test_X, pd.DataFrame):
        arr = test_X.values
    else:
        arr = np.asarray(test_X)

    # [T, 1, F] → [T, F]
    if arr.ndim == 3 and arr.shape[1] == 1:
        arr = arr[:, 0, :]
    if arr.ndim != 2:
        raise ValueError(
            f"Expected 2D array [T, F] or 3D [T,1,F], got shape {arr.shape}"
        )

    T, F = arr.shape

    # 레전드 라벨 준비
    if not feature_names or len(feature_names) != F:
        feature_names = [f"var{i+1}" for i in range(F)]

    fig = None
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.figure

    x = np.arange(T)
    for i in range(F):
        ax.plot(x, arr[:, i], label=feature_names[i])

    ax.set_xlabel("Time")
    ax.set_ylabel("Value")
    ax.set_title(title if title is not None else f"Time Series ({F} features)")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best", ncol=2, fontsize=8)

    # 외부에서 subplot 레이아웃을 관리할 수 있도록 ax가 주어지면 tight_layout은 호출하지 않음
    if ax is None and fig is not None:
        plt.tight_layout()

    if save_path is not None and fig is not None:
        dir_ = os.path.dirname(save_path)
        if dir_:
            os.makedirs(dir_, exist_ok=True)
        fig.savefig(save_path, dpi=150)

    # 외부 축(ax)로 그린 경우 show/close는 호출자가 제어
    if ax is None and fig is not None:
        if show:
            plt.show()
        else:
            plt.close(fig)

    if close is True and fig is not None:
        plt.close(fig)


def plot_diagnostics_triplet(
    t2_array,
    spe_array,
    CI_array,
    thresholds: list = None,
    sigma_levels=None,
    title=None,
    x_label="Time",
    y_labels=("T²", "SPE", "CI"),
    figsize=(12, 10),
    save_path=None,
    show=True,
    x_start=None,
    x_tick_step=1000,
    downsample: int = 1,
    tight_layout: bool = True,
    dpi: int = 150,
    png_compress_level=1,
):
    """
    세 개의 진단 시계열을 하나의 Figure에 3x1 서브플롯으로 그립니다.

    - 상단: T²(Hotelling)
    - 중단: SPE
    - 하단: CI(Comprehensive Index; CI)

    CI 서브플롯에만 `thresholds`로 전달된 값들을 수평선으로 표시합니다.

    Args:
        t2_array: (N,) 형태의 1D 배열
        spe_array: (N,) 형태의 1D 배열
        CI_array: (N,) 형태의 1D 배열
        thresholds: 수평 임계선 값 리스트/튜플(예: diagnosis_thresholds 결과)
        sigma_levels: 각 임계선 라벨 리스트(예: ["3σ", "4.5σ", "6σ"]). None이면 자동 라벨.
        title: 전체 Figure 타이틀
        x_label: 공통 x축 라벨(기본: "Time")
        y_labels: 각 서브플롯 y축 라벨 튜플(기본: ("T²", "SPE", "CI"))
        figsize: Figure 크기
        save_path: 저장 경로. None이면 저장하지 않음
        show: True면 화면 표시, False면 닫음
    """

    def _to_1d(a):
        if isinstance(a, pd.Series):
            return a.values
        return np.asarray(a).reshape(-1)

    t2 = _to_1d(t2_array[x_start:] if x_start is not None else t2_array)
    spe = _to_1d(spe_array[x_start:] if x_start is not None else spe_array)
    CI = _to_1d(CI_array[x_start:] if x_start is not None else CI_array)

    n = len(CI)
    # x_start가 지정되면 해당 값부터 시작하도록 x축 생성
    if x_start is not None:
        x = np.arange(n) + x_start
    else:
        x = np.arange(n)

    # 다운샘플링(플롯 성능 개선용): 시각화 목적이라면 k=5~20 정도가 효과적
    ds = int(downsample) if downsample is not None else 1
    if ds > 1:
        x = x[::ds]
        t2 = t2[::ds]
        spe = spe[::ds]
        CI = CI[::ds]

    fig, axes = plt.subplots(nrows=3, ncols=1, figsize=figsize, sharex=True)

    # Top: T²
    axes[0].plot(x, t2, color="tab:blue", lw=1.4)
    axes[0].set_ylabel(y_labels[0])
    axes[0].set_title("Hostelling's T² Statistic")
    axes[0].grid(True, alpha=0.3)
    # x축 범위를 x_start부터 최대값까지 고정
    # axes[0].set_xlim(x[0], x[-1]+100)

    # Middle: SPE
    axes[1].plot(x, spe, color="tab:blue", lw=1.4)
    axes[1].set_ylabel(y_labels[1])
    axes[1].set_title("Squared Prediction Error (SPE)")
    axes[1].grid(True, alpha=0.3)
    # axes[1].set_xlim(x[0], x[-1]+100)

    # Bottom: CI (CI) + thresholds
    axes[2].plot(x, CI, color="tab:blue", lw=1.4, label="CI")
    axes[2].set_ylabel(y_labels[2])
    axes[2].set_xlabel(x_label)
    axes[2].set_title("Comprehensive Index (CI)")
    axes[2].grid(True, alpha=0.3)
    # axes[2].set_xlim(x[0], x[-1]+100)

    # 시작 눈금을 포함시키기 위해 눈금 간격을 고정할 수 있는 옵션
    if x_tick_step is not None:
        ticks = (
            np.floor(np.arange(x[0], x[-1] + 1, x_tick_step) / x_tick_step)
            * x_tick_step
        )
        ticks[0] = x_start if x_start is not None else 0
        ticks = np.append(ticks, x[-1]) if ticks[-1] != x[-1] else ticks
        for ax in axes:
            ax.xaxis.set_major_locator(mtick.FixedLocator(ticks))

    threshold_color = ["tab:red", "tab:purple", "tab:green"]
    if thresholds is not None:
        thr = list(thresholds)
        # 최대 3개까지만 표시(요청사항 대응)
        thr = thr[:3]
        if sigma_levels is None:
            sigma_labels = [f"Threshold-{i}" for i in thr]
        else:
            sigma_labels = [
                f"Threshold {i} ({sigma_levels[i]:.1f}\u03c3)" for i in range(len(thr))
            ]
        for val, lab, col in zip(thr, sigma_labels, threshold_color):
            axes[2].axhline(val, color=col, linestyle="--", linewidth=1.2, label=lab)
        axes[2].legend(loc="best", fontsize=8)

    if title:
        fig.suptitle(title, y=0.98)

    if tight_layout:
        fig.tight_layout()
    else:
        # tight_layout은 비용이 큰 편이라, 고정 간격으로 빠르게 배치
        fig.subplots_adjust(hspace=0.25, top=0.92)

    if save_path is not None:
        dir_ = os.path.dirname(save_path)
        if dir_:
            os.makedirs(dir_, exist_ok=True)
        if str(save_path).lower().endswith(".png") and png_compress_level is not None:
            # PNG 저장이 보통 가장 느림: 압축 레벨을 낮추면 저장 시간↓(대신 파일 크기↑)
            fig.savefig(
                save_path,
                dpi=dpi,
                pil_kwargs={"compress_level": int(png_compress_level)},
            )
        else:
            fig.savefig(save_path, dpi=dpi)

    if show:
        plt.show()
    else:
        plt.close(fig)


def plot_distribution_with_stats(
    data,
    bins=50,
    kde=True,
    normal_fit=False,
    figsize=(10, 5),
    title="Distribution with mean/std",
    save_path=None,
    show=True,
    ddof=1,
    block=False,
):
    # 1D 벡터로 변환
    try:
        import torch

        if isinstance(data, torch.Tensor):
            x = data.detach().cpu().numpy().ravel()
        else:
            x = np.asarray(data).ravel()
    except Exception:
        x = np.asarray(data).ravel()

    # 통계
    mu = float(np.mean(x))
    std = float(np.std(x, ddof=ddof))
    var = float(np.var(x, ddof=ddof))
    n = int(x.size)

    # 플롯
    fig, ax = plt.subplots(figsize=figsize)
    used_seaborn = False
    if kde:
        try:
            import seaborn as sns

            sns.histplot(
                x, bins=bins, kde=True, stat="density", color="tab:blue", ax=ax
            )
            used_seaborn = True
        except Exception:
            ax.hist(x, bins=bins, density=True, color="tab:blue", alpha=0.7)
    else:
        ax.hist(x, bins=bins, density=True, color="tab:blue", alpha=0.7)

    # 평균/±σ 라인
    ax.axvline(mu, color="red", lw=2, label=f"mean = {mu:.3f}")
    ax.axvline(
        mu - std, color="purple", ls="--", lw=1.5, label=f"mean-σ = {mu - std:.3f}"
    )
    ax.axvline(
        mu + std, color="purple", ls="--", lw=1.5, label=f"mean+σ = {mu + std:.3f}"
    )

    # 정규분포 피팅(선택)
    if normal_fit:
        try:
            from scipy.stats import norm

            xs = np.linspace(np.min(x), np.max(x), 500)
            ax.plot(
                xs,
                norm.pdf(xs, loc=mu, scale=std),
                color="orange",
                lw=2,
                label="Normal fit (μ, σ)",
            )
        except Exception:
            pass

    # 텍스트 박스
    text = f"n = {n}\nmean = {mu:.3f}\nstd = {std:.3f}\nvar = {var:.3f}"
    ax.text(
        0.02,
        0.98,
        text,
        transform=ax.transAxes,
        va="top",
        ha="left",
        bbox=dict(boxstyle="round", fc="white", ec="gray", alpha=0.8),
    )

    ax.set_xlabel("Value")
    ax.set_ylabel("Density" if used_seaborn or kde else "Frequency")
    ax.set_title(title)
    ax.legend(loc="best")
    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150)

    if show:
        plt.show(block=block)
    else:
        plt.close(fig)

    return {"mean": mu, "std": std, "var": var, "count": n}


def make_model_path_based_timestamp(
    base="models", make: bool = False, prefix: str = None, tz="Asia/Seoul"
):
    # tzinfo 설정: zoneinfo 우선, 실패 시 pytz로 폴백
    try:
        from zoneinfo import ZoneInfo  # Python 3.9+

        tzinfo = ZoneInfo(tz)
    except Exception:
        import pytz  # pip install pytz 필요

        tzinfo = pytz.timezone(tz)

    now = datetime.now(tzinfo)
    timestamp = now.strftime("26%m%d_%H%M%S")
    if prefix is None:
        path = os.path.join(base, timestamp)
    else:
        path = os.path.join(base, f"{prefix}_{timestamp}")
    if make:
        os.makedirs(path, exist_ok=True)
    return path


def last_index_of(tensor, feature_idx, value, operator="=="):
    # NOTE: 반환값은 "조건을 만족하는 마지막 인덱스 + 1" 입니다. (없으면 0)
    # 기존 구현은 torch.Tensor에 대해 np.array(tensor)로 전체 복사 변환이 발생해 매우 느릴 수 있어,
    # torch 입력은 torch 연산으로만 처리합니다.

    if isinstance(tensor, torch.Tensor):
        if tensor.ndim < 2:
            raise ValueError(
                f"last_index_of expects 2D tensor, got shape {tuple(tensor.shape)}"
            )

        col = tensor[:, feature_idx]
        if operator == "==":
            mask = col == value
        elif operator == "<":
            mask = col < value
        elif operator == "<=":
            mask = col <= value
        elif operator == ">":
            mask = col > value
        elif operator == ">=":
            mask = col >= value
        elif operator == "!=":
            mask = col != value
        else:
            raise ValueError(f"Unsupported operator: {operator}")

        if not bool(mask.any()):
            return 0

        # 마지막 True 위치를 찾기 위해 뒤집어서 argmax 사용
        # (nonzero로 전체 인덱스 배열을 만들지 않아 메모리/시간이 유리할 수 있음)
        flipped = torch.flip(mask, dims=(0,))
        last_true_from_end = int(torch.argmax(flipped.to(dtype=torch.int8)).item())
        last_true = int(mask.shape[0] - 1 - last_true_from_end)
        return last_true + 1

    A = np.asarray(tensor)
    if A.ndim < 2:
        raise ValueError(
            f"last_index_of expects 2D array-like, got shape {getattr(A, 'shape', None)}"
        )

    col = A[:, feature_idx]
    if operator == "==":
        mask = col == value
    elif operator == "<":
        mask = col < value
    elif operator == "<=":
        mask = col <= value
    elif operator == ">":
        mask = col > value
    elif operator == ">=":
        mask = col >= value
    elif operator == "!=":
        mask = col != value
    else:
        raise ValueError(f"Unsupported operator: {operator}")

    idx = np.flatnonzero(mask)
    if idx.size == 0:
        return 0
    return int(idx[-1] + 1)


def get_start_index(tensor, feature_idx):
    start_idx = 0
    start_idx = np.max(
        [start_idx, last_index_of(tensor, feature_idx, operator="<")]
    )  # 0 means minimum voltage
    start_idx = np.max(
        [start_idx, last_index_of(tensor, feature_idx, operator="==")]
    )  # 0 means minimum voltage
    start_idx = np.max(
        [start_idx, last_index_of(tensor, feature_idx, operator=">")]
    )  # 5 means maximum voltage
    return start_idx


def is_abnormal(tensor):
    abnormal_data = False
    if (
        last_index_of(tensor, 0, 0, operator="<") != 0
        or last_index_of(tensor, 0, 5, operator=">") != 0
    ):
        abnormal_data = True
    return abnormal_data


def normalize_columns_0_to_1(X, eps=1e-12, exclude_cols=None, exclude_ranges=None):
    """
    입력 행렬의 각 열을 [0,1] 범위로 정규화합니다.

    - torch.Tensor와 numpy.ndarray 모두 지원합니다.
    - 분모(최대-최소)가 0인 열은 eps로 치환합니다.

    Args:
        X: (n×m) 배열/텐서
        eps: 분모가 0일 때 사용할 작은 값
        exclude_cols: 정규화에서 제외할 컬럼 인덱스들(0-based). 예: [0, 1, 2]
        exclude_ranges: 정규화에서 제외할 컬럼 구간들(0-based, inclusive).
            예: [(110, 220)]  -> 110~220 컬럼은 정규화하지 않음

    Returns:
        동일 타입(torch.Tensor 또는 np.ndarray)의 정규화된 행렬
    """

    def _build_normalize_mask(n_cols: int):
        mask = np.ones(n_cols, dtype=bool)

        if exclude_cols is not None:
            try:
                cols = np.asarray(exclude_cols)
            except Exception:
                cols = None

            if cols is None:
                for c in exclude_cols:
                    try:
                        ci = int(c)
                    except Exception:
                        continue
                    if 0 <= ci < n_cols:
                        mask[ci] = False
            else:
                # 벡터화: 유효 범위 인덱스만 False 처리
                try:
                    cols_i = cols.astype(np.int64, copy=False).ravel()
                except Exception:
                    cols_i = np.array([int(x) for x in cols.ravel()], dtype=np.int64)

                if cols_i.size:
                    valid = (cols_i >= 0) & (cols_i < n_cols)
                    if np.any(valid):
                        mask[cols_i[valid]] = False

        if exclude_ranges is not None:
            for r in exclude_ranges:
                if r is None:
                    continue
                try:
                    start, end = r
                except Exception:
                    continue
                try:
                    start_i = int(start)
                    end_i = int(end)
                except Exception:
                    continue
                if end_i < start_i:
                    start_i, end_i = end_i, start_i
                start_i = max(start_i, 0)
                end_i = min(end_i, n_cols - 1)
                if start_i <= end_i:
                    mask[start_i : end_i + 1] = False

        return mask

    if isinstance(X, torch.Tensor):
        # 불필요한 dtype 변환/복사 최소화
        if torch.is_floating_point(X):
            Xf = X
        else:
            Xf = X.to(dtype=torch.float64)
        if Xf.ndim != 2:
            raise ValueError(f"Expected 2D tensor (n×m), got shape {tuple(Xf.shape)}")

        n_cols = int(Xf.shape[1])
        np_mask = _build_normalize_mask(n_cols)
        if not np.any(np_mask):
            return Xf

        col_mask = torch.as_tensor(np_mask, dtype=torch.bool, device=Xf.device)
        X_sel = Xf[:, col_mask]
        col_min = X_sel.min(dim=0, keepdim=True).values
        col_max = X_sel.max(dim=0, keepdim=True).values
        denom = torch.clamp(col_max - col_min, min=float(eps))

        out = Xf.clone()  # 원본 보존(기존 동작 유지)
        out[:, col_mask] = (X_sel - col_min) / denom
        return out
    else:
        A = np.asarray(X, dtype=float)
        if A.ndim != 2:
            raise ValueError(f"Expected 2D array (n×m), got shape {A.shape}")

        n_cols = int(A.shape[1])
        mask = _build_normalize_mask(n_cols)
        if not np.any(mask):
            return A

        A_sel = A[:, mask]
        col_min = np.min(A_sel, axis=0, keepdims=True)
        col_max = np.max(A_sel, axis=0, keepdims=True)
        denom = np.maximum(col_max - col_min, eps)

        out = A.copy()
        out[:, mask] = (A_sel - col_min) / denom
        return out


def normalize_columns_by_mean_std(X, exclude_cols=None, exclude_ranges=None):
    """입력 행렬의 각 열을 (y - mu_y) / sigma_y 로 정규화합니다.

    - torch.Tensor와 numpy.ndarray 모두 지원합니다.
    - sigma_y(표준편차)가 0인 열은 eps로 치환합니다.
    - 특정 열(또는 열 구간)을 정규화에서 제외할 수 있습니다.

    Args:
        X: (n×m) 배열/텐서
        eps: 분모(std)가 0일 때 사용할 작은 값
        exclude_cols: 정규화에서 제외할 컬럼 인덱스들(0-based). 예: [0, 1, 2]
        exclude_ranges: 정규화에서 제외할 컬럼 구간들(0-based, inclusive).
            예: [(110, 220)]  -> 110~220 컬럼은 정규화하지 않음

    Returns:
        동일 타입(torch.Tensor 또는 np.ndarray)의 정규화된 행렬
    """

    def _build_normalize_mask(n_cols: int):
        mask = np.ones(n_cols, dtype=bool)

        if exclude_cols is not None:
            for c in exclude_cols:
                try:
                    ci = int(c)
                except Exception:
                    continue
                if 0 <= ci < n_cols:
                    mask[ci] = False

        if exclude_ranges is not None:
            for r in exclude_ranges:
                if r is None:
                    continue
                try:
                    start, end = r
                except Exception:
                    continue
                try:
                    start_i = int(start)
                    end_i = int(end)
                except Exception:
                    continue
                if end_i < start_i:
                    start_i, end_i = end_i, start_i
                start_i = max(start_i, 0)
                end_i = min(end_i, n_cols - 1)
                if start_i <= end_i:
                    mask[start_i : end_i + 1] = False

        return mask

    if isinstance(X, torch.Tensor):
        Xf = X.to(dtype=torch.float64)
        if Xf.ndim != 2:
            raise ValueError(f"Expected 2D tensor (n×m), got shape {tuple(Xf.shape)}")

        n_cols = int(Xf.shape[1])
        np_mask = _build_normalize_mask(n_cols)
        if not np.any(np_mask):
            return Xf

        col_mask = torch.as_tensor(np_mask, dtype=torch.bool, device=Xf.device)
        X_sel = Xf[:, col_mask]
        mu = X_sel.mean(dim=0, keepdim=True)
        std = X_sel.std(dim=0, keepdim=True, unbiased=False)

        out = Xf.clone()
        out[:, col_mask] = (X_sel - mu) / std
        return out

    A = np.asarray(X, dtype=float)
    if A.ndim != 2:
        raise ValueError(f"Expected 2D array (n×m), got shape {A.shape}")

    n_cols = int(A.shape[1])
    mask = _build_normalize_mask(n_cols)
    if not np.any(mask):
        return A

    A_sel = A[:, mask]
    mu = np.mean(A_sel, axis=0, keepdims=True)
    std = np.std(A_sel, axis=0, keepdims=True)

    out = A.copy()
    out[:, mask] = (A_sel - mu) / std
    return out


def _as_mapping(config: Any) -> dict:
    """Namespace/dict/dataclass/object 모두 dict 비슷하게 변환."""
    if config is None:
        return {}
    if isinstance(config, Mapping):
        return dict(config)
    if is_dataclass(config):
        return asdict(config)
    if hasattr(config, "__dict__"):
        return dict(vars(config))
    return {"config": config}


def _fmt(v: Any, max_len: int = 180) -> str:
    if v is None:
        s = "None"
    elif isinstance(v, (list, tuple, set)):
        s = "[" + ", ".join(map(str, list(v))) + "]"
    else:
        s = str(v)

    s = s.replace("\n", "\\n")
    if len(s) > max_len:
        s = s[: max_len - 3] + "..."
    return s


def print_sim_config(
    title: str = "Simulation Settings",
    config: Any = None,
    include: Sequence[str] | None = None,
    exclude: Sequence[str] | None = None,
    extra: Mapping[str, Any] | None = None,
    sort_keys: bool = True,
    show_system: bool = True,
) -> None:
    """
    시뮬레이션 세팅(실험 설정)을 깔끔하게 출력.

    - config: argparse.Namespace, dict, dataclass, 또는 일반 객체(속성 기반)
    - include: 출력할 key들만 선택 (None이면 전체)
    - exclude: 제외할 key들
    - extra: 추가로 더 붙이고 싶은 값들(예: device, seed, git hash 등)
    """
    cfg = _as_mapping(config)

    if include is not None:
        cfg = {k: cfg.get(k, None) for k in include}

    if exclude is not None:
        for k in exclude:
            cfg.pop(k, None)

    if extra:
        # extra가 우선권(override)
        cfg.update(dict(extra))

    items = list(cfg.items())
    if sort_keys:
        items.sort(key=lambda kv: kv[0])

    # 시스템 정보(선택)
    sys_lines = []
    if show_system:
        sys_lines = [
            ("timestamp", datetime.now().strftime("%Y-%m-%d %H:%M:%S")),
            ("python", sys.version.split()[0]),
            (
                "platform",
                f"{platform.system()} {platform.release()} ({platform.machine()})",
            ),
            ("cwd", os.getcwd()),
        ]

    # 출력 폭 계산
    all_items = sys_lines + items
    key_w = max([len(str(k)) for k, _ in all_items] + [3])
    val_w = 0
    for _, v in all_items:
        val_w = max(val_w, len(_fmt(v)))

    line_w = min(max(key_w + 3 + val_w + 2, 60), 140)

    def hr(ch: str = "=") -> str:
        return ch * line_w

    print(hr("="))
    print(title.center(line_w))
    print(hr("="))

    if sys_lines:
        for k, v in sys_lines:
            print(f"{str(k):<{key_w}} : {_fmt(v)}")
        print(hr("-"))

    for k, v in items:
        print(f"{str(k):<{key_w}} : {_fmt(v)}")

    print(hr("="))


from pathlib import Path
import re


def read_learning_case_from_sim_config(models_dir: str) -> int:
    sim_path = Path(models_dir) / "sim_config.txt"
    text = sim_path.read_text(encoding="utf-8")

    m = re.search(r"(?m)^\s*learning_case\s*:\s*(\d+)\s*$", text)
    if not m:
        raise ValueError(f"learning_case를 찾지 못했습니다: {sim_path}")
    return int(m.group(1))


def _coerce_sim_config_value(raw: str):
    s = str(raw).strip()
    if s == "":
        return ""

    sl = s.lower()
    if sl in ("true", "false"):
        return sl == "true"

    # int
    if re.fullmatch(r"-?\d+", s):
        try:
            return int(s)
        except Exception:
            pass

    # float
    if re.fullmatch(r"-?\d+(?:\.\d+)?", s):
        try:
            return float(s)
        except Exception:
            pass

    # list/tuple/dict literal (e.g., [1, 2, 3])
    if (
        (s.startswith("[") and s.endswith("]"))
        or (s.startswith("(") and s.endswith(")"))
        or (s.startswith("{") and s.endswith("}"))
    ):
        try:
            return ast.literal_eval(s)
        except Exception:
            return s

    return s


def read_values_from_sim_config(
    models_dir: str,
    names_to_find=None,
    *,
    strict: bool = True,
    exclude_keys: list | str | None = None,
) -> SimpleNamespace:
    """models_dir/sim_config.txt에서 특정 키들의 값을 읽어 SimpleNamespace로 반환.

    Args:
        models_dir: sim_config.txt가 있는 디렉터리 (예: ./models/260203_163103)
        names_to_find: 찾을 키 이름(문자열) 또는 키 이름 리스트/튜플. None이면 파일 내 모든 키를 반환.
        strict: True면 누락된 키가 있을 때 예외 발생, False면 누락 키는 None으로 반환 (names_to_find가 None이면 무시됨)
        exclude_keys: 결과에서 제외할 키 이름(문자열) 또는 리스트.

    Returns:
        SimpleNamespace: obj.key 형태로 접근 가능
        value는 bool/int/float/list/dict literal 등을 가능한 범위에서 자동 변환합니다.
    """
    sim_path = Path(models_dir) / "sim_config.txt"
    text = sim_path.read_text(encoding="utf-8")

    # sim_config.txt는 보통 "key : value" 형태로 정렬되어 출력됨
    parsed = {}
    for line in text.splitlines():
        if ":" not in line:
            continue
        # 구분선(====, ----) 등 제외
        stripped = line.strip()
        if not stripped or set(stripped) <= {"=", "-"}:
            continue

        k, v = line.split(":", 1)
        k = k.strip()
        v = v.strip()
        if k:
            parsed[k] = v

    # 제외할 키 처리
    if exclude_keys is None:
        exclude_set = set()
    elif isinstance(exclude_keys, str):
        exclude_set = {exclude_keys}
    else:
        exclude_set = set(exclude_keys)

    out = {}

    if names_to_find is None:
        # 모든 키 반환 모드
        keys = list(parsed.keys())
        missing = []  # 모든 키를 가져오므로 missing은 없음
    else:
        # 특정 키 반환 모드
        if isinstance(names_to_find, (str, bytes)):
            keys = [names_to_find]
        else:
            keys = list(names_to_find)
        missing = []

    for k in keys:
        if k in exclude_set:
            continue
        if k in parsed:
            out[k] = _coerce_sim_config_value(parsed[k])
        elif names_to_find is not None:
            # names_to_find가 명시되었는데 키가 없는 경우에만 처리
            missing.append(k)
            out[k] = None

    if strict and missing:
        raise ValueError(
            f"sim_config.txt에서 키를 찾지 못했습니다: {missing} (file={sim_path})"
        )
    return SimpleNamespace(**out)


def get_preprocessing_and_skip_charge_ready(learning_case: int):
    if learning_case == 1:
        PREPROCESSING = False
        SKIP_CHARGE_READY = True
    elif learning_case == 2:
        PREPROCESSING = True
        SKIP_CHARGE_READY = True
    elif learning_case == 3:
        PREPROCESSING = True
        SKIP_CHARGE_READY = False
    elif learning_case == 4:
        PREPROCESSING = False
        SKIP_CHARGE_READY = False
    else:
        raise ValueError(f"지원하지 않는 learning_case 값입니다: {learning_case}")
    return PREPROCESSING, SKIP_CHARGE_READY


def load_vehicle_ids_used_for_training(sim_config_path: str) -> list[int]:
    with open(sim_config_path, "r", encoding="utf-8") as f:
        for line in f:
            if "Vehicle IDs used for training" not in line:
                continue
            m = re.search(r":\s*(\[.*\])\s*$", line)
            if not m:
                raise ValueError(
                    f"sim_config.txt에서 차량 ID 리스트를 찾았지만 파싱에 실패했습니다: {line.strip()}"
                )
            value = ast.literal_eval(m.group(1))
            if isinstance(value, (list, tuple)):
                return [int(x) for x in value]
            raise ValueError(
                f"Vehicle IDs used for training 값이 list/tuple이 아닙니다: {type(value)}"
            )

    raise FileNotFoundError(
        f"sim_config.txt에서 'Vehicle IDs used for training' 라인을 찾지 못했습니다: {sim_config_path}"
    )


def get_input_dimensions(BATTERY_TYPE: str):
    dim_dict = dict()
    dim_dict["x"] = 2  # [estimated pack voltage 1, estimated pack volatage 2]
    dim_dict["y"] = (
        110 if BATTERY_TYPE == "QAS" else 85
    )  # DTI 일경우 85 이고 QAS일 경우 110인듯 [각 셀의 voltage]
    dim_dict["z"] = (
        110 if BATTERY_TYPE == "QAS" else 85
    )  # DTI 일경우 85 이고 QAS일 경우 110인듯 [각 셀의 estimated voltage diviation]
    dim_dict["q"] = 3  # [Board Temperature, Board-end SOC, Current]

    dim_dict["x2"] = 2  # [estimated pack SOC 1, estimated pack SOC 2]
    dim_dict["y2"] = (
        110 if BATTERY_TYPE == "QAS" else 85
    )  # DTI 일경우 85 이고 QAS일 경우 110인듯 [각 셀의 SOC]
    dim_dict["z2"] = (
        110 if BATTERY_TYPE == "QAS" else 85
    )  # DTI 일경우 85 이고 QAS일 경우 110인듯 [각 셀의 estimated SOC diviation]
    dim_dict["q2"] = 4  # [Board Temperature, Board-end SOC, Velocity, Current]

    return dim_dict


def preprocess_loaded_tensor(
    tensor,
    tensorx,
    dim_dict,
    BATTERY_TYPE,
    PREPROCESSING,
    SKIP_CHARGE_READY,
    normalize_dx: bool = False,
    normalize_val: float = 1.0,
):
    if BATTERY_TYPE == "DTI":
        # Remove unstable initial data dynamically
        # Uses column 0 to detect convergence (diff < 0.1 for 20 steps)
        start_idx_dti = 0
        if SKIP_CHARGE_READY:
            start_idx_dti = get_convergence_start_index(
                tensor, threshold=0.1, stable_window_size=20
            )
            # print(
            #     f"DTI: Detected convergence start index at {start_idx_dti} based on voltage stability."
            # )
            # print(
            #     f"Length before trimming: {tensor.shape[0]}, after trimming: {tensor.shape[0] - start_idx_dti}"
            # )
        tensor = tensor[start_idx_dti:, :]
        tensorx = tensorx[start_idx_dti:, :]

        # Voltage features scaling (mV -> V)
        # Indices 0 to 171: Pack Voltage(2) + Cell Voltages(85) + Voltage Deviations(85) = 172 columns
        volt_end = dim_dict["x"] + dim_dict["y"]
        # SOC features scaling (% -> Ratio)
        # Index 173: Board SOC (dim_dict["q"]=3: Temp(172), SOC(173), Current(174))
        soc_idx = volt_end + dim_dict["z"] + 1
        tensor = tensor.float()
        tensor[:, 1:volt_end] = tensor[:, 1:volt_end] / 1000.0
        tensor[:, soc_idx] = tensor[:, soc_idx] / 100.0

        # Target tensor (tensorx) scaling
        # Indices 0 to 171: Pack SOC(2) + Cell SOC(85) + SOC Deviations(85)
        # Index 173: Board SOC (dim_dict["q2"]=4)
        soc_end_x = dim_dict["x2"]
        soc_true_x = soc_end_x + dim_dict["y2"]
        soc_idx_x = soc_true_x + dim_dict["z2"] + 1

        tensorx = tensorx.float()
        # Note: Pack SOC (index 0-1) in DTI vin_3 appears to have large values (1200+).
        # Dividing by 100 as per general SOC % assumption.
        tensorx[:, :soc_end_x] = tensorx[:, :soc_end_x] / 1000.0
        tensorx[:, soc_end_x:soc_true_x] = tensorx[:, soc_end_x:soc_true_x] / 100.0
        tensorx[:, soc_idx_x] = tensorx[:, soc_idx_x] / 100.0
        tensorx[:, soc_idx_x + 1] = tensorx[:, soc_idx_x + 1] / 100.0  # velocity

        return tensor.double(), tensorx.double()

    if BATTERY_TYPE == "QAS":
        tensorx[:, dim_dict["x2"] + dim_dict["y2"] + dim_dict["z2"] + 2] = 0

    if SKIP_CHARGE_READY:
        charge_idx = 224 if BATTERY_TYPE == "QAS" else 174
        start_idx = last_index_of(
            tensor, feature_idx=charge_idx, value=-500, operator="<"
        )  # -500 means minimum current during charge ready
        if BATTERY_TYPE == "DTI":
            start_idx = np.max(
                [
                    start_idx,
                    last_index_of(tensor, feature_idx=0, value=5.2, operator=">="),
                ]
            )
            start_idx = np.max(
                [
                    start_idx,
                    last_index_of(tensor, feature_idx=0, value=0, operator="<="),
                ]
            )
        tensor = tensor[start_idx:, :]
        tensorx = tensorx[start_idx:, :]

    if PREPROCESSING:
        start_idx = np.max(
            [0, last_index_of(tensor, feature_idx=0, value=0, operator="<=")]
        )  # 0 means minimum voltage
        start_idx = np.max(
            [start_idx, last_index_of(tensor, feature_idx=0, value=5, operator=">")]
        )  # 5 means maximum voltage
        cell_div_range = (
            dim_dict["x"] + dim_dict["y"],
            dim_dict["x"] + dim_dict["y"] + dim_dict["z"] - 1,
        )
        tensor = normalize_columns_0_to_1(
            tensor[start_idx:, :], exclude_ranges=[cell_div_range]
        )
        tensorx = normalize_columns_0_to_1(
            tensorx[start_idx:, :], exclude_ranges=[cell_div_range]
        )
        if normalize_dx:
            dx_s = int(dim_dict["x"] + dim_dict["y"])
            dx_e = int(dx_s + dim_dict["z"])
            dx2_s = int(dim_dict["x2"] + dim_dict["y2"])
            dx2_e = int(dx2_s + dim_dict["z2"])

            # range 기반 advanced indexing은 불필요한 복사/할당이 발생할 수 있어 슬라이스로 교체
            # torch.Tensor면 in-place 연산으로 추가 할당을 줄임
            if isinstance(tensor, torch.Tensor):
                tensor[:, dx_s:dx_e].div_(normalize_val)
            else:
                tensor[:, dx_s:dx_e] = np.asarray(tensor[:, dx_s:dx_e]) / normalize_val

            if isinstance(tensorx, torch.Tensor):
                tensorx[:, dx2_s:dx2_e].div_(normalize_val)
            else:
                tensorx[:, dx2_s:dx2_e] = (
                    np.asarray(tensorx[:, dx2_s:dx2_e]) / normalize_val
                )
    return tensor, tensorx


def plot_ae_output_distribution(
    df_data, save_path=None, show=True, normal_fit=True, do_plot=False
):
    if not do_plot:
        return
    # plot_distribution_with_stats(
    #     ERRORU,
    #     bins=60,
    #     kde=True,
    #     normal_fit=normal_fit,
    #     title="Cell voltage sample Distribution",
    #     save_path=save_path,
    #     show=show,
    # )
    # plot_distribution_with_stats(
    #     ERRORX,
    #     bins=60,
    #     kde=True,
    #     normal_fit=normal_fit,
    #     title="Cell SoC sample Distribution",
    #     save_path=save_path,
    #     show=show,
    # )
    plot_distribution_with_stats(
        np.array(df_data.iloc[:, 0]),
        bins=60,
        kde=True,
        normal_fit=normal_fit,
        title="Cell $zu_2$ sample Distribution",
        save_path=save_path,
        show=show,
    )
    plot_distribution_with_stats(
        np.array(df_data.iloc[:, 1]),
        bins=60,
        kde=True,
        normal_fit=normal_fit,
        title="Cell $zx_2$ sample Distribution",
        save_path=save_path,
        show=show,
    )
    plot_distribution_with_stats(
        np.array(df_data.iloc[:, 2]),
        bins=60,
        kde=True,
        normal_fit=normal_fit,
        title="Cell $zu_1$ sample Distribution",
        save_path=save_path,
        show=show,
    )
    plot_distribution_with_stats(
        np.array(df_data.iloc[:, 3]),
        bins=60,
        kde=True,
        normal_fit=normal_fit,
        title="Cell $zx_1$ sample Distribution",
        save_path=save_path,
        show=show,
    )
    plot_distribution_with_stats(
        np.array(df_data.iloc[:, 4]),
        bins=60,
        kde=True,
        normal_fit=normal_fit,
        title="Cell $ewu$ sample Distribution",
        save_path=save_path,
        show=show,
    )
    plot_distribution_with_stats(
        np.array(df_data.iloc[:, 5]),
        bins=60,
        kde=True,
        normal_fit=normal_fit,
        title="Cell $ewx$ sample Distribution",
        save_path=save_path,
        show=show,
    )


def plot_net_layer_weight_bias_bars(
    net, layer_names=("fc1", "fc2", "fc3"), bins=50, show=True, save_dir=None
):
    """
    layer당 figure 1개 생성:
      - subplot(1,2): weight 분포 / bias 분포
    """
    for layer_name in layer_names:
        layer = getattr(net, layer_name, None)
        if layer is None:
            raise AttributeError(f"net에 '{layer_name}' 레이어가 없습니다.")

        w = layer.weight.detach().cpu().numpy().ravel()
        b = None
        if getattr(layer, "bias", None) is not None:
            b = layer.bias.detach().cpu().numpy().ravel()

        fig, axes = plt.subplots(1, 2, figsize=(10, 3.5))
        fig.suptitle(f"{layer_name} weight/bias distribution")

        axes[0].hist(w, bins=bins)
        axes[0].set_title(f"{layer_name}.weight (n={w.size})")
        axes[0].set_xlabel("value")
        axes[0].set_ylabel("count")

        if b is not None:
            axes[1].hist(b, bins=bins)
            axes[1].set_title(f"{layer_name}.bias (n={b.size})")
            axes[1].set_xlabel("value")
            axes[1].set_ylabel("count")
        else:
            axes[1].set_visible(False)

        fig.tight_layout()

        if save_dir is not None:
            fig.savefig(f"{save_dir}/{layer_name}_weight_bias_hist.png", dpi=150)

        if show:
            plt.show()
        else:
            plt.close(fig)


def plot_net_layer_params_by_index(
    net,
    layer_names=("fc1", "fc2", "fc3"),
    net_name="net",
    *,
    mode="bar",
    max_points=8000,
    show=True,
    save_dir=None,
    figsize=(12, 6),
    bar_width_ratio=0.65,
    edge_color="black",
    edge_linewidth=0.6,
    annotate_values=True,
    annotate_max=60,
    value_fmt="{:.4g}",
    value_fontsize=7,
    normalize_y=True,
    max_xtick_labels=30,  # <- 추가: xtick 라벨 너무 많으면 일부만 표시
):
    for layer_name in layer_names:
        layer = getattr(net, layer_name, None)
        if layer is None:
            raise AttributeError(f"net에 '{layer_name}' 레이어가 없습니다.")

        w = layer.weight.detach().cpu().numpy().ravel()
        b = (
            layer.bias.detach().cpu().numpy().ravel()
            if layer.bias is not None
            else None
        )

        def _downsample(arr, max_n):
            n = arr.size
            if n <= max_n:
                idx = np.arange(n)
                return idx, arr
            step = int(np.ceil(n / max_n))
            idx = np.arange(0, n, step)
            return idx, arr[idx]

        def _bar_width_from_idx(idx):
            if idx.size <= 1:
                base = 1.0
            else:
                diffs = np.diff(idx)
                base = float(np.median(diffs)) if diffs.size else 1.0
                if base <= 0:
                    base = 1.0
            return base * float(bar_width_ratio)

        def _normalize_minus1_to_1(y):
            y = np.asarray(y, dtype=float)
            y_min = float(np.nanmin(y))
            y_max = float(np.nanmax(y))
            if not np.isfinite(y_min) or not np.isfinite(y_max) or (y_max - y_min) == 0:
                return np.zeros_like(y, dtype=float)
            return 2.0 * (y - y_min) / (y_max - y_min) - 1.0

        def normalize_by_max_abs(y):
            y = np.asarray(y, dtype=float)
            max_abs = np.nanmax(np.abs(y))
            if not np.isfinite(max_abs) or max_abs == 0:
                return np.zeros_like(y, dtype=float)
            return y / max_abs

        def _annotate(ax, xs, ys):
            if not annotate_values:
                return
            n = len(xs)
            if n == 0:
                return
            if n > int(annotate_max):
                xs = xs[: int(annotate_max)]
                ys = ys[: int(annotate_max)]

            dy = 0.03 if normalize_y else 0.0
            for x0, y0 in zip(xs, ys):
                ax.text(
                    x0,
                    y0 + (dy if y0 >= 0 else -dy),
                    value_fmt.format(y0),
                    ha="center",
                    va="bottom" if y0 >= 0 else "top",
                    fontsize=value_fontsize,
                    rotation=0,  # <- 가로
                )

        def _set_integer_xticks(ax, x_positions):
            # x_positions는 이미 정수(1-based)라고 가정
            x_positions = np.asarray(x_positions, dtype=int)
            n = x_positions.size
            if n == 0:
                return

            if n <= max_xtick_labels:
                ticks = x_positions
            else:
                # 너무 많으면 일부만(그래도 정수만)
                ticks = np.unique(
                    np.linspace(x_positions.min(), x_positions.max(), max_xtick_labels)
                    .round()
                    .astype(int)
                )

            ax.set_xticks(ticks)
            ax.xaxis.set_major_locator(
                mtick.FixedLocator(ticks)
            )  # <- 소수 tick 방지(강제)
            ax.set_xticklabels([str(int(t)) for t in ticks])

        w_idx, w_plot = _downsample(w, max_points)
        if b is not None:
            b_idx, b_plot = _downsample(b, max_points)

        if normalize_y:
            w_plot = normalize_by_max_abs(w_plot)
            if b is not None:
                b_plot = normalize_by_max_abs(b_plot)

        # x축을 1-based 인덱스로 (요청: 2개면 1,2만)
        xw = w_idx + 1
        xb = (b_idx + 1) if b is not None else None

        fig, axes = plt.subplots(1, 2 if b is not None else 1, figsize=figsize)
        if not isinstance(axes, np.ndarray):
            axes = np.array([axes])

        fig.suptitle(
            f"{layer_name}: value vs index (mode={mode})"
            + (" [y normalized to -1~1]" if normalize_y else "")
        )

        # weight
        ax = axes[0]
        if mode == "bar":
            width = _bar_width_from_idx(xw)  # 1-based라도 간격은 동일
            ax.bar(
                xw,
                w_plot,
                width=width,
                edgecolor=edge_color,
                linewidth=edge_linewidth,
                color="tab:blue",
                align="center",
            )
            _annotate(ax, xw, w_plot)
        else:
            ax.plot(xw, w_plot, linewidth=0.8)

        ax.set_title(f"{layer_name}.weight  (orig_n={w.size}, plotted={w_plot.size})")
        ax.set_xlabel("weight index")
        ax.set_ylabel("value" + (" (normalized)" if normalize_y else ""))
        ax.grid(True, alpha=0.2)
        if normalize_y:
            ax.set_ylim(-1.05, 1.05)

        _set_integer_xticks(ax, xw)
        ax.set_xlim(xw.min() - 0.5, xw.max() + 0.5)

        # bias
        if b is not None:
            ax = axes[1]
            if mode == "bar":
                width = _bar_width_from_idx(xb)
                ax.bar(
                    xb,
                    b_plot,
                    width=width,
                    edgecolor=edge_color,
                    linewidth=edge_linewidth,
                    color="tab:orange",
                    align="center",
                )
                _annotate(ax, xb, b_plot)
            else:
                ax.plot(xb, b_plot, linewidth=0.8)

            ax.set_title(f"{layer_name}.bias  (orig_n={b.size}, plotted={b_plot.size})")
            ax.set_xlabel("bias index")
            ax.set_ylabel("value" + (" (normalized)" if normalize_y else ""))
            ax.grid(True, alpha=0.2)
            if normalize_y:
                ax.set_ylim(-1.05, 1.05)

            _set_integer_xticks(ax, xb)
            ax.set_xlim(xb.min() - 0.5, xb.max() + 0.5)

        fig.tight_layout()

        if save_dir is not None:
            if normalize_y:
                fig.savefig(
                    f"{save_dir}/{net_name}_{layer_name}_params_by_index_normalized.png",
                    dpi=150,
                )
            else:
                fig.savefig(
                    f"{save_dir}/{net_name}_{layer_name}_params_by_index.png", dpi=150
                )

        if show:
            plt.show()
        else:
            plt.close(fig)


def find_max_idx_from_array(array):
    dv = 0.00001
    if array.ndim == 1:
        max_err = np.asarray(array, dtype=float).max()
        max_err_time = np.where(np.asarray(array, dtype=float) > (max_err - dv))
        return max_err_time
    elif array.ndim == 2:
        max_array = np.asarray(array, dtype=float).max(axis=1)
        max_err = max_array.max()
        max_err_time = np.where(max_array > (max_err - dv))
        max_err_idx = np.where(
            np.asarray(array[max_err_time], dtype=float) > (max_err - dv)
        )
        return max_err_time, max_err_idx
    else:
        raise ValueError("Input array must be 1D or 2D.")


def find_min_idx_from_array(array):
    dv = 0.00001
    if array.ndim == 1:
        min_err = np.asarray(array, dtype=float).min()
        min_err_time = np.where(np.asarray(array, dtype=float) < (min_err + dv))
        return min_err_time
    elif array.ndim == 2:
        min_array = np.asarray(array, dtype=float).min(axis=1)
        min_err = min_array.min()
        min_err_time = np.where(min_array < (min_err + dv))
        min_err_idx = np.where(
            np.asarray(array[min_err_time], dtype=float) < (min_err + dv)
        )
        return min_err_time, min_err_idx
    else:
        raise ValueError("Input array must be 1D or 2D.")


def compare_max_error_data(max_idx_1, max_idx_2, df_data):
    return [
        [
            df_data.iloc[max_idx_1, 0],
            df_data.iloc[max_idx_1, 1],
            df_data.iloc[max_idx_1, 2],
            df_data.iloc[max_idx_1, 3],
            df_data.iloc[max_idx_1, 4],
            df_data.iloc[max_idx_1, 5],
        ],
        [
            df_data.iloc[max_idx_2, 0],
            df_data.iloc[max_idx_2, 1],
            df_data.iloc[max_idx_2, 2],
            df_data.iloc[max_idx_2, 3],
            df_data.iloc[max_idx_2, 4],
            df_data.iloc[max_idx_2, 5],
        ],
    ]


import numpy as np
import matplotlib.pyplot as plt

import numpy as np
import matplotlib.pyplot as plt


def plot_contrib_percent_stacked(
    contrib_array,  # (N,6)
    *,
    feature_names=None,  # 길이 6 리스트 (예: ["zx1","zx2","ewx","zu1","zu2","ewu"])
    t_indices=None,  # 그리고 싶은 t 인덱스들(list/ndarray). None이면 top_k 사용
    val_array=None,  # (N,) optional. top_k 고를 때 사용
    top_k=1e8,  # t_indices가 None일 때, val_array 큰 순으로 top_k
    mode="abs",  # "abs" 권장. ("raw"도 가능하지만 음수면 해석 어려움)
    kind="$T^2$",
    figsize=(14, 5),
    alpha=0.9,
    show=True,
    save_path=None,
    show_line=True,
    plot_step: int | None = None,
    max_bars: int | None = 5000,
    xtick_step: int | None = 1,
    xtick_max_labels: int | None = None,
):
    contrib_array = np.asarray(contrib_array, dtype=float)
    if contrib_array.ndim != 2:
        raise ValueError(
            f"contrib_array는 2D여야 합니다. got shape={contrib_array.shape}"
        )
    n, f = contrib_array.shape

    if feature_names is None:
        feature_names = [f"f{i}" for i in range(f)]
    if len(feature_names) != f:
        raise ValueError(f"feature_names 길이({len(feature_names)}) != feature 수({f})")

    # 어떤 t들을 그릴지 선택
    if t_indices is None:
        if val_array is None:
            raise ValueError(
                "t_indices가 None이면 val_array가 필요합니다(top_k 선택용)."
            )
        val_array = np.asarray(val_array, dtype=float).reshape(-1)
        if val_array.shape[0] != n:
            raise ValueError("val_array 길이가 contrib_array N과 다릅니다.")
        # 큰 T2 기준 top_k 시점 선택
        k = min(int(top_k), n)
        t_indices = np.argsort(val_array)[::-1][:k]
        t_indices = np.sort(t_indices)  # x축 보기 좋게 오름차순 정렬
    else:
        t_indices = np.asarray(t_indices, dtype=int)

    # ================= 성능 최적화 =================
    # 막대(stacked bar)는 K가 커지면 (feature 수 * K) 만큼 Rectangle을 만들어 매우 느려짐.
    # - plot_step: 사용자 지정 간격으로 t를 샘플링
    # - max_bars: 자동으로 최대 막대 개수 제한(기본 5000)
    t_indices = np.asarray(t_indices, dtype=int)
    t_indices = np.unique(t_indices)
    if t_indices.size == 0:
        raise ValueError("t_indices가 비어 있습니다.")
    t_indices = np.sort(t_indices)

    step_i = 1
    if plot_step is not None:
        try:
            step_i = int(plot_step)
        except Exception:
            step_i = 1
        if step_i < 1:
            step_i = 1

    if max_bars is not None:
        try:
            max_bars_i = int(max_bars)
        except Exception:
            max_bars_i = 0
        if max_bars_i > 0 and t_indices.size > max_bars_i:
            auto_step = int(np.ceil(t_indices.size / max_bars_i))
            step_i = max(step_i, auto_step)

    if step_i > 1:
        t_indices_ds = t_indices[::step_i]
        # 마지막 인덱스는 포함해서 끝값이 잘 안 보이는 문제 방지
        if t_indices_ds[-1] != t_indices[-1]:
            t_indices_ds = np.append(t_indices_ds, t_indices[-1])
        t_indices = t_indices_ds

    C = contrib_array[t_indices, :]  # (K, F)

    if mode == "abs":
        W = np.abs(C)
    elif mode == "raw":
        # raw는 음수 기여가 있으면 stacked-percent 해석이 애매해질 수 있음
        W = C.copy()
    else:
        raise ValueError("mode는 'abs' 또는 'raw'")

    denom = np.sum(W, axis=1, keepdims=True)  # (K,1)
    # 0 division 방지
    denom = np.where(denom == 0, 1.0, denom)
    P = 100.0 * (W / denom)  # (K,6) 퍼센트

    x = np.arange(len(t_indices))

    fig, ax = plt.subplots(figsize=figsize)
    bottom = np.zeros(len(t_indices), dtype=float)

    colors = plt.get_cmap("tab10").colors  # 최대 10개
    for j in range(f):
        ax.bar(
            x,
            P[:, j],
            bottom=bottom,
            color=colors[j % len(colors)],
            alpha=alpha,
            edgecolor="black",
            linewidth=0.4,
            label=feature_names[j],
        )
        bottom += P[:, j]

    ax.set_title(f"{kind} feature contribution (stacked % per time index)")
    ax.set_ylabel("percent (%)")
    ax.set_xlabel("t (time index)")
    ax.set_ylim(0, 100)

    # x축 라벨을 실제 t 인덱스로 표시
    # - 막대는 모두 그리되, tick label만 간격(xtick_step)으로 줄여 가독성 개선
    if xtick_step is None:
        xtick_step_i = 1
    else:
        xtick_step_i = int(xtick_step)
        if xtick_step_i < 1:
            xtick_step_i = 1

    if xtick_max_labels is not None:
        m = int(xtick_max_labels)
        if m > 0:
            # 표시 가능한 라벨 수를 넘으면 자동으로 step을 늘림
            xtick_step_i = max(xtick_step_i, int(np.ceil(len(x) / m)))

    tick_pos = x[::xtick_step_i]
    tick_lab = [str(int(t)) for t in t_indices[::xtick_step_i]]
    if len(x) > 0 and tick_pos[-1] != x[-1]:
        tick_pos = np.append(tick_pos, x[-1])
        tick_lab.append(str(int(t_indices[-1])))

    ax.set_xticks(tick_pos)
    ax.set_xticklabels(tick_lab, rotation=90)

    ax.grid(True, axis="y", alpha=0.2)
    # legend/tight_layout은 show_line(twinx) 여부에 따라 아래에서 한 번만 정리

    if show_line:
        if val_array is None:
            raise ValueError("show_line=True이면 val_array가 필요합니다.")
        val_vals = np.asarray(val_array, dtype=float)[t_indices]

        ax2 = ax.twinx()
        ax2.plot(
            x,
            val_vals,
            color="black",
            marker="o",
            linewidth=1.2,
            markersize=3,
            label=kind,
        )
        ax2.set_ylabel(f"{kind} value")
        ax2.grid(False)

        # 레전드는 1개만(우측 상단 레전드 제거): 왼쪽 상단에 통합 레전드
        h1, l1 = ax.get_legend_handles_labels()
        h2, l2 = ax2.get_legend_handles_labels()
        ax.legend(h1 + h2, l1 + l2, loc="upper left", fontsize=9, ncol=1)

        # twinx가 있으면 tight_layout이 우측 라벨을 잘 못 잡는 경우가 있어 right margin을 조금 확보
        fig.tight_layout()
        fig.subplots_adjust(right=0.95)
    else:
        ax.legend(ncol=1, fontsize=9, loc="upper left")
        fig.tight_layout()

    if save_path is not None:
        dir_ = os.path.dirname(save_path)
        if dir_:
            os.makedirs(dir_, exist_ok=True)
        fig.savefig(save_path, dpi=200)

    if show:
        plt.show()
    else:
        plt.close(fig)

    return P, t_indices


def plot_tsne_before_after_ci(
    *,
    X_list_before,
    Y_list_before,
    X_list_after,
    Y_list_after,
    alarm_masks,
    CI_array,
    thresholds,
    args,
    vehicle_idx,
):
    # ===== 리스트 -> 배열로 변환 (행 단위로 쌓였다는 가정) =====
    X_before = np.vstack(X_list_before).astype(float)  # (M1, D)
    y_before = np.asarray(Y_list_before, dtype=int)  # (M1,)

    X_after = np.vstack(X_list_after).astype(float)  # (M2, D)
    y_after = np.asarray(Y_list_after, dtype=int)  # (M2,)

    print("before:", X_before.shape, y_before.shape)
    print("after :", X_after.shape, y_after.shape)

    # ===== 합쳐서 같은 스케일러/같은 t-SNE로 임베딩 =====
    X_all = np.vstack([X_before, X_after])
    Xs_all = StandardScaler().fit_transform(X_all)

    tsne = TSNE(
        n_components=2,
        perplexity=10,
        learning_rate="auto",
        init="pca",
        random_state=0,
        max_iter=1000,
    )
    Z_all = tsne.fit_transform(Xs_all)

    n_before = X_before.shape[0]
    Z_before = Z_all[:n_before]
    Z_after = Z_all[n_before:]

    # ===== 1x3 plot (t-SNE 2개 + CI 1개) =====
    # t-SNE 패널(axes[0], axes[1])만 같은 좌표계로 비교하고,
    # CI 패널(axes[2])은 독립 축을 유지.
    fig, axes = plt.subplots(1, 3, figsize=(18, 6), sharex=False, sharey=False)
    axes[1].sharex(axes[0])
    axes[1].sharey(axes[0])

    for ax, Z, y, title in [
        (axes[0], Z_before, y_before, "Before (y_recovered)"),
        (axes[1], Z_after, y_after, "After (recon)"),
    ]:
        ax.scatter(Z[y == 0, 0], Z[y == 0, 1], s=4, alpha=0.5, label="normal")
        ax.scatter(Z[y == 1, 0], Z[y == 1, 1], s=4, alpha=0.5, label="fault")
        ax.set_title(title)
        ax.grid(True, alpha=0.2)

    CI = np.asarray(CI_array[args.x_start :]).reshape(-1)

    n = len(CI)
    # x_start가 지정되면 해당 값부터 시작하도록 x축 생성
    if args.x_start is not None:
        x = np.arange(n) + args.x_start
    else:
        x = np.arange(n)
    axes[2].plot(x, CI, color="tab:blue", lw=1.4, label="CI")
    axes[2].set_ylabel("CI")
    axes[2].set_xlabel("Time")
    axes[2].set_title("Comprehensive Index (CI)")
    axes[2].grid(True, alpha=0.3)

    if args.x_tick_step is not None:
        ticks = (
            np.floor(np.arange(x[0], x[-1] + 1, args.x_tick_step) / args.x_tick_step)
            * args.x_tick_step
        )
        ticks[0] = args.x_start if args.x_start is not None else 0
        ticks = np.append(ticks, x[-1]) if ticks[-1] != x[-1] else ticks
        axes[2].xaxis.set_major_locator(mtick.FixedLocator(ticks))

    threshold_color = ["tab:red", "tab:purple", "tab:green"]
    thr = list(thresholds)
    # 최대 3개까지만 표시
    thr = thr[:3]
    sigma_labels = [f"Threshold-{i}" for i in thr]
    for val, lab, col in zip(thr, sigma_labels, threshold_color):
        axes[2].axhline(val, color=col, linestyle="--", linewidth=1.2, label=lab)
    axes[2].legend(loc="best", fontsize=8)

    # alarm 샘플 강조(테두리만) — before/after 공통 처리
    alarm_style_base = {
        "alpha": 0.3,
        # "facecolors": "none",
    }

    alarm_style_normal_lo = {
        **alarm_style_base,
        "s": 4,
        "marker": "o",
        "edgecolors": "#7F3B08",
        "facecolors": "#7F3B08",
        "linewidths": 0.7,
        "zorder": 5,
        "label": f"normal (CI>{thresholds[0]})",
    }
    alarm_style_normal_hi = {
        **alarm_style_base,
        "s": 4,
        "marker": "o",
        "edgecolors": "#2A9D8F",
        "facecolors": "#2A9D8F",
        "linewidths": 0.7,
        "zorder": 5,
        "label": f"normal (CI>{thresholds[1]})",
    }

    alarm_style_fault_lo = {
        **alarm_style_base,
        "s": 7,
        "marker": "o",
        "edgecolors": "blue",
        "facecolors": "blue",
        "linewidths": 0.9,
        "zorder": 6,
        "label": f"fault (CI>{thresholds[0]})",
    }
    alarm_style_fault_hi = {
        **alarm_style_base,
        "s": 7,
        "marker": "o",
        "edgecolors": "red",
        "facecolors": "red",
        "linewidths": 0.9,
        "zorder": 6,
        "label": f"fault (CI>{thresholds[1]})",
    }

    for key, ax, Z, y in [
        ("before", axes[0], Z_before, y_before),
        ("after", axes[1], Z_after, y_after),
    ]:
        if not args.alarm:
            continue

        alarm_level = np.asarray(alarm_masks[key], dtype=np.int8)
        if alarm_level.shape[0] != Z.shape[0]:
            print(
                f"[WARN] alarm_mask_{key} 길이 불일치:",
                alarm_level.shape[0],
                f"vs Z_{key}:",
                Z.shape[0],
            )
            continue

        mask_normal_lo = (y == 0) & (alarm_level == 1)
        mask_normal_hi = (y == 0) & (alarm_level == 2)
        mask_fault_lo = (y == 1) & (alarm_level == 1)
        mask_fault_hi = (y == 1) & (alarm_level == 2)

        if args.alarm_nlo and np.any(mask_normal_lo):
            ax.scatter(
                Z[mask_normal_lo, 0],
                Z[mask_normal_lo, 1],
                **alarm_style_normal_lo,
            )
        if args.alarm_nhi and np.any(mask_normal_hi):
            ax.scatter(
                Z[mask_normal_hi, 0],
                Z[mask_normal_hi, 1],
                **alarm_style_normal_hi,
            )
        if args.alarm_flo and np.any(mask_fault_lo):
            ax.scatter(
                Z[mask_fault_lo, 0],
                Z[mask_fault_lo, 1],
                **alarm_style_fault_lo,
            )
        if args.alarm_fhi and np.any(mask_fault_hi):
            ax.scatter(
                Z[mask_fault_hi, 0],
                Z[mask_fault_hi, 1],
                **alarm_style_fault_hi,
            )

    axes[0].legend(loc="best")
    axes[1].legend(loc="best")
    plt.tight_layout()
    if args.save:
        save_path = (
            f"{args.models_dir}/{args.models_idx}/results_tsne/tSNE_{vehicle_idx}.png"
        )
        if save_path is not None and fig is not None:
            dir_ = os.path.dirname(save_path)
            if dir_:
                os.makedirs(dir_, exist_ok=True)
            fig.savefig(save_path, dpi=150)
    else:
        plt.show()

    return fig, axes


def get_rmse_stats_from_error(error_array):
    """
    error_array: (Sample_Size, Features) 형태의 2차원 배열 권장
    """
    # 1. 제곱 (Square)
    squared_errors = error_array**2

    # 2. 각 샘플마다의 평균 (Mean) -> 각 샘플의 MSE가 됨
    # 배열이 1차원이면 그냥 전체 RMSE 하나만 나옵니다.
    if error_array.ndim > 1:
        # 다차원일 경우, 마지막 차원(특징 차원)에 대해 평균을 냄
        sample_mse = np.mean(squared_errors, axis=1)
    else:
        sample_mse = squared_errors

    # 3. 제곱근 (Root) -> 각 샘플의 RMSE
    sample_rmse = np.sqrt(sample_mse)

    # 4. 통계 산출
    total_rmse = np.mean(sample_rmse)  # RMSE 평균
    rmse_std = np.std(sample_rmse)  # RMSE 표준편차

    return total_rmse, rmse_std


def draw_auc_heatmap(csv_path, save_path, title_suffix=""):
    """
    Draws AUC heatmap from a CSV file containing 'u_model_idx', 'x_model_idx', and 'AUC' columns.
    """
    # Read the CSV
    if not os.path.exists(csv_path):
        print(f"Error: CSV file not found at {csv_path}")
        return

    print(f"Loading AUC results from {csv_path} for heatmap...")
    auc_df = pd.read_csv(csv_path)
    print(f"Loaded {len(auc_df)} model combination results")

    required_columns = ["u_model_idx", "x_model_idx", "AUC"]
    if not all(col in auc_df.columns for col in required_columns):
        print(f"Error: CSV must contain {required_columns} columns.")
        return

    print("Creating AUC heatmap...")

    # Create pivot table for heatmap
    pivot_data = auc_df.pivot(index="x_model_idx", columns="u_model_idx", values="AUC")

    # Sort indices for better visualization
    pivot_data = pivot_data.sort_index(axis=0).sort_index(axis=1)

    u_indices = pivot_data.columns
    x_indices = pivot_data.index

    # Create heatmap
    fig, ax = plt.subplots(
        figsize=(
            max(10, len(u_indices) * 0.8),
            max(8, len(x_indices) * 0.6),
        )
    )

    # Calculate vmin/vmax based on data to enhance contrast
    data_values = pivot_data.values
    valid_values = data_values[~np.isnan(data_values)]

    if len(valid_values) > 0:
        val_min = np.min(valid_values)
        val_max = np.max(valid_values)
        # Add a tiny padding to avoid single-value issues
        if val_max == val_min:
            val_min -= 0.01
            val_max += 0.01

        # Use dynamic range instead of fixed 0.5-1.0
        vmin = val_min
        vmax = val_max
    else:
        vmin = 0.5
        vmax = 1.0

    # Use a colormap (viridis is good for AUC values)
    im = ax.imshow(
        pivot_data.values, cmap="viridis", aspect="auto", vmin=vmin, vmax=vmax
    )

    # Set ticks and labels
    ax.set_xticks(np.arange(len(pivot_data.columns)))
    ax.set_yticks(np.arange(len(pivot_data.index)))
    ax.set_xticklabels(pivot_data.columns)
    ax.set_yticklabels(pivot_data.index)

    # Rotate x labels for better readability
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

    # Add colorbar
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label("AUC Score", rotation=270, labelpad=20, fontsize=12)

    # Add text annotations with AUC values
    for i in range(len(pivot_data.index)):
        for j in range(len(pivot_data.columns)):
            val = pivot_data.values[i, j]
            if not np.isnan(val):
                text_color = "white" if val < 0.75 else "black"
                ax.text(
                    j,
                    i,
                    f"{val:.3f}",
                    ha="center",
                    va="center",
                    color=text_color,
                    fontsize=9,
                )

    # Labels and title
    ax.set_xlabel("U Model Index", fontsize=13, fontweight="bold")
    ax.set_ylabel("X Model Index", fontsize=13, fontweight="bold")
    ax.set_title(
        f"AUC Heatmap: Model Combination Performance {title_suffix}",
        fontsize=15,
        fontweight="bold",
        pad=20,
    )

    # Grid
    ax.set_xticks(np.arange(len(pivot_data.columns)) - 0.5, minor=True)
    ax.set_yticks(np.arange(len(pivot_data.index)) - 0.5, minor=True)
    ax.grid(which="minor", color="gray", linestyle="-", linewidth=0.5)
    ax.tick_params(which="minor", size=0)

    plt.tight_layout()

    # Ensure directory exists
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    plt.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"AUC heatmap saved to: {save_path}\n")
