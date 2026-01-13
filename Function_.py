import pickle
import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
import pandas as pd
import matplotlib.ticker as mtick
import os
import json
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

# Function to safely load a PyTorch model (serialized by GPU) or any pickled object to CPU
def _to_cpu(obj: Any):
    if torch.is_tensor(obj):
        print( "Its a tensor, moved to CPU")
        return obj.detach().cpu()
    if isinstance(obj, (list, tuple)):
        print( "Its a list or tuple, moved to CPU")
        return type(obj)(_to_cpu(x) for x in obj)
    if isinstance(obj, dict):
        print( "Its a dict, moved to CPU")
        return {k: _to_cpu(v) for k, v in obj.items()}
    return obj

def safe_load(path: str):
    try:
        print( "Trying torch.load")
        return _to_cpu(torch.load(path, map_location=torch.device('cpu'), weights_only=False))
    except Exception:
        print( "Trying pickle.load")
        class CPUUnpickler(pickle.Unpickler):
            def find_class(self, module, name):
                if module == 'torch.storage' and name == '_load_from_bytes':
                    return lambda b: torch.load(io.BytesIO(b), map_location=torch.device('cpu'), weights_only=False)
                return super().find_class(module, name)
        try:
            print( "Trying CPUUnpickler")
            with open(path, 'rb') as f:
                obj = CPUUnpickler(f).load()
            return _to_cpu(obj)
        except Exception:
            print( "Trying normal pickle.load")
            with open(path, 'rb') as f:
                obj = pickle.load(f)
            return _to_cpu(obj)

def calculate_volt_modepi(volt_all):
    """
    Simplified function to calculate volt_modepi and volt_di
    """
    volt_mode = volt_all.mean(axis=1)
    volt_std = volt_all.std(axis=1)
    volt_lamda = 1 / volt_std

    volt_pi1 = (1 / (2 * np.pi * volt_all.pow(3))).mul(volt_lamda, axis=0).pow(0.5)
    volt_pi2 = (((-1) * ((volt_all.sub(volt_mode, axis=0))).pow(2).mul(volt_lamda, axis=0)) / (2 * volt_all.mul(volt_mode.pow(2), axis=0))).apply(np.exp)
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
        OCV[k] = b[17] * x[1, k] ** 17 + b[16] * x[1, k] ** 16 + b[15] * x[1, k] ** 15 + b[14] * x[1, k] ** 14 + b[13] * \
                 x[1, k] ** 13 + \
                 b[12] * x[1, k] ** 12 + b[11] * x[1, k] ** 11 + b[10] * x[1, k] ** 10 + b[9] * x[1, k] ** 9 + b[8] * x[
                     1, k] ** 8 + \
                 b[7] * x[1, k] ** 7 + b[6] * x[1, k] ** 6 + b[5] * x[1, k] ** 5 + b[4] * x[1, k] ** 4 + b[3] * x[
                     1, k] ** 3 + \
                 b[2] * x[1, k] ** 2 + b[1] * x[1, k] + b[0] + b[18] * temp.iloc[k] + b[19] * temp.iloc[k] ** 2 + b[
                     20] * temp.iloc[k] ** 3
        C[k, :] = [-1, b[17] * x[1, k] ** 16 * 17 + b[16] * x[1, k] ** 15 * 16 + b[15] * x[1, k] ** 14 * 15 + b[14] * x[
            1, k] ** 13 * 14 +
                   b[13] * x[1, k] ** 12 * 13 + b[12] * x[1, k] ** 11 * 12 + b[11] * x[1, k] ** 10 * 11 + b[10] * x[
                       1, k] ** 9 * 10 +
                   b[9] * x[1, k] ** 8 * 9 + b[8] * x[1, k] ** 7 * 8 + b[7] * 7 * x[1, k] ** 6 + b[6] * 6 * x[
                       1, k] ** 5 +
                   b[5] * 5 * x[1, k] ** 4 + b[4] * 4 * x[1, k] ** 3 + 3 * b[3] * x[1, k] ** 2 + 2 * b[2] * x[1, k] + b[
                       1]]
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
            OCVi[k, j] = b[17] * (xi[k, j] + x[1, k]) ** 17 + b[16] * (xi[k, j] + x[1, k]) ** 16 + b[15] * (
                        xi[k, j] + x[1, k]) ** 15 + b[14] * (xi[k, j] + x[1, k]) ** 14 + b[13] * (
                                     xi[k, j] + x[1, k]) ** 13 + b[12] * (xi[k, j] + x[1, k]) ** 12 + b[11] * (
                                     xi[k, j] + x[1, k]) ** 11 + b[10] * (xi[k, j] + x[1, k]) ** 10 + b[9] * (
                                     xi[k, j] + x[1, k]) ** 9 + b[8] * (xi[k, j] + x[1, k]) ** 8 + b[7] * (
                                     xi[k, j] + x[1, k]) ** 7 + b[6] * (xi[k, j] + x[1, k]) ** 6 + b[5] * (
                                     xi[k, j] + x[1, k]) ** 5 + b[4] * (xi[k, j] + x[1, k]) ** 4 + b[3] * (
                                     xi[k, j] + x[1, k]) ** 3 + b[2] * (xi[k, j] + x[1, k]) ** 2 + b[1] * (
                                     xi[k, j] + x[1, k]) + b[0] + b[18] * temp[k] + b[19] * temp[k] ** 2 + b[20] * temp[
                             k] ** 3
            if OCVi[k, j] > OCV[k] + 0.1:
                OCVi[k, j] = OCV[k] + 0.1
            Ci = b[17] * (xi[k, j] + x[1, k]) ** 16 * 17 + b[16] * (xi[k, j] + x[1, k]) ** 15 * 16 + b[15] * (
                        xi[k, j] + x[1, k]) ** 14 * 15 + b[14] * (xi[k, j] + x[1, k]) ** 13 * 14 + b[13] * (
                             xi[k, j] + x[1, k]) ** 12 * 13 + b[12] * (xi[k, j] + x[1, k]) ** 11 * 12 + b[11] * (
                             xi[k, j] + x[1, k]) ** 10 * 11 + b[10] * (xi[k, j] + x[1, k]) ** 9 * 10 + b[9] * (
                             xi[k, j] + x[1, k]) ** 8 * 9 + b[8] * (xi[k, j] + x[1, k]) ** 7 * 8 + b[7] * 7 * (
                             xi[k, j] + x[1, k]) ** 6 + b[6] * 6 * (xi[k, j] + x[1, k]) ** 5 + b[5] * 5 * (
                             xi[k, j] + x[1, k]) ** 4 + b[4] * 4 * (xi[k, j] + x[1, k]) ** 3 + 3 * b[3] * (
                             xi[k, j] + x[1, k]) ** 2 + 2 * b[2] * (xi[k, j] + x[1, k]) + b[1]
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
    xipre[1:, :] = xi[:xi.shape[0] - 1, :]
    return xipre, xpre, Spre, Upre, Xipre, x, DUi, Xi


def custom_activation(x):
    return 2.5 + 1.8 * torch.sigmoid(x)


def PCA(data, l1, l2):
    # Data standardization
    data_mean = np.mean(data, 0)
    data_std = np.std(data, 0)
    data_nor = (data - data_mean) / data_std
    # Calculate covariance matrix for standardized data
    X = np.cov(data_nor.T)
    # Calculate singular values for covariance matrix
    P, v, P_t = np.linalg.svd(X)  # This function returns three values u s v
    v_ratio = np.cumsum(v) / np.sum(v) # Cumulative contribution rate of eigenvalues -> 이 중에서 상위 95%의 기여도를 차지하는 벡터들 선택.
    # Find the index of eigenvalues with a cumulative ratio greater than 0.95
    k = np.where(v_ratio > 0.95)[0]
    # New principal components
    p_k = P[:, :k[0]]
    v_I = np.diag(1 / v[:k[0]])
    # T2 statistic threshold calculation
    coe = k[0] * (np.shape(data)[0] - 1) * (np.shape(data)[0] + 1) / \
        ((np.shape(data)[0] - k[0]) * np.shape(data)[0])
    T_95_limit = coe * stats.f.ppf(0.95, k[0], (np.shape(data)[0] - k[0]))
    T_99_limit = coe * stats.f.ppf(l1, k[0], (np.shape(data)[0] - k[0]))
    # SPE statistic threshold calculation
    O1 = np.sum((v[k[0]:]) ** 1)
    O2 = np.sum((v[k[0]:]) ** 2)
    O3 = np.sum((v[k[0]:]) ** 3)
    h0 = 1 - (2 * O1 * O3) / (3 * (O2 ** 2))
    c_95 = norm.ppf(0.95)
    c_99 = norm.ppf(l2)
    SPE_95_limit = O1 * ((h0 * c_95 * ((2 * O2) ** 0.5) /
                         O1 + 1 + O2 * h0 * (h0 - 1) / (O1 ** 2)) ** (1 / h0))
    SPE_99_limit = O1 * ((h0 * c_99 * ((2 * O2) ** 0.5) /
                         O1 + 1 + O2 * h0 * (h0 - 1) / (O1 ** 2)) ** (1 / h0))
    return v_I, v, v_ratio, p_k, data_mean, data_std, T_95_limit, T_99_limit, SPE_95_limit, SPE_99_limit, P, k, P_t, X, data_nor

def T2(data_in, data_mean, data_std, p_k, v_I):
    data_nor = np.array((data_in - data_mean) / data_std)
    D = p_k @ v_I @ p_k.T
    t2 = np.dot(np.dot((data_nor).T, D), (data_nor))
    return t2  # T2 statistic

def T2_array(data_in, data_mean, data_std, p_k, v_I):
    D = p_k @ v_I @ p_k.T # (F×F)
    X = np.asarray((data_in - data_mean) / data_std, dtype=float) # (N×F)
    return np.sum((X @ D) * X, axis=1)

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
    C = p_k @ p_k.T                     # (F, F)
    M = np.eye(F_dim) - C               # residual projection
    R = X @ M                           # (N, F)
    return np.einsum('ij,ij->i', R, R)  # row-wise squared norm

def SPE(data_in, data_mean, data_std, p_k):
    # test_data_nor = ((data_in - data_mean) / data_std).reshape(len(data_in), 1)
    test_data_nor = np.array(((data_in - data_mean) / data_std)).reshape(6,1)
    I = np.eye(len(data_in))
    Q_count = np.dot(np.dot((I - np.dot(p_k, p_k.T)), test_data_nor).T,
                     np.dot((I - np.dot(p_k, p_k.T)), test_data_nor))
    return Q_count # Squared prediction error

def chi_square_dist_components(p_k, v_I, X, SPE_limit, T_limit):
    Pi = (p_k @ v_I @ p_k.T / SPE_limit) + ((np.eye(p_k.shape[0]) - p_k @ p_k.T) / T_limit)
    M = np.dot(X, Pi)
    tr_M = np.trace(M)
    tr_M2 = np.trace(np.dot(M, M))
    g = tr_M2 / tr_M
    h = (tr_M ** 2) / tr_M2
    return Pi, g, h

def diagnosis_thresholds(g, h, sigma_levels=[2, 3, 4.5, 6], show_plot=True):
    mean = h
    sigma = np.sqrt(2 * h)
    
    x = np.linspace(0, 30, 10000)
    pdf = chi2.pdf(x, h)
    
    thresholds = list()
    for k in sigma_levels:
        thresholds.append(g * (mean + k * sigma))
    
    # -----------------------------
    # Numerical output
    # -----------------------------
    print(f'Chi-square distribution (df = {h})')
    print(f'Mean = {mean:.2f}, SD = {sigma:.2f}\n')

    print(f'{"k(SD)":>6} {"Threshold":>12} {"Area":>12}')
    print('-' * 50)

    for i in range(len(sigma_levels)):
        area = chi2.cdf(thresholds[i], h)
        print(f'{sigma_levels[i]:6.1f} {thresholds[i]:12.4f} {area:12.6f}')
    
    if not show_plot:
        return thresholds
    
    # -----------------------------
    # Plot PDF
    # -----------------------------    
    plt.figure()
    plt.plot(g * x, pdf, linewidth=2)
    plt.grid(True)

    # Plot mean
    plt.axvline(mean, color='k', linewidth=2, label='Mean')

    # Plot SD ranges
    for i in range(len(sigma_levels)):
        plt.axvline(thresholds[i], linestyle='--', linewidth=1.2)
        plt.text(thresholds[i], max(pdf) * 0.9, f'{sigma_levels[i]} SD',
                rotation=90, verticalalignment='bottom')

    # Labels and title
    plt.xlabel('x')
    plt.ylabel('Probability Density')
    plt.title(f'Chi-square Distribution (df = {h}) with SD Ranges')
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
    (v_I, v, v_ratio, p_k, data_mean, data_std,
     T_95_limit, T_99_limit, SPE_95_limit, SPE_99_limit,
     P, k, P_t, X, data_nor) = pca_outputs

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
        X=np.asarray(X)
    )

    # Save data_nor: keep original type
    data_nor_type = "DataFrame" if isinstance(data_nor, pd.DataFrame) else "ndarray"
    if data_nor_type == "DataFrame":
        # No extra engines required; use CSV
        data_nor.to_csv(df_path, index=True)
        df_shape = list(data_nor.shape)
    else:
        # Store ndarray into separate NPZ for clarity
        np.savez_compressed(os.path.join(output_dir, "pca_data_nor.npz"), data_nor=np.asarray(data_nor))
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
            "data_nor": data_nor_type
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
            "data_nor": df_shape
        },
        "paths": {
            "arrays": arrays_path,
            "data_nor": df_path
        }
    }
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)

def load_pca_results(output_dir):
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

    arrays = np.load(manifest["paths"]["arrays"], allow_pickle=False)

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

    # Load data_nor according to stored type
    dn_type = manifest["types"]["data_nor"]
    dn_path = manifest["paths"]["data_nor"]
    if dn_type == "DataFrame":
        # Read back CSV with index preserved
        data_nor = pd.read_csv(dn_path, index_col=0)
    elif dn_type == "ndarray":
        dn = np.load(dn_path, allow_pickle=False)
        data_nor = dn["data_nor"]
    else:
        raise ValueError("Unknown data_nor type in manifest: " + str(dn_type))

    # Validate shapes
    expected = manifest["shapes"]
    def _chk(name, obj):
        shp = list(np.asarray(obj).shape) if name != "data_nor" else list(obj.shape)
        if shp != expected[name]:
            raise ValueError(f"Shape mismatch for {name}: loaded {shp}, expected {expected[name]}")

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
    _chk("data_nor", data_nor)

    return (
        v_I, v, v_ratio, p_k, data_mean, data_std,
        T_95_limit, T_99_limit, SPE_95_limit, SPE_99_limit,
        P, k, P_t, X, data_nor
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


def plot_array(arr, title=None, x_label=None, y_label=None,
            figsize=(12, 4), save_path=None, show=True):
    
    # x축 생성: 제공된 x가 없으면 DataFrame index 또는 0..N-1 사용
    x = np.arange(len(arr))

    # 플롯
    plt.figure(figsize=figsize)
    plt.plot(x, arr, color='tab:blue', lw=1.5)
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

def cont(X_col, data_mean, data_std, X_test, P, num_pc, lamda, T2UCL1):
    X_test = ((X_test - data_mean) / data_std)
    S = np.dot(X_test, P[:, :num_pc])
    r = []
    ee = (T2UCL1 / num_pc)
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
        for m in range (n):
            mean.append(np.mean(s[:n]))
        # mean = s[:n].tolist()
        for i in range(n, len(s)):
            select_s = s[i - n: i ]
            mean_s = np.mean(select_s)
            mean.append(mean_s)
    else:
        mean = s.tolist()
    return mean

def SlidingAverage(s, n):
    mean = []
    if len(s) > n:
        for m in range (n):
            mean.append(np.mean(s[:n]))
        for i in range(n, len(s)):
            select_s = s[i - n: i ]
            mean_s = np.mean(select_s)
            mean.append(mean_s)
    else:
        mean = s.tolist()
    return mean

def DiagnosisFeature(ERRORU,ERRORX):
    ERRORUm = pd.DataFrame(ERRORU).max(axis=1)
    ERRORXm = pd.DataFrame(ERRORX).max(axis=1)
    meanu = ERRORU.mean(axis=1)
    meanx = ERRORX.mean(axis=1)
    Z_U = (pd.DataFrame(ERRORU).sub(meanu, axis=0).div(ERRORU.std(axis=1), axis=0)).max(axis=1)
    Z_X = (pd.DataFrame(ERRORX).sub(meanx, axis=0).div(ERRORX.std(axis=1), axis=0)).max(axis=1)
    second_largest_ERRORU = pd.Series(np.apply_along_axis(lambda row: np.partition(row, -2)[-2], axis=1, arr=ERRORU))
    max_diff_ERRORU = (ERRORUm - second_largest_ERRORU).div(ERRORU.std(axis=1), axis=0)
    second_largest_ERRORX = pd.Series(np.apply_along_axis(lambda row: np.partition(row, -2)[-2], axis=1, arr=ERRORX))
    max_diff_ERRORX = (ERRORXm - second_largest_ERRORX).div(ERRORX.std(axis=1), axis=0)
    alpha = 0.2
    Z_U_smoothed = ERRORUm.ewm(alpha=alpha).mean()
    Z_X_smoothed = ERRORXm.ewm(alpha=alpha).mean()
    max_diff_ERRORX = pd.Series(SlidingAverage_list(max_diff_ERRORX , 100))
    Z_X = pd.Series(SlidingAverage_list(Z_X , 100))
    Z_X_smoothed = pd.Series(SlidingAverage_list(Z_X_smoothed , 100))
    df_data = pd.concat([max_diff_ERRORU,max_diff_ERRORX,Z_U,Z_X,Z_U_smoothed,Z_X_smoothed], axis=1) # zu_2, zx_2, zu_1, zx_1, emu, emx
    return df_data

def ClassifyFeature(temp_max,temp_avg,CONTN,insulation_resistance,threshold1,fai):
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
        max_count_U = np.sum(np.argmax(ERRORU[ttime - 3000:ttime, :], axis=1) == max_erroru_column)
        f2 = max_count_U / 3000
        f3 = temp_dif[f_time - 50:f_time].max()
        f4 = insulation_resistance.iloc[ttime-1000:ttime].min()
        f5 = volt_all.iloc[ttime-100:ttime,:].min().min()
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
		names += [('var%d(t-%d)' % (j+1, i)) for j in range(n_vars)]
	# forecast sequence (t, t+1, ... t+n)
	for i in range(0, n_out):
		cols.append(df.shift(-i))
		if i == 0:
			names += [('var%d(t)' % (j+1)) for j in range(n_vars)]
		else:
			names += [('var%d(t+%d)' % (j+1, i)) for j in range(n_vars)]
	# put it all together
	agg = concat(cols, axis=1)
	agg.columns = names
	# drop rows with NaN values
	if dropnan:
		agg.dropna(inplace=True)
	return agg

def prepare_training_data(test_X, INPUT_SIZE, TIME_STEP, device):
    test_X_df = pd.DataFrame(test_X.cpu().detach().numpy()[:,0,:])
    reframed = series_to_supervised(test_X_df, 1, 1)
    reframed.drop(reframed.columns[INPUT_SIZE:INPUT_SIZE * 2 - 2], axis=1, inplace=True)
    train = reframed.values
    train_X, train_y = train[:, :-2], train[:, -2:] # Last two columns are the targets (volt_modepi and soc)
    train_y = train_y.reshape(-1, 2)
    batch_train = int(reframed.shape[0] / TIME_STEP)

    train_X = torch.tensor(train_X)
    train_X = train_X.reshape(batch_train, TIME_STEP, INPUT_SIZE).to(device)
    train_y = torch.tensor(train_y)
    train_y = train_y.reshape(batch_train, TIME_STEP, 2).to(device)
    
    return train_X, train_y


def plot_testX_timeseries(input, feature_names=None, title=None, figsize=(12, 6), save_path=None, show=True, seperate=False, _range: list = [-1], start_idx=0):
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
        raise ValueError("Expected _range to be a 1D array with a single element -1 or a 2D array.")
    
    
    """ sperate가 True이면 for 문을 돌고 false 이면 한번에 그린다."""
    plot_all = True if _range == [-1] else False
    if seperate:
        _range = [0, input.shape[-1] - 1] if plot_all else _range
        for i in range(_range[0], _range[1] + 1):
            print(f"Plotting feature index: {i}")
            print(f"Input shape: {input.shape}")
            if plot_all: # 모든 특성 그리기
                if input.shape[-1] > 10:
                    raise Exception("특성 수가 너무 많아 개별 플롯으로 그릴 수 없습니다. 'seperate'를 False로 설정하세요.")

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
            plot_timeseries(test_X, feature_names, title_input, figsize, save_path_input, show)
    else:
        if plot_all: # 모든 특성 그리기
            print("모든 특성을 한 번에 플롯으로 그립니다.")
            test_X = input[start_idx:]
            title += f"-all({test_X.shape[-1]})"
            if save_path is not None:
                save_path += f"-all.png"
        else: # 특정 범위의 특성 그리기
            print(f"{_range[0]}부터 {_range[1]}까지의 특성을 한 번에 플롯으로 그립니다.")
            if input.ndim == 2:
                test_X = input[start_idx:, _range[0]:_range[1]+1]
            elif input.ndim == 3:
                test_X = input[start_idx:, :, _range[0]:_range[1]+1]
            else:
                raise ValueError(f"Expected 2D or 3D array, got shape {input.shape}")
            title += f"-range({_range[0]}-{_range[1]})"
            if save_path is not None:
                save_path += f"-range({_range[0]}-{_range[1]}).png"
                
        plot_timeseries(test_X, feature_names, title, figsize, save_path, show)
            
def plot_timeseries(test_X, feature_names=None, title=None, figsize=(12, 6), save_path=None, show=True):
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
            raise ValueError(f"Expected 2D array [T, F] or 3D [T,1,F], got shape {arr.shape}")

        T, F = arr.shape

        # 레전드 라벨 준비
        if not feature_names or len(feature_names) != F:
            feature_names = [f"var{i+1}" for i in range(F)]

        fig, ax = plt.subplots(figsize=figsize)
        x = np.arange(T)
        for i in range(F):
            ax.plot(x, arr[:, i], label=feature_names[i])

        ax.set_xlabel("Time")
        ax.set_ylabel("Value")
        ax.set_title(title if title is not None else f"Time Series ({F} features)")
        ax.grid(True, alpha=0.3)
        ax.legend(loc="best", ncol=2, fontsize=8)
        plt.tight_layout()

        if save_path is not None:
            dir_ = os.path.dirname(save_path)
            if dir_:
                os.makedirs(dir_, exist_ok=True)
            fig.savefig(save_path, dpi=150)

        if show:
            plt.show()
        else:
            plt.close(fig)

def plot_diagnostics_triplet(t2_array,
                             spe_array,
                             CI_array,
                             thresholds:list=None,
                             sigma_levels=None,
                             title=None,
                             x_label="Time",
                             y_labels=("T²", "SPE", "CI"),
                             figsize=(12, 10),
                             save_path=None,
                             show=True,
                             x_start=None,
                             x_tick_step=1000):
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

    fig, axes = plt.subplots(nrows=3, ncols=1, figsize=figsize, sharex=True)

    # Top: T²
    axes[0].plot(x, t2, color='tab:blue', lw=1.4)
    axes[0].set_ylabel(y_labels[0])
    axes[0].set_title("Hostelling's T² Statistic")
    axes[0].grid(True, alpha=0.3)
    # x축 범위를 x_start부터 최대값까지 고정
    axes[0].set_xlim(x[0], x[-1])

    # Middle: SPE
    axes[1].plot(x, spe, color='tab:blue', lw=1.4)
    axes[1].set_ylabel(y_labels[1])
    axes[1].set_title("Squared Prediction Error (SPE)")
    axes[1].grid(True, alpha=0.3)
    axes[1].set_xlim(x[0], x[-1])

    # Bottom: CI (CI) + thresholds
    axes[2].plot(x, CI, color='tab:blue', lw=1.4, label='CI')
    axes[2].set_ylabel(y_labels[2])
    axes[2].set_xlabel(x_label)
    axes[2].set_title("Comprehensive Index (CI)")
    axes[2].grid(True, alpha=0.3)
    axes[2].set_xlim(x[0], x[-1])

    # 시작 눈금을 포함시키기 위해 눈금 간격을 고정할 수 있는 옵션
    if x_tick_step is not None:
        ticks = np.floor(np.arange(x[0], x[-1] + 1, x_tick_step)/x_tick_step)*x_tick_step
        ticks[0] = x_start if x_start is not None else 0
        ticks = np.append(ticks, x[-1]) if ticks[-1] != x[-1] else ticks
        for ax in axes:
            ax.xaxis.set_major_locator(mtick.FixedLocator(ticks))

    threshold_color = ["tab:red", "tab:purple", "tab:green"]
    if thresholds is not None:
        thr = list(thresholds)
        # 최대 3개까지만 표시(요청사항 대응)
        thr = thr[:3]
        sigma_labels = [f"Threshold {i+1} ({sigma_levels[i]:.1f}\u03C3)" for i in range(len(thr))]
        for val, lab, col in zip(thr, sigma_labels, threshold_color):
            axes[2].axhline(val, color=col, linestyle='--', linewidth=1.2, label=lab)
        axes[2].legend(loc='best', fontsize=8)

    if title:
        fig.suptitle(title, y=0.98)
    fig.tight_layout()

    if save_path is not None:
        dir_ = os.path.dirname(save_path)
        if dir_:
            os.makedirs(dir_, exist_ok=True)
        fig.savefig(save_path, dpi=150)

    if show:
        plt.show()
    else:
        plt.close(fig)

def make_model_path_based_timestamp(base="models", make:bool=False, prefix:str=None, tz="Asia/Seoul"):
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