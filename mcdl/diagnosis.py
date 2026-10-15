"""진단 특징과 관리도.

차량 한 대의 재구성 오차에서 진단 특징을 뽑고, 정상 학습 진단 특징으로 맞춘 관리도(PCA 모델과
T²·SPE 관리한계)로 시점별 T²·SPE·CI와 최대 CI를 계산한다. 용어는 GLOSSARY.md를 따른다.
"""

from __future__ import annotations

import json
import os

import numpy as np
import pandas as pd
from scipy import stats
from scipy.stats import norm

_EPS = 1e-12
_EWMA_ALPHA = 0.2
_X_WINDOW = 100  # X 신호 특징에만 적용하는 직전 이동평균 길이
_LIMIT_99 = 0.99  # npz에 함께 저장하는 99% 관리한계의 분위수 (평가에는 95%만 쓴다)


def signal_features(signal: str, error):
    """한 신호("U" 또는 "X")의 재구성 오차 (시점, 셀)에서 시점별 특징 3개를 뽑는다.

    (1위−2위 오차)/셀 간 표준편차, 셀 간 최대 z-score, 최대 오차의 EWMA(α=0.2).
    X 신호는 세 특징 모두 직전 100-step 이동평균을 적용한다.
    반환값(Series 3개)은 Test_multi_models의 temp_cache 파일 형식과 같다.
    """
    arr = np.asarray(error, dtype=float)
    if arr.ndim != 2:
        raise ValueError(f"signal_features expects a 2D array, got {arr.ndim}D")

    arr_max = arr.max(axis=1)
    arr_mean = arr.mean(axis=1)
    arr_std = np.maximum(arr.std(axis=1), _EPS)

    z = ((arr - arr_mean[:, None]) / arr_std[:, None]).max(axis=1)
    second = np.partition(arr, -2, axis=1)[:, -2] if arr.shape[1] >= 2 else arr[:, 0]
    max_diff = (arr_max - second) / arr_std
    ewma = pd.Series(arr_max).ewm(alpha=_EWMA_ALPHA).mean()

    if signal == "U":
        return pd.Series(max_diff), pd.Series(z), ewma
    return (
        pd.Series(_sliding_average(max_diff, _X_WINDOW)),
        pd.Series(_sliding_average(z, _X_WINDOW)),
        pd.Series(_sliding_average(ewma.to_numpy(), _X_WINDOW)),
    )


def combine_signal_features(u, x) -> pd.DataFrame:
    """신호별 특징을 진단 특징 열 순서 [zu₂, zx₂, zu₁, zx₁, ewu, ewx]로 묶는다."""
    return pd.concat([u[0], x[0], u[1], x[1], u[2], x[2]], axis=1)


def diagnosis_features(error_u, error_x) -> pd.DataFrame:
    """차량 한 대의 진단 특징 (시점, 6). 차량마다 따로 계산해야 한다 (EWMA·이동평균이 차량 경계를 넘지 않게)."""
    return combine_signal_features(
        signal_features("U", error_u), signal_features("X", error_x)
    )


class ControlChart:
    """관리도: 정상 학습 진단 특징으로 맞춘 PCA 모델과 T²·SPE 관리한계(95%)."""

    def __init__(self, pca_outputs):
        # 학습 특징 행렬(data_nor)은 메모리를 아끼기 위해 들고 있지 않는다. save()가 다시 계산한다.
        (
            self._v_I,
            self._v,
            self._v_ratio,
            self._p_k,
            self._mean,  # fit이면 pandas Series, load면 ndarray. 출처 그대로 둔다.
            self._std,
            self.t2_limit,
            self._t2_limit_99,
            self.spe_limit,
            self._spe_limit_99,
            self._P,
            self._k,
            self._P_t,
            self._cov,
            _data_nor,
        ) = pca_outputs

    @classmethod
    def fit(cls, features: pd.DataFrame) -> "ControlChart":
        """정상 학습 진단 특징(차량별로 계산해 이어 붙인 DataFrame)으로 관리도를 맞춘다."""
        return cls(_pca(features, _LIMIT_99, _LIMIT_99))

    @classmethod
    def load(cls, run_dir) -> "ControlChart":
        """Run 폴더에 저장된 관리도(pca_arrays.npz)를 읽는다. pca_manifest.json이 있어야 한다."""
        return cls(_load_pca(run_dir))

    def save(self, run_dir, features: pd.DataFrame) -> None:
        """관리도를 run_dir에 저장한다 (pca_arrays.npz, pca_manifest.json, pca_data_nor.csv).

        features는 fit에 쓴 학습 진단 특징이다. pca_data_nor.csv는 fit과 같은 식으로 다시 계산한다.
        """
        data_nor = (features - self._mean) / self._std
        _save_pca(
            run_dir,
            (
                self._v_I,
                self._v,
                self._v_ratio,
                self._p_k,
                self._mean,
                self._std,
                self.t2_limit,
                self._t2_limit_99,
                self.spe_limit,
                self._spe_limit_99,
                self._P,
                self._k,
                self._P_t,
                self._cov,
                data_nor,
            ),
        )

    def score(self, features: pd.DataFrame):
        """시점별 (T², SPE, CI). CI = SPE/SPE₉₅ + T²/T²₉₅."""
        t2, _ = _t2(features, self._mean, self._std, self._p_k, self._v_I)
        spe, _ = _spe(features, self._mean, self._std, self._p_k)
        return t2, spe, (spe / self.spe_limit) + (t2 / self.t2_limit)

    def max_ci(self, features: pd.DataFrame) -> float:
        """최대 CI: 차량 한 대의 CI 시계열 최댓값 (NaN 제외)."""
        return float(np.nanmax(np.asarray(self.score(features)[2], dtype=float)))


def _sliding_average(s, n: int):
    """각 시점까지의 직전 n개 평균. 처음 n개는 첫 n개 평균, 길이가 n 이하면 그대로 둔다."""
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


def _pca(data, l1, l2):
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


def _t2(data_in, data_mean, data_std, p_k, v_I):
    D = p_k @ v_I @ p_k.T  # (F×F)
    X = np.asarray((data_in - data_mean) / data_std, dtype=float)  # (N×F)
    contrib_array = (X @ D) * X
    return np.sum(contrib_array, axis=1), contrib_array


def _spe(data_in, data_mean, data_std, p_k):
    """SPE_i = ||(I - p_k p_k^T) x_i||^2 (x_i는 표준화한 특징). (SPE, 특징별 잔차²)를 돌려준다."""
    X = np.asarray((data_in - data_mean) / data_std, dtype=float)
    if X.ndim == 1:
        X = X[None, :]  # (1, F)
    F_dim = p_k.shape[0]
    C = p_k @ p_k.T  # (F, F)
    M = np.eye(F_dim) - C  # residual projection
    R = X @ M  # (N, F)
    return np.einsum("ij,ij->i", R, R), R**2  # row-wise squared norm


def _save_pca(output_dir, pca_outputs):
    """관리도 15개 값을 pca_arrays.npz, pca_manifest.json, pca_data_nor.csv로 저장한다 (기존 형식 그대로)."""
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


def _load_pca(output_dir):
    """pca_arrays.npz에서 관리도 15개 값을 읽는다 (data_nor는 None). pca_manifest.json이 없으면 실패한다."""
    manifest_path = os.path.join(output_dir, "pca_manifest.json")
    if not os.path.exists(manifest_path):
        raise FileNotFoundError("pca_manifest.json not found in output_dir")

    arrays = np.load(os.path.join(output_dir, "pca_arrays.npz"), allow_pickle=False)
    return (
        arrays["v_I"],
        arrays["v"],
        arrays["v_ratio"],
        arrays["p_k"],
        arrays["data_mean"],
        arrays["data_std"],
        # Preserve numpy scalar type
        arrays["T_95_limit"][()],
        arrays["T_99_limit"][()],
        arrays["SPE_95_limit"][()],
        arrays["SPE_99_limit"][()],
        arrays["P"],
        arrays["k"],
        arrays["P_t"],
        arrays["X"],
        None,
    )
