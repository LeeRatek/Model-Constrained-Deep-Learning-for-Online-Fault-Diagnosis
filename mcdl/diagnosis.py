"""재구성 오차를 진단 특징으로 요약하고, PCA 관리도로 이상 정도(T²·SPE·CI)를 계산하는 코드.

전체 흐름
    1. 진단 특징: 차량 한 대의 재구성 오차(시점 × 셀)를 시점마다 숫자 6개로 요약한다(diagnosis_features).
       U-model 오차에서 3개, X-model 오차에서 3개다.
    2. 관리도: 정상 학습 차량들의 진단 특징으로 PCA(주성분 분석)를 해서 "정상일 때의 모양"을 잡고,
       정상 데이터의 95%가 들어오는 경계값(관리한계)을 정한다(ControlChart.fit).
    3. 점수: 평가할 차량의 진단 특징을 관리도에 넣어 시점마다
         T²  = PCA 축 안에서 평소 범위를 얼마나 벗어났는지
         SPE = PCA 축으로 설명되지 않고 남는 오차의 크기
         CI  = T²/T² 관리한계 + SPE/SPE 관리한계 (두 지표를 합친 종합 점수)
       를 계산한다(ControlChart.score). 차량의 판정 점수는 CI의 최댓값(최대 CI)이다.

누가 쓰나
    mcdl/reconstruction.py(진단 특징 계산), Train_.py(관리도 만들고 저장), Test_.py·Test_multi_models.py·
    Test_tsne.py·mcdl/chart_source.py(관리도 불러오기와 점수 계산).
    계산식은 원래 논문 코드(예전 Function_.py)와 같다.
"""

from __future__ import annotations

import json
import os

import numpy as np
import pandas as pd
from scipy import stats
from scipy.stats import norm

_EPS = 1e-12  # 표준편차가 0일 때 0으로 나누지 않도록 쓰는 아주 작은 값
_EWMA_ALPHA = 0.2  # 지수가중 이동평균(EWMA)의 가중치: 클수록 최근 값을 더 크게 반영한다
_X_WINDOW = 100  # X-model(SOC) 특징에만 적용하는 이동평균 길이(시점 수)
_LIMIT_99 = 0.99  # 파일에 함께 저장하는 99% 관리한계의 기준 (판정에는 95% 한계만 쓴다)


def signal_features(signal: str, error):
    """모델 하나(signal="U" 또는 "X")의 재구성 오차(시점 × 셀)에서 시점마다 특징 3개를 뽑는다.

    시점마다 셀들의 오차를 보고 다음 세 값을 만든다.
      1. (가장 큰 오차 − 두 번째로 큰 오차) / 셀 간 표준편차: 한 셀만 튀는 정도
      2. 셀 간 최대 z-score((오차 − 평균) / 표준편차의 최댓값): 가장 튀는 셀이 평균에서 떨어진 정도
      3. 가장 큰 오차의 지수가중 이동평균(EWMA, α=0.2): 큰 오차가 계속되는 정도
    X-model은 SOC 오차가 들쭉날쭉해서 세 값 모두 직전 100 시점 이동평균을 한 번 더 한다.
    돌려주는 값: pandas Series 3개의 튜플.
    쓰는 곳: 아래 diagnosis_features, mcdl/reconstruction.py의 training_signal_features.
    """
    arr = np.asarray(error, dtype=float)
    if arr.ndim != 2:
        raise ValueError(f"signal_features expects a 2D array, got {arr.ndim}D")

    arr_max = arr.max(axis=1)
    arr_mean = arr.mean(axis=1)
    arr_std = np.maximum(arr.std(axis=1), _EPS)

    z = ((arr - arr_mean[:, None]) / arr_std[:, None]).max(axis=1)
    second = np.partition(arr, -2, axis=1)[:, -2]  # 시점마다 두 번째로 큰 오차
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
    """U-model 특징 3개와 X-model 특징 3개를 진단 특징 6열 DataFrame으로 합친다.

    열 순서는 [zu₂, zx₂, zu₁, zx₁, ewu, ewx]이다(u는 전압, x는 SOC; 2·1·ew는 signal_features의 1·2·3번 특징).
    이 순서가 PCA 관리도 파일에 저장된 순서이므로 바꾸면 안 된다.
    쓰는 곳: 아래 diagnosis_features, Test_multi_models.py·mcdl/chart_source.py(캐시해 둔 특징을 합칠 때).
    """
    return pd.concat([u[0], x[0], u[1], x[1], u[2], x[2]], axis=1)


def diagnosis_features(error_u, error_x) -> pd.DataFrame:
    """차량 한 대의 진단 특징(시점 × 6 DataFrame)을 만든다. error_u, error_x는 U·X 재구성 오차(시점 × 셀).

    여러 차량의 오차를 이어 붙인 뒤 한 번에 계산하면 이동평균이 차량 경계를 넘어 섞이므로,
    반드시 차량마다 따로 계산한 뒤 결과를 이어 붙인다.
    쓰는 곳: mcdl/reconstruction.py의 vehicle_features, 테스트.
    """
    return combine_signal_features(
        signal_features("U", error_u), signal_features("X", error_x)
    )


class ControlChart:
    """관리도: 정상 데이터로 맞춘 PCA 모델과 T²·SPE 관리한계(95%)를 묶은 객체.

    만드는 방법은 두 가지다.
      - ControlChart.fit(학습 진단 특징): 정상 학습 차량들의 진단 특징으로 새로 계산한다 (Train_.py 등).
      - ControlChart.load(Run 폴더): Train_.py가 저장해 둔 pca_arrays.npz를 읽는다 (평가 스크립트).
    만든 뒤에는 score(시점별 T²·SPE·CI)와 max_ci(차량 판정 점수)로 쓴다.
    """

    def __init__(self, pca_outputs):
        """_pca 또는 _load_pca가 돌려준 값 15개를 속성으로 나눠 담는다. 직접 부르지 말고 fit/load를 쓴다."""
        # 표준화한 학습 특징 행렬(data_nor)은 메모리를 아끼려고 들고 있지 않는다. save()가 다시 계산한다.
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
        """정상 학습 진단 특징(차량별로 계산해 이어 붙인 시점 × 6 DataFrame)으로 관리도를 새로 만든다.

        쓰는 곳: Train_.py(학습 끝), Test_multi_models.py(체크포인트 조합마다), mcdl/chart_source.py.
        """
        return cls(_pca(features, _LIMIT_99, _LIMIT_99))

    @classmethod
    def load(cls, run_dir) -> "ControlChart":
        """Run 폴더에 저장된 관리도(pca_arrays.npz)를 읽는다. 같은 폴더에 pca_manifest.json도 있어야 한다.

        쓰는 곳: mcdl/chart_source.py (최종 모델 쌍을 평가할 때 학습 때 저장한 관리도를 그대로 쓴다).
        """
        return cls(_load_pca(run_dir))

    def save(self, run_dir, features: pd.DataFrame) -> None:
        """관리도를 run_dir에 파일 세 개로 저장한다.

        pca_arrays.npz(관리도 숫자들), pca_manifest.json(각 값의 종류·크기 목록),
        pca_data_nor.csv(표준화한 학습 특징, 기록용). features는 fit에 쓴 학습 진단 특징이다.
        쓰는 곳: Train_.py (학습 끝).
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
        """차량 한 대의 진단 특징으로 시점별 (T², SPE, CI) 세 배열을 계산한다.

        CI = SPE/SPE 관리한계 + T²/T² 관리한계. 두 지표가 모두 관리한계 이하이면 CI는 2 이하다.
        쓰는 곳: Test_.py(진단 그림 모드), Test_tsne.py(알람 시점 찾기), 아래 max_ci.
        """
        t2 = _t2(features, self._mean, self._std, self._p_k, self._v_I)
        spe = _spe(features, self._mean, self._std, self._p_k)
        return t2, spe, (spe / self.spe_limit) + (t2 / self.t2_limit)

    def max_ci(self, features: pd.DataFrame) -> float:
        """차량 한 대의 판정 점수(최대 CI)를 돌려준다: CI 시계열에서 가장 큰 값 (NaN은 무시).

        이 값이 임계값보다 크면 그 차량을 고장으로 판정한다(ROC 계산은 mcdl/roc.py).
        쓰는 곳: Test_.py, Test_multi_models.py.
        """
        return float(np.nanmax(np.asarray(self.score(features)[2], dtype=float)))


def _sliding_average(s, n: int):
    """signal_features 안에서 쓴다(X-model 특징). 각 시점에서 바로 앞 n개 값의 평균을 구한다.

    처음 n개 시점은 앞에 값이 모자라므로 처음 n개 값의 평균으로 채운다. 길이가 n 이하이면 그대로 돌려준다.
    """
    a = np.asarray(s, dtype=float).reshape(-1)
    L = a.shape[0]
    if L <= n:
        return a
    out = np.empty(L, dtype=float)
    out[:n] = float(np.mean(a[:n]))
    c = np.concatenate(([0.0], np.cumsum(a, dtype=float)))
    window_sums = c[n:] - c[:-n]  # ends at n..L
    out[n:] = window_sums[:-1] / n  # ends at n..L-1
    return out


def _pca(data, l1, l2):
    """ControlChart.fit 안에서 쓴다. 학습 진단 특징으로 PCA를 하고 T²·SPE 관리한계를 계산한다.

    순서: 특징별 표준화(평균 0, 표준편차 1) → 공분산 행렬 → 특이값 분해(SVD) → 누적 기여율이 95%를
    넘는 지점까지의 주성분 선택 → T² 한계(F 분포 근사)와 SPE 한계(Jackson–Mudholkar 근사) 계산.
    l1, l2는 함께 저장하는 두 번째 한계의 기준(0.99)이다.
    돌려주는 값 15개: (v_I, v, v_ratio, p_k, data_mean, data_std, T_95_limit, T_99_limit, SPE_95_limit,
    SPE_99_limit, P, k, P_t, X, data_nor). 이름의 뜻은 README 8장의 "pca_arrays.npz 키" 표에 있다.
    """
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
    """ControlChart.score 안에서 쓴다. 시점마다 호텔링 T²를 계산한다.

    특징을 학습 때의 평균·표준편차로 표준화한 뒤, 선택한 주성분 방향에서 각 성분을 그 분산으로 나눠
    제곱합한다(평소 퍼진 정도에 비해 얼마나 멀리 있는지).
    """
    D = p_k @ v_I @ p_k.T  # (F×F)
    X = np.asarray((data_in - data_mean) / data_std, dtype=float)  # (N×F)
    return np.sum((X @ D) * X, axis=1)


def _spe(data_in, data_mean, data_std, p_k):
    """ControlChart.score 안에서 쓴다. 시점마다 SPE(주성분으로 설명되지 않는 잔차의 제곱합)를 계산한다.

    식: SPE_i = ||(I - p_k p_k^T) x_i||^2 (x_i는 표준화한 특징 벡터).
    """
    X = np.asarray((data_in - data_mean) / data_std, dtype=float)  # (N×F)
    M = np.eye(p_k.shape[0]) - p_k @ p_k.T  # 주성분 공간 밖으로 보내는 투영 (F×F)
    R = X @ M  # 잔차 (N×F)
    return np.einsum("ij,ij->i", R, R)  # 행마다 제곱합


def _save_pca(output_dir, pca_outputs):
    """ControlChart.save 안에서 쓴다. 관리도 값 15개를 파일로 저장한다.

    숫자 14개는 pca_arrays.npz 하나에, 각 값의 종류·크기·경로는 pca_manifest.json에, 표준화한 학습 특징
    (data_nor)은 pca_data_nor.csv에 쓴다. 기존 Run 폴더와 같은 형식이라 예전 Run도 그대로 읽힌다.
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

    # data_nor(표준화한 학습 특징 DataFrame)는 기록용 CSV로 저장한다
    data_nor.to_csv(df_path, index=True)

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
            "data_nor": "DataFrame",
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
            "data_nor": list(data_nor.shape),
        },
        "paths": {"arrays": arrays_path, "data_nor": df_path},
    }
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)


def _load_pca(output_dir):
    """ControlChart.load 안에서 쓴다. pca_arrays.npz에서 관리도 값을 읽어 _pca와 같은 15개 순서로 돌려준다.

    표준화한 학습 특징(data_nor)은 평가에 필요 없어 읽지 않고 None으로 채운다.
    pca_manifest.json이 없으면 저장이 끝나지 않은 폴더로 보고 FileNotFoundError를 낸다.
    """
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
