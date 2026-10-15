"""원본 데이터 로딩과 전처리.

차량 폴더의 pkl(vin_2·vin_3)을 CPU 텐서로 읽고, battery_type·learning_case 설정대로 전처리한다.
차량 단위 로딩과 캐시는 mcdl/vehicles.py가, 분할을 만드는 이상 차량 판정은 data_filter.py가 이 module을 쓴다.
함수 본문은 예전 Function_.py에서 그대로 옮겼다.
"""

from __future__ import annotations

import io
import os
import pickle
from typing import Any

import numpy as np
import torch


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
