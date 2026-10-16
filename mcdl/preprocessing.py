"""원본 데이터 파일을 읽고 전처리하는 코드.

원본 데이터
    차량마다 data/<배터리 종류>/<차량 ID>/ 폴더에 vin_2.pkl(U-model 입력, 셀 전압 관련)과
    vin_3.pkl(X-model 입력, 셀 SOC 관련)이 있다. 둘 다 (시점 수 × 열 수) 크기의 torch 텐서를 pickle로
    저장한 파일이다. 열 구성은 get_input_dimensions가 정한다.

전처리
    learning_case 번호에 따라 충전 대기 구간(학습에 맞지 않는 앞부분)을 잘라내거나, 열을 0~1로
    정규화한다(README 5장). DTI 배터리는 단위 변환 등 별도 처리를 한다.

누가 쓰나
    mcdl/vehicles.py(차량 한 대씩 읽기), mcdl/run.py(모델 크기 정하기), data_filter.py(이상 차량 걸러내기).
    함수 본문은 예전 공통 함수 파일(Function_.py, 지금은 없음)에서 그대로 옮겨 왔다.
"""

from __future__ import annotations

import io
import os
import pickle
from typing import Any

import numpy as np
import torch


def get_convergence_start_index(data, threshold=0.1, stable_window_size=20):
    """DTI 데이터에서 값이 안정되기 시작하는 시점을 찾는다.

    data는 (시점 × 열) 2차원이고, 첫 번째 열(0번)만 본다. 바로 이웃한 시점 사이의 변화량이
    stable_window_size 시점 동안 계속 threshold보다 작은 첫 구간을 찾아 그 시작 위치(i + 1)를 돌려준다.
    그런 구간이 없으면 0(자르지 않음)을 돌려준다.
    쓰는 곳: preprocess_loaded_tensor (DTI의 불안정한 앞부분을 잘라낼 때).
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
    """safe_load 안에서 쓴다. 읽은 객체 안의 torch 텐서를 모두 CPU로 옮긴다 (리스트·튜플·dict 안쪽까지).

    GPU에서 저장된 파일을 GPU가 없는 컴퓨터에서도 쓸 수 있게 하려는 것이다.
    """

    def log(message: str) -> None:
        """verbose일 때만 메시지를 출력한다."""
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
    """파일 확장자에 맞는 방법으로 파일을 읽고, 안의 torch 텐서는 CPU로 옮겨 돌려준다.

    - .pkl/.pickle: pickle로 읽는다 (GPU에서 저장된 텐서도 CPU로 읽도록 처리)
    - .pt/.pth/.ckpt: torch.load로 읽는다 (실패하면 pickle로 다시 시도)
    - .npy/.npz: numpy.load로 읽는다
    - 그 밖의 확장자: pickle, 안 되면 torch.load 순서로 시도한다
    verbose=True면 어떤 방법으로 읽는지 출력한다.
    쓰는 곳: mcdl/vehicles.py의 load_vehicle(원본 vin_2·vin_3 읽기), data_filter.py.
    """

    def log(message: str) -> None:
        """verbose일 때만 메시지를 출력한다."""
        if verbose:
            print(message)

    ext = os.path.splitext(path)[1].lower()

    class CPUUnpickler(pickle.Unpickler):
        """pickle 안의 torch 텐서를 GPU가 아니라 CPU로 복원하는 pickle 읽기 도구."""

        def find_class(self, module, name):
            """torch 텐서 복원 함수를 만나면, CPU로 복원하는 함수로 바꿔 준다."""
            if module == "torch.storage" and name == "_load_from_bytes":
                return lambda b: torch.load(
                    io.BytesIO(b), map_location=torch.device("cpu"), weights_only=False
                )
            return super().find_class(module, name)

    def load_pickle():
        """pickle로 읽는다. CPU 복원 방식이 실패하면 일반 pickle.load로 다시 읽는다."""
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
        """torch.load로 CPU에 읽는다."""
        log("Loading with torch.load...")
        return _to_cpu(
            torch.load(path, map_location=torch.device("cpu"), weights_only=False),
            verbose=verbose,
        )

    def load_numpy():
        """numpy.load로 읽는다."""
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
    """한 열(feature_idx)에서 조건(값 operator value)을 만족하는 마지막 시점을 찾아 "그 위치 + 1"을 돌려준다.

    조건을 만족하는 시점이 없으면 0을 돌려준다. 그래서 돌려받은 값부터 데이터를 잘라 쓰면
    "조건을 만족한 마지막 시점 이후"만 남는다. operator는 "==", "<", "<=", ">", ">=", "!=" 중 하나.
    예: last_index_of(tensor, 224, -500, "<") → 전류가 -500보다 작았던 마지막 시점 + 1.
    쓰는 곳: preprocess_loaded_tensor(앞부분 자르기), is_abnormal(이상 차량 판정).
    """
    # torch 텐서를 numpy로 통째로 복사하면 매우 느릴 수 있어, torch 입력은 torch 연산으로만 처리한다.

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
    """차량 데이터가 비정상(측정 오류로 보이는 값 포함)인지 판정한다.

    첫 번째 열(팩 전압 추정값)에 0보다 작거나 5보다 큰 값이 한 시점이라도 있으면 True.
    쓰는 곳: data_filter.py (분할을 만들 때 이런 차량을 뺀다).
    """
    abnormal_data = False
    if (
        last_index_of(tensor, 0, 0, operator="<") != 0
        or last_index_of(tensor, 0, 5, operator=">") != 0
    ):
        abnormal_data = True
    return abnormal_data


def normalize_columns_0_to_1(X, eps=1e-12, exclude_cols=None, exclude_ranges=None):
    """
    입력 행렬의 각 열을 [0,1] 범위로 정규화합니다 (열마다 (값 - 최솟값) / (최댓값 - 최솟값)).

    쓰는 곳: preprocess_loaded_tensor (learning_case 2·3에서 0~1 정규화를 할 때).
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
        """정규화할 열은 True, 제외할 열(exclude_cols, exclude_ranges)은 False인 배열을 만든다."""
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
    """learning_case 번호(1~4)를 (0~1 정규화 여부, 충전 대기 구간 제거 여부) 두 값으로 바꾼다.

    1: (False, True), 2: (True, True), 3: (True, False), 4: (False, False). 그 밖의 번호는 ValueError.
    쓰는 곳: mcdl/run.py의 RunConfig.from_mapping (설정의 preprocessing·skip_charge_ready 계산).
    """
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
    """배터리 종류별로 차량 텐서의 열 구성(각 구간의 열 개수)을 돌려준다.

    vin_2(U-model 입력)는 [x | y | z | q], vin_3(X-model 입력)은 [x2 | y2 | z2 | q2] 순서로 열이 놓여 있다.
    셀 수는 QAS 110개, DTI 85개다.
    쓰는 곳: mcdl/run.py(모델 입력·출력 크기), mcdl/vehicles.py·data_filter.py(전처리), 테스트.
    """
    dim_dict = dict()
    dim_dict["x"] = 2  # 팩 전압 추정값 2개
    dim_dict["y"] = (
        110 if BATTERY_TYPE == "QAS" else 85
    )  # 셀 전압 (셀 수만큼): 모델이 재구성하는 대상
    dim_dict["z"] = (
        110 if BATTERY_TYPE == "QAS" else 85
    )  # 셀 전압 편차 추정값 (셀 수만큼, 코드에서 Δ 또는 dx로 부른다)
    dim_dict["q"] = 3  # 보드 온도, 보드단 SOC, 전류

    dim_dict["x2"] = 2  # 팩 SOC 추정값 2개
    dim_dict["y2"] = (
        110 if BATTERY_TYPE == "QAS" else 85
    )  # 셀 SOC (셀 수만큼): 모델이 재구성하는 대상
    dim_dict["z2"] = (
        110 if BATTERY_TYPE == "QAS" else 85
    )  # 셀 SOC 편차 추정값 (셀 수만큼)
    dim_dict["q2"] = 4  # 보드 온도, 보드단 SOC, 속도, 전류

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
    """차량 한 대의 원본 텐서(vin_2 → tensor, vin_3 → tensorx)를 학습·평가에 쓸 수 있게 전처리한다.

    - QAS: X-model 입력의 속도 열을 0으로 둔다. SKIP_CHARGE_READY면 전류가 -500보다 작았던 마지막
      시점까지(충전 대기 구간)를 잘라낸다. PREPROCESSING이면 팩 전압이 0 이하이거나 5 초과였던 마지막
      시점까지를 더 잘라낸 뒤, 편차(Δ) 구간을 뺀 모든 열을 0~1로 정규화한다. normalize_dx면 Δ를
      normalize_val로 나눈다.
    - DTI: 값이 안정되기 전의 앞부분을 잘라내고(get_convergence_start_index), 단위를 바꾼다
      (mV → V, % → 비율). 0~1 정규화는 하지 않는다.
    두 텐서는 같은 시점으로 함께 잘린다. dim_dict는 get_input_dimensions의 결과다.
    쓰는 곳: mcdl/vehicles.py의 load_vehicle(학습·평가의 모든 차량), data_filter.py.
    """
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
