"""원본 데이터 파일을 읽고 전처리하는 코드.

원본 데이터
    차량마다 data/<배터리 종류>/<차량 ID>/ 폴더에 vin_2.pkl(U-model 입력, 셀 전압 관련)과
    vin_3.pkl(X-model 입력, 셀 SOC 관련)이 있다. 둘 다 (시점 수 × 열 수) 크기의 float64 torch 텐서를 pickle로
    저장한 파일이다. 열 구성은 get_input_dimensions가 정한다.

전처리
    learning_case 번호에 따라 충전 대기 구간(학습에 맞지 않는 앞부분)을 잘라내거나, 열을 0~1로
    정규화한다(README 5장). DTI 배터리는 단위 변환 등 별도 처리를 한다.

누가 쓰나
    mcdl/vehicles.py(차량 한 대씩 읽기), mcdl/run.py(모델 크기 정하기), data_filter.py(이상 차량 걸러내기).
"""

from __future__ import annotations

import io
import operator
import pickle

import numpy as np
import torch

# learning_case 번호 → (0~1 정규화 여부, 충전 대기 구간 제거 여부)
_LEARNING_CASES = {1: (False, True), 2: (True, True), 3: (True, False), 4: (False, False)}

# last_index_of가 받는 비교 연산
_COMPARE = {"<": operator.lt, "<=": operator.le, ">": operator.gt}


class _CPUUnpickler(pickle.Unpickler):
    """pickle 안의 torch 텐서를 GPU가 아니라 CPU로 복원하는 pickle 읽기 도구. 쓰는 곳: safe_load."""

    def find_class(self, module, name):
        """torch 텐서 복원 함수를 만나면, CPU로 복원하는 함수로 바꿔 준다."""
        if module == "torch.storage" and name == "_load_from_bytes":
            return lambda b: torch.load(
                io.BytesIO(b), map_location=torch.device("cpu"), weights_only=False
            )
        return super().find_class(module, name)


def safe_load(path: str):
    """원본 텐서 파일(.pkl)을 읽는다. GPU에서 저장된 텐서도 GPU가 없는 컴퓨터에서 읽을 수 있게 CPU로 복원한다.

    쓰는 곳: mcdl/vehicles.py의 load_vehicle(원본 vin_2·vin_3 읽기), data_filter.py.
    """
    with open(path, "rb") as f:
        return _CPUUnpickler(f).load()


def get_convergence_start_index(tensor):
    """DTI 데이터에서 값이 안정되기 시작하는 시점을 찾는다.

    첫 번째 열(0번)만 본다. 바로 이웃한 시점 사이의 변화량이 20 시점 동안 계속 0.1보다 작은 첫 구간을 찾아
    그 시작 위치(i + 1)를 돌려준다. 그런 구간이 없으면 0(자르지 않음)을 돌려준다.
    쓰는 곳: preprocess_loaded_tensor (DTI의 불안정한 앞부분을 잘라낼 때).
    """
    threshold, stable_window_size = 0.1, 20
    diffs = np.abs(np.diff(tensor[:, 0].detach().cpu().numpy()))
    for i in range(len(diffs) - stable_window_size):
        if np.all(diffs[i : i + stable_window_size] < threshold):
            # diffs[i]는 i와 i+1 사이의 변화이므로, 안정 구간은 i+1부터 시작한다
            return i + 1
    return 0


def last_index_of(tensor, feature_idx, value, op):
    """한 열(feature_idx)에서 조건(값 op value)을 만족하는 마지막 시점을 찾아 "그 위치 + 1"을 돌려준다.

    조건을 만족하는 시점이 없으면 0을 돌려준다. 그래서 돌려받은 값부터 데이터를 잘라 쓰면
    "조건을 만족한 마지막 시점 이후"만 남는다. op는 "<", "<=", ">" 중 하나.
    예: last_index_of(tensor, 224, -500, "<") → 전류가 -500보다 작았던 마지막 시점 + 1.
    쓰는 곳: preprocess_loaded_tensor(앞부분 자르기), is_abnormal(이상 차량 판정).
    """
    hits = torch.nonzero(_COMPARE[op](tensor[:, feature_idx], value))
    return int(hits[-1]) + 1 if len(hits) else 0


def is_abnormal(tensor):
    """차량 데이터가 비정상(측정 오류로 보이는 값 포함)인지 판정한다.

    첫 번째 열(팩 전압 추정값)에 0보다 작거나 5보다 큰 값이 한 시점이라도 있으면 True.
    쓰는 곳: data_filter.py (분할을 만들 때 이런 차량을 뺀다).
    """
    return last_index_of(tensor, 0, 0, "<") != 0 or last_index_of(tensor, 0, 5, ">") != 0


def normalize_columns_0_to_1(X, exclude_range, eps=1e-12):
    """(시점 × 열) 텐서의 각 열을 (값 − 최솟값) / (최댓값 − 최솟값)으로 0~1 범위에 맞춘 새 텐서를 돌려준다.

    exclude_range=(시작 열, 끝 열) 구간(끝 포함)은 정규화하지 않고 그대로 둔다. 최댓값과 최솟값이 같은 열은
    0으로 나누지 않도록 분모를 eps로 바꾼다.
    쓰는 곳: preprocess_loaded_tensor (learning_case 2·3에서 0~1 정규화를 할 때).
    """
    col_mask = torch.ones(X.shape[1], dtype=torch.bool)
    col_mask[exclude_range[0] : exclude_range[1] + 1] = False
    X_sel = X[:, col_mask]
    col_min = X_sel.min(dim=0, keepdim=True).values
    col_max = X_sel.max(dim=0, keepdim=True).values
    out = X.clone()
    out[:, col_mask] = (X_sel - col_min) / torch.clamp(col_max - col_min, min=eps)
    return out


def get_preprocessing_and_skip_charge_ready(learning_case: int):
    """learning_case 번호(1~4)를 (0~1 정규화 여부, 충전 대기 구간 제거 여부) 두 값으로 바꾼다.

    1: (False, True), 2: (True, True), 3: (True, False), 4: (False, False). 그 밖의 번호는 ValueError.
    쓰는 곳: mcdl/run.py의 RunConfig.from_mapping (설정의 preprocessing·skip_charge_ready 계산).
    """
    if learning_case not in _LEARNING_CASES:
        raise ValueError(f"지원하지 않는 learning_case 값입니다: {learning_case}")
    return _LEARNING_CASES[learning_case]


def get_input_dimensions(BATTERY_TYPE: str):
    """배터리 종류별로 차량 텐서의 열 구성(각 구간의 열 개수)을 돌려준다.

    vin_2(U-model 입력)는 [x | y | z | q], vin_3(X-model 입력)은 [x2 | y2 | z2 | q2] 순서로 열이 놓여 있다.
    셀 수는 QAS 110개, DTI 85개다.
    쓰는 곳: mcdl/run.py(모델 입력·출력 크기), mcdl/vehicles.py·data_filter.py(전처리), 테스트.
    """
    n_cells = 110 if BATTERY_TYPE == "QAS" else 85
    return {
        "x": 2,  # 팩 전압 추정값 2개
        "y": n_cells,  # 셀 전압: 모델이 재구성하는 대상
        "z": n_cells,  # 셀 전압 편차 추정값 (코드에서 Δ 또는 dx로 부른다)
        "q": 3,  # 보드 온도, 보드단 SOC, 전류
        "x2": 2,  # 팩 SOC 추정값 2개
        "y2": n_cells,  # 셀 SOC: 모델이 재구성하는 대상
        "z2": n_cells,  # 셀 SOC 편차 추정값
        "q2": 4,  # 보드 온도, 보드단 SOC, 속도, 전류
    }


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
        # 0번 열이 안정되기 전(20 시점 동안 변화 < 0.1이 되기 전)의 앞부분을 잘라낸다
        start_idx_dti = get_convergence_start_index(tensor) if SKIP_CHARGE_READY else 0
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

    # 여기부터 QAS
    dx_start = dim_dict["x"] + dim_dict["y"]  # vin_2의 편차(Δ) 구간 시작
    q_start = dx_start + dim_dict["z"]  # vin_2의 q(온도, SOC, 전류) 시작
    dx2_start = dim_dict["x2"] + dim_dict["y2"]  # vin_3의 편차 구간 시작

    tensorx[:, dx2_start + dim_dict["z2"] + 2] = 0  # vin_3의 속도 열

    if SKIP_CHARGE_READY:
        start_idx = last_index_of(tensor, q_start + 2, -500, "<")  # 충전 대기 중에는 전류가 -500 미만이다
        tensor = tensor[start_idx:, :]
        tensorx = tensorx[start_idx:, :]

    if PREPROCESSING:
        # 팩 전압이 0 이하이거나 5 초과였던 마지막 시점까지 잘라낸다
        start_idx = max(last_index_of(tensor, 0, 0, "<="), last_index_of(tensor, 0, 5, ">"))
        cell_div_range = (dx_start, q_start - 1)
        tensor = normalize_columns_0_to_1(tensor[start_idx:, :], cell_div_range)
        tensorx = normalize_columns_0_to_1(tensorx[start_idx:, :], cell_div_range)
        if normalize_dx:
            tensor[:, dx_start:q_start].div_(normalize_val)
            tensorx[:, dx2_start : dx2_start + dim_dict["z2"]].div_(normalize_val)
    return tensor, tensorx
