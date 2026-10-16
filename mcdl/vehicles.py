"""차량 데이터를 읽는 코드: 분할 파일에서 차량 ID를 읽고, 차량별 원본을 전처리한 텐서를 돌려준다.

분할(split)
    차량 ID를 용도별로 나눈 목록이다. train(학습), validate(검증), test(정상 평가 차량), fault(고장 차량),
    normal(정상 전체)이 있고, 저장소 루트나 datasets/<이름>/ 폴더에 <배터리 종류>_filtered_vehicle_ids_<분할>.npy
    파일로 저장되어 있다(data_filter.py가 만든다).

차량 텐서
    차량 한 대의 데이터는 (U 텐서, X 텐서) 두 개다. U 텐서는 vin_2.pkl을, X 텐서는 vin_3.pkl을 전처리한 것으로
    각각 U-model(셀 전압)과 X-model(셀 SOC)의 입력이 된다.

.pt 캐시
    차량 수십~수백 대를 원본에서 읽고 전처리하는 데는 시간이 오래 걸린다. 그래서 여러 대를 한꺼번에 읽을 때는
    결과를 source_data_dir(기본 ./data)에 .pt 파일 하나로 저장해 두고, 다음 실행에서 같은 조건이면 그 파일을 읽는다.
    Test_multi_models.py가 이 방식을 쓴다. 파일 이름의 긴 문자열(해시)은 차량 목록과 전처리 설정으로 만든 값이라,
    조건이 바뀌면 다른 파일이 된다.

누가 쓰나
    Train_.py, Test_.py, Test_tsne.py(split_ids, load_vehicle), Test_multi_models.py(load_training_vehicles,
    load_test_vehicles), mcdl/chart_source.py(관리도를 다시 계산할 때).
"""

from __future__ import annotations

import hashlib
import json
import os

import numpy as np
import torch

from mcdl.preprocessing import get_input_dimensions, preprocess_loaded_tensor, safe_load


def split_ids(split_dir, battery_type: str, split: str, *, start: int = 0, end: int = -1) -> list[int]:
    """분할 파일에서 차량 ID 목록(int)을 읽는다. split은 "train", "validate", "test", "fault", "normal" 중 하나.

    start~end 구간만 돌려준다(end 포함, end=-1이면 끝까지). 학습할 때 --vehicle-start/--vehicle-end로
    train 분할의 일부만 쓰는 데 쓴다. 예: split_ids(".", "QAS", "train", start=0, end=9) → 앞의 10대.
    쓰는 곳: Train_.py, Test_.py, Test_multi_models.py, Test_tsne.py, mcdl/chart_source.py.
    """
    ids = (
        np.load(f"{split_dir}/{battery_type}_filtered_vehicle_ids_{split}.npy")
        .astype(np.int64)
        .tolist()
    )
    return ids[start : end + 1] if end != -1 else ids[start:]


def load_vehicle(source_data_dir, config, vehicle_id):
    """차량 한 대의 원본 vin_2.pkl·vin_3.pkl을 읽어 설정대로 전처리한 (U 텐서, X 텐서)를 돌려준다.

    config는 mcdl/run.py의 RunConfig이며, 배터리 종류와 전처리 방식(learning_case 등)을 담고 있다.
    학습 때와 같은 config로 읽어야 평가에서도 같은 전처리가 된다.
    쓰는 곳: Train_.py(학습·검증 차량), Test_.py·Test_tsne.py(평가 차량), 아래 두 함수, mcdl/chart_source.py.
    """
    folder = f"{source_data_dir}/{config.battery_type}/{vehicle_id}"
    tensor = safe_load(f"{folder}/vin_2.pkl")
    tensorx = safe_load(f"{folder}/vin_3.pkl")
    return preprocess_loaded_tensor(
        tensor,
        tensorx,
        get_input_dimensions(config.battery_type),
        config.battery_type,
        config.preprocessing,
        config.skip_charge_ready,
        normalize_dx=config.normalize_dx,
        normalize_val=config.normalize_val,
    )


def load_training_vehicles(source_data_dir, config, ids):
    """학습 차량 여러 대를 [(U 텐서, X 텐서), ...] 목록으로 돌려준다 (load_vehicle 결과 그대로, float64).

    같은 차량 목록·전처리 설정으로 만든 캐시 파일(source_data_dir/train_data_list_<배터리>_<해시>.pt)이 있으면
    그것을 읽고, 없으면 원본에서 한 대씩 읽은 뒤 캐시 파일을 만들어 둔다.
    쓰는 곳: Test_multi_models.py (체크포인트마다 학습 진단 특징을 계산할 때).
    """
    path = _cache_path(source_data_dir, config, "train_data_list", "train_list", ids)
    vehicles = None
    if os.path.exists(path):
        print(f"Found cached training data (list): {path}")
        print("Loading...")
        try:
            vehicles = torch.load(path)
            print("Successfully loaded cached training data.")
        except Exception as e:
            print(f"Error loading cache: {e}. Will regenerate.")

    if vehicles is None:
        print("Cache miss or error. Processing raw pickle files...")
        vehicles = []
        for count, vehicle_id in enumerate(ids, start=1):
            if count % 10 == 0:
                print(f"  Processing vehicle {count}/{len(ids)}")
            vehicles.append(load_vehicle(source_data_dir, config, vehicle_id))

        print(f"Saving training data list to cache: {path}")
        try:
            torch.save(vehicles, path)
            print("Cache saved successfully.")
        except Exception as e:
            print(f"Warning: Could not save cache to {path}: {e}")
    return vehicles


def load_test_vehicles(source_data_dir, config, ids):
    """평가 차량 여러 대를 {차량 ID: (U 텐서, X 텐서)} dict로 돌려준다.

    캐시 파일(source_data_dir/test_data_cache_<배터리>_<해시>.pt)이 있으면 그것을 읽고, 없으면 원본에서 읽은 뒤
    캐시 파일을 만들어 둔다. ids의 순서도 해시에 들어가므로, 같은 차량이라도 순서가 다르면 다른 캐시다.
    쓰는 곳: Test_multi_models.py (모든 체크포인트 조합에서 같은 평가 차량을 반복해 쓴다).
    """
    path = _cache_path(source_data_dir, config, "test_data_cache", "test_list", ids)
    vehicles = {}
    if os.path.exists(path):
        print(f"Found cached test data: {path}")
        print("Loading...")
        try:
            vehicles = torch.load(path)
            print(f"Successfully loaded cached test data: {len(vehicles)} vehicles")
        except Exception as e:
            print(f"Error loading test cache: {e}. Will regenerate.")

    if not vehicles:
        print("Cache miss or error. Processing raw test pickle files...")
        for count, vehicle_id in enumerate(ids, start=1):
            if count % 10 == 0:
                print(f"  Loading test vehicle {count}/{len(ids)}")
            vehicles[vehicle_id] = load_vehicle(source_data_dir, config, vehicle_id)

        print(f"Saving test data to cache: {path}")
        try:
            torch.save(vehicles, path)
            print("Test cache saved successfully.")
        except Exception as e:
            print(f"Warning: Could not save test cache to {path}: {e}")
    return vehicles


def _cache_path(source_data_dir, config, prefix: str, list_key: str, ids) -> str:
    """load_training_vehicles·load_test_vehicles 안에서 쓴다. .pt 캐시 파일 경로를 만든다.

    차량 ID 목록(순서 포함)과 전처리 설정(배터리 종류, 정규화 여부, 충전 대기 구간 제거 여부, Δ 나누기 여부)을
    JSON 문자열로 만들고, 그 문자열의 md5 해시(32자리 16진수 지문)를 파일 이름에 넣는다.
    조건이 하나라도 다르면 이름이 달라지므로 다른 조건의 캐시를 잘못 읽지 않는다.
    주의: 전처리 "코드"를 고친 것은 이름에 반영되지 않으므로, 그때는 캐시 파일을 직접 지워야 한다.
    """
    key = {
        list_key: list(ids),
        "battery_type": config.battery_type,
        "preprocessing": config.preprocessing,
        "skip_charge_ready": config.skip_charge_ready,
        "normalize_dx": config.normalize_dx,
    }
    # normalize_val은 normalize_dx를 켰을 때만 넣는다. 그래야 normalize_dx를 쓰지 않은 예전 Run의
    # 캐시 파일 이름이 이 항목이 생기기 전과 똑같이 유지되어 기존 캐시를 그대로 다시 쓸 수 있다.
    if config.normalize_dx:
        key["normalize_val"] = config.normalize_val
    digest = hashlib.md5(json.dumps(key, sort_keys=True).encode("utf-8")).hexdigest()
    return os.path.join(source_data_dir, f"{prefix}_{config.battery_type}_{digest}.pt")
