"""차량 데이터: 분할의 차량 ID와, 원본(vin_2·vin_3)을 Run 설정대로 전처리한 차량 텐서.

여러 차량을 한꺼번에 읽을 때는 source_data_dir에 .pt 캐시를 만들고 다음 실행에서 다시 쓴다.
캐시 파일 이름(해시 규칙)과 내용은 기존 Test_multi_models.py와 같다. 용어는 GLOSSARY.md를 따른다.
"""

from __future__ import annotations

import hashlib
import json
import os

import numpy as np
import torch

from mcdl.preprocessing import get_input_dimensions, preprocess_loaded_tensor, safe_load


def split_ids(split_dir, battery_type: str, split: str, *, start: int = 0, end: int = -1) -> list[int]:
    """분할(train/validate/test/fault/normal)의 차량 ID. start~end(끝 포함, -1이면 끝까지)만 돌려준다."""
    ids = (
        np.load(f"{split_dir}/{battery_type}_filtered_vehicle_ids_{split}.npy")
        .astype(np.int64)
        .tolist()
    )
    return ids[start : end + 1] if end != -1 else ids[start:]


def load_vehicle(source_data_dir, config, vehicle_id):
    """원본 vin_2·vin_3을 읽어 설정(RunConfig)대로 전처리한 (U 텐서, X 텐서)."""
    folder = f"{source_data_dir}/{config.battery_type}/{vehicle_id}"
    tensor = safe_load(f"{folder}/vin_2.pkl", verbose=False)
    tensorx = safe_load(f"{folder}/vin_3.pkl", verbose=False)
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
    """학습 차량들의 [(U, X)] (double). .pt 캐시(train_data_list_<BT>_<hash>.pt)가 있으면 그것을 쓴다."""
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
            tensor, tensorx = load_vehicle(source_data_dir, config, vehicle_id)
            vehicles.append((tensor.double(), tensorx.double()))

        print(f"Saving training data list to cache: {path}")
        try:
            torch.save(vehicles, path)
            print("Cache saved successfully.")
        except Exception as e:
            print(f"Warning: Could not save cache to {path}: {e}")
    return vehicles


def load_test_vehicles(source_data_dir, config, ids):
    """테스트 차량들의 {id: (U, X)}. .pt 캐시(test_data_cache_<BT>_<hash>.pt)가 있으면 그것을 쓴다."""
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
            vehicles = {}

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
    """캐시 파일 경로. 해시는 차량 ID 목록(순서 포함)과 전처리 설정의 md5다."""
    key = {
        list_key: list(ids),
        "battery_type": config.battery_type,
        "preprocessing": config.preprocessing,
        "skip_charge_ready": config.skip_charge_ready,
        "normalize_dx": config.normalize_dx,
    }
    # normalize_dx일 때만 넣어, 그 밖의 Run은 기존 캐시 해시를 그대로 쓴다
    if config.normalize_dx:
        key["normalize_val"] = config.normalize_val
    digest = hashlib.md5(json.dumps(key, sort_keys=True).encode("utf-8")).hexdigest()
    return os.path.join(source_data_dir, f"{prefix}_{config.battery_type}_{digest}.pt")
