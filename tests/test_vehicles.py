"""mcdl/vehicles.py 테스트: 분할 파일 읽기, 차량 원본 읽기·전처리, 차량 묶음 캐시(.pt 파일)를 확인한다.

임시 폴더에 가짜 원본 데이터와 분할 파일(fake_vehicles.py)을 만들어 쓴다. 캐시 파일 이름은 기존 Run들이 이미
만들어 둔 캐시를 그대로 다시 쓸 수 있도록 정해진 규칙(documented_cache_name)과 똑같아야 한다.
"""

import hashlib
import json
import shutil
from pathlib import Path

import numpy as np
import pytest
import torch

from fake_vehicles import fake_vehicle, write_fake_vehicles, write_split
from mcdl.preprocessing import get_input_dimensions, preprocess_loaded_tensor
from mcdl.run import RunConfig
from mcdl.vehicles import load_test_vehicles, load_training_vehicles, load_vehicle, split_ids

IDS = [4, 9, 15, 21]  # 가짜 차량 ID
CONFIG = RunConfig.from_mapping({"battery_type": "QAS", "learning_case": 1})  # 기본 학습 설정 (learning_case 1)


def documented_cache_name(prefix, list_key, ids, config):
    """캐시 파일 이름 규칙: 차량 ID 목록과 전처리 조건을 JSON으로 적어 MD5 해시(내용이 같으면 늘 같은 32자 문자열)를 낸다.

    설계 문서(.scratch/vehicle-data/spec.md)에 적힌 규칙을 mcdl/vehicles.py와 따로 다시 구현해,
    두 쪽이 같은 이름을 내는지 대조한다.
    """
    key = {
        list_key: ids,
        "battery_type": config.battery_type,
        "preprocessing": config.preprocessing,
        "skip_charge_ready": config.skip_charge_ready,
        "normalize_dx": config.normalize_dx,
    }
    if config.normalize_dx:
        key["normalize_val"] = config.normalize_val
    digest = hashlib.md5(json.dumps(key, sort_keys=True).encode("utf-8")).hexdigest()
    return f"{prefix}_{config.battery_type}_{digest}.pt"


@pytest.fixture
def data_dir(tmp_path):
    """가짜 차량 4대의 원본 데이터를 둔 임시 data 폴더 경로. 이 파일의 테스트가 인자 이름 data_dir로 받아 쓴다."""
    write_fake_vehicles(tmp_path / "data", IDS)
    return tmp_path / "data"


def test_split_ids_reads_float_npy_and_applies_range(tmp_path):
    """실수형으로 저장된 분할 파일도 정수 ID 목록으로 읽고, start·end 범위(end 포함, -1은 끝까지)를 적용하는지 본다."""
    write_split(tmp_path, "train", [5, 6, 7, 8, 9], dtype=np.float64)  # datasets/*의 분할은 float64로 저장되어 있다

    assert split_ids(tmp_path, "QAS", "train") == [5, 6, 7, 8, 9]
    assert all(type(i) is int for i in split_ids(tmp_path, "QAS", "train"))
    assert split_ids(tmp_path, "QAS", "train", start=1, end=2) == [6, 7]  # 끝 포함
    assert split_ids(tmp_path, "QAS", "train", start=3) == [8, 9]  # end=-1이면 끝까지


def test_load_vehicle_matches_manual_preprocessing(data_dir):
    """load_vehicle이 원본 파일을 읽어 전처리한 결과가, 같은 데이터를 직접 전처리한 결과와 같은지 본다."""
    u, x = load_vehicle(data_dir, CONFIG, 9)

    expected_u, expected_x = preprocess_loaded_tensor(
        *fake_vehicle(9), get_input_dimensions("QAS"), "QAS", False, True
    )
    assert torch.equal(u, expected_u) and torch.equal(x, expected_x)
    assert torch.all(x[:, 224] == 0)  # QAS는 X 텐서의 속도 열을 0으로 둔다


def test_training_vehicles_are_cached_and_reused(data_dir):
    """학습 차량 묶음을 처음 읽으면 규칙대로 이름 붙은 캐시가 생기고, 원본을 지워도 캐시로 같은 값을 읽는지 본다."""
    first = load_training_vehicles(data_dir, CONFIG, IDS)

    cache = data_dir / documented_cache_name("train_data_list", "train_list", IDS, CONFIG)
    assert cache.exists()
    assert all(u.dtype == torch.float64 and x.dtype == torch.float64 for u, x in first)

    shutil.rmtree(data_dir / "QAS")  # 원본이 없어도 캐시로 읽는다
    second = load_training_vehicles(data_dir, CONFIG, IDS)
    assert len(second) == len(IDS)
    for (u1, x1), (u2, x2) in zip(first, second):
        assert torch.equal(u1, u2) and torch.equal(x1, x2)


def test_test_vehicles_are_cached_by_id(data_dir):
    """테스트 차량 묶음은 {차량 ID: 데이터} 형태로 캐시되고, 원본을 지워도 캐시로 같은 값을 읽는지 본다."""
    first = load_test_vehicles(data_dir, CONFIG, IDS)

    assert list(first) == IDS
    assert (data_dir / documented_cache_name("test_data_cache", "test_list", IDS, CONFIG)).exists()
    shutil.rmtree(data_dir / "QAS")
    second = load_test_vehicles(data_dir, CONFIG, IDS)
    assert all(torch.equal(first[i][0], second[i][0]) for i in IDS)


def test_vehicle_order_is_part_of_the_cache_key(data_dir):
    """같은 차량이라도 순서가 다르면 다른 캐시 파일이 생기는지 본다(학습 순서가 결과에 영향을 주므로)."""
    load_training_vehicles(data_dir, CONFIG, IDS)
    load_training_vehicles(data_dir, CONFIG, list(reversed(IDS)))

    assert len(list(data_dir.glob("train_data_list_QAS_*.pt"))) == 2


def test_normalize_val_enters_the_key_only_with_normalize_dx(data_dir):
    """normalize_val은 Δ 정규화(normalize_dx)를 켰을 때만 캐시 이름에 영향을 주는지 본다(끄면 값이 쓰이지 않으므로)."""
    plain_4 = RunConfig.from_mapping({"battery_type": "QAS", "learning_case": 2, "normalize_val": 4})
    plain_8 = RunConfig.from_mapping({"battery_type": "QAS", "learning_case": 2, "normalize_val": 8})
    dx_4 = RunConfig.from_mapping({"battery_type": "QAS", "learning_case": 2, "normalize_dx": True, "normalize_val": 4})
    dx_8 = RunConfig.from_mapping({"battery_type": "QAS", "learning_case": 2, "normalize_dx": True, "normalize_val": 8})

    names = {c: documented_cache_name("test_data_cache", "test_list", IDS[:1], c) for c in (plain_4, plain_8, dx_4, dx_8)}
    assert names[plain_4] == names[plain_8]
    assert names[dx_4] != names[dx_8]
    for config in (plain_4, dx_4, dx_8):
        load_test_vehicles(data_dir, config, IDS[:1])
        assert (data_dir / names[config]).exists()


REAL_RUN = Path(__file__).resolve().parents[1] / "models" / "260224_090359"  # 있으면 실제 데이터 테스트에 쓴다
REAL_DATA = Path(__file__).resolve().parents[1] / "data"


@pytest.mark.realdata
@pytest.mark.skipif(not (REAL_RUN.exists() and REAL_DATA.exists()), reason="기준 Run 또는 data/ 없음")
def test_documented_rule_names_the_existing_real_caches(monkeypatch):
    """실제 Run(260224_090359)이 만들어 둔 학습·테스트 캐시 파일 이름이 이 규칙으로 똑같이 계산되는지 본다.

    models/260224_090359와 data/가 모두 있을 때만 돌고, 없으면 건너뛴다.
    """
    from mcdl.run import Run

    monkeypatch.chdir(REAL_RUN.parents[1])
    config = Run(REAL_RUN).config
    train = split_ids(config.split_dir, "QAS", "train", start=config.vehicle_start, end=config.vehicle_end)
    test = split_ids(config.split_dir, "QAS", "test") + split_ids(config.split_dir, "QAS", "fault")

    assert documented_cache_name("train_data_list", "train_list", train, config) == (
        "train_data_list_QAS_015ac371f38644b99d2e0d2c85ec0ba7.pt"
    )
    assert documented_cache_name("test_data_cache", "test_list", test, config) == (
        "test_data_cache_QAS_528fb02abeb3f5a1f220b9fe6857d592.pt"
    )
