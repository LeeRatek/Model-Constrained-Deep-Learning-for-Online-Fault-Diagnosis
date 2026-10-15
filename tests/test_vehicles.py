"""차량 데이터 테스트. tmp_path의 가짜 원본 데이터(QAS 열 구성 pkl)와 분할 .npy를 쓴다."""

import hashlib
import json
import shutil
from pathlib import Path

import numpy as np
import pytest
import torch

from fake_vehicles import fake_vehicle, write_fake_vehicles, write_split
from Function_ import get_input_dimensions, preprocess_loaded_tensor
from mcdl.run import RunConfig
from mcdl.vehicles import load_test_vehicles, load_training_vehicles, load_vehicle, split_ids

IDS = [4, 9, 15, 21]
CONFIG = RunConfig.from_mapping({"battery_type": "QAS", "learning_case": 1})


def documented_cache_name(prefix, list_key, ids, config):
    """spec에 적힌 해시 규칙 (module과 별개로 다시 구현해 파일 이름을 대조한다)."""
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
    write_fake_vehicles(tmp_path / "data", IDS)
    return tmp_path / "data"


def test_split_ids_reads_float_npy_and_applies_range(tmp_path):
    write_split(tmp_path, "train", [5, 6, 7, 8, 9], dtype=np.float64)  # datasets/*의 분할은 float64로 저장되어 있다

    assert split_ids(tmp_path, "QAS", "train") == [5, 6, 7, 8, 9]
    assert all(type(i) is int for i in split_ids(tmp_path, "QAS", "train"))
    assert split_ids(tmp_path, "QAS", "train", start=1, end=2) == [6, 7]  # 끝 포함
    assert split_ids(tmp_path, "QAS", "train", start=3) == [8, 9]  # end=-1이면 끝까지


def test_load_vehicle_matches_manual_preprocessing(data_dir):
    u, x = load_vehicle(data_dir, CONFIG, 9)

    expected_u, expected_x = preprocess_loaded_tensor(
        *fake_vehicle(9), get_input_dimensions("QAS"), "QAS", False, True
    )
    assert torch.equal(u, expected_u) and torch.equal(x, expected_x)
    assert torch.all(x[:, 224] == 0)  # QAS는 X 텐서의 속도 열을 0으로 둔다


def test_training_vehicles_are_cached_and_reused(data_dir):
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
    first = load_test_vehicles(data_dir, CONFIG, IDS)

    assert list(first) == IDS
    assert (data_dir / documented_cache_name("test_data_cache", "test_list", IDS, CONFIG)).exists()
    shutil.rmtree(data_dir / "QAS")
    second = load_test_vehicles(data_dir, CONFIG, IDS)
    assert all(torch.equal(first[i][0], second[i][0]) for i in IDS)


def test_vehicle_order_is_part_of_the_cache_key(data_dir):
    load_training_vehicles(data_dir, CONFIG, IDS)
    load_training_vehicles(data_dir, CONFIG, list(reversed(IDS)))

    assert len(list(data_dir.glob("train_data_list_QAS_*.pt"))) == 2


def test_normalize_val_enters_the_key_only_with_normalize_dx(data_dir):
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


REAL_RUN = Path(__file__).resolve().parents[1] / "models" / "260224_090359"
REAL_DATA = Path(__file__).resolve().parents[1] / "data"


@pytest.mark.realdata
@pytest.mark.skipif(not (REAL_RUN.exists() and REAL_DATA.exists()), reason="기준 Run 또는 data/ 없음")
def test_documented_rule_names_the_existing_real_caches(monkeypatch):
    """기준 Run(260224_090359)의 학습·테스트 캐시 이름이 이 규칙으로 계산된다."""
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
