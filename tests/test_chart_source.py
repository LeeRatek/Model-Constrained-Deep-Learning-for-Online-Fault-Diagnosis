"""관리도 공급원 테스트. tmp_path에 만든 Run과 작은 가짜 원본 데이터(QAS 열 구성)를 쓴다."""

import numpy as np
import pytest
import torch

from fake_vehicles import fake_vehicle, write_fake_vehicles, write_split
from mcdl.chart_source import (
    RECOMPUTE_BATCH_SIZE,
    chart_for_pair,
    load_cached_features,
    save_cached_features,
)
from mcdl.diagnosis import ControlChart, combine_signal_features, diagnosis_features
from mcdl.reconstruction import training_features, training_signal_features
from mcdl.run import Run, build_model_pair, checkpoint_filename

SIM_CONFIG = "battery_type : QAS\nlearning_case : 1\nadd_one_output_layer : True\nno_use_dx : False\n"
TRAIN_IDS = [3, 7, 11]


@pytest.fixture
def run(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)  # 분할 경로 "."은 cwd 기준이다
    write_split(tmp_path, "train", TRAIN_IDS)
    write_fake_vehicles(tmp_path / "data", TRAIN_IDS)

    run_dir = tmp_path / "run"
    (run_dir / "artifact").mkdir(parents=True)
    (run_dir / "sim_config.txt").write_text(SIM_CONFIG, encoding="utf-8")
    run = Run(run_dir)
    generator = torch.Generator().manual_seed(0)
    for model, net in zip(("U", "X"), build_model_pair(run.config)):
        for number in (-1, 0):
            state = {
                k: (torch.rand(v.shape, dtype=torch.float64, generator=generator) - 0.5) * 0.02
                for k, v in net.state_dict().items()
            }
            torch.save(state, run_dir / "artifact" / checkpoint_filename(model, number))
    return run


def preprocessed_training_vehicles(run):
    from Function_ import get_input_dimensions, preprocess_loaded_tensor

    dims = get_input_dimensions("QAS")
    return [
        preprocess_loaded_tensor(*fake_vehicle(v), dims, "QAS", False, True)
        for v in TRAIN_IDS
    ]


def recompute_conditions(abs_err=True):
    return {"abs_err": abs_err, "float32": False, "batch_size": RECOMPUTE_BATCH_SIZE}


def assert_same_chart(a, b):
    assert a.t2_limit == b.t2_limit
    assert a.spe_limit == b.spe_limit
    probe = diagnosis_features(*(np.random.default_rng(1).random((50, 110)) for _ in range(2)))
    for s, t in zip(a.score(probe), b.score(probe)):
        assert np.array_equal(s, t)


def test_recompute_matches_training_features(run, tmp_path):
    expected = ControlChart.fit(
        training_features(
            *run.load_model_pair(0, -1),
            preprocessed_training_vehicles(run),
            batch_size=RECOMPUTE_BATCH_SIZE,
            abs_err=True,
        )
    )

    chart = chart_for_pair(run, 0, -1, source_data_dir=tmp_path / "data")

    assert_same_chart(chart, expected)


def test_final_pair_without_cache_uses_saved_chart(run, tmp_path):
    features = training_features(
        *run.load_model_pair(-1, -1),
        preprocessed_training_vehicles(run),
        batch_size=RECOMPUTE_BATCH_SIZE,
        abs_err=True,
    )
    ControlChart.fit(features).save(f"{run.path}/", features)

    chart = chart_for_pair(run, -1, -1, source_data_dir=tmp_path / "missing")

    assert_same_chart(chart, ControlChart.load(run.path))


def test_cache_with_same_conditions_is_used(run, tmp_path):
    vehicles = preprocessed_training_vehicles(run)
    net, netx = run.load_model_pair(0, -1)
    u_feat = training_signal_features(net, "U", [u for u, _ in vehicles], batch_size=RECOMPUTE_BATCH_SIZE, abs_err=True)
    x_feat = training_signal_features(netx, "X", [x for _, x in vehicles], batch_size=RECOMPUTE_BATCH_SIZE, abs_err=True)
    save_cached_features(run, "U", 0, u_feat, conditions=recompute_conditions())
    save_cached_features(run, "X", -1, x_feat, conditions=recompute_conditions())

    # 원본 데이터가 없어도 캐시만으로 관리도를 얻는다
    chart = chart_for_pair(run, 0, -1, source_data_dir=tmp_path / "missing")

    assert_same_chart(chart, ControlChart.fit(combine_signal_features(u_feat, x_feat)))


def test_cache_with_other_conditions_is_ignored(run, tmp_path, capsys):
    vehicles = preprocessed_training_vehicles(run)
    net, netx = run.load_model_pair(0, -1)
    other = {"abs_err": True, "float32": False, "batch_size": 4096}
    save_cached_features(run, "U", 0, training_signal_features(net, "U", [u for u, _ in vehicles], batch_size=4096, abs_err=True), conditions=other)
    save_cached_features(run, "X", -1, training_signal_features(netx, "X", [x for _, x in vehicles], batch_size=4096, abs_err=True), conditions=other)

    chart_for_pair(run, 0, -1, source_data_dir=tmp_path / "data")

    assert "Recomputing the control chart" in capsys.readouterr().out


def test_cached_features_round_trip(run):
    features = tuple(np.arange(3.0) + i for i in range(3))

    path = save_cached_features(run, "X", 5, features, conditions=recompute_conditions(abs_err=False))

    assert path == run.feature_cache_path("X", 5, **recompute_conditions(abs_err=False))
    loaded = load_cached_features(run, "X", 5, conditions=recompute_conditions(abs_err=False))
    assert all(np.array_equal(a, b) for a, b in zip(loaded, features))
