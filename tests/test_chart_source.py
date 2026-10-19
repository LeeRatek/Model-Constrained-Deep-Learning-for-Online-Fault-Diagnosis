"""mcdl/chart_source.py 테스트: 모델 쌍마다 관리도를 어디서 가져오는지(캐시 → 저장된 파일 → 다시 계산) 확인한다.

실제 학습 결과 대신, 임시 폴더에 가짜 Run 폴더(sim_config.txt + 무작위 가중치 체크포인트)와 가짜 원본 데이터
(fake_vehicles.py)를 만들어 쓴다. 어느 경로로 얻든 같은 조건이면 똑같은 관리도가 나와야 한다.
"""

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

SIM_CONFIG = "battery_type : QAS\nlearning_case : 1\nadd_one_output_layer : True\nno_use_dx : False\n"  # 가짜 Run 설정
TRAIN_IDS = [3, 7, 11]  # 가짜 학습 차량 ID


@pytest.fixture
def run(tmp_path, monkeypatch):
    """테스트용 가짜 Run을 만들어 돌려준다. 이 파일의 테스트가 인자 이름 run으로 받아 쓴다.

    @pytest.fixture가 붙은 함수는 pytest가 테스트 실행 전에 불러, 돌려준 값을 같은 이름의 테스트 인자로 넣어 준다.
    tmp_path(테스트마다 새로 만드는 빈 임시 폴더)와 monkeypatch(테스트 동안만 설정을 바꾸고 끝나면 되돌리는 도구)도
    pytest가 기본으로 주는 fixture다.
    만드는 것: 임시 폴더에 기본 분할(datasets/basic)의 train .npy, 가짜 원본 데이터, run/sim_config.txt,
    체크포인트 번호 -1·0의 U·X 모델 가중치(시드 0의 작은 무작위 값).
    """
    monkeypatch.chdir(tmp_path)  # 분할 경로 "./datasets/basic"은 작업 폴더 기준이다
    split_dir = tmp_path / "datasets" / "basic"
    split_dir.mkdir(parents=True)
    write_split(split_dir, "train", TRAIN_IDS)
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
    """가짜 학습 차량들을 파일을 거치지 않고 만들어 전처리한 (vin_2, vin_3) 목록. 기대값을 직접 계산할 때 쓴다."""
    from mcdl.preprocessing import get_input_dimensions, preprocess_loaded_tensor

    dims = get_input_dimensions("QAS")
    return [
        preprocess_loaded_tensor(*fake_vehicle(v), dims, "QAS", False, True)
        for v in TRAIN_IDS
    ]


def recompute_conditions(abs_err=True):
    """chart_for_pair가 관리도를 다시 계산할 때 쓰는 조건. 특징 캐시는 이 조건이 같을 때만 다시 쓰인다."""
    return {"abs_err": abs_err, "float32": False, "batch_size": RECOMPUTE_BATCH_SIZE}


def assert_same_chart(a, b):
    """두 관리도가 같은지 확인한다: 관리한계가 같고, 같은 무작위 입력에 대한 T²·SPE·CI가 완전히 같아야 한다."""
    assert a.t2_limit == b.t2_limit
    assert a.spe_limit == b.spe_limit
    probe = diagnosis_features(*(np.random.default_rng(1).random((50, 110)) for _ in range(2)))
    for s, t in zip(a.score(probe), b.score(probe)):
        assert np.array_equal(s, t)


def test_recompute_matches_training_features(run, tmp_path):
    """캐시도 저장 파일도 없는 중간 체크포인트 쌍은 원본 데이터로 다시 계산하며, 직접 계산한 관리도와 같아야 한다."""
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
    """최종 모델 쌍(-1, -1)은 캐시가 없으면 Run 폴더에 저장된 관리도 파일을 쓴다(원본 데이터 폴더가 없어도 된다)."""
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
    """같은 조건으로 저장된 특징 캐시가 있으면 원본 데이터 없이 캐시만으로 관리도를 만든다."""
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
    """다른 조건(여기서는 batch 크기)으로 만든 캐시는 쓰지 않고 다시 계산한다.

    capsys는 테스트 중 화면 출력(print)을 잡아 두는 pytest 기본 fixture다. 다시 계산할 때 나오는 안내 문구로 확인한다.
    """
    vehicles = preprocessed_training_vehicles(run)
    net, netx = run.load_model_pair(0, -1)
    other = {"abs_err": True, "float32": False, "batch_size": 4096}
    save_cached_features(run, "U", 0, training_signal_features(net, "U", [u for u, _ in vehicles], batch_size=4096, abs_err=True), conditions=other)
    save_cached_features(run, "X", -1, training_signal_features(netx, "X", [x for _, x in vehicles], batch_size=4096, abs_err=True), conditions=other)

    chart_for_pair(run, 0, -1, source_data_dir=tmp_path / "data")

    assert "Recomputing the control chart" in capsys.readouterr().out


def test_cached_features_round_trip(run):
    """특징 캐시를 저장한 뒤 같은 조건으로 읽으면 같은 값이 돌아오고, 저장 경로가 Run.feature_cache_path와 같은지 본다."""
    features = tuple(np.arange(3.0) + i for i in range(3))

    path = save_cached_features(run, "X", 5, features, conditions=recompute_conditions(abs_err=False))

    assert path == run.feature_cache_path("X", 5, **recompute_conditions(abs_err=False))
    loaded = load_cached_features(run, "X", 5, conditions=recompute_conditions(abs_err=False))
    assert all(np.array_equal(a, b) for a, b in zip(loaded, features))
