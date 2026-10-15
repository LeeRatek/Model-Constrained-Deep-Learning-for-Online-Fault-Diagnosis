"""진입 스크립트 smoke 테스트. tmp 작업 폴더의 가짜 원본 데이터로 학습(Train_.main)부터 평가까지 잇는다."""

import contextlib
import importlib
import io
import sys

import matplotlib

matplotlib.use("Agg")

import pandas as pd
import pytest

from fake_vehicles import N_STEPS, write_fake_vehicles, write_split
from mcdl.diagnosis import ControlChart
from mcdl.run import Run

SPLITS = {"train": [0, 1], "validate": [2], "test": [3, 4], "fault": [5, 6]}
DEBUG_VEHICLE = 338  # Test_ --debug가 고정해 평가하는 차량
MAIN_SCRIPTS = ["Train_", "data_filter", "Test_", "Test_tsne", "Test_multi_models", "Plot_AUC_heatmap"]


@pytest.mark.parametrize("name", MAIN_SCRIPTS)
def test_import_runs_nothing(name, monkeypatch, capsys):
    monkeypatch.setattr(sys, "argv", ["prog", "--no-such-option"])  # import 때 파싱하면 SystemExit가 난다
    sys.modules.pop(name, None)

    module = importlib.import_module(name)

    assert callable(module.main)
    assert capsys.readouterr().out == ""


@pytest.fixture(scope="module")
def workspace(tmp_path_factory):
    """가짜 원본 데이터(data/QAS/<id>)와 루트 분할 .npy를 둔 작업 폴더. 스크립트는 상대 경로를 쓰므로 cwd를 옮긴다."""
    root = tmp_path_factory.mktemp("workspace")
    write_fake_vehicles(root / "data", [i for ids in SPLITS.values() for i in ids] + [DEBUG_VEHICLE])
    for split, ids in SPLITS.items():
        write_split(root, split, ids)

    with pytest.MonkeyPatch.context() as mp:
        mp.chdir(root)
        yield root


def run_main(name, argv):
    """스크립트의 main(argv)를 실행하고 콘솔 출력을 돌려준다."""
    module = importlib.import_module(name)
    out = io.StringIO()
    with contextlib.redirect_stdout(out):
        module.main(argv)
    return out.getvalue()


TRAIN_ARGS = [
    "--source-data-dir", "data",
    "--ae-u-epochs", "1",
    "--ae-x-epochs", "1",
    "--ae-u-save-per", "1",
    "--ae-x-save-per", "1",
]


@pytest.fixture(scope="module")
def trained_run(workspace):
    """Train_.main으로 학습 차량 2대·검증 차량 1대, 1 epoch Run을 만든다."""
    run_main("Train_", ["--models-dir", "models", *TRAIN_ARGS])
    (run_dir,) = (workspace / "models").iterdir()
    return run_dir


def training_rows(run_dir):
    """학습 진단 특징의 행 수 (pca_data_nor.csv). 가짜 차량은 전처리로 잘리지 않아 차량당 N_STEPS행이다."""
    return len(pd.read_csv(run_dir / "pca_data_nor.csv"))


def test_train_creates_a_run(trained_run):
    run = Run(trained_run)

    assert run.config.battery_type == "QAS"
    assert "debug" not in (trained_run / "sim_config.txt").read_text(encoding="utf-8")
    assert run.checkpoint_numbers("U") == [0, -1]
    assert run.checkpoint_numbers("X") == [0, -1]
    assert (trained_run / "loss_net_val.csv").exists()  # 검증 차량 로딩까지 거쳤다
    chart = ControlChart.load(trained_run)
    assert chart.t2_limit > 0 and chart.spe_limit > 0
    assert training_rows(trained_run) == len(SPLITS["train"]) * N_STEPS


def test_train_debug_uses_one_vehicle_and_writes_no_sim_config(workspace):
    out = run_main("Train_", ["--models-dir", "models_debug", "--debug", *TRAIN_ARGS])

    (run_dir,) = (workspace / "models_debug").iterdir()
    assert "현재 디버그 모드로 실행 중입니다." in out
    assert not (run_dir / "sim_config.txt").exists()
    assert (run_dir / "artifact" / "net.pth").exists()
    assert training_rows(run_dir) == N_STEPS  # 학습 차량 1대


def test_train_checkpoint_schedule_with_shuffle_and_validation(workspace):
    """3 epoch, save-per 2: 체크포인트와 검증 loss는 epoch 0, 2에만 남고, 학습 loss는 epoch마다 남는다."""
    run_main(
        "Train_",
        ["--models-dir", "models_schedule", "--source-data-dir", "data",
         "--ae-u-epochs", "3", "--ae-x-epochs", "3", "--ae-u-save-per", "2", "--ae-x-save-per", "2",
         "--ae-u-shuffle", "--ae-x-shuffle"],
    )

    (run_dir,) = (workspace / "models_schedule").iterdir()
    run = Run(run_dir)
    assert run.checkpoint_numbers("U") == [0, 2, -1]
    assert run.checkpoint_numbers("X") == [0, 2, -1]
    for name in ("loss_net", "loss_netx"):
        assert pd.read_csv(run_dir / f"{name}.csv").epoch.tolist() == [0, 1, 2]
        assert pd.read_csv(run_dir / f"{name}_val.csv").epoch.tolist() == [0, 2]


def test_test_evaluates_the_run_with_either_option_name(trained_run):
    for option in ("--models-folder", "--models-idx"):
        out = run_main("Test_", ["--models-dir", "models", option, trained_run.name, "--source-data-dir", "data"])

        assert "AUC = " in out
        assert "Falling back to saved 'pca_arrays.npz'" in out  # -1/-1은 학습 때 저장한 관리도
    assert (trained_run / "results" / "AUC_ROC1.png").exists()


def test_test_models_dir_defaults_to_model_dir_env(trained_run, monkeypatch):
    monkeypatch.setenv("MODEL_DIR", "models")
    module = importlib.reload(importlib.import_module("Test_"))  # parser 기본값은 import 때 정해진다

    out = io.StringIO()
    with contextlib.redirect_stdout(out):
        module.main(["--models-idx", trained_run.name, "--source-data-dir", "data"])

    assert "AUC = " in out.getvalue()


def test_test_debug_fixes_vehicle_and_keeps_chart_pair(trained_run):
    out = run_main(
        "Test_",
        ["--models-dir", "models", "--models-idx", trained_run.name, "--source-data-dir", "data",
         "--u-model-idx", "0", "--x-model-idx", "0", "--debug"],
    )

    assert f"vehicle ID={DEBUG_VEHICLE}..." in out
    assert all(f"vehicle ID={i}..." not in out for ids in (SPLITS["test"], SPLITS["fault"]) for i in ids)
    assert "Recomputing the control chart for (u=0, x=0)" in out  # 관리도도 모델과 같은 쌍 (known-bugs 11)


def test_tsne_saves_the_plot(trained_run):
    fault = SPLITS["fault"][0]

    run_main(
        "Test_tsne",
        ["--models-dir", "models", "--models-idx", trained_run.name, "--source-data-dir", "data",
         "--fault-idx", str(fault), "--per-vehicle", "50", "--save"],
    )

    assert (trained_run / "results_tsne" / f"tSNE_{fault}.png").exists()


def test_multi_evaluates_all_checkpoint_pairs_and_heatmap(trained_run):
    out = run_main(
        "Test_multi_models",
        ["--models-dir", "models", "--models-idx", trained_run.name, "--source-data-dir", "data"],
    )

    summary = pd.read_csv(trained_run / "AUC_summary_case1.csv")
    assert sorted(zip(summary.u_model_idx, summary.x_model_idx)) == [(-1, -1), (-1, 0), (0, -1), (0, 0)]
    assert (trained_run / "AUC_heatmap_case1.png").exists()
    assert not (trained_run / "temp_cache").exists()  # --keep-cache가 없으면 지운다
    assert "All model combinations tested successfully" in out

    run_main("Plot_AUC_heatmap", ["--models-dir", "models", "--models-idx", trained_run.name])
    assert (trained_run / "AUC_summary_case1_heatmap.png").exists()
