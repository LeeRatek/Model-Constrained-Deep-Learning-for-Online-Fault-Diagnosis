"""실행 스크립트(Train_.py, Test_.py 등 저장소 루트의 .py) smoke 테스트.

smoke 테스트: 결과 값이 정확한지까지는 따지지 않고, 전체 흐름이 오류 없이 끝까지 돌아가고 기대한 파일이
생기는지만 빠르게 확인하는 테스트. (전자 제품에 처음 전원을 넣어 연기가 나지 않는지 보는 데서 온 말이다.)

임시 작업 폴더에 가짜 원본 데이터(fake_vehicles.py)와 분할 파일을 만들고, 학습(Train_.main)으로 작은 Run을
만든 뒤 그 Run으로 Test_, Test_tsne, Test_multi_models, Plot_AUC_heatmap을 차례로 실행한다(약 20초).
"""

import contextlib
import importlib
import io
import shutil
import sys

import matplotlib

matplotlib.use("Agg")  # 그림 창을 띄우지 않고 파일로만 그린다 (테스트 중 창이 뜨지 않게)

import matplotlib.pyplot as plt
import pandas as pd
import pytest
import torch

from fake_vehicles import N_STEPS, write_fake_vehicles, write_split
from mcdl.diagnosis import ControlChart
from mcdl.lstm import load_lstm_model
from mcdl.run import Run

SPLITS = {"train": [0, 1], "validate": [2], "test": [3, 4], "fault": [5, 6]}  # 가짜 분할 (차량 ID)
DEBUG_VEHICLE = 338  # Test_ --debug가 고정해 평가하는 차량
MAIN_SCRIPTS = ["Train_", "Train_lstm", "data_filter", "Test_", "Test_tsne", "Test_multi_models", "Plot_AUC_heatmap"]


@pytest.mark.parametrize("name", MAIN_SCRIPTS)
def test_import_runs_nothing(name, monkeypatch, capsys):
    """스크립트를 import만 하면 아무것도 실행·출력되지 않고 main 함수만 준비되는지 본다.

    그래야 테스트나 다른 코드가 스크립트의 함수를 안전하게 가져다 쓸 수 있다. 명령줄에 없는 옵션을 넣어 두어,
    import 중에 옵션을 읽으려 하면 바로 실패하게 만든다.
    """
    monkeypatch.setattr(sys, "argv", ["prog", "--no-such-option"])  # import 때 파싱하면 SystemExit가 난다
    sys.modules.pop(name, None)

    module = importlib.import_module(name)

    assert callable(module.main)
    assert capsys.readouterr().out == ""


@pytest.fixture(scope="module")
def workspace(tmp_path_factory):
    """가짜 원본 데이터(data/QAS/<id>)와 기본 분할 .npy(datasets/basic/)를 둔 임시 작업 폴더. 스크립트는 상대 경로를
    쓰므로 이 파일의 테스트가 도는 동안 작업 폴더(cwd)를 여기로 옮겨 둔다.

    scope="module"이라 이 파일의 테스트 전체가 한 번 만든 폴더를 함께 쓴다(테스트마다 새로 만들지 않는다).
    """
    root = tmp_path_factory.mktemp("workspace")
    write_fake_vehicles(root / "data", [i for ids in SPLITS.values() for i in ids] + [DEBUG_VEHICLE])
    split_dir = root / "datasets" / "basic"
    split_dir.mkdir(parents=True)
    for split, ids in SPLITS.items():
        write_split(split_dir, split, ids)

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


TRAIN_ARGS = [  # 빨리 끝나도록 1 epoch만 학습하는 Train_.py 옵션
    "--source-data-dir", "data",
    "--ae-u-epochs", "1",
    "--ae-x-epochs", "1",
    "--ae-u-save-per", "1",
    "--ae-x-save-per", "1",
]


@pytest.fixture(scope="module")
def trained_run(workspace):
    """Train_.main으로 학습 차량 2대·검증 차량 1대, 1 epoch Run을 만들고 그 폴더 경로를 돌려준다.

    이 파일의 평가 스크립트 테스트들이 이 Run 하나를 함께 쓴다.
    """
    run_main("Train_", ["--models-dir", "models", *TRAIN_ARGS])
    (run_dir,) = (workspace / "models").iterdir()
    return run_dir


def training_rows(run_dir):
    """학습 진단 특징의 행 수 (pca_data_nor.csv). 가짜 차량은 전처리로 잘리지 않아 차량당 N_STEPS행이다."""
    return len(pd.read_csv(run_dir / "pca_data_nor.csv"))


def test_train_creates_a_run(trained_run):
    """학습이 Run 폴더에 설정 파일, 체크포인트(0번과 최종), 검증 loss, 관리도를 남기는지 본다."""
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
    """Train_ --debug는 학습 차량 1대만 쓰고 sim_config.txt를 남기지 않는지 본다."""
    out = run_main("Train_", ["--models-dir", "models_debug", "--debug", *TRAIN_ARGS])

    (run_dir,) = (workspace / "models_debug").iterdir()
    assert "현재 디버그 모드로 실행 중입니다." in out
    assert not (run_dir / "sim_config.txt").exists()
    assert (run_dir / "artifact" / "net.pth").exists()
    assert training_rows(run_dir) == N_STEPS  # 학습 차량 1대


def test_train_and_evaluation_read_the_same_named_split(workspace):
    """--dataset-name default(data_filter.py의 기본 이름)로 학습하면 datasets/default 분할로 학습하고,
    평가 쪽(Run)도 같은 폴더를 읽는지 본다. 예전에는 학습만 이 폴더를 쓰고 평가는 기본 분할을 읽었다
    (.scratch/known-bugs/issues/15-train-eval-split-dir-differ.md)."""
    split_dir = workspace / "datasets" / "default"
    split_dir.mkdir(parents=True)
    write_split(split_dir, "train", [2])  # 기본 분할(학습 차량 2대)과 다른 차량 1대
    write_split(split_dir, "validate", [3])

    run_main("Train_", ["--models-dir", "models_named", "--dataset-name", "default", *TRAIN_ARGS])

    (run_dir,) = (workspace / "models_named").iterdir()
    assert Run(run_dir).config.split_dir == "./datasets/default"
    assert training_rows(run_dir) == N_STEPS  # datasets/default의 학습 차량 1대로 학습했다


def test_train_stops_when_the_named_split_is_missing(workspace):
    """--dataset-name의 분할 폴더가 없으면 기본 분할로 학습하지 않고, Run 폴더를 만들기 전에 오류로 멈추는지 본다."""
    with pytest.raises(FileNotFoundError, match="datasets/no_such_split"):
        run_main("Train_", ["--models-dir", "models_missing", "--dataset-name", "no_such_split", *TRAIN_ARGS])

    assert not (workspace / "models_missing").exists()


def test_train_lstm_needs_only_vin1_and_saves_only_the_lstm(workspace):
    """Train_lstm.py가 MC-AE와 상관없이 LSTM만 학습하는지 본다.

    학습 차량의 vin_1.pkl만 있는 데이터 폴더로 실행해도 끝까지 돌고(vin_2·vin_3를 읽지 않는다), Run 폴더에는
    sim_config.txt, loss_lstm.csv, artifact/lstm.pth만 남는다(MC-AE 체크포인트·관리도 없음). 이 Run을 평가 쪽(Run)에
    주면 LSTM 전용 Run이라는 오류로 멈춘다. 같은 시드로 다시 학습하면 같은 가중치가 나온다.
    """
    for vehicle_id in SPLITS["train"]:
        folder = workspace / "data_vin1_only" / "QAS" / str(vehicle_id)
        folder.mkdir(parents=True)
        shutil.copy(workspace / "data" / "QAS" / str(vehicle_id) / "vin_1.pkl", folder)
    lstm_args = ["--source-data-dir", "data_vin1_only", "--epochs", "2"]

    run_main("Train_lstm", ["--models-dir", "models_lstm", *lstm_args])
    run_main("Train_lstm", ["--models-dir", "models_lstm_again", *lstm_args])

    (run_dir,) = (workspace / "models_lstm").iterdir()
    assert sorted(p.relative_to(run_dir).as_posix() for p in run_dir.rglob("*") if p.is_file()) == [
        "artifact/lstm.pth",
        "loss_lstm.csv",
        "sim_config.txt",
    ]
    assert pd.read_csv(run_dir / "loss_lstm.csv").epoch.tolist() == [0, 1]
    lstm = load_lstm_model(run_dir / "artifact")
    assert lstm(torch.rand(4, 1, 7, dtype=torch.float64)).shape == (4, 1, 2)
    with pytest.raises(ValueError, match="LSTM만 학습한 Run"):
        Run(run_dir)

    (again_dir,) = (workspace / "models_lstm_again").iterdir()
    again = load_lstm_model(again_dir / "artifact").state_dict()
    assert all(torch.equal(value, again[name]) for name, value in lstm.state_dict().items())


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
    """Test_가 Run을 평가해 AUC와 ROC 그림을 내는지, 옛 옵션 이름 --models-folder로도 되는지 본다."""
    for option in ("--models-folder", "--models-idx"):
        out = run_main("Test_", ["--models-dir", "models", option, trained_run.name, "--source-data-dir", "data"])

        assert "AUC = " in out
        assert "Falling back to saved 'pca_arrays.npz'" in out  # -1/-1은 학습 때 저장한 관리도
    assert (trained_run / "results" / "AUC_ROC1.png").exists()


def test_test_models_dir_defaults_to_model_dir_env(trained_run, monkeypatch):
    """--models-dir를 주지 않으면 환경 변수 MODEL_DIR의 폴더를 쓰는지 본다."""
    monkeypatch.setenv("MODEL_DIR", "models")
    module = importlib.reload(importlib.import_module("Test_"))  # parser 기본값은 import 때 정해진다

    out = io.StringIO()
    with contextlib.redirect_stdout(out):
        module.main(["--models-idx", trained_run.name, "--source-data-dir", "data"])

    assert "AUC = " in out.getvalue()


def test_test_debug_fixes_vehicle_and_keeps_chart_pair(trained_run):
    """Test_ --debug는 차량 338 하나만 평가하고, 관리도도 고른 모델 쌍(0, 0)으로 다시 계산하는지 본다.

    예전에는 --debug일 때 관리도만 최종 모델 쌍 것을 쓰는 버그가 있었다
    (.scratch/known-bugs/issues/11-test-debug-chart-pair-mismatch.md).
    """
    out = run_main(
        "Test_",
        ["--models-dir", "models", "--models-idx", trained_run.name, "--source-data-dir", "data",
         "--u-model-idx", "0", "--x-model-idx", "0", "--debug"],
    )

    assert f"vehicle ID={DEBUG_VEHICLE}..." in out
    assert all(f"vehicle ID={i}..." not in out for ids in (SPLITS["test"], SPLITS["fault"]) for i in ids)
    assert "Recomputing the control chart for (u=0, x=0)" in out  # 관리도도 모델과 같은 쌍


def test_tsne_saves_the_plot(trained_run):
    """Test_tsne --save가 고장 차량 하나의 t-SNE 그림 파일을 저장하고, 저장한 그림을 닫는지 본다.

    대상 차량마다 그림을 그리므로 닫지 않으면 차량 수만큼 그림이 메모리에 쌓인다
    (.scratch/known-bugs/issues/14-tsne-figures-not-closed.md).
    """
    fault = SPLITS["fault"][0]
    plt.close("all")  # 앞선 테스트가 열어 둔 그림(창 표시용 ROC 등)은 세지 않는다

    run_main(
        "Test_tsne",
        ["--models-dir", "models", "--models-idx", trained_run.name, "--source-data-dir", "data",
         "--fault-idx", str(fault), "--per-vehicle", "50", "--save"],
    )

    assert (trained_run / "results_tsne" / f"tSNE_{fault}.png").exists()
    assert plt.get_fignums() == []


def test_multi_evaluates_all_checkpoint_pairs_and_heatmap(trained_run):
    """Test_multi_models가 체크포인트 쌍 4개(U·X 각각 0번과 최종)를 모두 평가해 AUC 표와 히트맵을 남기고,
    임시 캐시를 지우는지 본다. 이어서 Plot_AUC_heatmap이 그 표로 히트맵을 다시 그리는지도 본다."""
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
