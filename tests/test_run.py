"""mcdl/run.py 테스트: 학습 결과 폴더(Run)의 설정 읽기·쓰기, 체크포인트 찾기, 모델 만들기·불러오기를 확인한다.

원본 차량 데이터 없이 CPU에서 돈다. 설정은 실제 Run의 sim_config.txt를 복사해 둔
tests/fixtures/sim_config_260224_090359.txt를 기준으로 쓴다(fixture: 테스트가 읽어 쓰려고 미리 준비해 둔 데이터 파일).
맨 아래 테스트 하나만 실제 Run 폴더(models/260224_090359)가 있을 때 돌고, 없으면 건너뛴다.
"""

import ast
import re
from argparse import Namespace
from pathlib import Path

import pytest
import torch

from Class_ import CombinedAE, CustomSigmoidFunc
from mcdl.preprocessing import get_input_dimensions
from mcdl.run import Run, RunConfig, build_model_pair, checkpoint_filename, write_sim_config

FIXTURES = Path(__file__).parent / "fixtures"
REAL_SIM_CONFIG = (FIXTURES / "sim_config_260224_090359.txt").read_text(encoding="utf-8")  # 실제 Run 설정 원문
REAL_RUN = Path(__file__).resolve().parents[1] / "models" / "260224_090359"  # 있으면 실제 Run 테스트에 쓴다

# 오래된 Run의 sim_config.txt에는 없을 수 있는 키. 없으면 예전 학습 때의 기본값으로 읽어야 한다.
LEGACY_KEYS = (
    "ae_u_scale",
    "ae_u_shift",
    "ae_x_scale",
    "ae_x_shift",
    "add_one_output_layer",
    "no_use_dx",
    "normalize_dx",
    "normalize_val",
    "no_abs_err",
    "vehicle_start",
    "vehicle_end",
)


def make_run(tmp_path, sim_config_text):
    """tmp_path(pytest가 테스트마다 주는 빈 임시 폴더)에 sim_config.txt와 빈 artifact/ 폴더만 있는 Run 폴더를 만든다."""
    run_dir = tmp_path / "run"
    (run_dir / "artifact").mkdir(parents=True)
    (run_dir / "sim_config.txt").write_text(sim_config_text, encoding="utf-8")
    return run_dir


def without_keys(text, keys):
    """sim_config 원문에서 keys에 해당하는 줄을 지운 글을 돌려준다(키가 없는 오래된 설정 흉내)."""
    return "\n".join(
        line for line in text.splitlines() if line.split(":", 1)[0].strip() not in keys
    )


def with_value(text, key, value):
    """sim_config 원문에서 key 줄의 값만 value로 바꾼 글을 돌려준다."""
    return re.sub(rf"(?m)^({key}\s*:\s*).*$", rf"\g<1>{value}", text)


def save_random_float64_weights(run_dir, model, number, config):
    """float32로 표현할 수 없는 값까지 담은 float64 가중치를 체크포인트 파일로 저장한다 (학습 결과와 같은 자료형).

    sigmoid가 포화되면 출력이 같아져 비교가 무의미하므로 0 근처의 작은 값을 쓴다.
    """
    u, x = build_model_pair(config)
    net = u if model == "U" else x
    generator = torch.Generator().manual_seed(number + 100)
    state = {
        k: (torch.rand(v.shape, dtype=torch.float64, generator=generator) - 0.5) * 0.02
        for k, v in net.state_dict().items()
    }
    torch.save(state, run_dir / "artifact" / checkpoint_filename(model, number))


def reference_load(run_dir, model, number, battery_type, scale, shift, use_dx, add_one_output_layer):
    """비교 기준이 되는 모델 불러오기: float64(double) 모델을 직접 만들고 저장된 가중치를 그대로 넣는다.

    학습 때와 같은 정밀도다. 예전 코드는 가중치를 float32로 잘라 불러왔는데, 그 버그를 고친 뒤의 올바른 방식이다
    (.scratch/known-bugs/issues/01-eval-weights-truncated-to-float32.md).
    """
    dims = get_input_dimensions(battery_type)
    keys = ("x", "q", "y") if model == "U" else ("x2", "q2", "y2")
    net = CombinedAE(
        input_size=dims[keys[0]],
        encode2_input_size=dims[keys[1]],
        output_size=dims[keys[2]],
        activation_fn=CustomSigmoidFunc(scale=scale, shift=shift),
        use_dx_in_forward=use_dx,
        add_one_output_layer=add_one_output_layer,
    ).double()
    stem = ("net" if model == "U" else "netx") + ("" if number == -1 else f"_e{number}")
    net.load_state_dict(torch.load(Path(run_dir) / "artifact" / f"{stem}.pth"))
    return net.eval()


def forward(net, model, battery_type="QAS", n=64):
    """고정 시드(7)의 무작위 입력 n개를 모델에 넣은 재구성 값. 두 모델이 같은 출력을 내는지 비교할 때 쓴다."""
    dims = get_input_dimensions(battery_type)
    keys = ("x", "z", "q") if model == "U" else ("x2", "z2", "q2")
    generator = torch.Generator().manual_seed(7)
    inputs = [torch.rand(n, dims[k], dtype=torch.float64, generator=generator) for k in keys]
    with torch.inference_mode():
        return net(*inputs)[0]


# ---------------------------------------------------------------- 설정


def test_real_sim_config_is_read_as_trained(tmp_path):
    """실제 Run의 설정 파일을 읽으면 학습 때 쓴 값(모델 출력 범위, Δ 사용 여부, 분할 폴더 등)이 그대로 나오는지 본다."""
    config = Run(make_run(tmp_path, REAL_SIM_CONFIG)).config

    assert config.battery_type == "QAS"
    assert config.learning_case == 1
    assert (config.ae_u_scale, config.ae_u_shift) == (1.8, 2.5)
    assert (config.ae_x_scale, config.ae_x_shift) == (1.0, 0)
    assert type(config.ae_x_shift) is int  # "0"은 정수로 읽힌다
    assert config.use_dx is False  # no_use_dx : True
    assert config.add_one_output_layer is True
    assert config.normalize_dx is False
    assert config.normalize_val == 4
    assert config.no_abs_err is False
    assert (config.vehicle_start, config.vehicle_end) == (0, -1)
    assert (config.preprocessing, config.skip_charge_ready) == (False, True)
    assert config.split_dir == "."  # dataset_name 키가 없는 Run은 루트 분할


def test_missing_keys_fall_back_to_legacy_defaults(tmp_path):
    """LEGACY_KEYS가 없는 오래된 설정은 예전 학습 때의 기본값으로 읽히는지 본다."""
    text = without_keys(REAL_SIM_CONFIG, LEGACY_KEYS)
    config = Run(make_run(tmp_path, text)).config

    assert (config.ae_u_scale, config.ae_u_shift) == (1.8, 2.5)
    assert (config.ae_x_scale, config.ae_x_shift) == (1.0, 0)
    assert config.use_dx is True
    assert config.add_one_output_layer is False
    assert config.normalize_dx is False
    assert config.normalize_val == 4
    assert config.no_abs_err is False
    assert (config.vehicle_start, config.vehicle_end) == (0, -1)


@pytest.mark.parametrize("key", ["battery_type", "learning_case"])
def test_missing_required_key_names_the_key_and_run(tmp_path, key):
    """꼭 있어야 하는 키(battery_type, learning_case)가 없으면, 빠진 키 이름과 파일 이름을 담은 ValueError가 나는지 본다."""
    run_dir = make_run(tmp_path, without_keys(REAL_SIM_CONFIG, (key,)))

    with pytest.raises(ValueError, match=key) as excinfo:
        Run(run_dir)
    assert "sim_config.txt" in str(excinfo.value)


def test_learning_case_derives_preprocessing_flags(tmp_path):
    """learning_case 값에서 전처리 여부(preprocessing, skip_charge_ready)가 계산되는지 본다."""
    text = with_value(REAL_SIM_CONFIG, "learning_case", 3)
    config = Run(make_run(tmp_path, text)).config

    assert (config.preprocessing, config.skip_charge_ready) == (True, False)


@pytest.mark.parametrize(
    "dataset_name, make_dir, expected",
    [
        ("10", "10", "./datasets/10"),
        ("05", "05", "./datasets/05"),  # 폴더 이름은 숫자로 바뀌지 않는다
        ("default", "default", "./datasets/default"),  # data_filter.py의 기본 이름도 보통 이름처럼 쓴다
        ("None", None, "."),  # 이름 없이 학습한 Run은 sim_config.txt에 None으로 기록된다
    ],
)
def test_split_dir_follows_dataset_name(tmp_path, monkeypatch, dataset_name, make_dir, expected):
    """dataset_name에 따라 분할 폴더가 정해지는지 본다. 이름이 None이면 루트("."), 그 밖에는 datasets/<이름>.

    monkeypatch.chdir는 테스트 동안만 작업 폴더를 tmp_path로 바꾼다(테스트가 끝나면 원래대로 돌아온다).
    """
    monkeypatch.chdir(tmp_path)
    if make_dir:
        (tmp_path / "datasets" / make_dir).mkdir(parents=True)
    text = f"battery_type : QAS\nlearning_case : 1\ndataset_name : {dataset_name}\n"

    assert Run(make_run(tmp_path, text)).config.split_dir == expected


def test_missing_split_folder_is_an_error(tmp_path, monkeypatch):
    """기록된 분할 폴더가 없으면 루트 분할로 넘어가지 않고, 폴더와 sim_config.txt 경로를 밝힌 오류로 멈추는지 본다.

    학습(Train_.py, RunConfig.from_mapping)과 평가(Run)가 같은 규칙을 쓰므로 둘 다 확인한다.
    (.scratch/known-bugs/issues/15-train-eval-split-dir-differ.md)
    """
    monkeypatch.chdir(tmp_path)
    text = "battery_type : QAS\nlearning_case : 1\ndataset_name : 11\n"

    with pytest.raises(FileNotFoundError, match="datasets/11") as excinfo:
        Run(make_run(tmp_path, text))
    assert "sim_config.txt" in str(excinfo.value)
    with pytest.raises(FileNotFoundError, match="datasets/11"):
        RunConfig.from_mapping({"battery_type": "QAS", "learning_case": 1, "dataset_name": "11"})


# ---------------------------------------------------------------- 체크포인트


def test_checkpoint_numbers_ascending_with_final_last(tmp_path):
    """체크포인트 번호를 작은 것부터(문자열 순서가 아니라 숫자 순서로) 나열하고, 최종 모델(-1)은 맨 뒤에 두는지 본다."""
    run_dir = make_run(tmp_path, REAL_SIM_CONFIG)
    for name in ["net.pth", "net_e10.pth", "net_e0.pth", "net_e2.pth", "netx.pth", "netx_e1.pth"]:
        (run_dir / "artifact" / name).write_bytes(b"")
    run = Run(run_dir)

    assert run.checkpoint_numbers("U") == [0, 2, 10, -1]
    assert run.checkpoint_numbers("X") == [1, -1]


def test_checkpoint_numbers_without_artifact_dir(tmp_path):
    """artifact/ 폴더가 없으면 오류 없이 빈 목록을 돌려주는지 본다."""
    run_dir = make_run(tmp_path, REAL_SIM_CONFIG)
    (run_dir / "artifact").rmdir()

    assert Run(run_dir).checkpoint_numbers("U") == []


def test_feature_cache_path_names_the_conditions(tmp_path):
    """특징 캐시 파일 이름에 계산 조건(모델, 체크포인트 번호, abs_err, float32, batch 크기)이 모두 들어가,
    조건이 하나라도 다르면 다른 파일이 되는지 본다."""
    run = Run(make_run(tmp_path, REAL_SIM_CONFIG))

    path = run.feature_cache_path("X", -1, abs_err=True, float32=False, batch_size=65536)

    assert path == run.path / "temp_cache" / "feat_x_-1_abs=1_float32=0_batch=65536.pkl"
    other_conditions = {
        run.feature_cache_path("X", -1, abs_err=False, float32=False, batch_size=65536),
        run.feature_cache_path("X", -1, abs_err=True, float32=True, batch_size=65536),
        run.feature_cache_path("X", -1, abs_err=True, float32=False, batch_size=4096),
        run.feature_cache_path("U", -1, abs_err=True, float32=False, batch_size=65536),
    }
    assert path not in other_conditions and len(other_conditions) == 4


def test_checkpoint_filename_round_trips_through_scan(tmp_path):
    """checkpoint_filename으로 만든 파일 이름을 checkpoint_numbers가 다시 같은 번호로 읽어 내는지 본다."""
    run_dir = make_run(tmp_path, REAL_SIM_CONFIG)
    for model in ("U", "X"):
        for number in (-1, 0, 7):
            (run_dir / "artifact" / checkpoint_filename(model, number)).write_bytes(b"")
    run = Run(run_dir)

    assert run.checkpoint_numbers("U") == [0, 7, -1]
    assert run.checkpoint_numbers("X") == [0, 7, -1]


# ---------------------------------------------------------------- 모델


@pytest.mark.parametrize("model, number", [("U", -1), ("U", 3), ("X", -1), ("X", 3)])
def test_load_model_matches_full_precision_reference(tmp_path, model, number):
    """Run.load_model로 불러온 모델이 기준 방식(reference_load)으로 불러온 모델과 완전히 같은 출력을 내는지 본다."""
    run_dir = make_run(tmp_path, REAL_SIM_CONFIG)
    run = Run(run_dir)
    save_random_float64_weights(run_dir, model, number, run.config)
    scale, shift = (1.8, 2.5) if model == "U" else (1.0, 0)

    loaded = run.load_model(model, number)
    reference = reference_load(run_dir, model, number, "QAS", scale, shift, False, True)

    out = forward(loaded, model)
    assert out.unique().numel() > 1  # 포화되지 않은 출력으로 비교한다
    assert torch.equal(out, forward(reference, model))


def test_load_model_keeps_saved_float64_weights(tmp_path):
    """저장된 float64 가중치가 float32로 잘리지 않고 그대로 쓰이고, 모델이 평가 모드인지 본다.

    예전 코드에 있던 정밀도 손실 버그(.scratch/known-bugs/issues/01-eval-weights-truncated-to-float32.md)가
    다시 생기지 않는지 지키는 테스트다.
    """
    run_dir = make_run(tmp_path, REAL_SIM_CONFIG)
    run = Run(run_dir)
    save_random_float64_weights(run_dir, "U", -1, run.config)
    saved = torch.load(run_dir / "artifact" / "net.pth")

    loaded = run.load_model("U", -1)

    assert not loaded.training
    for name, value in loaded.state_dict().items():
        assert value.dtype == torch.float64
        assert torch.equal(value, saved[name]), name


def test_load_model_pair_is_u_then_x(tmp_path):
    """load_model_pair(U 번호, X 번호)가 (U-model, X-model) 순서로, 각각 지정한 체크포인트를 불러오는지 본다."""
    run_dir = make_run(tmp_path, REAL_SIM_CONFIG)
    run = Run(run_dir)
    save_random_float64_weights(run_dir, "U", 2, run.config)
    save_random_float64_weights(run_dir, "X", 5, run.config)

    u, x = run.load_model_pair(2, 5)

    assert torch.equal(forward(u, "U"), forward(run.load_model("U", 2), "U"))
    assert torch.equal(forward(x, "X"), forward(run.load_model("X", 5), "X"))


def test_build_model_pair_shapes_follow_config(tmp_path):
    """새로 만든 모델 쌍의 층 크기, 추가 출력층(fc4) 여부, 출력 범위(scale·shift)가 설정을 따르는지 본다."""
    config = Run(make_run(tmp_path, REAL_SIM_CONFIG)).config
    u, x = build_model_pair(config)

    assert u.fc3.out_features == 110 and u.fc2.in_features == 3
    assert x.fc3.out_features == 110 and x.fc2.in_features == 4
    assert hasattr(u, "fc4") and hasattr(x, "fc4")  # add_one_output_layer : True
    assert (u.activation_fn.scale, u.activation_fn.shift) == (1.8, 2.5)
    assert (x.activation_fn.scale, x.activation_fn.shift) == (1.0, 0)


def test_build_model_pair_uses_unit_sigmoid_with_preprocessing(tmp_path):
    """0~1 정규화를 쓰는 learning_case(여기서는 2)에서는 두 모델 모두 출력 범위가 0~1(scale 1, shift 0)인지 본다."""
    text = with_value(REAL_SIM_CONFIG, "learning_case", 2)
    u, x = build_model_pair(Run(make_run(tmp_path, text)).config)

    assert (u.activation_fn.scale, u.activation_fn.shift) == (1, 0)
    assert (x.activation_fn.scale, x.activation_fn.shift) == (1, 0)


def test_build_model_pair_consumes_rng_u_then_x(tmp_path):
    """같은 시드에서 U-model을 먼저, X-model을 나중에 만들어, 예전 코드와 같은 초기 가중치가 나오는지 본다.

    난수 발생기(RNG)는 꺼내 쓸 때마다 다음 값으로 넘어가므로, 만드는 순서가 바뀌면 초기 가중치도 바뀐다.
    """
    config = Run(make_run(tmp_path, REAL_SIM_CONFIG)).config
    dims = get_input_dimensions("QAS")

    torch.manual_seed(0)
    u, x = build_model_pair(config)
    torch.manual_seed(0)
    u_ref = CombinedAE(dims["x"], dims["q"], dims["y"], False, True, CustomSigmoidFunc(1.8, 2.5))
    x_ref = CombinedAE(dims["x2"], dims["q2"], dims["y2"], False, True, CustomSigmoidFunc(1.0, 0))

    for net, ref in ((u, u_ref), (x, x_ref)):
        for k, v in ref.state_dict().items():
            assert torch.equal(net.state_dict()[k], v)


# ---------------------------------------------------------------- sim_config 쓰기

SYSTEM_KEYS = ("timestamp", "python", "platform", "cwd")  # 실행 환경마다 달라져 비교에서 빼는 키
EXTRA_KEYS = (  # 명령줄 옵션이 아니라 Train_.py가 extra로 덧붙여 쓰는 키
    "device",
    "Vehicle IDs used for training",
    "Amount of data used for training",
    "PREPROCESSING",
    "SKIP_CHARGE_READY",
)


def train_args_and_extra_from_fixture():
    """실제 Run의 sim_config 원문에서, 학습 때 Train_.py가 write_sim_config에 넘겼던 (명령줄 옵션 args, extra)를 되살린다."""
    values = {}
    for line in REAL_SIM_CONFIG.splitlines():
        if " : " in line:
            key, raw = line.split(" : ", 1)
            try:
                values[key.strip()] = ast.literal_eval(raw.strip())
            except (ValueError, SyntaxError):
                values[key.strip()] = raw.strip()

    # 차량 ID 목록은 180자에서 잘려 저장되어 있으므로, 보이는 ID 뒤에 아무 ID나 덧붙인다
    shown_ids = [int(n) for n in re.findall(r"\d+", values["Vehicle IDs used for training"])]
    values["Vehicle IDs used for training"] = shown_ids + list(range(1000, 1050))

    extra = {k: values.pop(k) for k in EXTRA_KEYS}
    for k in SYSTEM_KEYS:
        values.pop(k)
    return Namespace(**values), extra


def mask_system_lines(text):
    """실행 환경마다 달라지는 줄(SYSTEM_KEYS)의 값을 <system>으로 가려, 나머지 내용만 비교할 수 있게 한다."""
    return re.sub(rf"(?m)^({'|'.join(SYSTEM_KEYS)})(\s*:).*$", r"\1\2 <system>", text)


def test_write_sim_config_keeps_current_format(tmp_path):
    """같은 입력으로 sim_config.txt를 쓰면, 환경 관련 줄을 빼고 실제 Run의 파일과 글자 하나까지 같은지 본다."""
    args, extra = train_args_and_extra_from_fixture()

    text = write_sim_config(tmp_path, args, extra)

    written = (tmp_path / "sim_config.txt").read_text(encoding="utf-8")
    assert written == text
    assert mask_system_lines(written) == mask_system_lines(REAL_SIM_CONFIG)


def test_written_sim_config_reads_back_as_same_config(tmp_path):
    """쓴 sim_config.txt를 다시 읽으면, 쓸 때 넘긴 옵션과 같은 설정(RunConfig)이 되는지 본다."""
    args, extra = train_args_and_extra_from_fixture()

    write_sim_config(tmp_path, args, extra)

    assert Run(tmp_path).config == RunConfig.from_mapping(vars(args))


def test_round_trip_with_dataset_and_preprocessing(tmp_path, monkeypatch):
    """분할 이름·learning_case·Δ 정규화 같은 옵션을 바꿔 써도, 다시 읽으면 같은 설정이 되는지 본다."""
    monkeypatch.chdir(tmp_path)
    (tmp_path / "datasets" / "10").mkdir(parents=True)
    args, extra = train_args_and_extra_from_fixture()
    args.dataset_name = "10"
    args.learning_case = 2
    args.normalize_dx = True
    args.no_use_dx = False
    run_dir = tmp_path / "run"

    write_sim_config(run_dir, args, extra)
    config = Run(run_dir).config

    assert config == RunConfig.from_mapping(vars(args))
    assert config.split_dir == "./datasets/10"
    assert (config.preprocessing, config.normalize_dx, config.use_dx) == (True, True, True)


# ---------------------------------------------------------------- 실제 Run


@pytest.mark.realdata
@pytest.mark.skipif(not REAL_RUN.exists(), reason="models/260224_090359 없음")
def test_real_run_loads_saved_weights_exactly():
    """실제 Run에서 체크포인트 목록을 찾고, 저장된 가중치를 손실 없이 불러와 기준 방식과 같은 출력을 내는지 본다.

    @pytest.mark.realdata는 실제 데이터가 필요한 테스트라는 표시이고(pytest.ini에 등록), skipif는 Run 폴더가
    없을 때 이 테스트를 건너뛰게 한다.
    """
    run = Run(REAL_RUN)

    assert run.checkpoint_numbers("U") == list(range(13)) + [-1]
    for model, scale, shift in (("U", 1.8, 2.5), ("X", 1.0, 0)):
        for number in (-1, 0):
            loaded = run.load_model(model, number)
            saved = torch.load(REAL_RUN / "artifact" / checkpoint_filename(model, number))
            for name, value in loaded.state_dict().items():
                assert torch.equal(value, saved[name]), (model, number, name)
            reference = reference_load(REAL_RUN, model, number, "QAS", scale, shift, False, True)
            out = forward(loaded, model)
            assert out.unique().numel() > 1
            assert torch.equal(out, forward(reference, model))
