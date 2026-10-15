"""Run module 테스트. 원본 데이터 없이 CPU에서 돈다."""

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
REAL_SIM_CONFIG = (FIXTURES / "sim_config_260224_090359.txt").read_text(encoding="utf-8")
REAL_RUN = Path(__file__).resolve().parents[1] / "models" / "260224_090359"

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
    run_dir = tmp_path / "run"
    (run_dir / "artifact").mkdir(parents=True)
    (run_dir / "sim_config.txt").write_text(sim_config_text, encoding="utf-8")
    return run_dir


def without_keys(text, keys):
    return "\n".join(
        line for line in text.splitlines() if line.split(":", 1)[0].strip() not in keys
    )


def with_value(text, key, value):
    return re.sub(rf"(?m)^({key}\s*:\s*).*$", rf"\g<1>{value}", text)


def save_random_float64_weights(run_dir, model, number, config):
    """float32로 표현할 수 없는 값까지 담은 float64 가중치를 저장한다 (학습 산출물과 같은 dtype).

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
    """학습과 같은 정밀도의 참조 로드: double 모델 생성 → load_state_dict (known-bugs 01 수정 후)."""
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
    dims = get_input_dimensions(battery_type)
    keys = ("x", "z", "q") if model == "U" else ("x2", "z2", "q2")
    generator = torch.Generator().manual_seed(7)
    inputs = [torch.rand(n, dims[k], dtype=torch.float64, generator=generator) for k in keys]
    with torch.inference_mode():
        return net(*inputs)[0]


# ---------------------------------------------------------------- 설정


def test_real_sim_config_is_read_as_trained(tmp_path):
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
    run_dir = make_run(tmp_path, without_keys(REAL_SIM_CONFIG, (key,)))

    with pytest.raises(ValueError, match=key) as excinfo:
        Run(run_dir)
    assert "sim_config.txt" in str(excinfo.value)


def test_learning_case_derives_preprocessing_flags(tmp_path):
    text = with_value(REAL_SIM_CONFIG, "learning_case", 3)
    config = Run(make_run(tmp_path, text)).config

    assert (config.preprocessing, config.skip_charge_ready) == (True, False)


@pytest.mark.parametrize(
    "dataset_name, make_dir, expected",
    [
        ("10", "10", "./datasets/10"),
        ("05", "05", "./datasets/05"),  # 폴더 이름은 숫자로 바뀌지 않는다
        ("10", None, "."),  # 폴더가 없으면 루트 분할
        ("None", None, "."),
        ("default", "default", "."),
    ],
)
def test_split_dir_follows_dataset_name(tmp_path, monkeypatch, dataset_name, make_dir, expected):
    monkeypatch.chdir(tmp_path)
    if make_dir:
        (tmp_path / "datasets" / make_dir).mkdir(parents=True)
    text = f"battery_type : QAS\nlearning_case : 1\ndataset_name : {dataset_name}\n"

    assert Run(make_run(tmp_path, text)).config.split_dir == expected


# ---------------------------------------------------------------- 체크포인트


def test_checkpoint_numbers_ascending_with_final_last(tmp_path):
    run_dir = make_run(tmp_path, REAL_SIM_CONFIG)
    for name in ["net.pth", "net_e10.pth", "net_e0.pth", "net_e2.pth", "netx.pth", "netx_e1.pth"]:
        (run_dir / "artifact" / name).write_bytes(b"")
    run = Run(run_dir)

    assert run.checkpoint_numbers("U") == [0, 2, 10, -1]
    assert run.checkpoint_numbers("X") == [1, -1]


def test_checkpoint_numbers_without_artifact_dir(tmp_path):
    run_dir = make_run(tmp_path, REAL_SIM_CONFIG)
    (run_dir / "artifact").rmdir()

    assert Run(run_dir).checkpoint_numbers("U") == []


def test_feature_cache_path_names_the_conditions(tmp_path):
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
    """known-bugs 01 수정: 저장된 float64 가중치가 float32로 잘리지 않고 그대로 쓰인다."""
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
    run_dir = make_run(tmp_path, REAL_SIM_CONFIG)
    run = Run(run_dir)
    save_random_float64_weights(run_dir, "U", 2, run.config)
    save_random_float64_weights(run_dir, "X", 5, run.config)

    u, x = run.load_model_pair(2, 5)

    assert torch.equal(forward(u, "U"), forward(run.load_model("U", 2), "U"))
    assert torch.equal(forward(x, "X"), forward(run.load_model("X", 5), "X"))


def test_build_model_pair_shapes_follow_config(tmp_path):
    config = Run(make_run(tmp_path, REAL_SIM_CONFIG)).config
    u, x = build_model_pair(config)

    assert u.fc3.out_features == 110 and u.fc2.in_features == 3
    assert x.fc3.out_features == 110 and x.fc2.in_features == 4
    assert hasattr(u, "fc4") and hasattr(x, "fc4")  # add_one_output_layer : True
    assert (u.activation_fn.scale, u.activation_fn.shift) == (1.8, 2.5)
    assert (x.activation_fn.scale, x.activation_fn.shift) == (1.0, 0)


def test_build_model_pair_uses_unit_sigmoid_with_preprocessing(tmp_path):
    text = with_value(REAL_SIM_CONFIG, "learning_case", 2)
    u, x = build_model_pair(Run(make_run(tmp_path, text)).config)

    assert (u.activation_fn.scale, u.activation_fn.shift) == (1, 0)
    assert (x.activation_fn.scale, x.activation_fn.shift) == (1, 0)


def test_build_model_pair_consumes_rng_u_then_x(tmp_path):
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

SYSTEM_KEYS = ("timestamp", "python", "platform", "cwd")
EXTRA_KEYS = (
    "device",
    "Vehicle IDs used for training",
    "Amount of data used for training",
    "PREPROCESSING",
    "SKIP_CHARGE_READY",
)


def train_args_and_extra_from_fixture():
    """실제 Run의 sim_config에서 Train_이 넘겼던 args와 extra를 되살린다."""
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
    return re.sub(rf"(?m)^({'|'.join(SYSTEM_KEYS)})(\s*:).*$", r"\1\2 <system>", text)


def test_write_sim_config_keeps_current_format(tmp_path):
    args, extra = train_args_and_extra_from_fixture()

    text = write_sim_config(tmp_path, args, extra)

    written = (tmp_path / "sim_config.txt").read_text(encoding="utf-8")
    assert written == text
    assert mask_system_lines(written) == mask_system_lines(REAL_SIM_CONFIG)


def test_written_sim_config_reads_back_as_same_config(tmp_path):
    args, extra = train_args_and_extra_from_fixture()

    write_sim_config(tmp_path, args, extra)

    assert Run(tmp_path).config == RunConfig.from_mapping(vars(args))


def test_round_trip_with_dataset_and_preprocessing(tmp_path, monkeypatch):
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
