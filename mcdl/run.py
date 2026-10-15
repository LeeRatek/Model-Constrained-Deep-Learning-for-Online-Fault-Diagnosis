"""Run: 한 번의 학습 실행과 그 결과 폴더 models/<RUN_ID>/.

평가 스크립트는 Run을 열어 설정, 체크포인트 번호, 모델만 받는다. 학습 스크립트는 같은 module로
sim_config.txt를 쓰고, 체크포인트 파일 이름을 정하고, 모델 쌍을 만든다. sim_config.txt 형식,
키가 없을 때의 레거시 기본값, learning_case 파생값, 분할 폴더 경로, 체크포인트 파일 규칙,
U-model·X-model 생성과 로드는 모두 이 module 안에 있다.
"""

from __future__ import annotations

import ast
import io
import os
import platform
import re
import sys
from contextlib import redirect_stdout
from dataclasses import asdict, dataclass, is_dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch

from Class_ import CombinedAE, CustomSigmoidFunc
from mcdl.preprocessing import get_input_dimensions, get_preprocessing_and_skip_charge_ready

REQUIRED_KEYS = ("battery_type", "learning_case")

# sim_config.txt에 키가 없는 예전 Run에 쓰는 값
LEGACY_DEFAULTS = {
    "ae_u_scale": 1.8,
    "ae_u_shift": 2.5,
    "ae_x_scale": 1.0,
    "ae_x_shift": 0,
    "add_one_output_layer": False,
    "no_use_dx": False,
    "normalize_dx": False,
    "normalize_val": 4,
    "no_abs_err": False,
    "vehicle_start": 0,
    "vehicle_end": -1,
}

_FILE_PREFIX = {"U": "net", "X": "netx"}


@dataclass(frozen=True)
class RunConfig:
    """평가에 필요한 Run 설정. 필드 이름은 sim_config 키 이름을 따른다 (파생 필드 제외)."""

    battery_type: str
    learning_case: int
    ae_u_scale: float
    ae_u_shift: float
    ae_x_scale: float
    ae_x_shift: float
    use_dx: bool
    add_one_output_layer: bool
    normalize_dx: bool
    normalize_val: float
    no_abs_err: bool
    vehicle_start: int
    vehicle_end: int
    preprocessing: bool  # learning_case에서 파생
    skip_charge_ready: bool  # learning_case에서 파생
    split_dir: str  # "./datasets/<name>" 또는 "." (루트 분할)

    @classmethod
    def from_mapping(cls, values) -> "RunConfig":
        """sim_config 값(또는 학습 인자)에서 설정을 만든다. 레거시 기본값과 파생값은 여기서만 채운다."""
        missing = [k for k in REQUIRED_KEYS if values.get(k) is None]
        if missing:
            raise ValueError(f"Run 설정에 필수 키가 없습니다: {missing}")

        v = {k: values.get(k, default) for k, default in LEGACY_DEFAULTS.items()}
        preprocessing, skip_charge_ready = get_preprocessing_and_skip_charge_ready(
            values["learning_case"]
        )
        return cls(
            battery_type=values["battery_type"],
            learning_case=values["learning_case"],
            ae_u_scale=v["ae_u_scale"],
            ae_u_shift=v["ae_u_shift"],
            ae_x_scale=v["ae_x_scale"],
            ae_x_shift=v["ae_x_shift"],
            use_dx=not v["no_use_dx"],
            add_one_output_layer=v["add_one_output_layer"],
            normalize_dx=v["normalize_dx"],
            normalize_val=v["normalize_val"],
            no_abs_err=v["no_abs_err"],
            vehicle_start=v["vehicle_start"],
            vehicle_end=v["vehicle_end"],
            preprocessing=preprocessing,
            skip_charge_ready=skip_charge_ready,
            split_dir=_split_dir(values.get("dataset_name")),
        )


class Run:
    """models/<RUN_ID>/ 폴더 하나."""

    def __init__(self, path):
        self.path = Path(path)
        sim_path = self.path / "sim_config.txt"
        raw = _parse_sim_config(sim_path.read_text(encoding="utf-8"))
        # dataset_name은 폴더 이름이므로 변환하지 않는다 (예: "05"가 5가 되지 않게)
        values = {
            k: v if k == "dataset_name" else _coerce_value(v) for k, v in raw.items()
        }
        try:
            self.config = RunConfig.from_mapping(values)
        except ValueError as e:
            raise ValueError(f"{sim_path}: {e}") from e

    def checkpoint_numbers(self, model: str) -> list[int]:
        """artifact 폴더에 있는 체크포인트 번호. 오름차순이고 최종 모델(-1)은 맨 뒤에 둔다."""
        artifact_dir = self.path / "artifact"
        if not artifact_dir.exists():
            print(f"Warning: Artifact directory not found: {artifact_dir}")
            return []

        pattern = re.compile(rf"{_FILE_PREFIX[model]}(?:_e(\d+))?\.pth")
        numbers = set()
        for filename in os.listdir(artifact_dir):
            match = pattern.match(filename)
            if match:
                numbers.add(int(match.group(1)) if match.group(1) else -1)

        ordered = sorted(numbers - {-1})
        if -1 in numbers:
            ordered.append(-1)
        return ordered

    def load_model(self, model: str, number: int):
        """체크포인트 번호의 U-model 또는 X-model을 CPU·double·eval 상태로 돌려준다.

        학습 때와 같은 double 정밀도로 가중치를 load한다 (저장된 float64 값을 그대로 쓴다).
        """
        net = _build_model(self.config, model).double()
        state = torch.load(
            self.path / "artifact" / checkpoint_filename(model, number),
            map_location="cpu",
        )
        net.load_state_dict(state)
        return net.eval()

    def load_model_pair(self, u: int, x: int):
        """모델 쌍 (U-model, X-model)."""
        return self.load_model("U", u), self.load_model("X", x)

    def feature_cache_path(
        self, model: str, number: int, *, abs_err: bool, float32: bool, batch_size: int
    ) -> Path:
        """Test_multi_models가 temp_cache에 남기는 학습 특징 캐시 파일 경로.

        학습 특징을 계산한 조건(오차 abs 여부, float32 변환, 추론 batch 크기)이 이름에 들어가므로,
        조건이 다른 캐시를 섞어 쓰지 않는다.
        """
        prefix = {"U": "u", "X": "x"}[model]
        return (
            self.path
            / "temp_cache"
            / f"feat_{prefix}_{number}_abs={int(abs_err)}_float32={int(float32)}_batch={batch_size}.pkl"
        )


def build_model_pair(config: RunConfig):
    """설정에 맞는 새 (U-model, X-model). U를 먼저 만든다 (학습의 RNG 소비 순서)."""
    return _build_model(config, "U"), _build_model(config, "X")


def checkpoint_filename(model: str, number: int) -> str:
    """체크포인트 번호의 파일 이름. -1은 최종 모델."""
    prefix = _FILE_PREFIX[model]
    return f"{prefix}.pth" if number == -1 else f"{prefix}_e{number}.pth"


def write_sim_config(path, args, extra, exclude: Sequence[str] | None = None) -> str:
    """학습 인자와 추가 값을 path/sim_config.txt에 쓰고, 쓴 텍스트를 돌려준다. exclude 키는 쓰지 않는다."""
    buf = io.StringIO()
    with redirect_stdout(buf):
        print_sim_config(title="Train Run", config=args, extra=extra, exclude=exclude)
    text = buf.getvalue()
    os.makedirs(path, exist_ok=True)
    with open(f"{path}/sim_config.txt", "w", encoding="utf-8") as f:
        f.write(text)
    return text


def save_net_state(model, models_dir="./models", filename="net_model.pth"):
    """
    net(CombinedAE)의 state_dict를 CPU 텐서로 저장합니다.
    """
    os.makedirs(models_dir, exist_ok=True)
    path = os.path.join(models_dir, filename)
    state = {k: v.detach().cpu() for k, v in model.state_dict().items()}
    torch.save(state, path)


def make_model_path_based_timestamp(
    base="models", make: bool = False, prefix: str = None, tz="Asia/Seoul"
):
    # tzinfo 설정: zoneinfo 우선, 실패 시 pytz로 폴백
    try:
        from zoneinfo import ZoneInfo  # Python 3.9+

        tzinfo = ZoneInfo(tz)
    except Exception:
        import pytz  # pip install pytz 필요

        tzinfo = pytz.timezone(tz)

    now = datetime.now(tzinfo)
    timestamp = now.strftime("26%m%d_%H%M%S")
    if prefix is None:
        path = os.path.join(base, timestamp)
    else:
        path = os.path.join(base, f"{prefix}_{timestamp}")
    if make:
        os.makedirs(path, exist_ok=True)
    return path


def print_sim_config(
    title: str = "Simulation Settings",
    config: Any = None,
    include: Sequence[str] | None = None,
    exclude: Sequence[str] | None = None,
    extra: Mapping[str, Any] | None = None,
    sort_keys: bool = True,
    show_system: bool = True,
) -> None:
    """
    시뮬레이션 세팅(실험 설정)을 깔끔하게 출력.

    - config: argparse.Namespace, dict, dataclass, 또는 일반 객체(속성 기반)
    - include: 출력할 key들만 선택 (None이면 전체)
    - exclude: 제외할 key들
    - extra: 추가로 더 붙이고 싶은 값들(예: device, seed, git hash 등)
    """
    cfg = _as_mapping(config)

    if include is not None:
        cfg = {k: cfg.get(k, None) for k in include}

    if exclude is not None:
        for k in exclude:
            cfg.pop(k, None)

    if extra:
        # extra가 우선권(override)
        cfg.update(dict(extra))

    items = list(cfg.items())
    if sort_keys:
        items.sort(key=lambda kv: kv[0])

    # 시스템 정보(선택)
    sys_lines = []
    if show_system:
        sys_lines = [
            ("timestamp", datetime.now().strftime("%Y-%m-%d %H:%M:%S")),
            ("python", sys.version.split()[0]),
            (
                "platform",
                f"{platform.system()} {platform.release()} ({platform.machine()})",
            ),
            ("cwd", os.getcwd()),
        ]

    # 출력 폭 계산
    all_items = sys_lines + items
    key_w = max([len(str(k)) for k, _ in all_items] + [3])
    val_w = 0
    for _, v in all_items:
        val_w = max(val_w, len(_fmt(v)))

    line_w = min(max(key_w + 3 + val_w + 2, 60), 140)

    def hr(ch: str = "=") -> str:
        return ch * line_w

    print(hr("="))
    print(title.center(line_w))
    print(hr("="))

    if sys_lines:
        for k, v in sys_lines:
            print(f"{str(k):<{key_w}} : {_fmt(v)}")
        print(hr("-"))

    for k, v in items:
        print(f"{str(k):<{key_w}} : {_fmt(v)}")

    print(hr("="))


def _as_mapping(config: Any) -> dict:
    """Namespace/dict/dataclass/object 모두 dict 비슷하게 변환."""
    if config is None:
        return {}
    if isinstance(config, Mapping):
        return dict(config)
    if is_dataclass(config):
        return asdict(config)
    if hasattr(config, "__dict__"):
        return dict(vars(config))
    return {"config": config}


def _fmt(v: Any, max_len: int = 180) -> str:
    if v is None:
        s = "None"
    elif isinstance(v, (list, tuple, set)):
        s = "[" + ", ".join(map(str, list(v))) + "]"
    else:
        s = str(v)

    s = s.replace("\n", "\\n")
    if len(s) > max_len:
        s = s[: max_len - 3] + "..."
    return s


def _build_model(config: RunConfig, model: str):
    dims = get_input_dimensions(config.battery_type)
    if model == "U":
        x, q, y = dims["x"], dims["q"], dims["y"]
        scale, shift = config.ae_u_scale, config.ae_u_shift
    else:
        x, q, y = dims["x2"], dims["q2"], dims["y2"]
        scale, shift = config.ae_x_scale, config.ae_x_shift
    # 0~1 정규화를 쓰는 learning_case에서는 출력 범위도 0~1이다
    if config.preprocessing:
        scale, shift = 1, 0
    return CombinedAE(
        input_size=x,
        encode2_input_size=q,
        output_size=y,
        activation_fn=CustomSigmoidFunc(scale=scale, shift=shift),
        use_dx_in_forward=config.use_dx,
        add_one_output_layer=config.add_one_output_layer,
    )


def _split_dir(dataset_name) -> str:
    if dataset_name is None:
        return "."
    name = str(dataset_name).strip()
    if name and name.lower() != "none" and name != "default":
        candidate = f"./datasets/{name}"
        if os.path.isdir(candidate):
            return candidate
        print(
            f"[Warning] Dataset folder '{candidate}' recorded in config not found. Falling back to root."
        )
    return "."


def _parse_sim_config(text: str) -> dict:
    """sim_config.txt의 "key : value" 줄을 {key: 값 원문}으로 읽는다."""
    parsed = {}
    for line in text.splitlines():
        if ":" not in line:
            continue
        # 구분선(====, ----) 제외
        stripped = line.strip()
        if not stripped or set(stripped) <= {"=", "-"}:
            continue

        k, v = line.split(":", 1)
        k = k.strip()
        if k:
            parsed[k] = v.strip()
    return parsed


def _coerce_value(raw: str):
    """bool, 정수, 지수 표기 없는 실수, 리터럴만 변환한다. "None"·"1e-05" 등은 문자열 그대로 둔다."""
    s = str(raw).strip()
    if s == "":
        return ""

    sl = s.lower()
    if sl in ("true", "false"):
        return sl == "true"

    if re.fullmatch(r"-?\d+", s):
        return int(s)

    if re.fullmatch(r"-?\d+(?:\.\d+)?", s):
        return float(s)

    if (
        (s.startswith("[") and s.endswith("]"))
        or (s.startswith("(") and s.endswith(")"))
        or (s.startswith("{") and s.endswith("}"))
    ):
        try:
            return ast.literal_eval(s)
        except Exception:
            return s

    return s
