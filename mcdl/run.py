"""학습 결과 폴더(Run)를 읽고 쓰는 코드.

Run이란
    Train_.py를 한 번 실행하면 models/<RUN_ID>/ 폴더가 하나 생긴다(RUN_ID는 시작 시각, 예: 260224_090359).
    이 폴더를 "Run"이라고 부른다. 안에는 학습 설정 기록(sim_config.txt), 모델 가중치 파일(artifact/*.pth,
    "체크포인트"), PCA 관리도 파일(pca_arrays.npz 등)이 들어 있다.

누가 쓰나
    - 평가 스크립트(Test_.py, Test_multi_models.py, Test_tsne.py): Run(...)으로 폴더를 열어 설정과 모델을 받는다.
    - 학습 스크립트(Train_.py): 새 Run 폴더 이름을 정하고, 설정 기록을 쓰고, 새 모델을 만들고, 가중치를 저장한다.

이 파일이 정하는 규칙
    sim_config.txt의 형식, 예전 Run에 없는 설정 항목의 기본값, 체크포인트 파일 이름(net_e3.pth 등),
    U-model·X-model을 만드는 방법이 모두 여기에 있다. 규칙을 바꿀 때는 이 파일만 고치면 된다.
"""

from __future__ import annotations

import ast
import io
import os
import platform
import re
import sys
from contextlib import redirect_stdout
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Mapping, Sequence
from zoneinfo import ZoneInfo

import torch

from Class_ import CombinedAE, CustomSigmoidFunc
from mcdl.preprocessing import get_input_dimensions, get_preprocessing_and_skip_charge_ready

# sim_config.txt에 반드시 있어야 하는 항목. 없으면 어떤 데이터·모델로 학습했는지 알 수 없다.
REQUIRED_KEYS = ("battery_type", "learning_case")

# 예전에 학습한 Run의 sim_config.txt에는 나중에 생긴 옵션이 기록되어 있지 않다.
# 그런 Run을 읽을 때는 그 옵션이 생기기 전의 동작과 같은 아래 값으로 채운다.
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

# 체크포인트 파일 이름 앞부분. U-model은 net*.pth, X-model은 netx*.pth로 저장된다.
_FILE_PREFIX = {"U": "net", "X": "netx"}


@dataclass(frozen=True)
class RunConfig:
    """평가와 학습에 필요한 Run 설정 묶음 (한 번 만들면 바꿀 수 없는 값 객체).

    sim_config.txt를 읽어 만들거나(Run.__init__), 학습할 때 명령줄 인자로 만든다(Train_.py main).
    필드 이름은 sim_config.txt의 항목 이름과 같다. 마지막 세 필드는 다른 값에서 계산한 것이다.
    """

    battery_type: str  # 배터리 종류: "QAS" 또는 "DTI"
    learning_case: int  # 전처리 조합 번호 1~4 (README 5장)
    ae_u_scale: float  # U-model 출력 범위: shift ~ shift + scale
    ae_u_shift: float
    ae_x_scale: float  # X-model 출력 범위
    ae_x_shift: float
    use_dx: bool  # 셀 편차 추정값(Δ)을 모델 입력으로 쓰는지
    add_one_output_layer: bool  # 디코더 앞에 선형층을 하나 더 두는지
    normalize_dx: bool  # Δ를 normalize_val로 나누는지
    normalize_val: float
    no_abs_err: bool  # True면 재구성 오차를 절댓값 대신 부호 있는 값으로 쓴다
    vehicle_start: int  # 학습에 쓴 train 분할 구간 (끝 포함, -1이면 끝까지)
    vehicle_end: int
    preprocessing: bool  # learning_case에서 계산: 0~1 정규화를 하는지
    skip_charge_ready: bool  # learning_case에서 계산: 충전 대기 구간을 잘라내는지
    split_dir: str  # 분할 파일(.npy)이 있는 폴더: "./datasets/<이름>" 또는 저장소 루트 "."

    @classmethod
    def from_mapping(cls, values) -> "RunConfig":
        """{항목 이름: 값} 묶음에서 설정을 만든다.

        쓰는 곳: Run.__init__(sim_config.txt에서 읽은 값), Train_.py main(명령줄 인자 vars(args)).
        없는 항목은 LEGACY_DEFAULTS로 채우고, preprocessing·skip_charge_ready·split_dir는 여기서 계산한다.
        battery_type이나 learning_case가 없으면 ValueError를 낸다.
        """
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
    """학습 결과 폴더 models/<RUN_ID>/ 하나를 여는 객체.

    쓰는 곳: Test_.py, Test_multi_models.py, Test_tsne.py, mcdl/chart_source.py.
    예: run = Run("models/260224_090359"); run.config.battery_type; net, netx = run.load_model_pair(-1, -1)
    """

    def __init__(self, path):
        """폴더의 sim_config.txt를 읽어 self.config(RunConfig)를 만든다. 필수 항목이 없으면 파일 경로와 함께 ValueError."""
        self.path = Path(path)
        sim_path = self.path / "sim_config.txt"
        raw = _parse_sim_config(sim_path.read_text(encoding="utf-8"))
        # dataset_name은 폴더 이름이므로 숫자로 바꾸지 않는다 (예: "05"가 5가 되지 않게)
        values = {
            k: v if k == "dataset_name" else _coerce_value(v) for k, v in raw.items()
        }
        try:
            self.config = RunConfig.from_mapping(values)
        except ValueError as e:
            raise ValueError(f"{sim_path}: {e}") from e

    def checkpoint_numbers(self, model: str) -> list[int]:
        """artifact 폴더에 저장된 체크포인트 번호 목록. model은 "U" 또는 "X".

        파일 이름 net_e3.pth → 3, net.pth(학습이 끝난 최종 모델) → -1로 읽는다. 오름차순이고 -1은 맨 뒤에 둔다.
        쓰는 곳: Test_multi_models.py (`--u-model-idx auto`일 때 평가할 체크포인트를 찾는다).
        """
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
        """체크포인트 하나를 불러 평가용 모델로 돌려준다. model은 "U"/"X", number는 체크포인트 번호(-1은 최종).

        모델은 CPU에 두고, 학습 때와 같은 float64 정밀도로 만든 뒤 가중치를 넣고, 평가 모드(eval)로 바꾼다.
        쓰는 곳: Test_multi_models.py, 그리고 아래 load_model_pair.
        """
        net = _build_model(self.config, model).double()
        state = torch.load(
            self.path / "artifact" / checkpoint_filename(model, number),
            map_location="cpu",
        )
        net.load_state_dict(state)
        return net.eval()

    def load_model_pair(self, u: int, x: int):
        """함께 평가할 (U-model, X-model)을 한 번에 불러온다. u, x는 각 체크포인트 번호.

        쓰는 곳: Test_.py, Test_tsne.py, mcdl/chart_source.py(관리도를 다시 계산할 때).
        """
        return self.load_model("U", u), self.load_model("X", x)

    def feature_cache_path(
        self, model: str, number: int, *, abs_err: bool, float32: bool, batch_size: int
    ) -> Path:
        """체크포인트 하나의 학습 진단 특징을 저장해 두는 캐시 파일 경로를 돌려준다.

        학습 진단 특징(정상 학습 차량들의 진단 특징)은 계산에 몇 분이 걸려서, Test_multi_models.py가
        Run 폴더 안 temp_cache/에 파일로 저장해 두고 다시 쓴다. 계산 조건(오차 절댓값 여부, float32 변환,
        추론 batch 크기)이 결과에 영향을 주므로 파일 이름에 넣어, 조건이 다른 캐시를 섞어 쓰지 않게 한다.
        예: temp_cache/feat_u_3_abs=1_float32=0_batch=65536.pkl
        쓰는 곳: Test_multi_models.py, mcdl/chart_source.py.
        """
        prefix = {"U": "u", "X": "x"}[model]
        return (
            self.path
            / "temp_cache"
            / f"feat_{prefix}_{number}_abs={int(abs_err)}_float32={int(float32)}_batch={batch_size}.pkl"
        )


def build_model_pair(config: RunConfig):
    """설정에 맞는 새(학습 전) U-model과 X-model을 만든다.

    쓰는 곳: Train_.py (학습 시작 전). 가중치 초기값이 난수로 정해지므로, 항상 U-model을 먼저 만들어
    같은 시드에서 같은 초기값이 나오게 한다.
    """
    return _build_model(config, "U"), _build_model(config, "X")


def checkpoint_filename(model: str, number: int) -> str:
    """체크포인트 파일 이름을 정한다. 예: ("U", 3) → "net_e3.pth", ("X", -1) → "netx.pth" (-1은 최종 모델).

    쓰는 곳: Train_.py(저장), Run.load_model(불러오기).
    """
    prefix = _FILE_PREFIX[model]
    return f"{prefix}.pth" if number == -1 else f"{prefix}_e{number}.pth"


def write_sim_config(path, args, extra, exclude: Sequence[str] = ()) -> str:
    """학습 설정을 path/sim_config.txt에 쓰고, 쓴 내용을 문자열로 돌려준다.

    args는 명령줄 인자, extra는 덧붙일 값(학습 차량 ID 목록 등)이다. exclude에 든 항목은 쓰지 않는다
    (Train_.py는 기록할 필요가 없는 --debug를 뺀다). 형식은 print_sim_config의 출력과 같다.
    쓰는 곳: Train_.py.
    """
    buf = io.StringIO()
    with redirect_stdout(buf):
        print_sim_config(title="Train Run", config=args, extra=extra, exclude=exclude)
    text = buf.getvalue()
    os.makedirs(path, exist_ok=True)
    with open(f"{path}/sim_config.txt", "w", encoding="utf-8") as f:
        f.write(text)
    return text


def save_net_state(model, models_dir, filename):
    """모델 가중치(state_dict)를 CPU 텐서로 바꿔 models_dir/filename에 저장한다.

    GPU에서 학습해도 CPU에서 불러올 수 있게 하려고 CPU로 옮겨 저장한다.
    쓰는 곳: Train_.py의 train_autoencoder (중간 체크포인트와 최종 모델).
    """
    os.makedirs(models_dir, exist_ok=True)
    state = {k: v.detach().cpu() for k, v in model.state_dict().items()}
    torch.save(state, os.path.join(models_dir, filename))


def make_model_path_based_timestamp(base):
    """새 Run 폴더 경로 "<base>/<RUN_ID>"를 만든다. RUN_ID는 서울 시간 기준 "26MMDD_HHMMSS"이다.

    연도 자리는 "26"으로 고정되어 있다. 폴더 자체는 만들지 않는다(처음 저장할 때 생긴다).
    쓰는 곳: Train_.py (학습 시작 시 결과 폴더 이름을 정한다).
    """
    now = datetime.now(ZoneInfo("Asia/Seoul"))
    return os.path.join(base, now.strftime("26%m%d_%H%M%S"))


def print_sim_config(title: str, config, extra: Mapping[str, Any], exclude: Sequence[str] = ()) -> None:
    """설정을 "항목 : 값" 표로 화면에 출력한다. sim_config.txt도 이 출력 형식으로 저장된다.

    맨 위에 실행 환경(실행 시각, 파이썬 버전, 운영체제, 실행 폴더)을, 그 아래에 config(argparse 인자)와
    extra(덧붙일 값, 예: device·학습 차량 ID)를 항목 이름 순서로 출력한다. 이름이 같으면 extra 값을 쓴다.
    exclude에 든 항목은 출력하지 않는다.
    쓰는 곳: Test_.py·Test_multi_models.py·Test_tsne.py(시작할 때 설정 표시), write_sim_config(파일 저장).
    """
    cfg = dict(vars(config))
    for k in exclude:
        cfg.pop(k, None)
    cfg.update(extra)
    items = sorted(cfg.items(), key=lambda kv: kv[0])

    sys_lines = [
        ("timestamp", datetime.now().strftime("%Y-%m-%d %H:%M:%S")),
        ("python", sys.version.split()[0]),
        ("platform", f"{platform.system()} {platform.release()} ({platform.machine()})"),
        ("cwd", os.getcwd()),
    ]

    # 출력 폭: 가장 긴 항목 이름과 값에 맞추되 60~140자로 제한
    all_items = sys_lines + items
    key_w = max([len(str(k)) for k, _ in all_items] + [3])
    val_w = max(len(_fmt(v)) for _, v in all_items)
    line_w = min(max(key_w + 3 + val_w + 2, 60), 140)

    print("=" * line_w)
    print(title.center(line_w))
    print("=" * line_w)
    for k, v in sys_lines:
        print(f"{str(k):<{key_w}} : {_fmt(v)}")
    print("-" * line_w)
    for k, v in items:
        print(f"{str(k):<{key_w}} : {_fmt(v)}")
    print("=" * line_w)


def _fmt(v: Any) -> str:
    """print_sim_config 안에서 쓴다. 값 하나를 한 줄 문자열로 바꾼다 (목록은 [a, b], 줄바꿈은 \\n, 180자가 넘으면 ...)."""
    if isinstance(v, (list, tuple, set)):
        s = "[" + ", ".join(map(str, list(v))) + "]"
    else:
        s = str(v)

    s = s.replace("\n", "\\n")
    if len(s) > 180:
        s = s[:177] + "..."
    return s


def _build_model(config: RunConfig, model: str):
    """설정에 맞는 CombinedAE(MC-AE) 하나를 새로 만든다. model은 "U"(셀 전압) 또는 "X"(셀 SOC).

    입력·출력 크기는 배터리 종류의 열 구성(get_input_dimensions)에서, 출력 범위(scale, shift)와 옵션은
    설정에서 정한다. 쓰는 곳: build_model_pair(학습), Run.load_model(평가).
    """
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
    """RunConfig.from_mapping 안에서 쓴다. 학습 때 쓴 분할(--dataset-name)이 있는 폴더를 정한다.

    이름이 있고 ./datasets/<이름> 폴더가 있으면 그 폴더, 이름이 없거나 "default"면 저장소 루트(".")를 쓴다.
    기록된 폴더가 없으면 경고를 출력하고 루트를 쓴다.
    """
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
    """Run.__init__ 안에서 쓴다. sim_config.txt의 "항목 : 값" 줄을 {항목: 값 문자열}로 읽는다.

    표 구분선(====, ----)과 빈 줄은 건너뛴다. 값을 숫자 등으로 바꾸는 일은 _coerce_value가 한다.
    """
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
    """Run.__init__ 안에서 쓴다. sim_config.txt의 값 문자열을 파이썬 값으로 바꾼다.

    "True"/"False" → bool, "3" → int, "0.5" → float, "[1, 2]" 같은 목록 → list로 바꾼다.
    "None"이나 "1e-05"(지수 표기)처럼 애매한 값은 문자열 그대로 둔다.
    """
    s = str(raw).strip()
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
