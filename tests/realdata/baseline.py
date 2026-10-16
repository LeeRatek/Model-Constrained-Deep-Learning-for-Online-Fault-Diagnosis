"""실제 데이터로 계산 결과가 바뀌지 않았는지 확인하는 도구: 스크립트 실행 결과를 기록하고(capture) 두 기록을 비교한다(compare).

코드를 고치기 전에 한 번 기록하고 고친 뒤에 다시 기록해 둘을 비교하면, 실제 데이터에서 결과가 소수점
끝자리까지 같은지(bit 단위로 같은지) 알 수 있다. 비교의 기준이 되는 이 기록을 "기준선(baseline)"이라고 부른다.
실제 데이터(data/)와 학습된 Run이 있어야 해서 pytest 테스트에는 넣지 않았다(파일 이름이 test_로 시작하지 않는다).

실행 (저장소 루트에서)
    python tests/realdata/baseline.py capture <target> <tag>            # 기록
    python tests/realdata/baseline.py compare <target> <tag_a> <tag_b>  # 두 기록 비교

    target: 무엇을 실행할지. test | test_u0 | tsne | multi | train | train_full (명령은 아래 TARGETS)
    tag: 기록에 붙이는 이름(예: before, after). 결과는 models/_verify/baselines/<tag>/<target>.pkl과
         같은 이름의 .log(화면 출력)로 저장된다.
    평가 target은 models/_verify/260224_090359(실제 Run을 복사해 둔 폴더)가 있어야 한다.

어떻게 기록하나
    스크립트 코드는 고치지 않는다. 대신 진단 특징·PCA·T²/SPE·ROC·t-SNE 계산 함수를 잠시 "감싸는 함수
    (wrapper)"로 바꿔치기한다(install_wrappers). 감싸는 함수는 원래 함수를 그대로 불러 같은 결과를 돌려주면서,
    입력·출력의 지문(digest: 값이 1 bit만 달라도 바뀌는 짧은 문자열)을 RECORDS에 남긴다. 그 상태로 스크립트를
    명령줄에서 실행한 것과 똑같이 실행한다(runpy).
    WRAP_TARGETS에는 함수의 지금 위치(mcdl.*)와 리팩터링 전 위치(Function_)가 함께 적혀 있다. 예전 위치는
    옛 코드로 기준선을 다시 기록할 때만 쓰이고, 그 파일이 없으면(지금 코드) 건너뛴다.
"""

import os

os.environ.setdefault("MPLBACKEND", "Agg")  # matplotlib import 전에 설정해야 plt.show()가 멈추지 않는다

import argparse
import hashlib
import io
import json
import pickle
import platform
import runpy
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]  # 저장소 루트
VERIFY_DIR = ROOT / "models" / "_verify"  # 확인용 Run 복사본과 기록을 두는 폴더
BASELINE_DIR = VERIFY_DIR / "baselines"  # 기록 저장 위치: <tag>/<target>.pkl
RUN_ID = "260224_090359"  # 평가 target이 쓰는 Run (models/_verify/ 아래 복사본)
TRAIN_MODELS_DIR = "./models/_verify/train"  # 학습 target이 새 Run을 만드는 곳

# target 이름 → 실행할 명령 (스크립트 파일 + 옵션)
TARGETS = {
    "test": [
        "Test_.py",
        "--models-dir", "./models/_verify",
        "--models-folder", RUN_ID,
        "--u-model-idx", "-1",
        "--x-model-idx", "-1",
    ],
    # 캐시가 없을 때 학습 데이터로 관리도를 다시 계산하는 경로 (모델 쌍 U 0, X -1)
    "test_u0": [
        "Test_.py",
        "--models-dir", "./models/_verify",
        "--models-folder", RUN_ID,
        "--u-model-idx", "0",
        "--x-model-idx", "-1",
    ],
    "tsne": [
        "Test_tsne.py",
        "--models-dir", "./models/_verify",
        "--models-idx", RUN_ID,
        "--u-model-idx", "-1",
        "--x-model-idx", "-1",
        "--fault-idx", "338",
        "--save",
    ],
    "multi": [
        "Test_multi_models.py",
        "--models-dir", "./models/_verify",
        "--models-idx", RUN_ID,
        "--u-model-idx", "-1",
        "--x-model-idx", "-1",
    ],
    "train": [
        "Train_.py",
        "--vehicle-start", "0",
        "--vehicle-end", "0",
        "--ae-u-epochs", "1",
        "--ae-x-epochs", "1",
        "--ae-u-save-per", "1",
        "--ae-x-save-per", "1",
        "--no-validation",
        "--models-dir", TRAIN_MODELS_DIR,
    ],
    # 학습 루프의 RNG 순서·검증·체크포인트 간격까지 보는 학습 (차량 2대, 3 epoch, save-per 2, shuffle, 검증 50대)
    "train_full": [
        "Train_.py",
        "--vehicle-start", "0",
        "--vehicle-end", "1",
        "--ae-u-epochs", "3",
        "--ae-x-epochs", "3",
        "--ae-u-save-per", "2",
        "--ae-x-save-per", "2",
        "--ae-u-shuffle",
        "--ae-x-shuffle",
        "--models-dir", TRAIN_MODELS_DIR,
    ],
}
TRAIN_TARGETS = ("train", "train_full")

# sim_config.txt에서 실행 환경에 따라 달라지는 줄 (비교에서 값만 지운다)
SIM_CONFIG_VOLATILE_KEYS = ("timestamp", "python", "platform", "cwd")

# 실행 중 모은 기록: (기록 이름, {"in": 입력 지문, "out": 출력 지문, "full": 값 자체}) 목록. capture가 파일로 저장한다.
RECORDS = []


# ---------------------------------------------------------------- 지문(digest) 만들기


def digest(x):
    """값의 지문(짧은 문자열)을 만든다. 값·자료형(dtype)·모양(shape)·열 이름 중 하나라도 다르면 지문이 달라진다.

    배열은 메모리 내용의 SHA-256 해시 앞 16자리, 실수 하나는 16진 표기(float.hex)로 남겨 마지막 bit까지 구분한다.
    쓰는 곳: _wrap, install_wrappers, _digest_train_run.
    """
    import numpy as np
    import pandas as pd
    import torch

    if isinstance(x, torch.Tensor):
        x = x.detach().cpu().numpy()
    if isinstance(x, pd.DataFrame):
        return _hash(
            ["DataFrame", repr(list(x.columns)), digest(x.index.to_numpy()), digest(x.to_numpy())]
        )
    if isinstance(x, pd.Series):
        return _hash(["Series", repr(x.name), digest(x.index.to_numpy()), digest(x.to_numpy())])
    if isinstance(x, np.ndarray):
        if x.dtype == object:
            return _hash(["object", repr(x.shape)] + [digest(v) for v in x.ravel().tolist()])
        a = np.ascontiguousarray(x)
        return f"{a.dtype.str}{a.shape}:{hashlib.sha256(a.tobytes()).hexdigest()[:16]}"
    if isinstance(x, np.generic):
        return digest(np.asarray(x))
    if isinstance(x, dict):
        return _hash(["dict"] + [f"{k!r}={digest(v)}" for k, v in x.items()])
    if isinstance(x, (list, tuple)):
        return _hash([type(x).__name__] + [digest(v) for v in x])
    if isinstance(x, float):
        return float.hex(x)
    return repr(x)


def _hash(parts):
    """문자열 조각 목록을 하나로 합쳐 SHA-256 해시 앞 16자리로 줄인다. 쓰는 곳: digest."""
    return hashlib.sha256(json.dumps(parts).encode()).hexdigest()[:16]


# ---------------------------------------------------------------- 기록용 감싸는 함수(wrapper)


def _as_diagnosis_feature_output(features):
    """지금 diagnosis_features는 DataFrame 하나만 돌려주지만, 예전 DiagnosisFeature는 (DataFrame, None)을 돌려줬다.

    예전 코드로 뜬 기준선과 비교할 수 있게 예전 모양으로 바꿔 기록한다. 쓰는 곳: WRAP_TARGETS.
    """
    return features, None


# 기록할 함수 목록: (기록 이름, 함수가 있을 수 있는 위치들, 옵션). 함수를 다른 파일로 옮겨도 같은 기록 이름을 써서
# 예전에 뜬 기준선과 그대로 비교한다. 위치는 실제로 있는 것만 감싼다.
# 위치에 세 번째 값(dict)이 있으면 그 위치에만 추가 옵션을 적용한다. 옵션의 뜻은 _wrap 설명에 있다.
WRAP_TARGETS = [
    (
        "DiagnosisFeature",
        [
            ("Function_", "DiagnosisFeature"),
            ("mcdl.diagnosis", "diagnosis_features", {"legacy_out": _as_diagnosis_feature_output}),
        ],
        {},
    ),
    ("T2_array", [("Function_", "T2_array"), ("mcdl.diagnosis", "_t2")], {}),
    ("SPE_array", [("Function_", "SPE_array"), ("mcdl.diagnosis", "_spe")], {}),
    ("Custom_PCA", [("Function_", "Custom_PCA"), ("mcdl.diagnosis", "_pca")], {"inputs": True}),
    ("load_pca_results", [("Function_", "load_pca_results"), ("mcdl.diagnosis", "_load_pca")], {}),
    (
        "compute_roc_auc_from_threshold_matrix",
        [
            ("Function_", "compute_roc_auc_from_threshold_matrix"),
            ("mcdl.roc", "_roc_from_threshold_matrix"),
        ],
        {"inputs": True, "keep_full": True, "keep_last_args": True},
    ),
    (
        "compute_optimal_thresholds_from_roc",
        [
            ("Function_", "compute_optimal_thresholds_from_roc"),
            ("mcdl.roc", "_optimal_thresholds"),
        ],
        {"keep_full": True},
    ),
    # 반환값(fig, axes)은 객체 주소가 섞이므로 입력만 기록한다
    (
        "plot_tsne_before_after_ci",
        [("Function_", "plot_tsne_before_after_ci"), ("mcdl.plotting", "plot_tsne_before_after_ci")],
        {"inputs": True, "output": False, "drop_kwargs": ("args",)},
    ),
]

LAST_ARGS = {}  # 기록 이름 → 마지막 호출의 위치 인자 (Test_.result 구성용)


def _wrap(
    module,
    attr,
    name,
    *,
    inputs=False,
    output=True,
    keep_full=False,
    drop_kwargs=(),
    keep_last_args=False,
    legacy_out=None,
):
    """module의 함수 attr를 기록하는 감싸는 함수로 바꿔치기한다. 기록은 name이라는 이름으로 RECORDS에 쌓인다.

    옵션
        inputs=True: 입력(인자)의 지문도 남긴다.        output=False: 출력 지문을 남기지 않는다.
        keep_full=True: 출력 값 자체도 남긴다(다를 때 얼마나 다른지 보려고).
        drop_kwargs: 입력 지문에서 뺄 키워드 인자 이름들.
        keep_last_args=True: 마지막 호출의 위치 인자를 LAST_ARGS[name]에 둔다.
        legacy_out: 옮긴 함수의 반환 모양이 바뀐 경우, 예전 함수의 반환 모양으로 바꿔 기록하는 함수.
    쓰는 곳: install_wrappers.
    """
    orig = getattr(module, attr)

    def wrapper(*args, **kwargs):
        """원래 함수를 그대로 불러 결과를 돌려주고, 그 사이에 입력·출력 지문을 RECORDS에 남긴다."""
        out = orig(*args, **kwargs)
        recorded = legacy_out(out) if legacy_out else out
        payload = {}
        if inputs:
            kept = {k: v for k, v in kwargs.items() if k not in drop_kwargs}
            payload["in"] = digest([list(args), kept])
        if output:
            payload["out"] = digest(recorded)
        if keep_full:
            payload["full"] = recorded
        if keep_last_args:
            LAST_ARGS[name] = args
        RECORDS.append((name, payload))
        return out

    setattr(module, attr, wrapper)


def install_wrappers():
    """WRAP_TARGETS의 함수들과 t-SNE 계산(sklearn TSNE.fit_transform)을 기록하는 함수로 바꿔치기한다.

    스크립트를 실행하기 전에 한 번 부른다. 스크립트와 mcdl 파일들은 그 뒤에 import되므로 바뀐 함수를 쓰게 된다.
    쓰는 곳: capture.
    """
    import importlib

    from sklearn.manifold import TSNE

    for name, locations, options in WRAP_TARGETS:
        for module_name, attr, *location_options in locations:
            try:
                module = importlib.import_module(module_name)
            except ModuleNotFoundError:
                continue
            if hasattr(module, attr):
                _wrap(module, attr, name, **options, **(location_options[0] if location_options else {}))

    orig_fit_transform = TSNE.fit_transform

    def fit_transform(self, X, *args, **kwargs):
        """원래 t-SNE 계산을 그대로 하고, 입력·출력 지문과 출력 값 자체를 RECORDS에 남긴다."""
        Z = orig_fit_transform(self, X, *args, **kwargs)
        RECORDS.append(("TSNE.fit_transform", {"in": digest(X), "out": digest(Z), "full": Z}))
        return Z

    TSNE.fit_transform = fit_transform


# ---------------------------------------------------------------- capture


class _Tee(io.TextIOBase):
    """화면 출력을 여러 곳(화면과 .log 파일)에 동시에 쓰는 출력 대상. 쓰는 곳: capture."""

    def __init__(self, *streams):
        """출력을 함께 보낼 대상들(예: 원래 화면 출력, 열어 둔 로그 파일)을 받는다."""
        self.streams = streams

    def write(self, s):
        """글 s를 모든 대상에 쓴다."""
        for st in self.streams:
            st.write(s)
        return len(s)

    def flush(self):
        """모든 대상의 버퍼를 비워 실제로 출력되게 한다."""
        for st in self.streams:
            st.flush()


def capture(target, tag):
    """target 명령을 실행하며 기록을 모아 models/_verify/baselines/<tag>/<target>.pkl로 저장한다.

    화면 출력은 같은 이름의 .log에도 남긴다. 실행이 끝나면 target에 따라 기록을 하나 더 붙인다.
        test, test_u0: 최종 ROC 결과(임계값별 판정, 최적 임계값)
        multi: AUC_summary_case1.csv 내용
        train, train_full: 새로 생긴 학습 Run 폴더의 모든 파일 지문
    쓰는 곳: main.
    """
    import numpy as np
    import torch

    out_dir = BASELINE_DIR / tag
    out_dir.mkdir(parents=True, exist_ok=True)
    pkl_path = out_dir / f"{target}.pkl"
    log_path = out_dir / f"{target}.log"

    os.chdir(ROOT)
    sys.path.insert(0, str(ROOT))
    argv = TARGETS[target]
    train_runs_before = _train_run_dirs() if target in TRAIN_TARGETS else set()

    install_wrappers()
    start = time.perf_counter()
    with open(log_path, "w", encoding="utf-8") as log:
        real_stdout = sys.stdout
        sys.stdout = _Tee(real_stdout, log)
        try:
            sys.argv = list(argv)
            runpy.run_path(str(ROOT / argv[0]), run_name="__main__")
        finally:
            sys.stdout = real_stdout
    elapsed = time.perf_counter() - start

    meta = {
        "target": target,
        "tag": tag,
        "argv": argv,
        "elapsed_sec": round(elapsed, 1),
        "captured_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "numpy": np.__version__,
        "torch": torch.__version__,
    }

    if target in ("test", "test_u0"):
        # 스크립트 전역 변수 대신 ROC wrapper가 잡은 값으로 만든다 (리팩토링으로 변수 이름이 바뀌어도 비교 가능)
        y_true, predict_results = LAST_ARGS["compute_roc_auc_from_threshold_matrix"][:2]
        RECORDS.append(
            (
                "Test_.result",
                {
                    "full": {
                        "roc": _last_full("compute_roc_auc_from_threshold_matrix"),
                        "opt": _last_full("compute_optimal_thresholds_from_roc"),
                        "y_true": y_true,
                        "predict_results": predict_results,
                    }
                },
            )
        )
    elif target == "multi":
        run_dir = VERIFY_DIR / RUN_ID
        csv_text = (run_dir / "AUC_summary_case1.csv").read_text(encoding="utf-8")
        RECORDS.append(("AUC_summary_case1.csv", {"full": csv_text}))
        meta["temp_cache_left"] = (run_dir / "temp_cache").exists()
    elif target in TRAIN_TARGETS:
        new_runs = _train_run_dirs() - train_runs_before
        if len(new_runs) != 1:
            raise RuntimeError(f"새 학습 Run 폴더를 하나로 특정할 수 없습니다: {sorted(new_runs)}")
        run_dir = Path(TRAIN_MODELS_DIR) / new_runs.pop()
        meta["train_run_dir"] = str(run_dir)
        RECORDS.extend(_digest_train_run(run_dir))

    with open(pkl_path, "wb") as f:
        pickle.dump({"meta": meta, "records": RECORDS}, f)

    print(f"[baseline] {target} → {pkl_path} ({len(RECORDS)} records, {elapsed:.1f}s)")
    if meta.get("temp_cache_left"):
        print("[baseline] 경고: temp_cache가 남아 있습니다. Test_/Test_tsne의 PCA 출처가 바뀝니다.")


def _last_full(name):
    """RECORDS에서 name으로 남긴 마지막 기록의 값 자체(full)를 찾는다. 쓰는 곳: capture."""
    return next(payload["full"] for n, payload in reversed(RECORDS) if n == name)


def _train_run_dirs():
    """학습 target의 결과 폴더(models/_verify/train) 아래 Run 폴더 이름들.

    실행 전후의 목록을 비교해 새로 생긴 Run을 찾는다. 쓰는 곳: capture.
    """
    base = ROOT / TRAIN_MODELS_DIR
    return {p.name for p in base.iterdir() if p.is_dir()} if base.exists() else set()


def _digest_train_run(run_dir):
    """학습 Run 폴더 안 파일마다 기록을 하나씩 만든다.

    체크포인트(.pth)와 .npz는 항목별 지문, sim_config.txt는 환경 관련 줄을 지운 글, pca_manifest.json은 Run 폴더
    이름을 <RUN_ID>로 바꾼 글(실행할 때마다 폴더 이름이 달라지므로), 나머지는 파일 내용의 해시로 남긴다.
    쓰는 곳: capture.
    """
    import numpy as np
    import torch

    records = []
    for path in sorted(Path(run_dir).rglob("*")):
        if not path.is_file():
            continue
        rel = path.relative_to(run_dir).as_posix()
        if path.suffix == ".pth":
            state = torch.load(path, map_location="cpu")
            out = {k: digest(v) for k, v in state.items()}
        elif path.suffix == ".npz":
            with np.load(path, allow_pickle=True) as z:
                out = {k: digest(z[k]) for k in z.files}
        elif path.name == "sim_config.txt":
            out = _normalize_sim_config(path.read_text(encoding="utf-8"))
        elif path.name == "pca_manifest.json":
            out = path.read_text(encoding="utf-8").replace(path.parent.name, "<RUN_ID>")
        else:
            out = hashlib.sha256(path.read_bytes()).hexdigest()[:16]
        records.append((f"file:{rel}", {"out": out}))
    return records


def _normalize_sim_config(text):
    """sim_config.txt 글에서 실행할 때마다 달라지는 줄(SIM_CONFIG_VOLATILE_KEYS)의 값을 <ignored>로 바꾼다.

    쓰는 곳: _digest_train_run.
    """
    lines = []
    for line in text.splitlines():
        key = line.split(":", 1)[0].strip()
        if ":" in line and key in SIM_CONFIG_VOLATILE_KEYS:
            line = line.split(":", 1)[0] + ": <ignored>"
        lines.append(line)
    return "\n".join(lines)


# ---------------------------------------------------------------- compare


def compare(target, tag_a, tag_b):
    """두 기록(tag_a, tag_b)을 기록 순서대로 하나씩 비교해, 다른 곳을 최대 50개까지 출력한다.

    모두 같으면 0, 다르면 1을 돌려준다(명령줄 종료 코드). 쓰는 곳: main.
    """
    a = _load(tag_a, target)
    b = _load(tag_b, target)
    ra, rb = a["records"], b["records"]

    problems = []
    if [n for n, _ in ra] != [n for n, _ in rb]:
        problems.append(
            f"기록 순서/개수가 다릅니다: {len(ra)} vs {len(rb)} "
            f"(첫 차이: {_first_name_diff(ra, rb)})"
        )
    for i, ((name, pa), (_, pb)) in enumerate(zip(ra, rb)):
        for key in sorted(set(pa) | set(pb)):
            if not _equal(pa.get(key), pb.get(key)):
                problems.append(f"#{i} {name}.{key}: {_describe_diff(pa.get(key), pb.get(key))}")

    print(f"[compare] {target}: {tag_a} ({len(ra)} records, {a['meta']['elapsed_sec']}s) "
          f"vs {tag_b} ({len(rb)} records, {b['meta']['elapsed_sec']}s)")
    if problems:
        print(f"[compare] 불일치 {len(problems)}건")
        for p in problems[:50]:
            print("  - " + p)
        return 1
    print("[compare] 모든 기록이 bit 단위로 일치합니다.")
    return 0


def _load(tag, target):
    """저장된 기록 파일(baselines/<tag>/<target>.pkl) 하나를 읽는다. 쓰는 곳: compare."""
    with open(BASELINE_DIR / tag / f"{target}.pkl", "rb") as f:
        return pickle.load(f)


def _first_name_diff(ra, rb):
    """두 기록 목록에서 기록 이름이 처음 달라지는 위치를 설명하는 글. 쓰는 곳: compare."""
    for i, ((na, _), (nb, _)) in enumerate(zip(ra, rb)):
        if na != nb:
            return f"#{i} {na} vs {nb}"
    return f"#{min(len(ra), len(rb))} (한쪽이 끝남)"


def _equal(x, y):
    """두 기록 값이 완전히 같은지 본다. 배열은 자료형·모양·값(NaN끼리는 같다고 본다), dict·list는 안쪽까지 비교한다.

    쓰는 곳: compare, _describe_diff.
    """
    import numpy as np

    if type(x) is not type(y):
        return False
    if isinstance(x, np.ndarray):
        return (
            x.dtype == y.dtype
            and x.shape == y.shape
            and np.array_equal(x, y, equal_nan=x.dtype.kind in "fc")
        )
    if isinstance(x, dict):
        return x.keys() == y.keys() and all(_equal(x[k], y[k]) for k in x)
    if isinstance(x, (list, tuple)):
        return len(x) == len(y) and all(_equal(u, v) for u, v in zip(x, y))
    if isinstance(x, float):
        return x == y or (x != x and y != y)
    return x == y


def _describe_diff(x, y):
    """두 값이 어떻게 다른지 한 줄로 설명한다(실수 배열은 최대 절대 차이, 다른 배열은 다른 원소 수, dict는 다른 키).

    쓰는 곳: compare.
    """
    import numpy as np

    if isinstance(x, np.ndarray) and isinstance(y, np.ndarray) and x.shape == y.shape:
        if x.dtype.kind in "fc" and y.dtype.kind in "fc":
            return f"최대 절대 차이 {float(np.nanmax(np.abs(x - y))):.3e}"
        return f"다른 원소 {int(np.sum(x != y))}개"
    if isinstance(x, dict) and isinstance(y, dict):
        keys = [k for k in x.keys() | y.keys() if not _equal(x.get(k), y.get(k))]
        return "다른 키 " + ", ".join(map(str, sorted(keys, key=str))[:10])
    return f"{str(x)[:80]!r} vs {str(y)[:80]!r}"


# ---------------------------------------------------------------- main


def main():
    """명령줄에서 capture 또는 compare를 골라 실행한다. 쓰는 곳: 이 파일 맨 아래 실행부."""
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = parser.add_subparsers(dest="cmd", required=True)
    p_cap = sub.add_parser("capture")
    p_cap.add_argument("target", choices=TARGETS)
    p_cap.add_argument("tag")
    p_cmp = sub.add_parser("compare")
    p_cmp.add_argument("target", choices=TARGETS)
    p_cmp.add_argument("tag_a")
    p_cmp.add_argument("tag_b")
    args = parser.parse_args()

    if args.cmd == "capture":
        capture(args.target, args.tag)
        return 0
    return compare(args.target, args.tag_a, args.tag_b)


if __name__ == "__main__":
    sys.exit(main())
