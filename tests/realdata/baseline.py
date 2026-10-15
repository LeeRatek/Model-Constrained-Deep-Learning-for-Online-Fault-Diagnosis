"""실데이터 기준선 수집·비교 도구 (.scratch/run-module 티켓 01).

리팩토링 전후로 같은 명령을 실행해 결과를 bit 단위로 비교한다.
pytest가 수집하지 않는 파일 이름이며, 저장소 루트에서 직접 실행한다.

    python tests/realdata/baseline.py capture <target> <tag>
    python tests/realdata/baseline.py compare <target> <tag_a> <tag_b>

target: test | test_u0 | tsne | multi | train | train_full
결과:   models/_verify/baselines/<tag>/<target>.pkl  (+ 같은 이름의 .log)

스크립트 코드는 건드리지 않는다. 진단 특징·PCA·T²/SPE·ROC·t-SNE 입력 함수를 기록용 wrapper로
바꿔 둔 뒤 runpy로 스크립트를 실행한다. 스크립트와 mcdl module은 import할 때 이 wrapper를 가져간다.
WRAP_TARGETS의 위치 목록에는 지금 위치(mcdl.*)와 예전 위치(Function_)가 함께 있다. 예전 위치는 옛 코드로
기준선을 다시 뜰 때만 쓰이고, 없는 module이면 건너뛴다.
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

ROOT = Path(__file__).resolve().parents[2]
VERIFY_DIR = ROOT / "models" / "_verify"
BASELINE_DIR = VERIFY_DIR / "baselines"
RUN_ID = "260224_090359"
TRAIN_MODELS_DIR = "./models/_verify/train"

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

RECORDS = []


# ---------------------------------------------------------------- digest


def digest(x):
    """값을 bit 단위로 구분하는 짧은 문자열로 바꾼다 (dtype·shape·라벨 포함)."""
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
    return hashlib.sha256(json.dumps(parts).encode()).hexdigest()[:16]


# ---------------------------------------------------------------- 기록용 wrapper


def _as_diagnosis_feature_output(features):
    """diagnosis_features는 DataFrame만 돌려준다. 이전 DiagnosisFeature의 (df, None) 모양으로 기록한다."""
    return features, None


# (기록 이름, 함수가 있을 수 있는 위치들, 옵션). 함수를 다른 module로 옮겨도 같은 기록 이름을 써서
# 이전 기준선(before1)과 그대로 비교한다. 위치는 존재하는 것만 감싼다.
# 위치에 세 번째 값(dict)이 있으면 그 위치에만 추가 옵션을 적용한다.
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
    """legacy_out: 옮긴 함수의 반환 모양이 바뀐 경우, 이전 함수의 반환 모양으로 바꿔 기록한다."""
    orig = getattr(module, attr)

    def wrapper(*args, **kwargs):
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
        Z = orig_fit_transform(self, X, *args, **kwargs)
        RECORDS.append(("TSNE.fit_transform", {"in": digest(X), "out": digest(Z), "full": Z}))
        return Z

    TSNE.fit_transform = fit_transform


# ---------------------------------------------------------------- capture


class _Tee(io.TextIOBase):
    def __init__(self, *streams):
        self.streams = streams

    def write(self, s):
        for st in self.streams:
            st.write(s)
        return len(s)

    def flush(self):
        for st in self.streams:
            st.flush()


def capture(target, tag):
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
    return next(payload["full"] for n, payload in reversed(RECORDS) if n == name)


def _train_run_dirs():
    base = ROOT / TRAIN_MODELS_DIR
    return {p.name for p in base.iterdir() if p.is_dir()} if base.exists() else set()


def _digest_train_run(run_dir):
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
    lines = []
    for line in text.splitlines():
        key = line.split(":", 1)[0].strip()
        if ":" in line and key in SIM_CONFIG_VOLATILE_KEYS:
            line = line.split(":", 1)[0] + ": <ignored>"
        lines.append(line)
    return "\n".join(lines)


# ---------------------------------------------------------------- compare


def compare(target, tag_a, tag_b):
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
    with open(BASELINE_DIR / tag / f"{target}.pkl", "rb") as f:
        return pickle.load(f)


def _first_name_diff(ra, rb):
    for i, ((na, _), (nb, _)) in enumerate(zip(ra, rb)):
        if na != nb:
            return f"#{i} {na} vs {nb}"
    return f"#{min(len(ra), len(rb))} (한쪽이 끝남)"


def _equal(x, y):
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
