"""tests/fixtures/diagnosis_golden.npz(test_diagnosis.py의 기대값)를 지금 mcdl 구현으로 다시 만든다.

진단 특징·관리도·ROC 계산을 일부러 바꿨을 때 쓴다. 계산을 바꾸지 않았다면 다시 만들어도 지금 fixture와
bit 단위로 같은 값이 나온다. pytest가 수집하지 않는 파일 이름이며, 저장소 루트에서 직접 실행한다.

    python tests/make_diagnosis_golden.py            # fixture를 덮어쓴다
    python tests/make_diagnosis_golden.py --check    # 덮어쓰지 않고 지금 fixture와 비교만 한다
"""

import argparse
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(HERE.parent), str(HERE)]

from mcdl.diagnosis import ControlChart, diagnosis_features  # noqa: E402
from mcdl.roc import evaluate  # noqa: E402
from synthetic_errors import make_vehicle_errors  # noqa: E402

FIXTURE = HERE / "fixtures" / "diagnosis_golden.npz"


def build():
    """test_diagnosis.py가 비교하는 기대값 전부."""
    train, test_normal, test_fault = make_vehicle_errors()
    out = {}

    features = pd.concat([diagnosis_features(eu, ex) for eu, ex in train], axis=0, ignore_index=True)
    out["train_features"] = features.to_numpy()

    chart = ControlChart.fit(features)
    with tempfile.TemporaryDirectory() as tmp:
        chart.save(f"{tmp}/", features)
        with np.load(Path(tmp) / "pca_arrays.npz") as arrays:
            for name in arrays.files:
                out[f"pca_{name}"] = arrays[name]
        loaded = ControlChart.load(tmp)
    out["pca_data_nor"] = ((features - out["pca_data_mean"]) / out["pca_data_std"]).to_numpy()

    max_ci = []
    for j, (eu, ex) in enumerate(test_normal + test_fault):
        vehicle = diagnosis_features(eu, ex)
        out[f"test_features_{j}"] = vehicle.to_numpy()
        out[f"t2_{j}"], out[f"spe_{j}"], out[f"ci_{j}"] = (np.asarray(a) for a in chart.score(vehicle))
        out[f"ci_loaded_{j}"] = np.asarray(loaded.score(vehicle)[2])
        max_ci.append(chart.max_ci(vehicle))
    out["max_ci"] = np.asarray(max_ci)

    n_normal = len(test_normal)
    for tag, reverse in (("roc", False), ("roc_reverse", True)):
        normal = [(s, False) for s in max_ci[:n_normal]]
        fault = [(s, True) for s in max_ci[n_normal:]]
        vehicles = fault + normal if reverse else normal + fault  # Test_multi_models는 reverse일 때 고장 차량을 먼저 넘긴다
        result = evaluate([s for s, _ in vehicles], [f for _, f in vehicles], reverse=reverse)
        b, c = result.best, result.closest_to_01
        out[f"{tag}_fpr"] = np.asarray(result.fpr)
        out[f"{tag}_tpr"] = np.asarray(result.tpr)
        out[f"{tag}_auc"] = np.asarray(result.auc)
        out[f"{tag}_best"] = np.asarray([b["threshold"], b["fpr"], b["tpr"], b["youden_j"]])
        out[f"{tag}_best_index"] = np.asarray(b["index"])
        out[f"{tag}_closest"] = np.asarray([c["threshold"], c["fpr"], c["tpr"], c["distance"]])
        out[f"{tag}_closest_index"] = np.asarray(c["index"])
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--check", action="store_true", help="덮어쓰지 않고 지금 fixture와 비교만 한다")
    args = parser.parse_args()

    new = build()
    if not args.check:
        np.savez_compressed(FIXTURE, **new)
        print(f"저장: {FIXTURE} ({len(new)}개 항목)")
        return 0

    old = dict(np.load(FIXTURE))
    differ = [
        k for k in new
        if k not in old or old[k].dtype != new[k].dtype or old[k].shape != new[k].shape or not np.array_equal(old[k], new[k])
    ]
    for k in differ:
        print(f"다름: {k}")
    print(f"{len(new)}개 항목 중 {len(differ)}개가 다릅니다.")
    return 1 if differ else 0


if __name__ == "__main__":
    sys.exit(main())
