"""test_diagnosis.py가 정답으로 쓰는 기대값 파일(tests/fixtures/diagnosis_golden.npz)을 다시 만드는 도구.

이름에 쓰인 말
    golden: 테스트에서 "이 입력을 넣으면 정확히 이 값이 나와야 한다"고 미리 저장해 둔 정답 값을 golden 값
        (기준이 되는 값이라는 뜻)이라고 부른다. 이 도구는 그 정답 값을 만들기 때문에 이름이 make_..._golden이다.
    fixture: 테스트가 읽어 쓰려고 미리 준비해 둔 데이터 파일을 부르는 말이다. 그래서 그런 파일을
        tests/fixtures/ 폴더에 모아 둔다. 이 도구가 만드는 diagnosis_golden.npz도 그중 하나다.
    mcdl: 이 저장소의 계산 코드가 들어 있는 폴더(파이썬 패키지) 이름. 논문 제목 Model-Constrained Deep
        Learning의 머리글자다.

왜 필요한가
    test_diagnosis.py는 합성 오차(synthetic_errors.py)를 넣어 진단 특징·관리도(T², SPE, CI)·ROC를 계산하고,
    결과가 이 파일에 저장된 값과 소수점 끝자리까지 완전히 같은지(bit 단위로 같은지) 비교한다.
    - 계산 코드를 고친 적이 없는데 테스트가 실패하면, 무언가 의도치 않게 바뀐 것이다. 이 도구로 기대값을
      덮어쓰지 말고 원인을 찾는다.
    - 계산 방법을 일부러 바꿨다면, 이 도구로 기대값을 새로 만들어 저장하고 바뀐 이유를 커밋 메시지에 남긴다.
    계산을 바꾸지 않은 상태에서 다시 만들면 지금 파일과 똑같은 값이 나온다(--check로 확인할 수 있다).

실행 (저장소 루트에서)
    python tests/make_diagnosis_golden.py            # 기대값 파일을 덮어쓴다
    python tests/make_diagnosis_golden.py --check    # 덮어쓰지 않고 지금 파일과 비교만 한다

파일 이름이 test_로 시작하지 않으므로 pytest(테스트 실행 도구)는 이 파일을 테스트로 실행하지 않는다.
"""

import argparse
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
# 저장소 루트(mcdl 패키지)와 tests 폴더(synthetic_errors.py)를 import할 수 있게 경로에 넣는다.
sys.path[:0] = [str(HERE.parent), str(HERE)]

from mcdl.diagnosis import ControlChart, diagnosis_features  # noqa: E402
from mcdl.roc import evaluate  # noqa: E402
from synthetic_errors import make_vehicle_errors  # noqa: E402

FIXTURE = HERE / "fixtures" / "diagnosis_golden.npz"


def build():
    """test_diagnosis.py가 비교하는 기대값을 전부 계산해 {이름: 배열} dict로 돌려준다.

    합성 오차로: 학습 차량 진단 특징 → 관리도 학습(저장·불러오기 포함) → 테스트 차량별 T²·SPE·CI·최대 CI
    → ROC(정상 방향, 반대 방향) 순서로 계산한다. 쓰는 곳: main.
    """
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
    """build로 기대값을 만든 뒤, 기본은 파일에 덮어쓰고 --check면 지금 파일과 항목별로 비교만 한다.

    다른 항목이 있으면 1, 없으면 0을 돌려준다(명령줄 종료 코드). 쓰는 곳: 이 파일 맨 아래 실행부.
    """
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
