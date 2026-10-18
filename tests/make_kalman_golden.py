"""test_kalman.py가 정답으로 쓰는 기대값 파일(tests/fixtures/kalman_golden.npz)을 다시 만드는 도구.

지금 파일의 값은 칼만 필터를 지우기 직전의 원본 코드(커밋 a93ea3a^의 Function_.py)로 만들었다. 그래서
test_kalman.py가 통과하면 복원한 mcdl/kalman.py가 원본과 bit 단위로 같은 계산을 한다는 뜻이다.
golden(기준이 되는 정답 값)·fixture(테스트가 읽는 미리 준비된 데이터 파일)라는 말의 뜻은 tests/README.md에 있다.

    python tests/make_kalman_golden.py            # 지금 mcdl/kalman.py로 기대값 파일을 덮어쓴다
    python tests/make_kalman_golden.py --check    # 덮어쓰지 않고 지금 파일과 비교만 한다

계산 방법을 일부러 바꿨을 때만 덮어쓴다. 고친 적이 없는데 test_kalman.py가 실패하면 덮어쓰지 말고 원인을 찾는다.
"""

import argparse
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
# 저장소 루트(mcdl 패키지)와 tests 폴더(synthetic_battery.py)를 import할 수 있게 경로에 넣는다.
sys.path[:0] = [str(HERE.parent), str(HERE)]

from mcdl.kalman import calculate_volt_modepi, solvers  # noqa: E402
from synthetic_battery import make_kalman_inputs  # noqa: E402

FIXTURE = HERE / "fixtures" / "kalman_golden.npz"
OUTPUT_NAMES = ("xipre", "xpre", "Spre", "Upre", "Xipre", "x", "DUi", "Xi")  # solvers가 돌려주는 순서


def build(calculate_volt_modepi=calculate_volt_modepi, solvers=solvers):
    """합성 입력으로 calculate_volt_modepi와 solvers를 계산해 {이름: 배열} dict로 돌려준다.

    함수를 인자로 받을 수 있어, 원본 코드의 함수로도 같은 기대값을 만들 수 있다. 쓰는 곳: main, test_kalman.py.
    """
    inputs = make_kalman_inputs()
    volt_modepi, volt_di = calculate_volt_modepi(inputs["volt_all"])
    outputs = solvers(
        volt_modepi, volt_di, inputs["volt_all"], inputs["soc"], inputs["b"], inputs["current"], inputs["temp_avg"]
    )
    out = {"volt_modepi": volt_modepi.to_numpy(), "volt_di": volt_di.to_numpy()}
    out.update({name: np.asarray(value, dtype=float) for name, value in zip(OUTPUT_NAMES, outputs)})
    return out


def main():
    """build로 기대값을 만든 뒤, 기본은 파일에 덮어쓰고 --check면 지금 파일과 항목별로 비교만 한다.

    다른 항목이 있으면 1, 없으면 0을 돌려준다(명령줄 종료 코드). 쓰는 곳: 이 파일 맨 아래 실행부.
    """
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--check", action="store_true", help="덮어쓰지 않고 지금 기대값 파일과 비교만 한다")
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
