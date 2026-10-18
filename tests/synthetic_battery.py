"""칼만 필터 테스트에 쓰는 합성(인공으로 만든) 배터리 측정값. 테스트 파일이 아니라 테스트가 불러 쓰는 도우미다.

이 저장소에는 칼만 필터(mcdl/kalman.py)에 넣을 OCV 계수와 원시 측정값이 없다. 그래서 그럴듯한 OCV 곡선과
셀 전압·전류·온도·SOC를 난수로 만들어, 계산이 바뀌지 않았는지만 확인한다. 시드를 고정해 언제나 같은 값이 나온다.

쓰는 곳: test_kalman.py, make_kalman_golden.py (test_kalman.py가 비교할 기대값을 만들 때).
"""

import numpy as np
import pandas as pd

SEED = 20261019  # 고정 시드. 바꾸면 기대값 파일(fixtures/kalman_golden.npz)도 다시 만들어야 한다
N_STEPS = 150  # 시점 수
N_CELLS = 6  # 셀 수

# OCV 계수 21개: b[0..17]은 SOC 다항식, b[18..20]은 온도 1·2·3차 항. 고차 항까지 모두 계산되도록 작은 값을 넣었다
OCV_COEFFS = [3.2, 1.0, -0.3, 0.12] + [(-1) ** n * 0.01 / n for n in range(4, 18)] + [1e-3, -1e-5, 1e-8]


def make_kalman_inputs(seed=SEED):
    """solvers에 넣을 입력을 만든다: volt_all(셀 전압 DataFrame), soc(배열), b(OCV 계수), current·temp_avg(Series).

    SOC는 0.9에서 시작해 방전 전류로 조금씩 줄고, 셀 전압은 OCV에서 셀마다 다른 작은 편차와 잡음을 더한 값이다.
    쓰는 곳: test_kalman.py, make_kalman_golden.py.
    """
    rng = np.random.default_rng(seed)
    current = 30 + rng.normal(0, 5, N_STEPS)  # 양수가 방전
    soc = 0.9 - np.cumsum(current) / (23 * 3600)
    temp = 25 + rng.normal(0, 0.5, N_STEPS)

    b = OCV_COEFFS
    ocv = np.array([sum(b[n] * s ** n for n in range(18)) + b[18] * T + b[19] * T ** 2 + b[20] * T ** 3
                    for s, T in zip(soc, temp)])
    cell_offset = rng.normal(0, 0.005, N_CELLS)
    volt = ocv[:, None] - 1e-3 * current[:, None] + cell_offset[None, :] + rng.normal(0, 1e-3, (N_STEPS, N_CELLS))

    return {
        "volt_all": pd.DataFrame(volt),
        "soc": soc,
        "b": b,
        "current": pd.Series(current),
        "temp_avg": pd.Series(temp),
    }
