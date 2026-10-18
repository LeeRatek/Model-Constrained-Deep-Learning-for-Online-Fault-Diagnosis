"""mcdl/kalman.py 테스트: 복원한 칼만 필터가 논문 원본 코드와 같은 계산을 하는지 확인한다.

합성 측정값(synthetic_battery.py)을 넣어 대표 전압(calculate_volt_modepi)과 칼만 필터 추정값(solvers)을 계산하고,
원본 코드(커밋 a93ea3a^의 Function_.py)로 미리 만들어 둔 기대값 tests/fixtures/kalman_golden.npz와
소수점 끝자리까지 완전히 같은지(bit 단위로 같은지) 비교한다. 기대값을 다시 만드는 방법은 make_kalman_golden.py에 있다.
"""

from pathlib import Path

import numpy as np
import pytest

from make_kalman_golden import build

GOLDEN = dict(np.load(Path(__file__).parent / "fixtures" / "kalman_golden.npz"))  # 기대값 {이름: 배열}


@pytest.fixture(scope="module")
def outputs():
    """지금 mcdl/kalman.py로 계산한 값 {이름: 배열}. 이 파일의 테스트가 함께 쓴다(한 번만 계산)."""
    return build()


@pytest.mark.parametrize("name", sorted(GOLDEN))
def test_matches_original_implementation(outputs, name):
    """각 결과(대표 전압, 셀 편차, 팩·셀 추정값 8개)가 원본 코드의 결과와 자료형·모양·값까지 같은지 본다."""
    actual, expected = outputs[name], GOLDEN[name]
    assert actual.dtype == expected.dtype and actual.shape == expected.shape
    assert np.array_equal(actual, expected)


def test_cell_soc_is_measured_soc_plus_estimated_deviation(outputs):
    """셀별 SOC 추정값 Xi가 측정 SOC + 셀별 편차 추정값인지 본다(계산된 시점 1 ~ N−2).

    xipre는 셀 편차를 한 시점 뒤로 민 값이므로, 시점 k의 편차는 xipre[k + 1]이다.
    """
    from synthetic_battery import make_kalman_inputs

    soc = make_kalman_inputs()["soc"]
    n = len(soc)
    deviation = outputs["xipre"][2:n]  # 시점 1 ~ N−2의 셀 편차
    np.testing.assert_allclose(outputs["Xi"][1 : n - 1], soc[1 : n - 1, None] + deviation, rtol=0, atol=1e-15)
