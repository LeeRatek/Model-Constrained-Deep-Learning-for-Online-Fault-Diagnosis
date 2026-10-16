"""진단 계산 테스트에 쓰는 합성(인공으로 만든) 재구성 오차. 테스트 파일이 아니라 테스트가 불러 쓰는 도우미다.

진단 계산(T², SPE, CI, ROC)은 모델이 낸 재구성 오차 |실제 값 − 재구성 값|만 입력으로 받는다. 그래서 실제 차량
데이터와 학습된 모델 없이도, 정상·고장 차량을 흉내 낸 오차 행렬을 난수로 만들어 진단 계산만 따로 시험할 수 있다.
난수 시드(seed)를 고정해 두어 언제 실행해도 같은 행렬이 나오므로, 결과를 저장해 둔 기대값과 비교할 수 있다.

쓰는 곳: test_diagnosis.py, make_diagnosis_golden.py (test_diagnosis.py가 비교할 기대값을 만들 때).
"""

import numpy as np

N_CELLS = 110  # QAS 차량의 셀 수
SEED = 20261015  # 고정 시드. 바꾸면 기대값 파일(fixtures/diagnosis_golden.npz)도 다시 만들어야 한다


def make_vehicle_errors(seed=SEED):
    """차량별 (U 오차, X 오차) 행렬을 만든다. U는 셀 전압, X는 셀 SOC 쪽 오차이며 각 행렬은 (시점 수, 셀 110개)다.

    쓰는 곳: test_diagnosis.py, make_diagnosis_golden.py의 build.

    Returns:
        train: 정상 학습 차량 4대
        test_normal: 정상 테스트 차량 3대 (1대는 100 시점 미만이라 이동평균을 건너뛰는 경로를 탄다)
        test_fault: 고장 테스트 차량 3대 (후반부에 셀 하나의 오차가 서서히 커진다)
    """
    rng = np.random.default_rng(seed)

    def vehicle(n_steps, fault_size=0.0):
        """차량 한 대의 (U 오차, X 오차). fault_size > 0이면 무작위 셀 하나의 오차를 후반부에 0에서 fault_size까지 키운다."""
        error_u = np.abs(rng.normal(0.0, 0.01, size=(n_steps, N_CELLS)))
        error_x = np.abs(rng.normal(0.0, 0.005, size=(n_steps, N_CELLS)))
        if fault_size:
            cell = int(rng.integers(N_CELLS))
            start = n_steps // 2
            error_u[start:, cell] += np.linspace(0.0, fault_size, n_steps - start)
            error_x[start:, cell] += np.linspace(0.0, fault_size / 2, n_steps - start)
        return error_u, error_x

    train = [vehicle(n) for n in (300, 260, 340, 280)]
    test_normal = [vehicle(n) for n in (250, 80, 270)]
    test_fault = [vehicle(n, size) for n, size in ((290, 0.08), (320, 0.02), (260, 0.005))]
    return train, test_normal, test_fault
