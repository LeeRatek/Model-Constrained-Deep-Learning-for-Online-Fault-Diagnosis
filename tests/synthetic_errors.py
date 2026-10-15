"""진단 계산 테스트용 합성 재구성 오차 (고정 seed, 원본 데이터 없이 사용)."""

import numpy as np

N_CELLS = 110
SEED = 20261015


def make_vehicle_errors(seed=SEED):
    """차량별 (U 오차, X 오차) 행렬. 각 행렬은 (시점 수, 셀 110개)다.

    Returns:
        train: 정상 학습 차량 4대
        test_normal: 정상 테스트 차량 3대 (1대는 100 시점 미만이라 이동평균을 건너뛰는 경로를 탄다)
        test_fault: 고장 테스트 차량 3대 (후반부에 셀 하나의 오차가 서서히 커진다)
    """
    rng = np.random.default_rng(seed)

    def vehicle(n_steps, fault_size=0.0):
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
