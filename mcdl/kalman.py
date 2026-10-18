"""칼만 필터로 배터리 팩·셀 상태를 추정하는 코드 (논문 원본 코드의 Function_.py에서 복원).

무엇을 하나
    셀 전압 측정값에서 팩의 대표 전압과 셀별 편차를 구하고(calculate_volt_modepi), 등가회로 모델과 칼만 필터로
    팩의 분극 전압·SOC와 셀별 SOC 편차를 시점마다 추정한다(solvers). MC-AE의 입력(vin_2·vin_3의 팩 추정값 x와
    셀 편차 추정값 Δ)이 이런 모델 기반 추정값이다. 이 저장소의 데이터(vin_2.pkl, vin_3.pkl)에는 그 결과가 이미
    계산되어 들어 있어, 학습·평가 스크립트는 이 파일을 쓰지 않는다.

모델
    팩: 1차 RC 등가회로. 상태 x = [분극 전압, SOC], 단자 전압 U = OCV(SOC, 온도) − 분극 전압 − RO·전류.
        적응형 확장 칼만 필터(EKF)로 추정한다. 오차가 커지면 공분산을 키우는 계수(λ, 1~1.5)를 쓴다.
    셀: 셀마다 SOC 편차 하나를 스칼라 칼만 필터로 추정한다. 셀 전압 편차 = OCV(팩 SOC + 셀 편차) − OCV(팩 SOC) − 전류·ΔR.
    OCV(s, T) = b[0] + b[1]·s + … + b[17]·s¹⁷ + b[18]·T + b[19]·T² + b[20]·T³ (s: SOC, T: 온도).

주의
    OCV 계수 b(21개)는 이 저장소와 데이터에 없다. 배터리별로 따로 구해 넣어야 한다.
    계산 순서는 원본과 같게 두었다(tests/test_kalman.py가 원본 코드로 만든 기대값과 bit 단위로 비교한다).
"""

from __future__ import annotations

import numpy as np


def calculate_volt_modepi(volt_all):
    """셀 전압들(시점 × 셀 DataFrame)에서 시점마다 가중 대표 전압(volt_modepi)과 셀별 편차(volt_di)를 구한다.

    셀마다 역가우스 분포 모양의 가중치를 주어, 평균에서 많이 벗어난 셀의 영향을 줄인 대표 전압을 만든다.
    돌려주는 값: (volt_modepi: 시점별 대표 전압 Series, volt_di: 셀 전압 − 대표 전압 DataFrame).
    쓰는 곳: solvers의 입력 준비 (이 저장소의 스크립트에서는 쓰지 않는다).
    """
    volt_mode = volt_all.mean(axis=1)
    volt_std = volt_all.std(axis=1)
    volt_lamda = 1 / volt_std

    volt_pi1 = (1 / (2 * np.pi * volt_all.pow(3))).mul(volt_lamda, axis=0).pow(0.5)
    volt_pi2 = (
        ((-1) * ((volt_all.sub(volt_mode, axis=0))).pow(2).mul(volt_lamda, axis=0))
        / (2 * volt_all.mul(volt_mode.pow(2), axis=0))
    ).apply(np.exp)
    volt_pi = volt_pi1 * volt_pi2

    volt_modepi = ((volt_pi * volt_all).sum(axis=1)) / (volt_pi.sum(axis=1))
    volt_di = volt_all.sub(volt_modepi, axis=0)

    return volt_modepi, volt_di


def _ocv(b, s, temp):
    """SOC s, 온도 temp에서의 OCV. 덧셈 순서는 원본 그대로(17차 항부터)다. 쓰는 곳: solvers."""
    return (
        b[17] * s ** 17
        + b[16] * s ** 16
        + b[15] * s ** 15
        + b[14] * s ** 14
        + b[13] * s ** 13
        + b[12] * s ** 12
        + b[11] * s ** 11
        + b[10] * s ** 10
        + b[9] * s ** 9
        + b[8] * s ** 8
        + b[7] * s ** 7
        + b[6] * s ** 6
        + b[5] * s ** 5
        + b[4] * s ** 4
        + b[3] * s ** 3
        + b[2] * s ** 2
        + b[1] * s
        + b[0]
        + b[18] * temp
        + b[19] * temp ** 2
        + b[20] * temp ** 3
    )


def _docv_dsoc(b, s):
    """OCV를 SOC로 미분한 값 (칼만 필터의 관측 행렬). 곱셈·덧셈 순서는 원본 그대로다. 쓰는 곳: solvers."""
    return (
        b[17] * s ** 16 * 17
        + b[16] * s ** 15 * 16
        + b[15] * s ** 14 * 15
        + b[14] * s ** 13 * 14
        + b[13] * s ** 12 * 13
        + b[12] * s ** 11 * 12
        + b[11] * s ** 10 * 11
        + b[10] * s ** 9 * 10
        + b[9] * s ** 8 * 9
        + b[8] * s ** 7 * 8
        + b[7] * 7 * s ** 6
        + b[6] * 6 * s ** 5
        + b[5] * 5 * s ** 4
        + b[4] * 4 * s ** 3
        + 3 * b[3] * s ** 2
        + 2 * b[2] * s
        + b[1]
    )


def solvers(volt_modepi, volt_di, volt_all, soc, b, current, temp_avg):
    """칼만 필터로 팩 상태(분극 전압, SOC)와 셀별 SOC 편차를 시점마다 추정한다.

    입력 (시점 수 N, 셀 수 n. Series·DataFrame은 0부터 시작하는 기본 index여야 한다)
        volt_modepi, volt_di: calculate_volt_modepi의 결과 (N Series, N × n DataFrame)
        volt_all: 셀 전압 (N × n DataFrame). soc: 측정 SOC (N, 배열 또는 Series)
        b: OCV 계수 21개 (모듈 설명의 식). current: 전류 (N Series). temp_avg: 평균 온도 (N Series)
    돌려주는 값 (원본 순서)
        xipre: 한 시점 앞의 셀별 SOC 편차 추정값 (N × n)
        xpre: 다음 시점 팩 상태 예측 [분극 전압, SOC] (2 × (N+1))
        Spre: 다음 시점 SOC 예측 (측정 SOC 기준, 길이 N+1 목록). Upre: 다음 시점 단자 전압 예측 (길이 N+1 목록)
        Xipre: 다음 시점 셀별 SOC 예측 ((N+1) × n). x: 팩 상태 추정값 [분극 전압, SOC] (2 × N)
        DUi: 셀별 전압 편차 추정값 (N × n). Xi: 셀별 SOC 추정값 (측정 SOC + 셀 편차, N × n)
    시점 1 ~ N−2를 계산한다(첫 시점은 초기값, 마지막 시점은 계산하지 않는다).
    쓰는 곳: 이 저장소의 스크립트에서는 쓰지 않는다 (OCV 계수 b가 데이터에 없다).
    """
    n_steps, num = volt_all.shape

    # 팩 EKF: 공분산 초기값, 과정·관측 잡음
    P = np.eye(2)
    Q = np.array([[1e-5, 0], [0, 1e-5]])
    R = 0.1
    # 적응 계수(λ)를 정하는 상수
    rou = 1.2
    beta = 0.15
    gamma = 1.5
    # 셀 칼만 필터: 공분산 초기값(1000), 과정·관측 잡음
    Pi = np.zeros((n_steps, num))
    Pi[0, 0:num] = 1000 * np.ones((1, num))
    Qi = 1e-4
    Ri = 2

    # 등가회로 상수: 옴 저항 RO, 분극 저항 RP, 분극 용량 CP, 셀별 저항 차이 DRi
    RO = 1e-3
    RP = 5.2203e-4
    CP = 5e3
    DRi = 3 * 10 ** (-6) * np.ones(num)
    t = RP * CP
    # 상태 방정식 x[k] = A·x[k-1] + B·전류 (분극 전압은 RC 감쇠, SOC는 전류 적산)
    A = np.array([[np.exp(-1 / t), 0], [0, 1]])
    B = np.array([(1 - np.exp(-1 / t)) * RP, -1 / (23 * 3600)])

    x = np.zeros((2, n_steps))
    xpre = np.zeros((2, n_steps + 1))
    x[:, 0] = np.array([0, soc[0]]).T
    xi = np.zeros((n_steps, num))
    Xi = np.zeros((n_steps, num))
    Xi[0, 0:num] = 0.01 * soc[0] * np.ones((1, num))
    Xipre = np.zeros((n_steps + 1, num))

    I = current
    temp = temp_avg
    OCV = [0] * n_steps
    OCVi = np.zeros((n_steps, num))
    DUi = np.zeros((n_steps, num))
    ei = np.zeros((n_steps, num))
    DUt = np.array(volt_di)
    C = np.zeros((n_steps, 2))
    U = [0] * n_steps
    Upre = [0] * (n_steps + 1)
    Spre = [0] * (n_steps + 1)
    Ut = volt_modepi  # 팩 EKF의 관측값: 대표 셀 전압
    V = [0] * n_steps
    e = [0] * n_steps

    for k in range(1, n_steps - 1):
        # 팩 EKF 예측
        x[:, k] = np.dot(A, x[:, k - 1]) + B * I.iloc[k - 1]
        if x[1, k] > 1:
            x[1, k] = 1
        OCV[k] = _ocv(b, x[1, k], temp.iloc[k])
        C[k, :] = [-1, _docv_dsoc(b, x[1, k])]
        U[k] = OCV[k] - x[0, k] - RO * I.iloc[k]
        e[k] = Ut.iloc[k] - U[k]

        # 적응 계수 λ: 최근 오차의 크기(V)가 예상보다 크면 예측 공분산을 키운다 (1 ~ 1.5)
        if k == 2:
            V[k] = e[k] * e[k]
        else:
            V[k] = (rou * V[k - 1] + e[k] * e[k]) / (1 + rou)
        N = V[k] - beta * R - np.dot(np.dot(C[k, :], Q), (np.dot(C[k, :], Q)).T)
        M = np.dot(np.dot(C[k, :], A), np.dot(np.dot(P, A.T), np.dot(C[k, :], A).T))
        Ek = N / M
        if gamma * Ek > 1 and gamma * Ek < 1.5:
            lambda_k = gamma * Ek
        elif gamma * Ek >= 1.5:
            lambda_k = 1.5
        else:
            lambda_k = 1

        # 팩 EKF 갱신과 다음 시점 예측
        P = lambda_k * (A.dot(P)).dot(A.T) + Q
        K = np.dot(P, C[k, :].T) / (np.dot(C[k, :], np.dot(P, C[k, :].T)) + R)
        x[:, k] = x[:, k] + np.dot(K, e[k])
        P = P - np.dot(np.dot(K, C[k, :]), P)
        xpre[:, k + 1] = np.dot(A, x[:, k]) + B * I.iloc[k]
        Spre[k + 1] = soc[k] + B[1] * I.iloc[k]
        Upre[k + 1] = OCV[k] - xpre[0, k + 1] - RO * I.iloc[k + 1]

        # 셀마다 SOC 편차를 스칼라 칼만 필터로 갱신한다
        for j in range(num):
            xi[k, j] = xi[k - 1, j]
            Pi[k, j] = Pi[k - 1, j] + Qi
            cell_soc = xi[k, j] + x[1, k]
            OCVi[k, j] = _ocv(b, cell_soc, temp.iloc[k])
            if OCVi[k, j] > OCV[k] + 0.1:
                OCVi[k, j] = OCV[k] + 0.1
            Ci = _docv_dsoc(b, cell_soc)
            DUi[k, j] = OCVi[k, j] - OCV[k] - I.iloc[k] * DRi[j]
            ei[k, j] = DUt[k, j] - DUi[k, j]
            Ki = (Pi[k, j] * Ci) / ((Ci * Pi[k, j] * Ci + Ri))
            xi[k, j] = xi[k, j] + Ki * ei[k, j]
            Xi[k, j] = soc[k] + xi[k, j]
            Xipre[k + 1, j] = xpre[1, k + 1] + xi[k, j]
            Pi[k, j] = Pi[k, j] - Ki * Ci * Pi[k, j]

    xipre = np.zeros((xi.shape[0], xi.shape[1]))
    xipre[1:, :] = xi[: xi.shape[0] - 1, :]
    return xipre, xpre, Spre, Upre, Xipre, x, DUi, Xi
