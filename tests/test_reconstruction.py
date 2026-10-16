"""mcdl/reconstruction.py 테스트: 모델 재구성 값, 재구성 오차, 진단 특징 계산이 맞는지 확인한다.

실제 모델과 데이터 대신, 셀 4개짜리 작은 CombinedAE(Class_.py)와 난수로 만든 차량 텐서를 쓴다.
기준은 리팩터링 전 스크립트가 하던 계산 방식(manual_errors, script_features)이며, 결과가 완전히 같아야 한다.
"""

import numpy as np
import pandas as pd
import torch

from Class_ import CombinedAE, CustomSigmoidFunc
from mcdl.diagnosis import combine_signal_features, diagnosis_features
from mcdl.reconstruction import (
    reconstruct,
    reconstruction_errors,
    training_features,
    training_signal_features,
    vehicle_features,
)

N_CELLS = 4  # 테스트용 작은 모델의 셀 수


def small_model(q_dim, seed):
    """셀 4개짜리 작은 CombinedAE를 seed로 초기화해 float64·평가 모드로 돌려준다. q_dim은 q 입력 폭(U 3, X 4)."""
    torch.manual_seed(seed)
    model = CombinedAE(
        input_size=2,
        encode2_input_size=q_dim,
        output_size=N_CELLS,
        activation_fn=CustomSigmoidFunc(scale=1.8, shift=2.5),
        use_dx_in_forward=True,
        add_one_output_layer=True,
    )
    return model.double().eval()


def vehicle(n_steps, q_dim, seed):
    """[x(2) | y(셀) | Δ(셀) | q] 열 구성의 합성 차량 텐서 (float64, 값 0~4). 실제 vin_2·vin_3와 열 순서가 같다."""
    generator = torch.Generator().manual_seed(seed)
    return torch.rand(n_steps, 2 + 2 * N_CELLS + q_dim, dtype=torch.float64, generator=generator) * 4


def manual_errors(model, tensor, batch_size):
    """비교 기준이 되는 재구성 오차 계산. 리팩터링 전 Test_multi_models.py의 run_model_inference가 하던 방식
    (열을 직접 자르고 batch_size씩 나눠 |재구성 − 실제|)을 그대로 옮겼다."""
    x, y = tensor[:, :2], tensor[:, 2 : 2 + N_CELLS]
    dx, q = tensor[:, 2 + N_CELLS : 2 + 2 * N_CELLS], tensor[:, 2 + 2 * N_CELLS :]
    out = []
    with torch.inference_mode():
        for start in range(0, len(tensor), batch_size):
            recon, _ = model(x[start : start + batch_size], dx[start : start + batch_size], q[start : start + batch_size])
            out.append(torch.abs(recon - y[start : start + batch_size]).numpy())
    return np.concatenate(out, axis=0)


def script_features(net, netx, u, x):
    """비교 기준이 되는 테스트 차량 진단 특징 계산. 리팩터링 전 Test_·Test_multi_models·Test_tsne가 하던 방식
    (열을 직접 자르고, 차량 전체를 한 번에 모델에 넣고, 오차의 절댓값)을 그대로 옮겼다."""
    errors = []
    for model, tensor in ((net, u), (netx, x)):
        y = tensor[:, 2 : 2 + N_CELLS]
        with torch.inference_mode():
            recon = model(tensor[:, :2], tensor[:, 2 + N_CELLS : 2 + 2 * N_CELLS], tensor[:, 2 + 2 * N_CELLS :])
            errors.append(torch.abs(recon[0] - y).cpu().numpy())
    return diagnosis_features(*errors)


U_MODEL, X_MODEL = small_model(3, seed=1), small_model(4, seed=2)
# 가짜 학습 차량 3대 (시점 수 150, 90, 130). U는 셀 전압용, X는 셀 SOC용 텐서
TRAIN_U = [vehicle(n, 3, seed=10 + i) for i, n in enumerate((150, 90, 130))]
TRAIN_X = [vehicle(n, 4, seed=20 + i) for i, n in enumerate((150, 90, 130))]


def test_errors_match_reference_for_any_batch_size():
    """reconstruction_errors가 batch 크기(1000, 64, 7)와 상관없이 기준 계산(manual_errors)과 같은 오차를 내는지 본다."""
    tensor = TRAIN_U[0]
    for batch_size in (1000, 64, 7):
        errors = reconstruction_errors(U_MODEL, tensor, batch_size=batch_size, abs_err=True)
        assert errors.shape == (150, N_CELLS)
        assert np.array_equal(errors, manual_errors(U_MODEL, tensor, batch_size))


def test_signed_and_absolute_errors():
    """abs_err=False면 부호 있는 오차(음수 포함)를, True면 그 절댓값을 돌려주는지 본다."""
    signed = reconstruction_errors(U_MODEL, TRAIN_U[0], batch_size=64, abs_err=False)
    absolute = reconstruction_errors(U_MODEL, TRAIN_U[0], batch_size=64, abs_err=True)

    assert (signed < 0).any()
    assert np.array_equal(np.abs(signed), absolute)


def test_float32_option_rounds_errors():
    """float32=True면 오차를 float32로 바꿔(정밀도를 낮춰) 돌려주는지 본다."""
    exact = reconstruction_errors(U_MODEL, TRAIN_U[0], batch_size=64, abs_err=True)
    rounded = reconstruction_errors(U_MODEL, TRAIN_U[0], batch_size=64, abs_err=True, float32=True)

    assert rounded.dtype == np.float32
    assert np.array_equal(rounded, exact.astype(np.float32))


def test_training_features_are_per_vehicle_diagnosis_features():
    """training_features가 차량마다 진단 특징을 구해 이어 붙인 것과 같은지 본다(차량 경계를 넘어 섞이지 않아야 한다)."""
    features = training_features(U_MODEL, X_MODEL, zip(TRAIN_U, TRAIN_X), batch_size=64, abs_err=True)

    expected = pd.concat(
        [
            diagnosis_features(manual_errors(U_MODEL, u, 64), manual_errors(X_MODEL, x, 64))
            for u, x in zip(TRAIN_U, TRAIN_X)
        ],
        axis=0,
        ignore_index=True,
    )
    pd.testing.assert_frame_equal(features, expected, check_exact=True)


def test_signal_path_matches_paired_path():
    """U·X 신호별로 따로 구해 묶은 특징(Test_multi_models.py의 신호별 캐시 방식)과
    U·X를 쌍으로 한 번에 구한 특징(관리도를 다시 계산할 때의 방식)이 완전히 같은지 본다."""
    u = training_signal_features(U_MODEL, "U", TRAIN_U, batch_size=64, abs_err=True)
    x = training_signal_features(X_MODEL, "X", TRAIN_X, batch_size=64, abs_err=True)

    paired = training_features(U_MODEL, X_MODEL, zip(TRAIN_U, TRAIN_X), batch_size=64, abs_err=True)

    pd.testing.assert_frame_equal(combine_signal_features(u, x), paired, check_exact=True)


def test_reconstruct_returns_target_columns_and_whole_vehicle_by_default():
    """reconstruct가 (실제 셀 값 y, 재구성 값)을 돌려주고, 기본값이면 차량 전체를 한 번에 모델에 넣은 결과와 같은지 본다."""
    tensor = TRAIN_U[1]

    y, recon = reconstruct(U_MODEL, tensor)

    assert torch.equal(y, tensor[:, 2 : 2 + N_CELLS])
    with torch.inference_mode():
        whole, _ = U_MODEL(tensor[:, :2], tensor[:, 2 + N_CELLS : 2 + 2 * N_CELLS], tensor[:, 2 + 2 * N_CELLS :])
    assert torch.equal(recon, whole)
    assert torch.equal(reconstruct(U_MODEL, tensor, batch_size=1000)[1], whole)  # 한 batch에 다 들어가면 같다


def test_vehicle_features_match_script_computation():
    """vehicle_features가 리팩터링 전 평가 스크립트의 계산(script_features)과 완전히 같은 진단 특징을 내는지 본다."""
    for u, x in zip(TRAIN_U, TRAIN_X):
        features = vehicle_features(U_MODEL, X_MODEL, u, x, abs_err=True)

        pd.testing.assert_frame_equal(features, script_features(U_MODEL, X_MODEL, u, x), check_exact=True)
