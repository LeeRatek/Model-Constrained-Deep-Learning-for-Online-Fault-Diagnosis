"""재구성·재구성 오차·진단 특징 테스트. 작은 CombinedAE(셀 4개)와 합성 차량 텐서를 쓴다."""

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

N_CELLS = 4


def small_model(q_dim, seed):
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
    """[x(2) | y(셀) | Δ(셀) | q] 열 구성의 합성 차량 텐서 (float64)."""
    generator = torch.Generator().manual_seed(seed)
    return torch.rand(n_steps, 2 + 2 * N_CELLS + q_dim, dtype=torch.float64, generator=generator) * 4


def manual_errors(model, tensor, batch_size):
    """기존 스크립트(Test_multi_models.run_model_inference)와 같은 방식의 참조 계산."""
    x, y = tensor[:, :2], tensor[:, 2 : 2 + N_CELLS]
    dx, q = tensor[:, 2 + N_CELLS : 2 + 2 * N_CELLS], tensor[:, 2 + 2 * N_CELLS :]
    out = []
    with torch.inference_mode():
        for start in range(0, len(tensor), batch_size):
            recon, _ = model(x[start : start + batch_size], dx[start : start + batch_size], q[start : start + batch_size])
            out.append(torch.abs(recon - y[start : start + batch_size]).numpy())
    return np.concatenate(out, axis=0)


def script_features(net, netx, u, x):
    """기존 Test_·Test_multi_models·Test_tsne의 테스트 차량 계산: 열을 직접 자르고 차량 전체를 4인자로 추론한 뒤 abs."""
    errors = []
    for model, tensor in ((net, u), (netx, x)):
        y = tensor[:, 2 : 2 + N_CELLS]
        with torch.inference_mode():
            recon = model(tensor[:, :2], tensor[:, 2 + N_CELLS : 2 + 2 * N_CELLS], tensor[:, 2 + 2 * N_CELLS :], y)
            errors.append(torch.abs(recon[0] - y).cpu().numpy())
    return diagnosis_features(*errors)


U_MODEL, X_MODEL = small_model(3, seed=1), small_model(4, seed=2)
TRAIN_U = [vehicle(n, 3, seed=10 + i) for i, n in enumerate((150, 90, 130))]
TRAIN_X = [vehicle(n, 4, seed=20 + i) for i, n in enumerate((150, 90, 130))]


def test_errors_match_reference_for_any_batch_size():
    tensor = TRAIN_U[0]
    for batch_size in (1000, 64, 7):
        errors = reconstruction_errors(U_MODEL, tensor, batch_size=batch_size, abs_err=True)
        assert errors.shape == (150, N_CELLS)
        assert np.array_equal(errors, manual_errors(U_MODEL, tensor, batch_size))


def test_signed_and_absolute_errors():
    signed = reconstruction_errors(U_MODEL, TRAIN_U[0], batch_size=64, abs_err=False)
    absolute = reconstruction_errors(U_MODEL, TRAIN_U[0], batch_size=64, abs_err=True)

    assert (signed < 0).any()
    assert np.array_equal(np.abs(signed), absolute)


def test_float32_option_rounds_errors():
    exact = reconstruction_errors(U_MODEL, TRAIN_U[0], batch_size=64, abs_err=True)
    rounded = reconstruction_errors(U_MODEL, TRAIN_U[0], batch_size=64, abs_err=True, float32=True)

    assert rounded.dtype == np.float32
    assert np.array_equal(rounded, exact.astype(np.float32))


def test_training_features_are_per_vehicle_diagnosis_features():
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
    """Test_multi_models(신호별 캐시)와 관리도 재계산(U·X 쌍)이 같은 학습 진단 특징을 낸다."""
    u = training_signal_features(U_MODEL, "U", TRAIN_U, batch_size=64, abs_err=True)
    x = training_signal_features(X_MODEL, "X", TRAIN_X, batch_size=64, abs_err=True)

    paired = training_features(U_MODEL, X_MODEL, zip(TRAIN_U, TRAIN_X), batch_size=64, abs_err=True)

    pd.testing.assert_frame_equal(combine_signal_features(u, x), paired, check_exact=True)


def test_reconstruct_returns_target_columns_and_whole_vehicle_by_default():
    tensor = TRAIN_U[1]

    y, recon = reconstruct(U_MODEL, tensor)

    assert torch.equal(y, tensor[:, 2 : 2 + N_CELLS])
    with torch.inference_mode():
        whole, _ = U_MODEL(tensor[:, :2], tensor[:, 2 + N_CELLS : 2 + 2 * N_CELLS], tensor[:, 2 + 2 * N_CELLS :])
    assert torch.equal(recon, whole)
    assert torch.equal(reconstruct(U_MODEL, tensor, batch_size=1000)[1], whole)  # 한 batch에 다 들어가면 같다


def test_vehicle_features_match_script_computation():
    for u, x in zip(TRAIN_U, TRAIN_X):
        features = vehicle_features(U_MODEL, X_MODEL, u, x, abs_err=True)

        pd.testing.assert_frame_equal(features, script_features(U_MODEL, X_MODEL, u, x), check_exact=True)
