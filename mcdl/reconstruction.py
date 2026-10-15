"""재구성 오차와 진단 특징.

U-model·X-model로 차량 텐서 [x | y | Δ | q]를 재구성해 셀별 재구성 오차를 구하고, 차량 한 대의 진단 특징과
정상 학습 차량들의 학습 진단 특징을 만든다. 학습 쪽과 테스트 쪽이 같은 함수(vehicle_features)를 쓴다.
용어는 GLOSSARY.md를 따른다.
"""

from __future__ import annotations

import pandas as pd
import torch

from mcdl.diagnosis import diagnosis_features, signal_features


def reconstruct(model, vehicle_tensor, *, batch_size: int | None = None):
    """한 차량의 (재구성 대상 y, 재구성 결과 ŷ). 둘 다 (시점, 셀) double 텐서이고 ŷ는 CPU로 옮긴다.

    차량 텐서의 열 구성 [x | y | Δ | q]는 모델 크기(fc1 입력, fc3 출력)에서 읽는다. 모델이 있는 device에서
    batch_size 단위로 추론한다 (None이면 차량 전체를 한 번에). batch 크기에 따라 ŷ가 ULP 수준에서
    달라지므로 호출자가 정한다.
    """
    x_dim = model.fc1.in_features
    y_dim = model.fc3.out_features
    x = vehicle_tensor[:, :x_dim].double()
    y = vehicle_tensor[:, x_dim : x_dim + y_dim].double()
    dx = vehicle_tensor[:, x_dim + y_dim : x_dim + 2 * y_dim].double()
    q = vehicle_tensor[:, x_dim + 2 * y_dim :].double()
    device = next(model.parameters()).device
    size = batch_size or x.shape[0]

    recons = []
    with torch.inference_mode():
        for start in range(0, x.shape[0], size):
            end = min(start + size, x.shape[0])
            recon, _ = model(
                x[start:end].to(device), dx[start:end].to(device), q[start:end].to(device)
            )
            recons.append(recon.cpu())
    return y, torch.cat(recons, dim=0)


def reconstruction_errors(
    model, vehicle_tensor, *, batch_size: int | None, abs_err: bool, float32: bool = False
):
    """한 차량의 재구성 오차 ŷ - y (시점, 셀) numpy 배열. abs_err이면 절댓값이다.

    batch_size는 reconstruct와 같다. 오차는 원소별 계산이라 batch를 나눠도 ŷ만 영향을 받는다.
    float32이면 오차를 float32로 바꿔 돌려준다 (Test_multi_models --use-float32).
    """
    y, recon = reconstruct(model, vehicle_tensor, batch_size=batch_size)
    error = recon - y
    if abs_err:
        error = torch.abs(error)
    if float32:
        error = error.float()
    return error.numpy()


def vehicle_features(net, netx, u_tensor, x_tensor, *, batch_size: int | None = None, abs_err: bool):
    """차량 한 대의 진단 특징: U·X 재구성 오차 → diagnosis_features.

    학습 진단 특징(training_features)과 테스트 차량 평가가 모두 이 함수를 쓴다. 테스트 쪽은 차량 전체를
    한 번에 추론하고(batch_size=None), 학습 쪽은 batch 크기를 정해 부른다.
    """
    return diagnosis_features(
        reconstruction_errors(net, u_tensor, batch_size=batch_size, abs_err=abs_err),
        reconstruction_errors(netx, x_tensor, batch_size=batch_size, abs_err=abs_err),
    )


def training_features(net, netx, vehicles, *, batch_size: int, abs_err: bool) -> pd.DataFrame:
    """학습 진단 특징: (U 텐서, X 텐서) 쌍마다 vehicle_features를 구해 이어 붙인다.

    vehicles는 iterator여도 된다 (원본을 한 번만 읽으면서 처리할 수 있다).
    """
    features = [
        vehicle_features(net, netx, u, x, batch_size=batch_size, abs_err=abs_err)
        for u, x in vehicles
    ]
    return pd.concat(features, axis=0, ignore_index=True)


def training_signal_features(
    model, signal: str, vehicle_tensors, *, batch_size: int, abs_err: bool, float32: bool = False
):
    """한 신호("U"/"X")의 학습 진단 특징 3개: 차량마다 signal_features를 구해 신호별로 이어 붙인다.

    반환값(Series 3개)은 temp_cache 파일 형식과 같고, combine_signal_features로 진단 특징 열 순서가 된다.
    """
    per_vehicle = [
        signal_features(
            signal,
            reconstruction_errors(
                model, tensor, batch_size=batch_size, abs_err=abs_err, float32=float32
            ),
        )
        for tensor in vehicle_tensors
    ]
    return tuple(
        pd.concat([feats[i] for feats in per_vehicle], ignore_index=True) for i in range(3)
    )
