"""학습된 모델로 차량 데이터를 재구성하고, 재구성 오차와 진단 특징을 만드는 코드.

재구성과 재구성 오차
    U-model은 셀 전압을, X-model은 셀 SOC를 입력에서 "다시 계산"(재구성)한다. 정상 차량이면 재구성 값이
    실제 값과 비슷하고, 고장이 있으면 차이가 커진다. 이 차이(재구성 값 − 실제 값)를 재구성 오차라고 한다.

진단 특징
    재구성 오차(시점 × 셀)를 시점마다 숫자 6개로 요약한 것이다(계산은 mcdl/diagnosis.py).
    - 차량 한 대의 진단 특징: 평가할 차량의 이상 정도를 계산하는 데 쓴다.
    - 학습 진단 특징: 정상 학습 차량 여러 대의 진단 특징을 이어 붙인 것. PCA 관리도(정상 범위)를 만드는 데 쓴다.
    두 경우 모두 같은 함수(vehicle_features)로 계산하므로, 관리도와 평가가 항상 같은 방식의 특징을 쓴다.

누가 쓰나
    Train_.py(학습 끝에 관리도 만들기), Test_.py·Test_multi_models.py·Test_tsne.py(평가),
    mcdl/chart_source.py(관리도를 다시 계산할 때).
"""

from __future__ import annotations

import pandas as pd
import torch

from mcdl.diagnosis import diagnosis_features, signal_features


def reconstruct(model, vehicle_tensor, *, batch_size: int | None = None):
    """차량 한 대를 모델로 재구성해 (실제 값 y, 재구성 값 ŷ)를 돌려준다. 둘 다 (시점 × 셀) float64 텐서다.

    차량 텐서의 열은 [x | y | Δ | q] 순서다(mcdl/preprocessing.py의 get_input_dimensions). 각 구간의 폭은
    모델 크기에서 읽는다: x 폭 = 첫 번째 층(fc1)의 입력 수, 셀 수 = 출력 층(fc3)의 출력 수.
    모델이 있는 장치(CPU/GPU)에서 batch_size 시점씩 나눠 계산하고(None이면 차량 전체를 한 번에), ŷ는 CPU로 옮긴다.
    batch 크기에 따라 ŷ가 소수점 끝자리(ULP)에서 달라질 수 있어, 호출하는 쪽이 정하게 했다.
    쓰는 곳: 아래 reconstruction_errors, Test_tsne.py(t-SNE 그림에 실제 값과 재구성 값을 그릴 때).
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
    """차량 한 대의 재구성 오차(ŷ − y)를 (시점 × 셀) numpy 배열로 돌려준다.

    abs_err=True면 절댓값을 쓴다(기본 설정). float32=True면 메모리를 줄이려고 float32로 바꾼다
    (Test_multi_models.py --use-float32). batch_size의 뜻은 reconstruct와 같다.
    쓰는 곳: 아래 vehicle_features, training_signal_features.
    """
    y, recon = reconstruct(model, vehicle_tensor, batch_size=batch_size)
    error = recon - y
    if abs_err:
        error = torch.abs(error)
    if float32:
        error = error.float()
    return error.numpy()


def vehicle_features(net, netx, u_tensor, x_tensor, *, batch_size: int | None = None, abs_err: bool):
    """차량 한 대의 진단 특징(시점 × 6 DataFrame)을 계산한다.

    net(U-model)으로 U 텐서를, netx(X-model)로 X 텐서를 재구성해 재구성 오차를 구한 뒤
    mcdl/diagnosis.py의 diagnosis_features로 6개 특징을 만든다.
    abs_err는 학습 설정(sim_config의 no_abs_err)에 맞춰 넘겨야 관리도와 같은 방식이 된다.
    평가 스크립트는 차량 전체를 한 번에 추론하고(batch_size=None), 학습 쪽은 batch 크기를 정해 부른다.
    쓰는 곳: Test_.py, Test_multi_models.py, Test_tsne.py(평가 차량), 아래 training_features(학습 차량).
    """
    return diagnosis_features(
        reconstruction_errors(net, u_tensor, batch_size=batch_size, abs_err=abs_err),
        reconstruction_errors(netx, x_tensor, batch_size=batch_size, abs_err=abs_err),
    )


def training_features(net, netx, vehicles, *, batch_size: int, abs_err: bool) -> pd.DataFrame:
    """학습 진단 특징: 정상 학습 차량마다 진단 특징을 구해 위아래로 이어 붙인다 (관리도를 만드는 재료).

    vehicles는 (U 텐서, X 텐서) 쌍을 차례로 내주는 목록이나 반복자다. 반복자를 넘기면 차량을 한 대씩
    읽으면서 계산하므로 메모리를 적게 쓴다. 차량마다 따로 계산하는 이유는 이동평균 같은 계산이
    차량 경계를 넘어 섞이지 않게 하기 위해서다.
    쓰는 곳: Train_.py(학습 끝, batch 4096), mcdl/chart_source.py(관리도를 다시 계산할 때, batch 65536).
    """
    features = [
        vehicle_features(net, netx, u, x, batch_size=batch_size, abs_err=abs_err)
        for u, x in vehicles
    ]
    return pd.concat(features, axis=0, ignore_index=True)


def training_signal_features(
    model, signal: str, vehicle_tensors, *, batch_size: int, abs_err: bool, float32: bool = False
):
    """모델 하나(signal="U" 또는 "X")에 대한 학습 진단 특징 3개를 계산한다 (6개 중 그 모델 몫).

    Test_multi_models.py는 U-model 체크포인트와 X-model 체크포인트를 여러 개씩 조합해 평가한다. 조합마다
    6개 특징을 다시 계산하지 않도록, 체크포인트마다 자기 몫 3개만 한 번 계산해 캐시 파일(temp_cache/)에
    저장하고, 조합할 때 mcdl/diagnosis.py의 combine_signal_features로 6개를 맞춘다. 결과는
    training_features와 같다.
    돌려주는 값: pandas Series 3개의 튜플 (차량별 결과를 이어 붙인 것).
    쓰는 곳: Test_multi_models.py.
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
