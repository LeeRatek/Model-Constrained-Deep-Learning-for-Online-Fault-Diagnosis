"""모델 정의: 셀 전압·셀 SOC를 재구성하는 MC-AE(CombinedAE)와 그 출력 함수(CustomSigmoidFunc).

MC-AE는 Model-Constrained AutoEncoder의 줄임말이다. 입력을 받아 셀 값을 "다시 계산"(재구성)하도록 학습하는
작은 신경망으로, 논문 Fig. 1(b)(c)의 네트워크와 Fig. 1(d)의 directed linear layer에 해당한다.
같은 클래스로 두 모델을 만든다.
    U-model: 셀 전압을 재구성한다 (입력 vin_2.pkl, 체크포인트 파일 net*.pth)
    X-model: 셀 SOC를 재구성한다 (입력 vin_3.pkl, 체크포인트 파일 netx*.pth)

누가 쓰나
    모델은 직접 만들지 않고 mcdl/run.py의 build_model_pair(새 모델), Run.load_model(저장된 모델 불러오기)로 만든다.
    학습은 Train_.py, 재구성 계산은 mcdl/reconstruction.py가 한다.
"""

import torch
import torch.nn as nn


class CustomSigmoidFunc:
    """디코더 출력에 쓰는 함수: shift + scale × sigmoid(x). 출력 범위가 shift ~ shift + scale이 된다.

    예: U-model은 scale 1.8, shift 2.5로 출력이 2.5~4.3 V(셀 전압 범위), X-model은 scale 1, shift 0으로 0~1(SOC).
    0~1 정규화를 쓰는 learning_case에서는 두 모델 모두 scale 1, shift 0이다(mcdl/run.py의 _build_model).
    """

    # The authors use scale default = 1.8, shift default = 2.5
    # We set scale = 1, shift = 0 as default values for general use.
    def __init__(self, scale=1, shift=0):
        """출력 범위를 정하는 scale(폭)과 shift(시작값)를 저장한다."""
        self.scale = scale
        self.shift = shift

    def forward(self, x):
        """x에 sigmoid를 적용한 뒤 출력 범위(shift ~ shift + scale)로 늘리고 옮긴다."""
        return self.shift + self.scale * torch.sigmoid(x)


class CombinedAE(nn.Module):
    """MC-AE 본체. 입력 (x, Δ, q)로 셀 값 ŷ를 재구성한다.

    계산식 (README 2장)
        z = fc1(x) + sigmoid(fc2(q)) + Δ      x: 팩 추정값 2개, q: 온도·SOC·전류 등, Δ: 셀별 편차 추정값
        z = fc4(z)                            add_one_output_layer일 때만
        ŷ = shift + scale · sigmoid(fc3(z))   셀 수만큼의 재구성 값
    fc1의 입력 수(x 폭)와 fc3의 출력 수(셀 수)는 mcdl/reconstruction.py와 Train_.py가 차량 텐서를 나눌 때
    읽어 쓰므로, 층 이름과 크기의 의미를 바꾸면 그쪽도 함께 고쳐야 한다.
    """

    def __init__(
        self,
        input_size,
        encode2_input_size,
        output_size,
        use_dx_in_forward: bool = True,
        add_one_output_layer: bool = False,
        activation_fn: CustomSigmoidFunc = None,
    ):
        """층을 만든다. input_size = x 폭(2), encode2_input_size = q 폭(3 또는 4), output_size = 셀 수(110 또는 85).

        use_dx_in_forward=False면 Δ를 0으로 바꿔 쓰지 않는다(--no-use-dx). activation_fn은 CustomSigmoidFunc다.
        """
        super(CombinedAE, self).__init__()
        self.fc1 = nn.Linear(input_size, 1)
        self.fc2 = nn.Linear(encode2_input_size, 1)
        self.fc3 = nn.Linear(output_size, output_size)
        self.activation_fn = activation_fn
        self.use_dx_in_forward = use_dx_in_forward
        self.add_one_output_layer = add_one_output_layer
        if self.add_one_output_layer:
            self.fc4 = nn.Linear(output_size, output_size)

    def encode(self, x):
        """팩 추정값 x(2개)를 값 하나로 줄인다 (fc1)."""
        return self.fc1(x)

    def encode2(self, x):
        """q(온도·SOC·전류 등)를 0~1 사이 값 하나로 줄인다 (fc2 + sigmoid)."""
        return torch.sigmoid(self.fc2(x))

    def decode(self, z):
        """셀별 잠재값 z를 셀 값 범위의 재구성 값으로 바꾼다 (fc3 + CustomSigmoidFunc)."""
        return self.activation_fn.forward(self.fc3(z))

    def forward(self, x, dx, q, y=None):
        """재구성을 계산해 (재구성 값 ŷ, 잠재값 z)를 돌려준다. y는 쓰지 않는다(예전 호출 형태와 맞추려고 남겨 둔 인자).

        쓰는 곳: Train_.py의 train_autoencoder(학습), mcdl/reconstruction.py의 reconstruct(평가).
        """
        if not self.use_dx_in_forward:
            dx = dx * 0.0

        z = self.encode(x) + self.encode2(q) + dx
        if self.add_one_output_layer:
            z = self.fc4(z)
        re = self.decode(z)
        return re, z
