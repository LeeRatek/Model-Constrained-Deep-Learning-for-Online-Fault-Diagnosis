"""팩 상태 예측 LSTM(Class_.py의 LSTM)의 학습 데이터 만들기, 모델 저장·불러오기 (논문 원본 코드에서 복원).

학습 데이터
    차량마다 vin_1.pkl에 팩 단위 측정값 7개가 시점마다 들어 있다(모양: 시점 × 1 × 7, 0~1로 정규화된 값).
    원본 코드 주석에 따르면 마지막 두 열이 대표 셀 전압(volt_modepi)과 SOC다. LSTM은 한 시점 전의 7개 값으로
    현재 시점의 이 두 값을 예측하도록 학습한다(prepare_training_data).

누가 쓰나
    Train_.py(--lstm-training: 학습 차량 전체로 LSTM 하나를 학습해 Run 폴더 artifact/lstm.pth로 저장).
"""

from __future__ import annotations

import os

import pandas as pd
import torch

from Class_ import LSTM_INPUT_SIZE

LSTM_TIME_STEP = 1  # LSTM에 한 번에 넣는 시점 수


def series_to_supervised(data, n_in=1, n_out=1, dropnan=True):
    """시계열(시점 × 변수)을 지도학습용 표로 바꾼다: 과거 n_in 시점의 값(입력)과 현재부터 n_out 시점의 값(정답)을 한 행에.

    열 이름은 "var1(t-1)"(변수 1의 한 시점 전), "var1(t)"(현재)처럼 붙는다. dropnan=True면 앞뒤로 밀면서 생긴
    빈칸(NaN)이 있는 행을 지운다. 출처: https://machinelearningmastery.com/convert-time-series-supervised-learning-problem-python/
    쓰는 곳: prepare_training_data.
    """
    n_vars = 1 if type(data) is list else data.shape[1]
    df = pd.DataFrame(data)
    cols, names = list(), list()
    # 입력: t-n_in, …, t-1
    for i in range(n_in, 0, -1):
        cols.append(df.shift(i))
        names += [("var%d(t-%d)" % (j + 1, i)) for j in range(n_vars)]
    # 정답: t, t+1, …, t+n_out-1
    for i in range(0, n_out):
        cols.append(df.shift(-i))
        if i == 0:
            names += [("var%d(t)" % (j + 1)) for j in range(n_vars)]
        else:
            names += [("var%d(t+%d)" % (j + 1, i)) for j in range(n_vars)]
    agg = pd.concat(cols, axis=1)
    agg.columns = names
    if dropnan:
        agg.dropna(inplace=True)
    return agg


def prepare_training_data(vin1, device):
    """차량 한 대의 vin_1 텐서(시점 × 1 × 7)로 LSTM 학습 쌍을 만든다.

    입력 train_X: 시점 t-1의 7개 값 (시점 수 − 1, 1, 7). 정답 train_y: 시점 t의 마지막 두 열(대표 셀 전압, SOC)
    (시점 수 − 1, 1, 2). 둘 다 float64로 device에 올린다.
    쓰는 곳: Train_.py의 train_lstm (차량마다 따로 만들어 이어 붙이므로 차량 경계를 넘는 쌍은 생기지 않는다).
    """
    vin1_df = pd.DataFrame(vin1.cpu().detach().numpy()[:, 0, :])
    reframed = series_to_supervised(vin1_df, 1, 1)
    # 현재 시점(t)의 앞 다섯 열은 정답이 아니므로 버린다 → [t-1의 7개 | t의 마지막 2개]
    reframed.drop(reframed.columns[LSTM_INPUT_SIZE : LSTM_INPUT_SIZE * 2 - 2], axis=1, inplace=True)
    train = reframed.values
    train_X, train_y = train[:, :-2], train[:, -2:]
    batch_train = int(reframed.shape[0] / LSTM_TIME_STEP)

    train_X = torch.tensor(train_X).reshape(batch_train, LSTM_TIME_STEP, LSTM_INPUT_SIZE).to(device)
    train_y = torch.tensor(train_y.reshape(-1, 2)).reshape(batch_train, LSTM_TIME_STEP, 2).to(device)
    return train_X, train_y


def save_lstm_model(model, models_dir, filename="lstm.pth"):
    """학습한 LSTM을 CPU로 옮겨 모델 객체째 models_dir/filename에 저장한다 (원본 코드와 같은 저장 방식).

    모델 객체째 저장하므로 불러올 때 Class_.py의 LSTM 클래스가 있어야 한다. 쓰는 곳: Train_.py.
    """
    os.makedirs(models_dir, exist_ok=True)
    torch.save(model.to(torch.device("cpu")), os.path.join(models_dir, filename))


def load_lstm_model(models_dir, filename="lstm.pth", device=None):
    """save_lstm_model로 저장한 LSTM을 CPU로 불러온다. device를 주면 그 장치로 옮긴다.

    예: lstm = load_lstm_model("models/<RUN_ID>/artifact"); 예측 = lstm(입력) (입력 모양은 prepare_training_data의 train_X).
    쓰는 곳: 테스트(저장 결과 확인). 학습·평가 스크립트에서는 쓰지 않는다.
    """
    model = torch.load(os.path.join(models_dir, filename), map_location=torch.device("cpu"), weights_only=False)
    if device is not None:
        model = model.to(device)
    return model
