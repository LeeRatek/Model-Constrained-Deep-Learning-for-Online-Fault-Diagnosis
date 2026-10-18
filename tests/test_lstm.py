"""mcdl/lstm.py와 Class_.py의 LSTM 테스트: 학습 데이터 구성, 모델 출력 모양, 저장·불러오기를 확인한다.

LSTM은 vin_1.pkl(시점 × 1 × 7)의 한 시점 전 값 7개로 현재 시점의 마지막 두 열(대표 셀 전압, SOC)을 예측한다.
실제 데이터 대신 같은 모양의 가짜 vin_1(fake_vehicles.py)을 쓴다.
"""

import numpy as np
import torch

from Class_ import LSTM
from fake_vehicles import fake_vin1
from mcdl.lstm import load_lstm_model, prepare_training_data, save_lstm_model, series_to_supervised


def test_series_to_supervised_pairs_previous_and_current_values():
    """series_to_supervised가 [한 시점 전 값들 | 현재 값들]을 한 행에 놓고, 앞이 비는 첫 행을 지우는지 본다."""
    table = series_to_supervised(np.array([[1.0, 10.0], [2.0, 20.0], [3.0, 30.0]]), n_in=1, n_out=1)

    assert list(table.columns) == ["var1(t-1)", "var2(t-1)", "var1(t)", "var2(t)"]
    assert table.values.tolist() == [[1.0, 10.0, 2.0, 20.0], [2.0, 20.0, 3.0, 30.0]]


def test_training_pairs_predict_next_voltage_and_soc():
    """학습 입력은 시점 t-1의 7개 값, 정답은 시점 t의 마지막 두 열(대표 셀 전압, SOC)인지 본다."""
    vin1 = fake_vin1(0, n_steps=30)

    train_X, train_y = prepare_training_data(vin1, "cpu")

    assert train_X.shape == (29, 1, 7) and train_y.shape == (29, 1, 2)
    assert train_X.dtype == train_y.dtype == torch.float64
    assert torch.equal(train_X[:, 0, :], vin1[:-1, 0, :])
    assert torch.equal(train_y[:, 0, :], vin1[1:, 0, 5:7])


def test_lstm_outputs_two_values_per_step():
    """LSTM이 (batch, 시점, 7) 입력에 (batch, 시점, 2) 예측을 내는지 본다."""
    torch.manual_seed(0)
    lstm = LSTM().double()

    assert lstm(torch.rand(5, 1, 7, dtype=torch.float64)).shape == (5, 1, 2)
    assert lstm(torch.rand(3, 4, 7, dtype=torch.float64)).shape == (3, 4, 2)


def test_saved_lstm_loads_with_same_predictions(tmp_path):
    """save_lstm_model로 저장한 모델을 load_lstm_model로 불러오면 같은 예측을 내는지 본다."""
    torch.manual_seed(0)
    lstm = LSTM().double()
    inputs = torch.rand(8, 1, 7, dtype=torch.float64)
    expected = lstm(inputs)

    save_lstm_model(lstm, tmp_path / "artifact")
    loaded = load_lstm_model(tmp_path / "artifact")

    assert torch.equal(loaded(inputs), expected)
