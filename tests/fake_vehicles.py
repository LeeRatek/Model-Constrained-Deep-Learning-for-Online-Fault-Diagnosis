"""테스트용 가짜 차량 데이터를 만드는 도우미. 테스트 파일이 아니라 테스트가 불러 쓰는 함수 모음이다.

실제 차량 데이터(data/ 폴더)는 크고 저장소에 없으므로, 테스트는 이 파일로 실제와 열 개수만 같은
무작위 값 파일을 임시 폴더에 만들어 쓴다. 값 자체에는 의미가 없고, 파일을 읽고 나누고 계산하는 흐름이
끝까지 돌아가는지 확인하는 용도다. 차량 ID를 난수 시드로 쓰므로 같은 ID는 언제나 같은 값이 나온다.

    fake_vehicle: 차량 한 대의 가짜 vin_2(셀 전압용), vin_3(셀 SOC용) 텐서
    fake_vin1: 차량 한 대의 가짜 vin_1(LSTM용 팩 측정값 7개) 텐서
    write_fake_vehicles: 가짜 차량들을 data/QAS/<ID>/vin_1.pkl, vin_2.pkl, vin_3.pkl 파일로 저장
    write_split: 분할(train/validate/test/fault) 차량 목록 .npy 저장 (data_filter.py가 만드는 것과 같은 형식)

쓰는 곳: test_vehicles.py, test_chart_source.py, test_scripts.py.
"""

import pickle

import numpy as np
import torch

N_STEPS = 160  # 가짜 차량 한 대의 시점 수


def fake_vehicle(seed, n_steps=N_STEPS):
    """QAS 열 구성의 가짜 vin_2 (225열, 값 3.5~4.0), vin_3 (226열, 값 0~1)을 seed로 만들어 돌려준다.

    전처리는 전류(열 224)가 −500보다 작았던 마지막 시점까지를 충전 준비 구간으로 보고 잘라 낸다. 가짜 데이터는
    전류 열이 모두 −500 이상이라 잘리는 부분이 없고, 시점 수가 그대로 남는다.
    쓰는 곳: write_fake_vehicles, test_vehicles.py·test_chart_source.py (파일을 거치지 않고 기대값을 직접 만들 때).
    """
    generator = torch.Generator().manual_seed(seed)
    vin_2 = torch.rand(n_steps, 225, dtype=torch.float64, generator=generator) * 0.5 + 3.5
    vin_3 = torch.rand(n_steps, 226, dtype=torch.float64, generator=generator)
    return vin_2, vin_3


def fake_vin1(seed, n_steps=N_STEPS):
    """실제 vin_1.pkl과 같은 모양(시점 × 1 × 7, 값 0~1)의 가짜 LSTM 입력을 만든다.

    vin_2·vin_3와 다른 난수 시드(seed + 100000)를 써서, 이 함수를 더해도 기존 가짜 vin_2·vin_3 값은 그대로다.
    쓰는 곳: write_fake_vehicles, test_lstm.py.
    """
    generator = torch.Generator().manual_seed(seed + 100000)
    return torch.rand(n_steps, 1, 7, dtype=torch.float64, generator=generator)


def write_fake_vehicles(data_dir, ids):
    """data_dir/QAS/<id>/vin_1.pkl, vin_2.pkl, vin_3.pkl을 만든다 (차량 ID를 seed로 쓴다). 실제 원본과 같은 폴더 구조다.

    쓰는 곳: test_vehicles.py, test_chart_source.py, test_scripts.py.
    """
    for vehicle_id in ids:
        folder = data_dir / "QAS" / str(vehicle_id)
        folder.mkdir(parents=True)
        tensors = (fake_vin1(vehicle_id), *fake_vehicle(vehicle_id))
        for name, tensor in zip(("vin_1.pkl", "vin_2.pkl", "vin_3.pkl"), tensors):
            with open(folder / name, "wb") as f:
                pickle.dump(tensor, f)


def write_split(split_dir, split, ids, dtype=np.int64):
    """분할 split(train, validate, test, fault)의 차량 ID 목록을 split_dir/QAS_filtered_vehicle_ids_<split>.npy로 저장한다.

    dtype=np.float64로 주면 datasets/ 아래 실제 분할 파일처럼 ID를 실수형으로 저장한다(data_filter.py가 그렇게 저장한다).
    쓰는 곳: test_vehicles.py, test_chart_source.py, test_scripts.py.
    """
    np.save(split_dir / f"QAS_filtered_vehicle_ids_{split}.npy", np.asarray(ids, dtype=dtype))
