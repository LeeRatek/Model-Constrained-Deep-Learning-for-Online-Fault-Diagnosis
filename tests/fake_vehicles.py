"""테스트용 가짜 원본 데이터 (QAS 열 구성 pkl)와 분할 .npy."""

import pickle

import numpy as np
import torch

N_STEPS = 160


def fake_vehicle(seed, n_steps=N_STEPS):
    """QAS 열 구성의 가짜 vin_2 (225열), vin_3 (226열). 전류 열(224)은 −500 이상이라 잘리지 않는다."""
    generator = torch.Generator().manual_seed(seed)
    vin_2 = torch.rand(n_steps, 225, dtype=torch.float64, generator=generator) * 0.5 + 3.5
    vin_3 = torch.rand(n_steps, 226, dtype=torch.float64, generator=generator)
    return vin_2, vin_3


def write_fake_vehicles(data_dir, ids):
    """data_dir/QAS/<id>/vin_2.pkl, vin_3.pkl을 만든다 (차량 ID를 seed로 쓴다)."""
    for vehicle_id in ids:
        folder = data_dir / "QAS" / str(vehicle_id)
        folder.mkdir(parents=True)
        for name, tensor in zip(("vin_2.pkl", "vin_3.pkl"), fake_vehicle(vehicle_id)):
            with open(folder / name, "wb") as f:
                pickle.dump(tensor, f)


def write_split(split_dir, split, ids, dtype=np.int64):
    np.save(split_dir / f"QAS_filtered_vehicle_ids_{split}.npy", np.asarray(ids, dtype=dtype))
