"""학습 DataLoader(Train_.row_loader) 테스트. 시점마다 꺼내 collate로 쌓던 예전 방식과 배치·전역 RNG 소비가 같아야 한다."""

import numpy as np
import pytest
import torch
from torch.utils.data import DataLoader, Dataset

from Train_ import row_loader

X_DIM, Y_DIM, Q_DIM = 2, 4, 3
LENGTHS = (37, 5, 58)  # 짧은 차량을 끼워 배치가 차량 경계를 여러 번 걸치게 한다


class PerRow(Dataset):
    """예전 VehicleRows: 시점 하나씩 (x, y, Δ, q)를 double로 꺼낸다 (참조 구현)."""

    def __init__(self, tensors):
        self.tensors = tensors
        self.cumulative = np.cumsum([0] + [t.shape[0] for t in tensors])

    def __len__(self):
        return self.cumulative[-1]

    def __getitem__(self, idx):
        vehicle = np.searchsorted(self.cumulative, idx, side="right") - 1
        row = self.tensors[vehicle][idx - self.cumulative[vehicle]]
        y_start, dx_start, q_start = X_DIM, X_DIM + Y_DIM, X_DIM + 2 * Y_DIM
        return (
            row[:y_start].to(torch.double),
            row[y_start:dx_start].to(torch.double),
            row[dx_start:q_start].to(torch.double),
            row[q_start:].to(torch.double),
        )


def vehicles(dtype):
    generator = torch.Generator().manual_seed(0)
    return [torch.rand(n, X_DIM + 2 * Y_DIM + Q_DIM, generator=generator).to(dtype) for n in LENGTHS]


def batches_and_next_rand(loader):
    torch.manual_seed(7)
    batches = [tuple(t.clone() for t in batch) for batch in loader]
    return batches, torch.rand(4)


@pytest.mark.parametrize("dtype", [torch.float64, torch.float32])
@pytest.mark.parametrize("batch_size", [7, 1, 1000])
@pytest.mark.parametrize("shuffle", [False, True])
def test_row_loader_matches_per_row_loader(shuffle, batch_size, dtype):
    tensors = vehicles(dtype)

    expected, expected_rand = batches_and_next_rand(DataLoader(PerRow(tensors), batch_size=batch_size, shuffle=shuffle))
    actual, actual_rand = batches_and_next_rand(
        row_loader(tensors, X_DIM, Y_DIM, batch_size=batch_size, shuffle=shuffle)
    )

    assert len(actual) == len(expected)
    for got, want in zip(actual, expected):
        for a, b in zip(got, want):
            assert a.dtype == b.dtype == torch.float64
            assert a.stride() == b.stride()
            assert torch.equal(a, b)
    assert torch.equal(actual_rand, expected_rand)  # 순회 뒤 전역 RNG 상태도 같다
