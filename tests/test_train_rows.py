"""Train_.py의 row_loader(학습 데이터를 batch 단위로 꺼내 주는 함수) 테스트.

row_loader는 학습 속도를 높이려고, 시점 하나씩 꺼내 batch로 쌓던 예전 방식을 batch 단위로 한 번에 자르는
방식으로 바꾼 것이다. 학습 결과가 예전과 똑같으려면 다음이 모두 같아야 한다.
    - 각 batch의 값, 자료형(float64), 메모리 배치(stride: 텐서 값이 메모리에 놓인 간격. 달라지면 계산 순서가
      바뀌어 마지막 자릿수가 달라질 수 있다)
    - 순회를 마친 뒤 전역 난수 발생기(RNG)의 상태. 섞기(shuffle)에 난수를 쓰므로, 꺼내 쓴 난수 개수가 다르면
      그다음 학습 단계의 난수가 모두 달라진다.
비교 기준은 예전 방식을 그대로 옮긴 PerRow다.
"""

import numpy as np
import pytest
import torch
from torch.utils.data import DataLoader, Dataset

from Train_ import row_loader

X_DIM, Y_DIM, Q_DIM = 2, 4, 3  # 가짜 차량 텐서의 열 폭: x 2개, 셀 4개(y와 Δ), q 3개
LENGTHS = (37, 5, 58)  # 짧은 차량을 끼워 배치가 차량 경계를 여러 번 걸치게 한다


class PerRow(Dataset):
    """비교 기준: 예전 Train_.py의 VehicleRows. 시점 하나씩 (x, y, Δ, q)를 float64(double)로 꺼낸다."""

    def __init__(self, tensors):
        """차량 텐서 목록을 받고, 차량별 시작 위치(누적 시점 수)를 계산해 둔다."""
        self.tensors = tensors
        self.cumulative = np.cumsum([0] + [t.shape[0] for t in tensors])

    def __len__(self):
        """전체 시점 수 (모든 차량의 시점 수 합)."""
        return self.cumulative[-1]

    def __getitem__(self, idx):
        """전체에서 idx번째 시점이 어느 차량의 몇 번째 시점인지 찾아, 그 행을 (x, y, Δ, q)로 나눠 돌려준다."""
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
    """시점 수가 LENGTHS인 가짜 차량 텐서 3개를 시드 0으로 만들어 dtype으로 돌려준다."""
    generator = torch.Generator().manual_seed(0)
    return [torch.rand(n, X_DIM + 2 * Y_DIM + Q_DIM, generator=generator).to(dtype) for n in LENGTHS]


def batches_and_next_rand(loader):
    """같은 시드(7)에서 loader의 batch를 모두 꺼내고, 그 직후 전역 난수 4개를 뽑아 함께 돌려준다.

    직후 난수가 같으면 두 loader가 난수를 같은 만큼 썼다는 뜻이다.
    """
    torch.manual_seed(7)
    batches = [tuple(t.clone() for t in batch) for batch in loader]
    return batches, torch.rand(4)


@pytest.mark.parametrize("dtype", [torch.float64, torch.float32])
@pytest.mark.parametrize("batch_size", [7, 1, 1000])
@pytest.mark.parametrize("shuffle", [False, True])
def test_row_loader_matches_per_row_loader(shuffle, batch_size, dtype):
    """row_loader와 예전 방식(PerRow + DataLoader)이 batch 값·자료형·stride·난수 사용량까지 같은지 본다.

    입력 자료형(float64, float32) × batch 크기(7, 1, 1000) × 섞기 여부의 모든 조합(12가지)을 돈다.
    """
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
