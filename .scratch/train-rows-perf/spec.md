# Spec: 학습 데이터 로딩 성능 개선

**Status:** resolved

출처: 2026-10-15 6번 후보(Train_ 학습 루프) 뒤의 제안과 같은 날 grilling(P1, P2). 용어는 `GLOSSARY.md`를 따른다.

## Problem Statement

Train_의 학습·검증 DataLoader는 시점(행)마다 `VehicleRows.__getitem__`을 부른다. 각 호출은 이진 탐색, 행 인덱싱, 열 슬라이스 4개, dtype 변환을 하고, 그 결과를 collate로 쌓는다. 차량 20대(60만 행) 실측에서 한 바퀴 순회에 약 17.5초가 걸렸다. 실데이터 학습은 epoch당 약 636만 행이므로, 데이터 로딩에만 epoch당 약 3분이 드는 것으로 추정한다.

## Solution

DataLoader가 배치 인덱스를 한 번에 넘기게 한다(`sampler=BatchSampler(…)`, `batch_size=None`). `VehicleRows`는 배치 하나를 한 번에 모아 온다. 배치 값, 메모리 배치(stride), 전역 난수 소비는 지금과 bit 단위로 같다. 따라서 학습 결과도 같다.

## User Stories

1. 연구자로서, 학습 시간이 줄되 결과(체크포인트, loss, 관리도)는 bit 단위로 같길 원한다.
2. 연구자로서, shuffle을 켜도 같은 시드면 예전과 같은 결과를 원한다.
3. 개발자로서, 새 loader가 예전 행 단위 loader와 같은 배치를 낸다는 것을 테스트로 보고 싶다.

## Implementation Decisions

**실측 근거 (차량 20대, 60만 행, batch 100)**

| 방식 | shuffle 끔 | shuffle 켬 |
|---|---|---|
| 행 단위 (지금) | 17.5초 | 17.7초 |
| 배치 단위 (모아 쌓기) | 2.7초 | 2.9초 |
| 배치 단위 (+ 한 차량 안 연속 구간은 slice) | 1.25초 | 2.9초 |

세 방식 모두 배치 값, stride, 이후 전역 난수가 같다.

**interface**

```python
class VehicleRows(Dataset):                 # 인덱스 = 배치(시점 번호 목록). (x, y, Δ, q) double, 각각 contiguous
    def __init__(self, tensors, x_dim, y_dim)

def row_loader(tensors, x_dim, y_dim, *, batch_size, shuffle) -> DataLoader
    # DataLoader(rows, sampler=BatchSampler(RandomSampler 또는 SequentialSampler, batch_size, drop_last=False), batch_size=None)
```

- **sampler 구성:** `DataLoader(shuffle=…, batch_size=…)`가 안에서 만드는 것과 같다. 그래서 순회할 때의 전역 난수 소비(base seed 1번, shuffle이면 sampler seed 1번)가 같다.
- **배치 모으기**
  - 배치가 한 차량 안의 연속 구간이면 slice로 잘라 온다. shuffle을 끄면 거의 모든 배치가 여기에 해당한다(검증 포함).
  - 아니면 행을 하나씩 모아 `torch.stack`으로 쌓는다.
  - 어느 쪽이든 `.to(torch.double)` 뒤 열 구간마다 `.contiguous()`로 만든다. 예전 collate 결과와 stride가 같아야 같은 연산 경로를 탄다.
- `train_autoencoder`의 학습·검증 loader가 `row_loader`를 쓴다. LSTM 레거시 loader는 그대로다.

## Testing Decisions

- **회귀 테스트(`tests/test_train_rows.py`):** 테스트 안의 행 단위 참조 Dataset + `DataLoader(batch_size, shuffle)`와 `row_loader`를 비교한다.
  - 비교 항목: 배치 값, dtype, stride, 이후 `torch.rand` 값
  - 경우: shuffle 켬/끔, 차량 경계를 걸치는 batch 크기, 전체보다 큰 batch 크기, float32 차량
- **실데이터:** train(`after_bugs` 대비), train_full(`tl00a` 대비)을 bit 단위로 비교한다. 소요 시간도 기록한다(train_full은 지금 약 190초).

## Out of Scope

- 차량 텐서를 하나로 이어 붙이는 방식. 메모리 피크가 두 배가 된다.
- DataLoader worker 병렬화
- 평가 쪽 로딩
