# 01: 배치 단위 `VehicleRows`와 `row_loader`

**What to build:** 학습·검증 DataLoader가 배치 인덱스를 한 번에 넘기고, `VehicleRows`가 배치를 한 번에 모아 오게 한다. 학습 결과는 bit 단위로 같다.

**Blocked by:** None (can start immediately)

**Status:** resolved

- [x] `VehicleRows.__getitem__(indices)`: 한 차량 안의 연속 구간이면 slice로, 아니면 행을 모아 쌓는다. `.to(torch.double)` 뒤 열 구간마다 contiguous로 만든다.
- [x] `row_loader(tensors, x_dim, y_dim, *, batch_size, shuffle)`를 만들고, `train_autoencoder`의 학습·검증 loader가 이것을 쓴다.
- [x] `tests/test_train_rows.py`: 행 단위 참조 loader와 배치·dtype·stride·이후 난수가 같은지 확인한다. shuffle 켬/끔, 차량 경계, 큰 batch, float32를 포함한다.
- [x] 실데이터: train(`after_bugs`), train_full(`tl00a`)이 bit 단위로 일치해야 한다. 소요 시간과 함께 Comments에 남긴다.

## Comments

**2026-10-15 완료**

- `VehicleRows.__getitem__(indices)`
  - 배치가 한 차량 안의 연속 구간이면 slice로 잘라 오고, 아니면 행을 모아 `torch.stack`으로 쌓는다.
  - `.to(torch.double)` 뒤 열 구간마다 `clone(memory_format=torch.contiguous_format)`으로 표준 stride의 새 텐서를 만든다.
  - 처음에는 `.contiguous()`를 썼다. 그런데 batch가 1행이면 복사하지 않아 stride가 `(13, 1)`로 남았고(예전 collate는 `(2, 1)`), 회귀 테스트가 이를 잡아서 clone으로 바꿨다.
- `row_loader(tensors, x_dim, y_dim, *, batch_size, shuffle)`: `DataLoader(rows, sampler=BatchSampler(RandomSampler|SequentialSampler, batch_size, drop_last=False), batch_size=None)`. `train_autoencoder`의 학습·검증 loader가 이것을 쓴다.
- tests: `tests/test_train_rows.py`에 12개를 넣었다. pytest 82개 통과.
  - 비교 대상: 행 단위 참조 loader
  - 경우: shuffle 켬/끔 × batch 7/1/1000 × float64/float32
  - 확인: 배치 값, dtype, stride, 순회 뒤 `torch.rand`
- 실데이터(기준선 태그 `rp01`): 모두 bit 단위로 일치한다.

  | 대상 | 비교 기준 | 기록 | 소요 시간 |
  |---|---|---|---|
  | train | `after_bugs` | 14개 일치 | 4.1초 → 2.4초 |
  | train_full | `tl00a` | 17개 일치 | 191.2초 → 40.7초 (4.7배) |

  - train_full의 약 25초는 차량 52대(학습 2 + 검증 50) 로딩이다. 학습·검증 부분만 보면 약 165초 → 약 15초다.
