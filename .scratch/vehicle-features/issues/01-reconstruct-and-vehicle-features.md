# 01: `reconstruct`와 `vehicle_features`

**What to build:** `mcdl/reconstruction.py`에 `reconstruct`와 `vehicle_features`를 만들고, `reconstruction_errors`와 `training_features`를 그 위에 다시 쓴다. 학습 진단 특징은 그대로다.

**Blocked by:** None (can start immediately)

**Status:** resolved

- [x] `reconstruct(model, vehicle_tensor, *, batch_size=None)`: 열 구성은 모델 크기에서 읽는다. y와 ŷ를 CPU 텐서로 돌려주고, None이면 차량 전체를 한 번에 추론한다.
- [x] `reconstruction_errors`가 `reconstruct` 결과로 오차를 만든다. interface는 그대로다.
- [x] `vehicle_features(net, netx, u_tensor, x_tensor, *, batch_size=None, abs_err)`: `abs_err`는 필수 인자다.
- [x] `training_features`가 차량마다 `vehicle_features`를 이어 붙인다.
- [x] 테스트를 추가한다.
  - `vehicle_features`가 기존 스크립트 방식(직접 열 자르기 + 4인자 전체 추론 + abs + `diagnosis_features`)과 bit 단위로 같다.
  - `reconstruct`의 y가 입력 열과 같고, batch 크기에 상관없이 ŷ가 같다.
- [x] 실데이터: train, test_u0, multi가 `after_bugs`와 bit 단위로 일치해야 한다. 결과를 Comments에 남긴다.

## Comments

**2026-10-15 완료**

- `reconstruct`는 batch마다 ŷ를 CPU로 옮겨 이어 붙이고, y는 입력 텐서의 열을 그대로(double) 돌려준다. `batch_size=None`이면 차량 길이를 batch 크기로 쓴다.
- `reconstruction_errors`는 `reconstruct` 결과로 `ŷ - y`를 계산한 뒤 abs, float32 순으로 적용한다. 반환 배열은 예전처럼 C-contiguous다(합산 순서가 바뀌지 않게 확인함). `batch_size`는 필수 키워드로 남기고 None을 허용한다.
- `vehicle_features`의 `abs_err`는 기본값 없는 필수 키워드다. `training_features`는 이 함수를 이어 붙인다.
- 이 변경으로 쓰이지 않게 된 `numpy` import를 `mcdl/reconstruction.py`에서 지웠다.
- tests: `test_reconstruction.py`에 2개를 추가했다. pytest 56개 통과.
  - `reconstruct`의 y·ŷ가 직접 자른 열 및 전체 추론과 같은지. batch 크기는 차량 길이 이상(한 batch)일 때 None과 같다는 것만 확인했다. batch를 나누면 ŷ가 ULP 수준에서 달라질 수 있어서 batch끼리는 비교하지 않는다. 나눈 batch는 기존 `test_errors_match_reference_for_any_batch_size`가 같은 batch의 참조 계산과 비교한다.
  - `vehicle_features`가 기존 스크립트 방식(4인자 전체 추론 + abs + `diagnosis_features`)과 bit 단위로 같은지
- 실데이터(기준선 태그 `vf01`): `after_bugs`와 비교해 모두 bit 단위로 일치한다.
  - train: 14개 기록
  - test_u0: 513개 기록 (관리도 재계산, 학습 차량 209대)
  - multi: 304개 기록
