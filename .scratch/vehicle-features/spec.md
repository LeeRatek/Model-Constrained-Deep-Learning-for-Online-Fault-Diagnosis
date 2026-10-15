# Spec: 차량 한 대의 진단 특징

**Status:** resolved

출처: 2026-10-15 2차 아키텍처 리뷰의 4번 후보("테스트 차량 평가 루프")와 같은 날 grilling(Q1–Q8). 용어는 `GLOSSARY.md`를 따른다.

## Problem Statement

테스트 차량마다 "차량 텐서를 `dim_dict`로 x·y·Δ·q로 자르기 → U-model·X-model 추론 → 절댓값 재구성 오차 → 진단 특징"을 계산하는 코드가 세 스크립트에 따로 있다.
- Test_: 차량 루프 안
- Test_multi_models: `process_combination` 안
- Test_tsne: `collect_vehicle_points` 안

학습 쪽은 같은 계산을 `mcdl/reconstruction.py`로 한다. 그래서 "관리도를 맞춘 학습 진단 특징과 테스트 진단 특징이 같은 방식으로 계산된다"는 사실을 코드가 보장하지 못한다. 실제로 두 쪽은 이미 다르다. 학습 쪽은 설정의 `no_abs_err`를 따르지만 테스트 쪽은 항상 절댓값을 쓴다(known-bugs 10). 열 구성 `[x | y | Δ | q]`를 아는 곳도 네 군데다.

## Solution

차량 한 대를 재구성하는 함수(`reconstruct`)와, 차량 한 대의 진단 특징을 내는 함수(`vehicle_features`)를 `mcdl/reconstruction.py`에 둔다.
- `reconstruction_errors`와 `training_features`는 이 두 함수 위에 다시 쓴다.
- 세 스크립트는 차량 루프를 그대로 두고, 루프 본문에서 이 함수들을 부른다(Q1).
- 수치 결과는 바뀌지 않는다.

## User Stories

1. 연구자로서, 리팩토링 전후 평가 결과(CI, AUC, 임계값, t-SNE 입력)가 bit 단위로 같길 원한다.
2. 연구자로서, 학습 진단 특징과 테스트 진단 특징이 같은 함수에서 나오길 원한다. 그래야 관리도와 테스트 특징이 어긋나지 않는다.
3. 개발자로서, 차량 텐서의 열 구성을 한 곳에서만 알고 싶다.
4. 개발자로서, 테스트 쪽과 학습 쪽의 계산 차이(batch, abs)가 호출하는 곳에서 보이길 원한다.
5. 개발자로서, 차량 진단 특징 계산을 합성 텐서로 시험하고 싶다.
6. 리뷰어로서, 각 티켓을 마칠 때 기준선 비교 결과가 티켓에 남길 원한다.

## Implementation Decisions

### interface (grilling Q2에서 합의)

```python
# mcdl/reconstruction.py에 추가
def reconstruct(model, vehicle_tensor, *, batch_size=None)                        # (재구성 대상 y, 재구성 결과 ŷ), torch double
def vehicle_features(net, netx, u_tensor, x_tensor, *, batch_size=None, abs_err)  # 차량 한 대의 진단 특징 (DataFrame)
```

**`reconstruct`**
- 열 구성은 모델 크기(`fc1.in_features`, `fc3.out_features`)에서 읽는다.
- 모델이 있는 device에서, `torch.inference_mode`로, `batch_size` 단위로 추론하고 결과를 이어 붙인다.
- `batch_size=None`이면 차량 전체를 한 번에 추론한다. 지금의 테스트 쪽 동작이다(Q3).
- 반환하는 y와 ŷ는 CPU 텐서다.

**`reconstruction_errors`** (interface 그대로)
- `reconstruct` 결과로 `ŷ - y`를 계산한다. 이어서 `abs_err`이면 절댓값, `float32`이면 float32로 바꾼 numpy 배열을 돌려준다.
- 오차는 원소별 계산이라, batch마다 빼든 이어 붙인 뒤 빼든 결과가 bit 단위로 같다.

**`vehicle_features`**
- `diagnosis_features(reconstruction_errors(net, u, …), reconstruction_errors(netx, x, …))`이다.
- `abs_err`는 기본값 없는 필수 인자다. 세 스크립트는 `abs_err=True`를 명시해서 넘긴다(Q4, known-bugs 10).

**`training_features`**
- 차량마다 `vehicle_features`를 구해 이어 붙인다.

### 스크립트 전환

- **Test_:** 차량 루프 본문이 `load_vehicle` → `vehicle_features(net, netx, u, x, abs_err=True)` → `chart.max_ci`/`chart.score`가 된다. 디버그 모드의 차량 고정과 플롯 분기는 그대로다.
- **Test_multi_models:** `process_combination`의 차량 본문이 `vehicle_features(..., abs_err=True)` → `chart.max_ci`가 된다. `.double()` 변환과 `del` 문은 지운다(Q7).
- **Test_tsne:** `collect_vehicle_points`에서 CI는 `vehicle_features`로, t-SNE 점(원래 값 y와 재구성 값 ŷ)은 `reconstruct`로 얻는다(Q6). 추론이 차량당 한 번씩 늘지만, CPU 추론은 결정적이라 결과는 같다.
- 세 스크립트에서 쓰이지 않게 되는 `dim_dict`, `get_input_dimensions` 호출, `collect_vehicle_points`의 `dim_dict` 인자를 지운다(Q7).

### 바꾸지 않는 것

- 차량 루프, 진행 출력, 디버그 분기, 플롯
- 테스트 쪽 batch(차량 전체)와 abs(항상 절댓값)
- 학습 쪽 batch(Train_ 4096, 평가 쪽 재계산 65536)

## Testing Decisions

**좋은 테스트의 기준.** `reconstruct`, `reconstruction_errors`, `vehicle_features`, `training_features`의 interface만 거친다. 작은 CombinedAE와 합성 텐서를 쓴다(기존 `tests/test_reconstruction.py`).

**확인할 동작**
- `vehicle_features(batch_size=None, abs_err=True)`가 기존 스크립트 방식과 bit 단위로 같다. 기존 방식은 열을 직접 자르고, 4인자로 전체 추론한 뒤 abs를 취하고 `diagnosis_features`를 거친다.
- `reconstruct`의 y가 입력 텐서의 해당 열과 같다. ŷ는 batch 크기를 지정했을 때와 None일 때가 같다(합성 텐서 크기에서).
- 기존 테스트(`reconstruction_errors`의 batch·abs·float32, `training_features`, 신호별 경로와 쌍 경로 일치)가 그대로 통과한다.

**실데이터 기준선(`after_bugs`), 티켓별**
- 01: train, test_u0, multi (학습 진단 특징 경로)
- 02: test, tsne, multi

## Out of Scope

- known-bugs 10(테스트 쪽 abs 불일치)과 11(Test_ 디버그 모드 관리도 불일치) 수정
- 차량 루프를 module로 옮기는 것
- CLI 변경
