# Spec: 재구성 오차·학습 진단 특징 module과 관리도 공급원

**Status:** resolved (티켓 01–02 완료, 2026-10-15)

출처: 2026-10-15 2차 아키텍처 리뷰의 1번 후보("모델 쌍의 관리도 공급원을 한 곳으로")와 2번 후보("학습 특징 계산을 하나로"), 그리고 같은 날 grilling(Q1–Q6). 용어는 `GLOSSARY.md`를 따른다.

## Problem Statement

같은 모델 쌍이면 어느 스크립트로 평가해도 같은 관리도가 나와야 한다. 알려진 버그 05·06·07·08을 고치면서 이것이 가능해졌지만, 지금은 그 규칙들이 손으로 맞춰져 있을 뿐이다.

- **재구성 오차 → 학습 진단 특징 계산이 세 벌이다.** Test_multi_models의 `run_model_inference`, Train_pca_only의 `_reconstruction_errors`, Train_의 PCA 루프다. 특히 Test_multi_models와 Train_pca_only는 같은 학습 진단 특징을 내야 하는데, 두 구현을 손으로 맞춰 둔 상태다.
- **관리도 출처 규칙이 두 군데에 복사되어 있다.** 규칙은 "같은 조건의 캐시 → 최종 모델 쌍이면 npz → 그 밖엔 재계산"이고, Test_와 Test_tsne에 각각 있다. 캐시를 쓰는 조건과 형식은 Test_multi_models에 따로 있다.

## Solution

두 module로 나눈다.

- **재구성 오차 module:** 모델과 차량 텐서로 재구성 오차를 구하고, 학습 차량들의 학습 진단 특징을 만든다. 형태는 두 가지다. 하나는 U·X를 함께 처리하는 형태이고, 다른 하나는 신호별로 처리하는 형태다.
- **관리도 공급원 module:** Run과 모델 쌍을 받아 관리도를 돌려준다. temp_cache 읽기·쓰기 형식과, 읽어도 되는 캐시 조건을 소유한다.

Train_, Test_, Test_multi_models, Test_tsne는 이 두 module만 부른다. `Train_pca_only.py`는 없어진다. 디스크 형식(temp_cache 파일 이름과 pickle 내용), CLI, 수치 결과는 바뀌지 않는다.

## User Stories

1. 연구자로서, 같은 모델 쌍이면 Test_, Test_tsne, Test_multi_models 중 어느 것으로 평가해도 같은 관리도가 나오길 원한다.
2. 연구자로서, `Test_multi_models --keep-cache`가 남긴 캐시를 Test_와 Test_tsne가 지금처럼 찾아 쓰길 원한다. 단, 계산 조건이 같을 때만이다.
3. 연구자로서, 리팩토링 전후로 학습 체크포인트, `pca_arrays.npz`, AUC, 최적 임계값이 bit 단위로 같길 원한다.
4. 연구자로서, 최종 모델 쌍이 아닌 모델 쌍을 캐시 없이 평가하면 지금처럼 학습 데이터로 관리도를 다시 계산해 주길 원한다.
5. 개발자로서, 재구성 오차를 구하는 코드(열 구성 슬라이싱, batch 추론, abs, float32 변환)를 한 곳에서 찾고 싶다.
6. 개발자로서, Test_multi_models와 재계산 경로의 학습 진단 특징이 같다는 것을 코드 구조로 보장받고 싶다.
7. 개발자로서, batch 크기가 결과에 영향을 준다는 사실이 interface에 드러나길 원한다(Train_ 4096, 평가 65536).
8. 개발자로서, 관리도 출처 규칙(캐시 → npz → 재계산)을 한 함수에서 읽고 고치고 싶다.
9. 개발자로서, temp_cache 파일 형식(신호별 특징 3개 pickle)을 쓰는 코드와 읽는 코드가 한 module에 있길 원한다.
10. 개발자로서, 관리도 출처의 세 분기를 원본 데이터 없이 tmp Run과 작은 가짜 원본 데이터로 시험하고 싶다.
11. 개발자로서, 재구성 오차 계산을 작은 `CombinedAE`와 합성 차량 텐서로 시험하고 싶다.
12. 리뷰어로서, 각 티켓을 마칠 때 실데이터 기준선(`after_bugs`) 비교 결과가 티켓에 남길 원한다.

## Implementation Decisions

### interface (grilling Q2에서 합의)

```python
# mcdl/reconstruction.py — 재구성 오차와 학습 진단 특징
def reconstruction_errors(model, vehicle_tensor, *, batch_size, abs_err, float32=False)  # 한 차량 (시점, 셀)
def training_features(net, netx, vehicles, *, batch_size, abs_err)    # (U, X) 텐서 쌍 → 차량마다 진단 특징 → 이어 붙인 DataFrame
def training_signal_features(model, signal, vehicle_tensors, *, batch_size, abs_err, float32=False)  # 신호별 특징 3개

# mcdl/chart_source.py — 모델 쌍의 관리도 공급원
def chart_for_pair(run, u, x, *, source_data_dir)            # 같은 조건 캐시 → 최종 쌍이면 npz → 그 밖엔 재계산
def save_cached_features(run, model, number, features, *, conditions)
def load_cached_features(run, model, number, *, conditions)
```

### 재구성 오차

- **열 구성은 모델에서 읽는다(Q3).** 차량 텐서는 `[x | y | Δ | q]` 순서다.
  - x 열 수는 `fc1.in_features`
  - y와 Δ 열 수는 `fc3.out_features`
  - q 열 수는 `fc2.in_features`
- **입력과 batch.** 각 입력은 `.double()`로 바꾼다. batch 단위로 모델이 있는 device에서 추론하고, 오차는 CPU numpy로 돌려준다(Q4).
- **abs와 float32.** `abs_err`면 절댓값을 쓴다. `float32`면 batch 오차를 float32로 바꾼 뒤 numpy로 돌려준다. 지금 `--use-float32`와 같은 동작이다.
- **batch 크기는 결과의 일부다.** 호출자가 정한다.
  - Train_: 4096
  - Test_multi_models: `--inference-batch-size`(기본 65536)
  - 관리도 재계산: 65536
- **`training_features`.** `(U 텐서, X 텐서)` 쌍을 차례로 받는다. 차량마다 U 오차와 X 오차를 구한 뒤 `diagnosis_features`를 적용하고, 결과를 이어 붙인다(`ignore_index=True`). 쌍은 iterator로 받을 수 있어서, 원본을 한 번만 읽고 처리할 수 있다.
- **`training_signal_features`.** 한 신호의 차량별 `signal_features`를 구해 신호별로 이어 붙인다. 지금 Test_multi_models의 Phase 1·2와 같고, temp_cache 형식과도 같다.

### 관리도 공급원

- **재계산 조건은 상수 하나다(Q5).** 설정의 abs 여부, float32 끔, batch 65536이다. 이 상수를 다음 두 곳에 쓴다.
  - 읽어도 되는 캐시를 고르는 조건
  - 재계산할 때의 조건
- **`chart_for_pair(run, u, x, source_data_dir)` 순서**
  1. 위 조건으로 만든 U·X 캐시 파일이 둘 다 있으면, 캐시 특징을 `combine_signal_features` → `ControlChart.fit`으로 맞춘다. 캐시를 읽다 오류가 나면 지금처럼 메시지를 찍고 다음 단계로 넘어간다.
  2. U와 X가 모두 -1이면 `ControlChart.load(run.path)`를 쓴다.
  3. 그 밖에는 Run의 학습 차량 원본을 읽어 전처리한 뒤 `training_features`(batch 65536) → `ControlChart.fit`으로 계산한다.
- **콘솔 메시지는 지금과 같은 내용으로 찍는다.** "Loading cached training features", "Falling back to saved 'pca_arrays.npz'", "Recomputing …"이다.
- **`save_cached_features`와 `load_cached_features`.** 경로는 `run.feature_cache_path`를 쓰고, 내용은 신호별 특징 3개 튜플의 pickle이다. Test_multi_models는 자기 실행 조건(`--use-float32`, `--inference-batch-size`)으로 캐시를 쓰고 읽는다.

### 스크립트 전환

- **Train_**
  - 학습 끝 PCA 루프를 `training_features(net, netx, zip(학습 차량 U 텐서, X 텐서), batch_size=4096, abs_err=use_abs_err)`로 바꾼다.
  - `net.eval()`과 `netx.eval()`은 그대로 둔다.
- **Test_, Test_tsne:** 관리도 출처 분기 전체를 `chart_for_pair(run, args.u_model_idx, args.x_model_idx, source_data_dir=args.source_data_dir)` 한 줄로 바꾼다.
- **Test_multi_models**
  - Phase 1·2는 `training_signal_features`와 `save_cached_features`를 쓴다(disk). memory 모드는 dict에 그대로 둔다.
  - 조합 평가는 `load_cached_features`나 memory dict에서 특징을 읽는다.
  - `run_model_inference`는 삭제한다.
- **`Train_pca_only.py`:** 삭제한다. Test_multi_models의 쓰이지 않던 import도 함께 지운다.

### 바뀌지 않는 것

- temp_cache 파일 이름과 내용
- 콘솔 출력의 의미
- CLI
- 테스트 차량 forward: 차량 전체를 한 번에 처리하는 경로이고, 4번 후보에서 다룬다.
- 데이터 로딩과 `.pt` 캐시: 3번 후보에서 다룬다.

## Testing Decisions

**좋은 테스트의 기준.** `reconstruction_errors`, `training_features`, `training_signal_features`, `chart_for_pair`, 캐시 저장·로드의 interface만 거친다.

**재구성 오차 (작은 `CombinedAE`, 셀 4개, 합성 차량 텐서)**
- batch 크기가 차량 길이 이상이면, 한 번에 forward한 결과와 bit 단위로 같다.
- batch로 나눈 결과는 batch마다 forward해 이어 붙인 결과와 같다.
- abs와 부호 있는 오차를 구분한다. float32 옵션은 float32로 반올림된 값을 낸다.
- `training_features`는 `combine_signal_features(training_signal_features(U), training_signal_features(X))`와 같다(`assert_frame_equal`, exact).

**관리도 공급원 (`tmp_path` Run + 작은 가짜 원본 데이터)**
- 가짜 원본 데이터는 QAS 열 구성 `vin_2.pkl`·`vin_3.pkl`이 있는 차량 몇 대와 분할 `.npy`로 만든다.
- 세 분기를 모두 실행한다.
  - 캐시 분기: 같은 조건의 캐시가 있으면 그 특징으로 맞춘 관리도를 쓴다.
  - 조건이 다른 캐시는 무시한다.
  - npz 분기: -1/-1이면 저장된 관리도를 쓴다.
  - 재계산 분기: 학습 차량으로 계산한 관리도가 `training_features`를 직접 써서 맞춘 관리도와 같다.
- 캐시 저장 → 로드가 round-trip으로 같은 값을 낸다.

**실데이터 기준선(`after_bugs`)**
- train, test, test_u0(재계산 경로), tsne, multi가 bit 단위로 일치해야 한다.
- test_u0은 이번에 기준선 도구에 추가한 대상이다(Test_, U 0 / X -1, 캐시 없음).

## Out of Scope

- 테스트 차량 forward와 평가 루프 (4번 후보)
- 데이터 로딩, `.pt` 캐시, 재계산의 캐시 활용 (3번 후보)
- CLI 변경, 디버거 분기 (5번 후보)
- 진단 수식, 학습 알고리즘 변경
