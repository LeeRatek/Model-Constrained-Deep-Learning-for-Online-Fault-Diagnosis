# Spec: 차량 데이터 module

**Status:** resolved

출처: 2026-10-15 2차 아키텍처 리뷰의 3번 후보("차량 데이터 module")와 같은 날 grilling(Q1–Q8). 용어는 `GLOSSARY.md`를 따른다.

## Problem Statement

"분할의 차량 ID → 원본 `vin_2`·`vin_3` 로딩 → 전처리"가 스크립트 여섯 곳에 반복되어 있다.
- Train_(학습·검증)
- Test_
- Test_tsne
- Test_multi_models(학습·테스트, `.pt` 캐시 두 종류)
- 관리도 재계산

학습 차량 범위를 자르는 코드도 세 곳에 복사되어 있다. 전처리 인자(배터리 타입, learning_case 파생 플래그, `normalize_dx`, `normalize_val`)를 한 곳에서 빠뜨리면 다른 데이터가 나온다. 알려진 버그 02가 그런 경우였다. `.pt` 캐시의 해시 규칙과 파일 형식은 Test_multi_models 안에만 있어서 시험할 수 없다.

## Solution

분할 ID 읽기, 차량 하나 로딩·전처리, 캐시를 쓰는 여러 차량 로딩을 `mcdl/vehicles.py` 하나에 모은다. 스크립트는 Run 설정(또는 Train_ 인자로 만든 설정)과 차량 ID만 넘긴다. `.pt` 캐시 파일 이름과 내용은 지금과 같아서, 기존 캐시를 그대로 다시 쓴다. 수치 결과는 바뀌지 않는다.

## User Stories

1. 연구자로서, 리팩토링 뒤에도 기존 `.pt` 캐시(41 GB)를 그대로 다시 쓰길 원한다.
2. 연구자로서, 학습과 평가가 같은 규칙으로 차량을 전처리하길 원한다.
3. 연구자로서, 분할 이름을 잘못 적으면 학습이 지금처럼 실패하길 원한다. 다른 분할로 조용히 학습하면 안 된다.
4. 연구자로서, 리팩토링 전후 학습 결과와 평가 결과가 bit 단위로 같길 원한다.
5. 개발자로서, 차량 텐서가 어떻게 만들어지는지(경로, 전처리 인자, dtype) 한 곳에서 보고 싶다.
6. 개발자로서, 학습 차량 범위를 자르는 규칙(끝 포함, -1이면 끝까지)을 한 곳에서 보고 싶다.
7. 개발자로서, `.pt` 캐시 해시 규칙을 원본 데이터 없이 시험하고 싶다.
8. 개발자로서, 캐시가 있으면 원본을 읽지 않는다는 것을 테스트로 확인하고 싶다.
9. 개발자로서, Train_에서 데이터와 모델이 같은 설정 객체를 쓰길 원한다.
10. 리뷰어로서, 각 티켓을 마칠 때 기준선 비교 결과가 티켓에 남길 원한다.

## Implementation Decisions

### interface (grilling Q2에서 합의)

```python
# mcdl/vehicles.py — 차량 데이터
def split_ids(split_dir, battery_type, split, *, start=0, end=-1)   # 분할의 차량 ID (start~end 끝 포함, -1=끝)
def load_vehicle(source_data_dir, config, vehicle_id)               # 원본 → 전처리된 (U 텐서, X 텐서)
def load_training_vehicles(source_data_dir, config, ids)            # [(U, X)] double, .pt 캐시 train_data_list_<BT>_<hash>.pt
def load_test_vehicles(source_data_dir, config, ids)                # {id: (U, X)}, .pt 캐시 test_data_cache_<BT>_<hash>.pt
```

**config**
- `RunConfig`다.
- 쓰는 필드는 `battery_type`, `preprocessing`, `skip_charge_ready`, `normalize_dx`, `normalize_val`다.

**`split_ids`**
- `{split_dir}/{BT}_filtered_vehicle_ids_{split}.npy`를 `.astype(np.int64).tolist()`로 읽는다.
- `end != -1`이면 `[start : end + 1]`, 아니면 `[start:]`이다.

**`load_vehicle`**
- `{source_data_dir}/{BT}/{id}/vin_2.pkl`과 `vin_3.pkl`을 `safe_load(verbose=False)`로 읽는다.
- 이어서 `preprocess_loaded_tensor`에 설정의 전처리 인자를 그대로 넘긴다.

**`.pt` 캐시**
- 해시: 다음 dict를 `json.dumps(sort_keys=True)`로 만든 문자열의 md5다.
  - `{"<train_list|test_list>": ids, "battery_type", "preprocessing", "skip_charge_ready", "normalize_dx"}`
  - `normalize_dx`일 때만 `"normalize_val"`을 추가한다(known-bugs 02 수정).
- 파일 위치: `source_data_dir`
- 저장 형식
  - 학습 캐시: `(U.double(), X.double())`의 list
  - 테스트 캐시: `{id: (U, X)}` dict. double로 바꾸지 않는다.
- 캐시를 읽다 실패하면 메시지를 찍고 다시 만든다.
- 콘솔 메시지는 지금과 같은 내용으로 찍는다.

### 스크립트 전환

- **Test_, Test_tsne**
  - 분할 ID는 `split_ids(run.config.split_dir, …)`로, 차량은 `load_vehicle`로 읽는다. 캐시는 쓰지 않는다(Q6).
  - Test_tsne의 `collect_vehicle_points`는 배터리 타입·전처리 플래그 인자 대신 설정 객체를 받는다.
- **관리도 재계산(`mcdl/chart_source.py`):** `split_ids`(학습 차량 범위 포함)와 `load_vehicle`로 차량을 하나씩 읽어 처리한다. `.pt` 캐시는 쓰지 않는다(Q5). 학습 캐시는 26.7 GB라 통째로 메모리에 올려야 하는데, 지금의 스트리밍 방식은 약 75초면 된다.
- **Test_multi_models:** 학습·테스트 로딩 블록을 `split_ids` + `load_training_vehicles`·`load_test_vehicles`로 바꾼다.
- **Train_**
  - `RunConfig`를 앞쪽에서 한 번 만들어 데이터와 모델에 함께 쓴다. 디버그 모드의 Δ 해제도 반영한다(Q4).
  - 분할 경로는 Train_의 지금 규칙(`./datasets/{이름}`, 폴더가 없으면 실패)으로 정해 `split_ids`에 넘긴다(Q3).
  - 학습·검증 차량은 `load_vehicle`로 읽는다. LSTM `vin_1` 로딩(레거시)은 그 자리에 둔다.

### 바꾸지 않는 것

- `data_filter.py`(Q1)
- 테스트 차량 평가 루프(4번 후보)
- 디버그 모드 분기
- `.pt` 캐시 파일 이름과 형식

## Testing Decisions

**좋은 테스트의 기준.** `split_ids`, `load_vehicle`, `load_training_vehicles`, `load_test_vehicles`의 interface만 거친다.

**fixture.** `tmp_path`의 가짜 원본 데이터(QAS 열 구성 pkl)와 분할 `.npy`를 쓴다. 이 fixture는 test_chart_source와 함께 쓰는 helper로 옮긴다.

**확인할 동작**
- `split_ids`: float64로 저장된 `.npy`도 정수 ID로 읽고, 범위(끝 포함, -1)를 적용한다.
- `load_vehicle`: 직접 `safe_load` + `preprocess_loaded_tensor`를 한 결과와 같다.
- 캐시
  - 첫 호출 때 문서화된 규칙의 이름으로 캐시를 만든다.
  - 원본을 지운 뒤에도 두 번째 호출은 캐시로 같은 값을 돌려준다.
  - `normalize_dx`일 때만 해시에 `normalize_val`이 들어간다.
- realdata: 문서화된 해시 규칙으로 계산한 이름이, 기준 Run의 실제 캐시 파일 이름 두 개와 같다.
  - 학습 캐시: `015ac371…`
  - 테스트 캐시: `528fb02a…`

**실데이터 기준선(`after_bugs`), 티켓별**
- 01: test, test_u0, tsne
- 02: multi. 캐시를 다시 쓰는지는 소요 시간과 "Found cached" 메시지로 확인한다.
- 03: train

## Out of Scope

- `data_filter.py`
- 테스트 차량 forward와 평가 루프
- `.pt` 캐시를 재계산이나 Test_·Test_tsne에 새로 쓰는 것
- CLI 변경
