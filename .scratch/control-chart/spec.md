# Spec: 진단 특징 · 관리도 · ROC 평가 module

**Status:** resolved (티켓 01–03 완료, 2026-10-15)

출처: 2026-10-14 아키텍처 리뷰의 4번 후보("진단 모니터 module")와 5번 후보("차량 단위 ROC 평가 module"), 그리고 2026-10-15 grilling(Q1–Q8). 용어는 `GLOSSARY.md`를 따른다.

## Problem Statement

연구자가 보고하는 수치는 모두 같은 계산 사슬에서 나온다. PCA 관리한계, CI, AUC, 최적 임계값이 그렇다. 사슬은 차량별 진단 특징 → 관리도 → T²·SPE → CI → 최대 CI → ROC 순서다. 그런데 이 사슬을 잇는 접착 코드가 스크립트마다 따로 있다.

- 진단 특징 구현이 두 벌이다. Function_의 `DiagnosisFeature`와 Test_multi_models의 신호별 클론이다.
- 열 순서 `[u0,x0,u1,x1,u2,x2]`와 15-tuple 풀기가 3곳에 복사되어 있다.
- CI 식은 3곳에, 임계값 grid와 예측 행렬은 2곳에 있다.
- 이 규칙을 하나라도 빠뜨리면 다른 숫자가 나온다. Train_pca_only가 차량 경계를 넘어 특징을 계산하는 것(known-bugs 05)이 그 예다.
- 이 계산을 원본 데이터 없이 시험할 방법도 없다.

## Solution

계산 사슬을 module 두 개로 모은다.

- 진단 특징과 관리도 module: 스크립트는 차량별 재구성 오차를 넘겨 진단 특징을 받는다. 정상 학습 특징으로 관리도를 맞추거나 저장된 관리도를 읽은 뒤, 차량별 T²·SPE·CI와 최대 CI를 받는다.
- ROC 평가 module: 차량별 최대 CI와 고장 여부를 넘기면 AUC와 최적 임계값을 받는다.

디스크 형식(`pca_arrays.npz`, `pca_manifest.json`, `pca_data_nor.csv`, temp_cache), CLI, 수치 결과는 바뀌지 않는다. 지금 구현의 출력을 합성 오차 행렬로 고정한 fixture와 실데이터 기준선으로 bit 단위 보존을 확인한다.

## User Stories

1. 연구자로서, 리팩토링 전후 PCA 관리한계, 차량별 CI, AUC, 최적 임계값이 bit 단위로 같다는 증거를 보고 싶다.
2. 연구자로서, 기존 Run의 `pca_arrays.npz`를 그대로 읽어 평가하고 싶다.
3. 연구자로서, 새로 학습한 Run도 지금과 같은 `pca_arrays.npz`, `pca_manifest.json`, `pca_data_nor.csv`를 남기길 원한다.
4. 연구자로서, Test_multi_models가 이미 남긴 temp_cache 특징 파일을 Test_와 Test_tsne에서 지금처럼 쓰고 싶다.
5. 연구자로서, 진단 특징 정의(6개 값, X 쪽 100-step 이동평균, EWMA α=0.2)를 한 곳에서 확인하고 싶다.
6. 연구자로서, CI 식과 최대 CI 정의를 한 곳에서 확인하고 싶다.
7. 연구자로서, `--use-reverse-pos-neg` 판정 방향이 지금과 같게 동작하길 원한다.
8. 연구자로서, AUC_summary CSV와 ROC 그림이 지금과 같은 값으로 만들어지길 원한다.
9. 개발자로서, 진단 특징 구현이 하나뿐이길 원한다. 그래야 학습 경로와 Test_multi_models 경로가 어긋나지 않는다.
10. 개발자로서, 15-tuple을 위치로 풀지 않고 관리도 하나를 주고받고 싶다.
11. 개발자로서, 관리한계(T²₉₅, SPE₉₅)와 임계값을 코드에서 다른 이름으로 구분하고 싶다.
12. 개발자로서, 원본 데이터 없이 합성 오차 행렬로 진단 특징, 관리도, CI, ROC를 몇 초 안에 시험하고 싶다.
13. 개발자로서, 지금 구현의 출력이 fixture로 고정되어 있길 원한다. 그래야 이후 리팩토링이나 버그 수정이 수치를 바꾸면 바로 드러난다.
14. 개발자로서, 관리도를 저장했다가 다시 읽어도 같은 CI가 나오는지 확인하고 싶다.
15. 개발자로서, Test_multi_models가 조합마다 학습 특징 전체(약 300 MB)를 메모리에 들고 있지 않길 원한다.
16. 개발자로서, 알려진 버그 05·06·07·08·09가 이번 리팩토링에서 고쳐지지도, 섞이지도 않길 원한다.
17. 개발자로서, Function_.py에서 진단 계산이 빠져 탐색할 곳이 줄어들길 원한다.
18. 리뷰어로서, 각 티켓을 마칠 때 fixture 테스트와 실데이터 기준선 비교 결과가 티켓에 남길 원한다.

## Implementation Decisions

### module과 interface (grilling Q5에서 합의)

```python
# mcdl/diagnosis.py
def signal_features(signal, error)       # "U"|"X" 한 신호의 특징 3개 (Test_multi 신호별 캐시용)
def combine_signal_features(u, x)        # 신호별 특징 → 진단 특징 열 순서 [u0,x0,u1,x1,u2,x2]
def diagnosis_features(error_u, error_x) # 차량 1대의 진단 특징 (위 둘을 조합)

class ControlChart:                      # 관리도
    @classmethod
    def fit(cls, features)               # 정상 학습 진단 특징 (차량별로 계산해 이어 붙인 DataFrame)
    @classmethod
    def load(cls, run_dir)               # pca_arrays.npz
    t2_limit, spe_limit                  # 관리한계 (95%)
    def save(self, run_dir, features)    # npz + manifest + pca_data_nor.csv
    def score(self, features)            # → (T², SPE, CI) 시계열
    def max_ci(self, features)           # → 최대 CI

# mcdl/roc.py
def evaluate(max_ci, is_fault, reverse=False)   # → RocResult(fpr, tpr, auc, best, closest_to_01)
```

**진단 특징**
- 구현은 `signal_features` 하나다. `DiagnosisFeature`와 Test_multi_models의 `extract_features_from_error`는 연산이 같으므로 이것으로 대체한다.
  - 연산: eps 1e-12, 두 번째 최댓값은 `np.partition`, EWMA α=0.2, X 신호의 3개 특징에만 100-step 이동평균
- `get_true_feature` 옵션(이동평균 전 X 특징)은 옮기지 않는다. 쓰는 곳이 주석 처리된 그림 코드뿐이다.
- `signal_features`가 돌려주는 값(Series 3개 튜플)은 temp_cache pickle 형식과 같다. 기존 캐시 파일을 그대로 읽을 수 있다.

**관리도**
- `fit`은 받은 DataFrame을 그대로 PCA(l1 = l2 = 0.99)에 넘긴다. 15-tuple은 관리도 안에 숨긴다.
- 학습 특징 행렬 `data_nor`는 들고 있지 않는다(Q6). `save`는 같은 식 `(features − mean) / std`로 다시 계산해 `pca_data_nor.csv`에 쓴다.
- `load`는 기존 평가 경로와 같은 옵션을 쓴다: `data_nor` 없이, shape 검증 없이.
- `data_mean`·`data_std`의 표현은 출처 그대로 둔다. `fit`이면 pandas Series, `load`면 ndarray다. 바꾸면 T²·SPE 계산이 bit 단위로 달라질 수 있다.
- `score`의 CI 식은 `SPE / SPE₉₅ + T² / T²₉₅` 순서 그대로 쓴다. `max_ci`는 `float(np.nanmax(np.asarray(ci, dtype=float)))`다.
- `save`는 Train_이 지금 넘기는 출력 경로 문자열 형식(`"<Run 경로>/"`)을 그대로 받는다. manifest 안의 경로 문자열이 지금과 같아야 하기 때문이다.

**ROC 평가**
- `evaluate` 안에 숨기는 것:
  - 임계값 grid: `np.asarray(list(np.arange(0, 1000, 0.1)), dtype=float)`
  - 차량×임계값 int8 예측 행렬
  - 기본 `np.argsort`로 FPR 정렬 (known-bugs 09 유지)
  - Youden J 최적점과 (0,1) 최근접점
- 판정 규칙
  - 기본: 고장을 양성으로 보고, 최대 CI > θ이면 양성으로 판정한다.
  - `reverse`: 정상을 양성으로 보고, 최대 CI < θ이면 양성으로 판정한다.
  - 차량 순서는 호출자가 넘긴 순서를 그대로 쓴다.
- `RocResult`의 `best`·`closest_to_01`은 지금과 같은 dict다. CSV와 그림 코드를 그대로 쓰기 위해서다.

### 스크립트 전환

- **Train_**
  - 차량별 `diagnosis_features`를 구해 이어 붙인다(`ignore_index=True`).
  - 이어서 `ControlChart.fit`과 `save`를 부른다.
  - 학습 차량 forward의 batch 크기(4096)는 그대로 둔다.
- **Test_, Test_tsne**
  - PCA 출처 결정 정책(temp_cache → npz → train_pca_only)은 스크립트에 그대로 둔다. known-bugs 06·07·08이 이 정책에 걸려 있다.
  - 각 분기는 관리도를 얻기만 한다.
  - 테스트 차량은 `diagnosis_features` → `score`/`max_ci` 순서로 처리한다.
- **Test_multi_models**
  - 학습 특징 캐시는 `signal_features`로 만든다. 파일 이름과 키는 그대로다.
  - 조합마다 `combine_signal_features` → `ControlChart.fit`을 부른다.
  - 테스트 차량은 `diagnosis_features` → `max_ci`, 마지막에 `evaluate`를 부른다.
- **Train_pca_only (Q7 (b))**
  - 새 interface를 쓰고 `ControlChart`를 돌려준다.
  - 버그 05의 순서(차량을 이어 붙인 뒤 특징 계산)는 유지한다.
  - 이 경로는 known-bugs 06 때문에 지금도 도달할 수 없다.
- **ROC를 쓰는 스크립트 (Test_, Test_multi_models)**: `evaluate`를 쓴다. 그림, CSV, 콘솔 출력은 스크립트에 남긴다.

### Function_.py 정리 (Q2)

- 옮긴 함수의 원본은 지운다: `DiagnosisFeature`, `Custom_PCA`, `T2_array`, `SPE_array`, `save_pca_results`, `load_pca_results`, `compute_roc_auc_from_threshold_matrix`, `compute_optimal_thresholds_from_roc`.
- 원래부터 쓰이지 않던 `plot_auc_roc_curve_from_threshold_matrix`도 지운다. 사용자 승인을 받았고, ROC 함수를 옮기면 호출 대상이 없는 함수가 되기 때문이다.
- 이 변경으로 고아가 된 import도 지운다.

## Testing Decisions

**좋은 테스트의 기준.** 진단 특징 함수, `ControlChart`, `evaluate`의 interface만 거쳐 관찰 가능한 결과를 확인한다.

**지금 동작을 고정하는 fixture (티켓 01)**
- 고정 seed로 합성 오차 행렬을 만든다. 정상 학습 차량 몇 대와, 정상·고장 테스트 차량 몇 대로 구성하고, 셀은 110개다.
- 옮기기 전의 Function_ 구현으로 다음 결과를 계산해 `tests/fixtures/`에 저장한다.
  - 차량별 진단 특징과 이어 붙인 학습 특징
  - PCA 출력 15개
  - 저장 → 로드한 관리도로 계산한 CI
  - 차량별 T²·SPE·CI·최대 CI
  - 기본과 reverse 판정의 ROC 결과
- 새 module 테스트는 이 값을 bit 단위로 재현해야 한다.
- fixture는 이 Windows 환경(numpy 2.3.5, scipy 1.16.3)에서 만들었다. 다른 플랫폼에서는 ULP 수준 차이로 실패할 수 있다.

**추가 테스트**
- `combine_signal_features(signal_features(U), signal_features(X))`가 `diagnosis_features`와 같은지
- `save` → `load` round-trip 후 CI가 같은지(ndarray 경로)
- 관리한계 속성 값
- `evaluate`의 판정 방향과 입력 검증

**실데이터 기준선**
- `tests/realdata/baseline.py`로 `before1`과 비교한다.
- 도구는 옮긴 함수를 새 위치에서 같은 기록 이름으로 감싼다.
- `Test_.result`는 ROC wrapper가 잡은 입력과 출력에서 만든다.
- 티켓 01에서 갱신한 도구로 지금 코드를 다시 떠서 `before1`과 일치하는지 먼저 확인한다.

**참고할 기존 테스트:** `tests/test_run.py`(합성 Run fixture, 레거시 로드 순서를 참조 구현과 비교하는 방식)

## Out of Scope

- 모델 forward와 재구성 오차 계산: batch 크기, abs 여부, dtype 변환 (3번 후보)
- temp_cache 파일 관리, 캐시 키, PCA 출처 결정 정책 (known-bugs 06·07·08)
- 플롯, CSV 쓰기, 콘솔 출력
- known-bugs 05·06·07·08·09의 수정
- 진단 수식 변경, 새 기능

## Further Notes

- `T2`·`SPE`(배열이 아닌 버전) 등 원래부터 쓰이지 않던 Function_ 함수는 건드리지 않는다.
- Test_의 `CI_contrib`(기여도 합)는 지금도 계산만 하고 쓰지 않는다. 전환 후에는 계산하지 않는다. 그림을 그리는 진단 모드(`AUC_ANALYSIS = False`)는 T²·SPE·CI 시계열만 쓴다.
