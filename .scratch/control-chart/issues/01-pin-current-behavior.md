# 01: 진단 계산의 현재 동작 고정 (코드 이동 없음)

**What to build:** 진단 특징부터 ROC까지 지금 구현이 내는 값을 원본 데이터 없이 재현할 수 있게 고정한다. 함수를 새 위치로 옮긴 뒤에도 실데이터 기준선을 `before1`과 그대로 비교할 수 있게 기준선 도구를 갱신한다. 이 티켓에서는 루트 스크립트와 Function_.py를 바꾸지 않는다.

**Blocked by:** None (can start immediately)

**Status:** resolved

- [x] 고정 seed로 합성 오차 행렬을 만드는 helper를 tests 아래에 둔다. 정상 학습 차량과 정상·고장 테스트 차량, 셀 110개로 구성한다.
- [x] 지금 Function_ 구현으로 다음 결과를 계산해 `tests/fixtures/`에 저장한다.
  - 차량별 진단 특징
  - 이어 붙인 학습 특징
  - PCA 출력 15개
  - 저장 → 로드한 관리도로 계산한 CI
  - 차량별 T²·SPE·CI·최대 CI
  - 기본과 reverse 판정의 ROC 결과
  - 생성 방법(환경, seed)은 Comments에 기록한다.
- [x] fixture를 지금 구현과 비교하는 테스트를 추가한다. 티켓 02와 03에서는 같은 테스트가 새 module을 대상으로 바뀐다.
- [x] 기준선 도구를 갱신한다.
  - 감쌀 함수를 위치 목록으로 찾는다. 지금은 Function_이고, 티켓 02·03 뒤에는 새 module이다.
  - `Test_.result`는 ROC wrapper가 잡은 입력(`y_true`, 예측 행렬)과 출력에서 만든다.
- [x] 갱신한 도구로 지금 코드를 다시 떠서(`before4`) `before1`과 bit 단위로 일치하는지 확인한다. 대상은 train, test, tsne, multi이고, 결과를 Comments에 남긴다.

## Comments

### 2026-10-15 완료

**합성 오차**: `tests/synthetic_errors.py`의 `make_vehicle_errors()`. seed는 20261015이고, 셀은 110개다.

| 구성 | 시점 수 | 특징 |
|---|---|---|
| 정상 학습 4대 | 300, 260, 340, 280 | |
| 정상 테스트 3대 | 250, 80, 270 | 80 시점 차량은 100-step 이동평균을 건너뛰는 경로를 탄다 |
| 고장 테스트 3대 | 290, 320, 260 | 후반부에 셀 하나의 오차를 0.08, 0.02, 0.005까지 키운다 |

**fixture**: `tests/fixtures/diagnosis_golden.npz`
- 배열 65개, 274 KB
- 만든 환경: Windows 11, Python 3.12.13, numpy 2.3.5, pandas 2.3.3, scipy 1.16.3
- 옮기기 전 Function_ 구현(`DiagnosisFeature`, `Custom_PCA`, `T2_array`, `SPE_array`, `save_pca_results`/`load_pca_results`, ROC 함수 2개)으로 계산했다. Test_·Test_multi_models와 같은 임계값 grid와 판정 규칙을 썼다.
- 생성 스크립트는 옮기기 전 함수를 import하므로 저장소에 두지 않았다. 세션 scratchpad의 `make_diagnosis_golden.py`다.
- 최대 CI 값: 정상 3.43, 729.65, 3.29 / 고장 1203.96, 3.91, 1.98
  - 80 시점 정상 차량이 큰 값을 받는다(이동평균을 건너뛰기 때문).
  - 1203.96은 임계값 grid의 끝(999.9)을 넘는다.
- AUC: 기본 0.5556, reverse 0.5
- 정리하면 짧은 차량, grid 포화, 단순하지 않은 ROC 곡선을 모두 고정한다.

**테스트**: `tests/test_diagnosis.py`
- 6개 테스트를 추가했다: 진단 특징, PCA, 점수와 최대 CI, 저장·로드 후 CI, ROC 기본·reverse.
- 지금 구현을 대상으로 모두 bit 단위로 통과한다. 전체 pytest는 32개 통과, 약 5.5초다.

**기준선 도구 갱신** (`tests/realdata/baseline.py`)
- `WRAP_TARGETS`: 기록 이름별로 함수가 있을 수 있는 위치 목록을 둔다. 있는 위치만 감싼다.
- `legacy_out`: 반환 모양이 바뀐 함수를 이전 모양으로 바꿔 기록한다. 티켓 02에서 쓴다.
- `Test_.result`: ROC wrapper가 잡은 마지막 위치 인자(`y_true`, 예측 행렬)와 출력(roc, opt)으로 만든다.

**`before4`(갱신한 도구, 지금 코드) vs `before1`**: 네 대상 모두 모든 기록이 bit 단위로 일치했다.

| target | 기록 수 | before4 소요 시간 |
|---|---|---|
| train | 14 | 3.9 s |
| test | 304 | 44.6 s |
| tsne | 24 | 23.7 s |
| multi | 304 | 234.3 s |

`Test_.result`를 스크립트 전역 변수 대신 wrapper 값으로 만들어도 같은 기록이 나오는 것도 함께 확인되었다.
