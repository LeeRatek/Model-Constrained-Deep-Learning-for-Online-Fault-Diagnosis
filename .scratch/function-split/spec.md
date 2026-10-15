# Spec: Function_.py 정리

**Status:** resolved

출처: 2026-10-15 2차 아키텍처 리뷰의 7번 후보("Function_.py 나머지 정리")와 같은 날 grilling(Q1–Q5). 용어는 `GLOSSARY.md`를 따른다.

## Problem Statement

Function_.py(2,794줄)에는 성격이 다른 코드가 섞여 있다.
- 원본 로딩·전처리
- 그림
- Run 폴더 규칙
- 모델 활성화 함수
- LSTM 레거시
- 쓰이지 않는 코드

스크립트는 `from Function_ import *`로 이 모두를 받는다. 그래서 어떤 스크립트가 무엇에 기대는지 import만 보고 알 수 없다. AST로 실제 참조를 집계한 결과는 다음과 같다.

| 묶음 | 개수 | 줄 수 |
|---|---|---|
| 스크립트·mcdl·tests가 쓰는 것 | 17개 | 1,278줄 |
| 그 함수들의 helper | 8개 | 412줄 |
| 주석 처리된 코드에서만 쓰이는 것 | 9개 | 438줄 |
| 어디서도 쓰이지 않는 것 | 18개 | 524줄 |

`monitering.py`는 Test 계열 스크립트가 star import하지만 아무것도 쓰지 않는다. Class_의 `Dataset`(5-tensor)·`RMSELoss`도 쓰이지 않는다. 특히 `Dataset`은 star import로 torch의 `Dataset` 이름을 가린다.

## Solution

쓰이는 25개(공개 17개 + helper 8개)를 성격에 맞는 곳으로 옮기고, 나머지 27개와 Function_.py를 지운다. 스크립트의 star import는 명시적 import로 바꾼다. 함수 본문과 이름은 바꾸지 않으므로 수치 결과는 그대로다.

## User Stories

1. 연구자로서, 정리 뒤에도 모든 스크립트의 결과가 bit 단위로 같길 원한다.
2. 개발자로서, 스크립트의 import만 보고 무엇에 기대는지 알고 싶다.
3. 개발자로서, 원본 로딩·전처리 코드를 차량 데이터 module(`mcdl/vehicles.py`) 가까이에서 찾고 싶다.
4. 개발자로서, Run 폴더 규칙(폴더 이름, 체크포인트 저장)을 `mcdl/run.py` 한 곳에서 보고 싶다.
5. 개발자로서, 쓰이지 않는 코드 때문에 헷갈리고 싶지 않다. 필요하면 git 기록에서 찾는다.
6. 리뷰어로서, 각 티켓을 마칠 때 기준선 비교 결과가 티켓에 남길 원한다.

## Implementation Decisions

### 옮길 곳 (Q1)

| 옮길 곳 | 함수 |
|---|---|
| `mcdl/preprocessing.py` | `safe_load`, `get_input_dimensions`, `get_preprocessing_and_skip_charge_ready`, `preprocess_loaded_tensor`, `is_abnormal` + helper `_to_cpu`, `last_index_of`, `get_convergence_start_index`, `normalize_columns_0_to_1` |
| `mcdl/plotting.py` | `plot_loss_curve_all`, `plot_roc_curve`, `plot_diagnostics_triplet`, `plot_testX_timeseries`, `plot_distribution_with_stats`, `plot_net_layer_params_by_index`, `plot_tsne_before_after_ci`, `draw_auc_heatmap` + helper `plot_loss_curve`, `load_loss_csv`, `plot_timeseries` |
| `mcdl/run.py` | `save_net_state`, `make_model_path_based_timestamp` |
| `Class_.py` | `CustomSigmoidFunc` |
| `Train_.py` | `prepare_training_data`, `series_to_supervised` (LSTM 레거시) |

- 함수 본문과 이름은 그대로 옮긴다. 새 module에는 함수가 실제로 쓰는 import만 둔다.
- 의존 방향
  - `mcdl.run` → `Class_`, `mcdl.preprocessing`
  - `Class_` → `mcdl.plotting`
  - `mcdl.vehicles` → `mcdl.preprocessing`
  - `mcdl.plotting`과 `mcdl.preprocessing`은 프로젝트 내부 module을 import하지 않는다.
- Function_.py는 지운다. 호환용 파일은 두지 않는다.

### 지울 것 (Q2–Q4)

- **쓰이지 않는 18개:** `solvers`, `SPE`, `save_lstm_model`, `load_lstm_model`, `load_net_state_dict`, `load_net_into`, `cont`, `Ratio_cu`, `SlidingAverage_list`, `SlidingAverage`, `ClassifyFeature`, `get_start_index`, `normalize_columns_by_mean_std`, `read_learning_case_from_sim_config`, `plot_net_layer_weight_bias_bars`, `find_min_idx_from_array`, `compare_max_error_data`, `get_rmse_stats_from_error`
- **주석 처리된 코드에서만 쓰이는 9개:** `calculate_volt_modepi`, `T2`, `chi_square_dist_components`, `diagnosis_thresholds`, `plot_array`, `load_vehicle_ids_used_for_training`, `plot_ae_output_distribution`, `find_max_idx_from_array`, `plot_contrib_percent_stacked`
- **스크립트의 주석 블록:** 대상은 Test_, Test_tsne, Train_, data_filter다. 지운 함수나 이미 없는 변수(`ERRORU`, `recon_imtest`, `x_recovered` 등)를 참조하는 주석 블록이다.
- **기타:** `monitering.py`, Class_의 `Dataset`(5-tensor)과 `RMSELoss`
- 스크립트의 `from Function_ import *`, `from Class_ import *`, `from monitering import *`는 명시적 import로 바꾼다.

## Testing Decisions

- **pytest:** 기존 테스트의 import 경로를 새 위치로 바꾸고 전부 통과시킨다. smoke 테스트(`tests/test_scripts.py`)가 바뀐 import로 학습과 평가를 끝까지 실행한다.
- **기준선 도구:** t-SNE 입력 wrapper의 위치에 `mcdl.plotting`을 더한다.
- **실데이터(`after_bugs`, train_full은 `tl00a`), 티켓별**
  - 01: train, test_u0. data_filter는 정리 전 코드(git worktree)와 정리 뒤 코드로 한 번씩 실행해 분할 `.npy` 5개와 dataset_info를 비교한다.
  - 02: test, tsne, multi
  - 03: train, train_full, test, test_u0, tsne, multi 전체

## Out of Scope

- 함수 본문 수정, 이름 변경
- Class_의 다른 정리(쓰이지 않는 import 등)
- plotting 함수의 인자 정리. 예: `plot_tsne_before_after_ci`가 `args`를 받는 것
