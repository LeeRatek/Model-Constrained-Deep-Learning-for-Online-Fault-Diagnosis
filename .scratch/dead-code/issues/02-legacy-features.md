# 02: LSTM 레거시, 쓰이지 않는 CLI 옵션, `analyze_outputs`

**What to build:** LSTM 레거시 일체, 쓰이지 않는 CLI 옵션 6개, `CombinedAE.analyze_outputs`를 지운다. 학습·평가 결과는 그대로다. 새 Run의 sim_config에서는 지운 옵션 키만 빠진다.

**Blocked by:** 01

**Status:** resolved

- [x] Train_: `--lstm-*` 5개, LSTM 학습 블록, `prepare_training_data`, `series_to_supervised`를 지운다.
- [x] Class_: `LSTM`, `MyDataset`, `INPUT_SIZE`를 지운다. data_filter: `LSTM_LOAD` 분기를 지운다.
- [x] CLI 옵션: Test_ `--sigma-levels`, Test_multi `--x-start`·`--x-tick-step`·`--sigma-levels`, Train_ `--vin1/2/3-start`를 지운다.
- [x] Class_: `CombinedAE.analyze_outputs`와 `mcdl.plotting` import를 지운다.
- [x] 실데이터
  - 6개 대상의 sim_config 외 모든 기록이 bit 단위로 일치해야 한다.
  - train·train_full의 sim_config 차이는 지운 키 8줄뿐이어야 한다.
- [x] pytest(smoke 포함), 지운 옵션을 주면 argparse 오류가 나는지 확인한다.

## Comments

**2026-10-16 완료**

- Train_
  - `--lstm-*` 5개와 `--vin1/2/3-start`를 지웠다.
  - LSTM 레거시를 지웠다: helper 2개, `LSTM_TRAINING`·`LSTM_LOAD`, 하이퍼파라미터 블록, 차량 루프 안의 `vin_1` 로딩·LSTM 학습 블록
  - 이 때문에 쓰이지 않게 된 import도 지웠다: `Class_`(LSTM·MyDataset), pandas 2줄, `safe_load`
  - 중복 import `from torch import nn`(`import torch.nn as nn`과 같은 이름)도 지웠다. 기준선을 뜨는 중에 발견해 바로 뒤에 지웠고, pytest로 확인했다.
  - Train_.py는 439줄이 됐다.
- Class_: `MyDataset`, `INPUT_SIZE`, `LSTM`, `CombinedAE.analyze_outputs`와 그 import(`Dataset`, `mcdl.plotting`)를 지웠다. Class_.py는 53줄이다(`CustomSigmoidFunc`, `CombinedAE`).
- data_filter: `LSTM_LOAD`와 `vin_1` 로딩 블록을 지웠다.
- Test_ `--sigma-levels`, Test_multi `--x-start`·`--x-tick-step`·`--sigma-levels`를 지웠다.
- 지운 옵션(`--lstm-training`, `--vin1-start`, `--sigma-levels`, `--x-start`)을 주면 종료 코드 2와 "unrecognized arguments" 오류가 난다.
- 정적 검사: 정의되지 않은 이름 0건, 쓰이지 않는 import 0건. pytest 82개 통과(smoke 포함).
- 실데이터(기준선 태그 `dc02`, sim_config 외 기록은 bit 단위 비교)

  | 대상 | 비교 기준 | sim_config 외 불일치 |
  |---|---|---|
  | train | `after_bugs` | 0 / 14 |
  | train_full | `tl00a` | 0 / 17 |
  | test | `after_bugs` | 0 / 304 |
  | test_u0 | `after_bugs` | 0 / 513 |
  | tsne | `after_bugs` | 0 / 24 |
  | multi | `after_bugs` | 0 / 304 |

  - train·train_full의 sim_config에서는 지운 옵션 8줄만 빠졌다: `lstm_batchsize`, `lstm_epochs`, `lstm_load`, `lstm_lr`, `lstm_training`, `vin1_start`, `vin2_start`, `vin3_start`
