# Spec: LSTM 학습과 칼만 필터 복원

**Status:** resolved

출처: 2026-10-19 사용자 요청("main 브랜치의 LSTM이 어떻게 구현·동작하는지 보고 복원, Function_의 칼만 필터 등 복원")과
같은 날 결정(Q1–Q3). 용어는 `GLOSSARY.md`를 따른다.

## Problem Statement

- LSTM은 dead-code 정리(081bdbb, `.scratch/dead-code/spec.md` Q3)에서 지웠다. 원본(main, 260189c)에서도 결과를 저장하지 않았다.
  - 모델: 양방향 LSTM(입력 7, 은닉 18) + Linear(36 → 2). `vin_1.pkl`(T, 1, 7)의 한 시점 전 값 7개 → 현재 시점의 마지막 두 열(대표 셀 전압, SOC).
  - 학습: Adam lr 1e-4, batch 100, shuffle, MSE. main은 차량 0번 하나, 지우기 직전 Train_은 차량마다 새 LSTM을 학습했고 둘 다 저장하지 않았다.
  - main Test_은 `models/lstm.pth`를 불러 예측했지만 예측값을 쓰지 않았다.
- 칼만 필터(`calculate_volt_modepi`, `solvers`)는 함수 분리(a93ea3a, `.scratch/function-split/spec.md`)에서 지웠다. main에서도 부르는 곳이 없었다.
  - `solvers`: 팩 1차 RC 등가회로 + 적응형 확장 칼만 필터(상태: 분극 전압, SOC), 셀마다 스칼라 칼만 필터(SOC 편차).
  - OCV 계수 `b`(21개)는 저장소와 데이터에 없다. 데이터의 vin_2·vin_3 x·Δ 열에 이런 추정 결과가 이미 들어 있다(README 2장 "Kalman 예측, LSTM 예측").

## Decisions (2026-10-19)

- **Q1 LSTM 동작:** 저장되는 형태. `Train_.py --lstm-training`이면 학습 차량 전체의 vin_1로 LSTM 하나를 학습해
  `artifact/lstm.pth`(모델 객체째, 원본 저장 방식)와 `loss_lstm.csv`를 저장한다. 구조·하이퍼파라미터·데이터 구성은 원본 그대로.
  옵션은 지우기 전 이름·기본값(`--lstm-epochs 300`, `--lstm-lr 1e-4`, `--lstm-batchsize 100`). `--lstm-load`는 되살리지 않는다.
  MC-AE 학습·관리도 저장이 끝난 뒤에 학습해, 이 옵션이 MC-AE 결과를 바꾸지 않게 한다.
- **Q2 복원 범위:** 칼만 필터(`calculate_volt_modepi`, `solvers`)와 LSTM 도우미(`series_to_supervised`,
  `prepare_training_data`, `save_lstm_model`, `load_lstm_model`). 그 밖의 지운 함수와 `monitering.py`는 복원하지 않는다.
- **Q3 위치:** `mcdl/kalman.py`, `mcdl/lstm.py`, `Class_.py`의 `LSTM`.

### 2026-10-19 변경: LSTM 학습을 MC-AE와 분리 (Q1 대체)

- 사용자 지적: Q1 구현은 `Train_.py --lstm-training`이 MC-AE 학습 뒤에 LSTM을 덧붙이는 방식이라, LSTM만 학습할 수 없었다.
  LSTM은 MC-AE와 의존성이 없어야 한다.
- 사용자 결정: LSTM 전용 스크립트. `Train_lstm.py`를 만들고 `Train_.py`의 `--lstm-*` 옵션을 없앴다(Train_.py는 MC-AE 전용, LSTM을 넣기 전 f632fd3와 같은 동작).
  - `Train_lstm.py`: vin_1만 읽어 LSTM 하나를 학습하고 자기 Run 폴더에 `sim_config.txt`(`model : LSTM`), `loss_lstm.csv`, `artifact/lstm.pth`만 저장한다.
    옵션은 `--battery-type`, `--dataset-name`, `--vehicle-start/end`, `--models-dir`, `--source-data-dir`, `--epochs`(300), `--lr`(1e-4), `--batchsize`(100), `--seed`(42).
  - 함께 쓰는 코드: `train_lstm`은 `mcdl/lstm.py`로, `set_seed`는 `mcdl/run.py`로 옮겼다. 분할 폴더 규칙 `_split_dir`는 `resolve_split_dir`로 공개해
    learning_case 없이도 쓴다.
  - `Run`은 sim_config에 `model : LSTM`이 있으면 "LSTM만 학습한 Run" ValueError로 멈춘다.
- 검증: `tests/test_scripts.py::test_train_lstm_needs_only_vin1_and_saves_only_the_lstm`(vin_1만 있는 데이터 폴더로 실행, 저장 파일 3개, Run 거부,
  같은 시드 재현). 실데이터 train·train_full·test 기록(sp01)이 fx01과 sim_config까지 같다. 실데이터 LSTM 전용 학습(차량 2대, 5 epoch) 14초,
  loss 0.175 → 0.005.

## Verification

- `tests/test_kalman.py`: 원본 코드(a93ea3a^의 Function_.py)로 합성 입력에서 만든 기대값(`tests/fixtures/kalman_golden.npz`, 10개 항목)과 bit 단위로 같다.
- 원본 함수와 직접 비교: 실제 vin_1(QAS 0·338, DTI 0)의 `prepare_training_data` 결과, 같은 시드의 LSTM 초기 가중치·출력이 같다.
- `tests/test_lstm.py`, `tests/test_scripts.py::test_train_lstm_is_saved_and_leaves_mc_ae_unchanged`.
- 실데이터 기준선(태그 ls01, 비교 fx01): test는 모든 기록이 같고, train·train_full은 sim_config.txt에 `lstm_*` 4줄이 더해진 것 말고 같다.
- 실데이터 학습(차량 2대, 5 epoch): loss 0.101 → 0.0045, `lstm.pth` 저장·불러오기 확인. 이 짧은 학습으로는 학습에 쓰지 않은 차량 338에서
  "다음 값 = 지금 값" 예측보다 오차가 크다(전압 MSE 8.0e-3 vs 1.3e-3, SOC 6.9e-2 vs 2.0e-5). 예측 성능은 평가하지 않았다.
