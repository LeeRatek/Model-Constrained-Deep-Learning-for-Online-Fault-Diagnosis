# Spec: 불필요한 코드 제거

**Status:** resolved

출처: 2026-10-16 사용자 요청("불필요한 코드 제거")과 같은 날 grilling(Q1–Q8). 용어는 `GLOSSARY.md`를 따른다.

## Problem Statement

후보 1–7 리팩토링 뒤에도 실행 경로와 무관한 코드가 남아 있다. AST로 집계한 결과는 다음과 같다.

| 종류 | 내용 |
|---|---|
| 쓰이지 않는 import | Class_ 10개(torchvision, scipy.io, sklearn 포함), 스크립트 5개에서 각 5~7개 |
| 읽히지 않는 변수·덮어써지는 대입 | 예: Train_ `lr_decay_freq`, Test_multi의 첫 `combinations_phase2` |
| 남은 주석 코드 | 예전 구현이나 다른 선택지를 보존해 둔 줄 |
| LSTM 레거시 | `--lstm-*` 옵션 5개, 학습 코드, helper 2개, `LSTM`·`MyDataset`. 학습 결과를 저장하지 않는다 |
| 쓰이지 않는 CLI 옵션 | Test_ `--sigma-levels`, Test_multi `--x-start`·`--x-tick-step`·`--sigma-levels`, Train_ `--vin1/2/3-start` |
| 쓰이지 않는 메서드 | `CombinedAE.analyze_outputs`(268줄, 호출하는 곳 없음) |
| 파일 | `models/` 루트 구버전 산출물 5개(git 추적, 그대로 쓸 수 없음), `README.txt`, `check_data_stats.py` |

## Solution

1. 동작과 무관한 것(Q1·Q2)을 먼저 지우고, 결과가 bit 단위로 같은지 확인한다.
2. 레거시 기능과 쓰이지 않는 옵션·메서드(Q3–Q5)를 지운다. 이 단계에서 바뀌는 것은 CLI(지운 옵션은 오류)와 새 Run의 sim_config 줄(지운 옵션 키)뿐이다.
3. 파일을 정리하고(Q7) README를 갱신한다.

## User Stories

1. 연구자로서, 정리 뒤에도 학습·평가 결과(체크포인트, loss, 관리도, CI, AUC)가 bit 단위로 같길 원한다.
2. 연구자로서, 쓰지 않는 옵션과 레거시 코드 없이 스크립트를 읽고 싶다.
3. 연구자로서, 문서화된 분석 모드(Test_ 진단 그림, data_filter 그림·분포 모드)는 그대로 쓰고 싶다.
4. 개발자로서, 무거운 import(torchvision 등) 없이 스크립트를 빨리 시작하고 싶다.
5. 리뷰어로서, 결과가 바뀌는 지점(sim_config의 지운 키)이 정확히 무엇인지 티켓에서 보고 싶다.

## Implementation Decisions

- **Q1. 동작과 무관한 정리**
  - 대상: 쓰이지 않는 import, 읽히지 않는 변수, 덮어써지는 대입, 쓰이지 않는 지역 import(`tempfile`), 의미 없는 주석(`# from sklearn.datasets import load_boston`)
  - 오탐 제외: dataclass 필드, Enum 멤버, 튜플 언패킹의 일부(`fig, ax = …`처럼 한쪽만 쓰는 것)는 지우지 않는다.
- **Q2. 주석 코드:** 예전 구현이나 대안을 보존한 주석 코드를 지운다. 설명용 주석은 남긴다.
- **Q3. LSTM 레거시:** 다음을 지운다.
  - Train_: `--lstm-training`, `--lstm-load`, `--lstm-epochs`, `--lstm-lr`, `--lstm-batchsize`, LSTM 학습 블록, `prepare_training_data`, `series_to_supervised`
  - Class_: `LSTM`, `MyDataset`, `INPUT_SIZE`
  - data_filter: `LSTM_LOAD` 분기
  - `vin_1.pkl` 데이터는 건드리지 않는다.
- **Q4. 쓰이지 않는 CLI 옵션:** Test_ `--sigma-levels`, Test_multi `--x-start`·`--x-tick-step`·`--sigma-levels`, Train_ `--vin1-start`·`--vin2-start`·`--vin3-start`를 지운다. data_filter의 `vin*_start` 상수는 PLOT_MODE가 쓰므로 남긴다.
- **Q5.** `CombinedAE.analyze_outputs`를 지우고, Class_의 `mcdl.plotting` import도 지운다.
- **Q6. 남기는 것:** Test_ `AUC_ANALYSIS`, data_filter `mode`(PLOT/ANALYSIS)
- **Q7. 파일**
  - 지움: `models/net.pth`, `models/netx.pth`, `models/pca_arrays.npz`, `models/pca_data_nor.csv`, `models/pca_manifest.json`(git 추적 해제), `README.txt`, `check_data_stats.py`
  - 남김: `backup/`

## Testing Decisions

- **01:** pytest, 실데이터 6개 대상(train, train_full, test, test_u0, tsne, multi) bit 일치. 비교 기준은 train_full은 `tl00a`, 나머지는 `after_bugs`다.
- **02**
  - pytest(smoke 포함)
  - 실데이터 6개 대상: sim_config를 뺀 모든 기록이 bit 일치해야 한다.
  - train·train_full의 sim_config는 지운 옵션 키 8줄(`lstm_*` 5개, `vin*_start` 3개)만 빠져야 한다.
  - 지운 옵션을 주면 argparse 오류가 나는지 확인한다.
- **03:** pytest, README 링크와 구조 확인

## Out of Scope

- `AUC_ANALYSIS`, data_filter `mode`를 CLI 옵션으로 바꾸는 것(Q6)
- `backup/`
- mcdl module 함수의 인자 정리
