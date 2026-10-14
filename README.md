Model-Constrained Deep Learning for Online Fault Diagnosis in Li-ion Batteries over Stochastic Conditions for the Proposed Scenario Code. 
The meaning of the code file is as follows:
Train: Training of the network in Fig 1(b) and Fig 1(c) of the proposed method with the directed linear layer in Fig 1(d)
Test: Test on sample data
Class: the desired class
Function: the required function
Dataset acquisition link: https://doi.org/10.5281/zenodo.10656500

# Model-Constrained Deep Learning for Online Fault Diagnosis

**Model-Constrained Deep Learning for Online Fault Diagnosis in Li-ion Batteries over Stochastic Conditions** 논문의 제안 기법(MC-AE + PCA 기반 T²/SPE 진단) 코드를 바탕으로, 데이터 분할 → 학습 → 체크포인트별 평가(시뮬레이션) → 시각화 파이프라인을 정리한 저장소입니다.

- 원본 데이터셋: <https://doi.org/10.5281/zenodo.10656500>

## 목차

1. [빠른 시작](#1-빠른-시작)
2. [방법 개요](#2-방법-개요)
3. [저장소 구조](#3-저장소-구조)
4. [환경 설정](#4-환경-설정)
5. [데이터 준비](#5-데이터-준비)
6. [모듈 구성](#6-모듈-구성)
7. [스크립트별 사용법](#7-스크립트별-사용법)
8. [결과물 정리](#8-결과물-정리)
9. [주의사항 및 알려진 제약](#9-주의사항-및-알려진-제약)

---

## 1. 빠른 시작

모든 스크립트는 **저장소 루트에서** 실행합니다 (`./data`, `./datasets`, `./*.npy` 상대 경로를 사용).

```bash
# 0) (선택) 차량 분할 새로 만들기 → ./datasets/10/   (datasets/2, 5, 10, 15는 이미 포함되어 있음)
python data_filter.py --dataset-name 10 --train-ratio 0.1 --val-ratio 0.5 --seed 42

# 1) 학습 → ./models/<RUN_ID>/ 생성   (RUN_ID 예: 260224_090359)
python Train_.py --battery-type QAS --dataset-name 10 \
    --ae-u-epochs 13 --ae-x-epochs 13 --ae-u-save-per 1 --ae-x-save-per 1

# 2) 시뮬레이션: 저장된 모든 체크포인트 (U, X) 조합의 AUC 계산
python Test_multi_models.py --models-idx <RUN_ID> --keep-cache

# 3) (선택) 단일 조합 테스트 / t-SNE 시각화
python Test_.py --models-folder <RUN_ID>
python Test_tsne.py --models-idx <RUN_ID> --u-model-idx -1 --x-model-idx -1 --fault-idx 338 --save
```

`--dataset-name`을 생략하면 저장소 루트의 기본 분할(`QAS_filtered_vehicle_ids_*.npy`)을 사용합니다.

### 목적별 실행 파일

이 저장소에서 **시뮬레이션**은 학습된 체크포인트로 테스트 차량의 온라인 진단을 재현해 ROC/AUC 등 성능 지표를 산출하는 과정을 말합니다.

| 목적 | 실행 파일 | 주요 결과물 |
|---|---|---|
| 차량 분할 (train / validate / test / fault) | `data_filter.py` | `datasets/<name>/*.npy`, `dataset_info_<BT>.txt` |
| 학습 | `Train_.py` | `artifact/*.pth`, `loss_*.csv`, `pca_*`, `sim_config.txt` |
| 테스트 (모델 조합 1개) | `Test_.py` | `results/AUC_ROC<C>.png`, 콘솔의 AUC·최적 임계값 |
| 시뮬레이션 (체크포인트 조합별 성능 비교, 모델 선택) | `Test_multi_models.py` | `AUC_summary_case<C>.csv`, `results/AUC_ROC_*.png`, `AUC_heatmap_case<C>.png` |
| AUC 히트맵 다시 그리기 | `Plot_AUC_heatmap.py` | `AUC_summary_case<C>_heatmap.png` |
| t-SNE + CI 시각화 | `Test_tsne.py` | `results_tsne/tSNE_<id>.png` |
| 차량별 T²·SPE·CI 시계열 그림 | `Test_.py` (코드에서 `AUC_ANALYSIS = False`) | `results/<id>_testing_diagnostics_case<C>.png` |

- 결과물 경로는 `data_filter.py`를 제외하고 모두 `models/<RUN_ID>/` 기준입니다.
- `<C>`는 learning_case(기본 1), `<BT>`는 배터리 타입(QAS/DTI), `<id>`는 차량 ID입니다.

---

## 2. 방법 개요

두 개의 MC-AE(`CombinedAE`)를 **정상 차량 데이터**로 학습해 각각 셀 전압과 셀 SOC를 재구성하게 하고, 재구성 오차로 만든 진단 특징을 PCA 관리도(T², SPE)로 감시합니다.

```
정상 차량 데이터
  ├─ U-model (net)  : 셀 전압 U 재구성   ← vin_2.pkl
  └─ X-model (netx) : 셀 SOC  X 재구성   ← vin_3.pkl
        │
        ▼
셀별 재구성 오차 |Û - U|, |X̂ - X|   (QAS 110셀 / DTI 85셀)
        │
        ▼
시점별 진단 특징 6개                       DiagnosisFeature()
        │
        ▼
학습 데이터로 PCA → T², SPE 관리한계(95%)    Custom_PCA()
        │
        ▼
CI(t) = T²(t) / T²_95 + SPE(t) / SPE_95
        │
        ▼
차량 판정: max_t CI(t) > θ 이면 고장
(θ를 0 ~ 999.9, 0.1 간격으로 바꿔 가며 ROC/AUC 계산)
```

### CombinedAE 구조 ([Class_.py](Class_.py))

논문 Fig. 1(b)(c)의 네트워크와 Fig. 1(d)의 directed linear layer에 해당합니다.

```
z = fc1(x) + sigmoid(fc2(q)) + Δ        # fc1: Linear(2 → 1), fc2: Linear(q → 1)
z = fc4(z)                              # --add-one-output-layer 일 때만, Linear(N → N)
ŷ = shift + scale · sigmoid(fc3(z))     # fc3: Linear(N → N)
```

| 기호 | U-model (`net`) | X-model (`netx`) |
|---|---|---|
| x (2) | 팩 전압 추정값 2개 (Kalman 예측, LSTM 예측) | 팩 SOC 추정값 2개 |
| Δ (N) | 셀별 전압 편차 추정값 | 셀별 SOC 편차 추정값 |
| q | 보드 온도, 보드단 SOC, 전류 (3) | 보드 온도, 보드단 SOC, 속도, 전류 (4) |
| y, ŷ (N) | 셀 전압 (정답 / 재구성) | 셀 SOC (정답 / 재구성) |
| 출력 범위 | scale 1.8, shift 2.5 → 2.5 ~ 4.3 | scale 1.0, shift 0 → 0 ~ 1 |

- `--no-use-dx`이면 Δ = 0으로 둡니다.
- learning_case 2·3(0~1 정규화 사용)에서는 scale = 1, shift = 0을 사용합니다.
- 손실은 MSE(ŷ, y), 옵티마이저는 Adam이며 double precision으로 학습합니다.

### 진단 특징 6개 ([Function_.py](Function_.py) `DiagnosisFeature`)

매 시점마다 셀 방향으로 재구성 오차 e를 요약합니다.

| 열 | 표기 | 정의 |
|---|---|---|
| 0 | zu₂ | 전압 오차의 (1위 − 2위) / 셀 간 표준편차 |
| 1 | zx₂ | SOC 오차의 (1위 − 2위) / 셀 간 표준편차, 100-step 이동평균 |
| 2 | zu₁ | 전압 오차의 셀 간 최대 z-score |
| 3 | zx₁ | SOC 오차의 셀 간 최대 z-score, 100-step 이동평균 |
| 4 | ewu | 셀 최대 전압 오차의 EWMA (α = 0.2) |
| 5 | ewx | 셀 최대 SOC 오차의 EWMA (α = 0.2), 100-step 이동평균 |

- 특징은 차량별로 계산한 뒤 이어 붙입니다. 이동평균과 EWMA가 차량 경계를 넘지 않게 하기 위해서입니다.
- PCA는 표준화한 6개 특징에 대해 누적 기여율 95% 기준으로 주성분 수를 정합니다.
- T² 관리한계는 F 분포, SPE 관리한계는 Jackson–Mudholkar 근사로 계산합니다.

---

## 3. 저장소 구조

```
.
├── Train_.py               [실행] MC-AE(U·X) 학습 + 학습 데이터 PCA 관리한계 저장
├── Test_.py                [실행] 단일 (U, X) 모델 조합 평가 → ROC/AUC
├── Test_multi_models.py    [실행] 체크포인트 (U, X) 조합 전수 평가 → AUC 표·ROC·히트맵
├── Test_tsne.py            [실행] 기준 정상 차량 vs 대상 차량 t-SNE + CI 시각화
├── Plot_AUC_heatmap.py     [실행] AUC_summary CSV → 히트맵 (재)생성
├── data_filter.py          [실행] 이상 차량 제거 + train/validate/test 분할
├── Train_pca_only.py       [모듈] 지정 체크포인트로 PCA만 재계산 (Test_.py가 import)
├── Class_.py               [모듈] CombinedAE, LSTM, Dataset 클래스
├── Function_.py            [모듈] 로딩·전처리·진단 특징·PCA·T²/SPE·ROC·플롯·설정 I/O
├── monitering.py           [참고] 외부 T²/Q 관리도 클래스 (import만 되고 사용하지 않음)
├── check_data_stats.py     [분석] DTI 초기 수렴 구간 탐색용 일회성 스크립트
├── QAS_filtered_vehicle_ids_{normal,fault,train,validate,test}.npy   QAS 기본 분할
├── DTI_filtered_vehicle_ids_{normal,fault,train,validate,test}.npy   DTI 기본 분할
├── datasets/{2,5,10,15}/   data_filter.py로 만든 QAS 분할 (학습 비율 2·5·10·15 %)
├── backup/                 이전 분할 파일 백업 (20260205/, past/)
├── data/                   원본 데이터 (git 제외, 직접 배치)
├── models/                 학습·평가 결과 (git 제외, 실행 시 생성)
├── README.md
└── README.txt              구버전 간단 설명
```

### 차량 분할 파일

| 분할 | QAS 기본 (루트) | DTI 기본 (루트) | `datasets/2` | `datasets/5` | `datasets/10` | `datasets/15` |
|---|---|---|---|---|---|---|
| normal | 309 (ID 0~334) | 77 (ID 0~76) | 309 | 309 | 309 | 309 |
| fault | 50 (ID 335~392) | 3 (ID 77~79) | 50 | 50 | 50 | 50 |
| train | 209 | 39 | 6 | 15 | 30 | 46 |
| validate | 50 | 19 | 151 | 147 | 139 | 131 |
| test | 50 | 19 | 152 | 147 | 140 | 132 |

- normal = train + validate + test입니다. 평가에는 **test(정상) + fault(고장)** 차량을 사용합니다.
- `datasets/*`는 모두 `--val-ratio 0.5 --seed 42`로 생성되었고, 폴더마다 `dataset_info_QAS.txt`에 분할별 샘플 수가 기록되어 있습니다.

---

## 4. 환경 설정

기존 실행 기록(`sim_config.txt`) 기준 환경은 Linux, Python 3.12.12, CPU입니다.

| 패키지 | 버전 |
|---|---|
| torch / torchvision | 2.9.1 / 0.24.1 |
| numpy | 2.3.5 |
| pandas | 2.3.3 |
| scikit-learn | 1.8.0 |
| scipy | 1.16.3 |
| matplotlib | 3.10.8 |
| seaborn | 0.13.2 |

```bash
pip install torch==2.9.1 torchvision==0.24.1 numpy==2.3.5 pandas==2.3.3 \
    scikit-learn==1.8.0 scipy==1.16.3 matplotlib==3.10.8 seaborn==0.13.2
```

`requirements.txt`는 `.gitignore`에 포함되어 있어 저장소에 올라가지 않습니다. 로컬에 있으면 `pip install -r requirements.txt`로 설치해도 됩니다.

### 환경 변수

| 변수 | 대응 옵션 | 기본값 | 적용 스크립트 |
|---|---|---|---|
| `MODEL_DIR` | `--models-dir` | `./models` | Train_, Test_multi_models, Test_tsne, Plot_AUC_heatmap (Test_.py는 미적용) |
| `SOURCE_DIR` | `--source-data-dir` | `./data` | Train_, Test_, Test_multi_models, Test_tsne |

`data_filter.py`는 `./data` 경로를 코드에 고정해서 사용합니다.

---

## 5. 데이터 준비

Zenodo 데이터셋을 받아 `data/` 아래에 다음 구조로 둡니다.

```
data/
├── QAS/                    차량 ID 0 ~ 392 (393대), 셀 110개
│   ├── 0/
│   │   ├── vin_1.pkl       LSTM 입력 (T, 1, 7)
│   │   ├── vin_2.pkl       U-model 입력 (T, 225)
│   │   └── vin_3.pkl       X-model 입력 (T, 226)
│   ├── 1/ ...
│   └── Labels.xls
├── DTI/                    차량 ID 0 ~ 79 (80대), 셀 85개 → vin_2 (T, 175), vin_3 (T, 176)
├── GIS/                    데이터셋에 포함되어 있으나 현재 스크립트에서 지원하지 않음
└── *.xlsx                  논문 Figure 원본 데이터 (코드에서 사용하지 않음)
```

- 정상/고장은 **차량 ID 범위로 구분**합니다: QAS는 ID ≥ 335, DTI는 ID ≥ 77이 고장입니다. `Labels.xls`는 코드에서 읽지 않습니다.
- `.pkl`은 pickle된 torch 텐서이며, `Function_.safe_load()`가 CPU로 로드합니다.
- 원본 크기는 QAS 약 49 GB, DTI 약 9 GB입니다.

### `vin_2.pkl` / `vin_3.pkl` 열 구성

N = 110 (QAS) / 85 (DTI)이며, 열 구간은 `get_input_dimensions()`에 정의되어 있습니다.

| 열 구간 | `vin_2.pkl` (U-model) | `vin_3.pkl` (X-model) |
|---|---|---|
| `[0, 2)` → x | 팩 전압 추정값 2개 | 팩 SOC 추정값 2개 |
| `[2, 2+N)` → y | 셀 전압 (재구성 대상) | 셀 SOC (재구성 대상) |
| `[2+N, 2+2N)` → Δ | 셀 전압 편차 추정값 | 셀 SOC 편차 추정값 |
| `[2+2N, 끝)` → q | 보드 온도, 보드단 SOC, 전류 | 보드 온도, 보드단 SOC, 속도, 전류 |

### 전처리 (`learning_case`, `preprocess_loaded_tensor()`)

| learning_case | 충전 대기 구간 제거 (SKIP_CHARGE_READY) | 0~1 정규화 (PREPROCESSING) |
|---|---|---|
| **1 (기본)** | O | X |
| 2 | O | O |
| 3 | X | O |
| 4 | X | X |

- **충전 대기 구간 제거 (QAS)**: 전류 열(224)이 −500 미만인 마지막 시점까지를 잘라냅니다.
- **0~1 정규화 (QAS)**: 열 0이 0 이하이거나 5 초과인 마지막 시점까지를 잘라낸 뒤, Δ 구간을 제외한 모든 열을 min-max 정규화합니다. `--normalize-dx`를 주면 Δ를 `--normalize-val`로 나눕니다.
- QAS는 `vin_3`의 속도 열을 항상 0으로 둡니다.
- **DTI**는 별도 경로를 탑니다. 충전 대기 구간 제거 대신 열 0이 안정화되는 시점(20 step 연속 변화량 < 0.1)부터 사용하고, 단위 변환(mV → V, % → 비율 등)만 적용합니다. 0~1 정규화는 적용되지 않습니다.

---

## 6. 모듈 구성

### 의존 관계

| 파일 | import 하는 모듈 |
|---|---|
| `Train_.py` | Function_, Class_ |
| `Test_.py` | Function_, Class_, Train_pca_only, monitering |
| `Test_multi_models.py` | Function_, Class_, Train_pca_only(import만), monitering |
| `Test_tsne.py` | Function_, Class_, monitering |
| `data_filter.py` | Function_, Class_ |
| `Plot_AUC_heatmap.py` | Function_ (`draw_auc_heatmap`) |
| `Train_pca_only.py` | Function_, Class_, monitering |
| `Class_.py` | Function_ |

### Class_.py

| 클래스 | 설명 |
|---|---|
| `CombinedAE` | MC-AE 본체 (2장 구조 참고). `analyze_outputs()`로 중간 출력 시각화 가능 |
| `LSTM` | 양방향 LSTM (입력 7 → hidden 18×2 → 출력 2). `vin_1.pkl` 기반, 현재 파이프라인에서는 레거시 |
| `MyDataset`, `Dataset` | LSTM / 5-텐서용 Dataset |
| `RMSELoss` | RMSE 손실 (사용하지 않음) |

### Function_.py 주요 함수

| 분류 | 함수 |
|---|---|
| 로딩 | `safe_load` (pkl/pt/npy를 CPU로 로드) |
| 전처리 | `get_input_dimensions`, `get_preprocessing_and_skip_charge_ready`, `preprocess_loaded_tensor`, `is_abnormal`, `get_convergence_start_index`, `normalize_columns_0_to_1` |
| 진단 | `CustomSigmoidFunc`, `DiagnosisFeature`, `Custom_PCA`, `T2_array`, `SPE_array` |
| 평가 | `compute_roc_auc_from_threshold_matrix`, `compute_optimal_thresholds_from_roc` |
| 시각화 | `plot_roc_curve`, `draw_auc_heatmap`, `plot_diagnostics_triplet`, `plot_tsne_before_after_ci`, `plot_loss_curve_all`, `plot_net_layer_params_by_index`, `plot_testX_timeseries` |
| 저장·설정 | `save_net_state`, `save_pca_results`, `load_pca_results`, `make_model_path_based_timestamp`, `print_sim_config`, `read_values_from_sim_config`, `read_dataset_path_from_sim_config` |
| 현재 미사용 | `solvers`(Kalman 기반 추정), `calculate_volt_modepi`, `ClassifyFeature`, `cont`, `diagnosis_thresholds`, `plot_contrib_percent_stacked`, LSTM 저장/로드 등 |

### Train_pca_only.py

`train_pca_only(args, device, save=True, ...)`는 지정한 체크포인트(`args.u_model_idx`, `args.x_model_idx`)로 학습 차량의 재구성 오차 → 진단 특징 → PCA를 다시 계산합니다. `save=False`이면 PCA 결과 튜플을 반환하고, `save=True`이면 새 타임스탬프 폴더에 `pca_*`를 저장합니다. Test_.py의 PCA 대체 경로에서 사용합니다.

### monitering.py

외부 코드(Iurii Katser, control-charts)의 Hotelling T²/Q 통계 클래스입니다. 테스트 스크립트들이 `from monitering import *`로 불러오지만, 실제 진단에는 `Function_.py`의 `T2_array`/`SPE_array`를 사용합니다.

---

## 7. 스크립트별 사용법

### 7.1 데이터 분할 — `data_filter.py`

차량 0 ~ 최대 ID를 순회하며 전처리 후 `vin_2.pkl` 열 0 값이 0 미만이거나 5 초과인 시점이 있는 차량을 제외합니다(`is_abnormal`). 남은 차량을 ID 범위로 정상/고장으로 나누고, 정상 차량을 무작위로 train/validate/test로 분할합니다.

```bash
python data_filter.py --dataset-name 10 --train-ratio 0.1 --val-ratio 0.5 --seed 42
```

| 옵션 | 기본값 | 설명 |
|---|---|---|
| `--dataset-name` | `default` | 저장 폴더 `./datasets/<name>/` (`default`라는 이름은 피할 것, 9장) |
| `--train-ratio` | 없음 | 정상 차량 중 학습 비율. 미지정 시 QAS 109대 / DTI 39대 |
| `--val-ratio` | 없음 | **학습을 뺀 나머지** 중 검증 비율. 미지정 시 QAS 50대 / DTI 19대. 남은 차량이 test |
| `--seed` | 42 | 분할 난수 시드 |

코드 상단에 고정된 값이 있어, 바꾸려면 코드를 수정해야 합니다.

| 변수 | 고정값 | 비고 |
|---|---|---|
| `BATTERY_TYPE` | `"QAS"` | DTI 분할을 만들려면 `"DTI"`로 변경 |
| `SKIP_CHARGE_READY`, `PREPROCESSING` | `True`, `False` | learning_case 1과 동일 |
| `mode` | `MODE.FILTER_MODE` | 아래 모드 참고 |

| mode | 동작 |
|---|---|
| `FILTER_MODE` (기본) | 분할 파일 저장 |
| `PLOT_MODE` | 차량별 주요 열 시계열을 `data/data_analysis_at_charge_0/<BT>-<id>/vin2-<열>st.png`, `vin3-<열>st.png`로 저장 |
| `ANALYSIS_MODE` | 전체 차량을 이어 붙여 주요 변수(전압·온도·SOC·전류·편차) 분포를 화면에 표시 |

**결과물** (`./datasets/<name>/`)

- `<BT>_filtered_vehicle_ids_{normal,fault,train,validate,test}.npy`: 차량 ID 배열
- `dataset_info_<BT>.txt`: 분할별 차량 수, 샘플 수, 비율

### 7.2 학습 — `Train_.py`

```bash
python Train_.py --battery-type QAS --dataset-name 10 --learning-case 1 \
    --ae-u-epochs 13 --ae-x-epochs 13 --ae-u-save-per 1 --ae-x-save-per 1
```

**동작 순서**

1. 학습·검증 차량 리스트를 로드합니다 (`--dataset-name` 또는 루트 기본 분할).
2. 차량별 `vin_2.pkl`, `vin_3.pkl`을 로드하고 전처리합니다.
3. `models/<RUN_ID>/sim_config.txt`에 전체 인자, 학습 차량 ID, 학습 샘플 수 등을 저장합니다.
4. U-model을 학습합니다. `--ae-u-save-per` epoch마다 검증 loss를 계산하고 `net_e<epoch>.pth`를 저장하며, 종료 후 `net.pth`를 저장합니다.
5. X-model도 같은 방식으로 학습해 `netx_e<epoch>.pth`, `netx.pth`를 저장합니다.
6. 최종 모델로 학습 차량의 재구성 오차 → 진단 특징 → PCA를 계산해 `pca_*`를 저장합니다.

콘솔에는 저장 주기마다 `Epoch | Train Loss | Val Loss`와 총 소요 시간이 출력됩니다.

| 옵션 | 기본값 | 설명 |
|---|---|---|
| `--battery-type` | `QAS` | `QAS` 또는 `DTI` |
| `--dataset-name` | 없음 | `./datasets/<name>/` 분할 사용. 미지정 시 루트 `*.npy` |
| `--learning-case` | 1 | 전처리 조합 (5장 표) |
| `--models-dir` / `--source-data-dir` | `./models` / `./data` | 결과 저장 위치 / 원본 데이터 위치 |
| `--vehicle-start` / `--vehicle-end` | 0 / -1 | 학습 차량 리스트 슬라이스 (end 포함, -1은 끝까지) |
| `--ae-u-epochs` / `--ae-x-epochs` | **10 / 300** | U·X 모델 epoch. 기본값이 서로 다르므로 명시 권장 |
| `--ae-u-save-per` / `--ae-x-save-per` | 10 / 10 | 검증 + 체크포인트 저장 주기 (`epoch % N == 0`) |
| `--ae-u-lr` / `--ae-x-lr` | 5e-4 / 5e-4 | Adam 학습률 |
| `--ae-u-batchsize` / `--ae-x-batchsize` | 100 / 100 | 배치 크기 |
| `--ae-u-shuffle` / `--ae-x-shuffle` | off | 학습 데이터 셔플 |
| `--ae-u-scale`, `--ae-u-shift` | 1.8, 2.5 | U 출력 sigmoid 스케일·오프셋 (learning_case 1·4에서만 적용) |
| `--ae-x-scale`, `--ae-x-shift` | 1.0, 0 | X 출력 sigmoid 스케일·오프셋 (learning_case 1·4에서만 적용) |
| `--no-use-dx` | off | Δ(셀 편차) 입력 미사용 |
| `--add-one-output-layer` | off | 디코더 앞에 Linear(N → N) 추가 |
| `--no-abs-err` | off | 학습 데이터 PCA 특징 계산 시 부호 있는 오차 사용 (테스트 차량 오차는 항상 절댓값) |
| `--normalize-dx`, `--normalize-val` | off, 4 | learning_case 2·3에서 Δ를 normalize_val로 나눔 |
| `--no-validation` | off | 검증 데이터 로드·평가 생략 |
| `--no-save` | off | `.pth` 체크포인트 저장 생략 (loss·PCA·sim_config는 저장) |
| `--seed` | 42 | 난수 시드 |
| `--lstm-load`, `--lstm-training`, `--lstm-epochs`, `--lstm-lr`, `--lstm-batchsize` | off / 300 / 1e-4 / 100 | 레거시. `vin_1.pkl`로 LSTM을 학습하지만 결과는 저장하지 않음 |
| `--vin1-start`, `--vin2-start`, `--vin3-start` | 0 | 현재 사용하지 않음 (sim_config에 기록만 됨) |

**결과물**: `sim_config.txt`, `artifact/*.pth`, `loss_*.csv`, `pca_*` (상세는 [8장](#8-결과물-정리))

### 7.3 테스트 (단일 조합) — `Test_.py`

학습 폴더의 `sim_config.txt`에서 설정(battery_type, learning_case, 출력층 파라미터, dataset_name 등)을 읽어 학습과 같은 조건으로 평가합니다.

```bash
python Test_.py --models-folder <RUN_ID>                                    # 최종 모델 (net.pth, netx.pth)
python Test_.py --models-folder <RUN_ID> --u-model-idx 2 --x-model-idx 0    # 특정 체크포인트 (temp_cache 필요, 9장)
```

| 옵션 | 기본값 | 설명 |
|---|---|---|
| `--models-folder` | `260224_090359` | 평가할 `RUN_ID` (다른 스크립트의 `--models-idx`에 해당) |
| `--models-dir` | `./models` | `MODEL_DIR` 환경 변수 미적용 |
| `--source-data-dir` | `./data` | 원본 데이터 위치 |
| `--u-model-idx` / `--x-model-idx` | -1 / -1 | -1: `net.pth`/`netx.pth`, N: `net_eN.pth`/`netx_eN.pth` |
| `--x-start`, `--x-tick-step` | 20, 3000 | 진단 시계열 그림의 x축 (진단 그림 모드에서만 사용) |
| `--sigma-levels` | `3,4.5,6` | 현재 사용하지 않음 |

**PCA 관리한계를 정하는 순서**

1. `temp_cache/feat_u_<u>.pkl`, `feat_x_<x>.pkl`이 있으면 캐시된 학습 특징으로 PCA를 계산합니다.
2. 없고 u = -1이면 학습 때 저장한 `pca_arrays.npz`를 사용합니다.
3. 그 외에는 `train_pca_only()`로 다시 계산합니다 (현재 오류 발생, 9장).

**동작**: test(정상) + fault 차량마다 CI 최대값을 구하고, 임계값을 바꿔 가며 ROC/AUC를 계산합니다. 콘솔에 AUC, Youden J 기준 최적 임계값, (0, 1)에 가장 가까운 임계값을 출력하고 ROC 그림을 저장·표시합니다.

**결과물** (`models/<RUN_ID>/`)

- `results/AUC_ROC<C>.png`: ROC 곡선 + 최적점 2개
- `{net,netx}_fc{1,2}_params_by_index[_normalized].png` (8개): fc1·fc2 가중치·편향 막대그래프
- `loss_{net,netx}[_val].png`: loss 곡선 (없을 때만 생성)
- 코드의 `AUC_ANALYSIS = False`로 바꾸면 ROC 대신 차량별 `results/<id>_testing_diagnostics_case<C>.png`를 저장합니다. T²·SPE·CI 3단 시계열이고, CI에 임계선 10.4 / 87.5가 표시됩니다.

### 7.4 시뮬레이션 (체크포인트 조합 평가) — `Test_multi_models.py`

학습 중 저장된 U/X 체크포인트의 모든(또는 지정한) 조합에 대해, **조합마다 학습 데이터로 PCA를 다시 계산**하고 테스트 차량을 진단해 AUC를 비교합니다. 최적 epoch 조합을 고르는 메인 평가 스크립트입니다.

```bash
# artifact 폴더의 모든 체크포인트 조합 (N_U × N_X개)
python Test_multi_models.py --models-idx <RUN_ID>

# 조합 수 줄이기: 마지막 X로 최적 U 탐색 → 최적 U로 모든 X 평가 (N_U + N_X개)
python Test_multi_models.py --models-idx <RUN_ID> --fast-search

# 특정 체크포인트만 평가하고, 중단된 실행 이어서 하기
python Test_multi_models.py --models-idx <RUN_ID> --u-model-idx 0,5,10,-1 --x-model-idx -1 --resume
```

| 옵션 | 기본값 | 설명 |
|---|---|---|
| `--models-idx` | `260206_084839` | 평가할 `RUN_ID` |
| `--models-dir` / `--source-data-dir` | `./models` / `./data` | 결과 위치 / 원본 데이터 및 `.pt` 캐시 위치 |
| `--u-model-idx` / `--x-model-idx` | `auto` | `auto`: artifact 폴더 스캔, `0,5,10`: 직접 지정, `-1`: 최종 모델 |
| `--fast-search` | off | 2단계 탐색 (위 예시 참고) |
| `--use-validate` | off | validate 차량도 정상 테스트 차량에 포함 |
| `--use-reverse-pos-neg` | off | 정상을 Positive로 두고 `max CI < θ`를 양성으로 판정 |
| `--cache-strategy` | `disk` | 학습 특징 캐시 위치 (`disk`: `temp_cache/`, `memory`: RAM) |
| `--keep-cache` | off | 종료 후 `temp_cache/` 유지. Test_.py·Test_tsne.py와 다음 실행에서 재사용 |
| `--resume` | off | 기존 CSV에 있는 조합은 건너뜀. 미지정 시 기존 CSV를 지우고 시작 |
| `--inference-batch-size` | 65536 | 학습 데이터 추론 배치 크기 |
| `--use-float32` | off | 오차 행렬을 float32로 변환해 메모리 절약 |
| `--x-start`, `--x-tick-step`, `--sigma-levels` | | 현재 사용하지 않음 |

**처리 흐름**

1. 체크포인트를 모두 로드하고, 학습·테스트 데이터를 전처리합니다. 전처리 결과는 `--source-data-dir`에 `.pt` 캐시로 저장되고 다음 실행에서 재사용됩니다.
2. U·X 체크포인트별로 학습 데이터 특징을 계산해 캐시합니다 (`temp_cache/`).
3. 조합마다 PCA → 테스트 차량 CI → ROC/AUC를 계산하고, CSV에 한 줄씩 추가하며 ROC 그림을 저장합니다.
4. 이번 실행에서 2개 이상의 조합을 계산했으면 전체 ROC 비교 그림과 AUC 히트맵을 저장합니다.

**결과물**

- `AUC_summary_case<C>.csv`: 조합별 AUC·최적 임계값 (열 설명은 [8장](#8-결과물-정리))
- `results/AUC_ROC_u<U>_x<X>_case<C>.png`: 조합별 ROC 곡선
- `results/AUC_ROC_ALL_COMBINATIONS_case<C>.png`: 모든 조합의 ROC 곡선 비교
- `AUC_heatmap_case<C>.png`: 가로 U 체크포인트 × 세로 X 체크포인트 AUC 히트맵
- `temp_cache/feat_{u,x}_<idx>.pkl`: 학습 특징 캐시 (`--keep-cache`일 때만 남음)
- `<source-data-dir>/train_data_list_<BT>_<hash>.pt`, `test_data_cache_<BT>_<hash>.pt`: 전처리 데이터 캐시 (자동 삭제되지 않음)
- 콘솔: 조합별 AUC / Best threshold / TPR / FPR 요약 표

### 7.5 시각화

#### `Plot_AUC_heatmap.py`

`AUC_summary*.csv`로 히트맵만 다시 그립니다. `--resume`으로 일부 조합만 새로 계산해 히트맵이 갱신되지 않았을 때 등에 사용합니다.

```bash
python Plot_AUC_heatmap.py --models-idx <RUN_ID>
python Plot_AUC_heatmap.py --csv-path ./models/<RUN_ID>/AUC_summary_case1.csv
```

| 옵션 | 기본값 | 설명 |
|---|---|---|
| `--models-idx` | `260209_105821` | `RUN_ID` |
| `--models-dir` | `./models` | |
| `--search-pattern` | `AUC_summary*.csv` | 폴더에서 찾을 CSV 패턴 (처음 찾은 파일 사용) |
| `--csv-path` | 없음 | CSV 직접 지정 (지정 시 위 옵션 무시) |

**결과물**: CSV와 같은 폴더의 `<CSV이름>_heatmap.png` (예: `AUC_summary_case1_heatmap.png`)

#### `Test_tsne.py`

validate 앞쪽 `--cnt-max`대(기준 정상, label 0)와 대상 차량 1대(label 1)에서 차량당 `--per-vehicle`개 시점을 뽑습니다. CI 임계값을 넘은 시점은 임계값별로 최대 100개까지 추가합니다. 실제값(셀 전압 + 셀 SOC)과 재구성값을 하나의 t-SNE로 함께 임베딩해 비교합니다.

```bash
python Test_tsne.py --models-idx <RUN_ID> --u-model-idx -1 --x-model-idx -1 --fault-idx 338 --save   # 차량 338만
python Test_tsne.py --models-idx <RUN_ID> --u-model-idx -1 --x-model-idx -1 --save                   # test + fault 전체
```

| 옵션 | 기본값 | 설명 |
|---|---|---|
| `--models-idx` | `260206_084839` | `RUN_ID` |
| `--battery-type` | `QAS` | sim_config에서 읽지 않으므로 DTI 모델이면 직접 지정 |
| `--u-model-idx` / `--x-model-idx` | **10 / 10** | -1이 아니면 temp_cache 필요 (9장) |
| `--fault-idx` | -1 | -1: test(정상) + fault 전체를 차례로, N: 차량 N 하나 |
| `--thr THR0 THR1` | 10.4 87.5 | CI 알람 임계값 2개 |
| `--cnt-max` | 6 | 기준 정상 차량 수 |
| `--per-vehicle` | 500 | 차량당 샘플 수 |
| `--save` | off | 파일로 저장. 미지정 시 차량마다 창으로 표시 |
| `--no-alarm`, `--no-alarm-{nlo,nhi,flo,fhi}` | 모두 표시 | 알람 강조점 끄기 (n/f = 기준 정상/대상 차량, lo/hi = THR0/THR1 초과) |
| `--x-start`, `--x-tick-step` | 20, 3000 | CI 패널 x축 |
| `--normalize-dx` | off | 전처리 옵션 (학습 설정과 맞출 것) |

**결과물**: `results_tsne/tSNE_<id>.png` — [t-SNE (실제값) | t-SNE (재구성값) | 대상 차량 CI + 임계선] 1×3 그림

---

## 8. 결과물 정리

```
models/<RUN_ID>/                                  RUN_ID = 26MMDD_HHMMSS (KST, 연도 '26'은 코드에 고정)
├── sim_config.txt                                Train_             학습 설정 전체 (Test 계열이 읽음)
├── artifact/
│   ├── net_e<N>.pth, net.pth                     Train_             U-model state_dict (주기별 / 최종)
│   └── netx_e<N>.pth, netx.pth                   Train_             X-model state_dict (주기별 / 최종)
├── loss_net.csv, loss_netx.csv                   Train_             epoch별 평균 학습 loss
├── loss_net_val.csv, loss_netx_val.csv           Train_             save-per 주기마다의 검증 loss
├── pca_arrays.npz                                Train_             PCA 결과·관리한계 (최종 모델 기준)
├── pca_manifest.json                             Train_             pca_* 타입·shape·경로
├── pca_data_nor.csv                              Train_             표준화된 학습 특징 (N×6, 대용량)
├── loss_{net,netx}[_val].png                     Test 계열          loss 곡선 (없을 때만 생성)
├── {net,netx}_fc{1,2}_params_by_index[_normalized].png   Test_      fc1·fc2 파라미터 그림
├── AUC_summary_case<C>.csv                       Test_multi_models  조합별 AUC 표
├── AUC_heatmap_case<C>.png                       Test_multi_models  AUC 히트맵
├── AUC_summary_case<C>_heatmap.png               Plot_AUC_heatmap   AUC 히트맵 (CSV에서 재생성)
├── temp_cache/feat_{u,x}_<idx>.pkl               Test_multi_models  학습 특징 캐시 (--keep-cache)
├── results/
│   ├── AUC_ROC<C>.png                            Test_              단일 조합 ROC
│   ├── <id>_testing_diagnostics_case<C>.png      Test_              T²·SPE·CI 시계열 (AUC_ANALYSIS=False)
│   ├── AUC_ROC_u<U>_x<X>_case<C>.png             Test_multi_models  조합별 ROC
│   └── AUC_ROC_ALL_COMBINATIONS_case<C>.png      Test_multi_models  전체 ROC 비교
└── results_tsne/tSNE_<id>.png                    Test_tsne (--save) t-SNE + CI
```

- `<C>` = learning_case, `<idx>` = -1(최종 모델) 또는 epoch 번호입니다.
- 용량: `.pth`는 개당 약 100 KB입니다. `pca_data_nor.csv`는 학습 샘플 수에 비례하며, 약 636만 샘플 기준 약 815 MB입니다. 평가 스크립트는 `pca_data_nor.csv`를 읽지 않습니다.
- 기존 실행 폴더에 있는 `results_ci/`, `results_tsne_before/`, `results_tsne_u*_x*/` 등은 현재 주석 처리된 코드나 수동 정리로 생긴 폴더입니다.

### `AUC_summary_case<C>.csv` 열

| 열 | 의미 |
|---|---|
| `u_model_idx`, `x_model_idx` | 체크포인트 번호 (-1 = `net.pth` / `netx.pth`) |
| `AUC` | ROC 곡선 아래 면적 |
| `best_threshold`, `best_TPR`, `best_FPR` | Youden J (TPR − FPR)가 최대인 지점 |
| `closest_01_threshold`, `closest_01_TPR`, `closest_01_FPR`, `closest_01_distance` | (FPR, TPR) = (0, 1)에 가장 가까운 지점과 그 거리 |

### `pca_arrays.npz` 키

| 키 | 의미 |
|---|---|
| `p_k`, `v_I` | 선택된 주성분 (6×k), 고유값 역수 대각행렬 (k×k) |
| `data_mean`, `data_std` | 특징 표준화용 평균·표준편차 (6) |
| `T_95_limit`, `T_99_limit` | T² 관리한계 (95 %, 99 %) |
| `SPE_95_limit`, `SPE_99_limit` | SPE 관리한계 (95 %, 99 %) |
| `v`, `v_ratio`, `k` | 고유값, 누적 기여율, 누적 기여율이 95 %를 넘는 인덱스 |
| `P`, `P_t`, `X` | SVD 행렬, 표준화 특징의 공분산 행렬 |

---

## 9. 주의사항 및 알려진 제약

### 실행 시 주의사항

- **저장소 루트에서 실행**해야 합니다 (상대 경로 사용).
- **`sim_config.txt`를 지우거나 고치지 마세요.** Test 계열 스크립트가 학습 설정을 복원할 때 이 파일을 읽습니다. 학습 시 준 `--dataset-name`도 여기에 기록되어, 테스트가 같은 분할을 자동으로 사용합니다.
- **분할 이름으로 `default`는 피하세요.** `sim_config.txt`의 `dataset_name`이 `default`이면 Test 계열은 루트 기본 분할을 사용합니다. `data_filter.py`의 기본 이름이 `default`입니다.
- **디버거(VS Code 등)로 실행하면 동작이 바뀝니다.**

  | 스크립트 | 디버거로 실행할 때 |
  |---|---|
  | `Train_.py` | 학습·검증 차량 1대씩만 사용, `use_dx = False` 강제, `sim_config.txt` 미생성 |
  | `Test_.py` | u = `net_e2`, x = `netx_e1`로 고정, 모든 반복에서 차량 338만 평가 |
  | `data_filter.py` | `train_ratio = 0.02`, `val_ratio = 0.5`, `dataset_name = "2"`로 덮어씀 |

- **Test 계열은 CPU 실행을 권장합니다.** `Test_.py`, `Test_multi_models.py`, `Test_tsne.py`는 모델만 `cuda`로 옮기고 테스트 입력은 CPU에 둡니다. GPU가 있는 환경에서는 `CUDA_VISIBLE_DEVICES=""`를 지정해 실행하세요.
- **디스크 용량**: `Test_multi_models.py`가 `--source-data-dir`에 남기는 `.pt` 캐시는 학습·테스트 차량 데이터 전체를 담으므로 QAS 기본 분할 기준 수십 GB가 될 수 있습니다. 분할이나 전처리 설정이 바뀌면 해시가 달라져 새 파일이 생기므로 주기적으로 정리하세요. `temp_cache/`도 체크포인트마다 파일이 생깁니다.
- **체크포인트 저장 시점**: `epoch % save_per == 0`일 때만 `*_e<epoch>.pth`가 저장됩니다. 마지막 epoch 가중치는 `net.pth` / `netx.pth`(-1)에만 있습니다.
- **조합 수**: 300 epoch를 10 epoch마다 저장하면 U·X 각 31개 → 961개 조합입니다. `--fast-search`나 `--u-model-idx`/`--x-model-idx` 지정을 고려하세요.

### 알려진 제약

- `Test_.py`에서 `--u-model-idx`를 -1이 아닌 값으로 주고 `temp_cache`가 없으면, `train_pca_only()`에 Test 인자(`args`)가 넘어가 AttributeError가 납니다. 먼저 `Test_multi_models.py --keep-cache`로 캐시를 만들어 두세요.
- `Test_tsne.py`는 `temp_cache`가 없으면 체크포인트와 관계없이 최종 모델 기준 `pca_arrays.npz`를 사용합니다.
- `Train_pca_only.py`는 단독 실행되지 않습니다. `__main__`에서 `device` 인자 없이 호출하고, 필요한 인자(`--learning-case` 등)도 정의되어 있지 않습니다. Test_.py에서 함수로만 사용합니다.
- `--normalize-dx`를 쓰면 Test 계열은 `normalize_val`을 넘기지 않아(기본 1.0) 학습(기본 4)과 Δ 스케일이 달라집니다. learning_case 2·3에만 해당합니다.
- LSTM 관련 옵션은 레거시입니다. 학습한 LSTM은 저장되지 않으며, `--lstm-training`은 `--lstm-load`와 함께 써야 동작합니다.
- `check_data_stats.py`는 데이터 경로가 다른 PC 경로로 하드코딩되어 있어, 경로를 수정한 뒤 실행해야 합니다.
- `models/` 루트의 `net.pth`, `netx.pth`, `pca_*`는 구버전 산출물입니다. 현재 스크립트가 요구하는 `<RUN_ID>/artifact/`, `sim_config.txt` 구조가 아니어서 그대로는 사용할 수 없습니다.
- `data/GIS`는 `--battery-type` 선택지(QAS, DTI)에 없어 사용할 수 없습니다.
