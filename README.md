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
python Test_.py --models-idx <RUN_ID>
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
시점별 진단 특징 6개                       diagnosis_features()
        │
        ▼
학습 데이터로 PCA → T², SPE 관리한계(95%)    ControlChart.fit()
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

### 진단 특징 6개 ([mcdl/diagnosis.py](mcdl/diagnosis.py) `diagnosis_features`)

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
├── Class_.py               [모듈] CombinedAE, LSTM, Dataset 클래스
├── Function_.py            [모듈] 로딩·전처리·진단 특징·PCA·T²/SPE·ROC·플롯·설정 I/O
├── monitering.py           [참고] 외부 T²/Q 관리도 클래스 (import만 되고 사용하지 않음)
├── check_data_stats.py     [분석] DTI 초기 수렴 구간 탐색용 일회성 스크립트
├── mcdl/                   [모듈] deep module 패키지 (run.py: Run 폴더 읽기·쓰기, 모델 생성·로드 / reconstruction.py: 재구성·재구성 오차·차량 진단 특징 / diagnosis.py: 진단 특징·관리도 / chart_source.py: 모델 쌍의 관리도 공급원 / vehicles.py: 분할 차량 ID·차량 텐서 로딩 / roc.py: 차량 단위 ROC 평가)
├── tests/                  pytest 테스트, realdata/baseline.py: 실데이터 기준선 비교 도구
├── pytest.ini              pytest 설정 (저장소 루트를 import 경로에 추가)
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
| `MODEL_DIR` | `--models-dir` | `./models` | Train_, Test_, Test_multi_models, Test_tsne, Plot_AUC_heatmap |
| `SOURCE_DIR` | `--source-data-dir` | `./data` | Train_, Test_, Test_multi_models, Test_tsne |

`data_filter.py`는 `./data` 경로를 코드에 고정해서 사용합니다.

### 테스트

저장소 루트에서 pytest로 실행합니다. 원본 데이터 없이 CPU에서 몇 초 안에 끝납니다.

```bash
pip install pytest
python -m pytest
```

- `tests/test_run.py`: `mcdl/run.py` 테스트. `tmp_path`에 만든 합성 Run과 실제 sim_config 사본(`tests/fixtures/`)을 사용합니다.
- `tests/test_scripts.py`: 진입 스크립트 smoke 테스트. `tmp_path`의 가짜 원본 데이터로 `Train_.main` → `Test_.main` → `Test_tsne.main` → `Test_multi_models.main` → `Plot_AUC_heatmap.main`을 이어서 실행합니다(약 20초). 스크립트를 import만 하면 아무것도 실행되지 않는지도 확인합니다.
- `tests/test_chart_source.py`: 관리도 공급원 테스트. `tmp_path`의 Run과 작은 가짜 원본 데이터(QAS 열 구성 pkl)로 캐시·npz·재계산 분기를 모두 실행합니다.
- `tests/test_reconstruction.py`: 재구성·재구성 오차·진단 특징 테스트 (작은 CombinedAE와 합성 차량 텐서). 테스트 차량 진단 특징이 기존 스크립트 방식과 bit 단위로 같은지도 확인합니다.
- `tests/test_vehicles.py`: 차량 데이터 테스트. `tmp_path`의 가짜 원본 데이터와 분할 `.npy`로 분할 범위, 전처리, `.pt` 캐시 재사용과 해시 규칙을 확인합니다 (가짜 데이터는 `tests/fake_vehicles.py`).
- `tests/test_diagnosis.py`: 진단 특징 → 관리도 → CI → ROC 테스트. 고정 seed 합성 오차(`tests/synthetic_errors.py`)에 대한 기대값(`tests/fixtures/diagnosis_golden.npz`)과 bit 단위로 비교합니다. 기대값은 Windows·numpy 2.3.5 환경에서 만들었으므로 다른 플랫폼에서는 ULP 수준 차이로 실패할 수 있습니다.
- `realdata` 마커가 붙은 테스트는 `models/260224_090359`가 있을 때만 돕니다.

**실데이터 기준선 비교**: 리팩토링 전후 수치가 bit 단위로 같은지 확인하는 도구입니다. pytest가 수집하지 않으며 직접 실행합니다.

```bash
python tests/realdata/baseline.py capture <test|test_u0|tsne|multi|train|train_full> <tag>    # 기준선 저장
python tests/realdata/baseline.py compare <target> <tag_a> <tag_b>          # 두 기준선 비교
```

- test·test_u0·tsne·multi는 `models/_verify/260224_090359`(원본 Run에서 `pca_data_nor.csv`와 `results/`를 뺀 사본)에서 실행합니다. test_u0은 관리도를 다시 계산하는 경로(U 0, X -1)입니다.
- train·train_full은 `models/_verify/train/`에 새 Run을 만들고 Run 폴더의 모든 파일을 비교합니다.
  - train: 차량 1대, 1 epoch
  - train_full: 차량 2대, 3 epoch, `save-per 2`, shuffle, 검증 포함(약 3분). 학습 루프의 RNG 순서, 검증, 체크포인트 간격까지 봅니다.
- 결과는 `models/_verify/baselines/<tag>/`에 저장됩니다.
- multi는 `--source-data-dir`에 `.pt` 캐시(QAS 기본 분할 기준 약 41 GB)를 만들거나 재사용합니다.

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
| `Train_.py` | Function_, Class_, mcdl.run, mcdl.diagnosis, mcdl.reconstruction, mcdl.vehicles |
| `Test_.py` | Function_, Class_, monitering, mcdl.run, mcdl.reconstruction, mcdl.chart_source, mcdl.roc, mcdl.vehicles |
| `Test_multi_models.py` | Function_, Class_, monitering, mcdl.run, mcdl.diagnosis, mcdl.reconstruction, mcdl.chart_source, mcdl.roc, mcdl.vehicles |
| `Test_tsne.py` | Function_, Class_, monitering, mcdl.run, mcdl.reconstruction, mcdl.chart_source, mcdl.vehicles |
| `data_filter.py` | Function_, Class_ |
| `Plot_AUC_heatmap.py` | Function_ (`draw_auc_heatmap`) |
| `Class_.py` | Function_ |
| `mcdl/run.py` | Function_, Class_ |
| `mcdl/chart_source.py` | mcdl.diagnosis, mcdl.reconstruction, mcdl.vehicles |
| `mcdl/vehicles.py` | Function_ (원본 로딩·전처리) (+ torch) |
| `mcdl/reconstruction.py` | mcdl.diagnosis (+ torch) |
| `mcdl/diagnosis.py` | (numpy, pandas, scipy만 사용) |
| `mcdl/roc.py` | (numpy, scikit-learn만 사용) |

### mcdl/run.py

Run(`models/<RUN_ID>/`)을 다루는 module입니다. 평가 스크립트(Test_, Test_multi_models, Test_tsne)는 이 module로 Run을 열고, Train_은 이 module로 sim_config.txt를 쓰고 체크포인트 파일 이름을 정하고 모델 쌍을 만듭니다. 용어는 [GLOSSARY.md](GLOSSARY.md)를 따릅니다.

| 이름 | 설명 |
|---|---|
| `Run(path)` | sim_config.txt를 읽어 `.config`를 만듭니다. `battery_type`·`learning_case`가 없으면 ValueError |
| `Run.config` (`RunConfig`) | 평가에 필요한 설정. 키가 없는 예전 Run은 기본값(scale·shift 1.8/2.5/1.0/0, use_dx True 등)으로 채우고, learning_case에서 preprocessing·skip_charge_ready를, dataset_name에서 분할 폴더(`split_dir`)를 계산 |
| `Run.checkpoint_numbers("U"/"X")` | artifact 폴더의 체크포인트 번호 (오름차순, -1은 맨 뒤) |
| `Run.feature_cache_path("U"/"X", n, abs_err=, float32=, batch_size=)` | temp_cache 학습 특징 캐시 파일 경로. 계산 조건이 이름에 들어갑니다 |
| `Run.load_model("U"/"X", n)`, `Run.load_model_pair(u, x)` | CPU·double·eval 상태의 U-model·X-model. 학습 때와 같은 double 정밀도로 가중치를 load합니다 |
| `RunConfig.from_mapping(values)` | sim_config 값 또는 학습 인자(`vars(args)`)에서 설정을 만듭니다. 기본값·파생값 계산이 모이는 곳 |
| `build_model_pair(config)`, `checkpoint_filename(model, n)` | 설정에 맞는 새 모델 쌍 (U를 먼저 생성), 체크포인트 파일 이름 규칙 |
| `write_sim_config(path, args, extra)` | 학습 인자를 `path/sim_config.txt`에 쓰고 그 텍스트를 돌려줍니다 (Train_) |
| `print_sim_config(...)` | 설정 표 출력. sim_config.txt 형식이 여기서 정해지며, Test 계열은 시작할 때 콘솔 출력에 씁니다 |

### mcdl/reconstruction.py

U-model·X-model로 차량 텐서 `[x | y | Δ | q]`를 재구성해 재구성 오차를 구하고, 차량 한 대의 진단 특징과 학습 진단 특징을 만드는 module입니다. 열 구성은 모델 크기에서 읽으며, 재구성 오차를 계산하는 곳(학습 진단 특징, 테스트 차량 평가)은 모두 `reconstruct`를 거칩니다. 학습 쪽과 테스트 쪽이 같은 `vehicle_features`를 씁니다.

| 이름 | 설명 |
|---|---|
| `reconstruct(model, tensor, batch_size=None)` | 한 차량의 (재구성 대상 y, 재구성 결과 ŷ). 모델이 있는 device에서 batch 단위로 추론하고, `None`이면 차량 전체를 한 번에 추론합니다. batch 크기는 결과에 ULP 수준으로 영향을 주므로 호출자가 정합니다 (Train_ 4096, 평가 쪽 학습 특징 65536, 테스트 차량은 차량 전체) |
| `reconstruction_errors(model, tensor, batch_size=, abs_err=, float32=False)` | 한 차량의 재구성 오차 ŷ − y (시점 × 셀). `abs_err`이면 절댓값입니다 |
| `vehicle_features(net, netx, u, x, batch_size=None, abs_err=)` | 차량 한 대의 진단 특징. 테스트 스크립트는 차량 전체를 한 번에 추론하고, abs 여부는 학습 진단 특징과 같은 설정(`no_abs_err`)을 따릅니다 |
| `training_features(net, netx, vehicles, batch_size=, abs_err=)` | (U 텐서, X 텐서) 쌍마다 `vehicle_features`를 구해 이어 붙인 학습 진단 특징 (Train_, 관리도 재계산) |
| `training_signal_features(model, "U"/"X", tensors, …)` | 한 신호의 학습 진단 특징 3개 (Test_multi_models의 temp_cache 형식). 두 신호를 `combine_signal_features`로 묶으면 `training_features`와 같습니다 |

### mcdl/diagnosis.py

차량별 재구성 오차에서 진단 특징을 뽑고, 관리도(정상 학습 진단 특징으로 맞춘 PCA 모델과 T²·SPE 관리한계)로 CI를 계산하는 module입니다. 용어는 [GLOSSARY.md](GLOSSARY.md)를 따릅니다.

| 이름 | 설명 |
|---|---|
| `diagnosis_features(error_u, error_x)` | 차량 한 대의 진단 특징 (시점 × 6, 2장 표의 열 순서). 차량마다 따로 계산해야 합니다 |
| `signal_features("U"/"X", error)`, `combine_signal_features(u, x)` | 신호별 특징 3개와 그 조합. Test_multi_models의 temp_cache 파일(`Run.feature_cache_path`)이 이 형식입니다 |
| `ControlChart.fit(features)` | 정상 학습 진단 특징(차량별로 계산해 이어 붙인 것)으로 관리도를 맞춥니다 |
| `ControlChart.load(run_dir)` | Run 폴더의 `pca_arrays.npz`를 읽습니다 (`pca_manifest.json` 필요) |
| `chart.save(run_dir, features)` | `pca_arrays.npz`, `pca_manifest.json`, `pca_data_nor.csv`를 씁니다 (Train_) |
| `chart.t2_limit`, `chart.spe_limit` | 관리한계 T²₉₅, SPE₉₅ |
| `chart.score(features)`, `chart.max_ci(features)` | 시점별 (T², SPE, CI)와 최대 CI. CI = SPE/SPE₉₅ + T²/T²₉₅ |

### mcdl/roc.py

| 이름 | 설명 |
|---|---|
| `evaluate(max_ci, is_fault, reverse=False)` | 차량별 최대 CI와 고장 여부로 ROC를 평가합니다. 임계값 θ = 0 ~ 999.9 (0.1 간격). 기본은 고장이 양성이고 최대 CI > θ이면 양성, `reverse`는 정상이 양성이고 최대 CI < θ이면 양성입니다 |
| `RocResult` | `fpr`, `tpr`, `auc`, `best`(Youden J 최대점), `closest_to_01`((0,1) 최근접점). `best`·`closest_to_01`은 threshold·fpr·tpr·index와 youden_j/distance를 담은 dict입니다 |

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
| 진단 | `CustomSigmoidFunc` (진단 특징·관리도는 `mcdl/diagnosis.py`) |
| 시각화 | `plot_roc_curve`, `draw_auc_heatmap`, `plot_diagnostics_triplet`, `plot_tsne_before_after_ci`, `plot_loss_curve_all`, `plot_net_layer_params_by_index`, `plot_testX_timeseries` |
| 저장·설정 | `save_net_state`, `make_model_path_based_timestamp` (sim_config 읽기·쓰기는 `mcdl/run.py`, 관리도 저장·로드는 `mcdl/diagnosis.py`) |
| 현재 미사용 | `solvers`(Kalman 기반 추정), `calculate_volt_modepi`, `ClassifyFeature`, `cont`, `diagnosis_thresholds`, `plot_contrib_percent_stacked`, LSTM 저장/로드 등 |

### mcdl/chart_source.py

모델 쌍의 관리도를 어디서 가져올지 정하는 module입니다. Test_.py와 Test_tsne.py는 `chart_for_pair` 한 줄로 관리도를 얻고, Test_multi_models.py는 같은 module로 temp_cache를 쓰고 읽습니다.

| 이름 | 설명 |
|---|---|
| `chart_for_pair(run, u, x, source_data_dir=)` | ① 다시 계산할 때와 같은 조건(설정의 abs 여부, float32 끔, batch 65536)으로 만든 temp_cache 특징이 있으면 그것으로 관리도를 맞추고, ② U·X가 모두 -1이면 `pca_arrays.npz`를 읽고, ③ 그 밖에는 Run의 학습 차량 원본을 읽어 학습 진단 특징을 다시 계산합니다 (수 분) |
| `save_cached_features(run, "U"/"X", n, features, conditions=)`, `load_cached_features(…)` | temp_cache 파일(신호별 특징 3개의 pickle) 쓰기·읽기. 경로는 `Run.feature_cache_path` |

### mcdl/vehicles.py

분할의 차량 ID와, 원본(`vin_2`·`vin_3`)을 Run 설정대로 전처리한 차량 텐서를 돌려주는 module입니다. 네 스크립트와 관리도 재계산이 모두 이 module로 차량을 읽습니다. 용어는 [GLOSSARY.md](GLOSSARY.md)를 따릅니다.

| 이름 | 설명 |
|---|---|
| `split_ids(split_dir, battery_type, split, start=0, end=-1)` | 분할(train/validate/test/fault/normal)의 차량 ID (int 목록). `start`~`end`(끝 포함, -1이면 끝까지)가 학습 차량 범위입니다 |
| `load_vehicle(source_data_dir, config, vehicle_id)` | 한 차량의 (U 텐서, X 텐서). `RunConfig`의 battery_type·전처리 설정·normalize_dx·normalize_val로 `preprocess_loaded_tensor`를 적용합니다 |
| `load_training_vehicles(source_data_dir, config, ids)` | 학습 차량들의 `[(U, X)]` (double). `source_data_dir/train_data_list_<BT>_<hash>.pt`가 있으면 그것을 읽고, 없으면 만들어 저장합니다 (Test_multi_models) |
| `load_test_vehicles(source_data_dir, config, ids)` | 테스트 차량들의 `{id: (U, X)}`. 캐시 파일은 `test_data_cache_<BT>_<hash>.pt`입니다 (Test_multi_models) |

캐시 해시는 차량 ID 목록(순서 포함)과 battery_type·preprocessing·skip_charge_ready·normalize_dx의 md5이며, normalize_dx일 때만 normalize_val이 들어갑니다. 전처리 코드를 바꾸면 해시가 그대로이므로 캐시 파일을 직접 지워야 합니다.

### monitering.py

외부 코드(Iurii Katser, control-charts)의 Hotelling T²/Q 통계 클래스입니다. 테스트 스크립트들이 `from monitering import *`로 불러오지만, 실제 진단에는 `mcdl/diagnosis.py`의 관리도(`ControlChart`)를 사용합니다.

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
| `--debug` | off | 빠른 확인용: `train_ratio = 0.02`, `val_ratio = 0.5`, `dataset_name = "2"`로 덮어씀 |

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
5. X-model도 같은 방식으로 학습해 `netx_e<epoch>.pth`, `netx.pth`를 저장합니다. U·X 모두 같은 함수(`train_autoencoder`)로 학습합니다.
6. 최종 모델로 학습 차량의 재구성 오차 → 진단 특징 → PCA를 계산해 `pca_*`를 저장합니다.

콘솔에는 저장 주기마다 `Epoch | Train Loss | Val Loss`와 총 소요 시간이 출력됩니다.

**재현성**: 같은 인자와 `--seed`면 결과가 bit 단위로 같습니다. 다만 DataLoader는 순회할 때마다 전역 난수를 쓰므로(shuffle을 꺼도), shuffle을 켜면 X-model의 셔플 순서가 U-model 학습의 epoch 수와 `--ae-u-save-per`(검증 횟수)에 따라 달라집니다.

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
| `--no-abs-err` | off | 진단 특징 계산 시 부호 있는 오차 사용. 평가 스크립트도 sim_config의 이 값을 따라 테스트 차량 오차를 계산 |
| `--normalize-dx`, `--normalize-val` | off, 4 | learning_case 2·3에서 Δ를 normalize_val로 나눔 |
| `--no-validation` | off | 검증 데이터 로드·평가 생략 |
| `--no-save` | off | `.pth` 체크포인트 저장 생략 (loss·PCA·sim_config는 저장) |
| `--seed` | 42 | 난수 시드 |
| `--debug` | off | 빠른 확인용: 학습·검증 차량 1대씩, `use_dx = False`, `sim_config.txt` 미생성 |
| `--lstm-load`, `--lstm-training`, `--lstm-epochs`, `--lstm-lr`, `--lstm-batchsize` | off / 300 / 1e-4 / 100 | 레거시. `vin_1.pkl`로 LSTM을 학습하지만 결과는 저장하지 않음 |
| `--vin1-start`, `--vin2-start`, `--vin3-start` | 0 | 현재 사용하지 않음 (sim_config에 기록만 됨) |

**결과물**: `sim_config.txt`, `artifact/*.pth`, `loss_*.csv`, `pca_*` (상세는 [8장](#8-결과물-정리))

### 7.3 테스트 (단일 조합) — `Test_.py`

학습 폴더의 `sim_config.txt`에서 설정(battery_type, learning_case, 출력층 파라미터, dataset_name 등)을 읽어 학습과 같은 조건으로 평가합니다.

```bash
python Test_.py --models-idx <RUN_ID>                                       # 최종 모델 (net.pth, netx.pth)
python Test_.py --models-idx <RUN_ID> --u-model-idx 2 --x-model-idx 0       # 특정 체크포인트 (캐시가 없으면 관리도를 다시 계산, 수 분)
```

| 옵션 | 기본값 | 설명 |
|---|---|---|
| `--models-idx` (`--models-folder`) | `260224_090359` | 평가할 `RUN_ID`. 예전 이름 `--models-folder`도 그대로 동작 |
| `--models-dir` | `./models` | 결과 위치 (`MODEL_DIR` 환경 변수) |
| `--source-data-dir` | `./data` | 원본 데이터 위치 |
| `--u-model-idx` / `--x-model-idx` | -1 / -1 | -1: `net.pth`/`netx.pth`, N: `net_eN.pth`/`netx_eN.pth` |
| `--x-start`, `--x-tick-step` | 20, 3000 | 진단 시계열 그림의 x축 (진단 그림 모드에서만 사용) |
| `--sigma-levels` | `3,4.5,6` | 현재 사용하지 않음 |
| `--debug` | off | 빠른 확인용: 모든 반복에서 차량 338만 평가 (모델 쌍은 위 인자 그대로) |

**PCA 관리한계를 정하는 순서**

1. 같은 계산 조건(설정의 abs 여부, float32 끔, batch 65536)으로 만든 `temp_cache/feat_u_<u>_….pkl`, `feat_x_<x>_….pkl`이 있으면 캐시된 학습 특징으로 PCA를 계산합니다.
2. 없고 u = x = -1(최종 모델 쌍)이면 학습 때 저장한 `pca_arrays.npz`를 사용합니다.
3. 그 외에는 학습 차량의 진단 특징을 차량별로 다시 계산해 관리도를 만듭니다. 학습 차량 원본을 읽으므로 수 분이 걸립니다. Test_multi_models와 같은 계산(`mcdl/reconstruction.py`)이라 같은 모델 쌍이면 같은 결과가 나옵니다.

이 순서는 `mcdl/chart_source.py`의 `chart_for_pair`에 있으며, Test_tsne.py도 같은 순서를 씁니다.

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
- `temp_cache/feat_{u,x}_<idx>_abs=<0|1>_float32=<0|1>_batch=<N>.pkl`: 학습 특징 캐시 (`--keep-cache`일 때만 남음). 계산 조건이 이름에 들어가므로 조건이 다른 캐시는 섞여 쓰이지 않습니다
- `<source-data-dir>/train_data_list_<BT>_<hash>.pt`, `test_data_cache_<BT>_<hash>.pt`: 전처리 데이터 캐시 (자동 삭제되지 않음, 해시 규칙은 [6장 mcdl/vehicles.py](#mcdlvehiclespy))
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
| `--models-idx` | `260206_084839` | `RUN_ID`. 배터리 타입과 전처리 설정(`normalize_dx` 등)은 이 Run의 sim_config에서 읽습니다 |
| `--u-model-idx` / `--x-model-idx` | -1 / -1 | 관리도 출처는 Test_.py와 같은 순서 (7.3) |
| `--fault-idx` | -1 | -1: test(정상) + fault 전체를 차례로, N: 차량 N 하나 |
| `--thr THR0 THR1` | 10.4 87.5 | CI 알람 임계값 2개 |
| `--cnt-max` | 6 | 기준 정상 차량 수 |
| `--per-vehicle` | 500 | 차량당 샘플 수 |
| `--save` | off | 파일로 저장. 미지정 시 차량마다 창으로 표시 |
| `--no-alarm`, `--no-alarm-{nlo,nhi,flo,fhi}` | 모두 표시 | 알람 강조점 끄기 (n/f = 기준 정상/대상 차량, lo/hi = THR0/THR1 초과) |
| `--x-start`, `--x-tick-step` | 20, 3000 | CI 패널 x축 |

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
├── temp_cache/feat_{u,x}_<idx>_<조건>.pkl        Test_multi_models  학습 특징 캐시 (--keep-cache)
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
- **`--debug`를 주면 빠른 확인용으로 동작이 바뀝니다.** 디버거(VS Code 등)로 실행하는 것만으로는 동작이 바뀌지 않습니다. 디버거에서 이 동작을 쓰려면 launch.json의 `args`에 `--debug`를 넣으세요.

  | 스크립트 | `--debug`일 때 |
  |---|---|
  | `Train_.py` | 학습·검증 차량 1대씩만 사용, `use_dx = False` 강제, `sim_config.txt` 미생성 |
  | `Test_.py` | 모든 반복에서 차량 338만 평가. 모델 쌍과 관리도는 `--u-model-idx`·`--x-model-idx`를 그대로 따름 |
  | `data_filter.py` | `train_ratio = 0.02`, `val_ratio = 0.5`, `dataset_name = "2"`로 덮어씀 |

- **2026-10-15 이전 평가 결과와 비교할 때 주의하세요.** 그 전에는 평가 스크립트가 저장된 float64 가중치를 float32로 잘라 load했습니다(`.scratch/known-bugs` 01). 기존 Run 폴더의 `AUC_summary_case<C>.csv`, ROC 그림 등은 잘린 가중치 기준이며, 지금 다시 평가하면 값이 달라질 수 있습니다. 학습 때 저장한 `pca_arrays.npz`는 원래부터 full precision 기준입니다.
- **Test 계열은 CPU에서 실행됩니다.** `Test_.py`, `Test_multi_models.py`, `Test_tsne.py`는 GPU가 있어도 모델과 입력을 모두 CPU에 둡니다 (`mcdl/run.py`가 CPU 모델을 돌려줌).
- **디스크 용량**: `Test_multi_models.py`가 `--source-data-dir`에 남기는 `.pt` 캐시는 학습·테스트 차량 데이터 전체를 담으므로 QAS 기본 분할 기준 수십 GB가 될 수 있습니다. 분할이나 전처리 설정이 바뀌면 해시가 달라져 새 파일이 생기므로 주기적으로 정리하세요. `temp_cache/`도 체크포인트마다 파일이 생깁니다.
- **체크포인트 저장 시점**: `epoch % save_per == 0`일 때만 `*_e<epoch>.pth`가 저장됩니다. 마지막 epoch 가중치는 `net.pth` / `netx.pth`(-1)에만 있습니다.
- **조합 수**: 300 epoch를 10 epoch마다 저장하면 U·X 각 31개 → 961개 조합입니다. `--fast-search`나 `--u-model-idx`/`--x-model-idx` 지정을 고려하세요.

### 알려진 제약

- LSTM 관련 옵션은 레거시입니다. 학습한 LSTM은 저장되지 않으며, `--lstm-training`은 `--lstm-load`와 함께 써야 동작합니다.
- `check_data_stats.py`는 데이터 경로가 다른 PC 경로로 하드코딩되어 있어, 경로를 수정한 뒤 실행해야 합니다.
- `models/` 루트의 `net.pth`, `netx.pth`, `pca_*`는 구버전 산출물입니다. 현재 스크립트가 요구하는 `<RUN_ID>/artifact/`, `sim_config.txt` 구조가 아니어서 그대로는 사용할 수 없습니다.
- `data/GIS`는 `--battery-type` 선택지(QAS, DTI)에 없어 사용할 수 없습니다.
