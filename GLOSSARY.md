# MC-AE 온라인 고장 진단

리튬이온 배터리 팩의 셀 전압·SOC를 MC-AE로 재구성하고, 재구성 오차를 PCA 관리도로 감시해 고장 차량을 찾아내는 연구 코드베이스의 언어다.

## Language

### 학습 실행

**Run**:
한 번의 학습 실행과, 그 설정·가중치·PCA 결과가 저장된 폴더 `models/<RUN_ID>/`.
_Avoid_: models folder, models_idx, models_folder, 학습 폴더

**체크포인트 번호**:
Run 안에서 특정 epoch에 저장된 가중치를 가리키는 번호. 학습이 끝난 최종 모델은 -1로 가리킨다.
_Avoid_: model idx, epoch 번호

### 모델

**U-model**:
셀 전압을 재구성하는 MC-AE.
_Avoid_: net, 전압 모델

**X-model**:
셀 SOC를 재구성하는 MC-AE.
_Avoid_: netx, SOC 모델

**모델 쌍**:
함께 평가하는 U-model 체크포인트와 X-model 체크포인트의 짝.
_Avoid_: 조합, combination

### 차량 데이터

**분할**:
Run이 쓰는 차량 ID 묶음. 정상 차량(normal = train + validate + test)과 고장 차량(fault)으로 나뉜다.
_Avoid_: split list, dataset

**학습 차량 범위**:
train 분할 중 실제 학습에 쓴 구간(`vehicle_start`~`vehicle_end`, 끝 포함, -1이면 끝까지).
_Avoid_: vehicle slice

### 진단

**재구성 오차**:
한 차량의 시점·셀별 (재구성값 − 실제값). U-model은 셀 전압, X-model은 셀 SOC에 대해 구하며, 보통 절댓값을 쓴다.
_Avoid_: ERRORU, ERRORX, error matrix

**학습 진단 특징**:
정상 학습 차량마다 구한 진단 특징을 이어 붙인 것. 관리도를 맞추는 데 쓴다.
_Avoid_: df_data_train, training data

**진단 특징**:
한 차량의 시점별 재구성 오차에서 뽑은 6개 값(zu₂, zx₂, zu₁, zx₁, ewu, ewx). 차량마다 따로 계산한다.
_Avoid_: feature vector, df_data

**관리도**:
정상 차량의 학습 진단 특징으로 맞춘 PCA 모델과 그 T²·SPE 관리한계.
_Avoid_: PCA results, loads, monitor

**관리한계**:
관리도의 T²·SPE 95% 한계.
_Avoid_: threshold, 임계값

**CI**:
시점별 T²/T²₉₅ + SPE/SPE₉₅.
_Avoid_: health index

**최대 CI**:
차량 한 대의 CI 시계열 최댓값. 차량을 판정하는 점수다.
_Avoid_: ci_max

**임계값**:
최대 CI와 비교해 고장을 판정하는 값 θ.
_Avoid_: 관리한계, limit

## 코드와 테스트를 읽을 때 필요한 말

위 용어를 처음 보는 사람을 위해, 코드 주석에 자주 나오는 말을 쉬운 말로 풀어 둔다.

### 진단 계산

| 말 | 뜻 |
|---|---|
| MC-AE | Model-Constrained AutoEncoder. 배터리 모델 식을 구조에 넣은 작은 오토인코더(입력을 받아 같은 값을 다시 만들어 내도록 학습하는 신경망). 코드에서는 `Class_.py`의 `CombinedAE` |
| 재구성 | 모델이 입력을 보고 셀 전압(또는 SOC)을 다시 계산해 내는 것. 정상 차량이면 실제값과 비슷하고, 고장이면 차이(재구성 오차)가 커진다 |
| Δ (dx) | 셀별 편차 추정값. 모델 입력의 한 부분이다 |
| PCA | 주성분 분석. 여러 특징을 서로 독립인 몇 개의 축으로 바꾸는 통계 기법. 여기서는 정상 데이터의 "평소 모양"을 잡는 데 쓴다 |
| T² (호텔링 T²) | PCA 축 안에서 평소 범위를 얼마나 벗어났는지 나타내는 값 |
| SPE | PCA 축으로 설명되지 않고 남는 오차의 크기. 평소와 다른 모양이면 커진다 |
| 95% 관리한계 | 정상 데이터의 95%가 그 아래에 들어오도록 정한 T²·SPE의 경계값 |
| ROC 곡선, AUC | 임계값을 바꿔 가며 고장 차량을 고장으로 맞힌 비율(TPR)과 정상 차량을 잘못 고장으로 판정한 비율(FPR)을 그린 곡선과, 그 아래 면적. AUC가 1에 가까울수록 정상과 고장을 잘 가른다 |
| Youden J | TPR − FPR. 이 값이 가장 큰 임계값을 "최적 임계값"으로 고른다 |
| 칼만 필터 | 모델 예측과 측정값을 섞어 보이지 않는 상태(SOC 등)를 시점마다 추정하는 방법. 측정 잡음과 모델 오차의 크기에 따라 둘의 비중을 정한다. `mcdl/kalman.py`의 `solvers`가 팩 상태와 셀별 SOC 편차를 추정한다(확장 칼만 필터: 비선형 식을 그 시점에서 직선으로 근사해 쓰는 칼만 필터) |
| OCV | 개방 회로 전압. 전류가 흐르지 않을 때의 전압으로, SOC와 온도로 정해진다. `solvers`는 SOC 17차 다항식 + 온도 3차 항으로 계산하며 그 계수 `b`가 필요하다 |
| LSTM | 시계열을 다루는 신경망의 한 종류. 여기서는 한 시점 전의 팩 측정값 7개로 다음 시점의 대표 셀 전압과 SOC를 예측한다(`Class_.py`의 `LSTM`, MC-AE와 따로 `Train_lstm.py`로 학습) |

### 파일과 데이터

| 말 | 뜻 |
|---|---|
| mcdl | 논문 제목의 Model-Constrained Deep Learning 앞 글자. 여러 실행 스크립트가 함께 쓰는 공통 코드를 모은 폴더(파이썬 패키지) |
| 패키지, module | 파이썬에서 `.py` 파일 하나를 module, module을 모은 폴더(`__init__.py`가 있는 폴더)를 패키지라고 한다 |
| 텐서 | torch(파이토치)의 다차원 배열. numpy 배열과 비슷하다 |
| float64, float32 | 실수를 저장하는 정밀도. float64가 더 정밀하다. 이 코드는 학습·평가를 float64로 한다 |
| RUN_ID | 학습을 시작한 시각으로 만든 Run 폴더 이름. `26MMDD_HHMMSS` 형식이다 |
| `sim_config.txt` | Train_이 Run 폴더에 남기는 학습 설정 기록. 평가 스크립트가 이 파일을 읽어 학습 때와 같은 조건으로 평가한다 |
| 체크포인트 파일 (`.pth`) | 학습 중 저장한 모델 가중치. `net_e<번호>.pth`(중간), `net.pth`(최종) |
| 분할 파일 (`.npy`) | 차량 ID 목록을 담은 numpy 파일. `QAS_filtered_vehicle_ids_train.npy`처럼 이름에 배터리 종류와 분할 이름이 들어간다. `datasets/<이름>/` 폴더에 있고, 이름을 주지 않으면 기본 분할 `datasets/basic/`을 쓴다 |
| learning_case | 전처리 조합 번호(1~4). 충전 대기 구간을 자를지, 0~1로 정규화할지를 정한다 |
| 캐시 | 오래 걸리는 계산 결과를 파일로 저장해 두고 다음 실행에서 다시 쓰는 것 |
| `.pt` 캐시 | 전처리를 마친 차량 텐서를 묶어 저장한 파일. `data/` 폴더에 생기며 Test_multi_models가 만든다 |
| `temp_cache/` | Run 폴더 안에 생기는, 체크포인트별 학습 진단 특징을 저장한 캐시 폴더. Test_multi_models가 만든다 |

### 테스트

| 말 | 뜻 |
|---|---|
| pytest | 파이썬 테스트 실행 도구. `tests/` 안의 `test_*.py` 파일에서 `test_`로 시작하는 함수를 찾아 실행한다 |
| fixture | pytest 용어로, 테스트를 실행하기 전에 미리 준비해 두는 데이터나 환경(가짜 데이터 폴더, 작은 모델 등). `@pytest.fixture`가 붙은 함수가 이것을 만든다 |
| `tests/fixtures/` 폴더 | 테스트가 읽는 고정 데이터 파일을 두는 폴더. 이름은 위 fixture에서 온 관례다 |
| golden 값 (기대값) | 이미 맞다고 확인한 계산 결과를 파일로 저장해 둔 "정답지". 코드를 바꾼 뒤 같은 입력으로 다시 계산해 이것과 비교하면, 결과가 조금이라도 달라졌는지 바로 알 수 있다. `tests/fixtures/diagnosis_golden.npz`가 이것이다 |
| smoke 테스트 | 프로그램을 처음부터 끝까지 한 번 돌려 크게 깨진 곳이 없는지만 보는 간단한 테스트. `tests/test_scripts.py`가 가짜 데이터로 학습부터 평가까지 실행한다 |
| 실데이터 기준선 | 코드를 고치기 전에 실제 데이터로 스크립트를 돌려 결과를 기록해 둔 것. 고친 뒤 다시 돌려 기록과 비교한다(`tests/realdata/baseline.py`) |
| bit 단위로 같다 | 소수점 끝자리까지 완전히 같은 값이라는 뜻. "거의 같다"가 아니라 컴퓨터가 저장한 값이 100% 같다 |
| ULP | 부동소수점 수에서 바로 옆 수와의 최소 간격. 계산 순서만 바뀌어도 이 정도의 아주 작은 차이가 생길 수 있다 |
| 시드, 난수 상태 | 난수를 만드는 시작값. 같은 시드면 같은 난수가 나와 결과를 재현할 수 있다 |
