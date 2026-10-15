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
