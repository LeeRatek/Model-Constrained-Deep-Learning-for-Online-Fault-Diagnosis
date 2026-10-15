# Spec: Train_ 학습 루프 통합

**Status:** resolved

출처: 2026-10-15 2차 아키텍처 리뷰의 6번 후보("Train_ 학습 루프")와 같은 날 grilling(Q1–Q5). 용어는 `GLOSSARY.md`를 따른다.

## Problem Statement

Train_의 U-model 학습 루프와 X-model 학습 루프는 거의 같은 코드다. 학습, 검증, 체크포인트, loss CSV, 최종 모델 저장이 두 번씩 들어 있다. 차이는 세 가지뿐이다.
- 하이퍼파라미터 인자 이름(`ae_u_*` / `ae_x_*`)
- 파일 이름(`net`/`netx`)
- Dataset의 열 키(`x,y,z` / `x2,y2,z2`)

한쪽만 고치면 두 모델의 학습 방식이 조용히 달라진다. Dataset 클래스는 6개인데 그중 4개는 쓰이지 않는다. 쓰이는 2개도 열 키만 다르다.

지금 학습 기준선은 차량 1대, 1 epoch, shuffle 끔, 검증 생략이다. 그래서 RNG 소비 순서, 검증 경로, 체크포인트 간격이 바뀌어도 잡아내지 못한다.

## Solution

1. 리팩토링 전에 실데이터 기준선 `train_full`을 추가하고, 결정성을 확인한다.
2. 모델 하나를 학습하는 함수 `train_autoencoder`와 Dataset `VehicleRows`를 Train_.py 모듈 최상위에 둔다. main은 U, X 순서로 이 함수를 부른다.
3. 학습 결과(체크포인트, loss CSV, 관리도)는 bit 단위로 같다.

## User Stories

1. 연구자로서, 리팩토링 전후 학습 결과가 bit 단위로 같길 원한다. shuffle을 켜거나 검증을 포함한 학습도 마찬가지다.
2. 연구자로서, U-model과 X-model이 같은 방식으로 학습된다는 것을 코드에서 바로 확인하고 싶다.
3. 개발자로서, 학습 루프를 한 곳에서만 고치고 싶다.
4. 개발자로서, 차량 텐서의 열 구성을 재구성 쪽(`reconstruct`)과 같은 규칙(모델 크기)으로 읽고 싶다.
5. 리뷰어로서, RNG 순서를 바꾸지 않았다는 근거를 기준선 비교로 보고 싶다.

## Implementation Decisions

### RNG 사실 (torch 2.9.1 실측)

| 동작 | 전역 RNG |
|---|---|
| DataLoader 생성, Adam 생성, `module.double()` | 쓰지 않는다 |
| DataLoader 순회 (iterator를 만들 때마다) | 쓴다 (shuffle을 꺼도) |

- `.double()`은 Parameter 객체를 바꾸지 않는다. 먼저 만든 optimizer도 그대로 유효하다.
- 그래서 함수 안의 준비 순서(loader, optimizer, `.double()`)는 결과에 영향이 없다. 지켜야 할 것은 epoch마다 학습 loader를 순회하고, `epoch % save_per == 0`일 때 검증 loader를 순회하는 순서다. 그리고 U 학습 뒤에 X 학습을 한다.

### interface (grilling Q3)

```python
class VehicleRows(Dataset):
    def __init__(self, tensors, x_dim, y_dim)   # 시점 하나를 (x, y, Δ, q) double로 꺼낸다. 열 구성 [x | y | Δ | q]

def train_autoencoder(net, model, train_tensors, val_tensors, *, epochs, save_per, lr,
                      batch_size, shuffle, device, run_dir, save) -> None
```

- `x_dim`, `y_dim`은 모델 크기(`net.fc1.in_features`, `net.fc3.out_features`)에서 읽는다. `reconstruct`와 같은 규칙이다.
- `model`("U"/"X")은 체크포인트 이름(`checkpoint_filename`)과 loss CSV 이름(`loss_net*.csv`/`loss_netx*.csv`)에 쓴다.
- 검증 차량이 없으면(`--no-validation`) 검증 loader는 빈 목록이다. 지금과 같다.
- 함수 안의 준비 순서는 지금 U 쪽 순서다. 콘솔 출력과 CSV 형식은 지금과 같다.

### 지우는 것

- 쓰이지 않는 Dataset 4개: `VirtualConcatDataset`, `VirtualConcatDatasetX`, `VirtualConcatDatasetVal`, `VirtualDatasetVal`
- `VirtualDatasetU`, `VirtualDatasetX`: `VehicleRows`로 대신한다.
- Train_ main의 `dim_dict`: 이제 쓰이지 않는다.

### 바꾸지 않는 것 (Q1)

- LSTM 레거시, 데이터 로딩, 학습 진단 특징·관리도
- 행 단위 `__getitem__` 데이터 파이프라인과 그 성능

## Testing Decisions

**실데이터 기준선 `train_full` (Q4)**
- 설정: `--vehicle-start 0 --vehicle-end 1 --ae-u-epochs 3 --ae-x-epochs 3 --ae-u-save-per 2 --ae-x-save-per 2 --ae-u-shuffle --ae-x-shuffle`, 검증 포함(루트 분할 validate 50대)
- 리팩토링 전에 두 번(`tl00a`, `tl00b`) 떠서 서로 bit 단위로 같은지 확인한다. 리팩토링 뒤에는 `tl00a`와 비교한다.

**smoke 테스트:** 가짜 데이터로 3 epoch, `save-per 2`, shuffle, 검증을 넣어 Train_을 실행한다. 체크포인트 번호가 `[0, 2, -1]`이고, 검증 loss가 epoch 0과 2에 기록되는지 확인한다.

**티켓별 검증**
- 01: `tl00a` = `tl00b`. smoke 테스트는 리팩토링 전 코드에서 통과해야 한다.
- 02: train(`after_bugs` 대비), `train_full`(`tl00a` 대비), pytest

## Out of Scope

- 데이터 파이프라인 성능 개선(배치 단위 슬라이싱 등)
- `mcdl/`로 옮기는 것. 호출자가 Train_ 하나뿐이기 때문이다(Q2).
- LSTM 레거시
