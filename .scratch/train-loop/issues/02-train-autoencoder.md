# 02: `train_autoencoder`와 `VehicleRows`

**What to build:** U·X 학습 루프를 `train_autoencoder` 하나로, 쓰이는 Dataset 2개를 `VehicleRows` 하나로 합친다. 쓰이지 않는 Dataset 4개와 `dim_dict`를 지운다. 학습 결과는 bit 단위로 같다.

**Blocked by:** 01

**Status:** resolved

- [x] `VehicleRows(tensors, x_dim, y_dim)`: 시점 하나를 (x, y, Δ, q) double로 꺼낸다.
- [x] `train_autoencoder(net, model, train_tensors, val_tensors, *, …)`: 학습, 검증, 체크포인트, loss CSV, 최종 모델 저장을 맡는다. main은 U, X 순서로 부른다.
- [x] Dataset 6개(쓰이지 않는 4개 포함)와 main의 `dim_dict`를 지운다.
- [x] 실데이터: train이 `after_bugs`와, `train_full`이 `tl00a`와 bit 단위로 일치해야 한다. 결과를 Comments에 남긴다.
- [x] pytest가 통과한다.

## Comments

**2026-10-15 완료**

- `VehicleRows(tensors, x_dim, y_dim)`
  - `VirtualDatasetU`·`VirtualDatasetX`를 대신한다. 이진 탐색, 행 인덱싱, 열 구간, `.to(torch.double)`은 그대로다.
  - x·y 크기는 `train_autoencoder`가 `net.fc1.in_features`, `net.fc3.out_features`에서 읽어 넘긴다.
- `train_autoencoder(net, model, train_tensors, val_tensors, *, …)`
  - U 쪽 루프를 옮긴 것이다. 달라지는 부분은 `model`("U"/"X")로 정한다. 체크포인트 이름은 `checkpoint_filename`, loss CSV 이름은 `LOSS_CSV`가 정한다.
  - `--no-validation`이면 main이 `val_tensors=None`을 넘긴다. 이때 검증 loader는 예전처럼 빈 목록이다. 빈 목록을 순회하면 RNG를 쓰지 않는다.
- main은 `train_autoencoder(net, "U", …)` 다음 `train_autoencoder(netx, "X", …)`를 부른다.
- 지운 것
  - Dataset 6개(쓰이지 않던 4개 포함)
  - main의 `dim_dict`, `model_path`, `AE_U_*`·`AE_X_*` 변수
  - Train_.py는 527줄이 됐다(이 티켓 전 824줄).
- 실데이터(기준선 태그 `tl02`): 모두 bit 단위로 일치한다.
  - train: `after_bugs`와 14개 기록
  - train_full: `tl00a`와 17개 기록. 대상은 shuffle, 검증 50대, 체크포인트 0·2·-1, loss CSV 4개, 관리도다
- pytest 70개 통과. 체크포인트 간격 smoke 테스트가 리팩토링 전후 모두 통과한다.
- README
  - 4장: 기준선 대상 목록과 `train_full` 설명
  - 7.2: `train_autoencoder`와 재현성 주의(DataLoader 순회와 shuffle 순서)
