# 01: 학습 기준선 강화

**What to build:** 리팩토링 전에 학습 루프의 RNG 순서, 검증 경로, 체크포인트 간격을 비교할 수 있는 기준선을 만든다.

**Blocked by:** None (can start immediately)

**Status:** resolved

- [x] `tests/realdata/baseline.py`에 `train_full` 대상을 추가한다. 차량 2대, 3 epoch, `save-per 2`, U·X shuffle, 검증 포함이다.
- [x] 리팩토링 전 코드로 `train_full`을 두 번(`tl00a`, `tl00b`) 떠서 bit 단위로 같은지 확인한다.
- [x] `tests/test_scripts.py`에 smoke 테스트를 추가한다. 가짜 데이터로 3 epoch·`save-per 2`·shuffle·검증을 실행해, 체크포인트 번호 `[0, 2, -1]`과 검증 loss의 epoch(0, 2)를 확인한다. 리팩토링 전 코드에서 통과해야 한다.
- [x] 결과를 Comments에 남긴다.

## Comments

**2026-10-15 완료**

- `tests/realdata/baseline.py`에 `train_full` 대상을 추가했다. 결과 Run은 `train`과 같은 `models/_verify/train/`에 만들고, Run 폴더의 모든 파일을 기록한다.
  - 기록하는 파일: 체크포인트 6개(e0·e2·최종 × U·X), loss CSV 4개, 관리도, sim_config 등 17개
- 리팩토링 전 코드로 두 번 떴다. `tl00a` 191.2초, `tl00b` 180.9초이고, 두 결과가 bit 단위로 같다(결정적).
- 콘솔: U는 epoch 0·2에서 Train/Val loss 0.0092/0.0035 → 0.0030/0.0026, X는 0.0328/0.0325 → 0.0307/0.0329다. 검증 차량 50대 경로를 실제로 탔다.
- smoke 테스트 `test_train_checkpoint_schedule_with_shuffle_and_validation`을 추가했다. 리팩토링 전 코드에서 통과한다.
  - 체크포인트 `[0, 2, -1]`
  - 학습 loss epoch `[0, 1, 2]`
  - 검증 loss epoch `[0, 2]`
