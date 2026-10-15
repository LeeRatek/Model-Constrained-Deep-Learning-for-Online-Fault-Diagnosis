# 01: 재구성 오차·학습 진단 특징 module과 학습 쪽 세 경로 전환

**What to build:** 재구성 오차와 학습 진단 특징 계산을 `mcdl/reconstruction.py` 하나로 모은다. 이 module을 쓰도록 바꾸는 곳은 세 곳이다.
- Train_의 학습 끝 PCA 루프(batch 4096)
- Test_multi_models의 Phase 1·2(신호별, 캐시)
- 관리도 재계산(지금의 Train_pca_only)

Test_multi_models와 재계산이 같은 학습 진단 특징을 낸다는 것이 코드 구조로 보장된다. 수치 결과는 그대로다.

**Blocked by:** None (can start immediately)

**Status:** resolved

- [x] `reconstruction_errors`, `training_features`, `training_signal_features`를 spec대로 만든다. 열 구성은 모델에서 읽고, 추론은 모델이 있는 device에서 batch 단위로 한다.
- [x] Train_의 PCA 루프가 `training_features(…, batch_size=4096)`를 쓴다.
- [x] Test_multi_models의 Phase 1·2가 `training_signal_features`를 쓴다. `--use-float32`와 `--inference-batch-size`는 그대로 전달하고, `run_model_inference`는 삭제한다.
- [x] `Train_pca_only.train_pca_only`가 `training_features(…, batch_size=65536)`를 쓴다. 파일 삭제는 02에서 한다.
- [x] 다음 테스트를 추가한다. 모두 작은 `CombinedAE`와 합성 차량 텐서를 쓴다.
  - batch 분할 결과
  - abs와 부호 있는 오차의 구분
  - float32 옵션
  - 두 학습 특징 경로가 같은지
- [x] 실데이터: train, multi, test_u0이 `after_bugs`와 bit 단위로 일치해야 한다. 결과를 Comments에 남긴다.
- [x] README 6장(mcdl 소절, 의존 관계)을 갱신한다.

## Comments

### 2026-10-15 완료

**검증 결과**
- `python -m pytest`: 47개 통과(02용 테스트 5개 포함).
- 새 `tests/test_reconstruction.py`의 5개 테스트
  - batch 크기 1000/64/7에서 참조 계산과 bit 단위로 같음
  - abs와 부호 있는 오차의 구분
  - float32 옵션
  - `training_features`가 차량별 진단 특징을 이어 붙인 것과 같음
  - 신호별 경로와 U·X 쌍 경로가 같음(`assert_frame_equal`, exact)
- 실데이터(`compare <target> after_bugs cs01`): 세 대상 모두 bit 단위로 일치했다.

| target | 기록 수 | 소요 시간 |
|---|---|---|
| train | 14 | 4.0 s |
| test_u0 | 513 | 120.0 s |
| multi | 304 | 229.7 s |

- test_u0은 이번에 기준선 도구에 추가한 대상이다. Test_(U 0 / X -1)를 캐시 없이 돌려 관리도를 재계산하는 경로이며, 학습 차량 209대의 진단 특징까지 기록한다.
- `after_bugs` 기준선은 전환 전 코드로 떴다.

**정리한 것**
- Test_multi_models: `run_model_inference`(92줄), Phase 1·2의 차량 루프, `signal_features` import
- Train_: PCA 루프(약 80줄)와 `diagnosis_features` import
- Train_pca_only: `_reconstruction_errors`
