# 05: Train_pca_only가 차량 경계를 넘어 진단 특징을 계산한다

**Status:** resolved

**Blocked by:** None (can start immediately)

**현상**
- Train_pca_only는 학습 차량 텐서를 모두 이어 붙인 뒤 재구성 오차를 구하고, `DiagnosisFeature`를 한 번만 호출한다.
- Train_과 Test_multi_models는 차량별로 특징을 계산한 뒤에 이어 붙인다.
- 이 차이 때문에 EWMA와 100-step 이동평균이 차량 경계를 넘어 계산된다. 같은 체크포인트로 계산해도 `pca_arrays.npz`와 다른 PCA 관리한계가 나온다.

**영향:** 지금은 유일한 호출 경로(06)가 항상 실패하므로 드러나지 않는다.

**고칠 때 주의**
- 06과 함께 다뤄야 한다. 06만 고치면 이 차이가 곧바로 결과에 나타난다.
- 진단 모니터 후보(4번)에서 Train_pca_only를 흡수할지 함께 결정한다.

**출처:** 2026-10-14 아키텍처 리뷰.

## Comments

### 2026-10-15 위치 변경 (control-chart 02)

- Train_pca_only는 이제 `mcdl/diagnosis.py`의 `diagnosis_features`와 `ControlChart`를 쓰고, 관리도를 돌려준다.
- 이어 붙인 오차에 `diagnosis_features`를 한 번 적용하는 순서는 그대로 두었고, 해당 줄에 이 티켓 번호를 주석으로 달았다.
- 고칠 때는 Train_처럼 차량마다 `diagnosis_features`를 구해 이어 붙이면 된다. Test_multi_models처럼 신호별 특징을 쓰는 방법도 있다.

### 2026-10-15 수정 (06·07과 함께)

**바꾼 것**
- `train_pca_only`를 다시 썼다. 학습 차량마다 재구성 오차(batch 65536) → `diagnosis_features`를 구해 이어 붙인 뒤 `ControlChart.fit`을 한다.
- Test_multi_models의 학습 특징 계산과 같은 방식이다.

**검증**
- 기준 Run 260224_090359에서 `Test_.py --u-model-idx 0 --x-model-idx -1`을 캐시 없이 돌렸다.
- 원본 Test_multi_models CSV의 (0, -1) 행과 출력 정밀도 기준으로 같은 값을 냈다.
  - AUC 0.8558
  - Youden 최적점 26.1 (FPR 0, TPR 0.58)
  - (0,1) 최근접점 19.2 (FPR 0.12, TPR 0.68), 거리 0.3418
