# 01: 평가 시 가중치가 float32로 잘린 채 쓰인다

**Status:** resolved

**Blocked by:** None (can start immediately)

**현상**
- 학습은 double 정밀도로 진행되고, 체크포인트도 float64로 저장된다.
- 평가 경로 4곳은 기본 dtype(float32)으로 `CombinedAE`를 만든 뒤 `load_state_dict` → `.double()` 순서로 로드한다.
  - 해당 경로: Test_, Test_multi_models, Test_tsne, Train_pca_only
- 그래서 평가에 쓰이는 가중치는 float32로 반올림된 값이다.

**위치:** 각 평가 스크립트의 모델 로드 코드. run-module 02 이후에는 Run module의 `load_model_pair`가 이 순서를 유지한다.

**영향:** 학습이 끝날 때 계산하는 `pca_arrays.npz`는 full-precision 가중치를 쓰고, 테스트 차량 진단은 잘린 가중치를 쓴다. 같은 Run 안에서 두 정밀도가 섞인다.

**고칠 때 주의**
- 기존 Run의 AUC와 최적 임계값이 모두 달라진다.
- Run module 테스트의 "full-precision 로드와 다르다" 단언이 뒤집힌다.
- 무엇을 바꿀지보다 고칠지 말지를 먼저 결정해야 한다.

**출처:** 2026-10-14 아키텍처 리뷰. 체크포인트 dtype은 직접 확인했다. grilling Q6에서 별도 티켓으로 다루기로 했다.

## Comments

### 2026-10-15 수정

**바꾼 것**
- `Run.load_model`이 모델을 double로 만든 뒤 가중치를 load한다(grilling Q1). 이제 저장된 float64 가중치가 잘리지 않고 그대로 쓰인다.
- 이 동작을 고정하던 테스트를 뒤집었다. 지금은 `test_load_model_keeps_saved_float64_weights`가 load한 가중치가 저장된 값과 bit 단위로 같은지 확인한다.
- README 9장에 주의 문구를 넣었다. 2026-10-15 이전의 평가 결과는 잘린 가중치 기준이라는 내용이다.
- 기존 Run은 다시 평가하지 않았다(grilling Q1).

**기준 Run 260224_090359(-1/-1)에서 고치기 전후 비교**(`before1` vs `after_bugs`)
- train: 학습은 원래 double이었으므로 그대로이고, bit 단위로 일치했다.
- test: 차량 100대의 진단 특징, T², SPE가 모두 바뀌었다.
  - 그러나 차량 × 임계값(0.1 간격) 예측 행렬은 완전히 같았다. 판정이 바뀐 차량은 0대다.
  - 그래서 AUC 0.9148, Youden·(0,1) 최근접 임계값 23.900000000000002(FPR 0.02, TPR 0.78)가 그대로다.
- multi: 학습 특징으로 계산한 관리도(`Custom_PCA` 입력과 출력)와 테스트 차량 값이 바뀌었다. 그래도 AUC_summary CSV 행은 텍스트까지 같았다.
- tsne: CI 시계열과 t-SNE 입력이 바뀌었다(그림만 바뀜).
- 다른 체크포인트나 다른 Run에서는 판정이 달라질 수 있다.

**새 기준선:** 이후 리팩토링은 `models/_verify/baselines/after_bugs/`와 비교한다. 대상은 train, test, tsne, multi다.
