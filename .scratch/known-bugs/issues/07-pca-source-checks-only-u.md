# 07: Test_의 PCA 출처 조건이 U 체크포인트 번호만 검사한다

**Status:** resolved

**Blocked by:** None (can start immediately)

**현상**
- Test_는 temp_cache가 없을 때, U 체크포인트 번호가 -1이면 `pca_arrays.npz`를 쓴다.
- 이때 X 번호는 검사하지 않는다. 그래서 모델 쌍이 (U -1, X N)이면 최종 X-model 기준으로 계산된 npz가 조용히 쓰인다.
- Test_tsne는 체크포인트 번호와 상관없이 항상 npz로 fallback한다(README 9장).

**영향:** 모델 쌍과 맞지 않는 PCA 관리한계로 평가한다.

**고칠 때 주의:** 조건을 U와 X 모두 검사하도록 바꾸면, 해당 모델 쌍은 PCA 재계산 경로(06)로 넘어가 실패한다. 06과 05를 먼저 고쳐야 한다.

**출처:** 2026-10-14 아키텍처 리뷰.

## Comments

### 2026-10-15 수정 (05·06과 함께)

**바꾼 것**
- Test_와 Test_tsne는 캐시가 없을 때, U·X가 모두 -1일 때만 `pca_arrays.npz`를 쓴다. 그 밖의 모델 쌍은 `train_pca_only`로 관리도를 다시 계산한다.
- Test_tsne의 `--u-model-idx`·`--x-model-idx` 기본값을 10/10에서 -1/-1로 바꿨다(grilling Q5).
- README 7.3, 7.5, 9장을 고쳤다.

**검증**
- -1/-1 기준 경로는 `before1` 대비 test, tsne 모두 bit 단위로 일치했다(`fix07`).
- 다시 계산하는 경로는 05의 검증과 같다.
