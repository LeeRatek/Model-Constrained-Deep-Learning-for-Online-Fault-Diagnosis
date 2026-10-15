# 06: Test_에서 체크포인트를 지정하면 PCA 재계산 경로가 AttributeError로 실패한다

**Status:** resolved

**Blocked by:** None (can start immediately)

**현상**
- Test_에서 `--u-model-idx`를 -1이 아닌 값으로 주고 temp_cache가 없으면 `train_pca_only`를 호출한다.
- 이때 바로 앞에서 보정한 `vals`가 아니라 Test_의 `args`를 넘긴다.
- `args`에는 `battery_type`, `learning_case` 등이 없으므로 AttributeError가 난다.

**영향**
- README 9장에 이미 적혀 있다.
- 우회 방법은 `Test_multi_models --keep-cache`로 캐시를 먼저 만들어 두는 것이다.

**고칠 때 주의:** `vals`를 넘기도록 고치면 05 때문에 `pca_arrays.npz`와 다른 방식으로 PCA가 계산된다. 05와 함께 고쳐야 한다.

**출처:** README 9장, 2026-10-14 아키텍처 리뷰.

## Comments

### 2026-10-15 수정 (05·07과 함께)

**바꾼 것**
- `train_pca_only(run, u, x, source_data_dir, inference_batch_size)`는 Test_의 `args` 대신 Run을 받아, 설정을 Run에서 읽는다. 그래서 AttributeError가 나던 경로가 동작한다.
- 아무도 쓰지 않던 `save=True` 모드와 실행되지 않던 `__main__`은 지웠다(grilling Q4).
- README 9장의 해당 제약 두 항목을 지웠다.

**검증:** 05와 같다. `Test_.py --u-model-idx 0 --x-model-idx -1`이 원본 CSV의 (0, -1) 행과 같은 결과를 냈다.
