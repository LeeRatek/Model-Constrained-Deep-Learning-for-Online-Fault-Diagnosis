# 02: 평가 스크립트가 normalize_val을 넘기지 않는다

**Status:** resolved

**Blocked by:** None (can start immediately)

**현상**
- 학습 쪽(Train_, Train_pca_only)은 `preprocess_loaded_tensor`에 `normalize_val`(기본 4)을 넘긴다.
- 평가 스크립트 3개(Test_, Test_multi_models, Test_tsne)는 넘기지 않으므로 함수 기본값 1.0이 쓰인다.

**영향**
- `--normalize-dx`로 학습한 Run에서는 학습과 평가의 Δ 스케일이 다르다. learning_case 2·3에만 해당한다.
- README 9장에 이미 적혀 있다.

**고칠 때 주의**
- 해당 Run의 평가 수치가 바뀐다.
- Test_multi_models의 `.pt` 전처리 캐시 해시에는 `normalize_val`이 들어가지 않는다. 그래서 고친 뒤에도 예전 캐시가 조용히 재사용된다.
- 해시 키는 기존 해시가 유지되도록 조건부로 추가해야 한다. 예를 들어 `normalize_dx`가 켜져 있을 때만 추가한다.
- run-module 02 이후에는 호출부에 이 티켓 번호가 주석으로 달려 있다.

**출처:** 2026-10-14 아키텍처 리뷰.

## Comments

### 2026-10-15 수정

**바꾼 것**
- Test_, Test_multi_models, Test_tsne가 `preprocess_loaded_tensor`에 `normalize_val=run.config.normalize_val`을 넘긴다. Test_multi_models는 학습 데이터와 테스트 데이터 양쪽 모두 넘긴다.
- Test_multi_models의 `.pt` 캐시 해시에는 `normalize_dx`가 켜져 있을 때만 `normalize_val` 키를 넣는다. 그래서 `normalize_dx`가 꺼진 Run은 기존 해시와 캐시 파일을 그대로 쓴다.
- README 9장에서 이 버그 항목을 지웠다.

**영향**
- 기존 Run 10개는 모두 `normalize_dx=False`라서 결과가 바뀌지 않는다.
- `before1` 대비 test, tsne, multi 모두 bit 단위로 일치했고(`fix02`), multi는 기존 41 GB 캐시를 재사용했다.
- `normalize_dx=True`로 학습한 Run을 실제로 돌려 보지는 않았다. 그런 Run이 아직 없다.
