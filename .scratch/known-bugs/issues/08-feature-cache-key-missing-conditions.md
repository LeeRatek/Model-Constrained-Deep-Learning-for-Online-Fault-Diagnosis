# 08: temp_cache 특징 캐시의 키에 계산 조건이 빠져 있다

**Status:** resolved

**Blocked by:** None (can start immediately)

**현상**
- Test_multi_models의 학습 특징 캐시 파일 이름에는 체크포인트 번호만 들어간다.
- 오차 abs 여부(`no_abs_err`), `--use-float32`, 추론 batch 크기가 달라도 같은 파일을 다시 쓴다.
- 오래된 캐시를 지우는 코드는 주석 처리되어 있다.
- Test_와 Test_tsne는 `pca_arrays.npz`보다 이 캐시를 먼저 쓴다. 그래서 두 스크립트의 결과가 이전에 Test_multi_models를 어떤 옵션으로 돌렸는지에 따라 달라진다.

**영향:** 같은 모델 쌍을 평가해도 남아 있는 캐시에 따라 PCA 관리한계가 달라진다.

**고칠 때 주의:** 캐시 키를 바꾸면 기존 캐시가 모두 무효가 된다. `--keep-cache`를 쓰는 작업 흐름도 영향을 받는다.

**출처:** 2026-10-14 아키텍처 리뷰.

## Comments

### 2026-10-15 수정

**바꾼 것**
- 캐시 파일 이름 규칙을 `Run.feature_cache_path(model, 번호, abs_err=, float32=, batch_size=)` 한 곳에 두었다. 이름은 `temp_cache/feat_{u|x}_{번호}_abs={0|1}_float32={0|1}_batch={N}.pkl` 형식이다.
- Test_multi_models는 자기 실행 조건으로 캐시를 찾고 쓴다. 조건은 설정의 abs 여부, `--use-float32`, `--inference-batch-size`다.
- Test_와 Test_tsne는 자기가 직접 계산할 때의 조건과 이름이 같은 캐시만 쓴다. 조건은 설정의 abs 여부, float32 끔, batch 65536이다.
- 이름 형식이 옛것인 캐시(`feat_u_<번호>.pkl`)는 더 이상 쓰지 않는다.
- README 7장·8장의 캐시 이름 설명을 고쳤다.

**함께 살아난 경로**
- 지금까지 `--cache-strategy memory`에서는 `temp_cache_dir`가 `None`이라 `os.path.join(None, …)`에서 TypeError가 나 항상 실패했다.
- 이제 경로를 Run이 계산하므로 이 경로도 실패하지 않는다. spec에서 정한 "예외 경로가 살아나는 것은 허용" 정책에 해당한다. 실데이터 확인 결과는 아래 "memory 모드 실데이터 확인"에 있다.

**검증**
- `before1` 대비 test, tsne, multi 모두 bit 단위로 일치했다(`fix08`).
- `Test_multi_models --keep-cache`(-1/-1)는 `feat_u_-1_abs=1_float32=0_batch=65536.pkl`과 `feat_x_-1_…`을 남겼다. 이어서 돌린 `Test_.py`(-1/-1)는 이 캐시를 찾아 관리도를 계산했다("Loading cached training features"). AUC는 0.9148로 같았다. 확인한 뒤 캐시는 지웠다.
- 새 테스트 `test_feature_cache_path_names_the_conditions`를 추가했다.

### 2026-10-15 memory 모드 실데이터 확인

- `Test_multi_models.py --models-dir ./models/_verify --models-idx 260224_090359 --u-model-idx -1 --x-model-idx -1 --cache-strategy memory`가 끝까지 실행되었다(exit 0).
- 학습 특징을 메모리에 두고 썼고("Using pre-computed FEATURES from memory"), temp_cache 폴더는 만들지 않았다.
- AUC_summary CSV 행은 disk 모드 기준선(`after_bugs`)과 텍스트까지 같았다: `-1,-1,0.9148,23.900000000000002,0.78,0.02,23.900000000000002,0.78,0.02,0.2209072203437452`
