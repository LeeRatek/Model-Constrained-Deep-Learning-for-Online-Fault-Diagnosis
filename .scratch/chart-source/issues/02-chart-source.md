# 02: 관리도 공급원 module과 Test_·Test_tsne·Test_multi_models 전환

**What to build:** 모델 쌍의 관리도를 어디서 가져올지(같은 조건 캐시 → 최종 쌍이면 npz → 그 밖엔 재계산), 그리고 temp_cache 읽기·쓰기 형식을 `mcdl/chart_source.py` 하나에 둔다.
- Test_와 Test_tsne는 `chart_for_pair` 한 줄로 관리도를 얻는다.
- Test_multi_models는 같은 module로 캐시를 쓰고 읽는다.
- `Train_pca_only.py`는 없어진다.

**Blocked by:** 01 (재구성 오차·학습 진단 특징 module)

**Status:** resolved

- [x] `chart_for_pair`, `save_cached_features`, `load_cached_features`와 재계산 조건 상수를 spec대로 만든다. 재계산은 Run의 학습 차량 원본을 읽어 `training_features(batch 65536)`로 계산한다.
- [x] Test_와 Test_tsne의 관리도 출처 분기를 `chart_for_pair`로 바꾼다.
- [x] Test_multi_models의 캐시 쓰기(Phase 1·2, disk)와 읽기(조합 평가)를 `save_cached_features`와 `load_cached_features`로 바꾼다.
- [x] `Train_pca_only.py`와 그것을 가리키는 import를 지운다.
- [x] 다음 테스트를 추가한다. 모두 `tmp_path`의 Run과 작은 가짜 원본 데이터를 쓴다.
  - 캐시 분기
  - 조건이 다른 캐시를 무시하는지
  - npz 분기
  - 재계산 분기
  - 캐시 저장·로드 round-trip
- [x] 실데이터: test, test_u0, tsne, multi가 `after_bugs`와 bit 단위로 일치해야 한다. `--keep-cache` 캐시를 쓰고 Test_가 읽는 round-trip도 다시 확인한다. 결과를 Comments에 남긴다.
- [x] README(3장 구조, 6장, 7.3)를 갱신한다.

## Comments

### 2026-10-15 완료

**검증 결과**
- `python -m pytest`: 47개 통과.
- 새 `tests/test_chart_source.py`의 5개 테스트는 `tmp_path`의 Run과 가짜 QAS 원본 데이터(차량 3대, pkl)를 쓴다.
  - 재계산 분기: `training_features`로 직접 맞춘 관리도와 같다.
  - npz 분기: 최종 쌍이면 원본 데이터 없이 저장된 관리도를 쓴다.
  - 캐시 분기: 원본 데이터 없이 캐시만으로 맞춘다.
  - 조건(batch)이 다른 캐시는 무시하고 재계산한다.
  - 캐시 저장·로드 round-trip
- 실데이터(`compare <target> after_bugs cs02`): 네 대상 모두 bit 단위로 일치했다.

| target | 기록 수 | 소요 시간 |
|---|---|---|
| test | 304 | 46.3 s |
| test_u0 (재계산) | 513 | 119.9 s |
| tsne | 24 | 23.4 s |
| multi | 304 | 225.3 s |

- `--keep-cache` round-trip도 확인했다.
  - Test_multi_models는 `save_cached_features`로 `feat_u_-1_abs=1_float32=0_batch=65536.pkl`, `feat_x_-1_…`을 남겼다.
  - Test_는 `chart_for_pair`로 이 캐시를 찾아 썼다("Loading cached training features"). AUC는 0.9148, 최적 임계값은 23.9로 같았다.
  - 확인한 뒤 캐시는 지웠다.

**정리한 것**
- `Train_pca_only.py` 파일을 삭제했다. working tree에서만 지웠고, git에 stage하지 않았다.
- Test_와 Test_tsne에서 관리도 출처 분기(각 약 40줄), 캐시 조건 dict, 관련 import(`ControlChart`, `combine_signal_features`, `train_pca_only`, Test_tsne의 `import pickle`)를 지웠다.
- Test_multi_models에서 `u_error_files`·`x_error_files` dict, 함수 안의 `import pickle`, `Train_pca_only` import를 지웠다. help 문구와 주석의 `train_pca_only` 언급도 고쳤다.
- README의 3장, 6장(의존 관계, `mcdl/chart_source.py` 소절), 7.3, 테스트 목록을 고쳤다.
