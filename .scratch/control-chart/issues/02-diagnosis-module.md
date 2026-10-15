# 02: 진단 특징·관리도 module과 스크립트 5개 전환

**What to build:** 진단 특징과 관리도 계산을 `mcdl/diagnosis.py`로 모은다. 이 module을 쓰도록 바꾸는 스크립트는 Train_, Test_, Test_multi_models, Test_tsne, Train_pca_only다. 스크립트는 15-tuple을 풀지 않고 관리도를 주고받는다. 진단 특징 구현은 하나만 남는다. 디스크 형식과 수치 결과는 그대로다.

**Blocked by:** 01 (진단 계산의 현재 동작 고정)

**Status:** resolved

- [x] `signal_features`, `combine_signal_features`, `diagnosis_features`, `ControlChart`(`fit`, `load`, `t2_limit`, `spe_limit`, `save`, `score`, `max_ci`)를 spec대로 만든다.
- [x] `ControlChart`는 `data_nor`를 들고 있지 않는다. `save`가 다시 계산해서 쓴다.
- [x] Train_이 차량별 진단 특징 → `fit` → `save`를 쓴다. 출력 경로 문자열 형식은 지금과 같다.
- [x] Test_와 Test_tsne를 바꾼다.
  - PCA 출처 결정 정책은 그대로 둔다. 각 분기는 관리도를 얻기만 한다.
  - 테스트 차량 처리에는 `diagnosis_features`와 `score`/`max_ci`를 쓴다.
- [x] Test_multi_models를 바꾼다.
  - 학습 특징 캐시는 `signal_features`로 만든다. 클론 `extract_features_from_error`와 `sliding_average_np`는 삭제한다.
  - 조합마다 `combine_signal_features` → `fit`을 부른다.
  - 테스트 차량에는 `diagnosis_features` → `max_ci`를 쓴다.
- [x] Train_pca_only가 새 interface를 쓰고 `ControlChart`를 돌려준다. 버그 05의 순서(이어 붙인 뒤 특징 계산)는 유지한다. Test_의 해당 분기는 돌려받은 관리도를 그대로 쓴다.
- [x] Function_.py에서 `DiagnosisFeature`, `Custom_PCA`, `T2_array`, `SPE_array`, `save_pca_results`, `load_pca_results`를 지운다. 이 변경으로 고아가 된 import도 지운다.
- [x] 01의 fixture 테스트가 새 module을 대상으로 bit 단위로 통과한다. 다음 테스트를 추가한다.
  - 신호별 조합과 `diagnosis_features`가 같은지
  - `save` → `load` round-trip 후 CI가 같은지
- [x] 기준선 도구가 새 위치의 함수를 감싸게 한다. 실데이터 4종(train, test, tsne, multi)이 `before1`과 bit 단위로 일치해야 하고, 결과를 Comments에 남긴다.
- [x] README 6장(의존 관계, Function_.py 함수 표, mcdl 소절)을 갱신한다.

## Comments

### 2026-10-15 완료

**검증 결과**
- `python -m pytest`: 35개 통과, 약 9초.
  - `tests/test_diagnosis.py`를 새 module 대상으로 바꿨다.
  - 다음 테스트를 추가했다.
    - Test_multi_models 경로(신호별 특징을 차량마다 구해 신호별로 이어 붙인 뒤 묶기)와 Train_ 경로(차량별 진단 특징 이어 붙이기)가 `assert_frame_equal(check_exact=True)`로 같다.
    - 저장 형식(npz 키별 값)이 fixture와 같다.
    - 로드한 관리도의 CI가 fixture와 같다.
    - manifest가 없으면 로드가 실패한다.
    - 2D가 아닌 오차는 거부한다.
  - ROC 테스트는 아직 Function_ 구현을 쓴다(티켓 03).
- 기준선 비교 결과(`compare <target> before1 after4_02`): 네 대상 모두 모든 기록이 bit 단위로 일치했다.

| target | 기록 수 | 소요 시간 |
|---|---|---|
| train | 14 | 3.7 s |
| test | 304 | 43.4 s |
| tsne | 24 | 22.9 s |
| multi | 304 | 226.6 s (temp_cache 남지 않음) |

**구현 메모**
- 옮긴 계산 코드(`_pca`, `_t2`, `_spe`, `_save_pca`, `_sliding_average`)는 원본 그대로이고, 이름만 private으로 바꿨다.
- `_load_pca`는 쓰이던 경로만 남겼다. 그 경로는 `data_nor` 없이, shape 검증 없이 읽고, manifest가 없으면 실패한다.
- 기준선 도구의 `WRAP_TARGETS`에 새 위치를 추가했다.
  - `diagnosis_features`는 `legacy_out`으로 이전 `(df, None)` 모양을 만들어 같은 이름(`DiagnosisFeature`)으로 기록한다.
  - `_pca`, `_t2`, `_spe`, `_load_pca`도 이전 이름으로 기록한다.
- Test_의 진단 그림 모드(`AUC_ANALYSIS = False`)는 `score()`의 T²·SPE·CI를 쓴다. 기본 모드는 `max_ci()`를 쓴다. 차량마다 점수 계산은 한 번만 일어난다.
- 쓰지 않던 `CI_contrib` 계산과 `analyse_ae_output` 변수(`get_true_feature` 옵션용)는 없어졌다.
- Test_와 Test_tsne의 PCA 출처 분기에 known-bugs 07과 README 9장을 가리키는 주석을 달았다.

**정리한 것**
- Function_.py: 함수 6개(645줄), 그리고 import `json`, `scipy.stats`
- Test_multi_models.py: `sliding_average_np`, `extract_features_from_error`
- Test_.py: `analyse_ae_output`

**검증하지 못한 경로**
- Train_pca_only(known-bugs 06 때문에 도달할 수 없음)
- Test_의 진단 그림 모드(`AUC_ANALYSIS = False`, 코드 상수)
- temp_cache가 남아 있을 때의 Test_·Test_tsne 캐시 분기
- 세 경로 모두 실데이터로는 돌려 보지 않았다. 캐시 분기에서 쓰는 `combine_signal_features` → `fit`은 Test_multi_models의 실데이터 비교로 검증되었다.
