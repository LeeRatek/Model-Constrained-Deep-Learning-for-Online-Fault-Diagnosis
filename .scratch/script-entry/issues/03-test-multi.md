# 03: Test_multi_models를 `main(argv)`로

**What to build:** Test_multi_models를 `main(argv=None)` 구조로 바꾼다. `process_combination`은 main 안의 closure로 두고, `global` 문을 지운다. 평가 결과와 CSV는 그대로다.

**Blocked by:** 01

**Status:** resolved

- [x] 본문을 `main(argv)`로 옮긴다. import 시점 부수 효과를 main 안으로 옮긴다.
- [x] `process_combination`의 `global` 문을 지운다. 바깥 이름을 다시 할당하지 않는지 확인한다.
- [x] smoke 테스트: 01의 Run으로 `Test_multi_models.main`을 실행하고 `AUC_summary` CSV를 확인한다.
- [x] 실데이터: multi가 `after_bugs`와 bit 단위로 일치해야 한다. 결과를 Comments에 남긴다.
- [x] README 7장(CLI 예시를 `--models-idx`로 통일)과 9장(디버거 표 → `--debug`)을 갱신한다.

## Comments

**2026-10-15 완료**

- 본문을 `main(argv)`로 옮겼다. `process_combination`은 main 안의 closure다.
- `global total_test_vehicles, completed_combinations, all_roc_results`를 지웠다.
  - 세 이름 모두 다시 할당하지 않는다. `set.add`, `list.append`, 읽기만 한다.
  - main의 `"step1_results" in locals()`, `"train_data_cache" in locals()` 검사와 `del u_error_cache, x_error_cache`는 함수 안에서도 같은 의미라 그대로 두었다. closure가 참조하는 cell 변수도 `del`할 수 있다.
- tests: `test_multi_evaluates_all_checkpoint_pairs_and_heatmap`
  - 가짜 Run의 체크포인트 4개 조합(0/-1 × 0/-1)이 모두 CSV에 남는다.
  - AUC 히트맵이 만들어지고, temp_cache는 지워진다.
  - 이어서 `Plot_AUC_heatmap.main`이 같은 CSV로 히트맵을 만든다.
  - 전체 pytest 69개 통과.
- 실데이터(기준선 태그 `se03`): multi가 `after_bugs`와 304개 기록 모두 bit 단위로 일치한다. AUC_summary CSV 텍스트를 포함한다.
- README
  - 7장: Test_ 예시를 `--models-idx`로 통일하고, `--debug`와 `MODEL_DIR`을 반영했다.
  - 4장: `test_scripts.py`를 추가했다.
  - 9장: 디버거 표를 `--debug` 표로 바꿨다.
  - 7.2: `--no-abs-err` 설명을 버그 10 수정에 맞췄다.
