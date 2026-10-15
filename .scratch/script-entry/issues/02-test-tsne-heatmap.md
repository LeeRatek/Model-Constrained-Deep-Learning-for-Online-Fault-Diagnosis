# 02: Test_, Test_tsne, Plot_AUC_heatmap을 `main(argv)`로

**What to build:** Test_, Test_tsne를 `main(argv=None)` 구조로 바꾸고, Plot_AUC_heatmap은 파싱을 main 안으로 옮긴다. Test_에 `--debug`, `--models-idx` 별칭, `MODEL_DIR`을 넣는다. 평가 결과는 그대로다.

**Blocked by:** 01 (smoke 테스트가 Train_으로 Run을 만든다)

**Status:** resolved

- [x] Test_: 본문을 `main(argv)`로 옮긴다.
  - `--debug`이면 차량 338을 고정한다. 모델 쌍 강제는 없앤다(known-bugs 11 해소).
  - `--models-idx`를 `--models-folder`의 별칭으로 받는다.
  - `--models-dir` 기본값이 `MODEL_DIR`을 따른다.
- [x] Test_tsne: 본문을 `main(argv)`로 옮긴다. `add_vehicle_points`와 `collect_vehicle_points`는 모듈 최상위에 둔다.
- [x] Plot_AUC_heatmap: `main(argv)` 안에서 파싱한다.
- [x] smoke 테스트: 01의 Run으로 `Test_.main`과 `Test_tsne.main`을 실행한다. Test_는 `--models-folder`와 `--models-idx` 둘 다로 실행되어야 한다.
- [x] 실데이터: test, test_u0, tsne가 `after_bugs`와 bit 단위로 일치해야 한다. 결과를 Comments에 남긴다.
- [x] known-bugs 11을 닫는다.

## Comments

**2026-10-15 완료**

- Test_
  - 본문을 `main(argv)`로 옮겼다.
  - `--models-folder`와 `--models-idx`를 같은 dest(`models_folder`)로 받는다.
  - `--models-dir` 기본값이 `MODEL_DIR`을 따른다.
  - `--debug`이면 차량 338만 평가한다. 모델 쌍 (2, 1) 강제를 없애고, `chart_for_pair`도 `u_model_idx, x_model_idx`로 부른다. 이제 관리도와 모델은 항상 같은 쌍이다.
- Test_tsne: 본문을 `main(argv)`로 옮겼다. `add_vehicle_points`와 `collect_vehicle_points`는 모듈 최상위에 둔다.
- Plot_AUC_heatmap: 파싱을 `main(argv)` 안으로 옮겼다.
- 세 스크립트 모두 `KMP_DUPLICATE_LIB_OK` 설정, GPU 개수 출력, 경고 끄기를 main 맨 앞으로 옮겼다.
- tests: `tests/test_scripts.py`에 추가했다.
  - import 부수 효과 없음: Test_, Test_tsne, Plot_AUC_heatmap
  - Test_를 `--models-folder`와 `--models-idx`로 각각 실행해 AUC와 ROC 그림을 확인한다
  - `MODEL_DIR`만으로 Test_를 실행한다
  - Test_ `--debug --u-model-idx 0 --x-model-idx 0`: 차량 338만 평가하고, 관리도를 (u=0, x=0)으로 다시 계산한다(버그 11 해소 확인)
  - Test_tsne `--save`: t-SNE 그림을 확인한다
- 실데이터(기준선 태그 `se02`): `after_bugs`와 비교해 모두 bit 단위로 일치한다.
  - test: 304개 기록
  - test_u0: 513개 기록
  - tsne: 24개 기록
