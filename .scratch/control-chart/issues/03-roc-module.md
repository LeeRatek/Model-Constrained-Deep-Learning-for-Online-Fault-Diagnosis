# 03: ROC 평가 module과 Test_·Test_multi_models 전환

**What to build:** 차량별 최대 CI와 고장 여부로 AUC와 최적 임계값을 구하는 계산을 `mcdl/roc.py`의 `evaluate` 하나로 모은다. Test_와 Test_multi_models는 임계값 grid와 예측 행렬을 직접 만들지 않는다. 그림, CSV, 콘솔 출력은 지금과 같은 값으로 만들어진다.

**Blocked by:** 02 (진단 특징·관리도 module과 스크립트 5개 전환)

**Status:** resolved

- [x] `evaluate(max_ci, is_fault, reverse=False)`와 `RocResult`(fpr, tpr, auc, best, closest_to_01)를 spec대로 만든다. 임계값 grid, int8 예측 행렬, 기본 `argsort`, Youden J와 (0,1) 최근접점 계산을 안에 숨긴다.
- [x] Test_가 차량별 최대 CI를 모아 `evaluate`를 부른다.
- [x] Test_multi_models도 차량별 최대 CI를 모아 `evaluate`를 부른다. `--use-reverse-pos-neg`의 차량 순서와 판정 규칙은 지금과 같다. CSV 열과 값도 그대로다.
- [x] Function_.py에서 `compute_roc_auc_from_threshold_matrix`, `compute_optimal_thresholds_from_roc`, 원래부터 쓰이지 않던 `plot_auc_roc_curve_from_threshold_matrix`를 지운다. 이 변경으로 고아가 된 import도 지운다.
- [x] 01의 ROC fixture 테스트가 `evaluate`를 대상으로 bit 단위로 통과한다. 기본과 reverse 판정을 모두 확인한다.
- [x] 기준선 도구가 새 위치의 ROC 함수를 감싸게 한다. 실데이터 3종(test, tsne, multi)이 `before1`과 bit 단위로 일치해야 하고, 결과를 Comments에 남긴다.
- [x] README 6장을 갱신한다.

## Comments

### 2026-10-15 완료

**검증 결과**
- `python -m pytest`: 36개 통과, 약 7초.
  - ROC 테스트는 이제 `evaluate`를 대상으로 돈다. 고장 차량을 먼저 넘기는 reverse 순서(Test_multi_models와 같음)도 그대로 재현한다.
  - 정상이나 고장 한쪽만 있으면 거부하는지 확인하는 테스트를 추가했다.
- 기준선 비교 결과(`compare <target> before1 after4_03`): 네 대상 모두 모든 기록이 bit 단위로 일치했다. train은 티켓 범위 밖이지만 Function_.py가 바뀌어서 함께 확인했다.

| target | 기록 수 | 소요 시간 |
|---|---|---|
| train | 14 | 3.8 s |
| test | 304 | 41.5 s (`Test_.result` 포함) |
| tsne | 24 | 22.9 s |
| multi | 304 | 217.1 s (AUC_summary CSV 행 텍스트 포함, temp_cache 남지 않음) |

**구현 메모**
- ROC 계산 함수 두 개는 원본 그대로 옮겼다. 이름만 private(`_roc_from_threshold_matrix`, `_optimal_thresholds`)으로 바꿨다.
- 정렬 줄에 known-bugs 09 주석을 달았다.
- 임계값 grid는 module 상수다. 기존 스크립트와 같은 방식(`list`를 거쳐 만듦)으로 만든다.
- Test_multi_models
  - 테스트 데이터 캐시 해시에 `test_list` 순서가 들어가므로 `test_list`는 그대로 두었다.
  - 고장 여부는 리스트 identity(`vehicle_ids is fault_list`)로 판단한다.
  - `evaluate`는 `reverse`이면 정상을 양성으로 본다. 그래서 y_true가 예전 `enumerate` label과 같다. reverse일 때 고장이 0, 정상이 1이다.
- Test_에서 `test_idx`와 `total_test_vehicles`는 예측 행렬 인덱스로만 쓰였기 때문에 지웠다. Test_multi_models에서는 진행 출력에 쓰이므로 남겼다.

**정리한 것**
- Function_.py: 165줄. ROC 함수 2개와, 쓰이지 않던 wrapper(사용자 승인)
- Test_와 Test_multi_models: 임계값 grid, 예측 행렬, `y_true` 만드는 코드

**검증하지 못한 경로.** `--use-reverse-pos-neg`는 실데이터로 돌려 보지 않았다. 판정 규칙과 차량 순서는 합성 fixture 테스트(`test_roc[roc_reverse]`)로 확인했다.
