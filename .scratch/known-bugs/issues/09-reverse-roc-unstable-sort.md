# 09: reverse 모드 ROC에서 FPR 동률 정렬이 stable하지 않다

**Status:** resolved

**Blocked by:** None (can start immediately)

**현상**
- `compute_roc_auc_from_threshold_matrix`는 FPR을 기본 `np.argsort`로 정렬하는데, 이 정렬은 stable하지 않다(커밋 f65281c에서 도입).
- `--use-reverse-pos-neg`에서는 FPR이 같은 점들 안에서 순서가 정렬 구현에 따라 섞인다. 그 결과 AUC와 Youden 임계값이 달라질 수 있다.
- 리뷰 agent가 합성 데이터로 100회 실측한 결과는 다음과 같다(numpy 2.3.5).
  - AUC: 99회 차이, 최대 3.3e-3
  - Youden 임계값: 25회 차이
- 기본 방향은 입력이 이미 정렬되어 있어 영향이 없다.

**영향:** reverse 모드 결과를 재현할 수 있는지가 numpy 버전과 플랫폼에 달려 있다.

**고칠 때 주의**
- stable 정렬이나 lexsort로 바꾸면 reverse 모드 AUC가 바뀐다.
- FPR이 같은 점들 안에서 TPR을 어떤 순서로 둘지도 함께 정해야 한다. stable 정렬을 써도 rank 기반 AUC와 최대 5.7e-3 차이가 난다.
- `argmax`가 동률일 때 첫 번째 값을 고르는 방식도 이 정렬 순서에 의존한다.

**출처:** 2026-10-14 아키텍처 리뷰.

## Comments

### 2026-10-15 위치 변경 (control-chart 03)

- ROC 계산은 이제 `mcdl/roc.py`에 있다. 정렬은 `_roc_from_threshold_matrix` 안에서 일어나며, 해당 줄에 이 티켓 번호를 주석으로 달았다. 동작은 바뀌지 않았다.
- `tests/test_diagnosis.py`의 `test_roc[roc_reverse]`가 지금의 reverse 결과를 fixture로 고정하고 있다. 이 버그를 고치면 fixture를 다시 만들어야 한다.

### 2026-10-15 수정

**바꾼 것**
- `mcdl/roc.py`의 `_roc_from_threshold_matrix`에서 정렬을 `np.argsort(fpr)`에서 `np.lexsort((tpr, fpr))`로 바꿨다(grilling Q2).
- 이 정렬은 FPR로 먼저, 같으면 TPR로 오름차순 정렬하는 stable 정렬이다. 그래서 ROC 곡선이 항상 단조 계단 모양이 된다.

**영향(합성 fixture 기준)**
- 기본 판정(최대 CI > θ)은 입력이 이미 이 순서라 그대로다. fixture 테스트가 고치기 전 값 그대로 통과했다.
- reverse 판정:
  - AUC가 0.5에서 0.5556으로 바뀌었다. 순위 기반으로 직접 계산한 AUC 0.5556과 같다. 고치기 전에는 TPR이 단조가 아니어서 면적이 틀렸다.
  - Youden 최적 임계값이 998.0에서 999.9로 바뀌었다. 같은 (FPR, TPR) 점들 사이에서 어느 점을 고르느냐의 차이이며, 이제는 결정적이다.
  - `tests/fixtures/diagnosis_golden.npz`의 `roc_reverse_*` 항목만 새 동작으로 다시 만들었다.
- 기존 Run의 `--use-reverse-pos-neg` 결과(있다면)는 다시 계산하면 바뀔 수 있다. 기존 Run에서 reverse 모드를 썼는지는 CSV만 보고 알 수 없다.
