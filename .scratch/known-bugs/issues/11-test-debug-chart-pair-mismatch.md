# 11: Test_ 디버그 모드에서 관리도와 모델 쌍이 다르다

**Status:** resolved

**Blocked by:** None (can start immediately)

**현상**
- Test_.py는 디버그 모드(`sys.gettrace()` 또는 debugpy)이면 모델 쌍을 `(2, 1)`로 바꾼다(`u_model_idx, x_model_idx = 2, 1`).
- 그런데 관리도는 `chart_for_pair(run, args.u_model_idx, args.x_model_idx, …)`로 CLI 인자의 쌍(기본 -1/-1)에서 가져온다.
- 결과적으로 디버그 모드에서는 (2, 1) 모델의 테스트 특징을 (-1, -1) 관리도로 평가한다.

**영향:** 디버거에서 실행할 때만 생긴다. 일반 실행 결과에는 영향이 없다. 디버그 모드에서는 차량도 338로 고정되므로 AUC는 원래 의미가 없다.

**고칠 때 주의**
- `chart_for_pair`에 `u_model_idx, x_model_idx`를 넘기면 된다. 다만 (2, 1)은 최종 쌍이 아니다. 그래서 같은 조건의 temp_cache가 없으면 디버그 실행 때마다 관리도를 다시 계산한다(약 75초).
- 디버그 모드에서 모델 쌍을 바꾸는 것 자체가 필요한지도 함께 정해야 한다.

**출처:** 2026-10-15 4번 후보(테스트 차량 평가 루프) 탐색.

## Comments

### 2026-10-15 해소 (script-entry 02)

- 디버거 감지를 명시적 `--debug` 옵션으로 바꾸면서(script-entry grilling Q2·Q3), Test_의 모델 쌍 (2, 1) 강제를 없앴다.
- 모델 쌍과 관리도 쌍은 둘 다 `--u-model-idx`·`--x-model-idx`에서 온다. `chart_for_pair(run, u_model_idx, x_model_idx, …)`로 부른다.
- 검증: `tests/test_scripts.py::test_test_debug_fixes_vehicle_and_keeps_chart_pair`. `--debug --u-model-idx 0 --x-model-idx 0`이면 관리도를 (u=0, x=0)으로 다시 계산한다.
- 일반 실행(디버그가 아닐 때)의 결과는 그대로다. test·test_u0 기준선 `se02`가 `after_bugs`와 일치했다.
