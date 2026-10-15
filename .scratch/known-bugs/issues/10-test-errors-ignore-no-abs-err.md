# 10: 테스트 쪽 재구성 오차가 `no_abs_err` 설정을 따르지 않는다

**Status:** resolved

**Blocked by:** None (can start immediately)

**현상**
- 학습 진단 특징은 설정의 `no_abs_err`를 따른다. 거짓이면 절댓값 오차를, 참이면 부호 있는 오차를 쓴다. Train_, Test_multi_models의 temp_cache, 관리도 재계산이 모두 그렇다.
- 테스트 차량의 재구성 오차는 설정과 상관없이 항상 절댓값이다. Test_, Test_multi_models, Test_tsne가 모두 그렇다.
- 그래서 `--no-abs-err`로 학습한 Run을 평가하면 문제가 생긴다. 관리도는 부호 있는 오차의 특징으로 맞추고, 테스트 특징은 절댓값 오차로 계산한다.

**영향:** 지금 `models/`의 Run 10개는 모두 `no_abs_err : False`라 영향이 없다. `--no-abs-err`로 새로 학습한 Run의 CI·AUC만 틀린다.

**고칠 때 주의**
- 세 스크립트는 `vehicle_features(..., abs_err=True)`를 명시해서 부른다(vehicle-features 02 이후). 고칠 곳은 이 인자를 `not run.config.no_abs_err`로 바꾸는 것이다.
- 고치면 `no_abs_err`가 참인 Run의 평가 결과가 바뀐다. 기존 Run에는 해당이 없다.

**출처:** 2026-10-15 4번 후보(테스트 차량 평가 루프) 탐색.

## Comments

### 2026-10-15 수정

**바꾼 것**
- 세 스크립트가 테스트 차량 진단 특징을 설정의 abs 여부로 계산한다. 학습 진단 특징과 같은 설정이다.
  - Test_: `abs_err=not run.config.no_abs_err`
  - Test_multi_models: `abs_err=use_abs_err`. 같은 파일의 학습 특징이 이미 쓰던 변수다
  - Test_tsne: `abs_err=not config.no_abs_err`

**검증**
- 기존 Run(`no_abs_err : False`)의 결과는 그대로다. 기준선 태그 `fix10`과 `after_bugs`를 비교했다.
  - test: 304개 기록 일치
  - tsne: 24개 기록 일치
  - multi: 304개 기록 일치
- `--no-abs-err` smoke 실행
  - 학습: `Train_.py --no-abs-err`, 차량 1대, 1 epoch. Run은 `models/_verify/bug10/261015_153944`이다
  - 평가: Test_ -1/-1로 끝까지 실행된다
  - 같은 Run, 같은 테스트 차량 100대에서 결과는 다음과 같다.

    | 테스트 쪽 오차 | AUC | 최적 임계값 |
    |---|---|---|
    | 부호 있는 오차 (수정 후, 관리도와 같은 오차) | 0.8376 | 64 |
    | 절댓값 오차 (수정 전) | 0.8634 | 46.2 |

  - 이 Run은 차량 1대·1 epoch로 학습한 smoke용이라, 어느 쪽 AUC가 일반적으로 더 높은지는 말할 수 없다. 이 결과가 보여 주는 것은 수정 전에는 관리도와 다른 오차로 평가하고 있었고, 그 때문에 결과가 달라졌다는 점이다.
- 참고: smoke 출력을 파일로 리디렉션하자 Test_가 마지막 `print`의 `≈` 문자에서 `UnicodeEncodeError`(cp949)로 멈췄다. 콘솔 인코딩 문제이고 이 버그와는 무관하다. `PYTHONIOENCODING=utf-8`로 실행하면 끝까지 돈다. 기준선 도구는 stdout을 UTF-8로 바꾸므로 영향이 없다.
