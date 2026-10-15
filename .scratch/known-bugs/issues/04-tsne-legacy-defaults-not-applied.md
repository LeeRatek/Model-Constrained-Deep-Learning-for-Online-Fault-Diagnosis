# 04: Test_tsne에서 레거시 기본값이 적용되지 않는다

**Status:** resolved

**Blocked by:** None (can start immediately)

**현상**
- Test_tsne는 sim_config를 키 목록 지정 + `strict=False`로 읽는다. 이 경우 없는 키도 `None` 값의 속성으로 채워진다.
- 그래서 `hasattr`로 판단하는 하위 호환 기본값 분기가 항상 참이 되어 기본값이 적용되지 않는다.
- 결과적으로 오래된 config에서는 scale·shift가 `None`이 되고, `CustomSigmoidFunc(None, None)`이 만들어진다.

**영향:** `ae_*_scale`·`ae_*_shift` 키가 없는 오래된 Run에서 Test_tsne가 실패한다.

**예정**
- run-module 02에서 Run module이 레거시 기본값을 채우므로, 이 경로는 살아난다. spec에서 "지금 예외가 나는 경로는 살아나도 된다"고 정했기 때문이다.
- 02를 마치면 Comments에 기록하고, 이 티켓을 닫을지 정한다.

**출처:** 2026-10-14 아키텍처 리뷰.

## Comments

### 2026-10-14 run-module 02 완료

- 이제 Test_tsne는 Run module로 설정을 읽는다. sim_config에 키가 없으면 Run module이 레거시 기본값(1.8 / 2.5 / 1.0 / 0, use_dx True, add_one_output_layer False)을 채운다.
- 따라서 이 티켓의 실패 경로는 더 이상 일어나지 않는다. spec에서 정한 대로 "지금 예외가 나는 경로가 살아나는 것"에 해당한다.
- 키가 모두 있는 Run(260224_090359)의 결과는 기준선과 bit 단위로 같다.
- 오래된 config를 쓴 Run으로 실제 실행해 보지는 않았다. 레거시 기본값은 `tests/test_run.py`의 `test_missing_keys_fall_back_to_legacy_defaults`로 확인했다.
- 이 티켓을 닫을지는 triage에서 결정한다.
- 같은 이유로 Test_도 살아나는 경로가 하나 있다. 예전에는 `normalize_dx` 키가 없는 config에서 AttributeError가 났지만, 이제는 기본값 False로 동작한다.

### 2026-10-15 닫음

triage(버그 수정 grilling Q8)에서 run-module 02로 해소된 것으로 보고 닫았다. 추가 코드 변경은 없다.
