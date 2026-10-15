# 03: Test_tsne가 battery_type과 normalize_dx를 CLI에서 읽는다

**Status:** resolved

**Blocked by:** None (can start immediately)

**현상**
- Test_tsne는 `battery_type`(`--battery-type`, 기본 QAS)과 `normalize_dx`(`--normalize-dx`)를 sim_config가 아니라 CLI 옵션에서 읽는다.
- `--normalize-val`은 정의만 되어 있고 쓰이지 않는다.

**영향**
- DTI Run을 옵션 없이 돌리면 QAS 열 구성으로 처리된다.
- `normalize_dx`로 학습한 Run에서 옵션을 빠뜨리면 학습과 다른 전처리로 평가된다.

**고칠 때 주의**
- sim_config 값을 쓰도록 바꾸면 CLI 옵션의 의미가 달라진다. README 7.5를 함께 갱신해야 한다.
- run-module 02 이후에는 호출부에 이 티켓 번호가 주석으로 달려 있다.

**출처:** 2026-10-14 아키텍처 리뷰.

## Comments

### 2026-10-14 run-module 02 완료

- Test_tsne의 데이터 경로·열 구성·전처리는 지금도 CLI 값(`--battery-type`, `--normalize-dx`)을 따른다. 호출부에 이 티켓 번호를 주석으로 남겼다.
- 달라진 점이 하나 있다. 모델은 이제 Run module이 sim_config의 `battery_type`으로 만든다. 예전에는 CLI 값으로 만들었다.
  - 두 값이 같으면(일반적인 경우) 결과는 같다. 260224_090359 기준선과 bit 단위로 같다.
  - 두 값이 다르면 예전이나 지금이나 실패한다. 예전에는 `load_state_dict`에서 크기가 맞지 않았고, 지금은 forward에서 입력 열 수가 맞지 않는다.

### 2026-10-15 수정

**바꾼 것**
- Test_tsne는 이제 `battery_type`, `normalize_dx`, `normalize_val`을 모두 Run(sim_config)에서 읽는다.
- CLI 옵션 `--battery-type`, `--normalize-dx`, `--normalize-val`은 지웠다(grilling Q3). README 7.5의 옵션 표도 고쳤다. README 7장의 예시 명령은 이 옵션들을 쓰지 않으므로 그대로 동작한다.

**검증**
- QAS 기준 Run에서는 `before1` 대비 tsne가 bit 단위로 일치했다(`fix03`).
- DTI Run 260211_163108의 사본(`models/_verify/260211_163108`)으로 `Test_tsne.py --u-model-idx -1 --x-model-idx -1 --fault-idx 77 --save`를 옵션 없이 실행했다. 셀 85개 기준(입력 170차원)으로 끝까지 실행되어 `tSNE_77.png`를 만들었다.
