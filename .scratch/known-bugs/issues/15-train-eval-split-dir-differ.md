# 15: 학습과 평가가 분할 폴더를 다른 규칙으로 정한다

**Status:** resolved

**Blocked by:** None (can start immediately)

**현상**
- Train_.py는 `--dataset-name`이 있으면 무조건 `./datasets/<이름>`을 읽었다.
- 평가(`Run` → `RunConfig.split_dir`, `mcdl/run.py`의 `_split_dir`)는 이름이 `default`이면 루트 분할을 읽었고,
  기록된 폴더가 없으면 경고만 출력하고 루트 분할로 넘어갔다.

| `dataset_name` | 학습 | 평가 |
|---|---|---|
| 없음 / `None` | 루트 | 루트 |
| `default` | `datasets/default` | 루트 |
| 폴더가 없는 이름 | 파일 없음 오류 | 경고 후 루트 |

- `default`는 data_filter.py의 기본 분할 이름이다. 기본값으로 분할을 만들고 `--dataset-name default`로 학습하면
  평가가 다른 차량 목록(학습 차량이 섞일 수 있는 루트 분할)으로 이루어진다.

**영향:** 기존 Run(`models/2602*`)은 sim_config.txt에 `dataset_name`이 없어 학습·평가 모두 루트 분할이다. 영향 없음.

**출처:** 2026-10-16 코드 정리 중 발견.

## Comments

### 2026-10-16 해소

- 2026-10-16 사용자 결정: 엄격한 공통 규칙. 이름이 없거나 `None`이면 루트, 그 밖의 이름(`default` 포함)은 `./datasets/<이름>`,
  폴더가 없으면 학습·평가 모두 `FileNotFoundError`(폴더 이름과 sim_config.txt 경로 포함)로 멈춘다.
- Train_.py는 자체 경로 계산을 지우고 평가와 같은 `run_config.split_dir`를 쓴다.
- 검증:
  - `tests/test_run.py::test_split_dir_follows_dataset_name`, `::test_missing_split_folder_is_an_error`
  - `tests/test_scripts.py::test_train_and_evaluation_read_the_same_named_split`(수정 전 코드에서 실패),
    `::test_train_stops_when_the_named_split_is_missing`
  - 실제 데이터 train·train_full·test 기록(fx01)이 op01과 bit 단위로 같다(이름 없이 학습하는 경로는 그대로).
