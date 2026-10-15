# Spec: 진입 스크립트를 `main(argv)`로

**Status:** resolved

출처: 2026-10-15 2차 아키텍처 리뷰의 5번 후보("진입 스크립트를 main(argv) adapter로")와 같은 날 grilling(Q1–Q8). 용어는 `GLOSSARY.md`를 따른다.

## Problem Statement

진입 스크립트는 import만 해도 argv를 파싱하고 본문 전체를 실행한다. 그래서 스크립트 단위로 테스트할 수 없다. 리팩토링 검증은 실데이터 기준선 도구(runpy)에만 의존한다.

Train_, Test_, data_filter는 디버거가 붙었는지 감지해 동작을 바꾼다. 바뀌는 것은 차량 수, 모델 쌍, 분할 비율이다. 감지 코드는 세 곳에 복사되어 있다. 디버거로 실행하면 일반 실행과 다른 결과가 나오고, pytest를 디버거로 돌려도 마찬가지다. Test_의 디버그 분기는 모델 쌍만 바꾸고 관리도 쌍은 바꾸지 않는다(known-bugs 11).

Run을 지정하는 옵션은 Test_만 `--models-folder`이고, 나머지는 `--models-idx`다. Test_의 `--models-dir`만 `MODEL_DIR`을 따르지 않는다.

## Solution

다섯 스크립트를 `main(argv=None)`과 `if __name__ == "__main__": main()` 구조로 바꾼다. 대상은 Train_, Test_, Test_multi_models, Test_tsne, data_filter다. Plot_AUC_heatmap은 파싱만 main 안으로 옮긴다.
- 디버거 감지는 명시적 `--debug` 옵션으로 바꾼다.
- Test_는 `--models-idx` 별칭과 `MODEL_DIR`을 받는다.
- 가짜 원본 데이터로 학습부터 평가까지 잇는 smoke 테스트를 둔다.
- 일반 CLI 실행의 수치 결과는 바뀌지 않는다.

## User Stories

1. 연구자로서, README 7장의 명령이 지금처럼 동작하고 결과가 bit 단위로 같길 원한다.
2. 연구자로서, 디버거로 실행해도 일반 실행과 같은 동작을 원한다. 빠른 실행이 필요하면 `--debug`를 직접 준다.
3. 연구자로서, Test_ 디버그 실행에서 모델과 관리도가 같은 쌍이길 원한다.
4. 연구자로서, 모든 평가 스크립트에서 같은 옵션 이름(`--models-idx`)과 같은 환경 변수(`MODEL_DIR`)를 쓰고 싶다. 기존 `--models-folder`도 계속 동작해야 한다.
5. 개발자로서, 스크립트를 import해도 아무 일도 일어나지 않길 원한다.
6. 개발자로서, 학습 → 평가 전체 경로를 원본 데이터 없이 몇 초 만에 시험하고 싶다.
7. 리뷰어로서, 각 티켓을 마칠 때 기준선 비교 결과가 티켓에 남길 원한다.

## Implementation Decisions

### 구조 (Q1, Q6)

- 각 스크립트의 본문을 순서 그대로 `main(argv=None)` 안으로 옮긴다. 파싱은 `parser.parse_args(argv)`로 한다.
- 본문 안에 정의된 class와 함수는 모듈 최상위로 뺀다.
  - Train_: Dataset 클래스 6개
  - Test_tsne: `add_vehicle_points`, `collect_vehicle_points`
  - data_filter: `MODE`
- Test_multi_models의 `process_combination`은 main 안의 closure로 두고 `global` 문을 지운다. 바꾸는 것은 컨테이너 내용뿐이고 이름을 다시 할당하지 않는다.
- import 시점 부수 효과는 main 안으로 옮긴다. `KMP_DUPLICATE_LIB_OK` 설정, `warnings.filterwarnings("ignore")`, GPU 개수 출력이 여기에 해당한다.
- parser는 스크립트마다 둔다. 공용 CLI module은 만들지 않는다(Q5).
- 쓰이지 않는 옵션은 그대로 둔다. Test_·Test_multi의 `--sigma-levels`, Test_multi의 `--x-start`·`--x-tick-step`, Train_의 `--vin1/2/3-start`가 해당한다.

### `--debug` (Q2, Q3)

- `sys.gettrace()`·`debugpy` 감지를 지우고 `--debug`(store_true)로 대신한다.
- `--debug`일 때의 동작은 지금 디버거 실행과 같다. 단, Test_의 모델 쌍 강제는 예외다.
  - Train_: 학습·검증 차량 1대씩, `use_dx = False`, `sim_config.txt`를 쓰지 않음
  - Test_: 모든 반복에서 차량 338만 평가한다. 모델 쌍 (2, 1) 강제는 없앤다. 모델 쌍은 `--u-model-idx`·`--x-model-idx`로 정하고, 관리도도 같은 쌍을 쓴다(known-bugs 11 해소).
  - data_filter: `train_ratio = 0.02`, `val_ratio = 0.5`, `dataset_name = "2"`

### CLI 이름 (Q4)

- Test_: `--models-folder`와 `--models-idx`를 같은 dest(`models_folder`)로 받는다.
- Test_: `--models-dir` 기본값이 `MODEL_DIR` 환경 변수를 따른다(없으면 `./models`).
- README 예시는 `--models-idx`로 통일한다.

## Testing Decisions

**smoke 테스트 (`tests/test_scripts.py`, Q7)**
- `tmp_path`에 가짜 원본 데이터(`tests/fake_vehicles.py`)와 루트 분할 `.npy`를 만들고 cwd를 옮긴다.
- 다음 순서로 실행한다.
  1. `Train_.main`: 학습 차량 2대, 1 epoch, 검증 생략
  2. 그 Run으로 `Test_.main`
  3. `Test_tsne.main`
  4. `Test_multi_models.main`
- 확인할 것
  - Run 폴더(`sim_config.txt`, 체크포인트, `pca_arrays.npz`)
  - Test_의 ROC 그림
  - Test_tsne의 저장 그림
  - Test_multi의 `AUC_summary` CSV
- matplotlib은 Agg backend로 돌린다.
- data_filter는 `./data`와 차량 ID 범위가 하드코딩되어 있어 smoke 테스트에서 뺀다.

**import 부수 효과:** 각 스크립트를 import만 했을 때 argv를 파싱하지 않고 출력도 없다.

**실데이터 기준선(`after_bugs`), 티켓별**
- 01: train. data_filter는 리팩토링 전후로 `--dataset-name _verify_df_* --train-ratio 0.02 --val-ratio 0.5`를 실행해 분할 `.npy` 5개와 `dataset_info`를 비교한다.
- 02: test, test_u0, tsne
- 03: multi

## Out of Scope

- 본문 로직을 module로 더 쪼개는 것
- 쓰이지 않는 옵션 삭제
- check_data_stats.py
- 공용 CLI module
