# 01: Train_, data_filter를 `main(argv)`로

**What to build:** Train_과 data_filter를 `main(argv=None)` 구조로 바꾸고, 디버거 감지를 `--debug`로 대신한다. Train_ smoke 테스트를 둔다. 학습 결과와 분할 결과는 그대로다.

**Blocked by:** None (can start immediately)

**Status:** resolved

- [x] Train_: 본문을 `main(argv)`로 옮긴다. Dataset 클래스 6개와 `set_seed`는 모듈 최상위에 둔다. import 시점 부수 효과를 main 안으로 옮긴다.
- [x] Train_: `--debug`가 지금의 디버거 분기를 대신한다(차량 1대씩, `use_dx = False`, sim_config 생략).
- [x] data_filter: 본문을 `main(argv)`로 옮기고, `--debug`가 지금의 디버거 분기를 대신한다.
- [x] `tests/test_scripts.py`: 가짜 데이터로 `Train_.main`을 실행해 Run 폴더를 확인한다. 스크립트를 import만 하면 아무것도 실행되지 않는다는 것도 확인한다.
- [x] 실데이터
  - train이 `after_bugs`와 bit 단위로 일치해야 한다.
  - data_filter 전후 실행의 분할 `.npy` 5개와 `dataset_info_QAS.txt`가 같아야 한다.
  - 결과를 Comments에 남긴다.

## Comments

**2026-10-15 완료**

- Train_
  - parser 정의, `set_seed`, `device`, Dataset 클래스 6개는 모듈 최상위에 둔다. 나머지 본문은 순서 그대로 `main(argv=None)` 안으로 옮겼다.
  - `KMP_DUPLICATE_LIB_OK` 설정, GPU 개수 출력, 경고 끄기는 main 맨 앞으로 옮겼다.
- data_filter: 같은 방식이다. `MODE`는 모듈 최상위에 둔다. `exit()`와 `locals()` 검사는 함수 안에서도 같은 의미라 그대로 두었다.
- `--debug`가 디버거 감지(`sys.gettrace()`·debugpy)를 대신한다. 동작은 예전 디버거 분기와 같다.
- `write_sim_config`에 `exclude`를 추가했다(`print_sim_config`로 넘김). Train_은 `exclude=("debug",)`로 불러, `sim_config.txt` 형식이 바뀌지 않는다.
- tests: `tests/test_scripts.py`를 추가했다. 4개 통과, 약 7초.
  - 두 스크립트 모두 import만 하면 argv를 파싱하지 않고 아무것도 출력하지 않는다.
  - `Train_.main`: 가짜 데이터(학습 2대, 검증 1대, 1 epoch)로 Run을 만든다. sim_config(`debug` 줄 없음), 체크포인트 0/-1, 검증 loss, 관리도, 학습 진단 특징 행 수(2대분)를 확인한다. 이 테스트로 vehicle-data 03에서 실데이터로 돌리지 못한 검증 차량 로딩 경로도 실행된다.
  - `Train_.main --debug`: 학습 차량 1대분 행 수이고 sim_config가 없다.
- 실데이터
  - train(기준선 태그 `se01`): `after_bugs`와 14개 기록이 모두 bit 단위로 일치한다. 새 Run의 sim_config에 `debug` 줄이 없다.
  - data_filter: `--dataset-name _verify_df_{before,after} --train-ratio 0.02 --val-ratio 0.5`를 리팩토링 전후로 한 번씩 실행했다(각 약 57초).
    - 분할 `.npy` 5개가 dtype·shape·값까지 같다(normal 309, fault 50, train 6, validate 151, test 152).
    - `dataset_info_QAS.txt`는 이름 줄을 빼면 같다.
    - 비교 후 두 폴더는 지웠다.
