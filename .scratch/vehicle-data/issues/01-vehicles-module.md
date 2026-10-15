# 01: 차량 데이터 module과 평가 쪽 전환 (Test_, Test_tsne, 관리도 재계산)

**What to build:** 분할 ID 읽기, 차량 로딩·전처리, `.pt` 캐시 로딩을 `mcdl/vehicles.py`에 만든다. Test_, Test_tsne, 관리도 재계산이 이 module로 차량을 읽는다. 수치 결과는 그대로다.

**Blocked by:** None (can start immediately)

**Status:** resolved

- [x] `split_ids`, `load_vehicle`, `load_training_vehicles`, `load_test_vehicles`를 spec대로 만든다. 캐시 해시, 파일 이름, 저장 형식, 메시지는 지금의 Test_multi_models와 같다.
- [x] Test_와 Test_tsne가 `split_ids`와 `load_vehicle`을 쓴다. Test_tsne의 `collect_vehicle_points`는 설정 객체를 받는다.
- [x] `mcdl/chart_source.py`의 재계산이 `split_ids`(학습 차량 범위)와 `load_vehicle`을 쓴다.
- [x] 가짜 원본 데이터 helper를 tests 공용으로 옮기고, `tests/test_vehicles.py`를 추가한다. 다음을 확인한다.
  - 분할과 범위
  - 로딩 결과
  - 캐시 생성과 재사용
  - 해시의 `normalize_val` 조건
  - realdata: 해시 규칙으로 만든 이름과 실제 캐시 파일 이름이 같은지
- [x] 실데이터: test, test_u0, tsne가 `after_bugs`와 bit 단위로 일치해야 한다. 결과를 Comments에 남긴다.
- [x] README 6장(mcdl 소절, 의존 관계)을 갱신한다.

## Comments

**2026-10-15 완료**

- `mcdl/vehicles.py`: `split_ids`, `load_vehicle`, `load_training_vehicles`, `load_test_vehicles`. 캐시 해시·파일 이름·메시지는 Test_multi_models에서 그대로 옮겼다.
- Test_.py, Test_tsne.py: 분할은 `split_ids`, 차량은 `load_vehicle(args.source_data_dir, run.config, id)`로 읽는다. Test_tsne의 `collect_vehicle_points`는 `BATTERY_TYPE`·`PREPROCESSING`·`SKIP_CHARGE_READY` 대신 `config`를 받는다(`dim_dict`는 슬라이싱에 계속 쓰므로 그대로 받는다).
- `mcdl/chart_source.py`의 `_recompute`는 `split_ids(..., start=vehicle_start, end=vehicle_end)`와 `load_vehicle`을 쓴다. Function_ import가 없어졌다.
- tests: `tests/fake_vehicles.py`(가짜 원본·분할 helper)를 만들어 `test_chart_source.py`가 함께 쓰게 했다. `tests/test_vehicles.py`에 6개 + realdata 1개를 추가했다. pytest 54개 통과.
- 실데이터(기준선 태그 `vd01`), `after_bugs`와 비교:
  - test: 304개 기록 모두 bit 단위로 일치
  - test_u0: 513개 기록 모두 일치. 관리도 재계산 경로("Recomputing the control chart")를 실제로 탔다
  - tsne: 24개 기록 모두 일치
- README: 3장 구조, 4장 테스트 목록, 6장 의존 관계 표와 `mcdl/vehicles.py` 소절, 7.4 캐시 항목에서 해시 규칙으로 연결.
