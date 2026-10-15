# 01: 전처리·Run 규칙·모델 활성화·LSTM helper 옮기기

**What to build:** 원본 로딩·전처리 9개를 `mcdl/preprocessing.py`로 옮긴다. 그 밖에 `save_net_state`·`make_model_path_based_timestamp`는 `mcdl/run.py`로, `CustomSigmoidFunc`는 `Class_.py`로, LSTM helper 2개는 `Train_.py`로 옮긴다. 이 함수들을 쓰는 곳의 import를 바꾼다. 결과는 그대로다.

**Blocked by:** None (can start immediately)

**Status:** resolved

- [x] `mcdl/preprocessing.py`를 만든다. 9개 함수의 본문은 그대로 옮기고, 필요한 import만 둔다.
- [x] `mcdl/run.py`, `Class_.py`, `Train_.py`로 나머지 5개를 옮긴다. 옮긴 함수는 Function_.py에서 지운다.
- [x] import를 바꾼다.
  - 대상: `mcdl/vehicles.py`, `mcdl/run.py`, `Class_.py`, `Train_.py`, `data_filter.py`, tests
  - Train_은 Function_ star import를 지운다(남는 이름이 없다).
- [x] 실데이터
  - train, test_u0가 `after_bugs`와 bit 단위로 일치해야 한다.
  - data_filter는 정리 전후 실행의 분할 `.npy` 5개와 dataset_info가 같아야 한다.
- [x] pytest가 통과한다.

## Comments

**2026-10-15 완료**

- 옮긴 곳
  - `mcdl/preprocessing.py`(472줄): 9개 함수
  - `mcdl/run.py`: `save_net_state`, `make_model_path_based_timestamp`. `print_sim_config` 앞에 둔다
  - `Class_.py`: `CustomSigmoidFunc`
  - `Train_.py`: `prepare_training_data`, `series_to_supervised`
- 옮기는 방법: AST 줄 범위로 본문을 그대로 잘라 옮겼다(scratchpad의 `move_defs.py`). 줄 끝 형식은 파일마다 유지했다(mcdl·tests는 LF, 루트 스크립트는 CRLF).
- 바꾼 import 하나: Function_.py의 `from pyparsing import Any`(타입 힌트에만 쓰임, `from __future__ import annotations`라 실행되지 않음)를 새 module에서 `from typing import Any`로 바꿨다.
- 쓰는 곳 import
  - `mcdl/vehicles.py`, `mcdl/run.py`, `Class_.py`, tests 4개
  - Train_은 Function_ star import를 지우고 명시적 import로 바꿨다.
  - data_filter는 전처리 4개를 명시적으로 import한다(그림 함수는 02에서).
- Function_.py는 2,794줄에서 2,237줄이 됐다.
- pytest 82개 통과
- 실데이터(기준선 태그 `fs01`)
  - train: `after_bugs`와 14개 기록 일치
  - test_u0: `after_bugs`와 513개 기록 일치. 관리도 재계산이 `load_vehicle` → `preprocess_loaded_tensor`를 거친다
  - data_filter: 정리 전(커밋 `025ca33`과 같은 코드)과 정리 뒤로 `--train-ratio 0.02 --val-ratio 0.5`를 실행했다. 분할 `.npy` 5개(309/50/6/151/152)와 dataset_info(이름 줄 제외)가 같다. 비교 후 두 폴더는 지웠다.
