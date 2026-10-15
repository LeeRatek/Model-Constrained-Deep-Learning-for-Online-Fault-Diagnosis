# 03: 지우기와 import 정리

**What to build:** 쓰이지 않는 코드와 Function_.py, `monitering.py`를 지우고, 스크립트의 star import를 명시적 import로 바꾼다. 결과는 그대로다.

**Blocked by:** 02

**Status:** resolved

- [x] Function_.py에 남은 27개(미사용 18개, 주석 코드 전용 9개)와 Function_.py 파일을 지운다.
- [x] `monitering.py`, Class_의 `Dataset`(5-tensor)·`RMSELoss`를 지운다.
- [x] Test_, Test_tsne, Train_, data_filter에서, 지운 함수나 이미 없는 변수를 참조하는 주석 블록을 지운다.
- [x] 모든 스크립트의 `from Function_ import *`, `from Class_ import *`, `from monitering import *`를 명시적 import로 바꾼다.
- [x] README를 갱신한다(3장 구조, 6장 의존 관계·module 소절, Function_ 표).
- [x] 실데이터: train, train_full, test, test_u0, tsne, multi가 모두 bit 단위로 일치해야 한다. 비교 기준은 train_full은 `tl00a`, 나머지는 `after_bugs`다. pytest가 통과한다.

## Comments

**2026-10-15 완료**

- 지운 것
  - Function_.py: 남은 정의가 지울 27개와 정확히 같은지 assert로 확인한 뒤 파일째 지웠다.
  - `monitering.py`
  - Class_의 `Dataset`(5-tensor)·`RMSELoss`
- star import: 모든 스크립트의 `from Function_ import *`, `from Class_ import *`, `from monitering import *`를 지웠다. Train_은 `from Class_ import LSTM, MyDataset`을 쓴다.
- 주석 블록: 지운 함수나 없는 변수를 참조하는 것만 지웠다.
  - Test_: `load_vehicle_ids_used_for_training`, `ERRORU`·`recon_imtest`·`df_data2`·`t2_contrib` 분석, `chi_square_dist_components`·`diagnosis_thresholds`, `plot_contrib_percent_stacked` 블록
  - Test_tsne: `recon_imtest` t-SNE 대안, `X_list`/`y_list`
  - Train_: `v_combined_tensor` 줄, 끝의 `plot_array`/`T2_array` 블록
  - data_filter: `calculate_volt_modepi` 블록, 끝의 `dim_x` 슬라이싱 블록
  - 지금 있는 변수만 쓰는 예전 구현 주석은 남겼다(예: Train_의 `combined_tensor` concat, Test_tsne의 `normal_but_faulty`).
- 적용 중 한 번 멈췄다. Test_ 그림 분기에 `# plot_contrib_percent_stacked(` 블록이 두 개 연달아 있어서, 표식이 한 번만 나온다는 assert에 걸렸다. 멈춘 지점까지 적용된 것을 확인하고, 두 블록을 함께 자르도록 남은 단계만 다시 적용했다.
- 정적 검사(AST): 스크립트·mcdl·tests 전체에서 정의되지 않은 이름과 남은 star import가 모두 0건이다. smoke 테스트가 지나가지 않는 분기(LSTM, Test_ 그림 모드 등)까지 포함한다.
- 기준선 도구 docstring을 지금 구조에 맞게 고쳤다. `WRAP_TARGETS`의 예전 위치(Function_)는 옛 코드로 기준선을 뜰 때를 위해 남긴다.
- README: 3장 구조, 5장 `safe_load` 위치, 6장(의존 관계 표를 실제 import에서 다시 뽑음, Run 표에 Run 폴더·체크포인트 저장, Class_ 표, `mcdl/preprocessing.py`·`mcdl/plotting.py` 소절, monitering 소절 삭제)
- pytest 82개 통과
- 실데이터(기준선 태그 `fs03`): 모두 bit 단위로 일치한다.

  | 대상 | 비교 기준 | 기록 |
  |---|---|---|
  | train | `after_bugs` | 14개 |
  | train_full | `tl00a` | 17개 (45.0초) |
  | test | `after_bugs` | 304개 |
  | test_u0 | `after_bugs` | 513개 |
  | tsne | `after_bugs` | 24개 |
  | multi | `after_bugs` | 304개 |
