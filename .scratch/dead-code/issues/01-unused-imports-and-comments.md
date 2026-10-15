# 01: 쓰이지 않는 import·변수와 주석 코드

**What to build:** 동작과 무관한 것을 지운다. 대상은 쓰이지 않는 import, 읽히지 않는 변수, 덮어써지는 대입, 의미 없는 주석, 예전 구현·대안을 보존한 주석 코드다. 결과는 bit 단위로 같다.

**Blocked by:** None (can start immediately)

**Status:** resolved

- [x] 스크립트 6개, Class_, mcdl에서 쓰이지 않는 import를 지운다.
- [x] 읽히지 않는 변수, 덮어써지는 대입, 쓰이지 않는 지역 import를 지운다. 오탐(dataclass 필드, Enum 멤버, 튜플 언패킹 일부)은 남긴다.
- [x] 예전 구현·대안을 보존한 주석 코드와 의미 없는 주석을 지운다. 설명용 주석은 남긴다.
- [x] 정적 검사: 정의되지 않은 이름이 0건이어야 한다.
- [x] pytest, 실데이터 6개 대상이 bit 단위로 일치해야 한다. 결과를 Comments에 남긴다.

## Comments

**2026-10-16 완료**

- 8개 파일에서 133줄을 지웠다.
  - 대상 파일: Train_, Test_, Test_multi_models, Test_tsne, data_filter, Class_, `mcdl/plotting.py`, `mcdl/preprocessing.py`
- import
  - 스크립트의 matplotlib(`mpl`, `mtick`, `matplotlib`, 안 쓰는 `plt`), 안 쓰는 numpy·pandas, sklearn `train_test_split`, `F`, `pathlib.Path`
  - Class_의 sklearn 4개, `torch.optim`, `scipy.io`, torchvision `transforms`, `DataLoader`, `F`, 중복 `nn`, `plot_distribution_with_stats`
- 변수·대입
  - Train_ `FIRST_LOAD`(2곳), `total_val_samples`
  - Test_ 그림 분기의 `title`
  - Test_multi: `combinations_to_run`, `original_loop_code_marker`, 곧바로 덮어써지는 첫 `combinations_phase2`, 다시 쓰이지 않는 두 번째 `last_x_idx`, 지역 `import tempfile`
  - Test_tsne: 쓰지 않는 반환값 `fig, axes`
  - data_filter: `device`
- 주석 코드
  - Train_: `set_seed(42)`, `use_dx_in_forward = False`, `train_list = [0]` 등, 텐서 concat 예전 구현 2블록
  - Test_·Test_tsne: `VIN_data`, `SKIP_CHARGE_READY = False`, `fault_list = np.arange(…)`, `thresholds = [20, 30]`, `--results-dir`, `cnt_max = 6`, `normal_but_faulty` 대안
  - Test_multi: 캐시 정리 블록
  - data_filter: `vehicle_ids`와 `plot_testX_timeseries` 대안 2줄
  - Class_: `analyze_outputs` 호출 줄
  - plotting: `yscale="log"` 2줄, `set_xlim` 3줄
  - preprocessing: DTI 디버그 `print` 2블록
  - 모든 파일: `# from sklearn.datasets import load_boston`
- 남긴 것: 설명용 주석, data_filter의 `(vin1_start, vin2_start, vin3_start)`(튜플 일부가 PLOT_MODE에서 쓰인다). LSTM 블록 안의 `lr_decay_freq`·`hidden_state`·`print(os.getcwd())`는 02에서 블록째 지운다.
- 정적 검사: 정의되지 않은 이름 0건, 쓰이지 않는 import 0건
- pytest 82개 통과
- 실데이터(기준선 태그 `dc01`): 6개 대상이 모두 bit 단위로 일치한다.
  - train 14, train_full 17(`tl00a` 대비), test 304, test_u0 513, tsne 24, multi 304개 기록
