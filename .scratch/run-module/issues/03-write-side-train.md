# 03: Run module 쓰기 쪽과 Train_ 전환

**What to build:** Train_이 Run module을 써서 Run을 만든다. 설정은 `RunConfig.from_mapping`으로 만들고, 모델 쌍은 `build_model_pair`로, 체크포인트 파일 이름은 `checkpoint_filename`으로, sim_config.txt는 `write_sim_config`로 정한다. 이로써 학습과 평가가 같은 규칙으로 U-model·X-model을 만든다. sim_config 형식은 쓰는 쪽과 읽는 쪽이 한 module 안에 있게 된다. 학습 결과와 sim_config 텍스트는 지금과 같다.

**Blocked by:** 02 (Run module 읽기 쪽과 평가 스크립트 3개 전환)

**Status:** resolved

- [x] Run module에 `checkpoint_filename`과 `write_sim_config`를 추가한다. `print_sim_config`와 그 서식 helper를 Run module로 옮긴다. 평가 스크립트가 설정을 콘솔에 찍는 호출도 옮긴 함수를 쓴다. 이번 변경으로 쓰이지 않게 된 원본은 지운다.
- [x] Train_이 args로 `RunConfig.from_mapping`을 거쳐 설정을 만들고, `build_model_pair`로 모델을 만든다. 생성 순서(U 다음 X)와 그 앞뒤의 RNG 호출 순서는 지금과 같다.
- [x] Train_의 체크포인트 저장 경로(epoch 파일과 최종 파일)는 `checkpoint_filename`을 쓴다.
- [x] 테스트를 추가한다.
  - write → `Run`으로 read → 같은 `RunConfig` (round-trip)
  - 고정 args를 넣었을 때 `write_sim_config` 출력이 지금 형식과 바이트 단위로 같음. 시스템 정보 줄(timestamp, python, platform, cwd)은 제외한다.
  - 체크포인트 파일 이름과 `Run.checkpoint_numbers`가 서로 맞물림
- [x] 01의 Train_ 소형 학습 기준선과 비교한다.
  - `.pth`, `loss_*.csv`, `pca_arrays.npz`가 bit 단위로 같다.
  - sim_config.txt는 시스템 정보 줄을 빼고 텍스트가 같다.
  - 결과를 이 파일의 Comments에 남긴다.
- [x] 01의 평가 기준선 비교를 다시 돌려서, 평가 3종이 여전히 bit 단위로 같은지 확인한다.
- [x] README 6장(Function_.py 함수 표, 의존 관계)을 갱신한다.

## Comments

### 2026-10-14 완료

**검증 결과**
- `python -m pytest`: 26개 통과, 약 6초.
- 이번에 추가한 쓰기 쪽 테스트는 세 가지다.
  - 실제 Run 260224_090359의 sim_config에서 학습 인자와 extra를 되살려 `write_sim_config`로 썼다. 결과는 Linux에서 만든 원본과 시스템 정보 줄을 빼고 바이트 단위로 같았다.
  - round-trip을 두 경우로 확인했다. 하나는 원본 설정이고, 다른 하나는 dataset_name, learning_case 2, normalize_dx를 바꾼 설정이다.
  - 체크포인트 파일 이름과 스캔이 서로 맞물리는지는 02에서 이미 확인했다.
- 기준선 비교 결과(`compare <target> before1 after03`): 네 대상 모두 모든 기록이 bit 단위로 일치했다.

| target | 기록 수 | after03 소요 시간 | 비교한 것 |
|---|---|---|---|
| train | 14 | 3.8 s | 체크포인트 4개 tensor, loss CSV 4개, `pca_arrays.npz`, `pca_data_nor.csv`, manifest, sim_config(시스템 줄 제외), 학습 PCA 입출력 |
| test | 304 | 43.5 s | 02와 같음 |
| tsne | 24 | 22.5 s | 02와 같음 |
| multi | 304 | 207.7 s | 02와 같음. temp_cache 남지 않음 |

**구현 메모**
- Train_은 `RunConfig.from_mapping({**vars(args), "no_use_dx": not use_dx_in_forward})`로 설정을 만든다. 디버그 모드에서 Δ 사용을 끄는 기존 동작을 설정에 반영하기 위해서다.
- 모델은 지금도 U, X 순서로 연속 생성한 뒤 각각 `.to(device)`한다. 학습 device 처리는 바꾸지 않았다.
- `write_sim_config`는 파일을 쓰고 텍스트를 돌려준다. 콘솔 출력은 지금처럼 Train_이 한다.

**정리한 것.** 이번 변경으로 쓰이지 않게 된 것만 지웠다.
- Function_.py: `print_sim_config`, `_as_mapping`, `_fmt`, 그리고 import `Mapping`, `Sequence`, `asdict`, `is_dataclass`, `platform`
- Train_.py: `from contextlib import redirect_stdout`, 지역 변수 `add_one_output_layer`

**검증하지 못한 경로**
- 디버그 모드 학습과 GPU 학습은 실행해 보지 않았다.
- 디버그 모드에서 Δ를 끄는 동작은 코드로만 확인했다. 기존 `use_dx_in_forward` 값이 그대로 설정에 들어간다.
