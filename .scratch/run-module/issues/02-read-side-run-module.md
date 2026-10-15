# 02: Run module (읽기 쪽)과 평가 스크립트 3개 전환

**What to build:** 평가 스크립트 3개(Test_, Test_multi_models, Test_tsne)가 Run module로 Run을 연다. 스크립트는 설정, 체크포인트 번호 목록, 모델 쌍만 받는다. 각 스크립트에 복사되어 있던 sim_config 해석, 레거시 기본값, 모델 생성·로드 코드는 사라진다. README 7장 명령과 수치 결과는 그대로다. Run module은 원본 데이터 없이 도는 테스트로 검증한다.

**Blocked by:** 01 (실데이터 기준선 수집)

**Status:** resolved

- [x] `mcdl` 패키지와 Run module을 만든다. 구성은 `RunConfig`(`from_mapping` 포함), `Run`(`.config`, `.checkpoint_numbers`, `.load_model_pair`), `build_model_pair`다. 모양과 규칙은 spec의 Implementation Decisions를 따른다.
- [x] 루트에 pytest 설정을 둔다. import 경로에 저장소 루트를 넣고, 테스트 경로를 지정하고, `realdata` 마커를 등록한다. 원본 데이터 없이 CPU에서 `python -m pytest`가 통과해야 한다.
- [x] 다음 테스트를 작성한다. 모두 Run module interface만 거친다.
  - 실제 sim_config fixture에서 읽은 설정 필드 값
  - 레거시 키를 지운 변형의 기본값
  - 필수 키가 없을 때의 예외
  - split_dir 해석 3가지: 폴더가 있을 때, 없을 때, None·default일 때
  - 체크포인트 번호 목록과 그 순서
  - `load_model_pair` 출력이 참조 경로와 bit 단위로 같고 full-precision 로드와는 다름 (known-bugs 01 고정)
  - realdata: 실제 Run을 로드하는 테스트. `models/`가 없으면 skip
- [x] Test_, Test_multi_models, Test_tsne가 Run module을 쓴다. 평가 device는 CPU로 고정한다.
- [x] 스크립트별 차이는 그대로 두고, 호출부에 known-bugs 번호를 주석으로 남긴다.
  - 평가 스크립트의 normalize_val 미전달 (02)
  - Test_tsne가 CLI에서 읽는 battery_type·normalize_dx (03)
- [x] sim_config reader 3개를 Function_.py에서 Run module로 옮긴다. 이번 변경으로 쓰이지 않게 된 원본은 지운다. Train_pca_only의 분할 경로 호출 한 줄은 옮긴 함수를 가리키게 바꾼다.
- [x] README를 갱신한다.
  - 3장 구조: `mcdl/`, tests 추가
  - 6장: 의존 관계 표와 Function_.py 함수 표
  - 새 "테스트" 소절: pytest 명령, realdata 기준선 비교 절차
- [x] 01의 기준선 도구를 비교 모드로 돌려 평가 3종이 bit 단위로 일치하는지 확인하고, 결과를 이 파일의 Comments에 남긴다.
- [x] 지금 예외가 나던 경로 가운데 이번 변경으로 살아나는 것을 해당 known-bugs 티켓 Comments에 기록한다. 대상은 오래된 config에서의 Test_tsne(04)와 Test_다.

## Comments

### 2026-10-14 완료

**검증 결과**
- `python -m pytest`: 23개 통과, 약 6초. 원본 데이터 없이 CPU에서 돈다. realdata 테스트 1개는 `models/260224_090359`가 있어 실행되었다.
- 기준선 비교 결과(`compare <target> before1 after02`): 세 대상 모두 모든 기록이 bit 단위로 일치했다.

| target | 기록 수 | after02 소요 시간 |
|---|---|---|
| test | 304 | 47.9 s |
| tsne | 24 | 23.9 s |
| multi | 304 | 216.8 s (`.pt` 캐시 사용, temp_cache 남지 않음) |

**spec과 달라진 점**
- `Run.load_model(model, number)`를 추가했다.
  - Test_multi_models는 U-model과 X-model을 체크포인트 번호별로 따로 미리 로드하는데, `load_model_pair`만으로는 이 구조에 맞지 않았다.
  - `load_model_pair`는 `load_model`을 U, X 순서로 부른다. spec의 interface 절도 갱신했다.
- `checkpoint_filename`은 읽기 쪽에도 필요해서 이번 티켓에서 먼저 만들었다. 원래는 티켓 03 범위였다.
- Train_pca_only의 분할 경로 호출은 `read_dataset_path_from_sim_config` 대신 `Run(...).config.split_dir`을 쓴다.
  - 이 줄은 원래 Test_에서 도달하지 못한다. 함수 첫머리에서 known-bugs 06의 AttributeError가 먼저 나기 때문이다.
  - 차이는 sim_config에 필수 키가 없을 때뿐이다. 예전에는 "."로 넘어갔고, 지금은 ValueError가 난다.

**정리한 것.** 이번 변경으로 쓰이지 않게 된 것만 지웠다.
- Function_.py: `read_values_from_sim_config`, `read_dataset_path_from_sim_config`, `_coerce_sim_config_value`, 그리고 `SimpleNamespace` import
- Test_multi_models.py: `scan_model_indices`, `import re`
- 원래부터 쓰이지 않던 reader 2개(`read_learning_case_from_sim_config`, `load_vehicle_ids_used_for_training`)는 그대로 두었다.

**수치에는 영향이 없지만 달라진 점**
- Test_가 fc1·fc2 파라미터를 콘솔에 출력하고 그림으로 저장하는 시점에, 모델이 이제 double이다. 예전에는 float32였다. 값은 float32로 잘린 값 그대로라서 같고, 콘솔에 찍히는 숫자 형식만 달라질 수 있다.
- 콘솔 메시지가 일부 바뀌었다.
  - "Loaded U model:" 뒤에 파일 이름(`net_e3`) 대신 체크포인트 번호(`3`)를 찍는다.
  - 분할 폴더 경고는 Run을 열 때 찍힌다.

**검증하지 못한 경로**
- Test_multi_models의 `--u-model-idx auto` 스캔은 실데이터로 돌리지 않았다. 14×14 조합이라 오래 걸리기 때문이다.
- 대신 단위 테스트와 realdata 테스트로 확인했다. realdata 테스트에서 260224_090359의 U 체크포인트 번호는 0–12, -1로 나왔다.
