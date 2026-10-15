# Spec: Run module

**Status:** resolved (티켓 01–03 완료, 2026-10-14)

출처: 2026-10-14 아키텍처 리뷰의 1번 후보("Run 폴더를 하나의 deep module로")와 그 grilling(Q1–Q19). 용어는 `GLOSSARY.md`를 따른다.

## Problem Statement

연구자는 학습한 Run을 여러 평가 스크립트(Test_, Test_multi_models, Test_tsne)로 평가한다. 그런데 Run을 해석하는 규칙이 스크립트마다 따로 복사되어 있다. sim_config 읽기, 키가 없을 때의 기본값, learning_case에서 나오는 파생 설정, 분할 폴더 경로, 체크포인트 번호 ↔ 파일 규칙, U-model·X-model 생성과 로드가 모두 그렇다. 그 결과 다음 문제가 생긴다.

- 같은 Run을 스크립트마다 다르게 해석할 수 있다. Test_tsne는 battery_type을 sim_config가 아니라 CLI에서 읽는다.
- 한 곳을 고치면 다른 복사본을 놓친다. 모델 생성 코드만 10벌이다.
- 테스트가 하나도 없다. 스크립트는 import만 해도 전체를 실행한다. 원본 데이터(QAS 약 49 GB) 없이는 무엇이 바뀌었는지 확인할 방법이 없다.

## Solution

Run을 해석하는 지식을 Run module 하나에 모은다.

- 평가 스크립트는 Run을 열어 설정, 체크포인트 번호 목록, 모델 쌍만 받는다.
- 학습 스크립트는 같은 module로 sim_config를 쓰고, 체크포인트 파일 이름을 정하고, 모델 쌍을 만든다.

디스크 형식, CLI, 수치 결과는 바뀌지 않는다. Run module은 원본 데이터 없이 합성 Run으로 몇 초 안에 테스트한다. 수치 보존은 리팩토링 전에 떠 둔 실데이터 기준선과 bit 단위로 비교해 증명한다.

## User Stories

1. 연구자로서, 리팩토링 후에도 기존 Run 폴더(sim_config.txt, artifact, pca_arrays.npz)를 그대로 평가하고 싶다. 그래야 과거 실험과 비교할 수 있다.
2. 연구자로서, README 7장의 명령을 지금 그대로 쓰고 싶다. 그래야 실행 방법을 다시 익히지 않아도 된다.
3. 연구자로서, 리팩토링 전후 AUC, 최적 임계값, 차량별 T²·SPE가 bit 단위로 같다는 증거를 보고 싶다. 그래야 보고한 수치를 계속 믿을 수 있다.
4. 연구자로서, 평가 스크립트들이 같은 Run을 같은 규칙으로 해석하길 원한다. 의도적으로 남긴 차이는 표시된 채로 둔다. 그래야 스크립트 간 결과 차이의 원인을 찾을 수 있다.
5. 연구자로서, sim_config에 일부 키가 없는 오래된 Run도 지금과 같은 기본값으로 열리길 원한다.
6. 연구자로서, battery_type이나 learning_case가 없는 Run을 열면 무엇이 없는지 즉시 알려 주길 원한다. 그래야 엉뚱한 곳에서 None 때문에 죽지 않는다.
7. 연구자로서, 체크포인트 번호 -1과 N이 지금과 같은 가중치 파일을 가리키길 원한다.
8. 연구자로서, Test_multi_models의 auto 스캔이 지금과 같은 순서(오름차순, -1은 맨 뒤)로 체크포인트를 나열하길 원한다. 그래야 CSV와 히트맵 순서가 바뀌지 않는다.
9. 연구자로서, GPU가 있는 머신에서도 평가가 CPU로 일관되게 돌길 원한다. 그래야 입력과 모델의 device가 어긋나 죽지 않는다.
10. 연구자로서, 분할 폴더 해석 규칙(`dataset_name`이 없거나 None·default면 루트 분할, 폴더가 없으면 경고 후 루트)이 그대로이길 원한다.
11. 연구자로서, 새로 학습한 Run의 sim_config.txt가 지금과 같은 형식으로 쓰이길 원한다. 그래야 이전 코드로도 읽을 수 있다.
12. 연구자로서, 학습 결과(체크포인트, loss, PCA 관리한계)가 리팩토링 전후 같길 원한다.
13. 개발자로서, Run에 관한 규칙을 한 module에서 찾고 싶다. 그래야 스크립트 5개를 뒤지지 않아도 된다.
14. 개발자로서, 원본 데이터 없이 CPU에서 몇 초 안에 도는 테스트로 Run module을 검증하고 싶다.
15. 개발자로서, 테스트가 Run module의 interface만 거쳐 검증하길 원한다. 그래야 내부를 바꿔도 테스트가 살아남는다.
16. 개발자로서, 가중치 float32 절삭처럼 의도와 다를 수 있는 현재 동작이 테스트로 고정되어 있길 원한다. 그래야 실수로 바꾸면 바로 드러난다.
17. 개발자로서, 알려진 버그를 리팩토링과 섞지 않고 별도 티켓으로 다루고 싶다.
18. 개발자로서, 스크립트별로 일부러 남긴 차이가 호출부에 known-bugs 티켓 번호와 함께 표시되어 있길 원한다.
19. 개발자로서, 학습과 평가가 같은 규칙으로 U-model·X-model을 만들길 원한다. 출력 sigmoid의 scale·shift 선택, Δ 사용 여부, 추가 출력층 여부가 그 규칙이다.
20. 개발자로서, sim_config를 쓰고 다시 읽으면 같은 설정이 나오는지 확인하는 round-trip 테스트를 원한다.
21. 개발자로서, 실데이터 기준선 비교를 같은 도구로 언제든 다시 돌리고 싶다. 그래야 이후 후보(차량 데이터, 진단 모니터 등)의 리팩토링에도 재사용할 수 있다.
22. 개발자로서, GLOSSARY 용어(Run, 체크포인트 번호, U-model/X-model, 모델 쌍)가 코드 interface에도 그대로 쓰이길 원한다.
23. 리뷰어로서, 각 티켓을 마칠 때 기준선 비교 결과가 그 티켓에 기록되길 원한다.

## Implementation Decisions

### module 위치와 이름

- 새 deep module은 패키지 `mcdl`에 모은다. 이번 module은 `mcdl.run`이다.
- 루트의 기존 스크립트는 이 패키지를 import한다. 실행은 지금처럼 저장소 루트에서 한다.
- 테스트는 별도 tests 폴더에 두고, 루트의 pytest 설정에서 다음을 지정한다.
  - import 경로에 저장소 루트를 추가한다.
  - 테스트 경로를 지정한다.
  - `realdata` 마커를 등록한다.

### interface (grilling Q16에서 합의한 모양)

```python
@dataclass(frozen=True)
class RunConfig:                 # 필드 이름 = sim_config 키 이름 (파생 필드 제외)
    battery_type: str
    learning_case: int
    ae_u_scale; ae_u_shift; ae_x_scale; ae_x_shift
    use_dx: bool                 # not no_use_dx
    add_one_output_layer: bool
    normalize_dx: bool
    normalize_val
    no_abs_err: bool
    vehicle_start: int
    vehicle_end: int
    preprocessing: bool          # learning_case에서 파생
    skip_charge_ready: bool      # learning_case에서 파생
    split_dir: str               # "./datasets/<name>" 또는 "."

    @classmethod
    def from_mapping(cls, values) -> "RunConfig"

class Run:
    def __init__(self, path)                     # models/<RUN_ID>/
    config: RunConfig
    def checkpoint_numbers(self, model)          # model: "U" | "X"
    def load_model(self, model, number)          # → U-model 또는 X-model (티켓 02에서 추가)
    def load_model_pair(self, u, x)              # → (U-model, X-model)

def build_model_pair(config)                     # → (U-model, X-model)
def checkpoint_filename(model, number)           # 읽기·쓰기 공용 (티켓 02에서 만듦)
def write_sim_config(path, args, extra)          # 쓰기 쪽
```

- 평가 스크립트가 알아야 할 것은 `Run(path)`, `.config`, `.checkpoint_numbers`, 그리고 모델 로드(`.load_model_pair`, 또는 Test_multi_models처럼 U·X를 따로 미리 로드할 때는 `.load_model`)다.
- `load_model`은 grilling Q16 이후에 추가했다. Test_multi_models가 체크포인트 번호별로 U-model과 X-model을 따로 미리 로드하기 때문이다. `load_model_pair`는 `load_model`을 U, X 순서로 부른다.
- `checkpoint_filename`은 체크포인트를 읽을 때도 필요해서 티켓 02에서 먼저 만들었다.
- interface에서는 `"net"`/`"netx"` 대신 `"U"`/`"X"`를 쓴다. 파일 이름 규칙은 module 안에 숨긴다.
- sim_config에 있는 나머지 키(lr, epochs, lstm_* 등)는 노출하지 않는다.

### 설정 해석 규칙 (지금 동작을 그대로 옮김)

**레거시 기본값.** 키가 없을 때 쓰는 값이며, `from_mapping` 한 곳에서만 채운다.

| 키 | 기본값 |
|---|---|
| ae_u_scale / ae_u_shift | 1.8 / 2.5 |
| ae_x_scale / ae_x_shift | 1.0 / 0 |
| add_one_output_layer | False |
| no_use_dx | False (즉 use_dx = True) |
| normalize_dx | False |
| normalize_val | 4 (Train_ 기본값) |
| no_abs_err | False |
| vehicle_start / vehicle_end | 0 / -1 |

**필수 키.** battery_type이나 learning_case가 없으면, 키 이름과 Run 경로를 담은 예외를 바로 낸다.

**값 변환.** 지금 규칙을 그대로 쓴다.
- true/false, 정수, 지수 표기가 없는 실수, 리터럴(`[...]` 등)만 변환한다.
- `"None"`이나 `1e-05` 같은 문자열은 문자열 그대로 둔다.
- 정수 0과 실수 0.0의 타입 차이도 유지한다.

**파생값.**
- learning_case → preprocessing / skip_charge_ready: 기존 표를 그대로 쓴다.
- split_dir: 기존 규칙을 그대로 쓴다. cwd 기준 `./datasets/<name>` 폴더가 있으면 그 경로를, `dataset_name`이 없거나 None·default이거나 폴더가 없으면 `"."`를 쓴다. 폴더가 없을 때 경고를 출력하는 것도 유지한다.

**체크포인트 스캔.** 기존 규칙을 그대로 쓴다. 최종 모델 파일은 -1이고, epoch 파일은 그 번호다. 오름차순으로 정렬하고 -1은 맨 뒤에 둔다.

### 모델 쌍 생성과 로드

- `build_model_pair`가 설정에서 U-model과 X-model을 만든다. preprocessing이 켜져 있으면 sigmoid(1, 0)을 쓰는 분기도 여기에 둔다. 생성 순서는 U 다음 X로 고정한다. 학습의 RNG 소비 순서가 이 순서에 걸려 있다.
- `load_model_pair`는 지금 평가 경로와 같은 순서로 로드한다.
  1. 기본 dtype(float32)으로 모델을 만든다.
  2. 저장된 float64 state_dict를 `load_state_dict`로 넣는다.
  3. `.double().eval()`을 한다.
  - 이 순서 때문에 가중치가 float32로 잘린다(known-bugs 01). 순서는 module 안에서 이름 있는 단계로 드러낸다.
- 모델은 CPU에 둔다. 평가 스크립트의 device도 CPU로 고정한다.

### 스크립트별 차이와 버그 처리

**수치를 내는 경로.** 스크립트별로 다른 지금 동작을 그대로 유지한다. 차이는 호출부에 명시적으로 남기고 known-bugs 번호를 주석으로 붙인다.
- 평가 스크립트는 normalize_val을 넘기지 않는다. 즉 1.0이 쓰인다(known-bugs 02).
- Test_tsne는 battery_type과 normalize_dx를 CLI에서 읽는다(known-bugs 03).

**지금 예외가 나는 경로.** 리팩토링으로 살아나는 것은 허용한다. 해당 버그 티켓에는 기록만 남긴다.
- 오래된 config에서 Test_tsne가 `scale=None`으로 죽던 경로(known-bugs 04)
- 오래된 config에서 Test_가 normalize_dx를 읽지 못해 죽던 경로
- GPU 환경의 평가 스크립트

### Function_.py 정리

- sim_config를 읽는 구현(`read_values_from_sim_config`, `_coerce_sim_config_value`, `read_dataset_path_from_sim_config`)을 Run module로 옮긴다.
- 원본 함수는 이번 변경으로 쓰이지 않게 된 것만 지운다. Train_pca_only가 분할 경로를 읽는 호출 한 줄은 옮긴 함수를 가리키게 바꾼다.
- 쓰기 쪽 티켓에서는 `print_sim_config`와 그 서식 helper를 같은 방식으로 옮긴다. 평가 스크립트가 설정을 콘솔에 찍는 호출도 옮긴 함수를 쓰게 한다.
- `CombinedAE`, `CustomSigmoidFunc`, `get_input_dimensions`, `get_preprocessing_and_skip_charge_ready`는 다른 곳에서도 쓰므로 그 자리에 두고 import만 한다.
- 원래부터 쓰이지 않던 reader 2개는 건드리지 않는다.

### 쓰기 쪽 (Train_)

- Train_은 args로 `RunConfig.from_mapping`을 거쳐 설정을 만들고, `build_model_pair`로 모델을 만든다.
- 체크포인트 저장 경로는 `checkpoint_filename`을 쓴다.
- sim_config.txt는 `write_sim_config`로 쓴다. 내용은 지금과 바이트 단위로 같아야 한다. 시스템 정보 줄(timestamp, python, platform, cwd)만 예외다.

### 바꾸지 않는 것

- CLI는 그대로다. `--models-folder`(Test_)와 `--models-idx`(나머지)도 그대로 둔다.
- 디버거 분기와 import 시점 실행도 그대로 둔다.

## Testing Decisions

**좋은 테스트의 기준.** Run module의 interface(`Run`, `RunConfig.from_mapping`, `build_model_pair`, 쓰기 함수)만 거쳐서 관찰 가능한 결과를 확인한다. 내부 helper나 파싱 중간값은 직접 테스트하지 않는다.

**fixture**
- 실제 Run 260224_090359의 sim_config.txt 사본을 커밋한다. 파일 안에 서버 경로와 잘린 차량 ID 목록이 들어 있다.
- 위 사본에서 레거시 키(ae_*_scale/shift, add_one_output_layer, no_use_dx, normalize_dx 등)를 지운 변형은 테스트 안에서 만든다.
- 합성 Run은 `tmp_path`에 만든다.
  - sim_config를 쓴다.
  - 랜덤 float64 가중치로 최종 모델과 epoch 체크포인트 몇 개를 저장한다.
  - 차원은 실제 battery type(QAS 110셀)을 쓴다. 파일 하나가 200 KB 정도다.

**확인할 동작**
- 실제 sim_config에서 읽은 설정 필드 값
- 레거시 기본값
- 필수 키가 없을 때의 예외
- split_dir 해석: 폴더가 있을 때, 없을 때, None·default일 때
- 체크포인트 번호 목록과 그 순서
- `load_model_pair`의 출력: "float32 생성 → load → double" 참조 경로와 bit 단위로 같고, full-precision으로 로드했을 때와는 **다르다**. 지금의 절삭 동작을 고정하는 단언이다. known-bugs 01을 고치면 이 단언이 뒤집힌다.
- (쓰기 쪽) write → read round-trip, 그리고 지금 형식과 바이트 단위 일치(시스템 정보 줄 제외)

**realdata 테스트.** 실제 `models/260224_090359`의 체크포인트를 로드하는 테스트에는 `realdata` 마커를 붙인다. `models/`가 없으면 skip한다.

**실데이터 기준선.** pytest가 아니라 손으로 돌리는 도구다. 저장소의 tests 아래 realdata 폴더에 커밋하고, 결과물은 git에서 제외되는 `models/_verify/`에 둔다.
- 평가 쪽
  - 대상: Test_ (-1/-1), Test_multi_models (-1/-1 한 쌍, `--keep-cache` 없이), Test_tsne (차량 338, -1/-1, `--save`)
  - 방법: runpy로 실행하고, 결과 배열과 차량별 T²·SPE 해시 등을 pickle로 저장한다.
  - 실행 환경: `MPLBACKEND=Agg`. Run 사본에서 돌리고, 원본 Run 폴더에는 쓰지 않는다.
- 학습 쪽
  - 대상: Train_ 소형 학습 (첫 학습 차량 1대, U·X 각 1 epoch, save-per 1, `--no-validation`, seed 42)
  - 기준선을 두 번 떠서 결정적인지 먼저 확인한다.

**참고할 기존 테스트.** 저장소에 테스트가 없다. 이번 테스트가 첫 사례다.

## Out of Scope

- PCA 산출물(pca_arrays.npz)의 로드·소유, temp_cache 특징 캐시, PCA 출처 결정: 진단 모니터 후보
- 차량 데이터 로딩·전처리·`.pt` 캐시: 차량 데이터 후보
- argparse와 CLI 옵션 이름 통일, `main(argv)`, 디버거 분기: 진입 스크립트 후보
- Train_pca_only (분할 경로 호출 한 줄만 예외)
- known-bugs 01–09의 수정. 각각 별도 티켓에서 결정한다.
- 모델 구조, 학습 알고리즘, 진단 수식 변경, 새 기능
- 원래부터 쓰이지 않던 코드 삭제

## Further Notes

- **기준선은 이 Windows 환경에서 새로 뜬다.** 기존 `AUC_summary_case1.csv`는 Linux에서 만든 것이라 ULP 수준에서 다를 수 있으므로 기준선으로 쓰지 않는다.
- **디스크.** Test_multi_models 기준선을 처음 돌리면 원본 데이터 폴더에 수십 GB짜리 `.pt` 전처리 캐시가 생긴다. 리팩토링 후 실행에서는 같은 해시로 재사용한다. C 드라이브 여유 공간은 161 GB다.
- **기준선 순서.** Test_multi_models는 `--keep-cache` 없이 돌린다. temp_cache가 남으면 Test_와 Test_tsne가 PCA 출처로 그 캐시를 먼저 쓰게 되어 경로가 바뀐다.
- **bit 단위 동일성에 영향을 주는 요소.** Run module 바깥에 있지만 건드리지 않아야 한다.
  - 추론 batch 크기
  - Custom_PCA에 넘기는 DataFrame 입력
  - 임계값 grid `np.arange(0, 1000, 0.1)`
  - 기본 `np.argsort`
