# tests 폴더 안내

이 폴더에는 코드가 제대로 동작하는지 자동으로 확인하는 **테스트 코드**가 있습니다.
코드를 고친 뒤 아래 명령 하나로 전부 실행해서, 예전과 같은 결과가 나오는지 확인합니다.

```bash
python -m pytest            # 저장소 루트에서 실행. 약 20초
python -m pytest -m "not realdata"   # 실제 데이터가 필요한 테스트를 빼고 실행
```

**pytest**는 파이썬 테스트 실행 도구입니다. 이름이 `test_`로 시작하는 파일에서 `test_`로 시작하는 함수를 찾아 차례로
실행하고, 함수 안의 `assert`(이 조건이 참이어야 한다)가 하나라도 거짓이면 그 테스트를 실패로 보고합니다.
설정은 저장소 루트의 `pytest.ini`에 있습니다.

## 파일 구성

| 파일 | 하는 일 |
|---|---|
| `test_diagnosis.py` | 진단 계산(진단 특징 → 관리도 → CI → ROC)의 결과가 저장해 둔 기대값과 완전히 같은지 확인 |
| `test_reconstruction.py` | 모델의 재구성 값·재구성 오차·진단 특징 계산 확인 (`mcdl/reconstruction.py`) |
| `test_chart_source.py` | 관리도를 캐시·저장 파일·다시 계산 중 어디서 가져오는지 확인 (`mcdl/chart_source.py`) |
| `test_run.py` | 학습 결과 폴더(Run)의 설정 읽기·쓰기, 체크포인트 찾기, 모델 불러오기 확인 (`mcdl/run.py`) |
| `test_vehicles.py` | 분할 파일 읽기, 차량 원본 읽기·전처리, 차량 데이터 캐시 확인 (`mcdl/vehicles.py`) |
| `test_train_rows.py` | 학습 데이터를 batch로 꺼내는 `Train_.row_loader`가 예전 방식과 같은 batch를 내는지 확인 |
| `test_scripts.py` | 실행 스크립트(`Train_.py`, `Test_.py` 등)를 가짜 데이터로 처음부터 끝까지 실행해 보는 smoke 테스트 |
| `fake_vehicles.py` | (도우미) 테스트용 가짜 차량 원본 데이터와 분할 파일을 만드는 함수 |
| `synthetic_errors.py` | (도우미) 진단 계산 테스트용 합성 재구성 오차를 만드는 함수 |
| `make_diagnosis_golden.py` | (도구) `test_diagnosis.py`가 쓰는 기대값 파일을 다시 만드는 스크립트 |
| `fixtures/` | 테스트가 읽어 쓰는 미리 준비된 데이터 파일 |
| `realdata/baseline.py` | (도구) 실제 데이터로 코드 수정 전후 결과를 비교하는 스크립트 |

도우미·도구 파일은 이름이 `test_`로 시작하지 않으므로 pytest가 테스트로 실행하지 않습니다.

## 자주 나오는 말

**fixture (픽스처)**
테스트가 실행되려면 미리 갖춰져 있어야 하는 준비물을 말합니다. 이 저장소에서는 두 가지 뜻으로 씁니다.

- **fixture 파일**: 테스트가 읽어 쓰려고 미리 저장해 둔 데이터 파일. `fixtures/` 폴더 이름이 여기서 왔습니다.
  - `fixtures/diagnosis_golden.npz`: `test_diagnosis.py`가 비교하는 기대값(아래 golden 값)
  - `fixtures/sim_config_260224_090359.txt`: 실제 학습 결과(Run 260224_090359)의 설정 파일 복사본. `test_run.py`가 씁니다.
- **pytest fixture**: 함수 위에 `@pytest.fixture`가 붙은 준비 함수. 테스트 함수의 인자 이름이 이 함수 이름과 같으면,
  pytest가 테스트 전에 이 함수를 실행해 그 결과를 인자로 넣어 줍니다. 예: `test_chart_source.py`의 `run`은 가짜 Run 폴더를
  만들어 줍니다. pytest가 기본으로 주는 것도 있습니다.
  - `tmp_path`: 테스트마다 새로 만드는 빈 임시 폴더
  - `monkeypatch`: 테스트 동안만 작업 폴더·환경 변수 등을 바꾸고 끝나면 되돌리는 도구
  - `capsys`: 테스트 중 화면 출력을 잡아 두는 도구

**golden 값 (기대값)**
"이 입력을 넣으면 정확히 이 값이 나와야 한다"고 미리 저장해 둔 정답 값입니다. 기준이 되는 값이라는 뜻으로 golden이라고
부릅니다. `test_diagnosis.py`는 리팩터링 전 코드로 만든 golden 값과 지금 코드의 결과를 비교하므로, 이 테스트가 통과하면
계산이 예전과 같다는 뜻입니다.
계산 방법을 **일부러** 바꿨을 때만 `python tests/make_diagnosis_golden.py`로 golden 값을 다시 만듭니다. 고친 적이 없는데
실패한다면 다시 만들지 말고 원인을 찾아야 합니다.

**bit 단위로 같다**
소수점 끝자리까지 완전히 같다는 뜻입니다(컴퓨터 안의 0과 1이 하나도 다르지 않다). 이 저장소의 테스트는 대부분
"거의 같다"가 아니라 "완전히 같다"를 확인합니다. 다른 운영체제나 라이브러리 버전에서는 마지막 자릿수만 달라져 실패할
수 있습니다(`test_diagnosis.py` 맨 위 설명 참고).

**smoke 테스트**
결과 값이 정확한지까지는 따지지 않고, 전체 흐름이 오류 없이 끝까지 돌아가고 기대한 파일이 생기는지만 빠르게 확인하는
테스트입니다. 전자 제품에 처음 전원을 넣어 연기가 나지 않는지 보는 데서 온 말입니다. `test_scripts.py`가 여기에 해당합니다.

**realdata (실제 데이터 테스트)**
`@pytest.mark.realdata`가 붙은 테스트는 실제 학습 결과 `models/260224_090359`(와 `data/`)가 있어야 돌고, 없으면 건너뜁니다.
`realdata/baseline.py`는 pytest와 별개로, 실제 데이터로 스크립트를 실행해 코드 수정 전후의 결과를 bit 단위로 비교하는 도구입니다.
사용법은 그 파일 맨 위 설명에 있습니다.

**mcdl**
이 저장소의 계산 코드가 들어 있는 폴더(파이썬 패키지) 이름입니다. 논문 제목 **M**odel-**C**onstrained **D**eep **L**earning의
머리글자입니다. 테스트는 대부분 이 폴더의 파일을 하나씩 맡아 확인합니다.

용어 전체는 저장소 루트의 `GLOSSARY.md`에 있습니다.
