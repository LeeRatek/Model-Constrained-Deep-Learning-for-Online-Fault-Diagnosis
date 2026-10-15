# 03: 파일 정리와 README

**What to build:** 레거시 파일을 지우고, README를 바뀐 코드와 CLI에 맞춘다.

**Blocked by:** 02

**Status:** resolved

- [x] `models/` 루트 구버전 산출물 5개를 git 추적에서 지운다(`net.pth`, `netx.pth`, `pca_arrays.npz`, `pca_data_nor.csv`, `pca_manifest.json`).
- [x] `README.txt`, `check_data_stats.py`를 지운다. `backup/`은 남긴다.
- [x] README 갱신
  - 대상: 3장 구조, 6장 Class_ 표, 7장 옵션 표(지운 옵션), 9장(LSTM·check_data_stats·models 루트 산출물 항목)
  - README의 `README.txt` 언급과 데이터셋 링크를 확인한다.
- [x] pytest가 통과한다.

## Comments

**2026-10-16 완료**

- `git rm` 대상
  - `models/net.pth`, `models/netx.pth`, `models/pca_arrays.npz`, `models/pca_data_nor.csv`, `models/pca_manifest.json`
  - `README.txt`(데이터셋 링크는 README.md 1장과 5장에 있다)
  - `check_data_stats.py`
- `backup/`은 남겼다.
- README
  - 3장 구조: Class_ 설명, `check_data_stats.py`·`README.txt` 줄
  - 5장: `vin_1.pkl`은 현재 스크립트가 읽지 않는다고 적었다.
  - 6장: 의존 관계 표의 Train_·Class_, Class_ 표(LSTM 줄, `analyze_outputs` 언급)
  - 7장 옵션 표: Train_ LSTM·vin 줄, Test_ `--sigma-levels`, Test_multi 줄
  - 9장 알려진 제약: LSTM, check_data_stats, models 루트 산출물 항목
- pytest 82개 통과
