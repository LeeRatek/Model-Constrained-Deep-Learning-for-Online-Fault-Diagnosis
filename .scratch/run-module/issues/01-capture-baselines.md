# 01: 실데이터 기준선 수집 (코드 변경 없음)

**What to build:** 리팩토링 전 코드로 평가 3종과 학습 1종을 실행해 결과를 저장한다. 같은 도구로 나중에 새 결과와 bit 단위로 비교할 수 있게 한다. 이 티켓에서는 루트 스크립트, Function_.py, Class_.py를 바꾸지 않는다.

**Blocked by:** None (can start immediately)

**Status:** resolved

- [x] Run 260224_090359의 사본을 `models/_verify/` 아래에 만든다. `pca_data_nor.csv`와 `results/`는 빼고 복사한다. 원본 Run 폴더는 바뀌지 않는다.
- [x] 기준선 도구를 tests 아래 realdata 폴더에 둔다. pytest가 수집하지 않는 이름을 쓴다. 도구는 다음 대상을 runpy로 실행하고, 결과를 `models/_verify/` 아래 pickle로 저장한다(`MPLBACKEND=Agg`).
  - Test_ (-1/-1): `roc`, `opt`, `y_true`, `predict_results`, 차량별 T²·SPE 배열 해시
  - Test_multi_models (`--u-model-idx -1 --x-model-idx -1`, `--keep-cache` 없이): AUC_summary CSV의 해당 행 텍스트, ROC 배열
  - Test_tsne (`--fault-idx 338`, -1/-1, `--save`): CI 시계열, t-SNE 임베딩 배열
- [x] 같은 도구에 비교 모드를 둔다. 비교 모드는 저장된 기준선과 새 실행 결과를 bit 단위로 비교하고, 다른 항목을 이름별로 보고한다.
- [x] Train_ 소형 학습을 두 번 실행한다. 조건은 첫 학습 차량 1대, U·X 각 1 epoch, save-per 1, `--no-validation`, seed 42이고, 결과는 `models/_verify/` 아래에 둔다. 두 실행의 `.pth`, `loss_*.csv`, `pca_arrays.npz`가 bit 단위로 같은지 확인한다. 다르면 멈추고 보고한다.
- [x] 평가 3종의 기준선을 Test_ → Test_tsne → Test_multi_models 순서로 뜬다. 뜬 뒤 `models/_verify/` 사본에 temp_cache가 남지 않았는지 확인한다.
- [x] 각 기준선의 생성 명령, 소요 시간, 저장 위치를 이 파일의 Comments에 기록한다.

## Comments

### 2026-10-14 기준선 수집 완료

**도구:** `tests/realdata/baseline.py`

```
python tests/realdata/baseline.py capture <test|tsne|multi|train> <tag>
python tests/realdata/baseline.py compare <target> <tag_a> <tag_b>
```

- 저장소 루트에서 bms-mcdl 환경의 python으로 실행한다.
- 결과는 `models/_verify/baselines/<tag>/<target>.pkl`과 콘솔 로그 `.log`에 남는다.
- 동작 방식: Function_의 다음 함수를 기록용 wrapper로 바꾼 뒤 runpy로 스크립트를 실행한다. 스크립트 코드는 건드리지 않는다.
  - 대상 함수: `DiagnosisFeature`, `T2_array`, `SPE_array`, `Custom_PCA`(입력 포함), `load_pca_results`, ROC 2종(전체 값 보관), `plot_tsne_before_after_ci`(입력만)
  - `TSNE.fit_transform`도 함께 가로챈다.
- train 대상은 새 Run 폴더의 파일을 기록한다. `.pth`는 tensor별 해시, npz는 키별 해시를 남긴다. sim_config는 timestamp·python·platform·cwd 값을 지운 텍스트를 남긴다.

**기준선:** `before1`, `before2`. 모두 Windows 11, Python 3.12.13, numpy 2.3.5, torch 2.9.1+cpu 환경에서 떴다.

| target | 기록 수 | before1 | before2 | before1 vs before2 |
|---|---|---|---|---|
| train | 14 | 3.5 s | 3.6 s | bit 일치 |
| test | 304 | 49.4 s | 39.5 s | bit 일치 |
| tsne | 24 | 27.7 s | 23.1 s | bit 일치 |
| multi | 304 | 482.0 s (`.pt` 캐시 생성) | 222.1 s (캐시 사용) | bit 일치 |

**관찰한 점**
- 네 대상 모두 실행할 때마다 결과가 같았다(결정적). 리팩토링 후 결과는 `before1`과 비교하면 된다.
- Test_multi_models는 `.pt` 캐시가 없을 때와 있을 때 결과가 같았다.
- Test_ (-1/-1) 결과는 AUC 0.9148, Youden 최적 임계값 23.900000000000002 (FPR 0.02, TPR 0.78), (0,1) 최근접 거리 0.2209072203437452다.
  - Test_multi_models의 CSV 행도 같은 값이다.
  - Linux에서 만든 원본 `AUC_summary_case1.csv`의 (-1,-1) 행과 텍스트까지 같다.
- `data/`에 `.pt` 캐시가 생겼다: train_data_list_QAS_015ac371…pt 26.7 GB, test_data_cache_QAS_528fb02a…pt 14.3 GB. 리팩토링 후 비교 실행에서 재사용한다.
- temp_cache는 두 multi 실행 모두 남지 않았다.
- 원래 계획은 npz였지만, 기록에 dict·문자열·배열이 섞여 있어 pickle로 저장했다.
- 첫 시도에서 `save_pca_results`의 입력을 기록했더니, 입력에 RUN_ID가 들어간 저장 경로가 포함되어 실행마다 달랐다. 그래서 이 wrapper는 뺐다. 저장된 파일 자체는 train 대상의 파일 기록으로 비교된다.
