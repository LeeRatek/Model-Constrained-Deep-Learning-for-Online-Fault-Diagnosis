# 12: data_filter 분석 모드의 SOC 편차 분포가 vin_2 열을 읽는다

**Status:** resolved

**Blocked by:** None (can start immediately)

**현상**
- data_filter.py의 ANALYSIS_MODE는 `narray2 = np.array(combined_tensor)`로 vin_2 텐서를 한 번 더 복사했다. vin_3 텐서(`combined_tensorx`)는 만들기만 하고 쓰지 않았다.
- "SOC divation 1/2" 분포는 `narray2[:, x2 + y2]`, `[:, x2 + y2 + 1]`을 그렸다. QAS는 x2 + y2 = 112이므로, 실제로는 vin_2의 셀 전압 편차 0·1번 열이었다("Volatage divation 1/2"와 같은 값).

**영향:** 코드에서 `mode = MODE.ANALYSIS_MODE`로 바꿨을 때 화면에 띄우는 분포 그림 2장만 틀린다. 분할 파일(FILTER_MODE)과 학습·평가에는 영향이 없다.

**출처:** 2026-10-16 코드 정리(쓰이지 않는 변수 탐색) 중 발견.

## Comments

### 2026-10-16 해소 (코드 정리와 함께)

- ANALYSIS_MODE가 vin_2(`u_data`)와 vin_3(`x_data`)를 따로 이어 붙이고, SOC 편차 분포는 `x_data`의 셀 SOC 편차 0·1번 열에서 그린다.
- FILTER_MODE 결과는 그대로다. `--dataset-name _verify_df_op_before/after --train-ratio 0.02 --val-ratio 0.5`로 만든 분할 .npy 다섯 개가 정리 전후 같았다.
