# 14: Test_tsne --save가 저장한 그림을 닫지 않는다

**Status:** resolved

**Blocked by:** None (can start immediately)

**현상**
- `plot_tsne_before_after_ci`는 `args.save`이면 그림을 파일로 저장만 하고 `plt.close`를 하지 않았다.
- Test_tsne.py는 대상 차량마다 이 함수를 부르므로, `--fault-idx` 없이 test + fault 차량 전체(기본 분할 약 200대)를 그리면
  1×3 그림(18×6인치)이 차량 수만큼 메모리에 남는다. matplotlib은 그림이 20개를 넘으면 경고도 낸다.

**영향:** 메모리 사용량과 경고만 문제다. 저장되는 그림과 계산 결과에는 영향이 없다.

**출처:** 2026-10-16 코드 정리 중 발견.

## Comments

### 2026-10-16 해소

- 저장한 뒤 `plt.close(fig)`로 닫는다. `--save`가 없으면 예전처럼 창으로 띄운다.
- 검증: `tests/test_scripts.py::test_tsne_saves_the_plot`이 실행 뒤 열린 그림이 없는지 확인한다(수정 전 코드에서는 1개가 남아 실패).
- 실제 데이터 tsne 기록(fx01)이 op01과 같고, 다시 만든 `tSNE_338.png`도 바이트 단위로 같다.
