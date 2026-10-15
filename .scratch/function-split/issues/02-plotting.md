# 02: 그림 함수를 `mcdl/plotting.py`로

**What to build:** 스크립트와 Class_가 쓰는 그림 함수 8개와 helper 3개를 `mcdl/plotting.py`로 옮기고, 쓰는 곳의 import를 바꾼다. 결과는 그대로다.

**Blocked by:** 01

**Status:** resolved

- [x] `mcdl/plotting.py`를 만든다. 11개 함수의 본문은 그대로 옮기고, 필요한 import만 둔다.
- [x] import를 바꾼다. 대상은 Test_, Test_multi_models, Test_tsne, Plot_AUC_heatmap, Class_, data_filter다.
- [x] 기준선 도구의 `plot_tsne_before_after_ci` wrapper 위치에 `mcdl.plotting`을 더한다.
- [x] 실데이터: test, tsne, multi가 `after_bugs`와 bit 단위로 일치해야 한다. tsne는 t-SNE 입력 wrapper가 실제로 기록되는지도 확인한다.
- [x] pytest가 통과한다.

## Comments

**2026-10-15 완료**

- `mcdl/plotting.py`(1,201줄): 그림 함수 8개와 helper 3개를 본문 그대로 옮겼다. import는 함수가 실제로 쓰는 9개만 둔다(os, plt, mtick, np, pd, torch, scipy `norm`, sklearn `TSNE`·`StandardScaler`).
- 쓰는 곳 import: Test_, Test_multi_models, Test_tsne, Plot_AUC_heatmap, Class_, data_filter. 스크립트의 star import는 03에서 지운다.
- 기준선 도구: `plot_tsne_before_after_ci` wrapper 위치에 `("mcdl.plotting", …)`을 더했다.
- Function_.py에는 지울 27개만 남았다(1,056줄).
- pytest 82개 통과
- 실데이터(기준선 태그 `fs02`): `after_bugs`와 비교해 모두 bit 단위로 일치한다.
  - test: 304개 기록
  - tsne: 24개 기록. t-SNE 입력 wrapper가 새 위치에서 1건 기록됐다
  - multi: 304개 기록
