# 02: Test_, Test_multi_models, Test_tsne 전환

**What to build:** 세 스크립트의 테스트 차량 본문에서 열 자르기·추론·오차 계산을 걷어내고, `vehicle_features`(와 Test_tsne의 `reconstruct`)를 쓴다. 평가 결과는 그대로다.

**Blocked by:** 01

**Status:** resolved

- [x] Test_: `vehicle_features(net, netx, u, x, abs_err=True)` → `chart.max_ci`/`chart.score`. 디버그 차량 고정과 플롯 분기는 그대로 둔다.
- [x] Test_multi_models: `process_combination`의 본문을 `vehicle_features(..., abs_err=True)` → `chart.max_ci`로 바꾼다. `.double()` 변환과 `del` 문을 지운다.
- [x] Test_tsne: `collect_vehicle_points`에서 CI는 `vehicle_features`로, t-SNE 점은 `reconstruct`로 얻는다. `dim_dict` 인자를 지운다.
- [x] 세 스크립트에서 쓰이지 않게 된 `dim_dict`와 `get_input_dimensions` 호출을 지운다.
- [x] 실데이터: test, tsne, multi가 `after_bugs`와 bit 단위로 일치해야 한다. 결과를 Comments에 남긴다.
- [x] README(6장 `mcdl/reconstruction.py` 소절)를 갱신한다.

## Comments

**2026-10-15 완료**

- Test_: 차량 본문은 `load_vehicle` → `vehicle_features(..., abs_err=True)` → `chart.max_ci`/`chart.score`다. import는 `mcdl.diagnosis` 대신 `mcdl.reconstruction`이다.
- Test_multi_models: 차량 본문은 캐시 텐서 → `vehicle_features(..., abs_err=True)` → `chart.max_ci`다. `.double()` 변환 8줄과 `del` 3줄을 지웠다.
- Test_tsne: `collect_vehicle_points`가 CI는 `vehicle_features`로, t-SNE 점은 `reconstruct`로 얻는다. 변수 이름 `y_recovered`·`y_recovered2`는 그대로 두었고 재구성 결과는 `recon_u`·`recon_x`다. `dim_dict` 인자를 지웠다.
- 세 스크립트에서 `dim_dict = get_input_dimensions(...)`를 지웠다. 이제 재구성 오차를 계산할 때 열 구성을 아는 곳은 `mcdl/reconstruction.py`의 `reconstruct` 하나다. Train_의 학습 Dataset(`VirtualConcatDataset` 등)은 여전히 `dim_dict`로 자른다(6번 후보, Train_ 학습 루프 범위).
- 지우지 않은 것: Test_.py와 Test_tsne.py에 원래 주석 처리되어 있던 분석 코드다. 이제 없는 변수(`ERRORU`, `recon_imtest`, `x_recovered` 등)를 참조한다. 리팩토링 범위 밖의 죽은 코드라 그대로 두었다.
- 실데이터(기준선 태그 `vf02`): `after_bugs`와 비교해 모두 bit 단위로 일치한다.
  - test: 304개 기록
  - tsne: 24개 기록. `plot_tsne_before_after_ci`의 입력(t-SNE 점, 알람 마스크, CI)을 포함한다
  - multi: 304개 기록. 소요 시간 210.5초(after_bugs는 210.4초)
- README: 3장 구조, 4장 테스트 설명, 6장 의존 관계 표와 `mcdl/reconstruction.py` 소절, 9장(디버그 표에 버그 11, 알려진 제약에 버그 10)
