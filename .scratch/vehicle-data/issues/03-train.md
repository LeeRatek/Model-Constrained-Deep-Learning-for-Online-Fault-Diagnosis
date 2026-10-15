# 03: Train_의 학습·검증 로딩 전환

**What to build:** Train_이 `RunConfig`를 앞쪽에서 한 번 만들어 데이터 로딩과 모델 생성에 함께 쓰고, 학습·검증 차량을 `split_ids`와 `load_vehicle`로 읽는다. 분할 경로 규칙(`./datasets/{이름}`, 폴더가 없으면 실패)은 지금과 같다. 학습 결과는 그대로다.

**Blocked by:** 01 (차량 데이터 module)

**Status:** resolved

- [x] `RunConfig.from_mapping({**vars(args), "no_use_dx": not use_dx_in_forward})`를 디버그 감지 직후 한 번 만들고, 모델 생성에도 그것을 쓴다.
- [x] 학습 차량은 `split_ids(<Train_ 분할 경로>, …, "train", start=args.vehicle_start, end=args.vehicle_end)`로, 검증 차량은 `"validate"`로 읽는다.
- [x] 차량 로딩은 `load_vehicle`로 한다. LSTM `vin_1` 로딩과 디버그 모드의 "1대만" 동작은 그대로 둔다.
- [x] 실데이터: train이 `after_bugs`와 bit 단위로 일치해야 한다. 결과를 Comments에 남긴다.

## Comments

**2026-10-15 완료**

- `run_config = RunConfig.from_mapping({**vars(args), "no_use_dx": not use_dx_in_forward})`를 디버그 감지 직후 한 번 만든다. `PREPROCESSING`·`SKIP_CHARGE_READY`(sim_config 기록용)도 여기서 꺼내고, 모델 생성(`build_model_pair`)도 이것을 쓴다.
- 분할 경로 규칙은 그대로다. `--dataset-name`이 있으면 `./datasets/{이름}`(없으면 `np.load`에서 실패), 없으면 `.`이다. 이 경로를 `split_ids`에 넘기고, 학습 차량 범위도 `split_ids`의 `start`·`end`로 적용한다.
- 학습·검증 차량은 `load_vehicle(args.source_data_dir, run_config, id)`로 읽는다. LSTM `vin_1` 로딩, `vehicle_idxes`(문자열 ID), 디버그 모드의 "1대만" 동작은 그대로 두었다.
- 실데이터(기준선 태그 `vd03`): `after_bugs`와 14개 기록이 모두 bit 단위로 일치한다. 체크포인트, 관리도, sim_config가 포함된다.
- 검증하지 못한 경로: 기준선 train 대상은 `--no-validation`이라 검증 차량 로딩은 실데이터로 돌리지 않았다. 학습 차량과 같은 `load_vehicle` 호출이다. 디버그 모드도 실행하지 않았다.
