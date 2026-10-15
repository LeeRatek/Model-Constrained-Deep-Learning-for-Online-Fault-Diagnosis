# 02: Test_multi_models의 학습·테스트 로딩 전환

**What to build:** Test_multi_models의 학습 차량 로딩(학습 차량 범위, `.pt` 캐시)과 테스트 차량 로딩(`.pt` 캐시)을 `split_ids`, `load_training_vehicles`, `load_test_vehicles`로 바꾼다. 기존 41 GB 캐시를 그대로 다시 쓴다.

**Blocked by:** 01 (차량 데이터 module)

**Status:** resolved

- [x] 학습 차량: `split_ids(…, "train", start, end)`와 `load_training_vehicles`를 쓴다.
- [x] 테스트 차량: 분할 ID는 `split_ids`로 읽는다. test, `--use-validate`이면 validate, fault를 읽고, `--use-reverse-pos-neg`의 순서도 지금과 같다. 차량은 `load_test_vehicles`로 읽는다.
- [x] 캐시 해시 dict를 만들고 `.pt`를 저장·로드하던 코드를 지운다.
- [x] 실데이터: multi가 `after_bugs`와 bit 단위로 일치해야 한다. 기존 캐시를 다시 쓰는지는 "Found cached" 메시지와 소요 시간으로 확인한다. 결과를 Comments에 남긴다.

## Comments

**2026-10-15 완료**

- 분할: test, validate, fault를 `split_ids`로 읽는다. `--use-validate`와 `--use-reverse-pos-neg`에 따른 `test_list` 순서는 그대로다. 고장 판정(`vehicle_ids is fault_list`)도 같은 목록 객체를 쓰므로 그대로 동작한다.
- 학습: `split_ids(..., "train", start=vehicle_start, end=vehicle_end)` + `load_training_vehicles`. 테스트: `test_list`를 순서대로 펼친 목록으로 `load_test_vehicles`를 부른다.
- 해시 dict, `.pt` 저장·로드 코드와 이로 인해 쓰이지 않게 된 `hashlib`·`json` import, `normalize_dx_flag`·`normalize_val` 변수를 지웠다.
- 실데이터(기준선 태그 `vd02`): `after_bugs`와 304개 기록이 모두 bit 단위로 일치한다. 소요 시간은 228.7초(after_bugs는 210.4초)다.
- 캐시 재사용 확인: 두 실행 모두 같은 파일을 "Found cached"로 읽었다.
  - `train_data_list_QAS_015ac371f38644b99d2e0d2c85ec0ba7.pt`(209대, 6,357,186 샘플)
  - `test_data_cache_QAS_528fb02abeb3f5a1f220b9fe6857d592.pt`(100대)
