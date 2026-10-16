# 13: 실행 스크립트의 도움말·경고 문구가 실제 동작과 다르다

**Status:** resolved

**Blocked by:** None (can start immediately)

**현상**
- Test_tsne.py `--save`의 도움말이 "normal lo(CI>thr[0]) 오버레이를 끕니다."였다. `--no-alarm-nlo` 설명을 복사한 것으로, 실제로는 그림을 창 대신 파일로 저장하는 옵션이다.
- Test_multi_models.py는 체크포인트를 찾지 못하면 "Warning: No valid U model files found (excluding -1 and 0)."를 출력했다. 실제로는 -1(최종)과 0번도 찾는다(`Run.checkpoint_numbers`).

**영향:** `--help`와 콘솔 메시지만 틀린다. 계산 결과에는 영향이 없다.

**출처:** 2026-10-16 주석 작업·코드 정리 중 발견.

## Comments

### 2026-10-16 해소 (코드 정리와 함께)

- `--save` 도움말: "그림을 창으로 띄우지 않고 <models-dir>/<models-idx>/results_tsne/tSNE_<차량 ID>.png로 저장합니다."
- 경고: "Warning: No U model checkpoint files found." (X도 같은 형식)
