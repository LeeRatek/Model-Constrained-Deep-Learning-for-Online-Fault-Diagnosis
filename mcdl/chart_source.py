"""평가할 모델 쌍의 관리도를 어디서 가져올지 정하는 코드.

왜 필요한가
    관리도(PCA 모델 + 관리한계, mcdl/diagnosis.py)는 정상 학습 차량들의 진단 특징으로 만든다. 그런데 진단 특징은
    어떤 U-model·X-model 체크포인트로 재구성했는지에 따라 달라지므로, 관리도도 모델 쌍마다 따로 있어야 한다.
    Train_.py는 학습이 끝난 최종 모델 쌍(-1/-1)의 관리도만 pca_arrays.npz로 저장해 둔다.

가져오는 순서 (chart_for_pair)
    1. 캐시: Test_multi_models.py가 Run 폴더의 temp_cache/에 저장해 둔 학습 진단 특징이 있고, 계산 조건이
       아래 3번과 같으면 그 특징으로 관리도를 만든다(몇 초).
    2. 저장된 파일: 최종 모델 쌍(-1/-1)이면 학습 때 저장한 pca_arrays.npz를 읽는다.
    3. 다시 계산: 그 밖에는 Run의 학습 차량 원본을 한 대씩 읽어 학습 진단 특징을 계산하고 관리도를 만든다(수 분).

누가 쓰나
    Test_.py, Test_tsne.py(chart_for_pair), Test_multi_models.py(temp_cache 파일 쓰기·읽기).
"""

from __future__ import annotations

import pickle

from mcdl.diagnosis import ControlChart, combine_signal_features
from mcdl.reconstruction import training_features
from mcdl.vehicles import load_vehicle, split_ids

# 관리도를 다시 계산할 때 한 번에 추론할 시점 수(batch 크기). batch 크기가 결과의 소수점 끝자리에 영향을 주므로,
# Test_multi_models.py의 기본값과 같게 두어 두 스크립트가 같은 관리도를 얻게 한다.
RECOMPUTE_BATCH_SIZE = 65536


def chart_for_pair(run, u: int, x: int, *, source_data_dir) -> ControlChart:
    """모델 쌍(U 체크포인트 번호 u, X 체크포인트 번호 x)을 평가할 관리도를 돌려준다.

    위 "가져오는 순서"대로 캐시 → 저장된 파일(-1/-1일 때) → 다시 계산 순으로 찾는다.
    run은 mcdl/run.py의 Run, source_data_dir은 원본 데이터 폴더(다시 계산할 때만 쓴다).
    쓰는 곳: Test_.py, Test_tsne.py.
    """
    conditions = _recompute_conditions(run)
    u_path = run.feature_cache_path("U", u, **conditions)
    x_path = run.feature_cache_path("X", x, **conditions)

    if u_path.exists() and x_path.exists():
        try:
            print("Loading cached training features for PCA setup:")
            print(f"  U: {u_path}")
            print(f"  X: {x_path}")
            u_feat = load_cached_features(run, "U", u, conditions=conditions)
            x_feat = load_cached_features(run, "X", x, conditions=conditions)
            print("Computing PCA directly from cached training features...")
            return ControlChart.fit(combine_signal_features(u_feat, x_feat))
        except Exception as e:
            print(f"Error processing cached features: {e}.")

    if u == -1 and x == -1:
        print("Cached features not found or error. Falling back to saved 'pca_arrays.npz'.")
        return ControlChart.load(run.path)

    return _recompute(run, u, x, source_data_dir)


def save_cached_features(run, model: str, number: int, features, *, conditions):
    """체크포인트 하나의 학습 진단 특징(그 모델 몫 3개)을 temp_cache/ 파일로 저장하고 경로를 돌려준다.

    model은 "U"/"X", number는 체크포인트 번호, conditions는 계산 조건(파일 이름에 들어간다).
    파일 형식은 pickle(파이썬 객체를 그대로 저장하는 형식)이다.
    쓰는 곳: Test_multi_models.py (--cache-strategy disk일 때).
    """
    path = run.feature_cache_path(model, number, **conditions)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(features, f)
    return path


def load_cached_features(run, model: str, number: int, *, conditions):
    """save_cached_features로 저장한 학습 진단 특징(pandas Series 3개)을 읽는다.

    쓰는 곳: Test_multi_models.py, 위 chart_for_pair.
    """
    with open(run.feature_cache_path(model, number, **conditions), "rb") as f:
        return pickle.load(f)


def _recompute_conditions(run) -> dict:
    """chart_for_pair·_recompute 안에서 쓴다. 관리도를 다시 계산할 때의 계산 조건을 돌려준다.

    오차 절댓값 여부는 학습 설정을 따르고, float32 변환은 하지 않으며, batch 크기는 65536이다.
    캐시는 이 조건과 똑같이 만든 것만 읽어서, 캐시를 쓰든 다시 계산하든 같은 관리도가 나온다.
    """
    return {
        "abs_err": not run.config.no_abs_err,
        "float32": False,
        "batch_size": RECOMPUTE_BATCH_SIZE,
    }


def _recompute(run, u: int, x: int, source_data_dir) -> ControlChart:
    """chart_for_pair 안에서 쓴다. Run의 학습 차량 원본으로 학습 진단 특징을 계산해 관리도를 새로 만든다.

    학습에 썼던 차량 목록(sim_config의 분할과 학습 차량 범위)을 다시 읽어, 차량을 한 대씩 불러오며 계산한다
    (모든 차량을 한꺼번에 메모리에 올리지 않는다). 기본 분할 209대 기준 1~2분 걸린다.
    """
    config = run.config
    net, netx = run.load_model_pair(u, x)
    train_list = split_ids(
        config.split_dir, config.battery_type, "train", start=config.vehicle_start, end=config.vehicle_end
    )

    def vehicles():
        """학습 차량을 한 대씩 읽어 (U 텐서, X 텐서)를 차례로 내준다. 10대마다 진행 상황을 출력한다."""
        for count, vehicle_id in enumerate(train_list, start=1):
            if count % 10 == 0:
                print(f"Processing vehicle {count}/{len(train_list)}")
            yield load_vehicle(source_data_dir, config, vehicle_id)

    print(f"Recomputing the control chart for (u={u}, x={x}) from {len(train_list)} training vehicles...")
    features = training_features(
        net,
        netx,
        vehicles(),
        batch_size=RECOMPUTE_BATCH_SIZE,
        abs_err=_recompute_conditions(run)["abs_err"],
    )
    return ControlChart.fit(features)
