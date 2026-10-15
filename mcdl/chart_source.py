"""모델 쌍의 관리도 공급원.

Run과 모델 쌍(U·X 체크포인트 번호)을 받아 관리도를 돌려준다. 출처는 다음 순서로 정한다.

1. 다시 계산할 때와 같은 조건으로 만든 학습 진단 특징 캐시(temp_cache)가 있으면 그것으로 관리도를 맞춘다.
2. 최종 모델 쌍(-1/-1)이면 학습 때 저장한 관리도(pca_arrays.npz)를 읽는다.
3. 그 밖에는 Run의 학습 차량 원본으로 학습 진단 특징을 다시 계산해 관리도를 맞춘다.

temp_cache 파일(신호별 특징 3개의 pickle)을 쓰고 읽는 형식도 여기서 정한다. 용어는 GLOSSARY.md를 따른다.
"""

from __future__ import annotations

import pickle

from mcdl.diagnosis import ControlChart, combine_signal_features
from mcdl.reconstruction import training_features
from mcdl.vehicles import load_vehicle, split_ids

# 관리도를 다시 계산할 때의 추론 batch 크기. 결과의 일부이므로 Test_multi_models 기본값과 맞춘다.
RECOMPUTE_BATCH_SIZE = 65536


def chart_for_pair(run, u: int, x: int, *, source_data_dir) -> ControlChart:
    """모델 쌍의 관리도. 같은 조건 캐시 → 최종 쌍이면 저장된 관리도 → 그 밖엔 학습 데이터로 재계산."""
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
    """신호별 학습 진단 특징(Series 3개)을 temp_cache에 저장하고 경로를 돌려준다."""
    path = run.feature_cache_path(model, number, **conditions)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(features, f)
    return path


def load_cached_features(run, model: str, number: int, *, conditions):
    """temp_cache에서 신호별 학습 진단 특징(Series 3개)을 읽는다."""
    with open(run.feature_cache_path(model, number, **conditions), "rb") as f:
        return pickle.load(f)


def _recompute_conditions(run) -> dict:
    """다시 계산할 때의 조건. 이 조건으로 만든 캐시만 읽는다."""
    return {
        "abs_err": not run.config.no_abs_err,
        "float32": False,
        "batch_size": RECOMPUTE_BATCH_SIZE,
    }


def _recompute(run, u: int, x: int, source_data_dir) -> ControlChart:
    config = run.config
    net, netx = run.load_model_pair(u, x)
    train_list = split_ids(
        config.split_dir, config.battery_type, "train", start=config.vehicle_start, end=config.vehicle_end
    )

    def vehicles():
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
