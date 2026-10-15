"""진단 특징 → 관리도 → CI → ROC 계산을 fixture로 고정한 테스트.

fixture(tests/fixtures/diagnosis_golden.npz)는 .scratch/control-chart 티켓 01 시점의 Function_ 구현으로
합성 오차(tests/synthetic_errors.py)에서 만들었다. 값은 bit 단위로 비교한다.
reverse 판정 ROC 항목(roc_reverse_*)만 known-bugs 09 수정(FPR·TPR stable 정렬) 뒤의 mcdl.roc로 다시 만들었다.
생성 환경: Windows 11, Python 3.12.13, numpy 2.3.5, pandas 2.3.3, scipy 1.16.3.
다른 플랫폼에서는 ULP 수준 차이로 실패할 수 있다.
"""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from mcdl.diagnosis import (
    ControlChart,
    combine_signal_features,
    diagnosis_features,
    signal_features,
)
from mcdl.roc import evaluate
from synthetic_errors import make_vehicle_errors

GOLDEN = dict(np.load(Path(__file__).parent / "fixtures" / "diagnosis_golden.npz"))
TRAIN, TEST_NORMAL, TEST_FAULT = make_vehicle_errors()
TEST_VEHICLES = TEST_NORMAL + TEST_FAULT


def assert_same(actual, key):
    expected = GOLDEN[key]
    actual = np.asarray(actual)
    assert actual.dtype == expected.dtype, key
    assert actual.shape == expected.shape, key
    assert np.array_equal(actual, expected), key


def train_features():
    """Train_ 경로: 차량별 진단 특징을 이어 붙인다."""
    return pd.concat(
        [diagnosis_features(eu, ex) for eu, ex in TRAIN], axis=0, ignore_index=True
    )


# ---------------------------------------------------------------- 진단 특징


def test_diagnosis_features():
    assert_same(train_features().to_numpy(), "train_features")
    for j, (eu, ex) in enumerate(TEST_VEHICLES):
        assert_same(diagnosis_features(eu, ex).to_numpy(), f"test_features_{j}")


def test_signal_cache_path_matches_vehicle_path():
    """Test_multi_models 경로: 신호별 특징을 차량마다 구해 신호별로 이어 붙인 뒤 묶는다."""
    per_signal = {}
    for signal, index in (("U", 0), ("X", 1)):
        per_vehicle = [signal_features(signal, errors[index]) for errors in TRAIN]
        per_signal[signal] = tuple(
            pd.concat([feats[i] for feats in per_vehicle], ignore_index=True) for i in range(3)
        )

    combined = combine_signal_features(per_signal["U"], per_signal["X"])

    pd.testing.assert_frame_equal(combined, train_features(), check_exact=True)


def test_features_require_2d_errors():
    with pytest.raises(ValueError):
        signal_features("U", np.zeros(5))


# ---------------------------------------------------------------- 관리도


def test_fitted_chart_scores():
    chart = ControlChart.fit(train_features())

    assert_same(chart.t2_limit, "pca_T_95_limit")
    assert_same(chart.spe_limit, "pca_SPE_95_limit")
    max_ci = []
    for j, (eu, ex) in enumerate(TEST_VEHICLES):
        features = diagnosis_features(eu, ex)
        t2, spe, ci = chart.score(features)
        assert_same(t2, f"t2_{j}")
        assert_same(spe, f"spe_{j}")
        assert_same(ci, f"ci_{j}")
        max_ci.append(chart.max_ci(features))
    assert_same(max_ci, "max_ci")


def test_saved_chart_keeps_file_format(tmp_path):
    features = train_features()
    ControlChart.fit(features).save(f"{tmp_path}/", features)

    with np.load(tmp_path / "pca_arrays.npz") as arrays:
        for name in arrays.files:
            assert_same(arrays[name], f"pca_{name}")
    data_nor = pd.read_csv(tmp_path / "pca_data_nor.csv", index_col=0)
    np.testing.assert_allclose(data_nor.to_numpy(), GOLDEN["pca_data_nor"], rtol=0, atol=1e-15)
    assert (tmp_path / "pca_manifest.json").exists()


def test_loaded_chart_scores(tmp_path):
    features = train_features()
    ControlChart.fit(features).save(f"{tmp_path}/", features)

    chart = ControlChart.load(tmp_path)

    assert_same(chart.t2_limit, "pca_T_95_limit")
    assert_same(chart.spe_limit, "pca_SPE_95_limit")
    for j, (eu, ex) in enumerate(TEST_VEHICLES):
        assert_same(chart.score(diagnosis_features(eu, ex))[2], f"ci_loaded_{j}")


def test_load_requires_manifest(tmp_path):
    features = train_features()
    ControlChart.fit(features).save(f"{tmp_path}/", features)
    (tmp_path / "pca_manifest.json").unlink()

    with pytest.raises(FileNotFoundError):
        ControlChart.load(tmp_path)


# ---------------------------------------------------------------- ROC


@pytest.mark.parametrize("tag, reverse", [("roc", False), ("roc_reverse", True)])
def test_roc(tag, reverse):
    max_ci = list(GOLDEN["max_ci"])
    n_normal = len(TEST_NORMAL)
    normal = [(score, False) for score in max_ci[:n_normal]]
    fault = [(score, True) for score in max_ci[n_normal:]]
    # Test_multi_models는 reverse일 때 고장 차량을 먼저 넘긴다
    vehicles = fault + normal if reverse else normal + fault

    result = evaluate([s for s, _ in vehicles], [f for _, f in vehicles], reverse=reverse)

    assert_same(result.fpr, f"{tag}_fpr")
    assert_same(result.tpr, f"{tag}_tpr")
    assert_same(result.auc, f"{tag}_auc")
    b, c = result.best, result.closest_to_01
    assert_same([b["threshold"], b["fpr"], b["tpr"], b["youden_j"]], f"{tag}_best")
    assert_same(b["index"], f"{tag}_best_index")
    assert_same([c["threshold"], c["fpr"], c["tpr"], c["distance"]], f"{tag}_closest")
    assert_same(c["index"], f"{tag}_closest_index")


def test_roc_needs_both_normal_and_fault_vehicles():
    with pytest.raises(ValueError):
        evaluate([1.0, 2.0], [False, False])
