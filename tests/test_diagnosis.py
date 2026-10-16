"""진단 계산(진단 특징 → 관리도 → CI → ROC)의 결과가 바뀌지 않았는지 확인하는 테스트.

실제 데이터 대신 합성 오차(synthetic_errors.py)를 넣어 계산하고, 그 결과를 미리 저장해 둔 기대값 파일
tests/fixtures/diagnosis_golden.npz와 비교한다. 소수점 끝자리까지 완전히 같아야 통과한다(bit 단위 비교).
기대값 파일을 만드는 방법과 fixture·golden이라는 말의 뜻은 make_diagnosis_golden.py 맨 위 설명에 있다.

기대값은 어디서 왔나
    리팩터링 전 코드(지금은 mcdl/ 여러 파일로 나뉜 옛 Function_.py)로 만든 값이다. 그래서 이 테스트가 통과하면
    지금 코드가 예전과 같은 계산을 한다는 뜻이다. 단, 반대 방향 ROC 항목(roc_reverse_*)만은 FPR·TPR 정렬
    버그를 고친 뒤(.scratch/known-bugs/issues/09-reverse-roc-unstable-sort.md) 지금 mcdl/roc.py로 다시 만들었다.

주의: 기대값은 Windows 11, Python 3.12.13, numpy 2.3.5, pandas 2.3.3, scipy 1.16.3에서 만들었다.
다른 운영체제나 라이브러리 버전에서는 마지막 자릿수만큼(ULP: 실수 하나가 나타낼 수 있는 가장 작은 간격)
달라져 실패할 수 있다. 그때는 계산이 틀렸다기보다 환경 차이일 가능성이 크다.
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

GOLDEN = dict(np.load(Path(__file__).parent / "fixtures" / "diagnosis_golden.npz"))  # 기대값 {이름: 배열}
TRAIN, TEST_NORMAL, TEST_FAULT = make_vehicle_errors()
TEST_VEHICLES = TEST_NORMAL + TEST_FAULT


def assert_same(actual, key):
    """actual이 기대값 GOLDEN[key]와 자료형·모양·값까지 완전히 같은지 확인한다. 쓰는 곳: 이 파일의 테스트들."""
    expected = GOLDEN[key]
    actual = np.asarray(actual)
    assert actual.dtype == expected.dtype, key
    assert actual.shape == expected.shape, key
    assert np.array_equal(actual, expected), key


def train_features():
    """학습 차량 4대의 진단 특징을 차량별로 구해 이어 붙인다(Train_.py가 관리도를 만들 때와 같은 방식)."""
    return pd.concat(
        [diagnosis_features(eu, ex) for eu, ex in TRAIN], axis=0, ignore_index=True
    )


# ---------------------------------------------------------------- 진단 특징


def test_diagnosis_features():
    """학습·테스트 차량의 진단 특징(diagnosis_features)이 기대값과 같은지 본다."""
    assert_same(train_features().to_numpy(), "train_features")
    for j, (eu, ex) in enumerate(TEST_VEHICLES):
        assert_same(diagnosis_features(eu, ex).to_numpy(), f"test_features_{j}")


def test_signal_cache_path_matches_vehicle_path():
    """신호(U, X)별 특징을 따로 구해 나중에 묶어도(Test_multi_models.py가 쓰는 캐시 방식),
    차량별로 한 번에 구한 특징(train_features)과 완전히 같은지 본다."""
    per_signal = {}
    for signal, index in (("U", 0), ("X", 1)):
        per_vehicle = [signal_features(signal, errors[index]) for errors in TRAIN]
        per_signal[signal] = tuple(
            pd.concat([feats[i] for feats in per_vehicle], ignore_index=True) for i in range(3)
        )

    combined = combine_signal_features(per_signal["U"], per_signal["X"])

    pd.testing.assert_frame_equal(combined, train_features(), check_exact=True)


def test_features_require_2d_errors():
    """오차가 (시점 × 셀) 2차원 배열이 아니면 ValueError로 바로 알려 주는지 본다."""
    with pytest.raises(ValueError):
        signal_features("U", np.zeros(5))


# ---------------------------------------------------------------- 관리도


def test_fitted_chart_scores():
    """관리도를 학습한 직후의 95% 관리한계와, 테스트 차량별 T²·SPE·CI·최대 CI가 기대값과 같은지 본다."""
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
    """관리도를 저장한 파일(pca_arrays.npz, pca_data_nor.csv, pca_manifest.json)의 내용과 이름이 그대로인지 본다.

    tmp_path는 pytest가 테스트마다 만들어 주는 빈 임시 폴더다(테스트가 끝나면 지워도 되는 곳).
    """
    features = train_features()
    ControlChart.fit(features).save(f"{tmp_path}/", features)

    with np.load(tmp_path / "pca_arrays.npz") as arrays:
        for name in arrays.files:
            assert_same(arrays[name], f"pca_{name}")
    data_nor = pd.read_csv(tmp_path / "pca_data_nor.csv", index_col=0)
    np.testing.assert_allclose(data_nor.to_numpy(), GOLDEN["pca_data_nor"], rtol=0, atol=1e-15)
    assert (tmp_path / "pca_manifest.json").exists()


def test_loaded_chart_scores(tmp_path):
    """저장했다가 다시 불러온 관리도로 계산한 관리한계·CI가 기대값과 같은지 본다(평가 스크립트가 쓰는 경로)."""
    features = train_features()
    ControlChart.fit(features).save(f"{tmp_path}/", features)

    chart = ControlChart.load(tmp_path)

    assert_same(chart.t2_limit, "pca_T_95_limit")
    assert_same(chart.spe_limit, "pca_SPE_95_limit")
    for j, (eu, ex) in enumerate(TEST_VEHICLES):
        assert_same(chart.score(diagnosis_features(eu, ex))[2], f"ci_loaded_{j}")


def test_load_requires_manifest(tmp_path):
    """관리도 목록 파일(pca_manifest.json)이 없으면 불러오기가 FileNotFoundError로 멈추는지 본다."""
    features = train_features()
    ControlChart.fit(features).save(f"{tmp_path}/", features)
    (tmp_path / "pca_manifest.json").unlink()

    with pytest.raises(FileNotFoundError):
        ControlChart.load(tmp_path)


# ---------------------------------------------------------------- ROC


@pytest.mark.parametrize("tag, reverse", [("roc", False), ("roc_reverse", True)])
def test_roc(tag, reverse):
    """차량별 최대 CI로 계산한 ROC(FPR·TPR·AUC·최적 임계값)가 기대값과 같은지 본다.

    parametrize는 같은 테스트를 인자만 바꿔 여러 번 돌리는 pytest 기능이다. 여기서는 정상 방향(reverse=False)과
    반대 방향(reverse=True, 정상 차량을 양성으로 보고 최대 CI가 임계값보다 작으면 양성으로 판정) 두 번 돈다.
    """
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
    """정상 차량이나 고장 차량 중 한쪽만 있으면 ROC를 계산할 수 없으므로 ValueError가 나는지 본다."""
    with pytest.raises(ValueError):
        evaluate([1.0, 2.0], [False, False])
