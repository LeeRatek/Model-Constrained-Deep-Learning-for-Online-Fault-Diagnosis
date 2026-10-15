"""차량 단위 ROC 평가.

차량별 최대 CI와 고장 여부로, 임계값 θ를 0 ~ 999.9 (0.1 간격)로 바꿔 가며 ROC 곡선, AUC,
최적 임계값(Youden J 최대점, (0,1) 최근접점)을 구한다. 용어는 GLOSSARY.md를 따른다.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

# 임계값 grid. list를 거쳐 만드는 방식까지 기존 스크립트와 같게 둔다 (CSV에 0.30000000000000004 등이 그대로 찍힌다)
_THRESHOLDS = np.asarray(list(np.arange(0, 1000, 0.1)), dtype=float)


@dataclass(frozen=True)
class RocResult:
    fpr: np.ndarray
    tpr: np.ndarray
    auc: float
    best: dict  # Youden J 최대점: threshold, fpr, tpr, index, youden_j
    closest_to_01: dict  # (0,1) 최근접점: threshold, fpr, tpr, index, distance


def evaluate(max_ci, is_fault, reverse: bool = False) -> RocResult:
    """차량별 최대 CI와 고장 여부로 ROC를 평가한다. 차량 순서는 넘긴 순서를 그대로 쓴다.

    기본: 고장 차량이 양성이고, 최대 CI > θ이면 양성으로 판정한다.
    reverse: 정상 차량이 양성이고, 최대 CI < θ이면 양성으로 판정한다 (--use-reverse-pos-neg).
    """
    predict = np.zeros((len(max_ci), len(_THRESHOLDS)), dtype=np.int8)
    y_true = np.zeros(len(max_ci), dtype=np.int8)
    for i, (score, fault) in enumerate(zip(max_ci, is_fault)):
        hit = (score < _THRESHOLDS) if reverse else (score > _THRESHOLDS)
        predict[i, :] = hit.astype(np.int8)
        y_true[i] = np.int8(bool(fault) != reverse)  # reverse이면 정상이 양성(1)

    roc = _roc_from_threshold_matrix(y_true, predict, _THRESHOLDS)
    opt = _optimal_thresholds(
        roc["fpr"], roc["tpr"], roc["thresholds"], candidate_idx=roc["candidate_idx"]
    )
    return RocResult(
        fpr=roc["fpr"],
        tpr=roc["tpr"],
        auc=roc["auc"],
        best=opt["best"],
        closest_to_01=opt["closest_to_01"],
    )


def _roc_from_threshold_matrix(y_true, predict_results, predict_thresholds):
    """threshold grid 기반 ROC point들과 AUC를 계산.

    Returns:
        dict: fpr, tpr, thresholds(thr_arr), auc, candidate_idx(finite threshold index), pos_count, neg_count
    """
    from sklearn import metrics as sk_metrics

    y_true = np.asarray(y_true)
    if y_true.ndim != 1:
        y_true = y_true.reshape(-1)

    pos_count = int(np.sum(y_true == 1))
    neg_count = int(np.sum(y_true == 0))
    if pos_count == 0 or neg_count == 0:
        raise ValueError(
            f"ROC 계산을 위해 정상/고장 샘플이 모두 필요합니다. pos={pos_count}, neg={neg_count}"
        )

    predict_results = np.asarray(predict_results)
    predict_thresholds = np.asarray(predict_thresholds, dtype=float)
    if predict_results.ndim != 2:
        raise ValueError(
            f"predict_results는 2D여야 합니다. got ndim={predict_results.ndim}"
        )
    if predict_results.shape[1] != predict_thresholds.shape[0]:
        raise ValueError(
            f"predict_results.shape[1]({predict_results.shape[1]}) != len(predict_thresholds)({predict_thresholds.shape[0]})"
        )

    # threshold를 큰 값 -> 작은 값으로 낮추면, 예측 Positive가 늘어나며 ROC가 (0,0) -> (1,1)로 진행
    fpr_list = [0.0]
    tpr_list = [0.0]
    thr_list = [float("inf")]

    for j in range(len(predict_thresholds) - 1, -1, -1):
        y_pred = predict_results[:, j].astype(bool)
        tp = int(np.sum(y_pred & (y_true == 1)))
        fp = int(np.sum(y_pred & (y_true == 0)))
        tpr_list.append(tp / pos_count)
        fpr_list.append(fp / neg_count)
        thr_list.append(float(predict_thresholds[j]))

    fpr_list.append(1.0)
    tpr_list.append(1.0)
    thr_list.append(float("-inf"))

    fpr = np.asarray(fpr_list, dtype=float)
    tpr = np.asarray(tpr_list, dtype=float)
    thr_arr = np.asarray(thr_list, dtype=float)

    # FPR, 같으면 TPR 오름차순으로 정렬해 ROC 곡선을 단조 계단으로 만든다 (stable 정렬).
    # 기본 판정(score > thr)은 이미 이 순서라 그대로이고, reverse 판정(score < thr)은 역순 입력이 바로잡힌다.
    sort_idx = np.lexsort((tpr, fpr))
    fpr = fpr[sort_idx]
    tpr = tpr[sort_idx]
    thr_arr = thr_arr[sort_idx]

    auc_val = float(sk_metrics.auc(fpr, tpr))

    finite_mask = np.isfinite(thr_arr)
    candidate_idx = np.where(finite_mask)[0]

    return {
        "fpr": fpr,
        "tpr": tpr,
        "thresholds": thr_arr,
        "auc": auc_val,
        "candidate_idx": candidate_idx,
        "pos_count": pos_count,
        "neg_count": neg_count,
    }


def _optimal_thresholds(fpr, tpr, thresholds, *, candidate_idx=None):
    """ROC curve 상에서 최적 임계값(Youden J, (0,1) 최소거리)을 계산."""
    fpr = np.asarray(fpr, dtype=float)
    tpr = np.asarray(tpr, dtype=float)
    thr_arr = np.asarray(thresholds, dtype=float)
    if fpr.shape != tpr.shape or fpr.shape != thr_arr.shape:
        raise ValueError("fpr/tpr/thresholds는 같은 shape 여야 합니다.")

    if candidate_idx is None:
        candidate_idx = np.where(np.isfinite(thr_arr))[0]
    else:
        candidate_idx = np.asarray(candidate_idx, dtype=int)

    j_scores = tpr - fpr
    if candidate_idx.size > 0:
        best_idx = int(candidate_idx[np.argmax(j_scores[candidate_idx])])
    else:
        best_idx = int(np.argmax(j_scores))

    dist2 = (fpr**2) + ((1.0 - tpr) ** 2)
    if candidate_idx.size > 0:
        closest_idx = int(candidate_idx[np.argmin(dist2[candidate_idx])])
    else:
        closest_idx = int(np.argmin(dist2))

    ## ================ Youden J 기준 최적점 ================
    best = {
        "threshold": float(thr_arr[best_idx]),
        "fpr": float(fpr[best_idx]),
        "tpr": float(tpr[best_idx]),
        "index": best_idx,
        "youden_j": float(j_scores[best_idx]),
    }

    ## ================ (0,1) 최소거리 기준 최적점 ================
    closest_to_01 = {
        "threshold": float(thr_arr[closest_idx]),
        "fpr": float(fpr[closest_idx]),
        "tpr": float(tpr[closest_idx]),
        "index": closest_idx,
        "distance": float(np.sqrt(dist2[closest_idx])),
    }

    return {"best": best, "closest_to_01": closest_to_01}
