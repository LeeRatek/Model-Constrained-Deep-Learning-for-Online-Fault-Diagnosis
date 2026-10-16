"""차량 단위 판정 성능(ROC 곡선과 AUC)을 계산하는 코드.

무엇을 하나
    평가 차량마다 판정 점수(최대 CI, mcdl/diagnosis.py)와 실제 고장 여부가 있다. 임계값 θ를 0부터 999.9까지
    0.1 간격으로 바꿔 가며 "최대 CI > θ이면 고장"으로 판정하고, 임계값마다
      TPR(고장 차량 중 고장으로 맞힌 비율), FPR(정상 차량 중 잘못 고장으로 판정한 비율)
    을 구해 ROC 곡선을 만든다. 곡선 아래 면적(AUC)이 1에 가까울수록 정상과 고장을 잘 가른다.
    추천 임계값 두 개도 고른다.
      - Youden J 최대점: TPR − FPR이 가장 큰 임계값
      - (0,1) 최근접점: 곡선 위에서 완벽한 판정점(FPR 0, TPR 1)에 가장 가까운 임계값

누가 쓰나
    Test_.py, Test_multi_models.py (결과는 콘솔, ROC 그림, AUC_summary CSV에 남는다).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn import metrics

# 검사할 임계값 목록: 0, 0.1, 0.2, …, 999.9 (10,000개).
# list를 거쳐 만드는 방식까지 예전 스크립트와 같게 두어, 결과 CSV에 찍히는 값(0.30000000000000004 등)이 그대로다.
_THRESHOLDS = np.asarray(list(np.arange(0, 1000, 0.1)), dtype=float)


@dataclass(frozen=True)
class RocResult:
    """evaluate의 결과 묶음."""

    fpr: np.ndarray  # ROC 곡선의 FPR 값들 (오름차순)
    tpr: np.ndarray  # ROC 곡선의 TPR 값들
    auc: float  # ROC 곡선 아래 면적
    best: dict  # Youden J 최대점: threshold, fpr, tpr, index, youden_j
    closest_to_01: dict  # (0,1) 최근접점: threshold, fpr, tpr, index, distance


def evaluate(max_ci, is_fault, reverse: bool = False) -> RocResult:
    """차량별 최대 CI 목록과 고장 여부 목록으로 ROC 곡선, AUC, 추천 임계값을 계산한다.

    max_ci와 is_fault는 같은 차량 순서의 목록이다. 정상 차량과 고장 차량이 모두 있어야 한다(없으면 ValueError).
    - 기본: 고장 차량을 "양성"으로 보고, 최대 CI > θ이면 양성(고장)으로 판정한다.
    - reverse=True: 정상 차량을 양성으로 보고, 최대 CI < θ이면 양성으로 판정한다
      (Test_multi_models.py --use-reverse-pos-neg).
    쓰는 곳: Test_.py, Test_multi_models.py.
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
    """evaluate 안에서 쓴다. 임계값별 판정 결과 표로 ROC 곡선의 점들과 AUC를 계산한다.

    y_true는 차량별 정답(양성 1, 음성 0), predict_results는 (차량 수 × 임계값 수) 판정 결과(양성 1)다.
    곡선 양 끝에 (0,0)과 (1,1)을 더하고, FPR 오름차순(같으면 TPR 오름차순)으로 정렬해 계단 모양 곡선을 만든다.

    Returns:
        dict: fpr, tpr, thresholds(곡선 점마다의 임계값), auc, candidate_idx(양 끝 점을 뺀 실제 임계값의 위치)
    """
    pos_count = int(np.sum(y_true == 1))
    neg_count = int(np.sum(y_true == 0))
    if pos_count == 0 or neg_count == 0:
        raise ValueError(
            f"ROC 계산을 위해 정상/고장 샘플이 모두 필요합니다. pos={pos_count}, neg={neg_count}"
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

    auc_val = float(metrics.auc(fpr, tpr))

    return {
        "fpr": fpr,
        "tpr": tpr,
        "thresholds": thr_arr,
        "auc": auc_val,
        "candidate_idx": np.where(np.isfinite(thr_arr))[0],
    }


def _optimal_thresholds(fpr, tpr, thr_arr, *, candidate_idx):
    """evaluate 안에서 쓴다. ROC 곡선 위에서 추천 임계값 두 개를 고른다.

    Youden J(TPR − FPR)가 가장 큰 점과, (FPR 0, TPR 1)까지 거리가 가장 짧은 점이다. 같은 값이 여럿이면
    앞쪽(작은 FPR) 점을 고른다. candidate_idx(실제 임계값의 위치)에서만 골라, 곡선 양 끝의 가상 점
    (임계값 ±무한대)은 빼고 고른다.
    """
    j_scores = tpr - fpr
    best_idx = int(candidate_idx[np.argmax(j_scores[candidate_idx])])

    dist2 = (fpr**2) + ((1.0 - tpr) ** 2)
    closest_idx = int(candidate_idx[np.argmin(dist2[candidate_idx])])

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
