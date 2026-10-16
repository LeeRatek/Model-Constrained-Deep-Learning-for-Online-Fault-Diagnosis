"""시각화 스크립트 (t-SNE + CI): 정상 기준 차량과 대상 차량의 데이터 분포를 2차원 그림으로 비교한다.

t-SNE는 셀 220개 값(셀 전압 + 셀 SOC)처럼 차원이 높은 데이터를, 비슷한 시점끼리 가깝게 2차원 평면에 펼쳐
보여 주는 시각화 방법이다. 실제 값과 모델의 재구성 값을 각각 그려, 고장 차량이 정상 차량과 어떻게 다르게
놓이는지 본다. 세 번째 칸에는 대상 차량의 CI 시계열과 임계값 선을 그린다.

실행 예
    python Test_tsne.py --models-idx <RUN_ID> --fault-idx 338 --save   # 차량 338 하나
    python Test_tsne.py --models-idx <RUN_ID> --save                   # test + fault 차량 전체를 차례로
    옵션 전체는 README 7.5절에 있다.

하는 일 (main 순서)
    1. Run 설정을 읽고, 모델 쌍과 그 관리도를 가져온다(Test_.py와 같은 방식).
    2. validate 분할 앞쪽 --cnt-max대를 정상 기준 차량으로 한 번 모은다(collect_vehicle_points).
    3. 대상 차량마다 점을 모아 기준 차량 점과 합쳐 t-SNE 그림을 그린다(mcdl/plotting.py).
       --save면 models/<RUN_ID>/results_tsne/tSNE_<차량 ID>.png로 저장하고, 아니면 창으로 띄운다.
"""

import numpy as np
import os
import warnings
import torch

import argparse
import time
from mcdl.run import Run, print_sim_config
from mcdl.reconstruction import reconstruct, vehicle_features
from mcdl.chart_source import chart_for_pair
from mcdl.vehicles import load_vehicle, split_ids
from mcdl.plotting import plot_loss_curve_all, plot_tsne_before_after_ci

# 명령줄 옵션 정의. 실제 읽기(parse_args)는 main 안에서 한다 (import만 할 때는 아무것도 실행되지 않게).
parser = argparse.ArgumentParser(
    description="Run diagnostics plotting with CLI options"
)
parser.add_argument(
    "--models-dir",
    type=str,
    default=(
        "./models"
        if os.environ.get("MODEL_DIR") is None
        else os.environ.get("MODEL_DIR")
    ),
)
parser.add_argument(
    "--models-idx",
    type=str,
    default="260206_084839",  # 260206_084839 or 260209_105821
)
parser.add_argument(
    "--fault-idx",
    type=int,
    default=-1,
    help=("-1: 모든 비정상차량, 그 외: 특정 비정상 차량 인덱스 (예: 347, 348 등)"),
)
parser.add_argument(
    "--source-data-dir",
    type=str,
    default=(
        "./data"
        if os.environ.get("SOURCE_DIR") is None
        else os.environ.get("SOURCE_DIR")
    ),
)

parser.add_argument("--u-model-idx", type=int, default=-1)
parser.add_argument("--x-model-idx", type=int, default=-1)

parser.add_argument("--x-start", type=int, default=20)
parser.add_argument("--x-tick-step", type=int, default=3000)
parser.add_argument("--cnt-max", type=int, default=6)
parser.add_argument("--per-vehicle", type=int, default=500)
parser.add_argument(
    "--thr",
    nargs=2,
    type=float,
    default=[10.4, 87.5],
    metavar=("THR0", "THR1"),
    help="CI threshold 2개 지정. 예) --thr 18.7 28.7",
)


# t-SNE 알람(테두리 오버레이) 표시 옵션 (기본: 전부 표시)
# 약어: n/f=normal/fault, lo/hi=thresholds[0]/thresholds[1]
parser.add_argument(
    "--no-alarm",
    dest="alarm",
    action="store_false",
    default=True,
    help="알람 오버레이를 전부 끕니다.",
)
parser.add_argument(
    "--no-alarm-nlo",
    dest="alarm_nlo",
    action="store_false",
    default=True,
    help="normal lo(CI>thr[0]) 오버레이를 끕니다.",
)
parser.add_argument(
    "--no-alarm-nhi",
    dest="alarm_nhi",
    action="store_false",
    default=True,
    help="normal hi(CI>thr[1]) 오버레이를 끕니다.",
)
parser.add_argument(
    "--no-alarm-flo",
    dest="alarm_flo",
    action="store_false",
    default=True,
    help="fault lo(CI>thr[0]) 오버레이를 끕니다.",
)
parser.add_argument(
    "--no-alarm-fhi",
    dest="alarm_fhi",
    action="store_false",
    default=True,
    help="fault hi(CI>thr[1]) 오버레이를 끕니다.",
)
parser.add_argument(
    "--save",
    action="store_true",
    default=False,
    help="그림을 창으로 띄우지 않고 <models-dir>/<models-idx>/results_tsne/tSNE_<차량 ID>.png로 저장합니다.",
)


def collect_vehicle_points(vehicle_ids, label, *, args, config, net_loaded, netx_loaded, chart):
    """차량들(vehicle_ids)에서 t-SNE에 넣을 점(시점)을 골라 모은다. label은 0(정상 기준) 또는 1(대상 차량).

    차량마다: 원본을 읽고 → 진단 특징으로 CI를 계산하고 → 무작위 args.per_vehicle개 시점에, CI가 임계값(args.thr)을
    넘은 시점(임계값마다 최대 100개)을 더해 점을 고른다. 고른 시점의 실제 셀 값(before)과 재구성 값(after)을 모으고,
    점마다 알람 단계(0 없음, 1 첫 번째 임계값 초과, 2 두 번째 임계값 초과)를 기록한다. 시점은 시드 0으로 골라
    실행할 때마다 같다.
    돌려주는 값: ({"before", "after", "labels", "alarm_levels": 점마다의 목록}, 마지막 차량의 CI 시계열).
    쓰는 곳: main (기준 정상 차량 묶음 한 번, 대상 차량마다 한 번).
    """
    thresholds = args.thr
    points = {"before": [], "after": [], "labels": [], "alarm_levels": []}
    CI_last = None

    for i in vehicle_ids:
        combined_tensor, combined_tensorx = load_vehicle(args.source_data_dir, config, i)

        # CI는 차량 한 대의 진단 특징으로 (차량 전체를 한 번에 추론, 오차의 abs 여부는 학습 진단 특징과 같은 설정을 따른다),
        # t-SNE 점은 재구성 대상 y와 재구성 결과 ŷ로 만든다
        df_data = vehicle_features(
            net_loaded,
            netx_loaded,
            combined_tensor,
            combined_tensorx,
            abs_err=not config.no_abs_err,
        )
        y_recovered, recon_u = reconstruct(net_loaded, combined_tensor)
        y_recovered2, recon_x = reconstruct(netx_loaded, combined_tensorx)
        _, _, CI_array = chart.score(df_data)
        CI_last = CI_array

        alarmed_idx_hi = np.where(CI_array > thresholds[1])[0]
        alarmed_idx_lo = np.where(
            (CI_array > thresholds[0]) & (CI_array <= thresholds[1])
        )[0]

        # Limit valid alarms to 100 to avoid overcrowding the plot
        if len(alarmed_idx_hi) > 100:
            alarmed_idx_hi = np.random.default_rng(0).choice(
                alarmed_idx_hi, 100, replace=False
            )
        if len(alarmed_idx_lo) > 100:
            alarmed_idx_lo = np.random.default_rng(0).choice(
                alarmed_idx_lo, 100, replace=False
            )

        idx = np.random.default_rng(0).choice(
            y_recovered.shape[0],
            size=min(args.per_vehicle, y_recovered.shape[0]),
            replace=False,
        )
        # thresholds[0]/[1] 초과 샘플을 반드시 포함 (현재 차량 기준)
        to_add = []
        if alarmed_idx_lo.size > 0:
            to_add.append(np.asarray(alarmed_idx_lo, dtype=idx.dtype))
        if alarmed_idx_hi.size > 0:
            to_add.append(np.asarray(alarmed_idx_hi, dtype=idx.dtype))
        if len(to_add) > 0:
            idx = np.unique(np.concatenate([idx, *to_add]))

        # 고른 시점의 [셀 전압 | 셀 SOC] 행: 실제 값(before)과 재구성 값(after)
        points["before"].extend(np.asarray(torch.cat([y_recovered, y_recovered2], dim=1), dtype=float)[idx])
        points["after"].extend(np.asarray(torch.cat([recon_u, recon_x], dim=1), dtype=float)[idx])
        points["labels"].extend([label] * len(idx))

        # 고른 시점의 알람 단계 (0: 없음, 1: thresholds[0] 초과, 2: thresholds[1] 초과)
        alarm_levels = np.zeros(len(idx), dtype=np.int8)
        if alarmed_idx_lo.size > 0:
            alarm_levels[np.isin(idx, alarmed_idx_lo)] = 1
        if alarmed_idx_hi.size > 0:
            alarm_levels[np.isin(idx, alarmed_idx_hi)] = 2
        points["alarm_levels"].extend(alarm_levels.tolist())

        print(f"Done (label={label}, vehicle={i})")

    return points, CI_last


def main(argv=None):
    """명령줄 옵션을 읽어 t-SNE 그림을 그린다. 순서는 이 파일 맨 위 설명의 "하는 일"과 같다.

    argv를 주면 명령줄 대신 그 목록을 옵션으로 읽는다(테스트에서 이렇게 부른다). 쓰는 곳: 이 파일 맨 아래 실행부.
    """
    os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
    warnings.filterwarnings("ignore")

    args = parser.parse_args(argv)

    run = Run(f"{args.models_dir}/{args.models_idx}")
    config = run.config
    print(f"Using dataset from: {config.split_dir}")

    print_sim_config(
        title="Train Run",
        config=args,
        extra={
            "device": "cpu",  # 평가는 CPU에서 한다 (Run이 CPU 모델을 돌려준다)
            "learning_case": config.learning_case,
            "PREPROCESSING": config.preprocessing,
            "SKIP_CHARGE_READY": config.skip_charge_ready,
            "no_use_dx": not config.use_dx,
            "add_one_output_layer": config.add_one_output_layer,
        },
    )

    try:
        plot_loss_curve_all(path=f"{args.models_dir}/{args.models_idx}")
    except Exception as e:
        print(f"Error plotting loss curve: {e}")

    # 관리도: 같은 조건 캐시 → 최종 모델 쌍이면 저장된 관리도 → 그 밖엔 학습 데이터로 재계산
    chart = chart_for_pair(run, args.u_model_idx, args.x_model_idx, source_data_dir=args.source_data_dir)
    net_loaded, netx_loaded = run.load_model_pair(args.u_model_idx, args.x_model_idx)
    # collect_vehicle_points 호출마다 똑같이 넘기는 값
    shared = dict(args=args, config=config, net_loaded=net_loaded, netx_loaded=netx_loaded, chart=chart)

    start = time.perf_counter()
    # 정상 기준 점은 validate 분할 앞쪽 --cnt-max대에서 한 번만 모은다 (label 0)
    validate_list = split_ids(config.split_dir, config.battery_type, "validate")
    normal_points, _ = collect_vehicle_points(validate_list[: args.cnt_max], 0, **shared)

    # 대상 차량(label 1)마다 점을 모아 정상 기준 점과 합쳐 그린다
    if args.fault_idx == -1:
        target_ids = split_ids(config.split_dir, config.battery_type, "test") + split_ids(
            config.split_dir, config.battery_type, "fault"
        )
    else:
        target_ids = [args.fault_idx]
    for target_id in target_ids:
        target_points, CI_target = collect_vehicle_points([target_id], 1, **shared)

        print(f"\n=== Plotting test_vehicle_idx={target_id} ===")
        plot_tsne_before_after_ci(
            X_before=normal_points["before"] + target_points["before"],
            X_after=normal_points["after"] + target_points["after"],
            labels=normal_points["labels"] + target_points["labels"],
            alarm_levels=normal_points["alarm_levels"] + target_points["alarm_levels"],
            CI_array=CI_target,
            thresholds=args.thr,
            args=args,
            vehicle_idx=target_id,
        )

    elapsed = time.perf_counter() - start
    h, rem = divmod(elapsed, 3600)
    m, s = divmod(rem, 60)
    print(f"{int(h)}시간 {int(m)}분 {s:.3f}초")


if __name__ == "__main__":
    main()
