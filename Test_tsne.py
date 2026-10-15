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

device = torch.device("cpu")  # 평가는 CPU 전용 (Run이 CPU 모델을 돌려준다)

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
    help="normal lo(CI>thr[0]) 오버레이를 끕니다.",
)


# 예시: normal_list, fault_list를 이미 갖고 있다고 가정
# 각 vehicle에서 df_data (N×6) 만든 다음 아래만 수행
def add_vehicle_points(df_data, label, idx):
    Xv = np.asarray(df_data, dtype=float)
    return Xv[idx], np.full(len(idx), label, dtype=int)


def collect_vehicle_points(
    *,
    vehicle_ids,
    label,
    per_vehicle,
    args,
    config,
    thresholds,
    net_loaded,
    netx_loaded,
    chart,
):
    X_list_before, Y_list_before = [], []
    X_list_after, Y_list_after = [], []
    alarm_masks = {"before": [], "after": []}

    CI_last = None

    for i in vehicle_ids:
        VEHICLE_ID = f"{i}"
        combined_tensor, combined_tensorx = load_vehicle(args.source_data_dir, config, VEHICLE_ID)

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
            size=min(per_vehicle, y_recovered.shape[0]),
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

        x_before, y_before = add_vehicle_points(
            torch.cat([y_recovered, y_recovered2], dim=1),
            label,
            idx=idx,
        )
        x_after, y_after = add_vehicle_points(
            torch.cat([recon_u, recon_x], dim=1),
            label,
            idx=idx,
        )

        X_list_before.extend(x_before)
        Y_list_before.extend(y_before)
        X_list_after.extend(x_after)
        Y_list_after.extend(y_after)

        # 샘플링된 idx 중 알람 레벨 표시
        # 0: none, 1: thresholds[0] 초과, 2: thresholds[1] 초과
        alarm_levels = np.zeros(len(idx), dtype=np.int8)
        if alarmed_idx_lo.size > 0:
            alarm_levels[np.isin(idx, alarmed_idx_lo)] = 1
        if alarmed_idx_hi.size > 0:
            alarm_levels[np.isin(idx, alarmed_idx_hi)] = 2
        alarm_masks["before"].extend(alarm_levels.tolist())
        alarm_masks["after"].extend(alarm_levels.tolist())

        print(f"Done (label={label}, vehicle={i})")

    pack = {
        "X_list_before": X_list_before,
        "Y_list_before": Y_list_before,
        "X_list_after": X_list_after,
        "Y_list_after": Y_list_after,
        "alarm_masks": alarm_masks,
    }
    return pack, CI_last


def main(argv=None):
    os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
    print(torch.cuda.device_count())

    warnings.filterwarnings("ignore")

    args = parser.parse_args(argv)

    thresholds = list(args.thr)
    run = Run(f"{args.models_dir}/{args.models_idx}")
    learning_case = run.config.learning_case

    DATA_BASE_PATH = run.config.split_dir
    print(f"Using dataset from: {DATA_BASE_PATH}")
    add_one_output_layer = run.config.add_one_output_layer
    use_dx = run.config.use_dx

    # 플롯 결과 저장 폴더는 한 번만 생성
    os.makedirs(f"{args.models_dir}/{args.models_idx}/results", exist_ok=True)

    BATTERY_TYPE = run.config.battery_type  # 'DTI' or 'QAS'
    PREPROCESSING, SKIP_CHARGE_READY = run.config.preprocessing, run.config.skip_charge_ready

    print_sim_config(
        title="Train Run",
        config=args,
        extra={
            "device": str(device),
            "learning_case": learning_case,
            "PREPROCESSING": PREPROCESSING,
            "SKIP_CHARGE_READY": SKIP_CHARGE_READY,
            "no_use_dx": not use_dx,
            "add_one_output_layer": add_one_output_layer,
        },
    )

    try:
        plot_loss_curve_all(path=f"{args.models_dir}/{args.models_idx}")
    except Exception as e:
        print(f"Error plotting loss curve: {e}")


    ## =================== DTI fault index range = [77~79] ===================
    ## =================== QAS fault index range = [335~392] ===================
    validate_list = split_ids(DATA_BASE_PATH, BATTERY_TYPE, "validate")
    normal_list = split_ids(DATA_BASE_PATH, BATTERY_TYPE, "test")
    fault_list = split_ids(DATA_BASE_PATH, BATTERY_TYPE, "fault")


    ## =================== Load training data  =================== ##
    # 관리도: 같은 조건 캐시 → 최종 모델 쌍이면 저장된 관리도 → 그 밖엔 학습 데이터로 재계산
    chart = chart_for_pair(run, args.u_model_idx, args.x_model_idx, source_data_dir=args.source_data_dir)

    net_loaded, netx_loaded = run.load_model_pair(args.u_model_idx, args.x_model_idx)


    start = time.perf_counter()
    cnt_max = args.cnt_max
    per_vehicle = args.per_vehicle
    # label=0(정상) 차량은 1회만 고정 수집
    normal_vehicle_ids = validate_list[0:cnt_max]
    normal_pack, _ = collect_vehicle_points(
        vehicle_ids=normal_vehicle_ids,
        label=0,
        per_vehicle=per_vehicle,
        args=args,
        config=run.config,
        thresholds=thresholds,
        net_loaded=net_loaded,
        netx_loaded=netx_loaded,
        chart=chart,
    )

    # label=1(고장) 차량 인덱스는 바꿔가며 반복 플롯

    test_list = [*normal_list, *fault_list]
    test_idx = test_list if args.fault_idx == -1 else [args.fault_idx]
    test_vehicle_idxs = test_idx
    for test_vehicle_idx in test_vehicle_idxs:
        fault_pack, CI_fault = collect_vehicle_points(
            vehicle_ids=[test_vehicle_idx],
            label=1,
            per_vehicle=per_vehicle,
            args=args,
            config=run.config,
            thresholds=thresholds,
            net_loaded=net_loaded,
            netx_loaded=netx_loaded,
            chart=chart,
        )

        X_list_before = normal_pack["X_list_before"] + fault_pack["X_list_before"]
        Y_list_before = normal_pack["Y_list_before"] + fault_pack["Y_list_before"]
        X_list_after = normal_pack["X_list_after"] + fault_pack["X_list_after"]
        Y_list_after = normal_pack["Y_list_after"] + fault_pack["Y_list_after"]
        alarm_masks = {
            "before": normal_pack["alarm_masks"]["before"]
            + fault_pack["alarm_masks"]["before"],
            "after": normal_pack["alarm_masks"]["after"]
            + fault_pack["alarm_masks"]["after"],
        }

        print(f"\n=== Plotting test_vehicle_idx={test_vehicle_idx} ===")
        plot_tsne_before_after_ci(
            X_list_before=X_list_before,
            Y_list_before=Y_list_before,
            X_list_after=X_list_after,
            Y_list_after=Y_list_after,
            alarm_masks=alarm_masks,
            CI_array=CI_fault,
            thresholds=thresholds,
            args=args,
            vehicle_idx=test_vehicle_idx,
        )

    elapsed = time.perf_counter() - start
    h, rem = divmod(elapsed, 3600)
    m, s = divmod(rem, 60)
    print(f"{int(h)}시간 {int(m)}분 {s:.3f}초")


if __name__ == "__main__":
    main()
