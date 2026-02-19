import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
import pandas as pd
import matplotlib.ticker as mtick
import os
import warnings
import matplotlib
from Function_ import *
from Class_ import *
import torch

# from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split
import argparse
from monitering import *
import time

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
print(torch.cuda.device_count())

warnings.filterwarnings("ignore")
# VIN_data = pd.read_excel(r'{args.source_data_dir}/Name_list.xls')

parser = argparse.ArgumentParser(
    description="Run diagnostics plotting with CLI options"
)
parser.add_argument(
    "--battery-type",
    choices=["QAS", "DTI"],
    default="QAS",
    help="배터리 유형 선택 (DTI fault index range = [77~79], QAS fault index range = [335~392])",
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
# parser.add_argument("--results-dir", type=str, default="./results" if os.environ.get("RESULT_DIR") is None else os.environ.get("RESULT_DIR"))
parser.add_argument(
    "--source-data-dir",
    type=str,
    default=(
        "./data"
        if os.environ.get("SOURCE_DIR") is None
        else os.environ.get("SOURCE_DIR")
    ),
)

parser.add_argument("--u-model-idx", type=int, default=10)
parser.add_argument("--x-model-idx", type=int, default=10)

parser.add_argument("--x-start", type=int, default=20)
parser.add_argument("--x-tick-step", type=int, default=3000)
parser.add_argument("--normalize-dx", action="store_true")
parser.add_argument("--normalize-val", type=int, default=4)
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

args = parser.parse_args()

thresholds = list(args.thr)
vals = read_values_from_sim_config(
    f"{args.models_dir}/{args.models_idx}",
    names_to_find=[
        "learning_case",
        "ae_u_scale",
        "ae_u_shift",
        "ae_x_scale",
        "ae_x_shift",
        "no_use_dx",
        "add_one_output_layer",
    ],
    strict=False,  # True: 키가 하나라도 없을 시 예외, False: 없는 키는 None
)
learning_case = vals.learning_case
add_one_output_layer = (
    vals.add_one_output_layer if hasattr(vals, "add_one_output_layer") else False
)
use_dx = not vals.no_use_dx if hasattr(vals, "no_use_dx") else True

## ================== For backward compatibility =================== ##
ae_u_scale = vals.ae_u_scale if hasattr(vals, "ae_u_scale") else 1.8
ae_u_shift = vals.ae_u_shift if hasattr(vals, "ae_u_shift") else 2.5
ae_x_scale = vals.ae_x_scale if hasattr(vals, "ae_x_scale") else 1.0
ae_x_shift = vals.ae_x_shift if hasattr(vals, "ae_x_shift") else 0

# 플롯 결과 저장 폴더는 한 번만 생성
os.makedirs(f"{args.models_dir}/{args.models_idx}/results", exist_ok=True)

BATTERY_TYPE = args.battery_type  # 'DTI' or 'QAS'
PREPROCESSING, SKIP_CHARGE_READY = get_preprocessing_and_skip_charge_ready(
    learning_case
)
# SKIP_CHARGE_READY = False
dim_dict = get_input_dimensions(BATTERY_TYPE)

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
normal_list = (
    np.load(f"./{BATTERY_TYPE}_filtered_vehicle_ids_test.npy").astype(np.int64).tolist()
)
fault_list = (
    np.load(f"./{BATTERY_TYPE}_filtered_vehicle_ids_fault.npy")
    .astype(np.int64)
    .tolist()
)
# fault_list = np.arange(335, 393)


## =================== Load training data  =================== ##
import pickle

# Attempt to load feature cache for the specified u/x models
cache_dir = f"{args.models_dir}/{args.models_idx}/temp_cache"
u_feat_path = os.path.join(cache_dir, f"feat_u_{args.u_model_idx}.pkl")
x_feat_path = os.path.join(cache_dir, f"feat_x_{args.x_model_idx}.pkl")
loads = None

if os.path.exists(u_feat_path) and os.path.exists(x_feat_path):
    try:
        print(f"Loading cached training features for PCA setup:")
        print(f"  U: {u_feat_path}")
        print(f"  X: {x_feat_path}")

        with open(u_feat_path, "rb") as f:
            u_feat = pickle.load(f)
        with open(x_feat_path, "rb") as f:
            x_feat = pickle.load(f)

        # Structure: max_diff_ERRORU, max_diff_ERRORX, Z_U, Z_X, Z_U_smoothed, Z_X_smoothed
        df_data_train = pd.concat(
            [
                u_feat[0],  # max_diff_ERRORU
                x_feat[0],  # max_diff_ERRORX
                u_feat[1],  # Z_U
                x_feat[1],  # Z_X
                u_feat[2],  # Z_U_smoothed
                x_feat[2],  # Z_X_smoothed
            ],
            axis=1,
        )

        print("Computing PCA directly from cached training features...")
        loads = Custom_PCA(df_data_train, 0.99, 0.99)

        # Optimization: discard large arrays
        loads_list = list(loads)
        loads_list[13] = None  # Discard X (scores)
        loads_list[14] = None  # Discard data_nor
        loads = tuple(loads_list)

        del df_data_train, u_feat, x_feat

    except Exception as e:
        print(f"Error processing cached features: {e}.")
        loads = None

if loads is None:
    print("Cached features not found or error. Falling back to saved 'pca_arrays.npz'.")
    loads = load_pca_results(
        f"{args.models_dir}/{args.models_idx}",
        load_data_nor=False,
        validate_shapes=False,
    )

(
    v_I,
    v,
    v_ratio,
    p_k,
    data_mean,
    data_std,
    T_95_limit,
    T_99_limit,
    SPE_95_limit,
    SPE_99_limit,
    P,
    k,
    P_t,
    X,
    data_nor,
) = loads
net_loaded = CombinedAE(
    input_size=dim_dict["x"],
    encode2_input_size=dim_dict["q"],
    output_size=dim_dict["y"],
    activation_fn=(
        CustomSigmoidFunc(scale=ae_u_scale, shift=ae_u_shift)
        if PREPROCESSING == False
        else CustomSigmoidFunc(scale=1, shift=0)
    ),
    use_dx_in_forward=use_dx,
    add_one_output_layer=add_one_output_layer,
).to(device)
netx_loaded = CombinedAE(
    input_size=dim_dict["x2"],
    encode2_input_size=dim_dict["q2"],
    output_size=dim_dict["y2"],
    activation_fn=(
        CustomSigmoidFunc(scale=ae_x_scale, shift=ae_x_shift)
        if PREPROCESSING == False
        else CustomSigmoidFunc(scale=1, shift=0)
    ),
    use_dx_in_forward=use_dx,
    add_one_output_layer=add_one_output_layer,
).to(device)
u_model_idx = "net" if args.u_model_idx == -1 else f"net_e{args.u_model_idx}"
x_model_idx = "netx" if args.x_model_idx == -1 else f"netx_e{args.x_model_idx}"
net_state_dict = torch.load(
    os.path.join(f"{args.models_dir}/{args.models_idx}/artifact/", f"{u_model_idx}.pth")
)
net_loaded.load_state_dict(net_state_dict)
netx_state_dict = torch.load(
    os.path.join(f"{args.models_dir}/{args.models_idx}/artifact/", f"{x_model_idx}.pth")
)
netx_loaded.load_state_dict(netx_state_dict)


# 예시: normal_list, fault_list를 이미 갖고 있다고 가정
# 각 vehicle에서 df_data (N×6) 만든 다음 아래만 수행
def add_vehicle_points(df_data, label, idx):
    Xv = np.asarray(df_data, dtype=float)
    return Xv[idx], np.full(len(idx), label, dtype=int)
    # X_list.append(Xv[idx])
    # y_list.append(np.full(len(idx), label, dtype=int))


def collect_vehicle_points(
    *,
    vehicle_ids,
    label,
    per_vehicle,
    args,
    BATTERY_TYPE,
    dim_dict,
    PREPROCESSING,
    SKIP_CHARGE_READY,
    thresholds,
    net_loaded,
    netx_loaded,
    data_mean,
    data_std,
    p_k,
    v_I,
    T_95_limit,
    SPE_95_limit,
):
    X_list_before, Y_list_before = [], []
    X_list_after, Y_list_after = [], []
    alarm_masks = {"before": [], "after": []}

    CI_last = None

    for i in vehicle_ids:
        VEHICLE_ID = f"{i}"
        combined_tensor = safe_load(
            f"{args.source_data_dir}/{BATTERY_TYPE}/{VEHICLE_ID}/vin_2.pkl",
            verbose=False,
        )
        combined_tensorx = safe_load(
            f"{args.source_data_dir}/{BATTERY_TYPE}/{VEHICLE_ID}/vin_3.pkl",
            verbose=False,
        )

        combined_tensor, combined_tensorx = preprocess_loaded_tensor(
            combined_tensor,
            combined_tensorx,
            dim_dict,
            BATTERY_TYPE,
            PREPROCESSING,
            SKIP_CHARGE_READY,
            normalize_dx=args.normalize_dx,
        )

        # Use indexing to separate
        x_recovered = combined_tensor[:, : dim_dict["x"]]
        y_recovered = combined_tensor[:, dim_dict["x"] : dim_dict["x"] + dim_dict["y"]]
        z_recovered = combined_tensor[
            :,
            dim_dict["x"]
            + dim_dict["y"] : dim_dict["x"]
            + dim_dict["y"]
            + dim_dict["z"],
        ]
        q_recovered = combined_tensor[
            :, dim_dict["x"] + dim_dict["y"] + dim_dict["z"] :
        ]
        net_loaded = net_loaded.double().eval()

        with torch.inference_mode():
            recon_imtest = net_loaded(
                x_recovered, z_recovered, q_recovered, y_recovered
            )

        # Use indexing to separate
        x_recovered2 = combined_tensorx[:, : dim_dict["x2"]]
        y_recovered2 = combined_tensorx[
            :, dim_dict["x2"] : dim_dict["x2"] + dim_dict["y2"]
        ]
        z_recovered2 = combined_tensorx[
            :,
            dim_dict["x2"]
            + dim_dict["y2"] : dim_dict["x2"]
            + dim_dict["y2"]
            + dim_dict["z2"],
        ]
        q_recovered2 = combined_tensorx[
            :, dim_dict["x2"] + dim_dict["y2"] + dim_dict["z2"] :
        ]
        netx_loaded = netx_loaded.double().eval()

        with torch.inference_mode():
            reconx_imtest = netx_loaded(
                x_recovered2, z_recovered2, q_recovered2, y_recovered2
            )

            # 출력/정답을 각각 numpy로 변환하지 말고 torch에서 error를 먼저 계산 후 1회만 변환
            ERRORU = np.abs(recon_imtest[0] - y_recovered).cpu().numpy()
            ERRORX = np.abs(reconx_imtest[0] - y_recovered2).cpu().numpy()

        df_data, _ = DiagnosisFeature(ERRORU, ERRORX, get_true_feature=False)
        t2_array, _ = T2_array(df_data, data_mean, data_std, p_k, v_I)
        spe_array, _ = SPE_array(df_data, data_mean, data_std, p_k)
        CI_array = (spe_array / SPE_95_limit) + (t2_array / T_95_limit)
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
            torch.cat([recon_imtest[0], reconx_imtest[0]], dim=1),
            label,
            idx=idx,
        )

        # x_before, y_before = add_vehicle_points(
        #     torch.cat(
        #         [
        #             (recon_imtest[0] - y_recovered).detach(),
        #             (reconx_imtest[0] - y_recovered2).detach(),
        #         ],
        #         dim=1,
        #     )
        #     .cpu()
        #     .numpy(),
        #     label,
        #     idx=idx,
        # )
        # x_after, y_after = add_vehicle_points(
        #     torch.cat([recon_imtest[1], reconx_imtest[1]], dim=1),
        #     label,
        #     idx=idx,
        # )
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


start = time.perf_counter()
# cnt_max = 6
cnt_max = args.cnt_max
per_vehicle = args.per_vehicle
normal_but_faulty = [46, 184, 252]  # [46, 184, 252, 301]
normal = np.array(normal_list)[
    np.asarray(np.isin(normal_list, normal_but_faulty), dtype=bool)
].tolist()

# label=0(정상) 차량은 1회만 고정 수집
normal_vehicle_ids = [*normal_list[0:cnt_max], *normal]
normal_pack, _ = collect_vehicle_points(
    vehicle_ids=normal_vehicle_ids,
    label=0,
    per_vehicle=per_vehicle,
    args=args,
    BATTERY_TYPE=BATTERY_TYPE,
    dim_dict=dim_dict,
    PREPROCESSING=PREPROCESSING,
    SKIP_CHARGE_READY=SKIP_CHARGE_READY,
    thresholds=thresholds,
    net_loaded=net_loaded,
    netx_loaded=netx_loaded,
    data_mean=data_mean,
    data_std=data_std,
    p_k=p_k,
    v_I=v_I,
    T_95_limit=T_95_limit,
    SPE_95_limit=SPE_95_limit,
)

# label=1(고장) 차량 인덱스는 바꿔가며 반복 플롯
fault_idx = np.arange(335, 393) if args.fault_idx == -1 else [args.fault_idx]
fault_vehicle_idxs = fault_idx
for fault_vehicle_idx in fault_vehicle_idxs:
    fault_pack, CI_fault = collect_vehicle_points(
        vehicle_ids=[fault_vehicle_idx],
        label=1,
        per_vehicle=per_vehicle,
        args=args,
        BATTERY_TYPE=BATTERY_TYPE,
        dim_dict=dim_dict,
        PREPROCESSING=PREPROCESSING,
        SKIP_CHARGE_READY=SKIP_CHARGE_READY,
        thresholds=thresholds,
        net_loaded=net_loaded,
        netx_loaded=netx_loaded,
        data_mean=data_mean,
        data_std=data_std,
        p_k=p_k,
        v_I=v_I,
        T_95_limit=T_95_limit,
        SPE_95_limit=SPE_95_limit,
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

    print(f"\n=== Plotting fault_vehicle_idx={fault_vehicle_idx} ===")
    fig, axes = plot_tsne_before_after_ci(
        X_list_before=X_list_before,
        Y_list_before=Y_list_before,
        X_list_after=X_list_after,
        Y_list_after=Y_list_after,
        alarm_masks=alarm_masks,
        CI_array=CI_fault,
        thresholds=thresholds,
        args=args,
        fault_vehicle_idx=fault_vehicle_idx,
    )

elapsed = time.perf_counter() - start
h, rem = divmod(elapsed, 3600)
m, s = divmod(rem, 60)
print(f"{int(h)}시간 {int(m)}분 {s:.3f}초")
