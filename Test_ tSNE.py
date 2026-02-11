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
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler
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
    default="./models",  # 260206_084839 or 260209_105821
)  # 260123_082222 & 260126_100030
parser.add_argument(
    "--models-idx",
    type=str,
    default="260206_084839",  # 260206_084839 or 260209_105821
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

parser.add_argument("--x-start", type=int, default=20)
parser.add_argument("--x-tick-step", type=int, default=3000)
parser.add_argument("--sigma-levels", type=str, default="3,4.5,6")
parser.add_argument("--normalize-dx", action="store_true")
parser.add_argument("--normalize-val", type=int, default=4)

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

thresholds = [18.7, 28.7]
vals = read_values_from_sim_config(
    f"{args.models_dir}/{args.models_idx}",
    names_to_find=[
        "learning_case",
        "ae_u_scale",
        "ae_u_shift",
        "ae_x_scale",
        "ae_x_shift",
    ],
    strict=False,  # True: 키가 하나라도 없을 시 예외, False: 없는 키는 None
)
learning_case = vals["learning_case"]

## ================== For backward compatibility =================== ##
ae_u_scale = vals["ae_u_scale"] if vals["ae_u_scale"] is not None else 1.8
ae_u_shift = vals["ae_u_shift"] if vals["ae_u_shift"] is not None else 2.5
ae_x_scale = vals["ae_x_scale"] if vals["ae_x_scale"] is not None else 1.0
ae_x_shift = vals["ae_x_shift"] if vals["ae_x_shift"] is not None else 0

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
    },
)


try:
    plot_loss_curve_all(args)
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
loads = load_pca_results(
    f"{args.models_dir}/{args.models_idx}", load_data_nor=False, validate_shapes=False
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
if PREPROCESSING:
    net_loaded = CombinedAE(
        input_size=dim_dict["x"],
        encode2_input_size=dim_dict["q"],
        output_size=dim_dict["y"],
        activation_fn=torch.sigmoid,
        use_dx_in_forward=True,
    ).to(device)
else:
    net_loaded = CombinedAE(
        input_size=dim_dict["x"],
        encode2_input_size=dim_dict["q"],
        output_size=dim_dict["y"],
        activation_fn=CustomSigmoidFunc(scale=ae_u_scale, shift=ae_u_shift),
        use_dx_in_forward=True,
    ).to(device)
netx_loaded = CombinedAE(
    input_size=dim_dict["x2"],
    encode2_input_size=dim_dict["q2"],
    output_size=dim_dict["y2"],
    activation_fn=CustomSigmoidFunc(scale=ae_x_scale, shift=ae_x_shift),
    use_dx_in_forward=True,
).to(device)
net_state_dict = torch.load(
    os.path.join(f"{args.models_dir}/{args.models_idx}/artifact/", "net.pth")
)
net_loaded.load_state_dict(net_state_dict)
netx_state_dict = torch.load(
    os.path.join(f"{args.models_dir}/{args.models_idx}/artifact/", "netx.pth")
)
netx_loaded.load_state_dict(netx_state_dict)


# 예시: normal_list, fault_list를 이미 갖고 있다고 가정
# 각 vehicle에서 df_data (N×6) 만든 다음 아래만 수행
def add_vehicle_points(df_data, label, idx):
    Xv = np.asarray(df_data, dtype=float)
    return Xv[idx], np.full(len(idx), label, dtype=int)
    # X_list.append(Xv[idx])
    # y_list.append(np.full(len(idx), label, dtype=int))


start = time.perf_counter()
cnt_max = 6
per_vehicle = 500
# test_list = [normal_list[0:cnt_max], fault_list[0:cnt_max]]
fault_vehicle_idx = 347
normal = np.array(normal_list)[
    np.asarray(np.isin(normal_list, [46, 184, 252, 301]), dtype=bool)
].tolist()
test_list = [[*normal_list[0:cnt_max], *normal], [fault_vehicle_idx]]
X_list_before, Y_list_before = [], []
X_list_after, Y_list_after = [], []
alarm_masks = {
    "before": [],  # 0: none, 1: CI>thresholds[0](blue), 2: CI>thresholds[1](red)
    "after": [],  # 0: none, 1: CI>thresholds[0](blue), 2: CI>thresholds[1](red)
}
for label, vehicle_ids in enumerate(test_list):
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

        combined_tensor, combined_tensorx = preprocess_combined_tensor(
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
            # net_loaded.analyze_outputs(x_recovered, z_recovered, q_recovered, recon_imtest[0], y_recovered, cell_idx=1, combine=True, ncols=3, fill="spiral")

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
            # netx_loaded.analyze_outputs(x_recovered2, z_recovered2, q_recovered2, reconx_imtest[0], y_recovered2, cell_idx=1, combine=True, ncols=3, fill="spiral")

            # 출력/정답을 각각 numpy로 변환하지 말고 torch에서 error를 먼저 계산 후 1회만 변환
            ERRORU = np.abs(recon_imtest[0] - y_recovered).cpu().numpy()
            ERRORX = np.abs(reconx_imtest[0] - y_recovered2).cpu().numpy()

        df_data, _ = DiagnosisFeature(ERRORU, ERRORX, get_true_feature=False)
        t2_array, _ = T2_array(df_data, data_mean, data_std, p_k, v_I)
        spe_array, _ = SPE_array(df_data, data_mean, data_std, p_k)
        # CI_contrib = (spe_contrib / SPE_95_limit) + (t2_contrib / T_95_limit)
        CI_array = (spe_array / SPE_95_limit) + (t2_array / T_95_limit)
        # recon_merged = torch.cat([y_recovered, y_recovered2], dim=1)
        recon_merged = torch.cat(
            [recon_imtest[0], reconx_imtest[0]], dim=1
        )  # (N, y+y2)
        # add_vehicle_points(
        #     np.asarray((df_data - data_mean) / data_std, dtype=float),
        #     label,
        #     per_vehicle=1000,
        # )
        alarmed_idx_hi = np.where(CI_array > thresholds[1])[0]
        alarmed_idx_lo = np.where(
            (CI_array > thresholds[0]) & (CI_array <= thresholds[1])
        )[0]

        # alarmed_set = (
        #     set(np.where(CI_array > thresholds[1])[0].tolist()) if label == 1 else set()
        # )

        idx = np.random.default_rng(0).choice(
            y_recovered.shape[0],
            size=min(per_vehicle, y_recovered.shape[0]),
            replace=False,
        )
        # normal(label==0)/fault(label==1) 모두에서 thresholds[0]/[1] 초과 샘플을 반드시 포함 (현재 차량 기준)
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
        X_list_before.extend(x_before)
        Y_list_before.extend(y_before)
        X_list_after.extend(x_after)
        Y_list_after.extend(y_after)

        # 샘플링된 idx 중 알람 레벨 표시 (normal/fault 공통)
        # 0: none, 1: thresholds[0] 초과(blue), 2: thresholds[1] 초과(red)
        alarm_levels = np.zeros(len(idx), dtype=np.int8)
        if alarmed_idx_lo.size > 0:
            alarm_levels[np.isin(idx, alarmed_idx_lo)] = 1
        if alarmed_idx_hi.size > 0:
            alarm_levels[np.isin(idx, alarmed_idx_hi)] = 2
        alarm_masks["before"].extend(alarm_levels.tolist())
        alarm_masks["after"].extend(alarm_levels.tolist())

        print("Done")

# ===== 리스트 -> 배열로 변환 (행 단위로 쌓였다는 가정) =====
X_before = np.vstack(X_list_before).astype(float)  # (M1, D)
y_before = np.asarray(Y_list_before, dtype=int)  # (M1,)

X_after = np.vstack(X_list_after).astype(float)  # (M2, D)
y_after = np.asarray(Y_list_after, dtype=int)  # (M2,)

print("before:", X_before.shape, y_before.shape)
print("after :", X_after.shape, y_after.shape)

# ===== 합쳐서 같은 스케일러/같은 t-SNE로 임베딩 =====
X_all = np.vstack([X_before, X_after])
Xs_all = StandardScaler().fit_transform(X_all)

tsne = TSNE(
    n_components=2,
    perplexity=10,
    learning_rate="auto",
    init="pca",
    random_state=0,
    max_iter=1000,
)
Z_all = tsne.fit_transform(Xs_all)

n_before = X_before.shape[0]
Z_before = Z_all[:n_before]
Z_after = Z_all[n_before:]

# ===== 1x2 plot =====
fig, axes = plt.subplots(1, 2, figsize=(12, 6), sharex=True, sharey=True)


for ax, Z, y, title in [
    (axes[0], Z_before, y_before, "Before (y_recovered)"),
    (axes[1], Z_after, y_after, "After (recon)"),
]:
    ax.scatter(Z[y == 0, 0], Z[y == 0, 1], s=4, alpha=0.5, label="normal")
    ax.scatter(Z[y == 1, 0], Z[y == 1, 1], s=4, alpha=0.5, label="fault")
    ax.set_title(title)
    ax.grid(True, alpha=0.2)

# alarm 샘플 강조(테두리만) — before/after 공통 처리
alarm_style_base = {
    "s": 4,
    "alpha": 0.3,
    "facecolors": "none",
}

alarm_style_normal_lo = {
    **alarm_style_base,
    "marker": "o",
    "edgecolors": "#D100D1",
    "linewidths": 0.5,
    "zorder": 5,
    "label": f"normal (CI>{thresholds[0]})",
}
alarm_style_normal_hi = {
    **alarm_style_base,
    "marker": "o",
    "edgecolors": "#2A9D8F",
    "linewidths": 0.5,
    "zorder": 5,
    "label": f"normal (CI>{thresholds[1]})",
}

alarm_style_fault_lo = {
    **alarm_style_base,
    "marker": "o",
    "edgecolors": "blue",
    "linewidths": 0.9,
    "zorder": 6,
    "label": f"fault (CI>{thresholds[0]})",
}
alarm_style_fault_hi = {
    **alarm_style_base,
    "marker": "o",
    "edgecolors": "red",
    "linewidths": 0.9,
    "zorder": 6,
    "label": f"fault (CI>{thresholds[1]})",
}

for key, ax, Z, y in [
    ("before", axes[0], Z_before, y_before),
    ("after", axes[1], Z_after, y_after),
]:
    if not args.alarm:
        continue

    alarm_level = np.asarray(alarm_masks[key], dtype=np.int8)
    if alarm_level.shape[0] != Z.shape[0]:
        print(
            f"[WARN] alarm_mask_{key} 길이 불일치:",
            alarm_level.shape[0],
            f"vs Z_{key}:",
            Z.shape[0],
        )
        continue

    mask_normal_lo = (y == 0) & (alarm_level == 1)
    mask_normal_hi = (y == 0) & (alarm_level == 2)
    mask_fault_lo = (y == 1) & (alarm_level == 1)
    mask_fault_hi = (y == 1) & (alarm_level == 2)

    if args.alarm_nlo and np.any(mask_normal_lo):
        ax.scatter(Z[mask_normal_lo, 0], Z[mask_normal_lo, 1], **alarm_style_normal_lo)
    if args.alarm_nhi and np.any(mask_normal_hi):
        ax.scatter(Z[mask_normal_hi, 0], Z[mask_normal_hi, 1], **alarm_style_normal_hi)
    if args.alarm_flo and np.any(mask_fault_lo):
        ax.scatter(Z[mask_fault_lo, 0], Z[mask_fault_lo, 1], **alarm_style_fault_lo)
    if args.alarm_fhi and np.any(mask_fault_hi):
        ax.scatter(Z[mask_fault_hi, 0], Z[mask_fault_hi, 1], **alarm_style_fault_hi)

axes[0].legend(loc="best")
axes[1].legend(loc="best")
plt.tight_layout()
if args.save:
    save_path = (
        f"{args.models_dir}/{args.models_idx}/results/tSNE_{fault_vehicle_idx}.png"
    )
    if save_path is not None and fig is not None:
        dir_ = os.path.dirname(save_path)
        if dir_:
            os.makedirs(dir_, exist_ok=True)
        fig.savefig(save_path, dpi=150)
else:
    plt.show()

elapsed = time.perf_counter() - start
h, rem = divmod(elapsed, 3600)
m, s = divmod(rem, 60)
print(f"{int(h)}시간 {int(m)}분 {s:.3f}초")
