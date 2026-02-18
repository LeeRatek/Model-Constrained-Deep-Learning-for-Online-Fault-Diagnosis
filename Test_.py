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
from Train_pca_only import train_pca_only

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
    "--models-dir", type=str, default="./models"
)  # 260206_084839 & 260211_163108
parser.add_argument("--models-folder", type=str, default="260206_084839")
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
parser.add_argument("--u-model-idx", type=int, default=10)
parser.add_argument("--x-model-idx", type=int, default=10)
args = parser.parse_args()

vals = read_values_from_sim_config(
    f"{args.models_dir}/{args.models_folder}",
    exclude_keys=[
        "models_dir",
        "models_folder",
        "source_data_dir",
    ],
    strict=False,  # True: 키가 하나라도 없을 시 예외, False: 없는 키는 None
)

learning_case = vals.learning_case


add_one_output_layer = (
    vals.add_one_output_layer if hasattr(vals, "add_one_output_layer") else False
)
use_dx = not vals.no_use_dx if hasattr(vals, "no_use_dx") else True
u_model_idx = "net" if args.u_model_idx == -1 else f"net_e{args.u_model_idx}"
x_model_idx = "netx" if args.x_model_idx == -1 else f"netx_e{args.x_model_idx}"

## ================== For backward compatibility =================== ##
ae_u_scale = vals.ae_u_scale if hasattr(vals, "ae_u_scale") else 1.8
ae_u_shift = vals.ae_u_shift if hasattr(vals, "ae_u_shift") else 2.5
ae_x_scale = vals.ae_x_scale if hasattr(vals, "ae_x_scale") else 1.0
ae_x_shift = vals.ae_x_shift if hasattr(vals, "ae_x_shift") else 0

# 플롯 결과 저장 폴더는 한 번만 생성
os.makedirs(f"{args.models_dir}/{args.models_folder}/results", exist_ok=True)

BATTERY_TYPE = vals.battery_type  # 'DTI' or 'QAS'
PREPROCESSING, SKIP_CHARGE_READY = get_preprocessing_and_skip_charge_ready(
    learning_case
)
# SKIP_CHARGE_READY = False
dim_dict = get_input_dimensions(BATTERY_TYPE)
AUC_ANALYSIS = False
if not AUC_ANALYSIS:
    thresholds = [10.4, 87.5]
    # thresholds = [20, 30]
analyse_ae_output = False

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
    plot_loss_curve_all(args)
except Exception as e:
    print(f"Error plotting loss curve: {e}")


## =================== DTI fault index range = [77~79] ===================
## =================== QAS fault index range = [335~392] ===================

# sim_config_path = os.path.join(f"{args.models_dir}/{args.models_folder}", "sim_config.txt")
# train_vehicle_ids = load_vehicle_ids_used_for_training(sim_config_path)
# all_normal_vehicle_ids = np.load(f"./{BATTERY_TYPE}_filtered_vehicle_ids_normal.npy").astype(np.int64).tolist()
# normal_list = [vid for vid in all_normal_vehicle_ids if vid not in set(train_vehicle_ids)]
normal_list = (
    np.load(f"./{BATTERY_TYPE}_filtered_vehicle_ids_test.npy").astype(np.int64).tolist()
)
fault_list = (
    np.load(f"./{BATTERY_TYPE}_filtered_vehicle_ids_fault.npy")
    .astype(np.int64)
    .tolist()
)
# fault_list = np.arange(335, 393)
test_list = [normal_list, fault_list]


## =================== Initilize params for AUROC curve  =================== ##
total_test_vehicles = len(normal_list) + len(fault_list)
predict_threshold_array = np.arange(0, 1000, 0.1)
predict_thresholds = np.asarray(list(predict_threshold_array), dtype=float)
predict_results = np.zeros(
    (total_test_vehicles, len(predict_thresholds)), dtype=np.int8
)
y_true = np.zeros(total_test_vehicles, dtype=np.int8)  # normal=0, fault=1

## =================== Load training data  =================== ##
start_pca_load = time.perf_counter()
# Attempt to load feature cache for the specified u/x models
cache_dir = f"{args.models_dir}/{args.models_folder}/temp_cache"
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
        del df_data_train, u_feat, x_feat

    except Exception as e:
        print(f"Error processing cached features: {e}.")
        loads = None

if loads is None and args.u_model_idx == -1:
    print("Cached features not found or error. Falling back to saved 'pca_arrays.npz'.")
    loads = load_pca_results(
        f"{args.models_dir}/{args.models_idx}",
        load_data_nor=False,
        validate_shapes=False,
    )
elif loads is None:
    vals.u_model_idx = args.u_model_idx
    vals.x_model_idx = args.x_model_idx
    vals.models_dir = args.models_dir
    vals.models_folder = args.models_folder
    vals.source_data_dir = args.source_data_dir
    loads = train_pca_only(
        args, device, save=False, inference_batch_size=65536
    )  # Large batch for faster processing
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
elapsed_pca_load = time.perf_counter() - start_pca_load
print(f"PCA results loading time (once): {elapsed_pca_load:.3f} seconds")

net_loaded = CombinedAE(
    input_size=dim_dict["x"],
    encode2_input_size=dim_dict["q"],
    output_size=dim_dict["y"],
    activation_fn=(
        CustomSigmoidFunc(scale=ae_u_scale, shift=ae_u_shift)
        if PREPROCESSING == False
        else torch.sigmoid
    ),
    use_dx_in_forward=use_dx,
    add_one_output_layer=add_one_output_layer,
).to(device)
netx_loaded = CombinedAE(
    input_size=dim_dict["x2"],
    encode2_input_size=dim_dict["q2"],
    output_size=dim_dict["y2"],
    activation_fn=CustomSigmoidFunc(scale=ae_x_scale, shift=ae_x_shift),
    use_dx_in_forward=use_dx,
    add_one_output_layer=add_one_output_layer,
).to(device)

net_state_dict = torch.load(
    os.path.join(
        f"{args.models_dir}/{args.models_folder}/artifact/", f"{u_model_idx}.pth"
    )
)
net_loaded.load_state_dict(net_state_dict)
netx_state_dict = torch.load(
    os.path.join(
        f"{args.models_dir}/{args.models_folder}/artifact/", f"{x_model_idx}.pth"
    )
)
netx_loaded.load_state_dict(netx_state_dict)

## =================== Plot network layer parameters  =================== ##
showing_params = False
fc1_w = net_loaded.fc1.weight.detach().cpu().numpy()
fc1_b = net_loaded.fc1.bias.detach().cpu().numpy()
fc2_w = net_loaded.fc2.weight.detach().cpu().numpy()
fc2_b = net_loaded.fc2.bias.detach().cpu().numpy()
print(
    f"Weights and biases of net: fc1[w:{fc1_w}, b:{fc1_b}], fc2[w:{fc2_w}, b:{fc2_b}]"
)
plot_net_layer_params_by_index(
    net_loaded,
    net_name="net",
    mode="bar",
    max_points=8000,
    show=showing_params,
    save_dir=f"{args.models_dir}/{args.models_folder}",
    layer_names=("fc1", "fc2"),
    figsize=(6, 3),
)
plot_net_layer_params_by_index(
    net_loaded,
    net_name="net",
    mode="bar",
    max_points=8000,
    show=showing_params,
    save_dir=f"{args.models_dir}/{args.models_folder}",
    layer_names=("fc1", "fc2"),
    figsize=(6, 3),
    normalize_y=False,
)

x_fc1_w = netx_loaded.fc1.weight.detach().cpu().numpy()
x_fc1_b = netx_loaded.fc1.bias.detach().cpu().numpy()
x_fc2_w = netx_loaded.fc2.weight.detach().cpu().numpy()
x_fc2_b = netx_loaded.fc2.bias.detach().cpu().numpy()
print(
    f"Weights and biases of netx: fc1[w:{x_fc1_w}, b:{x_fc1_b}], fc2[w:{x_fc2_w}, b:{x_fc2_b}]"
)
plot_net_layer_params_by_index(
    netx_loaded,
    net_name="netx",
    mode="bar",
    max_points=8000,
    show=showing_params,
    save_dir=f"{args.models_dir}/{args.models_folder}",
    layer_names=("fc1", "fc2"),
    figsize=(6, 3),
)
plot_net_layer_params_by_index(
    netx_loaded,
    net_name="netx",
    mode="bar",
    max_points=8000,
    show=showing_params,
    save_dir=f"{args.models_dir}/{args.models_folder}",
    layer_names=("fc1", "fc2"),
    figsize=(6, 3),
    normalize_y=False,
)

start = time.perf_counter()
test_idx = 0
for label, vehicle_ids in enumerate(test_list):
    for i in vehicle_ids:
        i = 46  # 3, 46, 351, 362, 382
        print(f"Processing label={label} with vehicle ID={i}...")
        elapsed = time.perf_counter() - start
        h, rem = divmod(elapsed, 3600)
        m, s = divmod(rem, 60)
        print(f"Elapsed time: {int(h)}시간 {int(m)}분 {s:.3f}초")
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
            normalize_dx=vals.normalize_dx,
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
            # ERRORU = (recon_imtest[0] - y_recovered).cpu().numpy()
            # ERRORX = (reconx_imtest[0] - y_recovered2).cpu().numpy()

        # plot_testX_timeseries(y_recovered, feature_names="U", title="U", figsize=(6, 3), show=True, seperate=False, start_idx=0, _range=[0,0])
        # plot_testX_timeseries(y_recovered2, feature_names="X", title="X", figsize=(6, 3), show=True, seperate=False, start_idx=0, _range=[0,0])
        # plot_testX_timeseries(recon_imtest[0], feature_names="$\hat{U}$", title="$\hat{U}$", figsize=(6, 3), show=True, seperate=False, start_idx=0, _range=[0,0])
        # plot_testX_timeseries(reconx_imtest[0], feature_names="$\hat{X}$", title="$\hat{X}$", figsize=(6, 3), show=True, seperate=False, start_idx=0, _range=[0,0])
        # plot_testX_timeseries(ERRORU, feature_names="|U - $\hat{U}|$", title="|U - $\hat{U}|$", figsize=(6, 3), show=True, seperate=False, start_idx=0, _range=[0,0])
        # plot_testX_timeseries(ERRORX, feature_names="|X - $\hat{X}|$", title="|X - $\hat{X}|$", figsize=(6, 3), show=True, seperate=False, start_idx=0, _range=[0,0])
        # plot_testX_timeseries(np.asarray(ERRORU, dtype=float).max(axis=1).reshape(ERRORU.shape[0],1), feature_names="max(|U - $\hat{U}|$)", title="max(|U - $\hat{U}|$)", figsize=(6, 3), show=True, seperate=False, start_idx=0, _range=[0,0])
        # plot_testX_timeseries(np.asarray(ERRORX, dtype=float).max(axis=1).reshape(ERRORU.shape[0],1), feature_names="max(|X - $\hat{X}|$)", title="max(|X - $\hat{X}|$)", figsize=(6, 3), show=True, seperate=False, start_idx=0, _range=[0,0])

        # plot_testX_timeseries(np.asarray(np.abs(ERRORU), dtype=float).max(axis=1).reshape(ERRORU.shape[0],1), feature_names="|max(U - $\hat{U}$)|", title="|max(U - $\hat{U}$)|", figsize=(6, 3), show=True, seperate=False, start_idx=0, _range=[0,0])
        # plot_testX_timeseries(np.asarray(np.abs(ERRORX), dtype=float).max(axis=1).reshape(ERRORU.shape[0],1), feature_names="|max(X - $\hat{X}$)|", title="|max(X - $\hat{X}$)|", figsize=(6, 3), show=True, seperate=False, start_idx=0, _range=[0,0])

        # title = ["$zu_2$", "$zx_2$", "$zu_1$", "$zx_1$", "$ewu$", "$ewx$"]
        # for i in range(len(title)):
        #     plot_testX_timeseries(t2_contrib[:, i].to_numpy().reshape(df_data2.shape[0],1), feature_names=title[i], title=title[i], figsize=(6, 3), show=True, seperate=False, start_idx=0, _range=[0,0])

        # plot_testX_timeseries(df_data2.iloc[:,0].to_numpy().reshape(df_data2.shape[0],1), feature_names="$zu_2$", title="$zu_2$", figsize=(6, 3), show=True, seperate=False, start_idx=0, _range=[0,0])
        # plot_testX_timeseries(df_data2.iloc[:,1].to_numpy().reshape(df_data2.shape[0],1), feature_names="$zx_2$", title="$zx_2$", figsize=(6, 3), show=True, seperate=False, start_idx=0, _range=[0,0])
        # plot_testX_timeseries(df_data2.iloc[:,2].to_numpy().reshape(df_data2.shape[0],1), feature_names="$zu_1$", title="$zu_1$", figsize=(6, 3), show=True, seperate=False, start_idx=0, _range=[0,0])
        # plot_testX_timeseries(df_data2.iloc[:,3].to_numpy().reshape(df_data2.shape[0],1), feature_names="$zx_1$", title="$zx_1$", figsize=(6, 3), show=True, seperate=False, start_idx=0, _range=[0,0])
        # plot_testX_timeseries(df_data2.iloc[:,4].to_numpy().reshape(df_data2.shape[0],1), feature_names="$ewu$", title="$ewu$", figsize=(6, 3), show=True, seperate=False, start_idx=0, _range=[0,0])
        # plot_testX_timeseries(df_data2.iloc[:,5].to_numpy().reshape(df_data2.shape[0],1), feature_names="$ewx$", title="$ewx$", figsize=(6, 3), show=True, seperate=False, start_idx=0, _range=[0,0])
        # df_origin = (data_nor * data_std) + data_mean

        # plot_testX_timeseries(z_recovered, feature_names="$\Delta{U}$", title="$\Delta{U}$", figsize=(6, 3), show=True, seperate=False, start_idx=0, _range=[0,0])
        # plot_testX_timeseries(x_recovered[:, 0].reshape(x_recovered.shape[0],1), feature_names="$U^{pre1}$", title="$U^{pre1}$", figsize=(6, 3), show=True, seperate=False, start_idx=0, _range=[0,0])
        # plot_testX_timeseries((x_recovered[:, 0].unsqueeze(1) + z_recovered), feature_names="$U^{pre1}$ + $\Delta{U}$", title="$U^{pre1}$ + $\Delta{U}$", figsize=(6, 3), show=True, seperate=False, start_idx=0, _range=[0,0])
        # plot_testX_timeseries((x_recovered[:, 0].unsqueeze(1) - z_recovered), feature_names="$U^{pre1}$ - $\Delta{U}$", title="$U^{pre1}$ - $\Delta{U}$", figsize=(6, 3), show=True, seperate=False, start_idx=0, _range=[0,0])

        df_data, df_data2 = DiagnosisFeature(
            ERRORU, ERRORX, get_true_feature=analyse_ae_output
        )
        # plot_ae_output_distribution(df_data2, do_plot=analyse_ae_output)

        # u_max_idx = find_max_idx_from_array(df_data2.iloc[:,0])[0][0]
        # ci_max_idx = find_max_idx_from_array(CI_array)[0][0]
        # [[df_data2.iloc[u_max_idx,0],df_data2.iloc[u_max_idx,1],df_data2.iloc[u_max_idx,2],df_data2.iloc[u_max_idx,3],df_data2.iloc[u_max_idx,4],df_data2.iloc[u_max_idx,5]],
        #  [df_data2.iloc[ci_max_idx ,0],df_data2.iloc[ci_max_idx ,1],df_data2.iloc[ci_max_idx ,2],df_data2.iloc[ci_max_idx ,3],df_data2.iloc[ci_max_idx ,4],df_data2.iloc[ci_max_idx ,5]]]
        # [[df_data.iloc[u_max_idx,0],df_data.iloc[u_max_idx,1],df_data.iloc[u_max_idx,2],df_data.iloc[u_max_idx,3],df_data.iloc[u_max_idx,4],df_data.iloc[u_max_idx,5]],
        # [df_data.iloc[ci_max_idx ,0],df_data.iloc[ci_max_idx ,1],df_data.iloc[ci_max_idx ,2],df_data.iloc[ci_max_idx ,3],df_data.iloc[ci_max_idx ,4],df_data.iloc[ci_max_idx ,5]]]

        # max_err = np.array(df_data.iloc[:,2]).max()
        # max_err_time = np.where(np.array(df_data.iloc[:,2])> (max_err - 0.00001))
        # max_err2 = np.asarray(ERRORU[max_err_time,:], dtype=float).max(axis=1).max()
        # max_err_idx = np.where(np.asarray(ERRORU[max_err_time], dtype=float) > (max_err2 - 0.00001))

        # ## =================== Calculate thresholds =================== ###
        # sigma_levels = [float(s.strip()) for s in args.sigma_levels.split(',') if s.strip()]
        # new_X = (df_data - data_mean) / data_std
        # pi, g, h = chi_square_dist_components(p_k, v_I, np.cov(new_X.T), SPE_95_limit, T_95_limit)
        # thresholds = diagnosis_thresholds(g, h, sigma_levels=sigma_levels, show_plot=False)

        ## =================== Testing Diagnosis =================== ##
        # feature_names=["zx1","zx2","ewx","zu1","zu2","ewu"]
        t2_array, t2_contrib = T2_array(df_data, data_mean, data_std, p_k, v_I)
        spe_array, spe_contrib = SPE_array(df_data, data_mean, data_std, p_k)
        CI_contrib = (spe_contrib / SPE_95_limit) + (t2_contrib / T_95_limit)
        CI_array = (spe_array / SPE_95_limit) + (t2_array / T_95_limit)
        # title = ["$zu_2$", "$zx_2$", "$zu_1$", "$zx_1$", "$ewu$", "$ewx$"]
        # plot_contrib_percent_stacked(t2_contrib,val_array=t2_array,feature_names=title,top_k=100,mode="abs",kind="$T^2$",)
        # plot_contrib_percent_stacked(spe_contrib,val_array=spe_array,feature_names=title,top_k=100,mode="abs",kind="SPE",)
        # plot_contrib_percent_stacked(CI_contrib,val_array=CI_array,feature_names=title,top_k=100,mode="abs",kind="CI",)
        # plot_contrib_percent_stacked(
        #     CI_contrib, val_array=CI_array, feature_names=title,
        #     kind="CI", mode="abs", plot_step=100, xtick_step=10
        #     )

        # for i in range(len(title)):
        #     plot_testX_timeseries(t2_contrib[:, i].reshape(t2_contrib.shape[0],1), feature_names=title[i], title="t2_"+title[i], figsize=(6, 3), show=True, seperate=False, start_idx=0, _range=[0,0])

        # for i in range(len(title)):
        #     plot_testX_timeseries(spe_contrib[:, i].reshape(spe_contrib.shape[0],1), feature_names=title[i], title="spe_"+title[i], figsize=(6, 3), show=True, seperate=False, start_idx=0, _range=[0,0])

        if AUC_ANALYSIS:
            # =================== Prediction Matrix for ROC ===================
            # 각 threshold에 대해: CI_array에서 threshold를 넘는 값이 하나라도 있으면 해당 차량을 Positive로 예측
            ci_max = float(np.nanmax(np.asarray(CI_array, dtype=float)))
            predict_results[test_idx, :] = (ci_max > predict_thresholds).astype(np.int8)
            y_true[test_idx] = np.int8(label)
            test_idx += 1
        else:
            # =================== Plotting =================== ##
            title = ["$zu_2$", "$zx_2$", "$zu_1$", "$zx_1$", "$ewu$", "$ewx$"]
            start1 = time.perf_counter()
            # plot_contrib_percent_stacked(
            #     CI_contrib,
            #     val_array=CI_array,
            #     save_path=f"{args.models_dir}/{args.models_folder}/results_ci/{VEHICLE_ID}_testing_contrib_top{learning_case}.png",
            #     show=False,
            #     feature_names=title,
            #     top_k=100,
            #     mode="abs",
            #     kind="CI",
            # )
            # plot_contrib_percent_stacked(
            #     CI_contrib,
            #     val_array=CI_array,
            #     save_path=f"{args.models_dir}/{args.models_folder}/results_ci/{VEHICLE_ID}_testing_contrib{learning_case}.png",
            #     show=False,
            #     feature_names=title,
            #     kind="CI",
            #     mode="abs",
            #     plot_step=100,
            #     xtick_step=10,
            # )
            plot_diagnostics_triplet(
                t2_array,
                spe_array,
                CI_array,
                thresholds,
                sigma_levels=None,
                title="Test",
                x_label="Time",
                y_labels=("T²", "SPE", "CI"),
                figsize=(12, 10),
                save_path=f"{args.models_dir}/{args.models_folder}/results/{VEHICLE_ID}_testing_diagnostics_case{learning_case}.png",
                show=False,
                x_start=args.x_start,
                x_tick_step=args.x_tick_step,
                downsample=1,  # 1 ~ 10
                tight_layout=False,
                dpi=100,  # 100 ~ 150
                png_compress_level=1,
            )
            elapsed1 = time.perf_counter() - start1
            print(f"Plotting time for vehicle ID={VEHICLE_ID}: {elapsed1:.3f} seconds")
        print("Done")


elapsed = time.perf_counter() - start
h, rem = divmod(elapsed, 3600)
m, s = divmod(rem, 60)
print(f"{int(h)}시간 {int(m)}분 {s:.3f}초")

if AUC_ANALYSIS:
    # =================== ROC / AUC ===================
    roc_save_path = (
        f"{args.models_dir}/{args.models_folder}/results/AUC_ROC{learning_case}.png"
    )
    roc = compute_roc_auc_from_threshold_matrix(
        y_true, predict_results, predict_thresholds
    )
    opt = compute_optimal_thresholds_from_roc(
        roc["fpr"], roc["tpr"], roc["thresholds"], candidate_idx=roc["candidate_idx"]
    )

    print(f"AUC = {roc['auc']:.4f}")
    print(
        f"Best threshold (Youden J) ≈ {opt['best']['threshold']:.6g} @ (FPR={opt['best']['fpr']:.4f}, TPR={opt['best']['tpr']:.4f})"
    )
    print(
        f"Closest to (0,1) ≈ {opt['closest_to_01']['threshold']:.6g} @ (FPR={opt['closest_to_01']['fpr']:.4f}, TPR={opt['closest_to_01']['tpr']:.4f}), dist={opt['closest_to_01']['distance']:.4f}"
    )

    plot_roc_curve(
        roc["fpr"],
        roc["tpr"],
        roc["auc"],
        best_point=opt["best"],
        closest_to_01_point=opt["closest_to_01"],
        save_path=roc_save_path,
        title="AUC-ROC Curve",
        figsize=(6, 6),
        show=True,
        dpi=200,
    )
    print(f"ROC curve saved to: {roc_save_path}")
