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
    "--models-dir", type=str, default="./models/260206_084839"
)  # 260123_082222 & 260126_100030
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
args = parser.parse_args()

vals = read_values_from_sim_config(
    args.models_dir,
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
os.makedirs(f"{args.models_dir}/results", exist_ok=True)

BATTERY_TYPE = args.battery_type  # 'DTI' or 'QAS'
PREPROCESSING, SKIP_CHARGE_READY = get_preprocessing_and_skip_charge_ready(
    learning_case
)
# SKIP_CHARGE_READY = False
dim_dict = get_input_dimensions(BATTERY_TYPE)
AUC_ANALYSIS = False
if not AUC_ANALYSIS:
    thresholds = [18.7, 28.7]
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
    },
)


try:
    plot_loss_curve_all(args)
except Exception as e:
    print(f"Error plotting loss curve: {e}")


## =================== DTI fault index range = [77~79] ===================
## =================== QAS fault index range = [335~392] ===================

# sim_config_path = os.path.join(args.models_dir, "sim_config.txt")
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
loads = load_pca_results(args.models_dir, load_data_nor=False, validate_shapes=False)
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
net_state_dict = torch.load(os.path.join(f"{args.models_dir}/artifact/", "net.pth"))
net_loaded.load_state_dict(net_state_dict)
netx_state_dict = torch.load(os.path.join(f"{args.models_dir}/artifact/", "netx.pth"))
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
    save_dir=f"{args.models_dir}",
    layer_names=("fc1", "fc2"),
    figsize=(6, 3),
)
plot_net_layer_params_by_index(
    net_loaded,
    net_name="net",
    mode="bar",
    max_points=8000,
    show=showing_params,
    save_dir=f"{args.models_dir}",
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
    save_dir=f"{args.models_dir}",
    layer_names=("fc1", "fc2"),
    figsize=(6, 3),
)
plot_net_layer_params_by_index(
    netx_loaded,
    net_name="netx",
    mode="bar",
    max_points=8000,
    show=showing_params,
    save_dir=f"{args.models_dir}",
    layer_names=("fc1", "fc2"),
    figsize=(6, 3),
    normalize_y=False,
)

start = time.perf_counter()
test_idx = 0
for label, vehicle_ids in enumerate(test_list):
    for i in vehicle_ids:
        # i = 382
        print(f"Processing label={label} with vehicle ID={i}...")
        elapsed = time.perf_counter() - start
        h, rem = divmod(elapsed, 3600)
        m, s = divmod(rem, 60)
        print(f"Elapsed time: {int(h)}시간 {int(m)}분 {s:.3f}초")
        VEHICLE_ID = f"{i}"

        # lstm = torch.load('./models/lstm.pth').to(device)
        # test_X = safe_load(f'{args.source_data_dir}/{BATTERY_TYPE}/{VEHICLE_ID}/vin_1.pkl', verbose=False)
        # # test
        # lstm.eval()
        # prediction = lstm(test_X)

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

        # plot_testX_timeseries(z_recovered, feature_names="$\Delta{U}$", title="$\Delta{U}$", figsize=(6, 3), show=True, seperate=False, start_idx=0, _range=[0,0])
        # plot_testX_timeseries(x_recovered[:, 0].reshape(x_recovered.shape[0],1), feature_names="$U^{pre1}$", title="$U^{pre1}$", figsize=(6, 3), show=True, seperate=False, start_idx=0, _range=[0,0])
        # plot_testX_timeseries((x_recovered[:, 0].unsqueeze(1) + z_recovered), feature_names="$U^{pre1}$ + $\Delta{U}$", title="$U^{pre1}$ + $\Delta{U}$", figsize=(6, 3), show=True, seperate=False, start_idx=0, _range=[0,0])
        # plot_testX_timeseries((x_recovered[:, 0].unsqueeze(1) - z_recovered), feature_names="$U^{pre1}$ - $\Delta{U}$", title="$U^{pre1}$ - $\Delta{U}$", figsize=(6, 3), show=True, seperate=False, start_idx=0, _range=[0,0])

        df_data, df_data2 = DiagnosisFeature(
            ERRORU, ERRORX, get_true_feature=analyse_ae_output
        )
        plot_ae_output_distribution(ERRORU, ERRORX, df_data2, do_plot=analyse_ae_output)

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
            #     save_path=f"{args.models_dir}/results_ci/{VEHICLE_ID}_testing_contrib_top{learning_case}.png",
            #     show=False,
            #     feature_names=title,
            #     top_k=100,
            #     mode="abs",
            #     kind="CI",
            # )
            # plot_contrib_percent_stacked(
            #     CI_contrib,
            #     val_array=CI_array,
            #     save_path=f"{args.models_dir}/results_ci/{VEHICLE_ID}_testing_contrib{learning_case}.png",
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
                save_path=f"{args.models_dir}/results/{VEHICLE_ID}_testing_diagnostics_case{learning_case}.png",
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

        ## =================== Compare to others code =================== ##
        # df = pd.DataFrame((data_nor * data_std) + data_mean) # 원래 데이터 복원
        # test = T2(scaling=True,explained_variance=4,p_value=0.95)
        # test.fit(df)
        # t_array, q_array = test.predict(df_data)
        # ci_array = np.array(q_array) / SPE_95_limit +   np.array(t_array) / T_95_limit
        # plot_diagnostics_triplet(t_array,
        #                         q_array,
        #                         ci_array,
        #                         thresholds,
        #                         sigma_levels=sigma_levels,
        #                         title="Test",
        #                         x_label="Time",
        #                         y_labels=("T²", "SPE", "CI"),
        #                         figsize=(12, 10),
        #                         save_path=None,
        #                         show=True,
        #                         x_start=args.x_start,
        #                         x_tick_step=args.x_tick_step)
        print("Done")

elapsed = time.perf_counter() - start
h, rem = divmod(elapsed, 3600)
m, s = divmod(rem, 60)
print(f"{int(h)}시간 {int(m)}분 {s:.3f}초")

if AUC_ANALYSIS:
    # =================== ROC / AUC ===================
    roc_save_path = f"{args.models_dir}/results/AUC_ROC{learning_case}.png"
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
