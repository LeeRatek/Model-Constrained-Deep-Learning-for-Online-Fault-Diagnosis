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

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
print(torch.cuda.device_count())

warnings.filterwarnings('ignore')
# VIN_data = pd.read_excel(r'{args.source_data_dir}/Name_list.xls')

parser = argparse.ArgumentParser(description="Run diagnostics plotting with CLI options")
parser.add_argument("--battery-type", choices=["QAS","DTI"], default="QAS", help="배터리 유형 선택 (DTI fault index range = [77~79], QAS fault index range = [335~392])")
parser.add_argument("--models-dir", type=str, default="./models/260123_082222") # 260123_082222 & 260126_100030
# parser.add_argument("--results-dir", type=str, default="./results" if os.environ.get("RESULT_DIR") is None else os.environ.get("RESULT_DIR"))
parser.add_argument("--source-data-dir", type=str, default="./data" if os.environ.get("SOURCE_DIR") is None else os.environ.get("SOURCE_DIR"))
parser.add_argument("--x-start", type=int, default=20)
parser.add_argument("--x-tick-step", type=int, default=3000)
parser.add_argument("--sigma-levels", type=str, default="3,4.5,6")
parser.add_argument("--normalize-dx", action="store_true")
parser.add_argument("--normalize-val", type=int, default=4)
args = parser.parse_args()

learning_case = read_learning_case_from_sim_config(args.models_dir)

# 플롯 결과 저장 폴더는 한 번만 생성
os.makedirs(f"{args.models_dir}/results", exist_ok=True)

BATTERY_TYPE = args.battery_type # 'DTI' or 'QAS'
PREPROCESSING, SKIP_CHARGE_READY = get_preprocessing_and_skip_charge_ready(learning_case)
dim_dict = get_input_dimensions(BATTERY_TYPE)
AUC_ANALYSIS = False
if not AUC_ANALYSIS:
    thresholds = [40, 50]
    
print_sim_config(
    title="Train Run",
    config=args,
    extra={
        "device": str(device),
        "learning_case": learning_case,
        "PREPROCESSING": PREPROCESSING,
        "SKIP_CHARGE_READY": SKIP_CHARGE_READY
        },
    )

# DTI fault index range = [77~79]
# QAS fault index range = [335~392]
sim_config_path = os.path.join(args.models_dir, "sim_config.txt")
train_vehicle_ids = load_vehicle_ids_used_for_training(sim_config_path)
print(f"Vehicle IDs used for training: {train_vehicle_ids}")

first_fault_vehicle_id = 335 if BATTERY_TYPE == "QAS" else 77
all_normal_vehicle_ids = list(range(0, first_fault_vehicle_id))
normal_list = [vid for vid in all_normal_vehicle_ids if vid not in set(train_vehicle_ids)]
fault_list = np.load(f"./{BATTERY_TYPE}_filtered_vehicle_ids_fault.npy").astype(np.int64).tolist()
test_list = [normal_list, fault_list]

total_test_vehicles = len(normal_list) + len(fault_list)
    
predict_threshold_array = range(0, 1000, 2)
predict_thresholds = np.asarray(list(predict_threshold_array), dtype=float)
predict_results = np.zeros((total_test_vehicles, len(predict_thresholds)), dtype=np.int8)
y_true = np.zeros(total_test_vehicles, dtype=np.int8)  # normal=0, fault=1

## =================== Load training data  =================== ##
start_pca_load = time.perf_counter()
loads = load_pca_results(args.models_dir, load_data_nor=False, validate_shapes=False)
(v_I, v, v_ratio, p_k, data_mean, data_std,
    T_95_limit, T_99_limit, SPE_95_limit, SPE_99_limit,
    P, k, P_t, X, data_nor) = loads
elapsed_pca_load = time.perf_counter() - start_pca_load
print(f"PCA results loading time (once): {elapsed_pca_load:.3f} seconds")
if PREPROCESSING:
    net_loaded = CombinedAE(input_size=dim_dict["x"], encode2_input_size=dim_dict["q"], output_size=dim_dict["y"],
                        activation_fn=torch.sigmoid, use_dx_in_forward=True).to(device)
else:
    net_loaded = CombinedAE(input_size=dim_dict["x"], encode2_input_size=dim_dict["q"], output_size=dim_dict["y"],
                        activation_fn=custom_activation, use_dx_in_forward=True).to(device)
netx_loaded = CombinedAE(input_size=dim_dict["x2"], encode2_input_size=dim_dict["q2"], output_size=dim_dict["y2"], activation_fn=torch.sigmoid,
                        use_dx_in_forward=True).to(device)
net_state_dict = torch.load(os.path.join(args.models_dir, 'net.pth'))
net_loaded.load_state_dict(net_state_dict)
netx_state_dict = torch.load(os.path.join(args.models_dir, 'netx.pth'))
netx_loaded.load_state_dict(netx_state_dict)

start = time.perf_counter()
test_idx = 0
for label, vehicle_ids in enumerate(test_list):    
    for i in vehicle_ids:
        print(f"Processing label={label} with vehicle ID={i}...")
        elapsed = time.perf_counter() - start
        h, rem = divmod(elapsed, 3600)
        m, s = divmod(rem, 60)
        print(f"Elapsed time: {int(h)}시간 {int(m)}분 {s:.3f}초")
        VEHICLE_ID = f'{i}'
        
        # lstm = torch.load('./models/lstm.pth').to(device)
        # test_X = safe_load(f'{args.source_data_dir}/{BATTERY_TYPE}/{VEHICLE_ID}/vin_1.pkl', verbose=False)
        # # test
        # lstm.eval()
        # prediction = lstm(test_X)
        
        
        combined_tensor = safe_load(f'{args.source_data_dir}/{BATTERY_TYPE}/{VEHICLE_ID}/vin_2.pkl', verbose=False)
        combined_tensorx = safe_load(f'{args.source_data_dir}/{BATTERY_TYPE}/{VEHICLE_ID}/vin_3.pkl', verbose=False)
        
        combined_tensor, combined_tensorx = preprocess_combined_tensor(combined_tensor, combined_tensorx, dim_dict, 
                                                                       BATTERY_TYPE, PREPROCESSING, SKIP_CHARGE_READY, 
                                                                       normalize_dx=args.normalize_dx)
        
        # Use indexing to separate
        x_recovered = combined_tensor[:, :dim_dict["x"]]
        y_recovered = combined_tensor[:, dim_dict["x"]:dim_dict["x"] + dim_dict["y"]]
        z_recovered = combined_tensor[:, dim_dict["x"] + dim_dict["y"]: dim_dict["x"] + dim_dict["y"] + dim_dict["z"]]
        q_recovered = combined_tensor[:, dim_dict["x"] + dim_dict["y"] + dim_dict["z"]:]
        net_loaded = net_loaded.double().eval()

        with torch.inference_mode():
            recon_imtest = net_loaded(x_recovered, z_recovered, q_recovered)

        # Use indexing to separate
        x_recovered2 = combined_tensorx[:, :dim_dict["x2"]]
        y_recovered2 = combined_tensorx[:, dim_dict["x2"]:dim_dict["x2"] + dim_dict["y2"]]
        z_recovered2 = combined_tensorx[:, dim_dict["x2"] + dim_dict["y2"]: dim_dict["x2"] + dim_dict["y2"] + dim_dict["z2"]]
        q_recovered2 = combined_tensorx[:, dim_dict["x2"] + dim_dict["y2"] + dim_dict["z2"]:]
        netx_loaded = netx_loaded.double().eval()

        with torch.inference_mode():
            reconx_imtest = netx_loaded(x_recovered2, z_recovered2, q_recovered2)

            # 출력/정답을 각각 numpy로 변환하지 말고 torch에서 error를 먼저 계산 후 1회만 변환
            ERRORU = (recon_imtest[0] - y_recovered).cpu().numpy()
            ERRORX = (reconx_imtest[0] - y_recovered2).cpu().numpy()
        
        df_data = DiagnosisFeature(ERRORU,ERRORX)
        
        
        # ## =================== Calculate thresholds =================== ###        
        # sigma_levels = [float(s.strip()) for s in args.sigma_levels.split(',') if s.strip()]
        # new_X = (df_data - data_mean) / data_std
        # pi, g, h = chi_square_dist_components(p_k, v_I, np.cov(new_X.T), SPE_95_limit, T_95_limit)
        # thresholds = diagnosis_thresholds(g, h, sigma_levels=sigma_levels, show_plot=False)
        
        
        
        ## =================== Testing Diagnosis =================== ##
        t2_array = T2_array(df_data, data_mean, data_std, p_k, v_I)
        spe_array = SPE_array(df_data, data_mean, data_std, p_k)
        CI_array = (spe_array / SPE_95_limit) + (t2_array / T_95_limit)


        if AUC_ANALYSIS:    
            # =================== Prediction Matrix for ROC ===================
            # 각 threshold에 대해: CI_array에서 threshold를 넘는 값이 하나라도 있으면 해당 차량을 Positive로 예측
            ci_max = float(np.nanmax(np.asarray(CI_array, dtype=float)))
            predict_results[test_idx, :] = (ci_max > predict_thresholds).astype(np.int8)
            y_true[test_idx] = np.int8(label)
            test_idx += 1
        else:
            # =================== Plotting =================== ##
            start1 = time.perf_counter()
            plot_diagnostics_triplet(t2_array,
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
                                    downsample=10, # 1 ~ 10
                                    tight_layout=False,
                                    dpi=100, # 100 ~ 150
                                    png_compress_level=1)
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
    plot_auc_roc_curve_from_threshold_matrix(
        y_true,
        predict_results,
        predict_thresholds,
        save_path=roc_save_path,
        title="AUC-ROC Curve",
        figsize=(6, 6),
        show=True,
    )



