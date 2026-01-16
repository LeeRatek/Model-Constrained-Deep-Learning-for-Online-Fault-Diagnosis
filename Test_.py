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
import math
import math
from sklearn import preprocessing
import torch
import torch.nn as nn
import torch.optim as optim
# from sklearn.datasets import load_boston
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
import scipy.io as scio
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import torch.nn.functional as F
from torch import nn
from torchvision import transforms as tfs
import scipy.stats as stats
import seaborn as sns
import pickle
import argparse

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
print(torch.cuda.device_count())

warnings.filterwarnings('ignore')
# VIN_data = pd.read_excel(r'{args.source_data_dir}/Name_list.xls')

parser = argparse.ArgumentParser(description="Run diagnostics plotting with CLI options")
parser.add_argument("--battery-type", choices=["QAS","DTI"], default="QAS", help="배터리 유형 선택 (DTI fault index range = [77~79], QAS fault index range = [335~392])")
parser.add_argument("--vehicle-start", type=int, default=0, help="Filtered fault vehicle ID 시작 인덱스")
parser.add_argument("--vehicle-end", type=int, default=0, help="Filtered fault vehicle ID 끝 인덱스")
parser.add_argument("--models-dir", type=str, default="./models/260115_102724")
parser.add_argument("--results-dir", type=str, default="./results" if os.environ.get("RESULT_DIR") is None else os.environ.get("RESULT_DIR"))
parser.add_argument("--source-data-dir", type=str, default="./data" if os.environ.get("SOURCE_DIR") is None else os.environ.get("SOURCE_DIR"))
parser.add_argument("--x-start", type=int, default=20)
parser.add_argument("--x-tick-step", type=int, default=3000)
parser.add_argument("--sigma-levels", type=str, default="3,4.5,6")
parser.add_argument("--normalize-dx", action="store_true")
parser.add_argument("--normalize-val", type=int, default=4)
parser.add_argument("--learning-case", type=int, default=1, help="1: Skip and no nomalization, 2: Skip but doing normalization, 3: No skip but doing normalization, 4: No skip and no normalization.")
args = parser.parse_args()

BATTERY_TYPE = args.battery_type # 'DTI' or 'QAS'
if args.learning_case == 1:
    PREPROCESSING = False
    SKIP_CHARGE_READY = True
elif args.learning_case == 2:
    PREPROCESSING = True
    SKIP_CHARGE_READY = True
elif args.learning_case == 3:
    PREPROCESSING = True
    SKIP_CHARGE_READY = False
elif args.learning_case == 4:
    PREPROCESSING = False
    SKIP_CHARGE_READY = False
    
dim_x = 2 # [estimated pack voltage 1, estimated pack volatage 2]
dim_y = 110 if BATTERY_TYPE == 'QAS' else 85 # DTI 일경우 85 이고 QAS일 경우 110인듯 [각 셀의 voltage]
dim_z = 110 if BATTERY_TYPE == 'QAS' else 85 # DTI 일경우 85 이고 QAS일 경우 110인듯 [각 셀의 estimated voltage diviation]
dim_q = 3 # [Board Temperature, Board-end SOC, Current]

dim_x2 = 2 # [estimated pack SOC 1, estimated pack SOC 2]
dim_y2 = 110 if BATTERY_TYPE == 'QAS' else 85 # DTI 일경우 85 이고 QAS일 경우 110인듯 [각 셀의 SOC]
dim_z2 = 110 if BATTERY_TYPE == 'QAS' else 85 # DTI 일경우 85 이고 QAS일 경우 110인듯 [각 셀의 estimated SOC diviation]
dim_q2= 4 # [Board Temperature, Board-end SOC, Velocity, Current]

print_sim_config(
    title="Train Run",
    config=args,
    extra={
        "device": str(device),
    },
    # exclude=["password", "token"],  # 민감정보 방지용
    )
# DTI fault index range = [77~79]
# QAS fault index range = [335~392]
falt_list = np.load(f"./{BATTERY_TYPE}_filtered_vehicle_ids_fault.npy").astype(np.int64).tolist()
for i in falt_list[args.vehicle_start:args.vehicle_end+1]:
    VEHICLE_ID = f'{i}'
    path = f'{args.source_data_dir}/{BATTERY_TYPE}/{VEHICLE_ID}/'
    # vin = VIN_data.iloc[i, 0]
    # print(vin)

    # lstm = torch.load('./models/lstm.pth').to(device)
    # test_X = safe_load(f'{args.source_data_dir}/{BATTERY_TYPE}/{VEHICLE_ID}/vin_1.pkl')
    # # test
    # lstm.eval()
    # prediction = lstm(test_X)

    net_loaded = CombinedAE(input_size=2, encode2_input_size=3, output_size=110,
                            activation_fn=custom_activation, use_dx_in_forward=True).to(device)
    netx_loaded = CombinedAE(input_size=2, encode2_input_size=4, output_size=110, activation_fn=torch.sigmoid,
                             use_dx_in_forward=True).to(device)

    net_state_dict = torch.load(os.path.join(args.models_dir, 'net.pth'))
    net_loaded.load_state_dict(net_state_dict)

    netx_state_dict = torch.load(os.path.join(args.models_dir, 'netx.pth'))
    netx_loaded.load_state_dict(netx_state_dict)
    
    combined_tensor = safe_load(f'{args.source_data_dir}/{BATTERY_TYPE}/{VEHICLE_ID}/vin_2.pkl')
    combined_tensorx = safe_load(f'{args.source_data_dir}/{BATTERY_TYPE}/{VEHICLE_ID}/vin_3.pkl')
    
    
    if PREPROCESSING:
        start_idx = 0
        start_idx = np.max([start_idx,last_index_of(combined_tensor, feature_idx=0, value=0, operator='<=')]) # 0 means minimum voltage
        start_idx = np.max([start_idx,last_index_of(combined_tensor, feature_idx=0, value=5, operator='>')]) # 5 means maximum voltage
        cell_div_idxes = list(range(dim_x + dim_y, dim_x + dim_y + dim_z))
        combined_tensor = normalize_columns_0_to_1(combined_tensor[start_idx:, :], exclude_cols=cell_div_idxes)
        combined_tensorx = normalize_columns_0_to_1(combined_tensorx[start_idx:, :], exclude_cols=cell_div_idxes)
        if args.normalize_dx:
            combined_tensor[:,range(dim_x + dim_y, dim_x + dim_y + dim_z)] = combined_tensor[:,range(dim_x + dim_y, dim_x + dim_y + dim_z)].div(args.normalize_val)
            # tensorx[:,range(dim_x2 + dim_y2, dim_x2 + dim_y2 + dim_z2)] = tensorx[:,range(dim_x2 + dim_y2, dim_x2 + dim_y2 + dim_z2)].div(0.1)

    if SKIP_CHARGE_READY:
        charge_idx = 224 if BATTERY_TYPE == "QAS" else 174
        start_idx = last_index_of(combined_tensor, feature_idx=charge_idx, value=-500, operator='<') # -500 means minimum current during charge ready
        combined_tensor = combined_tensor[start_idx:, :]
        combined_tensorx = combined_tensorx[start_idx:, :]
    
    # Use indexing to separate
    x_recovered = combined_tensor[:, :dim_x]
    y_recovered = combined_tensor[:, dim_x:dim_x + dim_y]
    z_recovered = combined_tensor[:, dim_x + dim_y: dim_x + dim_y + dim_z]
    q_recovered = combined_tensor[:, dim_x + dim_y + dim_z:]
    net_loaded = net_loaded.double()
    recon_imtest = net_loaded(x_recovered, z_recovered, q_recovered)

    # Use indexing to separate
    x_recovered2 = combined_tensorx[:, :dim_x2]
    y_recovered2 = combined_tensorx[:, dim_x2:dim_x2 + dim_y2]
    z_recovered2 = combined_tensorx[:, dim_x2 + dim_y2: dim_x2 + dim_y2 + dim_z2]
    q_recovered2 = combined_tensorx[:, dim_x2 + dim_y2 + dim_z2:]
    netx_loaded = netx_loaded.double()
    reconx_imtest = netx_loaded(x_recovered2, z_recovered2, q_recovered2)

    AA = recon_imtest[0].cpu().detach().numpy()
    yTrainU = y_recovered.cpu().detach().numpy()
    ERRORU = AA - yTrainU

    BB = reconx_imtest[0].cpu().detach().numpy()
    yTrainX = y_recovered2.cpu().detach().numpy()
    ERRORX = BB - yTrainX

    df_data = DiagnosisFeature(ERRORU,ERRORX)
    
    loads = load_pca_results(args.models_dir)
    (v_I, v, v_ratio, p_k, data_mean, data_std,
        T_95_limit, T_99_limit, SPE_95_limit, SPE_99_limit,
        P, k, P_t, X, data_nor) = loads
    
    sigma_levels = [float(s.strip()) for s in args.sigma_levels.split(',') if s.strip()]
    pi, g, h = chi_square_dist_components(p_k, v_I, X, SPE_95_limit, T_95_limit)
    thresholds = diagnosis_thresholds(g, h, sigma_levels=sigma_levels, show_plot=False)
    
    ## =================== Training Diagnosis =================== ##
    training_data = (data_nor * data_std) + data_mean
    t2_array_train = T2_array(training_data, data_mean, data_std, p_k, v_I)
    spe_array_train = SPE_array(training_data, data_mean, data_std, p_k)
    CI_array_train = spe_array_train / SPE_95_limit + t2_array_train / T_95_limit
    
    
    plot_diagnostics_triplet(t2_array_train,
                             spe_array_train,
                             CI_array_train,
                             thresholds,
                             sigma_levels=sigma_levels,
                             title="Training",
                             x_label="Time",
                             y_labels=("T²", "SPE", "CI"),
                             figsize=(12, 10),
                             save_path=f"{args.results_dir}/training_diagnostics_case{args.learning_case}.png",
                             show=False,
                             x_start=args.x_start,
                             x_tick_step=args.x_tick_step)
    
    ## =================== Testing Diagnosis =================== ##
    t2_array = T2_array(df_data, data_mean, data_std, p_k, v_I)
    spe_array = SPE_array(df_data, data_mean, data_std, p_k)
    CI_array = spe_array / SPE_95_limit + t2_array / T_95_limit
    
    plot_diagnostics_triplet(t2_array,
                             spe_array,
                             CI_array,
                             thresholds,
                             sigma_levels=sigma_levels,
                             title="Test",
                             x_label="Time",
                             y_labels=("T²", "SPE", "CI"),
                             figsize=(12, 10),
                             save_path=f"{args.results_dir}/testing_diagnostics_case{args.learning_case}.png",
                             show=False,
                             x_start=args.x_start,
                             x_tick_step=args.x_tick_step)
    
    print("Done")    

    # lamda, CONTN, t_total, q_total, S, FAI, g, h, kesi, fai,  f_time, level, maxlevel, contTT, contQ, X_ratio, CContn, data_mean, data_std = chi_square_dist_components(
    #     df_data.values, data_mean, data_std, v.reshape(len(v),1), p_k, v_I, T_99_limit, SPE_99_limit, X, time)
    #
    # nm = 3000
    # mm = len(fai)
    #
    # threshold1 = np.mean(fai[nm:mm]) + 3*np.std(fai[nm:mm])
    # threshold2 = np.mean(fai[nm:mm]) + 4.5*np.std(fai[nm:mm])
    # threshold3 = np.mean(fai[nm:mm]) + 6*np.std(fai[nm:mm])



