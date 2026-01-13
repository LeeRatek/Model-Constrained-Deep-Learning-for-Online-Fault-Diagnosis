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

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
print(torch.cuda.device_count())

warnings.filterwarnings('ignore')
# VIN_data = pd.read_excel(r'./data/Name_list.xls')


# DTI fault index range = [77~79]
# QAS fault index range = [335~392]
# for i in range(1):
for i in [335]:
    BATTERY_TYPE = 'QAS' # 'DTI' or 'QAS'
    VEHICLE_ID = f'{i}'
    path = f'./data/{BATTERY_TYPE}/{VEHICLE_ID}/'
    # vin = VIN_data.iloc[i, 0]
    # print(vin)

    # lstm = torch.load('./models/lstm.pth').to(device)
    # test_X = safe_load(f'./data/{BATTERY_TYPE}/{VEHICLE_ID}/vin_1.pkl')
    # # test
    # lstm.eval()
    # prediction = lstm(test_X)

    net_loaded = CombinedAE(input_size=2, encode2_input_size=3, output_size=110,
                            activation_fn=custom_activation, use_dx_in_forward=True).to(device)
    netx_loaded = CombinedAE(input_size=2, encode2_input_size=4, output_size=110, activation_fn=torch.sigmoid,
                             use_dx_in_forward=True).to(device)

    net_state_dict = torch.load('./models/net.pth')
    net_loaded.load_state_dict(net_state_dict)

    netx_state_dict = torch.load('./models/netx.pth')
    netx_loaded.load_state_dict(netx_state_dict)
    
    combined_tensor = safe_load(f'./data/{BATTERY_TYPE}/{VEHICLE_ID}/vin_2.pkl')
    combined_tensorx = safe_load(f'./data/{BATTERY_TYPE}/{VEHICLE_ID}/vin_3.pkl')

    dim_x = 2
    dim_y = 110 
    dim_z = 110 
    dim_q = 3

    # Use indexing to separate
    x_recovered = combined_tensor[:, :dim_x]
    y_recovered = combined_tensor[:, dim_x:dim_x + dim_y]
    z_recovered = combined_tensor[:, dim_x + dim_y: dim_x + dim_y + dim_z]
    q_recovered = combined_tensor[:, dim_x + dim_y + dim_z:]
    net_loaded = net_loaded.double()
    recon_imtest = net_loaded(x_recovered, z_recovered, q_recovered)

    dim_x2 = 2
    dim_y2 = 110
    dim_z2 = 110
    dim_q2= 4

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
    
    loads = load_pca_results(f"./models/")
    (v_I, v, v_ratio, p_k, data_mean, data_std,
        T_95_limit, T_99_limit, SPE_95_limit, SPE_99_limit,
        P, k, P_t, X, data_nor) = loads
    
    sigma_level = [3, 4.5, 6]
    pi, g, h = chi_square_dist_components(p_k, v_I, X, SPE_95_limit, T_95_limit)
    thresholds = diagnosis_thresholds(g, h, sigma_level=sigma_level, show_plot=False)
    
    ## =================== Training Diagnosis =================== ##
    training_data = (data_nor * data_std) + data_mean
    t2_array_train = T2_array(training_data, data_mean, data_std, p_k, v_I)
    spe_array_train = SPE_array(training_data, data_mean, data_std, p_k)
    CI_array_train = spe_array_train / SPE_95_limit + t2_array_train / T_95_limit
    
    
    plot_diagnostics_triplet(t2_array_train,
                             spe_array_train,
                             CI_array_train,
                             thresholds,
                             sigma_labels=sigma_level,
                             title="Training Diagnostics",
                             x_label="Time",
                             y_labels=("T²", "SPE", "CI"),
                             figsize=(12, 10),
                             save_path="./results/training_diagnostics.png",
                             show=False,
                             x_start=20,
                             x_tick_step=3000)
    
    ## =================== Testing Diagnosis =================== ##
    t2_array = T2_array(df_data, data_mean, data_std, p_k, v_I)
    spe_array = SPE_array(df_data, data_mean, data_std, p_k)
    CI_array = spe_array / SPE_95_limit + t2_array / T_95_limit
    
    plot_diagnostics_triplet(t2_array,
                             spe_array,
                             CI_array,
                             thresholds,
                             sigma_labels=sigma_level,
                             title="Testing Diagnostics",
                             x_label="Time",
                             y_labels=("T²", "SPE", "CI"),
                             figsize=(12, 10),
                             save_path="./results/testing_diagnostics.png",
                             show=False,
                             x_start=20,
                             x_tick_step=3000)
    
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



