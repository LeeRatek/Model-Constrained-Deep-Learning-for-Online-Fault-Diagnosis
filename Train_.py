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
import torch.nn as nn
# from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import torch.nn.functional as F
from torch import nn

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
print(torch.cuda.device_count())

warnings.filterwarnings('ignore')

#----------------------------------------data loading------------------------------
# VIN_data = pd.read_excel('./data/Name_list.xls')  # Changed path and removed Chinese characters
# print(os.getcwd())
# VIN_data = pd.read_excel('./data/Fig3abfandFigS13.xlsx')  # Changed path and removed Chinese characters
# vin = VIN_data.iloc[1, 0]
# print(vin)

# lstm = torch.load('./models/lstm.pth').to(device)  # Changed path
# with open('./data/test/' + vin + '/vin_1.pkl', 'rb') as file:  # Changed path
#     test_X = pickle.load(file)
# test

# DTI는 79까지, QAS는 392까지
# for i in range(79 + 1):
#     vin = f'VIN_{i}'
#     print(vin)
#     test_X = safe_load(f'./data/DTI/{i}/vin_1.pkl')
#     if (sum(test_X[:,:,2])[0] != 0):
#         print(f"Skipping {vin} due to zero current data.")
#         plot_testX_timeseries(test_X, feature_names="combined_tensor", title="combined_tensor", figsize=(12, 6), save_path=f'./', show=True, seperate=True, start_idx=0,_range=[2,2])
#         continue


PLOT_MODE = False
LSTM_TRAINING = False
LSTM_LOAD = False if not LSTM_TRAINING else True
# for i in range(392 + 1):
for i in [0]:
    BATTERY_TYPE = 'QAS' # 'DTI' or 'QAS'
    VEHICLE_ID = f'{i}'
    PATH = f"./data/data_analysis/{BATTERY_TYPE}-{VEHICLE_ID}"
    # PATH = f"./temp/{BATTERY_TYPE}-{VEHICLE_ID}"
    dir_ = os.path.dirname(PATH)
    if dir_:
        os.makedirs(dir_, exist_ok=True)

    vin1_start = 1000
    vin2_start = 1000
    vin3_start = 1000   

    #----------------------------------------Data loading for LSTM (customized) ------------------------------
    # print(os.getcwd())
    if LSTM_LOAD:
        test_X = safe_load(f'./data/{BATTERY_TYPE}/{VEHICLE_ID}/vin_1.pkl')

    #----------------------------------------Hyper Parameter Setting---------------------
    TIME_STEP = 1  # rnn time step
    INPUT_SIZE = 7  # rnn input size (전압은 total voltage, 온도는 평균 온도, 전류는 동일하니 1개로 표현, SOC는 board end SOC)
    LR = 1e-4  # learning rate
    lr_decay_freq = 25
    EPOCH = 100
    BATCH_SIZE = 100

    #----------------------------------------Training for overall state prediction--------
    if LSTM_TRAINING:
        train_X, train_y = prepare_training_data(test_X, INPUT_SIZE, TIME_STEP, device)

        train_dataset = MyDataset(train_X, train_y)
        train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)

        lstm = LSTM().to(device)
        optimizer = torch.optim.Adam(lstm.parameters(), lr=LR)  # optimize all cnn parameters
        loss_func = nn.MSELoss()

        # LSTM Training
        i = 0
        hidden_state = None
        loss_train_100 = []
        for epoch in range(EPOCH):
            for step, (b_x, b_y) in enumerate(train_loader):  # gives batch data
                lstm = lstm.double() # Casts all floating point parameters and buffers to ``double`` datatype.
                output = lstm(b_x)  # rnn output
                loss = loss_func(b_y, output)  # cross entropy loss
                optimizer.zero_grad()  # clear gradients for this training step
                loss.backward()  # backpropagation, compute gradients
                optimizer.step()  # apply gradients
                if step % 50 == 0:
                    loss_train_100.append(loss.cpu().detach().numpy())

    #----------------------------------------Data loading for MC-AE (customized) ------------------------------
    combined_tensor = safe_load(f'./data/{BATTERY_TYPE}/{VEHICLE_ID}/vin_2.pkl')
    combined_tensorx = safe_load(f'./data/{BATTERY_TYPE}/{VEHICLE_ID}/vin_3.pkl')

    #----------------------------------------Training for MC-AE--------------------------
    dim_x = 2 # [estimated pack voltage 1, estimated pack volatage 2]
    dim_y = 110 if BATTERY_TYPE == 'QAS' else 85 # DTI 일경우 85 이고 QAS일 경우 110인듯 [각 셀의 voltage]
    dim_z = 110 if BATTERY_TYPE == 'QAS' else 85 # DTI 일경우 85 이고 QAS일 경우 110인듯 [각 셀의 estimated voltage diviation]
    dim_q = 3 # [Board Temperature, Board-end SOC, Current]

    x_recovered = combined_tensor[:, :dim_x] # 0~1
    y_recovered = combined_tensor[:, dim_x:dim_x + dim_y] # 2~111
    z_recovered = combined_tensor[:, dim_x + dim_y: dim_x + dim_y + dim_z] # 112~221
    q_recovered = combined_tensor[:, dim_x + dim_y + dim_z:] # 222~

    dim_x2 = 2 # [estimated pack SOC 1, estimated pack SOC 2]
    dim_y2 = 110 if BATTERY_TYPE == 'QAS' else 85 # DTI 일경우 85 이고 QAS일 경우 110인듯 [각 셀의 SOC]
    dim_z2 = 110 if BATTERY_TYPE == 'QAS' else 85 # DTI 일경우 85 이고 QAS일 경우 110인듯 [각 셀의 estimated SOC diviation]
    dim_q2= 4 # [Board Temperature, Board-end SOC, Velocity, Current]

    x_recovered2 = combined_tensorx[:, :dim_x2]
    y_recovered2 = combined_tensorx[:, dim_x2:dim_x2 + dim_y2]
    z_recovered2 = combined_tensorx[:, dim_x2 + dim_y2: dim_x2 + dim_y2 + dim_z2]
    q_recovered2 = combined_tensorx[:, dim_x2 + dim_y2 + dim_z2:]

    #----------------------------------------Plotting the input data--------------------------
    if PLOT_MODE:
        plot_testX_timeseries(test_X, feature_names="Data for LSTM", title="vin1", figsize=(12, 6), save_path=f'{PATH}/vin1', show=False, seperate=True, start_idx=vin1_start)
        plot_testX_timeseries(combined_tensor, feature_names="Data for voltage estimation", title="vin2", figsize=(12, 6), save_path=f'{PATH}/vin2', show=False, seperate=True, start_idx=vin2_start, _range=[0,dim_x+2])
        plot_testX_timeseries(combined_tensor, feature_names="Data for voltage estimation", title="vin2", figsize=(12, 6), save_path=f'{PATH}/vin2', show=False, seperate=True, start_idx=vin2_start, _range=[dim_x + dim_y,dim_x + dim_y + 2])
        plot_testX_timeseries(combined_tensor, feature_names="Data for voltage estimation", title="vin2", figsize=(12, 6), save_path=f'{PATH}/vin2', show=False, seperate=True, start_idx=vin2_start, _range=[dim_x + dim_y + dim_z,dim_x + dim_y + dim_z + dim_q - 1])
        # plot_testX_timeseries(combined_tensor, feature_names="Data for voltage estimation", title="vin2", figsize=(12, 6), save_path=f'{PATH}/vin2', show=False, seperate=True, start_idx=1000, _range=[0,combined_tensor.shape[-1]])
        plot_testX_timeseries(combined_tensorx, feature_names="Data for SOC estimation", title="vin3", figsize=(12, 6), save_path=f'{PATH}/vin3', show=False, seperate=True, start_idx=vin3_start, _range=[0,dim_x2+2])
        plot_testX_timeseries(combined_tensorx, feature_names="Data for SOC estimation", title="vin3", figsize=(12, 6), save_path=f'{PATH}/vin3', show=False, seperate=True, start_idx=vin3_start, _range=[dim_x2 + dim_y2,dim_x2 + dim_y2 + 2])
        plot_testX_timeseries(combined_tensorx, feature_names="Data for SOC estimation", title="vin3", figsize=(12, 6), save_path=f'{PATH}/vin3', show=False, seperate=True, start_idx=vin3_start, _range=[dim_x2 + dim_y2 + dim_z2,dim_x2 + dim_y2 + dim_z2 + dim_q2 - 1])

if PLOT_MODE:
    exit()

EPOCH = 300
LR = 5e-4
BATCHSIZE = 100
class Dataset(Dataset):
    def __init__(self, x, y, z, q):
        self.x = x.to(torch.double)
        self.y = y.to(torch.double)
        self.z = z.to(torch.double)
        self.q = q.to(torch.double)
    def __len__(self):
        return len(self.x)
    def __getitem__(self, idx):
        return self.x[idx], self.y[idx], self.z[idx], self.q[idx]
train_loader_u = DataLoader(Dataset(x_recovered, y_recovered, z_recovered, q_recovered), batch_size=BATCHSIZE,
                      shuffle=False)

# Instantiate the networks
net = CombinedAE(input_size=2, encode2_input_size=3, output_size=110, activation_fn=custom_activation, use_dx_in_forward=True).to(device)
netx = CombinedAE(input_size=2, encode2_input_size=4, output_size=110, activation_fn=torch.sigmoid, use_dx_in_forward=True).to(device)

optimizer = torch.optim.Adam(net.parameters(), lr=LR)
l1_lambda = 0.01
loss_f = nn.MSELoss()
for epoch in range(EPOCH):
    total_loss = 0
    num_batches = 0
    for iteration, (x, y, z, q) in enumerate(train_loader_u):
        x = x.to(device)
        y = y.to(device)
        z = z.to(device)
        q = q.to(device)
        net = net.double()
        recon_im , recon_p = net(x,z,q)
        loss_u = loss_f(y,recon_im)
        total_loss += loss_u.item()
        num_batches += 1
        optimizer.zero_grad()
        loss_u.backward()
        optimizer.step()
    avg_loss = total_loss / num_batches
    print('Epoch: {:2d} | Average Loss: {:.4f}'.format(epoch, avg_loss))
    
save_net_state(model=net, models_dir=f"./models/", filename='net.pth')
train_loader2 = DataLoader(Dataset(x_recovered, y_recovered, z_recovered, q_recovered), batch_size=len(x_recovered),
                        shuffle=False)
for iteration, (x, y, z, q) in enumerate(train_loader2):
    x = x.to(device)
    y = y.to(device)
    z = z.to(device)
    q = q.to(device)
    net = net.double()
    recon_imtest, recon = net(x, z, q)
AA = recon_imtest.cpu().detach().numpy()
yTrainU = y_recovered.cpu().detach().numpy()
ERRORU = AA - yTrainU



train_loader_soc = DataLoader(Dataset(x_recovered2, y_recovered2, z_recovered2, q_recovered2), batch_size=BATCHSIZE, shuffle=False)
optimizer = torch.optim.Adam(netx.parameters(), lr=LR)
loss_f = nn.MSELoss()
avg_loss_list_x = []
for epoch in range(EPOCH):
    total_loss = 0
    num_batches = 0
    for iteration, (x, y, z, q) in enumerate(train_loader_soc):
        x = x.to(device)
        y = y.to(device)
        z = z.to(device)
        q = q.to(device)
        netx = netx.double()
        recon_im , z  = netx(x,z,q)
        loss_x = loss_f(y,recon_im)
        total_loss += loss_x.item()
        num_batches += 1
        optimizer.zero_grad()
        loss_x.backward()
        optimizer.step()
    avg_loss = total_loss / num_batches
    avg_loss_list_x.append(avg_loss)
    print('Epoch: {:2d} | Average Loss: {:.4f}'.format(epoch, avg_loss))
save_net_state(model=netx, models_dir=f"./models/", filename='netx.pth')

train_loaderx2 = DataLoader(Dataset(x_recovered2, y_recovered2, z_recovered2, q_recovered2), batch_size=len(x_recovered2), shuffle=False)
for iteration, (x, y, z, q) in enumerate(train_loaderx2):
    x = x.to(device)
    y = y.to(device)
    z = z.to(device)
    q = q.to(device)
    netx = netx.double()
    recon_imtestx, z = netx(x, z, q)

BB = recon_imtestx.cpu().detach().numpy()
yTrainX = y_recovered2.cpu().detach().numpy()
ERRORX = BB - yTrainX

df_data = DiagnosisFeature(ERRORU,ERRORX)

results = PCA(df_data,0.99,0.99)

save_pca_results(f"./models/", results)
(v_I, v, v_ratio, p_k, data_mean, data_std, T_95_limit, T_99_limit, SPE_95_limit, SPE_99_limit, P, k, P_t, X, data_nor) = results
t2_array = T2_array(df_data, data_mean, data_std, p_k, v_I)
plot_array(t2_array, title="Hotelling T² over time",
            figsize=(12, 4), save_path=None, show=True)

spe_array = SPE_array(df_data, data_mean, data_std, p_k)
plot_array(spe_array, title="Squared Prediction Error over time",       
            figsize=(12, 4), save_path=None, show=True)

