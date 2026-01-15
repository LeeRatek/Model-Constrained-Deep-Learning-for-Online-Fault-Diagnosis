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
import time
import argparse
from contextlib import redirect_stdout

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
print(torch.cuda.device_count())

warnings.filterwarnings('ignore')

parser = argparse.ArgumentParser(description="Run MC-AE training with CLI options")
parser.add_argument("--battery-type", choices=["QAS","DTI"], default="QAS")
parser.add_argument("--vehicle-start", type=int, default=0, help="Filtered normal vehicle ID 시작 인덱스")
parser.add_argument("--vehicle-end", type=int, default=3, help="Filtered normal vehicle ID 끝 인덱스")
parser.add_argument("--lstm-training", action="store_true")
parser.add_argument("--lstm-load", action="store_true")
parser.add_argument("--models-dir", type=str, default="./models" if os.environ.get("MODEL_DIR") is None else os.environ.get("MODEL_DIR"))
parser.add_argument("--source-data-dir", type=str, default="./data" if os.environ.get("SOURCE_DIR") is None else os.environ.get("SOURCE_DIR"))
parser.add_argument("--vin1-start", type=int, default=0)
parser.add_argument("--vin2-start", type=int, default=0)
parser.add_argument("--vin3-start", type=int, default=0)
parser.add_argument("--ae-epochs", type=int, default=800)
parser.add_argument("--ae-lr", type=float, default=5e-4)
parser.add_argument("--ae-batchsize", type=int, default=100)
parser.add_argument("--lstm-epochs", type=int, default=300)
parser.add_argument("--lstm-lr", type=float, default=5e-4)
parser.add_argument("--lstm-batchsize", type=int, default=100)
parser.add_argument("--learning-case", type=int, default=1, help="1: Skip and no nomalization, 2: Skip but doing normalization, 3: No skip but doing normalization, 4: No skip and no normalization.")
args = parser.parse_args()


LSTM_TRAINING = args.lstm_training
LSTM_LOAD = args.lstm_load if args.lstm_training else args.lstm_load
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

# 모델 저장 경로 설정

model_path = make_model_path_based_timestamp(base=args.models_dir)

start = time.perf_counter()

normal_list = np.load(f"./{BATTERY_TYPE}_filtered_vehicle_ids_normal.npy").astype(np.int64).tolist()
vehicle_idxes = []
FIRST_LOAD = True
for i in normal_list[args.vehicle_start:args.vehicle_end+1]:
    VEHICLE_ID = f'{i}'
    vehicle_idxes.append(VEHICLE_ID)
    PATH = f"{args.source_data_dir}/data_analysis_start_at_0/{BATTERY_TYPE}-{VEHICLE_ID}"   

    #----------------------------------------Data loading for LSTM (customized) ------------------------------
    # print(os.getcwd())
    if LSTM_LOAD:
        test_X = safe_load(f'{args.source_data_dir}/{BATTERY_TYPE}/{VEHICLE_ID}/vin_1.pkl')

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
    tensor = safe_load(f'{args.source_data_dir}/{BATTERY_TYPE}/{VEHICLE_ID}/vin_2.pkl')
    tensorx = safe_load(f'{args.source_data_dir}/{BATTERY_TYPE}/{VEHICLE_ID}/vin_3.pkl')
    
    #----------------------------------------Preprocessing ------------------------------#
    
    if PREPROCESSING:
        start_idx = 0
        start_idx = np.max([start_idx,last_index_of(tensor, feature_idx=0, value=0, operator='<=')]) # 0 means minimum voltage
        start_idx = np.max([start_idx,last_index_of(tensor, feature_idx=0, value=5, operator='>')]) # 5 means maximum voltage
        cell_div_idxes = list(range(dim_x + dim_y, dim_x + dim_y + dim_z))
        tensor = normalize_columns_0_to_1(tensor[start_idx:, :], exclude_cols=cell_div_idxes)
        tensorx = normalize_columns_0_to_1(tensorx[start_idx:, :], exclude_cols=cell_div_idxes)
    
    if SKIP_CHARGE_READY:
        charge_idx = 224 if BATTERY_TYPE == "QAS" else 174
        start_idx = last_index_of(tensor, feature_idx=charge_idx, value=-500, operator='<') # -500 means minimum current during charge ready
        tensor = tensor[start_idx:, :]
        tensorx = tensorx[start_idx:, :]
    
    if FIRST_LOAD:
        FIRST_LOAD = False
        combined_tensor = tensor
        combined_tensorx = tensorx
    else:
        combined_tensor = torch.cat((combined_tensor, tensor), dim=0)
        combined_tensorx = torch.cat((combined_tensorx, tensorx), dim=0)

buf = io.StringIO()
with redirect_stdout(buf):
    print_sim_config(
    title="Train Run",
    config=args,
    extra={
        "device": str(device),
        "Vehicle IDs used for training": vehicle_idxes,
        "Amount of data used for training": combined_tensor.shape[0],
    },
    # exclude=["password", "token"],  # 민감정보 방지용
    )

text = buf.getvalue()
print(text, end="")  # 콘솔 출력
os.makedirs(model_path, exist_ok=True)
with open(f"{model_path}/sim_config.txt", "w", encoding="utf-8") as f:
    f.write(text)    # 파일 저장
#----------------------------------------Training for MC-AE--------------------------
x_recovered = combined_tensor[:, :dim_x] # 0~1
y_recovered = combined_tensor[:, dim_x:dim_x + dim_y] # 2~111
z_recovered = combined_tensor[:, dim_x + dim_y: dim_x + dim_y + dim_z] # 112~221
q_recovered = combined_tensor[:, dim_x + dim_y + dim_z:] # 222~

x_recovered2 = combined_tensorx[:, :dim_x2]
y_recovered2 = combined_tensorx[:, dim_x2:dim_x2 + dim_y2]
z_recovered2 = combined_tensorx[:, dim_x2 + dim_y2: dim_x2 + dim_y2 + dim_z2]
q_recovered2 = combined_tensorx[:, dim_x2 + dim_y2 + dim_z2:]

AE_EPOCH = args.ae_epochs
AE_LR = args.ae_lr
AE_BATCHSIZE = args.ae_batchsize
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
train_loader_u = DataLoader(Dataset(x_recovered, y_recovered, z_recovered, q_recovered), batch_size=AE_BATCHSIZE,
                      shuffle=False)

# Instantiate the networks
net = CombinedAE(input_size=2, encode2_input_size=3, output_size=dim_y, activation_fn=custom_activation, use_dx_in_forward=True).to(device)
netx = CombinedAE(input_size=2, encode2_input_size=4, output_size=dim_y, activation_fn=torch.sigmoid, use_dx_in_forward=True).to(device)

optimizer = torch.optim.Adam(net.parameters(), lr=AE_LR)
loss_f = nn.MSELoss()
for epoch in range(AE_EPOCH):
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
    if epoch % 50 == 0:
        print('Epoch: {:2d} | Average Loss: {:.4f}'.format(epoch, avg_loss))
    
save_net_state(model=net, models_dir=f"{model_path}/", filename='net.pth')
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

train_loader_soc = DataLoader(Dataset(x_recovered2, y_recovered2, z_recovered2, q_recovered2), batch_size=AE_BATCHSIZE, shuffle=False)
optimizer = torch.optim.Adam(netx.parameters(), lr=AE_LR)
loss_f = nn.MSELoss()
avg_loss_list_x = []
for epoch in range(AE_EPOCH):
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
    if epoch % 50 == 0:
        print('Epoch: {:2d} | Average Loss: {:.4f}'.format(epoch, avg_loss))
save_net_state(model=netx, models_dir=f"{model_path}/", filename='netx.pth')

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

elapsed = time.perf_counter() - start
h, rem = divmod(elapsed, 3600)
m, s = divmod(rem, 60)
print(f"{int(h)}시간 {int(m)}분 {s:.3f}초")

save_pca_results(f"{model_path}/", results)

# (v_I, v, v_ratio, p_k, data_mean, data_std, T_95_limit, T_99_limit, SPE_95_limit, SPE_99_limit, P, k, P_t, X, data_nor) = results

# t2_array = T2_array(df_data, data_mean, data_std, p_k, v_I)
# plot_array(t2_array, title="Hotelling T² over time",
#             figsize=(12, 4), save_path=None, show=True)

# spe_array = SPE_array(df_data, data_mean, data_std, p_k)
# plot_array(spe_array, title="Squared Prediction Error over time",       
#             figsize=(12, 4), save_path=None, show=True)

