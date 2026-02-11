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

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
print(torch.cuda.device_count())

warnings.filterwarnings("ignore")

parser = argparse.ArgumentParser(description="Run MC-AE training with CLI options")
parser.add_argument("--battery-type", choices=["QAS", "DTI"], default="DTI")
parser.add_argument(
    "--vehicle-start",
    type=int,
    default=0,
    help="Filtered normal vehicle ID 시작 인덱스",
)
parser.add_argument(
    "--vehicle-end", type=int, default=-1, help="Filtered normal vehicle ID 끝 인덱스"
)
parser.add_argument("--lstm-training", action="store_true")
parser.add_argument("--lstm-load", action="store_true")
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
    "--source-data-dir",
    type=str,
    default=(
        "./data"
        if os.environ.get("SOURCE_DIR") is None
        else os.environ.get("SOURCE_DIR")
    ),
)
parser.add_argument("--vin1-start", type=int, default=0)
parser.add_argument("--vin2-start", type=int, default=0)
parser.add_argument("--vin3-start", type=int, default=0)
parser.add_argument("--ae-u-scale", type=float, default=1.8)
parser.add_argument("--ae-u-shift", type=float, default=2.5)
parser.add_argument("--ae-u-epochs", type=int, default=10)  # default 300
parser.add_argument(
    "--ae-u-lr", type=float, default=5e-4
)  # based on original manuscript
parser.add_argument("--ae-u-batchsize", type=int, default=100)
parser.add_argument("--ae-x-scale", type=float, default=1.0)
parser.add_argument("--ae-x-shift", type=float, default=0)
parser.add_argument("--ae-x-epochs", type=int, default=300)  # default 300
parser.add_argument(
    "--ae-x-lr", type=float, default=5e-4
)  # based on original manuscript
parser.add_argument("--ae-x-batchsize", type=int, default=100)
parser.add_argument("--lstm-epochs", type=int, default=300)
parser.add_argument(
    "--lstm-lr", type=float, default=1e-4
)  # based on original manuscript
parser.add_argument(
    "--lstm-batchsize", type=int, default=100
)  # based on original manuscript
parser.add_argument("--normalize-dx", action="store_true")
parser.add_argument("--normalize-val", type=int, default=4)
parser.add_argument(
    "--learning-case",
    type=int,
    default=1,
    help="1: Skip and no nomalization, 2: Skip but doing normalization, 3: No skip but doing normalization, 4: No skip and no normalization.",
)
parser.add_argument("--no-use-dx", action="store_true")
parser.add_argument("--no-abs-err", action="store_true")
parser.add_argument("--add-one-output-layer", action="store_true")
args = parser.parse_args()


LSTM_TRAINING = args.lstm_training
LSTM_LOAD = args.lstm_load if args.lstm_training else args.lstm_load
BATTERY_TYPE = args.battery_type  # 'DTI' or 'QAS'
PREPROCESSING, SKIP_CHARGE_READY = get_preprocessing_and_skip_charge_ready(
    args.learning_case
)
use_dx_in_forward = not args.no_use_dx
use_abs_err = not args.no_abs_err
add_one_output_layer = args.add_one_output_layer
# use_dx_in_forward = False
dim_dict = get_input_dimensions(BATTERY_TYPE)

# 모델 저장 경로 설정

data_path = make_model_path_based_timestamp(base=args.models_dir)
model_path = data_path + "/artifact"

start = time.perf_counter()

train_list = (
    np.load(f"./{BATTERY_TYPE}_filtered_vehicle_ids_train.npy")
    .astype(np.int64)
    .tolist()
)
validate_list = (
    np.load(f"./{BATTERY_TYPE}_filtered_vehicle_ids_validate.npy")
    .astype(np.int64)
    .tolist()
)
vehicle_idxes = []
FIRST_LOAD = True

# ----------------------------------------Hyper Parameter Setting---------------------
TIME_STEP = 1  # rnn time step
INPUT_SIZE = 7  # rnn input size (전압은 total voltage, 온도는 평균 온도, 전류는 동일하니 1개로 표현, SOC는 board end SOC)
LR = args.lstm_lr  # learning rate
lr_decay_freq = 25
EPOCH = args.lstm_epochs
BATCH_SIZE = args.lstm_batchsize

train_list = (
    train_list[args.vehicle_start : args.vehicle_end + 1]
    if args.vehicle_end != -1
    else train_list[args.vehicle_start :]
)
# train_list = [0]
# validate_list = [0]
count = 0
for i in train_list:
    count += 1
    if count % 10 == 0:
        print(f"Processing vehicle {count}/{len(train_list)}")
    VEHICLE_ID = f"{i}"
    vehicle_idxes.append(VEHICLE_ID)
    # ----------------------------------------Data loading for LSTM (customized) ------------------------------
    # print(os.getcwd())
    if LSTM_LOAD:
        test_X = safe_load(
            f"{args.source_data_dir}/{BATTERY_TYPE}/{VEHICLE_ID}/vin_1.pkl"
        )

    # ----------------------------------------Training for overall state prediction--------
    if LSTM_TRAINING:
        train_X, train_y = prepare_training_data(test_X, INPUT_SIZE, TIME_STEP, device)

        train_dataset = MyDataset(train_X, train_y)
        train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)

        lstm = LSTM().to(device)
        optimizer = torch.optim.Adam(
            lstm.parameters(), lr=LR
        )  # optimize all cnn parameters
        loss_func = nn.MSELoss()

        # LSTM Training
        hidden_state = None
        loss_train_100 = []
        for epoch in range(EPOCH):
            for step, (b_x, b_y) in enumerate(train_loader):  # gives batch data
                lstm = (
                    lstm.double()
                )  # Casts all floating point parameters and buffers to ``double`` datatype.
                output = lstm(b_x)  # rnn output
                loss = loss_func(b_y, output)  # cross entropy loss
                optimizer.zero_grad()  # clear gradients for this training step
                loss.backward()  # backpropagation, compute gradients
                optimizer.step()  # apply gradients
                if step % 50 == 0:
                    loss_train_100.append(loss.cpu().detach().numpy())

    # ----------------------------------------Data loading for MC-AE (customized) ------------------------------
    tensor = safe_load(
        f"{args.source_data_dir}/{BATTERY_TYPE}/{VEHICLE_ID}/vin_2.pkl", verbose=False
    )
    tensorx = safe_load(
        f"{args.source_data_dir}/{BATTERY_TYPE}/{VEHICLE_ID}/vin_3.pkl", verbose=False
    )

    # ----------------------------------------Preprocessing ------------------------------#
    tensor, tensorx = preprocess_combined_tensor(
        tensor,
        tensorx,
        dim_dict,
        BATTERY_TYPE,
        PREPROCESSING,
        SKIP_CHARGE_READY,
        normalize_dx=args.normalize_dx,
        normalize_val=args.normalize_val,
    )

    if FIRST_LOAD:
        FIRST_LOAD = False
        combined_tensor = tensor
        combined_tensorx = tensorx
    else:
        combined_tensor = torch.cat((combined_tensor, tensor), dim=0)
        combined_tensorx = torch.cat((combined_tensorx, tensorx), dim=0)


# ----------------------------------------Training for MC-AE--------------------------
x_recovered = combined_tensor[:, : dim_dict["x"]]  # 0~1
y_recovered = combined_tensor[:, dim_dict["x"] : dim_dict["x"] + dim_dict["y"]]  # 2~111
z_recovered = combined_tensor[
    :, dim_dict["x"] + dim_dict["y"] : dim_dict["x"] + dim_dict["y"] + dim_dict["z"]
]  # 112~221
q_recovered = combined_tensor[
    :, dim_dict["x"] + dim_dict["y"] + dim_dict["z"] :
]  # 222~

x_recovered2 = combined_tensorx[:, : dim_dict["x2"]]
y_recovered2 = combined_tensorx[:, dim_dict["x2"] : dim_dict["x2"] + dim_dict["y2"]]
z_recovered2 = combined_tensorx[
    :,
    dim_dict["x2"] + dim_dict["y2"] : dim_dict["x2"] + dim_dict["y2"] + dim_dict["z2"],
]
q_recovered2 = combined_tensorx[:, dim_dict["x2"] + dim_dict["y2"] + dim_dict["z2"] :]


# ----------------------------------------Validation Data Loading ------------------------------
FIRST_LOAD = True
count = 0
for i in validate_list:
    count += 1
    if count % 10 == 0:
        print(f"Processing validation vehicle {count}/{len(validate_list)}")
    v_tensor = safe_load(
        f"{args.source_data_dir}/{BATTERY_TYPE}/{i}/vin_2.pkl", verbose=False
    )
    v_tensorx = safe_load(
        f"{args.source_data_dir}/{BATTERY_TYPE}/{i}/vin_3.pkl", verbose=False
    )
    v_tensor, v_tensorx = preprocess_combined_tensor(
        v_tensor,
        v_tensorx,
        dim_dict,
        BATTERY_TYPE,
        PREPROCESSING,
        SKIP_CHARGE_READY,
        normalize_dx=args.normalize_dx,
        normalize_val=args.normalize_val,
    )
    if FIRST_LOAD:
        FIRST_LOAD = False
        v_combined_tensor = v_tensor
        v_combined_tensorx = v_tensorx
    else:
        v_combined_tensor = torch.cat((v_combined_tensor, v_tensor), dim=0)
        v_combined_tensorx = torch.cat((v_combined_tensorx, v_tensorx), dim=0)

v_x_recovered = v_combined_tensor[:, : dim_dict["x"]]  # 0~1
v_y_recovered = v_combined_tensor[
    :, dim_dict["x"] : dim_dict["x"] + dim_dict["y"]
]  # 2~111
v_z_recovered = v_combined_tensor[
    :, dim_dict["x"] + dim_dict["y"] : dim_dict["x"] + dim_dict["y"] + dim_dict["z"]
]  # 112~221
v_q_recovered = v_combined_tensor[
    :, dim_dict["x"] + dim_dict["y"] + dim_dict["z"] :
]  # 222~

v_x_recovered2 = v_combined_tensorx[:, : dim_dict["x2"]]
v_y_recovered2 = v_combined_tensorx[:, dim_dict["x2"] : dim_dict["x2"] + dim_dict["y2"]]
v_z_recovered2 = v_combined_tensorx[
    :,
    dim_dict["x2"] + dim_dict["y2"] : dim_dict["x2"] + dim_dict["y2"] + dim_dict["z2"],
]
v_q_recovered2 = v_combined_tensorx[
    :, dim_dict["x2"] + dim_dict["y2"] + dim_dict["z2"] :
]

buf = io.StringIO()
with redirect_stdout(buf):
    print_sim_config(
        title="Train Run",
        config=args,
        extra={
            "device": str(device),
            "Vehicle IDs used for training": vehicle_idxes,
            "Amount of data used for training": combined_tensor.shape[0],
            "PREPROCESSING": PREPROCESSING,
            "SKIP_CHARGE_READY": SKIP_CHARGE_READY,
        },
        # exclude=["password", "token"],  # 민감정보 방지용
    )

text = buf.getvalue()
print(text, end="")  # 콘솔 출력
os.makedirs(data_path, exist_ok=True)
with open(f"{data_path}/sim_config.txt", "w", encoding="utf-8") as f:
    f.write(text)  # 파일 저장

AE_U_EPOCH = args.ae_u_epochs
AE_U_LR = args.ae_u_lr
AE_U_BATCHSIZE = args.ae_u_batchsize


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


train_loader_u = DataLoader(
    Dataset(x_recovered, y_recovered, z_recovered, q_recovered),
    batch_size=AE_U_BATCHSIZE,
    shuffle=False,
)
validate_loader_u = DataLoader(
    Dataset(v_x_recovered, v_y_recovered, v_z_recovered, v_q_recovered),
    batch_size=AE_U_BATCHSIZE,
    shuffle=False,
)

# Instantiate the networks
net = CombinedAE(
    input_size=dim_dict["x"],
    encode2_input_size=dim_dict["q"],
    output_size=dim_dict["y"],
    activation_fn=(
        CustomSigmoidFunc(scale=args.ae_u_scale, shift=args.ae_u_shift)
        if PREPROCESSING == False
        else torch.sigmoid
    ),
    use_dx_in_forward=use_dx_in_forward,
    add_one_output_layer=add_one_output_layer,
).to(device)
netx = CombinedAE(
    input_size=dim_dict["x2"],
    encode2_input_size=dim_dict["q2"],
    output_size=dim_dict["y2"],
    activation_fn=CustomSigmoidFunc(scale=args.ae_x_scale, shift=args.ae_x_shift),
    use_dx_in_forward=use_dx_in_forward,
    add_one_output_layer=add_one_output_layer,
).to(device)

## ================= Training MC-AE for voltage reconstruction ================= ##
optimizer = torch.optim.Adam(net.parameters(), lr=AE_U_LR)
loss_f = nn.MSELoss()
avg_loss_list_u = []
val_loss_points_u = []  # list of (epoch, avg_val_loss)

# double casting은 반복 호출할 필요가 없어 루프 밖에서 1회만 수행
net = net.double()
for epoch in range(AE_U_EPOCH):
    total_loss = 0
    num_batches = 0
    for iteration, (x, y, z, q) in enumerate(train_loader_u):
        x = x.to(device)
        y = y.to(device)
        z = z.to(device)
        q = q.to(device)
        net.train()
        recon_im, recon_p = net(x, z, q)
        loss_u = loss_f(y, recon_im)
        total_loss += loss_u.item()
        num_batches += 1
        optimizer.zero_grad()
        loss_u.backward()
        optimizer.step()
    avg_loss = total_loss / num_batches
    avg_loss_list_u.append(avg_loss)
    if epoch % 10 == 0:
        # Validation (fast: no grad)
        net.eval()
        v_total = 0.0
        v_batches = 0
        with torch.inference_mode():
            for vx, vy, vz, vq in validate_loader_u:
                vx = vx.to(device)
                vy = vy.to(device)
                vz = vz.to(device)
                vq = vq.to(device)
                v_recon_im, _ = net(vx, vz, vq)
                v_loss = loss_f(vy, v_recon_im)
                v_total += float(v_loss.item())
                v_batches += 1
        avg_val_loss = v_total / max(v_batches, 1)
        val_loss_points_u.append((int(epoch), float(avg_val_loss)))

        print(
            "Epoch: {:2d} | Train Loss: {:.4f} | Val Loss: {:.4f}".format(
                epoch, avg_loss, avg_val_loss
            )
        )
        # checkpoint (indexed)
        save_net_state(
            model=net, models_dir=f"{model_path}/", filename=f"net_e{epoch}.pth"
        )

# Save loss log (fast: dump once per model)
np.savetxt(
    os.path.join(f"{data_path}/", "loss_net.csv"),
    np.column_stack(
        (
            np.arange(len(avg_loss_list_u), dtype=np.int64),
            np.asarray(avg_loss_list_u, dtype=np.float64),
        )
    ),
    delimiter=",",
    fmt=["%.0f", "%.4f"],
    header="epoch,avg_loss",
    comments="",
)

# validation loss points (epoch,avg_loss)
if len(val_loss_points_u) > 0:
    np.savetxt(
        os.path.join(f"{data_path}/", "loss_net_val.csv"),
        np.asarray(val_loss_points_u, dtype=np.float64),
        delimiter=",",
        fmt=["%.0f", "%.4f"],
        header="epoch,avg_loss",
        comments="",
    )

save_net_state(model=net, models_dir=f"{model_path}/", filename="net.pth")

train_loader2 = DataLoader(
    Dataset(x_recovered, y_recovered, z_recovered, q_recovered),
    batch_size=len(x_recovered),
    shuffle=False,
)
for iteration, (x, y, z, q) in enumerate(train_loader2):
    x = x.to(device)
    y = y.to(device)
    z = z.to(device)
    q = q.to(device)
    net = net.double()
    recon_imtest, recon = net(x, z, q)
AA = recon_imtest.cpu().detach().numpy()
yTrainU = y_recovered.cpu().detach().numpy()
ERRORU = np.abs(AA - yTrainU) if use_abs_err else AA - yTrainU

## ================= Training MC-AE for SOC reconstruction ================= ##
AE_X_EPOCH = args.ae_x_epochs
AE_X_LR = args.ae_x_lr
AE_X_BATCHSIZE = args.ae_x_batchsize
train_loader_soc = DataLoader(
    Dataset(x_recovered2, y_recovered2, z_recovered2, q_recovered2),
    batch_size=AE_X_BATCHSIZE,
    shuffle=False,
)
validate_loader_soc = DataLoader(
    Dataset(v_x_recovered2, v_y_recovered2, v_z_recovered2, v_q_recovered2),
    batch_size=AE_X_BATCHSIZE,
    shuffle=False,
)
optimizer = torch.optim.Adam(netx.parameters(), lr=AE_X_LR)
loss_f = nn.MSELoss()
avg_loss_list_x = []
val_loss_points_x = []  # list of (epoch, avg_val_loss)

netx = netx.double()
for epoch in range(AE_X_EPOCH):
    total_loss = 0
    num_batches = 0
    for iteration, (x, y, z, q) in enumerate(train_loader_soc):
        x = x.to(device)
        y = y.to(device)
        z = z.to(device)
        q = q.to(device)
        netx.train()
        recon_im, z = netx(x, z, q)
        loss_x = loss_f(y, recon_im)
        total_loss += loss_x.item()
        num_batches += 1
        optimizer.zero_grad()
        loss_x.backward()
        optimizer.step()
    avg_loss = total_loss / num_batches
    avg_loss_list_x.append(avg_loss)
    if epoch % 10 == 0:
        netx.eval()
        v_total = 0.0
        v_batches = 0
        with torch.inference_mode():
            for vx, vy, vz, vq in validate_loader_soc:
                vx = vx.to(device)
                vy = vy.to(device)
                vz = vz.to(device)
                vq = vq.to(device)
                v_recon_im, _ = netx(vx, vz, vq)
                v_loss = loss_f(vy, v_recon_im)
                v_total += float(v_loss.item())
                v_batches += 1
        avg_val_loss = v_total / max(v_batches, 1)
        val_loss_points_x.append((int(epoch), float(avg_val_loss)))

        print(
            "Epoch: {:2d} | Train Loss: {:.4f} | Val Loss: {:.4f}".format(
                epoch, avg_loss, avg_val_loss
            )
        )
        # checkpoint (indexed)
        save_net_state(
            model=netx, models_dir=f"{model_path}/", filename=f"netx_e{epoch}.pth"
        )

# Save loss log (fast: dump once per model)
np.savetxt(
    os.path.join(f"{data_path}/", "loss_netx.csv"),
    np.column_stack(
        (
            np.arange(len(avg_loss_list_x), dtype=np.int64),
            np.asarray(avg_loss_list_x, dtype=np.float64),
        )
    ),
    delimiter=",",
    fmt=["%.0f", "%.4f"],
    header="epoch,avg_loss",
    comments="",
)

if len(val_loss_points_x) > 0:
    np.savetxt(
        os.path.join(f"{data_path}/", "loss_netx_val.csv"),
        np.asarray(val_loss_points_x, dtype=np.float64),
        delimiter=",",
        fmt=["%.0f", "%.4f"],
        header="epoch,avg_loss",
        comments="",
    )
save_net_state(model=netx, models_dir=f"{model_path}/", filename="netx.pth")

train_loaderx2 = DataLoader(
    Dataset(x_recovered2, y_recovered2, z_recovered2, q_recovered2),
    batch_size=len(x_recovered2),
    shuffle=False,
)
for iteration, (x, y, z, q) in enumerate(train_loaderx2):
    x = x.to(device)
    y = y.to(device)
    z = z.to(device)
    q = q.to(device)
    netx = netx.double()
    recon_imtestx, z = netx(x, z, q)

BB = recon_imtestx.cpu().detach().numpy()
yTrainX = y_recovered2.cpu().detach().numpy()
ERRORX = np.abs(BB - yTrainX) if use_abs_err else BB - yTrainX

df_data, _ = DiagnosisFeature(ERRORU, ERRORX)

results = Custom_PCA(df_data, 0.99, 0.99)

elapsed = time.perf_counter() - start
h, rem = divmod(elapsed, 3600)
m, s = divmod(rem, 60)
print(f"{int(h)}시간 {int(m)}분 {s:.3f}초")

save_pca_results(f"{data_path}/", results)

# (v_I, v, v_ratio, p_k, data_mean, data_std, T_95_limit, T_99_limit, SPE_95_limit, SPE_99_limit, P, k, P_t, X, data_nor) = results

# t2_array = T2_array(df_data, data_mean, data_std, p_k, v_I)
# plot_array(t2_array, title="Hotelling T² over time",
#             figsize=(12, 4), save_path=None, show=True)

# spe_array = SPE_array(df_data, data_mean, data_std, p_k)
# plot_array(spe_array, title="Squared Prediction Error over time",
#             figsize=(12, 4), save_path=None, show=True)
