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
import random


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# set_seed(42) # Moved to after args parsing

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
print(torch.cuda.device_count())

warnings.filterwarnings("ignore")

parser = argparse.ArgumentParser(description="Run MC-AE training with CLI options")
parser.add_argument("--battery-type", choices=["QAS", "DTI"], default="QAS")
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
parser.add_argument("--ae-u-save-per", type=int, default=10)  # default 300
parser.add_argument(
    "--ae-u-lr", type=float, default=5e-4
)  # based on original manuscript
parser.add_argument("--ae-u-shuffle", action="store_true")
parser.add_argument("--ae-u-batchsize", type=int, default=100)
parser.add_argument("--ae-x-scale", type=float, default=1.0)
parser.add_argument("--ae-x-shift", type=float, default=0)
parser.add_argument("--ae-x-epochs", type=int, default=300)  # default 300
parser.add_argument("--ae-x-save-per", type=int, default=10)  # default 300
parser.add_argument(
    "--ae-x-lr", type=float, default=5e-4
)  # based on original manuscript
parser.add_argument("--ae-x-shuffle", action="store_true")
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
parser.add_argument("--dataset-name", type=str, default=None, help="Name of the dataset folder (e.g. 'train_0.5_seed42') to load from ./datasets/")
parser.add_argument("--no-use-dx", action="store_true")
parser.add_argument("--no-abs-err", action="store_true")
parser.add_argument("--add-one-output-layer", action="store_true")
parser.add_argument("--no-save", action="store_true")
parser.add_argument(
    "--seed", type=int, default=42, help="Random seed for reproducibility"
)
args = parser.parse_args()

set_seed(args.seed)

LSTM_TRAINING = args.lstm_training
LSTM_LOAD = args.lstm_load if args.lstm_training else args.lstm_load
BATTERY_TYPE = args.battery_type  # 'DTI' or 'QAS'
PREPROCESSING, SKIP_CHARGE_READY = get_preprocessing_and_skip_charge_ready(
    args.learning_case
)
use_dx_in_forward = not args.no_use_dx
use_abs_err = not args.no_abs_err
add_one_output_layer = args.add_one_output_layer

# 디버그 모드인지 확인 (sys.gettrace() 또는 debugpy 모듈 로드 여부)
is_debug = sys.gettrace() is not None or "debugpy" in sys.modules

if is_debug:
    use_dx_in_forward = False
    print("현재 디버그 모드로 실행 중입니다.")

# use_dx_in_forward = False
dim_dict = get_input_dimensions(BATTERY_TYPE)

# 모델 저장 경로 설정

data_path = make_model_path_based_timestamp(base=args.models_dir)
model_path = data_path + "/artifact"

start = time.perf_counter()

if args.dataset_name:
    dataset_base_path = f"./datasets/{args.dataset_name}"
    print(f"Dataset path: {dataset_base_path}")
    train_file_path = f"{dataset_base_path}/{BATTERY_TYPE}_filtered_vehicle_ids_train.npy"
    val_file_path = f"{dataset_base_path}/{BATTERY_TYPE}_filtered_vehicle_ids_validate.npy"
else:
    print("Dataset path: . (Legacy)")
    train_file_path = f"./{BATTERY_TYPE}_filtered_vehicle_ids_train.npy"
    val_file_path = f"./{BATTERY_TYPE}_filtered_vehicle_ids_validate.npy"

train_list = (
    np.load(train_file_path)
    .astype(np.int64)
    .tolist()
)
validate_list = (
    np.load(val_file_path)
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

vehicle_tensors = []
vehicle_tensorsx = []

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
    tensor, tensorx = preprocess_loaded_tensor(
        tensor,
        tensorx,
        dim_dict,
        BATTERY_TYPE,
        PREPROCESSING,
        SKIP_CHARGE_READY,
        normalize_dx=args.normalize_dx,
        normalize_val=args.normalize_val,
    )

    vehicle_tensors.append(tensor)
    vehicle_tensorsx.append(tensorx)
    if is_debug:
        break

# combined_tensor를 생성하지 않고 리스트만 유지하여 메모리 피크 방지
# combined_tensor = torch.cat(vehicle_tensors, dim=0)
# combined_tensorx = torch.cat(vehicle_tensorsx, dim=0)
# del vehicle_tensors, vehicle_tensorsx # 기존 코드 주석 처리


# ----------------------------------------Training for MC-AE--------------------------
# 리스트 상태로 Dataset 생성 (메모리 효율화)
class VirtualConcatDataset(Dataset):
    def __init__(self, tensor_list_u, tensor_list_x, dim_dict):
        self.tensor_list_u = tensor_list_u
        self.tensor_list_x = tensor_list_x
        self.dim_dict = dim_dict

        # 각 텐서의 길이와 누적 인덱스 계산
        self.lengths = [t.shape[0] for t in tensor_list_u]
        self.cumulative_lengths = np.cumsum([0] + self.lengths)
        self.total_length = self.cumulative_lengths[-1]

    def __len__(self):
        return self.total_length

    def __getitem__(self, idx):
        # 이진 탐색으로 어떤 텐서에 속하는지 찾기
        bin_idx = np.searchsorted(self.cumulative_lengths, idx, side="right") - 1
        local_idx = idx - self.cumulative_lengths[bin_idx]

        # 해당 텐서 가져오기
        tensor_u = self.tensor_list_u[bin_idx][local_idx]
        tensor_x = self.tensor_list_x[bin_idx][local_idx]

        # 슬라이싱 (여기서 필요한 부분만 추출)
        # U-Model Inputs
        x = tensor_u[: self.dim_dict["x"]]
        y = tensor_u[self.dim_dict["x"] : self.dim_dict["x"] + self.dim_dict["y"]]
        z = tensor_u[
            self.dim_dict["x"]
            + self.dim_dict["y"] : self.dim_dict["x"]
            + self.dim_dict["y"]
            + self.dim_dict["z"]
        ]
        q = tensor_u[self.dim_dict["x"] + self.dim_dict["y"] + self.dim_dict["z"] :]

        # X-Model Inputs (필요 시 반환하도록 수정 가능, 현재 구조상 4개만 반환하므로 U모델용만 반환)
        # X 모델용 데이터가 필요할 땐 별도 Dataset 또는 item 반환 구조 변경 필요
        # 현재 코드 구조상 train_loader_u는 x,y,z,q만 씀.

        return (
            x.to(torch.double),
            y.to(torch.double),
            z.to(torch.double),
            q.to(torch.double),
        )


class VirtualConcatDatasetX(Dataset):
    def __init__(self, tensor_list_x, dim_dict):
        self.tensor_list_x = tensor_list_x
        self.dim_dict = dim_dict
        self.lengths = [t.shape[0] for t in tensor_list_x]
        self.cumulative_lengths = np.cumsum([0] + self.lengths)
        self.total_length = self.cumulative_lengths[-1]

    def __len__(self):
        return self.total_length

    def __getitem__(self, idx):
        bin_idx = np.searchsorted(self.cumulative_lengths, idx, side="right") - 1
        local_idx = idx - self.cumulative_lengths[bin_idx]
        tensor_x = self.tensor_list_x[bin_idx][local_idx]

        x2 = tensor_x[: self.dim_dict["x2"]]
        y2 = tensor_x[self.dim_dict["x2"] : self.dim_dict["x2"] + self.dim_dict["y2"]]
        z2 = tensor_x[
            self.dim_dict["x2"]
            + self.dim_dict["y2"] : self.dim_dict["x2"]
            + self.dim_dict["y2"]
            + self.dim_dict["z2"]
        ]
        q2 = tensor_x[self.dim_dict["x2"] + self.dim_dict["y2"] + self.dim_dict["z2"] :]

        return (
            x2.to(torch.double),
            y2.to(torch.double),
            z2.to(torch.double),
            q2.to(torch.double),
        )


# 검증 데이터용
class VirtualConcatDatasetVal(Dataset):
    def __init__(self, tensor_list_u, tensor_list_x, dim_dict):
        self.tensor_list_u = tensor_list_u
        self.tensor_list_x = tensor_list_x
        self.dim_dict = dim_dict
        self.lengths = [t.shape[0] for t in tensor_list_u]
        self.cumulative_lengths = np.cumsum([0] + self.lengths)
        self.total_length = self.cumulative_lengths[-1]

    def __len__(self):
        return self.total_length

    def __getitem__(self, idx):
        bin_idx = np.searchsorted(self.cumulative_lengths, idx, side="right") - 1
        local_idx = idx - self.cumulative_lengths[bin_idx]

        t_u = self.tensor_list_u[bin_idx][local_idx]
        t_x = self.tensor_list_x[bin_idx][local_idx]

        # U Inputs
        vx = t_u[: self.dim_dict["x"]]
        vy = t_u[self.dim_dict["x"] : self.dim_dict["x"] + self.dim_dict["y"]]
        vz = t_u[
            self.dim_dict["x"]
            + self.dim_dict["y"] : self.dim_dict["x"]
            + self.dim_dict["y"]
            + self.dim_dict["z"]
        ]
        vq = t_u[self.dim_dict["x"] + self.dim_dict["y"] + self.dim_dict["z"] :]

        # X Inputs
        vx2 = t_x[: self.dim_dict["x2"]]
        vy2 = t_x[self.dim_dict["x2"] : self.dim_dict["x2"] + self.dim_dict["y2"]]
        vz2 = t_x[
            self.dim_dict["x2"]
            + self.dim_dict["y2"] : self.dim_dict["x2"]
            + self.dim_dict["y2"]
            + self.dim_dict["z2"]
        ]
        vq2 = t_x[self.dim_dict["x2"] + self.dim_dict["y2"] + self.dim_dict["z2"] :]

        return (
            vx.to(torch.double),
            vy.to(torch.double),
            vz.to(torch.double),
            vq.to(torch.double),
            vx2.to(torch.double),
            vy2.to(torch.double),
            vz2.to(torch.double),
            vq2.to(torch.double),
        )


# Total training samples calculation
total_train_samples = sum(len(t) for t in vehicle_tensors)


# ----------------------------------------Validation Data Loading ------------------------------
FIRST_LOAD = True
count = 0
v_vehicle_tensors = []
v_vehicle_tensorsx = []

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
    v_tensor, v_tensorx = preprocess_loaded_tensor(
        v_tensor,
        v_tensorx,
        dim_dict,
        BATTERY_TYPE,
        PREPROCESSING,
        SKIP_CHARGE_READY,
        normalize_dx=args.normalize_dx,
        normalize_val=args.normalize_val,
    )

    v_vehicle_tensors.append(v_tensor)
    v_vehicle_tensorsx.append(v_tensorx)
    if is_debug:
        break

# combined_tensor 생성 제거
# if len(v_vehicle_tensors) > 0:
#     v_combined_tensor = torch.cat(v_vehicle_tensors, dim=0)
#     v_combined_tensorx = torch.cat(v_vehicle_tensorsx, dim=0)
# else:
#     v_combined_tensor = torch.empty(0)
#     v_combined_tensorx = torch.empty(0)

# del v_vehicle_tensors, v_vehicle_tensorsx

# v_x_recovered = v_combined_tensor[:, : dim_dict["x"]]  # 0~1
# ... (슬라이싱 제거)

total_val_samples = sum(len(t) for t in v_vehicle_tensors)

if not is_debug:
    buf = io.StringIO()
    with redirect_stdout(buf):
        print_sim_config(
            title="Train Run",
            config=args,
            extra={
                "device": str(device),
                "Vehicle IDs used for training": vehicle_idxes,
                "Amount of data used for training": total_train_samples,
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


# ---------------------------------------- Dataset Class for Virtual Concat ----------------------------------------
class VirtualDatasetU(Dataset):
    def __init__(self, tensor_list, dim_dict):
        self.tensor_list = tensor_list
        self.dim_list = [t.shape[0] for t in tensor_list]
        self.cumulative_dim = np.cumsum([0] + self.dim_list)
        self.total_len = self.cumulative_dim[-1]

        # Dimensions
        self.dx = dim_dict["x"]
        self.dy = dim_dict["y"]
        self.dz = dim_dict["z"]
        self.dq = dim_dict["q"]  # Not used but for consistency

    def __len__(self):
        return self.total_len

    def __getitem__(self, idx):
        # Binary search
        bin_idx = np.searchsorted(self.cumulative_dim, idx, side="right") - 1
        local_idx = idx - self.cumulative_dim[bin_idx]

        # Get raw tensor
        raw = self.tensor_list[bin_idx][local_idx]

        # Slice on the fly
        x = raw[: self.dx]
        y = raw[self.dx : self.dx + self.dy]
        z = raw[self.dx + self.dy : self.dx + self.dy + self.dz]
        q = raw[self.dx + self.dy + self.dz :]

        return (
            x.to(torch.double),
            y.to(torch.double),
            z.to(torch.double),
            q.to(torch.double),
        )


class VirtualDatasetX(Dataset):
    def __init__(self, tensor_list, dim_dict):
        self.tensor_list = tensor_list
        # Precompute cumulative lengths
        self.dim_list = [t.shape[0] for t in tensor_list]
        self.cumulative_dim = np.cumsum([0] + self.dim_list)
        self.total_len = self.cumulative_dim[-1]

        # Dimensions
        self.dx = dim_dict["x2"]
        self.dy = dim_dict["y2"]
        self.dz = dim_dict["z2"]
        # self.dq is rest

    def __len__(self):
        return self.total_len

    def __getitem__(self, idx):
        # Binary search
        bin_idx = np.searchsorted(self.cumulative_dim, idx, side="right") - 1
        local_idx = idx - self.cumulative_dim[bin_idx]

        raw = self.tensor_list[bin_idx][local_idx]

        # Slice on the fly
        x = raw[: self.dx]
        y = raw[self.dx : self.dx + self.dy]
        z = raw[self.dx + self.dy : self.dx + self.dy + self.dz]
        q = raw[self.dx + self.dy + self.dz :]

        return (
            x.to(torch.double),
            y.to(torch.double),
            z.to(torch.double),
            q.to(torch.double),
        )


class VirtualDatasetVal(Dataset):
    def __init__(self, tensor_list_u, tensor_list_x, dim_dict):
        self.tensor_list_u = tensor_list_u
        self.tensor_list_x = tensor_list_x
        self.dim_list = [t.shape[0] for t in tensor_list_u]
        self.cumulative_dim = np.cumsum([0] + self.dim_list)
        self.total_len = self.cumulative_dim[-1]
        self.dim_dict = dim_dict

    def __len__(self):
        return self.total_len

    def __getitem__(self, idx):
        bin_idx = np.searchsorted(self.cumulative_dim, idx, side="right") - 1
        local_idx = idx - self.cumulative_dim[bin_idx]

        raw_u = self.tensor_list_u[bin_idx][local_idx]
        raw_x = self.tensor_list_x[bin_idx][local_idx]

        # U
        vx = raw_u[: self.dim_dict["x"]]
        vy = raw_u[self.dim_dict["x"] : self.dim_dict["x"] + self.dim_dict["y"]]
        vz = raw_u[
            self.dim_dict["x"]
            + self.dim_dict["y"] : self.dim_dict["x"]
            + self.dim_dict["y"]
            + self.dim_dict["z"]
        ]
        vq = raw_u[self.dim_dict["x"] + self.dim_dict["y"] + self.dim_dict["z"] :]

        # X
        vx2 = raw_x[: self.dim_dict["x2"]]
        vy2 = raw_x[self.dim_dict["x2"] : self.dim_dict["x2"] + self.dim_dict["y2"]]
        vz2 = raw_x[
            self.dim_dict["x2"]
            + self.dim_dict["y2"] : self.dim_dict["x2"]
            + self.dim_dict["y2"]
            + self.dim_dict["z2"]
        ]
        vq2 = raw_x[self.dim_dict["x2"] + self.dim_dict["y2"] + self.dim_dict["z2"] :]

        return (
            vx.to(torch.double),
            vy.to(torch.double),
            vz.to(torch.double),
            vq.to(torch.double),
            vx2.to(torch.double),
            vy2.to(torch.double),
            vz2.to(torch.double),
            vq2.to(torch.double),
        )


# Instantiate the networks
net = CombinedAE(
    input_size=dim_dict["x"],
    encode2_input_size=dim_dict["q"],
    output_size=dim_dict["y"],
    activation_fn=(
        CustomSigmoidFunc(scale=args.ae_u_scale, shift=args.ae_u_shift)
        if PREPROCESSING == False
        else CustomSigmoidFunc(scale=1, shift=0)
    ),
    use_dx_in_forward=use_dx_in_forward,
    add_one_output_layer=add_one_output_layer,
).to(device)
netx = CombinedAE(
    input_size=dim_dict["x2"],
    encode2_input_size=dim_dict["q2"],
    output_size=dim_dict["y2"],
    activation_fn=(
        CustomSigmoidFunc(scale=args.ae_x_scale, shift=args.ae_x_shift)
        if PREPROCESSING == False
        else CustomSigmoidFunc(scale=1, shift=0)
    ),
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

train_loader_u = DataLoader(
    VirtualDatasetU(vehicle_tensors, dim_dict),
    batch_size=AE_U_BATCHSIZE,
    shuffle=args.ae_u_shuffle,  # Shuffle is important for training
)
validate_loader_u = DataLoader(
    VirtualDatasetU(v_vehicle_tensors, dim_dict),
    batch_size=AE_U_BATCHSIZE,
    shuffle=False,
)

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
    if epoch % args.ae_u_save_per == 0:
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
        if not args.no_save:
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

if not args.no_save:
    save_net_state(model=net, models_dir=f"{model_path}/", filename="net.pth")

# ERRORU calculation moved to the end to process per-vehicle

## ================= Training MC-AE for SOC reconstruction ================= ##
AE_X_EPOCH = args.ae_x_epochs
AE_X_LR = args.ae_x_lr
AE_X_BATCHSIZE = args.ae_x_batchsize
train_loader_soc = DataLoader(
    VirtualDatasetX(vehicle_tensorsx, dim_dict),
    batch_size=AE_X_BATCHSIZE,
    shuffle=args.ae_x_shuffle,  # Shuffle for training
)
validate_loader_soc = DataLoader(
    VirtualDatasetX(v_vehicle_tensorsx, dim_dict),
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
    if epoch % args.ae_x_save_per == 0:
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
        if not args.no_save:
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

if not args.no_save:
    save_net_state(model=netx, models_dir=f"{model_path}/", filename="netx.pth")

# Optimization: Calculate Diagnosis Features Per Vehicle to handle variance differences
print("Calculating features per vehicle...")
net.eval()
netx.eval()

df_data_list = []

with torch.no_grad():
    for i, (t_u, t_x) in enumerate(zip(vehicle_tensors, vehicle_tensorsx)):
        # Calculate ERROR U for specific vehicle
        # Batching for safety if vehicle data is large (though usually fits in memory)
        batch_size = 4096

        # U-Model
        x_u = t_u[:, : dim_dict["x"]]
        y_u = t_u[:, dim_dict["x"] : dim_dict["x"] + dim_dict["y"]]
        z_u = t_u[
            :,
            dim_dict["x"]
            + dim_dict["y"] : dim_dict["x"]
            + dim_dict["y"]
            + dim_dict["z"],
        ]
        q_u = t_u[:, dim_dict["x"] + dim_dict["y"] + dim_dict["z"] :]

        # X-Model
        x_x = t_x[:, : dim_dict["x2"]]
        y_x = t_x[:, dim_dict["x2"] : dim_dict["x2"] + dim_dict["y2"]]
        z_x = t_x[
            :,
            dim_dict["x2"]
            + dim_dict["y2"] : dim_dict["x2"]
            + dim_dict["y2"]
            + dim_dict["z2"],
        ]
        q_x = t_x[:, dim_dict["x2"] + dim_dict["y2"] + dim_dict["z2"] :]

        num_samples = t_u.shape[0]
        error_u_list = []
        error_x_list = []

        for start_idx in range(0, num_samples, batch_size):
            end_idx = min(start_idx + batch_size, num_samples)

            # Prepare batch for U
            b_x_u = x_u[start_idx:end_idx].to(device).double()
            b_y_u = y_u[start_idx:end_idx].to(device).double()
            b_z_u = z_u[start_idx:end_idx].to(device).double()
            b_q_u = q_u[start_idx:end_idx].to(device).double()

            recon_u, _ = net(b_x_u, b_z_u, b_q_u)

            aa = recon_u.cpu().numpy()
            yy = b_y_u.cpu().numpy()
            e_u = np.abs(aa - yy) if use_abs_err else aa - yy
            error_u_list.append(e_u)

            # Prepare batch for X
            b_x_x = x_x[start_idx:end_idx].to(device).double()
            b_y_x = y_x[start_idx:end_idx].to(device).double()
            b_z_x = z_x[start_idx:end_idx].to(device).double()
            b_q_x = q_x[start_idx:end_idx].to(device).double()

            recon_x, _ = netx(b_x_x, b_z_x, b_q_x)

            bb = recon_x.cpu().numpy()
            yy_x = b_y_x.cpu().numpy()
            e_x = np.abs(bb - yy_x) if use_abs_err else bb - yy_x
            error_x_list.append(e_x)

        # Concatenate errors for this vehicle
        ERRORU_veh = np.concatenate(error_u_list, axis=0)
        ERRORX_veh = np.concatenate(error_x_list, axis=0)

        # Calculate Diagnosis Feature specifically for this vehicle
        # This handles variance per vehicle
        df_veh, _ = DiagnosisFeature(ERRORU_veh, ERRORX_veh)
        df_data_list.append(df_veh)

# Concatenate all features
if df_data_list:
    df_data = pd.concat(df_data_list, axis=0, ignore_index=True)
else:
    print("Warning: No data to process!")
    df_data = pd.DataFrame()  # Empty fallback

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
