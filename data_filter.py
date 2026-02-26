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
import argparse
from enum import IntEnum

parser = argparse.ArgumentParser()
parser.add_argument(
    "--dataset-name", type=str, default="default", help="Name of the dataset folder"
)
parser.add_argument(
    "--train-ratio",
    type=float,
    default=None,
    help="Ratio of training data (0.0 to 1.0)",
)
parser.add_argument(
    "--val-ratio",
    type=float,
    default=None,
    help="Ratio of validation data (0.0 to 1.0)",
)
parser.add_argument("--seed", type=int, default=42, help="Random seed for splitting")
args = parser.parse_args()

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
print(torch.cuda.device_count())

warnings.filterwarnings("ignore")

# 디버그 모드인지 확인 (sys.gettrace() 또는 debugpy 모듈 로드 여부)
is_debug = sys.gettrace() is not None or "debugpy" in sys.modules

if is_debug:
    args.train_ratio = 0.15
    args.val_ratio = 0.5
    args.dataset_name = f"{int(args.train_ratio*100)}"
    print("현재 디버그 모드로 실행 중입니다.")

# Create dataset directory
dataset_dir = f"./datasets/{args.dataset_name}"
os.makedirs(dataset_dir, exist_ok=True)
print(f"Dataset will be saved to: {dataset_dir}")


class MODE(IntEnum):
    PLOT_MODE = 1
    FILTER_MODE = 2
    ANALYSIS_MODE = 3


mode = MODE.FILTER_MODE
LSTM_LOAD = False
FIRST_LOAD = True
BATTERY_TYPE = "QAS"  # 'DTI' or 'QAS'
SKIP_CHARGE_READY = True
PREPROCESSING = False
dim_dict = get_input_dimensions(BATTERY_TYPE)

max_idx = 392 if BATTERY_TYPE == "QAS" else 79
fault_idx = 335 if BATTERY_TYPE == "QAS" else 77

# Set random seed
np.random.seed(args.seed)

# Calculate Dataset Sizes
if args.train_ratio is not None and args.val_ratio is not None:
    # Use ratios
    pass  # Logic below will calculate based on total count
else:
    # Fallback to hardcoded values (if not specified) - though ratios are preferred for flexibility
    n_train_set = 109 if BATTERY_TYPE == "QAS" else 39
    n_validation_set = 50 if BATTERY_TYPE == "QAS" else 19

# "QAS" = [0~392], "DTI" = [0~79]
filtered_vehicle_ids = np.array([])
(vin1_start, vin2_start, vin3_start) = (0, 0, 0)

start_idxes = []
# vehicle_ids = range(0, max_idx + 1) if mode is not MODE.FILTER_MODE else []
for i in range(0, max_idx + 1):
    VEHICLE_ID = f"{i}"
    print(f"============== Processing Vehicle ID: {VEHICLE_ID} ==============")
    PATH = f"./data/data_analysis_at_charge_{vin2_start}/{BATTERY_TYPE}-{VEHICLE_ID}"

    # ----------------------------------------Data loading for LSTM (customized) ------------------------------
    if LSTM_LOAD:
        test_X = safe_load(
            f"./data/{BATTERY_TYPE}/{VEHICLE_ID}/vin_1.pkl", verbose=False
        )

    # ----------------------------------------Data loading for MC-AE (customized) ------------------------------
    tensor = safe_load(f"./data/{BATTERY_TYPE}/{VEHICLE_ID}/vin_2.pkl", verbose=False)
    tensorx = safe_load(f"./data/{BATTERY_TYPE}/{VEHICLE_ID}/vin_3.pkl", verbose=False)

    # volt_all = pd.DataFrame(tensor[:, dim_dict["x"] : dim_dict["x"] + dim_dict["y"]])
    # volt_modepi, volt_di = calculate_volt_modepi(
    #     pd.DataFrame(tensor[:, dim_dict["x"] : dim_dict["x"] + dim_dict["y"]])
    # )
    # Temperature = pd.DataFrame(tensor[:, dim_dict["x"] + dim_dict["y"] + dim_dict["z"]])
    # soc_all = pd.DataFrame(
    #     tensorx[:, dim_dict["x"] + dim_dict["y"] + dim_dict["z"] + 1]
    # )
    # current = pd.DataFrame(tensor[:, dim_dict["x"] + dim_dict["y"] + dim_dict["z"] + 2])

    tensor, tensorx = preprocess_loaded_tensor(
        tensor, tensorx, dim_dict, BATTERY_TYPE, PREPROCESSING, SKIP_CHARGE_READY
    )
    if is_abnormal(tensor):
        print(f"Vehicle ID {i} is filtered out due to abnormal data.")
        continue
    filtered_vehicle_ids = np.append(filtered_vehicle_ids, i)  # TBD

    # ---------------------------------------- Data normalization ------------------------------
    if mode == MODE.PLOT_MODE:
        # plot_testX_timeseries(test_X, feature_names="Data for LSTM", title="vin1", figsize=(12, 6), save_path=f'{PATH}/vin1', show=False, seperate=True, start_idx=vin1_start)
        plot_testX_timeseries(
            tensor,
            feature_names="Data for voltage estimation",
            title="vin2",
            figsize=(12, 6),
            save_path=f"{PATH}/vin2",
            show=False,
            seperate=True,
            start_idx=vin2_start,
            _range=[0, dim_dict["x"] + 2],
            close=True,
        )
        plot_testX_timeseries(
            tensor,
            feature_names="Data for voltage estimation",
            title="vin2",
            figsize=(12, 6),
            save_path=f"{PATH}/vin2",
            show=False,
            seperate=True,
            start_idx=vin2_start,
            _range=[dim_dict["x"] + dim_dict["y"], dim_dict["x"] + dim_dict["y"] + 2],
            close=True,
        )
        plot_testX_timeseries(
            tensor,
            feature_names="Data for voltage estimation",
            title="vin2",
            figsize=(12, 6),
            save_path=f"{PATH}/vin2",
            show=False,
            seperate=True,
            start_idx=vin2_start,
            _range=[
                dim_dict["x"] + dim_dict["y"] + dim_dict["z"],
                dim_dict["x"] + dim_dict["y"] + dim_dict["z"] + dim_dict["q"] - 1,
            ],
            close=True,
        )
        # plot_testX_timeseries(combined_tensor, feature_names="Data for voltage estimation", title="vin2", figsize=(12, 6), save_path=f'{PATH}/vin2', show=False, seperate=True, start_idx=1000, _range=[0,combined_tensor.shape[-1]])
        plot_testX_timeseries(
            tensorx,
            feature_names="Data for SOC estimation",
            title="vin3",
            figsize=(12, 6),
            save_path=f"{PATH}/vin3",
            show=False,
            seperate=True,
            start_idx=vin3_start,
            _range=[0, dim_dict["x2"] + 2],
            close=True,
        )
        plot_testX_timeseries(
            tensorx,
            feature_names="Data for SOC estimation",
            title="vin3",
            figsize=(12, 6),
            save_path=f"{PATH}/vin3",
            show=False,
            seperate=True,
            start_idx=vin3_start,
            _range=[
                dim_dict["x2"] + dim_dict["y2"],
                dim_dict["x2"] + dim_dict["y2"] + 2,
            ],
            close=True,
        )
        plot_testX_timeseries(
            tensorx,
            feature_names="Data for SOC estimation",
            title="vin3",
            figsize=(12, 6),
            save_path=f"{PATH}/vin3",
            show=False,
            seperate=True,
            start_idx=vin3_start,
            _range=[
                dim_dict["x2"] + dim_dict["y2"] + dim_dict["z2"],
                dim_dict["x2"] + dim_dict["y2"] + dim_dict["z2"] + dim_dict["q2"] - 1,
            ],
            close=True,
        )

    elif mode == MODE.ANALYSIS_MODE:
        if FIRST_LOAD:
            FIRST_LOAD = False
            combined_tensor = tensor
            combined_tensorx = tensorx
        else:
            combined_tensor = torch.cat((combined_tensor, tensor), dim=0)
            combined_tensorx = torch.cat((combined_tensorx, tensorx), dim=0)

print("Start indices for each vehicle ID:", start_idxes)
print(f"Filtered vehicle IDs: {filtered_vehicle_ids}")

if mode == MODE.PLOT_MODE:
    exit()

if mode == MODE.FILTER_MODE:
    fault_list = [i for i in filtered_vehicle_ids if i >= fault_idx]
    normal_list = [i for i in filtered_vehicle_ids if i < fault_idx]

    # Calculate split sizes if ratios are provided
    total_normal = len(normal_list)
    if args.train_ratio is not None:
        n_train_set = int(total_normal * args.train_ratio)

    if "n_train_set" not in locals():
        n_train_set = 109 if BATTERY_TYPE == "QAS" else 39

    total_remain = total_normal - n_train_set

    if args.val_ratio is not None:
        # User requested: val_ratio applies to the remaining data after train split
        n_validation_set = int(total_remain * args.val_ratio)

    if "n_validation_set" not in locals():
        n_validation_set = 50 if BATTERY_TYPE == "QAS" else 19

    if n_train_set + n_validation_set > total_normal:
        print(
            f"Error: Requested train ({n_train_set}) + val ({n_validation_set}) > Total ({total_normal})"
        )
        exit(1)

    print(
        f"Dataset Split - Train: {n_train_set}, Val: {n_validation_set}, Total Normal: {total_normal}"
    )

    # Ensure output directory exists (created earlier via dataset_dir)
    save_path = dataset_dir

    np.save(
        f"{save_path}/{BATTERY_TYPE}_filtered_vehicle_ids_normal.npy",
        np.array(normal_list),
    )
    np.save(
        f"{save_path}/{BATTERY_TYPE}_filtered_vehicle_ids_fault.npy",
        np.array(fault_list),
    )

    train_list = np.random.choice(normal_list, size=n_train_set, replace=False)
    train_list.sort()
    remain_list = [vid for vid in normal_list if vid not in set(train_list)]
    validate_list = np.random.choice(remain_list, size=n_validation_set, replace=False)
    validate_list.sort()
    test_list = [vid for vid in remain_list if vid not in set(validate_list)]
    test_list.sort()

    np.save(
        f"{save_path}/{BATTERY_TYPE}_filtered_vehicle_ids_train.npy",
        np.array(train_list),
    )
    np.save(
        f"{save_path}/{BATTERY_TYPE}_filtered_vehicle_ids_validate.npy",
        np.array(validate_list),
    )
    np.save(
        f"{save_path}/{BATTERY_TYPE}_filtered_vehicle_ids_test.npy", np.array(test_list)
    )

    # Save dataset info for traceability
    with open(f"{save_path}/dataset_info_{BATTERY_TYPE}.txt", "w") as f:
        f.write(f"Dataset Name: {args.dataset_name}\n")
        f.write(f"Battery Type: {BATTERY_TYPE}\n")
        f.write(f"Seed: {args.seed}\n")
        f.write(
            f"Train Count: {len(train_list)} ({len(train_list)/total_normal:.2%})\n"
        )
        f.write(
            f"Val Count: {len(validate_list)} ({len(validate_list)/total_normal:.2%})\n"
        )
        f.write(f"Test Count: {len(test_list)} ({len(test_list)/total_normal:.2%})\n")
        f.write(f"Fault Count: {len(fault_list)}\n")

    print(f"Dataset saved to {save_path}")
    exit()

narray1 = np.array(combined_tensor)
narray2 = np.array(combined_tensor)
upre1 = narray1[:, 0]
upre2 = narray1[:, 1]
temp = narray1[:, dim_dict["x"] + dim_dict["y"] + dim_dict["z"]]
soc = narray1[:, dim_dict["x"] + dim_dict["y"] + dim_dict["z"] + 1]
current = narray1[:, dim_dict["x"] + dim_dict["y"] + dim_dict["z"] + 2]
u1 = narray1[:, dim_dict["x"]]
d_u1 = narray1[:, dim_dict["x"] + dim_dict["y"]]
u2 = narray1[:, dim_dict["x"] + 1]
d_u2 = narray1[:, dim_dict["x"] + dim_dict["y"] + 1]
d_soc1 = narray2[:, dim_dict["x2"] + dim_dict["y2"]]
d_soc2 = narray2[:, dim_dict["x2"] + dim_dict["y2"] + 1]
for i in range(dim_dict["y"]):
    if i == 0:
        u = narray1[:, dim_dict["x"] + i]
        d_u = narray1[:, dim_dict["x"] + dim_dict["y"] + i]
    else:
        u = np.vstack((u, narray1[:, dim_dict["x"] + i]))
        d_u = np.vstack((d_u, narray1[:, dim_dict["x"] + dim_dict["y"] + i]))


plot_distribution_with_stats(
    upre1,
    bins=60,
    kde=True,
    normal_fit=False,
    title="upre1 Sample Distribution",
    save_path=None,
    show=True,
)
plot_distribution_with_stats(
    upre2,
    bins=60,
    kde=True,
    normal_fit=False,
    title="upre2 Sample Distribution",
    save_path=None,
    show=True,
)
plot_distribution_with_stats(
    temp,
    bins=60,
    kde=True,
    normal_fit=False,
    title="Temperature Sample Distribution",
    save_path=None,
    show=True,
)
plot_distribution_with_stats(
    soc,
    bins=60,
    kde=True,
    normal_fit=False,
    title="SOC Sample Distribution",
    save_path=None,
    show=True,
)
plot_distribution_with_stats(
    current,
    bins=60,
    kde=True,
    normal_fit=False,
    title="Current Sample Distribution",
    save_path=None,
    show=True,
)

plot_distribution_with_stats(
    u1,
    bins=60,
    kde=True,
    normal_fit=False,
    title="True Voltage 1 Sample Distribution",
    save_path=None,
    show=True,
)

plot_distribution_with_stats(
    u2,
    bins=60,
    kde=True,
    normal_fit=False,
    title="True Voltage 2 Sample Distribution",
    save_path=None,
    show=True,
)

plot_distribution_with_stats(
    d_u1,
    bins=60,
    kde=True,
    normal_fit=False,
    title="Volatage divation 1 Sample Distribution",
    save_path=None,
    show=True,
)

plot_distribution_with_stats(
    d_u2,
    bins=60,
    kde=True,
    normal_fit=False,
    title="Volatage divation 2 Sample Distribution",
    save_path=None,
    show=True,
)

plot_distribution_with_stats(
    d_soc1,
    bins=60,
    kde=True,
    normal_fit=False,
    title="SOC divation 1 Sample Distribution",
    save_path=None,
    show=True,
)

plot_distribution_with_stats(
    d_soc2,
    bins=60,
    kde=True,
    normal_fit=False,
    title="SOC divation 2 Sample Distribution",
    save_path=None,
    show=True,
)

print("Amount of data used for training:", combined_tensor.shape[0])

# ----------------------------------------Training for MC-AE--------------------------
# x_recovered = combined_tensor[:, :dim_x] # 0~1
# y_recovered = combined_tensor[:, dim_x:dim_x + dim_y] # 2~111
# z_recovered = combined_tensor[:, dim_x + dim_y: dim_x + dim_y + dim_z] # 112~221
# q_recovered = combined_tensor[:, dim_x + dim_y + dim_z:] # 222~

# x_recovered2 = combined_tensorx[:, :dim_x2]
# y_recovered2 = combined_tensorx[:, dim_x2:dim_x2 + dim_y2]
# z_recovered2 = combined_tensorx[:, dim_x2 + dim_y2: dim_x2 + dim_y2 + dim_z2]
# q_recovered2 = combined_tensorx[:, dim_x2 + dim_y2 + dim_z2:]
