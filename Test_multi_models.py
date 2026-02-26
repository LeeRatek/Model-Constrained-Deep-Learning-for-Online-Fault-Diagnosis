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
import re
from pathlib import Path
import hashlib
import json

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
print(torch.cuda.device_count())

warnings.filterwarnings("ignore")

parser = argparse.ArgumentParser(
    description="Run diagnostics plotting with CLI options for multiple model combinations"
)

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
    "--models-idx",
    type=str,
    default="260206_084839",  # 260206_084839 or 260209_105821
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
parser.add_argument("--x-start", type=int, default=20)
parser.add_argument("--x-tick-step", type=int, default=3000)
parser.add_argument("--sigma-levels", type=str, default="3,4.5,6")
parser.add_argument(
    "--u-model-idx",
    type=str,
    default="auto",
    help="U model indices: 'auto' to scan artifact folder, comma-separated for manual (e.g., '30,40,50'), or '-1' for saved PCA",
)
parser.add_argument(
    "--x-model-idx",
    type=str,
    default="auto",
    help="X model indices: 'auto' to scan artifact folder, comma-separated for manual (e.g., '130,140,150'), or '-1' for saved PCA",
)
parser.add_argument(
    "--inference-batch-size",
    type=int,
    default=65536,
    help="Batch size for model inference in train_pca_only (default: 65536, optimized for large datasets)",
)
parser.add_argument(
    "--cache-strategy",
    type=str,
    default="disk",
    choices=["memory", "disk"],
    help="Caching strategy for training errors: 'memory' (faster, high RAM usage) or 'disk' (slower, saves RAM). Default: 'disk'",
)
parser.add_argument(
    "--resume",
    action="store_true",
    help="Resume from previous run. If true, attempts to reuse error files in temp_cache and skips already computed combinations in CSV.",
)
parser.add_argument(
    "--keep-cache",
    action="store_true",
    help="Keep the disk cache files (error matrices) after the simulation finishes. Useful for debugging or subsequent runs with --resume.",
)
parser.add_argument(
    "--use-float32",
    action="store_true",
    help="Convert error matrices to float32 (Single Precision) to save memory/disk (approx 50% reduction). Optional optimization.",
)
parser.add_argument(
    "--fast-search",
    action="store_true",
    help="Optimize search: 1. Find best U using last X model. 2. Sweep X models using best U. Reduces combinations from N*M to N+M.",
)
parser.add_argument(
    "--use-validate",
    action="store_true",
    help="Use validation set for model selection and evaluation.",
)
parser.add_argument(
    "--use-reverse-pos-nag",
    action="store_true",
    help="Use reverse positive-negative strategy for model selection and evaluation.",
)
args = parser.parse_args()


def scan_model_indices(artifact_dir, model_prefix):
    """
    Scan artifact directory for model files and extract indices.

    Args:
        artifact_dir: Path to artifact directory
        model_prefix: 'net' or 'netx'

    Returns:
        List of indices (integers), excluding -1 and 0
    """
    indices = []

    if not os.path.exists(artifact_dir):
        print(f"Warning: Artifact directory not found: {artifact_dir}")
        return indices

    # Pattern: net.pth, net_e30.pth, netx.pth, netx_e130.pth
    pattern = re.compile(rf"{model_prefix}(?:_e(\d+))?\.pth")

    for filename in os.listdir(artifact_dir):
        match = pattern.match(filename)
        if match:
            if match.group(1):  # net_e{idx}.pth
                idx = int(match.group(1))
                # Include index 0 as well
                if idx >= 0:
                    indices.append(idx)
            else:
                # net.pth (idx=-1)
                indices.append(-1)

    # Sort: positive indices ascending, then -1 at the end
    unique_indices = sorted(set(indices))
    if -1 in unique_indices:
        unique_indices.remove(-1)
        unique_indices.append(-1)

    return unique_indices


def sliding_average_np(s, n: int):
    a = np.asarray(s, dtype=float).reshape(-1)
    L = int(a.shape[0])
    n = int(n)
    if L == 0:
        return a
    if L <= n or n <= 0:
        return a
    out = np.empty(L, dtype=float)
    first_mean = float(np.mean(a[:n]))
    out[:n] = first_mean
    c = np.cumsum(a, dtype=float)
    c = np.concatenate(([0.0], c))
    window_sums = c[n:] - c[:-n]
    out[n:] = window_sums[:-1] / n
    return out


def extract_features_from_error(mode, error_matrix):
    """
    Extract diagnosis features from error matrix to reduce memory/disk usage.
    mode: 'u' or 'x'
    Returns: tuple of (max_diff, Z, Z_smoothed) as pd.Series
    """
    arr = np.asarray(error_matrix, dtype=float)
    eps = 1e-12
    arr_max = arr.max(axis=1)
    arr_mean = arr.mean(axis=1)
    arr_std = np.maximum(arr.std(axis=1), eps)

    Z = ((arr - arr_mean[:, None]) / arr_std[:, None]).max(axis=1)

    if arr.shape[1] >= 2:
        arr_second = np.partition(arr, -2, axis=1)[:, -2]
    else:
        arr_second = arr[:, 0]

    max_diff = (arr_max - arr_second) / arr_std

    # EWM Smoothing (alpha=0.2)
    s_max = pd.Series(arr_max)
    Z_smoothed = s_max.ewm(alpha=0.2).mean()

    if mode == "u":
        # U returns: max_diff, Z, Z_smoothed (No sliding window)
        return (pd.Series(max_diff), pd.Series(Z), pd.Series(Z_smoothed))

    elif mode == "x":
        # X returns: max_diff, Z, Z_smoothed (WITH sliding window 100)
        max_diff_slid = pd.Series(sliding_average_np(max_diff, 100))
        Z_slid = pd.Series(sliding_average_np(Z, 100))
        Z_smoothed_slid = pd.Series(sliding_average_np(Z_smoothed.to_numpy(), 100))

        return (max_diff_slid, Z_slid, Z_smoothed_slid)


def run_model_inference(
    model,
    combined_tensor,
    dim_dict,
    batch_size,
    device,
    is_x_model=False,
    use_abs_err=True,
    use_float32=False,
):
    """
    Run inference on combined tensor and return error matrix.
    Optimized to avoid repeated data loading/slicing in multi-model tests.
    """
    # Helper to slice tensor based on model type
    if is_x_model:
        x_recovered = combined_tensor[:, : dim_dict["x2"]]
        y_recovered = combined_tensor[
            :, dim_dict["x2"] : dim_dict["x2"] + dim_dict["y2"]
        ]
        z_recovered = combined_tensor[
            :,
            dim_dict["x2"]
            + dim_dict["y2"] : dim_dict["x2"]
            + dim_dict["y2"]
            + dim_dict["z2"],
        ]
        q_recovered = combined_tensor[
            :, dim_dict["x2"] + dim_dict["y2"] + dim_dict["z2"] :
        ]
    else:
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

    # Move to GPU/CPU efficiently based on device type
    is_gpu = device.type == "cuda"

    if is_gpu:
        x_recovered = x_recovered.double().to(device, non_blocking=True)
        z_recovered = z_recovered.double().to(device, non_blocking=True)
        q_recovered = q_recovered.double().to(device, non_blocking=True)
        y_recovered_device = y_recovered.double().to(device, non_blocking=True)
    else:
        x_recovered = x_recovered.double()
        z_recovered = z_recovered.double()
        q_recovered = q_recovered.double()
        y_recovered_device = y_recovered.double()

    num_samples = x_recovered.shape[0]
    error_list = []

    with torch.inference_mode():
        for start_idx in range(0, num_samples, batch_size):
            end_idx = min(start_idx + batch_size, num_samples)

            x_batch = x_recovered[start_idx:end_idx]
            z_batch = z_recovered[start_idx:end_idx]
            q_batch = q_recovered[start_idx:end_idx]
            y_batch = y_recovered_device[start_idx:end_idx]

            recon, _ = model(x_batch, z_batch, q_batch)

            if use_abs_err:
                error_batch = torch.abs(recon - y_batch)
            else:
                error_batch = recon - y_batch

            # Optional optimization: Convert to float32
            if use_float32:
                error_batch = error_batch.float()

            error_list.append(
                error_batch.cpu().numpy() if is_gpu else error_batch.numpy()
            )
            del recon, error_batch

    # Free memory
    del x_recovered, z_recovered, q_recovered, y_recovered_device
    if is_gpu:
        torch.cuda.empty_cache()

    return np.concatenate(error_list, axis=0)


# Parse model indices
artifact_dir = f"{args.models_dir}/{args.models_idx}/artifact"

if args.u_model_idx.lower() == "auto":
    print(f"Scanning artifact directory for U models: {artifact_dir}")
    u_model_indices = scan_model_indices(artifact_dir, "net")
    if not u_model_indices:
        print("Warning: No valid U model files found (excluding -1 and 0).")
        u_model_indices = []
    else:
        print(f"Found U model indices: {u_model_indices}")
else:
    u_model_indices = [int(idx.strip()) for idx in args.u_model_idx.split(",")]

if args.x_model_idx.lower() == "auto":
    print(f"Scanning artifact directory for X models: {artifact_dir}")
    x_model_indices = scan_model_indices(artifact_dir, "netx")
    if not x_model_indices:
        print("Warning: No valid X model files found (excluding -1 and 0).")
        x_model_indices = []
    else:
        print(f"Found X model indices: {x_model_indices}")
else:
    x_model_indices = [int(idx.strip()) for idx in args.x_model_idx.split(",")]

# Exit if no valid models found
if not u_model_indices or not x_model_indices:
    print("\nError: No valid model combinations to test. Exiting.")
    exit(1)

vals = read_values_from_sim_config(
    f"{args.models_dir}/{args.models_idx}",
    exclude_keys=[
        "models_dir",
        "models_idx",
        "source_data_dir",
    ],
    strict=False,
)

learning_case = vals.learning_case

add_one_output_layer = (
    vals.add_one_output_layer if hasattr(vals, "add_one_output_layer") else False
)
use_dx = not vals.no_use_dx if hasattr(vals, "no_use_dx") else True

## ================== For backward compatibility =================== ##
ae_u_scale = vals.ae_u_scale if hasattr(vals, "ae_u_scale") else 1.8
ae_u_shift = vals.ae_u_shift if hasattr(vals, "ae_u_shift") else 2.5
ae_x_scale = vals.ae_x_scale if hasattr(vals, "ae_x_scale") else 1.0
ae_x_shift = vals.ae_x_shift if hasattr(vals, "ae_x_shift") else 0
normalize_dx_flag = vals.normalize_dx if hasattr(vals, "normalize_dx") else False

# 플롯 결과 저장 폴더는 한 번만 생성
os.makedirs(f"{args.models_dir}/{args.models_idx}/results", exist_ok=True)

BATTERY_TYPE = vals.battery_type  # 'DTI' or 'QAS'
PREPROCESSING, SKIP_CHARGE_READY = get_preprocessing_and_skip_charge_ready(
    learning_case
)

# Determine Dataset Path based on trained model config
DATA_BASE_PATH = read_dataset_path_from_sim_config(f"{args.models_dir}/{args.models_idx}")
print(f"Using dataset from: {DATA_BASE_PATH}")

dim_dict = get_input_dimensions(BATTERY_TYPE)

print_sim_config(
    title="Multi-Model Test Run",
    config=args,
    extra={
        "device": str(device),
        "learning_case": learning_case,
        "PREPROCESSING": PREPROCESSING,
        "SKIP_CHARGE_READY": SKIP_CHARGE_READY,
        "no_use_dx": not use_dx,
        "add_one_output_layer": add_one_output_layer,
        "u_model_indices": u_model_indices,
        "x_model_indices": x_model_indices,
        "total_combinations": len(u_model_indices) * len(x_model_indices),
    },
)

try:
    plot_loss_curve_all(path=f"{args.models_dir}/{args.models_idx}")
except Exception as e:
    print(f"Error plotting loss curve: {e}")

## =================== Load test data lists =================== ##
normal_test = (
    np.load(f"{DATA_BASE_PATH}/{BATTERY_TYPE}_filtered_vehicle_ids_test.npy")
    .astype(np.int64)
    .tolist()
)
normal_validate = (
    np.load(f"{DATA_BASE_PATH}/{BATTERY_TYPE}_filtered_vehicle_ids_validate.npy")
    .astype(np.int64)
    .tolist()
)
normal_list = [*normal_test, *normal_validate] if args.use_validate else normal_test
fault_list = (
    np.load(f"{DATA_BASE_PATH}/{BATTERY_TYPE}_filtered_vehicle_ids_fault.npy")
    .astype(np.int64)
    .tolist()
)

test_list = (
    [normal_list, fault_list]
    if not args.use_reverse_pos_nag
    else [fault_list, normal_list]
)

## =================== Initialize params for AUROC curve  =================== ##
total_test_vehicles = len(normal_list) + len(fault_list)
predict_threshold_array = np.arange(0, 1000, 0.1)
predict_thresholds = np.asarray(list(predict_threshold_array), dtype=float)

# Store results for all model combinations
all_roc_results = []

# Define CSV path for incremental saving
csv_path = f"{args.models_dir}/{args.models_idx}/AUC_summary_case{learning_case}.csv"

completed_combinations = set()

# Initialize completed combinations set if resuming
if args.resume and os.path.exists(csv_path):
    print(f"Resuming mode enabled. Checking cached results in {csv_path}...")
    try:
        existing_df = pd.read_csv(csv_path)
        # Check if columns exist
        if (
            "u_model_idx" in existing_df.columns
            and "x_model_idx" in existing_df.columns
        ):
            for _, row in existing_df.iterrows():
                completed_combinations.add(
                    (int(row["u_model_idx"]), int(row["x_model_idx"]))
                )
        print(
            f"  Found {len(completed_combinations)} already computed combinations. These will be skipped."
        )
    except Exception as e:
        print(f"Warning: Could not read existing CSV for resume: {e}")

# Clear existing CSV file if it exists to start fresh (ONLY if not resuming)
if not args.resume and os.path.exists(csv_path):
    os.remove(csv_path)
    print(f"Removed previous AUC summary file: {csv_path}")

# Pre-load all models to avoid repeated loading
print("\nPre-loading all models...")
models_u = {}
models_x = {}

for u_idx in u_model_indices:
    if u_idx == -1:
        u_model_idx = "net"
    else:
        u_model_idx = f"net_e{u_idx}"
    net_loaded = CombinedAE(
        input_size=dim_dict["x"],
        encode2_input_size=dim_dict["q"],
        output_size=dim_dict["y"],
        activation_fn=(
            CustomSigmoidFunc(scale=ae_u_scale, shift=ae_u_shift)
            if PREPROCESSING == False
            else CustomSigmoidFunc(scale=1, shift=0)
        ),
        use_dx_in_forward=use_dx,
        add_one_output_layer=add_one_output_layer,
    ).to(device)
    net_state_dict = torch.load(
        os.path.join(
            f"{args.models_dir}/{args.models_idx}/artifact/",
            f"{u_model_idx}.pth",
        ),
        map_location=device,
    )
    net_loaded.load_state_dict(net_state_dict)
    # Revert to double for precision accuracy (AUC integrity)
    net_loaded.double().eval()
    models_u[u_idx] = net_loaded
    del net_state_dict  # Free memory
    print(f"  Loaded U model: {u_model_idx}")

for x_idx in x_model_indices:
    if x_idx == -1:
        x_model_idx = "netx"
    else:
        x_model_idx = f"netx_e{x_idx}"
    netx_loaded = CombinedAE(
        input_size=dim_dict["x2"],
        encode2_input_size=dim_dict["q2"],
        output_size=dim_dict["y2"],
        activation_fn=(
            CustomSigmoidFunc(scale=ae_x_scale, shift=ae_x_shift)
            if PREPROCESSING == False
            else CustomSigmoidFunc(scale=1, shift=0)
        ),
        use_dx_in_forward=use_dx,
        add_one_output_layer=add_one_output_layer,
    ).to(device)
    netx_state_dict = torch.load(
        os.path.join(
            f"{args.models_dir}/{args.models_idx}/artifact/",
            f"{x_model_idx}.pth",
        ),
        map_location=device,
    )
    netx_loaded.load_state_dict(netx_state_dict)
    # Revert to double for precision accuracy
    netx_loaded.double().eval()
    models_x[x_idx] = netx_loaded
    del netx_state_dict  # Free memory
    print(f"  Loaded X model: {x_model_idx}")


print(
    f"All models loaded successfully. Total: {len(models_u)} U models, {len(models_x)} X models\n"
)

# ================= Check if all error/feature files already exist ================= #
# If we are using disk cache and all files exist, we can skip loading combined_tensor_train.
all_files_exist = False
if args.cache_strategy == "disk":
    temp_cache_dir_check = os.path.join(
        f"{args.models_dir}/{args.models_idx}", "temp_cache"
    )
    if os.path.exists(temp_cache_dir_check):
        missing_u = []
        for u_idx in u_model_indices:
            fpath = os.path.join(temp_cache_dir_check, f"feat_u_{u_idx}.pkl")
            if not os.path.exists(fpath):
                missing_u.append(u_idx)

        missing_x = []
        for x_idx in x_model_indices:
            fpath = os.path.join(temp_cache_dir_check, f"feat_x_{x_idx}.pkl")
            if not os.path.exists(fpath):
                missing_x.append(x_idx)

        if not missing_u and not missing_x:
            all_files_exist = True
            print(
                "All required feature files found in disk cache. Skipping training data loading."
            )
        else:
            print(
                f"Missing cached files: U={missing_u}, X={missing_x}. Will load training data."
            )

# Pre-load training data once to avoid repeated loading in train_pca_only
if not all_files_exist:
    print("Pre-loading training data...")
    train_list = (
        np.load(f"{DATA_BASE_PATH}/{BATTERY_TYPE}_filtered_vehicle_ids_train.npy")
        .astype(np.int64)
        .tolist()
    )

    # Apply vehicle range if specified in vals
    vehicle_start = vals.vehicle_start if hasattr(vals, "vehicle_start") else 0
    vehicle_end = vals.vehicle_end if hasattr(vals, "vehicle_end") else -1
    train_list = (
        train_list[vehicle_start : vehicle_end + 1]
        if vehicle_end != -1
        else train_list[vehicle_start:]
    )

    # ================= Check Cache for Combined Data ================= #

    # Generate unique hash for this configuration
    config_dict = {
        "train_list": train_list,
        "battery_type": BATTERY_TYPE,
        "preprocessing": PREPROCESSING,
        "skip_charge_ready": SKIP_CHARGE_READY,
        "normalize_dx": normalize_dx_flag,
    }
    config_str = json.dumps(config_dict, sort_keys=True)
    config_hash = hashlib.md5(config_str.encode("utf-8")).hexdigest()

    # Changed to list cache
    cache_filename = f"train_data_list_{BATTERY_TYPE}_{config_hash}.pt"
    cache_path = os.path.join(args.source_data_dir, cache_filename)

    train_data_cache = None  # List of (tensor, tensorx)

    if os.path.exists(cache_path):
        print(f"Found cached training data (list): {cache_path}")
        print("Loading...")
        try:
            train_data_cache = torch.load(cache_path)
            print("Successfully loaded cached training data.")
        except Exception as e:
            print(f"Error loading cache: {e}. Will regenerate.")

    if train_data_cache is None:
        print("Cache miss or error. Processing raw pickle files...")
        train_data_cache = []

        count = 0
        for i in train_list:
            count += 1
            if count % 10 == 0:
                print(f"  Processing vehicle {count}/{len(train_list)}")
            VEHICLE_ID = f"{i}"

            tensor = safe_load(
                f"{args.source_data_dir}/{BATTERY_TYPE}/{VEHICLE_ID}/vin_2.pkl",
                verbose=False,
            )
            tensorx = safe_load(
                f"{args.source_data_dir}/{BATTERY_TYPE}/{VEHICLE_ID}/vin_3.pkl",
                verbose=False,
            )

            tensor, tensorx = preprocess_loaded_tensor(
                tensor,
                tensorx,
                dim_dict,
                BATTERY_TYPE,
                PREPROCESSING,
                SKIP_CHARGE_READY,
                normalize_dx=normalize_dx_flag,
            )

            # Keep as Double
            tensor = tensor.double()
            tensorx = tensorx.double()

            train_data_cache.append((tensor, tensorx))

        # Save to cache
        print(f"Saving training data list to cache: {cache_path}")
        try:
            torch.save(train_data_cache, cache_path)
            print("Cache saved successfully.")
        except Exception as e:
            print(f"Warning: Could not save cache to {cache_path}: {e}")

    total_samples = sum(t[0].shape[0] for t in train_data_cache)
    print(
        f"Training data loaded: {len(train_data_cache)} vehicles, {total_samples} total samples (Double precision)\n"
    )
else:
    # Just needed to skip the loading block
    pass

# Pre-load all test data once to avoid repeated loading
print("Pre-loading test data...")
test_data_cache = {}  # Dictionary to store {vehicle_id: (tensor, tensor_x)}

# ================= Check Cache for Test Data ================= #
# Create a flat list of test vehicle IDs for hash generation
flat_test_list = []
for sublist in test_list:
    flat_test_list.extend(sublist)

test_config_dict = {
    "test_list": flat_test_list,
    "battery_type": BATTERY_TYPE,
    "preprocessing": PREPROCESSING,
    "skip_charge_ready": SKIP_CHARGE_READY,
    "normalize_dx": normalize_dx_flag,
}
test_config_str = json.dumps(test_config_dict, sort_keys=True)
test_config_hash = hashlib.md5(test_config_str.encode("utf-8")).hexdigest()

test_cache_filename = f"test_data_cache_{BATTERY_TYPE}_{test_config_hash}.pt"
test_cache_path = os.path.join(args.source_data_dir, test_cache_filename)

if os.path.exists(test_cache_path):
    print(f"Found cached test data: {test_cache_path}")
    print("Loading...")
    try:
        test_data_cache = torch.load(test_cache_path)
        print(f"Successfully loaded cached test data: {len(test_data_cache)} vehicles")
    except Exception as e:
        print(f"Error loading test cache: {e}. Will regenerate.")
        test_data_cache = {}

if not test_data_cache:
    print("Cache miss or error. Processing raw test pickle files...")
    count = 0
    for label, vehicle_ids in enumerate(test_list):
        for i in vehicle_ids:
            count += 1
            if count % 10 == 0:
                print(f"  Loading test vehicle {count}/{total_test_vehicles}")

            VEHICLE_ID = f"{i}"

            tensor = safe_load(
                f"{args.source_data_dir}/{BATTERY_TYPE}/{VEHICLE_ID}/vin_2.pkl",
                verbose=False,
            )
            tensor_x = safe_load(
                f"{args.source_data_dir}/{BATTERY_TYPE}/{VEHICLE_ID}/vin_3.pkl",
                verbose=False,
            )

            tensor, tensor_x = preprocess_loaded_tensor(
                tensor,
                tensor_x,
                dim_dict,
                BATTERY_TYPE,
                PREPROCESSING,
                SKIP_CHARGE_READY,
                normalize_dx=normalize_dx_flag,
            )

            test_data_cache[i] = (tensor, tensor_x)

    # Save to cache
    print(f"Saving test data to cache: {test_cache_path}")
    try:
        torch.save(test_data_cache, test_cache_path)
        print("Test cache saved successfully.")
    except Exception as e:
        print(f"Warning: Could not save test cache to {test_cache_path}: {e}")

print(f"Test data loaded: {len(test_data_cache)} vehicles\n")

# Initialize caches for training errors to avoid redundant computations
import tempfile
import shutil

# Caching Strategy Setup
cache_strategy = args.cache_strategy
temp_cache_dir = None
u_error_cache = {}  # Used for direct memory storage
x_error_cache = {}
u_error_files = {}  # Used for disk storage paths
x_error_files = {}

if cache_strategy == "disk":
    # Create a fixed temporary directory for disk caching (easier manual cleanup)
    temp_cache_dir = os.path.join(f"{args.models_dir}/{args.models_idx}", "temp_cache")

    # Clean up previous cache if it exists (e.g. from a crashed run), unless resuming
    # if os.path.exists(temp_cache_dir) and not args.resume:
    #     try:
    #         shutil.rmtree(temp_cache_dir)
    #         print(f"Cleared stale cache directory: {temp_cache_dir}")
    #     except OSError as e:
    #         print(f"Warning: Could not clear old cache directory {temp_cache_dir}: {e}")

    os.makedirs(temp_cache_dir, exist_ok=True)
    if args.resume:
        print(
            f"Using DISK cache for training FEATURES (not raw errors) at: {temp_cache_dir} (Resuming mode)"
        )
    else:
        print(
            f"Using DISK cache for training FEATURES (not raw errors) at: {temp_cache_dir}"
        )
else:
    print(
        "Using MEMORY cache for training FEATURES (not raw errors) (Optimized RAM usage)"
    )

# Use abs error based on vals (sim_config)
use_abs_err = not vals.no_abs_err if hasattr(vals, "no_abs_err") else True
inference_batch_size = args.inference_batch_size

# Phase 1: Compute and Cache U FEATURES (Extracted from errors)
print(f"\nPhase 1: Computing and caching U FEATURES to {cache_strategy}...")
for u_idx in u_model_indices:
    # Check for existing cache if resuming OR if we determined all files exist (which implies we should respect them even if resume flag not explicitly set, though likely it is set if we are here)
    # Actually, if all_files_exist is True, it means we found everything. We should just verify specific file exists.

    fpath = os.path.join(temp_cache_dir, f"feat_u_{u_idx}.pkl")
    if cache_strategy == "disk" and os.path.exists(fpath):
        print(f"  Found cached U features for model {u_idx}, skipping computation.")
        u_error_files[u_idx] = fpath
        continue

    # If we are here, we MUST compute.
    # If 'all_files_exist' was True but we are here, something is wrong (file deleted in between?) OR we are not using disk cache?
    # Whatever, if we are here, we need train_data_cache.
    # If all_files_exist was True, train_data_cache is NOT loaded.
    if "train_data_cache" not in locals() or train_data_cache is None:
        raise RuntimeError(
            "Training data not loaded but feature computation needed! (Cache inconsistency)"
        )

    print(
        f"  Computing training features for U model {u_idx} (Per-vehicle processing)..."
    )

    # Process per vehicle
    feat_u_list = []

    for v_tensor, _ in train_data_cache:
        error_u_veh = run_model_inference(
            models_u[u_idx],
            v_tensor,
            dim_dict,
            inference_batch_size,
            device,
            is_x_model=False,
            use_abs_err=use_abs_err,
            use_float32=args.use_float32,
        )
        # Extract features for this vehicle
        feat_veh_tuple = extract_features_from_error("u", error_u_veh)
        feat_u_list.append(feat_veh_tuple)
        del error_u_veh

    # Concatenate features
    if feat_u_list:
        feat_u_final = (
            pd.concat([t[0] for t in feat_u_list], ignore_index=True),
            pd.concat([t[1] for t in feat_u_list], ignore_index=True),
            pd.concat([t[2] for t in feat_u_list], ignore_index=True),
        )
    else:
        feat_u_final = (pd.Series(), pd.Series(), pd.Series())

    if cache_strategy == "disk":
        # Save to disk as pickle (pd.Series tuple)
        fpath = os.path.join(temp_cache_dir, f"feat_u_{u_idx}.pkl")
        with open(fpath, "wb") as f:
            import pickle

            pickle.dump(feat_u_final, f)
        u_error_files[u_idx] = fpath
    else:
        # Keep in memory
        u_error_cache[u_idx] = feat_u_final

# Phase 2: Compute and Cache X FEATURES
print(f"\nPhase 2: Computing and caching X FEATURES to {cache_strategy}...")
for x_idx in x_model_indices:
    fpath = os.path.join(temp_cache_dir, f"feat_x_{x_idx}.pkl")
    if cache_strategy == "disk" and os.path.exists(fpath):
        print(f"  Found cached X features for model {x_idx}, skipping computation.")
        x_error_files[x_idx] = fpath
        continue

    if "train_data_cache" not in locals() or train_data_cache is None:
        raise RuntimeError("Training data not loaded but feature computation needed!")

    print(
        f"  Computing training features for X model {x_idx} (Per-vehicle processing)..."
    )

    # Process per vehicle
    feat_x_list = []

    for _, v_tensorx in train_data_cache:
        error_x_veh = run_model_inference(
            models_x[x_idx],
            v_tensorx,
            dim_dict,
            inference_batch_size,
            device,
            is_x_model=True,
            use_abs_err=use_abs_err,
            use_float32=args.use_float32,
        )

        # Extract features for this vehicle
        feat_veh_tuple = extract_features_from_error("x", error_x_veh)
        feat_x_list.append(feat_veh_tuple)
        del error_x_veh

    # Concatenate features
    if feat_x_list:
        feat_x_final = (
            pd.concat([t[0] for t in feat_x_list], ignore_index=True),
            pd.concat([t[1] for t in feat_x_list], ignore_index=True),
            pd.concat([t[2] for t in feat_x_list], ignore_index=True),
        )
    else:
        feat_x_final = (pd.Series(), pd.Series(), pd.Series())

    if cache_strategy == "disk":
        # Save to disk
        fpath = os.path.join(temp_cache_dir, f"feat_x_{x_idx}.pkl")
        with open(fpath, "wb") as f:
            import pickle

            pickle.dump(feat_x_final, f)
        x_error_files[x_idx] = fpath
    else:
        # Keep in memory
        x_error_cache[x_idx] = feat_x_final

# Critical: Release original training data to free up RAM before PCA loop
print("Releasing training data from memory...")
if "train_data_cache" in locals() and train_data_cache is not None:
    del train_data_cache
if torch.cuda.is_available():
    torch.cuda.empty_cache()

# Determine the list of combinations to process based on fast-search strategy or full search
combinations_to_run = []

if args.fast_search and len(x_model_indices) > 0 and len(u_model_indices) > 0:
    print(f"\n{'='*80}")
    print("FAST SEARCH MODE ENABLED")
    print("Step 1: Finding best U model using the last X model...")
    print(f"{'='*80}\n")

    last_x_idx = x_model_indices[-1]

    # Step 1: Run all U models against the last X model
    step1_results = []

    # Check if we already have results for this step to avoid re-calculation if possible,
    # but we need the AUC values to sort. We'll rely on the resume logic inside the loop
    # or just let the main loop handle it, but here we need to enforce the order.

    # We will generate a list of (u, x) tuples to run in order.
    # However, fast search requires the RESULT of step 1 to proceed to step 2.
    # So we cannot pre-generate the full list. We must run the loop in two phases.

    # PHASE 1 LOOP
    combinations_phase1 = [(u, last_x_idx) for u in u_model_indices]
else:
    # Standard full grid search
    combinations_phase1 = [
        (u_idx, x_idx) for x_idx in x_model_indices for u_idx in u_model_indices
    ]


# Function to process a single combination (refactored from original loop)
def process_combination(u_idx, x_idx):
    global predict_thresholds, total_test_vehicles, completed_combinations, all_roc_results

    # SKIP LOGIC for RESUME (if result already in CSV)
    if (u_idx, x_idx) in completed_combinations:
        print(f"Skipping cached combination: u={u_idx}, x={x_idx}")
        # We need to retrieve the AUC if we are in fast search mode to determine the winner
        # Try to read from all_roc_results if available (populated from CSV earlier?)
        # Since we didn't populate all_roc_results from CSV, we might miss it.
        # But for fast search Step 1, we critically need the AUC.
        # If resuming, we should ideally read the AUC from the CSV file row.

        auc_val = 0.0
        if args.fast_search:
            # Try to find AUC in existing CSV data
            try:
                if os.path.exists(csv_path):
                    existing_df = pd.read_csv(csv_path)
                    match = existing_df[
                        (existing_df["u_model_idx"] == u_idx)
                        & (existing_df["x_model_idx"] == x_idx)
                    ]
                    if not match.empty:
                        auc_val = float(match.iloc[0]["AUC"])
            except:
                pass
        return auc_val

    print(f"\n{'='*80}")
    print(f"Testing model combination: u_model_idx={u_idx}, x_model_idx={x_idx}")
    print(f"{'='*80}\n")

    # Initialize prediction results for this combination
    predict_results = np.zeros(
        (total_test_vehicles, len(predict_thresholds)), dtype=np.int8
    )
    y_true = np.zeros(total_test_vehicles, dtype=np.int8)  # normal=0, fault=1

    ## =================== Load PCA data for each combination  =================== ##
    start_pca_load = time.perf_counter()

    # Load cached features (fast!)
    if cache_strategy == "disk":
        import pickle

        print("  Loading pre-computed FEATURES from disk...")
        with open(u_error_files[u_idx], "rb") as f:
            u_feat = pickle.load(f)
        with open(x_error_files[x_idx], "rb") as f:
            x_feat = pickle.load(f)
    else:
        print("  Using pre-computed FEATURES from memory...")
        u_feat = u_error_cache[u_idx]
        x_feat = x_error_cache[x_idx]

    # Construct df_data directly from features
    df_data = pd.concat(
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

    # Run PCA on features
    loads = Custom_PCA(df_data, 0.99, 0.99)

    # Clean up large PCA artifacts
    loads_list = list(loads)
    loads_list[13] = None  # Discard X (scores)
    loads_list[14] = None  # Discard data_nor
    loads = tuple(loads_list)

    # Explicitly delete temporary df_data to save memory
    del df_data, u_feat, x_feat

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

    del X, data_nor, loads
    elapsed_pca_load = time.perf_counter() - start_pca_load
    print(f"PCA results loading/computing time: {elapsed_pca_load:.3f} seconds")

    # Use pre-loaded models
    net_loaded = models_u[u_idx]
    netx_loaded = models_x[x_idx]

    start = time.perf_counter()
    test_idx = 0
    for label, vehicle_ids in enumerate(test_list):
        for i in vehicle_ids:
            if test_idx % 10 == 0:
                elapsed = time.perf_counter() - start
                h, rem = divmod(elapsed, 3600)
                m, s = divmod(rem, 60)
                print(
                    f"Processing vehicle {test_idx}/{total_test_vehicles} | Elapsed: {int(h)}h {int(m)}m {s:.1f}s"
                )

            # Use pre-loaded test data
            tensor, tensor_x = test_data_cache[i]

            # Use indexing to separate
            x_recovered = tensor[:, : dim_dict["x"]]
            y_recovered = tensor[:, dim_dict["x"] : dim_dict["x"] + dim_dict["y"]]
            z_recovered = tensor[
                :,
                dim_dict["x"]
                + dim_dict["y"] : dim_dict["x"]
                + dim_dict["y"]
                + dim_dict["z"],
            ]
            q_recovered = tensor[:, dim_dict["x"] + dim_dict["y"] + dim_dict["z"] :]

            # Use indexing to separate
            x_recovered2 = tensor_x[:, : dim_dict["x2"]]
            y_recovered2 = tensor_x[:, dim_dict["x2"] : dim_dict["x2"] + dim_dict["y2"]]
            z_recovered2 = tensor_x[
                :,
                dim_dict["x2"]
                + dim_dict["y2"] : dim_dict["x2"]
                + dim_dict["y2"]
                + dim_dict["z2"],
            ]
            q_recovered2 = tensor_x[
                :, dim_dict["x2"] + dim_dict["y2"] + dim_dict["z2"] :
            ]

            # Revert inputs to double
            x_recovered = x_recovered.double()
            y_recovered = y_recovered.double()
            z_recovered = z_recovered.double()
            q_recovered = q_recovered.double()

            x_recovered2 = x_recovered2.double()
            y_recovered2 = y_recovered2.double()
            z_recovered2 = z_recovered2.double()
            q_recovered2 = q_recovered2.double()

            with torch.inference_mode():
                recon_imtest = net_loaded(
                    x_recovered, z_recovered, q_recovered, y_recovered
                )
                reconx_imtest = netx_loaded(
                    x_recovered2, z_recovered2, q_recovered2, y_recovered2
                )

                ERRORU = torch.abs(recon_imtest[0] - y_recovered).cpu().numpy()
                ERRORX = torch.abs(reconx_imtest[0] - y_recovered2).cpu().numpy()

                del recon_imtest, reconx_imtest

            df_data, _ = DiagnosisFeature(ERRORU, ERRORX)

            ## =================== Testing Diagnosis =================== ##
            t2_array, _ = T2_array(df_data, data_mean, data_std, p_k, v_I)
            spe_array, _ = SPE_array(df_data, data_mean, data_std, p_k)
            CI_array = (spe_array / SPE_95_limit) + (t2_array / T_95_limit)

            ci_max = float(np.nanmax(np.asarray(CI_array, dtype=float)))
            predict_results[test_idx, :] = (ci_max > predict_thresholds).astype(np.int8)
            y_true[test_idx] = np.int8(label)
            test_idx += 1

            del x_recovered, y_recovered, z_recovered, q_recovered
            del x_recovered2, y_recovered2, z_recovered2, q_recovered2
            del ERRORU, ERRORX, df_data
            del t2_array, spe_array, CI_array

    elapsed = time.perf_counter() - start
    h, rem = divmod(elapsed, 3600)
    m, s = divmod(rem, 60)
    print(
        f"\nCombination (u={u_idx}, x={x_idx}) completed: {int(h)}h {int(m)}m {s:.3f}s\n"
    )

    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # =================== ROC / AUC ===================
    roc = compute_roc_auc_from_threshold_matrix(
        y_true, predict_results, predict_thresholds
    )
    opt = compute_optimal_thresholds_from_roc(
        roc["fpr"],
        roc["tpr"],
        roc["thresholds"],
        candidate_idx=roc["candidate_idx"],
    )

    print(f"AUC = {roc['auc']:.4f}")

    # Save individual ROC curve
    roc_save_path = f"{args.models_dir}/{args.models_idx}/results/AUC_ROC_u{u_idx}_x{x_idx}_case{learning_case}.png"
    plot_roc_curve(
        roc["fpr"],
        roc["tpr"],
        roc["auc"],
        best_point=opt["best"],
        closest_to_01_point=opt["closest_to_01"],
        save_path=roc_save_path,
        title=f"AUC-ROC Curve (u_model={u_idx}, x_model={x_idx})",
        figsize=(6, 6),
        show=False,
        dpi=200,
    )
    print(f"ROC curve saved to: {roc_save_path}\n")

    # Store results for combined plot
    all_roc_results.append(
        {
            "u_idx": u_idx,
            "x_idx": x_idx,
            "fpr": roc["fpr"],
            "tpr": roc["tpr"],
            "auc": roc["auc"],
            "best": opt["best"],
            "closest_to_01": opt["closest_to_01"],
        }
    )

    # =================== Save AUC result immediately to CSV =================== ##
    auc_record = pd.DataFrame(
        [
            {
                "u_model_idx": u_idx,
                "x_model_idx": x_idx,
                "AUC": roc["auc"],
                "best_threshold": opt["best"]["threshold"],
                "best_TPR": opt["best"]["tpr"],
                "best_FPR": opt["best"]["fpr"],
                "closest_01_threshold": opt["closest_to_01"]["threshold"],
                "closest_01_TPR": opt["closest_to_01"]["tpr"],
                "closest_01_FPR": opt["closest_to_01"]["fpr"],
                "closest_01_distance": opt["closest_to_01"]["distance"],
            }
        ]
    )

    if not os.path.exists(csv_path):
        auc_record.to_csv(csv_path, mode="w", header=True, index=False)
    else:
        auc_record.to_csv(csv_path, mode="a", header=False, index=False)

    # Mark as completed
    completed_combinations.add((u_idx, x_idx))

    return roc["auc"]


# Run Logic
best_u_idx = None

# Phase 1
for u_idx, x_idx in combinations_phase1:
    auc = process_combination(u_idx, x_idx)
    # Track best U if in fast search mode
    if args.fast_search and "step1_results" in locals():
        step1_results.append((u_idx, auc))

# Phase 2 (Only for Fast Search)
if args.fast_search and step1_results:
    # Find best U
    best_u_tuple = max(step1_results, key=lambda item: item[1])
    best_u_idx = best_u_tuple[0]
    best_auc = best_u_tuple[1]

    print(f"\n{'='*80}")
    print(f"Step 1 Complete. Best U model: u={best_u_idx} (AUC={best_auc:.4f})")
    print("Step 2: Sweeping all X models using the best U model...")
    print(f"{'='*80}\n")

    # Remove duplicates if last_x_idx search overlaps (it's already done)
    # But usually we just sweep all X for this U
    combinations_phase2 = [
        (best_u_idx, x) for x in x_model_indices if x != combinations_phase1[0][1]
    ]

    # In fast search phase 1 we did all U against LAST X.
    # Now we do BEST U against ALL X.
    # Note: (best_u_idx, last_x_idx) is already done in phase 1.

    last_x_idx = x_model_indices[-1]
    combinations_phase2 = [
        (best_u_idx, x)
        for x in x_model_indices
        if (best_u_idx, x) not in completed_combinations
    ]

    for u_idx, x_idx in combinations_phase2:
        process_combination(u_idx, x_idx)

original_loop_code_marker = (
    False  # Just a marker to ensure we broke the original loop flow structure
)


# Cleanup temp files if disk strategy was used and keep_cache is False
if cache_strategy == "disk" and temp_cache_dir and os.path.exists(temp_cache_dir):
    if args.keep_cache:
        print(
            f"Keeping temporary disk cache at: {temp_cache_dir} (--keep-cache enabled)"
        )
    else:
        print(f"Cleaning up temporary disk cache: {temp_cache_dir}")
        shutil.rmtree(temp_cache_dir)

if cache_strategy == "memory":
    # Clear memory cache explicitly
    u_error_cache.clear()
    x_error_cache.clear()
    del u_error_cache, x_error_cache


# =================== Plot all ROC curves on one figure =================== ##
if len(all_roc_results) > 1:
    print(f"\n{'='*80}")
    print("Plotting all ROC curves on a single figure...")
    print(f"{'='*80}\n")

    fig, ax = plt.subplots(figsize=(10, 8))

    # Plot each ROC curve
    for result in all_roc_results:
        label = f"u={result['u_idx']}, x={result['x_idx']} (AUC={result['auc']:.4f})"
        ax.plot(result["fpr"], result["tpr"], linewidth=2, label=label)

    # Plot diagonal reference line
    ax.plot([0, 1], [0, 1], "k--", linewidth=1, label="Random")

    ax.set_xlabel("False Positive Rate", fontsize=12)
    ax.set_ylabel("True Positive Rate", fontsize=12)
    ax.set_title(
        "ROC Curves Comparison - All Model Combinations", fontsize=14, fontweight="bold"
    )
    ax.legend(loc="lower right", fontsize=9)
    ax.grid(True, alpha=0.3)
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])

    combined_roc_path = f"{args.models_dir}/{args.models_idx}/results/AUC_ROC_ALL_COMBINATIONS_case{learning_case}.png"
    plt.tight_layout()
    plt.savefig(combined_roc_path, dpi=200, bbox_inches="tight")
    plt.close()

    print(f"Combined ROC curve saved to: {combined_roc_path}")

    # Print summary table
    print(f"\n{'='*80}")
    print("Summary of all model combinations:")
    print(f"{'='*80}")
    print(
        f"{'u_idx':<10} {'x_idx':<10} {'AUC':<10} {'Best Threshold':<15} {'Best TPR':<10} {'Best FPR':<10}"
    )
    print(f"{'-'*80}")
    for result in all_roc_results:
        print(
            f"{result['u_idx']:<10} {result['x_idx']:<10} {result['auc']:<10.4f} {result['best']['threshold']:<15.6g} {result['best']['tpr']:<10.4f} {result['best']['fpr']:<10.4f}"
        )
    print(f"{'='*80}\n")

    # =================== Create AUC heatmap visualization =================== ##
    heatmap_path = (
        f"{args.models_dir}/{args.models_idx}/AUC_heatmap_case{learning_case}.png"
    )
    draw_auc_heatmap(csv_path, heatmap_path)

print("\n✓ All model combinations tested successfully!")
