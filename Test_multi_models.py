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
                # Exclude indices -1 and 0
                if idx > 0:
                    indices.append(idx)
            # Skip net.pth (idx=-1)

    return sorted(set(indices))  # Remove duplicates and sort


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

# 플롯 결과 저장 폴더는 한 번만 생성
os.makedirs(f"{args.models_dir}/{args.models_idx}/results", exist_ok=True)

BATTERY_TYPE = vals.battery_type  # 'DTI' or 'QAS'
PREPROCESSING, SKIP_CHARGE_READY = get_preprocessing_and_skip_charge_ready(
    learning_case
)
dim_dict = get_input_dimensions(BATTERY_TYPE)
AUC_ANALYSIS = True
analyse_ae_output = False

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
    plot_loss_curve_all(args)
except Exception as e:
    print(f"Error plotting loss curve: {e}")

## =================== Load test data lists =================== ##
normal_list = (
    np.load(f"./{BATTERY_TYPE}_filtered_vehicle_ids_test.npy").astype(np.int64).tolist()
)
fault_list = (
    np.load(f"./{BATTERY_TYPE}_filtered_vehicle_ids_fault.npy")
    .astype(np.int64)
    .tolist()
)
test_list = [normal_list, fault_list]

## =================== Initialize params for AUROC curve  =================== ##
total_test_vehicles = len(normal_list) + len(fault_list)
predict_threshold_array = np.arange(0, 1000, 0.1)
predict_thresholds = np.asarray(list(predict_threshold_array), dtype=float)

# Store results for all model combinations
all_roc_results = []

# Pre-load all models to avoid repeated loading
print("\nPre-loading all models...")
models_u = {}
models_x = {}

for u_idx in u_model_indices:
    u_model_idx = "net" if u_idx == -1 else f"net_e{u_idx}"
    net_loaded = CombinedAE(
        input_size=dim_dict["x"],
        encode2_input_size=dim_dict["q"],
        output_size=dim_dict["y"],
        activation_fn=(
            CustomSigmoidFunc(scale=ae_u_scale, shift=ae_u_shift)
            if PREPROCESSING == False
            else torch.sigmoid
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
    net_loaded.double().eval()
    models_u[u_idx] = net_loaded
    del net_state_dict  # Free memory
    print(f"  Loaded U model: {u_model_idx}")

for x_idx in x_model_indices:
    x_model_idx = "netx" if x_idx == -1 else f"netx_e{x_idx}"
    netx_loaded = CombinedAE(
        input_size=dim_dict["x2"],
        encode2_input_size=dim_dict["q2"],
        output_size=dim_dict["y2"],
        activation_fn=CustomSigmoidFunc(scale=ae_x_scale, shift=ae_x_shift),
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
    netx_loaded.double().eval()
    models_x[x_idx] = netx_loaded
    del netx_state_dict  # Free memory
    print(f"  Loaded X model: {x_model_idx}")

print(
    f"All models loaded successfully. Total: {len(models_u)} U models, {len(models_x)} X models\n"
)

# Pre-load training data once to avoid repeated loading in train_pca_only
print("Pre-loading training data...")
train_list = (
    np.load(f"./{BATTERY_TYPE}_filtered_vehicle_ids_train.npy")
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

combined_tensor_list = []
combined_tensorx_list = []

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
        normalize_dx=(vals.normalize_dx if hasattr(vals, "normalize_dx") else False),
    )

    combined_tensor_list.append(tensor)
    combined_tensorx_list.append(tensorx)

if len(combined_tensor_list) > 0:
    combined_tensor_train = torch.cat(combined_tensor_list, dim=0)
    combined_tensorx_train = torch.cat(combined_tensorx_list, dim=0)
    del combined_tensor_list, combined_tensorx_list
else:
    combined_tensor_train = torch.empty(0)
    combined_tensorx_train = torch.empty(0)

print(f"Training data loaded: {combined_tensor_train.shape[0]} samples\n")

# Pre-load all test data once to avoid repeated loading
print("Pre-loading test data...")
test_data_cache = {}  # Dictionary to store {vehicle_id: (tensor, tensor_x)}

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
            normalize_dx=(
                vals.normalize_dx if hasattr(vals, "normalize_dx") else False
            ),
        )

        test_data_cache[i] = (tensor, tensor_x)

print(f"Test data loaded: {len(test_data_cache)} vehicles\n")

# Iterate over all u_model and x_model combinations
for u_idx in u_model_indices:
    for x_idx in x_model_indices:
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
        if u_idx == -1:
            loads = load_pca_results(
                f"{args.models_dir}/{args.models_idx}",
                load_data_nor=True,
                validate_shapes=False,
            )
        else:
            # Use pre-loaded models and training data
            net_for_pca = models_u[u_idx]
            netx_for_pca = models_x[x_idx]

            # Pass pre-loaded training data to avoid reloading
            loads = train_pca_only(
                vals,
                device,
                save=False,
                net=net_for_pca,
                netx=netx_for_pca,
                preloaded_train_data=(combined_tensor_train, combined_tensorx_train),
            )

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

        # Explicitly delete large unused arrays (X and data_nor)
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
                y_recovered2 = tensor_x[
                    :, dim_dict["x2"] : dim_dict["x2"] + dim_dict["y2"]
                ]
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

                with torch.inference_mode():
                    recon_imtest = net_loaded(
                        x_recovered, z_recovered, q_recovered, y_recovered
                    )
                    reconx_imtest = netx_loaded(
                        x_recovered2, z_recovered2, q_recovered2, y_recovered2
                    )

                    # Compute errors on GPU and move to CPU once
                    ERRORU = torch.abs(recon_imtest[0] - y_recovered).cpu().numpy()
                    ERRORX = torch.abs(reconx_imtest[0] - y_recovered2).cpu().numpy()

                    # Free GPU tensors immediately
                    del recon_imtest, reconx_imtest

                df_data, df_data2 = DiagnosisFeature(
                    ERRORU, ERRORX, get_true_feature=analyse_ae_output
                )

                ## =================== Testing Diagnosis =================== ##
                t2_array, _ = T2_array(df_data, data_mean, data_std, p_k, v_I)
                spe_array, _ = SPE_array(df_data, data_mean, data_std, p_k)
                CI_array = (spe_array / SPE_95_limit) + (t2_array / T_95_limit)

                if AUC_ANALYSIS:
                    ci_max = float(np.nanmax(np.asarray(CI_array, dtype=float)))
                    predict_results[test_idx, :] = (ci_max > predict_thresholds).astype(
                        np.int8
                    )
                    y_true[test_idx] = np.int8(label)
                    test_idx += 1

                # Free memory after processing each vehicle
                del x_recovered, y_recovered, z_recovered, q_recovered
                del x_recovered2, y_recovered2, z_recovered2, q_recovered2
                del ERRORU, ERRORX, df_data, df_data2
                del t2_array, spe_array, CI_array

        elapsed = time.perf_counter() - start
        h, rem = divmod(elapsed, 3600)
        m, s = divmod(rem, 60)
        print(
            f"\nCombination (u={u_idx}, x={x_idx}) completed: {int(h)}h {int(m)}m {s:.3f}s\n"
        )

        # Clear CUDA cache between combinations
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        if AUC_ANALYSIS:
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
            print(
                f"Best threshold (Youden J) ≈ {opt['best']['threshold']:.6g} @ (FPR={opt['best']['fpr']:.4f}, TPR={opt['best']['tpr']:.4f})"
            )
            print(
                f"Closest to (0,1) ≈ {opt['closest_to_01']['threshold']:.6g} @ (FPR={opt['closest_to_01']['fpr']:.4f}, TPR={opt['closest_to_01']['tpr']:.4f}), dist={opt['closest_to_01']['distance']:.4f}"
            )

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

# =================== Plot all ROC curves on one figure =================== ##
if AUC_ANALYSIS and len(all_roc_results) > 1:
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

print("\n✓ All model combinations tested successfully!")
