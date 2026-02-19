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
from monitering import *


def train_pca_only(
    args,
    device,
    save=True,
    net=None,
    netx=None,
    preloaded_train_data=None,
    inference_batch_size=None,
):
    from torch.utils.data import Dataset

    if save:
        data_path = make_model_path_based_timestamp(base=args.models_dir)
    use_dx_in_forward = not args.no_use_dx
    use_abs_err = not args.no_abs_err
    add_one_output_layer = args.add_one_output_layer

    BATTERY_TYPE = args.battery_type  # 'DTI' or 'QAS'
    PREPROCESSING, SKIP_CHARGE_READY = get_preprocessing_and_skip_charge_ready(
        args.learning_case
    )
    # SKIP_CHARGE_READY = False
    dim_dict = get_input_dimensions(BATTERY_TYPE)

    ## =================== Load training data  =================== ##
    if net is None:
        print("Loading trained umodel...")
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
        u_model_idx = "net" if args.u_model_idx == -1 else f"net_e{args.u_model_idx}"
        net_state_dict = torch.load(
            os.path.join(
                f"{args.models_dir}/{args.models_folder}/artifact/",
                f"{u_model_idx}.pth",
            ),
            map_location=device,
        )
        net.load_state_dict(net_state_dict)

    if netx is None:
        print("Loading trained x model...")
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
        x_model_idx = "netx" if args.x_model_idx == -1 else f"netx_e{args.x_model_idx}"
        netx_state_dict = torch.load(
            os.path.join(
                f"{args.models_dir}/{args.models_folder}/artifact/",
                f"{x_model_idx}.pth",
            ),
            map_location=device,
        )
        netx.load_state_dict(netx_state_dict)

    # 모델을 미리 eval/double 모드로 전환
    net.double().eval()
    netx.double().eval()

    # Use preloaded training data if provided, otherwise load from disk
    if preloaded_train_data is not None:
        print("Using pre-loaded training data")
        combined_tensor, combined_tensorx = preloaded_train_data
    else:
        print("Loading training data from disk...")
        train_list = (
            np.load(f"./{BATTERY_TYPE}_filtered_vehicle_ids_train.npy")
            .astype(np.int64)
            .tolist()
        )

        # ----------------------------------------Hyper Parameter Setting---------------------

        train_list = (
            train_list[args.vehicle_start : args.vehicle_end + 1]
            if args.vehicle_end != -1
            else train_list[args.vehicle_start :]
        )

        # Optimization 1: Use list to collect tensors instead of repeated concatenation
        combined_tensor_list = []
        combined_tensorx_list = []

        count = 0
        for i in train_list:
            count += 1
            if count % 10 == 0:
                print(f"Processing vehicle {count}/{len(train_list)}")
            VEHICLE_ID = f"{i}"

            # ----------------------------------------Data loading for MC-AE (customized) ------------------------------
            tensor = safe_load(
                f"{args.source_data_dir}/{BATTERY_TYPE}/{VEHICLE_ID}/vin_2.pkl",
                verbose=False,
            )
            tensorx = safe_load(
                f"{args.source_data_dir}/{BATTERY_TYPE}/{VEHICLE_ID}/vin_3.pkl",
                verbose=False,
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

            combined_tensor_list.append(tensor)
            combined_tensorx_list.append(tensorx)

        # Batch concatenation is much faster
        if len(combined_tensor_list) > 0:
            combined_tensor = torch.cat(combined_tensor_list, dim=0)
            combined_tensorx = torch.cat(combined_tensorx_list, dim=0)
            # 메모리 해제
            del combined_tensor_list, combined_tensorx_list
        else:
            # Handle empty case if needed
            combined_tensor = torch.empty(0)
            combined_tensorx = torch.empty(0)

    # ----------------------------------------Training for MC-AE--------------------------
    x_recovered = combined_tensor[:, : dim_dict["x"]]  # 0~1
    y_recovered = combined_tensor[
        :, dim_dict["x"] : dim_dict["x"] + dim_dict["y"]
    ]  # 2~111
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
        dim_dict["x2"]
        + dim_dict["y2"] : dim_dict["x2"]
        + dim_dict["y2"]
        + dim_dict["z2"],
    ]
    q_recovered2 = combined_tensorx[
        :, dim_dict["x2"] + dim_dict["y2"] + dim_dict["z2"] :
    ]

    ## ================= MC-AE inference for voltage reconstruction ================= ##
    # Optimize for both CPU and GPU environments
    # Use provided batch size or default for large datasets
    if inference_batch_size is None:
        INFERENCE_BATCH_SIZE = (
            65536  # Default: optimized for 50GB RAM with 6.3M samples
        )
    else:
        INFERENCE_BATCH_SIZE = inference_batch_size

    # For CPU: Keep data on CPU and use larger batches
    # For GPU: Move to GPU once to avoid repeated transfers
    is_gpu = device.type == "cuda"

    if is_gpu:
        # GPU: Move all data to GPU once
        x_recovered = x_recovered.double().to(device, non_blocking=True)
        z_recovered = z_recovered.double().to(device, non_blocking=True)
        q_recovered = q_recovered.double().to(device, non_blocking=True)
        y_recovered_device = y_recovered.double().to(device, non_blocking=True)
    else:
        # CPU: Convert to double in-place, stay on CPU
        x_recovered = x_recovered.double()
        z_recovered = z_recovered.double()
        q_recovered = q_recovered.double()
        y_recovered_device = y_recovered.double()

    num_samples = x_recovered.shape[0]
    ERRORU_list = []

    with torch.inference_mode():
        for start_idx in range(0, num_samples, INFERENCE_BATCH_SIZE):
            end_idx = min(start_idx + INFERENCE_BATCH_SIZE, num_samples)

            x_batch = x_recovered[start_idx:end_idx]
            z_batch = z_recovered[start_idx:end_idx]
            q_batch = q_recovered[start_idx:end_idx]
            y_batch = y_recovered_device[start_idx:end_idx]

            recon_imtest, _ = net(x_batch, z_batch, q_batch)

            # Compute error directly (on GPU or CPU)
            if use_abs_err:
                error_batch = torch.abs(recon_imtest - y_batch)
            else:
                error_batch = recon_imtest - y_batch

            # For CPU: already on CPU, for GPU: transfer back
            ERRORU_list.append(
                error_batch.cpu().numpy() if is_gpu else error_batch.numpy()
            )

            del recon_imtest, error_batch

    # Free memory
    del x_recovered, z_recovered, q_recovered, y_recovered, y_recovered_device

    ERRORU = np.concatenate(ERRORU_list, axis=0)
    del ERRORU_list

    ## ================= MC-AE inference for SOC reconstruction ================= ##
    # Same CPU/GPU optimization
    if is_gpu:
        x_recovered2 = x_recovered2.double().to(device, non_blocking=True)
        z_recovered2 = z_recovered2.double().to(device, non_blocking=True)
        q_recovered2 = q_recovered2.double().to(device, non_blocking=True)
        y_recovered2_device = y_recovered2.double().to(device, non_blocking=True)
    else:
        x_recovered2 = x_recovered2.double()
        z_recovered2 = z_recovered2.double()
        q_recovered2 = q_recovered2.double()
        y_recovered2_device = y_recovered2.double()

    num_samples2 = x_recovered2.shape[0]
    ERRORX_list = []

    with torch.inference_mode():
        for start_idx in range(0, num_samples2, INFERENCE_BATCH_SIZE):
            end_idx = min(start_idx + INFERENCE_BATCH_SIZE, num_samples2)

            x_batch = x_recovered2[start_idx:end_idx]
            z_batch = z_recovered2[start_idx:end_idx]
            q_batch = q_recovered2[start_idx:end_idx]
            y_batch = y_recovered2_device[start_idx:end_idx]

            recon_imtestx, _ = netx(x_batch, z_batch, q_batch)

            # Compute error
            if use_abs_err:
                error_batch = torch.abs(recon_imtestx - y_batch)
            else:
                error_batch = recon_imtestx - y_batch

            ERRORX_list.append(
                error_batch.cpu().numpy() if is_gpu else error_batch.numpy()
            )

            del recon_imtestx, error_batch

    # Free memory
    del x_recovered2, z_recovered2, q_recovered2, y_recovered2, y_recovered2_device

    ERRORX = np.concatenate(ERRORX_list, axis=0)
    del ERRORX_list

    df_data, _ = DiagnosisFeature(ERRORU, ERRORX)
    results = Custom_PCA(df_data, 0.99, 0.99)

    if save:
        save_pca_results(f"{data_path}/", results)
        return
    else:
        return results


if __name__ == "__main__":
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
    print(torch.cuda.device_count())

    warnings.filterwarnings("ignore")
    # VIN_data = pd.read_excel(r'{args.source_data_dir}/Name_list.xls')

    parser = argparse.ArgumentParser(
        description="Run diagnostics plotting with CLI options"
    )
    parser.add_argument(
        "--battery-type",
        choices=["QAS", "DTI"],
        default="QAS",
        help="배터리 유형 선택 (DTI fault index range = [77~79], QAS fault index range = [335~392])",
    )
    parser.add_argument(
        "--models-folder", type=str, default="260203_163103"
    )  # 260123_082222 & 260126_100030
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
    args = parser.parse_args()
    train_pca_only(args)
