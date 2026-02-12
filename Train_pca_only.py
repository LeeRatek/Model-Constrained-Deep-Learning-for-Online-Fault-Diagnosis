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


def train_pca_only(args, device, save=True, net=None, netx=None, preloaded_train_data=None):
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
                else torch.sigmoid
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
            activation_fn=CustomSigmoidFunc(
                scale=args.ae_x_scale, shift=args.ae_x_shift
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
    # DataLoader 오버헤드 제거, 직접 배치 처리
    INFERENCE_BATCH_SIZE = 4096  # 더 큰 배치 사용

    # y는 추론에 불필요하므로 CPU에 유지
    x_recovered = x_recovered.double()
    z_recovered = z_recovered.double()
    q_recovered = q_recovered.double()

    num_samples = x_recovered.shape[0]
    recon_imtest_list = []

    with torch.inference_mode():
        for start_idx in range(0, num_samples, INFERENCE_BATCH_SIZE):
            end_idx = min(start_idx + INFERENCE_BATCH_SIZE, num_samples)

            x_batch = x_recovered[start_idx:end_idx].to(device, non_blocking=True)
            z_batch = z_recovered[start_idx:end_idx].to(device, non_blocking=True)
            q_batch = q_recovered[start_idx:end_idx].to(device, non_blocking=True)

            recon_imtest, _ = net(x_batch, z_batch, q_batch)
            recon_imtest_list.append(recon_imtest.cpu().numpy())

            # GPU 메모리 즉시 해제
            del x_batch, z_batch, q_batch, recon_imtest

    AA = np.concatenate(recon_imtest_list, axis=0)
    del recon_imtest_list
    yTrainU = y_recovered.cpu().numpy()
    ERRORU = np.abs(AA - yTrainU) if use_abs_err else AA - yTrainU
    del AA, yTrainU

    ## ================= MC-AE inference for SOC reconstruction ================= ##
    x_recovered2 = x_recovered2.double()
    z_recovered2 = z_recovered2.double()
    q_recovered2 = q_recovered2.double()

    num_samples2 = x_recovered2.shape[0]
    recon_imtestx_list = []

    with torch.inference_mode():
        for start_idx in range(0, num_samples2, INFERENCE_BATCH_SIZE):
            end_idx = min(start_idx + INFERENCE_BATCH_SIZE, num_samples2)

            x_batch = x_recovered2[start_idx:end_idx].to(device, non_blocking=True)
            z_batch = z_recovered2[start_idx:end_idx].to(device, non_blocking=True)
            q_batch = q_recovered2[start_idx:end_idx].to(device, non_blocking=True)

            recon_imtestx, _ = netx(x_batch, z_batch, q_batch)
            recon_imtestx_list.append(recon_imtestx.cpu().numpy())

            # GPU 메모리 즉시 해제
            del x_batch, z_batch, q_batch, recon_imtestx

    BB = np.concatenate(recon_imtestx_list, axis=0)
    del recon_imtestx_list
    yTrainX = y_recovered2.cpu().numpy()
    ERRORX = np.abs(BB - yTrainX) if use_abs_err else BB - yTrainX
    del BB, yTrainX

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
