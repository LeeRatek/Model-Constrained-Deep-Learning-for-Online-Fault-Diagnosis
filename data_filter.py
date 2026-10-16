"""데이터 분할 스크립트: 원본 차량 데이터에서 이상한 차량을 빼고, 학습·검증·테스트·고장 차량 목록을 만든다.

학습(Train_.py)과 평가(Test_ 계열)를 하기 전에 한 번 실행한다. 만든 목록은 datasets/<이름>/ 폴더에
<BT>_filtered_vehicle_ids_<분할>.npy 파일로 저장되고(<BT>는 배터리 종류 QAS 또는 DTI),
Train_.py --dataset-name <이름>으로 그 분할을 고른다.

실행 예
    python data_filter.py --dataset-name 10 --train-ratio 0.1 --val-ratio 0.5 --seed 42
    옵션과 코드 안에 고정된 값(BATTERY_TYPE, max_idx, fault_idx, mode)은 README 7.1절에 정리되어 있다.

하는 일 (mode가 기본값 FILTER_MODE일 때)
    1. 차량 0 ~ max_idx의 원본(data/<BT>/<ID>/vin_2.pkl, vin_3.pkl)을 읽어 전처리한다.
    2. 값이 비정상인 차량을 뺀다(mcdl/preprocessing.py의 is_abnormal).
    3. 차량 ID가 fault_idx 이상이면 고장, 미만이면 정상 차량으로 나눈다.
    4. 정상 차량을 시드로 고정된 무작위 순서로 train / validate / test로 나눠 .npy로 저장하고,
       분할 요약을 dataset_info_<BT>.txt에 남긴다.
"""

import numpy as np
import os
import warnings
from mcdl.preprocessing import get_input_dimensions, is_abnormal, preprocess_loaded_tensor, safe_load
from mcdl.plotting import plot_distribution_with_stats, plot_testX_timeseries
import torch
import argparse
from enum import IntEnum

# 명령줄 옵션 정의. 실제 읽기(parse_args)는 main 안에서 한다 (import만 할 때는 아무것도 실행되지 않게).
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
parser.add_argument(
    "--debug",
    action="store_true",
    help='빠른 확인용: train_ratio 0.02, val_ratio 0.5, dataset_name "2"로 덮어씀',
)


class MODE(IntEnum):
    """main이 무엇을 할지 고르는 값. main 안의 `mode = ...` 줄을 고쳐서 바꾼다(명령줄 옵션은 없다).

    PLOT_MODE: 차량마다 주요 열의 시계열 그림을 data/data_analysis_at_charge_0/ 아래에 저장하고 끝낸다.
    FILTER_MODE: 분할 파일(.npy)과 요약(dataset_info_<BT>.txt)을 저장한다 (기본값).
    ANALYSIS_MODE: 남은 차량 데이터를 모두 이어 붙여, 주요 값의 분포 그림을 화면에 띄운다.
    """

    PLOT_MODE = 1
    FILTER_MODE = 2
    ANALYSIS_MODE = 3


def main(argv=None):
    """명령줄 옵션을 읽어 mode에 맞는 일을 한다. 기본(FILTER_MODE) 순서는 이 파일 맨 위 설명의 "하는 일"과 같다.

    argv를 주면 명령줄 대신 그 목록을 옵션으로 읽는다. 쓰는 곳: 이 파일 맨 아래 실행부.
    """
    args = parser.parse_args(argv)

    os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
    print(torch.cuda.device_count())

    warnings.filterwarnings("ignore")

    # --debug: 분할 비율과 이름을 덮어쓴다 (빠른 확인용)
    if args.debug:
        args.train_ratio = 0.02
        args.val_ratio = 0.5
        args.dataset_name = f"{int(args.train_ratio*100)}"
        print("현재 디버그 모드로 실행 중입니다.")

    # Create dataset directory
    dataset_dir = f"./datasets/{args.dataset_name}"
    os.makedirs(dataset_dir, exist_ok=True)
    print(f"Dataset will be saved to: {dataset_dir}")


    mode = MODE.FILTER_MODE
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
    vehicle_data_counts = {}  # Store sample count for each vehicle
    (vin1_start, vin2_start, vin3_start) = (0, 0, 0)

    start_idxes = []
    for i in range(0, max_idx + 1):
        VEHICLE_ID = f"{i}"
        print(f"============== Processing Vehicle ID: {VEHICLE_ID} ==============")
        PATH = f"./data/data_analysis_at_charge_{vin2_start}/{BATTERY_TYPE}-{VEHICLE_ID}"

        # ----------------------------------------Data loading for MC-AE (customized) ------------------------------
        tensor = safe_load(f"./data/{BATTERY_TYPE}/{VEHICLE_ID}/vin_2.pkl", verbose=False)
        tensorx = safe_load(f"./data/{BATTERY_TYPE}/{VEHICLE_ID}/vin_3.pkl", verbose=False)

        tensor, tensorx = preprocess_loaded_tensor(
            tensor, tensorx, dim_dict, BATTERY_TYPE, PREPROCESSING, SKIP_CHARGE_READY
        )
        if is_abnormal(tensor):
            print(f"Vehicle ID {i} is filtered out due to abnormal data.")
            continue
        filtered_vehicle_ids = np.append(filtered_vehicle_ids, i)  # TBD
        vehicle_data_counts[i] = tensor.shape[0]  # Store data count

        # ---------------------------------------- Data normalization ------------------------------
        if mode == MODE.PLOT_MODE:
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

        # Calculate total data points for each split
        train_data_count = sum(vehicle_data_counts[vid] for vid in train_list)
        val_data_count = sum(vehicle_data_counts[vid] for vid in validate_list)
        test_data_count = sum(vehicle_data_counts[vid] for vid in test_list)
        fault_data_count = sum(vehicle_data_counts[vid] for vid in fault_list)

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
        total_samples = train_data_count + val_data_count + test_data_count
        with open(f"{save_path}/dataset_info_{BATTERY_TYPE}.txt", "w") as f:
            f.write(f"Dataset Name: {args.dataset_name}\n")
            f.write(f"Battery Type: {BATTERY_TYPE}\n")
            f.write(f"Seed: {args.seed}\n")
            f.write(
                f"Train Count: {len(train_list)} ({len(train_list)/total_normal:.2%}), Samples: {train_data_count} ({train_data_count/total_samples:.2%})\n"
            )
            f.write(
                f"Val Count: {len(validate_list)} ({len(validate_list)/total_normal:.2%}), Samples: {val_data_count} ({val_data_count/total_samples:.2%})\n"
            )
            f.write(
                f"Test Count: {len(test_list)} ({len(test_list)/total_normal:.2%}), Samples: {test_data_count} ({test_data_count/total_samples:.2%})\n"
            )
            f.write(f"Fault Count: {len(fault_list)}, Samples: {fault_data_count}\n")

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


if __name__ == "__main__":
    main()
