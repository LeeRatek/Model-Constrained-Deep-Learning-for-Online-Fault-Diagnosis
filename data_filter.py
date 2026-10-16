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
    warnings.filterwarnings("ignore")

    # --debug: 분할 비율과 이름을 덮어쓴다 (빠른 확인용)
    if args.debug:
        args.train_ratio = 0.02
        args.val_ratio = 0.5
        args.dataset_name = f"{int(args.train_ratio*100)}"
        print("현재 디버그 모드로 실행 중입니다.")

    dataset_dir = f"./datasets/{args.dataset_name}"
    os.makedirs(dataset_dir, exist_ok=True)
    print(f"Dataset will be saved to: {dataset_dir}")

    mode = MODE.FILTER_MODE
    BATTERY_TYPE = "QAS"  # 'DTI' or 'QAS'
    SKIP_CHARGE_READY = True
    PREPROCESSING = False
    dim_dict = get_input_dimensions(BATTERY_TYPE)
    x, y, z, q = dim_dict["x"], dim_dict["y"], dim_dict["z"], dim_dict["q"]  # vin_2 열 구간 폭
    x2, y2, z2, q2 = dim_dict["x2"], dim_dict["y2"], dim_dict["z2"], dim_dict["q2"]  # vin_3 열 구간 폭

    max_idx = 392 if BATTERY_TYPE == "QAS" else 79
    fault_idx = 335 if BATTERY_TYPE == "QAS" else 77

    np.random.seed(args.seed)

    kept_ids = []
    vehicle_data_counts = {}  # 차량 ID → 전처리 후 시점 수
    u_tensors, x_tensors = [], []  # ANALYSIS_MODE에서 이어 붙일 차량 데이터
    for i in range(max_idx + 1):
        print(f"============== Processing Vehicle ID: {i} ==============")
        tensor = safe_load(f"./data/{BATTERY_TYPE}/{i}/vin_2.pkl")
        tensorx = safe_load(f"./data/{BATTERY_TYPE}/{i}/vin_3.pkl")
        tensor, tensorx = preprocess_loaded_tensor(
            tensor, tensorx, dim_dict, BATTERY_TYPE, PREPROCESSING, SKIP_CHARGE_READY
        )
        if is_abnormal(tensor):
            print(f"Vehicle ID {i} is filtered out due to abnormal data.")
            continue
        kept_ids.append(i)
        vehicle_data_counts[i] = tensor.shape[0]

        if mode == MODE.PLOT_MODE:
            # 열 구간마다 열 하나씩 그림 파일로 저장한다: [팩 추정값 2개 + 첫 셀], [편차 앞 3개], [q 열들]
            save_dir = f"./data/data_analysis_at_charge_0/{BATTERY_TYPE}-{i}"
            for data, name, columns in (
                (tensor, "vin2", (0, x + 2)),
                (tensor, "vin2", (x + y, x + y + 2)),
                (tensor, "vin2", (x + y + z, x + y + z + q - 1)),
                (tensorx, "vin3", (0, x2 + 2)),
                (tensorx, "vin3", (x2 + y2, x2 + y2 + 2)),
                (tensorx, "vin3", (x2 + y2 + z2, x2 + y2 + z2 + q2 - 1)),
            ):
                plot_testX_timeseries(data, title=name, save_path=f"{save_dir}/{name}", columns=columns)
        elif mode == MODE.ANALYSIS_MODE:
            u_tensors.append(tensor)
            x_tensors.append(tensorx)

    # 분할 파일에는 ID를 실수형(float64)으로 저장한다 (기존 datasets/의 분할 파일과 같은 형식)
    filtered_vehicle_ids = np.array(kept_ids, dtype=float)
    print(f"Filtered vehicle IDs: {filtered_vehicle_ids}")

    if mode == MODE.PLOT_MODE:
        return

    if mode == MODE.FILTER_MODE:
        fault_list = [i for i in filtered_vehicle_ids if i >= fault_idx]
        normal_list = [i for i in filtered_vehicle_ids if i < fault_idx]

        # 비율을 주지 않으면 고정 대수를 쓴다
        total_normal = len(normal_list)
        if args.train_ratio is not None:
            n_train_set = int(total_normal * args.train_ratio)
        else:
            n_train_set = 109 if BATTERY_TYPE == "QAS" else 39

        # val_ratio는 학습 분할을 뺀 나머지에 대한 비율이다
        total_remain = total_normal - n_train_set
        if args.val_ratio is not None:
            n_validation_set = int(total_remain * args.val_ratio)
        else:
            n_validation_set = 50 if BATTERY_TYPE == "QAS" else 19

        if n_train_set + n_validation_set > total_normal:
            print(
                f"Error: Requested train ({n_train_set}) + val ({n_validation_set}) > Total ({total_normal})"
            )
            exit(1)

        print(
            f"Dataset Split - Train: {n_train_set}, Val: {n_validation_set}, Total Normal: {total_normal}"
        )

        np.save(
            f"{dataset_dir}/{BATTERY_TYPE}_filtered_vehicle_ids_normal.npy",
            np.array(normal_list),
        )
        np.save(
            f"{dataset_dir}/{BATTERY_TYPE}_filtered_vehicle_ids_fault.npy",
            np.array(fault_list),
        )

        train_list = np.random.choice(normal_list, size=n_train_set, replace=False)
        train_list.sort()
        remain_list = [vid for vid in normal_list if vid not in set(train_list)]
        validate_list = np.random.choice(remain_list, size=n_validation_set, replace=False)
        validate_list.sort()
        test_list = [vid for vid in remain_list if vid not in set(validate_list)]
        test_list.sort()

        # 분할별 전체 시점 수
        train_data_count = sum(vehicle_data_counts[vid] for vid in train_list)
        val_data_count = sum(vehicle_data_counts[vid] for vid in validate_list)
        test_data_count = sum(vehicle_data_counts[vid] for vid in test_list)
        fault_data_count = sum(vehicle_data_counts[vid] for vid in fault_list)

        np.save(
            f"{dataset_dir}/{BATTERY_TYPE}_filtered_vehicle_ids_train.npy",
            np.array(train_list),
        )
        np.save(
            f"{dataset_dir}/{BATTERY_TYPE}_filtered_vehicle_ids_validate.npy",
            np.array(validate_list),
        )
        np.save(
            f"{dataset_dir}/{BATTERY_TYPE}_filtered_vehicle_ids_test.npy", np.array(test_list)
        )

        # 나중에 어떤 분할인지 확인할 수 있게 요약을 남긴다
        total_samples = train_data_count + val_data_count + test_data_count
        with open(f"{dataset_dir}/dataset_info_{BATTERY_TYPE}.txt", "w") as f:
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

        print(f"Dataset saved to {dataset_dir}")
        return

    # ANALYSIS_MODE: 남은 차량을 모두 이어 붙여 주요 열의 분포를 본다
    u_data = torch.cat(u_tensors).numpy()  # vin_2 (셀 전압 쪽)
    x_data = torch.cat(x_tensors).numpy()  # vin_3 (셀 SOC 쪽)
    q_start = x + y + z  # q 열(온도, SOC, 전류)의 시작
    dsoc_start = x2 + y2  # vin_3의 셀 SOC 편차 시작
    for values, title in (
        (u_data[:, 0], "upre1 Sample Distribution"),
        (u_data[:, 1], "upre2 Sample Distribution"),
        (u_data[:, q_start], "Temperature Sample Distribution"),
        (u_data[:, q_start + 1], "SOC Sample Distribution"),
        (u_data[:, q_start + 2], "Current Sample Distribution"),
        (u_data[:, x], "True Voltage 1 Sample Distribution"),
        (u_data[:, x + 1], "True Voltage 2 Sample Distribution"),
        (u_data[:, x + y], "Volatage divation 1 Sample Distribution"),
        (u_data[:, x + y + 1], "Volatage divation 2 Sample Distribution"),
        (x_data[:, dsoc_start], "SOC divation 1 Sample Distribution"),
        (x_data[:, dsoc_start + 1], "SOC divation 2 Sample Distribution"),
    ):
        plot_distribution_with_stats(values, title=title)

    print("Amount of data used for training:", u_data.shape[0])


if __name__ == "__main__":
    main()
