import numpy as np
import os
import warnings
import torch
import torch.nn as nn
from torch.utils.data import Dataset
from torch.utils.data import BatchSampler, DataLoader, RandomSampler, SequentialSampler
import time
import argparse
import random
from mcdl.run import (
    RunConfig,
    build_model_pair,
    checkpoint_filename,
    make_model_path_based_timestamp,
    save_net_state,
    write_sim_config,
)
from mcdl.diagnosis import ControlChart
from mcdl.reconstruction import training_features
from mcdl.vehicles import load_vehicle, split_ids


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

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
parser.add_argument("--normalize-dx", action="store_true")
parser.add_argument("--normalize-val", type=int, default=4)
parser.add_argument(
    "--learning-case",
    type=int,
    default=1,
    help="1: Skip and no nomalization, 2: Skip but doing normalization, 3: No skip but doing normalization, 4: No skip and no normalization.",
)
parser.add_argument(
    "--dataset-name",
    type=str,
    default=None,
    help="Name of the dataset folder (e.g. 'train_0.5_seed42') to load from ./datasets/",
)
parser.add_argument("--no-use-dx", action="store_true")
parser.add_argument("--no-abs-err", action="store_true")
parser.add_argument("--add-one-output-layer", action="store_true")
parser.add_argument("--no-save", action="store_true")
parser.add_argument(
    "--no-validation", action="store_true", help="Skip validation process"
)
parser.add_argument(
    "--seed", type=int, default=42, help="Random seed for reproducibility"
)
parser.add_argument(
    "--debug",
    action="store_true",
    help="빠른 확인용: 학습·검증 차량 1대씩, use_dx 끔, sim_config.txt 생략",
)


# ----------------------------------------Training for MC-AE--------------------------
class VehicleRows(Dataset):
    """여러 차량 텐서를 이어 붙이지 않고(메모리 피크 방지) 배치 하나를 (x, y, Δ, q) double로 꺼낸다.

    인덱스는 시점 번호의 목록(배치)이다 (row_loader의 BatchSampler가 넘긴다). 한 차량 안의 연속 구간이면
    slice로, 아니면 시점을 하나씩 모아 쌓는다. 어느 쪽이든 시점마다 꺼내 collate로 쌓던 예전 방식과
    값·stride가 같다. 열 구성 [x | y | Δ | q]의 x·y 크기는 호출자가 모델 크기에서 읽어 넘긴다
    (reconstruct와 같은 규칙).
    """

    def __init__(self, tensors, x_dim, y_dim):
        self.tensors = tensors
        self.cumulative = np.cumsum([0] + [t.shape[0] for t in tensors])
        self.x_dim = x_dim
        self.y_dim = y_dim

    def __len__(self):
        return self.cumulative[-1]

    def __getitem__(self, indices):
        # 이진 탐색으로 각 시점이 어느 차량의 몇 번째 시점인지 찾는다
        idx = np.asarray(indices)
        vehicle = np.searchsorted(self.cumulative, idx, side="right") - 1
        local = idx - self.cumulative[vehicle]
        if vehicle[0] == vehicle[-1] and np.all(np.diff(local) == 1):
            rows = self.tensors[vehicle[0]][local[0] : local[-1] + 1]  # 한 차량 안의 연속 구간
        else:
            rows = torch.stack([self.tensors[v][i] for v, i in zip(vehicle, local)])
        rows = rows.to(torch.double)

        # 열 구간마다 표준 stride의 새 텐서로 만든다 (예전 collate 결과와 stride가 같아야 같은 연산 경로를 탄다).
        # .contiguous()는 배치가 1행이면 복사하지 않아 원래 행 stride가 남으므로 clone을 쓴다.
        y_start = self.x_dim
        dx_start = y_start + self.y_dim
        q_start = dx_start + self.y_dim
        return tuple(
            rows[:, a:b].clone(memory_format=torch.contiguous_format)
            for a, b in ((0, y_start), (y_start, dx_start), (dx_start, q_start), (q_start, None))
        )


def row_loader(tensors, x_dim, y_dim, *, batch_size, shuffle):
    """VehicleRows를 배치 단위로 읽는 DataLoader.

    sampler 구성이 DataLoader(shuffle=…, batch_size=…)가 안에서 만드는 것과 같아서, 배치 순서와
    순회할 때의 전역 RNG 소비가 같다.
    """
    rows = VehicleRows(tensors, x_dim, y_dim)
    sampler = RandomSampler(rows) if shuffle else SequentialSampler(rows)
    return DataLoader(rows, sampler=BatchSampler(sampler, batch_size, drop_last=False), batch_size=None)


LOSS_CSV = {"U": "loss_net", "X": "loss_netx"}


def train_autoencoder(
    net, model, train_tensors, val_tensors, *, epochs, save_per, lr, batch_size, shuffle, device, run_dir, save
):
    """U-model 또는 X-model(model = "U"/"X") 하나를 학습한다.

    epoch % save_per == 0이면 검증 loss를 재고 체크포인트를 저장한다. 끝나면 loss CSV와 최종 모델(-1)을 저장한다.
    val_tensors가 None이면 검증을 건너뛴다 (--no-validation).
    전역 RNG는 DataLoader를 순회할 때마다 쓰인다 (shuffle을 꺼도). 그래서 학습 결과는 이 함수의 순회 순서
    (epoch마다 학습 → save_per마다 검증)와 main의 호출 순서(U → X)에 달려 있다.
    """
    optimizer = torch.optim.Adam(net.parameters(), lr=lr)
    loss_f = nn.MSELoss()
    avg_loss_list = []
    val_loss_points = []  # list of (epoch, avg_val_loss)

    # double casting은 반복 호출할 필요가 없어 루프 밖에서 1회만 수행
    net = net.double()

    x_dim, y_dim = net.fc1.in_features, net.fc3.out_features
    # Shuffle is important for training
    train_loader = row_loader(train_tensors, x_dim, y_dim, batch_size=batch_size, shuffle=shuffle)
    if val_tensors is None:
        validate_loader = []
    else:
        validate_loader = row_loader(val_tensors, x_dim, y_dim, batch_size=batch_size, shuffle=False)

    for epoch in range(epochs):
        total_loss = 0
        num_batches = 0
        for x, y, z, q in train_loader:
            x = x.to(device)
            y = y.to(device)
            z = z.to(device)
            q = q.to(device)
            net.train()
            recon_im, _ = net(x, z, q)
            loss = loss_f(y, recon_im)
            total_loss += loss.item()
            num_batches += 1
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        avg_loss = total_loss / num_batches
        avg_loss_list.append(avg_loss)
        if epoch % save_per == 0:
            # Validation (fast: no grad)
            net.eval()
            v_total = 0.0
            v_batches = 0
            with torch.inference_mode():
                for vx, vy, vz, vq in validate_loader:
                    vx = vx.to(device)
                    vy = vy.to(device)
                    vz = vz.to(device)
                    vq = vq.to(device)
                    v_recon_im, _ = net(vx, vz, vq)
                    v_loss = loss_f(vy, v_recon_im)
                    v_total += float(v_loss.item())
                    v_batches += 1
            avg_val_loss = v_total / max(v_batches, 1)
            val_loss_points.append((int(epoch), float(avg_val_loss)))

            print(
                "Epoch: {:2d} | Train Loss: {:.4f} | Val Loss: {:.4f}".format(
                    epoch, avg_loss, avg_val_loss
                )
            )
            # checkpoint (indexed)
            if save:
                save_net_state(
                    model=net, models_dir=f"{run_dir}/artifact/", filename=checkpoint_filename(model, epoch)
                )

    # Save loss log (fast: dump once per model)
    np.savetxt(
        os.path.join(f"{run_dir}/", f"{LOSS_CSV[model]}.csv"),
        np.column_stack(
            (
                np.arange(len(avg_loss_list), dtype=np.int64),
                np.asarray(avg_loss_list, dtype=np.float64),
            )
        ),
        delimiter=",",
        fmt=["%.0f", "%.4f"],
        header="epoch,avg_loss",
        comments="",
    )

    # validation loss points (epoch,avg_loss)
    if len(val_loss_points) > 0:
        np.savetxt(
            os.path.join(f"{run_dir}/", f"{LOSS_CSV[model]}_val.csv"),
            np.asarray(val_loss_points, dtype=np.float64),
            delimiter=",",
            fmt=["%.0f", "%.4f"],
            header="epoch,avg_loss",
            comments="",
        )

    if save:
        save_net_state(model=net, models_dir=f"{run_dir}/artifact/", filename=checkpoint_filename(model, -1))


def main(argv=None):
    os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
    print(torch.cuda.device_count())

    warnings.filterwarnings("ignore")

    args = parser.parse_args(argv)

    set_seed(args.seed)

    BATTERY_TYPE = args.battery_type  # 'DTI' or 'QAS'
    use_dx_in_forward = not args.no_use_dx
    use_abs_err = not args.no_abs_err

    # --debug: 학습·검증 차량 1대씩, use_dx 끔, sim_config.txt 생략 (빠른 확인용)
    is_debug = args.debug

    if is_debug:
        use_dx_in_forward = False
        print("현재 디버그 모드로 실행 중입니다.")

    # 데이터 로딩과 모델 생성에 함께 쓰는 Run 설정 (디버그 모드에서는 use_dx_in_forward가 False로 강제되어 있다)
    run_config = RunConfig.from_mapping({**vars(args), "no_use_dx": not use_dx_in_forward})
    PREPROCESSING, SKIP_CHARGE_READY = run_config.preprocessing, run_config.skip_charge_ready

    # 모델 저장 경로 설정

    data_path = make_model_path_based_timestamp(base=args.models_dir)

    start = time.perf_counter()

    if args.dataset_name:
        dataset_base_path = f"./datasets/{args.dataset_name}"
        print(f"Dataset path: {dataset_base_path}")
    else:
        print("Dataset path: . (Legacy)")
        dataset_base_path = "."

    train_list = split_ids(
        dataset_base_path, BATTERY_TYPE, "train", start=args.vehicle_start, end=args.vehicle_end
    )
    if not args.no_validation:
        validate_list = split_ids(dataset_base_path, BATTERY_TYPE, "validate")
    else:
        print("Validation skipped by user request.")
        validate_list = []

    vehicle_idxes = []

    vehicle_tensors = []
    vehicle_tensorsx = []

    count = 0
    for i in train_list:
        count += 1
        if count % 10 == 0:
            print(f"Processing vehicle {count}/{len(train_list)}")
        VEHICLE_ID = f"{i}"
        vehicle_idxes.append(VEHICLE_ID)
        # ----------------------------------------Data loading and preprocessing for MC-AE ------------------------------
        tensor, tensorx = load_vehicle(args.source_data_dir, run_config, VEHICLE_ID)

        vehicle_tensors.append(tensor)
        vehicle_tensorsx.append(tensorx)
        if is_debug:
            break

    # combined_tensor를 생성하지 않고 리스트만 유지하여 메모리 피크 방지


    # Total training samples calculation
    total_train_samples = sum(len(t) for t in vehicle_tensors)


    v_vehicle_tensors = []
    v_vehicle_tensorsx = []

    if args.no_validation:
        print("Skipping validation.")
    else:
        # ----------------------------------------Validation Data Loading ------------------------------
        count = 0

        for i in validate_list:
            count += 1
            if count % 10 == 0:
                print(f"Processing validation vehicle {count}/{len(validate_list)}")
            v_tensor, v_tensorx = load_vehicle(args.source_data_dir, run_config, i)

            v_vehicle_tensors.append(v_tensor)
            v_vehicle_tensorsx.append(v_tensorx)
            if is_debug:
                break

    if not is_debug:
        text = write_sim_config(
            data_path,
            args,
            exclude=("debug",),
            extra={
                "device": str(device),
                "Vehicle IDs used for training": vehicle_idxes,
                "Amount of data used for training": total_train_samples,
                "PREPROCESSING": PREPROCESSING,
                "SKIP_CHARGE_READY": SKIP_CHARGE_READY,
            },
        )
        print(text, end="")  # 콘솔 출력

    # Instantiate the networks
    net, netx = build_model_pair(run_config)
    net, netx = net.to(device), netx.to(device)

    ## ================= Training MC-AE for voltage reconstruction ================= ##
    train_autoencoder(
        net,
        "U",
        vehicle_tensors,
        None if args.no_validation else v_vehicle_tensors,
        epochs=args.ae_u_epochs,
        save_per=args.ae_u_save_per,
        lr=args.ae_u_lr,
        batch_size=args.ae_u_batchsize,
        shuffle=args.ae_u_shuffle,
        device=device,
        run_dir=data_path,
        save=not args.no_save,
    )

    ## ================= Training MC-AE for SOC reconstruction ================= ##
    train_autoencoder(
        netx,
        "X",
        vehicle_tensorsx,
        None if args.no_validation else v_vehicle_tensorsx,
        epochs=args.ae_x_epochs,
        save_per=args.ae_x_save_per,
        lr=args.ae_x_lr,
        batch_size=args.ae_x_batchsize,
        shuffle=args.ae_x_shuffle,
        device=device,
        run_dir=data_path,
        save=not args.no_save,
    )

    # Optimization: Calculate Diagnosis Features Per Vehicle to handle variance differences
    print("Calculating features per vehicle...")
    net.eval()
    netx.eval()

    # 학습 진단 특징: 차량마다 재구성 오차(batch 4096) → 진단 특징을 구해 이어 붙인다
    df_data = training_features(
        net, netx, zip(vehicle_tensors, vehicle_tensorsx), batch_size=4096, abs_err=use_abs_err
    )

    chart = ControlChart.fit(df_data)

    elapsed = time.perf_counter() - start
    h, rem = divmod(elapsed, 3600)
    m, s = divmod(rem, 60)
    print(f"{int(h)}시간 {int(m)}분 {s:.3f}초")

    chart.save(f"{data_path}/", df_data)


if __name__ == "__main__":
    main()
