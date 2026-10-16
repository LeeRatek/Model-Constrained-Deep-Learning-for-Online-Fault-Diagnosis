"""학습 스크립트: U-model(셀 전압)과 X-model(셀 SOC)을 학습하고, 학습 결과 폴더(Run)를 만든다.

실행 예
    python Train_.py --dataset-name 10 --ae-u-epochs 13 --ae-x-epochs 13 --ae-u-save-per 1 --ae-x-save-per 1
    옵션 전체는 README 7.2절, 처음 실행 순서는 README 1장에 있다.

하는 일 (main 순서)
    1. 분할 파일에서 학습 차량과 검증 차량 ID를 읽고, 차량마다 원본을 전처리해 메모리에 올린다.
    2. models/<RUN_ID>/ 폴더를 만들고 sim_config.txt(학습 설정 기록)를 쓴다.
    3. U-model을 학습한다. --ae-u-save-per epoch마다 검증 loss를 재고 체크포인트(net_e<epoch>.pth)를 저장한다.
    4. X-model도 같은 방식으로 학습한다(netx_e<epoch>.pth).
    5. 최종 모델로 정상 학습 차량의 진단 특징을 계산해 PCA 관리도를 만들고 pca_arrays.npz 등으로 저장한다.
       평가 스크립트(Test_.py 등)는 이 폴더를 읽어 평가한다.

다른 파일과의 관계
    데이터 읽기는 mcdl/vehicles.py, 모델 만들기·저장 규칙은 mcdl/run.py, 진단 특징은 mcdl/reconstruction.py,
    관리도는 mcdl/diagnosis.py를 쓴다. 모델 구조는 Class_.py에 있다.
"""

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
    """파이썬·numpy·torch 난수의 시작값(시드)을 고정해, 같은 시드면 같은 학습 결과가 나오게 한다. 쓰는 곳: main."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# 학습에 쓸 장치: GPU가 있으면 GPU, 없으면 CPU
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# 명령줄 옵션 정의. 실제 읽기(parse_args)는 main 안에서 한다 (import만 할 때는 아무것도 실행되지 않게).
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
    """학습 데이터를 배치(시점 묶음) 단위로 꺼내 주는 torch Dataset. 쓰는 곳: 아래 row_loader.

    여러 차량의 텐서를 하나로 이어 붙이면 메모리를 두 배로 쓰므로, 차량별 텐서 목록을 그대로 두고
    "전체 시점 번호 → 몇 번째 차량의 몇 번째 시점"을 계산해 필요한 행만 꺼낸다.
    한 번에 배치 하나(시점 번호 목록)를 받아 (x, y, Δ, q) 네 덩어리를 float64로 돌려준다.
    한 차량 안의 연속된 시점이면 한 번에 잘라 오고(빠름), 아니면 시점을 하나씩 모은다.
    x·y 폭은 호출하는 쪽이 모델 크기에서 읽어 넘긴다(mcdl/reconstruction.py와 같은 규칙).
    """

    def __init__(self, tensors, x_dim, y_dim):
        """tensors: 차량별 (시점 × 열) 텐서 목록, x_dim: x 열 수, y_dim: 셀 수. 차량별 시작 위치를 미리 계산해 둔다."""
        self.tensors = tensors
        self.cumulative = np.cumsum([0] + [t.shape[0] for t in tensors])
        self.x_dim = x_dim
        self.y_dim = y_dim

    def __len__(self):
        """전체 시점 수 (모든 차량의 시점 수 합)."""
        return self.cumulative[-1]

    def __getitem__(self, indices):
        """시점 번호 목록(배치) 하나를 받아 (x, y, Δ, q) 텐서 네 개를 돌려준다. DataLoader가 부른다."""
        # 이진 탐색으로 각 시점이 어느 차량의 몇 번째 시점인지 찾는다
        idx = np.asarray(indices)
        vehicle = np.searchsorted(self.cumulative, idx, side="right") - 1
        local = idx - self.cumulative[vehicle]
        if vehicle[0] == vehicle[-1] and np.all(np.diff(local) == 1):
            rows = self.tensors[vehicle[0]][local[0] : local[-1] + 1]  # 한 차량 안의 연속 구간
        else:
            rows = torch.stack([self.tensors[v][i] for v, i in zip(vehicle, local)])
        rows = rows.to(torch.double)

        # 열 구간마다 메모리에 빈틈없이 놓인 새 텐서로 복사한다. 예전 방식(시점을 하나씩 꺼내 쌓기)과 메모리
        # 배치까지 같아야 행렬 계산이 같은 경로를 타서 결과가 소수점 끝자리까지 같다.
        # .contiguous()는 배치가 1행일 때 복사하지 않아 메모리 배치가 달라지므로 clone을 쓴다.
        y_start = self.x_dim
        dx_start = y_start + self.y_dim
        q_start = dx_start + self.y_dim
        return tuple(
            rows[:, a:b].clone(memory_format=torch.contiguous_format)
            for a, b in ((0, y_start), (y_start, dx_start), (dx_start, q_start), (q_start, None))
        )


def row_loader(tensors, x_dim, y_dim, *, batch_size, shuffle):
    """학습·검증 데이터를 batch_size 시점씩 차례로 내주는 DataLoader를 만든다. 쓰는 곳: train_autoencoder.

    shuffle=True면 매 epoch 시점 순서를 섞는다. 내부 구성이 torch 기본 DataLoader(shuffle=…, batch_size=…)와
    같아서, 섞는 순서와 난수를 쓰는 횟수까지 예전 코드와 같다(그래야 같은 시드에서 같은 학습 결과가 나온다).
    """
    rows = VehicleRows(tensors, x_dim, y_dim)
    sampler = RandomSampler(rows) if shuffle else SequentialSampler(rows)
    return DataLoader(rows, sampler=BatchSampler(sampler, batch_size, drop_last=False), batch_size=None)


# 모델별 loss 기록 파일 이름 (Run 폴더에 <이름>.csv, <이름>_val.csv로 저장된다)
LOSS_CSV = {"U": "loss_net", "X": "loss_netx"}


def train_autoencoder(
    net, model, train_tensors, val_tensors, *, epochs, save_per, lr, batch_size, shuffle, device, run_dir, save
):
    """모델 하나(model = "U" 또는 "X")를 학습하고 체크포인트·loss 기록을 Run 폴더에 저장한다. 쓰는 곳: main.

    - 손실은 재구성 값과 실제 셀 값의 평균제곱오차(MSE), 최적화는 Adam, 계산은 float64다.
    - epoch 번호가 save_per의 배수일 때마다 검증 loss를 계산해 출력하고, 체크포인트(<run_dir>/artifact/)를 저장한다.
    - 끝나면 epoch별 학습 loss(loss_net.csv 등), 검증 loss(loss_net_val.csv 등), 최종 모델(net.pth/netx.pth)을 저장한다.
    - val_tensors가 None이면 검증을 건너뛴다(--no-validation). save=False면 .pth 파일을 저장하지 않는다(--no-save).
    재현성 주의: DataLoader는 데이터를 한 바퀴 돌 때마다(섞지 않아도) 난수를 쓴다. 그래서 이 함수 안의 순서
    (epoch마다 학습 → save_per마다 검증)와 main의 호출 순서(U → X)를 바꾸면 shuffle을 켠 학습 결과가 달라진다.
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
    """학습 전체를 실행한다 (파일 맨 위 "하는 일" 1~5). argv는 명령줄 옵션 목록이며, None이면 실제 명령줄을 읽는다.

    `python Train_.py ...`로 실행하면 맨 아래의 `if __name__ == "__main__"`이 이 함수를 부른다.
    테스트(tests/test_scripts.py)는 Train_.main([...])처럼 옵션 목록을 직접 넘겨 부른다.
    """
    os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"  # 일부 환경에서 수학 라이브러리 중복 로드 오류를 피하는 설정
    print(torch.cuda.device_count())  # 사용할 수 있는 GPU 수 (0이면 CPU로 학습)

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
