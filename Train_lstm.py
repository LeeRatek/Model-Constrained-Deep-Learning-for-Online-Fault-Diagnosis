"""LSTM 학습 스크립트: 팩 상태 예측 LSTM만 학습해 학습 결과 폴더(Run)에 저장한다 (MC-AE 학습과 따로).

LSTM은 한 시점 전의 팩 측정값 7개(vin_1.pkl)로 현재 시점의 대표 셀 전압과 SOC를 예측한다(Class_.py의 LSTM,
논문 원본 코드에서 복원). MC-AE(Train_.py)와 데이터·모델·결과를 공유하지 않으므로 하나만 학습해도 되고,
둘 다 필요하면 각각 실행한다.

실행 예
    python Train_lstm.py --dataset-name 10 --epochs 300
    옵션 전체는 README 7.6절에 있다.

하는 일 (main 순서)
    1. 분할 파일에서 학습 차량 ID를 읽고, 차량마다 vin_1.pkl을 읽는다.
       분할 폴더 규칙은 Train_.py·평가 스크립트와 같다(mcdl/run.py의 resolve_split_dir).
    2. models/<RUN_ID>/ 폴더를 만들고 sim_config.txt(학습 설정 기록, model : LSTM)를 쓴다.
    3. 학습 차량 전체로 LSTM 하나를 학습한다(mcdl/lstm.py의 train_lstm).
    4. epoch별 loss를 loss_lstm.csv에, 모델을 artifact/lstm.pth에 저장한다.
    이 Run 폴더에는 MC-AE 모델과 관리도가 없어서 평가 스크립트(Test_.py 등)는 이 폴더를 받으면 오류로 멈춘다.
"""

import argparse
import os
import time
import warnings

import numpy as np
import torch

from mcdl.lstm import save_lstm_model, train_lstm
from mcdl.preprocessing import safe_load
from mcdl.run import make_model_path_based_timestamp, resolve_split_dir, set_seed, write_sim_config
from mcdl.vehicles import split_ids

# 학습에 쓸 장치: GPU가 있으면 GPU, 없으면 CPU
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# 명령줄 옵션 정의. 실제 읽기(parse_args)는 main 안에서 한다 (import만 할 때는 아무것도 실행되지 않게).
parser = argparse.ArgumentParser(description="Train the pack state prediction LSTM (independent of the MC-AE)")
parser.add_argument("--battery-type", choices=["QAS", "DTI"], default="QAS")
parser.add_argument(
    "--dataset-name",
    type=str,
    default=None,
    help="./datasets/<이름>의 분할을 쓴다. 생략하면 저장소 루트의 기본 분할",
)
parser.add_argument("--vehicle-start", type=int, default=0, help="train 분할에서 쓸 차량 범위의 시작 위치")
parser.add_argument("--vehicle-end", type=int, default=-1, help="train 분할에서 쓸 차량 범위의 끝 위치 (포함, -1은 끝까지)")
parser.add_argument(
    "--models-dir",
    type=str,
    default=os.environ.get("MODEL_DIR") or "./models",
)
parser.add_argument(
    "--source-data-dir",
    type=str,
    default=os.environ.get("SOURCE_DIR") or "./data",
)
parser.add_argument("--epochs", type=int, default=300)
parser.add_argument("--lr", type=float, default=1e-4, help="Adam 학습률")
parser.add_argument("--batchsize", type=int, default=100)
parser.add_argument("--seed", type=int, default=42, help="난수 시드 (같은 시드면 같은 학습 결과)")


def main(argv=None):
    """LSTM 학습 전체를 실행한다 (파일 맨 위 "하는 일" 1~4). argv는 명령줄 옵션 목록이며, None이면 실제 명령줄을 읽는다.

    `python Train_lstm.py ...`로 실행하면 맨 아래의 `if __name__ == "__main__"`이 이 함수를 부른다.
    테스트(tests/test_scripts.py)는 Train_lstm.main([...])처럼 옵션 목록을 직접 넘겨 부른다.
    """
    os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"  # 일부 환경에서 수학 라이브러리 중복 로드 오류를 피하는 설정
    warnings.filterwarnings("ignore")

    args = parser.parse_args(argv)
    set_seed(args.seed)
    start = time.perf_counter()

    split_dir = resolve_split_dir(args.dataset_name)
    print(f"Dataset path: {split_dir}")
    train_list = split_ids(split_dir, args.battery_type, "train", start=args.vehicle_start, end=args.vehicle_end)

    vin1_tensors = []
    for count, vehicle_id in enumerate(train_list, start=1):
        if count % 10 == 0:
            print(f"Processing vehicle {count}/{len(train_list)}")
        vin1_tensors.append(safe_load(f"{args.source_data_dir}/{args.battery_type}/{vehicle_id}/vin_1.pkl"))

    run_dir = make_model_path_based_timestamp(base=args.models_dir)
    text = write_sim_config(
        run_dir,
        args,
        extra={
            "model": "LSTM",  # 평가 스크립트가 MC-AE Run과 구별하는 표시 (mcdl/run.py의 Run)
            "device": str(device),
            "Vehicle IDs used for training": train_list,
            "Amount of data used for training": sum(len(v) for v in vin1_tensors),
        },
    )
    print(text, end="")  # 콘솔 출력

    lstm, losses = train_lstm(
        vin1_tensors, epochs=args.epochs, lr=args.lr, batch_size=args.batchsize, device=device
    )

    np.savetxt(
        os.path.join(run_dir, "loss_lstm.csv"),
        np.column_stack((np.arange(len(losses)), losses)),
        delimiter=",",
        fmt=["%.0f", "%.6e"],
        header="epoch,avg_loss",
        comments="",
    )
    save_lstm_model(lstm, f"{run_dir}/artifact/")

    elapsed = time.perf_counter() - start
    h, rem = divmod(elapsed, 3600)
    m, s = divmod(rem, 60)
    print(f"LSTM saved to {run_dir}/artifact/lstm.pth ({int(h)}시간 {int(m)}분 {s:.3f}초)")


if __name__ == "__main__":
    main()
