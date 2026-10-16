"""평가 스크립트 (모델 쌍 하나): 학습된 Run의 U-model·X-model 한 쌍으로 평가 차량을 진단하고 ROC/AUC를 계산한다.

실행 예
    python Test_.py --models-idx <RUN_ID>                                  # 최종 모델 쌍 (-1, -1)
    python Test_.py --models-idx <RUN_ID> --u-model-idx 2 --x-model-idx 0  # 특정 체크포인트 쌍
    옵션 전체는 README 7.3절에 있다.

하는 일 (main 순서)
    1. Run 폴더의 sim_config.txt에서 학습 설정을 읽는다(배터리 종류, 전처리, 분할 등을 학습 때와 똑같이 쓰려고).
    2. 평가 차량 목록을 읽는다: test 분할(정상)과 fault 분할(고장).
    3. 모델 쌍의 관리도를 가져온다(mcdl/chart_source.py: 캐시 → 저장된 파일 → 다시 계산).
    4. 차량마다 원본을 읽어 진단 특징을 계산하고, 관리도로 최대 CI(차량의 이상 점수)를 구한다.
    5. 모든 차량의 최대 CI와 고장 여부로 ROC/AUC와 추천 임계값을 계산해 출력하고, ROC 그림을 저장·표시한다.
    코드의 AUC_ANALYSIS를 False로 바꾸면 4·5 대신 차량별 T²·SPE·CI 시계열 그림을 저장한다(진단 그림 모드).

결과물: models/<RUN_ID>/results/AUC_ROC<learning_case>.png, 모델 층 그림, loss 곡선 그림 (README 8장).
"""

import os
import warnings
import torch

import argparse
import time
from mcdl.run import Run, print_sim_config
from mcdl.reconstruction import vehicle_features
from mcdl.chart_source import chart_for_pair
from mcdl.vehicles import load_vehicle, split_ids
from mcdl.roc import evaluate
from mcdl.plotting import (
    plot_diagnostics_triplet,
    plot_loss_curve_all,
    plot_net_layer_params_by_index,
    plot_roc_curve,
)

device = torch.device("cpu")  # 평가는 CPU 전용 (Run이 CPU 모델을 돌려준다)

# 명령줄 옵션 정의. 실제 읽기(parse_args)는 main 안에서 한다 (import만 할 때는 아무것도 실행되지 않게).
parser = argparse.ArgumentParser(
    description="Run diagnostics plotting with CLI options"
)

parser.add_argument(
    "--models-dir",
    type=str,
    default=(
        "./models"
        if os.environ.get("MODEL_DIR") is None
        else os.environ.get("MODEL_DIR")
    ),
)  # 260206_084839 & 260211_163108 & 260219_111732 & 260224_090359
# --models-idx는 다른 스크립트와 같은 이름의 별칭
parser.add_argument(
    "--models-folder", "--models-idx", dest="models_folder", type=str, default="260224_090359"
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
parser.add_argument("--u-model-idx", type=int, default=-1)
parser.add_argument("--x-model-idx", type=int, default=-1)
parser.add_argument(
    "--debug",
    action="store_true",
    help="빠른 확인용: 모든 반복에서 차량 338만 평가",
)


def main(argv=None):
    """평가 전체를 실행한다 (파일 맨 위 "하는 일" 1~5). argv는 명령줄 옵션 목록이며, None이면 실제 명령줄을 읽는다.

    `python Test_.py ...`로 실행하면 맨 아래의 `if __name__ == "__main__"`이 이 함수를 부른다.
    테스트(tests/test_scripts.py)는 Test_.main([...])처럼 옵션 목록을 직접 넘겨 부른다.
    """
    os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"  # 일부 환경에서 수학 라이브러리 중복 로드 오류를 피하는 설정
    print(torch.cuda.device_count())  # 사용할 수 있는 GPU 수 (평가는 GPU가 있어도 CPU로 한다)

    warnings.filterwarnings("ignore")

    args = parser.parse_args(argv)

    run = Run(f"{args.models_dir}/{args.models_folder}")

    learning_case = run.config.learning_case
    add_one_output_layer = run.config.add_one_output_layer
    use_dx = run.config.use_dx
    u_model_idx, x_model_idx = args.u_model_idx, args.x_model_idx

    # --debug: 모든 반복에서 차량 338만 평가한다 (빠른 확인용). 모델 쌍은 CLI 인자 그대로다
    is_debug = args.debug

    if is_debug:
        print("현재 디버그 모드로 실행 중입니다.")

    # 플롯 결과 저장 폴더는 한 번만 생성
    os.makedirs(f"{args.models_dir}/{args.models_folder}/results", exist_ok=True)

    BATTERY_TYPE = run.config.battery_type  # 'DTI' or 'QAS'

    DATA_BASE_PATH = run.config.split_dir
    print(f"Using dataset from: {DATA_BASE_PATH}")

    PREPROCESSING, SKIP_CHARGE_READY = run.config.preprocessing, run.config.skip_charge_ready
    AUC_ANALYSIS = True
    if not AUC_ANALYSIS:
        thresholds = [10.4, 87.5]

    print_sim_config(
        title="Train Run",
        config=args,
        extra={
            "device": str(device),
            "learning_case": learning_case,
            "PREPROCESSING": PREPROCESSING,
            "SKIP_CHARGE_READY": SKIP_CHARGE_READY,
            "no_use_dx": not use_dx,
            "add_one_output_layer": add_one_output_layer,
        },
    )


    ## =================== DTI fault index range = [77~79] ===================
    ## =================== QAS fault index range = [335~392] ===================

    normal_list = split_ids(DATA_BASE_PATH, BATTERY_TYPE, "test")
    fault_list = split_ids(DATA_BASE_PATH, BATTERY_TYPE, "fault")
    test_list = [normal_list, fault_list]


    ## =================== Initilize params for AUROC curve  =================== ##
    vehicle_max_ci = []  # 차량별 최대 CI (test_list 순서)
    vehicle_is_fault = []

    ## =================== Load training data  =================== ##
    start_pca_load = time.perf_counter()
    try:
        plot_loss_curve_all(path=f"{args.models_dir}/{args.models_folder}")
    except Exception as e:
        print(f"Error plotting loss curve: {e}")

    # 관리도: 같은 조건 캐시 → 최종 모델 쌍이면 저장된 관리도 → 그 밖엔 학습 데이터로 재계산
    chart = chart_for_pair(run, u_model_idx, x_model_idx, source_data_dir=args.source_data_dir)
    elapsed_pca_load = time.perf_counter() - start_pca_load
    print(f"PCA results loading time (once): {elapsed_pca_load:.3f} seconds")

    net_loaded, netx_loaded = run.load_model_pair(u_model_idx, x_model_idx)

    ## =================== Plot network layer parameters  =================== ##
    showing_params = False
    fc1_w = net_loaded.fc1.weight.detach().cpu().numpy()
    fc1_b = net_loaded.fc1.bias.detach().cpu().numpy()
    fc2_w = net_loaded.fc2.weight.detach().cpu().numpy()
    fc2_b = net_loaded.fc2.bias.detach().cpu().numpy()
    print(
        f"Weights and biases of net: fc1[w:{fc1_w}, b:{fc1_b}], fc2[w:{fc2_w}, b:{fc2_b}]"
    )
    plot_net_layer_params_by_index(
        net_loaded,
        net_name="net",
        mode="bar",
        max_points=8000,
        show=showing_params,
        save_dir=f"{args.models_dir}/{args.models_folder}",
        layer_names=("fc1", "fc2"),
        figsize=(6, 3),
    )
    plot_net_layer_params_by_index(
        net_loaded,
        net_name="net",
        mode="bar",
        max_points=8000,
        show=showing_params,
        save_dir=f"{args.models_dir}/{args.models_folder}",
        layer_names=("fc1", "fc2"),
        figsize=(6, 3),
        normalize_y=False,
    )

    x_fc1_w = netx_loaded.fc1.weight.detach().cpu().numpy()
    x_fc1_b = netx_loaded.fc1.bias.detach().cpu().numpy()
    x_fc2_w = netx_loaded.fc2.weight.detach().cpu().numpy()
    x_fc2_b = netx_loaded.fc2.bias.detach().cpu().numpy()
    print(
        f"Weights and biases of netx: fc1[w:{x_fc1_w}, b:{x_fc1_b}], fc2[w:{x_fc2_w}, b:{x_fc2_b}]"
    )
    plot_net_layer_params_by_index(
        netx_loaded,
        net_name="netx",
        mode="bar",
        max_points=8000,
        show=showing_params,
        save_dir=f"{args.models_dir}/{args.models_folder}",
        layer_names=("fc1", "fc2"),
        figsize=(6, 3),
    )
    plot_net_layer_params_by_index(
        netx_loaded,
        net_name="netx",
        mode="bar",
        max_points=8000,
        show=showing_params,
        save_dir=f"{args.models_dir}/{args.models_folder}",
        layer_names=("fc1", "fc2"),
        figsize=(6, 3),
        normalize_y=False,
    )

    start = time.perf_counter()
    for label, vehicle_ids in enumerate(test_list):
        for i in vehicle_ids:
            if is_debug:
                i = 338  # 3, 46, 351, 362, 382
            print(f"Processing label={label} with vehicle ID={i}...")
            elapsed = time.perf_counter() - start
            h, rem = divmod(elapsed, 3600)
            m, s = divmod(rem, 60)
            print(f"Elapsed time: {int(h)}시간 {int(m)}분 {s:.3f}초")
            VEHICLE_ID = f"{i}"

            combined_tensor, combined_tensorx = load_vehicle(args.source_data_dir, run.config, VEHICLE_ID)

            # 차량 한 대의 진단 특징 (차량 전체를 한 번에 추론, 오차의 abs 여부는 학습 진단 특징과 같은 설정을 따른다)
            df_data = vehicle_features(
                net_loaded,
                netx_loaded,
                combined_tensor,
                combined_tensorx,
                abs_err=not run.config.no_abs_err,
            )

            ## =================== Testing Diagnosis =================== ##
            if AUC_ANALYSIS:
                # =================== 차량 판정 점수 (ROC는 루프가 끝난 뒤 evaluate에서) ===================
                vehicle_max_ci.append(chart.max_ci(df_data))
                vehicle_is_fault.append(label == 1)
            else:
                # =================== Plotting =================== ##
                t2_array, spe_array, CI_array = chart.score(df_data)
                start1 = time.perf_counter()
                plot_diagnostics_triplet(
                    t2_array,
                    spe_array,
                    CI_array,
                    thresholds,
                    sigma_levels=None,
                    title="Test",
                    x_label="Time",
                    y_labels=("T²", "SPE", "CI"),
                    figsize=(12, 10),
                    save_path=f"{args.models_dir}/{args.models_folder}/results/{VEHICLE_ID}_testing_diagnostics_case{learning_case}.png",
                    show=False,
                    x_start=args.x_start,
                    x_tick_step=args.x_tick_step,
                    downsample=1,  # 1 ~ 10
                    tight_layout=False,
                    dpi=100,  # 100 ~ 150
                    png_compress_level=1,
                )
                elapsed1 = time.perf_counter() - start1
                print(f"Plotting time for vehicle ID={VEHICLE_ID}: {elapsed1:.3f} seconds")
            print("Done")


    elapsed = time.perf_counter() - start
    h, rem = divmod(elapsed, 3600)
    m, s = divmod(rem, 60)
    print(f"{int(h)}시간 {int(m)}분 {s:.3f}초")

    if AUC_ANALYSIS:
        # =================== ROC / AUC ===================
        roc_save_path = (
            f"{args.models_dir}/{args.models_folder}/results/AUC_ROC{learning_case}.png"
        )
        result = evaluate(vehicle_max_ci, vehicle_is_fault)
        best, closest = result.best, result.closest_to_01

        print(f"AUC = {result.auc:.4f}")
        print(
            f"Best threshold (Youden J) ≈ {best['threshold']:.6g} @ (FPR={best['fpr']:.4f}, TPR={best['tpr']:.4f})"
        )
        print(
            f"Closest to (0,1) ≈ {closest['threshold']:.6g} @ (FPR={closest['fpr']:.4f}, TPR={closest['tpr']:.4f}), dist={closest['distance']:.4f}"
        )

        plot_roc_curve(
            result.fpr,
            result.tpr,
            result.auc,
            best_point=best,
            closest_to_01_point=closest,
            save_path=roc_save_path,
            title="AUC-ROC Curve",
            figsize=(6, 6),
            show=True,
            dpi=200,
        )
        print(f"ROC curve saved to: {roc_save_path}")


if __name__ == "__main__":
    main()
