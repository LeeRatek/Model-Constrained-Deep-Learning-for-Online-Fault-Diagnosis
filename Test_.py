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
from sklearn.model_selection import train_test_split
import argparse
from monitering import *
import time
from mcdl.run import Run, print_sim_config
from mcdl.reconstruction import vehicle_features
from mcdl.chart_source import chart_for_pair
from mcdl.vehicles import load_vehicle, split_ids
from mcdl.roc import evaluate

device = torch.device("cpu")  # 평가는 CPU 전용 (Run이 CPU 모델을 돌려준다)
# VIN_data = pd.read_excel(r'{args.source_data_dir}/Name_list.xls')

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
parser.add_argument("--sigma-levels", type=str, default="3,4.5,6")
parser.add_argument("--u-model-idx", type=int, default=-1)
parser.add_argument("--x-model-idx", type=int, default=-1)
parser.add_argument(
    "--debug",
    action="store_true",
    help="빠른 확인용: 모든 반복에서 차량 338만 평가",
)


def main(argv=None):
    os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
    print(torch.cuda.device_count())

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
    # SKIP_CHARGE_READY = False
    AUC_ANALYSIS = True
    if not AUC_ANALYSIS:
        thresholds = [10.4, 87.5]
        # thresholds = [20, 30]

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

    # sim_config_path = os.path.join(f"{args.models_dir}/{args.models_folder}", "sim_config.txt")
    # train_vehicle_ids = load_vehicle_ids_used_for_training(sim_config_path)
    # all_normal_vehicle_ids = np.load(f"{DATA_BASE_PATH}/{BATTERY_TYPE}_filtered_vehicle_ids_normal.npy").astype(np.int64).tolist()
    # normal_list = [vid for vid in all_normal_vehicle_ids if vid not in set(train_vehicle_ids)]
    normal_list = split_ids(DATA_BASE_PATH, BATTERY_TYPE, "test")
    fault_list = split_ids(DATA_BASE_PATH, BATTERY_TYPE, "fault")
    # fault_list = np.arange(335, 393)
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

            # interest_order = 5
            # do_show = False
            # time_idx = find_max_idx_from_array(np.asarray(ERRORU, dtype=float).max(axis=1), interest_order)[0][0]
            # print(f"Max ERRORU at time index {time_idx}, value: {ERRORU[time_idx].max():.4f}")
            # cell_idx = np.where(ERRORU[time_idx] == ERRORU[time_idx].max())[0][0]
            # print(f"Max ERRORU at cell index {cell_idx}, value: {ERRORU[time_idx][cell_idx]:.4f}")
            # if do_show:
            #     net_loaded.analyze_outputs(x_recovered, recon_imtest[1], q_recovered, recon_imtest[0], y_recovered, cell_idx=cell_idx, combine=True, ncols=3, fill="spiral")
            #     netx_loaded.analyze_outputs(x_recovered2, reconx_imtest[1], q_recovered2, reconx_imtest[0], y_recovered2, cell_idx=cell_idx, combine=True, ncols=3, fill="spiral")
            #     plot_testX_timeseries(ERRORU, feature_names="|U - $\hat{U}|$", title="|U - $\hat{U}|$", figsize=(6, 3), show=True, seperate=False, start_idx=0, _range=[cell_idx,cell_idx])
            #     plot_testX_timeseries(ERRORX, feature_names="|X - $\hat{X}|$", title="|X - $\hat{X}|$", figsize=(6, 3), show=True, seperate=False, start_idx=0, _range=[cell_idx,cell_idx])

            # plot_testX_timeseries(y_recovered, feature_names="U", title="U", figsize=(6, 3), show=True, seperate=False, start_idx=0, _range=[0,0])
            # plot_testX_timeseries(y_recovered2, feature_names="X", title="X", figsize=(6, 3), show=True, seperate=False, start_idx=0, _range=[0,0])
            # plot_testX_timeseries(recon_imtest[0], feature_names="$\hat{U}$", title="$\hat{U}$", figsize=(6, 3), show=True, seperate=False, start_idx=0, _range=[0,0])
            # plot_testX_timeseries(reconx_imtest[0], feature_names="$\hat{X}$", title="$\hat{X}$", figsize=(6, 3), show=True, seperate=False, start_idx=0, _range=[0,0])
            # plot_testX_timeseries(ERRORU, feature_names="|U - $\hat{U}|$", title="|U - $\hat{U}|$", figsize=(6, 3), show=True, seperate=False, start_idx=0, _range=[0,0])
            # plot_testX_timeseries(ERRORX, feature_names="|X - $\hat{X}|$", title="|X - $\hat{X}|$", figsize=(6, 3), show=True, seperate=False, start_idx=0, _range=[0,0])
            # plot_testX_timeseries(np.asarray(ERRORU, dtype=float).max(axis=1).reshape(ERRORU.shape[0],1), feature_names="max(|U - $\hat{U}|$)", title="max(|U - $\hat{U}|$)", figsize=(6, 3), show=True, seperate=False, start_idx=0, _range=[0,0])
            # plot_testX_timeseries(np.asarray(ERRORX, dtype=float).max(axis=1).reshape(ERRORU.shape[0],1), feature_names="max(|X - $\hat{X}|$)", title="max(|X - $\hat{X}|$)", figsize=(6, 3), show=True, seperate=False, start_idx=0, _range=[0,0])

            # plot_testX_timeseries(np.asarray(np.abs(ERRORU), dtype=float).max(axis=1).reshape(ERRORU.shape[0],1), feature_names="|max(U - $\hat{U}$)|", title="|max(U - $\hat{U}$)|", figsize=(6, 3), show=True, seperate=False, start_idx=0, _range=[0,0])
            # plot_testX_timeseries(np.asarray(np.abs(ERRORX), dtype=float).max(axis=1).reshape(ERRORU.shape[0],1), feature_names="|max(X - $\hat{X}$)|", title="|max(X - $\hat{X}$)|", figsize=(6, 3), show=True, seperate=False, start_idx=0, _range=[0,0])

            # title = ["$zu_2$", "$zx_2$", "$zu_1$", "$zx_1$", "$ewu$", "$ewx$"]
            # for i in range(len(title)):
            #     plot_testX_timeseries(t2_contrib[:, i].to_numpy().reshape(df_data2.shape[0],1), feature_names=title[i], title=title[i], figsize=(6, 3), show=True, seperate=False, start_idx=0, _range=[0,0])

            # plot_testX_timeseries(df_data2.iloc[:,0].to_numpy().reshape(df_data2.shape[0],1), feature_names="$zu_2$", title="$zu_2$", figsize=(6, 3), show=True, seperate=False, start_idx=0, _range=[0,0])
            # plot_testX_timeseries(df_data2.iloc[:,1].to_numpy().reshape(df_data2.shape[0],1), feature_names="$zx_2$", title="$zx_2$", figsize=(6, 3), show=True, seperate=False, start_idx=0, _range=[0,0])
            # plot_testX_timeseries(df_data2.iloc[:,2].to_numpy().reshape(df_data2.shape[0],1), feature_names="$zu_1$", title="$zu_1$", figsize=(6, 3), show=True, seperate=False, start_idx=0, _range=[0,0])
            # plot_testX_timeseries(df_data2.iloc[:,3].to_numpy().reshape(df_data2.shape[0],1), feature_names="$zx_1$", title="$zx_1$", figsize=(6, 3), show=True, seperate=False, start_idx=0, _range=[0,0])
            # plot_testX_timeseries(df_data2.iloc[:,4].to_numpy().reshape(df_data2.shape[0],1), feature_names="$ewu$", title="$ewu$", figsize=(6, 3), show=True, seperate=False, start_idx=0, _range=[0,0])
            # plot_testX_timeseries(df_data2.iloc[:,5].to_numpy().reshape(df_data2.shape[0],1), feature_names="$ewx$", title="$ewx$", figsize=(6, 3), show=True, seperate=False, start_idx=0, _range=[0,0])
            # df_origin = (data_nor * data_std) + data_mean

            # plot_testX_timeseries(z_recovered, feature_names="$\Delta{U}$", title="$\Delta{U}$", figsize=(6, 3), show=True, seperate=False, start_idx=0, _range=[0,0])
            # plot_testX_timeseries(x_recovered[:, 0].reshape(x_recovered.shape[0],1), feature_names="$U^{pre1}$", title="$U^{pre1}$", figsize=(6, 3), show=True, seperate=False, start_idx=0, _range=[0,0])
            # plot_testX_timeseries((x_recovered[:, 0].unsqueeze(1) + z_recovered), feature_names="$U^{pre1}$ + $\Delta{U}$", title="$U^{pre1}$ + $\Delta{U}$", figsize=(6, 3), show=True, seperate=False, start_idx=0, _range=[0,0])
            # plot_testX_timeseries((x_recovered[:, 0].unsqueeze(1) - z_recovered), feature_names="$U^{pre1}$ - $\Delta{U}$", title="$U^{pre1}$ - $\Delta{U}$", figsize=(6, 3), show=True, seperate=False, start_idx=0, _range=[0,0])

            # 차량 한 대의 진단 특징 (차량 전체를 한 번에 추론, 오차의 abs 여부는 학습 진단 특징과 같은 설정을 따른다)
            df_data = vehicle_features(
                net_loaded,
                netx_loaded,
                combined_tensor,
                combined_tensorx,
                abs_err=not run.config.no_abs_err,
            )
            # plot_ae_output_distribution(df_data2, do_plot=analyse_ae_output)

            # u_max_idx = find_max_idx_from_array(df_data2.iloc[:,0])[0][0]
            # ci_max_idx = find_max_idx_from_array(CI_array)[0][0]
            # [[df_data2.iloc[u_max_idx,0],df_data2.iloc[u_max_idx,1],df_data2.iloc[u_max_idx,2],df_data2.iloc[u_max_idx,3],df_data2.iloc[u_max_idx,4],df_data2.iloc[u_max_idx,5]],
            #  [df_data2.iloc[ci_max_idx ,0],df_data2.iloc[ci_max_idx ,1],df_data2.iloc[ci_max_idx ,2],df_data2.iloc[ci_max_idx ,3],df_data2.iloc[ci_max_idx ,4],df_data2.iloc[ci_max_idx ,5]]]
            # [[df_data.iloc[u_max_idx,0],df_data.iloc[u_max_idx,1],df_data.iloc[u_max_idx,2],df_data.iloc[u_max_idx,3],df_data.iloc[u_max_idx,4],df_data.iloc[u_max_idx,5]],
            # [df_data.iloc[ci_max_idx ,0],df_data.iloc[ci_max_idx ,1],df_data.iloc[ci_max_idx ,2],df_data.iloc[ci_max_idx ,3],df_data.iloc[ci_max_idx ,4],df_data.iloc[ci_max_idx ,5]]]

            # max_err = np.array(df_data.iloc[:,2]).max()
            # max_err_time = np.where(np.array(df_data.iloc[:,2])> (max_err - 0.00001))
            # max_err2 = np.asarray(ERRORU[max_err_time,:], dtype=float).max(axis=1).max()
            # max_err_idx = np.where(np.asarray(ERRORU[max_err_time], dtype=float) > (max_err2 - 0.00001))

            # ## =================== Calculate thresholds =================== ###
            # sigma_levels = [float(s.strip()) for s in args.sigma_levels.split(',') if s.strip()]
            # new_X = (df_data - data_mean) / data_std
            # pi, g, h = chi_square_dist_components(p_k, v_I, np.cov(new_X.T), SPE_95_limit, T_95_limit)
            # thresholds = diagnosis_thresholds(g, h, sigma_levels=sigma_levels, show_plot=False)

            ## =================== Testing Diagnosis =================== ##
            # feature_names=["zx1","zx2","ewx","zu1","zu2","ewu"]
            # title = ["$zu_2$", "$zx_2$", "$zu_1$", "$zx_1$", "$ewu$", "$ewx$"]
            # plot_contrib_percent_stacked(t2_contrib,val_array=t2_array,feature_names=title,top_k=100,mode="abs",kind="$T^2$",)
            # plot_contrib_percent_stacked(spe_contrib,val_array=spe_array,feature_names=title,top_k=100,mode="abs",kind="SPE",)
            # plot_contrib_percent_stacked(CI_contrib,val_array=CI_array,feature_names=title,top_k=100,mode="abs",kind="CI",)
            # plot_contrib_percent_stacked(
            #     CI_contrib, val_array=CI_array, feature_names=title,
            #     kind="CI", mode="abs", plot_step=100, xtick_step=10
            #     )

            # for i in range(len(title)):
            #     plot_testX_timeseries(t2_contrib[:, i].reshape(t2_contrib.shape[0],1), feature_names=title[i], title="t2_"+title[i], figsize=(6, 3), show=True, seperate=False, start_idx=0, _range=[0,0])

            # for i in range(len(title)):
            #     plot_testX_timeseries(spe_contrib[:, i].reshape(spe_contrib.shape[0],1), feature_names=title[i], title="spe_"+title[i], figsize=(6, 3), show=True, seperate=False, start_idx=0, _range=[0,0])

            if AUC_ANALYSIS:
                # =================== 차량 판정 점수 (ROC는 루프가 끝난 뒤 evaluate에서) ===================
                vehicle_max_ci.append(chart.max_ci(df_data))
                vehicle_is_fault.append(label == 1)
            else:
                # =================== Plotting =================== ##
                t2_array, spe_array, CI_array = chart.score(df_data)
                title = ["$zu_2$", "$zx_2$", "$zu_1$", "$zx_1$", "$ewu$", "$ewx$"]
                start1 = time.perf_counter()
                # plot_contrib_percent_stacked(
                #     CI_contrib,
                #     val_array=CI_array,
                #     save_path=f"{args.models_dir}/{args.models_folder}/results_ci/{VEHICLE_ID}_testing_contrib_top{learning_case}.png",
                #     show=False,
                #     feature_names=title,
                #     top_k=100,
                #     mode="abs",
                #     kind="CI",
                # )
                # plot_contrib_percent_stacked(
                #     CI_contrib,
                #     val_array=CI_array,
                #     save_path=f"{args.models_dir}/{args.models_folder}/results_ci/{VEHICLE_ID}_testing_contrib{learning_case}.png",
                #     show=False,
                #     feature_names=title,
                #     kind="CI",
                #     mode="abs",
                #     plot_step=100,
                #     xtick_step=10,
                # )
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
