"""시뮬레이션 스크립트 (체크포인트 조합 비교): Run에 저장된 여러 U·X 체크포인트 조합의 진단 성능(AUC)을 비교한다.

학습 중에는 epoch마다 체크포인트가 저장된다. 어느 epoch의 U-model과 어느 epoch의 X-model을 함께 쓸 때 진단이
가장 잘 되는지 고르기 위한 메인 평가 스크립트다.

실행 예
    python Test_multi_models.py --models-idx <RUN_ID>                 # 모든 체크포인트 조합
    python Test_multi_models.py --models-idx <RUN_ID> --fast-search   # 조합 수 줄이기 (U 먼저 고르고 X 훑기)
    옵션 전체는 README 7.4절에 있다.

하는 일 (main 순서)
    1. 평가할 U·X 체크포인트 번호를 정하고 모델을 모두 불러온다.
    2. 학습 차량과 평가 차량(test + fault)을 읽는다. 처음에는 원본에서 읽어 data/에 .pt 캐시 파일을 만들고,
       다음 실행부터는 캐시를 읽는다(메모리를 많이 쓴다: README 1장 준비물).
    3. 체크포인트마다 학습 진단 특징(자기 몫 3개)을 계산해 Run 폴더의 temp_cache/에 저장한다(--cache-strategy disk).
    4. 조합마다 학습 진단 특징으로 관리도를 만들고, 평가 차량의 최대 CI로 ROC/AUC를 계산해
       AUC_summary_case<C>.csv에 한 줄씩 쌓고 조합별 ROC 그림을 저장한다(process_combination).
    5. 2개 이상 계산했으면 모든 조합의 ROC 비교 그림과 AUC 히트맵을 저장한다.
       --keep-cache가 없으면 temp_cache/를 지운다(남겨 두면 Test_.py·Test_tsne.py가 재사용한다).
"""

import matplotlib.pyplot as plt
import pandas as pd
import os
import shutil
import warnings

import argparse
import time
from mcdl.run import Run, print_sim_config
from mcdl.diagnosis import ControlChart, combine_signal_features
from mcdl.reconstruction import training_signal_features, vehicle_features
from mcdl.chart_source import load_cached_features, save_cached_features
from mcdl.roc import evaluate
from mcdl.plotting import draw_auc_heatmap, plot_loss_curve_all, plot_roc_curve
from mcdl.vehicles import load_test_vehicles, load_training_vehicles, split_ids

# 명령줄 옵션 정의. 실제 읽기(parse_args)는 main 안에서 한다 (import만 할 때는 아무것도 실행되지 않게).
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
parser.add_argument(
    "--inference-batch-size",
    type=int,
    default=65536,
    help="Batch size for model inference on training data (default: 65536, same as the control chart recompute in Test_/Test_tsne)",
)
parser.add_argument(
    "--cache-strategy",
    type=str,
    default="disk",
    choices=["memory", "disk"],
    help="Caching strategy for training errors: 'memory' (faster, high RAM usage) or 'disk' (slower, saves RAM). Default: 'disk'",
)
parser.add_argument(
    "--resume",
    action="store_true",
    help="Resume from previous run. If true, attempts to reuse error files in temp_cache and skips already computed combinations in CSV.",
)
parser.add_argument(
    "--keep-cache",
    action="store_true",
    help="Keep the disk cache files (error matrices) after the simulation finishes. Useful for debugging or subsequent runs with --resume.",
)
parser.add_argument(
    "--use-float32",
    action="store_true",
    help="Convert error matrices to float32 (Single Precision) to save memory/disk (approx 50% reduction). Optional optimization.",
)
parser.add_argument(
    "--fast-search",
    action="store_true",
    help="Optimize search: 1. Find best U using last X model. 2. Sweep X models using best U. Reduces combinations from N*M to N+M.",
)
parser.add_argument(
    "--use-validate",
    action="store_true",
    help="Use validation set for model selection and evaluation.",
)
parser.add_argument(
    "--use-reverse-pos-neg",
    action="store_true",
    help="Use reverse positive-negative strategy for model selection and evaluation.",
)


def main(argv=None):
    """시뮬레이션 전체를 실행한다 (파일 맨 위 "하는 일" 1~5). argv는 명령줄 옵션 목록이며, None이면 실제 명령줄을 읽는다.

    `python Test_multi_models.py ...`로 실행하면 맨 아래의 `if __name__ == "__main__"`이 이 함수를 부른다.
    테스트(tests/test_scripts.py)는 Test_multi_models.main([...])처럼 옵션 목록을 직접 넘겨 부른다.
    """
    os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"  # 일부 환경에서 수학 라이브러리 중복 로드 오류를 피하는 설정
    warnings.filterwarnings("ignore")

    args = parser.parse_args(argv)

    run_dir = f"{args.models_dir}/{args.models_idx}"
    run = Run(run_dir)
    config = run.config

    def checkpoint_indices(model, option):
        """평가할 체크포인트 번호 목록: option이 "auto"면 artifact 폴더에서 찾고, 아니면 쉼표로 나눈 번호를 쓴다."""
        if option.lower() != "auto":
            return [int(idx.strip()) for idx in option.split(",")]
        print(f"Scanning artifact directory for {model} models: {run_dir}/artifact")
        indices = run.checkpoint_numbers(model)
        if indices:
            print(f"Found {model} model indices: {indices}")
        else:
            print(f"Warning: No {model} model checkpoint files found.")
        return indices

    u_model_indices = checkpoint_indices("U", args.u_model_idx)
    x_model_indices = checkpoint_indices("X", args.x_model_idx)
    if not u_model_indices or not x_model_indices:
        print("\nError: No valid model combinations to test. Exiting.")
        exit(1)

    # 학습 특징 캐시(temp_cache) 파일 이름에 들어가는 계산 조건
    cache_conditions = {
        "abs_err": not config.no_abs_err,
        "float32": args.use_float32,
        "batch_size": args.inference_batch_size,
    }

    os.makedirs(f"{run_dir}/results", exist_ok=True)
    print(f"Using dataset from: {config.split_dir}")

    print_sim_config(
        title="Multi-Model Test Run",
        config=args,
        extra={
            "device": "cpu",  # 평가는 CPU에서 한다 (Run이 CPU 모델을 돌려준다)
            "learning_case": config.learning_case,
            "PREPROCESSING": config.preprocessing,
            "SKIP_CHARGE_READY": config.skip_charge_ready,
            "no_use_dx": not config.use_dx,
            "add_one_output_layer": config.add_one_output_layer,
            "u_model_indices": u_model_indices,
            "x_model_indices": x_model_indices,
            "total_combinations": len(u_model_indices) * len(x_model_indices),
        },
    )

    try:
        plot_loss_curve_all(path=run_dir)
    except Exception as e:
        print(f"Error plotting loss curve: {e}")

    # 평가 차량. --use-reverse-pos-neg면 고장 차량을 먼저 둔다
    normal_list = split_ids(config.split_dir, config.battery_type, "test")
    if args.use_validate:
        normal_list += split_ids(config.split_dir, config.battery_type, "validate")
    fault_list = split_ids(config.split_dir, config.battery_type, "fault")
    test_list = [fault_list, normal_list] if args.use_reverse_pos_neg else [normal_list, fault_list]
    # (차량 ID, 고장 여부) 목록 (test_list 순서)
    test_vehicles = [(i, vehicle_ids is fault_list) for vehicle_ids in test_list for i in vehicle_ids]

    all_roc_results = []  # 조합별 ROC 결과 (마지막 비교 그림용)
    csv_path = f"{run_dir}/AUC_summary_case{config.learning_case}.csv"

    # --resume: CSV에 이미 있는 조합은 다시 계산하지 않는다
    completed_combinations = set()
    if args.resume and os.path.exists(csv_path):
        print(f"Resuming mode enabled. Checking cached results in {csv_path}...")
        try:
            existing_df = pd.read_csv(csv_path)
            if "u_model_idx" in existing_df.columns and "x_model_idx" in existing_df.columns:
                for _, row in existing_df.iterrows():
                    completed_combinations.add((int(row["u_model_idx"]), int(row["x_model_idx"])))
            print(
                f"  Found {len(completed_combinations)} already computed combinations. These will be skipped."
            )
        except Exception as e:
            print(f"Warning: Could not read existing CSV for resume: {e}")

    # 이어서 하지 않으면 예전 결과 CSV를 지우고 새로 쓴다
    if not args.resume and os.path.exists(csv_path):
        os.remove(csv_path)
        print(f"Removed previous AUC summary file: {csv_path}")

    print("\nPre-loading all models...")
    nets = {"U": {}, "X": {}}  # 모델 종류 → {체크포인트 번호: 모델}
    for model, indices in (("U", u_model_indices), ("X", x_model_indices)):
        for idx in indices:
            nets[model][idx] = run.load_model(model, idx)
            print(f"  Loaded {model} model: {idx}")
    print(
        f"All models loaded successfully. Total: {len(nets['U'])} U models, {len(nets['X'])} X models\n"
    )

    # disk 캐시에 모든 체크포인트의 학습 특징이 이미 있으면 학습 데이터를 읽지 않는다
    temp_cache_dir = os.path.join(run_dir, "temp_cache")
    all_files_exist = False
    if args.cache_strategy == "disk" and os.path.exists(temp_cache_dir):
        missing_u = [u for u in u_model_indices if not run.feature_cache_path("U", u, **cache_conditions).exists()]
        missing_x = [x for x in x_model_indices if not run.feature_cache_path("X", x, **cache_conditions).exists()]
        if not missing_u and not missing_x:
            all_files_exist = True
            print("All required feature files found in disk cache. Skipping training data loading.")
        else:
            print(f"Missing cached files: U={missing_u}, X={missing_x}. Will load training data.")

    train_data_cache = None  # [(U 텐서, X 텐서), ...]
    if not all_files_exist:
        print("Pre-loading training data...")
        train_list = split_ids(
            config.split_dir,
            config.battery_type,
            "train",
            start=config.vehicle_start,
            end=config.vehicle_end,
        )
        # source_data_dir의 .pt 캐시가 있으면 그것을 쓴다
        train_data_cache = load_training_vehicles(args.source_data_dir, config, train_list)
        total_samples = sum(t[0].shape[0] for t in train_data_cache)
        print(
            f"Training data loaded: {len(train_data_cache)} vehicles, {total_samples} total samples (Double precision)\n"
        )

    print("Pre-loading test data...")
    # {vehicle_id: (tensor, tensor_x)}. 캐시 키는 test_list 순서대로 펼친 차량 ID 목록이다
    test_data_cache = load_test_vehicles(args.source_data_dir, config, [i for i, _ in test_vehicles])
    print(f"Test data loaded: {len(test_data_cache)} vehicles\n")

    if args.cache_strategy == "disk":
        os.makedirs(temp_cache_dir, exist_ok=True)
        print(
            f"Using DISK cache for training FEATURES (not raw errors) at: {temp_cache_dir}"
            + (" (Resuming mode)" if args.resume else "")
        )
    else:
        print("Using MEMORY cache for training FEATURES (not raw errors) (Optimized RAM usage)")

    # 체크포인트마다 학습 진단 특징(자기 몫 3개)을 한 번만 계산한다: U 전체(Phase 1) → X 전체(Phase 2)
    memory_features = {"U": {}, "X": {}}  # memory 전략일 때 모델 종류 → {체크포인트 번호: 특징}
    for phase, model, indices in ((1, "U", u_model_indices), (2, "X", x_model_indices)):
        print(f"\nPhase {phase}: Computing and caching {model} FEATURES to {args.cache_strategy}...")
        for idx in indices:
            if args.cache_strategy == "disk" and run.feature_cache_path(model, idx, **cache_conditions).exists():
                print(f"  Found cached {model} features for model {idx}, skipping computation.")
                continue
            if train_data_cache is None:
                raise RuntimeError("Training data not loaded but feature computation needed! (Cache inconsistency)")

            print(f"  Computing training features for {model} model {idx} (Per-vehicle processing)...")
            column = 0 if model == "U" else 1  # (U 텐서, X 텐서) 중 이 모델의 입력
            features = training_signal_features(
                nets[model][idx],
                model,
                [pair[column] for pair in train_data_cache],
                batch_size=args.inference_batch_size,
                abs_err=cache_conditions["abs_err"],
                float32=args.use_float32,
            )
            if args.cache_strategy == "disk":
                save_cached_features(run, model, idx, features, conditions=cache_conditions)
            else:
                memory_features[model][idx] = features

    # 관리도 계산 전에 학습 데이터를 메모리에서 내린다
    print("Releasing training data from memory...")
    train_data_cache = None

    def process_combination(u_idx, x_idx):
        """체크포인트 조합 하나(U 번호 u_idx, X 번호 x_idx)를 평가하고 그 AUC를 돌려준다. main 안에서만 쓴다.

        학습 진단 특징(캐시 또는 메모리)으로 관리도를 만들고, 평가 차량마다 최대 CI를 구해 ROC/AUC를 계산한다.
        결과는 CSV에 한 줄 추가하고, 조합별 ROC 그림을 저장하고, 전체 비교 그림용 목록(all_roc_results)에 넣는다.
        --resume으로 이미 CSV에 있는 조합이면 계산하지 않고 건너뛴다(--fast-search면 CSV의 AUC를 읽어 돌려준다).
        """
        if (u_idx, x_idx) in completed_combinations:
            print(f"Skipping cached combination: u={u_idx}, x={x_idx}")
            # --fast-search는 1단계 AUC로 최고 U를 고르므로, 건너뛴 조합은 CSV에서 AUC를 읽는다
            if args.fast_search and os.path.exists(csv_path):
                existing_df = pd.read_csv(csv_path)
                match = existing_df[
                    (existing_df["u_model_idx"] == u_idx) & (existing_df["x_model_idx"] == x_idx)
                ]
                if not match.empty:
                    return float(match.iloc[0]["AUC"])
            return 0.0

        print(f"\n{'='*80}")
        print(f"Testing model combination: u_model_idx={u_idx}, x_model_idx={x_idx}")
        print(f"{'='*80}\n")

        start_pca_load = time.perf_counter()
        if args.cache_strategy == "disk":
            print("  Loading pre-computed FEATURES from disk...")
            u_feat = load_cached_features(run, "U", u_idx, conditions=cache_conditions)
            x_feat = load_cached_features(run, "X", x_idx, conditions=cache_conditions)
        else:
            print("  Using pre-computed FEATURES from memory...")
            u_feat = memory_features["U"][u_idx]
            x_feat = memory_features["X"][x_idx]

        # 학습 특징으로 관리도를 만든다 (관리도는 학습 특징 행렬을 들고 있지 않는다)
        chart = ControlChart.fit(combine_signal_features(u_feat, x_feat))
        del u_feat, x_feat
        elapsed_pca_load = time.perf_counter() - start_pca_load
        print(f"PCA results loading/computing time: {elapsed_pca_load:.3f} seconds")

        net_loaded = nets["U"][u_idx]
        netx_loaded = nets["X"][x_idx]

        vehicle_max_ci = []  # 차량별 최대 CI (test_vehicles 순서)
        vehicle_is_fault = []
        start = time.perf_counter()
        for n, (vehicle_id, is_fault) in enumerate(test_vehicles):
            if n % 10 == 0:
                elapsed = time.perf_counter() - start
                h, rem = divmod(elapsed, 3600)
                m, s = divmod(rem, 60)
                print(
                    f"Processing vehicle {n}/{len(test_vehicles)} | Elapsed: {int(h)}h {int(m)}m {s:.1f}s"
                )

            tensor, tensor_x = test_data_cache[vehicle_id]
            # 차량 한 대의 진단 특징 (차량 전체를 한 번에 추론, 오차의 abs 여부는 학습 진단 특징과 같은 설정을 따른다)
            df_data = vehicle_features(
                net_loaded, netx_loaded, tensor, tensor_x, abs_err=cache_conditions["abs_err"]
            )
            vehicle_max_ci.append(chart.max_ci(df_data))
            vehicle_is_fault.append(is_fault)

        elapsed = time.perf_counter() - start
        h, rem = divmod(elapsed, 3600)
        m, s = divmod(rem, 60)
        print(
            f"\nCombination (u={u_idx}, x={x_idx}) completed: {int(h)}h {int(m)}m {s:.3f}s\n"
        )

        # reverse이면 정상 차량을 양성으로 보고 최대 CI < 임계값을 양성으로 판정한다
        result = evaluate(vehicle_max_ci, vehicle_is_fault, reverse=args.use_reverse_pos_neg)
        best, closest = result.best, result.closest_to_01
        print(f"AUC = {result.auc:.4f}")

        roc_save_path = f"{run_dir}/results/AUC_ROC_u{u_idx}_x{x_idx}_case{config.learning_case}.png"
        plot_roc_curve(
            result.fpr,
            result.tpr,
            result.auc,
            best_point=best,
            closest_to_01_point=closest,
            save_path=roc_save_path,
            title=f"AUC-ROC Curve (u_model={u_idx}, x_model={x_idx})",
            show=False,
        )
        print(f"ROC curve saved to: {roc_save_path}\n")

        all_roc_results.append(
            {
                "u_idx": u_idx,
                "x_idx": x_idx,
                "fpr": result.fpr,
                "tpr": result.tpr,
                "auc": result.auc,
                "best": best,
            }
        )

        # 결과를 바로 CSV에 한 줄 추가한다 (중간에 멈춰도 --resume으로 이어 갈 수 있게)
        auc_record = pd.DataFrame(
            [
                {
                    "u_model_idx": u_idx,
                    "x_model_idx": x_idx,
                    "AUC": result.auc,
                    "best_threshold": best["threshold"],
                    "best_TPR": best["tpr"],
                    "best_FPR": best["fpr"],
                    "closest_01_threshold": closest["threshold"],
                    "closest_01_TPR": closest["tpr"],
                    "closest_01_FPR": closest["fpr"],
                    "closest_01_distance": closest["distance"],
                }
            ]
        )
        if not os.path.exists(csv_path):
            auc_record.to_csv(csv_path, mode="w", header=True, index=False)
        else:
            auc_record.to_csv(csv_path, mode="a", header=False, index=False)

        completed_combinations.add((u_idx, x_idx))
        return result.auc

    if args.fast_search:
        # 1단계: 마지막 X 체크포인트로 모든 U를 평가해 최고 U를 고른다
        print(f"\n{'='*80}")
        print("FAST SEARCH MODE ENABLED")
        print("Step 1: Finding best U model using the last X model...")
        print(f"{'='*80}\n")
        combinations = [(u, x_model_indices[-1]) for u in u_model_indices]
    else:
        # 모든 조합 (X 순서로 바깥, U 순서로 안쪽)
        combinations = [(u_idx, x_idx) for x_idx in x_model_indices for u_idx in u_model_indices]

    step1_aucs = [(u_idx, process_combination(u_idx, x_idx)) for u_idx, x_idx in combinations]

    if args.fast_search:
        # 2단계: 최고 U로 모든 X를 평가한다 ((최고 U, 마지막 X)는 1단계에서 이미 했다)
        best_u_idx, best_auc = max(step1_aucs, key=lambda item: item[1])
        print(f"\n{'='*80}")
        print(f"Step 1 Complete. Best U model: u={best_u_idx} (AUC={best_auc:.4f})")
        print("Step 2: Sweeping all X models using the best U model...")
        print(f"{'='*80}\n")
        for x_idx in x_model_indices:
            if (best_u_idx, x_idx) not in completed_combinations:
                process_combination(best_u_idx, x_idx)

    if args.cache_strategy == "disk" and os.path.exists(temp_cache_dir):
        if args.keep_cache:
            print(f"Keeping temporary disk cache at: {temp_cache_dir} (--keep-cache enabled)")
        else:
            print(f"Cleaning up temporary disk cache: {temp_cache_dir}")
            shutil.rmtree(temp_cache_dir)

    # 두 조합 이상이면 모든 ROC 곡선을 한 그림에 모으고, 요약 표와 AUC 히트맵을 남긴다
    if len(all_roc_results) > 1:
        print(f"\n{'='*80}")
        print("Plotting all ROC curves on a single figure...")
        print(f"{'='*80}\n")

        fig, ax = plt.subplots(figsize=(10, 8))
        for result in all_roc_results:
            label = f"u={result['u_idx']}, x={result['x_idx']} (AUC={result['auc']:.4f})"
            ax.plot(result["fpr"], result["tpr"], linewidth=2, label=label)
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

        combined_roc_path = f"{run_dir}/results/AUC_ROC_ALL_COMBINATIONS_case{config.learning_case}.png"
        plt.tight_layout()
        plt.savefig(combined_roc_path, dpi=200, bbox_inches="tight")
        plt.close()
        print(f"Combined ROC curve saved to: {combined_roc_path}")

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

        draw_auc_heatmap(csv_path, f"{run_dir}/AUC_heatmap_case{config.learning_case}.png")

    print("\n✓ All model combinations tested successfully!")


if __name__ == "__main__":
    main()
