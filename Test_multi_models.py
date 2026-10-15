import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
import pandas as pd
import matplotlib.ticker as mtick
import os
import warnings
import matplotlib
import torch

# from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split
import argparse
import time
from pathlib import Path
from mcdl.run import Run, print_sim_config
from mcdl.diagnosis import ControlChart, combine_signal_features
from mcdl.reconstruction import training_signal_features, vehicle_features
from mcdl.chart_source import load_cached_features, save_cached_features
from mcdl.roc import evaluate
from mcdl.plotting import draw_auc_heatmap, plot_loss_curve_all, plot_roc_curve
from mcdl.vehicles import load_test_vehicles, load_training_vehicles, split_ids

device = torch.device("cpu")  # 평가는 CPU 전용 (Run이 CPU 모델을 돌려준다)

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
parser.add_argument("--x-start", type=int, default=20)
parser.add_argument("--x-tick-step", type=int, default=3000)
parser.add_argument("--sigma-levels", type=str, default="3,4.5,6")
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
    os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
    print(torch.cuda.device_count())

    warnings.filterwarnings("ignore")

    args = parser.parse_args(argv)


    run = Run(f"{args.models_dir}/{args.models_idx}")

    # Parse model indices
    artifact_dir = f"{args.models_dir}/{args.models_idx}/artifact"

    if args.u_model_idx.lower() == "auto":
        print(f"Scanning artifact directory for U models: {artifact_dir}")
        u_model_indices = run.checkpoint_numbers("U")
        if not u_model_indices:
            print("Warning: No valid U model files found (excluding -1 and 0).")
            u_model_indices = []
        else:
            print(f"Found U model indices: {u_model_indices}")
    else:
        u_model_indices = [int(idx.strip()) for idx in args.u_model_idx.split(",")]

    if args.x_model_idx.lower() == "auto":
        print(f"Scanning artifact directory for X models: {artifact_dir}")
        x_model_indices = run.checkpoint_numbers("X")
        if not x_model_indices:
            print("Warning: No valid X model files found (excluding -1 and 0).")
            x_model_indices = []
        else:
            print(f"Found X model indices: {x_model_indices}")
    else:
        x_model_indices = [int(idx.strip()) for idx in args.x_model_idx.split(",")]

    # Exit if no valid models found
    if not u_model_indices or not x_model_indices:
        print("\nError: No valid model combinations to test. Exiting.")
        exit(1)

    learning_case = run.config.learning_case
    add_one_output_layer = run.config.add_one_output_layer
    use_dx = run.config.use_dx
    # 학습 특징 캐시(temp_cache) 파일 이름에 들어가는 계산 조건
    cache_conditions = {
        "abs_err": not run.config.no_abs_err,
        "float32": args.use_float32,
        "batch_size": args.inference_batch_size,
    }

    # 플롯 결과 저장 폴더는 한 번만 생성
    os.makedirs(f"{args.models_dir}/{args.models_idx}/results", exist_ok=True)

    BATTERY_TYPE = run.config.battery_type  # 'DTI' or 'QAS'
    PREPROCESSING, SKIP_CHARGE_READY = run.config.preprocessing, run.config.skip_charge_ready

    # Determine Dataset Path based on trained model config
    DATA_BASE_PATH = run.config.split_dir
    print(f"Using dataset from: {DATA_BASE_PATH}")

    print_sim_config(
        title="Multi-Model Test Run",
        config=args,
        extra={
            "device": str(device),
            "learning_case": learning_case,
            "PREPROCESSING": PREPROCESSING,
            "SKIP_CHARGE_READY": SKIP_CHARGE_READY,
            "no_use_dx": not use_dx,
            "add_one_output_layer": add_one_output_layer,
            "u_model_indices": u_model_indices,
            "x_model_indices": x_model_indices,
            "total_combinations": len(u_model_indices) * len(x_model_indices),
        },
    )

    try:
        plot_loss_curve_all(path=f"{args.models_dir}/{args.models_idx}")
    except Exception as e:
        print(f"Error plotting loss curve: {e}")

    ## =================== Load test data lists =================== ##
    normal_test = split_ids(DATA_BASE_PATH, BATTERY_TYPE, "test")
    normal_validate = split_ids(DATA_BASE_PATH, BATTERY_TYPE, "validate")
    normal_list = [*normal_test, *normal_validate] if args.use_validate else normal_test
    fault_list = split_ids(DATA_BASE_PATH, BATTERY_TYPE, "fault")

    test_list = (
        [normal_list, fault_list]
        if not args.use_reverse_pos_neg
        else [fault_list, normal_list]
    )

    ## =================== Initialize params for AUROC curve  =================== ##
    total_test_vehicles = len(normal_list) + len(fault_list)

    # Store results for all model combinations
    all_roc_results = []

    # Define CSV path for incremental saving
    csv_path = f"{args.models_dir}/{args.models_idx}/AUC_summary_case{learning_case}.csv"

    completed_combinations = set()

    # Initialize completed combinations set if resuming
    if args.resume and os.path.exists(csv_path):
        print(f"Resuming mode enabled. Checking cached results in {csv_path}...")
        try:
            existing_df = pd.read_csv(csv_path)
            # Check if columns exist
            if (
                "u_model_idx" in existing_df.columns
                and "x_model_idx" in existing_df.columns
            ):
                for _, row in existing_df.iterrows():
                    completed_combinations.add(
                        (int(row["u_model_idx"]), int(row["x_model_idx"]))
                    )
            print(
                f"  Found {len(completed_combinations)} already computed combinations. These will be skipped."
            )
        except Exception as e:
            print(f"Warning: Could not read existing CSV for resume: {e}")

    # Clear existing CSV file if it exists to start fresh (ONLY if not resuming)
    if not args.resume and os.path.exists(csv_path):
        os.remove(csv_path)
        print(f"Removed previous AUC summary file: {csv_path}")

    # Pre-load all models to avoid repeated loading
    print("\nPre-loading all models...")
    models_u = {}
    models_x = {}

    for u_idx in u_model_indices:
        models_u[u_idx] = run.load_model("U", u_idx)
        print(f"  Loaded U model: {u_idx}")

    for x_idx in x_model_indices:
        models_x[x_idx] = run.load_model("X", x_idx)
        print(f"  Loaded X model: {x_idx}")


    print(
        f"All models loaded successfully. Total: {len(models_u)} U models, {len(models_x)} X models\n"
    )

    # ================= Check if all error/feature files already exist ================= #
    # If we are using disk cache and all files exist, we can skip loading combined_tensor_train.
    all_files_exist = False
    if args.cache_strategy == "disk":
        temp_cache_dir_check = os.path.join(
            f"{args.models_dir}/{args.models_idx}", "temp_cache"
        )
        if os.path.exists(temp_cache_dir_check):
            missing_u = []
            for u_idx in u_model_indices:
                fpath = run.feature_cache_path("U", u_idx, **cache_conditions)
                if not os.path.exists(fpath):
                    missing_u.append(u_idx)

            missing_x = []
            for x_idx in x_model_indices:
                fpath = run.feature_cache_path("X", x_idx, **cache_conditions)
                if not os.path.exists(fpath):
                    missing_x.append(x_idx)

            if not missing_u and not missing_x:
                all_files_exist = True
                print(
                    "All required feature files found in disk cache. Skipping training data loading."
                )
            else:
                print(
                    f"Missing cached files: U={missing_u}, X={missing_x}. Will load training data."
                )

    # Pre-load training data once to avoid repeated loading per model
    if not all_files_exist:
        print("Pre-loading training data...")
        train_list = split_ids(
            DATA_BASE_PATH,
            BATTERY_TYPE,
            "train",
            start=run.config.vehicle_start,
            end=run.config.vehicle_end,
        )
        # List of (tensor, tensorx). source_data_dir의 .pt 캐시가 있으면 그것을 쓴다
        train_data_cache = load_training_vehicles(args.source_data_dir, run.config, train_list)

        total_samples = sum(t[0].shape[0] for t in train_data_cache)
        print(
            f"Training data loaded: {len(train_data_cache)} vehicles, {total_samples} total samples (Double precision)\n"
        )
    else:
        # Just needed to skip the loading block
        pass

    # Pre-load all test data once to avoid repeated loading
    print("Pre-loading test data...")
    # {vehicle_id: (tensor, tensor_x)}. 캐시 키는 test_list 순서대로 펼친 차량 ID 목록이다
    flat_test_list = [i for vehicle_ids in test_list for i in vehicle_ids]
    test_data_cache = load_test_vehicles(args.source_data_dir, run.config, flat_test_list)

    print(f"Test data loaded: {len(test_data_cache)} vehicles\n")

    # Initialize caches for training errors to avoid redundant computations
    import tempfile
    import shutil

    # Caching Strategy Setup
    cache_strategy = args.cache_strategy
    temp_cache_dir = None
    u_error_cache = {}  # Used for direct memory storage
    x_error_cache = {}

    if cache_strategy == "disk":
        # Create a fixed temporary directory for disk caching (easier manual cleanup)
        temp_cache_dir = os.path.join(f"{args.models_dir}/{args.models_idx}", "temp_cache")

        # Clean up previous cache if it exists (e.g. from a crashed run), unless resuming
        # if os.path.exists(temp_cache_dir) and not args.resume:
        #     try:
        #         shutil.rmtree(temp_cache_dir)
        #         print(f"Cleared stale cache directory: {temp_cache_dir}")
        #     except OSError as e:
        #         print(f"Warning: Could not clear old cache directory {temp_cache_dir}: {e}")

        os.makedirs(temp_cache_dir, exist_ok=True)
        if args.resume:
            print(
                f"Using DISK cache for training FEATURES (not raw errors) at: {temp_cache_dir} (Resuming mode)"
            )
        else:
            print(
                f"Using DISK cache for training FEATURES (not raw errors) at: {temp_cache_dir}"
            )
    else:
        print(
            "Using MEMORY cache for training FEATURES (not raw errors) (Optimized RAM usage)"
        )

    # Use abs error based on sim_config
    use_abs_err = not run.config.no_abs_err
    inference_batch_size = args.inference_batch_size

    # Phase 1: Compute and Cache U FEATURES (Extracted from errors)
    print(f"\nPhase 1: Computing and caching U FEATURES to {cache_strategy}...")
    for u_idx in u_model_indices:
        # Check for existing cache if resuming OR if we determined all files exist (which implies we should respect them even if resume flag not explicitly set, though likely it is set if we are here)
        # Actually, if all_files_exist is True, it means we found everything. We should just verify specific file exists.

        fpath = run.feature_cache_path("U", u_idx, **cache_conditions)
        if cache_strategy == "disk" and os.path.exists(fpath):
            print(f"  Found cached U features for model {u_idx}, skipping computation.")
            continue

        # If we are here, we MUST compute.
        # If 'all_files_exist' was True but we are here, something is wrong (file deleted in between?) OR we are not using disk cache?
        # Whatever, if we are here, we need train_data_cache.
        # If all_files_exist was True, train_data_cache is NOT loaded.
        if "train_data_cache" not in locals() or train_data_cache is None:
            raise RuntimeError(
                "Training data not loaded but feature computation needed! (Cache inconsistency)"
            )

        print(
            f"  Computing training features for U model {u_idx} (Per-vehicle processing)..."
        )

        # 차량마다 재구성 오차 → 신호별 특징을 구해 이어 붙인다
        feat_u_final = training_signal_features(
            models_u[u_idx],
            "U",
            [t for t, _ in train_data_cache],
            batch_size=inference_batch_size,
            abs_err=use_abs_err,
            float32=args.use_float32,
        )

        if cache_strategy == "disk":
            save_cached_features(run, "U", u_idx, feat_u_final, conditions=cache_conditions)
        else:
            # Keep in memory
            u_error_cache[u_idx] = feat_u_final

    # Phase 2: Compute and Cache X FEATURES
    print(f"\nPhase 2: Computing and caching X FEATURES to {cache_strategy}...")
    for x_idx in x_model_indices:
        fpath = run.feature_cache_path("X", x_idx, **cache_conditions)
        if cache_strategy == "disk" and os.path.exists(fpath):
            print(f"  Found cached X features for model {x_idx}, skipping computation.")
            continue

        if "train_data_cache" not in locals() or train_data_cache is None:
            raise RuntimeError("Training data not loaded but feature computation needed!")

        print(
            f"  Computing training features for X model {x_idx} (Per-vehicle processing)..."
        )

        # 차량마다 재구성 오차 → 신호별 특징을 구해 이어 붙인다
        feat_x_final = training_signal_features(
            models_x[x_idx],
            "X",
            [tx for _, tx in train_data_cache],
            batch_size=inference_batch_size,
            abs_err=use_abs_err,
            float32=args.use_float32,
        )

        if cache_strategy == "disk":
            save_cached_features(run, "X", x_idx, feat_x_final, conditions=cache_conditions)
        else:
            # Keep in memory
            x_error_cache[x_idx] = feat_x_final

    # Critical: Release original training data to free up RAM before PCA loop
    print("Releasing training data from memory...")
    if "train_data_cache" in locals() and train_data_cache is not None:
        del train_data_cache
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # Determine the list of combinations to process based on fast-search strategy or full search
    combinations_to_run = []

    if args.fast_search and len(x_model_indices) > 0 and len(u_model_indices) > 0:
        print(f"\n{'='*80}")
        print("FAST SEARCH MODE ENABLED")
        print("Step 1: Finding best U model using the last X model...")
        print(f"{'='*80}\n")

        last_x_idx = x_model_indices[-1]

        # Step 1: Run all U models against the last X model
        step1_results = []

        # Check if we already have results for this step to avoid re-calculation if possible,
        # but we need the AUC values to sort. We'll rely on the resume logic inside the loop
        # or just let the main loop handle it, but here we need to enforce the order.

        # We will generate a list of (u, x) tuples to run in order.
        # However, fast search requires the RESULT of step 1 to proceed to step 2.
        # So we cannot pre-generate the full list. We must run the loop in two phases.

        # PHASE 1 LOOP
        combinations_phase1 = [(u, last_x_idx) for u in u_model_indices]
    else:
        # Standard full grid search
        combinations_phase1 = [
            (u_idx, x_idx) for x_idx in x_model_indices for u_idx in u_model_indices
        ]


    # Function to process a single combination (refactored from original loop)
    def process_combination(u_idx, x_idx):
        # SKIP LOGIC for RESUME (if result already in CSV)
        if (u_idx, x_idx) in completed_combinations:
            print(f"Skipping cached combination: u={u_idx}, x={x_idx}")
            # We need to retrieve the AUC if we are in fast search mode to determine the winner
            # Try to read from all_roc_results if available (populated from CSV earlier?)
            # Since we didn't populate all_roc_results from CSV, we might miss it.
            # But for fast search Step 1, we critically need the AUC.
            # If resuming, we should ideally read the AUC from the CSV file row.

            auc_val = 0.0
            if args.fast_search:
                # Try to find AUC in existing CSV data
                try:
                    if os.path.exists(csv_path):
                        existing_df = pd.read_csv(csv_path)
                        match = existing_df[
                            (existing_df["u_model_idx"] == u_idx)
                            & (existing_df["x_model_idx"] == x_idx)
                        ]
                        if not match.empty:
                            auc_val = float(match.iloc[0]["AUC"])
                except:
                    pass
            return auc_val

        print(f"\n{'='*80}")
        print(f"Testing model combination: u_model_idx={u_idx}, x_model_idx={x_idx}")
        print(f"{'='*80}\n")

        # 차량별 최대 CI와 고장 여부 (test_list 순서)
        vehicle_max_ci = []
        vehicle_is_fault = []

        ## =================== Load PCA data for each combination  =================== ##
        start_pca_load = time.perf_counter()

        # Load cached features (fast!)
        if cache_strategy == "disk":
            print("  Loading pre-computed FEATURES from disk...")
            u_feat = load_cached_features(run, "U", u_idx, conditions=cache_conditions)
            x_feat = load_cached_features(run, "X", x_idx, conditions=cache_conditions)
        else:
            print("  Using pre-computed FEATURES from memory...")
            u_feat = u_error_cache[u_idx]
            x_feat = x_error_cache[x_idx]

        # Fit the control chart on the cached training features (관리도는 학습 특징 행렬을 들고 있지 않는다)
        chart = ControlChart.fit(combine_signal_features(u_feat, x_feat))
        del u_feat, x_feat
        elapsed_pca_load = time.perf_counter() - start_pca_load
        print(f"PCA results loading/computing time: {elapsed_pca_load:.3f} seconds")

        # Use pre-loaded models
        net_loaded = models_u[u_idx]
        netx_loaded = models_x[x_idx]

        start = time.perf_counter()
        test_idx = 0
        for vehicle_ids in test_list:
            for i in vehicle_ids:
                if test_idx % 10 == 0:
                    elapsed = time.perf_counter() - start
                    h, rem = divmod(elapsed, 3600)
                    m, s = divmod(rem, 60)
                    print(
                        f"Processing vehicle {test_idx}/{total_test_vehicles} | Elapsed: {int(h)}h {int(m)}m {s:.1f}s"
                    )

                # Use pre-loaded test data
                tensor, tensor_x = test_data_cache[i]

                # 차량 한 대의 진단 특징 (차량 전체를 한 번에 추론, 오차의 abs 여부는 학습 진단 특징과 같은 설정을 따른다)
                df_data = vehicle_features(net_loaded, netx_loaded, tensor, tensor_x, abs_err=use_abs_err)

                ## =================== Testing Diagnosis =================== ##
                vehicle_max_ci.append(chart.max_ci(df_data))
                vehicle_is_fault.append(vehicle_ids is fault_list)
                test_idx += 1

        elapsed = time.perf_counter() - start
        h, rem = divmod(elapsed, 3600)
        m, s = divmod(rem, 60)
        print(
            f"\nCombination (u={u_idx}, x={x_idx}) completed: {int(h)}h {int(m)}m {s:.3f}s\n"
        )

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        # =================== ROC / AUC ===================
        # reverse이면 정상 차량을 양성으로 보고 최대 CI < 임계값을 양성으로 판정한다
        result = evaluate(vehicle_max_ci, vehicle_is_fault, reverse=args.use_reverse_pos_neg)
        best, closest = result.best, result.closest_to_01

        print(f"AUC = {result.auc:.4f}")

        # Save individual ROC curve
        roc_save_path = f"{args.models_dir}/{args.models_idx}/results/AUC_ROC_u{u_idx}_x{x_idx}_case{learning_case}.png"
        plot_roc_curve(
            result.fpr,
            result.tpr,
            result.auc,
            best_point=best,
            closest_to_01_point=closest,
            save_path=roc_save_path,
            title=f"AUC-ROC Curve (u_model={u_idx}, x_model={x_idx})",
            figsize=(6, 6),
            show=False,
            dpi=200,
        )
        print(f"ROC curve saved to: {roc_save_path}\n")

        # Store results for combined plot
        all_roc_results.append(
            {
                "u_idx": u_idx,
                "x_idx": x_idx,
                "fpr": result.fpr,
                "tpr": result.tpr,
                "auc": result.auc,
                "best": best,
                "closest_to_01": closest,
            }
        )

        # =================== Save AUC result immediately to CSV =================== ##
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

        # Mark as completed
        completed_combinations.add((u_idx, x_idx))

        return result.auc


    # Run Logic
    best_u_idx = None

    # Phase 1
    for u_idx, x_idx in combinations_phase1:
        auc = process_combination(u_idx, x_idx)
        # Track best U if in fast search mode
        if args.fast_search and "step1_results" in locals():
            step1_results.append((u_idx, auc))

    # Phase 2 (Only for Fast Search)
    if args.fast_search and step1_results:
        # Find best U
        best_u_tuple = max(step1_results, key=lambda item: item[1])
        best_u_idx = best_u_tuple[0]
        best_auc = best_u_tuple[1]

        print(f"\n{'='*80}")
        print(f"Step 1 Complete. Best U model: u={best_u_idx} (AUC={best_auc:.4f})")
        print("Step 2: Sweeping all X models using the best U model...")
        print(f"{'='*80}\n")

        # Remove duplicates if last_x_idx search overlaps (it's already done)
        # But usually we just sweep all X for this U
        combinations_phase2 = [
            (best_u_idx, x) for x in x_model_indices if x != combinations_phase1[0][1]
        ]

        # In fast search phase 1 we did all U against LAST X.
        # Now we do BEST U against ALL X.
        # Note: (best_u_idx, last_x_idx) is already done in phase 1.

        last_x_idx = x_model_indices[-1]
        combinations_phase2 = [
            (best_u_idx, x)
            for x in x_model_indices
            if (best_u_idx, x) not in completed_combinations
        ]

        for u_idx, x_idx in combinations_phase2:
            process_combination(u_idx, x_idx)

    original_loop_code_marker = (
        False  # Just a marker to ensure we broke the original loop flow structure
    )


    # Cleanup temp files if disk strategy was used and keep_cache is False
    if cache_strategy == "disk" and temp_cache_dir and os.path.exists(temp_cache_dir):
        if args.keep_cache:
            print(
                f"Keeping temporary disk cache at: {temp_cache_dir} (--keep-cache enabled)"
            )
        else:
            print(f"Cleaning up temporary disk cache: {temp_cache_dir}")
            shutil.rmtree(temp_cache_dir)

    if cache_strategy == "memory":
        # Clear memory cache explicitly
        u_error_cache.clear()
        x_error_cache.clear()
        del u_error_cache, x_error_cache


    # =================== Plot all ROC curves on one figure =================== ##
    if len(all_roc_results) > 1:
        print(f"\n{'='*80}")
        print("Plotting all ROC curves on a single figure...")
        print(f"{'='*80}\n")

        fig, ax = plt.subplots(figsize=(10, 8))

        # Plot each ROC curve
        for result in all_roc_results:
            label = f"u={result['u_idx']}, x={result['x_idx']} (AUC={result['auc']:.4f})"
            ax.plot(result["fpr"], result["tpr"], linewidth=2, label=label)

        # Plot diagonal reference line
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

        combined_roc_path = f"{args.models_dir}/{args.models_idx}/results/AUC_ROC_ALL_COMBINATIONS_case{learning_case}.png"
        plt.tight_layout()
        plt.savefig(combined_roc_path, dpi=200, bbox_inches="tight")
        plt.close()

        print(f"Combined ROC curve saved to: {combined_roc_path}")

        # Print summary table
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

        # =================== Create AUC heatmap visualization =================== ##
        heatmap_path = (
            f"{args.models_dir}/{args.models_idx}/AUC_heatmap_case{learning_case}.png"
        )
        draw_auc_heatmap(csv_path, heatmap_path)

    print("\n✓ All model combinations tested successfully!")


if __name__ == "__main__":
    main()
