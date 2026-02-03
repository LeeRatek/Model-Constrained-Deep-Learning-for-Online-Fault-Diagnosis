import argparse
import csv
import os
import re
import time

import numpy as np
import torch

from Function_ import (
    DiagnosisFeature,
    SPE_array,
    T2_array,
    compute_roc_auc_from_threshold_matrix,
    get_input_dimensions,
    get_preprocessing_and_skip_charge_ready,
    load_net_state_dict,
    load_pca_results,
    load_vehicle_ids_used_for_training,
    preprocess_combined_tensor,
    read_learning_case_from_sim_config,
    safe_load,
)
from Class_ import CombinedAE, custom_activation


def _find_checkpoints(models_dir: str):
    net_pat = re.compile(r"^net_e(\d+)\.pth$", re.IGNORECASE)
    netx_pat = re.compile(r"^netx_e(\d+)\.pth$", re.IGNORECASE)

    net_map = {}
    netx_map = {}

    for name in os.listdir(models_dir):
        m = net_pat.match(name)
        if m:
            net_map[int(m.group(1))] = name
            continue
        m = netx_pat.match(name)
        if m:
            netx_map[int(m.group(1))] = name

    return net_map, netx_map


def _build_models(*, battery_type: str, learning_case: int, device):
    preprocessing, skip_charge_ready = get_preprocessing_and_skip_charge_ready(learning_case)
    dim_dict = get_input_dimensions(battery_type)

    if preprocessing:
        net = CombinedAE(
            input_size=dim_dict["x"],
            encode2_input_size=dim_dict["q"],
            output_size=dim_dict["y"],
            activation_fn=torch.sigmoid,
            use_dx_in_forward=True,
        ).to(device)
    else:
        net = CombinedAE(
            input_size=dim_dict["x"],
            encode2_input_size=dim_dict["q"],
            output_size=dim_dict["y"],
            activation_fn=custom_activation,
            use_dx_in_forward=True,
        ).to(device)

    netx = CombinedAE(
        input_size=dim_dict["x2"],
        encode2_input_size=dim_dict["q2"],
        output_size=dim_dict["y2"],
        activation_fn=torch.sigmoid,
        use_dx_in_forward=True,
    ).to(device)

    return net, netx, dim_dict, preprocessing, skip_charge_ready


def _iter_test_vehicles(battery_type: str, models_dir: str):
    sim_config_path = os.path.join(models_dir, "sim_config.txt")
    train_vehicle_ids = load_vehicle_ids_used_for_training(sim_config_path)

    first_fault_vehicle_id = 335 if battery_type == "QAS" else 77
    all_normal_vehicle_ids = list(range(0, first_fault_vehicle_id))
    normal_list = [vid for vid in all_normal_vehicle_ids if vid not in set(train_vehicle_ids)]
    fault_list = np.load(f"./{battery_type}_filtered_vehicle_ids_fault.npy").astype(np.int64).tolist()

    # label: normal=0, fault=1
    for vid in normal_list:
        yield int(vid), 0
    for vid in fault_list:
        yield int(vid), 1


def _preprocess_one_vehicle(
    *,
    vehicle_id: int,
    label: int,
    battery_type: str,
    source_data_dir: str,
    dim_dict,
    preprocessing: bool,
    skip_charge_ready: bool,
    normalize_dx: bool,
    normalize_val: int,
):
    vid = f"{vehicle_id}"
    combined_tensor = safe_load(f"{source_data_dir}/{battery_type}/{vid}/vin_2.pkl", verbose=False)
    combined_tensorx = safe_load(f"{source_data_dir}/{battery_type}/{vid}/vin_3.pkl", verbose=False)
    combined_tensor, combined_tensorx = preprocess_combined_tensor(
        combined_tensor,
        combined_tensorx,
        dim_dict,
        battery_type,
        preprocessing,
        skip_charge_ready,
        normalize_dx=normalize_dx,
        normalize_val=normalize_val,
    )

    # split (torch tensor)
    x = combined_tensor[:, : dim_dict["x"]]
    y = combined_tensor[:, dim_dict["x"] : dim_dict["x"] + dim_dict["y"]]
    z = combined_tensor[:, dim_dict["x"] + dim_dict["y"] : dim_dict["x"] + dim_dict["y"] + dim_dict["z"]]
    q = combined_tensor[:, dim_dict["x"] + dim_dict["y"] + dim_dict["z"] :]

    x2 = combined_tensorx[:, : dim_dict["x2"]]
    y2 = combined_tensorx[:, dim_dict["x2"] : dim_dict["x2"] + dim_dict["y2"]]
    z2 = combined_tensorx[:, dim_dict["x2"] + dim_dict["y2"] : dim_dict["x2"] + dim_dict["y2"] + dim_dict["z2"]]
    q2 = combined_tensorx[:, dim_dict["x2"] + dim_dict["y2"] + dim_dict["z2"] :]

    return {
        "vehicle_id": vehicle_id,
        "label": int(label),
        "x": x,
        "y": y,
        "z": z,
        "q": q,
        "x2": x2,
        "y2": y2,
        "z2": z2,
        "q2": q2,
    }


def _compute_auc_for_checkpoint(
    *,
    net,
    netx,
    dim_dict,
    pca_pack,
    device,
    items,
    predict_thresholds,
):
    (v_I, v, v_ratio, p_k, data_mean, data_std,
        T_95_limit, T_99_limit, SPE_95_limit, SPE_99_limit,
        P, k, P_t, X, data_nor) = pca_pack

    ci_max_list = []
    y_true = []

    net = net.double().eval()
    netx = netx.double().eval()

    with torch.inference_mode():
        for it in items:
            x = it["x"].to(device)
            y = it["y"].to(device)
            z = it["z"].to(device)
            q = it["q"].to(device)

            x2 = it["x2"].to(device)
            y2 = it["y2"].to(device)
            z2 = it["z2"].to(device)
            q2 = it["q2"].to(device)

            recon_u = net(x, z, q)
            recon_x = netx(x2, z2, q2)

            # Train_/Test_ 코드와 동일하게 tuple[0] 사용
            err_u = (recon_u[0] - y).cpu().numpy()
            err_x = (recon_x[0] - y2).cpu().numpy()

            df_data = DiagnosisFeature(err_u, err_x)
            t2_array = T2_array(df_data, data_mean, data_std, p_k, v_I)
            spe_array = SPE_array(df_data, data_mean, data_std, p_k)
            ci_array = (spe_array / SPE_95_limit) + (t2_array / T_95_limit)

            ci_max_list.append(float(np.nanmax(np.asarray(ci_array, dtype=float))))
            y_true.append(np.int8(it["label"]))

    ci_maxs = np.asarray(ci_max_list, dtype=float)
    y_true = np.asarray(y_true, dtype=np.int8)
    predict_results = (ci_maxs[:, None] > predict_thresholds[None, :]).astype(np.int8)

    roc = compute_roc_auc_from_threshold_matrix(y_true, predict_results, predict_thresholds)
    return float(roc["auc"])


def main():
    parser = argparse.ArgumentParser(
        description="Validation: 저장된 체크포인트를 로드해 epoch별 AUC를 계산/저장"
    )
    parser.add_argument("--battery-type", choices=["QAS", "DTI"], default="QAS")
    parser.add_argument("--models-dir", type=str, required=True, help="Train_.py가 만든 model_path 폴더")
    parser.add_argument("--source-data-dir", type=str, default="./data")
    parser.add_argument("--normalize-dx", action="store_true")
    parser.add_argument("--normalize-val", type=int, default=4)
    parser.add_argument(
        "--sweep",
        choices=["netx", "net", "pair"],
        default="netx",
        help=(
            "체크포인트 sweep 방식. "
            "netx: net은 최종 net.pth 고정 + netx_eXXXX만 평가(순차학습 구조에 권장). "
            "net: netx는 최종 netx.pth 고정 + net_eXXXX만 평가. "
            "pair: 같은 epoch의 net_eXXXX/netx_eXXXX를 쌍으로 평가(순차학습이면 의미가 약할 수 있음)."
        ),
    )
    parser.add_argument("--include-final", action="store_true", help="최종 net.pth/netx.pth도 함께 기록")
    parser.add_argument("--output", type=str, default=None, help="결과 CSV 저장 경로(기본: models-dir/validation_auc.csv)")
    parser.add_argument("--thr-max", type=int, default=1000)
    parser.add_argument("--thr-step", type=int, default=2)
    parser.add_argument(
        "--cache-preprocessed",
        action="store_true",
        help="차량별 전처리 결과를 메모리에 캐시(속도↑, 메모리↑)",
    )
    args = parser.parse_args()

    models_dir = args.models_dir
    if not os.path.isdir(models_dir):
        raise FileNotFoundError(f"models-dir not found: {models_dir}")

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    learning_case = read_learning_case_from_sim_config(models_dir)
    net_map, netx_map = _find_checkpoints(models_dir)

    predict_thresholds = np.asarray(list(range(0, int(args.thr_max), int(args.thr_step))), dtype=float)

    # PCA 결과는 한 번만 로드
    pca_pack = load_pca_results(models_dir, load_data_nor=False, validate_shapes=False)

    # 모델 템플릿 생성(가중치는 체크포인트마다 로드)
    net_tpl, netx_tpl, dim_dict, preprocessing, skip_charge_ready = _build_models(
        battery_type=args.battery_type,
        learning_case=learning_case,
        device=device,
    )

    # 차량 데이터 준비 (선택적으로 캐시)
    items = []
    started_prep = time.perf_counter()
    for vehicle_id, label in _iter_test_vehicles(args.battery_type, models_dir):
        if args.cache_preprocessed:
            it = _preprocess_one_vehicle(
                vehicle_id=vehicle_id,
                label=label,
                battery_type=args.battery_type,
                source_data_dir=args.source_data_dir,
                dim_dict=dim_dict,
                preprocessing=preprocessing,
                skip_charge_ready=skip_charge_ready,
                normalize_dx=args.normalize_dx,
                normalize_val=args.normalize_val,
            )
        else:
            # 캐시를 끄면, 여기서는 ID/라벨만 들고 있다가 평가 시점에 매번 전처리
            it = {"vehicle_id": vehicle_id, "label": int(label)}
        items.append(it)
    prep_elapsed = time.perf_counter() - started_prep
    print(f"[Validation] vehicles={len(items)} prepared (cache={args.cache_preprocessed}) in {prep_elapsed:.3f}s")

    def _get_items_for_eval():
        if args.cache_preprocessed:
            return items
        # no-cache: 평가 시점에 전처리 수행
        out = []
        for it in items:
            out.append(
                _preprocess_one_vehicle(
                    vehicle_id=it["vehicle_id"],
                    label=it["label"],
                    battery_type=args.battery_type,
                    source_data_dir=args.source_data_dir,
                    dim_dict=dim_dict,
                    preprocessing=preprocessing,
                    skip_charge_ready=skip_charge_ready,
                    normalize_dx=args.normalize_dx,
                    normalize_val=args.normalize_val,
                )
            )
        return out

    out_csv = args.output or os.path.join(models_dir, "validation_auc.csv")
    os.makedirs(os.path.dirname(out_csv) or ".", exist_ok=True)

    rows = []
    best_key = None
    best_auc = float("-inf")

    def _eval_one(key, net_file, netx_file):
        nonlocal best_key, best_auc

        # fresh model instances (state_dict 로드 후 eval)
        net = net_tpl
        netx = netx_tpl
        net.load_state_dict(load_net_state_dict(models_dir, net_file))
        netx.load_state_dict(load_net_state_dict(models_dir, netx_file))

        started = time.perf_counter()
        auc_val = _compute_auc_for_checkpoint(
            net=net,
            netx=netx,
            dim_dict=dim_dict,
            pca_pack=pca_pack,
            device=device,
            items=_get_items_for_eval(),
            predict_thresholds=predict_thresholds,
        )
        elapsed = time.perf_counter() - started

        row = {
            "key": key,
            "net_file": net_file,
            "netx_file": netx_file,
            "auc": float(auc_val),
            "seconds": float(elapsed),
        }
        rows.append(row)
        print(f"[Validation] {key}: AUC={auc_val:.4f} ({elapsed:.2f}s) net={net_file}, netx={netx_file}")

        if auc_val > best_auc:
            best_auc = auc_val
            best_key = key

    # sweep
    if args.sweep == "netx":
        # 순차학습 구조상 net은 최종 net.pth 고정이 더 자연스러움
        epochs = sorted(netx_map.keys())
        for ep in epochs:
            _eval_one(f"netx_e{ep:04d}", net_file="net.pth", netx_file=netx_map[ep])
        if args.include_final:
            _eval_one("final", net_file="net.pth", netx_file="netx.pth")
    elif args.sweep == "net":
        epochs = sorted(net_map.keys())
        for ep in epochs:
            _eval_one(f"net_e{ep:04d}", net_file=net_map[ep], netx_file="netx.pth")
        if args.include_final:
            _eval_one("final", net_file="net.pth", netx_file="netx.pth")
    else:
        epochs = sorted(set(net_map.keys()) & set(netx_map.keys()))
        for ep in epochs:
            _eval_one(f"pair_e{ep:04d}", net_file=net_map[ep], netx_file=netx_map[ep])
        if args.include_final:
            _eval_one("final", net_file="net.pth", netx_file="netx.pth")

    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["key", "net_file", "netx_file", "auc", "seconds"])
        for r in rows:
            w.writerow([r["key"], r["net_file"], r["netx_file"], f"{r['auc']:.6f}", f"{r['seconds']:.3f}"])

    print(f"[Validation] Saved: {out_csv}")
    if best_key is not None and best_auc != float("-inf"):
        print(f"[Validation] Best: {best_key}, AUC={best_auc:.4f}")
    else:
        print("[Validation] Best AUC not found.")


if __name__ == "__main__":
    main()
