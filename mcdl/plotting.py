"""실행 스크립트들이 결과를 그림(PNG)으로 남길 때 쓰는 그림 함수 모음.

함수와 쓰는 곳
    plot_loss_curve_all             학습·검증 loss 곡선                     Test_.py, Test_multi_models.py, Test_tsne.py
    plot_roc_curve                  ROC 곡선과 추천 임계값 두 점             Test_.py, Test_multi_models.py
    plot_diagnostics_triplet        차량 한 대의 T²·SPE·CI 시계열(3단)       Test_.py (진단 그림 모드)
    plot_net_layer_params_by_index  모델 층의 가중치·편향 막대그래프         Test_.py
    plot_tsne_before_after_ci       실제 값·재구성 값 t-SNE와 CI             Test_tsne.py
    draw_auc_heatmap                체크포인트 조합별 AUC 히트맵             Test_multi_models.py, Plot_AUC_heatmap.py
    plot_testX_timeseries           입력 데이터 열들의 시계열               data_filter.py (그림 모드)
    plot_distribution_with_stats    값 분포 히스토그램과 평균·표준편차      data_filter.py (분석 모드)
그림 크기·해상도 같은 모양 설정은 각 함수 안에 값으로 적혀 있다. 그림 함수는 계산 결과(수치)에 영향을 주지 않는다.
"""

from __future__ import annotations

import os

import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import numpy as np
import pandas as pd
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler

# 임계선 색 (첫 번째, 두 번째 임계값 순서)
_THRESHOLD_COLORS = ["tab:red", "tab:purple", "tab:green"]


def plot_loss_curve_all(path):
    """Run 폴더(path)의 loss CSV 네 개로 loss 곡선 그림 네 장을 그려 같은 폴더에 저장한다.

    loss_net.png(U-model 학습), loss_netx.png(X-model 학습), loss_net_val.png·loss_netx_val.png(검증).
    이미 그림 파일이 있으면 다시 그리지 않는다. 화면에는 띄우지 않는다.
    쓰는 곳: Test_.py, Test_multi_models.py, Test_tsne.py (평가 시작 때, 실패해도 평가는 계속한다).
    """
    for name, title, dpi, png_compress_level in (
        ("loss_net", "Voltage train loss", 200, 9),
        ("loss_netx", "SoC train loss", 200, 9),
        ("loss_net_val", "Voltage validation loss (every 10 epochs)", 150, 1),
        ("loss_netx_val", "SoC validation loss (every 10 epochs)", 150, 1),
    ):
        save_path = f"{path}/{name}.png"
        if not os.path.exists(save_path):
            _plot_loss_curve(f"{path}/{name}.csv", title, save_path, dpi, png_compress_level)


def _plot_loss_curve(loss_csv_path, title, save_path, dpi, png_compress_level):
    """plot_loss_curve_all 안에서 쓴다. Train_.py가 저장한 loss CSV(열: epoch, avg_loss)로 곡선 하나를 그려 저장한다."""
    data = np.loadtxt(loss_csv_path, delimiter=",", skiprows=1)
    if data.ndim == 1:  # epoch가 하나뿐이면 1차원으로 읽힌다
        data = data.reshape(1, -1)
    epochs = data[:, 0].astype(np.int64, copy=False)
    losses = data[:, 1].astype(np.float64, copy=False)

    os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)

    fig = plt.figure(figsize=(10, 4))
    ax = fig.gca()
    ax.plot(epochs, losses, color="tab:blue", lw=1.5)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Average Loss")
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    # tight_layout은 느려서 여백을 직접 정한다
    fig.subplots_adjust(left=0.08, right=0.98, top=0.88, bottom=0.15)
    fig.savefig(save_path, dpi=dpi, pil_kwargs={"compress_level": png_compress_level})
    plt.close(fig)


def plot_roc_curve(fpr, tpr, auc_val, *, best_point, closest_to_01_point, save_path, title, show):
    """ROC 곡선을 그리고, 추천 임계값 두 점(Youden J 최대점, (0,1) 최근접점)을 표시해 save_path에 저장한다.

    fpr, tpr, auc_val과 두 점은 mcdl/roc.py evaluate의 결과를 그대로 넘긴다. show=True면 창으로도 띄운다
    (창을 닫아야 다음으로 진행된다).
    쓰는 곳: Test_.py(results/AUC_ROC<C>.png), Test_multi_models.py(조합별 ROC).
    """
    os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)

    plt.figure(figsize=(6, 6))
    plt.plot(fpr, tpr, label=f"ROC (AUC={auc_val:.3f})")
    plt.plot([0, 1], [0, 1], linestyle="--", color="gray", label="Random")

    # 두 점과 설명 글. 글은 점 옆에 두되, 그림 오른쪽·위쪽 끝에 가까우면 반대쪽으로 옮긴다
    best_fpr, best_tpr, best_thr = best_point["fpr"], best_point["tpr"], best_point["threshold"]
    dx = 0.05 if best_fpr <= 0.7 else -0.25
    dy = 0.05 if best_tpr <= 0.7 else -0.25
    plt.scatter([best_fpr], [best_tpr], color="red", s=60, zorder=5, label=f"Best thr={best_thr:.3g}")
    plt.annotate(
        f"thr={best_thr:.3g}\nFPR={best_fpr:.2f}, TPR={best_tpr:.2f}",
        xy=(best_fpr, best_tpr),
        xytext=(min(max(best_fpr + dx, 0.0), 1.0), min(max(best_tpr + dy, 0.0), 1.0)),
        textcoords="data",
        arrowprops=dict(arrowstyle="->", color="red", lw=1.0),
        fontsize=9,
        color="red",
    )

    closest_fpr = closest_to_01_point["fpr"]
    closest_tpr = closest_to_01_point["tpr"]
    closest_thr = closest_to_01_point["threshold"]
    dx2 = 0.05 if closest_fpr <= 0.7 else -0.25
    dy2 = -0.12 if closest_tpr >= 0.5 else 0.08
    plt.scatter(
        [closest_fpr], [closest_tpr], color="blue", s=60, zorder=5, label=f"Closest(0,1) thr={closest_thr:.3g}"
    )
    plt.annotate(
        f"thr={closest_thr:.3g}\nFPR={closest_fpr:.2f}, TPR={closest_tpr:.2f}",
        xy=(closest_fpr, closest_tpr),
        xytext=(min(max(closest_fpr + dx2, 0.0), 1.0), min(max(closest_tpr + dy2, 0.0), 1.0)),
        textcoords="data",
        arrowprops=dict(arrowstyle="->", color="blue", lw=1.0),
        fontsize=9,
        color="blue",
    )

    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(title)
    plt.grid(True, alpha=0.3)
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(save_path, dpi=200)

    if show:
        plt.show()
    else:
        plt.close()


def plot_testX_timeseries(data, title, save_path, columns):
    """data(시점 × 열 텐서)에서 columns=(시작 열, 끝 열) 구간(끝 포함)의 열마다 시계열 그림을 한 장씩 저장한다.

    파일 이름은 "<save_path>-<열 번호>st.png"이다.
    쓰는 곳: data_filter.py (PLOT_MODE에서 차량별 입력 열을 그림 파일로 저장).
    """
    for i in range(columns[0], columns[1] + 1):
        print(f"{i}번째 특성을 개별 플롯으로 그립니다.")
        values = data[:, i].detach().cpu().numpy()

        fig, ax = plt.subplots(figsize=(12, 6))
        ax.plot(np.arange(len(values)), values, label="var1")
        ax.set_xlabel("Time")
        ax.set_ylabel("Value")
        ax.set_title(f"{title}-{i}st")
        ax.grid(True, alpha=0.3)
        ax.legend(loc="best", ncol=2, fontsize=8)

        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        fig.savefig(f"{save_path}-{i}st.png", dpi=150)
        plt.close(fig)


def plot_diagnostics_triplet(t2_array, spe_array, CI_array, thresholds, *, title, save_path, x_start, x_tick_step):
    """차량 한 대의 T²·SPE·CI 시계열을 위아래 3단 그림 하나로 그려 save_path에 저장한다.

    x_start 시점부터 그리고, x축 눈금은 x_tick_step 간격이다. CI 그림에만 thresholds의 값들을 가로 점선으로 표시한다.
    쓰는 곳: Test_.py (AUC_ANALYSIS = False인 진단 그림 모드, 차량마다 한 장).
    """
    t2 = np.asarray(t2_array[x_start:]).reshape(-1)
    spe = np.asarray(spe_array[x_start:]).reshape(-1)
    CI = np.asarray(CI_array[x_start:]).reshape(-1)
    x = np.arange(len(CI)) + x_start  # x축은 x_start부터

    fig, axes = plt.subplots(nrows=3, ncols=1, figsize=(12, 10), sharex=True)

    axes[0].plot(x, t2, color="tab:blue", lw=1.4)
    axes[0].set_ylabel("T²")
    axes[0].set_title("Hostelling's T² Statistic")
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(x, spe, color="tab:blue", lw=1.4)
    axes[1].set_ylabel("SPE")
    axes[1].set_title("Squared Prediction Error (SPE)")
    axes[1].grid(True, alpha=0.3)

    axes[2].plot(x, CI, color="tab:blue", lw=1.4, label="CI")
    axes[2].set_ylabel("CI")
    axes[2].set_xlabel("Time")
    axes[2].set_title("Comprehensive Index (CI)")
    axes[2].grid(True, alpha=0.3)

    _set_time_ticks([axes[0], axes[1], axes[2]], x, x_start, x_tick_step)

    for val, col in zip(thresholds, _THRESHOLD_COLORS):
        axes[2].axhline(val, color=col, linestyle="--", linewidth=1.2, label=f"Threshold-{val}")
    axes[2].legend(loc="best", fontsize=8)

    fig.suptitle(title, y=0.98)
    # tight_layout은 느려서 간격을 직접 정한다
    fig.subplots_adjust(hspace=0.25, top=0.92)

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    # PNG 압축 수준을 낮추면 저장이 빨라진다(대신 파일이 커진다)
    fig.savefig(save_path, dpi=100, pil_kwargs={"compress_level": 1})
    plt.close(fig)


def _set_time_ticks(axes, x, x_start, x_tick_step):
    """plot_diagnostics_triplet·plot_tsne_before_after_ci 안에서 쓴다. 시간 축 눈금을 x_tick_step 간격으로 고정한다.

    첫 눈금은 x_start, 마지막 눈금은 마지막 시점이 되도록 맞춘다.
    """
    ticks = np.floor(np.arange(x[0], x[-1] + 1, x_tick_step) / x_tick_step) * x_tick_step
    ticks[0] = x_start
    ticks = np.append(ticks, x[-1]) if ticks[-1] != x[-1] else ticks
    for ax in axes:
        ax.xaxis.set_major_locator(mtick.FixedLocator(ticks))


def plot_distribution_with_stats(data, title):
    """값들의 분포(히스토그램과 밀도 곡선)를 그리고 평균과 평균±표준편차 선, 개수·평균·표준편차·분산을 함께 표시한다.

    밀도 곡선은 seaborn으로 그리고, seaborn이 없으면 히스토그램만 그린다. 그림은 저장하지 않고 창으로 띄운다.
    쓰는 곳: data_filter.py (ANALYSIS_MODE에서 전압·온도·SOC·전류 등의 분포를 화면에 표시).
    """
    bins = 60
    x = np.asarray(data).ravel()
    mu = float(np.mean(x))
    std = float(np.std(x, ddof=1))
    var = float(np.var(x, ddof=1))
    n = int(x.size)

    fig, ax = plt.subplots(figsize=(10, 5))
    try:
        import seaborn as sns

        sns.histplot(x, bins=bins, kde=True, stat="density", color="tab:blue", ax=ax)
    except Exception:
        ax.hist(x, bins=bins, density=True, color="tab:blue", alpha=0.7)

    ax.axvline(mu, color="red", lw=2, label=f"mean = {mu:.3f}")
    ax.axvline(mu - std, color="purple", ls="--", lw=1.5, label=f"mean-σ = {mu - std:.3f}")
    ax.axvline(mu + std, color="purple", ls="--", lw=1.5, label=f"mean+σ = {mu + std:.3f}")

    ax.text(
        0.02,
        0.98,
        f"n = {n}\nmean = {mu:.3f}\nstd = {std:.3f}\nvar = {var:.3f}",
        transform=ax.transAxes,
        va="top",
        ha="left",
        bbox=dict(boxstyle="round", fc="white", ec="gray", alpha=0.8),
    )

    ax.set_xlabel("Value")
    ax.set_ylabel("Density")
    ax.set_title(title)
    ax.legend(loc="best")
    plt.tight_layout()
    plt.show(block=False)


def plot_net_layer_params_by_index(net, net_name, *, save_dir, normalize_y):
    """모델(net)의 fc1·fc2 층마다 가중치와 편향 값을 막대그래프로 그려 저장한다 (학습된 모델이 어떤 값을 갖는지 보는 용도).

    층마다 그림 한 장(왼쪽 가중치, 오른쪽 편향)을 "<save_dir>/<net_name>_<층 이름>_params_by_index[_normalized].png"로
    저장한다. normalize_y=True면 값을 최대 절댓값으로 나눠 -1~1 범위로 맞춘 그림이고 파일 이름 끝에 _normalized가 붙는다.
    쓰는 곳: Test_.py (U-model·X-model 각각 normalize_y True/False 두 번).
    """
    for layer_name in ("fc1", "fc2"):
        layer = getattr(net, layer_name)
        w = layer.weight.detach().cpu().numpy().ravel()
        b = layer.bias.detach().cpu().numpy().ravel()

        fig, axes = plt.subplots(1, 2, figsize=(6, 3))
        fig.suptitle(
            f"{layer_name}: value vs index (mode=bar)"
            + (" [y normalized to -1~1]" if normalize_y else "")
        )

        for ax, kind, raw, color in ((axes[0], "weight", w, "tab:blue"), (axes[1], "bias", b, "tab:orange")):
            values = _normalize_by_max_abs(raw) if normalize_y else raw
            x = np.arange(raw.size) + 1  # 값 번호는 1부터

            ax.bar(x, values, width=0.65, edgecolor="black", linewidth=0.6, color=color, align="center")
            # 막대 위(음수면 아래)에 값을 적는다
            dy = 0.03 if normalize_y else 0.0
            for x0, y0 in zip(x, values):
                ax.text(
                    x0,
                    y0 + (dy if y0 >= 0 else -dy),
                    f"{y0:.4g}",
                    ha="center",
                    va="bottom" if y0 >= 0 else "top",
                    fontsize=7,
                    rotation=0,
                )

            ax.set_title(f"{layer_name}.{kind}  (orig_n={raw.size}, plotted={values.size})")
            ax.set_xlabel(f"{kind} index")
            ax.set_ylabel("value" + (" (normalized)" if normalize_y else ""))
            ax.grid(True, alpha=0.2)
            if normalize_y:
                ax.set_ylim(-1.05, 1.05)

            # x축 눈금은 정수 값 번호만 (소수 눈금이 생기지 않게 고정)
            ax.set_xticks(x)
            ax.xaxis.set_major_locator(mtick.FixedLocator(x))
            ax.set_xticklabels([str(int(t)) for t in x])
            ax.set_xlim(x.min() - 0.5, x.max() + 0.5)

        fig.tight_layout()
        suffix = "_normalized" if normalize_y else ""
        fig.savefig(f"{save_dir}/{net_name}_{layer_name}_params_by_index{suffix}.png", dpi=150)
        plt.close(fig)


def _normalize_by_max_abs(y):
    """plot_net_layer_params_by_index 안에서 쓴다. 값을 최대 절댓값으로 나눠 -1~1 범위로 맞춘다(0의 위치와 부호는 그대로)."""
    y = np.asarray(y, dtype=float)
    max_abs = np.nanmax(np.abs(y))
    if not np.isfinite(max_abs) or max_abs == 0:
        return np.zeros_like(y, dtype=float)
    return y / max_abs


def plot_tsne_before_after_ci(*, X_before, X_after, labels, alarm_levels, CI_array, thresholds, args, vehicle_idx):
    """기준 정상 차량과 대상 차량의 데이터를 t-SNE로 2차원에 그려 비교하고, 대상 차량의 CI도 함께 그린다.

    t-SNE는 고차원 데이터를 비슷한 점끼리 가깝게 2차원에 펼쳐 보여 주는 시각화 방법이다.
    그림은 1×3: [실제 셀 값(before)의 t-SNE | 재구성 값(after)의 t-SNE | 대상 차량 CI 시계열과 임계선].
    X_before·X_after는 같은 시점들의 실제 값과 재구성 값(행 목록), labels는 점마다 0(정상 기준) 또는 1(대상 차량),
    alarm_levels는 점마다 CI가 임계값을 넘었는지(0 없음, 1 첫 번째, 2 두 번째 임계값 초과)다.
    args에서 그림 옵션(--x-start, --x-tick-step, --no-alarm*, --save)과 저장 위치를 읽는다.
    args.save면 <models_dir>/<models_idx>/results_tsne/tSNE_<vehicle_idx>.png로 저장하고, 아니면 창으로 띄운다.
    쓰는 곳: Test_tsne.py.
    """
    X_before = np.vstack(X_before).astype(float)  # (점 수, 셀 값 수)
    X_after = np.vstack(X_after).astype(float)
    y = np.asarray(labels, dtype=int)

    print("before:", X_before.shape, y.shape)
    print("after :", X_after.shape, y.shape)

    # before·after를 합쳐 같은 표준화·같은 t-SNE로 2차원 좌표를 구한다 (두 그림을 같은 좌표계로 비교하려고)
    Xs_all = StandardScaler().fit_transform(np.vstack([X_before, X_after]))
    tsne = TSNE(
        n_components=2,
        perplexity=10,
        learning_rate="auto",
        init="pca",
        random_state=0,
        max_iter=1000,
    )
    Z_all = tsne.fit_transform(Xs_all)
    Z_before = Z_all[: X_before.shape[0]]
    Z_after = Z_all[X_before.shape[0] :]

    # t-SNE 두 칸(axes[0], axes[1])은 축을 공유하고, CI 칸(axes[2])은 따로 둔다
    fig, axes = plt.subplots(1, 3, figsize=(18, 6), sharex=False, sharey=False)
    axes[1].sharex(axes[0])
    axes[1].sharey(axes[0])

    for ax, Z, title in ((axes[0], Z_before, "Before (y_recovered)"), (axes[1], Z_after, "After (recon)")):
        ax.scatter(Z[y == 0, 0], Z[y == 0, 1], s=4, alpha=0.5, label="normal")
        ax.scatter(Z[y == 1, 0], Z[y == 1, 1], s=4, alpha=0.5, label="fault")
        ax.set_title(title)
        ax.grid(True, alpha=0.2)

    CI = np.asarray(CI_array[args.x_start :]).reshape(-1)
    x = np.arange(len(CI)) + args.x_start  # x축은 x_start부터
    axes[2].plot(x, CI, color="tab:blue", lw=1.4, label="CI")
    axes[2].set_ylabel("CI")
    axes[2].set_xlabel("Time")
    axes[2].set_title("Comprehensive Index (CI)")
    axes[2].grid(True, alpha=0.3)
    _set_time_ticks([axes[2]], x, args.x_start, args.x_tick_step)

    for val, col in zip(thresholds, _THRESHOLD_COLORS):
        axes[2].axhline(val, color=col, linestyle="--", linewidth=1.2, label=f"Threshold-{val}")
    axes[2].legend(loc="best", fontsize=8)

    # 알람 점 강조: (켜짐 여부, 라벨, 알람 단계, 점 크기, 색, 테두리 굵기, 그리는 층, 범례)
    alarm_overlays = [
        (args.alarm_nlo, 0, 1, 4, "#7F3B08", 0.7, 5, f"normal (CI>{thresholds[0]})"),
        (args.alarm_nhi, 0, 2, 4, "#2A9D8F", 0.7, 5, f"normal (CI>{thresholds[1]})"),
        (args.alarm_flo, 1, 1, 7, "blue", 0.9, 6, f"fault (CI>{thresholds[0]})"),
        (args.alarm_fhi, 1, 2, 7, "red", 0.9, 6, f"fault (CI>{thresholds[1]})"),
    ]
    if args.alarm:
        alarm_level = np.asarray(alarm_levels, dtype=np.int8)
        for ax, Z in ((axes[0], Z_before), (axes[1], Z_after)):
            for enabled, label, level, size, color, width, zorder, legend in alarm_overlays:
                mask = (y == label) & (alarm_level == level)
                if enabled and np.any(mask):
                    ax.scatter(
                        Z[mask, 0],
                        Z[mask, 1],
                        alpha=0.3,
                        s=size,
                        marker="o",
                        edgecolors=color,
                        facecolors=color,
                        linewidths=width,
                        zorder=zorder,
                        label=legend,
                    )

    axes[0].legend(loc="best")
    axes[1].legend(loc="best")
    plt.tight_layout()
    if args.save:
        save_path = f"{args.models_dir}/{args.models_idx}/results_tsne/tSNE_{vehicle_idx}.png"
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        fig.savefig(save_path, dpi=150)
    else:
        plt.show()


def draw_auc_heatmap(csv_path, save_path):
    """
    AUC_summary CSV(열: u_model_idx, x_model_idx, AUC)로 체크포인트 조합별 AUC 히트맵을 그려 save_path에 저장한다.

    가로축은 U-model 체크포인트 번호, 세로축은 X-model 체크포인트 번호다. CSV가 없거나 열이 모자라면
    오류 메시지만 출력하고 끝낸다.
    쓰는 곳: Test_multi_models.py(평가가 끝날 때), Plot_AUC_heatmap.py(CSV로 다시 그릴 때).
    """
    if not os.path.exists(csv_path):
        print(f"Error: CSV file not found at {csv_path}")
        return

    print(f"Loading AUC results from {csv_path} for heatmap...")
    auc_df = pd.read_csv(csv_path)
    print(f"Loaded {len(auc_df)} model combination results")

    required_columns = ["u_model_idx", "x_model_idx", "AUC"]
    if not all(col in auc_df.columns for col in required_columns):
        print(f"Error: CSV must contain {required_columns} columns.")
        return

    print("Creating AUC heatmap...")

    # 행: X-model 번호, 열: U-model 번호 (번호 순으로 정렬)
    pivot_data = auc_df.pivot(index="x_model_idx", columns="u_model_idx", values="AUC")
    pivot_data = pivot_data.sort_index(axis=0).sort_index(axis=1)

    fig, ax = plt.subplots(
        figsize=(
            max(10, len(pivot_data.columns) * 0.8),
            max(8, len(pivot_data.index) * 0.6),
        )
    )

    # 색 범위는 실제 AUC 값의 최솟값~최댓값 (값이 하나뿐이면 ±0.01, 값이 없으면 0.5~1.0)
    valid_values = pivot_data.values[~np.isnan(pivot_data.values)]
    if len(valid_values) > 0:
        vmin, vmax = np.min(valid_values), np.max(valid_values)
        if vmax == vmin:
            vmin -= 0.01
            vmax += 0.01
    else:
        vmin, vmax = 0.5, 1.0

    im = ax.imshow(pivot_data.values, cmap="viridis", aspect="auto", vmin=vmin, vmax=vmax)

    ax.set_xticks(np.arange(len(pivot_data.columns)))
    ax.set_yticks(np.arange(len(pivot_data.index)))
    ax.set_xticklabels(pivot_data.columns)
    ax.set_yticklabels(pivot_data.index)
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label("AUC Score", rotation=270, labelpad=20, fontsize=12)

    # 칸마다 AUC 값을 적는다 (어두운 칸은 흰 글씨)
    for i in range(len(pivot_data.index)):
        for j in range(len(pivot_data.columns)):
            val = pivot_data.values[i, j]
            if not np.isnan(val):
                ax.text(
                    j,
                    i,
                    f"{val:.3f}",
                    ha="center",
                    va="center",
                    color="white" if val < 0.75 else "black",
                    fontsize=9,
                )

    ax.set_xlabel("U Model Index", fontsize=13, fontweight="bold")
    ax.set_ylabel("X Model Index", fontsize=13, fontweight="bold")
    ax.set_title(
        "AUC Heatmap: Model Combination Performance",
        fontsize=15,
        fontweight="bold",
        pad=20,
    )

    # 칸 경계선
    ax.set_xticks(np.arange(len(pivot_data.columns)) - 0.5, minor=True)
    ax.set_yticks(np.arange(len(pivot_data.index)) - 0.5, minor=True)
    ax.grid(which="minor", color="gray", linestyle="-", linewidth=0.5)
    ax.tick_params(which="minor", size=0)

    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"AUC heatmap saved to: {save_path}\n")
