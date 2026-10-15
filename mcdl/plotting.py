"""스크립트가 쓰는 그림 함수.

loss 곡선, ROC 곡선, 진단 시계열(T²·SPE·CI), 입력 시계열·분포, 모델 파라미터, t-SNE, AUC 히트맵을 그린다.
함수 본문은 예전 Function_.py에서 그대로 옮겼다.
"""

from __future__ import annotations

import os

import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import numpy as np
import pandas as pd
import torch
from scipy.stats import norm
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler


def load_loss_csv(loss_csv_path, *, delimiter=",", skiprows: int = 1):
    """Train_.py에서 저장한 loss CSV(epoch,avg_loss)를 로드.

    Returns:
        (epochs, losses): np.ndarray, np.ndarray
    """
    data = np.loadtxt(loss_csv_path, delimiter=delimiter, skiprows=skiprows)
    if data.ndim == 1:
        data = data.reshape(1, -1)
    if data.shape[1] < 2:
        raise ValueError(
            f"loss CSV는 최소 2개 컬럼(epoch,loss)이 필요합니다: {loss_csv_path}"
        )
    epochs = data[:, 0].astype(np.int64, copy=False)
    losses = data[:, 1].astype(np.float64, copy=False)
    return epochs, losses


def plot_loss_curve(
    *,
    loss_csv_path=None,
    epochs=None,
    losses=None,
    title: str = "Loss Curve",
    x_label: str = "Epoch",
    y_label: str = "Average Loss",
    figsize=(10, 4),
    save_path=None,
    show: bool = True,
    dpi: int = 150,
    tight_layout: bool = False,
    downsample: int = 1,
    yscale=None,
    png_compress_level=1,
):
    """Loss curve를 플로팅.

    입력은 (1) loss_csv_path 또는 (2) epochs/losses 배열 둘 중 하나.
    속도를 위해 downsample, tight_layout 옵션을 제공.

    Returns:
        (epochs, losses): 플롯에 사용된 배열
    """
    if loss_csv_path is not None:
        epochs, losses = load_loss_csv(loss_csv_path)
    else:
        if epochs is None or losses is None:
            raise ValueError(
                "loss_csv_path 또는 (epochs, losses) 중 하나는 반드시 제공해야 합니다."
            )
        epochs = np.asarray(epochs)
        losses = np.asarray(losses, dtype=np.float64)

    if epochs.shape[0] != losses.shape[0]:
        raise ValueError(
            f"epochs/losses 길이가 다릅니다: {epochs.shape[0]} vs {losses.shape[0]}"
        )

    ds = int(downsample) if downsample is not None else 1
    if ds < 1:
        ds = 1
    if ds > 1:
        epochs_p = epochs[::ds]
        losses_p = losses[::ds]
    else:
        epochs_p = epochs
        losses_p = losses

    if save_path is not None:
        os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)

    fig = plt.figure(figsize=figsize)
    ax = fig.gca()
    ax.plot(epochs_p, losses_p, color="tab:blue", lw=1.5)
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    if yscale is not None:
        ax.set_yscale(yscale)

    if tight_layout:
        fig.tight_layout()
    else:
        # tight_layout은 비용이 커서 기본은 빠른 여백으로
        fig.subplots_adjust(left=0.08, right=0.98, top=0.88, bottom=0.15)

    if save_path is not None:
        if str(save_path).lower().endswith(".png") and png_compress_level is not None:
            fig.savefig(
                save_path,
                dpi=dpi,
                pil_kwargs={"compress_level": int(png_compress_level)},
            )
        else:
            fig.savefig(save_path, dpi=dpi)

    if show:
        plt.show()
    else:
        plt.close(fig)

    return epochs, losses


def plot_loss_curve_all(path):
    save_path = f"{path}/loss_net.png"
    if not os.path.exists(save_path):
        plot_loss_curve(
            loss_csv_path=f"{path}/loss_net.csv",
            title="Voltage train loss",
            save_path=save_path,
            show=False,  # 저장만 하고 창은 안 띄움
            dpi=200,
            downsample=1,  # 1이면 전체 epoch 표시
            tight_layout=False,
            png_compress_level=9,
        )

    save_path = f"{path}/loss_netx.png"
    if not os.path.exists(save_path):
        plot_loss_curve(
            loss_csv_path=f"{path}/loss_netx.csv",
            title="SoC train loss",
            save_path=save_path,
            show=False,  # 저장만 하고 창은 안 띄움
            dpi=200,
            downsample=1,  # 1이면 전체 epoch 표시
            tight_layout=False,
            png_compress_level=9,
        )

    save_path = f"{path}/loss_net_val.png"
    if not os.path.exists(save_path):
        plot_loss_curve(
            loss_csv_path=f"{path}/loss_net_val.csv",
            title="Voltage validation loss (every 10 epochs)",
            save_path=save_path,
            show=False,
            downsample=1,
        )

    save_path = f"{path}/loss_netx_val.png"
    if not os.path.exists(save_path):
        plot_loss_curve(
            loss_csv_path=f"{path}/loss_netx_val.csv",
            title="SoC validation loss (every 10 epochs)",
            save_path=save_path,
            show=False,
            downsample=1,
        )


def plot_roc_curve(
    fpr,
    tpr,
    auc_val,
    *,
    best_point=None,
    closest_to_01_point=None,
    save_path=None,
    title: str = "AUC-ROC Curve",
    figsize=(6, 6),
    show: bool = True,
    dpi: int = 200,
):
    """ROC 커브를 플로팅하고(옵션으로) 최적점들을 표시."""

    fpr = np.asarray(fpr, dtype=float)
    tpr = np.asarray(tpr, dtype=float)
    auc_val = float(auc_val)

    if save_path is not None:
        os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)

    plt.figure(figsize=figsize)
    plt.plot(fpr, tpr, label=f"ROC (AUC={auc_val:.3f})")
    plt.plot([0, 1], [0, 1], linestyle="--", color="gray", label="Random")

    if best_point is not None:
        best_fpr = float(best_point.get("fpr"))
        best_tpr = float(best_point.get("tpr"))
        best_thr = float(best_point.get("threshold"))
        dx = 0.05 if best_fpr <= 0.7 else -0.25
        dy = 0.05 if best_tpr <= 0.7 else -0.25
        plt.scatter(
            [best_fpr],
            [best_tpr],
            color="red",
            s=60,
            zorder=5,
            label=f"Best thr={best_thr:.3g}",
        )
        plt.annotate(
            f"thr={best_thr:.3g}\nFPR={best_fpr:.2f}, TPR={best_tpr:.2f}",
            xy=(best_fpr, best_tpr),
            xytext=(
                min(max(best_fpr + dx, 0.0), 1.0),
                min(max(best_tpr + dy, 0.0), 1.0),
            ),
            textcoords="data",
            arrowprops=dict(arrowstyle="->", color="red", lw=1.0),
            fontsize=9,
            color="red",
        )

    if closest_to_01_point is not None:
        closest_fpr = float(closest_to_01_point.get("fpr"))
        closest_tpr = float(closest_to_01_point.get("tpr"))
        closest_thr = float(closest_to_01_point.get("threshold"))
        dx2 = 0.05 if closest_fpr <= 0.7 else -0.25
        dy2 = -0.12 if closest_tpr >= 0.5 else 0.08
        plt.scatter(
            [closest_fpr],
            [closest_tpr],
            color="blue",
            s=60,
            zorder=5,
            label=f"Closest(0,1) thr={closest_thr:.3g}",
        )
        plt.annotate(
            f"thr={closest_thr:.3g}\nFPR={closest_fpr:.2f}, TPR={closest_tpr:.2f}",
            xy=(closest_fpr, closest_tpr),
            xytext=(
                min(max(closest_fpr + dx2, 0.0), 1.0),
                min(max(closest_tpr + dy2, 0.0), 1.0),
            ),
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

    if save_path is not None:
        plt.savefig(save_path, dpi=dpi)

    if show:
        plt.show()
    else:
        plt.close()


def plot_testX_timeseries(
    input,
    feature_names=None,
    title=None,
    figsize=(12, 6),
    save_path=None,
    show=True,
    seperate=False,
    _range: list = [-1],
    start_idx=0,
    ax=None,
    close=None,
):
    """
    test_X 시계열 데이터(형상: [T, 1, 7] 또는 [T, 7])를 시간(x축) 대비 다중 라인(y축)으로 그립니다.

    Args:
        test_X: torch.Tensor | np.ndarray | pd.DataFrame
            - 권장 형상: [T, 1, 7] (tensor) 또는 [T, 7]
        feature_names: 특성 이름 리스트(길이 = 특성 수). None이면 기본 이름 사용.
        title: 그래프 제목. None이면 기본 제목 사용.
        figsize: matplotlib figure size.
        save_path: 저장 경로(확장자 포함). None이면 저장하지 않음.
        show: True면 화면에 표시, False면 닫음(서버/배치 환경용).
    """

    if not (len(_range) == 1 and _range[0] == -1) and len(_range) != 2:
        raise ValueError(
            "Expected _range to be a 1D array with a single element -1 or a 2D array."
        )

    """ sperate가 True이면 for 문을 돌고 false 이면 한번에 그린다."""
    plot_all = True if _range == [-1] else False
    if seperate:
        _range = [0, input.shape[-1] - 1] if plot_all else _range
        for i in range(_range[0], _range[1] + 1):
            print(f"Plotting feature index: {i}")
            print(f"Input shape: {input.shape}")
            if plot_all:  # 모든 특성 그리기
                if input.shape[-1] > 10:
                    raise Exception(
                        "특성 수가 너무 많아 개별 플롯으로 그릴 수 없습니다. 'seperate'를 False로 설정하세요."
                    )

            print(f"{i}번째 특성을 개별 플롯으로 그립니다.")
            if input.ndim == 2:
                test_X = input[start_idx:, i].unsqueeze(1)
            elif input.ndim == 3:
                test_X = input[start_idx:, :, i].unsqueeze(1)
            else:
                raise ValueError(f"Expected 2D or 3D array, got shape {input.shape}")

            title_input = title + f"-{i}st"
            if save_path is not None:
                save_path_input = save_path + f"-{i}st.png"

            print(f"test_X shape for plotting: {test_X.shape}")
            # seperate=True는 개별 figure를 전제로 하므로, 외부 ax가 있어도 전달하지 않음
            plot_timeseries(
                test_X,
                feature_names,
                title_input,
                figsize,
                save_path_input,
                show,
                ax=None,
                close=close,
            )
    else:
        if plot_all:  # 모든 특성 그리기
            print("모든 특성을 한 번에 플롯으로 그립니다.")
            test_X = input[start_idx:]
            title += f"-all({test_X.shape[-1]})"
            if save_path is not None:
                save_path += f"-all.png"
        else:  # 특정 범위의 특성 그리기
            print(
                f"{_range[0]}부터 {_range[1]}까지의 특성을 한 번에 플롯으로 그립니다."
            )
            if input.ndim == 2:
                test_X = input[start_idx:, _range[0] : _range[1] + 1]
            elif input.ndim == 3:
                test_X = input[start_idx:, :, _range[0] : _range[1] + 1]
            else:
                raise ValueError(f"Expected 2D or 3D array, got shape {input.shape}")
            title += f"-range({_range[0]}-{_range[1]})"
            if save_path is not None:
                save_path += f"-range({_range[0]}-{_range[1]}).png"

        plot_timeseries(
            test_X, feature_names, title, figsize, save_path, show, ax=ax, close=close
        )


def plot_timeseries(
    test_X,
    feature_names=None,
    title=None,
    figsize=(12, 6),
    save_path=None,
    show=True,
    ax=None,
    close=None,
):
    # 입력을 numpy 2D [T, F]로 변환
    if isinstance(test_X, torch.Tensor):
        arr = test_X.detach().cpu().numpy()
    elif isinstance(test_X, pd.DataFrame):
        arr = test_X.values
    else:
        arr = np.asarray(test_X)

    # [T, 1, F] → [T, F]
    if arr.ndim == 3 and arr.shape[1] == 1:
        arr = arr[:, 0, :]
    if arr.ndim != 2:
        raise ValueError(
            f"Expected 2D array [T, F] or 3D [T,1,F], got shape {arr.shape}"
        )

    T, F = arr.shape

    # 레전드 라벨 준비
    if not feature_names or len(feature_names) != F:
        feature_names = [f"var{i+1}" for i in range(F)]

    fig = None
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.figure

    x = np.arange(T)
    for i in range(F):
        ax.plot(x, arr[:, i], label=feature_names[i])

    ax.set_xlabel("Time")
    ax.set_ylabel("Value")
    ax.set_title(title if title is not None else f"Time Series ({F} features)")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best", ncol=2, fontsize=8)

    # 외부에서 subplot 레이아웃을 관리할 수 있도록 ax가 주어지면 tight_layout은 호출하지 않음
    if ax is None and fig is not None:
        plt.tight_layout()

    if save_path is not None and fig is not None:
        dir_ = os.path.dirname(save_path)
        if dir_:
            os.makedirs(dir_, exist_ok=True)
        fig.savefig(save_path, dpi=150)

    # 외부 축(ax)로 그린 경우 show/close는 호출자가 제어
    if ax is None and fig is not None:
        if show:
            plt.show()
        else:
            plt.close(fig)

    if close is True and fig is not None:
        plt.close(fig)


def plot_diagnostics_triplet(
    t2_array,
    spe_array,
    CI_array,
    thresholds: list = None,
    sigma_levels=None,
    title=None,
    x_label="Time",
    y_labels=("T²", "SPE", "CI"),
    figsize=(12, 10),
    save_path=None,
    show=True,
    x_start=None,
    x_tick_step=1000,
    downsample: int = 1,
    tight_layout: bool = True,
    dpi: int = 150,
    png_compress_level=1,
):
    """
    세 개의 진단 시계열을 하나의 Figure에 3x1 서브플롯으로 그립니다.

    - 상단: T²(Hotelling)
    - 중단: SPE
    - 하단: CI(Comprehensive Index; CI)

    CI 서브플롯에만 `thresholds`로 전달된 값들을 수평선으로 표시합니다.

    Args:
        t2_array: (N,) 형태의 1D 배열
        spe_array: (N,) 형태의 1D 배열
        CI_array: (N,) 형태의 1D 배열
        thresholds: 수평 임계선 값 리스트/튜플(예: diagnosis_thresholds 결과)
        sigma_levels: 각 임계선 라벨 리스트(예: ["3σ", "4.5σ", "6σ"]). None이면 자동 라벨.
        title: 전체 Figure 타이틀
        x_label: 공통 x축 라벨(기본: "Time")
        y_labels: 각 서브플롯 y축 라벨 튜플(기본: ("T²", "SPE", "CI"))
        figsize: Figure 크기
        save_path: 저장 경로. None이면 저장하지 않음
        show: True면 화면 표시, False면 닫음
    """

    def _to_1d(a):
        if isinstance(a, pd.Series):
            return a.values
        return np.asarray(a).reshape(-1)

    t2 = _to_1d(t2_array[x_start:] if x_start is not None else t2_array)
    spe = _to_1d(spe_array[x_start:] if x_start is not None else spe_array)
    CI = _to_1d(CI_array[x_start:] if x_start is not None else CI_array)

    n = len(CI)
    # x_start가 지정되면 해당 값부터 시작하도록 x축 생성
    if x_start is not None:
        x = np.arange(n) + x_start
    else:
        x = np.arange(n)

    # 다운샘플링(플롯 성능 개선용): 시각화 목적이라면 k=5~20 정도가 효과적
    ds = int(downsample) if downsample is not None else 1
    if ds > 1:
        x = x[::ds]
        t2 = t2[::ds]
        spe = spe[::ds]
        CI = CI[::ds]

    fig, axes = plt.subplots(nrows=3, ncols=1, figsize=figsize, sharex=True)

    # Top: T²
    axes[0].plot(x, t2, color="tab:blue", lw=1.4)
    axes[0].set_ylabel(y_labels[0])
    axes[0].set_title("Hostelling's T² Statistic")
    axes[0].grid(True, alpha=0.3)

    # Middle: SPE
    axes[1].plot(x, spe, color="tab:blue", lw=1.4)
    axes[1].set_ylabel(y_labels[1])
    axes[1].set_title("Squared Prediction Error (SPE)")
    axes[1].grid(True, alpha=0.3)

    # Bottom: CI (CI) + thresholds
    axes[2].plot(x, CI, color="tab:blue", lw=1.4, label="CI")
    axes[2].set_ylabel(y_labels[2])
    axes[2].set_xlabel(x_label)
    axes[2].set_title("Comprehensive Index (CI)")
    axes[2].grid(True, alpha=0.3)

    # 시작 눈금을 포함시키기 위해 눈금 간격을 고정할 수 있는 옵션
    if x_tick_step is not None:
        ticks = (
            np.floor(np.arange(x[0], x[-1] + 1, x_tick_step) / x_tick_step)
            * x_tick_step
        )
        ticks[0] = x_start if x_start is not None else 0
        ticks = np.append(ticks, x[-1]) if ticks[-1] != x[-1] else ticks
        for ax in axes:
            ax.xaxis.set_major_locator(mtick.FixedLocator(ticks))

    threshold_color = ["tab:red", "tab:purple", "tab:green"]
    if thresholds is not None:
        thr = list(thresholds)
        # 최대 3개까지만 표시(요청사항 대응)
        thr = thr[:3]
        if sigma_levels is None:
            sigma_labels = [f"Threshold-{i}" for i in thr]
        else:
            sigma_labels = [
                f"Threshold {i} ({sigma_levels[i]:.1f}\u03c3)" for i in range(len(thr))
            ]
        for val, lab, col in zip(thr, sigma_labels, threshold_color):
            axes[2].axhline(val, color=col, linestyle="--", linewidth=1.2, label=lab)
        axes[2].legend(loc="best", fontsize=8)

    if title:
        fig.suptitle(title, y=0.98)

    if tight_layout:
        fig.tight_layout()
    else:
        # tight_layout은 비용이 큰 편이라, 고정 간격으로 빠르게 배치
        fig.subplots_adjust(hspace=0.25, top=0.92)

    if save_path is not None:
        dir_ = os.path.dirname(save_path)
        if dir_:
            os.makedirs(dir_, exist_ok=True)
        if str(save_path).lower().endswith(".png") and png_compress_level is not None:
            # PNG 저장이 보통 가장 느림: 압축 레벨을 낮추면 저장 시간↓(대신 파일 크기↑)
            fig.savefig(
                save_path,
                dpi=dpi,
                pil_kwargs={"compress_level": int(png_compress_level)},
            )
        else:
            fig.savefig(save_path, dpi=dpi)

    if show:
        plt.show()
    else:
        plt.close(fig)


def plot_distribution_with_stats(
    data,
    bins=50,
    kde=True,
    normal_fit=False,
    figsize=(10, 5),
    title="Distribution with mean/std",
    save_path=None,
    show=True,
    ddof=1,
    block=False,
):
    # 1D 벡터로 변환
    try:
        import torch

        if isinstance(data, torch.Tensor):
            x = data.detach().cpu().numpy().ravel()
        else:
            x = np.asarray(data).ravel()
    except Exception:
        x = np.asarray(data).ravel()

    # 통계
    mu = float(np.mean(x))
    std = float(np.std(x, ddof=ddof))
    var = float(np.var(x, ddof=ddof))
    n = int(x.size)

    # 플롯
    fig, ax = plt.subplots(figsize=figsize)
    used_seaborn = False
    if kde:
        try:
            import seaborn as sns

            sns.histplot(
                x, bins=bins, kde=True, stat="density", color="tab:blue", ax=ax
            )
            used_seaborn = True
        except Exception:
            ax.hist(x, bins=bins, density=True, color="tab:blue", alpha=0.7)
    else:
        ax.hist(x, bins=bins, density=True, color="tab:blue", alpha=0.7)

    # 평균/±σ 라인
    ax.axvline(mu, color="red", lw=2, label=f"mean = {mu:.3f}")
    ax.axvline(
        mu - std, color="purple", ls="--", lw=1.5, label=f"mean-σ = {mu - std:.3f}"
    )
    ax.axvline(
        mu + std, color="purple", ls="--", lw=1.5, label=f"mean+σ = {mu + std:.3f}"
    )

    # 정규분포 피팅(선택)
    if normal_fit:
        try:
            from scipy.stats import norm

            xs = np.linspace(np.min(x), np.max(x), 500)
            ax.plot(
                xs,
                norm.pdf(xs, loc=mu, scale=std),
                color="orange",
                lw=2,
                label="Normal fit (μ, σ)",
            )
        except Exception:
            pass

    # 텍스트 박스
    text = f"n = {n}\nmean = {mu:.3f}\nstd = {std:.3f}\nvar = {var:.3f}"
    ax.text(
        0.02,
        0.98,
        text,
        transform=ax.transAxes,
        va="top",
        ha="left",
        bbox=dict(boxstyle="round", fc="white", ec="gray", alpha=0.8),
    )

    ax.set_xlabel("Value")
    ax.set_ylabel("Density" if used_seaborn or kde else "Frequency")
    ax.set_title(title)
    ax.legend(loc="best")
    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150)

    if show:
        plt.show(block=block)
    else:
        plt.close(fig)

    return {"mean": mu, "std": std, "var": var, "count": n}


def plot_net_layer_params_by_index(
    net,
    layer_names=("fc1", "fc2", "fc3"),
    net_name="net",
    *,
    mode="bar",
    max_points=8000,
    show=True,
    save_dir=None,
    figsize=(12, 6),
    bar_width_ratio=0.65,
    edge_color="black",
    edge_linewidth=0.6,
    annotate_values=True,
    annotate_max=60,
    value_fmt="{:.4g}",
    value_fontsize=7,
    normalize_y=True,
    max_xtick_labels=30,  # <- 추가: xtick 라벨 너무 많으면 일부만 표시
):
    for layer_name in layer_names:
        layer = getattr(net, layer_name, None)
        if layer is None:
            raise AttributeError(f"net에 '{layer_name}' 레이어가 없습니다.")

        w = layer.weight.detach().cpu().numpy().ravel()
        b = (
            layer.bias.detach().cpu().numpy().ravel()
            if layer.bias is not None
            else None
        )

        def _downsample(arr, max_n):
            n = arr.size
            if n <= max_n:
                idx = np.arange(n)
                return idx, arr
            step = int(np.ceil(n / max_n))
            idx = np.arange(0, n, step)
            return idx, arr[idx]

        def _bar_width_from_idx(idx):
            if idx.size <= 1:
                base = 1.0
            else:
                diffs = np.diff(idx)
                base = float(np.median(diffs)) if diffs.size else 1.0
                if base <= 0:
                    base = 1.0
            return base * float(bar_width_ratio)

        def _normalize_minus1_to_1(y):
            y = np.asarray(y, dtype=float)
            y_min = float(np.nanmin(y))
            y_max = float(np.nanmax(y))
            if not np.isfinite(y_min) or not np.isfinite(y_max) or (y_max - y_min) == 0:
                return np.zeros_like(y, dtype=float)
            return 2.0 * (y - y_min) / (y_max - y_min) - 1.0

        def normalize_by_max_abs(y):
            y = np.asarray(y, dtype=float)
            max_abs = np.nanmax(np.abs(y))
            if not np.isfinite(max_abs) or max_abs == 0:
                return np.zeros_like(y, dtype=float)
            return y / max_abs

        def _annotate(ax, xs, ys):
            if not annotate_values:
                return
            n = len(xs)
            if n == 0:
                return
            if n > int(annotate_max):
                xs = xs[: int(annotate_max)]
                ys = ys[: int(annotate_max)]

            dy = 0.03 if normalize_y else 0.0
            for x0, y0 in zip(xs, ys):
                ax.text(
                    x0,
                    y0 + (dy if y0 >= 0 else -dy),
                    value_fmt.format(y0),
                    ha="center",
                    va="bottom" if y0 >= 0 else "top",
                    fontsize=value_fontsize,
                    rotation=0,  # <- 가로
                )

        def _set_integer_xticks(ax, x_positions):
            # x_positions는 이미 정수(1-based)라고 가정
            x_positions = np.asarray(x_positions, dtype=int)
            n = x_positions.size
            if n == 0:
                return

            if n <= max_xtick_labels:
                ticks = x_positions
            else:
                # 너무 많으면 일부만(그래도 정수만)
                ticks = np.unique(
                    np.linspace(x_positions.min(), x_positions.max(), max_xtick_labels)
                    .round()
                    .astype(int)
                )

            ax.set_xticks(ticks)
            ax.xaxis.set_major_locator(
                mtick.FixedLocator(ticks)
            )  # <- 소수 tick 방지(강제)
            ax.set_xticklabels([str(int(t)) for t in ticks])

        w_idx, w_plot = _downsample(w, max_points)
        if b is not None:
            b_idx, b_plot = _downsample(b, max_points)

        if normalize_y:
            w_plot = normalize_by_max_abs(w_plot)
            if b is not None:
                b_plot = normalize_by_max_abs(b_plot)

        # x축을 1-based 인덱스로 (요청: 2개면 1,2만)
        xw = w_idx + 1
        xb = (b_idx + 1) if b is not None else None

        fig, axes = plt.subplots(1, 2 if b is not None else 1, figsize=figsize)
        if not isinstance(axes, np.ndarray):
            axes = np.array([axes])

        fig.suptitle(
            f"{layer_name}: value vs index (mode={mode})"
            + (" [y normalized to -1~1]" if normalize_y else "")
        )

        # weight
        ax = axes[0]
        if mode == "bar":
            width = _bar_width_from_idx(xw)  # 1-based라도 간격은 동일
            ax.bar(
                xw,
                w_plot,
                width=width,
                edgecolor=edge_color,
                linewidth=edge_linewidth,
                color="tab:blue",
                align="center",
            )
            _annotate(ax, xw, w_plot)
        else:
            ax.plot(xw, w_plot, linewidth=0.8)

        ax.set_title(f"{layer_name}.weight  (orig_n={w.size}, plotted={w_plot.size})")
        ax.set_xlabel("weight index")
        ax.set_ylabel("value" + (" (normalized)" if normalize_y else ""))
        ax.grid(True, alpha=0.2)
        if normalize_y:
            ax.set_ylim(-1.05, 1.05)

        _set_integer_xticks(ax, xw)
        ax.set_xlim(xw.min() - 0.5, xw.max() + 0.5)

        # bias
        if b is not None:
            ax = axes[1]
            if mode == "bar":
                width = _bar_width_from_idx(xb)
                ax.bar(
                    xb,
                    b_plot,
                    width=width,
                    edgecolor=edge_color,
                    linewidth=edge_linewidth,
                    color="tab:orange",
                    align="center",
                )
                _annotate(ax, xb, b_plot)
            else:
                ax.plot(xb, b_plot, linewidth=0.8)

            ax.set_title(f"{layer_name}.bias  (orig_n={b.size}, plotted={b_plot.size})")
            ax.set_xlabel("bias index")
            ax.set_ylabel("value" + (" (normalized)" if normalize_y else ""))
            ax.grid(True, alpha=0.2)
            if normalize_y:
                ax.set_ylim(-1.05, 1.05)

            _set_integer_xticks(ax, xb)
            ax.set_xlim(xb.min() - 0.5, xb.max() + 0.5)

        fig.tight_layout()

        if save_dir is not None:
            if normalize_y:
                fig.savefig(
                    f"{save_dir}/{net_name}_{layer_name}_params_by_index_normalized.png",
                    dpi=150,
                )
            else:
                fig.savefig(
                    f"{save_dir}/{net_name}_{layer_name}_params_by_index.png", dpi=150
                )

        if show:
            plt.show()
        else:
            plt.close(fig)


def plot_tsne_before_after_ci(
    *,
    X_list_before,
    Y_list_before,
    X_list_after,
    Y_list_after,
    alarm_masks,
    CI_array,
    thresholds,
    args,
    vehicle_idx,
):
    # ===== 리스트 -> 배열로 변환 (행 단위로 쌓였다는 가정) =====
    X_before = np.vstack(X_list_before).astype(float)  # (M1, D)
    y_before = np.asarray(Y_list_before, dtype=int)  # (M1,)

    X_after = np.vstack(X_list_after).astype(float)  # (M2, D)
    y_after = np.asarray(Y_list_after, dtype=int)  # (M2,)

    print("before:", X_before.shape, y_before.shape)
    print("after :", X_after.shape, y_after.shape)

    # ===== 합쳐서 같은 스케일러/같은 t-SNE로 임베딩 =====
    X_all = np.vstack([X_before, X_after])
    Xs_all = StandardScaler().fit_transform(X_all)

    tsne = TSNE(
        n_components=2,
        perplexity=10,
        learning_rate="auto",
        init="pca",
        random_state=0,
        max_iter=1000,
    )
    Z_all = tsne.fit_transform(Xs_all)

    n_before = X_before.shape[0]
    Z_before = Z_all[:n_before]
    Z_after = Z_all[n_before:]

    # ===== 1x3 plot (t-SNE 2개 + CI 1개) =====
    # t-SNE 패널(axes[0], axes[1])만 같은 좌표계로 비교하고,
    # CI 패널(axes[2])은 독립 축을 유지.
    fig, axes = plt.subplots(1, 3, figsize=(18, 6), sharex=False, sharey=False)
    axes[1].sharex(axes[0])
    axes[1].sharey(axes[0])

    for ax, Z, y, title in [
        (axes[0], Z_before, y_before, "Before (y_recovered)"),
        (axes[1], Z_after, y_after, "After (recon)"),
    ]:
        ax.scatter(Z[y == 0, 0], Z[y == 0, 1], s=4, alpha=0.5, label="normal")
        ax.scatter(Z[y == 1, 0], Z[y == 1, 1], s=4, alpha=0.5, label="fault")
        ax.set_title(title)
        ax.grid(True, alpha=0.2)

    CI = np.asarray(CI_array[args.x_start :]).reshape(-1)

    n = len(CI)
    # x_start가 지정되면 해당 값부터 시작하도록 x축 생성
    if args.x_start is not None:
        x = np.arange(n) + args.x_start
    else:
        x = np.arange(n)
    axes[2].plot(x, CI, color="tab:blue", lw=1.4, label="CI")
    axes[2].set_ylabel("CI")
    axes[2].set_xlabel("Time")
    axes[2].set_title("Comprehensive Index (CI)")
    axes[2].grid(True, alpha=0.3)

    if args.x_tick_step is not None:
        ticks = (
            np.floor(np.arange(x[0], x[-1] + 1, args.x_tick_step) / args.x_tick_step)
            * args.x_tick_step
        )
        ticks[0] = args.x_start if args.x_start is not None else 0
        ticks = np.append(ticks, x[-1]) if ticks[-1] != x[-1] else ticks
        axes[2].xaxis.set_major_locator(mtick.FixedLocator(ticks))

    threshold_color = ["tab:red", "tab:purple", "tab:green"]
    thr = list(thresholds)
    # 최대 3개까지만 표시
    thr = thr[:3]
    sigma_labels = [f"Threshold-{i}" for i in thr]
    for val, lab, col in zip(thr, sigma_labels, threshold_color):
        axes[2].axhline(val, color=col, linestyle="--", linewidth=1.2, label=lab)
    axes[2].legend(loc="best", fontsize=8)

    # alarm 샘플 강조(테두리만) — before/after 공통 처리
    alarm_style_base = {
        "alpha": 0.3,
        # "facecolors": "none",
    }

    alarm_style_normal_lo = {
        **alarm_style_base,
        "s": 4,
        "marker": "o",
        "edgecolors": "#7F3B08",
        "facecolors": "#7F3B08",
        "linewidths": 0.7,
        "zorder": 5,
        "label": f"normal (CI>{thresholds[0]})",
    }
    alarm_style_normal_hi = {
        **alarm_style_base,
        "s": 4,
        "marker": "o",
        "edgecolors": "#2A9D8F",
        "facecolors": "#2A9D8F",
        "linewidths": 0.7,
        "zorder": 5,
        "label": f"normal (CI>{thresholds[1]})",
    }

    alarm_style_fault_lo = {
        **alarm_style_base,
        "s": 7,
        "marker": "o",
        "edgecolors": "blue",
        "facecolors": "blue",
        "linewidths": 0.9,
        "zorder": 6,
        "label": f"fault (CI>{thresholds[0]})",
    }
    alarm_style_fault_hi = {
        **alarm_style_base,
        "s": 7,
        "marker": "o",
        "edgecolors": "red",
        "facecolors": "red",
        "linewidths": 0.9,
        "zorder": 6,
        "label": f"fault (CI>{thresholds[1]})",
    }

    for key, ax, Z, y in [
        ("before", axes[0], Z_before, y_before),
        ("after", axes[1], Z_after, y_after),
    ]:
        if not args.alarm:
            continue

        alarm_level = np.asarray(alarm_masks[key], dtype=np.int8)
        if alarm_level.shape[0] != Z.shape[0]:
            print(
                f"[WARN] alarm_mask_{key} 길이 불일치:",
                alarm_level.shape[0],
                f"vs Z_{key}:",
                Z.shape[0],
            )
            continue

        mask_normal_lo = (y == 0) & (alarm_level == 1)
        mask_normal_hi = (y == 0) & (alarm_level == 2)
        mask_fault_lo = (y == 1) & (alarm_level == 1)
        mask_fault_hi = (y == 1) & (alarm_level == 2)

        if args.alarm_nlo and np.any(mask_normal_lo):
            ax.scatter(
                Z[mask_normal_lo, 0],
                Z[mask_normal_lo, 1],
                **alarm_style_normal_lo,
            )
        if args.alarm_nhi and np.any(mask_normal_hi):
            ax.scatter(
                Z[mask_normal_hi, 0],
                Z[mask_normal_hi, 1],
                **alarm_style_normal_hi,
            )
        if args.alarm_flo and np.any(mask_fault_lo):
            ax.scatter(
                Z[mask_fault_lo, 0],
                Z[mask_fault_lo, 1],
                **alarm_style_fault_lo,
            )
        if args.alarm_fhi and np.any(mask_fault_hi):
            ax.scatter(
                Z[mask_fault_hi, 0],
                Z[mask_fault_hi, 1],
                **alarm_style_fault_hi,
            )

    axes[0].legend(loc="best")
    axes[1].legend(loc="best")
    plt.tight_layout()
    if args.save:
        save_path = (
            f"{args.models_dir}/{args.models_idx}/results_tsne/tSNE_{vehicle_idx}.png"
        )
        if save_path is not None and fig is not None:
            dir_ = os.path.dirname(save_path)
            if dir_:
                os.makedirs(dir_, exist_ok=True)
            fig.savefig(save_path, dpi=150)
    else:
        plt.show()

    return fig, axes


def draw_auc_heatmap(csv_path, save_path, title_suffix=""):
    """
    Draws AUC heatmap from a CSV file containing 'u_model_idx', 'x_model_idx', and 'AUC' columns.
    """
    # Read the CSV
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

    # Create pivot table for heatmap
    pivot_data = auc_df.pivot(index="x_model_idx", columns="u_model_idx", values="AUC")

    # Sort indices for better visualization
    pivot_data = pivot_data.sort_index(axis=0).sort_index(axis=1)

    u_indices = pivot_data.columns
    x_indices = pivot_data.index

    # Create heatmap
    fig, ax = plt.subplots(
        figsize=(
            max(10, len(u_indices) * 0.8),
            max(8, len(x_indices) * 0.6),
        )
    )

    # Calculate vmin/vmax based on data to enhance contrast
    data_values = pivot_data.values
    valid_values = data_values[~np.isnan(data_values)]

    if len(valid_values) > 0:
        val_min = np.min(valid_values)
        val_max = np.max(valid_values)
        # Add a tiny padding to avoid single-value issues
        if val_max == val_min:
            val_min -= 0.01
            val_max += 0.01

        # Use dynamic range instead of fixed 0.5-1.0
        vmin = val_min
        vmax = val_max
    else:
        vmin = 0.5
        vmax = 1.0

    # Use a colormap (viridis is good for AUC values)
    im = ax.imshow(
        pivot_data.values, cmap="viridis", aspect="auto", vmin=vmin, vmax=vmax
    )

    # Set ticks and labels
    ax.set_xticks(np.arange(len(pivot_data.columns)))
    ax.set_yticks(np.arange(len(pivot_data.index)))
    ax.set_xticklabels(pivot_data.columns)
    ax.set_yticklabels(pivot_data.index)

    # Rotate x labels for better readability
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

    # Add colorbar
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label("AUC Score", rotation=270, labelpad=20, fontsize=12)

    # Add text annotations with AUC values
    for i in range(len(pivot_data.index)):
        for j in range(len(pivot_data.columns)):
            val = pivot_data.values[i, j]
            if not np.isnan(val):
                text_color = "white" if val < 0.75 else "black"
                ax.text(
                    j,
                    i,
                    f"{val:.3f}",
                    ha="center",
                    va="center",
                    color=text_color,
                    fontsize=9,
                )

    # Labels and title
    ax.set_xlabel("U Model Index", fontsize=13, fontweight="bold")
    ax.set_ylabel("X Model Index", fontsize=13, fontweight="bold")
    ax.set_title(
        f"AUC Heatmap: Model Combination Performance {title_suffix}",
        fontsize=15,
        fontweight="bold",
        pad=20,
    )

    # Grid
    ax.set_xticks(np.arange(len(pivot_data.columns)) - 0.5, minor=True)
    ax.set_yticks(np.arange(len(pivot_data.index)) - 0.5, minor=True)
    ax.grid(which="minor", color="gray", linestyle="-", linewidth=0.5)
    ax.tick_params(which="minor", size=0)

    plt.tight_layout()

    # Ensure directory exists
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    plt.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"AUC heatmap saved to: {save_path}\n")
