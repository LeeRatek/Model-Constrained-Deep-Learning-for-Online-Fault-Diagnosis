from sklearn import preprocessing
import torch
import torch.nn as nn
import torch.optim as optim

# from sklearn.datasets import load_boston
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
import scipy.io as scio
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import torch.nn.functional as F
from torch import nn
from torchvision import transforms as tfs
from Function_ import (
    CustomSigmoidFunc,
    plot_distribution_with_stats,
    plot_testX_timeseries,
)


class MyDataset(Dataset):
    def __init__(self, data, target):
        self.data = data
        self.target = target

    def __getitem__(self, index):
        x = self.data[index]
        y = self.target[index]
        return x, y

    def __len__(self):
        return len(self.data)


INPUT_SIZE = 7  # rnn input size


class LSTM(nn.Module):
    def __init__(self):
        super(LSTM, self).__init__()
        self.lstm = nn.LSTM(
            input_size=INPUT_SIZE,
            hidden_size=18,  # rnn hidden unit
            num_layers=1,  # number of rnn layer
            batch_first=True,  # input & output will has batch size as 1s dimension. e.g. (batch, time_step, input_size)
            bidirectional=True,
        )
        self.out = nn.Linear(
            36, 2
        )  # Bi-directional LSTM 이기 때문에, 최종 추론에는 순방향 18 hidden node 그리고 역방향 18 hidden node 가중치를 모두 포함.

    def forward(self, x):
        r_out, (hidden_state1, hidden_state2) = self.lstm(x, None)
        outs = []
        for time_step in range(r_out.size(1)):
            outs.append(self.out(r_out[:, time_step, :]))
        return torch.stack(outs, dim=1)


class Dataset(Dataset):
    def __init__(self, x, y, z, q, w):
        self.x = torch.from_numpy(x).to(torch.double)
        self.y = torch.from_numpy(y).to(torch.double)
        self.z = torch.from_numpy(z).to(torch.double)
        self.q = torch.from_numpy(q).to(torch.double)
        self.w = torch.from_numpy(w).to(torch.double)

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        return self.x[idx], self.y[idx], self.z[idx], self.q[idx], self.w[idx]


class CombinedAE(nn.Module):
    def __init__(
        self,
        input_size,
        encode2_input_size,
        output_size,
        use_dx_in_forward: bool = True,
        add_one_output_layer: bool = False,
        activation_fn: CustomSigmoidFunc = None,
    ):
        super(CombinedAE, self).__init__()
        self.fc1 = nn.Linear(input_size, 1)
        self.fc2 = nn.Linear(encode2_input_size, 1)
        self.fc3 = nn.Linear(output_size, output_size)
        self.activation_fn = activation_fn
        self.use_dx_in_forward = use_dx_in_forward
        self.add_one_output_layer = add_one_output_layer
        if self.add_one_output_layer:
            self.fc4 = nn.Linear(output_size, output_size)

    def encode(self, x):
        return self.fc1(x)

    def encode2(self, x):
        return torch.sigmoid(self.fc2(x))

    def decode(self, z):
        return self.activation_fn.forward(self.fc3(z))

    def forward(self, x, dx, q, y=None):
        if not self.use_dx_in_forward:
            dx = dx * 0.0

        z = self.encode(x) + self.encode2(q) + dx
        if self.add_one_output_layer:
            z = self.fc4(z)
        re = self.decode(z)
        # self.analyze_outputs(x, dx, q, re, y, combine=True, ncols=3, fill="spiral")
        return re, z

    def analyze_outputs(
        self,
        x,
        z,
        q,
        re,
        y=None,
        cell_idx: int = 0,
        figsize=(6, 3),
        *,
        combine: bool = False,
        layout: tuple[int, int] | None = None,
        ncols: int = 1,
        fill: str = "row",
        save_path: str | None = None,
        show: bool = True,
    ):
        """모델 내부 중간 결과 7개를 시각화합니다.

        Args:
            x, dx, q, re: forward 입력/중간/출력 텐서
            figsize: 단일 플롯 기준 크기. combine=True이면 (width, height*7)로 확장합니다.
            combine: True면 7개 결과를 하나의 figure(7x1)로 묶어 출력합니다.
            layout: (rows, cols)로 그리드 레이아웃을 직접 지정합니다.
            ncols: layout을 주지 않았을 때 사용할 열 개수입니다.
            fill: combine=True일 때 축을 채우는 순서.
                - 'row': 행 우선(기본) (1,1)->(1,2)->...
                - 'col': 열 우선 (1,1)->(2,1)->...
                - 'outside_in_columns': 열을 바깥→안쪽(0,last,1,...) 순으로 위→아래 채움
                - 'spiral': 좌상단에서 시작해 아래→오른쪽→위→왼쪽으로 회전하며 안쪽으로 채움
            save_path: combine=True일 때 figure 전체를 저장할 경로(예: "out.png").
            show: True면 표시, False면 닫음.
        """

        if not combine:
            plot_testX_timeseries(
                x,
                feature_names="Kalman prediction",
                title="Kalman prediction",
                figsize=figsize,
                show=show,
                seperate=False,
                start_idx=0,
                _range=[0, 0],
            )
            plot_testX_timeseries(
                x,
                feature_names="LSTM prediction",
                title="LSTM prediction",
                figsize=figsize,
                show=show,
                seperate=False,
                start_idx=0,
                _range=[1, 1],
            )
            plot_testX_timeseries(
                self.encode2(q),
                feature_names="Encode vechile information",
                title="Encode vechile information",
                figsize=figsize,
                show=show,
                seperate=False,
                start_idx=0,
                _range=[0, 0],
            )
            plot_testX_timeseries(
                self.encode(x),
                feature_names="Encode Kalman + LSTM prediction",
                title="Encode Kalman + LSTM prediction",
                figsize=figsize,
                show=show,
                seperate=False,
                start_idx=0,
                _range=[0, 0],
            )
            plot_testX_timeseries(
                self.encode(x) + self.encode2(q),
                feature_names="Sum of all system information",
                title="Sum of all system information",
                figsize=figsize,
                show=show,
                seperate=False,
                start_idx=0,
                _range=[0, 0],
            )
            plot_testX_timeseries(
                z,
                feature_names=f"Latent information of {cell_idx+1}th cell",
                title=f"Latent information of {cell_idx+1}th cell",
                figsize=figsize,
                show=show,
                seperate=False,
                start_idx=0,
                _range=[cell_idx, cell_idx],
            )
            plot_testX_timeseries(
                re,
                feature_names=f"Decoder's output of {cell_idx+1}th cell",
                title=f"Decoder's output of {cell_idx+1}th cell",
                figsize=figsize,
                show=show,
                seperate=False,
                start_idx=0,
                _range=[cell_idx, cell_idx],
            )
            if y is not None:
                plot_testX_timeseries(
                    y,
                    feature_names=f"True value of {cell_idx+1}th cell",
                    title=f"True value of {cell_idx+1}th cell",
                    figsize=figsize,
                    show=show,
                    seperate=False,
                    start_idx=0,
                    _range=[cell_idx, cell_idx],
                )
            return

        # combine=True: 하나의 figure에 7개 서브플롯로 묶기 (layout/ncols 지원)
        import matplotlib.pyplot as plt
        import math
        import numpy as np

        w, h = figsize

        items = [
            (x, "Kalman prediction", [0, 0]),
            (x, "LSTM prediction", [1, 1]),
            (self.encode(x), "Encode Kalman + LSTM prediction", [0, 0]),
            (self.encode2(q), "Encode vechile information", [0, 0]),
            (self.encode(x) + self.encode2(q), "Sum of all system information", [0, 0]),
            (
                z,
                f"Latent information of {cell_idx+1}th cell",
                [cell_idx, cell_idx],
            ),
            (re, f"Decoder's output of {cell_idx+1}th cell", [cell_idx, cell_idx]),
        ]
        items += (
            []
            if y is None
            else [(y, f"True value of {cell_idx+1}th cell", [cell_idx, cell_idx])]
        )

        nplots = len(items)
        if layout is not None:
            nrows, ncols2 = int(layout[0]), int(layout[1])
            if nrows < 1 or ncols2 < 1:
                raise ValueError(
                    f"layout은 양의 정수 (rows, cols) 여야 합니다: {layout}"
                )
        else:
            ncols2 = int(ncols) if ncols is not None else 1
            if ncols2 < 1:
                ncols2 = 1
            nrows = int(math.ceil(nplots / ncols2))

        fig, axes = plt.subplots(
            nrows, ncols2, figsize=(w * ncols2, h * nrows), sharex=True
        )

        axes_arr = np.asarray(axes)
        if axes_arr.ndim == 0:
            axes_arr = axes_arr.reshape(1, 1)
        elif axes_arr.ndim == 1:
            if nrows == 1:
                axes_arr = axes_arr.reshape(1, -1)
            elif ncols2 == 1:
                axes_arr = axes_arr.reshape(-1, 1)
            else:
                axes_arr = axes_arr.reshape(nrows, ncols2)

        def _outside_in_col_order(num_cols: int) -> list[int]:
            order = []
            left = 0
            right = num_cols - 1
            while left <= right:
                if left == right:
                    order.append(left)
                    break
                order.append(left)
                order.append(right)
                left += 1
                right -= 1
            return order

        def _spiral_positions(num_rows: int, num_cols: int) -> list[tuple[int, int]]:
            """(0,0)에서 시작해 아래→오른쪽→위→왼쪽 순으로 테두리를 돌며 안쪽으로 채웁니다."""
            if num_rows <= 0 or num_cols <= 0:
                return []

            top = 0
            bottom = num_rows - 1
            left = 0
            right = num_cols - 1
            pos: list[tuple[int, int]] = []

            while left <= right and top <= bottom:
                # down along left column
                for r in range(top, bottom + 1):
                    pos.append((r, left))
                left += 1
                if left > right:
                    break

                # right along bottom row
                for c in range(left, right + 1):
                    pos.append((bottom, c))
                bottom -= 1
                if top > bottom:
                    break

                # up along right column
                for r in range(bottom, top - 1, -1):
                    pos.append((r, right))
                right -= 1
                if left > right:
                    break

                # left along top row
                for c in range(right, left - 1, -1):
                    pos.append((top, c))
                top += 1

            return pos

        fill_l = str(fill).strip().lower() if fill is not None else "row"
        if fill_l == "row":
            positions = [(r, c) for r in range(nrows) for c in range(ncols2)]
        elif fill_l == "col":
            positions = [(r, c) for c in range(ncols2) for r in range(nrows)]
        elif fill_l in ("outside_in_columns", "outside-in-columns", "outside_in"):
            col_order = _outside_in_col_order(ncols2)
            positions = [(r, c) for c in col_order for r in range(nrows)]
        elif fill_l in ("spiral", "spiral_down_right", "spiral-down-right"):
            positions = _spiral_positions(nrows, ncols2)
        else:
            raise ValueError(
                f"지원하지 않는 fill 옵션입니다: {fill} (row|col|outside_in_columns|spiral)"
            )

        for i, (arr, t, r) in enumerate(items):
            plot_testX_timeseries(
                arr,
                feature_names=t,
                title=t,
                figsize=figsize,
                show=False,
                seperate=False,
                start_idx=0,
                _range=r,
                ax=axes_arr[positions[i][0], positions[i][1]],
            )

        # 남는 축은 숨김 (fill 순서에 따라 "빈 칸" 위치가 달라짐)
        used = set(positions[: len(items)])
        for r, c in positions:
            if (r, c) not in used:
                axes_arr[r, c].set_visible(False)

        fig.tight_layout()
        if save_path is not None:
            fig.savefig(save_path, dpi=150)

        if show:
            plt.show()
        else:
            plt.close(fig)


class RMSELoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.mse = nn.MSELoss()

    def forward(self, yhat, y):
        return torch.sqrt(self.mse(yhat, y))
