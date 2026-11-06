r"""261106_RRWaveNet_Original_Rate: the original RRWaveNet, which regresses the
respiratory rate of a PPG window directly (one number per window), trained
and tested on CapnoBase, BIDMC and STEAM2 for comparison with the
RRWaveNet-inspired waveform model (261007 / 261013 / 261105).

Source
------
P. Osathitporn et al., "RRWaveNet: A compact end-to-end multiscale residual
CNN for robust PPG respiratory rate estimation," IEEE Internet of Things
Journal 10(18), 15943-15952, 2023. Code: github.com/IoBT-VISTEC/RRWaveNet
(main.py, utils.py, models/base.py, models/rrwavenet.py; TensorFlow/Keras).
The network, the training and the cross-validation of that code are ported
to PyTorch here without change, except one bug fix (below).
models/deeplearning.py (Bian et al. 2020), respnet.py and respwatch.py
(imported in models/__init__.py) are the repository's other baselines and
are not used.

Taken from the authors' code
----------------------------
* Input: one PPG window of W s, resampled to 50 Hz by scipy.signal.resample
  (utils.resampling; W x 50 samples). Output: the respiratory rate of the
  window (breaths/min).
* Network (models/rrwavenet.py):
  - three stems Conv1D(1 filter, kernel 16 / 32 / 64, stride 5, "same")
    -> BatchNorm -> ReLU -> MaxPool(3, stride 2, "same"), concatenated
    (3 channels);
  - eight residual blocks with 64, 64, 128, 128, 256, 256, 512, 512 filters
    (Conv1D k3 -> BN -> ReLU -> Conv1D k3 -> BN). The block input is
    concatenated with the block output, not added, so 1,923 channels
    remain at the end;
  - global average pooling -> ReLU -> Dense 128 -> ReLU -> Dense 64 -> tanh
    -> Dense 1;
  - He-normal convolution weights, Glorot-uniform dense weights, zero
    biases, Keras BatchNorm settings (momentum 0.99, epsilon 1e-3).
  - 6,857,594 trainable parameters.
* Training:
  - MSE loss; AdaBelief (adabelief_tf, ported below) with learning rate
    1e-3, epsilon 1e-14 and rectify; batch 100; at most 69,420 epochs;
  - EarlyStopping on the validation loss with patience 10. As in the Keras
    default, the weights of the last epoch are kept, not the best;
  - ReduceLROnPlateau on the validation loss: factor 0.25, patience 4,
    Keras min_delta 1e-4.
* Cross-validation (models/base.py):
  - leave-one-subject-out;
  - the other subjects are split by KFold(5, shuffle) into training and
    validation subjects, so each test subject gets five models (one per
    split), each trained from new weights;
  - a subject's MAE is the mean of its five models' MAEs, reported as the
    mean and SD over subjects.

Fixed in base.py
----------------
base.py passes the positions returned by kf.split(persons) to the data as
subject indices. So the test subject could land in the training or
validation data, and the last subject was never used. Here each position is
mapped to its subject (persons[i]).

Not in the authors' code
------------------------
The README asks the user to prepare data/{dataset}/X_{W}.npy and
RR_{W}.npy. The following comes from the paper or is chosen here:
* Windows: W = 16, 32 or 64 s with a 2 s stride (paper, Sec. III-A). The
  paper's headline results use W = 64 s: 1.66 +- 1.01 (BIDMC) and
  1.59 +- 1.08 (CapnoBase).
* Data and window quality: CapnoBase, BIDMC and STEAM2 are read by
  261105_RRWaveNet_Unified_Optimized.py, which is imported. That gives the
  same subjects, the same PPG at 64 Hz, the same quality maps and the same
  reference breaths. The paper removed windows with an SQI below 0.9.
  Here a window is used for training, validation and the window MAE when
  both hold:
  - it passes the Train_OK rule of 261105 (PPG and reference usable
    >= 50%, reference usable for training >= 50%, no labelled reference
    artifact);
  - it contains at least two reference breaths.
* RR label of a window: 60 / the mean interval of the reference breaths
  inside it. The breaths are the 261105 ground truth: expert CO2 labels for
  CapnoBase, breaths detected on the reference for BIDMC and STEAM2.
* PPG scale: min-max [0, 1] per window after the resampling. This is the
  input scaling of the 261105 model; the paper states no normalization.

Paper vs code: the paper gives learning rate 1e-4 and early-stopping
patience 5, while rrwavenet.py has 1e-3 and 10. The code is followed.

Evaluation
----------
1. Window MAE (the authors' metric, base.py):
   - per model, the mean |predicted - label| over the test subject's
     labelled windows;
   - per subject, the mean over its five models;
   - mean +- SD over subjects (SD as base.py prints it, np.std, and with
     ddof=1).
2. 60 s minutes, for comparison with 261105 and the other models:
   - Minutes and ground truth: the scored minutes of 261105, from its
     evaluate_subject_minutes. The minutes, the reference validity rule and
     the GT (the number of reference breaths in the minute) are the same.
   - Rate of a minute, W = 16 / 32 s: the mean prediction of the 2 s-stride
     windows that lie inside the minute.
   - Rate of a minute, W = 64 s: the prediction of the 2 s-stride window
     that contains the minute and whose centre is closest to the minute
     centre. This is the window from 2 s before to 2 s after the minute,
     shifted inside the recording at its ends.
   - Per minute, the absolute error (and the error) of the five models is
     averaged. The subject MAE (the mean over its minutes) is then the mean
     of its five models' MAEs, as in base.py.
   - Reported: minute MAE +- SD, bias +- SD, Pearson r, within 1 / 2
     breaths/min, and the subject MAE as mean +- SD and median [IQR].

Outputs
-------
PROJECT_ROOT = D:\PPG2RSP_RRWaveNet_Inspired\261106_RRWaveNet_Original_Rate
  W{W}\{dataset}\Folds\{subject}_fold{k}.json, _history.csv, _windows.csv
  W{W}\{dataset}\Minutes\{subject}_minutes.csv
  W{W}\{dataset}\mae_{W}.npz   (subjects x 5 window MAEs, as base.save_mae)
  Summary\: summary.csv, per_subject.csv, per_fold.csv, minutes.csv and
            261106_RRWaveNet_Original_Rate_results.xlsx

Run
---
Run with a GPU, from the folder of this script. 261105_RRWaveNet_Unified_Optimized.py
is looked for next to this script, in the current folder and in
D:\PPG2RSP_RRWaveNet_Inspired\261105_RRWaveNet_Unified_Optimized; or give
--base-script.
  python 261106_RRWaveNet_Original_Rate.py                 (W = 64 s, all datasets)
  python 261106_RRWaveNet_Original_Rate.py --winsize 16 32 64
  --datasets bidmc                       one dataset
  --shard 0/2, --shard 1/2               two parallel runs, then --summary-only
  Finished folds are kept, so an interrupted run continues with the same
  command.
  --dry-run --no-require-cuda            data check without training
"""

import argparse
import importlib.util
import json
import math
import random
import sys
import time
import zlib
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy import stats
from scipy.signal import resample
from sklearn.model_selection import KFold


# ---------------------------------------------------------------------------
# Paths and settings
# ---------------------------------------------------------------------------

SCRIPT_NAME = "261106_RRWaveNet_Original_Rate"
PROJECT_ROOT = Path(r"D:\PPG2RSP_RRWaveNet_Inspired\261106_RRWaveNet_Original_Rate")
WORKSPACE = Path(r"D:\PPG2RSP_RRWaveNet_Inspired")
BASE_SCRIPT = "261105_RRWaveNet_Unified_Optimized.py"
SCRIPT_DIR = Path(__file__).resolve().parent

DATASETS = ("capnobase", "bidmc", "steam2")
DATASET_LABELS = {"capnobase": "CapnoBase", "bidmc": "BIDMC", "steam2": "STEAM2"}

# Authors' code: main.py, models/rrwavenet.py, models/base.py
SAMPLING_RATE = 50          # main.py --sampling_rate
SEED = 69420                # main.py --seed
LEARNING_RATE = 1e-3        # rrwavenet.py (the paper states 1e-4)
EPSILON = 1e-14             # AdaBeliefOptimizer epsilon
EPOCHS = 69420              # rrwavenet.py: training ends by early stopping
BATCH_SIZE = 100
EARLY_STOP_PATIENCE = 10    # rrwavenet.py (the paper states 5)
LR_FACTOR = 0.25
LR_PATIENCE = 4
LR_MIN_DELTA = 1e-4         # Keras ReduceLROnPlateau default
N_SPLITS = 5                # base.py KFold
STEM_KERNELS = (16, 32, 64)
STEM_STRIDE = 5
RES_CHANNELS = (64, 64, 128, 128, 256, 256, 512, 512)
HIDDEN = 64                 # hp_intermediate
BN_MOMENTUM = 0.01          # Keras BatchNormalization momentum 0.99
BN_EPSILON = 1e-3           # Keras BatchNormalization epsilon

# Paper, Sec. III-A
WINDOW_SIZES = (16, 32, 64)
DEFAULT_WINDOW_SIZES = (64,)
WINDOW_STRIDE_SEC = 2.0

MINUTE_SEC = 60.0


# ---------------------------------------------------------------------------
# 261105 (data reading, quality rules and the scored minutes)
# ---------------------------------------------------------------------------

def find_base_script(explicit: Optional[str]) -> Path:
    if explicit:
        path = Path(explicit)
        if path.is_file():
            return path.resolve()
        raise FileNotFoundError(f"--base-script: {explicit} not found")
    folders = [
        SCRIPT_DIR,
        Path.cwd(),
        WORKSPACE / "261105_RRWaveNet_Unified_Optimized",
        SCRIPT_DIR.parent / "261105_RRWaveNet_Unified_Optimized",
    ]
    for folder in folders:
        path = folder / BASE_SCRIPT
        if path.is_file():
            return path.resolve()
    raise FileNotFoundError(
        f"{BASE_SCRIPT} not found (looked in {', '.join(str(folder) for folder in folders)}); "
        "give it with --base-script"
    )


def load_base(path: Path):
    """Import 261105 as a module (its data reading only; nothing is trained)."""
    name = "rrwavenet_261105"
    spec = importlib.util.spec_from_file_location(name, str(path))
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def stable_seed(*parts: object) -> int:
    return zlib.crc32("|".join(str(part) for part in parts).encode("utf-8")) % (2**31 - 1)


# ---------------------------------------------------------------------------
# Network (models/rrwavenet.py)
# ---------------------------------------------------------------------------

def same_padding(length: int, kernel: int, stride: int) -> Tuple[int, int]:
    """Left / right padding of TensorFlow's padding="same" (also for stride > 1)."""
    out = -(-length // stride)
    total = max((out - 1) * stride + kernel - length, 0)
    return total // 2, total - total // 2


class SameConv1d(nn.Conv1d):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        left, right = same_padding(x.shape[-1], self.kernel_size[0], self.stride[0])
        return super().forward(F.pad(x, (left, right)))


class SameMaxPool1d(nn.Module):
    def __init__(self, pool_size: int, stride: int):
        super().__init__()
        self.pool_size = pool_size
        self.stride = stride

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        left, right = same_padding(x.shape[-1], self.pool_size, self.stride)
        x = F.pad(x, (left, right), value=float("-inf"))
        return F.max_pool1d(x, self.pool_size, self.stride)


def batch_norm(channels: int) -> nn.BatchNorm1d:
    return nn.BatchNorm1d(channels, eps=BN_EPSILON, momentum=BN_MOMENTUM)


class MultiConv(nn.Module):
    """_multi_conv: Conv1D(1, k, stride 5, same) -> BN -> ReLU -> MaxPool(3, 2, same)."""

    def __init__(self, kernel_size: int):
        super().__init__()
        self.conv = SameConv1d(1, 1, kernel_size, stride=STEM_STRIDE)
        self.bn = batch_norm(1)
        self.pool = SameMaxPool1d(3, 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.pool(F.relu(self.bn(self.conv(x))))


class ResConv(nn.Module):
    """_resconv: Conv1D(N, 3) -> BN -> ReLU -> Conv1D(N, 3) -> BN, then
    Concatenate([block input, block output])."""

    def __init__(self, in_channels: int, filters: int):
        super().__init__()
        self.conv1 = SameConv1d(in_channels, filters, 3)
        self.bn1 = batch_norm(filters)
        self.conv2 = SameConv1d(filters, filters, 3)
        self.bn2 = batch_norm(filters)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.bn2(self.conv2(F.relu(self.bn1(self.conv1(x)))))
        return torch.cat([x, y], dim=1)


class RRWaveNetOriginal(nn.Module):
    def __init__(self):
        super().__init__()
        self.stems = nn.ModuleList(MultiConv(kernel) for kernel in STEM_KERNELS)
        channels = len(STEM_KERNELS)
        blocks = []
        for filters in RES_CHANNELS:
            blocks.append(ResConv(channels, filters))
            channels += filters
        self.blocks = nn.Sequential(*blocks)
        self.dense1 = nn.Linear(channels, 2 * HIDDEN)
        self.dense2 = nn.Linear(2 * HIDDEN, HIDDEN)
        self.dense3 = nn.Linear(HIDDEN, 1)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        """Keras initial weights: convolutions HeNormal (truncated normal,
        stddev sqrt(2 / fan_in)), Dense glorot_uniform, biases zero."""
        for module in self.modules():
            if isinstance(module, nn.Conv1d):
                fan_in = module.in_channels * module.kernel_size[0]
                std = math.sqrt(2.0 / fan_in) / 0.87962566103423978
                nn.init.trunc_normal_(module.weight, 0.0, std, -2.0 * std, 2.0 * std)
                nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                nn.init.zeros_(module.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (batch, 1, W * 50) -> (batch, 1) breaths/min."""
        x = torch.cat([stem(x) for stem in self.stems], dim=1)
        x = self.blocks(x)
        x = F.relu(x.mean(dim=-1))
        x = F.relu(self.dense1(x))
        x = torch.tanh(self.dense2(x))
        return self.dense3(x)


def count_parameters(model: nn.Module) -> int:
    return int(sum(p.numel() for p in model.parameters() if p.requires_grad))


class AdaBelief(torch.optim.Optimizer):
    """adabelief_tf.AdaBeliefOptimizer(learning_rate, epsilon=1e-14,
    rectify=True) as used in rrwavenet.py: beta_1 0.9, beta_2 0.999, no
    weight decay, no AMSGrad, no warm-up, SMA threshold 5.

    m = b1 m + (1 - b1) g;  s = b2 s + (1 - b2) (g - m)^2 + eps
    rectified (SMA >= 5): p -= lr r_t m_hat / (sqrt(s / (1 - b2^t)) + eps)
    otherwise:            p -= lr m_hat
    """

    def __init__(self, params, lr: float, eps: float = EPSILON,
                 betas: Tuple[float, float] = (0.9, 0.999), sma_threshold: float = 5.0):
        super().__init__(params, dict(lr=lr, eps=eps, betas=betas, sma_threshold=sma_threshold))

    @torch.no_grad()
    def step(self, closure=None):
        for group in self.param_groups:
            beta1, beta2 = group["betas"]
            eps = group["eps"]
            sma_inf = 2.0 / (1.0 - beta2) - 1.0
            for param in group["params"]:
                if param.grad is None:
                    continue
                state = self.state[param]
                if not state:
                    state["step"] = 0
                    state["m"] = torch.zeros_like(param)
                    state["s"] = torch.zeros_like(param)
                state["step"] += 1
                step = state["step"]
                m, s, grad = state["m"], state["s"], param.grad
                m.mul_(beta1).add_(grad, alpha=1.0 - beta1)
                diff = grad - m
                s.mul_(beta2).addcmul_(diff, diff, value=1.0 - beta2).add_(eps)
                bias1 = 1.0 - beta1 ** step
                bias2 = 1.0 - beta2 ** step
                m_hat = m / bias1
                sma = sma_inf - 2.0 * step * beta2 ** step / bias2
                if sma >= group["sma_threshold"]:
                    r = math.sqrt(
                        (sma - 4.0) / (sma_inf - 4.0) * (sma - 2.0) / (sma_inf - 2.0) * sma_inf / sma
                    )
                    update = r * m_hat / ((s / bias2).sqrt() + eps)
                else:
                    update = m_hat
                param.sub_(group["lr"] * update)
        return None


# ---------------------------------------------------------------------------
# Windows and labels
# ---------------------------------------------------------------------------

def minmax_rows(windows: np.ndarray) -> np.ndarray:
    low = windows.min(axis=1, keepdims=True)
    span = windows.max(axis=1, keepdims=True) - low
    out = np.zeros_like(windows)
    ok = span[:, 0] > 1e-12
    out[ok] = (windows[ok] - low[ok]) / span[ok]
    return out


def window_label(breaths: np.ndarray, start: int, end: int, fs: int) -> float:
    """60 / mean interval of the reference breaths in [start, end) (breaths/min)."""
    inside = breaths[(breaths >= start) & (breaths < end)]
    if len(inside) < 2 or inside[-1] <= inside[0]:
        return np.nan
    return float(60.0 * fs * (len(inside) - 1) / (inside[-1] - inside[0]))


def subject_windows(base, record: dict, winsize: int) -> dict:
    """All W s windows of one subject with a 2 s stride (every recording
    segment of 261105, no window across a segment end): the 50 Hz input,
    the quality rule and the RR label."""
    fs = int(record["fs"])
    length = int(round(winsize * fs))
    hop = int(round(WINDOW_STRIDE_SEC * fs))
    inputs, rows = [], []
    for segment_id in sorted(record["references"]):
        reference = record["references"][segment_id]
        signal = np.asarray(reference["model_input"], dtype=np.float64)
        if len(signal) < length:
            continue
        breaths = np.sort(np.asarray(reference["breaths"], dtype=np.float64))
        starts = np.arange(0, len(signal) - length + 1, hop, dtype=np.int64)
        windows = np.lib.stride_tricks.sliding_window_view(signal, length)[starts]
        windows = resample(windows, SAMPLING_RATE * winsize, axis=1)  # utils.resampling
        inputs.append(minmax_rows(windows).astype(np.float32))
        for start in starts:
            end = int(start) + length
            _, train_ok = base.window_rule(reference["crop_maps"], int(start), end)
            label = window_label(breaths, int(start), end, fs)
            rows.append(
                {
                    "Segment_ID": int(segment_id),
                    "Start_Sample": int(start),
                    "Window_Start_Sec": float(start) / fs,
                    "Window_End_Sec": float(end) / fs,
                    "Train_OK": int(train_ok),
                    "Label_RR": label,
                    "Labelled": int(train_ok and np.isfinite(label)),
                }
            )
    if not inputs:
        return {"x": np.empty((0, SAMPLING_RATE * winsize), np.float32), "windows": pd.DataFrame(rows), "fs": fs}
    return {"x": np.concatenate(inputs), "windows": pd.DataFrame(rows), "fs": fs}


def minute_grid(base, record: dict) -> pd.DataFrame:
    """The 60 s minutes of a subject as 261105 scores them.

    evaluate_subject_minutes of 261105 builds the minute grid over the
    subject's kept 10 s windows and applies the reference validity rule and
    the ground truth. It also scores a predicted waveform; it gets zeros
    here, and those columns are dropped.
    """
    metadata = [
        {"segment_id": int(segment_id), "start": int(start)}
        for segment_id, start in zip(record["segment_ids"], record["starts"])
    ]
    outputs = {"predictions": np.zeros_like(record["rsp"]), "targets": record["rsp"]}
    _, frame = base.evaluate_subject_minutes(outputs, metadata, record["references"], int(record["fs"]))
    if frame.empty:
        return frame
    frame = frame.drop(columns=["Pred_Wave_FFT_RR", "AE_Wave_FFT", "Wave_PCC"], errors="ignore")
    frame["GT_RR"] = frame["GT_Breath_Count"].astype(float) * (60.0 / float(base.EVAL_WINDOW_SEC))
    return frame


def minute_rates(windows: pd.DataFrame, predictions: np.ndarray, minutes: pd.DataFrame,
                 winsize: int, fs: int) -> np.ndarray:
    """Rate of every minute from the window predictions (see Evaluation)."""
    length = int(round(winsize * fs))
    by_segment = {}
    for segment_id, group in windows.assign(Pred=predictions).groupby("Segment_ID"):
        by_segment[int(segment_id)] = (
            group["Start_Sample"].to_numpy(np.int64),
            group["Pred"].to_numpy(np.float64),
        )
    out = np.full(len(minutes), np.nan)
    for index, (segment_id, start_sec, end_sec) in enumerate(
        minutes[["Segment_ID", "RR_Window_Start_Sec", "RR_Window_End_Sec"]].itertuples(index=False)
    ):
        if int(segment_id) not in by_segment:
            continue
        starts, values = by_segment[int(segment_id)]
        first, last = int(round(start_sec * fs)), int(round(end_sec * fs))
        if winsize <= MINUTE_SEC:
            inside = (starts >= first) & (starts + length <= last)
            if inside.any():
                out[index] = float(values[inside].mean())
        else:
            covering = np.flatnonzero((starts <= first) & (starts + length >= last))
            if len(covering):
                offset = np.abs(starts[covering] + length / 2.0 - (first + last) / 2.0)
                out[index] = float(values[covering[int(np.argmin(offset))]])
    return out


# ---------------------------------------------------------------------------
# Training (models/base.py and the Keras callbacks of rrwavenet.py)
# ---------------------------------------------------------------------------

def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed % (2**32))
    torch.manual_seed(seed)


def stack_subjects(data: Dict[str, dict], subjects: Sequence[str]) -> Tuple[np.ndarray, np.ndarray]:
    xs, ys = [], []
    for subject in subjects:
        item = data[subject]
        if item["windows"].empty:
            continue
        keep = item["windows"]["Labelled"].to_numpy(bool)
        xs.append(item["x"][keep])
        ys.append(item["windows"]["Label_RR"].to_numpy(np.float32)[keep])
    if not xs:
        return np.empty((0, 0), np.float32), np.empty(0, np.float32)
    return np.concatenate(xs), np.concatenate(ys)


@torch.no_grad()
def predict(model: nn.Module, x: np.ndarray, device: torch.device) -> np.ndarray:
    model.eval()
    out = []
    for first in range(0, len(x), BATCH_SIZE):
        batch = torch.from_numpy(x[first:first + BATCH_SIZE]).unsqueeze(1).to(device)
        out.append(model(batch)[:, 0].float().cpu().numpy())
    return np.concatenate(out) if out else np.empty(0, np.float32)


@torch.no_grad()
def mean_squared_error(model: nn.Module, x: torch.Tensor, y: torch.Tensor) -> float:
    model.eval()
    total = torch.zeros((), device=x.device)
    for first in range(0, len(x), BATCH_SIZE):
        total += F.mse_loss(model(x[first:first + BATCH_SIZE]), y[first:first + BATCH_SIZE], reduction="sum")
    return float(total.item() / len(x))


def train_model(x_train: np.ndarray, y_train: np.ndarray, x_val: np.ndarray, y_val: np.ndarray,
                device: torch.device, seed: int) -> Tuple[nn.Module, List[dict], dict]:
    """model.fit(x_train, y_train, epochs=69420, batch_size=100,
    validation_data=(x_val, y_val), callbacks=[EarlyStopping, ReduceLROnPlateau])."""
    set_seed(seed)
    model = RRWaveNetOriginal().to(device)
    optimizer = AdaBelief(model.parameters(), lr=LEARNING_RATE, eps=EPSILON)
    xt = torch.from_numpy(x_train).unsqueeze(1).to(device)
    yt = torch.from_numpy(y_train).unsqueeze(1).to(device)
    xv = torch.from_numpy(x_val).unsqueeze(1).to(device)
    yv = torch.from_numpy(y_val).unsqueeze(1).to(device)
    generator = torch.Generator().manual_seed(seed)
    history = []
    stop_best, stop_wait, best_epoch = math.inf, 0, 0
    plateau_best, plateau_wait = math.inf, 0
    for epoch in range(1, EPOCHS + 1):
        model.train()
        order = torch.randperm(len(xt), generator=generator).to(device)  # fit(shuffle=True)
        total = torch.zeros((), device=device)
        for first in range(0, len(order), BATCH_SIZE):
            index = order[first:first + BATCH_SIZE]
            optimizer.zero_grad(set_to_none=True)
            loss = F.mse_loss(model(xt[index]), yt[index])
            loss.backward()
            optimizer.step()
            total += loss.detach() * len(index)
        train_loss = float(total.item() / len(xt))
        val_loss = mean_squared_error(model, xv, yv)
        learning_rate = optimizer.param_groups[0]["lr"]
        history.append({"Epoch": epoch, "Loss": train_loss, "Val_Loss": val_loss, "LR": learning_rate})
        # EarlyStopping(monitor="val_loss", patience=10): min_delta 0; the
        # weights of the last epoch are kept (restore_best_weights=False).
        if val_loss < stop_best:
            stop_best, stop_wait, best_epoch = val_loss, 0, epoch
        else:
            stop_wait += 1
        # ReduceLROnPlateau(monitor="val_loss", factor=0.25, patience=4):
        # min_delta 1e-4, cooldown 0, min_lr 0.
        if val_loss < plateau_best - LR_MIN_DELTA:
            plateau_best, plateau_wait = val_loss, 0
        else:
            plateau_wait += 1
            if plateau_wait >= LR_PATIENCE:
                for group in optimizer.param_groups:
                    group["lr"] = learning_rate * LR_FACTOR
                plateau_wait = 0
        if stop_wait >= EARLY_STOP_PATIENCE:
            break
    info = {
        "Epochs_Run": len(history),
        "Lowest_Val_Loss_Epoch": best_epoch,
        "Lowest_Val_Loss": stop_best,
        "Final_Val_Loss": history[-1]["Val_Loss"],
        "Final_LR": optimizer.param_groups[0]["lr"],
    }
    del xt, yt, xv, yv, optimizer
    return model, history, info


# ---------------------------------------------------------------------------
# Folds of one test subject
# ---------------------------------------------------------------------------

def dataset_dir(results_root: Path, winsize: int, dataset_key: str) -> Path:
    return results_root / f"W{winsize}" / DATASET_LABELS[dataset_key]


def fold_paths(folder: Path, subject: str, fold: int) -> Dict[str, Path]:
    stem = folder / "Folds" / f"{subject}_fold{fold}"
    return {
        "json": stem.with_name(stem.name + ".json"),
        "history": stem.with_name(stem.name + "_history.csv"),
        "windows": stem.with_name(stem.name + "_windows.csv"),
    }


def fold_done(folder: Path, subject: str, fold: int) -> bool:
    paths = fold_paths(folder, subject, fold)
    return paths["json"].is_file() and paths["windows"].is_file()


def subject_folds(dataset_key: str, subjects: Sequence[str], test_subject: str) -> List[Tuple[List[str], List[str]]]:
    """base.py: KFold(5, shuffle) over the other subjects. Fixed: the
    positions of kf.split(persons) are mapped to the subjects."""
    persons = [subject for subject in subjects if subject != test_subject]
    folds = KFold(n_splits=N_SPLITS, shuffle=True, random_state=stable_seed(SEED, dataset_key, test_subject))
    return [
        ([persons[i] for i in train_positions], [persons[i] for i in val_positions])
        for train_positions, val_positions in folds.split(persons)
    ]


def run_subject(dataset_key: str, winsize: int, subjects: Sequence[str], test_subject: str,
                data: Dict[str, dict], folder: Path, device: torch.device) -> None:
    label = DATASET_LABELS[dataset_key]
    for fold, (train_subjects, val_subjects) in enumerate(subject_folds(dataset_key, subjects, test_subject), start=1):
        if fold_done(folder, test_subject, fold):
            continue
        paths = fold_paths(folder, test_subject, fold)
        paths["json"].parent.mkdir(parents=True, exist_ok=True)
        x_train, y_train = stack_subjects(data, train_subjects)
        x_val, y_val = stack_subjects(data, val_subjects)
        if len(x_train) == 0 or len(x_val) == 0:
            raise RuntimeError(f"{label} {test_subject} fold {fold}: no labelled training or validation window")
        seed = stable_seed(SEED, dataset_key, winsize, test_subject, fold)
        began = time.time()
        model, history, info = train_model(x_train, y_train, x_val, y_val, device, seed)
        test = data[test_subject]
        table = test["windows"].copy()
        table["Pred_RR"] = predict(model, test["x"], device) if len(test["x"]) else np.empty(0)
        labelled = table["Labelled"].to_numpy(bool) if len(table) else np.zeros(0, bool)
        errors = (table["Pred_RR"] - table["Label_RR"]).to_numpy(float)[labelled]
        result = {
            "Dataset": label,
            "Window_Sec": winsize,
            "Test_Subject": test_subject,
            "Fold": fold,
            "Train_Subjects": train_subjects,
            "Val_Subjects": val_subjects,
            "N_Train_Windows": int(len(x_train)),
            "N_Val_Windows": int(len(x_val)),
            "N_Test_Windows_Labelled": int(labelled.sum()),
            **info,
            "Test_Window_MAE": float(np.mean(np.abs(errors))) if len(errors) else np.nan,
            "Test_Window_MSE": float(np.mean(errors**2)) if len(errors) else np.nan,
            "Trainable_Params": count_parameters(model),
            "Seed": seed,
            "Device": str(device),
            "Elapsed_Sec": float(time.time() - began),
        }
        pd.DataFrame(history).to_csv(paths["history"], index=False)
        table.to_csv(paths["windows"], index=False)
        paths["json"].write_text(json.dumps(result, indent=2), encoding="utf-8")
        print(
            f"{label} W{winsize} {test_subject} fold {fold}/{N_SPLITS}: {len(x_train)} training / "
            f"{len(x_val)} validation windows, {info['Epochs_Run']} epochs (lowest validation loss at "
            f"epoch {info['Lowest_Val_Loss_Epoch']}), window MAE {result['Test_Window_MAE']:.3f} "
            f"({result['Elapsed_Sec']:.0f} s)",
            flush=True,
        )
        del model, x_train, y_train, x_val, y_val
        if device.type == "cuda":
            torch.cuda.empty_cache()


def write_subject_minutes(base, dataset_key: str, winsize: int, subject: str, record: dict,
                          windows: dict, folder: Path) -> None:
    """Rates of the scored minutes from the five models of a subject."""
    minutes = minute_grid(base, record)
    out = folder / "Minutes" / f"{subject}_minutes.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    if minutes.empty:
        minutes.to_csv(out, index=False)
        return
    minutes.insert(0, "Subject", subject)
    minutes.insert(0, "Window_Sec", winsize)
    minutes.insert(0, "Dataset", DATASET_LABELS[dataset_key])
    predictions, errors = [], []
    for fold in range(1, N_SPLITS + 1):
        table = pd.read_csv(fold_paths(folder, subject, fold)["windows"])
        rates = minute_rates(table, table["Pred_RR"].to_numpy(float), minutes, winsize, windows["fs"])
        minutes[f"Pred_RR_Fold{fold}"] = rates
        minutes[f"AE_Fold{fold}"] = np.abs(rates - minutes["GT_RR"].to_numpy(float))
        predictions.append(rates)
        errors.append(rates - minutes["GT_RR"].to_numpy(float))
    with np.errstate(invalid="ignore"):
        minutes["Pred_RR_Mean_Folds"] = np.nanmean(np.vstack(predictions), axis=0) if predictions else np.nan
        minutes["Error_Mean_Folds"] = np.nanmean(np.vstack(errors), axis=0)
        minutes["AE_Mean_Folds"] = np.nanmean(np.abs(np.vstack(errors)), axis=0)
    minutes.to_csv(out, index=False)


# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------

def sd(values, ddof: int = 1) -> float:
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    return float(np.std(values, ddof=ddof)) if values.size > ddof else np.nan


def read_folds(results_root: Path, winsizes: Sequence[int], dataset_keys: Sequence[str]) -> pd.DataFrame:
    rows = []
    for winsize in winsizes:
        for dataset_key in dataset_keys:
            for path in sorted((dataset_dir(results_root, winsize, dataset_key) / "Folds").glob("*_fold*.json")):
                item = json.loads(path.read_text(encoding="utf-8"))
                item["Train_Subjects"] = ",".join(item["Train_Subjects"])
                item["Val_Subjects"] = ",".join(item["Val_Subjects"])
                rows.append(item)
    frame = pd.DataFrame(rows)
    if not frame.empty:
        frame["Test_Subject"] = frame["Test_Subject"].astype(str)
    return frame


def read_minutes(results_root: Path, winsizes: Sequence[int], dataset_keys: Sequence[str]) -> pd.DataFrame:
    frames = []
    for winsize in winsizes:
        for dataset_key in dataset_keys:
            for path in sorted((dataset_dir(results_root, winsize, dataset_key) / "Minutes").glob("*_minutes.csv")):
                frame = pd.read_csv(path, dtype={"Subject": str})
                if not frame.empty:
                    frames.append(frame)
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def scored(minutes: pd.DataFrame) -> pd.DataFrame:
    if minutes.empty:
        return minutes
    valid = minutes[minutes["Ref_Valid"].astype(bool)]
    return valid[np.isfinite(valid["AE_Mean_Folds"].to_numpy(float))]


def summarize(folds: pd.DataFrame, minutes: pd.DataFrame, results_root: Path) -> None:
    out_dir = results_root / "Summary"
    out_dir.mkdir(parents=True, exist_ok=True)
    if folds.empty:
        print("No finished fold found; nothing to summarize.")
        return
    complete = folds.groupby(["Window_Sec", "Dataset", "Test_Subject"])["Fold"].transform("nunique") == N_SPLITS
    folds = folds[complete]
    subject_rows, summary_rows = [], []
    have_minutes = scored(minutes)
    for (winsize, dataset), group in folds.groupby(["Window_Sec", "Dataset"], sort=False):
        window_mae = group.pivot_table(index="Test_Subject", columns="Fold", values="Test_Window_MAE")
        subject_window_mae = window_mae.mean(axis=1)
        dataset_key = next(key for key, value in DATASET_LABELS.items() if value == dataset)
        np.savez(
            dataset_dir(results_root, int(winsize), dataset_key) / f"mae_{int(winsize)}",
            window_mae.to_numpy(float),
            subjects=np.asarray(window_mae.index, dtype=str),
        )
        part = have_minutes[(have_minutes["Window_Sec"] == winsize) & (have_minutes["Dataset"] == dataset)] \
            if not have_minutes.empty else pd.DataFrame()
        subject_minute = part.groupby("Subject")["AE_Mean_Folds"].mean() if not part.empty else pd.Series(dtype=float)
        for subject in window_mae.index:
            sub = part[part["Subject"] == subject] if not part.empty else pd.DataFrame()
            subject_rows.append(
                {
                    "Dataset": dataset,
                    "Window_Sec": int(winsize),
                    "Subject": subject,
                    "N_Test_Windows_Labelled": int(group.loc[group["Test_Subject"] == subject, "N_Test_Windows_Labelled"].iloc[0]),
                    "Window_MAE": float(subject_window_mae[subject]),
                    "Window_MAE_SD_Folds": sd(window_mae.loc[subject]),
                    "N_Minutes_Scored": int(len(sub)),
                    "GT_RR_Mean": float(sub["GT_RR"].mean()) if len(sub) else np.nan,
                    "Pred_RR_Mean": float(sub["Pred_RR_Mean_Folds"].mean()) if len(sub) else np.nan,
                    "Minute_MAE": float(subject_minute.get(subject, np.nan)),
                    "Minute_Bias": float(sub["Error_Mean_Folds"].mean()) if len(sub) else np.nan,
                }
            )
        row = {
            "Dataset": dataset,
            "Window_Sec": int(winsize),
            "N_Subjects": int(window_mae.shape[0]),
            "Window_MAE_Subject_Mean": float(subject_window_mae.mean()),
            "Window_MAE_Subject_SD_base_py": sd(subject_window_mae, ddof=0),
            "Window_MAE_Subject_SD": sd(subject_window_mae),
            "Window_MAE_Subject_Median": float(subject_window_mae.median()),
            "Window_MAE_Subject_Q1": float(subject_window_mae.quantile(0.25)),
            "Window_MAE_Subject_Q3": float(subject_window_mae.quantile(0.75)),
            "N_Subjects_Minutes": int(subject_minute.size),
            "N_Minutes": int(len(part)),
        }
        if len(part):
            ae = part["AE_Mean_Folds"].to_numpy(float)
            error = part["Error_Mean_Folds"].to_numpy(float)
            row.update(
                {
                    "Minute_MAE": float(ae.mean()),
                    "Minute_MAE_SD": sd(ae),
                    "Minute_Bias": float(error.mean()),
                    "Minute_Bias_SD": sd(error),
                    "Minute_Pearson_r": float(stats.pearsonr(part["Pred_RR_Mean_Folds"], part["GT_RR"])[0])
                    if len(part) > 2 and part["Pred_RR_Mean_Folds"].std() > 0 and part["GT_RR"].std() > 0
                    else np.nan,
                    "Minute_Within_1_BPM_Pct": float(np.mean(ae <= 1.0) * 100.0),
                    "Minute_Within_2_BPM_Pct": float(np.mean(ae <= 2.0) * 100.0),
                    "Subject_Minute_MAE_Mean": float(subject_minute.mean()),
                    "Subject_Minute_MAE_SD": sd(subject_minute),
                    "Subject_Minute_MAE_Median": float(subject_minute.median()),
                    "Subject_Minute_MAE_Q1": float(subject_minute.quantile(0.25)),
                    "Subject_Minute_MAE_Q3": float(subject_minute.quantile(0.75)),
                }
            )
        row.update(
            {
                "Epochs_Run_Median": float(group["Epochs_Run"].median()),
                "Lowest_Val_Loss_Epoch_Median": float(group["Lowest_Val_Loss_Epoch"].median()),
                "Train_Windows_Median": float(group["N_Train_Windows"].median()),
                "Trainable_Params": int(group["Trainable_Params"].iloc[0]),
            }
        )
        summary_rows.append(row)
        print(
            f"{dataset} W{int(winsize)} ({row['N_Subjects']} subjects): window MAE "
            f"{row['Window_MAE_Subject_Mean']:.3f} +- {row['Window_MAE_Subject_SD_base_py']:.3f} (np.std as base.py)"
            + (
                f"; 60 s minutes: subject MAE {row['Subject_Minute_MAE_Mean']:.3f} +- {row['Subject_Minute_MAE_SD']:.3f}, "
                f"median {row['Subject_Minute_MAE_Median']:.3f} [{row['Subject_Minute_MAE_Q1']:.3f}, "
                f"{row['Subject_Minute_MAE_Q3']:.3f}], {row['N_Minutes']} minutes"
                if len(part) else ""
            )
        )
    summary = pd.DataFrame(summary_rows)
    per_subject = pd.DataFrame(subject_rows)
    settings = pd.DataFrame(
        [
            ("Window_Stride_Sec", WINDOW_STRIDE_SEC),
            ("Sampling_Rate_Hz", SAMPLING_RATE),
            ("Learning_Rate", LEARNING_RATE),
            ("AdaBelief_Epsilon", EPSILON),
            ("Batch_Size", BATCH_SIZE),
            ("Max_Epochs", EPOCHS),
            ("EarlyStopping_Patience (last weights kept)", EARLY_STOP_PATIENCE),
            ("ReduceLROnPlateau", f"factor {LR_FACTOR}, patience {LR_PATIENCE}, min_delta {LR_MIN_DELTA}"),
            ("Cross_Validation", f"LOSO, KFold({N_SPLITS}, shuffle) of the other subjects into training / validation"),
            ("RR_Label", "60 / mean interval of the reference breaths in the window (>= 2 breaths)"),
            ("Window_Rule", "261105 Train_OK"),
            ("Input_Scale", "min-max [0, 1] per window after resampling to 50 Hz"),
            ("Minute_Rate", "W <= 60 s: mean of the windows inside the minute; W > 60 s: the window around the minute"),
            ("Seed", SEED),
        ],
        columns=["Setting", "Value"],
    )
    summary.to_csv(out_dir / "summary.csv", index=False)
    per_subject.to_csv(out_dir / "per_subject.csv", index=False)
    folds.to_csv(out_dir / "per_fold.csv", index=False)
    if not minutes.empty:
        minutes.to_csv(out_dir / "minutes.csv", index=False)
    try:
        with pd.ExcelWriter(out_dir / f"{SCRIPT_NAME}_results.xlsx", engine="openpyxl") as writer:
            summary.to_excel(writer, sheet_name="Summary", index=False)
            per_subject.to_excel(writer, sheet_name="Per_Subject", index=False)
            folds.to_excel(writer, sheet_name="Per_Fold", index=False)
            settings.to_excel(writer, sheet_name="Settings", index=False)
    except Exception as exc:  # the CSV files hold the same tables
        print(f"Workbook not written ({exc}); see the CSV files in {out_dir}")
    print(f"Summary written to {out_dir}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_shard(value: Optional[str]) -> Tuple[int, int]:
    if not value:
        return 0, 1
    index, count = (int(part) for part in value.split("/"))
    if count < 1 or not 0 <= index < count:
        raise ValueError("--shard must be K/N with 0 <= K < N")
    return index, count


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=SCRIPT_NAME)
    parser.add_argument("--datasets", nargs="+", choices=DATASETS, default=list(DATASETS))
    parser.add_argument("--winsize", nargs="+", type=int, choices=WINDOW_SIZES, default=list(DEFAULT_WINDOW_SIZES))
    parser.add_argument("--subjects", nargs="+", default=None, help="test subjects to run (default: all)")
    parser.add_argument("--results-root", default=str(PROJECT_ROOT))
    parser.add_argument("--base-script", default=None, help=f"path of {BASE_SCRIPT}")
    parser.add_argument("--shard", default=None, help="K/N: run every N-th test subject from K")
    parser.add_argument("--summary-only", action="store_true")
    parser.add_argument("--dry-run", action="store_true", help="read the data and count windows, no training")
    parser.add_argument("--no-require-cuda", action="store_true")
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> None:
    args = parse_args(argv)
    results_root = Path(args.results_root).resolve()
    winsizes = sorted(set(args.winsize))
    dataset_keys = [key for key in DATASETS if key in args.datasets]
    shard_index, shard_count = parse_shard(args.shard)

    if args.summary_only:
        summarize(read_folds(results_root, winsizes, dataset_keys),
                  read_minutes(results_root, winsizes, dataset_keys), results_root)
        return

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda" and not (args.no_require_cuda or args.dry_run):
        raise RuntimeError("No CUDA device; use --no-require-cuda to train on the CPU")
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True
    base_path = find_base_script(args.base_script)
    base = load_base(base_path)
    spec = base.WINDOW_SPECS[0]
    print(f"Data reading of {base_path}; device={device}; RRWaveNet "
          f"{count_parameters(RRWaveNetOriginal()):,} trainable parameters")

    for dataset_key in dataset_keys:
        label = DATASET_LABELS[dataset_key]
        try:
            records = base.load_dataset(dataset_key, spec)
        except Exception as exc:
            print(f"{label}: could not be loaded, skipped ({exc})")
            continue
        subjects = sorted(records)
        tests = [subject for subject in subjects if not args.subjects or subject in set(args.subjects)]
        tests = tests[shard_index::shard_count]
        for winsize in winsizes:
            folder = dataset_dir(results_root, winsize, dataset_key)
            data = {subject: subject_windows(base, records[subject], winsize) for subject in subjects}
            if args.dry_run:
                labelled = np.concatenate([d["windows"]["Label_RR"].to_numpy(float)[d["windows"]["Labelled"].to_numpy(bool)]
                                           for d in data.values() if not d["windows"].empty])
                per_subject = [int(d["windows"]["Labelled"].sum()) if not d["windows"].empty else 0 for d in data.values()]
                minutes = sum(int(minute_grid(base, records[s])["Ref_Valid"].astype(bool).sum())
                              if len(records[s]["starts"]) else 0 for s in subjects)
                print(
                    f"{label} W{winsize}: {len(subjects)} subjects, "
                    f"{sum(len(d['windows']) for d in data.values())} windows, {len(labelled)} labelled "
                    f"(per subject {min(per_subject)}-{max(per_subject)}), label {np.mean(labelled):.2f} +- "
                    f"{np.std(labelled, ddof=1):.2f} breaths/min ({np.min(labelled):.1f}-{np.max(labelled):.1f}), "
                    f"{minutes} reference-valid 60 s minutes"
                )
                continue
            for test_subject in tests:
                run_subject(dataset_key, winsize, subjects, test_subject, data, folder, device)
                write_subject_minutes(base, dataset_key, winsize, test_subject, records[test_subject],
                                      data[test_subject], folder)
            del data
        del records

    if not args.dry_run and shard_count == 1:
        summarize(read_folds(results_root, winsizes, dataset_keys),
                  read_minutes(results_root, winsizes, dataset_keys), results_root)
    elif not args.dry_run:
        print("Shard finished; run --summary-only when every shard is done.")


if __name__ == "__main__":
    main()
