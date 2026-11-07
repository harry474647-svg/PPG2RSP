# -*- coding: utf-8 -*-
r"""
261107_RRWaveNet_PaperStyle_RR_W16_BIDMC_CapnoBase: the 16 s windows of BIDMC
and CapnoBase first, written into the results folder of the 261106 run that
is still training STEAM2.

261106_RRWaveNet_PaperStyle_RR (and part B of 261106_DeepCorrencoder_
PaperStyle_All) runs dataset by dataset, every window size within a dataset:
STEAM2 W16, W32, W64, then BIDMC W16, W32, W64, then CapnoBase W16, W32, W64.
This script runs, in this order (default --jobs):
    BIDMC W16, CapnoBase W16
with the 261106 code unchanged (data, RR labels at 0.1 breaths/min, 0.1-0.5 Hz
input only, model, training, LOSO with inner subject-level 5-fold CV,
evaluation, file names); only the order and the results folder are new.

Results folder (the same as the running run, found automatically when
--output-root is not given): the first of
    D:\PPG2RSP_RRWaveNet_Inspired\261106_DeepCorrencoder_PaperStyle_All\RRWaveNet_PaperStyle_RR
                                         (the one-file run, part B)
    <this script's folder>\261106_RRWaveNet_PaperStyle_RR_Results
    D:\PPG2RSP_RRWaveNet_Inspired\261106_RRWaveNet_PaperStyle_RR\261106_RRWaveNet_PaperStyle_RR_Results
    D:\PPG2RSP_RRWaveNet_Inspired\261106_RRWaveNet_PaperStyle_RR_Results
that holds a STEAM2 folder (the one written most recently when several do);
without any, the script stops and asks for --output-root. Every job writes
<results folder>\<Dataset>\W<size>\ (cache, models, metrics, curves), exactly
where the 261106 run would write it.

A job is skipped when
  * its final results exist (metrics\RRWaveNet_PaperStyle_Final_Summary.csv;
    --redo runs it again), or
  * its running results were written in the last 30 minutes and no final
    results exist (another process is training it; --redo runs it anyway).
Every job starts from the same seed (SEED), so its result does not depend on
the order of the jobs.
At the end, the global tables of every finished dataset / window in the
results folder (also those of the 261106 run) are written as
RRWaveNet_PaperStyle_Global{Summary,PerSubject,PerFold}_AllFinished.csv (the
261106 run writes its own RRWaveNet_PaperStyle_Global*.csv when it ends).

!! The running 261106 run still has BIDMC W16 and CapnoBase W16 in its queue.
   Stop it (Ctrl+C) when its STEAM2 part is done, i.e. when its log prints
   "Dataset: BIDMC | Window: 16"; otherwise it trains them again and
   overwrites these results. The windows it leaves undone (BIDMC and
   CapnoBase W32 and W64) are run afterwards with
     python 261107_RRWaveNet_PaperStyle_RR_W16_BIDMC_CapnoBase.py --jobs bidmc:32 capnobase:32 bidmc:64 capnobase:64

Run (a second terminal while the 261106 run goes on; both use GPU 0):
    python 261107_RRWaveNet_PaperStyle_RR_W16_BIDMC_CapnoBase.py
Options: --jobs dataset:window ... (order kept), --output-root <folder>,
    --redo, --max-epochs, --batch-size

The experiment (unchanged from 261106_RRWaveNet_PaperStyle_RR)
---------------------------------------------------------------
261106_RRWaveNet_PaperStyle_RR (from 260514_RRWaveNet_PaperStyle_RR): the
same experiment on every dataset currently used, read as in the current
pipeline (261105_RRWaveNet_Unified_Optimized):
- CapnoBase: every case with a signal file in .csv or .tab form (CSV and
  Borealis TAB), columns found by name (pleth / co2), the sampling rate of
  each signal from the param file (CO2 resampled to the PPG rate when they
  differ); RR labels 3-60 breaths/min (neonates and children above
  45 breaths/min are kept).
- STEAM2: every session (EEGLAB .mat: EEG.data, EEG.srate, channel label
  PPG / RSP), PPG and RSP files paired by identical name only (never by
  position), a file whose label names the other signal rejected, sessions
  split at EEGLAB "boundary" events (removed data) so that no window spans a
  gap, all sessions of a person in one subject.
- BIDMC: unchanged.
RR labels: Welch peak of the reference window as in 260514, with the
spectrum zero-padded to 0.1 breaths/min bins (RR_LABEL_RESOLUTION_BPM); in
260514 the bins were 7.5 breaths/min apart.
Input: only the 0.1-0.5 Hz PPG (input case low_only; ACTIVE_INPUT_CASES).
The low_hilbert case (high-band PPG and its Hilbert envelope and frequency)
is not trained.
Model, preprocessing (SQI, filters, z-score), training and evaluation are
those of 260514. Results go to a new folder
(261106_RRWaveNet_PaperStyle_RR_Results), so no cache of 260514 is reused.

Paper-style RRWaveNet for direct respiration-rate regression with input-case
comparison.

Goal
- Predict scalar respiration rate (RR) directly, in the style of the RRWaveNet
  paper.
- Reuse the same STEAM / BIDMC / CapnoBase data sources used by the earlier
  waveform script.
- Compare two input cases inspired by the user's Hilbert-comparison code:
    * `low_only`: low-frequency PPG only
    * `low_hilbert`: low-frequency PPG + high-frequency Hilbert branch
- Evaluate with LOSO and inner subject-level 5-fold CV.
- Report MAE in breaths per minute.

Input-case definition
- `low_only`
  Uses only the low-band PPG (0.1-0.5 Hz).
- `low_hilbert`
  Uses the same low-band PPG plus a second branch built from:
    * high-band PPG (0.5-5.0 Hz)
    * Hilbert envelope of the high-band PPG
    * Hilbert instantaneous frequency of the high-band PPG

Paper-facing design choices
- Targets are scalar RR labels derived from the reference respiration window.
- Windows are resampled to 50 Hz with 2-second stride.
- SQI filtering, MSE training, and MAE reporting are retained.
- The RRWaveNet-style backbone uses multi-scale convolution, 8 residual blocks,
  global average pooling, and a small FC head.
"""

from __future__ import annotations

import argparse
import csv
import math
import os
import re
import random
import sys
import time
from contextlib import nullcontext
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import matplotlib
matplotlib.use("Agg")

import numpy as np
import pandas as pd
import scipy.io as sio
import torch
import torch.nn as nn
import torch.nn.functional as F
from matplotlib import pyplot as plt
from scipy.signal import butter, find_peaks, hilbert, resample, resample_poly, savgol_filter, sosfiltfilt, welch
from torch.optim import Optimizer
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader, Dataset


# ===========================================================
# 1. Global Config
# ===========================================================
SEED = 42
GPU_NUM = 0
device = torch.device(f"cuda:{GPU_NUM}" if torch.cuda.is_available() else "cpu")
if device.type == "cuda":
    torch.cuda.set_device(device)

USE_AMP = True
AMP_ENABLED = USE_AMP and device.type == "cuda"
NUM_WORKERS = 0 if os.name == "nt" else min(4, os.cpu_count() or 1)
PIN_MEMORY = device.type == "cuda"

WORKSPACE_ROOT = os.path.abspath(os.path.dirname(__file__))
DEFAULT_SAVE_ROOT = os.path.join(WORKSPACE_ROOT, "261106_RRWaveNet_PaperStyle_RR_Results")

# Dataset sources of the current pipeline (261105).
STEAM2_PPG_DIR = r"D:\STEAM\Multimodal_prediction\PPG-RSP\Preprocessed\PPG\2back_raw_v2"
STEAM2_RSP_DIR = r"D:\STEAM\Multimodal_prediction\PPG-RSP\Preprocessed\RSP\2back_raw_v2"
BIDMC_DATA_PATH = (
    r"D:\STEAM\Multimodal_prediction\PPG-RSP\Preprocessed"
    r"\bidmc-ppg-and-respiration-dataset-1.0.0"
    r"\bidmc-ppg-and-respiration-dataset-1.0.0\bidmc_data.mat"
)
CAPNOBASE_CSV_DIR = r"D:\STEAM\Multimodal_prediction\PPG-RSP\Opendata\Capnobase\data\csv"

ACTIVE_DATASETS = ("steam2", "bidmc", "capnobase")
WINDOW_SIZES_SEC = (16.0, 32.0, 64.0)
STRIDE_SEC = 2.0
TARGET_FS = 50

INPUT_CASES = {
    "low_only": {
        "label": "Low Only",
        "use_hilbert_branch": False,
    },
    "low_hilbert": {
        "label": "Low + Hilbert Branch",
        "use_hilbert_branch": True,
    },
}
# 261106: only the 0.1-0.5 Hz PPG input. Add "low_hilbert" to train the
# Hilbert-branch case of 260514 again.
ACTIVE_INPUT_CASES = ("low_only",)

# Paper-facing preprocessing
ENABLE_SQI_FILTER = True
SQI_THRESHOLD = 0.90
FLATLINE_DELTA = 0.02
FLATLINE_MIN_POINTS = 30
PEAK_MATCH_MS = 150

# RR label extraction from the reference respiration waveform. RR_MIN_BPM /
# RR_MAX_BPM are the default range; a dataset can set its own (rr_band_bpm in
# DATASET_CONFIGS: CapnoBase 3-60 breaths/min as in 261105).
RR_MIN_BPM = 4.0
RR_MAX_BPM = 45.0
RESP_LOW_HZ = RR_MIN_BPM / 60.0
RESP_HIGH_HZ = RR_MAX_BPM / 60.0
RESP_BANDPASS_ORDER = 3
# RR label resolution: the Welch spectrum is zero-padded to
# nfft = fs * 60 / RR_LABEL_RESOLUTION_BPM points (30000 at 50 Hz), so its
# bins are 0.1 breaths/min apart. Without it (nfft = nperseg = 8 s) the bins
# were 0.125 Hz = 7.5 breaths/min apart and the labels took only the values
# 7.5, 15, 22.5, ...
RR_LABEL_RESOLUTION_BPM = 0.1
LOW_BAND_HZ = (0.1, 0.5)
HIGH_BAND_HZ = (0.5, 5.0)

# Model config.
# The paper figure explicitly exposes a k=32 branch. The companion branches are
# kept configurable here.
MULTISCALE_KERNELS = (32, 64, 128)
RESIDUAL_FEATURES = (64, 64, 128, 128, 256, 256, 512, 512)
HILBERT_AUX_RESIDUAL_FEATURES = (16, 16, 32, 32, 64, 64, 96, 96)
RESIDUAL_KERNEL_SIZE = 5
FC_HIDDEN_DIMS = (128, 64)
LOW_HEAD_DROPOUT = 0.20
HILBERT_HEAD_DROPOUT = 0.30
FEATURE_DROPOUT_LOW = 0.10
FEATURE_DROPOUT_HIGH = 0.20
FUSION_HIDDEN_DIM = 64

# Training config from the paper text where available.
INNER_FOLDS = 5
BATCH_SIZE = 64
MAX_EPOCHS = 69420
EARLY_STOP_PATIENCE = 10
LEARNING_RATE = 1.0e-4
ADABELIEF_EPS = 1.0e-13
ADABELIEF_BETAS = (0.9, 0.999)
WEIGHT_DECAY = 1.0e-4
SCHEDULER_FACTOR = 0.5
SCHEDULER_PATIENCE = 3
MIN_LR = 5.0e-6
GRAD_CLIP_NORM = 1.0
TARGET_STD_EPS = 1.0e-6
LOSS_BETA = 0.75


DATASET_CONFIGS = {
    "steam2": {
        "label": "STEAM2",
        "save_name": "STEAM2",
        "ppg_dir": STEAM2_PPG_DIR,
        "rsp_dir": STEAM2_RSP_DIR,
        "source_fs": None,
        # EEGLAB sessions (EEG.srate = 200 Hz); used only when a file has no srate.
        "default_fs": 200,
        "rr_band_bpm": (RR_MIN_BPM, RR_MAX_BPM),
    },
    "bidmc": {
        "label": "BIDMC",
        "save_name": "BIDMC",
        "data_path": BIDMC_DATA_PATH,
        "source_fs": 125,
        "rr_band_bpm": (RR_MIN_BPM, RR_MAX_BPM),
    },
    "capnobase": {
        "label": "CapnoBase",
        "save_name": "CapnoBase",
        "data_path": CAPNOBASE_CSV_DIR,
        "source_fs": None,
        # Used only when the param file has no sampling rate.
        "default_fs": 300,
        # Neonates and children breathe above 45 breaths/min (261105:
        # capnography_wide_rr, 3-60 breaths/min).
        "rr_band_bpm": (3.0, 60.0),
    },
}


def set_seed(seed: int = SEED, deterministic: bool = False) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = deterministic
        torch.backends.cudnn.benchmark = not deterministic
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        try:
            torch.set_float32_matmul_precision("high")
        except Exception:
            pass


set_seed(SEED, deterministic=False)


# ===========================================================
# 2. Paths / Small Helpers
# ===========================================================
def safe_tag(text: str) -> str:
    return "".join(ch if ch.isalnum() or ch in ("-", "_", ".") else "_" for ch in str(text))


def prepare_output_dirs(output_root: str, dataset_key: str, window_sec: float) -> Dict[str, str]:
    root = os.path.join(output_root, DATASET_CONFIGS[dataset_key]["save_name"], f"W{int(window_sec)}")
    dirs = {
        "root": root,
        "cache": os.path.join(root, "cache"),
        "models": os.path.join(root, "models"),
        "metrics": os.path.join(root, "metrics"),
        "curves": os.path.join(root, "curves"),
    }
    for path in dirs.values():
        os.makedirs(path, exist_ok=True)
    return dirs


def autocast_context():
    if AMP_ENABLED:
        return torch.autocast(device_type="cuda", dtype=torch.float16)
    return nullcontext()


def count_trainable_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


# ===========================================================
# 3. Signal Utilities
# ===========================================================
def sanitize_signal(sig: np.ndarray) -> np.ndarray:
    sig = np.asarray(sig, dtype=np.float64).squeeze()
    if sig.ndim != 1:
        sig = sig.reshape(-1)
    if sig.size == 0:
        return sig.astype(np.float64)
    finite_mask = np.isfinite(sig)
    if not finite_mask.any():
        return np.zeros_like(sig, dtype=np.float64)
    fill_value = float(np.median(sig[finite_mask]))
    out = sig.copy()
    out[~finite_mask] = fill_value
    return out.astype(np.float64)


def bandpass_filter(sig: np.ndarray, fs: float, low_hz: float, high_hz: float, order: int = 3) -> np.ndarray:
    sig = sanitize_signal(sig)
    if sig.size < max(32, order * 6 + 1):
        return sig.astype(np.float32)
    nyq = 0.5 * float(fs)
    low_hz = max(float(low_hz), 1e-5)
    high_hz = min(float(high_hz), nyq * 0.99)
    if not np.isfinite(low_hz) or not np.isfinite(high_hz) or high_hz <= low_hz:
        return sig.astype(np.float32)
    try:
        sos = butter(order, [low_hz / nyq, high_hz / nyq], btype="band", output="sos")
        return sosfiltfilt(sos, sig).astype(np.float32)
    except Exception:
        return sig.astype(np.float32)


def get_hilbert_features(sig: np.ndarray, fs: float) -> Tuple[np.ndarray, np.ndarray]:
    sig = sanitize_signal(sig)
    analytic = hilbert(sig)
    envelope = np.abs(analytic)
    phase = np.unwrap(np.angle(analytic))
    inst_freq = np.diff(phase) / (2.0 * np.pi) * float(fs)
    if inst_freq.size == 0:
        inst_freq = np.zeros_like(sig)
    else:
        inst_freq = np.insert(inst_freq, 0, inst_freq[0])
    envelope = np.nan_to_num(envelope, nan=0.0, posinf=0.0, neginf=0.0)
    inst_freq = np.nan_to_num(inst_freq, nan=0.0, posinf=0.0, neginf=0.0)
    return envelope.astype(np.float32), inst_freq.astype(np.float32)


def maybe_resample(sig: np.ndarray, orig_fs: float, target_fs: float) -> np.ndarray:
    sig = sanitize_signal(sig)
    if int(round(orig_fs)) == int(round(target_fs)):
        return sig.astype(np.float32)
    target_len = int(round(len(sig) * float(target_fs) / float(orig_fs)))
    target_len = max(target_len, 1)
    return resample(sig, target_len).astype(np.float32)


def resample_signal_poly(sig: np.ndarray, src_fs: float, dst_fs: float) -> np.ndarray:
    sig = sanitize_signal(sig)
    src_fs = float(src_fs)
    dst_fs = float(dst_fs)
    if int(round(src_fs)) == int(round(dst_fs)):
        return sig.astype(np.float32)
    src_i = int(round(src_fs))
    dst_i = int(round(dst_fs))
    div = math.gcd(src_i, dst_i)
    up = max(dst_i // div, 1)
    down = max(src_i // div, 1)
    return resample_poly(sig, up=up, down=down).astype(np.float32)


def extract_windows(sig: np.ndarray, fs: float, win_sec: float, stride_sec: float) -> np.ndarray:
    sig = np.asarray(sig)
    win_len = int(round(float(fs) * float(win_sec)))
    stride_len = int(round(float(fs) * float(stride_sec)))
    if sig.shape[-1] < win_len or win_len <= 0 or stride_len <= 0:
        return np.array([])
    windows = []
    for start in range(0, sig.shape[-1] - win_len + 1, stride_len):
        end = start + win_len
        windows.append(sig[start:end])
    return np.asarray(windows)


def zscore_per_window(data: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    data = np.asarray(data, dtype=np.float32)
    mean = np.mean(data, axis=-1, keepdims=True)
    std = np.std(data, axis=-1, keepdims=True)
    std = np.where(std < eps, 1.0, std)
    return ((data - mean) / std).astype(np.float32)


def normalize_unit_range(sig: np.ndarray) -> np.ndarray:
    sig = sanitize_signal(sig)
    lo = float(np.min(sig))
    hi = float(np.max(sig))
    if not np.isfinite(lo) or not np.isfinite(hi) or hi - lo < 1e-8:
        return np.zeros_like(sig, dtype=np.float64)
    return ((sig - lo) / (hi - lo)).astype(np.float64)


# ===========================================================
# 4. SQI and RR Label Extraction
# ===========================================================
def _flatline_mask(sig: np.ndarray, delta: float = FLATLINE_DELTA, min_points: int = FLATLINE_MIN_POINTS) -> np.ndarray:
    sig = normalize_unit_range(sig)
    if sig.size == 0:
        return np.array([], dtype=bool)
    diff = np.abs(np.diff(sig))
    mask = np.zeros(sig.shape[0], dtype=bool)
    run_start = None
    for idx, value in enumerate(diff):
        if value < delta:
            if run_start is None:
                run_start = idx
        else:
            if run_start is not None and (idx - run_start + 1) >= min_points:
                mask[run_start:idx + 1] = True
            run_start = None
    if run_start is not None and (len(diff) - run_start + 1) >= min_points:
        mask[run_start:] = True
    return mask


def _smooth_for_peak_detection(sig: np.ndarray) -> np.ndarray:
    sig = normalize_unit_range(sig)
    if sig.size < 7:
        return sig
    window = min(sig.size - (1 - sig.size % 2), 31)
    window = max(window, 7)
    if window % 2 == 0:
        window -= 1
    try:
        return savgol_filter(sig, window_length=window, polyorder=3, mode="interp")
    except Exception:
        return sig


def primary_peak_detector(sig: np.ndarray, fs: float) -> np.ndarray:
    x = _smooth_for_peak_detection(sig)
    distance = max(1, int(round(0.30 * fs)))
    prominence = max(0.03, float(np.std(x)) * 0.12)
    peaks, _ = find_peaks(x, distance=distance, prominence=prominence)
    return peaks.astype(np.int64)


def secondary_peak_detector(sig: np.ndarray, fs: float) -> np.ndarray:
    x = _smooth_for_peak_detection(sig)
    distance = max(1, int(round(0.28 * fs)))
    prominence = max(0.02, float(np.std(x)) * 0.08)
    width_lo = max(1, int(round(0.03 * fs)))
    width_hi = max(width_lo + 1, int(round(0.40 * fs)))
    peaks, _ = find_peaks(x, distance=distance, prominence=prominence, width=(width_lo, width_hi))
    return peaks.astype(np.int64)


def f1_score_from_peaks(peaks_a: np.ndarray, peaks_b: np.ndarray, tolerance_samples: int) -> float:
    peaks_a = np.asarray(peaks_a, dtype=np.int64)
    peaks_b = np.asarray(peaks_b, dtype=np.int64)
    if peaks_a.size == 0 and peaks_b.size == 0:
        return 1.0
    if peaks_a.size == 0 or peaks_b.size == 0:
        return 0.0

    matched_b = np.zeros(peaks_b.size, dtype=bool)
    tp = 0
    for peak in peaks_a:
        distances = np.abs(peaks_b - peak)
        idx = int(np.argmin(distances))
        if distances[idx] <= tolerance_samples and not matched_b[idx]:
            matched_b[idx] = True
            tp += 1
    fp = int(peaks_a.size - tp)
    fn = int(peaks_b.size - tp)
    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    if precision + recall <= 0:
        return 0.0
    return float(2.0 * precision * recall / (precision + recall))


def compute_sqi(sig: np.ndarray, fs: float) -> float:
    sig = sanitize_signal(sig)
    flat_mask = _flatline_mask(sig)
    if flat_mask.size == 0:
        return 0.0
    k_value = float(np.sum(~flat_mask) / max(flat_mask.size, 1))
    tol = max(1, int(round((PEAK_MATCH_MS / 1000.0) * fs)))
    peaks_a = primary_peak_detector(sig, fs)
    peaks_b = secondary_peak_detector(sig, fs)
    f1 = f1_score_from_peaks(peaks_a, peaks_b, tol)
    return float(k_value * f1)


def estimate_rr_bpm_from_reference(
    sig: np.ndarray, fs: float, rr_band_bpm: Tuple[float, float] = (RR_MIN_BPM, RR_MAX_BPM)
) -> float:
    rr_min, rr_max = float(rr_band_bpm[0]), float(rr_band_bpm[1])
    low_hz, high_hz = rr_min / 60.0, rr_max / 60.0
    sig = sanitize_signal(sig)
    sig = bandpass_filter(sig, fs, low_hz, high_hz, order=RESP_BANDPASS_ORDER)
    if sig.size < int(fs * 4):
        return np.nan
    nperseg = min(sig.size, int(fs * 8))
    nfft = max(nperseg, int(round(float(fs) * 60.0 / RR_LABEL_RESOLUTION_BPM)))
    freqs, psd = welch(sig, fs=fs, nperseg=nperseg, nfft=nfft)
    mask = (freqs >= low_hz) & (freqs <= high_hz)
    if not np.any(mask):
        return np.nan
    band_freqs = freqs[mask]
    band_psd = psd[mask]
    if band_freqs.size == 0 or not np.any(np.isfinite(band_psd)):
        return np.nan
    peak_idx = int(np.argmax(band_psd))
    bpm = float(np.round(band_freqs[peak_idx] * 60.0 / RR_LABEL_RESOLUTION_BPM) * RR_LABEL_RESOLUTION_BPM)
    if not np.isfinite(bpm) or bpm < rr_min or bpm > rr_max:
        return np.nan
    return bpm


@dataclass
class SubjectRRPack:
    x_low: np.ndarray
    x_high: np.ndarray
    y_rr: np.ndarray
    fs: int


@dataclass
class RRTargetScaler:
    mean: float
    std: float

    @classmethod
    def fit(cls, values: np.ndarray) -> "RRTargetScaler":
        arr = np.asarray(values, dtype=np.float32).reshape(-1)
        mean = float(np.mean(arr))
        std = float(np.std(arr))
        if not np.isfinite(std) or std < TARGET_STD_EPS:
            std = 1.0
        return cls(mean=mean, std=std)

    def transform(self, values: np.ndarray) -> np.ndarray:
        arr = np.asarray(values, dtype=np.float32)
        return ((arr - self.mean) / self.std).astype(np.float32)

    def inverse_transform(self, values: np.ndarray) -> np.ndarray:
        arr = np.asarray(values, dtype=np.float32)
        return (arr * self.std + self.mean).astype(np.float32)


def _collect_rr_windows(
    raw_windows: np.ndarray,
    low_windows: np.ndarray,
    high_windows: np.ndarray,
    env_windows: np.ndarray,
    freq_windows: np.ndarray,
    y_windows: np.ndarray,
    target_fs: int,
    apply_sqi: bool,
    rr_band_bpm: Tuple[float, float] = (RR_MIN_BPM, RR_MAX_BPM),
) -> Tuple[List[np.ndarray], List[np.ndarray], List[float]]:
    kept_x_low: List[np.ndarray] = []
    kept_x_high: List[np.ndarray] = []
    kept_y_rr: List[float] = []

    n_windows = min(len(raw_windows), len(low_windows), len(high_windows), len(env_windows), len(freq_windows), len(y_windows))
    for idx in range(n_windows):
        x_raw_win = raw_windows[idx].astype(np.float32)
        x_low_win = low_windows[idx].astype(np.float32)
        x_high_win = high_windows[idx].astype(np.float32)
        env_win = env_windows[idx].astype(np.float32)
        freq_win = freq_windows[idx].astype(np.float32)
        y_win = y_windows[idx].astype(np.float32)

        if apply_sqi:
            sqi = compute_sqi(x_raw_win, target_fs)
            if not np.isfinite(sqi) or sqi < SQI_THRESHOLD:
                continue

        rr_bpm = estimate_rr_bpm_from_reference(y_win, target_fs, rr_band_bpm)
        if not np.isfinite(rr_bpm):
            continue

        if float(np.std(x_low_win)) < 1e-8:
            continue

        high_stack = np.stack([x_high_win, env_win, freq_win], axis=0).astype(np.float32)
        kept_x_low.append(x_low_win.copy())
        kept_x_high.append(high_stack.copy())
        kept_y_rr.append(float(rr_bpm))

    return kept_x_low, kept_x_high, kept_y_rr


def preprocess_subject_rr_windows(
    ppg_raw: np.ndarray,
    target_raw: np.ndarray,
    orig_fs: float,
    window_sec: float,
    stride_sec: float = STRIDE_SEC,
    target_fs: int = TARGET_FS,
    enable_sqi: bool = ENABLE_SQI_FILTER,
    rr_band_bpm: Tuple[float, float] = (RR_MIN_BPM, RR_MAX_BPM),
) -> SubjectRRPack | None:
    ppg_raw = sanitize_signal(ppg_raw)
    target_raw = sanitize_signal(target_raw)
    min_len = min(len(ppg_raw), len(target_raw))
    ppg_raw = ppg_raw[:min_len]
    target_raw = target_raw[:min_len]

    valid = np.isfinite(ppg_raw) & np.isfinite(target_raw)
    ppg_raw = ppg_raw[valid]
    target_raw = target_raw[valid]
    if ppg_raw.size == 0 or target_raw.size == 0:
        return None

    ppg = maybe_resample(ppg_raw, orig_fs, target_fs).astype(np.float32)
    target = maybe_resample(target_raw, orig_fs, target_fs).astype(np.float32)
    min_len = min(len(ppg), len(target))
    ppg = ppg[:min_len]
    target = target[:min_len]

    ppg_low = bandpass_filter(ppg, target_fs, LOW_BAND_HZ[0], LOW_BAND_HZ[1], order=RESP_BANDPASS_ORDER)
    ppg_high = bandpass_filter(ppg, target_fs, HIGH_BAND_HZ[0], HIGH_BAND_HZ[1], order=RESP_BANDPASS_ORDER)
    env, freq = get_hilbert_features(ppg_high, target_fs)
    target_resp = bandpass_filter(
        target, target_fs, rr_band_bpm[0] / 60.0, rr_band_bpm[1] / 60.0, order=RESP_BANDPASS_ORDER
    )

    raw_windows = extract_windows(ppg, target_fs, window_sec, stride_sec)
    low_windows = extract_windows(ppg_low, target_fs, window_sec, stride_sec)
    high_windows = extract_windows(ppg_high, target_fs, window_sec, stride_sec)
    env_windows = extract_windows(env, target_fs, window_sec, stride_sec)
    freq_windows = extract_windows(freq, target_fs, window_sec, stride_sec)
    y_windows = extract_windows(target_resp, target_fs, window_sec, stride_sec)
    n_windows = min(len(raw_windows), len(low_windows), len(high_windows), len(env_windows), len(freq_windows), len(y_windows))
    if n_windows == 0:
        return None

    kept_x_low, kept_x_high, kept_y_rr = _collect_rr_windows(
        raw_windows=raw_windows,
        low_windows=low_windows,
        high_windows=high_windows,
        env_windows=env_windows,
        freq_windows=freq_windows,
        y_windows=y_windows,
        target_fs=target_fs,
        apply_sqi=bool(enable_sqi),
        rr_band_bpm=rr_band_bpm,
    )

    if not kept_x_low and enable_sqi:
        kept_x_low, kept_x_high, kept_y_rr = _collect_rr_windows(
            raw_windows=raw_windows,
            low_windows=low_windows,
            high_windows=high_windows,
            env_windows=env_windows,
            freq_windows=freq_windows,
            y_windows=y_windows,
            target_fs=target_fs,
            apply_sqi=False,
            rr_band_bpm=rr_band_bpm,
        )

    if not kept_x_low:
        return None

    x_low_arr = np.stack(kept_x_low, axis=0).astype(np.float32)
    x_high_arr = np.stack(kept_x_high, axis=0).astype(np.float32)
    x_low_arr = zscore_per_window(x_low_arr)
    x_high_arr = zscore_per_window(x_high_arr)
    y_arr = np.asarray(kept_y_rr, dtype=np.float32)
    return SubjectRRPack(x_low=x_low_arr, x_high=x_high_arr, y_rr=y_arr, fs=int(target_fs))


# ===========================================================
# 5. Dataset Loading
# ===========================================================
def get_cache_path(cache_dir: str, dataset_key: str, subject: str, window_sec: float, enable_sqi: bool) -> str:
    sqi_tag = "sqi1" if enable_sqi else "sqi0"
    # The label resolution is part of the name: caches with the old 7.5
    # breaths/min labels are never read.
    tag = f"{dataset_key}_{safe_tag(subject)}_w{float(window_sec):g}_{sqi_tag}_rr{RR_LABEL_RESOLUTION_BPM:g}.npz"
    return os.path.join(cache_dir, tag)


def load_subject_cache(cache_path: str) -> SubjectRRPack | None:
    if not os.path.exists(cache_path):
        return None
    try:
        cached = np.load(cache_path, allow_pickle=False)
        return SubjectRRPack(
            x_low=cached["x_low"].astype(np.float32),
            x_high=cached["x_high"].astype(np.float32),
            y_rr=cached["y_rr"].astype(np.float32),
            fs=int(cached["fs"]),
        )
    except Exception:
        return None


def save_subject_cache(cache_path: str, pack: SubjectRRPack) -> None:
    os.makedirs(os.path.dirname(cache_path), exist_ok=True)
    np.savez_compressed(
        cache_path,
        x_low=pack.x_low.astype(np.float32),
        x_high=pack.x_high.astype(np.float32),
        y_rr=pack.y_rr.astype(np.float32),
        fs=np.asarray(pack.fs, dtype=np.int32),
    )


# STEAM2 (EEGLAB .mat sessions) and CapnoBase (CSV / TAB) readers of the
# current pipeline (261105_RRWaveNet_Unified_Optimized).
STEAM2_CHANNELS = {"PPG": ("PPG", "PLETH", "BVP"), "RSP": ("RSP", "RESP", "RESPIRATION", "BELT")}
STEAM2_SESSION_PATTERN = re.compile(r"^(?P<subject>.+?)[_\- ]*session[_\- ]*(?P<session>\d+)$", re.IGNORECASE)
CAPNOBASE_FILE_SUFFIXES = (".csv", ".tab")
_NUMBER_PATTERN = re.compile(r"[-+]?(?:\d+\.?\d*|\.\d+)(?:[eE][-+]?\d+)?")


def _unwrap_mat_value(value):
    arr = np.asarray(value)
    while arr.dtype == object and arr.size == 1:
        arr = np.asarray(arr.item())
    return arr


def _first_mat_payload(mat):
    valid_keys = [k for k in mat.keys() if not k.startswith("_")]
    if not valid_keys:
        raise KeyError("No valid keys found in MAT file.")
    payload = _unwrap_mat_value(mat[valid_keys[0]])
    if payload.ndim >= 2 and payload.shape[0] == 1 and payload.shape[1] == 1:
        payload = payload[0, 0]
    return payload


def _extract_signal_and_fs(payload, default_fs: float):
    signal = sanitize_signal(_unwrap_mat_value(payload["data"]))
    try:
        srate = float(np.asarray(_unwrap_mat_value(payload["srate"])).squeeze())
    except Exception:
        srate = float(default_fs)
    if not np.isfinite(srate) or srate <= 0:
        srate = float(default_fs)
    return signal, srate


def _as_list(value) -> list:
    if value is None:
        return []
    if isinstance(value, np.ndarray):
        return list(value.ravel()) if value.size else []
    return [value]


def steam2_session_key(stem: str) -> Tuple[str, int]:
    """(subject, session number) from a file stem such as 13_LSB_Session2."""
    match = STEAM2_SESSION_PATTERN.match(stem)
    if match is None:
        return stem, 0
    return match.group("subject"), int(match.group("session"))


def read_eeglab_channel(path: str, kind: str, default_fs: float) -> dict:
    """One channel of an EEGLAB .mat file, chosen by its label; a one-channel
    file without a label is accepted, one labelled as the other signal is
    rejected (swapped files). The positions of "boundary" events inside the
    recording are returned so that the session is split there."""
    mat = sio.loadmat(path, squeeze_me=True, struct_as_record=False)
    names = [key for key in mat if not key.startswith("_")]
    payload = mat["EEG"] if "EEG" in mat else (mat[names[0]] if names else None)
    if payload is None or not hasattr(payload, "data"):
        signal, fs = _extract_signal_and_fs(_first_mat_payload(sio.loadmat(path)), default_fs)
        return {"signal": signal, "fs": fs, "boundaries": [], "label": ""}
    data = np.asarray(payload.data, dtype=np.float64)
    data = data.reshape(1, -1) if data.ndim == 1 else data
    labels = [str(getattr(item, "labels", "")).strip() for item in _as_list(getattr(payload, "chanlocs", None))]
    wanted = {name.upper() for name in STEAM2_CHANNELS[kind]}
    other = {name.upper() for key, names in STEAM2_CHANNELS.items() if key != kind for name in names}
    index = next((i for i, label in enumerate(labels) if label.upper() in wanted), None)
    if index is None:
        if data.shape[0] != 1:
            raise RuntimeError(f"{os.path.basename(path)}: no {kind} channel among {labels}")
        if labels and labels[0].upper() in other:
            raise RuntimeError(f"{os.path.basename(path)}: channel '{labels[0]}' found where {kind} was expected")
        index = 0
    signal = sanitize_signal(data[index])
    try:
        fs = float(np.asarray(payload.srate).squeeze())
    except Exception:
        fs = float(default_fs)
    if not np.isfinite(fs) or fs <= 0:
        fs = float(default_fs)
    boundaries = []
    for event in _as_list(getattr(payload, "event", None)):
        if str(getattr(event, "type", "")).strip().lower() != "boundary":
            continue
        try:
            latency = float(getattr(event, "latency"))
        except (TypeError, ValueError):
            continue
        # EEGLAB latencies are 1-based; a boundary at k.5 lies between samples
        # k and k + 1. Markers at the start and at the end are not gaps.
        position = int(np.ceil(latency - 1.0))
        if 0 < position < signal.size:
            boundaries.append(position)
    label = labels[index] if index < len(labels) else ""
    return {"signal": signal, "fs": fs, "boundaries": sorted(set(boundaries)), "label": label}


def steam2_session_files(ppg_root: str, rsp_root: str) -> Dict[str, List[Tuple[str, str, str]]]:
    """Subject -> [(session name, PPG file, RSP file)], sessions in order.
    Subject folders are used when present; otherwise the files of one flat
    folder are grouped by the subject part of their name. PPG and RSP files
    are paired by identical name only; a file without a partner is reported
    and left out."""
    ppg_root, rsp_root = Path(ppg_root), Path(rsp_root)
    sessions: Dict[str, List[Tuple[str, str, str]]] = {}
    subject_dirs = sorted(path for path in ppg_root.iterdir() if path.is_dir())
    if subject_dirs:
        for subject_dir in subject_dirs:
            rsp_dir = rsp_root / subject_dir.name
            if not rsp_dir.is_dir():
                print(f"   [Skip] STEAM2 {subject_dir.name}: missing RSP folder")
                continue
            rsp_files = {path.stem: path for path in rsp_dir.glob("*.mat")}
            ppg_files = sorted(subject_dir.glob("*.mat"))
            for ppg_path in ppg_files:
                if ppg_path.stem not in rsp_files:
                    print(f"   [Skip] STEAM2 {subject_dir.name} {ppg_path.name}: no RSP file of the same name")
                    continue
                sessions.setdefault(subject_dir.name, []).append(
                    (ppg_path.stem, str(ppg_path), str(rsp_files[ppg_path.stem]))
                )
            for stem in sorted(set(rsp_files) - {path.stem for path in ppg_files}):
                print(f"   [Skip] STEAM2 {subject_dir.name} {stem}.mat: no PPG file of the same name")
    else:
        rsp_files = {path.stem: path for path in rsp_root.glob("*.mat")}
        for ppg_path in sorted(ppg_root.glob("*.mat")):
            if ppg_path.stem not in rsp_files:
                print(f"   [Skip] STEAM2 {ppg_path.name}: no RSP file of the same name")
                continue
            subject, _ = steam2_session_key(ppg_path.stem)
            sessions.setdefault(subject, []).append((ppg_path.stem, str(ppg_path), str(rsp_files[ppg_path.stem])))
    for subject in sessions:
        sessions[subject].sort(key=lambda item: (steam2_session_key(item[0])[1], item[0]))
    return sessions


def steam2_session_pieces(ppg: dict, rsp: dict) -> Tuple[List[Tuple[np.ndarray, np.ndarray, float, float]], float]:
    """The (PPG, RSP, PPG fs, RSP fs) pieces of one session, split at the
    boundary events of either file, and the PPG - RSP length difference (s)."""
    ppg_sec = ppg["signal"].size / ppg["fs"]
    rsp_sec = rsp["signal"].size / rsp["fs"]
    cuts = sorted({b / ppg["fs"] for b in ppg["boundaries"]} | {b / rsp["fs"] for b in rsp["boundaries"]})
    edges = [0.0] + [cut for cut in cuts if 0.0 < cut < min(ppg_sec, rsp_sec)] + [min(ppg_sec, rsp_sec)]
    pieces = []
    for start, end in zip(edges[:-1], edges[1:]):
        p = ppg["signal"][int(round(start * ppg["fs"])):int(round(end * ppg["fs"]))]
        r = rsp["signal"][int(round(start * rsp["fs"])):int(round(end * rsp["fs"]))]
        if p.size and r.size:
            pieces.append((p, r, ppg["fs"], rsp["fs"]))
    return pieces, ppg_sec - rsp_sec


def capnobase_file(csv_dir: str, case_id: str, kind: str) -> Optional[str]:
    for suffix in CAPNOBASE_FILE_SUFFIXES:
        path = os.path.join(csv_dir, f"{case_id}_8min_{kind}{suffix}")
        if os.path.exists(path):
            return path
    return None


def capnobase_case_ids(csv_dir: str) -> List[str]:
    return sorted(
        {
            name.split("_")[0]
            for name in os.listdir(csv_dir)
            if any(name.endswith(f"_8min_signal{suffix}") for suffix in CAPNOBASE_FILE_SUFFIXES)
        }
    )


def _sniff_delimiter(path: str) -> str:
    with open(path, newline="", encoding="utf-8-sig") as handle:
        first = handle.readline()
    return "\t" if first.count("\t") > first.count(",") else ","


def _normalize_field_name(name: object) -> str:
    return re.sub(r"[^0-9a-z]+", "_", str(name).lower()).strip("_")


def read_capnobase_fields(path: str) -> Dict[str, List[str]]:
    """Fields of a CapnoBase param file, one per column (header row, then
    value rows) or one per row (name in the first cell, values after it)."""
    with open(path, newline="", encoding="utf-8-sig") as handle:
        rows = [
            [cell.strip() for cell in row]
            for row in csv.reader(handle, delimiter=_sniff_delimiter(path))
            if any(cell.strip() for cell in row)
        ]
    fields: Dict[str, List[str]] = {}
    if not rows:
        return fields
    header = rows[0]
    column_layout = all(len(row) <= len(header) for row in rows[1:]) and not any(
        _NUMBER_PATTERN.fullmatch(cell) for cell in header if cell
    )
    if column_layout and len(header) == 2 and len(rows) > 2:
        names = [row[0] for row in rows[1:] if row]
        if names and not any(_NUMBER_PATTERN.fullmatch(name) for name in names if name):
            column_layout = False
            rows = rows[1:]
    if column_layout:
        for index, name in enumerate(header):
            if name:
                fields[_normalize_field_name(name)] = [
                    token for row in rows[1:] if index < len(row) for token in row[index].split()
                ]
    else:
        for row in rows:
            if row and row[0]:
                fields[_normalize_field_name(row[0])] = [token for cell in row[1:] for token in cell.split()]
    return fields


def capnobase_sampling_rate(fields: Dict[str, List[str]], signal_name: str, default: float) -> float:
    names = [name for name in fields if "samplingrate" in name and signal_name in name] or [
        name for name in fields if "rate" in name and signal_name in name
    ]
    if names:
        values = [float(m) for token in fields[min(names, key=len)] for m in _NUMBER_PATTERN.findall(token)]
        values = [value for value in values if np.isfinite(value) and value > 0]
        if values:
            return float(values[0])
    return float(default)


def read_capnobase_signals(path: str) -> Tuple[np.ndarray, np.ndarray]:
    """PPG (pleth) and capnogram (co2) columns of a signal file (CSV or TAB)."""
    frame = pd.read_csv(path, sep=_sniff_delimiter(path))
    frame.columns = [_normalize_field_name(column) for column in frame.columns]

    def column(signal_name: str) -> str:
        names = [name for name in frame.columns if signal_name in name]
        names = [name for name in names if name.endswith("_y")] or names
        if not names:
            raise KeyError(f"{os.path.basename(path)}: no {signal_name} column in {list(frame.columns)}")
        return min(names, key=len)

    ppg = pd.to_numeric(frame[column("pleth")], errors="coerce").to_numpy(np.float64)
    co2 = pd.to_numeric(frame[column("co2")], errors="coerce").to_numpy(np.float64)
    return ppg, co2


def load_bidmc_dataset(window_sec: float, dirs: Dict[str, str], enable_sqi: bool = ENABLE_SQI_FILTER) -> Dict[str, SubjectRRPack]:
    subject_data: Dict[str, SubjectRRPack] = {}
    mat = sio.loadmat(DATASET_CONFIGS["bidmc"]["data_path"])
    all_data = mat["data"][0]
    for idx in range(len(all_data)):
        subject = f"{idx + 1:02d}"
        cache_path = get_cache_path(dirs["cache"], "bidmc", subject, window_sec, enable_sqi=enable_sqi)
        cached = load_subject_cache(cache_path)
        if cached is not None:
            subject_data[subject] = cached
            print(f"Loaded BIDMC cache {subject}: {len(cached.x_low)} windows")
            continue

        try:
            subj_data = all_data[idx]
            ppg_raw = np.array(subj_data["ppg"][0, 0]["v"]).squeeze()
            target_raw = np.array(subj_data["ref"][0, 0]["resp_sig"][0, 0]["imp"][0, 0]["v"]).squeeze()
            pack = preprocess_subject_rr_windows(
                ppg_raw=ppg_raw,
                target_raw=target_raw,
                orig_fs=float(DATASET_CONFIGS["bidmc"]["source_fs"]),
                window_sec=window_sec,
                enable_sqi=enable_sqi,
                rr_band_bpm=DATASET_CONFIGS["bidmc"]["rr_band_bpm"],
            )
            if pack is not None:
                subject_data[subject] = pack
                save_subject_cache(cache_path, pack)
                print(f"Loaded BIDMC {subject}: {len(pack.x_low)} windows")
        except Exception as exc:
            print(f"   [Load Fail] BIDMC {subject}: {exc}")
    return subject_data


def load_capnobase_dataset(window_sec: float, dirs: Dict[str, str], enable_sqi: bool = ENABLE_SQI_FILTER) -> Dict[str, SubjectRRPack]:
    subject_data: Dict[str, SubjectRRPack] = {}
    cfg = DATASET_CONFIGS["capnobase"]
    csv_dir = cfg["data_path"]
    case_ids = capnobase_case_ids(csv_dir)
    for case_id in case_ids:
        cache_path = get_cache_path(dirs["cache"], "capnobase", case_id, window_sec, enable_sqi=enable_sqi)
        cached = load_subject_cache(cache_path)
        if cached is not None:
            subject_data[case_id] = cached
            print(f"Loaded CapnoBase cache {case_id}: {len(cached.x_low)} windows")
            continue

        try:
            signal_path = capnobase_file(csv_dir, case_id, "signal")
            param_path = capnobase_file(csv_dir, case_id, "param")
            ppg_raw, target_raw = read_capnobase_signals(signal_path)
            fields = read_capnobase_fields(param_path) if param_path else {}
            orig_fs = capnobase_sampling_rate(fields, "pleth", cfg["default_fs"])
            co2_fs = capnobase_sampling_rate(fields, "co2", orig_fs)
            if int(round(co2_fs)) != int(round(orig_fs)):
                target_raw = resample_signal_poly(target_raw, co2_fs, orig_fs)
            pack = preprocess_subject_rr_windows(
                ppg_raw=ppg_raw,
                target_raw=target_raw,
                orig_fs=orig_fs,
                window_sec=window_sec,
                enable_sqi=enable_sqi,
                rr_band_bpm=cfg["rr_band_bpm"],
            )
            if pack is not None:
                subject_data[case_id] = pack
                save_subject_cache(cache_path, pack)
                print(f"Loaded CapnoBase {case_id}: {len(pack.x_low)} windows")
        except Exception as exc:
            print(f"   [Load Fail] CapnoBase {case_id}: {exc}")
    return subject_data


def load_steam2_dataset(window_sec: float, dirs: Dict[str, str], enable_sqi: bool = ENABLE_SQI_FILTER) -> Dict[str, SubjectRRPack]:
    """Every piece of every session (sessions split at boundary events) is
    resampled to TARGET_FS and windowed on its own; the windows of a subject's
    pieces are stacked in session order."""
    subject_data: Dict[str, SubjectRRPack] = {}
    cfg = DATASET_CONFIGS["steam2"]
    if not os.path.isdir(cfg["ppg_dir"]) or not os.path.isdir(cfg["rsp_dir"]):
        raise FileNotFoundError(f"STEAM2 folders not found: {cfg['ppg_dir']} / {cfg['rsp_dir']}")
    for subject, items in steam2_session_files(cfg["ppg_dir"], cfg["rsp_dir"]).items():
        cache_path = get_cache_path(dirs["cache"], "steam2", subject, window_sec, enable_sqi=enable_sqi)
        cached = load_subject_cache(cache_path)
        if cached is not None:
            subject_data[subject] = cached
            print(f"Loaded STEAM2 cache {subject}: {len(cached.x_low)} windows")
            continue

        x_low_parts: List[np.ndarray] = []
        x_high_parts: List[np.ndarray] = []
        y_parts: List[np.ndarray] = []
        for name, ppg_path, rsp_path in items:
            try:
                ppg = read_eeglab_channel(ppg_path, "PPG", cfg["default_fs"])
                rsp = read_eeglab_channel(rsp_path, "RSP", cfg["default_fs"])
                pieces, length_difference = steam2_session_pieces(ppg, rsp)
                if abs(length_difference) > 1.0:
                    print(f"   [Note] STEAM2 {subject} {name}: PPG and RSP differ by {length_difference:.1f} s; "
                          "the common part is used")
                for ppg_piece, rsp_piece, fs_ppg, fs_rsp in pieces:
                    ppg_res = resample_signal_poly(ppg_piece, fs_ppg, TARGET_FS)
                    rsp_res = resample_signal_poly(rsp_piece, fs_rsp, TARGET_FS)
                    pack = preprocess_subject_rr_windows(
                        ppg_raw=ppg_res,
                        target_raw=rsp_res,
                        orig_fs=float(TARGET_FS),
                        window_sec=window_sec,
                        enable_sqi=enable_sqi,
                        rr_band_bpm=cfg["rr_band_bpm"],
                    )
                    if pack is not None:
                        x_low_parts.append(pack.x_low)
                        x_high_parts.append(pack.x_high)
                        y_parts.append(pack.y_rr)
            except Exception as exc:
                print(f"   [Skip] STEAM2 {subject} session {name}: {exc}")

        if x_low_parts:
            merged = SubjectRRPack(
                x_low=np.concatenate(x_low_parts, axis=0).astype(np.float32),
                x_high=np.concatenate(x_high_parts, axis=0).astype(np.float32),
                y_rr=np.concatenate(y_parts, axis=0).astype(np.float32),
                fs=TARGET_FS,
            )
            subject_data[subject] = merged
            save_subject_cache(cache_path, merged)
            print(f"Loaded STEAM2 {subject}: {len(merged.x_low)} windows from {len(items)} sessions")
        else:
            print(f"   [Load Fail] STEAM2 {subject}: no usable window")
    return subject_data


def load_dataset(
    dataset_key: str,
    window_sec: float,
    dirs: Dict[str, str],
    enable_sqi: bool = ENABLE_SQI_FILTER,
) -> Dict[str, SubjectRRPack]:
    if dataset_key == "steam2":
        return load_steam2_dataset(window_sec, dirs, enable_sqi=enable_sqi)
    if dataset_key == "bidmc":
        return load_bidmc_dataset(window_sec, dirs, enable_sqi=enable_sqi)
    if dataset_key == "capnobase":
        return load_capnobase_dataset(window_sec, dirs, enable_sqi=enable_sqi)
    raise KeyError(f"Unsupported dataset_key: {dataset_key}")


# ===========================================================
# 6. Dataset Objects / Folds
# ===========================================================
class RRScalarDataset(Dataset):
    def __init__(self, x_low: np.ndarray, x_high: np.ndarray, y: np.ndarray):
        self.x_low = torch.as_tensor(x_low, dtype=torch.float32)
        self.x_high = torch.as_tensor(x_high, dtype=torch.float32)
        self.y = torch.as_tensor(y, dtype=torch.float32)

    def __len__(self) -> int:
        return len(self.x_low)

    def __getitem__(self, idx: int):
        return self.x_low[idx].unsqueeze(0), self.x_high[idx], self.y[idx]


def concat_subject_arrays(subject_data: Dict[str, SubjectRRPack], subjects: Sequence[str]) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    x_low_all = [subject_data[subj].x_low for subj in subjects]
    x_high_all = [subject_data[subj].x_high for subj in subjects]
    y_all = [subject_data[subj].y_rr for subj in subjects]
    return (
        np.concatenate(x_low_all, axis=0).astype(np.float32),
        np.concatenate(x_high_all, axis=0).astype(np.float32),
        np.concatenate(y_all, axis=0).astype(np.float32),
    )


def make_loader(x_low: np.ndarray, x_high: np.ndarray, y: np.ndarray, batch_size: int, shuffle: bool) -> DataLoader:
    dataset = RRScalarDataset(x_low, x_high, y)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=NUM_WORKERS,
        pin_memory=PIN_MEMORY,
    )


def build_subject_folds(subjects: Sequence[str], n_splits: int = INNER_FOLDS, seed: int = SEED) -> List[List[str]]:
    subjects = list(subjects)
    rng = random.Random(seed)
    rng.shuffle(subjects)
    n_splits = max(1, min(int(n_splits), len(subjects)))
    folds = [[] for _ in range(n_splits)]
    for idx, subject in enumerate(subjects):
        folds[idx % n_splits].append(subject)
    return [fold for fold in folds if fold]


# ===========================================================
# 7. Paper-Style RRWaveNet
# ===========================================================
class Conv1dSame(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, stride: int = 1, bias: bool = False):
        super().__init__()
        self.kernel_size = int(kernel_size)
        self.stride = int(stride)
        self.conv = nn.Conv1d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=self.kernel_size,
            stride=self.stride,
            padding=0,
            bias=bias,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        in_len = int(x.shape[-1])
        out_len = math.ceil(in_len / self.stride)
        pad_needed = max(0, (out_len - 1) * self.stride + self.kernel_size - in_len)
        pad_left = pad_needed // 2
        pad_right = pad_needed - pad_left
        if pad_needed > 0:
            x = F.pad(x, (pad_left, pad_right))
        return self.conv(x)


class MultiScaleBranch(nn.Module):
    def __init__(self, in_channels: int, kernel_size: int, out_channels: int = 1):
        super().__init__()
        self.conv = Conv1dSame(in_channels, out_channels, kernel_size=kernel_size, stride=5, bias=False)
        self.bn = nn.BatchNorm1d(out_channels)
        self.act = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(self.bn(self.conv(x)))


class DenseResidualBlock(nn.Module):
    """
    A paper-faithful channel-growth interpretation:
    output_channels = input_channels + 2 * growth_channels
    """

    def __init__(self, in_channels: int, growth_channels: int, temporal_kernel_size: int = RESIDUAL_KERNEL_SIZE):
        super().__init__()
        self.expand = nn.Conv1d(in_channels, growth_channels, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm1d(growth_channels)
        self.temporal = Conv1dSame(growth_channels, growth_channels, kernel_size=temporal_kernel_size, stride=1, bias=False)
        self.bn2 = nn.BatchNorm1d(growth_channels)
        self.act = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y1 = self.act(self.bn1(self.expand(x)))
        y2 = self.act(self.bn2(self.temporal(y1)))
        return torch.cat([x, y1, y2], dim=1)


def init_module_weights(module: nn.Module) -> None:
    for submodule in module.modules():
        if isinstance(submodule, (nn.Conv1d, nn.Linear)):
            nn.init.kaiming_normal_(submodule.weight, nonlinearity="relu")
            if submodule.bias is not None:
                nn.init.zeros_(submodule.bias)
        elif isinstance(submodule, nn.BatchNorm1d):
            nn.init.ones_(submodule.weight)
            nn.init.zeros_(submodule.bias)


class RRWaveNetFeatureExtractor(nn.Module):
    def __init__(
        self,
        in_channels: int,
        branch_kernels: Sequence[int] = MULTISCALE_KERNELS,
        residual_features: Sequence[int] = RESIDUAL_FEATURES,
        residual_kernel_size: int = RESIDUAL_KERNEL_SIZE,
        feature_dropout: float = 0.0,
    ):
        super().__init__()
        self.branches = nn.ModuleList([MultiScaleBranch(in_channels, k) for k in branch_kernels])
        channels = len(branch_kernels)
        blocks = []
        for feat in residual_features:
            blocks.append(DenseResidualBlock(channels, feat, temporal_kernel_size=residual_kernel_size))
            channels = channels + 2 * feat
        self.blocks = nn.Sequential(*blocks)
        self.gap = nn.AdaptiveAvgPool1d(1)
        self.dropout = nn.Dropout(feature_dropout) if feature_dropout > 0.0 else nn.Identity()
        self.output_dim = int(channels)
        init_module_weights(self)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        multi_scale = torch.cat([branch(x) for branch in self.branches], dim=1)
        feat = self.blocks(multi_scale)
        pooled = self.gap(feat).squeeze(-1)
        return self.dropout(pooled)


class RegressionHead(nn.Module):
    def __init__(self, in_dim: int, hidden_dims: Sequence[int], dropout: float):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, int(hidden_dims[0])),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(int(hidden_dims[0]), int(hidden_dims[1])),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(int(hidden_dims[1]), 1),
        )
        init_module_weights(self)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)


class RRWaveNetLowOnly(nn.Module):
    def __init__(self, fc_hidden_dims: Sequence[int] = FC_HIDDEN_DIMS):
        super().__init__()
        self.backbone = RRWaveNetFeatureExtractor(
            in_channels=1,
            residual_features=RESIDUAL_FEATURES,
            feature_dropout=FEATURE_DROPOUT_LOW,
        )
        self.head = RegressionHead(self.backbone.output_dim, fc_hidden_dims, dropout=LOW_HEAD_DROPOUT)

    def forward(self, x_low: torch.Tensor, x_high: torch.Tensor | None = None) -> torch.Tensor:
        feat = self.backbone(x_low)
        return self.head(feat)


class RRWaveNetLowHilbert(nn.Module):
    def __init__(self, fc_hidden_dims: Sequence[int] = FC_HIDDEN_DIMS):
        super().__init__()
        self.low_backbone = RRWaveNetFeatureExtractor(
            in_channels=1,
            residual_features=RESIDUAL_FEATURES,
            feature_dropout=FEATURE_DROPOUT_LOW,
        )
        self.high_backbone = RRWaveNetFeatureExtractor(
            in_channels=3,
            residual_features=HILBERT_AUX_RESIDUAL_FEATURES,
            feature_dropout=FEATURE_DROPOUT_HIGH,
        )
        self.low_proj = nn.Sequential(
            nn.Linear(self.low_backbone.output_dim, FUSION_HIDDEN_DIM),
            nn.ReLU(inplace=True),
            nn.Dropout(HILBERT_HEAD_DROPOUT),
        )
        self.high_proj = nn.Sequential(
            nn.Linear(self.high_backbone.output_dim, FUSION_HIDDEN_DIM),
            nn.ReLU(inplace=True),
            nn.Dropout(HILBERT_HEAD_DROPOUT),
        )
        self.gate = nn.Sequential(
            nn.Linear(FUSION_HIDDEN_DIM * 2, FUSION_HIDDEN_DIM),
            nn.ReLU(inplace=True),
            nn.Linear(FUSION_HIDDEN_DIM, FUSION_HIDDEN_DIM),
            nn.Sigmoid(),
        )
        self.head = RegressionHead(FUSION_HIDDEN_DIM, fc_hidden_dims, dropout=HILBERT_HEAD_DROPOUT)
        init_module_weights(self.low_proj)
        init_module_weights(self.high_proj)
        init_module_weights(self.gate)

    def forward(self, x_low: torch.Tensor, x_high: torch.Tensor | None = None) -> torch.Tensor:
        if x_high is None:
            raise ValueError("x_high is required for RRWaveNetLowHilbert.")
        low_feat = self.low_backbone(x_low)
        high_feat = self.high_backbone(x_high)
        low_proj = self.low_proj(low_feat)
        high_proj = self.high_proj(high_feat)
        gate = self.gate(torch.cat([low_proj, high_proj], dim=1))
        fused = low_proj + gate * high_proj
        return self.head(fused)


def build_model(case_key: str) -> nn.Module:
    if case_key == "low_only":
        return RRWaveNetLowOnly()
    if case_key == "low_hilbert":
        return RRWaveNetLowHilbert()
    raise KeyError(f"Unsupported input case: {case_key}")


# ===========================================================
# 8. Optimizer
# ===========================================================
class AdaBelief(Optimizer):
    def __init__(
        self,
        params,
        lr: float = LEARNING_RATE,
        betas: Tuple[float, float] = ADABELIEF_BETAS,
        eps: float = ADABELIEF_EPS,
        weight_decay: float = WEIGHT_DECAY,
    ):
        if lr <= 0.0:
            raise ValueError("Invalid learning rate")
        if eps <= 0.0:
            raise ValueError("Invalid epsilon")
        if not 0.0 <= betas[0] < 1.0 or not 0.0 <= betas[1] < 1.0:
            raise ValueError("Invalid betas")
        defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay)
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            lr = group["lr"]
            beta1, beta2 = group["betas"]
            eps = group["eps"]
            weight_decay = group["weight_decay"]

            for p in group["params"]:
                if p.grad is None:
                    continue
                grad = p.grad
                if grad.is_sparse:
                    raise RuntimeError("AdaBelief does not support sparse gradients.")

                state = self.state[p]
                if len(state) == 0:
                    state["step"] = 0
                    state["exp_avg"] = torch.zeros_like(p)
                    state["exp_avg_var"] = torch.zeros_like(p)

                exp_avg = state["exp_avg"]
                exp_avg_var = state["exp_avg_var"]
                state["step"] += 1

                if weight_decay != 0.0:
                    grad = grad.add(p, alpha=weight_decay)

                exp_avg.mul_(beta1).add_(grad, alpha=1.0 - beta1)
                grad_residual = grad - exp_avg
                exp_avg_var.mul_(beta2).addcmul_(grad_residual, grad_residual, value=1.0 - beta2)

                bias_correction1 = 1.0 - beta1 ** state["step"]
                bias_correction2 = 1.0 - beta2 ** state["step"]

                step_size = lr / bias_correction1
                denom = (exp_avg_var / bias_correction2).sqrt().add_(eps)
                p.addcdiv_(exp_avg, denom, value=-step_size)

        return loss


# ===========================================================
# 9. Train / Eval
# ===========================================================
def evaluate_model(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    scaler: RRTargetScaler,
) -> Tuple[float, float, np.ndarray, np.ndarray]:
    model.eval()
    total_loss = 0.0
    preds_scaled = []
    trues_scaled = []
    with torch.no_grad():
        for x_low, x_high, y in loader:
            x_low = x_low.to(device, non_blocking=True)
            x_high = x_high.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            with autocast_context():
                pred = model(x_low, x_high)
                loss = criterion(pred, y)
            total_loss += float(loss.item())
            preds_scaled.append(pred.float().cpu().numpy())
            trues_scaled.append(y.float().cpu().numpy())
    total_loss /= max(len(loader), 1)
    preds_scaled_np = np.concatenate(preds_scaled, axis=0) if preds_scaled else np.array([], dtype=np.float32)
    trues_scaled_np = np.concatenate(trues_scaled, axis=0) if trues_scaled else np.array([], dtype=np.float32)
    preds_np = scaler.inverse_transform(preds_scaled_np)
    trues_np = scaler.inverse_transform(trues_scaled_np)
    mae = float(np.mean(np.abs(preds_np - trues_np))) if preds_np.size else np.nan
    return total_loss, mae, preds_np, trues_np


def save_learning_curve(train_losses: List[float], val_losses: List[float], save_path: str, title: str) -> None:
    plt.figure(figsize=(10, 5))
    plt.plot(np.arange(1, len(train_losses) + 1), train_losses, label="Train Loss")
    plt.plot(np.arange(1, len(val_losses) + 1), val_losses, label="Val Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title(title)
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path, dpi=180)
    plt.close()


def train_one_fold(
    dataset_key: str,
    window_sec: float,
    case_key: str,
    test_subject: str,
    fold_idx: int,
    subject_data: Dict[str, SubjectRRPack],
    train_subjects: Sequence[str],
    val_subjects: Sequence[str],
    dirs: Dict[str, str],
    max_epochs: int = MAX_EPOCHS,
    batch_size: int = BATCH_SIZE,
) -> Dict[str, object]:
    x_low_train, x_high_train, y_train_raw = concat_subject_arrays(subject_data, train_subjects)
    x_low_val, x_high_val, y_val_raw = concat_subject_arrays(subject_data, val_subjects)
    x_low_test, x_high_test, y_test_raw = concat_subject_arrays(subject_data, [test_subject])

    target_scaler = RRTargetScaler.fit(y_train_raw)
    y_train = target_scaler.transform(y_train_raw)
    y_val = target_scaler.transform(y_val_raw)
    y_test = target_scaler.transform(y_test_raw)

    train_loader = make_loader(x_low_train, x_high_train, y_train, batch_size=batch_size, shuffle=True)
    val_loader = make_loader(x_low_val, x_high_val, y_val, batch_size=batch_size, shuffle=False)
    test_loader = make_loader(x_low_test, x_high_test, y_test, batch_size=batch_size, shuffle=False)

    model = build_model(case_key).to(device)
    trainable_params = count_trainable_params(model)
    criterion = nn.SmoothL1Loss(beta=LOSS_BETA)
    optimizer = AdaBelief(
        model.parameters(),
        lr=LEARNING_RATE,
        betas=ADABELIEF_BETAS,
        eps=ADABELIEF_EPS,
        weight_decay=WEIGHT_DECAY,
    )
    scheduler = ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=SCHEDULER_FACTOR,
        patience=SCHEDULER_PATIENCE,
        min_lr=MIN_LR,
    )

    best_val_loss = np.inf
    best_val_mae = np.inf
    best_epoch = 0
    patience_counter = 0
    train_losses: List[float] = []
    val_losses: List[float] = []
    model_path = os.path.join(
        dirs["models"],
        f"{safe_tag(dataset_key)}_{safe_tag(case_key)}_W{int(window_sec)}_{safe_tag(test_subject)}_fold{fold_idx + 1}_best.pth",
    )

    for epoch in range(int(max_epochs)):
        model.train()
        running_loss = 0.0
        n_batches = 0
        for x_low, x_high, y in train_loader:
            x_low = x_low.to(device, non_blocking=True)
            x_high = x_high.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with autocast_context():
                pred = model(x_low, x_high)
                loss = criterion(pred, y)
            loss.backward()
            if GRAD_CLIP_NORM is not None:
                torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP_NORM)
            optimizer.step()
            running_loss += float(loss.item())
            n_batches += 1

        train_loss = running_loss / max(n_batches, 1)
        val_loss, val_mae, _, _ = evaluate_model(model, val_loader, criterion, target_scaler)
        scheduler.step(val_mae)
        train_losses.append(train_loss)
        val_losses.append(val_loss)

        improved = val_mae < (best_val_mae - 1.0e-4)
        if improved:
            best_val_loss = val_loss
            best_val_mae = val_mae
            best_epoch = epoch + 1
            patience_counter = 0
            torch.save(
                {
                    "model_state": model.state_dict(),
                    "target_mean": target_scaler.mean,
                    "target_std": target_scaler.std,
                    "best_val_mae": best_val_mae,
                },
                model_path,
            )
        else:
            patience_counter += 1

        print(
            f"[{DATASET_CONFIGS[dataset_key]['label']}] W{int(window_sec)} | "
            f"{INPUT_CASES[case_key]['label']} | Test {test_subject} | Fold {fold_idx + 1}/{INNER_FOLDS} | "
            f"Ep {epoch + 1:05d} | TrainLoss {train_loss:.5f} | ValLoss {val_loss:.5f} | "
            f"ValMAE {val_mae:.5f} | LR {optimizer.param_groups[0]['lr']:.6f}"
        )

        if patience_counter >= EARLY_STOP_PATIENCE:
            print(f"Early stop at epoch {epoch + 1} for {test_subject} fold {fold_idx + 1}")
            break

    if os.path.exists(model_path):
        checkpoint = torch.load(model_path, map_location=device)
        if isinstance(checkpoint, dict) and "model_state" in checkpoint:
            model.load_state_dict(checkpoint["model_state"])
        else:
            model.load_state_dict(checkpoint)

    test_loss, test_mae, preds, trues = evaluate_model(model, test_loader, criterion, target_scaler)
    curve_path = os.path.join(
        dirs["curves"],
        f"{safe_tag(dataset_key)}_{safe_tag(case_key)}_W{int(window_sec)}_{safe_tag(test_subject)}_fold{fold_idx + 1}.png",
    )
    save_learning_curve(
        train_losses=train_losses,
        val_losses=val_losses,
        save_path=curve_path,
        title=(
            f"{DATASET_CONFIGS[dataset_key]['label']} | {INPUT_CASES[case_key]['label']} | "
            f"W{int(window_sec)} | {test_subject} | Fold {fold_idx + 1}"
        ),
    )

    return {
        "model_path": model_path,
        "curve_path": curve_path,
        "best_epoch": best_epoch,
        "best_val_loss": float(best_val_loss),
        "best_val_mae": float(best_val_mae),
        "test_loss": float(test_loss),
        "test_mae": float(test_mae),
        "preds": preds.astype(np.float32),
        "trues": trues.astype(np.float32),
        "num_train_windows": int(len(x_low_train)),
        "num_val_windows": int(len(x_low_val)),
        "num_test_windows": int(len(x_low_test)),
        "trainable_params": int(trainable_params),
        "target_mean": float(target_scaler.mean),
        "target_std": float(target_scaler.std),
    }


# ===========================================================
# 10. Reporting / Runner
# ===========================================================
def save_dataframe_bundle(metrics_dir: str, prefix: str, frames: Dict[str, pd.DataFrame]) -> Dict[str, str]:
    os.makedirs(metrics_dir, exist_ok=True)
    saved_paths: Dict[str, str] = {}
    for name, frame in frames.items():
        path = os.path.join(metrics_dir, f"{prefix}_{name}.csv")
        frame.to_csv(path, index=False, encoding="utf-8-sig")
        saved_paths[name] = path
    return saved_paths


def summarize_subject_window_results(per_window_df: pd.DataFrame) -> pd.DataFrame:
    if per_window_df.empty:
        return pd.DataFrame()
    return (
        per_window_df.groupby(["Dataset", "Window_Sec", "Input_Case", "Input_Label", "Test_Subject"], as_index=False)
        .agg(
            Fold_MAE_Mean=("Abs_Error_BPM", "mean"),
            Fold_MAE_Std=("Abs_Error_BPM", "std"),
            Num_Windows=("Window_Index", "count"),
        )
        .sort_values(["Dataset", "Window_Sec", "Test_Subject"])
    )


def summarize_dataset_results(per_window_df: pd.DataFrame) -> pd.DataFrame:
    if per_window_df.empty:
        return pd.DataFrame()

    per_subject = summarize_subject_window_results(per_window_df)
    window_level = (
        per_window_df.groupby(["Dataset", "Window_Sec", "Input_Case", "Input_Label"], as_index=False)
        .agg(
            Window_MAE_Global=("Abs_Error_BPM", "mean"),
            Window_MAE_Std=("Abs_Error_BPM", "std"),
            Num_Eval_Windows=("Window_Index", "count"),
            Num_Subjects=("Test_Subject", "nunique"),
        )
    )
    subject_level = (
        per_subject.groupby(["Dataset", "Window_Sec", "Input_Case", "Input_Label"], as_index=False)
        .agg(
            Subject_MAE_Mean=("Fold_MAE_Mean", "mean"),
            Subject_MAE_Std=("Fold_MAE_Mean", "std"),
        )
    )
    summary = window_level.merge(subject_level, on=["Dataset", "Window_Sec", "Input_Case", "Input_Label"], how="left")
    return summary.sort_values(["Dataset", "Window_Sec", "Input_Case"])


def run_dataset_experiment(
    dataset_key: str,
    window_sec: float,
    output_root: str,
    max_epochs: int = MAX_EPOCHS,
    batch_size: int = BATCH_SIZE,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    dirs = prepare_output_dirs(output_root, dataset_key, window_sec)
    print(f"\n{'=' * 88}")
    print(f"Dataset: {DATASET_CONFIGS[dataset_key]['label']} | Window: {int(window_sec)} s")
    print(f"Output root: {dirs['root']}")
    print(f"{'=' * 88}")

    subject_data = load_dataset(dataset_key, window_sec, dirs, enable_sqi=True)
    subjects = sorted(subject_data.keys())
    if len(subjects) < 3:
        print(
            f"[Warn] {DATASET_CONFIGS[dataset_key]['label']} W{int(window_sec)}: "
            f"only {len(subjects)} valid subjects with SQI enabled. Retrying with SQI disabled."
        )
        subject_data = load_dataset(dataset_key, window_sec, dirs, enable_sqi=False)
        subjects = sorted(subject_data.keys())
    if len(subjects) < 3:
        print(
            f"[Skip] {DATASET_CONFIGS[dataset_key]['label']} W{int(window_sec)}: "
            f"still only {len(subjects)} valid subjects after SQI fallback."
        )
        empty = pd.DataFrame()
        return empty, empty, empty

    fold_rows = []
    per_window_rows = []
    start_all = time.time()

    for test_subject in subjects:
        remaining_subjects = [subj for subj in subjects if subj != test_subject]
        folds = build_subject_folds(remaining_subjects, n_splits=INNER_FOLDS, seed=SEED)
        print(f"\n===== Test Subject: {test_subject} | Num inner folds: {len(folds)} =====")

        for fold_idx, val_subjects in enumerate(folds):
            train_subjects = [subj for subj in remaining_subjects if subj not in val_subjects]
            if not train_subjects or not val_subjects:
                continue
            for case_key in ACTIVE_INPUT_CASES:
                fold_start = time.time()
                result = train_one_fold(
                    dataset_key=dataset_key,
                    window_sec=window_sec,
                    case_key=case_key,
                    test_subject=test_subject,
                    fold_idx=fold_idx,
                    subject_data=subject_data,
                    train_subjects=train_subjects,
                    val_subjects=val_subjects,
                    dirs=dirs,
                    max_epochs=max_epochs,
                    batch_size=batch_size,
                )

                elapsed = time.time() - fold_start
                fold_rows.append(
                    {
                        "Dataset": DATASET_CONFIGS[dataset_key]["label"],
                        "Dataset_Key": dataset_key,
                        "Window_Sec": int(window_sec),
                        "Input_Case": case_key,
                        "Input_Label": INPUT_CASES[case_key]["label"],
                        "Test_Subject": test_subject,
                        "Fold_Index": fold_idx + 1,
                        "Train_Subjects": ",".join(train_subjects),
                        "Val_Subjects": ",".join(val_subjects),
                        "Num_Train_Windows": result["num_train_windows"],
                        "Num_Val_Windows": result["num_val_windows"],
                        "Num_Test_Windows": result["num_test_windows"],
                        "Best_Epoch": result["best_epoch"],
                        "Best_Val_Loss": result["best_val_loss"],
                        "Best_Val_MAE": result["best_val_mae"],
                        "Test_Loss": result["test_loss"],
                        "Test_MAE": result["test_mae"],
                        "Trainable_Params": result["trainable_params"],
                        "Target_Mean_BPM": result["target_mean"],
                        "Target_Std_BPM": result["target_std"],
                        "Curve_Path": result["curve_path"],
                        "Model_Path": result["model_path"],
                        "Elapsed_Sec": elapsed,
                    }
                )

                preds = np.asarray(result["preds"], dtype=np.float32)
                trues = np.asarray(result["trues"], dtype=np.float32)
                abs_err = np.abs(preds - trues)
                for win_idx, (pred_rr, true_rr, err) in enumerate(zip(preds, trues, abs_err), start=1):
                    per_window_rows.append(
                        {
                            "Dataset": DATASET_CONFIGS[dataset_key]["label"],
                            "Dataset_Key": dataset_key,
                            "Window_Sec": int(window_sec),
                            "Input_Case": case_key,
                            "Input_Label": INPUT_CASES[case_key]["label"],
                            "Test_Subject": test_subject,
                            "Fold_Index": fold_idx + 1,
                            "Window_Index": win_idx,
                            "Pred_RR_BPM": float(pred_rr),
                            "True_RR_BPM": float(true_rr),
                            "Abs_Error_BPM": float(err),
                        }
                    )

                running_fold_df = pd.DataFrame(fold_rows)
                running_window_df = pd.DataFrame(per_window_rows)
                running_subject_df = summarize_subject_window_results(running_window_df)
                save_dataframe_bundle(
                    dirs["metrics"],
                    "RRWaveNet_PaperStyle_Running",
                    {
                        "PerFold": running_fold_df,
                        "PerWindow": running_window_df,
                        "PerSubject": running_subject_df,
                    },
                )

    total_elapsed = time.time() - start_all
    fold_df = pd.DataFrame(fold_rows)
    per_window_df = pd.DataFrame(per_window_rows)
    per_subject_df = summarize_subject_window_results(per_window_df)
    summary_df = summarize_dataset_results(per_window_df)
    if not summary_df.empty:
        summary_df["Total_Elapsed_Sec"] = total_elapsed

    save_dataframe_bundle(
        dirs["metrics"],
        "RRWaveNet_PaperStyle_Final",
        {
            "PerFold": fold_df,
            "PerWindow": per_window_df,
            "PerSubject": per_subject_df,
            "Summary": summary_df,
        },
    )

    return summary_df, per_subject_df, fold_df


# ===========================================================
# 11. CLI / Main (261107: BIDMC and CapnoBase W16 first, in the results
#     folder of the running 261106 run)
# ===========================================================
DEFAULT_JOBS = ("bidmc:16", "capnobase:16")
MERGED_RESULTS_ROOT = (
    r"D:\PPG2RSP_RRWaveNet_Inspired\261106_DeepCorrencoder_PaperStyle_All\RRWaveNet_PaperStyle_RR"
)
WORKSPACE_DRIVE_ROOT = r"D:\PPG2RSP_RRWaveNet_Inspired"
FINAL_PREFIX = "RRWaveNet_PaperStyle_Final"
RUNNING_PREFIX = "RRWaveNet_PaperStyle_Running"
BUSY_MINUTES = 30.0


def candidate_results_roots() -> List[str]:
    roots = [
        MERGED_RESULTS_ROOT,
        os.path.join(WORKSPACE_ROOT, "261106_RRWaveNet_PaperStyle_RR_Results"),
        os.path.join(WORKSPACE_DRIVE_ROOT, "261106_RRWaveNet_PaperStyle_RR", "261106_RRWaveNet_PaperStyle_RR_Results"),
        os.path.join(WORKSPACE_DRIVE_ROOT, "261106_RRWaveNet_PaperStyle_RR_Results"),
    ]
    return list(dict.fromkeys(os.path.abspath(root) for root in roots))


def latest_write(folder: str) -> float:
    """Newest modification time of the metrics files of every window folder
    (0 when there are none)."""
    newest = 0.0
    if not os.path.isdir(folder):
        return newest
    for window in os.scandir(folder):
        metrics = os.path.join(window.path, "metrics")
        if window.is_dir() and os.path.isdir(metrics):
            for entry in os.scandir(metrics):
                newest = max(newest, entry.stat().st_mtime)
    return newest


def find_results_root() -> Optional[str]:
    """The candidate folder holding a STEAM2 folder (the most recently
    written one when several do)."""
    found = []
    for root in candidate_results_roots():
        steam2 = os.path.join(root, DATASET_CONFIGS["steam2"]["save_name"])
        if os.path.isdir(steam2):
            found.append((latest_write(steam2), root))
    return max(found)[1] if found else None


def job_metrics_dir(output_root: str, dataset_key: str, window_sec: float) -> str:
    return os.path.join(output_root, DATASET_CONFIGS[dataset_key]["save_name"], f"W{int(window_sec)}", "metrics")


def job_state(output_root: str, dataset_key: str, window_sec: float) -> str:
    """'finished', 'busy' (running results written in the last BUSY_MINUTES,
    no final results), 'partial' or 'new'."""
    metrics = job_metrics_dir(output_root, dataset_key, window_sec)
    if os.path.exists(os.path.join(metrics, f"{FINAL_PREFIX}_Summary.csv")):
        return "finished"
    running = os.path.join(metrics, f"{RUNNING_PREFIX}_PerFold.csv")
    if os.path.exists(running):
        age_min = (time.time() - os.path.getmtime(running)) / 60.0
        return "busy" if age_min < BUSY_MINUTES else "partial"
    return "new"


def parse_job(text: str) -> Tuple[str, float]:
    try:
        dataset_key, window = text.split(":")
        dataset_key = dataset_key.strip().lower()
        window_sec = float(window)
    except ValueError:
        raise argparse.ArgumentTypeError(f"job '{text}': use dataset:window, e.g. bidmc:16")
    if dataset_key not in DATASET_CONFIGS:
        raise argparse.ArgumentTypeError(f"job '{text}': dataset one of {sorted(DATASET_CONFIGS)}")
    return dataset_key, window_sec


def write_global_tables(output_root: str) -> None:
    """The final tables of every finished dataset / window in the results
    folder, written as RRWaveNet_PaperStyle_Global*_AllFinished.csv."""
    frames = {"Summary": [], "PerSubject": [], "PerFold": []}
    for dataset_key in DATASET_CONFIGS:
        folder = os.path.join(output_root, DATASET_CONFIGS[dataset_key]["save_name"])
        if not os.path.isdir(folder):
            continue
        for window in sorted(os.scandir(folder), key=lambda entry: entry.name):
            metrics = os.path.join(window.path, "metrics")
            if not (window.is_dir() and os.path.exists(os.path.join(metrics, f"{FINAL_PREFIX}_Summary.csv"))):
                continue
            for name in frames:
                path = os.path.join(metrics, f"{FINAL_PREFIX}_{name}.csv")
                if os.path.exists(path) and os.path.getsize(path) > 0:
                    try:
                        frame = pd.read_csv(path)
                    except pd.errors.EmptyDataError:
                        continue
                    if not frame.empty:
                        frames[name].append(frame)
    for name, parts in frames.items():
        if parts:
            pd.concat(parts, ignore_index=True).to_csv(
                os.path.join(output_root, f"RRWaveNet_PaperStyle_Global{name}_AllFinished.csv"),
                index=False,
                encoding="utf-8-sig",
            )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Paper-style RRWaveNet RR: BIDMC and CapnoBase W16 first, in the results folder of the 261106 run"
    )
    parser.add_argument("--jobs", nargs="+", type=parse_job, default=[parse_job(job) for job in DEFAULT_JOBS],
                        help="dataset:window in the order to run, e.g. bidmc:16 capnobase:16 (default)")
    parser.add_argument("--output-root", type=str, default=None,
                        help="results folder of the 261106 run (default: found automatically)")
    parser.add_argument("--redo", action="store_true",
                        help="also run a job whose final results exist or that another process is training")
    parser.add_argument("--max-epochs", type=int, default=MAX_EPOCHS)
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_root = args.output_root or find_results_root()
    if output_root is None:
        print("The results folder of the 261106 run was not found in:")
        for root in candidate_results_roots():
            print(f"  {root}")
        print("Give it with --output-root <folder> (the folder that holds STEAM2\\W16 ...).")
        sys.exit(1)
    output_root = os.path.abspath(output_root)
    print(f"Device: {device}")
    print(f"AMP enabled: {AMP_ENABLED}")
    print(f"Input cases: {list(ACTIVE_INPUT_CASES)}")
    print(f"Results folder: {output_root}" + ("" if args.output_root else " (found automatically)"))
    print("Jobs: " + ", ".join(f"{DATASET_CONFIGS[d]['label']} W{int(w)}" for d, w in args.jobs))

    status = []
    for dataset_key, window_sec in args.jobs:
        label = f"{DATASET_CONFIGS[dataset_key]['label']} W{int(window_sec)}"
        state = job_state(output_root, dataset_key, window_sec)
        if state == "finished" and not args.redo:
            print(f"\n[Skip] {label}: final results exist (--redo runs it again)")
            status.append(f"{label}: skipped, already finished")
            continue
        if state == "busy" and not args.redo:
            print(f"\n[Skip] {label}: running results written in the last {BUSY_MINUTES:g} min and no final "
                  "results: another process seems to train it (--redo runs it anyway)")
            status.append(f"{label}: skipped, being trained by another process")
            continue
        if state == "partial":
            print(f"\n[Note] {label}: an earlier run stopped before the end; it is trained from the start")
        set_seed(SEED, deterministic=False)
        started = time.time()
        summary_df, _, _ = run_dataset_experiment(
            dataset_key=dataset_key,
            window_sec=float(window_sec),
            output_root=output_root,
            max_epochs=args.max_epochs,
            batch_size=args.batch_size,
        )
        status.append(f"{label}: {'finished' if not summary_df.empty else 'no result (too few subjects)'} "
                      f"({(time.time() - started) / 3600:.2f} h)")

    write_global_tables(output_root)
    print(f"\n{'=' * 88}")
    for line in status:
        print(line)
    print(f"Global tables of every finished dataset / window: {output_root}\\RRWaveNet_PaperStyle_Global*_AllFinished.csv")
    print("Reminder: stop the 261106 run when its log reaches 'Dataset: BIDMC | Window: 16', or it trains BIDMC and "
          "CapnoBase W16 again and overwrites these results.")


if __name__ == "__main__":
    main()
