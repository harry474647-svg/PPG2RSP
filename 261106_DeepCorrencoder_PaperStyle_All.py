# -*- coding: utf-8 -*-
r"""
261106_DeepCorrencoder_PaperStyle_All: 261106_Deep_Correncoder and
261106_RRWaveNet_PaperStyle_RR in one file, run one after the other by one
command; each keeps its own results folder.

Parts (each code unchanged; in part B the top-level names it shares with
part A carry a ps_ / PS_ prefix so that the two codes keep their own
settings and functions):
  A. Deep corr-encoder (from 261106_Deep_Correncoder): CorrEncoderNet of the
     260504 ablation, only the 0.1-0.5 Hz PPG input (model 01_low), STEAM2
     (EEGLAB sessions, split at boundary events), 10 s windows, LOSO, 30 s
     evaluation.
  B. Paper-style RRWaveNet RR regression (from 261106_RRWaveNet_PaperStyle_RR):
     only the 0.1-0.5 Hz PPG input (low_only), STEAM2, BIDMC and CapnoBase,
     windows of 16, 32 and 64 s with a 2 s stride, RR labels from the Welch
     peak of the reference at 0.1 breaths/min resolution, LOSO with inner
     subject-level 5-fold CV.
The run order is A, then B. Each part is seeded as when its file runs on its
own. When one part fails, the error is printed and the other part still runs.

Outputs (OUTPUT_ROOT = D:\PPG2RSP_RRWaveNet_Inspired\261106_DeepCorrencoder_PaperStyle_All)
  Deep_Correncoder\STEAM2\        Models, Learning_Curves, Waveform_Comparison_30sec,
                                  Metrics, Processed_Subject_Cache
  Deep_Correncoder\Combined_Metrics\
  RRWaveNet_PaperStyle_RR\<Dataset>\W<16|32|64>\   cache, models, metrics, curves
  RRWaveNet_PaperStyle_RR\RRWaveNet_PaperStyle_Global*.csv

Run:
  python 261106_DeepCorrencoder_PaperStyle_All.py
Options:
  --parts deep_correncoder paperstyle   which parts (default both)
  --output-root <folder>
  --ce-datasets steam2 bidmc capnobase  part A datasets (default steam2)
  --ps-datasets steam2 bidmc capnobase  part B datasets (default all three)
  --window-sizes 16 32 64               part B windows (s)
  --max-epochs, --batch-size            part B
"""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import os
import re
import time
import random
import numpy as np
import pandas as pd
import scipy.io as sio
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
import argparse
import csv
import math
import traceback
from collections import Counter
from contextlib import nullcontext
from itertools import combinations
from math import gcd
from pathlib import Path
from scipy.signal import butter, sosfiltfilt, correlate, find_peaks, hilbert, resample_poly, resample, savgol_filter, welch
from scipy.stats import pearsonr, skew
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader, TensorDataset, WeightedRandomSampler, Dataset
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple
from torch.optim import Optimizer

# ===========================================================================
# PART A. Deep corr-encoder (261106_Deep_Correncoder)
# ===========================================================================
"""
261106_Deep_Correncoder: Corr-Encoder all-cases ablation
(260504_AutoEncoder_Opendata_ablation) on STEAM2.

This script keeps the CorrEncoderNet model family from the user's
AutoEncoder_AllCases experiment and changes the data pipeline to open datasets:

- BIDMC: PPG -> impedance respiration
- CapnoBase: plethysmography -> CO2-derived respiration
- STEAM2: PPG -> respiratory belt (RSP)   [added in 261106]

Only STEAM2 is run (ACTIVE_DATASETS), and only the 0.1-0.5 Hz PPG input
("low", model 01_low; ACTIVE_FEATURE_SETS): the high-band and Hilbert inputs
are not trained. Everything else (preprocessing, quality check, model, loss,
training, validation subjects, 30 s evaluation, outputs) is that of
260504_AutoEncoder_Opendata_ablation.

STEAM2 reading (the reader of 261028_Deep_correncoder / 261105): every
session (one 2-back block) is an EEGLAB .mat file (EEG.data, EEG.srate =
200 Hz, one channel labelled PPG or RSP), PPG/<subject>/<subject>_Session<k>.mat
and RSP/<subject>/<subject>_Session<k>.mat with the same name. PPG and RSP
files are paired by name only; EEGLAB "boundary" events (removed data) split
a session into pieces. Every piece is resampled from 200 Hz to 125 Hz and
processed and windowed on its own (no window spans two sessions or a
boundary); all sessions of a person form one subject, so a person is never
in the training and the test data of one LOSO fold.

260504 trains every non-empty combination of:
- low:  0.1-0.5 Hz PPG time series
- high: 0.5-5.0 Hz PPG time series
- env:  Hilbert envelope of high
- freq: Hilbert instantaneous frequency of high
(15 feature-combination models per LOSO test subject per dataset); here only
"low" (1 model per LOSO test subject).

Saved outputs:
1. Learning curves
2. Best model checkpoints
3. Per-subject, per-segment, summary CSV files
4. 30-second comparison plots
"""



matplotlib.use("Agg")



# ===========================================================
# 1. Environment and global config
# ===========================================================
GPU_NUM = 0
device = torch.device(f"cuda:{GPU_NUM}" if torch.cuda.is_available() else "cpu")
if device.type == "cuda":
    torch.cuda.set_device(device)
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = False
    if hasattr(torch, "set_float32_matmul_precision"):
        torch.set_float32_matmul_precision("high")


BIDMC_DATA_PATH = (
    r"D:\STEAM\Multimodal_prediction\PPG-RSP\Preprocessed"
    r"\bidmc-ppg-and-respiration-dataset-1.0.0"
    r"\bidmc-ppg-and-respiration-dataset-1.0.0\bidmc_data.mat"
)
CAPNOBASE_CSV_DIR = r"D:\STEAM\Multimodal_prediction\PPG-RSP\Opendata\Capnobase\data\csv"
STEAM2_PPG_DIR = r"D:\STEAM\Multimodal_prediction\PPG-RSP\Preprocessed\PPG\2back_raw_v2"
STEAM2_RSP_DIR = r"D:\STEAM\Multimodal_prediction\PPG-RSP\Preprocessed\RSP\2back_raw_v2"
# Own results folder, so the BIDMC / CapnoBase results of 260504
# (CorrEncoder_OpenDataset_v1, Combined_Metrics) are not overwritten.
SAVE_ROOT = r"D:\PPG2RSP_RRWaveNet_Inspired\261106_Deep_Correncoder"

ACTIVE_DATASETS = ["steam2"]

DATASET_CONFIGS = {
    "bidmc": {
        "label": "BIDMC",
        "save_name": "BIDMC",
        "data_path": BIDMC_DATA_PATH,
        "target_fs": 125,
        "target_column_label": "Impedance Respiration",
    },
    "capnobase": {
        "label": "CapnoBase",
        "save_name": "CapnoBase",
        "data_path": CAPNOBASE_CSV_DIR,
        "target_fs": 125,
        "target_column_label": "CO2-derived Respiration",
    },
    "steam2": {
        "label": "STEAM2",
        "save_name": "STEAM2",
        "ppg_dir": STEAM2_PPG_DIR,
        "rsp_dir": STEAM2_RSP_DIR,
        # EEGLAB sessions (EEG.srate = 200 Hz); default_fs is used only when a
        # file has no srate.
        "default_fs": 200,
        "target_fs": 125,
        "target_column_label": "Respiratory Belt (RSP)",
    },
}

FS_METRIC = 125
WIN_SEC = 10.0
STRIDE_SEC = 10.0
SEGMENT_WINDOWS = 3
SEGMENT_SEC = WIN_SEC * SEGMENT_WINDOWS

EPOCHS = 200
BATCH_SIZE = 32
PATIENCE = 40
MIN_EPOCHS = 40
MIN_DELTA = 1e-4
LR = 2e-4
MIN_LR = 1e-6
WEIGHT_DECAY = 5e-4
INPUT_NOISE_STD = 0.01
INPUT_CLIP_VALUE = 5.0
MAX_GRAD_NORM = 1.0

FILTERS = [16, 32, 64]
K_SIZES = [31, 15, 7]
STEM_DROPOUT = 0.10
BLOCK_DROPOUT = 0.25
FUSION_DROPOUT = 0.20

VAL_RATIO = 0.12
MIN_VAL_SUBJECTS = 3
BALANCE_TRAIN_SUBJECTS = True
SCHEDULER_PATIENCE = 8
VAL_EMA_ALPHA = 0.25
EARLY_STOP_MIN_DELTA = 1e-4
EARLY_STOP_PLATEAU_TOL = 5e-3

LOSS_POINT_WEIGHT = 0.30
LOSS_CORR_WEIGHT = 0.55
LOSS_DERIV_WEIGHT = 0.15

USE_AMP = True
AMP_ENABLED = USE_AMP and device.type == "cuda"
USE_DATA_CACHE = True
QUALITY_MIN_CORR = 0.03
MIN_VALID_SUBJECTS = 4

NUM_WORKERS = 0 if os.name == "nt" else min(4, os.cpu_count() or 1)
PIN_MEMORY = device.type == "cuda"
PREFETCH_FACTOR = 2

FEATURE_ORDER = ["low", "high", "env", "freq"]
FEATURE_LABELS = {
    "low": "Low",
    "high": "High",
    "env": "Env",
    "freq": "Freq",
}
PLOT_COLORS = [
    "#d62728",
    "#1f77b4",
    "#2ca02c",
    "#9467bd",
    "#ff7f0e",
    "#17becf",
    "#8c564b",
    "#e377c2",
    "#7f7f7f",
    "#bcbd22",
    "#393b79",
    "#637939",
    "#8c6d31",
    "#843c39",
    "#3182bd",
]


def set_seed(seed=42):
    print(f"Seed: {seed}")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(seed)


set_seed()


def build_model_configs():
    configs = []
    idx = 1
    for n_features in range(1, len(FEATURE_ORDER) + 1):
        for feature_keys in combinations(FEATURE_ORDER, n_features):
            feature_keys = list(feature_keys)
            key_name = "_".join(feature_keys)
            label = " + ".join(FEATURE_LABELS[key] for key in feature_keys)
            configs.append(
                {
                    "name": f"{idx:02d}_{key_name}",
                    "label": label,
                    "feature_keys": feature_keys,
                    "color": PLOT_COLORS[idx - 1],
                }
            )
            idx += 1
    return configs


# 261106: only the 0.1-0.5 Hz PPG input ("low"). Add feature sets here to
# train other combinations of the 260504 ablation again.
ACTIVE_FEATURE_SETS = (("low",),)
MODEL_CONFIGS = [cfg for cfg in build_model_configs() if tuple(cfg["feature_keys"]) in ACTIVE_FEATURE_SETS]


# ===========================================================
# 2. Utility functions
# ===========================================================
def safe_tag(text):
    return "".join(c if c.isalnum() or c in "-_." else "_" for c in str(text))


def prepare_output_dirs(dataset_key):
    dataset_cfg = DATASET_CONFIGS[dataset_key]
    root = os.path.join(SAVE_ROOT, dataset_cfg["save_name"])
    dirs = {
        "root": root,
        "models": os.path.join(root, "Models"),
        "curves": os.path.join(root, "Learning_Curves"),
        "plots": os.path.join(root, "Waveform_Comparison_30sec"),
        "metrics": os.path.join(root, "Metrics"),
        "cache": os.path.join(root, "Processed_Subject_Cache"),
    }
    for path in dirs.values():
        os.makedirs(path, exist_ok=True)
    return dirs


def autocast_context():
    if not AMP_ENABLED:
        return nullcontext()
    try:
        return torch.amp.autocast(device_type="cuda", enabled=True)
    except Exception:
        return torch.cuda.amp.autocast(enabled=True)


class DummyGradScaler:
    def scale(self, loss):
        return loss

    def unscale_(self, optimizer):
        return None

    def step(self, optimizer):
        optimizer.step()

    def update(self):
        return None


def build_grad_scaler():
    if not AMP_ENABLED:
        return DummyGradScaler()
    try:
        return torch.amp.GradScaler("cuda", enabled=True)
    except Exception:
        return torch.cuda.amp.GradScaler(enabled=True)


def count_trainable_params(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def sanitize_signal(sig):
    sig = np.asarray(sig, dtype=np.float64).squeeze()
    if sig.ndim != 1:
        sig = sig.reshape(-1)
    if sig.size == 0:
        return sig.astype(np.float64)

    finite_mask = np.isfinite(sig)
    if not finite_mask.any():
        return np.zeros_like(sig, dtype=np.float64)

    fill_value = float(np.median(sig[finite_mask]))
    sig = sig.copy()
    sig[~finite_mask] = fill_value
    return sig.astype(np.float64)


def bandpass(sig, fs, low, high, order=3):
    sig = sanitize_signal(sig)
    if sig.size < max(32, order * 12):
        return sig.copy()

    nyq = 0.5 * float(fs)
    low = max(float(low), 1e-5)
    high = min(float(high), nyq * 0.99)
    if not np.isfinite(low) or not np.isfinite(high) or high <= low:
        return sig.copy()

    try:
        sos = butter(order, [low / nyq, high / nyq], btype="band", output="sos")
        return sosfiltfilt(sos, sig).astype(np.float64)
    except ValueError:
        return sig.copy()


def maybe_resample(sig, src_fs, dst_fs):
    sig = sanitize_signal(sig)
    src_fs = int(round(float(src_fs)))
    dst_fs = int(round(float(dst_fs)))
    if src_fs <= 0 or dst_fs <= 0:
        raise ValueError(f"Invalid sampling rate: src={src_fs}, dst={dst_fs}")
    if sig.size == 0 or src_fs == dst_fs:
        return sig.astype(np.float64)

    factor = gcd(src_fs, dst_fs)
    up = dst_fs // factor
    down = src_fs // factor
    return resample_poly(sig, up, down).astype(np.float64)


def safe_pearsonr(a, b):
    a = sanitize_signal(a)
    b = sanitize_signal(b)
    if len(a) != len(b):
        m = min(len(a), len(b))
        a = a[:m]
        b = b[:m]
    if len(a) < 2 or np.std(a) < 1e-8 or np.std(b) < 1e-8:
        return np.nan
    return float(pearsonr(a, b)[0])


def correct_polarity_skew(sig):
    sig = sanitize_signal(sig)
    if sig.size < 3:
        return sig
    return -sig if skew(sig) < 0 else sig


def align_phase_only(ppg_ref, target_sig):
    ppg_ref = sanitize_signal(ppg_ref)
    target_sig = sanitize_signal(target_sig)
    m = min(len(ppg_ref), len(target_sig))
    ppg_ref = ppg_ref[:m]
    target_sig = target_sig[:m]
    corr = safe_pearsonr(ppg_ref, target_sig)
    return -target_sig if np.isfinite(corr) and corr < 0 else target_sig


def get_hilbert_features(sig, fs):
    sig = sanitize_signal(sig)
    if sig.size < 3:
        return np.zeros_like(sig), np.zeros_like(sig)

    analytic_signal = hilbert(sig)
    envelope = np.abs(analytic_signal)
    phase = np.unwrap(np.angle(analytic_signal))
    inst_freq = np.diff(phase) / (2.0 * np.pi) * float(fs)
    if inst_freq.size == 0:
        inst_freq = np.zeros_like(sig)
    else:
        inst_freq = np.insert(inst_freq, 0, inst_freq[0])

    envelope = np.nan_to_num(envelope, nan=0.0, posinf=0.0, neginf=0.0)
    inst_freq = np.nan_to_num(inst_freq, nan=0.0, posinf=0.0, neginf=0.0)
    if inst_freq.size > 0:
        lo, hi = np.percentile(inst_freq, [0.5, 99.5])
        if np.isfinite(lo) and np.isfinite(hi) and hi > lo:
            inst_freq = np.clip(inst_freq, lo, hi)

    return envelope.astype(np.float64), inst_freq.astype(np.float64)


def extract_windows_stride(sig, fs, win_sec=WIN_SEC, stride_sec=STRIDE_SEC):
    sig = np.asarray(sig)
    win_len = int(round(win_sec * fs))
    step = int(round(stride_sec * fs))
    if win_len <= 0 or step <= 0:
        raise ValueError(f"Invalid window config: win_len={win_len}, step={step}")

    length = sig.shape[-1]
    if length < win_len:
        return np.array([])

    windows = []
    is_multichannel = sig.ndim > 1
    for st in range(0, length - win_len + 1, step):
        if is_multichannel:
            windows.append(sig[:, st : st + win_len])
        else:
            windows.append(sig[st : st + win_len])
    return np.asarray(windows)


def instance_zscore_normalize(data, clip_value=None):
    data = np.asarray(data, dtype=np.float32)
    time_axis = data.ndim - 1
    mean_val = data.mean(axis=time_axis, keepdims=True)
    std_val = data.std(axis=time_axis, keepdims=True)
    std_val[std_val < 1e-6] = 1.0
    norm = (data - mean_val) / std_val
    if clip_value is not None:
        norm = np.clip(norm, -clip_value, clip_value)
    return norm.astype(np.float32)


def normalize_for_plot(sig):
    sig = sanitize_signal(sig)
    return ((sig - np.mean(sig)) / (np.std(sig) + 1e-8)).astype(np.float64)


def waveform_mae(a, b):
    a = sanitize_signal(a)
    b = sanitize_signal(b)
    m = min(len(a), len(b))
    if m == 0:
        return np.nan
    return float(np.mean(np.abs(a[:m] - b[:m])))


def breaths_per_min_zc_single(pred, true, window_sec, fs=FS_METRIC):
    pred = bandpass(pred, fs, 0.1, 0.5, order=3)
    true = bandpass(true, fs, 0.1, 0.5, order=3)
    scale = 60.0 / float(window_sec)
    p_c = pred - np.mean(pred)
    t_c = true - np.mean(true)
    if len(p_c) < 2 or len(t_c) < 2:
        return np.nan
    zc_pred = ((p_c[:-1] * p_c[1:]) < 0).sum()
    zc_true = ((t_c[:-1] * t_c[1:]) < 0).sum()
    bpm_pred = (zc_pred / 2.0) * scale
    bpm_true = (zc_true / 2.0) * scale
    return float(abs(bpm_pred - bpm_true))


def check_quality_real_world(x_low, y_true, fs):
    x_low = sanitize_signal(x_low)
    y_true = sanitize_signal(y_true)
    if len(x_low) != len(y_true):
        m = min(len(x_low), len(y_true))
        x_low = x_low[:m]
        y_true = y_true[:m]
    if len(y_true) < int(4 * fs):
        return False

    y_norm = y_true - np.mean(y_true)
    if np.std(y_norm) < 1e-4:
        return False

    acor = correlate(y_norm, y_norm, mode="full")
    acor = acor[len(acor) // 2 :]
    acor /= acor[0] + 1e-8
    peaks, _ = find_peaks(acor, height=0.20)
    min_peak = int(1.5 * fs)
    max_peak = int(10.0 * fs)
    if not any(min_peak < p < max_peak for p in peaks):
        return False

    corr = safe_pearsonr(x_low, y_true)
    if not np.isfinite(corr) or corr < QUALITY_MIN_CORR:
        return False
    return True


def get_cache_path(cache_dir, dataset_key, subject):
    return os.path.join(cache_dir, f"{safe_tag(dataset_key)}_{safe_tag(subject)}_processed.npz")


def load_subject_cache(cache_dir, dataset_key, subject):
    if not USE_DATA_CACHE:
        return None

    cache_path = get_cache_path(cache_dir, dataset_key, subject)
    if not os.path.exists(cache_path):
        return None

    try:
        cached = np.load(cache_path, allow_pickle=False)
        return {
            "low": cached["low"].astype(np.float32),
            "high": cached["high"].astype(np.float32),
            "env": cached["env"].astype(np.float32),
            "freq": cached["freq"].astype(np.float32),
            "y": cached["y"].astype(np.float32),
            "fs": int(cached["fs"]),
        }
    except Exception as exc:
        print(f"   [Cache Load Fail] {dataset_key}/{subject}: {exc}")
        return None


def save_subject_cache(cache_dir, dataset_key, subject, data):
    if not USE_DATA_CACHE:
        return

    cache_path = get_cache_path(cache_dir, dataset_key, subject)
    try:
        np.savez_compressed(
            cache_path,
            low=data["low"].astype(np.float32),
            high=data["high"].astype(np.float32),
            env=data["env"].astype(np.float32),
            freq=data["freq"].astype(np.float32),
            y=data["y"].astype(np.float32),
            fs=np.asarray(data["fs"], dtype=np.int32),
        )
    except Exception as exc:
        print(f"   [Cache Save Fail] {dataset_key}/{subject}: {exc}")


def preprocess_subject_windows(dataset_key, subject, ppg_raw, target_raw, fs, cache_dir):
    cached = load_subject_cache(cache_dir, dataset_key, subject)
    if cached is not None:
        print(f"Loaded cache {subject}: {len(cached['low'])} windows | fs={cached['fs']}")
        return cached

    ppg_raw = sanitize_signal(ppg_raw)
    target_raw = sanitize_signal(target_raw)
    m = min(len(ppg_raw), len(target_raw))
    if m < int(WIN_SEC * fs):
        return None

    ppg_raw = ppg_raw[:m]
    target_raw = target_raw[:m]
    valid = np.isfinite(ppg_raw) & np.isfinite(target_raw)
    ppg_raw = ppg_raw[valid]
    target_raw = target_raw[valid]
    if len(ppg_raw) < int(WIN_SEC * fs):
        return None

    ppg_raw = correct_polarity_skew(ppg_raw)
    p_low = bandpass(ppg_raw, fs, 0.1, 0.5)
    p_high = bandpass(ppg_raw, fs, 0.5, 5.0)
    target_f = bandpass(target_raw, fs, 0.1, 0.5)
    target_f = align_phase_only(p_low, target_f)
    h_env, h_freq = get_hilbert_features(p_high, fs)

    low_win = extract_windows_stride(p_low, fs)
    high_win = extract_windows_stride(p_high, fs)
    env_win = extract_windows_stride(h_env, fs)
    freq_win = extract_windows_stride(h_freq, fs)
    y_win = extract_windows_stride(target_f, fs)

    n_win = min(len(low_win), len(high_win), len(env_win), len(freq_win), len(y_win))
    if n_win == 0:
        return None

    low_all, high_all, env_all, freq_all, y_all = [], [], [], [], []
    for i in range(n_win):
        if check_quality_real_world(low_win[i], y_win[i], fs):
            low_all.append(low_win[i].astype(np.float32).copy())
            high_all.append(high_win[i].astype(np.float32).copy())
            env_all.append(env_win[i].astype(np.float32).copy())
            freq_all.append(freq_win[i].astype(np.float32).copy())
            y_all.append(y_win[i].astype(np.float32).copy())

    if not low_all:
        return None

    data = {
        "low": np.stack(low_all, axis=0).astype(np.float32),
        "high": np.stack(high_all, axis=0).astype(np.float32),
        "env": np.stack(env_all, axis=0).astype(np.float32),
        "freq": np.stack(freq_all, axis=0).astype(np.float32),
        "y": np.stack(y_all, axis=0).astype(np.float32),
        "fs": int(fs),
    }
    save_subject_cache(cache_dir, dataset_key, subject, data)
    return data


# ===========================================================
# 3. Dataset loading
# ===========================================================
def load_bidmc_dataset(dirs):
    cfg = DATASET_CONFIGS["bidmc"]
    subject_data = {}

    if not os.path.exists(cfg["data_path"]):
        raise FileNotFoundError(f"BIDMC file not found: {cfg['data_path']}")

    mat = sio.loadmat(cfg["data_path"])
    all_data = mat["data"][0]

    for i in range(len(all_data)):
        subject = f"{i + 1:02d}"
        try:
            subj_data = all_data[i]
            ppg_raw = np.asarray(subj_data["ppg"][0, 0]["v"]).squeeze()
            target_raw = np.asarray(subj_data["ref"][0, 0]["resp_sig"][0, 0]["imp"][0, 0]["v"]).squeeze()
            fs = int(cfg["target_fs"])

            result = preprocess_subject_windows(
                dataset_key="bidmc",
                subject=subject,
                ppg_raw=ppg_raw,
                target_raw=target_raw,
                fs=fs,
                cache_dir=dirs["cache"],
            )
            if result is not None:
                subject_data[subject] = result
                print(f"Loaded BIDMC {subject}: {len(result['low'])} windows | fs={result['fs']}")
        except Exception as exc:
            print(f"   [Load Fail] BIDMC {subject}: {exc}")

    return subject_data


def load_capnobase_dataset(dirs):
    cfg = DATASET_CONFIGS["capnobase"]
    subject_data = {}
    csv_dir = cfg["data_path"]

    if not os.path.isdir(csv_dir):
        raise FileNotFoundError(f"CapnoBase CSV directory not found: {csv_dir}")

    all_files = os.listdir(csv_dir)
    case_ids = sorted({f.split("_")[0] for f in all_files if f.endswith("_8min_signal.csv")})

    for case_id in case_ids:
        try:
            signal_path = os.path.join(csv_dir, f"{case_id}_8min_signal.csv")
            param_path = os.path.join(csv_dir, f"{case_id}_8min_param.csv")
            sig_df = pd.read_csv(signal_path)
            param_df = pd.read_csv(param_path)

            orig_fs = int(param_df["samplingrate_pleth"].values[0])
            ppg_raw = sig_df["pleth_y"].values.astype(np.float64)
            target_raw = sig_df["co2_y"].values.astype(np.float64)

            ppg_res = maybe_resample(ppg_raw, orig_fs, cfg["target_fs"])
            target_res = maybe_resample(target_raw, orig_fs, cfg["target_fs"])

            result = preprocess_subject_windows(
                dataset_key="capnobase",
                subject=case_id,
                ppg_raw=ppg_res,
                target_raw=target_res,
                fs=int(cfg["target_fs"]),
                cache_dir=dirs["cache"],
            )
            if result is not None:
                subject_data[case_id] = result
                print(f"Loaded CapnoBase {case_id}: {len(result['low'])} windows | fs={result['fs']}")
        except Exception as exc:
            print(f"   [Load Fail] CapnoBase {case_id}: {exc}")

    return subject_data


# STEAM2 readers (EEGLAB .mat sessions), as in 261028_Deep_correncoder / 261105
STEAM2_CHANNELS = {"PPG": ("PPG", "PLETH", "BVP"), "RSP": ("RSP", "RESP", "RESPIRATION", "BELT")}
STEAM2_SESSION_PATTERN = re.compile(r"^(?P<subject>.+?)[_\- ]*session[_\- ]*(?P<session>\d+)$", re.IGNORECASE)


def _unwrap_mat(value):
    arr = np.asarray(value)
    while arr.dtype == object and arr.size == 1:
        arr = np.asarray(arr.item())
    return arr


def _first_mat_payload(mat):
    keys = [key for key in mat if not key.startswith("_")]
    if not keys:
        raise KeyError("No non-private MATLAB variable found")
    payload = _unwrap_mat(mat[keys[0]])
    if payload.ndim >= 2 and payload.shape[0] == 1 and payload.shape[1] == 1:
        payload = payload[0, 0]
    return payload


def _extract_signal_and_fs(payload, default_fs):
    signal = np.asarray(_unwrap_mat(payload["data"]), dtype=np.float64).reshape(-1)
    try:
        fs = float(np.asarray(_unwrap_mat(payload["srate"])).squeeze())
    except Exception:
        fs = float(default_fs)
    return signal.astype(np.float32), fs


def _as_list(value):
    if value is None:
        return []
    if isinstance(value, np.ndarray):
        return list(value.ravel()) if value.size else []
    return [value]


def steam2_session_key(stem):
    """(subject, session number) from a file stem such as 13_LSB_Session2."""
    match = STEAM2_SESSION_PATTERN.match(stem)
    if match is None:
        return stem, 0
    return match.group("subject"), int(match.group("session"))


def read_eeglab_channel(path, kind, default_fs):
    """One channel of an EEGLAB .mat file, chosen by its label; a one-channel
    file without a label is accepted, one labelled as the other signal is
    rejected (swapped files). The positions of "boundary" events inside the
    recording are returned so that the session is split there."""
    mat = sio.loadmat(path, squeeze_me=True, struct_as_record=False)
    names = [key for key in mat if not key.startswith("_")]
    payload = mat["EEG"] if "EEG" in mat else (mat[names[0]] if names else None)
    if payload is None or not hasattr(payload, "data"):
        signal, fs = _extract_signal_and_fs(_first_mat_payload(sio.loadmat(path)), default_fs)
        return {"signal": signal, "fs": fs, "boundaries": [], "label": "", "channels": 1}
    data = np.asarray(payload.data, dtype=np.float64)
    data = data.reshape(1, -1) if data.ndim == 1 else data
    labels = [str(getattr(item, "labels", "")).strip() for item in _as_list(getattr(payload, "chanlocs", None))]
    wanted = {name.upper() for name in STEAM2_CHANNELS[kind]}
    other = {name.upper() for key, names in STEAM2_CHANNELS.items() if key != kind for name in names}
    index = next((i for i, label in enumerate(labels) if label.upper() in wanted), None)
    if index is None:
        if data.shape[0] != 1:
            raise RuntimeError(f"{Path(path).name}: no {kind} channel among {labels}")
        if labels and labels[0].upper() in other:
            raise RuntimeError(f"{Path(path).name}: channel '{labels[0]}' found where {kind} was expected")
        index = 0
    signal = data[index].astype(np.float32)
    try:
        fs = float(np.asarray(payload.srate).squeeze())
    except Exception:
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
        # k and k + 1. Markers at the start and the end are not gaps.
        position = int(np.ceil(latency - 1.0))
        if 0 < position < signal.size:
            boundaries.append(position)
    label = labels[index] if index < len(labels) else ""
    return {"signal": signal, "fs": fs, "boundaries": sorted(set(boundaries)), "label": label,
            "channels": int(data.shape[0])}


def steam2_session_files(ppg_root, rsp_root):
    """Subject -> [(session name, PPG file, RSP file)], sessions in order.
    Subject folders are used when present; otherwise the files of one flat
    folder are grouped by the subject part of their name."""
    ppg_root, rsp_root = Path(ppg_root), Path(rsp_root)
    sessions = {}
    subject_dirs = sorted(path for path in ppg_root.iterdir() if path.is_dir())
    if subject_dirs:
        for subject_dir in subject_dirs:
            rsp_dir = rsp_root / subject_dir.name
            if not rsp_dir.is_dir():
                print(f"[STEAM2 {subject_dir.name}] no RSP folder, skipped")
                continue
            rsp_files = {path.stem: path for path in rsp_dir.glob("*.mat")}
            for ppg_path in sorted(subject_dir.glob("*.mat")):
                if ppg_path.stem not in rsp_files:
                    print(f"[STEAM2 {subject_dir.name}] {ppg_path.name}: no RSP file of the same name, left out")
                    continue
                sessions.setdefault(subject_dir.name, []).append(
                    (ppg_path.stem, ppg_path, rsp_files[ppg_path.stem])
                )
            for stem in sorted(set(rsp_files) - {path.stem for path in subject_dir.glob("*.mat")}):
                print(f"[STEAM2 {subject_dir.name}] {stem}.mat: no PPG file of the same name, left out")
    else:
        rsp_files = {path.stem: path for path in rsp_root.glob("*.mat")}
        for ppg_path in sorted(ppg_root.glob("*.mat")):
            if ppg_path.stem not in rsp_files:
                print(f"[STEAM2 {ppg_path.name}] no RSP file of the same name, skipped")
                continue
            subject, _ = steam2_session_key(ppg_path.stem)
            sessions.setdefault(subject, []).append((ppg_path.stem, ppg_path, rsp_files[ppg_path.stem]))
    for subject in sessions:
        sessions[subject].sort(key=lambda item: (steam2_session_key(item[0])[1], item[0]))
    return sessions


def steam2_session_pieces(ppg, rsp):
    """The (PPG, RSP, PPG fs, RSP fs) pieces of one session, split at the
    boundary events of either file, and a description of the session."""
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
    info = {"PPG_Sec": ppg_sec, "RSP_Sec": rsp_sec, "Length_Difference_Sec": ppg_sec - rsp_sec,
            "Internal_Boundaries": len(edges) - 2, "Pieces": len(pieces)}
    return pieces, info


def load_steam2_dataset(dirs):
    """Every piece of every session is resampled to 125 Hz and goes through
    preprocess_subject_windows on its own (cached per piece); the windows of
    a subject's pieces are then stacked in session order."""
    cfg = DATASET_CONFIGS["steam2"]
    subject_data = {}
    ppg_root, rsp_root = Path(cfg["ppg_dir"]), Path(cfg["rsp_dir"])
    if not ppg_root.is_dir() or not rsp_root.is_dir():
        raise FileNotFoundError(f"STEAM2 folders not found: {ppg_root} / {rsp_root}")
    fs = int(cfg["target_fs"])
    keys = ("low", "high", "env", "freq", "y")

    for subject, items in steam2_session_files(ppg_root, rsp_root).items():
        parts = []
        try:
            for name, ppg_path, rsp_path in items:
                ppg = read_eeglab_channel(ppg_path, "PPG", cfg["default_fs"])
                rsp = read_eeglab_channel(rsp_path, "RSP", cfg["default_fs"])
                pieces, info = steam2_session_pieces(ppg, rsp)
                if abs(info["Length_Difference_Sec"]) > 1.0:
                    print(f"[STEAM2 {subject} {name}] PPG {info['PPG_Sec']:.1f} s vs RSP "
                          f"{info['RSP_Sec']:.1f} s; the common part is used")
                for k, (ppg_piece, rsp_piece, ppg_fs, rsp_fs) in enumerate(pieces):
                    result = preprocess_subject_windows(
                        dataset_key="steam2",
                        subject=f"{subject}__{name}_piece{k + 1}",
                        ppg_raw=maybe_resample(ppg_piece, ppg_fs, fs),
                        target_raw=maybe_resample(rsp_piece, rsp_fs, fs),
                        fs=fs,
                        cache_dir=dirs["cache"],
                    )
                    if result is not None:
                        parts.append(result)
        except Exception as exc:
            print(f"   [Load Fail] STEAM2 {subject}: {exc}")
            continue
        if parts:
            data = {key: np.concatenate([part[key] for part in parts], axis=0).astype(np.float32) for key in keys}
            data["fs"] = fs
            subject_data[subject] = data
            print(f"Loaded STEAM2 {subject}: {len(data['low'])} windows | sessions={len(items)} | fs={fs}")

    return subject_data


def load_dataset(dataset_key, dirs):
    print(f"\nLoading dataset: {DATASET_CONFIGS[dataset_key]['label']}")
    if dataset_key == "bidmc":
        return load_bidmc_dataset(dirs)
    if dataset_key == "capnobase":
        return load_capnobase_dataset(dirs)
    if dataset_key == "steam2":
        return load_steam2_dataset(dirs)
    raise KeyError(f"Unsupported dataset key: {dataset_key}")


# ===========================================================
# 4. Loss
# ===========================================================
class PearsonLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, pred, target):
        if pred.dim() > 2:
            pred = pred.squeeze(1)
        if target.dim() > 2:
            target = target.squeeze(1)
        v_pred = pred - torch.mean(pred, dim=1, keepdim=True)
        v_target = target - torch.mean(target, dim=1, keepdim=True)
        denom = torch.sqrt(torch.sum(v_pred**2, dim=1) * torch.sum(v_target**2, dim=1)) + 1e-6
        corr = torch.sum(v_pred * v_target, dim=1) / denom
        return 1.0 - torch.mean(corr)


class TemporalHybridLoss(nn.Module):
    def __init__(
        self,
        point_weight=LOSS_POINT_WEIGHT,
        corr_weight=LOSS_CORR_WEIGHT,
        deriv_weight=LOSS_DERIV_WEIGHT,
    ):
        super().__init__()
        total = point_weight + corr_weight + deriv_weight
        self.point_weight = point_weight / total
        self.corr_weight = corr_weight / total
        self.deriv_weight = deriv_weight / total
        self.point = nn.SmoothL1Loss(beta=0.5)
        self.corr = PearsonLoss()

    def forward(self, pred, target):
        if target.dim() == 2:
            target = target.unsqueeze(1)
        point_loss = self.point(pred, target)
        corr_loss = self.corr(pred, target)
        pred_d = pred[:, :, 1:] - pred[:, :, :-1]
        target_d = target[:, :, 1:] - target[:, :, :-1]
        deriv_loss = self.point(pred_d, target_d)
        return (
            self.point_weight * point_loss
            + self.corr_weight * corr_loss
            + self.deriv_weight * deriv_loss
        )


# ===========================================================
# 5. Corr-Encoder model
# ===========================================================
def init_weights(module):
    if isinstance(module, (nn.Conv1d, nn.ConvTranspose1d)):
        nn.init.kaiming_normal_(module.weight, nonlinearity="relu")
        if module.bias is not None:
            nn.init.zeros_(module.bias)
    elif isinstance(module, nn.BatchNorm1d):
        nn.init.ones_(module.weight)
        nn.init.zeros_(module.bias)


class ConvBlock(nn.Module):
    def __init__(self, in_c, out_c, kernel, dropout=0.2):
        super().__init__()
        padding = kernel // 2
        self.conv1 = nn.Conv1d(in_c, out_c, kernel_size=kernel, padding=padding)
        self.bn1 = nn.BatchNorm1d(out_c)
        self.conv2 = nn.Conv1d(out_c, out_c, kernel_size=kernel, padding=padding)
        self.bn2 = nn.BatchNorm1d(out_c)
        self.shortcut = nn.Conv1d(in_c, out_c, kernel_size=1) if in_c != out_c else nn.Identity()
        self.act = nn.GELU()
        self.drop = nn.Dropout(dropout)

    def forward(self, x):
        residual = self.shortcut(x)
        out = self.act(self.bn1(self.conv1(x)))
        out = self.drop(out)
        out = self.bn2(self.conv2(out))
        return self.act(out + residual)


class SingleEncoder(nn.Module):
    def __init__(self, n_in, filters, k_sizes):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv1d(n_in, filters[0], kernel_size=15, stride=4, padding=5),
            nn.BatchNorm1d(filters[0]),
            nn.GELU(),
            nn.Dropout(STEM_DROPOUT),
        )
        self.enc1 = ConvBlock(filters[0], filters[0], k_sizes[0], BLOCK_DROPOUT)
        self.enc2 = ConvBlock(filters[0], filters[1], k_sizes[1], BLOCK_DROPOUT)
        self.enc3 = ConvBlock(filters[1], filters[2], k_sizes[2], BLOCK_DROPOUT)

    def forward(self, x):
        stem = self.stem(x)
        e1 = self.enc1(stem)
        e2 = self.enc2(e1)
        e3 = self.enc3(e2)
        return stem, e3


class DecoderBlock(nn.Module):
    def __init__(self, in_c, out_c, kernel, dropout=0.2):
        super().__init__()
        padding = kernel // 2
        self.conv = nn.Conv1d(in_c, out_c, kernel_size=kernel, padding=padding)
        self.bn = nn.BatchNorm1d(out_c)
        self.act = nn.GELU()
        self.drop = nn.Dropout(dropout)

    def forward(self, x):
        return self.drop(self.act(self.bn(self.conv(x))))


class CorrEncoderNet(nn.Module):
    def __init__(self, n_primary_channels=1, n_aux_channels=0):
        super().__init__()
        filters = FILTERS
        k_sizes = K_SIZES
        self.n_primary_channels = n_primary_channels
        self.n_aux_channels = n_aux_channels

        self.encoder_primary = SingleEncoder(n_primary_channels, filters, k_sizes)
        if n_aux_channels > 0:
            self.encoder_aux = SingleEncoder(n_aux_channels, filters, k_sizes)
            fusion_in = filters[2] * 2
        else:
            self.encoder_aux = None
            fusion_in = filters[2]

        self.fusion = nn.Sequential(
            nn.Conv1d(fusion_in, filters[2], kernel_size=1),
            nn.BatchNorm1d(filters[2]),
            nn.GELU(),
            nn.Dropout(FUSION_DROPOUT),
        )
        self.dec1 = DecoderBlock(filters[2], filters[1], k_sizes[2], BLOCK_DROPOUT)
        self.dec2 = DecoderBlock(filters[1], filters[0], k_sizes[1], BLOCK_DROPOUT)
        self.dec3 = DecoderBlock(filters[0], filters[0], k_sizes[0], BLOCK_DROPOUT)
        self.head = nn.ConvTranspose1d(filters[0], 1, kernel_size=15, stride=4, padding=5)
        self.apply(init_weights)

    def forward(self, x_primary, x_aux=None):
        stem_primary, feat_primary = self.encoder_primary(x_primary)
        if self.encoder_aux is not None:
            if x_aux is None:
                raise ValueError("x_aux is required when n_aux_channels > 0.")
            _, feat_aux = self.encoder_aux(x_aux)
            fused_input = torch.cat([feat_primary, feat_aux], dim=1)
        else:
            fused_input = feat_primary

        fused = self.fusion(fused_input)
        dec = self.dec3(self.dec2(self.dec1(fused)))
        out = self.head(dec + stem_primary)
        if out.shape[-1] != x_primary.shape[-1]:
            out = F.interpolate(out, size=x_primary.shape[-1], mode="linear", align_corners=False)
        return out


# ===========================================================
# 6. Split construction and dataloaders
# ===========================================================
def concat_subjects(subject_data, subjects, keys, return_subject_ids=False):
    out_lists = {key: [] for key in keys}
    subject_ids = []

    for subj in subjects:
        subj_pack = subject_data[subj]
        n_samples = len(subj_pack["y"])
        if n_samples == 0:
            continue
        for key in keys:
            out_lists[key].append(subj_pack[key])
        if return_subject_ids:
            subject_ids.extend([subj] * n_samples)

    if not out_lists[keys[0]]:
        raise RuntimeError(f"No samples found for subjects: {subjects}")

    out = {key: np.vstack(out_lists[key]) for key in keys}
    if return_subject_ids:
        return out, np.asarray(subject_ids)
    return out


def make_feature_tensor(data_dict, feature_keys):
    channels = [data_dict[key] for key in feature_keys]
    return np.stack(channels, axis=1).astype(np.float32)


def build_split_arrays(subject_data, subjects, feature_keys, return_subject_ids=False):
    if return_subject_ids:
        raw, subject_ids = concat_subjects(
            subject_data,
            subjects,
            ["low", "high", "env", "freq", "y"],
            return_subject_ids=True,
        )
    else:
        raw = concat_subjects(subject_data, subjects, ["low", "high", "env", "freq", "y"])
        subject_ids = None

    y = instance_zscore_normalize(raw["y"], clip_value=None)

    if "low" in feature_keys:
        primary = instance_zscore_normalize(raw["low"], clip_value=INPUT_CLIP_VALUE)
        aux_keys = [key for key in feature_keys if key != "low"]
        aux = None
        if aux_keys:
            aux_raw = make_feature_tensor(raw, aux_keys)
            aux = instance_zscore_normalize(aux_raw, clip_value=INPUT_CLIP_VALUE)
    else:
        primary_raw = make_feature_tensor(raw, feature_keys)
        primary = instance_zscore_normalize(primary_raw, clip_value=INPUT_CLIP_VALUE)
        aux = None

    return primary, aux, y, subject_ids


def to_channel_tensor(x):
    x_t = torch.tensor(x, dtype=torch.float32)
    if x_t.dim() == 2:
        x_t = x_t.unsqueeze(1)
    return x_t


def make_balanced_sampler(subject_ids):
    counts = Counter(subject_ids.tolist())
    weights = np.asarray([1.0 / counts[subj] for subj in subject_ids], dtype=np.float64)
    return WeightedRandomSampler(
        weights=torch.from_numpy(weights),
        num_samples=len(weights),
        replacement=True,
    )


def build_loader(x_primary, x_aux, y, batch_size=BATCH_SIZE, shuffle=False, subject_ids=None):
    primary_t = to_channel_tensor(x_primary)
    y_t = torch.tensor(y, dtype=torch.float32)
    if x_aux is None:
        aux_t = torch.zeros((len(x_primary), 1, x_primary.shape[-1]), dtype=torch.float32)
    else:
        aux_t = to_channel_tensor(x_aux)

    dataset = TensorDataset(primary_t, aux_t, y_t)
    sampler = None
    if shuffle and BALANCE_TRAIN_SUBJECTS and subject_ids is not None:
        sampler = make_balanced_sampler(subject_ids)
        shuffle = False

    drop_last = (shuffle or sampler is not None) and len(x_primary) > batch_size
    kwargs = {
        "dataset": dataset,
        "batch_size": batch_size,
        "shuffle": shuffle if sampler is None else False,
        "sampler": sampler,
        "drop_last": drop_last,
        "num_workers": NUM_WORKERS,
        "pin_memory": PIN_MEMORY,
    }
    if NUM_WORKERS > 0:
        kwargs["persistent_workers"] = True
        kwargs["prefetch_factor"] = PREFETCH_FACTOR
    return DataLoader(**kwargs)


def add_input_noise(x, std=INPUT_NOISE_STD, clip_value=INPUT_CLIP_VALUE):
    if std <= 0:
        return x
    x = x + torch.randn_like(x) * std
    if clip_value is not None:
        x = torch.clamp(x, -clip_value, clip_value)
    return x


def select_validation_subjects(all_subjects, test_subj):
    train_val_subjs = [subj for subj in all_subjects if subj != test_subj]
    n_available = len(train_val_subjs)
    n_val = max(MIN_VAL_SUBJECTS, int(round(n_available * VAL_RATIO)))
    n_val = min(n_val, max(1, n_available - 1))

    pivot = all_subjects.index(test_subj) % n_available
    ordered = train_val_subjs[pivot:] + train_val_subjs[:pivot]
    stride = max(1, n_available // n_val)

    val_subjs = []
    cursor = 0
    while len(val_subjs) < n_val:
        candidate = ordered[cursor % n_available]
        if candidate not in val_subjs:
            val_subjs.append(candidate)
        cursor += stride

    train_subjs = [subj for subj in train_val_subjs if subj not in val_subjs]
    if not train_subjs:
        raise RuntimeError("No training subjects left after validation split.")
    return train_subjs, val_subjs


# ===========================================================
# 7. Training, evaluation, plotting
# ===========================================================
def save_learning_curve(train_losses, val_losses, val_ema_losses, dataset_key, test_subj, model_name, dirs):
    plt.figure(figsize=(10, 5))
    plt.plot(train_losses, label="Train")
    plt.plot(val_losses, label="Val")
    if val_ema_losses:
        plt.plot(val_ema_losses, label="Val EMA", linestyle="--")
    plt.title(f"{DATASET_CONFIGS[dataset_key]['label']} | Loss - {test_subj} - {model_name}")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(
        os.path.join(dirs["curves"], f"{safe_tag(test_subj)}_{model_name}_loss.png"),
        dpi=150,
    )
    plt.close()


def build_model_from_config(config):
    feature_keys = config["feature_keys"]
    use_low = "low" in feature_keys
    aux_keys = [key for key in feature_keys if key != "low"] if use_low else []
    n_primary_channels = 1 if use_low else len(feature_keys)
    n_aux_channels = len(aux_keys)
    return CorrEncoderNet(
        n_primary_channels=n_primary_channels,
        n_aux_channels=n_aux_channels,
    ), n_aux_channels


def train_one_model(config, dataset_key, test_subj, train_subjs, val_subjs, subject_data, dirs):
    model_name = config["name"]
    feature_keys = config["feature_keys"]

    X_p_tr, X_a_tr, y_tr, train_ids = build_split_arrays(
        subject_data,
        train_subjs,
        feature_keys,
        return_subject_ids=True,
    )
    X_p_val, X_a_val, y_val, _ = build_split_arrays(subject_data, val_subjs, feature_keys)
    X_p_te, X_a_te, y_te, _ = build_split_arrays(subject_data, [test_subj], feature_keys)

    train_loader = build_loader(X_p_tr, X_a_tr, y_tr, shuffle=True, subject_ids=train_ids)
    val_loader = build_loader(X_p_val, X_a_val, y_val)
    test_loader = build_loader(X_p_te, X_a_te, y_te)

    model, n_aux_channels = build_model_from_config(config)
    model = model.to(device)
    criterion = TemporalHybridLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    scheduler = ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=0.5,
        patience=SCHEDULER_PATIENCE,
        threshold=MIN_DELTA,
        min_lr=MIN_LR,
    )
    scaler = build_grad_scaler()

    best_val_loss = np.inf
    best_monitor_loss = np.inf
    val_ema = None
    best_state = None
    counter = 0
    train_losses, val_losses, val_ema_losses = [], [], []
    model_path = os.path.join(dirs["models"], f"{safe_tag(test_subj)}_{model_name}_best.pth")
    trainable_params = count_trainable_params(model)

    print(
        f"Windows | Train={len(train_loader.dataset)} | Val={len(val_loader.dataset)} | "
        f"Test={len(test_loader.dataset)} | Params={trainable_params:,}"
    )

    for ep in range(EPOCHS):
        model.train()
        tr_loss = 0.0
        n_train_batches = 0

        for xp, xa, y in train_loader:
            xp = xp.to(device, non_blocking=PIN_MEMORY)
            xa = xa.to(device, non_blocking=PIN_MEMORY)
            y = y.to(device, non_blocking=PIN_MEMORY)

            xp = add_input_noise(xp)
            if n_aux_channels > 0:
                xa = add_input_noise(xa)

            optimizer.zero_grad(set_to_none=True)
            xa_in = xa if n_aux_channels > 0 else None

            with autocast_context():
                out = model(xp, xa_in)
                loss = criterion(out, y.unsqueeze(1))

            if not torch.isfinite(loss):
                continue

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), MAX_GRAD_NORM)
            scaler.step(optimizer)
            scaler.update()

            tr_loss += float(loss.item())
            n_train_batches += 1

        if n_train_batches == 0:
            raise RuntimeError(f"No valid training batches for {test_subj} / {model_name}")
        tr_loss /= n_train_batches

        model.eval()
        v_loss = 0.0
        n_val_batches = 0
        with torch.no_grad():
            for xp, xa, y in val_loader:
                xp = xp.to(device, non_blocking=PIN_MEMORY)
                xa = xa.to(device, non_blocking=PIN_MEMORY)
                y = y.to(device, non_blocking=PIN_MEMORY)
                xa_in = xa if n_aux_channels > 0 else None

                with autocast_context():
                    pred = model(xp, xa_in)
                    loss = criterion(pred, y.unsqueeze(1))

                if torch.isfinite(loss):
                    v_loss += float(loss.item())
                    n_val_batches += 1

        v_loss = v_loss / n_val_batches if n_val_batches > 0 else np.inf
        if np.isfinite(v_loss):
            val_ema = v_loss if val_ema is None else VAL_EMA_ALPHA * v_loss + (1.0 - VAL_EMA_ALPHA) * val_ema
        else:
            val_ema = np.inf

        monitor_loss = val_ema
        scheduler.step(monitor_loss)
        train_losses.append(tr_loss)
        val_losses.append(v_loss)
        val_ema_losses.append(monitor_loss)

        raw_improved = np.isfinite(v_loss) and (best_val_loss - v_loss > MIN_DELTA)
        if raw_improved:
            best_val_loss = v_loss
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            torch.save(best_state, model_path)

        monitor_improved = (
            np.isfinite(monitor_loss)
            and (best_monitor_loss - monitor_loss > EARLY_STOP_MIN_DELTA)
        )
        near_best_plateau = (
            np.isfinite(monitor_loss)
            and np.isfinite(best_monitor_loss)
            and monitor_loss <= best_monitor_loss + EARLY_STOP_PLATEAU_TOL
        )

        if monitor_improved:
            best_monitor_loss = monitor_loss
            counter = 0
        elif near_best_plateau:
            counter = counter
        else:
            counter += 1

        print(
            f"[{DATASET_CONFIGS[dataset_key]['label']} | {test_subj} | {model_name}] "
            f"Ep {ep + 1:03d} | Tr={tr_loss:.4f} | Val={v_loss:.4f} | "
            f"ValEMA={monitor_loss:.4f} | Best={best_val_loss:.4f} | "
            f"BestEMA={best_monitor_loss:.4f} | LR={optimizer.param_groups[0]['lr']:.6f} | "
            f"Patience={counter}/{PATIENCE}"
        )

        if (ep + 1) >= MIN_EPOCHS and counter >= PATIENCE:
            print(f"[{test_subj} | {model_name}] Early stop at epoch {ep + 1}")
            break

    if best_state is None:
        best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        torch.save(best_state, model_path)

    save_learning_curve(train_losses, val_losses, val_ema_losses, dataset_key, test_subj, model_name, dirs)

    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    all_preds = []
    with torch.no_grad():
        for xp, xa, _ in test_loader:
            xp = xp.to(device, non_blocking=PIN_MEMORY)
            xa = xa.to(device, non_blocking=PIN_MEMORY)
            xa_in = xa if n_aux_channels > 0 else None
            with autocast_context():
                pred = model(xp, xa_in)
            all_preds.append(pred.squeeze(1).float().cpu().numpy())

    all_preds = np.concatenate(all_preds, axis=0)
    return {
        "pred": all_preds,
        "true": y_te,
        "train_losses": train_losses,
        "val_losses": val_losses,
        "val_ema_losses": val_ema_losses,
        "best_val_loss": float(best_val_loss),
        "trainable_params": trainable_params,
    }


def evaluate_30sec_segments(pred_windows, true_windows, fs=FS_METRIC):
    segment_results = []
    num_segments = len(true_windows) // SEGMENT_WINDOWS

    for i in range(num_segments):
        st = i * SEGMENT_WINDOWS
        ed = st + SEGMENT_WINDOWS
        pred_seg = np.concatenate(pred_windows[st:ed])
        true_seg = np.concatenate(true_windows[st:ed])
        pred_f = bandpass(pred_seg, fs, 0.1, 0.5)
        true_f = bandpass(true_seg, fs, 0.1, 0.5)
        pred_n = normalize_for_plot(pred_f)
        true_n = normalize_for_plot(true_f)
        corr = safe_pearsonr(pred_n, true_n)
        mae = waveform_mae(pred_n, true_n)
        bpm_err = breaths_per_min_zc_single(pred_f, true_f, SEGMENT_SEC, fs=fs)
        segment_results.append(
            {
                "segment_idx": i,
                "corr": corr,
                "mae": mae,
                "bpm_err": bpm_err,
                "pred": pred_n,
                "true": true_n,
            }
        )

    return segment_results


def evaluate_overall(pred_windows, true_windows, fs=FS_METRIC):
    if len(pred_windows) == 0 or len(true_windows) == 0:
        return {"corr": np.nan, "mae": np.nan}

    pred_all = bandpass(np.concatenate(pred_windows), fs, 0.1, 0.5)
    true_all = bandpass(np.concatenate(true_windows), fs, 0.1, 0.5)
    pred_n = normalize_for_plot(pred_all)
    true_n = normalize_for_plot(true_all)
    return {
        "corr": safe_pearsonr(pred_n, true_n),
        "mae": waveform_mae(pred_n, true_n),
    }


def plot_subject_comparisons(dataset_key, test_subj, model_outputs, configs, dirs):
    true_windows = next(iter(model_outputs.values()))["true"]
    num_segments = len(true_windows) // SEGMENT_WINDOWS
    subject_dir = os.path.join(dirs["plots"], safe_tag(test_subj))
    os.makedirs(subject_dir, exist_ok=True)

    for seg_idx in range(num_segments):
        st = seg_idx * SEGMENT_WINDOWS
        ed = st + SEGMENT_WINDOWS
        true_seg = normalize_for_plot(
            bandpass(np.concatenate(true_windows[st:ed]), FS_METRIC, 0.1, 0.5)
        )
        t_axis = np.arange(len(true_seg)) / FS_METRIC

        plt.figure(figsize=(16, 7))
        plt.plot(t_axis, true_seg, "k-", linewidth=2.2, label="Ground truth", alpha=0.85)

        for config in configs:
            name = config["name"]
            pred_seg = normalize_for_plot(
                bandpass(np.concatenate(model_outputs[name]["pred"][st:ed]), FS_METRIC, 0.1, 0.5)
            )
            plt.plot(
                t_axis,
                pred_seg,
                linestyle="--",
                linewidth=1.25,
                color=config["color"],
                label=config["label"],
                alpha=0.90,
            )

        plt.title(
            f"{DATASET_CONFIGS[dataset_key]['label']} | {test_subj} | "
            f"30-sec Segment {seg_idx + 1:03d} | Corr-Encoder Ablation"
        )
        plt.xlabel("Time (sec)")
        plt.ylabel("Normalized amplitude")
        plt.legend(loc="upper center", bbox_to_anchor=(0.5, -0.12), ncol=4, fontsize=8)
        plt.grid(True, alpha=0.3)
        plt.tight_layout(rect=[0, 0.08, 1, 1])
        plt.savefig(
            os.path.join(subject_dir, f"{safe_tag(test_subj)}_segment_{seg_idx + 1:03d}_comparison.jpg"),
            dpi=150,
        )
        plt.close()


# ===========================================================
# 8. Experiment orchestration
# ===========================================================
def save_running_metrics(all_metrics, all_segment_metrics, dirs):
    pd.DataFrame(all_metrics).to_csv(
        os.path.join(dirs["metrics"], "LOSO_CorrEncoder_Ablation_Running.csv"),
        index=False,
    )
    pd.DataFrame(all_segment_metrics).to_csv(
        os.path.join(dirs["metrics"], "LOSO_CorrEncoder_Ablation_Segment_Running.csv"),
        index=False,
    )


def make_summary(metrics_df):
    if metrics_df.empty:
        return pd.DataFrame()

    return (
        metrics_df.groupby(["Dataset", "Model", "Label", "Features"], as_index=False)
        .agg(
            BPM_MAE_Mean=("BPM_MAE", "mean"),
            BPM_MAE_Std=("BPM_MAE", "std"),
            Waveform_MAE_30sec_Mean=("Waveform_MAE_30sec_Mean", "mean"),
            Waveform_MAE_30sec_Std=("Waveform_MAE_30sec_Mean", "std"),
            Waveform_Corr_30sec_Mean=("Waveform_Corr_30sec_Mean", "mean"),
            Waveform_Corr_30sec_Std=("Waveform_Corr_30sec_Mean", "std"),
            Waveform_Corr_Overall_Mean=("Waveform_Corr_Overall", "mean"),
            Waveform_Corr_Overall_Std=("Waveform_Corr_Overall", "std"),
            Waveform_MAE_Overall_Mean=("Waveform_MAE_Overall", "mean"),
            Waveform_MAE_Overall_Std=("Waveform_MAE_Overall", "std"),
            Best_Val_Loss_Mean=("Best_Val_Loss", "mean"),
            Trainable_Params=("Trainable_Params", "first"),
        )
        .sort_values(["BPM_MAE_Mean", "Waveform_Corr_30sec_Mean"], ascending=[True, False])
    )


def run_dataset_ablation(dataset_key):
    dataset_cfg = DATASET_CONFIGS[dataset_key]
    dirs = prepare_output_dirs(dataset_key)
    subject_data = load_dataset(dataset_key, dirs)
    all_subjects = sorted(subject_data.keys())

    if len(all_subjects) < MIN_VALID_SUBJECTS:
        raise RuntimeError(
            f"{dataset_cfg['label']} needs at least {MIN_VALID_SUBJECTS} valid subjects. "
            f"Found {len(all_subjects)}."
        )

    print(f"\nDataset: {dataset_cfg['label']}")
    print(f"Valid subjects: {len(all_subjects)}")
    print(f"Total model cases: {len(MODEL_CONFIGS)}")
    for cfg in MODEL_CONFIGS:
        print(f"  {cfg['name']}: {cfg['label']}")

    all_metrics = []
    all_segment_metrics = []

    for test_subj in all_subjects:
        print(f"\n===== [{dataset_cfg['label']}] LOSO Test Subject: {test_subj} =====")
        start_time = time.time()

        train_subjs, val_subjs = select_validation_subjects(all_subjects, test_subj)
        print(f"Validation Subjects ({len(val_subjs)}): {', '.join(val_subjs)}")
        print(f"Train Subjects: {len(train_subjs)}")

        subject_outputs = {}
        for config in MODEL_CONFIGS:
            print(f"\n--- Training {config['name']} ({config['label']}) ---")
            output = train_one_model(
                config=config,
                dataset_key=dataset_key,
                test_subj=test_subj,
                train_subjs=train_subjs,
                val_subjs=val_subjs,
                subject_data=subject_data,
                dirs=dirs,
            )
            subject_outputs[config["name"]] = output

            segment_results = evaluate_30sec_segments(output["pred"], output["true"])
            corr_values = [x["corr"] for x in segment_results if np.isfinite(x["corr"])]
            mae_values = [x["mae"] for x in segment_results if np.isfinite(x["mae"])]
            bpm_values = [x["bpm_err"] for x in segment_results if np.isfinite(x["bpm_err"])]
            avg_corr = float(np.mean(corr_values)) if corr_values else np.nan
            avg_mae = float(np.mean(mae_values)) if mae_values else np.nan
            avg_bpm = float(np.mean(bpm_values)) if bpm_values else np.nan
            overall = evaluate_overall(output["pred"], output["true"])

            elapsed = time.time() - start_time
            all_metrics.append(
                {
                    "Dataset": dataset_cfg["label"],
                    "Subject": test_subj,
                    "Model": config["name"],
                    "Label": config["label"],
                    "Features": "+".join(config["feature_keys"]),
                    "BPM_MAE": avg_bpm,
                    "Waveform_MAE_30sec_Mean": avg_mae,
                    "Waveform_Corr_30sec_Mean": avg_corr,
                    "Waveform_Corr_Overall": overall["corr"],
                    "Waveform_MAE_Overall": overall["mae"],
                    "Best_Val_Loss": output["best_val_loss"],
                    "Trainable_Params": output["trainable_params"],
                    "Elapsed_Sec_From_Subject_Start": elapsed,
                }
            )

            for seg in segment_results:
                all_segment_metrics.append(
                    {
                        "Dataset": dataset_cfg["label"],
                        "Subject": test_subj,
                        "Model": config["name"],
                        "Label": config["label"],
                        "Features": "+".join(config["feature_keys"]),
                        "Segment": seg["segment_idx"] + 1,
                        "BPM_Error": seg["bpm_err"],
                        "Waveform_MAE": seg["mae"],
                        "Waveform_Corr": seg["corr"],
                    }
                )

            print(
                f"Result [{config['name']}]: "
                f"BPM MAE={avg_bpm:.2f}, 30s MAE={avg_mae:.4f}, "
                f"30s Corr={avg_corr:.4f}, Overall Corr={overall['corr']:.4f}"
            )

            save_running_metrics(all_metrics, all_segment_metrics, dirs)

        plot_subject_comparisons(dataset_key, test_subj, subject_outputs, MODEL_CONFIGS, dirs)

    metrics_df = pd.DataFrame(all_metrics)
    segment_df = pd.DataFrame(all_segment_metrics)
    summary_df = make_summary(metrics_df)

    metrics_df.to_csv(
        os.path.join(dirs["metrics"], "LOSO_CorrEncoder_Ablation_PerSubject.csv"),
        index=False,
    )
    segment_df.to_csv(
        os.path.join(dirs["metrics"], "LOSO_CorrEncoder_Ablation_PerSegment.csv"),
        index=False,
    )
    summary_df.to_csv(
        os.path.join(dirs["metrics"], "LOSO_CorrEncoder_Ablation_Summary.csv"),
        index=False,
    )

    print(f"\n=== {dataset_cfg['label']} Final Summary ===")
    print(summary_df)
    print(f"\nSaved to: {dirs['root']}")
    return metrics_df, segment_df, summary_df


def ce_main():
    all_dataset_metrics = []
    all_dataset_segments = []
    all_dataset_summaries = []

    for dataset_key in ACTIVE_DATASETS:
        metrics_df, segment_df, summary_df = run_dataset_ablation(dataset_key)
        all_dataset_metrics.append(metrics_df)
        all_dataset_segments.append(segment_df)
        all_dataset_summaries.append(summary_df)

    combined_metric_dir = os.path.join(SAVE_ROOT, "Combined_Metrics")
    os.makedirs(combined_metric_dir, exist_ok=True)

    if all_dataset_metrics:
        pd.concat(all_dataset_metrics, ignore_index=True).to_csv(
            os.path.join(combined_metric_dir, "LOSO_CorrEncoder_Ablation_AllDatasets_PerSubject.csv"),
            index=False,
        )
    if all_dataset_segments:
        pd.concat(all_dataset_segments, ignore_index=True).to_csv(
            os.path.join(combined_metric_dir, "LOSO_CorrEncoder_Ablation_AllDatasets_PerSegment.csv"),
            index=False,
        )
    if all_dataset_summaries:
        combined_summary = pd.concat(all_dataset_summaries, ignore_index=True)
        combined_summary.to_csv(
            os.path.join(combined_metric_dir, "LOSO_CorrEncoder_Ablation_AllDatasets_Summary.csv"),
            index=False,
        )
        print("\n=== Combined Summary ===")
        print(combined_summary)

    print(f"\nAll Corr-Encoder ablation tasks finished: {', '.join(ACTIVE_DATASETS)}.")
    print(f"Combined metrics saved to: {combined_metric_dir}")



# ===========================================================================
# PART B. Paper-style RRWaveNet RR (261106_RRWaveNet_PaperStyle_RR)
# ===========================================================================
"""
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



matplotlib.use("Agg")



# ===========================================================
# 1. Global Config
# ===========================================================
SEED = 42
PS_GPU_NUM = 0
ps_device = torch.device(f"cuda:{PS_GPU_NUM}" if torch.cuda.is_available() else "cpu")
if ps_device.type == "cuda":
    torch.cuda.set_device(ps_device)

PS_USE_AMP = True
PS_AMP_ENABLED = PS_USE_AMP and ps_device.type == "cuda"
PS_NUM_WORKERS = 0 if os.name == "nt" else min(4, os.cpu_count() or 1)
PS_PIN_MEMORY = ps_device.type == "cuda"

WORKSPACE_ROOT = os.path.abspath(os.path.dirname(__file__))
DEFAULT_SAVE_ROOT = os.path.join(WORKSPACE_ROOT, "261106_RRWaveNet_PaperStyle_RR_Results")

# Dataset sources of the current pipeline (261105).
PS_STEAM2_PPG_DIR = r"D:\STEAM\Multimodal_prediction\PPG-RSP\Preprocessed\PPG\2back_raw_v2"
PS_STEAM2_RSP_DIR = r"D:\STEAM\Multimodal_prediction\PPG-RSP\Preprocessed\RSP\2back_raw_v2"
PS_BIDMC_DATA_PATH = (
    r"D:\STEAM\Multimodal_prediction\PPG-RSP\Preprocessed"
    r"\bidmc-ppg-and-respiration-dataset-1.0.0"
    r"\bidmc-ppg-and-respiration-dataset-1.0.0\bidmc_data.mat"
)
PS_CAPNOBASE_CSV_DIR = r"D:\STEAM\Multimodal_prediction\PPG-RSP\Opendata\Capnobase\data\csv"

PS_ACTIVE_DATASETS = ("steam2", "bidmc", "capnobase")
WINDOW_SIZES_SEC = (16.0, 32.0, 64.0)
PS_STRIDE_SEC = 2.0
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
PS_BATCH_SIZE = 64
MAX_EPOCHS = 69420
EARLY_STOP_PATIENCE = 10
LEARNING_RATE = 1.0e-4
ADABELIEF_EPS = 1.0e-13
ADABELIEF_BETAS = (0.9, 0.999)
PS_WEIGHT_DECAY = 1.0e-4
SCHEDULER_FACTOR = 0.5
PS_SCHEDULER_PATIENCE = 3
PS_MIN_LR = 5.0e-6
GRAD_CLIP_NORM = 1.0
TARGET_STD_EPS = 1.0e-6
LOSS_BETA = 0.75


PS_DATASET_CONFIGS = {
    "steam2": {
        "label": "STEAM2",
        "save_name": "STEAM2",
        "ppg_dir": PS_STEAM2_PPG_DIR,
        "rsp_dir": PS_STEAM2_RSP_DIR,
        "source_fs": None,
        # EEGLAB sessions (EEG.srate = 200 Hz); used only when a file has no srate.
        "default_fs": 200,
        "rr_band_bpm": (RR_MIN_BPM, RR_MAX_BPM),
    },
    "bidmc": {
        "label": "BIDMC",
        "save_name": "BIDMC",
        "data_path": PS_BIDMC_DATA_PATH,
        "source_fs": 125,
        "rr_band_bpm": (RR_MIN_BPM, RR_MAX_BPM),
    },
    "capnobase": {
        "label": "CapnoBase",
        "save_name": "CapnoBase",
        "data_path": PS_CAPNOBASE_CSV_DIR,
        "source_fs": None,
        # Used only when the param file has no sampling rate.
        "default_fs": 300,
        # Neonates and children breathe above 45 breaths/min (261105:
        # capnography_wide_rr, 3-60 breaths/min).
        "rr_band_bpm": (3.0, 60.0),
    },
}


def ps_set_seed(seed: int = SEED, deterministic: bool = False) -> None:
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


ps_set_seed(SEED, deterministic=False)


# ===========================================================
# 2. Paths / Small Helpers
# ===========================================================
def ps_safe_tag(text: str) -> str:
    return "".join(ch if ch.isalnum() or ch in ("-", "_", ".") else "_" for ch in str(text))


def ps_prepare_output_dirs(output_root: str, dataset_key: str, window_sec: float) -> Dict[str, str]:
    root = os.path.join(output_root, PS_DATASET_CONFIGS[dataset_key]["save_name"], f"W{int(window_sec)}")
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


def ps_autocast_context():
    if PS_AMP_ENABLED:
        return torch.autocast(device_type="cuda", dtype=torch.float16)
    return nullcontext()


def ps_count_trainable_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


# ===========================================================
# 3. Signal Utilities
# ===========================================================
def ps_sanitize_signal(sig: np.ndarray) -> np.ndarray:
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
    sig = ps_sanitize_signal(sig)
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


def ps_get_hilbert_features(sig: np.ndarray, fs: float) -> Tuple[np.ndarray, np.ndarray]:
    sig = ps_sanitize_signal(sig)
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


def ps_maybe_resample(sig: np.ndarray, orig_fs: float, target_fs: float) -> np.ndarray:
    sig = ps_sanitize_signal(sig)
    if int(round(orig_fs)) == int(round(target_fs)):
        return sig.astype(np.float32)
    target_len = int(round(len(sig) * float(target_fs) / float(orig_fs)))
    target_len = max(target_len, 1)
    return resample(sig, target_len).astype(np.float32)


def resample_signal_poly(sig: np.ndarray, src_fs: float, dst_fs: float) -> np.ndarray:
    sig = ps_sanitize_signal(sig)
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
    sig = ps_sanitize_signal(sig)
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
    sig = ps_sanitize_signal(sig)
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
    sig = ps_sanitize_signal(sig)
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
    stride_sec: float = PS_STRIDE_SEC,
    target_fs: int = TARGET_FS,
    enable_sqi: bool = ENABLE_SQI_FILTER,
    rr_band_bpm: Tuple[float, float] = (RR_MIN_BPM, RR_MAX_BPM),
) -> SubjectRRPack | None:
    ppg_raw = ps_sanitize_signal(ppg_raw)
    target_raw = ps_sanitize_signal(target_raw)
    min_len = min(len(ppg_raw), len(target_raw))
    ppg_raw = ppg_raw[:min_len]
    target_raw = target_raw[:min_len]

    valid = np.isfinite(ppg_raw) & np.isfinite(target_raw)
    ppg_raw = ppg_raw[valid]
    target_raw = target_raw[valid]
    if ppg_raw.size == 0 or target_raw.size == 0:
        return None

    ppg = ps_maybe_resample(ppg_raw, orig_fs, target_fs).astype(np.float32)
    target = ps_maybe_resample(target_raw, orig_fs, target_fs).astype(np.float32)
    min_len = min(len(ppg), len(target))
    ppg = ppg[:min_len]
    target = target[:min_len]

    ppg_low = bandpass_filter(ppg, target_fs, LOW_BAND_HZ[0], LOW_BAND_HZ[1], order=RESP_BANDPASS_ORDER)
    ppg_high = bandpass_filter(ppg, target_fs, HIGH_BAND_HZ[0], HIGH_BAND_HZ[1], order=RESP_BANDPASS_ORDER)
    env, freq = ps_get_hilbert_features(ppg_high, target_fs)
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
def ps_get_cache_path(cache_dir: str, dataset_key: str, subject: str, window_sec: float, enable_sqi: bool) -> str:
    sqi_tag = "sqi1" if enable_sqi else "sqi0"
    # The label resolution is part of the name: caches with the old 7.5
    # breaths/min labels are never read.
    tag = f"{dataset_key}_{ps_safe_tag(subject)}_w{float(window_sec):g}_{sqi_tag}_rr{RR_LABEL_RESOLUTION_BPM:g}.npz"
    return os.path.join(cache_dir, tag)


def ps_load_subject_cache(cache_path: str) -> SubjectRRPack | None:
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


def ps_save_subject_cache(cache_path: str, pack: SubjectRRPack) -> None:
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
PS_STEAM2_CHANNELS = {"PPG": ("PPG", "PLETH", "BVP"), "RSP": ("RSP", "RESP", "RESPIRATION", "BELT")}
PS_STEAM2_SESSION_PATTERN = re.compile(r"^(?P<subject>.+?)[_\- ]*session[_\- ]*(?P<session>\d+)$", re.IGNORECASE)
CAPNOBASE_FILE_SUFFIXES = (".csv", ".tab")
_NUMBER_PATTERN = re.compile(r"[-+]?(?:\d+\.?\d*|\.\d+)(?:[eE][-+]?\d+)?")


def _unwrap_mat_value(value):
    arr = np.asarray(value)
    while arr.dtype == object and arr.size == 1:
        arr = np.asarray(arr.item())
    return arr


def _ps_first_mat_payload(mat):
    valid_keys = [k for k in mat.keys() if not k.startswith("_")]
    if not valid_keys:
        raise KeyError("No valid keys found in MAT file.")
    payload = _unwrap_mat_value(mat[valid_keys[0]])
    if payload.ndim >= 2 and payload.shape[0] == 1 and payload.shape[1] == 1:
        payload = payload[0, 0]
    return payload


def _ps_extract_signal_and_fs(payload, default_fs: float):
    signal = ps_sanitize_signal(_unwrap_mat_value(payload["data"]))
    try:
        srate = float(np.asarray(_unwrap_mat_value(payload["srate"])).squeeze())
    except Exception:
        srate = float(default_fs)
    if not np.isfinite(srate) or srate <= 0:
        srate = float(default_fs)
    return signal, srate


def _ps_as_list(value) -> list:
    if value is None:
        return []
    if isinstance(value, np.ndarray):
        return list(value.ravel()) if value.size else []
    return [value]


def ps_steam2_session_key(stem: str) -> Tuple[str, int]:
    """(subject, session number) from a file stem such as 13_LSB_Session2."""
    match = PS_STEAM2_SESSION_PATTERN.match(stem)
    if match is None:
        return stem, 0
    return match.group("subject"), int(match.group("session"))


def ps_read_eeglab_channel(path: str, kind: str, default_fs: float) -> dict:
    """One channel of an EEGLAB .mat file, chosen by its label; a one-channel
    file without a label is accepted, one labelled as the other signal is
    rejected (swapped files). The positions of "boundary" events inside the
    recording are returned so that the session is split there."""
    mat = sio.loadmat(path, squeeze_me=True, struct_as_record=False)
    names = [key for key in mat if not key.startswith("_")]
    payload = mat["EEG"] if "EEG" in mat else (mat[names[0]] if names else None)
    if payload is None or not hasattr(payload, "data"):
        signal, fs = _ps_extract_signal_and_fs(_ps_first_mat_payload(sio.loadmat(path)), default_fs)
        return {"signal": signal, "fs": fs, "boundaries": [], "label": ""}
    data = np.asarray(payload.data, dtype=np.float64)
    data = data.reshape(1, -1) if data.ndim == 1 else data
    labels = [str(getattr(item, "labels", "")).strip() for item in _ps_as_list(getattr(payload, "chanlocs", None))]
    wanted = {name.upper() for name in PS_STEAM2_CHANNELS[kind]}
    other = {name.upper() for key, names in PS_STEAM2_CHANNELS.items() if key != kind for name in names}
    index = next((i for i, label in enumerate(labels) if label.upper() in wanted), None)
    if index is None:
        if data.shape[0] != 1:
            raise RuntimeError(f"{os.path.basename(path)}: no {kind} channel among {labels}")
        if labels and labels[0].upper() in other:
            raise RuntimeError(f"{os.path.basename(path)}: channel '{labels[0]}' found where {kind} was expected")
        index = 0
    signal = ps_sanitize_signal(data[index])
    try:
        fs = float(np.asarray(payload.srate).squeeze())
    except Exception:
        fs = float(default_fs)
    if not np.isfinite(fs) or fs <= 0:
        fs = float(default_fs)
    boundaries = []
    for event in _ps_as_list(getattr(payload, "event", None)):
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


def ps_steam2_session_files(ppg_root: str, rsp_root: str) -> Dict[str, List[Tuple[str, str, str]]]:
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
            subject, _ = ps_steam2_session_key(ppg_path.stem)
            sessions.setdefault(subject, []).append((ppg_path.stem, str(ppg_path), str(rsp_files[ppg_path.stem])))
    for subject in sessions:
        sessions[subject].sort(key=lambda item: (ps_steam2_session_key(item[0])[1], item[0]))
    return sessions


def ps_steam2_session_pieces(ppg: dict, rsp: dict) -> Tuple[List[Tuple[np.ndarray, np.ndarray, float, float]], float]:
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


def ps_load_bidmc_dataset(window_sec: float, dirs: Dict[str, str], enable_sqi: bool = ENABLE_SQI_FILTER) -> Dict[str, SubjectRRPack]:
    subject_data: Dict[str, SubjectRRPack] = {}
    mat = sio.loadmat(PS_DATASET_CONFIGS["bidmc"]["data_path"])
    all_data = mat["data"][0]
    for idx in range(len(all_data)):
        subject = f"{idx + 1:02d}"
        cache_path = ps_get_cache_path(dirs["cache"], "bidmc", subject, window_sec, enable_sqi=enable_sqi)
        cached = ps_load_subject_cache(cache_path)
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
                orig_fs=float(PS_DATASET_CONFIGS["bidmc"]["source_fs"]),
                window_sec=window_sec,
                enable_sqi=enable_sqi,
                rr_band_bpm=PS_DATASET_CONFIGS["bidmc"]["rr_band_bpm"],
            )
            if pack is not None:
                subject_data[subject] = pack
                ps_save_subject_cache(cache_path, pack)
                print(f"Loaded BIDMC {subject}: {len(pack.x_low)} windows")
        except Exception as exc:
            print(f"   [Load Fail] BIDMC {subject}: {exc}")
    return subject_data


def ps_load_capnobase_dataset(window_sec: float, dirs: Dict[str, str], enable_sqi: bool = ENABLE_SQI_FILTER) -> Dict[str, SubjectRRPack]:
    subject_data: Dict[str, SubjectRRPack] = {}
    cfg = PS_DATASET_CONFIGS["capnobase"]
    csv_dir = cfg["data_path"]
    case_ids = capnobase_case_ids(csv_dir)
    for case_id in case_ids:
        cache_path = ps_get_cache_path(dirs["cache"], "capnobase", case_id, window_sec, enable_sqi=enable_sqi)
        cached = ps_load_subject_cache(cache_path)
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
                ps_save_subject_cache(cache_path, pack)
                print(f"Loaded CapnoBase {case_id}: {len(pack.x_low)} windows")
        except Exception as exc:
            print(f"   [Load Fail] CapnoBase {case_id}: {exc}")
    return subject_data


def ps_load_steam2_dataset(window_sec: float, dirs: Dict[str, str], enable_sqi: bool = ENABLE_SQI_FILTER) -> Dict[str, SubjectRRPack]:
    """Every piece of every session (sessions split at boundary events) is
    resampled to TARGET_FS and windowed on its own; the windows of a subject's
    pieces are stacked in session order."""
    subject_data: Dict[str, SubjectRRPack] = {}
    cfg = PS_DATASET_CONFIGS["steam2"]
    if not os.path.isdir(cfg["ppg_dir"]) or not os.path.isdir(cfg["rsp_dir"]):
        raise FileNotFoundError(f"STEAM2 folders not found: {cfg['ppg_dir']} / {cfg['rsp_dir']}")
    for subject, items in ps_steam2_session_files(cfg["ppg_dir"], cfg["rsp_dir"]).items():
        cache_path = ps_get_cache_path(dirs["cache"], "steam2", subject, window_sec, enable_sqi=enable_sqi)
        cached = ps_load_subject_cache(cache_path)
        if cached is not None:
            subject_data[subject] = cached
            print(f"Loaded STEAM2 cache {subject}: {len(cached.x_low)} windows")
            continue

        x_low_parts: List[np.ndarray] = []
        x_high_parts: List[np.ndarray] = []
        y_parts: List[np.ndarray] = []
        for name, ppg_path, rsp_path in items:
            try:
                ppg = ps_read_eeglab_channel(ppg_path, "PPG", cfg["default_fs"])
                rsp = ps_read_eeglab_channel(rsp_path, "RSP", cfg["default_fs"])
                pieces, length_difference = ps_steam2_session_pieces(ppg, rsp)
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
            ps_save_subject_cache(cache_path, merged)
            print(f"Loaded STEAM2 {subject}: {len(merged.x_low)} windows from {len(items)} sessions")
        else:
            print(f"   [Load Fail] STEAM2 {subject}: no usable window")
    return subject_data


def ps_load_dataset(
    dataset_key: str,
    window_sec: float,
    dirs: Dict[str, str],
    enable_sqi: bool = ENABLE_SQI_FILTER,
) -> Dict[str, SubjectRRPack]:
    if dataset_key == "steam2":
        return ps_load_steam2_dataset(window_sec, dirs, enable_sqi=enable_sqi)
    if dataset_key == "bidmc":
        return ps_load_bidmc_dataset(window_sec, dirs, enable_sqi=enable_sqi)
    if dataset_key == "capnobase":
        return ps_load_capnobase_dataset(window_sec, dirs, enable_sqi=enable_sqi)
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
        num_workers=PS_NUM_WORKERS,
        pin_memory=PS_PIN_MEMORY,
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
        weight_decay: float = PS_WEIGHT_DECAY,
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
            x_low = x_low.to(ps_device, non_blocking=True)
            x_high = x_high.to(ps_device, non_blocking=True)
            y = y.to(ps_device, non_blocking=True)
            with ps_autocast_context():
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


def ps_save_learning_curve(train_losses: List[float], val_losses: List[float], save_path: str, title: str) -> None:
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
    batch_size: int = PS_BATCH_SIZE,
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

    model = build_model(case_key).to(ps_device)
    trainable_params = ps_count_trainable_params(model)
    criterion = nn.SmoothL1Loss(beta=LOSS_BETA)
    optimizer = AdaBelief(
        model.parameters(),
        lr=LEARNING_RATE,
        betas=ADABELIEF_BETAS,
        eps=ADABELIEF_EPS,
        weight_decay=PS_WEIGHT_DECAY,
    )
    scheduler = ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=SCHEDULER_FACTOR,
        patience=PS_SCHEDULER_PATIENCE,
        min_lr=PS_MIN_LR,
    )

    best_val_loss = np.inf
    best_val_mae = np.inf
    best_epoch = 0
    patience_counter = 0
    train_losses: List[float] = []
    val_losses: List[float] = []
    model_path = os.path.join(
        dirs["models"],
        f"{ps_safe_tag(dataset_key)}_{ps_safe_tag(case_key)}_W{int(window_sec)}_{ps_safe_tag(test_subject)}_fold{fold_idx + 1}_best.pth",
    )

    for epoch in range(int(max_epochs)):
        model.train()
        running_loss = 0.0
        n_batches = 0
        for x_low, x_high, y in train_loader:
            x_low = x_low.to(ps_device, non_blocking=True)
            x_high = x_high.to(ps_device, non_blocking=True)
            y = y.to(ps_device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with ps_autocast_context():
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
            f"[{PS_DATASET_CONFIGS[dataset_key]['label']}] W{int(window_sec)} | "
            f"{INPUT_CASES[case_key]['label']} | Test {test_subject} | Fold {fold_idx + 1}/{INNER_FOLDS} | "
            f"Ep {epoch + 1:05d} | TrainLoss {train_loss:.5f} | ValLoss {val_loss:.5f} | "
            f"ValMAE {val_mae:.5f} | LR {optimizer.param_groups[0]['lr']:.6f}"
        )

        if patience_counter >= EARLY_STOP_PATIENCE:
            print(f"Early stop at epoch {epoch + 1} for {test_subject} fold {fold_idx + 1}")
            break

    if os.path.exists(model_path):
        checkpoint = torch.load(model_path, map_location=ps_device)
        if isinstance(checkpoint, dict) and "model_state" in checkpoint:
            model.load_state_dict(checkpoint["model_state"])
        else:
            model.load_state_dict(checkpoint)

    test_loss, test_mae, preds, trues = evaluate_model(model, test_loader, criterion, target_scaler)
    curve_path = os.path.join(
        dirs["curves"],
        f"{ps_safe_tag(dataset_key)}_{ps_safe_tag(case_key)}_W{int(window_sec)}_{ps_safe_tag(test_subject)}_fold{fold_idx + 1}.png",
    )
    ps_save_learning_curve(
        train_losses=train_losses,
        val_losses=val_losses,
        save_path=curve_path,
        title=(
            f"{PS_DATASET_CONFIGS[dataset_key]['label']} | {INPUT_CASES[case_key]['label']} | "
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
    batch_size: int = PS_BATCH_SIZE,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    dirs = ps_prepare_output_dirs(output_root, dataset_key, window_sec)
    print(f"\n{'=' * 88}")
    print(f"Dataset: {PS_DATASET_CONFIGS[dataset_key]['label']} | Window: {int(window_sec)} s")
    print(f"Output root: {dirs['root']}")
    print(f"{'=' * 88}")

    subject_data = ps_load_dataset(dataset_key, window_sec, dirs, enable_sqi=True)
    subjects = sorted(subject_data.keys())
    if len(subjects) < 3:
        print(
            f"[Warn] {PS_DATASET_CONFIGS[dataset_key]['label']} W{int(window_sec)}: "
            f"only {len(subjects)} valid subjects with SQI enabled. Retrying with SQI disabled."
        )
        subject_data = ps_load_dataset(dataset_key, window_sec, dirs, enable_sqi=False)
        subjects = sorted(subject_data.keys())
    if len(subjects) < 3:
        print(
            f"[Skip] {PS_DATASET_CONFIGS[dataset_key]['label']} W{int(window_sec)}: "
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
                        "Dataset": PS_DATASET_CONFIGS[dataset_key]["label"],
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
                            "Dataset": PS_DATASET_CONFIGS[dataset_key]["label"],
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
# 11. CLI / Main
# ===========================================================
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Paper-style RRWaveNet direct RR regression with input-case comparison")
    parser.add_argument("--datasets", nargs="+", default=list(PS_ACTIVE_DATASETS), choices=list(PS_DATASET_CONFIGS.keys()))
    parser.add_argument("--window-sizes", nargs="+", type=float, default=list(WINDOW_SIZES_SEC))
    parser.add_argument("--output-root", type=str, default=DEFAULT_SAVE_ROOT)
    parser.add_argument("--max-epochs", type=int, default=MAX_EPOCHS)
    parser.add_argument("--batch-size", type=int, default=PS_BATCH_SIZE)
    return parser.parse_args()


def ps_main(args: argparse.Namespace | None = None) -> None:
    args = parse_args() if args is None else args
    print(f"Device: {ps_device}")
    print(f"AMP enabled: {PS_AMP_ENABLED}")
    print(f"Datasets: {args.datasets}")
    print(f"Window sizes: {args.window_sizes}")
    print(f"Input cases: {list(ACTIVE_INPUT_CASES)}")
    print(f"Output root: {args.output_root}")

    combined_summary = []
    combined_subject = []
    combined_fold = []

    for dataset_key in args.datasets:
        for window_sec in args.window_sizes:
            summary_df, subject_df, fold_df = run_dataset_experiment(
                dataset_key=dataset_key,
                window_sec=float(window_sec),
                output_root=args.output_root,
                max_epochs=args.max_epochs,
                batch_size=args.batch_size,
            )
            if not summary_df.empty:
                combined_summary.append(summary_df)
            if not subject_df.empty:
                combined_subject.append(subject_df)
            if not fold_df.empty:
                combined_fold.append(fold_df)

    output_root = os.path.abspath(args.output_root)
    os.makedirs(output_root, exist_ok=True)

    if combined_summary:
        pd.concat(combined_summary, ignore_index=True).to_csv(
            os.path.join(output_root, "RRWaveNet_PaperStyle_GlobalSummary.csv"),
            index=False,
            encoding="utf-8-sig",
        )
    if combined_subject:
        pd.concat(combined_subject, ignore_index=True).to_csv(
            os.path.join(output_root, "RRWaveNet_PaperStyle_GlobalPerSubject.csv"),
            index=False,
            encoding="utf-8-sig",
        )
    if combined_fold:
        pd.concat(combined_fold, ignore_index=True).to_csv(
            os.path.join(output_root, "RRWaveNet_PaperStyle_GlobalPerFold.csv"),
            index=False,
            encoding="utf-8-sig",
        )

    print("\nAll RRWaveNet paper-style RR experiments finished.")



# ===========================================================
# Runner: part A, then part B, each with its own results folder
# ===========================================================
OUTPUT_ROOT = r"D:\PPG2RSP_RRWaveNet_Inspired\261106_DeepCorrencoder_PaperStyle_All"
CE_FOLDER = "Deep_Correncoder"
PS_FOLDER = "RRWaveNet_PaperStyle_RR"
PARTS = ("deep_correncoder", "paperstyle")


def parse_run_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="261106 deep corr-encoder and paper-style RRWaveNet, one run")
    parser.add_argument("--parts", nargs="+", choices=PARTS, default=list(PARTS))
    parser.add_argument("--output-root", type=str, default=OUTPUT_ROOT)
    parser.add_argument("--ce-datasets", nargs="+", choices=list(DATASET_CONFIGS), default=list(ACTIVE_DATASETS))
    parser.add_argument("--ps-datasets", nargs="+", choices=list(PS_DATASET_CONFIGS), default=list(PS_ACTIVE_DATASETS))
    parser.add_argument("--window-sizes", nargs="+", type=float, default=list(WINDOW_SIZES_SEC))
    parser.add_argument("--max-epochs", type=int, default=MAX_EPOCHS)
    parser.add_argument("--batch-size", type=int, default=PS_BATCH_SIZE)
    return parser.parse_args()


def run_all() -> None:
    global SAVE_ROOT, ACTIVE_DATASETS
    args = parse_run_args()
    output_root = os.path.abspath(args.output_root)
    status = {}

    if "deep_correncoder" in args.parts:
        print(f"\n{'#' * 88}\n# A. Deep corr-encoder (0.1-0.5 Hz input): {', '.join(args.ce_datasets)}\n{'#' * 88}")
        SAVE_ROOT = os.path.join(output_root, CE_FOLDER)
        ACTIVE_DATASETS = list(args.ce_datasets)
        started = time.time()
        try:
            set_seed()
            ce_main()
            status["A. Deep corr-encoder"] = f"finished ({(time.time() - started) / 3600:.2f} h) -> {SAVE_ROOT}"
        except Exception:
            traceback.print_exc()
            status["A. Deep corr-encoder"] = "FAILED (see the error above)"

    if "paperstyle" in args.parts:
        print(f"\n{'#' * 88}\n# B. Paper-style RRWaveNet RR (0.1-0.5 Hz input): {', '.join(args.ps_datasets)}, "
              f"W {', '.join(f'{w:g}' for w in args.window_sizes)} s\n{'#' * 88}")
        ps_args = argparse.Namespace(
            datasets=list(args.ps_datasets),
            window_sizes=[float(w) for w in args.window_sizes],
            output_root=os.path.join(output_root, PS_FOLDER),
            max_epochs=int(args.max_epochs),
            batch_size=int(args.batch_size),
        )
        started = time.time()
        try:
            ps_set_seed(SEED, deterministic=False)
            ps_main(ps_args)
            status["B. Paper-style RRWaveNet RR"] = (f"finished ({(time.time() - started) / 3600:.2f} h) -> "
                                                    f"{ps_args.output_root}")
        except Exception:
            traceback.print_exc()
            status["B. Paper-style RRWaveNet RR"] = "FAILED (see the error above)"

    print(f"\n{'=' * 88}")
    for part, text in status.items():
        print(f"{part}: {text}")


if __name__ == "__main__":
    run_all()
