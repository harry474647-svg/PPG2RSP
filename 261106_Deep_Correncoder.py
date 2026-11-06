# -*- coding: utf-8 -*-
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

import os
import re
import time
import random
from collections import Counter
from contextlib import nullcontext
from itertools import combinations
from math import gcd
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.io as sio
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.signal import butter, sosfiltfilt, correlate, find_peaks, hilbert, resample_poly
from scipy.stats import pearsonr, skew

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader, TensorDataset, WeightedRandomSampler


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


def main():
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


if __name__ == "__main__":
    main()
