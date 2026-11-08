# -*- coding: utf-8 -*-
"""
RespDiff waveform model: CapnoBase only, Baseline only
(from 260519_RespDiff_AllDatasets_v3; model, preprocessing, losses, training,
sampling, evaluation and output layout unchanged).

What this script does
1. Trains the diffusion waveform model (LOSO) on CapnoBase only (CSV folder),
   with the Baseline input setting only (low-band PPG). The Proposed setting
   (low/high/envelope/instantaneous-frequency) is not trained.
2. Keeps what is already in the results folder:
   - the results go where v3 put them: {save-root}\CapnoBase\{run tag}\
     (Models, Learning_Curves, Waveform_Comparison, Metrics, Predictions);
     the run tag comes from the speed settings, so run with the settings of the
     earlier run (default --speed-preset balanced) to continue that folder
   - a test subject whose Baseline result is already stored (RespDiff_Final_* /
     RespDiff_Running_* PerSubject) is skipped; its stored rows (Baseline and,
     from the earlier v3 run, Proposed), predictions and models stay as they are
   - only the subjects without a stored Baseline result are trained; a subject
     interrupted mid-training (no stored result) is trained from the start
   - --force-subjects retrains the Baseline of the named subjects only; their
     stored Proposed rows and Proposed predictions are kept
   - subject IDs match with or without leading zeros (0009 = 9)
   - _Combined\{run tag}: the CapnoBase rows are replaced by the updated ones,
     the rows of the other datasets (STEAM, BIDMC) are kept
3. At the start, every run folder already under {save-root}\CapnoBase is listed
   with its number of subjects with a Baseline result.
4. Learning curves (total loss and each loss component), the 30 s waveform
   plot (GT vs Baseline) and the prediction bundle are saved per subject.

Run:
  python 261108_RespDiff_CapnoBase_Baseline_v3.py
Options: --speed-preset (default balanced), --run-tag, --epochs, --sample-steps,
  --batch-size, --val-every, --train/val/test-window-step,
  --max-train/val/test-windows-per-subject, --subjects, --force-subjects,
  --save-root
"""

import argparse
import gc
import json
import os
import random
import time
from contextlib import nullcontext
from math import gcd

import numpy as np
import pandas as pd
import scipy.io as sio
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.signal import (
    butter,
    correlate,
    find_peaks,
    hilbert,
    resample,
    resample_poly,
    savgol_filter,
    sosfiltfilt,
)
from scipy.stats import pearsonr, skew
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader, Dataset


# ===========================================================
# 1. Environment / Global Config
# ===========================================================
GPU_NUM = 0
device = torch.device(f"cuda:{GPU_NUM}" if torch.cuda.is_available() else "cpu")
if device.type == "cuda":
    torch.cuda.set_device(device)


def set_seed(seed=777, deterministic=False):
    print(f"Seed: {seed}")
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


set_seed(deterministic=False)


BASE_SAVE_ROOT = (
    r"D:\STEAM\Multimodal_prediction\PPG-RSP\Results\Waveform"
    r"\Hilbert_OpenDataset\RespDiff_AllDatasets_v3"
)

STEAM_PPG_DIR = r"D:\STEAM\Multimodal_prediction\PPG-RSP\Preprocessed\PPG\2back_raw_v2"
STEAM_RSP_DIR = r"D:\STEAM\Multimodal_prediction\PPG-RSP\Preprocessed\RSP\2back_raw_v2"
BIDMC_DATA_PATH = (
    r"D:\STEAM\Multimodal_prediction\PPG-RSP\Preprocessed"
    r"\bidmc-ppg-and-respiration-dataset-1.0.0"
    r"\bidmc-ppg-and-respiration-dataset-1.0.0\bidmc_data.mat"
)
CAPNOBASE_CSV_DIR = r"D:\STEAM\Multimodal_prediction\PPG-RSP\Opendata\Capnobase\data\csv"

DEFAULT_ACTIVE_DATASETS = ["capnobase"]  # this version: CapnoBase only
ACTIVE_MODES = ["baseline"]  # this version: Baseline only (Proposed is not trained)
LOSS_KEYS = ["total", "diff", "spec", "wave", "corr", "grad", "smooth"]

WINDOW_SEC = 32.0
STRIDE_SEC = 16.0
PLOT_SEC = 30.0
DIFF_TIMESTEPS = 1000
SAMPLE_STEPS = 200
EPOCHS = 150
BATCH_SIZE = 32
MIN_EPOCHS = 25
PATIENCE = 25
VAL_RATIO = 0.25
MIN_VAL_SUBJECTS = 2
EARLY_STOP_MIN_DELTA = 1e-4

USE_AMP = True
AMP_ENABLED = USE_AMP and device.type == "cuda"
NUM_WORKERS = 0 if os.name == "nt" else min(4, os.cpu_count() or 1)
PIN_MEMORY = device.type == "cuda"

AUTO_RESUME = True
RUNNING_PREFIX = "RespDiff_Running"
FINAL_PREFIX = "RespDiff_Final"

SMOOTH_PRED_SEC = 0.20
SMOOTH_TRUE_SEC = 0.10

SPEED_PRESETS = {
    "full": {
        "epochs": None,
        "sample_steps": None,
        "batch_size": None,
        "val_every": 1,
        "train_window_step": 1,
        "val_window_step": 1,
        "test_window_step": 1,
        "max_train_windows_per_subject": None,
        "max_val_windows_per_subject": None,
        "max_test_windows_per_subject": None,
    },
    "balanced": {
        "epochs": 90,
        "sample_steps": 80,
        "batch_size": None,
        "val_every": 2,
        "train_window_step": 2,
        "val_window_step": 2,
        "test_window_step": 1,
        "max_train_windows_per_subject": 240,
        "max_val_windows_per_subject": 160,
        "max_test_windows_per_subject": None,
    },
    "turbo": {
        "epochs": 60,
        "sample_steps": 40,
        "batch_size": None,
        "val_every": 3,
        "train_window_step": 3,
        "val_window_step": 3,
        "test_window_step": 2,
        "max_train_windows_per_subject": 160,
        "max_val_windows_per_subject": 96,
        "max_test_windows_per_subject": 128,
    },
}

DATASET_CONFIGS = {
    "steam": {
        "label": "STEAM",
        "save_name": "STEAM",
        "ppg_dir": STEAM_PPG_DIR,
        "rsp_dir": STEAM_RSP_DIR,
        "target_fs": 200,
        "window_sec": WINDOW_SEC,
        "stride_sec": STRIDE_SEC,
        "plot_target_label": "Respiration",
    },
    "bidmc": {
        "label": "BIDMC",
        "save_name": "BIDMC",
        "data_path": BIDMC_DATA_PATH,
        "target_fs": 125,
        "window_sec": WINDOW_SEC,
        "stride_sec": STRIDE_SEC,
        "plot_target_label": "Respiration",
    },
    "capnobase": {
        "label": "CapnoBase",
        "save_name": "CapnoBase",
        "data_path": CAPNOBASE_CSV_DIR,
        "target_fs": 125,
        "window_sec": WINDOW_SEC,
        "stride_sec": STRIDE_SEC,
        "plot_target_label": "CO2-derived Respiration",
    },
}

MODE_CONFIGS = {
    "baseline": {
        "label": "Baseline",
        "input_features": ["low"],
        "base_dim": 32,
        "rnn_hidden": 128,
        "dropout": 0.05,
        "feature_dropout": 0.0,
        "lr": 2e-4,
        "weight_decay": 1e-3,
        "lambda_spec": 0.05,
        "lambda_wave": 0.10,
        "lambda_corr": 0.05,
        "lambda_grad": 0.04,
        "lambda_smooth": 0.02,
        "epochs": EPOCHS,
        "batch_size": BATCH_SIZE,
        "min_epochs": MIN_EPOCHS,
        "patience": PATIENCE,
        "sample_steps": SAMPLE_STEPS,
        "color": "tab:blue",
    },
    "proposed": {
        "label": "Proposed",
        "input_features": ["low", "high", "env", "freq"],
        "base_dim": 24,
        "rnn_hidden": 96,
        "dropout": 0.20,
        "feature_dropout": 0.25,
        "lr": 1e-4,
        "weight_decay": 2e-3,
        "lambda_spec": 0.05,
        "lambda_wave": 0.15,
        "lambda_corr": 0.08,
        "lambda_grad": 0.06,
        "lambda_smooth": 0.03,
        "epochs": EPOCHS,
        "batch_size": BATCH_SIZE,
        "min_epochs": 30,
        "patience": 20,
        "sample_steps": SAMPLE_STEPS,
        "color": "tab:red",
    },
}


def safe_tag(text):
    return "".join(c if c.isalnum() or c in "-_." else "_" for c in str(text))


def build_run_tag(args):
    if getattr(args, "run_tag", None):
        return safe_tag(args.run_tag)
    parts = [f"preset_{safe_tag(args.speed_preset)}"]
    if args.epochs is not None:
        parts.append(f"e{int(args.epochs)}")
    if args.sample_steps is not None:
        parts.append(f"ss{int(args.sample_steps)}")
    parts.append(f"ve{int(args.val_every)}")
    parts.append(f"tr{int(args.train_window_step)}")
    parts.append(f"va{int(args.val_window_step)}")
    parts.append(f"te{int(args.test_window_step)}")
    if args.max_train_windows_per_subject is not None:
        parts.append(f"mtr{int(args.max_train_windows_per_subject)}")
    if args.max_val_windows_per_subject is not None:
        parts.append(f"mva{int(args.max_val_windows_per_subject)}")
    if args.max_test_windows_per_subject is not None:
        parts.append(f"mte{int(args.max_test_windows_per_subject)}")
    if args.batch_size is not None:
        parts.append(f"bs{int(args.batch_size)}")
    return "_".join(parts)


def apply_speed_preset(args):
    preset = SPEED_PRESETS[args.speed_preset]
    for key, value in preset.items():
        if getattr(args, key) is None:
            setattr(args, key, value)
    if args.val_every is None:
        args.val_every = 1
    if args.train_window_step is None:
        args.train_window_step = 1
    if args.val_window_step is None:
        args.val_window_step = 1
    if args.test_window_step is None:
        args.test_window_step = 1
    return args


def get_mode_config(mode_key, args):
    base = MODE_CONFIGS[mode_key]
    cfg = {k: (list(v) if isinstance(v, list) else v) for k, v in base.items()}
    if args.epochs is not None:
        cfg["epochs"] = int(args.epochs)
        cfg["min_epochs"] = min(int(cfg["min_epochs"]), int(cfg["epochs"]))
        cfg["patience"] = min(int(cfg["patience"]), max(1, int(cfg["epochs"]) - int(cfg["min_epochs"]) + 1))
    if args.sample_steps is not None:
        cfg["sample_steps"] = int(args.sample_steps)
    if args.batch_size is not None:
        cfg["batch_size"] = int(args.batch_size)
    cfg["val_every"] = int(args.val_every)
    cfg["train_window_step"] = int(args.train_window_step)
    cfg["val_window_step"] = int(args.val_window_step)
    cfg["test_window_step"] = int(args.test_window_step)
    cfg["max_train_windows_per_subject"] = args.max_train_windows_per_subject
    cfg["max_val_windows_per_subject"] = args.max_val_windows_per_subject
    cfg["max_test_windows_per_subject"] = args.max_test_windows_per_subject
    cfg["early_stop_patience_evals"] = max(1, int(np.ceil(float(cfg["patience"]) / float(cfg["val_every"]))))
    return cfg


def prepare_output_dirs(save_root, dataset_key, args):
    dataset_base = os.path.join(save_root, DATASET_CONFIGS[dataset_key]["save_name"])
    run_root = os.path.join(dataset_base, build_run_tag(args))
    dirs = {
        "dataset_base": dataset_base,
        "root": run_root,
        "models": os.path.join(run_root, "Models"),
        "curves": os.path.join(run_root, "Learning_Curves"),
        "plots": os.path.join(run_root, "Waveform_Comparison"),
        "metrics": os.path.join(run_root, "Metrics"),
        "predictions": os.path.join(run_root, "Predictions"),
        "cache": os.path.join(dataset_base, "Processed_Subject_Cache"),
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


def clear_runtime_memory():
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()


def count_trainable_params(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def get_pyplot():
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        return plt
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "matplotlib is required to save learning curves and waveform plots. "
            "Please install matplotlib in the Python environment used to run this script."
        ) from exc


# ===========================================================
# 2. Signal / IO Utilities
# ===========================================================
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


def bandpass_filter(sig, fs, low=0.1, high=0.5, order=3):
    sig = sanitize_signal(sig)
    if sig.size < max(32, order * 6 + 1):
        return sig.astype(np.float32)
    nyq = 0.5 * float(fs)
    low = max(float(low), 1e-5)
    high = min(float(high), nyq * 0.99)
    if not np.isfinite(low) or not np.isfinite(high) or high <= low:
        return sig.astype(np.float32)
    try:
        sos = butter(order, [low / nyq, high / nyq], btype="band", output="sos")
        return sosfiltfilt(sos, sig).astype(np.float32)
    except Exception:
        return sig.astype(np.float32)


def moving_average_same(x, kernel_size):
    x = sanitize_signal(x)
    kernel_size = int(max(1, kernel_size))
    if kernel_size <= 1 or x.size == 0:
        return x.astype(np.float32)
    kernel = np.ones(kernel_size, dtype=np.float64) / float(kernel_size)
    return np.convolve(x, kernel, mode="same").astype(np.float32)


def valid_odd_window(length, desired):
    if length < 5:
        return None
    win = int(max(5, desired))
    if win % 2 == 0:
        win += 1
    max_win = length if length % 2 == 1 else length - 1
    win = min(win, max_win)
    if win < 5:
        return None
    return win


def smooth_signal(sig, fs, smooth_sec=0.20, polyorder=3):
    sig = sanitize_signal(sig).astype(np.float32)
    if sig.size < 5:
        return sig
    desired = int(round(float(smooth_sec) * float(fs)))
    win = valid_odd_window(sig.size, desired)
    if win is None:
        return sig
    use_poly = min(int(polyorder), win - 1)
    if use_poly < 1:
        return sig
    try:
        return savgol_filter(sig, window_length=win, polyorder=use_poly, mode="interp").astype(np.float32)
    except Exception:
        return sig


def normalize_for_plot(sig):
    sig = sanitize_signal(sig)
    return (sig - np.mean(sig)) / (np.std(sig) + 1e-8)


def safe_pearsonr(x, y):
    x = sanitize_signal(x)
    y = sanitize_signal(y)
    m = min(len(x), len(y))
    x = x[:m]
    y = y[:m]
    if len(x) < 2 or np.std(x) < 1e-8 or np.std(y) < 1e-8:
        return np.nan
    return pearsonr(x, y)[0]


def waveform_mae(x, y):
    x = np.asarray(x)
    y = np.asarray(y)
    m = min(len(x), len(y))
    return float(np.mean(np.abs(x[:m] - y[:m])))


def correct_polarity_skew(sig):
    sig = sanitize_signal(sig)
    if sig.size < 3:
        return sig.astype(np.float32)
    return (-sig if skew(sig) < 0 else sig).astype(np.float32)


def align_phase_only(ppg_ref, rsp_sig):
    m = min(len(ppg_ref), len(rsp_sig))
    ppg_ref = sanitize_signal(ppg_ref)[:m]
    rsp_sig = sanitize_signal(rsp_sig)[:m]
    corr = safe_pearsonr(ppg_ref, rsp_sig)
    return (-rsp_sig if np.isfinite(corr) and corr < 0 else rsp_sig).astype(np.float32)


def get_hilbert_features(sig, fs):
    sig = sanitize_signal(sig)
    analytic_signal = hilbert(sig)
    envelope = np.abs(analytic_signal)
    phase = np.unwrap(np.angle(analytic_signal))
    inst_freq = np.diff(phase) / (2.0 * np.pi) * fs
    if len(inst_freq) == 0:
        inst_freq = np.zeros_like(sig)
    else:
        inst_freq = np.insert(inst_freq, 0, inst_freq[0])
    envelope = np.nan_to_num(envelope, nan=0.0, posinf=0.0, neginf=0.0)
    inst_freq = np.nan_to_num(inst_freq, nan=0.0, posinf=0.0, neginf=0.0)
    if inst_freq.size > 0:
        lo, hi = np.percentile(inst_freq, [1.0, 99.0])
        if np.isfinite(lo) and np.isfinite(hi) and hi > lo:
            inst_freq = np.clip(inst_freq, lo, hi)
        inst_freq = moving_average_same(inst_freq, int(max(1, 0.25 * fs)))
    return envelope.astype(np.float32), inst_freq.astype(np.float32)


def extract_windows(sig, fs, win_sec, stride_sec):
    win_len = int(round(win_sec * fs))
    stride_len = int(round(stride_sec * fs))
    sig = np.asarray(sig)
    total_len = sig.shape[-1]
    is_multichannel = sig.ndim > 1
    if total_len < win_len:
        return np.array([])
    windows = []
    for st in range(0, total_len - win_len + 1, stride_len):
        ed = st + win_len
        windows.append(sig[:, st:ed] if is_multichannel else sig[st:ed])
    return np.asarray(windows)


def instance_normalize(data):
    data = np.asarray(data, dtype=np.float32)
    mean_val = np.mean(data, axis=-1, keepdims=True)
    std_val = np.std(data, axis=-1, keepdims=True)
    std_val[std_val < 1e-6] = 1.0
    return ((data - mean_val) / std_val).astype(np.float32)


def maybe_resample(sig, orig_fs, target_fs):
    sig = sanitize_signal(sig).astype(np.float32)
    if int(round(orig_fs)) == int(round(target_fs)):
        return sig
    target_len = int(round(len(sig) * float(target_fs) / float(orig_fs)))
    target_len = max(target_len, 1)
    return resample(sig, target_len).astype(np.float32)


def resample_signal_poly(sig, src_fs, dst_fs):
    sig = sanitize_signal(sig)
    src_fs = int(round(float(src_fs)))
    dst_fs = int(round(float(dst_fs)))
    if src_fs <= 0 or dst_fs <= 0:
        raise ValueError(f"Invalid sampling rate: src={src_fs}, dst={dst_fs}")
    if sig.size == 0 or src_fs == dst_fs:
        return sig.astype(np.float32)
    factor = gcd(src_fs, dst_fs)
    up = dst_fs // factor
    down = src_fs // factor
    return resample_poly(sig, up, down).astype(np.float32)


def check_quality_real_world(x_low, y_true, fs):
    x_low = sanitize_signal(x_low)
    y_true = sanitize_signal(y_true)
    centered = y_true - np.mean(y_true)
    if np.std(centered) < 1e-4:
        return False
    acor = correlate(centered, centered, mode="full")
    acor = acor[len(acor) // 2 :]
    acor = acor / (acor[0] + 1e-8)
    min_peak = int(1.5 * fs)
    max_peak = int(10.0 * fs)
    peaks, _ = find_peaks(acor, height=0.20)
    if not any(min_peak < p < max_peak for p in peaks):
        return False
    corr = safe_pearsonr(x_low, y_true)
    return bool(np.isfinite(corr) and corr >= 0.05)


def postprocess_prediction_window(sig, fs):
    sig = bandpass_filter(sig, fs, 0.08, 0.70)
    sig = smooth_signal(sig, fs, smooth_sec=SMOOTH_PRED_SEC, polyorder=3)
    return sig.astype(np.float32)


def postprocess_true_window(sig, fs):
    sig = bandpass_filter(sig, fs, 0.10, 0.50)
    sig = smooth_signal(sig, fs, smooth_sec=SMOOTH_TRUE_SEC, polyorder=3)
    return sig.astype(np.float32)


def compute_window_metrics(pred_sig, true_sig, fs):
    pred_p = postprocess_prediction_window(pred_sig, fs)
    true_p = postprocess_true_window(true_sig, fs)
    pred_n = normalize_for_plot(pred_p)
    true_n = normalize_for_plot(true_p)
    return {
        "pred_proc": pred_p,
        "true_proc": true_p,
        "corr": safe_pearsonr(true_n, pred_n),
        "mae": waveform_mae(true_n, pred_n),
    }


def _unwrap_mat_value(value):
    arr = np.asarray(value)
    while arr.dtype == object and arr.size == 1:
        arr = np.asarray(arr.item())
    return arr


def _first_mat_payload(mat):
    valid_keys = [k for k in mat.keys() if not k.startswith("_")]
    if not valid_keys:
        raise KeyError("No valid keys found in MAT file.")
    payload = mat[valid_keys[0]]
    payload = _unwrap_mat_value(payload)
    if payload.ndim >= 2 and payload.shape[0] == 1 and payload.shape[1] == 1:
        payload = payload[0, 0]
    return payload


def _extract_signal_and_fs(payload):
    signal = sanitize_signal(_unwrap_mat_value(payload["data"]))
    srate = float(np.asarray(_unwrap_mat_value(payload["srate"])).squeeze())
    if not np.isfinite(srate) or srate <= 0:
        raise ValueError(f"Invalid srate: {srate}")
    return signal, srate


def stem_name(filename):
    return os.path.splitext(filename)[0]


def match_mat_file_pairs(ppg_folder, rsp_folder):
    ppg_files = sorted([f for f in os.listdir(ppg_folder) if f.endswith(".mat")])
    rsp_files = sorted([f for f in os.listdir(rsp_folder) if f.endswith(".mat")])
    if not ppg_files or not rsp_files:
        return []
    ppg_map = {stem_name(f): f for f in ppg_files}
    rsp_map = {stem_name(f): f for f in rsp_files}
    common = sorted(set(ppg_map).intersection(rsp_map))
    if common:
        return [(ppg_map[key], rsp_map[key]) for key in common]
    if len(ppg_files) != len(rsp_files):
        raise RuntimeError(f"File count mismatch without common stems: PPG={len(ppg_files)}, RSP={len(rsp_files)}")
    print("Warning: no common file stems found. Falling back to sorted-order pairing.")
    return list(zip(ppg_files, rsp_files))


# ===========================================================
# 3. Subject Cache / Dataset Loading
# ===========================================================
def get_cache_path(cache_dir, dataset_key, subject, window_sec, stride_sec):
    tag = f"w{float(window_sec):g}_s{float(stride_sec):g}"
    return os.path.join(cache_dir, f"{safe_tag(dataset_key)}_{safe_tag(subject)}_{safe_tag(tag)}_processed.npz")


def load_subject_cache(cache_dir, dataset_key, subject, window_sec, stride_sec):
    cache_path = get_cache_path(cache_dir, dataset_key, subject, window_sec, stride_sec)
    if not os.path.exists(cache_path):
        return None
    try:
        cached = np.load(cache_path, allow_pickle=False)
        return {
            "low": cached["low"].astype(np.float32),
            "high": cached["high"].astype(np.float32),
            "env": cached["env"].astype(np.float32),
            "freq": cached["freq"].astype(np.float32),
            "label": cached["label"].astype(np.float32),
            "fs": int(cached["fs"]),
        }
    except Exception as exc:
        print(f"   [Cache Load Fail] {subject}: {exc}")
        return None


def save_subject_cache(cache_dir, dataset_key, subject, data, window_sec, stride_sec):
    cache_path = get_cache_path(cache_dir, dataset_key, subject, window_sec, stride_sec)
    try:
        np.savez_compressed(
            cache_path,
            low=data["low"].astype(np.float32),
            high=data["high"].astype(np.float32),
            env=data["env"].astype(np.float32),
            freq=data["freq"].astype(np.float32),
            label=data["label"].astype(np.float32),
            fs=np.array(data["fs"], dtype=np.int32),
            window_sec=np.array(window_sec, dtype=np.float32),
            stride_sec=np.array(stride_sec, dtype=np.float32),
        )
    except Exception as exc:
        print(f"   [Cache Save Fail] {subject}: {exc}")


def preprocess_subject_windows(dataset_key, subject, ppg_raw, target_raw, fs, cache_dir, window_sec, stride_sec):
    cached = load_subject_cache(cache_dir, dataset_key, subject, window_sec, stride_sec)
    if cached is not None:
        print(f"Loaded cache {subject}: {len(cached['low'])} windows | fs={cached['fs']}")
        return cached

    ppg_raw = sanitize_signal(ppg_raw).astype(np.float32)
    target_raw = sanitize_signal(target_raw).astype(np.float32)
    m = min(len(ppg_raw), len(target_raw))
    ppg_raw = ppg_raw[:m]
    target_raw = target_raw[:m]
    valid = np.isfinite(ppg_raw) & np.isfinite(target_raw)
    ppg_raw = ppg_raw[valid]
    target_raw = target_raw[valid]

    if len(ppg_raw) < int(window_sec * fs) or len(target_raw) < int(window_sec * fs):
        return None

    ppg_raw = correct_polarity_skew(ppg_raw)
    p_low = bandpass_filter(ppg_raw, fs, 0.1, 0.5)
    p_high = bandpass_filter(ppg_raw, fs, 0.5, 5.0)
    target_f = bandpass_filter(target_raw, fs, 0.1, 0.5)
    target_f = align_phase_only(p_low, target_f)
    env, freq = get_hilbert_features(p_high, fs)

    low_win = extract_windows(p_low, fs, window_sec, stride_sec)
    high_win = extract_windows(p_high, fs, window_sec, stride_sec)
    env_win = extract_windows(env, fs, window_sec, stride_sec)
    freq_win = extract_windows(freq, fs, window_sec, stride_sec)
    label_win = extract_windows(target_f, fs, window_sec, stride_sec)

    n_win = min(len(low_win), len(high_win), len(env_win), len(freq_win), len(label_win))
    if n_win == 0:
        return None

    low_all = []
    high_all = []
    env_all = []
    freq_all = []
    label_all = []
    for i in range(n_win):
        if check_quality_real_world(low_win[i], label_win[i], fs):
            low_all.append(low_win[i].astype(np.float32).copy())
            high_all.append(high_win[i].astype(np.float32).copy())
            env_all.append(env_win[i].astype(np.float32).copy())
            freq_all.append(freq_win[i].astype(np.float32).copy())
            label_all.append(label_win[i].astype(np.float32).copy())

    if not low_all:
        return None

    data = {
        "low": np.stack(low_all, axis=0).astype(np.float32),
        "high": np.stack(high_all, axis=0).astype(np.float32),
        "env": np.stack(env_all, axis=0).astype(np.float32),
        "freq": np.stack(freq_all, axis=0).astype(np.float32),
        "label": np.stack(label_all, axis=0).astype(np.float32),
        "fs": int(fs),
    }
    save_subject_cache(cache_dir, dataset_key, subject, data, window_sec, stride_sec)
    return data


def load_bidmc_dataset(cfg, dirs):
    if not os.path.exists(cfg["data_path"]):
        raise FileNotFoundError(f"BIDMC file not found: {cfg['data_path']}")
    subject_data = {}
    mat = sio.loadmat(cfg["data_path"])
    all_data = mat["data"][0]
    for i in range(len(all_data)):
        subject = f"{i + 1:02d}"
        try:
            subj_data = all_data[i]
            ppg_raw = np.array(subj_data["ppg"][0, 0]["v"]).squeeze()
            target_raw = np.array(subj_data["ref"][0, 0]["resp_sig"][0, 0]["imp"][0, 0]["v"]).squeeze()
            result = preprocess_subject_windows(
                "bidmc",
                subject,
                ppg_raw,
                target_raw,
                int(cfg["target_fs"]),
                dirs["cache"],
                cfg["window_sec"],
                cfg["stride_sec"],
            )
            if result is not None:
                subject_data[subject] = result
                print(f"Loaded BIDMC {subject}: {len(result['low'])} windows | fs={result['fs']}")
        except Exception as exc:
            print(f"   [Load Fail] BIDMC {subject}: {exc}")
    return subject_data


def load_capnobase_dataset(cfg, dirs):
    csv_dir = cfg["data_path"]
    if not os.path.isdir(csv_dir):
        raise FileNotFoundError(f"CapnoBase CSV directory not found: {csv_dir}")
    subject_data = {}
    all_files = os.listdir(csv_dir)
    case_ids = sorted({f.split("_")[0] for f in all_files if f.endswith("_8min_signal.csv")})
    for case_id in case_ids:
        try:
            signal_path = os.path.join(csv_dir, f"{case_id}_8min_signal.csv")
            param_path = os.path.join(csv_dir, f"{case_id}_8min_param.csv")
            sig_df = pd.read_csv(signal_path)
            param_df = pd.read_csv(param_path)
            orig_fs = int(param_df["samplingrate_pleth"].values[0])
            ppg_raw = sig_df["pleth_y"].values.astype(np.float32)
            target_raw = sig_df["co2_y"].values.astype(np.float32)
            ppg_res = maybe_resample(ppg_raw, orig_fs, int(cfg["target_fs"]))
            target_res = maybe_resample(target_raw, orig_fs, int(cfg["target_fs"]))
            result = preprocess_subject_windows(
                "capnobase",
                case_id,
                ppg_res,
                target_res,
                int(cfg["target_fs"]),
                dirs["cache"],
                cfg["window_sec"],
                cfg["stride_sec"],
            )
            if result is not None:
                subject_data[case_id] = result
                print(f"Loaded CapnoBase {case_id}: {len(result['low'])} windows | fs={result['fs']}")
        except Exception as exc:
            print(f"   [Load Fail] CapnoBase {case_id}: {exc}")
    return subject_data


def load_steam_dataset(cfg, dirs):
    ppg_dir = cfg["ppg_dir"]
    rsp_dir = cfg["rsp_dir"]
    if not os.path.isdir(ppg_dir):
        raise FileNotFoundError(f"STEAM PPG directory not found: {ppg_dir}")
    if not os.path.isdir(rsp_dir):
        raise FileNotFoundError(f"STEAM RSP directory not found: {rsp_dir}")
    subject_data = {}
    subjects = sorted([d for d in os.listdir(ppg_dir) if os.path.isdir(os.path.join(ppg_dir, d))])
    target_fs = int(cfg["target_fs"])

    for subject in subjects:
        try:
            subj_ppg_dir = os.path.join(ppg_dir, subject)
            subj_rsp_dir = os.path.join(rsp_dir, subject)
            if not os.path.isdir(subj_rsp_dir):
                print(f"   [Skip] STEAM {subject}: missing RSP folder")
                continue
            pairs = match_mat_file_pairs(subj_ppg_dir, subj_rsp_dir)
            subject_packs = []
            for p_file, r_file in pairs:
                try:
                    p_payload = _first_mat_payload(sio.loadmat(os.path.join(subj_ppg_dir, p_file)))
                    r_payload = _first_mat_payload(sio.loadmat(os.path.join(subj_rsp_dir, r_file)))
                    ppg_raw, fs_ppg = _extract_signal_and_fs(p_payload)
                    rsp_raw, fs_rsp = _extract_signal_and_fs(r_payload)
                    ppg_res = resample_signal_poly(ppg_raw, fs_ppg, target_fs)
                    rsp_res = resample_signal_poly(rsp_raw, fs_rsp, target_fs)
                    pack = preprocess_subject_windows(
                        "steam",
                        f"{subject}_{stem_name(p_file)}",
                        ppg_res,
                        rsp_res,
                        target_fs,
                        dirs["cache"],
                        cfg["window_sec"],
                        cfg["stride_sec"],
                    )
                    if pack is not None:
                        subject_packs.append(pack)
                except Exception as exc:
                    print(f"   [Skip] STEAM {subject} pair {p_file}/{r_file}: {exc}")
            if subject_packs:
                subject_data[subject] = {
                    "low": np.concatenate([p["low"] for p in subject_packs], axis=0).astype(np.float32),
                    "high": np.concatenate([p["high"] for p in subject_packs], axis=0).astype(np.float32),
                    "env": np.concatenate([p["env"] for p in subject_packs], axis=0).astype(np.float32),
                    "freq": np.concatenate([p["freq"] for p in subject_packs], axis=0).astype(np.float32),
                    "label": np.concatenate([p["label"] for p in subject_packs], axis=0).astype(np.float32),
                    "fs": target_fs,
                }
                print(f"Loaded STEAM {subject}: {len(subject_data[subject]['low'])} windows | fs={target_fs}")
        except Exception as exc:
            print(f"   [Load Fail] STEAM {subject}: {exc}")
    return subject_data


def load_dataset(dataset_key, dirs):
    cfg = DATASET_CONFIGS[dataset_key]
    print(f"\nLoading dataset: {cfg['label']}")
    if dataset_key == "steam":
        return load_steam_dataset(cfg, dirs)
    if dataset_key == "bidmc":
        return load_bidmc_dataset(cfg, dirs)
    if dataset_key == "capnobase":
        return load_capnobase_dataset(cfg, dirs)
    raise KeyError(f"Unsupported dataset key: {dataset_key}")


# ===========================================================
# 4. Dataset / Model
# ===========================================================
class DiffDataset(Dataset):
    def __init__(self, x, y, augment=False, mode_key="baseline"):
        self.x = torch.as_tensor(x, dtype=torch.float32)
        self.y = torch.as_tensor(y, dtype=torch.float32)
        self.augment = augment
        self.mode_key = mode_key

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        x = self.x[idx].clone()
        y = self.y[idx].clone()

        if self.augment:
            if self.mode_key == "proposed":
                shift = random.randint(-24, 24)
                noise_std = random.uniform(0.002, 0.010)
                aux_drop_prob = 0.20
            else:
                shift = random.randint(-12, 12)
                noise_std = random.uniform(0.001, 0.004)
                aux_drop_prob = 0.0

            if shift != 0:
                x = torch.roll(x, shifts=shift, dims=-1)
                y = torch.roll(y, shifts=shift, dims=-1)

            x = x + torch.randn_like(x) * noise_std

            if x.shape[0] > 1:
                for ch in range(1, x.shape[0]):
                    if random.random() < aux_drop_prob:
                        x[ch] = 0.0

        return x, y


def build_window_indices(length, step=1, max_windows=None):
    indices = np.arange(0, length, max(1, int(step)), dtype=np.int64)
    if max_windows is not None and len(indices) > int(max_windows):
        sampled = np.linspace(0, len(indices) - 1, int(max_windows))
        indices = indices[np.round(sampled).astype(np.int64)]
    return indices


def build_xy(subject_data, subjects, mode_key, window_step=1, max_windows_per_subject=None):
    cfg = MODE_CONFIGS[mode_key]
    input_keys = cfg["input_features"]
    x_all = []
    y_all = []

    for subj in subjects:
        data = subject_data[subj]
        indices = build_window_indices(
            len(data["label"]),
            step=window_step,
            max_windows=max_windows_per_subject,
        )
        x = np.stack([data[key][indices] for key in input_keys], axis=1)
        y = data["label"][indices][:, np.newaxis, :]
        x_all.append(x.astype(np.float32))
        y_all.append(y.astype(np.float32))

    x_all = instance_normalize(np.concatenate(x_all, axis=0))
    y_all = instance_normalize(np.concatenate(y_all, axis=0))
    return x_all, y_all


def make_loader(dataset, batch_size, shuffle):
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        drop_last=False,
        num_workers=NUM_WORKERS,
        pin_memory=PIN_MEMORY,
    )


def create_loaders(subject_data, train_subjects, val_subjects, test_subject, mode_key, batch_size, cfg):
    x_train, y_train = build_xy(
        subject_data,
        train_subjects,
        mode_key,
        window_step=cfg["train_window_step"],
        max_windows_per_subject=cfg["max_train_windows_per_subject"],
    )
    x_val, y_val = build_xy(
        subject_data,
        val_subjects,
        mode_key,
        window_step=cfg["val_window_step"],
        max_windows_per_subject=cfg["max_val_windows_per_subject"],
    )
    x_test, y_test = build_xy(
        subject_data,
        [test_subject],
        mode_key,
        window_step=cfg["test_window_step"],
        max_windows_per_subject=cfg["max_test_windows_per_subject"],
    )

    train_loader = make_loader(DiffDataset(x_train, y_train, augment=True, mode_key=mode_key), batch_size, True)
    val_loader = make_loader(DiffDataset(x_val, y_val, augment=False, mode_key=mode_key), batch_size, False)
    test_loader = make_loader(DiffDataset(x_test, y_test, augment=False, mode_key=mode_key), batch_size, False)
    return train_loader, val_loader, test_loader


class CNNEncoder(nn.Module):
    def __init__(self, in_channels, base_dim=32, dropout=0.0):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(in_channels, base_dim, kernel_size=15, padding=7),
            nn.GroupNorm(1, base_dim),
            nn.Mish(),
            nn.Dropout(dropout),
            nn.Conv1d(base_dim, base_dim, kernel_size=15, padding=7),
            nn.GroupNorm(1, base_dim),
            nn.Mish(),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)


class DenoiserV6(nn.Module):
    def __init__(self, ppg_channels=1, base_dim=32, rnn_hidden=128, dropout=0.0, feature_dropout=0.0):
        super().__init__()
        self.feature_dropout = feature_dropout
        self.ppg_enc = CNNEncoder(ppg_channels, base_dim, dropout=dropout)
        self.rsp_enc = CNNEncoder(1, base_dim, dropout=dropout)
        self.time_mlp = nn.Sequential(
            nn.Linear(1, 64),
            nn.Mish(),
            nn.Dropout(dropout),
            nn.Linear(64, 64),
        )
        self.lstm = nn.LSTM(
            base_dim * 2 + 64,
            rnn_hidden,
            num_layers=2,
            batch_first=True,
            bidirectional=True,
            dropout=dropout,
        )
        self.out_head = nn.Sequential(
            nn.Linear(rnn_hidden * 2, rnn_hidden),
            nn.Mish(),
            nn.Dropout(dropout),
            nn.Linear(rnn_hidden, 1),
        )

    def _drop_aux_features(self, x):
        if not self.training or self.feature_dropout <= 0.0 or x.shape[1] <= 1:
            return x
        keep_mask = (
            torch.rand((x.shape[0], x.shape[1] - 1, 1), device=x.device)
            > self.feature_dropout
        ).float()
        x = x.clone()
        x[:, 1:, :] = x[:, 1:, :] * keep_mask
        return x

    def forward(self, y_t, t, x_ppg):
        x_ppg = self._drop_aux_features(x_ppg)
        f_ppg = self.ppg_enc(x_ppg)
        f_rsp = self.rsp_enc(y_t)
        t_emb = self.time_mlp(t.float().view(-1, 1) / 1000.0)
        t_emb = t_emb.unsqueeze(-1).repeat(1, 1, y_t.shape[2])
        feat = torch.cat([f_ppg, f_rsp, t_emb], dim=1).permute(0, 2, 1)
        rnn_out, _ = self.lstm(feat)
        return self.out_head(rnn_out).permute(0, 2, 1)


class RespDiffHilbert(nn.Module):
    def __init__(self, denoiser, timesteps=1000):
        super().__init__()
        self.denoiser = denoiser
        self.timesteps = timesteps

        beta = torch.linspace(1e-4, 0.02, timesteps)
        alpha = 1.0 - beta
        alpha_bar = torch.cumprod(alpha, dim=0)

        self.register_buffer("beta", beta)
        self.register_buffer("alpha", alpha)
        self.register_buffer("alpha_bar", alpha_bar)
        self.register_buffer("sqrt_alpha_bar", torch.sqrt(alpha_bar))
        self.register_buffer("sqrt_one_minus_alpha_bar", torch.sqrt(1.0 - alpha_bar))

    @staticmethod
    def _first_diff(x):
        return x[:, :, 1:] - x[:, :, :-1]

    @staticmethod
    def _second_diff(x):
        return x[:, :, 2:] - 2.0 * x[:, :, 1:-1] + x[:, :, :-2]

    def _pearson_loss(self, pred, target):
        pred = pred.squeeze(1)
        target = target.squeeze(1)
        pred_c = pred - pred.mean(dim=1, keepdim=True)
        target_c = target - target.mean(dim=1, keepdim=True)
        cov = torch.sum(pred_c * target_c, dim=1)
        pred_pow = torch.sum(pred_c ** 2, dim=1)
        target_pow = torch.sum(target_c ** 2, dim=1)
        corr = cov / (torch.sqrt(pred_pow * target_pow) + 1e-6)
        return 1.0 - torch.mean(corr)

    def loss_components(
        self,
        y_0,
        x_ppg,
        lambda_spec=0.05,
        lambda_wave=0.10,
        lambda_corr=0.05,
        lambda_grad=0.04,
        lambda_smooth=0.02,
    ):
        t = torch.randint(0, self.timesteps, (y_0.shape[0],), device=y_0.device).long()
        noise = torch.randn_like(y_0)

        sqrt_ab_t = self.sqrt_alpha_bar[t].view(-1, 1, 1)
        sqrt_1m_ab_t = self.sqrt_one_minus_alpha_bar[t].view(-1, 1, 1)
        y_t = sqrt_ab_t * y_0 + sqrt_1m_ab_t * noise

        pred_noise = self.denoiser(y_t, t, x_ppg)
        l_diff = F.mse_loss(pred_noise, noise)

        y0_recon = (y_t - sqrt_1m_ab_t * pred_noise) / (sqrt_ab_t + 1e-6)
        y0_recon = torch.clamp(y0_recon, -5.0, 5.0)

        y0_fft = torch.fft.rfft(y_0, dim=-1).abs()
        y0_recon_fft = torch.fft.rfft(y0_recon, dim=-1).abs()
        l_spec = F.mse_loss(torch.log1p(y0_recon_fft), torch.log1p(y0_fft))

        l_wave = F.smooth_l1_loss(y0_recon, y_0)
        l_corr = self._pearson_loss(y0_recon, y_0)
        l_grad = F.smooth_l1_loss(self._first_diff(y0_recon), self._first_diff(y_0))
        l_smooth = F.smooth_l1_loss(self._second_diff(y0_recon), self._second_diff(y_0))

        total = (
            l_diff
            + lambda_spec * l_spec
            + lambda_wave * l_wave
            + lambda_corr * l_corr
            + lambda_grad * l_grad
            + lambda_smooth * l_smooth
        )
        return total, {
            "total": float(total.detach().item()),
            "diff": float(l_diff.detach().item()),
            "spec": float(l_spec.detach().item()),
            "wave": float(l_wave.detach().item()),
            "corr": float(l_corr.detach().item()),
            "grad": float(l_grad.detach().item()),
            "smooth": float(l_smooth.detach().item()),
        }

    @torch.no_grad()
    def sample(self, x_ppg, sample_steps=200):
        y = torch.randn((x_ppg.shape[0], 1, x_ppg.shape[2]), device=x_ppg.device)
        step_ids = torch.linspace(
            self.timesteps - 1,
            0,
            steps=min(sample_steps, self.timesteps),
            device=x_ppg.device,
        ).long()
        step_ids = torch.unique_consecutive(step_ids)

        for idx, step in enumerate(step_ids):
            i = int(step.item())
            t = torch.full((x_ppg.shape[0],), i, device=x_ppg.device, dtype=torch.long)
            pred_noise = self.denoiser(y, t, x_ppg)

            ab_t = self.alpha_bar[i].view(1, 1, 1)
            x0_pred = (y - torch.sqrt(1.0 - ab_t) * pred_noise) / (torch.sqrt(ab_t) + 1e-8)
            x0_pred = torch.clamp(x0_pred, -5.0, 5.0)

            if idx == len(step_ids) - 1:
                y = x0_pred
            else:
                prev_i = int(step_ids[idx + 1].item())
                ab_prev = self.alpha_bar[prev_i].view(1, 1, 1)
                y = torch.sqrt(ab_prev) * x0_pred + torch.sqrt(1.0 - ab_prev) * pred_noise

        return y


def build_model(mode_key, args):
    cfg = get_mode_config(mode_key, args)
    denoiser = DenoiserV6(
        ppg_channels=len(cfg["input_features"]),
        base_dim=cfg["base_dim"],
        rnn_hidden=cfg["rnn_hidden"],
        dropout=cfg["dropout"],
        feature_dropout=cfg["feature_dropout"],
    ).to(device)
    return RespDiffHilbert(denoiser, timesteps=DIFF_TIMESTEPS).to(device)


def load_state_dict_compat(model_path):
    try:
        return torch.load(model_path, map_location=device, weights_only=True)
    except TypeError:
        return torch.load(model_path, map_location=device)


# ===========================================================
# 5. Train / Evaluate
# ===========================================================
def split_train_val_subjects(all_subjects, test_subject, val_ratio=VAL_RATIO, min_val_subjects=MIN_VAL_SUBJECTS):
    pool = [s for s in all_subjects if s != test_subject]
    if len(pool) < 2:
        raise RuntimeError("At least 3 valid subjects are required for LOSO training.")
    rotate = all_subjects.index(test_subject) % len(pool)
    pool = pool[rotate:] + pool[:rotate]
    val_count = max(1, int(round(len(pool) * val_ratio)))
    if len(pool) >= 4:
        val_count = max(val_count, min_val_subjects)
    val_count = min(val_count, max(1, len(pool) - 1))
    val_subjects = pool[:val_count]
    train_subjects = pool[val_count:]
    if not train_subjects:
        train_subjects = val_subjects[-1:]
        val_subjects = val_subjects[:-1]
    return train_subjects, val_subjects


def empty_history():
    history = {}
    for split in ["train", "val"]:
        for key in LOSS_KEYS:
            history[f"{split}_{key}"] = []
    return history


def append_history(history, split, values):
    for key in LOSS_KEYS:
        history[f"{split}_{key}"].append(float(values.get(key, np.nan)))


def average_loss_dict(loss_dicts):
    if not loss_dicts:
        return {key: np.nan for key in LOSS_KEYS}
    return {key: float(np.mean([item[key] for item in loss_dicts])) for key in LOSS_KEYS}


def evaluate_loader(model, loader, cfg):
    model.eval()
    losses = []
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            with autocast_context():
                _, loss_info = model.loss_components(
                    y,
                    x,
                    lambda_spec=cfg["lambda_spec"],
                    lambda_wave=cfg["lambda_wave"],
                    lambda_corr=cfg["lambda_corr"],
                    lambda_grad=cfg["lambda_grad"],
                    lambda_smooth=cfg["lambda_smooth"],
                )
            losses.append(loss_info)
    return average_loss_dict(losses)


def sample_loader(model, loader, sample_steps):
    model.eval()
    preds = []
    trues = []
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device, non_blocking=True)
            pred = model.sample(x, sample_steps=sample_steps)
            preds.append(pred.squeeze(1).float().cpu().numpy())
            trues.append(y.squeeze(1).float().cpu().numpy())
    pred_arr = np.concatenate(preds, axis=0) if preds else np.array([])
    true_arr = np.concatenate(trues, axis=0) if trues else np.array([])
    return pred_arr.astype(np.float32), true_arr.astype(np.float32)


def train_one_mode(dataset_key, test_subject, subject_data, train_subjects, val_subjects, mode_key, dirs, args):
    dataset_cfg = DATASET_CONFIGS[dataset_key]
    cfg = get_mode_config(mode_key, args)
    mode_label = cfg["label"]

    print(f"\n--- [{dataset_cfg['label']}] {test_subject} | MODE: {mode_label.upper()} ---")
    print(f"Train subjects ({len(train_subjects)}): {train_subjects}")
    print(f"Val subjects ({len(val_subjects)}): {val_subjects}")
    print(
        "Runtime speed config | "
        f"epochs={cfg['epochs']}, sample_steps={cfg['sample_steps']}, batch_size={cfg['batch_size']}, "
        f"val_every={cfg['val_every']}, train_step={cfg['train_window_step']}, "
        f"val_step={cfg['val_window_step']}, test_step={cfg['test_window_step']}, "
        f"patience_evals={cfg['early_stop_patience_evals']}, "
        f"max_train={cfg['max_train_windows_per_subject']}, "
        f"max_val={cfg['max_val_windows_per_subject']}, "
        f"max_test={cfg['max_test_windows_per_subject']}"
    )

    train_loader, val_loader, test_loader = create_loaders(
        subject_data=subject_data,
        train_subjects=train_subjects,
        val_subjects=val_subjects,
        test_subject=test_subject,
        mode_key=mode_key,
        batch_size=cfg["batch_size"],
        cfg=cfg,
    )

    model = build_model(mode_key, args)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg["lr"], weight_decay=cfg["weight_decay"])
    scheduler = CosineAnnealingLR(optimizer, T_max=cfg["epochs"], eta_min=1e-6)
    scaler = build_grad_scaler()
    model_path = os.path.join(dirs["models"], f"{safe_tag(test_subject)}_{safe_tag(mode_key)}_best.pt")

    history = empty_history()
    best_val_loss = np.inf
    best_epoch = 0
    wait = 0
    trainable_params = count_trainable_params(model)
    last_val = {key: np.nan for key in LOSS_KEYS}

    for ep in range(cfg["epochs"]):
        model.train()
        train_losses = []

        for x, y in train_loader:
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)
            with autocast_context():
                loss, loss_info = model.loss_components(
                    y,
                    x,
                    lambda_spec=cfg["lambda_spec"],
                    lambda_wave=cfg["lambda_wave"],
                    lambda_corr=cfg["lambda_corr"],
                    lambda_grad=cfg["lambda_grad"],
                    lambda_smooth=cfg["lambda_smooth"],
                )

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()

            train_losses.append(loss_info)

        avg_train = average_loss_dict(train_losses)
        append_history(history, "train", avg_train)
        should_eval = (ep == 0) or ((ep + 1) % cfg["val_every"] == 0) or ((ep + 1) == cfg["epochs"])
        if should_eval:
            avg_val = evaluate_loader(model, val_loader, cfg)
            last_val = avg_val
        else:
            avg_val = {key: np.nan for key in LOSS_KEYS}
        append_history(history, "val", avg_val)
        scheduler.step()

        if should_eval:
            improved = last_val["total"] < best_val_loss - EARLY_STOP_MIN_DELTA
            if improved:
                best_val_loss = last_val["total"]
                best_epoch = ep + 1
                wait = 0
                torch.save(model.state_dict(), model_path)
            elif ep + 1 >= cfg["min_epochs"]:
                wait += 1

        if (ep + 1) % 10 == 0 or ep == 0:
            if should_eval:
                val_msg = f"Val(total): {last_val['total']:.4f} | Val(wave): {last_val['wave']:.4f} | Val(corr): {last_val['corr']:.4f}"
            else:
                val_msg = f"Val(skip, every {cfg['val_every']} ep) | Last Val(total): {last_val['total']:.4f}"
            print(
                f"Ep {ep + 1:03d} | "
                f"Tr(total): {avg_train['total']:.4f} | "
                f"{val_msg} | "
                f"LR: {optimizer.param_groups[0]['lr']:.6f}"
            )

        if ep + 1 >= cfg["min_epochs"] and wait >= cfg["early_stop_patience_evals"]:
            print(f"Early stop at epoch {ep + 1}")
            break

    if not os.path.exists(model_path):
        torch.save(model.state_dict(), model_path)
        best_epoch = max(best_epoch, cfg["epochs"])
        best_val_loss = last_val["total"] if np.isfinite(last_val["total"]) else np.nan
    model.load_state_dict(load_state_dict_compat(model_path))
    preds, y_true = sample_loader(model, test_loader, cfg["sample_steps"])
    output = {
        "history": history,
        "preds": preds,
        "true": y_true,
        "best_val_loss": float(best_val_loss),
        "best_epoch": int(best_epoch),
        "trainable_params": int(trainable_params),
        "model_path": model_path,
    }
    del model, train_loader, val_loader, test_loader
    clear_runtime_memory()
    return output


# ===========================================================
# 6. Plot / Metrics / Resume Utilities
# ===========================================================
def concat_dataframes(frames):
    valid = [df for df in frames if df is not None and not df.empty]
    if not valid:
        return pd.DataFrame()
    return pd.concat(valid, ignore_index=True)


def metric_csv_path(metrics_dir, prefix, name):
    return os.path.join(metrics_dir, f"{prefix}_{name}.csv")


def read_metric_csv(path):
    if not os.path.exists(path):
        return pd.DataFrame()
    dtype_map = {"Subject": str, "Model": str}
    try:
        return pd.read_csv(path, dtype=dtype_map)
    except Exception:
        return pd.read_csv(path)


def merge_metric_versions(metrics_dir, name, dedupe_cols):
    final_df = read_metric_csv(metric_csv_path(metrics_dir, FINAL_PREFIX, name))
    running_df = read_metric_csv(metric_csv_path(metrics_dir, RUNNING_PREFIX, name))
    merged = concat_dataframes([final_df, running_df])
    if merged.empty:
        return merged
    return merged.drop_duplicates(subset=list(dedupe_cols), keep="last").reset_index(drop=True)


def load_existing_metrics(metrics_dir):
    per_subject_df = merge_metric_versions(metrics_dir, "PerSubject", ("Subject", "Model"))
    per_window_df = merge_metric_versions(metrics_dir, "PerWindow", ("Subject", "Model", "Window_Index"))
    return per_subject_df, per_window_df


def subject_key(value):
    """Subject ID for matching stored rows: '0009', '9' and '9.0' are the same subject."""
    text = str(value).strip()
    if text.endswith(".0") and text[:-2].isdigit():
        text = text[:-2]
    return (text.lstrip("0") or "0") if text.isdigit() else text


def subject_mask(df, subject):
    return df["Subject"].map(subject_key) == subject_key(subject)


def active_mode_mask(df):
    return df["Model"].astype(str).isin(ACTIVE_MODES)


def completed_mode_count(per_subject_df, subject):
    """How many of the modes run here (Baseline) already have a stored result for the subject."""
    if per_subject_df.empty or "Subject" not in per_subject_df.columns or "Model" not in per_subject_df.columns:
        return 0
    rows = per_subject_df[subject_mask(per_subject_df, subject) & active_mode_mask(per_subject_df)]
    return rows["Model"].astype(str).nunique()


def subject_is_complete(per_subject_df, subject):
    return completed_mode_count(per_subject_df, subject) >= len(ACTIVE_MODES)


def drop_subject_rows(per_subject_df, subject):
    """Drop the subject's rows of the modes run here only (stored Proposed rows are kept)."""
    if per_subject_df.empty or "Subject" not in per_subject_df.columns or "Model" not in per_subject_df.columns:
        return per_subject_df
    keep = ~(subject_mask(per_subject_df, subject) & active_mode_mask(per_subject_df))
    return per_subject_df[keep].reset_index(drop=True)


def drop_subject_windows(per_window_df, subject):
    """Drop the subject's windows of the modes run here only (stored Proposed windows are kept)."""
    if per_window_df.empty or "Subject" not in per_window_df.columns or "Model" not in per_window_df.columns:
        return per_window_df
    keep = ~(subject_mask(per_window_df, subject) & active_mode_mask(per_window_df))
    return per_window_df[keep].reset_index(drop=True)


def report_existing_runs(save_root, dataset_key, run_tag):
    """List the run folders already under {save-root}\{dataset}: subjects with a stored Baseline result each."""
    dataset_base = os.path.join(save_root, DATASET_CONFIGS[dataset_key]["save_name"])
    if not os.path.isdir(dataset_base):
        print(f"No earlier results under {dataset_base}")
        return
    found = []
    for name in sorted(os.listdir(dataset_base)):
        metrics_dir = os.path.join(dataset_base, name, "Metrics")
        if not os.path.isdir(metrics_dir):
            continue
        per_subject_df, _ = load_existing_metrics(metrics_dir)
        done = set()
        if not per_subject_df.empty and {"Subject", "Model"} <= set(per_subject_df.columns):
            done = set(per_subject_df.loc[active_mode_mask(per_subject_df), "Subject"].map(subject_key))
        found.append((name, len(done)))
    if not found:
        print(f"No earlier run folders under {dataset_base}")
        return
    print(f"Earlier run folders under {dataset_base}:")
    for name, n_done in found:
        print(f"  {'*' if name == run_tag else ' '} {name}: {n_done} subjects with a Baseline result")
    current = dict(found).get(run_tag, 0)
    others = [name for name, n_done in found if name != run_tag and n_done > 0]
    if current == 0 and others:
        print(f"  Note: this run ({run_tag}) has no stored result yet; to continue an earlier folder, run with "
              f"its speed settings or --run-tag (e.g. --run-tag {others[0]}).")


def save_dataframe_bundle(metrics_dir, prefix, dataframes):
    os.makedirs(metrics_dir, exist_ok=True)
    paths = {}
    for name, df in dataframes.items():
        csv_path = os.path.join(metrics_dir, f"{prefix}_{name}.csv")
        df.to_csv(csv_path, index=False, encoding="utf-8-sig")
        paths[name] = csv_path
    xlsx_path = os.path.join(metrics_dir, f"{prefix}.xlsx")
    try:
        with pd.ExcelWriter(xlsx_path, engine="openpyxl") as writer:
            for name, df in dataframes.items():
                df.to_excel(writer, sheet_name=name[:31], index=False)
        paths["excel"] = xlsx_path
    except Exception as exc:
        print(f"Excel save skipped for {prefix}: {exc}")
    return paths


def save_progress_json(metrics_dir, dataset_key, all_subjects, per_subject_df, args):
    completed = sorted([str(s) for s in all_subjects if subject_is_complete(per_subject_df, s)])
    payload = {
        "run_tag": build_run_tag(args),
        "speed_preset": args.speed_preset,
        "dataset_key": dataset_key,
        "dataset_label": DATASET_CONFIGS[dataset_key]["label"],
        "updated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "completed_subjects": completed,
        "num_completed": len(completed),
        "num_total": len(all_subjects),
        "mode_keys": ACTIVE_MODES,
        "note": "CapnoBase only, Baseline only (completed = a stored Baseline result)",
    }
    save_path = os.path.join(metrics_dir, "progress.json")
    with open(save_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    return save_path


def build_parameter_table(dataset_keys, args):
    rows = []
    for dataset_key in dataset_keys:
        dcfg = DATASET_CONFIGS[dataset_key]
        for mode_key in ACTIVE_MODES:
            mcfg = get_mode_config(mode_key, args)
            rows.append(
                {
                    "Run_Tag": build_run_tag(args),
                    "Speed_Preset": args.speed_preset,
                    "Dataset": dcfg["label"],
                    "Dataset_Key": dataset_key,
                    "Model": mode_key,
                    "Model_Label": mcfg["label"],
                    "Input_Features": "+".join(mcfg["input_features"]),
                    "Num_Input_Channels": len(mcfg["input_features"]),
                    "Target_FS": dcfg["target_fs"],
                    "Window_Sec": dcfg["window_sec"],
                    "Stride_Sec": dcfg["stride_sec"],
                    "Epochs": mcfg["epochs"],
                    "Batch_Size": mcfg["batch_size"],
                    "Sample_Steps": mcfg["sample_steps"],
                    "Val_Every": mcfg["val_every"],
                    "Early_Stop_Patience_Evals": mcfg["early_stop_patience_evals"],
                    "Train_Window_Step": mcfg["train_window_step"],
                    "Val_Window_Step": mcfg["val_window_step"],
                    "Test_Window_Step": mcfg["test_window_step"],
                    "Max_Train_Windows_Per_Subject": mcfg["max_train_windows_per_subject"],
                    "Max_Val_Windows_Per_Subject": mcfg["max_val_windows_per_subject"],
                    "Max_Test_Windows_Per_Subject": mcfg["max_test_windows_per_subject"],
                    "LR": mcfg["lr"],
                    "Weight_Decay": mcfg["weight_decay"],
                    "lambda_spec": mcfg["lambda_spec"],
                    "lambda_wave": mcfg["lambda_wave"],
                    "lambda_corr": mcfg["lambda_corr"],
                    "lambda_grad": mcfg["lambda_grad"],
                    "lambda_smooth": mcfg["lambda_smooth"],
                    "base_dim": mcfg["base_dim"],
                    "rnn_hidden": mcfg["rnn_hidden"],
                    "dropout": mcfg["dropout"],
                    "feature_dropout": mcfg["feature_dropout"],
                }
            )
    return pd.DataFrame(rows)


def subject_tokens(value):
    text = str(value)
    tokens = {text}
    stripped = text.lstrip("0") or "0"
    tokens.add(stripped)
    if text.isdigit():
        tokens.add(str(int(text)))
    return tokens


def normalize_subject_selector(values):
    if not values:
        return None
    selector = set()
    for value in values:
        selector.update(subject_tokens(value))
    return selector


def subject_in_selection(subject, selector):
    if selector is None:
        return True
    return any(token in selector for token in subject_tokens(subject))


def save_learning_curve_plot(dataset_key, subject, histories, dirs, args):
    plt = get_pyplot()
    dataset_label = DATASET_CONFIGS[dataset_key]["label"]
    components = [
        ("total", "Total"),
        ("diff", "Diffusion"),
        ("spec", "Spectral"),
        ("wave", "Waveform"),
        ("corr", "Correlation"),
        ("grad", "Gradient"),
        ("smooth", "Curvature"),
    ]

    fig, axes = plt.subplots(4, 2, figsize=(16, 14))
    axes = axes.flatten()
    for ax, (comp_key, title) in zip(axes, components):
        for mode_key in ACTIVE_MODES:
            cfg = get_mode_config(mode_key, args)
            history = histories[mode_key]
            color = cfg["color"]
            ax.plot(history[f"train_{comp_key}"], color=color, label=f"{cfg['label']} Train")
            ax.plot(history[f"val_{comp_key}"], color=color, linestyle="--", label=f"{cfg['label']} Val")
        ax.set_title(title)
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Loss")
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=9)

    if len(axes) > len(components):
        for ax in axes[len(components) :]:
            ax.axis("off")

    fig.suptitle(f"Learning Curves - {dataset_label} - {subject}", fontsize=15)
    save_path = os.path.join(dirs["curves"], f"{safe_tag(subject)}_Learning_Curves.png")
    plt.tight_layout(rect=[0, 0, 1, 0.98])
    plt.savefig(save_path, dpi=200)
    plt.close(fig)
    return save_path


def save_waveform_comparison_plot(dataset_key, subject, fs, true_sig, mode_sigs, mode_rs, dirs):
    """GT and every mode run here (Baseline), the first 30 s."""
    plt = get_pyplot()
    plot_len = min(len(true_sig), int(PLOT_SEC * fs))
    true_norm = normalize_for_plot(true_sig[:plot_len])
    t_axis = np.arange(plot_len) / float(fs)

    plt.figure(figsize=(18, 5))
    plt.plot(t_axis, true_norm, "k-", label="Ground truth", alpha=0.90)
    for mode_key in ACTIVE_MODES:
        mode_norm = normalize_for_plot(mode_sigs[mode_key][:plot_len])
        plt.plot(t_axis, mode_norm, color=MODE_CONFIGS[mode_key]["color"], linestyle="--",
                 label=f"{MODE_CONFIGS[mode_key]['label']} (r={mode_rs[mode_key]:.3f})", alpha=0.90)
    plt.title(f"{DATASET_CONFIGS[dataset_key]['label']} - {subject} - 30 s Waveform Comparison")
    plt.xlabel("Time (s)")
    plt.ylabel("Normalized Amplitude")
    plt.xlim(0, min(PLOT_SEC, t_axis[-1] if len(t_axis) > 0 else PLOT_SEC))
    plt.grid(True, alpha=0.3)
    plt.legend(loc="upper right")
    save_path = os.path.join(dirs["plots"], f"{safe_tag(subject)}_30s_Comparison.png")
    plt.tight_layout()
    plt.savefig(save_path, dpi=200)
    plt.close()
    return save_path


def save_subject_prediction_bundle(subject, fs, true_windows, raw_pred_dict, proc_pred_dict, dirs):
    """{mode}_raw / {mode}_processed of the modes run here; arrays of other modes already in the file (e.g.
    proposed_* of an earlier v3 run, when the subject is forced) are kept."""
    save_path = os.path.join(dirs["predictions"], f"{safe_tag(subject)}_predictions.npz")
    arrays = {}
    if os.path.exists(save_path):
        try:
            with np.load(save_path, allow_pickle=False) as old:
                written = {"fs", "true"} | {f"{m}_{kind}" for m in ACTIVE_MODES for kind in ("raw", "processed")}
                arrays = {key: old[key] for key in old.files if key not in written}
        except Exception as exc:
            print(f"   [Prediction bundle] earlier file not read ({exc}); written anew")
    arrays.update({"fs": np.array(fs, dtype=np.int32), "true": true_windows.astype(np.float32)})
    for mode_key in ACTIVE_MODES:
        arrays[f"{mode_key}_raw"] = raw_pred_dict[mode_key].astype(np.float32)
        arrays[f"{mode_key}_processed"] = proc_pred_dict[mode_key].astype(np.float32)
    np.savez_compressed(save_path, **arrays)
    return save_path


def evaluate_subject_predictions(dataset_key, subject, subject_data, pred_dict, true_windows, dirs):
    fs = int(subject_data[subject]["fs"])
    n_windows = min(len(true_windows), *(len(pred_dict[m]) for m in ACTIVE_MODES))
    if n_windows == 0:
        raise RuntimeError(f"No valid evaluation windows for subject {subject}")

    proc_pred_dict = {mode_key: [] for mode_key in ACTIVE_MODES}
    proc_true = []
    rows = []
    plot_payload = None

    for i in range(n_windows):
        true_sig = true_windows[i]
        true_proc = postprocess_true_window(true_sig, fs)
        proc_true.append(true_proc)

        mode_metrics = {}
        for mode_key in ACTIVE_MODES:
            metrics = compute_window_metrics(pred_dict[mode_key][i], true_sig, fs)
            proc_pred_dict[mode_key].append(metrics["pred_proc"])
            mode_metrics[mode_key] = metrics
            rows.append(
                {
                    "Dataset": DATASET_CONFIGS[dataset_key]["label"],
                    "Subject": str(subject),
                    "Window_Index": i + 1,
                    "Model": mode_key,
                    "Model_Label": MODE_CONFIGS[mode_key]["label"],
                    "MAE": metrics["mae"],
                    "Waveform_Corr": metrics["corr"],
                }
            )

        if plot_payload is None:
            plot_payload = {
                "true": true_proc,
                "sigs": {mode_key: mode_metrics[mode_key]["pred_proc"] for mode_key in ACTIVE_MODES},
                "rs": {mode_key: mode_metrics[mode_key]["corr"] for mode_key in ACTIVE_MODES},
            }

    proc_pred_dict = {k: np.stack(v, axis=0).astype(np.float32) for k, v in proc_pred_dict.items()}
    proc_true = np.stack(proc_true, axis=0).astype(np.float32)
    plot_path = save_waveform_comparison_plot(
        dataset_key=dataset_key,
        subject=subject,
        fs=fs,
        true_sig=plot_payload["true"],
        mode_sigs=plot_payload["sigs"],
        mode_rs=plot_payload["rs"],
        dirs=dirs,
    )
    pred_path = save_subject_prediction_bundle(subject, fs, proc_true, pred_dict, proc_pred_dict, dirs)
    return pd.DataFrame(rows), plot_path, pred_path, n_windows


def summarize_subject_results(dataset_key, subject, window_df, train_info, curve_path, plot_path, pred_path, elapsed, args):
    dataset_cfg = DATASET_CONFIGS[dataset_key]
    grouped = (
        window_df.groupby(["Model", "Model_Label"], as_index=False)
        .agg(
            MAE_Mean=("MAE", "mean"),
            MAE_Std=("MAE", "std"),
            Waveform_Corr_Mean=("Waveform_Corr", "mean"),
            Waveform_Corr_Std=("Waveform_Corr", "std"),
            Num_Windows=("Window_Index", "count"),
        )
    )

    rows = []
    for _, row in grouped.iterrows():
        mode_key = row["Model"]
        cfg = get_mode_config(mode_key, args)
        info = train_info[mode_key]
        rows.append(
            {
                "Run_Tag": build_run_tag(args),
                "Speed_Preset": args.speed_preset,
                "Dataset": dataset_cfg["label"],
                "Subject": str(subject),
                "Model": mode_key,
                "Model_Label": cfg["label"],
                "Input_Features": "+".join(cfg["input_features"]),
                "Num_Input_Channels": len(cfg["input_features"]),
                "Target_FS": dataset_cfg["target_fs"],
                "Window_Sec": dataset_cfg["window_sec"],
                "Stride_Sec": dataset_cfg["stride_sec"],
                "Epochs_Config": cfg["epochs"],
                "Batch_Size": cfg["batch_size"],
                "Sample_Steps": cfg["sample_steps"],
                "Val_Every": cfg["val_every"],
                "Early_Stop_Patience_Evals": cfg["early_stop_patience_evals"],
                "Train_Window_Step": cfg["train_window_step"],
                "Val_Window_Step": cfg["val_window_step"],
                "Test_Window_Step": cfg["test_window_step"],
                "Max_Train_Windows_Per_Subject": cfg["max_train_windows_per_subject"],
                "Max_Val_Windows_Per_Subject": cfg["max_val_windows_per_subject"],
                "Max_Test_Windows_Per_Subject": cfg["max_test_windows_per_subject"],
                "LR": cfg["lr"],
                "Weight_Decay": cfg["weight_decay"],
                "lambda_spec": cfg["lambda_spec"],
                "lambda_wave": cfg["lambda_wave"],
                "lambda_corr": cfg["lambda_corr"],
                "lambda_grad": cfg["lambda_grad"],
                "lambda_smooth": cfg["lambda_smooth"],
                "Trainable_Params": info["trainable_params"],
                "Best_Epoch": info["best_epoch"],
                "Best_Val_Loss": info["best_val_loss"],
                "MAE_Mean": row["MAE_Mean"],
                "MAE_Std": row["MAE_Std"],
                "Waveform_Corr_Mean": row["Waveform_Corr_Mean"],
                "Waveform_Corr_Std": row["Waveform_Corr_Std"],
                "Num_Windows": row["Num_Windows"],
                "Elapsed_Sec": elapsed,
                "Learning_Curve_Path": curve_path,
                "Waveform_Plot_Path": plot_path,
                "Prediction_Path": pred_path,
                "Model_Path": info["model_path"],
            }
        )
    return pd.DataFrame(rows)


def build_summary_dataframe(per_subject_df):
    if per_subject_df.empty:
        return pd.DataFrame()
    summary_df = (
        per_subject_df.groupby(
            ["Run_Tag", "Speed_Preset", "Dataset", "Model", "Model_Label", "Input_Features", "Num_Input_Channels"],
            as_index=False,
        )
        .agg(
            Num_Subjects=("Subject", "nunique"),
            Target_FS=("Target_FS", "mean"),
            Window_Sec=("Window_Sec", "mean"),
            Stride_Sec=("Stride_Sec", "mean"),
            Val_Every=("Val_Every", "mean"),
            Train_Window_Step=("Train_Window_Step", "mean"),
            Val_Window_Step=("Val_Window_Step", "mean"),
            Test_Window_Step=("Test_Window_Step", "mean"),
            Trainable_Params_Mean=("Trainable_Params", "mean"),
            Best_Val_Loss_Mean=("Best_Val_Loss", "mean"),
            MAE_Mean=("MAE_Mean", "mean"),
            MAE_Std=("MAE_Mean", "std"),
            Waveform_Corr_Mean=("Waveform_Corr_Mean", "mean"),
            Waveform_Corr_Std=("Waveform_Corr_Mean", "std"),
            Num_Windows_Mean=("Num_Windows", "mean"),
            Elapsed_Sec_Mean=("Elapsed_Sec", "mean"),
        )
    )
    return summary_df


# ===========================================================
# 7. Dataset Runner
# ===========================================================
def run_dataset_comparison(dataset_key, args):
    dataset_cfg = DATASET_CONFIGS[dataset_key]
    dirs = prepare_output_dirs(args.save_root, dataset_key, args)
    print("\n" + "=" * 78)
    print(f"Dataset: {dataset_cfg['label']}")
    print(f"Save root: {dirs['root']}")
    print(f"Run tag: {build_run_tag(args)}")
    print(f"Speed preset: {args.speed_preset}")
    print(f"Modes: {ACTIVE_MODES} | stored results are kept (subjects with a Baseline result are skipped)")
    print("=" * 78)

    per_subject_df, per_window_df = load_existing_metrics(dirs["metrics"])

    subject_data = load_dataset(dataset_key, dirs)
    all_subjects = sorted(subject_data.keys())
    if len(all_subjects) < 3:
        raise RuntimeError(f"{dataset_cfg['label']}: At least 3 valid subjects are required.")

    test_selector = normalize_subject_selector(args.subjects)
    force_selector = normalize_subject_selector(args.force_subjects) or set()
    test_subjects = [s for s in all_subjects if subject_in_selection(s, test_selector)]
    if not test_subjects:
        print(f"No matching test subjects for dataset {dataset_cfg['label']}.")
        return build_summary_dataframe(per_subject_df), per_subject_df, per_window_df

    completed_subjects = sum(subject_is_complete(per_subject_df, s) for s in all_subjects)
    print(f"Existing completed subjects (Baseline): {completed_subjects}/{len(all_subjects)}")

    for test_subject in test_subjects:
        done_count = completed_mode_count(per_subject_df, test_subject)
        force_this_subject = subject_in_selection(test_subject, force_selector)
        if not force_this_subject and subject_is_complete(per_subject_df, test_subject):
            print(f"\n===== [{dataset_cfg['label']}] LOSO Test Subject: {test_subject} =====")
            print(f"Skip subject {test_subject}: already complete ({done_count}/{len(ACTIVE_MODES)} modes).")
            continue

        if done_count > 0:
            print(f"\nForced rerun of {test_subject}: its stored Baseline result is replaced (other modes kept).")
        print(f"\n===== [{dataset_cfg['label']}] LOSO Test Subject: {test_subject} =====")

        per_subject_df = drop_subject_rows(per_subject_df, test_subject)
        per_window_df = drop_subject_windows(per_window_df, test_subject)

        start_time = time.time()
        train_subjects, val_subjects = split_train_val_subjects(
            all_subjects,
            test_subject,
            val_ratio=VAL_RATIO,
            min_val_subjects=MIN_VAL_SUBJECTS,
        )

        histories = {}
        pred_dict = {}
        train_info = {}
        y_test_reference = None

        for mode_key in ACTIVE_MODES:
            output = train_one_mode(
                dataset_key=dataset_key,
                test_subject=test_subject,
                subject_data=subject_data,
                train_subjects=train_subjects,
                val_subjects=val_subjects,
                mode_key=mode_key,
                dirs=dirs,
                args=args,
            )
            histories[mode_key] = output["history"]
            pred_dict[mode_key] = output["preds"]
            train_info[mode_key] = output
            if y_test_reference is None:
                y_test_reference = output["true"]

        curve_path = save_learning_curve_plot(dataset_key, test_subject, histories, dirs, args)
        window_df, plot_path, pred_path, n_windows = evaluate_subject_predictions(
            dataset_key=dataset_key,
            subject=test_subject,
            subject_data=subject_data,
            pred_dict=pred_dict,
            true_windows=y_test_reference,
            dirs=dirs,
        )
        elapsed = time.time() - start_time
        subject_result_df = summarize_subject_results(
            dataset_key=dataset_key,
            subject=test_subject,
            window_df=window_df,
            train_info=train_info,
            curve_path=curve_path,
            plot_path=plot_path,
            pred_path=pred_path,
            elapsed=elapsed,
            args=args,
        )

        per_subject_df = concat_dataframes([per_subject_df, subject_result_df])
        per_window_df = concat_dataframes([per_window_df, window_df])
        running_summary = build_summary_dataframe(per_subject_df)
        save_dataframe_bundle(
            dirs["metrics"],
            RUNNING_PREFIX,
            {
                "ParameterSet": build_parameter_table([dataset_key], args),
                "Summary": running_summary,
                "PerSubject": per_subject_df,
                "PerWindow": per_window_df,
            },
        )
        progress_path = save_progress_json(dirs["metrics"], dataset_key, all_subjects, per_subject_df, args)

        best_line = subject_result_df[["Model_Label", "MAE_Mean", "Waveform_Corr_Mean"]].sort_values(
            ["MAE_Mean", "Waveform_Corr_Mean"], ascending=[True, False]
        )
        print(f"Saved learning curves: {curve_path}")
        print(f"Saved waveform plot: {plot_path}")
        print(f"Saved prediction bundle: {pred_path}")
        print(f"Saved progress file: {progress_path}")
        print(f"Evaluation windows: {n_windows}")
        print(best_line.to_string(index=False))
        clear_runtime_memory()

    summary_df = build_summary_dataframe(per_subject_df)
    save_dataframe_bundle(
        dirs["metrics"],
        FINAL_PREFIX,
        {
            "ParameterSet": build_parameter_table([dataset_key], args),
            "Summary": summary_df,
            "PerSubject": per_subject_df,
            "PerWindow": per_window_df,
        },
    )
    save_progress_json(dirs["metrics"], dataset_key, all_subjects, per_subject_df, args)
    return summary_df, per_subject_df, per_window_df


def replace_dataset_rows(existing_df, new_df, labels):
    """The rows of the other datasets as stored, then the new rows of the datasets run here."""
    if existing_df.empty or "Dataset" not in existing_df.columns:
        return new_df
    kept = existing_df[~existing_df["Dataset"].astype(str).isin(labels)]
    return concat_dataframes([kept, new_df])


def save_global_outputs(save_root, args, summary_frames, subject_frames, window_frames):
    combined_summary = concat_dataframes(summary_frames)
    combined_subject = concat_dataframes(subject_frames)
    combined_window = concat_dataframes(window_frames)
    if combined_summary.empty and combined_subject.empty and combined_window.empty:
        return

    combined_dir = os.path.join(save_root, "_Combined", build_run_tag(args))
    os.makedirs(combined_dir, exist_ok=True)
    labels = [DATASET_CONFIGS[k]["label"] for k in args.datasets]
    stored = {name: read_metric_csv(metric_csv_path(combined_dir, FINAL_PREFIX, name))
              for name in ("ParameterSet", "DatasetSummary", "PerSubject", "PerWindow")}
    combined_subject = replace_dataset_rows(stored["PerSubject"], combined_subject, labels)
    combined_window = replace_dataset_rows(stored["PerWindow"], combined_window, labels)
    combined_summary = replace_dataset_rows(stored["DatasetSummary"], combined_summary, labels)
    parameter_set = replace_dataset_rows(stored["ParameterSet"], build_parameter_table(args.datasets, args), labels)
    global_summary = build_summary_dataframe(combined_subject)
    save_dataframe_bundle(
        combined_dir,
        FINAL_PREFIX,
        {
            "ParameterSet": parameter_set,
            "DatasetSummary": combined_summary,
            "GlobalSummary": global_summary,
            "PerSubject": combined_subject,
            "PerWindow": combined_window,
        },
    )


# ===========================================================
# 8. Main
# ===========================================================
def parse_args():
    parser = argparse.ArgumentParser(description="RespDiff waveform model, CapnoBase only, Baseline only; stored "
                                                 "results are kept and only the remaining subjects are trained.")
    parser.add_argument(
        "--subjects",
        nargs="*",
        default=None,
        help="Optional subset of LOSO test subjects/cases to run. Training still uses the full dataset pool.",
    )
    parser.add_argument(
        "--force-subjects",
        nargs="*",
        default=None,
        help="Optional subjects/cases whose Baseline is rerun even if already stored (their Proposed rows are kept).",
    )
    parser.add_argument("--save-root", type=str, default=BASE_SAVE_ROOT, help="Base result directory.")
    parser.add_argument(
        "--speed-preset",
        type=str,
        choices=sorted(SPEED_PRESETS.keys()),
        default="balanced",
        help="Speed/quality preset. Default: balanced",
    )
    parser.add_argument("--run-tag", type=str, default=None, help="Optional custom result subfolder name.")
    parser.add_argument("--epochs", type=int, default=None, help="Override epoch count for both modes.")
    parser.add_argument("--sample-steps", type=int, default=None, help="Override diffusion sampling steps.")
    parser.add_argument("--batch-size", type=int, default=None, help="Override batch size for both modes.")
    parser.add_argument("--val-every", type=int, default=None, help="Run validation every N epochs.")
    parser.add_argument(
        "--train-window-step",
        type=int,
        default=None,
        help="Use every Nth training window per subject after preprocessing.",
    )
    parser.add_argument(
        "--val-window-step",
        type=int,
        default=None,
        help="Use every Nth validation window per subject after preprocessing.",
    )
    parser.add_argument(
        "--test-window-step",
        type=int,
        default=None,
        help="Use every Nth test window per subject during final sampling/evaluation.",
    )
    parser.add_argument(
        "--max-train-windows-per-subject",
        type=int,
        default=None,
        help="Cap the number of training windows kept per subject after stepping.",
    )
    parser.add_argument(
        "--max-val-windows-per-subject",
        type=int,
        default=None,
        help="Cap the number of validation windows kept per subject after stepping.",
    )
    parser.add_argument(
        "--max-test-windows-per-subject",
        type=int,
        default=None,
        help="Cap the number of test windows kept per subject after stepping.",
    )
    args = parser.parse_args()
    args.datasets = list(DEFAULT_ACTIVE_DATASETS)  # CapnoBase only
    return args


def validate_args(args):
    for dataset_key in args.datasets:
        if dataset_key not in DATASET_CONFIGS:
            raise KeyError(f"Unsupported dataset key: {dataset_key}")
    if args.epochs is not None and args.epochs <= 0:
        raise ValueError("--epochs must be positive.")
    if args.sample_steps is not None and args.sample_steps <= 0:
        raise ValueError("--sample-steps must be positive.")
    if args.batch_size is not None and args.batch_size <= 0:
        raise ValueError("--batch-size must be positive.")
    for arg_name in [
        "val_every",
        "train_window_step",
        "val_window_step",
        "test_window_step",
        "max_train_windows_per_subject",
        "max_val_windows_per_subject",
        "max_test_windows_per_subject",
    ]:
        value = getattr(args, arg_name)
        if value is not None and value <= 0:
            raise ValueError(f"--{arg_name.replace('_', '-')} must be positive.")


if __name__ == "__main__":
    args = parse_args()
    args = apply_speed_preset(args)
    validate_args(args)

    print(f"Device: {device}")
    print(f"AMP enabled: {AMP_ENABLED}")
    print(f"DataLoader workers: {NUM_WORKERS} | Pin memory: {PIN_MEMORY}")
    print(f"Datasets: {args.datasets} | Modes: {ACTIVE_MODES}")
    print(f"Subjects filter: {args.subjects}")
    print(f"Force subjects: {args.force_subjects}")
    print(f"Save root: {args.save_root}")
    print(f"Speed preset: {args.speed_preset}")
    print(f"Run tag: {build_run_tag(args)}")
    print(
        "Runtime config | "
        f"epochs={args.epochs}, sample_steps={args.sample_steps}, batch_size={args.batch_size}, "
        f"val_every={args.val_every}, train_window_step={args.train_window_step}, "
        f"val_window_step={args.val_window_step}, test_window_step={args.test_window_step}, "
        f"max_train={args.max_train_windows_per_subject}, "
        f"max_val={args.max_val_windows_per_subject}, "
        f"max_test={args.max_test_windows_per_subject}"
    )

    combined_summary_frames = []
    combined_subject_frames = []
    combined_window_frames = []

    for dataset_key in args.datasets:
        report_existing_runs(args.save_root, dataset_key, build_run_tag(args))
        summary_df, per_subject_df, per_window_df = run_dataset_comparison(dataset_key, args)
        if not summary_df.empty:
            combined_summary_frames.append(summary_df)
        if not per_subject_df.empty:
            combined_subject_frames.append(per_subject_df)
        if not per_window_df.empty:
            combined_window_frames.append(per_window_df)

    save_global_outputs(
        save_root=args.save_root,
        args=args,
        summary_frames=combined_summary_frames,
        subject_frames=combined_subject_frames,
        window_frames=combined_window_frames,
    )
    print("\nRespDiff CapnoBase (Baseline) finished.")
