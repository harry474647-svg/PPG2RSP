r"""261006_RRWaveNet_Ensemble: RRWaveNet-style PPG -> respiration model with a
five-model ensemble, LOSOCV and an ablation study on every dataset.

Model (the 260928 layout; no identity channel, no gating, no added module)
---------------------------------------------------------------------------
raw PPG (min-max per 10 s window) at 64 Hz
-> three stem branches: Conv1d with kernels 32 / 64 / 128 samples
   (0.5 / 1 / 2 s), 8 filters each, GroupNorm, GELU  -> 24 channels
-> 1x1 stem fusion (24 -> 24) + GroupNorm
-> four depthwise residual encoder blocks, kernel 15, dilations 1/2/4/8
   (receptive field 6.58 s)
-> decoder Conv(24 -> 16, k7, dilation 12) -> Conv(16 -> 8, k5, dilation 24)
   (receptive field 2.64 s), GroupNorm + GELU + dropout after each
-> waveform head (1x1 conv + tanh): the only output of the trained model.
During training a breath-event head (1x1 conv on the same decoder
features) carries the auxiliary event and count losses; it is removed
after training and the waveform path does not pass through it, so
removing it changes no output.
Dropout 0.10 in the encoder and decoder. Compared with 260928 only the
sampling rate (128 -> 64 Hz, so the same kernel sizes span twice the time),
the decoder dilations (1/1 -> 12/24) and the dropout (0.20 -> 0.10) differ.
9,386 parameters during training (9,377 after the event head is
removed); without the 1x1 stem fusion 8,738 (8,729) parameters.

Training
--------
Raw PPG input without a band-pass; target = respiratory-phase waveform
from the expert CO2 breath labels (CapnoBase) or the respiratory-band
reference (BIDMC, STEAM2); loss = v16 waveform loss + breath-event heatmap
BCE + soft-count loss (the last two on the training-only event head;
the validation loss that picks the epoch is the same sum). Validation is
the last 20% of every training subject's windows; the epoch with the
lowest validation loss is restored
(AdamW, learning rate 2e-4 with cosine decay, up to 240 epochs, at least 20,
patience 25). Training windows keep the v16 jitter (random shift up to
0.094 s, gain 0.95-1.05, small noise); there is no speed augmentation and
no rate-balanced sampling.

Selection of the training settings (CapnoBase 5-fold pilot, rate-ranked
folds, pooled waveform-FFT MAE over all reference-valid minutes; BIDMC and
STEAM2 were not used to choose anything):
* dropout 0.20 -> 0.10: pooled FFT MAE 2.064 -> 1.934 breaths/min (seeds 1-2,
  40 epochs; Davies mMAE 0.461 -> 0.427). Decoder dilations 10/20 (2.207) and
  dropout 0.10 with dilations 10/20 (1.970) did not improve on it, so the
  decoder keeps the dilations 12/24 of model A.
* longer training: 80 instead of 40 epochs lowered the pooled FFT MAE
  further (1.934 -> 1.786; Davies mAE 0.266 -> 0.250, mMAE 0.427 -> 0.450)
  and the best epoch stayed near the end (67-80 of 80). The LOSOCV schedule
  is doubled the same way, 120 -> 240 epochs (120-epoch LOSOCV runs of the
  earlier model reached the cap in 40 of 42 folds, best epoch 106 on average).
* model A itself (decoder dilations 12/24) was the best of the structure
  pilot: decoders with two or three layers and increasing or decreasing
  dilations, two seeds each.

Ensemble
--------
Every LOSOCV fold trains 5 models that differ only in the random seed
(initial weights, batch order, jitter and dropout). On the held-out
subject their waveform outputs are averaged and the averaged waveform
is scored. Seeds depend only on the dataset,
the window condition, the held-out subject and the member index, so every
ablation variant uses the same five seeds as the final model.

Evaluation (Davies & Mandic 2022 style; no subject is excluded)
-----------------------------------------------------------------
Every complete 60 s minute with a valid reference is scored. Breathing rate
= frequency of the largest FFT peak of the predicted waveform (3-60
breaths/min for CapnoBase, 6-30 for BIDMC and STEAM2); ground truth = the
number of reference breaths in the minute (CapnoBase expert labels;
breaths detected on the reference for BIDMC and STEAM2). Reported per
dataset:
* FFT metrics over all minutes: MAE, RMSE, bias and 95% limits of
  agreement, Pearson r, share of minutes within 1 and 2 breaths/min, and the
  mean +- SD of the per-subject MAE (as RRWaveNet reports);
* Davies & Mandic (2022): median absolute error over all minutes (mAE) and
  median of the per-subject mean absolute error (mMAE), each with its
  interquartile range;
* waveform agreement: Pearson r between the predicted and the target
  waveform in every minute (mean and median over minutes);
* every subject with its median true and predicted rate and its MAE
  (Davies & Mandic, Fig. 3b), without any grouping of subjects.

Ablation study
--------------
Each variant repeats the LOSOCV with the same seeds and the same ensemble
and changes one thing:
* full: the final model above;
* w/o ensemble: the five single models of the full model, each scored
  alone (metrics averaged over the members; no extra training);
* no_stem_fusion: the 1x1 stem fusion and its GroupNorm removed, the 24
  concatenated stem channels feed the encoder;
* dropout_0.20: dropout of model A (0.20);
* epochs_120: up to 120 epochs as for model A.
Comparison: per dataset, the FFT and Davies & Mandic metrics of every variant
side by side, and paired Wilcoxon signed-rank tests of the full model
against each variant on the per-subject MAE and on the per-minute absolute
error (Benjamini-Hochberg over the variants of one dataset).

Data and outputs
----------------
Datasets: CapnoBase, BIDMC and STEAM2 (paths in DATASET_CONFIGS, or
--capnobase-path / --bidmc-path / --steam2-ppg-dir / --steam2-rsp-dir).
All results go to D:\PPG2RSP_RRWaveNet_Inspired\261006_Ensemble:
  {variant}\01_w10s_nonoverlap\{dataset}\Members\m{k}\   models, learning curves,
      training audits and test predictions of ensemble member k
  {variant}\01_w10s_nonoverlap\{dataset}\Fold_Summaries\   per-subject JSON and
      per-minute CSV of every member and of the ensemble
  {variant}\01_w10s_nonoverlap\{dataset}\Ensemble_Predictions\   averaged outputs
  Data_Quality\   window and subject quality of every dataset
  Summary\   fft_davies_overall.csv, fft_davies_single_models.csv,
      ablation_comparison.csv, per_subject_fft.csv, per-minute tables,
      figures and 261006_RRWaveNet_Ensemble_results.xlsx

Run (GPU):                       python 261006_RRWaveNet_Ensemble.py
Resume after an interruption:    python 261006_RRWaveNet_Ensemble.py --resume
                                 (finished members are reused, an interrupted
                                 member continues from its last epoch)
Some datasets or variants:       --datasets capnobase --variants full no_stem_fusion
Parallel workers (e.g. 4):       --resume --shard 0/4 ... --shard 3/4, then
                                 --resume --summary-only
Summary of stored results only:  --summary-only
Check the data without training: --dry-run --no-require-cuda
"""

import argparse
import csv
import gc
import hashlib
import json
import math
import os
import random
import re
import time
import warnings
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.io as sio
import torch
import torch.nn as nn
from scipy import stats
from scipy.signal import (
    butter,
    correlate,
    detrend,
    filtfilt,
    find_peaks,
    resample,
    resample_poly,
    sosfiltfilt,
    welch,
)
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader, Dataset


# ---------------------------------------------------------------------------
# Paths, datasets, v16 constants, and requested experiment order
# ---------------------------------------------------------------------------

PROJECT_ROOT = Path(r"D:\PPG2RSP_RRWaveNet_Inspired\261006_Ensemble")
# Every result of this script is written under PROJECT_ROOT.
DEFAULT_RESULTS_ROOT = PROJECT_ROOT

DATASET_CONFIGS = {
    "bidmc": {
        "label": "BIDMC",
        "data_path": (
            r"D:\STEAM\Multimodal_prediction\PPG-RSP\Preprocessed"
            r"\bidmc-ppg-and-respiration-dataset-1.0.0"
            r"\bidmc-ppg-and-respiration-dataset-1.0.0\bidmc_data.mat"
        ),
        "target_fs": 128,
        "default_fs": 125,
        "profile": "adult_fixed_band",
    },
    "capnobase": {
        "label": "CapnoBase",
        "data_path": (
            r"D:\STEAM\Multimodal_prediction\PPG-RSP\Opendata"
            r"\Capnobase\data\csv"
        ),
        "target_fs": 128,
        "default_fs": 300,
        "profile": "capnography_wide_rr",
        # Expert CO2 breath annotations ({case}_8min_labels.csv) are the
        # ground truth when present; detection is the fallback.
        "expert_breath_labels": True,
        # Expert artifact spans: PPG artifacts are unusable input samples,
        # CO2 artifacts invalidate the reference (training windows and
        # scored minutes that overlap them are excluded).
        "use_artifact_labels": True,
    },
    "steam2": {
        "label": "STEAM2",
        "ppg_dir": (
            r"D:\STEAM\Multimodal_prediction\PPG-RSP\Preprocessed"
            r"\PPG\2back_raw_v2"
        ),
        "rsp_dir": (
            r"D:\STEAM\Multimodal_prediction\PPG-RSP\Preprocessed"
            r"\RSP\2back_raw_v2"
        ),
        "target_fs": 128,
        "default_fs": 128,
        "profile": "adult_fixed_band",
    },
}

DATASET_ORDER = ("capnobase", "bidmc", "steam2")
# Datasets and window conditions run when none are given on the command line.
DEFAULT_DATASETS = ("capnobase", "bidmc", "steam2")
DEFAULT_WINDOWS = ("w10s_nonoverlap",)


@dataclass(frozen=True)
class WindowSpec:
    order: int
    window_sec: float
    stride_sec: float
    name: str
    label: str


WINDOW_SPECS = (
    WindowSpec(1, 10.0, 10.0, "w10s_nonoverlap", "10 s / 0% overlap"),
    WindowSpec(2, 20.0, 20.0, "w20s_nonoverlap", "20 s / 0% overlap"),
    WindowSpec(3, 60.0, 60.0, "w60s_nonoverlap", "60 s / 0% overlap"),
)

# Model input: resampled PPG time series, min-max [0, 1] per window (no filter).
PPG_INPUT = "raw"

# v16 input/output and training constants.
TARGET_FS = 64  # model and analysis sampling rate (Hz)
EVAL_WINDOW_SEC = 60.0
EVAL_STRIDE_SEC = 60.0
BATCH_SIZE = 48
# Maximum epochs of the final model (model A: 120).
DEFAULT_EPOCHS = 240
MIN_EPOCHS = 20
PATIENCE = 25
LEARNING_RATE = 2e-4
WEIGHT_DECAY = 5e-5
LOSS_WEIGHTS = (0.45, 0.45, 0.10)
VAL_WINDOW_RATIO = 0.20

# Loss = wave * v16 hybrid + event * heatmap BCE + count * soft-count error
# (breaths/min). The count term is small because its unit is bpm and it only
# needs to calibrate the heatmap integral, not drive localization.
LOSS_WEIGHT_WAVE = 1.0
LOSS_WEIGHT_EVENT = 0.5
LOSS_WEIGHT_COUNT = 0.1  # was 0.02; count above a background threshold
# The event head starts from near-zero weights (so its first output is the
# prior) and Adam moves each weight by about one learning rate per step. At
# the base rate it cannot leave the prior within the ~840 steps a 60 s
# condition gets (7 steps/epoch x 120 epochs), so the new head uses 10x.
EVENT_HEAD_LR_MULTIPLIER = 10.0

# Breath detection and reference quality (RRest / Charlton et al. 2016).
RESP_BAND_HZ = (0.1, 0.5)
RR_BAND_BPM = (6.0, 30.0)  # RR representable in the 0.1-0.5 Hz analysis band
CTO_PEAK_FRACTION = 0.2  # count-orig: peaks above 0.2 x Q3 of all peak values
CTO_PEAK_QUANTILE = 0.75
FFT_MIN_POINTS = 16384
REF_QC_WINDOW_SEC = 32.0  # RRest window length
REF_QC_HOP_SEC = 8.0
REF_AGREEMENT_BPM = 2.0  # RRest: count-orig RR and FFT RR within 2 bpm
# Share of the breathing time that must lie in clean cycles (exactly one
# sub-zero trough between consecutive breaths). A cycle with more troughs
# skipped a breath below the count-orig threshold, so the count and the RR
# disagree and the reference is ambiguous.
REF_MIN_VALID_CYCLE_FRACTION = 0.75
BREATH_BLOCK_SEC = 60.0  # continuous breath detection in blocks with margins
BREATH_BLOCK_MARGIN_SEC = 10.0
EVENT_SIGMA_SEC = 0.3  # event width when a breath has no neighbour
# Event width proportional to the local breath period (0.12 x period).
EVENT_SIGMA_PERIOD_FRACTION = 0.12
EVENT_SIGMA_LIMITS_SEC = (0.15, 0.5)
# Count loss: heatmap counted above this background level.
EVENT_COUNT_THRESHOLD = 0.1
# v16 jitter of training windows: random shift up to 0.094 s (12 samples
# at 128 Hz), gain 0.95-1.05 and small noise.
SHIFT_AUG_SEC = 0.094
# conv1x1: 1x1 stem fusion + GroupNorm; none: ablation, the concatenated
# stem branches feed the encoder.
STEM_FUSION_MODES = ("conv1x1", "none")
EVENT_PRIOR = 0.15  # initial event probability (bias of the training-only event head)

# PPG-only quality rules. These are applied to each candidate input window.
PPG_SQI_THRESHOLD = 0.86
PPG_HR_MIN_BPM = 40.0
PPG_HR_MAX_BPM = 180.0
PPG_MAX_GAP_SEC = 3.0
PPG_MAX_INTERVAL_RATIO = 2.2
PPG_MIN_PULSES = 5
PPG_PULSE_BAND = (0.5, 5.0)
PPG_MIN_PEAK_DISTANCE_SEC = 60.0 / PPG_HR_MAX_BPM
TECHNICAL_MIN_FINITE_RATIO = 0.50
TECHNICAL_MAX_FLAT_FRACTION = 0.995
TECHNICAL_MAX_SATURATION_FRACTION = 0.80

# Dataset analysis profiles. BIDMC and STEAM2 (adults, impedance or belt
# reference) keep the v16 processing. CapnoBase differs in three documented
# ways: RR spans 5-48 bpm (median 11.7; 29 children), the reference is a
# sidestream capnogram (trapezoid waveform, 1-3 s transport delay), and
# infant heart rates can exceed 180 bpm.
ANALYSIS_PROFILES = {
    "adult_fixed_band": {
        "description": "v16 processing: fixed 0.1-0.5 Hz, sign alignment",
        "band_hz": RESP_BAND_HZ,
        "adaptive_upper": False,
        "rr_band_bpm": RR_BAND_BPM,
        "min_breath_interval_sec": 1.5,
        "ppg_hr_range_bpm": (PPG_HR_MIN_BPM, PPG_HR_MAX_BPM),
        "ppg_spectral_beat_distance": False,
        "max_target_lag_sec": 0.0,  # no lag search: v16 zero-lag sign alignment
        "fixed_target_polarity": False,
        "fft_subharmonic_check": False,
        "resampler": "fft",
    },
    "capnography_wide_rr": {
        "description": (
            "0.05 Hz to clip(0.5 x PPG heart rate, 0.5, 1.0) Hz, capnogram "
            "delay alignment with fixed polarity, 40-220 bpm PPG beats"
        ),
        # Lower edge 0.05 Hz keeps 5 bpm breathing; the upper edge follows
        # the PPG heart rate (an octave below it, so the cardiac component
        # stays strongly attenuated) up to 1.0 Hz = 60 bpm.
        "band_hz": (0.05, 1.0),
        "adaptive_upper": True,
        "rr_band_bpm": (3.0, 60.0),
        "min_breath_interval_sec": 0.8,  # up to 75 bpm
        "ppg_hr_range_bpm": (40.0, 220.0),
        "ppg_spectral_beat_distance": True,
        # Lag search window: at least 3 s (sidestream transport delay) and at
        # least half a breath period, because the PPG follows breathing
        # mechanics while CO2 follows gas exchange (about a quarter cycle
        # later), and positive-pressure ventilation inverts the PPG
        # modulation. Polarity is fixed: high CO2 is expiration.
        "max_target_lag_sec": 3.0,
        "fixed_target_polarity": True,
        # Waveform target when expert breath labels exist: "phase" is a
        # respiratory-phase waveform built from the labels (see
        # label_phase_waveform); "co2" is the band-passed capnogram.
        "capnography_target": "phase",
        "fft_subharmonic_check": True,  # trapezoid capnogram harmonics
        "resampler": "poly",
    },
}
MAX_TARGET_LAG_SEC = 15.0  # upper bound of the lag window (half of 4 bpm)
ADAPTIVE_UPPER_HR_FRACTION = 0.5
ADAPTIVE_UPPER_LIMITS_HZ = (0.5, 1.0)
# FFT RR: a spectral peak at f/2 or f/3 with at least this share of the main
# peak's power is taken as the fundamental (trapezoid capnograms have strong
# harmonics, so the largest peak can be the 2nd harmonic at slow rates).
FFT_SUBHARMONIC_POWER_RATIO = 0.3

# Network of the final model (260928 layout). Only the sampling rate
# (TARGET_FS), the decoder dilations and the dropout differ from 260928.
MODEL_CONFIG = {
    # Stem kernels 32 / 64 / 128 samples = 0.5 / 1 / 2 s at 64 Hz, 8 filters each.
    "stem_kernel_sizes": (32, 64, 128),
    "stem_base_channels": 8,
    "hidden_channels": 24,
    "encoder_kernel_size": 15,
    # Encoder receptive field 6.58 s.
    "encoder_dilations": (1, 2, 4, 8),
    "dropout": 0.10,
    "decoder_mid_channels": (16, 8),
    "decoder_kernel_sizes": (7, 5),
    # Decoder receptive field 2.64 s.
    "decoder_dilations": (12, 24),
    "decoder_heads": ("waveform_tanh", "breath_event_logit"),
}
# Model A (selected in the structure pilot) differs from the final model in
# these training settings; the ablation study reverts each changed one.
A_SETTINGS = {"dropout": 0.20, "decoder_dilations": (12, 24), "epochs": 120}
ENSEMBLE_MEMBERS = 5
# Post-fusion representation diagnostics written per member.
FEATURE_METRICS = (
    "Feature_OffDiag_SignedMean",
    "Feature_OffDiag_AbsMean",
    "Feature_EffectiveRank",
    "Feature_EffectiveRank_Normalized",
    "Feature_ParticipationRatio",
)


@dataclass(frozen=True)
class ModelOptions:
    stem_fusion: str = "conv1x1"
    dropout: float = MODEL_CONFIG["dropout"]
    decoder_dilations: Tuple[int, ...] = tuple(MODEL_CONFIG["decoder_dilations"])

    @property
    def key(self) -> str:
        return "rrwavenet_fs{}_{}_dec{}_p{:.2f}".format(
            TARGET_FS,
            "stemfusion" if self.stem_fusion == "conv1x1" else "nostemfusion",
            "-".join(str(value) for value in self.decoder_dilations),
            self.dropout,
        )

    @property
    def label(self) -> str:
        fusion = (
            "1x1 stem fusion + GroupNorm"
            if self.stem_fusion == "conv1x1"
            else "no stem fusion (concatenated stem into the encoder)"
        )
        dilations = "/".join(str(value) for value in self.decoder_dilations)
        return (
            f"3 stem branches x 8 filters, {fusion}, residual encoder, "
            f"decoder dilations {dilations}, dropout {self.dropout:.2f}"
        )

    def build(self) -> "DeepOnlyV16EventDecoder":
        return DeepOnlyV16EventDecoder(
            stem_fusion=self.stem_fusion,
            dropout=self.dropout,
            decoder_dilations=self.decoder_dilations,
        )


@dataclass(frozen=True)
class Variant:
    """One LOSOCV run of the ablation study (same seeds, same ensemble)."""

    name: str
    label: str
    options: ModelOptions
    epochs: int


def build_variants() -> Dict[str, Variant]:
    final = ModelOptions()
    variants = {
        "full": Variant("full", "Full model", final, DEFAULT_EPOCHS),
        "no_stem_fusion": Variant(
            "no_stem_fusion",
            "w/o 1x1 stem fusion",
            replace(final, stem_fusion="none"),
            DEFAULT_EPOCHS,
        ),
    }
    if final.dropout != A_SETTINGS["dropout"]:
        name = "dropout_{:.2f}".format(A_SETTINGS["dropout"])
        variants[name] = Variant(
            name,
            "dropout {:.2f} (model A)".format(A_SETTINGS["dropout"]),
            replace(final, dropout=A_SETTINGS["dropout"]),
            DEFAULT_EPOCHS,
        )
    if tuple(final.decoder_dilations) != tuple(A_SETTINGS["decoder_dilations"]):
        name = "decoder_d" + "_".join(str(v) for v in A_SETTINGS["decoder_dilations"])
        variants[name] = Variant(
            name,
            "decoder dilations {} (model A)".format(
                "/".join(str(v) for v in A_SETTINGS["decoder_dilations"])
            ),
            replace(final, decoder_dilations=tuple(A_SETTINGS["decoder_dilations"])),
            DEFAULT_EPOCHS,
        )
    if DEFAULT_EPOCHS != A_SETTINGS["epochs"]:
        name = "epochs_{}".format(A_SETTINGS["epochs"])
        variants[name] = Variant(
            name,
            "up to {} epochs (model A)".format(A_SETTINGS["epochs"]),
            final,
            A_SETTINGS["epochs"],
        )
    return variants


VARIANTS = build_variants()
# The single models of "full" scored one by one (no training of their own).
NO_ENSEMBLE_LABEL = "w/o ensemble (single models)"


# ---------------------------------------------------------------------------
# Small utilities and signal preprocessing
# ---------------------------------------------------------------------------

def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = False
        torch.backends.cudnn.benchmark = True


def stable_seed(*parts: object) -> int:
    text = "|".join(str(part) for part in parts)
    digest = hashlib.sha256(text.encode("utf-8")).hexdigest()
    return int(digest[:8], 16) % (2**31 - 1)


def safe_tag(value: object) -> str:
    return "".join(
        char if char.isalnum() or char in "-_ ." else "_"
        for char in str(value)
    ).replace(" ", "_")


def sanitize_signal(signal: np.ndarray) -> Tuple[np.ndarray, int]:
    arr = np.asarray(signal, dtype=np.float64).reshape(-1)
    if arr.size == 0:
        return arr, 0
    finite = np.isfinite(arr)
    replaced = int((~finite).sum())
    if not finite.any():
        return np.zeros_like(arr), replaced
    if replaced:
        arr = arr.copy()
        arr[~finite] = float(np.median(arr[finite]))
    return arr, replaced


def bandpass_filter(
    signal: np.ndarray,
    fs: int,
    low: float,
    high: float,
    order: int = 3,
) -> Tuple[np.ndarray, int]:
    arr, replaced = sanitize_signal(signal)
    nyquist = 0.5 * float(fs)
    low = max(float(low), 1e-5)
    high = min(float(high), nyquist * 0.99)
    if arr.size < max(32, order * 6 + 1) or high <= low:
        return arr.astype(np.float64), replaced
    b, a = butter(order, [low / nyquist, high / nyquist], btype="band")
    try:
        filtered = filtfilt(b, a, arr)
    except ValueError:
        filtered = arr
    return np.asarray(filtered, dtype=np.float64), replaced


def maybe_resample(
    signal: np.ndarray,
    source_fs: float,
    target_fs: float,
) -> np.ndarray:
    arr, _ = sanitize_signal(signal)
    if int(round(source_fs)) == int(round(target_fs)):
        return arr.astype(np.float32)
    target_len = max(1, int(round(len(arr) * target_fs / float(source_fs))))
    return resample(arr, target_len).astype(np.float32)


def maybe_resample_with_mask(
    signal: np.ndarray,
    source_fs: float,
    target_fs: float,
    method: str = "fft",
) -> Tuple[np.ndarray, np.ndarray]:
    """Resample a signal and retain a finite-sample validity mask.

    "fft" is the v16 resampler. "poly" (anti-aliased polyphase) is used for
    CapnoBase: FFT resampling treats the 8 min record as periodic and rings
    at the sharp edges of the capnogram.
    """
    raw = np.asarray(signal, dtype=np.float64).reshape(-1)
    finite = np.isfinite(raw).astype(np.float64)
    clean, _ = sanitize_signal(raw)
    if int(round(source_fs)) == int(round(target_fs)):
        return clean.astype(np.float32), finite >= 0.5
    if method == "poly":
        source, target = int(round(source_fs)), int(round(target_fs))
        divisor = math.gcd(source, target)
        up, down = target // divisor, source // divisor
        signal_out = resample_poly(clean, up, down).astype(np.float32)
        mask_out = resample_poly(finite, up, down) >= 0.5
        return signal_out, mask_out
    target_len = max(1, int(round(len(clean) * target_fs / float(source_fs))))
    signal_out = resample(clean, target_len).astype(np.float32)
    mask_out = resample(finite, target_len) >= 0.5
    return signal_out, mask_out


def resample_signal_poly(
    signal: np.ndarray,
    source_fs: float,
    target_fs: float,
) -> np.ndarray:
    arr, _ = sanitize_signal(signal)
    source_fs = int(round(float(source_fs)))
    target_fs = int(round(float(target_fs)))
    if source_fs == target_fs:
        return arr.astype(np.float32)
    divisor = math.gcd(source_fs, target_fs)
    return resample_poly(
        arr,
        target_fs // divisor,
        source_fs // divisor,
    ).astype(np.float32)


def align_target_orientation(
    ppg_low: np.ndarray,
    rsp_low: np.ndarray,
) -> np.ndarray:
    """Preserve v16 target orientation without using it for quality selection."""
    x, _ = sanitize_signal(ppg_low)
    y, _ = sanitize_signal(rsp_low)
    length = min(len(x), len(y))
    x = x[:length]
    y = y[:length]
    if length >= 2:
        x_centered = x - float(np.mean(x))
        y_centered = y - float(np.mean(y))
        if float(np.mean(x_centered * y_centered)) < 0.0:
            y = -y
    return y.astype(np.float32)


def sos_bandpass(
    signal: np.ndarray,
    fs: int,
    low: float,
    high: float,
    order: int = 3,
) -> np.ndarray:
    """Zero-phase Butterworth band-pass in second-order sections.

    Used for the wide capnography band: at 0.05 Hz and 128 Hz the transfer
    function form of the v16 filter is numerically fragile.
    """
    arr, _ = sanitize_signal(signal)
    nyquist = 0.5 * float(fs)
    high = min(float(high), nyquist * 0.99)
    if arr.size < max(32, order * 6 + 1) or high <= low:
        return arr.astype(np.float64)
    sos = butter(order, [low / nyquist, high / nyquist], btype="band", output="sos")
    try:
        return np.asarray(sosfiltfilt(sos, arr), dtype=np.float64)
    except ValueError:
        return arr.astype(np.float64)


def estimate_ppg_heart_rate_hz(
    ppg_signal: np.ndarray,
    fs: int,
    hr_range_bpm: Tuple[float, float],
) -> float:
    """Spectral heart rate of the PPG pulse band (Welch, 16 s segments)."""
    pulse, _ = bandpass_filter(ppg_signal, fs, PPG_PULSE_BAND[0], PPG_PULSE_BAND[1])
    if len(pulse) < 4 * fs or float(np.std(pulse)) < 1e-8:
        return np.nan
    freqs, power = welch(pulse, fs=fs, nperseg=min(len(pulse), int(16 * fs)))
    band = (freqs >= hr_range_bpm[0] / 60.0) & (freqs <= hr_range_bpm[1] / 60.0)
    if not band.any():
        return np.nan
    return float(freqs[band][np.argmax(power[band])])


def profile_name(profile: dict) -> str:
    for name, candidate in ANALYSIS_PROFILES.items():
        if candidate is profile:
            return name
    return "custom"


def resolve_respiratory_band(profile: dict, hr_hz: float) -> Tuple[float, float]:
    """Analysis band of one segment, from the profile and the PPG heart rate."""
    low, high = profile["band_hz"]
    if profile["adaptive_upper"]:
        if np.isfinite(hr_hz):
            high = float(
                np.clip(ADAPTIVE_UPPER_HR_FRACTION * hr_hz, *ADAPTIVE_UPPER_LIMITS_HZ)
            )
        else:
            high = ADAPTIVE_UPPER_LIMITS_HZ[0]
    return float(low), float(high)


def respiratory_filter(
    signal: np.ndarray,
    fs: int,
    band: Tuple[float, float],
    profile: dict,
) -> np.ndarray:
    """Respiratory band-pass: the v16 filter for the adult profile, SOS otherwise."""
    if not profile["adaptive_upper"] and tuple(band) == tuple(RESP_BAND_HZ):
        return bandpass_filter(signal, fs, band[0], band[1])[0]
    return sos_bandpass(signal, fs, band[0], band[1])


def label_phase_waveform(labels: np.ndarray, length: int) -> np.ndarray:
    """Respiratory-phase waveform from expert breath onsets.

    The phase rises linearly from 0 to 1 between consecutive labels (and is
    extrapolated with the first and last interval), and the waveform is
    sin(2 pi phase), so every labelled breath is one cycle whose upward zero
    crossing is exactly the label. RR needs only breath timing; the rest of
    a capnogram (trapezoid shape, inspiratory/expiratory split, EtCO2 level)
    varies between subjects and cannot be inferred from the PPG, so this
    target keeps the timing and drops what the model cannot predict.
    """
    labels = np.unique(np.asarray(labels, dtype=np.float64))
    samples = np.arange(length, dtype=np.float64)
    intervals = np.diff(labels)
    index = np.clip(np.searchsorted(labels, samples, side="right") - 1, 0, len(labels) - 2)
    phase = (samples - labels[index]) / intervals[index]
    return np.sin(2.0 * np.pi * phase).astype(np.float32)


def shift_signal(signal: np.ndarray, lag: int, fill=None) -> np.ndarray:
    """Return z with z[t] = signal[t + lag]; edges padded with `fill` or edge values."""
    arr = np.asarray(signal)
    if lag == 0 or len(arr) == 0:
        return arr.copy()
    if lag > 0:
        pad = np.full(lag, arr[-1] if fill is None else fill, dtype=arr.dtype)
        return np.concatenate([arr[lag:], pad])
    pad = np.full(-lag, arr[0] if fill is None else fill, dtype=arr.dtype)
    return np.concatenate([pad, arr[:lag]])


def align_target_lag_and_sign(
    ppg_low: np.ndarray,
    rsp_low: np.ndarray,
    fs: int,
    max_lag_sec: float,
    allow_sign_flip: bool = True,
) -> Tuple[np.ndarray, int, float, float]:
    """Delay-align the reference to the PPG respiratory component.

    The PPG modulation follows breathing mechanics; a capnogram follows gas
    exchange, about a quarter cycle later, plus the sidestream transport
    delay (1-3 s), and positive-pressure ventilation inverts the PPG
    modulation. The zero-lag correlation that v16 used to pick the sign
    therefore changes sign with each subject's RR, delay and ventilation
    mode. The lag within +-max_lag_sec with the best correlation is removed.

    With allow_sign_flip=False (capnography) only positive correlation is
    accepted: a capnogram has a physical polarity (high CO2 = expiration),
    and flipping it would also move the breath-onset targets by about half
    a cycle relative to the PPG for some subjects but not others. A window
    of at least half a breath period lets the lag alone reach every phase.
    Breath counts do not depend on this shift.

    Returns (aligned reference, lag in samples, sign, correlation); a positive
    lag means the reference lagged the PPG.
    """
    x, _ = sanitize_signal(ppg_low)
    y, _ = sanitize_signal(rsp_low)
    length = min(len(x), len(y))
    x = x[:length] - float(np.mean(x[:length]))
    y = y[:length] - float(np.mean(y[:length]))
    scale = float(np.std(x) * np.std(y))
    if length < 2 or scale < 1e-12:
        return y.astype(np.float32), 0, 1.0, np.nan
    full = correlate(y, x, mode="full", method="fft")
    lags = np.arange(-(length - 1), length)
    window = np.abs(lags) <= int(round(max_lag_sec * fs))
    values = full[window] / ((length - np.abs(lags[window])) * scale)
    best = int(np.argmax(np.abs(values) if allow_sign_flip else values))
    lag = int(lags[window][best])
    corr = float(values[best])
    sign = 1.0 if corr >= 0 or not allow_sign_flip else -1.0
    return (sign * shift_signal(y, lag)).astype(np.float32), lag, sign, corr


def instance_minmax_normalize(
    data: np.ndarray,
    feature_range: Tuple[float, float],
) -> np.ndarray:
    arr = np.asarray(data, dtype=np.float32)
    min_value = np.min(arr, axis=-1, keepdims=True)
    max_value = np.max(arr, axis=-1, keepdims=True)
    denominator = max_value - min_value
    denominator[denominator < 1e-8] = 1.0
    scaled = (arr - min_value) / denominator
    lower, upper = feature_range
    return (scaled * (upper - lower) + lower).astype(np.float32)


def extract_window_starts(
    signal_length: int,
    fs: int,
    window_sec: float,
    stride_sec: float,
) -> np.ndarray:
    window_len = int(round(window_sec * fs))
    stride_len = int(round(stride_sec * fs))
    if signal_length < window_len:
        return np.empty(0, dtype=np.int64)
    return np.arange(
        0,
        signal_length - window_len + 1,
        stride_len,
        dtype=np.int64,
    )


# ---------------------------------------------------------------------------
# Window-independent PPG and RSP quality maps
# ---------------------------------------------------------------------------

PPG_QC_UNIT_SEC = 10.0
PPG_USABLE_SQI_THRESHOLD = 0.50
MIN_USABLE_PROPORTION = 0.50


def finite_ratio(signal: np.ndarray) -> float:
    arr = np.asarray(signal).reshape(-1)
    return float(np.isfinite(arr).mean()) if arr.size else 0.0


def flat_fraction(signal: np.ndarray) -> float:
    arr, _ = sanitize_signal(signal)
    if len(arr) < 2:
        return 1.0
    robust_range = float(np.percentile(arr, 95) - np.percentile(arr, 5))
    tolerance = max(1e-12, robust_range * 1e-4)
    return float(np.mean(np.abs(np.diff(arr)) <= tolerance))


def saturation_fraction(signal: np.ndarray) -> float:
    arr, _ = sanitize_signal(signal)
    if len(arr) < 2:
        return 1.0
    low = float(np.min(arr))
    high = float(np.max(arr))
    span = max(1e-12, high - low)
    tolerance = span * 1e-6
    saturated = (np.abs(arr - low) <= tolerance) | (
        np.abs(arr - high) <= tolerance
    )
    return float(np.mean(saturated))


def pulse_template_diagnostics(
    ppg_signal: np.ndarray,
    structural_mask: np.ndarray,
    fs: int,
    hr_range_bpm: Tuple[float, float] = (PPG_HR_MIN_BPM, PPG_HR_MAX_BPM),
    min_peak_distance_sec: float = PPG_MIN_PEAK_DISTANCE_SEC,
) -> Tuple[np.ndarray, dict]:
    """Build a continuous beat-quality map from the full PPG segment.

    The defaults are the v16 adult rules. The capnography profile passes a
    40-220 bpm range and a beat spacing derived from the PPG's own spectral
    heart rate, because infants can exceed 180 bpm.
    """
    clean, _ = sanitize_signal(ppg_signal)
    pulse_band, _ = bandpass_filter(
        clean,
        fs,
        PPG_PULSE_BAND[0],
        PPG_PULSE_BAND[1],
    )
    spread = float(np.std(pulse_band))
    quality_map = np.zeros(len(clean), dtype=np.int8)
    if spread < 1e-8:
        return quality_map, {
            "PPG_SQI": np.nan,
            "PPG_Pulse_Count": 0,
            "PPG_Usable_Pulse_Count": 0,
            "PPG_High_Pulse_Count": 0,
            "PPG_Max_Pulse_Gap_Sec": np.nan,
            "PPG_Interval_Ratio": np.nan,
        }
    distance = max(1, int(round(min_peak_distance_sec * fs)))
    peaks, _ = find_peaks(
        pulse_band,
        distance=distance,
        prominence=max(1e-8, 0.10 * spread),
    )
    peaks = np.asarray(peaks, dtype=int)
    if len(peaks) < 2:
        return quality_map, {
            "PPG_SQI": np.nan,
            "PPG_Pulse_Count": int(len(peaks)),
            "PPG_Usable_Pulse_Count": 0,
            "PPG_High_Pulse_Count": 0,
            "PPG_Max_Pulse_Gap_Sec": np.nan,
            "PPG_Interval_Ratio": np.nan,
        }
    intervals = np.diff(peaks).astype(float) / float(fs)
    median_interval = float(np.median(intervals))
    heart_rate = 60.0 / max(1e-8, median_interval)
    max_gap = float(np.max(intervals))
    interval_ratio = float(np.max(intervals) / max(1e-8, np.min(intervals)))

    pulses = []
    for left, right in zip(peaks[:-1], peaks[1:]):
        if right - left < 2:
            continue
        source_x = np.linspace(0.0, 1.0, right - left, endpoint=False)
        target_x = np.linspace(0.0, 1.0, 64, endpoint=False)
        pulse = np.interp(target_x, source_x, pulse_band[left:right])
        pulse = pulse - float(np.mean(pulse))
        pulse_std = float(np.std(pulse))
        if pulse_std >= 1e-8:
            pulses.append(pulse / pulse_std)

    correlations = []
    if len(pulses) >= max(2, PPG_MIN_PULSES - 1):
        matrix = np.stack(pulses, axis=0)
        template = np.mean(matrix, axis=0)
        template_norm = np.linalg.norm(template)
        if template_norm >= 1e-8:
            for pulse in matrix:
                denominator = template_norm * np.linalg.norm(pulse)
                if denominator >= 1e-8:
                    correlations.append(float(np.dot(template, pulse) / denominator))

    correlations = np.asarray(correlations, dtype=float)
    sqi = float(np.mean(correlations)) if len(correlations) else np.nan
    interval_ok = (
        (60.0 / np.maximum(intervals, 1e-8) >= hr_range_bpm[0])
        & (60.0 / np.maximum(intervals, 1e-8) <= hr_range_bpm[1])
        & (intervals <= PPG_MAX_GAP_SEC)
        & (intervals / max(1e-8, median_interval) <= PPG_MAX_INTERVAL_RATIO)
        & (median_interval / np.maximum(intervals, 1e-8) <= PPG_MAX_INTERVAL_RATIO)
    )
    pulse_quality = np.zeros(len(intervals), dtype=np.int8)
    for index in range(len(intervals)):
        corr = correlations[index] if index < len(correlations) else np.nan
        if not interval_ok[index] or not np.isfinite(corr):
            pulse_quality[index] = 0
        elif corr >= PPG_SQI_THRESHOLD:
            pulse_quality[index] = 2
        elif corr >= PPG_USABLE_SQI_THRESHOLD:
            pulse_quality[index] = 1

    for index, (left, right) in enumerate(zip(peaks[:-1], peaks[1:])):
        if pulse_quality[index] > 0:
            quality_map[left:right] = pulse_quality[index]
    quality_map[~np.asarray(structural_mask, dtype=bool)] = 0
    return quality_map, {
        "PPG_SQI": sqi,
        "PPG_Pulse_Count": int(len(peaks)),
        "PPG_Usable_Pulse_Count": int((pulse_quality > 0).sum()),
        "PPG_High_Pulse_Count": int((pulse_quality == 2).sum()),
        "PPG_Max_Pulse_Gap_Sec": max_gap,
        "PPG_Interval_Ratio": interval_ratio,
        "PPG_Median_HR_BPM": heart_rate,
    }


def build_structural_map(
    signal: np.ndarray,
    finite_mask: np.ndarray,
    fs: int,
    unit_sec: float,
) -> np.ndarray:
    clean, _ = sanitize_signal(signal)
    valid = np.asarray(finite_mask, dtype=bool).copy()
    unit_len = max(1, int(round(unit_sec * fs)))
    for start in range(0, len(clean), unit_len):
        end = min(len(clean), start + unit_len)
        block = clean[start:end]
        block_valid = valid[start:end]
        robust_range = (
            float(np.percentile(block, 95) - np.percentile(block, 5))
            if len(block)
            else 0.0
        )
        invalid = bool(
            block_valid.mean() < TECHNICAL_MIN_FINITE_RATIO
            or len(block) < 2
            or float(np.std(block)) < 1e-8
            or robust_range < 1e-8
            or flat_fraction(block) > TECHNICAL_MAX_FLAT_FRACTION
            or saturation_fraction(block) > TECHNICAL_MAX_SATURATION_FRACTION
        )
        if invalid:
            valid[start:end] = False
    return valid


def build_ppg_quality_map(
    ppg_signal: np.ndarray,
    ppg_finite_mask: np.ndarray,
    fs: int,
    hr_range_bpm: Tuple[float, float] = (PPG_HR_MIN_BPM, PPG_HR_MAX_BPM),
    min_peak_distance_sec: float = PPG_MIN_PEAK_DISTANCE_SEC,
) -> Tuple[np.ndarray, dict]:
    structural = build_structural_map(
        ppg_signal,
        ppg_finite_mask,
        fs,
        PPG_QC_UNIT_SEC,
    )
    quality_map, diagnostics = pulse_template_diagnostics(
        ppg_signal,
        structural,
        fs,
        hr_range_bpm,
        min_peak_distance_sec,
    )
    diagnostics.update(
        {
            "PPG_Structural_Valid_Fraction": float(structural.mean())
            if len(structural)
            else 0.0,
            "PPG_Usable_Sample_Fraction": float(
                (quality_map > 0).mean()
            )
            if len(quality_map)
            else 0.0,
            "PPG_High_Sample_Fraction": float(
                (quality_map == 2).mean()
            )
            if len(quality_map)
            else 0.0,
        }
    )
    return quality_map, diagnostics


def build_rsp_quality_map(
    rsp_signal: np.ndarray,
    rsp_finite_mask: np.ndarray,
    fs: int,
) -> Tuple[np.ndarray, dict]:
    quality_map = build_structural_map(
        rsp_signal,
        rsp_finite_mask,
        fs,
        PPG_QC_UNIT_SEC,
    )
    return quality_map.astype(np.int8), {
        "RSP_Structural_Valid_Fraction": float(quality_map.mean())
        if len(quality_map)
        else 0.0,
    }


# ---------------------------------------------------------------------------
# Breath detection, reference quality and breath-event targets
# ---------------------------------------------------------------------------

def _local_extrema(x: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Strict local maxima and minima, as in RRest ref_cto."""
    slope = np.diff(x)
    peaks = np.flatnonzero((slope[:-1] > 0) & (slope[1:] < 0)) + 1
    troughs = np.flatnonzero((slope[:-1] < 0) & (slope[1:] > 0)) + 1
    return peaks, troughs


def _count_between(events: np.ndarray, marks: np.ndarray) -> np.ndarray:
    """Number of events strictly between consecutive marks."""
    return np.searchsorted(events, marks[1:]) - np.searchsorted(
        events, marks[:-1], side="right"
    )


def count_orig_breaths(
    signal: np.ndarray,
    fs: int,
) -> Tuple[np.ndarray, np.ndarray, float, float]:
    """Count-orig breath detection on one analysis window.

    Follows RRest ref_cto (Schaefer and Kratky 2008): after z-scoring, peaks
    above 0.2 x the 75th percentile of all peak values are candidate breath
    apexes, and consecutive apexes without a dip below zero between them are
    one breath (the higher apex is kept), so a hump is never a breath.

    Two changes make it valid for capnograms as well as impedance signals:
    * a cycle is clean when the signal crosses zero upwards exactly once
      between consecutive apexes (RRest asks for exactly one sub-zero
      trough, which rejects every cycle of a flat-bottomed capnogram whose
      band-passed trough is W-shaped);
    * each breath is timed at its upward zero crossing (onset) rather than
      its apex. On long capnogram plateaus the band-passed apex jumps between
      the start and the end of the plateau, which moves breaths across minute
      edges; the onset is the point RRest uses for CapnoBase (co2_zex) and
      corresponds to the expert "start of expiration" labels.

    Returns (apex indices, onset indices with -1 where the onset lies before
    the window, cycle-based RR from clean cycles, share of the breathing time
    in clean cycles).
    """
    x, _ = sanitize_signal(signal)
    empty = np.empty(0, dtype=np.int64)
    spread = float(np.std(x)) if len(x) else 0.0
    if len(x) < 3 or spread < 1e-8:
        return empty, empty, np.nan, np.nan
    z = (x - float(np.mean(x))) / spread
    peaks, troughs = _local_extrema(z)
    if len(peaks) == 0:
        return empty, empty, np.nan, np.nan
    threshold = CTO_PEAK_FRACTION * float(np.quantile(z[peaks], CTO_PEAK_QUANTILE))
    rel_peaks = peaks[z[peaks] > threshold]
    rel_troughs = troughs[z[troughs] < 0.0]
    if len(rel_peaks) == 0:
        return empty, empty, np.nan, np.nan
    apexes = [int(rel_peaks[0])]
    for peak, n_troughs in zip(rel_peaks[1:], _count_between(rel_troughs, rel_peaks)):
        if n_troughs >= 1:
            apexes.append(int(peak))
        elif z[peak] > z[apexes[-1]]:
            apexes[-1] = int(peak)
    apexes = np.asarray(apexes, dtype=np.int64)
    upward = np.flatnonzero((z[:-1] < 0.0) & (z[1:] >= 0.0)) + 1
    previous = np.concatenate([[-1], apexes[:-1]])
    onsets = np.full(len(apexes), -1, dtype=np.int64)
    for index, (apex, before) in enumerate(zip(apexes, previous)):
        candidates = upward[(upward > before) & (upward <= apex)]
        if len(candidates):
            onsets[index] = int(candidates[-1])
    if len(apexes) < 2:
        return apexes, onsets, np.nan, np.nan
    clean = _count_between(upward, apexes) == 1
    cycles = np.diff(apexes) / float(fs)
    rr = float(60.0 / np.mean(cycles[clean])) if clean.any() else np.nan
    return apexes, onsets, rr, float(cycles[clean].sum() / cycles.sum())


def fft_rr_bpm(
    signal: np.ndarray,
    fs: int,
    rr_band_bpm: Tuple[float, float] = RR_BAND_BPM,
    subharmonic_check: bool = False,
) -> float:
    """Spectral-peak RR (RRest ref_fft: detrend + Hamming) within the RR band.

    With subharmonic_check, a peak at f/2 or f/3 holding at least 30% of the
    main peak's power is taken as the fundamental: a trapezoid capnogram at
    slow rates puts more power in its 2nd harmonic than in its fundamental.
    """
    x, _ = sanitize_signal(signal)
    if len(x) < 8 or float(np.std(x)) < 1e-8:
        return np.nan
    windowed = detrend(x) * np.hamming(len(x))
    n_fft = max(FFT_MIN_POINTS, 1 << int(math.ceil(math.log2(len(x)))))
    power = np.abs(np.fft.rfft(windowed, n=n_fft)) ** 2
    freqs_bpm = np.fft.rfftfreq(n_fft, d=1.0 / float(fs)) * 60.0
    in_band = (freqs_bpm >= rr_band_bpm[0]) & (freqs_bpm <= rr_band_bpm[1])
    if not in_band.any():
        return np.nan
    best = int(np.flatnonzero(in_band)[np.argmax(power[in_band])])
    rr = float(freqs_bpm[best])
    if subharmonic_check:
        resolution = float(freqs_bpm[1] - freqs_bpm[0])
        for divisor in (3, 2):
            target = rr / divisor
            near = in_band & (np.abs(freqs_bpm - target) <= max(2 * resolution, 0.05 * target))
            if near.any() and power[near].max() >= FFT_SUBHARMONIC_POWER_RATIO * power[best]:
                return float(freqs_bpm[near][np.argmax(power[near])])
    return rr


def reference_agreement(
    signal: np.ndarray,
    fs: int,
    rr_band_bpm: Tuple[float, float] = RR_BAND_BPM,
    subharmonic_check: bool = False,
    label_times: Optional[np.ndarray] = None,
) -> Tuple[bool, str, float, float]:
    """Respiratory quality of one reference window (reference signal only).

    Without expert labels this is the RRest agreement rule: count-orig RR and
    FFT RR within 2 bpm. With expert labels (sample indices relative to the
    window) the waveform's count-orig RR must instead agree with the label
    RR, i.e. the band-passed target reproduces the annotated breaths. In both
    cases at least 75% of the breathing time must be in clean cycles and the
    RR must lie in the profile's RR band.

    Returns (valid, exclusion_reason, count-orig RR, comparison RR).
    """
    _, _, rr_cto, clean_time = count_orig_breaths(signal, fs)
    if label_times is not None:
        labels = np.sort(np.asarray(label_times, dtype=np.float64))
        rr_other = (
            float(60.0 * fs / np.mean(np.diff(labels))) if len(labels) >= 2 else np.nan
        )
    else:
        rr_other = fft_rr_bpm(signal, fs, rr_band_bpm, subharmonic_check)
    if not (np.isfinite(rr_cto) and np.isfinite(rr_other)):
        return False, "ref_rr_undefined", rr_cto, rr_other
    if clean_time < REF_MIN_VALID_CYCLE_FRACTION:
        return False, "ref_irregular_cycles", rr_cto, rr_other
    if abs(rr_cto - rr_other) >= REF_AGREEMENT_BPM:
        return False, "ref_disagreement", rr_cto, rr_other
    rr_mean = 0.5 * (rr_cto + rr_other)
    if not (rr_band_bpm[0] <= rr_mean <= rr_band_bpm[1]):
        return False, "ref_out_of_band", rr_cto, rr_other
    return True, "", rr_cto, rr_other


def build_reference_resp_quality_map(
    ref_filtered: np.ndarray,
    fs: int,
    rr_band_bpm: Tuple[float, float] = RR_BAND_BPM,
    subharmonic_check: bool = False,
    label_times: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, dict]:
    """Sample-level respiratory quality of the reference RSP.

    The quality rule is evaluated on 32 s windows every 8 s; a sample is
    usable when at least half of the windows covering it pass. Only the
    reference (and its expert labels, if any) is used, never the PPG or a
    model output.
    """
    length = len(ref_filtered)
    window = int(round(REF_QC_WINDOW_SEC * fs))
    hop = int(round(REF_QC_HOP_SEC * fs))
    votes = np.zeros(length, dtype=np.float64)
    cover = np.zeros(length, dtype=np.float64)
    reasons = {
        "ref_rr_undefined": 0,
        "ref_irregular_cycles": 0,
        "ref_disagreement": 0,
        "ref_out_of_band": 0,
    }
    passed = 0
    starts = list(range(0, length - window + 1, hop)) if length >= window else []
    if starts and starts[-1] != length - window:
        starts.append(length - window)
    for start in starts:
        window_labels = None
        if label_times is not None:
            inside = label_times[(label_times >= start) & (label_times < start + window)]
            window_labels = inside - start
        valid, reason, _, _ = reference_agreement(
            ref_filtered[start : start + window],
            fs,
            rr_band_bpm,
            subharmonic_check,
            window_labels,
        )
        cover[start : start + window] += 1.0
        if valid:
            votes[start : start + window] += 1.0
            passed += 1
        else:
            reasons[reason] += 1
    quality = (cover > 0) & (votes >= 0.5 * cover)
    diagnostics = {
        "RSP_Resp_QC_Windows": int(len(starts)),
        "RSP_Resp_QC_Pass_Windows": int(passed),
        "RSP_Resp_Quality_Fraction": float(quality.mean()) if length else 0.0,
        **{f"RSP_Resp_QC_{key}": int(value) for key, value in reasons.items()},
    }
    return quality, diagnostics


def detect_breaths_continuous(filtered: np.ndarray, fs: int) -> np.ndarray:
    """Breath onsets of a continuous filtered signal.

    Count-orig runs on 60 s blocks with 10 s of context on each side, so the
    thresholds stay local when breathing depth drifts; a breath belongs to
    the block that contains its onset (upward zero crossing). Detections
    closer than 0.5 s are merged as block-edge duplicates.
    """
    length = len(filtered)
    block = int(round(BREATH_BLOCK_SEC * fs))
    margin = int(round(BREATH_BLOCK_MARGIN_SEC * fs))
    found = []
    for block_start in range(0, length, block):
        block_end = min(length, block_start + block)
        low = max(0, block_start - margin)
        high = min(length, block_end + margin)
        _, onsets, _, _ = count_orig_breaths(filtered[low:high], fs)
        onsets = onsets[onsets >= 0] + low
        found.append(onsets[(onsets >= block_start) & (onsets < block_end)])
    if not found:
        return np.empty(0, dtype=np.int64)
    breaths = np.unique(np.concatenate(found)).astype(np.int64)
    if len(breaths) > 1:
        keep = np.concatenate([[True], np.diff(breaths) >= int(round(0.5 * fs))])
        breaths = breaths[keep]
    return breaths


def _thresholded_bump_area(threshold: float) -> float:
    """Area of max(exp(-x^2/2) - threshold, 0) for unit sigma."""
    x = np.linspace(-8.0, 8.0, 160001)
    y = np.clip(np.exp(-0.5 * x * x) - threshold, 0.0, None)
    return float(np.sum(y) * (x[1] - x[0]))


COUNT_BUMP_AREA_FACTOR = _thresholded_bump_area(EVENT_COUNT_THRESHOLD)


def event_sigmas_samples(peaks: np.ndarray, fs: float) -> np.ndarray:
    """Gaussian width of every breath: a share of its local breath period.

    The period is the mean of the intervals to the neighbouring breaths;
    sigma = EVENT_SIGMA_PERIOD_FRACTION x period, limited to
    EVENT_SIGMA_LIMITS_SEC (0.15-0.5 s), so a slow breath is one broad bump
    and the bumps of fast breaths stay apart.
    """
    peaks = np.asarray(peaks, dtype=np.float64)
    if len(peaks) < 2:
        return np.full(len(peaks), EVENT_SIGMA_SEC * float(fs))
    gaps = np.diff(peaks)
    left = np.concatenate([[gaps[0]], gaps])
    right = np.concatenate([gaps, [gaps[-1]]])
    period_sec = 0.5 * (left + right) / float(fs)
    low, high = EVENT_SIGMA_LIMITS_SEC
    return np.clip(EVENT_SIGMA_PERIOD_FRACTION * period_sec, low, high) * float(fs)


def breath_event_target_and_norm(
    peaks: np.ndarray,
    start: float,
    length: int,
    fs: float,
) -> Tuple[np.ndarray, float]:
    """Breath heatmap of one window and its count normalizer.

    Peaks may be fractional sample positions (speed-augmented windows). The
    normalizer is the thresholded area of one bump with the window's mean
    width, so sum(max(heatmap - threshold, 0)) / normalizer counts breaths.
    """
    peaks = np.sort(np.asarray(peaks, dtype=np.float64))
    default_norm = COUNT_BUMP_AREA_FACTOR * EVENT_SIGMA_SEC * float(fs)
    if len(peaks) == 0:
        return np.zeros(length, dtype=np.float32), default_norm
    sigmas = event_sigmas_samples(peaks, fs)
    reach = 4.0 * sigmas
    near = (peaks + reach >= start) & (peaks - reach < start + length)
    if not np.any(near):
        return np.zeros(length, dtype=np.float32), default_norm
    samples = np.arange(length, dtype=np.float64) + float(start)
    bumps = np.exp(
        -0.5 * ((samples[None, :] - peaks[near, None]) / sigmas[near, None]) ** 2
    )
    norm = COUNT_BUMP_AREA_FACTOR * float(np.mean(sigmas[near]))
    return bumps.max(axis=0).astype(np.float32), norm


def breath_event_target(
    peaks: np.ndarray,
    start: int,
    length: int,
    fs: int,
) -> np.ndarray:
    """Breath heatmap (height 1 at every reference breath onset)."""
    return breath_event_target_and_norm(peaks, start, length, fs)[0]


def preprocess_segment(
    ppg_raw: np.ndarray,
    rsp_raw: np.ndarray,
    fs: int,
    spec: WindowSpec,
    segment_id: int,
    ppg_valid_mask: Optional[np.ndarray] = None,
    rsp_valid_mask: Optional[np.ndarray] = None,
    profile: Optional[dict] = None,
    label_samples: Optional[np.ndarray] = None,
    ppg_artifact_mask: Optional[np.ndarray] = None,
    rsp_artifact_mask: Optional[np.ndarray] = None,
) -> Optional[dict]:
    """Quality maps, fixed ground truth and kept windows of one segment.

    `profile` is the dataset analysis profile; `label_samples` are expert
    breath onsets (sample indices at `fs`) when the dataset provides them;
    the artifact masks mark expert-labelled artifacts (True = artifact).
    """
    profile = ANALYSIS_PROFILES["adult_fixed_band"] if profile is None else profile
    ppg_source = np.asarray(ppg_raw, dtype=np.float64).reshape(-1)
    ppg_raw, _ = sanitize_signal(ppg_source)
    rsp_raw_arr = np.asarray(rsp_raw, dtype=np.float64).reshape(-1)
    length = min(len(ppg_raw), len(rsp_raw_arr))
    ppg_raw = ppg_raw[:length]
    rsp_raw_arr = rsp_raw_arr[:length]
    if ppg_valid_mask is None:
        ppg_valid_mask = np.ones(length, dtype=bool)
    else:
        ppg_valid_mask = np.asarray(ppg_valid_mask, dtype=bool)[:length]
    if rsp_valid_mask is None:
        rsp_valid_mask = np.ones(length, dtype=bool)
    else:
        rsp_valid_mask = np.asarray(rsp_valid_mask, dtype=bool)[:length]
    window_len = int(round(spec.window_sec * fs))
    if length < window_len:
        return None

    def as_artifact_mask(mask: Optional[np.ndarray]) -> np.ndarray:
        out = np.zeros(length, dtype=bool)
        if mask is not None:
            mask = np.asarray(mask, dtype=bool)[:length]
            out[: len(mask)] = mask
        return out

    # Expert-labelled artifacts are unusable samples of their signal.
    ppg_artifact = as_artifact_mask(ppg_artifact_mask)
    rsp_artifact = as_artifact_mask(rsp_artifact_mask)
    ppg_valid_mask = ppg_valid_mask & ~ppg_artifact
    rsp_valid_mask = rsp_valid_mask & ~rsp_artifact

    # Analysis band: fixed for the adult profile; for the capnography profile
    # the upper edge follows the PPG heart rate (PPG only, so it is also
    # available at inference).
    ppg_hr_hz = estimate_ppg_heart_rate_hz(ppg_raw, fs, profile["ppg_hr_range_bpm"])
    band = resolve_respiratory_band(profile, ppg_hr_hz)
    ppg_low = respiratory_filter(ppg_raw, fs, band, profile)
    rsp_low = respiratory_filter(rsp_raw_arr, fs, band, profile)
    use_phase_target = bool(
        profile.get("capnography_target") == "phase"
        and label_samples is not None
        and len(label_samples) >= 3
    )
    if profile["max_target_lag_sec"] > 0:
        # Half a breath period from the expert labels (or breaths detected on
        # the unshifted reference), never below the profile minimum.
        if label_samples is not None and len(label_samples) >= 3:
            period_sec = float(np.median(np.diff(np.sort(label_samples)))) / fs
        else:
            onsets = detect_breaths_continuous(rsp_low, fs)
            period_sec = float(np.median(np.diff(onsets))) / fs if len(onsets) >= 3 else 0.0
        max_lag_sec = float(
            np.clip(0.5 * period_sec, profile["max_target_lag_sec"], MAX_TARGET_LAG_SEC)
        )
        # The waveform target (label phase waveform or capnogram) is aligned
        # to the PPG; the capnogram reference moves by the same lag.
        unaligned_target = (
            label_phase_waveform(label_samples, length) if use_phase_target else rsp_low
        )
        target_signal, lag, sign, alignment_corr = align_target_lag_and_sign(
            ppg_low,
            unaligned_target,
            fs,
            max_lag_sec,
            allow_sign_flip=not profile["fixed_target_polarity"],
        )
        rsp_low = target_signal if not use_phase_target else (
            sign * shift_signal(rsp_low, lag)
        ).astype(np.float32)
    else:
        centered = (ppg_low - np.mean(ppg_low)) * (rsp_low - np.mean(rsp_low))
        sign = -1.0 if float(np.mean(centered)) < 0.0 else 1.0
        rsp_low = align_target_orientation(ppg_low, rsp_low)
        target_signal = rsp_low
        lag = 0
        alignment_corr = safe_pcc(ppg_low, rsp_low)
    beat_distance = (
        0.6 / ppg_hr_hz
        if profile["ppg_spectral_beat_distance"] and np.isfinite(ppg_hr_hz)
        else PPG_MIN_PEAK_DISTANCE_SEC
    )
    ppg_quality_map, ppg_diagnostics = build_ppg_quality_map(
        ppg_raw,
        ppg_valid_mask,
        fs,
        profile["ppg_hr_range_bpm"],
        beat_distance,
    )
    rsp_quality_map, rsp_diagnostics = build_rsp_quality_map(
        rsp_raw_arr,
        rsp_valid_mask,
        fs,
    )
    # The reference was shifted by `lag`; shift its validity maps with it.
    rsp_quality_map = shift_signal(rsp_quality_map, lag, fill=0)
    rsp_valid_mask = shift_signal(rsp_valid_mask, lag, fill=False)
    rsp_artifact = shift_signal(rsp_artifact, lag, fill=False)

    # Fixed ground truth: expert breath labels when available (shifted with
    # the reference), otherwise breaths detected once on the continuous
    # reference. Its respiratory quality (reference and labels only) decides
    # which windows may be used for training and validation.
    detected_breaths = detect_breaths_continuous(rsp_low, fs)
    labels = None
    if label_samples is not None and len(label_samples) >= 2:
        labels = np.round(np.asarray(label_samples, dtype=np.float64)).astype(np.int64) - lag
        labels = np.unique(labels[(labels >= 0) & (labels < length)])
    label_based = labels is not None and len(labels) >= 2
    reference_breaths = labels if label_based else detected_breaths
    resp_quality_map, resp_diagnostics = build_reference_resp_quality_map(
        rsp_low,
        fs,
        profile["rr_band_bpm"],
        profile["fft_subharmonic_check"],
        reference_breaths if label_based else None,
    )
    reference_structural = rsp_quality_map > 0
    segment_diagnostics = {
        "Segment_ID": int(segment_id),
        "Profile": profile_name(profile),
        "Resp_Band_Low_Hz": band[0],
        "Resp_Band_High_Hz": band[1],
        "PPG_HR_BPM_Spectral": float(60.0 * ppg_hr_hz) if np.isfinite(ppg_hr_hz) else np.nan,
        "Target_Lag_Sec": float(lag / float(fs)),
        "Target_Sign": float(sign),
        "Target_Alignment_Corr": float(alignment_corr),
        "Expert_Labels_Used": bool(label_based),
        "Waveform_Target": "label_phase" if use_phase_target else "reference_signal",
        "GT_Breaths": int(len(reference_breaths)),
        "Detected_Breaths": int(len(detected_breaths)),
        "PPG_Artifact_Label_Sec": float(ppg_artifact.sum() / float(fs)),
        "RSP_Artifact_Label_Sec": float(rsp_artifact.sum() / float(fs)),
    }
    train_quality_map = resp_quality_map & reference_structural
    starts = extract_window_starts(
        length,
        fs,
        spec.window_sec,
        spec.stride_sec,
    )

    model_input = (ppg_raw if PPG_INPUT == "raw" else ppg_low).astype(np.float32)
    low_kept = []
    rsp_kept = []
    event_kept = []
    event_norm_kept = []
    train_ok_kept = []
    kept_starts = []
    quality_rows = []
    for index, start in enumerate(starts):
        end = int(start) + window_len
        ppg_quality = ppg_quality_map[int(start):end]
        rsp_quality = rsp_quality_map[int(start):end]
        ppg_usable_ratio = float((ppg_quality > 0).mean())
        ppg_high_ratio = float((ppg_quality == 2).mean())
        rsp_usable_ratio = float((rsp_quality > 0).mean())
        passed = bool(
            ppg_usable_ratio >= MIN_USABLE_PROPORTION
            and rsp_usable_ratio >= MIN_USABLE_PROPORTION
        )
        resp_usable_ratio = float(train_quality_map[int(start):end].mean())
        ref_artifact_ratio = float(rsp_artifact[int(start):end].mean())
        # Labels may be missing inside a labelled reference artifact, so such
        # windows would teach the event head "no breath" there.
        train_ok = bool(
            passed
            and resp_usable_ratio >= MIN_USABLE_PROPORTION
            and ref_artifact_ratio == 0.0
        )
        diagnostics = {
            **ppg_diagnostics,
            **rsp_diagnostics,
            **resp_diagnostics,
            **segment_diagnostics,
            "PPG_Usable_Proportion": ppg_usable_ratio,
            "PPG_High_Quality_Proportion": ppg_high_ratio,
            "RSP_Usable_Proportion": rsp_usable_ratio,
            "Structural_Invalid_Proportion": float(
                1.0 - min(ppg_usable_ratio, rsp_usable_ratio)
            ),
        }
        diagnostics.update(
            {
                "Segment_ID": int(segment_id),
                "Candidate_Window_Index": int(index),
                "Window_Start_Sec": float(start / float(fs)),
                "Window_End_Sec": float(end / float(fs)),
                "Target_Finite_Ratio": float(
                    rsp_valid_mask[int(start):end].mean()
                ),
                "Target_Technical_Valid": int(
                    rsp_usable_ratio >= MIN_USABLE_PROPORTION
                ),
                "Kept": int(passed),
                "RSP_Resp_Usable_Proportion": resp_usable_ratio,
                "PPG_Artifact_Label_Proportion": float(
                    ppg_artifact[int(start):end].mean()
                ),
                "RSP_Artifact_Label_Proportion": ref_artifact_ratio,
                "Train_OK": int(train_ok),
            }
        )
        quality_rows.append(diagnostics)
        if passed:
            # Model input window: the PPG itself (min-max normalized per
            # window in the dataset) or, with --ppg-input resp_band, the
            # respiratory-band PPG of 261001_RRWaveNet_fv.
            low_kept.append(model_input[int(start):end])
            rsp_kept.append(target_signal[int(start):end])
            event_window, event_norm = breath_event_target_and_norm(
                reference_breaths, int(start), window_len, fs
            )
            event_kept.append(event_window)
            event_norm_kept.append(event_norm)
            train_ok_kept.append(train_ok)
            kept_starts.append(int(start))

    reference = {
        "signal": rsp_low.astype(np.float32),
        "breaths": reference_breaths,
        "detected_breaths": detected_breaths,
        "label_based": bool(label_based),
        "structural_valid": reference_structural,
        "artifact_mask": rsp_artifact if rsp_artifact.any() else None,
        "ppg_artifact_mask": ppg_artifact if ppg_artifact.any() else None,
        "band_hz": band,
        "profile": profile,
        "diagnostics": segment_diagnostics,
        # Continuous arrays for speed augmentation: model input, waveform
        # target and the samples usable for training (reference quality,
        # no labelled reference artifact, usable PPG).
        "model_input": model_input,
        "target_signal": np.asarray(target_signal, dtype=np.float32),
        "train_map": train_quality_map & ~rsp_artifact & (ppg_quality_map > 0),
    }
    if not low_kept:
        return {
            "low": np.empty((0, window_len), dtype=np.float32),
            "rsp": np.empty((0, window_len), dtype=np.float32),
            "event": np.empty((0, window_len), dtype=np.float32),
            "event_norm": np.empty(0, dtype=np.float32),
            "train_ok": np.empty(0, dtype=bool),
            "starts": np.empty(0, dtype=np.int64),
            "segment_ids": np.empty(0, dtype=np.int64),
            "quality": quality_rows,
            "reference": reference,
            "segment_id": int(segment_id),
            "original_length": int(length),
            "fs": int(fs),
        }
    return {
        "low": np.stack(low_kept).astype(np.float32),
        "rsp": np.stack(rsp_kept).astype(np.float32),
        "event": np.stack(event_kept).astype(np.float32),
        "event_norm": np.asarray(event_norm_kept, dtype=np.float32),
        "train_ok": np.asarray(train_ok_kept, dtype=bool),
        "starts": np.asarray(kept_starts, dtype=np.int64),
        "segment_ids": np.full(len(kept_starts), segment_id, dtype=np.int64),
        "quality": quality_rows,
        "reference": reference,
        "segment_id": int(segment_id),
        "original_length": int(length),
        "fs": int(fs),
    }


def subject_reference_diagnostics(references: Dict[int, dict], fs: int) -> dict:
    """Breathing-rate profile of one subject's ground truth.

    Reports where the subject's breathing lies relative to the v16 band
    (6-30 bpm), so subjects whose RR the v16 processing could not represent
    can be identified, and how well breath detection on the reference agrees
    with the expert labels (per-minute count error) when labels exist.
    """
    rates = []
    detection_errors = []
    label_based = False
    for reference in references.values():
        breaths = np.asarray(reference["breaths"], dtype=np.float64)
        if len(breaths) > 1:
            rates.append(60.0 * fs / np.diff(breaths))
        if reference["label_based"]:
            label_based = True
            detected = np.asarray(reference["detected_breaths"])
            minute = int(round(60.0 * fs))
            for start in range(0, len(reference["signal"]) - minute + 1, minute):
                in_labels = int(((breaths >= start) & (breaths < start + minute)).sum())
                in_detected = int(((detected >= start) & (detected < start + minute)).sum())
                detection_errors.append(abs(in_labels - in_detected))
    rates = np.concatenate(rates) if rates else np.empty(0)
    diagnostics = {
        "GT_Source": "expert_labels" if label_based else "detected_breaths",
        "GT_RR_Median_BPM": float(np.median(rates)) if len(rates) else np.nan,
        "GT_RR_P05_BPM": float(np.percentile(rates, 5)) if len(rates) else np.nan,
        "GT_RR_P95_BPM": float(np.percentile(rates, 95)) if len(rates) else np.nan,
        "GT_RR_Below_6_Fraction": float(np.mean(rates < 6.0)) if len(rates) else np.nan,
        "GT_RR_Above_30_Fraction": float(np.mean(rates > 30.0)) if len(rates) else np.nan,
        "Detection_vs_Label_Count_MAE_per_min": (
            float(np.mean(detection_errors)) if detection_errors else np.nan
        ),
    }
    segment_rows = [reference["diagnostics"] for reference in references.values()]
    for key in (
        "Resp_Band_Low_Hz",
        "Resp_Band_High_Hz",
        "PPG_HR_BPM_Spectral",
        "Target_Lag_Sec",
        "Target_Sign",
        "Target_Alignment_Corr",
    ):
        values = [row[key] for row in segment_rows if np.isfinite(row[key])]
        diagnostics[key] = float(np.mean(values)) if values else np.nan
    return diagnostics


def combine_segments(
    segments: Sequence[dict],
    spec: WindowSpec,
    fs: int,
) -> Optional[dict]:
    valid = [segment for segment in segments if len(segment["low"])]
    if not valid:
        return None
    low = np.concatenate([segment["low"] for segment in valid], axis=0)
    rsp = np.concatenate([segment["rsp"] for segment in valid], axis=0)
    event = np.concatenate([segment["event"] for segment in valid], axis=0)
    event_norm = np.concatenate([segment["event_norm"] for segment in valid], axis=0)
    train_ok = np.concatenate([segment["train_ok"] for segment in valid], axis=0)
    starts = np.concatenate([segment["starts"] for segment in valid], axis=0)
    segment_ids = np.concatenate(
        [segment["segment_ids"] for segment in valid], axis=0
    )
    quality = []
    for segment in segments:
        quality.extend(segment["quality"])
    references = {
        int(segment["segment_id"]): segment["reference"] for segment in valid
    }
    return {
        "low": low.astype(np.float32),
        "rsp": rsp.astype(np.float32),
        "event": event.astype(np.float32),
        "event_norm": event_norm.astype(np.float32),
        "train_ok": train_ok.astype(bool),
        "starts": starts.astype(np.int64),
        "segment_ids": segment_ids.astype(np.int64),
        "references": references,
        "reference_diagnostics": subject_reference_diagnostics(references, fs),
        "quality": quality,
        "fs": int(fs),
        "window_sec": float(spec.window_sec),
        "stride_sec": float(spec.stride_sec),
    }


# ---------------------------------------------------------------------------
# Dataset readers
# ---------------------------------------------------------------------------

def _unwrap_mat(value):
    arr = np.asarray(value)
    while arr.dtype == object and arr.size == 1:
        arr = np.asarray(arr.item())
    return arr


def _first_mat_payload(mat: dict):
    keys = [key for key in mat if not key.startswith("_")]
    if not keys:
        raise KeyError("No non-private MATLAB variable found")
    payload = _unwrap_mat(mat[keys[0]])
    if payload.ndim >= 2 and payload.shape[0] == 1 and payload.shape[1] == 1:
        payload = payload[0, 0]
    return payload


def _extract_signal_and_fs(
    payload,
    default_fs: int,
) -> Tuple[np.ndarray, float]:
    signal = np.asarray(
        _unwrap_mat(payload["data"]),
        dtype=np.float64,
    ).reshape(-1)
    try:
        fs = float(np.asarray(_unwrap_mat(payload["srate"])).squeeze())
    except Exception:
        fs = float(default_fs)
    return signal.astype(np.float32), fs


def spans_to_mask(spans_sec: np.ndarray, length: int, fs: float) -> np.ndarray:
    """Boolean mask (True inside any (start, end) span given in seconds)."""
    mask = np.zeros(length, dtype=bool)
    for start, end in np.asarray(spans_sec, dtype=np.float64).reshape(-1, 2):
        low = max(0, int(np.floor(start * fs)))
        high = min(length, int(np.ceil(end * fs)) + 1)
        if high > low:
            mask[low:high] = True
    return mask


def preprocess_subject_from_pairs(
    pairs: Sequence[Tuple[np.ndarray, np.ndarray, float, float]],
    spec: WindowSpec,
    profile: Optional[dict] = None,
    pair_labels: Optional[Sequence[Optional[np.ndarray]]] = None,
    pair_artifacts: Optional[Sequence[Optional[Tuple[np.ndarray, np.ndarray]]]] = None,
) -> Optional[dict]:
    """Preprocess (PPG, RSP, PPG fs, RSP fs) pairs of one subject.

    `pair_labels` holds, per pair, expert breath onsets as sample indices at
    the RSP sampling rate (or None); `pair_artifacts` holds, per pair,
    expert artifact spans in seconds as (PPG spans, RSP spans), each of
    shape (k, 2) (or None).
    """
    profile = ANALYSIS_PROFILES["adult_fixed_band"] if profile is None else profile
    segments = []
    quality_segment_id = 0
    for pair_index, (ppg, rsp, ppg_fs, rsp_fs) in enumerate(pairs):
        labels = None
        if pair_labels is not None and pair_labels[pair_index] is not None:
            labels = np.asarray(pair_labels[pair_index], dtype=np.float64) * (
                TARGET_FS / float(rsp_fs)
            )
        if int(round(ppg_fs)) != int(round(rsp_fs)):
            rsp = resample_signal_poly(rsp, rsp_fs, ppg_fs)
        ppg_target, ppg_mask = maybe_resample_with_mask(
            ppg,
            ppg_fs,
            TARGET_FS,
            profile["resampler"],
        )
        rsp_target, rsp_mask = maybe_resample_with_mask(
            rsp,
            ppg_fs,
            TARGET_FS,
            profile["resampler"],
        )
        ppg_artifact = rsp_artifact = None
        if pair_artifacts is not None and pair_artifacts[pair_index] is not None:
            ppg_spans, rsp_spans = pair_artifacts[pair_index]
            ppg_artifact = spans_to_mask(ppg_spans, len(ppg_target), TARGET_FS)
            rsp_artifact = spans_to_mask(rsp_spans, len(rsp_target), TARGET_FS)
        segment = preprocess_segment(
            ppg_target,
            rsp_target,
            TARGET_FS,
            spec,
            quality_segment_id,
            ppg_valid_mask=ppg_mask,
            rsp_valid_mask=rsp_mask,
            profile=profile,
            label_samples=labels,
            ppg_artifact_mask=ppg_artifact,
            rsp_artifact_mask=rsp_artifact,
        )
        if segment is not None:
            segments.append(segment)
        quality_segment_id += 1
    return combine_segments(segments, spec, TARGET_FS)


def load_bidmc_subjects(spec: WindowSpec) -> Dict[str, dict]:
    cfg = DATASET_CONFIGS["bidmc"]
    data = {}
    mat = sio.loadmat(cfg["data_path"])
    all_data = mat["data"][0]
    for index, subject_record in enumerate(all_data):
        subject = f"{index + 1:02d}"
        try:
            ppg = np.asarray(subject_record["ppg"][0, 0]["v"]).squeeze()
            rsp = np.asarray(
                subject_record["ref"][0, 0]["resp_sig"][0, 0]
                ["imp"][0, 0]["v"]
            ).squeeze()
            result = preprocess_subject_from_pairs(
                [(ppg, rsp, cfg["default_fs"], cfg["default_fs"])],
                spec,
                ANALYSIS_PROFILES[cfg["profile"]],
            )
            if result is not None:
                data[subject] = result
        except Exception as exc:
            print(f"[BIDMC {subject}] skipped: {exc}")
    return data


# CapnoBase IEEE TBME RR benchmark (Borealis doi:10.5683/SP2/NLB8IT), one
# group of files per case (CSV; Borealis also serves them as .tab):
#   {case}_8min_signal.csv     co2_y, pleth_y (and ecg_y), one row per sample
#   {case}_8min_param.csv      samplingrate_co2, samplingrate_pleth, ...
#   {case}_8min_labels.csv     expert annotations: co2_startexp_x,
#                              co2_startinsp_x, pleth_peak_x, *_artif_x, units_x
#   {case}_8min_meta.csv       demographics and treatment (ventilation)
#   {case}_8min_reference.csv  trends derived from the labels (RR, HR)
# MATLAB structure fields are flattened into names (labels.co2.startexp.x ->
# co2_startexp_x). An annotation field keeps all of its events in one cell
# as space-separated sample numbers with MATLAB (one-based) indexing.
CAPNOBASE_LABEL_INDEX_BASE = 1
CAPNOBASE_RECORD_SEC = 480.0
CAPNOBASE_FILE_SUFFIXES = (".csv", ".tab")
CAPNOBASE_RR_CHECK_BPM = 2.0
_NUMBER_PATTERN = re.compile(r"[-+]?(?:\d+\.?\d*|\.\d+)(?:[eE][-+]?\d+)?")


def capnobase_file(csv_dir: Path, case_id: str, kind: str) -> Optional[Path]:
    for suffix in CAPNOBASE_FILE_SUFFIXES:
        path = csv_dir / f"{case_id}_8min_{kind}{suffix}"
        if path.exists():
            return path
    return None


def _sniff_delimiter(path: Path) -> str:
    with open(path, newline="", encoding="utf-8-sig") as handle:
        first = handle.readline()
    return "\t" if first.count("\t") > first.count(",") else ","


def _normalize_field_name(name: object) -> str:
    return re.sub(r"[^0-9a-z]+", "_", str(name).lower()).strip("_")


def read_capnobase_fields(path: Path) -> Dict[str, List[str]]:
    """Fields of a CapnoBase param, meta, labels or reference file.

    Fields are read one per column (header row, then value rows) or one per
    row (name in the first cell, values after it), comma or tab separated.
    Every cell is split on whitespace, so a field whose events are stored in
    one cell (" 974 1990 ...") yields one token per event. Names are lower
    case with every non-alphanumeric run replaced by "_".
    """
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
        # A two-column "name, value" table is a row layout under a header.
        names = [row[0] for row in rows[1:] if row]
        if names and not any(_NUMBER_PATTERN.fullmatch(name) for name in names if name):
            column_layout = False
            rows = rows[1:]
    if column_layout:
        for index, name in enumerate(header):
            if name:
                fields[_normalize_field_name(name)] = [
                    token
                    for row in rows[1:]
                    if index < len(row)
                    for token in row[index].split()
                ]
    else:
        for row in rows:
            if row and row[0]:
                fields[_normalize_field_name(row[0])] = [
                    token for cell in row[1:] for token in cell.split()
                ]
    return fields


def _field_numbers(tokens: Sequence[str]) -> np.ndarray:
    return np.asarray(
        [float(match) for token in tokens for match in _NUMBER_PATTERN.findall(token)],
        dtype=np.float64,
    )


def _find_field(fields: Dict[str, List[str]], *parts: str, suffix: str = "") -> Optional[str]:
    """Shortest field name containing every part; names ending in `suffix` win."""
    names = [name for name in fields if all(part in name for part in parts)]
    if suffix:
        names = [name for name in names if name.endswith(suffix)] or names
    return min(names, key=len) if names else None


def _sampling_rate(
    fields: Dict[str, List[str]],
    signal_name: str,
    default: float,
) -> Tuple[float, str]:
    name = _find_field(fields, "samplingrate", signal_name) or _find_field(
        fields, "rate", signal_name
    )
    values = _field_numbers(fields[name]) if name else np.empty(0)
    values = values[np.isfinite(values) & (values > 0)]
    if len(values):
        return float(values[0]), name
    return float(default), "default"


def read_capnobase_signals(path: Path) -> Tuple[np.ndarray, np.ndarray, str]:
    """PPG (pleth) and capnogram (co2) columns of a signal file."""
    frame = pd.read_csv(path, sep=_sniff_delimiter(path))
    frame.columns = [_normalize_field_name(column) for column in frame.columns]

    def column(signal_name: str) -> str:
        names = [name for name in frame.columns if signal_name in name]
        names = [name for name in names if name.endswith("_y")] or names
        if not names:
            raise KeyError(f"{path.name}: no {signal_name} column in {list(frame.columns)}")
        return min(names, key=len)

    ppg_column, co2_column = column("pleth"), column("co2")
    ppg = pd.to_numeric(frame[ppg_column], errors="coerce").to_numpy(np.float32)
    co2 = pd.to_numeric(frame[co2_column], errors="coerce").to_numpy(np.float32)
    return ppg, co2, f"{ppg_column},{co2_column}"


def _artifact_spans(values: np.ndarray) -> Tuple[np.ndarray, str]:
    """(start, end) rows of an artifact field.

    The bounds are stored either interleaved (start1 end1 start2 end2 ...)
    or as all starts followed by all ends (a column-major N x 2 matrix); the
    interleaved order is the one whose values never decrease.
    """
    values = values[np.isfinite(values)]
    note = ""
    if len(values) % 2:
        values = values[:-1]
        note = "odd number of artifact bounds, last ignored"
    if len(values) < 2:
        return np.empty((0, 2)), note
    half = len(values) // 2
    if np.all(np.diff(values) >= 0):
        return values.reshape(-1, 2), note
    starts, ends = values[:half], values[half:]
    if np.all(ends >= starts) and np.all(np.diff(starts) >= 0):
        return np.column_stack([starts, ends]), note
    note = "; ".join(filter(None, [note, "artifact bounds out of order"]))
    return np.sort(values.reshape(-1, 2), axis=1), note


def read_capnobase_annotations(
    path: Path,
    ppg_fs: float,
    co2_fs: float,
    ppg_length: int,
    co2_length: int,
) -> dict:
    """Expert breaths, pulse peaks and artifact spans of one labels file.

    Event positions become zero-based sample indices of their own signal.
    The unit comes from units_x ("samples" in CapnoBase); without it, a
    breath field ending within the record duration is read as seconds. The
    one-based MATLAB index is removed unless a zero position shows that the
    file is zero-based. Breaths are the starts of expiration (the CO2
    upstroke, where the capnogram crosses zero upward after band-pass
    filtering); starts of inspiration are the fallback.
    """
    fields = read_capnobase_fields(path)
    units_name = _find_field(fields, "unit", suffix="_x")
    units_text = " ".join(fields[units_name]).lower() if units_name else ""
    events = {
        name: _field_numbers(tokens)
        for name, tokens in fields.items()
        if name.endswith("_x") and "unit" not in name
    }
    event_fields = {name: fields[name] for name in events}
    breath_name = None
    for key in ("startexp", "startinsp"):
        name = _find_field(event_fields, "co2", key)
        if name is not None and len(events[name]) >= 2:
            breath_name = name
            break
    if "sample" in units_text:
        units = "samples"
    elif "sec" in units_text or units_text.strip() == "s":
        units = "seconds"
    elif breath_name is not None and events[breath_name].max() <= CAPNOBASE_RECORD_SEC + 1.0:
        units = "seconds (inferred)"
    else:
        units = "samples (inferred)"
    zero_based = any(np.any(values == 0) for values in events.values() if len(values))
    index_base = 0 if (zero_based or units.startswith("seconds")) else CAPNOBASE_LABEL_INDEX_BASE

    warnings_list = []

    def to_samples(values: np.ndarray, fs: float) -> np.ndarray:
        values = values[np.isfinite(values)]
        return values * fs if units.startswith("seconds") else values - index_base

    def events_of(name: Optional[str], fs: float, length: int) -> np.ndarray:
        if name is None:
            return np.empty(0, dtype=np.int64)
        samples = np.round(to_samples(events[name], fs)).astype(np.int64)
        inside = (samples >= 0) & (samples < length)
        if not np.all(inside):
            warnings_list.append(
                f"{name}: {int((~inside).sum())} of {len(samples)} positions outside the record"
            )
        return np.unique(samples[inside])

    breaths = events_of(breath_name, co2_fs, co2_length)
    startinsp_name = _find_field(event_fields, "co2", "startinsp")
    pleth_peak_name = _find_field(event_fields, "pleth", "peak")
    artifacts = {"pleth": [], "co2": []}
    for name, values in events.items():
        if "artif" not in name:
            continue
        signal_name = "pleth" if "pleth" in name else "co2" if "co2" in name else None
        if signal_name is None:
            continue
        fs = ppg_fs if signal_name == "pleth" else co2_fs
        spans, note = _artifact_spans(to_samples(values, fs))
        if note:
            warnings_list.append(f"{name}: {note}")
        if len(spans):
            artifacts[signal_name].append(spans / fs)
    artifacts = {
        key: np.concatenate(spans, axis=0) if spans else np.empty((0, 2))
        for key, spans in artifacts.items()
    }
    if breath_name is None:
        warnings_list.append(f"no co2 startexp/startinsp field (fields: {list(events)[:12]})")
    elif len(breaths) < 2:
        warnings_list.append(f"{breath_name}: fewer than two breaths inside the record")
    return {
        "breaths": breaths if len(breaths) >= 2 else None,
        "breath_field": breath_name or "",
        "startinsp": events_of(startinsp_name, co2_fs, co2_length)
        if startinsp_name != breath_name
        else np.empty(0, dtype=np.int64),
        "pleth_peaks": events_of(pleth_peak_name, ppg_fs, ppg_length),
        "ppg_artifacts_sec": artifacts["pleth"],
        "co2_artifacts_sec": artifacts["co2"],
        "units": units,
        "index_base": index_base,
        "fields": sorted(events),
        "warnings": warnings_list,
    }


def read_capnobase_reference_rr(path: Path) -> Tuple[float, str]:
    """Median of the label-derived CO2 respiratory-rate trend (reference file)."""
    fields = read_capnobase_fields(path)
    name = _find_field(fields, "rr", "co2", suffix="_y")
    values = _field_numbers(fields[name]) if name else np.empty(0)
    values = values[np.isfinite(values) & (values > 0) & (values < 150)]
    return (float(np.median(values)) if len(values) else np.nan), name or ""


def _text_field(fields: Dict[str, List[str]], *parts: str) -> str:
    name = _find_field(fields, *parts)
    return " ".join(fields[name]) if name else ""


def read_capnobase_case(
    csv_dir: Path,
    case_id: str,
    default_fs: float,
    read_labels: bool = True,
) -> dict:
    """Signals, sampling rates, expert annotations and metadata of one case.

    Breath labels are zero-based sample indices at the CO2 rate, artifact
    spans are (start, end) seconds. Every assumption of the conversion
    (field names, units, index base, sampling rate source) is recorded in
    `diagnostics`, inconsistencies in `warnings`.
    """
    signal_path = capnobase_file(csv_dir, case_id, "signal")
    if signal_path is None:
        raise FileNotFoundError(f"{case_id}_8min_signal.csv not found in {csv_dir}")
    ppg, co2, signal_columns = read_capnobase_signals(signal_path)
    warnings_list = []
    param_path = capnobase_file(csv_dir, case_id, "param")
    param = read_capnobase_fields(param_path) if param_path else {}
    if param_path is None:
        warnings_list.append(f"param file missing, {default_fs:g} Hz assumed")
    ppg_fs, ppg_fs_source = _sampling_rate(param, "pleth", default_fs)
    co2_fs, co2_fs_source = _sampling_rate(param, "co2", ppg_fs)
    record_sec = len(ppg) / ppg_fs
    if abs(record_sec - CAPNOBASE_RECORD_SEC) > 2.0:
        warnings_list.append(
            f"record is {record_sec:.1f} s at {ppg_fs:g} Hz "
            f"(CapnoBase cases are {CAPNOBASE_RECORD_SEC:.0f} s): check the sampling rate"
        )
    if int(round(ppg_fs)) != int(round(co2_fs)):
        warnings_list.append(
            f"pleth {ppg_fs:g} Hz and co2 {co2_fs:g} Hz share one table of {len(ppg)} rows"
        )
    meta_path = capnobase_file(csv_dir, case_id, "meta")
    meta = read_capnobase_fields(meta_path) if meta_path else {}

    annotations = None
    labels_path = capnobase_file(csv_dir, case_id, "labels") if read_labels else None
    if read_labels and labels_path is None:
        warnings_list.append("labels file missing, breaths are detected instead")
    if labels_path is not None:
        annotations = read_capnobase_annotations(
            labels_path, ppg_fs, co2_fs, len(ppg), len(co2)
        )
        warnings_list.extend(annotations["warnings"])
    breaths = annotations["breaths"] if annotations else None
    label_rr = (
        float(60.0 * co2_fs / np.median(np.diff(breaths)))
        if breaths is not None
        else np.nan
    )
    reference_path = capnobase_file(csv_dir, case_id, "reference")
    reference_rr, reference_field = (
        read_capnobase_reference_rr(reference_path) if reference_path else (np.nan, "")
    )
    if np.isfinite(label_rr) and np.isfinite(reference_rr):
        if abs(label_rr - reference_rr) > CAPNOBASE_RR_CHECK_BPM:
            warnings_list.append(
                f"label RR {label_rr:.1f} vs reference-file RR {reference_rr:.1f} bpm: "
                "check label units and sampling rate"
            )
    empty_spans = np.empty((0, 2))
    ppg_artifacts = annotations["ppg_artifacts_sec"] if annotations else empty_spans
    co2_artifacts = annotations["co2_artifacts_sec"] if annotations else empty_spans
    age_values = [
        _field_numbers(tokens)
        for name, tokens in meta.items()
        if "age" in name.split("_")
    ]
    age_values = [values[0] for values in age_values if len(values)]
    diagnostics = {
        "CapnoBase_Signal_Columns": signal_columns,
        "CapnoBase_FS_Pleth": ppg_fs,
        "CapnoBase_FS_CO2": co2_fs,
        "CapnoBase_FS_Source": f"{ppg_fs_source},{co2_fs_source}",
        "CapnoBase_Record_Sec": float(record_sec),
        "CapnoBase_Label_Field": annotations["breath_field"] if annotations else "",
        "CapnoBase_Label_Units": annotations["units"] if annotations else "",
        "CapnoBase_Label_Index_Base": annotations["index_base"] if annotations else np.nan,
        "CapnoBase_Breath_Labels": int(len(breaths)) if breaths is not None else 0,
        "CapnoBase_Label_RR_Median_BPM": label_rr,
        "CapnoBase_Reference_RR_Median_BPM": reference_rr,
        "CapnoBase_Reference_RR_Field": reference_field,
        "CapnoBase_PPG_Artifacts": int(len(ppg_artifacts)),
        "CapnoBase_PPG_Artifact_Sec": float(np.sum(ppg_artifacts[:, 1] - ppg_artifacts[:, 0])),
        "CapnoBase_CO2_Artifacts": int(len(co2_artifacts)),
        "CapnoBase_CO2_Artifact_Sec": float(np.sum(co2_artifacts[:, 1] - co2_artifacts[:, 0])),
        "CapnoBase_Ventilation": _text_field(param, "vent") or _text_field(meta, "vent"),
        "CapnoBase_Age": float(age_values[0]) if age_values else np.nan,
        "CapnoBase_Loading_Warnings": " | ".join(warnings_list),
    }
    return {
        "ppg": ppg,
        "co2": co2,
        "ppg_fs": ppg_fs,
        "co2_fs": co2_fs,
        "breaths": breaths,
        "startinsp": annotations["startinsp"] if annotations else np.empty(0, dtype=np.int64),
        "pleth_peaks": annotations["pleth_peaks"] if annotations else np.empty(0, dtype=np.int64),
        "ppg_artifacts_sec": ppg_artifacts,
        "co2_artifacts_sec": co2_artifacts,
        "label_fields": annotations["fields"] if annotations else [],
        "diagnostics": diagnostics,
        "warnings": warnings_list,
    }


def capnobase_case_ids(csv_dir: Path) -> List[str]:
    return sorted(
        {
            path.name.split("_")[0]
            for suffix in CAPNOBASE_FILE_SUFFIXES
            for path in csv_dir.glob(f"*_8min_signal{suffix}")
        }
    )


def load_capnobase_subjects(spec: WindowSpec) -> Dict[str, dict]:
    cfg = DATASET_CONFIGS["capnobase"]
    data = {}
    csv_dir = Path(cfg["data_path"])
    for case_id in capnobase_case_ids(csv_dir):
        try:
            case = read_capnobase_case(
                csv_dir,
                case_id,
                cfg["default_fs"],
                read_labels=bool(cfg.get("expert_breath_labels")),
            )
            for message in case["warnings"]:
                print(f"[CapnoBase {case_id}] {message}")
            artifacts = (
                [(case["ppg_artifacts_sec"], case["co2_artifacts_sec"])]
                if cfg.get("use_artifact_labels")
                else None
            )
            result = preprocess_subject_from_pairs(
                [(case["ppg"], case["co2"], case["ppg_fs"], case["co2_fs"])],
                spec,
                ANALYSIS_PROFILES[cfg["profile"]],
                [case["breaths"]],
                artifacts,
            )
            if result is not None:
                result["reference_diagnostics"].update(case["diagnostics"])
                data[case_id] = result
        except Exception as exc:
            print(f"[CapnoBase {case_id}] skipped: {exc}")
    return data


def capnobase_label_checks(case: dict) -> dict:
    """Checks that the labels sit where the signals say they should.

    A start of expiration is where CO2 rises (mean of the 0.5 s after the
    label above the 0.5 s before it), a start of inspiration is where it
    falls, the two alternate, and an expert pulse peak is the PPG maximum
    within +-0.1 s (offset 0 means index base and sampling rate agree).
    """
    co2 = np.asarray(case["co2"], dtype=np.float64)
    ppg = np.asarray(case["ppg"], dtype=np.float64)
    half = int(round(0.5 * case["co2_fs"]))

    def co2_step(labels: np.ndarray) -> np.ndarray:
        labels = labels[(labels >= half) & (labels < len(co2) - half)]
        return np.asarray(
            [
                np.nanmean(co2[index : index + half]) - np.nanmean(co2[index - half : index])
                for index in labels
            ]
        )

    breaths = case["breaths"] if case["breaths"] is not None else np.empty(0, dtype=np.int64)
    exp_step = co2_step(breaths)
    insp_step = co2_step(case["startinsp"])
    alternation = np.nan
    if len(breaths) > 1 and len(case["startinsp"]):
        between = np.searchsorted(case["startinsp"], breaths[1:]) - np.searchsorted(
            case["startinsp"], breaths[:-1]
        )
        alternation = float(np.mean(between == 1))
    reach = int(round(0.1 * case["ppg_fs"]))
    offsets = [
        int(np.nanargmax(ppg[p - reach : p + reach + 1])) - reach
        for p in case["pleth_peaks"]
        if reach <= p < len(ppg) - reach and np.any(np.isfinite(ppg[p - reach : p + reach + 1]))
    ]
    offsets = np.asarray(offsets, dtype=np.float64)
    return {
        "Check_CO2_Rises_At_Breath_Label": float(np.mean(exp_step > 0)) if len(exp_step) else np.nan,
        "Check_CO2_Falls_At_StartInsp": float(np.mean(insp_step < 0)) if len(insp_step) else np.nan,
        "Check_One_StartInsp_Per_Breath": alternation,
        "Check_Pleth_Peak_Labels": int(len(case["pleth_peaks"])),
        "Check_Pleth_Peak_Offset_Median_Samples": float(np.median(offsets)) if len(offsets) else np.nan,
        "Check_Pleth_Peak_Within_10ms": float(
            np.mean(np.abs(offsets) <= 0.01 * case["ppg_fs"])
        )
        if len(offsets)
        else np.nan,
        "Check_Label_HR_Median_BPM": float(
            60.0 * case["ppg_fs"] / np.median(np.diff(case["pleth_peaks"]))
        )
        if len(case["pleth_peaks"]) > 1
        else np.nan,
    }


def inspect_capnobase(results_root: Path) -> pd.DataFrame:
    """Print and save how every CapnoBase case is read (no training).

    Writes capnobase_loading_check.csv. A correct load shows 300 Hz, a
    480 s record, label RR equal to the reference-file RR, CO2 rising at
    nearly every breath label, and a median pulse-peak offset of 0 samples.
    """
    cfg = DATASET_CONFIGS["capnobase"]
    csv_dir = Path(cfg["data_path"])
    case_ids = capnobase_case_ids(csv_dir)
    print(f"CapnoBase directory: {csv_dir} ({len(case_ids)} cases)")
    rows = []
    for case_id in case_ids:
        try:
            case = read_capnobase_case(csv_dir, case_id, cfg["default_fs"])
            row = {
                "Case": case_id,
                **case["diagnostics"],
                **capnobase_label_checks(case),
                "CapnoBase_Label_Fields": " ".join(case["label_fields"]),
            }
        except Exception as exc:
            row = {"Case": case_id, "CapnoBase_Loading_Warnings": f"failed: {exc}"}
        rows.append(row)
    frame = pd.DataFrame(rows)
    if frame.empty:
        print("No {case}_8min_signal.csv files found.")
        return frame
    results_root.mkdir(parents=True, exist_ok=True)
    path = results_root / "capnobase_loading_check.csv"
    frame.to_csv(path, index=False, encoding="utf-8-sig")
    shown = {
        "Case": "case",
        "CapnoBase_FS_Pleth": "fs",
        "CapnoBase_Record_Sec": "sec",
        "CapnoBase_Breath_Labels": "breaths",
        "CapnoBase_Label_RR_Median_BPM": "RR_lab",
        "CapnoBase_Reference_RR_Median_BPM": "RR_ref",
        "Check_CO2_Rises_At_Breath_Label": "CO2up",
        "Check_Pleth_Peak_Offset_Median_Samples": "pk_off",
        "CapnoBase_PPG_Artifact_Sec": "ppg_art_s",
        "CapnoBase_CO2_Artifact_Sec": "co2_art_s",
        "CapnoBase_Ventilation": "vent",
    }
    table = frame[[column for column in shown if column in frame.columns]].rename(columns=shown)
    with pd.option_context("display.max_rows", None, "display.width", 160):
        print(table.to_string(index=False, float_format=lambda value: f"{value:.2f}"))
    warned = frame[frame["CapnoBase_Loading_Warnings"].fillna("").astype(str) != ""]
    for _, row in warned.iterrows():
        print(f"[CapnoBase {row['Case']}] {row['CapnoBase_Loading_Warnings']}")
    print(f"Saved: {path}")
    return frame


def _mat_stem(filename: str) -> str:
    return os.path.splitext(filename)[0]


def _match_mat_file_pairs(
    ppg_dir: Path,
    rsp_dir: Path,
) -> List[Tuple[str, str]]:
    ppg_files = sorted(path.name for path in ppg_dir.glob("*.mat"))
    rsp_files = sorted(path.name for path in rsp_dir.glob("*.mat"))
    ppg_map = {_mat_stem(name): name for name in ppg_files}
    rsp_map = {_mat_stem(name): name for name in rsp_files}
    common = sorted(set(ppg_map).intersection(rsp_map))
    if common:
        return [(ppg_map[key], rsp_map[key]) for key in common]
    if len(ppg_files) != len(rsp_files):
        raise RuntimeError(
            "PPG/RSP MAT file counts differ and no common stems exist"
        )
    return list(zip(ppg_files, rsp_files))


def load_steam2_subjects(spec: WindowSpec) -> Dict[str, dict]:
    cfg = DATASET_CONFIGS["steam2"]
    data = {}
    ppg_root = Path(cfg["ppg_dir"])
    rsp_root = Path(cfg["rsp_dir"])
    for subject_dir in sorted(
        path for path in ppg_root.iterdir() if path.is_dir()
    ):
        rsp_subject_dir = rsp_root / subject_dir.name
        if not rsp_subject_dir.is_dir():
            continue
        pairs = []
        try:
            for ppg_name, rsp_name in _match_mat_file_pairs(
                subject_dir,
                rsp_subject_dir,
            ):
                p_payload = _first_mat_payload(
                    sio.loadmat(subject_dir / ppg_name)
                )
                r_payload = _first_mat_payload(
                    sio.loadmat(rsp_subject_dir / rsp_name)
                )
                ppg, ppg_fs = _extract_signal_and_fs(
                    p_payload,
                    cfg["default_fs"],
                )
                rsp, rsp_fs = _extract_signal_and_fs(
                    r_payload,
                    cfg["default_fs"],
                )
                pairs.append((ppg, rsp, ppg_fs, rsp_fs))
            result = preprocess_subject_from_pairs(
                pairs,
                spec,
                ANALYSIS_PROFILES[cfg["profile"]],
            )
            if result is not None:
                data[subject_dir.name] = result
        except Exception as exc:
            print(f"[STEAM2 {subject_dir.name}] skipped: {exc}")
    return data


def load_dataset(dataset_key: str, spec: WindowSpec) -> Dict[str, dict]:
    if dataset_key == "bidmc":
        return load_bidmc_subjects(spec)
    if dataset_key == "capnobase":
        return load_capnobase_subjects(spec)
    if dataset_key == "steam2":
        return load_steam2_subjects(spec)
    raise KeyError(dataset_key)


# ---------------------------------------------------------------------------
# LOSOCV window datasets
# ---------------------------------------------------------------------------

def select_subject_indices(
    n_windows: int,
    split: str,
) -> np.ndarray:
    if split == "test":
        return np.arange(n_windows, dtype=int)
    val_count = (
        max(1, int(round(n_windows * VAL_WINDOW_RATIO)))
        if n_windows > 1
        else 0
    )
    val_count = min(val_count, max(0, n_windows - 1))
    if split == "train":
        return np.arange(0, n_windows - val_count, dtype=int)
    if split == "val":
        return (
            np.arange(n_windows - val_count, n_windows, dtype=int)
            if val_count
            else np.empty(0, dtype=int)
        )
    raise ValueError(split)


def test_metadata(
    subject_data: Dict[str, dict],
    test_subject: str,
) -> List[dict]:
    record = subject_data[test_subject]
    return [
        {
            "segment_id": int(segment_id),
            "start": int(start),
        }
        for segment_id, start in zip(
            record["segment_ids"],
            record["starts"],
        )
    ]


def release_memory(device: Optional[torch.device] = None) -> None:
    """Release objects from completed folds before the next LOSOCV fold."""
    gc.collect()
    if device is not None and device.type == "cuda":
        torch.cuda.empty_cache()


class SubjectWindowDataset(Dataset):
    """Lazy window view that avoids train/validation/test array duplication.

    Splits: "train"/"val" are the temporal split of every subject (last 20%
    of its kept windows for validation, Train_OK only); "test" is every kept
    window (inference never depends on the reference). Training windows get
    the v16 jitter (random shift up to SHIFT_AUG_SEC, gain 0.95-1.05, small
    noise).
    """

    def __init__(
        self,
        subject_data: Dict[str, dict],
        subjects: Sequence[str],
        split: str,
        augment: bool,
    ):
        self.subject_data = subject_data
        self.samples = []
        for subject in subjects:
            record = subject_data[subject]
            indices = select_subject_indices(len(record["low"]), split)
            if split in ("train", "val"):
                indices = indices[record["train_ok"][indices]]
            self.samples.extend((subject, int(index)) for index in indices)
        if not self.samples:
            raise RuntimeError(f"No windows available for split={split}")
        self.augment = augment

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int):
        subject, window_index = self.samples[index]
        record = self.subject_data[subject]
        x = record["low"][window_index]
        y = record["rsp"][window_index]
        e = record["event"][window_index]
        norm = float(record["event_norm"][window_index])
        x = instance_minmax_normalize(np.asarray(x, dtype=np.float32)[None, :], (0.0, 1.0))
        y = instance_minmax_normalize(np.asarray(y, dtype=np.float32)[None, :], (-1.0, 1.0))[0]
        x_tensor = torch.from_numpy(np.asarray(x, dtype=np.float32).copy())
        y_tensor = torch.from_numpy(np.asarray(y, dtype=np.float32).copy())
        e_tensor = torch.from_numpy(np.asarray(e, dtype=np.float32).copy())
        if self.augment:
            reach = int(round(SHIFT_AUG_SEC * float(record["fs"])))
            shift = random.randint(-reach, reach)
            scale = random.uniform(0.95, 1.05)
            noise_std = random.uniform(0.002, 0.01)
            if shift:
                x_tensor = torch.roll(x_tensor, shifts=shift, dims=-1)
                y_tensor = torch.roll(y_tensor, shifts=shift, dims=-1)
                e_tensor = torch.roll(e_tensor, shifts=shift, dims=-1)
            x_tensor = x_tensor * scale + torch.randn_like(x_tensor) * noise_std
        return (
            x_tensor,
            y_tensor.unsqueeze(0),
            e_tensor.unsqueeze(0),
            torch.tensor(norm, dtype=torch.float32),
        )


def make_train_loader(
    dataset: SubjectWindowDataset,
    batch_size: int,
    device: torch.device,
) -> DataLoader:
    return DataLoader(
        dataset,
        batch_size=int(batch_size),
        shuffle=True,
        drop_last=False,
        num_workers=0,
        pin_memory=device.type == "cuda",
    )


def make_eval_loader(dataset: SubjectWindowDataset, batch_size: int, device: torch.device) -> DataLoader:
    return DataLoader(
        dataset,
        batch_size=int(batch_size),
        shuffle=False,
        drop_last=False,
        num_workers=0,
        pin_memory=device.type == "cuda",
    )


# ---------------------------------------------------------------------------
# v16 Deep-only building blocks (unchanged)
# ---------------------------------------------------------------------------

def choose_group_count(channels: int) -> int:
    for groups in (8, 4, 2):
        if channels % groups == 0:
            return groups
    return 1


class ResidualConvBlock(nn.Module):
    """v16 depthwise dilated encoder block: GELU(block(x) + x)."""

    def __init__(
        self,
        channels: int,
        kernel_size: int,
        dilation: int,
        dropout: float,
    ):
        super().__init__()
        padding = int(dilation) * (int(kernel_size) // 2)
        groups = int(channels)
        self.conv1 = nn.Conv1d(
            channels,
            channels,
            kernel_size,
            padding=padding,
            dilation=dilation,
            groups=groups,
        )
        self.norm1 = nn.GroupNorm(channels, channels)
        self.conv2 = nn.Conv1d(
            channels,
            channels,
            kernel_size,
            padding=padding,
            dilation=dilation,
            groups=groups,
        )
        self.norm2 = nn.GroupNorm(channels, channels)
        self.act = nn.GELU()
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.drop(self.act(self.norm1(self.conv1(x))))
        out = self.drop(self.norm2(self.conv2(out)))
        return self.act(out + x)


class Conv1dExplicitSame(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
    ):
        super().__init__()
        total_pad = int(kernel_size) - 1
        self.pad = nn.ConstantPad1d(
            (total_pad // 2, total_pad - total_pad // 2),
            0.0,
        )
        self.conv = nn.Conv1d(
            in_channels,
            out_channels,
            kernel_size,
            padding=0,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(self.pad(x))


def make_encoder(channels: int, dropout: float) -> nn.Sequential:
    return nn.Sequential(
        *[
            ResidualConvBlock(
                channels,
                MODEL_CONFIG["encoder_kernel_size"],
                dilation,
                dropout,
            )
            for dilation in MODEL_CONFIG["encoder_dilations"]
        ]
    )


# ---------------------------------------------------------------------------
# Multi-scale stem
# ---------------------------------------------------------------------------

class StemBranch(nn.Module):
    """v16 stem branch: explicit-same Conv1d -> GroupNorm -> GELU."""

    def __init__(self, stem_channels: int, kernel_size: int):
        super().__init__()
        self.conv = Conv1dExplicitSame(1, stem_channels, kernel_size)
        self.norm = nn.GroupNorm(
            choose_group_count(stem_channels),
            stem_channels,
        )
        self.act = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(self.norm(self.conv(x)))


# ---------------------------------------------------------------------------
# Decoder and network
# ---------------------------------------------------------------------------

def make_breath_decoder_trunk(
    channels: int,
    dilations: Sequence[int],
    dropout: float,
) -> nn.Sequential:
    """v16 decoder layers (24 -> 16 -> 8 channels) with dilated convolutions."""
    width = int(channels)
    mids = tuple(MODEL_CONFIG["decoder_mid_channels"])
    kernels = tuple(MODEL_CONFIG["decoder_kernel_sizes"])
    dilation_list = tuple(dilations)
    if not len(kernels) == len(dilation_list) == len(mids):
        raise ValueError("decoder channels, kernels and dilations differ in length")
    layers = []
    # Conv -> GroupNorm -> GELU -> Dropout per layer, as in the v16 decoder.
    for out_channels, kernel, dilation in zip(mids, kernels, dilation_list):
        layers += [
            nn.Conv1d(
                width,
                out_channels,
                kernel,
                padding=dilation * (kernel // 2),
                dilation=dilation,
            ),
            nn.GroupNorm(choose_group_count(out_channels), out_channels),
            nn.GELU(),
            nn.Dropout(dropout),
        ]
        width = out_channels
    return nn.Sequential(*layers)


class DeepOnlyV16EventDecoder(nn.Module):
    """260928 early stem fusion network with the breath-event decoder.

    three stem branches -> concatenation (24 channels) -> 1x1 fusion +
    GroupNorm (no GELU) -> depthwise residual encoder -> dilated decoder
    trunk -> waveform head (tanh). The breath-event head (1x1 conv on the
    decoder features) exists for the auxiliary training losses and is
    removed after training (remove_event_head). With stem_fusion="none"
    (ablation) the concatenated stem channels feed the encoder directly.
    """

    def __init__(
        self,
        stem_fusion: str = "conv1x1",
        dropout: Optional[float] = None,
        decoder_dilations: Optional[Sequence[int]] = None,
    ):
        super().__init__()
        if stem_fusion not in STEM_FUSION_MODES:
            raise ValueError(f"stem_fusion must be one of {STEM_FUSION_MODES}")
        self.stem_fusion = stem_fusion
        stem_channels = int(MODEL_CONFIG["stem_base_channels"])
        hidden = MODEL_CONFIG["hidden_channels"]
        kernel_sizes = tuple(MODEL_CONFIG["stem_kernel_sizes"])
        dropout = float(MODEL_CONFIG["dropout"] if dropout is None else dropout)
        dilations = tuple(
            MODEL_CONFIG["decoder_dilations"]
            if decoder_dilations is None
            else decoder_dilations
        )
        self.stem_base_channels = stem_channels
        self.branch_labels = tuple(f"k{size}" for size in kernel_sizes)
        self.stem_branches = nn.ModuleList(
            [StemBranch(stem_channels, size) for size in kernel_sizes]
        )
        stem_width = stem_channels * len(kernel_sizes)
        if stem_fusion == "conv1x1":
            self.stem_fuse = nn.Sequential(
                nn.Conv1d(stem_width, hidden, 1),
                nn.GroupNorm(choose_group_count(hidden), hidden),
            )
            width = hidden
        else:
            # Ablation: no 1x1 fusion and no GroupNorm after it. Every stem
            # branch already ends in GroupNorm + GELU, so the concatenation
            # is the encoder input.
            self.stem_fuse = nn.Identity()
            width = stem_width
        self.feature_channels = width
        self.encoder = make_encoder(width, dropout)
        self.decoder = make_breath_decoder_trunk(width, dilations, dropout)
        decoder_channels = MODEL_CONFIG["decoder_mid_channels"][-1]
        self.wave_head = nn.Conv1d(decoder_channels, 1, 1)
        self.event_head = nn.Conv1d(decoder_channels, 1, 1)
        self._init_weights()
        self._init_event_head()

    def _init_weights(self) -> None:
        for module in self.modules():
            if isinstance(module, (nn.Conv1d, nn.Linear)):
                nn.init.kaiming_normal_(
                    module.weight,
                    mode="fan_out",
                    nonlinearity="relu",
                )
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0.0)

    def _init_event_head(self) -> None:
        """Start the event head at the prior breath-apex probability.

        Small weights (std 0.01, as for detection heads in RetinaNet and
        CenterNet) so the bias sets the first output; Kaiming fan-out on a
        single output channel would give std 1.4.
        """
        nn.init.normal_(self.event_head.weight, mean=0.0, std=0.01)
        nn.init.constant_(
            self.event_head.bias,
            math.log(EVENT_PRIOR / (1.0 - EVENT_PRIOR)),
        )

    def remove_event_head(self) -> None:
        """Drop the training-only event head; the model then outputs the waveform only."""
        self.event_head = None

    def fused_representation(self, x: torch.Tensor) -> torch.Tensor:
        stem = torch.cat([branch(x) for branch in self.stem_branches], dim=1)
        return self.stem_fuse(stem)

    def forward_with_diagnostics(
        self,
        x: torch.Tensor,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], torch.Tensor]:
        """Return (waveform, event logits or None, post-fusion features)."""
        fused = self.fused_representation(x)
        features = self.decoder(self.encoder(fused))
        waveform = torch.tanh(self.wave_head(features))
        logits = self.event_head(features) if self.event_head is not None else None
        return waveform, logits, fused

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        waveform, event_logits, _ = self.forward_with_diagnostics(x)
        return waveform, event_logits


def count_parameters(model: nn.Module) -> int:
    return int(sum(p.numel() for p in model.parameters() if p.requires_grad))


# ---------------------------------------------------------------------------
# Exact v16 loss, training, checkpointing, and audit
# ---------------------------------------------------------------------------

class PearsonLoss(nn.Module):
    def forward(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
    ) -> torch.Tensor:
        pred = pred.squeeze(1)
        target = target.squeeze(1)
        pred_centered = pred - pred.mean(dim=1, keepdim=True)
        target_centered = target - target.mean(dim=1, keepdim=True)
        covariance = (pred_centered * target_centered).sum(dim=1)
        denominator = torch.sqrt(
            (pred_centered**2).sum(dim=1)
            * (target_centered**2).sum(dim=1)
        ) + 1e-6
        return 1.0 - (covariance / denominator).mean()


class DerivativeLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.base = nn.SmoothL1Loss()

    def forward(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
    ) -> torch.Tensor:
        return self.base(
            pred[:, :, 1:] - pred[:, :, :-1],
            target[:, :, 1:] - target[:, :, :-1],
        )


class HybridLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.time_loss = nn.SmoothL1Loss()
        self.corr_loss = PearsonLoss()
        self.shape_loss = DerivativeLoss()

    def forward(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
    ) -> torch.Tensor:
        w_time, w_corr, w_shape = LOSS_WEIGHTS
        return (
            w_time * self.time_loss(pred, target)
            + w_corr * self.corr_loss(pred, target)
            + w_shape * self.shape_loss(pred, target)
        )


class WaveEventLoss(nn.Module):
    """v16 waveform loss plus breath-event losses.

    * wave: the v16 hybrid loss on the tanh waveform (keeps PCC and shape).
    * event: BCE between event logits and the Gaussian breath heatmap. Every
      breath has target height 1, so a shallow breath weighs as much as a
      deep one, and humps between breaths are explicit negatives.
    * count: |soft predicted count - soft target count| in breaths/min. A
      soft count is the heatmap above EVENT_COUNT_THRESHOLD, integrated and
      divided by the thresholded area of one bump of the window's mean
      width (event_norm), for prediction and target alike, so background
      probability below the threshold is not counted.
    """

    def __init__(self, fs: int = TARGET_FS):
        super().__init__()
        self.wave_loss = HybridLoss()
        self.event_loss = nn.BCEWithLogitsLoss()
        self.fs = float(fs)

    def forward(
        self,
        wave_pred: torch.Tensor,
        event_logits: torch.Tensor,
        wave_target: torch.Tensor,
        event_target: torch.Tensor,
        event_norm: torch.Tensor,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        wave = self.wave_loss(wave_pred, wave_target)
        event = self.event_loss(event_logits, event_target)
        minutes = event_target.shape[-1] / self.fs / 60.0
        norm = event_norm.reshape(-1, 1)
        threshold = EVENT_COUNT_THRESHOLD
        predicted = torch.relu(torch.sigmoid(event_logits) - threshold).sum(dim=-1) / norm
        target = torch.relu(event_target - threshold).sum(dim=-1) / norm
        count = (torch.abs(predicted - target) / minutes).mean()
        total = (
            LOSS_WEIGHT_WAVE * wave
            + LOSS_WEIGHT_EVENT * event
            + LOSS_WEIGHT_COUNT * count
        )
        parts = {
            "wave": wave.detach(),
            "event": event.detach(),
            "count_bpm": count.detach(),
        }
        return total, parts


def evaluate_loader(
    model: nn.Module,
    loader: DataLoader,
    criterion: WaveEventLoss,
    device: torch.device,
) -> Tuple[float, Dict[str, float]]:
    """Mean validation loss and its components over the loader."""
    model.eval()
    sums = {"total": 0.0, "wave": 0.0, "event": 0.0, "count_bpm": 0.0}
    batches = 0
    with torch.no_grad():
        for x, y, e, n in loader:
            wave, logits = model(x.to(device))
            loss, parts = criterion(
                wave, logits, y.to(device), e.to(device), n.to(device)
            )
            sums["total"] += float(loss.item())
            for key, value in parts.items():
                sums[key] += float(value.item())
            batches += 1
    if batches == 0:
        return np.nan, {key: np.nan for key in sums}
    means = {key: value / batches for key, value in sums.items()}
    return means["total"], means


def atomic_torch_save(state: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    torch.save(state, temporary)
    if path.exists():
        path.unlink()
    os.replace(temporary, path)


def build_training_audit(history: Sequence[dict], checkpoint: Path) -> dict:
    frame = pd.DataFrame(history)
    required = {"Epoch", "Train_Loss", "Val_Loss"}
    if frame.empty or not required.issubset(frame.columns):
        raise RuntimeError(f"Invalid training history for {checkpoint}")
    numeric = frame[["Epoch", "Train_Loss", "Val_Loss"]].apply(
        pd.to_numeric,
        errors="coerce",
    )
    if not np.isfinite(numeric.to_numpy(dtype=float)).all():
        raise RuntimeError(f"Non-finite loss detected: {checkpoint}")
    best_index = int(numeric["Val_Loss"].idxmin())
    best_row = numeric.loc[best_index]
    final_row = numeric.iloc[-1]
    best_position = int(numeric.index.get_loc(best_index))
    after_best = numeric.iloc[best_position + 1 :]
    val_increase = after_best["Val_Loss"] > best_row["Val_Loss"]
    train_decrease = after_best["Train_Loss"] < best_row["Train_Loss"]
    signature = val_increase & train_decrease
    return {
        "History_Epochs": int(len(numeric)),
        "Best_Epoch": int(best_row["Epoch"]),
        "Final_Epoch": int(final_row["Epoch"]),
        "Best_Train_Loss": float(best_row["Train_Loss"]),
        "Best_Val_Loss": float(best_row["Val_Loss"]),
        "Final_Train_Loss": float(final_row["Train_Loss"]),
        "Final_Val_Loss": float(final_row["Val_Loss"]),
        "Late_Val_Degradation": float(
            final_row["Val_Loss"] - best_row["Val_Loss"]
        ),
        "Generalization_Gap_At_Best": float(
            best_row["Val_Loss"] - best_row["Train_Loss"]
        ),
        "Post_Best_Overfit_Signature_Epochs": int(signature.sum()),
        "Post_Best_Overfit_Signature_Observed": int(signature.any()),
        "Losses_Finite": 1,
        "Best_Checkpoint_Exists": int(checkpoint.exists()),
        "Test_Evaluated_From_Best_Validation_Checkpoint": 1,
        "Model_Selection": "lowest_validation_loss_then_restore_checkpoint",
        "Overfit_Control": "early_stopping_plus_best_validation_checkpoint_restore",
        "Audit_Status": (
            "PASS_BEST_VALIDATION_CHECKPOINT"
            if checkpoint.exists()
            else "FAIL_MISSING_CHECKPOINT"
        ),
    }


def convergence_epoch_95pct(history: Sequence[dict]) -> int:
    """Return the first epoch reaching 95% of the initial-to-best val improvement."""
    frame = pd.DataFrame(history)
    if frame.empty:
        return 0
    values = pd.to_numeric(frame["Val_Loss"], errors="coerce").to_numpy(float)
    epochs = pd.to_numeric(frame["Epoch"], errors="coerce").to_numpy(int)
    finite = np.isfinite(values)
    if not finite.any():
        return 0
    values = values[finite]
    epochs = epochs[finite]
    best_index = int(np.argmin(values))
    initial = float(values[0])
    best = float(values[best_index])
    improvement = initial - best
    if improvement <= 1e-12:
        return int(epochs[best_index])
    threshold = initial - 0.95 * improvement
    reached = np.flatnonzero(values <= threshold)
    if len(reached) == 0:
        return int(epochs[best_index])
    return int(epochs[int(reached[0])])


# ---------------------------------------------------------------------------
# Test-time outputs and diagnostics of the fused representation
# ---------------------------------------------------------------------------

def predict_with_diagnostics(
    model: DeepOnlyV16EventDecoder,
    loader: DataLoader,
    device: torch.device,
) -> Dict[str, np.ndarray]:
    """Predicted waveforms, targets and post-fusion features."""
    model.eval()
    chunks = {
        "predictions": [],
        "targets": [],
        "features": [],
    }
    with torch.no_grad():
        for x, y, _, _ in loader:
            wave, _, fused = model.forward_with_diagnostics(x.to(device))
            chunks["predictions"].append(wave.squeeze(1).cpu().numpy())
            chunks["targets"].append(y.squeeze(1).numpy())
            chunks["features"].append(fused.cpu().numpy().astype(np.float32))
    if not chunks["predictions"]:
        empty = np.empty((0, 0), dtype=np.float32)
        outputs = {key: empty for key in chunks}
        outputs["features"] = np.empty((0, 0, 0), dtype=np.float32)
        return outputs
    return {
        key: np.concatenate(values).astype(np.float32)
        for key, values in chunks.items()
    }


def feature_representation_summary(features: np.ndarray) -> dict:
    """Summarize the 24-channel stem representation for one held-out subject."""
    result = {metric: np.nan for metric in FEATURE_METRICS}
    result["Feature_N_Windows"] = 0
    result["Feature_N_Samples"] = 0
    values = np.asarray(features, dtype=np.float64)
    if values.ndim != 3 or values.shape[0] == 0:
        return result
    windows, channels, samples = values.shape
    matrix = np.transpose(values, (1, 0, 2)).reshape(channels, -1)
    matrix = np.nan_to_num(matrix, nan=0.0, posinf=0.0, neginf=0.0)
    if channels < 2 or matrix.shape[1] < 3:
        return result
    matrix = matrix - np.mean(matrix, axis=1, keepdims=True)
    covariance = (matrix @ matrix.T) / max(1, matrix.shape[1] - 1)
    covariance = (covariance + covariance.T) / 2.0
    eigenvalues = np.clip(np.linalg.eigvalsh(covariance), 0.0, None)
    total = float(eigenvalues.sum())
    if total > 1e-12:
        probabilities = eigenvalues / total
        positive = probabilities > 0
        effective_rank = float(
            np.exp(-np.sum(probabilities[positive] * np.log(probabilities[positive])))
        )
        result["Feature_EffectiveRank"] = effective_rank
        result["Feature_EffectiveRank_Normalized"] = effective_rank / channels
        denom = float(np.sum(eigenvalues**2))
        if denom > 1e-12:
            result["Feature_ParticipationRatio"] = float(total**2 / denom)
    std = np.sqrt(np.maximum(np.diag(covariance), 0.0))
    denominator = np.outer(std, std)
    with np.errstate(divide="ignore", invalid="ignore"):
        correlation = covariance / denominator
    correlation = np.where(np.isfinite(correlation), correlation, 0.0)
    off_diagonal = correlation[~np.eye(channels, dtype=bool)]
    if off_diagonal.size:
        result["Feature_OffDiag_SignedMean"] = float(np.mean(off_diagonal))
        result["Feature_OffDiag_AbsMean"] = float(np.mean(np.abs(off_diagonal)))
    result["Feature_N_Windows"] = int(windows)
    result["Feature_N_Samples"] = int(matrix.shape[1])
    return result


# ---------------------------------------------------------------------------
# One LOSOCV fold
# ---------------------------------------------------------------------------

def make_optimizer(model: DeepOnlyV16EventDecoder, epochs: int):
    event_head_ids = {id(param) for param in model.event_head.parameters()}
    optimizer = torch.optim.AdamW(
        [
            {"params": [p for p in model.parameters() if id(p) not in event_head_ids]},
            {
                "params": list(model.event_head.parameters()),
                "lr": LEARNING_RATE * EVENT_HEAD_LR_MULTIPLIER,
            },
        ],
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
    )
    return optimizer, CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)


def fit_model(
    model: DeepOnlyV16EventDecoder,
    train_loader: DataLoader,
    val_loader: DataLoader,
    device: torch.device,
    epochs: int,
    min_epochs: int,
    patience: int,
    checkpoint: Path,
    state_path: Optional[Path] = None,
) -> List[dict]:
    """Train with early stopping on the validation loss.

    The best validation state is saved to `checkpoint`. The cosine schedule
    spans `epochs`. With `state_path`, the full training state (weights,
    optimizer, schedule, early-stopping counters, history and the random
    number generators) is saved after every epoch, and an existing state is
    continued from, so an interrupted run resumes exactly where it stopped.
    """
    criterion = WaveEventLoss()
    optimizer, scheduler = make_optimizer(model, epochs)
    best_val = np.inf
    patience_counter = 0
    history = []
    start_epoch = 0
    if state_path is not None and state_path.exists():
        state = torch.load(state_path, map_location=device, weights_only=False)
        model.load_state_dict(state["model"])
        optimizer.load_state_dict(state["optimizer"])
        scheduler.load_state_dict(state["scheduler"])
        best_val = float(state["best_val"])
        patience_counter = int(state["patience_counter"])
        history = list(state["history"])
        start_epoch = int(state["epoch"])
        random.setstate(state["python_rng"])
        np.random.set_state(state["numpy_rng"])
        torch.set_rng_state(state["torch_rng"])
        if torch.cuda.is_available() and state.get("cuda_rng") is not None:
            torch.cuda.set_rng_state_all(state["cuda_rng"])
        if state.get("stopped", False):
            return history
    for epoch in range(start_epoch, epochs):
        model.train()
        train_loss = 0.0
        for x, y, e, n in train_loader:
            optimizer.zero_grad(set_to_none=True)
            wave, logits = model(x.to(device))
            loss, _ = criterion(wave, logits, y.to(device), e.to(device), n.to(device))
            if not torch.isfinite(loss):
                raise RuntimeError(f"Non-finite training loss at epoch {epoch + 1}")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            train_loss += float(loss.item())
        train_loss /= max(1, len(train_loader))
        val_loss, val_parts = evaluate_loader(model, val_loader, criterion, device)
        if not np.isfinite(val_loss):
            raise RuntimeError(f"Non-finite validation loss at epoch {epoch + 1}")
        row = {
            "Epoch": epoch + 1,
            "Train_Loss": train_loss,
            "Val_Loss": val_loss,
            "Val_Wave_Loss": val_parts["wave"],
            "Val_Event_BCE": val_parts["event"],
            "Val_Count_Error_BPM": val_parts["count_bpm"],
        }
        scheduler.step()
        history.append(row)
        if row["Val_Loss"] < best_val - 1e-4:
            best_val = row["Val_Loss"]
            atomic_torch_save(model.state_dict(), checkpoint)
            patience_counter = 0
        elif epoch + 1 >= min_epochs:
            patience_counter += 1
        stop = epoch + 1 >= min_epochs and patience_counter >= patience
        if state_path is not None:
            atomic_torch_save(
                {
                    "epoch": epoch + 1,
                    "stopped": bool(stop),
                    "model": model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "scheduler": scheduler.state_dict(),
                    "best_val": best_val,
                    "patience_counter": patience_counter,
                    "history": history,
                    "python_rng": random.getstate(),
                    "numpy_rng": np.random.get_state(),
                    "torch_rng": torch.get_rng_state(),
                    "cuda_rng": torch.cuda.get_rng_state_all()
                    if torch.cuda.is_available()
                    else None,
                },
                state_path,
            )
        if stop:
            break
    return history


def train_one(
    dataset_key: str,
    spec: WindowSpec,
    test_subject: str,
    subject_data: Dict[str, dict],
    train_subjects: Sequence[str],
    member_root: Path,
    device: torch.device,
    epochs: int,
    min_epochs: int,
    patience: int,
    batch_size: int,
    seed: int,
    model_options: ModelOptions,
    save_features: bool = False,
    resume: bool = False,
) -> Tuple[Dict[str, np.ndarray], List[dict], dict, Path, dict]:
    """Train one ensemble member of one LOSOCV fold and predict the held-out subject.

    Validation: the last 20% of each training subject's windows (v16); the
    best validation state is restored before testing. The training state is
    saved after every epoch; with `resume` an interrupted member continues
    from it, otherwise it starts over.
    """
    set_seed(seed)
    model_root = member_root / "Models"
    curve_root = member_root / "Learning_Curves"
    audit_root = member_root / "Training_Audits"
    prediction_root = member_root / "Predictions"
    for path in (model_root, curve_root, audit_root, prediction_root):
        path.mkdir(parents=True, exist_ok=True)

    tag = safe_tag(test_subject)
    checkpoint = model_root / f"{tag}_best.pth"
    history_path = curve_root / f"{tag}_Learning_Curve.csv"
    audit_path = audit_root / f"{tag}_Training_Audit.csv"
    prediction_path = prediction_root / f"{tag}_test_outputs.npz"
    state_path = model_root / f"{tag}_training_state.pt"
    if not resume and state_path.exists():
        state_path.unlink()

    train_dataset = SubjectWindowDataset(subject_data, train_subjects, "train", True)
    val_dataset = SubjectWindowDataset(subject_data, train_subjects, "val", False)
    test_dataset = SubjectWindowDataset(subject_data, [test_subject], "test", False)
    train_loader = make_train_loader(train_dataset, batch_size, device)
    val_loader = make_eval_loader(val_dataset, batch_size, device)
    test_loader = make_eval_loader(test_dataset, batch_size, device)
    metadata = test_metadata(subject_data, test_subject)

    model = model_options.build().to(device)
    history = fit_model(
        model,
        train_loader,
        val_loader,
        device,
        epochs,
        min_epochs,
        patience,
        checkpoint,
        state_path,
    )
    history_frame = pd.DataFrame(history)
    history_frame.to_csv(history_path, index=False, encoding="utf-8-sig")
    audit = build_training_audit(history, checkpoint)
    audit["Convergence_Epoch_95pct"] = convergence_epoch_95pct(history)
    if audit["Audit_Status"] != "PASS_BEST_VALIDATION_CHECKPOINT":
        raise RuntimeError(f"No valid checkpoint: {checkpoint}")
    audit["Validation_Mode"] = "temporal (last 20% of each training subject)"
    pd.DataFrame([audit]).to_csv(audit_path, index=False, encoding="utf-8-sig")

    model.load_state_dict(torch.load(checkpoint, map_location=device))
    training_parameters = count_parameters(model)
    # The event head only served the auxiliary training losses.
    model.remove_event_head()
    atomic_torch_save(model.state_dict(), model_root / f"{tag}_waveform_model.pth")
    outputs = predict_with_diagnostics(model, test_loader, device)
    features = outputs.pop("features")
    best_row = history_frame.loc[history_frame["Epoch"] == audit["Best_Epoch"]].iloc[0]
    diagnostics = {
        "N_Params_Training": training_parameters,
        "N_Params": count_parameters(model),
        "Train_Windows": len(train_loader.dataset),
        "Val_Windows": len(val_loader.dataset),
        "Val_Count_Error_BPM_At_Best": float(best_row["Val_Count_Error_BPM"]),
        "Val_Event_BCE_At_Best": float(best_row["Val_Event_BCE"]),
    }
    diagnostics.update(feature_representation_summary(features))
    arrays = {
        "predictions": outputs["predictions"],
        "targets": outputs["targets"],
        "segment_ids": np.asarray(
            [item["segment_id"] for item in metadata],
            dtype=np.int64,
        ),
        "window_starts": np.asarray(
            [item["start"] for item in metadata],
            dtype=np.int64,
        ),
        "fs": np.asarray(TARGET_FS, dtype=np.int32),
        "input_window_sec": np.asarray(spec.window_sec, dtype=np.float32),
        "input_stride_sec": np.asarray(spec.stride_sec, dtype=np.float32),
        "model_key": np.asarray(model_options.key),
        "seed": np.asarray(seed, dtype=np.int64),
    }
    if save_features:
        arrays["feature_representation"] = features.astype(np.float32)
        arrays["feature_representation_location"] = np.asarray(
            "post_stem_fusion"
        )
    np.savez_compressed(prediction_path, **arrays)
    if state_path.exists():
        state_path.unlink()
    del features, arrays, train_loader, val_loader, test_loader
    del model
    release_memory(device)
    return outputs, metadata, audit, prediction_path, diagnostics


# ---------------------------------------------------------------------------
# Timeline reconstruction and direct 60-second RR evaluation
# ---------------------------------------------------------------------------

def safe_pcc(x: np.ndarray, y: np.ndarray) -> float:
    x_arr, _ = sanitize_signal(x)
    y_arr, _ = sanitize_signal(y)
    length = min(len(x_arr), len(y_arr))
    x_arr = x_arr[:length]
    y_arr = y_arr[:length]
    if length < 2:
        return np.nan
    if np.std(x_arr) < 1e-8 or np.std(y_arr) < 1e-8:
        return np.nan
    return float(stats.pearsonr(x_arr, y_arr)[0])


def reconstruct_segment(
    predictions: np.ndarray,
    targets: np.ndarray,
    metadata: Sequence[dict],
    segment_id: int,
    fs: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    selected = [
        index
        for index, item in enumerate(metadata)
        if int(item["segment_id"]) == int(segment_id)
    ]
    if not selected:
        return (
            np.empty(0, dtype=np.float64),
            np.empty(0, dtype=np.float64),
            np.empty(0, dtype=bool),
        )
    window_len = int(predictions.shape[1])
    max_end = max(int(metadata[index]["start"]) for index in selected)
    max_end += window_len
    pred_sum = np.zeros(max_end, dtype=np.float64)
    true_sum = np.zeros(max_end, dtype=np.float64)
    coverage = np.zeros(max_end, dtype=np.int32)
    for index in selected:
        start = int(metadata[index]["start"])
        end = start + window_len
        pred = np.asarray(predictions[index], dtype=np.float64)
        true = np.asarray(targets[index], dtype=np.float64)
        pred, _ = sanitize_signal(pred)
        true, _ = sanitize_signal(true)
        pred_sum[start:end] += pred[:window_len]
        true_sum[start:end] += true[:window_len]
        coverage[start:end] += 1
    valid = coverage > 0
    pred_out = np.full(max_end, np.nan, dtype=np.float64)
    true_out = np.full(max_end, np.nan, dtype=np.float64)
    pred_out[valid] = pred_sum[valid] / coverage[valid]
    true_out[valid] = true_sum[valid] / coverage[valid]
    return pred_out, true_out, valid


def contiguous_runs(mask: np.ndarray) -> List[Tuple[int, int]]:
    mask = np.asarray(mask, dtype=bool)
    if not mask.any():
        return []
    padded = np.concatenate([[False], mask, [False]])
    changes = np.flatnonzero(padded[1:] != padded[:-1])
    return [(int(left), int(right)) for left, right in changes.reshape(-1, 2)]


REFERENCE_EXCLUSION_REASONS = (
    "ref_structural_invalid",
    "ref_artifact_label",
    "ref_rr_undefined",
    "ref_irregular_cycles",
    "ref_disagreement",
    "ref_out_of_band",
)


def _mean_or_nan(values: Sequence[float]) -> float:
    values = np.asarray(values, dtype=np.float64)
    values = values[np.isfinite(values)]
    return float(np.mean(values)) if len(values) else np.nan


def summarize_subject_minutes(frame: pd.DataFrame) -> dict:
    """Subject-level summary of the per-minute evaluation table."""
    summary = {
        "N60": 0,
        "N60_RefValid": 0,
        "Ref_Valid_Ratio": np.nan,
        **{f"N60_Excluded_{reason}": 0 for reason in REFERENCE_EXCLUSION_REASONS},
        "RR_MAE_Wave_FFT": np.nan,
        "RR_Bias_Wave_FFT": np.nan,
        "Wave_PCC_Mean": np.nan,
    }
    if frame.empty:
        return summary
    summary["N60"] = int(len(frame))
    for reason in REFERENCE_EXCLUSION_REASONS:
        summary[f"N60_Excluded_{reason}"] = int(
            (frame["Ref_Exclusion_Reason"] == reason).sum()
        )
    valid = frame[frame["Ref_Valid"].astype(bool)]
    summary["N60_RefValid"] = int(len(valid))
    summary["Ref_Valid_Ratio"] = float(len(valid) / len(frame))
    if valid.empty:
        return summary
    error = valid["Pred_Wave_FFT_RR"] - valid["GT_Breath_Count"] * (60.0 / EVAL_WINDOW_SEC)
    summary.update(
        {
            "RR_MAE_Wave_FFT": _mean_or_nan(valid["AE_Wave_FFT"]),
            "RR_Bias_Wave_FFT": _mean_or_nan(error),
            "Wave_PCC_Mean": _mean_or_nan(valid["Wave_PCC"]),
        }
    )
    return summary


def evaluate_subject_minutes(
    outputs: Dict[str, np.ndarray],
    metadata: Sequence[dict],
    references: Dict[int, dict],
    fs: int,
) -> Tuple[dict, pd.DataFrame]:
    """Score every complete 60 s minute of one held-out subject (waveform only).

    Minutes are non-overlapping 60 s slices of each contiguous run of
    reconstructed windows (overlap-add mean of the window outputs). Per minute:

    * reference: with expert labels (CapnoBase) the labels are the ground
      truth and a minute needs a valid reference, two labelled breaths and a
      rate in band; otherwise the minute of the continuous reference must
      pass the RRest agreement rule and the ground truth is the number of
      breath onsets detected once on the continuous reference. Excluded
      minutes keep a reason. Band and rate range follow the segment's profile;
    * breathing rate: largest FFT peak of the predicted waveform within the
      profile's rate band, after the profile's respiratory filter is applied
      to the whole run;
    * waveform agreement: Pearson r of the predicted and target waveforms.
    """
    predictions = outputs["predictions"]
    targets = outputs["targets"]
    if len(predictions) == 0:
        return summarize_subject_minutes(pd.DataFrame()), pd.DataFrame()

    eval_len = int(round(EVAL_WINDOW_SEC * fs))
    per_minute = 60.0 / EVAL_WINDOW_SEC
    rows = []
    for segment_id in sorted({int(item["segment_id"]) for item in metadata}):
        pred_series, true_series, coverage = reconstruct_segment(
            predictions, targets, metadata, segment_id, fs
        )
        reference = references[segment_id]
        ref_signal = reference["signal"]
        ref_breaths = reference["breaths"]
        ref_structural = reference["structural_valid"]
        profile = reference["profile"]
        rr_band = profile["rr_band_bpm"]
        subharmonic = profile["fft_subharmonic_check"]
        for run_index, (run_start, run_end) in enumerate(
            contiguous_runs(coverage),
            start=1,
        ):
            if run_end - run_start < eval_len:
                continue
            minute_starts = list(range(run_start, run_end - eval_len + 1, eval_len))
            wave_run = respiratory_filter(
                pred_series[run_start:run_end], fs, reference["band_hz"], profile
            )
            for local_index, start in enumerate(minute_starts, start=1):
                end = start + eval_len
                # Reference validity and fixed ground truth. With expert labels
                # the labels are the ground truth: the minute needs a valid
                # reference signal, at least two labelled breaths and an RR in
                # the profile's band. Without labels the agreement rule decides.
                in_minute = (ref_breaths >= start) & (ref_breaths < end)
                gt_count = int(in_minute.sum())
                gt_rate = gt_count * per_minute
                minute_labels = ref_breaths[in_minute]
                label_rr = (
                    float(60.0 * fs / np.mean(np.diff(minute_labels)))
                    if reference["label_based"] and len(minute_labels) >= 2
                    else np.nan
                )
                structural = float(np.mean(ref_structural[start:end]))
                ref_artifact = (
                    float(np.mean(reference["artifact_mask"][start:end]))
                    if reference.get("artifact_mask") is not None
                    else 0.0
                )
                ppg_artifact = (
                    float(np.mean(reference["ppg_artifact_mask"][start:end]))
                    if reference.get("ppg_artifact_mask") is not None
                    else 0.0
                )
                _, _, ref_rr_cto, _ = count_orig_breaths(ref_signal[start:end], fs)
                ref_rr_fft = fft_rr_bpm(ref_signal[start:end], fs, rr_band, subharmonic)
                if structural < MIN_USABLE_PROPORTION:
                    ref_valid, reason = False, "ref_structural_invalid"
                elif ref_artifact > 0.0:
                    # Expert-labelled reference artifact: breaths inside it
                    # may be unlabelled, so the minute has no ground truth.
                    ref_valid, reason = False, "ref_artifact_label"
                elif reference["label_based"]:
                    if not np.isfinite(label_rr):
                        ref_valid, reason = False, "ref_rr_undefined"
                    elif not (rr_band[0] <= label_rr <= rr_band[1]):
                        ref_valid, reason = False, "ref_out_of_band"
                    else:
                        ref_valid, reason = True, ""
                else:
                    ref_valid, reason, _, _ = reference_agreement(
                        ref_signal[start:end], fs, rr_band, subharmonic
                    )

                # Breathing rate of the predicted waveform. The f/2, f/3 check
                # is for capnogram references; on predicted waveforms it
                # halved fast rates, so it is not used here.
                wave_fft = fft_rr_bpm(
                    wave_run[start - run_start : end - run_start],
                    fs,
                    rr_band,
                    False,
                )
                rows.append(
                    {
                        "Segment_ID": segment_id,
                        "Coverage_Run_Index": run_index,
                        "RR_Window_Index": local_index,
                        "RR_Window_Start_Sec": start / float(fs),
                        "RR_Window_End_Sec": end / float(fs),
                        "Ref_Valid": bool(ref_valid),
                        "Ref_Exclusion_Reason": reason,
                        "Ref_Structural_Fraction": structural,
                        "Ref_Artifact_Label_Fraction": ref_artifact,
                        "PPG_Artifact_Label_Fraction": ppg_artifact,
                        "Ref_RR_CtO": ref_rr_cto,
                        "Ref_RR_FFT": ref_rr_fft,
                        "Ref_RR_Label": label_rr,
                        "GT_Source": "expert_labels"
                        if reference["label_based"]
                        else "detected_breaths",
                        "GT_Breath_Count": gt_count,
                        "Pred_Wave_FFT_RR": wave_fft,
                        "AE_Wave_FFT": abs(wave_fft - gt_rate)
                        if np.isfinite(wave_fft)
                        else np.nan,
                        "Wave_PCC": safe_pcc(true_series[start:end], pred_series[start:end]),
                    }
                )
    frame = pd.DataFrame(rows)
    return summarize_subject_minutes(frame), frame


# ---------------------------------------------------------------------------
# Statistics, plots, and output tables
# ---------------------------------------------------------------------------

def safe_pvalue(function, x: np.ndarray, y: np.ndarray) -> float:
    try:
        if len(x) < 2 or len(y) < 2:
            return np.nan
        if np.allclose(np.asarray(x, dtype=float), np.asarray(y, dtype=float)):
            return 1.0
        return float(function(x, y).pvalue)
    except Exception:
        return np.nan


def bh_fdr(values: Sequence[float]) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    output = np.full(values.shape, np.nan, dtype=float)
    finite = np.isfinite(values)
    if not finite.any():
        return output
    raw = values[finite]
    order = np.argsort(raw)
    ranked = raw[order]
    adjusted = ranked * len(ranked) / np.arange(1, len(ranked) + 1)
    adjusted = np.minimum.accumulate(adjusted[::-1])[::-1]
    adjusted = np.clip(adjusted, 0.0, 1.0)
    restored = np.empty_like(adjusted)
    restored[order] = adjusted
    output[finite] = restored
    return output


MINUTE_KEY = ["Subject", "Segment_ID", "RR_Window_Start_Sec"]


def scored_minutes(frame: pd.DataFrame) -> pd.DataFrame:
    """Reference-valid minutes with the FFT rate error (breaths/min)."""
    if frame.empty:
        return frame.copy()
    valid = frame[frame["Ref_Valid"].astype(bool)].copy()
    valid["Subject"] = valid["Subject"].astype(str)
    valid["GT_RR"] = pd.to_numeric(valid["GT_Breath_Count"], errors="coerce") * (
        60.0 / EVAL_WINDOW_SEC
    )
    valid["FFT_RR"] = pd.to_numeric(valid["Pred_Wave_FFT_RR"], errors="coerce")
    valid["FFT_Error"] = valid["FFT_RR"] - valid["GT_RR"]
    valid["FFT_AE"] = valid["FFT_Error"].abs()
    return valid


def fft_davies_metrics(minutes: pd.DataFrame) -> dict:
    """FFT metrics over all minutes and the Davies & Mandic (2022) medians.

    mAE: median absolute error over all minutes; mMAE: median over subjects
    of each subject's mean absolute error. Minutes without an FFT peak are
    counted in N_Minutes_No_FFT and left out of the error statistics.
    """
    result = {
        "N_Subjects": 0,
        "N_Minutes": 0,
        "N_Minutes_No_FFT": 0,
        "FFT_MAE": np.nan,
        "FFT_RMSE": np.nan,
        "FFT_Bias": np.nan,
        "FFT_LoA_Lower": np.nan,
        "FFT_LoA_Upper": np.nan,
        "FFT_Pearson_r": np.nan,
        "FFT_Within_1_BPM_Pct": np.nan,
        "FFT_Within_2_BPM_Pct": np.nan,
        "FFT_Subject_MAE_Mean": np.nan,
        "FFT_Subject_MAE_SD": np.nan,
        "Davies_mAE": np.nan,
        "Davies_mAE_Q1": np.nan,
        "Davies_mAE_Q3": np.nan,
        "Davies_mMAE": np.nan,
        "Davies_mMAE_Q1": np.nan,
        "Davies_mMAE_Q3": np.nan,
        "Wave_PCC_Mean": np.nan,
        "Wave_PCC_Median": np.nan,
    }
    if minutes.empty:
        return result
    pcc = pd.to_numeric(minutes["Wave_PCC"], errors="coerce").dropna()
    if len(pcc):
        result["Wave_PCC_Mean"] = float(pcc.mean())
        result["Wave_PCC_Median"] = float(pcc.median())
    have = minutes[np.isfinite(minutes["FFT_AE"])]
    result["N_Minutes"] = int(len(minutes))
    result["N_Minutes_No_FFT"] = int(len(minutes) - len(have))
    if have.empty:
        return result
    errors = have["FFT_Error"].to_numpy(dtype=float)
    absolute = np.abs(errors)
    subject_mae = have.groupby("Subject")["FFT_AE"].mean()
    sd = float(np.std(errors, ddof=1)) if len(errors) > 1 else np.nan
    result.update(
        {
            "N_Subjects": int(subject_mae.size),
            "FFT_MAE": float(np.mean(absolute)),
            "FFT_RMSE": float(np.sqrt(np.mean(errors**2))),
            "FFT_Bias": float(np.mean(errors)),
            "FFT_LoA_Lower": float(np.mean(errors) - 1.96 * sd),
            "FFT_LoA_Upper": float(np.mean(errors) + 1.96 * sd),
            "FFT_Pearson_r": safe_pcc(
                have["FFT_RR"].to_numpy(dtype=float),
                have["GT_RR"].to_numpy(dtype=float),
            ),
            "FFT_Within_1_BPM_Pct": float(np.mean(absolute <= 1.0) * 100.0),
            "FFT_Within_2_BPM_Pct": float(np.mean(absolute <= 2.0) * 100.0),
            "FFT_Subject_MAE_Mean": float(subject_mae.mean()),
            "FFT_Subject_MAE_SD": float(subject_mae.std(ddof=1))
            if subject_mae.size > 1
            else np.nan,
            "Davies_mAE": float(np.median(absolute)),
            "Davies_mAE_Q1": float(np.percentile(absolute, 25)),
            "Davies_mAE_Q3": float(np.percentile(absolute, 75)),
            "Davies_mMAE": float(subject_mae.median()),
            "Davies_mMAE_Q1": float(subject_mae.quantile(0.25)),
            "Davies_mMAE_Q3": float(subject_mae.quantile(0.75)),
        }
    )
    return result


SUMMARY_METRICS = (
    "FFT_MAE",
    "FFT_RMSE",
    "FFT_Bias",
    "FFT_LoA_Lower",
    "FFT_LoA_Upper",
    "FFT_Pearson_r",
    "FFT_Within_1_BPM_Pct",
    "FFT_Within_2_BPM_Pct",
    "FFT_Subject_MAE_Mean",
    "FFT_Subject_MAE_SD",
    "Davies_mAE",
    "Davies_mAE_Q1",
    "Davies_mAE_Q3",
    "Davies_mMAE",
    "Davies_mMAE_Q1",
    "Davies_mMAE_Q3",
    "Wave_PCC_Mean",
    "Wave_PCC_Median",
)


def member_average_metrics(minutes: pd.DataFrame) -> dict:
    """Metrics of each single model (member), averaged over the members."""
    rows = [
        fft_davies_metrics(frame)
        for _, frame in minutes.groupby("Member", sort=True)
    ]
    if not rows:
        return fft_davies_metrics(pd.DataFrame())
    table = pd.DataFrame(rows)
    result = {
        "N_Subjects": int(table["N_Subjects"].max()),
        "N_Minutes": int(table["N_Minutes"].max()),
        "N_Minutes_No_FFT": float(table["N_Minutes_No_FFT"].mean()),
        "N_Members": int(len(table)),
    }
    for metric in SUMMARY_METRICS:
        result[metric] = float(table[metric].mean())
        result[f"{metric}_SD_Members"] = (
            float(table[metric].std(ddof=1)) if len(table) > 1 else np.nan
        )
    return result


def subject_errors(minutes: pd.DataFrame) -> pd.Series:
    """Per-subject FFT MAE; members are averaged per subject first."""
    have = minutes[np.isfinite(minutes["FFT_AE"])]
    if "Member" in have.columns:
        return have.groupby(["Member", "Subject"])["FFT_AE"].mean().groupby("Subject").mean()
    return have.groupby("Subject")["FFT_AE"].mean()


def minute_errors(minutes: pd.DataFrame) -> pd.Series:
    """Per-minute absolute FFT error; members are averaged per minute."""
    have = minutes[np.isfinite(minutes["FFT_AE"])]
    return have.groupby(MINUTE_KEY)["FFT_AE"].mean()


def paired_comparison(full: pd.DataFrame, other: pd.DataFrame) -> dict:
    """Full model vs one variant: per-subject and per-minute Wilcoxon tests."""
    a, b = subject_errors(full), subject_errors(other)
    subjects = a.index.intersection(b.index)
    sa, sb = a.loc[subjects].to_numpy(float), b.loc[subjects].to_numpy(float)
    ma, mb = minute_errors(full), minute_errors(other)
    keys = ma.index.intersection(mb.index)
    xa, xb = ma.loc[keys].to_numpy(float), mb.loc[keys].to_numpy(float)
    difference = sb - sa
    return {
        "N_Subjects_Paired": int(len(subjects)),
        "Subject_MAE_Full": float(np.mean(sa)) if len(sa) else np.nan,
        "Subject_MAE_Variant": float(np.mean(sb)) if len(sb) else np.nan,
        "Subject_MAE_Diff_Variant_minus_Full": float(np.mean(difference))
        if len(difference)
        else np.nan,
        "N_Subjects_Full_Better": int(np.sum(difference > 1e-9)),
        "N_Subjects_Variant_Better": int(np.sum(difference < -1e-9)),
        "N_Subjects_Tied": int(np.sum(np.abs(difference) <= 1e-9)),
        "Subject_Wilcoxon_p": safe_pvalue(stats.wilcoxon, sa, sb),
        "N_Minutes_Paired": int(len(keys)),
        "Minute_MAE_Full": float(np.mean(xa)) if len(xa) else np.nan,
        "Minute_MAE_Variant": float(np.mean(xb)) if len(xb) else np.nan,
        "Minute_Wilcoxon_p": safe_pvalue(stats.wilcoxon, xa, xb),
    }


def per_subject_table(minutes: pd.DataFrame) -> pd.DataFrame:
    """Every subject: minutes, median true and predicted rate, MAE and bias."""
    have = minutes[np.isfinite(minutes["FFT_AE"])]
    if have.empty:
        return pd.DataFrame()
    if "Member" in have.columns:
        have = have.groupby(["Subject", *MINUTE_KEY[1:]], as_index=False).agg(
            GT_RR=("GT_RR", "first"),
            FFT_RR=("FFT_RR", "mean"),
            FFT_Error=("FFT_Error", "mean"),
            FFT_AE=("FFT_AE", "mean"),
            Wave_PCC=("Wave_PCC", "mean"),
        )
    return (
        have.groupby("Subject")
        .agg(
            N_Minutes=("FFT_AE", "size"),
            GT_RR_Median=("GT_RR", "median"),
            FFT_RR_Median=("FFT_RR", "median"),
            FFT_MAE=("FFT_AE", "mean"),
            FFT_Bias=("FFT_Error", "mean"),
            Wave_PCC_Mean=("Wave_PCC", "mean"),
        )
        .reset_index()
    )


def save_summary_figures(
    summary_root: Path,
    dataset_label: str,
    spec: WindowSpec,
    rows: Sequence[Tuple[str, pd.DataFrame]],
) -> None:
    """Per-subject FFT MAE of every model, and the full model's per-subject
    median rates against the reference (Davies & Mandic, Fig. 3b)."""
    figure_root = summary_root / "Figures"
    figure_root.mkdir(parents=True, exist_ok=True)
    stem = f"{dataset_label}_{spec.name}"
    labels, values = [], []
    for label, minutes in rows:
        errors = subject_errors(minutes)
        if errors.size:
            labels.append(label)
            values.append(errors.to_numpy(float))
    if values:
        fig, axis = plt.subplots(figsize=(max(6.0, 1.6 * len(values)), 4.5))
        axis.boxplot(values, showfliers=False)
        for position, data in enumerate(values, start=1):
            jitter = np.random.default_rng(position).uniform(-0.12, 0.12, len(data))
            axis.scatter(position + jitter, data, s=10, alpha=0.6, color="#4c78a8")
        axis.set_xticks(range(1, len(labels) + 1))
        axis.set_xticklabels(labels, rotation=20, ha="right")
        axis.set_ylabel("Per-subject FFT MAE (breaths/min)")
        axis.set_title(f"{dataset_label}, {spec.label}")
        fig.tight_layout()
        fig.savefig(figure_root / f"{stem}_subject_fft_mae.png", dpi=150)
        plt.close(fig)
    if rows:
        table = per_subject_table(rows[0][1])
        if not table.empty:
            fig, axis = plt.subplots(figsize=(4.8, 4.8))
            low = float(min(table["GT_RR_Median"].min(), table["FFT_RR_Median"].min()))
            high = float(max(table["GT_RR_Median"].max(), table["FFT_RR_Median"].max()))
            axis.plot([low, high], [low, high], color="#888888", linewidth=1)
            axis.scatter(table["GT_RR_Median"], table["FFT_RR_Median"], s=18, color="#4c78a8")
            axis.set_xlabel("Reference median rate (breaths/min)")
            axis.set_ylabel("Predicted median rate (breaths/min)")
            axis.set_title(f"{dataset_label}: {rows[0][0]}")
            fig.tight_layout()
            fig.savefig(figure_root / f"{stem}_subject_median_rate.png", dpi=150)
            plt.close(fig)


def write_results_excel(path: Path, sheets: Dict[str, pd.DataFrame]) -> Path:
    try:
        with pd.ExcelWriter(path, engine="openpyxl") as writer:
            for name, frame in sheets.items():
                frame.to_excel(writer, index=False, sheet_name=name[:31])
    except Exception as exc:
        warnings.warn(f"Excel export failed: {exc}")
    return path


# ---------------------------------------------------------------------------
# Main experiment runner
# ---------------------------------------------------------------------------

def write_quality_report(
    quality_root: Path,
    dataset_label: str,
    subject_data: Dict[str, dict],
) -> None:
    rows = []
    quality_root.mkdir(parents=True, exist_ok=True)
    all_window_rows = []
    for subject, record in subject_data.items():
        quality = pd.DataFrame(record["quality"])
        candidate = len(quality)
        kept = int(quality["Kept"].sum()) if not quality.empty else 0
        train_ok = int(quality["Train_OK"].sum()) if not quality.empty else 0
        row = {
            "Dataset": dataset_label,
            "Subject": subject,
            "Candidate_Windows": candidate,
            "Kept_Windows": kept,
            "Kept_Ratio": float(kept / candidate) if candidate else np.nan,
            "Train_OK_Windows": train_ok,
            "Train_OK_Ratio_Of_Kept": float(train_ok / kept) if kept else np.nan,
            "Reference_Breaths": int(
                sum(len(ref["breaths"]) for ref in record["references"].values())
            ),
            **record["reference_diagnostics"],
            "Window_Sec": record["window_sec"],
            "Stride_Sec": record["stride_sec"],
        }
        rows.append(row)
        if not quality.empty:
            quality.insert(0, "Subject", subject)
            all_window_rows.append(quality)
    pd.DataFrame(rows).to_csv(
        quality_root / f"{dataset_label}_subject_quality.csv",
        index=False,
        encoding="utf-8-sig",
    )
    if all_window_rows:
        pd.concat(all_window_rows, ignore_index=True).to_csv(
            quality_root / f"{dataset_label}_window_quality.csv",
            index=False,
            encoding="utf-8-sig",
        )


def _json_default(value):
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f"Not JSON serializable: {type(value)!r}")


def build_manifest(
    args: argparse.Namespace,
    results_root: Path,
    device: torch.device,
    variants: Sequence[Variant],
    window_specs: Sequence[WindowSpec],
    dataset_keys: Sequence[str],
) -> dict:
    return {
        "script": str(Path(__file__).resolve()),
        "results_root": str(results_root),
        "device": str(device),
        "model": {
            "path": (
                "raw PPG (min-max per window) -> 3 stem branches (kernels "
                f"{list(MODEL_CONFIG['stem_kernel_sizes'])} samples at {TARGET_FS} Hz, "
                f"{MODEL_CONFIG['stem_base_channels']} filters, GroupNorm, GELU) -> "
                "1x1 fusion + GroupNorm -> depthwise residual encoder (kernel "
                f"{MODEL_CONFIG['encoder_kernel_size']}, dilations "
                f"{list(MODEL_CONFIG['encoder_dilations'])}) -> decoder "
                f"(channels {list(MODEL_CONFIG['decoder_mid_channels'])}, kernels "
                f"{list(MODEL_CONFIG['decoder_kernel_sizes'])}, dilations "
                f"{list(MODEL_CONFIG['decoder_dilations'])}) -> waveform head (tanh); "
                "breath-event head for the auxiliary training losses only, removed after training"
            ),
            "identity_channel": False,
            "gating": False,
            "model_config": MODEL_CONFIG,
            "model_fs": TARGET_FS,
        },
        "variants": {
            variant.name: {
                "label": variant.label,
                "model_key": variant.options.key,
                "model_label": variant.options.label,
                "max_epochs": variant.epochs,
                "parameters": count_parameters(variant.options.build()),
            }
            for variant in variants
        },
        "ensemble": {
            "members": args.members,
            "combination": "mean of the members' waveform outputs",
            "seed": "stable_seed('261006-ensemble-member', window, dataset, subject, member)",
            "same_seeds_for_every_variant": True,
        },
        "training": {
            "min_epochs": args.min_epochs,
            "patience": args.patience,
            "epoch_cap": args.epoch_cap,
            "batch_size": args.batch_size,
            "learning_rate": LEARNING_RATE,
            "weight_decay": WEIGHT_DECAY,
            "event_head_lr_multiplier": EVENT_HEAD_LR_MULTIPLIER,
            "loss_weights": {
                "wave": LOSS_WEIGHT_WAVE,
                "event": LOSS_WEIGHT_EVENT,
                "count": LOSS_WEIGHT_COUNT,
            },
            "validation": "last 20% of each training subject's windows, best checkpoint",
            "window_jitter": f"shift up to {SHIFT_AUG_SEC} s, gain 0.95-1.05, noise 0.002-0.01",
            "speed_augmentation": False,
            "rate_balanced_sampling": False,
        },
        "evaluation": {
            "unit": f"complete {EVAL_WINDOW_SEC:g} s minutes with a valid reference",
            "rate_estimate": "largest FFT peak of the predicted waveform",
            "waveform_agreement": "Pearson r of predicted and target waveform per minute",
            "ground_truth": "reference breaths in the minute (CapnoBase expert labels)",
            "fft_metrics": list(SUMMARY_METRICS),
            "davies_mandic_2022": "mAE = median minute AE, mMAE = median subject MAE",
            "subjects_excluded": "none",
            "smart_fusion": "not used",
        },
        "analysis_profiles": {
            DATASET_CONFIGS[key]["label"]: {
                "profile": DATASET_CONFIGS[key]["profile"],
                **ANALYSIS_PROFILES[DATASET_CONFIGS[key]["profile"]],
            }
            for key in dataset_keys
        },
        "data_paths": {
            DATASET_CONFIGS[key]["label"]: {
                field: DATASET_CONFIGS[key][field]
                for field in ("data_path", "ppg_dir", "rsp_dir")
                if field in DATASET_CONFIGS[key]
            }
            for key in dataset_keys
        },
        "window_conditions": [spec.name for spec in window_specs],
        "datasets": [DATASET_CONFIGS[key]["label"] for key in dataset_keys],
        "cross_validation": "Leave-One-Subject-Out; each subject is held out once",
    }


def condition_root(results_root: Path, variant: str, spec: WindowSpec, dataset_key: str) -> Path:
    return (
        results_root
        / variant
        / f"{spec.order:02d}_{spec.name}"
        / DATASET_CONFIGS[dataset_key]["label"]
    )


def member_seed(spec: WindowSpec, dataset_key: str, subject: str, member: int) -> int:
    return stable_seed("261006-ensemble-member", spec.name, dataset_key, subject, member)


def member_paths(root: Path, subject: str, member: int) -> Tuple[Path, Path, Path]:
    tag = safe_tag(subject)
    summaries = root / "Fold_Summaries"
    return (
        summaries / f"{tag}_m{member}_summary.json",
        summaries / f"{tag}_m{member}_RR60s_windows.csv",
        root / "Members" / f"m{member}" / "Predictions" / f"{tag}_test_outputs.npz",
    )


def ensemble_paths(root: Path, subject: str) -> Tuple[Path, Path, Path]:
    tag = safe_tag(subject)
    return (
        root / "Fold_Summaries" / f"{tag}_ensemble_summary.json",
        root / "Fold_Summaries" / f"{tag}_ensemble_RR60s_windows.csv",
        root / "Ensemble_Predictions" / f"{tag}_ensemble_outputs.npz",
    )


def read_json(path: Path) -> Optional[dict]:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


def member_done(root: Path, variant: Variant, subject: str, member: int, epochs: int) -> bool:
    summary_path, window_path, prediction_path = member_paths(root, subject, member)
    summary = read_json(summary_path)
    return bool(
        summary is not None
        and summary.get("Model") == variant.options.key
        and int(summary.get("Max_Epochs", -1)) == int(epochs)
        and window_path.exists()
        and prediction_path.exists()
    )


def run_member(
    args: argparse.Namespace,
    results_root: Path,
    variant: Variant,
    spec: WindowSpec,
    dataset_key: str,
    subject_data: Dict[str, dict],
    subjects: Sequence[str],
    test_subject: str,
    member: int,
    device: torch.device,
) -> Optional[dict]:
    """Train and score one member of one fold; returns its summary."""
    root = condition_root(results_root, variant.name, spec, dataset_key)
    summary_path, window_path, _ = member_paths(root, test_subject, member)
    epochs = variant.epochs if args.epoch_cap is None else min(variant.epochs, args.epoch_cap)
    if args.resume and member_done(root, variant, test_subject, member, epochs):
        return None
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    dataset_label = DATASET_CONFIGS[dataset_key]["label"]
    seed = member_seed(spec, dataset_key, test_subject, member)
    train_subjects = [subject for subject in subjects if subject != test_subject]
    start_time = time.time()
    outputs, metadata, audit, prediction_path, diagnostics = train_one(
        dataset_key,
        spec,
        test_subject,
        subject_data,
        train_subjects,
        root / "Members" / f"m{member}",
        device,
        epochs,
        args.min_epochs,
        args.patience,
        args.batch_size,
        seed,
        variant.options,
        args.save_features,
        args.resume,
    )
    summary, window_frame = evaluate_subject_minutes(
        outputs,
        metadata,
        subject_data[test_subject]["references"],
        TARGET_FS,
    )
    del outputs
    summary.update(
        {
            "Dataset": dataset_label,
            "Subject": str(test_subject),
            "Variant": variant.name,
            "Member": int(member),
            "Seed": int(seed),
            "LOSO_Fold_Index": subjects.index(test_subject) + 1,
            "Window_Config": spec.name,
            "Model": variant.options.key,
            "Model_Label": variant.options.label,
            "Max_Epochs": int(epochs),
            "Best_Val_Loss": audit["Best_Val_Loss"],
            "Best_Epoch": audit["Best_Epoch"],
            "Convergence_Epoch_95pct": audit["Convergence_Epoch_95pct"],
            "History_Epochs": audit["History_Epochs"],
            "Generalization_Gap_At_Best": audit["Generalization_Gap_At_Best"],
            "Elapsed_Sec": time.time() - start_time,
            "Prediction_File": str(prediction_path),
        }
    )
    summary.update(diagnostics)
    summary.update(subject_data[test_subject]["reference_diagnostics"])
    if not window_frame.empty:
        window_frame.insert(0, "Member", int(member))
        window_frame.insert(0, "Variant", variant.name)
        window_frame.insert(0, "Subject", str(test_subject))
        window_frame.insert(0, "Dataset", dataset_label)
    window_frame.to_csv(window_path, index=False, encoding="utf-8-sig")
    # Written last: its presence marks the member as complete.
    summary_path.write_text(
        json.dumps(summary, indent=2, default=_json_default),
        encoding="utf-8",
    )
    return summary


def evaluate_ensemble(
    args: argparse.Namespace,
    results_root: Path,
    variant: Variant,
    spec: WindowSpec,
    dataset_key: str,
    subject_data: Dict[str, dict],
    test_subject: str,
) -> Optional[dict]:
    """Average the members' outputs on the held-out subject and score them.

    Returns None while a member is missing. A stored ensemble result newer
    than all its members is reused.
    """
    root = condition_root(results_root, variant.name, spec, dataset_key)
    epochs = variant.epochs if args.epoch_cap is None else min(variant.epochs, args.epoch_cap)
    members = list(range(args.members))
    if not all(member_done(root, variant, test_subject, m, epochs) for m in members):
        return None
    summary_path, window_path, prediction_path = ensemble_paths(root, test_subject)
    member_files = [member_paths(root, test_subject, m) for m in members]
    newest_member = max(path.stat().st_mtime for files in member_files for path in files)
    stored = read_json(summary_path)
    if (
        stored is not None
        and stored.get("Members") == members
        and window_path.exists()
        and summary_path.stat().st_mtime >= newest_member
    ):
        return stored
    arrays = [dict(np.load(files[2], allow_pickle=False)) for files in member_files]
    for other in arrays[1:]:
        if not (
            np.array_equal(other["segment_ids"], arrays[0]["segment_ids"])
            and np.array_equal(other["window_starts"], arrays[0]["window_starts"])
        ):
            raise RuntimeError(f"{test_subject}: members scored different windows")
    outputs = {
        "predictions": np.mean([a["predictions"] for a in arrays], axis=0).astype(np.float32),
        "targets": arrays[0]["targets"],
    }
    summary, window_frame = evaluate_subject_minutes(
        outputs,
        test_metadata(subject_data, test_subject),
        subject_data[test_subject]["references"],
        TARGET_FS,
    )
    prediction_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        prediction_path,
        **outputs,
        segment_ids=arrays[0]["segment_ids"],
        window_starts=arrays[0]["window_starts"],
        seeds=np.asarray([int(a["seed"]) for a in arrays], dtype=np.int64),
        model_key=np.asarray(variant.options.key),
    )
    dataset_label = DATASET_CONFIGS[dataset_key]["label"]
    summary.update(
        {
            "Dataset": dataset_label,
            "Subject": str(test_subject),
            "Variant": variant.name,
            "Members": members,
            "Seeds": [int(a["seed"]) for a in arrays],
            "Window_Config": spec.name,
            "Model": variant.options.key,
            "Model_Label": variant.options.label,
        }
    )
    summary.update(subject_data[test_subject]["reference_diagnostics"])
    if not window_frame.empty:
        window_frame.insert(0, "Variant", variant.name)
        window_frame.insert(0, "Subject", str(test_subject))
        window_frame.insert(0, "Dataset", dataset_label)
    window_frame.to_csv(window_path, index=False, encoding="utf-8-sig")
    summary_path.write_text(
        json.dumps(summary, indent=2, default=_json_default),
        encoding="utf-8",
    )
    return summary


def read_minutes(paths: Sequence[Path]) -> pd.DataFrame:
    frames = []
    for path in paths:
        if path.exists() and path.stat().st_size > 0:
            frame = pd.read_csv(path, dtype={"Subject": str}, encoding="utf-8-sig")
            if not frame.empty:
                frames.append(frame)
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def summarize_results(
    args: argparse.Namespace,
    results_root: Path,
    variants: Sequence[Variant],
    window_specs: Sequence[WindowSpec],
    dataset_keys: Sequence[str],
) -> None:
    """Score the ensembles and write the FFT / Davies & Mandic comparison."""
    summary_root = results_root / "Summary"
    summary_root.mkdir(parents=True, exist_ok=True)
    overall_rows, single_rows, comparison_rows, subject_tables = [], [], [], []
    ensemble_frames, member_frames, missing_rows = [], [], []
    for spec in window_specs:
        for dataset_key in dataset_keys:
            dataset_label = DATASET_CONFIGS[dataset_key]["label"]
            try:
                subject_data = load_dataset(dataset_key, spec)
            except Exception as exc:
                print(f"{dataset_label}: not summarized ({exc})")
                continue
            subjects = selected_subjects(args, subject_data)
            minutes_by_row: List[Tuple[str, str, pd.DataFrame]] = []
            for variant in variants:
                root = condition_root(results_root, variant.name, spec, dataset_key)
                epochs = (
                    variant.epochs
                    if args.epoch_cap is None
                    else min(variant.epochs, args.epoch_cap)
                )
                ensemble_paths_done, member_paths_done = [], []
                for subject in subjects:
                    done = [
                        m
                        for m in range(args.members)
                        if member_done(root, variant, subject, m, epochs)
                    ]
                    member_paths_done += [member_paths(root, subject, m)[1] for m in done]
                    if len(done) == args.members:
                        evaluate_ensemble(
                            args, results_root, variant, spec, dataset_key, subject_data, subject
                        )
                        ensemble_paths_done.append(ensemble_paths(root, subject)[1])
                    else:
                        missing_rows.append(
                            {
                                "Dataset": dataset_label,
                                "Window_Config": spec.name,
                                "Variant": variant.name,
                                "Subject": subject,
                                "Members_Done": len(done),
                                "Members_Required": args.members,
                            }
                        )
                ensemble = scored_minutes(read_minutes(ensemble_paths_done))
                members = scored_minutes(read_minutes(member_paths_done))
                if not ensemble.empty:
                    ensemble_frames.append(ensemble)
                if not members.empty:
                    member_frames.append(members)
                    single = member_average_metrics(members)
                    single.update(
                        {
                            "Dataset": dataset_label,
                            "Window_Config": spec.name,
                            "Variant": variant.name,
                            "Model": f"{variant.label}, single models",
                            "Subjects_Complete": int(len(ensemble_paths_done)),
                        }
                    )
                    single_rows.append(single)
                if variant.name == "full" and not ensemble.empty:
                    minutes_by_row.append(
                        (f"{variant.label} (ensemble of {args.members})", "full", ensemble)
                    )
                    if not members.empty:
                        complete = members[members["Subject"].isin(ensemble["Subject"].unique())]
                        minutes_by_row.append((NO_ENSEMBLE_LABEL, "no_ensemble", complete))
                elif not ensemble.empty:
                    minutes_by_row.append(
                        (f"{variant.label} (ensemble of {args.members})", variant.name, ensemble)
                    )
            del subject_data
            release_memory()
            if not minutes_by_row:
                continue
            for label, row_name, minutes in minutes_by_row:
                metrics = (
                    member_average_metrics(minutes)
                    if row_name == "no_ensemble"
                    else fft_davies_metrics(minutes)
                )
                metrics.update(
                    {
                        "Dataset": dataset_label,
                        "Window_Config": spec.name,
                        "Row": row_name,
                        "Model": label,
                    }
                )
                overall_rows.append(metrics)
                table = per_subject_table(minutes)
                if not table.empty:
                    table.insert(0, "Model", label)
                    table.insert(0, "Row", row_name)
                    table.insert(0, "Window_Config", spec.name)
                    table.insert(0, "Dataset", dataset_label)
                    subject_tables.append(table)
            full_rows = [item for item in minutes_by_row if item[1] == "full"]
            if full_rows:
                full_minutes = full_rows[0][2]
                block = []
                for label, row_name, minutes in minutes_by_row:
                    if row_name == "full":
                        continue
                    result = paired_comparison(full_minutes, minutes)
                    result.update(
                        {
                            "Dataset": dataset_label,
                            "Window_Config": spec.name,
                            "Comparison": f"Full model vs {label}",
                            "Row": row_name,
                        }
                    )
                    block.append(result)
                for key in ("Subject_Wilcoxon_p", "Minute_Wilcoxon_p"):
                    adjusted = bh_fdr([row[key] for row in block])
                    for row, value in zip(block, adjusted):
                        row[key.replace("_p", "_FDR_p")] = value
                comparison_rows += block
            save_summary_figures(
                summary_root,
                dataset_label,
                spec,
                [(label, minutes) for label, _, minutes in minutes_by_row],
            )

    def ordered(frame: pd.DataFrame, first: Sequence[str]) -> pd.DataFrame:
        if frame.empty:
            return frame
        columns = [c for c in first if c in frame.columns]
        return frame[columns + [c for c in frame.columns if c not in columns]]

    overall = ordered(
        pd.DataFrame(overall_rows),
        ["Dataset", "Window_Config", "Row", "Model", "N_Subjects", "N_Minutes"],
    )
    single = ordered(
        pd.DataFrame(single_rows),
        ["Dataset", "Window_Config", "Variant", "Model", "N_Members", "N_Subjects", "N_Minutes"],
    )
    comparison = ordered(
        pd.DataFrame(comparison_rows),
        ["Dataset", "Window_Config", "Comparison", "Row"],
    )
    per_subject = pd.concat(subject_tables, ignore_index=True) if subject_tables else pd.DataFrame()
    ensemble_all = pd.concat(ensemble_frames, ignore_index=True) if ensemble_frames else pd.DataFrame()
    member_all = pd.concat(member_frames, ignore_index=True) if member_frames else pd.DataFrame()
    missing = pd.DataFrame(missing_rows)
    tables = {
        "fft_davies_overall": overall,
        "fft_davies_single_models": single,
        "ablation_comparison": comparison,
        "per_subject_fft": per_subject,
        "minutes_ensemble": ensemble_all,
        "minutes_single_models": member_all,
        "incomplete_folds": missing,
    }
    for name, frame in tables.items():
        frame.to_csv(summary_root / f"{name}.csv", index=False, encoding="utf-8-sig")
    excel_path = write_results_excel(
        summary_root / "261006_RRWaveNet_Ensemble_results.xlsx",
        {
            "Overall_FFT_Davies": overall,
            "Single_models": single,
            "Ablation_comparison": comparison,
            "Per_subject": per_subject,
            "Minutes_ensemble": ensemble_all,
            "Incomplete_folds": missing,
        },
    )
    if overall.empty:
        print("\nNo complete fold to summarize yet.")
        return
    print(
        "\nFFT and Davies & Mandic metrics (breaths/min; all reference-valid minutes; "
        "PCC = waveform Pearson r):"
    )
    header = (
        f"  {'Dataset':<10s} {'Model':<44s} {'Subj':>4s} {'Min':>5s} {'MAE':>6s} "
        f"{'RMSE':>6s} {'r':>5s} {'<=2':>5s} {'subjMAE':>12s} {'mAE [IQR]':>18s} "
        f"{'mMAE [IQR]':>18s} {'PCC':>5s}"
    )
    print(header)
    for _, row in overall.iterrows():
        print(
            f"  {row['Dataset']:<10s} {str(row['Model'])[:44]:<44s} {int(row['N_Subjects']):>4d} "
            f"{int(row['N_Minutes']):>5d} {row['FFT_MAE']:>6.3f} {row['FFT_RMSE']:>6.3f} "
            f"{row['FFT_Pearson_r']:>5.2f} {row['FFT_Within_2_BPM_Pct']:>4.0f}% "
            f"{row['FFT_Subject_MAE_Mean']:>5.2f}+-{row['FFT_Subject_MAE_SD']:<5.2f} "
            f"{row['Davies_mAE']:>5.2f} [{row['Davies_mAE_Q1']:.2f}-{row['Davies_mAE_Q3']:.2f}] "
            f"{row['Davies_mMAE']:>5.2f} [{row['Davies_mMAE_Q1']:.2f}-{row['Davies_mMAE_Q3']:.2f}] "
            f"{row['Wave_PCC_Mean']:>5.2f}"
        )
    if not comparison.empty:
        print("\nAblation (paired with the full model; per-subject FFT MAE):")
        for _, row in comparison.iterrows():
            print(
                f"  {row['Dataset']:<10s} {row['Comparison']:<60s} "
                f"diff {row['Subject_MAE_Diff_Variant_minus_Full']:+.3f}, full better "
                f"{row['N_Subjects_Full_Better']}/{row['N_Subjects_Paired']}, "
                f"p={row['Subject_Wilcoxon_p']:.3g} (FDR {row['Subject_Wilcoxon_FDR_p']:.3g}); "
                f"minutes p={row['Minute_Wilcoxon_p']:.3g}"
            )
    if not missing.empty:
        print(f"\n{len(missing)} fold(s) incomplete; see {summary_root / 'incomplete_folds.csv'}")
    print(f"\nSaved the summary under: {summary_root}")
    print(f"Excel workbook: {excel_path}")


def selected_subjects(args: argparse.Namespace, subject_data: Dict[str, dict]) -> List[str]:
    subjects = sorted(subject_data)
    if args.subjects:
        wanted = {str(subject) for subject in args.subjects}
        subjects = [subject for subject in subjects if subject in wanted]
    return subjects


def parse_shard(value: Optional[str]) -> Tuple[int, int]:
    if not value:
        return 0, 1
    index, count = (int(part) for part in value.split("/"))
    if count < 1 or not 0 <= index < count:
        raise ValueError("--shard must be K/N with 0 <= K < N")
    return index, count


def run_experiment(args: argparse.Namespace) -> None:
    results_root = Path(args.results_root).resolve()
    results_root.mkdir(parents=True, exist_ok=True)
    DATASET_CONFIGS["capnobase"]["use_artifact_labels"] = args.capnobase_artifacts
    if args.capnobase_path:
        DATASET_CONFIGS["capnobase"]["data_path"] = str(args.capnobase_path)
    if args.bidmc_path:
        DATASET_CONFIGS["bidmc"]["data_path"] = str(args.bidmc_path)
    if args.steam2_ppg_dir:
        DATASET_CONFIGS["steam2"]["ppg_dir"] = str(args.steam2_ppg_dir)
    if args.steam2_rsp_dir:
        DATASET_CONFIGS["steam2"]["rsp_dir"] = str(args.steam2_rsp_dir)
    if args.inspect_capnobase:
        inspect_capnobase(results_root)
        return
    if args.threads:
        torch.set_num_threads(int(args.threads))
    if args.require_cuda and not torch.cuda.is_available() and not args.summary_only:
        raise RuntimeError(
            "CUDA is required by default. Use --no-require-cuda to run on the CPU."
        )
    device = torch.device(
        f"cuda:{args.gpu}" if torch.cuda.is_available() and not args.force_cpu else "cpu"
    )
    variants = [VARIANTS[name] for name in VARIANTS if name in args.variants]
    window_specs = [spec for spec in WINDOW_SPECS if spec.name in args.windows]
    dataset_keys = [key for key in DATASET_ORDER if key in args.datasets]
    shard_index, shard_count = parse_shard(args.shard)
    manifest = build_manifest(args, results_root, device, variants, window_specs, dataset_keys)
    manifest_name = "run_manifest.json" if shard_count == 1 else f"run_manifest_shard{shard_index}.json"
    (results_root / manifest_name).write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False, default=_json_default),
        encoding="utf-8",
    )
    for variant in variants:
        print(
            f"{variant.name}: {variant.options.label}; up to {variant.epochs} epochs; "
            f"{count_parameters(variant.options.build())} parameters"
        )
    print(f"Ensemble of {args.members} models per fold; device={device}")

    if args.dry_run:
        for spec in window_specs:
            for dataset_key in dataset_keys:
                label = DATASET_CONFIGS[dataset_key]["label"]
                try:
                    data = load_dataset(dataset_key, spec)
                except Exception as exc:
                    print(f"{label}: could not be loaded ({exc})")
                    continue
                print(
                    f"{label} ({spec.label}): "
                    f"{len(data)} valid subjects = {len(data)} LOSOCV folds"
                )
                del data
                release_memory()
        return

    if not args.summary_only:
        failed_path = results_root / (
            "failed_members.csv" if shard_count == 1 else f"failed_members_shard{shard_index}.csv"
        )
        failed_rows = []
        for spec in window_specs:
            for dataset_key in dataset_keys:
                dataset_label = DATASET_CONFIGS[dataset_key]["label"]
                try:
                    subject_data = load_dataset(dataset_key, spec)
                except Exception as exc:
                    print(f"{dataset_label}: could not be loaded, skipped ({exc})")
                    failed_rows.append(
                        {"Dataset": dataset_label, "Subject": "", "Error": repr(exc)}
                    )
                    continue
                all_subjects = sorted(subject_data)
                if len(all_subjects) < 3:
                    raise RuntimeError(f"{dataset_label}: fewer than three valid subjects")
                if shard_index == 0:
                    write_quality_report(
                        results_root / "Data_Quality" / f"{spec.order:02d}_{spec.name}",
                        dataset_label,
                        subject_data,
                    )
                subjects = selected_subjects(args, subject_data)
                jobs = [
                    (variant, subject, member)
                    for variant in variants
                    for subject in subjects
                    for member in range(args.members)
                ]
                mine = [job for index, job in enumerate(jobs) if index % shard_count == shard_index]
                print(
                    f"\n{dataset_label} ({spec.label}): {len(all_subjects)} LOSOCV folds, "
                    f"{len(jobs)} member runs ({len(mine)} in this worker)"
                )
                for variant, subject, member in mine:
                    try:
                        summary = run_member(
                            args,
                            results_root,
                            variant,
                            spec,
                            dataset_key,
                            subject_data,
                            all_subjects,
                            subject,
                            member,
                            device,
                        )
                        if summary is not None:
                            print(
                                f"[{time.strftime('%H:%M:%S')}] {dataset_label} {variant.name} "
                                f"{subject} m{member}: FFT MAE={summary['RR_MAE_Wave_FFT']:.3f} "
                                f"(valid min {summary['N60_RefValid']}), best epoch "
                                f"{summary['Best_Epoch']}/{summary['History_Epochs']}, "
                                f"{summary['Elapsed_Sec'] / 60.0:.1f} min"
                            )
                        ensemble = evaluate_ensemble(
                            args, results_root, variant, spec, dataset_key, subject_data, subject
                        )
                        if ensemble is not None and summary is not None:
                            print(
                                f"    {dataset_label} {variant.name} {subject}: ensemble of "
                                f"{args.members} FFT MAE={ensemble['RR_MAE_Wave_FFT']:.3f}"
                            )
                    except Exception as exc:
                        failed_rows.append(
                            {
                                "Dataset": dataset_label,
                                "Window_Config": spec.name,
                                "Variant": variant.name,
                                "Subject": subject,
                                "Member": member,
                                "Error": repr(exc),
                            }
                        )
                        print(f"{dataset_label} {variant.name} {subject} m{member}: FAILED {exc}")
                    finally:
                        release_memory(device)
                del subject_data
                release_memory(device)
        pd.DataFrame(failed_rows).to_csv(failed_path, index=False, encoding="utf-8-sig")
        if shard_count > 1:
            print(
                "\nWorker finished. When every worker is done, run with "
                "--resume --summary-only to build the summary."
            )
            return
    summarize_results(args, results_root, variants, window_specs, dataset_keys)


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--results-root", type=Path, default=DEFAULT_RESULTS_ROOT)
    parser.add_argument(
        "--datasets",
        nargs="+",
        choices=DATASET_ORDER,
        default=list(DEFAULT_DATASETS),
    )
    parser.add_argument(
        "--windows",
        nargs="+",
        choices=[spec.name for spec in WINDOW_SPECS],
        default=list(DEFAULT_WINDOWS),
    )
    parser.add_argument(
        "--variants",
        nargs="+",
        choices=list(VARIANTS),
        default=list(VARIANTS),
        help="Ablation variants to run (default: all). 'full' is the final model.",
    )
    parser.add_argument(
        "--members",
        type=int,
        default=ENSEMBLE_MEMBERS,
        help="Models per fold in the ensemble (default %(default)s).",
    )
    parser.add_argument("--min-epochs", type=int, default=MIN_EPOCHS)
    parser.add_argument("--patience", type=int, default=PATIENCE)
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    parser.add_argument(
        "--epoch-cap",
        type=int,
        default=None,
        help="Upper limit on every variant's epochs (quick tests only).",
    )
    parser.add_argument(
        "--subjects",
        nargs="+",
        default=None,
        help="Only these held-out subjects (quick tests); training still uses all others.",
    )
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--force-cpu", action="store_true")
    parser.add_argument(
        "--threads",
        type=int,
        default=None,
        help="torch CPU threads per process (e.g. 1 with several --shard workers).",
    )
    parser.add_argument(
        "--no-require-cuda",
        dest="require_cuda",
        action="store_false",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help=(
            "Reuse members whose summary, minutes and predictions already exist "
            "and continue interrupted members from their last epoch."
        ),
    )
    parser.add_argument(
        "--shard",
        default=None,
        help="K/N: run every N-th member run starting at K (parallel workers).",
    )
    parser.add_argument(
        "--summary-only",
        action="store_true",
        help="Do not train; score the ensembles and write the summary.",
    )
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--capnobase-path", type=Path, default=None)
    parser.add_argument("--bidmc-path", type=Path, default=None)
    parser.add_argument("--steam2-ppg-dir", type=Path, default=None)
    parser.add_argument("--steam2-rsp-dir", type=Path, default=None)
    parser.add_argument(
        "--inspect-capnobase",
        action="store_true",
        help=(
            "Only read every CapnoBase case, print how it was loaded and save "
            "capnobase_loading_check.csv (no training, no GPU needed)."
        ),
    )
    parser.add_argument(
        "--ignore-capnobase-artifacts",
        dest="capnobase_artifacts",
        action="store_false",
        help="Do not use the expert artifact labels of CapnoBase.",
    )
    parser.add_argument(
        "--save-features",
        action="store_true",
        help="Also store the post-fusion features in each prediction file.",
    )
    parser.set_defaults(
        capnobase_artifacts=DATASET_CONFIGS["capnobase"]["use_artifact_labels"],
        require_cuda=True,
    )
    return parser.parse_args(argv)


if __name__ == "__main__":
    run_experiment(parse_args())
