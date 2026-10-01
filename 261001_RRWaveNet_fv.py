"""261001_RRWaveNet_fv: final version of the v16 early-fusion RR pipeline.

Model: v16 Deep-only early stem fusion with the identity channel (the scale
gate is off by default; --scale-gate turns it back on), a dilated CNN
decoder with a waveform head and a breath-event head,
trained with the v16 waveform loss + heatmap BCE + soft-count loss. RR is
scored per complete 60 s minute against a fixed ground truth, with reference
quality control, breath matching and smart fusion (Karlen et al. 2013).

CapnoBase fixes (why about half of the CapnoBase subjects failed)
-----------------------------------------------------------------
CapnoBase (42 cases, 29 children, 13 adults, 300 Hz) breathes at 5-48 bpm,
median 11.7, and its reference is a sidestream capnogram. The v16 pipeline
assumed adult impedance/belt references at 6-30 bpm:

1. Band. Input, target and RR counting used a fixed 0.1-0.5 Hz band. A
   capnogram is a trapezoid with strong harmonics: for slow breathing the
   band keeps the harmonics, so the reference shows extra peaks (legacy
   count 15/min for true 5 and 8, 17/min for true 12 on synthetic
   capnograms), and above 30 bpm the fundamental is cut (22.9/min for true
   40). The capnography profile uses 0.05 Hz up to clip(0.5 x PPG heart
   rate, 0.5, 1.0) Hz (second-order-section filter), an RR band of 3-60
   bpm, and a 0.8 s minimum breath spacing.
2. Breath rules. RRest count-orig asks for exactly one sub-zero trough per
   cycle; a band-passed capnogram has a W-shaped trough, so every cycle was
   rejected and slow subjects lost all training windows. Clean cycles are
   now defined by one upward zero crossing, and breaths are timed at that
   onset (RRest uses zero crossings for CapnoBase, co2_zex) because the
   band-passed apex wanders across long plateaus. FFT RR checks f/2 and f/3
   so a harmonic is not taken for the breathing rate.
3. Ground truth. CapnoBase ships expert CO2 breath labels; they replace
   detection as the ground truth (start of expiration). Detection-vs-label
   agreement is reported per subject. In {case}_8min_labels.csv every
   annotation field (co2_startexp_x, co2_startinsp_x, pleth_peak_x, the
   *_artif_x artifact spans) holds all of its events in one cell as
   space-separated one-based sample numbers, with the unit in units_x. A
   reader that expects one number per cell finds no labels and silently
   falls back to detection; the reader here splits the cells, removes the
   MATLAB index base, takes sampling rates from the param file, checks the
   record length and cross-checks the label RR against the reference file.
   Expert artifact spans are used too: PPG artifacts are unusable input
   samples, and training windows or scored minutes that overlap a CO2
   artifact are excluded. --inspect-capnobase prints and saves how every
   case was read (fs, labels, label vs reference RR, CO2 rise at the
   labels, pulse-peak offset, artifacts) without training.
4. Delay and polarity. The PPG follows breathing mechanics, the capnogram
   follows gas exchange (about a quarter cycle later) plus the sidestream
   transport delay (1-3 s), and positive-pressure ventilation inverts the
   PPG modulation, so the zero-lag correlation that set the target sign
   changed sign with each subject's RR, delay and ventilation mode. The
   capnogram keeps its physical polarity (high CO2 = expiration) and is
   shifted by the lag of maximal positive PPG correlation within half a
   breath period (at least 3 s); counts do not depend on this shift.
   Flipping it instead (an earlier draft) moved the breath-onset targets by
   half a cycle for some subjects and kept the event head from learning.
5. Pulses and resampling. Infant heart rates exceed the v16 180 bpm limit;
   PPG beat QC uses 40-220 bpm and a beat spacing from the PPG's spectral
   heart rate. 300 -> 128 Hz uses polyphase resampling instead of FFT
   resampling, which treats the record as periodic and rings at the
   capnogram's edges.

BIDMC and STEAM2 keep the v16 adult processing (fixed 0.1-0.5 Hz, sign
alignment, 40-180 bpm); the breath-rule changes (zero-crossing cycles, onset
timing) apply to all datasets. Per-subject diagnostics (GT RR range, share
below 6 / above 30 bpm, band, lag, heart rate, detection-vs-label error)
are written to the per-subject results and the data-quality report.

Decoder (unchanged from the event-decoder version)
--------------------------------------------------
RR is scored by counting breaths, so every breath weighs the same, while
waveform losses weigh a breath by its squared amplitude. The decoder keeps
the v16 layers with dilations (8, 16) (receptive field 0.09 s -> 0.88 s, no
extra parameters) and adds a 1x1 breath-event head (+9 parameters) trained
on Gaussian bumps at reference breaths; it starts at the prior probability
and uses 10x the base learning rate. Event breaths are peaks with at least
30% of the local probability range.

Evaluation: legacy v16 count and PCC (unchanged definitions), event /
count-orig / FFT RR against the fixed ground truth on reference-valid
minutes, missed and extra breaths per minute, smart fusion (withheld when
the three estimates' SD exceeds 4 bpm) with the retained share, the median
of the three estimates on every valid minute (RR_MAE_Median3), and pooled
minute-level MAE/RMSE.

CapnoBase target choice: on a heterogeneous synthetic CapnoBase set (6-44
bpm, varying I:E split, EtCO2, delay, and inverted PPG modulation under
ventilation) the label phase target with the median of three gave the
lowest RR MAE (pooled 0.39 bpm vs 0.51 with the capnogram target; event
head 0.67 vs 1.00), so "phase" is the default. Synthetic results do not
guarantee the same order on the real recordings; --capnography-target co2
reproduces the alternative. The legacy count is not valid for capnograms outside
6-30 bpm; it is kept only for comparison with earlier results.
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
from dataclasses import dataclass
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

PROJECT_ROOT = Path(
    r"D:\PPG2RSP_RRWaveNet_Inspired\261001_RRWaveNet_fv"
)
DEFAULT_RESULTS_ROOT = PROJECT_ROOT / "Results"

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

DATASET_ORDER = ("bidmc", "capnobase", "steam2")
DATASET_LABELS = ("BIDMC", "CapnoBase", "STEAM2")


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

# Per-subject results of the v16 early-fusion model used as the paired
# baseline. The first file holds 10, 20 and 60 s; the second holds 10 and 20 s.
# Rows are filtered to Fusion_Position == "early_fusion"; the first file wins
# when the same dataset/subject/window appears in both.
DEFAULT_BASELINE_CSVS = (
    Path(
        r"D:\PPG2RSP_RRWaveNet_Inspired"
        r"\Ablation_StemFusion_Early_NoFusion_10s20s60sNonOverlap_v1\Results"
        r"\v16_deep_only_stem_fusion_position_depthwise_10s20s_nonoverlap_per_subject.csv"
    ),
    Path(
        r"D:\PPG2RSP_RRWaveNet_Inspired"
        r"\Ablation_StemFusion_Position_Depthwise_10s20sNonOverlap_v1\Results"
        r"\v16_deep_only_stem_fusion_position_depthwise_10s20s_nonoverlap_per_subject.csv"
    ),
)
BASELINE_KEY = "baseline_early_fusion"
BASELINE_LABEL = "v16 early fusion (baseline)"

# v16 input/output and training constants.
TARGET_FS = 128
EVAL_WINDOW_SEC = 60.0
EVAL_STRIDE_SEC = 60.0
BATCH_SIZE = 48
DEFAULT_EPOCHS = 120
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
LOSS_WEIGHT_COUNT = 0.02
# The event head starts from near-zero weights (so its first output is the
# prior) and Adam moves each weight by about one learning rate per step. At
# the base rate it cannot leave the prior within the ~840 steps a 60 s
# condition gets (7 steps/epoch x 120 epochs), so the new head uses 10x.
EVENT_HEAD_LR_MULTIPLIER = 10.0

# Breath detection, reference quality and smart fusion (Karlen et al. 2013;
# RRest / Charlton et al. 2016).
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
EVENT_SIGMA_SEC = 0.3  # Gaussian width of the breath-event target
EVENT_PRIOR = 0.15  # initial event probability (bias of the event head)
# Event peaks are local maxima of the event probability whose prominence is
# at least 30% of the local probability range (5th-95th percentile over the
# 60 s block and its context). A relative rule, because a heatmap trained with
# BCE lowers its peaks where breath timing is uncertain, so a fixed height
# would drop correct but less confident breaths.
EVENT_MIN_PROMINENCE_FRACTION = 0.3
EVENT_MIN_RANGE = 0.05  # below this the head is flat: no breath events
EVENT_MIN_DISTANCE_SEC = 1.5  # same minimum breath spacing as the v16 counter
PEAK_MATCH_TOLERANCE_SEC = 1.0  # used when fewer than two reference breaths
# Breath matching tolerance as a share of the local breath interval: a
# quarter cycle (1 s at 15 bpm, 2.5 s at 6 bpm, 0.31 s at 48 bpm), so timing
# is judged on the same phase scale at every breathing rate.
PEAK_MATCH_TOLERANCE_CYCLE_FRACTION = 0.25
SMART_FUSION_SD_BPM = 4.0  # Karlen et al. 2013
DISCORDANT_PCC = 0.8  # legacy minutes with high PCC but a large count error
DISCORDANT_AE_BPM = 3.0

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

# v16 Deep-only early-fusion configuration with proposals 4 and 5 and the
# breath-event decoder.
MODEL_CONFIG = {
    "stem_kernel_sizes": (32, 64, 128),
    "stem_base_channels": 8,
    "hidden_channels": 24,
    "encoder_kernel_size": 15,
    "encoder_dilations": (1, 2, 4, 8),
    "dropout": 0.20,
    # Keep the 24-channel representation as the decoder input; do not expand
    # it to 64 channels before reducing to the scalar waveform output.
    "decoder_mid_channels": (16, 8),
    "decoder_kernel_sizes": (7, 5, 1),
    # Breath-event decoder: dilations widen the decoder receptive field from
    # 11 samples (0.09 s) to 113 samples (0.88 s) with the same parameters;
    # a second 1x1 head predicts breath-apex logits next to the waveform.
    "decoder_dilations": (8, 16),
    "decoder_heads": ("waveform_tanh", "breath_event_logit"),
    "event_sigma_sec": EVENT_SIGMA_SEC,
    "stem_fusion_position": "early_fusion",
    "stem_fusion_activation": "none_after_1x1_conv_groupnorm",
    "encoder_channel_mixing": "depthwise_temporal_conv_groups24",
    "residual_encoder_cross_channel_mixing": False,
    "encoder_normalization": "per_channel_groupnorm_groups24",
    # Proposal 4: identity channel into the early stem fusion.
    "use_identity_channel": True,
    "identity_channel_normalization": "per_window_zscore",
    "identity_channel_bypasses_scale_gate": True,
    "identity_fusion_weight_init": "zeros",
    # Proposal 5: window-adaptive scale gate before the early stem fusion.
    # Off by default: the stem branches enter the fusion with fixed weights,
    # as in v16. --scale-gate turns it on.
    "use_scale_gate": False,
    "scale_gate_descriptor": "relative_log_energy_of_pre_groupnorm_stem_responses",
    "scale_gate_hidden": 8,
    "scale_gate_output": "softmax_over_branches_times_branch_count",
    "scale_gate_output_init": "zeros_uniform_branch_weights",
}
BRANCH_LABELS = tuple(f"k{size}" for size in MODEL_CONFIG["stem_kernel_sizes"])

# Legacy v16 metrics (direct count on stitched, window-normalized signals).
PERFORMANCE_METRICS = (
    "RR_MAE_BPM",
    "PCC_60s_Mean",
)
# Metrics against the fixed continuous ground truth on reference-valid minutes.
GT_PERFORMANCE_METRICS = (
    "RR_MAE_Event",
    "RR_MAE_Wave_CtO",
    "RR_MAE_Wave_FFT",
    "RR_Bias_Event",
    "Peak_Wrong_per_min",
    "Peak_Missed_per_min",
    "Peak_Extra_per_min",
    "Peak_Sensitivity",
    "Peak_PPV",
    "Peak_Timing_Error_ms",
    "RR_MAE_SmartFusion",
    "SF_Retained_Ratio",
    "RR_MAE_Median3",
    "Ref_Valid_Ratio",
)
COMPARISON_METRICS = (*PERFORMANCE_METRICS, *GT_PERFORMANCE_METRICS)
LOWER_IS_BETTER_METRICS = {
    "RR_MAE_BPM",
    "RR_MAE_Event",
    "RR_MAE_Wave_CtO",
    "RR_MAE_Wave_FFT",
    "Peak_Wrong_per_min",
    "Peak_Missed_per_min",
    "Peak_Extra_per_min",
    "Peak_Timing_Error_ms",
    "RR_MAE_SmartFusion",
    "RR_MAE_Median3",
}

FEATURE_METRICS = (
    "Feature_OffDiag_SignedMean",
    "Feature_OffDiag_AbsMean",
    "Feature_EffectiveRank",
    "Feature_EffectiveRank_Normalized",
    "Feature_ParticipationRatio",
)

GATE_METRICS = (
    *(f"Gate_{label}_Mean" for label in BRANCH_LABELS),
    *(f"Gate_{label}_SD" for label in BRANCH_LABELS),
    *(f"Gate_{label}_Spearman_vs_Target_Freq" for label in BRANCH_LABELS),
    "Gate_Normalized_Entropy_Mean",
    "Target_Dominant_Freq_Hz_Mean",
    "Fusion_Identity_Column_Norm",
    "Fusion_Identity_to_Stem_Norm_Ratio",
)

CONVERGENCE_METRICS = (
    "Best_Epoch",
    "Convergence_Epoch_95pct",
    "History_Epochs",
)

MODEL_COLORS = {
    BASELINE_KEY: "#4c78a8",
    "proposed": "#54a24b",
}
BRANCH_COLORS = ("#f58518", "#e45756", "#b279a2")


@dataclass(frozen=True)
class ModelOptions:
    use_identity_channel: bool = True
    use_scale_gate: bool = False

    @property
    def key(self) -> str:
        parts = ["early_fusion"]
        if self.use_identity_channel:
            parts.append("identity")
        if self.use_scale_gate:
            parts.append("scalegate")
        parts.append("eventdecoder")
        return "_".join(parts)

    @property
    def label(self) -> str:
        additions = []
        if self.use_identity_channel:
            additions.append("identity channel")
        if self.use_scale_gate:
            additions.append("scale gate")
        additions.append("breath-event decoder")
        return "v16 early fusion + " + " + ".join(additions)

    def build(self) -> "DeepOnlyV16EventDecoder":
        return DeepOnlyV16EventDecoder(
            use_identity_channel=self.use_identity_channel,
            use_scale_gate=self.use_scale_gate,
        )


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


def detect_event_breaths(
    probability: np.ndarray,
    fs: int,
    min_interval_sec: float = EVENT_MIN_DISTANCE_SEC,
    rate_hint_breaths: Optional[np.ndarray] = None,
) -> np.ndarray:
    """Breath events from a continuous event-probability trace.

    Processed in the same 60 s blocks with 10 s context as the ground truth,
    so the relative prominence threshold follows slow changes in confidence.

    `rate_hint_breaths` are breath onsets from the model's waveform head
    (same time base). When given, peaks closer than half the local median
    breath interval are treated as one breath: at slow rates breath timing
    from the PPG is uncertain by a second or more, the event head answers
    with a broad bump, and a fixed 0.8 s spacing (sized for infant rates)
    splits that bump into two or three events. Only model outputs are used.
    """
    length = len(probability)
    block = int(round(BREATH_BLOCK_SEC * fs))
    margin = int(round(BREATH_BLOCK_MARGIN_SEC * fs))
    found = []
    for block_start in range(0, length, block):
        block_end = min(length, block_start + block)
        low = max(0, block_start - margin)
        high = min(length, block_end + margin)
        local = probability[low:high]
        spread = float(np.percentile(local, 95) - np.percentile(local, 5))
        if spread < EVENT_MIN_RANGE:
            continue
        interval_sec = min_interval_sec
        if rate_hint_breaths is not None:
            hints = rate_hint_breaths[
                (rate_hint_breaths >= low) & (rate_hint_breaths < high)
            ]
            if len(hints) >= 3:
                interval_sec = max(
                    min_interval_sec,
                    0.5 * float(np.median(np.diff(hints))) / float(fs),
                )
        peaks, _ = find_peaks(
            local,
            distance=max(1, int(round(interval_sec * fs))),
            prominence=EVENT_MIN_PROMINENCE_FRACTION * spread,
        )
        peaks = peaks + low
        found.append(peaks[(peaks >= block_start) & (peaks < block_end)])
    if not found:
        return np.empty(0, dtype=np.int64)
    return np.unique(np.concatenate(found)).astype(np.int64)


def breath_event_target(
    peaks: np.ndarray,
    start: int,
    length: int,
    fs: int,
) -> np.ndarray:
    """Gaussian breath heatmap (height 1 at every reference breath onset)."""
    sigma = EVENT_SIGMA_SEC * float(fs)
    reach = int(math.ceil(4.0 * sigma))
    near = peaks[(peaks >= start - reach) & (peaks < start + length + reach)]
    if len(near) == 0:
        return np.zeros(length, dtype=np.float32)
    samples = np.arange(start, start + length, dtype=np.float64)
    bumps = np.exp(-0.5 * ((samples[None, :] - near[:, None]) / sigma) ** 2)
    return bumps.max(axis=0).astype(np.float32)


def match_breaths(
    reference: np.ndarray,
    predicted: np.ndarray,
    tolerance: float,
) -> Tuple[np.ndarray, np.ndarray]:
    """Greedy one-to-one matching of breath times within a tolerance.

    Returns index arrays (into reference, into predicted) of matched pairs,
    closest pairs first.
    """
    reference = np.asarray(reference, dtype=np.float64)
    predicted = np.asarray(predicted, dtype=np.float64)
    if len(reference) == 0 or len(predicted) == 0:
        empty = np.empty(0, dtype=np.int64)
        return empty, empty
    distance = np.abs(reference[:, None] - predicted[None, :])
    rows, cols = np.nonzero(distance <= tolerance)
    order = np.argsort(distance[rows, cols], kind="stable")
    used_ref = np.zeros(len(reference), dtype=bool)
    used_pred = np.zeros(len(predicted), dtype=bool)
    matched_ref = []
    matched_pred = []
    for row, col in zip(rows[order], cols[order]):
        if not used_ref[row] and not used_pred[col]:
            used_ref[row] = True
            used_pred[col] = True
            matched_ref.append(row)
            matched_pred.append(col)
    return np.asarray(matched_ref, dtype=np.int64), np.asarray(matched_pred, dtype=np.int64)


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

    low_kept = []
    rsp_kept = []
    event_kept = []
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
            low_kept.append(ppg_low[int(start):end])
            rsp_kept.append(target_signal[int(start):end])
            event_kept.append(
                breath_event_target(reference_breaths, int(start), window_len, fs)
            )
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
    }
    if not low_kept:
        return {
            "low": np.empty((0, window_len), dtype=np.float32),
            "rsp": np.empty((0, window_len), dtype=np.float32),
            "event": np.empty((0, window_len), dtype=np.float32),
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

    Training and validation keep only windows whose reference passes the
    respiratory quality rule (Train_OK); the split itself is made on all kept
    windows first so the temporal train/validation boundary is unchanged.
    Test uses every kept window: inference never depends on the reference.
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
        x = record["low"][window_index][None, :].astype(
            np.float32,
            copy=False,
        )
        y = record["rsp"][window_index][None, :].astype(
            np.float32,
            copy=False,
        )
        x = instance_minmax_normalize(x, (0.0, 1.0))
        y = instance_minmax_normalize(y, (-1.0, 1.0))[0]
        x_tensor = torch.from_numpy(np.asarray(x, dtype=np.float32).copy())
        y_tensor = torch.from_numpy(np.asarray(y, dtype=np.float32).copy())
        e_tensor = torch.from_numpy(
            np.asarray(record["event"][window_index], dtype=np.float32).copy()
        )
        if self.augment:
            shift = random.randint(-12, 12)
            scale = random.uniform(0.95, 1.05)
            noise_std = random.uniform(0.002, 0.01)
            if shift:
                x_tensor = torch.roll(x_tensor, shifts=shift, dims=-1)
                y_tensor = torch.roll(y_tensor, shifts=shift, dims=-1)
                e_tensor = torch.roll(e_tensor, shifts=shift, dims=-1)
            x_tensor = x_tensor * scale + torch.randn_like(x_tensor) * noise_std
        return x_tensor, y_tensor.unsqueeze(0), e_tensor.unsqueeze(0)


def make_loaders(
    subject_data: Dict[str, dict],
    train_subjects: Sequence[str],
    test_subject: str,
    device: torch.device,
    batch_size: int = BATCH_SIZE,
):
    train_dataset = SubjectWindowDataset(
        subject_data,
        train_subjects,
        "train",
        True,
    )
    val_dataset = SubjectWindowDataset(
        subject_data,
        train_subjects,
        "val",
        False,
    )
    test_dataset = SubjectWindowDataset(
        subject_data,
        [test_subject],
        "test",
        False,
    )
    kwargs = {
        "batch_size": int(batch_size),
        "drop_last": False,
        "num_workers": 0,
        "pin_memory": device.type == "cuda",
    }
    return (
        DataLoader(
            train_dataset,
            shuffle=True,
            **kwargs,
        ),
        DataLoader(
            val_dataset,
            shuffle=False,
            **kwargs,
        ),
        DataLoader(
            test_dataset,
            shuffle=False,
            **kwargs,
        ),
        test_metadata(subject_data, test_subject),
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
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        dilation: int,
        dropout: float,
    ):
        super().__init__()
        if in_channels != out_channels:
            raise ValueError(
                "Depthwise residual encoder requires equal input and output channels"
            )
        padding = int(dilation) * (int(kernel_size) // 2)
        groups = int(in_channels)
        self.conv1 = nn.Conv1d(
            in_channels,
            out_channels,
            kernel_size,
            padding=padding,
            dilation=dilation,
            groups=groups,
        )
        self.norm1 = nn.GroupNorm(out_channels, out_channels)
        self.conv2 = nn.Conv1d(
            out_channels,
            out_channels,
            kernel_size,
            padding=padding,
            dilation=dilation,
            groups=groups,
        )
        self.norm2 = nn.GroupNorm(out_channels, out_channels)
        self.act = nn.GELU()
        self.drop = nn.Dropout(dropout)
        self.shortcut = (
            nn.Conv1d(in_channels, out_channels, 1)
            if in_channels != out_channels
            else nn.Identity()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = self.shortcut(x)
        out = self.drop(self.act(self.norm1(self.conv1(x))))
        out = self.drop(self.norm2(self.conv2(out)))
        return self.act(out + residual)


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


def make_encoder() -> nn.Sequential:
    hidden = MODEL_CONFIG["hidden_channels"]
    return nn.Sequential(
        *[
            ResidualConvBlock(
                hidden,
                hidden,
                MODEL_CONFIG["encoder_kernel_size"],
                dilation,
                MODEL_CONFIG["dropout"],
            )
            for dilation in MODEL_CONFIG["encoder_dilations"]
        ]
    )


# ---------------------------------------------------------------------------
# Early stem fusion with identity channel (proposal 4) and scale gate (proposal 5)
# ---------------------------------------------------------------------------

SCALE_GATE_ENERGY_EPS = 1e-6
IDENTITY_CHANNEL_EPS = 1e-6


class StemBranch(nn.Module):
    """v16 stem branch that also reports its pre-normalization response energy.

    The feature path (explicit-same Conv1d -> GroupNorm -> GELU) is the v16
    branch. The per-channel GroupNorm removes each window's response energy,
    so the log-energy is taken from the convolution output before the norm.
    """

    def __init__(self, stem_channels: int, kernel_size: int):
        super().__init__()
        self.conv = Conv1dExplicitSame(1, stem_channels, kernel_size)
        self.norm = nn.GroupNorm(
            choose_group_count(stem_channels),
            stem_channels,
        )
        self.act = nn.GELU()

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        response = self.conv(x)
        log_energy = 0.5 * torch.log(
            response.var(dim=-1, unbiased=False) + SCALE_GATE_ENERGY_EPS
        )
        return self.act(self.norm(response)), log_energy


class ScaleGate(nn.Module):
    """Window-adaptive weighting of the multi-scale stem branches.

    The descriptor is the log-energy of every stem filter response relative
    to the window mean, i.e. the window's spectral profile seen through the
    stem filters. Subtracting the mean makes it invariant to the input
    amplitude, like the rest of the stem. The softmax keeps the total branch
    weight fixed, so the gate redistributes emphasis between scales instead
    of changing the overall stem gain.
    """

    def __init__(self, stem_channels: int, branch_count: int, hidden: int):
        super().__init__()
        self.stem_channels = int(stem_channels)
        self.branch_count = int(branch_count)
        self.mlp = nn.Sequential(
            nn.Linear(self.stem_channels * self.branch_count, int(hidden)),
            nn.GELU(),
            nn.Linear(int(hidden), self.branch_count),
        )

    def reset_to_uniform(self) -> None:
        nn.init.zeros_(self.mlp[-1].weight)
        nn.init.zeros_(self.mlp[-1].bias)

    def forward(
        self,
        stem: torch.Tensor,
        log_energy: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        descriptor = log_energy - log_energy.mean(dim=1, keepdim=True)
        weights = torch.softmax(self.mlp(descriptor), dim=-1) * self.branch_count
        channel_weights = weights.repeat_interleave(self.stem_channels, dim=1)
        return stem * channel_weights.unsqueeze(-1), weights


def standardized_identity_channel(x: torch.Tensor) -> torch.Tensor:
    """Per-window z-scored copy of the model input."""
    centered = x - x.mean(dim=-1, keepdim=True)
    scale = torch.sqrt(
        centered.pow(2).mean(dim=-1, keepdim=True) + IDENTITY_CHANNEL_EPS
    )
    return centered / scale


# ---------------------------------------------------------------------------
# Breath-event decoder (the only structural change to the network)
# ---------------------------------------------------------------------------

def make_breath_decoder_trunk() -> nn.Sequential:
    """v16 decoder layers with dilated convolutions and no output layer.

    Same layers, channels, kernels and parameter count as the v16 decoder;
    dilations (8, 16) widen the receptive field from 11 to 113 samples
    (0.09 s -> 0.88 s at 128 Hz), enough to see the rise and fall around a
    breath apex instead of mapping each time point on its own.
    """
    hidden = MODEL_CONFIG["hidden_channels"]
    mid1, mid2 = MODEL_CONFIG["decoder_mid_channels"]
    kernel1, kernel2, _ = MODEL_CONFIG["decoder_kernel_sizes"]
    dilation1, dilation2 = MODEL_CONFIG["decoder_dilations"]
    return nn.Sequential(
        nn.Conv1d(
            hidden,
            mid1,
            kernel1,
            padding=dilation1 * (kernel1 // 2),
            dilation=dilation1,
        ),
        nn.GroupNorm(choose_group_count(mid1), mid1),
        nn.GELU(),
        nn.Dropout(MODEL_CONFIG["dropout"]),
        nn.Conv1d(
            mid1,
            mid2,
            kernel2,
            padding=dilation2 * (kernel2 // 2),
            dilation=dilation2,
        ),
        nn.GroupNorm(choose_group_count(mid2), mid2),
        nn.GELU(),
        nn.Dropout(MODEL_CONFIG["dropout"]),
    )


class DeepOnlyV16EventDecoder(nn.Module):
    """v16 early stem fusion (identity channel, optional scale gate) + event decoder.

    stem branches -> [scale gate, off by default] -> [stem, identity channel]
    -> 1x1 fusion + GroupNorm (no GELU) -> depthwise residual encoder
    -> dilated decoder trunk -> {waveform head (tanh), breath-event head}.
    The waveform head is the v16 output layer (1x1 conv + tanh); the event
    head is a parallel 1x1 conv producing breath-apex logits.
    """

    def __init__(
        self,
        use_identity_channel: bool = True,
        use_scale_gate: bool = False,
        stem_base_channels: Optional[int] = None,
    ):
        super().__init__()
        stem_channels = int(
            MODEL_CONFIG["stem_base_channels"]
            if stem_base_channels is None
            else stem_base_channels
        )
        if stem_channels <= 0:
            raise ValueError("stem_base_channels must be positive")
        hidden = MODEL_CONFIG["hidden_channels"]
        kernel_sizes = tuple(MODEL_CONFIG["stem_kernel_sizes"])
        self.stem_base_channels = stem_channels
        self.branch_labels = tuple(f"k{size}" for size in kernel_sizes)
        self.use_identity_channel = bool(use_identity_channel)
        self.stem_branches = nn.ModuleList(
            [StemBranch(stem_channels, size) for size in kernel_sizes]
        )
        self.scale_gate = (
            ScaleGate(
                stem_channels,
                len(kernel_sizes),
                MODEL_CONFIG["scale_gate_hidden"],
            )
            if use_scale_gate
            else None
        )
        fusion_inputs = stem_channels * len(kernel_sizes) + int(
            self.use_identity_channel
        )
        self.stem_fuse = nn.Sequential(
            nn.Conv1d(fusion_inputs, hidden, 1),
            nn.GroupNorm(choose_group_count(hidden), hidden),
        )
        self.encoder = make_encoder()
        decoder_channels = MODEL_CONFIG["decoder_mid_channels"][-1]
        self.decoder = make_breath_decoder_trunk()
        self.wave_head = nn.Conv1d(decoder_channels, 1, 1)
        self.event_head = nn.Conv1d(decoder_channels, 1, 1)
        self._init_weights()
        self._init_additions()

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

    def _init_additions(self) -> None:
        """Start the additions from neutral values.

        Runs after _init_weights, which would otherwise overwrite these
        initializations with Kaiming weights. The gate starts uniform, the
        identity channel starts unused, and the event head starts at the
        prior breath-apex probability: small weights (std 0.01, as for
        detection heads in RetinaNet/CenterNet) so the bias sets the output,
        because Kaiming fan-out on a single output channel gives std 1.4.
        """
        if self.scale_gate is not None:
            self.scale_gate.reset_to_uniform()
        if self.use_identity_channel:
            with torch.no_grad():
                self.stem_fuse[0].weight[:, -1:, :].zero_()
        nn.init.normal_(self.event_head.weight, mean=0.0, std=0.01)
        nn.init.constant_(
            self.event_head.bias,
            math.log(EVENT_PRIOR / (1.0 - EVENT_PRIOR)),
        )

    def fused_representation(
        self,
        x: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        responses = [branch(x) for branch in self.stem_branches]
        stem = torch.cat([features for features, _ in responses], dim=1)
        if self.scale_gate is None:
            gate = stem.new_ones(stem.shape[0], len(responses))
        else:
            log_energy = torch.cat([energy for _, energy in responses], dim=1)
            stem, gate = self.scale_gate(stem, log_energy)
        if self.use_identity_channel:
            stem = torch.cat([stem, standardized_identity_channel(x)], dim=1)
        return self.stem_fuse(stem), gate

    def forward_with_diagnostics(
        self,
        x: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return (waveform, event logits, post-fusion features, gate weights)."""
        fused, gate = self.fused_representation(x)
        features = self.decoder(self.encoder(fused))
        waveform = torch.tanh(self.wave_head(features))
        return waveform, self.event_head(features), fused, gate

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        waveform, event_logits, _, _ = self.forward_with_diagnostics(x)
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
      soft count is the heatmap integral divided by the area of one Gaussian
      bump, so this is a differentiable surrogate of the per-minute count
      error that RR MAE measures; it also calibrates the heatmap so each
      breath yields one bump of the right size.
    """

    def __init__(self, fs: int = TARGET_FS):
        super().__init__()
        self.wave_loss = HybridLoss()
        self.event_loss = nn.BCEWithLogitsLoss()
        self.fs = float(fs)
        self.bump_area = EVENT_SIGMA_SEC * math.sqrt(2.0 * math.pi) * float(fs)

    def forward(
        self,
        wave_pred: torch.Tensor,
        event_logits: torch.Tensor,
        wave_target: torch.Tensor,
        event_target: torch.Tensor,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        wave = self.wave_loss(wave_pred, wave_target)
        event = self.event_loss(event_logits, event_target)
        minutes = event_target.shape[-1] / self.fs / 60.0
        predicted = torch.sigmoid(event_logits).sum(dim=-1) / self.bump_area
        target = event_target.sum(dim=-1) / self.bump_area
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
        for x, y, e in loader:
            wave, logits = model(x.to(device))
            loss, parts = criterion(wave, logits, y.to(device), e.to(device))
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
# Test-time diagnostics of the fused representation and the scale gate
# ---------------------------------------------------------------------------

def predict_with_diagnostics(
    model: DeepOnlyV16EventDecoder,
    loader: DataLoader,
    device: torch.device,
) -> Dict[str, np.ndarray]:
    """Waveforms, event probabilities, targets, post-fusion features, gates."""
    model.eval()
    chunks = {
        "predictions": [],
        "event_probabilities": [],
        "targets": [],
        "event_targets": [],
        "features": [],
        "gates": [],
    }
    with torch.no_grad():
        for x, y, e in loader:
            wave, logits, fused, gate = model.forward_with_diagnostics(x.to(device))
            chunks["predictions"].append(wave.squeeze(1).cpu().numpy())
            chunks["event_probabilities"].append(
                torch.sigmoid(logits).squeeze(1).cpu().numpy()
            )
            chunks["targets"].append(y.squeeze(1).numpy())
            chunks["event_targets"].append(e.squeeze(1).numpy())
            chunks["features"].append(fused.cpu().numpy().astype(np.float32))
            chunks["gates"].append(gate.cpu().numpy().astype(np.float32))
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


def dominant_frequency_hz(
    windows: np.ndarray,
    fs: int,
    band: Tuple[float, float] = (0.1, 0.5),
) -> np.ndarray:
    """Dominant respiratory-band frequency of each target window."""
    values = np.asarray(windows, dtype=np.float64)
    if values.ndim != 2 or values.shape[0] == 0:
        return np.empty(0, dtype=np.float64)
    n_fft = max(8192, values.shape[1])
    centered = values - values.mean(axis=1, keepdims=True)
    spectrum = np.abs(np.fft.rfft(centered, n=n_fft, axis=1))
    freqs = np.fft.rfftfreq(n_fft, d=1.0 / float(fs))
    in_band = (freqs >= band[0]) & (freqs <= band[1])
    return freqs[in_band][np.argmax(spectrum[:, in_band], axis=1)]


def safe_spearman(x: np.ndarray, y: np.ndarray) -> float:
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    if len(x) < 3 or len(x) != len(y):
        return np.nan
    if np.ptp(x) < 1e-12 or np.ptp(y) < 1e-12:
        return np.nan
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return float(stats.spearmanr(x, y)[0])


def gate_diagnostics(
    gates: np.ndarray,
    targets: np.ndarray,
    fs: int,
    branch_labels: Sequence[str],
) -> dict:
    """Summarize per-window branch weights of one held-out subject.

    Spearman correlations relate each branch weight to the target's dominant
    respiratory frequency, i.e. whether the gate follows the breathing rate.
    """
    labels = list(branch_labels)
    result = {
        "Gate_N_Windows": 0,
        "Gate_Normalized_Entropy_Mean": np.nan,
        "Target_Dominant_Freq_Hz_Mean": np.nan,
    }
    for label in labels:
        result[f"Gate_{label}_Mean"] = np.nan
        result[f"Gate_{label}_SD"] = np.nan
        result[f"Gate_{label}_Spearman_vs_Target_Freq"] = np.nan
    weights = np.asarray(gates, dtype=np.float64)
    if weights.ndim != 2 or weights.shape[0] == 0 or weights.shape[1] != len(labels):
        return result
    count = int(weights.shape[0])
    result["Gate_N_Windows"] = count
    if len(labels) > 1:
        probabilities = weights / np.maximum(weights.sum(axis=1, keepdims=True), 1e-12)
        probabilities = np.clip(probabilities, 1e-12, 1.0)
        entropy = -(probabilities * np.log(probabilities)).sum(axis=1)
        result["Gate_Normalized_Entropy_Mean"] = float(
            np.mean(entropy / np.log(len(labels)))
        )
    frequency = dominant_frequency_hz(targets, fs)
    if len(frequency):
        result["Target_Dominant_Freq_Hz_Mean"] = float(np.mean(frequency))
    for index, label in enumerate(labels):
        column = weights[:, index]
        result[f"Gate_{label}_Mean"] = float(np.mean(column))
        result[f"Gate_{label}_SD"] = (
            float(np.std(column, ddof=1)) if count > 1 else 0.0
        )
        if len(frequency) == count:
            result[f"Gate_{label}_Spearman_vs_Target_Freq"] = safe_spearman(
                column,
                frequency,
            )
    return result


def fusion_weight_diagnostics(model: DeepOnlyV16EventDecoder) -> dict:
    """Column norms of the 1x1 fusion weight per stem branch and identity."""
    weight = model.stem_fuse[0].weight.detach().cpu().numpy()[:, :, 0]
    column_norms = np.linalg.norm(weight.astype(np.float64), axis=0)
    stem_channels = model.stem_base_channels
    stem_norms = column_norms[: stem_channels * len(model.branch_labels)]
    stem_mean = float(np.mean(stem_norms))
    result = {"Fusion_Stem_Column_Norm_Mean": stem_mean}
    for index, label in enumerate(model.branch_labels):
        block = column_norms[index * stem_channels : (index + 1) * stem_channels]
        result[f"Fusion_{label}_Column_Norm_Mean"] = float(np.mean(block))
    if model.use_identity_channel:
        identity = float(column_norms[-1])
        result["Fusion_Identity_Column_Norm"] = identity
        result["Fusion_Identity_to_Stem_Norm_Ratio"] = identity / max(1e-12, stem_mean)
    else:
        result["Fusion_Identity_Column_Norm"] = np.nan
        result["Fusion_Identity_to_Stem_Norm_Ratio"] = np.nan
    return result


# ---------------------------------------------------------------------------
# One LOSOCV fold
# ---------------------------------------------------------------------------

def train_one(
    dataset_key: str,
    spec: WindowSpec,
    test_subject: str,
    subject_data: Dict[str, dict],
    train_subjects: Sequence[str],
    fold_root: Path,
    device: torch.device,
    epochs: int,
    min_epochs: int,
    patience: int,
    batch_size: int,
    fold_seed: int,
    model_options: ModelOptions,
    save_features: bool = False,
) -> Tuple[Dict[str, np.ndarray], List[dict], dict, Path, dict]:
    set_seed(fold_seed)
    model_root = fold_root / "Models"
    curve_root = fold_root / "Learning_Curves"
    audit_root = fold_root / "Training_Audits"
    prediction_root = fold_root / "Predictions"
    for path in (model_root, curve_root, audit_root, prediction_root):
        path.mkdir(parents=True, exist_ok=True)

    tag = safe_tag(test_subject)
    checkpoint = model_root / f"{tag}_best.pth"
    history_path = curve_root / f"{tag}_Learning_Curve.csv"
    audit_path = audit_root / f"{tag}_Training_Audit.csv"
    prediction_path = prediction_root / f"{tag}_test_outputs.npz"

    train_loader, val_loader, test_loader, metadata = make_loaders(
        subject_data,
        train_subjects,
        test_subject,
        device,
        batch_size,
    )
    model = model_options.build().to(device)
    criterion = WaveEventLoss()
    event_head_params = list(model.event_head.parameters())
    event_head_ids = {id(param) for param in event_head_params}
    optimizer = torch.optim.AdamW(
        [
            {
                "params": [
                    param
                    for param in model.parameters()
                    if id(param) not in event_head_ids
                ]
            },
            {
                "params": event_head_params,
                "lr": LEARNING_RATE * EVENT_HEAD_LR_MULTIPLIER,
            },
        ],
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
    )
    scheduler = CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)
    best_val = np.inf
    patience_counter = 0
    history = []
    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        for x, y, e in train_loader:
            x = x.to(device)
            y = y.to(device)
            e = e.to(device)
            optimizer.zero_grad(set_to_none=True)
            wave, logits = model(x)
            loss, _ = criterion(wave, logits, y, e)
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
        scheduler.step()
        history.append(
            {
                "Epoch": epoch + 1,
                "Train_Loss": train_loss,
                "Val_Loss": val_loss,
                "Val_Wave_Loss": val_parts["wave"],
                "Val_Event_BCE": val_parts["event"],
                "Val_Count_Error_BPM": val_parts["count_bpm"],
            }
        )
        if val_loss < best_val - 1e-4:
            best_val = val_loss
            atomic_torch_save(model.state_dict(), checkpoint)
            patience_counter = 0
        elif epoch + 1 >= min_epochs:
            patience_counter += 1
        if epoch + 1 >= min_epochs and patience_counter >= patience:
            break

    history_frame = pd.DataFrame(history)
    history_frame.to_csv(history_path, index=False, encoding="utf-8-sig")
    audit = build_training_audit(history, checkpoint)
    audit["Convergence_Epoch_95pct"] = convergence_epoch_95pct(history)
    pd.DataFrame([audit]).to_csv(audit_path, index=False, encoding="utf-8-sig")
    if audit["Audit_Status"] != "PASS_BEST_VALIDATION_CHECKPOINT":
        raise RuntimeError(f"No valid checkpoint: {checkpoint}")

    model.load_state_dict(torch.load(checkpoint, map_location=device))
    outputs = predict_with_diagnostics(model, test_loader, device)
    features = outputs.pop("features")
    gates = outputs["gates"]
    targets = outputs["targets"]
    best_row = history_frame.loc[history_frame["Epoch"] == audit["Best_Epoch"]].iloc[0]
    diagnostics = {
        "N_Params": count_parameters(model),
        "Train_Windows": len(train_loader.dataset),
        "Val_Windows": len(val_loader.dataset),
        "Val_Count_Error_BPM_At_Best": float(best_row["Val_Count_Error_BPM"]),
        "Val_Event_BCE_At_Best": float(best_row["Val_Event_BCE"]),
    }
    diagnostics.update(feature_representation_summary(features))
    diagnostics.update(
        gate_diagnostics(gates, targets, TARGET_FS, model.branch_labels)
    )
    diagnostics.update(fusion_weight_diagnostics(model))
    arrays = {
        "predictions": outputs["predictions"],
        "event_probabilities": outputs["event_probabilities"],
        "targets": targets,
        "event_targets": outputs["event_targets"],
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
        "gate_weights": gates.astype(np.float32),
        "gate_branch_labels": np.asarray(model.branch_labels),
        "target_dominant_freq_hz": dominant_frequency_hz(
            targets,
            TARGET_FS,
        ).astype(np.float32),
        "model_key": np.asarray(model_options.key),
    }
    if save_features:
        arrays["feature_representation"] = features.astype(np.float32)
        arrays["feature_representation_location"] = np.asarray(
            "post_stem_fusion"
        )
    np.savez_compressed(prediction_path, **arrays)
    del features, arrays, train_loader, val_loader, test_loader
    del model, criterion, optimizer, scheduler
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


def direct_rr_bpm(
    signal: np.ndarray,
    fs: int,
) -> Tuple[float, int, np.ndarray]:
    filtered, _ = bandpass_filter(signal, fs, 0.1, 0.5)
    spread = float(np.std(filtered))
    if len(filtered) < 2 or spread < 1e-8:
        return np.nan, 0, filtered
    distance = max(1, int(round(1.5 * fs)))
    peaks, _ = find_peaks(
        filtered,
        distance=distance,
        prominence=max(1e-8, 0.15 * spread),
    )
    duration_sec = len(filtered) / float(fs)
    bpm = float(len(peaks) * 60.0 / duration_sec)
    return bpm, int(len(peaks)), filtered


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


def _sd_or_nan(values: Sequence[float]) -> float:
    values = np.asarray(values, dtype=np.float64)
    values = values[np.isfinite(values)]
    if len(values) > 1:
        return float(np.std(values, ddof=1))
    return 0.0 if len(values) == 1 else np.nan


def summarize_subject_minutes(frame: pd.DataFrame) -> dict:
    """Subject-level summary of the per-minute evaluation table."""
    summary = {
        "RR_MAE_BPM": np.nan,
        "RR_MAE_BPM_SD_60s": np.nan,
        "PCC_60s_Mean": np.nan,
        "PCC_60s_SD": np.nan,
        "N60": 0,
        "N_Valid_RR_60s": 0,
        "N_Valid_PCC_60s": 0,
        "N60_RefValid": 0,
        "Ref_Valid_Ratio": np.nan,
        **{f"N60_Excluded_{reason}": 0 for reason in REFERENCE_EXCLUSION_REASONS},
        "RR_MAE_Event": np.nan,
        "RR_MAE_Event_SD_60s": np.nan,
        "RR_MAE_Wave_CtO": np.nan,
        "RR_MAE_Wave_FFT": np.nan,
        "RR_Bias_Event": np.nan,
        "Peak_Wrong_per_min": np.nan,
        "Peak_Missed_per_min": np.nan,
        "Peak_Extra_per_min": np.nan,
        "Peak_Sensitivity": np.nan,
        "Peak_PPV": np.nan,
        "Peak_Timing_Error_ms": np.nan,
        "SF_N_Retained": 0,
        "SF_Retained_Ratio": np.nan,
        "RR_MAE_SmartFusion": np.nan,
        "RR_MAE_Median3": np.nan,
        "N60_Discordant_Legacy": 0,
    }
    if frame.empty:
        return summary
    legacy_rr = pd.to_numeric(frame["RR_AE_BPM_Counting"], errors="coerce").dropna()
    legacy_pcc = pd.to_numeric(frame["PCC_60s_Counting"], errors="coerce").dropna()
    summary.update(
        {
            "RR_MAE_BPM": _mean_or_nan(legacy_rr),
            "RR_MAE_BPM_SD_60s": _sd_or_nan(legacy_rr),
            "PCC_60s_Mean": _mean_or_nan(legacy_pcc),
            "PCC_60s_SD": _sd_or_nan(legacy_pcc),
            "N60": int(len(frame)),
            "N_Valid_RR_60s": int(len(legacy_rr)),
            "N_Valid_PCC_60s": int(len(legacy_pcc)),
            "N60_Discordant_Legacy": int(frame["Discordant_Legacy"].sum()),
        }
    )
    for reason in REFERENCE_EXCLUSION_REASONS:
        summary[f"N60_Excluded_{reason}"] = int(
            (frame["Ref_Exclusion_Reason"] == reason).sum()
        )
    valid = frame[frame["Ref_Valid"].astype(bool)]
    summary["N60_RefValid"] = int(len(valid))
    summary["Ref_Valid_Ratio"] = float(len(valid) / len(frame))
    if valid.empty:
        return summary
    true_positive = float(valid["Peak_TP"].sum())
    missed = float(valid["Peak_Missed"].sum())
    extra = float(valid["Peak_Extra"].sum())
    matched_time = float(valid["Peak_Timing_Error_Sum_ms"].sum())
    retained = valid[valid["SF_Retained"].astype(bool)]
    summary.update(
        {
            "RR_MAE_Event": _mean_or_nan(valid["AE_Event"]),
            "RR_MAE_Event_SD_60s": _sd_or_nan(valid["AE_Event"]),
            "RR_MAE_Wave_CtO": _mean_or_nan(valid["AE_Wave_CtO"]),
            "RR_MAE_Wave_FFT": _mean_or_nan(valid["AE_Wave_FFT"]),
            "RR_Bias_Event": _mean_or_nan(valid["Signed_Error_Event"]),
            "Peak_Wrong_per_min": _mean_or_nan(valid["Peak_Wrong"]),
            "Peak_Missed_per_min": _mean_or_nan(valid["Peak_Missed"]),
            "Peak_Extra_per_min": _mean_or_nan(valid["Peak_Extra"]),
            "Peak_Sensitivity": true_positive / (true_positive + missed)
            if true_positive + missed > 0
            else np.nan,
            "Peak_PPV": true_positive / (true_positive + extra)
            if true_positive + extra > 0
            else np.nan,
            "Peak_Timing_Error_ms": matched_time / true_positive
            if true_positive > 0
            else np.nan,
            "SF_N_Retained": int(len(retained)),
            "SF_Retained_Ratio": float(len(retained) / len(valid)),
            "RR_MAE_SmartFusion": _mean_or_nan(retained["AE_SmartFusion"]),
            "RR_MAE_Median3": _mean_or_nan(valid["AE_Median3"]),
        }
    )
    return summary


def evaluate_subject_minutes(
    outputs: Dict[str, np.ndarray],
    metadata: Sequence[dict],
    references: Dict[int, dict],
    fs: int,
) -> Tuple[dict, pd.DataFrame]:
    """Score every complete 60 s minute of one held-out subject.

    Minutes are the same as in v16 (non-overlapping 60 s slices of each
    contiguous run of reconstructed windows). Per minute:

    * legacy: direct peak count and PCC on the stitched, window-normalized
      prediction and target (unchanged v16 definition);
    * reference: with expert labels (CapnoBase) the labels are the ground
      truth and a minute needs two labelled breaths and an RR in band;
      otherwise the minute of the continuous reference must pass the RRest
      agreement rule and the ground truth is the number of breath onsets
      detected once on the continuous reference; excluded minutes keep a
      reason. Band, RR range and breath spacing follow the segment's profile;
    * model estimates: event-head breaths (relative-prominence peaks of the
      event probability, at least half the waveform head's local breath
      interval apart), count-orig breaths of the
      predicted waveform, and the FFT RR of the predicted waveform;
    * breath matching: event breaths vs reference breaths within a quarter
      of the median breath interval, matched on the whole run so that
      minute edges do not split a pair;
    * smart fusion: mean of the three estimates, withheld when their SD
      exceeds 4 breaths/min (Karlen et al. 2013);
    * median of the three estimates on every valid minute (no withholding).
    """
    predictions = outputs["predictions"]
    targets = outputs["targets"]
    events = outputs["event_probabilities"]
    if len(predictions) == 0:
        return summarize_subject_minutes(pd.DataFrame()), pd.DataFrame()

    eval_len = int(round(EVAL_WINDOW_SEC * fs))
    per_minute = 60.0 / EVAL_WINDOW_SEC
    rows = []
    for segment_id in sorted({int(item["segment_id"]) for item in metadata}):
        pred_series, true_series, coverage = reconstruct_segment(
            predictions, targets, metadata, segment_id, fs
        )
        event_series, _, _ = reconstruct_segment(
            events, events, metadata, segment_id, fs
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
            span_start, span_end = minute_starts[0], minute_starts[-1] + eval_len

            wave_run = respiratory_filter(
                pred_series[run_start:run_end], fs, reference["band_hz"], profile
            )
            wave_breaths = detect_breaths_continuous(wave_run, fs) + run_start
            event_breaths = (
                detect_event_breaths(
                    event_series[run_start:run_end],
                    fs,
                    profile["min_breath_interval_sec"],
                    wave_breaths - run_start,
                )
                + run_start
            )
            gt_span = ref_breaths[(ref_breaths >= span_start) & (ref_breaths < span_end)]
            ev_span = event_breaths[
                (event_breaths >= span_start) & (event_breaths < span_end)
            ]
            tolerance = PEAK_MATCH_TOLERANCE_SEC * fs
            if len(gt_span) > 1:
                tolerance = PEAK_MATCH_TOLERANCE_CYCLE_FRACTION * float(
                    np.median(np.diff(gt_span))
                )
            gt_idx, ev_idx = match_breaths(gt_span, ev_span, tolerance)
            gt_matched = np.zeros(len(gt_span), dtype=bool)
            ev_matched = np.zeros(len(ev_span), dtype=bool)
            gt_matched[gt_idx] = True
            ev_matched[ev_idx] = True
            pair_gt_time = gt_span[gt_idx]
            pair_error_ms = np.abs(gt_span[gt_idx] - ev_span[ev_idx]) * 1000.0 / fs

            for local_index, start in enumerate(minute_starts, start=1):
                end = start + eval_len
                # Legacy v16 definition (stitched, window-normalized signals).
                pred_rr, pred_count, pred_filtered = direct_rr_bpm(
                    pred_series[start:end], fs
                )
                true_rr, true_count, true_filtered = direct_rr_bpm(
                    true_series[start:end], fs
                )
                legacy_ae = (
                    abs(pred_rr - true_rr)
                    if np.isfinite(pred_rr) and np.isfinite(true_rr)
                    else np.nan
                )
                legacy_pcc = safe_pcc(true_filtered, pred_filtered)

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

                # Model estimates.
                event_count = int(((event_breaths >= start) & (event_breaths < end)).sum())
                wave_count = int(((wave_breaths >= start) & (wave_breaths < end)).sum())
                wave_fft = fft_rr_bpm(
                    wave_run[start - run_start : end - run_start],
                    fs,
                    rr_band,
                    subharmonic,
                )
                estimates = np.array(
                    [event_count * per_minute, wave_count * per_minute, wave_fft]
                )
                estimates_sd = (
                    float(np.std(estimates, ddof=1))
                    if np.all(np.isfinite(estimates))
                    else np.nan
                )
                sf_retained = bool(
                    np.isfinite(estimates_sd) and estimates_sd <= SMART_FUSION_SD_BPM
                )
                fused = float(np.mean(estimates)) if sf_retained else np.nan
                # Median of the three estimates: no minute is withheld, and a
                # single failing estimate cannot move the result.
                finite = estimates[np.isfinite(estimates)]
                median3 = float(np.median(finite)) if len(finite) else np.nan

                # Breath-level matching attributed by breath time.
                gt_in = (gt_span >= start) & (gt_span < end)
                ev_in = (ev_span >= start) & (ev_span < end)
                pairs_in = (pair_gt_time >= start) & (pair_gt_time < end)
                true_positive = int(pairs_in.sum())
                missed = int((gt_in & ~gt_matched).sum())
                extra = int((ev_in & ~ev_matched).sum())

                rows.append(
                    {
                        "Segment_ID": segment_id,
                        "Coverage_Run_Index": run_index,
                        "RR_Window_Index": local_index,
                        "RR_Window_Start_Sec": start / float(fs),
                        "RR_Window_End_Sec": end / float(fs),
                        "Target_Breath_Count": true_count,
                        "Prediction_Breath_Count": pred_count,
                        "Target_RR_BPM_Counting": true_rr,
                        "Prediction_RR_BPM_Counting": pred_rr,
                        "RR_AE_BPM_Counting": legacy_ae,
                        "PCC_60s_Counting": legacy_pcc,
                        "Discordant_Legacy": bool(
                            np.isfinite(legacy_pcc)
                            and np.isfinite(legacy_ae)
                            and legacy_pcc >= DISCORDANT_PCC
                            and legacy_ae >= DISCORDANT_AE_BPM
                        ),
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
                        "Pred_Event_Count": event_count,
                        "Pred_Wave_CtO_Count": wave_count,
                        "Pred_Wave_FFT_RR": wave_fft,
                        "AE_Event": abs(event_count - gt_count) * per_minute,
                        "AE_Wave_CtO": abs(wave_count - gt_count) * per_minute,
                        "AE_Wave_FFT": abs(wave_fft - gt_rate)
                        if np.isfinite(wave_fft)
                        else np.nan,
                        "Signed_Error_Event": (event_count - gt_count) * per_minute,
                        "Peak_TP": true_positive,
                        "Peak_Missed": missed,
                        "Peak_Extra": extra,
                        "Peak_Wrong": missed + extra,
                        "Peak_Timing_Error_Sum_ms": float(pair_error_ms[pairs_in].sum()),
                        "SF_Estimates_SD": estimates_sd,
                        "SF_Retained": sf_retained,
                        "SF_Fused_RR": fused,
                        "AE_SmartFusion": abs(fused - gt_rate) if sf_retained else np.nan,
                        "Median3_RR": median3,
                        "AE_Median3": abs(median3 - gt_rate)
                        if np.isfinite(median3)
                        else np.nan,
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
        return float(function(x, y).pvalue)
    except Exception:
        return np.nan


def safe_friedman(matrix: np.ndarray) -> float:
    try:
        if matrix.shape[0] < 2 or matrix.shape[1] < 3:
            return np.nan
        return float(stats.friedmanchisquare(*matrix.T).pvalue)
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


def build_global_friedman(per_subject: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for dataset in list(DATASET_LABELS) + ["CombinedAllDatasets"]:
        frame = per_subject.copy()
        if dataset != "CombinedAllDatasets":
            frame = frame[frame["Dataset"] == dataset]
        frame["_PairID"] = (
            frame["Dataset"].astype(str)
            + "::"
            + frame["Subject"].astype(str)
        )
        pivot_rr = frame.pivot(
            index="_PairID",
            columns="Window_Order",
            values="RR_MAE_BPM",
        ).reindex(columns=[spec.order for spec in WINDOW_SPECS])
        pivot_pcc = frame.pivot(
            index="_PairID",
            columns="Window_Order",
            values="PCC_60s_Mean",
        ).reindex(columns=[spec.order for spec in WINDOW_SPECS])
        rr_complete = pivot_rr.dropna()
        pcc_complete = pivot_pcc.dropna()
        rows.append(
            {
                "Dataset": dataset,
                "N_Complete_RR_Subjects": int(len(rr_complete)),
                "RR_Friedman_p": safe_friedman(rr_complete.to_numpy(float)),
                "N_Complete_PCC_Subjects": int(len(pcc_complete)),
                "PCC_Friedman_p": safe_friedman(pcc_complete.to_numpy(float)),
            }
        )
    result = pd.DataFrame(rows)
    result["RR_Friedman_FDR_p"] = bh_fdr(
        result["RR_Friedman_p"].to_numpy(float)
    )
    result["PCC_Friedman_FDR_p"] = bh_fdr(
        result["PCC_Friedman_p"].to_numpy(float)
    )
    return result


def build_pairwise_stats(per_subject: pd.DataFrame) -> pd.DataFrame:
    rows = []
    datasets = list(DATASET_LABELS) + ["CombinedAllDatasets"]
    for dataset in datasets:
        frame = per_subject.copy()
        if dataset != "CombinedAllDatasets":
            frame = frame[frame["Dataset"] == dataset]
        for left_spec in WINDOW_SPECS:
            for right_spec in WINDOW_SPECS:
                if left_spec.order >= right_spec.order:
                    continue
                left = frame[frame["Window_Order"] == left_spec.order]
                right = frame[frame["Window_Order"] == right_spec.order]
                left = left.copy()
                right = right.copy()
                left["_PairID"] = (
                    left["Dataset"].astype(str)
                    + "::"
                    + left["Subject"].astype(str)
                )
                right["_PairID"] = (
                    right["Dataset"].astype(str)
                    + "::"
                    + right["Subject"].astype(str)
                )
                merged = left.merge(
                    right,
                    on="_PairID",
                    suffixes=("_left", "_right"),
                )
                rr_left = pd.to_numeric(
                    merged["RR_MAE_BPM_left"], errors="coerce"
                )
                rr_right = pd.to_numeric(
                    merged["RR_MAE_BPM_right"], errors="coerce"
                )
                pcc_left = pd.to_numeric(
                    merged["PCC_60s_Mean_left"], errors="coerce"
                )
                pcc_right = pd.to_numeric(
                    merged["PCC_60s_Mean_right"], errors="coerce"
                )
                rr_mask = np.isfinite(rr_left) & np.isfinite(rr_right)
                pcc_mask = np.isfinite(pcc_left) & np.isfinite(pcc_right)
                rr_l = rr_left[rr_mask].to_numpy(float)
                rr_r = rr_right[rr_mask].to_numpy(float)
                pcc_l = pcc_left[pcc_mask].to_numpy(float)
                pcc_r = pcc_right[pcc_mask].to_numpy(float)
                rows.append(
                    {
                        "Dataset": dataset,
                        "Left_Window_Order": left_spec.order,
                        "Left_Window_Config": left_spec.name,
                        "Left_Window_Label": left_spec.label,
                        "Right_Window_Order": right_spec.order,
                        "Right_Window_Config": right_spec.name,
                        "Right_Window_Label": right_spec.label,
                        "N_Paired_RR_Subjects": len(rr_l),
                        "RR_Left_Mean": float(np.mean(rr_l)) if len(rr_l) else np.nan,
                        "RR_Right_Mean": float(np.mean(rr_r)) if len(rr_r) else np.nan,
                        "Right_Minus_Left_RR_Mean": (
                            float(np.mean(rr_r - rr_l)) if len(rr_l) else np.nan
                        ),
                        "RR_Paired_t_p": safe_pvalue(stats.ttest_rel, rr_r, rr_l),
                        "RR_Wilcoxon_p": safe_pvalue(stats.wilcoxon, rr_r, rr_l),
                        "N_Paired_PCC_Subjects": len(pcc_l),
                        "PCC_Left_Mean": float(np.mean(pcc_l)) if len(pcc_l) else np.nan,
                        "PCC_Right_Mean": float(np.mean(pcc_r)) if len(pcc_r) else np.nan,
                        "Right_Minus_Left_PCC_Mean": (
                            float(np.mean(pcc_r - pcc_l)) if len(pcc_l) else np.nan
                        ),
                        "PCC_Paired_t_p": safe_pvalue(stats.ttest_rel, pcc_r, pcc_l),
                        "PCC_Wilcoxon_p": safe_pvalue(stats.wilcoxon, pcc_r, pcc_l),
                    }
                )
    result = pd.DataFrame(rows)
    result["RR_Wilcoxon_FDR_p"] = np.nan
    result["PCC_Wilcoxon_FDR_p"] = np.nan
    for dataset in result["Dataset"].unique():
        mask = result["Dataset"] == dataset
        result.loc[mask, "RR_Wilcoxon_FDR_p"] = bh_fdr(
            result.loc[mask, "RR_Wilcoxon_p"].to_numpy(float)
        )
        result.loc[mask, "PCC_Wilcoxon_FDR_p"] = bh_fdr(
            result.loc[mask, "PCC_Wilcoxon_p"].to_numpy(float)
        )
    result["RR_Significant_FDR05"] = result["RR_Wilcoxon_FDR_p"] < 0.05
    result["PCC_Significant_FDR05"] = result["PCC_Wilcoxon_FDR_p"] < 0.05
    return result


def _dataset_subset(frame: pd.DataFrame, dataset: str) -> pd.DataFrame:
    if dataset == "CombinedAllDatasets" or frame.empty:
        return frame
    return frame[frame["Dataset"] == dataset]


def _sd(values: np.ndarray) -> float:
    if len(values) > 1:
        return float(np.std(values, ddof=1))
    return 0.0 if len(values) == 1 else np.nan


def build_metric_means(
    frame: pd.DataFrame,
    metrics: Sequence[str],
) -> pd.DataFrame:
    rows = []
    for dataset in [*DATASET_LABELS, "CombinedAllDatasets"]:
        dataset_frame = _dataset_subset(frame, dataset)
        for spec in WINDOW_SPECS:
            window_frame = dataset_frame[dataset_frame["Window_Config"] == spec.name]
            row = {
                "Dataset": dataset,
                "Window_Order": spec.order,
                "Window_Config": spec.name,
                "Window_Label": spec.label,
                "N_Subjects": int(window_frame["Subject"].nunique()),
            }
            for metric in metrics:
                values = (
                    pd.to_numeric(window_frame[metric], errors="coerce")
                    .dropna()
                    .to_numpy(float)
                    if metric in window_frame.columns
                    else np.empty(0)
                )
                row[f"{metric}_Mean"] = float(np.mean(values)) if len(values) else np.nan
                row[f"{metric}_SD"] = _sd(values)
                row[f"{metric}_N"] = int(len(values))
            rows.append(row)
    return pd.DataFrame(rows)


def build_minute_level_summary(windows: pd.DataFrame) -> pd.DataFrame:
    """Pooled minute-level results per dataset and window condition.

    Karlen-style reporting: errors are pooled over minutes (MAE and RMSE),
    with the number of minutes that had a valid reference and the share
    retained by smart fusion. Legacy errors use every minute; the other
    estimates use reference-valid minutes; smart fusion uses retained ones.
    """
    if windows.empty:
        return pd.DataFrame()
    rows = []
    for dataset in [*DATASET_LABELS, "CombinedAllDatasets"]:
        dataset_frame = _dataset_subset(windows, dataset)
        for spec in WINDOW_SPECS:
            frame = dataset_frame[dataset_frame["Window_Config"] == spec.name]
            valid = frame[frame["Ref_Valid"].astype(bool)]
            retained = valid[valid["SF_Retained"].astype(bool)]
            row = {
                "Dataset": dataset,
                "Window_Order": spec.order,
                "Window_Config": spec.name,
                "Window_Label": spec.label,
                "N_Subjects": int(frame["Subject"].nunique()),
                "N_Minutes": int(len(frame)),
                "N_Ref_Valid": int(len(valid)),
                "Ref_Valid_Ratio": float(len(valid) / len(frame)) if len(frame) else np.nan,
            }
            for reason in REFERENCE_EXCLUSION_REASONS:
                row[f"N_Excluded_{reason}"] = int(
                    (frame["Ref_Exclusion_Reason"] == reason).sum()
                )
            for name, column, subset in (
                ("Legacy", "RR_AE_BPM_Counting", frame),
                ("Event", "AE_Event", valid),
                ("Wave_CtO", "AE_Wave_CtO", valid),
                ("Wave_FFT", "AE_Wave_FFT", valid),
                ("SmartFusion", "AE_SmartFusion", retained),
                ("Median3", "AE_Median3", valid),
            ):
                errors = (
                    pd.to_numeric(subset[column], errors="coerce").dropna().to_numpy(float)
                )
                row[f"{name}_N"] = int(len(errors))
                row[f"{name}_MAE"] = float(np.mean(errors)) if len(errors) else np.nan
                row[f"{name}_RMSE"] = (
                    float(np.sqrt(np.mean(errors**2))) if len(errors) else np.nan
                )
            row["SF_N_Retained"] = int(len(retained))
            row["SF_Retained_Ratio"] = (
                float(len(retained) / len(valid)) if len(valid) else np.nan
            )
            for column in ("Peak_Missed", "Peak_Extra", "Peak_Wrong"):
                values = pd.to_numeric(valid[column], errors="coerce").dropna()
                row[f"{column}_per_min"] = float(values.mean()) if len(values) else np.nan
            rows.append(row)
    return pd.DataFrame(rows)


def load_baseline_results(paths: Sequence[Path]) -> pd.DataFrame:
    """Load baseline per-subject rows for the paired comparison.

    Defaults to the v16 early-fusion results; a per-subject CSV of an earlier
    run of this family of scripts also works. Every comparison metric present
    in the file is kept, so legacy files compare on the legacy metrics only.
    """
    key_columns = ["Dataset", "Subject", "Window_Config"]
    window_names = {spec.name for spec in WINDOW_SPECS}
    frames = []
    for path in paths:
        path = Path(path)
        if not path.exists():
            print(f"Baseline file not found (skipped): {path}")
            continue
        frame = pd.read_csv(path, dtype={"Subject": str}, encoding="utf-8-sig")
        if "Fusion_Position" in frame.columns:
            frame = frame[frame["Fusion_Position"] == "early_fusion"]
        missing = [column for column in key_columns if column not in frame.columns]
        metrics = [metric for metric in COMPARISON_METRICS if metric in frame.columns]
        if missing or not metrics:
            warnings.warn(f"Baseline file {path} lacks key or metric columns; skipped")
            continue
        frame = frame[frame["Window_Config"].isin(window_names)][
            key_columns + metrics
        ].copy()
        print(f"Baseline rows: {len(frame)} ({', '.join(metrics)}) from {path}")
        frames.append(frame)
    if not frames:
        return pd.DataFrame(columns=key_columns)
    return pd.concat(frames, ignore_index=True).drop_duplicates(
        subset=["Dataset", "Subject", "Window_Config"],
        keep="first",
    )


def _paired_frame(frame: pd.DataFrame, dataset: str, spec: WindowSpec) -> pd.DataFrame:
    subset = _dataset_subset(frame, dataset)
    subset = subset[subset["Window_Config"] == spec.name].copy()
    subset["_PairID"] = (
        subset["Dataset"].astype(str) + "::" + subset["Subject"].astype(str)
    )
    subset = subset.drop_duplicates(subset="_PairID", keep="first")
    return subset.set_index("_PairID")


def build_baseline_comparison(
    proposed: pd.DataFrame,
    baseline: pd.DataFrame,
) -> pd.DataFrame:
    """Paired subject-level comparison of this model against the baseline."""
    metrics = [
        metric
        for metric in COMPARISON_METRICS
        if metric in baseline.columns and metric in proposed.columns
    ]
    if proposed.empty or baseline.empty or not metrics:
        return pd.DataFrame()
    rows = []
    for dataset in [*DATASET_LABELS, "CombinedAllDatasets"]:
        for spec in WINDOW_SPECS:
            left = _paired_frame(baseline, dataset, spec)
            right = _paired_frame(proposed, dataset, spec)
            common = left.index.intersection(right.index)
            for metric in metrics:
                left_values = pd.to_numeric(
                    left.loc[common, metric], errors="coerce"
                ).to_numpy(float)
                right_values = pd.to_numeric(
                    right.loc[common, metric], errors="coerce"
                ).to_numpy(float)
                valid = np.isfinite(left_values) & np.isfinite(right_values)
                left_values = left_values[valid]
                right_values = right_values[valid]
                difference = right_values - left_values
                lower_is_better = metric in LOWER_IS_BETTER_METRICS
                improved = difference < 0 if lower_is_better else difference > 0
                worsened = difference > 0 if lower_is_better else difference < 0
                if len(left_values) >= 2:
                    p_t = safe_pvalue(stats.ttest_rel, right_values, left_values)
                    try:
                        p_w = float(
                            stats.wilcoxon(
                                right_values,
                                left_values,
                                zero_method="wilcox",
                                alternative="two-sided",
                            ).pvalue
                        )
                    except Exception:
                        p_w = np.nan
                else:
                    p_t = np.nan
                    p_w = np.nan
                rows.append(
                    {
                        "Dataset": dataset,
                        "Window_Order": spec.order,
                        "Window_Config": spec.name,
                        "Window_Label": spec.label,
                        "Metric": metric,
                        "Better_Direction": "lower" if lower_is_better else "higher",
                        "N_Paired": int(len(left_values)),
                        "Baseline_Mean": float(np.mean(left_values))
                        if len(left_values)
                        else np.nan,
                        "Baseline_SD": _sd(left_values),
                        "Proposed_Mean": float(np.mean(right_values))
                        if len(right_values)
                        else np.nan,
                        "Proposed_SD": _sd(right_values),
                        "Proposed_minus_Baseline_Mean": float(np.mean(difference))
                        if len(difference)
                        else np.nan,
                        "Proposed_minus_Baseline_SD": _sd(difference),
                        "N_Proposed_Better": int(improved.sum()),
                        "N_Proposed_Worse": int(worsened.sum()),
                        "Paired_t_p": p_t,
                        "Wilcoxon_p": p_w,
                    }
                )
    result = pd.DataFrame(rows)
    result["Paired_t_FDR_p"] = bh_fdr(result["Paired_t_p"].to_numpy(float))
    result["Wilcoxon_FDR_p"] = bh_fdr(result["Wilcoxon_p"].to_numpy(float))
    result["Significant_Wilcoxon_p05"] = result["Wilcoxon_p"] < 0.05
    result["Significant_Wilcoxon_FDR05"] = result["Wilcoxon_FDR_p"] < 0.05
    return result


def _style_boxes(box: dict, colors: Sequence[str]) -> None:
    for patch, color in zip(box["boxes"], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.55)
    for median in box["medians"]:
        median.set_color("black")
        median.set_linewidth(1.4)


METRIC_LABELS = {
    "RR_MAE_BPM": "Legacy direct-count RR MAE (BPM)",
    "PCC_60s_Mean": "PCC over same 60 s interval",
    "RR_MAE_Event": "Event-head RR MAE vs fixed GT (BPM)",
    "RR_MAE_Wave_CtO": "Waveform count-orig RR MAE vs fixed GT (BPM)",
    "RR_MAE_Wave_FFT": "Waveform FFT RR MAE vs fixed GT (BPM)",
    "RR_Bias_Event": "Event-head RR bias (BPM)",
    "Peak_Wrong_per_min": "Wrong breaths per minute (missed + extra)",
    "Peak_Missed_per_min": "Missed breaths per minute",
    "Peak_Extra_per_min": "Extra breaths per minute",
    "Peak_Sensitivity": "Breath sensitivity",
    "Peak_PPV": "Breath positive predictive value",
    "Peak_Timing_Error_ms": "Matched breath timing error (ms)",
    "RR_MAE_SmartFusion": "Smart-fusion RR MAE on retained minutes (BPM)",
    "SF_Retained_Ratio": "Smart-fusion retained share of valid minutes",
    "RR_MAE_Median3": "Median-of-three RR MAE, all valid minutes (BPM)",
    "Ref_Valid_Ratio": "Share of minutes with a valid reference",
}


def save_performance_boxplots(
    proposed: pd.DataFrame,
    baseline: pd.DataFrame,
    comparison: pd.DataFrame,
    output_root: Path,
    model_label: str,
    metrics: Sequence[str] = COMPARISON_METRICS,
) -> None:
    plot_root = output_root / "Boxplots"
    plot_root.mkdir(parents=True, exist_ok=True)
    for metric in metrics:
        if metric not in proposed.columns:
            continue
        groups = [("proposed", "proposed", proposed)]
        if metric in baseline.columns and baseline[metric].notna().any():
            groups.insert(0, (BASELINE_KEY, "baseline", baseline))
        fig, axes = plt.subplots(1, 4, figsize=(24, 6), constrained_layout=True)
        for axis, dataset in zip(axes, [*DATASET_LABELS, "CombinedAllDatasets"]):
            values = []
            labels = []
            colors = []
            for spec in WINDOW_SPECS:
                for key, short, frame in groups:
                    subset = _dataset_subset(frame, dataset)
                    subset = subset[subset["Window_Config"] == spec.name]
                    values.append(
                        pd.to_numeric(subset[metric], errors="coerce")
                        .dropna()
                        .to_numpy(float)
                    )
                    labels.append(f"{spec.window_sec:g}s\n{short}")
                    colors.append(MODEL_COLORS[key])
            box = axis.boxplot(
                values,
                tick_labels=labels,
                patch_artist=True,
                widths=0.60,
                showfliers=True,
            )
            _style_boxes(box, colors)
            axis.set_title(dataset)
            axis.set_ylabel(METRIC_LABELS.get(metric, metric))
            axis.grid(axis="y", alpha=0.25)
            axis.set_axisbelow(True)
            finite = [array for array in values if len(array)]
            if comparison.empty or len(groups) < 2 or not finite:
                continue
            ymin = min(float(np.min(array)) for array in finite)
            ymax = max(float(np.max(array)) for array in finite)
            span = max(1e-6, ymax - ymin)
            significant = comparison[
                (comparison["Dataset"] == dataset)
                & (comparison["Metric"] == metric)
                & comparison["Significant_Wilcoxon_FDR05"].fillna(False)
            ]
            for _, row in significant.iterrows():
                group = int(row["Window_Order"]) - 1
                left = 2 * group + 1
                right = left + 1
                y = ymax + 0.10 * span
                axis.plot(
                    [left, left, right, right],
                    [y, y + 0.025 * span, y + 0.025 * span, y],
                    color="black",
                    linewidth=1.0,
                )
                axis.text((left + right) / 2.0, y + 0.035 * span, "*", ha="center")
            if len(significant):
                axis.set_ylim(top=ymax + 0.22 * span)
        title = model_label
        if len(groups) > 1:
            title += " vs baseline (* Wilcoxon FDR < 0.05)"
        fig.suptitle(title)
        fig.savefig(
            plot_root / f"{metric}_boxplot.png",
            dpi=220,
            bbox_inches="tight",
        )
        plt.close(fig)


def save_gate_boxplots(proposed: pd.DataFrame, output_root: Path) -> None:
    metrics = [f"Gate_{label}_Mean" for label in BRANCH_LABELS]
    if proposed.empty or any(metric not in proposed.columns for metric in metrics):
        return
    plot_root = output_root / "Boxplots"
    plot_root.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 4, figsize=(24, 6), constrained_layout=True)
    for axis, dataset in zip(axes, [*DATASET_LABELS, "CombinedAllDatasets"]):
        subset = _dataset_subset(proposed, dataset)
        values = []
        labels = []
        colors = []
        for spec in WINDOW_SPECS:
            window_frame = subset[subset["Window_Config"] == spec.name]
            for label, metric, color in zip(BRANCH_LABELS, metrics, BRANCH_COLORS):
                values.append(
                    pd.to_numeric(window_frame[metric], errors="coerce")
                    .dropna()
                    .to_numpy(float)
                )
                labels.append(f"{spec.window_sec:g}s\n{label}")
                colors.append(color)
        box = axis.boxplot(
            values,
            tick_labels=labels,
            patch_artist=True,
            widths=0.60,
            showfliers=True,
        )
        _style_boxes(box, colors)
        axis.axhline(1.0, color="gray", linestyle="--", linewidth=1.0)
        axis.set_title(dataset)
        axis.set_ylabel("Mean scale-gate weight per subject (uniform = 1)")
        axis.grid(axis="y", alpha=0.25)
        axis.set_axisbelow(True)
    fig.suptitle("Scale-gate branch weights on held-out subjects")
    fig.savefig(
        plot_root / "scale_gate_branch_weights_boxplot.png",
        dpi=220,
        bbox_inches="tight",
    )
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
    condition_root: Path,
    dataset_label: str,
    subject_data: Dict[str, dict],
) -> None:
    rows = []
    quality_root = condition_root / "Data_Quality"
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
    model_options: ModelOptions,
    window_specs: Sequence[WindowSpec],
    dataset_keys: Sequence[str],
) -> dict:
    reference = ModelOptions(False, False).build()
    proposed = model_options.build()
    event_head = count_parameters(proposed.event_head)
    return {
        "script": str(Path(__file__).resolve()),
        "results_root": str(results_root),
        "device": str(device),
        "experiment": (
            "v16 early stem fusion + breath-event decoder, reference QC, "
            "fixed ground truth and smart fusion"
        ),
        "model_key": model_options.key,
        "model_label": model_options.label,
        "parameters": {
            "v16_early_fusion": count_parameters(reference) - event_head,
            "v16_with_stem_options": count_parameters(proposed) - event_head,
            "proposed": count_parameters(proposed),
        },
        "model_path": (
            "stem branches (k=32/64/128, 8 filters) -> "
            + ("scale gate -> " if model_options.use_scale_gate else "")
            + (
                "[stem, z-scored identity] -> "
                if model_options.use_identity_channel
                else ""
            )
            + "1x1 fusion + GroupNorm (no GELU) "
            "-> depthwise residual encoder -> dilated decoder trunk "
            "-> {waveform head (tanh), breath-event head (logits)}"
        ),
        "decoder_change": {
            "trunk": "v16 decoder convs (24->16 k7, 16->8 k5) with dilations (8, 16)",
            "receptive_field_samples": {"v16": 11, "proposed": 113},
            "added_parameters": event_head,
            "event_target": f"Gaussian bumps (sigma {EVENT_SIGMA_SEC} s) at reference breaths",
            "event_head_init": f"weights N(0, 0.01), bias logit({EVENT_PRIOR})",
        },
        "loss": {
            "total": "wave * v16 hybrid + event * BCE(heatmap) + count * soft-count |error| (bpm)",
            "weights": {
                "wave": LOSS_WEIGHT_WAVE,
                "event": LOSS_WEIGHT_EVENT,
                "count": LOSS_WEIGHT_COUNT,
            },
            "v16_hybrid_weights": LOSS_WEIGHTS,
        },
        "reference_quality": {
            "breath_detector": "count-orig (RRest ref_cto): peaks > 0.2 x Q3, trough < 0 between breaths",
            "agreement_rule": f"count-orig RR vs FFT RR < {REF_AGREEMENT_BPM} bpm",
            "min_clean_cycle_time_fraction": REF_MIN_VALID_CYCLE_FRACTION,
            "rr_band_bpm": list(RR_BAND_BPM),
            "training_map": f"{REF_QC_WINDOW_SEC:g} s windows every {REF_QC_HOP_SEC:g} s",
            "training_use": "train/val windows need >= 50% reference-quality samples",
            "evaluation_use": "each 60 s minute must pass the rule to be scored",
        },
        "ground_truth": (
            "expert breath labels where the dataset provides them (CapnoBase "
            "co2 start of expiration); otherwise breath onsets detected once on "
            "the continuous reference RSP (60 s blocks, 10 s context)"
        ),
        "analysis_profiles": {
            DATASET_CONFIGS[key]["label"]: {
                "profile": DATASET_CONFIGS[key]["profile"],
                **ANALYSIS_PROFILES[DATASET_CONFIGS[key]["profile"]],
                "expert_breath_labels": bool(
                    DATASET_CONFIGS[key].get("expert_breath_labels", False)
                ),
                "expert_artifact_labels": bool(
                    DATASET_CONFIGS[key].get("use_artifact_labels", False)
                ),
            }
            for key in dataset_keys
        },
        "capnobase_files": {
            "labels": (
                "co2_startexp_x (fallback co2_startinsp_x), one cell of "
                f"space-separated sample numbers, index base {CAPNOBASE_LABEL_INDEX_BASE} "
                "removed, unit from units_x"
            ),
            "sampling_rates": "param samplingrate_pleth / samplingrate_co2",
            "artifacts": (
                "pleth *_artif_x -> unusable PPG samples; co2 *_artif_x -> "
                "no training window and no scored minute overlapping them"
            ),
            "check": "--inspect-capnobase writes capnobase_loading_check.csv",
        },
        "capnography_adaptive_upper_edge": (
            f"clip({ADAPTIVE_UPPER_HR_FRACTION} x PPG spectral HR, "
            f"{ADAPTIVE_UPPER_LIMITS_HZ[0]}, {ADAPTIVE_UPPER_LIMITS_HZ[1]}) Hz"
        ),
        "smart_fusion": {
            "estimates": ["event-head count", "waveform count-orig count", "waveform FFT RR"],
            "fusion": "mean",
            "withhold_if_sd_bpm_above": SMART_FUSION_SD_BPM,
            "reported": "MAE on retained minutes and retained share of valid minutes",
        },
        "model_config": {
            **MODEL_CONFIG,
            "use_identity_channel": model_options.use_identity_channel,
            "use_scale_gate": model_options.use_scale_gate,
        },
        "window_conditions": [
            {
                "order": spec.order,
                "name": spec.name,
                "window_sec": spec.window_sec,
                "stride_sec": spec.stride_sec,
                "label": spec.label,
            }
            for spec in window_specs
        ],
        "datasets": [DATASET_CONFIGS[key]["label"] for key in dataset_keys],
        "cross_validation": {
            "method": "Leave-One-Subject-Out Cross-Validation",
            "fold_definition": "one held-out subject per fold",
            "subject_reuse": "each valid subject is held out exactly once",
        },
        "validation": "last 20 percent of each training subject's windows",
        "input_ppg_band_hz": [0.1, 0.5],
        "quality_selection": {
            "quality_map_before_model_windowing": True,
            "qc_unit_sec": PPG_QC_UNIT_SEC,
            "ppg_sqi": "continuous beat-level pulse-template quality map",
            "ppg_high_sqi_threshold": PPG_SQI_THRESHOLD,
            "ppg_usable_sqi_threshold": PPG_USABLE_SQI_THRESHOLD,
            "minimum_usable_proportion": MIN_USABLE_PROPORTION,
            "rsp_structural_validity": True,
            "target_similarity_gate": False,
        },
        "evaluation": {
            "window_sec": EVAL_WINDOW_SEC,
            "stride_sec": EVAL_STRIDE_SEC,
            "timeline_reconstruction": "overlap-add mean using original window starts",
            "legacy_rr_estimator": "direct respiratory peak count (v16)",
            "legacy_pcc": "waveform Pearson correlation on the same complete 60 s interval",
            "gt_metrics": "event / count-orig / FFT RR vs fixed ground truth on reference-valid minutes",
            "breath_matching_tolerance": (
                f"{PEAK_MATCH_TOLERANCE_CYCLE_FRACTION} x median reference breath interval"
            ),
        },
        "baseline_comparison": {
            "baseline": BASELINE_LABEL,
            "files": [str(path) for path in args.baseline_csv],
            "pairing": "same dataset, subject, and window condition",
            "tests": ["paired t-test", "Wilcoxon signed-rank test"],
            "multiple_testing": "Benjamini-Hochberg FDR over all baseline comparisons",
        },
        "training": {
            "epochs": args.epochs,
            "min_epochs": args.min_epochs,
            "patience": args.patience,
            "batch_size": args.batch_size,
            "learning_rate": LEARNING_RATE,
            "weight_decay": WEIGHT_DECAY,
            "event_head_lr_multiplier": EVENT_HEAD_LR_MULTIPLIER,
            "early_stopping": "validation loss (total) with best-checkpoint restore",
            "train_val_windows": "reference-quality windows only (Train_OK)",
        },
    }


def run_experiment(args: argparse.Namespace) -> None:
    results_root = Path(args.results_root).resolve()
    results_root.mkdir(parents=True, exist_ok=True)
    DATASET_CONFIGS["capnobase"]["use_artifact_labels"] = args.capnobase_artifacts
    if args.inspect_capnobase:
        inspect_capnobase(results_root)
        return
    if args.require_cuda and not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA is required by default. Use --no-require-cuda for a smoke test."
        )
    device = torch.device(
        f"cuda:{args.gpu}" if torch.cuda.is_available() and not args.force_cpu else "cpu"
    )
    model_options = ModelOptions(
        use_identity_channel=args.use_identity_channel,
        use_scale_gate=args.use_scale_gate,
    )
    ANALYSIS_PROFILES["capnography_wide_rr"]["capnography_target"] = args.capnography_target
    window_specs = [spec for spec in WINDOW_SPECS if spec.name in args.windows]
    dataset_keys = [key for key in DATASET_ORDER if key in args.datasets]
    manifest = build_manifest(
        args,
        results_root,
        device,
        model_options,
        window_specs,
        dataset_keys,
    )
    (results_root / "run_manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False, default=_json_default),
        encoding="utf-8",
    )
    print(
        f"{model_options.label}: {manifest['parameters']['proposed']} parameters "
        f"(v16 early fusion: {manifest['parameters']['v16_early_fusion']})"
    )

    if args.dry_run:
        for spec in window_specs:
            print(f"\n[Window condition {spec.order}/{len(WINDOW_SPECS)}] {spec.label}")
            for dataset_key in dataset_keys:
                data = load_dataset(dataset_key, spec)
                print(
                    f"{DATASET_CONFIGS[dataset_key]['label']}: "
                    f"{len(data)} valid subjects = {len(data)} LOSOCV folds"
                )
                del data
                release_memory()
        return

    baseline = load_baseline_results(args.baseline_csv)
    all_subject_rows = []
    window_paths = []
    failed_rows = []
    for spec in window_specs:
        condition_root = results_root / f"{spec.order:02d}_{spec.name}"
        condition_root.mkdir(parents=True, exist_ok=True)
        condition_subject_rows = []
        condition_window_frames = []
        print(f"\n[Window condition {spec.order}/{len(WINDOW_SPECS)}] {spec.label}")
        for dataset_key in dataset_keys:
            dataset_label = DATASET_CONFIGS[dataset_key]["label"]
            subject_data = load_dataset(dataset_key, spec)
            subjects = sorted(subject_data)
            if len(subjects) < 3:
                raise RuntimeError(f"{dataset_label}: fewer than three valid subjects")
            write_quality_report(condition_root, dataset_label, subject_data)
            print(f"{dataset_label}: {len(subjects)} LOSOCV folds; device={device}")
            fold_root = condition_root / dataset_label
            summary_root = fold_root / "Fold_Summaries"
            summary_root.mkdir(parents=True, exist_ok=True)
            for fold_index, test_subject in enumerate(subjects, start=1):
                tag = safe_tag(test_subject)
                summary_path = summary_root / f"{tag}_summary.json"
                fold_window_path = summary_root / f"{tag}_RR60s_windows.csv"
                summary = (
                    json.loads(summary_path.read_text(encoding="utf-8"))
                    if args.resume and summary_path.exists()
                    else None
                )
                if summary is not None and summary.get("Model") != model_options.key:
                    # A fold of another model variant (e.g. with the scale
                    # gate) in the same results root is retrained, not reused.
                    print(
                        f"{dataset_label} {test_subject}: stored fold is "
                        f"{summary.get('Model')}, retraining as {model_options.key}"
                    )
                    summary = None
                if summary is not None:
                    window_frame = (
                        pd.read_csv(
                            fold_window_path,
                            dtype={"Subject": str},
                            encoding="utf-8-sig",
                        )
                        if fold_window_path.exists()
                        else pd.DataFrame()
                    )
                    all_subject_rows.append(summary)
                    condition_subject_rows.append(summary)
                    if not window_frame.empty:
                        condition_window_frames.append(window_frame)
                    print(f"{dataset_label} {test_subject}: resumed from {summary_path}")
                    continue
                train_subjects = [
                    subject for subject in subjects if subject != test_subject
                ]
                fold_seed = stable_seed(
                    "v16-early-fusion-identity-scale-gate",
                    model_options.key,
                    spec.name,
                    dataset_key,
                    test_subject,
                )
                start_time = time.time()
                try:
                    (
                        outputs,
                        metadata,
                        audit,
                        prediction_path,
                        diagnostics,
                    ) = train_one(
                        dataset_key,
                        spec,
                        test_subject,
                        subject_data,
                        train_subjects,
                        fold_root,
                        device,
                        args.epochs,
                        args.min_epochs,
                        args.patience,
                        args.batch_size,
                        fold_seed,
                        model_options,
                        args.save_features,
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
                            "LOSO_Fold_Index": fold_index,
                            "Window_Order": spec.order,
                            "Window_Config": spec.name,
                            "Window_Label": spec.label,
                            "Input_Window_Sec": spec.window_sec,
                            "Input_Stride_Sec": spec.stride_sec,
                            "Model": model_options.key,
                            "Model_Label": model_options.label,
                            "Stem_Filters_Per_Branch": MODEL_CONFIG["stem_base_channels"],
                            "Stem_Fusion_Position": "early_fusion",
                            "Stem_Fusion_Activation": MODEL_CONFIG["stem_fusion_activation"],
                            "Identity_Channel": model_options.use_identity_channel,
                            "Scale_Gate": model_options.use_scale_gate,
                            "Residual_Encoder_Type": MODEL_CONFIG["encoder_channel_mixing"],
                            "Decoder": "dilated_cnn_wave_and_breath_event_heads",
                            "Best_Val_Loss": audit["Best_Val_Loss"],
                            "Best_Epoch": audit["Best_Epoch"],
                            "Convergence_Epoch_95pct": audit["Convergence_Epoch_95pct"],
                            "History_Epochs": audit["History_Epochs"],
                            "Generalization_Gap_At_Best": audit["Generalization_Gap_At_Best"],
                            "Audit_Status": audit["Audit_Status"],
                            "Post_Best_Overfit_Signature_Observed": audit[
                                "Post_Best_Overfit_Signature_Observed"
                            ],
                            "Elapsed_Sec": time.time() - start_time,
                            "Prediction_File": str(prediction_path),
                        }
                    )
                    summary.update(diagnostics)
                    summary.update(subject_data[test_subject]["reference_diagnostics"])
                    if not window_frame.empty:
                        window_frame.insert(0, "Model", model_options.key)
                        window_frame.insert(0, "Subject", str(test_subject))
                        window_frame.insert(0, "Dataset", dataset_label)
                        window_frame["Window_Order"] = spec.order
                        window_frame["Window_Config"] = spec.name
                        window_frame["Window_Label"] = spec.label
                        window_frame.to_csv(
                            fold_window_path,
                            index=False,
                            encoding="utf-8-sig",
                        )
                        condition_window_frames.append(window_frame)
                    # Written last: its presence marks the fold as complete.
                    summary_path.write_text(
                        json.dumps(summary, indent=2, default=_json_default),
                        encoding="utf-8",
                    )
                    all_subject_rows.append(summary)
                    condition_subject_rows.append(summary)
                    print(
                        f"{dataset_label} {test_subject}: "
                        f"legacy RR_MAE={summary['RR_MAE_BPM']:.3f}, "
                        f"PCC={summary['PCC_60s_Mean']:.3f} | "
                        f"valid min {summary['N60_RefValid']}/{summary['N60']}: "
                        f"Event MAE={summary['RR_MAE_Event']:.3f}, "
                        f"wrong/min={summary['Peak_Wrong_per_min']:.2f}, "
                        f"SF MAE={summary['RR_MAE_SmartFusion']:.3f} "
                        f"(kept {summary['SF_Retained_Ratio']:.0%}), "
                        f"Median3 MAE={summary['RR_MAE_Median3']:.3f} | "
                        f"BestEpoch={audit['Best_Epoch']}"
                    )
                except Exception as exc:
                    failed_rows.append(
                        {
                            "Dataset": dataset_label,
                            "Subject": test_subject,
                            "LOSO_Fold_Index": fold_index,
                            "Window_Order": spec.order,
                            "Window_Config": spec.name,
                            "Model": model_options.key,
                            "Error": repr(exc),
                        }
                    )
                    print(f"{dataset_label} {test_subject}: FAILED {exc}")
                finally:
                    release_memory(device)
            del subject_data
            release_memory(device)
        pd.DataFrame(condition_subject_rows).to_csv(
            condition_root / "per_subject.csv",
            index=False,
            encoding="utf-8-sig",
        )
        if condition_window_frames:
            condition_window_path = condition_root / "RR60s_windows.csv"
            pd.concat(condition_window_frames, ignore_index=True).to_csv(
                condition_window_path,
                index=False,
                encoding="utf-8-sig",
            )
            window_paths.append(condition_window_path)
        del condition_window_frames
        release_memory(device)

    per_subject = pd.DataFrame(all_subject_rows)
    if per_subject.empty:
        raise RuntimeError("No successful folds were produced.")
    per_subject["Subject"] = per_subject["Subject"].astype(str)
    window_frames = [
        pd.read_csv(path, dtype={"Subject": str}, encoding="utf-8-sig")
        for path in window_paths
        if path.exists()
    ]
    windows = (
        pd.concat(window_frames, ignore_index=True)
        if window_frames
        else pd.DataFrame()
    )
    del window_frames

    performance_means = build_metric_means(per_subject, COMPARISON_METRICS)
    minute_level = build_minute_level_summary(windows)
    gate_means = build_metric_means(per_subject, GATE_METRICS)
    feature_means = build_metric_means(per_subject, FEATURE_METRICS)
    convergence_means = build_metric_means(per_subject, CONVERGENCE_METRICS)
    baseline_comparison = build_baseline_comparison(per_subject, baseline)
    baseline_means = (
        build_metric_means(
            baseline,
            [metric for metric in COMPARISON_METRICS if metric in baseline.columns],
        )
        if not baseline.empty
        else pd.DataFrame()
    )
    window_friedman = build_global_friedman(per_subject)
    window_pairwise = build_pairwise_stats(per_subject)

    prefix = f"v16_{model_options.key}_10s20s60s_nonoverlap"
    tables = {
        "per_subject": per_subject,
        "RR60s_windows": windows,
        "performance_means": performance_means,
        "minute_level_summary": minute_level,
        "baseline_performance_means": baseline_means,
        "baseline_comparison": baseline_comparison,
        "window_friedman": window_friedman,
        "window_pairwise": window_pairwise,
        "gate_means": gate_means,
        "feature_means": feature_means,
        "convergence_means": convergence_means,
    }
    for name, frame in tables.items():
        frame.to_csv(
            results_root / f"{prefix}_{name}.csv",
            index=False,
            encoding="utf-8-sig",
        )
    pd.DataFrame(failed_rows).to_csv(
        results_root / "failed_folds.csv",
        index=False,
        encoding="utf-8-sig",
    )
    save_performance_boxplots(
        per_subject,
        baseline,
        baseline_comparison,
        results_root,
        model_options.label,
    )
    if model_options.use_scale_gate:
        save_gate_boxplots(per_subject, results_root)
    excel_path = write_results_excel(
        results_root / f"{prefix}_results.xlsx",
        {
            "Per_subject": per_subject,
            "RR60s_windows": windows,
            "Performance_means": performance_means,
            "Minute_level_summary": minute_level,
            "Baseline_means": baseline_means,
            "Baseline_comparison": baseline_comparison,
            "Window_friedman": window_friedman,
            "Window_pairwise": window_pairwise,
            "Gate_means": gate_means,
            "Feature_means": feature_means,
            "Convergence_means": convergence_means,
        },
    )

    print("\nRR MAE per dataset (breaths/min; mean over held-out subjects):")
    print(
        f"  {'Dataset':<20s} {'Window':<18s} {'N':>3s} {'Median3':>8s} "
        f"{'Event':>7s} {'SmartF':>7s} {'kept':>5s} {'Legacy':>7s} {'PCC':>6s}"
    )
    for _, row in performance_means.iterrows():
        if row["N_Subjects"] == 0:
            continue
        print(
            f"  {row['Dataset']:<20s} {row['Window_Label']:<18s} {row['N_Subjects']:>3d} "
            f"{row['RR_MAE_Median3_Mean']:>8.3f} {row['RR_MAE_Event_Mean']:>7.3f} "
            f"{row['RR_MAE_SmartFusion_Mean']:>7.3f} {row['SF_Retained_Ratio_Mean']:>5.0%} "
            f"{row['RR_MAE_BPM_Mean']:>7.3f} {row['PCC_60s_Mean_Mean']:>6.3f}"
        )
    combined = performance_means[performance_means["Dataset"] == "CombinedAllDatasets"]
    print("\nCombined performance (mean over held-out subjects):")
    for _, row in combined.iterrows():
        print(
            f"  {row['Window_Label']} (N={row['N_Subjects']}): "
            f"legacy RR_MAE={row['RR_MAE_BPM_Mean']:.3f}, "
            f"PCC={row['PCC_60s_Mean_Mean']:.3f} | "
            f"Event MAE={row['RR_MAE_Event_Mean']:.3f}, "
            f"wrong/min={row['Peak_Wrong_per_min_Mean']:.2f}, "
            f"SF MAE={row['RR_MAE_SmartFusion_Mean']:.3f} "
            f"(kept {row['SF_Retained_Ratio_Mean']:.0%}), "
            f"Median3 MAE={row['RR_MAE_Median3_Mean']:.3f}, "
            f"valid ref {row['Ref_Valid_Ratio_Mean']:.0%}"
        )
    pooled = minute_level[minute_level["Dataset"] == "CombinedAllDatasets"]
    print("Pooled over minutes (CombinedAllDatasets):")
    for _, row in pooled.iterrows():
        print(
            f"  {row['Window_Label']}: minutes {row['N_Ref_Valid']}/{row['N_Minutes']} valid | "
            f"Event MAE={row['Event_MAE']:.3f}, RMSE={row['Event_RMSE']:.3f} | "
            f"SF MAE={row['SmartFusion_MAE']:.3f}, RMSE={row['SmartFusion_RMSE']:.3f}, "
            f"kept {row['SF_Retained_Ratio']:.0%} | "
            f"Median3 MAE={row['Median3_MAE']:.3f}, RMSE={row['Median3_RMSE']:.3f}"
        )
    if not baseline_comparison.empty:
        print("Paired vs baseline (CombinedAllDatasets):")
        combined_stats = baseline_comparison[
            baseline_comparison["Dataset"] == "CombinedAllDatasets"
        ]
        for _, row in combined_stats.iterrows():
            print(
                f"  {row['Window_Label']} {row['Metric']}: "
                f"baseline={row['Baseline_Mean']:.4f}, "
                f"proposed={row['Proposed_Mean']:.4f}, "
                f"diff={row['Proposed_minus_Baseline_Mean']:+.4f}, "
                f"better={row['N_Proposed_Better']}/{row['N_Paired']}, "
                f"Wilcoxon FDR p={row['Wilcoxon_FDR_p']:.4g}"
            )
    print(f"\nSaved results under: {results_root}")
    print(f"Per-subject: {results_root / f'{prefix}_per_subject.csv'}")
    print(f"Minute-level summary: {results_root / f'{prefix}_minute_level_summary.csv'}")
    print(f"Baseline comparison: {results_root / f'{prefix}_baseline_comparison.csv'}")
    print(f"Failed folds: {results_root / 'failed_folds.csv'}")
    print(f"Excel workbook: {excel_path}")


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--results-root",
        type=Path,
        default=DEFAULT_RESULTS_ROOT,
    )
    parser.add_argument("--epochs", type=int, default=DEFAULT_EPOCHS)
    parser.add_argument("--min-epochs", type=int, default=MIN_EPOCHS)
    parser.add_argument("--patience", type=int, default=PATIENCE)
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--force-cpu", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--no-require-cuda",
        dest="require_cuda",
        action="store_false",
    )
    parser.add_argument(
        "--no-identity-channel",
        dest="use_identity_channel",
        action="store_false",
        help="Disable proposal 4 (identity channel into the stem fusion).",
    )
    parser.add_argument(
        "--scale-gate",
        dest="use_scale_gate",
        action="store_true",
        help="Enable proposal 5 (window-adaptive scale gate); off by default.",
    )
    parser.add_argument(
        "--no-scale-gate",
        dest="use_scale_gate",
        action="store_false",
        help="Disable the scale gate (the default).",
    )
    parser.add_argument(
        "--datasets",
        nargs="+",
        choices=DATASET_ORDER,
        default=list(DATASET_ORDER),
    )
    parser.add_argument(
        "--windows",
        nargs="+",
        choices=[spec.name for spec in WINDOW_SPECS],
        default=[spec.name for spec in WINDOW_SPECS],
    )
    parser.add_argument(
        "--baseline-csv",
        nargs="*",
        type=Path,
        default=list(DEFAULT_BASELINE_CSVS),
        help="v16 early-fusion per-subject CSVs; pass no value to skip.",
    )
    parser.add_argument(
        "--capnography-target",
        choices=["phase", "co2"],
        default=ANALYSIS_PROFILES["capnography_wide_rr"]["capnography_target"],
        help="CapnoBase waveform target: label phase waveform or capnogram.",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Reuse folds whose Fold_Summaries JSON already exists.",
    )
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
        use_identity_channel=MODEL_CONFIG["use_identity_channel"],
        use_scale_gate=MODEL_CONFIG["use_scale_gate"],
    )
    return parser.parse_args(argv)


if __name__ == "__main__":
    run_experiment(parse_args())
