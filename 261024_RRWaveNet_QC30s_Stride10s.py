r"""261024_RRWaveNet_QC30s_Stride10s: number of subjects and input windows
before and after QC for CapnoBase, BIDMC and STEAM2, with the respiratory
reference QC on 30 s windows every 10 s.

Only the data are read and checked. The data reading and QC are those of
261007_RRWaveNet_Ensemble_NoEvent for CapnoBase and of
261013_RRWaveNet_Adult_SubjectVal_WD for BIDMC and STEAM2, unchanged except
the respiratory reference QC window and stride (REF_QC_WINDOW_SEC = 30 s,
REF_QC_HOP_SEC = 10 s; 32 s / 8 s in 261007 and 261013).

QC stages
---------
Before QC           every input window (10 s, 0% overlap; --windows) of every
                    recording, resampled to 64 Hz (STEAM2: of every piece of
                    a session between EEGLAB boundary events).
After signal QC     at least 50% of the window's samples with usable PPG and
                    at least 50% with structurally valid RSP. Structurally
                    valid: in 10 s blocks, at least 50% finite samples, not
                    flat, not saturated. Usable PPG: structurally valid and
                    inside a beat whose interval fits a heart rate of
                    40-180 bpm (CapnoBase 40-220), is at most 3 s and at most
                    2.2 times (or 1/2.2 of) the median interval, and whose
                    correlation with the pulse template is at least 0.5.
                    CapnoBase: samples in an expert-annotated artifact of the
                    PPG or the capnogram are not valid, and the capnogram is
                    first aligned to the PPG (transport delay).
After reference QC  passes the signal QC, and at least 50% of the window's
                    samples are valid in the respiratory reference QC and none
                    lies in an expert-annotated reference artifact
                    (CapnoBase). Respiratory reference QC: the reference is
                    checked on 30 s windows every 10 s; a 30 s window passes
                    when the count-orig breathing rate agrees with the FFT
                    rate (CapnoBase: with the rate of the expert breath
                    labels) within 2 breaths/min, at least 75% of the
                    breathing time lies in clean cycles and the rate is in
                    the dataset's band (6-30 breaths/min; CapnoBase 3-60). A
                    sample is valid when at least half of the 30 s windows
                    covering it pass.
Subjects: found in the data (STEAM2: with at least one session whose PPG
and RSP files pair up), read without error, with at least one window
before QC, and with at least one window after each QC stage. Subjects with
no window left after the signal QC are counted as well (with their windows
before QC); the 261007 / 261013 data reading leaves them out.

Outputs (CSV, Excel-compatible UTF-8 BOM; PROJECT_ROOT =
D:\PPG2RSP_RRWaveNet_Inspired\261024_RRWaveNet_QC30s_Stride10s)
  qc_counts_by_dataset.csv   subjects and windows before / after QC, per
                             dataset and input window
  qc_counts_by_subject.csv   windows before / after QC per subject
  qc_unread_subjects.csv     subjects found but not read (reading error)
  qc_window_detail.csv       with --save-window-detail: every window with
                             its QC values

Run:                     python 261024_RRWaveNet_QC30s_Stride10s.py
Some datasets:           --datasets capnobase
Other input windows:     --windows w10s_nonoverlap w20s_nonoverlap
Data paths:              --capnobase-path, --bidmc-path, --steam2-ppg-dir,
                         --steam2-rsp-dir
Without the CapnoBase expert artifact labels: --ignore-capnobase-artifacts
"""

import argparse
import csv
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import scipy.io as sio
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


# ---------------------------------------------------------------------------
# Paths, datasets and QC constants
# ---------------------------------------------------------------------------

PROJECT_ROOT = Path(r"D:\PPG2RSP_RRWaveNet_Inspired\261024_RRWaveNet_QC30s_Stride10s")
# The count tables are written to PROJECT_ROOT.
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
        # Expert artifact spans: PPG artifacts are unusable PPG samples,
        # CO2 artifacts invalidate the reference (windows that overlap them
        # do not pass the reference QC).
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
        # EEGLAB sessions (EEG.data, EEG.srate = 200 Hz, one channel labelled
        # PPG or RSP); default_fs is used only when a file has no srate.
        "default_fs": 200,
        "profile": "adult_belt_as_recorded",
    },
}

DATASET_ORDER = ("capnobase", "bidmc", "steam2")
# Data reading and QC of each dataset: CapnoBase that of
# 261007_RRWaveNet_Ensemble_NoEvent, BIDMC and STEAM2 that of
# 261013_RRWaveNet_Adult_SubjectVal_WD.
DATASET_PROTOCOL = {"capnobase": "261007", "bidmc": "261013", "steam2": "261013"}


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

# PPG of each window: the resampled PPG time series (no filter).
PPG_INPUT = "raw"
TARGET_FS = 64  # analysis sampling rate (Hz)

# Breath detection and reference quality (RRest / Charlton et al. 2016).
RESP_BAND_HZ = (0.1, 0.5)
RR_BAND_BPM = (6.0, 30.0)  # RR representable in the 0.1-0.5 Hz analysis band
CTO_PEAK_FRACTION = 0.2  # count-orig: peaks above 0.2 x Q3 of all peak values
CTO_PEAK_QUANTILE = 0.75
FFT_MIN_POINTS = 16384
# Respiratory reference QC: 30 s windows every 10 s (32 s / 8 s in 261007
# and 261013).
REF_QC_WINDOW_SEC = 30.0  # respiratory reference QC window (s)
REF_QC_HOP_SEC = 10.0
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
# Breath heatmap counted above this background level.
EVENT_COUNT_THRESHOLD = 0.1

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
        "target_polarity": "zero_lag_sign",
        "fft_subharmonic_check": False,
        "resampler": "fft",
    },
    # STEAM2 belt: as adult_fixed_band, but the reference keeps the polarity it
    # was recorded with (one belt type for every subject). The v16 zero-lag
    # sign rule picks the sign from the correlation of the PPG and the belt in
    # the respiratory band at zero lag, which is near zero for the belt (the
    # PPG respiratory modulation is a variable fraction of a cycle off the
    # chest movement), so it flips the target at random between sessions.
    "adult_belt_as_recorded": {
        "description": "v16 processing: fixed 0.1-0.5 Hz, reference polarity as recorded",
        "band_hz": RESP_BAND_HZ,
        "adaptive_upper": False,
        "rr_band_bpm": RR_BAND_BPM,
        "min_breath_interval_sec": 1.5,
        "ppg_hr_range_bpm": (PPG_HR_MIN_BPM, PPG_HR_MAX_BPM),
        "ppg_spectral_beat_distance": False,
        "max_target_lag_sec": 0.0,  # no lag search
        "fixed_target_polarity": True,
        "target_polarity": "as_recorded",
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


# ---------------------------------------------------------------------------
# Small utilities and signal preprocessing
# ---------------------------------------------------------------------------


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
    waveform keeps only the breath timing.
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
# Breath detection and reference quality
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

    The quality rule is evaluated on 30 s windows every 10 s; a sample is
    usable when at least half of the windows covering it pass. Only the
    reference (and its expert labels, if any) is used, never the PPG.
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

    Peaks may be fractional sample positions. The
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
    elif profile.get("target_polarity") == "as_recorded":
        # Polarity as recorded, no lag: the zero-lag correlation is kept only
        # as a diagnostic.
        sign = 1.0
        target_signal = rsp_low
        lag = 0
        alignment_corr = safe_pcc(ppg_low, rsp_low)
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
    # which windows pass the reference QC.
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
    reference_qc_map = resp_quality_map & reference_structural
    starts = extract_window_starts(
        length,
        fs,
        spec.window_sec,
        spec.stride_sec,
    )

    ppg_input = (ppg_raw if PPG_INPUT == "raw" else ppg_low).astype(np.float32)
    low_kept = []
    rsp_kept = []
    event_kept = []
    event_norm_kept = []
    reference_qc_kept = []
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
        resp_usable_ratio = float(reference_qc_map[int(start):end].mean())
        ref_artifact_ratio = float(rsp_artifact[int(start):end].mean())
        # Labels may be missing inside a labelled reference artifact, so such
        # windows do not pass the reference QC.
        reference_qc_ok = bool(
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
                "Reference_QC_OK": int(reference_qc_ok),
            }
        )
        quality_rows.append(diagnostics)
        if passed:
            # Window kept by the signal QC: its PPG (PPG_INPUT) and its
            # reference waveform.
            low_kept.append(ppg_input[int(start):end])
            rsp_kept.append(target_signal[int(start):end])
            event_window, event_norm = breath_event_target_and_norm(
                reference_breaths, int(start), window_len, fs
            )
            event_kept.append(event_window)
            event_norm_kept.append(event_norm)
            reference_qc_kept.append(reference_qc_ok)
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
        # Continuous arrays: PPG, reference waveform and the samples that
        # pass the reference QC (reference quality, no labelled reference
        # artifact, usable PPG).
        "ppg_input": ppg_input,
        "target_signal": np.asarray(target_signal, dtype=np.float32),
        "reference_qc_map": reference_qc_map & ~rsp_artifact & (ppg_quality_map > 0),
    }
    if not low_kept:
        return {
            "low": np.empty((0, window_len), dtype=np.float32),
            "rsp": np.empty((0, window_len), dtype=np.float32),
            "event": np.empty((0, window_len), dtype=np.float32),
            "event_norm": np.empty(0, dtype=np.float32),
            "reference_qc_ok": np.empty(0, dtype=bool),
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
        "reference_qc_ok": np.asarray(reference_qc_kept, dtype=bool),
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
    reference_qc_ok = np.concatenate([segment["reference_qc_ok"] for segment in valid], axis=0)
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
        "reference_qc_ok": reference_qc_ok.astype(bool),
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


# STEAM2 sessions are EEGLAB structures saved as .mat files (EEG.data,
# EEG.srate, EEG.chanlocs.labels, EEG.event), one channel per file: the
# PPG file and the RSP (belt) file of a session have the same name in the
# PPG and RSP folders and come from the same recording (same length).
STEAM2_CHANNELS = {"PPG": ("PPG", "PLETH", "BVP"), "RSP": ("RSP", "RESP", "RESPIRATION", "BELT")}
STEAM2_SESSION_PATTERN = re.compile(r"^(?P<subject>.+?)[_\- ]*session[_\- ]*(?P<session>\d+)$", re.IGNORECASE)


def steam2_session_key(stem: str) -> Tuple[str, int]:
    """(subject, session number) from a file stem such as 13_LSB_Session2."""
    match = STEAM2_SESSION_PATTERN.match(stem)
    if match is None:
        return stem, 0
    return match.group("subject"), int(match.group("session"))


def _as_list(value) -> list:
    if value is None:
        return []
    if isinstance(value, np.ndarray):
        return list(value.ravel()) if value.size else []
    return [value]


def read_eeglab_channel(path: Path, kind: str, default_fs: float) -> dict:
    """One channel of an EEGLAB .mat file.

    The channel is chosen by its label (STEAM2_CHANNELS[kind]); a file with
    one channel is accepted when it has no label, and rejected when its label
    names the other signal (swapped files). EEGLAB "boundary" events inside
    the recording mark discontinuities (removed data); their positions are
    returned so that the session is split there. Files without an EEG
    structure fall back to the generic reader (data and srate fields).
    """
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
            raise RuntimeError(f"{path.name}: no {kind} channel among {labels}")
        if labels and labels[0].upper() in other:
            raise RuntimeError(f"{path.name}: channel '{labels[0]}' found where {kind} was expected")
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
        # EEGLAB latencies are 1-based and a boundary at k.5 lies between
        # samples k and k + 1: the next piece starts at 0-based index k. The
        # markers at the start (0.5, "New Segment" at 1) and at the end are
        # not discontinuities inside the data.
        position = int(np.ceil(latency - 1.0))
        if 0 < position < signal.size:
            boundaries.append(position)
    label = labels[index] if index < len(labels) else ""
    return {"signal": signal, "fs": fs, "boundaries": sorted(set(boundaries)), "label": label,
            "channels": int(data.shape[0])}


def steam2_session_files(ppg_root: Path, rsp_root: Path) -> Dict[str, List[Tuple[str, Path, Path]]]:
    """Subject -> [(session name, PPG file, RSP file)], sessions in order.

    Subject folders (PPG\\<subject>\\<session>.mat and RSP\\<subject>\\...) are
    used when present; otherwise the files of one flat folder are grouped by
    the subject part of their name (<subject>_Session<k>.mat). Every session
    of a person stays in one subject.
    """
    sessions: Dict[str, List[Tuple[str, Path, Path]]] = {}
    subject_dirs = sorted(path for path in ppg_root.iterdir() if path.is_dir())
    if subject_dirs:
        for subject_dir in subject_dirs:
            rsp_dir = rsp_root / subject_dir.name
            if not rsp_dir.is_dir():
                print(f"[STEAM2 {subject_dir.name}] no RSP folder, skipped")
                continue
            # PPG and RSP files of a session have the same name; files without
            # a partner are reported and left out (never paired by position).
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


def steam2_session_pieces(ppg: dict, rsp: dict) -> Tuple[List[Tuple[np.ndarray, np.ndarray, float, float]], dict]:
    """The (PPG, RSP, fs, fs) pieces of one session, split at the boundary
    events of either file, and a description of the session."""
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
    info = {
        "PPG_Label": ppg["label"], "RSP_Label": rsp["label"],
        "PPG_FS": ppg["fs"], "RSP_FS": rsp["fs"],
        "PPG_Sec": ppg_sec, "RSP_Sec": rsp_sec, "Length_Difference_Sec": ppg_sec - rsp_sec,
        "Internal_Boundaries": len(edges) - 2, "Pieces": len(pieces),
        "Piece_Sec": " ".join(f"{end - start:.1f}" for start, end in zip(edges[:-1], edges[1:])),
    }
    return pieces, info


def load_steam2_subjects(spec: WindowSpec) -> Dict[str, dict]:
    cfg = DATASET_CONFIGS["steam2"]
    profile = ANALYSIS_PROFILES[cfg["profile"]]
    data = {}
    sessions = steam2_session_files(Path(cfg["ppg_dir"]), Path(cfg["rsp_dir"]))
    for subject, items in sessions.items():
        try:
            pairs = []
            for name, ppg_path, rsp_path in items:
                ppg = read_eeglab_channel(ppg_path, "PPG", cfg["default_fs"])
                rsp = read_eeglab_channel(rsp_path, "RSP", cfg["default_fs"])
                pieces, info = steam2_session_pieces(ppg, rsp)
                if abs(info["Length_Difference_Sec"]) > 1.0:
                    print(f"[STEAM2 {subject} {name}] PPG {info['PPG_Sec']:.1f} s vs RSP "
                          f"{info['RSP_Sec']:.1f} s; the common part is used")
                pairs += pieces
            result = preprocess_subject_from_pairs(pairs, spec, profile)
            if result is not None:
                data[subject] = result
        except Exception as exc:
            print(f"[STEAM2 {subject}] skipped: {exc}")
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
# Subjects and windows before and after QC
# ---------------------------------------------------------------------------

STAGES = ("Before_QC", "After_Signal_QC", "After_Reference_QC")


def source_subjects(dataset_key: str) -> List[str]:
    """Subjects found in the data of a dataset (before reading and QC)."""
    cfg = DATASET_CONFIGS[dataset_key]
    if dataset_key == "capnobase":
        return capnobase_case_ids(Path(cfg["data_path"]))
    if dataset_key == "bidmc":
        records = sio.loadmat(cfg["data_path"])["data"][0]
        return [f"{index + 1:02d}" for index in range(len(records))]
    if dataset_key == "steam2":
        return sorted(steam2_session_files(Path(cfg["ppg_dir"]), Path(cfg["rsp_dir"])))
    raise KeyError(dataset_key)


def load_all_subjects(dataset_key: str, spec: WindowSpec) -> Dict[str, dict]:
    """load_dataset, with the subjects that it leaves out because no window
    passes the signal QC: those are returned with the QC rows of their
    windows only."""
    global combine_segments
    combine = combine_segments

    def keep_every_subject(segments, spec, fs):
        combined = combine(segments, spec, fs)
        if combined is not None:
            return combined
        return {
            "quality": [row for segment in segments for row in segment["quality"]],
            "reference_diagnostics": {},
        }

    combine_segments = keep_every_subject
    try:
        return load_dataset(dataset_key, spec)
    finally:
        combine_segments = combine


def subject_counts(dataset_key: str, spec: WindowSpec, subject: str, record: dict) -> dict:
    quality = pd.DataFrame(record["quality"])
    before = len(quality)
    signal = int(quality["Kept"].sum()) if before else 0
    reference = int(quality["Reference_QC_OK"].sum()) if before else 0
    # Segment-level values, repeated on every window of a segment.
    qc_windows = qc_passed = 0
    if before:
        segments = quality.drop_duplicates(subset=["Segment_ID"])
        qc_windows = int(segments["RSP_Resp_QC_Windows"].fillna(0).sum())
        qc_passed = int(segments["RSP_Resp_QC_Pass_Windows"].fillna(0).sum())
    return {
        "Dataset": DATASET_CONFIGS[dataset_key]["label"],
        "QC_Code": DATASET_PROTOCOL[dataset_key],
        "Input_Window_Sec": spec.window_sec,
        "Input_Stride_Sec": spec.stride_sec,
        "Subject": str(subject),
        "Windows_Before_QC": before,
        "Windows_After_Signal_QC": signal,
        "Windows_After_Reference_QC": reference,
        "Removed_By_Signal_QC": before - signal,
        "Removed_By_Reference_QC": signal - reference,
        "Removed_Total": before - reference,
        "Pct_Left_After_Signal_QC": 100.0 * signal / before if before else np.nan,
        "Pct_Left_After_Reference_QC": 100.0 * reference / before if before else np.nan,
        "Reference_QC_30s_Windows": qc_windows,
        "Reference_QC_30s_Windows_Passed": qc_passed,
    }


def dataset_counts(rows: pd.DataFrame, found: Dict[str, int]) -> pd.DataFrame:
    """Subjects and windows of every dataset and input window."""
    table = []
    keys = ["Dataset", "QC_Code", "Input_Window_Sec", "Input_Stride_Sec"]
    for key, frame in rows.groupby(keys, sort=False):
        before = int(frame["Windows_Before_QC"].sum())
        signal = int(frame["Windows_After_Signal_QC"].sum())
        reference = int(frame["Windows_After_Reference_QC"].sum())
        qc_windows = int(frame["Reference_QC_30s_Windows"].sum())
        qc_passed = int(frame["Reference_QC_30s_Windows_Passed"].sum())
        table.append({
            **dict(zip(keys, key)),
            "Subjects_Found": found.get(key[0], np.nan),
            "Subjects_Read": int(len(frame)),
            "Subjects_Before_QC": int((frame["Windows_Before_QC"] > 0).sum()),
            "Subjects_After_Signal_QC": int((frame["Windows_After_Signal_QC"] > 0).sum()),
            "Subjects_After_Reference_QC": int((frame["Windows_After_Reference_QC"] > 0).sum()),
            "Windows_Before_QC": before,
            "Windows_After_Signal_QC": signal,
            "Windows_After_Reference_QC": reference,
            "Removed_By_Signal_QC": before - signal,
            "Removed_By_Reference_QC": signal - reference,
            "Removed_Total": before - reference,
            "Pct_Left_After_Signal_QC": 100.0 * signal / before if before else np.nan,
            "Pct_Left_After_Reference_QC": 100.0 * reference / before if before else np.nan,
            "Reference_QC_30s_Windows": qc_windows,
            "Reference_QC_30s_Windows_Passed": qc_passed,
            "Pct_Reference_QC_30s_Windows_Passed": 100.0 * qc_passed / qc_windows if qc_windows else np.nan,
        })
    return pd.DataFrame(table)


def save_csv(frame: pd.DataFrame, path: Path) -> None:
    frame.to_csv(path, index=False, encoding="utf-8-sig")
    print("Saved:", path)


def count_qc(args: argparse.Namespace) -> None:
    results_root = Path(args.results_root).resolve()
    results_root.mkdir(parents=True, exist_ok=True)
    dataset_keys = [key for key in DATASET_ORDER if key in args.datasets]
    specs = [spec for spec in WINDOW_SPECS if spec.name in args.windows]
    print(f"Respiratory reference QC: {REF_QC_WINDOW_SEC:g} s windows every {REF_QC_HOP_SEC:g} s")

    rows, unread, details, found = [], [], [], {}
    for dataset_key in dataset_keys:
        label = DATASET_CONFIGS[dataset_key]["label"]
        try:
            subjects = source_subjects(dataset_key)
        except Exception as exc:
            print(f"{label}: data not found ({exc})")
            continue
        found[label] = len(subjects)
        for spec in specs:
            print(f"\n{label} ({DATASET_PROTOCOL[dataset_key]} data reading and QC), {spec.label}: "
                  f"{len(subjects)} subjects found")
            records = load_all_subjects(dataset_key, spec)
            for subject in subjects:
                if subject not in records:
                    unread.append({"Dataset": label, "Input_Window_Sec": spec.window_sec, "Subject": subject})
            for subject in sorted(records):
                rows.append(subject_counts(dataset_key, spec, subject, records[subject]))
                if args.save_window_detail and records[subject]["quality"]:
                    detail = pd.DataFrame(records[subject]["quality"]).rename(
                        columns={"Kept": "After_Signal_QC", "Reference_QC_OK": "After_Reference_QC"}
                    )
                    detail.insert(0, "Subject", str(subject))
                    detail.insert(0, "Input_Window_Sec", spec.window_sec)
                    detail.insert(0, "Dataset", label)
                    details.append(detail)
            mine = [row for row in rows if row["Dataset"] == label and row["Input_Window_Sec"] == spec.window_sec]
            print(
                f"  windows: before QC {sum(r['Windows_Before_QC'] for r in mine)}, after signal QC "
                f"{sum(r['Windows_After_Signal_QC'] for r in mine)}, after reference QC "
                f"{sum(r['Windows_After_Reference_QC'] for r in mine)}"
            )
            del records

    if not rows:
        raise SystemExit("No data read; check the dataset paths.")
    per_subject = pd.DataFrame(rows)
    per_dataset = dataset_counts(per_subject, found)
    save_csv(per_dataset, results_root / "qc_counts_by_dataset.csv")
    save_csv(per_subject, results_root / "qc_counts_by_subject.csv")
    save_csv(pd.DataFrame(unread, columns=["Dataset", "Input_Window_Sec", "Subject"]),
             results_root / "qc_unread_subjects.csv")
    if details:
        save_csv(pd.concat(details, ignore_index=True), results_root / "qc_window_detail.csv")

    print("\nSubjects and windows before / after QC:")
    show = ["Dataset", "Input_Window_Sec", "Subjects_Found", "Subjects_Before_QC", "Subjects_After_Signal_QC",
            "Subjects_After_Reference_QC", "Windows_Before_QC", "Windows_After_Signal_QC",
            "Windows_After_Reference_QC", "Pct_Left_After_Reference_QC"]
    print(per_dataset[show].to_string(index=False, float_format=lambda value: f"{value:.1f}"))


# ---------------------------------------------------------------------------
# Command line
# ---------------------------------------------------------------------------

def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--results-root", type=Path, default=DEFAULT_RESULTS_ROOT)
    parser.add_argument("--datasets", nargs="+", choices=DATASET_ORDER, default=list(DATASET_ORDER))
    parser.add_argument(
        "--windows",
        nargs="+",
        choices=[spec.name for spec in WINDOW_SPECS],
        default=["w10s_nonoverlap"],
        help="Input windows counted (default %(default)s).",
    )
    parser.add_argument("--capnobase-path", type=Path, default=None)
    parser.add_argument("--bidmc-path", type=Path, default=None)
    parser.add_argument("--steam2-ppg-dir", type=Path, default=None)
    parser.add_argument("--steam2-rsp-dir", type=Path, default=None)
    parser.add_argument(
        "--ignore-capnobase-artifacts",
        dest="capnobase_artifacts",
        action="store_false",
        help="Do not use the expert artifact labels of CapnoBase.",
    )
    parser.add_argument(
        "--save-window-detail",
        action="store_true",
        help="Also save every window with its QC values (qc_window_detail.csv).",
    )
    parser.set_defaults(capnobase_artifacts=DATASET_CONFIGS["capnobase"]["use_artifact_labels"])
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> None:
    args = parse_args(argv)
    DATASET_CONFIGS["capnobase"]["use_artifact_labels"] = args.capnobase_artifacts
    if args.capnobase_path:
        DATASET_CONFIGS["capnobase"]["data_path"] = str(args.capnobase_path)
    if args.bidmc_path:
        DATASET_CONFIGS["bidmc"]["data_path"] = str(args.bidmc_path)
    if args.steam2_ppg_dir:
        DATASET_CONFIGS["steam2"]["ppg_dir"] = str(args.steam2_ppg_dir)
    if args.steam2_rsp_dir:
        DATASET_CONFIGS["steam2"]["rsp_dir"] = str(args.steam2_rsp_dir)
    count_qc(args)


if __name__ == "__main__":
    main()
