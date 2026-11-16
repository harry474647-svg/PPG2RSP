# -*- coding: utf-8 -*-
r"""261116_PPG_RIIV_RIAV_RIFV_FFT: the three respiratory modulations of the PPG, RIIV, RIAV and RIFV, as
time series and their FFT spectra, for every recording of the datasets of the final results.

  RIIV  respiratory-induced intensity variation: the PPG value at every pulse peak (baseline / intensity)
  RIAV  respiratory-induced amplitude variation: pulse amplitude, peak minus the trough before it
  RIFV  respiratory-induced frequency variation: instantaneous pulse rate, 60 / (peak-to-peak interval)
  (definitions of Karlen et al. 2013, IEEE TBME 60(7):1946-53, as used in RRest / Charlton et al. 2016)

Data: the PPG as the models read it (the continuous model input, 64 Hz), from the preprocessing of the
script of the final results, loaded as a module (its data paths are used; the path options override):
  CapnoBase      261007_RRWaveNet_Ensemble_NoEvent.py
  BIDMC, STEAM2  261013_RRWaveNet_Adult_SubjectVal_WD.py
The subjects with an MAE issue are left out as in the final results (CapnoBase 0115, BIDMC 27, STEAM2
12_YJJ, 22_LMS, 26_CKK; --keep-issue-subjects keeps them).

Extraction (per recording)
  1. Beats: peaks of the PPG pulse band (the source script's 0.5-5 Hz band-pass, zero phase), at least
     60 / (highest heart rate of the dataset profile) apart, prominence >= 0.1 x SD (as the PPG quality
     map of the source script).
  2. Values on the PPG low-passed at 8 Hz (zero phase, no high-pass, so the baseline stays for RIIV): the
     peak = the maximum within +-0.15 s of the detected beat, the trough = the minimum between the
     previous peak and this one.
  3. A beat is used when the interval to the previous beat lies in the heart-rate range of the profile
     and differs by less than 30% from the median of the 7 intervals around it, and (CapnoBase) the beat
     is outside the expert PPG artifact labels.
  4. Each series (one value per used beat, at the peak time) resampled to 4 Hz by cubic spline between
     the first and the last used beat.
  5. Each series band-passed with the recording's respiratory band of the source script (the band its
     target was filtered with: BIDMC / STEAM2 0.1-0.5 Hz; CapnoBase 0.05 Hz to the heart-rate dependent
     upper edge, at most 1.0 Hz), 3rd-order Butterworth, zero phase, so slow trends and pulse-rate drifts
     do not cover the respiratory peaks.
Windows: steps 1-5 run on the whole recording; the series are then cut into windows of every length in
--window-sec (default 60 and 30 s: both are drawn), on a grid from the start of the recording (60 s: 0-60,
60-120 s, ... = the minutes of the scoring; 30 s: 0-30, 30-60 s, ... = the halves of those minutes); a
window is used when the series cover at least 90% of it and it holds at least 10 used beats.
A length of 0 = the whole recording as one window (first to last used beat), e.g. --window-sec 60 30 0.
Spectrum of each window and series: detrend, Hamming window, zero padding (at least 4096 points); |FFT|
divided by its largest value in the rate band so the three curves share one axis (their units differ);
peak = the largest |FFT| in the rate band, 0.1-0.5 Hz (6-30 /min; --rate-band-hz).
GT rate of the window (for comparison): GT breaths in the window / its length (with 60 s windows: the GT
breath count of the minute, as in the scoring; with 30 s windows: twice the count of the half minute), with the GT of the final results (CapnoBase: breaths
detected on the capnogram; BIDMC: annotator 1 (ann1) of bidmc_data.mat; STEAM2: the stored breaths of
the belt).

Figures per window (per recording for the whole recording), in two styles, each in its own folder
(--styles, default both); GT (black) = the continuous reference signal of the source script (CapnoBase
capnogram, BIDMC impedance, STEAM2 belt; band-passed and aligned as in the scoring)
  overlap   (a) GT, RIIV, RIAV and RIFV over time (band-passed), each z-scored (mean 0, SD 1), on one axis
            (b) their |FFT| (peak = 1) over each other; the peak of each marked, its frequency in the legend;
                the GT rate (breath count / length) dotted, the rate band dashed
  separate  the same curves, one row per signal (GT, RIIV, RIAV, RIFV), each with its own axes:
            left the time series, right the |FFT| (peak = 1) with its peak, the GT rate dotted, the band dashed

Outputs (D:\PPG2RSP_RRWaveNet_Inspired\261116_PPG_RIIV_RIAV_RIFV_FFT), {style} = overlap or separate
  windows of W s (default W = 60 and 30)
    {style}\{Dataset}\{subject}\w{W}s\{subject}[_rec{k}]_{t0}-{t1}s_RIIV_RIAV_RIFV.png   one figure per window
    {style}\{Dataset}\{subject}\w{W}s\{Dataset}_{subject}_w{W}s_RIIV_RIAV_RIFV.pdf       every W s window
  whole recording (length 0)
    {style}\{Dataset}\{subject}[_rec{k}]_RIIV_RIAV_RIFV.png    one figure per recording
    {style}\{Dataset}\{Dataset}_RIIV_RIAV_RIFV.pdf              every recording of the dataset
  261116_PPG_RIIV_RIAV_RIFV_FFT.xlsx
    Windows     every window (Window_Sec 60 / 30; 0 = whole recording): start, end, beats, mean pulse rate,
                GT breaths, the GT rate (Hz),
                the peak frequency of the GT reference and of RIIV / RIAV / RIFV (Hz), |peak - GT rate| of each
    Summary     per dataset and window length: mean |peak - GT rate| of each modulation, recordings within 0.02 Hz of the GT rate,
                and how often each is the closest (ties counted for each)
    Notes

Run: python 261116_PPG_RIIV_RIAV_RIFV_FFT.py     (Spyder: runfile(...), no options needed)
Options: --window-sec (default 60 30; 0 = whole recording), --styles (default overlap separate), --datasets, --subjects CapnoBase:0332 BIDMC:05 ...,
--keep-issue-subjects, --rate-band-hz,
--max-freq-hz (x axis, default 1.0), --output-root, --script-261007, --script-261013, --root-261007,
--root-261013, --capnobase-path, --bidmc-path, --steam2-ppg-dir, --steam2-rsp-dir
"""

import argparse
import importlib.util
import json
import math
import re
import sys
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.io as sio
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.lines import Line2D
from scipy.interpolate import interp1d
from scipy.signal import butter, detrend, filtfilt, find_peaks, sosfiltfilt

# ---------------------------------------------------------------------------
# Settings
# ---------------------------------------------------------------------------

NAME = "261116_PPG_RIIV_RIAV_RIFV_FFT"
HERE = Path(__file__).resolve().parent
WORKSPACE = Path(r"D:\PPG2RSP_RRWaveNet_Inspired")
OUTPUT_ROOT = WORKSPACE / NAME
ROOT_261007 = WORKSPACE / "261007_RRWaveNet_NoEvent_v1"
ROOT_261013 = WORKSPACE / "261013_RRWavNet_STEAM_BIDMC_v1"
SCRIPTS = {"261007": "261007_RRWaveNet_Ensemble_NoEvent.py", "261013": "261013_RRWaveNet_Adult_SubjectVal_WD.py"}
WINDOW = "01_w10s_nonoverlap"
DATASETS = ("CapnoBase", "BIDMC", "STEAM2")
KEYS = {"CapnoBase": "capnobase", "BIDMC": "bidmc", "STEAM2": "steam2"}
SOURCE = {"CapnoBase": "261007", "BIDMC": "261013", "STEAM2": "261013"}
GT_NAME = {"CapnoBase": "breaths detected on the capnogram", "BIDMC": "ann1", "STEAM2": "belt breaths"}
MODALITY = {"CapnoBase": "capnogram", "BIDMC": "impedance", "STEAM2": "belt"}
EXCLUDED = {"STEAM2": ("22_LMS", "26_CKK", "12_YJJ"), "BIDMC": ("27",), "CapnoBase": ("0115",)}
MODS = ("RIIV", "RIAV", "RIFV")
LONG = {"RIIV": "RIIV (intensity)", "RIAV": "RIAV (amplitude)", "RIFV": "RIFV (pulse rate)"}
COLORS = {"RIIV": "#2a78d6", "RIAV": "#eb6834", "RIFV": "#1baf7a"}  # validated categorical slots 1-3
GT_COLOR, BAND_COLOR, INK, MUTED, GRID = "#000000", "#b84fb8", "#1f1f1f", "#6b6a66", "#e4e3df"
RESAMPLE_HZ = 4.0
PPG_LOWPASS_HZ = 8.0
PEAK_SEARCH_SEC = 0.15
INTERVAL_TOLERANCE = 0.30
MEDIAN_BEATS = 7
FFT_MIN_POINTS = 4096
RATE_BAND_HZ = (0.1, 0.5)
MAX_FREQ_HZ = 1.0
WINDOW_SECS = (60.0, 30.0)  # every length is drawn; 0 = the whole recording
STYLES = ("overlap", "separate")  # each style in its own folder
MIN_WINDOW_COVER = 0.9
MIN_WINDOW_BEATS = 10


# ---------------------------------------------------------------------------
# Source scripts
# ---------------------------------------------------------------------------

_MODULES: Dict[str, object] = {}


def find_script(name: str, given: Optional[Path], roots: Sequence[Path]) -> Path:
    """The script of the results: given, written in a run manifest, or found nearby."""
    candidates: List[Path] = [Path(given)] if given else []
    for root in roots:
        for manifest in sorted(Path(root).glob("run_manifest*.json")):
            try:
                script = json.loads(manifest.read_text(encoding="utf-8")).get("script")
            except (OSError, ValueError, AttributeError):
                continue
            if script and Path(script).name == name:
                candidates.append(Path(script))
    folders = [HERE, HERE.parent, WORKSPACE] + [Path(r) for r in roots]
    if WORKSPACE.is_dir():
        folders += sorted(p for p in WORKSPACE.iterdir() if p.is_dir())
    candidates += [f / name for f in folders]
    for path in candidates:
        if path.is_file():
            return path.resolve()
    raise FileNotFoundError(f"{name} not found (run manifests of {', '.join(map(str, roots))}, {HERE}, {WORKSPACE}); "
                            f"give it with --script-{name[:6]}")


def load_script(path: Path):
    key = str(path)
    if key not in _MODULES:
        name = "respmod_src_" + re.sub(r"\W", "_", path.stem)
        spec = importlib.util.spec_from_file_location(name, key)
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module
        spec.loader.exec_module(module)
        _MODULES[key] = module
    return _MODULES[key]


# ---------------------------------------------------------------------------
# GT breaths (as the final results)
# ---------------------------------------------------------------------------

def read_ann1(mod) -> Dict[str, np.ndarray]:
    mat = sio.loadmat(mod.DATASET_CONFIGS["bidmc"]["data_path"])
    return {f"{i + 1:02d}": np.asarray(r["ref"][0, 0]["breaths"][0, 0]["ann1"], float).ravel()
            for i, r in enumerate(mat["data"][0])}


def ann1_timeline(mod, ann1: np.ndarray, reference: dict) -> np.ndarray:
    """ann1 (MATLAB sample numbers at 125 Hz) as sample indices of the reference timeline."""
    fs_in = float(mod.DATASET_CONFIGS["bidmc"]["default_fs"])
    lag = int(round(float(reference["diagnostics"].get("Target_Lag_Sec", 0.0)) * mod.TARGET_FS))
    labels = np.round((np.asarray(ann1, float) - 1.0) * (mod.TARGET_FS / fs_in)).astype(np.int64) - lag
    return np.unique(labels[(labels >= 0) & (labels < len(reference["signal"]))])


def gt_breaths(dataset: str, mod, ref: dict, ann1: Optional[np.ndarray]) -> np.ndarray:
    if dataset == "CapnoBase":
        return np.asarray(ref["detected_breaths"], dtype=np.int64)
    if dataset == "BIDMC":
        return ann1_timeline(mod, ann1, ref) if ann1 is not None else np.asarray(ref["breaths"], dtype=np.int64)
    return np.asarray(ref["breaths"], dtype=np.int64)


# ---------------------------------------------------------------------------
# RIIV, RIAV, RIFV
# ---------------------------------------------------------------------------

def lowpass(signal: np.ndarray, fs: float, cutoff: float, order: int = 3) -> np.ndarray:
    b, a = butter(order, cutoff / (0.5 * fs), btype="low")
    return filtfilt(b, a, signal)


def extract_modulations(mod, ppg: np.ndarray, fs: float, hr_range_bpm: Tuple[float, float],
                        artifact: Optional[np.ndarray]) -> dict:
    """Beats of the PPG and the RIIV / RIAV / RIFV value of every used beat."""
    ppg, _ = mod.sanitize_signal(np.asarray(ppg, dtype=np.float64))
    pulse = mod.bandpass_filter(ppg, fs, mod.PPG_PULSE_BAND[0], mod.PPG_PULSE_BAND[1])[0]
    smooth = lowpass(ppg, fs, PPG_LOWPASS_HZ)
    spread = float(np.std(pulse))
    if spread < 1e-8:
        return {"n_detected": 0}
    distance = max(1, int(round(60.0 / hr_range_bpm[1] * fs)))
    detected, _ = find_peaks(pulse, distance=distance, prominence=max(1e-8, 0.10 * spread))
    half = int(round(PEAK_SEARCH_SEC * fs))
    peaks = np.array([max(0, p - half) + int(np.argmax(smooth[max(0, p - half):p + half + 1])) for p in detected], int)
    peaks = np.unique(peaks)
    if len(peaks) < 3:
        return {"n_detected": int(len(peaks))}
    intervals = np.diff(peaks) / fs
    troughs = np.array([a + int(np.argmin(smooth[a:b + 1])) for a, b in zip(peaks[:-1], peaks[1:])], int)
    beats, prev = peaks[1:], peaks[:-1]
    ok = (intervals >= 60.0 / hr_range_bpm[1]) & (intervals <= 60.0 / hr_range_bpm[0])
    median = pd.Series(intervals).rolling(MEDIAN_BEATS, center=True, min_periods=1).median().to_numpy()
    ok &= np.abs(intervals - median) <= INTERVAL_TOLERANCE * median
    if artifact is not None:
        artifact = np.asarray(artifact, bool)
        ok &= ~artifact[beats] & ~artifact[prev]
    values = {"RIIV": smooth[beats], "RIAV": smooth[beats] - smooth[troughs], "RIFV": 60.0 / intervals}
    return {"n_detected": int(len(peaks)), "times": beats[ok] / fs, "values": {k: v[ok] for k, v in values.items()},
            "peaks": beats[ok], "troughs": troughs[ok], "mean_hr": float(np.mean(60.0 / intervals[ok])) if ok.any() else math.nan}


def resample(times: np.ndarray, values: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """One value per beat -> a 4 Hz series between the first and the last beat (cubic spline)."""
    grid = np.arange(times[0], times[-1], 1.0 / RESAMPLE_HZ)
    kind = "cubic" if len(times) >= 4 else "linear"
    return grid, interp1d(times, values, kind=kind)(grid)


def respiratory_band(x: np.ndarray, fs: float, band: Tuple[float, float], order: int = 3) -> np.ndarray:
    """Zero-phase Butterworth band-pass of a modulation series (second-order sections)."""
    high = min(float(band[1]), 0.45 * fs)
    sos = butter(order, [float(band[0]) / (0.5 * fs), high / (0.5 * fs)], btype="band", output="sos")
    try:
        return sosfiltfilt(sos, x)
    except ValueError:  # series too short for the filter
        return x - np.mean(x)


def spectrum(x: np.ndarray, fs: float) -> Tuple[np.ndarray, np.ndarray]:
    """Frequency (Hz) and |FFT|: detrend, Hamming window, zero padding."""
    x = np.asarray(x, dtype=np.float64)
    if len(x) < 8 or float(np.std(x)) < 1e-12:
        return np.zeros(1), np.full(1, np.nan)
    windowed = detrend(x) * np.hamming(len(x))
    n_fft = max(FFT_MIN_POINTS, 1 << int(math.ceil(math.log2(len(x)))))
    return np.fft.rfftfreq(n_fft, d=1.0 / fs), np.abs(np.fft.rfft(windowed, n=n_fft))


def band_peak(freq: np.ndarray, mag: np.ndarray, band: Tuple[float, float]) -> Tuple[float, float]:
    keep = (freq >= band[0]) & (freq <= band[1]) & np.isfinite(mag)
    if not keep.any():
        return math.nan, math.nan
    k = int(np.flatnonzero(keep)[np.argmax(mag[keep])])
    return float(freq[k]), float(mag[k])


# ---------------------------------------------------------------------------
# Figure
# ---------------------------------------------------------------------------

def style(ax) -> None:
    ax.grid(True, color=GRID, lw=0.6)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)


def draw_overlap(path: Path, pdf, header: Tuple[str, str], series: Dict[str, Tuple[np.ndarray, np.ndarray]],
         spectra: Dict[str, dict], gt: dict, gt_rate_hz: float, band: Tuple[float, float], max_freq: float,
         span_label: str) -> None:
    width, side, left, top_in, bottom_in = 12.6, 3.3, 0.8, 1.0, 0.6
    h_ts, h_fft, gap = 2.3, 2.9, 0.75
    height = top_in + bottom_in + h_ts + h_fft + gap
    fig = plt.figure(figsize=(width, height))
    plot_w = (width - left - side - 0.25) / width
    x_leg = (width - side + 0.1) / width

    # (a) time series, z-scored
    y0 = bottom_in + h_fft + gap
    ax = fig.add_axes([left / width, y0 / height, plot_w, h_ts / height])
    t_gt, v_gt = gt["series"]
    sd_gt = float(np.std(v_gt))
    ax.plot(t_gt, (v_gt - np.mean(v_gt)) / sd_gt if sd_gt > 0 else v_gt * 0, color=GT_COLOR, lw=1.5, zorder=2)
    for key in MODS:
        t, v = series[key]
        sd = float(np.std(v))
        ax.plot(t, (v - np.mean(v)) / sd if sd > 0 else v * 0, color=COLORS[key], lw=1.0, zorder=3)
    ax.set_xlim(min(series[k][0][0] for k in MODS), max(series[k][0][-1] for k in MODS))
    ax.set_xlabel("Time in the recording (s)", fontsize=9)
    ax.set_ylabel("z-score", fontsize=9)
    ax.set_title("(a) GT, RIIV, RIAV, RIFV over time (band-passed, each z-scored)", fontsize=9.5, loc="left", pad=3)
    style(ax)
    fig.legend([Line2D([], [], color=GT_COLOR, lw=1.6)] + [Line2D([], [], color=COLORS[k], lw=1.6) for k in MODS],
               [gt["label"]] + [LONG[k] for k in MODS], loc="upper left",
               bbox_to_anchor=(x_leg, (y0 + h_ts) / height), frameon=False, fontsize=8.2, labelspacing=0.7,
               borderaxespad=0.0)

    # (b) spectra, |FFT| / peak in the band
    ax = fig.add_axes([left / width, bottom_in / height, plot_w, h_fft / height])
    top = 1.0
    if np.isfinite(gt["peak_mag"]) and gt["peak_mag"] > 0:
        keep = gt["freq"] <= max_freq
        y = gt["mag"][keep] / gt["peak_mag"]
        top = max(top, float(np.nanmax(y)))
        ax.plot(gt["freq"][keep], y, color=GT_COLOR, lw=1.7, zorder=3)
        ax.plot([gt["peak_hz"]], [1.0], marker="s", ms=5.5, color=GT_COLOR, mec="white", mew=0.8, zorder=8)
    for z, key in enumerate(MODS):
        s = spectra[key]
        keep = s["freq"] <= max_freq
        if np.isfinite(s["peak_mag"]) and s["peak_mag"] > 0:
            y = s["mag"][keep] / s["peak_mag"]
            top = max(top, float(np.nanmax(y)))
            ax.plot(s["freq"][keep], y, color=COLORS[key], lw=1.7, zorder=4 + z)
            ax.axvline(s["peak_hz"], color=COLORS[key], lw=1.0, alpha=0.9, zorder=3)
            ax.plot([s["peak_hz"]], [1.0], marker="s", ms=5.5, color=COLORS[key], mec="white", mew=0.8, zorder=8)
    if np.isfinite(gt_rate_hz):
        ax.axvline(gt_rate_hz, color=GT_COLOR, lw=1.1, ls=(0, (1, 2)), zorder=9)
    for edge in band:
        ax.axvline(edge, color=BAND_COLOR, lw=1.1, ls=(0, (4, 3)), zorder=1)
    ax.set_xlim(0.0, max_freq)
    ax.set_ylim(0.0, top * 1.08)
    ax.set_xlabel("Frequency (Hz)", fontsize=9)
    ax.set_ylabel("|FFT| (peak = 1)", fontsize=9)
    ax.set_title(f"(b) FFT of GT, RIIV, RIAV, RIFV ({span_label})", fontsize=9.5, loc="left", pad=3)
    style(ax)
    handles = [Line2D([], [], color=GT_COLOR, lw=1.7, marker="s", ms=5.5, mec="white", mew=0.8)]
    labels = [f"{gt['label']} peak {gt['peak_hz']:.3f} Hz\n({gt['peak_hz'] * 60:.1f} /min)"
              if np.isfinite(gt["peak_hz"]) else f"{gt['label']} peak n/a"]
    handles += [Line2D([], [], color=COLORS[k], lw=1.7, marker="s", ms=5.5, mec="white", mew=0.8) for k in MODS]
    labels += [f"{k} peak {spectra[k]['peak_hz']:.3f} Hz\n({spectra[k]['peak_hz'] * 60:.1f} /min)"
               if np.isfinite(spectra[k]["peak_hz"]) else f"{k} peak n/a" for k in MODS]
    handles += [Line2D([], [], color=GT_COLOR, lw=1.1, ls=(0, (1, 2))),
                Line2D([], [], color=BAND_COLOR, lw=1.1, ls=(0, (4, 3)))]
    labels += [f"GT rate (breath count)\n{gt_rate_hz:.3f} Hz ({gt_rate_hz * 60:.1f} /min)" if np.isfinite(gt_rate_hz)
               else "GT rate n/a",
               f"rate band {band[0]:.2f}-{band[1]:.2f} Hz"]
    fig.legend(handles, labels, loc="upper left", bbox_to_anchor=(x_leg, (bottom_in + h_fft) / height), frameon=False,
               fontsize=7.9, labelspacing=0.55, borderaxespad=0.0)

    fig.text(left / width, 1.0 - 0.18 / height, header[0], fontsize=10.5, color=INK, va="top")
    fig.text(left / width, 1.0 - 0.48 / height, header[1], fontsize=7.6, color=MUTED, va="top")
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=200)
    pdf.savefig(fig)
    plt.close(fig)


def draw_separate(path: Path, pdf, header: Tuple[str, str], series: Dict[str, Tuple[np.ndarray, np.ndarray]],
                  spectra: Dict[str, dict], gt: dict, gt_rate_hz: float, band: Tuple[float, float], max_freq: float,
                  span_label: str) -> None:
    """One row per signal (GT, RIIV, RIAV, RIFV): the time series (z-scored) left, the |FFT| (peak = 1) right."""
    keys = ("GT",) + MODS
    names = {"GT": gt["label"], **LONG}
    colors = {"GT": GT_COLOR, **COLORS}
    width, left, right, col_gap, top_in, bottom_in, row_h, gap = 12.6, 1.05, 0.3, 0.75, 1.3, 0.62, 1.25, 0.3
    height = top_in + bottom_in + len(keys) * row_h + (len(keys) - 1) * gap
    fig = plt.figure(figsize=(width, height))
    col_w = (width - left - right - col_gap) / 2
    ts_axes, fft_axes = [], []
    for i, key in enumerate(keys):
        y = bottom_in + (len(keys) - 1 - i) * (row_h + gap)
        ax_t = fig.add_axes([left / width, y / height, col_w / width, row_h / height],
                            sharex=ts_axes[0] if ts_axes else None)
        ax_f = fig.add_axes([(left + col_w + col_gap) / width, y / height, col_w / width, row_h / height],
                            sharex=fft_axes[0] if fft_axes else None)
        ts_axes.append(ax_t)
        fft_axes.append(ax_f)
        t, v = gt["series"] if key == "GT" else series[key]
        sd = float(np.std(v))
        ax_t.plot(t, (v - np.mean(v)) / sd if sd > 0 else v * 0, color=colors[key], lw=1.3 if key == "GT" else 1.1)
        ax_t.set_xlim(t[0], t[-1])
        ax_t.set_ylabel(names[key].replace(" (", "\n("), fontsize=8.5)
        s = gt if key == "GT" else spectra[key]
        keep = s["freq"] <= max_freq
        top = 1.0
        if np.isfinite(s["peak_mag"]) and s["peak_mag"] > 0:
            yv = s["mag"][keep] / s["peak_mag"]
            top = max(top, float(np.nanmax(yv)))
            ax_f.plot(s["freq"][keep], yv, color=colors[key], lw=1.6, zorder=4)
            ax_f.axvline(s["peak_hz"], color=colors[key], lw=1.0, alpha=0.9, zorder=3)
            ax_f.plot([s["peak_hz"]], [1.0], marker="s", ms=5.5, color=colors[key], mec="white", mew=0.8, zorder=6)
        if np.isfinite(gt_rate_hz):
            ax_f.axvline(gt_rate_hz, color=GT_COLOR, lw=1.1, ls=(0, (1, 2)), zorder=5)
        for edge in band:
            ax_f.axvline(edge, color=BAND_COLOR, lw=1.1, ls=(0, (4, 3)), zorder=1)
        ax_f.set_xlim(0.0, max_freq)
        ax_f.set_ylim(0.0, top * 1.12)
        peak = s["peak_hz"]
        ax_f.text(0.99, 0.93, f"peak {peak:.3f} Hz ({peak * 60:.1f} /min)" if np.isfinite(peak) else "peak n/a",
                  transform=ax_f.transAxes, ha="right", va="top", fontsize=8.2, color=INK)
        for ax in (ax_t, ax_f):
            style(ax)
            ax.tick_params(labelsize=8)
            if i < len(keys) - 1:
                ax.tick_params(labelbottom=False)
    ts_axes[0].set_title("(a) over time (band-passed, each z-scored)", fontsize=9.5, loc="left", pad=3)
    fft_axes[0].set_title(f"(b) |FFT| (peak = 1), {span_label}", fontsize=9.5, loc="left", pad=3)
    ts_axes[-1].set_xlabel("Time in the recording (s)", fontsize=9)
    fft_axes[-1].set_xlabel("Frequency (Hz)", fontsize=9)
    rate = (f"GT rate (breath count) {gt_rate_hz:.3f} Hz ({gt_rate_hz * 60:.1f} /min)" if np.isfinite(gt_rate_hz)
            else "GT rate n/a")
    fig.legend([Line2D([], [], color=GT_COLOR, lw=1.1, ls=(0, (1, 2))),
                Line2D([], [], color=BAND_COLOR, lw=1.1, ls=(0, (4, 3)))],
               [rate, f"rate band {band[0]:.2f}-{band[1]:.2f} Hz"], loc="upper left", ncol=2, frameon=False,
               fontsize=8.2, bbox_to_anchor=(left / width, 1.0 - 0.72 / height), borderaxespad=0.0)
    fig.text(left / width, 1.0 - 0.18 / height, header[0], fontsize=10.5, color=INK, va="top")
    fig.text(left / width, 1.0 - 0.48 / height, header[1], fontsize=7.6, color=MUTED, va="top")
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=200)
    pdf.savefig(fig)
    plt.close(fig)


DRAW = {"overlap": draw_overlap, "separate": draw_separate}


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def same_subject(a: object, b: object) -> bool:
    a, b = str(a).strip(), str(b).strip()
    return a == b or (a.isdigit() and b.isdigit() and int(a) == int(b))


def parse_subjects(items: Optional[Sequence[str]]) -> Dict[str, List[str]]:
    chosen: Dict[str, List[str]] = {}
    for item in items or []:
        dataset, _, subject = str(item).partition(":")
        if dataset not in DATASETS or not subject:
            raise SystemExit(f"--subjects {item}: write Dataset:Subject with Dataset in {', '.join(DATASETS)}")
        chosen.setdefault(dataset, []).append(subject)
    return chosen


def window_spans(t_first: float, t_last: float, window_sec: float) -> List[Tuple[float, float]]:
    """Windows on a grid from the start (0, W, 2W s ...) covered at least 90% by the series; W = 0: the whole span."""
    if window_sec <= 0:
        return [(t_first, t_last)]
    spans = []
    for k in range(int(math.floor(t_last / window_sec)) + 1):
        a, b = k * window_sec, (k + 1) * window_sec
        if min(b, t_last) - max(a, t_first) >= MIN_WINDOW_COVER * window_sec:
            spans.append((a, b))
    return spans


def dataset_rows(args, dataset: str, chosen: Dict[str, List[str]], out_root: Path) -> List[dict]:
    source = SOURCE[dataset]
    roots = [args.root_261007] if source == "261007" else [args.root_261013]
    script = find_script(SCRIPTS[source], args.script_261007 if source == "261007" else args.script_261013, roots)
    mod = load_script(script)
    for key, field, value in (("capnobase", "data_path", args.capnobase_path), ("bidmc", "data_path", args.bidmc_path),
                              ("steam2", "ppg_dir", args.steam2_ppg_dir), ("steam2", "rsp_dir", args.steam2_rsp_dir)):
        if value:
            mod.DATASET_CONFIGS[key][field] = str(value)
    spec = next(s for s in mod.WINDOW_SPECS if f"{s.order:02d}_{s.name}" == WINDOW)
    print(f"\n{dataset}: preprocessing of {script.name}", flush=True)
    data = mod.load_dataset(KEYS[dataset], spec)
    ann1_all = read_ann1(mod) if dataset == "BIDMC" else {}
    fs = float(mod.TARGET_FS)
    band = tuple(args.rate_band_hz)
    lengths = sorted({float(w) for w in args.window_sec}, reverse=True)
    excluded = [] if args.keep_issue_subjects else list(EXCLUDED[dataset])
    rows = []
    styles = [st for st in STYLES if st in args.styles]
    folders = {st: out_root / st / dataset for st in styles}
    dataset_pdfs = {st: {"pdf": None, "path": folders[st] / f"{dataset}_RIIV_RIAV_RIFV.pdf", "pages": 0}
                    for st in styles}  # whole recordings

    def open_pdf(slot: dict):
        if slot["pdf"] is None:
            slot["path"].parent.mkdir(parents=True, exist_ok=True)
            slot["pdf"] = PdfPages(slot["path"])
        slot["pages"] += 1
        return slot["pdf"]

    def close_pdf(slot: dict) -> None:
        if slot["pdf"] is not None:
            slot["pdf"].close()
            slot["pdf"] = None

    try:
        for subject in sorted(data):
            if dataset in chosen and not any(same_subject(subject, s) for s in chosen[dataset]):
                continue
            if any(same_subject(subject, x) for x in excluded):
                print(f"  {subject}: left out (MAE issue)")
                continue
            tag = mod.safe_tag(subject)
            subject_pdfs = {(st, w): {"pdf": None, "pages": 0, "path": folders[st] / tag / f"w{w:g}s"
                                      / f"{dataset}_{tag}_w{w:g}s_RIIV_RIAV_RIFV.pdf"}
                            for st in styles for w in lengths if w > 0}
            try:
                references = data[subject]["references"]
                multi = len(references) > 1
                for segment_id, ref in sorted(references.items()):
                    profile = ref["profile"]
                    beats = extract_modulations(mod, ref["model_input"], fs, tuple(profile["ppg_hr_range_bpm"]),
                                                ref.get("ppg_artifact_mask"))
                    if "times" not in beats or len(beats["times"]) < MIN_WINDOW_BEATS:
                        print(f"  {subject} recording {segment_id}: too few usable beats "
                              f"({beats.get('n_detected', 0)} detected)")
                        continue
                    resp_band = tuple(float(v) for v in ref["band_hz"])
                    full = {}
                    for k in MODS:
                        grid, values = resample(beats["times"], beats["values"][k])
                        full[k] = (grid, respiratory_band(values, RESAMPLE_HZ, resp_band))
                    breaths_all = gt_breaths(dataset, mod, ref, ann1_all.get(subject))
                    ref_signal = np.asarray(ref["signal"], dtype=np.float64)
                    times = beats["times"]
                    rec = f" · recording {segment_id}" if multi else ""
                    drawn = {}
                    for length in lengths:
                        whole = length <= 0
                        drawn[length] = 0
                        for a, b in window_spans(float(times[0]), float(times[-1]), length):
                            in_window = (times >= a) & (times < b) if not whole else np.ones(len(times), bool)
                            if int(in_window.sum()) < MIN_WINDOW_BEATS:
                                continue
                            series = {}
                            for k in MODS:
                                grid, values = full[k]
                                keep = (grid >= a) & (grid < b) if not whole else np.ones(len(grid), bool)
                                series[k] = (grid[keep], values[keep])
                            if min(len(series[k][0]) for k in MODS) < 8:
                                continue
                            spectra = {}
                            for k in MODS:
                                freq, mag = spectrum(series[k][1], RESAMPLE_HZ)
                                peak_hz, peak_mag = band_peak(freq, mag, band)
                                spectra[k] = {"freq": freq, "mag": mag, "peak_hz": peak_hz, "peak_mag": peak_mag}
                            g0 = max(0, int(round((a if not whole else float(times[0])) * fs)))
                            g1 = min(len(ref_signal), int(round((b if not whole else float(times[-1])) * fs)))
                            gt_values = ref_signal[g0:g1]
                            ok = np.isfinite(gt_values)
                            gt_freq, gt_mag = spectrum(gt_values[ok], fs)
                            gt_peak_hz, gt_peak_mag = band_peak(gt_freq, gt_mag, band)
                            gt = {"label": f"GT ({MODALITY[dataset]})", "series": (np.arange(g0, g1)[ok] / fs, gt_values[ok]),
                                  "freq": gt_freq, "mag": gt_mag, "peak_hz": gt_peak_hz, "peak_mag": gt_peak_mag}
                            breaths = breaths_all[(breaths_all >= a * fs) & (breaths_all < b * fs)]
                            gt_rate_hz = len(breaths) / (b - a) if b > a else math.nan
                            pulse_rate = float(np.mean(beats["values"]["RIFV"][in_window]))
                            span = f"{a:.0f}-{b:.0f} s"
                            header = (f"{dataset} {subject}{rec}: respiratory modulations of the PPG · "
                                      + (f"whole recording {span}" if whole else f"{length:g} s window {span}")
                                      + f" · {int(in_window.sum())} beats, mean pulse rate {pulse_rate:.0f} /min",
                                      f"PPG = model input (64 Hz); beats on the 0.5-5 Hz pulse band; series at "
                                      f"{RESAMPLE_HZ:g} Hz, band-passed {resp_band[0]:.2f}-{resp_band[1]:.2f} Hz; GT rate = "
                                      f"{GT_NAME[dataset]} / {'recording' if whole else 'window'} length")
                            stem = (f"{tag}{f'_rec{segment_id}' if multi else ''}"
                                    + ("" if whole else f"_{a:.0f}-{b:.0f}s") + "_RIIV_RIAV_RIFV")
                            figures = {}
                            for st in styles:
                                target = folders[st] if whole else folders[st] / tag / f"w{length:g}s"
                                pdf = open_pdf(dataset_pdfs[st] if whole else subject_pdfs[(st, length)])
                                DRAW[st](target / f"{stem}.png", pdf, header, series, spectra, gt, gt_rate_hz, band,
                                         args.max_freq_hz, "whole recording" if whole else f"{length:g} s window {span}")
                                figures[st] = str((target / f"{stem}.png").relative_to(out_root))
                            drawn[length] += 1
                            row = {"Dataset": dataset, "Subject": subject, "Segment_ID": int(segment_id),
                                   "Window_Sec": 0.0 if whole else length, "Start_Sec": a, "End_Sec": b,
                                   "Beats_Used": int(in_window.sum()), "Mean_Pulse_Rate_BPM": pulse_rate,
                                   "Filter_Band_Low_Hz": resp_band[0], "Filter_Band_High_Hz": resp_band[1],
                                   "GT_Breaths": int(len(breaths)), "GT_Rate_Hz": gt_rate_hz, "GT_Ref_Peak_Hz": gt_peak_hz}
                            for k in MODS:
                                row[f"{k}_Peak_Hz"] = spectra[k]["peak_hz"]
                                row[f"{k}_Abs_Diff_To_GT_Hz"] = abs(spectra[k]["peak_hz"] - gt_rate_hz)
                            diffs = {k: row[f"{k}_Abs_Diff_To_GT_Hz"] for k in MODS
                                     if np.isfinite(row[f"{k}_Abs_Diff_To_GT_Hz"])}
                            best = min(diffs.values()) if diffs else math.nan
                            row["Closest_To_GT"] = "+".join(k for k, v in diffs.items() if v <= best + 1e-9)  # ties
                            for st in styles:
                                row[f"Figure_{st.capitalize()}"] = figures[st]
                            rows.append(row)
                    done = ", ".join("whole recording" if w <= 0 else f"{n} windows of {w:g} s" for w, n in drawn.items())
                    print(f"  {subject}{f' rec {segment_id}' if multi else ''}: {len(times)} beats used; {done} drawn",
                          flush=True)
            finally:
                for slot in subject_pdfs.values():
                    close_pdf(slot)
    finally:
        for slot in dataset_pdfs.values():
            close_pdf(slot)
    del data
    return rows


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="RIIV, RIAV, RIFV of the PPG: time series and FFT spectra per window")
    p.add_argument("--window-sec", nargs="+", type=float, default=list(WINDOW_SECS),
                   help="window lengths in s, every one drawn (default 60 30); 0 = the whole recording")
    p.add_argument("--styles", nargs="+", choices=list(STYLES), default=list(STYLES),
                   help="figure styles, each in its own folder (default overlap separate)")
    p.add_argument("--datasets", nargs="+", choices=list(DATASETS), default=list(DATASETS))
    p.add_argument("--subjects", nargs="+", default=None, help="Dataset:Subject (default: every subject)")
    p.add_argument("--keep-issue-subjects", action="store_true")
    p.add_argument("--rate-band-hz", nargs=2, type=float, default=list(RATE_BAND_HZ))
    p.add_argument("--max-freq-hz", type=float, default=MAX_FREQ_HZ)
    p.add_argument("--root-261007", type=Path, default=ROOT_261007)
    p.add_argument("--root-261013", type=Path, default=ROOT_261013)
    p.add_argument("--output-root", type=Path, default=OUTPUT_ROOT)
    p.add_argument("--script-261007", type=Path, default=None)
    p.add_argument("--script-261013", type=Path, default=None)
    p.add_argument("--capnobase-path", type=Path, default=None)
    p.add_argument("--bidmc-path", type=Path, default=None)
    p.add_argument("--steam2-ppg-dir", type=Path, default=None)
    p.add_argument("--steam2-rsp-dir", type=Path, default=None)
    return p.parse_args(list(sys.argv[1:] if argv is None else argv))


def main(argv: Optional[Sequence[str]] = None) -> None:
    args = parse_args(argv)
    chosen = parse_subjects(args.subjects)
    datasets = [d for d in args.datasets if not chosen or d in chosen]
    out_root = Path(args.output_root)
    out_root.mkdir(parents=True, exist_ok=True)
    rows = []
    for dataset in datasets:
        rows += dataset_rows(args, dataset, chosen, out_root)
    table = pd.DataFrame(rows)
    summary = []
    if not table.empty:
        for (dataset, length), g in table.groupby(["Dataset", "Window_Sec"], sort=False):
            row = {"Dataset": dataset, "Window_Sec": length, "Windows": int(len(g))}
            for k in MODS:
                d = g[f"{k}_Abs_Diff_To_GT_Hz"].dropna()
                row[f"{k}_Mean_Abs_Diff_Hz"] = float(d.mean()) if len(d) else math.nan
                row[f"{k}_Within_0.02Hz"] = int((d <= 0.02).sum())
                row[f"{k}_Closest"] = int(g["Closest_To_GT"].str.split("+").apply(lambda ks: k in ks).sum())
            summary.append(row)
    notes = pd.DataFrame([
        ("RIIV", "PPG value at every pulse peak (PPG low-passed at 8 Hz, baseline kept)"),
        ("RIAV", "pulse amplitude: peak minus the trough before it"),
        ("RIFV", "instantaneous pulse rate: 60 / peak-to-peak interval (beats per minute)"),
        ("Beats", "peaks of the 0.5-5 Hz pulse band; used when the interval is in the profile's heart-rate range, "
                  f"within {INTERVAL_TOLERANCE:.0%} of the median of {MEDIAN_BEATS} intervals, outside expert PPG artifacts"),
        ("Series", f"one value per used beat at the peak time, resampled to {RESAMPLE_HZ:g} Hz by cubic spline, then "
                   "band-passed with the recording's respiratory band of the source script (Filter_Band_*; 3rd-order "
                   "Butterworth, zero phase)"),
        ("Windows", "lengths " + ", ".join("whole recording" if w <= 0 else f"{w:g} s" for w in args.window_sec)
                    + f"; windows on a grid from the start of the recording, covered at least {MIN_WINDOW_COVER:.0%} by "
                    f"the series, at least {MIN_WINDOW_BEATS} used beats; extraction and filtering on the whole "
                    "recording, then cut; Window_Sec 0 = whole recording"),
        ("Spectrum", "per window: detrend, Hamming window, zero padding (>= 4096 points); figure |FFT| / its peak "
                     f"in {args.rate_band_hz[0]:g}-{args.rate_band_hz[1]:g} Hz"),
        ("GT rate", "GT breaths in the window / its length (CapnoBase capnogram breaths, BIDMC ann1, STEAM2 belt "
                    "breaths); with 60 s windows the GT breath count of the minute, with 30 s twice the half-minute count"),
        ("Reference", "Karlen W et al. IEEE TBME 2013;60(7):1946-53; Charlton PH et al. Physiol Meas 2016;37(4):610-26"),
    ], columns=["Item", "Description"])
    path = out_root / f"{NAME}.xlsx"
    with pd.ExcelWriter(path, engine="openpyxl") as writer:
        table.round(5).to_excel(writer, sheet_name="Windows", index=False)
        pd.DataFrame(summary).round(5).to_excel(writer, sheet_name="Summary", index=False)
        notes.to_excel(writer, sheet_name="Notes", index=False)
    print(f"\nFigures under {out_root}; table: {path}")


if __name__ == "__main__":
    main()
