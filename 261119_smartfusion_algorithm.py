# -*- coding: utf-8 -*-
r"""261119_smartfusion_algorithm: (1) the PPG, its respiratory components (RIIV, RIAV, RIFV) and the reference breathing
in every 30 s window of every subject, saved as one PDF per subject with a per-subject summary; (2) the respiratory
rate of the smart fusion algorithm (Karlen et al. 2013) in every scored minute, its MAE per subject and per dataset,
compared with the MAE of the final RRWaveNet results on the same minutes (MAE only).

Data: the signals of the preprocessing of the script of the final results, loaded as a module (its data paths are
used; the path options override):
  CapnoBase      261007_RRWaveNet_Ensemble_NoEvent.py      reference = capnogram
  BIDMC, STEAM2  261013_RRWaveNet_Adult_SubjectVal_WD.py   reference = impedance (BIDMC), belt (STEAM2)
  PPG  the continuous model input (64 Hz); reference = the continuous reference of the scoring (64 Hz, band-passed with
  the recording's respiratory band and aligned to the PPG as the source script does)

Components of the PPG (Karlen et al. 2013; Charlton et al. 2016), as 261119_PPG_RespMod_Sync
  Beats   peaks of the 0.5-5 Hz pulse band, at least 60 / (highest heart rate of the profile) and at least 0.6 x the
          recording's beat interval apart (spectral heart rate of the PPG, so a dicrotic / secondary wave is not counted
          as a second beat); the peak = maximum of the PPG low-passed at 8 Hz within +-0.15 s, the trough = minimum
          between the previous peak and this one; a beat is used when its interval lies in the heart-rate range of the
          profile, within 30% of the median of the 7 intervals around it, and (CapnoBase) outside the expert PPG artifacts
  RIIV    PPG value at every used pulse peak
  RIAV    pulse amplitude: peak minus the trough before it
  RIFV    instantaneous pulse rate: 60 / peak-to-peak interval
  Series  one value per used beat, resampled to 4 Hz by cubic spline, band-passed with the recording's respiratory band
          (3rd-order Butterworth, zero phase) on the whole recording; left empty between used beats more than 3 s apart

Part 1 - figures (every subject, the MAE issue subjects included and marked)
  Every 30 s window from the start of the recording covered at least 90% by the series and with at least 10 used beats:
    (a) reference breathing, (b) PPG with its pulse peaks (RIIV), troughs (peak - trough = RIAV) and baseline (the PPG
    low-passed at the upper edge of the respiratory band), (c) RIIV, RIAV, RIFV (z-scored; a component in antiphase with
    RIIV is drawn x -1 and marked), (d) reference vs the mean of the three components, (e) |FFT| of the components and
    the reference with their peaks; beside the panels: beats, GT breaths, pairwise PLV, PLV with the reference (with the
    mean phase difference) and the FFT rates; GT breaths as dashed vertical lines.
  PLV = |mean over the window of exp(i (phase A - phase B))| of the Hilbert phases (analytic signals of the whole
  recording, cut into the window): 1 = a constant phase relation (in phase, antiphase or a fixed delay), 0 = none.
  Per subject: Figures\{Dataset}\{Dataset}_{subject}_all_windows.pdf (one page per window) and
  Figures\{Dataset}\{Dataset}_{subject}_windows_summary.png (PLV and FFT rates per window); --window-png also saves
  every window as Figures\{Dataset}\{subject}\{Dataset}_{subject}_{t0}-{t1}s.png.

Part 2 - smart fusion vs RRWaveNet (MAE only)
  Minutes  the 60 s minutes of the stored RRWaveNet results (the scoring grid), scored as the final results: CapnoBase
           breaths detected on the capnogram with the RRest agreement rule; BIDMC annotator 1 (ann1) with the
           expert-label rule of 261013; STEAM2 the stored GT; GT = breath count of the minute
  Smart fusion (Karlen W, Raman S, Ansermino JM, Dumont GA. IEEE TBME 2013;60(7):1946-53): in every minute the rate of
           RIIV, RIAV and RIFV = the peak of the |FFT| of its series (detrend, Hamming window, zero padding) in the rate
           band of the scoring (BIDMC / STEAM2 6-30, CapnoBase 3-60 /min); the minute is estimated when the standard
           deviation of the three rates (N - 1) is below 4 breaths/min: smart fusion rate = their mean; otherwise the
           minute has no estimate (rejected). A minute needs at least 10 used beats and every series covering 90% of it.
           Karlen et al. used 32 s windows; here the scored minutes are used so both methods are judged on the same
           minutes and GT.
  RRWaveNet the final results: the ensemble of the members (mean of their outputs) of every LOSOCV fold, its stored rate
           of the minute (Pred_Wave_FFT_RR of Fold_Summaries\{subject}_ensemble_RR60s_windows.csv)
             CapnoBase      261007 (D:\PPG2RSP_RRWaveNet_Inspired\261007_RRWaveNet_NoEvent_v1), variant full
             BIDMC, STEAM2  261013 (D:\PPG2RSP_RRWaveNet_Inspired\261013_RRWavNet_STEAM_BIDMC_v1), the weight decay
                            selected in the fold (WD_Selection)
  MAE per subject = mean |estimate - GT| over the minutes:
    SF_MAE                    smart fusion on the minutes it estimated (accepted minutes)
    RRWaveNet_MAE_SF_Minutes  RRWaveNet on the same accepted minutes (the paired comparison)
    RRWaveNet_MAE             RRWaveNet on every scored minute (the final results)
    SF_NoRejection_MAE        mean of the three rates on every scored minute with three rates (no rejection)
    SF_Coverage               accepted minutes / scored minutes
  Per dataset and over all datasets: mean and SD of the subject MAEs, the MAE over every minute (pooled), the coverage,
  and a Wilcoxon signed-rank test of SF_MAE vs RRWaveNet_MAE_SF_Minutes over the subjects (at least 5). A subject without
  stored RRWaveNet results gets smart fusion on its own 60 s grid (Subjects, Minutes) but is left out of the comparison
  (Missing). The MAE issue subjects of the
  final results (CapnoBase 0115, BIDMC 27, STEAM2 12_YJJ, 22_LMS, 26_CKK) are left out of part 2 as in the final results
  (--keep-issue-subjects keeps them).

Outputs (D:\PPG2RSP_RRWaveNet_Inspired\261119_smartfusion_algorithm)
  Figures\{Dataset}\...                         part 1
  SmartFusion_vs_RRWaveNet_MAE.png / .pdf       subject MAE per dataset and all datasets: smart fusion (accepted
                                                minutes), RRWaveNet on the same minutes, RRWaveNet on every scored minute
  261119_smartfusion_algorithm.xlsx
    Datasets  per dataset and all: subjects, minutes, coverage, MAE of every method, Wilcoxon p
    Subjects  per subject: minutes, coverage, MAE of every method
    Minutes   every minute: GT, the three component rates, their SD, accepted, smart fusion rate, RRWaveNet rate, errors
    Windows   every 30 s window of part 1: beats, GT, pairwise PLV, PLV with the reference, phases, FFT rates
    Missing   subjects without stored RRWaveNet results or without a scored minute
    Notes

Run: python 261119_smartfusion_algorithm.py     (Spyder: runfile(...), no options needed)
Options: --datasets, --subjects CapnoBase:0121 BIDMC:05 ..., --no-figures (part 2 only), --window-png,
--keep-issue-subjects, --sd-threshold (default 4), --root-261007, --root-261013, --output-root, --script-261007,
--script-261013, --capnobase-path, --bidmc-path, --steam2-ppg-dir, --steam2-rsp-dir
"""

import argparse
import importlib.util
import json
import math
import re
import sys
import warnings
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
from scipy.signal import butter, detrend, filtfilt, find_peaks, hilbert, sosfiltfilt
from scipy.stats import wilcoxon

# ---------------------------------------------------------------------------
# Settings
# ---------------------------------------------------------------------------

NAME = "261119_smartfusion_algorithm"
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
MODALITY = {"CapnoBase": "capnogram", "BIDMC": "impedance", "STEAM2": "belt"}
GT_NAME = {"CapnoBase": "breaths detected on the capnogram (RRest rule)", "BIDMC": "ann1 (expert-label rule)",
           "STEAM2": "stored GT (belt breaths)"}
ISSUE_SUBJECTS = {"STEAM2": ("22_LMS", "26_CKK", "12_YJJ"), "BIDMC": ("27",), "CapnoBase": ("0115",)}
FINAL_VARIANT_261007 = "full"
MODS = ("RIIV", "RIAV", "RIFV")
PAIRS = (("RIIV", "RIAV"), ("RIIV", "RIFV"), ("RIAV", "RIFV"))
RESAMPLE_HZ = 4.0
PPG_LOWPASS_HZ = 8.0
PEAK_SEARCH_SEC = 0.15
INTERVAL_TOLERANCE = 0.30
MEDIAN_BEATS = 7
MIN_BEAT_DISTANCE_FRACTION = 0.6  # beats at least 0.6 x the spectral beat interval apart (no dicrotic double count)
MAX_BEAT_GAP_SEC = 3.0            # series left empty between used beats further apart than this
FFT_MIN_POINTS = 4096
WINDOW_SEC = 30.0                 # figure windows
MIN_WINDOW_COVER = 0.9
MIN_WINDOW_BEATS = 10
SD_THRESHOLD = 4.0                # smart fusion: SD of the three rates below this (breaths/min)
MAX_LAG = 20                      # samples of the 4 Hz grid (+-5 s) for the lagged r of the measures
CHANCE_PLV = 0.51                 # 95th percentile of the PLV of unrelated 30 s windows (261119 analysis)
COLORS = {"RIIV": "#2a78d6", "RIAV": "#eb6834", "RIFV": "#1baf7a", "REF": "#111111", "MEAN": "#5f5c55",
          "BR": "#b9b6ae", "PPG": "#3b5bdb", "PEAK": "#e03131", "TROUGH": "#5f5c55", "BASE": "#c2255c"}
METHOD_COLORS = {"SF": "#eb6834", "RRW_SF": "#2a78d6", "RRW_ALL": "#8fb8e8"}
INK, MUTED, GRID = "#1f1f1f", "#6b6a66", "#e4e3df"

_MODULES: Dict[str, object] = {}


# ---------------------------------------------------------------------------
# Source scripts and stored results
# ---------------------------------------------------------------------------

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
        name = "smartfusion_src_" + re.sub(r"\W", "_", path.stem)
        spec = importlib.util.spec_from_file_location(name, key)
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module
        spec.loader.exec_module(module)
        _MODULES[key] = module
    return _MODULES[key]


def read_json(path: Path) -> Optional[dict]:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


def rrwavenet_table(args, mod, spec, dataset: str, subject: str) -> Tuple[Optional[pd.DataFrame], str]:
    """Minute table of the final RRWaveNet ensemble of the subject's fold (stored by the source script), or None and
    why it is missing."""
    key = KEYS[dataset]
    if SOURCE[dataset] == "261007":
        root = Path(args.root_261007)
        variant = FINAL_VARIANT_261007
    else:
        root = Path(args.root_261013)
        found = mod.stored_selection(root, list(mod.VARIANTS.values()), spec, key, subject)
        if found is not None:
            variant = found[1].name
        else:  # selection stored with other candidates than the module's defaults: the variant it names
            variant = (read_json(mod.selection_path(root, spec, key, subject)) or {}).get("Selected_Variant")
        if not variant:
            return None, f"no weight decay selection ({mod.selection_path(root, spec, key, subject)})"
    path = mod.ensemble_paths(mod.condition_root(root, variant, spec, key), subject)[1]
    if not path.exists():
        return None, f"no ensemble minute table ({path})"
    return pd.read_csv(path, dtype={"Subject": str}, encoding="utf-8-sig"), str(path)


# ---------------------------------------------------------------------------
# GT (as the final results)
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
    """GT breaths of the final results (sample indices of the 64 Hz reference)."""
    if dataset == "CapnoBase":
        return np.asarray(ref["detected_breaths"], dtype=np.int64)
    if dataset == "BIDMC":
        return ann1_timeline(mod, ann1, ref) if ann1 is not None else np.asarray(ref["breaths"], dtype=np.int64)
    return np.asarray(ref["breaths"], dtype=np.int64)


def usable(mod, ref: dict, start: int, end: int) -> Tuple[bool, str]:
    if float(np.mean(ref["structural_valid"][start:end])) < mod.MIN_USABLE_PROPORTION:
        return False, "ref_structural_invalid"
    if ref.get("artifact_mask") is not None and float(np.mean(ref["artifact_mask"][start:end])) > 0.0:
        return False, "ref_artifact_label"
    return True, ""


def minute_gt(mod, dataset: str, ref: dict, start: int, end: int, breaths: np.ndarray,
              stored: Optional[pd.Series]) -> Tuple[bool, str, int]:
    """Validity, reason and GT breath count of one minute, as in the final results."""
    fs = mod.TARGET_FS
    if dataset == "STEAM2":  # stored GT of the source
        if stored is None:
            return False, "no stored minute", 0
        return (bool(stored["Ref_Valid"]), str(stored.get("Ref_Exclusion_Reason", "") or ""),
                int(stored["GT_Breath_Count"]))
    profile = ref["profile"]
    rr_band = profile["rr_band_bpm"]
    inside = breaths[(breaths >= start) & (breaths < end)]
    count = int(len(inside))
    ok, reason = usable(mod, ref, start, end)
    if not ok:
        return False, reason, count
    if dataset == "BIDMC":  # ann1, expert-label rule of 261013
        label_rr = float(60.0 * fs / np.mean(np.diff(inside))) if len(inside) >= 2 else math.nan
        if not np.isfinite(label_rr):
            return False, "ref_rr_undefined", count
        if not (rr_band[0] <= label_rr <= rr_band[1]):
            return False, "ref_out_of_band", count
        return True, "", count
    agree, why, _, _ = mod.reference_agreement(ref["signal"][start:end], fs, rr_band, profile["fft_subharmonic_check"])
    return bool(agree), why, count


# ---------------------------------------------------------------------------
# Beats and the components
# ---------------------------------------------------------------------------

def lowpass(signal: np.ndarray, fs: float, cutoff: float, order: int = 3) -> np.ndarray:
    b, a = butter(order, cutoff / (0.5 * fs), btype="low")
    return filtfilt(b, a, signal)


def detect_beats(mod, ppg: np.ndarray, fs: float, hr_range_bpm: Tuple[float, float],
                 artifact: Optional[np.ndarray]) -> dict:
    """Pulse peaks, troughs, the used-beat mask and the RIIV / RIAV / RIFV value of every used beat."""
    ppg, _ = mod.sanitize_signal(np.asarray(ppg, dtype=np.float64))
    pulse = mod.bandpass_filter(ppg, fs, mod.PPG_PULSE_BAND[0], mod.PPG_PULSE_BAND[1])[0]
    smooth = lowpass(ppg, fs, PPG_LOWPASS_HZ)
    spread = float(np.std(pulse))
    if spread < 1e-8:
        return {"n_detected": 0}
    distance = 60.0 / hr_range_bpm[1]
    hr_hz = mod.estimate_ppg_heart_rate_hz(ppg, int(round(fs)), hr_range_bpm)
    if np.isfinite(hr_hz) and hr_hz > 0:
        distance = max(distance, MIN_BEAT_DISTANCE_FRACTION / hr_hz)
    detected, _ = find_peaks(pulse, distance=max(1, int(round(distance * fs))), prominence=max(1e-8, 0.10 * spread))
    half = int(round(PEAK_SEARCH_SEC * fs))
    peaks = np.unique(np.array([max(0, p - half) + int(np.argmax(smooth[max(0, p - half):p + half + 1]))
                                for p in detected], int))
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
    return {"n_detected": int(len(peaks)), "smooth": smooth, "beats": beats, "troughs": troughs, "ok": ok,
            "times": beats[ok] / fs, "values": {k: v[ok] for k, v in values.items()},
            "hr_bpm": float(hr_hz * 60.0) if np.isfinite(hr_hz) else math.nan}


def respiratory_band(x: np.ndarray, fs: float, band: Tuple[float, float], order: int = 3) -> np.ndarray:
    high = min(float(band[1]), 0.45 * fs)
    sos = butter(order, [float(band[0]) / (0.5 * fs), high / (0.5 * fs)], btype="band", output="sos")
    try:
        return sosfiltfilt(sos, x)
    except ValueError:  # series too short for the filter
        return x - np.mean(x)


def component_series(beats: dict, band: Tuple[float, float]) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
    """The three components on a 4 Hz grid between the first and the last used beat, band-passed, empty in gaps."""
    times = beats["times"]
    grid = np.arange(times[0], times[-1], 1.0 / RESAMPLE_HZ)
    kind = "cubic" if len(times) >= 4 else "linear"
    after = np.clip(np.searchsorted(times, grid, side="right"), 1, len(times) - 1)
    gap = times[after] - times[after - 1]
    comp = {}
    for k in MODS:
        values = respiratory_band(interp1d(times, beats["values"][k], kind=kind)(grid), RESAMPLE_HZ, band)
        comp[k] = np.where(gap > MAX_BEAT_GAP_SEC, np.nan, values)
    return grid, comp


def spectrum(x: np.ndarray, fs: float) -> Tuple[np.ndarray, np.ndarray]:
    """Rate (/min) axis and |FFT|: detrend, Hamming window, zero padding."""
    x = np.asarray(x, float)
    x = x[np.isfinite(x)]
    if len(x) < 8 or float(np.std(x)) < 1e-12:
        return np.zeros(1), np.full(1, np.nan)
    n = max(FFT_MIN_POINTS, 1 << int(math.ceil(math.log2(len(x)))))
    return np.fft.rfftfreq(n, d=1.0 / fs) * 60.0, np.abs(np.fft.rfft(detrend(x) * np.hamming(len(x)), n=n))


def spectrum_rate(x: np.ndarray, fs: float, band_bpm: Tuple[float, float]) -> float:
    rate, mag = spectrum(x, fs)
    keep = (rate >= band_bpm[0]) & (rate <= band_bpm[1]) & np.isfinite(mag)
    return float(rate[keep][np.argmax(mag[keep])]) if keep.any() else math.nan


# ---------------------------------------------------------------------------
# Synchronization measures
# ---------------------------------------------------------------------------

def pearson(a: np.ndarray, b: np.ndarray) -> float:
    ok = np.isfinite(a) & np.isfinite(b)
    if ok.sum() < 8 or np.std(a[ok]) < 1e-12 or np.std(b[ok]) < 1e-12:
        return math.nan
    return float(np.corrcoef(a[ok], b[ok])[0, 1])


def analytic_phase(x: np.ndarray) -> np.ndarray:
    ok = np.isfinite(x)
    out = np.full(len(x), np.nan)
    if ok.sum() >= 8:
        out[ok] = np.angle(hilbert(np.where(ok, x, 0.0)))[ok]
    return out


def plv(phase_a: np.ndarray, phase_b: np.ndarray, idx: np.ndarray) -> Tuple[float, float]:
    """Phase locking value and mean phase difference (degrees, a - b) over the grid indices idx."""
    d = phase_a[idx] - phase_b[idx]
    d = d[np.isfinite(d)]
    if len(d) < 8:
        return math.nan, math.nan
    z = np.mean(np.exp(1j * d))
    return float(np.abs(z)), float(np.degrees(np.angle(z)))


def zscore(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, float)
    ok = np.isfinite(x)
    if ok.sum() < 3 or np.std(x[ok]) < 1e-12:
        return np.full_like(x, np.nan)
    return (x - np.nanmean(x)) / np.nanstd(x)


# ---------------------------------------------------------------------------
# Part 1: figures
# ---------------------------------------------------------------------------

def style(ax) -> None:
    ax.grid(True, color=GRID, lw=0.6)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.tick_params(labelsize=8.5)


def window_page(ctx: dict, a: float, b: float) -> Tuple[plt.Figure, dict]:
    """One 30 s window: (a) reference, (b) PPG with peaks / troughs / baseline, (c) components, (d) reference vs
    their mean, (e) spectra; and its measures."""
    fs, grid, comp, refg, ph = ctx["fs"], ctx["grid"], ctx["comp"], ctx["refg"], ctx["phase"]
    rs, smooth, base = ctx["ref_signal"], ctx["smooth"], ctx["baseline"]
    bk, tr, okb = ctx["beats"], ctx["troughs"], ctx["ok"]
    rb, band, br = ctx["rate_band"], ctx["band"], ctx["breaths_sec"]
    idx = np.flatnonzero((grid >= a) & (grid < b))
    tg = grid[idx]
    z = {k: zscore(comp[k][idx]) for k in MODS}
    zr = zscore(refg[idx])
    sign = {k: (1.0 if k == "RIIV" or not (pearson(comp[k][idx], comp["RIIV"][idx]) < 0) else -1.0) for k in MODS}
    z = {k: sign[k] * z[k] for k in MODS}
    with warnings.catch_warnings():  # samples where every component is empty stay empty
        warnings.simplefilter("ignore", RuntimeWarning)
        zmean = zscore(np.nanmean(np.vstack([z[k] for k in MODS]), axis=0)) if len(idx) else np.zeros(0)
    pair = {(x, y): plv(ph[x], ph[y], idx) for x, y in PAIRS}
    withref = {k: plv(ph[k], ph["REF"], idx) for k in MODS}
    rates = {k: spectrum_rate(comp[k][idx], RESAMPLE_HZ, rb) for k in MODS}
    i0, i1 = int(round(a * fs)), min(len(rs), int(round(b * fs)))
    ref_rate = spectrum_rate(rs[i0:i1], fs, rb)
    nb = int(((br >= a) & (br < b)).sum())
    rr_mean = float(np.nanmean(list(rates.values())))
    used = (bk >= i0) & (bk < i1) & okb
    rej = (bk >= i0) & (bk < i1) & ~okb
    tu = (tr >= i0) & (tr < i1) & okb
    fig = plt.figure(figsize=(12.5, 12.4))
    L, W = 0.07, 0.64
    x0 = L + W + 0.025
    axr = fig.add_axes([L, 0.805, W, 0.095])
    axp = fig.add_axes([L, 0.57, W, 0.2], sharex=axr)
    axa = fig.add_axes([L, 0.405, W, 0.125], sharex=axr)
    axb = fig.add_axes([L, 0.235, W, 0.125], sharex=axr)
    axc = fig.add_axes([L, 0.09, W, 0.075])
    for ax in (axr, axp, axa, axb):
        style(ax)
        for x in br[(br >= a) & (br < b)]:
            ax.axvline(x, color=COLORS["BR"], lw=0.9, ls=(0, (3, 3)), zorder=1)
    t = np.arange(i0, i1) / fs
    axr.plot(t, rs[i0:i1], color=COLORS["REF"], lw=2.4, zorder=3)
    axr.set_yticks([])
    axr.set_title(f"(a) Breathing: reference ({ctx['modality']}, band-passed)", loc="left", fontsize=10.5, pad=4)
    axr.tick_params(labelbottom=False)
    j1 = min(i1, len(smooth))
    axp.plot(np.arange(i0, j1) / fs, smooth[i0:j1], color=COLORS["PPG"], lw=1.3, zorder=2)
    axp.plot(np.arange(i0, j1) / fs, base[i0:j1], color=COLORS["BASE"], lw=1.6, ls=(0, (5, 3)), zorder=3)
    axp.plot(bk[used] / fs, smooth[bk[used]], color=COLORS["PEAK"], lw=1.6, zorder=4)
    axp.plot(bk[used] / fs, smooth[bk[used]], ls="none", marker="o", ms=6.5, color=COLORS["PEAK"], mec="white",
             mew=0.8, zorder=5)
    axp.plot(tr[tu] / fs, smooth[tr[tu]], color=COLORS["TROUGH"], lw=1.2, zorder=4)
    axp.plot(tr[tu] / fs, smooth[tr[tu]], ls="none", marker="o", ms=4.5, color=COLORS["TROUGH"], mec="white",
             mew=0.6, zorder=5)
    if rej.any():
        axp.plot(bk[rej] / fs, smooth[bk[rej]], ls="none", marker="x", ms=6, color="#868e96", mew=1.4, zorder=5)
    axp.set_yticks([])
    axp.tick_params(labelbottom=False)
    axp.set_title("(b) PPG (model input, low-passed 8 Hz) with its pulse peaks (red: RIIV), troughs (grey: peak - trough "
                  "= RIAV) and baseline (dashed)", loc="left", fontsize=10.0, pad=4)
    for k in MODS:
        axa.plot(tg, z[k], color=COLORS[k], lw=1.8, zorder=3)
    flipped = [k for k in MODS if sign[k] < 0]
    axa.set_title("(c) RIIV, RIAV and RIFV" + (f" ({', '.join(flipped)} x -1: antiphase with RIIV)" if flipped else ""),
                  loc="left", fontsize=10.5, pad=4)
    axa.set_ylabel("z", fontsize=9)
    axa.tick_params(labelbottom=False)
    axb.plot(tg, zr, color=COLORS["REF"], lw=2.0, zorder=3)
    axb.plot(tg, zmean, color=COLORS["MEAN"], lw=1.8, ls=(0, (6, 2)), zorder=4)
    axb.set_title("(d) Reference breathing vs the mean of the three components", loc="left", fontsize=10.5, pad=4)
    axb.set_ylabel("z", fontsize=9)
    axb.set_xlabel("Time in the recording (s)", fontsize=9)
    axb.set_xlim(a, b)
    style(axc)
    top = 1.0
    xmax = 40.0 if rb[1] <= 40 else float(rb[1])  # CapnoBase (children): up to 60 /min
    for k in list(MODS) + ["REF"]:
        rate, mag = spectrum(rs[i0:i1], fs) if k == "REF" else spectrum(comp[k][idx], RESAMPLE_HZ)
        keep = (rate >= rb[0]) & (rate <= rb[1]) & np.isfinite(mag)
        if not keep.any() or np.max(mag[keep]) <= 0:
            continue
        mm = mag / np.max(mag[keep])
        sel = rate <= xmax
        axc.plot(rate[sel], mm[sel], color=COLORS[k], lw=2.0 if k == "REF" else 1.6, zorder=4 if k == "REF" else 3)
        axc.plot([rate[keep][np.argmax(mag[keep])]], [1.0], marker="s", ms=6, color=COLORS[k], mec="white", mew=0.8,
                 zorder=6)
        top = max(top, float(np.nanmax(mm[sel])))
    axc.set_xlim(0, xmax)
    axc.set_ylim(0, min(top, 2.5) * 1.08)
    axc.set_xlabel("breaths/min", fontsize=9)
    axc.set_ylabel("|FFT|", fontsize=9)
    axc.set_title("(e) |FFT| (peak = 1) of the components and the reference, peaks marked", loc="left", fontsize=10.5,
                  pad=4)
    fig.text(x0, 0.9, f"window {a:.0f}-{b:.0f} s\nbeats used {int(used.sum())}"
             + (f", rejected {int(rej.sum())} (x)" if rej.any() else "") + f"\nGT breaths {nb} ({nb * 60.0 / (b - a):.1f} /min)",
             family="monospace", fontsize=8.6, va="top", color=INK)
    fig.text(x0, 0.53, "pairwise PLV\n" + "\n".join(f"{x}-{y}  {pair[(x, y)][0]:.2f}" for x, y in PAIRS),
             family="monospace", fontsize=8.6, va="top", color=INK)
    fig.text(x0, 0.36, "with the reference\n" + "\n".join(
        f"{k} PLV {withref[k][0]:.2f} ({withref[k][1]:+.0f} deg)" for k in MODS),
        family="monospace", fontsize=8.6, va="top", color=INK)
    fig.text(x0, 0.165, "FFT rate (/min)\n" + "\n".join(f"{k:<5} {rates[k]:5.1f}" for k in MODS)
             + f"\nmean  {rr_mean:5.1f}\nref.  {ref_rate:5.1f}\nerror {rr_mean - ref_rate:+5.1f}",
             family="monospace", fontsize=8.6, va="top", color=INK)
    h = [Line2D([], [], color=COLORS["REF"], lw=2.2), Line2D([], [], color=COLORS["PPG"], lw=1.3),
         Line2D([], [], color=COLORS["PEAK"], lw=1.6, marker="o", ms=6, mec="white"),
         Line2D([], [], color=COLORS["TROUGH"], lw=1.2, marker="o", ms=4.5, mec="white"),
         Line2D([], [], color=COLORS["BASE"], lw=1.6, ls=(0, (5, 3))),
         Line2D([], [], color=COLORS["BR"], lw=0.9, ls=(0, (3, 3)))]
    lab = ["Reference breathing", "PPG", "Pulse peaks (RIIV)", "Troughs (RIAV = peak - trough)", "PPG baseline",
           "GT breath"]
    h2 = [Line2D([], [], color=COLORS[k], lw=1.8) for k in MODS] + [Line2D([], [], color=COLORS["MEAN"], lw=1.8,
                                                                          ls=(0, (6, 2)))]
    lab2 = ["RIIV (intensity)", "RIAV (amplitude)", f"RIFV (pulse rate){' x -1' if sign['RIFV'] < 0 else ''}",
            "Mean of the 3 components"]
    fig.legend(h, lab, loc="lower left", bbox_to_anchor=(L, 0.02), ncol=6, frameon=False, fontsize=8.2)
    fig.legend(h2, lab2, loc="lower left", bbox_to_anchor=(L, -0.002), ncol=4, frameon=False, fontsize=8.2)
    fig.text(L, 0.985, f"{ctx['dataset']} {ctx['subject']}{ctx['rec']} · {a:.0f}-{b:.0f} s: the PPG, its respiratory "
             f"components (RIIV, RIAV, RIFV) and the breathing" + (" · MAE issue subject" if ctx["issue"] else ""),
             fontsize=12, color=INK, va="top")
    fig.text(L, 0.962, f"Components from the used pulse beats, 4 Hz, band-passed {band[0]:.2f}-{band[1]:.2f} Hz like "
             "the reference; (c)-(d) z-scored in the window; PLV = phase locking value; dashed vertical lines = GT breaths",
             fontsize=8.0, color=MUTED, va="top")
    row = {"Start_Sec": a, "End_Sec": b, "Beats_Used": int(used.sum()), "Beats_Rejected": int(rej.sum()),
           "GT_Breaths": nb, "GT_Rate": nb * 60.0 / (b - a), "Ref_FFT_Rate": ref_rate, "RR_Components_Mean": rr_mean,
           "Error_vs_Ref": rr_mean - ref_rate,
           "Spread_of_Component_Rates": float(np.nanmax(list(rates.values())) - np.nanmin(list(rates.values())))}
    for (x, y), (p, d) in pair.items():
        row[f"PLV_{x}_{y}"], row[f"Phase_{x}_{y}_deg"] = p, d
    for k in MODS:
        row[f"{k}_Rate"] = rates[k]
        row[f"PLV_{k}_Ref"], row[f"Phase_{k}_Ref_deg"] = withref[k]
    row["PLV_Components_Mean"] = float(np.nanmean([p for p, _ in pair.values()]))
    row["PLV_Ref_Mean"] = float(np.nanmean([p for p, _ in withref.values()]))
    return fig, row


def summary_figure(path: Path, dataset: str, subject: str, rows: pd.DataFrame, rate_band: Tuple[float, float]) -> None:
    """PLV and FFT rates of every window of a subject."""
    fig, (a1, a2) = plt.subplots(2, 1, figsize=(12.5, 7.2), sharex=True)
    fig.subplots_adjust(left=0.07, right=0.8, top=0.86, bottom=0.1, hspace=0.32)
    label = [f"{int(s)}" + (f" r{int(g)}" if rows.Segment_ID.nunique() > 1 else "") for s, g in
             zip(rows.Start_Sec + 15, rows.Segment_ID)]
    xc = np.arange(len(rows))
    for ax in (a1, a2):
        style(ax)
    a1.axhspan(0, CHANCE_PLV, color="#f1f0ec", lw=0, zorder=0)
    a1.plot(xc, rows.PLV_Components_Mean, color=COLORS["MEAN"], lw=2.2, marker="o", ms=6, mec="white",
            label="among RIIV, RIAV, RIFV\n(mean of 3 pairs)", zorder=4)
    for k in MODS:
        a1.plot(xc, rows[f"PLV_{k}_Ref"], color=COLORS[k], lw=1.6, marker="o", ms=5, mec="white",
                label=f"{k} with the reference", zorder=3)
    a1.set_ylim(0, 1.05)
    a1.set_ylabel("PLV", fontsize=9)
    a1.set_title(f"(a) Phase locking per {WINDOW_SEC:g} s window (shaded: up to the chance level, 95th percentile "
                 f"{CHANCE_PLV:g})", loc="left", fontsize=10.5)
    a1.legend(loc="upper left", bbox_to_anchor=(1.01, 1.0), frameon=False, fontsize=8.5)
    a2.plot(xc, rows.Ref_FFT_Rate, color=COLORS["REF"], lw=2.4, marker="s", ms=6, mec="white", label="reference (FFT)",
            zorder=4)
    a2.plot(xc, rows.GT_Rate, color=COLORS["REF"], lw=1.0, ls=(0, (3, 3)), label="GT breaths per minute", zorder=3)
    for k in MODS:
        a2.plot(xc, rows[f"{k}_Rate"], color=COLORS[k], lw=1.6, marker="o", ms=5, mec="white", label=k, zorder=3)
    a2.set_ylabel("breaths/min", fontsize=9)
    a2.set_xlabel("window centre (s)", fontsize=9)
    a2.set_title(f"(b) FFT rate per {WINDOW_SEC:g} s window", loc="left", fontsize=10.5)
    a2.legend(loc="upper left", bbox_to_anchor=(1.01, 1.0), frameon=False, fontsize=8.5)
    step = max(1, int(math.ceil(len(xc) / 24)))
    a2.set_xticks(xc[::step])
    a2.set_xticklabels(label[::step], fontsize=8)
    fig.text(0.07, 0.96, f"{dataset} {subject}: every {WINDOW_SEC:g} s window", fontsize=12, color=INK, va="top")
    fig.text(0.07, 0.925, "PLV = |mean exp(i x phase difference)| of the Hilbert phases (band-passed like the reference); "
             f"rates = FFT peak in {rate_band[0]:g}-{rate_band[1]:g} /min", fontsize=8.2, color=MUTED, va="top")
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=160)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Part 2: smart fusion
# ---------------------------------------------------------------------------

def smart_fusion(comp: Dict[str, np.ndarray], grid: np.ndarray, times: np.ndarray, a: float, b: float,
                 rate_band: Tuple[float, float], sd_threshold: float) -> dict:
    """Component rates of one minute and the smart fusion rate (mean when their SD is below the threshold)."""
    out = {f"{k}_Rate": math.nan for k in MODS}
    out.update({"SF_Beats": int(np.sum((times >= a) & (times < b))), "SF_SD": math.nan, "SF_Accepted": False,
                "SF_Rate": math.nan, "SF_Mean_All": math.nan, "SF_Reason": ""})
    idx = np.flatnonzero((grid >= a) & (grid < b))
    if out["SF_Beats"] < MIN_WINDOW_BEATS:
        out["SF_Reason"] = "too few beats"
        return out
    need = MIN_WINDOW_COVER * (b - a) * RESAMPLE_HZ
    if any(np.isfinite(comp[k][idx]).sum() < need for k in MODS):
        out["SF_Reason"] = "series cover < 90%"
        return out
    rates = [spectrum_rate(comp[k][idx], RESAMPLE_HZ, rate_band) for k in MODS]
    for k, r in zip(MODS, rates):
        out[f"{k}_Rate"] = r
    if not all(np.isfinite(rates)):
        out["SF_Reason"] = "rate undefined"
        return out
    sd = float(np.std(rates, ddof=1))
    out["SF_SD"], out["SF_Mean_All"] = sd, float(np.mean(rates))
    if sd < sd_threshold:
        out["SF_Accepted"], out["SF_Rate"] = True, float(np.mean(rates))
    else:
        out["SF_Reason"] = f"SD >= {sd_threshold:g}"
    return out


def mean_sd(x: pd.Series) -> Tuple[float, float]:
    x = pd.to_numeric(x, errors="coerce").dropna()
    return (float(x.mean()), float(x.std(ddof=1))) if len(x) else (math.nan, math.nan)


def comparison_figure(subjects: pd.DataFrame, out_root: Path) -> None:
    """Subject MAE per dataset: smart fusion, RRWaveNet on the same minutes, RRWaveNet on every scored minute."""
    groups = [d for d in DATASETS if d in set(subjects.Dataset)] + ["All"]
    methods = (("SF_MAE", "Smart fusion\n(accepted minutes)", "SF"),
               ("RRWaveNet_MAE_SF_Minutes", "RRWaveNet\n(same minutes)", "RRW_SF"),
               ("RRWaveNet_MAE", "RRWaveNet\n(every scored minute)", "RRW_ALL"))
    fig, axes = plt.subplots(1, len(groups), figsize=(max(10.0, 4.2 * len(groups)), 5.9), sharey=True)
    axes = np.atleast_1d(axes)
    rng = np.random.default_rng(0)
    top = 0.0
    for ax, g in zip(axes, groups):
        d = subjects if g == "All" else subjects[subjects.Dataset == g]
        style(ax)
        ax.grid(False, axis="x")
        for i, (col, _, ckey) in enumerate(methods):
            v = pd.to_numeric(d[col], errors="coerce").dropna().to_numpy()
            if not len(v):
                continue
            ax.boxplot([v], positions=[i], widths=0.55, showfliers=False, patch_artist=True,
                       boxprops=dict(facecolor="white", edgecolor=METHOD_COLORS[ckey], lw=1.6),
                       medianprops=dict(color=METHOD_COLORS[ckey], lw=2.2), whiskerprops=dict(color=METHOD_COLORS[ckey]),
                       capprops=dict(color=METHOD_COLORS[ckey]))
            ax.scatter(i + rng.uniform(-0.17, 0.17, len(v)), v, s=14, color=METHOD_COLORS[ckey], alpha=0.75,
                       edgecolor="white", linewidth=0.4, zorder=3)
            m, s = mean_sd(pd.Series(v))
            ax.text(i, -0.035, f"{m:.2f} ± {s:.2f}\nn = {len(v)}", transform=ax.get_xaxis_transform(), ha="center",
                    va="top", fontsize=8.4, color=INK, linespacing=1.3)
            top = max(top, float(np.max(v)))
        cov = d.SF_Accepted_Minutes.sum() / max(1, d.Scored_Minutes.sum())
        ax.set_title(f"{g} (n = {len(d)})", loc="left", fontsize=11.5, pad=18)
        ax.text(0, 1.01, f"smart fusion estimated {cov:.0%} of the scored minutes", transform=ax.transAxes, fontsize=8.2,
                color=MUTED, va="bottom")
        ax.set_xticks(range(len(methods)))
        ax.set_xticklabels([m[1] for m in methods], fontsize=8.6)
        ax.tick_params(axis="x", length=0, pad=36)
    axes[0].set_ylabel("Subject MAE (breaths/min)", fontsize=10)
    for ax in axes:
        ax.set_ylim(0, max(1.0, top) * 1.05)
    fig.text(0.01, 0.985, "Smart fusion (RIIV, RIAV, RIFV) vs RRWaveNet: subject MAE", fontsize=13, color=INK, va="top")
    fig.text(0.01, 0.945, "Smart fusion: mean of the three FFT rates of the minute when their SD < 4 breaths/min (Karlen et "
             "al. 2013); RRWaveNet: final ensemble; GT and minutes of the final results\nBelow each box: mean ± SD of the "
             "subject MAEs and the subjects with an MAE (a subject without accepted minutes has no smart fusion MAE)",
             fontsize=8.4, color=MUTED, va="top")
    fig.subplots_adjust(left=0.06, right=0.99, top=0.79, bottom=0.24, wspace=0.08)
    for ext in ("png", "pdf"):
        fig.savefig(out_root / f"SmartFusion_vs_RRWaveNet_MAE.{ext}", dpi=200)
    plt.close(fig)


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
    spans = []
    for k in range(int(math.floor(t_last / window_sec)) + 1):
        a, b = k * window_sec, (k + 1) * window_sec
        if min(b, t_last) - max(a, t_first) >= MIN_WINDOW_COVER * window_sec:
            spans.append((a, b))
    return spans


def process_dataset(args, dataset: str, chosen: Dict[str, List[str]], out_root: Path, store: dict) -> None:
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
    for subject in sorted(data):
        if dataset in chosen and not any(same_subject(subject, s) for s in chosen[dataset]):
            continue
        tag = mod.safe_tag(subject)
        issue = any(same_subject(subject, x) for x in ISSUE_SUBJECTS[dataset])
        table, table_path = rrwavenet_table(args, mod, spec, dataset, subject)
        if table is None:
            store["missing"].append({"Dataset": dataset, "Subject": subject, "Missing": table_path})
        references = data[subject]["references"]
        multi = len(references) > 1
        pdf = None
        window_rows, minute_rows = [], []
        try:
            for segment_id, ref in sorted(references.items()):
                profile = ref["profile"]
                beats = detect_beats(mod, ref["model_input"], fs, tuple(profile["ppg_hr_range_bpm"]),
                                     ref.get("ppg_artifact_mask"))
                if "times" not in beats or len(beats["times"]) < MIN_WINDOW_BEATS:
                    print(f"  {subject} recording {segment_id}: too few usable beats "
                          f"({beats.get('n_detected', 0)} detected)", flush=True)
                    continue
                band = tuple(float(v) for v in ref["band_hz"])
                rate_band = tuple(float(v) for v in profile["rr_band_bpm"])
                grid, comp = component_series(beats, band)
                rs = np.asarray(ref["signal"], dtype=np.float64)
                t64 = np.arange(len(rs)) / fs
                ok_ref = np.isfinite(rs)
                refg = np.interp(grid, t64[ok_ref], rs[ok_ref], left=np.nan, right=np.nan)
                phase = {k: analytic_phase(comp[k]) for k in MODS}
                phase["REF"] = analytic_phase(refg)
                breaths = gt_breaths(dataset, mod, ref, ann1_all.get(subject))
                times = beats["times"]
                # Part 1: every 30 s window
                if not args.no_figures:
                    sos = butter(3, band[1] / (0.5 * fs), btype="low", output="sos")
                    ctx = {"fs": fs, "grid": grid, "comp": comp, "refg": refg, "phase": phase, "ref_signal": rs,
                           "smooth": beats["smooth"], "baseline": sosfiltfilt(sos, beats["smooth"]),
                           "beats": beats["beats"], "troughs": beats["troughs"], "ok": beats["ok"],
                           "rate_band": rate_band, "band": band, "breaths_sec": breaths / fs,
                           "modality": MODALITY[dataset], "dataset": dataset, "subject": subject,
                           "rec": f" · recording {segment_id}" if multi else "", "issue": issue}
                    for a, b in window_spans(float(times[0]), float(times[-1]), WINDOW_SEC):
                        if int(((times >= a) & (times < b)).sum()) < MIN_WINDOW_BEATS:
                            continue
                        if np.flatnonzero((grid >= a) & (grid < b)).size < 8:
                            continue
                        fig, row = window_page(ctx, a, b)
                        if pdf is None:
                            folder = out_root / "Figures" / dataset
                            folder.mkdir(parents=True, exist_ok=True)
                            pdf = PdfPages(folder / f"{dataset}_{tag}_all_windows.pdf")
                        pdf.savefig(fig)
                        if args.window_png:
                            png = (out_root / "Figures" / dataset / tag
                                   / f"{dataset}_{tag}{f'_rec{segment_id}' if multi else ''}_{a:03.0f}-{b:03.0f}s.png")
                            png.parent.mkdir(parents=True, exist_ok=True)
                            fig.savefig(png, dpi=160)
                        plt.close(fig)
                        row.update({"Dataset": dataset, "Subject": subject, "Issue_Subject": issue,
                                    "Segment_ID": int(segment_id)})
                        window_rows.append(row)
                # Part 2: the scored minutes
                if table is not None:
                    seg = table[table["Segment_ID"].astype(int) == int(segment_id)]
                    grid_minutes = [(float(r.RR_Window_Start_Sec), float(r.RR_Window_End_Sec), r) for _, r in seg.iterrows()]
                else:
                    length = len(rs) / fs
                    grid_minutes = [(k * 60.0, (k + 1) * 60.0, None) for k in range(int(length // 60))]
                for t0, t1, stored_row in grid_minutes:
                    a, b = int(round(t0 * fs)), int(round(t1 * fs))
                    valid, reason, count = minute_gt(mod, dataset, ref, a, b, breaths, stored_row)
                    sf = smart_fusion(comp, grid, times, t0, t1, rate_band, args.sd_threshold)
                    rrw = float(stored_row["Pred_Wave_FFT_RR"]) if stored_row is not None else math.nan
                    row = {"Dataset": dataset, "Subject": subject, "Issue_Subject": issue, "Segment_ID": int(segment_id),
                           "Start_Sec": t0, "End_Sec": t1, "Scored": valid, "Reason": reason, "GT_Count": count}
                    row.update(sf)
                    row["RRWaveNet_Rate"] = rrw
                    row["AE_SF"] = abs(sf["SF_Rate"] - count) if valid and sf["SF_Accepted"] else math.nan
                    row["AE_SF_NoRejection"] = abs(sf["SF_Mean_All"] - count) if valid else math.nan
                    row["AE_RRWaveNet"] = abs(rrw - count) if valid else math.nan
                    minute_rows.append(row)
        finally:
            if pdf is not None:
                pdf.close()
        if window_rows:
            w = pd.DataFrame(window_rows)
            summary_figure(out_root / "Figures" / dataset / f"{dataset}_{tag}_windows_summary.png", dataset, subject, w,
                           rate_band)
            store["windows"].append(w)
        if minute_rows:
            m = pd.DataFrame(minute_rows)
            store["minutes"].append(m)
            sc = m[m.Scored]
            acc = sc[sc.SF_Accepted]
            srow = {"Dataset": dataset, "Subject": subject, "Issue_Subject": issue, "Has_RRWaveNet": table is not None,
                    "RRWaveNet_Table": table_path if table is not None else "", "Scored_Minutes": int(len(sc)),
                    "SF_Accepted_Minutes": int(len(acc)),
                    "SF_Coverage": len(acc) / len(sc) if len(sc) else math.nan,
                    "SF_MAE": float(acc.AE_SF.mean()) if len(acc) else math.nan,
                    "RRWaveNet_MAE_SF_Minutes": float(acc.AE_RRWaveNet.mean()) if len(acc) else math.nan,
                    "RRWaveNet_MAE": float(sc.AE_RRWaveNet.mean()) if len(sc) else math.nan,
                    "SF_NoRejection_MAE": float(sc.AE_SF_NoRejection.mean()) if sc.AE_SF_NoRejection.notna().any()
                    else math.nan,
                    "SF_NoRejection_Minutes": int(sc.AE_SF_NoRejection.notna().sum())}
            srow["SF_minus_RRWaveNet_same_minutes"] = srow["SF_MAE"] - srow["RRWaveNet_MAE_SF_Minutes"]
            store["subjects"].append(srow)
            if not len(sc):
                store["missing"].append({"Dataset": dataset, "Subject": subject, "Missing": "no scored minute"})
            print(f"  {subject}: {len(window_rows)} windows drawn; {len(sc)} scored minutes, smart fusion "
                  f"{len(acc)} ({srow['SF_Coverage']:.0%}); MAE SF {srow['SF_MAE']:.2f}, RRWaveNet same minutes "
                  f"{srow['RRWaveNet_MAE_SF_Minutes']:.2f}, every minute {srow['RRWaveNet_MAE']:.2f}"
                  f"{' [MAE issue]' if issue else ''}", flush=True)
    del data


def dataset_rows(subjects: pd.DataFrame, minutes: pd.DataFrame) -> pd.DataFrame:
    rows = []
    groups = [d for d in DATASETS if d in set(subjects.Dataset)] + ["All"]
    for g in groups:
        s = subjects if g == "All" else subjects[subjects.Dataset == g]
        m = minutes[minutes.Subject.isin(s.Subject) & minutes.Dataset.isin(s.Dataset)]
        m = m.merge(s[["Dataset", "Subject"]], on=["Dataset", "Subject"])
        sc = m[m.Scored]
        acc = sc[sc.SF_Accepted]
        row = {"Dataset": g, "Subjects": int(len(s)), "Subjects_with_SF": int(s.SF_MAE.notna().sum()),
               "Scored_Minutes": int(len(sc)), "SF_Accepted_Minutes": int(len(acc)),
               "SF_Coverage": len(acc) / len(sc) if len(sc) else math.nan}
        for col in ("SF_MAE", "RRWaveNet_MAE_SF_Minutes", "RRWaveNet_MAE", "SF_NoRejection_MAE"):
            row[f"{col}_Mean"], row[f"{col}_SD"] = mean_sd(s[col])
        row["Pooled_SF_MAE"] = float(acc.AE_SF.mean()) if len(acc) else math.nan
        row["Pooled_RRWaveNet_MAE_SF_Minutes"] = float(acc.AE_RRWaveNet.mean()) if len(acc) else math.nan
        row["Pooled_RRWaveNet_MAE"] = float(sc.AE_RRWaveNet.mean()) if len(sc) else math.nan
        row["Pooled_SF_NoRejection_MAE"] = float(sc.AE_SF_NoRejection.mean()) if len(sc) else math.nan
        pair = s[["SF_MAE", "RRWaveNet_MAE_SF_Minutes"]].dropna()
        try:
            row["Wilcoxon_p_SF_vs_RRWaveNet_same_minutes"] = (float(wilcoxon(pair.SF_MAE, pair.RRWaveNet_MAE_SF_Minutes).pvalue)
                                                             if len(pair) >= 5 else math.nan)
        except ValueError:
            row["Wilcoxon_p_SF_vs_RRWaveNet_same_minutes"] = math.nan
        row["Subjects_SF_Better"] = int((pair.SF_MAE < pair.RRWaveNet_MAE_SF_Minutes).sum())
        row["Subjects_RRWaveNet_Better"] = int((pair.SF_MAE > pair.RRWaveNet_MAE_SF_Minutes).sum())
        rows.append(row)
    return pd.DataFrame(rows)


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="PPG respiratory components of every 30 s window of every subject, and the "
                                            "smart fusion MAE compared with RRWaveNet")
    p.add_argument("--datasets", nargs="+", choices=list(DATASETS), default=list(DATASETS))
    p.add_argument("--subjects", nargs="+", default=None, help="Dataset:Subject (default: every subject)")
    p.add_argument("--no-figures", action="store_true", help="part 2 only (no window figures)")
    p.add_argument("--window-png", action="store_true", help="also save every window as PNG")
    p.add_argument("--keep-issue-subjects", action="store_true",
                   help="keep the MAE issue subjects in the smart fusion / RRWaveNet comparison")
    p.add_argument("--sd-threshold", type=float, default=SD_THRESHOLD, help="smart fusion SD limit (default 4 /min)")
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
    store = {"windows": [], "minutes": [], "subjects": [], "missing": []}
    for dataset in datasets:
        process_dataset(args, dataset, chosen, out_root, store)
    windows = pd.concat(store["windows"], ignore_index=True) if store["windows"] else pd.DataFrame()
    minutes = pd.concat(store["minutes"], ignore_index=True) if store["minutes"] else pd.DataFrame()
    subjects_all = pd.DataFrame(store["subjects"])
    compared = subjects_all
    if not subjects_all.empty:
        compared = subjects_all if args.keep_issue_subjects else subjects_all[~subjects_all.Issue_Subject.astype(bool)]
        compared = compared[(compared.Scored_Minutes > 0) & compared.Has_RRWaveNet.astype(bool)]
    if compared.empty:
        datasets_table = pd.DataFrame()
        print("\nNo subject with scored minutes and stored RRWaveNet results: no comparison")
    else:
        datasets_table = dataset_rows(compared, minutes)
        comparison_figure(compared, out_root)
        print("\nSubject MAE (mean ± SD), breaths/min")
        for r in datasets_table.itertuples():
            print(f"  {r.Dataset:<9} n {r.Subjects:3d}: smart fusion {r.SF_MAE_Mean:.2f} ± {r.SF_MAE_SD:.2f} "
                  f"(coverage {r.SF_Coverage:.0%}) | RRWaveNet same minutes {r.RRWaveNet_MAE_SF_Minutes_Mean:.2f} ± "
                  f"{r.RRWaveNet_MAE_SF_Minutes_SD:.2f} | RRWaveNet every minute {r.RRWaveNet_MAE_Mean:.2f} ± "
                  f"{r.RRWaveNet_MAE_SD:.2f} | SF without rejection {r.SF_NoRejection_MAE_Mean:.2f} ± "
                  f"{r.SF_NoRejection_MAE_SD:.2f} | Wilcoxon p {r.Wilcoxon_p_SF_vs_RRWaveNet_same_minutes:.3g}")
    notes = pd.DataFrame([
        ("Components", "RIIV (PPG at the pulse peak), RIAV (peak - trough), RIFV (60 / peak-to-peak interval) of the used "
                       "beats; beats of the 0.5-5 Hz pulse band at least 0.6 x the spectral beat interval apart; series "
                       "at 4 Hz, band-passed with the recording's respiratory band, empty between beats > 3 s apart"),
        ("Smart fusion", f"per scored minute: the FFT rate of RIIV, RIAV and RIFV; SD (N - 1) < {args.sd_threshold:g} "
                         "breaths/min -> their mean, otherwise no estimate (Karlen W et al. IEEE TBME 2013;60(7):1946-53); "
                         "the minute needs >= 10 used beats and every series covering 90% of it; Karlen et al. used 32 s "
                         "windows, here the scored minutes of the final results"),
        ("RRWaveNet", "final results: the ensemble of every LOSOCV fold (CapnoBase 261007 full; BIDMC / STEAM2 261013, "
                      "weight decay selected in the fold), stored rate of the minute (Pred_Wave_FFT_RR)"),
        ("GT", "as the final results: CapnoBase breaths detected on the capnogram (RRest agreement rule); BIDMC ann1 "
               "(expert-label rule of 261013); STEAM2 the stored GT; GT = breath count of the minute"),
        ("MAE", "per subject, mean |estimate - GT| over the minutes: SF_MAE on the minutes smart fusion estimated; "
                "RRWaveNet_MAE_SF_Minutes on the same minutes; RRWaveNet_MAE on every scored minute; SF_NoRejection_MAE = "
                "mean of the three rates on every scored minute; per dataset the mean and SD of the subject MAEs, the "
                "pooled MAE over the minutes, and a Wilcoxon signed-rank test (SF_MAE vs RRWaveNet_MAE_SF_Minutes)"),
        ("Subjects", "figures: every subject; comparison: MAE issue subjects (CapnoBase 0115, BIDMC 27, STEAM2 12_YJJ, "
                     "22_LMS, 26_CKK) " + ("kept" if args.keep_issue_subjects else "left out as in the final results")),
        ("Windows", f"figure windows of {WINDOW_SEC:g} s from the start of the recording, covered at least "
                    f"{MIN_WINDOW_COVER:.0%} by the series, at least {MIN_WINDOW_BEATS} used beats; PLV = |mean exp(i dphi)| "
                    "of the Hilbert phases (analytic signals of the whole recording, cut into windows); chance level of "
                    f"unrelated 30 s windows: 95th percentile about {CHANCE_PLV:g}"),
        ("References", "Karlen W et al. IEEE TBME 2013;60(7):1946-53; Charlton PH et al. Physiol Meas 2016;37(4):610-26; "
                       "Lachaux JP et al. Hum Brain Mapp 1999;8(4):194-208"),
    ], columns=["Item", "Description"])
    path = out_root / f"{NAME}.xlsx"
    with pd.ExcelWriter(path, engine="openpyxl") as writer:
        datasets_table.round(4).to_excel(writer, sheet_name="Datasets", index=False)
        (subjects_all.round(4) if not subjects_all.empty else subjects_all).to_excel(writer, sheet_name="Subjects",
                                                                                     index=False)
        (minutes.round(4) if not minutes.empty else minutes).to_excel(writer, sheet_name="Minutes", index=False)
        (windows.round(4) if not windows.empty else windows).to_excel(writer, sheet_name="Windows", index=False)
        pd.DataFrame(store["missing"], columns=["Dataset", "Subject", "Missing"]).to_excel(writer, sheet_name="Missing",
                                                                                            index=False)
        notes.to_excel(writer, sheet_name="Notes", index=False)
    print(f"\nFigures and table under {out_root}")


if __name__ == "__main__":
    main()
