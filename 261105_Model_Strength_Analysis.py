r"""261105_Model_Strength_Analysis: where each model - RespDiff, the deep
corr-encoder, RRWaveNet - reproduces the breathing best, over every subject of
the stored results (nothing is trained), in non-overlapping 30 s and 60 s
windows: which model wins which windows and subjects, which breathing and PPG
conditions go with the wins of each model (clusters, rules, correlations), and
what RRWaveNet does better or worse than the other two.

Stages (--stage, default all)
-----------------------------
extract  per recording: GT, PPG and the model outputs on one 30 Hz axis, read
         as 261104_Waveform_Overlay reads them (its readers are imported); every
         model aligned by one rule; per window the scores of every model and
         the features of the GT and the PPG (no model output in the features)
         -> windows_30s.csv, windows_60s.csv, recordings.csv
analyze  from those CSVs only, so it can be rerun with other settings without
         reading the results again
         -> model_strength.xlsx, profiles_{size}s.png/pdf,
            clusters_{size}s.png/pdf, summary.txt

Results read (test outputs only: every subject was held out of the training of
its own fold, so every window is an out-of-subject prediction)
------------
  RespDiff           {--respdiff-root}\respdiff\{dataset}\Predictions\{subject}_ddpm50.npz
  Deep corr-encoder  {--correncoder-root}\{dataset}\Predictions\{subject}_outputs.npz
  RRWaveNet          the ensemble (--rrwavenet-member ensemble, default; or m0 /
                     m1 / m2) of CapnoBase: 261007 ({--rrwavenet-261007-root}\full\...)
                     and BIDMC / STEAM2: 261013 (the weight decay selected in
                     every fold), as 261104; or every dataset from one 261102
                     folder (--rrwavenet-unified-root, the layout of 261013)
A subject is compared when every model of --models has a result for it; a
window when the GT and every model have output over all of it.

Alignment (the same rule for every model, as in the overlay figures)
---------
RRWaveNet: the sign / lag its target was given in training undone (Data_Quality
table, as 261104) and its output low-pass filtered at 1 Hz (Butterworth order
8, zero phase: the filter RespDiff applies to its own output). Then every model
shifted by at most half the median breath period and, if needed, inverted,
once per recording, to its largest Pearson r with the GT.

Scores of every model in every window
-------------------------------------
  Rate        FFT peak of the window in the rate band (CapnoBase 3-60, BIDMC
              and STEAM2 6-30 breaths/min)
  AE          |Rate - GT rate|; GT rate from the breaths in the window
              (CapnoBase: expert labels; BIDMC: ann1; STEAM2 has none: the
              cycles of the GT found by 261030's detector)
  PCC         Pearson r with the GT in the window
  Cycles      complete breathing cycles inside the window (261030's detector on
              every continuous stretch of >= 30 s with all waveforms);
              Cycle_Error = model cycles - GT cycles
  TiTtot      median inspiratory share of the cycles inside the window;
              TiTtot_AE against the GT cycles without clipped samples
  Score       PCC - 0.1 x min(AE, 5) (--ae-weight, --ae-cap): rate and shape
              in one number (an AE of 1 breath/min costs 0.1 of PCC)
  Advantage   the model's Score minus the best Score of the other models
Winner of a window: the model with the highest Score; "Tie" when the second is
within 0.05 (--tie-margin). Winner of a subject: from the mean Score of its
windows, with the same margin.

Features of every window (GT and PPG only)
------------------------------------------
  GT_RR             breathing rate (breaths/min)
  GT_Breath_CV      CV of the breath intervals (irregularity)
  GT_Rate_Change    |rate of the 2nd half - rate of the 1st half| (breaths/min)
  GT_TiTtot         median Ti/Ttot of the GT cycles without clipped samples
  GT_IE_Asymmetry   |GT_TiTtot - 0.5|
  GT_Harmonic2      power at 2 x the breathing rate / power at the rate
                    (non-sinusoidal, e.g. square capnograms)
  GT_Clipped_Share  share of clipped GT samples (impedance at its limits)
  PPG_HR            heart rate (beats/min)
  PPG_Pulse_CV      CV of the pulse intervals (arrhythmia, artifacts)
  PPG_Coupling      largest |r| of the PPG baseline (RIIV, 0.05-1 Hz) and the
                    GT (0.05-1 Hz) within half a breath of lag: how strongly
                    the PPG carries the breathing
  PPG_Coupling_Lag  that lag in breath cycles (absolute)
  RIIV / RIAV / RIFV_Rate_Error
                    |rate of the baseline / amplitude / frequency modulation
                    of the PPG - GT rate| (breaths/min): which modulation
                    carries the breathing
Pooled datasets are compared on within-dataset percentiles of every feature
(0 = lowest of its dataset, 1 = highest), because the GT differs between
datasets (capnogram, impedance, belt).

Analysis (every window size, pooled and per dataset)
--------
* Overview: mean AE, PCC, Score, Ti/Ttot AE of every model; share of windows
  and subjects each model wins.
* Associations: Spearman rho of every model's Advantage with every feature,
  between subjects (subject means; p and Benjamini-Hochberg q) and between
  windows (descriptive: windows of one subject are not independent).
* Profiles: AE, PCC and win share of every model in the within-dataset
  quintiles of every feature.
* Clusters: k-means (k = --clusters, default 3) of the windows and, apart, of
  the subjects on the feature percentiles (--cluster-features); per cluster
  its features, every model's scores, its best model; silhouette for k 2-6.
* Clusters of every model's wins: k-means of the windows each model wins: the
  kinds of windows where it is best, and their subjects.
* Rules: a decision tree (depth 3) predicting the winner from the feature
  percentiles, cross-validated by subject (balanced accuracy against chance).
* summary.txt: the main lines of all of this.

Run:            python 261105_Model_Strength_Analysis.py
Analysis only:  python 261105_Model_Strength_Analysis.py --stage analyze
Two models:     --models "Deep corr-encoder" RRWaveNet
Result folders: --respdiff-root, --correncoder-root, --rrwavenet-261007-root,
                --rrwavenet-261013-root, --rrwavenet-unified-root, --output-root
Data files:     --capnobase-path, --bidmc-path, --steam2-ppg-dir, --steam2-rsp-dir
"""

import argparse
import importlib.util
import math
import sys
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.io as sio
from scipy import stats
from scipy.signal import butter, detrend, find_peaks, sosfiltfilt

HERE = Path(__file__).resolve().parent


def load_script(name: str, filename: str):
    spec = importlib.util.spec_from_file_location(name, HERE / filename)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


ov = load_script("overlay_261104", "261104_Waveform_Overlay.py")
m30 = load_script("morphology_261030", "261030_Waveform_Morphology_Analysis.py")
_M07 = None


def capnobase_reader():
    """261007's CapnoBase reader (signals and expert breath labels); loaded
    only when CapnoBase is read."""
    global _M07
    if _M07 is None:
        _M07 = load_script("ensemble_261007", "261007_RRWaveNet_Ensemble_NoEvent.py")
    return _M07


# ---------------------------------------------------------------------------
# Settings
# ---------------------------------------------------------------------------

PROJECT_ROOT = Path(r"D:\PPG2RSP_RRWaveNet_Inspired\261105_Model_Strength_Analysis")
FS = ov.FS  # 30 Hz
MODELS = tuple(ov.MODELS)
RATE_BAND_BPM = {"capnobase": (3.0, 60.0), "bidmc": (6.0, 30.0), "steam2": (6.0, 30.0)}
WINDOW_SIZES = (30, 60)
RESP_BAND_HZ = (0.05, 1.0)
PULSE_BAND_HZ = (0.5, 5.0)
SURROGATE_FS = 4.0
MIN_RUN_SEC = 30.0
MIN_BREATHS = 3
AE_WEIGHT, AE_CAP, TIE_MARGIN = 0.1, 5.0, 0.05
N_CLUSTERS = 3

FEATURES = {
    "GT_RR": "breathing rate (breaths/min)",
    "GT_Breath_CV": "irregularity: CV of the breath intervals",
    "GT_Rate_Change": "rate change in the window (breaths/min)",
    "GT_TiTtot": "inspiration share of a breath (Ti/Ttot)",
    "GT_IE_Asymmetry": "|Ti/Ttot - 0.5|",
    "GT_Harmonic2": "2nd harmonic / fundamental (non-sinusoidal shape)",
    "GT_Clipped_Share": "share of clipped GT samples",
    "PPG_HR": "heart rate (beats/min)",
    "PPG_Pulse_CV": "CV of the pulse intervals",
    "PPG_Coupling": "|r| of PPG baseline and GT (breathing in the PPG)",
    "PPG_Coupling_Lag": "lag of PPG baseline to GT (breath cycles)",
    "RIIV_Rate_Error": "rate error of the PPG baseline modulation",
    "RIAV_Rate_Error": "rate error of the PPG amplitude modulation",
    "RIFV_Rate_Error": "rate error of the PPG pulse-rate modulation",
}
CLUSTER_FEATURES = ("GT_RR", "GT_Breath_CV", "GT_TiTtot", "GT_Harmonic2", "PPG_HR", "PPG_Coupling",
                    "RIIV_Rate_Error")
SHORT = {"RespDiff": "RD", "Deep corr-encoder": "CE", "RRWaveNet": "RR"}


# ---------------------------------------------------------------------------
# Data files: GT, PPG and breath times of every recording
# ---------------------------------------------------------------------------

def read_recordings(dataset_key: str) -> Dict[str, Dict[str, dict]]:
    """Subject -> {recording name: GT, PPG (30 Hz), clipped GT samples, the PPG
    at its own rate and the breath times (s; None without labels)}, named and
    cut as 261104 (and the corr-encoder) name and cut them."""
    cfg = ov.DATASET_CONFIGS[dataset_key]
    modality = ov.MODALITIES[dataset_key]
    out: Dict[str, Dict[str, dict]] = {}

    def add(subject, name, ppg, resp, ppg_fs, resp_fs, breaths):
        ppg30, resp30 = ov.to_fs(ppg, ppg_fs), ov.to_fs(resp, resp_fs)
        length = min(len(ppg30), len(resp30))
        clipped = ov.to_fs_mask(ov.saturated_samples(resp, resp_fs, modality, ov.SATURATION_CONFIG), resp_fs,
                                length, ov.SATURATION_CONFIG["saturation_margin_sec"])
        ppg_raw, _ = ov.fill_missing(ppg)
        out.setdefault(subject, {})[name] = {"gt": resp30[:length], "ppg": ppg30[:length], "clipped": clipped,
                                             "ppg_raw": ppg_raw, "ppg_fs": float(ppg_fs), "breaths": breaths}

    if dataset_key == "capnobase":
        m07 = capnobase_reader()
        csv_dir = Path(cfg["data_path"])
        for case_id in ov.capnobase_case_ids(csv_dir):
            try:
                case = m07.read_capnobase_case(csv_dir, case_id, cfg["default_fs"])
                breaths = case.get("breaths")
                breaths = None if breaths is None else np.sort(np.asarray(breaths, float)) / float(case["co2_fs"])
                add(case_id, case_id, case["ppg"], case["co2"], case["ppg_fs"], case["co2_fs"], breaths)
            except Exception as exc:
                print(f"[CapnoBase {case_id}] not read: {exc}")
    elif dataset_key == "bidmc":
        fs = float(cfg["default_fs"])
        mat = sio.loadmat(cfg["data_path"])
        for index, record in enumerate(mat["data"][0]):
            subject = f"{index + 1:02d}"
            try:
                ppg = np.asarray(record["ppg"][0, 0]["v"]).squeeze()
                ref = record["ref"][0, 0]
                imp = np.asarray(ref["resp_sig"][0, 0]["imp"][0, 0]["v"]).squeeze()
                ann = np.asarray(ref["breaths"][0, 0]["ann1"], float).ravel()  # MATLAB sample numbers
                add(subject, subject, ppg, imp, fs, fs, np.sort(ann - 1.0) / fs)
            except Exception as exc:
                print(f"[BIDMC {subject}] not read: {exc}")
    else:
        for subject, items in ov.steam2_session_files(Path(cfg["ppg_dir"]), Path(cfg["rsp_dir"])).items():
            try:
                for name, ppg_path, rsp_path in items:
                    ppg = ov.read_eeglab_channel(ppg_path, "PPG", cfg["default_fs"])
                    rsp = ov.read_eeglab_channel(rsp_path, "RSP", cfg["default_fs"])
                    pieces, _ = ov.steam2_session_pieces(ppg, rsp)
                    for k, (p, r, p_fs, r_fs) in enumerate(pieces):
                        add(subject, f"{name}_piece{k + 1}", p, r, p_fs, r_fs, None)
            except Exception as exc:
                print(f"[STEAM2 {subject}] not read: {exc}")
    return out


# ---------------------------------------------------------------------------
# Stored results
# ---------------------------------------------------------------------------

def rrwavenet_sources(dataset_key: str, args) -> Tuple[Dict[str, Tuple[Path, str]], Optional[Path]]:
    """Subject -> (npz of the test outputs, run name) and the folder holding
    the Data_Quality table."""
    label = ov.DATASET_CONFIGS[dataset_key]["label"]
    member = args.rrwavenet_member
    window = ov.RRWAVENET_WINDOW

    def output_of(condition: Path, subject: str) -> Optional[Path]:
        if member == "ensemble":
            path = condition / "Ensemble_Predictions" / f"{subject}_ensemble_outputs.npz"
        else:
            path = condition / "Members" / member / "Predictions" / f"{subject}_test_outputs.npz"
        return path if path.exists() and (condition / "Fold_Summaries" / f"{subject}_{member}_summary.json").exists() \
            else None

    files: Dict[str, Tuple[Path, str]] = {}
    if dataset_key == "capnobase" and args.rrwavenet_unified_root is None:
        root = Path(args.rrwavenet_261007_root)
        condition = root / "full" / window / label
        for summary in sorted((condition / "Fold_Summaries").glob(f"*_{member}_summary.json")):
            subject = summary.name[: -len(f"_{member}_summary.json")]
            path = output_of(condition, subject)
            if path is not None:
                files[subject] = (path, f"261007 full {member}")
        return files, root
    root = Path(args.rrwavenet_unified_root or args.rrwavenet_261013_root)
    run = "261102" if args.rrwavenet_unified_root else "261013"
    for selection in sorted((root / "WD_Selection" / window / label).glob("*_selection.json")):
        subject = selection.name[: -len("_selection.json")]
        variant = (ov.read_json(selection) or {}).get("Selected_Variant")
        if variant:
            path = output_of(root / variant / window / label, subject)
            if path is not None:
                files[subject] = (path, f"{run} {variant} {member}")
    return files, root


def rrwavenet_track(segment: dict, stored: Optional[dict], dataset_key: str, gt: np.ndarray, lowpass_hz: float):
    """RRWaveNet on the GT axis with its training sign / lag undone and the
    1 Hz low-pass (as 261104)."""
    lag = stored["lag_sec"] if stored else 0.0
    if stored:
        sign = stored["sign"]
    elif dataset_key != "capnobase":
        r = ov.target_reference_r(ov.Track(segment["target"], ov.RRWAVENET_FS, lag), ov.Track(gt, FS))
        sign = -1.0 if r < 0 else 1.0
    else:
        sign = 1.0
    smoothed = ov.lowpass_runs(sign * segment["output"], ov.RRWAVENET_FS, lowpass_hz, ov.RRWAVENET_LOWPASS_ORDER)
    return ov.Track(smoothed, ov.RRWAVENET_FS, lag).at(np.arange(len(gt)) / FS)


# ---------------------------------------------------------------------------
# Signal helpers
# ---------------------------------------------------------------------------

def bandpass(x: np.ndarray, fs: float, low: float, high: float, order: int = 2) -> np.ndarray:
    high = min(high, 0.45 * fs)
    return sosfiltfilt(butter(order, [low, high], "bandpass", fs=fs, output="sos"), np.asarray(x, float))


def fft_rate(x: np.ndarray, fs: float, band_bpm: Tuple[float, float]) -> float:
    """Frequency (per minute) of the largest FFT peak in the band."""
    x = np.asarray(x, float)
    x = x[np.isfinite(x)]
    if len(x) < 8 or np.std(x) <= 1e-12:
        return math.nan
    n = max(16384, len(x))
    f = np.fft.rfftfreq(n, 1.0 / fs) * 60.0
    p = np.abs(np.fft.rfft(x - np.mean(x), n))
    keep = (f >= band_bpm[0]) & (f <= band_bpm[1])
    return float(f[keep][np.argmax(p[keep])])


def harmonic2(x: np.ndarray, fs: float, rate_bpm: float) -> float:
    """Power at 2 x the breathing rate / power at the rate (+-0.03 Hz)."""
    if not np.isfinite(rate_bpm) or len(x) < 8:
        return math.nan
    y = detrend(np.asarray(x, float)) * np.hanning(len(x))
    n = max(16384, len(y))
    f = np.fft.rfftfreq(n, 1.0 / fs)
    p = np.abs(np.fft.rfft(y, n)) ** 2
    f0 = rate_bpm / 60.0

    def power(c):
        return float(p[(f >= c - 0.03) & (f <= c + 0.03)].sum())

    return power(2 * f0) / max(power(f0), 1e-12)


def shifted(values: np.ndarray, shift: int) -> np.ndarray:
    """values moved later by shift samples (earlier when negative), NaN padded."""
    out = np.full(len(values), np.nan)
    if shift >= 0:
        out[shift:] = values[: len(values) - shift]
    else:
        out[:shift] = values[-shift:]
    return out


def align(gt: np.ndarray, values: np.ndarray, period_sec: float) -> Tuple[np.ndarray, dict]:
    """The model shifted by at most half a breath and, if needed, inverted, to
    its largest Pearson r with the GT over the recording."""
    best = (-np.inf, 1.0, 0)
    reach = int(round(0.5 * period_sec * FS))
    for shift in range(-reach, reach + 1):
        r = ov.pearson(gt, shifted(values, shift))
        if np.isfinite(r):
            for sign in (1.0, -1.0):
                if sign * r > best[0]:
                    best = (sign * r, sign, shift)
    r_before = ov.pearson(gt, values)
    if not np.isfinite(best[0]):
        return np.asarray(values, float), {"r_as_is": r_before, "r_aligned": math.nan, "shift_sec": 0.0, "sign": 1.0}
    return best[1] * shifted(values, best[2]), {"r_as_is": r_before, "r_aligned": best[0],
                                                "shift_sec": best[2] / FS, "sign": best[1]}


def recording_cycles(values: Dict[str, np.ndarray], modality: str) -> Dict[str, List[dict]]:
    """Cycles (start, transition, end in s of the recording) of every waveform,
    by 261030's detector, on the stretches (>= 30 s) where all exist."""
    valid = np.ones(len(values["GT"]), bool)
    for v in values.values():
        valid &= np.isfinite(v)
    out = {key: [] for key in values}
    for s, e in m30.finite_runs(valid, int(MIN_RUN_SEC * FS)):
        for key, v in values.items():
            y = m30.band_filter(v[s:e], FS, modality, m30.CONFIG)
            cycles, _ = m30.detect_cycles(y, FS, modality, m30.CONFIG)
            out[key] += [{k: c[k] + s / FS for k in ("start", "transition", "end")} for c in cycles]
    return out


def ppg_signals(rec: dict, gt: np.ndarray) -> dict:
    """PPG modulations of the whole recording: RIIV and the GT in the same band
    (30 Hz), pulse beats (at the PPG's own rate), RIAV / RIFV on a 4 Hz grid."""
    riiv = bandpass(rec["ppg"], FS, *RESP_BAND_HZ)
    gt_band = bandpass(gt, FS, *RESP_BAND_HZ)
    fs = rec["ppg_fs"]
    pulse = bandpass(rec["ppg_raw"], fs, *PULSE_BAND_HZ)
    peaks, _ = find_peaks(pulse, distance=max(1, int(0.3 * fs)), prominence=0.3 * np.std(pulse))
    beats = peaks / fs
    grid = np.arange(0.0, len(gt) / FS, 1.0 / SURROGATE_FS)
    riav = rifv = np.full(len(grid), np.nan)
    if len(peaks) >= 4:
        inside = (grid >= beats[1]) & (grid <= beats[-1])
        riav = np.where(inside, np.interp(grid, beats, pulse[peaks]), np.nan)
        rifv = np.where(inside, np.interp(grid, beats[1:], np.diff(beats)), np.nan)
    return {"riiv": riiv, "gt_band": gt_band, "beats": beats, "grid": grid, "riav": riav, "rifv": rifv,
            "pulse": pulse, "pulse_fs": fs}


def coupling(a: np.ndarray, b: np.ndarray, reach: int) -> Tuple[float, int]:
    """Largest |r| of a and b (z-scored) within +-reach samples of lag, and that lag."""
    if np.std(a) <= 1e-12 or np.std(b) <= 1e-12:
        return math.nan, 0
    a, b = (a - a.mean()) / a.std(), (b - b.mean()) / b.std()
    n = len(a)
    full = np.correlate(b, a, "full")  # index n - 1 + lag: sum a[i] b[i + lag]
    lags = np.arange(-min(reach, n - 2), min(reach, n - 2) + 1)
    r = full[n - 1 + lags] / (n - np.abs(lags))
    k = int(np.argmax(np.abs(r)))
    return float(abs(r[k])), int(lags[k])


# ---------------------------------------------------------------------------
# Windows of one recording
# ---------------------------------------------------------------------------

def inside(cycles: List[dict], t0: float, t1: float) -> List[dict]:
    return [c for c in cycles if c["start"] >= t0 and c["end"] <= t1]


def ti_ttot(cycles: List[dict]) -> float:
    return float(np.median([(c["transition"] - c["start"]) / (c["end"] - c["start"]) for c in cycles])) \
        if cycles else math.nan


def rate_of(times: np.ndarray) -> float:
    return 60.0 / float(np.mean(np.diff(times))) if len(times) >= 2 else math.nan


def window_features(t0: float, t1: float, gt_seg: np.ndarray, breaths: np.ndarray, gt_cycles: List[dict],
                    clipped: np.ndarray, sig: dict, band: Tuple[float, float]) -> Optional[dict]:
    """GT and PPG features of one window (None: fewer than 3 breaths)."""
    br = breaths[(breaths >= t0) & (breaths < t1)]
    if len(br) < MIN_BREATHS:
        return None
    ibi = np.diff(br)
    rr = rate_of(br)
    mid = 0.5 * (t0 + t1)
    first, second = br[br < mid], br[br >= mid]
    clean = [c for c in inside(gt_cycles, t0, t1) if not c["clipped"]]
    gt_ti = ti_ttot(clean)
    a, b = int(round(t0 * FS)), int(round(t1 * FS))
    row = {"GT_RR": rr, "GT_Breath_CV": float(np.std(ibi) / np.mean(ibi)),
           "GT_Rate_Change": abs(rate_of(second) - rate_of(first)) if len(first) >= 2 and len(second) >= 2
           else math.nan,
           "GT_TiTtot": gt_ti, "GT_IE_Asymmetry": abs(gt_ti - 0.5) if np.isfinite(gt_ti) else math.nan,
           "GT_Harmonic2": harmonic2(gt_seg, FS, rr), "GT_Clipped_Share": float(np.mean(clipped[a:b])),
           "GT_Cycles": len(inside(gt_cycles, t0, t1)), "GT_Clean_Cycles": len(clean)}
    fs = sig["pulse_fs"]
    pa, pb = int(round(t0 * fs)), int(round(t1 * fs))
    row["PPG_HR"] = fft_rate(sig["pulse"][pa:pb], fs, (36.0, 210.0))
    beats = sig["beats"][(sig["beats"] >= t0) & (sig["beats"] < t1)]
    row["PPG_Pulse_CV"] = float(np.std(np.diff(beats)) / np.mean(np.diff(beats))) if len(beats) >= 3 else math.nan
    reach = int(round(0.5 * 60.0 / rr * FS))
    r, lag = coupling(sig["riiv"][a:b], sig["gt_band"][a:b], reach)
    row["PPG_Coupling"] = r
    row["PPG_Coupling_Lag"] = abs(lag) / FS / (60.0 / rr)
    row["RIIV_Rate_Error"] = abs(fft_rate(sig["riiv"][a:b], FS, band) - rr)
    g = (sig["grid"] >= t0) & (sig["grid"] < t1)
    for name in ("riav", "rifv"):
        seg = sig[name][g]
        ok = np.isfinite(seg).mean() >= 0.9
        row[f"{name.upper()}_Rate_Error"] = abs(fft_rate(seg, SURROGATE_FS, band) - rr) if ok else math.nan
    return row


def recording_windows(dataset_key: str, label: str, subject: str, name: str, rec: dict,
                      raw: Dict[str, np.ndarray], sizes: Sequence[int]) -> Tuple[Dict[int, List[dict]], dict]:
    gt = np.asarray(rec["gt"], float)
    modality = ov.MODALITIES[dataset_key]
    band = RATE_BAND_BPM[dataset_key]
    gt_only = recording_cycles({"GT": gt}, modality)["GT"]
    breaths = rec["breaths"]
    source = "labels"
    if breaths is None or len(breaths) < MIN_BREATHS:
        breaths, source = np.array([c["start"] for c in gt_only]), "GT cycles"
    period = float(np.median(np.diff(breaths))) if len(breaths) >= 2 else 60.0 / np.mean(band)
    values = {"GT": gt}
    info = {"Dataset": label, "Subject": subject, "Recording": name, "Minutes": len(gt) / FS / 60.0,
            "Breath_Period_Sec": period, "GT_Rate_Source": source}
    for model, v in raw.items():
        values[model], a = align(gt, v, period)
        info.update({f"{model}_r_as_is": a["r_as_is"], f"{model}_r_aligned": a["r_aligned"],
                     f"{model}_Shift_Sec": a["shift_sec"], f"{model}_Inverted": a["sign"] < 0,
                     f"{model}_Coverage": float(np.mean(np.isfinite(v)))})
    cycles = recording_cycles(values, modality)
    clipped = np.asarray(rec["clipped"], bool)
    for c in cycles["GT"]:
        c["clipped"] = bool(clipped[int(c["start"] * FS):int(math.ceil(c["end"] * FS)) + 1].any())
    sig = ppg_signals(rec, gt)
    rows: Dict[int, List[dict]] = {size: [] for size in sizes}
    for size in sizes:
        n = int(size * FS)
        skipped = {"no output": 0, "few breaths": 0}
        for a in range(0, len(gt) - n + 1, n):
            t0, t1 = a / FS, (a + n) / FS
            segs = {key: v[a:a + n] for key, v in values.items()}
            if not all(np.isfinite(s).all() for s in segs.values()):
                skipped["no output"] += 1
                continue
            feats = window_features(t0, t1, segs["GT"], breaths, cycles["GT"], clipped, sig, band)
            if feats is None:
                skipped["few breaths"] += 1
                continue
            row = {"Dataset": label, "Subject": subject, "Recording": name, "Size_Sec": size, "Start_Sec": t0, **feats}
            for model in raw:
                rate = fft_rate(segs[model], FS, band)
                inw = inside(cycles[model], t0, t1)
                ti = ti_ttot(inw)
                row.update({f"{model}_Rate": rate, f"{model}_AE": abs(rate - feats["GT_RR"]),
                            f"{model}_PCC": ov.pearson(segs["GT"], segs[model]), f"{model}_Cycles": len(inw),
                            f"{model}_Cycle_Error": len(inw) - feats["GT_Cycles"], f"{model}_TiTtot": ti,
                            f"{model}_TiTtot_AE": abs(ti - feats["GT_TiTtot"])})
            rows[size].append(row)
        info[f"Windows_{size}s"] = len(rows[size])
        info[f"Skipped_{size}s"] = ", ".join(f"{k} {v}" for k, v in skipped.items() if v)
    return rows, info


# ---------------------------------------------------------------------------
# Stage 1: extract
# ---------------------------------------------------------------------------

def extract(args) -> None:
    out = Path(args.output_root)
    out.mkdir(parents=True, exist_ok=True)
    windows: Dict[int, List[dict]] = {size: [] for size in args.sizes}
    infos = []
    for dataset_key in [k for k in ov.DATASET_ORDER if k in args.datasets]:
        label = ov.DATASET_CONFIGS[dataset_key]["label"]
        corr = ov.load_correncoder(Path(args.correncoder_root), label) if "Deep corr-encoder" in args.models else {}
        respdiff = ov.load_respdiff(Path(args.respdiff_root), label, args.respdiff_sampling) \
            if "RespDiff" in args.models else {}
        rr_files, rr_root = rrwavenet_sources(dataset_key, args) if "RRWaveNet" in args.models else ({}, None)
        rr_alignment = ov.rrwavenet_alignment_table(rr_root, label) if rr_root else {}
        found = {"RespDiff": set(respdiff), "Deep corr-encoder": set(corr), "RRWaveNet": set(rr_files)}
        print(f"{label}: results - " + ", ".join(f"{m} {len(found[m])}" for m in args.models) + " subjects",
              flush=True)
        candidates = set.intersection(*[found[m] for m in args.models])
        if args.subjects:
            candidates &= {str(s) for s in args.subjects}
        if not candidates:
            print(f"  no subject with a result of every model ({', '.join(args.models)})")
            continue
        print("  reading the data files ...", flush=True)
        recordings = read_recordings(dataset_key)
        for subject in sorted(candidates):
            if subject not in recordings:
                infos.append({"Dataset": label, "Subject": subject, "Status": "data file not read"})
                continue
            rr_segments = ov.load_rrwavenet(rr_files[subject][0]) if "RRWaveNet" in args.models else {}
            corr_by_name = {r["name"]: r for r in corr.get(subject, [])}
            for k, (name, rec) in enumerate(recordings[subject].items()):
                times = np.arange(len(rec["gt"])) / FS
                raw = {}
                index = corr_by_name[name]["index"] if name in corr_by_name else k
                if "RespDiff" in args.models and name in respdiff.get(subject, {}):
                    raw["RespDiff"] = ov.Track(respdiff[subject][name], FS).at(times)
                if "Deep corr-encoder" in args.models and name in corr_by_name:
                    raw["Deep corr-encoder"] = ov.Track(corr_by_name[name]["output"], FS).at(times)
                if "RRWaveNet" in args.models and index in rr_segments:
                    raw["RRWaveNet"] = rrwavenet_track(rr_segments[index], rr_alignment.get((subject, index)),
                                                       dataset_key, rec["gt"], args.rrwavenet_lowpass_hz)
                missing = [m for m in args.models if m not in raw]
                if missing:
                    infos.append({"Dataset": label, "Subject": subject, "Recording": name,
                                  "Status": "no output of " + ", ".join(missing)})
                    continue
                rows, info = recording_windows(dataset_key, label, subject, name, rec, raw, args.sizes)
                info["Status"] = "used"
                if "RRWaveNet" in raw:
                    info["RRWaveNet_Run"] = rr_files[subject][1]
                infos.append(info)
                for size in args.sizes:
                    windows[size] += rows[size]
            print(f"  {subject}: " + ", ".join(f"{sum(1 for r in windows[s] if r['Subject'] == subject and r['Dataset'] == label)}"
                                              f" x {s} s" for s in args.sizes), flush=True)
    for size in args.sizes:
        pd.DataFrame(windows[size]).to_csv(out / f"windows_{size}s.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(infos).to_csv(out / "recordings.csv", index=False, encoding="utf-8-sig")
    print("windows: " + ", ".join(f"{len(windows[s])} x {s} s" for s in args.sizes) + f" -> {out}")


# ---------------------------------------------------------------------------
# Stage 2: analyze
# ---------------------------------------------------------------------------

def bh(p: Sequence[float]) -> np.ndarray:
    """Benjamini-Hochberg q values (NaN kept)."""
    p = np.asarray(p, float)
    q = np.full(len(p), np.nan)
    ok = np.flatnonzero(np.isfinite(p))
    if len(ok):
        order = ok[np.argsort(p[ok])]
        ranked = p[order] * len(ok) / np.arange(1, len(ok) + 1)
        q[order] = np.minimum(1.0, np.minimum.accumulate(ranked[::-1])[::-1])
    return q


def add_scores(win: pd.DataFrame, models: List[str], args) -> pd.DataFrame:
    win = win.copy()
    for m in models:
        win[f"{m}_Score"] = win[f"{m}_PCC"] - args.ae_weight * win[f"{m}_AE"].clip(upper=args.ae_cap)
    scores = win[[f"{m}_Score" for m in models]].to_numpy()
    order = np.argsort(-scores, axis=1)
    best = scores[np.arange(len(win)), order[:, 0]]
    second = scores[np.arange(len(win)), order[:, 1]] if len(models) > 1 else best
    win["Winner"] = np.where(best - second >= args.tie_margin, np.array(models)[order[:, 0]], "Tie")
    win["Winner_Margin"] = best - second
    for k, m in enumerate(models):
        others = np.delete(scores, k, axis=1)
        win[f"{m}_Advantage"] = scores[:, k] - others.max(axis=1)
    ae = win[[f"{m}_AE" for m in models]].to_numpy()
    win["Winner_by_AE"] = np.array(models)[np.argmin(ae, axis=1)]
    win["Winner_by_PCC"] = np.array(models)[np.argmax(win[[f"{m}_PCC" for m in models]].to_numpy(), axis=1)]
    for f in FEATURES:
        win[f + "_pct"] = win.groupby("Dataset")[f].rank(pct=True)
    return win


def subject_table(win: pd.DataFrame, models: List[str], args) -> pd.DataFrame:
    agg = {"Windows": ("Start_Sec", "size")}
    for m in models:
        for k in ("AE", "PCC", "Score", "Advantage", "TiTtot_AE"):
            agg[f"{m}_{k}"] = (f"{m}_{k}", "mean")
        agg[f"{m}_Win_Share"] = ("Winner", lambda s, m=m: float((s == m).mean()))
    agg["Tie_Share"] = ("Winner", lambda s: float((s == "Tie").mean()))
    for f in FEATURES:
        agg[f] = (f, "median")
    sub = win.groupby(["Dataset", "Subject"]).agg(**agg).reset_index()
    scores = sub[[f"{m}_Score" for m in models]].to_numpy()
    order = np.argsort(-scores, axis=1)
    best = scores[np.arange(len(sub)), order[:, 0]]
    second = scores[np.arange(len(sub)), order[:, 1]] if len(models) > 1 else best
    sub["Winner"] = np.where(best - second >= args.tie_margin, np.array(models)[order[:, 0]], "Tie")
    sub["Winner_Margin"] = best - second
    for f in FEATURES:
        sub[f + "_pct"] = sub.groupby("Dataset")[f].rank(pct=True)
    return sub


def scopes(frame: pd.DataFrame):
    yield "All", frame
    for ds in sorted(frame["Dataset"].unique()):
        yield ds, frame[frame["Dataset"] == ds]


def overview(win: pd.DataFrame, sub: pd.DataFrame, models: List[str]) -> pd.DataFrame:
    rows = []
    for (scope, w), (_, s) in zip(scopes(win), scopes(sub)):
        row = {"Scope": scope, "Subjects": len(s), "Windows": len(w),
               "Tie_Windows": float((w.Winner == "Tie").mean()), "Tie_Subjects": float((s.Winner == "Tie").mean())}
        for m in models:
            row.update({f"{m}_AE_mean": w[f"{m}_AE"].mean(), f"{m}_AE_median": w[f"{m}_AE"].median(),
                        f"{m}_PCC_mean": w[f"{m}_PCC"].mean(), f"{m}_Score_mean": w[f"{m}_Score"].mean(),
                        f"{m}_TiTtot_AE_mean": w[f"{m}_TiTtot_AE"].mean(),
                        f"{m}_Won_Windows": float((w.Winner == m).mean()),
                        f"{m}_Won_Subjects": float((s.Winner == m).mean())})
        rows.append(row)
    return pd.DataFrame(rows)


def spearman(x: pd.Series, y: pd.Series, minimum: int) -> Tuple[float, float]:
    keep = x.notna() & y.notna()
    if keep.sum() < minimum or x[keep].nunique() < 2 or y[keep].nunique() < 2:
        return math.nan, math.nan
    rho, p = stats.spearmanr(x[keep], y[keep])
    return float(rho), float(p)


def associations(win: pd.DataFrame, sub: pd.DataFrame, models: List[str]) -> pd.DataFrame:
    rows = []
    for (scope, w), (_, s) in zip(scopes(win), scopes(sub)):
        suffix = "_pct" if scope == "All" else ""
        for m in models:
            for f in FEATURES:
                rho, p = spearman(s[f + suffix], s[f"{m}_Advantage"], 5)
                rho_w, _ = spearman(w[f + suffix], w[f"{m}_Advantage"], 10)
                rows.append({"Scope": scope, "Model": m, "Feature": f, "Meaning": FEATURES[f],
                             "Subjects": int(s[f].notna().sum()), "rho_subjects": rho, "p_subjects": p,
                             "Windows": int(w[f].notna().sum()), "rho_windows": rho_w})
    table = pd.DataFrame(rows)
    table["q_subjects"] = np.nan
    for _, idx in table.groupby("Scope").groups.items():
        table.loc[idx, "q_subjects"] = bh(table.loc[idx, "p_subjects"])
    table["Reading"] = [
        (f"{r.Model} gains on {'higher' if r.rho_subjects > 0 else 'lower'} {r.Feature}" if r.q_subjects < 0.1 else "")
        for r in table.itertuples()]
    return table


def profiles(win: pd.DataFrame, models: List[str]) -> pd.DataFrame:
    rows = []
    for f in FEATURES:
        bins = pd.cut(win[f + "_pct"], [0, 0.2, 0.4, 0.6, 0.8, 1.0], labels=[1, 2, 3, 4, 5], include_lowest=True)
        for b, w in win.groupby(bins, observed=True):
            row = {"Feature": f, "Quintile": int(b), "Windows": len(w), "Subjects": w.Subject.nunique(),
                   "Feature_median": w[f].median()}
            for m in models:
                row.update({f"{m}_AE": w[f"{m}_AE"].mean(), f"{m}_PCC": w[f"{m}_PCC"].mean(),
                            f"{m}_Won": float((w.Winner == m).mean())})
            row["Tie"] = float((w.Winner == "Tie").mean())
            rows.append(row)
    return pd.DataFrame(rows)


def describe(medians: pd.Series) -> str:
    """'high X, low Y' from the median percentiles of a cluster."""
    parts = [(abs(v - 0.5), f"{'high' if v > 0.5 else 'low'} {k}") for k, v in medians.items()
             if np.isfinite(v) and abs(v - 0.5) >= 0.15]
    return ", ".join(text for _, text in sorted(parts, reverse=True)) or "middle of every feature"


def kmeans(X: np.ndarray, k: int, seed: int):
    from sklearn.cluster import KMeans
    return KMeans(n_clusters=k, n_init=20, random_state=seed).fit(X)


def cluster_profile(frame: pd.DataFrame, labels: np.ndarray, feats: List[str], models: List[str]) -> pd.DataFrame:
    rows = []
    sizes = pd.Series(labels).value_counts()
    rank = {c: i + 1 for i, c in enumerate(sizes.index)}  # 1 = largest
    for c in sizes.index:
        part = frame[labels == c]
        med = part[[f + "_pct" for f in feats]].median()
        med.index = feats
        row = {"Cluster": rank[c], "Units": len(part), "Subjects": part.Subject.nunique(),
               "Datasets": ", ".join(f"{d} {n}" for d, n in part.Dataset.value_counts().items()),
               "Description": describe(med)}
        row.update({f"{f}_pct_median": med[f] for f in feats})
        row.update({f"{f}_median": part[f].median() for f in feats})
        for m in models:
            row.update({f"{m}_AE": part[f"{m}_AE"].mean(), f"{m}_PCC": part[f"{m}_PCC"].mean(),
                        f"{m}_Score": part[f"{m}_Score"].mean(), f"{m}_Won": float((part.Winner == m).mean())})
        row["Tie"] = float((part.Winner == "Tie").mean())
        row["Best_Model"] = max(models, key=lambda m: row[f"{m}_Score"])
        rows.append(row)
    return pd.DataFrame(rows).sort_values("Cluster")


def cluster_analysis(win: pd.DataFrame, sub: pd.DataFrame, models: List[str], args) -> Dict[str, pd.DataFrame]:
    from sklearn.metrics import silhouette_score
    feats = list(args.cluster_features)
    out = {}
    for unit, frame in (("Windows", win), ("Subjects", sub)):
        X = frame[[f + "_pct" for f in feats]].fillna(0.5).to_numpy()
        if len(frame) < 3 * args.clusters:
            continue
        sil = []
        for k in range(2, min(7, len(frame) - 1)):
            labels = kmeans(X, k, args.seed).labels_
            sample = np.random.default_rng(args.seed).choice(len(X), min(len(X), 4000), replace=False)
            sil.append({"Unit": unit, "k": k, "Silhouette": silhouette_score(X[sample], labels[sample])})
        out[f"Silhouette_{unit}"] = pd.DataFrame(sil)
        labels = kmeans(X, args.clusters, args.seed).labels_
        prof = cluster_profile(frame, labels, feats, models)
        out[f"Clusters_{unit}"] = prof
        rank = dict(zip(pd.Series(labels).value_counts().index, range(1, args.clusters + 1)))
        member = frame[["Dataset", "Subject"]].copy()
        member["Cluster"] = [rank[c] for c in labels]
        if unit == "Windows":
            share = member.groupby(["Dataset", "Subject"]).Cluster.agg(
                Main_Cluster=lambda s: int(s.mode().iloc[0]),
                Main_Share=lambda s: float((s == s.mode().iloc[0]).mean())).reset_index()
            out["Window_Cluster_of_Subjects"] = share.merge(sub[["Dataset", "Subject", "Winner"]],
                                                            on=["Dataset", "Subject"])
        else:
            out["Subject_Clusters"] = member.merge(sub[["Dataset", "Subject", "Winner"] +
                                                       [f"{m}_Score" for m in models] + feats],
                                                   on=["Dataset", "Subject"]).sort_values(["Cluster", "Dataset"])
    return out


def model_win_clusters(win: pd.DataFrame, models: List[str], args) -> pd.DataFrame:
    """k-means of the windows each model wins: the kinds of windows where it is best."""
    feats = list(args.cluster_features)
    rows = []
    for m in models:
        wins = win[win.Winner == m]
        if len(wins) < 3 * args.clusters:
            rows.append({"Model": m, "Cluster": 0, "Units": len(wins), "Description": "too few wins to cluster"})
            continue
        labels = kmeans(wins[[f + "_pct" for f in feats]].fillna(0.5).to_numpy(), args.clusters, args.seed).labels_
        prof = cluster_profile(wins, labels, feats, models)
        rank = dict(zip(pd.Series(labels).value_counts().index, range(1, args.clusters + 1)))
        tops = []
        for c in range(1, args.clusters + 1):
            part = wins[np.array([rank[x] for x in labels]) == c]
            counts = part.groupby(["Dataset", "Subject"]).size().sort_values(ascending=False).head(8)
            tops.append(", ".join(f"{d[:5]} {s} ({n})" for (d, s), n in counts.items()))
            others = [o for o in models if o != m]
            runner = part[[f"{o}_Score" for o in others]].mean().idxmax()[: -len("_Score")] if others else ""
            prof.loc[prof.Cluster == c, "Runner_Up"] = runner
            prof.loc[prof.Cluster == c, f"{m}_Advantage"] = part[f"{m}_Advantage"].mean()
        prof["Top_Subjects (windows)"] = tops
        prof.insert(0, "Model", m)
        rows += prof.to_dict("records")
    return pd.DataFrame(rows)


def rules(win: pd.DataFrame, args) -> Tuple[pd.DataFrame, str]:
    """Depth-3 decision tree: winner (Tie left out) from the feature
    percentiles, cross-validated by subject."""
    from sklearn.metrics import balanced_accuracy_score
    from sklearn.model_selection import GroupKFold
    from sklearn.tree import DecisionTreeClassifier, export_text
    data = win[win.Winner != "Tie"]
    feats = list(FEATURES)
    X = data[[f + "_pct" for f in feats]].fillna(0.5).to_numpy()
    y = data["Winner"].to_numpy()
    groups = (data.Dataset + "_" + data.Subject).to_numpy()
    classes = np.unique(y)
    if len(classes) < 2 or len(np.unique(groups)) < 5:
        return pd.DataFrame([{"Note": "too few winners or subjects"}]), ""
    leaf = max(20, int(0.03 * len(data)))

    def tree():
        return DecisionTreeClassifier(max_depth=3, min_samples_leaf=leaf, class_weight="balanced",
                                      random_state=args.seed)

    pred = np.empty(len(y), dtype=object)
    for train, test in GroupKFold(n_splits=min(5, len(np.unique(groups)))).split(X, y, groups):
        pred[test] = tree().fit(X[train], y[train]).predict(X[test])
    model = tree().fit(X, y)
    text = export_text(model, feature_names=[f + "_pct" for f in feats], show_weights=True)
    imp = pd.DataFrame({"Feature": feats, "Importance": model.feature_importances_}).sort_values(
        "Importance", ascending=False)
    imp.insert(0, "Balanced_accuracy_by_subject_CV", balanced_accuracy_score(y, pred.astype(str)))
    imp.insert(1, "Chance", 1.0 / len(classes))
    imp.insert(2, "Windows", len(y))
    return imp, text


def profile_figure(prof: pd.DataFrame, models: List[str], size: int, path: Path) -> None:
    feats = list(FEATURES)
    fig, axes = plt.subplots(len(feats), 3, figsize=(13.5, 2.05 * len(feats) + 1.2), squeeze=False)
    for i, f in enumerate(feats):
        part = prof[prof.Feature == f]
        for j, (key, title) in enumerate((("AE", "rate AE (breaths/min)"), ("PCC", "PCC with GT"),
                                          ("Won", "share of windows won"))):
            ax = axes[i, j]
            for m in models:
                ax.plot(part.Quintile, part[f"{m}_{key}"], marker="o", ms=5, lw=2, color=ov.MODEL_COLORS[m], label=m)
            if key == "Won":
                ax.plot(part.Quintile, part["Tie"], marker="o", ms=4, lw=1.2, ls="--", color="#8f8e8a", label="Tie")
            ax.set_xticks([1, 2, 3, 4, 5])
            ax.set_xticklabels(["lowest\n20%", "", "middle", "", "highest\n20%"] if i == len(feats) - 1 else [])
            ov.style_axis(ax)
            if i == 0:
                ax.set_title(title, fontsize=10.5, loc="left")
            if j == 0:
                ax.set_ylabel(f"{f}\n" + "\n".join(_wrap(FEATURES[f], 26)), fontsize=8.2)
    handles, labels = axes[0, 2].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=len(labels), frameon=False, fontsize=10,
               bbox_to_anchor=(0.5, 1.0))
    fig.text(0.01, 0.003, f"{size} s windows; x: within-dataset quintile of the feature (pooled datasets). "
             "Win: highest Score = PCC - 0.1 x min(AE, 5); Tie: second within the tie margin.",
             fontsize=8, color=ov.TEXT_SECONDARY)
    fig.tight_layout(rect=(0, 0.012, 1, 0.985))
    fig.savefig(path.with_suffix(".png"), dpi=110)
    fig.savefig(path.with_suffix(".pdf"))
    plt.close(fig)


def _wrap(text: str, width: int) -> List[str]:
    words, lines, line = text.split(), [], ""
    for w in words:
        if len(line) + len(w) + 1 > width and line:
            lines.append(line)
            line = w
        else:
            line = f"{line} {w}".strip()
    return lines + [line] if line else lines


def cluster_figure(prof: pd.DataFrame, feats: List[str], models: List[str], title: str, unit: str,
                   path: Path) -> None:
    height = 2.6 + 0.75 * len(prof)
    fig = plt.figure(figsize=(18, height))
    bottom, top = 1.25 / height, 1 - 0.75 / height
    heat = prof[[f"{f}_pct_median" for f in feats]].to_numpy(float)
    ax = fig.add_axes([0.11, bottom, 0.34, top - bottom])
    ax.imshow(heat, cmap="RdBu_r", vmin=0, vmax=1, aspect="auto")
    for (i, j), v in np.ndenumerate(heat):
        ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=9,
                color="white" if abs(v - 0.5) > 0.3 else ov.TEXT_PRIMARY)
    ax.set_xticks(range(len(feats)))
    ax.set_xticklabels(feats, rotation=30, ha="right", fontsize=9)
    ax.set_yticks(range(len(prof)))
    ax.set_yticklabels([f"C{int(c)} (n={int(n)}, {int(s)} subj.)"
                        for c, n, s in zip(prof.Cluster, prof.Units, prof.Subjects)], fontsize=9)
    ax.set_title("median within-dataset percentile (blue low, red high)", fontsize=10, loc="left")
    y = np.arange(len(prof))
    h = 0.8 / len(models)
    panels = (("PCC", "mean PCC"), ("AE", "mean rate AE (breaths/min)"),
              ("Won", f"share of {unit} won (gray: tie)"))
    for k, (key, label) in enumerate(panels):
        ax = fig.add_axes([0.5 + k * 0.17, bottom, 0.14, top - bottom])
        for i, m in enumerate(models):
            ax.barh(y - 0.4 + (i + 0.5) * h, prof[f"{m}_{key}"], height=h * 0.9, color=ov.MODEL_COLORS[m], label=m)
        if key == "Won":
            ax.scatter(prof["Tie"], y, marker="|", s=300, color="#8f8e8a", zorder=4, label="Tie")
            ax.set_xlim(0, 1)
        ax.set_yticks(y)
        ax.set_yticklabels([f"C{int(c)}" for c in prof.Cluster])
        ax.set_ylim(len(prof) - 0.5, -0.5)
        ov.style_axis(ax)
        ax.set_title(label, fontsize=10, loc="left")
    handles = [plt.Rectangle((0, 0), 1, 1, color=ov.MODEL_COLORS[m]) for m in models] + \
              [plt.Line2D([0], [0], marker="|", ls="", ms=14, color="#8f8e8a")]
    fig.legend(handles, list(models) + ["Tie"], loc="lower center", ncol=len(models) + 1, frameon=False,
               fontsize=9.5, bbox_to_anchor=(0.67, 0.0))
    fig.text(0.01, 1 - 0.3 / height, title, fontsize=11, ha="left", va="center")
    fig.savefig(path.with_suffix(".png"), dpi=120)
    fig.savefig(path.with_suffix(".pdf"))
    plt.close(fig)


def summary_lines(size: int, ovw: pd.DataFrame, assoc: pd.DataFrame, clus: Dict[str, pd.DataFrame],
                  mwc: pd.DataFrame, imp: pd.DataFrame, models: List[str]) -> List[str]:
    lines = [f"===== {size} s windows ====="]
    for r in ovw.to_dict("records"):
        lines.append(f"[{r['Scope']}] {r['Subjects']} subjects, {r['Windows']} windows; ties {r['Tie_Windows']:.0%} "
                     "of windows")
        for m in models:
            lines.append(f"   {m:18s} AE {r[f'{m}_AE_mean']:.2f} (median {r[f'{m}_AE_median']:.2f}), PCC "
                         f"{r[f'{m}_PCC_mean']:.2f}, Ti/Ttot AE {r[f'{m}_TiTtot_AE_mean']:.3f}; wins "
                         f"{r[f'{m}_Won_Windows']:.0%} of windows, {r[f'{m}_Won_Subjects']:.0%} of subjects")
    lines.append("Where each model gains on the others (subjects, Spearman of its Advantage, q < 0.1):")
    for m in models:
        part = assoc[(assoc.Model == m) & (assoc.q_subjects < 0.1)].sort_values("p_subjects")
        if part.empty:
            lines.append(f"   {m}: no feature with q < 0.1")
        for r in part.to_dict("records"):
            lines.append(f"   {m} [{r['Scope']}]: {'higher' if r['rho_subjects'] > 0 else 'lower'} {r['Feature']} "
                         f"({r['Meaning']}); rho {r['rho_subjects']:+.2f}, q {r['q_subjects']:.3f}, "
                         f"{r['Subjects']} subjects")
    for unit in ("Windows", "Subjects"):
        prof = clus.get(f"Clusters_{unit}")
        if prof is None:
            continue
        lines.append(f"Clusters of the {unit.lower()}:")
        for r in prof.to_dict("records"):
            won = ", ".join(f"{SHORT.get(m, m)} {r[f'{m}_Won']:.0%}" for m in models)
            count = f"{r['Units']} subjects" if unit == "Subjects" else f"{r['Units']} windows, {r['Subjects']} subjects"
            lines.append(f"   C{r['Cluster']} ({count}): "
                         f"{r['Description']}; best {r['Best_Model']}; won {won}, tie {r['Tie']:.0%}")
    lines.append("Kinds of windows each model wins (k-means of its wins):")
    for r in mwc.to_dict("records"):
        if r["Cluster"] == 0:
            lines.append(f"   {r['Model']}: {r['Description']}")
            continue
        lines.append(f"   {r['Model']} W{int(r['Cluster'])} ({int(r['Units'])} windows, {int(r['Subjects'])} subjects): "
                     f"{r['Description']}; runner-up {r['Runner_Up']}")
    if "Balanced_accuracy_by_subject_CV" in imp:
        lines.append(f"Rules (depth-3 tree): balanced accuracy {imp.Balanced_accuracy_by_subject_CV.iloc[0]:.2f} "
                     f"(chance {imp.Chance.iloc[0]:.2f}); top features: "
                     + ", ".join(f"{f} {v:.2f}" for f, v in zip(imp.Feature.head(4), imp.Importance.head(4))))
    return lines


def analyze(args) -> None:
    out = Path(args.output_root)
    sheets: Dict[str, pd.DataFrame] = {}
    text: List[str] = []
    trees: List[str] = []
    for size in args.sizes:
        path = out / f"windows_{size}s.csv"
        if not path.exists():
            print(f"{path} not found (run --stage extract first)")
            continue
        win = pd.read_csv(path, dtype={"Subject": str, "Recording": str})
        models = [m for m in args.models if f"{m}_AE" in win.columns]
        win = win.dropna(subset=[f"{m}_{k}" for m in models for k in ("AE", "PCC")]).reset_index(drop=True)
        if win.empty or len(models) < 2:
            print(f"{size} s: nothing to compare")
            continue
        win = add_scores(win, models, args)
        sub = subject_table(win, models, args)
        ovw = overview(win, sub, models)
        assoc = associations(win, sub, models)
        prof = profiles(win, models)
        clus = cluster_analysis(win, sub, models, args)
        mwc = model_win_clusters(win, models, args)
        imp, tree_text = rules(win, args)
        tag = f"{size}s"
        sheets[f"Overview_{tag}"] = ovw
        sheets[f"Subjects_{tag}"] = sub
        sheets[f"Associations_{tag}"] = assoc
        sheets[f"Profiles_{tag}"] = prof
        for name, frame in clus.items():
            sheets[f"{name}_{tag}"[:31]] = frame
        sheets[f"Model_Win_Clusters_{tag}"] = mwc
        sheets[f"Rules_{tag}"] = imp
        trees.append(f"===== {size} s =====\n{tree_text}")
        win.to_csv(out / f"windows_{tag}_scored.csv", index=False, encoding="utf-8-sig")
        profile_figure(prof, models, size, out / f"profiles_{tag}")
        for unit in ("Windows", "Subjects"):
            if f"Clusters_{unit}" in clus:
                cluster_figure(clus[f"Clusters_{unit}"], list(args.cluster_features), models,
                               f"{size} s windows: k-means of the {unit.lower()} (k = {args.clusters})",
                               unit.lower(), out / f"clusters_{unit.lower()}_{tag}")
        text += summary_lines(size, ovw, assoc, clus, mwc, imp, models) + [""]
    if not sheets:
        return
    notes = pd.DataFrame({"Item": ["Score", "Winner", "Advantage", "Percentiles", "Associations", "Profiles",
                                   "Clusters", "Model_Win_Clusters", "Rules", "Caution"] + list(FEATURES),
                          "Meaning": [
                              f"PCC - {args.ae_weight} x min(AE, {args.ae_cap}) per window and model",
                              f"highest Score; Tie when the second is within {args.tie_margin}; a subject's from "
                              "the mean Score of its windows",
                              "Score minus the best Score of the other models (> 0: this model is best)",
                              "feature ranks within each dataset (0-1), used for pooling and clustering",
                              "Spearman of every model's Advantage with every feature: between subjects (p, BH q "
                              "per scope) and between windows (descriptive)",
                              "mean AE, PCC and win share in the within-dataset quintiles of every feature",
                              f"k-means (k = {args.clusters}) on the percentiles of {', '.join(args.cluster_features)} "
                              "(missing -> 0.5); Cluster 1 = largest; Best_Model = highest mean Score",
                              "k-means of the windows each model wins; Runner_Up = best other model there",
                              "depth-3 tree predicting the winner from all feature percentiles; balanced accuracy "
                              "from 5-fold cross-validation by subject (rules in rules.txt)",
                              "windows of one subject are not independent; read the subject-level q, not the "
                              "window-level rho, for significance"] + list(FEATURES.values())})
    sheets["Notes"] = notes
    with pd.ExcelWriter(out / "model_strength.xlsx") as writer:
        for name, frame in sheets.items():
            frame.to_excel(writer, sheet_name=name[:31], index=False)
    (out / "summary.txt").write_text("\n".join(text), encoding="utf-8")
    (out / "rules.txt").write_text("\n".join(trees), encoding="utf-8")
    print("\n".join(text))
    print(f"saved under {out}")


# ---------------------------------------------------------------------------
# Command line
# ---------------------------------------------------------------------------

def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--stage", choices=["all", "extract", "analyze"], default="all")
    parser.add_argument("--datasets", nargs="+", choices=list(ov.DATASET_ORDER), default=list(ov.DATASET_ORDER))
    parser.add_argument("--subjects", nargs="+", default=None, help="Only these subjects.")
    parser.add_argument("--models", nargs="+", choices=list(MODELS), default=list(MODELS))
    parser.add_argument("--sizes", nargs="+", type=int, default=list(WINDOW_SIZES), help="Window sizes (s).")
    parser.add_argument("--respdiff-root", type=Path, default=ov.RESPDIFF_ROOT)
    parser.add_argument("--respdiff-sampling", default=ov.RESPDIFF_SAMPLING)
    parser.add_argument("--correncoder-root", type=Path, default=ov.CORRENCODER_ROOT)
    parser.add_argument("--rrwavenet-261007-root", type=Path, default=ov.RRWAVENET_261007_ROOT)
    parser.add_argument("--rrwavenet-261013-root", type=Path, default=ov.RRWAVENET_261013_ROOT)
    parser.add_argument("--rrwavenet-unified-root", type=Path, default=None,
                        help="One 261102 result folder (layout of 261013) for every dataset.")
    parser.add_argument("--rrwavenet-member", choices=["ensemble", "m0", "m1", "m2"], default="ensemble")
    parser.add_argument("--rrwavenet-lowpass-hz", type=float, default=ov.RRWAVENET_LOWPASS_HZ)
    parser.add_argument("--capnobase-path", type=Path, default=None)
    parser.add_argument("--bidmc-path", type=Path, default=None)
    parser.add_argument("--steam2-ppg-dir", type=Path, default=None)
    parser.add_argument("--steam2-rsp-dir", type=Path, default=None)
    parser.add_argument("--output-root", type=Path, default=PROJECT_ROOT)
    parser.add_argument("--ae-weight", type=float, default=AE_WEIGHT)
    parser.add_argument("--ae-cap", type=float, default=AE_CAP)
    parser.add_argument("--tie-margin", type=float, default=TIE_MARGIN)
    parser.add_argument("--clusters", type=int, default=N_CLUSTERS)
    parser.add_argument("--cluster-features", nargs="+", choices=list(FEATURES), default=list(CLUSTER_FEATURES))
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> None:
    args = parse_args(argv)
    if args.capnobase_path:
        ov.DATASET_CONFIGS["capnobase"]["data_path"] = str(args.capnobase_path)
    if args.bidmc_path:
        ov.DATASET_CONFIGS["bidmc"]["data_path"] = str(args.bidmc_path)
    if args.steam2_ppg_dir:
        ov.DATASET_CONFIGS["steam2"]["ppg_dir"] = str(args.steam2_ppg_dir)
    if args.steam2_rsp_dir:
        ov.DATASET_CONFIGS["steam2"]["rsp_dir"] = str(args.steam2_rsp_dir)
    if args.stage in ("all", "extract"):
        extract(args)
    if args.stage in ("all", "analyze"):
        analyze(args)


if __name__ == "__main__":
    main()
