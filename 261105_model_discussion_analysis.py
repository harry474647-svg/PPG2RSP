r"""261105_model_discussion_analysis: the subjects and the 30 s / 60 s windows for
the discussion of RespDiff, the deep corr-encoder and RRWaveNet (stored test
results of every subject; nothing is trained). Two analyses on the same
windows:

  1. Observations: the subjects (per dataset, the --top) where a known
     behaviour of a model is clearest, and the window that shows it best
  2. Worst subjects: per model, the subjects (the --worst-top) on which it
     performs worst, and the window where it is worst

Waveforms
---------
  RespDiff, corr-encoder   as stored (30 Hz)
  RRWaveNet                CapnoBase: 261007, ensemble of the full model;
                           BIDMC / STEAM2: 261013, ensemble of the weight decay
                           selected per fold (--rrwavenet-member m0 / m1 / m2
                           for one member instead); its training sign / lag
                           undone and low-pass 1 Hz (order 8), as 261104
  every model is shifted by at most half a breath and, if needed, inverted once
  per recording to its best r with the GT (the same rule for all). A recording
  is used when RRWaveNet and at least one other model exist.

In a window
-----------
  rate     FFT peak of the model's waveform in the rate band (CapnoBase 3-60,
           BIDMC / STEAM2 6-30 breaths/min); ratio = rate / GT rate;
           AE = |rate - GT rate|, GT rate from the breaths in the window
           (CapnoBase expert labels, BIDMC ann1; STEAM2: the GT cycles); a
           window needs >= 3 breaths
  PCC      Pearson r of the aligned model and the GT in the window
  cycles   breathing cycles of the GT and of every model (261030's detector
           on the aligned waveforms); a GT cycle inside the window is matched
           by a model cycle starting within half its length (1:1);
           Missed = GT cycles not matched, Extra = model cycles inside the
           window not matched, Coverage = matched / GT cycles
  score    PCC - 0.1 x min(AE, 5) (an AE of 1 breath/min costs 0.1 of PCC)
Windows of every size start every --step s; the non-overlapping ones (from
0 s, Grid) are used to rank the subjects, all of them to pick the window drawn.

1. Observations ("RRWaveNet right": AE <= 1 breath/min)
  ce_double    corr-encoder ratio 1.75-2.25, RRWaveNet right; score = AE gap
  rd_missed    RRWaveNet right and missing <= 1 cycle, RespDiff missing >= 2
               cycles more than RRWaveNet or at ratio 0.4-0.6; score = cycles
               missed by RespDiff minus by RRWaveNet + 0.1 x AE gap (<= 5)
  rd_double    RespDiff ratio 1.75-2.25, RRWaveNet right; score = AE gap
  rr_coverage  >= 3 GT cycles, RRWaveNet coverage >= the best other + 0.15,
               RRWaveNet extra <= 1 (so that cycles found twice do not count);
               score = the coverage margin
  Subjects: per size, the share of Grid windows that show the observation and
  the mean score of those; ranked once per dataset by the mean of both over
  the sizes; subjects without such a window are not listed. Window: per
  listed subject and size, the one with the highest score.

2. Worst subjects (per model)
  Subject score = the mean score of the model over the Grid windows per size,
  then the mean over the sizes; lowest first, pooled over the datasets (or per
  dataset with --worst-per-dataset). Beside it: the mean AE, PCC, coverage and
  score of every model, and
    Deficit   = the model's score - the best score of the other models
    Pattern   "model-specific" when Deficit <= -0.2 (the others do clearly
              better), "others worse" when Deficit >= +0.2 (the model is low
              but still clearly the best), otherwise "similar to others"
  Window: per listed subject and size, the one with the model's lowest score.

Outputs (default D:\PPG2RSP_RRWaveNet_Inspired\261105_model_discussion_analysis)
  discussion_analysis.xlsx   Obs_{observation} (every subject, ranked, Listed =
                             among the top), Obs_Windows, Worst_{model} (every
                             subject, ranked), Worst_Windows, Notes
  windows_30s.csv, windows_60s.csv   every window (start every --step s)
  observations\{observation}_{dataset}.pdf,
  observations\{observation}\{dataset}\{rank}_{subject}_{size}s.png
  worst\worst_{model}.pdf, worst\worst_{model}\{rank}_{dataset}_{subject}_{size}s.png
      one figure per listed subject and size: one row per model (GT and the
      model, min-max scaled in the window), GT inspirations as a gray
      background, the model's inspirations as bars, GT cycles the model missed
      marked, rate / AE / PCC / cycles beside every row

Run:             python 261105_model_discussion_analysis.py
Result folders:  --respdiff-root, --correncoder-root, --rrwavenet-261007-root,
                 --rrwavenet-261013-root
Data files:      --capnobase-path, --bidmc-path, --steam2-ppg-dir, --steam2-rsp-dir
Options:         --datasets capnobase bidmc, --sizes 30 60, --step 5, --top 5,
                 --worst-top 10, --worst-per-dataset, --no-png
Uses (same folder): 261104_Waveform_Overlay.py (data and result readers),
                 261030_Waveform_Morphology_Analysis.py (cycle detector),
                 261007_RRWaveNet_Ensemble_NoEvent.py (CapnoBase reader)
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
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from scipy.signal import detrend

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

PROJECT_ROOT = Path(r"D:\PPG2RSP_RRWaveNet_Inspired\261105_model_discussion_analysis")
FS = ov.FS  # 30 Hz
MODELS = ("RespDiff", "Deep corr-encoder", "RRWaveNet")
OTHERS = ("RespDiff", "Deep corr-encoder")
SHORT = {"RespDiff": "RespDiff", "Deep corr-encoder": "CorrEncoder", "RRWaveNet": "RRWaveNet"}
RATE_BAND_BPM = {"capnobase": (3.0, 60.0), "bidmc": (6.0, 30.0), "steam2": (6.0, 30.0)}
WINDOW_SIZES = (30, 60)
STEP_SEC = 5.0
MIN_RUN_SEC = 30.0
MIN_BREATHS = 3
AE_WEIGHT, AE_CAP = 0.1, 5.0

# 1. observations
TOP = 5
RATE_OK_BPM = 1.0
DOUBLE = (1.75, 2.25)
HALF = (0.4, 0.6)
MISSED_MIN = 2
COVERAGE_MARGIN = 0.15
OBSERVATIONS = {
    "ce_double": ("Deep corr-encoder at twice the rate, RRWaveNet right", ("Deep corr-encoder", "RRWaveNet")),
    "rd_missed": ("RespDiff misses breathing cycles / at half the rate, RRWaveNet right", ("RespDiff", "RRWaveNet")),
    "rd_double": ("RespDiff at twice the rate, RRWaveNet right", ("RespDiff", "RRWaveNet")),
    "rr_coverage": ("RRWaveNet detects the breathing cycles the others miss", MODELS),
}

# 2. worst subjects
WORST_TOP = 10
SPECIFIC_DEFICIT = -0.2

# figures
GT_BAND = "#dddcd8"
MISSED_COLOR = "#b3261e"


# ---------------------------------------------------------------------------
# Data files: GT and breath times of every recording
# ---------------------------------------------------------------------------

def read_recordings(dataset_key: str) -> Dict[str, Dict[str, dict]]:
    """Subject -> {recording name: GT (30 Hz), clipped GT samples and the
    breath times (s; None without labels)}, named and cut as 261104 (and the
    corr-encoder) name and cut them."""
    cfg = ov.DATASET_CONFIGS[dataset_key]
    modality = ov.MODALITIES[dataset_key]
    out: Dict[str, Dict[str, dict]] = {}

    def add(subject, name, ppg, resp, ppg_fs, resp_fs, breaths):
        ppg30, resp30 = ov.to_fs(ppg, ppg_fs), ov.to_fs(resp, resp_fs)
        length = min(len(ppg30), len(resp30))
        clipped = ov.to_fs_mask(ov.saturated_samples(resp, resp_fs, modality, ov.SATURATION_CONFIG), resp_fs,
                                length, ov.SATURATION_CONFIG["saturation_margin_sec"])
        out.setdefault(subject, {})[name] = {"gt": resp30[:length], "clipped": clipped, "breaths": breaths}

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
# RRWaveNet results: 261007 (CapnoBase) and 261013 (BIDMC, STEAM2)
# ---------------------------------------------------------------------------

def rrwavenet_sources(dataset_key: str, args) -> Tuple[Dict[str, Tuple[Path, str]], Path]:
    """Subject -> (npz of the test outputs, run name) and the folder holding
    the Data_Quality table (training sign / lag)."""
    label = ov.DATASET_CONFIGS[dataset_key]["label"]
    member = args.rrwavenet_member
    window = ov.RRWAVENET_WINDOW

    def output_of(condition: Path, subject: str) -> Optional[Path]:
        if member == "ensemble":
            path = condition / "Ensemble_Predictions" / f"{subject}_ensemble_outputs.npz"
        else:
            path = condition / "Members" / member / "Predictions" / f"{subject}_test_outputs.npz"
        summary = condition / "Fold_Summaries" / f"{subject}_{member}_summary.json"
        return path if path.exists() and summary.exists() else None

    files: Dict[str, Tuple[Path, str]] = {}
    if dataset_key == "capnobase":
        root = Path(args.rrwavenet_261007_root)
        condition = root / "full" / window / label
        for summary in sorted((condition / "Fold_Summaries").glob(f"*_{member}_summary.json")):
            subject = summary.name[: -len(f"_{member}_summary.json")]
            path = output_of(condition, subject)
            if path is not None:
                files[subject] = (path, f"261007 full {member}")
        return files, root
    root = Path(args.rrwavenet_261013_root)
    for selection in sorted((root / "WD_Selection" / window / label).glob("*_selection.json")):
        subject = selection.name[: -len("_selection.json")]
        variant = (ov.read_json(selection) or {}).get("Selected_Variant")
        if variant:
            path = output_of(root / variant / window / label, subject)
            if path is not None:
                files[subject] = (path, f"261013 {variant} {member}")
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


def align(gt: np.ndarray, values: np.ndarray, period_sec: float) -> np.ndarray:
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
    if not np.isfinite(best[0]):
        return np.asarray(values, float)
    return best[1] * shifted(values, best[2])


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


def inside(cycles: List[dict], t0: float, t1: float) -> List[dict]:
    return [c for c in cycles if c["start"] >= t0 and c["end"] <= t1]


def ti_ttot(cycles: List[dict]) -> float:
    return float(np.median([(c["transition"] - c["start"]) / (c["end"] - c["start"]) for c in cycles])) \
        if cycles else math.nan


def rate_of(times: np.ndarray) -> float:
    return 60.0 / float(np.mean(np.diff(times))) if len(times) >= 2 else math.nan


# ---------------------------------------------------------------------------
# Windows
# ---------------------------------------------------------------------------

def match_cycles(gt: List[dict], model: List[dict]) -> Tuple[int, List[dict]]:
    """1:1 matching: a GT cycle is matched by the nearest unmatched model cycle
    starting within half the GT cycle's length; returns the number matched and
    the GT cycles left unmatched."""
    used, missed, matched = set(), [], 0
    starts = np.array([c["start"] for c in model]) if model else np.empty(0)
    for c in gt:
        tolerance = 0.5 * (c["end"] - c["start"])
        best, best_gap = None, np.inf
        for k in np.flatnonzero(np.abs(starts - c["start"]) <= tolerance):
            gap = abs(starts[k] - c["start"])
            if k not in used and gap < best_gap:
                best, best_gap = int(k), gap
        if best is None:
            missed.append(c)
        else:
            used.add(best)
            matched += 1
    return matched, missed


def window_row(values: Dict[str, np.ndarray], cycles: Dict[str, List[dict]], breaths: np.ndarray, t0: float,
               t1: float, band: Tuple[float, float]) -> Optional[dict]:
    a, b = int(round(t0 * FS)), int(round(t1 * FS))
    if not all(np.isfinite(v[a:b]).all() for v in values.values()):
        return None
    br = breaths[(breaths >= t0) & (breaths < t1)]
    if len(br) < MIN_BREATHS:
        return None
    gt_cycles = inside(cycles["GT"], t0, t1)
    clean = [c for c in gt_cycles if not c.get("clipped", False)]
    gt_rr = rate_of(br)
    row = {"Start_Sec": t0, "End_Sec": t1, "GT_RR": gt_rr, "GT_Cycles": len(gt_cycles),
           "GT_Clipped_Cycles": len(gt_cycles) - len(clean), "GT_TiTtot": ti_ttot(clean),
           "GT_Harmonic2": harmonic2(values["GT"][a:b], FS, gt_rr)}
    for model in MODELS:
        if model not in values:
            continue
        seg = values[model][a:b]
        rate = fft_rate(seg, FS, band)
        near = [c for c in cycles[model] if c["end"] > t0 and c["start"] < t1]
        matched, _ = match_cycles(gt_cycles, near)
        model_in = inside(cycles[model], t0, t1)
        ae = abs(rate - gt_rr)
        pcc = ov.pearson(values["GT"][a:b], seg)
        row.update({f"{model}_Rate": rate, f"{model}_Ratio": rate / gt_rr, f"{model}_AE": ae,
                    f"{model}_PCC": pcc, f"{model}_Score": pcc - AE_WEIGHT * np.minimum(ae, AE_CAP),
                    f"{model}_Cycles": len(model_in), f"{model}_Matched": matched,
                    f"{model}_Missed": len(gt_cycles) - matched, f"{model}_Extra": max(0, len(model_in) - matched),
                    f"{model}_Coverage": matched / len(gt_cycles) if gt_cycles else math.nan})
    return row


def observe(row: dict, key: str) -> Tuple[bool, float]:
    """(the window shows the observation, its score)."""
    needs = OBSERVATIONS[key][1]
    if any(not np.isfinite(row.get(f"{m}_AE", math.nan)) for m in needs):
        return False, math.nan
    rr_ok = row["RRWaveNet_AE"] <= RATE_OK_BPM
    if key in ("ce_double", "rd_double"):
        model = "Deep corr-encoder" if key == "ce_double" else "RespDiff"
        ok = rr_ok and DOUBLE[0] <= row[f"{model}_Ratio"] <= DOUBLE[1]
        return bool(ok), row[f"{model}_AE"] - row["RRWaveNet_AE"]
    if key == "rd_missed":
        more = row["RespDiff_Missed"] - row["RRWaveNet_Missed"]
        half = HALF[0] <= row["RespDiff_Ratio"] <= HALF[1]
        ok = rr_ok and row["RRWaveNet_Missed"] <= 1 and (more >= MISSED_MIN or half)
        return bool(ok), more + 0.1 * min(max(row["RespDiff_AE"] - row["RRWaveNet_AE"], 0.0), 5.0)
    others = max(row[f"{m}_Coverage"] for m in OTHERS)
    margin = row["RRWaveNet_Coverage"] - others
    ok = row["GT_Cycles"] >= 3 and margin >= COVERAGE_MARGIN and row["RRWaveNet_Extra"] <= 1
    return bool(ok), margin


def dataset_windows(dataset_key: str, args, store: dict) -> List[dict]:
    """Every window of every recording with RRWaveNet and another model; the
    aligned waveforms and cycles go to store for the figures."""
    label = ov.DATASET_CONFIGS[dataset_key]["label"]
    modality = ov.MODALITIES[dataset_key]
    band = RATE_BAND_BPM[dataset_key]
    corr = ov.load_correncoder(Path(args.correncoder_root), label)
    respdiff = ov.load_respdiff(Path(args.respdiff_root), label, args.respdiff_sampling)
    rr_files, rr_root = rrwavenet_sources(dataset_key, args)
    rr_alignment = ov.rrwavenet_alignment_table(rr_root, label)
    subjects = sorted(set(rr_files) & (set(corr) | set(respdiff)))
    if args.subjects:
        subjects = [s for s in subjects if s in {str(x) for x in args.subjects}]
    print(f"{label}: results - RespDiff {len(respdiff)}, corr-encoder {len(corr)}, RRWaveNet {len(rr_files)}; "
          f"{len(subjects)} subjects with RRWaveNet and at least one other model", flush=True)
    if not subjects:
        return []
    recordings = read_recordings(dataset_key)
    rows = []
    for subject in subjects:
        if subject not in recordings:
            continue
        segments = ov.load_rrwavenet(rr_files[subject][0])
        corr_by_name = {r["name"]: r for r in corr.get(subject, [])}
        for k, (name, rec) in enumerate(recordings[subject].items()):
            gt = np.asarray(rec["gt"], float)
            times = np.arange(len(gt)) / FS
            index = corr_by_name[name]["index"] if name in corr_by_name else k
            raw = {}
            if name in respdiff.get(subject, {}):
                raw["RespDiff"] = ov.Track(respdiff[subject][name], FS).at(times)
            if name in corr_by_name:
                raw["Deep corr-encoder"] = ov.Track(corr_by_name[name]["output"], FS).at(times)
            if index in segments:
                raw["RRWaveNet"] = rrwavenet_track(segments[index], rr_alignment.get((subject, index)),
                                                   dataset_key, gt, args.rrwavenet_lowpass_hz)
            if "RRWaveNet" not in raw or len(raw) < 2:
                continue
            breaths = rec["breaths"]
            if breaths is None or len(breaths) < MIN_BREATHS:
                breaths = np.array([c["start"] for c in recording_cycles({"GT": gt}, modality)["GT"]])
            period = float(np.median(np.diff(breaths))) if len(breaths) >= 2 else 60.0 / np.mean(band)
            values = {"GT": gt}
            for model, v in raw.items():
                values[model] = align(gt, v, period)
            cycles = recording_cycles(values, modality)
            clipped = np.asarray(rec["clipped"], bool)
            for c in cycles["GT"]:
                c["clipped"] = bool(clipped[int(c["start"] * FS):int(math.ceil(c["end"] * FS)) + 1].any())
            store[(label, subject, name)] = {"values": values, "cycles": cycles}
            length = len(gt) / FS
            for size in args.sizes:
                grid = set(np.arange(0.0, length - size + 1e-9, size).round(3))
                starts = sorted(grid | set(np.arange(0.0, length - size + 1e-9, args.step).round(3)))
                for t0 in starts:
                    row = window_row(values, cycles, breaths, float(t0), float(t0) + size, band)
                    if row is None:
                        continue
                    row = {"Dataset": label, "Subject": subject, "Recording": name, "Size_Sec": size,
                           "Grid": t0 in grid, **row}
                    for key in OBSERVATIONS:
                        row[f"{key}_shown"], row[f"{key}_score"] = observe(row, key)
                    rows.append(row)
        print(f"  {subject}: models {', '.join(m for m in MODELS if m in raw)}", flush=True)
    return rows


# ---------------------------------------------------------------------------
# 1. Observations: subjects and windows
# ---------------------------------------------------------------------------

def observation_size_table(win: pd.DataFrame, key: str, size: int) -> pd.DataFrame:
    needs = OBSERVATIONS[key][1]
    part = win[(win.Size_Sec == size) & win.Grid]
    part = part[part[[f"{m}_AE" for m in needs]].notna().all(axis=1)]
    rows = []
    for (ds, s), d in part.groupby(["Dataset", "Subject"]):
        shown = d[d[f"{key}_shown"]]
        row = {"Dataset": ds, "Subject": s, "Size_Sec": size, "Windows": len(d), "Shown_Windows": len(shown),
               "Shown_Share": len(shown) / len(d), "Mean_Score_Shown": shown[f"{key}_score"].mean(),
               "GT_RR_median": d.GT_RR.median(), "GT_TiTtot_median": d.GT_TiTtot.median(),
               "GT_Harmonic2_median": d.GT_Harmonic2.median(), "GT_Cycles": int(d.GT_Cycles.sum())}
        for m in MODELS:
            if f"{m}_AE" in d and d[f"{m}_AE"].notna().all():
                row.update({f"{m}_AE_mean": d[f"{m}_AE"].mean(), f"{m}_Ratio_median": d[f"{m}_Ratio"].median(),
                            f"{m}_PCC_mean": d[f"{m}_PCC"].mean(),
                            f"{m}_Coverage": d[f"{m}_Matched"].sum() / max(d.GT_Cycles.sum(), 1),
                            f"{m}_Extra": int(d[f"{m}_Extra"].sum())})
        rows.append(row)
    return pd.DataFrame(rows)


def observation_subjects(win: pd.DataFrame, key: str, sizes: Sequence[int], top: int) -> pd.DataFrame:
    """Every subject per window size, ranked once per dataset by the mean over
    the sizes of the share of windows showing the observation, then of the
    mean score of those windows; Listed = the top subjects with such a window."""
    tables = [t for t in (observation_size_table(win, key, size) for size in sizes) if not t.empty]
    if not tables:
        return pd.DataFrame()
    long = pd.concat(tables, ignore_index=True)
    combined = long.groupby(["Dataset", "Subject"]).agg(
        Combined_Share=("Shown_Share", "mean"), Combined_Score=("Mean_Score_Shown", "mean"),
        Shown_Windows_All_Sizes=("Shown_Windows", "sum")).reset_index()
    combined = combined.sort_values(["Dataset", "Combined_Share", "Combined_Score"], ascending=[True, False, False])
    combined["Rank"] = combined.groupby("Dataset").cumcount() + 1
    combined["Listed"] = (combined.Shown_Windows_All_Sizes > 0) & (combined.Rank <= top)
    return long.merge(combined, on=["Dataset", "Subject"]).sort_values(["Dataset", "Rank", "Size_Sec"])


def observation_windows(win: pd.DataFrame, listed: pd.DataFrame, key: str, size: int) -> List[dict]:
    rows = []
    for r in listed.to_dict("records"):
        part = win[(win.Dataset == r["Dataset"]) & (win.Subject == r["Subject"]) & (win.Size_Sec == size)
                   & win[f"{key}_shown"]]
        if part.empty:
            continue
        best = part.loc[part[f"{key}_score"].idxmax()]
        rows.append({"Observation": key, "Rank": int(r["Rank"]), **best.to_dict()})
    return rows


# ---------------------------------------------------------------------------
# 2. Worst subjects of every model
# ---------------------------------------------------------------------------

def worst_subjects(win: pd.DataFrame, model: str, sizes: Sequence[int]) -> pd.DataFrame:
    grid = win[win.Grid & win[f"{model}_Score"].notna()]
    rows = []
    for (ds, s), d in grid.groupby(["Dataset", "Subject"]):
        row = {"Dataset": ds, "Subject": s, "Windows": len(d)}
        scores = []
        for size in sizes:
            part = d[d.Size_Sec == size]
            sc = part[f"{model}_Score"].mean() if len(part) else np.nan
            row[f"Score_{size}s"] = sc
            scores.append(sc)
        row["Score"] = np.nanmean(scores)
        others = []
        for m in MODELS:
            if f"{m}_AE" not in d or d[f"{m}_AE"].isna().all():
                continue
            ok = d[f"{m}_AE"].notna()
            row[f"{m}_AE"] = d.loc[ok, f"{m}_AE"].mean()
            row[f"{m}_PCC"] = d.loc[ok, f"{m}_PCC"].mean()
            row[f"{m}_Coverage"] = d.loc[ok, f"{m}_Matched"].sum() / max(d.loc[ok, "GT_Cycles"].sum(), 1)
            row[f"{m}_Score"] = np.nanmean([d.loc[ok & (d.Size_Sec == z), f"{m}_Score"].mean() for z in sizes])
            if m != model:
                others.append(row[f"{m}_Score"])
        row["GT_RR_median"] = d.GT_RR.median()
        row["Others_Best_Score"] = max(others) if others else np.nan
        row["Deficit"] = row["Score"] - row["Others_Best_Score"]
        deficit = row["Deficit"]
        row["Pattern"] = ("model-specific" if np.isfinite(deficit) and deficit <= SPECIFIC_DEFICIT
                          else "others worse" if np.isfinite(deficit) and deficit >= -SPECIFIC_DEFICIT
                          else "similar to others")
        rows.append(row)
    return pd.DataFrame(rows)


def ranked_worst(table: pd.DataFrame, per_dataset: bool, top: int) -> pd.DataFrame:
    groups = table.groupby("Dataset") if per_dataset else [("All", table)]
    listed = []
    for _, g in groups:
        g = g.sort_values("Score").copy()
        g["Rank"] = np.arange(1, len(g) + 1)
        listed.append(g)
    table = pd.concat(listed, ignore_index=True)
    table["Listed"] = table.Rank <= top
    return table


def worst_windows(win: pd.DataFrame, listed: List[dict], model: str, sizes: Sequence[int]) -> List[dict]:
    rows = []
    for r in listed:
        for size in sizes:
            part = win[(win.Dataset == r["Dataset"]) & (win.Subject == r["Subject"]) & (win.Size_Sec == size)
                       & win[f"{model}_Score"].notna()]
            if part.empty:
                continue
            worst = part.loc[part[f"{model}_Score"].idxmin()]
            rows.append({"Model": model, "Rank": int(r["Rank"]), "Pattern": r["Pattern"], **worst.to_dict()})
    return rows


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------

def draw(title: str, row: dict, data: dict, path: Path) -> plt.Figure:
    values, cycles = data["values"], data["cycles"]
    models = [m for m in MODELS if m in values]
    t0, t1 = row["Start_Sec"], row["End_Sec"]
    a, b = int(round(t0 * FS)), int(round(t1 * FS))
    t = np.arange(a, b) / FS
    gt_in = inside(cycles["GT"], t0, t1)
    fig = plt.figure(figsize=(15.5, 1.55 + 1.7 * len(models)))
    h = fig.get_figheight()
    top, bottom, gap = 1 - 0.95 / h, 0.62 / h, 0.25 / h
    ph = (top - bottom - gap * (len(models) - 1)) / len(models)
    for p, model in enumerate(models):
        ax = fig.add_axes([0.05, top - (p + 1) * ph - p * gap, 0.665, ph])
        for c in cycles["GT"]:
            if c["end"] > t0 and c["start"] < t1:
                ax.axvspan(max(c["start"], t0), min(c["transition"], t1), color=GT_BAND, lw=0, zorder=0)
        for c in cycles[model]:
            if c["end"] > t0 and c["start"] < t1:
                ax.fill_between([max(c["start"], t0), min(c["transition"], t1)], -1.42, -1.24,
                                color=ov.MODEL_COLORS[model], alpha=0.8, lw=0)
        _, missed = match_cycles(gt_in, [c for c in cycles[model] if c["end"] > t0 and c["start"] < t1])
        if missed:
            ax.plot([c["start"] for c in missed], [1.28] * len(missed), "v", color=MISSED_COLOR, ms=7, zorder=5)
        ax.plot(t, ov.minmax(values["GT"][a:b]), color=ov.GT_COLOR, lw=2.0, alpha=0.85, zorder=2)
        ax.plot(t, ov.minmax(values[model][a:b]), color=ov.MODEL_COLORS[model], lw=1.6, zorder=3)
        ax.set_xlim(t0, t1)
        ax.set_ylim(-1.5, 1.45)
        ax.set_yticks([])
        ov.style_axis(ax)
        ax.spines["left"].set_visible(False)
        if p < len(models) - 1:
            ax.set_xticklabels([])
        text = (f"{model}\nrate {row[f'{model}_Rate']:5.1f} /min ({row[f'{model}_Ratio']:.2f} x GT)  "
                f"AE {row[f'{model}_AE']:5.2f}\nPCC {row[f'{model}_PCC']:5.2f}   cycles matched "
                f"{int(row[f'{model}_Matched'])}/{int(row['GT_Cycles'])}, extra {int(row[f'{model}_Extra'])}")
        fig.text(0.725, top - p * (ph + gap) - ph / 2, text, family="monospace", fontsize=8.6,
                 color=ov.TEXT_PRIMARY, va="center", ha="left")
    recording = "" if str(row["Recording"]) == str(row["Subject"]) else f" ({row['Recording']})"
    fig.text(0.05, 1 - 0.32 / h, f"{title}: {row['Dataset']} {row['Subject']}{recording}, "
             f"{t0:.0f}-{t1:.0f} s ({t1 - t0:.0f} s)", fontsize=11.5, color=ov.TEXT_PRIMARY, va="center")
    fig.text(0.05, 1 - 0.62 / h, f"GT {row['GT_RR']:.1f} breaths/min, {int(row['GT_Cycles'])} cycles in the window; x axis: time "
             f"in the recording (s); each waveform min-max scaled to [-1, 1] in the window; every model shifted <= half a breath and, "
             f"if needed, inverted once per recording (same rule for all)", fontsize=8.5, color=ov.TEXT_SECONDARY,
             va="center")
    handles = [Line2D([0], [0], color=ov.GT_COLOR, lw=2.0)] + \
              [Line2D([0], [0], color=ov.MODEL_COLORS[m], lw=1.8) for m in models] + \
              [Patch(color=GT_BAND), Patch(color="#8f8e8a"),
               Line2D([0], [0], marker="v", ls="", color=MISSED_COLOR, ms=7)]
    labels = ["GT"] + list(models) + ["GT inspiration", "model inspiration (bar, model colour)",
                                       "GT cycle the model missed"]
    fig.legend(handles, labels, loc="lower center", ncol=len(labels), frameon=False, fontsize=8.8,
               bbox_to_anchor=(0.45, 0.0))
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=130)
    return fig


def draw_pdf(pdf_path: Path, title: str, rows: List[dict], store: dict, png_path, no_png: bool) -> None:
    pdf_path.parent.mkdir(parents=True, exist_ok=True)
    with PdfPages(pdf_path) as pdf:
        for r in rows:
            path = png_path(r)
            fig = draw(title, r, store[(r["Dataset"], r["Subject"], r["Recording"])], path)
            if no_png:
                path.unlink(missing_ok=True)
            pdf.savefig(fig)
            plt.close(fig)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--datasets", nargs="+", choices=list(ov.DATASET_ORDER), default=list(ov.DATASET_ORDER))
    parser.add_argument("--subjects", nargs="+", default=None, help="Only these subjects.")
    parser.add_argument("--sizes", nargs="+", type=int, default=list(WINDOW_SIZES), help="Window sizes (s).")
    parser.add_argument("--step", type=float, default=STEP_SEC, help="Start of a window every this many s.")
    parser.add_argument("--top", type=int, default=TOP, help="1. Subjects listed per observation and dataset.")
    parser.add_argument("--worst-top", type=int, default=WORST_TOP,
                        help="2. Subjects listed per model (per dataset with --worst-per-dataset).")
    parser.add_argument("--worst-per-dataset", action="store_true",
                        help="2. Rank the worst subjects within every dataset instead of pooled.")
    parser.add_argument("--respdiff-root", type=Path, default=ov.RESPDIFF_ROOT)
    parser.add_argument("--respdiff-sampling", default=ov.RESPDIFF_SAMPLING)
    parser.add_argument("--correncoder-root", type=Path, default=ov.CORRENCODER_ROOT)
    parser.add_argument("--rrwavenet-261007-root", type=Path, default=ov.RRWAVENET_261007_ROOT,
                        help="261007 results folder (CapnoBase, full model).")
    parser.add_argument("--rrwavenet-261013-root", type=Path, default=ov.RRWAVENET_261013_ROOT,
                        help="261013 results folder (BIDMC / STEAM2, weight decay selected per fold).")
    parser.add_argument("--rrwavenet-member", choices=["ensemble", "m0", "m1", "m2"], default="ensemble",
                        help="RRWaveNet output: the ensemble (default) or one member.")
    parser.add_argument("--rrwavenet-lowpass-hz", type=float, default=ov.RRWAVENET_LOWPASS_HZ)
    parser.add_argument("--capnobase-path", type=Path, default=None)
    parser.add_argument("--bidmc-path", type=Path, default=None)
    parser.add_argument("--steam2-ppg-dir", type=Path, default=None)
    parser.add_argument("--steam2-rsp-dir", type=Path, default=None)
    parser.add_argument("--output-root", type=Path, default=PROJECT_ROOT)
    parser.add_argument("--no-png", action="store_true", help="Only the PDFs.")
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> None:
    args = parse_args(argv)
    for key, path in (("capnobase", args.capnobase_path), ("bidmc", args.bidmc_path)):
        if path:
            ov.DATASET_CONFIGS[key]["data_path"] = str(path)
    if args.steam2_ppg_dir:
        ov.DATASET_CONFIGS["steam2"]["ppg_dir"] = str(args.steam2_ppg_dir)
    if args.steam2_rsp_dir:
        ov.DATASET_CONFIGS["steam2"]["rsp_dir"] = str(args.steam2_rsp_dir)
    out = Path(args.output_root)
    out.mkdir(parents=True, exist_ok=True)

    store: dict = {}
    rows = []
    for dataset_key in [k for k in ov.DATASET_ORDER if k in args.datasets]:
        rows += dataset_windows(dataset_key, args, store)
    win = pd.DataFrame(rows)
    if win.empty:
        print("No subject with RRWaveNet and another model was found (check the result folders).")
        return
    for size in args.sizes:
        win[win.Size_Sec == size].to_csv(out / f"windows_{size}s.csv", index=False, encoding="utf-8-sig")
    sheets = {}

    # 1. observations
    chosen = []
    for key in OBSERVATIONS:
        table = observation_subjects(win, key, args.sizes, args.top)
        if table.empty:
            continue
        sheets[f"Obs_{key}"] = table
        listed = table[table.Listed].drop_duplicates(["Dataset", "Subject"])
        for size in args.sizes:
            chosen += observation_windows(win, listed, key, size)
    obs_windows = pd.DataFrame(chosen)
    print("\n1. Observations: the listed subjects per dataset and the window drawn")
    for key, (title, _) in OBSERVATIONS.items():
        part = obs_windows[obs_windows.Observation == key] if not obs_windows.empty else obs_windows
        print(f"{key}: {title}")
        if part.empty:
            print("   no subject shows it")
            continue
        for ds, d in part.groupby("Dataset", sort=False):
            for (rank, s), e in d.sort_values(["Rank", "Size_Sec"]).groupby(["Rank", "Subject"], sort=True):
                spans = ", ".join(f"{r['Size_Sec']:.0f} s: {r['Start_Sec']:.0f}-{r['End_Sec']:.0f} s"
                                  for r in e.to_dict("records"))
                print(f"   {ds} #{rank} {s}: {spans}")
            draw_pdf(out / "observations" / f"{key}_{ds}.pdf", title,
                     d.sort_values(["Rank", "Size_Sec"]).to_dict("records"), store,
                     lambda r, key=key, ds=ds: out / "observations" / key / ds /
                     f"{r['Rank']:02d}_{ov.safe_name(r['Subject'])}_{int(r['Size_Sec'])}s.png", args.no_png)
    if not obs_windows.empty:
        sheets["Obs_Windows"] = obs_windows

    # 2. worst subjects
    chosen = []
    print(f"\n2. Worst subjects: lowest score first ({'per dataset' if args.worst_per_dataset else 'pooled'})")
    for model in MODELS:
        if f"{model}_Score" not in win or win[f"{model}_Score"].isna().all():
            continue
        table = ranked_worst(worst_subjects(win, model, args.sizes), args.worst_per_dataset, args.worst_top)
        sheets[f"Worst_{SHORT[model]}"] = table
        listed = table[table.Listed].to_dict("records")
        model_windows = worst_windows(win, listed, model, args.sizes)
        chosen += model_windows
        print(f"{model}")
        for r in listed:
            others = ", ".join(f"{SHORT[m]} {r[f'{m}_Score']:.2f}" for m in MODELS
                               if m != model and np.isfinite(r.get(f"{m}_Score", np.nan)))
            print(f"   #{int(r['Rank']):2d} {r['Dataset']} {r['Subject']}: score {r['Score']:.2f} (AE "
                  f"{r[f'{model}_AE']:.2f}, PCC {r[f'{model}_PCC']:.2f}); others {others}; {r['Pattern']}")
        order = ["Dataset", "Rank", "Size_Sec"] if args.worst_per_dataset else ["Rank", "Size_Sec"]
        if model_windows:
            name = f"worst_{SHORT[model]}"
            draw_pdf(out / "worst" / f"{name}.pdf", f"{model}: lowest score of the model",
                     pd.DataFrame(model_windows).sort_values(order).to_dict("records"), store,
                     lambda r, name=name: out / "worst" / name /
                     f"{r['Rank']:02d}_{r['Dataset']}_{ov.safe_name(r['Subject'])}_{int(r['Size_Sec'])}s.png",
                     args.no_png)
    if chosen:
        sheets["Worst_Windows"] = pd.DataFrame(chosen)

    notes = pd.DataFrame({"Item": [
        "Window score", "Cycles", "RRWaveNet output",
        "1. RRWaveNet right", "1. ce_double", "1. rd_double", "1. rd_missed", "1. rr_coverage", "1. Subjects",
        "1. Windows", "2. Subject score", "2. Deficit", "2. Pattern", "2. Windows"],
        "Meaning": [
            f"PCC - {AE_WEIGHT} x min(rate AE, {AE_CAP})",
            "261030 detector on the aligned waveforms; a GT cycle is matched by a model cycle starting within "
            "half its length (1:1); clipped GT cycles are counted (GT_Clipped_Cycles)",
            f"{args.rrwavenet_member}: CapnoBase from 261007, full model ({args.rrwavenet_261007_root}); "
            f"BIDMC / STEAM2 from 261013, weight decay selected per fold ({args.rrwavenet_261013_root})",
            f"AE <= {RATE_OK_BPM} breath/min",
            f"corr-encoder rate / GT rate {DOUBLE[0]}-{DOUBLE[1]}, RRWaveNet right; score = AE gap",
            f"RespDiff rate / GT rate {DOUBLE[0]}-{DOUBLE[1]}, RRWaveNet right; score = AE gap",
            f"RRWaveNet right and missing <= 1 cycle; RespDiff missing >= {MISSED_MIN} cycles more or at "
            f"{HALF[0]}-{HALF[1]} x the rate; score = cycles missed by RespDiff - by RRWaveNet + 0.1 x AE gap (<= 5)",
            f">= 3 GT cycles; RRWaveNet coverage >= best other + {COVERAGE_MARGIN}; RRWaveNet extra <= 1; "
            "score = coverage margin",
            "non-overlapping windows from 0 s, per size: share of windows showing it and the mean score of those; "
            "ranked once per dataset by the mean over the sizes (Combined_Share, then Combined_Score); the top "
            f"{args.top} with at least one such window are listed",
            f"per listed subject and size, the window (start every {args.step:g} s) with the highest score",
            "mean window score of the model over the non-overlapping windows per size, then the mean over the "
            f"sizes; lowest first, the {args.worst_top} lowest listed"
            + (" per dataset" if args.worst_per_dataset else " (pooled over the datasets)"),
            "the model's subject score - the best subject score of the other models",
            f"model-specific when Deficit <= {SPECIFIC_DEFICIT}, others worse when Deficit >= "
            f"{-SPECIFIC_DEFICIT}, otherwise similar to others",
            f"per listed subject and size, the window (start every {args.step:g} s) with the lowest score of "
            "the model"]})
    with pd.ExcelWriter(out / "discussion_analysis.xlsx") as writer:
        for name, frame in sheets.items():
            frame.round(4).to_excel(writer, sheet_name=name[:31], index=False)
        notes.to_excel(writer, sheet_name="Notes", index=False)
    print(f"\nsaved under {out}")


if __name__ == "__main__":
    main()
