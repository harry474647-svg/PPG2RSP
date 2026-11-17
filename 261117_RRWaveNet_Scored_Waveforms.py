# -*- coding: utf-8 -*-
r"""261117_RRWaveNet_Scored_Waveforms: RRWaveNet waveform reconstruction and the RIIV / RIAV / RIFV spectra of
the PPG, drawn only for the minutes scored with the rescored ground truth (minutes that have a model input,
a model output and a valid ground truth), in 60 s and 30 s windows and in two figure styles.

Model: RRWaveNet as in the final results, the ensemble (mean of the members' stored test outputs):
  CapnoBase      261007_RRWaveNet_Ensemble_NoEvent.py, variant "full"
  BIDMC, STEAM2  261013_RRWaveNet_Adult_SubjectVal_WD.py, the weight decay selected per fold (WD_Selection)
Nothing is trained; the stored outputs of every held-out subject are read.

Scored minutes (rescored ground truth, scored with the minute scoring of the source script through the
helpers of 261113_RRWaveNet_Rescore_ReferenceGT, so the minutes are the ones of the rescored report):
  CapnoBase  "peak": breaths detected by count-orig on the capnogram, RRest agreement rule, expert CO2
             artifact spans excluded
  BIDMC      "ann1": annotator 1 breaths, expert-label rule (--bidmc-gt ann1_nogap: minutes overlapping an
             ann1 gap left out as well)
  STEAM2     "original": the stored ground truth of 261013 (belt breaths, RRest agreement rule)
Only minutes with Ref_Valid = True are drawn and listed; their minute values (GT breaths, RRWaveNet rate,
AE, waveform PCC) are the values of the report. The subjects with an MAE issue are left out as in the final
results (CapnoBase 0115, BIDMC 27, STEAM2 12_YJJ, 22_LMS, 26_CKK; --keep-issue-subjects keeps them).

Windows: 60 s = each scored minute; 30 s = the two halves of each scored minute (--window-sec 60 30).

Input issue check (every 10 s model window inside the scored minutes; these windows passed the data selection,
so the model used them):
  1. Input issue: a window that passed the selection (>= 50% usable PPG and >= 50% structurally valid RSP)
     but still has one of these
       PPG_flat             PPG nearly constant (step <= 1e-4 of the recording's 5-95% range) for >= 1 s
                            (the selection drops a 10 s block only when >= 99.5% of it is flat)
       PPG_low_amplitude    pulse amplitude (0.5-5 Hz, 95th-5th percentile per 2 s) < 25% of the recording's
                            median (the selection has no amplitude check)
       PPG_quality_margin   >= 25% of the window unusable by the source's own PPG quality map (pulse template,
                            beat intervals, expert artifacts; the selection allows up to 50%)
       PPG_artifact_label   expert PPG artifact label inside the window (CapnoBase)
       RSP_collapse         envelope of the reference <= 50% of the recording's median for >= 10 s (CapnoBase
                            also >= 2 breath periods); "severe" when <= 10%
       RSP_quality_low      reference respiratory quality (RRest rule) on < 50% of the window: such windows
                            fail Train_OK, so they were used for testing only
       RSP_structural_part  reference structurally invalid for >= 1 s (the alignment-shift edges excluded)
     PPG is the model input; RSP is the reference (the training target and the GT of the scoring).
  2. Result worse: a scored minute with an issue window is compared with the same subject's issue-free scored
     minutes (needs >= 2). It is worse when
       rate      AE - median AE of the issue-free minutes >= the AE limit, or
       waveform  median |PCC| of the issue-free minutes - |PCC| >= the PCC limit.
     The limits are the 95th percentile of the same differences among the issue-free minutes themselves
     (each against the subject's other issue-free minutes, per dataset): the normal minute-to-minute
     fluctuation. The AE limit is at least 1 /min (the rate resolution of a 60 s minute).
     Likely source: PPG input, reference, or both (from the issue types); for a rate error the GT is also
     checked against the reference's own spectrum (Ref_RR_FFT).
  Each part is reported separately (windows, minutes, subjects); when there is no case, the log says "none".

Figures (each kind and style in its own folder)
  waveform\overlap    GT (black) and RRWaveNet (red) on one axis, each min-max scaled to [-1, 1] in the window
  waveform\separate   (a) GT and (b) RRWaveNet in two rows with a shared time axis, each in its own units
      GT = the continuous respiratory reference of the source script (CapnoBase capnogram, BIDMC impedance,
      STEAM2 belt; band-passed and aligned as in the scoring); RRWaveNet = the ensemble output band-passed
      with the recording's respiratory filter over each run of windows (as scored)
      Title: PCC as scored (stitched target vs. ensemble output in the window; for a 60 s window = the
      minute's reported PCC). Subtitle: the minute's rescored GT, RRWaveNet rate and AE (the 60 s scoring
      unit, also for its 30 s halves) and r of the drawn curves.
  respmod\overlap     (a) GT, RIIV, RIAV, RIFV over time (band-passed, z-scored) on one axis; (b) their |FFT|
  respmod\separate    one row per signal (GT, RIIV, RIAV, RIFV): time series left, |FFT| right
      Drawn with the functions of 261116_PPG_RIIV_RIAV_RIFV_FFT: beats of the PPG model input (0.5-5 Hz pulse
      band), RIIV (peak value), RIAV (peak - preceding trough), RIFV (60 / interval), 4 Hz cubic spline,
      band-passed with the recording's respiratory band over the whole recording, then cut to the window;
      spectrum: detrend, Hamming, zero padding; peak searched in the recording's rate band of the scoring
      profile. A window needs >= 90% series cover and >= 10 used beats (otherwise only the waveform is drawn).
  issues\A_input_issue\{style}   every 60 s / 30 s window with an input issue window inside
  issues\B_result_worse\{style}  the same figures for the minutes judged worse
      overlap: (a) PPG model input, (b) GT and RRWaveNet overlapped; separate: (a) PPG, (b) GT, (c) RRWaveNet.
      Issue windows are shaded and tagged; the issue samples (flat, low amplitude, collapse, invalid reference)
      are marked at the bottom of their panel. Subtitle: minute values, the comparison with the subject's
      issue-free minutes and the issue of each window.

Outputs (D:\PPG2RSP_RRWaveNet_Inspired\261117_RRWaveNet_Scored_Waveforms)
  {kind}\{style}\{Dataset}\{subject}\w{W}s\{subject}[_rec{k}]_{t0}-{t1}s.png       one figure per window
  {kind}\{style}\{Dataset}\{subject}\w{W}s\{Dataset}_{subject}_w{W}s_{kind}_{style}.pdf
  issues\{A_input_issue|B_result_worse}\{style}\{Dataset}\{subject}\w{W}s\...      issue figures and PDFs
  261117_RRWaveNet_Scored_Waveforms.xlsx
    Windows         every drawn window: minute, window, rescored GT, RRWaveNet rate, AE, PCC as scored, r of the
                    drawn curves, RIIV / RIAV / RIFV / GT peak frequencies, input issues, figure paths
    Minutes         every scored minute (the report's minutes) with its input issues and the decision
    Subjects        per subject: scored minutes, MAE, SD, mean PCC, mean |PCC|
    Datasets        per dataset: subjects, minutes, MAE (minute mean, subject mean), PCC
    Issue_Summary   per dataset: windows checked, issue windows per type, issue minutes, worse minutes, limits
    Issue_Windows   1. every 10 s window with an input issue (window unit)
    Issue_Minutes   1. every scored minute with an issue window, compared with the issue-free minutes
    Issue_Subjects  1. every subject with an issue minute: MAE and |PCC| with and without the issue minutes
    Worse_Windows   2. the issue windows of the minutes judged worse
    Worse_Minutes   2. the minutes judged worse
    Worse_Subjects  2. the subjects with a minute judged worse
    Window_Checks   every 10 s window of the scored minutes with all check values
    Missing         subjects without stored results, selection or matching windows
    Notes
  261117_RRWaveNet_Scored_Waveforms_issue_check.txt   the log of the issue check (also printed)

Run: python 261117_RRWaveNet_Scored_Waveforms.py     (Spyder: runfile(...), no options needed)
Options: --window-sec (default 60 30), --kinds (waveform respmod issues), --styles (overlap separate), --datasets,
--subjects CapnoBase:0332 BIDMC:05 ..., --keep-issue-subjects, --bidmc-gt (ann1 | ann1_nogap), --members,
--output-root, --root-261007, --root-261013, --script-261007, --script-261013, --script-261113,
--script-261116, --capnobase-path, --bidmc-path, --steam2-ppg-dir, --steam2-rsp-dir
"""

import argparse
import importlib.util
import json
import math
import re
import sys
import textwrap
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from matplotlib.transforms import blended_transform_factory
from scipy.signal import butter, hilbert, sosfiltfilt

# ---------------------------------------------------------------------------
# Settings
# ---------------------------------------------------------------------------

NAME = "261117_RRWaveNet_Scored_Waveforms"
HERE = Path(__file__).resolve().parent
WORKSPACE = Path(r"D:\PPG2RSP_RRWaveNet_Inspired")
OUTPUT_ROOT = WORKSPACE / NAME
ROOT_261007 = WORKSPACE / "261007_RRWaveNet_NoEvent_v1"
ROOTS_261013 = (WORKSPACE / "261013_RRWavNet_STEAM_BIDMC_v1", WORKSPACE / "261013_RRWaveNet_Adult_SubjectVal_WD")
SCRIPTS = {"261007": "261007_RRWaveNet_Ensemble_NoEvent.py", "261013": "261013_RRWaveNet_Adult_SubjectVal_WD.py",
           "261113": "261113_RRWaveNet_Rescore_ReferenceGT.py", "261116": "261116_PPG_RIIV_RIAV_RIFV_FFT.py"}
WINDOW = "01_w10s_nonoverlap"
DATASETS = ("CapnoBase", "BIDMC", "STEAM2")
KEYS = {"CapnoBase": "capnobase", "BIDMC": "bidmc", "STEAM2": "steam2"}
SOURCE = {"CapnoBase": "261007", "BIDMC": "261013", "STEAM2": "261013"}
GT = {"CapnoBase": "peak", "BIDMC": "ann1", "STEAM2": "original"}
GT_TEXT = {"peak": "capnogram peak detection, RRest rule", "ann1": "BIDMC ann1, expert-label rule",
           "ann1_nogap": "BIDMC ann1 without gap minutes", "original": "stored GT"}
MODALITY = {"CapnoBase": "capnogram", "BIDMC": "impedance", "STEAM2": "belt"}
EXCLUDED = {"STEAM2": ("22_LMS", "26_CKK", "12_YJJ"), "BIDMC": ("27",), "CapnoBase": ("0115",)}
WINDOW_SECS = (60.0, 30.0)
KINDS = ("waveform", "respmod", "issues")
STYLES = ("overlap", "separate")
MEMBERS = 3
EVAL_SEC = 60.0
GT_COLOR, EST_COLOR = "#000000", "#f64636"  # as 261108 / 261116: GT black, RRWaveNet red
INK, MUTED, GRID = "#1f1f1f", "#6b6a66", "#e4e3df"
PPG_COLOR = "#2f5f98"
ISSUE_FILL, ISSUE_MARK = "#f1cf8f", "#9a5b00"  # issue windows (shaded) / issue samples (bar at the panel bottom)

# Input issue check of the 10 s model windows inside the scored minutes
FLAT_MIN_SEC = 1.0           # PPG flat run: nearly constant for >= 1 s
FLAT_TOLERANCE = 1e-4        # step <= 1e-4 of the recording's 5-95% range (as the source's flat check)
PULSE_BAND_HZ = (0.5, 5.0)   # pulse band of the PPG
PULSE_BLOCK_SEC = 2.0        # pulse amplitude per 2 s
PULSE_LOW_RELATIVE = 0.25    # pulse amplitude < 25% of the recording's median
PPG_UNUSABLE_MIN = 0.25      # >= 25% of the window unusable by the source's PPG quality map (selection: < 50%)
RSP_QUALITY_MIN = 0.50       # reference respiratory quality on < 50% of the window (fails Train_OK)
STRUCTURAL_MIN_SEC = 1.0     # reference structurally invalid for >= 1 s
COLLAPSE_SMOOTH_SEC = 2.0    # envelope smoothing
COLLAPSE_LEVELS = (("marked", 0.50), ("severe", 0.10))  # envelope <= 50% / 10% of the recording's median
COLLAPSE_MIN_SEC = 10.0      # for >= 10 s (CapnoBase: also >= 2 breath periods)
WORSE_PERCENTILE = 95.0      # limit = 95th percentile of the fluctuation among issue-free minutes
AE_RESOLUTION = 1.0          # the AE limit is at least 1 /min (rate resolution of a 60 s minute)
MIN_CLEAN_MINUTES = 2        # issue-free minutes a subject needs for the comparison
ISSUES = {  # type: (signal, short tag in the figures, description)
    "PPG_flat": ("PPG", "flat", "PPG nearly constant for >= 1 s"),
    "PPG_low_amplitude": ("PPG", "low amp", "PPG pulse amplitude < 25% of the recording's median"),
    "PPG_quality_margin": ("PPG", "PPG quality", ">= 25% of the window unusable by the source's PPG quality map"),
    "PPG_artifact_label": ("PPG", "PPG artifact", "expert PPG artifact label (CapnoBase)"),
    "RSP_collapse": ("RSP", "collapse", "reference envelope <= 50% of the recording's median for >= 10 s"),
    "RSP_quality_low": ("RSP", "RSP quality", "reference respiratory quality on < 50% of the window (Train_OK = 0)"),
    "RSP_structural_part": ("RSP", "RSP invalid", "reference structurally invalid for >= 1 s"),
}

# ---------------------------------------------------------------------------
# Scripts used as modules
# ---------------------------------------------------------------------------

_MODULES: Dict[str, object] = {}


def find_script(name: str, given: Optional[Path], roots: Sequence[Path]) -> Path:
    """A script: given, written in a run manifest of the results, or found nearby."""
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
        name = "scored_wave_src_" + re.sub(r"\W", "_", path.stem)
        spec = importlib.util.spec_from_file_location(name, key)
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module
        spec.loader.exec_module(module)
        _MODULES[key] = module
    return _MODULES[key]


def root_261013(args) -> Path:
    if args.root_261013:
        return Path(args.root_261013)
    for root in ROOTS_261013:
        if any(root.glob("*/*/BIDMC/Fold_Summaries")) or any(root.glob("*/*/STEAM2/Fold_Summaries")):
            return root
    return next((root for root in ROOTS_261013 if root.is_dir()), ROOTS_261013[0])


# ---------------------------------------------------------------------------
# Signals
# ---------------------------------------------------------------------------

def as_scored(mod, values: np.ndarray, ref: dict) -> np.ndarray:
    """Every run of contiguous windows band-passed with the recording's respiratory filter (as scored)."""
    out = np.full(len(values), np.nan)
    ok = np.isfinite(values)
    padded = np.concatenate([[False], ok, [False]])
    for a, b in np.flatnonzero(padded[1:] != padded[:-1]).reshape(-1, 2):
        out[a:b] = mod.respiratory_filter(values[a:b], mod.TARGET_FS, ref["band_hz"], ref["profile"])
    return out


def minmax(v: np.ndarray) -> np.ndarray:
    v = np.asarray(v, float)
    ok = np.isfinite(v)
    if ok.sum() < 3 or np.ptp(v[ok]) <= 1e-12:
        return np.full_like(v, np.nan)
    return 2.0 * (v - np.nanmin(v)) / (np.nanmax(v) - np.nanmin(v)) - 1.0


def pearson(a: np.ndarray, b: np.ndarray) -> float:
    a, b = np.asarray(a, float), np.asarray(b, float)
    ok = np.isfinite(a) & np.isfinite(b)
    if ok.sum() < 3 or np.std(a[ok]) < 1e-12 or np.std(b[ok]) < 1e-12:
        return math.nan
    return float(np.corrcoef(a[ok], b[ok])[0, 1])


def window_spans(start: float, end: float, lengths: Sequence[float]) -> List[Tuple[float, float, float]]:
    """(length, t0, t1) of every window inside one scored minute: the minute itself for 60 s, its parts otherwise."""
    spans = []
    for w in lengths:
        if w >= end - start - 1e-6:
            spans.append((w, start, end))
            continue
        k = 0
        while start + (k + 1) * w <= end + 1e-6:
            spans.append((w, start + k * w, start + (k + 1) * w))
            k += 1
    return spans


# ---------------------------------------------------------------------------
# Input issue check
# ---------------------------------------------------------------------------

def true_runs(mask: np.ndarray) -> np.ndarray:
    """(start, end) of every run of True (end exclusive)."""
    padded = np.concatenate([[False], np.asarray(mask, dtype=bool), [False]])
    return np.flatnonzero(padded[1:] != padded[:-1]).reshape(-1, 2)


def ppg_checks(ppg: np.ndarray, fs: float) -> Dict[str, np.ndarray]:
    """Flat runs, clipped flat runs and the relative pulse amplitude of one recording's PPG (model input)."""
    x = np.asarray(ppg, dtype=np.float64)
    finite = np.isfinite(x)
    x = np.where(finite, x, np.nanmedian(x) if finite.any() else 0.0)
    span = float(np.percentile(x, 95) - np.percentile(x, 5)) if x.size else 0.0
    still = np.concatenate([np.abs(np.diff(x)) <= max(1e-12, FLAT_TOLERANCE * span), [False]])
    flat = np.zeros(x.size, dtype=bool)
    for a, b in true_runs(still):
        if b - a + 1 >= FLAT_MIN_SEC * fs:  # b - a steps = b - a + 1 samples
            flat[a:b + 1] = True
    full = max(1e-12, float(x.max() - x.min())) if x.size else 1.0
    clipped = flat & ((np.abs(x - x.max()) <= 1e-6 * full) | (np.abs(x - x.min()) <= 1e-6 * full))
    relative = np.full(x.size, np.nan)
    if x.size > 3 * fs:
        sos = butter(3, PULSE_BAND_HZ, btype="band", fs=fs, output="sos")
        pulse = sosfiltfilt(sos, x - x.mean())
        pulse[flat] = np.nan
        n = max(1, int(round(PULSE_BLOCK_SEC * fs)))
        amp = np.full(x.size, np.nan)
        for a in range(0, x.size, n):
            block = pulse[a:a + n]
            block = block[np.isfinite(block)]
            if block.size > 2:
                amp[a:a + n] = np.percentile(block, 95) - np.percentile(block, 5)
        median = np.nanmedian(amp) if np.isfinite(amp).any() else np.nan
        if np.isfinite(median) and median > 0:
            relative = amp / median
    return {"flat": flat, "clipped": clipped, "relative": relative}


def reference_checks(ref: dict, fs: float, breath_rule: bool) -> Dict[str, np.ndarray]:
    """Collapse of the reference (envelope vs. the recording's median) and its structurally invalid samples."""
    x = np.asarray(ref["signal"], dtype=np.float64)
    finite = np.isfinite(x)
    out = {"envelope": np.full(x.size, np.nan)}
    for name, _ in COLLAPSE_LEVELS:
        out[name] = np.zeros(x.size, dtype=bool)
    if finite.sum() > 3 * fs:
        centred = np.where(finite, x - x[finite].mean(), 0.0)
        k = max(1, int(round(COLLAPSE_SMOOTH_SEC * fs)))
        env = np.convolve(np.abs(hilbert(centred)), np.ones(k) / k, mode="same")
        base = float(np.median(env[finite]))
        if base > 0:
            out["envelope"] = env / base
            breaths = np.sort(np.asarray(ref.get("breaths", []), dtype=np.float64))
            period = float(np.median(np.diff(breaths))) / fs if breaths.size > 2 else math.nan
            min_sec = COLLAPSE_MIN_SEC
            if breath_rule and np.isfinite(period):
                min_sec = min(min_sec, 2.0 * period)
            for name, level in COLLAPSE_LEVELS:
                for a, b in true_runs((env <= level * base) & finite):
                    if (b - a) / fs >= min_sec:
                        out[name][a:b] = True
    valid = ref.get("structural_valid")
    invalid = np.zeros(x.size, dtype=bool) if valid is None else ~np.asarray(valid, dtype=bool)[: x.size]
    edge = int(math.ceil(abs(float(ref.get("diagnostics", {}).get("Target_Lag_Sec", 0.0))) * fs)) + 1
    if edge:  # the reference was shifted by the alignment lag; its edges are filled as invalid
        invalid[:edge] = False
        invalid[max(0, invalid.size - edge):] = False
    out["invalid"] = invalid
    return out


def window_issue_types(row: dict) -> List[str]:
    found = []
    if row["PPG_Flat_Sec"] > 0:
        found.append("PPG_flat")
    if row["PPG_Low_Amp_Sec"] > 0:
        found.append("PPG_low_amplitude")
    if row["PPG_Unusable_Fraction"] >= PPG_UNUSABLE_MIN:
        found.append("PPG_quality_margin")
    if row["PPG_Artifact_Label_Sec"] > 0:
        found.append("PPG_artifact_label")
    if row["RSP_Collapse_Sec"] > 0:
        found.append("RSP_collapse")
    if row["RSP_Quality_Fraction"] < RSP_QUALITY_MIN:
        found.append("RSP_quality_low")
    if row["RSP_Invalid_Sec"] >= STRUCTURAL_MIN_SEC:
        found.append("RSP_structural_part")
    return found


def issue_detail(row: dict, kind: str) -> str:
    if kind == "PPG_flat":
        clip = ", clipped" if row["PPG_Clipped_Sec"] > 0 else ""
        return f"PPG flat {row['PPG_Flat_Sec']:.1f} s{clip}"
    if kind == "PPG_low_amplitude":
        return f"PPG amplitude {row['PPG_Min_Relative_Amp']:.0%} of median for {row['PPG_Low_Amp_Sec']:.0f} s"
    if kind == "PPG_quality_margin":
        return f"PPG unusable {row['PPG_Unusable_Fraction'] * row['Window_Sec']:.1f} s"
    if kind == "PPG_artifact_label":
        return f"PPG artifact {row['PPG_Artifact_Label_Sec']:.1f} s"
    if kind == "RSP_collapse":
        level = "severe " if row["RSP_Collapse_Severe_Sec"] > 0 else ""
        return f"RSP {level}collapse {row['RSP_Collapse_Sec']:.1f} s (envelope {row['RSP_Min_Relative_Env']:.0%})"
    if kind == "RSP_quality_low":
        return f"RSP quality {row['RSP_Quality_Fraction']:.0%} of the window"
    return f"RSP invalid {row['RSP_Invalid_Sec']:.1f} s"


def check_windows(dataset: str, subject: str, segment_id: int, starts: Sequence[int], window_len: int, fs: float,
                  quality: Dict[Tuple[int, float], dict], ppg: Dict[str, np.ndarray], rsp: Dict[str, np.ndarray],
                  minute: Tuple[float, float]) -> List[dict]:
    """Check values and issue types of the model windows of one scored minute."""
    rows = []
    window_sec = window_len / fs
    for start in starts:
        a, b = int(start), int(start) + window_len
        q = quality.get((int(segment_id), round(a / fs, 3)), {})
        rel = ppg["relative"][a:b]
        env = rsp["envelope"][a:b]
        row = {"Dataset": dataset, "Subject": subject, "Segment_ID": int(segment_id),
               "Window_Start_Sec": a / fs, "Window_End_Sec": b / fs, "Window_Sec": window_sec,
               "Minute_Start_Sec": minute[0], "Minute_End_Sec": minute[1],
               "Train_OK": int(q.get("Train_OK", 0)),
               "Used_For": "test scoring" + ("; training/validation of the other folds" if q.get("Train_OK", 0) else ""),
               "PPG_Usable_Fraction": float(q.get("PPG_Usable_Proportion", math.nan)),
               "PPG_Unusable_Fraction": 1.0 - float(q.get("PPG_Usable_Proportion", 1.0)),
               "PPG_Flat_Sec": float(ppg["flat"][a:b].sum() / fs),
               "PPG_Clipped_Sec": float(ppg["clipped"][a:b].sum() / fs),
               "PPG_Low_Amp_Sec": float((rel < PULSE_LOW_RELATIVE).sum() / fs),
               "PPG_Min_Relative_Amp": float(np.nanmin(rel)) if np.isfinite(rel).any() else math.nan,
               "PPG_Artifact_Label_Sec": float(q.get("PPG_Artifact_Label_Proportion", 0.0)) * window_sec,
               "RSP_Structural_Fraction": float(q.get("RSP_Usable_Proportion", math.nan)),
               "RSP_Quality_Fraction": float(q.get("RSP_Resp_Usable_Proportion", 1.0)),
               "RSP_Invalid_Sec": float(rsp["invalid"][a:b].sum() / fs),
               "RSP_Collapse_Sec": float(rsp["marked"][a:b].sum() / fs),
               "RSP_Collapse_Severe_Sec": float(rsp["severe"][a:b].sum() / fs),
               "RSP_Min_Relative_Env": float(np.nanmin(env)) if np.isfinite(env).any() else math.nan}
        kinds = window_issue_types(row)
        row["Issue_Types"] = "; ".join(kinds)
        row["Issue_Detail"] = "; ".join(issue_detail(row, k) for k in kinds)
        row["PPG_Issue"] = any(ISSUES[k][0] == "PPG" for k in kinds)
        row["RSP_Issue"] = any(ISSUES[k][0] == "RSP" for k in kinds)
        row["Has_Issue"] = bool(kinds)
        rows.append(row)
    return rows


def judge_minutes(minutes: pd.DataFrame) -> Tuple[pd.DataFrame, dict]:
    """Compare every minute with an issue window with the same subject's issue-free minutes (one dataset)."""
    m = minutes.copy()
    m["AE"] = pd.to_numeric(m["AE_Wave_FFT"], errors="coerce")
    m["Abs_PCC"] = pd.to_numeric(m["Wave_PCC"], errors="coerce").abs()
    dev_ae, dev_pcc = [], []
    for _, g in m.groupby("Subject", sort=False):
        clean = g[~g["Has_Input_Issue"]]
        if len(clean) < MIN_CLEAN_MINUTES:
            continue
        for i in clean.index:  # each issue-free minute against the subject's other issue-free minutes
            others = clean.drop(index=i)
            dev_ae.append(clean.at[i, "AE"] - others["AE"].median())
            dev_pcc.append(others["Abs_PCC"].median() - clean.at[i, "Abs_PCC"])
    dev_ae = np.asarray([v for v in dev_ae if np.isfinite(v)])
    dev_pcc = np.asarray([v for v in dev_pcc if np.isfinite(v)])
    limits = {"AE_Limit": max(AE_RESOLUTION, float(np.percentile(dev_ae, WORSE_PERCENTILE))) if dev_ae.size else math.nan,
              "AE_Fluctuation_P95": float(np.percentile(dev_ae, WORSE_PERCENTILE)) if dev_ae.size else math.nan,
              "PCC_Limit": float(np.percentile(dev_pcc, WORSE_PERCENTILE)) if dev_pcc.size else math.nan,
              "Limit_Minutes": int(dev_ae.size)}
    cols = {"Clean_Minutes": 0, "Clean_Median_AE": math.nan, "Clean_Max_AE": math.nan, "Clean_Median_Abs_PCC": math.nan,
            "AE_Excess": math.nan, "Abs_PCC_Drop": math.nan, "Compared": False, "Worse_Rate": False,
            "Worse_Waveform": False, "Result_Worse": False, "Worse_By": "", "Comparison": ""}
    for k, v in cols.items():
        m[k] = v
    m["Comparison"] = m["Comparison"].astype(object)
    m["Worse_By"] = m["Worse_By"].astype(object)
    for _, g in m.groupby("Subject", sort=False):
        clean = g[~g["Has_Input_Issue"]]
        for i in g.index[g["Has_Input_Issue"]]:
            m.at[i, "Clean_Minutes"] = len(clean)
            if len(clean) < MIN_CLEAN_MINUTES or not np.isfinite(limits["AE_Limit"]):
                m.at[i, "Comparison"] = (f"not compared: {len(clean)} issue-free minute(s) in the subject"
                                         if len(clean) < MIN_CLEAN_MINUTES else "not compared: no limit")
                continue
            ae_c, pcc_c = clean["AE"].median(), clean["Abs_PCC"].median()
            excess, drop = m.at[i, "AE"] - ae_c, pcc_c - m.at[i, "Abs_PCC"]
            worse_rate = bool(np.isfinite(excess) and excess >= limits["AE_Limit"])
            worse_wave = bool(np.isfinite(drop) and np.isfinite(limits["PCC_Limit"]) and drop >= limits["PCC_Limit"])
            by = [n for n, w in (("rate", worse_rate), ("waveform", worse_wave)) if w]
            m.loc[i, ["Clean_Median_AE", "Clean_Max_AE", "Clean_Median_Abs_PCC", "AE_Excess", "Abs_PCC_Drop"]] = [
                ae_c, clean["AE"].max(), pcc_c, excess, drop]
            m.at[i, "Compared"] = True
            m.at[i, "Worse_Rate"], m.at[i, "Worse_Waveform"] = worse_rate, worse_wave
            m.at[i, "Result_Worse"] = bool(by)
            m.at[i, "Worse_By"] = " + ".join(by)
            m.at[i, "Comparison"] = "worse: " + " + ".join(by) if by else "not worse"
    # Likely source of the issue, and for a rate error whether the GT agrees with the reference's own spectrum
    m["Likely_Source"] = ""
    m["GT_Check"] = ""
    for i in m.index[m["Has_Input_Issue"]]:
        ppg_i, rsp_i = bool(m.at[i, "PPG_Input_Issue"]), bool(m.at[i, "RSP_Input_Issue"])
        m.at[i, "Likely_Source"] = ("PPG input + reference" if ppg_i and rsp_i
                                    else "PPG input (model input)" if ppg_i else "reference (training target / GT)")
        gt_rate = float(m.at[i, "GT_Breath_Count"]) * 60.0 / EVAL_SEC
        ref_fft, pred = float(m.at[i, "Ref_RR_FFT"]), float(m.at[i, "Pred_Wave_FFT_RR"])
        if np.isfinite(ref_fft) and np.isfinite(pred):
            if abs(pred - ref_fft) < AE_RESOLUTION <= abs(gt_rate - ref_fft):
                m.at[i, "GT_Check"] = (f"GT {gt_rate:.0f} vs reference spectrum {ref_fft:.1f} /min: RRWaveNet "
                                       f"({pred:.1f}) follows the reference spectrum; the error is on the GT side")
            elif abs(pred - ref_fft) >= AE_RESOLUTION:
                m.at[i, "GT_Check"] = (f"RRWaveNet {pred:.1f} vs reference spectrum {ref_fft:.1f} /min: "
                                       f"the model differs from the reference")
            else:
                m.at[i, "GT_Check"] = f"GT, reference spectrum and RRWaveNet within {AE_RESOLUTION:g} /min"
    return m, limits


# ---------------------------------------------------------------------------
# Waveform figures
# ---------------------------------------------------------------------------

def style_axis(ax) -> None:
    ax.grid(True, color=GRID, lw=0.8)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.tick_params(labelsize=8.5)


def draw_wave_overlap(path: Path, pdf, title: str, subtitle: str, t: np.ndarray, gt: np.ndarray, est: np.ndarray,
                      labels: Tuple[str, str], t0: float, t1: float) -> None:
    fig = plt.figure(figsize=(10.0 if t1 - t0 <= 30 else 13.0, 4.3))
    ax = fig.add_axes([0.07, 0.13, 0.9, 0.64])
    ax.plot(t, minmax(gt), color=GT_COLOR, lw=2.0, zorder=3)
    ax.plot(t, minmax(est), color=EST_COLOR, lw=2.2, zorder=4)
    ax.set_xlim(t0, t1)
    ax.set_ylim(-1.15, 1.15)
    ax.set_xlabel("Time in the recording (s)", fontsize=9)
    ax.set_ylabel("Min-max scaled", fontsize=9)
    style_axis(ax)
    fig.legend([Line2D([], [], color=GT_COLOR, lw=2.0), Line2D([], [], color=EST_COLOR, lw=2.2)], list(labels),
               loc="upper right", bbox_to_anchor=(0.97, 0.985), ncol=2, frameon=False, fontsize=8.5)
    fig.text(0.07, 0.955, title, fontsize=10.5, color=INK, va="top")
    fig.text(0.07, 0.885, subtitle, fontsize=7.6, color=MUTED, va="top", linespacing=1.5)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=160)
    pdf.savefig(fig)
    plt.close(fig)


def draw_wave_separate(path: Path, pdf, title: str, subtitle: str, t: np.ndarray, gt: np.ndarray, est: np.ndarray,
                       labels: Tuple[str, str], t0: float, t1: float) -> None:
    fig = plt.figure(figsize=(10.0 if t1 - t0 <= 30 else 13.0, 6.0))
    ax_g = fig.add_axes([0.08, 0.52, 0.89, 0.29])
    ax_e = fig.add_axes([0.08, 0.1, 0.89, 0.31], sharex=ax_g)
    ax_g.plot(t, gt, color=GT_COLOR, lw=1.8)
    ax_e.plot(t, est, color=EST_COLOR, lw=1.9)
    for ax, letter, label in ((ax_g, "a", labels[0]), (ax_e, "b", labels[1])):
        style_axis(ax)
        ax.set_title(f"({letter}) {label}", loc="left", fontsize=9.5, color=INK, pad=3)
        ax.set_ylabel("a.u.", fontsize=9)
    ax_e.set_xlim(t0, t1)
    ax_e.set_xlabel("Time in the recording (s)", fontsize=9)
    ax_g.tick_params(labelbottom=False)
    fig.text(0.08, 0.965, title, fontsize=10.5, color=INK, va="top")
    fig.text(0.08, 0.92, subtitle, fontsize=7.6, color=MUTED, va="top", linespacing=1.5)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=160)
    pdf.savefig(fig)
    plt.close(fig)


WAVE_DRAW = {"overlap": draw_wave_overlap, "separate": draw_wave_separate}


def mark_issues(ax, windows: Sequence[Tuple[float, float, str, str]], spans: Sequence[Tuple[float, float]],
                tag: Optional[str]) -> None:
    """Shade the issue windows, mark the issue samples at the bottom and name the PPG or RSP issues on top."""
    for w0, w1, ppg_tag, rsp_tag in windows:
        ax.axvspan(w0, w1, color=ISSUE_FILL, alpha=0.45, lw=0, zorder=0)
        label = {"PPG": ppg_tag, "RSP": rsp_tag}.get(tag, "")
        if label:
            ax.text((w0 + w1) / 2.0, 0.97, label, transform=blended_transform_factory(ax.transData, ax.transAxes),
                    ha="center", va="top", fontsize=6.8, color=ISSUE_MARK, linespacing=1.2, zorder=6)
    for a, b in spans:
        ax.axvspan(a, b, ymin=0.0, ymax=0.06, color=ISSUE_MARK, lw=0, zorder=5)


def headroom(ax, values: np.ndarray, top: float) -> None:
    """Room below the trace for the issue-sample bars and (top) above it for the issue tags."""
    v = np.asarray(values, dtype=float)
    v = v[np.isfinite(v)]
    if v.size:
        lo, hi = float(v.min()), float(v.max())
        r = max(hi - lo, 1e-9)
        ax.set_ylim(lo - 0.14 * r, hi + top * r)


def issue_layout(heights: Sequence[float], subtitle: str, width: float):
    """Figure and axes with a header that fits the subtitle (sizes in inches)."""
    lines = subtitle.count("\n") + 1
    header = 0.42 + 0.165 * lines + 0.55  # title, subtitle, legend row and the first panel's title
    gap, bottom = 0.55, 0.6
    height = header + sum(heights) + gap * (len(heights) - 1) + bottom
    fig = plt.figure(figsize=(width, height))
    axes, y = [], height - header
    left = 0.07 if width > 11 else 0.085
    for h in heights:
        y -= h
        axes.append(fig.add_axes([left, y / height, 0.97 - left, h / height],
                                 sharex=axes[0] if axes else None))
        y -= gap
    return fig, axes, left, height, header


def issue_header(fig, title: str, subtitle: str, left: float, height: float, header: float, extra) -> None:
    fig.text(left, 1 - 0.16 / height, title, fontsize=10.5, color=INK, va="top")
    fig.text(left, 1 - 0.46 / height, subtitle, fontsize=7.4, color=MUTED, va="top", linespacing=1.45)
    handles = [h for h, _ in extra] + [Patch(color=ISSUE_FILL, alpha=0.6), Patch(color=ISSUE_MARK)]
    labels = [lab for _, lab in extra] + ["issue window (10 s model input)", "issue samples"]
    fig.legend(handles, labels, loc="lower right", bbox_to_anchor=(0.97, 1 - (header - 0.06) / height),
               ncol=len(handles), frameon=False, fontsize=7.8, handlelength=1.6, columnspacing=1.2)


def issue_targets(fig, targets: Sequence[Tuple[Path, PdfPages]]) -> None:
    for path, pdf in targets:
        path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(path, dpi=160)
        pdf.savefig(fig)
    plt.close(fig)


def draw_issue_overlap(targets, title: str, subtitle: str, item: dict, labels: Tuple[str, str]) -> None:
    t0, t1, t = item["t0"], item["t1"], item["t"]
    fig, (ax_p, ax_o), left, height, header = issue_layout((1.7, 2.2), subtitle, 10.0 if t1 - t0 <= 30 else 13.0)
    ax_p.plot(t, item["ppg"], color=PPG_COLOR, lw=1.0, zorder=3)
    ax_o.plot(t, minmax(item["gt"]), color=GT_COLOR, lw=2.0, zorder=3)
    ax_o.plot(t, minmax(item["est"]), color=EST_COLOR, lw=2.2, zorder=4)
    headroom(ax_p, item["ppg"], 0.5 if any(w[2] for w in item["windows"]) else 0.08)
    ax_o.set_ylim(-1.32, 1.75 if any(w[3] for w in item["windows"]) else 1.15)
    mark_issues(ax_p, item["windows"], item["ppg_spans"], tag="PPG")
    mark_issues(ax_o, item["windows"], item["rsp_spans"], tag="RSP")
    for ax, letter, text, unit in ((ax_p, "a", "PPG (model input)", "a.u."),
                                   (ax_o, "b", f"{labels[0]} and {labels[1]}", "Min-max scaled")):
        style_axis(ax)
        ax.set_title(f"({letter}) {text}", loc="left", fontsize=9.5, color=INK, pad=3)
        ax.set_ylabel(unit, fontsize=9)
    ax_o.set_xlim(t0, t1)
    ax_o.set_xlabel("Time in the recording (s)", fontsize=9)
    ax_p.tick_params(labelbottom=False)
    issue_header(fig, title, subtitle, left, height, header,
                 [(Line2D([], [], color=GT_COLOR, lw=2.0), labels[0]), (Line2D([], [], color=EST_COLOR, lw=2.2), labels[1])])
    issue_targets(fig, targets)


def draw_issue_separate(targets, title: str, subtitle: str, item: dict, labels: Tuple[str, str]) -> None:
    t0, t1, t = item["t0"], item["t1"], item["t"]
    fig, (ax_p, ax_g, ax_e), left, height, header = issue_layout((1.7, 1.6, 1.6), subtitle,
                                                                   10.0 if t1 - t0 <= 30 else 13.0)
    ax_p.plot(t, item["ppg"], color=PPG_COLOR, lw=1.0, zorder=3)
    ax_g.plot(t, item["gt"], color=GT_COLOR, lw=1.8, zorder=3)
    ax_e.plot(t, item["est"], color=EST_COLOR, lw=1.9, zorder=3)
    headroom(ax_p, item["ppg"], 0.5 if any(w[2] for w in item["windows"]) else 0.08)
    headroom(ax_g, item["gt"], 0.5 if any(w[3] for w in item["windows"]) else 0.08)
    mark_issues(ax_p, item["windows"], item["ppg_spans"], tag="PPG")
    mark_issues(ax_g, item["windows"], item["rsp_spans"], tag="RSP")
    mark_issues(ax_e, item["windows"], [], tag=None)
    for ax, letter, text in ((ax_p, "a", "PPG (model input)"), (ax_g, "b", labels[0]), (ax_e, "c", labels[1])):
        style_axis(ax)
        ax.set_title(f"({letter}) {text}", loc="left", fontsize=9.5, color=INK, pad=3)
        ax.set_ylabel("a.u.", fontsize=9)
    ax_e.set_xlim(t0, t1)
    ax_e.set_xlabel("Time in the recording (s)", fontsize=9)
    for ax in (ax_p, ax_g):
        ax.tick_params(labelbottom=False)
    issue_header(fig, title, subtitle, left, height, header, [])
    issue_targets(fig, targets)


ISSUE_DRAW = {"overlap": draw_issue_overlap, "separate": draw_issue_separate}


# ---------------------------------------------------------------------------
# Run
# ---------------------------------------------------------------------------

def same_subject(a: object, b: object) -> bool:
    a, b = str(a).strip(), str(b).strip()
    return a == b or (a.isdigit() and b.isdigit() and int(a) == int(b))


def parse_subjects(items: Optional[Sequence[str]]) -> Dict[str, List[str]]:
    chosen: Dict[str, List[str]] = {}
    for item in items or []:
        if ":" not in item:
            raise ValueError(f"--subjects takes Dataset:Subject, got {item}")
        dataset, subject = item.split(":", 1)
        match = [d for d in DATASETS if d.lower() == dataset.lower()]
        if not match:
            raise ValueError(f"unknown dataset in {item} ({', '.join(DATASETS)})")
        chosen.setdefault(match[0], []).append(subject)
    return chosen


class Pdfs:
    """One PDF per (kind, style, length) of a subject, opened on first use."""

    def __init__(self):
        self.files: Dict[Tuple[str, str, float], PdfPages] = {}

    def get(self, key: Tuple[str, str, float], path: Path) -> PdfPages:
        if key not in self.files:
            path.parent.mkdir(parents=True, exist_ok=True)
            self.files[key] = PdfPages(path)
        return self.files[key]

    def close(self) -> None:
        for pdf in self.files.values():
            pdf.close()
        self.files = {}


def modulation_series(rm, mod, ref: dict, fs: float):
    """RIIV / RIAV / RIFV of the whole recording, band-passed (functions of 261116)."""
    profile = ref["profile"]
    beats = rm.extract_modulations(mod, ref["model_input"], fs, tuple(profile["ppg_hr_range_bpm"]),
                                   ref.get("ppg_artifact_mask"))
    if "times" not in beats or len(beats["times"]) < rm.MIN_WINDOW_BEATS:
        return None
    band = tuple(float(v) for v in ref["band_hz"])
    full = {}
    for k in rm.MODS:
        grid, values = rm.resample(beats["times"], beats["values"][k])
        full[k] = (grid, rm.respiratory_band(values, rm.RESAMPLE_HZ, band))
    return {"beats": beats, "full": full, "band": band}


LOG: List[str] = []


def log(message: str) -> None:
    """Print a line of the issue check and keep it for the log file."""
    print(message, flush=True)
    LOG.append(message)


def spans_in(mask: np.ndarray, a: int, b: int, fs: float) -> List[Tuple[float, float]]:
    return [((a + lo) / fs, (a + hi) / fs) for lo, hi in true_runs(np.asarray(mask[a:b], dtype=bool))]


def flagged(windows: Sequence[dict], signal: str, size: int, fs: float) -> np.ndarray:
    """Samples of the windows with an issue of the signal (PPG or RSP)."""
    mask = np.zeros(size, dtype=bool)
    for w in windows:
        if w[f"{signal}_Issue"]:
            mask[int(round(w["Window_Start_Sec"] * fs)):int(round(w["Window_End_Sec"] * fs))] = True
    return mask


def minute_key(subject: object, segment_id: object, start: object) -> Tuple[str, int, float]:
    return str(subject), int(segment_id), round(float(start), 3)


def issue_kinds(windows: Sequence[dict]) -> List[str]:
    """Issue types found in any of the windows, in the order of ISSUES."""
    found = {k for w in windows for k in w["Issue_Types"].split("; ") if k}
    return [k for k in ISSUES if k in found]


def dataset_rows(args, dataset: str, chosen: Dict[str, List[str]], out_root: Path, rescore, rm) -> dict:
    source = SOURCE[dataset]
    root = Path(args.root_261007) if source == "261007" else root_261013(args)
    script = find_script(SCRIPTS[source], args.script_261007 if source == "261007" else args.script_261013, [root])
    mod = load_script(script)
    for key, field, value in (("capnobase", "data_path", args.capnobase_path), ("bidmc", "data_path", args.bidmc_path),
                              ("steam2", "ppg_dir", args.steam2_ppg_dir), ("steam2", "rsp_dir", args.steam2_rsp_dir)):
        if value:
            mod.DATASET_CONFIGS[key][field] = str(value)
    dkey = KEYS[dataset]
    gt_name = args.bidmc_gt if dataset == "BIDMC" else GT[dataset]
    spec = next(s for s in mod.WINDOW_SPECS if f"{s.order:02d}_{s.name}" == WINDOW)
    print(f"\n{dataset}: results {root}; source {script.name}; GT {gt_name}", flush=True)
    data = mod.load_dataset(dkey, spec)
    ann1 = rescore.read_bidmc_ann1(mod) if gt_name.startswith("ann1") else None
    fs = float(mod.TARGET_FS)
    window_len = int(round(spec.window_sec * fs))
    lengths = sorted({float(w) for w in args.window_sec}, reverse=True)
    excluded = [] if args.keep_issue_subjects else list(EXCLUDED[dataset])
    rows, minutes, missing, checks, pending = [], [], [], [], []
    for subject in sorted(data):
        if dataset in chosen and not any(same_subject(subject, s) for s in chosen[dataset]):
            continue
        if any(same_subject(subject, x) for x in excluded):
            continue
        variant = rescore.final_variant(mod, source, root, spec, dkey, subject)
        if variant is None:
            missing.append({"Dataset": dataset, "Subject": subject, "Missing": "final variant / weight decay selection"})
            continue
        outputs, problem = rescore.stored_outputs(mod, root, variant, spec, dkey, subject, args.members)
        if outputs is None:
            missing.append({"Dataset": dataset, "Subject": subject, "Missing": problem})
            continue
        ens = outputs["ensemble"]
        metadata = mod.test_metadata(data, subject)
        if not (np.array_equal(ens["segment_ids"], [m["segment_id"] for m in metadata])
                and np.array_equal(ens["window_starts"], [m["start"] for m in metadata])):
            missing.append({"Dataset": dataset, "Subject": subject,
                            "Missing": "stored windows differ from the windows of the data read now"})
            continue
        references = data[subject]["references"]
        ann1_labels, spans = None, {}
        if ann1 is not None:
            if subject not in ann1 or len(ann1[subject]) < 2:
                missing.append({"Dataset": dataset, "Subject": subject, "Missing": "no ann1 breaths"})
                continue
            ann1_labels = {sid: rescore.ann1_on_timeline(mod, ann1[subject], ref) for sid, ref in references.items()}
            spans = {sid: rescore.gap_spans(ann1_labels[sid], len(references[sid]["signal"]), rescore.ANN1_GAP_FACTOR)
                     for sid in references}
        refs = rescore.reference_variant(references, "ann1" if gt_name.startswith("ann1") else gt_name, ann1_labels)
        frame = rescore.score(mod, ens, metadata, refs)
        if frame.empty:
            missing.append({"Dataset": dataset, "Subject": subject, "Missing": "no scored minute"})
            continue
        if gt_name == "ann1_nogap":
            gap = rescore.flag_gap_minutes(frame, spans, mod.TARGET_FS)
            frame.loc[gap & frame["Ref_Valid"].astype(bool), "Ref_Exclusion_Reason"] = "ref_ann1_gap"
            frame.loc[gap, "Ref_Valid"] = False
        scored = frame[frame["Ref_Valid"].astype(bool)].copy()
        scored.insert(0, "Subject", str(subject))
        scored.insert(0, "Dataset", dataset)
        scored.insert(2, "GT", gt_name)
        scored.insert(3, "Variant", variant.name)
        quality = {(int(q["Segment_ID"]), round(float(q["Window_Start_Sec"]), 3)): q
                   for q in data[subject]["quality"] if int(q.get("Kept", 0))}
        for col, value in (("Model_Windows", 0), ("Input_Issue_Windows", 0), ("Input_Issue_Types", ""),
                           ("Input_Issue_Detail", ""), ("Has_Input_Issue", False), ("PPG_Input_Issue", False),
                           ("RSP_Input_Issue", False), ("Issue_Windows_Train_OK", 0)):
            scored[col] = value
        tag = mod.safe_tag(subject)
        multi = len(references) > 1
        pdfs = Pdfs()
        drawn = 0
        subject_checks: List[dict] = []
        try:
            for segment_id, seg_minutes in scored.groupby("Segment_ID"):
                ref = refs[int(segment_id)]
                pred, true, _ = mod.reconstruct_segment(ens["predictions"], ens["targets"], metadata, int(segment_id),
                                                         mod.TARGET_FS)
                est = as_scored(mod, pred, ref)
                gt_sig = np.asarray(ref["signal"], dtype=np.float64)
                ppg_sig = np.asarray(ref["model_input"], dtype=np.float64)
                breaths = np.asarray(ref["breaths"], dtype=np.float64) / fs
                ppg_chk = ppg_checks(ppg_sig, fs)
                rsp_chk = reference_checks(ref, fs, breath_rule=dataset == "CapnoBase")
                seg_starts = [int(m["start"]) for m in metadata if int(m["segment_id"]) == int(segment_id)]
                mods = modulation_series(rm, mod, ref, fs) if "respmod" in args.kinds else None
                rate_band = tuple(np.asarray(ref["profile"]["rr_band_bpm"], float) / 60.0)
                max_freq = max(1.0, rate_band[1])
                rec_tag = f"_rec{segment_id}" if multi else ""
                rec_text = f" · recording {segment_id}" if multi else ""
                for index, m in seg_minutes.iterrows():
                    ma, mb = float(m["RR_Window_Start_Sec"]), float(m["RR_Window_End_Sec"])
                    gt_count, rate = float(m["GT_Breath_Count"]), float(m["Pred_Wave_FFT_RR"])
                    ae, pcc_min = float(m["AE_Wave_FFT"]), float(m["Wave_PCC"])
                    # 1. input issues of the model windows of this scored minute
                    a_min, b_min = int(round(ma * fs)), int(round(mb * fs))
                    win = check_windows(dataset, str(subject), int(segment_id),
                                        [s for s in seg_starts if a_min <= s < b_min], window_len, fs, quality,
                                        ppg_chk, rsp_chk, (ma, mb))
                    subject_checks += win
                    issue_win = [w for w in win if w["Has_Issue"]]
                    kinds = issue_kinds(issue_win)
                    scored.loc[index, "Model_Windows"] = len(win)
                    scored.loc[index, "Input_Issue_Windows"] = len(issue_win)
                    scored.loc[index, "Input_Issue_Types"] = "; ".join(kinds)
                    scored.loc[index, "Input_Issue_Detail"] = " | ".join(
                        f"{w['Window_Start_Sec']:.0f}-{w['Window_End_Sec']:.0f} s: {w['Issue_Detail']}" for w in issue_win)
                    scored.loc[index, "Has_Input_Issue"] = bool(issue_win)
                    scored.loc[index, "PPG_Input_Issue"] = any(w["PPG_Issue"] for w in issue_win)
                    scored.loc[index, "RSP_Input_Issue"] = any(w["RSP_Issue"] for w in issue_win)
                    scored.loc[index, "Issue_Windows_Train_OK"] = sum(w["Train_OK"] for w in issue_win)
                    for length, t0, t1 in window_spans(ma, mb, lengths):
                        a, b = int(round(t0 * fs)), int(round(t1 * fs))
                        t = np.arange(a, b) / fs
                        n = min(b, len(gt_sig), len(est))
                        g_w, e_w = gt_sig[a:n], est[a:n]
                        pcc = pearson(true[a:n], pred[a:n])  # as scored: stitched target vs. ensemble output
                        r_drawn = pearson(g_w, e_w)
                        stem = f"{tag}{rec_tag}_{t0:.0f}-{t1:.0f}s"
                        in_span = [w for w in issue_win if w["Window_Start_Sec"] < t1 - 1e-6
                                   and w["Window_End_Sec"] > t0 + 1e-6]
                        row = {"Dataset": dataset, "Subject": subject, "Segment_ID": int(segment_id), "GT": gt_name,
                               "Variant": variant.name, "Window_Sec": length, "Start_Sec": t0, "End_Sec": t1,
                               "Minute_Start_Sec": ma, "Minute_End_Sec": mb, "GT_Breaths_Minute": gt_count,
                               "RRWaveNet_Rate_Minute": rate, "AE_Minute": ae, "PCC_Minute_Reported": pcc_min,
                               "PCC_As_Scored_Window": pcc, "r_Drawn_Curves": r_drawn,
                               "Input_Issue_Windows": len(in_span),
                               "Input_Issue_Types": "; ".join(issue_kinds(in_span))}
                        title = f"{dataset} {subject}{rec_text} · {t0:.0f}-{t1:.0f} s · PCC (as scored) = {pcc:.2f}"
                        subtitle = (f"scored minute {ma:.0f}-{mb:.0f} s: GT {gt_count:.0f} /min ({GT_TEXT[gt_name]}), "
                                    f"RRWaveNet {rate:.1f} /min, AE {ae:.2f}\n"
                                    f"r of the drawn curves {r_drawn:.2f} · RRWaveNet ensemble ({source} {variant.name}), "
                                    f"band-passed as scored")
                        labels = (f"GT ({MODALITY[dataset]})", "RRWaveNet")
                        if "waveform" in args.kinds:
                            for st in args.styles:
                                folder = out_root / "waveform" / st / dataset / tag / f"w{length:g}s"
                                pdf = pdfs.get(("waveform", st, length),
                                               folder / f"{dataset}_{tag}_w{length:g}s_waveform_{st}.pdf")
                                WAVE_DRAW[st](folder / f"{stem}.png", pdf, title, subtitle, t[: n - a], g_w, e_w,
                                              labels, t0, t1)
                                row[f"Figure_Waveform_{st.capitalize()}"] = str((folder / f"{stem}.png").relative_to(out_root))
                        if "issues" in args.kinds and in_span:
                            # drawn after the dataset's minutes are judged (the folder depends on the decision)
                            pending.append({
                                "key": minute_key(subject, segment_id, ma), "subject": subject, "tag": tag,
                                "stem": stem, "length": length, "rec_text": rec_text, "variant": variant.name,
                                "t0": t0, "t1": t1, "t": t[: n - a].astype(np.float32),
                                "ppg": ppg_sig[a:n].astype(np.float32), "gt": g_w.astype(np.float32),
                                "est": e_w.astype(np.float32), "kinds": issue_kinds(in_span),
                                "windows": [(w["Window_Start_Sec"], w["Window_End_Sec"],
                                             "\n".join(ISSUES[k][1] for k in issue_kinds([w]) if ISSUES[k][0] == "PPG"),
                                             "\n".join(ISSUES[k][1] for k in issue_kinds([w]) if ISSUES[k][0] == "RSP"))
                                            for w in in_span],
                                "details": [f"{w['Window_Start_Sec']:.0f}-{w['Window_End_Sec']:.0f} s "
                                            f"{w['Issue_Detail']}" for w in in_span],
                                "ppg_spans": spans_in((ppg_chk["flat"] | (ppg_chk["relative"] < PULSE_LOW_RELATIVE))
                                                      & flagged(in_span, "PPG", ppg_sig.size, fs), a, n, fs),
                                "rsp_spans": spans_in((rsp_chk["marked"] | rsp_chk["invalid"])
                                                      & flagged(in_span, "RSP", gt_sig.size, fs), a, n, fs),
                                "minute": (ma, mb, gt_count, rate, ae, pcc_min), "labels": labels})
                        if mods is not None:
                            times = mods["beats"]["times"]
                            in_window = (times >= t0) & (times < t1)
                            series = {}
                            for k in rm.MODS:
                                grid, values = mods["full"][k]
                                keep = (grid >= t0) & (grid < t1)
                                series[k] = (grid[keep], values[keep])
                            cover = min(len(series[k][0]) for k in rm.MODS) / (rm.RESAMPLE_HZ * (t1 - t0))
                            if int(in_window.sum()) >= rm.MIN_WINDOW_BEATS and cover >= rm.MIN_WINDOW_COVER:
                                spectra = {}
                                for k in rm.MODS:
                                    freq, mag = rm.spectrum(series[k][1], rm.RESAMPLE_HZ)
                                    peak_hz, peak_mag = rm.band_peak(freq, mag, rate_band)
                                    spectra[k] = {"freq": freq, "mag": mag, "peak_hz": peak_hz, "peak_mag": peak_mag}
                                    row[f"{k}_Peak_Hz"] = peak_hz
                                ok = np.isfinite(g_w)
                                gt_freq, gt_mag = rm.spectrum(g_w[ok], fs)
                                gt_peak_hz, gt_peak_mag = rm.band_peak(gt_freq, gt_mag, rate_band)
                                gt_info = {"label": f"GT ({MODALITY[dataset]})", "series": (t[: n - a][ok], g_w[ok]),
                                           "freq": gt_freq, "mag": gt_mag, "peak_hz": gt_peak_hz, "peak_mag": gt_peak_mag}
                                gt_rate_hz = float(((breaths >= t0) & (breaths < t1)).sum()) / (t1 - t0)
                                row.update({"GT_Ref_Peak_Hz": gt_peak_hz, "GT_Rate_Window_Hz": gt_rate_hz,
                                            "Beats_Used": int(in_window.sum())})
                                pulse = float(np.mean(mods["beats"]["values"]["RIFV"][in_window]))
                                header = (f"{dataset} {subject}{rec_text}: respiratory modulations of the PPG · "
                                          f"{length:g} s window {t0:.0f}-{t1:.0f} s · {int(in_window.sum())} beats, "
                                          f"mean pulse rate {pulse:.0f} /min",
                                          f"scored minute {ma:.0f}-{mb:.0f} s: GT {gt_count:.0f} /min ({GT_TEXT[gt_name]}), "
                                          f"RRWaveNet {rate:.1f} /min; PPG = model input; series at {rm.RESAMPLE_HZ:g} Hz, "
                                          f"band-passed {mods['band'][0]:.2f}-{mods['band'][1]:.2f} Hz; peaks in "
                                          f"{rate_band[0]:.2f}-{rate_band[1]:.2f} Hz")
                                for st in args.styles:
                                    folder = out_root / "respmod" / st / dataset / tag / f"w{length:g}s"
                                    pdf = pdfs.get(("respmod", st, length),
                                                   folder / f"{dataset}_{tag}_w{length:g}s_respmod_{st}.pdf")
                                    rm.DRAW[st](folder / f"{stem}_RIIV_RIAV_RIFV.png", pdf, header, series, spectra,
                                                gt_info, gt_rate_hz, rate_band, max_freq, f"{length:g} s window")
                                    row[f"Figure_Respmod_{st.capitalize()}"] = str(
                                        (folder / f"{stem}_RIIV_RIAV_RIFV.png").relative_to(out_root))
                            else:
                                row["Respmod_Skipped"] = (f"{int(in_window.sum())} usable beats, "
                                                          f"series cover {cover:.0%}")
                        elif "respmod" in args.kinds:
                            row["Respmod_Skipped"] = "too few usable beats in the recording"
                        rows.append(row)
                        drawn += 1
        finally:
            pdfs.close()
        minutes.append(scored)
        checks += subject_checks
        n_issue = sum(w["Has_Issue"] for w in subject_checks)
        print(f"  {subject}: {len(scored)} scored minutes, {drawn} windows drawn; input issue windows {n_issue} "
              f"in {int(scored['Has_Input_Issue'].sum())} minutes", flush=True)
    del data
    if not minutes:
        return {"rows": rows, "minutes": [], "missing": missing, "checks": checks, "limits": {}}

    # 2. result worse: every minute with an issue window against the subject's issue-free minutes
    judged, limits = judge_minutes(pd.concat(minutes, ignore_index=True))
    decisions = {minute_key(r["Subject"], r["Segment_ID"], r["RR_Window_Start_Sec"]): r
                 for r in judged.to_dict("records")}
    for row in rows:
        d = decisions.get(minute_key(row["Subject"], row["Segment_ID"], row["Minute_Start_Sec"]))
        if d is not None and d["Has_Input_Issue"]:
            row["Minute_Result_Worse"] = bool(d["Result_Worse"])
            row["Minute_Comparison"] = d["Comparison"]

    # issue figures: A = every window with an input issue, B = the minutes judged worse
    figures: Dict[Tuple[str, int, float], Dict[str, str]] = {}
    by_subject: Dict[str, List[dict]] = {}
    for item in pending:
        by_subject.setdefault(str(item["subject"]), []).append(item)
    for subject, items in by_subject.items():
        pdfs = Pdfs()
        try:
            for item in items:
                d = decisions[item["key"]]
                ma, mb, gt_count, rate, ae, pcc_min = item["minute"]
                title = (f"{dataset} {subject}{item['rec_text']} · {item['t0']:.0f}-{item['t1']:.0f} s · input issue: "
                         + ", ".join(k.replace("_", " ") for k in item["kinds"]))
                if d["Compared"]:
                    verdict = (f"vs. the subject's issue-free minutes (n = {int(d['Clean_Minutes'])}, median AE "
                               f"{d['Clean_Median_AE']:.2f}, median |PCC| {d['Clean_Median_Abs_PCC']:.2f}): AE "
                               f"{d['AE_Excess']:+.2f} (worse if >= +{limits['AE_Limit']:.2f}), |PCC| "
                               f"{-d['Abs_PCC_Drop']:+.2f} (worse if <= -{limits['PCC_Limit']:.2f}) → "
                               + ("RESULT WORSE (" + d["Worse_By"] + ")" if d["Result_Worse"] else "not worse"))
                else:
                    verdict = d["Comparison"]
                source_text = d["Likely_Source"] + (f"; {d['GT_Check']}" if d["Worse_Rate"] and d["GT_Check"] else "")
                width = 150 if item["t1"] - item["t0"] <= 30 else 200
                subtitle = "\n".join(sum((textwrap.wrap(text, width, subsequent_indent="    ") for text in (
                    f"scored minute {ma:.0f}-{mb:.0f} s: GT {gt_count:.0f} /min ({GT_TEXT[gt_name]}), RRWaveNet "
                    f"{rate:.1f} /min, AE {ae:.2f}, PCC {pcc_min:.2f} · RRWaveNet ensemble ({source} {item['variant']})",
                    verdict, "likely source: " + source_text,
                    "issue windows: " + " | ".join(item["details"]))), []))
                groups = ["A_input_issue"] + (["B_result_worse"] if d["Result_Worse"] else [])
                for st in args.styles:
                    targets = []
                    for group in groups:
                        folder = out_root / "issues" / group / st / dataset / item["tag"] / f"w{item['length']:g}s"
                        pdf = pdfs.get((group, st, item["length"]), folder / (
                            f"{dataset}_{item['tag']}_w{item['length']:g}s_issues_{group[0]}_{st}.pdf"))
                        targets.append((folder / f"{item['stem']}_issue.png", pdf))
                    ISSUE_DRAW[st](targets, title, subtitle, item, item["labels"])
                    if item["length"] >= EVAL_SEC - 1e-6:
                        figures.setdefault(item["key"], {})[f"Figure_Issue_{st.capitalize()}"] = str(
                            targets[0][0].relative_to(out_root))
        finally:
            pdfs.close()
    for col in ("Figure_Issue_Overlap", "Figure_Issue_Separate"):
        judged[col] = [figures.get(minute_key(r["Subject"], r["Segment_ID"], r["RR_Window_Start_Sec"]), {}).get(col, "")
                       for r in judged.to_dict("records")]
    for row in checks:
        d = decisions.get(minute_key(row["Subject"], row["Segment_ID"], row["Minute_Start_Sec"]))
        row.update({"Minute_AE": d["AE"], "Minute_Abs_PCC": d["Abs_PCC"], "Minute_Compared": d["Compared"],
                    "Minute_Result_Worse": bool(d["Result_Worse"]), "Minute_Worse_By": d["Worse_By"],
                    "Minute_Comparison": d["Comparison"], "Likely_Source": d["Likely_Source"]})
    report_dataset(dataset, judged, pd.DataFrame(checks), limits)
    return {"rows": rows, "minutes": [judged], "missing": missing, "checks": checks, "limits": limits}


def report_dataset(dataset: str, judged: pd.DataFrame, checks: pd.DataFrame, limits: dict) -> None:
    """Log lines of the issue check of one dataset ("none" when there is no case)."""
    log(f"\n[Issue check] {dataset}: {len(checks)} model windows (10 s) in {len(judged)} scored minutes of "
        f"{judged['Subject'].nunique()} subjects checked")
    issue = checks[checks["Has_Issue"]] if not checks.empty else checks
    if issue.empty:
        log(f"[1. Input issue] {dataset}: none - no model window of the scored minutes has an input issue")
    else:
        im = judged[judged["Has_Input_Issue"]]
        log(f"[1. Input issue] {dataset}: {len(issue)} windows ({len(issue) / len(checks):.1%}) in {len(im)} minutes of "
            f"{issue['Subject'].nunique()} subjects; {int(issue['Train_OK'].sum())} of them were also used for "
            f"training/validation of the other folds")
        for k in ISSUES:
            g = issue[issue["Issue_Types"].str.split("; ").apply(lambda v, k=k: k in v)]
            if len(g):
                log(f"    {k:20s} {len(g):4d} windows, {g['Subject'].nunique():3d} subjects  ({ISSUES[k][2]})")
    if not np.isfinite(limits.get("AE_Limit", math.nan)):
        log(f"[2. Result worse] {dataset}: no limit (no subject with >= {MIN_CLEAN_MINUTES} issue-free minutes)")
        return
    note = "" if limits["Limit_Minutes"] >= 20 else " - few minutes: run every subject for the reported limits"
    log(f"[2. Result worse] {dataset}: limits AE +{limits['AE_Limit']:.2f} /min, |PCC| -{limits['PCC_Limit']:.2f} "
        f"({WORSE_PERCENTILE:g}th percentile of {limits['Limit_Minutes']} issue-free minutes){note}")
    if issue.empty:
        log(f"[2. Result worse] {dataset}: none - there is no issue minute to compare")
        return
    im = judged[judged["Has_Input_Issue"]]
    worse = im[im["Result_Worse"]]
    skipped = int((~im["Compared"]).sum())
    if worse.empty:
        log(f"[2. Result worse] {dataset}: none - no issue minute is worse than the subject's issue-free minutes "
            f"beyond the normal fluctuation ({int(im['Compared'].sum())} compared, {skipped} not compared)")
        return
    log(f"[2. Result worse] {dataset}: {len(worse)} of {int(im['Compared'].sum())} compared issue minutes in "
        f"{worse['Subject'].nunique()} subjects (rate {int(worse['Worse_Rate'].sum())}, waveform "
        f"{int(worse['Worse_Waveform'].sum())}); {skipped} issue minutes not compared")
    for _, r in worse.iterrows():
        log(f"    {r['Subject']} {r['RR_Window_Start_Sec']:.0f}-{r['RR_Window_End_Sec']:.0f} s: {r['Worse_By']}; "
            f"AE {r['AE']:.2f} (issue-free {r['Clean_Median_AE']:.2f}), |PCC| {r['Abs_PCC']:.2f} "
            f"(issue-free {r['Clean_Median_Abs_PCC']:.2f}); {r['Input_Issue_Types']}; {r['Likely_Source']}"
            + (f"; {r['GT_Check']}" if r["Worse_Rate"] and r["GT_Check"] else ""))


def summaries(minutes: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    m = minutes.copy()
    m["AE"] = m["AE_Wave_FFT"].astype(float)
    m["Abs_PCC"] = m["Wave_PCC"].abs()
    subj = m.groupby(["Dataset", "Subject"], sort=False).agg(
        GT=("GT", "first"), Scored_Minutes=("AE", "size"), MAE=("AE", "mean"), AE_SD=("AE", "std"),
        Mean_PCC=("Wave_PCC", "mean"), Mean_Abs_PCC=("Abs_PCC", "mean")).reset_index()
    ds = []
    for dataset, g in m.groupby("Dataset", sort=False):
        s = subj[subj.Dataset == dataset]
        ds.append({"Dataset": dataset, "GT": g["GT"].iloc[0], "Subjects": int(len(s)), "Scored_Minutes": int(len(g)),
                   "MAE_Minutes": float(g.AE.mean()), "SD_Minutes": float(g.AE.std()),
                   "MAE_Subject_Mean": float(s.MAE.mean()), "MAE_Subject_SD": float(s.MAE.std()),
                   "PCC_Minutes": float(g.Wave_PCC.mean()), "Abs_PCC_Minutes": float(g.Abs_PCC.mean())})
    return subj, pd.DataFrame(ds)


MINUTE_COLUMNS = ["Dataset", "Subject", "Segment_ID", "RR_Window_Start_Sec", "RR_Window_End_Sec", "GT", "Variant",
                  "Model_Windows", "Input_Issue_Windows", "Input_Issue_Types", "Input_Issue_Detail", "Likely_Source",
                  "Issue_Windows_Train_OK", "GT_Breath_Count", "Pred_Wave_FFT_RR", "Ref_RR_FFT", "AE", "Wave_PCC",
                  "Abs_PCC", "Clean_Minutes", "Clean_Median_AE", "Clean_Max_AE", "Clean_Median_Abs_PCC", "AE_Excess",
                  "AE_Limit", "Abs_PCC_Drop", "PCC_Limit", "Compared", "Worse_Rate", "Worse_Waveform", "Result_Worse",
                  "Worse_By", "Comparison", "GT_Check", "Figure_Issue_Overlap", "Figure_Issue_Separate"]
WINDOW_COLUMNS = ["Dataset", "Subject", "Segment_ID", "Window_Start_Sec", "Window_End_Sec", "Minute_Start_Sec",
                  "Minute_End_Sec", "Issue_Types", "Issue_Detail", "PPG_Issue", "RSP_Issue", "Train_OK", "Used_For",
                  "Minute_AE", "Minute_Abs_PCC", "Minute_Compared", "Minute_Result_Worse", "Minute_Worse_By",
                  "Minute_Comparison", "Likely_Source", "PPG_Usable_Fraction", "PPG_Unusable_Fraction", "PPG_Flat_Sec",
                  "PPG_Clipped_Sec", "PPG_Low_Amp_Sec", "PPG_Min_Relative_Amp", "PPG_Artifact_Label_Sec",
                  "RSP_Structural_Fraction", "RSP_Quality_Fraction", "RSP_Invalid_Sec", "RSP_Collapse_Sec",
                  "RSP_Collapse_Severe_Sec", "RSP_Min_Relative_Env", "Has_Issue"]


def or_none(frame: pd.DataFrame, message: str) -> pd.DataFrame:
    """The table, or one row saying that there is no case."""
    return frame if not frame.empty else pd.DataFrame({"Result": [message]})


def issue_tables(minutes: pd.DataFrame, checks: pd.DataFrame, limits: Dict[str, dict]) -> Dict[str, pd.DataFrame]:
    """Sheets of the issue check: summary, issue windows / minutes / subjects and the worse subsets."""
    m = minutes.copy()
    for dataset, lim in limits.items():
        m.loc[m["Dataset"] == dataset, "AE_Limit"] = lim.get("AE_Limit", math.nan)
        m.loc[m["Dataset"] == dataset, "PCC_Limit"] = lim.get("PCC_Limit", math.nan)
    im = m[m["Has_Input_Issue"]]
    issue_minutes = im[[c for c in MINUTE_COLUMNS if c in im.columns]]
    worse_minutes = issue_minutes[issue_minutes["Result_Worse"].astype(bool)]
    windows = checks[checks["Has_Issue"]] if not checks.empty else checks
    issue_windows = windows[[c for c in WINDOW_COLUMNS if c in windows.columns]]
    worse_windows = issue_windows[issue_windows["Minute_Result_Worse"].astype(bool)] if not windows.empty else windows
    subjects = []
    for (dataset, subject), g in m.groupby(["Dataset", "Subject"], sort=False):
        issue = g[g["Has_Input_Issue"]]
        if issue.empty:
            continue
        clean = g[~g["Has_Input_Issue"]]
        worse = issue[issue["Result_Worse"]]
        subjects.append({
            "Dataset": dataset, "Subject": subject, "Scored_Minutes": len(g), "Issue_Minutes": len(issue),
            "Issue_Free_Minutes": len(clean), "Compared_Minutes": int(issue["Compared"].sum()),
            "Worse_Minutes": len(worse), "Worse_Rate_Minutes": int(issue["Worse_Rate"].sum()),
            "Worse_Waveform_Minutes": int(issue["Worse_Waveform"].sum()),
            "Issue_Types": "; ".join(k for k in ISSUES if issue["Input_Issue_Types"].str.contains(k, regex=False).any()),
            "Likely_Source": "; ".join(sorted(set(issue["Likely_Source"]))),
            "MAE_All": g["AE"].mean(), "MAE_Without_Issue_Minutes": clean["AE"].mean(),
            "MAE_Change": clean["AE"].mean() - g["AE"].mean(),
            "Mean_Abs_PCC_All": g["Abs_PCC"].mean(), "Mean_Abs_PCC_Without_Issue_Minutes": clean["Abs_PCC"].mean(),
            "Abs_PCC_Change": clean["Abs_PCC"].mean() - g["Abs_PCC"].mean(),
            "Issue_Minutes_List": "; ".join(f"{r.RR_Window_Start_Sec:.0f}-{r.RR_Window_End_Sec:.0f} s"
                                            for r in issue.itertuples()),
            "Worse_Minutes_List": "; ".join(f"{r.RR_Window_Start_Sec:.0f}-{r.RR_Window_End_Sec:.0f} s ({r.Worse_By})"
                                            for r in worse.itertuples())})
    subjects = pd.DataFrame(subjects)
    worse_subjects = subjects[subjects["Worse_Minutes"] > 0] if not subjects.empty else subjects
    summary = []
    for dataset, g in m.groupby("Dataset", sort=False):
        c = checks[checks["Dataset"] == dataset] if not checks.empty else checks
        ci = c[c["Has_Issue"]] if not c.empty else c
        issue = g[g["Has_Input_Issue"]]
        worse = issue[issue["Result_Worse"]]
        subj_all = g.groupby("Subject")["AE"].mean()
        subj_clean = g[~g["Has_Input_Issue"]].groupby("Subject")["AE"].mean()
        pcc_all = g.groupby("Subject")["Abs_PCC"].mean()
        pcc_clean = g[~g["Has_Input_Issue"]].groupby("Subject")["Abs_PCC"].mean()
        row = {"Dataset": dataset, "Subjects": g["Subject"].nunique(), "Scored_Minutes": len(g),
               "Windows_Checked": len(c), "Issue_Windows": len(ci),
               "Issue_Window_Fraction": len(ci) / len(c) if len(c) else math.nan,
               "Issue_Windows_Also_Train_OK": int(ci["Train_OK"].sum()) if len(ci) else 0}
        for k in ISSUES:
            row[f"Windows_{k}"] = int(ci["Issue_Types"].str.split("; ").apply(lambda v, k=k: k in v).sum()) if len(ci) else 0
        row.update({"Issue_Minutes": len(issue), "Issue_Subjects": issue["Subject"].nunique(),
                    "Compared_Minutes": int(issue["Compared"].sum()), "Worse_Minutes": len(worse),
                    "Worse_Rate_Minutes": int(issue["Worse_Rate"].sum()),
                    "Worse_Waveform_Minutes": int(issue["Worse_Waveform"].sum()),
                    "Worse_Subjects": worse["Subject"].nunique(),
                    "AE_Limit": limits.get(dataset, {}).get("AE_Limit", math.nan),
                    "AE_Fluctuation_P95": limits.get(dataset, {}).get("AE_Fluctuation_P95", math.nan),
                    "PCC_Limit": limits.get(dataset, {}).get("PCC_Limit", math.nan),
                    "Limit_Minutes": limits.get(dataset, {}).get("Limit_Minutes", 0),
                    "MAE_Subject_Mean_All": subj_all.mean(),
                    "MAE_Subject_Mean_Without_Issue_Minutes": subj_clean.mean(),
                    "Abs_PCC_Subject_Mean_All": pcc_all.mean(),
                    "Abs_PCC_Subject_Mean_Without_Issue_Minutes": pcc_clean.mean(),
                    "Input_Issue": "none" if ci.empty else f"{len(ci)} windows in {len(issue)} minutes",
                    "Result_Worse": "none" if worse.empty else f"{len(worse)} minutes in {worse['Subject'].nunique()} subjects"})
        summary.append(row)
    none_issue = "none: no 10 s model window of the scored minutes has an input issue"
    none_worse = ("none: no issue minute is worse than the subject's issue-free minutes beyond the normal fluctuation"
                  if not im.empty else "none: there is no issue minute to compare")
    return {"Issue_Summary": pd.DataFrame(summary),
            "Issue_Windows": or_none(issue_windows, none_issue),
            "Issue_Minutes": or_none(issue_minutes, none_issue),
            "Issue_Subjects": or_none(subjects, none_issue),
            "Worse_Windows": or_none(worse_windows, none_worse),
            "Worse_Minutes": or_none(worse_minutes, none_worse),
            "Worse_Subjects": or_none(worse_subjects, none_worse),
            "Window_Checks": checks[[c for c in WINDOW_COLUMNS if c in checks.columns]] if not checks.empty else checks}


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="RRWaveNet waveforms and RIIV / RIAV / RIFV spectra of the scored minutes")
    p.add_argument("--window-sec", nargs="+", type=float, default=list(WINDOW_SECS),
                   help="window lengths in s (default 60 30): 60 = the scored minute, shorter = its parts")
    p.add_argument("--kinds", nargs="+", choices=list(KINDS), default=list(KINDS))
    p.add_argument("--styles", nargs="+", choices=list(STYLES), default=list(STYLES))
    p.add_argument("--datasets", nargs="+", choices=list(DATASETS), default=list(DATASETS))
    p.add_argument("--subjects", nargs="+", default=None, help="Dataset:Subject (default: every subject)")
    p.add_argument("--keep-issue-subjects", action="store_true")
    p.add_argument("--bidmc-gt", choices=("ann1", "ann1_nogap"), default="ann1")
    p.add_argument("--members", type=int, default=MEMBERS)
    p.add_argument("--output-root", type=Path, default=OUTPUT_ROOT)
    p.add_argument("--root-261007", type=Path, default=ROOT_261007)
    p.add_argument("--root-261013", type=Path, default=None)
    p.add_argument("--script-261007", type=Path, default=None)
    p.add_argument("--script-261013", type=Path, default=None)
    p.add_argument("--script-261113", type=Path, default=None)
    p.add_argument("--script-261116", type=Path, default=None)
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
    rescore = load_script(find_script(SCRIPTS["261113"], args.script_261113, []))
    rm = load_script(find_script(SCRIPTS["261116"], args.script_261116, []))
    rows, minutes, missing, checks, limits = [], [], [], [], {}
    LOG.clear()
    for dataset in datasets:
        part = dataset_rows(args, dataset, chosen, out_root, rescore, rm)
        rows += part["rows"]
        minutes += part["minutes"]
        missing += part["missing"]
        checks += part["checks"]
        if part["limits"]:
            limits[dataset] = part["limits"]
    table = pd.DataFrame(rows)
    minute_table = pd.concat(minutes, ignore_index=True) if minutes else pd.DataFrame()
    subj, ds = summaries(minute_table) if not minute_table.empty else (pd.DataFrame(), pd.DataFrame())
    check_table = pd.DataFrame(checks)
    sheets = issue_tables(minute_table, check_table, limits) if not minute_table.empty else {}
    if sheets:
        s = sheets["Issue_Summary"]
        n_issue, n_worse = int(s["Issue_Windows"].sum()), int(s["Worse_Minutes"].sum())
        log("\n[Issue check] all datasets: "
            + (f"input issue in {n_issue} of {int(s['Windows_Checked'].sum())} model windows "
               f"({int(s['Issue_Minutes'].sum())} minutes, {int(s['Issue_Subjects'].sum())} subjects)" if n_issue
               else "input issue: none")
            + "; " + (f"result worse in {n_worse} minutes of {int(s['Worse_Subjects'].sum())} subjects" if n_worse
                      else "result worse: none"))
    notes = pd.DataFrame([
        ("Model", "RRWaveNet ensemble (mean of the members' stored test outputs): CapnoBase 261007 variant full; "
                  "BIDMC, STEAM2 261013 weight decay selected per fold"),
        ("Scored minutes", "rescored ground truth, scored with the source script's minute scoring (helpers of 261113): "
                           "CapnoBase peak (capnogram peak detection, RRest rule), BIDMC " + args.bidmc_gt
                           + ", STEAM2 stored GT; only Ref_Valid minutes; every minute has a model input and output"),
        ("Windows", "60 s = the scored minute; 30 s = its halves; minute values (GT, rate, AE, PCC_Minute_Reported) "
                    "are those of the scoring unit (60 s)"),
        ("PCC_As_Scored_Window", "Pearson r of the stitched target and the ensemble output in the window (the scoring "
                                 "definition; equals PCC_Minute_Reported for 60 s windows)"),
        ("r_Drawn_Curves", "Pearson r of the drawn curves: continuous reference vs. output band-passed as scored"),
        ("Respmod", "functions of 261116_PPG_RIIV_RIAV_RIFV_FFT; peaks in the recording's rate band of the scoring "
                    "profile; windows need >= 90% series cover and >= 10 used beats"),
        ("Subjects", "issue subjects left out (CapnoBase 0115, BIDMC 27, STEAM2 12_YJJ, 22_LMS, 26_CKK)"
                     if not args.keep_issue_subjects else "every subject (issue subjects kept)"),
        ("Input issue check", "every 10 s model window inside the scored minutes (all passed the data selection: "
                              ">= 50% usable PPG and >= 50% structurally valid reference, so the model used them)"),
    ] + [(f"Issue {k}", f"{ISSUES[k][0]}: {ISSUES[k][2]}") for k in ISSUES] + [
        ("Issue measures", f"PPG flat: step <= {FLAT_TOLERANCE:g} of the recording's 5-95% range for >= "
                           f"{FLAT_MIN_SEC:g} s (clipped: at the recording's max / min); PPG amplitude: 95th-5th "
                           f"percentile of the {PULSE_BAND_HZ[0]:g}-{PULSE_BAND_HZ[1]:g} Hz pulse per "
                           f"{PULSE_BLOCK_SEC:g} s / the recording's median; RSP collapse: Hilbert envelope "
                           f"({COLLAPSE_SMOOTH_SEC:g} s mean) / the recording's median, <= 50% (marked) or <= 10% "
                           f"(severe) for >= {COLLAPSE_MIN_SEC:g} s (CapnoBase: also >= 2 breath periods); PPG quality "
                           "and RSP quality: the source script's own maps (PPG_Usable_Proportion, "
                           "RSP_Resp_Usable_Proportion)"),
        ("Used_For", "every checked window was used for test scoring; Train_OK windows were also used for "
                     "training / validation of the other folds' models (never for the subject's own fold)"),
        ("Result worse", f"a minute with an issue window vs. the same subject's issue-free scored minutes (>= "
                         f"{MIN_CLEAN_MINUTES}): rate worse when AE - their median AE >= AE_Limit; waveform worse when "
                         "their median |PCC| - |PCC| >= PCC_Limit"),
        ("AE_Limit, PCC_Limit", f"per dataset, the {WORSE_PERCENTILE:g}th percentile of the same differences among the "
                                "issue-free minutes (each against the subject's other issue-free minutes) = normal "
                                f"minute-to-minute fluctuation; AE_Limit is at least {AE_RESOLUTION:g} /min (rate "
                                "resolution of a 60 s minute); |PCC| because the polarity of some references is "
                                "arbitrary"),
        ("Likely_Source, GT_Check", "PPG issue = model input; RSP issue = reference (training target and GT of the "
                                    "scoring); GT_Check compares the GT, RRWaveNet and the reference's own spectrum "
                                    "(Ref_RR_FFT) to tell a GT-side error from a model error"),
        ("Issue figures", "issues\\A_input_issue: every 60 s / 30 s window with an issue window inside; "
                          "issues\\B_result_worse: the same figures of the minutes judged worse"),
    ], columns=["Item", "Description"])
    path = out_root / f"{NAME}.xlsx"
    with pd.ExcelWriter(path, engine="openpyxl") as writer:
        table.round(5).to_excel(writer, sheet_name="Windows", index=False)
        minute_table.round(5).to_excel(writer, sheet_name="Minutes", index=False)
        subj.round(4).to_excel(writer, sheet_name="Subjects", index=False)
        ds.round(4).to_excel(writer, sheet_name="Datasets", index=False)
        for name, frame in sheets.items():
            frame.round(5).to_excel(writer, sheet_name=name, index=False)
        pd.DataFrame(missing, columns=["Dataset", "Subject", "Missing"]).to_excel(writer, sheet_name="Missing", index=False)
        notes.to_excel(writer, sheet_name="Notes", index=False)
    log_path = out_root / f"{NAME}_issue_check.txt"
    log_path.write_text("\n".join(LOG).lstrip("\n") + "\n", encoding="utf-8")
    print(f"\n{len(table)} windows drawn from {len(minute_table)} scored minutes; figures under {out_root}; table: {path}; "
          f"issue check log: {log_path}")
    if not ds.empty:
        print(ds.round(3).to_string(index=False))


if __name__ == "__main__":
    main()
