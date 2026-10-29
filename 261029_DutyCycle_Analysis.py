r"""261029_DutyCycle_Analysis: inspiratory duty cycle of the respiratory
waveforms estimated by RespDiff, the deep corr-encoder and RRWaveNet, as in
Davies & Mandic, "Rapid extraction of respiratory waveforms from
photoplethysmography: A deep corr-encoder approach", Biomed. Signal Process.
Control 85 (2023) 104992 (Sections 2.3 and 3.3), and its statistical
comparison between the three models. Only stored results are read; nothing is
trained or sampled.

Duty cycle (paper, Section 2.3)
-------------------------------
"given that the waveforms were trained to an amplitude between 0 and 1, the
inspiratory duty cycle was estimated as the number of samples below 0.5
divided by the total number of samples. This was then multiplied by 100 to
give a percentage. The same operation was performed on the gold standard
capnography for comparison. Both the mean and median absolute error in duty
cycle are presented." Section 3.3: the duty cycle of the output against that
of the capnogram, "median Pearson's correlation coefficient of 0.16 (P =
1.8e-3)". The paper evaluates the duty cycle on CapnoBase only (capnography:
low CO2 = inspiration, so the samples below half amplitude are the
inspiration). It does not state over which windows the duty cycle is taken or
which test gives its P value; here:
* evaluation windows: non-overlapping 30 s and 60 s windows from the start of
  every recording (--windows);
* correlation: the Pearson r of model and reference duty cycle over the
  windows of each subject, reported as median (IQR) over subjects with a
  Wilcoxon signed-rank test against 0, and the pooled Pearson r over all
  windows with its p value (the two readings of the paper's sentence).
BIDMC (impedance pneumogram) and STEAM2 (respiratory belt) have no duty cycle
in the paper; the same operation is applied and gives the share of time the
waveform spends below half of its amplitude, which is not the inspiratory
phase of these signals. Their results are reported as such.

Two rules for the 0-1 amplitude (both reported):
* paper: every waveform on the amplitude scale it was trained to, as in the
  paper: deep corr-encoder and RespDiff outputs and the reference 0 to 1,
  threshold 0.5; RRWaveNet -1 to 1 (tanh output, target min-max scaled to
  -1..1 per 10 s window), threshold 0 (its midpoint);
* window_minmax: every waveform (outputs and reference) min-max scaled to
  0-1 inside the evaluation window, threshold 0.5.

Models and stored results (the results of the scripts named)
-------------------------------------------------------------
  RespDiff           261020_RespDiff / 261028_RespDiff_v3, DDPM 50:
                     {root}\respdiff\{dataset}\Predictions\{subject}_ddpm50.npz
                     (joined and low-pass filtered 5 s outputs, 30 Hz)
  Deep corr-encoder  261028_Deep_correncoder:
                     {root}\{dataset}\Predictions\{subject}_outputs.npz
                     (9.6 s outputs shifted 1 s and averaged, 30 Hz)
  RRWaveNet          CapnoBase: 261007_RRWaveNet_Ensemble_NoEvent, full model,
                     10 s windows, ensemble of 3:
                     {root}\full\01_w10s_nonoverlap\CapnoBase\Ensemble_Predictions\
                     BIDMC, STEAM2: 261013_RRWaveNet_Adult_SubjectVal_WD, the
                     weight decay selected in every fold
                     (WD_Selection\01_w10s_nonoverlap\{dataset}\{subject}_selection.json
                     -> {root}\full_wd{wd}\01_w10s_nonoverlap\{dataset}\Ensemble_Predictions\)
                     10 s windows at 64 Hz with their segment and start
A fold counts only when its result file is complete (RespDiff
Fold_Results\{subject}_ddpm50.json, corr-encoder
Fold_Results\{subject}_summary.json, RRWaveNet
Fold_Summaries\{subject}_ensemble_summary.json).

Gold standard and alignment
---------------------------
* Reference: the reference waveform as the paper prepares it, stored with the
  deep corr-encoder results (recording{k}_reference): 30 Hz, every 9.6 s window
  min-max scaled to 0-1, the windows shifted 1 s and averaged. CapnoBase the
  capnogram, BIDMC the impedance pneumogram, STEAM2 the belt as recorded. The
  same reference is used for all three models.
* Recordings: CapnoBase and BIDMC one per subject; STEAM2 every session piece
  (split at boundary events; the three scripts read the sessions with the same
  reader): RespDiff by name ({session}_piece{k}), RRWaveNet by segment number
  (segment k = recording k of the corr-encoder).
* Time: all outputs start at the start of their recording. RRWaveNet
  (CapnoBase) shifted its target by the lag of the capnogram behind the PPG
  (Target_Lag_Sec), so its output at time t belongs to the reference at t +
  lag; the lag is applied.
* Polarity: RespDiff and the corr-encoder are trained on the reference as
  recorded. RRWaveNet's BIDMC target was multiplied by the sign of its
  zero-lag correlation with the PPG (Target_Sign), so its output is
  multiplied by that sign; CapnoBase and STEAM2 keep the recorded polarity.
  Sign and lag of every segment come from
  {root}\Data_Quality\01_w10s_nonoverlap\{dataset}_window_quality.csv; without
  that file the BIDMC sign is taken from the correlation of RRWaveNet's stored
  targets with the reference, and the CapnoBase lag is 0 (both listed in
  alignment.csv). alignment.csv also gives, per segment, the correlation of
  RRWaveNet's stored target with the reference after alignment (a check that
  segments and polarity match).
* Windows: every waveform is taken at the same times, the 30 Hz samples of
  the reference in the window (RRWaveNet, 64 Hz: the nearest sample), and only
  the samples where the reference and all three models have output count; a
  window is scored when they cover at least 90% of it (--min-coverage).
  RRWaveNet scores only the 10 s windows that passed its quality check, so
  every model is compared on the same windows, samples and subjects (subjects
  with all three models). coverage.csv lists the windows each model covers
  and the common ones.

Metrics (per dataset, window length, rule and model)
----------------------------------------------------
* windows: duty cycle of the reference and the model (mean +- SD), error
  (model - reference: bias +- SD, 95% limits of agreement), absolute error
  mean +- SD and median (IQR) (the paper's mean and median absolute error),
  pooled Pearson r with p;
* subjects: subject MAE (mean +- SD, median (IQR)), subject Pearson r over
  the subject's windows (at least 3 windows): median (IQR), Wilcoxon
  signed-rank against 0.

Statistics (subject = unit; the three models on the same windows)
------------------------------------------------------------------
* Friedman test over the three models (Kendall's W) of the subject duty-cycle
  MAE and of the subject duty-cycle r;
* pairwise Wilcoxon signed-rank tests (two-sided) with the median of the
  paired differences and the matched-pairs rank-biserial correlation;
  Benjamini-Hochberg over the three pairs of each family (dataset, window,
  rule, metric);
* the same on the absolute error of every window, as a secondary result (the
  windows of one subject are not independent).

What the models were trained to reproduce (for the interpretation)
-------------------------------------------------------------------
RespDiff (5 s segments) and the corr-encoder (9.6 s windows) are trained on
the reference waveform itself, scaled to 0-1. RRWaveNet is not: on CapnoBase
its target is a sine wave of the respiratory phase between the expert breath
labels (261007, label_phase_waveform), whose duty cycle is 50% by
construction; on BIDMC and STEAM2 the reference band-passed to 0.1-0.5 Hz.
Its duty cycle therefore measures what is left of the inspiratory timing
after a target that removed it by design.

Outputs (PROJECT_ROOT = D:\PPG2RSP_RRWaveNet_Inspired\261029_DutyCycle_Analysis)
  windows.csv       every scored window: duty cycle of the reference and of
                    every model, both rules
  per_subject.csv   per subject, model, window length and rule
  summary.csv       per dataset, window length, rule and model
  statistics.csv    Friedman and pairwise Wilcoxon tests
  coverage.csv      windows per subject and model, common windows
  alignment.csv     RRWaveNet sign, lag and target-reference correlation per
                    segment; recording lengths of the three models
  261029_DutyCycle_Analysis.xlsx  all tables
  Figures\{dataset}_{window}s_{rule}_scatter.png / _subject_mae.png

Run:                     python 261029_DutyCycle_Analysis.py
Some datasets / windows: --datasets capnobase --windows 30 60
Result folders:          --respdiff-root, --correncoder-root,
                         --rrwavenet-261007-root, --rrwavenet-261013-root,
                         --output-root
"""

import argparse
import json
import math
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats


# ---------------------------------------------------------------------------
# Result folders and settings
# ---------------------------------------------------------------------------

PROJECT_ROOT = Path(r"D:\PPG2RSP_RRWaveNet_Inspired\261029_DutyCycle_Analysis")
RESPDIFF_ROOT = Path(r"D:\PPG2RSP_RRWaveNet_Inspired\261020_RespDiff_v2")
CORRENCODER_ROOT = Path(r"D:\PPG2RSP_RRWaveNet_Inspired\261028_Deep_correncoder")
RRWAVENET_261007_ROOT = Path(r"D:\PPG2RSP_RRWaveNet_Inspired\261007_RRWaveNet_NoEvent_v1")
RRWAVENET_261013_ROOT = Path(r"D:\PPG2RSP_RRWaveNet_Inspired\261013_RRWavNet_STEAM_BIDMC_v1")

DATASET_LABELS = {"capnobase": "CapnoBase", "bidmc": "BIDMC", "steam2": "STEAM2"}
DATASET_ORDER = ("capnobase", "bidmc", "steam2")
REFERENCE_SIGNALS = {
    "capnobase": "capnogram (inspiratory duty cycle)",
    "bidmc": "impedance pneumogram (time below half amplitude)",
    "steam2": "respiratory belt (time below half amplitude)",
}

# Models in a fixed order (table order and figure colours).
MODELS = ("RespDiff", "Deep corr-encoder", "RRWaveNet")
MODEL_COLORS = {"RespDiff": "#2a78d6", "Deep corr-encoder": "#eb6834", "RRWaveNet": "#1baf7a"}
PAIRS = tuple((a, b) for i, a in enumerate(MODELS) for b in MODELS[i + 1:])

RESPDIFF_SAMPLING = "ddpm50"
RESPDIFF_FS = 30
CORRENCODER_FS = 30
RRWAVENET_FS = 64
RRWAVENET_WINDOW = "01_w10s_nonoverlap"

EVAL_WINDOWS_SEC = (30.0, 60.0)
MIN_COVERAGE = 0.90
MIN_WINDOWS_FOR_R = 3
RULES = ("paper", "window_minmax")
# Paper rule: threshold at the middle of the amplitude range a waveform was
# trained to (0-1, RRWaveNet -1..1).
PAPER_THRESHOLDS = {"Reference": 0.5, "RespDiff": 0.5, "Deep corr-encoder": 0.5, "RRWaveNet": 0.0}

TEXT_PRIMARY = "#0b0b0b"
TEXT_SECONDARY = "#52514e"
GRID_COLOR = "#e4e3df"
SURFACE = "#fcfcfb"


@dataclass
class Track:
    """A waveform on the reference time axis: sample i lies at
    offset_sec + i / fs seconds of the reference; NaN = no output."""

    values: np.ndarray
    fs: float
    offset_sec: float = 0.0

    def at(self, times: np.ndarray) -> np.ndarray:
        """The sample nearest to every time (s on the reference axis); NaN
        outside the waveform or where it has no output."""
        index = np.round((np.asarray(times) - self.offset_sec) * self.fs).astype(np.int64)
        inside = (index >= 0) & (index < len(self.values))
        out = np.full(index.shape, np.nan)
        out[inside] = self.values[index[inside]]
        return out


def read_json(path: Path) -> Optional[dict]:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


def pearson(a: Sequence[float], b: Sequence[float]) -> Tuple[float, float]:
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    keep = np.isfinite(a) & np.isfinite(b)
    a, b = a[keep], b[keep]
    if len(a) < 3 or a.std() <= 1e-12 or b.std() <= 1e-12:
        return math.nan, math.nan
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        r, p = stats.pearsonr(a, b)
    return float(r), float(p)


# ---------------------------------------------------------------------------
# Stored results of the three models
# ---------------------------------------------------------------------------

def load_correncoder(root: Path, label: str) -> Dict[str, List[dict]]:
    """Subject -> recordings (index k, name, output, reference) of the deep
    corr-encoder; only folds whose summary was written (complete)."""
    folder = root / label
    subjects = {}
    for path in sorted((folder / "Predictions").glob("*_outputs.npz")):
        subject = path.name[: -len("_outputs.npz")]
        if not (folder / "Fold_Results" / f"{subject}_summary.json").exists():
            continue
        with np.load(path, allow_pickle=False) as data:
            names = [str(name) for name in data["recording_names"]]
            recordings = []
            for k, name in enumerate(names):
                if f"recording{k}_output" not in data.files:
                    continue  # shorter than one 9.6 s window
                recordings.append({
                    "index": k,
                    "name": name,
                    "output": data[f"recording{k}_output"].astype(np.float64),
                    "reference": data[f"recording{k}_reference"].astype(np.float64),
                })
        subjects[subject] = recordings
    return subjects


def load_respdiff(root: Path, label: str, sampling: str) -> Dict[str, Dict[str, np.ndarray]]:
    """Subject -> {recording name: output} of RespDiff (one sampling)."""
    folder = root / "respdiff" / label
    subjects = {}
    for path in sorted((folder / "Predictions").glob(f"*_{sampling}.npz")):
        subject = path.name[: -len(f"_{sampling}.npz")]
        if not (folder / "Fold_Results" / f"{subject}_{sampling}.json").exists():
            continue
        with np.load(path, allow_pickle=False) as data:
            names = [str(name) for name in data["recording_names"]]
            subjects[subject] = {
                name: data[f"recording{i}_prediction"].astype(np.float64)
                for i, name in enumerate(names)
                if f"recording{i}_prediction" in data.files
            }
    return subjects


def rrwavenet_files(dataset_key: str, root_261007: Path, root_261013: Path) -> Tuple[Dict[str, Tuple[Path, str]], Path]:
    """Subject -> (ensemble prediction file, run) of RRWaveNet, and the root
    of the run (for its data quality tables)."""
    label = DATASET_LABELS[dataset_key]
    files = {}
    if dataset_key == "capnobase":
        root = root_261007
        condition = root / "full" / RRWAVENET_WINDOW / label
        for path in sorted((condition / "Ensemble_Predictions").glob("*_ensemble_outputs.npz")):
            subject = path.name[: -len("_ensemble_outputs.npz")]
            if (condition / "Fold_Summaries" / f"{subject}_ensemble_summary.json").exists():
                files[subject] = (path, "261007 full")
        return files, root
    root = root_261013
    for selection in sorted((root / "WD_Selection" / RRWAVENET_WINDOW / label).glob("*_selection.json")):
        subject = selection.name[: -len("_selection.json")]
        record = read_json(selection) or {}
        variant = record.get("Selected_Variant")
        if not variant:
            continue
        condition = root / variant / RRWAVENET_WINDOW / label
        path = condition / "Ensemble_Predictions" / f"{subject}_ensemble_outputs.npz"
        if path.exists() and (condition / "Fold_Summaries" / f"{subject}_ensemble_summary.json").exists():
            files[subject] = (path, f"261013 {variant}")
    return files, root


def rrwavenet_alignment_table(root: Path, label: str) -> Dict[Tuple[str, int], dict]:
    """(subject, segment) -> target sign and lag of RRWaveNet, from its data
    quality table (empty when the table is not there)."""
    path = root / "Data_Quality" / RRWAVENET_WINDOW / f"{label}_window_quality.csv"
    if not path.exists():
        return {}
    wanted = {"Subject", "Segment_ID", "Target_Sign", "Target_Lag_Sec"}
    frame = pd.read_csv(path, dtype={"Subject": str}, usecols=lambda column: column in wanted)
    if not wanted <= set(frame.columns):
        return {}
    first = frame.groupby(["Subject", "Segment_ID"], sort=False).first()
    return {
        (str(subject), int(segment)): {"sign": float(row["Target_Sign"]), "lag_sec": float(row["Target_Lag_Sec"])}
        for (subject, segment), row in first.iterrows()
    }


def load_rrwavenet(path: Path) -> Dict[int, dict]:
    """Segment -> output and target of RRWaveNet's ensemble on the 64 Hz
    axis of the segment (NaN where no window was scored)."""
    segments = {}
    with np.load(path, allow_pickle=False) as data:
        predictions = data["predictions"].astype(np.float64)
        targets = data["targets"].astype(np.float64)
        segment_ids = data["segment_ids"].astype(np.int64)
        starts = data["window_starts"].astype(np.int64)
    if predictions.ndim != 2 or not len(predictions):
        return segments
    width = predictions.shape[1]
    for segment in np.unique(segment_ids):
        rows = np.flatnonzero(segment_ids == segment)
        length = int(starts[rows].max()) + width
        output = np.full(length, np.nan)
        target = np.full(length, np.nan)
        for row in rows:
            output[starts[row]:starts[row] + width] = predictions[row]
            target[starts[row]:starts[row] + width] = targets[row]
        segments[int(segment)] = {"output": output, "target": target, "windows": len(rows)}
    return segments


def target_reference_r(target: Track, reference: Track) -> float:
    """Pearson r of RRWaveNet's stored target and the reference at the same
    reference times (a check of segment, lag and polarity)."""
    times = target.offset_sec + np.arange(len(target.values)) / target.fs
    reference_times = np.arange(len(reference.values)) / reference.fs
    keep = np.isfinite(target.values) & (times >= 0) & (times <= reference_times[-1])
    if keep.sum() < 10:
        return math.nan
    resampled = np.interp(times[keep], reference_times, reference.values)
    return pearson(target.values[keep], resampled)[0]


# ---------------------------------------------------------------------------
# Duty cycle of every window
# ---------------------------------------------------------------------------

def duty_cycle(values: np.ndarray, rule: str, model: str) -> float:
    """Percentage of samples below half amplitude (paper rule: the midpoint
    of the trained amplitude range; window_minmax: after min-max scaling to
    0-1 inside the window)."""
    values = np.asarray(values, dtype=np.float64)
    if not len(values):
        return math.nan
    if rule == "paper":
        return float(100.0 * np.mean(values < PAPER_THRESHOLDS[model]))
    low, high = float(values.min()), float(values.max())
    if high - low <= 1e-12:
        return math.nan
    return float(100.0 * np.mean((values - low) / (high - low) < 0.5))


def dataset_windows(args: argparse.Namespace, dataset_key: str) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Windows of one dataset (both rules, every window length), the coverage
    and the alignment tables."""
    label = DATASET_LABELS[dataset_key]
    corr = load_correncoder(Path(args.correncoder_root), label)
    respdiff = load_respdiff(Path(args.respdiff_root), label, args.respdiff_sampling)
    rr_files, rr_root = rrwavenet_files(dataset_key, Path(args.rrwavenet_261007_root), Path(args.rrwavenet_261013_root))
    alignment_table = rrwavenet_alignment_table(rr_root, label)
    found = {"RespDiff": set(respdiff), "Deep corr-encoder": set(corr), "RRWaveNet": set(rr_files)}
    subjects = sorted(set.intersection(*found.values()))
    print(f"{label}: subjects with results - RespDiff {len(respdiff)}, corr-encoder {len(corr)}, "
          f"RRWaveNet {len(rr_files)}; all three {len(subjects)}"
          + ("" if alignment_table or not rr_files else f" (no RRWaveNet data quality table under {rr_root}: "
             "sign from the stored targets, lag 0)"), flush=True)
    window_rows, coverage_rows, alignment_rows = [], [], []
    for subject in sorted(set.union(*found.values())):
        if subject not in subjects:
            coverage_rows.append({"Dataset": label, "Subject": subject, "Included": False,
                                  **{f"Has_{m}": subject in found[m] for m in MODELS}})
    for subject in subjects:
        rr_segments = load_rrwavenet(rr_files[subject][0])
        for recording in corr[subject]:
            reference = Track(recording["reference"], CORRENCODER_FS)
            tracks = {"Deep corr-encoder": Track(recording["output"], CORRENCODER_FS)}
            if recording["name"] in respdiff[subject]:
                tracks["RespDiff"] = Track(respdiff[subject][recording["name"]], RESPDIFF_FS)
            segment = rr_segments.get(recording["index"])
            info = {"Dataset": label, "Subject": subject, "Recording": recording["name"],
                    "Segment": recording["index"], "RRWaveNet_Run": rr_files[subject][1],
                    "Reference_Sec": len(reference.values) / reference.fs,
                    "RespDiff_Sec": (len(tracks["RespDiff"].values) / RESPDIFF_FS) if "RespDiff" in tracks else math.nan}
            if segment is not None:
                stored = alignment_table.get((subject, recording["index"]))
                lag = stored["lag_sec"] if stored else 0.0
                target = Track(segment["target"], RRWAVENET_FS, lag)
                if stored:
                    sign, source = stored["sign"], "data quality table"
                elif dataset_key == "bidmc":
                    r = target_reference_r(target, reference)
                    sign, source = (-1.0 if r < 0 else 1.0), "correlation of stored target and reference"
                else:
                    sign, source = 1.0, "recorded polarity (no data quality table)"
                tracks["RRWaveNet"] = Track(sign * segment["output"], RRWAVENET_FS, lag)
                info.update({
                    "RRWaveNet_Sign": sign, "RRWaveNet_Lag_Sec": lag, "Alignment_Source": source,
                    "RRWaveNet_Windows": segment["windows"],
                    "RRWaveNet_Last_Sec": lag + len(segment["output"]) / RRWAVENET_FS,
                    "Target_Reference_r": target_reference_r(Track(sign * segment["target"], RRWAVENET_FS, lag),
                                                             reference),
                })
            alignment_rows.append(info)
            duration = len(reference.values) / reference.fs
            for window_sec in args.windows:
                count = int(math.ceil(duration / window_sec))
                covered = {m: 0 for m in MODELS}
                common = 0
                for index in range(count):
                    start = index * window_sec
                    # Every waveform at the same times (the 30 Hz samples of
                    # the reference in the window).
                    times = start + np.arange(int(round(window_sec * reference.fs))) / reference.fs
                    ref_values = reference.at(times)
                    if np.isfinite(ref_values).mean() < args.min_coverage:
                        continue
                    values = {m: tracks[m].at(times) for m in MODELS if m in tracks}
                    for model, part in values.items():
                        if np.isfinite(part).mean() >= args.min_coverage:
                            covered[model] += 1
                    if len(values) < len(MODELS):
                        continue
                    valid = np.isfinite(ref_values) & np.all([np.isfinite(v) for v in values.values()], axis=0)
                    if valid.mean() < args.min_coverage:
                        continue
                    common += 1
                    for rule in RULES:
                        row = {"Dataset": label, "Subject": subject, "Recording": recording["name"],
                               "Window_Sec": window_sec, "Window": index + 1, "Start_Sec": start,
                               "Coverage": float(valid.mean()), "Rule": rule,
                               "DC_Reference": duty_cycle(ref_values[valid], rule, "Reference")}
                        row.update({f"DC_{m}": duty_cycle(values[m][valid], rule, m) for m in MODELS})
                        window_rows.append(row)
                coverage_rows.append({"Dataset": label, "Subject": subject, "Included": True,
                                      **{f"Has_{m}": True for m in MODELS},
                                      "Recording": recording["name"], "Window_Sec": window_sec,
                                      "Reference_Windows": count, **{f"Windows_{m}": covered[m] for m in MODELS},
                                      "Common_Windows": common})
    return pd.DataFrame(window_rows), pd.DataFrame(coverage_rows), pd.DataFrame(alignment_rows)


# ---------------------------------------------------------------------------
# Metrics and statistics
# ---------------------------------------------------------------------------

def mean_sd(values: Sequence[float]) -> str:
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if not len(values):
        return ""
    sd = values.std(ddof=1) if len(values) > 1 else math.nan
    return f"{values.mean():.2f} ± {sd:.2f}"


def median_iqr(values: Sequence[float], digits: int = 2) -> str:
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if not len(values):
        return ""
    q1, q2, q3 = np.percentile(values, [25, 50, 75])
    return f"{q2:.{digits}f} ({q1:.{digits}f}-{q3:.{digits}f})"


def with_errors(windows: pd.DataFrame) -> pd.DataFrame:
    frame = windows.copy()
    for model in MODELS:
        frame[f"Error_{model}"] = frame[f"DC_{model}"] - frame["DC_Reference"]
        frame[f"AE_{model}"] = frame[f"Error_{model}"].abs()
    return frame


def subject_table(windows: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (dataset, window_sec, rule, subject), part in windows.groupby(["Dataset", "Window_Sec", "Rule", "Subject"],
                                                                      sort=False):
        for model in MODELS:
            have = part[np.isfinite(part[f"DC_{model}"]) & np.isfinite(part["DC_Reference"])]
            r, p = pearson(have[f"DC_{model}"], have["DC_Reference"]) if len(have) >= MIN_WINDOWS_FOR_R \
                else (math.nan, math.nan)
            rows.append({
                "Dataset": dataset, "Window_Sec": window_sec, "Rule": rule, "Subject": subject, "Model": model,
                "N_Windows": int(len(have)),
                "DC_Reference_Mean": float(have["DC_Reference"].mean()) if len(have) else math.nan,
                "DC_Model_Mean": float(have[f"DC_{model}"].mean()) if len(have) else math.nan,
                "Bias": float(have[f"Error_{model}"].mean()) if len(have) else math.nan,
                "MAE": float(have[f"AE_{model}"].mean()) if len(have) else math.nan,
                "Median_AE": float(have[f"AE_{model}"].median()) if len(have) else math.nan,
                "r": r, "r_p": p,
            })
    return pd.DataFrame(rows)


def summary_table(windows: pd.DataFrame, subjects: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (dataset, window_sec, rule), part in windows.groupby(["Dataset", "Window_Sec", "Rule"], sort=False):
        for model in MODELS:
            have = part[np.isfinite(part[f"DC_{model}"]) & np.isfinite(part["DC_Reference"])]
            error, ae = have[f"Error_{model}"].to_numpy(float), have[f"AE_{model}"].to_numpy(float)
            sub = subjects[(subjects["Dataset"] == dataset) & (subjects["Window_Sec"] == window_sec)
                           & (subjects["Rule"] == rule) & (subjects["Model"] == model)]
            sub_r = sub["r"].dropna()
            pooled_r, pooled_p = pearson(have[f"DC_{model}"], have["DC_Reference"])
            wilcoxon_p = math.nan
            if len(sub_r) >= 3 and np.any(sub_r != 0):
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    wilcoxon_p = float(stats.wilcoxon(sub_r).pvalue)
            bias = float(error.mean()) if len(error) else math.nan
            sd = float(error.std(ddof=1)) if len(error) > 1 else math.nan
            rows.append({
                "Dataset": dataset, "Window_Sec": window_sec, "Rule": rule, "Model": model,
                "N_Subjects": int(sub["Subject"].nunique()), "N_Windows": int(len(have)),
                "DC reference (mean ± SD)": mean_sd(have["DC_Reference"]),
                "DC model (mean ± SD)": mean_sd(have[f"DC_{model}"]),
                "Bias ± SD": f"{bias:.2f} ± {sd:.2f}" if len(error) else "",
                "95% LoA": f"{bias - 1.96 * sd:.2f} to {bias + 1.96 * sd:.2f}" if len(error) > 1 else "",
                "AE (mean ± SD)": mean_sd(ae),
                "AE (median, IQR)": median_iqr(ae),
                "Subject MAE (mean ± SD)": mean_sd(sub["MAE"]),
                "Subject MAE (median, IQR)": median_iqr(sub["MAE"]),
                "Pooled r (p)": f"{pooled_r:.3f} ({pooled_p:.2g})" if np.isfinite(pooled_r) else "",
                "Subject r (median, IQR)": median_iqr(sub_r, 3),
                "Subject r: N, Wilcoxon p vs 0": f"{len(sub_r)}, {wilcoxon_p:.2g}" if len(sub_r) else "",
                "DC_Reference_Mean": float(have["DC_Reference"].mean()) if len(have) else math.nan,
                "DC_Model_Mean": float(have[f"DC_{model}"].mean()) if len(have) else math.nan,
                "DC_Model_SD": float(have[f"DC_{model}"].std(ddof=1)) if len(have) > 1 else math.nan,
                "Bias": bias, "Bias_SD": sd,
                "AE_Mean": float(ae.mean()) if len(ae) else math.nan,
                "AE_SD": float(ae.std(ddof=1)) if len(ae) > 1 else math.nan,
                "AE_Median": float(np.median(ae)) if len(ae) else math.nan,
                "Subject_MAE_Mean": float(sub["MAE"].mean()) if len(sub) else math.nan,
                "Subject_MAE_Median": float(sub["MAE"].median()) if len(sub) else math.nan,
                "Pooled_r": pooled_r, "Pooled_r_p": pooled_p,
                "Subject_r_Median": float(sub_r.median()) if len(sub_r) else math.nan,
                "Subject_r_N": int(len(sub_r)), "Subject_r_Wilcoxon_p": wilcoxon_p,
            })
    return pd.DataFrame(rows)


def benjamini_hochberg(p_values: Sequence[float]) -> np.ndarray:
    p = np.asarray(p_values, dtype=float)
    out = np.full(p.shape, np.nan)
    finite = np.flatnonzero(np.isfinite(p))
    if not len(finite):
        return out
    order = finite[np.argsort(p[finite])]
    m = len(order)
    adjusted = p[order] * m / np.arange(1, m + 1)
    adjusted = np.minimum.accumulate(adjusted[::-1])[::-1]
    out[order] = np.clip(adjusted, 0.0, 1.0)
    return out


def paired_wilcoxon(a: np.ndarray, b: np.ndarray) -> dict:
    """Two-sided Wilcoxon signed-rank test of a - b, the median difference and
    the matched-pairs rank-biserial correlation (positive: a larger)."""
    difference = np.asarray(a, dtype=float) - np.asarray(b, dtype=float)
    nonzero = difference[difference != 0]
    result = {"N": int(len(difference)), "Median_A": float(np.median(a)), "Median_B": float(np.median(b)),
              "Median_Difference": float(np.median(difference)), "W": math.nan, "p": math.nan,
              "Rank_Biserial": math.nan}
    if len(nonzero):
        ranks = stats.rankdata(np.abs(nonzero))
        result["Rank_Biserial"] = float((ranks[nonzero > 0].sum() - ranks[nonzero < 0].sum()) / ranks.sum())
    if len(nonzero) >= 1:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            test = stats.wilcoxon(a, b)
        result["W"], result["p"] = float(test.statistic), float(test.pvalue)
    return result


def statistics_table(windows: pd.DataFrame, subjects: pd.DataFrame) -> pd.DataFrame:
    rows = []
    families = []
    for (dataset, window_sec, rule), part in subjects.groupby(["Dataset", "Window_Sec", "Rule"], sort=False):
        for metric, unit in (("MAE", "subject"), ("r", "subject")):
            wide = part.pivot(index="Subject", columns="Model", values=metric)[list(MODELS)].dropna()
            families.append((dataset, window_sec, rule, f"Subject {metric}", unit, wide))
        win = windows[(windows["Dataset"] == dataset) & (windows["Window_Sec"] == window_sec)
                      & (windows["Rule"] == rule)]
        wide = win[[f"AE_{m}" for m in MODELS]].dropna()
        wide.columns = list(MODELS)
        families.append((dataset, window_sec, rule, "Window AE", "window (not independent)", wide))
    for dataset, window_sec, rule, metric, unit, wide in families:
        base = {"Dataset": dataset, "Window_Sec": window_sec, "Rule": rule, "Metric": metric, "Unit": unit}
        n = len(wide)
        friedman = {"Test": "Friedman", "Comparison": " vs ".join(MODELS), "N": n}
        if n >= 3:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                chi2, p = stats.friedmanchisquare(*[wide[m].to_numpy(float) for m in MODELS])
            friedman.update({"Statistic": float(chi2), "p": float(p),
                             "Kendall_W": float(chi2 / (n * (len(MODELS) - 1))) if n else math.nan})
        rows.append({**base, **friedman})
        pair_rows = []
        for a, b in PAIRS:
            result = paired_wilcoxon(wide[a].to_numpy(float), wide[b].to_numpy(float)) if n >= 1 else {"N": 0}
            pair_rows.append({**base, "Test": "Wilcoxon signed-rank", "Comparison": f"{a} vs {b}",
                              "Statistic": result.get("W", math.nan), **result})
        adjusted = benjamini_hochberg([row.get("p", math.nan) for row in pair_rows])
        for row, q in zip(pair_rows, adjusted):
            row["p_BH"] = q
            rows.append(row)
    columns = ["Dataset", "Window_Sec", "Rule", "Metric", "Unit", "Test", "Comparison", "N", "Statistic", "p",
               "p_BH", "Kendall_W", "Median_A", "Median_B", "Median_Difference", "Rank_Biserial"]
    frame = pd.DataFrame(rows)
    return frame.reindex(columns=columns + [c for c in frame.columns if c not in columns and c != "W"])


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------

def style_axis(axis) -> None:
    axis.set_facecolor(SURFACE)
    axis.grid(True, color=GRID_COLOR, linewidth=0.8)
    axis.set_axisbelow(True)
    for side in ("top", "right"):
        axis.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        axis.spines[side].set_color(TEXT_SECONDARY)
    axis.tick_params(colors=TEXT_SECONDARY, labelsize=9)


def save_figures(windows: pd.DataFrame, subjects: pd.DataFrame, figure_root: Path) -> None:
    figure_root.mkdir(parents=True, exist_ok=True)
    for (dataset, window_sec, rule), part in windows.groupby(["Dataset", "Window_Sec", "Rule"], sort=False):
        tag = f"{dataset}_{window_sec:g}s_{rule}"
        values = part[["DC_Reference"] + [f"DC_{m}" for m in MODELS]].to_numpy(float)
        low, high = np.nanmin(values), np.nanmax(values)
        pad = 0.05 * max(high - low, 1.0)
        limits = (max(0.0, low - pad), min(100.0, high + pad))
        figure, axes = plt.subplots(1, len(MODELS), figsize=(12, 4.2), sharex=True, sharey=True,
                                    facecolor=SURFACE)
        for axis, model in zip(axes, MODELS):
            style_axis(axis)
            axis.plot(limits, limits, color=TEXT_SECONDARY, linewidth=1, linestyle="--")
            axis.scatter(part["DC_Reference"], part[f"DC_{model}"], s=14, color=MODEL_COLORS[model],
                         alpha=0.55, edgecolors=SURFACE, linewidths=0.5)
            r, _ = pearson(part[f"DC_{model}"], part["DC_Reference"])
            mae = float(part[f"AE_{model}"].mean())
            axis.set_title(f"{model}\npooled r = {r:.2f}, MAE = {mae:.1f} %", color=TEXT_PRIMARY, fontsize=10)
            axis.set_xlabel("Reference duty cycle (%)", color=TEXT_PRIMARY, fontsize=9)
            axis.set_xlim(limits)
            axis.set_ylim(limits)
        axes[0].set_ylabel("Model duty cycle (%)", color=TEXT_PRIMARY, fontsize=9)
        figure.suptitle(f"{dataset}: duty cycle per {window_sec:g} s window ({rule} rule)", color=TEXT_PRIMARY,
                        fontsize=11)
        figure.tight_layout()
        figure.savefig(figure_root / f"{tag}_scatter.png", dpi=150, facecolor=SURFACE)
        plt.close(figure)

        sub = subjects[(subjects["Dataset"] == dataset) & (subjects["Window_Sec"] == window_sec)
                       & (subjects["Rule"] == rule)]
        wide = sub.pivot(index="Subject", columns="Model", values="MAE")[list(MODELS)].dropna()
        if wide.empty:
            continue
        figure, axis = plt.subplots(figsize=(6, 4.2), facecolor=SURFACE)
        style_axis(axis)
        positions = np.arange(1, len(MODELS) + 1)
        for _, row in wide.iterrows():
            axis.plot(positions, row.to_numpy(float), color=GRID_COLOR, linewidth=0.8, zorder=1)
        box = axis.boxplot([wide[m] for m in MODELS], positions=positions, widths=0.45, patch_artist=True,
                           showfliers=False, medianprops={"color": TEXT_PRIMARY, "linewidth": 1.5})
        for patch, model in zip(box["boxes"], MODELS):
            patch.set_facecolor(MODEL_COLORS[model])
            patch.set_alpha(0.35)
            patch.set_edgecolor(MODEL_COLORS[model])
        for position, model in zip(positions, MODELS):
            jitter = np.random.default_rng(0).uniform(-0.08, 0.08, len(wide))
            axis.scatter(position + jitter, wide[model], s=12, color=MODEL_COLORS[model], zorder=3,
                         edgecolors=SURFACE, linewidths=0.5)
        axis.set_xticks(positions)
        axis.set_xticklabels(MODELS, color=TEXT_PRIMARY, fontsize=9)
        axis.set_ylabel("Subject duty-cycle MAE (%)", color=TEXT_PRIMARY, fontsize=9)
        axis.set_title(f"{dataset}: subject duty-cycle MAE, {window_sec:g} s windows ({rule} rule), "
                       f"n = {len(wide)}", color=TEXT_PRIMARY, fontsize=10)
        figure.tight_layout()
        figure.savefig(figure_root / f"{tag}_subject_mae.png", dpi=150, facecolor=SURFACE)
        plt.close(figure)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def write_excel(path: Path, sheets: Dict[str, pd.DataFrame]) -> None:
    try:
        with pd.ExcelWriter(path, engine="openpyxl") as writer:
            for name, frame in sheets.items():
                frame.to_excel(writer, index=False, sheet_name=name[:31])
    except Exception as exc:
        print(f"Excel export failed ({exc}); the CSV files are complete.")


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--datasets", nargs="+", choices=list(DATASET_ORDER), default=list(DATASET_ORDER))
    parser.add_argument("--windows", nargs="+", type=float, default=list(EVAL_WINDOWS_SEC),
                        help="Evaluation window lengths (s), non-overlapping.")
    parser.add_argument("--min-coverage", type=float, default=MIN_COVERAGE,
                        help="Share of a window that the reference and every model must cover.")
    parser.add_argument("--respdiff-sampling", default=RESPDIFF_SAMPLING)
    parser.add_argument("--respdiff-root", type=Path, default=RESPDIFF_ROOT)
    parser.add_argument("--correncoder-root", type=Path, default=CORRENCODER_ROOT)
    parser.add_argument("--rrwavenet-261007-root", type=Path, default=RRWAVENET_261007_ROOT,
                        help="261007 results (RRWaveNet, CapnoBase).")
    parser.add_argument("--rrwavenet-261013-root", type=Path, default=RRWAVENET_261013_ROOT,
                        help="261013 results (RRWaveNet, BIDMC and STEAM2).")
    parser.add_argument("--output-root", type=Path, default=PROJECT_ROOT)
    parser.add_argument("--no-figures", action="store_true")
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> None:
    args = parse_args(argv)
    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    window_frames, coverage_frames, alignment_frames = [], [], []
    for dataset_key in [key for key in DATASET_ORDER if key in args.datasets]:
        windows, coverage, alignment = dataset_windows(args, dataset_key)
        window_frames.append(windows)
        coverage_frames.append(coverage)
        alignment_frames.append(alignment)
    windows = pd.concat([f for f in window_frames if not f.empty], ignore_index=True) \
        if any(not f.empty for f in window_frames) else pd.DataFrame()
    coverage = pd.concat(coverage_frames, ignore_index=True)
    alignment = pd.concat(alignment_frames, ignore_index=True)
    coverage.to_csv(output_root / "coverage.csv", index=False, encoding="utf-8-sig")
    alignment.to_csv(output_root / "alignment.csv", index=False, encoding="utf-8-sig")
    if windows.empty:
        print("No window with all three models; see coverage.csv.")
        return
    windows = with_errors(windows)
    subjects = subject_table(windows)
    summary = summary_table(windows, subjects)
    statistics = statistics_table(windows, subjects)
    windows.to_csv(output_root / "windows.csv", index=False, encoding="utf-8-sig")
    subjects.to_csv(output_root / "per_subject.csv", index=False, encoding="utf-8-sig")
    summary.to_csv(output_root / "summary.csv", index=False, encoding="utf-8-sig")
    statistics.to_csv(output_root / "statistics.csv", index=False, encoding="utf-8-sig")
    write_excel(output_root / "261029_DutyCycle_Analysis.xlsx",
                {"Summary": summary, "Statistics": statistics, "Per_subject": subjects, "Windows": windows,
                 "Coverage": coverage, "Alignment": alignment})
    if not args.no_figures:
        save_figures(windows, subjects, output_root / "Figures")

    print("\nDuty cycle (% samples below half amplitude; model - reference):")
    for (dataset, window_sec, rule), part in summary.groupby(["Dataset", "Window_Sec", "Rule"], sort=False):
        print(f"\n  {dataset}, {window_sec:g} s windows, {rule} rule "
              f"({part['N_Subjects'].iloc[0]} subjects, {part['N_Windows'].iloc[0]} windows; reference DC "
              f"{part['DC reference (mean ± SD)'].iloc[0]})")
        for _, row in part.iterrows():
            print(f"    {row['Model']:<18s} DC {row['DC model (mean ± SD)']:<14s} AE {row['AE (mean ± SD)']:<14s} "
                  f"median AE {row['AE (median, IQR)']:<20s} subject r {row['Subject r (median, IQR)']:<22s} "
                  f"pooled r {row['Pooled r (p)']}")
        tests = statistics[(statistics["Dataset"] == dataset) & (statistics["Window_Sec"] == window_sec)
                           & (statistics["Rule"] == rule) & (statistics["Metric"] == "Subject MAE")]
        for _, row in tests.iterrows():
            p = row["p_BH"] if row["Test"] != "Friedman" else row["p"]
            label = "p" if row["Test"] == "Friedman" else "p (BH)"
            print(f"    subject MAE, {row['Test']:<21s} {row['Comparison']:<45s} {label} {p:.3g}")
    print(f"\nSaved under: {output_root}")


if __name__ == "__main__":
    main()
