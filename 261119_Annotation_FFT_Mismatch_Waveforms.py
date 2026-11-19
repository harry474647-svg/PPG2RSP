# -*- coding: utf-8 -*-
r"""261119_Annotation_FFT_Mismatch_Waveforms: every subject of the datasets with expert breath annotations
(BIDMC, CapnoBase) whose annotated breathing rate and the FFT rate of the preprocessed reference differ by
2 breaths/min or more in a 30 s window - the raw reference, the preprocessed reference, the reconstructed
waveform and the |FFT| of the three with their peaks, in every 30 s window, with the subject's MAE and PCC.

Annotations
  BIDMC      ann1 and ann2 (bidmc_data.mat ref.breaths; two annotators), --bidmc-annotators (default both)
  CapnoBase  the expert breath labels of the capnogram (co2_startexp of {case}_8min_labels)
  (STEAM2 has no annotation and is not part of this script.)

Selection (every 30 s window of every recording from its start: 0-30, 30-60, ... s)
  annotation rate  60 / mean breath interval of the annotation in the window (>= 2 breaths)
  FFT rate         peak of the |FFT| of the preprocessed reference in the window (detrend, Hamming window,
                   zero padding to >= 32768 points, as 261116) in the rate band of the scoring
                   (BIDMC 6-30, CapnoBase 3-60 /min)
  usable window    the reference is usable in the window as in the scoring: structurally valid for at least
                   half of it and no expert artifact span (CapnoBase CO2 artifacts); other windows are drawn
                   but never select a subject
  mismatch         usable window with |annotation rate - FFT rate| >= --threshold-bpm (default 2) for an
                   annotation (BIDMC: ann1 or ann2)
  A subject with at least one mismatch window is drawn (every window of it; the mismatch windows are
  shaded and their files end in _mismatch). The MAE issue subjects (BIDMC 27, CapnoBase 0115) are kept and
  marked.

Model (stored outputs, nothing is trained): the final model with the 1x1 stem fusion, member m0
  BIDMC      261013 (D:\PPG2RSP_RRWaveNet_Inspired\261013_RRWavNet_STEAM_BIDMC_v1), the weight decay selected
             in the fold (WD_Selection)
  CapnoBase  261007 (D:\PPG2RSP_RRWaveNet_Inspired\261007_RRWaveNet_NoEvent_v1), full
  (261028_Stem_ablation_noise_v1\full\... when the source has no stored result of the subject)
  Fold_Summaries\{subject}_m0_RR60s_windows.csv       every 60 s minute: stored rate and Wave_PCC
  Members\m0\Predictions\{subject}_test_outputs.npz    the predicted waveform (64 Hz)

Signals (the preprocessing of the script that produced the results, loaded as a module)
  raw            BIDMC impedance (125 Hz), CapnoBase capnogram (300 Hz), as recorded (no filter)
  preprocessed   the reference of the scoring: resampled to 64 Hz and band-passed (BIDMC 0.1-0.5 Hz; CapnoBase
                 0.05 Hz to the PPG-adaptive upper edge, aligned to the PPG); the GT waveform of the PCC
  reconstructed  the predicted waveform, band-passed as the rate was scored (--raw-predictions: raw output)

Scores per drawn subject (the stored minutes of member m0, as in the final results)
  MAE (final GT)  mean |stored rate - GT breath count| over the scored minutes: BIDMC ann1 with the
                  expert-label rule of 261013; CapnoBase count-orig breaths of the capnogram with the RRest rule
  MAE (each annotation)  the same minutes with the breath count of the annotation (BIDMC ann1 = final GT, ann2;
                  CapnoBase expert labels)
  |PCC|           mean |Wave_PCC| over the scored minutes

Figures ({Dataset}\{Dataset}_{subject}\ under the output folder)
  {Dataset}_{subject}_overview.png           the whole recording
  {Dataset}_{subject}_{t0}-{t1}s[_mismatch].png   every 30 s window
  {Dataset}_{subject}_all.pdf                all of them, one page each
  (a) raw (raw units) with the annotated breaths (BIDMC: ann1 triangles at the top, ann2 at the bottom;
      CapnoBase: expert labels at the top); (b) preprocessed reference, min-max scaled in the window, with
      the same marks; (c) reconstructed waveform over the preprocessed reference, min-max scaled;
  (d) |FFT| of the raw, preprocessed and reconstructed waveform of the same seconds, each scaled to its peak = 1,
      the peak in the rate band marked by a vertical line and a square, the annotation rates as dashed lines.
  Beside the panels: annotation and FFT rates and their difference; the stored rate, GT and error of the 60 s
  minute that holds the window; PCC; the FFT peaks.

Outputs (D:\PPG2RSP_RRWaveNet_Inspired\261119_Annotation_FFT_Mismatch_Waveforms)
  261119_Annotation_FFT_Mismatch.xlsx
    Summary   per dataset: subjects, usable windows, mismatch windows, subjects drawn
    Subjects  every subject: usable / mismatch windows, largest difference, drawn, MAE and |PCC| (drawn subjects)
    Windows   every 30 s window of every subject: annotation rates, FFT rate, differences, usable, mismatch
    Missing   subjects without stored results (still drawn without the reconstructed waveform)
    Notes

Run: python 261119_Annotation_FFT_Mismatch_Waveforms.py     (Spyder: runfile(...), no options needed)
Options: --threshold-bpm, --window-sec, --datasets, --bidmc-annotators, --raw-predictions, --root-261007,
--root-261013, --root-noise, --output-root, --script-261007, --script-261013, --capnobase-path, --bidmc-path
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
from matplotlib.patches import Patch
from matplotlib.transforms import blended_transform_factory
from scipy.signal import detrend

# ---------------------------------------------------------------------------
# Settings
# ---------------------------------------------------------------------------

NAME = "261119_Annotation_FFT_Mismatch_Waveforms"
HERE = Path(__file__).resolve().parent
WORKSPACE = Path(r"D:\PPG2RSP_RRWaveNet_Inspired")
OUTPUT_ROOT = WORKSPACE / NAME
ROOT_261007 = WORKSPACE / "261007_RRWaveNet_NoEvent_v1"
ROOT_261013 = WORKSPACE / "261013_RRWavNet_STEAM_BIDMC_v1"
ROOT_NOISE = WORKSPACE / "261028_Stem_ablation_noise_v1"
SCRIPTS = {"261007": "261007_RRWaveNet_Ensemble_NoEvent.py", "261013": "261013_RRWaveNet_Adult_SubjectVal_WD.py"}
WINDOW = "01_w10s_nonoverlap"
DATASETS = ("BIDMC", "CapnoBase")
KEYS = {"CapnoBase": "capnobase", "BIDMC": "bidmc"}
SOURCE = {"CapnoBase": "261007", "BIDMC": "261013"}
RAW_LABEL = {"BIDMC": "impedance", "CapnoBase": "capnogram"}
ISSUE_SUBJECTS = {"BIDMC": ("27",), "CapnoBase": ("0115",)}  # MAE issue subjects of the final results
BIDMC_ANNOTATORS = ("ann1", "ann2")
ANNOTATION_TEXT = {"ann1": "ann1 (annotator 1)", "ann2": "ann2 (annotator 2)", "expert": "expert label"}
THRESHOLD_BPM = 2.0
WINDOW_SEC = 30.0
MIN_WINDOW_SEC = 10.0  # a last window shorter than this is not drawn
MEMBER = 0
FFT_MIN_POINTS = 32768  # zero padding of the spectrum (at least)
MAX_FREQ_HZ = 1.0       # frequency axis of the |FFT| panel (at least the rate band)
COLORS = {"raw": "#111111", "raw_fft": "#9a9893", "GT": "#111111", "model": "#d7191c",
          "ann1": "#e08214", "ann2": "#5e3c99", "expert": "#e08214"}
MARKERS = {"ann1": ("v", 0.95), "ann2": ("^", 0.05), "expert": ("v", 0.95)}
SHADE = "#fbe7a1"
MODEL_LW = 2.0
INK, MUTED, GRID = "#1f1f1f", "#6b6a66", "#e4e3df"

_MODULES: Dict[str, object] = {}


# ---------------------------------------------------------------------------
# Scripts and stored results
# ---------------------------------------------------------------------------

def find_script(name: str, given: Optional[Path], roots: Sequence[Path]) -> Path:
    """The script that produced the results: given, written in a run manifest, or found nearby."""
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
        name = "annofft_src_" + re.sub(r"\W", "_", path.stem)
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


def model_files(args: argparse.Namespace, mod, dataset: str, subject: str) -> Optional[Dict[str, Path]]:
    """Minute table and predictions of member m0 of the final model: 261013 (weight decay selected in the
    fold) or 261007 (full), then 261028_Stem_ablation_noise_v1 (full)."""
    tag = mod.safe_tag(subject)
    roots = []
    if SOURCE[dataset] == "261007":
        roots.append(Path(args.root_261007) / "full" / WINDOW / dataset)
    else:
        record = read_json(Path(args.root_261013) / "WD_Selection" / WINDOW / dataset / f"{tag}_selection.json") or {}
        if record.get("Selected_Variant"):
            roots.append(Path(args.root_261013) / record["Selected_Variant"] / WINDOW / dataset)
    roots.append(Path(args.root_noise) / "full" / WINDOW / dataset)
    for root in roots:
        minutes = root / "Fold_Summaries" / f"{tag}_m{MEMBER}_RR60s_windows.csv"
        outputs = root / "Members" / f"m{MEMBER}" / "Predictions" / f"{tag}_test_outputs.npz"
        if minutes.exists() and outputs.exists():
            return {"minutes": minutes, "outputs": outputs, "root": root}
    return None


# ---------------------------------------------------------------------------
# Raw signals and annotations
# ---------------------------------------------------------------------------

def read_bidmc(path: str) -> Dict[str, dict]:
    """Raw impedance (125 Hz) and the breaths of both annotators (MATLAB sample numbers) of every subject."""
    mat = sio.loadmat(path)
    out = {}
    for i, r in enumerate(mat["data"][0]):
        ref = r["ref"][0, 0]
        imp = ref["resp_sig"][0, 0]["imp"][0, 0]
        breaths = ref["breaths"][0, 0]
        out[f"{i + 1:02d}"] = {"raw": np.asarray(imp["v"], float).ravel(),
                               "fs": float(np.asarray(imp["fs"], float).ravel()[0]),
                               "ann1": np.asarray(breaths["ann1"], float).ravel(),
                               "ann2": np.asarray(breaths["ann2"], float).ravel()}
    return out


def read_capnobase_raw(mod, subject: str) -> Optional[dict]:
    """Raw capnogram of a CapnoBase case (read_capnobase_case of 261007)."""
    cfg = mod.DATASET_CONFIGS["capnobase"]
    try:
        case = mod.read_capnobase_case(Path(cfg["data_path"]), subject, cfg["default_fs"], read_labels=False)
    except (OSError, KeyError, ValueError) as exc:
        print(f"  [CapnoBase {subject}] raw capnogram not read: {exc}", flush=True)
        return None
    return {"raw": np.asarray(case["co2"], float), "fs": float(case["co2_fs"])}


def lag_sec(reference: dict) -> float:
    return float(reference["diagnostics"].get("Target_Lag_Sec", 0.0))


def to_timeline(mod, labels: np.ndarray, fs_in: float, reference: dict) -> np.ndarray:
    """Breaths (MATLAB sample numbers of the raw recording) as sample indices of the 64 Hz reference."""
    lag = int(round(lag_sec(reference) * mod.TARGET_FS))
    idx = np.round((np.asarray(labels, float) - 1.0) * (mod.TARGET_FS / fs_in)).astype(np.int64) - lag
    return np.unique(idx[(idx >= 0) & (idx < len(reference["signal"]))])


def annotations_of(mod, dataset: str, ref: dict, raw: Optional[dict], annotators: Sequence[str]) -> Dict[str, np.ndarray]:
    """Annotated breaths (sample indices of the 64 Hz reference) of one recording."""
    if dataset == "BIDMC":
        return {k: to_timeline(mod, raw[k], raw["fs"], ref) for k in annotators} if raw else {}
    if ref.get("label_based") and ref.get("breaths") is not None:
        return {"expert": np.asarray(ref["breaths"], dtype=np.int64)}
    return {}


def label_rate(breaths: np.ndarray, start: int, end: int, fs: float) -> Tuple[int, float]:
    """Breath count and rate (60 / mean breath interval) of an annotation in [start, end)."""
    inside = breaths[(breaths >= start) & (breaths < end)]
    rate = float(60.0 * fs / np.mean(np.diff(inside))) if len(inside) >= 2 else math.nan
    return int(len(inside)), rate


def usable(mod, ref: dict, start: int, end: int) -> Tuple[bool, str]:
    """Reference usable in [start, end) as in the scoring (structure and expert artifact spans)."""
    if float(np.mean(ref["structural_valid"][start:end])) < mod.MIN_USABLE_PROPORTION:
        return False, "ref_structural_invalid"
    if ref.get("artifact_mask") is not None and float(np.mean(ref["artifact_mask"][start:end])) > 0.0:
        return False, "ref_artifact_label"
    return True, ""


# ---------------------------------------------------------------------------
# Spectra
# ---------------------------------------------------------------------------

def longest_finite(x: np.ndarray) -> np.ndarray:
    """The longest run of finite samples (the whole window when it has no gap)."""
    x = np.asarray(x, float)
    ok = np.isfinite(x)
    if ok.all() or not ok.any():
        return x if ok.all() else x[:0]
    padded = np.concatenate([[False], ok, [False]])
    runs = np.flatnonzero(padded[1:] != padded[:-1]).reshape(-1, 2)
    a, b = runs[np.argmax(runs[:, 1] - runs[:, 0])]
    return x[a:b]


def spectrum(x: np.ndarray, fs: float) -> Tuple[np.ndarray, np.ndarray]:
    """Frequency (Hz) and |FFT| of a window: detrend, Hamming window, zero padding (as 261116)."""
    x = longest_finite(x)
    if len(x) < 8 or float(np.std(x)) < 1e-12:
        return np.zeros(1), np.full(1, np.nan)
    n = max(FFT_MIN_POINTS, 1 << int(math.ceil(math.log2(len(x)))))
    return np.fft.rfftfreq(n, d=1.0 / fs), np.abs(np.fft.rfft(detrend(x) * np.hamming(len(x)), n=n))


def band_peak(freq: np.ndarray, mag: np.ndarray, band_bpm: Tuple[float, float]) -> Tuple[float, float]:
    """Largest |FFT| in the rate band: (frequency in Hz, magnitude)."""
    keep = (freq * 60.0 >= band_bpm[0]) & (freq * 60.0 <= band_bpm[1]) & np.isfinite(mag)
    if not keep.any():
        return math.nan, math.nan
    k = int(np.flatnonzero(keep)[np.argmax(mag[keep])])
    return float(freq[k]), float(mag[k])


def fft_rate(x: np.ndarray, fs: float, band_bpm: Tuple[float, float]) -> float:
    """Rate (/min) of the largest |FFT| in the rate band."""
    return band_peak(*spectrum(x, fs), band_bpm)[0] * 60.0


def spectra_of(raw_v: np.ndarray, raw_fs: float, ref_v: np.ndarray, pred_v: np.ndarray, fs: float,
               band_bpm: Tuple[float, float]) -> Dict[str, dict]:
    """|FFT| and its peak in the rate band of the raw, preprocessed and reconstructed waveform of a window."""
    out = {}
    for key, values, rate in (("raw", raw_v, raw_fs), ("GT", ref_v, fs), ("model", pred_v, fs)):
        freq, mag = spectrum(values, rate)
        peak_hz, peak_mag = band_peak(freq, mag, band_bpm)
        out[key] = {"freq": freq, "mag": mag, "peak_hz": peak_hz, "peak_mag": peak_mag}
    return out


# ---------------------------------------------------------------------------
# Windows (selection) and minutes (scores)
# ---------------------------------------------------------------------------

def subject_windows(args, mod, dataset: str, subject: str, record: dict, raw: Optional[dict]) -> pd.DataFrame:
    """Every 30 s window of every recording: annotation rates, FFT rate of the preprocessed reference,
    differences, usable, mismatch."""
    fs = mod.TARGET_FS
    size = int(round(args.window_sec * fs))
    rows = []
    for segment_id, ref in sorted(record["references"].items()):
        ann = annotations_of(mod, dataset, ref, raw, args.bidmc_annotators)
        band = tuple(float(v) for v in ref["profile"]["rr_band_bpm"])
        length = len(ref["signal"])
        for a in range(0, length, size):
            b = min(length, a + size)
            if (b - a) / fs < MIN_WINDOW_SEC - 1e-9:
                continue
            ok, reason = usable(mod, ref, a, b)
            ref_rate = fft_rate(np.asarray(ref["signal"][a:b], float), fs, band)
            row = {"Dataset": dataset, "Subject": subject, "Segment_ID": int(segment_id),
                   "Start_Sec": a / fs, "End_Sec": b / fs, "Usable": ok, "Reason": reason, "FFT_Rate": ref_rate}
            diffs = []
            for key, labels in ann.items():
                n, rate = label_rate(labels, a, b, fs)
                row[f"{key}_Count"], row[f"{key}_Rate"] = n, rate
                row[f"{key}_minus_FFT"] = rate - ref_rate
                if np.isfinite(rate) and np.isfinite(ref_rate):
                    diffs.append((key, abs(rate - ref_rate)))
            over = [k for k, d in diffs if d >= args.threshold_bpm]
            row["Max_Abs_Diff"] = max((d for _, d in diffs), default=math.nan)
            row["Mismatch"] = bool(ok and over)
            row["Mismatch_Annotations"] = ", ".join(over) if ok else ""
            rows.append(row)
    return pd.DataFrame(rows)


def minute_gt(mod, dataset: str, ref: dict, start: int, end: int, ann: Dict[str, np.ndarray]) -> Tuple[bool, str, int]:
    """Validity, reason and GT breath count of one minute, as in the final results (BIDMC: ann1 with the
    expert-label rule of 261013; CapnoBase: count-orig breaths of the capnogram with the RRest rule)."""
    fs = mod.TARGET_FS
    profile = ref["profile"]
    rr_band = profile["rr_band_bpm"]
    ok, reason = usable(mod, ref, start, end)
    if dataset == "BIDMC":
        breaths = ann.get("ann1", np.empty(0, dtype=np.int64))
        count, label_rr = label_rate(breaths, start, end, fs)
        if not ok:
            return False, reason, count
        if not np.isfinite(label_rr):
            return False, "ref_rr_undefined", count
        if not (rr_band[0] <= label_rr <= rr_band[1]):
            return False, "ref_out_of_band", count
        return True, "", count
    breaths = np.asarray(ref["detected_breaths"], dtype=np.int64)
    count = int(np.sum((breaths >= start) & (breaths < end)))
    if not ok:
        return False, reason, count
    agree, why, _, _ = mod.reference_agreement(ref["signal"][start:end], fs, rr_band, profile["fft_subharmonic_check"])
    return bool(agree), why, count


def subject_minutes(args, mod, dataset: str, subject: str, record: dict, raw: Optional[dict],
                    files: Dict[str, Path]) -> pd.DataFrame:
    """Every stored minute: validity and GT of the final results, annotation counts, stored rate, errors, PCC."""
    fs = mod.TARGET_FS
    table = pd.read_csv(files["minutes"], dtype={"Subject": str}, encoding="utf-8-sig")
    rows = []
    for m in table.itertuples(index=False):
        ref = record["references"].get(int(m.Segment_ID))
        if ref is None:
            continue
        t0, t1 = float(m.RR_Window_Start_Sec), float(m.RR_Window_End_Sec)
        a, b = int(round(t0 * fs)), int(round(t1 * fs))
        # BIDMC: ann1 always (the final GT), then the other compared annotator
        keys = ("ann1",) + tuple(k for k in args.bidmc_annotators if k != "ann1")
        ann = annotations_of(mod, dataset, ref, raw, keys)
        valid, reason, count = minute_gt(mod, dataset, ref, a, b, ann)
        rate = float(m.Pred_Wave_FFT_RR)
        row = {"Dataset": dataset, "Subject": subject, "Segment_ID": int(m.Segment_ID), "Start_Sec": t0,
               "End_Sec": t1, "Scored": valid, "Reason": reason, "GT_Count": count, "Model_Rate": rate,
               "AE_GT": abs(rate - count) if valid else math.nan,
               "Wave_PCC": float(getattr(m, "Wave_PCC", math.nan))}
        for key, labels in ann.items():
            n = int(np.sum((labels >= a) & (labels < b)))
            row[f"{key}_Count"] = n
            row[f"AE_{key}"] = abs(rate - n) if valid else math.nan
        rows.append(row)
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Signals of the figures
# ---------------------------------------------------------------------------

def series(outputs: Path, segment_id: int, length: int) -> np.ndarray:
    """Predicted waveform of one recording on its 64 Hz axis (overlap-add mean; NaN where no window)."""
    with np.load(outputs, allow_pickle=False) as data:
        pred, seg, starts = data["predictions"], data["segment_ids"].astype(int), data["window_starts"].astype(int)
    pick = seg == segment_id
    out_sum, cover = np.zeros(length), np.zeros(length)
    for p, s in zip(pred[pick], starts[pick]):
        e = min(length, s + p.shape[0])
        out_sum[s:e] += np.nan_to_num(np.asarray(p, float)[: e - s])
        cover[s:e] += 1
    out = np.full(length, np.nan)
    out[cover > 0] = out_sum[cover > 0] / cover[cover > 0]
    return out


def as_scored(mod, values: np.ndarray, ref: dict) -> np.ndarray:
    """The predicted waveform as the rate was scored: every run of contiguous windows band-passed with the
    recording's respiratory filter (respiratory_filter of the source script)."""
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
    ok = np.isfinite(a) & np.isfinite(b)
    if ok.sum() < 3 or np.std(a[ok]) < 1e-12 or np.std(b[ok]) < 1e-12:
        return math.nan
    return float(np.corrcoef(a[ok], b[ok])[0, 1])


def fmt(v: float, spec: str = "5.1f") -> str:
    return format(v, spec) if v is not None and np.isfinite(v) else "  n/a"


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------

def style(ax) -> None:
    ax.grid(True, color=GRID, lw=0.6)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.tick_params(labelsize=8)


def mark_breaths(ax, ann_sec: Dict[str, np.ndarray], t0: float, t1: float) -> None:
    """Annotated breaths as triangles along the top (ann1, expert) and bottom (ann2) edge of the panel."""
    tr = blended_transform_factory(ax.transData, ax.transAxes)
    for key, t in ann_sec.items():
        marker, y = MARKERS[key]
        t = t[(t >= t0) & (t <= t1)]
        ax.plot(t, np.full(len(t), y), linestyle="none", marker=marker, ms=6, color=COLORS[key],
                mec="white", mew=0.6, transform=tr, zorder=5, clip_on=False)


def draw_spectrum(ax, spectra: Dict[str, dict], ann_rates: Dict[str, float], band_bpm: Tuple[float, float],
                  max_freq: float) -> None:
    """|FFT| of the raw, preprocessed and reconstructed waveform, each scaled to its peak = 1; the peak in the
    rate band marked by a vertical line and a square; the annotation rates as dashed lines."""
    style(ax)
    ax.axvspan(band_bpm[0] / 60.0, band_bpm[1] / 60.0, color="#f3f2ee", lw=0, zorder=0)
    top = 1.0
    for key, color, lw, z in (("raw", COLORS["raw_fft"], 1.2, 2), ("GT", COLORS["GT"], 1.4, 3),
                              ("model", COLORS["model"], MODEL_LW, 4)):
        sp = spectra[key]
        if not (np.isfinite(sp["peak_mag"]) and sp["peak_mag"] > 0):
            continue
        keep = sp["freq"] <= max_freq
        yv = sp["mag"][keep] / sp["peak_mag"]
        top = max(top, float(np.nanmax(yv)))
        ax.plot(sp["freq"][keep], yv, color=color, lw=lw, zorder=z + 2)
        ax.axvline(sp["peak_hz"], color=color, lw=1.0, alpha=0.9, zorder=z)
        ax.plot([sp["peak_hz"]], [1.0], marker="s", ms=6, color=color, mec="white", mew=0.8, zorder=z + 5)
    for key, rate in ann_rates.items():
        if np.isfinite(rate):
            ax.axvline(rate / 60.0, color=COLORS[key], lw=1.3, ls=(0, (4, 2.5)), zorder=7)
    ax.set_xlim(0.0, max_freq)
    ax.set_ylim(0.0, min(top, 4.0) * 1.12)
    ax.set_ylabel("|FFT| (peak = 1)", fontsize=8)
    ax.set_xlabel("Frequency (Hz)", fontsize=8.5)
    sec = ax.secondary_xaxis("top", functions=(lambda f: f * 60.0, lambda r: r / 60.0))
    sec.set_xlabel("breaths/min", fontsize=7.8, color=MUTED)
    sec.tick_params(labelsize=7.5, colors=MUTED, length=0)


def peak_text(spectra: Dict[str, dict], ann_rates: Dict[str, float], band_bpm: Tuple[float, float]) -> str:
    def one(key: str) -> str:
        hz = spectra[key]["peak_hz"]
        return f"{hz:.3f} Hz {hz * 60:5.1f} /min" if np.isfinite(hz) else "n/a"
    lines = [f"FFT peak in {band_bpm[0]:g}-{band_bpm[1]:g} /min", f"raw           {one('raw')}",
             f"preprocessed  {one('GT')}", f"reconstructed {one('model')}"]
    lines += [f"{key:<13} {fmt(rate)} /min" for key, rate in ann_rates.items()]
    return "\n".join(lines)


def draw_page(targets: Sequence[Tuple[Path, PdfPages]], header: Tuple[str, str], titles: Sequence[str],
              t0: float, t1: float, raw_t: np.ndarray, raw_v: np.ndarray, raw_unit: str, ref_t: np.ndarray,
              ref_v: np.ndarray, pred_v: np.ndarray, ann_sec: Dict[str, np.ndarray],
              shades: Sequence[Tuple[float, float]], texts: Sequence[str], spectra: Dict[str, dict],
              ann_rates: Dict[str, float], band_bpm: Tuple[float, float]) -> None:
    """(a) raw, (b) preprocessed, (c) reconstructed over the preprocessed, for the seconds t0-t1, and (d) the
    |FFT| of the three with their peaks."""
    width, side, left, panel, gap, top_in, bottom_in = 13.0, 3.5, 0.75, 1.7, 0.5, 1.25, 1.35
    fft_panel, fft_gap = 1.9, 1.65  # (d) below (c): room for the time axis of (c) and the top axis of (d)
    height = top_in + bottom_in + 3 * panel + 2 * gap + fft_gap + fft_panel
    fig = plt.figure(figsize=(width, height))
    axes = []
    g = minmax(ref_v)
    plot_w = (width - left - side - 0.2) / width
    x_text = (width - side + 0.05) / width
    for k in range(3):
        bottom = bottom_in + fft_panel + fft_gap + (2 - k) * (panel + gap)
        ax = fig.add_axes([left / width, bottom / height, plot_w, panel / height], sharex=axes[0] if axes else None)
        axes.append(ax)
        style(ax)
        for s0, s1 in shades:
            ax.axvspan(max(s0, t0), min(s1, t1), color=SHADE, lw=0, zorder=0)
        if k == 0:
            ax.plot(raw_t, raw_v, color=COLORS["raw"], lw=0.9, zorder=2)
            ax.set_ylabel(raw_unit, fontsize=8)
            mark_breaths(ax, ann_sec, t0, t1)
        elif k == 1:
            ax.plot(ref_t, g, color=COLORS["GT"], lw=1.2, zorder=2)
            ax.set_ylim(-1.3, 1.3)
            ax.set_ylabel("Min-max scaled", fontsize=8)
            mark_breaths(ax, ann_sec, t0, t1)
        else:
            ax.plot(ref_t, g, color=COLORS["GT"], lw=1.1, zorder=2)
            ax.plot(ref_t, minmax(pred_v), color=COLORS["model"], lw=MODEL_LW, zorder=3)
            ax.set_ylim(-1.3, 1.3)
            ax.set_ylabel("Min-max scaled", fontsize=8)
        ax.set_xlim(t0, t1)
        ax.set_title(titles[k], fontsize=9.5, loc="left", pad=10 if k < 2 else 4)
        if k < 2:
            ax.tick_params(labelbottom=False)
        fig.text(x_text, (bottom + panel) / height, texts[k], family="monospace", fontsize=7.6, color=INK, va="top")
    axes[-1].set_xlabel("Time in the recording (s)", fontsize=8.5)
    max_freq = max(MAX_FREQ_HZ, band_bpm[1] / 60.0)
    ax = fig.add_axes([left / width, bottom_in / height, plot_w, fft_panel / height])
    draw_spectrum(ax, spectra, ann_rates, band_bpm, max_freq)
    ax.set_title("(d) |FFT| of the window (detrend, Hamming, zero padding), each scaled to its peak = 1; "
                 f"peak = largest |FFT| in {band_bpm[0]:g}-{band_bpm[1]:g} /min (shaded)",
                 fontsize=9.5, loc="left", pad=24)
    fig.text(x_text, (bottom_in + fft_panel) / height, peak_text(spectra, ann_rates, band_bpm), family="monospace",
             fontsize=7.6, color=INK, va="top")
    handles = [Line2D([], [], color=COLORS["raw"], lw=1.1), Line2D([], [], color=COLORS["model"], lw=MODEL_LW)]
    labels = ["Raw / preprocessed reference", "Reconstructed (final model, m0)"]
    for key in ann_sec:
        marker, _ = MARKERS[key]
        handles.append(Line2D([], [], linestyle="none", marker=marker, ms=6, color=COLORS[key]))
        labels.append(f"{ANNOTATION_TEXT[key]} breath")
    if shades:
        handles.append(Patch(facecolor=SHADE, edgecolor="none"))
        labels.append("annotation vs FFT mismatch (window)")
    fig.legend(handles, labels, loc="lower left", bbox_to_anchor=(left / width, 0.42 / height), ncol=len(labels),
               frameon=False, fontsize=7.8, handlelength=1.8, columnspacing=1.6)
    fft_handles = [Line2D([], [], color=COLORS["raw_fft"], lw=1.2), Line2D([], [], color=COLORS["GT"], lw=1.4),
                   Line2D([], [], color=COLORS["model"], lw=MODEL_LW),
                   Line2D([], [], linestyle="none", marker="s", ms=6, color=INK)]
    fft_labels = ["|FFT| raw", "|FFT| preprocessed", "|FFT| reconstructed", "FFT peak (line and square)"]
    for key in ann_rates:
        fft_handles.append(Line2D([], [], color=COLORS[key], lw=1.3, ls=(0, (4, 2.5))))
        fft_labels.append(f"{key} rate")
    fig.legend(fft_handles, fft_labels, loc="lower left", bbox_to_anchor=(left / width, 0.12 / height),
               ncol=len(fft_labels), frameon=False, fontsize=7.8, handlelength=1.8, columnspacing=1.6)
    fig.text(left / width, 1.0 - 0.2 / height, header[0], fontsize=10.5, color=INK, va="top")
    fig.text(left / width, 1.0 - 0.5 / height, header[1], fontsize=7.8, color=MUTED, va="top", linespacing=1.4)
    for path, pdf in targets:
        path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(path, dpi=160)
        pdf.savefig(fig)
    plt.close(fig)


def panel_titles(dataset: str, ref: dict, raw: Optional[dict]) -> Tuple[str, str, str]:
    band = ref["band_hz"]
    raw_fs = f", {raw['fs']:g} Hz" if raw else ""
    pre = (f"(b) Preprocessed reference (64 Hz, band-passed {band[0]:g}-{band[1]:.2f} Hz"
           + ("" if dataset == "BIDMC" else ", aligned to the PPG") + "; GT waveform of the scoring)")
    return (f"(a) Raw {RAW_LABEL[dataset]} (as recorded{raw_fs})" + ("" if raw else ": not read"), pre,
            "(c) Reconstructed waveform (final model with the stem fusion, m0) over the preprocessed reference")


def draw_subject(args, mod, dataset: str, subject: str, record: dict, raw: Optional[dict],
                 files: Optional[Dict[str, Path]], windows: pd.DataFrame, minutes: pd.DataFrame, row: dict,
                 folder: Path) -> int:
    """Overview and every 30 s window of one subject."""
    fs = mod.TARGET_FS
    tag = mod.safe_tag(subject)
    folder.mkdir(parents=True, exist_ok=True)
    issue = " · MAE issue subject (left out of the final results)" if row["Issue_Subject"] else ""
    if files is None:
        scores = "no stored result of the final model"
    else:
        ann_mae = ", ".join(f"MAE ({k}) {row.get(f'MAE_{k}', math.nan):.2f}" for k in row["Annotations"])
        scores = (f"MAE (final GT) {row['MAE_GT']:.2f}; {ann_mae} breaths/min; |PCC| {row['AbsPCC']:.2f}; "
                  f"{row['Scored_Minutes']} scored minutes")
    head = f"{dataset} {subject}: {scores}{issue}"
    gt_text = "ann1 (expert-label rule)" if dataset == "BIDMC" else "count-orig breaths of the capnogram (RRest rule)"
    sub = (f"Mismatch: |annotation rate - FFT rate of the preprocessed reference| >= {args.threshold_bpm:g} /min in a "
           f"usable {args.window_sec:g} s window (shaded; {row['Mismatch_Windows']} of {row['Usable_Windows']} usable "
           f"windows)\nfinal GT = {gt_text}; model: member m0, "
           + ("raw output" if args.raw_predictions else "band-passed as scored"))
    count = 0
    multi = len(record["references"]) > 1
    with PdfPages(folder / f"{dataset}_{tag}_all.pdf") as pdf:
        for segment_id, ref in sorted(record["references"].items()):
            length = len(ref["signal"])
            pred = series(files["outputs"], int(segment_id), length) if files else np.full(length, np.nan)
            if files and not args.raw_predictions:
                pred = as_scored(mod, pred, ref)
            ref_t = np.arange(length) / fs
            ref_v = np.asarray(ref["signal"], float)
            band = tuple(float(v) for v in ref["profile"]["rr_band_bpm"])
            shift = lag_sec(ref)
            ann = annotations_of(mod, dataset, ref, raw, args.bidmc_annotators)
            ann_sec = {k: v / fs for k, v in ann.items()}
            if raw:
                raw_t = np.arange(len(raw["raw"])) / raw["fs"] - shift
                raw_v, raw_fs = raw["raw"], raw["fs"]
            else:
                raw_t, raw_v, raw_fs = np.zeros(0), np.zeros(0), 1.0
            raw_unit = "Impedance (raw)" if dataset == "BIDMC" else "CO2 (raw)"
            titles = panel_titles(dataset, ref, raw)
            wins = windows[windows["Segment_ID"] == int(segment_id)].sort_values("Start_Sec")
            mins = minutes[minutes["Segment_ID"] == int(segment_id)] if not minutes.empty else minutes
            shades = [(w.Start_Sec, w.End_Sec) for w in wins.itertuples() if w.Mismatch]
            rec = f"rec{segment_id}_" if multi else ""
            # overview: the whole recording
            t0, t1 = 0.0, length / fs
            listing = [f"{w.Start_Sec:.0f}-{w.End_Sec:.0f} s: " + ", ".join(
                f"{k} {getattr(w, f'{k}_Rate'):.1f}" for k in ann) + f" vs FFT {w.FFT_Rate:.1f}"
                for w in wins.itertuples() if w.Mismatch]
            shown = listing[:12] + ([f"... {len(listing) - 12} more"] if len(listing) > 12 else [])
            texts = ["\n".join([f"{k} breaths {len(v):4d}" for k, v in ann.items()]
                               + ["mismatch windows"] + ["  " + s for s in (shown or ["none"])]),
                     f"usable windows   {int(wins.Usable.sum())} of {len(wins)}",
                     ("no stored result" if files is None else
                      f"MAE (final GT) {row['MAE_GT']:5.2f} /min\n"
                      + "".join(f"MAE ({k}) {row.get(f'MAE_{k}', math.nan):8.2f} /min\n" for k in row["Annotations"])
                      + f"|PCC|          {row['AbsPCC']:5.2f}")]
            raw_k = (raw_t >= t0) & (raw_t < t1)
            spectra = spectra_of(raw_v[raw_k], raw_fs, ref_v, pred, fs, band)
            whole = {k: label_rate(v, 0, length, fs)[1] for k, v in ann.items()}
            draw_page([(folder / f"{dataset}_{tag}_{rec}overview.png", pdf)], (f"{head} · whole recording", sub),
                      titles, t0, t1, raw_t[raw_k], raw_v[raw_k], raw_unit, ref_t, ref_v, pred, ann_sec, shades,
                      texts, spectra, whole, band)
            count += 1
            # every 30 s window
            for w in wins.itertuples():
                t0, t1 = float(w.Start_Sec), float(w.End_Sec)
                a, b = int(round(t0 * fs)), int(round(t1 * fs))
                rates = {k: getattr(w, f"{k}_Rate") for k in ann}
                lines = [f"window {t0:.0f}-{t1:.0f} s" + ("" if w.Usable else f"  (not usable: {w.Reason})")]
                lines += [f"{k:<6} {getattr(w, f'{k}_Count'):3d} breaths {fmt(rates[k])} /min" for k in ann]
                lines += [f"FFT (preprocessed)    {fmt(w.FFT_Rate)} /min"]
                lines += [f"{k} - FFT {fmt(getattr(w, f'{k}_minus_FFT'), '+5.1f')} /min" for k in ann]
                if w.Mismatch:
                    lines.append(f">= {args.threshold_bpm:g} /min: {w.Mismatch_Annotations}  <- mismatch")
                inside = mins[(mins.Start_Sec <= t0 + 1e-6) & (mins.End_Sec >= t1 - 1e-6)] if not mins.empty else mins
                r = pearson(ref_v[a:b], pred[a:b])
                if files is None:
                    model_text = "no stored result"
                elif inside.empty:
                    model_text = f"not in a stored 60 s minute\nr (window)     {fmt(r, '5.2f')}"
                else:
                    m = inside.iloc[0]
                    status = "scored" if m.Scored else f"not scored ({m.Reason})"
                    model_text = (f"minute {m.Start_Sec:.0f}-{m.End_Sec:.0f} s, {status}\n"
                                  f"model rate     {fmt(m.Model_Rate)} /min\n"
                                  f"GT (final)     {m.GT_Count:5d} breaths\n"
                                  f"error (final)  {fmt(m.AE_GT, '5.2f')} /min\n"
                                  + "".join(f"error ({k}) {fmt(m.get(f'AE_{k}', math.nan), '7.2f')} /min\n"
                                            for k in row["Annotations"] if k != "ann1")
                                  + f"PCC (minute)   {fmt(m.Wave_PCC, '5.2f')}\n"
                                  f"r (window)     {fmt(r, '5.2f')}")
                texts = ["\n".join(lines), f"FFT rate   {fmt(w.FFT_Rate)} /min", model_text]
                raw_k = (raw_t >= t0) & (raw_t < t1)
                ref_k = slice(a, b)
                spectra = spectra_of(raw_v[raw_k], raw_fs, ref_v[ref_k], pred[ref_k], fs, band)
                name = f"{dataset}_{tag}_{rec}{t0:03.0f}-{t1:03.0f}s{'_mismatch' if w.Mismatch else ''}.png"
                draw_page([(folder / name, pdf)], (f"{head} · {t0:.0f}-{t1:.0f} s", sub), titles, t0, t1,
                          raw_t[raw_k], raw_v[raw_k], raw_unit, ref_t[ref_k], ref_v[ref_k], pred[ref_k], ann_sec,
                          [(t0, t1)] if w.Mismatch else [], texts, spectra, rates, band)
                count += 1
    return count


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def same_subject(a: object, b: object) -> bool:
    a, b = str(a).strip(), str(b).strip()
    return a == b or (a.isdigit() and b.isdigit() and int(a) == int(b))


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Subjects whose annotated breathing rate and the FFT rate of the "
                                            "preprocessed reference differ in a 30 s window: raw, preprocessed and "
                                            "reconstructed waveforms with their FFT peaks, MAE and PCC")
    p.add_argument("--threshold-bpm", type=float, default=THRESHOLD_BPM,
                   help="mismatch when |annotation rate - FFT rate| >= this (default 2 /min)")
    p.add_argument("--window-sec", type=float, default=WINDOW_SEC, help="window of the comparison and the figures")
    p.add_argument("--datasets", nargs="+", choices=list(DATASETS), default=list(DATASETS))
    p.add_argument("--bidmc-annotators", nargs="+", choices=list(BIDMC_ANNOTATORS), default=list(BIDMC_ANNOTATORS),
                   help="BIDMC annotations compared with the FFT rate (default ann1 ann2)")
    p.add_argument("--raw-predictions", action="store_true",
                   help="draw the raw model output (default: band-passed as the rate was scored)")
    p.add_argument("--root-261007", type=Path, default=ROOT_261007)
    p.add_argument("--root-261013", type=Path, default=ROOT_261013)
    p.add_argument("--root-noise", type=Path, default=ROOT_NOISE, help="results of 261028_Stem_ablation_noise_v1")
    p.add_argument("--output-root", type=Path, default=OUTPUT_ROOT)
    p.add_argument("--script-261007", type=Path, default=None)
    p.add_argument("--script-261013", type=Path, default=None)
    p.add_argument("--capnobase-path", type=Path, default=None)
    p.add_argument("--bidmc-path", type=Path, default=None)
    return p.parse_args(list(sys.argv[1:] if argv is None else argv))


def main(argv: Optional[Sequence[str]] = None) -> None:
    args = parse_args(argv)
    out_root = Path(args.output_root)
    out_root.mkdir(parents=True, exist_ok=True)
    window_frames, minute_frames, subject_rows, summary, missing = [], [], [], [], []
    for dataset in args.datasets:
        source = SOURCE[dataset]
        root = args.root_261007 if source == "261007" else args.root_261013
        script = find_script(SCRIPTS[source], args.script_261007 if source == "261007" else args.script_261013, [root])
        mod = load_script(script)
        for key, value in (("capnobase", args.capnobase_path), ("bidmc", args.bidmc_path)):
            if value:
                mod.DATASET_CONFIGS[key]["data_path"] = str(value)
        spec = next(s for s in mod.WINDOW_SPECS if f"{s.order:02d}_{s.name}" == WINDOW)
        print(f"\n{dataset}: preprocessing of {script.name}", flush=True)
        data = mod.load_dataset(KEYS[dataset], spec)
        bidmc_raw = read_bidmc(mod.DATASET_CONFIGS["bidmc"]["data_path"]) if dataset == "BIDMC" else {}
        annotations = tuple(args.bidmc_annotators) if dataset == "BIDMC" else ("expert",)
        drawn = 0
        n_windows = n_usable = n_mismatch = 0
        for subject in sorted(data):
            record = data[subject]
            if dataset == "BIDMC":
                raw = next((v for k, v in bidmc_raw.items() if same_subject(k, subject)), None)
            else:
                raw = read_capnobase_raw(mod, subject)
            windows = subject_windows(args, mod, dataset, subject, record, raw)
            if windows.empty:
                continue
            window_frames.append(windows)
            issue = any(same_subject(subject, x) for x in ISSUE_SUBJECTS[dataset])
            mism = windows[windows.Mismatch]
            n_windows += len(windows)
            n_usable += int(windows.Usable.sum())
            n_mismatch += len(mism)
            row = {"Dataset": dataset, "Subject": subject, "Issue_Subject": issue, "Windows": int(len(windows)),
                   "Usable_Windows": int(windows.Usable.sum()), "Mismatch_Windows": int(len(mism)),
                   "Max_Abs_Diff_Usable": float(windows.loc[windows.Usable, "Max_Abs_Diff"].max())
                   if windows.Usable.any() else math.nan,
                   "Mismatch_List": "; ".join(f"{w.Start_Sec:.0f}-{w.End_Sec:.0f} s ({w.Mismatch_Annotations})"
                                              for w in mism.itertuples()) or "none",
                   "Drawn": False, "Annotations": [k for k in annotations if f"{k}_Rate" in windows.columns]}
            if not mism.empty:
                files = model_files(args, mod, dataset, subject)
                minutes = pd.DataFrame()
                if files is None:
                    missing.append({"Dataset": dataset, "Subject": subject,
                                    "Missing": "no stored result of the final model (m0); drawn without it"})
                else:
                    minutes = subject_minutes(args, mod, dataset, subject, record, raw, files)
                    scored = minutes[minutes.Scored] if not minutes.empty else minutes
                    row["Scored_Minutes"] = int(len(scored))
                    row["MAE_GT"] = float(scored.AE_GT.mean()) if len(scored) else math.nan
                    for k in row["Annotations"]:
                        col = "AE_GT" if (dataset == "BIDMC" and k == "ann1") else f"AE_{k}"
                        row[f"MAE_{k}"] = float(scored[col].mean()) if len(scored) and col in scored else math.nan
                    row["AbsPCC"] = float(scored.Wave_PCC.abs().mean()) if len(scored) else math.nan
                    row["Files"] = str(files["root"])
                    minute_frames.append(minutes)
                folder = out_root / dataset / f"{dataset}_{mod.safe_tag(subject)}"
                row["Figures"] = draw_subject(args, mod, dataset, subject, record, raw, files, windows, minutes, row,
                                              folder)
                row["Drawn"], row["Folder"] = True, str(folder)
                drawn += 1
                scores = ("no stored result" if files is None else
                          f"MAE (final GT) {row['MAE_GT']:.2f}, |PCC| {row['AbsPCC']:.2f}")
                print(f"  {subject}: {len(mism)} of {row['Usable_Windows']} usable windows mismatch "
                      f"(largest |difference| {row['Max_Abs_Diff_Usable']:.1f} /min); {scores}"
                      f"{' [issue]' if issue else ''}; {row['Figures']} figures", flush=True)
            row["Annotations"] = ", ".join(row["Annotations"])
            subject_rows.append(row)
        subjects_ds = [r for r in subject_rows if r["Dataset"] == dataset]
        summary.append({"Dataset": dataset, "Annotations": ", ".join(annotations), "Subjects": len(subjects_ds),
                        "Windows": n_windows, "Usable_Windows": n_usable, "Mismatch_Windows": n_mismatch,
                        "Mismatch_Share_of_Usable": n_mismatch / n_usable if n_usable else math.nan,
                        "Subjects_Drawn": drawn,
                        "Subjects_Drawn_List": ", ".join(r["Subject"] for r in subjects_ds if r["Drawn"]) or "none"})
        print(f"  {dataset}: {drawn} of {len(subjects_ds)} subjects drawn; {n_mismatch} of {n_usable} usable "
              f"{args.window_sec:g} s windows mismatch", flush=True)
        del data
    notes = pd.DataFrame([
        ("Annotations", "BIDMC: ann1 and ann2 (" + ", ".join(args.bidmc_annotators) + " compared); CapnoBase: expert "
                        "breath labels of the capnogram (co2_startexp)"),
        ("Windows", f"every {args.window_sec:g} s window of every recording from its start (a last window shorter "
                    f"than {MIN_WINDOW_SEC:g} s is left out)"),
        ("Annotation rate", "60 / mean breath interval of the annotation in the window (>= 2 breaths)"),
        ("FFT rate", "peak of the |FFT| of the preprocessed reference in the window (detrend, Hamming, zero padding to "
                     ">= 32768 points) in the rate band of the scoring (BIDMC 6-30, CapnoBase 3-60 /min)"),
        ("Usable", "reference structurally valid in at least half of the window and no expert artifact span"),
        ("Mismatch", f"usable window with |annotation rate - FFT rate| >= {args.threshold_bpm:g} /min for an "
                     "annotation; a subject with at least one is drawn"),
        ("Model", "final model with the 1x1 stem fusion, member m0: BIDMC 261013 (weight decay selected in the fold), "
                  "CapnoBase 261007 (full)"),
        ("MAE", "mean |stored rate - breath count| over the scored minutes of the final results: final GT (BIDMC "
                "ann1 with the expert-label rule, CapnoBase count-orig breaths with the RRest rule) and each "
                "annotation on the same minutes"),
        ("|PCC|", "mean |Wave_PCC| of the stored minute tables over the scored minutes"),
        ("Issue subjects", "BIDMC 27 and CapnoBase 0115 (MAE issue subjects left out of the final results) are "
                           "kept and marked"),
    ], columns=["Item", "Description"])
    path = out_root / "261119_Annotation_FFT_Mismatch.xlsx"
    with pd.ExcelWriter(path, engine="openpyxl") as writer:
        pd.DataFrame(summary).round(4).to_excel(writer, sheet_name="Summary", index=False)
        subjects = pd.DataFrame(subject_rows)
        (subjects.sort_values(["Dataset", "Drawn", "Mismatch_Windows"], ascending=[True, False, False]).round(4)
         if not subjects.empty else subjects).to_excel(writer, sheet_name="Subjects", index=False)
        (pd.concat(window_frames, ignore_index=True).round(4) if window_frames else pd.DataFrame()).to_excel(
            writer, sheet_name="Windows", index=False)
        (pd.concat(minute_frames, ignore_index=True).round(4) if minute_frames else pd.DataFrame()).to_excel(
            writer, sheet_name="Minutes", index=False)
        pd.DataFrame(missing, columns=["Dataset", "Subject", "Missing"]).to_excel(writer, sheet_name="Missing",
                                                                                   index=False)
        notes.to_excel(writer, sheet_name="Notes", index=False)
    print(f"\nFigures and table under {out_root}")


if __name__ == "__main__":
    main()
