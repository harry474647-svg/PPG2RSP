# -*- coding: utf-8 -*-
r"""261118_StemFusion_Threshold_Groups_Waveforms: the subjects grouped by how the 1x1 stem fusion changes their
RR error at 2 and 3 breaths/min, and their waveforms in every 30 s window (the figures of
261115_StemFusion_MAEDiff_Waveforms).

Groups, for every threshold T (--thresholds, default 2 and 3 breaths/min); MAE = subject MAE of a model
  1_change     |MAE O - MAE X| >= T, both directions (Fusion O better: O < X; Fusion O worse: O > X)
  2_improved   MAE X > T and MAE O < T: above T without the stem fusion, below T with it
  3_worsened   MAE X < T and MAE O > T: the opposite
Clinical criteria (--clinical-mae, --clinical-pcc, default 3.0 breaths/min and 0.6), with |PCC| = the mean over the
scored minutes of the absolute waveform PCC of every minute (Wave_PCC of the stored minute tables); each criterion
on its own, so a subject can be in several of them
  1_MAE        MAE < 3
  2_absPCC     |PCC| > 0.6
  3_both       MAE < 3 and |PCC| > 0.6
  better       Fusion X does not meet the criterion, Fusion O meets it
  worse        Fusion X meets the criterion, Fusion O does not
A subject can be in several groups and at both thresholds; it is drawn once and every figure is saved in each
of its group folders. The subjects with an MAE issue (CapnoBase 0115, BIDMC 27, STEAM2 12_YJJ, 22_LMS,
26_CKK) are left out of the groups as in the final results (--keep-issue-subjects keeps them, marked).

Models (stored outputs, nothing is trained): the single model m0 of every LOSOCV fold of the
final configuration, with (Fusion O) and without (Fusion X) the 1x1 stem fusion
  CapnoBase      261007 (D:\PPG2RSP_RRWaveNet_Inspired\261007_RRWaveNet_NoEvent_v1):
                 full / no_stem_fusion, member m0
  BIDMC, STEAM2  Fusion O: 261013 (D:\PPG2RSP_RRWaveNet_Inspired\261013_RRWavNet_STEAM_BIDMC_v1),
                 member m0 of the weight decay selected in the fold (WD_Selection);
                 Fusion X: 261028_Stem_ablation_noise_v1 (no_stem_fusion, member m0)
  (each model is read from 261028_Stem_ablation_noise_v1 first when it is there, otherwise from
  261007 / 261013, as 261028_Stem_ablation_quantative_result reads them)
  Fold_Summaries\{subject}_m0_RR60s_windows.csv            every 60 s minute: the stored rate
                                                           of the predicted waveform
  Members\m0\Predictions\{subject}_test_outputs.npz         the predicted waveform (64 Hz)

Ground truth of the rates (the rescored GT of the stem fusion tables)
  CapnoBase  breaths detected by count-orig peak detection on the capnogram; a minute is scored
             when it passes the RRest agreement rule (expert CO2 artifact spans excluded)
  BIDMC      breaths of annotator 1 (bidmc_data.mat ref.breaths.ann1); a minute is scored with the
             expert-label rule of 261013 (usable reference, >= 2 breaths, rate in 6-30 /min);
             --ann1-nogap also leaves out minutes overlapping an ann1 gap (> 2.5 x median interval)
  STEAM2     the stored GT (no expert annotation): breaths detected on the belt, RRest rule
  The reference signal and the breaths come from the preprocessing of the script that produced
  the results (261007 for CapnoBase, 261013 for BIDMC / STEAM2), loaded as a module: its data
  paths are used (--capnobase-path, --bidmc-path, --steam2-ppg-dir, --steam2-rsp-dir override).
  Per subject, FFT MAE = mean |stored rate - GT breath count| over the scored minutes, for each model on
  the same minutes.

Figures (as 261115_StemFusion_MAEDiff_Waveforms): every 30 s window of every recording of a grouped subject
(the windows of a minute are its first and second half; stretches outside the scored minute grid in 30 s steps):
  (a) GT vs Fusion O (top, deep red), (b) GT vs Fusion X (bottom, deep blue; both drawn thicker than the GT
  and above it), the same seconds, no model shifted in time; every waveform min-max scaled in the window;
  GT = the reference signal the GT breaths were found on (CapnoBase capnogram, BIDMC impedance, STEAM2 belt;
  band-passed and aligned as in the scoring); beside each panel the minute's GT rate, the model's rate and error
  and the Pearson r in the 30 s window; the header names every group of the subject.
  The model waveforms are drawn as the rate was scored: every run of contiguous windows band-passed with the
  recording's respiratory filter (--raw-predictions: the raw output).
FFT figures (FFT\ folder of every subject folder): the |FFT| of the same 30 s window (as 261116: detrend,
  Hamming window, zero padding to >= 32768 points), (a) GT vs Fusion O, (b) GT vs Fusion X (Fusion O and X are not
  drawn together); every spectrum scaled to its peak = 1, the peak (largest |FFT| in the recording's rate band of
  the scoring) marked by a vertical line and a square; the peaks beside each panel. Windows shorter than 10 s
  get no FFT figure.

Outputs (D:\PPG2RSP_RRWaveNet_Inspired\261118_StemFusion_Threshold_Groups_Waveforms)
  T{T}bpm\1_change_ge{T}bpm\{Dataset}_{subject}_{O_better|O_worse}\{recording}_{t0}-{t1}s.png
  T{T}bpm\2_improved_Xover{T}_Ounder{T}\{Dataset}_{subject}\...
  T{T}bpm\3_worsened_Xunder{T}_Oover{T}\{Dataset}_{subject}\...
  Clinical_MAE{3}_absPCC{0.6}\{1_MAE_lt3 | 2_absPCC_gt0.6 | 3_MAE_lt3_and_absPCC_gt0.6}\
      1_better_Xfails_Omeets\{Dataset}_{subject}\...  and  2_worse_Xmeets_Ofails\{Dataset}_{subject}\...
      every 30 s window, and {Dataset}_{subject}_windows.pdf (all windows, one page each) in every folder;
      FFT\{recording}_{t0}-{t1}s_FFT.png and FFT\{Dataset}_{subject}_FFT.pdf in every subject folder
  261118_StemFusion_Threshold_Groups.xlsx
    Group_Summary  per criterion (threshold or clinical), group and dataset: number of subjects and their list
                   ("none" when empty)
    Groups         every subject of every group: MAE X, MAE O, difference, |PCC| X, |PCC| O, direction, folder
    Subjects       every subject: MAE X, MAE O, difference, |PCC| X, |PCC| O, each clinical criterion with each
                   model, scored minutes, its groups
    Minutes        every minute of the grouped subjects: GT, rates and errors of both models
    Missing        subjects without stored results of both models or without a scored minute
    Notes

Run: python 261118_StemFusion_Threshold_Groups_Waveforms.py     (Spyder: runfile(...), no options needed)
Options: --thresholds (default 2 3), --clinical-mae (3), --clinical-pcc (0.6), --datasets, --keep-issue-subjects,
--ann1-nogap, --window-sec,
--raw-predictions, --root-261007, --root-261013, --root-noise, --output-root, --script-261007, --script-261013,
--capnobase-path, --bidmc-path, --steam2-ppg-dir, --steam2-rsp-dir
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
import scipy.io as sio
from scipy.signal import detrend
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.lines import Line2D

# ---------------------------------------------------------------------------
# Settings
# ---------------------------------------------------------------------------

NAME = "261118_StemFusion_Threshold_Groups_Waveforms"
HERE = Path(__file__).resolve().parent
WORKSPACE = Path(r"D:\PPG2RSP_RRWaveNet_Inspired")
OUTPUT_ROOT = WORKSPACE / NAME
ROOT_261007 = WORKSPACE / "261007_RRWaveNet_NoEvent_v1"
ROOT_261013 = WORKSPACE / "261013_RRWavNet_STEAM_BIDMC_v1"
ROOT_NOISE = WORKSPACE / "261028_Stem_ablation_noise_v1"
SCRIPTS = {"261007": "261007_RRWaveNet_Ensemble_NoEvent.py", "261013": "261013_RRWaveNet_Adult_SubjectVal_WD.py"}
WINDOW = "01_w10s_nonoverlap"
DATASETS = ("CapnoBase", "BIDMC", "STEAM2")
KEYS = {"CapnoBase": "capnobase", "BIDMC": "bidmc", "STEAM2": "steam2"}
SOURCE = {"CapnoBase": "261007", "BIDMC": "261013", "STEAM2": "261013"}
NEW_GT = {"CapnoBase": "peak detection on the capnogram", "BIDMC": "ann1", "STEAM2": "stored GT (no annotation)"}
MODALITY = {"CapnoBase": "capnogram", "BIDMC": "impedance", "STEAM2": "belt"}
EXCLUDED = {"STEAM2": ("22_LMS", "26_CKK", "12_YJJ"), "BIDMC": ("27",), "CapnoBase": ("0115",)}
THRESHOLDS = (2.0, 3.0)  # breaths/min
GROUPS = ("1_change", "2_improved", "3_worsened")
CLINICAL_MAE = 3.0  # clinical criteria: MAE < 3.0 breaths/min, |PCC| > 0.6, and both
CLINICAL_PCC = 0.6
CLINICAL_CRITERIA = ("mae", "pcc", "both")
CLINICAL_GROUPS = ("better", "worse")
WINDOW_SEC = 30.0
ANN1_GAP_FACTOR = 2.5
MEMBER = 0
COLORS = {"GT": "#111111", "O": "#d7191c", "X": "#0b5fc4"}  # top panel (Fusion O) deep red, bottom (Fusion X) deep blue
MODEL_LW = 2.2  # model lines drawn thicker than the GT and above it
FFT_MIN_POINTS = 32768  # zero padding of the spectrum (64 Hz: 0.002 Hz steps, a smooth curve)
MAX_FREQ_HZ = 1.0      # frequency axis of the FFT figures (at least the rate band)
FFT_MIN_SEC = 10.0     # shorter windows get no FFT figure
LABELS = {"X": "Fusion X (without stem fusion)", "O": "Fusion O (with stem fusion)"}
INK, MUTED, GRID = "#1f1f1f", "#6b6a66", "#e4e3df"


def group_folder(group: str, threshold: float) -> str:
    t = f"{threshold:g}"
    return {"1_change": f"1_change_ge{t}bpm", "2_improved": f"2_improved_Xover{t}_Ounder{t}",
            "3_worsened": f"3_worsened_Xunder{t}_Oover{t}"}[group]


def group_text(group: str, threshold: float, diff: float) -> str:
    t = f"{threshold:g}"
    if group == "1_change":
        return f"|O - X| >= {t} ({'Fusion O better' if diff < 0 else 'Fusion O worse'})"
    if group == "2_improved":
        return f"improved: X > {t} -> O < {t}"
    return f"worsened: X < {t} -> O > {t}"


def clinical_dir(args) -> str:
    return f"Clinical_MAE{args.clinical_mae:g}_absPCC{args.clinical_pcc:g}"


def clinical_folder(group: str) -> str:
    return {"better": "1_better_Xfails_Omeets", "worse": "2_worse_Xmeets_Ofails"}[group]


def clinical_rule(criterion: str, args) -> str:
    mae, pcc = f"MAE < {args.clinical_mae:g}", f"|PCC| > {args.clinical_pcc:g}"
    return {"mae": mae, "pcc": pcc, "both": f"{mae} and {pcc}"}[criterion]


def clinical_criterion_folder(criterion: str, args) -> str:
    mae, pcc = f"MAE_lt{args.clinical_mae:g}", f"absPCC_gt{args.clinical_pcc:g}"
    return {"mae": f"1_{mae}", "pcc": f"2_{pcc}", "both": f"3_{mae}_and_{pcc}"}[criterion]


def meets_clinical(criterion: str, mae: float, abs_pcc: float, args) -> bool:
    ok_mae, ok_pcc = bool(mae < args.clinical_mae), bool(abs_pcc > args.clinical_pcc)
    return {"mae": ok_mae, "pcc": ok_pcc, "both": ok_mae and ok_pcc}[criterion]


def clinical_failed(mae: float, abs_pcc: float, args) -> str:
    """Which part of the clinical criterion a model misses."""
    parts = [n for n, bad in (("MAE", not mae < args.clinical_mae), ("|PCC|", not abs_pcc > args.clinical_pcc)) if bad]
    return " + ".join(parts)


def clinical_memberships(row: dict, args) -> List[Tuple[str, str]]:
    """(criterion, better / worse) of every clinical criterion whose status changes: better = Fusion X misses the
    criterion and Fusion O meets it, worse = the opposite."""
    out = []
    for criterion in CLINICAL_CRITERIA:
        x = meets_clinical(criterion, row["MAE_X"], row["AbsPCC_X"], args)
        o = meets_clinical(criterion, row["MAE_O"], row["AbsPCC_O"], args)
        if o and not x:
            out.append((criterion, "better"))
        elif x and not o:
            out.append((criterion, "worse"))
    return out


def clinical_text(criterion: str, group: str, row: dict, args) -> str:
    rule = clinical_rule(criterion, args)
    if criterion == "mae":
        values = f"MAE {row['MAE_X']:.2f} -> {row['MAE_O']:.2f}"
    elif criterion == "pcc":
        values = f"|PCC| {row['AbsPCC_X']:.2f} -> {row['AbsPCC_O']:.2f}"
    elif group == "better":
        values = f"X misses {clinical_failed(row['MAE_X'], row['AbsPCC_X'], args)}"
    else:
        values = f"O misses {clinical_failed(row['MAE_O'], row['AbsPCC_O'], args)}"
    return f"clinical ({rule}) {group}: {values}"


def memberships(mae_x: float, mae_o: float, thresholds: Sequence[float]) -> List[Tuple[float, str]]:
    """(threshold, group) of every group the subject belongs to."""
    out = []
    for t in thresholds:
        if abs(mae_o - mae_x) >= t:
            out.append((t, "1_change"))
        if mae_x > t and mae_o < t:
            out.append((t, "2_improved"))
        if mae_x < t and mae_o > t:
            out.append((t, "3_worsened"))
    return out


# ---------------------------------------------------------------------------
# Source scripts and stored files
# ---------------------------------------------------------------------------

_MODULES: Dict[str, object] = {}


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
        name = "maediff_src_" + re.sub(r"\W", "_", path.stem)
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


def model_files(args: argparse.Namespace, mod, dataset: str, model: str, subject: str) -> Optional[Dict[str, Path]]:
    """Minute table and predictions of member m0 of a fold (Fusion O / X): 261028_Stem_ablation_noise_v1 first,
    then 261007 (CapnoBase) or 261013 (BIDMC, STEAM2, Fusion O only)."""
    tag = mod.safe_tag(subject)
    variant = "full" if model == "O" else "no_stem_fusion"
    roots = [Path(args.root_noise) / variant / WINDOW / dataset]
    if SOURCE[dataset] == "261007":
        roots.append(Path(args.root_261007) / variant / WINDOW / dataset)
    elif model == "O":
        record = read_json(Path(args.root_261013) / "WD_Selection" / WINDOW / dataset / f"{tag}_selection.json") or {}
        if record.get("Selected_Variant"):
            roots.append(Path(args.root_261013) / record["Selected_Variant"] / WINDOW / dataset)
    for root in roots:
        minutes = root / "Fold_Summaries" / f"{tag}_m{MEMBER}_RR60s_windows.csv"
        outputs = root / "Members" / f"m{MEMBER}" / "Predictions" / f"{tag}_test_outputs.npz"
        if minutes.exists() and outputs.exists():
            return {"minutes": minutes, "outputs": outputs, "root": root}
    return None


# ---------------------------------------------------------------------------
# Ground truth (the rescored GT)
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


def gap_spans(labels: np.ndarray, length: int, factor: float = ANN1_GAP_FACTOR) -> List[Tuple[int, int]]:
    if len(labels) < 3:
        return []
    limit = factor * float(np.median(np.diff(labels)))
    spans = [(0, int(labels[0]))] if labels[0] > limit else []
    spans += [(int(a) + 1, int(b)) for a, b in zip(labels[:-1], labels[1:]) if b - a > limit]
    if length - 1 - labels[-1] > limit:
        spans.append((int(labels[-1]) + 1, int(length)))
    return spans


def minute_gt(mod, ref: dict, start: int, end: int, breaths: np.ndarray, label_based: bool) -> Tuple[bool, str, int]:
    """The reference part of evaluate_subject_minutes: validity, reason and GT breath count of one minute."""
    fs = mod.TARGET_FS
    profile = ref["profile"]
    rr_band, sub = profile["rr_band_bpm"], profile["fft_subharmonic_check"]
    inside = (breaths >= start) & (breaths < end)
    count, labels = int(inside.sum()), breaths[inside]
    label_rr = float(60.0 * fs / np.mean(np.diff(labels))) if label_based and len(labels) >= 2 else math.nan
    if float(np.mean(ref["structural_valid"][start:end])) < mod.MIN_USABLE_PROPORTION:
        return False, "ref_structural_invalid", count
    if ref.get("artifact_mask") is not None and float(np.mean(ref["artifact_mask"][start:end])) > 0.0:
        return False, "ref_artifact_label", count
    if label_based:
        if not np.isfinite(label_rr):
            return False, "ref_rr_undefined", count
        if not (rr_band[0] <= label_rr <= rr_band[1]):
            return False, "ref_out_of_band", count
        return True, "", count
    ok, reason, _, _ = mod.reference_agreement(ref["signal"][start:end], fs, rr_band, sub)
    return bool(ok), reason, count


def gt_breaths(dataset: str, mod, ref: dict, ann1: Optional[np.ndarray]) -> Tuple[np.ndarray, bool]:
    """GT breaths of a recording and whether the expert-label rule applies."""
    if dataset == "CapnoBase":
        return np.asarray(ref["detected_breaths"], dtype=np.int64), False
    if dataset == "BIDMC":
        return ann1_timeline(mod, ann1, ref), True
    return np.asarray(ref["breaths"], dtype=np.int64), bool(ref["label_based"])


# ---------------------------------------------------------------------------
# Subjects
# ---------------------------------------------------------------------------

def subject_minutes(args, mod, dataset: str, subject: str, record: dict, files: Dict[str, dict],
                    ann1: Optional[np.ndarray]) -> pd.DataFrame:
    """Every minute of a subject with the rescored GT and the stored rates of both models."""
    fs = mod.TARGET_FS
    tables = {}
    for model in ("O", "X"):
        t = pd.read_csv(files[model]["minutes"], dtype={"Subject": str}, encoding="utf-8-sig")
        tables[model] = t.set_index(["Segment_ID", "RR_Window_Start_Sec"])
    keys = tables["O"].index.intersection(tables["X"].index)
    rows = []
    for segment_id, t0 in keys:
        ref = record["references"].get(int(segment_id))
        if ref is None:
            continue
        o, x = tables["O"].loc[(segment_id, t0)], tables["X"].loc[(segment_id, t0)]
        t1 = float(o["RR_Window_End_Sec"])
        a, b = int(round(float(t0) * fs)), int(round(t1 * fs))
        if dataset == "STEAM2":  # no new GT: the stored one
            valid, reason, count = bool(o["Ref_Valid"]), str(o.get("Ref_Exclusion_Reason", "") or ""), int(o["GT_Breath_Count"])
            gap = False
        else:
            breaths, label_based = gt_breaths(dataset, mod, ref, ann1)
            valid, reason, count = minute_gt(mod, ref, a, b, breaths, label_based)
            gap = dataset == "BIDMC" and any(lo < b and hi > a for lo, hi in gap_spans(breaths, len(ref["signal"])))
            if gap and args.ann1_nogap and valid:
                valid, reason = False, "ref_ann1_gap"
        rows.append({"Dataset": dataset, "Subject": subject, "Segment_ID": int(segment_id), "Start_Sec": float(t0),
                     "End_Sec": t1, "Scored": valid, "Reason": "" if valid else reason, "GT_Count": count,
                     "Ann1_Gap": gap, "Rate_X": float(x["Pred_Wave_FFT_RR"]), "Rate_O": float(o["Pred_Wave_FFT_RR"]),
                     "AE_X": abs(float(x["Pred_Wave_FFT_RR"]) - count) if valid else math.nan,
                     "AE_O": abs(float(o["Pred_Wave_FFT_RR"]) - count) if valid else math.nan,
                     "PCC_X": float(x.get("Wave_PCC", math.nan)), "PCC_O": float(o.get("Wave_PCC", math.nan))})
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Figures
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


def spectrum(x: np.ndarray, fs: float) -> Tuple[np.ndarray, np.ndarray]:
    """Frequency (Hz) and |FFT| of a window: detrend, Hamming window, zero padding (as 261116)."""
    x = np.asarray(x, dtype=np.float64)
    if len(x) < 8 or not np.isfinite(x).all() or float(np.std(x)) < 1e-12:
        return np.zeros(1), np.full(1, np.nan)
    windowed = detrend(x) * np.hamming(len(x))
    n_fft = max(FFT_MIN_POINTS, 1 << int(math.ceil(math.log2(len(x)))))
    return np.fft.rfftfreq(n_fft, d=1.0 / fs), np.abs(np.fft.rfft(windowed, n=n_fft))


def band_peak(freq: np.ndarray, mag: np.ndarray, band: Tuple[float, float]) -> Tuple[float, float]:
    """Largest |FFT| in the rate band: (frequency, magnitude)."""
    keep = (freq >= band[0]) & (freq <= band[1]) & np.isfinite(mag)
    if not keep.any():
        return math.nan, math.nan
    k = int(np.flatnonzero(keep)[np.argmax(mag[keep])])
    return float(freq[k]), float(mag[k])


def windows_of(minutes: pd.DataFrame, covered: np.ndarray, fs: int, size: float) -> List[Tuple[float, float]]:
    """Every 30 s window: the halves of every minute, then the covered stretches outside the minutes."""
    out = []
    taken = np.zeros(len(covered), bool)
    for m in minutes.itertuples(index=False):
        for k in range(int(round((m.End_Sec - m.Start_Sec) / size))):
            t0 = m.Start_Sec + k * size
            out.append((t0, t0 + size))
        taken[int(round(m.Start_Sec * fs)):int(round(m.End_Sec * fs))] = True
    rest = covered & ~taken
    padded = np.concatenate([[False], rest, [False]])
    edges = np.flatnonzero(padded[1:] != padded[:-1]).reshape(-1, 2)
    for a, b in edges:
        t = a / fs
        while t < b / fs - 1e-9:
            out.append((t, min(t + size, b / fs)))
            t += size
    return sorted(out)


def draw_window(targets: Sequence[Tuple[Path, PdfPages]], header: Tuple[str, str, str], t: np.ndarray, gt: np.ndarray,
                preds: Dict[str, np.ndarray], texts: Dict[str, str]) -> None:
    """One 30 s window, saved as PNG and PDF page in every target (group folder)."""
    extra = header[2].count("\n")  # the group line may wrap
    width, side, left, panel, gap, top_in, bottom_in = 12.0, 3.4, 0.7, 1.75, 0.45, 1.15 + 0.17 * extra, 0.6
    height = top_in + bottom_in + 2 * panel + gap
    fig = plt.figure(figsize=(width, height))
    g = minmax(gt)
    axes = []
    for k, model in enumerate(("O", "X")):
        bottom = bottom_in + (1 - k) * (panel + gap)
        ax = fig.add_axes([left / width, bottom / height, (width - left - side - 0.2) / width, panel / height],
                          sharex=axes[0] if axes else None)
        axes.append(ax)
        ax.plot(t, g, color=COLORS["GT"], lw=1.4, label="GT", zorder=2)
        ax.plot(t, minmax(preds[model]), color=COLORS[model], lw=MODEL_LW, zorder=3)
        ax.set_xlim(t[0], t[-1])
        ax.set_ylim(-1.2, 1.25)
        ax.set_ylabel("Min-max scaled", fontsize=8)
        ax.grid(True, color=GRID, lw=0.6)
        ax.set_axisbelow(True)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
        ax.set_title(f"({'ab'[k]}) GT vs {LABELS[model]}", fontsize=9.5, loc="left", pad=3)
        if k == 0:
            ax.tick_params(labelbottom=False)
        x = (width - side + 0.05) / width
        y = (bottom + panel) / height
        fig.legend([Line2D([], [], color=COLORS["GT"], lw=1.4), Line2D([], [], color=COLORS[model], lw=MODEL_LW)],
                   ["GT", LABELS[model]], loc="upper left", bbox_to_anchor=(x, y), frameon=False,
                   fontsize=7.8, borderaxespad=0.0)
        fig.text(x + 0.004, y - 0.75 / height, texts[model], family="monospace", fontsize=7.6, color=INK, va="top")
    axes[-1].set_xlabel("Time in the recording (s)")
    fig.text(left / width, 1.0 - 0.18 / height, header[0], fontsize=10.5, color=INK, va="top")
    fig.text(left / width, 1.0 - 0.47 / height, header[1], fontsize=7.6, color=MUTED, va="top")
    fig.text(left / width, 1.0 - 0.68 / height, header[2], fontsize=7.6, color=INK, va="top", linespacing=1.4)
    for path, pdf in targets:
        path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(path, dpi=160)
        pdf.savefig(fig)
    plt.close(fig)


def draw_fft(targets: Sequence[Tuple[Path, PdfPages]], header: Tuple[str, str, str], spectra: Dict[str, dict],
             texts: Dict[str, str], max_freq: float) -> None:
    """|FFT| of one window: (a) GT vs Fusion O, (b) GT vs Fusion X; each spectrum scaled to its peak = 1."""
    extra = header[2].count("\n")
    width, side, left, panel, gap, top_in, bottom_in = 12.0, 3.4, 0.7, 1.75, 0.45, 1.15 + 0.17 * extra, 0.6
    height = top_in + bottom_in + 2 * panel + gap
    fig = plt.figure(figsize=(width, height))
    axes = []
    for k, model in enumerate(("O", "X")):
        bottom = bottom_in + (1 - k) * (panel + gap)
        ax = fig.add_axes([left / width, bottom / height, (width - left - side - 0.2) / width, panel / height],
                          sharex=axes[0] if axes else None)
        axes.append(ax)
        top = 1.0
        for key, lw, z in (("GT", 1.4, 2), (model, MODEL_LW, 3)):
            sp = spectra[key]
            if not (np.isfinite(sp["peak_mag"]) and sp["peak_mag"] > 0):
                continue
            keep = sp["freq"] <= max_freq
            yv = sp["mag"][keep] / sp["peak_mag"]
            top = max(top, float(np.nanmax(yv)))
            ax.plot(sp["freq"][keep], yv, color=COLORS[key], lw=lw, zorder=z + 2)
            ax.axvline(sp["peak_hz"], color=COLORS[key], lw=1.0, alpha=0.9, zorder=z)
            ax.plot([sp["peak_hz"]], [1.0], marker="s", ms=5.5, color=COLORS[key], mec="white", mew=0.8, zorder=z + 4)
        ax.set_xlim(0.0, max_freq)
        ax.set_ylim(0.0, top * 1.12)
        ax.set_ylabel("|FFT| (peak = 1)", fontsize=8)
        ax.grid(True, color=GRID, lw=0.6)
        ax.set_axisbelow(True)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
        ax.set_title(f"({'ab'[k]}) |FFT|: GT vs {LABELS[model]}", fontsize=9.5, loc="left", pad=3)
        if k == 0:
            ax.tick_params(labelbottom=False)
        x = (width - side + 0.05) / width
        y = (bottom + panel) / height
        fig.legend([Line2D([], [], color=COLORS["GT"], lw=1.4), Line2D([], [], color=COLORS[model], lw=MODEL_LW)],
                   ["GT", LABELS[model]], loc="upper left", bbox_to_anchor=(x, y), frameon=False,
                   fontsize=7.8, borderaxespad=0.0)
        fig.text(x + 0.004, y - 0.75 / height, texts[model], family="monospace", fontsize=7.6, color=INK, va="top")
    axes[-1].set_xlabel("Frequency (Hz)")
    fig.text(left / width, 1.0 - 0.18 / height, header[0], fontsize=10.5, color=INK, va="top")
    fig.text(left / width, 1.0 - 0.47 / height, header[1], fontsize=7.6, color=MUTED, va="top")
    fig.text(left / width, 1.0 - 0.68 / height, header[2], fontsize=7.6, color=INK, va="top", linespacing=1.4)
    for path, pdf in targets:
        path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(path, dpi=160)
        pdf.savefig(fig)
    plt.close(fig)


def draw_subject(args, mod, dataset: str, subject: str, record: dict, files: Dict[str, dict], minutes: pd.DataFrame,
                 summary: dict, places: List[Tuple[tuple, Path, str]]) -> int:
    """Every 30 s window of a subject (waveform and FFT figure), saved in the folder of each of its groups
    (places: key, folder, text); the FFT figures in the FFT subfolder."""
    fs = mod.TARGET_FS
    tag = mod.safe_tag(subject)
    diff = summary["Diff_O_minus_X"]
    folders = {key: folder for key, folder, _ in places}
    multi = len(record["references"]) > 1
    count = 0
    issue = " · MAE issue subject" if summary["Issue_Subject"] else ""
    head = (f"{dataset} {subject}: MAE X {summary['MAE_X']:.2f}, MAE O {summary['MAE_O']:.2f} breaths/min "
            f"(O - X {diff:+.2f}; {'Fusion O better' if diff < 0 else 'Fusion O worse'}); "
            f"|PCC| X {summary['AbsPCC_X']:.2f}, O {summary['AbsPCC_O']:.2f}{issue}")
    group_line = "\n".join(textwrap.wrap("groups: " + "; ".join(text for _, _, text in places), 185,
                                          subsequent_indent="    "))
    pdfs, fft_pdfs = {}, {}
    try:
        for key, folder in folders.items():
            (folder / "FFT").mkdir(parents=True, exist_ok=True)
            pdfs[key] = PdfPages(folder / f"{dataset}_{tag}_windows.pdf")
            fft_pdfs[key] = PdfPages(folder / "FFT" / f"{dataset}_{tag}_FFT.pdf")
        for segment_id, ref in sorted(record["references"].items()):
            length = len(ref["signal"])
            preds = {m: series(files[m]["outputs"], int(segment_id), length) for m in ("O", "X")}
            if not args.raw_predictions:
                preds = {m: as_scored(mod, v, ref) for m, v in preds.items()}
            covered = np.isfinite(preds["O"]) & np.isfinite(preds["X"])
            if not covered.any():
                continue
            mins = minutes[minutes["Segment_ID"] == int(segment_id)].sort_values("Start_Sec")
            rate_band = tuple(float(v) / 60.0 for v in ref["profile"]["rr_band_bpm"])
            max_freq = max(MAX_FREQ_HZ, rate_band[1])
            for t0, t1 in windows_of(mins, covered, fs, args.window_sec):
                a, b = int(round(t0 * fs)), int(round(t1 * fs))
                if b - a < 2 * fs:
                    continue
                t = np.arange(a, b) / fs
                inside = mins[(mins.Start_Sec <= t0 + 1e-6) & (mins.End_Sec >= t1 - 1e-6)]
                texts = {}
                for m in ("X", "O"):
                    r = pearson(np.asarray(ref["signal"][a:b], float), preds[m][a:b])
                    if inside.empty:
                        texts[m] = f"not in a scored 60 s minute\nr (this window) {r:.2f}"
                    else:
                        row = inside.iloc[0]
                        status = "scored" if row.Scored else f"not scored ({row.Reason})"
                        ae = f"{row[f'AE_{m}']:.2f}" if row.Scored else "-"
                        texts[m] = (f"minute {row.Start_Sec:.0f}-{row.End_Sec:.0f} s, {status}\n"
                                    f"GT rate      {row.GT_Count:5.1f} /min\n"
                                    f"model rate   {row[f'Rate_{m}']:5.1f} /min\n"
                                    f"error (AE)   {ae:>5} /min\n"
                                    f"r (window)   {r:5.2f}")
                rec = f" · recording {segment_id}" if multi else ""
                sub = (f"GT: {MODALITY[dataset]} as scored, GT breaths = {NEW_GT[dataset]}; models: member m0, "
                       + ("raw output" if args.raw_predictions else "band-passed as scored")
                       + "; each waveform min-max scaled in the window")
                name = f"{f'rec{segment_id}_' if multi else ''}{t0:06.1f}-{t1:06.1f}s.png".replace(".0-", "-").replace(".0s", "s")
                gt_w = np.asarray(ref["signal"][a:b], float)
                draw_window([(folders[k] / name, pdfs[k]) for k in folders],
                            (f"{head}{rec} · {t0:.0f}-{t1:.0f} s", sub, group_line), t,
                            gt_w, {m: preds[m][a:b] for m in ("O", "X")}, texts)
                count += 1
                if (b - a) / fs < FFT_MIN_SEC - 1e-6:
                    continue
                spectra = {}
                for key, values in (("GT", gt_w), ("O", preds["O"][a:b]), ("X", preds["X"][a:b])):
                    freq, mag = spectrum(values, fs)
                    peak_hz, peak_mag = band_peak(freq, mag, rate_band)
                    spectra[key] = {"freq": freq, "mag": mag, "peak_hz": peak_hz, "peak_mag": peak_mag}
                status = ("not in a scored 60 s minute" if inside.empty else
                          f"minute {inside.iloc[0].Start_Sec:.0f}-{inside.iloc[0].End_Sec:.0f} s, "
                          + ("scored" if inside.iloc[0].Scored else "not scored"))
                peak = {k: (f"{v['peak_hz']:.3f} Hz {v['peak_hz'] * 60:5.1f} /min" if np.isfinite(v["peak_hz"])
                            else "n/a") for k, v in spectra.items()}
                fft_texts = {m: (f"peak in {rate_band[0] * 60:g}-{rate_band[1] * 60:g} /min\n"
                                 f"GT         {peak['GT']}\n"
                                 f"Fusion {m}   {peak[m]}\n"
                                 f"{status}") for m in ("O", "X")}
                fft_sub = (f"|FFT| of the {(b - a) / fs:g} s window (detrend, Hamming, zero padding), each spectrum "
                           f"scaled to its peak = 1 (largest |FFT| in the rate band of the scoring); GT: "
                           f"{MODALITY[dataset]} as scored; models: member m0, "
                           + ("raw output" if args.raw_predictions else "band-passed as scored"))
                draw_fft([(folders[k] / "FFT" / name.replace(".png", "_FFT.png"), fft_pdfs[k]) for k in folders],
                         (f"{head}{rec} · {t0:.0f}-{t1:.0f} s", fft_sub, group_line), spectra, fft_texts, max_freq)
    finally:
        for pdf in list(pdfs.values()) + list(fft_pdfs.values()):
            pdf.close()
    return count


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def same_subject(a: object, b: object) -> bool:
    a, b = str(a).strip(), str(b).strip()
    return a == b or (a.isdigit() and b.isdigit() and int(a) == int(b))


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Stem fusion subjects grouped at 2 and 3 breaths/min (change, improved "
                                            "below the threshold, worsened above it) and their 30 s waveforms")
    p.add_argument("--thresholds", nargs="+", type=float, default=list(THRESHOLDS), help="breaths/min (default 2 3)")
    p.add_argument("--clinical-mae", type=float, default=CLINICAL_MAE, help="clinical criterion: MAE below (default 3.0)")
    p.add_argument("--clinical-pcc", type=float, default=CLINICAL_PCC,
                   help="clinical criterion: mean |PCC| above (default 0.6)")
    p.add_argument("--datasets", nargs="+", choices=list(DATASETS), default=list(DATASETS))
    p.add_argument("--keep-issue-subjects", action="store_true", help="also group the MAE issue subjects (marked)")
    p.add_argument("--ann1-nogap", action="store_true", help="BIDMC: leave out minutes overlapping an ann1 gap")
    p.add_argument("--window-sec", type=float, default=WINDOW_SEC)
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
    p.add_argument("--steam2-ppg-dir", type=Path, default=None)
    p.add_argument("--steam2-rsp-dir", type=Path, default=None)
    return p.parse_args(list(sys.argv[1:] if argv is None else argv))


def main(argv: Optional[Sequence[str]] = None) -> None:
    args = parse_args(argv)
    thresholds = sorted({float(t) for t in args.thresholds})
    out_root = Path(args.output_root)
    out_root.mkdir(parents=True, exist_ok=True)
    subjects_rows, minute_frames, group_rows, missing = [], [], [], []
    for dataset in args.datasets:
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
        found = 0
        for subject in sorted(data):
            issue = any(same_subject(subject, x) for x in EXCLUDED[dataset])
            files = {m: model_files(args, mod, dataset, m, subject) for m in ("O", "X")}
            if not files["O"] or not files["X"]:
                missing.append({"Dataset": dataset, "Subject": subject,
                                "Missing": "stored results of " + " and ".join(LABELS[m] for m in ("O", "X") if not files[m])})
                continue
            minutes = subject_minutes(args, mod, dataset, subject, data[subject], files, ann1_all.get(subject))
            scored = minutes[minutes.Scored]
            if scored.empty:
                missing.append({"Dataset": dataset, "Subject": subject, "Missing": "no scored minute"})
                continue
            found += 1
            row = {"Dataset": dataset, "Subject": subject, "Issue_Subject": issue, "Scored_Minutes": int(len(scored)),
                   "MAE_X": float(scored.AE_X.mean()), "MAE_O": float(scored.AE_O.mean())}
            row["Diff_O_minus_X"] = row["MAE_O"] - row["MAE_X"]
            # |PCC|: absolute PCC of every scored minute, averaged over the subject's scored minutes
            row["AbsPCC_X"], row["AbsPCC_O"] = float(scored.PCC_X.abs().mean()), float(scored.PCC_O.abs().mean())
            for criterion in CLINICAL_CRITERIA:
                for side in ("X", "O"):
                    row[f"Meets_{criterion.upper()}_{side}"] = meets_clinical(criterion, row[f"MAE_{side}"],
                                                                              row[f"AbsPCC_{side}"], args)
            row["Files_O"], row["Files_X"] = str(files["O"]["root"]), str(files["X"]["root"])
            grouped = not (issue and not args.keep_issue_subjects)
            groups = memberships(row["MAE_X"], row["MAE_O"], thresholds) if grouped else []
            clinical = clinical_memberships(row, args) if grouped else []
            row["Clinical_Changes"] = "; ".join(f"{clinical_rule(c, args)} {g}" for c, g in clinical)
            row["Groups"] = "; ".join([f"{t:g} bpm {g}" for t, g in groups]
                                      + [f"clinical {c} {g}" for c, g in clinical])
            subjects_rows.append(row)
            tag = mod.safe_tag(subject)
            diff = row["Diff_O_minus_X"]
            direction = "O_better" if diff < 0 else "O_worse"
            places = [(("threshold", t, g), out_root / f"T{t:g}bpm" / group_folder(g, t)
                       / (f"{dataset}_{tag}_{direction}" if g == "1_change" else f"{dataset}_{tag}"),
                       f"{t:g} bpm {group_text(g, t, diff)}") for t, g in groups]
            places += [(("clinical", c, g), out_root / clinical_dir(args) / clinical_criterion_folder(c, args)
                        / clinical_folder(g) / f"{dataset}_{tag}", clinical_text(c, g, row, args)) for c, g in clinical]
            if not places:
                continue
            n = draw_subject(args, mod, dataset, subject, data[subject], files, minutes, row, places)
            for key, folder, _ in places:
                is_clinical = key[0] == "clinical"
                group_rows.append({"Criterion": f"clinical {clinical_rule(key[1], args)}" if is_clinical
                                   else f"{key[1]:g} bpm",
                                   "Threshold_BPM": math.nan if is_clinical else key[1],
                                   "Group": key[2], "Dataset": dataset, "Subject": subject,
                                   "Issue_Subject": issue, "MAE_X": row["MAE_X"], "MAE_O": row["MAE_O"],
                                   "Diff_O_minus_X": diff, "AbsPCC_X": row["AbsPCC_X"], "AbsPCC_O": row["AbsPCC_O"],
                                   "Meets_Both_X": row["Meets_BOTH_X"], "Meets_Both_O": row["Meets_BOTH_O"],
                                   "Missed_By_X": clinical_failed(row["MAE_X"], row["AbsPCC_X"], args),
                                   "Missed_By_O": clinical_failed(row["MAE_O"], row["AbsPCC_O"], args),
                                   "Direction": "Fusion O better" if diff < 0 else "Fusion O worse",
                                   "Scored_Minutes": row["Scored_Minutes"], "Windows_Drawn": n, "Folder": str(folder)})
            minute_frames.append(minutes)
            print(f"  {subject}: MAE X {row['MAE_X']:.2f}, O {row['MAE_O']:.2f} (O - X {row['Diff_O_minus_X']:+.2f})"
                  f"{' [issue]' if issue else ''} -> {row['Groups']}; {n} windows drawn", flush=True)
        print(f"  {found} subjects with both models")
        del data
    subjects = pd.DataFrame(subjects_rows)
    groups = pd.DataFrame(group_rows)
    summary = []
    datasets_read = list(dict.fromkeys(subjects["Dataset"])) if not subjects.empty else []
    print("\nGroups (MAE = subject MAE of a model)")
    sets = [(f"{t:g} bpm", t, g, {"1_change": f"|MAE O - MAE X| >= {t:g}", "2_improved": f"MAE X > {t:g} and MAE O < {t:g}",
                                  "3_worsened": f"MAE X < {t:g} and MAE O > {t:g}"}[g]) for t in thresholds for g in GROUPS]
    sets += [(f"clinical {clinical_rule(c, args)}", math.nan, g,
              {"better": "X misses and O meets", "worse": "X meets and O misses"}[g])
             for c in CLINICAL_CRITERIA for g in CLINICAL_GROUPS]
    criterion_key = {f"clinical {clinical_rule(c, args)}": c for c in CLINICAL_CRITERIA}
    for criterion, t, g, label in sets:
        sel = groups[(groups["Criterion"] == criterion) & (groups["Group"] == g)] if not groups.empty else groups
        listing = ", ".join(f"{r.Dataset} {r.Subject} (MAE {r.MAE_X:.2f} -> {r.MAE_O:.2f}, |PCC| {r.AbsPCC_X:.2f} -> "
                            f"{r.AbsPCC_O:.2f})" for r in sel.itertuples())
        print(f"  {criterion} {g} [{label}]: {len(sel)} subjects" + (f": {listing}" if len(sel) else ": none"))
        for d in datasets_read:
            s = sel[sel["Dataset"] == d] if not sel.empty else sel
            compared = subjects["Dataset"] == d
            if not args.keep_issue_subjects:
                compared &= ~subjects["Issue_Subject"].astype(bool)
            row = {"Criterion": criterion, "Threshold_BPM": t, "Group": g, "Rule": label, "Dataset": d,
                   "Subjects_in_Group": len(s), "Subjects_Compared": int(compared.sum()),
                   "Fusion_O_Better": int((s["Diff_O_minus_X"] < 0).sum()) if len(s) else 0,
                   "Fusion_O_Worse": int((s["Diff_O_minus_X"] > 0).sum()) if len(s) else 0,
                   "Subjects": ", ".join(f"{r.Subject} (MAE {r.MAE_X:.2f} -> {r.MAE_O:.2f}, |PCC| {r.AbsPCC_X:.2f} -> "
                                         f"{r.AbsPCC_O:.2f})" for r in s.itertuples()) if len(s) else "none"}
            if criterion in criterion_key:
                c = criterion_key[criterion].upper()
                row["Meeting_X"] = int(subjects.loc[compared, f"Meets_{c}_X"].sum())
                row["Meeting_O"] = int(subjects.loc[compared, f"Meets_{c}_O"].sum())
            summary.append(row)
    notes = pd.DataFrame([
        ("Models", "member m0 of every LOSOCV fold; Fusion O = with the 1x1 stem fusion, Fusion X = without"),
        ("GT", "CapnoBase: peak detection on the capnogram (RRest rule); BIDMC: ann1 (expert-label rule"
               + (", minutes with an ann1 gap left out" if args.ann1_nogap else "") + "); STEAM2: stored GT"),
        ("MAE", "per subject, mean |stored rate - GT breath count| over the scored minutes, the same minutes for both"),
        ("Groups", "for every threshold T: 1_change = |MAE O - MAE X| >= T (both directions); 2_improved = MAE X > T "
                   "and MAE O < T; 3_worsened = MAE X < T and MAE O > T; thresholds "
                   + ", ".join(f"{t:g}" for t in thresholds) + " breaths/min"),
        ("Clinical", "three criteria, each on its own: " + "; ".join(clinical_rule(c, args) for c in CLINICAL_CRITERIA)
                     + "; |PCC| = mean over the scored minutes of |Wave_PCC| of every minute; better = Fusion X "
                     "misses and Fusion O meets the criterion, worse = the opposite; a subject is saved in the folder "
                     "of every criterion it changes"),
        ("Subjects", "MAE issue subjects (CapnoBase 0115, BIDMC 27, STEAM2 12_YJJ, 22_LMS, 26_CKK) "
                     + ("kept and marked" if args.keep_issue_subjects else "left out of the groups")),
        ("Figures", f"every {args.window_sec:g} s window of every recording of a grouped subject, saved in every "
                    "folder of its groups; (a) GT vs Fusion O (deep red), (b) GT vs Fusion X (deep blue); model lines "
                    "thicker than the GT and above it; min-max scaled in the window"),
        ("FFT figures", "FFT\\ folder of every subject folder: |FFT| of the same window (detrend, Hamming, zero padding "
                        "to >= 32768 points), (a) GT vs Fusion O, (b) GT vs Fusion X; every spectrum scaled to its peak = 1; "
                        "peak = largest |FFT| in the recording's rate band of the scoring (CapnoBase 3-60, BIDMC / STEAM2 "
                        "6-30 /min); windows shorter than 10 s have none"),
    ], columns=["Item", "Description"])
    path = out_root / "261118_StemFusion_Threshold_Groups.xlsx"
    with pd.ExcelWriter(path, engine="openpyxl") as writer:
        pd.DataFrame(summary).to_excel(writer, sheet_name="Group_Summary", index=False)
        (groups.sort_values(["Criterion", "Group", "Diff_O_minus_X"]).round(4) if not groups.empty
         else pd.DataFrame({"Result": ["none: no subject in any group"]})).to_excel(writer, sheet_name="Groups", index=False)
        (subjects.sort_values("Diff_O_minus_X").round(4) if not subjects.empty else subjects).to_excel(
            writer, sheet_name="Subjects", index=False)
        (pd.concat(minute_frames, ignore_index=True).round(4) if minute_frames else pd.DataFrame()).to_excel(
            writer, sheet_name="Minutes", index=False)
        pd.DataFrame(missing, columns=["Dataset", "Subject", "Missing"]).to_excel(writer, sheet_name="Missing", index=False)
        notes.to_excel(writer, sheet_name="Notes", index=False)
    print(f"\n{groups['Subject'].nunique() if not groups.empty else 0} subjects drawn; figures under {out_root}; "
          f"table: {path}")


if __name__ == "__main__":
    main()
