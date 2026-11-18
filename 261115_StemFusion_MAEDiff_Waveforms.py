# -*- coding: utf-8 -*-
r"""261115_StemFusion_MAEDiff_Waveforms: the subjects whose RR error changes by at least 2
breaths/min with the 1x1 stem fusion, and their waveforms in every 30 s window.

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

Subjects drawn (--subjects): by default the seven subjects discussed for the stem fusion,
  CapnoBase 0332, 0312, 0018, 0035, 0142 and BIDMC 38, 30, whatever their MAE difference;
  "--subjects detected" draws, as before, every subject with |MAE O - MAE X| >= 2 breaths/min
  (--threshold), both directions (Fusion O better: O < X; Fusion O worse: O > X); both can be given
  ("--subjects detected CapnoBase:0332"). Per subject, FFT MAE = mean |stored rate - GT breath count|
  over the scored minutes, for each model on the same minutes. The subjects with an MAE issue
  (CapnoBase 0115, BIDMC 27, STEAM2 12_YJJ, 22_LMS, 26_CKK) are marked (--skip-issue-subjects leaves
  them out of the threshold detection).

Figures: every 30 s window of every recording of a drawn subject (the windows of a minute
are its first and second half; stretches outside the scored minute grid in 30 s steps):
  (a) GT vs Fusion O (top, deep red), (b) GT vs Fusion X (bottom, deep blue; both drawn thicker than the GT
  and above it), the same seconds, no model
  shifted in time; every waveform min-max scaled in the window; GT = the reference signal the GT
  breaths were found on
  (CapnoBase capnogram, BIDMC impedance, STEAM2 belt; band-passed and aligned as in the scoring),
  its GT breaths marked; beside each panel the minute's GT rate, the model's rate and error and
  the Pearson r in the 30 s window. The model waveforms are drawn as the rate was scored: every run of
contiguous windows band-passed with the recording's respiratory filter (--raw-predictions: the raw output).

Outputs (D:\PPG2RSP_RRWaveNet_Inspired\261115_StemFusion_MAEDiff_Waveforms)
  {Dataset}_{subject}_{O_better|O_worse}\{recording}_{t0}-{t1}s.png   every 30 s window
  {Dataset}_{subject}_{O_better|O_worse}\{Dataset}_{subject}_windows.pdf   all windows, one page each
  261115_StemFusion_MAEDiff.xlsx
    Drawn      the drawn subjects (MAE, difference, minutes, why drawn, folders)
    Subjects   every subject of the datasets read: MAE X, MAE O, difference, scored minutes
    Minutes    every minute of the drawn subjects: GT, rates and errors of both models
    Missing    listed subjects without stored results of both models or without a scored minute
    Notes

Run: python 261115_StemFusion_MAEDiff_Waveforms.py     (Spyder: runfile(...), no options needed)
Options: --subjects, --threshold, --datasets, --skip-issue-subjects, --ann1-nogap, --window-sec, --root-261007,
--root-261013, --root-noise, --output-root, --script-261007, --script-261013, data paths
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

# ---------------------------------------------------------------------------
# Settings
# ---------------------------------------------------------------------------

NAME = "261115_StemFusion_MAEDiff_Waveforms"
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
THRESHOLD_BPM = 2.0
# The subjects discussed for the stem fusion (MAE X -> O with the original GT: 0332 9.64 -> 2.31,
# 0312 6.06 -> 2.57, 0018 3.62 -> 0.46, BIDMC 38 3.33 -> 1.05, 0035 5.15 -> 2.88, 0142 0.30 -> 2.28,
# BIDMC 30 1.54 -> 3.54); drawn by default whatever their difference with the GT used here
SUBJECTS = ("CapnoBase:0332", "CapnoBase:0312", "CapnoBase:0018", "BIDMC:38", "CapnoBase:0035",
            "CapnoBase:0142", "BIDMC:30")
WINDOW_SEC = 30.0
ANN1_GAP_FACTOR = 2.5
MEMBER = 0
COLORS = {"GT": "#111111", "O": "#d7191c", "X": "#0b5fc4"}  # top panel (Fusion O) deep red, bottom (Fusion X) deep blue
MODEL_LW = 2.2  # model lines drawn thicker than the GT and above it
LABELS = {"X": "Fusion X (without stem fusion)", "O": "Fusion O (with stem fusion)"}
INK, MUTED, GRID = "#1f1f1f", "#6b6a66", "#e4e3df"


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
                     "AE_O": abs(float(o["Pred_Wave_FFT_RR"]) - count) if valid else math.nan})
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


def draw_window(path: Path, pdf, header: str, t: np.ndarray, gt: np.ndarray, breaths_sec: np.ndarray,
                preds: Dict[str, np.ndarray], texts: Dict[str, str]) -> None:
    width, side, left, panel, gap, top_in, bottom_in = 12.0, 3.4, 0.7, 1.75, 0.45, 0.95, 0.6
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
        for b in breaths_sec:
            ax.axvline(b, ymin=0.94, ymax=1.0, color=COLORS["GT"], lw=1.2)
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
        fig.legend([Line2D([], [], color=COLORS["GT"], lw=1.4), Line2D([], [], color=COLORS[model], lw=MODEL_LW),
                    Line2D([], [], color=COLORS["GT"], lw=1.2, marker="|", ls="none", ms=8)],
                   ["GT", LABELS[model], "GT breaths"], loc="upper left", bbox_to_anchor=(x, y), frameon=False,
                   fontsize=7.8, borderaxespad=0.0)
        fig.text(x + 0.004, y - 0.95 / height, texts[model], family="monospace", fontsize=7.6, color=INK, va="top")
    axes[-1].set_xlabel("Time in the recording (s)")
    fig.text(left / width, 1.0 - 0.18 / height, header[0], fontsize=10.5, color=INK, va="top")
    fig.text(left / width, 1.0 - 0.48 / height, header[1], fontsize=7.6, color=MUTED, va="top")
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=160)
    pdf.savefig(fig)
    plt.close(fig)


def draw_subject(args, mod, dataset: str, subject: str, record: dict, files: Dict[str, dict], minutes: pd.DataFrame,
                 summary: dict, ann1: Optional[np.ndarray], out_root: Path) -> Tuple[str, int]:
    fs = mod.TARGET_FS
    direction = "O_better" if summary["Diff_O_minus_X"] < 0 else "O_worse"
    folder = out_root / f"{dataset}_{mod.safe_tag(subject)}_{direction}"
    folder.mkdir(parents=True, exist_ok=True)
    multi = len(record["references"]) > 1
    count = 0
    issue = " · MAE issue subject" if summary["Issue_Subject"] else ""
    head = (f"{dataset} {subject}: MAE X {summary['MAE_X']:.2f}, MAE O {summary['MAE_O']:.2f} breaths/min "
            f"(O - X {summary['Diff_O_minus_X']:+.2f}; {'Fusion O better' if direction == 'O_better' else 'Fusion O worse'})"
            f"{issue}")
    with PdfPages(folder / f"{dataset}_{mod.safe_tag(subject)}_windows.pdf") as pdf:
        for segment_id, ref in sorted(record["references"].items()):
            length = len(ref["signal"])
            preds = {m: series(files[m]["outputs"], int(segment_id), length) for m in ("O", "X")}
            if not args.raw_predictions:
                preds = {m: as_scored(mod, v, ref) for m, v in preds.items()}
            covered = np.isfinite(preds["O"]) & np.isfinite(preds["X"])
            if not covered.any():
                continue
            mins = minutes[minutes["Segment_ID"] == int(segment_id)].sort_values("Start_Sec")
            breaths, _ = gt_breaths(dataset, mod, ref, ann1)
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
                bs = breaths[(breaths >= a) & (breaths < b)] / fs
                draw_window(folder / name, pdf, (f"{head}{rec} · {t0:.0f}-{t1:.0f} s", sub), t,
                            np.asarray(ref["signal"][a:b], float), bs, {m: preds[m][a:b] for m in ("O", "X")}, texts)
                count += 1
    return str(folder), count


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def same_subject(a: object, b: object) -> bool:
    a, b = str(a).strip(), str(b).strip()
    return a == b or (a.isdigit() and b.isdigit() and int(a) == int(b))


def parse_subjects(items: Sequence[str]) -> Tuple[Dict[str, List[str]], bool]:
    """Dataset:Subject items, and whether "detected" (the threshold detection) was asked for."""
    listed: Dict[str, List[str]] = {}
    detect = False
    for item in items:
        if item.strip().lower() == "detected":
            detect = True
            continue
        if ":" not in item:
            raise ValueError(f"--subjects takes Dataset:Subject or detected, got {item}")
        dataset, subject = item.split(":", 1)
        match = [d for d in DATASETS if d.lower() == dataset.strip().lower()]
        if not match:
            raise ValueError(f"unknown dataset in {item} ({', '.join(DATASETS)})")
        listed.setdefault(match[0], []).append(subject.strip())
    return listed, detect


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="GT vs Fusion O / Fusion X waveforms in every 30 s window of the stem "
                                            "fusion subjects")
    p.add_argument("--subjects", nargs="+", default=list(SUBJECTS),
                   help="Dataset:Subject items and/or 'detected' (|MAE O - MAE X| >= threshold); default: the seven "
                        "discussed subjects")
    p.add_argument("--threshold", type=float, default=THRESHOLD_BPM, help="|MAE O - MAE X| (breaths/min)")
    p.add_argument("--datasets", nargs="+", choices=list(DATASETS), default=list(DATASETS))
    p.add_argument("--skip-issue-subjects", action="store_true")
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
    out_root = Path(args.output_root)
    out_root.mkdir(parents=True, exist_ok=True)
    listed, detect = parse_subjects(args.subjects)
    subjects_rows, minute_frames, detected, missing = [], [], [], []
    for dataset in [d for d in args.datasets if detect or d in listed]:
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
        wanted = listed.get(dataset, [])
        for name in wanted:
            if not any(same_subject(name, s) for s in data):
                missing.append({"Dataset": dataset, "Subject": name, "Missing": "not in the data read"})
        for subject in sorted(data):
            is_listed = any(same_subject(subject, s) for s in wanted)
            files = {m: model_files(args, mod, dataset, m, subject) for m in ("O", "X")}
            if not files["O"] or not files["X"]:
                if is_listed:
                    missing.append({"Dataset": dataset, "Subject": subject,
                                    "Missing": "stored results of " + " and ".join(
                                        LABELS[m] for m in ("O", "X") if not files[m])})
                continue
            issue = any(same_subject(subject, x) for x in EXCLUDED[dataset])
            minutes = subject_minutes(args, mod, dataset, subject, data[subject], files, ann1_all.get(subject))
            scored = minutes[minutes.Scored]
            if scored.empty:
                if is_listed:
                    missing.append({"Dataset": dataset, "Subject": subject, "Missing": "no scored minute"})
                continue
            found += 1
            row = {"Dataset": dataset, "Subject": subject, "Issue_Subject": issue, "Scored_Minutes": int(len(scored)),
                   "MAE_X": float(scored.AE_X.mean()), "MAE_O": float(scored.AE_O.mean())}
            row["Diff_O_minus_X"] = row["MAE_O"] - row["MAE_X"]
            row["Files_O"], row["Files_X"] = str(files["O"]["root"]), str(files["X"]["root"])
            subjects_rows.append(row)
            passes = abs(row["Diff_O_minus_X"]) >= args.threshold and not (issue and args.skip_issue_subjects)
            if is_listed or (detect and passes):
                folder, n = draw_subject(args, mod, dataset, subject, data[subject], files, minutes, row,
                                         ann1_all.get(subject), out_root)
                detected.append({**row, "Direction": "Fusion O better" if row["Diff_O_minus_X"] < 0 else "Fusion O worse",
                                 "Drawn_Because": "listed" if is_listed else f"|O - X| >= {args.threshold:g}",
                                 "Passes_Threshold": bool(abs(row["Diff_O_minus_X"]) >= args.threshold),
                                 "Windows_Drawn": n, "Folder": folder})
                minute_frames.append(minutes)
                print(f"  {subject}: MAE X {row['MAE_X']:.2f}, O {row['MAE_O']:.2f} (O - X {row['Diff_O_minus_X']:+.2f})"
                      f"{' [issue]' if issue else ''} -> {n} windows drawn", flush=True)
        print(f"  {found} subjects with both models")
        del data
    subjects = pd.DataFrame(subjects_rows)
    det = pd.DataFrame(detected)
    if not det.empty:
        det = det.sort_values(["Direction", "Diff_O_minus_X"])
    notes = pd.DataFrame([
        ("Models", "member m0 of every LOSOCV fold; Fusion O = with the 1x1 stem fusion, Fusion X = without"),
        ("GT", "CapnoBase: peak detection on the capnogram (RRest rule); BIDMC: ann1 (expert-label rule"
               + (", minutes with an ann1 gap left out" if args.ann1_nogap else "") + "); STEAM2: stored GT"),
        ("MAE", "per subject, mean |stored rate - GT breath count| over the scored minutes, the same minutes for both"),
        ("Subjects drawn", "listed: " + (", ".join(f"{d} {s}" for d, v in listed.items() for s in v) or "none")
                           + ("; and every subject with |MAE O - MAE X| >= " + f"{args.threshold:g} breaths/min"
                              + ("; MAE issue subjects left out" if args.skip_issue_subjects else "")
                              if detect else "")),
        ("Figures", f"every {args.window_sec:g} s window of every recording of a drawn subject; (a) GT vs Fusion O "
                    "(deep red), (b) GT vs Fusion X (deep blue); model lines thicker than the GT and above it; "
                    "min-max scaled in the window; GT breaths marked at the top"),
    ], columns=["Item", "Description"])
    path = out_root / "261115_StemFusion_MAEDiff.xlsx"
    with pd.ExcelWriter(path, engine="openpyxl") as writer:
        (det.round(4) if not det.empty else det).to_excel(writer, sheet_name="Drawn", index=False)
        (subjects.sort_values("Diff_O_minus_X").round(4) if not subjects.empty else subjects).to_excel(
            writer, sheet_name="Subjects", index=False)
        (pd.concat(minute_frames, ignore_index=True).round(4) if minute_frames else pd.DataFrame()).to_excel(
            writer, sheet_name="Minutes", index=False)
        pd.DataFrame(missing, columns=["Dataset", "Subject", "Missing"]).to_excel(writer, sheet_name="Missing", index=False)
        notes.to_excel(writer, sheet_name="Notes", index=False)
    for m in missing:
        print(f"  not drawn: {m['Dataset']} {m['Subject']} ({m['Missing']})")
    print(f"\n{len(det)} subjects drawn; figures under {out_root}; table: {path}")


if __name__ == "__main__":
    main()
