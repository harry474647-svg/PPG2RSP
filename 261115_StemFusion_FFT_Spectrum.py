# -*- coding: utf-8 -*-
r"""261115_StemFusion_FFT_Spectrum: FFT spectrum of the whole recording for the ground truth (GT) and the
model with (Fusion O) and without (Fusion X) the 1x1 stem fusion, for the chosen subjects (default:
CapnoBase 0332), in the style of 261114_RRWaveNet_FFT_Spectrum (x = frequency in Hz, the peak frequency
of each curve in the legend beside the plot); y = |FFT| as computed, not scaled.

Models (stored outputs, nothing is trained): member m0 of the LOSOCV fold of the subject, as
261115_StemFusion_MAEDiff_Waveforms reads them
  CapnoBase      261007 (D:\PPG2RSP_RRWaveNet_Inspired\261007_RRWaveNet_NoEvent_v1): full / no_stem_fusion
  BIDMC, STEAM2  Fusion O: 261013 (D:\PPG2RSP_RRWaveNet_Inspired\261013_RRWavNet_STEAM_BIDMC_v1), the weight
                 decay selected in the fold; Fusion X: 261028_Stem_ablation_noise_v1 (no_stem_fusion)
  (each model is read from 261028_Stem_ablation_noise_v1 first when it is there)
  Fold_Summaries\{subject}_m0_RR60s_windows.csv    every 60 s minute: the stored rate of the model
  Members\m0\Predictions\{subject}_test_outputs.npz  predictions and targets of every test window (64 Hz)

Signals of a recording
  The windows are put back on the recording's time axis (overlap-add mean). Model: every run of
  contiguous windows band-passed with the recording's respiratory filter (respiratory_filter of the
  source script, as in the scoring). GT: the stored target. The whole recording = every sample covered
  by the target and both models (runs joined in time order when the coverage has gaps; the number of
  runs is listed).
  Spectrum of the whole recording: detrend, Hamming window, zero padding (at least 16384 points);
  y = |FFT| (magnitude of the complex FFT, not divided by anything). Peak: the largest |FFT| in the rate
  band, 0.1-0.5 Hz (6-30 /min).

Beside each panel: the GT mean rate of the recording (GT breaths in the covered samples / their
duration; CapnoBase: peak detection on the capnogram, BIDMC: ann1, STEAM2: the stored GT breaths), the
model's peak, and the FFT MAE of the model over the scored minutes of the recording (the rescored GT of
the stem fusion tables, as 261115_StemFusion_MAEDiff_Waveforms; --ann1-nogap for BIDMC).

Figure: (a) GT vs Fusion O (top, red), (b) GT vs Fusion X (bottom, blue), the same recording; GT black,
rate band dashed.

Outputs (D:\PPG2RSP_RRWaveNet_Inspired\261115_StemFusion_FFT_Spectrum)
  {Dataset}_{subject}\{subject}[_rec{k}]_whole_fft.png   one figure per recording
  {Dataset}_{subject}\{Dataset}_{subject}_whole_fft.pdf   every recording of the subject, one page each
  261115_StemFusion_FFT_Spectrum.xlsx
    Recordings  every recording: duration, runs, GT mean rate, peak frequency and |FFT| at the peak of GT /
                Fusion O / Fusion X, |peak difference| to the GT, MAE of both models over the scored minutes
    Notes

Run: python 261115_StemFusion_FFT_Spectrum.py     (Spyder: runfile(...), no options needed)
Options: --subjects CapnoBase:0332 BIDMC:38 ..., --ann1-nogap, --max-freq-hz (x axis, default 1.2),
--root-261007, --root-261013, --root-noise, --output-root, --script-261007, --script-261013, data paths
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
from scipy.signal import detrend

# ---------------------------------------------------------------------------
# Settings
# ---------------------------------------------------------------------------

NAME = "261115_StemFusion_FFT_Spectrum"
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
NEW_GT = {"CapnoBase": "peak detection on the capnogram", "BIDMC": "ann1", "STEAM2": "stored GT"}
MODALITY = {"CapnoBase": "capnography", "BIDMC": "impedance", "STEAM2": "belt"}
SUBJECTS = ("CapnoBase:0332",)
RATE_BAND_BPM = (6.0, 30.0)  # 0.1-0.5 Hz
FFT_MIN_POINTS = 16384
MAX_FREQ_HZ = 1.2
ANN1_GAP_FACTOR = 2.5
MEMBER = 0
COLORS = {"GT": "#000000", "O": "#ed665a", "X": "#559ed5"}  # Fusion O red, Fusion X blue
LABELS = {"O": "Fusion O (with stem fusion)", "X": "Fusion X (without stem fusion)"}
BAND_COLOR, INK, MUTED, GRID = "#b84fb8", "#1f1f1f", "#6b6a66", "#e4e3df"


# ---------------------------------------------------------------------------
# Source scripts and stored files (as 261115_StemFusion_MAEDiff_Waveforms)
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
        name = "fftspec_src_" + re.sub(r"\W", "_", path.stem)
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
# Ground truth (the rescored GT, as 261115_StemFusion_MAEDiff_Waveforms)
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
        else:
            breaths, label_based = gt_breaths(dataset, mod, ref, ann1)
            valid, reason, count = minute_gt(mod, ref, a, b, breaths, label_based)
            gap = dataset == "BIDMC" and any(lo < b and hi > a for lo, hi in gap_spans(breaths, len(ref["signal"])))
            if gap and args.ann1_nogap and valid:
                valid, reason = False, "ref_ann1_gap"
        rows.append({"Dataset": dataset, "Subject": subject, "Segment_ID": int(segment_id), "Start_Sec": float(t0),
                     "End_Sec": t1, "Scored": valid, "Reason": "" if valid else reason, "GT_Count": count,
                     "Rate_X": float(x["Pred_Wave_FFT_RR"]), "Rate_O": float(o["Pred_Wave_FFT_RR"]),
                     "AE_X": abs(float(x["Pred_Wave_FFT_RR"]) - count) if valid else math.nan,
                     "AE_O": abs(float(o["Pred_Wave_FFT_RR"]) - count) if valid else math.nan})
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Signals and spectra (as 261114_RRWaveNet_FFT_Spectrum)
# ---------------------------------------------------------------------------

def sanitize(signal: np.ndarray) -> np.ndarray:
    arr = np.asarray(signal, dtype=np.float64).reshape(-1)
    if arr.size == 0:
        return arr
    finite = np.isfinite(arr)
    if not finite.any():
        return np.zeros_like(arr)
    if not finite.all():
        arr = arr.copy()
        arr[~finite] = float(np.median(arr[finite]))
    return arr


def series(outputs: Path, segment_id: int, length: int, key: str) -> np.ndarray:
    """Predictions or targets of one recording on its 64 Hz axis (overlap-add mean; NaN where no window)."""
    with np.load(outputs, allow_pickle=False) as data:
        values, seg, starts = data[key], data["segment_ids"].astype(int), data["window_starts"].astype(int)
    pick = seg == segment_id
    out_sum, cover = np.zeros(length), np.zeros(length)
    for p, s in zip(values[pick], starts[pick]):
        e = min(length, s + p.shape[0])
        if e <= s:
            continue
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


def spectrum(x: np.ndarray, fs: float) -> Tuple[np.ndarray, np.ndarray]:
    """Frequency (Hz) and |FFT| as fft_rr_bpm: detrend, Hamming window, zero padding."""
    x = sanitize(x)
    if len(x) < 8 or float(np.std(x)) < 1e-8:
        return np.zeros(1), np.full(1, np.nan)
    windowed = detrend(x) * np.hamming(len(x))
    n_fft = max(FFT_MIN_POINTS, 1 << int(math.ceil(math.log2(len(x)))))
    return np.fft.rfftfreq(n_fft, d=1.0 / float(fs)), np.abs(np.fft.rfft(windowed, n=n_fft))


def band_peak(freq: np.ndarray, mag: np.ndarray) -> Tuple[float, float]:
    """Frequency (Hz) and value of the largest |FFT| in the rate band."""
    keep = (freq * 60.0 >= RATE_BAND_BPM[0]) & (freq * 60.0 <= RATE_BAND_BPM[1]) & np.isfinite(mag)
    if not keep.any():
        return math.nan, math.nan
    k = int(np.flatnonzero(keep)[np.argmax(mag[keep])])
    return float(freq[k]), float(mag[k])


def curve(label: str, color: str, lw: float, z: int, signal: np.ndarray, fs: float) -> dict:
    freq, mag = spectrum(signal, fs)
    peak_hz, peak_mag = band_peak(freq, mag)
    return {"label": label, "color": color, "lw": lw, "z": z, "freq": freq, "y": mag, "peak_hz": peak_hz,
            "peak_mag": peak_mag}


def contiguous_runs(mask: np.ndarray) -> List[Tuple[int, int]]:
    padded = np.concatenate([[False], np.asarray(mask, bool), [False]])
    return [(int(a), int(b)) for a, b in np.flatnonzero(padded[1:] != padded[:-1]).reshape(-1, 2)]


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------

BAND_HZ = (RATE_BAND_BPM[0] / 60.0, RATE_BAND_BPM[1] / 60.0)


def plot_spectra(ax, curves: List[dict], max_freq: float, marker_size: float = 5.5) -> None:
    """|FFT| of the curves as computed (no scaling); the peak in the rate band marked."""
    top = 0.0
    for c in curves:
        keep = c["freq"] <= max_freq
        if np.isfinite(c["y"][keep]).any():
            top = max(top, float(np.nanmax(c["y"][keep])))
            ax.plot(c["freq"][keep], c["y"][keep], color=c["color"], lw=c["lw"], solid_capstyle="round", zorder=c["z"])
        if np.isfinite(c["peak_hz"]):
            ax.axvline(c["peak_hz"], color=c["color"], lw=1.0, alpha=0.9, zorder=c["z"] - 1)
            ax.plot([c["peak_hz"]], [c["peak_mag"]], marker="s", ms=marker_size, color=c["color"], mec="white",
                    mew=0.8, zorder=c["z"] + 1)
    for edge in BAND_HZ:
        ax.axvline(edge, color=BAND_COLOR, lw=1.1, ls=(0, (4, 3)), zorder=1)
    ax.set_xlim(0.0, max_freq)
    ax.set_ylim(0.0, (top if top > 0 else 1.0) * 1.08)
    ax.grid(True, color=GRID, lw=0.6)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)


def peak_text(c: dict) -> str:
    return f"{c['label']}\npeak {c['peak_hz']:.3f} Hz" if np.isfinite(c["peak_hz"]) else f"{c['label']}\npeak n/a"


def draw_recording(path: Path, pdf, header: Tuple[str, str], panels: Dict[str, List[dict]], texts: Dict[str, str],
                   max_freq: float) -> None:
    """(a) GT vs Fusion O, (b) GT vs Fusion X for the whole recording; peaks and rates beside each panel."""
    width, side, left, panel, gap, top_in, bottom_in = 10.6, 3.6, 0.8, 1.85, 0.55, 0.95, 0.6
    height = top_in + bottom_in + 2 * panel + gap
    fig = plt.figure(figsize=(width, height))
    axes = []
    for k, model in enumerate(("O", "X")):
        bottom = bottom_in + (1 - k) * (panel + gap)
        ax = fig.add_axes([left / width, bottom / height, (width - left - side - 0.25) / width, panel / height],
                          sharex=axes[0] if axes else None)
        axes.append(ax)
        plot_spectra(ax, panels[model], max_freq)
        ax.set_ylabel("|FFT|", fontsize=9)
        ax.set_title(f"({'ab'[k]}) GT vs {LABELS[model]}", fontsize=9.5, loc="left", pad=3)
        if k == 0:
            ax.tick_params(labelbottom=False)
        x = (width - side + 0.05) / width
        y = (bottom + panel) / height
        handles = [Line2D([], [], color=c["color"], lw=c["lw"], marker="s", ms=5.5, mec="white", mew=0.8)
                   for c in panels[model]] + [Line2D([], [], color=BAND_COLOR, lw=1.1, ls=(0, (4, 3)))]
        labels = [peak_text(c) for c in panels[model]] + [f"rate band {BAND_HZ[0]:.2f}-{BAND_HZ[1]:.2f} Hz"]
        fig.legend(handles, labels, loc="upper left", bbox_to_anchor=(x, y), frameon=False, fontsize=7.8,
                   labelspacing=0.8, borderaxespad=0.0, handlelength=2.2)
        fig.text(x + 0.004, y - 1.42 / height, texts[model], family="monospace", fontsize=7.4, color=INK, va="top")
    axes[-1].set_xlabel("Frequency (Hz)")
    fig.text(left / width, 1.0 - 0.18 / height, header[0], fontsize=10.5, color=INK, va="top")
    fig.text(left / width, 1.0 - 0.48 / height, header[1], fontsize=7.4, color=MUTED, va="top")
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=200)
    pdf.savefig(fig)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def same_subject(a: object, b: object) -> bool:
    a, b = str(a).strip(), str(b).strip()
    return a == b or (a.isdigit() and b.isdigit() and int(a) == int(b))


def parse_subjects(items: Sequence[str]) -> Dict[str, List[str]]:
    chosen: Dict[str, List[str]] = {}
    for item in items:
        dataset, _, subject = str(item).partition(":")
        if dataset not in DATASETS or not subject:
            raise SystemExit(f"--subjects {item}: write Dataset:Subject with Dataset in {', '.join(DATASETS)}")
        chosen.setdefault(dataset, []).append(subject)
    return chosen


def draw_subject(args, mod, dataset: str, subject: str, record: dict, files: Dict[str, dict], minutes: pd.DataFrame,
                 ann1: Optional[np.ndarray], out_root: Path) -> List[dict]:
    """The whole-recording spectra of every recording of a subject."""
    fs = mod.TARGET_FS
    folder = out_root / f"{dataset}_{mod.safe_tag(subject)}"
    folder.mkdir(parents=True, exist_ok=True)
    multi = len(record["references"]) > 1
    sub = (f"GT: {MODALITY[dataset]} target; models: m0, band-passed as scored; FFT of the whole recording "
           "(detrend, Hamming); y = |FFT|, not scaled; peak in 0.1-0.5 Hz")
    rows = []
    with PdfPages(folder / f"{dataset}_{mod.safe_tag(subject)}_whole_fft.pdf") as pdf:
        for segment_id, ref in sorted(record["references"].items()):
            length = len(ref["signal"])
            preds = {m: as_scored(mod, series(files[m]["outputs"], int(segment_id), length, "predictions"), ref)
                     for m in ("O", "X")}
            target = series(files["O"]["outputs"], int(segment_id), length, "targets")
            covered = np.isfinite(target) & np.isfinite(preds["O"]) & np.isfinite(preds["X"])
            if covered.sum() < 8 * fs:
                continue
            runs = contiguous_runs(covered)
            duration = float(covered.sum()) / fs
            breaths, _ = gt_breaths(dataset, mod, ref, ann1)
            breaths = breaths[(breaths >= 0) & (breaths < length)]
            gt_rate = float(covered[breaths].sum()) * 60.0 / duration
            gt = curve(f"GT ({MODALITY[dataset]})", COLORS["GT"], 1.6, 4, target[covered], fs)
            panels = {m: [gt, curve(LABELS[m], COLORS[m], 1.9, 5, preds[m][covered], fs)] for m in ("O", "X")}
            mins = minutes[(minutes.Segment_ID == int(segment_id)) & minutes.Scored]
            mae = {m: float(mins[f"AE_{m}"].mean()) if len(mins) else math.nan for m in ("O", "X")}
            first, last = runs[0][0] / fs, runs[-1][1] / fs
            span = f"{first:.0f}-{last:.0f} s, {duration:.0f} s covered" + (f", {len(runs)} runs joined" if len(runs) > 1 else "")
            texts = {}
            for m in ("O", "X"):
                peak = panels[m][1]["peak_hz"]
                texts[m] = (f"whole recording {span}\n"
                            f"GT mean rate {gt_rate:5.1f} /min ({gt_rate / 60:.3f} Hz)\n"
                            f"model peak   {peak * 60:5.1f} /min ({peak:.3f} Hz)\n"
                            f"MAE, {len(mins)} scored min " + (f"{mae[m]:.2f} /min" if np.isfinite(mae[m]) else "-"))
            rec = f" · recording {segment_id}" if multi else ""
            head = (f"{dataset} {subject}{rec}: FFT of the whole recording · MAE X {mae['X']:.2f}, "
                    f"MAE O {mae['O']:.2f} breaths/min (scored minutes)")
            stem = f"{mod.safe_tag(subject)}{f'_rec{segment_id}' if multi else ''}_whole_fft"
            draw_recording(folder / f"{stem}.png", pdf, (head, sub), panels, texts, args.max_freq_hz)
            row = {"Dataset": dataset, "Subject": subject, "Segment_ID": int(segment_id), "First_Sec": first,
                   "Last_Sec": last, "Covered_Sec": duration, "Runs": len(runs), "GT_Breaths": int(covered[breaths].sum()),
                   "GT_Mean_Rate_Hz": gt_rate / 60.0, "GT_Peak_Hz": gt["peak_hz"], "GT_Peak_FFT": gt["peak_mag"]}
            for m in ("O", "X"):
                est = panels[m][1]
                row[f"Peak_Hz_{m}"] = est["peak_hz"]
                row[f"Peak_FFT_{m}"] = est["peak_mag"]
                row[f"Abs_Peak_Diff_Hz_{m}"] = abs(est["peak_hz"] - gt["peak_hz"])
                row[f"MAE_{m}"] = mae[m]
            row["Scored_Minutes"] = int(len(mins))
            row["Figure"] = str((folder / f"{stem}.png").relative_to(out_root))
            rows.append(row)
    return rows


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="FFT of the whole recording: GT, Fusion O and Fusion X")
    p.add_argument("--subjects", nargs="+", default=list(SUBJECTS), help="Dataset:Subject (default CapnoBase:0332)")
    p.add_argument("--ann1-nogap", action="store_true", help="BIDMC: leave out minutes overlapping an ann1 gap")
    p.add_argument("--max-freq-hz", type=float, default=MAX_FREQ_HZ)
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
    chosen = parse_subjects(args.subjects)
    out_root = Path(args.output_root)
    out_root.mkdir(parents=True, exist_ok=True)
    recording_rows = []
    for dataset in [d for d in DATASETS if d in chosen]:
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
        for wanted in chosen[dataset]:
            subject = next((s for s in sorted(data) if same_subject(s, wanted)), None)
            if subject is None:
                print(f"  {wanted}: not in the {dataset} data")
                continue
            files = {m: model_files(args, mod, dataset, m, subject) for m in ("O", "X")}
            if not files["O"] or not files["X"]:
                print(f"  {subject}: stored outputs of {'Fusion O' if not files['O'] else 'Fusion X'} (m0) not found")
                continue
            minutes = subject_minutes(args, mod, dataset, subject, data[subject], files, ann1_all.get(subject))
            rows = draw_subject(args, mod, dataset, subject, data[subject], files, minutes, ann1_all.get(subject),
                                out_root)
            recording_rows += rows
            for r in rows:
                print(f"  {subject} recording {r['Segment_ID']}: {r['Covered_Sec']:.0f} s; GT mean rate "
                      f"{r['GT_Mean_Rate_Hz']:.3f} Hz, peak GT {r['GT_Peak_Hz']:.3f} Hz, O {r['Peak_Hz_O']:.3f} Hz, "
                      f"X {r['Peak_Hz_X']:.3f} Hz; MAE X {r['MAE_X']:.2f}, O {r['MAE_O']:.2f}", flush=True)
            if not rows:
                print(f"  {subject}: no recording with stored outputs of both models")
        del data
    notes = pd.DataFrame([
        ("Models", "member m0 of the LOSOCV fold; Fusion O = with the 1x1 stem fusion, Fusion X = without"),
        ("Signals", "the whole recording: every sample covered by the stored target and both models (runs joined in "
                    "time order when there are gaps); model waveform band-passed as scored; GT = stored target"),
        ("Spectrum", "detrend, Hamming window, zero padding (>= 16384 points); |FFT| as computed, not scaled"),
        ("Peak", "largest |FFT| in 0.1-0.5 Hz (6-30 breaths/min); Abs_Peak_Diff = |model peak - GT peak|"),
        ("GT mean rate", "GT breaths in the covered samples / covered duration (CapnoBase peak detection on the "
                         "capnogram, BIDMC ann1, STEAM2 stored breaths)"),
        ("MAE", "mean |stored rate - GT breath count| over the scored minutes of the recording (rescored GT)"),
    ], columns=["Item", "Description"])
    path = out_root / f"{NAME}.xlsx"
    with pd.ExcelWriter(path, engine="openpyxl") as writer:
        pd.DataFrame(recording_rows).round(5).to_excel(writer, sheet_name="Recordings", index=False)
        notes.to_excel(writer, sheet_name="Notes", index=False)
    print(f"\nFigures under {out_root}; table: {path}")


if __name__ == "__main__":
    main()
