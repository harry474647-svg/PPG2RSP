# -*- coding: utf-8 -*-
r"""261119_BIDMC_Annotation_Disagreement_Waveforms: BIDMC subjects whose two expert annotators disagree
(ann1 vs ann2) - the raw impedance, the preprocessed reference and the reconstructed waveform, with the
subject's MAE and PCC.

Subjects (--subjects, default 27 34 46): BIDMC subjects with a minute in which the breathing rates of
annotator 1 and annotator 2 differ by about 5 breaths/min or more (46: 9.7 vs 16.3, 27: 8.3 vs 14.3,
34: 14.7 vs 20.0 /min). BIDMC 27 is the MAE issue subject left out of the final results; it is drawn and
marked.

Model (stored outputs, nothing is trained): the final model, 261013 with the 1x1 stem fusion (Fusion O),
member m0 of the weight decay selected in the fold (WD_Selection)
  D:\PPG2RSP_RRWaveNet_Inspired\261013_RRWavNet_STEAM_BIDMC_v1\{variant}\01_w10s_nonoverlap\BIDMC\
      Fold_Summaries\{subject}_m0_RR60s_windows.csv       every 60 s minute: stored rate and Wave_PCC
      Members\m0\Predictions\{subject}_test_outputs.npz    the predicted waveform (64 Hz)
  (261028_Stem_ablation_noise_v1\full\... when 261013 has no stored result of the subject)

Signals
  raw           impedance of bidmc_data.mat as recorded (ref.resp_sig.imp, 125 Hz, no filter)
  preprocessed  the reference of the preprocessing of 261013 (loaded as a module): impedance resampled
                to 64 Hz and band-passed 0.1-0.5 Hz, the training target and the GT waveform of the scoring
  reconstructed the predicted waveform of the model, band-passed as the rate was scored (every run of
                contiguous windows with the respiratory filter of 261013; --raw-predictions: raw output)
  ann1 / ann2   breaths of annotator 1 / annotator 2 (ref.breaths.ann1 / ann2)

Scores (as in the final results)
  MAE (ann1)    mean |stored rate - ann1 breath count| over the scored minutes (expert-label rule of
                261013: usable reference, >= 2 breaths, rate in 6-30 /min), the MAE of the final results
  MAE (ann2)    the same minutes scored with the ann2 breath count
  |PCC|         mean |Wave_PCC| of the stored minute tables over the scored minutes
  Disagreement  a minute in which the ann1 and ann2 rates (60 / mean breath interval) differ by more
                than --disagree-bpm (default 2 breaths/min)

Figures (every subject in its folder)
  {subject}_overview.png       the whole recording: (a) raw, (b) preprocessed, (c) reconstructed over the
                               preprocessed, (d) |FFT| of the three; ann1 / ann2 breaths marked in (a) and (b);
                               disagreement minutes shaded
  {subject}_{t0}-{t1}s.png     every stored 60 s minute, the same four panels; beside each panel the
                               annotator rates, the reference FFT rate, the model rate, its error against
                               ann1 and ann2, the PCC and the FFT peaks
  {subject}_all.pdf            all of them, one page each
  (a) is in raw units; (b) and (c) are min-max scaled in the window.
  (d) |FFT| of the raw, preprocessed and reconstructed waveform of the same seconds (detrend, Hamming window,
      zero padding to >= 32768 points, as 261116), each scaled to its peak = 1; the peak (largest |FFT| in the
      rate band of the scoring, 6-30 /min) marked by a vertical line and a square; the ann1 and ann2 rates
      (60 / mean breath interval) as dashed lines.

Outputs (D:\PPG2RSP_RRWaveNet_Inspired\261119_BIDMC_Annotation_Disagreement_Waveforms)
  BIDMC_{subject}\...                                 the figures
  261119_BIDMC_Annotation_Disagreement.xlsx
    Subjects  MAE (ann1), MAE (ann2), |PCC|, scored minutes, disagreement minutes
    Minutes   every minute: ann1 / ann2 counts and rates, reference FFT rate, model rate, errors, PCC
    Missing   subjects without stored results
    Notes

Run: python 261119_BIDMC_Annotation_Disagreement_Waveforms.py     (Spyder: runfile(...), no options needed)
Options: --subjects, --disagree-bpm, --raw-predictions, --root-261013, --root-noise, --output-root,
--script-261013, --bidmc-path
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

NAME = "261119_BIDMC_Annotation_Disagreement_Waveforms"
HERE = Path(__file__).resolve().parent
WORKSPACE = Path(r"D:\PPG2RSP_RRWaveNet_Inspired")
OUTPUT_ROOT = WORKSPACE / NAME
ROOT_261013 = WORKSPACE / "261013_RRWavNet_STEAM_BIDMC_v1"
ROOT_NOISE = WORKSPACE / "261028_Stem_ablation_noise_v1"
SCRIPT_261013 = "261013_RRWaveNet_Adult_SubjectVal_WD.py"
WINDOW = "01_w10s_nonoverlap"
DATASET = "BIDMC"
SUBJECTS = ("27", "34", "46")
ISSUE_SUBJECTS = ("27",)  # MAE issue subject left out of the final results
DISAGREE_BPM = 2.0
MEMBER = 0
FFT_MIN_POINTS = 32768  # zero padding of the spectrum (at least)
MAX_FREQ_HZ = 1.0       # frequency axis of the |FFT| panel
COLORS = {"raw": "#111111", "raw_fft": "#9a9893", "GT": "#111111", "model": "#d7191c", "ann1": "#e08214",
          "ann2": "#5e3c99"}
SHADE = "#fbe7a1"
MODEL_LW = 2.0
INK, MUTED, GRID = "#1f1f1f", "#6b6a66", "#e4e3df"

_MODULES: Dict[str, object] = {}


# ---------------------------------------------------------------------------
# Script and stored results
# ---------------------------------------------------------------------------

def find_script(given: Optional[Path], roots: Sequence[Path]) -> Path:
    """261013: given, written in a run manifest, or found nearby."""
    candidates: List[Path] = [Path(given)] if given else []
    for root in roots:
        for manifest in sorted(Path(root).glob("run_manifest*.json")):
            try:
                script = json.loads(manifest.read_text(encoding="utf-8")).get("script")
            except (OSError, ValueError, AttributeError):
                continue
            if script and Path(script).name == SCRIPT_261013:
                candidates.append(Path(script))
    folders = [HERE, HERE.parent, WORKSPACE] + [Path(r) for r in roots]
    if WORKSPACE.is_dir():
        folders += sorted(p for p in WORKSPACE.iterdir() if p.is_dir())
    candidates += [f / SCRIPT_261013 for f in folders]
    for path in candidates:
        if path.is_file():
            return path.resolve()
    raise FileNotFoundError(f"{SCRIPT_261013} not found (run manifests of {', '.join(map(str, roots))}, {HERE}, "
                            f"{WORKSPACE}); give it with --script-261013")


def load_script(path: Path):
    key = str(path)
    if key not in _MODULES:
        name = "annodis_src_" + re.sub(r"\W", "_", path.stem)
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


def model_files(args: argparse.Namespace, mod, subject: str) -> Optional[Dict[str, Path]]:
    """Minute table and predictions of member m0 of the final model: 261013 (weight decay selected in the
    fold), then 261028_Stem_ablation_noise_v1 (full)."""
    tag = mod.safe_tag(subject)
    roots = []
    record = read_json(Path(args.root_261013) / "WD_Selection" / WINDOW / DATASET / f"{tag}_selection.json") or {}
    if record.get("Selected_Variant"):
        roots.append(Path(args.root_261013) / record["Selected_Variant"] / WINDOW / DATASET)
    roots.append(Path(args.root_noise) / "full" / WINDOW / DATASET)
    for root in roots:
        minutes = root / "Fold_Summaries" / f"{tag}_m{MEMBER}_RR60s_windows.csv"
        outputs = root / "Members" / f"m{MEMBER}" / "Predictions" / f"{tag}_test_outputs.npz"
        if minutes.exists() and outputs.exists():
            return {"minutes": minutes, "outputs": outputs, "root": root}
    return None


# ---------------------------------------------------------------------------
# Annotations and raw signal
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


def lag_sec(reference: dict) -> float:
    return float(reference["diagnostics"].get("Target_Lag_Sec", 0.0))


def to_timeline(mod, labels: np.ndarray, fs_in: float, reference: dict) -> np.ndarray:
    """Breaths (MATLAB sample numbers of the raw recording) as sample indices of the 64 Hz reference."""
    lag = int(round(lag_sec(reference) * mod.TARGET_FS))
    idx = np.round((np.asarray(labels, float) - 1.0) * (mod.TARGET_FS / fs_in)).astype(np.int64) - lag
    return np.unique(idx[(idx >= 0) & (idx < len(reference["signal"]))])


def minute_valid(mod, ref: dict, start: int, end: int, breaths: np.ndarray) -> Tuple[bool, str]:
    """Expert-label rule of 261013 for one minute (as the final results score BIDMC with ann1)."""
    fs = mod.TARGET_FS
    rr_band = ref["profile"]["rr_band_bpm"]
    labels = breaths[(breaths >= start) & (breaths < end)]
    label_rr = float(60.0 * fs / np.mean(np.diff(labels))) if len(labels) >= 2 else math.nan
    if float(np.mean(ref["structural_valid"][start:end])) < mod.MIN_USABLE_PROPORTION:
        return False, "ref_structural_invalid"
    if ref.get("artifact_mask") is not None and float(np.mean(ref["artifact_mask"][start:end])) > 0.0:
        return False, "ref_artifact_label"
    if not np.isfinite(label_rr):
        return False, "ref_rr_undefined"
    if not (rr_band[0] <= label_rr <= rr_band[1]):
        return False, "ref_out_of_band"
    return True, ""


def label_rate(breaths: np.ndarray, start: int, end: int, fs: float) -> Tuple[int, float]:
    """Breath count and rate (60 / mean breath interval) of one annotator in one minute."""
    inside = breaths[(breaths >= start) & (breaths < end)]
    rate = float(60.0 * fs / np.mean(np.diff(inside))) if len(inside) >= 2 else math.nan
    return int(len(inside)), rate


# ---------------------------------------------------------------------------
# Signals
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
    recording's respiratory filter (respiratory_filter of 261013)."""
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
# Minutes
# ---------------------------------------------------------------------------

def subject_minutes(args, mod, subject: str, record: dict, raw: dict, files: Dict[str, Path]) -> pd.DataFrame:
    """Every stored minute: validity (ann1 rule), both annotators, reference FFT rate, model rate, errors, PCC."""
    fs = mod.TARGET_FS
    table = pd.read_csv(files["minutes"], dtype={"Subject": str}, encoding="utf-8-sig")
    rows = []
    for m in table.itertuples(index=False):
        ref = record["references"].get(int(m.Segment_ID))
        if ref is None:
            continue
        t0, t1 = float(m.RR_Window_Start_Sec), float(m.RR_Window_End_Sec)
        a, b = int(round(t0 * fs)), int(round(t1 * fs))
        ann = {k: to_timeline(mod, raw[k], raw["fs"], ref) for k in ("ann1", "ann2")}
        valid, reason = minute_valid(mod, ref, a, b, ann["ann1"])
        n1, r1 = label_rate(ann["ann1"], a, b, fs)
        n2, r2 = label_rate(ann["ann2"], a, b, fs)
        rate = float(m.Pred_Wave_FFT_RR)
        stored_fft = float(getattr(m, "Ref_RR_FFT", math.nan))
        ref_fft = stored_fft if np.isfinite(stored_fft) else fft_rate(
            np.asarray(ref["signal"][a:b], float), fs, tuple(ref["profile"]["rr_band_bpm"]))
        disagree = bool(np.isfinite(r1) and np.isfinite(r2) and abs(r1 - r2) > args.disagree_bpm)
        rows.append({"Dataset": DATASET, "Subject": subject, "Segment_ID": int(m.Segment_ID), "Start_Sec": t0,
                     "End_Sec": t1, "Scored": valid, "Reason": reason,
                     "Ann1_Count": n1, "Ann2_Count": n2, "Ann1_Rate": r1, "Ann2_Rate": r2,
                     "Ann1_minus_Ann2_Rate": r1 - r2, "Disagreement": disagree,
                     "Ref_FFT_Rate": ref_fft, "Model_Rate": rate,
                     "AE_Ann1": abs(rate - n1) if valid else math.nan,
                     "AE_Ann2": abs(rate - n2) if valid else math.nan,
                     "Wave_PCC": float(getattr(m, "Wave_PCC", math.nan))})
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------

def style(ax) -> None:
    ax.grid(True, color=GRID, lw=0.6)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.tick_params(labelsize=8)


def mark_breaths(ax, ann: Dict[str, np.ndarray], t0: float, t1: float) -> None:
    """ann1 breaths as triangles along the top edge, ann2 along the bottom edge of the panel."""
    tr = blended_transform_factory(ax.transData, ax.transAxes)
    for key, y, marker in (("ann1", 0.95, "v"), ("ann2", 0.05, "^")):
        t = ann[key][(ann[key] >= t0) & (ann[key] <= t1)]
        ax.plot(t, np.full(len(t), y), linestyle="none", marker=marker, ms=6, color=COLORS[key],
                mec="white", mew=0.6, transform=tr, zorder=5, clip_on=False)


def draw_spectrum(ax, spectra: Dict[str, dict], ann_rates: Dict[str, float], band_bpm: Tuple[float, float]) -> None:
    """(d) |FFT| of the raw, preprocessed and reconstructed waveform, each scaled to its peak = 1; the peak in
    the rate band marked by a vertical line and a square; the ann1 / ann2 rates as dashed lines."""
    style(ax)
    ax.axvspan(band_bpm[0] / 60.0, band_bpm[1] / 60.0, color="#f3f2ee", lw=0, zorder=0)
    top = 1.0
    for key, color, lw, z in (("raw", COLORS["raw_fft"], 1.2, 2), ("GT", COLORS["GT"], 1.4, 3),
                              ("model", COLORS["model"], MODEL_LW, 4)):
        sp = spectra[key]
        if not (np.isfinite(sp["peak_mag"]) and sp["peak_mag"] > 0):
            continue
        keep = sp["freq"] <= MAX_FREQ_HZ
        yv = sp["mag"][keep] / sp["peak_mag"]
        top = max(top, float(np.nanmax(yv)))
        ax.plot(sp["freq"][keep], yv, color=color, lw=lw, zorder=z + 2)
        ax.axvline(sp["peak_hz"], color=color, lw=1.0, alpha=0.9, zorder=z)
        ax.plot([sp["peak_hz"]], [1.0], marker="s", ms=6, color=color, mec="white", mew=0.8, zorder=z + 5)
    for key in ("ann1", "ann2"):
        if np.isfinite(ann_rates[key]):
            ax.axvline(ann_rates[key] / 60.0, color=COLORS[key], lw=1.3, ls=(0, (4, 2.5)), zorder=7)
    ax.set_xlim(0.0, MAX_FREQ_HZ)
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
    return (f"FFT peak in {band_bpm[0]:g}-{band_bpm[1]:g} /min\n"
            f"raw           {one('raw')}\n"
            f"preprocessed  {one('GT')}\n"
            f"reconstructed {one('model')}\n"
            f"ann1 rate     {fmt(ann_rates['ann1'])} /min\n"
            f"ann2 rate     {fmt(ann_rates['ann2'])} /min")


def draw_page(targets: Sequence[Tuple[Path, PdfPages]], header: Tuple[str, str], t0: float, t1: float,
              raw_t: np.ndarray, raw_v: np.ndarray, ref_t: np.ndarray, ref_v: np.ndarray, pred_v: np.ndarray,
              ann: Dict[str, np.ndarray], shades: Sequence[Tuple[float, float]], texts: Sequence[str],
              spectra: Dict[str, dict], ann_rates: Dict[str, float], band_bpm: Tuple[float, float]) -> None:
    """(a) raw, (b) preprocessed, (c) reconstructed over the preprocessed, for the seconds t0-t1, and (d) the
    |FFT| of the three with their peaks."""
    width, side, left, panel, gap, top_in, bottom_in = 13.0, 3.5, 0.75, 1.7, 0.5, 1.0, 1.35
    fft_panel, fft_gap = 1.9, 1.65  # (d) below (c): room for the time axis of (c) and the top axis of (d)
    height = top_in + bottom_in + 3 * panel + 2 * gap + fft_gap + fft_panel
    fig = plt.figure(figsize=(width, height))
    axes = []
    titles = ("(a) Raw impedance (as recorded, 125 Hz)",
              "(b) Preprocessed reference (64 Hz, band-passed 0.1-0.5 Hz; training target and GT waveform)",
              "(c) Reconstructed waveform (Fusion O, m0) over the preprocessed reference")
    g = minmax(ref_v)
    plot_w = (width - left - side - 0.2) / width
    x_text = (width - side + 0.05) / width
    for k in range(3):
        bottom = bottom_in + fft_panel + fft_gap + (2 - k) * (panel + gap)
        ax = fig.add_axes([left / width, bottom / height, plot_w, panel / height],
                          sharex=axes[0] if axes else None)
        axes.append(ax)
        style(ax)
        for s0, s1 in shades:
            ax.axvspan(max(s0, t0), min(s1, t1), color=SHADE, lw=0, zorder=0)
        if k == 0:
            ax.plot(raw_t, raw_v, color=COLORS["raw"], lw=0.9, zorder=2)
            ax.set_ylabel("Impedance (raw)", fontsize=8)
            mark_breaths(ax, ann, t0, t1)
        elif k == 1:
            ax.plot(ref_t, g, color=COLORS["GT"], lw=1.2, zorder=2)
            ax.set_ylim(-1.3, 1.3)
            ax.set_ylabel("Min-max scaled", fontsize=8)
            mark_breaths(ax, ann, t0, t1)
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
    ax = fig.add_axes([left / width, bottom_in / height, plot_w, fft_panel / height])
    draw_spectrum(ax, spectra, ann_rates, band_bpm)
    ax.set_title("(d) |FFT| of the window (detrend, Hamming, zero padding), each scaled to its peak = 1; "
                 f"peak = largest |FFT| in {band_bpm[0]:g}-{band_bpm[1]:g} /min (shaded)",
                 fontsize=9.5, loc="left", pad=24)
    fig.text(x_text, (bottom_in + fft_panel) / height, peak_text(spectra, ann_rates, band_bpm), family="monospace",
             fontsize=7.6, color=INK, va="top")
    handles = [Line2D([], [], color=COLORS["raw"], lw=1.1),
               Line2D([], [], color=COLORS["model"], lw=MODEL_LW),
               Line2D([], [], linestyle="none", marker="v", ms=6, color=COLORS["ann1"]),
               Line2D([], [], linestyle="none", marker="^", ms=6, color=COLORS["ann2"])]
    labels = ["Raw / preprocessed reference", "Reconstructed (Fusion O, m0)", "ann1 breath (annotator 1)",
              "ann2 breath (annotator 2)"]
    if shades:
        handles.append(Patch(facecolor=SHADE, edgecolor="none"))
        labels.append("ann1 and ann2 disagree (minute)")
    fig.legend(handles, labels, loc="lower left", bbox_to_anchor=(left / width, 0.42 / height), ncol=len(labels),
               frameon=False, fontsize=7.8, handlelength=1.8, columnspacing=1.6)
    fft_handles = [Line2D([], [], color=COLORS["raw_fft"], lw=1.2), Line2D([], [], color=COLORS["GT"], lw=1.4),
                   Line2D([], [], color=COLORS["model"], lw=MODEL_LW),
                   Line2D([], [], linestyle="none", marker="s", ms=6, color=INK),
                   Line2D([], [], color=COLORS["ann1"], lw=1.3, ls=(0, (4, 2.5))),
                   Line2D([], [], color=COLORS["ann2"], lw=1.3, ls=(0, (4, 2.5)))]
    fft_labels = ["|FFT| raw", "|FFT| preprocessed", "|FFT| reconstructed", "FFT peak (line and square)",
                  "ann1 rate", "ann2 rate"]
    fig.legend(fft_handles, fft_labels, loc="lower left", bbox_to_anchor=(left / width, 0.12 / height),
               ncol=len(fft_labels), frameon=False, fontsize=7.8, handlelength=1.8, columnspacing=1.6)
    fig.text(left / width, 1.0 - 0.2 / height, header[0], fontsize=10.5, color=INK, va="top")
    fig.text(left / width, 1.0 - 0.5 / height, header[1], fontsize=7.8, color=MUTED, va="top")
    for path, pdf in targets:
        path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(path, dpi=160)
        pdf.savefig(fig)
    plt.close(fig)


def fmt(v: float, spec: str = "5.1f") -> str:
    return format(v, spec) if np.isfinite(v) else "  n/a"


def draw_subject(args, mod, subject: str, record: dict, raw: dict, files: Dict[str, Path], minutes: pd.DataFrame,
                 row: dict, folder: Path) -> int:
    """Overview of every recording and every stored minute of one subject."""
    fs = mod.TARGET_FS
    folder.mkdir(parents=True, exist_ok=True)
    issue = " · MAE issue subject (left out of the final results)" if row["Issue_Subject"] else ""
    head = (f"BIDMC {subject}: MAE (ann1) {row['MAE_Ann1']:.2f}, MAE (ann2) {row['MAE_Ann2']:.2f} breaths/min; "
            f"|PCC| {row['AbsPCC']:.2f}; {row['Scored_Minutes']} scored minutes{issue}")
    sub = ("Model: 261013 with the stem fusion (Fusion O), member m0; reconstructed waveform "
           + ("raw output" if args.raw_predictions else "band-passed as scored")
           + f"; MAE (ann1) = final results; MAE (ann2) on the same minutes; shaded: |ann1 - ann2| rate "
             f"> {args.disagree_bpm:g} /min")
    count = 0
    multi = len(record["references"]) > 1
    with PdfPages(folder / f"BIDMC_{mod.safe_tag(subject)}_all.pdf") as pdf:
        for segment_id, ref in sorted(record["references"].items()):
            length = len(ref["signal"])
            pred = series(files["outputs"], int(segment_id), length)
            if not args.raw_predictions:
                pred = as_scored(mod, pred, ref)
            if not np.isfinite(pred).any():
                continue
            ref_t = np.arange(length) / fs
            ref_v = np.asarray(ref["signal"], float)
            band = tuple(float(v) for v in ref["profile"]["rr_band_bpm"])
            shift = lag_sec(ref)
            raw_t = np.arange(len(raw["raw"])) / raw["fs"] - shift
            ann = {k: (np.asarray(raw[k], float) - 1.0) / raw["fs"] - shift for k in ("ann1", "ann2")}
            mins = minutes[minutes["Segment_ID"] == int(segment_id)].sort_values("Start_Sec")
            shades = [(m.Start_Sec, m.End_Sec) for m in mins.itertuples() if m.Disagreement]
            rec = f"rec{segment_id}_" if multi else ""
            # overview: the whole recording
            covered = np.flatnonzero(np.isfinite(pred))
            t0, t1 = 0.0, max(length / fs, float(covered[-1] + 1) / fs)
            listing = ", ".join(f"{m.Start_Sec:.0f}-{m.End_Sec:.0f} s ({m.Ann1_Rate:.1f} vs {m.Ann2_Rate:.1f})"
                                for m in mins.itertuples() if m.Disagreement) or "none"
            texts = [f"ann1 breaths {int(np.sum((ann['ann1'] >= t0) & (ann['ann1'] < t1))):4d}\n"
                     f"ann2 breaths {int(np.sum((ann['ann2'] >= t0) & (ann['ann2'] < t1))):4d}\n"
                     f"disagreement minutes\n  " + "\n  ".join(listing.split(", ")),
                     f"minutes stored  {len(mins)}\nscored (ann1)   {int(mins.Scored.sum())}",
                     f"MAE (ann1)  {row['MAE_Ann1']:5.2f} /min\nMAE (ann2)  {row['MAE_Ann2']:5.2f} /min\n"
                     f"|PCC|       {row['AbsPCC']:5.2f}"]
            raw_k = (raw_t >= t0) & (raw_t <= t1)
            spectra = spectra_of(raw["raw"][raw_k], raw["fs"], ref_v, pred, fs, band)
            whole = {k: label_rate(np.asarray(ann[k]) * fs, int(round(t0 * fs)), int(round(t1 * fs)), fs)[1]
                     for k in ("ann1", "ann2")}
            draw_page([(folder / f"BIDMC_{mod.safe_tag(subject)}_{rec}overview.png", pdf)],
                      (f"{head} · whole recording", sub), t0, t1, raw_t[raw_k], raw["raw"][raw_k],
                      ref_t, ref_v, pred, ann, shades, texts, spectra, whole, band)
            count += 1
            # every stored minute
            for m in mins.itertuples():
                t0, t1 = float(m.Start_Sec), float(m.End_Sec)
                a, b = int(round(t0 * fs)), int(round(t1 * fs))
                r = pearson(ref_v[a:b], pred[a:b])
                status = "scored" if m.Scored else f"not scored ({m.Reason})"
                flag = "  <- disagree" if m.Disagreement else ""
                texts = [f"minute {t0:.0f}-{t1:.0f} s\n"
                         f"ann1 {m.Ann1_Count:3d} breaths {fmt(m.Ann1_Rate)} /min\n"
                         f"ann2 {m.Ann2_Count:3d} breaths {fmt(m.Ann2_Rate)} /min\n"
                         f"ann1 - ann2 {fmt(m.Ann1_minus_Ann2_Rate, '+5.1f')} /min{flag}",
                         f"FFT rate     {fmt(m.Ref_FFT_Rate)} /min\n{status}",
                         f"model rate   {fmt(m.Model_Rate)} /min\n"
                         f"error ann1   {fmt(m.AE_Ann1, '5.2f')} /min\n"
                         f"error ann2   {fmt(m.AE_Ann2, '5.2f')} /min\n"
                         f"PCC (stored) {fmt(m.Wave_PCC, '5.2f')}\n"
                         f"r (window)   {fmt(r, '5.2f')}"]
                raw_k = (raw_t >= t0) & (raw_t < t1)
                ref_k = slice(a, b)
                spectra = spectra_of(raw["raw"][raw_k], raw["fs"], ref_v[ref_k], pred[ref_k], fs, band)
                name = f"BIDMC_{mod.safe_tag(subject)}_{rec}{t0:03.0f}-{t1:03.0f}s.png"
                draw_page([(folder / name, pdf)], (f"{head} · {t0:.0f}-{t1:.0f} s", sub), t0, t1,
                          raw_t[raw_k], raw["raw"][raw_k], ref_t[ref_k], ref_v[ref_k], pred[ref_k], ann,
                          shades if m.Disagreement else [], texts, spectra,
                          {"ann1": m.Ann1_Rate, "ann2": m.Ann2_Rate}, band)
                count += 1
    return count


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def same_subject(a: object, b: object) -> bool:
    a, b = str(a).strip(), str(b).strip()
    return a == b or (a.isdigit() and b.isdigit() and int(a) == int(b))


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="BIDMC subjects with ann1 / ann2 disagreement: raw, preprocessed and "
                                            "reconstructed waveforms, MAE and PCC")
    p.add_argument("--subjects", nargs="+", default=list(SUBJECTS), help="BIDMC subjects (default 27 34 46)")
    p.add_argument("--disagree-bpm", type=float, default=DISAGREE_BPM,
                   help="minute shaded when the ann1 and ann2 rates differ by more (default 2 /min)")
    p.add_argument("--raw-predictions", action="store_true",
                   help="draw the raw model output (default: band-passed as the rate was scored)")
    p.add_argument("--root-261013", type=Path, default=ROOT_261013)
    p.add_argument("--root-noise", type=Path, default=ROOT_NOISE, help="results of 261028_Stem_ablation_noise_v1")
    p.add_argument("--output-root", type=Path, default=OUTPUT_ROOT)
    p.add_argument("--script-261013", type=Path, default=None)
    p.add_argument("--bidmc-path", type=Path, default=None)
    return p.parse_args(list(sys.argv[1:] if argv is None else argv))


def main(argv: Optional[Sequence[str]] = None) -> None:
    args = parse_args(argv)
    out_root = Path(args.output_root)
    out_root.mkdir(parents=True, exist_ok=True)
    script = find_script(args.script_261013, [args.root_261013])
    mod = load_script(script)
    if args.bidmc_path:
        mod.DATASET_CONFIGS["bidmc"]["data_path"] = str(args.bidmc_path)
    spec = next(s for s in mod.WINDOW_SPECS if f"{s.order:02d}_{s.name}" == WINDOW)
    print(f"BIDMC: preprocessing of {script.name}", flush=True)
    data = mod.load_dataset("bidmc", spec)
    raw_all = read_bidmc(mod.DATASET_CONFIGS["bidmc"]["data_path"])
    subject_rows, minute_frames, missing = [], [], []
    for wanted in args.subjects:
        subject = next((s for s in data if same_subject(s, wanted)), None)
        raw_key = next((s for s in raw_all if same_subject(s, wanted)), None)
        if subject is None or raw_key is None:
            missing.append({"Subject": wanted, "Missing": "not a BIDMC subject of the preprocessing"})
            print(f"  {wanted}: not a BIDMC subject of the preprocessing", flush=True)
            continue
        files = model_files(args, mod, subject)
        if files is None:
            missing.append({"Subject": subject, "Missing": "no stored result of Fusion O (m0)"})
            print(f"  {subject}: no stored result of Fusion O (m0)", flush=True)
            continue
        minutes = subject_minutes(args, mod, subject, data[subject], raw_all[raw_key], files)
        scored = minutes[minutes.Scored] if not minutes.empty else minutes
        if scored.empty:
            missing.append({"Subject": subject, "Missing": "no scored minute"})
            print(f"  {subject}: no scored minute", flush=True)
            continue
        dis = minutes[minutes.Disagreement]
        row = {"Dataset": DATASET, "Subject": subject,
               "Issue_Subject": any(same_subject(subject, x) for x in ISSUE_SUBJECTS),
               "Scored_Minutes": int(len(scored)), "Stored_Minutes": int(len(minutes)),
               "MAE_Ann1": float(scored.AE_Ann1.mean()), "MAE_Ann2": float(scored.AE_Ann2.mean()),
               "AbsPCC": float(scored.Wave_PCC.abs().mean()),
               "Disagreement_Minutes": int(len(dis)),
               "Disagreement_List": "; ".join(f"{m.Start_Sec:.0f}-{m.End_Sec:.0f} s: ann1 {m.Ann1_Rate:.1f}, "
                                              f"ann2 {m.Ann2_Rate:.1f} /min" for m in dis.itertuples()) or "none",
               "Files": str(files["root"])}
        folder = out_root / f"BIDMC_{mod.safe_tag(subject)}"
        n = draw_subject(args, mod, subject, data[subject], raw_all[raw_key], files, minutes, row, folder)
        row["Figures"] = n
        row["Folder"] = str(folder)
        subject_rows.append(row)
        minute_frames.append(minutes)
        print(f"  {subject}: MAE (ann1) {row['MAE_Ann1']:.2f}, MAE (ann2) {row['MAE_Ann2']:.2f} /min, "
              f"|PCC| {row['AbsPCC']:.2f}, {row['Scored_Minutes']} scored minutes, "
              f"disagreement: {row['Disagreement_List']}{' [issue]' if row['Issue_Subject'] else ''}; "
              f"{n} figures -> {folder}", flush=True)
    del data
    notes = pd.DataFrame([
        ("Model", "261013 with the 1x1 stem fusion (Fusion O), member m0 of the weight decay selected in the fold"),
        ("Raw", "bidmc_data.mat ref.resp_sig.imp, as recorded (125 Hz, no filter)"),
        ("Preprocessed", "reference of the 261013 preprocessing: 64 Hz, band-passed 0.1-0.5 Hz (training target, "
                         "GT waveform of the scoring)"),
        ("Reconstructed", "predicted waveform, " + ("raw output" if args.raw_predictions
                                                   else "band-passed as the rate was scored")),
        ("MAE (ann1)", "mean |stored rate - ann1 breath count| over the scored minutes (expert-label rule of "
                       "261013): the MAE of the final results"),
        ("MAE (ann2)", "the same minutes with the ann2 breath count"),
        ("|PCC|", "mean |Wave_PCC| of the stored minute tables over the scored minutes"),
        ("Disagreement", f"minute with |ann1 rate - ann2 rate| > {args.disagree_bpm:g} /min (rate = 60 / mean "
                         "breath interval in the minute)"),
        ("Issue subject", "BIDMC 27 is the MAE issue subject left out of the final results"),
    ], columns=["Item", "Description"])
    path = out_root / "261119_BIDMC_Annotation_Disagreement.xlsx"
    with pd.ExcelWriter(path, engine="openpyxl") as writer:
        pd.DataFrame(subject_rows).round(4).to_excel(writer, sheet_name="Subjects", index=False)
        (pd.concat(minute_frames, ignore_index=True).round(4) if minute_frames else pd.DataFrame()).to_excel(
            writer, sheet_name="Minutes", index=False)
        pd.DataFrame(missing, columns=["Subject", "Missing"]).to_excel(writer, sheet_name="Missing", index=False)
        notes.to_excel(writer, sheet_name="Notes", index=False)
    print(f"\n{len(subject_rows)} subjects drawn; figures and table under {out_root}")


if __name__ == "__main__":
    main()
