# -*- coding: utf-8 -*-
r"""261114_RRWaveNet_FFT_Spectrum: FFT spectra of the ground truth (GT) and the RRWaveNet
estimate drawn over each other (x = frequency in Hz), with the peak frequency of each in the
legend beside the plot, for every scored 60 s minute of the final RRWaveNet results:

  CapnoBase       261007 (D:\PPG2RSP_RRWaveNet_Inspired\261007_RRWaveNet_NoEvent_v1),
                  variant full, ensemble of 3
  BIDMC, STEAM2   261013 (D:\PPG2RSP_RRWaveNet_Inspired\261013_RRWavNet_STEAM_BIDMC_v1),
                  the weight decay selected per fold (WD_Selection), ensemble of 3

Nothing is trained and no other script is needed: only the stored results are read
(window 01_w10s_nonoverlap):
  Ensemble_Predictions\{subject}_ensemble_outputs.npz   predictions and targets of every
      10 s test window of the held-out subject (64 Hz) and where each window lies
  Fold_Summaries\{subject}_ensemble_RR60s_windows.csv   the scored minutes (start, end,
      Ref_Valid, GT breath count, the stored RRWaveNet rate Pred_Wave_FFT_RR)
  Data_Quality\01_w10s_nonoverlap\{Dataset}_window_quality.csv   the respiratory band of
      every recording (Resp_Band_Low_Hz / Resp_Band_High_Hz)

Signals of a minute (exactly as 261007 / 261013 score it)
  The windows are put back on the recording's time axis (overlap-add mean, as
  reconstruct_segment). RRWaveNet: the run of contiguous windows is band-passed with the
  recording's respiratory band (the run's own filter: 3rd-order Butterworth, zero phase;
  transfer-function form for 0.1-0.5 Hz, second-order sections otherwise), then the minute
  is cut out. GT: the stored target of the same seconds (the reference the model was trained
  and scored against: BIDMC impedance, STEAM2 belt, CapnoBase the capnography target of
  261007), band-passed already.
  Spectrum: detrend, Hamming window, zero padding to 16384 points (fft_rr_bpm of 261007 /
  261013); |FFT| of each curve divided by its largest value in the rate band.
  Peak: the largest value in the rate band, 0.1-0.5 Hz (6-30 breaths/min) for every dataset.
  Where the scoring used the same band (BIDMC, STEAM2), the RRWaveNet peak is the stored
  Pred_Wave_FFT_RR / 60; the table checks it for every minute (Matches_Stored_Rate).

Minutes: the scored ones (Ref_Valid) of every subject; the subjects with an MAE issue are left
out as in the final results (STEAM2 22_LMS, 26_CKK, 12_YJJ; BIDMC 27; CapnoBase 0115).
--all-minutes draws the not scored minutes too; --keep-issue-subjects keeps those subjects.

Outputs (D:\PPG2RSP_RRWaveNet_Inspired\261114_RRWaveNet_FFT_Spectrum)
  {Dataset}\{subject}\{subject}[_seg{k}]_{t0}-{t1}s_fft.png    one figure per minute
  {Dataset}\{Dataset}_fft.pdf                                   every minute, one page each
  261114_RRWaveNet_FFT_Spectrum.xlsx
    Minutes   every minute: GT and RRWaveNet peak frequency (Hz), |difference|, the GT rate
              from the breaths (Hz), the stored RRWaveNet rate (Hz), Matches_Stored_Rate, figure
    Subjects  per subject: minutes, mean +- SD of the peak frequencies and |difference|
    Notes

Run: python 261114_RRWaveNet_FFT_Spectrum.py      (Spyder: runfile(...), no options needed)
Options: --root-261007, --root-261013, --output-root, --datasets, --subjects, --all-minutes,
--keep-issue-subjects, --max-freq-hz (x axis, default 1.2)
"""

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.lines import Line2D
from scipy.signal import butter, detrend, filtfilt, sosfiltfilt

# ---------------------------------------------------------------------------
# Settings
# ---------------------------------------------------------------------------

NAME = "261114_RRWaveNet_FFT_Spectrum"
WORKSPACE = Path(r"D:\PPG2RSP_RRWaveNet_Inspired")
OUTPUT_ROOT = WORKSPACE / NAME
ROOT_261007 = WORKSPACE / "261007_RRWaveNet_NoEvent_v1"
ROOT_261013 = WORKSPACE / "261013_RRWavNet_STEAM_BIDMC_v1"
WINDOW = "01_w10s_nonoverlap"
FS = 64  # sampling rate of the stored RRWaveNet outputs (Hz)
DATASETS = ("CapnoBase", "BIDMC", "STEAM2")
# dataset -> run of the final results, rate band (breaths/min), GT modality
DATASET_RUN = {"CapnoBase": "261007", "BIDMC": "261013", "STEAM2": "261013"}
RATE_BAND_BPM = {"CapnoBase": (6.0, 30.0), "BIDMC": (6.0, 30.0), "STEAM2": (6.0, 30.0)}  # 0.1-0.5 Hz
MODALITY = {"CapnoBase": "capnography", "BIDMC": "impedance", "STEAM2": "belt"}
ADULT_BAND_HZ = (0.1, 0.5)  # RESP_BAND_HZ of 261007 / 261013 (transfer-function filter)
CAPNOBASE_DEFAULT_BAND_HZ = (0.05, 0.5)  # only if the quality table has no band for a recording
EXCLUDED = {"STEAM2": ("22_LMS", "26_CKK", "12_YJJ"), "BIDMC": ("27",), "CapnoBase": ("0115",)}
FFT_MIN_POINTS = 16384
MAX_FREQ_HZ = 1.2
GT_COLOR, EST_COLOR = "#000000", "#f64636"
BAND_COLOR, INK, MUTED, GRID = "#b84fb8", "#1f1f1f", "#6b6a66", "#e4e3df"


# ---------------------------------------------------------------------------
# Signal processing of 261007 / 261013 (the same steps as their scoring)
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


def bandpass_ba(signal: np.ndarray, fs: float, low: float, high: float, order: int = 3) -> np.ndarray:
    """bandpass_filter of 261007 / 261013 (transfer-function form, zero phase)."""
    arr = sanitize(signal)
    nyquist = 0.5 * float(fs)
    low, high = max(float(low), 1e-5), min(float(high), nyquist * 0.99)
    if arr.size < max(32, order * 6 + 1) or high <= low:
        return arr
    b, a = butter(order, [low / nyquist, high / nyquist], btype="band")
    try:
        return np.asarray(filtfilt(b, a, arr), dtype=np.float64)
    except ValueError:
        return arr


def bandpass_sos(signal: np.ndarray, fs: float, low: float, high: float, order: int = 3) -> np.ndarray:
    """sos_bandpass of 261007 / 261013 (second-order sections, zero phase)."""
    arr = sanitize(signal)
    nyquist = 0.5 * float(fs)
    high = min(float(high), nyquist * 0.99)
    if arr.size < max(32, order * 6 + 1) or high <= low:
        return arr
    sos = butter(order, [low / nyquist, high / nyquist], btype="band", output="sos")
    try:
        return np.asarray(sosfiltfilt(sos, arr), dtype=np.float64)
    except ValueError:
        return arr


def respiratory_filter(signal: np.ndarray, fs: float, band: Tuple[float, float]) -> np.ndarray:
    """respiratory_filter of 261007 / 261013: the transfer-function filter for the fixed 0.1-0.5 Hz band,
    second-order sections for any other band (CapnoBase: 0.05 Hz to the heart-rate dependent upper edge)."""
    if np.allclose(band, ADULT_BAND_HZ):
        return bandpass_ba(signal, fs, band[0], band[1])
    return bandpass_sos(signal, fs, band[0], band[1])


def spectrum(x: np.ndarray, fs: float) -> Tuple[np.ndarray, np.ndarray]:
    """Frequency (Hz) and |FFT| as fft_rr_bpm: detrend, Hamming window, zero padding."""
    x = sanitize(x)
    if len(x) < 8 or float(np.std(x)) < 1e-8:
        return np.zeros(1), np.full(1, np.nan)
    windowed = detrend(x) * np.hamming(len(x))
    n_fft = max(FFT_MIN_POINTS, 1 << int(math.ceil(math.log2(len(x)))))
    return np.fft.rfftfreq(n_fft, d=1.0 / float(fs)), np.abs(np.fft.rfft(windowed, n=n_fft))


def band_peak(freq: np.ndarray, mag: np.ndarray, band_bpm: Tuple[float, float]) -> Tuple[float, float]:
    """Frequency (Hz) and value of the largest |FFT| in the rate band (the bins fft_rr_bpm compares, in breaths/min)."""
    keep = (freq * 60.0 >= band_bpm[0]) & (freq * 60.0 <= band_bpm[1]) & np.isfinite(mag)
    if not keep.any():
        return math.nan, math.nan
    k = int(np.flatnonzero(keep)[np.argmax(mag[keep])])
    return float(freq[k]), float(mag[k])


def reconstruct(predictions: np.ndarray, targets: np.ndarray, starts: np.ndarray
                ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Windows of one recording back on its time axis (overlap-add mean, as reconstruct_segment)."""
    length = int(predictions.shape[1])
    end = int(starts.max()) + length
    pred_sum, true_sum = np.zeros(end), np.zeros(end)
    coverage = np.zeros(end, dtype=np.int32)
    for k, start in enumerate(starts):
        start = int(start)
        pred_sum[start:start + length] += sanitize(predictions[k])[:length]
        true_sum[start:start + length] += sanitize(targets[k])[:length]
        coverage[start:start + length] += 1
    valid = coverage > 0
    pred, true = np.full(end, np.nan), np.full(end, np.nan)
    pred[valid] = pred_sum[valid] / coverage[valid]
    true[valid] = true_sum[valid] / coverage[valid]
    return pred, true, valid


def contiguous_runs(mask: np.ndarray) -> List[Tuple[int, int]]:
    mask = np.asarray(mask, dtype=bool)
    if not mask.any():
        return []
    padded = np.concatenate([[False], mask, [False]])
    changes = np.flatnonzero(padded[1:] != padded[:-1])
    return [(int(a), int(b)) for a, b in changes.reshape(-1, 2)]


# ---------------------------------------------------------------------------
# Stored results
# ---------------------------------------------------------------------------

def read_json(path: Path) -> Optional[dict]:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


def fold_files(root: Path, dataset: str, run: str) -> List[dict]:
    """Every fold of a dataset: subject tag, model folder, per-minute table and ensemble outputs."""
    folds = []
    if run == "261013":  # the weight decay selected per fold
        for path in sorted((root / "WD_Selection" / WINDOW / dataset).glob("*_selection.json")):
            variant = (read_json(path) or {}).get("Selected_Variant")
            if variant:
                folds.append({"Tag": path.name[: -len("_selection.json")], "Variant": variant})
    else:  # 261007: the full model
        for path in sorted((root / "full" / WINDOW / dataset / "Fold_Summaries").glob("*_ensemble_RR60s_windows.csv")):
            if not path.name.endswith("_selected_ensemble_RR60s_windows.csv"):
                folds.append({"Tag": path.name[: -len("_ensemble_RR60s_windows.csv")], "Variant": "full"})
    for fold in folds:
        condition = root / fold["Variant"] / WINDOW / dataset
        fold["Minutes_File"] = condition / "Fold_Summaries" / f"{fold['Tag']}_ensemble_RR60s_windows.csv"
        fold["Outputs_File"] = condition / "Ensemble_Predictions" / f"{fold['Tag']}_ensemble_outputs.npz"
    return folds


def recording_bands(root: Path, dataset: str) -> Dict[Tuple[str, int], Tuple[float, float]]:
    """Respiratory band of every recording (subject, segment) from the stored quality table."""
    path = root / "Data_Quality" / WINDOW / f"{dataset}_window_quality.csv"
    bands = {}
    if path.exists():
        frame = pd.read_csv(path, dtype={"Subject": str}, encoding="utf-8-sig",
                            usecols=lambda c: c in ("Subject", "Segment_ID", "Resp_Band_Low_Hz", "Resp_Band_High_Hz"))
        if {"Subject", "Segment_ID", "Resp_Band_Low_Hz", "Resp_Band_High_Hz"} <= set(frame.columns):
            for row in frame.drop_duplicates(["Subject", "Segment_ID"]).itertuples(index=False):
                bands[(str(row.Subject), int(row.Segment_ID))] = (float(row.Resp_Band_Low_Hz),
                                                                  float(row.Resp_Band_High_Hz))
    return bands


def same_subject(a: object, b: object) -> bool:
    a, b = str(a).strip(), str(b).strip()
    return a == b or (a.isdigit() and b.isdigit() and int(a) == int(b))


def safe_name(text: str) -> str:
    return "".join(ch if ch.isalnum() or ch in "-_." else "_" for ch in str(text))


# ---------------------------------------------------------------------------
# Figure
# ---------------------------------------------------------------------------

def draw(path: Path, pdf, title: str, subtitle: str, curves: List[dict], band_hz: Tuple[float, float],
         max_freq: float) -> None:
    """GT and RRWaveNet spectra in one plot; the legend with the peak frequencies beside it."""
    width, height = 7.8, 3.0
    fig = plt.figure(figsize=(width, height))
    ax = fig.add_axes([0.75 / width, 0.62 / height, 4.55 / width, 1.78 / height])
    handles, labels, top = [], [], 1.0
    for c in curves:
        keep = c["freq"] <= max_freq
        ok = np.isfinite(c["peak_mag"]) and c["peak_mag"] > 0
        y = c["mag"] / c["peak_mag"] if ok else np.full_like(c["mag"], np.nan)
        if np.isfinite(y[keep]).any():
            top = max(top, float(np.nanmax(y[keep])))
            ax.plot(c["freq"][keep], y[keep], color=c["color"], lw=c["lw"], solid_capstyle="round", zorder=c["z"])
        if np.isfinite(c["peak_hz"]):
            ax.axvline(c["peak_hz"], color=c["color"], lw=1.0, alpha=0.9, zorder=c["z"] - 1)
            ax.plot([c["peak_hz"]], [1.0], marker="s", ms=5.5, color=c["color"], mec="white", mew=0.8,
                    zorder=c["z"] + 1)
        handles.append(Line2D([], [], color=c["color"], lw=c["lw"], marker="s", ms=5.5, mec="white", mew=0.8))
        labels.append(f"{c['label']}\npeak {c['peak_hz']:.3f} Hz" if np.isfinite(c["peak_hz"])
                      else f"{c['label']}\npeak n/a")
    for edge in band_hz:
        ax.axvline(edge, color=BAND_COLOR, lw=1.1, ls=(0, (4, 3)), zorder=1)
    handles.append(Line2D([], [], color=BAND_COLOR, lw=1.1, ls=(0, (4, 3))))
    labels.append(f"rate band {band_hz[0]:.2f}-{band_hz[1]:.2f} Hz")
    ax.set_xlim(0.0, max_freq)
    ax.set_ylim(0.0, top * 1.08)
    ax.set_xlabel("Frequency (Hz)")
    ax.set_ylabel("|FFT| (peak = 1)", fontsize=8.5)
    ax.grid(True, color=GRID, lw=0.6)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    fig.legend(handles, labels, loc="upper left", bbox_to_anchor=(5.5 / width, 2.42 / height), frameon=False,
               fontsize=8.2, labelspacing=1.0, borderaxespad=0.0, handlelength=2.2)
    fig.text(0.75 / width, 1.0 - 0.16 / height, title, fontsize=10, color=INK, va="top")
    fig.text(0.75 / width, 1.0 - 0.40 / height, subtitle, fontsize=7.4, color=MUTED, va="top")
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=200)
    pdf.savefig(fig)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def dataset_minutes(args: argparse.Namespace, dataset: str, out_root: Path) -> List[dict]:
    run = DATASET_RUN[dataset]
    root = Path(args.root_261007 if run == "261007" else args.root_261013)
    if not root.is_dir():
        print(f"{dataset}: results folder of {run} not found ({root}); skipped")
        return []
    band_hz = (RATE_BAND_BPM[dataset][0] / 60.0, RATE_BAND_BPM[dataset][1] / 60.0)
    bands = recording_bands(root, dataset)
    folds = [f for f in fold_files(root, dataset, run) if f["Outputs_File"].exists() and f["Minutes_File"].exists()]
    excluded = [] if args.keep_issue_subjects else list(EXCLUDED[dataset])
    print(f"\n{dataset}: RRWaveNet {run} ({root}), {len(folds)} subjects with stored outputs", flush=True)
    rows = []
    folder = out_root / dataset
    folder.mkdir(parents=True, exist_ok=True)
    with PdfPages(folder / f"{dataset}_fft.pdf") as pdf:
        for fold in folds:
            minutes = pd.read_csv(fold["Minutes_File"], dtype={"Subject": str}, encoding="utf-8-sig")
            subject = str(minutes["Subject"].iloc[0]) if "Subject" in minutes and len(minutes) else fold["Tag"]
            if any(same_subject(subject, x) for x in excluded):
                print(f"  {subject}: left out (MAE issue)")
                continue
            if args.subjects and not any(same_subject(subject, x) for x in args.subjects):
                continue
            if not args.all_minutes:
                minutes = minutes[minutes["Ref_Valid"].astype(bool)]
            if minutes.empty:
                print(f"  {subject}: no scored minute")
                continue
            with np.load(fold["Outputs_File"], allow_pickle=False) as data:
                predictions, targets = data["predictions"], data["targets"]
                segment_ids, starts = data["segment_ids"].astype(int), data["window_starts"].astype(int)
            multi = len(np.unique(segment_ids)) > 1
            count = 0
            for segment_id in sorted(set(minutes["Segment_ID"].astype(int))):
                pick = segment_ids == segment_id
                if not pick.any():
                    continue
                pred, true, valid = reconstruct(predictions[pick], targets[pick], starts[pick])
                band = bands.get((subject, segment_id)) or bands.get((fold["Tag"], segment_id))
                if band is None:
                    band = CAPNOBASE_DEFAULT_BAND_HZ if dataset == "CapnoBase" else ADULT_BAND_HZ
                    print(f"  {subject} segment {segment_id}: no band in the quality table, {band} used")
                filtered = np.full(len(pred), np.nan)
                for a, b in contiguous_runs(valid):
                    filtered[a:b] = respiratory_filter(pred[a:b], FS, band)
                for minute in minutes[minutes["Segment_ID"].astype(int) == segment_id].itertuples(index=False):
                    t0, t1 = float(minute.RR_Window_Start_Sec), float(minute.RR_Window_End_Sec)
                    a, b = int(round(t0 * FS)), int(round(t1 * FS))
                    if b > len(pred) or not valid[a:b].all():
                        continue
                    curves = []
                    for label, series, color, lw, z in (
                            (f"GT ({MODALITY[dataset]})", true[a:b], GT_COLOR, 1.6, 4),
                            (f"Estimated (RRWaveNet {run})", filtered[a:b], EST_COLOR, 1.9, 5)):
                        freq, mag = spectrum(series, FS)
                        peak_hz, peak_mag = band_peak(freq, mag, RATE_BAND_BPM[dataset])
                        curves.append({"label": label, "color": color, "lw": lw, "z": z, "freq": freq, "mag": mag,
                                       "peak_hz": peak_hz, "peak_mag": peak_mag})
                    gt_rate_hz = float(minute.GT_Breath_Count) * (60.0 / (t1 - t0)) / 60.0
                    stored_hz = float(minute.Pred_Wave_FFT_RR) / 60.0
                    seg_text = f"_seg{segment_id}" if multi else ""
                    stem = f"{safe_name(subject)}{seg_text}_{t0:.0f}-{t1:.0f}s"
                    path = folder / safe_name(subject) / f"{stem}_fft.png"
                    title = (f"{dataset} {subject}" + (f" · recording {segment_id}" if multi else "")
                             + f" · minute {t0:.0f}-{t1:.0f} s · FFT spectrum, GT vs RRWaveNet")
                    subtitle = (f"RRWaveNet {run} ensemble ({fold['Variant']}); |FFT| (detrend, Hamming) / its peak in "
                                f"the rate band; GT rate from the breaths {gt_rate_hz:.3f} Hz"
                                + ("" if bool(minute.Ref_Valid) else " · not scored (reference)"))
                    draw(path, pdf, title, subtitle, curves, band_hz, args.max_freq_hz)
                    gt, est = curves
                    rows.append({
                        "Dataset": dataset, "Run": run, "Variant": fold["Variant"], "Subject": subject,
                        "Segment_ID": segment_id, "Start_Sec": t0, "End_Sec": t1, "Scored": bool(minute.Ref_Valid),
                        "Rate_Band_Low_Hz": band_hz[0], "Rate_Band_High_Hz": band_hz[1],
                        "Filter_Band_Low_Hz": band[0], "Filter_Band_High_Hz": band[1],
                        "GT_Peak_Hz": gt["peak_hz"], "RRWaveNet_Peak_Hz": est["peak_hz"],
                        "Abs_Diff_Peak_Hz": abs(est["peak_hz"] - gt["peak_hz"]),
                        "GT_Rate_From_Breaths_Hz": gt_rate_hz,
                        "Stored_RRWaveNet_Rate_Hz": stored_hz,
                        "Matches_Stored_Rate": bool(np.isfinite(stored_hz) and abs(est["peak_hz"] - stored_hz) < 1e-6),
                        "Figure": str(path.relative_to(out_root)),
                    })
                    count += 1
            print(f"  {subject}: {count} minutes", flush=True)
    return rows


def subject_table(minutes: pd.DataFrame) -> pd.DataFrame:
    if minutes.empty:
        return minutes
    out = []
    for (dataset, subject), g in minutes.groupby(["Dataset", "Subject"], sort=False):
        row = {"Dataset": dataset, "Run": g["Run"].iloc[0], "Subject": subject, "Minutes": int(len(g))}
        for col in ("GT_Peak_Hz", "RRWaveNet_Peak_Hz", "Abs_Diff_Peak_Hz"):
            v = g[col].to_numpy(float)
            v = v[np.isfinite(v)]
            row[f"{col}_Mean"] = float(v.mean()) if v.size else math.nan
            row[f"{col}_SD"] = float(v.std(ddof=1)) if v.size > 1 else math.nan
        row["Matches_Stored_Rate"] = f"{int(g['Matches_Stored_Rate'].sum())}/{len(g)}"
        out.append(row)
    return pd.DataFrame(out)


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="FFT spectra (Hz) of GT and RRWaveNet per scored minute: "
                                                 "CapnoBase 261007, BIDMC and STEAM2 261013")
    parser.add_argument("--root-261007", type=Path, default=ROOT_261007)
    parser.add_argument("--root-261013", type=Path, default=ROOT_261013)
    parser.add_argument("--output-root", type=Path, default=OUTPUT_ROOT)
    parser.add_argument("--datasets", nargs="+", choices=list(DATASETS), default=list(DATASETS))
    parser.add_argument("--subjects", nargs="+", default=None, help="only these subjects")
    parser.add_argument("--all-minutes", action="store_true", help="also the minutes that were not scored")
    parser.add_argument("--keep-issue-subjects", action="store_true", help="keep the subjects with an MAE issue")
    parser.add_argument("--max-freq-hz", type=float, default=MAX_FREQ_HZ, help="right end of the frequency axis")
    return parser.parse_args(list(sys.argv[1:] if argv is None else argv))


def main(argv: Optional[Sequence[str]] = None) -> None:
    args = parse_args(argv)
    out_root = Path(args.output_root)
    out_root.mkdir(parents=True, exist_ok=True)
    rows = []
    for dataset in args.datasets:
        rows += dataset_minutes(args, dataset, out_root)
    minutes = pd.DataFrame(rows)
    notes = pd.DataFrame([
        ("Results", f"CapnoBase: 261007 ({args.root_261007}, variant full); BIDMC, STEAM2: 261013 "
                    f"({args.root_261013}, weight decay selected per fold); ensemble of 3; window {WINDOW}"),
        ("Minutes", "the scored 60 s minutes (Ref_Valid) of the stored per-minute tables"
                    + (" and the not scored ones" if args.all_minutes else "")),
        ("GT", "the stored target of the minute (the reference the model was trained and scored against)"),
        ("RRWaveNet", "the stored ensemble output; the run of contiguous windows band-passed with the recording's "
                      "band (as the scoring), then the minute cut out"),
        ("Spectrum", f"detrend, Hamming window, zero padding to {FFT_MIN_POINTS} points (fft_rr_bpm); "
                     "each curve / its peak in the rate band"),
        ("Peak", "largest |FFT| in the rate band 0.1-0.5 Hz (every dataset); Matches_Stored_Rate: RRWaveNet "
                 "peak = stored Pred_Wave_FFT_RR / 60 (CapnoBase was scored in 0.05-1.0 Hz, so it can differ there)"),
        ("Left out", "none (--keep-issue-subjects)" if args.keep_issue_subjects else
         "; ".join(f"{d} {', '.join(s)}" for d, s in EXCLUDED.items())),
    ], columns=["Item", "Description"])
    path = out_root / f"{NAME}.xlsx"
    with pd.ExcelWriter(path, engine="openpyxl") as writer:
        (minutes.round(5) if not minutes.empty else minutes).to_excel(writer, sheet_name="Minutes", index=False)
        subject_table(minutes).round(5).to_excel(writer, sheet_name="Subjects", index=False)
        notes.to_excel(writer, sheet_name="Notes", index=False)
    if not minutes.empty:
        print(f"\nRRWaveNet peak = stored rate in {int(minutes['Matches_Stored_Rate'].sum())} of {len(minutes)} minutes")
    print(f"Figures under {out_root}\\<Dataset>\\<subject>; table: {path}")


if __name__ == "__main__":
    main()
