# -*- coding: utf-8 -*-
r"""261114_RRWaveNet_FFT_Spectrum: the FFT spectra of the final waveform figures.
For every window drawn by 261108_Wavefom_Comparison (the same subjects, recordings and
seconds, chosen by its own code), the spectrum of the ground truth (GT) and of the
RRWaveNet estimate in that window are drawn over each other in one plot (x = frequency in
Hz), and the legend beside the plot gives the peak frequency of each.

Signals and spectrum (as 261108 scores the rate)
  GT          the reference band-passed 0.1-0.5 Hz (261108's GT of the figures), 30 Hz
  RRWaveNet   the stored ensemble output on the GT's time axis (training sign / lag undone,
              low-pass 1 Hz, inverted when r < 0 over the recording: 261108's alignment)
  spectrum    |FFT| of the window with its mean removed, zero-padded to 16384 points
              (261108's fft_rate), each curve divided by its largest value in the rate band
  peak        the largest value in the rate band of the dataset (CapnoBase 3-60 breaths/min =
              0.05-1.0 Hz; BIDMC, STEAM2 6-30 breaths/min = 0.1-0.5 Hz): the frequency of
              261108's RRWaveNet_Rate (breaths/min / 60). The band edges are dashed lines.

Nothing is trained and nothing of 261108 is changed: 261108_Wavefom_Comparison.py is loaded
as a module (found in --script-261108, the folder of this file and its parent,
D:\PPG2RSP_RRWaveNet_Inspired\261108_Wavefom_Comparison, D:\PPG2RSP_RRWaveNet_Inspired and
the folders under it), and its readers, windows and selection are used with its default
settings. Its options can be given here and are passed on unchanged (--datasets, --sizes,
--top, --subjects, --root-261007, --root-261013, --correncoder-root, --respdiff-root, ...).

Outputs (D:\PPG2RSP_RRWaveNet_Inspired\261114_RRWaveNet_FFT_Spectrum)
  All_Subjects\{Dataset}\{30s,60s}\{rank}_{subject}_{t0}-{t1}s_fft.png   (the name of 261108's
      overlap figure with _fft; --top N: {Dataset}\{30s,60s}\... as 261108)
  All_Subjects\{Dataset}\{30s,60s}\{Dataset}_{size}s_fft.pdf             every window, one page each
  261114_RRWaveNet_FFT_Spectrum.xlsx   every window: GT and RRWaveNet peak frequency (Hz),
      their difference, 261108's GT rate from the breaths, and the figure files

Run: python 261114_RRWaveNet_FFT_Spectrum.py     (Spyder: runfile(...), no options needed)
Options of this file: --output-root, --max-freq-hz (x axis, default 1.2), --script-261108.
"""

import argparse
import importlib.util
import math
import sys
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.lines import Line2D

NAME = "261114_RRWaveNet_FFT_Spectrum"
WORKSPACE = Path(r"D:\PPG2RSP_RRWaveNet_Inspired")
OUTPUT_ROOT = WORKSPACE / NAME
SCRIPT_261108 = "261108_Wavefom_Comparison.py"
HERE = Path(__file__).resolve().parent
MAX_FREQ_HZ = 1.2  # right end of the frequency axis
FFT_POINTS = 16384  # zero padding of 261108's fft_rate
GT_COLOR, EST_COLOR = "#000000", "#f64636"  # GT black, RRWaveNet coral (the colours of 261108's figures)
BAND_COLOR, INK, MUTED, GRID = "#b84fb8", "#1f1f1f", "#6b6a66", "#e4e3df"


# ---------------------------------------------------------------------------
# 261108
# ---------------------------------------------------------------------------

def find_261108(given: Optional[Path]) -> Path:
    folders: List[Path] = [Path(given)] if given else []
    folders += [HERE, HERE.parent, WORKSPACE / Path(SCRIPT_261108).stem, WORKSPACE]
    for base in (WORKSPACE, HERE.parent):
        if base.is_dir():
            folders += sorted(child for child in base.iterdir() if child.is_dir())
    for folder in folders:
        path = folder / SCRIPT_261108 if folder.suffix != ".py" else folder
        if path.is_file():
            return path.resolve()
    raise FileNotFoundError(f"{SCRIPT_261108} not found (looked in the folder of this file, its parent and "
                            f"{WORKSPACE}); give it with --script-261108")


def load_261108(path: Path):
    name = "waveform_261108"
    spec = importlib.util.spec_from_file_location(name, str(path))
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


# ---------------------------------------------------------------------------
# Spectrum and figure
# ---------------------------------------------------------------------------

def spectrum(x: np.ndarray, fs: float) -> Tuple[np.ndarray, np.ndarray]:
    """Frequency (Hz) and |FFT| of the window, mean removed, zero-padded as 261108's fft_rate."""
    x = np.asarray(x, float)
    x = x[np.isfinite(x)]
    n = max(FFT_POINTS, len(x))
    return np.fft.rfftfreq(n, 1.0 / fs), np.abs(np.fft.rfft(x - np.mean(x), n))


def band_peak(freq: np.ndarray, mag: np.ndarray, band_hz: Tuple[float, float]) -> Tuple[float, float]:
    """Frequency and value of the largest value inside the band."""
    keep = (freq >= band_hz[0]) & (freq <= band_hz[1])
    if not keep.any() or not np.isfinite(mag[keep]).any() or mag[keep].max() <= 0:
        return math.nan, math.nan
    k = int(np.argmax(np.where(keep, mag, -np.inf)))
    return float(freq[k]), float(mag[k])


def draw(path: Path, pdf, title: str, subtitle: str, curves: List[dict], band_hz: Tuple[float, float],
         max_freq: float) -> None:
    """GT and RRWaveNet spectra in one plot; legend with the peak frequencies beside it."""
    width, height = 7.8, 3.0
    fig = plt.figure(figsize=(width, height))
    ax = fig.add_axes([0.75 / width, 0.62 / height, 4.55 / width, 1.78 / height])
    handles, labels = [], []
    top = 1.0
    for c in curves:
        keep = c["freq"] <= max_freq
        y = c["mag"] / c["peak_mag"] if np.isfinite(c["peak_mag"]) and c["peak_mag"] > 0 else c["mag"] * np.nan
        top = max(top, float(np.nanmax(y[keep])) if np.isfinite(y[keep]).any() else 1.0)
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

def parse_args(argv: Optional[Sequence[str]] = None) -> Tuple[argparse.Namespace, List[str]]:
    parser = argparse.ArgumentParser(description="FFT spectra (Hz) of GT and RRWaveNet in the windows of 261108")
    parser.add_argument("--output-root", type=Path, default=OUTPUT_ROOT)
    parser.add_argument("--max-freq-hz", type=float, default=MAX_FREQ_HZ, help="right end of the frequency axis")
    parser.add_argument("--script-261108", type=Path, default=None, help="261108_Wavefom_Comparison.py or its folder")
    return parser.parse_known_args(list(sys.argv[1:] if argv is None else argv))


def main(argv: Optional[Sequence[str]] = None) -> None:
    args, rest = parse_args(argv)
    script = find_261108(args.script_261108)
    print(f"{NAME}: windows of {script}")
    m = load_261108(script)
    margs = m.parse_args(rest)  # 261108's options (its defaults when none are given)
    every = margs.top <= 0
    out_root = Path(args.output_root)
    fig_root = out_root / m.ALL_FOLDER if every else out_root
    fig_root.mkdir(parents=True, exist_ok=True)
    m.set_data_paths(margs)
    datasets = list(margs.datasets)
    if "CapnoBase" in datasets and not m.load_capnobase_reader(margs):
        datasets.remove("CapnoBase")
    table = []
    for dataset in datasets:
        run = next(r for r, ds in m.RUN_DATASETS.items() if dataset in ds)
        root = Path(getattr(margs, f"root_{run}"))
        if not root.exists():
            print(f"[{run}] results folder not found: {root} ({dataset} skipped)")
            continue
        rows, store, n_recordings = m.dataset_windows(margs, dataset, run, root)
        if not rows:
            continue
        key = m.DATASET_KEYS[dataset]
        band_bpm = m.RATE_BAND_BPM[key]
        band_hz = (band_bpm[0] / 60.0, band_bpm[1] / 60.0)
        gt_band = f"{m.GT_BAND_HZ[0]:g}-{m.GT_BAND_HZ[1]:g} Hz"
        for size in margs.sizes:
            ranked = m.best_windows(rows, size)  # 261108's choice of the drawn windows
            pick = [dict(r, Rank=r["Order"]) for r in ranked] if every else m.choose(ranked, margs.top)
            digits = len(str(len(pick))) if every else 1
            folder = fig_root / dataset / f"{size}s"
            folder.mkdir(parents=True, exist_ok=True)
            with PdfPages(folder / f"{dataset}_{size}s_fft.pdf") as pdf:
                for c in pick:
                    values = store[(c["Subject"], c["Recording"])]
                    a, b = int(round(c["Start_Sec"] * m.FS30)), int(round(c["End_Sec"] * m.FS30))
                    curves = []
                    for label, series, color, lw, z in (
                            (f"GT ({m.MODALITIES[key]}, {gt_band})", "GT", GT_COLOR, 1.6, 4),
                            (f"Estimated (RRWaveNet {run})", "RRWaveNet", EST_COLOR, 1.9, 5)):
                        freq, mag = spectrum(values[series][a:b], m.FS30)
                        peak_hz, peak_mag = band_peak(freq, mag, band_hz)
                        curves.append({"label": label, "series": series, "color": color, "lw": lw, "z": z,
                                       "freq": freq, "mag": mag, "peak_hz": peak_hz, "peak_mag": peak_mag})
                    single = n_recordings.get(c["Subject"], 1) == 1
                    who = c["Subject"] if single else f"{c['Subject']} ({c['Recording']})"
                    stem = (f"{c['Rank']:0{digits}d}_{m.safe_name(c['Subject'] if single else c['Recording'])}_"
                            f"{c['Start_Sec']:.0f}-{c['End_Sec']:.0f}s")
                    title = (f"{dataset} {who} · {size} s window {c['Start_Sec']:.0f}-{c['End_Sec']:.0f} s · "
                             f"FFT spectrum, GT vs RRWaveNet")
                    subtitle = (f"window of 261108 figure {stem}_overlap.png · each |FFT| / its peak in the rate band "
                                f"· GT rate from the breaths {c['GT_RR'] / 60.0:.3f} Hz")
                    path = folder / f"{stem}_fft.png"
                    draw(path, pdf, title, subtitle, curves, band_hz, args.max_freq_hz)
                    gt, est = curves
                    table.append({
                        "Dataset": dataset, "Run": run, "Size_Sec": size, "Rank": c["Rank"], "Subject": c["Subject"],
                        "Recording": c["Recording"], "Start_Sec": c["Start_Sec"], "End_Sec": c["End_Sec"],
                        "Rate_Band_Low_Hz": band_hz[0], "Rate_Band_High_Hz": band_hz[1],
                        "GT_Peak_Hz": gt["peak_hz"], "RRWaveNet_Peak_Hz": est["peak_hz"],
                        "Abs_Diff_Hz": abs(est["peak_hz"] - gt["peak_hz"]),
                        "GT_Rate_From_Breaths_Hz": c["GT_RR"] / 60.0,
                        "RRWaveNet_Rate_261108_Hz": c.get("RRWaveNet_Rate", math.nan) / 60.0,
                        "FFT_Figure": str(path.relative_to(fig_root)),
                        "Overlap_Figure_261108": str(Path(dataset) / f"{size}s" / f"{stem}_overlap.png"),
                    })
            print(f"  {dataset} {size} s: {len(pick)} spectrum figures -> {folder}", flush=True)
    frame = pd.DataFrame(table)
    path = out_root / f"{NAME}.xlsx"
    notes = pd.DataFrame([
        ("Windows", "the windows drawn by 261108_Wavefom_Comparison (its best_windows / --top selection)"),
        ("GT", f"the reference band-passed {m.GT_BAND_HZ[0]:g}-{m.GT_BAND_HZ[1]:g} Hz at 30 Hz (261108)"),
        ("RRWaveNet", "the stored ensemble output on the GT's time axis, 261108's alignment and 1 Hz low-pass"),
        ("Spectrum", f"|FFT| of the window, mean removed, zero-padded to {FFT_POINTS} points; each curve divided "
                     "by its largest value in the rate band"),
        ("Peak", "largest value in the rate band (CapnoBase 0.05-1.0 Hz, BIDMC / STEAM2 0.1-0.5 Hz); "
                 "RRWaveNet_Peak_Hz = 261108's RRWaveNet_Rate / 60"),
    ], columns=["Item", "Description"])
    with pd.ExcelWriter(path, engine="openpyxl") as writer:
        (frame.round(4) if not frame.empty else frame).to_excel(writer, sheet_name="Peaks", index=False)
        notes.to_excel(writer, sheet_name="Notes", index=False)
    print(f"\nFigures under {fig_root}; table: {path}")


if __name__ == "__main__":
    main()
