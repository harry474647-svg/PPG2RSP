# -*- coding: utf-8 -*-
r"""261119_PPG_RespMod_Sync: how closely the respiratory components of the PPG - RIIV, RIAV, RIFV and the PPG's own
respiratory band - follow the reference respiratory signal, in every 30 s window of every subject of the
datasets of the final results; the PPG, the reference and the three modulations drawn over each other (overlap)
and one by one (separate), with synchronization measures of every component against the reference.

Data: the signals as the models read them, from the preprocessing of the script of the final results, loaded
as a module (its data paths are used; the path options override):
  CapnoBase      261007_RRWaveNet_Ensemble_NoEvent.py
  BIDMC, STEAM2  261013_RRWaveNet_Adult_SubjectVal_WD.py
  Reference  the continuous reference signal of the scoring (CapnoBase capnogram, BIDMC impedance, STEAM2 belt),
             64 Hz, band-passed with the recording's respiratory band and aligned to the PPG as the source script
             does (CapnoBase: lag and fixed polarity; BIDMC / STEAM2: sign alignment)
  PPG        the continuous model input (64 Hz, as the model reads it)
  GT breaths the breaths of the final results (CapnoBase: detected on the capnogram; BIDMC: annotator 1 (ann1)
             of bidmc_data.mat; STEAM2: the stored breaths of the belt), drawn as vertical guide lines
All subjects are drawn; the MAE issue subjects of the final results (CapnoBase 0115, BIDMC 27, STEAM2 12_YJJ,
22_LMS, 26_CKK) are marked.

Components of the PPG (as 261116_PPG_RIIV_RIAV_RIFV_FFT; Karlen et al. 2013, Charlton et al. 2016)
  PPG resp.  the PPG band-passed with the recording's respiratory band (its respiratory baseline)
  RIIV       respiratory-induced intensity variation: the PPG value at every pulse peak
  RIAV       respiratory-induced amplitude variation: pulse amplitude, peak minus the trough before it
  RIFV       respiratory-induced frequency variation: instantaneous pulse rate, 60 / peak-to-peak interval
  Beats: peaks of the 0.5-5 Hz pulse band; a beat is used when its interval lies in the heart-rate range of the
  profile and within 30% of the median of the 7 intervals around it, and (CapnoBase) outside the expert PPG
  artifact labels. Each modulation: one value per used beat, resampled to 4 Hz by cubic spline, band-passed with
  the recording's respiratory band (3rd-order Butterworth, zero phase). Extraction and filtering run on the whole
  recording; the series are then cut into windows. Between two used beats more than 3 s apart (PPG dropouts,
  rejected beats) the modulations and the PPG respiratory band are left empty (not drawn, not in the measures), so
  no interpolated or filtered curve stands in for missing beats; the raw PPG is drawn as it is.

Synchronization of every component with the reference (both on the 4 Hz grid of the modulations; the reference
and the PPG respiratory band interpolated onto it), per window
  r (lag 0)    Pearson r at zero lag (sign: + in phase, - in antiphase)
  r (best lag) the r of largest magnitude within +-max lag (--max-lag-sec, default 5 s) and its lag; a positive
               lag = the component follows the reference
  PLV          phase locking value |mean(exp(i (phase component - phase reference)))| of the Hilbert phases
               (analytic signals of the whole recording, then cut): 1 = a constant phase relation in the window
               (in phase, antiphase or shifted), 0 = none; with the mean phase difference in degrees
  FFT rate     peak of the |FFT| of the window (detrend, Hamming, zero padding) in the rate band of the scoring
               (BIDMC / STEAM2 6-30, CapnoBase 3-60 /min), of every component and of the reference
  Most synchronized: the component with the highest PLV in the window.

Windows: every 30 s window (--window-sec) on a grid from the start of the recording (0-30, 30-60 s, ...), covered
at least 90% by the modulation series and with at least 10 used beats.

Figures, in two styles, each in its own folder (--styles, default both)
  overlap   (a) the reference, the PPG respiratory band, RIIV, RIAV and RIFV over each other; (b)-(e) the reference
            with each component on its own: (b) the PPG (raw, thin) and its respiratory band, (c) RIIV, (d) RIAV,
            (e) RIFV; every curve z-scored in the window (mean 0, SD 1)
  separate  one row per signal with its own y axis in its own units, a shared time axis: (a) reference, (b) PPG
            (raw) with its respiratory band, (c) RIIV, (d) RIAV, (e) RIFV (band-passed series)
  Both: GT breaths as vertical dashed lines in every panel; beside the panels the window (beats, pulse rate, GT
  rate, reference FFT rate, most synchronized component) and the measures of every component.

Outputs (D:\PPG2RSP_RRWaveNet_Inspired\261119_PPG_RespMod_Sync), {style} = overlap or separate
  {style}\{Dataset}\{subject}\{subject}[_rec{k}]_{t0}-{t1}s.png      one figure per window
  {style}\{Dataset}\{subject}\{Dataset}_{subject}_{style}.pdf        every window of the subject, one per page
  261119_PPG_RespMod_Sync.xlsx
    Windows   every window: beats, pulse rate, GT breaths and rate, reference FFT rate, and per component r (lag 0),
              r (best lag) and its lag, PLV, mean phase difference, FFT rate, most synchronized component
    Subjects  per subject and component: windows, median PLV, median r (lag 0), median |r (best lag)|, median lag,
              windows in which the component is the most synchronized
    Summary   per dataset and component: the same over the windows of the dataset (median and quartiles of PLV)
    Notes

Run: python 261119_PPG_RespMod_Sync.py     (Spyder: runfile(...), no options needed)
Options: --window-sec (default 30), --styles (default overlap separate), --datasets, --subjects CapnoBase:0332
BIDMC:05 ..., --max-lag-sec (default 5), --dpi (default 130), --output-root, --script-261007, --script-261013,
--root-261007, --root-261013, --capnobase-path, --bidmc-path, --steam2-ppg-dir, --steam2-rsp-dir
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
from scipy.interpolate import interp1d
from scipy.signal import butter, detrend, filtfilt, find_peaks, hilbert, sosfiltfilt

# ---------------------------------------------------------------------------
# Settings
# ---------------------------------------------------------------------------

NAME = "261119_PPG_RespMod_Sync"
HERE = Path(__file__).resolve().parent
WORKSPACE = Path(r"D:\PPG2RSP_RRWaveNet_Inspired")
OUTPUT_ROOT = WORKSPACE / NAME
ROOT_261007 = WORKSPACE / "261007_RRWaveNet_NoEvent_v1"
ROOT_261013 = WORKSPACE / "261013_RRWavNet_STEAM_BIDMC_v1"
SCRIPTS = {"261007": "261007_RRWaveNet_Ensemble_NoEvent.py", "261013": "261013_RRWaveNet_Adult_SubjectVal_WD.py"}
WINDOW = "01_w10s_nonoverlap"
DATASETS = ("CapnoBase", "BIDMC", "STEAM2")
KEYS = {"CapnoBase": "capnobase", "BIDMC": "bidmc", "STEAM2": "steam2"}
SOURCE = {"CapnoBase": "261007", "BIDMC": "261013", "STEAM2": "261013"}
MODALITY = {"CapnoBase": "capnogram", "BIDMC": "impedance", "STEAM2": "belt"}
GT_NAME = {"CapnoBase": "breaths detected on the capnogram", "BIDMC": "ann1", "STEAM2": "belt breaths"}
ISSUE_SUBJECTS = {"STEAM2": ("22_LMS", "26_CKK", "12_YJJ"), "BIDMC": ("27",), "CapnoBase": ("0115",)}
MODS = ("RIIV", "RIAV", "RIFV")
COMPONENTS = ("PPG", "RIIV", "RIAV", "RIFV")  # PPG = the PPG's respiratory band
LABEL = {"PPG": "PPG resp. band", "RIIV": "RIIV (intensity)", "RIAV": "RIAV (amplitude)", "RIFV": "RIFV (pulse rate)"}
COLORS = {"REF": "#111111", "PPG_RAW": "#a3a19b", "PPG": "#7b4fb8", "RIIV": "#2a78d6", "RIAV": "#eb6834",
          "RIFV": "#1baf7a", "BREATH": "#b9b6ae"}
INK, MUTED, GRID = "#1f1f1f", "#6b6a66", "#e4e3df"
RESAMPLE_HZ = 4.0
PPG_LOWPASS_HZ = 8.0
PEAK_SEARCH_SEC = 0.15
INTERVAL_TOLERANCE = 0.30
MEDIAN_BEATS = 7
FFT_MIN_POINTS = 4096
WINDOW_SEC = 30.0
MAX_LAG_SEC = 5.0
STYLES = ("overlap", "separate")
MIN_WINDOW_COVER = 0.9
MIN_WINDOW_BEATS = 10
MAX_BEAT_GAP_SEC = 3.0  # modulations left empty between used beats further apart than this
DPI = 130

_MODULES: Dict[str, object] = {}


# ---------------------------------------------------------------------------
# Source scripts and GT breaths
# ---------------------------------------------------------------------------

def find_script(name: str, given: Optional[Path], roots: Sequence[Path]) -> Path:
    """The script of the results: given, written in a run manifest, or found nearby."""
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
        name = "respsync_src_" + re.sub(r"\W", "_", path.stem)
        spec = importlib.util.spec_from_file_location(name, key)
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module
        spec.loader.exec_module(module)
        _MODULES[key] = module
    return _MODULES[key]


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


def gt_breaths(dataset: str, mod, ref: dict, ann1: Optional[np.ndarray]) -> np.ndarray:
    """GT breaths of the final results (sample indices of the 64 Hz reference)."""
    if dataset == "CapnoBase":
        return np.asarray(ref["detected_breaths"], dtype=np.int64)
    if dataset == "BIDMC":
        return ann1_timeline(mod, ann1, ref) if ann1 is not None else np.asarray(ref["breaths"], dtype=np.int64)
    return np.asarray(ref["breaths"], dtype=np.int64)


# ---------------------------------------------------------------------------
# RIIV, RIAV, RIFV (as 261116_PPG_RIIV_RIAV_RIFV_FFT)
# ---------------------------------------------------------------------------

def lowpass(signal: np.ndarray, fs: float, cutoff: float, order: int = 3) -> np.ndarray:
    b, a = butter(order, cutoff / (0.5 * fs), btype="low")
    return filtfilt(b, a, signal)


def extract_modulations(mod, ppg: np.ndarray, fs: float, hr_range_bpm: Tuple[float, float],
                        artifact: Optional[np.ndarray]) -> dict:
    """Beats of the PPG and the RIIV / RIAV / RIFV value of every used beat."""
    ppg, _ = mod.sanitize_signal(np.asarray(ppg, dtype=np.float64))
    pulse = mod.bandpass_filter(ppg, fs, mod.PPG_PULSE_BAND[0], mod.PPG_PULSE_BAND[1])[0]
    smooth = lowpass(ppg, fs, PPG_LOWPASS_HZ)
    spread = float(np.std(pulse))
    if spread < 1e-8:
        return {"n_detected": 0}
    distance = max(1, int(round(60.0 / hr_range_bpm[1] * fs)))
    detected, _ = find_peaks(pulse, distance=distance, prominence=max(1e-8, 0.10 * spread))
    half = int(round(PEAK_SEARCH_SEC * fs))
    peaks = np.array([max(0, p - half) + int(np.argmax(smooth[max(0, p - half):p + half + 1])) for p in detected], int)
    peaks = np.unique(peaks)
    if len(peaks) < 3:
        return {"n_detected": int(len(peaks))}
    intervals = np.diff(peaks) / fs
    troughs = np.array([a + int(np.argmin(smooth[a:b + 1])) for a, b in zip(peaks[:-1], peaks[1:])], int)
    beats, prev = peaks[1:], peaks[:-1]
    ok = (intervals >= 60.0 / hr_range_bpm[1]) & (intervals <= 60.0 / hr_range_bpm[0])
    median = pd.Series(intervals).rolling(MEDIAN_BEATS, center=True, min_periods=1).median().to_numpy()
    ok &= np.abs(intervals - median) <= INTERVAL_TOLERANCE * median
    if artifact is not None:
        artifact = np.asarray(artifact, bool)
        ok &= ~artifact[beats] & ~artifact[prev]
    values = {"RIIV": smooth[beats], "RIAV": smooth[beats] - smooth[troughs], "RIFV": 60.0 / intervals}
    return {"n_detected": int(len(peaks)), "times": beats[ok] / fs, "values": {k: v[ok] for k, v in values.items()},
            "ppg": ppg}


def resample(times: np.ndarray, values: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """One value per beat -> a 4 Hz series between the first and the last beat (cubic spline)."""
    grid = np.arange(times[0], times[-1], 1.0 / RESAMPLE_HZ)
    kind = "cubic" if len(times) >= 4 else "linear"
    return grid, interp1d(times, values, kind=kind)(grid)


def respiratory_band(x: np.ndarray, fs: float, band: Tuple[float, float], order: int = 3) -> np.ndarray:
    """Zero-phase Butterworth band-pass (second-order sections)."""
    high = min(float(band[1]), 0.45 * fs)
    sos = butter(order, [float(band[0]) / (0.5 * fs), high / (0.5 * fs)], btype="band", output="sos")
    try:
        return sosfiltfilt(sos, x)
    except ValueError:  # series too short for the filter
        return x - np.mean(x)


def spectrum_rate(x: np.ndarray, fs: float, band_bpm: Tuple[float, float]) -> float:
    """Rate (/min) of the largest |FFT| in the rate band: detrend, Hamming window, zero padding."""
    x = np.asarray(x, dtype=np.float64)
    x = x[np.isfinite(x)]
    if len(x) < 8 or float(np.std(x)) < 1e-12:
        return math.nan
    n_fft = max(FFT_MIN_POINTS, 1 << int(math.ceil(math.log2(len(x)))))
    mag = np.abs(np.fft.rfft(detrend(x) * np.hamming(len(x)), n=n_fft))
    rate = np.fft.rfftfreq(n_fft, d=1.0 / fs) * 60.0
    keep = (rate >= band_bpm[0]) & (rate <= band_bpm[1])
    return float(rate[keep][np.argmax(mag[keep])]) if keep.any() else math.nan


# ---------------------------------------------------------------------------
# Synchronization
# ---------------------------------------------------------------------------

def pearson(a: np.ndarray, b: np.ndarray) -> float:
    ok = np.isfinite(a) & np.isfinite(b)
    if ok.sum() < 8 or np.std(a[ok]) < 1e-12 or np.std(b[ok]) < 1e-12:
        return math.nan
    return float(np.corrcoef(a[ok], b[ok])[0, 1])


def analytic_phase(x: np.ndarray) -> np.ndarray:
    """Hilbert phase of a band-passed series (NaN where the series is missing)."""
    ok = np.isfinite(x)
    out = np.full(len(x), np.nan)
    if ok.sum() >= 8:
        out[ok] = np.angle(hilbert(np.where(ok, x, 0.0)))[ok]
    return out


def sync_measures(comp: np.ndarray, ref: np.ndarray, phase_c: np.ndarray, phase_r: np.ndarray, idx: np.ndarray,
                  max_lag: int) -> dict:
    """r (lag 0), r of largest magnitude within +-max_lag samples and its lag (component after the reference: +),
    PLV and mean phase difference, over the grid indices idx of a window."""
    r0 = pearson(comp[idx], ref[idx])
    best_r, best_k = math.nan, 0
    for k in range(-max_lag, max_lag + 1):
        j = idx + k
        keep = (j >= 0) & (j < len(comp))
        r = pearson(comp[j[keep]], ref[idx[keep]])
        if np.isfinite(r) and (not np.isfinite(best_r) or abs(r) > abs(best_r)):
            best_r, best_k = r, k
    d = phase_c[idx] - phase_r[idx]
    d = d[np.isfinite(d)]
    if len(d) >= 8:
        z = np.mean(np.exp(1j * d))
        plv, phase = float(np.abs(z)), float(np.degrees(np.angle(z)))
    else:
        plv, phase = math.nan, math.nan
    return {"r0": r0, "r_best": best_r, "lag": best_k / RESAMPLE_HZ, "plv": plv, "phase": phase}


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------

def style_axis(ax) -> None:
    ax.grid(True, color=GRID, lw=0.6)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.tick_params(labelsize=8)


def zscore(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, float)
    ok = np.isfinite(x)
    if ok.sum() < 3 or np.std(x[ok]) < 1e-12:
        return np.full_like(x, np.nan)
    return (x - np.nanmean(x)) / np.nanstd(x)


def fmt(v: float, spec: str = ".2f") -> str:
    return format(v, spec) if v is not None and np.isfinite(v) else "n/a"


def measure_text(name: str, m: dict, rate: float, ref_rate: float) -> str:
    return (f"{LABEL[name]} vs reference\n"
            f"r (lag 0)     {fmt(m['r0'], '+.2f')}\n"
            f"r (best lag)  {fmt(m['r_best'], '+.2f')} at {fmt(m['lag'], '+.2f')} s\n"
            f"PLV           {fmt(m['plv'])}  (phase {fmt(m['phase'], '+.0f')} deg)\n"
            f"FFT rate      {fmt(rate, '.1f')} vs {fmt(ref_rate, '.1f')} /min")


def draw_window(style: str, targets: Sequence[Tuple[Path, PdfPages]], header: Tuple[str, str], data: dict,
                texts: Dict[str, str], dpi: int) -> None:
    """One window: overlap (all components, then the reference with each) or separate (one row per signal)."""
    width, side, left, panel, gap, top_in, bottom_in = 13.0, 3.7, 0.8, 1.45, 0.42, 1.2, 1.05
    n = 5
    height = top_in + bottom_in + n * panel + (n - 1) * gap
    fig = plt.figure(figsize=(width, height))
    plot_w = (width - left - side - 0.2) / width
    x_text = (width - side + 0.08) / width
    t0, t1 = data["span"]
    axes = []
    for k in range(n):
        bottom = bottom_in + (n - 1 - k) * (panel + gap)
        ax = fig.add_axes([left / width, bottom / height, plot_w, panel / height], sharex=axes[0] if axes else None)
        axes.append(ax)
        style_axis(ax)
        for b in data["breaths"]:
            ax.axvline(b, color=COLORS["BREATH"], lw=0.9, ls=(0, (3, 3)), zorder=1)
        key = ("ALL", "PPG", "RIIV", "RIAV", "RIFV")[k]
        tr, ref = data["ref"]
        if style == "overlap":
            if key == "ALL":
                ax.plot(tr, zscore(ref), color=COLORS["REF"], lw=2.0, zorder=4)
                for c in COMPONENTS:
                    tc, vc = data[c]
                    ax.plot(tc, zscore(vc), color=COLORS[c], lw=1.3, zorder=3)
                title = "(a) Reference and every component of the PPG (z-scored)"
            else:
                if key == "PPG":
                    tp, vp = data["PPG_RAW"]
                    ax.plot(tp, zscore(vp), color=COLORS["PPG_RAW"], lw=0.7, zorder=2)
                tc, vc = data[key]
                ax.plot(tr, zscore(ref), color=COLORS["REF"], lw=1.8, zorder=3)
                ax.plot(tc, zscore(vc), color=COLORS[key], lw=1.6, zorder=4)
                title = (f"({'abcde'[k]}) Reference vs {LABEL[key]}"
                         + (" (raw PPG thin grey)" if key == "PPG" else "") + " (z-scored)")
            ax.set_ylabel("z", fontsize=8)
        else:
            if key == "ALL":
                ax.plot(tr, ref, color=COLORS["REF"], lw=1.6, zorder=3)
                title = f"(a) Reference ({data['modality']}, band-passed, aligned as in the scoring)"
                ax.set_ylabel("reference", fontsize=8)
            elif key == "PPG":
                tp, vp = data["PPG_RAW"]
                tc, vc = data["PPG"]
                ax.plot(tp, vp - np.nanmean(vp), color=COLORS["PPG_RAW"], lw=0.7, zorder=2)
                ax.plot(tc, vc, color=COLORS["PPG"], lw=1.6, zorder=3)
                title = "(b) PPG (model input, mean removed, grey) and its respiratory band (purple)"
                ax.set_ylabel("PPG", fontsize=8)
            else:
                tc, vc = data[key]
                ax.plot(tc, vc, color=COLORS[key], lw=1.6, zorder=3)
                title = f"({'abcde'[k]}) {LABEL[key]}, band-passed"
                ax.set_ylabel({"RIIV": "PPG units", "RIAV": "PPG units", "RIFV": "beats/min"}[key], fontsize=8)
        ax.set_xlim(t0, t1)
        ax.set_title(title, fontsize=9.3, loc="left", pad=3)
        if k < n - 1:
            ax.tick_params(labelbottom=False)
        fig.text(x_text, (bottom + panel) / height, texts[key], family="monospace", fontsize=7.4, color=INK, va="top")
    axes[-1].set_xlabel("Time in the recording (s)", fontsize=8.5)
    handles = [Line2D([], [], color=COLORS["REF"], lw=1.8), Line2D([], [], color=COLORS["PPG_RAW"], lw=1.0)]
    labels = [f"Reference ({data['modality']})", "PPG (raw)"]
    for c in COMPONENTS:
        handles.append(Line2D([], [], color=COLORS[c], lw=1.6))
        labels.append(LABEL[c])
    handles.append(Line2D([], [], color=COLORS["BREATH"], lw=0.9, ls=(0, (3, 3))))
    labels.append("GT breath")
    fig.legend(handles, labels, loc="lower left", bbox_to_anchor=(left / width, 0.1 / height), ncol=len(labels),
               frameon=False, fontsize=7.8, handlelength=1.8, columnspacing=1.4)
    fig.text(left / width, 1.0 - 0.2 / height, header[0], fontsize=10.5, color=INK, va="top")
    fig.text(left / width, 1.0 - 0.5 / height, header[1], fontsize=7.6, color=MUTED, va="top", linespacing=1.4)
    for path, pdf in targets:
        path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(path, dpi=dpi)
        pdf.savefig(fig)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Datasets
# ---------------------------------------------------------------------------

def same_subject(a: object, b: object) -> bool:
    a, b = str(a).strip(), str(b).strip()
    return a == b or (a.isdigit() and b.isdigit() and int(a) == int(b))


def parse_subjects(items: Optional[Sequence[str]]) -> Dict[str, List[str]]:
    chosen: Dict[str, List[str]] = {}
    for item in items or []:
        dataset, _, subject = str(item).partition(":")
        if dataset not in DATASETS or not subject:
            raise SystemExit(f"--subjects {item}: write Dataset:Subject with Dataset in {', '.join(DATASETS)}")
        chosen.setdefault(dataset, []).append(subject)
    return chosen


def window_spans(t_first: float, t_last: float, window_sec: float) -> List[Tuple[float, float]]:
    """Windows on a grid from the start (0, W, 2W s ...) covered at least 90% by the series."""
    spans = []
    for k in range(int(math.floor(t_last / window_sec)) + 1):
        a, b = k * window_sec, (k + 1) * window_sec
        if min(b, t_last) - max(a, t_first) >= MIN_WINDOW_COVER * window_sec:
            spans.append((a, b))
    return spans


def dataset_rows(args, dataset: str, chosen: Dict[str, List[str]], out_root: Path) -> List[dict]:
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
    fs = float(mod.TARGET_FS)
    max_lag = int(round(args.max_lag_sec * RESAMPLE_HZ))
    styles = [st for st in STYLES if st in args.styles]
    rows = []
    for subject in sorted(data):
        if dataset in chosen and not any(same_subject(subject, s) for s in chosen[dataset]):
            continue
        issue = any(same_subject(subject, x) for x in ISSUE_SUBJECTS[dataset])
        tag = mod.safe_tag(subject)
        folders = {st: out_root / st / dataset / tag for st in styles}
        pdfs: Dict[str, PdfPages] = {}
        drawn = 0
        try:
            references = data[subject]["references"]
            multi = len(references) > 1
            for segment_id, ref in sorted(references.items()):
                profile = ref["profile"]
                beats = extract_modulations(mod, ref["model_input"], fs, tuple(profile["ppg_hr_range_bpm"]),
                                            ref.get("ppg_artifact_mask"))
                if "times" not in beats or len(beats["times"]) < MIN_WINDOW_BEATS:
                    print(f"  {subject} recording {segment_id}: too few usable beats "
                          f"({beats.get('n_detected', 0)} detected)", flush=True)
                    continue
                resp_band = tuple(float(v) for v in ref["band_hz"])
                rate_band = tuple(float(v) for v in profile["rr_band_bpm"])
                # modulations on the 4 Hz grid; the reference and the PPG respiratory band interpolated onto it
                comp: Dict[str, np.ndarray] = {}
                grid = None
                for k in MODS:
                    grid, values = resample(beats["times"], beats["values"][k])
                    comp[k] = respiratory_band(values, RESAMPLE_HZ, resp_band)
                # no modulation between two used beats more than MAX_BEAT_GAP_SEC apart (dropouts, rejected beats)
                after = np.clip(np.searchsorted(beats["times"], grid, side="right"), 1, len(beats["times"]) - 1)
                gap = beats["times"][after] - beats["times"][after - 1]
                for k in MODS:
                    comp[k] = np.where(gap > MAX_BEAT_GAP_SEC, np.nan, comp[k])
                ref_signal = np.asarray(ref["signal"], dtype=np.float64)
                ppg_raw = np.asarray(beats["ppg"], dtype=np.float64)
                ppg_resp = respiratory_band(ppg_raw, fs, resp_band)
                # the PPG respiratory band left empty where the modulations are (gaps between used beats, outside them)
                t_ppg = np.arange(len(ppg_resp)) / fs
                after = np.clip(np.searchsorted(beats["times"], t_ppg, side="right"), 1, len(beats["times"]) - 1)
                outside = (t_ppg < beats["times"][0]) | (t_ppg > beats["times"][-1])
                gap64 = beats["times"][after] - beats["times"][after - 1]
                ppg_resp = np.where(outside | (gap64 > MAX_BEAT_GAP_SEC), np.nan, ppg_resp)
                t64 = np.arange(len(ref_signal)) / fs
                ok_ref = np.isfinite(ref_signal)
                ref_grid = np.interp(grid, t64[ok_ref], ref_signal[ok_ref], left=np.nan, right=np.nan)
                comp["PPG"] = np.interp(grid, np.arange(len(ppg_resp)) / fs, ppg_resp, left=np.nan, right=np.nan)
                phase_ref = analytic_phase(ref_grid)
                phases = {c: analytic_phase(comp[c]) for c in COMPONENTS}
                breaths_all = gt_breaths(dataset, mod, ref, ann1_all.get(subject)) / fs
                times = beats["times"]
                rec = f" · recording {segment_id}" if multi else ""
                for a, b in window_spans(float(times[0]), float(times[-1]), args.window_sec):
                    in_window = (times >= a) & (times < b)
                    if int(in_window.sum()) < MIN_WINDOW_BEATS:
                        continue
                    idx = np.flatnonzero((grid >= a) & (grid < b))
                    if len(idx) < 8:
                        continue
                    i0, i1 = int(round(a * fs)), min(len(ref_signal), int(round(b * fs)))
                    ref_rate = spectrum_rate(ref_signal[i0:i1], fs, rate_band)
                    measures, rates = {}, {}
                    for c in COMPONENTS:
                        measures[c] = sync_measures(comp[c], ref_grid, phases[c], phase_ref, idx, max_lag)
                        rates[c] = spectrum_rate(comp[c][idx], RESAMPLE_HZ, rate_band)
                    plvs = {c: measures[c]["plv"] for c in COMPONENTS if np.isfinite(measures[c]["plv"])}
                    best = max(plvs, key=plvs.get) if plvs else ""
                    breaths = breaths_all[(breaths_all >= a) & (breaths_all < b)]
                    gt_rate = len(breaths) * 60.0 / (b - a)
                    pulse_rate = float(np.mean(beats["values"]["RIFV"][in_window]))
                    span = f"{a:.0f}-{b:.0f} s"
                    header = (f"{dataset} {subject}{rec} · {args.window_sec:g} s window {span}: synchronization of the "
                              f"PPG's respiratory components with the reference"
                              + (" · MAE issue subject" if issue else ""),
                              f"PPG = model input (64 Hz); RIIV / RIAV / RIFV from the beats of the 0.5-5 Hz pulse band, "
                              f"{RESAMPLE_HZ:g} Hz, band-passed {resp_band[0]:.2f}-{resp_band[1]:.2f} Hz like the "
                              f"reference; PLV = phase locking value (Hilbert); best lag within +-{args.max_lag_sec:g} s "
                              f"(+: component after the reference)\nGT breaths ({GT_NAME[dataset]}) as dashed lines")
                    texts = {"ALL": (f"window {span}\n"
                                     f"beats {int(in_window.sum())}, pulse {pulse_rate:.0f} /min\n"
                                     f"GT breaths {len(breaths)} ({gt_rate:.1f} /min)\n"
                                     f"reference FFT {fmt(ref_rate, '.1f')} /min\n"
                                     f"most synchronized (PLV): {LABEL.get(best, 'n/a')}\n"
                                     + "PLV  " + "  ".join(f"{c} {fmt(measures[c]['plv'])}" for c in ('PPG', 'RIIV'))
                                     + "\n     " + "  ".join(f"{c} {fmt(measures[c]['plv'])}" for c in ('RIAV', 'RIFV')))}
                    for c in COMPONENTS:
                        texts[c] = measure_text(c, measures[c], rates[c], ref_rate)
                    ppg_k = slice(i0, min(len(ppg_raw), i1))
                    payload = {"span": (a, b), "modality": MODALITY[dataset], "breaths": breaths,
                               "ref": (t64[i0:i1], ref_signal[i0:i1]),
                               "PPG_RAW": (np.arange(ppg_k.start, ppg_k.stop) / fs, ppg_raw[ppg_k]),
                               "PPG": (np.arange(ppg_k.start, ppg_k.stop) / fs, ppg_resp[ppg_k])}
                    for k in MODS:
                        payload[k] = (grid[idx], comp[k][idx])
                    stem = f"{tag}{f'_rec{segment_id}' if multi else ''}_{a:03.0f}-{b:03.0f}s"
                    for st in styles:
                        if st not in pdfs:
                            folders[st].mkdir(parents=True, exist_ok=True)
                            pdfs[st] = PdfPages(folders[st] / f"{dataset}_{tag}_{st}.pdf")
                        draw_window(st, [(folders[st] / f"{stem}.png", pdfs[st])], header, payload, texts, args.dpi)
                    drawn += 1
                    row = {"Dataset": dataset, "Subject": subject, "Issue_Subject": issue, "Segment_ID": int(segment_id),
                           "Start_Sec": a, "End_Sec": b, "Beats_Used": int(in_window.sum()),
                           "Mean_Pulse_Rate_BPM": pulse_rate, "Filter_Band_Low_Hz": resp_band[0],
                           "Filter_Band_High_Hz": resp_band[1], "GT_Breaths": int(len(breaths)),
                           "GT_Rate_BPM": gt_rate, "Ref_FFT_Rate_BPM": ref_rate, "Most_Synchronized_PLV": best}
                    for c in COMPONENTS:
                        m = measures[c]
                        row[f"{c}_r_Lag0"] = m["r0"]
                        row[f"{c}_r_BestLag"] = m["r_best"]
                        row[f"{c}_BestLag_Sec"] = m["lag"]
                        row[f"{c}_PLV"] = m["plv"]
                        row[f"{c}_Phase_Deg"] = m["phase"]
                        row[f"{c}_FFT_Rate_BPM"] = rates[c]
                        row[f"{c}_Abs_Rate_Diff_to_Ref_BPM"] = abs(rates[c] - ref_rate)
                    for st in styles:
                        row[f"Figure_{st.capitalize()}"] = str((folders[st] / f"{stem}.png").relative_to(out_root))
                    rows.append(row)
        finally:
            for pdf in pdfs.values():
                pdf.close()
        print(f"  {subject}: {drawn} windows of {args.window_sec:g} s drawn{' [MAE issue]' if issue else ''}", flush=True)
    del data
    return rows


def summarize(table: pd.DataFrame, keys: Sequence[str]) -> pd.DataFrame:
    rows = []
    if table.empty:
        return pd.DataFrame()
    for group, g in table.groupby(list(keys), sort=False):
        group = group if isinstance(group, tuple) else (group,)
        for c in COMPONENTS:
            plv = g[f"{c}_PLV"].dropna()
            row = dict(zip(keys, group))
            row.update({"Component": LABEL[c], "Windows": int(len(g)),
                        "PLV_Median": float(plv.median()) if len(plv) else math.nan,
                        "PLV_Q1": float(plv.quantile(0.25)) if len(plv) else math.nan,
                        "PLV_Q3": float(plv.quantile(0.75)) if len(plv) else math.nan,
                        "r_Lag0_Median": float(g[f"{c}_r_Lag0"].median()),
                        "Abs_r_BestLag_Median": float(g[f"{c}_r_BestLag"].abs().median()),
                        "BestLag_Sec_Median": float(g[f"{c}_BestLag_Sec"].median()),
                        "Abs_Rate_Diff_to_Ref_BPM_Median": float(g[f"{c}_Abs_Rate_Diff_to_Ref_BPM"].median()),
                        "Windows_Most_Synchronized": int((g["Most_Synchronized_PLV"] == c).sum())})
            rows.append(row)
    return pd.DataFrame(rows)


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Synchronization of the PPG's respiratory components (PPG respiratory "
                                            "band, RIIV, RIAV, RIFV) with the reference, every 30 s window")
    p.add_argument("--window-sec", type=float, default=WINDOW_SEC, help="window length in s (default 30)")
    p.add_argument("--styles", nargs="+", choices=list(STYLES), default=list(STYLES),
                   help="figure styles, each in its own folder (default overlap separate)")
    p.add_argument("--datasets", nargs="+", choices=list(DATASETS), default=list(DATASETS))
    p.add_argument("--subjects", nargs="+", default=None, help="Dataset:Subject (default: every subject)")
    p.add_argument("--max-lag-sec", type=float, default=MAX_LAG_SEC, help="lag range of r (best lag), default 5 s")
    p.add_argument("--dpi", type=int, default=DPI)
    p.add_argument("--root-261007", type=Path, default=ROOT_261007)
    p.add_argument("--root-261013", type=Path, default=ROOT_261013)
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
    datasets = [d for d in args.datasets if not chosen or d in chosen]
    out_root = Path(args.output_root)
    out_root.mkdir(parents=True, exist_ok=True)
    rows = []
    for dataset in datasets:
        rows += dataset_rows(args, dataset, chosen, out_root)
    table = pd.DataFrame(rows)
    subjects = summarize(table, ("Dataset", "Subject", "Issue_Subject"))
    summary = summarize(table, ("Dataset",))
    if not summary.empty:
        print("\nMedian PLV per dataset (most synchronized windows):")
        for r in summary.itertuples():
            print(f"  {r.Dataset:<9} {r.Component:<18} PLV {r.PLV_Median:.2f} [{r.PLV_Q1:.2f}-{r.PLV_Q3:.2f}], "
                  f"r lag 0 {r.r_Lag0_Median:+.2f}, |r best lag| {r.Abs_r_BestLag_Median:.2f} at "
                  f"{r.BestLag_Sec_Median:+.2f} s; most synchronized in {r.Windows_Most_Synchronized} of {r.Windows}")
    notes = pd.DataFrame([
        ("Reference", "continuous reference signal of the scoring (CapnoBase capnogram, BIDMC impedance, STEAM2 belt), "
                      "64 Hz, band-passed and aligned to the PPG as the source script does"),
        ("PPG", "continuous model input (64 Hz); PPG resp. band = the PPG band-passed with the recording's respiratory "
                "band (3rd-order Butterworth, zero phase)"),
        ("RIIV / RIAV / RIFV", "PPG value at every pulse peak / peak minus the trough before it / 60 / peak-to-peak "
                               f"interval; beats of the 0.5-5 Hz pulse band, interval within {INTERVAL_TOLERANCE:.0%} "
                               f"of the median of {MEDIAN_BEATS}, outside expert PPG artifacts; {RESAMPLE_HZ:g} Hz cubic "
                               "spline, band-passed with the respiratory band (as 261116_PPG_RIIV_RIAV_RIFV_FFT)"),
        ("Grid", f"every measure on the {RESAMPLE_HZ:g} Hz grid of the modulations; the reference and the PPG "
                 "respiratory band linearly interpolated onto it"),
        ("r", f"Pearson r at lag 0 and the r of largest magnitude within +-{args.max_lag_sec:g} s (BestLag_Sec > 0: the "
              "component follows the reference)"),
        ("PLV", "phase locking value |mean(exp(i dphi))| of the Hilbert phases (analytic signals of the whole recording, "
                "cut into windows); 1 = constant phase relation, 0 = none; Phase_Deg = mean phase difference "
                "(component - reference)"),
        ("FFT rate", "peak of the |FFT| of the window (detrend, Hamming, zero padding >= 4096 points) in the rate band "
                     "of the scoring (BIDMC / STEAM2 6-30, CapnoBase 3-60 /min)"),
        ("Windows", f"{args.window_sec:g} s windows on a grid from the start of the recording, covered at least "
                    f"{MIN_WINDOW_COVER:.0%} by the modulation series, at least {MIN_WINDOW_BEATS} used beats"),
        ("GT breaths", "drawn as guide lines: CapnoBase breaths detected on the capnogram, BIDMC ann1, STEAM2 belt "
                       "breaths; GT_Rate_BPM = breaths in the window x 60 / its length"),
        ("Subjects", "every subject; MAE issue subjects (CapnoBase 0115, BIDMC 27, STEAM2 12_YJJ, 22_LMS, 26_CKK) "
                     "marked in Issue_Subject"),
        ("References", "Karlen W et al. IEEE TBME 2013;60(7):1946-53; Charlton PH et al. Physiol Meas 2016;37(4):610-26; "
                       "Lachaux JP et al. Hum Brain Mapp 1999;8(4):194-208 (phase locking value)"),
    ], columns=["Item", "Description"])
    path = out_root / f"{NAME}.xlsx"
    with pd.ExcelWriter(path, engine="openpyxl") as writer:
        table.round(4).to_excel(writer, sheet_name="Windows", index=False)
        subjects.round(4).to_excel(writer, sheet_name="Subjects", index=False)
        summary.round(4).to_excel(writer, sheet_name="Summary", index=False)
        notes.to_excel(writer, sheet_name="Notes", index=False)
    print(f"\nFigures under {out_root}; table: {path}")


if __name__ == "__main__":
    main()
