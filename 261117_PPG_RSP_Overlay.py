# -*- coding: utf-8 -*-
r"""261117_PPG_RSP_Overlay: the PPG and the respiration reference signal of every recording over each other,
band-passed, per time window, in two figure styles (overlap and separate).

Signals: the raw recordings as stored (not the model input), read with the readers of the script of the
final results, loaded as a module (its data paths are used; the path options override):
  CapnoBase      261007_RRWaveNet_Ensemble_NoEvent.py      PPG = pleth, RSP = capnogram (co2), 300 Hz
  BIDMC          261013_RRWaveNet_Adult_SubjectVal_WD.py   PPG = PLETH, RSP = impedance (ref.resp_sig.imp), 125 Hz
  STEAM2         261013_RRWaveNet_Adult_SubjectVal_WD.py   PPG and RSP (belt) EEGLAB sessions, split at the
                                                           boundary events like the source script (recordings
                                                           rec0, rec1, ... in that order)
Each signal keeps its own sampling rate and time axis (seconds from the start of the recording); the two
signals are shown as recorded: no time shift, no sign change, no resampling.
The subjects with an MAE issue are left out as in the final results (CapnoBase 0115, BIDMC 27, STEAM2
12_YJJ, 22_LMS, 26_CKK; --keep-issue-subjects keeps them).

Filtering (per recording, on the whole recording, then cut into windows)
  PPG  band-pass 0.1-5.0 Hz (--ppg-band-hz)
  RSP  band-pass 0.1-0.5 Hz (--rsp-band-hz)
  Butterworth, 4th order (--filter-order), zero phase (forward-backward, second-order sections).
  Missing samples (NaN) are bridged by linear interpolation for the filter and shown as gaps.

Windows: every length in --window-sec (default 60 and 10 s: both are drawn), on a grid from the start of
the recording (60 s: 0-60, 60-120 s, ...; 10 s: 0-10, 10-20 s, ...), over the time both signals cover;
a window is drawn when both signals have at least 90% valid (not missing) samples in it.
A length of 0 = the whole recording as one window, e.g. --window-sec 60 10 0.

Figures, in two styles, each in its own folder (--styles, default both)
  overlap   PPG (blue) and RSP (black) on one axis, each z-scored in the window (mean 0, SD 1)
  separate  two rows with a shared time axis, each signal with its own y axis in its own units:
            (a) PPG band-passed, (b) RSP band-passed

Outputs (D:\PPG2RSP_RRWaveNet_Inspired\261117_PPG_RSP_Overlay), {style} = overlap or separate
  windows of W s (default W = 60 and 10)
    {style}\{Dataset}\{subject}\w{W}s\{subject}[_rec{k}]_{t0}-{t1}s.png          one figure per window
    {style}\{Dataset}\{subject}\w{W}s\{Dataset}_{subject}_w{W}s_{style}.pdf     every W s window, one per page
  whole recording (length 0)
    {style}\{Dataset}\{subject}[_rec{k}]_whole.png                             one figure per recording
    {style}\{Dataset}\{Dataset}_whole_{style}.pdf                              every recording of the dataset
  261117_PPG_RSP_Overlay.xlsx
    Windows     every drawn window: dataset, subject, recording, start, end, sampling rates, valid fractions,
                figure paths
    Notes       filters and settings

Run: python 261117_PPG_RSP_Overlay.py     (Spyder: runfile(...), no options needed)
Options: --window-sec (default 60 10; 0 = whole recording), --styles (default overlap separate),
--datasets, --subjects CapnoBase:0104 BIDMC:09 ..., --keep-issue-subjects, --ppg-band-hz (default 0.1 5.0),
--rsp-band-hz (default 0.1 0.5), --filter-order (default 4), --dpi (default 150), --output-root,
--script-261007, --script-261013, --root-261007, --root-261013, --capnobase-path, --bidmc-path,
--steam2-ppg-dir, --steam2-rsp-dir
"""

import argparse
import importlib.util
import json
import re
import sys
from pathlib import Path
from typing import Dict, Iterator, List, Optional, Sequence, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.io as sio
from matplotlib.backends.backend_pdf import PdfPages
from scipy.signal import butter, sosfiltfilt

# ---------------------------------------------------------------------------
# Settings
# ---------------------------------------------------------------------------

NAME = "261117_PPG_RSP_Overlay"
HERE = Path(__file__).resolve().parent
WORKSPACE = Path(r"D:\PPG2RSP_RRWaveNet_Inspired")
OUTPUT_ROOT = WORKSPACE / NAME
ROOT_261007 = WORKSPACE / "261007_RRWaveNet_NoEvent_v1"
ROOT_261013 = WORKSPACE / "261013_RRWavNet_STEAM_BIDMC_v1"
SCRIPTS = {"261007": "261007_RRWaveNet_Ensemble_NoEvent.py", "261013": "261013_RRWaveNet_Adult_SubjectVal_WD.py"}
DATASETS = ("CapnoBase", "BIDMC", "STEAM2")
KEYS = {"CapnoBase": "capnobase", "BIDMC": "bidmc", "STEAM2": "steam2"}
SOURCE = {"CapnoBase": "261007", "BIDMC": "261013", "STEAM2": "261013"}
MODALITY = {"CapnoBase": "capnogram", "BIDMC": "impedance", "STEAM2": "belt"}
RSP_UNIT = {"CapnoBase": "CO2, mmHg", "BIDMC": "a.u.", "STEAM2": "a.u."}
EXCLUDED = {"STEAM2": ("22_LMS", "26_CKK", "12_YJJ"), "BIDMC": ("27",), "CapnoBase": ("0115",)}

PPG_BAND_HZ = (0.1, 5.0)
RSP_BAND_HZ = (0.1, 0.5)
FILTER_ORDER = 4
WINDOW_SECS = (60.0, 10.0)  # every length is drawn; 0 = the whole recording
STYLES = ("overlap", "separate")  # each style in its own folder
MIN_VALID_FRACTION = 0.9
DPI = 150

PPG_COLOR, RSP_COLOR = "#2a78d6", "#000000"
INK, MUTED, GRID = "#1f1f1f", "#6b6a66", "#e4e3df"

# ---------------------------------------------------------------------------
# Source scripts
# ---------------------------------------------------------------------------

_MODULES: Dict[str, object] = {}


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
        name = "overlay_src_" + re.sub(r"\W", "_", path.stem)
        spec = importlib.util.spec_from_file_location(name, key)
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module
        spec.loader.exec_module(module)
        _MODULES[key] = module
    return _MODULES[key]


# ---------------------------------------------------------------------------
# Raw recordings: (subject, [recording, ...]); a recording is a dict with
# name, ppg, ppg_fs, rsp, rsp_fs
# ---------------------------------------------------------------------------

def same_subject(a: object, b: object) -> bool:
    a, b = str(a), str(b)
    return a == b or (a.isdigit() and b.isdigit() and int(a) == int(b))


def wanted(subject: str, dataset: str, chosen: Dict[str, List[str]], excluded: Sequence[str]) -> bool:
    if dataset in chosen and not any(same_subject(subject, s) for s in chosen[dataset]):
        return False
    return not any(same_subject(subject, x) for x in excluded)


def capnobase_recordings(mod, chosen, excluded) -> Iterator[Tuple[str, List[dict]]]:
    cfg = mod.DATASET_CONFIGS["capnobase"]
    csv_dir = Path(cfg["data_path"])
    for case_id in mod.capnobase_case_ids(csv_dir):
        if not wanted(case_id, "CapnoBase", chosen, excluded):
            continue
        try:
            case = mod.read_capnobase_case(csv_dir, case_id, cfg["default_fs"], read_labels=False)
        except Exception as exc:  # report and go on with the next case
            print(f"  [CapnoBase {case_id}] skipped: {exc}")
            continue
        yield case_id, [{"name": "", "ppg": case["ppg"], "ppg_fs": float(case["ppg_fs"]),
                         "rsp": case["co2"], "rsp_fs": float(case["co2_fs"])}]


def bidmc_recordings(mod, chosen, excluded) -> Iterator[Tuple[str, List[dict]]]:
    cfg = mod.DATASET_CONFIGS["bidmc"]
    records = sio.loadmat(cfg["data_path"])["data"][0]
    fs = float(cfg["default_fs"])
    for index, record in enumerate(records):
        subject = f"{index + 1:02d}"
        if not wanted(subject, "BIDMC", chosen, excluded):
            continue
        try:
            ppg = np.asarray(record["ppg"][0, 0]["v"]).squeeze()
            rsp = np.asarray(record["ref"][0, 0]["resp_sig"][0, 0]["imp"][0, 0]["v"]).squeeze()
        except Exception as exc:
            print(f"  [BIDMC {subject}] skipped: {exc}")
            continue
        yield subject, [{"name": "", "ppg": ppg, "ppg_fs": fs, "rsp": rsp, "rsp_fs": fs}]


def steam2_recordings(mod, chosen, excluded) -> Iterator[Tuple[str, List[dict]]]:
    cfg = mod.DATASET_CONFIGS["steam2"]
    sessions = mod.steam2_session_files(Path(cfg["ppg_dir"]), Path(cfg["rsp_dir"]))
    for subject, items in sessions.items():
        if not wanted(subject, "STEAM2", chosen, excluded):
            continue
        recordings = []
        try:
            for session, ppg_path, rsp_path in items:
                ppg = mod.read_eeglab_channel(ppg_path, "PPG", cfg["default_fs"])
                rsp = mod.read_eeglab_channel(rsp_path, "RSP", cfg["default_fs"])
                pieces, info = mod.steam2_session_pieces(ppg, rsp)
                if abs(info["Length_Difference_Sec"]) > 1.0:
                    print(f"  [STEAM2 {subject} {session}] PPG {info['PPG_Sec']:.1f} s vs RSP "
                          f"{info['RSP_Sec']:.1f} s; the common part is used")
                for k, (p, r, p_fs, r_fs) in enumerate(pieces):
                    name = session if len(pieces) == 1 else f"{session} piece {k + 1}"
                    recordings.append({"name": name, "ppg": p, "ppg_fs": float(p_fs), "rsp": r, "rsp_fs": float(r_fs)})
        except Exception as exc:
            print(f"  [STEAM2 {subject}] skipped: {exc}")
            continue
        yield subject, recordings


READERS = {"CapnoBase": capnobase_recordings, "BIDMC": bidmc_recordings, "STEAM2": steam2_recordings}

# ---------------------------------------------------------------------------
# Signals
# ---------------------------------------------------------------------------


def bandpass(signal: np.ndarray, fs: float, band: Tuple[float, float], order: int) -> Tuple[np.ndarray, np.ndarray]:
    """Zero-phase Butterworth band-pass of the whole recording; returns (filtered, valid mask).

    Missing samples are bridged by linear interpolation for the filter and set back to NaN after it.
    """
    x = np.asarray(signal, dtype=np.float64).ravel()
    valid = np.isfinite(x)
    if valid.sum() < 2:
        return np.full(x.shape, np.nan), valid
    if not valid.all():
        idx = np.arange(x.size)
        x = np.interp(idx, idx[valid], x[valid])
    high = min(float(band[1]), 0.45 * fs)
    sos = butter(order, [float(band[0]), high], btype="bandpass", fs=fs, output="sos")
    try:
        y = sosfiltfilt(sos, x - np.mean(x))
    except ValueError:  # recording too short for the filter padding
        y = sosfiltfilt(sos, x - np.mean(x), padlen=max(0, x.size - 1))
    y[~valid] = np.nan
    return y, valid


def window_spans(duration: float, window_sec: float) -> List[Tuple[float, float]]:
    if window_sec <= 0:
        return [(0.0, duration)]
    count = int(np.floor(duration / window_sec + 1e-9))
    return [(k * window_sec, (k + 1) * window_sec) for k in range(count)]


def cut(y: np.ndarray, valid: np.ndarray, fs: float, a: float, b: float) -> Tuple[np.ndarray, np.ndarray, float]:
    i0, i1 = int(round(a * fs)), min(int(round(b * fs)), y.size)
    t = np.arange(i0, i1) / fs
    return t, y[i0:i1], float(valid[i0:i1].mean()) if i1 > i0 else 0.0


def zscore(x: np.ndarray) -> np.ndarray:
    sd = np.nanstd(x)
    return (x - np.nanmean(x)) / sd if sd > 0 else x - np.nanmean(x)


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------

def style_axis(ax) -> None:
    ax.grid(True, color=GRID, lw=0.6)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color("#9a9994")
    ax.tick_params(colors=MUTED, labelsize=9)


def figure_width(length: float) -> float:
    return 10.0 if length <= 10 else 16.0 if length <= 60 else 22.0


def draw_overlap(path: Path, pdf, header: Tuple[str, str], t_p, ppg, t_r, rsp, labels: Tuple[str, str], dpi: int) -> None:
    length = max(t_p[-1], t_r[-1]) - min(t_p[0], t_r[0])
    fig, ax = plt.subplots(figsize=(figure_width(length), 4.4))
    ax.plot(t_p, zscore(ppg), color=PPG_COLOR, lw=0.8, label=labels[0], zorder=2)
    ax.plot(t_r, zscore(rsp), color=RSP_COLOR, lw=1.8, label=labels[1], zorder=3)
    style_axis(ax)
    ax.set_xlim(min(t_p[0], t_r[0]), max(t_p[-1], t_r[-1]))
    ax.set_xlabel("Time (s)", fontsize=10, color=INK)
    ax.set_ylabel("z-score", fontsize=10, color=INK)
    fig.legend(loc="upper right", bbox_to_anchor=(0.995, 0.995), ncol=2, frameon=False, fontsize=9.5)
    fig.suptitle(header[0], x=0.01, ha="left", fontsize=11.5, fontweight="bold", color=INK)
    fig.text(0.01, 0.885, header[1], ha="left", fontsize=9, color=MUTED)
    fig.tight_layout(rect=(0, 0, 1, 0.875))
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=dpi)
    pdf.savefig(fig)
    plt.close(fig)


def draw_separate(path: Path, pdf, header: Tuple[str, str], t_p, ppg, t_r, rsp, labels: Tuple[str, str],
                  units: Tuple[str, str], dpi: int) -> None:
    length = max(t_p[-1], t_r[-1]) - min(t_p[0], t_r[0])
    fig, (ax_p, ax_r) = plt.subplots(2, 1, figsize=(figure_width(length), 6.4), sharex=True)
    ax_p.plot(t_p, ppg, color=PPG_COLOR, lw=0.8)
    ax_r.plot(t_r, rsp, color=RSP_COLOR, lw=1.6)
    for ax, letter, label, unit in ((ax_p, "a", labels[0], units[0]), (ax_r, "b", labels[1], units[1])):
        style_axis(ax)
        ax.set_title(f"({letter}) {label}", loc="left", fontsize=10, color=INK)
        ax.set_ylabel(unit, fontsize=9.5, color=INK)
    ax_r.set_xlim(min(t_p[0], t_r[0]), max(t_p[-1], t_r[-1]))
    ax_r.set_xlabel("Time (s)", fontsize=10, color=INK)
    fig.suptitle(header[0], x=0.01, ha="left", fontsize=11.5, fontweight="bold", color=INK)
    fig.text(0.01, 0.93, header[1], ha="left", fontsize=9, color=MUTED)
    fig.tight_layout(rect=(0, 0, 1, 0.935))
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=dpi)
    pdf.savefig(fig)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Run
# ---------------------------------------------------------------------------

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


def dataset_rows(args, dataset: str, chosen: Dict[str, List[str]], out_root: Path) -> List[dict]:
    source = SOURCE[dataset]
    roots = [args.root_261007] if source == "261007" else [args.root_261013]
    script = find_script(SCRIPTS[source], args.script_261007 if source == "261007" else args.script_261013, roots)
    mod = load_script(script)
    for key, field, value in (("capnobase", "data_path", args.capnobase_path), ("bidmc", "data_path", args.bidmc_path),
                              ("steam2", "ppg_dir", args.steam2_ppg_dir), ("steam2", "rsp_dir", args.steam2_rsp_dir)):
        if value:
            mod.DATASET_CONFIGS[key][field] = str(value)
    print(f"\n{dataset}: raw recordings read with {script.name}", flush=True)
    excluded = [] if args.keep_issue_subjects else list(EXCLUDED[dataset])
    lengths = sorted({float(w) for w in args.window_sec}, reverse=True)
    styles = [st for st in STYLES if st in args.styles]
    folders = {st: out_root / st / dataset for st in styles}
    ppg_band, rsp_band = tuple(args.ppg_band_hz), tuple(args.rsp_band_hz)
    labels = (f"PPG ({ppg_band[0]:g}-{ppg_band[1]:g} Hz)",
              f"RSP reference: {MODALITY[dataset]} ({rsp_band[0]:g}-{rsp_band[1]:g} Hz)")
    units = ("PPG (a.u.)", f"{MODALITY[dataset].capitalize()} ({RSP_UNIT[dataset]})")
    rows: List[dict] = []
    whole_pdfs: Dict[str, Optional[PdfPages]] = {st: None for st in styles}
    try:
        for subject, recordings in READERS[dataset](mod, chosen, excluded):
            tag = mod.safe_tag(subject) if hasattr(mod, "safe_tag") else re.sub(r"[^\w.-]", "_", str(subject))
            multi = len(recordings) > 1
            subject_pdfs: Dict[Tuple[str, float], PdfPages] = {}
            drawn = {w: 0 for w in lengths}
            try:
                for rec_index, rec in enumerate(recordings):
                    ppg, ppg_ok = bandpass(rec["ppg"], rec["ppg_fs"], ppg_band, args.filter_order)
                    rsp, rsp_ok = bandpass(rec["rsp"], rec["rsp_fs"], rsp_band, args.filter_order)
                    duration = min(ppg.size / rec["ppg_fs"], rsp.size / rec["rsp_fs"])
                    rec_tag = f"_rec{rec_index}" if multi else ""
                    rec_text = f" · rec{rec_index}" + (f" ({rec['name']})" if rec["name"] else "") if multi else ""
                    for length in lengths:
                        whole = length <= 0
                        for a, b in window_spans(duration, length):
                            t_p, y_p, v_p = cut(ppg, ppg_ok, rec["ppg_fs"], a, b)
                            t_r, y_r, v_r = cut(rsp, rsp_ok, rec["rsp_fs"], a, b)
                            if min(v_p, v_r) < MIN_VALID_FRACTION or t_p.size < 2 or t_r.size < 2:
                                continue
                            span = f"{a:.0f}-{b:.0f} s"
                            header = (f"{dataset} {subject}{rec_text} · "
                                      + (f"whole recording {span}" if whole else f"{length:g} s window {span}"),
                                      f"PPG {ppg_band[0]:g}-{ppg_band[1]:g} Hz ({rec['ppg_fs']:g} Hz) · RSP "
                                      f"({MODALITY[dataset]}) {rsp_band[0]:g}-{rsp_band[1]:g} Hz ({rec['rsp_fs']:g} Hz) · "
                                      f"Butterworth order {args.filter_order}, zero phase, filtered over the whole recording")
                            stem = f"{tag}{rec_tag}_whole" if whole else f"{tag}{rec_tag}_{a:.0f}-{b:.0f}s"
                            figures = {}
                            for st in styles:
                                folder = folders[st] if whole else folders[st] / tag / f"w{length:g}s"
                                if whole:
                                    if whole_pdfs[st] is None:
                                        folders[st].mkdir(parents=True, exist_ok=True)
                                        whole_pdfs[st] = PdfPages(folders[st] / f"{dataset}_whole_{st}.pdf")
                                    pdf = whole_pdfs[st]
                                else:
                                    if (st, length) not in subject_pdfs:
                                        folder.mkdir(parents=True, exist_ok=True)
                                        subject_pdfs[(st, length)] = PdfPages(
                                            folder / f"{dataset}_{tag}_w{length:g}s_{st}.pdf")
                                    pdf = subject_pdfs[(st, length)]
                                png = folder / f"{stem}.png"
                                if st == "overlap":
                                    draw_overlap(png, pdf, header, t_p, y_p, t_r, y_r, labels, args.dpi)
                                else:
                                    draw_separate(png, pdf, header, t_p, y_p, t_r, y_r, labels, units, args.dpi)
                                figures[st] = str(png.relative_to(out_root))
                            drawn[length] += 1
                            row = {"Dataset": dataset, "Subject": subject, "Recording_Index": rec_index,
                                   "Recording": rec["name"], "Window_Sec": 0.0 if whole else length,
                                   "Start_Sec": a, "End_Sec": b, "PPG_FS_Hz": rec["ppg_fs"], "RSP_FS_Hz": rec["rsp_fs"],
                                   "PPG_Valid_Fraction": v_p, "RSP_Valid_Fraction": v_r}
                            for st in styles:
                                row[f"Figure_{st.capitalize()}"] = figures[st]
                            rows.append(row)
            finally:
                for pdf in subject_pdfs.values():
                    pdf.close()
            done = ", ".join(f"{n} {'whole' if w <= 0 else f'{w:g} s'}" for w, n in drawn.items())
            print(f"  {subject}: {len(recordings)} recording(s); {done} drawn", flush=True)
    finally:
        for pdf in whole_pdfs.values():
            if pdf is not None:
                pdf.close()
    return rows


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="PPG and respiration reference over each other, band-passed, per window")
    p.add_argument("--window-sec", nargs="+", type=float, default=list(WINDOW_SECS),
                   help="window lengths in s, every one drawn (default 60 10); 0 = the whole recording")
    p.add_argument("--styles", nargs="+", choices=list(STYLES), default=list(STYLES),
                   help="figure styles, each in its own folder (default overlap separate)")
    p.add_argument("--datasets", nargs="+", choices=list(DATASETS), default=list(DATASETS))
    p.add_argument("--subjects", nargs="+", default=None, help="Dataset:Subject (default: every subject)")
    p.add_argument("--keep-issue-subjects", action="store_true")
    p.add_argument("--ppg-band-hz", nargs=2, type=float, default=list(PPG_BAND_HZ))
    p.add_argument("--rsp-band-hz", nargs=2, type=float, default=list(RSP_BAND_HZ))
    p.add_argument("--filter-order", type=int, default=FILTER_ORDER)
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
    rows: List[dict] = []
    for dataset in datasets:
        rows += dataset_rows(args, dataset, chosen, out_root)
    notes = pd.DataFrame([
        ("Signals", "raw recordings as stored, read with the readers of the source scripts (CapnoBase 261007; "
                    "BIDMC, STEAM2 261013); each signal at its own sampling rate; no time shift, sign change or "
                    "resampling"),
        ("RSP reference", "CapnoBase capnogram (co2), BIDMC impedance (ref.resp_sig.imp), STEAM2 belt (RSP channel)"),
        ("STEAM2 recordings", "sessions split at the EEGLAB boundary events as in the source script; Recording_Index "
                              "in that order (rec0, rec1, ...)"),
        ("PPG filter", f"band-pass {args.ppg_band_hz[0]:g}-{args.ppg_band_hz[1]:g} Hz"),
        ("RSP filter", f"band-pass {args.rsp_band_hz[0]:g}-{args.rsp_band_hz[1]:g} Hz"),
        ("Filter", f"Butterworth order {args.filter_order}, zero phase (forward-backward, second-order sections), "
                   "on the whole recording before the windows are cut; missing samples bridged by linear "
                   "interpolation for the filter and shown as gaps"),
        ("Windows", "lengths " + ", ".join("whole recording" if w <= 0 else f"{w:g} s" for w in args.window_sec)
                    + "; grid from the start of the recording over the time both signals cover; drawn when both "
                    f"signals have at least {MIN_VALID_FRACTION:.0%} valid samples; Window_Sec 0 = whole recording"),
        ("overlap", "PPG (blue) and RSP (black) on one axis, each z-scored in the window"),
        ("separate", "(a) PPG and (b) RSP in two rows with a shared time axis, each in its own units"),
        ("Subjects", "issue subjects left out (CapnoBase 0115, BIDMC 27, STEAM2 12_YJJ, 22_LMS, 26_CKK)"
                     if not args.keep_issue_subjects else "every subject (issue subjects kept)"),
    ], columns=["Item", "Description"])
    path = out_root / f"{NAME}.xlsx"
    with pd.ExcelWriter(path, engine="openpyxl") as writer:
        pd.DataFrame(rows).round(4).to_excel(writer, sheet_name="Windows", index=False)
        notes.to_excel(writer, sheet_name="Notes", index=False)
    print(f"\n{len(rows)} windows drawn; figures under {out_root}; table: {path}")


if __name__ == "__main__":
    main()
