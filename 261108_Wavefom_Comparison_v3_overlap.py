# -*- coding: utf-8 -*-
r"""261108_Wavefom_Comparison_v3_overlap: the respiratory waveforms of RRWaveNet, the deep corr-encoder (paper version)
and RespDiff against the ground truth (GT) as in 261108_Wavefom_Comparison, but all three models and the GT on ONE
axis (one figure per window), in 30 s windows covering every second of every subject (the subjects that had an MAE
issue left out). Nothing is trained: the stored outputs are read.

What is the same as 261108_Wavefom_Comparison (its code, unchanged)
  Models (stored outputs)
    RRWaveNet          ensemble output of its run: CapnoBase 261007 (261007_RRWaveNet_NoEvent_v1), BIDMC and STEAM2
                       261013 (261013_RRWavNet_STEAM_BIDMC_v1)
    Deep Corr-Encoder  261107_DeepCorrencoder_PaperVersion_v1 (the paper's model, LOSOCV):
                       {correncoder-root}\{Dataset}\Predictions\{subject}_outputs.npz (fused 30 Hz output)
    RespDiff           Baseline outputs (low-band PPG input) of 260519_RespDiff_AllDatasets_v3: CapnoBase first from
                       261108_RespDiff_CapnoBase_Baseline_v3, BIDMC and STEAM2 from RespDiff_AllDatasets_v3; its 32 s
                       windows (16 s apart) put back on the recording's time axis (matched through the stored
                       reference, |r| >= 0.98; overlaps joined by a weighted mean of the z-scored windows; windows v3
                       left out stay gaps; RespDiff_Placement sheet)
  GT: the reference at 30 Hz (CapnoBase capnogram, BIDMC impedance, STEAM2 belt), band-passed 0.1-0.5 Hz (3rd-order
  Butterworth, zero phase) for the figures and every r; the GT rate from the breaths (CapnoBase expert labels, BIDMC
  ann1, STEAM2 the GT cycles).
  Alignment: no model shifted in time; a model inverted once per recording when its r with the band-passed GT is
  negative (the same rule for all); RRWaveNet with its training sign / lag undone and low-pass 1 Hz.
  Measures in a window: AE = |FFT rate of the waveform - GT rate| (breaths/min), r = Pearson r with the band-passed
  GT, score = r - 0.1 x min(AE, 5), Advantage = RRWaveNet's score - the better score of the two others.
  Subjects left out (MAE issue): STEAM2 22_LMS, 26_CKK, 12_YJJ; BIDMC 27; CapnoBase 0115 (--exclude-*).

What is new
  Windows: 30 s windows covering EVERY second of every recording with a model output (--sizes for other lengths),
  not one chosen window per subject; every window is drawn, also one without a model output in it (the GT alone).
    grid   0-30, 30-60, ... s, one after the other
    end    when the recording does not end on a window border: one more 30 s window ending at the end of the
           recording (it overlaps the window before, so that the last seconds are drawn too)
    whole  a recording shorter than 30 s: one window over all of it (the x axis still 30 s long)
  Scored: RRWaveNet and another model in the recording, every model complete in the window and >= 3 GT breaths -
  the measures of 261108; the scored grid windows make the subject means (as the "subject mean over non-overlapping
  windows" of 261108; end and whole windows are not in them). Not scored: a gap in a model (e.g. a RespDiff window
  v3 left out), < 3 GT breaths, or a recording without RRWaveNet / another model: drawn as it is (a gap = no line
  there); AE and r of every model complete in the window (AE needs >= 3 GT breaths); the reason in
  Not_Scored_Because.
  Figure (one per window): the GT and the three models overlapped on one axis, the same seconds of the recording;
  every waveform min-max scaled in the window; GT black, RRWaveNet red (the red of 261108 a little darker, #f64636 ->
  #dc2a1c), Deep Corr-Encoder blue and RespDiff green (the colours of 261108, #2d93e1 and #24a085), the line widths of
  261108 (GT 1.5, models 1.7); the models drawn over the GT (RRWaveNet on top). Beside the axis: the legend (GT with
  its rate) and a table of every model's AE and r in the window and its subject mean (MAE, PCC).

Outputs (D:\PPG2RSP_RRWaveNet_Inspired\261108_Wavefom_Comparison\v3_overlap)
  {Dataset}\30s\{subject}\{nn}_{subject}_{t0}-{t1}s.png               every window in time order (nn = window number of
                                                                     the subject; STEAM2 with several recordings:
                                                                     {nn}_{recording}_{t0}-{t1}s.png)
  {Dataset}\30s\{subject}\{Dataset}_{subject}_30s_overlap.pdf          every window of the subject, one page each
  261108_Waveform_v3_overlap.xlsx
    Windows            every drawn window: kind (grid / end / whole), scored or not (why), GT rate, AE / r / rate of
                       every model, coverage of every model, Advantage, figure
    Skipped            recordings without any model output (and subjects without a data file): not drawn
    Subject_Summary    per subject and size: scored windows, mean MAE / PCC of every model
    RespDiff_Placement how the RespDiff windows of every subject were placed
    Notes
Run:
  python 261108_Wavefom_Comparison_v3_overlap.py          (Spyder: runfile(...), no options needed)
Options: --sizes, --datasets, --subjects, --root-261007, --root-261013, --correncoder-root, --respdiff-root,
  --respdiff-capnobase-root, --respdiff-run-tag, --scripts-dir (CapnoBase breath labels:
  261007_RRWaveNet_Ensemble_NoEvent.py), --capnobase-path, --bidmc-path, --steam2-ppg-dir, --steam2-rsp-dir,
  --exclude-steam2, --exclude-bidmc, --exclude-capnobase, --rrwavenet-lowpass-hz, --output-root
"""

import argparse
import importlib.util
import json
import math
import re
import sys
from dataclasses import dataclass
from fractions import Fraction
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
from scipy.signal import butter, detrend, resample, resample_poly, savgol_filter, sosfiltfilt

# ---------------------------------------------------------------------------
# Folders and settings
# ---------------------------------------------------------------------------

WORKSPACE = Path(r"D:\PPG2RSP_RRWaveNet_Inspired")
PROJECT_ROOT = WORKSPACE / "261108_Wavefom_Comparison"
FS = 64  # model and analysis sampling rate of the RRWaveNet runs (Hz)
WINDOW = "01_w10s_nonoverlap"
DATASETS = ("CapnoBase", "BIDMC", "STEAM2")
FIGURE_ORDER = ("STEAM2", "BIDMC", "CapnoBase")
INK, MUTED, GRID = "#1f1f1f", "#6b6a66", "#e4e3df"

# RRWaveNet runs in use: run -> results folder, how the fold's model folder is found, title; and their datasets
RUNS = {
    "261007": {
        "root": WORKSPACE / "261007_RRWaveNet_NoEvent_v1",
        "selected": False,
        "title": "261007 (full model, temporal validation, ensemble of 3)",
    },
    "261013": {
        "root": WORKSPACE / "261013_RRWavNet_STEAM_BIDMC_v1",
        "selected": True,
        "title": "261013 (subject-wise validation, WD per fold, ensemble of 3)",
    },
}
RUN_DATASETS = {"261007": ("CapnoBase",), "261013": ("BIDMC", "STEAM2")}
EXCLUDED = {"STEAM2": ("22_LMS", "26_CKK", "12_YJJ"), "BIDMC": ("27",), "CapnoBase": ("0115",)}
GT_BAND_HZ = (0.1, 0.5)  # GT band-pass for the figures and r (RespDiff v3's bandpass_filter)
GT_BAND_ORDER = 3
OUTPUT_FOLDER = "v3_overlap"  # the results: PROJECT_ROOT\v3_overlap

# Figures: the GT black; RRWaveNet the red of 261108 a little darker (#f64636 -> #dc2a1c), Deep Corr-Encoder and
# RespDiff the blue and green of 261108 (#2d93e1, #24a085), lighter than the red; the three pass the categorical
# palette checks (lightness band, chroma, colour-blind and normal-vision separation, contrast >= 3:1 on white); all
# solid lines, the widths of 261108 (GT 1.5, models 1.7) as there
GT_LINE = "#000000"
FIG_MODELS = ("RRWaveNet", "Deep corr-encoder", "RespDiff")
FIG_NAME = {"RRWaveNet": "RRWaveNet", "Deep corr-encoder": "Deep Corr-Encoder", "RespDiff": "RespDiff"}
FIG_COLOR = {"RRWaveNet": "#dc2a1c", "Deep corr-encoder": "#2d93e1", "RespDiff": "#24a085"}
GT_WIDTH, MODEL_WIDTH = 1.5, 1.7
SIZES = (30,)  # 30 s windows covering every second (--sizes)
DRAW_ORDER = ("RespDiff", "Deep corr-encoder", "RRWaveNet")  # drawn over the GT in this order (RRWaveNet on top)


# ===========================================================================
# From 261106_RRWaveNet_Performance_Boxplots_v2 (readers, alignment, window measures; unchanged)
# ===========================================================================

# RRWaveNet run results (fold files)

def read_json(path: Path) -> Optional[dict]:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


def fold_files(root: Path, dataset: str, selected: bool) -> List[dict]:
    """Every fold of a dataset: subject tag, the model folder of the fold and
    its stored per-minute table and ensemble outputs."""
    folds = []
    if selected:
        for path in sorted((root / "WD_Selection" / WINDOW / dataset).glob("*_selection.json")):
            tag = path.name[: -len("_selection.json")]
            variant = (read_json(path) or {}).get("Selected_Variant")
            folds.append({"Tag": tag, "Variant": variant or ""})
    else:
        summaries = root / "full" / WINDOW / dataset / "Fold_Summaries"
        for path in sorted(summaries.glob("*_ensemble_RR60s_windows.csv")):
            if path.name.endswith("_selected_ensemble_RR60s_windows.csv"):
                continue
            folds.append({"Tag": path.name[: -len("_ensemble_RR60s_windows.csv")], "Variant": "full"})
    for fold in folds:
        condition = root / fold["Variant"] / WINDOW / dataset
        fold["Minutes_File"] = condition / "Fold_Summaries" / f"{fold['Tag']}_ensemble_RR60s_windows.csv"
        fold["Outputs_File"] = condition / "Ensemble_Predictions" / f"{fold['Tag']}_ensemble_outputs.npz"
    return folds














def same_subject(a: object, b: object) -> bool:
    """'0115' == '115' == '0115'; '27' == '27'; other names must match exactly."""
    a, b = str(a).strip(), str(b).strip()
    return a == b or (a.isdigit() and b.isdigit() and int(a) == int(b))





# ---------------------------------------------------------------------------
# Waveform overlays: GT, RespDiff, deep corr-encoder and RRWaveNet
# (readers of 261104_Waveform_Overlay, window measures of
# 261105_model_discussion_analysis)
# ---------------------------------------------------------------------------

SCRIPT_DIR = Path(__file__).resolve().parent
# CapnoBase reader with the expert breath labels: read_capnobase_case of the
# 261007 script (or of the 261105 script, which keeps it unchanged)
CAPNOBASE_READERS = (("261007", "261007_RRWaveNet_Ensemble_NoEvent.py"),
                     ("261105", "261105_RRWaveNet_Unified_Optimized.py"))
# RespDiff: the Baseline outputs of 260519_RespDiff_AllDatasets_v3 (all datasets) and of
# 261108_RespDiff_CapnoBase_Baseline_v3 (CapnoBase, the current run; used first)
RESPDIFF_ROOT = Path(r"D:\STEAM\Multimodal_prediction\PPG-RSP\Results\Waveform\Hilbert_OpenDataset"
                     r"\RespDiff_AllDatasets_v3")
RESPDIFF_CAPNOBASE_ROOT = Path(r"D:\STEAM\Multimodal_prediction\PPG-RSP\Results\Waveform\Hilbert_OpenDataset"
                               r"\RespDiff_CapnoBase_Baseline_261108")
RESPDIFF_RUN_TAG = "preset_balanced_e90_ss80_ve2_tr2_va2_te1_mtr240_mva160"
CORRENCODER_ROOT = WORKSPACE / "261107_DeepCorrencoder_PaperVersion_v1"  # the paper version (LOSOCV)
RESPDIFF_MODE = "baseline"  # only the Baseline outputs of RespDiff v3 are used
RESPDIFF_FOLDERS = {"CapnoBase": "CapnoBase", "BIDMC": "BIDMC", "STEAM2": "STEAM"}  # v3's dataset folders
RESPDIFF_FS = {"capnobase": 125, "bidmc": 125, "steam2": 200}  # v3's target_fs
RESPDIFF_WINDOW_SEC, RESPDIFF_STRIDE_SEC = 32.0, 16.0  # v3's WINDOW_SEC / STRIDE_SEC
RESPDIFF_MATCH_MIN_R = 0.98  # a stored window is placed when its reference matches a recording window this well
RRWAVENET_LOWPASS_HZ = 1.0
FS30 = 30  # time axis of the overlays: the 30 Hz of the corr-encoder and the GT
MODELS = ("RespDiff", "Deep corr-encoder", "RRWaveNet")
OTHERS = MODELS[:2]
SHORT = {"RespDiff": "RespDiff", "Deep corr-encoder": "CorrEncoder", "RRWaveNet": "RRWaveNet"}
MODEL_COLORS = {"RespDiff": "#2a78d6", "Deep corr-encoder": "#eb6834", "RRWaveNet": "#1baf7a"}
DATASET_KEYS = {"CapnoBase": "capnobase", "BIDMC": "bidmc", "STEAM2": "steam2"}
RATE_BAND_BPM = {"capnobase": (3.0, 60.0), "bidmc": (6.0, 30.0), "steam2": (6.0, 30.0)}
WINDOW_SIZES = (30, 60)
STEP_SEC = 5.0
MIN_RUN_SEC = 30.0
MIN_BREATHS = 3
AE_WEIGHT, AE_CAP = 0.1, 5.0
RATE_OK_BPM = 1.0
DOUBLE, HALF = (1.75, 2.25), (0.4, 0.6)

m07 = None  # module holding read_capnobase_case, loaded by load_capnobase_reader


# ---------------------------------------------------------------------------
# Copied unchanged from 261104_Waveform_Overlay (data / result readers, Track,
# filters; its 30 Hz FS is FS30 here) and 261030_Waveform_Morphology_Analysis
# (breathing cycle detector), so that no other script is needed for the
# overlays except the CapnoBase reader (expert breath labels) of 261007
# ---------------------------------------------------------------------------

# Data files of the GT (as in 261104 and the three scripts).
DATASET_CONFIGS = {
    "capnobase": {
        "label": "CapnoBase",
        "data_path": r"D:\STEAM\Multimodal_prediction\PPG-RSP\Opendata\Capnobase\data\csv",
        "default_fs": 300,
    },
    "bidmc": {
        "label": "BIDMC",
        "data_path": (
            r"D:\STEAM\Multimodal_prediction\PPG-RSP\Preprocessed"
            r"\bidmc-ppg-and-respiration-dataset-1.0.0"
            r"\bidmc-ppg-and-respiration-dataset-1.0.0\bidmc_data.mat"
        ),
        "default_fs": 125,
    },
    "steam2": {
        "label": "STEAM2",
        "ppg_dir": r"D:\STEAM\Multimodal_prediction\PPG-RSP\Preprocessed\PPG\2back_raw_v2",
        "rsp_dir": r"D:\STEAM\Multimodal_prediction\PPG-RSP\Preprocessed\RSP\2back_raw_v2",
        "default_fs": 200,
    },
}
MODALITIES = {"capnobase": "capnogram", "bidmc": "impedance", "steam2": "belt"}
CAPNOBASE_FILE_SUFFIXES = (".csv", ".tab")
STEAM2_CHANNELS = {"PPG": ("PPG", "PLETH", "BVP"), "RSP": ("RSP", "RESP", "RESPIRATION", "BELT")}
STEAM2_SESSION_PATTERN = re.compile(r"^(?P<subject>.+?)[_\- ]*session[_\- ]*(?P<session>\d+)$", re.IGNORECASE)
SATURATION_CONFIG = {"saturation_min_sec": 0.1, "saturation_margin_sec": 0.1}
RRWAVENET_LOWPASS_ORDER = 8
RRWAVENET_FS = FS  # 64 Hz of the RRWaveNet outputs
CYCLE_CONFIG = {
    "highpass_hz": 0.03,
    "lowpass_hz": {"capnogram": 2.0, "impedance": 1.0, "belt": 2.0},
    "filter_order": 4,
    "range_window_sec": 20.0,
    "hysteresis": (0.35, 0.65),
    "min_cycle_sec": {"capnogram": 0.6, "impedance": 1.5, "belt": 1.5},
    "max_cycle_sec": 20.0,
    "min_phase_sec": 0.2,
    "match_tolerance": 0.5,
    "polarity_min_r": -0.5,
    "amplitude_scale": "zscore",
    "capnogram_transition_fraction": 0.25,
    "slope_eps": 1e-6,
    "sigma_eps": 1e-9,
    "saturation_min_sec": 0.1,
    "saturation_margin_sec": 0.1,
}


def _unwrap_mat(value):
    arr = np.asarray(value)
    while arr.dtype == object and arr.size == 1:
        arr = np.asarray(arr.item())
    return arr


def _first_mat_payload(mat: dict):
    keys = [key for key in mat if not key.startswith("_")]
    if not keys:
        raise KeyError("No non-private MATLAB variable found")
    payload = _unwrap_mat(mat[keys[0]])
    if payload.ndim >= 2 and payload.shape[0] == 1 and payload.shape[1] == 1:
        payload = payload[0, 0]
    return payload


def _extract_signal_and_fs(
    payload,
    default_fs: int,
) -> Tuple[np.ndarray, float]:
    signal = np.asarray(
        _unwrap_mat(payload["data"]),
        dtype=np.float64,
    ).reshape(-1)
    try:
        fs = float(np.asarray(_unwrap_mat(payload["srate"])).squeeze())
    except Exception:
        fs = float(default_fs)
    return signal.astype(np.float32), fs


def capnobase_case_ids(csv_dir: Path) -> List[str]:
    return sorted(
        {
            path.name.split("_")[0]
            for suffix in CAPNOBASE_FILE_SUFFIXES
            for path in csv_dir.glob(f"*_8min_signal{suffix}")
        }
    )


def steam2_session_key(stem: str) -> Tuple[str, int]:
    """(subject, session number) from a file stem such as 13_LSB_Session2."""
    match = STEAM2_SESSION_PATTERN.match(stem)
    if match is None:
        return stem, 0
    return match.group("subject"), int(match.group("session"))


def _as_list(value) -> list:
    if value is None:
        return []
    if isinstance(value, np.ndarray):
        return list(value.ravel()) if value.size else []
    return [value]


def read_eeglab_channel(path: Path, kind: str, default_fs: float) -> dict:
    """One channel of an EEGLAB .mat file.

    The channel is chosen by its label (STEAM2_CHANNELS[kind]); a file with
    one channel is accepted when it has no label, and rejected when its label
    names the other signal (swapped files). EEGLAB "boundary" events inside
    the recording mark discontinuities (removed data); their positions are
    returned so that the session is split there. Files without an EEG
    structure fall back to the generic reader (data and srate fields).
    """
    mat = sio.loadmat(path, squeeze_me=True, struct_as_record=False)
    names = [key for key in mat if not key.startswith("_")]
    payload = mat["EEG"] if "EEG" in mat else (mat[names[0]] if names else None)
    if payload is None or not hasattr(payload, "data"):
        signal, fs = _extract_signal_and_fs(_first_mat_payload(sio.loadmat(path)), default_fs)
        return {"signal": signal, "fs": fs, "boundaries": [], "label": "", "channels": 1}
    data = np.asarray(payload.data, dtype=np.float64)
    data = data.reshape(1, -1) if data.ndim == 1 else data
    labels = [str(getattr(item, "labels", "")).strip() for item in _as_list(getattr(payload, "chanlocs", None))]
    wanted = {name.upper() for name in STEAM2_CHANNELS[kind]}
    other = {name.upper() for key, names in STEAM2_CHANNELS.items() if key != kind for name in names}
    index = next((i for i, label in enumerate(labels) if label.upper() in wanted), None)
    if index is None:
        if data.shape[0] != 1:
            raise RuntimeError(f"{path.name}: no {kind} channel among {labels}")
        if labels and labels[0].upper() in other:
            raise RuntimeError(f"{path.name}: channel '{labels[0]}' found where {kind} was expected")
        index = 0
    signal = data[index].astype(np.float32)
    try:
        fs = float(np.asarray(payload.srate).squeeze())
    except Exception:
        fs = float(default_fs)
    boundaries = []
    for event in _as_list(getattr(payload, "event", None)):
        if str(getattr(event, "type", "")).strip().lower() != "boundary":
            continue
        try:
            latency = float(getattr(event, "latency"))
        except (TypeError, ValueError):
            continue
        # EEGLAB latencies are 1-based and a boundary at k.5 lies between
        # samples k and k + 1: the next piece starts at 0-based index k. The
        # markers at the start (0.5, "New Segment" at 1) and at the end are
        # not discontinuities inside the data.
        position = int(np.ceil(latency - 1.0))
        if 0 < position < signal.size:
            boundaries.append(position)
    label = labels[index] if index < len(labels) else ""
    return {"signal": signal, "fs": fs, "boundaries": sorted(set(boundaries)), "label": label,
            "channels": int(data.shape[0])}


def steam2_session_files(ppg_root: Path, rsp_root: Path) -> Dict[str, List[Tuple[str, Path, Path]]]:
    """Subject -> [(session name, PPG file, RSP file)], sessions in order.

    Subject folders (PPG\\<subject>\\<session>.mat and RSP\\<subject>\\...) are
    used when present; otherwise the files of one flat folder are grouped by
    the subject part of their name (<subject>_Session<k>.mat). Every session
    of a person stays in one subject, so a person is never in the training
    and the test data of the same fold.
    """
    sessions: Dict[str, List[Tuple[str, Path, Path]]] = {}
    subject_dirs = sorted(path for path in ppg_root.iterdir() if path.is_dir())
    if subject_dirs:
        for subject_dir in subject_dirs:
            rsp_dir = rsp_root / subject_dir.name
            if not rsp_dir.is_dir():
                print(f"[STEAM2 {subject_dir.name}] no RSP folder, skipped")
                continue
            # PPG and RSP files of a session have the same name; files without
            # a partner are reported and left out (never paired by position).
            rsp_files = {path.stem: path for path in rsp_dir.glob("*.mat")}
            for ppg_path in sorted(subject_dir.glob("*.mat")):
                if ppg_path.stem not in rsp_files:
                    print(f"[STEAM2 {subject_dir.name}] {ppg_path.name}: no RSP file of the same name, left out")
                    continue
                sessions.setdefault(subject_dir.name, []).append(
                    (ppg_path.stem, ppg_path, rsp_files[ppg_path.stem])
                )
            for stem in sorted(set(rsp_files) - {path.stem for path in subject_dir.glob("*.mat")}):
                print(f"[STEAM2 {subject_dir.name}] {stem}.mat: no PPG file of the same name, left out")
    else:
        rsp_files = {path.stem: path for path in rsp_root.glob("*.mat")}
        for ppg_path in sorted(ppg_root.glob("*.mat")):
            if ppg_path.stem not in rsp_files:
                print(f"[STEAM2 {ppg_path.name}] no RSP file of the same name, skipped")
                continue
            subject, _ = steam2_session_key(ppg_path.stem)
            sessions.setdefault(subject, []).append((ppg_path.stem, ppg_path, rsp_files[ppg_path.stem]))
    for subject in sessions:
        sessions[subject].sort(key=lambda item: (steam2_session_key(item[0])[1], item[0]))
    return sessions


def steam2_session_pieces(ppg: dict, rsp: dict) -> Tuple[List[Tuple[np.ndarray, np.ndarray, float, float]], dict]:
    """The (PPG, RSP, fs, fs) pieces of one session, split at the boundary
    events of either file, and a description of the session."""
    ppg_sec = ppg["signal"].size / ppg["fs"]
    rsp_sec = rsp["signal"].size / rsp["fs"]
    cuts = sorted({b / ppg["fs"] for b in ppg["boundaries"]} | {b / rsp["fs"] for b in rsp["boundaries"]})
    edges = [0.0] + [cut for cut in cuts if 0.0 < cut < min(ppg_sec, rsp_sec)] + [min(ppg_sec, rsp_sec)]
    pieces = []
    for start, end in zip(edges[:-1], edges[1:]):
        p = ppg["signal"][int(round(start * ppg["fs"])):int(round(end * ppg["fs"]))]
        r = rsp["signal"][int(round(start * rsp["fs"])):int(round(end * rsp["fs"]))]
        if p.size and r.size:
            pieces.append((p, r, ppg["fs"], rsp["fs"]))
    info = {
        "PPG_Label": ppg["label"], "RSP_Label": rsp["label"],
        "PPG_FS": ppg["fs"], "RSP_FS": rsp["fs"],
        "PPG_Sec": ppg_sec, "RSP_Sec": rsp_sec, "Length_Difference_Sec": ppg_sec - rsp_sec,
        "Internal_Boundaries": len(edges) - 2, "Pieces": len(pieces),
        "Piece_Sec": " ".join(f"{end - start:.1f}" for start, end in zip(edges[:-1], edges[1:])),
    }
    return pieces, info


def fill_missing(signal: np.ndarray) -> Tuple[np.ndarray, int]:
    """Missing samples (NaN) interpolated linearly; returns the count."""
    x = np.asarray(signal, dtype=np.float64).reshape(-1)
    bad = ~np.isfinite(x)
    if bad.any() and not bad.all():
        index = np.arange(x.size)
        x = x.copy()
        x[bad] = np.interp(index[bad], index[~bad], x[~bad])
    return x, int(bad.sum())


def to_fs(signal: np.ndarray, fs: float) -> np.ndarray:
    """Resampled to 30 Hz (polyphase with anti-aliasing, as the deep
    corr-encoder), missing samples interpolated first."""
    x, _ = fill_missing(signal)
    ratio = Fraction(FS30 / float(fs)).limit_denominator(1000)
    return resample_poly(x, ratio.numerator, ratio.denominator) if ratio != 1 else x


def saturated_samples(signal: np.ndarray, fs: float, modality: str, config: dict) -> np.ndarray:
    """Clipped samples of a recording at its own rate: runs of at least 0.1 s
    exactly at the largest or smallest value of the recording (the capnogram
    only at the largest: its inspiratory baseline at the floor is no
    clipping)."""
    x = np.asarray(signal, dtype=float).reshape(-1)
    mask = np.zeros(x.size, dtype=bool)
    finite = np.isfinite(x)
    if finite.sum() < 2:
        return mask
    low, high = np.min(x[finite]), np.max(x[finite])
    tolerance = 1e-9 * max(high - low, 1e-12)
    for level in ([high] if modality == "capnogram" else [high, low]):
        at_level = finite & (np.abs(x - level) <= tolerance)
        for s, e in finite_runs(at_level, max(2, int(round(config["saturation_min_sec"] * fs)))):
            mask[s:e] = True
    return mask


def to_fs_mask(mask: np.ndarray, fs: float, length: int, margin_sec: float) -> np.ndarray:
    """A mask of samples at fs on the 30 Hz grid, widened by margin_sec."""
    out = np.zeros(length, dtype=bool)
    index = np.round(np.flatnonzero(mask) * FS30 / float(fs)).astype(np.int64)
    out[index[(index >= 0) & (index < length)]] = True
    width = int(round(margin_sec * FS30))
    if width and out.any():
        out = np.convolve(out.astype(float), np.ones(2 * width + 1), mode="same") > 0.5
    return out


@dataclass
class Track:
    """A waveform on the GT time axis: sample i lies at offset_sec + i / fs
    seconds of the GT; NaN = no output."""

    values: np.ndarray
    fs: float
    offset_sec: float = 0.0

    def at(self, times: np.ndarray) -> np.ndarray:
        index = np.round((np.asarray(times) - self.offset_sec) * self.fs).astype(np.int64)
        inside = (index >= 0) & (index < len(self.values))
        out = np.full(index.shape, np.nan)
        out[inside] = self.values[index[inside]]
        return out


def load_correncoder(root: Path, label: str) -> Dict[str, List[dict]]:
    """Subject -> recordings (index k, name, output, stored reference)."""
    folder = root / label
    subjects = {}
    for path in sorted((folder / "Predictions").glob("*_outputs.npz")):
        subject = path.name[: -len("_outputs.npz")]
        if not (folder / "Fold_Results" / f"{subject}_summary.json").exists():
            continue
        with np.load(path, allow_pickle=False) as data:
            names = [str(name) for name in data["recording_names"]]
            subjects[subject] = [
                {"index": k, "name": name,
                 "output": data[f"recording{k}_output"].astype(np.float64),
                 "reference": data[f"recording{k}_reference"].astype(np.float64)}
                for k, name in enumerate(names) if f"recording{k}_output" in data.files
            ]
    return subjects


# --- RespDiff v3 (Baseline): v3's preprocessing, copied unchanged (sanitize_signal, bandpass_filter,
# valid_odd_window, smooth_signal, maybe_resample, resample_signal_poly, instance_normalize,
# postprocess_true_window, _extract_signal_and_fs), to find where every stored window lies in the recording

def v3_sanitize_signal(sig):
    sig = np.asarray(sig, dtype=np.float64).squeeze()
    if sig.ndim != 1:
        sig = sig.reshape(-1)
    if sig.size == 0:
        return sig.astype(np.float64)
    finite_mask = np.isfinite(sig)
    if not finite_mask.any():
        return np.zeros_like(sig, dtype=np.float64)
    fill_value = float(np.median(sig[finite_mask]))
    sig = sig.copy()
    sig[~finite_mask] = fill_value
    return sig.astype(np.float64)


def v3_bandpass_filter(sig, fs, low=0.1, high=0.5, order=3):
    sig = v3_sanitize_signal(sig)
    if sig.size < max(32, order * 6 + 1):
        return sig.astype(np.float32)
    nyq = 0.5 * float(fs)
    low = max(float(low), 1e-5)
    high = min(float(high), nyq * 0.99)
    if not np.isfinite(low) or not np.isfinite(high) or high <= low:
        return sig.astype(np.float32)
    try:
        sos = butter(order, [low / nyq, high / nyq], btype="band", output="sos")
        return sosfiltfilt(sos, sig).astype(np.float32)
    except Exception:
        return sig.astype(np.float32)


def v3_valid_odd_window(length, desired):
    if length < 5:
        return None
    win = int(max(5, desired))
    if win % 2 == 0:
        win += 1
    max_win = length if length % 2 == 1 else length - 1
    win = min(win, max_win)
    if win < 5:
        return None
    return win


def v3_smooth_signal(sig, fs, smooth_sec=0.20, polyorder=3):
    sig = v3_sanitize_signal(sig).astype(np.float32)
    if sig.size < 5:
        return sig
    desired = int(round(float(smooth_sec) * float(fs)))
    win = v3_valid_odd_window(sig.size, desired)
    if win is None:
        return sig
    use_poly = min(int(polyorder), win - 1)
    if use_poly < 1:
        return sig
    try:
        return savgol_filter(sig, window_length=win, polyorder=use_poly, mode="interp").astype(np.float32)
    except Exception:
        return sig


def v3_maybe_resample(sig, orig_fs, target_fs):
    sig = v3_sanitize_signal(sig).astype(np.float32)
    if int(round(orig_fs)) == int(round(target_fs)):
        return sig
    target_len = int(round(len(sig) * float(target_fs) / float(orig_fs)))
    target_len = max(target_len, 1)
    return resample(sig, target_len).astype(np.float32)


def v3_resample_signal_poly(sig, src_fs, dst_fs):
    sig = v3_sanitize_signal(sig)
    src_fs = int(round(float(src_fs)))
    dst_fs = int(round(float(dst_fs)))
    if src_fs <= 0 or dst_fs <= 0:
        raise ValueError(f"Invalid sampling rate: src={src_fs}, dst={dst_fs}")
    if sig.size == 0 or src_fs == dst_fs:
        return sig.astype(np.float32)
    factor = math.gcd(src_fs, dst_fs)
    return resample_poly(sig, dst_fs // factor, src_fs // factor).astype(np.float32)


def v3_instance_normalize(x):
    x = np.asarray(x, dtype=np.float32)
    std = float(np.std(x))
    return ((x - np.mean(x)) / (std if std >= 1e-6 else 1.0)).astype(np.float32)


def v3_postprocess_true_window(sig, fs):
    sig = v3_bandpass_filter(sig, fs, 0.10, 0.50)
    sig = v3_smooth_signal(sig, fs, smooth_sec=0.10, polyorder=3)
    return sig.astype(np.float32)


def v3_signal_and_fs(path: Path) -> Tuple[np.ndarray, float]:
    """v3's STEAM reader: the first variable of the MAT file, its data and srate."""
    payload = _first_mat_payload(sio.loadmat(path))
    signal = v3_sanitize_signal(_unwrap_mat(payload["data"]))
    srate = float(np.asarray(_unwrap_mat(payload["srate"])).squeeze())
    if not np.isfinite(srate) or srate <= 0:
        raise ValueError(f"Invalid srate: {srate}")
    return signal, srate


_BIDMC_MAT: Dict[str, dict] = {}
RESPDIFF_PLACEMENT: List[dict] = []  # per subject: how its stored windows were placed (sheet RespDiff_Placement)


def v3_sources(dataset_key: str, subject: str) -> List[Tuple[str, np.ndarray, np.ndarray]]:
    """v3's (source name, PPG, target) of a subject at v3's rate, in v3's order: CapnoBase CSV (resampled to
    125 Hz), BIDMC (125 Hz), STEAM: every PPG / RSP file pair of the subject folder (common stems, sorted;
    resampled to 200 Hz)."""
    cfg = DATASET_CONFIGS[dataset_key]
    fs = RESPDIFF_FS[dataset_key]
    if dataset_key == "capnobase":
        csv_dir = Path(cfg["data_path"])
        signal = pd.read_csv(csv_dir / f"{subject}_8min_signal.csv")
        param = pd.read_csv(csv_dir / f"{subject}_8min_param.csv")
        orig_fs = int(param["samplingrate_pleth"].values[0])
        return [(subject, v3_maybe_resample(signal["pleth_y"].values.astype(np.float32), orig_fs, fs),
                 v3_maybe_resample(signal["co2_y"].values.astype(np.float32), orig_fs, fs))]
    if dataset_key == "bidmc":
        path = str(cfg["data_path"])
        if path not in _BIDMC_MAT:
            _BIDMC_MAT[path] = sio.loadmat(path)
        record = _BIDMC_MAT[path]["data"][0][int(subject) - 1]
        ppg = np.array(record["ppg"][0, 0]["v"]).squeeze()
        imp = np.array(record["ref"][0, 0]["resp_sig"][0, 0]["imp"][0, 0]["v"]).squeeze()
        return [(subject, ppg, imp)]
    ppg_dir, rsp_dir = Path(cfg["ppg_dir"]) / subject, Path(cfg["rsp_dir"]) / subject
    common = sorted({p.stem for p in ppg_dir.glob("*.mat")} & {p.stem for p in rsp_dir.glob("*.mat")})
    out = []
    for stem in common:
        ppg, ppg_fs = v3_signal_and_fs(ppg_dir / f"{stem}.mat")
        rsp, rsp_fs = v3_signal_and_fs(rsp_dir / f"{stem}.mat")
        out.append((stem, v3_resample_signal_poly(ppg, ppg_fs, fs), v3_resample_signal_poly(rsp, rsp_fs, fs)))
    return out


def v3_candidate_windows(dataset_key: str, subject: str, fs: int) -> Tuple[List[dict], Dict[str, int]]:
    """Every 32 s window (16 s apart) of v3's target of the subject, before v3's quality check: source,
    start (s) and the reference as v3 stored it for a test window (instance-normalized, post-processed)."""
    candidates, lengths = [], {}
    win, stride = int(round(RESPDIFF_WINDOW_SEC * fs)), int(round(RESPDIFF_STRIDE_SEC * fs))
    for name, ppg, target in v3_sources(dataset_key, subject):
        ppg = v3_sanitize_signal(ppg).astype(np.float32)
        target = v3_sanitize_signal(target).astype(np.float32)
        m = min(len(ppg), len(target))
        if m < win:
            continue
        target_f = v3_bandpass_filter(target[:m], fs, 0.1, 0.5)
        lengths[name] = m
        for start in range(0, m - win + 1, stride):
            reference = v3_postprocess_true_window(v3_instance_normalize(target_f[start:start + win]), fs)
            candidates.append({"source": name, "start": start, "reference": reference})
    return candidates, lengths


def place_windows(stored_references: np.ndarray, candidates: List[dict]) -> List[Tuple[Optional[int], float]]:
    """Every stored window: the recording window whose reference it is (|r| >= RESPDIFF_MATCH_MIN_R, in
    order; None when none matches)."""
    placed, first = [], 0
    for reference in stored_references:
        best_k, best_r = None, 0.0
        for k in range(first, len(candidates)):
            if len(candidates[k]["reference"]) != len(reference):
                continue
            r = abs(pearson(reference, candidates[k]["reference"]))
            if np.isfinite(r) and r > best_r:
                best_k, best_r = k, r
                if r > 0.9999:
                    break
        if best_k is not None and best_r >= RESPDIFF_MATCH_MIN_R:
            placed.append((best_k, best_r))
            first = best_k + 1
        else:
            placed.append((None, best_r))
    return placed


def respdiff_run_dir(base: Path, run_tag: str) -> Optional[Path]:
    """{base}\\{run tag} when it has Predictions, else the run folder with the most prediction files."""
    if (base / run_tag / "Predictions").is_dir():
        return base / run_tag
    runs = sorted(p for p in base.glob("*") if (p / "Predictions").is_dir()) if base.is_dir() else []
    if not runs:
        return None
    best = max(runs, key=lambda p: len(list((p / "Predictions").glob("*_predictions.npz"))))
    print(f"  RespDiff: {base / run_tag} not found; using {best}")
    return best


def respdiff_files(args: argparse.Namespace, dataset: str) -> Dict[str, Path]:
    """Subject -> RespDiff v3 prediction file with Baseline outputs: RespDiff_AllDatasets_v3, then (CapnoBase)
    the 261108 run, which wins for a subject in both."""
    roots = [Path(args.respdiff_root)]
    if dataset == "CapnoBase":
        roots.append(Path(args.respdiff_capnobase_root))
    files = {}
    for root in roots:
        run_dir = respdiff_run_dir(root / RESPDIFF_FOLDERS[dataset], args.respdiff_run_tag)
        if run_dir is None:
            continue
        for path in sorted((run_dir / "Predictions").glob("*_predictions.npz")):
            files[path.name[: -len("_predictions.npz")]] = path
    return files


def respdiff_tracks(dataset: str, subject: str, path: Path) -> Dict[str, Track]:
    """Recording name (as read_recordings names it) -> RespDiff v3 Baseline track on its time axis."""
    key = DATASET_KEYS[dataset]
    info = {"Dataset": dataset, "Subject": subject, "File": str(path)}
    try:
        with np.load(path, allow_pickle=False) as data:
            if f"{RESPDIFF_MODE}_processed" not in data.files or "true" not in data.files:
                raise KeyError(f"{RESPDIFF_MODE}_processed / true not in the file")
            fs = int(data["fs"]) if "fs" in data.files else RESPDIFF_FS[key]
            outputs = data[f"{RESPDIFF_MODE}_processed"].astype(np.float64)
            stored = data["true"].astype(np.float64)
        candidates, lengths = v3_candidate_windows(key, subject, fs)
    except Exception as exc:
        RESPDIFF_PLACEMENT.append({**info, "Note": f"not used: {type(exc).__name__}: {exc}"})
        print(f"  RespDiff {dataset} {subject}: not used ({exc})")
        return {}
    placed = place_windows(stored, candidates)
    rs = np.array([r for k, r in placed if k is not None])
    n_win = outputs.shape[1] if outputs.ndim == 2 else 0
    weight = 0.05 + 0.95 * (1.0 - np.abs(np.linspace(-1.0, 1.0, n_win)))
    sums = {name: np.zeros(n) for name, n in lengths.items()}
    weights = {name: np.zeros(n) for name, n in lengths.items()}
    for (k, _), output in zip(placed, outputs):
        if k is None:
            continue
        c = candidates[k]
        z = (output - np.mean(output)) / (np.std(output) + 1e-8)
        end = min(c["start"] + len(z), lengths[c["source"]])
        sums[c["source"]][c["start"]:end] += (weight * z)[: end - c["start"]]
        weights[c["source"]][c["start"]:end] += weight[: end - c["start"]]
    sessions = {}
    for name in lengths:
        if weights[name].any():
            with np.errstate(invalid="ignore", divide="ignore"):
                sessions[name] = np.where(weights[name] > 0, sums[name] / weights[name], np.nan)
    tracks = {}
    if key == "steam2":
        # read_recordings cuts every STEAM2 session at its EEGLAB boundary events: {session}_piece{k}
        cfg = DATASET_CONFIGS[key]
        for name, values in sessions.items():
            try:
                ppg = read_eeglab_channel(Path(cfg["ppg_dir"]) / subject / f"{name}.mat", "PPG", cfg["default_fs"])
                rsp = read_eeglab_channel(Path(cfg["rsp_dir"]) / subject / f"{name}.mat", "RSP", cfg["default_fs"])
            except Exception as exc:
                print(f"  RespDiff {dataset} {subject} {name}: pieces not read ({exc})")
                continue
            ppg_sec, rsp_sec = ppg["signal"].size / ppg["fs"], rsp["signal"].size / rsp["fs"]
            cuts = sorted({b / ppg["fs"] for b in ppg["boundaries"]} | {b / rsp["fs"] for b in rsp["boundaries"]})
            edges = [0.0] + [c for c in cuts if 0.0 < c < min(ppg_sec, rsp_sec)] + [min(ppg_sec, rsp_sec)]
            piece = 0
            for start, end in zip(edges[:-1], edges[1:]):
                n_ppg = int(round(end * ppg["fs"])) - int(round(start * ppg["fs"]))
                n_rsp = int(round(end * rsp["fs"])) - int(round(start * rsp["fs"]))
                if n_ppg > 0 and n_rsp > 0:
                    piece += 1
                    tracks[f"{name}_piece{piece}"] = Track(values, fs, -start)
    else:
        tracks = {name: Track(values, fs) for name, values in sessions.items()}
    RESPDIFF_PLACEMENT.append({
        **info, "FS": fs, "Stored_Windows": len(placed), "Placed_Windows": int(len(rs)),
        "Recording_Windows": len(candidates), "Min_Match_r": float(rs.min()) if rs.size else math.nan,
        "Median_Match_r": float(np.median(rs)) if rs.size else math.nan,
        "Sources": ", ".join(sessions), "Recordings": ", ".join(tracks),
        "Note": "" if len(rs) == len(placed) else f"{len(placed) - len(rs)} stored windows not placed (left out)"})
    if len(rs) < len(placed):
        print(f"  RespDiff {dataset} {subject}: {len(placed) - len(rs)} of {len(placed)} windows not placed")
    return tracks


def rrwavenet_alignment_table(root: Path, label: str) -> Dict[Tuple[str, int], dict]:
    path = root / "Data_Quality" / WINDOW / f"{label}_window_quality.csv"
    if not path.exists():
        return {}
    wanted = {"Subject", "Segment_ID", "Target_Sign", "Target_Lag_Sec"}
    frame = pd.read_csv(path, dtype={"Subject": str}, usecols=lambda column: column in wanted)
    if not wanted <= set(frame.columns):
        return {}
    first = frame.groupby(["Subject", "Segment_ID"], sort=False).first()
    return {(str(s), int(g)): {"sign": float(r["Target_Sign"]), "lag_sec": float(r["Target_Lag_Sec"])}
            for (s, g), r in first.iterrows()}


def load_rrwavenet(path: Path) -> Dict[int, dict]:
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
        output, target = np.full(length, np.nan), np.full(length, np.nan)
        for row in rows:
            output[starts[row]:starts[row] + width] = predictions[row]
            target[starts[row]:starts[row] + width] = targets[row]
        segments[int(segment)] = {"output": output, "target": target, "windows": len(rows)}
    return segments


def pearson(a: Sequence[float], b: Sequence[float]) -> float:
    a, b = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    keep = np.isfinite(a) & np.isfinite(b)
    a, b = a[keep], b[keep]
    if len(a) < 3 or a.std() <= 1e-12 or b.std() <= 1e-12:
        return math.nan
    return float(np.corrcoef(a, b)[0, 1])


def target_reference_r(target: Track, reference: Track) -> float:
    times = target.offset_sec + np.arange(len(target.values)) / target.fs
    ref_times = np.arange(len(reference.values)) / reference.fs
    keep = np.isfinite(target.values) & (times >= 0) & (times <= ref_times[-1])
    if keep.sum() < 10:
        return math.nan
    return pearson(target.values[keep], np.interp(times[keep], ref_times, reference.values))


def finite_runs(valid: np.ndarray, min_length: int) -> List[Tuple[int, int]]:
    """Continuous stretches [start, end) where valid is True."""
    edges = np.diff(np.concatenate([[0], valid.astype(int), [0]]))
    starts, ends = np.flatnonzero(edges == 1), np.flatnonzero(edges == -1)
    return [(int(s), int(e)) for s, e in zip(starts, ends) if e - s >= min_length]


def lowpass_runs(values: np.ndarray, fs: float, cutoff_hz: float, order: int) -> np.ndarray:
    """Zero-phase Butterworth low-pass of every continuous run of finite
    samples separately (NaN gaps stay); runs too short to filter are kept as
    they are. cutoff_hz <= 0: unchanged."""
    out = np.array(values, dtype=float)
    if cutoff_hz <= 0:
        return out
    sos = butter(order, cutoff_hz, btype="low", fs=fs, output="sos")
    padlen = 3 * (2 * sos.shape[0] + 1)
    for start, end in finite_runs(np.isfinite(out), 1):
        if end - start > padlen:
            out[start:end] = sosfiltfilt(sos, out[start:end])
    return out


def minmax(values: np.ndarray) -> np.ndarray:
    """Smallest finite sample -> -1, largest -> +1; NaN stays NaN (all NaN
    when fewer than 3 samples or no variation)."""
    x = np.asarray(values, dtype=float)
    out = np.full(x.shape, np.nan)
    keep = np.isfinite(x)
    if keep.sum() >= 3:
        low, high = float(np.min(x[keep])), float(np.max(x[keep]))
        if high - low > 1e-12:
            out[keep] = 2.0 * (x[keep] - low) / (high - low) - 1.0
    return out


def safe_name(text: str) -> str:
    return re.sub(r"[^0-9A-Za-z._-]+", "_", str(text))


def band_filter(x: np.ndarray, fs: float, modality: str, config: dict) -> np.ndarray:
    """Linear trend removed, then zero-phase Butterworth high-pass and
    low-pass (a low-pass edge of 0 = none)."""
    x = detrend(np.asarray(x, dtype=float))
    nyquist = 0.5 * fs
    sections = [butter(2, config["highpass_hz"] / nyquist, btype="highpass", output="sos")]
    high = config["lowpass_hz"][modality]
    if high and high < 0.95 * nyquist:
        sections.append(butter(config["filter_order"], high / nyquist, btype="lowpass", output="sos"))
    sos = np.vstack(sections)
    if len(x) <= 3 * (2 * len(sos) + 1):
        return x
    return sosfiltfilt(sos, x, padlen=min(len(x) - 1, int(round(fs / config["highpass_hz"]))))


def local_unit(y: np.ndarray, fs: float, window_sec: float) -> np.ndarray:
    """The waveform scaled to its local range (5th-95th percentile in a
    moving window)."""
    width = max(3, int(round(window_sec * fs)) | 1)
    series = pd.Series(y)
    rolling = series.rolling(width, center=True, min_periods=max(3, width // 3))
    low = rolling.quantile(0.05).bfill().ffill().to_numpy()
    high = rolling.quantile(0.95).bfill().ffill().to_numpy()
    span = high - low
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(span > 1e-12, (y - low) / span, np.nan)


def half_crossings(u: np.ndarray, fs: float, low: float, high: float) -> List[Tuple[float, str]]:
    """Crossings of 0.5 with hysteresis (the waveform must pass low and high
    between crossings): [(time, 'up' / 'down')], times interpolated."""
    n = len(u)
    state = np.where(u >= high, 1, np.where(u <= low, -1, 0))
    state[~np.isfinite(u)] = 0
    index = np.where(state != 0, np.arange(n), 0)
    index = np.maximum.accumulate(index)
    filled = state[index]
    first = np.flatnonzero(state != 0)
    if not len(first):
        return []
    filled[:first[0]] = 0
    changes = np.flatnonzero((filled[1:] != filled[:-1]) & (filled[:-1] != 0)) + 1
    events = []
    for k in changes:
        kind = "up" if filled[k] == 1 else "down"
        j = k
        while j > 0 and not ((u[j - 1] - 0.5) * (u[j] - 0.5) <= 0 and u[j - 1] != u[j]):
            j -= 1
        if j == 0:
            continue
        t = (j - 1 + (0.5 - u[j - 1]) / (u[j] - u[j - 1])) / fs
        events.append((float(t), kind))
    return events


def detect_cycles(y: np.ndarray, fs: float, modality: str, config: dict) -> Tuple[List[dict], Dict[str, int]]:
    """Complete breathing cycles of one filtered waveform (start, transition,
    end in seconds; inspiration = start -> transition) and the counts of the
    cycles left out."""
    qc: Dict[str, int] = {}
    if len(y) < 3 or np.nanstd(y) <= config["sigma_eps"]:
        return [], {"constant signal": 1}
    u = local_unit(y, fs, config["range_window_sec"])
    events = half_crossings(u, fs, *config["hysteresis"])
    raw, extrema = [], []
    if modality == "capnogram":
        for (t0, k0), (t1, k1), (t2, k2) in zip(events, events[1:], events[2:]):
            if (k0, k1, k2) == ("down", "up", "down"):
                raw.append((t0, t1, t2))
    else:
        # one trough between a downward and the next upward crossing, one peak
        # between an upward and the next downward crossing
        for (t0, k0), (t1, k1) in zip(events, events[1:]):
            i0, i1 = int(math.ceil(t0 * fs)), int(math.floor(t1 * fs)) + 1
            if i1 - i0 < 1:
                continue
            part = y[i0:i1]
            if k0 == "down" and k1 == "up":
                extrema.append(((i0 + int(np.argmin(part))) / fs, "trough"))
            elif k0 == "up" and k1 == "down":
                extrema.append(((i0 + int(np.argmax(part))) / fs, "peak"))
        for (t0, k0), (t1, k1), (t2, k2) in zip(extrema, extrema[1:], extrema[2:]):
            if (k0, k1, k2) == ("trough", "peak", "trough"):
                raw.append((t0, t1, t2))
    marks = events if modality == "capnogram" else extrema
    if raw:
        partial = int(marks[0][0] < raw[0][0]) + int(marks[-1][0] > raw[-1][2])
    else:
        partial = int(bool(marks))
    if partial:
        qc["incomplete cycle at a stretch end"] = partial
    cycles = []
    min_cycle = config["min_cycle_sec"][modality]
    for start, transition, end in raw:
        t_i, t_e = transition - start, end - transition
        if not (min_cycle <= end - start <= config["max_cycle_sec"]):
            qc["cycle length out of range"] = qc.get("cycle length out of range", 0) + 1
            continue
        if min(t_i, t_e) < config["min_phase_sec"]:
            qc["phase shorter than the minimum"] = qc.get("phase shorter than the minimum", 0) + 1
            continue
        cycles.append({"start": start, "transition": transition, "end": end})
    return cycles, qc


def find_script(name: str, first: Sequence[Path]) -> Optional[Path]:
    """A script: the first found in the given folders, this script's folder,
    the current folder, the workspace, the parent folder and their
    subfolders."""
    folders = [Path(f) for f in first if f]
    folders += [SCRIPT_DIR, Path.cwd(), WORKSPACE, SCRIPT_DIR.parent]
    for base in (WORKSPACE, SCRIPT_DIR.parent):
        if base.is_dir():
            folders += sorted(child for child in base.iterdir() if child.is_dir())
    seen = []
    for folder in folders:
        path = folder / name
        if path not in seen:
            seen.append(path)
            if path.is_file():
                return path.resolve()
    return None


def load_capnobase_reader(args: argparse.Namespace) -> bool:
    """Import read_capnobase_case from the 261007 script (its results folder,
    --scripts-dir or the folders of find_script), else from the 261105 script."""
    global m07
    for run, name in CAPNOBASE_READERS:
        path = find_script(name, [args.scripts_dir, getattr(args, f"root_{run}")])
        if path is None:
            continue
        spec = importlib.util.spec_from_file_location(f"capnobase_reader_{run}", str(path))
        module = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = module
        try:
            spec.loader.exec_module(module)
        except Exception as exc:
            print(f"  {path} not imported ({type(exc).__name__}: {exc})")
            continue
        if hasattr(module, "read_capnobase_case"):
            m07 = module
            print(f"  CapnoBase reader (expert breath labels): {path}")
            return True
    print("  CapnoBase reader not found or not imported (261007_RRWaveNet_Ensemble_NoEvent.py or "
          "261105_RRWaveNet_Unified_Optimized.py; --scripts-dir): CapnoBase overlays skipped")
    return False


def set_data_paths(args: argparse.Namespace) -> None:
    for key, path in (("capnobase", args.capnobase_path), ("bidmc", args.bidmc_path)):
        if path:
            DATASET_CONFIGS[key]["data_path"] = str(path)
    if args.steam2_ppg_dir:
        DATASET_CONFIGS["steam2"]["ppg_dir"] = str(args.steam2_ppg_dir)
    if args.steam2_rsp_dir:
        DATASET_CONFIGS["steam2"]["rsp_dir"] = str(args.steam2_rsp_dir)


def read_recordings(dataset_key: str) -> Dict[str, Dict[str, dict]]:
    """Subject -> {recording name: GT (30 Hz), clipped GT samples, breath times
    (s; None without labels)}, named and cut as 261104 and the corr-encoder
    (as 261105_model_discussion_analysis)."""
    cfg = DATASET_CONFIGS[dataset_key]
    modality = MODALITIES[dataset_key]
    out: Dict[str, Dict[str, dict]] = {}

    def add(subject, name, ppg, resp, ppg_fs, resp_fs, breaths):
        ppg30, resp30 = to_fs(ppg, ppg_fs), to_fs(resp, resp_fs)
        length = min(len(ppg30), len(resp30))
        clipped = to_fs_mask(saturated_samples(resp, resp_fs, modality, SATURATION_CONFIG), resp_fs,
                             length, SATURATION_CONFIG["saturation_margin_sec"])
        out.setdefault(subject, {})[name] = {"gt": resp30[:length], "clipped": clipped, "breaths": breaths}

    if dataset_key == "capnobase":
        csv_dir = Path(cfg["data_path"])
        for case_id in capnobase_case_ids(csv_dir):
            try:
                case = m07.read_capnobase_case(csv_dir, case_id, cfg["default_fs"])
                breaths = case.get("breaths")
                breaths = None if breaths is None else np.sort(np.asarray(breaths, float)) / float(case["co2_fs"])
                add(case_id, case_id, case["ppg"], case["co2"], case["ppg_fs"], case["co2_fs"], breaths)
            except Exception as exc:
                print(f"[CapnoBase {case_id}] not read: {exc}")
    elif dataset_key == "bidmc":
        fs = float(cfg["default_fs"])
        mat = sio.loadmat(cfg["data_path"])
        for index, record in enumerate(mat["data"][0]):
            subject = f"{index + 1:02d}"
            try:
                ppg = np.asarray(record["ppg"][0, 0]["v"]).squeeze()
                ref = record["ref"][0, 0]
                imp = np.asarray(ref["resp_sig"][0, 0]["imp"][0, 0]["v"]).squeeze()
                ann = np.asarray(ref["breaths"][0, 0]["ann1"], float).ravel()  # MATLAB sample numbers
                add(subject, subject, ppg, imp, fs, fs, np.sort(ann - 1.0) / fs)
            except Exception as exc:
                print(f"[BIDMC {subject}] not read: {exc}")
    else:
        for subject, items in steam2_session_files(Path(cfg["ppg_dir"]), Path(cfg["rsp_dir"])).items():
            try:
                for name, ppg_path, rsp_path in items:
                    ppg = read_eeglab_channel(ppg_path, "PPG", cfg["default_fs"])
                    rsp = read_eeglab_channel(rsp_path, "RSP", cfg["default_fs"])
                    pieces, _ = steam2_session_pieces(ppg, rsp)
                    for k, (p, r, p_fs, r_fs) in enumerate(pieces):
                        add(subject, f"{name}_piece{k + 1}", p, r, p_fs, r_fs, None)
            except Exception as exc:
                print(f"[STEAM2 {subject}] not read: {exc}")
    return out


def rrwavenet_track(segment: dict, stored: Optional[dict], dataset_key: str, gt: np.ndarray,
                    lowpass_hz: float) -> np.ndarray:
    """RRWaveNet on the GT axis with its training sign / lag undone and the
    1 Hz low-pass (as 261104)."""
    lag = stored["lag_sec"] if stored else 0.0
    if stored:
        sign = stored["sign"]
    elif dataset_key != "capnobase":
        r = target_reference_r(Track(segment["target"], RRWAVENET_FS, lag), Track(gt, FS30))
        sign = -1.0 if r < 0 else 1.0
    else:
        sign = 1.0
    smoothed = lowpass_runs(sign * segment["output"], RRWAVENET_FS, lowpass_hz, RRWAVENET_LOWPASS_ORDER)
    return Track(smoothed, RRWAVENET_FS, lag).at(np.arange(len(gt)) / FS30)


def fft_rate(x: np.ndarray, fs: float, band_bpm: Tuple[float, float]) -> float:
    """Frequency (per minute) of the largest FFT peak in the band."""
    x = np.asarray(x, float)
    x = x[np.isfinite(x)]
    if len(x) < 8 or np.std(x) <= 1e-12:
        return math.nan
    n = max(16384, len(x))
    f = np.fft.rfftfreq(n, 1.0 / fs) * 60.0
    p = np.abs(np.fft.rfft(x - np.mean(x), n))
    keep = (f >= band_bpm[0]) & (f <= band_bpm[1])
    return float(f[keep][np.argmax(p[keep])])


def align(gt: np.ndarray, values: np.ndarray) -> np.ndarray:
    """The model on the GT's own time axis, not shifted in time (every sample
    at the same second as the GT sample beside it); inverted when its Pearson
    r with the GT over the recording is negative (every model alike)."""
    values = np.asarray(values, float)
    r = pearson(gt, values)
    return -values if np.isfinite(r) and r < 0 else values


def recording_cycles(values: Dict[str, np.ndarray], modality: str) -> Dict[str, List[dict]]:
    """Breathing cycles of every waveform (261030's detector) on the stretches
    (>= 30 s) where all exist."""
    valid = np.ones(len(values["GT"]), bool)
    for v in values.values():
        valid &= np.isfinite(v)
    out = {key: [] for key in values}
    for s, e in finite_runs(valid, int(MIN_RUN_SEC * FS30)):
        for key, v in values.items():
            y = band_filter(v[s:e], FS30, modality, CYCLE_CONFIG)
            cycles, _ = detect_cycles(y, FS30, modality, CYCLE_CONFIG)
            out[key] += [{k: c[k] + s / FS30 for k in ("start", "transition", "end")} for c in cycles]
    return out


def inside(cycles: List[dict], t0: float, t1: float) -> List[dict]:
    return [c for c in cycles if c["start"] >= t0 and c["end"] <= t1]


def match_cycles(gt: List[dict], model: List[dict]) -> int:
    """GT cycles matched 1:1 by a model cycle starting within half their length."""
    used, matched = set(), 0
    starts = np.array([c["start"] for c in model]) if model else np.empty(0)
    for c in gt:
        tolerance = 0.5 * (c["end"] - c["start"])
        best, best_gap = None, np.inf
        for k in np.flatnonzero(np.abs(starts - c["start"]) <= tolerance):
            gap = abs(starts[k] - c["start"])
            if k not in used and gap < best_gap:
                best, best_gap = int(k), gap
        if best is not None:
            used.add(best)
            matched += 1
    return matched


def window_row(values: Dict[str, np.ndarray], cycles: Dict[str, List[dict]], breaths: np.ndarray, t0: float,
               t1: float, band: Tuple[float, float]) -> Optional[dict]:
    """Rate, AE, PCC, score and cycles of every model in one window (None when
    a waveform has a gap in it or the GT has fewer than 3 breaths)."""
    a, b = int(round(t0 * FS30)), int(round(t1 * FS30))
    if not all(np.isfinite(v[a:b]).all() for v in values.values()):
        return None
    br = breaths[(breaths >= t0) & (breaths < t1)]
    if len(br) < MIN_BREATHS:
        return None
    gt_cycles = inside(cycles["GT"], t0, t1)
    gt_rr = 60.0 / float(np.mean(np.diff(br)))
    row = {"Start_Sec": t0, "End_Sec": t1, "GT_RR": gt_rr, "GT_Cycles": len(gt_cycles),
           "GT_Clipped_Cycles": sum(bool(c.get("clipped")) for c in gt_cycles)}
    for model in MODELS:
        if model not in values:
            continue
        seg = values[model][a:b]
        rate = fft_rate(seg, FS30, band)
        near = [c for c in cycles[model] if c["end"] > t0 and c["start"] < t1]
        matched = match_cycles(gt_cycles, near)
        n_in = len(inside(cycles[model], t0, t1))
        ae = abs(rate - gt_rr)
        pcc = pearson(values["GT"][a:b], seg)
        s = SHORT[model]
        row.update({f"{s}_Rate": rate, f"{s}_Ratio": rate / gt_rr, f"{s}_AE": ae, f"{s}_PCC": pcc,
                    f"{s}_Score": pcc - AE_WEIGHT * min(ae, AE_CAP), f"{s}_Cycles": n_in,
                    f"{s}_Missed": len(gt_cycles) - matched, f"{s}_Extra": max(0, n_in - matched),
                    f"{s}_Coverage": matched / len(gt_cycles) if gt_cycles else math.nan})
    others = [row[f"{SHORT[m]}_Score"] for m in OTHERS if np.isfinite(row.get(f"{SHORT[m]}_Score", math.nan))]
    row["Advantage"] = row["RRWaveNet_Score"] - max(others) if others else math.nan
    return row


def why_text(row: dict) -> str:
    """What the other models do in a window where RRWaveNet is right."""
    parts = []
    for model in OTHERS:
        s = SHORT[model]
        ratio = row.get(f"{s}_Ratio", math.nan)
        if not np.isfinite(ratio):
            continue
        if DOUBLE[0] <= ratio <= DOUBLE[1]:
            parts.append(f"{s} at 2x rate")
        elif HALF[0] <= ratio <= HALF[1]:
            parts.append(f"{s} at half rate")
        elif row.get(f"{s}_AE", 0) > 2:
            parts.append(f"{s} rate off by {row[f'{s}_AE']:.1f}")
        missed = row.get(f"{s}_Missed", 0) - row.get("RRWaveNet_Missed", 0)
        if missed >= 2:
            parts.append(f"{s} misses {int(missed)} more cycles")
        gap = row.get("RRWaveNet_PCC", math.nan) - row.get(f"{s}_PCC", math.nan)
        if np.isfinite(gap) and gap >= 0.3:
            parts.append(f"{s} PCC lower by {gap:.2f}")
    return "; ".join(parts)


def subject_means(rows: List[dict], size: int) -> Tuple[dict, str]:
    """Per model the mean AE (MAE), PCC and score over the subject's
    non-overlapping windows of a size (all its windows when none is complete)."""
    grid = [r for r in rows if r["Size_Sec"] == size and r["Grid"]]
    basis = "non-overlapping windows"
    if not grid:
        grid = [r for r in rows if r["Size_Sec"] == size]
        basis = "overlapping windows (no complete non-overlapping one)"
    out = {"N_Windows": len(grid)}
    for model in MODELS:
        s = SHORT[model]
        ae = [r[f"{s}_AE"] for r in grid if f"{s}_AE" in r and np.isfinite(r[f"{s}_AE"])]
        if ae:
            pcc = [r[f"{s}_PCC"] for r in grid if np.isfinite(r.get(f"{s}_PCC", math.nan))]
            out.update({f"{s}_MAE": float(np.mean(ae)), f"{s}_PCC": float(np.mean(pcc)) if pcc else math.nan,
                        f"{s}_Score": float(np.mean([r[f"{s}_Score"] for r in grid if f"{s}_Score" in r]))})
    wins = [r["Advantage"] > 0 for r in grid if np.isfinite(r.get("Advantage", math.nan))]
    out["RRWaveNet_Best_Window_Share"] = float(np.mean(wins)) if wins else math.nan
    return out, basis


# ===========================================================================
# Windows of every subject, the drawn windows and the figures (261108)
# ===========================================================================

ID_DIGITS = {"CapnoBase": 4, "BIDMC": 2}
THREE = ("RRWaveNet", "Deep corr-encoder", "RespDiff")


def norm_id(dataset: str, subject) -> str:
    """Subject IDs as the data files name them (CapnoBase 0009, BIDMC 01)."""
    text = re.sub(r"\.0+$", "", str(subject).strip())
    if dataset in ID_DIGITS and text.isdigit():
        text = text.zfill(ID_DIGITS[dataset])
    return text


def gt_bandpass(gt: np.ndarray) -> np.ndarray:
    """The GT band-passed GT_BAND_HZ (3rd-order Butterworth, zero phase; RespDiff v3's bandpass_filter)."""
    return v3_bandpass_filter(gt, FS30, GT_BAND_HZ[0], GT_BAND_HZ[1], GT_BAND_ORDER).astype(np.float64)


def window_measures(values: Dict[str, np.ndarray], cycles: Dict[str, List[dict]], breaths: np.ndarray, t0: float,
                    t1: float, band: Tuple[float, float], note: str = "") -> dict:
    """One window: the measures of 261108 (window_row) when it can be scored (RRWaveNet and another model in the
    recording, every waveform complete, >= 3 GT breaths); otherwise AE and r of every model complete in the window
    and why it is not scored (note: a reason of the whole recording). With the share of the window every model
    covers."""
    a, b = int(round(t0 * FS30)), int(round(t1 * FS30))
    cover = {m: float(np.isfinite(values[m][a:b]).mean()) for m in MODELS if m in values}
    row = None if note else window_row(values, cycles, breaths, t0, t1, band)
    if row is not None:
        row = dict(row, Scored=True, Not_Scored_Because="")
    else:
        br = breaths[(breaths >= t0) & (breaths < t1)]
        gt_rr = 60.0 / float(np.mean(np.diff(br))) if len(br) >= MIN_BREATHS else math.nan
        gt_cycles = inside(cycles["GT"], t0, t1)
        row = {"Start_Sec": t0, "End_Sec": t1, "GT_RR": gt_rr, "GT_Cycles": len(gt_cycles),
               "GT_Clipped_Cycles": sum(bool(c.get("clipped")) for c in gt_cycles)}
        for model, share in cover.items():
            if share < 1.0:
                continue
            seg = values[model][a:b]
            rate = fft_rate(seg, FS30, band)
            s = SHORT[model]
            row.update({f"{s}_Rate": rate, f"{s}_AE": abs(rate - gt_rr) if np.isfinite(gt_rr) else math.nan,
                        f"{s}_PCC": pearson(values["GT"][a:b], seg)})
        reasons = ([note] if note else []) + [
            f"no {FIG_NAME[m]} output in the window" if share == 0 else
            f"{FIG_NAME[m]} covers {100 * share:.0f}% of the window" for m, share in cover.items() if share < 1.0]
        if len(br) < MIN_BREATHS:
            reasons.append(f"fewer than {MIN_BREATHS} GT breaths")
        row.update(Scored=False, Not_Scored_Because="; ".join(reasons) or "-")
    for model, share in cover.items():
        row[f"{SHORT[model]}_Coverage_Pct"] = 100.0 * share
    return row


def dataset_windows(args: argparse.Namespace, dataset: str, run: str, root: Path
                    ) -> Tuple[List[dict], List[dict], Dict[Tuple[str, str], Dict[str, np.ndarray]], Dict[str, int]]:
    """30 s windows covering every second of every recording (with a model output) of every subject with RRWaveNet and
    another model (readers, alignment and cycles of 261108's dataset_windows): the windows with their measures, the
    recordings not drawn, and the aligned waveforms of every recording."""
    key = DATASET_KEYS[dataset]
    band = RATE_BAND_BPM[key]
    modality = MODALITIES[key]
    corr = load_correncoder(Path(args.correncoder_root), dataset)
    respdiff = respdiff_files(args, dataset)
    files = {f["Tag"]: f["Outputs_File"] for f in fold_files(root, dataset, RUNS[run]["selected"])
             if f["Outputs_File"].exists()}
    subjects = sorted(set(files) & (set(corr) | set(respdiff)))
    if args.subjects:
        subjects = [s for s in subjects if any(same_subject(s, x) for x in args.subjects)]
    left_out = args.excluded.get(dataset, [])
    subjects = [s for s in subjects if not any(same_subject(s, x) for x in left_out)]
    print(f"\n{dataset} [RRWaveNet {run}]: RRWaveNet {len(files)}, Deep Corr-Encoder {len(corr)}, RespDiff "
          f"{len(respdiff)} subjects; {len(subjects)} with RRWaveNet and another model (left out: "
          f"{', '.join(left_out) or '-'})", flush=True)
    rows, skipped, store, n_recordings = [], [], {}, {}
    if not subjects:
        return rows, skipped, store, n_recordings
    recordings = read_recordings(key)
    alignment = rrwavenet_alignment_table(root, dataset)
    for subject in subjects:
        if subject not in recordings:
            print(f"  {dataset} {subject}: no data file read (skipped)")
            skipped.append({"Run": run, "Dataset": dataset, "Subject": subject, "Reason": "no data file read"})
            continue
        segments = load_rrwavenet(files[subject])
        respdiff_by_name = respdiff_tracks(dataset, subject, respdiff[subject]) if subject in respdiff else {}
        corr_by_name = {r["name"]: r for r in corr.get(subject, [])}
        n_recordings[subject] = len(recordings[subject])
        count = 0
        for k, (name, rec) in enumerate(recordings[subject].items()):
            gt = np.asarray(rec["gt"], float)
            times = np.arange(len(gt)) / FS30
            index = corr_by_name[name]["index"] if name in corr_by_name else k
            raw = {}
            if name in respdiff_by_name:
                raw["RespDiff"] = respdiff_by_name[name].at(times)
            if name in corr_by_name:
                raw["Deep corr-encoder"] = Track(corr_by_name[name]["output"], FS30).at(times)
            if index in segments:
                raw["RRWaveNet"] = rrwavenet_track(segments[index], alignment.get((subject, index)), key, gt,
                                                   args.rrwavenet_lowpass_hz)
            if not raw:
                skipped.append({"Run": run, "Dataset": dataset, "Subject": subject, "Recording": name,
                                "Reason": "no model output in the recording"})
                continue
            # 261108 scores a recording only with RRWaveNet and another model: the others are drawn, not scored
            note = ("" if "RRWaveNet" in raw and len(raw) >= 2 else
                    "no RRWaveNet output in the recording" if "RRWaveNet" not in raw else
                    "no other model output in the recording")
            breaths = rec["breaths"]
            if breaths is None or len(breaths) < MIN_BREATHS:
                breaths = np.array([c["start"] for c in recording_cycles({"GT": gt}, modality)["GT"]])
            # the GT of the figures and of every r: band-passed 0.1-0.5 Hz; breathing cycles on the GT as recorded
            gt_band = gt_bandpass(gt)
            values = {"GT": gt_band, **{model: align(gt_band, v) for model, v in raw.items()}}
            cycles = recording_cycles({**values, "GT": gt}, modality)
            clipped = np.asarray(rec["clipped"], bool)
            for c in cycles["GT"]:
                c["clipped"] = bool(clipped[int(c["start"] * FS30):int(math.ceil(c["end"] * FS30)) + 1].any())
            store[(subject, name)] = values
            length = len(gt) / FS30
            for size in args.sizes:
                for t0, t1, kind in cover_windows(length, size):
                    rows.append({"Run": run, "Dataset": dataset, "Subject": subject, "Recording": name,
                                 "Size_Sec": size, "Window_Kind": kind, "Grid": kind == "grid",
                                 "Models": ", ".join(SHORT[m] for m in MODELS if m in values),
                                 **window_measures(values, cycles, breaths, t0, t1, band, note)})
                    count += 1
        print(f"  {dataset} {subject}: {count} windows", flush=True)
    return rows, skipped, store, n_recordings


def cover_windows(length: float, size: float) -> List[Tuple[float, float, str]]:
    """Windows of `size` s covering every second of a recording of `length` s: one after the other from 0 s (grid);
    when the recording does not end on a border, one more ending at its end (end; it overlaps the window before); a
    recording shorter than a window: one window over all of it (whole)."""
    if length < size:
        return [(0.0, float(length), "whole")]
    starts = np.arange(0.0, length - size + 1e-9, size).round(3)
    out = [(float(t0), float(t0) + size, "grid") for t0 in starts]
    if length - out[-1][1] > 1.0 / FS30:
        out.append((float(length) - size, float(length), "end"))
    return out


def subject_means_v3(rows: List[dict], size: int) -> dict:
    """Per model the mean AE (MAE) and r (PCC) over the subject's scored non-overlapping windows of a size (the end
    window and a short recording's window are drawn, not in the mean)."""
    scored = [r for r in rows if r["Size_Sec"] == size and r["Scored"] and r["Grid"]]
    out = {"N_Windows": sum(r["Size_Sec"] == size for r in rows), "N_Scored_Windows": len(scored)}
    for model in MODELS:
        s = SHORT[model]
        ae = [r[f"{s}_AE"] for r in scored if np.isfinite(r.get(f"{s}_AE", math.nan))]
        pcc = [r[f"{s}_PCC"] for r in scored if np.isfinite(r.get(f"{s}_PCC", math.nan))]
        if ae or pcc:
            out[f"{s}_MAE"] = float(np.mean(ae)) if ae else math.nan
            out[f"{s}_PCC"] = float(np.mean(pcc)) if pcc else math.nan
    return out


# ===========================================================================
# Figures (v3_overlap): the GT and every model on one axis
# ===========================================================================

def number(value: float) -> str:
    return f"{value:.2f}" if np.isfinite(value) else "n/a"


def window_table(row: dict, means: dict, size: int, present: Dict[str, str], names: Dict[str, str]) -> str:
    """AE and r of every model in the window and its subject mean (MAE, PCC), as a monospace table."""
    width = max(len(names[m]) for m in FIG_MODELS) + 2
    lines = [f"{'':{width}}{'This window':>14}   Subject mean",
             f"{'':{width}}{'AE':>6}{'r':>8}   {'MAE':>6}{'PCC':>8}"]
    for m in FIG_MODELS:
        s = SHORT[m]
        if present[m] == "none" and f"{s}_MAE" not in means:
            lines.append(f"{names[m]:<{width}}no stored output")
            continue
        if present[m] in ("none", "gap"):
            here = f"{'no output' if present[m] == 'none' else 'gap':^14}"
        else:
            here = f"{number(row.get(f'{s}_AE', math.nan)):>6}{number(row.get(f'{s}_PCC', math.nan)):>8}"
        lines.append(f"{names[m]:<{width}}{here}   {number(means.get(f'{s}_MAE', math.nan)):>6}"
                     f"{number(means.get(f'{s}_PCC', math.nan)):>8}")
    lines.append(f"Subject mean over {means['N_Scored_Windows']} scored non-overlapping windows of {size} s")
    return "\n".join(lines)


def overlap_page(path: Path, pdf, header: str, subheader: str, notes_text: str, t: np.ndarray, size: float,
                 gt_line: tuple, model_lines: Dict[str, tuple], table_text: str) -> None:
    """One figure: the GT and the models on one axis (the models drawn over the GT, RRWaveNet on top); beside the axis
    the legend and the table."""
    side_in, left_in, top_in, bottom_in, panel_in = 3.9, 0.75, 1.6, 0.6, 2.9
    width = 7.0 + 4.0 * size / 60.0 + 0.6
    height = top_in + bottom_in + panel_in
    fig = plt.figure(figsize=(width, height))
    ax = fig.add_axes([left_in / width, bottom_in / height, (width - left_in - side_in - 0.15) / width,
                       panel_in / height])
    values, color, label, lw = gt_line
    ax.plot(t, values, color=color, lw=lw, solid_capstyle="round", zorder=2)
    for z, m in enumerate(DRAW_ORDER, start=3):
        if m in model_lines:
            v, c, _, w = model_lines[m]
            ax.plot(t, v, color=c, lw=w, solid_capstyle="round", zorder=z)
    handles = [Line2D([], [], color=color, lw=lw)]
    labels = [label]
    for m in FIG_MODELS:  # legend: GT, RRWaveNet, Deep Corr-Encoder, RespDiff
        if m in model_lines:
            _, c, lab, w = model_lines[m]
            handles.append(Line2D([], [], color=c, lw=w))
            labels.append(lab)
    ax.set_xlim(t[0], t[0] + size if len(t) < size * FS30 else t[-1])
    ax.set_ylim(-1.18, 1.18)
    ax.set_ylabel("Min-max scaled", fontsize=8)
    ax.set_xlabel("Time in the recording (s)")
    ax.grid(True, color=GRID, lw=0.6)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    x = (width - side_in + 0.12) / width
    top = (bottom_in + panel_in) / height
    fig.legend(handles, labels, loc="upper left", bbox_to_anchor=(x, top), frameon=False, fontsize=7.6,
               borderaxespad=0.0)
    fig.text(x + 0.004, top - (0.3 + 0.2 * len(labels)) / height, table_text, family="monospace", fontsize=7.0,
             color=INK, va="top")
    fig.text(left_in / width, 1.0 - 0.22 / height, header, fontsize=10.5, color=INK, va="top")
    if subheader:
        fig.text(left_in / width, 1.0 - 0.50 / height, subheader, fontsize=8.5, color=INK, va="top")
    fig.text(left_in / width, 1.0 - 0.76 / height, notes_text, fontsize=7.2, color=MUTED, va="top", linespacing=1.4)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=200)
    pdf.savefig(fig)
    plt.close(fig)


def draw_window(folder: Path, pdf, dataset: str, run: str, row: dict, values: Dict[str, np.ndarray], means: dict,
                single: bool, k: int, n: int, digits: int) -> Path:
    """The overlap figure of one window: the GT and every model of the recording on one axis. Returns its file."""
    size = int(row["Size_Sec"])
    t0, t1 = row["Start_Sec"], row["End_Sec"]
    a, b = int(round(t0 * FS30)), int(round(t1 * FS30))
    t = t0 + np.arange(b - a) / FS30
    band = f"{GT_BAND_HZ[0]:g}-{GT_BAND_HZ[1]:g} Hz"
    rate = f"{row['GT_RR']:.1f} breaths/min" if np.isfinite(row.get("GT_RR", math.nan)) else \
        f"fewer than {MIN_BREATHS} breaths"
    gt_line = (minmax(values["GT"][a:b]), GT_LINE, f"GT ({MODALITIES[DATASET_KEYS[dataset]]}, {band}), {rate}",
               GT_WIDTH)
    names = {m: f"RRWaveNet ({run})" if m == "RRWaveNet" else FIG_NAME[m] for m in FIG_MODELS}
    present, model_lines = {}, {}
    for m in FIG_MODELS:
        if m not in values:
            present[m] = "none"
            continue
        seg = values[m][a:b]
        present[m] = "full" if np.isfinite(seg).all() else ("gap" if np.isfinite(seg).any() else "none")
        if present[m] != "none":
            label = names[m] + ("" if present[m] == "full" else
                                f" (covers {100 * np.isfinite(seg).mean():.0f}% of the window)")
            model_lines[m] = (minmax(seg), FIG_COLOR[m], label, MODEL_WIDTH)
    who = row["Subject"] + ("" if single else f" ({row['Recording']})")
    header = f"{dataset} {who} · {size} s window {t0:.0f}-{t1:.0f} s · window {k} of {n}"
    if row["Scored"] and np.isfinite(row.get("Advantage", math.nan)):
        header += f" · Advantage {row['Advantage']:.2f}"
    sub = []
    if row["Window_Kind"] == "end":
        sub.append(f"the last {size} s of the recording, overlapping the window before")
    elif row["Window_Kind"] == "whole":
        sub.append(f"the whole recording ({t1 - t0:.0f} s, shorter than {size} s)")
    if not row["Scored"]:
        sub.append(f"not scored: {row['Not_Scored_Because']}")
    subheader = "; ".join(sub)
    notes_text = ("GT and every model on one axis, the same seconds of the recording (no model shifted in time; a model "
                  f"inverted when r < 0 over the recording)\nGT black, band-passed {band}; each waveform min-max scaled "
                  "in the window; RRWaveNet low-pass 1 Hz; no line = no stored output there\nAE = |FFT rate - GT rate| "
                  "(breaths/min); r = Pearson r with the band-passed GT; subject mean over the scored non-overlapping "
                  "windows")
    stem = f"{k:0{digits}d}_{safe_name(row['Subject'] if single else row['Recording'])}_{t0:.0f}-{t1:.0f}s"
    path = folder / f"{stem}.png"
    overlap_page(path, pdf, header, subheader, notes_text, t, size, gt_line, model_lines,
                 window_table(row, means, size, present, names))
    return path


def notes(args: argparse.Namespace) -> pd.DataFrame:
    return pd.DataFrame([
        ("Models", f"RRWaveNet: CapnoBase 261007, BIDMC / STEAM2 261013 (stored ensemble output); Deep Corr-Encoder: "
                   f"{args.correncoder_root} (261107 paper version); RespDiff: Baseline outputs of "
                   f"260519_RespDiff_AllDatasets_v3 ({args.respdiff_root}; CapnoBase first from "
                   f"{args.respdiff_capnobase_root}), run {args.respdiff_run_tag}"),
        ("RespDiff on the time axis", "v3's 32 s windows (16 s apart) placed where v3 cut them (as 261108): v3's "
                                      "preprocessing of the same data redone and every stored window matched through "
                                      f"its stored reference (|r| >= {RESPDIFF_MATCH_MIN_R}); overlaps joined by a "
                                      "weighted mean of the z-scored windows; windows v3 left out stay gaps "
                                      "(RespDiff_Placement)"),
        ("GT", f"the reference at 30 Hz, band-passed {GT_BAND_HZ[0]:g}-{GT_BAND_HZ[1]:g} Hz ({GT_BAND_ORDER}rd-order "
               "Butterworth, zero phase) for the figures and every r; GT rate from the breaths in the window (CapnoBase "
               "expert labels, BIDMC ann1, STEAM2 the cycles of the GT as recorded)"),
        ("Alignment", "no model shifted in time; a model inverted once per recording when its r with the band-passed "
                      "GT is negative; RRWaveNet training sign / lag undone, low-pass 1 Hz"),
        ("Windows", f"every second of every recording with a model output: windows of {', '.join(f'{s} s' for s in args.sizes)} "
                    "from 0 s, one after the other (Window_Kind grid); when the recording does not end on a window "
                    "border, one more window ending at its end (end, overlaps the window before); a recording shorter "
                    "than a window in one window (whole); every window drawn, also without a model output in it"),
        ("Scored", f"RRWaveNet and another model in the recording, every model complete in the window and >= "
                   f"{MIN_BREATHS} GT breaths: the measures of 261108. The scored grid windows make the subject means "
                   "(as 261108). Not scored: why in Not_Scored_Because; AE and r then for every model complete in the "
                   "window"),
        ("AE / r / score", f"AE = |FFT rate - GT rate| (breaths/min); r = Pearson r with the band-passed GT; score = r - "
                           f"{AE_WEIGHT} x min(AE, {AE_CAP}); Advantage = RRWaveNet's score - the better of the others"),
        ("Coverage_Pct", "share of the window in which a model has an output (%)"),
        ("Left out", "; ".join(f"{d} {', '.join(s)}" for d, s in args.excluded.items() if s)),
        ("Figures", "one per window: the GT and the three models on one axis (the same seconds, no model shifted; the "
                    "models over the GT, RRWaveNet on top); GT black, RRWaveNet #dc2a1c (the 261108 red a little darker), "
                    "Deep Corr-Encoder #2d93e1, RespDiff #24a085 (the 261108 blue and green); lines GT 1.5, "
                    "models 1.7 (as 261108); "
                    "every waveform min-max scaled in the window; beside it the legend and the AE / r table"),
    ], columns=["Item", "Description"])


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="261108 waveform comparison, v3_overlap: RRWaveNet, Deep Corr-Encoder "
                                                 "(paper version) and RespDiff against the GT on one axis, every "
                                                 "non-overlapping window of every subject")
    for run, config in RUNS.items():
        parser.add_argument(f"--root-{run}", type=Path, default=config["root"],
                            help=f"results folder of {run} (default {config['root']})")
    parser.add_argument("--output-root", type=Path, default=PROJECT_ROOT / OUTPUT_FOLDER)
    parser.add_argument("--datasets", nargs="+", choices=list(FIGURE_ORDER), default=list(FIGURE_ORDER))
    parser.add_argument("--sizes", nargs="+", type=int, default=list(SIZES), help="window sizes (s), default 30")
    parser.add_argument("--subjects", nargs="+", default=None, help="only these subjects")
    parser.add_argument("--correncoder-root", type=Path, default=CORRENCODER_ROOT)
    parser.add_argument("--respdiff-root", type=Path, default=RESPDIFF_ROOT,
                        help="results of 260519_RespDiff_AllDatasets_v3 (Baseline outputs used)")
    parser.add_argument("--respdiff-capnobase-root", type=Path, default=RESPDIFF_CAPNOBASE_ROOT,
                        help="results of 261108_RespDiff_CapnoBase_Baseline_v3 (CapnoBase, used first)")
    parser.add_argument("--respdiff-run-tag", default=RESPDIFF_RUN_TAG, help="run folder of the RespDiff results")
    parser.add_argument("--rrwavenet-lowpass-hz", type=float, default=RRWAVENET_LOWPASS_HZ)
    parser.add_argument("--scripts-dir", type=Path, default=None,
                        help="folder of 261007_RRWaveNet_Ensemble_NoEvent.py (CapnoBase breath labels)")
    parser.add_argument("--capnobase-path", type=Path, default=None)
    parser.add_argument("--bidmc-path", type=Path, default=None)
    parser.add_argument("--steam2-ppg-dir", type=Path, default=None)
    parser.add_argument("--steam2-rsp-dir", type=Path, default=None)
    parser.add_argument("--exclude-steam2", nargs="*", default=list(EXCLUDED["STEAM2"]))
    parser.add_argument("--exclude-bidmc", nargs="*", default=list(EXCLUDED["BIDMC"]))
    parser.add_argument("--exclude-capnobase", nargs="*", default=list(EXCLUDED["CapnoBase"]))
    args = parser.parse_args(argv)
    args.root_261105 = None  # the CapnoBase reader is looked for in the 261007 folder (and the usual places)
    args.excluded = {"STEAM2": [norm_id("STEAM2", s) for s in args.exclude_steam2],
                     "BIDMC": [norm_id("BIDMC", s) for s in args.exclude_bidmc],
                     "CapnoBase": [norm_id("CapnoBase", s) for s in args.exclude_capnobase]}
    return args


FIRST_COLUMNS = ["Dataset", "Run", "Size_Sec", "Subject", "Recording", "Window_No", "Windows_Of_Subject", "Start_Sec",
                 "End_Sec", "Window_Kind", "Grid", "Scored", "Not_Scored_Because", "GT_RR", "GT_Clipped_Cycles", "Advantage",
                 "RRWaveNet_AE", "RRWaveNet_PCC", "RRWaveNet_Rate", "CorrEncoder_AE", "CorrEncoder_PCC",
                 "CorrEncoder_Rate", "RespDiff_AE", "RespDiff_PCC", "RespDiff_Rate", "RRWaveNet_Coverage_Pct",
                 "CorrEncoder_Coverage_Pct", "RespDiff_Coverage_Pct", "Figure"]


def ordered(rows: List[dict]) -> pd.DataFrame:
    frame = pd.DataFrame(rows)
    if frame.empty:
        return frame
    return frame[[c for c in FIRST_COLUMNS if c in frame] + [c for c in frame if c not in FIRST_COLUMNS]]


def main(argv: Optional[Sequence[str]] = None) -> None:
    args = parse_args(argv)
    out_root = Path(args.output_root)
    out_root.mkdir(parents=True, exist_ok=True)
    print(f"Output: {out_root} (every second of every subject in {', '.join(map(str, args.sizes))} s windows)\n"
          f"Deep Corr-Encoder: {args.correncoder_root}\nRespDiff (Baseline): {args.respdiff_root}; CapnoBase "
          f"first {args.respdiff_capnobase_root} ({args.respdiff_run_tag})")
    set_data_paths(args)
    datasets = list(args.datasets)
    if "CapnoBase" in datasets and not load_capnobase_reader(args):
        datasets.remove("CapnoBase")
    drawn, skipped, subject_rows = [], [], []
    for dataset in datasets:
        run = next(r for r, ds in RUN_DATASETS.items() if dataset in ds)
        root = Path(getattr(args, f"root_{run}"))
        if not root.exists():
            print(f"[{run}] results folder not found: {root} ({dataset} skipped)")
            continue
        rows, skip, store, n_recordings = dataset_windows(args, dataset, run, root)
        skipped += skip
        for size in args.sizes:
            for subject in sorted({r["Subject"] for r in rows if r["Size_Sec"] == size}):
                windows = [r for r in rows if r["Subject"] == subject and r["Size_Sec"] == size]
                means = subject_means_v3(windows, size)
                subject_rows.append({"Dataset": dataset, "Run": run, "Subject": subject, "Size_Sec": size, **means})
                folder = out_root / dataset / f"{size}s" / safe_name(subject)
                folder.mkdir(parents=True, exist_ok=True)
                single = n_recordings.get(subject, 1) == 1
                digits = len(str(len(windows)))
                with PdfPages(folder / f"{dataset}_{safe_name(subject)}_{size}s_overlap.pdf") as pdf:
                    for k, row in enumerate(windows, start=1):
                        path = draw_window(folder, pdf, dataset, run, row, store[(subject, row["Recording"])], means,
                                           single, k, len(windows), digits)
                        drawn.append({**row, "Window_No": k, "Windows_Of_Subject": len(windows),
                                      "Figure": str(path.relative_to(out_root))})
                print(f"  {dataset} {size} s {subject}: {len(windows)} windows drawn "
                      f"({means['N_Scored_Windows']} scored)", flush=True)
        del store
    path = out_root / "261108_Waveform_v3_overlap.xlsx"
    sheets = {"Windows": ordered(drawn),
              "Skipped": pd.DataFrame(skipped, columns=["Run", "Dataset", "Subject", "Recording", "Reason"]),
              "Subject_Summary": pd.DataFrame(subject_rows), "RespDiff_Placement": pd.DataFrame(RESPDIFF_PLACEMENT),
              "Notes": notes(args)}
    with pd.ExcelWriter(path, engine="openpyxl") as writer:
        for name, frame in sheets.items():
            (frame.round(4) if name != "Notes" else frame).to_excel(writer, sheet_name=name, index=False)
    print(f"\n{len(drawn)} windows drawn under {out_root}\\<Dataset>\\<size>s\\<subject>; {len(skipped)} "
          f"recordings / subjects not drawn (Skipped); table: {path}")


if __name__ == "__main__":
    main()
