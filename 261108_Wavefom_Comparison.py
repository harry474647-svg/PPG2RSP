# -*- coding: utf-8 -*-
r"""261108_Wavefom_Comparison: the respiratory waveforms of RRWaveNet, the
deep corr-encoder (paper version) and RespDiff against the ground truth (GT),
for the windows that show RRWaveNet's advantage: 4 subjects per dataset and
window size, chosen from every subject's windows, with the subjects that had
an MAE issue left out. Nothing is trained: the stored outputs are read.

Models (stored outputs)
-----------------------
  RRWaveNet          ensemble output of its run: CapnoBase 261007
                     (261007_RRWaveNet_NoEvent_v1), BIDMC and STEAM2 261013
                     (261013_RRWavNet_STEAM_BIDMC_v1)
  Deep Corr-Encoder  261107_DeepCorrencoder_PaperVersion_v1 (the paper's model,
                     LOSOCV): {correncoder-root}\{Dataset}\Predictions\{subject}_outputs.npz
                     (fused 30 Hz output of every recording)
  RespDiff           261020_RespDiff_v2: respdiff\{Dataset}\Predictions\{subject}_ddpm50.npz
GT: the reference as recorded, at 30 Hz (CapnoBase capnogram, BIDMC impedance,
STEAM2 belt), with the breaths (CapnoBase expert labels, BIDMC ann1, STEAM2 the
GT cycles). Readers, alignment and window measures: those of
261106_RRWaveNet_Performance_Boxplots_v2 (copied unchanged): no model is shifted
in time; a model is inverted once per recording when its r with the GT is
negative (the same rule for all); RRWaveNet with its training sign / lag
undone and low-pass 1 Hz.

Windows and the choice of the subjects
--------------------------------------
Windows of 30 s and 60 s (--sizes), one starting every 5 s (--step). In every
window, per model: AE = |FFT rate of the waveform - GT rate from the breaths|
(breaths/min), r = Pearson r with the GT, score = r - 0.1 x min(AE, 5);
Advantage = RRWaveNet's score - the better score of the two others.
Per subject and size the window with the largest Advantage; the subjects with
all three models are ranked by that Advantage and the first 4 (--top) whose
window has RRWaveNet AE <= 1 breath/min and Advantage > 0 are drawn (when fewer
qualify, the next ones by Advantage fill up, marked in the table).
Subjects left out (MAE issue): STEAM2 22_LMS, 26_CKK, 12_YJJ; BIDMC 27;
CapnoBase 0115 (--exclude-*).

Figures (every chosen window, all panels the same seconds of the recording,
every waveform min-max scaled in the window)
  overlap   (a) GT vs RRWaveNet, (b) GT vs Deep Corr-Encoder, (c) GT vs RespDiff:
            each model drawn over the GT
  separate  GT, RRWaveNet, Deep Corr-Encoder, RespDiff, each in its own panel
GT black solid; RRWaveNet coral, Deep Corr-Encoder blue, RespDiff teal (solid).
Beside every model panel: its AE and r in the window and its subject means.

Outputs (default D:\PPG2RSP_RRWaveNet_Inspired\261108_Wavefom_Comparison)
  {Dataset}\{30s,60s}\{rank}_{subject}_{t0}-{t1}s_overlap.png / _separate.png
     (STEAM2, several recordings per subject: {rank}_{recording}_{t0}-{t1}s_...)
  {Dataset}\{30s,60s}\{Dataset}_{size}s_overlap.pdf / _separate.pdf (every page)
  261108_Waveform_Candidates.xlsx
     Candidates       the drawn windows (rank, AE, r, rate of every model, Advantage)
     Best_Windows     every subject's best window per size (the ranking)
     Subject_Summary  every subject's mean MAE / PCC per model and size
     Notes
Run:
  python 261108_Wavefom_Comparison.py
Options: --top, --sizes, --step, --datasets, --subjects, --root-261007,
  --root-261013, --correncoder-root, --respdiff-root, --respdiff-sampling,
  --scripts-dir (CapnoBase breath labels: 261007_RRWaveNet_Ensemble_NoEvent.py),
  --capnobase-path, --bidmc-path, --steam2-ppg-dir, --steam2-rsp-dir,
  --exclude-steam2, --exclude-bidmc, --exclude-capnobase, --rrwavenet-lowpass-hz,
  --output-root
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
from scipy.signal import butter, detrend, resample_poly, sosfiltfilt

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
TOP = 4

# Figures: the GT black; the models in the colours of the user's legend (RRWaveNet coral, Deep Corr-Encoder blue,
# RespDiff teal), all solid lines
GT_LINE = "#000000"
FIG_MODELS = ("RRWaveNet", "Deep corr-encoder", "RespDiff")
FIG_NAME = {"RRWaveNet": "RRWaveNet", "Deep corr-encoder": "Deep Corr-Encoder", "RespDiff": "RespDiff"}
FIG_COLOR = {"RRWaveNet": "#ef7f75", "Deep corr-encoder": "#70addb", "RespDiff": "#7cb5b0"}


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
RESPDIFF_ROOT = WORKSPACE / "261020_RespDiff_v2"
CORRENCODER_ROOT = WORKSPACE / "261107_DeepCorrencoder_PaperVersion_v1"  # the paper version (LOSOCV)
RESPDIFF_SAMPLING = "ddpm50"
RRWAVENET_LOWPASS_HZ = 1.0
FS30 = 30  # time axis of the overlays: the 30 Hz of RespDiff, the corr-encoder and the GT
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


def load_respdiff(root: Path, label: str, sampling: str) -> Dict[str, Dict[str, np.ndarray]]:
    folder = root / "respdiff" / label
    subjects = {}
    for path in sorted((folder / "Predictions").glob(f"*_{sampling}.npz")):
        subject = path.name[: -len(f"_{sampling}.npz")]
        if not (folder / "Fold_Results" / f"{subject}_{sampling}.json").exists():
            continue
        with np.load(path, allow_pickle=False) as data:
            names = [str(name) for name in data["recording_names"]]
            subjects[subject] = {name: data[f"recording{i}_prediction"].astype(np.float64)
                                 for i, name in enumerate(names) if f"recording{i}_prediction" in data.files}
    return subjects


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


# Panels of a waveform page, top to bottom: the run's RRWaveNet, RespDiff, the
# deep corr-encoder.


# ===========================================================================
# Windows of every subject, the chosen subjects and the figures (261108)
# ===========================================================================

ID_DIGITS = {"CapnoBase": 4, "BIDMC": 2}
THREE = ("RRWaveNet", "Deep corr-encoder", "RespDiff")


def norm_id(dataset: str, subject) -> str:
    """Subject IDs as the data files name them (CapnoBase 0009, BIDMC 01)."""
    text = re.sub(r"\.0+$", "", str(subject).strip())
    if dataset in ID_DIGITS and text.isdigit():
        text = text.zfill(ID_DIGITS[dataset])
    return text


def dataset_windows(args: argparse.Namespace, dataset: str, run: str, root: Path
                    ) -> Tuple[List[dict], Dict[Tuple[str, str], Dict[str, np.ndarray]], Dict[str, int]]:
    """Every window of every subject with RRWaveNet and another model (measures of v2's overlay_dataset), and the
    aligned waveforms of every recording."""
    key = DATASET_KEYS[dataset]
    band = RATE_BAND_BPM[key]
    modality = MODALITIES[key]
    corr = load_correncoder(Path(args.correncoder_root), dataset)
    respdiff = load_respdiff(Path(args.respdiff_root), dataset, args.respdiff_sampling)
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
    rows, store, n_recordings = [], {}, {}
    if not subjects:
        return rows, store, n_recordings
    recordings = read_recordings(key)
    alignment = rrwavenet_alignment_table(root, dataset)
    for subject in subjects:
        if subject not in recordings:
            print(f"  {dataset} {subject}: no data file read (skipped)")
            continue
        segments = load_rrwavenet(files[subject])
        corr_by_name = {r["name"]: r for r in corr.get(subject, [])}
        n_recordings[subject] = len(recordings[subject])
        count = 0
        for k, (name, rec) in enumerate(recordings[subject].items()):
            gt = np.asarray(rec["gt"], float)
            times = np.arange(len(gt)) / FS30
            index = corr_by_name[name]["index"] if name in corr_by_name else k
            raw = {}
            if name in respdiff.get(subject, {}):
                raw["RespDiff"] = Track(respdiff[subject][name], FS30).at(times)
            if name in corr_by_name:
                raw["Deep corr-encoder"] = Track(corr_by_name[name]["output"], FS30).at(times)
            if index in segments:
                raw["RRWaveNet"] = rrwavenet_track(segments[index], alignment.get((subject, index)), key, gt,
                                                   args.rrwavenet_lowpass_hz)
            if "RRWaveNet" not in raw or len(raw) < 2:
                continue
            breaths = rec["breaths"]
            if breaths is None or len(breaths) < MIN_BREATHS:
                breaths = np.array([c["start"] for c in recording_cycles({"GT": gt}, modality)["GT"]])
            values = {"GT": gt, **{model: align(gt, v) for model, v in raw.items()}}
            cycles = recording_cycles(values, modality)
            clipped = np.asarray(rec["clipped"], bool)
            for c in cycles["GT"]:
                c["clipped"] = bool(clipped[int(c["start"] * FS30):int(math.ceil(c["end"] * FS30)) + 1].any())
            store[(subject, name)] = values
            length = len(gt) / FS30
            for size in args.sizes:
                grid = set(np.arange(0.0, length - size + 1e-9, size).round(3))
                starts = sorted(grid | set(np.arange(0.0, length - size + 1e-9, args.step).round(3)))
                for t0 in starts:
                    row = window_row(values, cycles, breaths, float(t0), float(t0) + size, band)
                    if row is not None:
                        rows.append({"Run": run, "Dataset": dataset, "Subject": subject, "Recording": name,
                                     "Size_Sec": size, "Grid": t0 in grid,
                                     "Models": ", ".join(SHORT[m] for m in MODELS if m in values), **row})
                        count += 1
        print(f"  {dataset} {subject}: {count} windows", flush=True)
    return rows, store, n_recordings


def has_three(row: dict) -> bool:
    return all(np.isfinite(row.get(f"{SHORT[m]}_AE", math.nan)) for m in THREE)


def choose(rows: List[dict], size: int, top: int) -> Tuple[List[dict], List[dict]]:
    """Every subject's best window of a size (largest Advantage, all three models present), ranked; the first `top`
    with RRWaveNet AE <= RATE_OK_BPM and Advantage > 0 (filled up by Advantage when fewer qualify)."""
    best = {}
    for r in rows:
        if r["Size_Sec"] != size or not has_three(r) or not np.isfinite(r.get("Advantage", math.nan)):
            continue
        if r["Subject"] not in best or (r["Advantage"], r["RRWaveNet_PCC"]) > (best[r["Subject"]]["Advantage"],
                                                                                best[r["Subject"]]["RRWaveNet_PCC"]):
            best[r["Subject"]] = r
    ranked = sorted(best.values(), key=lambda r: r["Advantage"], reverse=True)
    qualified = [r for r in ranked if r["RRWaveNet_AE"] <= RATE_OK_BPM and r["Advantage"] > 0]
    chosen = [dict(r, Filled=False) for r in qualified[:top]]
    for r in ranked:
        if len(chosen) >= top:
            break
        if all(r["Subject"] != c["Subject"] for c in chosen):
            chosen.append(dict(r, Filled=True))
    for rank, r in enumerate(chosen, start=1):
        r["Rank"] = rank
    best_rows = [dict(r, Qualifies=r in qualified, Order=i + 1) for i, r in enumerate(ranked)]
    return best_rows, chosen


def page(path: Path, pdf, header: str, notes_text: str, t: np.ndarray, panels: List[dict]) -> None:
    """One figure: panels stacked, the same seconds; beside every panel its legend and text."""
    side_in, left_in, top_in, bottom_in, panel_in, gap_in = 3.3, 0.75, 1.35, 0.6, 1.6, 0.38
    size = t[-1] - t[0]
    width = 7.0 + 4.0 * size / 60.0
    height = top_in + bottom_in + len(panels) * panel_in + (len(panels) - 1) * gap_in
    fig = plt.figure(figsize=(width, height))
    axes = []
    for k, p in enumerate(panels):
        bottom = bottom_in + (len(panels) - 1 - k) * (panel_in + gap_in)
        ax = fig.add_axes([left_in / width, bottom / height, (width - left_in - side_in - 0.15) / width,
                           panel_in / height], sharex=axes[0] if axes else None)
        axes.append(ax)
        handles, labels = [], []
        for values, color, label, lw in p["lines"]:
            ax.plot(t, values, color=color, lw=lw, solid_capstyle="round")
            handles.append(Line2D([], [], color=color, lw=lw))
            labels.append(label)
        ax.set_xlim(t[0], t[-1])
        ax.set_ylim(-1.18, 1.18)
        ax.set_ylabel("Min-max scaled", fontsize=8)
        ax.grid(True, color=GRID, lw=0.6)
        ax.set_axisbelow(True)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        ax.set_title(p["title"], fontsize=9.5, loc="left", pad=3)
        if k < len(panels) - 1:
            ax.tick_params(labelbottom=False)
        x = (width - side_in + 0.12) / width
        top = (bottom + panel_in) / height
        fig.legend(handles, labels, loc="upper left", bbox_to_anchor=(x, top), frameon=False, fontsize=7.6,
                   borderaxespad=0.0)
        if p.get("text"):
            fig.text(x + 0.004, top - (0.24 + 0.2 * len(labels)) / height, p["text"], family="monospace",
                     fontsize=7.2, color=INK, va="top")
    axes[-1].set_xlabel("Time in the recording (s)")
    fig.text(left_in / width, 1.0 - 0.22 / height, header, fontsize=10.5, color=INK, va="top")
    fig.text(left_in / width, 1.0 - 0.50 / height, notes_text, fontsize=7.2, color=MUTED, va="top", linespacing=1.4)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=200)
    pdf.savefig(fig)
    plt.close(fig)


def model_text(model: str, row: dict, means: dict, size: int) -> str:
    s = SHORT[model]
    lines = [f"This window  AE {row[f'{s}_AE']:5.2f}  r {row[f'{s}_PCC']:5.2f}"]
    if f"{s}_MAE" in means:
        lines.append(f"Subject mean ({means['N_Windows']} x {size} s)")
        lines.append(f"             MAE {means[f'{s}_MAE']:4.2f}  PCC {means[f'{s}_PCC']:4.2f}")
    return "\n".join(lines)


def draw_candidate(out_root: Path, pdfs: dict, dataset: str, run: str, row: dict, values: Dict[str, np.ndarray],
                   means: dict, basis: str, single: bool) -> Dict[str, str]:
    size = int(row["Size_Sec"])
    t0, t1 = row["Start_Sec"], row["End_Sec"]
    a, b = int(round(t0 * FS30)), int(round(t1 * FS30))
    t = t0 + np.arange(b - a) / FS30
    gt = minmax(values["GT"][a:b])
    gt_label = f"GT ({MODALITIES[DATASET_KEYS[dataset]]}), {row['GT_RR']:.1f} breaths/min"
    name = {m: f"RRWaveNet ({run})" if m == "RRWaveNet" else FIG_NAME[m] for m in FIG_MODELS}
    who = row["Subject"] + ("" if single else f" ({row['Recording']})")
    header = (f"{dataset} {who} · {size} s window {t0:.0f}-{t1:.0f} s · candidate {row['Rank']}"
              + (" (filled up)" if row.get("Filled") else "") + f" · Advantage {row['Advantage']:.2f}")
    notes_text = ("all panels the same seconds of the recording (no model shifted in time; a model inverted when r < 0 "
                  "over the recording); GT black\neach waveform min-max scaled in the window; RRWaveNet low-pass 1 Hz\n"
                  f"AE = |FFT rate - GT rate| (breaths/min); r = Pearson r with the GT; subject mean over {basis}")
    stem = f"{row['Rank']}_{safe_name(row['Subject'] if single else row['Recording'])}_{t0:.0f}-{t1:.0f}s"
    folder = out_root / dataset / f"{size}s"
    overlap = [{"title": f"({'abc'[k]}) GT vs {name[m]}",
                "lines": [(gt, GT_LINE, gt_label, 1.5), (minmax(values[m][a:b]), FIG_COLOR[m], name[m], 1.7)],
                "text": model_text(m, row, means, size)} for k, m in enumerate(FIG_MODELS)]
    separate = [{"title": f"(a) GT ({MODALITIES[DATASET_KEYS[dataset]]})", "lines": [(gt, GT_LINE, gt_label, 1.5)]}]
    separate += [{"title": f"({'bcd'[k]}) {name[m]}", "lines": [(minmax(values[m][a:b]), FIG_COLOR[m], name[m], 1.7)],
                  "text": model_text(m, row, means, size)} for k, m in enumerate(FIG_MODELS)]
    paths = {}
    for kind, panels in (("overlap", overlap), ("separate", separate)):
        if (size, kind) not in pdfs:
            pdf_path = folder / f"{dataset}_{size}s_{kind}.pdf"
            pdf_path.parent.mkdir(parents=True, exist_ok=True)
            pdfs[(size, kind)] = PdfPages(pdf_path)
        path = folder / f"{stem}_{kind}.png"
        page(path, pdfs[(size, kind)], header, notes_text, t, panels)
        paths[kind] = str(path.relative_to(out_root))
    return paths


def notes(args: argparse.Namespace) -> pd.DataFrame:
    return pd.DataFrame([
        ("Models", f"RRWaveNet: CapnoBase 261007, BIDMC / STEAM2 261013 (stored ensemble output); Deep Corr-Encoder: "
                   f"{args.correncoder_root} (261107 paper version); RespDiff: {args.respdiff_root} "
                   f"({args.respdiff_sampling})"),
        ("GT", "the reference as recorded at 30 Hz; GT rate from the breaths in the window (CapnoBase expert labels, "
               "BIDMC ann1, STEAM2 the GT cycles); a window needs >= 3 breaths and every waveform without a gap"),
        ("Alignment", "no model shifted in time; a model inverted once per recording when its r with the GT is "
                      "negative; RRWaveNet training sign / lag undone, low-pass 1 Hz"),
        ("AE / r / score", f"AE = |FFT rate - GT rate| (breaths/min); r = Pearson r with the GT; score = r - "
                           f"{AE_WEIGHT} x min(AE, {AE_CAP}); Advantage = RRWaveNet's score - the better of the others"),
        ("Choice", f"per subject and size the window with the largest Advantage (all three models present); the first "
                   f"{args.top} subjects by Advantage with RRWaveNet AE <= {RATE_OK_BPM} and Advantage > 0 (Filled = "
                   "taken by Advantage because fewer qualified)"),
        ("Left out", "; ".join(f"{d} {', '.join(s)}" for d, s in args.excluded.items() if s)),
        ("Figures", "overlap: each model over the GT; separate: GT and every model in its own panel; GT black, "
                    "RRWaveNet coral, Deep Corr-Encoder blue, RespDiff teal; every waveform min-max scaled in the window"),
    ], columns=["Item", "Description"])


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="261108 waveform comparison: RRWaveNet, Deep Corr-Encoder (paper "
                                                 "version) and RespDiff against the GT")
    for run, config in RUNS.items():
        parser.add_argument(f"--root-{run}", type=Path, default=config["root"],
                            help=f"results folder of {run} (default {config['root']})")
    parser.add_argument("--output-root", type=Path, default=PROJECT_ROOT)
    parser.add_argument("--datasets", nargs="+", choices=list(FIGURE_ORDER), default=list(FIGURE_ORDER))
    parser.add_argument("--top", type=int, default=TOP, help="subjects drawn per dataset and size")
    parser.add_argument("--sizes", nargs="+", type=int, default=list(WINDOW_SIZES), help="window sizes (s)")
    parser.add_argument("--step", type=float, default=STEP_SEC, help="a window starts every this many s")
    parser.add_argument("--subjects", nargs="+", default=None, help="only these subjects")
    parser.add_argument("--correncoder-root", type=Path, default=CORRENCODER_ROOT)
    parser.add_argument("--respdiff-root", type=Path, default=RESPDIFF_ROOT)
    parser.add_argument("--respdiff-sampling", default=RESPDIFF_SAMPLING)
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


def main(argv: Optional[Sequence[str]] = None) -> None:
    args = parse_args(argv)
    out_root = Path(args.output_root)
    out_root.mkdir(parents=True, exist_ok=True)
    print(f"Output: {out_root}\nDeep Corr-Encoder: {args.correncoder_root}\nRespDiff: {args.respdiff_root}")
    set_data_paths(args)
    datasets = list(args.datasets)
    if "CapnoBase" in datasets and not load_capnobase_reader(args):
        datasets.remove("CapnoBase")
    candidates, best_all, subject_rows = [], [], []
    for dataset in datasets:
        run = next(r for r, ds in RUN_DATASETS.items() if dataset in ds)
        root = Path(getattr(args, f"root_{run}"))
        if not root.exists():
            print(f"[{run}] results folder not found: {root} ({dataset} skipped)")
            continue
        rows, store, n_recordings = dataset_windows(args, dataset, run, root)
        if not rows:
            continue
        for subject in sorted({r["Subject"] for r in rows}):
            for size in args.sizes:
                means, basis = subject_means([r for r in rows if r["Subject"] == subject], size)
                subject_rows.append({"Dataset": dataset, "Run": run, "Subject": subject, "Size_Sec": size,
                                     "Subject_Means_Over": basis, **means})
        pdfs = {}
        for size in args.sizes:
            best, chosen = choose(rows, size, args.top)
            best_all += best
            print(f"  {dataset} {size} s: " + ("; ".join(
                f"{c['Rank']}. {c['Subject']} {c['Start_Sec']:.0f}-{c['End_Sec']:.0f}s (Advantage {c['Advantage']:.2f}, "
                f"RRWaveNet AE {c['RRWaveNet_AE']:.2f} r {c['RRWaveNet_PCC']:.2f}{', filled' if c['Filled'] else ''})"
                for c in chosen) or "no window with all three models"))
            for c in chosen:
                means, basis = subject_means([r for r in rows if r["Subject"] == c["Subject"]], size)
                paths = draw_candidate(out_root, pdfs, dataset, run, c, store[(c["Subject"], c["Recording"])], means,
                                       basis, n_recordings.get(c["Subject"], 1) == 1)
                candidates.append({**c, "Overlap_Figure": paths["overlap"], "Separate_Figure": paths["separate"]})
        for pdf in pdfs.values():
            pdf.close()
    first = ["Dataset", "Run", "Size_Sec", "Rank", "Filled", "Subject", "Recording", "Start_Sec", "End_Sec", "GT_RR",
             "Advantage", "RRWaveNet_AE", "RRWaveNet_PCC", "RRWaveNet_Rate", "CorrEncoder_AE", "CorrEncoder_PCC",
             "CorrEncoder_Rate", "RespDiff_AE", "RespDiff_PCC", "RespDiff_Rate", "Overlap_Figure", "Separate_Figure"]
    cand = pd.DataFrame(candidates)
    if not cand.empty:
        cand = cand[[c for c in first if c in cand] + [c for c in cand if c not in first]]
    best_frame = pd.DataFrame(best_all)
    sheets = {"Candidates": cand, "Best_Windows": best_frame, "Subject_Summary": pd.DataFrame(subject_rows),
              "Notes": notes(args)}
    path = out_root / "261108_Waveform_Candidates.xlsx"
    with pd.ExcelWriter(path, engine="openpyxl") as writer:
        for name, frame in sheets.items():
            (frame.round(4) if name != "Notes" else frame).to_excel(writer, sheet_name=name, index=False)
    print(f"\nFigures under {out_root}\\<Dataset>\\<size>s; table: {path}")


if __name__ == "__main__":
    main()
