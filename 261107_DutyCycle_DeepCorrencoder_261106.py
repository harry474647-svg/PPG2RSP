# -*- coding: utf-8 -*-
r"""261107_DutyCycle_DeepCorrencoder_261106: the outputs of the 261106 deep
corr-encoder (CorrEncoderNet, 0.1-0.5 Hz PPG input, model 01_low) made again
from its checkpoints, on the time axis of the recordings, and their duty cycle
by the rule of 261107_DutyCycle_RespDiff_DeepCorrencoder (official RespDiff
code), so the values can be set next to those of RRWaveNet, RespDiff and the
261028 corr-encoder in 261107_DutyCycle_Results. Nothing is trained.

Checkpoints (261106_DeepCorrencoder_PaperStyle_All and its v2)
  <results folder>\Deep_Correncoder\{STEAM2, BIDMC, CapnoBase}\Models\{subject}_01_low_best.pth
  (the LOSO model of every test subject) and Metrics\LOSO_CorrEncoder_Ablation_PerSubject.csv
  (the run's own scores, used for the check below).

Outputs made again (code of 261106, copied unchanged)
  Data as the corr-encoder reads it (BIDMC 125 Hz; CapnoBase pleth / co2 of
  {case}_8min_signal.csv at the PPG rate, resampled to 125 Hz; STEAM2 every
  piece of every session, split at boundary events, resampled to 125 Hz),
  preprocess_subject_windows: PPG polarity by skewness, PPG and reference
  band-passed 0.1-0.5 Hz, the reference inverted when its r with the PPG band
  is negative (training sign), non-overlapping 10 s windows, quality check;
  every window's PPG z-scored (clip 5) and given to the test subject's model.
  Every window gets an output (the run scored only the windows that passed
  the quality check; QC below).
  Check: the run's own 30 s scores (Waveform_Corr_30sec_Mean, BPM_MAE,
  Waveform_MAE_30sec_Mean) computed again from the new outputs of the
  windows that passed the quality check, next to the stored values (sheet
  Check); the same numbers when the outputs are those of the run (on the GPU
  with mixed precision up to small rounding differences).

Duty cycle (rule of 261107_DutyCycle_RespDiff_DeepCorrencoder, functions copied)
  * Reference: the respiratory waveform of the data file at 30 Hz (CapnoBase
    capnogram, BIDMC impedance, STEAM2 belt; every STEAM2 piece a recording),
    low-pass 1 Hz (Butterworth 8, zero phase), non-overlapping 5 s segments
    each min-max scaled to [0, 1], joined.
  * Corr-encoder: the output of every 10 s window with the training sign
    undone (inverted back where the reference was inverted), min-max scaled
    to [0, 1] in its window (the model was trained on z-scored windows; as
    RRWaveNet's (y + 1) / 2 of its [-1, 1] windows), put on the recording's
    time axis (125 Hz), taken at the nearest 30 Hz sample, low-pass 1 Hz as
    the reference. No time shift.
  * 60 s windows moved by 20 s; duty cycle = share of samples >= 0.5; a
    window with a sample without output (the end of a recording after the
    last whole 10 s window) is left out (N_Windows_Skipped).
  * Per subject (all windows of all its recordings): Duty_Cycle_Corr (Pearson
    r of reference and model duty cycle over the windows), Duty_MAE, Duty_Bias,
    Duty_Reference_Mean, Duty_Predicted_Mean; the same over the windows whose
    every sample lies in a 10 s window that passed the quality check
    (QC_Duty_*; QC_Share per window in Duty_Windows).

Outputs (default <results folder>\Deep_Correncoder\DutyCycle_From_Checkpoints)
  261107_DutyCycle_DeepCorrencoder_261106.xlsx
    Summary                   per dataset: Mean ± SD (Median) over subjects
    CorrEncoder261106_Duty    per subject (columns of CorrEncoder_Duty of 261107, + QC_*)
    Check                     the run's 30 s scores: stored and from the new outputs
    Duty_Windows, Coverage, Notes
  and the same tables as CSV files;
  {Dataset}\Predictions\{subject}_outputs.npz   per recording: recording_names,
      recording{k}_output (30 Hz, [0, 1] per window, sign undone, NaN = no output;
      before the 1 Hz low-pass), recording{k}_reference (GT at 30 Hz),
      recording{k}_qc (30 Hz, 1 = in a window that passed the quality check),
      recording{k}_output_125hz (the model output, sign undone, not scaled)
  {Dataset}\Fold_Results\{subject}_summary.json

The results folder is the folder of this file, or else the current folder,
when it is named 261106_DeepCorrencoder_PaperStyle_All; otherwise
RESULTS_ROOT; --results-root sets another one.

Run (in D:\PPG2RSP_RRWaveNet_Inspired\261106_DeepCorrencoder_PaperStyle_All):
  python 261107_DutyCycle_DeepCorrencoder_261106.py
Options: --datasets steam2 bidmc capnobase, --results-root, --output-root,
  --capnobase-path, --bidmc-path, --steam2-ppg-dir, --steam2-rsp-dir
A dataset without checkpoints (not trained yet) is listed in Coverage and
left out.
"""

import argparse
import csv
import json
import math
import os
import re
from contextlib import nullcontext
from dataclasses import dataclass
from fractions import Fraction
from math import gcd
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import scipy.io as sio
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.signal import butter, correlate, find_peaks, resample_poly, sosfiltfilt
from scipy.stats import pearsonr, skew

# ---------------------------------------------------------------------------
# Folders and settings
# ---------------------------------------------------------------------------

WORKSPACE = Path(r"D:\PPG2RSP_RRWaveNet_Inspired")
RESULTS_FOLDER_NAME = "261106_DeepCorrencoder_PaperStyle_All"
RESULTS_ROOT = WORKSPACE / RESULTS_FOLDER_NAME
CE_FOLDER = "Deep_Correncoder"
DUTY_FOLDER = "DutyCycle_From_Checkpoints"
MODEL_NAME = "01_low"  # the 0.1-0.5 Hz input model of 261106
MODEL_LABEL = "Deep corr-encoder 261106"
DATASET_ORDER = ("steam2", "bidmc", "capnobase")
STORED_PER_SUBJECT = "LOSO_CorrEncoder_Ablation_PerSubject.csv"
INFERENCE_BATCH = 256

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Duty cycle (official RespDiff code; as 261107_DutyCycle_RespDiff_DeepCorrencoder)
FS = 30
FS30 = FS
SEGMENT = 5 * FS  # 5 s segments of the reference, each min-max scaled to [0, 1]
DUTY_WINDOW = 60 * FS
DUTY_HOP = 20 * FS
DUTY_THRESHOLD = 0.5
DUTY_LOWPASS_HZ = 1.0
DUTY_LOWPASS_ORDER = 8

# Corr-encoder (261106, unchanged)
FS_METRIC = 125
WIN_SEC = 10.0
STRIDE_SEC = 10.0
SEGMENT_WINDOWS = 3
SEGMENT_SEC = WIN_SEC * SEGMENT_WINDOWS
INPUT_CLIP_VALUE = 5.0
FILTERS = [16, 32, 64]
K_SIZES = [31, 15, 7]
STEM_DROPOUT = 0.10
BLOCK_DROPOUT = 0.25
FUSION_DROPOUT = 0.20
QUALITY_MIN_CORR = 0.03
USE_AMP = True
AMP_ENABLED = USE_AMP and device.type == "cuda"

# Data files (as the corr-encoder and 261107 read them)
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
CAPNOBASE_FILE_SUFFIXES = (".csv", ".tab")
_NUMBER_PATTERN = re.compile(r"[-+]?(?:\d+\.?\d*|\.\d+)(?:[eE][-+]?\d+)?")
STEAM2_CHANNELS = {"PPG": ("PPG", "PLETH", "BVP"), "RSP": ("RSP", "RESP", "RESPIRATION", "BELT")}
STEAM2_SESSION_PATTERN = re.compile(r"^(?P<subject>.+?)[_\- ]*session[_\- ]*(?P<session>\d+)$", re.IGNORECASE)


# ---------------------------------------------------------------------------
# Copied from 261107_DutyCycle_RespDiff_DeepCorrencoder (data files of the GT, duty cycle rule)
# ---------------------------------------------------------------------------

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


def capnobase_file(csv_dir: Path, case_id: str, kind: str) -> Optional[Path]:
    for suffix in CAPNOBASE_FILE_SUFFIXES:
        path = csv_dir / f"{case_id}_8min_{kind}{suffix}"
        if path.exists():
            return path
    return None


def _sniff_delimiter(path: Path) -> str:
    with open(path, newline="", encoding="utf-8-sig") as handle:
        first = handle.readline()
    return "\t" if first.count("\t") > first.count(",") else ","


def _normalize_field_name(name: object) -> str:
    return re.sub(r"[^0-9a-z]+", "_", str(name).lower()).strip("_")


def read_capnobase_fields(path: Path) -> Dict[str, List[str]]:
    """Fields of a CapnoBase param, meta, labels or reference file.

    Fields are read one per column (header row, then value rows) or one per
    row (name in the first cell, values after it), comma or tab separated.
    Every cell is split on whitespace, so a field whose events are stored in
    one cell (" 974 1990 ...") yields one token per event. Names are lower
    case with every non-alphanumeric run replaced by "_".
    """
    with open(path, newline="", encoding="utf-8-sig") as handle:
        rows = [
            [cell.strip() for cell in row]
            for row in csv.reader(handle, delimiter=_sniff_delimiter(path))
            if any(cell.strip() for cell in row)
        ]
    fields: Dict[str, List[str]] = {}
    if not rows:
        return fields
    header = rows[0]
    column_layout = all(len(row) <= len(header) for row in rows[1:]) and not any(
        _NUMBER_PATTERN.fullmatch(cell) for cell in header if cell
    )
    if column_layout and len(header) == 2 and len(rows) > 2:
        # A two-column "name, value" table is a row layout under a header.
        names = [row[0] for row in rows[1:] if row]
        if names and not any(_NUMBER_PATTERN.fullmatch(name) for name in names if name):
            column_layout = False
            rows = rows[1:]
    if column_layout:
        for index, name in enumerate(header):
            if name:
                fields[_normalize_field_name(name)] = [
                    token
                    for row in rows[1:]
                    if index < len(row)
                    for token in row[index].split()
                ]
    else:
        for row in rows:
            if row and row[0]:
                fields[_normalize_field_name(row[0])] = [
                    token for cell in row[1:] for token in cell.split()
                ]
    return fields


def _field_numbers(tokens: Sequence[str]) -> np.ndarray:
    return np.asarray(
        [float(match) for token in tokens for match in _NUMBER_PATTERN.findall(token)],
        dtype=np.float64,
    )


def _find_field(fields: Dict[str, List[str]], *parts: str, suffix: str = "") -> Optional[str]:
    """Shortest field name containing every part; names ending in `suffix` win."""
    names = [name for name in fields if all(part in name for part in parts)]
    if suffix:
        names = [name for name in names if name.endswith(suffix)] or names
    return min(names, key=len) if names else None


def _sampling_rate(
    fields: Dict[str, List[str]],
    signal_name: str,
    default: float,
) -> Tuple[float, str]:
    name = _find_field(fields, "samplingrate", signal_name) or _find_field(
        fields, "rate", signal_name
    )
    values = _field_numbers(fields[name]) if name else np.empty(0)
    values = values[np.isfinite(values) & (values > 0)]
    if len(values):
        return float(values[0]), name
    return float(default), "default"


def read_capnobase_signals(path: Path) -> Tuple[np.ndarray, np.ndarray, str]:
    """PPG (pleth) and capnogram (co2) columns of a signal file."""
    frame = pd.read_csv(path, sep=_sniff_delimiter(path))
    frame.columns = [_normalize_field_name(column) for column in frame.columns]

    def column(signal_name: str) -> str:
        names = [name for name in frame.columns if signal_name in name]
        names = [name for name in names if name.endswith("_y")] or names
        if not names:
            raise KeyError(f"{path.name}: no {signal_name} column in {list(frame.columns)}")
        return min(names, key=len)

    ppg_column, co2_column = column("pleth"), column("co2")
    ppg = pd.to_numeric(frame[ppg_column], errors="coerce").to_numpy(np.float32)
    co2 = pd.to_numeric(frame[co2_column], errors="coerce").to_numpy(np.float32)
    return ppg, co2, f"{ppg_column},{co2_column}"


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
    ratio = Fraction(FS / float(fs)).limit_denominator(1000)
    return resample_poly(x, ratio.numerator, ratio.denominator) if ratio != 1 else x


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


def lowpass_official(x: np.ndarray) -> np.ndarray:
    """Butterworth order 8, 1 Hz, zero phase (every continuous stretch of
    finite samples on its own)."""
    return lowpass_runs(np.asarray(x, dtype=float), FS30, DUTY_LOWPASS_HZ, DUTY_LOWPASS_ORDER)


def official_reference(gt: np.ndarray) -> np.ndarray:
    """The GT as the RespDiff reference: low-pass filtered, non-overlapping 5 s
    segments from the start, each min-max scaled to [0, 1] (constant segment
    = 0), joined; the rest after the last whole segment is dropped."""
    x = lowpass_official(gt)
    n = (len(x) // SEGMENT) * SEGMENT
    segments = x[:n].reshape(-1, SEGMENT)
    low = segments.min(axis=1, keepdims=True)
    span = segments.max(axis=1, keepdims=True) - low
    return ((segments - low) / np.where(span > 0, span, 1.0)).reshape(-1)


def duty_windows(reference: np.ndarray, prediction: np.ndarray) -> List[dict]:
    """60 s windows moved by 20 s over the reference (official code): the
    share of samples >= 0.5 of reference and model; a window where either
    has no value is marked not valid."""
    n = min(len(reference), len(prediction))
    rows = []
    for k, start in enumerate(range(0, n - DUTY_WINDOW + 1, DUTY_HOP)):
        r = reference[start:start + DUTY_WINDOW]
        p = prediction[start:start + DUTY_WINDOW]
        valid = bool(np.isfinite(r).all() and np.isfinite(p).all())
        rows.append({"Window": k, "Start_Sec": start / FS30, "End_Sec": (start + DUTY_WINDOW) / FS30,
                     "Valid": valid,
                     "Duty_Reference": float((r >= DUTY_THRESHOLD).mean()) if valid else math.nan,
                     "Duty_Predicted": float((p >= DUTY_THRESHOLD).mean()) if valid else math.nan})
    return rows


def safe_corr(a: Sequence[float], b: Sequence[float]) -> float:
    a, b = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    if len(a) < 2 or np.std(a) <= 0 or np.std(b) <= 0:
        return math.nan
    return float(np.corrcoef(a, b)[0, 1])


def duty_summary(rows: List[dict]) -> dict:
    valid = [r for r in rows if r["Valid"]]
    ref = np.array([r["Duty_Reference"] for r in valid], dtype=float)
    pred = np.array([r["Duty_Predicted"] for r in valid], dtype=float)
    return {"N_Duty_Windows": len(valid), "N_Windows_Skipped": len(rows) - len(valid),
            "Duty_Cycle_Corr": safe_corr(ref, pred),
            "Duty_MAE": float(np.mean(np.abs(pred - ref))) if len(valid) else math.nan,
            "Duty_Bias": float(np.mean(pred - ref)) if len(valid) else math.nan,
            "Duty_Reference_Mean": float(ref.mean()) if len(valid) else math.nan,
            "Duty_Predicted_Mean": float(pred.mean()) if len(valid) else math.nan}


def mean_sd_median(values) -> Tuple[float, float, float, int]:
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if not len(values):
        return math.nan, math.nan, math.nan, 0
    sd = float(values.std(ddof=1)) if len(values) > 1 else math.nan
    return float(values.mean()), sd, float(np.median(values)), int(len(values))


def write_excel(path: Path, sheets: Dict[str, pd.DataFrame]) -> None:
    try:
        with pd.ExcelWriter(path, engine="openpyxl") as writer:
            for name, frame in sheets.items():
                frame.to_excel(writer, index=False, sheet_name=name[:31])
    except Exception as exc:
        print(f"Excel export failed ({exc}); the CSV files are complete.")



# ---------------------------------------------------------------------------
# Copied from 261106_DeepCorrencoder_PaperStyle_All(_v2), unchanged (corr-encoder preprocessing,
# model and 30 s scores)
# ---------------------------------------------------------------------------

def safe_tag(text):
    return "".join(c if c.isalnum() or c in "-_." else "_" for c in str(text))


def autocast_context():
    if not AMP_ENABLED:
        return nullcontext()
    try:
        return torch.amp.autocast(device_type="cuda", enabled=True)
    except Exception:
        return torch.cuda.amp.autocast(enabled=True)


def sanitize_signal(sig):
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


def bandpass(sig, fs, low, high, order=3):
    sig = sanitize_signal(sig)
    if sig.size < max(32, order * 12):
        return sig.copy()

    nyq = 0.5 * float(fs)
    low = max(float(low), 1e-5)
    high = min(float(high), nyq * 0.99)
    if not np.isfinite(low) or not np.isfinite(high) or high <= low:
        return sig.copy()

    try:
        sos = butter(order, [low / nyq, high / nyq], btype="band", output="sos")
        return sosfiltfilt(sos, sig).astype(np.float64)
    except ValueError:
        return sig.copy()


def maybe_resample(sig, src_fs, dst_fs):
    sig = sanitize_signal(sig)
    src_fs = int(round(float(src_fs)))
    dst_fs = int(round(float(dst_fs)))
    if src_fs <= 0 or dst_fs <= 0:
        raise ValueError(f"Invalid sampling rate: src={src_fs}, dst={dst_fs}")
    if sig.size == 0 or src_fs == dst_fs:
        return sig.astype(np.float64)

    factor = gcd(src_fs, dst_fs)
    up = dst_fs // factor
    down = src_fs // factor
    return resample_poly(sig, up, down).astype(np.float64)


def safe_pearsonr(a, b):
    a = sanitize_signal(a)
    b = sanitize_signal(b)
    if len(a) != len(b):
        m = min(len(a), len(b))
        a = a[:m]
        b = b[:m]
    if len(a) < 2 or np.std(a) < 1e-8 or np.std(b) < 1e-8:
        return np.nan
    return float(pearsonr(a, b)[0])


def correct_polarity_skew(sig):
    sig = sanitize_signal(sig)
    if sig.size < 3:
        return sig
    return -sig if skew(sig) < 0 else sig


def align_phase_only(ppg_ref, target_sig):
    ppg_ref = sanitize_signal(ppg_ref)
    target_sig = sanitize_signal(target_sig)
    m = min(len(ppg_ref), len(target_sig))
    ppg_ref = ppg_ref[:m]
    target_sig = target_sig[:m]
    corr = safe_pearsonr(ppg_ref, target_sig)
    return -target_sig if np.isfinite(corr) and corr < 0 else target_sig


def extract_windows_stride(sig, fs, win_sec=WIN_SEC, stride_sec=STRIDE_SEC):
    sig = np.asarray(sig)
    win_len = int(round(win_sec * fs))
    step = int(round(stride_sec * fs))
    if win_len <= 0 or step <= 0:
        raise ValueError(f"Invalid window config: win_len={win_len}, step={step}")

    length = sig.shape[-1]
    if length < win_len:
        return np.array([])

    windows = []
    is_multichannel = sig.ndim > 1
    for st in range(0, length - win_len + 1, step):
        if is_multichannel:
            windows.append(sig[:, st : st + win_len])
        else:
            windows.append(sig[st : st + win_len])
    return np.asarray(windows)


def instance_zscore_normalize(data, clip_value=None):
    data = np.asarray(data, dtype=np.float32)
    time_axis = data.ndim - 1
    mean_val = data.mean(axis=time_axis, keepdims=True)
    std_val = data.std(axis=time_axis, keepdims=True)
    std_val[std_val < 1e-6] = 1.0
    norm = (data - mean_val) / std_val
    if clip_value is not None:
        norm = np.clip(norm, -clip_value, clip_value)
    return norm.astype(np.float32)


def normalize_for_plot(sig):
    sig = sanitize_signal(sig)
    return ((sig - np.mean(sig)) / (np.std(sig) + 1e-8)).astype(np.float64)


def waveform_mae(a, b):
    a = sanitize_signal(a)
    b = sanitize_signal(b)
    m = min(len(a), len(b))
    if m == 0:
        return np.nan
    return float(np.mean(np.abs(a[:m] - b[:m])))


def breaths_per_min_zc_single(pred, true, window_sec, fs=FS_METRIC):
    pred = bandpass(pred, fs, 0.1, 0.5, order=3)
    true = bandpass(true, fs, 0.1, 0.5, order=3)
    scale = 60.0 / float(window_sec)
    p_c = pred - np.mean(pred)
    t_c = true - np.mean(true)
    if len(p_c) < 2 or len(t_c) < 2:
        return np.nan
    zc_pred = ((p_c[:-1] * p_c[1:]) < 0).sum()
    zc_true = ((t_c[:-1] * t_c[1:]) < 0).sum()
    bpm_pred = (zc_pred / 2.0) * scale
    bpm_true = (zc_true / 2.0) * scale
    return float(abs(bpm_pred - bpm_true))


def check_quality_real_world(x_low, y_true, fs):
    x_low = sanitize_signal(x_low)
    y_true = sanitize_signal(y_true)
    if len(x_low) != len(y_true):
        m = min(len(x_low), len(y_true))
        x_low = x_low[:m]
        y_true = y_true[:m]
    if len(y_true) < int(4 * fs):
        return False

    y_norm = y_true - np.mean(y_true)
    if np.std(y_norm) < 1e-4:
        return False

    acor = correlate(y_norm, y_norm, mode="full")
    acor = acor[len(acor) // 2 :]
    acor /= acor[0] + 1e-8
    peaks, _ = find_peaks(acor, height=0.20)
    min_peak = int(1.5 * fs)
    max_peak = int(10.0 * fs)
    if not any(min_peak < p < max_peak for p in peaks):
        return False

    corr = safe_pearsonr(x_low, y_true)
    if not np.isfinite(corr) or corr < QUALITY_MIN_CORR:
        return False
    return True


def evaluate_30sec_segments(pred_windows, true_windows, fs=FS_METRIC):
    segment_results = []
    num_segments = len(true_windows) // SEGMENT_WINDOWS

    for i in range(num_segments):
        st = i * SEGMENT_WINDOWS
        ed = st + SEGMENT_WINDOWS
        pred_seg = np.concatenate(pred_windows[st:ed])
        true_seg = np.concatenate(true_windows[st:ed])
        pred_f = bandpass(pred_seg, fs, 0.1, 0.5)
        true_f = bandpass(true_seg, fs, 0.1, 0.5)
        pred_n = normalize_for_plot(pred_f)
        true_n = normalize_for_plot(true_f)
        corr = safe_pearsonr(pred_n, true_n)
        mae = waveform_mae(pred_n, true_n)
        bpm_err = breaths_per_min_zc_single(pred_f, true_f, SEGMENT_SEC, fs=fs)
        segment_results.append(
            {
                "segment_idx": i,
                "corr": corr,
                "mae": mae,
                "bpm_err": bpm_err,
                "pred": pred_n,
                "true": true_n,
            }
        )

    return segment_results


def init_weights(module):
    if isinstance(module, (nn.Conv1d, nn.ConvTranspose1d)):
        nn.init.kaiming_normal_(module.weight, nonlinearity="relu")
        if module.bias is not None:
            nn.init.zeros_(module.bias)
    elif isinstance(module, nn.BatchNorm1d):
        nn.init.ones_(module.weight)
        nn.init.zeros_(module.bias)


class ConvBlock(nn.Module):
    def __init__(self, in_c, out_c, kernel, dropout=0.2):
        super().__init__()
        padding = kernel // 2
        self.conv1 = nn.Conv1d(in_c, out_c, kernel_size=kernel, padding=padding)
        self.bn1 = nn.BatchNorm1d(out_c)
        self.conv2 = nn.Conv1d(out_c, out_c, kernel_size=kernel, padding=padding)
        self.bn2 = nn.BatchNorm1d(out_c)
        self.shortcut = nn.Conv1d(in_c, out_c, kernel_size=1) if in_c != out_c else nn.Identity()
        self.act = nn.GELU()
        self.drop = nn.Dropout(dropout)

    def forward(self, x):
        residual = self.shortcut(x)
        out = self.act(self.bn1(self.conv1(x)))
        out = self.drop(out)
        out = self.bn2(self.conv2(out))
        return self.act(out + residual)


class SingleEncoder(nn.Module):
    def __init__(self, n_in, filters, k_sizes):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv1d(n_in, filters[0], kernel_size=15, stride=4, padding=5),
            nn.BatchNorm1d(filters[0]),
            nn.GELU(),
            nn.Dropout(STEM_DROPOUT),
        )
        self.enc1 = ConvBlock(filters[0], filters[0], k_sizes[0], BLOCK_DROPOUT)
        self.enc2 = ConvBlock(filters[0], filters[1], k_sizes[1], BLOCK_DROPOUT)
        self.enc3 = ConvBlock(filters[1], filters[2], k_sizes[2], BLOCK_DROPOUT)

    def forward(self, x):
        stem = self.stem(x)
        e1 = self.enc1(stem)
        e2 = self.enc2(e1)
        e3 = self.enc3(e2)
        return stem, e3


class DecoderBlock(nn.Module):
    def __init__(self, in_c, out_c, kernel, dropout=0.2):
        super().__init__()
        padding = kernel // 2
        self.conv = nn.Conv1d(in_c, out_c, kernel_size=kernel, padding=padding)
        self.bn = nn.BatchNorm1d(out_c)
        self.act = nn.GELU()
        self.drop = nn.Dropout(dropout)

    def forward(self, x):
        return self.drop(self.act(self.bn(self.conv(x))))


class CorrEncoderNet(nn.Module):
    def __init__(self, n_primary_channels=1, n_aux_channels=0):
        super().__init__()
        filters = FILTERS
        k_sizes = K_SIZES
        self.n_primary_channels = n_primary_channels
        self.n_aux_channels = n_aux_channels

        self.encoder_primary = SingleEncoder(n_primary_channels, filters, k_sizes)
        if n_aux_channels > 0:
            self.encoder_aux = SingleEncoder(n_aux_channels, filters, k_sizes)
            fusion_in = filters[2] * 2
        else:
            self.encoder_aux = None
            fusion_in = filters[2]

        self.fusion = nn.Sequential(
            nn.Conv1d(fusion_in, filters[2], kernel_size=1),
            nn.BatchNorm1d(filters[2]),
            nn.GELU(),
            nn.Dropout(FUSION_DROPOUT),
        )
        self.dec1 = DecoderBlock(filters[2], filters[1], k_sizes[2], BLOCK_DROPOUT)
        self.dec2 = DecoderBlock(filters[1], filters[0], k_sizes[1], BLOCK_DROPOUT)
        self.dec3 = DecoderBlock(filters[0], filters[0], k_sizes[0], BLOCK_DROPOUT)
        self.head = nn.ConvTranspose1d(filters[0], 1, kernel_size=15, stride=4, padding=5)
        self.apply(init_weights)

    def forward(self, x_primary, x_aux=None):
        stem_primary, feat_primary = self.encoder_primary(x_primary)
        if self.encoder_aux is not None:
            if x_aux is None:
                raise ValueError("x_aux is required when n_aux_channels > 0.")
            _, feat_aux = self.encoder_aux(x_aux)
            fused_input = torch.cat([feat_primary, feat_aux], dim=1)
        else:
            fused_input = feat_primary

        fused = self.fusion(fused_input)
        dec = self.dec3(self.dec2(self.dec1(fused)))
        out = self.head(dec + stem_primary)
        if out.shape[-1] != x_primary.shape[-1]:
            out = F.interpolate(out, size=x_primary.shape[-1], mode="linear", align_corners=False)
        return out



# ---------------------------------------------------------------------------
# Recordings: the GT at 30 Hz (as 261107) and the corr-encoder's input at 125 Hz (as 261106)
# ---------------------------------------------------------------------------

def gt_30hz(ppg: np.ndarray, resp: np.ndarray, ppg_fs: float, resp_fs: float) -> np.ndarray:
    """The GT of a recording at 30 Hz, cut to the shorter of PPG and GT at 30 Hz (as
    reference_recordings of 261107)."""
    ppg30, resp30 = to_fs(ppg, ppg_fs), to_fs(resp, resp_fs)
    return resp30[:min(len(ppg30), len(resp30))]


def capnobase_correncoder_input(csv_dir: str, case_id: str) -> Optional[Tuple[np.ndarray, np.ndarray]]:
    """PPG and CO2 of a case at 125 Hz as load_capnobase_dataset of 261106 reads them."""
    signal_path = os.path.join(csv_dir, f"{case_id}_8min_signal.csv")
    param_path = os.path.join(csv_dir, f"{case_id}_8min_param.csv")
    if not (os.path.isfile(signal_path) and os.path.isfile(param_path)):
        return None
    sig_df = pd.read_csv(signal_path)
    param_df = pd.read_csv(param_path)
    orig_fs = int(param_df["samplingrate_pleth"].values[0])
    ppg_raw = sig_df["pleth_y"].values.astype(np.float64)
    target_raw = sig_df["co2_y"].values.astype(np.float64)
    return maybe_resample(ppg_raw, orig_fs, FS_METRIC), maybe_resample(target_raw, orig_fs, FS_METRIC)


def dataset_recordings(dataset_key: str, tags: Sequence[str]) -> Tuple[Dict[str, List[dict]], List[dict]]:
    """Subject -> recordings in the corr-encoder's order: {name, gt (30 Hz), ppg125, resp125}; only the
    subjects whose safe_tag is among tags (the checkpoint names)."""
    cfg = DATASET_CONFIGS[dataset_key]
    tags = set(tags)

    def wanted_subject(subject: str) -> bool:
        return safe_tag(subject) in tags
    out: Dict[str, List[dict]] = {}
    problems: List[dict] = []
    if dataset_key == "capnobase":
        csv_dir = Path(cfg["data_path"])
        for case_id in capnobase_case_ids(csv_dir):
            if not wanted_subject(case_id):
                continue
            try:
                ppg, co2, _ = read_capnobase_signals(capnobase_file(csv_dir, case_id, "signal"))
                param_path = capnobase_file(csv_dir, case_id, "param")
                param = read_capnobase_fields(param_path) if param_path else {}
                ppg_fs, _ = _sampling_rate(param, "pleth", cfg["default_fs"])
                co2_fs, _ = _sampling_rate(param, "co2", ppg_fs)
                ce_input = capnobase_correncoder_input(str(csv_dir), case_id)
                if ce_input is None:
                    raise FileNotFoundError(f"{case_id}_8min_signal.csv / _param.csv (the corr-encoder's files)")
                out[case_id] = [{"name": case_id, "gt": gt_30hz(ppg, co2, ppg_fs, co2_fs),
                                 "ppg125": ce_input[0], "resp125": ce_input[1]}]
            except Exception as exc:
                problems.append({"Subject": case_id, "Status": f"not read: {exc}"})
    elif dataset_key == "bidmc":
        mat = sio.loadmat(cfg["data_path"])
        for index, record in enumerate(mat["data"][0]):
            subject = f"{index + 1:02d}"
            if not wanted_subject(subject):
                continue
            try:
                ppg = np.asarray(record["ppg"][0, 0]["v"]).squeeze()
                imp = np.asarray(record["ref"][0, 0]["resp_sig"][0, 0]["imp"][0, 0]["v"]).squeeze()
                fs = float(cfg["default_fs"])
                out[subject] = [{"name": subject, "gt": gt_30hz(ppg, imp, fs, fs), "ppg125": ppg, "resp125": imp}]
            except Exception as exc:
                problems.append({"Subject": subject, "Status": f"not read: {exc}"})
    else:
        for subject, items in steam2_session_files(Path(cfg["ppg_dir"]), Path(cfg["rsp_dir"])).items():
            if not wanted_subject(subject):
                continue
            try:
                recs = []
                for name, ppg_path, rsp_path in items:
                    ppg = read_eeglab_channel(ppg_path, "PPG", cfg["default_fs"])
                    rsp = read_eeglab_channel(rsp_path, "RSP", cfg["default_fs"])
                    pieces, _ = steam2_session_pieces(ppg, rsp)
                    for k, (p, r, p_fs, r_fs) in enumerate(pieces):
                        recs.append({"name": f"{name}_piece{k + 1}", "gt": gt_30hz(p, r, p_fs, r_fs),
                                     "ppg125": maybe_resample(p, p_fs, FS_METRIC),
                                     "resp125": maybe_resample(r, r_fs, FS_METRIC)})
                out[subject] = recs
            except Exception as exc:
                problems.append({"Subject": subject, "Status": f"not read: {exc}"})
    return out, problems


# ---------------------------------------------------------------------------
# Corr-encoder windows and outputs
# ---------------------------------------------------------------------------

def correncoder_windows(ppg_raw: np.ndarray, target_raw: np.ndarray, fs: int = FS_METRIC) -> Optional[dict]:
    """The windows of one recording as preprocess_subject_windows of 261106 makes them (same steps),
    and where they lie: low (PPG 0.1-0.5 Hz windows), y (reference windows, training sign), qc,
    positions (samples of every window on the recording's 125 Hz axis), flipped, length."""
    ppg_raw = sanitize_signal(ppg_raw)
    target_raw = sanitize_signal(target_raw)
    m = min(len(ppg_raw), len(target_raw))
    if m < int(WIN_SEC * fs):
        return None
    ppg_raw = ppg_raw[:m]
    target_raw = target_raw[:m]
    valid = np.isfinite(ppg_raw) & np.isfinite(target_raw)
    index = np.flatnonzero(valid)
    ppg_raw = ppg_raw[valid]
    target_raw = target_raw[valid]
    if len(ppg_raw) < int(WIN_SEC * fs):
        return None
    ppg_raw = correct_polarity_skew(ppg_raw)
    p_low = bandpass(ppg_raw, fs, 0.1, 0.5)
    target_f = bandpass(target_raw, fs, 0.1, 0.5)
    r = safe_pearsonr(p_low[:len(target_f)], target_f[:len(p_low)])
    flipped = bool(np.isfinite(r) and r < 0)  # what align_phase_only does
    target_f = align_phase_only(p_low, target_f)
    low_win = extract_windows_stride(p_low, fs)
    y_win = extract_windows_stride(target_f, fs)
    n_win = min(len(low_win), len(y_win))
    if n_win == 0:
        return None
    width = int(round(WIN_SEC * fs))
    step = int(round(STRIDE_SEC * fs))
    positions = np.stack([index[i * step:i * step + width] for i in range(n_win)])
    qc = np.array([check_quality_real_world(low_win[i], y_win[i], fs) for i in range(n_win)], dtype=bool)
    return {"low": np.asarray(low_win[:n_win], dtype=np.float32), "y": np.asarray(y_win[:n_win], dtype=np.float32),
            "qc": qc, "positions": positions, "flipped": flipped, "length": m}


def load_model(path: Path) -> nn.Module:
    model = CorrEncoderNet(n_primary_channels=1, n_aux_channels=0)  # build_model_from_config of 01_low
    model.load_state_dict(torch.load(str(path), map_location=device))
    return model.to(device).eval()


def predict(model: nn.Module, low_windows: np.ndarray) -> np.ndarray:
    """Model output of every window: the PPG window z-scored (clip 5) as build_split_arrays of 261106."""
    x = instance_zscore_normalize(low_windows, clip_value=INPUT_CLIP_VALUE)
    outputs = []
    with torch.no_grad():
        for start in range(0, len(x), INFERENCE_BATCH):
            batch = torch.tensor(x[start:start + INFERENCE_BATCH], dtype=torch.float32).unsqueeze(1).to(device)
            with autocast_context():
                out = model(batch, None)
            outputs.append(out.squeeze(1).float().cpu().numpy())
    return np.concatenate(outputs, axis=0)


def own_scores(pred_windows: List[np.ndarray], true_windows: List[np.ndarray]) -> dict:
    """The run's 30 s scores (run_dataset_ablation of 261106) of the windows that passed the quality check."""
    if not pred_windows:
        return {"Waveform_Corr_30sec_Mean": math.nan, "BPM_MAE": math.nan, "Waveform_MAE_30sec_Mean": math.nan,
                "N_QC_Windows": 0}
    y = instance_zscore_normalize(np.stack(true_windows), clip_value=None)
    segments = evaluate_30sec_segments(np.stack(pred_windows), y)
    corr = [s["corr"] for s in segments if np.isfinite(s["corr"])]
    mae = [s["mae"] for s in segments if np.isfinite(s["mae"])]
    bpm = [s["bpm_err"] for s in segments if np.isfinite(s["bpm_err"])]
    return {"Waveform_Corr_30sec_Mean": float(np.mean(corr)) if corr else math.nan,
            "BPM_MAE": float(np.mean(bpm)) if bpm else math.nan,
            "Waveform_MAE_30sec_Mean": float(np.mean(mae)) if mae else math.nan,
            "N_QC_Windows": len(pred_windows)}


def unit_range(window: np.ndarray) -> np.ndarray:
    low, high = float(np.min(window)), float(np.max(window))
    return (window - low) / (high - low) if high > low else np.zeros_like(window)


# ---------------------------------------------------------------------------
# One dataset
# ---------------------------------------------------------------------------

def dataset_duty(key: str, ce_root: Path, out_root: Path) -> dict:
    label = DATASET_CONFIGS[key]["label"]
    models_dir = ce_root / label / "Models"
    stored_path = ce_root / label / "Metrics" / STORED_PER_SUBJECT
    suffix = f"_{MODEL_NAME}_best.pth"
    checkpoints = {path.name[: -len(suffix)]: path for path in sorted(models_dir.glob(f"*{suffix}"))} \
        if models_dir.is_dir() else {}
    empty = {"duty": pd.DataFrame(), "windows": pd.DataFrame(), "check": pd.DataFrame()}
    if not checkpoints:
        print(f"[{label}] no checkpoints in {models_dir}; left out", flush=True)
        return {**empty, "coverage": pd.DataFrame([{"Dataset": label,
                                                    "Status": f"no checkpoints in {models_dir}; not analysed"}])}
    coverage = []
    stored = pd.read_csv(stored_path, dtype={"Subject": str}) if stored_path.is_file() else pd.DataFrame()
    if not stored.empty:
        stored = stored[stored["Model"] == MODEL_NAME].drop_duplicates("Subject", keep="last").set_index("Subject")
    else:
        coverage.append({"Dataset": label, "Status": f"{stored_path} not found: no check against the run's scores"})
    print(f"[{label}] {len(checkpoints)} checkpoints in {models_dir}; reading the data files ...", flush=True)
    recordings, problems = dataset_recordings(key, list(checkpoints))
    coverage += [{"Dataset": label, **p} for p in problems]
    found = {safe_tag(subject) for subject in recordings}
    coverage += [{"Dataset": label, "Subject": tag, "Status": "checkpoint without a subject in the data files"}
                 for tag in sorted(set(checkpoints) - found)]
    pred_dir = out_root / label / "Predictions"
    fold_dir = out_root / label / "Fold_Results"
    pred_dir.mkdir(parents=True, exist_ok=True)
    fold_dir.mkdir(parents=True, exist_ok=True)

    subject_rows, window_rows, check_rows = [], [], []
    for subject in sorted(recordings):
        model = load_model(checkpoints[safe_tag(subject)])
        pooled, qc_pred, qc_true, arrays, names = [], [], [], {}, []
        n_windows = n_qc = n_flipped = 0
        for k, rec in enumerate(recordings[subject]):
            names.append(rec["name"])
            reference = official_reference(rec["gt"])
            arrays[f"recording{k}_reference"] = np.asarray(rec["gt"], dtype=np.float32)
            win = correncoder_windows(rec["ppg125"], rec["resp125"])
            values125 = np.full(len(rec["ppg125"]), np.nan)
            qc125 = np.zeros(len(rec["ppg125"]))
            raw125 = np.full(len(rec["ppg125"]), np.nan)
            if win is not None:
                out = predict(model, win["low"])
                sign = -1.0 if win["flipped"] else 1.0  # training sign undone
                for i in range(len(out)):
                    pos = win["positions"][i]
                    raw125[pos] = sign * out[i]
                    values125[pos] = unit_range(sign * out[i])
                    qc125[pos] = float(win["qc"][i])
                qc_pred += [out[i] for i in np.flatnonzero(win["qc"])]
                qc_true += [win["y"][i] for i in np.flatnonzero(win["qc"])]
                n_windows += len(out)
                n_qc += int(win["qc"].sum())
                n_flipped += int(win["flipped"])
            times = np.arange(len(reference)) / FS30
            track = Track(values125, float(FS_METRIC)).at(times)
            qc30 = np.nan_to_num(Track(qc125, float(FS_METRIC)).at(times), nan=0.0)
            prediction = lowpass_official(track)
            rows = duty_windows(reference, prediction)
            for r in rows:
                a = int(round(r["Start_Sec"] * FS30))
                r["QC_Share"] = float(qc30[a:a + DUTY_WINDOW].mean())
            pooled += rows
            window_rows += [{"Dataset": label, "Model": MODEL_LABEL, "Subject": subject, "Recording": rec["name"], **r}
                            for r in rows]
            arrays[f"recording{k}_output"] = Track(values125, float(FS_METRIC)).at(
                np.arange(len(rec["gt"])) / FS30).astype(np.float32)
            arrays[f"recording{k}_qc"] = np.nan_to_num(Track(qc125, float(FS_METRIC)).at(
                np.arange(len(rec["gt"])) / FS30), nan=0.0).astype(np.float32)
            arrays[f"recording{k}_output_125hz"] = raw125.astype(np.float32)
            arrays[f"recording{k}_flipped"] = np.array(bool(win["flipped"]) if win is not None else False)
        np.savez_compressed(pred_dir / f"{subject}_outputs.npz", recording_names=np.array(names), **arrays)

        qc_rows = [r for r in pooled if r["Valid"] and r["QC_Share"] >= 1.0]
        row = {"Dataset": label, "Model": MODEL_LABEL, "Subject": subject, "Recordings": len(recordings[subject]),
               **duty_summary(pooled),
               **{f"QC_{name}": value for name, value in duty_summary(qc_rows).items()
                  if name not in ("N_Windows_Skipped",)},
               "N_Model_Windows": n_windows, "N_QC_Model_Windows": n_qc, "Recordings_Sign_Inverted": n_flipped}
        subject_rows.append(row)
        scores = own_scores(qc_pred, qc_true)
        check = {"Dataset": label, "Subject": subject, "N_QC_Windows": scores["N_QC_Windows"]}
        for measure in ("Waveform_Corr_30sec_Mean", "BPM_MAE", "Waveform_MAE_30sec_Mean"):
            stored_value = float(stored.at[subject, measure]) if (not stored.empty and subject in stored.index
                                                                  and measure in stored.columns) else math.nan
            check[f"{measure}_Stored"] = stored_value
            check[f"{measure}_From_New_Outputs"] = scores[measure]
            check[f"{measure}_Difference"] = scores[measure] - stored_value
        check_rows.append(check)
        (fold_dir / f"{subject}_summary.json").write_text(json.dumps(
            {**{k: (None if isinstance(v, float) and not math.isfinite(v) else v) for k, v in row.items()},
             "check": {k: (None if isinstance(v, float) and not math.isfinite(v) else v) for k, v in check.items()},
             "checkpoint": str(checkpoints[safe_tag(subject)])}, indent=2), encoding="utf-8")
        print(f"  [{label}] {subject}: {n_windows} windows ({n_qc} passed QC), duty r {row['Duty_Cycle_Corr']:.3f}, "
              f"duty MAE {row['Duty_MAE']:.3f} | check corr30 {check['Waveform_Corr_30sec_Mean_Stored']:.4f} -> "
              f"{check['Waveform_Corr_30sec_Mean_From_New_Outputs']:.4f}", flush=True)
        del model
    return {"duty": pd.DataFrame(subject_rows), "windows": pd.DataFrame(window_rows),
            "check": pd.DataFrame(check_rows), "coverage": pd.DataFrame(coverage)}


# ---------------------------------------------------------------------------
# Summary, workbook, main
# ---------------------------------------------------------------------------

SUMMARY_MEASURES = ("Duty_Cycle_Corr", "Duty_MAE", "Duty_Bias", "Duty_Reference_Mean", "Duty_Predicted_Mean",
                    "QC_Duty_Cycle_Corr", "QC_Duty_MAE", "QC_Duty_Bias")


def summary_table(duty: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for key in DATASET_ORDER:
        label = DATASET_CONFIGS[key]["label"]
        part = duty[duty["Dataset"] == label] if not duty.empty else pd.DataFrame()
        if part.empty:
            continue
        row = {"Dataset": label, "Model": MODEL_LABEL, "N_Subjects": int(part["Subject"].nunique())}
        for measure in SUMMARY_MEASURES:
            m, s, md, n = mean_sd_median(part[measure]) if measure in part else (math.nan, math.nan, math.nan, 0)
            row[measure] = f"{m:.3f} ± {s:.3f} ({md:.3f})" if n else ""
        for measure in SUMMARY_MEASURES:
            m, s, md, n = mean_sd_median(part[measure]) if measure in part else (math.nan, math.nan, math.nan, 0)
            row.update({f"{measure}_Mean": m, f"{measure}_SD": s, f"{measure}_Median": md, f"{measure}_N": n})
        rows.append(row)
    return pd.DataFrame(rows)


def notes() -> pd.DataFrame:
    lines = [
        ("Model", "261106 deep corr-encoder (CorrEncoderNet, 0.1-0.5 Hz PPG input, model 01_low), the LOSO "
                  "checkpoint of every test subject; outputs made again, nothing trained"),
        ("Windows", "non-overlapping 10 s windows of every recording (125 Hz) as the run cut them; every window gets "
                    "an output (the run scored only those that passed its quality check)"),
        ("Duty cycle", "share of samples >= 0.5 in every 60 s window (moved by 20 s) of the [0, 1] reference and "
                       "model waveform (official RespDiff code; rule of 261107_DutyCycle_RespDiff_DeepCorrencoder)"),
        ("Reference", "GT of the data file, 30 Hz, low-pass 1 Hz (Butterworth 8, zero phase), 5 s segments each "
                      "min-max scaled to [0, 1], joined"),
        ("Corr-encoder waveform", "output of every 10 s window with the training sign undone, min-max scaled to "
                                  "[0, 1] in its window, at the nearest 30 Hz sample, low-pass as the reference; "
                                  "no time shift"),
        ("Duty_Cycle_Corr", "Pearson r of reference and model duty cycle over all windows of the subject"),
        ("Duty_MAE / Duty_Bias", "mean |model - reference| / mean (model - reference) of the window duty cycles"),
        ("QC_*", "the same over the 60 s windows whose every sample lies in a 10 s window that passed the "
                 "run's quality check (QC_Share = 1 in Duty_Windows)"),
        ("Check", "the run's own 30 s scores computed again from the new outputs of the windows that passed the "
                  "quality check, next to the stored ones (Metrics\\LOSO_CorrEncoder_Ablation_PerSubject.csv)"),
        ("Summary", "Mean ± SD (Median) over subjects"),
    ]
    return pd.DataFrame(lines, columns=["Item", "Description"])


def default_results_root() -> Path:
    """The folder of this file, or else the current folder, when it is named
    261106_DeepCorrencoder_PaperStyle_All; otherwise RESULTS_ROOT."""
    candidates = []
    try:
        candidates.append(Path(os.path.abspath(__file__)).parent)
    except NameError:  # run line by line in a console
        pass
    candidates.append(Path.cwd())
    for folder in candidates:
        if folder.name.lower() == RESULTS_FOLDER_NAME.lower():
            return folder
    return RESULTS_ROOT


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="261107 duty cycle of the 261106 deep corr-encoder "
                                                 "(outputs made again from its checkpoints)")
    parser.add_argument("--datasets", nargs="+", choices=list(DATASET_ORDER), default=list(DATASET_ORDER))
    parser.add_argument("--results-root", type=Path, default=None,
                        help=f"results folder of 261106_DeepCorrencoder_PaperStyle_All (default: this file's "
                             f"folder when named {RESULTS_FOLDER_NAME}, else {RESULTS_ROOT})")
    parser.add_argument("--output-root", type=Path, default=None,
                        help=f"default: <results folder>\\{CE_FOLDER}\\{DUTY_FOLDER}")
    parser.add_argument("--capnobase-path", type=Path, default=None)
    parser.add_argument("--bidmc-path", type=Path, default=None)
    parser.add_argument("--steam2-ppg-dir", type=Path, default=None)
    parser.add_argument("--steam2-rsp-dir", type=Path, default=None)
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> None:
    args = parse_args(argv)
    for key, path in (("capnobase", args.capnobase_path), ("bidmc", args.bidmc_path)):
        if path:
            DATASET_CONFIGS[key]["data_path"] = str(path)
    if args.steam2_ppg_dir:
        DATASET_CONFIGS["steam2"]["ppg_dir"] = str(args.steam2_ppg_dir)
    if args.steam2_rsp_dir:
        DATASET_CONFIGS["steam2"]["rsp_dir"] = str(args.steam2_rsp_dir)
    results_root = Path(args.results_root or default_results_root()).resolve()
    ce_root = results_root / CE_FOLDER
    out = Path(args.output_root or ce_root / DUTY_FOLDER).resolve()
    out.mkdir(parents=True, exist_ok=True)
    print(f"Corr-encoder results: {ce_root}\nOutputs: {out}\nDevice: {device}", flush=True)
    parts = {"duty": [], "windows": [], "check": [], "coverage": []}
    for key in DATASET_ORDER:
        if key not in args.datasets:
            continue
        try:
            result = dataset_duty(key, ce_root, out)
        except Exception as exc:
            print(f"[{DATASET_CONFIGS[key]['label']}] not analysed: {type(exc).__name__}: {exc}", flush=True)
            parts["coverage"].append(pd.DataFrame([{"Dataset": DATASET_CONFIGS[key]["label"],
                                                    "Status": f"not analysed: {type(exc).__name__}: {exc}"}]))
            continue
        for name in parts:
            if not result[name].empty:
                parts[name].append(result[name])
    frames = {name: pd.concat(p, ignore_index=True) if p else pd.DataFrame() for name, p in parts.items()}
    tables = {"Summary": summary_table(frames["duty"]), "CorrEncoder261106_Duty": frames["duty"],
              "Check": frames["check"], "Duty_Windows": frames["windows"], "Coverage": frames["coverage"],
              "Notes": notes()}
    for name, frame in tables.items():
        frame.to_csv(out / f"{name}.csv", index=False, encoding="utf-8-sig")
    write_excel(out / "261107_DutyCycle_DeepCorrencoder_261106.xlsx", tables)
    settings = {"results_root": str(results_root), "model": MODEL_NAME, "device": str(device),
                "duty_window_sec": DUTY_WINDOW / FS30, "duty_hop_sec": DUTY_HOP / FS30,
                "duty_threshold": DUTY_THRESHOLD, "segment_sec": SEGMENT / FS30,
                "lowpass_hz": DUTY_LOWPASS_HZ, "lowpass_order": DUTY_LOWPASS_ORDER,
                "correncoder_window_sec": WIN_SEC, "correncoder_fs": FS_METRIC}
    (out / "settings.json").write_text(json.dumps(settings, indent=2), encoding="utf-8")
    print("\nMean ± SD (Median) over subjects:")
    for _, row in tables["Summary"].iterrows():
        print(f"  {row['Dataset']:<9s} n={row['N_Subjects']:<3d} duty r {row['Duty_Cycle_Corr']} | "
              f"duty MAE {row['Duty_MAE']} | QC windows: duty MAE {row['QC_Duty_MAE']}")
    check = frames["check"]
    if not check.empty:
        diff = check["Waveform_Corr_30sec_Mean_Difference"].abs()
        print(f"Check (run's 30 s corr, stored vs new outputs): largest difference {diff.max():.2g} "
              f"over {int(diff.notna().sum())} subjects")
    print(f"\nSaved under: {out}")


if __name__ == "__main__":
    main()
