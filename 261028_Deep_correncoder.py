r"""261028_Deep_correncoder: the deep corr-encoder of Davies & Mandic, "Rapid
extraction of respiratory waveforms from photoplethysmography: A deep
corr-encoder approach", Biomed. Signal Process. Control 85 (2023) 104992,
doi:10.1016/j.bspc.2023.104992, as published: the authors' training code
(correncoder_capnobase_training.py, Harry J Davies, 2023) and the data
preparation and evaluation of the paper, on CapnoBase, BIDMC and STEAM2
(leave-one-subject-out cross-validation); results as respiratory-rate MAE and
waveform PCC. One file: it reads the data files itself and saves every result
under PROJECT_ROOT (the folder of this file).

Model (paper Section 2.2; the authors' code, unchanged)
-------------------------------------------------------
3 convolution "encoder" layers with 8 kernels each of sizes 150, 75 and 50
samples, mirrored by 3 transpose convolution "decoder" layers; ReLU on the
first two (and last two) layers, sigmoid on the innermost layers; dropout 0.5
on the encoder layers; padding 20, 20 and 10, mirrored in the decoder. Input
PPG and output respiratory waveform: 288 samples (9.6 s at 30 Hz).
Training (the authors' code): mean squared error, Adam with learning rate
1e-3, 80 epochs, batch 30; the training windows are shuffled once and the
batches taken in that order every epoch; no validation, the model after epoch
80 is used. The training loss, test loss and zero-crossing breaths error
(breaths_per_min_zc) that the authors' script prints every epoch are written
to the training log. Seed 55, set at the start of every fold (the authors'
script sets it once before the folds) so that every fold can be resumed.

Data (paper Section 2.1)
------------------------
* CapnoBase (42 subjects, 8 min): finger PPG and capnography (reference),
  downsampled to 30 Hz; BIDMC (53 subjects, 8 min): PPG and impedance
  pneumography (reference), resampled to 30 Hz. Resampling: polyphase with an
  anti-aliasing filter (as MATLAB resample). The PPG is used as recorded (it
  is high-pass filtered by the pulse oximeter); all data are used, artefacts
  are not removed;
* every window of PPG re-scaled to between -1 and 1, every window of the
  reference to between 0 and 1;
* training: every recording divided into non-overlapping 9.6 s windows (480 s
  = 50 per subject);
* testing: a sliding window of 9.6 s shifted 1 s each time (471 test windows
  per 480 s recording); the outputs (and the reference windows) of an
  evaluation window are fused and averaged.
STEAM2 (not in the paper) is prepared in the same way: PPG and respiratory
belt of every session (EEGLAB .mat at 200 Hz, split at boundary events; all
sessions of a person form one subject), resampled to 30 Hz.
Every dataset is trained and tested in its own leave-one-subject-out
cross-validation (for BIDMC the paper reports this "trained exclusively on
BIDMC" result next to training on CapnoBase).

Evaluation (paper Section 2.3) and results
------------------------------------------
Evaluation windows of 30.6 s and 60.6 s (the paper's Table 1 and Section
3.2), i.e. 22 and 52 consecutive test windows, without overlap between
evaluation windows. In each:
* respiratory rate = the frequency of the maximum FFT power (mean removed,
  FFT zero-padded to 16384 points) x 60, breaths/min. Reference rate:
  CapnoBase the manual reference rate of the dataset (rr_co2 of the
  reference file, mean in the window); BIDMC and STEAM2 the same FFT rate of
  the fused reference waveform;
* MAE = |predicted - reference rate|: mean +- SD over windows, mAE (median
  over windows) and mMAE (median of the subject MAEs) with IQR, as the paper;
* PCC = Pearson r of the fused output and reference waveforms; waveform MAE =
  mean |output - reference| (0-1 scale; the paper's waveform error).

Outputs (PROJECT_ROOT = D:\PPG2RSP_RRWaveNet_Inspired\261028_Deep_correncoder)
  {dataset}\Models\model_sub{subject}.pth          model after epoch 80
  {dataset}\Training_Logs\{subject}_training_log.csv  every epoch
  {dataset}\Predictions\{subject}_outputs.npz      30 Hz output, reference and
                                                   PPG of every recording
  {dataset}\Fold_Results\{subject}_windows.csv / _summary.json
  Summary\   correncoder_results.csv, per_subject.csv, windows.csv,
             data_check.csv, run_settings.json,
             261028_Deep_correncoder_results.xlsx

Run (GPU or CPU):                python 261028_Deep_correncoder.py
Resume after an interruption:    --resume (finished folds are kept)
Some datasets:                   --datasets capnobase bidmc
Summary of stored folds only:    --summary-only
Data paths:                      --capnobase-path, --bidmc-path,
                                 --steam2-ppg-dir, --steam2-rsp-dir
"""

import argparse
import csv
import json
import math
import random
import re
import time
from fractions import Fraction
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import scipy.io as sio
import torch
import torch.nn as nn
from scipy.signal import resample_poly


# ---------------------------------------------------------------------------
# Paths and data files
# ---------------------------------------------------------------------------

PROJECT_ROOT = Path(r"D:\PPG2RSP_RRWaveNet_Inspired\261028_Deep_correncoder")
# Every result of this script is written under PROJECT_ROOT.
DEFAULT_RESULTS_ROOT = PROJECT_ROOT

DATASET_CONFIGS = {
    "capnobase": {
        "label": "CapnoBase",
        "data_path": (
            r"D:\STEAM\Multimodal_prediction\PPG-RSP\Opendata"
            r"\Capnobase\data\csv"
        ),
        "default_fs": 300,
        "reference": "CO2 capnogram",
    },
    "bidmc": {
        "label": "BIDMC",
        "data_path": (
            r"D:\STEAM\Multimodal_prediction\PPG-RSP\Preprocessed"
            r"\bidmc-ppg-and-respiration-dataset-1.0.0"
            r"\bidmc-ppg-and-respiration-dataset-1.0.0\bidmc_data.mat"
        ),
        "default_fs": 125,
        "reference": "impedance pneumogram (ref.resp_sig.imp)",
    },
    "steam2": {
        "label": "STEAM2",
        "ppg_dir": (
            r"D:\STEAM\Multimodal_prediction\PPG-RSP\Preprocessed"
            r"\PPG\2back_raw_v2"
        ),
        "rsp_dir": (
            r"D:\STEAM\Multimodal_prediction\PPG-RSP\Preprocessed"
            r"\RSP\2back_raw_v2"
        ),
        # EEGLAB sessions (EEG.data, EEG.srate = 200 Hz); default_fs is used
        # only when a file has no srate.
        "default_fs": 200,
        "reference": "respiratory belt (RSP), as recorded",
    },
}

DATASET_ORDER = ("capnobase", "bidmc", "steam2")


# ---------------------------------------------------------------------------
# The authors' code (correncoder_capnobase_training.py, Harry J Davies, 2023)
# ---------------------------------------------------------------------------

# define number of epochs and batch size
num_epochs = 80
batch_size = 30

#define number of kernels per layer
n_in, n_out = 1, 8
n_out2 = 8
n_out3 = 8
n_outputs = 1

# define kernel lengths, padding, dilation, stride, and dropout
kernel_size = 150
kernel_size2 = 75
kernel_size3 = 50
padding = 20
dilation = 1
stride = 1
dropout_val = 0.5
padding2 = 20
padding3 = 10
dilation2 = 1
dilation3 = 1
stride2 = 1
stride3 = 1

# set a seed for evaluation (optional)
seed_val = 55

# set the learning rate for Adam optimisation
learning_rate = 0.001

#define the model
class Correncoder_model(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer1 = nn.Sequential(
            nn.Conv1d(n_in, n_out, kernel_size=kernel_size, padding=padding),
            nn.ReLU(),
            nn.Dropout(dropout_val)
        )
        self.layer2 = nn.Sequential(
            nn.Conv1d(n_out, n_out2, kernel_size=kernel_size2, padding=padding2),
            nn.ReLU(),
            nn.Dropout(dropout_val)
        )
        self.layer3 = nn.Sequential(
            nn.Conv1d(n_out2, n_out3, kernel_size=kernel_size3, padding=padding3),
            nn.Sigmoid(),
            nn.Dropout(dropout_val)
        )
        self.layer4 = nn.Sequential(
            nn.ConvTranspose1d(n_out3, n_out2, kernel_size=kernel_size3, padding=padding3),
            nn.Sigmoid()
        )
        self.layer5 = nn.Sequential(
            nn.ConvTranspose1d(n_out2, n_out, kernel_size=kernel_size2, padding=padding2),
            nn.ReLU()
        )
        self.layer6 = nn.Sequential(
            nn.ConvTranspose1d(n_out, n_in, kernel_size=kernel_size, padding=padding)
        )

    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = self.layer5(out)
        out = self.layer6(out)
        return out


# a function that gives a rough indication of breaths per minute error by examining the crossings of 0.5 
# this assumes that the respiratory reference is normalised between 0 and 1.
def breaths_per_min_zc(output_array_zc, input_array_zc):
    peak_count_output = []
    peak_count_cap = []
    for ind_output in range(output_array_zc.shape[0]):
        output_array_zc_temp = output_array_zc[ind_output, 0, :]
        input_array_zc_temp = input_array_zc[ind_output, :]
        output_array_zc_temp = output_array_zc_temp - 0.5
        input_array_zc_temp = input_array_zc_temp - 0.5
        zero_crossings_output = ((output_array_zc_temp[:-1] * output_array_zc_temp[1:]) < 0).sum()
        zero_crossings_input = ((input_array_zc_temp[:-1] * input_array_zc_temp[1:]) < 0).sum()
        peak_count_output.append(zero_crossings_output)
        peak_count_cap.append(zero_crossings_input)
        # breaths_per_min_output = (zero_crossings_output / 2)*6.25
    peak_count_output = np.array(peak_count_output)
    peak_count_cap = np.array(peak_count_cap)
    #6.5 is used ot scale up to 1 minute, as each segment here is 60/6.5 seconds long.
    mean_error = ((np.mean(peak_count_output - peak_count_cap)) / 2) * 6.5 
    mean_abs_error = ((np.mean(np.abs(peak_count_output - peak_count_cap))) / 2) * 6.5
    return mean_abs_error, mean_error


# ---------------------------------------------------------------------------
# Data file readers: CapnoBase CSV files, STEAM2 EEGLAB .mat files
# ---------------------------------------------------------------------------

CAPNOBASE_FILE_SUFFIXES = (".csv", ".tab")

_NUMBER_PATTERN = re.compile(r"[-+]?(?:\d+\.?\d*|\.\d+)(?:[eE][-+]?\d+)?")

STEAM2_CHANNELS = {"PPG": ("PPG", "PLETH", "BVP"), "RSP": ("RSP", "RESP", "RESPIRATION", "BELT")}

STEAM2_SESSION_PATTERN = re.compile(r"^(?P<subject>.+?)[_\- ]*session[_\- ]*(?P<session>\d+)$", re.IGNORECASE)

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


# ---------------------------------------------------------------------------
# Data of the paper: 30 Hz, 9.6 s (288-sample) windows
# ---------------------------------------------------------------------------

CORR_FS = 30
SEGMENT_LEN = 288  # 9.6 s
TEST_HOP = 30  # test windows shifted by 1 s
PPG_RANGE = (-1.0, 1.0)  # every PPG window re-scaled to between -1 and 1
REFERENCE_RANGE = (0.0, 1.0)  # every reference window re-scaled to between 0 and 1
# Evaluation windows of the paper: 30.6 s (Table 1) and 60.6 s, i.e. 22 and 52
# consecutive test windows (9.6 s + 1 s per further window).
EVAL_WINDOWS_SEC = (30.6, 60.6)
FFT_POINTS = 16384  # zero-padded FFT: 0.11 breaths/min per bin at 30 Hz
PREDICT_BATCH = 512


def fill_missing(signal: np.ndarray) -> Tuple[np.ndarray, int]:
    """Missing samples (NaN) interpolated linearly; returns the count."""
    x = np.asarray(signal, dtype=np.float64).reshape(-1)
    bad = ~np.isfinite(x)
    if bad.any() and not bad.all():
        index = np.arange(x.size)
        x = x.copy()
        x[bad] = np.interp(index[bad], index[~bad], x[~bad])
    return x, int(bad.sum())


def to_corr_fs(signal: np.ndarray, fs: float) -> Tuple[np.ndarray, int]:
    """Resampled to 30 Hz (polyphase with anti-aliasing, as MATLAB resample)."""
    x, missing = fill_missing(signal)
    ratio = Fraction(CORR_FS / float(fs)).limit_denominator(1000)
    if ratio != 1:
        x = resample_poly(x, ratio.numerator, ratio.denominator)
    return x, missing


def make_recording(name: str, ppg: np.ndarray, resp: np.ndarray, ppg_fs: float, resp_fs: float,
                   rr_times: Optional[np.ndarray] = None, rr_values: Optional[np.ndarray] = None) -> dict:
    ppg30, missing_ppg = to_corr_fs(ppg, ppg_fs)
    resp30, missing_resp = to_corr_fs(resp, resp_fs)
    length = min(len(ppg30), len(resp30))
    return {"name": name, "ppg": ppg30[:length], "resp": resp30[:length], "rr_times": rr_times,
            "rr_values": rr_values, "missing_ppg": missing_ppg, "missing_resp": missing_resp}


def capnobase_subjects(cfg: dict):
    """CapnoBase: PPG (pleth), capnogram (co2) and the manual reference
    respiratory rate of the capnogram (reference file, rr_co2)."""
    csv_dir = Path(cfg["data_path"])
    for case_id in capnobase_case_ids(csv_dir):
        try:
            signal_path = capnobase_file(csv_dir, case_id, "signal")
            if signal_path is None:
                raise FileNotFoundError(f"{case_id}_8min_signal.csv not found in {csv_dir}")
            ppg, co2, _ = read_capnobase_signals(signal_path)
            param_path = capnobase_file(csv_dir, case_id, "param")
            param = read_capnobase_fields(param_path) if param_path else {}
            ppg_fs, _ = _sampling_rate(param, "pleth", cfg["default_fs"])
            co2_fs, _ = _sampling_rate(param, "co2", ppg_fs)
            rr_times = rr_values = None
            reference_path = capnobase_file(csv_dir, case_id, "reference")
            if reference_path is not None:
                fields = read_capnobase_fields(reference_path)
                x_name = _find_field(fields, "rr", "co2", suffix="_x")
                y_name = _find_field(fields, "rr", "co2", suffix="_y")
                if x_name and y_name:
                    rr_times, rr_values = _field_numbers(fields[x_name]), _field_numbers(fields[y_name])
                    count = min(len(rr_times), len(rr_values))
                    rr_times, rr_values = rr_times[:count], rr_values[:count]
            if rr_times is None:
                raise FileNotFoundError(f"{case_id}: no reference respiratory rate (rr_co2) found")
        except Exception as exc:
            print(f"[CapnoBase {case_id}] skipped: {exc}")
            continue
        yield case_id, [make_recording(case_id, ppg, co2, ppg_fs, co2_fs, rr_times, rr_values)]


def bidmc_subjects(cfg: dict):
    """BIDMC: PPG and impedance pneumogram (ref.resp_sig.imp), 125 Hz."""
    mat = sio.loadmat(cfg["data_path"])
    for index, record in enumerate(mat["data"][0]):
        subject = f"{index + 1:02d}"
        try:
            ppg = np.asarray(record["ppg"][0, 0]["v"]).squeeze()
            imp = np.asarray(record["ref"][0, 0]["resp_sig"][0, 0]["imp"][0, 0]["v"]).squeeze()
        except Exception as exc:
            print(f"[BIDMC {subject}] skipped: {exc}")
            continue
        fs = float(cfg["default_fs"])
        yield subject, [make_recording(subject, ppg, imp, fs, fs)]


def steam2_subjects(cfg: dict):
    """STEAM2: PPG and respiratory belt (RSP) of every session (EEGLAB .mat),
    split at boundary events; all sessions of a person form one subject."""
    for subject, items in steam2_session_files(Path(cfg["ppg_dir"]), Path(cfg["rsp_dir"])).items():
        recordings = []
        try:
            for name, ppg_path, rsp_path in items:
                ppg = read_eeglab_channel(ppg_path, "PPG", cfg["default_fs"])
                rsp = read_eeglab_channel(rsp_path, "RSP", cfg["default_fs"])
                pieces, info = steam2_session_pieces(ppg, rsp)
                if abs(info["Length_Difference_Sec"]) > 1.0:
                    print(f"[STEAM2 {subject} {name}] PPG {info['PPG_Sec']:.1f} s vs RSP "
                          f"{info['RSP_Sec']:.1f} s; the common part is used")
                for k, (p, r, p_fs, r_fs) in enumerate(pieces):
                    recordings.append(make_recording(f"{name}_piece{k + 1}", p, r, p_fs, r_fs))
        except Exception as exc:
            print(f"[STEAM2 {subject}] skipped: {exc}")
            continue
        if recordings:
            yield subject, recordings


def load_dataset(dataset_key: str) -> Tuple[Dict[str, List[dict]], List[dict]]:
    """The 30 Hz recordings of every subject and a data check per subject."""
    cfg = DATASET_CONFIGS[dataset_key]
    readers = {"capnobase": capnobase_subjects, "bidmc": bidmc_subjects, "steam2": steam2_subjects}
    data, checks = {}, []
    for subject, recordings in readers[dataset_key](cfg):
        x, y = training_segments(recordings)
        checks.append({
            "Dataset": cfg["label"], "Subject": subject, "Recordings": len(recordings),
            "Seconds": float(sum(len(r["ppg"]) for r in recordings) / CORR_FS),
            "Training_Segments": int(len(x)),
            "Test_Windows": int(sum(len(test_starts(len(r["ppg"]))) for r in recordings)),
            "Constant_PPG_Segments": scale(x, PPG_RANGE)[1], "Constant_Reference_Segments": scale(y, REFERENCE_RANGE)[1],
            "Missing_PPG_Samples": int(sum(r["missing_ppg"] for r in recordings)),
            "Missing_Reference_Samples": int(sum(r["missing_resp"] for r in recordings)),
        })
        if len(x):
            data[subject] = recordings
        else:
            print(f"[{cfg['label']} {subject}] shorter than one 9.6 s window, skipped")
    return data, checks


def scale(segments: np.ndarray, value_range: Tuple[float, float]) -> Tuple[np.ndarray, int]:
    """Every window re-scaled (min-max) to value_range; a constant window is
    set to the lower bound. Returns the count of constant windows."""
    if not len(segments):
        return segments, 0
    low, high = value_range
    minimum = segments.min(axis=1, keepdims=True)
    span = segments.max(axis=1, keepdims=True) - minimum
    constant = span[:, 0] <= 1e-12
    scaled = (segments - minimum) / np.where(span > 1e-12, span, 1.0)
    return low + (high - low) * scaled, int(constant.sum())


def training_segments(recordings: Sequence[dict]) -> Tuple[np.ndarray, np.ndarray]:
    """Training: the recordings divided into non-overlapping 9.6 s segments
    (CapnoBase / BIDMC 480 s: 50 per subject), from the start."""
    xs, ys = [], []
    for recording in recordings:
        count = len(recording["ppg"]) // SEGMENT_LEN
        if count:
            xs.append(recording["ppg"][:count * SEGMENT_LEN].reshape(count, SEGMENT_LEN))
            ys.append(recording["resp"][:count * SEGMENT_LEN].reshape(count, SEGMENT_LEN))
    if not xs:
        return np.empty((0, SEGMENT_LEN)), np.empty((0, SEGMENT_LEN))
    return np.concatenate(xs), np.concatenate(ys)


def test_starts(length: int) -> List[int]:
    """Testing: a sliding window of 9.6 s shifted 1 s each time (CapnoBase 480
    s: 471 test windows per subject)."""
    return list(range(0, length - SEGMENT_LEN + 1, TEST_HOP)) if length >= SEGMENT_LEN else []


# ---------------------------------------------------------------------------
# Training (the authors' loop) and testing
# ---------------------------------------------------------------------------

def set_seed() -> None:
    torch.manual_seed(seed_val)
    random.seed(seed_val)
    np.random.seed(seed_val)


def train_fold(trainX: np.ndarray, trainy: np.ndarray, testX: np.ndarray, testy: np.ndarray,
               epochs: int, device: torch.device) -> Tuple[nn.Module, pd.DataFrame]:
    """The authors' training loop; returns the model after the last epoch and
    what their script prints every epoch (training loss, test loss and
    zero-crossing breaths error on the held-out subject's segments, in
    training mode as in their script; logged only, never used for a choice)."""
    set_seed()
    # shuffle training data (as sklearn.utils.shuffle, global numpy state)
    order = np.arange(len(trainX))
    np.random.shuffle(order)
    trainX, trainy = trainX[order], trainy[order]

    # initialise new model
    model = Correncoder_model().to(device)
    # Loss and optimizer
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    # ensure correct shape, can also transpose here instead of reshaping
    L_in = trainX.shape[-1]
    trainX = trainX.reshape((trainX.shape[0], 1, L_in))
    testX = testX.reshape((testX.shape[0], 1, L_in))

    total_step = trainX.shape[0]

    # transformation of data into torch tensors
    trainXT = torch.from_numpy(trainX.astype('float32')).to(device)
    trainyT = torch.from_numpy(trainy.astype('float32')).to(device)
    testXT = torch.from_numpy(testX.astype('float32')).to(device)
    testyT = torch.from_numpy(testy.astype('float32')).to(device)
    # used for input to the breaths per minute calculator
    input_array = testyT.cpu().detach().numpy()

    history = []
    for epoch in range(epochs):
        started = time.time()
        model.train()
        for i in range(total_step // batch_size):  # split data into batches
            trainXT_seg = trainXT[i * batch_size:(i + 1) * batch_size, :, :]
            trainyT_seg = trainyT[i * batch_size:(i + 1) * batch_size, None]
            # Run the forward pass
            outputs = model(trainXT_seg)

            loss = criterion(outputs, trainyT_seg)

            # Backprop and perform Adam optimisation
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        # calculate test error at the end of each epoch
        with torch.no_grad():
            test_output = model(testXT)
            loss_test = criterion(test_output, testyT[:, None])
        output_array = test_output.cpu().detach().numpy()
        mean_error_bpm = breaths_per_min_zc(output_array, input_array)
        history.append({
            "Epoch": epoch + 1,
            "Training_Loss": float(loss.item()),
            "Test_Loss": float(loss_test.item()),
            "Peaks_Error_Abs": float(mean_error_bpm[0]),
            "Peaks_Error_Bias": float(mean_error_bpm[1]),
            "Seconds": time.time() - started,
        })
    return model, pd.DataFrame(history)


def predict(model: nn.Module, windows: np.ndarray, device: torch.device) -> np.ndarray:
    """Outputs of the trained model (evaluation mode: dropout off)."""
    model.eval()
    outputs = []
    with torch.no_grad():
        for start in range(0, len(windows), PREDICT_BATCH):
            batch = torch.from_numpy(windows[start:start + PREDICT_BATCH, None, :].astype("float32")).to(device)
            outputs.append(model(batch)[:, 0, :].cpu().numpy().astype(np.float64))
    return np.concatenate(outputs) if outputs else np.empty((0, SEGMENT_LEN))


def fuse(windows: np.ndarray, starts: Sequence[int], length: int) -> np.ndarray:
    """Overlapping windows averaged into one waveform."""
    total = np.zeros(length)
    count = np.zeros(length)
    for window, start in zip(windows, starts):
        total[start:start + SEGMENT_LEN] += window
        count[start:start + SEGMENT_LEN] += 1
    out = np.full(length, np.nan)
    out[count > 0] = total[count > 0] / count[count > 0]
    return out


# ---------------------------------------------------------------------------
# Evaluation: respiratory rate (MAE) and waveform (PCC, MAE)
# ---------------------------------------------------------------------------

def fft_rate(signal: np.ndarray) -> float:
    """Respiratory rate (breaths/min): the frequency of the maximum FFT power
    (mean removed) x 60."""
    x = np.asarray(signal, dtype=np.float64)
    x = x - x.mean()
    n = max(FFT_POINTS, x.size)
    power = np.abs(np.fft.rfft(x, n)) ** 2
    if power[1:].max() <= 0:
        return math.nan
    return float(np.fft.rfftfreq(n, 1.0 / CORR_FS)[1 + int(np.argmax(power[1:]))] * 60.0)


def pearson(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    if a.size < 2 or a.std() <= 1e-12 or b.std() <= 1e-12:
        return math.nan
    return float(np.corrcoef(a, b)[0, 1])


def evaluate_recording(recording: dict, starts: Sequence[int], outputs: np.ndarray, references: np.ndarray,
                       window_sec: float) -> List[dict]:
    """Every evaluation window: consecutive test windows fused (outputs and
    references averaged), the respiratory rate of both by FFT (CapnoBase: the
    manual reference rate of the dataset in that span), PCC and waveform MAE."""
    per_window = int(round(window_sec - SEGMENT_LEN / CORR_FS)) + 1
    rows = []
    for index in range(len(starts) // per_window):
        chosen = slice(index * per_window, (index + 1) * per_window)
        first = starts[chosen][0]
        local = [s - first for s in starts[chosen]]
        length = local[-1] + SEGMENT_LEN
        output = fuse(outputs[chosen], local, length)
        reference = fuse(references[chosen], local, length)
        start_sec, end_sec = first / CORR_FS, (first + length) / CORR_FS
        if recording["rr_times"] is not None:
            inside = (recording["rr_times"] >= start_sec) & (recording["rr_times"] < end_sec)
            ref_rr = float(np.mean(recording["rr_values"][inside])) if inside.any() else math.nan
            ref_source = "manual (dataset)"
        else:
            ref_rr, ref_source = fft_rate(reference), "reference waveform FFT"
        pred_rr = fft_rate(output)
        rows.append({
            "Window_Sec": window_sec, "Recording": recording["name"], "Window": index + 1,
            "Start_Sec": start_sec, "End_Sec": end_sec, "Ref_RR": ref_rr, "Ref_RR_Source": ref_source,
            "Pred_RR": pred_rr, "Error": pred_rr - ref_rr, "AE": abs(pred_rr - ref_rr),
            "PCC": pearson(output, reference), "Waveform_MAE": float(np.mean(np.abs(output - reference))),
        })
    return rows


def dataset_metrics(windows: pd.DataFrame) -> dict:
    """Respiratory rate MAE (mean +- SD), mAE / mMAE with IQR (as the paper),
    PCC and waveform MAE of one dataset and evaluation window size."""
    def sd(values) -> float:
        values = np.asarray(values, dtype=float)
        values = values[np.isfinite(values)]
        return float(np.std(values, ddof=1)) if values.size > 1 else math.nan

    have = windows[np.isfinite(windows["AE"])]
    ae, error, pcc = have["AE"].to_numpy(float), have["Error"].to_numpy(float), have["PCC"].to_numpy(float)
    subjects = have.groupby("Subject").agg(MAE=("AE", "mean"), PCC=("PCC", "mean"),
                                           Waveform_MAE=("Waveform_MAE", "mean"))
    return {
        "N_Subjects": int(len(subjects)), "N_Windows": int(len(have)),
        "MAE": float(ae.mean()), "MAE_SD": sd(ae), "RMSE": float(np.sqrt(np.mean(error ** 2))),
        "Bias": float(error.mean()), "Bias_SD": sd(error),
        "mAE": float(np.median(ae)), "mAE_Q1": float(np.percentile(ae, 25)), "mAE_Q3": float(np.percentile(ae, 75)),
        "Subject_MAE_Mean": float(subjects["MAE"].mean()), "Subject_MAE_SD": sd(subjects["MAE"]),
        "mMAE": float(subjects["MAE"].median()), "mMAE_Q1": float(subjects["MAE"].quantile(0.25)),
        "mMAE_Q3": float(subjects["MAE"].quantile(0.75)),
        "PCC_Mean": float(np.nanmean(pcc)), "PCC_SD": sd(pcc), "PCC_Median": float(np.nanmedian(pcc)),
        "Subject_PCC_Mean": float(subjects["PCC"].mean()), "Subject_PCC_SD": sd(subjects["PCC"]),
        "Waveform_MAE_Median": float(subjects["Waveform_MAE"].median()),
        "Waveform_MAE_Q1": float(subjects["Waveform_MAE"].quantile(0.25)),
        "Waveform_MAE_Q3": float(subjects["Waveform_MAE"].quantile(0.75)),
    }


# ---------------------------------------------------------------------------
# Leave-one-subject-out cross-validation
# ---------------------------------------------------------------------------

def fold_paths(results_root: Path, label: str, subject: str) -> Dict[str, Path]:
    root = results_root / label
    return {
        "model": root / "Models" / f"model_sub{subject}.pth",
        "log": root / "Training_Logs" / f"{subject}_training_log.csv",
        "predictions": root / "Predictions" / f"{subject}_outputs.npz",
        "windows": root / "Fold_Results" / f"{subject}_windows.csv",
        "summary": root / "Fold_Results" / f"{subject}_summary.json",
    }


def read_json(path: Path) -> Optional[dict]:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


def run_settings(args: argparse.Namespace) -> dict:
    return {
        "Epochs": int(args.epochs), "Batch_Size": batch_size, "Learning_Rate": learning_rate, "Seed": seed_val,
        "Kernels": [kernel_size, kernel_size2, kernel_size3], "Padding": [padding, padding2, padding3],
        "Dropout": dropout_val, "Sampling_Rate_Hz": CORR_FS, "Window_Samples": SEGMENT_LEN,
        "Test_Shift_Samples": TEST_HOP, "PPG_Range": list(PPG_RANGE), "Reference_Range": list(REFERENCE_RANGE),
        "Evaluation_Windows_Sec": list(EVAL_WINDOWS_SEC),
    }


def run_dataset(args: argparse.Namespace, dataset_key: str, results_root: Path, device: torch.device) -> List[dict]:
    label = DATASET_CONFIGS[dataset_key]["label"]
    print(f"\n[{time.strftime('%H:%M:%S')}] {label}: reading the data ...", flush=True)
    data, checks = load_dataset(dataset_key)
    subjects_all = sorted(data)
    if len(subjects_all) < 2:
        print(f"{label}: fewer than two subjects, skipped")
        return checks
    segments = {}
    for subject in subjects_all:
        x, y = training_segments(data[subject])
        segments[subject] = (scale(x, PPG_RANGE)[0], scale(y, REFERENCE_RANGE)[0])
    print(f"{label}: {len(subjects_all)} subjects, {sum(len(x) for x, _ in segments.values())} training "
          f"windows of 9.6 s at {CORR_FS} Hz (reference: {DATASET_CONFIGS[dataset_key]['reference']})", flush=True)
    subjects = [s for s in subjects_all if not args.subjects or s in {str(x) for x in args.subjects}]
    settings = run_settings(args)
    for subject in subjects:
        paths = fold_paths(results_root, label, subject)
        stored = read_json(paths["summary"]) if args.resume else None
        if stored is not None and stored.get("Settings") == settings and paths["windows"].exists():
            print(f"[{time.strftime('%H:%M:%S')}] {label} {subject}: stored, kept", flush=True)
            continue
        train_subjects = [s for s in subjects_all if s != subject]
        trainX = np.concatenate([segments[s][0] for s in train_subjects])
        trainy = np.concatenate([segments[s][1] for s in train_subjects])
        testX, testy = segments[subject]
        print(f"[{time.strftime('%H:%M:%S')}] {label} test subject {subject}: training on {len(trainX)} windows of "
              f"{len(train_subjects)} subjects, {args.epochs} epochs", flush=True)
        started = time.time()
        model, history = train_fold(trainX, trainy, testX, testy, args.epochs, device)
        training_seconds = time.time() - started

        rows, arrays = [], {}
        for k, recording in enumerate(data[subject]):
            starts = test_starts(len(recording["ppg"]))
            if not starts:
                continue
            x = scale(np.stack([recording["ppg"][s:s + SEGMENT_LEN] for s in starts]), PPG_RANGE)[0]
            y = scale(np.stack([recording["resp"][s:s + SEGMENT_LEN] for s in starts]), REFERENCE_RANGE)[0]
            outputs = predict(model, x, device)
            for window_sec in EVAL_WINDOWS_SEC:
                rows += [{"Dataset": label, "Subject": subject, **row}
                         for row in evaluate_recording(recording, starts, outputs, y, window_sec)]
            length = starts[-1] + SEGMENT_LEN
            arrays[f"recording{k}_output"] = fuse(outputs, starts, length).astype(np.float32)
            arrays[f"recording{k}_reference"] = fuse(y, starts, length).astype(np.float32)
            arrays[f"recording{k}_ppg"] = recording["ppg"][:length].astype(np.float32)
        windows = pd.DataFrame(rows)
        for path in paths.values():
            path.parent.mkdir(parents=True, exist_ok=True)
        # save the PyTorch model files
        torch.save(model.state_dict(), paths["model"])
        history.to_csv(paths["log"], index=False, encoding="utf-8-sig")
        np.savez_compressed(paths["predictions"], recording_names=np.asarray([r["name"] for r in data[subject]]),
                            sampling_rate=np.asarray(CORR_FS), **arrays)
        windows.to_csv(paths["windows"], index=False, encoding="utf-8-sig")
        summary = {"Dataset": label, "Subject": subject, "Settings": settings,
                   "Train_Subjects": len(train_subjects), "Train_Windows": int(len(trainX)),
                   "Training_Seconds": training_seconds}
        for window_sec in EVAL_WINDOWS_SEC:
            part = windows[windows["Window_Sec"] == window_sec] if len(windows) else windows
            tag = f"{window_sec:g}s"
            summary[f"MAE_{tag}"] = float(part["AE"].mean()) if len(part) else math.nan
            summary[f"PCC_{tag}"] = float(part["PCC"].mean()) if len(part) else math.nan
            summary[f"Waveform_MAE_{tag}"] = float(part["Waveform_MAE"].mean()) if len(part) else math.nan
        # Written last: its presence marks the fold as complete.
        paths["summary"].write_text(json.dumps(summary, indent=2), encoding="utf-8")
        tag = f"{EVAL_WINDOWS_SEC[-1]:g}s"
        print(f"[{time.strftime('%H:%M:%S')}] {label} test subject {subject}: {tag} windows MAE "
              f"{summary[f'MAE_{tag}']:.3f} breaths/min, PCC {summary[f'PCC_{tag}']:.3f}, waveform MAE "
              f"{summary[f'Waveform_MAE_{tag}']:.3f}; training {training_seconds / 60:.1f} min", flush=True)
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()
    return checks


def write_excel(path: Path, sheets: Dict[str, pd.DataFrame]) -> None:
    try:
        with pd.ExcelWriter(path, engine="openpyxl") as writer:
            for name, frame in sheets.items():
                frame.to_excel(writer, index=False, sheet_name=name[:31])
    except Exception as exc:
        print(f"Excel export failed ({exc}); the CSV files are complete.")


def summarize(results_root: Path, labels: Sequence[str], checks: pd.DataFrame) -> None:
    summary_root = results_root / "Summary"
    summary_root.mkdir(parents=True, exist_ok=True)
    frames = []
    for label in labels:
        for path in sorted((results_root / label / "Fold_Results").glob("*_windows.csv")):
            frame = pd.read_csv(path, dtype={"Subject": str}, encoding="utf-8-sig")
            if not frame.empty:
                frames.append(frame)
    if not frames:
        print("No fold results to summarize yet.")
        return
    windows = pd.concat(frames, ignore_index=True)
    results, subject_tables = [], []
    for label in labels:
        for window_sec in EVAL_WINDOWS_SEC:
            part = windows[(windows["Dataset"] == label) & np.isclose(windows["Window_Sec"], window_sec)]
            if part.empty:
                continue
            m = dataset_metrics(part)
            results.append({
                "Dataset": label, "Evaluation_Window_Sec": window_sec,
                "Reference_RR": part["Ref_RR_Source"].iloc[0],
                "N_Subjects": m["N_Subjects"], "N_Windows": m["N_Windows"],
                "MAE (mean ± SD over windows)": f"{m['MAE']:.3f} ± {m['MAE_SD']:.3f}",
                "Subject MAE (mean ± SD over subjects)": f"{m['Subject_MAE_Mean']:.3f} ± {m['Subject_MAE_SD']:.3f}",
                "mAE [IQR]": f"{m['mAE']:.2f} ({m['mAE_Q1']:.2f}-{m['mAE_Q3']:.2f})",
                "mMAE [IQR]": f"{m['mMAE']:.2f} ({m['mMAE_Q1']:.2f}-{m['mMAE_Q3']:.2f})",
                "PCC (mean ± SD over windows)": f"{m['PCC_Mean']:.3f} ± {m['PCC_SD']:.3f}",
                "Subject PCC (mean ± SD over subjects)": f"{m['Subject_PCC_Mean']:.3f} ± {m['Subject_PCC_SD']:.3f}",
                "Waveform MAE (median [IQR] over subjects)":
                    f"{m['Waveform_MAE_Median']:.3f} ({m['Waveform_MAE_Q1']:.3f}-{m['Waveform_MAE_Q3']:.3f})",
                **m,
            })
            table = part.groupby("Subject").agg(
                N_Windows=("AE", "size"), Ref_RR_Median=("Ref_RR", "median"), Pred_RR_Median=("Pred_RR", "median"),
                MAE=("AE", "mean"), Bias=("Error", "mean"), PCC=("PCC", "mean"),
                Waveform_MAE=("Waveform_MAE", "mean"),
            ).reset_index()
            table.insert(0, "Evaluation_Window_Sec", window_sec)
            table.insert(0, "Dataset", label)
            subject_tables.append(table)
    results = pd.DataFrame(results)
    per_subject = pd.concat(subject_tables, ignore_index=True)
    results.to_csv(summary_root / "correncoder_results.csv", index=False, encoding="utf-8-sig")
    per_subject.to_csv(summary_root / "per_subject.csv", index=False, encoding="utf-8-sig")
    windows.to_csv(summary_root / "windows.csv", index=False, encoding="utf-8-sig")
    checks.to_csv(summary_root / "data_check.csv", index=False, encoding="utf-8-sig")
    write_excel(summary_root / "261028_Deep_correncoder_results.xlsx",
                {"Results": results, "Per_subject": per_subject, "Windows": windows, "Data_check": checks})
    print("\nDeep corr-encoder (breaths/min; mean ± SD; mAE / mMAE median (IQR) as in the paper):")
    for _, row in results.iterrows():
        print(f"  {row['Dataset']:<10s} {row['Evaluation_Window_Sec']:>4.1f} s  subjects {row['N_Subjects']:>3d}  "
              f"windows {row['N_Windows']:>5d}  MAE {row['MAE (mean ± SD over windows)']}  "
              f"mAE {row['mAE [IQR]']}  mMAE {row['mMAE [IQR]']}  PCC {row['PCC (mean ± SD over windows)']}  "
              f"waveform MAE {row['Waveform MAE (median [IQR] over subjects)']}")
    print(f"\nSaved the summary under: {summary_root}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--results-root", type=Path, default=DEFAULT_RESULTS_ROOT)
    parser.add_argument("--datasets", nargs="+", choices=list(DATASET_ORDER), default=list(DATASET_ORDER))
    parser.add_argument("--subjects", nargs="+", default=None,
                        help="Only these test subjects (quick tests); training still uses all others.")
    parser.add_argument("--epochs", type=int, default=num_epochs, help="Epochs (authors: %(default)s).")
    parser.add_argument("--resume", action="store_true", help="Keep the folds already finished.")
    parser.add_argument("--summary-only", action="store_true")
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--force-cpu", action="store_true")
    parser.add_argument("--threads", type=int, default=None)
    parser.add_argument("--capnobase-path", type=Path, default=None)
    parser.add_argument("--bidmc-path", type=Path, default=None)
    parser.add_argument("--steam2-ppg-dir", type=Path, default=None)
    parser.add_argument("--steam2-rsp-dir", type=Path, default=None)
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> None:
    args = parse_args(argv)
    if args.capnobase_path:
        DATASET_CONFIGS["capnobase"]["data_path"] = str(args.capnobase_path)
    if args.bidmc_path:
        DATASET_CONFIGS["bidmc"]["data_path"] = str(args.bidmc_path)
    if args.steam2_ppg_dir:
        DATASET_CONFIGS["steam2"]["ppg_dir"] = str(args.steam2_ppg_dir)
    if args.steam2_rsp_dir:
        DATASET_CONFIGS["steam2"]["rsp_dir"] = str(args.steam2_rsp_dir)
    if args.threads:
        torch.set_num_threads(int(args.threads))
    results_root = Path(args.results_root).resolve()
    (results_root / "Summary").mkdir(parents=True, exist_ok=True)
    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() and not args.force_cpu else "cpu")
    dataset_keys = [key for key in DATASET_ORDER if key in args.datasets]
    labels = [DATASET_CONFIGS[key]["label"] for key in dataset_keys]
    (results_root / "Summary" / "run_settings.json").write_text(
        json.dumps({**run_settings(args), "Datasets": labels, "Device": str(device)}, indent=2), encoding="utf-8"
    )
    print(f"Deep corr-encoder ({sum(p.numel() for p in Correncoder_model().parameters())} parameters): "
          f"{args.epochs} epochs, batch {batch_size}, learning rate {learning_rate:g}, seed {seed_val}; "
          f"9.6 s windows at {CORR_FS} Hz; device={device}")
    print(f"Results: {results_root}")
    check_path = results_root / "Summary" / "data_check.csv"
    if args.summary_only:
        checks = pd.read_csv(check_path, dtype={"Subject": str}) if check_path.exists() else pd.DataFrame()
    else:
        rows = []
        for key in dataset_keys:
            rows += run_dataset(args, key, results_root, device)
        checks = pd.DataFrame(rows)
    summarize(results_root, labels, checks)


if __name__ == "__main__":
    main()
