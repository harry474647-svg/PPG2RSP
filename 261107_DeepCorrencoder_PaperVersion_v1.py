r"""261107_DeepCorrencoder_PaperVersion_v1: the deep corr-encoder exactly as in
Davies & Mandic, "Rapid extraction of respiratory waveforms from
photoplethysmography: A deep corr-encoder approach", Biomed. Signal Process.
Control 85 (2023) 104992, doi:10.1016/j.bspc.2023.104992, trained and tested
with leave-one-subject-out cross-validation (LOSOCV) on STEAM2, BIDMC and
CapnoBase, every subject scored by respiratory-rate MAE, PCC, waveform MAE
and the duty cycle measures used so far. Everything is written to a new
folder.

Model (paper Section 2.2 and Fig. 1)
------------------------------------
Encoder: 3 one-dimensional convolution layers with 8 kernels each, of sizes
150, 75 and 50 samples; decoder: the mirror, 3 transposed convolution layers
with 8 kernels of sizes 50, 75 and 150 (the last one maps the 8 channels to
the 1-channel output). Padding: the input and output of the first layer by 20,
the output of the second layer by 10 (conv padding 20, 20, 10; mirrored in
the transposed convolutions). Activations: ReLU on the first two layers,
sigmoid on the innermost layers (encoder layer 3, decoder layer 1), ReLU on
the last two layers (Fig. 1; "ReLU was implemented in the shallow layers
considering that the output signal was normalised between 0 and 1"). Dropout
0.5 on the encoder layers. Input PPG and output respiratory waveform: 288
samples (9.6 s at 30 Hz). 18,441 parameters (6 layers, 48 kernels).
  The authors' training code (as used in 261028_Deep_correncoder) has no
  activation after the last transposed convolution; this file follows the
  paper (OUTPUT_RELU = True; False gives the authors' code).
Training (paper; the authors' loop): mean squared error, Adam with learning
rate 1e-3, batch 30, 80 epochs (training stopped at epoch 80, no validation
set), the training windows shuffled once; seed 55 at the start of every fold.

Data (paper Section 2.1)
------------------------
All data are used, artefacts are not removed. PPG and reference resampled to
30 Hz (polyphase with anti-aliasing, as MATLAB resample); every 9.6 s window
of PPG re-scaled to between -1 and 1, every window of the reference to
between 0 and 1.
  CapnoBase (42 subjects, 8 min): finger PPG, capnogram; reference rate the
    manual rate of the dataset (rr_co2 of the reference file).
  BIDMC (53 subjects, 8 min): PPG, impedance pneumogram.
  STEAM2: PPG and respiratory belt of every session (EEGLAB .mat at 200 Hz,
    split at boundary events into pieces; all sessions of a person form one
    subject).
Training windows: every recording divided into non-overlapping 9.6 s windows
from the start (480 s = 50 per subject). Test windows: 9.6 s shifted by 1 s
(471 per 480 s recording); the outputs of overlapping windows are fused
(averaged).

LOSOCV (every dataset on its own, as the paper's CapnoBase validation)
-----------------------------------------------------------------------
For every subject: the model is trained on all other subjects of the dataset
and tested on that subject.

Scores of every test subject
----------------------------
* Respiratory rate and waveform (paper Section 2.3) in evaluation windows of
  30.6 s and 60.6 s (22 and 52 consecutive test windows, fused): rate =
  frequency of the maximum FFT power x 60 (zero-padded FFT, 16384 points);
  reference rate: CapnoBase the manual rate of the dataset in the window,
  BIDMC and STEAM2 the FFT rate of the fused reference window;
  MAE = mean |predicted - reference rate| (breaths/min), PCC = Pearson r of
  the fused output and reference, waveform MAE = mean |output - reference|.
  Dataset tables: MAE mean ± SD, mAE / mMAE with IQR (as the paper), PCC.
* Duty cycle (rule of 261107_DutyCycle_RespDiff_DeepCorrencoder, the official
  RespDiff code, as for RRWaveNet, RespDiff and the 261028 corr-encoder):
  reference = the recording's reference at 30 Hz, low-pass 1 Hz (Butterworth
  8, zero phase), 5 s segments each min-max scaled to [0, 1], joined; model =
  the fused output (no time shift), low-pass 1 Hz; 60 s windows moved by 20 s,
  duty cycle = share of samples >= 0.5; per subject Duty_Cycle_Corr (Pearson
  r of reference and model duty cycle over the windows), Duty_MAE, Duty_Bias,
  Duty_Reference_Mean, Duty_Predicted_Mean. (The paper's inspiratory duty
  cycle counts the samples below 0.5, i.e. 1 - this share: the same absolute
  errors and correlation.)

Results folder (new)
--------------------
  D:\PPG2RSP_RRWaveNet_Inspired\261107_DeepCorrencoder_PaperVersion_v1
  (the folder of this file, or else the current folder, when it is named so;
  otherwise RESULTS_ROOT; --results-root sets another one)
    {Dataset}\Models\model_sub{subject}.pth              model after epoch 80
    {Dataset}\Training_Logs\{subject}_training_log.csv   every epoch
    {Dataset}\Predictions\{subject}_outputs.npz          fused 30 Hz output,
        reference (fused windows), PPG and GT (recording{k}_gt) per recording
    {Dataset}\Fold_Results\{subject}_windows.csv          30.6 / 60.6 s windows
    {Dataset}\Fold_Results\{subject}_duty_windows.csv     60 s duty windows
    {Dataset}\Fold_Results\{subject}_summary.json         (written last)
    Summary\   261107_DeepCorrencoder_PaperVersion_v1_results.xlsx and CSV:
        Results (paper tables), Summary (Mean ± SD (Median) over subjects of
        MAE, PCC, waveform MAE, duty measures), Per_subject, Windows,
        Duty_Windows, Data_check; run_settings.json
A fold whose summary exists with the same settings is kept when the script
runs again (an interrupted run goes on); --rerun trains every fold again.

GPU: as 261028 (the run stops when the GPU is occupied; --allow-cpu trains
on the CPU instead, --force-cpu always on the CPU).

Run (in D:\PPG2RSP_RRWaveNet_Inspired\261107_DeepCorrencoder_PaperVersion_v1):
    python 261107_DeepCorrencoder_PaperVersion_v1.py
Options: --datasets steam2 bidmc capnobase, --subjects, --epochs (paper: 80),
    --rerun, --summary-only, --gpu, --allow-cpu, --force-cpu, --threads,
    --results-root, --capnobase-path, --bidmc-path, --steam2-ppg-dir,
    --steam2-rsp-dir
"""

import argparse
import csv
import gc
import json
import math
import random
import re
import sys
import time
from fractions import Fraction
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import scipy.io as sio
import torch
import torch.nn as nn
from scipy.signal import butter, resample_poly, sosfiltfilt


# ---------------------------------------------------------------------------
# Paths and data files
# ---------------------------------------------------------------------------

RESULTS_FOLDER_NAME = "261107_DeepCorrencoder_PaperVersion_v1"
RESULTS_ROOT = Path(r"D:\PPG2RSP_RRWaveNet_Inspired") / RESULTS_FOLDER_NAME

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

DATASET_ORDER = ("steam2", "bidmc", "capnobase")


# ---------------------------------------------------------------------------
# The authors' code (correncoder_capnobase_training.py, Harry J Davies, 2023); the output layer as the
# paper (OUTPUT_RELU)
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

# ReLU after the last transposed convolution as in the paper (Fig. 1, Section 2.2); the authors' training code
# has no activation there (False).
OUTPUT_RELU = True

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
            nn.ConvTranspose1d(n_out, n_in, kernel_size=kernel_size, padding=padding),
            *([nn.ReLU()] if OUTPUT_RELU else [])
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
# The GPU is used only with at least this much free memory (MB); otherwise the CPU.
MIN_FREE_GPU_MB = 256


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
# GPU memory
# ---------------------------------------------------------------------------

def release_memory(release_traceback: bool = False) -> None:
    """Free unused memory. With release_traceback: also an earlier run stopped
    by an error in the same console (Spyder), whose stored traceback keeps its
    GPU tensors alive."""
    if release_traceback:
        for name in ("last_type", "last_value", "last_traceback"):
            if hasattr(sys, name):
                setattr(sys, name, None)
    gc.collect()
    if torch.cuda.is_available():
        try:
            torch.cuda.empty_cache()
        except RuntimeError:
            pass


def cuda_failure(exc: BaseException) -> bool:
    text = f"{type(exc).__name__}: {exc}"
    return any(word in text for word in ("out of memory", "CUDA", "CUBLAS", "cuDNN", "CUDNN"))


def gpu_report(device: torch.device) -> str:
    """Free / total memory of the GPU and the programs using it (if known)."""
    lines = []
    try:
        free, total = torch.cuda.mem_get_info(device)
        lines.append(f"GPU memory: {free / 2 ** 20:.0f} MB free of {total / 2 ** 20:.0f} MB")
    except Exception:
        pass
    try:
        lines.append(torch.cuda.list_gpu_processes(device))
    except Exception:
        pass
    return "\n".join(lines)


def choose_device(args: argparse.Namespace) -> torch.device:
    """The GPU when it has at least MIN_FREE_GPU_MB free and a test allocation
    succeeds. Otherwise the run stops with the reason (the CPU only with
    --allow-cpu or --force-cpu)."""
    if args.force_cpu:
        return torch.device("cpu")
    if not torch.cuda.is_available():
        reason, device = "PyTorch finds no CUDA GPU (CUDA build of PyTorch and NVIDIA driver needed)", None
    else:
        device = torch.device(f"cuda:{args.gpu}")
        reason = ""
        try:
            mem_get_info = getattr(torch.cuda, "mem_get_info", None)
            if mem_get_info is not None:
                free, total = mem_get_info(device)
                if free < MIN_FREE_GPU_MB * 2 ** 20:
                    reason = (f"only {free / 2 ** 20:.0f} MB of {total / 2 ** 20:.0f} MB free "
                              "(used by other programs)")
            if not reason:
                probe = torch.empty(MIN_FREE_GPU_MB * 2 ** 20 // 4, dtype=torch.float32, device=device)
                nn.functional.dropout(probe[:4096], dropout_val, True)
                torch.cuda.synchronize(device)
                del probe
        except RuntimeError as exc:
            reason = str(exc).splitlines()[0]
        release_memory()
    if not reason:
        print(f"GPU {args.gpu}: {torch.cuda.get_device_name(device)}", flush=True)
        return device
    report = gpu_report(device) if device is not None else ""
    if args.allow_cpu:
        print(f"GPU {args.gpu} not used: {reason}. Training on the CPU (--allow-cpu).", flush=True)
        if report:
            print(report, flush=True)
        return torch.device("cpu")
    print(f"\nGPU {args.gpu} cannot be used: {reason}.", flush=True)
    if report:
        print(report, flush=True)
    print("Free the GPU (end or pause the other training on it, or restart the console: Spyder Consoles > "
          "Restart kernel) and run again; or run with --allow-cpu to train on the CPU (slower).", flush=True)
    raise SystemExit(1)


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
# Duty cycle (copied from 261107_DutyCycle_RespDiff_DeepCorrencoder: official RespDiff code)
# ---------------------------------------------------------------------------

FS30 = CORR_FS
SEGMENT = 5 * FS30  # 5 s segments of the reference, each min-max scaled to [0, 1]
DUTY_WINDOW = 60 * FS30
DUTY_HOP = 20 * FS30
DUTY_THRESHOLD = 0.5
DUTY_LOWPASS_HZ = 1.0
DUTY_LOWPASS_ORDER = 8


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



def recording_duty(recording: dict, fused_output: np.ndarray) -> List[dict]:
    """60 s duty windows of one recording: the recording's reference (30 Hz) against the fused output on
    the same time axis (no shift; NaN after the last test window), both as 261107 treats them."""
    reference = official_reference(recording["resp"])
    prediction = np.full(len(reference), np.nan)
    count = min(len(reference), len(fused_output))
    prediction[:count] = fused_output[:count]
    return duty_windows(reference, lowpass_official(prediction))


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
        "duty": root / "Fold_Results" / f"{subject}_duty_windows.csv",
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
        "Evaluation_Windows_Sec": list(EVAL_WINDOWS_SEC), "Output_ReLU": OUTPUT_RELU,
        "Duty": {"Window_Sec": DUTY_WINDOW / FS30, "Hop_Sec": DUTY_HOP / FS30, "Threshold": DUTY_THRESHOLD,
                 "Segment_Sec": SEGMENT / FS30, "Lowpass_Hz": DUTY_LOWPASS_HZ, "Lowpass_Order": DUTY_LOWPASS_ORDER},
    }


def run_fold(recordings: Sequence[dict], label: str, subject: str, trainX: np.ndarray, trainy: np.ndarray,
             testX: np.ndarray, testy: np.ndarray, epochs: int, device: torch.device):
    """Train the model of one fold and test it on the held-out subject's
    recordings: model, training log, evaluation rows, waveforms, seconds."""
    started = time.time()
    model, history = train_fold(trainX, trainy, testX, testy, epochs, device)
    training_seconds = time.time() - started
    rows, duty_rows, arrays = [], [], {}
    for k, recording in enumerate(recordings):
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
        fused = fuse(outputs, starts, length)
        arrays[f"recording{k}_output"] = fused.astype(np.float32)
        arrays[f"recording{k}_reference"] = fuse(y, starts, length).astype(np.float32)
        arrays[f"recording{k}_ppg"] = recording["ppg"][:length].astype(np.float32)
        arrays[f"recording{k}_gt"] = np.asarray(recording["resp"], dtype=np.float32)
        duty_rows += [{"Dataset": label, "Subject": subject, "Recording": recording["name"], **row}
                      for row in recording_duty(recording, fused)]
    return model.cpu(), history, rows, duty_rows, arrays, training_seconds


def run_dataset(args: argparse.Namespace, dataset_key: str, results_root: Path,
                device: torch.device) -> Tuple[List[dict], torch.device]:
    """LOSOCV of one dataset; returns the data check and the device in use
    (the CPU after a GPU failure)."""
    label = DATASET_CONFIGS[dataset_key]["label"]
    print(f"\n[{time.strftime('%H:%M:%S')}] {label}: reading the data ...", flush=True)
    data, checks = load_dataset(dataset_key)
    subjects_all = sorted(data)
    if len(subjects_all) < 2:
        print(f"{label}: fewer than two subjects, skipped")
        return checks, device
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
        stored = None if args.rerun else read_json(paths["summary"])
        if stored is not None and stored.get("Settings") == settings and paths["windows"].exists() \
                and paths["duty"].exists():
            print(f"[{time.strftime('%H:%M:%S')}] {label} {subject}: stored, kept", flush=True)
            continue
        train_subjects = [s for s in subjects_all if s != subject]
        trainX = np.concatenate([segments[s][0] for s in train_subjects])
        trainy = np.concatenate([segments[s][1] for s in train_subjects])
        testX, testy = segments[subject]
        print(f"[{time.strftime('%H:%M:%S')}] {label} test subject {subject}: training on {len(trainX)} windows of "
              f"{len(train_subjects)} subjects, {args.epochs} epochs ({device.type})", flush=True)
        try:
            fold = run_fold(data[subject], label, subject, trainX, trainy, testX, testy, args.epochs, device)
        except RuntimeError as exc:
            if device.type != "cuda" or not cuda_failure(exc):
                raise
            message = str(exc).splitlines()[0]
            fold = None
        if fold is None:
            print(f"[{time.strftime('%H:%M:%S')}] {label} test subject {subject}: GPU failed ({message}).",
                  flush=True)
            report = gpu_report(device)
            if report:
                print(report, flush=True)
            if not args.allow_cpu:
                print("Stopped: the GPU is occupied by other programs. The finished folds are kept; free the GPU "
                      "and run again (or with --allow-cpu to go on on the CPU).", flush=True)
                raise SystemExit(1)
            print("This fold is run again on the CPU, and so are the following folds (--allow-cpu).", flush=True)
            release_memory()
            device = torch.device("cpu")
            fold = run_fold(data[subject], label, subject, trainX, trainy, testX, testy, args.epochs, device)
        model, history, rows, duty_rows, arrays, training_seconds = fold
        windows = pd.DataFrame(rows)
        duty_frame = pd.DataFrame(duty_rows)
        for path in paths.values():
            path.parent.mkdir(parents=True, exist_ok=True)
        # save the PyTorch model files
        torch.save(model.state_dict(), paths["model"])
        history.to_csv(paths["log"], index=False, encoding="utf-8-sig")
        np.savez_compressed(paths["predictions"], recording_names=np.asarray([r["name"] for r in data[subject]]),
                            sampling_rate=np.asarray(CORR_FS), **arrays)
        windows.to_csv(paths["windows"], index=False, encoding="utf-8-sig")
        duty_frame.to_csv(paths["duty"], index=False, encoding="utf-8-sig")
        summary = {"Dataset": label, "Subject": subject, "Settings": settings,
                   "Train_Subjects": len(train_subjects), "Train_Windows": int(len(trainX)),
                   "Training_Seconds": training_seconds}
        for window_sec in EVAL_WINDOWS_SEC:
            part = windows[windows["Window_Sec"] == window_sec] if len(windows) else windows
            tag = f"{window_sec:g}s"
            summary[f"MAE_{tag}"] = float(part["AE"].mean()) if len(part) else math.nan
            summary[f"PCC_{tag}"] = float(part["PCC"].mean()) if len(part) else math.nan
            summary[f"Waveform_MAE_{tag}"] = float(part["Waveform_MAE"].mean()) if len(part) else math.nan
        summary.update({key: (None if isinstance(value, float) and not math.isfinite(value) else value)
                        for key, value in duty_summary(duty_rows).items()})
        # Written last: its presence marks the fold as complete.
        paths["summary"].write_text(json.dumps(summary, indent=2), encoding="utf-8")
        tag = f"{EVAL_WINDOWS_SEC[-1]:g}s"
        print(f"[{time.strftime('%H:%M:%S')}] {label} test subject {subject}: {tag} windows MAE "
              f"{summary[f'MAE_{tag}']:.3f} breaths/min, PCC {summary[f'PCC_{tag}']:.3f}, waveform MAE "
              f"{summary[f'Waveform_MAE_{tag}']:.3f}, duty MAE {summary['Duty_MAE'] if summary['Duty_MAE'] is not None else math.nan:.3f} "
              f"(r {summary['Duty_Cycle_Corr'] if summary['Duty_Cycle_Corr'] is not None else math.nan:.3f}); "
              f"training {training_seconds / 60:.1f} min", flush=True)
        del model, fold
        release_memory()
    return checks, device


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
    duty_frames = []
    for label in labels:
        for path in sorted((results_root / label / "Fold_Results").glob("*_duty_windows.csv")):
            try:
                frame = pd.read_csv(path, dtype={"Subject": str}, encoding="utf-8-sig")
            except pd.errors.EmptyDataError:
                continue
            if not frame.empty:
                duty_frames.append(frame)
    duty_windows_all = pd.concat(duty_frames, ignore_index=True) if duty_frames else pd.DataFrame()
    duty_subjects = []
    if not duty_windows_all.empty:
        for (label, subject), part in duty_windows_all.groupby(["Dataset", "Subject"], sort=False):
            duty_subjects.append({"Dataset": label, "Subject": subject, **duty_summary(part.to_dict("records"))})
    duty_subjects = pd.DataFrame(duty_subjects)
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
    if not duty_subjects.empty:
        per_subject = per_subject.merge(duty_subjects, on=["Dataset", "Subject"], how="left")
    mean_sd = summary_table(per_subject, labels)
    results.to_csv(summary_root / "correncoder_results.csv", index=False, encoding="utf-8-sig")
    per_subject.to_csv(summary_root / "per_subject.csv", index=False, encoding="utf-8-sig")
    windows.to_csv(summary_root / "windows.csv", index=False, encoding="utf-8-sig")
    checks.to_csv(summary_root / "data_check.csv", index=False, encoding="utf-8-sig")
    duty_windows_all.to_csv(summary_root / "duty_windows.csv", index=False, encoding="utf-8-sig")
    mean_sd.to_csv(summary_root / "summary_mean_sd_median.csv", index=False, encoding="utf-8-sig")
    write_excel(summary_root / "261107_DeepCorrencoder_PaperVersion_v1_results.xlsx",
                {"Summary": mean_sd, "Results": results, "Per_subject": per_subject, "Windows": windows,
                 "Duty_Windows": duty_windows_all, "Data_check": checks})
    print("\nDeep corr-encoder (breaths/min; mean ± SD; mAE / mMAE median (IQR) as in the paper):")
    for _, row in results.iterrows():
        print(f"  {row['Dataset']:<10s} {row['Evaluation_Window_Sec']:>4.1f} s  subjects {row['N_Subjects']:>3d}  "
              f"windows {row['N_Windows']:>5d}  MAE {row['MAE (mean ± SD over windows)']}  "
              f"mAE {row['mAE [IQR]']}  mMAE {row['mMAE [IQR]']}  PCC {row['PCC (mean ± SD over windows)']}  "
              f"waveform MAE {row['Waveform MAE (median [IQR] over subjects)']}")
    print("\nMean ± SD (Median) over subjects:")
    for _, row in mean_sd.iterrows():
        print(f"  {row['Dataset']:<10s} {row['Evaluation_Window_Sec']:>4.1f} s  n={row['N_Subjects']:<3d} "
              f"MAE {row['MAE']} | PCC {row['PCC']} | duty r {row['Duty_Cycle_Corr']} | duty MAE {row['Duty_MAE']}")
    print(f"\nSaved the summary under: {summary_root}")


SUMMARY_MEASURES = (("MAE", 2), ("PCC", 3), ("Waveform_MAE", 3), ("Duty_Cycle_Corr", 3), ("Duty_MAE", 3),
                    ("Duty_Bias", 3), ("Duty_Reference_Mean", 3), ("Duty_Predicted_Mean", 3))


def summary_table(per_subject: pd.DataFrame, labels: Sequence[str]) -> pd.DataFrame:
    """Per dataset and evaluation window: Mean ± SD (Median) over subjects (the duty measures are the same
    in both rows of a dataset: 60 s duty windows)."""
    rows = []
    for label in labels:
        for window_sec in EVAL_WINDOWS_SEC:
            part = per_subject[(per_subject["Dataset"] == label)
                               & np.isclose(per_subject["Evaluation_Window_Sec"], window_sec)]
            if part.empty:
                continue
            row = {"Dataset": label, "Evaluation_Window_Sec": window_sec, "N_Subjects": int(part["Subject"].nunique())}
            stats = {}
            for measure, digits in SUMMARY_MEASURES:
                m, s, md, n = mean_sd_median(part[measure]) if measure in part else (math.nan, math.nan, math.nan, 0)
                row[measure] = f"{m:.{digits}f} ± {s:.{digits}f} ({md:.{digits}f})" if n else ""
                stats.update({f"{measure}_Mean": m, f"{measure}_SD": s, f"{measure}_Median": md, f"{measure}_N": n})
            rows.append({**row, **stats})
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def default_results_root() -> Path:
    """The folder of this file, or else the current folder, when it is named RESULTS_FOLDER_NAME;
    otherwise RESULTS_ROOT."""
    candidates = []
    try:
        candidates.append(Path(__file__).resolve().parent)
    except NameError:  # run line by line in a console
        pass
    candidates.append(Path.cwd())
    for folder in candidates:
        if folder.name.lower() == RESULTS_FOLDER_NAME.lower():
            return folder
    return RESULTS_ROOT


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--results-root", type=Path, default=None,
                        help=f"results folder (default: this file's folder when named {RESULTS_FOLDER_NAME}, "
                             f"else {RESULTS_ROOT})")
    parser.add_argument("--datasets", nargs="+", choices=list(DATASET_ORDER), default=list(DATASET_ORDER))
    parser.add_argument("--subjects", nargs="+", default=None,
                        help="Only these test subjects (quick tests); training still uses all others.")
    parser.add_argument("--epochs", type=int, default=num_epochs, help="Epochs (authors: %(default)s).")
    parser.add_argument("--rerun", action="store_true",
                        help="Train every fold again (default: folds finished with the same settings are kept).")
    parser.add_argument("--summary-only", action="store_true")
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--allow-cpu", action="store_true",
                        help="Train on the CPU when the GPU is not available or occupied (default: stop).")
    parser.add_argument("--force-cpu", action="store_true", help="Always train on the CPU.")
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
    results_root = Path(args.results_root or default_results_root()).resolve()
    (results_root / "Summary").mkdir(parents=True, exist_ok=True)
    release_memory(release_traceback=True)
    device = torch.device("cpu") if args.summary_only else choose_device(args)
    dataset_keys = [key for key in DATASET_ORDER if key in args.datasets]
    labels = [DATASET_CONFIGS[key]["label"] for key in dataset_keys]
    (results_root / "Summary" / "run_settings.json").write_text(
        json.dumps({**run_settings(args), "Datasets": labels, "Device": str(device)}, indent=2), encoding="utf-8"
    )
    print(f"Deep corr-encoder, paper version ({sum(p.numel() for p in Correncoder_model().parameters())} parameters, "
          f"output ReLU {OUTPUT_RELU}): "
          f"{args.epochs} epochs, batch {batch_size}, learning rate {learning_rate:g}, seed {seed_val}; "
          f"9.6 s windows at {CORR_FS} Hz; device={device}")
    print(f"Results: {results_root}")
    check_path = results_root / "Summary" / "data_check.csv"
    if args.summary_only:
        checks = pd.read_csv(check_path, dtype={"Subject": str}) if check_path.exists() else pd.DataFrame()
    else:
        rows = []
        for key in dataset_keys:
            dataset_rows, device = run_dataset(args, key, results_root, device)
            rows += dataset_rows
        checks = pd.DataFrame(rows)
    summarize(results_root, labels, checks)


if __name__ == "__main__":
    main()
