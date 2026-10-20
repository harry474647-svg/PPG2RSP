r"""261020_RespDiff: RespDiff (Miao, Chen, Li & Mandic, "RespDiff: An End-to-End
Multi-scale RNN Diffusion Model for Respiratory Waveform Estimation from PPG
Signals", arXiv:2410.04366, 2024) on CapnoBase, BIDMC and STEAM2, reproduced
as in the paper and its official code (github.com/MYY311/RespDiff, commit
3ff0554: model_fft.py / breathing_bidmc_fft.py with the spectral loss,
model.py / breathing_bidmc.py without it). LOSOCV, one model per fold.

Model (official code)
---------------------
A conditional diffusion model (DDPM) of the respiratory waveform y given the
PPG x, 50 diffusion steps, beta linear from 1e-4 to 0.5. The noise predictor:
* E_fine: Conv1d(1 -> 32) with kernels 1, 3, 5, 7, 9, 11, concatenated (192
  channels), Kaiming-normal initialization; one for the PPG, one for the
  noisy respiratory waveform y_t;
* E_coarse: for kernels 3, 5, 7, 9, 11, 13 a 1x1 Conv1d(1 -> 32) and three
  residual dilated Conv1d(32 -> 32) with dilations 1, 2, 4, each with ReLU and
  BatchNorm (batch statistics only), concatenated (192 channels);
* f_ppg = E_fine(x) + step embedding + lambda_ppg * E_coarse(x), lambda_ppg a
  learned weight per channel (initialized to 1); the step embedding is
  sinusoidal (128) -> Linear(128, 192) -> SiLU -> Linear(192, 192) -> SiLU;
* [f_ppg, E_fine(y_t)] (384 channels) -> bidirectional RNN (tanh, 6 layers,
  1024 units) -> Linear(2048, 1024) -> ReLU -> Linear(1024, 128) -> ReLU ->
  Linear(128, 512) -> ReLU -> Linear(512, 1) = predicted noise.

Training (official code): t uniform over the 50 steps; loss = sum of the
squared noise error / L, plus 0.01 x the spectral loss (mean squared
difference of the FFT magnitudes, orthonormal FFT, between the one-step
estimate y0_hat = (y_t - sqrt(1 - abar_t) eps) / sqrt(abar_t) and y0); Adam,
learning rate 1e-4, batch 128, 400 epochs, learning rate x 0.1 after epochs
280 and 396; no validation set, the model after the last epoch is used.
Sampling (official code): DDPM, 50 steps (NFE 50), or DDIM, 6 steps (NFE 6;
the x0 estimate is min-max scaled at every step as in the official code);
100 samples per 5 s segment are averaged, test batch 64.

Data (paper, official code)
---------------------------
PPG and respiratory waveform resampled to 30 Hz (FFT resampling,
scipy.signal.resample), the respiratory waveform low-pass filtered
(Butterworth, order 8, 1 Hz, zero phase), cut into non-overlapping 5 s
segments; every segment is kept (no quality-based removal). Per segment
min-max scaling: PPG to [-1, 1], respiratory waveform to [0, 1] (official
code; the paper text names the two ranges the other way round; the DDIM
scaling and the duty-cycle threshold 0.5 of the code need [0, 1]). The PPG
is not low-pass filtered (official code; the paper text says both are).
Respiratory waveform: CapnoBase the CO2 capnogram, BIDMC the impedance
pneumogram (ref.resp_sig.imp, as in the official code), STEAM2 the belt (RSP)
as recorded. The files are read with the readers of 261007 (CapnoBase) and
261013 (BIDMC, STEAM2) (signals only: no expert labels, artifact spans or
quality checks are used).

Evaluation (paper, official code)
---------------------------------
The 5 s outputs of a recording are joined and low-pass filtered (as above);
over 60 s windows moved by 20 s, the breathing rate of the output and of
the reference waveform is the largest positive-frequency FFT component x 60
(breaths/min). Per subject: RR error = mean absolute error over its windows,
waveform MAE = mean |output - reference| over the recording, and the
duty-cycle correlation of the official code (samples >= 0.5 per window).
Dataset result = mean (+- SD) over subjects, as in the paper. The official
code divides the summed window errors by (number of windows - 1); that
value is reported as RR_MAE_Official_Code (the number the paper's 1.18 bpm
corresponds to), RR_MAE is the mean over the windows.

Models (paper, Tables I and II and the multi-scale ablation)
-------------------------------------------------------------
  folder             model                                     sampling
  respdiff           RespDiff (spectral loss; model_fft.py)    DDPM 50, DDIM 6
  no_spectral_loss   w/o spectral loss (model.py)              DDPM 50, DDIM 6
  no_multiscale      w/o spectral loss, E_fine kernels all 3   DDPM 50
The official model.py (w/o spectral loss) also adds the step embedding to
the features of y_t; this is kept. Paper (BIDMC): RespDiff 1.18 / 1.30 bpm
(NFE 50 / 6), w/o spectral loss 1.44 / 1.46, kernel sizes all 3 1.53;
waveform MAE 0.307 (0.316 w/o spectral loss).

Needed for these datasets (not in the paper or the official code)
-----------------------------------------------------------------
* resampling to 30 Hz from the sampling rate of the file: int(N x 30 / fs)
  samples (BIDMC 125 Hz: N x 0.24 as in the official code);
* STEAM2 sessions are split at their boundary events; every continuous piece
  is processed and scored as its own recording (no 60 s window crosses a
  boundary) and all sessions of a person form one subject;
* a constant segment (max = min) is scaled to the lower bound of its range
  instead of NaN, missing samples (NaN) are interpolated linearly; both are
  counted in Summary/data_check.csv (CapnoBase and BIDMC have none);
* fixed seeds per fold (the official code sets none), the training state is
  saved every 25 epochs to resume an interrupted run, every dataset and model
  is trained in its own process (memory), the loss of every epoch is printed.

Outputs (PROJECT_ROOT = D:\PPG2RSP_RRWaveNet_Inspired\261020_RespDiff)
-----------------------------------------------------------------------
  {model}\{dataset}\Models\{subject}_final.pth          model after training
  {model}\{dataset}\Training_Logs\{subject}_loss.csv    loss of every epoch
  {model}\{dataset}\Predictions\{subject}_{sampling}.npz output, reference
                                                        and PPG per recording
  {model}\{dataset}\Fold_Results\{subject}_{sampling}.json / _windows.csv
  Summary\respdiff_results.csv        every dataset, model and sampling
  Summary\table_spectral_loss.csv     Table II layout
  Summary\table_multiscale.csv        multi-scale encoder ablation
  Summary\per_subject.csv, windows.csv, data_check.csv (--dry-run)

Run (GPU):                       python 261020_RespDiff.py
Resume after an interruption:    python 261020_RespDiff.py --resume
Some datasets / models:          --datasets bidmc --variants respdiff
Summary of stored results only:  --summary-only
Check the data without training: --dry-run --no-require-cuda
"""

import argparse
import csv
import gc
import hashlib
import json
import os
import random
import re
import subprocess
import sys
import time
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import scipy.io as sio
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.signal import butter, filtfilt, resample
from torch.utils.data import DataLoader, Dataset


# ---------------------------------------------------------------------------
# Paths and datasets
# ---------------------------------------------------------------------------

PROJECT_ROOT = Path(r"D:\PPG2RSP_RRWaveNet_Inspired\261020_RespDiff")
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
        # only when a file has no srate (261013).
        "default_fs": 200,
        "reference": "respiratory belt (RSP), as recorded",
    },
}

DATASET_ORDER = ("capnobase", "bidmc", "steam2")


# ---------------------------------------------------------------------------
# RespDiff settings (paper and official code)
# ---------------------------------------------------------------------------

FS = 30
SEGMENT_SEC = 5
SEGMENT_LEN = FS * SEGMENT_SEC
LOWPASS_HZ = 1.0
LOWPASS_ORDER = 8
LOWPASS_B, LOWPASS_A = butter(LOWPASS_ORDER, LOWPASS_HZ / (0.5 * FS), btype="low", analog=False)
PPG_RANGE = (-1.0, 1.0)
RESP_RANGE = (0.0, 1.0)

NUM_STEPS = 50
BETA_START = 1e-4
BETA_END = 0.5
RNN_INPUT_DIM = 384
RNN_HIDDEN_DIM = 1024
RNN_LAYERS = 6
RNN_OUTPUT_DIM = 128
DECODER_HIDDEN_DIM = 512
ENCODER_CHANNELS = 32
FINE_KERNELS = (1, 3, 5, 7, 9, 11)
COARSE_KERNELS = (3, 5, 7, 9, 11, 13)
EMBEDDING_DIM = 128
EMBEDDING_PROJECTION_DIM = 192
SPECTRAL_WEIGHT = 1e-2

LEARNING_RATE = 1e-4
EPOCHS = 400
LR_MILESTONE_FRACTIONS = (0.7, 0.99)
LR_GAMMA = 0.1
TRAIN_BATCH = 128
TEST_BATCH = 64
N_SAMPLES = 100

RR_WINDOW = 60 * FS
RR_HOP = 20 * FS
DUTY_THRESHOLD = 0.5

# Training state saved every CHECKPOINT_EVERY epochs (resume).
CHECKPOINT_EVERY = 25
CODE_NAME = "261020_RespDiff"


@dataclass(frozen=True)
class Variant:
    """One model of the paper's experiments."""

    name: str
    label: str
    spectral_loss: bool
    # model.py (w/o spectral loss) adds the step embedding to E_fine(y_t).
    step_embedding_on_noise: bool
    fine_kernels: Tuple[int, ...]
    samplers: Tuple[str, ...]


VARIANTS = {
    "respdiff": Variant("respdiff", "RespDiff", True, False, FINE_KERNELS, ("ddpm50", "ddim6")),
    "no_spectral_loss": Variant(
        "no_spectral_loss", "w/o spectral loss", False, True, FINE_KERNELS, ("ddpm50", "ddim6")
    ),
    "no_multiscale": Variant(
        "no_multiscale", "w/o spectral loss, kernel sizes all 3", False, True, (3,) * 6, ("ddpm50",)
    ),
}
# sampling -> (sampler, number of function evaluations)
SAMPLERS = {"ddpm50": ("DDPM", 50), "ddim6": ("DDIM", 6)}


# ---------------------------------------------------------------------------
# Data readers of 261007 (CapnoBase) and 261013 (BIDMC, STEAM2), unchanged
# ---------------------------------------------------------------------------

CAPNOBASE_FILE_SUFFIXES = (".csv", ".tab")
_NUMBER_PATTERN = re.compile(r"[-+]?(?:\d+\.?\d*|\.\d+)(?:[eE][-+]?\d+)?")
STEAM2_CHANNELS = {"PPG": ("PPG", "PLETH", "BVP"), "RSP": ("RSP", "RESP", "RESPIRATION", "BELT")}
STEAM2_SESSION_PATTERN = re.compile(r"^(?P<subject>.+?)[_\- ]*session[_\- ]*(?P<session>\d+)$", re.IGNORECASE)
FATAL_ERROR_WORDS = ("CUDA", "cuDNN", "CUDNN", "CUBLAS", "cublas", "out of memory", "Memory allocation")


def stable_seed(*parts: object) -> int:
    text = "|".join(str(part) for part in parts)
    digest = hashlib.sha256(text.encode("utf-8")).hexdigest()
    return int(digest[:8], 16) % (2**31 - 1)


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


def is_fatal_error(exc: BaseException) -> bool:
    """Out of host or GPU memory, or a CUDA / cuDNN failure: the process (and
    its CUDA context) cannot be trusted any more, so it stops instead of
    failing every following fold; it is restarted with --resume."""
    if isinstance(exc, MemoryError):
        return True
    text = f"{type(exc).__name__}: {exc}"
    return any(word in text for word in FATAL_ERROR_WORDS)


def release_memory(device: Optional[torch.device] = None) -> None:
    """Release objects from completed folds before the next LOSOCV fold."""
    gc.collect()
    if device is not None and device.type == "cuda":
        torch.cuda.empty_cache()


def run_worker(command: Sequence[str]) -> int:
    """Run a worker process and print its output here line by line, so it is
    seen in any console (IDE consoles such as Spyder do not show the output a
    child process writes itself). Stopping the main process stops the worker."""
    env = dict(os.environ, PYTHONUNBUFFERED="1", PYTHONIOENCODING="utf-8")
    process = subprocess.Popen(
        list(command),
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        encoding="utf-8",
        errors="replace",
        bufsize=1,
        env=env,
    )
    try:
        for line in process.stdout:
            print(line, end="", flush=True)
        return process.wait()
    except BaseException:
        process.kill()
        process.wait()
        raise


# ---------------------------------------------------------------------------
# RespDiff (official code: model_fft.py, model.py)
# ---------------------------------------------------------------------------

class FFTLoss(nn.Module):
    """Spectral loss: mean squared difference of the FFT magnitudes
    (orthonormal FFT), averaged over the frequency bins and the batch."""

    def forward(self, predicted: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        pred_magnitude = torch.abs(torch.fft.fft(predicted, dim=-1, norm="ortho"))
        target_magnitude = torch.abs(torch.fft.fft(target, dim=-1, norm="ortho"))
        return torch.mean((pred_magnitude - target_magnitude) ** 2, dim=-1).mean()


class ConditionalTimeGrad(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, num_layers: int, output_dim: int):
        super().__init__()
        self.rnn = nn.RNN(input_dim, hidden_dim, num_layers, nonlinearity="tanh", batch_first=True,
                          bidirectional=True)
        self.fc1 = nn.Linear(2 * hidden_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        rnn_out, _ = self.rnn(x)
        return self.fc2(F.relu(self.fc1(rnn_out)))


class Decoder(nn.Module):
    def __init__(self, input_dimension: int, hidden_dimension: int = DECODER_HIDDEN_DIM, out_dimension: int = 1):
        super().__init__()
        self.fc1 = nn.Linear(input_dimension, hidden_dimension)
        self.fc2 = nn.Linear(hidden_dimension, out_dimension)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc2(F.relu(self.fc1(x)))


class SignalEncoder(nn.Module):
    """E_fine: one Conv1d(1 -> 32) per kernel size, concatenated."""

    def __init__(self, kernel_sizes: Sequence[int] = FINE_KERNELS):
        super().__init__()
        self.conv_layers = nn.ModuleList([
            nn.Conv1d(1, ENCODER_CHANNELS, kernel_size=ks, padding=ks // 2) for ks in kernel_sizes
        ])
        for layer in self.conv_layers:
            nn.init.kaiming_normal_(layer.weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.cat([conv(x) for conv in self.conv_layers], dim=1)


def conv1d_with_init(in_channels: int, out_channels: int, kernel_size: int) -> nn.Conv1d:
    layer = nn.Conv1d(in_channels, out_channels, kernel_size)
    nn.init.kaiming_normal_(layer.weight)
    return layer


class DilatedConv(nn.Module):
    """One branch of E_coarse: 1x1 bottleneck, then three residual dilated
    convolutions (dilations 1, 2, 4) with ReLU and BatchNorm (batch
    statistics only, as track_running_stats=False in the official code)."""

    def __init__(self, kernel_size: int, in_channels: int = 1, out_channels: int = ENCODER_CHANNELS):
        super().__init__()
        self.layer0 = nn.Conv1d(out_channels, out_channels, kernel_size=kernel_size, dilation=1,
                                padding=kernel_size // 2 * 1)
        self.layer1 = nn.Conv1d(out_channels, out_channels, kernel_size=kernel_size, dilation=2,
                                padding=kernel_size // 2 * 2)
        self.layer2 = nn.Conv1d(out_channels, out_channels, kernel_size=kernel_size, dilation=4,
                                padding=kernel_size // 2 * 4)
        self.bn0 = nn.BatchNorm1d(out_channels, track_running_stats=False)
        self.bn1 = nn.BatchNorm1d(out_channels, track_running_stats=False)
        self.bn2 = nn.BatchNorm1d(out_channels, track_running_stats=False)
        self.relu = nn.ReLU()
        for layer in (self.layer0, self.layer1, self.layer2):
            nn.init.kaiming_normal_(layer.weight, nonlinearity="relu")
        self.bottle = conv1d_with_init(in_channels, out_channels, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.bottle(x)
        x = self.bn0(self.relu(self.layer0(x)) + x)
        x = self.bn1(self.relu(self.layer1(x)) + x)
        x = self.bn2(self.relu(self.layer2(x)) + x)
        return x


class SignalEncoderDil(nn.Module):
    """E_coarse: one DilatedConv branch per kernel size, concatenated."""

    def __init__(self, kernel_sizes: Sequence[int] = COARSE_KERNELS):
        super().__init__()
        self.conv_layers = nn.ModuleList([
            DilatedConv(kernel_size=ks, in_channels=1, out_channels=ENCODER_CHANNELS) for ks in kernel_sizes
        ])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.cat([conv(x) for conv in self.conv_layers], dim=1)


class DiffusionEmbedding(nn.Module):
    def __init__(self, num_steps: int, embedding_dim: int = EMBEDDING_DIM, projection_dim: Optional[int] = None):
        super().__init__()
        if projection_dim is None:
            projection_dim = embedding_dim
        self.register_buffer("embedding", self._build_embedding(num_steps, embedding_dim / 2), persistent=False)
        self.projection1 = nn.Linear(embedding_dim, projection_dim)
        self.projection2 = nn.Linear(projection_dim, projection_dim)

    def forward(self, diffusion_step: torch.Tensor) -> torch.Tensor:
        x = self.embedding[diffusion_step]
        x = F.silu(self.projection1(x))
        return F.silu(self.projection2(x))

    @staticmethod
    def _build_embedding(num_steps: int, dim: float = 64) -> torch.Tensor:
        steps = torch.arange(num_steps).unsqueeze(1)
        frequencies = 10.0 ** (torch.arange(dim) / (dim - 1) * 4.0).unsqueeze(0)
        table = steps * frequencies
        return torch.cat([torch.sin(table), torch.cos(table)], dim=1)


class DiffusionBaseModel(nn.Module):
    """Noise predictor (module names as in the official code, so that its
    state dicts load)."""

    def __init__(self, variant: Variant):
        super().__init__()
        self.diff_model = ConditionalTimeGrad(RNN_INPUT_DIM, RNN_HIDDEN_DIM, RNN_LAYERS, RNN_OUTPUT_DIM)
        self.diffusion_embedding = DiffusionEmbedding(
            num_steps=NUM_STEPS, embedding_dim=EMBEDDING_DIM, projection_dim=EMBEDDING_PROJECTION_DIM
        )
        self.ppg_encoder1 = SignalEncoderDil()
        self.ppg_encoder2 = SignalEncoder(variant.fine_kernels)
        self.noise_encoder = SignalEncoder(variant.fine_kernels)
        self.de = Decoder(input_dimension=RNN_OUTPUT_DIM)
        self.weight = nn.Parameter(torch.ones(1, EMBEDDING_PROJECTION_DIM, 1), requires_grad=True)
        self.step_embedding_on_noise = variant.step_embedding_on_noise

    def forward(self, ppg: torch.Tensor, noise: torch.Tensor, step: torch.Tensor) -> torch.Tensor:
        embedding = self.diffusion_embedding(step).unsqueeze(2)
        f1 = self.ppg_encoder2(ppg) + embedding + self.weight * self.ppg_encoder1(ppg)
        f2 = self.noise_encoder(noise)
        if self.step_embedding_on_noise:
            f2 = f2 + embedding
        f = torch.concat([f1, f2], dim=1).permute(0, 2, 1)
        return self.de(F.relu(self.diff_model(f)))


class DiffusionPipeline(nn.Module):
    def __init__(self, variant: Variant, device: torch.device):
        super().__init__()
        self.diffusion_model = DiffusionBaseModel(variant)
        self.num_steps = NUM_STEPS
        self.beta = np.linspace(BETA_START, BETA_END, self.num_steps)
        self.alpha_hat = 1 - self.beta
        self.alpha = np.cumprod(self.alpha_hat)
        self.register_buffer("alpha_torch", torch.tensor(self.alpha).float().unsqueeze(1).unsqueeze(1))
        self.spectral_loss = variant.spectral_loss
        self.loss_fft = FFTLoss()
        self.device = device

    def train_process(self, ppg: torch.Tensor, resp: torch.Tensor) -> torch.Tensor:
        B, K, L = ppg.shape
        t = torch.randint(0, self.num_steps, [B]).to(self.device)
        current_alpha = self.alpha_torch[t]
        noise = torch.randn(B, K, L).to(self.device)
        noisy_resp = (current_alpha ** 0.5) * resp + (1.0 - current_alpha) ** 0.5 * noise
        predicted = self.diffusion_model(ppg, noisy_resp, t).permute(0, 2, 1)
        residual = noise - predicted
        loss = (residual ** 2).sum() / L
        if self.spectral_loss:
            predicted_resp = (noisy_resp - (1.0 - current_alpha) ** 0.5 * predicted) / (current_alpha ** 0.5)
            loss = loss + self.loss_fft(predicted_resp, resp) * SPECTRAL_WEIGHT
        return loss

    @torch.no_grad()
    def ddpm_sampling(self, ppg: torch.Tensor, n_samples: int) -> torch.Tensor:
        """DDPM over all 50 steps (official imputation)."""
        B, K, L = ppg.shape
        samples = torch.ones(B, n_samples, K, L)
        for i in range(n_samples):
            sample = torch.randn(B, K, L).to(self.device)
            for t in range(self.num_steps - 1, -1, -1):
                steps = torch.full((B,), t).to(self.device)
                noise_predicted = self.diffusion_model(ppg, sample, steps).permute(0, 2, 1)
                coeff1 = 1 / self.alpha_hat[t] ** 0.5
                coeff2 = (1 - self.alpha_hat[t]) / (1 - self.alpha[t]) ** 0.5
                noise = torch.randn_like(noise_predicted)
                sigma = ((1.0 - self.alpha[t - 1]) / (1.0 - self.alpha[t]) * self.beta[t]) ** 0.5
                sample = coeff1 * (sample - coeff2 * noise_predicted)
                if t > 0:
                    sample = sample + sigma * noise
            samples[:, i] = sample.detach().cpu()
        return samples

    @torch.no_grad()
    def ddim_sampling(self, ppg: torch.Tensor, n_samples: int, n_steps: int) -> torch.Tensor:
        """Deterministic DDIM over n_steps of the 50 (official ddim_sampling,
        including the min-max scaling of the x0 estimate at every step)."""
        B, K, L = ppg.shape
        samples = torch.ones(B, n_samples, K, L)
        timesteps = np.linspace(self.num_steps - 1, 0, n_steps).astype(int)
        for i in range(n_samples):
            sample = torch.randn(B, K, L).to(self.device)
            for index, t in enumerate(timesteps):
                steps = torch.full((B,), int(t)).to(self.device)
                noise_predicted = self.diffusion_model(ppg, sample, steps).permute(0, 2, 1)
                alpha_t = self.alpha[t]
                alpha_next = self.alpha[timesteps[index + 1]] if index + 1 < len(timesteps) else 1.0
                x0_pred = (sample - (1 - alpha_t) ** 0.5 * noise_predicted) / alpha_t ** 0.5
                x0_pred = (x0_pred - x0_pred.min()) / (x0_pred.max() - x0_pred.min())
                if index == len(timesteps) - 1:
                    sample = x0_pred
                else:
                    sample = alpha_next ** 0.5 * x0_pred + (1 - alpha_next) ** 0.5 * noise_predicted
            samples[:, i] = sample.detach().cpu()
        return samples

    def sample(self, ppg: torch.Tensor, sampling: str, n_samples: int) -> torch.Tensor:
        sampler, nfe = SAMPLERS[sampling]
        if sampler == "DDPM":
            return self.ddpm_sampling(ppg, n_samples)
        return self.ddim_sampling(ppg, n_samples, nfe)


# ---------------------------------------------------------------------------
# Data: recordings, 30 Hz, 5 s segments (paper, official code)
# ---------------------------------------------------------------------------

def fill_missing(signal: np.ndarray) -> Tuple[np.ndarray, int]:
    """Linear interpolation of missing samples (NaN); returns the count."""
    signal = np.asarray(signal, dtype=np.float64).reshape(-1)
    missing = ~np.isfinite(signal)
    count = int(missing.sum())
    if count and count < signal.size:
        index = np.arange(signal.size)
        signal = signal.copy()
        signal[missing] = np.interp(index[missing], index[~missing], signal[~missing])
    return signal, count


def capnobase_recordings(cfg: dict) -> Dict[str, List[dict]]:
    csv_dir = Path(cfg["data_path"])
    data = {}
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
            data[case_id] = [{"name": case_id, "ppg": ppg, "resp": co2, "ppg_fs": ppg_fs, "resp_fs": co2_fs}]
        except Exception as exc:
            if is_fatal_error(exc):
                raise
            print(f"[CapnoBase {case_id}] skipped: {exc}")
    return data


def bidmc_recordings(cfg: dict) -> Dict[str, List[dict]]:
    mat = sio.loadmat(cfg["data_path"])
    data = {}
    for index, subject_record in enumerate(mat["data"][0]):
        subject = f"{index + 1:02d}"
        try:
            ppg = np.asarray(subject_record["ppg"][0, 0]["v"]).squeeze()
            resp = np.asarray(subject_record["ref"][0, 0]["resp_sig"][0, 0]["imp"][0, 0]["v"]).squeeze()
            data[subject] = [{"name": subject, "ppg": ppg, "resp": resp, "ppg_fs": float(cfg["default_fs"]),
                              "resp_fs": float(cfg["default_fs"])}]
        except Exception as exc:
            if is_fatal_error(exc):
                raise
            print(f"[BIDMC {subject}] skipped: {exc}")
    return data


def steam2_recordings(cfg: dict) -> Dict[str, List[dict]]:
    data = {}
    for subject, items in steam2_session_files(Path(cfg["ppg_dir"]), Path(cfg["rsp_dir"])).items():
        try:
            recordings = []
            for name, ppg_path, rsp_path in items:
                ppg = read_eeglab_channel(ppg_path, "PPG", cfg["default_fs"])
                rsp = read_eeglab_channel(rsp_path, "RSP", cfg["default_fs"])
                pieces, info = steam2_session_pieces(ppg, rsp)
                if abs(info["Length_Difference_Sec"]) > 1.0:
                    print(f"[STEAM2 {subject} {name}] PPG {info['PPG_Sec']:.1f} s vs RSP "
                          f"{info['RSP_Sec']:.1f} s; the common part is used")
                for piece_index, (p, r, p_fs, r_fs) in enumerate(pieces):
                    recordings.append({"name": f"{name}_piece{piece_index + 1}", "ppg": p, "resp": r,
                                       "ppg_fs": p_fs, "resp_fs": r_fs})
            if recordings:
                data[subject] = recordings
        except Exception as exc:
            if is_fatal_error(exc):
                raise
            print(f"[STEAM2 {subject}] skipped: {exc}")
    return data


def segment_recording(recording: dict) -> Optional[dict]:
    """30 Hz, respiratory low-pass, non-overlapping 5 s segments (raw
    amplitudes; scaled per segment when a fold is built)."""
    ppg, ppg_missing = fill_missing(recording["ppg"])
    resp, resp_missing = fill_missing(recording["resp"])
    ppg = resample(ppg, int(ppg.shape[0] * (FS / recording["ppg_fs"])))
    resp = resample(resp, int(resp.shape[0] * (FS / recording["resp_fs"])))
    length = min(len(ppg), len(resp))
    count = length // SEGMENT_LEN
    if count < 1:
        return None
    resp = filtfilt(LOWPASS_B, LOWPASS_A, resp[:length])
    ppg = ppg[:length]
    return {
        "name": recording["name"],
        "ppg": ppg[: count * SEGMENT_LEN].reshape(count, SEGMENT_LEN),
        "resp": resp[: count * SEGMENT_LEN].reshape(count, SEGMENT_LEN),
        "ppg_fs": float(recording["ppg_fs"]),
        "resp_fs": float(recording["resp_fs"]),
        "ppg_missing": ppg_missing,
        "resp_missing": resp_missing,
    }


def load_dataset(dataset_key: str) -> Dict[str, List[dict]]:
    cfg = DATASET_CONFIGS[dataset_key]
    readers = {"capnobase": capnobase_recordings, "bidmc": bidmc_recordings, "steam2": steam2_recordings}
    data = {}
    for subject, recordings in readers[dataset_key](cfg).items():
        segmented = [item for item in (segment_recording(r) for r in recordings) if item is not None]
        if segmented:
            data[subject] = segmented
        else:
            print(f"[{cfg['label']} {subject}] shorter than one {SEGMENT_SEC} s segment, skipped")
    return data


def scale_segments(segments: np.ndarray, value_range: Tuple[float, float]) -> Tuple[np.ndarray, int]:
    """Per segment min-max scaling to value_range (official code); a constant
    segment is set to the lower bound instead of NaN. Returns the count of
    constant segments."""
    low, high = value_range
    minimum = segments.min(axis=1, keepdims=True)
    span = segments.max(axis=1, keepdims=True) - minimum
    constant = span[:, 0] <= 0
    scaled = (segments - minimum) / np.where(span > 0, span, 1.0)
    return low + (high - low) * scaled, int(constant.sum())


def stack_scaled(recordings: Sequence[dict]) -> Tuple[np.ndarray, np.ndarray, int]:
    ppg = np.concatenate([r["ppg"] for r in recordings], axis=0)
    resp = np.concatenate([r["resp"] for r in recordings], axis=0)
    ppg, ppg_constant = scale_segments(ppg, PPG_RANGE)
    resp, resp_constant = scale_segments(resp, RESP_RANGE)
    return ppg, resp, ppg_constant + resp_constant


class SegmentDataset(Dataset):
    def __init__(self, ppg: np.ndarray, resp: np.ndarray):
        self.ppg = ppg
        self.resp = resp

    def __len__(self) -> int:
        return len(self.ppg)

    def __getitem__(self, index: int):
        return (
            torch.tensor(self.ppg[index, None, :], dtype=torch.float32),
            torch.tensor(self.resp[index, None, :], dtype=torch.float32),
        )


def data_check_rows(dataset_key: str, data: Dict[str, List[dict]]) -> List[dict]:
    rows = []
    for subject, recordings in data.items():
        for recording in recordings:
            _, ppg_constant = scale_segments(recording["ppg"], PPG_RANGE)
            _, resp_constant = scale_segments(recording["resp"], RESP_RANGE)
            rows.append({
                "Dataset": DATASET_CONFIGS[dataset_key]["label"],
                "Subject": subject,
                "Recording": recording["name"],
                "PPG_FS": recording["ppg_fs"],
                "Resp_FS": recording["resp_fs"],
                "Reference": DATASET_CONFIGS[dataset_key]["reference"],
                "Segments": int(len(recording["ppg"])),
                "Seconds_30Hz": float(len(recording["ppg"]) * SEGMENT_SEC),
                "PPG_Missing_Samples": recording["ppg_missing"],
                "Resp_Missing_Samples": recording["resp_missing"],
                "PPG_Constant_Segments": ppg_constant,
                "Resp_Constant_Segments": resp_constant,
            })
    return rows


# ---------------------------------------------------------------------------
# Evaluation (official code)
# ---------------------------------------------------------------------------

def respiratory_rate(signal: np.ndarray, sampling_rate: float = FS) -> float:
    """Largest positive-frequency FFT component x 60 (breaths/min)."""
    magnitude = np.abs(np.fft.fft(signal))
    freq = np.fft.fftfreq(len(signal), d=1 / sampling_rate)
    positive = freq > 0
    return float(freq[positive][np.argmax(magnitude[positive])] * 60)


def score_recording(prediction_segments: np.ndarray, reference_segments: np.ndarray) -> dict:
    """Joined and low-pass filtered output against the joined reference:
    60 s windows moved by 20 s (official code)."""
    prediction = filtfilt(LOWPASS_B, LOWPASS_A, np.concatenate(prediction_segments, axis=0))
    reference = np.concatenate(reference_segments, axis=0)
    windows = []
    divisor = 0
    if len(prediction) >= RR_WINDOW:
        divisor = (len(prediction) - RR_WINDOW) // RR_HOP
        for index in range(divisor + 1):
            r_current = prediction[RR_HOP * index:RR_HOP * index + RR_WINDOW].copy()
            c_current = reference[RR_HOP * index:RR_HOP * index + RR_WINDOW].copy()
            rr_reference = respiratory_rate(c_current)
            rr_predicted = respiratory_rate(r_current)
            windows.append({
                "Window": index,
                "Start_Sec": RR_HOP * index / FS,
                "RR_Reference": rr_reference,
                "RR_Predicted": rr_predicted,
                "AE": abs(rr_reference - rr_predicted),
                "Duty_Reference": float((c_current >= DUTY_THRESHOLD).sum()),
                "Duty_Predicted": float((r_current >= DUTY_THRESHOLD).sum()),
            })
    return {
        "prediction": prediction,
        "reference": reference,
        "windows": windows,
        "official_divisor": divisor,
        "absolute_error": np.abs(prediction - reference),
    }


def safe_corrcoef(a: Sequence[float], b: Sequence[float]) -> float:
    if len(a) < 2:
        return float("nan")
    with np.errstate(all="ignore"), warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return float(np.corrcoef(a, b)[0, 1])


# ---------------------------------------------------------------------------
# Training and testing of one fold
# ---------------------------------------------------------------------------

def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def fold_root(results_root: Path, variant: Variant, dataset_key: str) -> Path:
    return results_root / variant.name / DATASET_CONFIGS[dataset_key]["label"]


def atomic_torch_save(obj, path: Path) -> None:
    temporary = path.with_name(path.name + ".tmp")
    torch.save(obj, temporary)
    os.replace(temporary, path)


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, default=float), encoding="utf-8")
    os.replace(temporary, path)


def read_json(path: Path) -> Optional[dict]:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


def training_key(args: argparse.Namespace, variant: Variant, dataset_key: str, train_subjects: Sequence[str],
                 train_segments: int) -> dict:
    """What a stored model was trained with (it is reused only if equal)."""
    return {
        "Code": CODE_NAME,
        "Model": variant.name,
        "Dataset": DATASET_CONFIGS[dataset_key]["label"],
        "Epochs": int(args.epochs),
        "Train_Subjects": len(train_subjects),
        "Train_Segments": int(train_segments),
    }


def train_fold(
    args: argparse.Namespace,
    variant: Variant,
    dataset_key: str,
    subject: str,
    train_ppg: np.ndarray,
    train_resp: np.ndarray,
    key: dict,
    root: Path,
    device: torch.device,
) -> Tuple[DiffusionPipeline, float]:
    """Official training loop; returns the model after the last epoch."""
    model_dir, log_dir = root / "Models", root / "Training_Logs"
    model_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)
    final_path = model_dir / f"{subject}_final.pth"
    key_path = model_dir / f"{subject}_final.json"
    state_path = model_dir / f"{subject}_training_state.pt"
    seed = stable_seed("261020-respdiff-train", dataset_key, subject)
    set_seed(seed)
    model = DiffusionPipeline(variant, device).to(device)
    if args.resume and final_path.exists() and read_json(key_path) == key:
        model.load_state_dict(torch.load(final_path, map_location=device))
        print(f"    trained model reused ({final_path.name})", flush=True)
        return model, 0.0
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    milestones = [int(fraction * args.epochs) for fraction in LR_MILESTONE_FRACTIONS]
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=milestones, gamma=LR_GAMMA)
    generator = torch.Generator().manual_seed(seed)
    loader = DataLoader(SegmentDataset(train_ppg, train_resp), batch_size=TRAIN_BATCH, shuffle=True,
                        generator=generator)
    history, start_epoch, elapsed = [], 0, 0.0
    if args.resume and state_path.exists():
        state = torch.load(state_path, map_location=device, weights_only=False)
        if state.get("key") == key:
            model.load_state_dict(state["model"])
            optimizer.load_state_dict(state["optimizer"])
            scheduler.load_state_dict(state["scheduler"])
            generator.set_state(state["loader_rng"])
            torch.set_rng_state(state["torch_rng"])
            if torch.cuda.is_available() and state.get("cuda_rng") is not None:
                torch.cuda.set_rng_state_all(state["cuda_rng"])
            history, start_epoch, elapsed = list(state["history"]), int(state["epoch"]), float(state["seconds"])
            print(f"    resumed after epoch {start_epoch}", flush=True)
    elif state_path.exists():
        state_path.unlink()
    for epoch in range(start_epoch, args.epochs):
        started = time.time()
        model.train()
        avg_loss, batch_no = 0.0, 0
        learning_rate = float(optimizer.param_groups[0]["lr"])
        for batch_no, (ppg, resp) in enumerate(loader, start=1):
            optimizer.zero_grad()
            loss = model.train_process(ppg.to(device), resp.to(device))
            loss.backward()
            avg_loss += loss.item() / ppg.shape[0]
            optimizer.step()
        scheduler.step()
        seconds = time.time() - started
        elapsed += seconds
        epoch_loss = avg_loss / max(batch_no, 1)
        history.append({"Epoch": epoch + 1, "Train_Loss": epoch_loss, "LR": learning_rate, "Seconds": seconds})
        print(f"    epoch {epoch + 1:3d}/{args.epochs}  train loss {epoch_loss:.5f}  lr {learning_rate:.1e}  "
              f"{seconds:.1f} s", flush=True)
        if (epoch + 1) % CHECKPOINT_EVERY == 0 and epoch + 1 < args.epochs:
            atomic_torch_save(
                {
                    "key": key,
                    "epoch": epoch + 1,
                    "model": model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "scheduler": scheduler.state_dict(),
                    "loader_rng": generator.get_state(),
                    "torch_rng": torch.get_rng_state(),
                    "cuda_rng": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
                    "history": history,
                    "seconds": elapsed,
                },
                state_path,
            )
    atomic_torch_save(model.state_dict(), final_path)
    write_json(key_path, key)
    pd.DataFrame(history).to_csv(log_dir / f"{subject}_loss.csv", index=False, encoding="utf-8-sig")
    if state_path.exists():
        state_path.unlink()
    return model, elapsed


def test_fold(
    args: argparse.Namespace,
    model: DiffusionPipeline,
    variant: Variant,
    dataset_key: str,
    subject: str,
    recordings: Sequence[dict],
    sampling: str,
    root: Path,
    device: torch.device,
    key: dict,
    training_seconds: float,
) -> dict:
    """Samples every 5 s segment of the held-out subject (batches of 64 in
    recording order, as in the official code) and scores every recording."""
    test_ppg, test_resp, constant = stack_scaled(recordings)
    set_seed(stable_seed("261020-respdiff-sample", dataset_key, subject, sampling))
    model.eval()
    loader = DataLoader(SegmentDataset(test_ppg, test_resp), batch_size=TEST_BATCH, shuffle=False)
    started = time.time()
    outputs = []
    for ppg, _ in loader:
        samples = model.sample(ppg.to(device), sampling, args.n_samples)
        outputs.append(samples[:, :, 0, :].mean(dim=1).numpy().astype(np.float64))
    sampling_seconds = time.time() - started
    output = np.concatenate(outputs, axis=0)
    segment_mae = float(np.mean(np.abs(output - test_resp).mean(axis=1)))

    sampler, nfe = SAMPLERS[sampling]
    label = DATASET_CONFIGS[dataset_key]["label"]
    window_rows, errors, duty_reference, duty_predicted = [], [], [], []
    divisor, offset, arrays = 0, 0, {}
    for index, recording in enumerate(recordings):
        count = len(recording["ppg"])
        scored = score_recording(output[offset:offset + count], test_resp[offset:offset + count])
        divisor += scored["official_divisor"]
        errors.append(scored["absolute_error"])
        for row in scored["windows"]:
            duty_reference.append(row["Duty_Reference"])
            duty_predicted.append(row["Duty_Predicted"])
            window_rows.append({"Dataset": label, "Model": variant.label, "Sampling": f"{sampler} {nfe}",
                                "Subject": subject, "Recording": recording["name"], **row})
        arrays[f"recording{index}_prediction"] = scored["prediction"].astype(np.float32)
        arrays[f"recording{index}_reference"] = scored["reference"].astype(np.float32)
        arrays[f"recording{index}_ppg"] = np.concatenate(test_ppg[offset:offset + count]).astype(np.float32)
        offset += count
    window_errors = np.asarray([row["AE"] for row in window_rows], dtype=np.float64)
    summary = {
        "Dataset": label,
        "Model": variant.label,
        "Folder": variant.name,
        "Sampling": f"{sampler} {nfe}",
        "Sampler": sampler,
        "NFE": nfe,
        "Subject": subject,
        "Recordings": len(recordings),
        "Segments": int(len(test_ppg)),
        "Constant_Segments": constant,
        "N_Windows": int(len(window_rows)),
        "RR_MAE": float(window_errors.mean()) if len(window_errors) else float("nan"),
        "RR_MAE_Official_Code": float(window_errors.sum() / divisor) if divisor > 0 else float("nan"),
        "Waveform_MAE": float(np.concatenate(errors).mean()),
        "Segment_MAE": segment_mae,
        "Duty_Cycle_Corr": safe_corrcoef(duty_reference, duty_predicted),
        "N_Samples": int(args.n_samples),
        "Training_Seconds": float(training_seconds),
        "Sampling_Seconds": float(sampling_seconds),
        "Training_Key": key,
    }
    result_dir, prediction_dir = root / "Fold_Results", root / "Predictions"
    result_dir.mkdir(parents=True, exist_ok=True)
    prediction_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        prediction_dir / f"{subject}_{sampling}.npz",
        recording_names=np.asarray([r["name"] for r in recordings]),
        segment_output=output.astype(np.float32),
        **arrays,
    )
    pd.DataFrame(window_rows).to_csv(result_dir / f"{subject}_{sampling}_windows.csv", index=False,
                                     encoding="utf-8-sig")
    write_json(result_dir / f"{subject}_{sampling}.json", summary)
    return summary


def fold_done(root: Path, subject: str, sampling: str, key: dict, n_samples: int) -> bool:
    summary = read_json(root / "Fold_Results" / f"{subject}_{sampling}.json")
    return bool(summary and summary.get("Training_Key") == key and summary.get("N_Samples") == n_samples)


# ---------------------------------------------------------------------------
# Experiment
# ---------------------------------------------------------------------------

def run_task(args: argparse.Namespace, results_root: Path, dataset_key: str, variants: Sequence[Variant],
             device: torch.device) -> None:
    label = DATASET_CONFIGS[dataset_key]["label"]
    print(f"[{time.strftime('%H:%M:%S')}] {label}: reading the data ...", flush=True)
    data = load_dataset(dataset_key)
    all_subjects = sorted(data)
    if len(all_subjects) < 2:
        raise RuntimeError(f"{label}: fewer than two subjects")
    subjects = [s for s in all_subjects if not args.subjects or s in {str(x) for x in args.subjects}]
    segments = sum(len(r["ppg"]) for items in data.values() for r in items)
    print(f"{label}: {len(all_subjects)} subjects, {segments} segments of {SEGMENT_SEC} s at {FS} Hz "
          f"(reference: {DATASET_CONFIGS[dataset_key]['reference']})", flush=True)
    failed = []
    for variant in variants:
        root = fold_root(results_root, variant, dataset_key)
        print(f"\n{label}, {variant.label}: LOSOCV over {len(subjects)} of {len(all_subjects)} subjects", flush=True)
        for subject in subjects:
            train_subjects = [s for s in all_subjects if s != subject]
            train_recordings = [r for s in train_subjects for r in data[s]]
            train_ppg, train_resp, _ = stack_scaled(train_recordings)
            key = training_key(args, variant, dataset_key, train_subjects, len(train_ppg))
            todo = [s for s in variant.samplers if not (args.resume and fold_done(root, subject, s, key, args.n_samples))]
            if not todo:
                continue
            try:
                print(f"[{time.strftime('%H:%M:%S')}] {label} {variant.label} {subject}: training on "
                      f"{len(train_ppg)} segments of {len(train_subjects)} subjects, {args.epochs} epochs",
                      flush=True)
                model, seconds = train_fold(args, variant, dataset_key, subject, train_ppg, train_resp, key,
                                            root, device)
                for sampling in todo:
                    summary = test_fold(args, model, variant, dataset_key, subject, data[subject], sampling, root,
                                        device, key, seconds)
                    print(f"[{time.strftime('%H:%M:%S')}] {label} {variant.label} {subject} {summary['Sampling']}: "
                          f"RR MAE {summary['RR_MAE']:.3f} bpm (official code {summary['RR_MAE_Official_Code']:.3f}), "
                          f"waveform MAE {summary['Waveform_MAE']:.3f}, {summary['N_Windows']} windows, "
                          f"sampling {summary['Sampling_Seconds']:.0f} s", flush=True)
                del model
            except Exception as exc:
                failed.append({"Dataset": label, "Model": variant.label, "Subject": subject, "Error": repr(exc)})
                print(f"{label} {variant.label} {subject}: FAILED {exc}", flush=True)
                if is_fatal_error(exc):
                    write_failed(results_root, dataset_key, variants, failed)
                    print("Fatal error (memory / CUDA): the worker stops; it continues with --resume.", flush=True)
                    raise SystemExit(3)
            finally:
                release_memory(device)
    write_failed(results_root, dataset_key, variants, failed)


def write_failed(results_root: Path, dataset_key: str, variants: Sequence[Variant], rows: List[dict]) -> None:
    name = "_".join([DATASET_CONFIGS[dataset_key]["label"]] + [v.name for v in variants])
    results_root.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows, columns=["Dataset", "Model", "Subject", "Error"]).to_csv(
        results_root / f"failed_folds_{name}.csv", index=False, encoding="utf-8-sig"
    )


def task_progress(results_root: Path, dataset_key: str, variant: Variant) -> int:
    folder = fold_root(results_root, variant, dataset_key) / "Fold_Results"
    return len(list(folder.glob("*.json"))) if folder.exists() else 0


def run_tasks_in_subprocesses(args: argparse.Namespace, results_root: Path, dataset_keys: Sequence[str],
                              variants: Sequence[Variant]) -> List[str]:
    """Every dataset and model in its own Python process, so its host and GPU
    memory is returned before the next one; a worker stopped by a fatal error
    is started again with --resume (twice without progress: left)."""
    script = str(Path(__file__).resolve())
    incomplete = []
    for dataset_key in dataset_keys:
        label = DATASET_CONFIGS[dataset_key]["label"]
        for variant in variants:
            attempt, stalled = 0, 0
            while True:
                before = task_progress(results_root, dataset_key, variant)
                command = [sys.executable, "-u", script, *sys.argv[1:], "--datasets", dataset_key,
                           "--variants", variant.name, "--child-task"]
                if attempt > 0 and "--resume" not in command:
                    command.append("--resume")
                print(f"\n=== {label}, {variant.label}: worker process (attempt {attempt + 1}) ===", flush=True)
                code = run_worker(command)
                if code == 0:
                    break
                attempt += 1
                stalled = stalled + 1 if task_progress(results_root, dataset_key, variant) <= before else 0
                if stalled >= 2:
                    print(f"{label}, {variant.label}: worker stopped (exit code {code}) twice without a new "
                          "result; left incomplete (run again with --resume)", flush=True)
                    incomplete.append(f"{label} / {variant.label}")
                    break
                print(f"{label}, {variant.label}: worker stopped (exit code {code}); restarting with --resume",
                      flush=True)
    return incomplete


# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------

def mean_sd(values: pd.Series) -> str:
    values = pd.to_numeric(values, errors="coerce").dropna()
    if values.empty:
        return ""
    sd = values.std(ddof=1) if len(values) > 1 else float("nan")
    return f"{values.mean():.3f} ± {sd:.3f}"


def summarize_results(results_root: Path, dataset_keys: Sequence[str], variants: Sequence[Variant]) -> None:
    summary_root = results_root / "Summary"
    summary_root.mkdir(parents=True, exist_ok=True)
    rows, windows = [], []
    for variant in variants:
        for dataset_key in dataset_keys:
            folder = fold_root(results_root, variant, dataset_key) / "Fold_Results"
            if not folder.exists():
                continue
            for path in sorted(folder.glob("*.json")):
                summary = read_json(path)
                if summary is None:
                    continue
                summary.pop("Training_Key", None)
                rows.append(summary)
                window_path = path.with_name(path.stem + "_windows.csv")
                if window_path.exists():
                    windows.append(pd.read_csv(window_path, dtype={"Subject": str}))
    if not rows:
        print("No fold results to summarize yet.")
        return
    per_subject = pd.DataFrame(rows)
    per_subject.to_csv(summary_root / "per_subject.csv", index=False, encoding="utf-8-sig")
    if windows:
        pd.concat(windows, ignore_index=True).to_csv(summary_root / "windows.csv", index=False, encoding="utf-8-sig")

    table = []
    order = {v.label: i for i, v in enumerate(VARIANTS.values())}
    for (dataset, model, sampling), frame in per_subject.groupby(["Dataset", "Model", "Sampling"], sort=False):
        table.append({
            "Dataset": dataset,
            "Model": model,
            "Sampling": sampling,
            "NFE": int(frame["NFE"].iloc[0]),
            "Window Size": f"{RR_WINDOW // FS}s",
            "N_Subjects": int(frame["Subject"].nunique()),
            "N_Windows": int(frame["N_Windows"].sum()),
            "RR error (bpm, mean ± SD over subjects)": mean_sd(frame["RR_MAE"]),
            "RR error, official code (bpm)": mean_sd(frame["RR_MAE_Official_Code"]),
            "Waveform MAE": mean_sd(frame["Waveform_MAE"]),
            "Segment MAE": mean_sd(frame["Segment_MAE"]),
            "Duty-cycle correlation": mean_sd(frame["Duty_Cycle_Corr"]),
            "RR_MAE_Mean": float(frame["RR_MAE"].mean()),
            "RR_MAE_SD": float(frame["RR_MAE"].std(ddof=1)),
            "RR_MAE_Official_Code_Mean": float(frame["RR_MAE_Official_Code"].mean()),
            "Waveform_MAE_Mean": float(frame["Waveform_MAE"].mean()),
            "Waveform_MAE_SD": float(frame["Waveform_MAE"].std(ddof=1)),
            "_order": (DATASET_ORDER.index(next(k for k, c in DATASET_CONFIGS.items() if c["label"] == dataset)),
                       order.get(model, 99), -int(frame["NFE"].iloc[0])),
        })
    results = pd.DataFrame(table).sort_values("_order").drop(columns="_order")
    results.to_csv(summary_root / "respdiff_results.csv", index=False, encoding="utf-8-sig")

    def layout(models: Sequence[str]) -> pd.DataFrame:
        part = results[results["Model"].isin(models)]
        out = []
        for (model, sampling), frame in part.groupby(["Model", "Sampling"], sort=False):
            row = {"Model": model, "NFE": int(frame["NFE"].iloc[0]), "Window Size": f"{RR_WINDOW // FS}s"}
            for _, item in frame.iterrows():
                row[f"{item['Dataset']} RR error (bpm)"] = item["RR error (bpm, mean ± SD over subjects)"]
                row[f"{item['Dataset']} RR error, official code (bpm)"] = item["RR error, official code (bpm)"]
                row[f"{item['Dataset']} waveform MAE"] = item["Waveform MAE"]
            out.append(row)
        return pd.DataFrame(out)

    layout([VARIANTS["respdiff"].label, VARIANTS["no_spectral_loss"].label]).to_csv(
        summary_root / "table_spectral_loss.csv", index=False, encoding="utf-8-sig")
    multiscale = layout([VARIANTS["no_spectral_loss"].label, VARIANTS["no_multiscale"].label])
    if not multiscale.empty:
        multiscale = multiscale[multiscale["NFE"] == SAMPLERS["ddpm50"][1]]
    multiscale.to_csv(summary_root / "table_multiscale.csv", index=False, encoding="utf-8-sig")

    print("\nRespDiff (60 s windows; mean ± SD over subjects):")
    for _, row in results.iterrows():
        print(f"  {row['Dataset']:9s} {row['Model']:38s} {row['Sampling']:8s} n={row['N_Subjects']:2d}  "
              f"RR {row['RR error (bpm, mean ± SD over subjects)']} bpm (official code "
              f"{row['RR error, official code (bpm)']}), waveform MAE {row['Waveform MAE']}")
    print(f"\nSaved the summary under: {summary_root}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--results-root", type=Path, default=DEFAULT_RESULTS_ROOT)
    parser.add_argument("--datasets", nargs="+", choices=list(DATASET_ORDER), default=list(DATASET_ORDER))
    parser.add_argument("--variants", nargs="+", choices=list(VARIANTS), default=list(VARIANTS),
                        help="Models (folder names; see the table above).")
    parser.add_argument("--subjects", nargs="+", default=None,
                        help="Only these held-out subjects; training still uses all others.")
    parser.add_argument("--epochs", type=int, default=EPOCHS)
    parser.add_argument("--n-samples", type=int, default=N_SAMPLES,
                        help="Samples averaged per segment (official code: 100).")
    parser.add_argument("--resume", action="store_true",
                        help="Reuse finished folds and trained models; continue interrupted training.")
    parser.add_argument("--summary-only", action="store_true")
    parser.add_argument("--dry-run", action="store_true", help="Read and check the data only.")
    parser.add_argument("--single-process", action="store_true",
                        help="Train every dataset and model in this process.")
    parser.add_argument("--child-task", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--no-require-cuda", action="store_true")
    parser.add_argument("--force-cpu", action="store_true")
    parser.add_argument("--threads", type=int, default=None)
    parser.add_argument("--capnobase-path", type=Path, default=None)
    parser.add_argument("--bidmc-path", type=Path, default=None)
    parser.add_argument("--steam2-ppg-dir", type=Path, default=None)
    parser.add_argument("--steam2-rsp-dir", type=Path, default=None)
    return parser.parse_args(argv)


def apply_paths(args: argparse.Namespace) -> None:
    if args.capnobase_path:
        DATASET_CONFIGS["capnobase"]["data_path"] = str(args.capnobase_path)
    if args.bidmc_path:
        DATASET_CONFIGS["bidmc"]["data_path"] = str(args.bidmc_path)
    if args.steam2_ppg_dir:
        DATASET_CONFIGS["steam2"]["ppg_dir"] = str(args.steam2_ppg_dir)
    if args.steam2_rsp_dir:
        DATASET_CONFIGS["steam2"]["rsp_dir"] = str(args.steam2_rsp_dir)


def main(argv: Optional[Sequence[str]] = None) -> None:
    args = parse_args(argv)
    apply_paths(args)
    if args.threads:
        torch.set_num_threads(args.threads)
    results_root = Path(args.results_root)
    dataset_keys = [key for key in DATASET_ORDER if key in args.datasets]
    variants = [VARIANTS[name] for name in VARIANTS if name in args.variants]

    if args.dry_run:
        rows = []
        for dataset_key in dataset_keys:
            data = load_dataset(dataset_key)
            rows += data_check_rows(dataset_key, data)
            label = DATASET_CONFIGS[dataset_key]["label"]
            segments = sum(len(r["ppg"]) for items in data.values() for r in items)
            print(f"{label}: {len(data)} subjects, {sum(len(v) for v in data.values())} recordings, "
                  f"{segments} segments of {SEGMENT_SEC} s at {FS} Hz")
        (results_root / "Summary").mkdir(parents=True, exist_ok=True)
        pd.DataFrame(rows).to_csv(results_root / "Summary" / "data_check.csv", index=False, encoding="utf-8-sig")
        return
    if args.summary_only:
        summarize_results(results_root, dataset_keys, variants)
        return

    if not args.child_task and not args.single_process:
        incomplete = run_tasks_in_subprocesses(args, results_root, dataset_keys, variants)
        if incomplete:
            print("\nIncomplete tasks: " + "; ".join(incomplete))
        summarize_results(results_root, dataset_keys, variants)
        return

    use_cuda = torch.cuda.is_available() and not args.force_cpu
    if not use_cuda and not args.no_require_cuda:
        raise RuntimeError("CUDA is not available (use --no-require-cuda to run on the CPU)")
    device = torch.device("cuda" if use_cuda else "cpu")
    print(f"RespDiff ({CODE_NAME}): device={device}, {args.epochs} epochs, {args.n_samples} samples per segment",
          flush=True)
    for dataset_key in dataset_keys:
        try:
            run_task(args, results_root, dataset_key, variants, device)
        except SystemExit:
            raise
        except Exception as exc:
            if is_fatal_error(exc):
                print(f"{DATASET_CONFIGS[dataset_key]['label']}: fatal error while reading ({exc}); the worker "
                      "stops", flush=True)
                raise SystemExit(3)
            raise
    if not args.child_task:
        summarize_results(results_root, dataset_keys, variants)


if __name__ == "__main__":
    main()
