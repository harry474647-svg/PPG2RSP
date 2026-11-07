r"""261107_DutyCycle_RespDiff_DeepCorrencoder: the duty cycle of the stored
results of RespDiff and of the deep corr-encoder, and, with the same rule,
of the RRWaveNet results in use (CapnoBase: 261007; BIDMC, STEAM2: 261013),
in one workbook with the MAE FFT and PCC of those RRWaveNet results. Only
stored results and the data files are read; nothing is trained.

Duty cycle (the rule of the official RespDiff code, as in 261020_RespDiff /
261028_RespDiff_v3, score_recording)
------------------------------------------------------------------------
* Reference: the respiratory waveform of the data file at 30 Hz (CapnoBase
  capnogram, BIDMC impedance, STEAM2 belt), low-pass filtered (Butterworth
  order 8, 1 Hz, zero phase), cut into non-overlapping 5 s segments from the
  start of the recording, every segment min-max scaled to [0, 1] (a constant
  segment = 0), joined (the rest after the last whole segment is dropped).
* Model output on the same time axis, on [0, 1], low-pass filtered as above:
    RespDiff           the stored prediction (the joined 5 s outputs, already
                       low-pass filtered by the RespDiff run; [0, 1] scale)
    Deep corr-encoder  the stored output (9.6 s windows shifted 1 s and
                       averaged; trained on [0, 1] windows), low-pass filtered
    RRWaveNet          the stored ensemble output (10 s windows at 64 Hz,
                       trained on [-1, 1] windows; its training sign / lag
                       undone, taken at the nearest 30 Hz sample) mapped to
                       [0, 1] by (y + 1) / 2, low-pass filtered (every
                       continuous stretch on its own)
  No output is shifted in time, inverted or rescaled otherwise.
* 60 s windows moved by 20 s; in every window the duty cycle = the share of
  samples >= 0.5 (the official code counts them; share = count / 1800). A
  window where the model has no output (RRWaveNet windows that failed its
  quality check) is left out (N_Windows_Skipped).
* Per subject (all windows of all its recordings, as the official code):
    Duty_Cycle_Corr      Pearson r of reference and model duty cycle over the
                         windows (the official "duty-cycle correlation")
    Duty_MAE, Duty_Bias  mean |model - reference| and mean (model - reference)
                         of the window duty cycles (share, 0-1)
    Duty_Reference_Mean, Duty_Predicted_Mean
* RespDiff check: the same correlation on its own stored reference
  (Duty_Cycle_Corr_Own_Reference) and the value its run stored
  (Fold_Results\{subject}_ddpm50.json, Duty_Cycle_Corr_Stored); the two are
  the same number when this code follows the official rule.
* CapnoBase RRWaveNet was trained on a sine of the respiratory phase between
  the expert breath labels (261007), not on the capnogram: its duty cycle is
  that of a sine (about 0.5) whatever the capnogram does.

RRWaveNet MAE FFT and PCC (as the run reports them, per_subject_fft)
--------------------------------------------------------------------
Scored minutes of the run ({subject}_ensemble_RR60s_windows.csv: reference
valid and an FFT rate): MAE FFT = mean |FFT rate of the predicted waveform -
reference rate| (breaths/min), PCC = mean Pearson r of the predicted and
target waveform in the minute.
  CapnoBase  261007 ({root-261007}\full\01_w10s_nonoverlap\CapnoBase\), ensemble of 3
  BIDMC      261013 ({root-261013}\{selected wd}\01_w10s_nonoverlap\BIDMC\), WD of every fold
  STEAM2     261013 (as BIDMC)

Stored results (subfolders of)
------------------------------
  RespDiff           D:\PPG2RSP_RRWaveNet_Inspired\261020_RespDiff_v2
                     respdiff\{dataset}\Predictions\{subject}_ddpm50.npz,
                     respdiff\{dataset}\Fold_Results\{subject}_ddpm50.json
  Deep corr-encoder  D:\PPG2RSP_RRWaveNet_Inspired\261028_Deep_correncoder
                     {dataset}\Predictions\{subject}_outputs.npz,
                     {dataset}\Fold_Results\{subject}_summary.json
  RRWaveNet          D:\PPG2RSP_RRWaveNet_Inspired\261007_RRWaveNet_NoEvent_v1,
                     D:\PPG2RSP_RRWaveNet_Inspired\261013_RRWavNet_STEAM_BIDMC_v1
  GT                 the data files of the three scripts (as 261030)

Outputs (default D:\PPG2RSP_RRWaveNet_Inspired\261107_DutyCycle_RespDiff_DeepCorrencoder)
  261107_DutyCycle_Results.xlsx
    Summary            per dataset and model: Mean ± SD (Median) over subjects
                       of MAE FFT, PCC (RRWaveNet) and the duty cycle measures
    RRWaveNet_Results  per subject: MAE FFT, PCC, duty cycle (261007 CapnoBase,
                       261013 BIDMC / STEAM2)
    RespDiff_Duty, CorrEncoder_Duty   per subject
    Duty_Windows       every 60 s window of every model
    Coverage, Notes
  and the same tables as CSV files.

Run (in D:\PPG2RSP_RRWaveNet_Inspired\261107_DutyCycle_RespDiff_DeepCorrencoder):
  python 261107_DutyCycle_RespDiff_DeepCorrencoder.py
Options: --datasets capnobase bidmc steam2, --respdiff-root, --respdiff-sampling,
  --correncoder-root, --root-261007, --root-261013, --output-root,
  --capnobase-path, --bidmc-path, --steam2-ppg-dir, --steam2-rsp-dir
"""

import argparse
import csv
import json
import math
import re
from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import scipy.io as sio
from scipy.signal import butter, resample_poly, sosfiltfilt

# ---------------------------------------------------------------------------
# Folders and settings
# ---------------------------------------------------------------------------

WORKSPACE = Path(r"D:\PPG2RSP_RRWaveNet_Inspired")
PROJECT_ROOT = WORKSPACE / "261107_DutyCycle_RespDiff_DeepCorrencoder"
RESPDIFF_ROOT = WORKSPACE / "261020_RespDiff_v2"
CORRENCODER_ROOT = WORKSPACE / "261028_Deep_correncoder"
ROOT_261007 = WORKSPACE / "261007_RRWaveNet_NoEvent_v1"
ROOT_261013 = WORKSPACE / "261013_RRWavNet_STEAM_BIDMC_v1"
RESPDIFF_SAMPLING = "ddpm50"

FS = 30  # the 30 Hz of the GT, RespDiff and the corr-encoder
FS30 = FS
RRWAVENET_FS = 64
RRWAVENET_WINDOW = "01_w10s_nonoverlap"
WINDOW = RRWAVENET_WINDOW
RRWAVENET_LOWPASS_ORDER = 8

# Duty cycle (official RespDiff code)
SEGMENT = 5 * FS  # 5 s segments of the reference, each min-max scaled to [0, 1]
DUTY_WINDOW = 60 * FS
DUTY_HOP = 20 * FS
DUTY_THRESHOLD = 0.5
DUTY_LOWPASS_HZ = 1.0
DUTY_LOWPASS_ORDER = 8

MODELS = ("RRWaveNet", "RespDiff", "Deep corr-encoder")
DATASET_ORDER = ("capnobase", "bidmc", "steam2")
# dataset -> RRWaveNet run in use: (run, folder argument, WD selected per fold)
RRWAVENET_RUNS = {"capnobase": ("261007", "root_261007", False), "bidmc": ("261013", "root_261013", True),
                  "steam2": ("261013", "root_261013", True)}

# Data files of the GT (as 261030 and the three scripts)
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
_NUMBER_PATTERN = re.compile(r"[-+]?(?:\d+\.?\d*|\.\d+)(?:[eE][-+]?\d+)?")
STEAM2_CHANNELS = {"PPG": ("PPG", "PLETH", "BVP"), "RSP": ("RSP", "RESP", "RESPIRATION", "BELT")}
STEAM2_SESSION_PATTERN = re.compile(r"^(?P<subject>.+?)[_\- ]*session[_\- ]*(?P<session>\d+)$", re.IGNORECASE)
# clipped samples of the GT (261030; listed in Coverage, the duty cycle uses every sample as the official code)
CONFIG = {"saturation_min_sec": 0.1, "saturation_margin_sec": 0.1}


# ---------------------------------------------------------------------------
# Copied from 261030_Waveform_Morphology_Analysis (data files of the GT, stored results)
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
    index = np.round(np.flatnonzero(mask) * FS / float(fs)).astype(np.int64)
    out[index[(index >= 0) & (index < length)]] = True
    width = int(round(margin_sec * FS))
    if width and out.any():
        out = np.convolve(out.astype(float), np.ones(2 * width + 1), mode="same") > 0.5
    return out


def reference_recordings(dataset_key: str, config: dict) -> Dict[str, Dict[str, Tuple[np.ndarray, np.ndarray]]]:
    """Subject -> {recording name: (GT at 30 Hz, its clipped samples)}, named
    and cut as the deep corr-encoder names and cuts them (the length of the
    shorter of PPG and reference at 30 Hz)."""
    cfg = DATASET_CONFIGS[dataset_key]
    modality = MODALITIES[dataset_key]
    out: Dict[str, Dict[str, Tuple[np.ndarray, np.ndarray]]] = {}

    def add(subject: str, name: str, ppg, resp, ppg_fs: float, resp_fs: float) -> None:
        ppg30, resp30 = to_fs(ppg, ppg_fs), to_fs(resp, resp_fs)
        length = min(len(ppg30), len(resp30))
        clipped = to_fs_mask(saturated_samples(resp, resp_fs, modality, config), resp_fs, length,
                             config["saturation_margin_sec"])
        out.setdefault(subject, {})[name] = (resp30[:length], clipped)

    if dataset_key == "capnobase":
        csv_dir = Path(cfg["data_path"])
        for case_id in capnobase_case_ids(csv_dir):
            try:
                signal_path = capnobase_file(csv_dir, case_id, "signal")
                ppg, co2, _ = read_capnobase_signals(signal_path)
                param_path = capnobase_file(csv_dir, case_id, "param")
                param = read_capnobase_fields(param_path) if param_path else {}
                ppg_fs, _ = _sampling_rate(param, "pleth", cfg["default_fs"])
                co2_fs, _ = _sampling_rate(param, "co2", ppg_fs)
                add(case_id, case_id, ppg, co2, ppg_fs, co2_fs)
            except Exception as exc:
                print(f"[CapnoBase {case_id}] GT not read: {exc}")
    elif dataset_key == "bidmc":
        mat = sio.loadmat(cfg["data_path"])
        for index, record in enumerate(mat["data"][0]):
            subject = f"{index + 1:02d}"
            try:
                ppg = np.asarray(record["ppg"][0, 0]["v"]).squeeze()
                imp = np.asarray(record["ref"][0, 0]["resp_sig"][0, 0]["imp"][0, 0]["v"]).squeeze()
                add(subject, subject, ppg, imp, float(cfg["default_fs"]), float(cfg["default_fs"]))
            except Exception as exc:
                print(f"[BIDMC {subject}] GT not read: {exc}")
    else:
        for subject, items in steam2_session_files(Path(cfg["ppg_dir"]), Path(cfg["rsp_dir"])).items():
            try:
                for name, ppg_path, rsp_path in items:
                    ppg = read_eeglab_channel(ppg_path, "PPG", cfg["default_fs"])
                    rsp = read_eeglab_channel(rsp_path, "RSP", cfg["default_fs"])
                    pieces, _ = steam2_session_pieces(ppg, rsp)
                    for k, (p, r, p_fs, r_fs) in enumerate(pieces):
                        add(subject, f"{name}_piece{k + 1}", p, r, p_fs, r_fs)
            except Exception as exc:
                print(f"[STEAM2 {subject}] GT not read: {exc}")
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


def read_json(path: Path) -> Optional[dict]:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


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
    path = root / "Data_Quality" / RRWAVENET_WINDOW / f"{label}_window_quality.csv"
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



# ---------------------------------------------------------------------------
# Copied from 261106_RRWaveNet_Performance_Boxplots_v2 (RRWaveNet runs: fold files, scored 
# minutes = MAE FFT and PCC of the run, RRWaveNet on the GT axis)
# ---------------------------------------------------------------------------

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


def finite_or_median(x: np.ndarray) -> np.ndarray:
    """The runs' sanitize_signal: non-finite samples -> median of the rest."""
    x = np.asarray(x, dtype=np.float64).reshape(-1)
    ok = np.isfinite(x)
    if ok.all():
        return x
    if not ok.any():
        return np.zeros_like(x)
    x = x.copy()
    x[~ok] = float(np.median(x[ok]))
    return x


def reconstruct(arrays: dict, segment_id: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Predicted and target waveform of one segment on its time axis
    (overlap-add mean of the windows, as reconstruct_segment of the runs)."""
    predictions, targets = arrays["predictions"], arrays["targets"]
    index = np.flatnonzero(np.asarray(arrays["segment_ids"]).astype(int) == int(segment_id))
    if not len(index):
        return np.empty(0), np.empty(0), np.empty(0, bool)
    n = int(predictions.shape[1])
    starts = np.asarray(arrays["window_starts"]).astype(int)
    length = int(starts[index].max()) + n
    p_sum, t_sum, cover = np.zeros(length), np.zeros(length), np.zeros(length, int)
    for i in index:
        a = starts[i]
        p_sum[a:a + n] += finite_or_median(predictions[i])[:n]
        t_sum[a:a + n] += finite_or_median(targets[i])[:n]
        cover[a:a + n] += 1
    valid = cover > 0
    pred, true = np.full(length, np.nan), np.full(length, np.nan)
    pred[valid] = p_sum[valid] / cover[valid]
    true[valid] = t_sum[valid] / cover[valid]
    return pred, true, valid


def run_pcc(x: np.ndarray, y: np.ndarray) -> float:
    x, y = finite_or_median(x), finite_or_median(y)
    if len(x) < 2 or np.std(x) < 1e-8 or np.std(y) < 1e-8:
        return np.nan
    return float(np.corrcoef(x, y)[0, 1])


def true_flags(values: pd.Series) -> pd.Series:
    return values.astype(str).str.strip().str.lower().isin(("true", "1", "1.0"))


def fold_minutes(fold: dict, dataset: str) -> Tuple[pd.DataFrame, dict]:
    """Scored minutes of one fold with the waveform MAE, and a coverage row."""
    info = {"Dataset": dataset, "Subject_Tag": fold["Tag"], "Variant": fold["Variant"],
            "Minutes_File": str(fold["Minutes_File"]), "Outputs_File": str(fold["Outputs_File"])}
    for key, path in (("Minutes_File", fold["Minutes_File"]), ("Outputs_File", fold["Outputs_File"])):
        if not path.exists():
            info["Status"] = f"missing {key.replace('_', ' ').lower()}"
            return pd.DataFrame(), info
    minutes = pd.read_csv(fold["Minutes_File"], dtype={"Subject": str}, encoding="utf-8-sig")
    if minutes.empty:
        info["Status"] = "no minutes"
        return pd.DataFrame(), info
    minutes["AE_Wave_FFT"] = pd.to_numeric(minutes["AE_Wave_FFT"], errors="coerce")
    scored = minutes[true_flags(minutes["Ref_Valid"]) & np.isfinite(minutes["AE_Wave_FFT"])].copy()
    info.update({"N_Minutes_Stored": int(len(minutes)), "N_Minutes_Scored": int(len(scored))})
    if scored.empty:
        info["Status"] = "no scored minute"
        return pd.DataFrame(), info
    with np.load(fold["Outputs_File"], allow_pickle=False) as data:
        arrays = {key: data[key] for key in ("predictions", "targets", "segment_ids", "window_starts")}
    series: Dict[int, tuple] = {}
    wave_mae, pcc_check = [], []
    for _, row in scored.iterrows():
        segment = int(row["Segment_ID"])
        if segment not in series:
            series[segment] = reconstruct(arrays, segment)
        pred, true, valid = series[segment]
        a = int(round(float(row["RR_Window_Start_Sec"]) * RRWAVENET_FS))
        b = int(round(float(row["RR_Window_End_Sec"]) * RRWAVENET_FS))
        keep = valid[a:b]
        if b > len(pred) or not keep.any():
            wave_mae.append(np.nan)
            pcc_check.append(np.nan)
            continue
        p, t = pred[a:b][keep], true[a:b][keep]
        wave_mae.append(float(np.mean(np.abs(p - t))))
        pcc_check.append(run_pcc(t, p))
    scored["Waveform_MAE"] = wave_mae
    scored["PCC_Recomputed"] = pcc_check
    stored = pd.to_numeric(scored["Wave_PCC"], errors="coerce").to_numpy()
    diff = np.abs(stored - np.asarray(pcc_check, float))
    info["PCC_Check_Max_Abs_Diff"] = float(np.nanmax(diff)) if np.isfinite(diff).any() else np.nan
    info["N_Minutes_No_Waveform"] = int(np.isnan(wave_mae).sum())
    info["Status"] = "ok"
    subject = str(scored["Subject"].iloc[0]) if "Subject" in scored.columns else fold["Tag"]
    info["Subject"] = subject
    out = pd.DataFrame({
        "Dataset": dataset,
        "Subject": subject,
        "Variant": fold["Variant"],
        "Segment_ID": scored["Segment_ID"].astype(int).to_numpy(),
        "Minute_Start_Sec": scored["RR_Window_Start_Sec"].to_numpy(float),
        "GT_RR": pd.to_numeric(scored["GT_Breath_Count"], errors="coerce").to_numpy(float),
        "Pred_RR": pd.to_numeric(scored["Pred_Wave_FFT_RR"], errors="coerce").to_numpy(float),
        "RR_AE": scored["AE_Wave_FFT"].to_numpy(float),
        "PCC": stored,
        "Waveform_MAE": scored["Waveform_MAE"].to_numpy(float),
    })
    return out, info


def per_subject(minutes: pd.DataFrame) -> pd.DataFrame:
    return (
        minutes.groupby(["Dataset", "Subject", "Variant"], sort=False)
        .agg(N_Minutes=("RR_AE", "size"), RR_MAE=("RR_AE", "mean"), RR_MAE_SD=("RR_AE", "std"),
             PCC=("PCC", "mean"), PCC_SD=("PCC", "std"),
             Waveform_MAE=("Waveform_MAE", "mean"), Waveform_MAE_SD=("Waveform_MAE", "std"))
        .reset_index()
    )


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



# ---------------------------------------------------------------------------
# Duty cycle (official RespDiff code)
# ---------------------------------------------------------------------------

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


def load_respdiff_references(root: Path, label: str, sampling: str) -> Dict[str, Dict[str, np.ndarray]]:
    """Subject -> {recording name: the reference RespDiff stored with its prediction}."""
    folder = root / "respdiff" / label
    subjects = {}
    for path in sorted((folder / "Predictions").glob(f"*_{sampling}.npz")):
        subject = path.name[: -len(f"_{sampling}.npz")]
        with np.load(path, allow_pickle=False) as data:
            names = [str(name) for name in data["recording_names"]]
            subjects[subject] = {name: data[f"recording{i}_reference"].astype(np.float64)
                                 for i, name in enumerate(names) if f"recording{i}_reference" in data.files}
    return subjects


# ---------------------------------------------------------------------------
# One dataset
# ---------------------------------------------------------------------------

def dataset_duty(args: argparse.Namespace, key: str) -> dict:
    label = DATASET_CONFIGS[key]["label"]
    run, root_arg, selected = RRWAVENET_RUNS[key]
    rr_root = Path(getattr(args, root_arg))
    rd_root, ce_root = Path(args.respdiff_root), Path(args.correncoder_root)
    corr = load_correncoder(ce_root, label)
    respdiff = load_respdiff(rd_root, label, args.respdiff_sampling)
    respdiff_refs = load_respdiff_references(rd_root, label, args.respdiff_sampling)
    folds = fold_files(rr_root, label, selected)
    minutes, coverage = [], []
    for fold in folds:
        frame, info = fold_minutes(fold, label)
        coverage.append({"Model": f"RRWaveNet {run}", **info})
        if not frame.empty:
            minutes.append(frame)
    rr_subjects = per_subject(pd.concat(minutes, ignore_index=True)) if minutes else pd.DataFrame()
    outputs = {fold["Tag"]: fold["Outputs_File"] for fold in folds if fold["Outputs_File"].exists()}
    alignment = rrwavenet_alignment_table(rr_root, label)
    print(f"{label}: RRWaveNet {run} {len(outputs)} subjects (alignment table "
          f"{'found' if alignment else 'not found, lag 0'}), RespDiff {len(respdiff)}, corr-encoder {len(corr)}",
          flush=True)
    print(f"[{label}] reading the GT from the data files ...", flush=True)
    recordings = reference_recordings(key, CONFIG)
    subjects = sorted(set(recordings) & (set(corr) | set(respdiff) | set(outputs)))
    for name, found in (("RespDiff", respdiff), ("Deep corr-encoder", corr), (f"RRWaveNet {run}", outputs)):
        for subject in sorted(set(found) - set(recordings)):
            coverage.append({"Model": name, "Dataset": label, "Subject": subject,
                             "Status": "no GT recording of this subject in the data files"})
    subject_rows, window_rows = [], []
    for subject in subjects:
        corr_by_name = {r["name"]: r for r in corr.get(subject, [])}
        segments = load_rrwavenet(outputs[subject]) if subject in outputs else {}
        pooled = {model: [] for model in MODELS}
        own = []
        for k, (name, (gt, clipped)) in enumerate(recordings[subject].items()):
            reference = official_reference(gt)
            times = np.arange(len(reference)) / FS30
            predictions = {}
            index = corr_by_name[name]["index"] if name in corr_by_name else k
            if index in segments:
                y = rrwavenet_track(segments[index], alignment.get((subject, index)), key, np.asarray(gt, float),
                                    0.0)[:len(reference)]
                predictions["RRWaveNet"] = lowpass_official(0.5 * (y + 1.0))
            if name in respdiff.get(subject, {}):
                predictions["RespDiff"] = Track(respdiff[subject][name], FS30).at(times)
            if name in corr_by_name:
                predictions["Deep corr-encoder"] = lowpass_official(Track(corr_by_name[name]["output"],
                                                                         FS30).at(times))
            for model, prediction in predictions.items():
                rows = duty_windows(reference, prediction)
                pooled[model] += rows
                window_rows += [{"Dataset": label, "Model": model, "Subject": subject, "Recording": name, **r}
                                for r in rows]
            if name in respdiff.get(subject, {}) and name in respdiff_refs.get(subject, {}):
                own += duty_windows(respdiff_refs[subject][name], respdiff[subject][name])
        for model, rows in pooled.items():
            if not rows:
                continue
            row = {"Dataset": label, "Model": model, "Subject": subject, "Recordings": len(recordings[subject]),
                   **duty_summary(rows)}
            if model == "RespDiff":
                stored = read_json(rd_root / "respdiff" / label / "Fold_Results"
                                   / f"{subject}_{args.respdiff_sampling}.json") or {}
                row["Duty_Cycle_Corr_Own_Reference"] = duty_summary(own)["Duty_Cycle_Corr"] if own else math.nan
                row["Duty_Cycle_Corr_Stored"] = float(stored.get("Duty_Cycle_Corr", math.nan))
            subject_rows.append(row)
        print(f"  [{label}] {subject}: " + ", ".join(f"{m} {len(r)} windows" for m, r in pooled.items() if r),
              flush=True)
    duty = pd.DataFrame(subject_rows)
    rr = pd.DataFrame()
    if not rr_subjects.empty:
        rr = rr_subjects.rename(columns={"RR_MAE": "MAE_FFT", "RR_MAE_SD": "MAE_FFT_SD"})
        rr.insert(1, "Run", run)
        own_duty = duty[duty["Model"] == "RRWaveNet"].drop(columns=["Dataset", "Model"]) if not duty.empty \
            else pd.DataFrame(columns=["Subject"])
        own_duty = own_duty.drop(columns=[c for c in ("Duty_Cycle_Corr_Own_Reference", "Duty_Cycle_Corr_Stored")
                                          if c in own_duty])
        rr = rr.merge(own_duty, on="Subject", how="left")
    return {"duty": duty, "windows": pd.DataFrame(window_rows), "rr": rr, "coverage": pd.DataFrame(coverage)}


# ---------------------------------------------------------------------------
# Summary, workbook, main
# ---------------------------------------------------------------------------

SUMMARY_MEASURES = (("MAE_FFT", 2), ("PCC", 3), ("Duty_Cycle_Corr", 3), ("Duty_MAE", 3), ("Duty_Bias", 3),
                    ("Duty_Reference_Mean", 3), ("Duty_Predicted_Mean", 3))


def mean_sd_median(values) -> Tuple[float, float, float, int]:
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if not len(values):
        return math.nan, math.nan, math.nan, 0
    sd = float(values.std(ddof=1)) if len(values) > 1 else math.nan
    return float(values.mean()), sd, float(np.median(values)), int(len(values))


def summary_table(rr: pd.DataFrame, duty: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for dataset in [DATASET_CONFIGS[k]["label"] for k in DATASET_ORDER]:
        for model in MODELS:
            part = rr[rr["Dataset"] == dataset] if model == "RRWaveNet" and not rr.empty else \
                duty[(duty["Dataset"] == dataset) & (duty["Model"] == model)] if not duty.empty else pd.DataFrame()
            if part.empty:
                continue
            name = f"RRWaveNet {part['Run'].iloc[0]}" if model == "RRWaveNet" else model
            row = {"Dataset": dataset, "Model": name, "N_Subjects": int(part["Subject"].nunique())}
            for measure, digits in SUMMARY_MEASURES:
                m, s, md, n = mean_sd_median(part[measure]) if measure in part else (math.nan, math.nan, math.nan, 0)
                row[measure] = f"{m:.{digits}f} ± {s:.{digits}f} ({md:.{digits}f})" if n else ""
            for measure, _ in SUMMARY_MEASURES:
                m, s, md, n = mean_sd_median(part[measure]) if measure in part else (math.nan, math.nan, math.nan, 0)
                row.update({f"{measure}_Mean": m, f"{measure}_SD": s, f"{measure}_Median": md, f"{measure}_N": n})
            rows.append(row)
    return pd.DataFrame(rows)


def notes() -> pd.DataFrame:
    lines = [
        ("Duty cycle", "share of samples >= 0.5 in every 60 s window (moved by 20 s) of the [0, 1] reference and "
                       "model waveform (official RespDiff code; the code counts the samples)"),
        ("Reference", "GT of the data file, 30 Hz, low-pass 1 Hz (Butterworth 8, zero phase), 5 s segments each "
                      "min-max scaled to [0, 1], joined"),
        ("RespDiff", "stored prediction (joined 5 s outputs, low-pass filtered by its run)"),
        ("Deep corr-encoder", "stored output, low-pass filtered as the reference"),
        ("RRWaveNet", "stored ensemble output (training sign / lag undone) mapped to [0, 1] by (y + 1) / 2, "
                      "low-pass filtered; windows without output left out"),
        ("Duty_Cycle_Corr", "Pearson r of reference and model duty cycle over all windows of the subject "
                            "(official duty-cycle correlation)"),
        ("Duty_MAE / Duty_Bias", "mean |model - reference| / mean (model - reference) of the window duty cycles"),
        ("RespDiff check", "Duty_Cycle_Corr_Own_Reference: on RespDiff's own stored reference; "
                           "Duty_Cycle_Corr_Stored: the value its run stored"),
        ("MAE FFT / PCC", "RRWaveNet: mean over the run's scored minutes of |FFT rate - reference rate| and of "
                          "the waveform Pearson r (per_subject_fft of the run); CapnoBase 261007, BIDMC / STEAM2 "
                          "261013"),
        ("Summary", "Mean ± SD (Median) over subjects"),
        ("CapnoBase RRWaveNet", "trained on a sine of the respiratory phase (expert breath labels), not on the "
                                "capnogram: its duty cycle is that of a sine"),
    ]
    return pd.DataFrame(lines, columns=["Item", "Description"])


def write_excel(path: Path, sheets: Dict[str, pd.DataFrame]) -> None:
    try:
        with pd.ExcelWriter(path, engine="openpyxl") as writer:
            for name, frame in sheets.items():
                frame.to_excel(writer, index=False, sheet_name=name[:31])
    except Exception as exc:
        print(f"Excel export failed ({exc}); the CSV files are complete.")


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="261107 duty cycle of RespDiff, the deep corr-encoder and "
                                                 "RRWaveNet (261007 CapnoBase, 261013 BIDMC / STEAM2)")
    parser.add_argument("--datasets", nargs="+", choices=list(DATASET_ORDER), default=list(DATASET_ORDER))
    parser.add_argument("--respdiff-root", type=Path, default=RESPDIFF_ROOT)
    parser.add_argument("--respdiff-sampling", default=RESPDIFF_SAMPLING)
    parser.add_argument("--correncoder-root", type=Path, default=CORRENCODER_ROOT)
    parser.add_argument("--root-261007", type=Path, default=ROOT_261007)
    parser.add_argument("--root-261013", type=Path, default=ROOT_261013)
    parser.add_argument("--output-root", type=Path, default=PROJECT_ROOT)
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
    out = Path(args.output_root)
    out.mkdir(parents=True, exist_ok=True)
    parts = {"duty": [], "windows": [], "rr": [], "coverage": []}
    for key in DATASET_ORDER:
        if key not in args.datasets:
            continue
        try:
            result = dataset_duty(args, key)
        except Exception as exc:
            print(f"[{DATASET_CONFIGS[key]['label']}] not analysed: {type(exc).__name__}: {exc}")
            parts["coverage"].append(pd.DataFrame([{"Dataset": DATASET_CONFIGS[key]["label"],
                                                    "Status": f"not analysed: {exc}"}]))
            continue
        for name in parts:
            if not result[name].empty:
                parts[name].append(result[name])
    frames = {name: pd.concat(p, ignore_index=True) if p else pd.DataFrame() for name, p in parts.items()}
    duty, rr = frames["duty"], frames["rr"]
    summary = summary_table(rr, duty)
    respdiff = duty[duty["Model"] == "RespDiff"].drop(columns="Model") if not duty.empty else pd.DataFrame()
    corr = duty[duty["Model"] == "Deep corr-encoder"].drop(columns="Model") if not duty.empty else pd.DataFrame()
    respdiff = respdiff.dropna(axis=1, how="all")
    corr = corr.drop(columns=[c for c in ("Duty_Cycle_Corr_Own_Reference", "Duty_Cycle_Corr_Stored")
                              if c in corr])
    tables = {"Summary": summary, "RRWaveNet_Results": rr, "RespDiff_Duty": respdiff, "CorrEncoder_Duty": corr,
              "Duty_Windows": frames["windows"], "Coverage": frames["coverage"], "Notes": notes()}
    for name, frame in tables.items():
        frame.to_csv(out / f"{name}.csv", index=False, encoding="utf-8-sig")
    write_excel(out / "261107_DutyCycle_Results.xlsx", tables)
    settings = {"respdiff_root": str(args.respdiff_root), "correncoder_root": str(args.correncoder_root),
                "root_261007": str(args.root_261007), "root_261013": str(args.root_261013),
                "duty_window_sec": DUTY_WINDOW / FS30, "duty_hop_sec": DUTY_HOP / FS30,
                "duty_threshold": DUTY_THRESHOLD, "segment_sec": SEGMENT / FS30,
                "lowpass_hz": DUTY_LOWPASS_HZ, "lowpass_order": DUTY_LOWPASS_ORDER}
    (out / "settings.json").write_text(json.dumps(settings, indent=2), encoding="utf-8")
    print("\nMean ± SD (Median) over subjects:")
    for _, row in summary.iterrows():
        print(f"  {row['Dataset']:<9s} {row['Model']:<18s} n={row['N_Subjects']:<3d} "
              + (f"MAE FFT {row['MAE_FFT']} | PCC {row['PCC']} | " if row.get("MAE_FFT") else "")
              + f"duty r {row['Duty_Cycle_Corr']} | duty MAE {row['Duty_MAE']}")
    if not respdiff.empty and "Duty_Cycle_Corr_Stored" in respdiff:
        diff = (respdiff["Duty_Cycle_Corr_Own_Reference"] - respdiff["Duty_Cycle_Corr_Stored"]).abs()
        print(f"RespDiff check (own reference vs stored): largest difference {diff.max():.2g} "
              f"over {int(diff.notna().sum())} subjects")
    print(f"\nSaved under: {out}")


if __name__ == "__main__":
    main()
