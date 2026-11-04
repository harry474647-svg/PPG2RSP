r"""261104_Waveform_Overlay: the respiratory waveforms of the reference (GT),
RespDiff, the deep corr-encoder and RRWaveNet drawn over one another, every
recording of every subject, from the stored results (nothing is trained).

Models and stored results (the newest of each; read as in
261030_Waveform_Morphology_Analysis, whose readers this script copies)
-------------------------------------------------------------------------
  RespDiff           261028_RespDiff_v3 / 261101_RespDiff_v4 (the same folder,
                     DDPM 50): {root}\respdiff\{dataset}\Predictions\{subject}_ddpm50.npz
                     (joined and low-pass filtered 5 s outputs, 30 Hz); a fold
                     counts when Fold_Results\{subject}_ddpm50.json exists
  Deep corr-encoder  261028_Deep_correncoder:
                     {root}\{dataset}\Predictions\{subject}_outputs.npz
                     (9.6 s outputs shifted 1 s and averaged, 30 Hz; with the
                     names of the recordings and the stored reference)
  RRWaveNet          CapnoBase: 261007_RRWaveNet_Ensemble_NoEvent, full model,
                     10 s windows, ensemble of 3
                     ({root}\full\01_w10s_nonoverlap\CapnoBase\Ensemble_Predictions\);
                     BIDMC, STEAM2: 261013_RRWaveNet_Adult_SubjectVal_WD, the
                     weight decay selected in every fold
                     (WD_Selection\...\{subject}_selection.json ->
                     {root}\full_wd{wd}\01_w10s_nonoverlap\{dataset}\Ensemble_Predictions\);
                     10 s windows at 64 Hz with their segment and start
Every subject with a deep corr-encoder result is drawn (its results carry the
recordings and their names); RespDiff and RRWaveNet are drawn where their
results exist, so subjects whose RespDiff fold is not finished yet show the
other models only (named in the figure and in overlay_index.csv).

GT and time axis
----------------
* GT: the reference as recorded (--reference raw, default): CapnoBase the
  capnogram, BIDMC the impedance pneumogram, STEAM2 the respiratory belt, read
  from the data files as the three scripts read them and resampled to 30 Hz.
  --reference stored (or when the data files cannot be read) uses the
  reference stored with the corr-encoder results instead.
* Every model on the 30 Hz time axis of the GT: RespDiff and the corr-encoder
  as stored; RRWaveNet (64 Hz) at the nearest sample, with the capnogram lag
  (CapnoBase) and the sign (BIDMC) that its target was given in training
  undone (Data_Quality\01_w10s_nonoverlap\{dataset}_window_quality.csv), as
  in 261030. STEAM2: every session piece is one recording.
* Gaps: time without output (RRWaveNet scores only the 10 s windows that
  passed its quality check) is left empty, never filled.

What is drawn
-------------
* One figure per recording: rows of 60 s (--panel-sec), GT in near-black
  (thick, underneath), RespDiff blue, deep corr-encoder orange, RRWaveNet
  green (the colours of 261030), one legend for the figure.
* Scale: the targets of all three models were min-max scaled per window in
  training (RespDiff 5 s to [0, 1], corr-encoder 9.6 s, RRWaveNet 10 s to
  [-1, 1]), so their values are on no common scale. Every waveform (GT and
  each model) is min-max scaled to [-1, 1] on its own (--scale minmax,
  default; its smallest value -> -1, its largest -> +1) or z-scored (--scale
  zscore: mean 0, SD 1), in each 60 s row (--scale-span row, default) or over
  the whole recording (--scale-span recording). Min-max uses the single
  largest and smallest sample, so one spike (e.g. a movement artifact of the
  GT) shrinks the rest of that waveform in its row; z-score is less affected.
  Scaling changes neither the timing nor the shape, nor the r in the legend.
* Smoothing: RRWaveNet's output (an ensemble of 10 s windows at 64 Hz) is
  low-pass filtered with the filter RespDiff applies to its own joined
  output (Butterworth order 8, 1 Hz, zero phase; here in second-order
  sections, which order 8 at 1 Hz needs at 64 Hz), on every continuous run of
  output separately (gaps stay gaps); --rrwavenet-lowpass-hz 0 draws it as
  stored. The corr-encoder (averaged shifted windows) and RespDiff (its own
  1 Hz low-pass) are drawn as stored, the GT unfiltered. The 1 Hz edge is
  above every breathing rate scored (BIDMC, STEAM2 up to 30, CapnoBase up to
  60 breaths/min); the legend r of RRWaveNet is that of the smoothed
  waveform.
* Legend: per model the Pearson r with the GT over the recording (on the
  samples where both exist) and the share of the GT time with output.
* Note for RRWaveNet: on CapnoBase its target is a sine wave of the
  respiratory phase between the expert breath labels (261007), on BIDMC and
  STEAM2 the reference band-passed to 0.1-0.5 Hz; it reproduces the timing of
  the breaths, not the shape of the capnogram or of the raw reference.

Outputs (default D:\PPG2RSP_RRWaveNet_Inspired\261104_Waveform_Overlay)
  {dataset}\{subject}.png or {dataset}\{subject}__{recording}.png
  {dataset}_overlays.pdf   all figures of the dataset, one page each
  overlay_index.csv        per recording: GT source, length, and per model
                           the result used, r with the GT and coverage

Run:               python 261104_Waveform_Overlay.py
Some datasets:     --datasets capnobase bidmc
Some subjects:     --subjects 0009 27
Result folders:    --respdiff-root, --correncoder-root, --rrwavenet-261007-root,
                   --rrwavenet-261013-root, --output-root
Data files (GT):   --capnobase-path, --bidmc-path, --steam2-ppg-dir,
                   --steam2-rsp-dir (default: the paths of the scripts)
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

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd
import scipy.io as sio
from scipy.signal import butter, resample_poly, sosfiltfilt


# ---------------------------------------------------------------------------
# Result folders, data files and settings (as in 261030)
# ---------------------------------------------------------------------------

PROJECT_ROOT = Path(r"D:\PPG2RSP_RRWaveNet_Inspired\261104_Waveform_Overlay")
RESPDIFF_ROOT = Path(r"D:\PPG2RSP_RRWaveNet_Inspired\261020_RespDiff_v2")
CORRENCODER_ROOT = Path(r"D:\PPG2RSP_RRWaveNet_Inspired\261028_Deep_correncoder")
RRWAVENET_261007_ROOT = Path(r"D:\PPG2RSP_RRWaveNet_Inspired\261007_RRWaveNet_NoEvent_v1")
RRWAVENET_261013_ROOT = Path(r"D:\PPG2RSP_RRWaveNet_Inspired\261013_RRWavNet_STEAM_BIDMC_v1")

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
DATASET_ORDER = ("capnobase", "bidmc", "steam2")
MODALITIES = {"capnobase": "capnogram", "bidmc": "impedance", "steam2": "belt"}
MODELS = ("RespDiff", "Deep corr-encoder", "RRWaveNet")
MODEL_COLORS = {"RespDiff": "#2a78d6", "Deep corr-encoder": "#eb6834", "RRWaveNet": "#1baf7a"}
TEXT_PRIMARY = "#0b0b0b"
TEXT_SECONDARY = "#52514e"
GRID_COLOR = "#e4e3df"
FS = 30  # analysis rate: the 30 Hz of RespDiff, the corr-encoder and the GT
RESPDIFF_SAMPLING = "ddpm50"
RRWAVENET_FS = 64
RRWAVENET_WINDOW = "01_w10s_nonoverlap"

CAPNOBASE_FILE_SUFFIXES = (".csv", ".tab")

_NUMBER_PATTERN = re.compile(r"[-+]?(?:\d+\.?\d*|\.\d+)(?:[eE][-+]?\d+)?")

STEAM2_CHANNELS = {"PPG": ("PPG", "PLETH", "BVP"), "RSP": ("RSP", "RESP", "RESPIRATION", "BELT")}

STEAM2_SESSION_PATTERN = re.compile(r"^(?P<subject>.+?)[_\- ]*session[_\- ]*(?P<session>\d+)$", re.IGNORECASE)


# ---------------------------------------------------------------------------
# Readers of the data files and of the stored results (copied unchanged from
# 261030_Waveform_Morphology_Analysis)
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


def rrwavenet_files(dataset_key: str, root_261007: Path, root_261013: Path) -> Tuple[Dict[str, Tuple[Path, str]], Path]:
    label = DATASET_CONFIGS[dataset_key]["label"]
    files = {}
    if dataset_key == "capnobase":
        condition = root_261007 / "full" / RRWAVENET_WINDOW / label
        for path in sorted((condition / "Ensemble_Predictions").glob("*_ensemble_outputs.npz")):
            subject = path.name[: -len("_ensemble_outputs.npz")]
            if (condition / "Fold_Summaries" / f"{subject}_ensemble_summary.json").exists():
                files[subject] = (path, "261007 full")
        return files, root_261007
    for selection in sorted((root_261013 / "WD_Selection" / RRWAVENET_WINDOW / label).glob("*_selection.json")):
        subject = selection.name[: -len("_selection.json")]
        variant = (read_json(selection) or {}).get("Selected_Variant")
        if not variant:
            continue
        condition = root_261013 / variant / RRWAVENET_WINDOW / label
        path = condition / "Ensemble_Predictions" / f"{subject}_ensemble_outputs.npz"
        if path.exists() and (condition / "Fold_Summaries" / f"{subject}_ensemble_summary.json").exists():
            files[subject] = (path, f"261013 {variant}")
    return files, root_261013


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
# Overlay figures
# ---------------------------------------------------------------------------

# Clipped-sample settings of 261030 (reference_recordings marks them; they
# are not used for drawing).
SATURATION_CONFIG = {"saturation_min_sec": 0.1, "saturation_margin_sec": 0.1}
GT_COLOR = TEXT_PRIMARY
GT_NAMES = {"capnobase": "capnogram", "bidmc": "impedance pneumogram", "steam2": "respiratory belt"}
PANEL_SEC = 60.0
RRWAVENET_LOWPASS_HZ = 1.0  # the low-pass of RespDiff's output (order 8, zero phase)
RRWAVENET_LOWPASS_ORDER = 8


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


def zscore(values: np.ndarray) -> np.ndarray:
    """(x - mean) / SD over the finite samples; NaN stays NaN (all NaN when
    fewer than 3 samples or no variation)."""
    x = np.asarray(values, dtype=float)
    out = np.full(x.shape, np.nan)
    keep = np.isfinite(x)
    if keep.sum() >= 3:
        spread = float(np.std(x[keep]))
        if spread > 1e-12:
            out[keep] = (x[keep] - float(np.mean(x[keep]))) / spread
    return out


def safe_name(text: str) -> str:
    return re.sub(r"[^0-9A-Za-z._-]+", "_", str(text))


def style_axis(axis) -> None:
    axis.grid(True, color=GRID_COLOR, linewidth=0.6)
    axis.set_axisbelow(True)
    for side in ("top", "right"):
        axis.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        axis.spines[side].set_color(GRID_COLOR)
    axis.tick_params(colors=TEXT_SECONDARY, labelsize=8)


def overlay_figure(values: Dict[str, np.ndarray], labels: Dict[str, str], title: str, subtitle: str,
                   panel_sec: float, scale: str, span: str):
    """Rows of panel_sec seconds; the GT underneath, then the models."""
    length = len(values["GT"])
    total_sec = length / FS
    # A remainder under 1 s (e.g. 480.03 s) is drawn in the last row, not in a row of its own.
    panels = max(1, int(math.ceil((total_sec - 1.0) / panel_sec)))
    keys = ["GT"] + [m for m in MODELS if m in values]
    scaler = minmax if scale == "minmax" else zscore
    if span == "recording":
        values = {k: scaler(v) for k, v in values.items()}
    fig, axes = plt.subplots(panels, 1, figsize=(15.0, 1.3 + 1.9 * panels), squeeze=False)
    for row, axis in enumerate(axes[:, 0]):
        start = int(round(row * panel_sec * FS))
        end = length if row == panels - 1 else min(length, int(round((row + 1) * panel_sec * FS)))
        times = np.arange(start, end) / FS
        for order, key in enumerate(keys):
            y = values[key][start:end]
            if span == "row":
                y = scaler(y)
            if key == "GT":
                axis.plot(times, y, color=GT_COLOR, linewidth=2.2, alpha=0.85, zorder=1, solid_capstyle="round")
            else:
                axis.plot(times, y, color=MODEL_COLORS[key], linewidth=1.25, zorder=2 + order,
                          solid_capstyle="round")
        axis.set_xlim(row * panel_sec, max((row + 1) * panel_sec, end / FS))
        if scale == "minmax":
            axis.set_ylim(-1.15, 1.15)
            axis.set_yticks([-1, 0, 1])
        axis.set_ylabel("scaled" if scale == "minmax" else "z", color=TEXT_SECONDARY, fontsize=8)
        style_axis(axis)
    axes[-1, 0].set_xlabel("time in the recording (s)", color=TEXT_SECONDARY, fontsize=9)
    handles = [Line2D([0], [0], color=GT_COLOR if k == "GT" else MODEL_COLORS[k],
                      linewidth=2.2 if k == "GT" else 1.6) for k in keys]
    height = fig.get_figheight()
    fig.suptitle(title, x=0.01, ha="left", y=1.0 - 0.12 / height, fontsize=11, color=TEXT_PRIMARY)
    fig.text(0.01, 1.0 - 0.42 / height, subtitle, ha="left", va="top", fontsize=8, color=TEXT_SECONDARY)
    fig.legend(handles, [labels[k] for k in keys], loc="upper left", ncol=len(keys), frameon=False,
               fontsize=9, bbox_to_anchor=(0.005, 1.0 - 0.62 / height))
    fig.subplots_adjust(left=0.045, right=0.99, top=1.0 - 1.05 / height, bottom=0.6 / height,
                        hspace=0.42)
    return fig


def dataset_overlays(args: argparse.Namespace, dataset_key: str, output_root: Path) -> List[dict]:
    label = DATASET_CONFIGS[dataset_key]["label"]
    corr = load_correncoder(Path(args.correncoder_root), label)
    respdiff = load_respdiff(Path(args.respdiff_root), label, args.respdiff_sampling)
    rr_files, rr_root = rrwavenet_files(dataset_key, Path(args.rrwavenet_261007_root),
                                        Path(args.rrwavenet_261013_root))
    alignment_table = rrwavenet_alignment_table(rr_root, label)
    subjects = sorted(corr)
    if args.subjects:
        wanted = {str(s) for s in args.subjects}
        subjects = [s for s in subjects if s in wanted]
    without_corr = sorted((set(respdiff) | set(rr_files)) - set(corr))
    print(f"{label}: results - corr-encoder {len(corr)}, RespDiff {len(respdiff)}, RRWaveNet {len(rr_files)} "
          f"subjects; drawn {len(subjects)}", flush=True)
    if without_corr:
        print(f"  not drawn (no corr-encoder result, which carries the recordings): {' '.join(without_corr)}")
    if not subjects:
        return []
    raw = {}
    if args.reference == "raw":
        print("  reading the GT from the data files ...", flush=True)
        try:
            raw = reference_recordings(dataset_key, SATURATION_CONFIG)
        except Exception as exc:
            print(f"  the data files could not be read ({exc}); the stored corr-encoder reference is used")
    figure_root = output_root / label
    if not args.no_png:
        figure_root.mkdir(parents=True, exist_ok=True)
    pdf = None if args.no_pdf else PdfPages(output_root / f"{label}_overlays.pdf")
    rows = []
    try:
        for subject in subjects:
            rr_segments = load_rrwavenet(rr_files[subject][0]) if subject in rr_files else {}
            for recording in corr[subject]:
                name = recording["name"]
                stored = recording["reference"]
                gt_values, gt_source = stored, "stored corr-encoder reference"
                if raw and name in raw.get(subject, {}):
                    gt_values, gt_source = raw[subject][name][0], "data file"
                times = np.arange(len(gt_values)) / FS
                values = {"GT": np.asarray(gt_values, dtype=float),
                          "Deep corr-encoder": Track(recording["output"], FS).at(times)}
                if name in respdiff.get(subject, {}):
                    values["RespDiff"] = Track(respdiff[subject][name], FS).at(times)
                info = {"Dataset": label, "Subject": subject, "Recording": name, "GT": gt_source,
                        "GT_Min": len(gt_values) / FS / 60.0}
                segment = rr_segments.get(recording["index"])
                if segment is not None:
                    stored_alignment = alignment_table.get((subject, recording["index"]))
                    lag = stored_alignment["lag_sec"] if stored_alignment else 0.0
                    if stored_alignment:
                        sign = stored_alignment["sign"]
                    elif dataset_key == "bidmc":
                        r = target_reference_r(Track(segment["target"], RRWAVENET_FS, lag), Track(stored, FS))
                        sign = -1.0 if r < 0 else 1.0
                    else:
                        sign = 1.0
                    smoothed = lowpass_runs(sign * segment["output"], RRWAVENET_FS, args.rrwavenet_lowpass_hz,
                                            RRWAVENET_LOWPASS_ORDER)
                    values["RRWaveNet"] = Track(smoothed, RRWAVENET_FS, lag).at(times)
                    info.update({"RRWaveNet_Run": rr_files[subject][1], "RRWaveNet_Sign": sign,
                                 "RRWaveNet_Lag_Sec": lag})
                legend = {"GT": f"GT: {GT_NAMES[dataset_key]} ({'data file' if gt_source == 'data file' else 'stored'})"}
                for model in MODELS:
                    if model not in values:
                        continue
                    r = pearson(values["GT"], values[model])
                    covered = float(np.mean(np.isfinite(values[model])))
                    info[f"{model}_r"] = r
                    info[f"{model}_Coverage"] = covered
                    smoothed_note = (f", low-pass {args.rrwavenet_lowpass_hz:g} Hz"
                                     if model == "RRWaveNet" and args.rrwavenet_lowpass_hz > 0 else "")
                    legend[model] = f"{model} (r = {r:.2f}, {100 * covered:.0f}% of the time{smoothed_note})"
                missing = [m for m in MODELS if m not in values]
                info["No_Result"] = ", ".join(missing)
                title = f"{label} {subject}" + ("" if name == subject else f" - {name}")
                subtitle = (f"{len(gt_values) / FS / 60:.1f} min; each waveform "
                            f"{'min-max scaled to [-1, 1]' if args.scale == 'minmax' else 'z-scored'} "
                            f"{'per row' if args.scale_span == 'row' else 'over the recording'}; "
                            + (f"RRWaveNet: {rr_files[subject][1]}; " if "RRWaveNet" in values else "")
                            + (f"no result: {', '.join(missing)}" if missing else "all three models"))
                fig = overlay_figure(values, legend, title, subtitle, args.panel_sec, args.scale, args.scale_span)
                if not args.no_png:
                    stem = safe_name(subject) if name == subject else f"{safe_name(subject)}__{safe_name(name)}"
                    fig.savefig(figure_root / f"{stem}.png", dpi=args.dpi)
                if pdf is not None:
                    pdf.savefig(fig)
                plt.close(fig)
                rows.append(info)
            print(f"  {subject}: {len(corr[subject])} recording(s)", flush=True)
    finally:
        if pdf is not None:
            pdf.close()
    return rows


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--datasets", nargs="+", choices=list(DATASET_ORDER), default=list(DATASET_ORDER))
    parser.add_argument("--subjects", nargs="+", default=None, help="Only these subjects.")
    parser.add_argument("--reference", choices=["raw", "stored"], default="raw",
                        help="GT: the reference of the data files (raw) or the one stored with the corr-encoder "
                             "results (stored).")
    parser.add_argument("--scale", choices=["minmax", "zscore"], default="minmax",
                        help="Every waveform min-max scaled to [-1, 1] (default) or z-scored, on its own.")
    parser.add_argument("--scale-span", choices=["row", "recording"], default="row",
                        help="Scale every waveform per row (default) or over the whole recording.")
    parser.add_argument("--panel-sec", type=float, default=PANEL_SEC, help="Seconds per row (default 60).")
    parser.add_argument("--rrwavenet-lowpass-hz", type=float, default=RRWAVENET_LOWPASS_HZ,
                        help="Low-pass of RRWaveNet's output (Butterworth order 8, zero phase; default 1.0 Hz as "
                             "RespDiff's output filter; 0 = as stored).")
    parser.add_argument("--respdiff-sampling", default=RESPDIFF_SAMPLING)
    parser.add_argument("--respdiff-root", type=Path, default=RESPDIFF_ROOT)
    parser.add_argument("--correncoder-root", type=Path, default=CORRENCODER_ROOT)
    parser.add_argument("--rrwavenet-261007-root", type=Path, default=RRWAVENET_261007_ROOT)
    parser.add_argument("--rrwavenet-261013-root", type=Path, default=RRWAVENET_261013_ROOT)
    parser.add_argument("--output-root", type=Path, default=PROJECT_ROOT)
    parser.add_argument("--capnobase-path", type=Path, default=None)
    parser.add_argument("--bidmc-path", type=Path, default=None)
    parser.add_argument("--steam2-ppg-dir", type=Path, default=None)
    parser.add_argument("--steam2-rsp-dir", type=Path, default=None)
    parser.add_argument("--dpi", type=int, default=130)
    parser.add_argument("--no-png", action="store_true", help="Only the PDF of every dataset.")
    parser.add_argument("--no-pdf", action="store_true", help="Only the PNG of every recording.")
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
    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    rows = []
    for dataset_key in [key for key in DATASET_ORDER if key in args.datasets]:
        rows += dataset_overlays(args, dataset_key, output_root)
    index = pd.DataFrame(rows)
    index.to_csv(output_root / "overlay_index.csv", index=False, encoding="utf-8-sig")
    if index.empty:
        print("Nothing drawn: no subject with a deep corr-encoder result was found (check the result folders).")
        return
    print(f"\n{len(index)} figure(s) of {index['Subject'].nunique()} subject(s) saved under: {output_root}")
    for dataset, part in index.groupby("Dataset", sort=False):
        text = []
        for model in MODELS:
            column = f"{model}_r"
            if column in part and part[column].notna().any():
                text.append(f"{model} r median {part[column].median():.2f} ({part[column].notna().sum()} recordings)")
        print(f"  {dataset}: " + "; ".join(text))


if __name__ == "__main__":
    main()
