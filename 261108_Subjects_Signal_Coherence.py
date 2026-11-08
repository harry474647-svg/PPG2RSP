# -*- coding: utf-8 -*-
r"""261108_Subjects_Signal_Coherence: how strongly the PPG waveform is related
to the reference respiration (RSP) waveform, per subject, with ONE criterion for
STEAM2, BIDMC and CapnoBase. Nothing is trained; the data files are read.

Criterion (the same for every dataset)
--------------------------------------
Measure: magnitude-squared coherence (MSC) between the PPG and the reference
respiration at the breathing frequency. MSC (0-1) is the share of the PPG
variation at that frequency that the reference explains linearly (a squared
correlation per frequency). A constant time delay (capnogram 1-3 s behind the
PPG) changes only the phase and a reversed polarity (impedance, belt) only the
sign, so MSC is unaffected by both and no dataset needs its own alignment rule.

  Signals    PPG and reference as recorded (CapnoBase capnogram, BIDMC
             impedance, STEAM2 belt; STEAM2 sessions cut at their EEGLAB
             boundary events), missing samples interpolated, both resampled
             to 30 Hz (polyphase, anti-aliasing).
  Windows    120 s, not overlapping (--window-sec).
  Estimate   Welch: 20 s Hann segments, not overlapping (--segment-sec), so
             K = 6 segments per window; every segment linearly detrended;
             frequency resolution 0.05 Hz.
  Frequency  the breathing frequency of the window = the largest peak of the
             reference's spectrum within 0.05-1.0 Hz (3-60 breaths/min, all
             three datasets); the MSC of that frequency bin is the window's value.
  Left out   windows with missing samples, a flat PPG or reference (standard
             deviation below 1% of the recording's median window), or >= 10 %
             clipped (saturated) reference samples (--max-clipped-fraction).

Threshold ("low relationship")
------------------------------
With K independent segments, the MSC of two unrelated signals stays below
1 - alpha^(1/(K-1)) with probability 1 - alpha (Carter 1987; Bendat & Piersol).
K = 6, alpha = 0.05: 1 - 0.05^(1/5) = 0.451.
  window   MSC < 0.451: no significant coupling at the breathing frequency
  subject  median window MSC < 0.451 (fewer than half of its windows
           significant): "Low"; otherwise "Related"; fewer than 2 valid
           windows: "Undetermined"
The threshold follows K: 4 -> 0.63, 6 -> 0.45, 8 -> 0.35, 12 -> 0.24.

Supplementary (reported, not used for the decision): the lag-tolerant
correlation |r|: both signals band-passed around the window's breathing
frequency (0.5-2 x, 3rd-order Butterworth, zero phase), the largest |r| within
+-half a breath period; Cohen's classes |r| < 0.3 weak, 0.3-0.5 moderate,
>= 0.5 strong.

Subjects with an MAE issue (STEAM2 22_LMS, 26_CKK, 12_YJJ; BIDMC 27;
CapnoBase 0115) are computed and marked (Issue_Subject) but left out of the
Low_Relationship list and of the dataset counts.

Outputs (default D:\PPG2RSP_RRWaveNet_Inspired\261108_Subjects_Signal_Coherence)
  261108_Subjects_Signal_Coherence.xlsx
    Criterion         every rule and parameter, the threshold
    Subjects          every subject of every dataset: MSC median / mean / IQR /
                      min, significant-window share, breathing frequency,
                      left-out windows by reason, decision, supplementary |r|
    Low_Relationship  the subjects judged "Low" (issue subjects left out),
                      lowest MSC first
    Dataset_Summary   per dataset: subjects, Low / Related / Undetermined,
                      median MSC
    Recordings        every recording (STEAM2: every session piece)
    Windows           every 120 s window (MSC, frequency, |r|, lag, reason left out)
  and Subjects.csv, Windows.csv, Recordings.csv in the same folder.

Run:
  python 261108_Subjects_Signal_Coherence.py
Options: --datasets, --subjects, --output-root, --window-sec, --segment-sec,
  --alpha, --band-hz, --max-clipped-fraction, --capnobase-path, --bidmc-path,
  --steam2-ppg-dir, --steam2-rsp-dir, --exclude-steam2, --exclude-bidmc,
  --exclude-capnobase
"""

import argparse
import math
import re
from fractions import Fraction
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import scipy.io as sio
from scipy.signal import butter, coherence, correlate, resample_poly, sosfiltfilt, welch

# ---------------------------------------------------------------------------
# Folders and settings
# ---------------------------------------------------------------------------

WORKSPACE = Path(r"D:\PPG2RSP_RRWaveNet_Inspired")
NAME = "261108_Subjects_Signal_Coherence"
PROJECT_ROOT = WORKSPACE / NAME
FS30 = 30  # common rate of PPG and reference (Hz)
FIGURE_ORDER = ("STEAM2", "BIDMC", "CapnoBase")
DATASET_KEYS = {"CapnoBase": "capnobase", "BIDMC": "bidmc", "STEAM2": "steam2"}
EXCLUDED = {"STEAM2": ("22_LMS", "26_CKK", "12_YJJ"), "BIDMC": ("27",), "CapnoBase": ("0115",)}
ID_DIGITS = {"CapnoBase": 4, "BIDMC": 2}

WINDOW_SEC = 120.0
SEGMENT_SEC = 20.0
ALPHA = 0.05
BAND_HZ = (0.05, 1.0)  # breathing frequencies looked for (3-60 breaths/min)
MAX_CLIPPED_FRACTION = 0.10
FLAT_FRACTION = 0.01  # a window is flat when its SD is below 1% of the recording's median window SD
SUPP_BAND_FACTORS = (0.5, 2.0)  # supplementary |r|: band 0.5-2 x the breathing frequency
SUPP_ORDER = 3
COHEN = ((0.5, "strong"), (0.3, "moderate"), (0.0, "weak"))

# Data files (as 261104 / 261108_Wavefom_Comparison)
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
SATURATION_CONFIG = {"saturation_min_sec": 0.1, "saturation_margin_sec": 0.0}


# ---------------------------------------------------------------------------
# Data readers, copied unchanged from 261108_Wavefom_Comparison (261104)
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


def finite_runs(valid: np.ndarray, min_length: int) -> List[Tuple[int, int]]:
    """Continuous stretches [start, end) where valid is True."""
    edges = np.diff(np.concatenate([[0], valid.astype(int), [0]]))
    starts, ends = np.flatnonzero(edges == 1), np.flatnonzero(edges == -1)
    return [(int(s), int(e)) for s, e in zip(starts, ends) if e - s >= min_length]



# ---------------------------------------------------------------------------
# Reading every recording at 30 Hz (PPG, reference, missing and clipped samples)
# ---------------------------------------------------------------------------

def same_subject(a: object, b: object) -> bool:
    """'0115' == '115'; other names must match exactly."""
    a, b = str(a).strip(), str(b).strip()
    return a == b or (a.isdigit() and b.isdigit() and int(a) == int(b))


def norm_id(dataset: str, subject) -> str:
    text = re.sub(r"\.0+$", "", str(subject).strip())
    if dataset in ID_DIGITS and text.isdigit():
        text = text.zfill(ID_DIGITS[dataset])
    return text


def subject_order(subject: str):
    match = re.match(r"\d+", str(subject))
    return (int(match.group()) if match else 10 ** 9, str(subject))


def mask_to_30(mask: np.ndarray, fs: float, length: int) -> np.ndarray:
    """A mask at fs on the 30 Hz grid (the sample at or before every 30 Hz sample)."""
    mask = np.asarray(mask, bool).reshape(-1)
    if not mask.any():
        return np.zeros(length, bool)
    index = np.minimum(np.floor(np.arange(length) * float(fs) / FS30).astype(np.int64), mask.size - 1)
    return mask[index]


def make_recording(ppg, rsp, ppg_fs: float, rsp_fs: float, modality: str) -> dict:
    ppg = np.asarray(ppg, dtype=np.float64).reshape(-1)
    rsp = np.asarray(rsp, dtype=np.float64).reshape(-1)
    ppg30, rsp30 = to_fs(ppg, ppg_fs), to_fs(rsp, rsp_fs)
    n = min(len(ppg30), len(rsp30))
    missing = mask_to_30(~np.isfinite(ppg), ppg_fs, n) | mask_to_30(~np.isfinite(rsp), rsp_fs, n)
    clipped = mask_to_30(saturated_samples(rsp, rsp_fs, modality, SATURATION_CONFIG), rsp_fs, n)
    return {"ppg": ppg30[:n], "rsp": rsp30[:n], "missing": missing, "clipped": clipped,
            "ppg_fs": float(ppg_fs), "rsp_fs": float(rsp_fs)}


def _norm_name(name) -> str:
    return re.sub(r"[^a-z0-9]+", "_", str(name).strip().lower()).strip("_")


def capnobase_file(csv_dir: Path, case_id: str, kind: str) -> Optional[Path]:
    for suffix in CAPNOBASE_FILE_SUFFIXES:
        path = csv_dir / f"{case_id}_8min_{kind}{suffix}"
        if path.exists():
            return path
    return None


def capnobase_rates(path: Optional[Path], default_fs: float) -> Tuple[float, float]:
    """Sampling rates of pleth and co2 from the param file (columns, or name / value rows)."""
    rates: Dict[str, float] = {}
    if path is not None:
        try:
            frame = pd.read_csv(path, sep=None, engine="python")
        except Exception:
            frame = pd.DataFrame()
        columns = {_norm_name(c): c for c in frame.columns}
        for signal in ("pleth", "co2"):
            name = next((c for n, c in columns.items() if "samplingrate" in n and signal in n), None)
            if name is not None:
                values = pd.to_numeric(frame[name], errors="coerce").dropna()
                values = values[values > 0]
                if len(values):
                    rates[signal] = float(values.iloc[0])
        if len(rates) < 2 and frame.shape[1] >= 2:
            for _, row in frame.iterrows():
                key = _norm_name(row.iloc[0])
                for signal in ("pleth", "co2"):
                    if signal not in rates and "rate" in key and signal in key:
                        values = pd.to_numeric(pd.Series(list(row.iloc[1:])), errors="coerce").dropna()
                        if len(values) and values.iloc[0] > 0:
                            rates[signal] = float(values.iloc[0])
    ppg_fs = rates.get("pleth", float(default_fs))
    return ppg_fs, rates.get("co2", ppg_fs)


def capnobase_signals(path: Path) -> Tuple[np.ndarray, np.ndarray]:
    frame = pd.read_csv(path, sep=None, engine="python")
    frame.columns = [_norm_name(c) for c in frame.columns]

    def column(signal: str) -> str:
        names = [n for n in frame.columns if signal in n]
        names = [n for n in names if n.endswith("_y")] or names
        if not names:
            raise KeyError(f"{path.name}: no {signal} column in {list(frame.columns)}")
        return min(names, key=len)

    return (pd.to_numeric(frame[column("pleth")], errors="coerce").to_numpy(np.float64),
            pd.to_numeric(frame[column("co2")], errors="coerce").to_numpy(np.float64))


def read_dataset(dataset: str, wanted) -> Dict[str, Dict[str, dict]]:
    """Subject -> {recording name: recording}, named and cut as 261108_Wavefom_Comparison."""
    key = DATASET_KEYS[dataset]
    cfg = DATASET_CONFIGS[key]
    modality = MODALITIES[key]
    out: Dict[str, Dict[str, dict]] = {}
    if key == "capnobase":
        csv_dir = Path(cfg["data_path"])
        for case_id in capnobase_case_ids(csv_dir):
            if not wanted(case_id):
                continue
            try:
                ppg, co2 = capnobase_signals(capnobase_file(csv_dir, case_id, "signal"))
                ppg_fs, co2_fs = capnobase_rates(capnobase_file(csv_dir, case_id, "param"), cfg["default_fs"])
                out[case_id] = {case_id: make_recording(ppg, co2, ppg_fs, co2_fs, modality)}
            except Exception as exc:
                print(f"  [CapnoBase {case_id}] not read: {exc}")
    elif key == "bidmc":
        mat = sio.loadmat(cfg["data_path"])
        for index, record in enumerate(mat["data"][0]):
            subject = f"{index + 1:02d}"
            if not wanted(subject):
                continue
            try:
                ppg_struct = record["ppg"][0, 0]
                ppg = np.asarray(ppg_struct["v"]).squeeze()
                imp_struct = record["ref"][0, 0]["resp_sig"][0, 0]["imp"][0, 0]
                imp = np.asarray(imp_struct["v"]).squeeze()
                fs_ppg = float(np.asarray(ppg_struct["fs"]).squeeze()) if "fs" in ppg_struct.dtype.names else cfg["default_fs"]
                fs_imp = float(np.asarray(imp_struct["fs"]).squeeze()) if "fs" in imp_struct.dtype.names else cfg["default_fs"]
                out[subject] = {subject: make_recording(ppg, imp, fs_ppg, fs_imp, modality)}
            except Exception as exc:
                print(f"  [BIDMC {subject}] not read: {exc}")
    else:
        for subject, items in steam2_session_files(Path(cfg["ppg_dir"]), Path(cfg["rsp_dir"])).items():
            if not wanted(subject):
                continue
            try:
                for name, ppg_path, rsp_path in items:
                    ppg = read_eeglab_channel(ppg_path, "PPG", cfg["default_fs"])
                    rsp = read_eeglab_channel(rsp_path, "RSP", cfg["default_fs"])
                    pieces, _ = steam2_session_pieces(ppg, rsp)
                    for k, (p, r, p_fs, r_fs) in enumerate(pieces):
                        out.setdefault(subject, {})[f"{name}_piece{k + 1}"] = make_recording(p, r, p_fs, r_fs, modality)
            except Exception as exc:
                print(f"  [STEAM2 {subject}] not read: {exc}")
    return out


# ---------------------------------------------------------------------------
# The criterion: coherence at the breathing frequency, per 120 s window
# ---------------------------------------------------------------------------

def msc_threshold(k_segments: int, alpha: float) -> float:
    """MSC that two unrelated signals exceed with probability alpha (K independent segments)."""
    return 1.0 - alpha ** (1.0 / (k_segments - 1))


def cohen_class(r: float) -> str:
    if not np.isfinite(r):
        return ""
    return next(label for limit, label in COHEN if abs(r) >= limit)


def lag_tolerant_r(x: np.ndarray, y: np.ndarray, f_resp: float) -> Tuple[float, float]:
    """Supplementary: the largest |r| within +-half a breath period, both signals band-passed 0.5-2 x f_resp;
    (|r|, lag in s: positive = the reference later than the PPG)."""
    nyq = FS30 / 2.0
    low = max(SUPP_BAND_FACTORS[0] * f_resp, 0.01)
    high = min(SUPP_BAND_FACTORS[1] * f_resp, 0.9 * nyq)
    if not (0 < low < high):
        return math.nan, math.nan
    sos = butter(SUPP_ORDER, [low / nyq, high / nyq], btype="band", output="sos")
    xf = sosfiltfilt(sos, x - np.mean(x))
    yf = sosfiltfilt(sos, y - np.mean(y))
    xf, yf = xf - np.mean(xf), yf - np.mean(yf)
    n = len(xf)
    scale = float(np.std(xf) * np.std(yf))
    if n < 2 or scale < 1e-12:
        return math.nan, math.nan
    full = correlate(yf, xf, mode="full", method="fft")
    lags = np.arange(-(n - 1), n)
    keep = np.abs(lags) <= int(round(0.5 / f_resp * FS30))
    values = full[keep] / ((n - np.abs(lags[keep])) * scale)
    best = int(np.argmax(np.abs(values)))
    return float(abs(values[best])), float(lags[keep][best] / FS30)


def window_rows(dataset: str, subject: str, name: str, rec: dict, args: argparse.Namespace,
                threshold: float) -> List[dict]:
    n_win, n_seg = int(round(args.window_sec * FS30)), int(round(args.segment_sec * FS30))
    ppg, rsp = rec["ppg"], rec["rsp"]
    starts = list(range(0, len(ppg) - n_win + 1, n_win))
    sd_p = np.array([np.std(ppg[a:a + n_win]) for a in starts])
    sd_r = np.array([np.std(rsp[a:a + n_win]) for a in starts])
    med_p = float(np.median(sd_p[sd_p > 0])) if (sd_p > 0).any() else 0.0
    med_r = float(np.median(sd_r[sd_r > 0])) if (sd_r > 0).any() else 0.0
    rows = []
    for i, a in enumerate(starts):
        b = a + n_win
        x, y = ppg[a:b], rsp[a:b]
        clipped = float(np.mean(rec["clipped"][a:b]))
        row = {"Dataset": dataset, "Subject": subject, "Recording": name, "Window": i + 1,
               "Start_Sec": a / FS30, "End_Sec": b / FS30, "Clipped_Reference_Fraction": clipped}
        if rec["missing"][a:b].any():
            row["Left_Out"] = "missing samples"
        elif sd_p[i] <= FLAT_FRACTION * med_p or sd_r[i] <= FLAT_FRACTION * med_r:
            row["Left_Out"] = "flat signal"
        elif clipped >= args.max_clipped_fraction:
            row["Left_Out"] = "clipped reference"
        if row.get("Left_Out"):
            rows.append(row)
            continue
        f, pyy = welch(y, fs=FS30, window="hann", nperseg=n_seg, noverlap=0, nfft=n_seg, detrend="linear")
        _, cxy = coherence(x, y, fs=FS30, window="hann", nperseg=n_seg, noverlap=0, nfft=n_seg, detrend="linear")
        band = np.flatnonzero((f >= args.band_hz[0] - 1e-9) & (f <= args.band_hz[1] + 1e-9))
        k = int(band[np.argmax(pyy[band])])
        f_resp = float(f[k])
        abs_r, lag = lag_tolerant_r(x, y, f_resp)
        row.update({"Left_Out": "", "Breathing_Freq_Hz": f_resp, "Breathing_Rate_bpm": 60.0 * f_resp,
                    "Reference_Peak_Prominence": float(pyy[k] / max(np.median(pyy[band]), 1e-30)),
                    "MSC": float(cxy[k]), "Significant": bool(cxy[k] >= threshold),
                    "Supp_Abs_r": abs_r, "Supp_Lag_Sec": lag, "Supp_Cohen_Class": cohen_class(abs_r)})
        rows.append(row)
    return rows


def summarize(rows: List[dict], threshold: float) -> dict:
    """MSC statistics and the decision of a subject (or a recording)."""
    frame = pd.DataFrame(rows)
    valid = frame[frame["Left_Out"] == ""] if not frame.empty and "Left_Out" in frame else pd.DataFrame()
    msc = valid["MSC"].to_numpy(float) if not valid.empty else np.empty(0)
    reasons = frame["Left_Out"].value_counts() if not frame.empty and "Left_Out" in frame else pd.Series(dtype=int)
    out = {"Windows_Total": int(len(frame)), "Windows_Valid": int(msc.size),
           "Left_Out_Missing": int(reasons.get("missing samples", 0)),
           "Left_Out_Flat": int(reasons.get("flat signal", 0)),
           "Left_Out_Clipped": int(reasons.get("clipped reference", 0))}
    if msc.size:
        out.update({"MSC_Median": float(np.median(msc)), "MSC_Mean": float(np.mean(msc)),
                    "MSC_Q1": float(np.percentile(msc, 25)), "MSC_Q3": float(np.percentile(msc, 75)),
                    "MSC_Min": float(np.min(msc)), "MSC_Max": float(np.max(msc)),
                    "Significant_Windows": int(np.sum(msc >= threshold)),
                    "Significant_Share": float(np.mean(msc >= threshold)),
                    "Breathing_Rate_Median_bpm": float(np.median(valid["Breathing_Rate_bpm"])),
                    "Reference_Peak_Prominence_Median": float(np.median(valid["Reference_Peak_Prominence"])),
                    "Supp_Abs_r_Median": float(np.nanmedian(valid["Supp_Abs_r"])),
                    "Supp_Lag_Median_Sec": float(np.nanmedian(valid["Supp_Lag_Sec"]))})
        out["Supp_Cohen_Class"] = cohen_class(out["Supp_Abs_r_Median"])
    out["Decision"] = ("Undetermined" if msc.size < 2 else ("Low" if out["MSC_Median"] < threshold else "Related"))
    return out


# ---------------------------------------------------------------------------
# Tables
# ---------------------------------------------------------------------------

def criterion_table(args: argparse.Namespace, k_segments: int, threshold: float) -> pd.DataFrame:
    return pd.DataFrame([
        ("Measure", "MSC", "magnitude-squared coherence between the PPG and the reference respiration at the breathing "
                           "frequency: the share of the PPG variation at that frequency explained linearly by the "
                           "reference; unaffected by a constant time delay and by the polarity, so one rule fits all "
                           "datasets"),
        ("Signals", "as recorded", "PPG and reference (CapnoBase capnogram, BIDMC impedance, STEAM2 belt; STEAM2 sessions "
                                   "cut at their EEGLAB boundaries); missing samples interpolated; both resampled to "
                                   f"{FS30} Hz"),
        ("Window", f"{args.window_sec:g} s", "not overlapping; only complete windows"),
        ("Welch segments", f"{args.segment_sec:g} s Hann, K = {k_segments}",
         "not overlapping, linearly detrended; frequency resolution "
         f"{1.0 / args.segment_sec:.3f} Hz"),
        ("Breathing frequency", f"{args.band_hz[0]:g}-{args.band_hz[1]:g} Hz",
         "the largest peak of the reference's spectrum in this range (3-60 breaths/min); the MSC of that bin"),
        ("Windows left out", f"clipped >= {args.max_clipped_fraction:.0%}",
         f"missing samples; flat PPG or reference (SD below {FLAT_FRACTION:.0%} of the recording's median window SD); "
         "clipped (saturated) reference samples at or above this share"),
        ("Threshold", f"{threshold:.3f}", f"1 - alpha^(1/(K-1)), alpha = {args.alpha:g}, K = {k_segments}: the MSC two "
                                          "unrelated signals stay below with probability 1 - alpha (Carter 1987; "
                                          "Bendat & Piersol)"),
        ("Window decision", f"MSC < {threshold:.3f}", "no significant coupling at the breathing frequency"),
        ("Subject decision", "Low / Related / Undetermined",
         f"Low: median window MSC < {threshold:.3f} (fewer than half of the windows significant); Related: otherwise; "
         "Undetermined: fewer than 2 valid windows"),
        ("Threshold vs K", "4: 0.63, 6: 0.45, 8: 0.35, 12: 0.24", "alpha = 0.05; the threshold follows the design"),
        ("Supplementary", "lag-tolerant |r| (not used for the decision)",
         f"both signals band-passed {SUPP_BAND_FACTORS[0]:g}-{SUPP_BAND_FACTORS[1]:g} x the breathing frequency "
         f"({SUPP_ORDER}rd-order Butterworth, zero phase); the largest |r| within +-half a breath period; Cohen: "
         "< 0.3 weak, 0.3-0.5 moderate, >= 0.5 strong"),
        ("Issue subjects", "; ".join(f"{d} {', '.join(s)}" for d, s in args.excluded.items() if s),
         "computed and marked (Issue_Subject); left out of Low_Relationship and Dataset_Summary"),
    ], columns=["Item", "Value", "Description"])


def write_tables(out_dir: Path, sheets: Dict[str, pd.DataFrame]) -> Path:
    """The workbook (and Subjects / Recordings / Windows as CSV); a file held by another program (e.g. open in
    Excel) is written under a time-stamped name instead."""
    stamp = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")

    def write(path: Path, writer) -> Path:
        try:
            writer(path)
            return path
        except PermissionError:
            alt = path.with_name(f"{path.stem}_{stamp}{path.suffix}")
            writer(alt)
            print(f"  [Locked] {path.name} is held by another program; saved as {alt.name}")
            return alt

    def xlsx(path: Path) -> None:
        with pd.ExcelWriter(path, engine="openpyxl") as w:
            for name, frame in sheets.items():
                frame.to_excel(w, sheet_name=name[:31], index=False)

    for name in ("Subjects", "Recordings", "Windows"):
        write(out_dir / f"{name}.csv", lambda p, f=sheets[name]: f.to_csv(p, index=False, encoding="utf-8-sig"))
    return write(out_dir / f"{NAME}.xlsx", xlsx)


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="PPG-reference coherence at the breathing frequency, per subject, "
                                                 "one criterion for STEAM2, BIDMC and CapnoBase")
    parser.add_argument("--datasets", nargs="+", choices=list(FIGURE_ORDER), default=list(FIGURE_ORDER))
    parser.add_argument("--subjects", nargs="+", default=None, help="only these subjects")
    parser.add_argument("--output-root", type=Path, default=PROJECT_ROOT)
    parser.add_argument("--window-sec", type=float, default=WINDOW_SEC)
    parser.add_argument("--segment-sec", type=float, default=SEGMENT_SEC)
    parser.add_argument("--alpha", type=float, default=ALPHA)
    parser.add_argument("--band-hz", nargs=2, type=float, default=list(BAND_HZ))
    parser.add_argument("--max-clipped-fraction", type=float, default=MAX_CLIPPED_FRACTION)
    parser.add_argument("--capnobase-path", type=Path, default=None)
    parser.add_argument("--bidmc-path", type=Path, default=None)
    parser.add_argument("--steam2-ppg-dir", type=Path, default=None)
    parser.add_argument("--steam2-rsp-dir", type=Path, default=None)
    parser.add_argument("--exclude-steam2", nargs="*", default=list(EXCLUDED["STEAM2"]))
    parser.add_argument("--exclude-bidmc", nargs="*", default=list(EXCLUDED["BIDMC"]))
    parser.add_argument("--exclude-capnobase", nargs="*", default=list(EXCLUDED["CapnoBase"]))
    args = parser.parse_args(argv)
    args.excluded = {"STEAM2": [norm_id("STEAM2", s) for s in args.exclude_steam2],
                     "BIDMC": [norm_id("BIDMC", s) for s in args.exclude_bidmc],
                     "CapnoBase": [norm_id("CapnoBase", s) for s in args.exclude_capnobase]}
    for key, path in (("capnobase", args.capnobase_path), ("bidmc", args.bidmc_path)):
        if path:
            DATASET_CONFIGS[key]["data_path"] = str(path)
    if args.steam2_ppg_dir:
        DATASET_CONFIGS["steam2"]["ppg_dir"] = str(args.steam2_ppg_dir)
    if args.steam2_rsp_dir:
        DATASET_CONFIGS["steam2"]["rsp_dir"] = str(args.steam2_rsp_dir)
    k_segments = int(round(args.window_sec / args.segment_sec))
    if k_segments < 2 or abs(k_segments * args.segment_sec - args.window_sec) > 1e-6:
        parser.error("--window-sec must be a multiple (>= 2) of --segment-sec")
    return args


def main(argv: Optional[Sequence[str]] = None) -> None:
    args = parse_args(argv)
    k_segments = int(round(args.window_sec / args.segment_sec))
    threshold = msc_threshold(k_segments, args.alpha)
    out_dir = Path(args.output_root)
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"Output: {out_dir}\nCriterion: MSC at the breathing frequency, {args.window_sec:g} s windows, "
          f"{args.segment_sec:g} s segments (K = {k_segments}); Low when the median window MSC < {threshold:.3f}")
    windows, recordings, subjects = [], [], []
    for dataset in args.datasets:
        wanted = (lambda s: True) if not args.subjects else (lambda s: any(same_subject(s, x) for x in args.subjects))
        print(f"\n{dataset}: reading", flush=True)
        data = read_dataset(dataset, wanted)
        for subject in sorted(data, key=subject_order):
            issue = any(same_subject(subject, x) for x in args.excluded.get(dataset, []))
            subject_windows = []
            for name, rec in data[subject].items():
                rows = window_rows(dataset, subject, name, rec, args, threshold)
                subject_windows += rows
                recordings.append({"Dataset": dataset, "Subject": subject, "Recording": name, "Issue_Subject": issue,
                                   "Seconds": len(rec["ppg"]) / FS30, "PPG_FS": rec["ppg_fs"],
                                   "Reference_FS": rec["rsp_fs"], **summarize(rows, threshold)})
            windows += subject_windows
            subjects.append({"Dataset": dataset, "Subject": subject, "Issue_Subject": issue,
                             "Reference": MODALITIES[DATASET_KEYS[dataset]], "Recordings": len(data[subject]),
                             "Seconds": sum(len(r["ppg"]) for r in data[subject].values()) / FS30,
                             **summarize(subject_windows, threshold)})
            s = subjects[-1]
            print(f"  {subject}{' (issue subject)' if issue else ''}: {s['Windows_Valid']}/{s['Windows_Total']} "
                  f"windows, MSC median {s.get('MSC_Median', math.nan):.3f} -> {s['Decision']}", flush=True)
    subject_frame = pd.DataFrame(subjects)
    if subject_frame.empty:
        print("No subject read.")
        return
    first = ["Dataset", "Subject", "Issue_Subject", "Decision", "Rank_In_Dataset", "MSC_Median", "Significant_Share",
             "Windows_Valid", "Windows_Total", "MSC_Mean", "MSC_Q1", "MSC_Q3", "MSC_Min", "MSC_Max",
             "Significant_Windows", "Breathing_Rate_Median_bpm", "Reference_Peak_Prominence_Median",
             "Left_Out_Missing", "Left_Out_Flat", "Left_Out_Clipped", "Supp_Abs_r_Median", "Supp_Cohen_Class",
             "Supp_Lag_Median_Sec", "Reference", "Recordings", "Seconds"]
    ranked = subject_frame[~subject_frame["Issue_Subject"] & subject_frame["MSC_Median"].notna()]
    subject_frame["Rank_In_Dataset"] = ranked.groupby("Dataset")["MSC_Median"].rank(method="min").reindex(
        subject_frame.index)
    subject_frame = subject_frame[[c for c in first if c in subject_frame] + [c for c in subject_frame if c not in first]]
    usable = subject_frame[~subject_frame["Issue_Subject"]]
    low = usable[usable["Decision"] == "Low"].sort_values(["MSC_Median"]).reset_index(drop=True)
    summary = []
    for dataset in [d for d in FIGURE_ORDER if d in set(subject_frame["Dataset"])]:
        part = usable[usable["Dataset"] == dataset]
        med = part["MSC_Median"].dropna()
        summary.append({"Dataset": dataset, "Subjects": len(part),
                        "Issue_Subjects_Left_Out": int((subject_frame["Dataset"] == dataset).sum() - len(part)),
                        "Low": int((part["Decision"] == "Low").sum()),
                        "Related": int((part["Decision"] == "Related").sum()),
                        "Undetermined": int((part["Decision"] == "Undetermined").sum()),
                        "Low_Share": float((part["Decision"] == "Low").mean()) if len(part) else math.nan,
                        "MSC_Median_of_Subjects": float(med.median()) if len(med) else math.nan,
                        "MSC_Q1_of_Subjects": float(med.quantile(0.25)) if len(med) else math.nan,
                        "MSC_Q3_of_Subjects": float(med.quantile(0.75)) if len(med) else math.nan,
                        "Low_Subjects": ", ".join(low.loc[low["Dataset"] == dataset, "Subject"])})
    sheets = {"Criterion": criterion_table(args, k_segments, threshold), "Subjects": subject_frame,
              "Low_Relationship": low, "Dataset_Summary": pd.DataFrame(summary),
              "Recordings": pd.DataFrame(recordings), "Windows": pd.DataFrame(windows)}
    path = write_tables(out_dir, sheets)
    pd.set_option("display.width", 200)
    print("\n" + pd.DataFrame(summary)[["Dataset", "Subjects", "Low", "Related", "Undetermined",
                                        "MSC_Median_of_Subjects"]].to_string(index=False))
    if not low.empty:
        print("\nLow relationship (median window MSC < %.3f):" % threshold)
        print(low[["Dataset", "Subject", "MSC_Median", "Significant_Share", "Windows_Valid"]].round(3).to_string(index=False))
    print(f"\nSaved: {path}")


if __name__ == "__main__":
    main()
