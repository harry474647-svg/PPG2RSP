# -*- coding: utf-8 -*-
r"""261109_Subjects_BreathAmplitude_Coherence: per subject, how strongly the PPG
is related to the reference respiration by two criteria - the breath-to-breath
amplitude (BBA) and the coherence (MSC) - with the PPG band-passed 0.1-8.0 Hz
and the respiration band-passed 0.1-1.0 Hz, and the subjects labelled
  Both weak / BBA only weak / Coherence only weak / Strong (/ Undetermined).
One rule for STEAM2, BIDMC and CapnoBase. Nothing is trained; the data files are read.

Signals
  PPG and reference as recorded (CapnoBase capnogram, BIDMC impedance, STEAM2 belt;
  STEAM2 sessions cut at their EEGLAB boundary events); missing samples
  interpolated; both resampled to 64 Hz (polyphase, anti-aliasing); then
    PPG          band-pass 0.1-8.0 Hz  (3rd-order Butterworth, zero phase, whole recording)
    respiration  band-pass 0.1-1.0 Hz  (3rd-order Butterworth, zero phase, whole recording)
  Windows: 120 s, not overlapping; left out (both criteria) with missing samples, a flat
  PPG or reference (SD below 1% of the recording's median window SD) or >= 10 % clipped
  (saturated) reference samples.

Criterion 1 - breath-to-breath amplitude (Johansson & Oberg 1999; Johansson & Stromberg
2000; Nilsson et al. 2003): the amplitude of the respiratory-induced intensity variation
(RIIV) of the PPG in every breath, against the amplitude of the same breath in the
reference respiration. In the papers the RIIV amplitude (peak-to-valley, per breath)
rises with the tidal volume within a subject (R = 0.89 on average per subject).
  Breaths      on the respiration (0.1-1.0 Hz): count-orig style (Schafer & Kratky 2008):
               peaks above 0.2 x the 75th percentile of the peaks, troughs below 0.2 x
               the 25th percentile of the troughs, alternating (of two neighbours of the
               same kind the more extreme one is kept); a breath = trough - peak - trough,
               1-12 s long. Breaths with clipped reference samples are left out.
  RIIV         the respiratory-induced intensity variation of the 0.1-8.0 Hz PPG: the mean
               of every cardiac cycle (onset to onset; onsets = pulse troughs at least
               0.6 x the spectral beat period apart; cycles of 0.27-1.5 s), interpolated
               linearly between the cycle centres (gaps over 2 s left empty). Averaging
               over each cycle removes the pulse itself and keeps its intensity level,
               i.e. the RIIV of the papers, without a narrower filter.
  Lag          per window, the lag of the RIIV against the respiration with the largest
               |r| within +-half a breath period (the PPG and the reference are not in
               phase: venous / arterial transmission, capnogram transport delay); the
               breaths of the PPG are the reference breaths shifted by this lag.
  Amplitude    peak-to-valley (maximum - minimum) within the breath, for the respiration
               and for the RIIV (polarity does not matter).
  Window       Spearman rho of the RIIV amplitudes and the respiration amplitudes of its
               breaths (monotonic: the papers fitted 1st- and 2nd-order regressions;
               robust to single disturbed breaths), at least 8 breaths;
               significant: rho > 0 with one-sided p < 0.05 (larger breath, larger RIIV).
  Subject      Strong: at least half of its valid windows significant; Weak: fewer;
               Undetermined: fewer than 2 valid windows.
Criterion 2 - coherence (as 261108_Subjects_Signal_Coherence, on the filtered signals):
  magnitude-squared coherence (Welch: 20 s Hann segments, not overlapping, linearly
  detrended, K = 6) at the breathing frequency = the largest peak of the respiration
  spectrum within 0.1-1.0 Hz; window significant: MSC >= 1 - alpha^(1/(K-1)) = 0.451
  (alpha 0.05; Carter 1987); subject Strong / Weak / Undetermined by the same
  half-of-the-windows rule. (In theory a filter does not change the MSC inside its pass
  band; the filters change the estimate only through less spectral leakage.)
Labels
  Both weak            coherence Weak and BBA Weak
  BBA only weak        BBA Weak, coherence Strong   (timing related, amplitudes not)
  Coherence only weak  coherence Weak, BBA Strong   (amplitudes related, timing not stable)
  Strong               both Strong
  Undetermined         either criterion Undetermined
Subjects with an MAE issue (STEAM2 22_LMS, 26_CKK, 12_YJJ; BIDMC 27; CapnoBase 0115)
are computed and marked (Issue_Subject) but left out of the label counts.
Note: the CapnoBase reference is a capnogram; its breath amplitude follows the CO2
level, not the tidal volume, so the BBA criterion is weaker evidence there.

Outputs (default D:\PPG2RSP_RRWaveNet_Inspired\261109_Subjects_BreathAmplitude_Coherence)
  261109_Subjects_BreathAmplitude_Coherence.xlsx
    Criterion        every rule and parameter
    Subjects         every subject: label; BBA (decision, significant windows, median rho,
                     pooled rho, breaths, amplitude variation); coherence (decision,
                     significant windows, median MSC); breathing rate; left-out windows
    Labels           per dataset: subjects per label and their names (issue subjects left out)
    Not_Strong       the subjects that are not Strong, by label, weakest first
    Recordings       every recording (STEAM2: every session piece)
    Windows          every 120 s window: MSC, breathing frequency, BBA rho / p / breaths,
                     lag, amplitude variation, reasons left out
    Breaths          every breath used: times, respiration and RIIV amplitude, lag
  and Subjects.csv, Windows.csv, Breaths.csv, Recordings.csv in the same folder.

Run:
  python 261109_Subjects_BreathAmplitude_Coherence.py
Options: --datasets, --subjects, --output-root, --ppg-band, --resp-band, --window-sec,
  --segment-sec, --alpha, --min-breaths, --max-clipped-fraction, --capnobase-path,
  --bidmc-path, --steam2-ppg-dir, --steam2-rsp-dir, --exclude-steam2, --exclude-bidmc,
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
from scipy.signal import butter, coherence, correlate, find_peaks, resample_poly, sosfiltfilt, welch
from scipy.stats import pearsonr, spearmanr

# ---------------------------------------------------------------------------
# Folders and settings
# ---------------------------------------------------------------------------

WORKSPACE = Path(r"D:\PPG2RSP_RRWaveNet_Inspired")
NAME = "261109_Subjects_BreathAmplitude_Coherence"
PROJECT_ROOT = WORKSPACE / NAME
FS = 64  # common rate of PPG and reference (Hz)
FIGURE_ORDER = ("STEAM2", "BIDMC", "CapnoBase")
DATASET_KEYS = {"CapnoBase": "capnobase", "BIDMC": "bidmc", "STEAM2": "steam2"}
EXCLUDED = {"STEAM2": ("22_LMS", "26_CKK", "12_YJJ"), "BIDMC": ("27",), "CapnoBase": ("0115",)}
ID_DIGITS = {"CapnoBase": 4, "BIDMC": 2}

PPG_BAND_HZ = (0.1, 8.0)
RESP_BAND_HZ = (0.1, 1.0)
FILTER_ORDER = 3
WINDOW_SEC = 120.0
SEGMENT_SEC = 20.0
ALPHA = 0.05
MIN_BREATHS = 8
MAX_CLIPPED_FRACTION = 0.10
FLAT_FRACTION = 0.01  # a window is flat when its SD is below 1% of the recording's median window SD
BREATH_SEC = (1.0, 12.0)  # breath durations kept (5-60 breaths/min)
PEAK_FRACTION = 0.2  # count-orig thresholds: 0.2 x the 75th percentile of the peaks / 25th of the troughs
HR_RANGE_BPM = (40.0, 220.0)
BEAT_DISTANCE_FRACTION = 0.6  # pulse onsets at least 0.6 x the spectral beat period apart
BEAT_SEC = (60.0 / 220.0, 1.5)  # cardiac cycles kept
MAX_BEAT_GAP_SEC = 2.0  # the RIIV is not interpolated across larger gaps
MAX_RIIV_MISSING = 0.10  # a window needs the RIIV on at least 90 % of its samples
LABELS = ("Both weak", "BBA only weak", "Coherence only weak", "Strong", "Undetermined")

# Data files (as 261104 / 261108_Wavefom_Comparison / 261108_Subjects_Signal_Coherence)
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
# Data readers, copied unchanged from 261108_Subjects_Signal_Coherence (261104)
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


def _extract_signal_and_fs(payload, default_fs: int) -> Tuple[np.ndarray, float]:
    signal = np.asarray(_unwrap_mat(payload["data"]), dtype=np.float64).reshape(-1)
    try:
        fs = float(np.asarray(_unwrap_mat(payload["srate"])).squeeze())
    except Exception:
        fs = float(default_fs)
    return signal.astype(np.float32), fs


def capnobase_case_ids(csv_dir: Path) -> List[str]:
    return sorted({path.name.split("_")[0] for suffix in CAPNOBASE_FILE_SUFFIXES
                   for path in csv_dir.glob(f"*_8min_signal{suffix}")})


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
    """One channel of an EEGLAB .mat file (chosen by its label), its rate and its internal boundary events."""
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
        # EEGLAB latencies are 1-based and a boundary at k.5 lies between samples k and k + 1: the next piece
        # starts at 0-based index k; the markers at the start and the end are not discontinuities
        position = int(np.ceil(latency - 1.0))
        if 0 < position < signal.size:
            boundaries.append(position)
    label = labels[index] if index < len(labels) else ""
    return {"signal": signal, "fs": fs, "boundaries": sorted(set(boundaries)), "label": label,
            "channels": int(data.shape[0])}


def steam2_session_files(ppg_root: Path, rsp_root: Path) -> Dict[str, List[Tuple[str, Path, Path]]]:
    """Subject -> [(session name, PPG file, RSP file)], sessions in order (files paired by name only)."""
    sessions: Dict[str, List[Tuple[str, Path, Path]]] = {}
    subject_dirs = sorted(path for path in ppg_root.iterdir() if path.is_dir())
    if subject_dirs:
        for subject_dir in subject_dirs:
            rsp_dir = rsp_root / subject_dir.name
            if not rsp_dir.is_dir():
                print(f"[STEAM2 {subject_dir.name}] no RSP folder, skipped")
                continue
            rsp_files = {path.stem: path for path in rsp_dir.glob("*.mat")}
            for ppg_path in sorted(subject_dir.glob("*.mat")):
                if ppg_path.stem not in rsp_files:
                    print(f"[STEAM2 {subject_dir.name}] {ppg_path.name}: no RSP file of the same name, left out")
                    continue
                sessions.setdefault(subject_dir.name, []).append((ppg_path.stem, ppg_path, rsp_files[ppg_path.stem]))
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
    """The (PPG, RSP, fs, fs) pieces of one session, split at the boundary events of either file."""
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
    info = {"PPG_Sec": ppg_sec, "RSP_Sec": rsp_sec, "Internal_Boundaries": len(edges) - 2, "Pieces": len(pieces)}
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
    """Resampled to FS (polyphase with anti-aliasing), missing samples interpolated first."""
    x, _ = fill_missing(signal)
    ratio = Fraction(FS / float(fs)).limit_denominator(1000)
    return resample_poly(x, ratio.numerator, ratio.denominator) if ratio != 1 else x


def saturated_samples(signal: np.ndarray, fs: float, modality: str, config: dict) -> np.ndarray:
    """Clipped samples: runs of at least 0.1 s exactly at the largest or smallest value of the recording (the
    capnogram only at the largest: its inspiratory baseline at the floor is no clipping)."""
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
# Reading every recording at FS, filtered (PPG 0.1-8.0 Hz, respiration 0.1-1.0 Hz)
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


def mask_to_fs(mask: np.ndarray, fs: float, length: int) -> np.ndarray:
    """A mask at fs on the FS grid (the sample at or before every FS sample)."""
    mask = np.asarray(mask, bool).reshape(-1)
    if not mask.any():
        return np.zeros(length, bool)
    index = np.minimum(np.floor(np.arange(length) * float(fs) / FS).astype(np.int64), mask.size - 1)
    return mask[index]


def bandpass(signal: np.ndarray, band: Tuple[float, float]) -> np.ndarray:
    """Zero-phase Butterworth band-pass (second-order sections) at FS."""
    nyquist = FS / 2.0
    low, high = float(band[0]), min(float(band[1]), 0.99 * nyquist)
    sos = butter(FILTER_ORDER, [low / nyquist, high / nyquist], btype="band", output="sos")
    x = np.asarray(signal, dtype=np.float64)
    if x.size <= 3 * (2 * len(sos) + 1):
        return x - np.mean(x) if x.size else x
    return sosfiltfilt(sos, x - np.mean(x))


def make_recording(ppg, rsp, ppg_fs: float, rsp_fs: float, modality: str, args: argparse.Namespace) -> dict:
    ppg = np.asarray(ppg, dtype=np.float64).reshape(-1)
    rsp = np.asarray(rsp, dtype=np.float64).reshape(-1)
    ppg_fs_signal, rsp_fs_signal = to_fs(ppg, ppg_fs), to_fs(rsp, rsp_fs)
    n = min(len(ppg_fs_signal), len(rsp_fs_signal))
    missing = mask_to_fs(~np.isfinite(ppg), ppg_fs, n) | mask_to_fs(~np.isfinite(rsp), rsp_fs, n)
    clipped = mask_to_fs(saturated_samples(rsp, rsp_fs, modality, SATURATION_CONFIG), rsp_fs, n)
    return {"ppg": bandpass(ppg_fs_signal[:n], args.ppg_band), "rsp": bandpass(rsp_fs_signal[:n], args.resp_band),
            "missing": missing, "clipped": clipped, "ppg_fs": float(ppg_fs), "rsp_fs": float(rsp_fs)}


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


def read_dataset(dataset: str, wanted, args: argparse.Namespace) -> Dict[str, Dict[str, dict]]:
    """Subject -> {recording name: recording}, named and cut as 261108_Subjects_Signal_Coherence."""
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
                out[case_id] = {case_id: make_recording(ppg, co2, ppg_fs, co2_fs, modality, args)}
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
                out[subject] = {subject: make_recording(ppg, imp, fs_ppg, fs_imp, modality, args)}
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
                        out.setdefault(subject, {})[f"{name}_piece{k + 1}"] = make_recording(p, r, p_fs, r_fs,
                                                                                             modality, args)
            except Exception as exc:
                print(f"  [STEAM2 {subject}] not read: {exc}")
    return out


# ---------------------------------------------------------------------------
# Breaths of the respiration and the RIIV of the PPG
# ---------------------------------------------------------------------------

def detect_breaths(resp: np.ndarray, fs: float = FS) -> List[Tuple[int, int, int]]:
    """Breaths (trough, peak, next trough) of the band-passed respiration, count-orig style."""
    x = np.asarray(resp, dtype=np.float64) - float(np.median(resp))
    peaks, _ = find_peaks(x)
    troughs, _ = find_peaks(-x)
    if len(peaks) < 3 or len(troughs) < 3:
        return []
    peaks = peaks[x[peaks] > max(PEAK_FRACTION * np.percentile(x[peaks], 75), 0.0)]
    troughs = troughs[x[troughs] < min(PEAK_FRACTION * np.percentile(x[troughs], 25), 0.0)]
    events = sorted([(int(i), 1) for i in peaks] + [(int(i), -1) for i in troughs])
    sequence: List[Tuple[int, int]] = []
    for index, kind in events:
        if sequence and sequence[-1][1] == kind:
            last = sequence[-1][0]
            if (kind == 1 and x[index] > x[last]) or (kind == -1 and x[index] < x[last]):
                sequence[-1] = (index, kind)
        else:
            sequence.append((index, kind))
    breaths = []
    for a, b, c in zip(sequence, sequence[1:], sequence[2:]):
        if a[1] == -1 and b[1] == 1 and c[1] == -1 and BREATH_SEC[0] <= (c[0] - a[0]) / fs <= BREATH_SEC[1]:
            breaths.append((a[0], b[0], c[0]))
    return breaths


def beat_onsets(ppg: np.ndarray, fs: float = FS) -> Tuple[np.ndarray, float]:
    """Pulse onsets (troughs) of the band-passed PPG, at least 0.6 x the spectral beat period apart; the heart
    rate (bpm) of the recording from the Welch spectrum (40-220 bpm)."""
    x = np.asarray(ppg, dtype=np.float64)
    if x.size < 8 * fs:
        return np.empty(0, dtype=np.int64), math.nan
    f, p = welch(x - np.mean(x), fs=fs, nperseg=int(min(x.size, 16 * fs)))
    band = (f >= HR_RANGE_BPM[0] / 60.0) & (f <= HR_RANGE_BPM[1] / 60.0)
    if not band.any():
        return np.empty(0, dtype=np.int64), math.nan
    hr_hz = float(f[band][np.argmax(p[band])])
    distance = max(int(BEAT_DISTANCE_FRACTION * fs / hr_hz), int(BEAT_SEC[0] * fs))
    onsets, props = find_peaks(-x, distance=distance, prominence=0.0)
    if onsets.size:
        prominence = props["prominences"]
        onsets = onsets[prominence >= 0.3 * np.median(prominence)]  # small notches are no onsets
    return onsets.astype(np.int64), 60.0 * hr_hz


def riiv_series(ppg: np.ndarray, missing: np.ndarray, fs: float = FS) -> Tuple[np.ndarray, np.ndarray, float]:
    """The RIIV: the mean of every cardiac cycle of the band-passed PPG, interpolated linearly between the cycle
    centres (NaN outside them and across gaps over MAX_BEAT_GAP_SEC); the cycle centres; the heart rate."""
    n = len(ppg)
    out = np.full(n, np.nan)
    onsets, hr = beat_onsets(ppg, fs)
    if onsets.size < 3:
        return out, np.empty(0), hr
    starts, ends = onsets[:-1], onsets[1:]
    duration = (ends - starts) / fs
    keep = (duration >= BEAT_SEC[0]) & (duration <= BEAT_SEC[1])
    keep &= np.array([not missing[s:e].any() for s, e in zip(starts, ends)], dtype=bool)
    if keep.sum() < 3:
        return out, np.empty(0), hr
    centres = 0.5 * (starts[keep] + ends[keep])
    means = np.array([np.mean(ppg[s:e]) for s, e in zip(starts[keep], ends[keep])])
    grid = np.arange(n, dtype=np.float64)
    inside = (grid >= centres[0]) & (grid <= centres[-1])
    out[inside] = np.interp(grid[inside], centres, means)
    gap = int(round(MAX_BEAT_GAP_SEC * fs))
    for a, b in zip(centres[:-1], centres[1:]):
        if b - a > gap:
            out[int(np.ceil(a)) + 1:int(np.floor(b))] = np.nan
    return out, centres, hr


def lag_of_riiv(riiv: np.ndarray, resp: np.ndarray, period_sec: float, fs: float = FS) -> Tuple[int, float]:
    """The lag (samples; positive = RIIV later) with the largest |r| within +-half a breath period."""
    ok = np.isfinite(riiv)
    if ok.mean() < 1.0 - MAX_RIIV_MISSING or not np.isfinite(period_sec):
        return 0, math.nan
    x, _ = fill_missing(np.where(ok, riiv, np.nan))
    x, y = x - np.mean(x), np.asarray(resp, float) - np.mean(resp)
    scale = float(np.std(x) * np.std(y))
    n = len(x)
    if n < 2 or scale < 1e-12:
        return 0, math.nan
    full = correlate(x, y, mode="full", method="fft")
    lags = np.arange(-(n - 1), n)
    keep = np.abs(lags) <= max(1, int(round(0.5 * period_sec * fs)))
    values = full[keep] / ((n - np.abs(lags[keep])) * scale)
    best = int(np.argmax(np.abs(values)))
    return int(lags[keep][best]), float(values[best])


def one_sided_greater(rho: float, p_two_sided: float) -> float:
    if not np.isfinite(rho) or not np.isfinite(p_two_sided):
        return math.nan
    return p_two_sided / 2.0 if rho > 0 else 1.0 - p_two_sided / 2.0


def cv(values: np.ndarray) -> float:
    values = np.asarray(values, float)
    return float(np.std(values, ddof=1) / np.mean(values)) if values.size > 1 and np.mean(values) > 0 else math.nan


# ---------------------------------------------------------------------------
# Both criteria per 120 s window
# ---------------------------------------------------------------------------

def msc_threshold(k_segments: int, alpha: float) -> float:
    """MSC that two unrelated signals exceed with probability alpha (K independent segments)."""
    return 1.0 - alpha ** (1.0 / (k_segments - 1))


def window_rows(dataset: str, subject: str, name: str, rec: dict, args: argparse.Namespace,
                threshold: float) -> Tuple[List[dict], List[dict]]:
    n_win, n_seg = int(round(args.window_sec * FS)), int(round(args.segment_sec * FS))
    ppg, rsp, n = rec["ppg"], rec["rsp"], len(rec["ppg"])
    breaths = detect_breaths(rsp)
    riiv, centres, hr = riiv_series(ppg, rec["missing"])
    starts = list(range(0, n - n_win + 1, n_win))
    sd_p = np.array([np.std(ppg[a:a + n_win]) for a in starts])
    sd_r = np.array([np.std(rsp[a:a + n_win]) for a in starts])
    med_p = float(np.median(sd_p[sd_p > 0])) if (sd_p > 0).any() else 0.0
    med_r = float(np.median(sd_r[sd_r > 0])) if (sd_r > 0).any() else 0.0
    rows, breath_rows = [], []
    for i, a in enumerate(starts):
        b = a + n_win
        clipped = float(np.mean(rec["clipped"][a:b]))
        row = {"Dataset": dataset, "Subject": subject, "Recording": name, "Window": i + 1,
               "Start_Sec": a / FS, "End_Sec": b / FS, "Clipped_Reference_Fraction": clipped,
               "Heart_Rate_bpm": hr, "Left_Out": "", "BBA_Left_Out": ""}
        if rec["missing"][a:b].any():
            row["Left_Out"] = "missing samples"
        elif sd_p[i] <= FLAT_FRACTION * med_p or sd_r[i] <= FLAT_FRACTION * med_r:
            row["Left_Out"] = "flat signal"
        elif clipped >= args.max_clipped_fraction:
            row["Left_Out"] = "clipped reference"
        if row["Left_Out"]:
            rows.append(row)
            continue
        x, y = ppg[a:b], rsp[a:b]
        # coherence at the breathing frequency
        f, pyy = welch(y, fs=FS, window="hann", nperseg=n_seg, noverlap=0, nfft=n_seg, detrend="linear")
        _, cxy = coherence(x, y, fs=FS, window="hann", nperseg=n_seg, noverlap=0, nfft=n_seg, detrend="linear")
        band = np.flatnonzero((f >= args.resp_band[0] - 1e-9) & (f <= args.resp_band[1] + 1e-9))
        k = int(band[np.argmax(pyy[band])])
        f_resp = float(f[k])
        row.update({"Breathing_Freq_Hz": f_resp, "Breathing_Rate_bpm": 60.0 * f_resp, "MSC": float(cxy[k]),
                    "MSC_Significant": bool(cxy[k] >= threshold)})
        # breath-to-breath amplitude
        inside = [br for br in breaths if br[0] >= a and br[2] <= b]
        period = float(np.median([(c - s) / FS for s, _, c in inside])) if inside else 1.0 / f_resp
        riiv_window = riiv[a:b]
        if np.mean(np.isfinite(riiv_window)) < 1.0 - MAX_RIIV_MISSING:
            row["BBA_Left_Out"] = "PPG beats not detected"
            rows.append(row)
            continue
        lag, lag_r = lag_of_riiv(riiv_window, y, period)
        amp_r, amp_p, kept = [], [], []
        for s, pk, e in inside:
            if rec["clipped"][s:e].any():
                continue
            s2, e2 = s + lag, e + lag
            if s2 < 0 or e2 > n:
                continue
            seg = riiv[s2:e2]
            beats = int(np.sum((centres >= s2) & (centres < e2)))
            if seg.size < 2 or not np.isfinite(seg).all() or beats < 2:
                continue
            amp_r.append(float(np.max(rsp[s:e]) - np.min(rsp[s:e])))
            amp_p.append(float(np.max(seg) - np.min(seg)))
            kept.append((s, pk, e, beats))
        row.update({"Breaths_Detected": len(inside), "BBA_Breaths": len(kept), "Lag_Sec": lag / FS,
                    "Lag_r": lag_r, "Breath_Period_Median_Sec": period,
                    "RSP_Amplitude_CV": cv(np.array(amp_r)), "RIIV_Amplitude_CV": cv(np.array(amp_p))})
        if len(kept) < args.min_breaths:
            row["BBA_Left_Out"] = f"fewer than {args.min_breaths} breaths"
            rows.append(row)
            continue
        ar, ap = np.array(amp_r), np.array(amp_p)
        if np.ptp(ar) <= 0 or np.ptp(ap) <= 0:
            row["BBA_Left_Out"] = "constant amplitude"
            rows.append(row)
            continue
        rho, p_two = spearmanr(ap, ar)
        r, _ = pearsonr(ap, ar)
        p_one = one_sided_greater(float(rho), float(p_two))
        row.update({"BBA_rho": float(rho), "BBA_p_one_sided": p_one, "BBA_Pearson_r": float(r),
                    "BBA_Significant": bool(rho > 0 and p_one < args.alpha)})
        rows.append(row)
        for (s, pk, e, beats), r_amp, p_amp in zip(kept, ar, ap):
            breath_rows.append({"Dataset": dataset, "Subject": subject, "Recording": name, "Window": i + 1,
                                "Breath_Start_Sec": s / FS, "Breath_Peak_Sec": pk / FS, "Breath_End_Sec": e / FS,
                                "Breath_Sec": (e - s) / FS, "Lag_Sec": lag / FS, "Beats": beats,
                                "RSP_Amplitude": r_amp, "RIIV_Amplitude": p_amp,
                                "RSP_Amplitude_Norm": r_amp / float(np.mean(ar)),
                                "RIIV_Amplitude_Norm": p_amp / float(np.mean(ap))})
    return rows, breath_rows


# ---------------------------------------------------------------------------
# Subject decisions and labels
# ---------------------------------------------------------------------------

def decision(significant: np.ndarray) -> str:
    """Strong: at least half of the valid windows significant; Weak: fewer; Undetermined: < 2 valid windows."""
    if significant.size < 2:
        return "Undetermined"
    return "Strong" if np.mean(significant) >= 0.5 else "Weak"


def label_of(coh: str, bba: str) -> str:
    if "Undetermined" in (coh, bba):
        return "Undetermined"
    if coh == "Weak" and bba == "Weak":
        return "Both weak"
    if bba == "Weak":
        return "BBA only weak"
    if coh == "Weak":
        return "Coherence only weak"
    return "Strong"


def summarize(rows: List[dict], breaths: List[dict]) -> dict:
    frame = pd.DataFrame(rows)
    out = {"Windows_Total": int(len(frame))}
    if frame.empty:
        out.update({"Coherence_Decision": "Undetermined", "BBA_Decision": "Undetermined", "Label": "Undetermined"})
        return out
    valid = frame[frame["Left_Out"] == ""]
    reasons = frame["Left_Out"].value_counts()
    out.update({"Windows_Valid": int(len(valid)), "Left_Out_Missing": int(reasons.get("missing samples", 0)),
                "Left_Out_Flat": int(reasons.get("flat signal", 0)),
                "Left_Out_Clipped": int(reasons.get("clipped reference", 0))})
    msc_sig = valid["MSC_Significant"].to_numpy(bool) if "MSC_Significant" in valid else np.empty(0, bool)
    if len(valid):
        msc = valid["MSC"].to_numpy(float)
        out.update({"MSC_Median": float(np.median(msc)), "MSC_Q1": float(np.percentile(msc, 25)),
                    "MSC_Q3": float(np.percentile(msc, 75)), "MSC_Significant_Windows": int(msc_sig.sum()),
                    "MSC_Significant_Share": float(np.mean(msc_sig)),
                    "Breathing_Rate_Median_bpm": float(np.median(valid["Breathing_Rate_bpm"]))})
    bba = valid[valid["BBA_Left_Out"] == ""] if "BBA_Left_Out" in valid else valid.iloc[0:0]
    bba_sig = bba["BBA_Significant"].to_numpy(bool) if "BBA_Significant" in bba else np.empty(0, bool)
    out["BBA_Windows_Valid"] = int(len(bba))
    out["BBA_Left_Out_No_Beats"] = int((valid.get("BBA_Left_Out", pd.Series(dtype=str)) == "PPG beats not detected").sum())
    out["BBA_Left_Out_Few_Breaths"] = int(valid.get("BBA_Left_Out", pd.Series(dtype=str)).astype(str)
                                          .str.startswith("fewer than").sum())
    if len(bba):
        rho = bba["BBA_rho"].to_numpy(float)
        out.update({"BBA_rho_Median": float(np.median(rho)), "BBA_rho_Q1": float(np.percentile(rho, 25)),
                    "BBA_rho_Q3": float(np.percentile(rho, 75)), "BBA_Significant_Windows": int(bba_sig.sum()),
                    "BBA_Significant_Share": float(np.mean(bba_sig)),
                    "BBA_Breaths": int(bba["BBA_Breaths"].sum()),
                    "Lag_Median_Sec": float(np.median(bba["Lag_Sec"])),
                    "RSP_Amplitude_CV_Median": float(np.nanmedian(bba["RSP_Amplitude_CV"])),
                    "RIIV_Amplitude_CV_Median": float(np.nanmedian(bba["RIIV_Amplitude_CV"]))})
        b = pd.DataFrame(breaths)
        if len(b) >= 3:
            rho_all, p_all = spearmanr(b["RIIV_Amplitude_Norm"], b["RSP_Amplitude_Norm"])
            out.update({"BBA_Pooled_rho": float(rho_all), "BBA_Pooled_p_one_sided": one_sided_greater(rho_all, p_all)})
    out["Coherence_Decision"] = decision(msc_sig)
    out["BBA_Decision"] = decision(bba_sig)
    out["Label"] = label_of(out["Coherence_Decision"], out["BBA_Decision"])
    return out


# ---------------------------------------------------------------------------
# Tables
# ---------------------------------------------------------------------------

def criterion_table(args: argparse.Namespace, k_segments: int, threshold: float) -> pd.DataFrame:
    return pd.DataFrame([
        ("Signals", f"{FS} Hz", "PPG and reference as recorded (CapnoBase capnogram, BIDMC impedance, STEAM2 belt; STEAM2 "
                                "sessions cut at their EEGLAB boundaries); missing samples interpolated; polyphase "
                                "resampling"),
        ("PPG filter", f"{args.ppg_band[0]:g}-{args.ppg_band[1]:g} Hz",
         f"{FILTER_ORDER}rd-order Butterworth band-pass, zero phase, whole recording"),
        ("Respiration filter", f"{args.resp_band[0]:g}-{args.resp_band[1]:g} Hz",
         f"{FILTER_ORDER}rd-order Butterworth band-pass, zero phase, whole recording"),
        ("Window", f"{args.window_sec:g} s", "not overlapping; only complete windows; the same windows for both "
                                             "criteria"),
        ("Windows left out", f"clipped >= {args.max_clipped_fraction:.0%}",
         f"missing samples; flat PPG or reference (SD below {FLAT_FRACTION:.0%} of the recording's median window SD); "
         "clipped (saturated) reference samples at or above this share"),
        ("BBA: breaths", "count-orig", f"respiration peaks > {PEAK_FRACTION:g} x 75th percentile of the peaks, troughs < "
                                       f"{PEAK_FRACTION:g} x 25th percentile of the troughs, alternating; breath = trough-"
                                       f"peak-trough of {BREATH_SEC[0]:g}-{BREATH_SEC[1]:g} s; breaths with clipped "
                                       "reference samples left out (Schafer & Kratky 2008)"),
        ("BBA: RIIV", "cycle means of the 0.1-8 Hz PPG",
         f"mean of every cardiac cycle (onset to onset; onsets = pulse troughs >= {BEAT_DISTANCE_FRACTION:g} x the "
         f"spectral beat period apart, {HR_RANGE_BPM[0]:g}-{HR_RANGE_BPM[1]:g} bpm; cycles "
         f"{BEAT_SEC[0]:.2f}-{BEAT_SEC[1]:g} s), linearly interpolated between the cycle centres; no interpolation "
         f"across gaps > {MAX_BEAT_GAP_SEC:g} s; a window needs it on >= {1 - MAX_RIIV_MISSING:.0%} of its samples. "
         "The respiratory-induced intensity variation of Johansson & Oberg 1999 / Nilsson et al. 2003"),
        ("BBA: lag", "+-half a breath period", "per window, the RIIV lag with the largest |r| against the respiration; "
                                               "the PPG breaths are the reference breaths shifted by it"),
        ("BBA: amplitude", "peak-to-valley", "maximum - minimum within each breath, respiration and RIIV (independent "
                                             "of the polarity); a PPG breath needs >= 2 cardiac cycles"),
        ("BBA: window", f"Spearman rho, >= {args.min_breaths} breaths",
         f"significant: rho > 0, one-sided p < {args.alpha:g} (a larger breath gives a larger RIIV)"),
        ("Coherence", f"MSC >= {threshold:.3f}",
         f"magnitude-squared coherence (Welch: {args.segment_sec:g} s Hann segments, not overlapping, linearly "
         f"detrended, K = {k_segments}) at the largest peak of the respiration spectrum in "
         f"{args.resp_band[0]:g}-{args.resp_band[1]:g} Hz; threshold 1 - alpha^(1/(K-1)) (Carter 1987)"),
        ("Subject decision", "Strong / Weak / Undetermined",
         "per criterion: Strong when at least half of the valid windows are significant; Weak otherwise; "
         "Undetermined with fewer than 2 valid windows"),
        ("Labels", " / ".join(LABELS),
         "Both weak: both Weak; BBA only weak: BBA Weak, coherence Strong; Coherence only weak: coherence Weak, BBA "
         "Strong; Strong: both Strong; Undetermined: either Undetermined"),
        ("Supplementary", "BBA_Pooled_rho, Pearson r, CV",
         "pooled Spearman rho over all breaths of a subject with the amplitudes divided by their window mean (as the "
         "per-subject normalisation of the papers); Pearson r per window; amplitude CV (little breath-to-breath "
         "variation of the reference leaves little to correlate)"),
        ("CapnoBase", "capnogram reference", "the breath amplitude of a capnogram follows the CO2 level, not the tidal "
                                             "volume: the BBA criterion is weaker evidence there"),
        ("Issue subjects", "; ".join(f"{d} {', '.join(s)}" for d, s in args.excluded.items() if s),
         "computed and marked (Issue_Subject); left out of Labels and Not_Strong"),
    ], columns=["Item", "Value", "Description"])


def write_tables(out_dir: Path, sheets: Dict[str, pd.DataFrame]) -> Path:
    """The workbook and the CSV files; a file held by another program (e.g. open in Excel) is written under a
    time-stamped name instead."""
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

    for name in ("Subjects", "Recordings", "Windows", "Breaths"):
        write(out_dir / f"{name}.csv", lambda p, f=sheets[name]: f.to_csv(p, index=False, encoding="utf-8-sig"))
    return write(out_dir / f"{NAME}.xlsx", xlsx)


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="PPG-respiration relationship per subject by breath-to-breath "
                                                 "amplitude and coherence (PPG 0.1-8 Hz, respiration 0.1-1 Hz)")
    parser.add_argument("--datasets", nargs="+", choices=list(FIGURE_ORDER), default=list(FIGURE_ORDER))
    parser.add_argument("--subjects", nargs="+", default=None, help="only these subjects")
    parser.add_argument("--output-root", type=Path, default=PROJECT_ROOT)
    parser.add_argument("--ppg-band", nargs=2, type=float, default=list(PPG_BAND_HZ))
    parser.add_argument("--resp-band", nargs=2, type=float, default=list(RESP_BAND_HZ))
    parser.add_argument("--window-sec", type=float, default=WINDOW_SEC)
    parser.add_argument("--segment-sec", type=float, default=SEGMENT_SEC)
    parser.add_argument("--alpha", type=float, default=ALPHA)
    parser.add_argument("--min-breaths", type=int, default=MIN_BREATHS)
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
    for name in ("ppg_band", "resp_band"):
        low, high = getattr(args, name)
        if not 0 < low < high < FS / 2:
            parser.error(f"--{name.replace('_', '-')}: 0 < low < high < {FS / 2:g} Hz")
    return args


def main(argv: Optional[Sequence[str]] = None) -> None:
    args = parse_args(argv)
    k_segments = int(round(args.window_sec / args.segment_sec))
    threshold = msc_threshold(k_segments, args.alpha)
    out_dir = Path(args.output_root)
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"Output: {out_dir}\nPPG {args.ppg_band[0]:g}-{args.ppg_band[1]:g} Hz, respiration "
          f"{args.resp_band[0]:g}-{args.resp_band[1]:g} Hz; {args.window_sec:g} s windows; BBA: Spearman rho of the "
          f"breath amplitudes (one-sided p < {args.alpha:g}); coherence: MSC >= {threshold:.3f} (K = {k_segments}); "
          "Strong when at least half of the windows are significant")
    windows, breaths, recordings, subjects = [], [], [], []
    for dataset in args.datasets:
        wanted = (lambda s: True) if not args.subjects else (lambda s: any(same_subject(s, x) for x in args.subjects))
        print(f"\n{dataset}: reading", flush=True)
        data = read_dataset(dataset, wanted, args)
        for subject in sorted(data, key=subject_order):
            issue = any(same_subject(subject, x) for x in args.excluded.get(dataset, []))
            subject_windows, subject_breaths = [], []
            for name, rec in data[subject].items():
                rows, breath_rows = window_rows(dataset, subject, name, rec, args, threshold)
                subject_windows += rows
                subject_breaths += breath_rows
                recordings.append({"Dataset": dataset, "Subject": subject, "Recording": name, "Issue_Subject": issue,
                                   "Seconds": len(rec["ppg"]) / FS, "PPG_FS": rec["ppg_fs"],
                                   "Reference_FS": rec["rsp_fs"], **summarize(rows, breath_rows)})
            windows += subject_windows
            breaths += subject_breaths
            subjects.append({"Dataset": dataset, "Subject": subject, "Issue_Subject": issue,
                             "Reference": MODALITIES[DATASET_KEYS[dataset]], "Recordings": len(data[subject]),
                             "Seconds": sum(len(r["ppg"]) for r in data[subject].values()) / FS,
                             **summarize(subject_windows, subject_breaths)})
            s = subjects[-1]
            print(f"  {subject}{' (issue subject)' if issue else ''}: coherence {s['Coherence_Decision']} "
                  f"({s.get('MSC_Significant_Windows', 0)}/{s.get('Windows_Valid', 0)} windows, median MSC "
                  f"{s.get('MSC_Median', math.nan):.3f}); BBA {s['BBA_Decision']} "
                  f"({s.get('BBA_Significant_Windows', 0)}/{s.get('BBA_Windows_Valid', 0)} windows, median rho "
                  f"{s.get('BBA_rho_Median', math.nan):.2f}) -> {s['Label']}", flush=True)
    subject_frame = pd.DataFrame(subjects)
    if subject_frame.empty:
        print("No subject read.")
        return
    first = ["Dataset", "Subject", "Issue_Subject", "Label", "BBA_Decision", "Coherence_Decision",
             "BBA_Significant_Share", "BBA_Significant_Windows", "BBA_Windows_Valid", "BBA_rho_Median", "BBA_rho_Q1",
             "BBA_rho_Q3", "BBA_Pooled_rho", "BBA_Pooled_p_one_sided", "BBA_Breaths", "RSP_Amplitude_CV_Median",
             "RIIV_Amplitude_CV_Median", "Lag_Median_Sec", "MSC_Significant_Share", "MSC_Significant_Windows",
             "Windows_Valid", "MSC_Median", "MSC_Q1", "MSC_Q3", "Breathing_Rate_Median_bpm", "Windows_Total",
             "Left_Out_Missing", "Left_Out_Flat", "Left_Out_Clipped", "BBA_Left_Out_No_Beats",
             "BBA_Left_Out_Few_Breaths", "Reference", "Recordings", "Seconds"]
    subject_frame = subject_frame[[c for c in first if c in subject_frame] + [c for c in subject_frame if c not in first]]
    usable = subject_frame[~subject_frame["Issue_Subject"]]
    order = {label: i for i, label in enumerate(LABELS)}
    not_strong = usable[usable["Label"] != "Strong"].copy()
    not_strong["_order"] = not_strong["Label"].map(order)
    not_strong["_dataset"] = not_strong["Dataset"].map({d: i for i, d in enumerate(FIGURE_ORDER)})
    sort_cols = [c for c in ("_order", "_dataset", "BBA_Significant_Share", "MSC_Significant_Share") if c in not_strong]
    not_strong = not_strong.sort_values(sort_cols).drop(columns=["_order", "_dataset"]).reset_index(drop=True)
    summary = []
    for dataset in [d for d in FIGURE_ORDER if d in set(subject_frame["Dataset"])]:
        part = usable[usable["Dataset"] == dataset]
        row = {"Dataset": dataset, "Subjects": len(part),
               "Issue_Subjects_Left_Out": int((subject_frame["Dataset"] == dataset).sum() - len(part))}
        for label in LABELS:
            row[label] = int((part["Label"] == label).sum())
        row["BBA_Weak"] = int((part["BBA_Decision"] == "Weak").sum())
        row["Coherence_Weak"] = int((part["Coherence_Decision"] == "Weak").sum())
        for label in LABELS[:-1]:
            row[f"{label}: subjects"] = ", ".join(part.loc[part["Label"] == label, "Subject"])
        row["Undetermined: subjects"] = ", ".join(part.loc[part["Label"] == "Undetermined", "Subject"])
        summary.append(row)
    sheets = {"Criterion": criterion_table(args, k_segments, threshold), "Subjects": subject_frame,
              "Labels": pd.DataFrame(summary), "Not_Strong": not_strong, "Recordings": pd.DataFrame(recordings),
              "Windows": pd.DataFrame(windows), "Breaths": pd.DataFrame(breaths)}
    path = write_tables(out_dir, sheets)
    pd.set_option("display.width", 220)
    print("\n" + pd.DataFrame(summary)[["Dataset", "Subjects", *LABELS]].to_string(index=False))
    if not not_strong.empty:
        print("\nNot strong (issue subjects left out):")
        cols = [c for c in ("Dataset", "Subject", "Label", "BBA_Significant_Share", "BBA_rho_Median",
                            "MSC_Significant_Share", "MSC_Median") if c in not_strong]
        print(not_strong[cols].round(3).to_string(index=False))
    print(f"\nSaved: {path}")


if __name__ == "__main__":
    main()
