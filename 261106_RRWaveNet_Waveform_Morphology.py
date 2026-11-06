r"""261106_RRWaveNet_Waveform_Morphology: how much of the breath shape
(duty cycle T_I/T_tot, I:E, phase durations, cycle shape, onsets) the 261105
RRWaveNet (final ensemble), RespDiff and the deep corr-encoder recover, per
breathing cycle against the recorded GT and against the GT as RRWaveNet was
trained on it; in the windows drawn by 261106_RRWaveNet_Performance_Boxplots_v2
(the showcase windows) and over whole recordings. Only stored results are
read; nothing is trained. No statistical test (descriptive: mean ± SD
(median) of the subject means).

Why a second reference ("Training target")
------------------------------------------
RRWaveNet is not trained on the recorded waveform:
  BIDMC, STEAM2  the reference band-passed to 0.1-0.5 Hz (261105 RESP_BAND_HZ,
                 Butterworth order 3, zero phase; then min-max scaled per 10 s
                 window). The 2nd and higher harmonics of a breath carry its
                 asymmetry (T_I != T_E); at 15-30 breaths/min they lie at
                 0.5-1.5 Hz, outside the band, so the target itself is close
                 to a sine: its T_I/T_tot is pushed towards 0.5.
  CapnoBase      a sine of the respiratory phase between the expert breath
                 labels (261105 label_phase_waveform, capnography_target
                 "phase"): T_I/T_tot = 0.5 and I:E = 1 by construction; only
                 the breath timing (onsets, breath length) is in the target.
The item "Training target" is the GT transformed exactly so (on the GT time
axis: no lag, the training lag / sign of RRWaveNet are undone on its output).
  * Against the GT, its errors are the ceiling of a model that reproduced its
    target perfectly: a model error close to them is everything the target
    allows; the rest of the shape was removed by the target, not missed by
    the model.
  * As the reference ("reference = Training target"), the models are measured
    against what RRWaveNet was taught (fair to RRWaveNet; RespDiff and the
    corr-encoder were trained on the recorded waveform, scaled per window).
CapnoBase: duty cycle, I:E and shape errors of RRWaveNet only say how far the
capnogram is from a sine; use the onset and breath length errors and the
coverage there.

Models and GT (as 261106_RRWaveNet_Performance_Boxplots_v2)
-----------------------------------------------------------
  RRWaveNet  261105 final ensemble ({root-261105}\{selected variant}\
             01_w10s_nonoverlap\{dataset}\Ensemble_Predictions\
             {subject}_ensemble_outputs.npz, variant from WD_Selection),
             10 s windows at 64 Hz, training sign / lag undone
             (Data_Quality\01_w10s_nonoverlap\{dataset}_window_quality.csv),
             taken at the nearest 30 Hz sample; no extra low-pass
             (--rrwavenet-lowpass-hz 1 = as the v2 pages)
  RespDiff   {respdiff-root}\respdiff\{dataset}\Predictions\{subject}_ddpm50.npz
  Deep corr-encoder  {correncoder-root}\{dataset}\Predictions\{subject}_outputs.npz
  GT         the reference as recorded at 30 Hz (CapnoBase capnogram, BIDMC
             impedance, STEAM2 belt), read and cut as v2; CapnoBase expert
             breath labels from read_capnobase_case of the 261007 (or 261105)
             script (found as in v2, or --scripts-dir)
Time: every waveform on the GT time axis, never shifted (as the v2 pages);
polarity: a waveform is inverted once per recording when its Pearson r with
the GT is negative (as the v2 pages; the sign used is in Alignment).
Clipped GT cycles (BIDMC impedance at the recorder's range) are not evaluated
(--keep-clipped evaluates them).

Per breathing cycle (261030_Waveform_Morphology_Analysis, unchanged)
-------------------------------------------------------------------
Every stretch: trend removed, high-pass 0.03 Hz, low-pass 2 Hz (capnogram,
belt) or 1 Hz (impedance), the same for the reference and every item;
cycles by the modality's detector (belt / impedance: trough -> peak ->
trough, inspiration = rising; capnogram: CO2 fall -> rise -> fall,
inspiration = low CO2); every reference cycle matched 1:1 to the item cycle
starting nearest (within half its length).
  T_I, T_E, T_tot (s), Ti_Ttot = T_I / T_tot (duty cycle), IE = T_I / T_E,
  A (cycle amplitude, z-score of the stretch), slope_ratio (peak inspiratory
  / expiratory slope), skew (cycle skewness)
  errors E_x = |item - reference| and biases B_x = item - reference:
  E_TiTtot, B_TiTtot, E_IE, E_logIE, E_TI, E_TE, B_TI, B_TE, E_Ttot, B_Ttot,
  E_A, E_slope, E_skew
  added here: E_shape = 1 - r of the two cycle shapes (each cycle from its own
  start to its own end resampled to 101 points and z-scored: the shape
  independent of timing and size); E_onset = |item start - reference start|
  (s), B_onset (s)
  coverage = matched / reference cycles; false cycle rate = unmatched item
  cycles / item cycles
Per unit (window, subject, dataset): Duty_r = Pearson r of reference and item
T_I/T_tot over the matched cycles (does the item follow the breath-to-breath
changes of the duty cycle? a sine-like output gives r ~ 0); Duty_SD_ratio =
SD of the item's T_I/T_tot / SD of the reference's (1 = as variable, ~0 =
a fixed duty cycle).

Analyses (--mode windows | recordings | both)
--------------------------------------------
windows     the showcase windows (default: the 3 subjects per dataset and
            size picked from the v2 Candidates, SHOWCASE below; --windows-csv
            with Dataset, Subject, Recording, Size_Sec, Start_Sec, End_Sec;
            or --windows-from-summary Waveform_Summary_all_runs.xlsx = every
            drawn window of 261105). The stretch is the window widened by
            --pad-sec (15 s) on both sides inside the item's continuous
            output, so that the cycles at the window edges are complete; the
            cycles starting in the window are kept. Every item is evaluated
            (a missing model is marked).
recordings  every subject with all three models: the continuous stretches of
            at least 30 s where the GT and all three models have output (as
            261030); per subject the mean of every measure over the cycles
            matched by all items of the reference (common set; coverage and
            false cycles over each item's own cycles).

Outputs (default D:\PPG2RSP_RRWaveNet_Inspired\261106_RRWaveNet_Waveform_Morphology)
  Windows\{30s,60s}\{dataset}\{subject or recording}_{t0}-{t1}s_morphology.png
          a page per window: (a) GT with its cycles (inspiration shaded),
          (b) RRWaveNet with its cycles, (c) mean cycle shape (reference GT,
          ± SD) of the GT and of every item in its own panel, (d) T_I/T_tot
          per cycle, GT vs item, in its own panel; the numbers beside
  Windows\{30s,60s}\Morphology_{dataset}_{size}s.pdf
  Windows\window_summary.csv, window_cycles.csv
  Recordings\{dataset}_shapes_duty.png (as (c), (d) over all common cycles),
          {dataset}_subject_measures.png (subject means: E_TiTtot, B_TiTtot,
          E_shape, Duty_SD_ratio, E_onset, coverage)
  Recordings\per_subject.csv, cycle_table.csv, summary.csv
  261106_RRWaveNet_Waveform_Morphology.xlsx  Summary_Recordings (Mean ± SD
          (Median) of the subject means), Per_Subject, Window_Summary, QC,
          Alignment, Notes
  settings.json

Run (in D:\PPG2RSP_RRWaveNet_Inspired\261106_RRWaveNet_Waveform_Morphology):
  python 261106_RRWaveNet_Waveform_Morphology.py --self-test   (check first)
  python 261106_RRWaveNet_Waveform_Morphology.py
Options: --mode, --datasets capnobase bidmc steam2, --subjects, --windows-csv,
  --windows-from-summary, --pad-sec, --rrwavenet-lowpass-hz, --keep-clipped,
  --min-run-sec, --root-261105, --respdiff-root, --correncoder-root,
  --scripts-dir, --output-root, --capnobase-path, --bidmc-path,
  --steam2-ppg-dir, --steam2-rsp-dir, --no-figures
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
from matplotlib.patches import Patch
from scipy.signal import butter, detrend, resample_poly, sosfiltfilt

# ---------------------------------------------------------------------------
# Folders and settings
# ---------------------------------------------------------------------------

WORKSPACE = Path(r"D:\PPG2RSP_RRWaveNet_Inspired")
PROJECT_ROOT = WORKSPACE / "261106_RRWaveNet_Waveform_Morphology"
ROOT_261105 = WORKSPACE / "261105_RRWaveNet_Unified_Optimized"
RESPDIFF_ROOT = WORKSPACE / "261020_RespDiff_v2"
CORRENCODER_ROOT = WORKSPACE / "261028_Deep_correncoder"
RESPDIFF_SAMPLING = "ddpm50"
SCRIPT_DIR = Path(__file__).resolve().parent
# CapnoBase reader with the expert breath labels (as v2)
CAPNOBASE_READERS = (("261007", "261007_RRWaveNet_Ensemble_NoEvent.py"),
                     ("261105", "261105_RRWaveNet_Unified_Optimized.py"))
m07 = None  # module holding read_capnobase_case, loaded by load_capnobase_reader

WINDOW = "01_w10s_nonoverlap"
FS30 = 30  # analysis rate: the 30 Hz of the GT, RespDiff and the corr-encoder
RRWAVENET_FS = 64
RRWAVENET_LOWPASS_HZ = 0.0  # no extra low-pass (1.0 = as the v2 pages)
RRWAVENET_LOWPASS_ORDER = 8
DATASETS = ("CapnoBase", "BIDMC", "STEAM2")
DATASET_KEYS = {"CapnoBase": "capnobase", "BIDMC": "bidmc", "STEAM2": "steam2"}
MODALITIES = {"capnobase": "capnogram", "bidmc": "impedance", "steam2": "belt"}

# The GT as RRWaveNet was trained on it (261105): BIDMC / STEAM2 band-pass
TARGET_BAND_HZ = (0.1, 0.5)
TARGET_BAND_ORDER = 3

TARGET = "Training target"
MODELS = ("RRWaveNet", "RespDiff", "Deep corr-encoder")
ITEMS = (TARGET,) + MODELS
REFERENCES = {"GT": ITEMS, TARGET: MODELS}  # reference -> the items measured against it
COLORS = {TARGET: "#8c8b87", "RRWaveNet": "#1baf7a", "RespDiff": "#2a78d6", "Deep corr-encoder": "#eb6834"}
GT_COLOR = "#000000"
INK, MUTED, GRID = "#1f1f1f", "#6b6a66", "#e4e3df"
TARGET_TEXT = {"capnobase": "label-phase sine", "bidmc": "GT 0.1-0.5 Hz", "steam2": "GT 0.1-0.5 Hz"}

PAD_SEC = 15.0
MIN_RUN_SEC = 30.0
SHAPE_POINTS = 101

# The showcase windows (261105, drawn by v2; picked from its Candidates:
# RRWaveNet rate right and PCC high in the window, the others failing there,
# RRWaveNet best for the subject, no outlier subject, BIDMC without clipped
# GT cycles): (dataset, size, subject, recording, start, end)
SHOWCASE = (
    ("CapnoBase", 30, "0330", "0330", 285, 315), ("CapnoBase", 30, "0149", "0149", 185, 215),
    ("CapnoBase", 30, "0123", "0123", 235, 265),
    ("CapnoBase", 60, "0330", "0330", 320, 380), ("CapnoBase", 60, "0149", "0149", 170, 230),
    ("CapnoBase", 60, "0123", "0123", 410, 470),
    ("BIDMC", 30, "25", "25", 385, 415), ("BIDMC", 30, "29", "29", 15, 45), ("BIDMC", 30, "09", "09", 25, 55),
    ("BIDMC", 60, "25", "25", 180, 240), ("BIDMC", 60, "29", "29", 20, 80), ("BIDMC", 60, "09", "09", 0, 60),
    ("STEAM2", 30, "41_JYM", "41_JYM_Session1_piece1", 120, 150),
    ("STEAM2", 30, "44_JSJ", "44_JSJ_Session1_piece1", 10, 40),
    ("STEAM2", 30, "19_KSM", "19_KSM_Session3_piece1", 130, 160),
    ("STEAM2", 60, "28_WSY", "28_WSY_Session4_piece1", 100, 160),
    ("STEAM2", 60, "43_KDY", "43_KDY_Session2_piece1", 110, 170),
    ("STEAM2", 60, "48_KHH", "48_KHH_Session2_piece1", 25, 85),
)

# Data files of the GT (as v2 and the three scripts)
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
STEAM2_CHANNELS = {"PPG": ("PPG", "PLETH", "BVP"), "RSP": ("RSP", "RESP", "RESPIRATION", "BELT")}
STEAM2_SESSION_PATTERN = re.compile(r"^(?P<subject>.+?)[_\- ]*session[_\- ]*(?P<session>\d+)$", re.IGNORECASE)
SATURATION_CONFIG = {"saturation_min_sec": 0.1, "saturation_margin_sec": 0.1}

# Cycle detection and measures (261030 CONFIG, unchanged)
CONFIG = {
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
FEATURES = ("T_I", "T_E", "T_tot", "IE", "Ti_Ttot", "A", "peak_I", "peak_E", "slope_ratio", "skew")
ERRORS = (
    ("E_TiTtot", "duty cycle error |T_I/T_tot|", ""),
    ("B_TiTtot", "duty cycle bias (item - reference)", ""),
    ("E_IE", "I:E error", ""),
    ("E_logIE", "|log(IE_item / IE_ref)|", ""),
    ("E_TI", "inspiratory duration error", "s"),
    ("E_TE", "expiratory duration error", "s"),
    ("B_TI", "inspiratory duration bias", "s"),
    ("B_TE", "expiratory duration bias", "s"),
    ("E_Ttot", "breath length error", "s"),
    ("B_Ttot", "breath length bias", "s"),
    ("E_A", "cycle amplitude error", "SD"),
    ("E_slope", "peak-slope ratio error", ""),
    ("E_skew", "skewness error", ""),
    ("E_shape", "cycle shape error 1 - r", ""),
    ("E_onset", "cycle onset error", "s"),
    ("B_onset", "cycle onset bias", "s"),
)
ERROR_COLUMNS = [e[0] for e in ERRORS]
EVALUATED = ("matched", "missed", "polarity_refused")
# Columns of the Mean ± SD (median) summary
KEY_MEASURES = ("coverage", "false_cycle_rate", "ref_Ti_Ttot", "item_Ti_Ttot", "E_TiTtot", "B_TiTtot", "Duty_r",
                "Duty_SD_ratio", "E_logIE", "E_TI", "E_TE", "E_Ttot", "E_shape", "E_onset", "E_skew", "E_slope",
                "E_A")


def cycle_shape(y: np.ndarray, fs: float, cycle: dict, points: int = SHAPE_POINTS) -> np.ndarray:
    """The cycle from its start to its end resampled to `points` samples and
    z-scored (the shape independent of timing and size)."""
    t = np.linspace(cycle["start"], cycle["end"], points) * fs
    x = np.interp(t, np.arange(len(y)), y)
    sd = float(np.std(x))
    return (x - np.mean(x)) / sd if sd > 1e-12 else np.full(points, np.nan)


# ---------------------------------------------------------------------------
# Copied from 261106_RRWaveNet_Performance_Boxplots_v2 (data files, stored results of the 
# three models, RRWaveNet on the GT axis)
# ---------------------------------------------------------------------------

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


def safe_name(text: str) -> str:
    return re.sub(r"[^0-9A-Za-z._-]+", "_", str(text))


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
        path = find_script(name, [args.scripts_dir, getattr(args, f"root_{run}", None)])
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
          "261105_RRWaveNet_Unified_Optimized.py; --scripts-dir): CapnoBase skipped")
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



# ---------------------------------------------------------------------------
# Copied from 261030_Waveform_Morphology_Analysis (per-cycle morphology engine; added: the 
# T_I/T_tot bias, the cycle shape and onset errors, the added self-test cases)
# ---------------------------------------------------------------------------

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


def scaled(y: np.ndarray, config: dict) -> np.ndarray:
    if config["amplitude_scale"] == "zscore":
        sd = np.std(y)
        return (y - np.mean(y)) / sd if sd > config["sigma_eps"] else y * np.nan
    return y


def cycle_features(y: np.ndarray, ys: np.ndarray, dy: np.ndarray, fs: float, cycle: dict, modality: str,
                   config: dict, qc: Dict[str, int]) -> dict:
    """Features of one cycle: y the filtered waveform, ys the scaled one, dy
    the derivative of ys."""
    start, transition, end = cycle["start"], cycle["transition"], cycle["end"]
    n = len(y)

    def idx(t: float) -> int:
        return int(min(max(round(t * fs), 0), n))

    i0, it, i1 = idx(start), idx(transition), idx(end)
    t_i, t_e = transition - start, end - transition
    out = {"T_I": t_i, "T_E": t_e, "T_tot": t_i + t_e, "Ti_Ttot": t_i / (t_i + t_e),
           "IE": t_i / t_e if t_e >= config["min_phase_sec"] else math.nan}
    seg = ys[i0:i1]
    out["A"] = float(np.max(seg) - np.min(seg)) if len(seg) and np.all(np.isfinite(seg)) else math.nan
    if modality == "capnogram":
        h = config["capnogram_transition_fraction"] * (t_i + t_e)
        down = dy[idx(start - h):idx(start + h)]
        up = dy[idx(transition - h):idx(transition + h)]
        peak_i = float(np.max(-down)) if len(down) else math.nan
        peak_e = float(np.max(up)) if len(up) else math.nan
    else:
        rise, fall = dy[i0:it], dy[it:i1]
        peak_i = float(np.max(rise)) if len(rise) else math.nan
        peak_e = float(np.max(-fall)) if len(fall) else math.nan
    out["peak_I"], out["peak_E"] = peak_i, peak_e
    if np.isfinite(peak_i) and np.isfinite(peak_e) and peak_e > config["slope_eps"] and peak_i > 0:
        out["slope_ratio"] = peak_i / peak_e
    else:
        out["slope_ratio"] = math.nan
        qc["slope ratio invalid (denominator ~0)"] = qc.get("slope ratio invalid (denominator ~0)", 0) + 1
    x = y[i0:i1]
    sigma = float(np.std(x)) if len(x) else 0.0
    if sigma > config["sigma_eps"]:
        out["skew"] = float(np.mean(((x - np.mean(x)) / sigma) ** 3))
    else:
        out["skew"] = math.nan
        qc["skewness invalid (cycle SD ~0)"] = qc.get("skewness invalid (cycle SD ~0)", 0) + 1
    if not np.isfinite(out["IE"]):
        qc["I:E invalid (T_E too short)"] = qc.get("I:E invalid (T_E too short)", 0) + 1
    return out


def match_cycles(gt: List[dict], pred: List[dict], tolerance: float) -> List[Optional[int]]:
    """For every GT cycle in order, the index of its model cycle (start
    nearest, within tolerance x the GT cycle length, order kept, 1:1) or
    None."""
    starts = np.array([c["start"] for c in pred]) if pred else np.empty(0)
    matched, next_free = [], 0
    for cycle in gt:
        limit = tolerance * (cycle["end"] - cycle["start"])
        best, best_gap = None, None
        for j in range(next_free, len(pred)):
            gap = starts[j] - cycle["start"]
            if gap > limit:
                break
            if abs(gap) <= limit and (best_gap is None or abs(gap) < best_gap):
                best, best_gap = j, abs(gap)
        matched.append(best)
        if best is not None:
            next_free = best + 1
    return matched


def cycle_errors(g: dict, p: dict) -> dict:
    out = {
        "E_IE": abs(g["IE"] - p["IE"]),
        "E_TI": abs(g["T_I"] - p["T_I"]), "E_TE": abs(g["T_E"] - p["T_E"]),
        "B_TI": p["T_I"] - g["T_I"], "B_TE": p["T_E"] - g["T_E"],
        "E_A": abs(g["A"] - p["A"]),
        "E_slope": abs(g["slope_ratio"] - p["slope_ratio"]),
        "E_skew": abs(g["skew"] - p["skew"]),
        "E_TiTtot": abs(g["Ti_Ttot"] - p["Ti_Ttot"]), "B_TiTtot": p["Ti_Ttot"] - g["Ti_Ttot"],
        "E_logIE": abs(math.log(p["IE"] / g["IE"])) if g["IE"] > 0 and p["IE"] > 0 else math.nan,
        "E_Ttot": abs(g["T_tot"] - p["T_tot"]), "B_Ttot": p["T_tot"] - g["T_tot"],
    }
    return {k: (float(v) if v is not None and np.isfinite(v) else math.nan) for k, v in out.items()}


def evaluate_morphology(gt: np.ndarray, pred: np.ndarray, fs: float, modality: str, subject_id: str,
                        config: dict) -> dict:
    """The interface of the specification: GT and model waveform of one
    continuous stretch (same rate and time axis) -> {cycle_table, summary,
    qc, shapes}; shapes: GT cycle id -> (GT, model) cycle shape (phase-normalized,
    see cycle_shape) of every matched cycle that is evaluated. config may carry 'window_id', 'time_offset_sec' (start of the
    stretch in its recording) and 'gt_saturated' (boolean mask of clipped GT
    samples: the GT cycles that contain one are not evaluated, for any
    model)."""
    window_id = config.get("window_id", "")
    offset = float(config.get("time_offset_sec", 0.0))
    qc: Dict[str, int] = {}
    gt = np.asarray(gt, dtype=float)
    pred = np.asarray(pred, dtype=float)
    if len(gt) != len(pred) or not np.all(np.isfinite(gt)) or not np.all(np.isfinite(pred)):
        raise ValueError("GT and model need the same length and only finite samples")
    gy, py = band_filter(gt, fs, modality, config), band_filter(pred, fs, modality, config)
    gt_cycles, gt_qc = detect_cycles(gy, fs, modality, config)
    for key, value in gt_qc.items():
        qc[f"GT: {key}"] = value
    r = pearson(gy, py)
    refused = np.isfinite(r) and r <= config["polarity_min_r"]
    if refused:
        qc["model: inverted polarity (not compared)"] = 1
        pred_cycles = []
    else:
        pred_cycles, pred_qc = detect_cycles(py, fs, modality, config)
        for key, value in pred_qc.items():
            qc[f"model: {key}"] = value
    gys, pys = scaled(gy, config), scaled(py, config)
    gdy, pdy = np.gradient(gys) * fs, np.gradient(pys) * fs
    gt_features = [cycle_features(gy, gys, gdy, fs, c, modality, config, qc) for c in gt_cycles]
    pred_features = [cycle_features(py, pys, pdy, fs, c, modality, config, qc) for c in pred_cycles]
    if refused:
        matches = [None] * len(gt_cycles)
    else:
        matches = match_cycles(gt_cycles, pred_cycles, config["match_tolerance"])
    mask = config.get("gt_saturated")
    saturated = [bool(mask is not None and np.any(mask[max(int(round(c["start"] * fs)), 0):
                                                       int(round(c["end"] * fs)) + 1]))
                 for c in gt_cycles]
    if any(saturated):
        qc["GT: cycle with clipped (saturated) samples, not evaluated"] = int(sum(saturated))
    rows, shapes = [], {}
    for i, (cycle, features, j) in enumerate(zip(gt_cycles, gt_features, matches)):
        status = "polarity_refused" if refused else ("matched" if j is not None else "missed")
        row = {"subject_id": subject_id, "window_id": window_id, "cycle_id": i,
               "match_status": "gt_saturated" if saturated[i] else status,
               "gt_t_start": offset + cycle["start"], "gt_t_transition": offset + cycle["transition"],
               "gt_t_end": offset + cycle["end"]}
        row.update({f"gt_{k}": features[k] for k in FEATURES})
        if j is not None:
            p = pred_cycles[j]
            row.update({"pred_t_start": offset + p["start"], "pred_t_transition": offset + p["transition"],
                        "pred_t_end": offset + p["end"]})
            row.update({f"pred_{k}": pred_features[j][k] for k in FEATURES})
            if not saturated[i]:
                row.update(cycle_errors(features, pred_features[j]))
                g_shape, p_shape = cycle_shape(gy, fs, cycle), cycle_shape(py, fs, p)
                r_shape = pearson(g_shape, p_shape)
                row.update({"E_shape": 1.0 - r_shape if np.isfinite(r_shape) else math.nan,
                            "E_onset": abs(p["start"] - cycle["start"]), "B_onset": p["start"] - cycle["start"]})
                shapes[i] = (g_shape.astype(np.float32), p_shape.astype(np.float32))
        rows.append(row)
    used = {j for j in matches if j is not None}
    clipped_spans = [(c["start"], c["end"]) for c, s in zip(gt_cycles, saturated) if s]
    n_in_clipped = 0
    for j, (cycle, features) in enumerate(zip(pred_cycles, pred_features)):
        if j not in used:
            # a model cycle in the time of a clipped GT cycle is left out, not false
            in_clipped = any(a <= cycle["start"] < b for a, b in clipped_spans)
            n_in_clipped += in_clipped
            row = {"subject_id": subject_id, "window_id": window_id, "cycle_id": -1,
                   "match_status": "in_gt_saturated" if in_clipped else "false",
                   "pred_t_start": offset + cycle["start"], "pred_t_transition": offset + cycle["transition"],
                   "pred_t_end": offset + cycle["end"]}
            row.update({f"pred_{k}": features[k] for k in FEATURES})
            rows.append(row)
    table = pd.DataFrame(rows)
    n_gt = len(gt_cycles) - sum(saturated)
    n_matched = sum(1 for j, s in zip(matches, saturated) if j is not None and not s)
    n_pred = len(pred_cycles) - sum(1 for j, s in zip(matches, saturated) if j is not None and s) - n_in_clipped
    summary = {"subject_id": subject_id, "window_id": window_id, "polarity_r": r, "refused": bool(refused),
               "n_gt_cycles": n_gt, "n_gt_saturated": int(sum(saturated)), "n_pred_cycles": n_pred,
               "n_matched": n_matched, "cycle_coverage": n_matched / n_gt if n_gt else math.nan,
               "false_cycle_rate": (n_pred - n_matched) / max(n_pred, 1)}
    if n_matched:
        matched = table[table["match_status"] == "matched"]
        for column in ERROR_COLUMNS:
            summary[column] = float(matched[column].mean())
    return {"cycle_table": table, "summary": summary, "qc": qc, "shapes": shapes}


def synthetic_breaths(durations: Sequence[Tuple[float, float]], fs: float, kind: str = "volume") -> np.ndarray:
    """Breaths with the given (T_I, T_E): volume-like (half-cosine rise and
    fall between 0 and 1) or capnogram-like (CO2 low in inspiration, high in
    expiration, short ramps)."""
    out = []
    for t_i, t_e in durations:
        n_i, n_e = int(round(t_i * fs)), int(round(t_e * fs))
        if kind == "volume":
            out.append(0.5 - 0.5 * np.cos(np.pi * np.arange(n_i) / n_i))
            out.append(0.5 + 0.5 * np.cos(np.pi * np.arange(n_e) / n_e))
        else:
            ramp = max(2, int(0.15 * fs))
            low = np.zeros(n_i)
            low[:ramp] = np.linspace(1.0, 0.0, ramp)
            high = np.ones(n_e)
            high[:ramp] = np.linspace(0.0, 1.0, ramp)
            out += [low, high]
    return np.concatenate(out)


def self_test() -> bool:
    fs = 30.0
    rng = np.random.default_rng(3)
    base = [(1.5, 2.5)] * 4 + [(1.2, 2.8), (1.8, 2.2)] * 4 + [(1.5, 2.5)] * 4
    gt = synthetic_breaths(base, fs)
    config = dict(CONFIG)
    results = []

    def run(pred, modality="belt", gt_=gt, **overrides):
        return evaluate_morphology(gt_, pred, fs, modality, "test", {**config, **overrides})

    def report(name, ok, detail):
        results.append(ok)
        print(f"  [{'PASS' if ok else 'FAIL'}] {name}: {detail}")

    s = run(gt.copy())["summary"]
    report("identical (pred = gt)", s["cycle_coverage"] == 1 and all(abs(s[c]) < 1e-9 for c in ERROR_COLUMNS),
           f"coverage {s['cycle_coverage']:.2f}, largest error "
           f"{max(abs(s[c]) for c in ERROR_COLUMNS):.2g}")

    s = run(0.6 * gt, amplitude_scale="none")["summary"]
    timing = max(s["E_IE"], s["E_TI"], s["E_TE"], s["E_skew"], s["E_slope"])
    report("amplitude x 0.6 (common scale)", timing < 1e-9 and s["E_A"] > 0.1,
           f"timing / skewness / slope ratio errors {timing:.2g}, E_A {s['E_A']:.3f}")
    s = run(0.6 * gt)["summary"]
    report("amplitude x 0.6 (z-score, default)", s["E_A"] < 1e-9,
           f"E_A {s['E_A']:.2g}: a global gain is not measured with --amplitude-scale zscore")

    changed = list(base)
    changed[6] = (2.4, 1.6)
    s = run(synthetic_breaths(changed, fs))["summary"]
    table = run(synthetic_breaths(changed, fs))["cycle_table"]
    worst = table.loc[table["E_IE"].idxmax()]
    report("time allocation of one cycle changed", worst["E_IE"] > 0.5 and max(worst["E_TI"], worst["E_TE"]) > 0.5,
           f"largest cycle E_IE {worst['E_IE']:.2f}, E_TI {worst['E_TI']:.2f} s, E_TE {worst['E_TE']:.2f} s")

    result = run(-gt)
    s = result["summary"]
    report("inverted polarity (pred = -gt)", s["refused"] and s["n_matched"] == 0,
           f"r {s['polarity_r']:.2f}, refused, qc: {list(result['qc'])}")

    short = [(1.5, 2.5)] * 8
    merged = [(1.5, 2.5)] * 3 + [(3.0, 5.0)] + [(1.5, 2.5)] * 3
    gt_short = synthetic_breaths(short, fs)
    result = evaluate_morphology(gt_short, synthetic_breaths(merged, fs)[:len(gt_short)], fs, "belt", "test", config)
    s = result["summary"]
    missed = (result["cycle_table"]["match_status"] == "missed").sum()
    report("missing breath in the model", missed >= 1 and s["cycle_coverage"] < 1,
           f"GT {s['n_gt_cycles']} cycles, model {s['n_pred_cycles']}, missed {missed}, "
           f"coverage {s['cycle_coverage']:.2f}")

    flat = np.full_like(gt, 0.3)
    result = run(flat)
    report("constant signal", result["summary"]["n_pred_cycles"] == 0 and "model: constant signal" in result["qc"],
           f"qc: {list(result['qc'])}")
    result = evaluate_morphology(flat, gt, fs, "belt", "test", config)
    report("constant GT", result["summary"]["n_gt_cycles"] == 0 and "GT: constant signal" in result["qc"],
           f"qc: {list(result['qc'])}")

    big = gt.copy()
    big[int(16 * fs):int(20 * fs)] *= 1.3
    big = np.minimum(big, 1.0)  # one deeper breath clipped by the recorder
    mask = to_fs_mask(saturated_samples(big, fs, "belt", config), fs, len(big), config["saturation_margin_sec"])
    result = run(big, gt_=big, gt_saturated=mask)
    s, table = result["summary"], result["cycle_table"]
    flagged = table.loc[table["match_status"] == "gt_saturated", "gt_t_start"].round(1).tolist()
    report("clipped GT breath", s["n_gt_saturated"] == 1 and flagged == [16.0] and s["cycle_coverage"] == 1
           and s["E_TI"] < 1e-9, f"left out: the cycle at {flagged} s; {s['n_gt_cycles']} cycles evaluated, "
           f"coverage {s['cycle_coverage']:.2f}")

    capno_breaths = [(1.6, 2.4)] * 12
    capno = synthetic_breaths(capno_breaths, fs, kind="capnogram") + 0.01 * rng.standard_normal(int(4.0 * fs) * 12)
    table = evaluate_morphology(capno, capno, fs, "capnogram", "test", config)["cycle_table"]
    t_i = table["gt_T_I"].median() if len(table) else math.nan
    report("CO2 shape (capnogram detector)", abs(t_i - 1.6) < 0.15,
           f"{len(table)} cycles, T_I {t_i:.2f} s (true 1.6 s: CO2 low = inspiration, not rising = inspiration)")
    self_test_added(report)
    ok = all(results)
    print(f"\nSelf-test: {sum(results)} of {len(results)} cases passed" + ("" if ok else " - do not use the results"))
    return ok



# ---------------------------------------------------------------------------
# Copied from 261105_RRWaveNet_Unified_Optimized (CapnoBase training target)
# ---------------------------------------------------------------------------

def label_phase_waveform(labels: np.ndarray, length: int) -> np.ndarray:
    """Respiratory-phase waveform from expert breath onsets.

    The phase rises linearly from 0 to 1 between consecutive labels (and is
    extrapolated with the first and last interval), and the waveform is
    sin(2 pi phase), so every labelled breath is one cycle whose upward zero
    crossing is exactly the label. RR needs only breath timing; the rest of
    a capnogram (trapezoid shape, inspiratory/expiratory split, EtCO2 level)
    varies between subjects and cannot be inferred from the PPG, so this
    target keeps the timing and drops what the model cannot predict.
    """
    labels = np.unique(np.asarray(labels, dtype=np.float64))
    samples = np.arange(length, dtype=np.float64)
    intervals = np.diff(labels)
    index = np.clip(np.searchsorted(labels, samples, side="right") - 1, 0, len(labels) - 2)
    phase = (samples - labels[index]) / intervals[index]
    return np.sin(2.0 * np.pi * phase).astype(np.float32)



# ---------------------------------------------------------------------------
# The GT as RRWaveNet was trained on it, polarity, the waveforms of a recording
# ---------------------------------------------------------------------------

def training_target(gt: np.ndarray, breaths: Optional[np.ndarray], dataset_key: str) -> Optional[np.ndarray]:
    """The GT as RRWaveNet (261105) was trained on it, on the GT time axis:
    BIDMC / STEAM2 band-passed 0.1-0.5 Hz (Butterworth order 3, zero phase);
    CapnoBase the label-phase sine of the expert breath labels (s); None
    without at least 3 labels or 10 GT samples."""
    if dataset_key == "capnobase":
        if breaths is None or len(breaths) < 3:
            return None
        return label_phase_waveform(np.asarray(breaths, float) * FS30, len(gt)).astype(np.float64)
    x = np.asarray(gt, dtype=float)
    keep = np.isfinite(x)
    if keep.sum() < 10:
        return None
    x = np.where(keep, x, np.median(x[keep]))
    sos = butter(TARGET_BAND_ORDER, TARGET_BAND_HZ, btype="band", fs=FS30, output="sos")
    out = sosfiltfilt(sos, x - np.mean(x))
    out[~keep] = np.nan
    return out


def sign_matched(gt: np.ndarray, values: np.ndarray) -> Tuple[np.ndarray, float]:
    """The waveform inverted when its Pearson r with the GT over the recording
    is negative (as the v2 pages), and the sign used."""
    values = np.asarray(values, dtype=float)
    r = pearson(gt, values)
    return (-values, -1.0) if np.isfinite(r) and r < 0 else (values, 1.0)


def dataset_sources(args: argparse.Namespace, dataset: str) -> dict:
    corr = load_correncoder(Path(args.correncoder_root), dataset)
    respdiff = load_respdiff(Path(args.respdiff_root), dataset, args.respdiff_sampling)
    files = {f["Tag"]: f["Outputs_File"] for f in fold_files(Path(args.root_261105), dataset, True)
             if f["Outputs_File"].exists()}
    alignment = rrwavenet_alignment_table(Path(args.root_261105), dataset)
    print(f"{dataset}: RRWaveNet 261105 {len(files)} subjects (alignment table: "
          f"{'found' if alignment else 'not found, lag 0'}), RespDiff {len(respdiff)}, corr-encoder {len(corr)}",
          flush=True)
    return {"corr": corr, "respdiff": respdiff, "files": files, "alignment": alignment}


def subject_values(args: argparse.Namespace, dataset: str, subject: str, recordings: Dict[str, dict],
                   sources: dict, alignment_rows: List[dict]) -> Dict[str, dict]:
    """Recording name -> the GT, the training target and the three models on
    the GT time axis (30 Hz, not shifted; sign-matched to the GT) and the
    clipped GT samples."""
    key = DATASET_KEYS[dataset]
    corr_by_name = {r["name"]: r for r in sources["corr"].get(subject, [])}
    segments = load_rrwavenet(sources["files"][subject]) if subject in sources["files"] else {}
    out = {}
    for k, (name, rec) in enumerate(recordings.items()):
        gt = np.asarray(rec["gt"], dtype=float)
        times = np.arange(len(gt)) / FS30
        index = corr_by_name[name]["index"] if name in corr_by_name else k
        values = {"GT": gt}
        target = training_target(gt, rec.get("breaths"), key)
        if target is not None:
            values[TARGET] = target
        stored = sources["alignment"].get((subject, index))
        if index in segments:
            values["RRWaveNet"] = rrwavenet_track(segments[index], stored, key, gt, args.rrwavenet_lowpass_hz)
        if name in sources["respdiff"].get(subject, {}):
            values["RespDiff"] = Track(sources["respdiff"][subject][name], FS30).at(times)
        if name in corr_by_name:
            values["Deep corr-encoder"] = Track(corr_by_name[name]["output"], FS30).at(times)
        signs = {}
        for item in ITEMS:
            if item in values:
                values[item], signs[item] = sign_matched(gt, values[item])
        clipped = np.asarray(rec["clipped"], dtype=bool)
        alignment_rows.append({
            "Dataset": dataset, "Subject": subject, "Recording": name, "Segment": index,
            "GT_Sec": len(gt) / FS30, "GT_Clipped_Sec": float(clipped.sum() / FS30),
            "Items": ", ".join(item for item in ITEMS if item in values),
            "RRWaveNet_Training_Lag_Sec": stored["lag_sec"] if stored else (0.0 if index in segments else math.nan),
            "RRWaveNet_Training_Sign": stored["sign"] if stored else math.nan,
            "RRWaveNet_Alignment_Source": ("data quality table" if stored else
                                           "none found (lag 0)" if index in segments else "no RRWaveNet output"),
            **{f"Sign_{item}": signs.get(item, math.nan) for item in ITEMS}})
        out[name] = {"values": values, "clipped": clipped}
    return out


# ---------------------------------------------------------------------------
# Counts and duty cycle measures of a set of cycles
# ---------------------------------------------------------------------------

def sd(values) -> float:
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    return float(values.std(ddof=1)) if len(values) > 1 else math.nan


def duty_stats(matched: pd.DataFrame) -> dict:
    """Mean and SD of T_I/T_tot of the reference and the item over matched
    cycles, their Pearson r (Duty_r) and the SD ratio item / reference."""
    if matched.empty or "gt_Ti_Ttot" not in matched or "pred_Ti_Ttot" not in matched:
        return {k: math.nan for k in ("ref_Ti_Ttot", "ref_Ti_Ttot_SD", "item_Ti_Ttot", "item_Ti_Ttot_SD",
                                      "Duty_r", "Duty_SD_ratio")}
    g, p = matched["gt_Ti_Ttot"].to_numpy(float), matched["pred_Ti_Ttot"].to_numpy(float)
    keep = np.isfinite(g) & np.isfinite(p)
    g, p = g[keep], p[keep]
    sd_g, sd_p = sd(g), sd(p)
    return {"ref_Ti_Ttot": float(g.mean()) if len(g) else math.nan, "ref_Ti_Ttot_SD": sd_g,
            "item_Ti_Ttot": float(p.mean()) if len(p) else math.nan, "item_Ti_Ttot_SD": sd_p,
            "Duty_r": pearson(g, p) if len(g) >= 3 else math.nan,
            "Duty_SD_ratio": sd_p / sd_g if np.isfinite(sd_g) and sd_g > 1e-9 else math.nan}


def unit_stats(table: pd.DataFrame) -> dict:
    """Counts, coverage, false cycle rate, duty cycle measures and the mean of
    every error over the matched cycles of one item in one unit."""
    if table.empty or "match_status" not in table:
        out = {"n_ref_cycles": 0, "n_ref_clipped": 0, "n_matched": 0, "n_false": 0, "coverage": math.nan,
               "false_cycle_rate": math.nan, **duty_stats(pd.DataFrame())}
        out.update({column: math.nan for column in ERROR_COLUMNS})
        return out
    status = table["match_status"]
    reference = table[status.isin(EVALUATED)]
    matched = reference[reference["match_status"] == "matched"]
    n_false = int((status == "false").sum())
    n_pred = len(matched) + n_false
    out = {"n_ref_cycles": int(len(reference)), "n_ref_clipped": int((status == "gt_saturated").sum()),
           "n_matched": int(len(matched)), "n_false": n_false,
           "coverage": len(matched) / len(reference) if len(reference) else math.nan,
           "false_cycle_rate": n_false / n_pred if n_pred else math.nan}
    out.update(duty_stats(matched))
    for column in ERROR_COLUMNS:
        out[column] = float(matched[column].mean()) if column in matched and len(matched) else math.nan
    return out


# ---------------------------------------------------------------------------
# Showcase windows
# ---------------------------------------------------------------------------

def load_windows(args: argparse.Namespace) -> pd.DataFrame:
    columns = ["Dataset", "Size_Sec", "Subject", "Recording", "Start_Sec", "End_Sec"]
    if args.windows_csv:
        frame = pd.read_csv(args.windows_csv, dtype={"Subject": str, "Recording": str})
    elif args.windows_from_summary:
        frame = pd.read_excel(args.windows_from_summary, sheet_name="Drawn_Windows",
                              dtype={"Subject": str, "Recording": str})
        frame = frame[frame["Run"].astype(str) == "261105"]
    else:
        frame = pd.DataFrame(SHOWCASE, columns=columns)
    missing = [c for c in columns if c not in frame]
    if missing:
        raise ValueError(f"windows: columns missing: {missing}")
    frame = frame[columns].copy()
    frame["Recording"] = frame["Recording"].where(frame["Recording"].notna(), frame["Subject"]).astype(str)
    frame["Subject"] = frame["Subject"].astype(str)
    return frame


def window_analysis(args: argparse.Namespace, dataset: str, windows: pd.DataFrame, values_of, config: dict,
                    out_root: Path, qc_rows: List[dict]) -> Tuple[List[pd.DataFrame], List[dict]]:
    """Every item against both references in every window of a dataset; a
    morphology page per window."""
    key = DATASET_KEYS[dataset]
    modality = MODALITIES[key]
    tables, summaries, pdfs = [], [], {}
    pad = int(round(args.pad_sec * FS30))
    for _, w in windows.iterrows():
        subject, name, size = str(w["Subject"]), str(w["Recording"]), int(w["Size_Sec"])
        t0, t1 = float(w["Start_Sec"]), float(w["End_Sec"])
        window_id = f"{size}s {t0:.0f}-{t1:.0f}s"
        meta = {"Dataset": dataset, "Size_Sec": size, "Subject": subject, "Recording": name,
                "Start_Sec": t0, "End_Sec": t1}
        rec = values_of(subject).get(name)
        if rec is None:
            qc_rows.append({**meta, "reason": "recording not read or no result of any model", "count": 1})
            continue
        values, clipped = rec["values"], rec["clipped"]
        n = len(values["GT"])
        a, b = int(round(t0 * FS30)), min(int(round(t1 * FS30)), n)
        results = {}
        for reference, items in REFERENCES.items():
            if reference not in values:
                qc_rows.append({**meta, "Reference": reference, "reason": "no training target (no breath labels)",
                                "count": 1})
                continue
            ref = values[reference]
            for item in items:
                row = {**meta, "Reference": reference, "Item": item}
                if item not in values:
                    summaries.append({**row, "Status": "no result for this subject"})
                    continue
                valid = np.isfinite(ref) & np.isfinite(values[item])
                if b - a < FS30 or not valid[a:b].all():
                    summaries.append({**row, "Status": "output has a gap in the window"})
                    continue
                s, e = a, b
                low, high = max(0, a - pad), min(n, b + pad)
                while s > low and valid[s - 1]:
                    s -= 1
                while e < high and valid[e]:
                    e += 1
                result = evaluate_morphology(ref[s:e], values[item][s:e], FS30, modality, subject,
                                             {**config, "window_id": window_id, "time_offset_sec": s / FS30,
                                              "gt_saturated": None if args.keep_clipped else clipped[s:e]})
                for reason, count in result["qc"].items():
                    qc_rows.append({**row, "reason": reason, "count": count})
                table = result["cycle_table"]
                if not table.empty:
                    start = table["gt_t_start"] if "gt_t_start" in table else pd.Series(np.nan, index=table.index)
                    if "pred_t_start" in table:
                        start = start.where(start.notna(), table["pred_t_start"])
                    table = table[(start >= t0) & (start < t1)].copy()
                    for column, value in reversed(list(row.items())):
                        table.insert(0, column, value)
                    tables.append(table)
                shapes = {cid: result["shapes"][cid] for cid in (table["cycle_id"] if not table.empty else [])
                          if cid in result["shapes"]}
                results[(reference, item)] = {"table": table, "shapes": shapes, "stretch": (s, e)}
                summaries.append({**row, "Status": "ok", **unit_stats(table)})
        if args.no_figures or not results:
            continue
        stem = (safe_name(subject) if name == subject else safe_name(name) if name.startswith(subject)
                else f"{safe_name(subject)}__{safe_name(name)}")
        path = out_root / "Windows" / f"{size}s" / dataset / f"{stem}_{t0:.0f}-{t1:.0f}s_morphology.png"
        if size not in pdfs:
            pdf_path = out_root / "Windows" / f"{size}s" / f"Morphology_{dataset}_{size}s.pdf"
            pdf_path.parent.mkdir(parents=True, exist_ok=True)
            pdfs[size] = PdfPages(pdf_path)
        window_page(path, pdfs[size], meta, values, results, key)
    for pdf in pdfs.values():
        pdf.close()
    return tables, summaries


# ---------------------------------------------------------------------------
# Whole recordings
# ---------------------------------------------------------------------------

def recording_analysis(args: argparse.Namespace, dataset: str, subjects: Sequence[str], values_of, config: dict,
                       qc_rows: List[dict]) -> Tuple[List[pd.DataFrame], dict]:
    """Every recording of a subject: the stretches (>= min_run_sec) where the
    GT and all items have output; every item against both references."""
    modality = MODALITIES[DATASET_KEYS[dataset]]
    min_run = int(round(args.min_run_sec * FS30))
    tables, shapes = [], {}
    for subject in subjects:
        n_before = len(tables)
        for name, rec in values_of(subject).items():
            values, clipped = rec["values"], rec["clipped"]
            meta = {"Dataset": dataset, "Subject": subject, "Recording": name}
            missing = [item for item in ITEMS if item not in values]
            if missing:
                qc_rows.append({**meta, "reason": "not analysed, no " + ", ".join(missing), "count": 1})
                continue
            valid = np.isfinite(values["GT"])
            for item in ITEMS:
                valid &= np.isfinite(values[item])
            runs = finite_runs(valid, min_run)
            if not runs:
                qc_rows.append({**meta, "reason": f"no common stretch of {args.min_run_sec:g} s", "count": 1})
                continue
            for k, (s, e) in enumerate(runs):
                window_id = f"{name}#{k + 1}"
                for reference, items in REFERENCES.items():
                    for item in items:
                        result = evaluate_morphology(values[reference][s:e], values[item][s:e], FS30, modality,
                                                     subject, {**config, "window_id": window_id,
                                                               "time_offset_sec": s / FS30,
                                                               "gt_saturated": None if args.keep_clipped
                                                               else clipped[s:e]})
                        for reason, count in result["qc"].items():
                            qc_rows.append({**meta, "Window": window_id, "Reference": reference, "Item": item,
                                            "reason": reason, "count": count})
                        table = result["cycle_table"]
                        if table.empty:
                            continue
                        table.insert(0, "Item", item)
                        table.insert(0, "Reference", reference)
                        table.insert(0, "Recording", name)
                        table.insert(0, "Dataset", dataset)
                        tables.append(table)
                        for cid, pair in result["shapes"].items():
                            shapes[(reference, item, subject, window_id, cid)] = pair
        print(f"  [{dataset}] {subject}: {len(tables) - n_before} item/reference stretches", flush=True)
    return tables, shapes


def mark_common(cycles: pd.DataFrame) -> pd.DataFrame:
    """common = the reference cycle is matched by every item of its reference."""
    cycles = cycles.copy()
    cycles["common"] = False
    key = ["Dataset", "subject_id", "window_id", "cycle_id"]
    for reference, items in REFERENCES.items():
        mask = (cycles["Reference"] == reference) & (cycles["match_status"] == "matched")
        counts = cycles[mask].groupby(key)["Item"].nunique()
        keys = set(counts[counts == len(items)].index)
        cycles.loc[mask, "common"] = [k in keys for k in cycles.loc[mask, key].itertuples(index=False, name=None)]
    return cycles


def subject_table(cycles: pd.DataFrame) -> pd.DataFrame:
    """Per dataset, reference, item and subject: counts, coverage and false
    cycles over the item's own cycles; duty cycle measures and errors over the
    common cycles."""
    rows = []
    for (dataset, reference, item, subject), part in cycles.groupby(["Dataset", "Reference", "Item", "subject_id"],
                                                                     sort=False):
        own = unit_stats(part)
        common = unit_stats(part[part["common"]])
        row = {"Dataset": dataset, "Reference": reference, "Item": item, "Subject": subject,
               **{k: own[k] for k in ("n_ref_cycles", "n_ref_clipped", "n_matched", "n_false", "coverage",
                                      "false_cycle_rate")},
               "n_common": common["n_matched"]}
        row.update({k: v for k, v in common.items() if k not in row and not k.startswith("n_")
                    and k not in ("coverage", "false_cycle_rate")})
        rows.append(row)
    return pd.DataFrame(rows)


def mean_sd_median(values) -> Tuple[float, float, float, int]:
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if not len(values):
        return math.nan, math.nan, math.nan, 0
    return float(values.mean()), sd(values), float(np.median(values)), int(len(values))


def summary_table(subjects: pd.DataFrame, cycles: pd.DataFrame) -> pd.DataFrame:
    """Per dataset, reference and item: Mean ± SD (Median) of the subject
    means of the key measures, and the duty cycle r / SD ratio over all
    common cycles."""
    rows = []
    for (dataset, reference, item), part in subjects.groupby(["Dataset", "Reference", "Item"], sort=False):
        row = {"Dataset": dataset, "Reference": reference, "Item": item, "N_Subjects": int(len(part)),
               "N_Ref_Cycles": int(part["n_ref_cycles"].sum()), "N_Common_Cycles": int(part["n_common"].sum())}
        for measure in KEY_MEASURES:
            m, s, md, n = mean_sd_median(part[measure]) if measure in part else (math.nan, math.nan, math.nan, 0)
            row[measure] = f"{m:.3f} ± {s:.3f} ({md:.3f})" if n else ""
        pooled = cycles[(cycles["Dataset"] == dataset) & (cycles["Reference"] == reference)
                        & (cycles["Item"] == item) & cycles["common"]]
        duty = duty_stats(pooled)
        row["Duty_r_all_cycles"] = duty["Duty_r"]
        row["Duty_SD_ratio_all_cycles"] = duty["Duty_SD_ratio"]
        for measure in KEY_MEASURES:
            m, s, md, n = mean_sd_median(part[measure]) if measure in part else (math.nan, math.nan, math.nan, 0)
            row.update({f"{measure}_Mean": m, f"{measure}_SD": s, f"{measure}_Median": md, f"{measure}_N": n})
        rows.append(row)
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------

def style(axis) -> None:
    axis.grid(True, color=GRID, linewidth=0.6)
    axis.set_axisbelow(True)
    for side in ("top", "right"):
        axis.spines[side].set_visible(False)
    axis.tick_params(labelsize=7.5, colors=MUTED)


def zscore(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    s = np.nanstd(x)
    return (x - np.nanmean(x)) / s if s > 1e-12 else x * 0.0


def item_label(item: str, dataset_key: str) -> str:
    if item == TARGET:
        return f"Training target ({TARGET_TEXT[dataset_key]})"
    return "RRWaveNet (261105)" if item == "RRWaveNet" else item


def draw_waveform(axis, values: np.ndarray, stretch: Tuple[int, int], t0: float, t1: float, table: pd.DataFrame,
                  who: str, color: str, modality: str, onsets: Sequence[float]) -> None:
    """The filtered waveform of the window (as analysed), z-scored, with the
    inspiration of its cycles shaded and the GT cycle onsets dashed."""
    s, e = stretch
    y = band_filter(values[s:e], FS30, modality, CONFIG)
    t = s / FS30 + np.arange(e - s) / FS30
    inside = (t >= t0) & (t <= t1)
    axis.plot(t[inside], zscore(y[inside]), color=color, lw=1.4, zorder=3)
    if not table.empty and f"{who}_t_start" in table:
        for _, c in table.dropna(subset=[f"{who}_t_start", f"{who}_t_transition"]).iterrows():
            if c["match_status"] in ("matched", "missed", "false", "gt_saturated"):
                a, b = max(c[f"{who}_t_start"], t0), min(c[f"{who}_t_transition"], t1)
                if b > a:
                    axis.axvspan(a, b, color="#d9d8d4" if who == "gt" else color, alpha=0.35 if who == "gt"
                                 else 0.15, lw=0, zorder=1)
    for onset in onsets:
        if t0 <= onset <= t1:
            axis.axvline(onset, color=MUTED, lw=0.7, ls="--", zorder=2)
    axis.set_xlim(t0, t1)
    style(axis)


def draw_shapes(axis, shapes: Dict[int, Tuple[np.ndarray, np.ndarray]], table: pd.DataFrame, item: str) -> None:
    """Mean ± SD cycle shape (phase 0-1 of the cycle) of the reference and
    the item, and their mean T_I/T_tot (dashed)."""
    style(axis)
    phase = np.linspace(0.0, 1.0, SHAPE_POINTS)
    if not shapes:
        axis.text(0.5, 0.5, "no matched cycle", transform=axis.transAxes, ha="center", va="center", fontsize=8,
                  color=MUTED)
        return
    g = np.array([v[0] for v in shapes.values()], dtype=float)
    p = np.array([v[1] for v in shapes.values()], dtype=float)
    for data, color in ((g, GT_COLOR), (p, COLORS[item])):
        mean, spread = np.nanmean(data, axis=0), np.nanstd(data, axis=0)
        axis.fill_between(phase, mean - spread, mean + spread, color=color, alpha=0.12, lw=0)
        axis.plot(phase, mean, color=color, lw=1.6 if color == GT_COLOR else 1.4)
    matched = table[table["match_status"] == "matched"] if not table.empty else table
    if not matched.empty:
        axis.axvline(np.nanmean(matched["gt_Ti_Ttot"]), color=GT_COLOR, lw=0.9, ls="--")
        axis.axvline(np.nanmean(matched["pred_Ti_Ttot"]), color=COLORS[item], lw=0.9, ls="--")
    axis.set_xlim(0, 1)
    axis.set_ylim(-2.6, 2.6)
    axis.set_xlabel("phase of the cycle (start -> end)", fontsize=7.5)


def draw_duty(axis, table: pd.DataFrame, item: str, alpha: float = 0.9) -> None:
    style(axis)
    axis.plot([0.15, 0.85], [0.15, 0.85], color=MUTED, lw=0.8, ls="--")
    matched = table[table["match_status"] == "matched"] if not table.empty else table
    if not matched.empty:
        axis.scatter(matched["gt_Ti_Ttot"], matched["pred_Ti_Ttot"], s=14 if alpha > 0.5 else 5,
                     color=COLORS[item], alpha=alpha, edgecolors="white" if alpha > 0.5 else "none", linewidths=0.4)
    else:
        axis.text(0.5, 0.5, "no matched cycle", transform=axis.transAxes, ha="center", va="center", fontsize=8,
                  color=MUTED)
    axis.set_xlim(0.15, 0.85)
    axis.set_ylim(0.15, 0.85)
    axis.set_xlabel("GT  T_I / T_tot", fontsize=7.5)


def numbers_text(stats: dict, unit: str = "") -> str:
    def f(key, digits=2):
        v = stats.get(key, math.nan)
        return f"{v:.{digits}f}" if isinstance(v, (int, float)) and np.isfinite(v) else "-"
    return (f"matched {stats.get('n_matched', 0)}/{stats.get('n_ref_cycles', 0)} (coverage {f('coverage')}),"
            f" false {stats.get('n_false', 0)}\n"
            f"T_I/T_tot  GT {f('ref_Ti_Ttot')} ± {f('ref_Ti_Ttot_SD')}, item {f('item_Ti_Ttot')} ± "
            f"{f('item_Ti_Ttot_SD')}\n"
            f"duty error {f('E_TiTtot', 3)}, bias {f('B_TiTtot', 3)}\n"
            f"duty r {f('Duty_r')}, SD ratio {f('Duty_SD_ratio')}\n"
            f"shape 1-r {f('E_shape', 3)}, onset {f('E_onset')} s{unit}")


def page_layout(width: float, rows: Sequence[Tuple[str, float]], gap: float, top: float, bottom: float,
                gaps: Optional[Dict[str, float]] = None):
    """Figure height and (bottom, height) in figure fractions of every row;
    gaps: the gap after a row when not `gap`."""
    gaps = gaps or {}
    after = [gaps.get(name, gap) for name, _ in rows[:-1]]
    height = top + bottom + sum(h for _, h in rows) + sum(after)
    y, boxes = height - top, {}
    for k, (name, h) in enumerate(rows):
        y -= h
        boxes[name] = (y / height, h / height)
        y -= after[k] if k < len(after) else 0.0
    return height, boxes


def legend_handles(dataset_key: str):
    handles = [Line2D([], [], color=GT_COLOR, lw=1.6)] + [Line2D([], [], color=COLORS[i], lw=1.4) for i in ITEMS]
    labels = ["GT (recorded)"] + [item_label(i, dataset_key) for i in ITEMS]
    handles += [Patch(color="#d9d8d4", alpha=0.6), Line2D([], [], color=MUTED, lw=0.7, ls="--")]
    labels += ["inspiration (T_I) of a cycle", "GT cycle onset / mean T_I/T_tot"]
    return handles, labels


def four_panels(fig, y0: float, h: float, left: float, right: float, width: float, gap: float):
    panel = (right - left - gap * (len(ITEMS) - 1)) / len(ITEMS)
    return [fig.add_axes([(left + k * (panel + gap)) / width, y0, panel / width, h]) for k in range(len(ITEMS))]


def window_page(path: Path, pdf, meta: dict, values: Dict[str, np.ndarray], results: dict, dataset_key: str) -> None:
    """(a) GT and (b) RRWaveNet of the window with their cycles, (c) the mean
    cycle shapes and (d) T_I/T_tot per cycle of every item against the GT."""
    modality = MODALITIES[dataset_key]
    t0, t1 = meta["Start_Sec"], meta["End_Sec"]
    width, left, side, gap = 15.0, 0.8, 3.0, 0.45
    height, box = page_layout(width, [("a", 1.45), ("b", 1.45), ("c", 1.9), ("d", 1.95)], 0.62, 1.0, 1.45,
                              gaps={"b": 1.0})
    fig = plt.figure(figsize=(width, height))
    right = width - side
    base = next((results[k] for k in (("GT", "RRWaveNet"),) + tuple(("GT", i) for i in ITEMS) if k in results), None)
    gt_table = base["table"] if base else pd.DataFrame()
    onsets = gt_table["gt_t_start"].dropna().unique().tolist() if "gt_t_start" in gt_table else []
    ax_a = fig.add_axes([left / width, box["a"][0], (right - left) / width, box["a"][1]])
    if base:
        draw_waveform(ax_a, values["GT"], base["stretch"], t0, t1, gt_table, "gt", GT_COLOR, modality, onsets)
    ax_a.set_title("(a) GT (recorded, filtered as analysed) and its breathing cycles", fontsize=9.5, loc="left")
    ax_a.tick_params(labelbottom=False)
    ax_b = fig.add_axes([left / width, box["b"][0], (right - left) / width, box["b"][1]], sharex=ax_a)
    rr = results.get(("GT", "RRWaveNet"))
    if rr:
        draw_waveform(ax_b, values["RRWaveNet"], rr["stretch"], t0, t1, rr["table"], "pred", COLORS["RRWaveNet"],
                      modality, onsets)
    else:
        style(ax_b)
        ax_b.set_yticks([])
        ax_b.set_xlim(t0, t1)
        ax_b.text(0.5, 0.5, "RRWaveNet: no output in this window", transform=ax_b.transAxes, ha="center",
                  fontsize=8, color=MUTED)
    ax_b.set_title("(b) RRWaveNet (261105) and its cycles (dashed: GT onsets)", fontsize=9.5, loc="left")
    ax_b.set_xlabel("Time in the recording (s)", fontsize=8)
    shape_axes = four_panels(fig, box["c"][0], box["c"][1], left, right, width, gap)
    duty_axes = four_panels(fig, box["d"][0], box["d"][1], left, right, width, gap)
    for k, (item, ax_c, ax_d) in enumerate(zip(ITEMS, shape_axes, duty_axes)):
        result = results.get(("GT", item))
        label = item_label(item, dataset_key)
        ax_c.set_title(f"(c{k + 1}) cycle shape: GT vs\n{label}", fontsize=8.5, loc="left")
        ax_d.set_title(f"(d{k + 1}) T_I/T_tot per cycle", fontsize=8.5, loc="left")
        if result is None:
            for axis in (ax_c, ax_d):
                style(axis)
                axis.set_xticks([])
                axis.set_yticks([])
                axis.text(0.5, 0.5, f"{item}:\nno result in this window", transform=axis.transAxes,
                          ha="center", va="center", fontsize=8, color=MUTED)
            continue
        draw_shapes(ax_c, result["shapes"], result["table"], item)
        draw_duty(ax_d, result["table"], item)
        if k == 0:
            ax_d.set_ylabel("item  T_I / T_tot", fontsize=7.5)
            ax_c.set_ylabel("z-scored cycle", fontsize=7.5)
        x0 = ax_d.get_position().x0
        fig.text(x0, box["d"][0] - 0.5 / height, numbers_text(unit_stats(result["table"])), fontsize=6.9,
                 family="monospace", color=INK, va="top")
    handles, labels = legend_handles(dataset_key)
    fig.legend(handles, labels, loc="upper left", bbox_to_anchor=((right + 0.25) / width, box["a"][0] + box["a"][1]),
               frameon=False, fontsize=8)
    fig.text((right + 0.25) / width, box["c"][0] + box["c"][1], (
        "Reference: the recorded GT.\n(c) each cycle from its start to its\nend (101 points, z-scored); mean ± SD;\n"
        "dashed: mean T_I/T_tot.\n(d) dots on the diagonal: the item\nhas the GT's duty cycle; a flat row\n"
        "at 0.5: a sine-like output.\nTraining target = the GT as RRWaveNet\nwas trained on it: its errors are the\n"
        "ceiling of a perfect RRWaveNet."), fontsize=7.2, color=MUTED, va="top")
    name = meta["Subject"] + ("" if meta["Recording"] == meta["Subject"] else f" ({meta['Recording']})")
    fig.text(left / width, 1.0 - 0.25 / height, f"{meta['Dataset']} {name} · {meta['Size_Sec']} s window "
             f"{t0:.0f}-{t1:.0f} s · breath morphology (261105 RRWaveNet, RespDiff, deep corr-encoder)",
             fontsize=10.5, color=INK, va="top")
    fig.text(left / width, 1.0 - 0.55 / height,
             "no waveform shifted in time; inverted when r < 0 over the recording; every waveform filtered alike "
             f"(high-pass 0.03 Hz, low-pass {CONFIG['lowpass_hz'][modality]:g} Hz); clipped GT cycles not evaluated",
             fontsize=7.2, color=MUTED, va="top")
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=170)
    pdf.savefig(fig)
    plt.close(fig)


def recording_figures(cycles: pd.DataFrame, shapes: dict, subjects: pd.DataFrame, root: Path) -> None:
    root.mkdir(parents=True, exist_ok=True)
    for dataset in cycles["Dataset"].unique():
        key = DATASET_KEYS[dataset]
        width, left, side, gap = 15.0, 0.8, 3.0, 0.45
        height, box = page_layout(width, [("c", 2.1), ("d", 2.3)], 0.75, 1.3, 1.5)
        fig = plt.figure(figsize=(width, height))
        right = width - side
        shape_axes = four_panels(fig, box["c"][0], box["c"][1], left, right, width, gap)
        duty_axes = four_panels(fig, box["d"][0], box["d"][1], left, right, width, gap)
        part_all = cycles[(cycles["Dataset"] == dataset) & (cycles["Reference"] == "GT") & cycles["common"]]
        for k, (item, ax_c, ax_d) in enumerate(zip(ITEMS, shape_axes, duty_axes)):
            table = part_all[part_all["Item"] == item]
            keys = set(table[["subject_id", "window_id", "cycle_id"]].itertuples(index=False, name=None))
            item_shapes = {i: v for i, ((ref, it, s, w, c), v) in enumerate(shapes.items())
                           if ref == "GT" and it == item and (s, w, c) in keys}
            label = item_label(item, key)
            ax_c.set_title(f"(a{k + 1}) cycle shape: GT vs\n{label}", fontsize=8.5, loc="left")
            ax_d.set_title(f"(b{k + 1}) T_I/T_tot per cycle", fontsize=8.5, loc="left")
            draw_shapes(ax_c, item_shapes, table, item)
            draw_duty(ax_d, table, item, alpha=0.25)
            if k == 0:
                ax_c.set_ylabel("z-scored cycle", fontsize=7.5)
                ax_d.set_ylabel("item  T_I / T_tot", fontsize=7.5)
            sub = subjects[(subjects["Dataset"] == dataset) & (subjects["Reference"] == "GT")
                           & (subjects["Item"] == item)]
            lines = []
            for measure, digits in (("E_TiTtot", 3), ("B_TiTtot", 3), ("Duty_r", 2), ("Duty_SD_ratio", 2),
                                    ("E_shape", 3), ("E_onset", 2), ("coverage", 2)):
                m, s, md, n = mean_sd_median(sub[measure]) if measure in sub else (math.nan, math.nan, math.nan, 0)
                lines.append(f"{measure:<13s} {m:.{digits}f} ± {s:.{digits}f} ({md:.{digits}f})" if n
                             else f"{measure:<13s} -")
            fig.text(ax_d.get_position().x0, box["d"][0] - 0.5 / height,
                     "subject means: mean ± SD (median)\n" + "\n".join(lines), fontsize=6.8, family="monospace",
                     color=INK, va="top")
        handles, labels = legend_handles(key)
        fig.legend(handles[:5] + handles[-1:], labels[:5] + ["mean T_I/T_tot"], loc="upper left",
                   bbox_to_anchor=((right + 0.25) / width, box["c"][0] + box["c"][1]), frameon=False, fontsize=8)
        n_subjects = part_all["subject_id"].nunique()
        n_cycles = int((part_all["Item"] == "RRWaveNet").sum())
        fig.text(left / width, 1.0 - 0.25 / height, f"{dataset}: breath morphology over whole recordings "
                 f"(reference: recorded GT; {n_cycles} cycles matched by every item, {n_subjects} subjects)",
                 fontsize=10.5, color=INK, va="top")
        fig.text(left / width, 1.0 - 0.55 / height, "cycles matched by the training target and all three models; "
                 "(a) mean ± SD of the z-scored cycles; (b) every matched cycle", fontsize=7.2, color=MUTED, va="top")
        fig.savefig(root / f"{dataset}_shapes_duty.png", dpi=160)
        plt.close(fig)

        measures = (("E_TiTtot", "duty cycle error"), ("B_TiTtot", "duty cycle bias (item - GT)"),
                    ("E_shape", "cycle shape error 1 - r"), ("Duty_SD_ratio", "duty cycle SD ratio (item / GT)"),
                    ("E_onset", "onset error (s)"), ("coverage", "cycle coverage"))
        fig, axes = plt.subplots(2, 3, figsize=(13.5, 7.2))
        sub = subjects[(subjects["Dataset"] == dataset) & (subjects["Reference"] == "GT")]
        for axis, (measure, title) in zip(axes.ravel(), measures):
            style(axis)
            data = [sub.loc[sub["Item"] == item, measure].dropna().to_numpy(float) for item in ITEMS]
            positions = np.arange(1, len(ITEMS) + 1)
            box_ = axis.boxplot(data, positions=positions, widths=0.5, patch_artist=True, showfliers=False,
                                medianprops={"color": INK, "linewidth": 1.3})
            for patch, item in zip(box_["boxes"], ITEMS):
                patch.set_facecolor(COLORS[item])
                patch.set_alpha(0.3)
            rng = np.random.default_rng(0)
            for position, values, item in zip(positions, data, ITEMS):
                axis.scatter(position + rng.uniform(-0.1, 0.1, len(values)), values, s=9, color=COLORS[item],
                             edgecolors="white", linewidths=0.4, zorder=3)
            if measure == "B_TiTtot":
                axis.axhline(0, color=MUTED, lw=0.8, ls="--")
            if measure == "Duty_SD_ratio":
                axis.axhline(1, color=MUTED, lw=0.8, ls="--")
            axis.set_xticks(positions)
            axis.set_xticklabels(["Training\ntarget", "RRWaveNet", "RespDiff", "Corr-enc."], fontsize=7.5)
            axis.set_title(title, fontsize=9, loc="left")
        fig.suptitle(f"{dataset}: subject means (reference: recorded GT; errors over the cycles matched by every "
                     "item, coverage over each item's own cycles)", fontsize=10)
        fig.tight_layout(rect=(0, 0, 1, 0.95))
        fig.savefig(root / f"{dataset}_subject_measures.png", dpi=160)
        plt.close(fig)


# ---------------------------------------------------------------------------
# Self-test: the added cases
# ---------------------------------------------------------------------------

def self_test_added(report) -> None:
    fs = float(FS30)
    config = dict(CONFIG)
    gt = synthetic_breaths([(1.2, 2.8)] * 16, fs)  # T_I/T_tot 0.30 at 15 breaths/min
    table = evaluate_morphology(gt, training_target(gt, None, "bidmc"), fs, "belt", "test", config)["cycle_table"]
    m = table[table["match_status"] == "matched"]
    g, p = m["gt_Ti_Ttot"].median(), m["pred_Ti_Ttot"].median()
    report("training target 0.1-0.5 Hz of asymmetric breaths", abs(p - 0.5) < abs(g - 0.5) - 0.03,
           f"T_I/T_tot GT {g:.2f}, training target {p:.2f} (the band pulls it towards 0.5)")

    labels = np.arange(1.0, 60.0, 4.0)
    sine = training_target(np.zeros(int(64 * fs)), labels, "capnobase")
    table = evaluate_morphology(sine, sine, fs, "belt", "test", config)["cycle_table"]
    d = table["gt_Ti_Ttot"].median()
    report("label-phase sine (CapnoBase training target)", abs(d - 0.5) < 0.02,
           f"T_I/T_tot {d:.3f} (0.5 by construction)")

    later = np.concatenate([np.full(int(round(0.3 * fs)), gt[0]), gt[:len(gt) - int(round(0.3 * fs))]])
    table = evaluate_morphology(gt, later, fs, "belt", "test", config)["cycle_table"]
    m = table[table["match_status"] == "matched"]
    report("the same breaths 0.3 s later", m["E_shape"].median() < 0.01 and abs(m["B_onset"].median() - 0.3) < 0.05
           and m["E_TiTtot"].median() < 0.01,
           f"shape error {m['E_shape'].median():.4f}, onset bias {m['B_onset'].median():.2f} s, duty error "
           f"{m['E_TiTtot'].median():.4f}")

    varied = synthetic_breaths([(1.0, 3.0), (1.4, 2.6), (1.8, 2.2)] * 6, fs)
    same = unit_stats(evaluate_morphology(varied, varied, fs, "belt", "test", config)["cycle_table"])
    band = unit_stats(evaluate_morphology(varied, training_target(varied, None, "steam2"), fs, "belt", "test",
                                          config)["cycle_table"])
    report("duty cycle r / SD ratio", abs(same["Duty_r"] - 1) < 1e-6 and abs(same["Duty_SD_ratio"] - 1) < 1e-6
           and band["Duty_SD_ratio"] < 0.8,
           f"identical: r {same['Duty_r']:.2f}, SD ratio {same['Duty_SD_ratio']:.2f}; training target: r "
           f"{band['Duty_r']:.2f}, SD ratio {band['Duty_SD_ratio']:.2f}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

RENAME = {"subject_id": "Subject_ID", "window_id": "Window_ID", "cycle_id": "Cycle_ID"}


def export_cycles(frame: pd.DataFrame) -> pd.DataFrame:
    """Cycle table with ref_ / item_ instead of gt_ / pred_ (the reference can
    be the training target)."""
    columns = {c: ("ref_" + c[3:] if c.startswith("gt_") else "item_" + c[5:] if c.startswith("pred_") else c)
               for c in frame.columns}
    return frame.rename(columns={**columns, **RENAME})


def write_excel(path: Path, sheets: Dict[str, pd.DataFrame]) -> None:
    try:
        with pd.ExcelWriter(path, engine="openpyxl") as writer:
            for name, frame in sheets.items():
                frame.to_excel(writer, index=False, sheet_name=name[:31])
    except Exception as exc:
        print(f"Excel export failed ({exc}); the CSV files are complete.")


def notes(args: argparse.Namespace) -> pd.DataFrame:
    lines = [
        ("RRWaveNet", f"261105 final ensemble ({args.root_261105}), training sign / lag undone, not shifted; extra "
                      f"low-pass {args.rrwavenet_lowpass_hz:g} Hz (0 = none)"),
        ("Training target", "BIDMC / STEAM2: the GT band-passed 0.1-0.5 Hz (Butterworth 3, zero phase); CapnoBase: "
                            "the label-phase sine of the expert breath labels (T_I/T_tot 0.5); as reference item: "
                            "the ceiling of a model reproducing its target"),
        ("Reference GT / Training target", "every item measured against the recorded GT; the three models also "
                                           "against the training target"),
        ("Time / polarity", "no waveform shifted; inverted once per recording when r < 0 with the GT"),
        ("Cycles", "261030 detector and matching, the same filter for all; clipped GT cycles not evaluated"),
        ("Duty cycle", "T_I / T_tot per cycle (belt / impedance: rising time; capnogram: low-CO2 time)"),
        ("Duty_r / Duty_SD_ratio", "Pearson r of reference and item T_I/T_tot over the matched cycles; SD of the "
                                   "item's T_I/T_tot / SD of the reference's"),
        ("E_shape", "1 - r of the two cycles, each from its own start to end resampled to 101 points, z-scored"),
        ("E_onset / B_onset", "|item cycle start - reference cycle start| (s) / signed"),
        ("Summary", "recordings: per subject the mean over the cycles matched by every item of the reference "
                    "(coverage, false cycles: own cycles); Mean ± SD (Median) of the subject means; no test"),
        ("Windows", f"window widened by {args.pad_sec:g} s on both sides inside the item's output; the cycles "
                    "starting in the window"),
    ]
    return pd.DataFrame(lines, columns=["Item", "Description"])


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="261106 breath morphology of 261105 RRWaveNet, RespDiff and the "
                                                 "deep corr-encoder")
    parser.add_argument("--mode", choices=["windows", "recordings", "both"], default="both")
    parser.add_argument("--datasets", nargs="+", choices=["capnobase", "bidmc", "steam2"],
                        default=["capnobase", "bidmc", "steam2"])
    parser.add_argument("--subjects", nargs="+", default=None, help="recordings mode: only these subjects")
    parser.add_argument("--windows-csv", type=Path, default=None)
    parser.add_argument("--windows-from-summary", type=Path, default=None,
                        help="Waveform_Summary_all_runs.xlsx of v2: every drawn window of 261105")
    parser.add_argument("--pad-sec", type=float, default=PAD_SEC)
    parser.add_argument("--min-run-sec", type=float, default=MIN_RUN_SEC)
    parser.add_argument("--rrwavenet-lowpass-hz", type=float, default=RRWAVENET_LOWPASS_HZ)
    parser.add_argument("--keep-clipped", action="store_true")
    parser.add_argument("--root-261105", type=Path, default=ROOT_261105)
    parser.add_argument("--respdiff-root", type=Path, default=RESPDIFF_ROOT)
    parser.add_argument("--respdiff-sampling", default=RESPDIFF_SAMPLING)
    parser.add_argument("--correncoder-root", type=Path, default=CORRENCODER_ROOT)
    parser.add_argument("--scripts-dir", type=Path, default=None)
    parser.add_argument("--output-root", type=Path, default=PROJECT_ROOT)
    parser.add_argument("--capnobase-path", type=Path, default=None)
    parser.add_argument("--bidmc-path", type=Path, default=None)
    parser.add_argument("--steam2-ppg-dir", type=Path, default=None)
    parser.add_argument("--steam2-rsp-dir", type=Path, default=None)
    parser.add_argument("--no-figures", action="store_true")
    parser.add_argument("--self-test", action="store_true", help="run the synthetic test cases and stop")
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> None:
    args = parse_args(argv)
    if args.self_test:
        print("Test cases (synthetic signals, 30 Hz):")
        sys.exit(0 if self_test() else 1)
    set_data_paths(args)
    out_root = Path(args.output_root)
    out_root.mkdir(parents=True, exist_ok=True)
    config = dict(CONFIG)
    windows = load_windows(args) if args.mode in ("windows", "both") else pd.DataFrame()
    window_tables, window_rows, cycle_tables, all_shapes = [], [], [], {}
    qc_rows, alignment_rows = [], []
    for dataset in DATASETS:
        key = DATASET_KEYS[dataset]
        if key not in args.datasets:
            continue
        if key == "capnobase" and not load_capnobase_reader(args):
            continue
        sources = dataset_sources(args, dataset)
        print(f"[{dataset}] reading the GT data files ...", flush=True)
        recordings = read_recordings(key)
        cache: Dict[str, Dict[str, dict]] = {}

        def values_of(subject: str, dataset=dataset, sources=sources, recordings=recordings, cache=cache):
            if subject not in cache:
                cache[subject] = subject_values(args, dataset, subject, recordings.get(subject, {}), sources,
                                                alignment_rows)
            return cache[subject]

        if args.mode in ("windows", "both"):
            mine = windows[windows["Dataset"] == dataset]
            if len(mine):
                print(f"[{dataset}] {len(mine)} windows", flush=True)
                tables, rows = window_analysis(args, dataset, mine, values_of, config, out_root, qc_rows)
                window_tables += tables
                window_rows += rows
        if args.mode in ("recordings", "both"):
            subjects = sorted(set(sources["files"]) & set(sources["corr"]) & set(sources["respdiff"])
                              & set(recordings))
            if args.subjects:
                subjects = [s for s in subjects if s in set(args.subjects)]
            print(f"[{dataset}] whole recordings of {len(subjects)} subjects with all three models", flush=True)
            tables, shapes = recording_analysis(args, dataset, subjects, values_of, config, qc_rows)
            cycle_tables += tables
            all_shapes.update({(dataset,) + k: v for k, v in shapes.items()})
        cache.clear()

    sheets = {}
    if window_rows:
        window_summary = pd.DataFrame(window_rows)
        (out_root / "Windows").mkdir(parents=True, exist_ok=True)
        window_summary.to_csv(out_root / "Windows" / "window_summary.csv", index=False, encoding="utf-8-sig")
        if window_tables:
            export_cycles(pd.concat(window_tables, ignore_index=True)).to_csv(
                out_root / "Windows" / "window_cycles.csv", index=False, encoding="utf-8-sig")
        sheets["Window_Summary"] = window_summary
    if cycle_tables:
        cycles = mark_common(pd.concat(cycle_tables, ignore_index=True))
        subjects = subject_table(cycles)
        summary = summary_table(subjects, cycles)
        rec_root = out_root / "Recordings"
        rec_root.mkdir(parents=True, exist_ok=True)
        export_cycles(cycles).to_csv(rec_root / "cycle_table.csv", index=False, encoding="utf-8-sig")
        subjects.to_csv(rec_root / "per_subject.csv", index=False, encoding="utf-8-sig")
        summary.to_csv(rec_root / "summary.csv", index=False, encoding="utf-8-sig")
        sheets = {"Summary_Recordings": summary[[c for c in summary.columns if not c.endswith(("_Mean", "_SD",
                                                                                              "_Median", "_N"))]],
                  "Summary_Recordings_Numbers": summary, "Per_Subject": subjects, **sheets}
        if not args.no_figures:
            for dataset in cycles["Dataset"].unique():
                recording_figures(cycles[cycles["Dataset"] == dataset],
                                  {k[1:]: v for k, v in all_shapes.items() if k[0] == dataset}, subjects, rec_root)
        print("\nWhole recordings, reference GT: mean ± SD (median) of the subject means")
        for _, row in summary[summary["Reference"] == "GT"].iterrows():
            print(f"  {row['Dataset']:<9s} {row['Item']:<18s} T_I/T_tot GT {row['ref_Ti_Ttot']} | item "
                  f"{row['item_Ti_Ttot']} | duty error {row['E_TiTtot']} | duty r {row['Duty_r']} | shape "
                  f"{row['E_shape']}")
    qc = pd.DataFrame(qc_rows)
    alignment = pd.DataFrame(alignment_rows)
    qc.to_csv(out_root / "qc.csv", index=False, encoding="utf-8-sig")
    alignment.to_csv(out_root / "alignment.csv", index=False, encoding="utf-8-sig")
    sheets.update({"QC": qc, "Alignment": alignment, "Notes": notes(args)})
    write_excel(out_root / "261106_RRWaveNet_Waveform_Morphology.xlsx", sheets)
    settings = {**config, "mode": args.mode, "pad_sec": args.pad_sec, "min_run_sec": args.min_run_sec,
                "rrwavenet_lowpass_hz": args.rrwavenet_lowpass_hz, "keep_clipped": args.keep_clipped,
                "target_band_hz": TARGET_BAND_HZ, "target_band_order": TARGET_BAND_ORDER,
                "root_261105": str(args.root_261105)}
    (out_root / "settings.json").write_text(json.dumps(settings, indent=2), encoding="utf-8")
    if window_rows:
        print("\nShowcase windows, reference GT (T_I/T_tot GT -> item, duty error, shape error):")
        for _, r in pd.DataFrame(window_rows).iterrows():
            if r.get("Reference") == "GT" and r.get("Status") == "ok":
                print(f"  {r['Dataset']:<9s} {r['Size_Sec']}s {r['Subject']:<8s} {r['Start_Sec']:.0f}-"
                      f"{r['End_Sec']:.0f}s {r['Item']:<18s} {r['ref_Ti_Ttot']:.2f} -> {r['item_Ti_Ttot']:.2f}  "
                      f"E {r['E_TiTtot']:.3f}  shape {r['E_shape']:.3f}  coverage {r['coverage']:.2f}")
    print(f"\nSaved under: {out_root}")


if __name__ == "__main__":
    main()
