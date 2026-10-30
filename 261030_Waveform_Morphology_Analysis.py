r"""261030_Waveform_Morphology_Analysis: respiratory waveform morphology of
RespDiff, the deep corr-encoder and RRWaveNet against the reference, per
breathing cycle, as specified in "PPG -> RSP 세부 호흡 파형 평가 지표 5종"
(GT-예측 파형 비교 명세, 2026-10-30): I:E ratio error, inspiratory /
expiratory duration error, cycle amplitude error, peak-slope asymmetry error
and waveform skewness error, with cycle matching, coverage and false cycles;
statistical comparison of the three models per dataset. Only stored results
are read; nothing is trained.

Models and stored results (the results of the scripts named)
-------------------------------------------------------------
  RespDiff           261020_RespDiff / 261028_RespDiff_v3, DDPM 50:
                     {root}\respdiff\{dataset}\Predictions\{subject}_ddpm50.npz
                     (joined and low-pass filtered 5 s outputs, 30 Hz)
  Deep corr-encoder  261028_Deep_correncoder:
                     {root}\{dataset}\Predictions\{subject}_outputs.npz
                     (9.6 s outputs shifted 1 s and averaged, 30 Hz)
  RRWaveNet          CapnoBase: 261007_RRWaveNet_Ensemble_NoEvent, full model,
                     10 s windows, ensemble of 3:
                     {root}\full\01_w10s_nonoverlap\CapnoBase\Ensemble_Predictions\
                     BIDMC, STEAM2: 261013_RRWaveNet_Adult_SubjectVal_WD, the
                     weight decay selected in every fold
                     (WD_Selection\01_w10s_nonoverlap\{dataset}\{subject}_selection.json
                     -> {root}\full_wd{wd}\01_w10s_nonoverlap\{dataset}\Ensemble_Predictions\);
                     10 s windows at 64 Hz with their segment and start
A fold counts only when its result file is complete (RespDiff
Fold_Results\{subject}_ddpm50.json, corr-encoder Fold_Results\
{subject}_summary.json, RRWaveNet Fold_Summaries\{subject}_ensemble_summary.json).
Subjects with all three models are analysed.

Reference (GT) and alignment
----------------------------
* GT: the reference waveform as recorded (--reference raw, default): CapnoBase
  the capnogram (co2), BIDMC the impedance pneumogram (ref.resp_sig.imp),
  STEAM2 the respiratory belt, read from the data files as the three scripts
  read them and resampled to 30 Hz (polyphase, as the deep corr-encoder), on
  the time axis of the model outputs. --reference stored uses instead the
  reference stored with the corr-encoder results (every 9.6 s window min-max
  scaled, shifted 1 s and averaged), when the data files are not at hand.
* Recordings: CapnoBase and BIDMC one per subject; STEAM2 every session piece
  (split at boundary events; the scripts read the sessions with the same
  reader): RespDiff by name ({session}_piece{k}), RRWaveNet by segment number
  (segment k = recording k of the corr-encoder).
* Time and polarity as stored, without searching a lag per measure (the
  specification): RRWaveNet's CapnoBase target was shifted by the capnogram
  lag (Target_Lag_Sec) and its BIDMC target multiplied by the sign of its
  zero-lag correlation with the PPG (Target_Sign); both are undone with the
  values of {root}\Data_Quality\01_w10s_nonoverlap\{dataset}_window_quality.csv
  (without that file: the BIDMC sign from the stored targets, the lag 0).
  RespDiff and the corr-encoder follow the recorded polarity.
* Clipped GT: the BIDMC impedance is stored scaled to 0-1 and sticks at 1 or
  0 for 0.1-1 s in many breaths (clipped at the range of the recorder; 44 of
  the 53 recordings). A clipped breath has no true peak, amplitude or shape,
  so every GT cycle that contains a clipped sample (runs >= 0.1 s exactly at
  the largest or smallest value of the recording, at its own rate, widened
  by 0.1 s; the capnogram only at its largest value) is left out for all
  models (match_status gt_saturated, qc.csv, GT_Clipped_Sec in
  alignment.csv): it counts neither as GT cycle nor any model cycle in its
  time (matched or not: in_gt_saturated) as model or false cycle. BIDMC:
  3481 of 7063 GT cycles (49 %); 27 recordings have half of their cycles or
  more clipped, 9 none. --keep-clipped evaluates them too (sensitivity
  check). Not detectable with --reference stored.
* Common stretches: every waveform is taken at the 30 Hz samples of the GT
  (RRWaveNet, 64 Hz: the nearest sample); only continuous stretches where the
  GT and all three models have output, at least 30 s long (--min-run-sec; the
  specification recommends the 30 s joined waveforms), are analysed.
  RRWaveNet scores only the 10 s windows that passed its quality check, so
  its gaps also remove those stretches for the other models: every model is
  evaluated on the same GT cycles (the specification).

Processing of every stretch (the same for the GT and every model)
------------------------------------------------------------------
1. Filter: linear trend removed, zero-phase Butterworth high-pass 0.03 Hz
   (order 2) and low-pass 2.0 Hz (capnogram, belt) or 1.0 Hz (impedance:
   its cardiac ripple, 1-2 Hz, would dominate the slopes), order 4
   (--highpass-hz, --lowpass-hz; --lowpass-hz 0 = no low-pass, for the
   sensitivity check of the specification). The same filter for the GT and
   every model, no other smoothing. Any smoothing moves the peaks of an
   asymmetric breath towards its flatter side (synthetic breaths: T_I
   +0.06 s at 2 Hz, +0.12 s at 1 Hz, for the GT and the models alike), so
   duration differences are slightly compressed, never created.
2. Polarity check: Pearson r of model and GT over the stretch; r <= -0.5
   (--polarity-min-r) is taken as an inverted waveform and the model is not
   compared there (qc.csv), never flipped.
3. Breathing cycles, by modality (the specification: one rule does not fit
   belt, impedance and CO2):
   * the waveform is scaled to its local range (5th-95th percentile in a
     moving 20 s window, --range-window-sec), and its crossings of 0.5 are
     found with hysteresis (it must pass 0.35 and 0.65 between crossings);
     the crossing times are interpolated;
   * belt, impedance (volume-like, inspiration = rising): one trough between
     every downward and the next upward crossing and one peak between every
     upward and the next downward crossing; cycle = trough -> peak (inspiration)
     -> next trough (expiration). STEAM2 and BIDMC are taken as recorded
     (rising = inspiration); --invert-bidmc / --invert-steam2 invert the GT and
     all models when the recording polarity is the opposite;
   * capnogram (CO2 rises in expiration): cycle = downward crossing (start of
     inspiration, CO2 falls) -> upward crossing (start of expiration, CO2
     rises) -> next downward crossing; T_I = time below half amplitude;
   * kept: complete cycles of 0.6-20 s (capnogram) or 1.5-20 s (belt,
     impedance) with both phases >= 0.2 s; the others are counted in qc.csv;
     the incomplete cycles at the ends of a stretch never form a cycle (also
     counted in qc.csv);
   * checked on the GT against the expert labels: CapnoBase 0009 and 0015,
     exactly the labelled breaths (151 of 151, 229 of 229); the transitions
     lie at half amplitude, 0.2-0.4 s after the labelled onsets of the CO2
     fall and rise (T_I by the onsets 0.87 s, at half amplitude 1.02 s in
     0015). BIDMC 01 and 02: the detected peaks lie 0.04-0.08 s (median)
     from the annotated breaths (ann1, ann2: impedance maxima), i.e. rising
     impedance = inspiration as recorded.
4. Matching: every GT cycle in order is matched to the model cycle whose
   start is nearest, within half the GT cycle length (--match-tolerance),
   keeping the order, each model cycle at most once (1:1). Unmatched GT
   cycles are missed breaths, unmatched model cycles false breaths; no
   model cycle is ever made up.
     cycle_coverage   = n_matched / n_gt_cycles (GT cycles of stretches with
                        inverted polarity count as not matched)
     false_cycle_rate = n_unmatched_pred / n_pred_cycles

Measures of every matched cycle (GT and model each on its own cycle)
--------------------------------------------------------------------
  01 I:E ratio        IE = T_I / T_E;           E_IE = |IE_gt - IE_pred|
  02 Phase durations  T_I, T_E (s);             E_TI, E_TE = |gt - pred|,
                                                B_TI, B_TE = pred - gt
  03 Cycle amplitude  A = max - min in the cycle; E_A = |A_gt - A_pred|
  04 Peak-slope ratio R = peak_I / peak_E;      E_slope = |R_gt - R_pred|
  05 Skewness         third standardized moment of the cycle's samples;
                                                E_skew = |skew_gt - skew_pred|
  added (as recommended in the evaluation of the specification):
     Ti_Ttot = T_I / (T_I + T_E), E_TiTtot (bounded 0-1, the cycle duty cycle);
     E_logIE = |log(IE_pred / IE_gt)| (the same error up and down);
     T_tot = T_I + T_E, E_Ttot and B_Ttot (separates the breath length, i.e.
     the rate error, from the I/E split).
* Amplitude needs a common scale (the specification). The model outputs are
  scaled per window in training (RespDiff 5 s, corr-encoder 9.6 s, RRWaveNet
  10 s), so an absolute amplitude cannot be compared: every waveform is
  standardized (z-score) over the stretch (--amplitude-scale zscore), and E_A
  is the error of the cycle amplitude in SDs of its stretch, i.e. of the
  relative amplitude pattern; a global gain difference is not measured.
  --amplitude-scale none compares the values as they are (only for waveforms
  on one scale).
* Slopes: derivative of the filtered, scaled waveform. Belt, impedance:
  peak_I = largest rise in the inspiration, peak_E = largest fall in the
  expiration. Capnogram: peak_I = largest fall of CO2 within +-0.25 T_tot of
  the start of inspiration, peak_E = largest rise within +-0.25 T_tot of the
  start of expiration (the transitions of the capnogram; they are changes of
  CO2, not airflow). The ratio is invalid when peak_E is ~0.
* Invalid values (T_E or a phase too short, slope denominator ~0, cycle SD
  ~0) are left out of that measure and counted in qc.csv.

Summaries and statistics
------------------------
* own set: every model on all its matched cycles (with its coverage and
  false cycle rate); common set: the GT cycles matched by all three models
  (and valid in the measure for all three), the paired comparison.
* Per subject (the LOSOCV unit): mean of every error over its cycles; per
  dataset: mean, SD and median of the subject means, and over cycles (mean,
  SD, median as an auxiliary value, N). Datasets are never pooled (different
  sensors).
* Statistics on the subject means of the common set (subjects with at least
  10 cycles, --min-subject-cycles): Friedman test of the three models
  (Kendall's W), pairwise Wilcoxon signed-rank tests (two-sided) with the
  median of the paired differences and the matched-pairs rank-biserial
  correlation, Benjamini-Hochberg over all tests of a dataset; the same for
  the coverage and the false cycle rate (own set). The waveform measures are
  exploratory and complementary to the rate and PCC results.

Interpretation (what the models were trained to reproduce)
-----------------------------------------------------------
RespDiff and the corr-encoder are trained on the reference waveform itself
(scaled per window). RRWaveNet is not: on CapnoBase its target is a sine wave
of the respiratory phase between the expert breath labels (261007,
label_phase_waveform), whose I:E is 1 and skewness 0 by construction; on BIDMC
and STEAM2 the reference band-passed to 0.1-0.5 Hz, which removes most of the
asymmetry of a breath. Its morphology errors therefore measure what is left of
the breath shape after a target that removed it by design.

Outputs (PROJECT_ROOT = D:\PPG2RSP_RRWaveNet_Inspired\261030_Waveform_Morphology_Analysis)
  cycle_table.csv   every GT cycle x model (match status, the times and every
                    feature of GT and model, the errors) and every false cycle
  summary.csv       per dataset, model and set (own / common): counts, coverage,
                    false cycle rate; every error as E_x = mean of the
                    subject means, E_x_SD, E_x_median (auxiliary) and over
                    cycles E_x_cycle_mean, _cycle_SD, _cycle_median, _N_cycles
  per_subject.csv   per dataset, subject, model and set
  statistics.csv    Friedman and pairwise Wilcoxon tests
  qc.csv            per dataset, subject and stretch: reason and count
  alignment.csv     RRWaveNet sign and lag per segment, analysed time
  261030_Waveform_Morphology_Analysis.xlsx   all tables but the cycle table
  Figures\{dataset}_IE_scatter.png, _subject_errors.png, _example.png

Run:                         python 261030_Waveform_Morphology_Analysis.py
Check the implementation:    python 261030_Waveform_Morphology_Analysis.py --self-test
                             (the test cases of the specification on synthetic
                             signals; run it before the real analysis)
Some datasets:               --datasets capnobase bidmc
Result folders:              --respdiff-root, --correncoder-root,
                             --rrwavenet-261007-root, --rrwavenet-261013-root,
                             --output-root
Data files (GT):             --capnobase-path, --bidmc-path, --steam2-ppg-dir,
                             --steam2-rsp-dir (default: the paths of the scripts)
"""

import argparse
import csv
import json
import math
import re
import sys
import warnings
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
from scipy import stats
from scipy.signal import butter, detrend, resample_poly, sosfiltfilt


# ---------------------------------------------------------------------------
# Result folders, data files and settings
# ---------------------------------------------------------------------------

PROJECT_ROOT = Path(r"D:\PPG2RSP_RRWaveNet_Inspired\261030_Waveform_Morphology_Analysis")
RESPDIFF_ROOT = Path(r"D:\PPG2RSP_RRWaveNet_Inspired\261020_RespDiff_v2")
CORRENCODER_ROOT = Path(r"D:\PPG2RSP_RRWaveNet_Inspired\261028_Deep_correncoder")
RRWAVENET_261007_ROOT = Path(r"D:\PPG2RSP_RRWaveNet_Inspired\261007_RRWaveNet_NoEvent_v1")
RRWAVENET_261013_ROOT = Path(r"D:\PPG2RSP_RRWaveNet_Inspired\261013_RRWavNet_STEAM_BIDMC_v1")

# The data files of the three scripts (GT read as recorded).
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

# Models in a fixed order (table order and figure colours).
MODELS = ("RespDiff", "Deep corr-encoder", "RRWaveNet")
MODEL_COLORS = {"RespDiff": "#2a78d6", "Deep corr-encoder": "#eb6834", "RRWaveNet": "#1baf7a"}
PAIRS = tuple((a, b) for i, a in enumerate(MODELS) for b in MODELS[i + 1:])
TEXT_PRIMARY = "#0b0b0b"
TEXT_SECONDARY = "#52514e"
GRID_COLOR = "#e4e3df"

FS = 30  # analysis rate: the 30 Hz of RespDiff, the corr-encoder and the GT
RESPDIFF_SAMPLING = "ddpm50"
RRWAVENET_FS = 64
RRWAVENET_WINDOW = "01_w10s_nonoverlap"

# Cycle detection and measures (the same for the GT and every model).
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
MIN_RUN_SEC = 30.0
MIN_SUBJECT_CYCLES = 10
EXAMPLE_SEC = 60.0

FEATURES = ("T_I", "T_E", "T_tot", "IE", "Ti_Ttot", "A", "peak_I", "peak_E", "slope_ratio", "skew")
# (column, label, unit) of every error measure; the first six are the five
# measures of the specification (02 has two).
ERRORS = (
    ("E_IE", "01 I:E ratio error |IE_gt - IE_pred|", ""),
    ("E_TI", "02 Inspiratory duration error", "s"),
    ("E_TE", "02 Expiratory duration error", "s"),
    ("E_A", "03 Cycle amplitude error", "SD"),
    ("E_slope", "04 Peak-slope ratio error", ""),
    ("E_skew", "05 Skewness error", ""),
    ("B_TI", "02 Inspiratory duration bias (pred - gt)", "s"),
    ("B_TE", "02 Expiratory duration bias (pred - gt)", "s"),
    ("E_TiTtot", "added: T_I/T_tot error", ""),
    ("E_logIE", "added: |log(IE_pred / IE_gt)|", ""),
    ("E_Ttot", "added: breath length error", "s"),
    ("B_Ttot", "added: breath length bias (pred - gt)", "s"),
)
ERROR_COLUMNS = [e[0] for e in ERRORS]
# GT cycles of the evaluation (not: clipped GT cycles, false model cycles)
EVALUATED = ("matched", "missed", "polarity_refused")
TESTED_ERRORS = ("E_IE", "E_TI", "E_TE", "E_A", "E_slope", "E_skew", "E_TiTtot", "E_logIE", "E_Ttot")


# ---------------------------------------------------------------------------
# Data file readers of the three scripts (CapnoBase CSV, BIDMC .mat, STEAM2
# EEGLAB .mat), as in 261028_Deep_correncoder
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


# ---------------------------------------------------------------------------
# Stored results of the three models (as 261029_DutyCycle_Analysis)
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# Morphology of one GT / model pair (interface of the specification)
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
        "E_TiTtot": abs(g["Ti_Ttot"] - p["Ti_Ttot"]),
        "E_logIE": abs(math.log(p["IE"] / g["IE"])) if g["IE"] > 0 and p["IE"] > 0 else math.nan,
        "E_Ttot": abs(g["T_tot"] - p["T_tot"]), "B_Ttot": p["T_tot"] - g["T_tot"],
    }
    return {k: (float(v) if v is not None and np.isfinite(v) else math.nan) for k, v in out.items()}


def evaluate_morphology(gt: np.ndarray, pred: np.ndarray, fs: float, modality: str, subject_id: str,
                        config: dict) -> dict:
    """The interface of the specification: GT and model waveform of one
    continuous stretch (same rate and time axis) -> {cycle_table, summary,
    qc}. config may carry 'window_id', 'time_offset_sec' (start of the
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
    rows = []
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
    return {"cycle_table": table, "summary": summary, "qc": qc}


# ---------------------------------------------------------------------------
# All recordings of a dataset
# ---------------------------------------------------------------------------

def finite_runs(valid: np.ndarray, min_length: int) -> List[Tuple[int, int]]:
    """Continuous stretches [start, end) where valid is True."""
    edges = np.diff(np.concatenate([[0], valid.astype(int), [0]]))
    starts, ends = np.flatnonzero(edges == 1), np.flatnonzero(edges == -1)
    return [(int(s), int(e)) for s, e in zip(starts, ends) if e - s >= min_length]


def dataset_morphology(args: argparse.Namespace, dataset_key: str, config: dict):
    label = DATASET_CONFIGS[dataset_key]["label"]
    modality = MODALITIES[dataset_key]
    invert = (dataset_key == "bidmc" and args.invert_bidmc) or (dataset_key == "steam2" and args.invert_steam2)
    corr = load_correncoder(Path(args.correncoder_root), label)
    respdiff = load_respdiff(Path(args.respdiff_root), label, args.respdiff_sampling)
    rr_files, rr_root = rrwavenet_files(dataset_key, Path(args.rrwavenet_261007_root), Path(args.rrwavenet_261013_root))
    alignment_table = rrwavenet_alignment_table(rr_root, label)
    found = {"RespDiff": set(respdiff), "Deep corr-encoder": set(corr), "RRWaveNet": set(rr_files)}
    subjects = sorted(set.intersection(*found.values()))
    if args.subjects:
        subjects = [s for s in subjects if s in set(args.subjects)]
    print(f"{label} ({modality}): subjects with results - RespDiff {len(respdiff)}, corr-encoder {len(corr)}, "
          f"RRWaveNet {len(rr_files)}; all three {len(subjects)}", flush=True)
    qc_rows, alignment_rows, tables, summaries, examples = [], [], [], [], []
    for subject in sorted(set.union(*found.values())):
        missing = [m for m in MODELS if subject not in found[m]]
        if missing:
            qc_rows.append({"dataset": label, "subject_id": subject, "window_id": "",
                            "reason": "no result of " + ", ".join(missing), "count": 1})
    raw = {}
    if subjects and args.reference == "raw":
        print(f"[{label}] reading the GT from the data files ...", flush=True)
        try:
            raw = reference_recordings(dataset_key, config)
        except Exception as exc:
            print(f"[{label}] the data files could not be read ({exc}); the stored corr-encoder reference is used")
            qc_rows.append({"dataset": label, "subject_id": "", "window_id": "",
                            "reason": f"GT data files not read ({exc}); stored reference used", "count": 1})
    min_run = int(round(args.min_run_sec * FS))
    for subject in subjects:
        rr_segments = load_rrwavenet(rr_files[subject][0])
        for recording in corr[subject]:
            name = recording["name"]
            stored = recording["reference"]
            gt_values, gt_source = stored, "stored corr-encoder reference"
            clipped = np.zeros(len(stored), dtype=bool)  # not detectable in the stored reference
            if args.reference == "raw" and raw:
                if name in raw.get(subject, {}):
                    (gt_values, clipped), gt_source = raw[subject][name], "data file"
                else:
                    qc_rows.append({"dataset": label, "subject_id": subject, "window_id": name,
                                    "reason": "recording not in the data files read; stored reference used",
                                    "count": 1})
            gt = Track(gt_values, FS)
            tracks = {"Deep corr-encoder": Track(recording["output"], FS)}
            if name in respdiff[subject]:
                tracks["RespDiff"] = Track(respdiff[subject][name], FS)
            segment = rr_segments.get(recording["index"])
            info = {"Dataset": label, "Subject": subject, "Recording": name, "Segment": recording["index"],
                    "GT": gt_source, "GT_Sec": len(gt_values) / FS, "GT_Clipped_Sec": float(clipped.sum() / FS),
                    "RRWaveNet_Run": rr_files[subject][1]}
            if segment is not None:
                stored_alignment = alignment_table.get((subject, recording["index"]))
                lag = stored_alignment["lag_sec"] if stored_alignment else 0.0
                if stored_alignment:
                    sign, source = stored_alignment["sign"], "data quality table"
                elif dataset_key == "bidmc":
                    r = target_reference_r(Track(segment["target"], RRWAVENET_FS, lag), Track(stored, FS))
                    sign, source = (-1.0 if r < 0 else 1.0), "correlation of stored target and reference"
                else:
                    sign, source = 1.0, "recorded polarity (no data quality table)"
                tracks["RRWaveNet"] = Track(sign * segment["output"], RRWAVENET_FS, lag)
                info.update({"RRWaveNet_Sign": sign, "RRWaveNet_Lag_Sec": lag, "Alignment_Source": source})
            absent = [m for m in MODELS if m not in tracks]
            times = np.arange(len(gt_values)) / FS
            values = {"GT": gt.at(times), **{m: tracks[m].at(times) for m in tracks}}
            if invert:
                values = {k: -v for k, v in values.items()}
            valid = np.isfinite(values["GT"])
            for model in MODELS:
                valid &= np.isfinite(values[model]) if model in values else False
            runs = finite_runs(valid, min_run)
            info.update({"Analysed_Sec": float(sum(e - s for s, e in runs) / FS), "Stretches": len(runs)})
            alignment_rows.append(info)
            if absent:
                qc_rows.append({"dataset": label, "subject_id": subject, "window_id": name,
                                "reason": "no output of " + ", ".join(absent), "count": 1})
                continue
            if not runs:
                qc_rows.append({"dataset": label, "subject_id": subject, "window_id": name,
                                "reason": f"no common stretch of {args.min_run_sec:g} s", "count": 1})
                continue
            for k, (s, e) in enumerate(runs):
                window_id = f"{name}#{k + 1}"
                for model in MODELS:
                    result = evaluate_morphology(values["GT"][s:e], values[model][s:e], FS, modality, subject,
                                                 {**config, "window_id": window_id, "time_offset_sec": s / FS,
                                                  "gt_saturated": None if args.keep_clipped else clipped[s:e]})
                    table = result["cycle_table"]
                    if not table.empty:
                        table.insert(0, "model", model)
                        table.insert(0, "dataset", label)
                        tables.append(table)
                    summaries.append({"dataset": label, "model": model, **result["summary"]})
                    for reason, count in result["qc"].items():
                        qc_rows.append({"dataset": label, "subject_id": subject, "window_id": window_id,
                                        "model": model, "reason": reason, "count": count})
                if len(examples) < 1 and e - s >= int(EXAMPLE_SEC * FS):
                    examples.append({"dataset": label, "subject": subject, "window_id": window_id, "start": s,
                                     "values": {k: v[s:s + int(EXAMPLE_SEC * FS)] for k, v in values.items()}})
    cycles = pd.concat(tables, ignore_index=True) if tables else pd.DataFrame()
    return cycles, pd.DataFrame(summaries), pd.DataFrame(qc_rows), pd.DataFrame(alignment_rows), examples


# ---------------------------------------------------------------------------
# Summaries and statistics
# ---------------------------------------------------------------------------

def mark_common(cycles: pd.DataFrame) -> pd.DataFrame:
    """common = the GT cycle is matched by all three models."""
    cycles = cycles.copy()
    matched_rows = cycles["match_status"] == "matched"
    key = ["dataset", "subject_id", "window_id", "cycle_id"]
    matched = cycles[matched_rows].groupby(key)["model"].nunique()
    common_keys = set(matched[matched == len(MODELS)].index)
    cycles["common"] = [m and tuple(k) in common_keys
                        for m, k in zip(matched_rows, cycles[key].itertuples(index=False, name=None))]
    return cycles


def subject_table(cycles: pd.DataFrame) -> pd.DataFrame:
    """Per dataset, subject, model and set: cycle counts, coverage, false
    cycle rate and the mean of every error (common set: the cycles valid in
    that measure for all three models)."""
    rows = []
    key = ["dataset", "subject_id", "window_id", "cycle_id"]
    for (dataset, subject, model), part in cycles.groupby(["dataset", "subject_id", "model"], sort=False):
        gt = part[part["match_status"].isin(EVALUATED)]
        matched = gt[gt["match_status"] == "matched"]
        n_pred = int(len(matched) + (part["match_status"] == "false").sum())
        own = {"dataset": dataset, "subject_id": subject, "model": model, "set": "own",
               "n_gt_cycles": int(len(gt)), "n_refused": int((gt["match_status"] == "polarity_refused").sum()),
               "n_gt_saturated": int((part["match_status"] == "gt_saturated").sum()),
               "n_pred_cycles": n_pred, "n_matched": int(len(matched)),
               "coverage": len(matched) / len(gt) if len(gt) else math.nan,
               "false_cycle_rate": (n_pred - len(matched)) / n_pred if n_pred else math.nan}
        for column in ERROR_COLUMNS:
            own[column] = float(matched[column].mean()) if len(matched) else math.nan
            own[f"N_{column}"] = int(matched[column].notna().sum())
        rows.append(own)
    common = cycles[cycles["common"]]
    for (dataset, subject), part in common.groupby(["dataset", "subject_id"], sort=False):
        for model in MODELS:
            rows.append({"dataset": dataset, "subject_id": subject, "model": model, "set": "common",
                         "n_matched": int((part["model"] == model).sum())})
        for column in ERROR_COLUMNS:
            wide = part.pivot_table(index=key, columns="model", values=column, aggfunc="first")
            wide = wide.reindex(columns=list(MODELS)).dropna()
            for row in rows[-len(MODELS):]:
                row[column] = float(wide[row["model"]].mean()) if len(wide) else math.nan
                row[f"N_{column}"] = int(len(wide))
    return pd.DataFrame(rows)


def sd(values) -> float:
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    return float(values.std(ddof=1)) if len(values) > 1 else math.nan


def summary_table(cycles: pd.DataFrame, subjects: pd.DataFrame) -> pd.DataFrame:
    rows = []
    key = ["dataset", "subject_id", "window_id", "cycle_id"]
    for dataset in cycles["dataset"].unique():
        for cycle_set in ("own", "common"):
            for model in MODELS:
                sub = subjects[(subjects["dataset"] == dataset) & (subjects["model"] == model)
                               & (subjects["set"] == cycle_set)]
                part = cycles[(cycles["dataset"] == dataset) & (cycles["model"] == model)]
                row = {"dataset": dataset, "model": model, "set": cycle_set, "n_subjects": int(len(sub))}
                if cycle_set == "own":
                    gt = part[part["match_status"].isin(EVALUATED)]
                    matched = gt[gt["match_status"] == "matched"]
                    n_pred = int(len(matched) + (part["match_status"] == "false").sum())
                    row.update({"n_gt_cycles": int(len(gt)), "n_pred_cycles": n_pred, "n_matched": int(len(matched)),
                                "n_refused": int((gt["match_status"] == "polarity_refused").sum()),
                                "n_gt_saturated": int((part["match_status"] == "gt_saturated").sum()),
                                "coverage": len(matched) / len(gt) if len(gt) else math.nan,
                                "false_cycle_rate": (n_pred - len(matched)) / n_pred if n_pred else math.nan,
                                "coverage_subject_mean": float(sub["coverage"].mean()),
                                "coverage_subject_SD": sd(sub["coverage"]),
                                "false_cycle_rate_subject_mean": float(sub["false_cycle_rate"].mean()),
                                "false_cycle_rate_subject_SD": sd(sub["false_cycle_rate"])})
                    pool = matched
                else:
                    pool = cycles[(cycles["dataset"] == dataset) & cycles["common"]]
                    row["n_matched"] = int((pool["model"] == model).sum())
                for column in ERROR_COLUMNS:
                    if cycle_set == "own":
                        values = pool[column].to_numpy(float)
                    else:
                        wide = pool.pivot_table(index=key, columns="model", values=column, aggfunc="first")
                        wide = wide.reindex(columns=list(MODELS)).dropna()
                        values = wide[model].to_numpy(float) if len(wide) else np.array([])
                    values = values[np.isfinite(values)]
                    means = sub[column].to_numpy(float) if column in sub else np.array([])
                    means = means[np.isfinite(means)]
                    row[column] = float(means.mean()) if len(means) else math.nan
                    row[f"{column}_SD"] = sd(means)
                    row[f"{column}_median"] = float(np.median(means)) if len(means) else math.nan
                    row[f"{column}_cycle_mean"] = float(values.mean()) if len(values) else math.nan
                    row[f"{column}_cycle_SD"] = sd(values)
                    row[f"{column}_cycle_median"] = float(np.median(values)) if len(values) else math.nan
                    row[f"{column}_N_cycles"] = int(len(values))
                rows.append(row)
    return pd.DataFrame(rows)


def benjamini_hochberg(p_values: Sequence[float]) -> np.ndarray:
    p = np.asarray(p_values, dtype=float)
    out = np.full(p.shape, np.nan)
    finite = np.flatnonzero(np.isfinite(p))
    if not len(finite):
        return out
    order = finite[np.argsort(p[finite])]
    adjusted = p[order] * len(order) / np.arange(1, len(order) + 1)
    out[order] = np.clip(np.minimum.accumulate(adjusted[::-1])[::-1], 0.0, 1.0)
    return out


def paired_test(a: np.ndarray, b: np.ndarray) -> dict:
    difference = a - b
    nonzero = difference[difference != 0]
    out = {"N": int(len(a)), "Median_A": float(np.median(a)), "Median_B": float(np.median(b)),
           "Median_Difference_A_minus_B": float(np.median(difference)), "Statistic": math.nan, "p": math.nan,
           "Rank_Biserial_A_minus_B": math.nan}
    if len(nonzero):
        ranks = stats.rankdata(np.abs(nonzero))
        out["Rank_Biserial_A_minus_B"] = float((ranks[nonzero > 0].sum() - ranks[nonzero < 0].sum()) / ranks.sum())
        if len(a) >= 3:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                test = stats.wilcoxon(a, b)
            out["Statistic"], out["p"] = float(test.statistic), float(test.pvalue)
    elif len(a):
        out["p"] = 1.0
    return out


def statistics_table(subjects: pd.DataFrame, min_cycles: int) -> pd.DataFrame:
    """Tests on the subject means of the subjects with at least min_cycles
    cycles in that measure (common set) or GT cycles (coverage, false cycle
    rate) for every model."""
    rows = []
    for dataset in subjects["dataset"].unique():
        family = []
        measures = [(m, "common") for m in TESTED_ERRORS] + [("coverage", "own"), ("false_cycle_rate", "own")]
        for measure, cycle_set in measures:
            part = subjects[(subjects["dataset"] == dataset) & (subjects["set"] == cycle_set)]
            if measure not in part:
                continue
            count = f"N_{measure}" if cycle_set == "common" else "n_gt_cycles"
            part = part[part[count] >= min_cycles]
            wide = part.pivot_table(index="subject_id", columns="model", values=measure, aggfunc="first")
            wide = wide.reindex(columns=list(MODELS)).dropna()
            base = {"dataset": dataset, "measure": measure, "set": cycle_set, "unit": "subject mean"}
            friedman = {**base, "test": "Friedman", "comparison": " vs ".join(MODELS), "N": int(len(wide))}
            if len(wide) >= 3:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    try:
                        chi2, p = stats.friedmanchisquare(*[wide[m].to_numpy(float) for m in MODELS])
                        friedman.update({"Statistic": float(chi2), "p": float(p),
                                         "Kendall_W": float(chi2 / (len(wide) * (len(MODELS) - 1)))})
                    except ValueError:
                        pass
            rows.append(friedman)
            for a, b in PAIRS:
                result = paired_test(wide[a].to_numpy(float), wide[b].to_numpy(float)) if len(wide) else {"N": 0}
                family.append({**base, "test": "Wilcoxon signed-rank", "comparison": f"{a} vs {b}", **result})
        for row, q in zip(family, benjamini_hochberg([r.get("p", math.nan) for r in family])):
            row["p_BH"] = q
            rows.append(row)
    columns = ["dataset", "measure", "set", "unit", "test", "comparison", "N", "Statistic", "p", "p_BH", "Kendall_W",
               "Median_A", "Median_B", "Median_Difference_A_minus_B", "Rank_Biserial_A_minus_B"]
    frame = pd.DataFrame(rows)
    return frame.reindex(columns=columns)


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------

def style_axis(axis) -> None:
    axis.grid(True, color=GRID_COLOR, linewidth=0.8)
    axis.set_axisbelow(True)
    for side in ("top", "right"):
        axis.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        axis.spines[side].set_color(TEXT_SECONDARY)
    axis.tick_params(colors=TEXT_SECONDARY, labelsize=8)


def figures(cycles: pd.DataFrame, subjects: pd.DataFrame, statistics: pd.DataFrame, examples: List[dict],
            root: Path) -> None:
    root.mkdir(parents=True, exist_ok=True)
    for dataset in cycles["dataset"].unique():
        common = cycles[(cycles["dataset"] == dataset) & cycles["common"]]
        if not common.empty:
            figure, axes = plt.subplots(1, len(MODELS), figsize=(12, 4.2), sharex=True, sharey=True)
            values = common[["gt_IE", "pred_IE"]].to_numpy(float)
            values = values[np.isfinite(values) & (values > 0)]
            low, high = (np.percentile(values, 1) * 0.8, np.percentile(values, 99) * 1.25) if len(values) else (0.2, 5)
            for axis, model in zip(axes, MODELS):
                style_axis(axis)
                part = common[common["model"] == model]
                axis.plot([low, high], [low, high], color=TEXT_SECONDARY, linewidth=1, linestyle="--")
                axis.scatter(part["gt_IE"], part["pred_IE"], s=8, alpha=0.4, color=MODEL_COLORS[model],
                             edgecolors="none")
                axis.set_xscale("log")
                axis.set_yscale("log")
                axis.set_xlim(low, high)
                axis.set_ylim(low, high)
                r = pearson(np.log(part["gt_IE"].clip(lower=1e-6)), np.log(part["pred_IE"].clip(lower=1e-6)))
                axis.set_title(f"{model}\nE_IE = {part['E_IE'].mean():.3f}, r(log) = {r:.2f}", color=TEXT_PRIMARY,
                               fontsize=9)
                axis.set_xlabel("GT I:E (T_I / T_E)", color=TEXT_PRIMARY, fontsize=8)
            axes[0].set_ylabel("Model I:E", color=TEXT_PRIMARY, fontsize=8)
            figure.suptitle(f"{dataset}: I:E per breathing cycle (cycles matched by all three models, "
                            f"n = {len(common) // len(MODELS)})", color=TEXT_PRIMARY, fontsize=10)
            figure.tight_layout()
            figure.savefig(root / f"{dataset}_IE_scatter.png", dpi=150)
            plt.close(figure)
        panels = [("E_IE", "common"), ("E_TI", "common"), ("E_TE", "common"), ("E_A", "common"),
                  ("E_slope", "common"), ("E_skew", "common"), ("coverage", "own"), ("false_cycle_rate", "own")]
        figure, axes = plt.subplots(2, 4, figsize=(14, 6.6))
        for axis, (measure, cycle_set) in zip(axes.ravel(), panels):
            style_axis(axis)
            part = subjects[(subjects["dataset"] == dataset) & (subjects["set"] == cycle_set)]
            if measure not in part:
                axis.set_visible(False)
                continue
            wide = part.pivot_table(index="subject_id", columns="model", values=measure, aggfunc="first")
            wide = wide.reindex(columns=list(MODELS)).dropna()
            if wide.empty:
                axis.set_visible(False)
                continue
            positions = np.arange(1, len(MODELS) + 1)
            for _, row in wide.iterrows():
                axis.plot(positions, row.to_numpy(float), color=GRID_COLOR, linewidth=0.7, zorder=1)
            box = axis.boxplot([wide[m] for m in MODELS], positions=positions, widths=0.5, patch_artist=True,
                               showfliers=False, medianprops={"color": TEXT_PRIMARY, "linewidth": 1.3})
            for patch, model in zip(box["boxes"], MODELS):
                patch.set_facecolor(MODEL_COLORS[model])
                patch.set_alpha(0.35)
            rng = np.random.default_rng(0)
            for position, model in zip(positions, MODELS):
                axis.scatter(position + rng.uniform(-0.08, 0.08, len(wide)), wide[model], s=8,
                             color=MODEL_COLORS[model], zorder=3, edgecolors="white", linewidths=0.4)
            axis.set_xticks(positions)
            axis.set_xticklabels(["RespDiff", "Corr-enc.", "RRWaveNet"], color=TEXT_PRIMARY, fontsize=8)
            test = statistics[(statistics["dataset"] == dataset) & (statistics["measure"] == measure)
                              & (statistics["test"] == "Friedman")]
            p = test["p"].iloc[0] if len(test) else math.nan
            label = dict((e[0], e[1]) for e in ERRORS).get(measure, measure.replace("_", " "))
            axis.set_title(f"{label}\nFriedman p = {p:.2g}" if np.isfinite(p) else label, color=TEXT_PRIMARY,
                           fontsize=8)
        figure.suptitle(f"{dataset}: subject means (errors: common cycles; coverage, false cycles: own cycles)",
                        color=TEXT_PRIMARY, fontsize=10)
        figure.tight_layout()
        figure.savefig(root / f"{dataset}_subject_errors.png", dpi=150)
        plt.close(figure)
    for example in examples:
        figure, axis = plt.subplots(figsize=(12, 4.6))
        style_axis(axis)
        t = example["start"] / FS + np.arange(len(example["values"]["GT"])) / FS
        names = ["GT"] + list(MODELS)
        for offset, name in enumerate(names):
            y = example["values"][name]
            y = (y - np.mean(y)) / (np.std(y) or 1.0)
            color = TEXT_PRIMARY if name == "GT" else MODEL_COLORS[name]
            axis.plot(t, y - 4.0 * offset, color=color, linewidth=1.2)
        part = cycles[(cycles["dataset"] == example["dataset"]) & (cycles["subject_id"] == example["subject"])
                      & (cycles["window_id"] == example["window_id"]) & (cycles["model"] == MODELS[0])
                      & cycles["gt_t_start"].notna()]
        for _, row in part.iterrows():
            if t[0] <= row["gt_t_start"] <= t[-1]:
                axis.axvline(row["gt_t_start"], color=TEXT_SECONDARY, linewidth=0.6, linestyle="--")
            if t[0] <= row["gt_t_transition"] <= t[-1]:
                axis.axvline(row["gt_t_transition"], color=GRID_COLOR, linewidth=0.6)
        axis.set_yticks([-4.0 * k for k in range(len(names))])
        axis.set_yticklabels(names, color=TEXT_PRIMARY, fontsize=8)
        axis.set_xlabel("Time in the recording (s)", color=TEXT_PRIMARY, fontsize=8)
        axis.set_title(f"{example['dataset']} {example['subject']} {example['window_id']}: waveforms (z-scored) and "
                       "GT cycles (dashed: start, light: inspiration/expiration transition)", color=TEXT_PRIMARY,
                       fontsize=9)
        figure.tight_layout()
        figure.savefig(root / f"{example['dataset']}_example.png", dpi=150)
        plt.close(figure)


# ---------------------------------------------------------------------------
# Test cases of the specification (synthetic signals)
# ---------------------------------------------------------------------------

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
    ok = all(results)
    print(f"\nSelf-test: {sum(results)} of {len(results)} cases passed" + ("" if ok else " - do not use the results"))
    return ok


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def write_excel(path: Path, sheets: Dict[str, pd.DataFrame]) -> None:
    try:
        with pd.ExcelWriter(path, engine="openpyxl") as writer:
            for name, frame in sheets.items():
                frame.to_excel(writer, index=False, sheet_name=name[:31])
    except Exception as exc:
        print(f"Excel export failed ({exc}); the CSV files are complete.")


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--datasets", nargs="+", choices=list(DATASET_ORDER), default=list(DATASET_ORDER))
    parser.add_argument("--subjects", nargs="+", default=None, help="Only these subjects (quick tests).")
    parser.add_argument("--reference", choices=["raw", "stored"], default="raw",
                        help="GT: the reference of the data files (raw) or the one stored with the corr-encoder "
                             "results (stored).")
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
    parser.add_argument("--amplitude-scale", choices=["zscore", "none"], default=CONFIG["amplitude_scale"])
    parser.add_argument("--highpass-hz", type=float, default=CONFIG["highpass_hz"])
    parser.add_argument("--lowpass-hz", type=float, default=None,
                        help="One low-pass edge for every modality, 0 = none (default: capnogram and belt 2.0, "
                             "impedance 1.0).")
    parser.add_argument("--range-window-sec", type=float, default=CONFIG["range_window_sec"])
    parser.add_argument("--match-tolerance", type=float, default=CONFIG["match_tolerance"])
    parser.add_argument("--polarity-min-r", type=float, default=CONFIG["polarity_min_r"])
    parser.add_argument("--min-run-sec", type=float, default=MIN_RUN_SEC)
    parser.add_argument("--min-subject-cycles", type=int, default=MIN_SUBJECT_CYCLES,
                        help="Subjects with fewer cycles are left out of the statistical tests.")
    parser.add_argument("--keep-clipped", action="store_true",
                        help="Evaluate the GT cycles with clipped samples too (sensitivity check).")
    parser.add_argument("--invert-bidmc", action="store_true",
                        help="Invert the GT and all models of BIDMC (when the impedance falls in inspiration).")
    parser.add_argument("--invert-steam2", action="store_true",
                        help="Invert the GT and all models of STEAM2 (when the belt signal falls in inspiration).")
    parser.add_argument("--no-figures", action="store_true")
    parser.add_argument("--self-test", action="store_true", help="Run the test cases of the specification and stop.")
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> None:
    args = parse_args(argv)
    if args.self_test:
        print("Test cases of the specification (synthetic signals, 30 Hz):")
        sys.exit(0 if self_test() else 1)
    if args.capnobase_path:
        DATASET_CONFIGS["capnobase"]["data_path"] = str(args.capnobase_path)
    if args.bidmc_path:
        DATASET_CONFIGS["bidmc"]["data_path"] = str(args.bidmc_path)
    if args.steam2_ppg_dir:
        DATASET_CONFIGS["steam2"]["ppg_dir"] = str(args.steam2_ppg_dir)
    if args.steam2_rsp_dir:
        DATASET_CONFIGS["steam2"]["rsp_dir"] = str(args.steam2_rsp_dir)
    config = dict(CONFIG)
    config.update({"amplitude_scale": args.amplitude_scale, "highpass_hz": args.highpass_hz,
                   "range_window_sec": args.range_window_sec, "match_tolerance": args.match_tolerance,
                   "polarity_min_r": args.polarity_min_r})
    if args.lowpass_hz is not None:
        config["lowpass_hz"] = {key: args.lowpass_hz for key in CONFIG["lowpass_hz"]}
    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    frames = {"cycles": [], "summaries": [], "qc": [], "alignment": []}
    examples = []
    for dataset_key in [key for key in DATASET_ORDER if key in args.datasets]:
        cycles, summaries, qc, alignment, example = dataset_morphology(args, dataset_key, config)
        for name, frame in (("cycles", cycles), ("summaries", summaries), ("qc", qc), ("alignment", alignment)):
            if not frame.empty:
                frames[name].append(frame)
        examples += example
    concat = {name: pd.concat(parts, ignore_index=True) if parts else pd.DataFrame() for name, parts in frames.items()}
    concat["qc"].to_csv(output_root / "qc.csv", index=False, encoding="utf-8-sig")
    concat["alignment"].to_csv(output_root / "alignment.csv", index=False, encoding="utf-8-sig")
    if concat["cycles"].empty:
        print("No breathing cycle analysed; see qc.csv.")
        return
    cycles = mark_common(concat["cycles"])
    order = (["dataset", "model", "subject_id", "window_id", "cycle_id", "match_status", "common"]
             + [f"{who}_t_{when}" for who in ("gt", "pred") for when in ("start", "transition", "end")]
             + [f"{who}_{k}" for k in FEATURES for who in ("gt", "pred")] + ERROR_COLUMNS)
    cycles = cycles.reindex(columns=order + [c for c in cycles.columns if c not in order])
    subjects = subject_table(cycles)
    summary = summary_table(cycles, subjects)
    statistics = statistics_table(subjects, args.min_subject_cycles)
    cycles.to_csv(output_root / "cycle_table.csv", index=False, encoding="utf-8-sig")
    summary.to_csv(output_root / "summary.csv", index=False, encoding="utf-8-sig")
    subjects.to_csv(output_root / "per_subject.csv", index=False, encoding="utf-8-sig")
    statistics.to_csv(output_root / "statistics.csv", index=False, encoding="utf-8-sig")
    settings = {**config, "reference": args.reference, "min_run_sec": args.min_run_sec,
                "min_subject_cycles": args.min_subject_cycles, "keep_clipped": args.keep_clipped,
                "invert_bidmc": args.invert_bidmc, "invert_steam2": args.invert_steam2}
    (output_root / "settings.json").write_text(json.dumps(settings, indent=2), encoding="utf-8")
    write_excel(output_root / "261030_Waveform_Morphology_Analysis.xlsx",
                {"Summary": summary, "Statistics": statistics, "Per_subject": subjects, "QC": concat["qc"],
                 "Alignment": concat["alignment"]})
    if not args.no_figures:
        figures(cycles, subjects, statistics, examples, output_root / "Figures")

    print("\nWaveform morphology (errors: subject means over the cycles matched by all three models):")
    for dataset in summary["dataset"].unique():
        own = summary[(summary["dataset"] == dataset) & (summary["set"] == "own")]
        common = summary[(summary["dataset"] == dataset) & (summary["set"] == "common")]
        print(f"\n  {dataset}: {int(own['n_gt_cycles'].max())} GT cycles, {int(common['n_matched'].max())} matched "
              "by all three models")
        for (_, o), (_, c) in zip(own.iterrows(), common.iterrows()):
            print(f"    {o['model']:<18s} coverage {o['coverage']:.2f} false {o['false_cycle_rate']:.2f} | "
                  + "  ".join(f"{col} {c[col]:.3g}" for col in
                              ("E_IE", "E_TI", "E_TE", "E_A", "E_slope", "E_skew")))
        tests = statistics[(statistics["dataset"] == dataset) & (statistics["test"] == "Friedman")]
        print("    Friedman p: " + ", ".join(f"{r['measure']} {r['p']:.2g}" for _, r in tests.iterrows()
                                            if np.isfinite(r["p"])))
    print(f"\nSaved under: {output_root}")


if __name__ == "__main__":
    main()
