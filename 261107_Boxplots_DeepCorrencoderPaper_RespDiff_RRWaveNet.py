# -*- coding: utf-8 -*-
r"""261107_Boxplots_DeepCorrencoderPaper_RespDiff_RRWaveNet: the three models
side by side, waveforms and box plots in one run, with the subjects that had
an MAE issue left out:
    RRWaveNet           CapnoBase 261007, BIDMC 261013, STEAM2 261013
    Deep corr-encoder   261107_DeepCorrencoder_PaperVersion_v1 (the paper's model, LOSOCV)
    RespDiff            261020_RespDiff_v2 (ddpm50)
Nothing is trained: the stored outputs and result workbooks are read.

Subjects left out of every model, of the waveform pages and of the box plots
(MAE issue; subject set 04 of the RRWaveNet runs): STEAM2 22_LMS, 26_CKK,
12_YJJ; BIDMC 27; CapnoBase 0115 (--exclude-steam2 / --exclude-bidmc /
--exclude-capnobase). The figures do not list them; the workbook does.

Part 1. Waveform comparison and pages (261106_RRWaveNet_Performance_Boxplots_v2,
code copied unchanged except the corr-encoder folder, its name on the pages,
the runs and the subjects left out)
-------------------------------------------------------------------------
* GT = the reference as recorded at 30 Hz (CapnoBase capnogram, BIDMC
  impedance, STEAM2 belt); RRWaveNet = the run's stored ensemble output with
  its training sign / lag undone (put back on the GT's time axis), low-pass
  1 Hz; RespDiff and the corr-encoder = their stored 30 Hz outputs; no model
  is shifted in time; a model is inverted once per recording when its r with
  the GT is negative (the same rule for all).
* Windows of 30 s and 60 s (--sizes), one starting every 5 s (--step); in a
  window per model: AE = |FFT rate - GT rate| (breaths/min; GT rate from the
  breaths: CapnoBase expert labels, BIDMC ann1, STEAM2 the GT cycles), PCC
  with the GT; score = PCC - 0.1 x min(AE, 5); Advantage = RRWaveNet's score
  - the best other score. Subject MAE / PCC = mean over the subject's
  non-overlapping windows of a size.
* One page per subject and size: the window with the largest Advantage, three
  panels of the same seconds of the recording, (a) GT vs RRWaveNet of the run,
  (b) GT vs RespDiff, (c) GT vs the deep corr-encoder (261107 paper version);
  GT black, every waveform min-max scaled in the window, beside every panel
  the model's subject MAE / PCC and this window's AE / r.
* The CapnoBase expert breath labels are read with read_capnobase_case of
  261007_RRWaveNet_Ensemble_NoEvent.py (found in the 261007 results folder,
  the folder of this file, the workspace, or --scripts-dir).
Outputs: {run}\Waveforms\{30s,60s}\{dataset}\{subject}_{t0}-{t1}s.png,
{run}\Waveforms\{30s,60s}\{run}_{dataset}_{size}s_waveforms.pdf (every page),
{run}\Waveforms\{run}_Waveform_Summary.xlsx, and for both runs
Waveform_Summary_all_runs.xlsx (Candidates, Subject_Summary, Dataset_Summary,
Drawn_Windows, Notes) in the output folder (run 261007: CapnoBase, run
261013: BIDMC and STEAM2).

Part 2. Box plots of the three models
-------------------------------------
Per dataset, the same subjects; measures and where every value is read:
  Deep corr-encoder: the results stored by 261107_DeepCorrencoder_PaperVersion_v1, as they are,
      {correncoder-root}\Summary\261107_DeepCorrencoder_PaperVersion_v1_results.xlsx, sheet
      Per_subject, rows Evaluation_Window_Sec = 60.6: MAE, PCC, Duty_MAE
      ({correncoder-root} = D:\PPG2RSP_RRWaveNet_Inspired\261107_DeepCorrencoder_PaperVersion_v1;
      MAE = |FFT rate of the fused output - reference rate|, PCC = Pearson r of fused output and
      reference, both over its 60.6 s windows, the paper's evaluation)
  RRWaveNet and RespDiff:
      MAE, PCC    Subject_Summary of part 1 (Size_Sec 60): RRWaveNet_MAE / _PCC, RespDiff_MAE / _PCC
      Duty error  261107_DutyCycle_Results.xlsx: RRWaveNet_Results (Run 261007 / 261013), RespDiff_Duty
                  (Duty_MAE; rule of 261107_DutyCycle_RespDiff_DeepCorrencoder, as the corr-encoder's)
Friedman test of the three models (subjects with all three), Kendall's W;
paired Wilcoxon signed-rank tests of the three pairs, Benjamini-Hochberg over
the 3 pairs; only significant pairs (FDR < 0.05) marked (* < 0.05, ** < 0.01,
*** < 0.001, **** < 1e-4). No reference lines.
Outputs: Boxplots\{MAE, PCC, Duty_error}_per_dataset.png (panels (a) STEAM2,
(b) BIDMC, (c) CapnoBase; every subject a point), ..._no_points.png, and one
dataset per figure (..._{STEAM2, BIDMC, CapnoBase}.png);
261107_Boxplot_Tests.xlsx (Friedman, Pairwise, Values, Sources, Excluded).

Run (default output D:\PPG2RSP_RRWaveNet_Inspired\261107_Boxplots_DeepCorrencoderPaper_RespDiff_RRWaveNet):
  python 261107_Boxplots_DeepCorrencoderPaper_RespDiff_RRWaveNet.py
Box plots again without the waveforms (from Waveform_Summary_all_runs.xlsx of an earlier run):
  python 261107_Boxplots_DeepCorrencoderPaper_RespDiff_RRWaveNet.py --no-waveforms
Options: --root-261007, --root-261013, --output-root, --correncoder-root,
  --respdiff-root, --respdiff-sampling, --sizes, --step, --subjects,
  --scripts-dir, --capnobase-path, --bidmc-path, --steam2-ppg-dir,
  --steam2-rsp-dir, --duty-results, --correncoder-results (default: the
  workbook in --correncoder-root), --waveform-summary, --exclude-*
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
from scipy import stats
from scipy.signal import butter, detrend, resample_poly, sosfiltfilt

# ---------------------------------------------------------------------------
# Folders and settings
# ---------------------------------------------------------------------------

WORKSPACE = Path(r"D:\PPG2RSP_RRWaveNet_Inspired")
PROJECT_ROOT = WORKSPACE / "261107_Boxplots_DeepCorrencoderPaper_RespDiff_RRWaveNet"
FS = 64  # model and analysis sampling rate of the RRWaveNet runs (Hz)
WINDOW = "01_w10s_nonoverlap"
DATASETS = ("CapnoBase", "BIDMC", "STEAM2")
INK, MUTED, GRID = "#1f1f1f", "#6b6a66", "#e4e3df"

# RRWaveNet runs in use: run -> results folder, how the fold's model folder is found, title; and their datasets
RUNS = {
    "261007": {
        "root": WORKSPACE / "261007_RRWaveNet_NoEvent_v1",
        "selected": False,
        "title": "261007 (full model, temporal validation, ensemble of 3)",
    },
    "261013": {
        "root": WORKSPACE / "261013_RRWavNet_STEAM_BIDMC_v1",
        "selected": True,
        "title": "261013 (subject-wise validation, WD per fold, ensemble of 3)",
    },
}
RUN_DATASETS = {"261007": ("CapnoBase",), "261013": ("BIDMC", "STEAM2")}

# Subjects with an MAE issue, left out of every model
EXCLUDED = {"STEAM2": ("22_LMS", "26_CKK", "12_YJJ"), "BIDMC": ("27",), "CapnoBase": ("0115",)}

# Box plots
# The deep corr-encoder: its results folder (outputs for the waveform pages, stored results for the box plots)
# and, in it, the workbook of its stored per-subject results.
CORRENCODER_FOLDER = WORKSPACE / "261107_DeepCorrencoder_PaperVersion_v1"
CORRENCODER_RESULTS_FILE = Path("Summary") / "261107_DeepCorrencoder_PaperVersion_v1_results.xlsx"
WAVEFORM_SUMMARY = PROJECT_ROOT / "Waveform_Summary_all_runs.xlsx"
DUTY_RESULTS = WORKSPACE / "261107_DutyCycle_RespDiff_DeepCorrencoder" / "261107_DutyCycle_Results.xlsx"
BOX_DATASETS = ("STEAM2", "BIDMC", "CapnoBase")  # panel order (a), (b), (c)
RRWAVENET_RUN = {"CapnoBase": "261007", "BIDMC": "261013", "STEAM2": "261013"}
SIZE_SEC = 60
CORRENCODER_WINDOW_SEC = 60.6  # the corr-encoder's stored per-subject rows used (30.6 / 60.6 s; duty is the same)
ID_DIGITS = {"CapnoBase": 4, "BIDMC": 2}
BOX_MODELS = ("RRWaveNet", "Deep corr-encoder", "RespDiff")
BOX_LABEL = {"RRWaveNet": "RRWaveNet", "Deep corr-encoder": "Corr-encoder", "RespDiff": "RespDiff"}
BOX_LEGEND = {"RRWaveNet": "RRWaveNet\n(STEAM2 / BIDMC 261013,\nCapnoBase 261007)",
              "Deep corr-encoder": "Deep corr-encoder\n(261107 paper version)", "RespDiff": "RespDiff"}
BOX_COLOR = {"RRWaveNet": "#1baf7a", "Deep corr-encoder": "#eb6834", "RespDiff": "#2a78d6"}
BOX_PAIRS = (("RRWaveNet", "Deep corr-encoder"), ("Deep corr-encoder", "RespDiff"), ("RRWaveNet", "RespDiff"))

# measure -> (better, title, y label, note)
MEASURES = {
    "MAE": ("lower", "Respiratory rate MAE", "Subject RR MAE (breaths/min)",
            "MAE: |FFT rate - reference rate|,\nmean over the subject's windows.\nRRWaveNet, RespDiff: 60 s windows,\n"
            "GT rate from the breaths.\nCorr-encoder: its stored result\n(60.6 s windows, paper evaluation)."),
    "PCC": ("higher", "Waveform PCC", "Subject waveform PCC (r)",
            "PCC: Pearson r with the GT, mean\nover the subject's windows.\nRRWaveNet, RespDiff: 60 s windows,\n"
            "sign matched once per recording.\nCorr-encoder: its stored result\n(60.6 s windows, paper evaluation)."),
    "Duty_error": ("lower", "Duty cycle error", "Subject duty cycle error |model - GT| (0-1)",
                   "Duty = share of time the [0, 1]\nwaveform is >= 0.5 in 60 s\nwindows (RespDiff definition).\n"
                   "Corr-encoder: its stored result."),
}


# ===========================================================================
# Part 1. Waveform comparison and pages (from 261106_RRWaveNet_Performance_Boxplots_v2, unchanged
# except the corr-encoder folder, the corr-encoder's name on the pages and the subjects left out)
# ===========================================================================

# RRWaveNet run results (fold files, scored minutes = the run's own MAE / PCC)

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
        a = int(round(float(row["RR_Window_Start_Sec"]) * FS))
        b = int(round(float(row["RR_Window_End_Sec"]) * FS))
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


def same_subject(a: object, b: object) -> bool:
    """'0115' == '115' == '0115'; '27' == '27'; other names must match exactly."""
    a, b = str(a).strip(), str(b).strip()
    return a == b or (a.isdigit() and b.isdigit() and int(a) == int(b))


def write_excel(path: Path, sheets: Dict[str, pd.DataFrame]) -> None:
    with pd.ExcelWriter(path, engine="openpyxl") as writer:
        for name, frame in sheets.items():
            frame.to_excel(writer, sheet_name=name, index=False)
            sheet = writer.sheets[name]
            for i, column in enumerate(frame.columns):
                width = max([len(str(column))] + [len(str(v)) for v in frame[column].head(200)])
                sheet.column_dimensions[sheet.cell(1, i + 1).column_letter].width = min(max(width + 2, 8), 80)



# ---------------------------------------------------------------------------
# Waveform overlays: GT, RespDiff, deep corr-encoder and RRWaveNet
# (readers of 261104_Waveform_Overlay, window measures of
# 261105_model_discussion_analysis)
# ---------------------------------------------------------------------------

SCRIPT_DIR = Path(__file__).resolve().parent
# CapnoBase reader with the expert breath labels: read_capnobase_case of the
# 261007 script (or of the 261105 script, which keeps it unchanged)
CAPNOBASE_READERS = (("261007", "261007_RRWaveNet_Ensemble_NoEvent.py"),
                     ("261105", "261105_RRWaveNet_Unified_Optimized.py"))
RESPDIFF_ROOT = WORKSPACE / "261020_RespDiff_v2"
CORRENCODER_ROOT = WORKSPACE / "261107_DeepCorrencoder_PaperVersion_v1"  # the paper version (LOSOCV)
RESPDIFF_SAMPLING = "ddpm50"
RRWAVENET_LOWPASS_HZ = 1.0
FS30 = 30  # time axis of the overlays: the 30 Hz of RespDiff, the corr-encoder and the GT
MODELS = ("RespDiff", "Deep corr-encoder", "RRWaveNet")
OTHERS = MODELS[:2]
SHORT = {"RespDiff": "RespDiff", "Deep corr-encoder": "CorrEncoder", "RRWaveNet": "RRWaveNet"}
MODEL_COLORS = {"RespDiff": "#2a78d6", "Deep corr-encoder": "#eb6834", "RRWaveNet": "#1baf7a"}
DATASET_KEYS = {"CapnoBase": "capnobase", "BIDMC": "bidmc", "STEAM2": "steam2"}
RATE_BAND_BPM = {"capnobase": (3.0, 60.0), "bidmc": (6.0, 30.0), "steam2": (6.0, 30.0)}
WINDOW_SIZES = (30, 60)
STEP_SEC = 5.0
MIN_RUN_SEC = 30.0
MIN_BREATHS = 3
AE_WEIGHT, AE_CAP = 0.1, 5.0
RATE_OK_BPM = 1.0
DOUBLE, HALF = (1.75, 2.25), (0.4, 0.6)

m07 = None  # module holding read_capnobase_case, loaded by load_capnobase_reader


# ---------------------------------------------------------------------------
# Copied unchanged from 261104_Waveform_Overlay (data / result readers, Track,
# filters; its 30 Hz FS is FS30 here) and 261030_Waveform_Morphology_Analysis
# (breathing cycle detector), so that no other script is needed for the
# overlays except the CapnoBase reader (expert breath labels) of 261007
# ---------------------------------------------------------------------------

# Data files of the GT (as in 261104 and the three scripts).
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
SATURATION_CONFIG = {"saturation_min_sec": 0.1, "saturation_margin_sec": 0.1}
RRWAVENET_LOWPASS_ORDER = 8
RRWAVENET_FS = FS  # 64 Hz of the RRWaveNet outputs
CYCLE_CONFIG = {
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


def safe_name(text: str) -> str:
    return re.sub(r"[^0-9A-Za-z._-]+", "_", str(text))


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
        path = find_script(name, [args.scripts_dir, getattr(args, f"root_{run}")])
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
          "261105_RRWaveNet_Unified_Optimized.py; --scripts-dir): CapnoBase overlays skipped")
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


def fft_rate(x: np.ndarray, fs: float, band_bpm: Tuple[float, float]) -> float:
    """Frequency (per minute) of the largest FFT peak in the band."""
    x = np.asarray(x, float)
    x = x[np.isfinite(x)]
    if len(x) < 8 or np.std(x) <= 1e-12:
        return math.nan
    n = max(16384, len(x))
    f = np.fft.rfftfreq(n, 1.0 / fs) * 60.0
    p = np.abs(np.fft.rfft(x - np.mean(x), n))
    keep = (f >= band_bpm[0]) & (f <= band_bpm[1])
    return float(f[keep][np.argmax(p[keep])])


def align(gt: np.ndarray, values: np.ndarray) -> np.ndarray:
    """The model on the GT's own time axis, not shifted in time (every sample
    at the same second as the GT sample beside it); inverted when its Pearson
    r with the GT over the recording is negative (every model alike)."""
    values = np.asarray(values, float)
    r = pearson(gt, values)
    return -values if np.isfinite(r) and r < 0 else values


def recording_cycles(values: Dict[str, np.ndarray], modality: str) -> Dict[str, List[dict]]:
    """Breathing cycles of every waveform (261030's detector) on the stretches
    (>= 30 s) where all exist."""
    valid = np.ones(len(values["GT"]), bool)
    for v in values.values():
        valid &= np.isfinite(v)
    out = {key: [] for key in values}
    for s, e in finite_runs(valid, int(MIN_RUN_SEC * FS30)):
        for key, v in values.items():
            y = band_filter(v[s:e], FS30, modality, CYCLE_CONFIG)
            cycles, _ = detect_cycles(y, FS30, modality, CYCLE_CONFIG)
            out[key] += [{k: c[k] + s / FS30 for k in ("start", "transition", "end")} for c in cycles]
    return out


def inside(cycles: List[dict], t0: float, t1: float) -> List[dict]:
    return [c for c in cycles if c["start"] >= t0 and c["end"] <= t1]


def match_cycles(gt: List[dict], model: List[dict]) -> int:
    """GT cycles matched 1:1 by a model cycle starting within half their length."""
    used, matched = set(), 0
    starts = np.array([c["start"] for c in model]) if model else np.empty(0)
    for c in gt:
        tolerance = 0.5 * (c["end"] - c["start"])
        best, best_gap = None, np.inf
        for k in np.flatnonzero(np.abs(starts - c["start"]) <= tolerance):
            gap = abs(starts[k] - c["start"])
            if k not in used and gap < best_gap:
                best, best_gap = int(k), gap
        if best is not None:
            used.add(best)
            matched += 1
    return matched


def window_row(values: Dict[str, np.ndarray], cycles: Dict[str, List[dict]], breaths: np.ndarray, t0: float,
               t1: float, band: Tuple[float, float]) -> Optional[dict]:
    """Rate, AE, PCC, score and cycles of every model in one window (None when
    a waveform has a gap in it or the GT has fewer than 3 breaths)."""
    a, b = int(round(t0 * FS30)), int(round(t1 * FS30))
    if not all(np.isfinite(v[a:b]).all() for v in values.values()):
        return None
    br = breaths[(breaths >= t0) & (breaths < t1)]
    if len(br) < MIN_BREATHS:
        return None
    gt_cycles = inside(cycles["GT"], t0, t1)
    gt_rr = 60.0 / float(np.mean(np.diff(br)))
    row = {"Start_Sec": t0, "End_Sec": t1, "GT_RR": gt_rr, "GT_Cycles": len(gt_cycles),
           "GT_Clipped_Cycles": sum(bool(c.get("clipped")) for c in gt_cycles)}
    for model in MODELS:
        if model not in values:
            continue
        seg = values[model][a:b]
        rate = fft_rate(seg, FS30, band)
        near = [c for c in cycles[model] if c["end"] > t0 and c["start"] < t1]
        matched = match_cycles(gt_cycles, near)
        n_in = len(inside(cycles[model], t0, t1))
        ae = abs(rate - gt_rr)
        pcc = pearson(values["GT"][a:b], seg)
        s = SHORT[model]
        row.update({f"{s}_Rate": rate, f"{s}_Ratio": rate / gt_rr, f"{s}_AE": ae, f"{s}_PCC": pcc,
                    f"{s}_Score": pcc - AE_WEIGHT * min(ae, AE_CAP), f"{s}_Cycles": n_in,
                    f"{s}_Missed": len(gt_cycles) - matched, f"{s}_Extra": max(0, n_in - matched),
                    f"{s}_Coverage": matched / len(gt_cycles) if gt_cycles else math.nan})
    others = [row[f"{SHORT[m]}_Score"] for m in OTHERS if np.isfinite(row.get(f"{SHORT[m]}_Score", math.nan))]
    row["Advantage"] = row["RRWaveNet_Score"] - max(others) if others else math.nan
    return row


def why_text(row: dict) -> str:
    """What the other models do in a window where RRWaveNet is right."""
    parts = []
    for model in OTHERS:
        s = SHORT[model]
        ratio = row.get(f"{s}_Ratio", math.nan)
        if not np.isfinite(ratio):
            continue
        if DOUBLE[0] <= ratio <= DOUBLE[1]:
            parts.append(f"{s} at 2x rate")
        elif HALF[0] <= ratio <= HALF[1]:
            parts.append(f"{s} at half rate")
        elif row.get(f"{s}_AE", 0) > 2:
            parts.append(f"{s} rate off by {row[f'{s}_AE']:.1f}")
        missed = row.get(f"{s}_Missed", 0) - row.get("RRWaveNet_Missed", 0)
        if missed >= 2:
            parts.append(f"{s} misses {int(missed)} more cycles")
        gap = row.get("RRWaveNet_PCC", math.nan) - row.get(f"{s}_PCC", math.nan)
        if np.isfinite(gap) and gap >= 0.3:
            parts.append(f"{s} PCC lower by {gap:.2f}")
    return "; ".join(parts)


def subject_means(rows: List[dict], size: int) -> Tuple[dict, str]:
    """Per model the mean AE (MAE), PCC and score over the subject's
    non-overlapping windows of a size (all its windows when none is complete)."""
    grid = [r for r in rows if r["Size_Sec"] == size and r["Grid"]]
    basis = "non-overlapping windows"
    if not grid:
        grid = [r for r in rows if r["Size_Sec"] == size]
        basis = "overlapping windows (no complete non-overlapping one)"
    out = {"N_Windows": len(grid)}
    for model in MODELS:
        s = SHORT[model]
        ae = [r[f"{s}_AE"] for r in grid if f"{s}_AE" in r and np.isfinite(r[f"{s}_AE"])]
        if ae:
            pcc = [r[f"{s}_PCC"] for r in grid if np.isfinite(r.get(f"{s}_PCC", math.nan))]
            out.update({f"{s}_MAE": float(np.mean(ae)), f"{s}_PCC": float(np.mean(pcc)) if pcc else math.nan,
                        f"{s}_Score": float(np.mean([r[f"{s}_Score"] for r in grid if f"{s}_Score" in r]))})
    wins = [r["Advantage"] > 0 for r in grid if np.isfinite(r.get("Advantage", math.nan))]
    out["RRWaveNet_Best_Window_Share"] = float(np.mean(wins)) if wins else math.nan
    return out, basis


# Panels of a waveform page, top to bottom: the run's RRWaveNet, RespDiff, the
# deep corr-encoder.
PANEL_ORDER = ("RRWaveNet", "RespDiff", "Deep corr-encoder")
PAGE_NAMES = {"Deep corr-encoder": "Deep corr-encoder (261107 paper version)"}
GT_COLOR = "#000000"  # the GT in black in every panel


def waveform_page(path: Path, pdf, run: str, dataset: str, subject: str, recording: Optional[str], size: int,
                  row: dict, values: Dict[str, np.ndarray], means: dict, basis: str, gt_name: str) -> None:
    """One page per drawn window: the GT with each model in its own panel, all
    panels the same time interval, (a) RRWaveNet of the run, (b) RespDiff,
    (c) the deep corr-encoder; every waveform min-max scaled in the window.
    Beside each panel (outside the waveforms): its legend, the model's
    subject MAE and PCC and this window's AE and r."""
    t0, t1 = row["Start_Sec"], row["End_Sec"]
    a, b = int(round(t0 * FS30)), int(round(t1 * FS30))
    t = t0 + np.arange(b - a) / FS30
    side_in, left_in, top_in, bottom_in, panel_in, gap_in = 3.2, 0.75, 1.35, 0.6, 1.75, 0.38
    width = 7.0 + 4.0 * size / 60.0
    height = top_in + bottom_in + len(PANEL_ORDER) * panel_in + (len(PANEL_ORDER) - 1) * gap_in
    fig = plt.figure(figsize=(width, height))
    gt = minmax(values["GT"][a:b])
    axes = []
    for k, model in enumerate(PANEL_ORDER):
        bottom = bottom_in + (len(PANEL_ORDER) - 1 - k) * (panel_in + gap_in)
        ax = fig.add_axes([left_in / width, bottom / height, (width - left_in - side_in - 0.15) / width,
                           panel_in / height], sharex=axes[0] if axes else None)
        axes.append(ax)
        s = SHORT[model]
        name = f"RRWaveNet ({run})" if model == "RRWaveNet" else PAGE_NAMES.get(model, model)
        available = model in values and f"{s}_MAE" in means and f"{s}_AE" in row
        ax.plot(t, gt, color=GT_COLOR, lw=1.5, zorder=1)
        if available:
            ax.plot(t, minmax(values[model][a:b]), color=MODEL_COLORS[model], lw=1.3, zorder=2)
        else:
            ax.text(0.5, 0.5, f"{model}: no result for this subject", transform=ax.transAxes, ha="center",
                    va="center", fontsize=9, color=INK, zorder=3,
                    bbox=dict(boxstyle="round,pad=0.4", facecolor="white", edgecolor=GRID, alpha=0.95))
        ax.set_xlim(t0, t1)
        ax.set_ylim(-1.18, 1.18)
        ax.set_ylabel("Min-max scaled", fontsize=8)
        ax.grid(True, color=GRID, lw=0.6)
        ax.set_axisbelow(True)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        ax.set_title(f"({'abc'[k]}) GT vs {name}", fontsize=9.5, loc="left", pad=3)
        if k < len(PANEL_ORDER) - 1:
            ax.tick_params(labelbottom=False)
        x = (width - side_in + 0.12) / width
        top = (bottom + panel_in) / height
        handles = [Line2D([], [], color=GT_COLOR, lw=1.5)]
        labels = [f"GT ({gt_name}), {row['GT_RR']:.1f} breaths/min"]
        if available:
            handles.append(Line2D([], [], color=MODEL_COLORS[model], lw=1.3))
            labels.append(name)
        fig.legend(handles, labels, loc="upper left", bbox_to_anchor=(x, top), frameon=False, fontsize=7.5,
                   borderaxespad=0.0)
        if available:
            lines = [f"Subject mean ({means['N_Windows']} x {size} s)",
                     f"  MAE {means[f'{s}_MAE']:5.2f}   PCC {means[f'{s}_PCC']:5.2f}",
                     "This window",
                     f"  AE  {row[f'{s}_AE']:5.2f}   r   {row[f'{s}_PCC']:5.2f}"]
            fig.text(x + 0.004, top - 0.5 / height, "\n".join(lines), family="monospace", fontsize=7.2,
                     color=INK, va="top")
        else:
            fig.text(x + 0.004, top - 0.3 / height, f"{model}: no result\nfor this subject", fontsize=7.2,
                     color=MUTED, va="top")
    axes[-1].set_xlabel("Time in the recording (s)")
    name = f"{subject}" + (f" ({recording})" if recording else "")
    fig.text(left_in / width, 1.0 - 0.22 / height, f"{run} · {dataset} {name} · {size} s window {t0:.0f}-{t1:.0f} s",
             fontsize=10.5, color=INK, va="top")
    fig.text(left_in / width, 1.0 - 0.50 / height,
             "all panels the same seconds of the recording (no model shifted in time; inverted when r < 0 over the "
             "recording); GT black\neach waveform min-max scaled in the window; RRWaveNet low-pass 1 Hz\n"
             f"MAE / AE = |FFT rate - GT rate| (breaths/min); subject mean over {basis}",
             fontsize=7.2, color=MUTED, va="top", linespacing=1.4)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=200)
    pdf.savefig(fig)
    plt.close(fig)


def excluded_sets(exclusions: Optional[pd.DataFrame], dataset: str, subject: str) -> str:
    if exclusions is None or exclusions.empty:
        return ""
    found = exclusions[exclusions["Found"].astype(bool) & (exclusions["Dataset"] == dataset)]
    hits = found[found["Subject"].map(lambda s: same_subject(s, subject))]
    return ", ".join(sorted(set(hits["Subject_Set"].astype(str))))


def overlay_dataset(args: argparse.Namespace, dataset: str, runs: Dict[str, dict], out_root: Path) -> None:
    """Every subject of every run with RRWaveNet and at least one other model:
    windows of every size, the subject means and one figure per size (the
    window where RRWaveNet's score is furthest above the best other model)."""
    key = DATASET_KEYS[dataset]
    band = RATE_BAND_BPM[key]
    modality = MODALITIES[key]
    corr = load_correncoder(Path(args.correncoder_root), dataset)
    respdiff = load_respdiff(Path(args.respdiff_root), dataset, args.respdiff_sampling)
    plan = {}
    for run, info in runs.items():
        files = {f["Tag"]: f["Outputs_File"] for f in fold_files(info["root"], dataset, RUNS[run]["selected"])
                 if f["Outputs_File"].exists()}
        subjects = sorted(set(files) & (set(corr) | set(respdiff)))
        if args.subjects:
            subjects = [s for s in subjects if any(same_subject(s, x) for x in args.subjects)]
        left_out = args.excluded.get(dataset, [])
        subjects = [s for s in subjects if not any(same_subject(s, x) for x in left_out)]
        print(f"{dataset} [{run}]: RespDiff {len(respdiff)}, corr-encoder {len(corr)}, RRWaveNet {len(files)}; "
              f"{len(subjects)} subjects with RRWaveNet and another model", flush=True)
        if subjects:
            plan[run] = (files, subjects)
    if not plan:
        return
    recordings = read_recordings(key)
    aligned = {}  # (subject, recording, model) -> RespDiff / corr-encoder sign-matched to the GT (same in every run)
    for run, (files, subjects) in plan.items():
        info = runs[run]
        alignment = rrwavenet_alignment_table(info["root"], dataset)
        pdfs = {}
        for subject in subjects:
            if subject not in recordings:
                print(f"  [{run}] {dataset} {subject}: no data file read (skipped)")
                continue
            segments = load_rrwavenet(files[subject])
            corr_by_name = {r["name"]: r for r in corr.get(subject, [])}
            store, rows = {}, []
            for k, (name, rec) in enumerate(recordings[subject].items()):
                gt = np.asarray(rec["gt"], float)
                times = np.arange(len(gt)) / FS30
                index = corr_by_name[name]["index"] if name in corr_by_name else k
                raw = {}
                if name in respdiff.get(subject, {}):
                    raw["RespDiff"] = Track(respdiff[subject][name], FS30).at(times)
                if name in corr_by_name:
                    raw["Deep corr-encoder"] = Track(corr_by_name[name]["output"], FS30).at(times)
                if index in segments:
                    raw["RRWaveNet"] = rrwavenet_track(segments[index], alignment.get((subject, index)), key, gt,
                                                       args.rrwavenet_lowpass_hz)
                if "RRWaveNet" not in raw or len(raw) < 2:
                    continue
                breaths = rec["breaths"]
                if breaths is None or len(breaths) < MIN_BREATHS:
                    breaths = np.array([c["start"] for c in recording_cycles({"GT": gt}, modality)["GT"]])
                values = {"GT": gt}
                for model, v in raw.items():
                    if model == "RRWaveNet":
                        values[model] = align(gt, v)
                    else:
                        if (subject, name, model) not in aligned:
                            aligned[(subject, name, model)] = align(gt, v)
                        values[model] = aligned[(subject, name, model)]
                cycles = recording_cycles(values, modality)
                clipped = np.asarray(rec["clipped"], bool)
                for c in cycles["GT"]:
                    c["clipped"] = bool(clipped[int(c["start"] * FS30):int(math.ceil(c["end"] * FS30)) + 1].any())
                store[name] = values
                length = len(gt) / FS30
                for size in args.sizes:
                    grid = set(np.arange(0.0, length - size + 1e-9, size).round(3))
                    starts = sorted(grid | set(np.arange(0.0, length - size + 1e-9, args.step).round(3)))
                    for t0 in starts:
                        row = window_row(values, cycles, breaths, float(t0), float(t0) + size, band)
                        if row is not None:
                            rows.append({"Run": run, "Dataset": dataset, "Subject": subject, "Recording": name,
                                         "Size_Sec": size, "Grid": t0 in grid,
                                         "Models": ", ".join(SHORT[m] for m in MODELS if m in values), **row})
            info["windows"] += rows
            flag = excluded_sets(info["exclusions"], dataset, subject)
            official = info["official"].get((dataset, subject), {})
            for size in args.sizes:
                sized = [r for r in rows if r["Size_Sec"] == size and np.isfinite(r["Advantage"])]
                if not sized:
                    continue
                means, basis = subject_means(rows, size)
                best = max(sized, key=lambda r: (r["Advantage"], r["RRWaveNet_PCC"]))
                single = len(recordings[subject]) == 1
                rec_name = None if single else best["Recording"]
                recording = str(best["Recording"])
                stem = (safe_name(subject) if single else safe_name(recording) if
                        recording.startswith(subject) else f"{safe_name(subject)}__{safe_name(recording)}")
                figure = (out_root / run / "Waveforms" / f"{size}s" / dataset /
                          f"{stem}_{best['Start_Sec']:.0f}-{best['End_Sec']:.0f}s.png")
                if size not in pdfs:
                    pdf_path = out_root / run / "Waveforms" / f"{size}s" / f"{run}_{dataset}_{size}s_waveforms.pdf"
                    pdf_path.parent.mkdir(parents=True, exist_ok=True)
                    pdfs[size] = PdfPages(pdf_path)
                waveform_page(figure, pdfs[size], run, dataset, subject, rec_name, size, best,
                              store[best["Recording"]], means, basis, MODALITIES[key])
                summary = {"Run": run, "Dataset": dataset, "Subject": subject, "Size_Sec": size,
                           "Subject_Means_Over": basis, **means,
                           "Official_RR_MAE": official.get("RR_MAE", math.nan),
                           "Official_PCC": official.get("PCC", math.nan), "In_Excluded_Sets": flag}
                info["subjects"].append(summary)
                drawn = {**best, "Why": why_text(best), "In_Excluded_Sets": flag,
                         **{f"Subject_{k}": v for k, v in means.items()},
                         "Figure": str(figure.relative_to(out_root))}
                info["drawn"].append(drawn)
            print(f"  [{run}] {dataset} {subject}: {len(rows)} windows, models "
                  f"{', '.join(sorted({m for r in rows for m in r['Models'].split(', ')}))}", flush=True)
        for pdf in pdfs.values():
            pdf.close()


def ranks(frame: pd.DataFrame, metric: str, lower_better: bool) -> pd.Series:
    """Rank of RRWaveNet among the models present (1 = best)."""
    columns = [f"{SHORT[m]}_{metric}" for m in MODELS if f"{SHORT[m]}_{metric}" in frame.columns]
    values = frame[columns]
    rr = values[f"RRWaveNet_{metric}"]
    better = (values.lt(rr, axis=0) if lower_better else values.gt(rr, axis=0)).sum(axis=1)
    return (better + 1).where(rr.notna())


def subject_table(subjects: pd.DataFrame) -> pd.DataFrame:
    if subjects.empty:
        return subjects
    out = subjects.copy()
    others_mae = out[[c for c in ("RespDiff_MAE", "CorrEncoder_MAE") if c in out]].min(axis=1)
    others_pcc = out[[c for c in ("RespDiff_PCC", "CorrEncoder_PCC") if c in out]].max(axis=1)
    out["RRWaveNet_MAE_Rank"] = ranks(out, "MAE", True)
    out["RRWaveNet_PCC_Rank"] = ranks(out, "PCC", False)
    out["MAE_Gain_vs_Best_Other"] = others_mae - out["RRWaveNet_MAE"]
    out["PCC_Gain_vs_Best_Other"] = out["RRWaveNet_PCC"] - others_pcc
    return out


def dataset_table(subjects: pd.DataFrame) -> pd.DataFrame:
    """Per run, dataset and size: means and medians of the subject values of
    the subjects with every model of the dataset, and how often RRWaveNet is
    best."""
    rows = []
    if subjects.empty:
        return pd.DataFrame()
    for (run, dataset, size), d in subjects.groupby(["Run", "Dataset", "Size_Sec"], sort=False):
        present = [f"{SHORT[m]}_MAE" for m in MODELS if f"{SHORT[m]}_MAE" in d and d[f"{SHORT[m]}_MAE"].notna().any()]
        full = d.dropna(subset=present)
        row = {"Run": run, "Dataset": dataset, "Size_Sec": size, "N_Subjects": len(d),
               "Models_Compared": ", ".join(c[:-4] for c in present), "N_Subjects_All_Models": len(full)}
        for model in MODELS:
            s = SHORT[model]
            for metric in ("MAE", "PCC"):
                if f"{s}_{metric}" in full:
                    row[f"{s}_{metric}_Mean"] = full[f"{s}_{metric}"].mean()
                    row[f"{s}_{metric}_Median"] = full[f"{s}_{metric}"].median()
        if len(full):
            row["RRWaveNet_Best_MAE_Subjects"] = int((full["RRWaveNet_MAE_Rank"] == 1).sum())
            row["RRWaveNet_Best_PCC_Subjects"] = int((full["RRWaveNet_PCC_Rank"] == 1).sum())
            row["RRWaveNet_Best_Window_Share_Mean"] = full["RRWaveNet_Best_Window_Share"].mean()
        rows.append(row)
    return pd.DataFrame(rows)


def candidates(drawn: pd.DataFrame) -> pd.DataFrame:
    """Drawn windows where RRWaveNet is right (AE <= 1 breath/min) and ahead
    of the best other model, the largest lead first."""
    if drawn.empty:
        return drawn
    keep = drawn[(drawn["RRWaveNet_AE"] <= RATE_OK_BPM) & (drawn["Advantage"] > 0)].copy()
    keep = keep.sort_values("Advantage", ascending=False)
    keep.insert(0, "Rank", np.arange(1, len(keep) + 1))
    first = ["Rank", "Run", "Dataset", "Subject", "Recording", "Size_Sec", "Start_Sec", "End_Sec", "GT_RR",
             "Advantage", "Why", "RRWaveNet_AE", "RRWaveNet_PCC", "RespDiff_AE", "RespDiff_PCC", "RespDiff_Ratio",
             "CorrEncoder_AE", "CorrEncoder_PCC", "CorrEncoder_Ratio", "Subject_RRWaveNet_MAE",
             "Subject_RRWaveNet_PCC", "Subject_RespDiff_MAE", "Subject_RespDiff_PCC", "Subject_CorrEncoder_MAE",
             "Subject_CorrEncoder_PCC", "In_Excluded_Sets", "Figure"]
    first = [c for c in first if c in keep.columns]
    return keep[first + [c for c in keep.columns if c not in first]]


def overlay_notes(args: argparse.Namespace) -> pd.DataFrame:
    lines = [
        ("Models", f"RespDiff ({args.respdiff_root}, {args.respdiff_sampling}), deep corr-encoder "
                   f"({args.correncoder_root}), RRWaveNet = the run's stored ensemble output (as the box plots)"),
        ("GT", "the reference as recorded (CapnoBase capnogram, BIDMC impedance, STEAM2 belt) at 30 Hz, read as "
               "261104; GT rate from the breaths in the window (CapnoBase expert labels, BIDMC ann1, STEAM2 the "
               "GT cycles); a window needs >= 3 breaths and every waveform without a gap"),
        ("Alignment", "no model is shifted in time: every waveform of a page covers the same seconds of the "
                      "recording as the GT; a model is inverted once per recording when its r with the GT is "
                      "negative (the same rule for all); RRWaveNet: training sign / lag undone (its output put "
                      "back on the GT's time axis), "
                      f"low-pass {args.rrwavenet_lowpass_hz:g} Hz (as 261104 / 261105_model_discussion_analysis)"),
        ("AE / MAE", "|FFT peak rate of the model's waveform in the rate band - GT rate| (breaths/min); "
                     "CapnoBase band 3-60, BIDMC / STEAM2 6-30; subject MAE = mean AE over the subject's "
                     "non-overlapping windows of a size (all its windows when none is complete)"),
        ("PCC", "Pearson r of the model and the GT over the same samples of the window (no time shift); subject "
                "PCC = mean over the same windows"),
        ("Score / Advantage", f"score = PCC - {AE_WEIGHT} x min(AE, {AE_CAP}); Advantage = RRWaveNet's score - the "
                              "best score of the other models in the window (> 0: RRWaveNet ahead)"),
        ("Drawn window", "per subject and size the window (start every "
                         f"{args.step:g} s) with the largest Advantage, one page with three panels of the same "
                         "interval: (a) GT vs RRWaveNet of the run, (b) GT vs RespDiff, (c) GT vs the deep "
                         "corr-encoder, all the same seconds of the recording, GT black; beside each panel the "
                         "model's subject means and this window's AE and r"),
        ("Candidates", f"drawn windows with RRWaveNet AE <= {RATE_OK_BPM} and Advantage > 0, largest Advantage "
                       "first; Why: the other models at twice / half the rate, rate off by > 2, >= 2 more GT "
                       "cycles missed than RRWaveNet, PCC lower by >= 0.3"),
        ("Official_RR_MAE / PCC", "the run's own per-subject values (60 s minutes, its target; the box plots), "
                                  "for reference; the overlay values use the recorded GT and the windows above"),
        ("In_Excluded_Sets", "the subject sets of the box plots that leave this subject out"),
        ("Cycles", "261030 detector on the waveforms as drawn (not shifted in time); Missed = GT cycles without a "
                   "model cycle starting within half their length (1:1); Extra = model cycles not matched; "
                   "Coverage = matched / GT"),
    ]
    return pd.DataFrame(lines, columns=["Item", "Description"])


def write_overlay_tables(args: argparse.Namespace, runs: Dict[str, dict], out_root: Path) -> None:
    combined = {"Candidates": [], "Subject_Summary": [], "Dataset_Summary": [], "Drawn_Windows": []}
    for run, info in runs.items():
        if not info["windows"]:
            continue
        folder = out_root / run / "Waveforms"
        folder.mkdir(parents=True, exist_ok=True)
        windows = pd.DataFrame(info["windows"])
        subjects = subject_table(pd.DataFrame(info["subjects"]))
        drawn = pd.DataFrame(info["drawn"])
        cand = candidates(drawn)
        data = dataset_table(subjects)
        windows.to_csv(folder / f"{run}_waveform_windows_all.csv", index=False, encoding="utf-8-sig")
        sheets = {"Candidates": cand, "Drawn_Windows": drawn, "Subject_Summary": subjects,
                  "Dataset_Summary": data, "Grid_Windows": windows[windows["Grid"]]}
        for name, frame in sheets.items():
            frame.to_csv(folder / f"{run}_waveform_{name.lower()}.csv", index=False, encoding="utf-8-sig")
        write_excel(folder / f"{run}_Waveform_Summary.xlsx",
                    {**{k: v.round(4) for k, v in sheets.items()}, "Notes": overlay_notes(args)})
        for name in combined:
            combined[name].append(sheets[name])
        print(f"[{run}] waveforms: {len(drawn)} pages, {len(cand)} candidate windows; saved under {folder}")
    frames = {k: pd.concat(v, ignore_index=True) for k, v in combined.items() if v}
    if not frames:
        return
    if "Candidates" in frames and not frames["Candidates"].empty:
        cand = frames["Candidates"].drop(columns="Rank").sort_values("Advantage", ascending=False)
        cand.insert(0, "Rank", np.arange(1, len(cand) + 1))
        frames["Candidates"] = cand
    path = out_root / "Waveform_Summary_all_runs.xlsx"
    write_excel(path, {**{k: v.round(4) for k, v in frames.items()}, "Notes": overlay_notes(args)})
    print(f"Waveform summary of every run: {path}")




# ===========================================================================
# Part 2. Box plots of the three models (MAE, PCC, duty cycle error)
# ===========================================================================

# ---------------------------------------------------------------------------
# Reading the workbooks
# ---------------------------------------------------------------------------

def norm_id(dataset: str, subject) -> str:
    """Subject IDs as the data files name them (CapnoBase 0009, BIDMC 01), also when a workbook cell lost the
    leading zeros."""
    text = str(subject).strip()
    text = re.sub(r"\.0+$", "", text)
    if dataset in ID_DIGITS and text.isdigit():
        text = text.zfill(ID_DIGITS[dataset])
    return text


def read_sheet(path: Path, sheet: str) -> pd.DataFrame:
    frame = pd.read_excel(path, sheet_name=sheet, dtype={"Subject": str})
    frame["Subject"] = [norm_id(d, s) for d, s in zip(frame["Dataset"], frame["Subject"])]
    return frame


def waveform_values(path: Path) -> pd.DataFrame:
    """MAE and PCC of RRWaveNet and RespDiff (Subject_Summary of part 1, 60 s)."""
    summary = read_sheet(path, "Subject_Summary")
    summary["Run"] = summary["Run"].astype(str).str.replace(r"\.0+$", "", regex=True)
    rows = []
    for dataset in BOX_DATASETS:
        part = summary[(summary["Dataset"] == dataset) & (summary["Run"] == RRWAVENET_RUN[dataset])
                       & (pd.to_numeric(summary["Size_Sec"], errors="coerce") == SIZE_SEC)]
        for measure in ("MAE", "PCC"):
            for model in ("RRWaveNet", "RespDiff"):
                column = f"{model}_{measure}"
                if column not in part:
                    continue
                for subject, value in zip(part["Subject"], pd.to_numeric(part[column], errors="coerce")):
                    rows.append({"Dataset": dataset, "Subject": subject, "Model": model, "Measure": measure,
                                 "Value": value, "Source": f"{path}: Subject_Summary {column} "
                                                           f"(Run {RRWAVENET_RUN[dataset]}, {SIZE_SEC} s)"})
    return pd.DataFrame(rows)


def duty_values(duty_path: Path) -> pd.DataFrame:
    """Duty_MAE of RRWaveNet and RespDiff (261107 duty workbook)."""
    rows = []
    rr = read_sheet(duty_path, "RRWaveNet_Results")
    rr["Run"] = rr["Run"].astype(str).str.replace(r"\.0+$", "", regex=True)
    rd = read_sheet(duty_path, "RespDiff_Duty")
    for dataset in BOX_DATASETS:
        sources = (
            ("RRWaveNet", rr[(rr["Dataset"] == dataset) & (rr["Run"] == RRWAVENET_RUN[dataset])],
             f"{duty_path}: RRWaveNet_Results Duty_MAE (Run {RRWAVENET_RUN[dataset]})"),
            ("RespDiff", rd[rd["Dataset"] == dataset], f"{duty_path}: RespDiff_Duty Duty_MAE"),
        )
        for model, part, source in sources:
            if "Duty_MAE" not in part:
                continue
            for subject, value in zip(part["Subject"], pd.to_numeric(part["Duty_MAE"], errors="coerce")):
                rows.append({"Dataset": dataset, "Subject": subject, "Model": model, "Measure": "Duty_error",
                             "Value": value, "Source": source})
    return pd.DataFrame(rows)


def correncoder_values(path: Path) -> pd.DataFrame:
    """The deep corr-encoder's results as 261107_DeepCorrencoder_PaperVersion_v1 stored them (sheet Per_subject,
    rows of the CORRENCODER_WINDOW_SEC evaluation windows): MAE, PCC and Duty_MAE, unchanged."""
    ce = read_sheet(path, "Per_subject")
    window = pd.to_numeric(ce["Evaluation_Window_Sec"], errors="coerce")
    ce = ce[np.isclose(window, CORRENCODER_WINDOW_SEC)]
    if ce.empty:
        raise SystemExit(f"{path}: no Per_subject rows with Evaluation_Window_Sec = {CORRENCODER_WINDOW_SEC}")
    rows = []
    for measure, column in (("MAE", "MAE"), ("PCC", "PCC"), ("Duty_error", "Duty_MAE")):
        if column not in ce:
            print(f"[Warning] {path}: no column {column}; corr-encoder {measure} left out")
            continue
        for dataset, subject, value in zip(ce["Dataset"], ce["Subject"], pd.to_numeric(ce[column], errors="coerce")):
            if dataset in BOX_DATASETS:
                rows.append({"Dataset": dataset, "Subject": subject, "Model": "Deep corr-encoder", "Measure": measure,
                             "Value": value, "Source": f"{path}: Per_subject {column} "
                                                       f"(Evaluation_Window_Sec {CORRENCODER_WINDOW_SEC:g})"})
    print(f"Deep corr-encoder: stored results of {path} ("
          + ", ".join(f"{d} {int((ce['Dataset'] == d).sum())}" for d in BOX_DATASETS) + " subjects)")
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Statistics
# ---------------------------------------------------------------------------

def bh(p: Sequence[float]) -> np.ndarray:
    p = np.asarray(p, float)
    out = np.full(p.shape, np.nan)
    ok = np.isfinite(p)
    if ok.any():
        q = p[ok]
        order = np.argsort(q)
        adj = q[order] * len(q) / np.arange(1, len(q) + 1)
        adj = np.minimum.accumulate(adj[::-1])[::-1].clip(0, 1)
        back = np.empty_like(adj)
        back[order] = adj
        out[ok] = back
    return out


def stars(q: float) -> str:
    if not np.isfinite(q):
        return ""
    return "****" if q < 1e-4 else "***" if q < 1e-3 else "**" if q < 1e-2 else "*" if q < 0.05 else ""


def pstr(p: float) -> str:
    return "n/a" if not np.isfinite(p) else f"{p:.1e}" if p < 1e-3 else f"{p:.3f}"


def tests(values: pd.DataFrame, measure: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
    better = MEASURES[measure][0]
    friedman, pairwise = [], []
    for dataset in BOX_DATASETS:
        part = values[(values["Measure"] == measure) & (values["Dataset"] == dataset)]
        wide = part.pivot_table(index="Subject", columns="Model", values="Value", aggfunc="first")
        wide = wide.reindex(columns=list(BOX_MODELS))
        full = wide.dropna()
        chi2 = p = math.nan
        if len(full) >= 3:
            try:
                chi2, p = stats.friedmanchisquare(*[full[m] for m in BOX_MODELS])
            except ValueError:
                pass
        friedman.append({"Measure": measure, "Dataset": dataset, "N_Subjects_All_Three": len(full), "Chi2": chi2,
                         "Kendall_W": chi2 / (len(full) * 2) if len(full) and np.isfinite(chi2) else math.nan, "P": p,
                         **{f"N_{BOX_LABEL[m]}": int(wide[m].notna().sum()) for m in BOX_MODELS},
                         **{f"Median_{BOX_LABEL[m]}": wide[m].median() for m in BOX_MODELS},
                         **{f"Mean_{BOX_LABEL[m]}": wide[m].mean() for m in BOX_MODELS},
                         **{f"SD_{BOX_LABEL[m]}": wide[m].std() for m in BOX_MODELS}})
        rows = []
        for a, b in BOX_PAIRS:
            pair = wide[[a, b]].dropna()
            diff = pair[a] - pair[b]
            lower = better == "lower"
            p_pair = math.nan
            if len(pair) >= 2 and (diff != 0).any():
                try:
                    p_pair = float(stats.wilcoxon(pair[a], pair[b]).pvalue)
                except ValueError:
                    pass
            rows.append({"Measure": measure, "Dataset": dataset, "A": a, "B": b, "N_Pairs": len(pair),
                         "A_Median": pair[a].median(), "B_Median": pair[b].median(),
                         "N_A_Better": int(((diff < 0) if lower else (diff > 0)).sum()),
                         "N_B_Better": int(((diff > 0) if lower else (diff < 0)).sum()), "P": p_pair})
        for row, q in zip(rows, bh([r["P"] for r in rows])):
            row["FDR_within_dataset"] = q
            row["Stars"] = stars(q)
            row["Better"] = ((row["A"] if (row["A_Median"] < row["B_Median"]) == (better == "lower") else row["B"])
                             if np.isfinite(q) and q < 0.05 else "")
        pairwise += rows
    return pd.DataFrame(friedman), pd.DataFrame(pairwise)


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------

def nice_step(span: float) -> float:
    for step in (0.05, 0.1, 0.2, 0.25, 0.5, 1.0, 2.0, 2.5, 5.0, 10.0, 20.0):
        if span / step <= 10:
            return step
    return 50.0


def whisker_range(v: np.ndarray) -> Tuple[float, float]:
    q1, q3 = np.percentile(v, [25, 75])
    iqr = q3 - q1
    return float(v[v >= q1 - 1.5 * iqr].min()), float(v[v <= q3 + 1.5 * iqr].max())


def draw(values: pd.DataFrame, fr: pd.DataFrame, pw: pd.DataFrame, measure: str, points: bool,
         out: Path) -> int:
    better, title, ylabel, note_extra = MEASURES[measure]
    data = values[values["Measure"] == measure]
    vals = {(d, m): data[(data["Dataset"] == d) & (data["Model"] == m)]["Value"].dropna().to_numpy(float)
            for d in BOX_DATASETS for m in BOX_MODELS}
    present = [v for v in vals.values() if len(v)]
    if not present:
        print(f"[{measure}] no values; no figure")
        return 0
    if points:
        lo, top = min(v.min() for v in present), max(v.max() for v in present)
    else:
        ranges = [whisker_range(v) for v in present]
        lo, top = min(r[0] for r in ranges), max(r[1] for r in ranges)
    span = max(top - lo, 1e-6)
    ylim = (lo - 0.05 * span, top + 0.05 * span + 3 * 0.09 * span + 0.03 * span)
    step = nice_step(span)
    cap = 1.0 if measure == "PCC" else top + 0.02 * span
    drawn: List[tuple] = []

    def panel(ax, dataset: str, ptitle: str, show_ylabel: bool) -> None:
        rng = np.random.default_rng(0)
        for j, model in enumerate(BOX_MODELS):
            v = vals[(dataset, model)]
            if not len(v):
                ax.text(j, 0.5, "no results", transform=ax.get_xaxis_transform(), ha="center", va="center",
                        fontsize=9, color=MUTED)
                continue
            box = ax.boxplot([v], positions=[j], widths=0.62, patch_artist=True, showfliers=False,
                             medianprops={"color": INK, "lw": 1.8}, whiskerprops={"color": MUTED, "lw": 1.1},
                             capprops={"color": MUTED, "lw": 1.1})
            box["boxes"][0].set(facecolor=BOX_COLOR[model], alpha=0.3 if points else 0.45, edgecolor=BOX_COLOR[model], lw=1.2)
            if points:
                ax.scatter(j + rng.uniform(-0.18, 0.18, len(v)), v, s=10, color=BOX_COLOR[model], edgecolors="white",
                           lw=0.3, zorder=3)
            ax.text(j, -0.1, f"median {np.median(v):.3f}\nn = {len(v)}", transform=ax.get_xaxis_transform(),
                    ha="center", va="top", fontsize=7.5, color=MUTED)
        local = [(v.max() if points else whisker_range(v)[1]) for m in BOX_MODELS
                 for v in [vals[(dataset, m)]] if len(v)]
        local_top = max(local) if local else top
        sig = pw[(pw["Dataset"] == dataset) & (pw["FDR_within_dataset"] < 0.05)]
        ordered = sorted(sig.itertuples(), key=lambda r: abs(BOX_MODELS.index(r.B) - BOX_MODELS.index(r.A)))
        for level, r in enumerate(ordered):
            x1, x2 = sorted((BOX_MODELS.index(r.A), BOX_MODELS.index(r.B)))
            y = local_top + 0.05 * span + level * 0.09 * span
            ax.plot([x1 + 0.04, x1 + 0.04, x2 - 0.04, x2 - 0.04], [y, y + 0.02 * span, y + 0.02 * span, y],
                    color=INK, lw=0.9, clip_on=False)
            ax.text((x1 + x2) / 2, y + 0.022 * span, r.Stars, ha="center", va="bottom", fontsize=10, color=INK)
            drawn.append((dataset, r.A, r.B))
        f = fr[fr["Dataset"] == dataset].iloc[0]
        ax.set_title(f"{ptitle}\nFriedman p = {pstr(f['P'])}, Kendall's W = "
                     f"{f['Kendall_W']:.2f}, n = {f['N_Subjects_All_Three']}" if np.isfinite(f["Kendall_W"]) else
                     f"{ptitle}\nFriedman: n/a (n = {f['N_Subjects_All_Three']} with all three)",
                     loc="left", fontsize=9.5, pad=8)
        ax.set_ylim(*ylim)
        ticks = np.arange(math.floor(ylim[0] / step) * step, top + 1e-9, step)
        ax.set_yticks([t for t in ticks if ylim[0] <= t <= cap])
        ax.set_xticks(range(len(BOX_MODELS)))
        ax.set_xticklabels([BOX_LABEL[m] for m in BOX_MODELS], fontsize=9.5)
        if show_ylabel:
            ax.set_ylabel(ylabel, fontsize=9)
        ax.grid(True, axis="y", color=GRID, lw=0.7)
        ax.set_axisbelow(True)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)

    handles = [plt.Rectangle((0, 0), 1, 1, facecolor=BOX_COLOR[m], alpha=0.5, edgecolor=BOX_COLOR[m]) for m in BOX_MODELS]
    note = ("Only significant pairs are marked:\npaired Wilcoxon signed-rank test\n(same subjects), Benjamini-Hochberg\n"
            "over the 3 pairs of the dataset.\n* < 0.05, ** < 0.01,\n*** < 0.001, **** < 1e-4\n\n"
            "Friedman: the three models\ntogether (subjects with all three).\n\n" + note_extra)
    sub = ("box: median and IQR, whiskers 1.5 IQR" + ("; every subject a point" if points else " (points not drawn)")
           + "; the same y range in every panel")
    head = f"{title}: the three models compared within each dataset ({better} = better)"
    name = f"{measure}_per_dataset" + ("" if points else "_no_points")

    fig, axes = plt.subplots(1, 3, figsize=(17, 6.2))
    fig.subplots_adjust(left=0.05, right=0.83, top=0.78, bottom=0.13, wspace=0.16)
    for c, dataset in enumerate(BOX_DATASETS):
        panel(axes[c], dataset, f"({'abc'[c]}) {dataset}", c == 0)
    fig.legend(handles, [BOX_LEGEND[m] for m in BOX_MODELS], loc="upper left", bbox_to_anchor=(0.845, 0.78), frameon=False,
               fontsize=9)
    fig.text(0.845, 0.46, note, fontsize=8.3, color=MUTED, va="top")
    fig.text(0.05, 0.94, head, fontsize=13, color=INK)
    fig.text(0.05, 0.905, sub, fontsize=9, color=MUTED)
    fig.savefig(out / f"{name}.png", dpi=200)
    plt.close(fig)
    significant = {(r.Dataset, r.A, r.B) for r in pw.itertuples() if r.FDR_within_dataset < 0.05}
    assert significant == set(drawn), (measure, significant, drawn)

    run_of = {"STEAM2": "261013", "BIDMC": "261013", "CapnoBase": "261007"}
    for dataset in BOX_DATASETS:
        fig, ax = plt.subplots(figsize=(8.6, 6.2))
        fig.subplots_adjust(left=0.1, right=0.66, top=0.78, bottom=0.13)
        panel(ax, dataset, dataset, True)
        labels = [f"RRWaveNet ({run_of[dataset]})", BOX_LEGEND["Deep corr-encoder"], "RespDiff"]
        fig.legend(handles, labels, loc="upper left", bbox_to_anchor=(0.68, 0.78), frameon=False, fontsize=9)
        fig.text(0.68, 0.48, note, fontsize=8, color=MUTED, va="top")
        fig.text(0.1, 0.94, f"{dataset}: {title[0].lower() + title[1:]} of the three models ({better} = better)", fontsize=12,
                 color=INK)
        fig.text(0.1, 0.905, sub.replace("; the same y range in every panel", ""), fontsize=8.5, color=MUTED)
        fig.savefig(out / f"{name}_{dataset}.png", dpi=200)
        plt.close(fig)
    return len(significant)


# ===========================================================================
# Runner: part 1 (waveforms), then part 2 (box plots)
# ===========================================================================

def official_values(run: str, root: Path) -> Dict[Tuple[str, str], dict]:
    """The run's own per-subject RR MAE / PCC (its scored minutes; Official_* columns of the Subject_Summary)."""
    frames = []
    for dataset in RUN_DATASETS[run]:
        for fold in fold_files(root, dataset, RUNS[run]["selected"]):
            frame, _ = fold_minutes(fold, dataset)
            if not frame.empty:
                frames.append(frame)
    if not frames:
        return {}
    subjects = per_subject(pd.concat(frames, ignore_index=True))
    return {(r["Dataset"], str(r["Subject"])): r for r in subjects.to_dict("records")}


def waveforms(args: argparse.Namespace, out_root: Path) -> None:
    runs = {}
    for run in RUN_DATASETS:
        root = Path(getattr(args, f"root_{run}"))
        if not root.exists():
            print(f"[{run}] results folder not found: {root} (skipped)")
            continue
        runs[run] = {"root": root, "exclusions": None, "official": official_values(run, root),
                     "windows": [], "subjects": [], "drawn": []}
    datasets = [d for d in DATASETS if any(d in RUN_DATASETS[run] for run in runs)]
    if not datasets:
        print("\nWaveforms: no RRWaveNet result found (skipped)")
        return
    print("\nWaveforms: GT vs RRWaveNet / RespDiff / deep corr-encoder (paper version), one page per window")
    set_data_paths(args)
    if "CapnoBase" in datasets and not load_capnobase_reader(args):
        datasets.remove("CapnoBase")
    for dataset in datasets:
        overlay_dataset(args, dataset, {run: info for run, info in runs.items() if dataset in RUN_DATASETS[run]},
                        out_root)
    write_overlay_tables(args, runs, out_root)


def boxplots(args: argparse.Namespace, out_root: Path, summary_path: Path) -> None:
    excluded = args.excluded
    for path in (summary_path, Path(args.duty_results), Path(args.correncoder_results)):
        if not path.is_file():
            raise SystemExit(f"Not found: {path}")
    values = pd.concat([waveform_values(summary_path), duty_values(Path(args.duty_results)),
                        correncoder_values(Path(args.correncoder_results))], ignore_index=True)
    removed = []
    for dataset, subjects in excluded.items():
        for subject in subjects:
            hit = (values["Dataset"] == dataset) & (values["Subject"] == subject)
            removed.append({"Dataset": dataset, "Subject": subject, "Rows_Removed": int(hit.sum()),
                            "Status": "left out of every model" if hit.any() else
                            "not in the values (left out before or not found)"})
    keep = ~values.apply(lambda r: r["Subject"] in excluded.get(r["Dataset"], []), axis=1)
    values = values[keep].reset_index(drop=True)

    figures = out_root / "Boxplots"
    figures.mkdir(parents=True, exist_ok=True)
    friedman, pairwise, counts = [], [], {}
    for measure in MEASURES:
        fr, pw = tests(values, measure)
        friedman.append(fr)
        pairwise.append(pw)
        counts[measure] = [draw(values, fr, pw, measure, points, figures) for points in (True, False)]
    friedman = pd.concat(friedman, ignore_index=True)
    pairwise = pd.concat(pairwise, ignore_index=True)
    wide = values.pivot_table(index=["Dataset", "Subject"], columns=["Measure", "Model"], values="Value",
                              aggfunc="first")
    wide.columns = [f"{m}__{BOX_LABEL[model]}" for m, model in wide.columns]
    sources = values.drop_duplicates(["Dataset", "Model", "Measure"])[["Dataset", "Model", "Measure", "Source"]]
    tables = {"Friedman": friedman, "Pairwise": pairwise, "Values": wide.reset_index(), "Sources": sources,
              "Excluded": pd.DataFrame(removed)}
    with pd.ExcelWriter(out_root / "261107_Boxplot_Tests.xlsx", engine="openpyxl") as writer:
        for name, frame in tables.items():
            frame.to_excel(writer, sheet_name=name, index=False)
    for name, frame in tables.items():
        frame.to_csv(figures / f"{name}.csv", index=False, encoding="utf-8-sig")
    pd.set_option("display.width", 220)
    print("\nBox plots (subjects with an MAE issue left out):")
    print(friedman[["Measure", "Dataset", "N_Subjects_All_Three", "P", "Kendall_W", "Median_RRWaveNet",
                    "Median_Corr-encoder", "Median_RespDiff"]].round(4).to_string(index=False))
    sig = pairwise[pairwise["FDR_within_dataset"] < 0.05]
    print(sig[["Measure", "Dataset", "A", "B", "N_Pairs", "FDR_within_dataset", "Stars", "Better"]]
          .round(5).to_string(index=False) if not sig.empty else "no significant pair")
    print("significant pairs drawn (with points, without):", counts)
    print(f"Box plots: {figures}; tests: {out_root / '261107_Boxplot_Tests.xlsx'}")


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="261107: waveforms and box plots of the paper-version deep "
                                                 "corr-encoder, RespDiff and RRWaveNet (261007 CapnoBase, 261013 "
                                                 "BIDMC / STEAM2)")
    for run, config in RUNS.items():
        parser.add_argument(f"--root-{run}", type=Path, default=config["root"],
                            help=f"results folder of {run} (default {config['root']})")
    parser.add_argument("--output-root", type=Path, default=PROJECT_ROOT)
    parser.add_argument("--exclude-steam2", nargs="*", default=list(EXCLUDED["STEAM2"]))
    parser.add_argument("--exclude-bidmc", nargs="*", default=list(EXCLUDED["BIDMC"]))
    parser.add_argument("--exclude-capnobase", nargs="*", default=list(EXCLUDED["CapnoBase"]))
    wave = parser.add_argument_group("waveforms (part 1)")
    wave.add_argument("--no-waveforms", action="store_true",
                      help="box plots only, from Waveform_Summary_all_runs.xlsx of an earlier run")
    wave.add_argument("--sizes", nargs="+", type=int, default=list(WINDOW_SIZES), help="window sizes (s)")
    wave.add_argument("--step", type=float, default=STEP_SEC, help="a window starts every this many s")
    wave.add_argument("--subjects", nargs="+", default=None, help="only these subjects")
    wave.add_argument("--respdiff-root", type=Path, default=RESPDIFF_ROOT)
    wave.add_argument("--respdiff-sampling", default=RESPDIFF_SAMPLING)
    wave.add_argument("--correncoder-root", type=Path, default=CORRENCODER_FOLDER,
                      help="results folder of 261107_DeepCorrencoder_PaperVersion_v1 (its outputs for the pages, "
                           "its stored results for the box plots)")
    wave.add_argument("--rrwavenet-lowpass-hz", type=float, default=RRWAVENET_LOWPASS_HZ)
    wave.add_argument("--scripts-dir", type=Path, default=None,
                      help="folder of 261007_RRWaveNet_Ensemble_NoEvent.py (CapnoBase breath labels)")
    wave.add_argument("--capnobase-path", type=Path, default=None)
    wave.add_argument("--bidmc-path", type=Path, default=None)
    wave.add_argument("--steam2-ppg-dir", type=Path, default=None)
    wave.add_argument("--steam2-rsp-dir", type=Path, default=None)
    box = parser.add_argument_group("box plots (part 2)")
    box.add_argument("--waveform-summary", type=Path, default=None,
                     help="Waveform_Summary_all_runs.xlsx (default: the one in the output folder)")
    box.add_argument("--duty-results", type=Path, default=DUTY_RESULTS, help="261107_DutyCycle_Results.xlsx")
    box.add_argument("--correncoder-results", type=Path, default=None,
                     help=f"the corr-encoder's stored results (default: {{correncoder-root}}\\{CORRENCODER_RESULTS_FILE})")
    args = parser.parse_args(argv)
    if args.correncoder_results is None:
        args.correncoder_results = Path(args.correncoder_root) / CORRENCODER_RESULTS_FILE
    args.root_261105 = None  # the CapnoBase reader is looked for in the 261007 folder (and the usual places)
    args.excluded = {"STEAM2": [norm_id("STEAM2", s) for s in args.exclude_steam2],
                     "BIDMC": [norm_id("BIDMC", s) for s in args.exclude_bidmc],
                     "CapnoBase": [norm_id("CapnoBase", s) for s in args.exclude_capnobase]}
    if SIZE_SEC not in args.sizes:
        args.sizes = sorted(set(args.sizes) | {SIZE_SEC})
        print(f"--sizes: {SIZE_SEC} s added (the box plots use the {SIZE_SEC} s windows)")
    return args


def main(argv: Optional[Sequence[str]] = None) -> None:
    args = parse_args(argv)
    out_root = Path(args.output_root)
    out_root.mkdir(parents=True, exist_ok=True)
    print(f"Output: {out_root}\nCorr-encoder: {args.correncoder_root} (box plots: {args.correncoder_results})\n"
          f"RespDiff: {args.respdiff_root}")
    print("Left out: " + "; ".join(f"{d} {', '.join(s)}" for d, s in args.excluded.items() if s))
    if not args.no_waveforms:
        waveforms(args, out_root)
    boxplots(args, out_root, Path(args.waveform_summary or out_root / "Waveform_Summary_all_runs.xlsx"))


if __name__ == "__main__":
    main()
