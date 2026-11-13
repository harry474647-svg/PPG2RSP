r"""261113_RRWaveNet_Rescore_ReferenceGT: the stored RRWaveNet results of 261013
and 261007 scored again against another respiratory-rate ground truth. Nothing is
trained: the stored test outputs (predicted waveforms of every held-out subject)
are read and every 60 s minute is scored once more with the scoring code of the
script that produced them; only the reference breaths change.

Jobs and result folders
-----------------------
1. 261013, BIDMC (D:\PPG2RSP_RRWaveNet_Inspired\261013_RRWaveNet_Adult_SubjectVal_WD):
   ground truth = the breaths of annotator 1 (bidmc_data.mat,
   data(i).ref.breaths.ann1, manual annotations on the impedance signal).
   261013 used breaths detected by count-orig on the band-passed impedance
   signal and kept a minute only when it passed the RRest agreement rule.
   With ann1 the expert-label rule of 261013 (the one used for the CapnoBase
   labels) applies: a minute is scored when its reference signal is usable
   (structural fraction >= 0.5), it holds at least two ann1 breaths and the
   rate 60 / mean ann1 interval lies in the rate band (6-30 breaths/min);
   ground truth = number of ann1 breaths in the minute.
   ann1 leaves some breaths unannotated (for example bidmc_46, whose first
   ann1 breath is at 35 s, and bidmc_15/19/34/40/48, whose ann1 intervals
   reach 3-4 times the median where annotator 2 marked breaths). In such a
   minute the breath count is too low. The results are therefore given twice:
   "ann1" (every minute the rule above keeps) and "ann1_nogap" (in addition,
   minutes that overlap an ann1 gap are left out; gap = an interval between
   consecutive ann1 breaths, or from the recording start to the first / from
   the last breath to the recording end, longer than ANN1_GAP_FACTOR x the
   subject's median ann1 interval).
2. 261013, CapnoBase (D:\PPG2RSP_RRWaveNet_Inspired\261112_RRWaveNet_261013_CapnoBase,
   the 261013 code run on CapnoBase): ground truth = breaths detected by peak
   detection on the capnogram (count-orig on the band-passed, delay-aligned
   capnogram, the detector 261013 applies to the BIDMC and STEAM2
   references, run once on the whole recording); a minute is scored when it
   passes the same RRest agreement rule as the BIDMC and STEAM2 minutes of
   261013 (count-orig rate and FFT rate of the reference within 2
   breaths/min, regular cycles, rate in the band 3-60 breaths/min). The
   original ground truth is the expert CO2 breath labels. Expert CO2
   artifact spans still exclude a minute (as in the original scoring).
3. 261007, CapnoBase (D:\PPG2RSP_RRWaveNet_Inspired\261007_RRWaveNet_NoEvent_v1):
   as job 2 for the 261007 results (variant "full", every window
   configuration with stored results, normally 10 s and 20 s).

Every job also scores the minutes with the original ground truth through the
same code ("original"); these rows must match the stored results of the
source (reproduction_check.csv compares them minute by minute) and are the
baseline of the comparison.

Source scripts
--------------
The preprocessing and the minute scoring of the script that produced the
results are used (its data paths included), so that script is loaded as a
module (its main is not run). It is found, in this order, at:
SCRIPT_PATHS / --script-* (when set); the "script" path written in the
run_manifest.json of the results folder (the script that produced it); the
folder of this script and its parent; the results folder and its parent;
every folder directly under WORKSPACE_ROOT. The script used is printed and
listed in Summary\jobs.csv.

Models scored (as in the source summaries)
------------------------------------------
* ensemble: mean of the members' outputs (261013: the members of the weight
  decay selected per fold, from WD_Selection; 261007: variant "full");
* single models: every member scored alone; metrics averaged over members.

Metrics (the 261013 definitions, mean +- SD with ddof=1)
---------------------------------------------------------
rate = largest FFT peak of the predicted waveform in the rate band (stored
scoring, unchanged); FFT MAE +- SD over the minutes, RMSE, bias +- SD, LoA,
Pearson r, within 1 / 2 breaths/min, subject MAE mean +- SD, Davies & Mandic
mAE and mMAE with IQR, waveform PCC. Paired Wilcoxon test of the subject MAE
(ensemble) between the original and the new ground truth.

Outputs (PROJECT_ROOT = D:\PPG2RSP_RRWaveNet_Inspired\261113_RRWaveNet_Rescore_ReferenceGT)
  Minutes\{source}_{dataset}_{window}_minutes.csv   every minute, model and
      ground truth (reference validity, exclusion reason, GT count, rate, AE)
  Summary\overall.csv, overall_mean_sd.csv, exclusions.csv, per_subject.csv,
      gt_paired_tests.csv, reproduction_check.csv, missing.csv, jobs.csv and
      the workbook 261113_RRWaveNet_Rescore_ReferenceGT.xlsx
  run_manifest.json

Run (Spyder or a terminal):   python 261113_RRWaveNet_Rescore_ReferenceGT.py
One job:                      --jobs 261013_bidmc_ann1
Other result folders:         --root-261013 ..., --root-261013-capnobase ...,
                              --root-261007 ...
Source script given by hand:  --script-261013-bidmc ..., --script-261013-capnobase ...,
                              --script-261007 ...
"""

import argparse
import importlib.util
import json
import re
import sys
import warnings
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import scipy.io as sio
from scipy import stats


# ---------------------------------------------------------------------------
# Paths and settings
# ---------------------------------------------------------------------------

HERE = Path(__file__).resolve().parent
WORKSPACE_ROOT = Path(r"D:\PPG2RSP_RRWaveNet_Inspired")
PROJECT_ROOT = WORKSPACE_ROOT / "261113_RRWaveNet_Rescore_ReferenceGT"
ROOT_261013 = WORKSPACE_ROOT / "261013_RRWaveNet_Adult_SubjectVal_WD"
ROOT_261013_CAPNOBASE = WORKSPACE_ROOT / "261112_RRWaveNet_261013_CapnoBase"
ROOT_261007 = WORKSPACE_ROOT / "261007_RRWaveNet_NoEvent_v1"
# Source script of a job; None = found automatically (see the docstring).
SCRIPT_PATHS: Dict[str, Optional[str]] = {
    "261013_bidmc_ann1": None,
    "261013_capnobase_peak": None,
    "261007_capnobase_peak": None,
}
FINAL_VARIANT_261007 = "full"
MEMBERS = 3
ANN1_GAP_FACTOR = 2.5  # ann1 gap: interval > 2.5 x the subject's median ann1 interval
BIDMC_LABEL_INDEX_BASE = 1  # ann1 holds MATLAB (one-based) sample numbers
EVAL_WINDOW_SEC = 60.0  # scored minute (as in 261007 / 261013)

JOBS = {
    "261013_bidmc_ann1": {
        "source": "261013", "dataset": "bidmc", "new_gt": "ann1",
        "scripts": ("261013_RRWaveNet_Adult_SubjectVal_WD.py",),
    },
    "261013_capnobase_peak": {
        "source": "261013", "dataset": "capnobase", "new_gt": "peak",
        # 261112_RRWaveNet_261013_CapnoBase is the 261013 code for CapnoBase.
        "scripts": ("261112_RRWaveNet_261013_CapnoBase.py", "261013_RRWaveNet_Adult_SubjectVal_WD.py"),
    },
    "261007_capnobase_peak": {
        "source": "261007", "dataset": "capnobase", "new_gt": "peak",
        "scripts": ("261007_RRWaveNet_Ensemble_NoEvent.py",),
    },
}
GT_DESCRIPTIONS = {
    ("bidmc", "original"): "261013: breaths detected by count-orig on the impedance signal, RRest agreement rule",
    ("bidmc", "ann1"): "ann1 expert breaths (count in the minute), expert-label rule of 261013",
    ("bidmc", "ann1_nogap"): "ann1 expert breaths, minutes overlapping an ann1 gap left out",
    ("capnobase", "original"): "expert CO2 breath labels (count in the minute), expert-label rule",
    ("capnobase", "peak"): "breaths detected by count-orig peak detection on the capnogram, RRest agreement rule",
}
MODEL_ROWS = (("ensemble", "Ensemble (final)"), ("single", "Single models (mean of members)"))


# ---------------------------------------------------------------------------
# Source scripts
# ---------------------------------------------------------------------------

_MODULES: Dict[str, object] = {}


def job_root(args: argparse.Namespace, job_name: str) -> Path:
    if job_name == "261013_bidmc_ann1":
        return Path(args.root_261013)
    if job_name == "261013_capnobase_peak":
        return Path(args.root_261013_capnobase)
    return Path(args.root_261007)


def explicit_script(args: argparse.Namespace, job_name: str) -> Optional[str]:
    given = {
        "261013_bidmc_ann1": args.script_261013_bidmc,
        "261013_capnobase_peak": args.script_261013_capnobase,
        "261007_capnobase_peak": args.script_261007,
    }[job_name]
    return str(given) if given else SCRIPT_PATHS.get(job_name)


def manifest_scripts(root: Path) -> List[Path]:
    """Script paths written in the run manifests of a results folder."""
    paths = []
    for manifest in sorted(root.glob("run_manifest*.json")):
        try:
            script = json.loads(manifest.read_text(encoding="utf-8")).get("script")
        except (OSError, ValueError, AttributeError):
            continue
        if script:
            paths.append(Path(script))
    return paths


def find_script(args: argparse.Namespace, job_name: str, job: dict, root: Path) -> Path:
    """The source script of a job (see the docstring for the search order)."""
    candidates: List[Path] = []
    given = explicit_script(args, job_name)
    if given:
        candidates.append(Path(given))
    candidates.extend(path for path in manifest_scripts(root) if path.name in job["scripts"])
    folders = [HERE, HERE.parent, root, root.parent, WORKSPACE_ROOT]
    if WORKSPACE_ROOT.is_dir():
        folders.extend(sorted(path for path in WORKSPACE_ROOT.iterdir() if path.is_dir()))
    for name in job["scripts"]:
        candidates.extend(folder / name for folder in folders)
    for path in candidates:
        if path.is_file():
            return path.resolve()
    raise FileNotFoundError(
        "{}: none of {} found (looked at the run_manifest.json of {}, {}, {}, {}, and the folders under {}). "
        "Set SCRIPT_PATHS['{}'] at the top of this script or pass --script-....".format(
            job_name, ", ".join(job["scripts"]), root, HERE, HERE.parent, root, WORKSPACE_ROOT, job_name)
    )


def load_script(path: Path):
    """A source script loaded as a module (its main is not run); cached by path."""
    key = str(path)
    if key not in _MODULES:
        name = "rescore_src_" + re.sub(r"\W", "_", path.stem)
        spec = importlib.util.spec_from_file_location(name, key)
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module
        spec.loader.exec_module(module)
        _MODULES[key] = module
    return _MODULES[key]


def stored_specs(module, root: Path, dataset_key: str, windows: Optional[Sequence[str]]) -> list:
    """Window configurations with a stored results folder for the dataset."""
    specs = []
    for spec in module.WINDOW_SPECS:
        if windows and spec.name not in windows:
            continue
        if any(module.condition_root(root, name, spec, dataset_key).exists() for name in module.VARIANTS):
            specs.append(spec)
    return specs


def final_variant(module, source: str, root: Path, spec, dataset_key: str, subject: str):
    """The variant whose members form the final ensemble of a fold."""
    if source == "261007":
        return module.VARIANTS.get(FINAL_VARIANT_261007)
    found = module.stored_selection(root, list(module.VARIANTS.values()), spec, dataset_key, subject)
    return None if found is None else found[1]


def stored_outputs(module, root: Path, variant, spec, dataset_key: str, subject: str,
                   members: int) -> Tuple[Optional[Dict[str, dict]], str]:
    """Member outputs and their ensemble (mean of the members, as in the
    source's evaluate_ensemble)."""
    condition = module.condition_root(root, variant.name, spec, dataset_key)
    arrays = []
    for member in range(members):
        path = module.member_paths(condition, subject, member)[2]
        if not path.exists():
            return None, f"member m{member} outputs missing ({variant.name})"
        with np.load(path, allow_pickle=False) as data:
            arrays.append({key: data[key] for key in ("predictions", "targets", "segment_ids", "window_starts")})
    for other in arrays[1:]:
        if not (np.array_equal(other["segment_ids"], arrays[0]["segment_ids"])
                and np.array_equal(other["window_starts"], arrays[0]["window_starts"])):
            return None, "members scored different windows"
    outputs = {f"m{member}": arrays[member] for member in range(members)}
    outputs["ensemble"] = {
        "predictions": np.mean([a["predictions"] for a in arrays], axis=0).astype(np.float32),
        "targets": arrays[0]["targets"],
        "segment_ids": arrays[0]["segment_ids"],
        "window_starts": arrays[0]["window_starts"],
    }
    return outputs, ""


# ---------------------------------------------------------------------------
# Reference breaths
# ---------------------------------------------------------------------------

def read_bidmc_ann1(module) -> Dict[str, np.ndarray]:
    """ann1 of every BIDMC record (sample numbers at the BIDMC rate), keyed by
    the subject names of the source script (01, 02, ...)."""
    mat = sio.loadmat(module.DATASET_CONFIGS["bidmc"]["data_path"])
    out = {}
    for index, record in enumerate(mat["data"][0]):
        breaths = record["ref"][0, 0]["breaths"][0, 0]
        out[f"{index + 1:02d}"] = np.asarray(breaths["ann1"], dtype=np.float64).ravel()
    return out


def ann1_on_timeline(module, ann1: np.ndarray, reference: dict) -> np.ndarray:
    """ann1 as sample indices of the reference timeline (TARGET_FS, the
    segment's target lag removed), as the source maps expert labels."""
    fs_in = float(module.DATASET_CONFIGS["bidmc"]["default_fs"])
    lag = int(round(float(reference["diagnostics"].get("Target_Lag_Sec", 0.0)) * module.TARGET_FS))
    samples = (np.asarray(ann1, dtype=np.float64) - BIDMC_LABEL_INDEX_BASE) * (module.TARGET_FS / fs_in)
    labels = np.round(samples).astype(np.int64) - lag
    length = len(reference["signal"])
    return np.unique(labels[(labels >= 0) & (labels < length)])


def gap_spans(labels: np.ndarray, length: int, factor: float) -> List[Tuple[int, int]]:
    """Spans (start, end) of the timeline without ann1 breaths that are longer
    than factor x the median ann1 interval (also before the first and after
    the last breath)."""
    if factor <= 0 or len(labels) < 3:
        return []
    limit = factor * float(np.median(np.diff(labels)))
    spans = []
    if labels[0] > limit:
        spans.append((0, int(labels[0])))
    for left, right in zip(labels[:-1], labels[1:]):
        if right - left > limit:
            spans.append((int(left) + 1, int(right)))
    if length - 1 - labels[-1] > limit:
        spans.append((int(labels[-1]) + 1, int(length)))
    return spans


def reference_variant(references: Dict[int, dict], gt: str,
                      ann1_labels: Optional[Dict[int, np.ndarray]] = None) -> Dict[int, dict]:
    """References of one subject with the breaths of a ground truth."""
    if gt == "original":
        return references
    out = {}
    for segment_id, reference in references.items():
        copy = dict(reference)
        if gt == "ann1":
            copy["breaths"] = ann1_labels[segment_id]
            copy["label_based"] = True
        elif gt == "peak":
            copy["breaths"] = np.asarray(reference["detected_breaths"], dtype=np.int64)
            copy["label_based"] = False
        else:
            raise ValueError(gt)
        out[segment_id] = copy
    return out


# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------

def score(module, outputs: dict, metadata: list, references: Dict[int, dict]) -> pd.DataFrame:
    """Minutes of one held-out subject scored by the source's own code."""
    _, frame = module.evaluate_subject_minutes(
        {"predictions": outputs["predictions"], "targets": outputs["targets"]},
        metadata,
        references,
        module.TARGET_FS,
    )
    return frame


def flag_gap_minutes(frame: pd.DataFrame, spans_by_segment: Dict[int, List[Tuple[int, int]]], fs: int) -> pd.Series:
    flags = []
    for _, row in frame.iterrows():
        start = int(round(float(row["RR_Window_Start_Sec"]) * fs))
        end = int(round(float(row["RR_Window_End_Sec"]) * fs))
        spans = spans_by_segment.get(int(row["Segment_ID"]), [])
        flags.append(any(lo < end and hi > start for lo, hi in spans))
    return pd.Series(flags, index=frame.index, dtype=bool)


def reproduction_row(module, root: Path, variant, spec, dataset_key: str, subject: str,
                     frame: pd.DataFrame) -> dict:
    """Original-GT ensemble minutes rescored here vs the stored ensemble table."""
    condition = module.condition_root(root, variant.name, spec, dataset_key)
    path = module.ensemble_paths(condition, subject)[1]
    row = {"Subject": subject, "Stored_Table": str(path)}
    if not path.exists():
        row["Status"] = "no stored ensemble table"
        return row
    stored = pd.read_csv(path, encoding="utf-8-sig")
    key = ["Segment_ID", "RR_Window_Start_Sec"]
    merged = stored.merge(frame, on=key, how="outer", suffixes=("_stored", "_rescored"), indicator=True)
    both = merged[merged["_merge"] == "both"]
    rate_diff = (pd.to_numeric(both["Pred_Wave_FFT_RR_stored"], errors="coerce")
                 - pd.to_numeric(both["Pred_Wave_FFT_RR_rescored"], errors="coerce")).abs()
    count_diff = (pd.to_numeric(both["GT_Breath_Count_stored"], errors="coerce")
                  - pd.to_numeric(both["GT_Breath_Count_rescored"], errors="coerce")).abs()
    valid_mismatch = int((both["Ref_Valid_stored"].astype(bool) != both["Ref_Valid_rescored"].astype(bool)).sum())
    row.update({
        "Minutes_Stored": int(len(stored)),
        "Minutes_Rescored": int(len(frame)),
        "Minutes_Unmatched": int((merged["_merge"] != "both").sum()),
        "Max_Abs_Diff_Pred_Rate": float(rate_diff.max()) if len(rate_diff) else np.nan,
        "Max_Abs_Diff_GT_Count": float(count_diff.max()) if len(count_diff) else np.nan,
        "Ref_Valid_Mismatches": valid_mismatch,
    })
    row["Status"] = "match" if (
        row["Minutes_Unmatched"] == 0 and valid_mismatch == 0
        and np.nan_to_num(row["Max_Abs_Diff_Pred_Rate"]) < 1e-6
        and np.nan_to_num(row["Max_Abs_Diff_GT_Count"]) == 0
    ) else "differs"
    return row


def run_job(args: argparse.Namespace, job_name: str, job: dict) -> Tuple[List[pd.DataFrame], List[dict], List[dict], List[dict]]:
    """Rescore one job; returns minute tables, reproduction rows, missing rows
    and the description rows of the job."""
    frames: List[pd.DataFrame] = []
    repro: List[dict] = []
    missing: List[dict] = []
    info: List[dict] = []
    root = job_root(args, job_name)
    if not root.is_dir():
        print(f"[{job_name}] results folder not found: {root}")
        missing.append({"Job": job_name, "Subject": "", "Missing": f"results folder not found: {root}"})
        return frames, repro, missing, info
    script = find_script(args, job_name, job, root)
    print(f"[{job_name}] source script: {script}")
    module = load_script(script)
    dataset_key = job["dataset"]
    label = module.DATASET_CONFIGS[dataset_key]["label"]
    if args.bidmc_path:
        module.DATASET_CONFIGS["bidmc"]["data_path"] = str(args.bidmc_path)
    if args.capnobase_path:
        module.DATASET_CONFIGS["capnobase"]["data_path"] = str(args.capnobase_path)
    specs = stored_specs(module, root, dataset_key, args.windows)
    if not specs:
        print(f"[{job_name}] no stored {label} results in {root}")
        missing.append({"Job": job_name, "Subject": "", "Missing": f"no stored {label} results in {root}"})
        return frames, repro, missing, info
    gts = ["original", job["new_gt"]]
    ann1 = read_bidmc_ann1(module) if job["new_gt"] == "ann1" else None
    for spec in specs:
        print(f"[{job_name}] {label} {spec.name}: results {root}")
        info.append({"Job": job_name, "Source": job["source"], "Dataset": label, "Window_Config": spec.name,
                     "Results_Root": str(root), "Script": str(script),
                     "Data_Path": str(module.DATASET_CONFIGS[dataset_key].get("data_path", ""))})
        subject_data = module.load_dataset(dataset_key, spec)
        subjects = [s for s in subject_data if not args.subjects or s in args.subjects]
        for subject in subjects:
            variant = final_variant(module, job["source"], root, spec, dataset_key, subject)
            if variant is None:
                missing.append({"Job": job_name, "Window_Config": spec.name, "Subject": subject,
                                "Missing": "weight decay selection / final variant"})
                continue
            outputs, problem = stored_outputs(module, root, variant, spec, dataset_key, subject, args.members)
            if outputs is None:
                missing.append({"Job": job_name, "Window_Config": spec.name, "Subject": subject, "Missing": problem})
                continue
            metadata = module.test_metadata(subject_data, subject)
            if not (np.array_equal(outputs["ensemble"]["segment_ids"], [m["segment_id"] for m in metadata])
                    and np.array_equal(outputs["ensemble"]["window_starts"], [m["start"] for m in metadata])):
                missing.append({"Job": job_name, "Window_Config": spec.name, "Subject": subject,
                                "Missing": "stored windows differ from the windows of the data read now"})
                continue
            references = subject_data[subject]["references"]
            ann1_labels, spans = None, {}
            if ann1 is not None:
                if subject not in ann1 or len(ann1[subject]) < 2:
                    missing.append({"Job": job_name, "Window_Config": spec.name, "Subject": subject,
                                    "Missing": "no ann1 breaths"})
                    continue
                ann1_labels = {sid: ann1_on_timeline(module, ann1[subject], ref) for sid, ref in references.items()}
                spans = {sid: gap_spans(ann1_labels[sid], len(references[sid]["signal"]), args.ann1_gap_factor)
                         for sid in references}
            for gt in gts:
                refs = reference_variant(references, gt, ann1_labels)
                for model_name, model_outputs in outputs.items():
                    frame = score(module, model_outputs, metadata, refs)
                    if frame.empty:
                        continue
                    if gt == "original" and model_name == "ensemble":
                        row = reproduction_row(module, root, variant, spec, dataset_key, subject, frame)
                        row.update({"Job": job_name, "Window_Config": spec.name})
                        repro.append(row)
                    frame.insert(0, "Member", -1 if model_name == "ensemble" else int(model_name[1:]))
                    frame.insert(0, "Model", model_name)
                    frame.insert(0, "Variant", variant.name)
                    frame.insert(0, "Subject", str(subject))
                    frame.insert(0, "GT", gt)
                    frame.insert(0, "Window_Config", spec.name)
                    frame.insert(0, "Dataset", label)
                    frame.insert(0, "Source", job["source"])
                    frame.insert(0, "Job", job_name)
                    frames.append(frame)
                    if gt == "ann1":
                        frame["Ann1_Gap_Minute"] = flag_gap_minutes(frame, spans, module.TARGET_FS)
                        nogap = frame.copy()
                        nogap["GT"] = "ann1_nogap"
                        drop = nogap["Ann1_Gap_Minute"] & nogap["Ref_Valid"].astype(bool)
                        nogap.loc[drop, "Ref_Valid"] = False
                        nogap.loc[drop, "Ref_Exclusion_Reason"] = "ref_ann1_gap"
                        frames.append(nogap)
        del subject_data
    return frames, repro, missing, info


# ---------------------------------------------------------------------------
# Metrics (the 261013 definitions)
# ---------------------------------------------------------------------------

SUMMARY_METRICS = (
    "FFT_MAE", "FFT_MAE_SD", "FFT_RMSE", "FFT_Bias", "FFT_Bias_SD", "FFT_LoA_Lower", "FFT_LoA_Upper",
    "FFT_Pearson_r", "FFT_Within_1_BPM_Pct", "FFT_Within_2_BPM_Pct", "FFT_Subject_MAE_Mean",
    "FFT_Subject_MAE_SD", "Davies_mAE", "Davies_mAE_Q1", "Davies_mAE_Q3", "Davies_mMAE",
    "Davies_mMAE_Q1", "Davies_mMAE_Q3", "Wave_PCC_Mean", "Wave_PCC_SD", "Wave_PCC_Median",
)


def sd(values) -> float:
    """Sample SD (ddof=1) of the finite values."""
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    return float(np.std(values, ddof=1)) if values.size > 1 else np.nan


def pearson(x: np.ndarray, y: np.ndarray) -> float:
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    keep = np.isfinite(x) & np.isfinite(y)
    x, y = x[keep], y[keep]
    if x.size < 2 or np.std(x) < 1e-8 or np.std(y) < 1e-8:
        return np.nan
    return float(stats.pearsonr(x, y)[0])


def scored_minutes(frame: pd.DataFrame) -> pd.DataFrame:
    """Reference-valid minutes with the FFT rate error (breaths/min)."""
    if frame.empty:
        return frame.copy()
    valid = frame[frame["Ref_Valid"].astype(bool)].copy()
    valid["Subject"] = valid["Subject"].astype(str)
    valid["GT_RR"] = pd.to_numeric(valid["GT_Breath_Count"], errors="coerce") * (60.0 / EVAL_WINDOW_SEC)
    valid["FFT_RR"] = pd.to_numeric(valid["Pred_Wave_FFT_RR"], errors="coerce")
    valid["FFT_Error"] = valid["FFT_RR"] - valid["GT_RR"]
    valid["FFT_AE"] = valid["FFT_Error"].abs()
    return valid


def fft_davies_metrics(minutes: pd.DataFrame) -> dict:
    """FFT metrics over all minutes and the Davies & Mandic (2022) medians
    (mAE: median absolute error over the minutes; mMAE: median over subjects
    of each subject's mean absolute error)."""
    result = {"N_Subjects": 0, "N_Minutes": 0, "N_Minutes_No_FFT": 0}
    result.update({metric: np.nan for metric in SUMMARY_METRICS})
    if minutes.empty:
        return result
    pcc = pd.to_numeric(minutes["Wave_PCC"], errors="coerce").dropna()
    if len(pcc):
        result["Wave_PCC_Mean"] = float(pcc.mean())
        result["Wave_PCC_SD"] = sd(pcc)
        result["Wave_PCC_Median"] = float(pcc.median())
    have = minutes[np.isfinite(minutes["FFT_AE"])]
    result["N_Minutes"] = int(len(minutes))
    result["N_Minutes_No_FFT"] = int(len(minutes) - len(have))
    if have.empty:
        return result
    errors = have["FFT_Error"].to_numpy(dtype=float)
    absolute = np.abs(errors)
    subject_mae = have.groupby("Subject")["FFT_AE"].mean()
    error_sd = sd(errors)
    result.update({
        "N_Subjects": int(subject_mae.size),
        "FFT_MAE": float(np.mean(absolute)),
        "FFT_MAE_SD": sd(absolute),
        "FFT_RMSE": float(np.sqrt(np.mean(errors ** 2))),
        "FFT_Bias": float(np.mean(errors)),
        "FFT_Bias_SD": error_sd,
        "FFT_LoA_Lower": float(np.mean(errors) - 1.96 * error_sd),
        "FFT_LoA_Upper": float(np.mean(errors) + 1.96 * error_sd),
        "FFT_Pearson_r": pearson(have["FFT_RR"].to_numpy(dtype=float), have["GT_RR"].to_numpy(dtype=float)),
        "FFT_Within_1_BPM_Pct": float(np.mean(absolute <= 1.0) * 100.0),
        "FFT_Within_2_BPM_Pct": float(np.mean(absolute <= 2.0) * 100.0),
        "FFT_Subject_MAE_Mean": float(subject_mae.mean()),
        "FFT_Subject_MAE_SD": sd(subject_mae),
        "Davies_mAE": float(np.median(absolute)),
        "Davies_mAE_Q1": float(np.percentile(absolute, 25)),
        "Davies_mAE_Q3": float(np.percentile(absolute, 75)),
        "Davies_mMAE": float(subject_mae.median()),
        "Davies_mMAE_Q1": float(subject_mae.quantile(0.25)),
        "Davies_mMAE_Q3": float(subject_mae.quantile(0.75)),
    })
    return result


def member_average_metrics(minutes: pd.DataFrame) -> dict:
    """Metrics of each single model (member), averaged over the members."""
    rows = [fft_davies_metrics(frame) for _, frame in minutes.groupby("Member", sort=True)] if not minutes.empty else []
    if not rows:
        return fft_davies_metrics(pd.DataFrame())
    table = pd.DataFrame(rows)
    result = {
        "N_Subjects": int(table["N_Subjects"].max()),
        "N_Minutes": int(table["N_Minutes"].max()),
        "N_Minutes_No_FFT": float(table["N_Minutes_No_FFT"].mean()),
        "N_Members": int(len(table)),
    }
    for metric in SUMMARY_METRICS:
        result[metric] = float(table[metric].mean())
        result[f"{metric}_SD_Members"] = float(table[metric].std(ddof=1)) if len(table) > 1 else np.nan
    return result


# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------

def mean_sd(mean: float, spread: float, digits: int = 3) -> str:
    if not np.isfinite(mean):
        return ""
    if not np.isfinite(spread):
        return f"{mean:.{digits}f}"
    return f"{mean:.{digits}f} ± {spread:.{digits}f}"


def summarize(minutes: pd.DataFrame) -> Dict[str, pd.DataFrame]:
    overall_rows, exclusion_rows, subject_rows, test_rows = [], [], [], []
    group_keys = ["Job", "Source", "Dataset", "Window_Config"]
    for keys, group in minutes.groupby(group_keys, sort=False):
        base = dict(zip(group_keys, keys))
        dataset_key = "bidmc" if base["Dataset"] == "BIDMC" else "capnobase"
        subject_mae = {}
        for gt, gt_frame in group.groupby("GT", sort=False):
            ensemble = scored_minutes(gt_frame[gt_frame["Model"] == "ensemble"].drop(columns=["Member"]))
            single = scored_minutes(gt_frame[gt_frame["Model"] != "ensemble"])
            for model_key, model_label in MODEL_ROWS:
                result = fft_davies_metrics(ensemble) if model_key == "ensemble" else member_average_metrics(single)
                overall_rows.append({**base, "GT": gt, "GT_Description": GT_DESCRIPTIONS.get((dataset_key, gt), gt),
                                     "Model": model_label, **result})
            all_ensemble = gt_frame[gt_frame["Model"] == "ensemble"]
            reasons = all_ensemble.loc[~all_ensemble["Ref_Valid"].astype(bool), "Ref_Exclusion_Reason"].value_counts()
            exclusion_rows.append({**base, "GT": gt, "N60": int(len(all_ensemble)),
                                   "N60_RefValid": int(all_ensemble["Ref_Valid"].astype(bool).sum()),
                                   **{f"Excluded_{reason}": int(count) for reason, count in reasons.items()}})
            have = ensemble[np.isfinite(ensemble["FFT_AE"])] if not ensemble.empty else ensemble
            per_subject = (have.groupby("Subject")["FFT_AE"].agg(["mean", "size"]) if not have.empty
                           else pd.DataFrame(columns=["mean", "size"]))
            have_single = single[np.isfinite(single["FFT_AE"])] if not single.empty else single
            single_subject = (have_single.groupby(["Member", "Subject"])["FFT_AE"].mean().groupby("Subject").mean()
                              if not have_single.empty else pd.Series(dtype=float))
            subject_mae[gt] = per_subject["mean"]
            for subject in sorted(set(all_ensemble["Subject"].astype(str))):
                subject_rows.append({
                    **base, "GT": gt, "Subject": subject,
                    "N60": int((all_ensemble["Subject"].astype(str) == subject).sum()),
                    "N60_Scored": int(per_subject["size"].get(subject, 0)),
                    "FFT_MAE_Ensemble": float(per_subject["mean"].get(subject, np.nan)),
                    "FFT_MAE_Single_Mean": float(single_subject.get(subject, np.nan)),
                })
        if "original" in subject_mae:
            for gt, values in subject_mae.items():
                if gt == "original":
                    continue
                pair = pd.concat([subject_mae["original"], values], axis=1, keys=["original", gt]).dropna()
                diff = pair[gt] - pair["original"]
                p_value = np.nan
                if len(pair) > 1:
                    if np.allclose(diff, 0.0):
                        p_value = 1.0
                    else:
                        try:
                            p_value = float(stats.wilcoxon(pair[gt], pair["original"]).pvalue)
                        except ValueError:
                            p_value = np.nan
                test_rows.append({**base, "GT_New": gt, "GT_Original": "original", "N_Subjects": int(len(pair)),
                                  "Subject_MAE_Original": mean_sd(pair["original"].mean(), pair["original"].std(ddof=1)),
                                  "Subject_MAE_New": mean_sd(pair[gt].mean(), pair[gt].std(ddof=1)),
                                  "Diff_New_minus_Original": mean_sd(diff.mean(), diff.std(ddof=1)),
                                  "Wilcoxon_p": p_value})
    overall = pd.DataFrame(overall_rows)
    text_rows = []
    for _, row in overall.iterrows():
        text_rows.append({
            "Job": row["Job"], "Dataset": row["Dataset"], "Window_Config": row["Window_Config"],
            "GT": row["GT"], "Model": row["Model"],
            "N_Subjects": row.get("N_Subjects"), "N_Minutes": row.get("N_Minutes"),
            "FFT_MAE (minutes)": mean_sd(row.get("FFT_MAE", np.nan), row.get("FFT_MAE_SD", np.nan)),
            "Subject_MAE": mean_sd(row.get("FFT_Subject_MAE_Mean", np.nan), row.get("FFT_Subject_MAE_SD", np.nan)),
            "Bias": mean_sd(row.get("FFT_Bias", np.nan), row.get("FFT_Bias_SD", np.nan)),
            "RMSE": mean_sd(row.get("FFT_RMSE", np.nan), np.nan),
            "Within_2_BPM_Pct": mean_sd(row.get("FFT_Within_2_BPM_Pct", np.nan), np.nan, 1),
            "Davies_mMAE": mean_sd(row.get("Davies_mMAE", np.nan), np.nan),
            "Wave_PCC": mean_sd(row.get("Wave_PCC_Mean", np.nan), row.get("Wave_PCC_SD", np.nan)),
        })
    return {
        "Overall_mean_SD": pd.DataFrame(text_rows),
        "Overall": overall,
        "Exclusions": pd.DataFrame(exclusion_rows).fillna(0),
        "Per_subject": pd.DataFrame(subject_rows),
        "GT_paired_tests": pd.DataFrame(test_rows),
    }


def write_csv(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False, encoding="utf-8-sig")


def write_excel(path: Path, sheets: Dict[str, pd.DataFrame]) -> None:
    try:
        with pd.ExcelWriter(path, engine="openpyxl") as writer:
            for name, frame in sheets.items():
                frame.to_excel(writer, index=False, sheet_name=name[:31])
    except Exception as exc:
        warnings.warn(f"Excel export failed: {exc}")


def run(args: argparse.Namespace) -> None:
    results_root = Path(args.results_root)
    results_root.mkdir(parents=True, exist_ok=True)
    all_frames, repro, missing, info = [], [], [], []
    for job_name in args.jobs:
        try:
            frames, job_repro, job_missing, job_info = run_job(args, job_name, JOBS[job_name])
        except FileNotFoundError as exc:
            print(f"[{job_name}] not run: {exc}")
            missing.append({"Job": job_name, "Subject": "", "Missing": str(exc)})
            continue
        all_frames.extend(frames)
        repro.extend(job_repro)
        missing.extend(job_missing)
        info.extend(job_info)
    sheets: Dict[str, pd.DataFrame] = {}
    if all_frames:
        minutes = pd.concat(all_frames, ignore_index=True, sort=False)
        for keys, group in minutes.groupby(["Source", "Dataset", "Window_Config"], sort=False):
            write_csv(group, results_root / "Minutes" / "{}_{}_{}_minutes.csv".format(*keys))
        sheets = summarize(minutes)
    sheets["Reproduction_check"] = pd.DataFrame(repro)
    sheets["Missing"] = pd.DataFrame(missing)
    sheets["Jobs"] = pd.DataFrame(info)
    sheets["GT_definitions"] = pd.DataFrame(
        [{"Dataset": d, "GT": g, "Description": text} for (d, g), text in GT_DESCRIPTIONS.items()]
    )
    summary_root = results_root / "Summary"
    for name, frame in sheets.items():
        write_csv(frame, summary_root / f"{name.lower()}.csv")
    workbook = summary_root / "261113_RRWaveNet_Rescore_ReferenceGT.xlsx"
    write_excel(workbook, sheets)
    manifest = {
        "script": str(Path(__file__).resolve()),
        "jobs": {name: {**JOBS[name], "results_root": str(job_root(args, name))} for name in args.jobs},
        "source_scripts": [row["Script"] for row in info],
        "members": args.members,
        "ann1_gap_factor": args.ann1_gap_factor,
        "gt_definitions": {f"{d}:{g}": text for (d, g), text in GT_DESCRIPTIONS.items()},
    }
    (results_root / "run_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    print("\nResults (mean ± SD; FFT MAE over minutes, breaths/min)")
    table = sheets.get("Overall_mean_SD")
    if table is not None and not table.empty:
        for _, row in table.iterrows():
            print(f"  {row['Job']:<22} {row['Window_Config']:<16} {row['GT']:<11} {row['Model']:<32} "
                  f"subjects {row['N_Subjects']}, minutes {row['N_Minutes']}: MAE {row['FFT_MAE (minutes)']}, "
                  f"subject MAE {row['Subject_MAE']}")
    repro_frame = sheets["Reproduction_check"]
    if not repro_frame.empty:
        differs = int((repro_frame["Status"] != "match").sum())
        print(f"Reproduction of the stored original-GT ensemble minutes: {len(repro_frame) - differs} subjects match, "
              f"{differs} differ (see reproduction_check.csv)")
    if missing:
        print(f"Not scored: {len(missing)} (see missing.csv)")
    print(f"Saved under: {results_root}")
    print(f"Excel workbook: {workbook}")


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--results-root", type=Path, default=PROJECT_ROOT)
    parser.add_argument("--jobs", nargs="+", choices=list(JOBS), default=list(JOBS))
    parser.add_argument("--root-261013", type=Path, default=ROOT_261013)
    parser.add_argument("--root-261013-capnobase", type=Path, default=ROOT_261013_CAPNOBASE)
    parser.add_argument("--root-261007", type=Path, default=ROOT_261007)
    parser.add_argument("--script-261013-bidmc", type=Path, default=None)
    parser.add_argument("--script-261013-capnobase", type=Path, default=None)
    parser.add_argument("--script-261007", type=Path, default=None)
    parser.add_argument("--bidmc-path", type=Path, default=None)
    parser.add_argument("--capnobase-path", type=Path, default=None)
    parser.add_argument("--windows", nargs="+", default=None, help="window configurations (default: all stored)")
    parser.add_argument("--subjects", nargs="+", default=None)
    parser.add_argument("--members", type=int, default=MEMBERS)
    parser.add_argument("--ann1-gap-factor", type=float, default=ANN1_GAP_FACTOR,
                        help="ann1 gap = interval > factor x median ann1 interval (0: no gap check)")
    return parser.parse_args(argv)


if __name__ == "__main__":
    run(parse_args())
