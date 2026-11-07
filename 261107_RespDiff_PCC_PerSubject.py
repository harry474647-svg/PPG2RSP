# -*- coding: utf-8 -*-
r"""261107_RespDiff_PCC_PerSubject: the waveform PCC of the stored RespDiff
results, per subject, from the RespDiff results folder only (nothing else is
read, nothing is trained).

The RespDiff run (261020_RespDiff code, official RespDiff evaluation) stores
per subject RR_MAE, Waveform_MAE, Segment_MAE and Duty_Cycle_Corr, but no
PCC. This script computes the PCC from what that run stored:
  {respdiff-root}\{model}\{dataset}\Predictions\{subject}_{sampling}.npz
      recording{i}_prediction   the joined RespDiff output, low-pass filtered (as its evaluation)
      recording{i}_reference    the joined reference (5 s segments, each min-max scaled to [0, 1])
  {respdiff-root}\{model}\{dataset}\Fold_Results\{subject}_{sampling}.json (its own scores, copied)
({model} = respdiff; every model folder and sampling found, e.g. ddpm50 and
ddim6, is read; --samplings chooses.)

PCC (prediction vs the reference RespDiff stored, the signals of its own
evaluation; no time shift, no sign change)
  * windows of the RespDiff evaluation: 60 s moved by 20 s over every
    recording (score_recording of 261020_RespDiff); PCC = Pearson r of the
    prediction and the reference in the window;
  * per subject: PCC_Mean / PCC_SD / PCC_Median over all windows of all its
    recordings (as RespDiff pools its windows per subject), N_Windows;
    PCC_Whole = Pearson r over all samples of all its recordings.
  The RespDiff run's own scores of the subject (RR_MAE, RR_MAE_Official_Code,
  Waveform_MAE, Segment_MAE, Duty_Cycle_Corr) are added from its json.

Output (default {respdiff-root}\PCC_Results\261107_RespDiff_PCC_PerSubject.xlsx; nothing of
the RespDiff run is changed)
  Summary      per model, sampling and dataset: Mean ± SD (Median) over subjects
  STEAM2, BIDMC, CapnoBase   per subject: PCC_Mean, PCC_SD, PCC_Median, PCC_Whole,
               N_Windows, Recordings and the run's own scores
  Windows      every 60 s window (recording, start, PCC)
  Notes
  and the same tables as CSV files.

Run:
  python 261107_RespDiff_PCC_PerSubject.py
Options: --respdiff-root (default D:\PPG2RSP_RRWaveNet_Inspired\261020_RespDiff_v2),
  --samplings ddpm50 ddim6, --datasets STEAM2 BIDMC CapnoBase, --output-root
"""

import argparse
import json
import math
import re
from pathlib import Path
from typing import List, Optional, Sequence

import numpy as np
import pandas as pd

RESPDIFF_ROOT = Path(r"D:\PPG2RSP_RRWaveNet_Inspired\261020_RespDiff_v2")
OUTPUT_FOLDER = "PCC_Results"
OUTPUT_NAME = "261107_RespDiff_PCC_PerSubject.xlsx"
DATASETS = ("STEAM2", "BIDMC", "CapnoBase")
# windows of the RespDiff evaluation (261020_RespDiff: FS, RR_WINDOW, RR_HOP)
FS = 30
WINDOW = 60 * FS
HOP = 20 * FS
OWN_SCORES = ("N_Windows", "RR_MAE", "RR_MAE_Official_Code", "Waveform_MAE", "Segment_MAE", "Duty_Cycle_Corr")


def pearson(a: np.ndarray, b: np.ndarray) -> float:
    a, b = np.asarray(a, float), np.asarray(b, float)
    ok = np.isfinite(a) & np.isfinite(b)
    a, b = a[ok], b[ok]
    if a.size < 2 or a.std() <= 1e-12 or b.std() <= 1e-12:
        return math.nan
    return float(np.corrcoef(a, b)[0, 1])


def window_pccs(prediction: np.ndarray, reference: np.ndarray) -> List[dict]:
    """The 60 s windows moved by 20 s of the RespDiff evaluation (score_recording)."""
    n = min(len(prediction), len(reference))
    rows = []
    if n >= WINDOW:
        for index in range((n - WINDOW) // HOP + 1):
            a = HOP * index
            rows.append({"Window": index, "Start_Sec": a / FS, "End_Sec": (a + WINDOW) / FS,
                         "PCC": pearson(prediction[a:a + WINDOW], reference[a:a + WINDOW])})
    return rows


def read_json(path: Path) -> dict:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}


def subject_order(subject: str):
    match = re.match(r"\d+", subject)
    return (int(match.group()) if match else 10 ** 9, subject)


def scan(root: Path, datasets: Sequence[str], samplings: Optional[Sequence[str]]) -> List[dict]:
    """Every stored prediction file: model folder, dataset, subject, sampling, paths."""
    found = []
    for model_dir in sorted(p for p in root.iterdir() if p.is_dir()):
        for dataset in datasets:
            pred_dir = model_dir / dataset / "Predictions"
            if not pred_dir.is_dir():
                continue
            for path in sorted(pred_dir.glob("*.npz")):
                subject, _, sampling = path.stem.rpartition("_")
                if not subject or (samplings and sampling not in samplings):
                    continue
                found.append({"Model": model_dir.name, "Dataset": dataset, "Subject": subject, "Sampling": sampling,
                              "npz": path, "json": model_dir / dataset / "Fold_Results" / f"{subject}_{sampling}.json"})
    return found


def subject_rows(item: dict) -> (dict, List[dict]):
    windows, pred_all, ref_all, recordings = [], [], [], 0
    with np.load(item["npz"], allow_pickle=False) as data:
        names = [str(n) for n in data["recording_names"]] if "recording_names" in data.files else []
        k = 0
        while f"recording{k}_prediction" in data.files:
            prediction = data[f"recording{k}_prediction"].astype(np.float64)
            reference = data[f"recording{k}_reference"].astype(np.float64)
            n = min(len(prediction), len(reference))
            recordings += 1
            name = names[k] if k < len(names) else f"recording{k}"
            windows += [{"Model": item["Model"], "Dataset": item["Dataset"], "Sampling": item["Sampling"],
                         "Subject": item["Subject"], "Recording": name, **row}
                        for row in window_pccs(prediction[:n], reference[:n])]
            pred_all.append(prediction[:n])
            ref_all.append(reference[:n])
            k += 1
    pcc = np.array([w["PCC"] for w in windows], dtype=float)
    pcc = pcc[np.isfinite(pcc)]
    own = read_json(item["json"])
    row = {"Model": item["Model"], "Dataset": item["Dataset"], "Sampling": item["Sampling"], "Subject": item["Subject"],
           "Recordings": recordings, "N_PCC_Windows": int(pcc.size),
           "PCC_Mean": float(pcc.mean()) if pcc.size else math.nan,
           "PCC_SD": float(pcc.std(ddof=1)) if pcc.size > 1 else math.nan,
           "PCC_Median": float(np.median(pcc)) if pcc.size else math.nan,
           "PCC_Whole": pearson(np.concatenate(pred_all), np.concatenate(ref_all)) if pred_all else math.nan,
           **{f"Own_{key}": own.get(key, math.nan) for key in OWN_SCORES}}
    return row, windows


def mean_sd_median(values) -> str:
    v = pd.to_numeric(pd.Series(values), errors="coerce").dropna().to_numpy(float)
    if not v.size:
        return ""
    sd = v.std(ddof=1) if v.size > 1 else math.nan
    return f"{v.mean():.3f} ± {sd:.3f} ({np.median(v):.3f})"


def notes() -> pd.DataFrame:
    return pd.DataFrame([
        ("Source", "the stored RespDiff results only: Predictions\\{subject}_{sampling}.npz (recording{i}_prediction, "
                   "recording{i}_reference) and Fold_Results\\{subject}_{sampling}.json"),
        ("PCC", "Pearson r of RespDiff's prediction (joined, low-pass filtered, as its evaluation) and the reference it "
                "stored (joined 5 s segments, each min-max scaled to [0, 1]); no time shift, no sign change"),
        ("Windows", "60 s moved by 20 s over every recording (the windows of RespDiff's RR / duty evaluation)"),
        ("PCC_Mean / SD / Median", "over all windows of all recordings of the subject"),
        ("PCC_Whole", "Pearson r over all samples of all recordings of the subject"),
        ("Own_*", "the RespDiff run's own scores of the subject (its json): RR_MAE (breaths/min), Waveform_MAE, "
                  "Segment_MAE, Duty_Cycle_Corr, N_Windows"),
        ("Summary", "Mean ± SD (Median) over subjects"),
    ], columns=["Item", "Description"])


def main(argv: Optional[Sequence[str]] = None) -> None:
    parser = argparse.ArgumentParser(description="Per-subject PCC of the stored RespDiff results")
    parser.add_argument("--respdiff-root", type=Path, default=RESPDIFF_ROOT)
    parser.add_argument("--samplings", nargs="*", default=None, help="e.g. ddpm50 ddim6 (default: every one found)")
    parser.add_argument("--datasets", nargs="+", choices=list(DATASETS), default=list(DATASETS))
    parser.add_argument("--output-root", type=Path, default=None,
                        help=f"default: {{respdiff-root}}\\{OUTPUT_FOLDER}")
    args = parser.parse_args(argv)
    root = Path(args.respdiff_root)
    if not root.is_dir():
        raise SystemExit(f"RespDiff results folder not found: {root}")
    items = scan(root, args.datasets, args.samplings)
    if not items:
        raise SystemExit(f"No {{model}}\\{{dataset}}\\Predictions\\*.npz under {root}")
    print(f"RespDiff results: {root} ({len(items)} prediction files)")
    subjects, windows = [], []
    for item in items:
        row, rows = subject_rows(item)
        subjects.append(row)
        windows += rows
    subjects = pd.DataFrame(subjects)
    subjects = subjects.iloc[sorted(range(len(subjects)), key=lambda i: (
        subjects["Model"].iloc[i], subjects["Sampling"].iloc[i], DATASETS.index(subjects["Dataset"].iloc[i]),
        subject_order(subjects["Subject"].iloc[i])))].reset_index(drop=True)
    windows = pd.DataFrame(windows)
    summary = []
    for (model, sampling, dataset), part in subjects.groupby(["Model", "Sampling", "Dataset"], sort=False):
        summary.append({"Model": model, "Sampling": sampling, "Dataset": dataset, "N_Subjects": len(part),
                        "PCC (window mean)": mean_sd_median(part["PCC_Mean"]),
                        "PCC (whole recording)": mean_sd_median(part["PCC_Whole"]),
                        "Own RR_MAE": mean_sd_median(part["Own_RR_MAE"]),
                        "Own Waveform_MAE": mean_sd_median(part["Own_Waveform_MAE"]),
                        "Own Duty_Cycle_Corr": mean_sd_median(part["Own_Duty_Cycle_Corr"])})
    summary = pd.DataFrame(summary)
    out = Path(args.output_root) if args.output_root else root / OUTPUT_FOLDER
    out.mkdir(parents=True, exist_ok=True)
    sheets = {"Summary": summary}
    for dataset in DATASETS:
        part = subjects[subjects["Dataset"] == dataset]
        if not part.empty:
            sheets[dataset] = part.drop(columns="Dataset")
    sheets.update({"Windows": windows, "Notes": notes()})
    path = out / OUTPUT_NAME
    with pd.ExcelWriter(path, engine="openpyxl") as writer:
        for name, frame in sheets.items():
            frame.to_excel(writer, sheet_name=name, index=False)
    for name, frame in sheets.items():
        frame.to_csv(out / f"RespDiff_PCC_{name}.csv", index=False, encoding="utf-8-sig")
    pd.set_option("display.width", 220)
    print("\nMean ± SD (Median) over subjects:")
    print(summary.to_string(index=False))
    print(f"\nSaved: {path}")


if __name__ == "__main__":
    main()
