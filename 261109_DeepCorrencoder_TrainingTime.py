# -*- coding: utf-8 -*-
r"""261109_DeepCorrencoder_TrainingTime: the training time of the deep
corr-encoder in every LOSO fold of 261106_DeepCorrencoder_PaperStyle_All
(STEAM2) and 261106_DeepCorrencoder_PaperStyle_All_v2 (BIDMC, CapnoBase),
read from what those runs stored. Nothing is trained; nothing of the runs is
changed. The paper-style RRWaveNet RR part (RRWaveNet_PaperStyle_RR) is not read.

Where the time comes from (per fold = per LOSO test subject)
  Fold_Time_Sec   Elapsed_Sec_From_Subject_Start of the run's per-subject table
                  (Metrics\LOSO_CorrEncoder_Ablation_PerSubject.csv, else the
                  running table LOSO_CorrEncoder_Ablation_Running.csv, else
                  Combined_Metrics\LOSO_CorrEncoder_Ablation_AllDatasets_PerSubject.csv).
                  The run starts this clock when the fold starts and reads it
                  after the test evaluation of the model: data split, every
                  training epoch with its validation, reload of the best model,
                  test prediction and the 30 s evaluation (training is nearly
                  all of it). The comparison figure drawn after is not inside.
                  With several models in a fold (other ablation feature sets),
                  the clock runs on, so a model's time is the difference to the
                  model before it; the deep corr-encoder of 261106 is 01_low,
                  the only model there.
  Cross-check     file times written by the run:
                  Learning_Curves\<subject>_<model>_loss.png  saved when training ends,
                  Models\<subject>_<model>_best.pth           saved at the last improvement.
                  Curve_Interval_Sec = time between the end of training of this
                  fold and of the fold before (also holds the previous fold's
                  figure); intervals far above the dataset median (> 3 x) are
                  marked as a likely pause or restart of the run.
  Settings        EPOCHS / MIN_EPOCHS / PATIENCE read from the two scripts when
                  they are in the results folder: STEAM2 was trained by
                  261106_DeepCorrencoder_PaperStyle_All.py, BIDMC and CapnoBase
                  by 261106_DeepCorrencoder_PaperStyle_All_v2.py. The number of
                  epochs a fold ran is not stored by the runs.

Results folder: the folder of this file, or else the current folder, when it is
named 261106_DeepCorrencoder_PaperStyle_All; otherwise
D:\PPG2RSP_RRWaveNet_Inspired\261106_DeepCorrencoder_PaperStyle_All
(--output-root sets another one).

Outputs: <results folder>\Deep_Correncoder\Training_Time\
  261109_DeepCorrencoder_TrainingTime.xlsx
    Folds     every fold: dataset, subject, model, Fold_Time_Sec / _Min, source
              table, curve and model file times, Curve_Interval_Sec, note
    Summary   per dataset: folds, total (h), mean +- SD, median, min, max
              (min per fold), the run's EPOCHS / MIN_EPOCHS / PATIENCE
    Notes
  Folds.csv, Summary.csv

Run (in D:\PPG2RSP_RRWaveNet_Inspired\261106_DeepCorrencoder_PaperStyle_All):
  python 261109_DeepCorrencoder_TrainingTime.py
Options: --output-root <folder>, --model 01_low (the corr-encoder model name)
"""

import argparse
import math
import re
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import numpy as np
import pandas as pd

RUN_NAME = "261106_DeepCorrencoder_PaperStyle_All"
DEFAULT_ROOT = Path(r"D:\PPG2RSP_RRWaveNet_Inspired") / RUN_NAME
CE_FOLDER = "Deep_Correncoder"
DATASETS = ("STEAM2", "BIDMC", "CapnoBase")
TRAINED_BY = {"STEAM2": f"{RUN_NAME}.py", "BIDMC": f"{RUN_NAME}_v2.py", "CapnoBase": f"{RUN_NAME}_v2.py"}
MODEL = "01_low"
PAUSE_FACTOR = 3.0
TABLES = ("LOSO_CorrEncoder_Ablation_PerSubject.csv", "LOSO_CorrEncoder_Ablation_Running.csv")
COMBINED = "LOSO_CorrEncoder_Ablation_AllDatasets_PerSubject.csv"
OUT_NAME = "261109_DeepCorrencoder_TrainingTime"


def results_root(argument: Optional[Path]) -> Path:
    if argument:
        return Path(argument)
    for folder in (Path(__file__).resolve().parent, Path.cwd()):
        if folder.name == RUN_NAME:
            return folder
    return DEFAULT_ROOT


def safe_tag(text) -> str:
    """As the runs name their files."""
    return "".join(c if c.isalnum() or c in "-_." else "_" for c in str(text))


def subject_order(subject: str):
    match = re.match(r"\d+", str(subject))
    return (int(match.group()) if match else 10 ** 9, str(subject))


def read_table(path: Path) -> pd.DataFrame:
    try:
        return pd.read_csv(path, dtype={"Subject": str})
    except PermissionError:
        raise SystemExit(f"Cannot read {path}: it is held by another program (e.g. open in Excel). Close it.")


def fold_times(dataset: str, ce_root: Path) -> pd.DataFrame:
    """Every fold and model of a dataset with the time of that model (from the run's clock)."""
    metrics = ce_root / dataset / "Metrics"
    for name in TABLES:
        if (metrics / name).exists():
            table, source = read_table(metrics / name), f"{dataset}\\Metrics\\{name}"
            break
    else:
        combined = ce_root / "Combined_Metrics" / COMBINED
        if not combined.exists():
            return pd.DataFrame()
        table = read_table(combined)
        table = table[table["Dataset"].astype(str) == dataset]
        source = f"Combined_Metrics\\{COMBINED}"
    if table.empty or "Elapsed_Sec_From_Subject_Start" not in table.columns:
        return pd.DataFrame()
    rows = []
    for subject, part in table.groupby("Subject", sort=False):
        previous = 0.0
        for _, r in part.iterrows():  # the run's order: models one after the other in a fold
            elapsed = float(r["Elapsed_Sec_From_Subject_Start"])
            rows.append({"Dataset": dataset, "Subject": str(subject), "Model": str(r.get("Model", "")),
                         "Fold_Time_Sec": elapsed - previous, "Clock_At_End_Sec": elapsed, "Source": source,
                         "Best_Val_Loss": r.get("Best_Val_Loss", math.nan),
                         "Trainable_Params": r.get("Trainable_Params", math.nan)})
            previous = elapsed
    return pd.DataFrame(rows)


def file_time(path: Path) -> Optional[datetime]:
    return datetime.fromtimestamp(path.stat().st_mtime) if path.exists() else None


def add_file_times(frame: pd.DataFrame, ce_root: Path) -> pd.DataFrame:
    """Training-end (learning curve) and last-improvement (model) file times, and the interval between folds."""
    out = []
    for dataset, part in frame.groupby("Dataset", sort=False):
        part = part.copy()
        base = ce_root / dataset
        part["Curve_Saved"] = [file_time(base / "Learning_Curves" / f"{safe_tag(s)}_{m}_loss.png")
                               for s, m in zip(part["Subject"], part["Model"])]
        part["Model_Saved"] = [file_time(base / "Models" / f"{safe_tag(s)}_{m}_best.pth")
                               for s, m in zip(part["Subject"], part["Model"])]
        ordered = part.dropna(subset=["Curve_Saved"]).sort_values("Curve_Saved")
        intervals = ordered["Curve_Saved"].diff().dt.total_seconds()
        part["Curve_Interval_Sec"] = intervals.reindex(part.index)
        median = float(np.nanmedian(part["Curve_Interval_Sec"])) if part["Curve_Interval_Sec"].notna().any() else math.nan
        notes = []
        for _, r in part.iterrows():
            note = []
            if r["Curve_Saved"] is None or pd.isna(r["Curve_Saved"]):
                note.append("no learning-curve file")
            if np.isfinite(median) and pd.notna(r["Curve_Interval_Sec"]) and r["Curve_Interval_Sec"] > PAUSE_FACTOR * median:
                note.append(f"interval > {PAUSE_FACTOR:g} x median: pause or restart before this fold")
            notes.append("; ".join(note))
        part["Note"] = notes
        out.append(part)
    return pd.concat(out, ignore_index=True) if out else frame


def run_settings(root: Path) -> Dict[str, Dict[str, object]]:
    """EPOCHS / MIN_EPOCHS / PATIENCE of the scripts found in the results folder."""
    settings = {}
    for dataset, script in TRAINED_BY.items():
        path = root / script
        values: Dict[str, object] = {"Trained_By": script, "Script_Found": path.exists()}
        if path.exists():
            text = path.read_text(encoding="utf-8", errors="replace")
            for key in ("EPOCHS", "MIN_EPOCHS", "PATIENCE"):
                match = re.search(rf"^{key}\s*=\s*([0-9.]+)", text, re.MULTILINE)
                values[key] = float(match.group(1)) if match else math.nan
        settings[dataset] = values
    return settings


def summary_table(folds: pd.DataFrame, settings: Dict[str, Dict[str, object]]) -> pd.DataFrame:
    rows = []
    for dataset in [d for d in DATASETS if d in set(folds["Dataset"])]:
        minutes = folds.loc[folds["Dataset"] == dataset, "Fold_Time_Min"].to_numpy(float)
        minutes = minutes[np.isfinite(minutes)]
        rows.append({"Dataset": dataset, "Folds": int(minutes.size),
                     "Total_Hours": float(minutes.sum() / 60.0),
                     "Mean_Min_Per_Fold": float(minutes.mean()) if minutes.size else math.nan,
                     "SD_Min": float(minutes.std(ddof=1)) if minutes.size > 1 else math.nan,
                     "Median_Min": float(np.median(minutes)) if minutes.size else math.nan,
                     "Min_Min": float(minutes.min()) if minutes.size else math.nan,
                     "Max_Min": float(minutes.max()) if minutes.size else math.nan,
                     "Mean_SD_Text": f"{minutes.mean():.1f} ± {minutes.std(ddof=1):.1f} min" if minutes.size > 1 else "",
                     **{k: v for k, v in settings.get(dataset, {}).items()}})
    if rows:
        allm = folds["Fold_Time_Min"].to_numpy(float)
        allm = allm[np.isfinite(allm)]
        rows.append({"Dataset": "All", "Folds": int(allm.size), "Total_Hours": float(allm.sum() / 60.0),
                     "Mean_Min_Per_Fold": float(allm.mean()), "SD_Min": float(allm.std(ddof=1)) if allm.size > 1 else math.nan,
                     "Median_Min": float(np.median(allm)), "Min_Min": float(allm.min()), "Max_Min": float(allm.max()),
                     "Mean_SD_Text": f"{allm.mean():.1f} ± {allm.std(ddof=1):.1f} min" if allm.size > 1 else ""})
    return pd.DataFrame(rows)


def notes(root: Path, model: str) -> pd.DataFrame:
    return pd.DataFrame([
        ("Results folder", str(root)),
        ("Model", f"{model} (CorrEncoderNet, 0.1-0.5 Hz PPG input): the deep corr-encoder of 261106"),
        ("Fold_Time_Sec", "Elapsed_Sec_From_Subject_Start of the run (for a later model of the same fold: its difference "
                          "to the model before): from the start of the fold to after the test evaluation; data split, "
                          "training epochs with validation, best-model reload, test prediction, 30 s evaluation"),
        ("Not inside", "the comparison figure after the fold, reading the data and caches before the first fold"),
        ("Curve_Saved / Model_Saved", "file times of Learning_Curves\\<subject>_<model>_loss.png (end of training) and "
                                      "Models\\<subject>_<model>_best.pth (last improvement)"),
        ("Curve_Interval_Sec", "end of training of this fold minus that of the fold before (in time order); also holds the "
                               f"previous fold's figure; > {PAUSE_FACTOR:g} x the dataset median: pause or restart"),
        ("Settings", "EPOCHS / MIN_EPOCHS / PATIENCE of the script that trained the dataset, read from the script in the "
                     "results folder (STEAM2: 261106_DeepCorrencoder_PaperStyle_All.py; BIDMC, CapnoBase: _v2.py); "
                     "the epochs a fold ran are not stored by the runs"),
        ("Hardware", "not stored by the runs: times are those of the computer that ran them"),
    ], columns=["Item", "Description"])


def write_outputs(out_dir: Path, sheets: Dict[str, pd.DataFrame]) -> Path:
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")

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
                frame.to_excel(w, sheet_name=name, index=False)

    for name in ("Folds", "Summary"):
        write(out_dir / f"{name}.csv", lambda p, f=sheets[name]: f.to_csv(p, index=False, encoding="utf-8-sig"))
    return write(out_dir / f"{OUT_NAME}.xlsx", xlsx)


def main(argv: Optional[Sequence[str]] = None) -> None:
    parser = argparse.ArgumentParser(description="Training time of the deep corr-encoder per LOSO fold (261106 runs)")
    parser.add_argument("--output-root", type=Path, default=None, help=f"results folder ({RUN_NAME})")
    parser.add_argument("--model", default=MODEL, help="model name of the corr-encoder in the run's tables")
    args = parser.parse_args(argv)
    root = results_root(args.output_root)
    ce_root = root / CE_FOLDER
    if not ce_root.is_dir():
        raise SystemExit(f"{ce_root} not found: run this in {RUN_NAME} or give --output-root")
    print(f"Results folder: {root}")
    frames: List[pd.DataFrame] = []
    for dataset in DATASETS:
        frame = fold_times(dataset, ce_root)
        if frame.empty:
            print(f"  {dataset}: no per-subject table with Elapsed_Sec_From_Subject_Start")
            continue
        frame = frame[frame["Model"] == args.model]
        print(f"  {dataset}: {len(frame)} folds of {args.model} ({frame['Source'].iloc[0] if len(frame) else '-'})")
        frames.append(frame)
    if not frames or all(f.empty for f in frames):
        raise SystemExit(f"No fold of model {args.model} found under {ce_root}")
    folds = add_file_times(pd.concat(frames, ignore_index=True), ce_root)
    folds["Fold_Time_Min"] = folds["Fold_Time_Sec"] / 60.0
    folds = folds.iloc[sorted(range(len(folds)), key=lambda i: (DATASETS.index(folds["Dataset"].iloc[i]),
                                                                subject_order(folds["Subject"].iloc[i])))]
    first = ["Dataset", "Subject", "Model", "Fold_Time_Sec", "Fold_Time_Min", "Curve_Saved", "Model_Saved",
             "Curve_Interval_Sec", "Note", "Source"]
    folds = folds[[c for c in first if c in folds] + [c for c in folds if c not in first]].reset_index(drop=True)
    settings = run_settings(root)
    summary = summary_table(folds, settings)
    out_dir = ce_root / "Training_Time"
    out_dir.mkdir(parents=True, exist_ok=True)
    path = write_outputs(out_dir, {"Folds": folds, "Summary": summary, "Notes": notes(root, args.model)})
    pd.set_option("display.width", 200)
    show = ["Dataset", "Folds", "Total_Hours", "Mean_SD_Text", "Median_Min", "Min_Min", "Max_Min", "PATIENCE"]
    print("\n" + summary[[c for c in show if c in summary]].round(2).to_string(index=False))
    flagged = folds[folds["Note"].astype(str) != ""]
    if not flagged.empty:
        print("\nNotes:\n" + flagged[["Dataset", "Subject", "Note"]].to_string(index=False))
    print(f"\nSaved: {path}")


if __name__ == "__main__":
    main()
