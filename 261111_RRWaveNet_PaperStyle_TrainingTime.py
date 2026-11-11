# -*- coding: utf-8 -*-
r"""261111_RRWaveNet_PaperStyle_TrainingTime: the training time per LOSO fold and the
parameter count of the paper-style RRWaveNet RR model, read from what its runs stored.
Nothing is trained; nothing of the runs is changed.

Runs read (every folder that exists; --roots adds others)
  D:\PPG2RSP_RRWaveNet_Inspired\261106_DeepCorrencoder_PaperStyle_All\RRWaveNet_PaperStyle_RR
      (261106 one-file run, part B, and 261107_RRWaveNet_PaperStyle_RR_W16_BIDMC_CapnoBase:
       LOSO with an inner subject-level 5-fold CV = 5 models per test subject)
  D:\PPG2RSP_RRWaveNet_Inspired\261106_RRWaveNet_PaperStyle_RR\261106_RRWaveNet_PaperStyle_RR_Results
  D:\PPG2RSP_RRWaveNet_Inspired\261106_RRWaveNet_PaperStyle_RR_Results
  D:\PPG2RSP_RRWaveNet_Inspired\261107_RRWaveNet_PaperStyle_LOSO_W16
      (one model per test subject)
  In each: <Dataset>\W<size>\metrics\RRWaveNet_PaperStyle_Final_PerFold.csv, or the running
  table RRWaveNet_PaperStyle_Running_PerFold.csv of a job that is not finished.

Where the time comes from
  Elapsed_Sec of every row of the PerFold table: one model (test subject x inner fold x
  input case), timed by the run from the start of the model's fold to the end of its test
  prediction (data split, every epoch with its validation, early stopping, reload of the
  best model, test prediction; training is nearly all of it).
  Per LOSO fold (test subject) = the sum over the models of that test subject.
  Trainable_Params: stored by the run for every model.
  A fold kept after a resume keeps the time of its first training.

Outputs (default D:\PPG2RSP_RRWaveNet_Inspired\261111_RRWaveNet_PaperStyle_TrainingTime)
  RRWaveNet_PaperStyle_TrainingTime.xlsx: Summary, LOSO_Folds, Models, Sources, Notes (+ csv)

Run:  python 261111_RRWaveNet_PaperStyle_TrainingTime.py
      --roots <folder> ...   other results folders     --output-root <folder>
"""

import argparse
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

WORKSPACE = Path(r"D:\PPG2RSP_RRWaveNet_Inspired")
NAME = "261111_RRWaveNet_PaperStyle_TrainingTime"
OUTPUT_ROOT = WORKSPACE / NAME
DEFAULT_ROOTS = (
    WORKSPACE / "261106_DeepCorrencoder_PaperStyle_All" / "RRWaveNet_PaperStyle_RR",
    WORKSPACE / "261106_RRWaveNet_PaperStyle_RR" / "261106_RRWaveNet_PaperStyle_RR_Results",
    WORKSPACE / "261106_RRWaveNet_PaperStyle_RR_Results",
    WORKSPACE / "261107_RRWaveNet_PaperStyle_LOSO_W16",
)
DATASET_ORDER = {"STEAM2": 0, "BIDMC": 1, "CapnoBase": 2}


def candidate_roots(extra: Sequence[Path]) -> List[Path]:
    here = Path(__file__).resolve().parent
    found, seen = [], set()
    for root in list(extra) + list(DEFAULT_ROOTS) + [here / "RRWaveNet_PaperStyle_RR",
                                                     here / "261106_RRWaveNet_PaperStyle_RR_Results"]:
        root = Path(root)
        key = str(root).lower()
        if key not in seen and root.is_dir():
            seen.add(key)
            found.append(root)
    return found


def run_label(root: Path) -> str:
    return root.name if root.name != "RRWaveNet_PaperStyle_RR" else f"{root.parent.name}\\{root.name}"


def read_tables(roots: List[Path]) -> Tuple[pd.DataFrame, pd.DataFrame]:
    frames, sources = [], []
    for root in roots:
        for metrics in sorted(root.glob("*/W*/metrics")):
            final = metrics / "RRWaveNet_PaperStyle_Final_PerFold.csv"
            running = metrics / "RRWaveNet_PaperStyle_Running_PerFold.csv"
            path, status = (final, "finished") if final.is_file() else (running, "running (not finished)")
            if not path.is_file():
                continue
            try:
                frame = pd.read_csv(path, encoding="utf-8-sig", dtype={"Test_Subject": str})
            except Exception as exc:  # a table being written at this moment
                sources.append({"Run": run_label(root), "Table": str(path), "Status": f"not read ({exc})"})
                continue
            if frame.empty or "Elapsed_Sec" not in frame.columns:
                continue
            frame.insert(0, "Status", status)
            frame.insert(0, "Run", run_label(root))
            frames.append(frame)
            sources.append({"Run": run_label(root), "Table": str(path), "Status": status, "Rows": len(frame)})
    models = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    return models, pd.DataFrame(sources)


def text(values: np.ndarray, scale: float = 60.0) -> str:
    v = np.asarray(values, float) / scale
    v = v[np.isfinite(v)]
    if not v.size:
        return "-"
    sd = f"{v.std(ddof=1):.1f}" if v.size > 1 else "-"
    return f"{v.mean():.1f} ± {sd} (median {np.median(v):.1f}, {v.min():.1f}-{v.max():.1f})"


def summarize(models: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    keys = ["Run", "Status", "Dataset", "Window_Sec", "Input_Case"]
    folds = (models.groupby(keys + ["Test_Subject"], sort=False)
             .agg(Models=("Elapsed_Sec", "size"), Fold_Time_Sec=("Elapsed_Sec", "sum"),
                  Model_Time_Mean_Sec=("Elapsed_Sec", "mean"), Best_Epoch_Median=("Best_Epoch", "median"))
             .reset_index())
    folds["Fold_Time_Min"] = folds["Fold_Time_Sec"] / 60.0
    rows = []
    for key, part in models.groupby(keys, sort=False):
        fold_part = folds.set_index(keys).loc[key]
        fold_part = fold_part if isinstance(fold_part, pd.DataFrame) else fold_part.to_frame().T
        params = sorted(set(int(p) for p in part["Trainable_Params"].dropna())) if "Trainable_Params" in part else []
        epochs = part["Best_Epoch"].to_numpy(float) if "Best_Epoch" in part else np.array([])
        rows.append({**dict(zip(keys, key)),
                     "Test_Subjects": int(part["Test_Subject"].nunique()),
                     "Models": int(len(part)),
                     "Models_Per_Fold": int(fold_part["Models"].median()),
                     "Total_Hours": float(part["Elapsed_Sec"].sum() / 3600.0),
                     "Fold_Time_Mean_Min": float(fold_part["Fold_Time_Sec"].astype(float).mean() / 60.0),
                     "Fold_Time_SD_Min": float(fold_part["Fold_Time_Sec"].astype(float).std(ddof=1) / 60.0)
                     if len(fold_part) > 1 else np.nan,
                     "Fold_Time_Median_Min": float(fold_part["Fold_Time_Sec"].astype(float).median() / 60.0),
                     "Fold_Time_Text_Min": text(fold_part["Fold_Time_Sec"].to_numpy(float)),
                     "Model_Time_Text_Min": text(part["Elapsed_Sec"].to_numpy(float)),
                     "Best_Epoch_Median": float(np.nanmedian(epochs)) if epochs.size else np.nan,
                     "Best_Epoch_Range": f"{np.nanmin(epochs):.0f}-{np.nanmax(epochs):.0f}" if epochs.size else "-",
                     "Trainable_Params": ", ".join(f"{p:,}" for p in params)})
    summary = pd.DataFrame(rows)
    order = summary["Dataset"].map(DATASET_ORDER).fillna(9)
    summary = summary.assign(_o=order).sort_values(["Run", "_o", "Window_Sec"], kind="mergesort").drop(columns="_o")
    return summary.reset_index(drop=True), folds


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--roots", nargs="+", type=Path, default=[], help="other results folders of the runs")
    parser.add_argument("--output-root", type=Path, default=OUTPUT_ROOT)
    args = parser.parse_args(argv)
    roots = candidate_roots(args.roots)
    if not roots:
        print("No results folder of the paper-style RRWaveNet found; give it with --roots")
        return 1
    models, sources = read_tables(roots)
    if models.empty:
        print("No PerFold table with Elapsed_Sec found in:\n  " + "\n  ".join(str(r) for r in roots))
        return 1
    summary, folds = summarize(models)
    out = Path(args.output_root)
    out.mkdir(parents=True, exist_ok=True)
    notes = pd.DataFrame({"Item": ["Model time", "Fold time", "Parameters", "Resume"], "Description": [
        "Elapsed_Sec of the run's PerFold table: start of the model's fold to the end of its test prediction",
        "sum over the models of a LOSO test subject (5 inner folds in the 261106 run, 1 model in LOSO_W16)",
        "Trainable_Params stored by the run (the same for every window size: global average pooling)",
        "a fold kept after a resume keeps the time of its first training"]})
    sheets = {"Summary": summary, "LOSO_Folds": folds, "Models": models, "Sources": sources, "Notes": notes}
    try:
        with pd.ExcelWriter(out / "RRWaveNet_PaperStyle_TrainingTime.xlsx", engine="openpyxl") as writer:
            for name, frame in sheets.items():
                frame.to_excel(writer, sheet_name=name, index=False)
    except PermissionError:
        print("  RRWaveNet_PaperStyle_TrainingTime.xlsx is open in another program; only the csv files are written")
    for name in ("Summary", "LOSO_Folds"):
        sheets[name].to_csv(out / f"RRWaveNet_PaperStyle_TrainingTime_{name}.csv", index=False, encoding="utf-8-sig")
    pd.set_option("display.width", 250)
    pd.set_option("display.max_columns", 20)
    print(summary[["Run", "Status", "Dataset", "Window_Sec", "Test_Subjects", "Models_Per_Fold", "Total_Hours",
                   "Fold_Time_Text_Min", "Trainable_Params"]].to_string(index=False,
                                                                          float_format=lambda v: f"{v:.2f}"))
    print(f"\nFold time in minutes: mean ± SD (median, min-max). Saved in {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
