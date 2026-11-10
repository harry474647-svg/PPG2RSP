# -*- coding: utf-8 -*-
r"""261110_StemFusion_TrainLoss_BoxPlot: box plots of the last-epoch training loss with and
without the 1x1 stem fusion (261007 CapnoBase / BIDMC / STEAM2, 261013 BIDMC / STEAM2), in
the layout of the earlier StemFusion_Training_Loss figure (one panel per dataset, without /
with stem fusion, bracket and stars when significant). Nothing is trained or re-run.

Values
  Curve_Final_Train_Loss of Summary\fold_measures.csv of 261106_StemFusion_Generalization_FeatureLevel:
  the training loss of the last epoch of the learning curve (training mode: dropout and jitter),
  single model m0 per fold; O = with stem fusion, X = without (261007: full vs no_stem_fusion;
  261013: full of the selected weight decay vs 261028_Stem_ablation_noise_v1 with that weight decay).
Statistics
  paired by fold: mean +- SD, difference O - X, relative difference (O - X) / X x 100 (mean +- SD
  over folds), folds lower with stem fusion, two-sided Wilcoxon signed-rank test. The FDR p and the
  stars are those of the analysis (Tests: Benjamini-Hochberg over all stages and measures of a run
  and dataset); when its tests are not found, BH over the panels here (written in the Notes).
  Stars: * < 0.05, ** < 0.01, *** < 0.001, **** < 1e-4.

Outputs (default D:\PPG2RSP_RRWaveNet_Inspired\261110_StemFusion_TrainLoss_BoxPlot)
  StemFusion_LastEpoch_TrainLoss_all.png / .pdf      five panels (261007, then 261013)
  StemFusion_LastEpoch_TrainLoss_261007.png / .pdf   three panels
  StemFusion_LastEpoch_TrainLoss_261013.png / .pdf   two panels
  StemFusion_LastEpoch_TrainLoss.xlsx (Summary, Per_Fold, Notes), _summary.csv

Run:  python 261110_StemFusion_TrainLoss_BoxPlot.py
      --summary-dir <261106_StemFusion_Generalization_FeatureLevel\Summary>   (fold_measures.csv and
      tests.csv or the _results.xlsx there), or --fold-measures <csv> --tests <csv or xlsx>
      --output-root <folder>
"""

import argparse
from pathlib import Path
from typing import List, Optional, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

WORKSPACE = Path(r"D:\PPG2RSP_RRWaveNet_Inspired")
SUMMARY_DIR = WORKSPACE / "261106_StemFusion_Generalization_FeatureLevel" / "Summary"
OUTPUT_ROOT = WORKSPACE / "261110_StemFusion_TrainLoss_BoxPlot"
MEASURE = "Curve_Final_Train_Loss"
NAME = "StemFusion_LastEpoch_TrainLoss"
PANELS = (("261007", "CapnoBase"), ("261007", "BIDMC"), ("261007", "STEAM2"), ("261013", "BIDMC"), ("261013", "STEAM2"))
COLORS = {"X": "#ed665a", "O": "#559ed5"}  # without / with stem fusion, as in the earlier figure
LABELS = {"X": "without\nstem fusion", "O": "with\nstem fusion"}
EDGE, WHISKER, GRID = "#3e3e3e", "#3e3e3e", "#e4e3df"


def stars(p: float) -> str:
    for threshold, mark in ((1e-4, "****"), (1e-3, "***"), (1e-2, "**"), (0.05, "*")):
        if np.isfinite(p) and p < threshold:
            return mark
    return "n.s."


def bh(p: Sequence[float]) -> np.ndarray:
    p = np.asarray(p, float)
    out = np.full(len(p), np.nan)
    ok = np.isfinite(p)
    if ok.any():
        q = p[ok]
        order = np.argsort(q)
        ranked = np.minimum.accumulate((q[order] * len(q) / (np.arange(len(q)) + 1))[::-1])[::-1]
        values = np.empty(len(q))
        values[order] = np.clip(ranked, 0, 1)
        out[ok] = values
    return out


def read_tests(path: Optional[Path]) -> pd.DataFrame:
    if path is None or not path.is_file():
        return pd.DataFrame()
    if path.suffix.lower() in (".xlsx", ".xls"):
        return pd.read_excel(path, sheet_name="Tests", dtype={"Run": str})
    return pd.read_csv(path, encoding="utf-8-sig", dtype={"Run": str})


def per_fold_values(frame: pd.DataFrame) -> pd.DataFrame:
    frame = frame.copy()
    frame["Run"] = frame["Run"].astype(str)
    wide = frame.pivot_table(index=["Run", "Dataset", "Subject"], columns="Model", values=MEASURE).reset_index()
    wide = wide.dropna(subset=["O", "X"]).rename(columns={"X": "Without_Stem_Fusion", "O": "With_Stem_Fusion"})
    wide["Difference_With_minus_Without"] = wide["With_Stem_Fusion"] - wide["Without_Stem_Fusion"]
    wide["Relative_Difference_Pct"] = 100.0 * wide["Difference_With_minus_Without"] / wide["Without_Stem_Fusion"]
    order = {panel: i for i, panel in enumerate(PANELS)}
    wide["_order"] = [order.get((r, d), 99) for r, d in zip(wide["Run"], wide["Dataset"])]
    wide = wide.sort_values(["_order", "Subject"], kind="mergesort").drop(columns="_order").reset_index(drop=True)
    wide.columns.name = None
    return wide


def summarize(per_fold: pd.DataFrame, tests: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for run, dataset in PANELS:
        d = per_fold[(per_fold["Run"] == run) & (per_fold["Dataset"] == dataset)]
        if d.empty:
            continue
        o, x = d["With_Stem_Fusion"].to_numpy(float), d["Without_Stem_Fusion"].to_numpy(float)
        diff, rel = o - x, d["Relative_Difference_Pct"].to_numpy(float)
        p = float(stats.wilcoxon(o, x).pvalue) if np.any(diff != 0) else 1.0
        row = {"Run": run, "Dataset": dataset, "N_Folds": len(d),
               "Without_Mean": x.mean(), "Without_SD": x.std(ddof=1), "With_Mean": o.mean(), "With_SD": o.std(ddof=1),
               "Difference_Mean": diff.mean(), "Difference_SD": diff.std(ddof=1),
               "Relative_Difference_Mean_Pct": rel.mean(), "Relative_Difference_SD_Pct": rel.std(ddof=1),
               "Without_Median": np.median(x), "With_Median": np.median(o),
               "N_Lower_With_Stem_Fusion": int((diff < 0).sum()), "N_Higher_With_Stem_Fusion": int((diff > 0).sum()),
               "Wilcoxon_p": p, "FDR_p": np.nan, "FDR_Source": ""}
        if not tests.empty:
            found = tests[(tests["Run"].astype(str) == run) & (tests["Dataset"] == dataset)
                          & (tests["Measure"] == MEASURE)]
            if not found.empty:
                row["FDR_p"] = float(found.iloc[0]["FDR"])
                row["FDR_Source"] = "analysis tests (BH over all stages and measures of the run and dataset)"
                row["Analysis_p"] = float(found.iloc[0]["P"])
        rows.append(row)
    summary = pd.DataFrame(rows)
    missing = summary["FDR_p"].isna()
    if missing.any():
        summary.loc[missing, "FDR_p"] = bh(summary.loc[missing, "Wilcoxon_p"].to_numpy())
        summary.loc[missing, "FDR_Source"] = "BH over the panels of this figure"
    summary["Significance"] = [stars(q) for q in summary["FDR_p"]]
    summary["Without_Text"] = [f"{m:.4f} ± {s:.4f}" for m, s in zip(summary.Without_Mean, summary.Without_SD)]
    summary["With_Text"] = [f"{m:.4f} ± {s:.4f}" for m, s in zip(summary.With_Mean, summary.With_SD)]
    summary["Relative_Text"] = [f"{m:+.1f} ± {s:.1f}%" for m, s in
                                zip(summary.Relative_Difference_Mean_Pct, summary.Relative_Difference_SD_Pct)]
    return summary


def figure(per_fold: pd.DataFrame, summary: pd.DataFrame, panels: List[tuple], path: Path) -> None:
    fig, axes = plt.subplots(1, len(panels), figsize=(4.8 * len(panels), 4.6), squeeze=False)
    for ax, (run, dataset) in zip(axes[0], panels):
        d = per_fold[(per_fold["Run"] == run) & (per_fold["Dataset"] == dataset)]
        s = summary[(summary["Run"] == run) & (summary["Dataset"] == dataset)].iloc[0]
        values = [d["Without_Stem_Fusion"].to_numpy(float), d["With_Stem_Fusion"].to_numpy(float)]
        boxes = ax.boxplot(values, widths=0.5, patch_artist=True, showfliers=False,
                           medianprops={"color": "#111111", "lw": 1.8},
                           whiskerprops={"color": WHISKER, "lw": 1.0}, capprops={"color": WHISKER, "lw": 1.0},
                           boxprops={"edgecolor": EDGE, "lw": 1.0})
        for patch, key in zip(boxes["boxes"], ("X", "O")):
            patch.set_facecolor(COLORS[key])
        ends = np.concatenate([w.get_ydata() for w in boxes["whiskers"]])
        top, bottom = float(np.max(ends)), float(np.min(ends))
        span = top - bottom if top > bottom else abs(top) or 1.0
        if s.Significance != "n.s.":
            y = top + 0.10 * span
            ax.plot([1, 1, 2, 2], [y, y + 0.04 * span, y + 0.04 * span, y], color="#111111", lw=1.2)
            ax.text(1.5, y + 0.13 * span, s.Significance, ha="center", va="bottom", fontsize=11, fontweight="bold")
        ax.set_ylim(bottom - 0.06 * span, top + 0.36 * span)
        ax.set_xticks([1, 2])
        ax.set_xticklabels([LABELS["X"], LABELS["O"]], fontsize=9.5)
        ax.set_title(f"{dataset} ({run})\nlower with stem fusion in {s.N_Lower_With_Stem_Fusion}/{s.N_Folds} folds, "
                     f"mean {s.Relative_Difference_Mean_Pct:+.1f} ± {s.Relative_Difference_SD_Pct:.1f}%", fontsize=10)
        ax.grid(axis="y", color=GRID, lw=0.6)
        ax.set_axisbelow(True)
        ax.tick_params(labelsize=9)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
    axes[0, 0].set_ylabel("Training loss (last epoch)", fontsize=10)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path.with_suffix(".png"), dpi=300)
    fig.savefig(path.with_suffix(".pdf"))
    plt.close(fig)


def main(argv: Optional[Sequence[str]] = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--summary-dir", type=Path, default=SUMMARY_DIR)
    parser.add_argument("--fold-measures", type=Path, default=None, help="fold_measures.csv (default: in --summary-dir)")
    parser.add_argument("--tests", type=Path, default=None,
                        help="tests.csv or the _results.xlsx (Tests sheet) of the analysis (default: in --summary-dir)")
    parser.add_argument("--output-root", type=Path, default=OUTPUT_ROOT)
    args = parser.parse_args(argv)
    fold_path = args.fold_measures or args.summary_dir / "fold_measures.csv"
    tests_path = args.tests
    if tests_path is None:
        for candidate in (args.summary_dir / "tests.csv",
                          args.summary_dir / "261106_StemFusion_Generalization_FeatureLevel_results.xlsx"):
            if candidate.is_file():
                tests_path = candidate
                break
    per_fold = per_fold_values(pd.read_csv(fold_path, encoding="utf-8-sig", dtype={"Subject": str, "Run": str}))
    summary = summarize(per_fold, read_tests(tests_path))
    out = Path(args.output_root)
    out.mkdir(parents=True, exist_ok=True)
    present = [p for p in PANELS if ((summary["Run"] == p[0]) & (summary["Dataset"] == p[1])).any()]
    figure(per_fold, summary, present, out / f"{NAME}_all")
    for run in ("261007", "261013"):
        panels = [p for p in present if p[0] == run]
        if panels:
            figure(per_fold, summary, panels, out / f"{NAME}_{run}")
    notes = pd.DataFrame({"Item": ["Value", "Models", "Statistics", "Inputs"], "Meaning": [
        f"{MEASURE}: training loss of the last epoch of the learning curve (training mode: dropout and jitter)",
        "single model m0 per fold; O = with stem fusion, X = without; 261007 full vs no_stem_fusion; 261013 full "
        "(selected weight decay) vs 261028_Stem_ablation_noise_v1 (same weight decay)",
        "paired by fold; mean +- SD; relative difference (O - X) / X x 100 per fold; two-sided Wilcoxon signed-rank; "
        "FDR as in FDR_Source; stars from the FDR p",
        f"{fold_path}; tests {tests_path}"]})
    with pd.ExcelWriter(out / f"{NAME}.xlsx", engine="openpyxl") as writer:
        summary.to_excel(writer, sheet_name="Summary", index=False)
        per_fold.to_excel(writer, sheet_name="Per_Fold", index=False)
        notes.to_excel(writer, sheet_name="Notes", index=False)
    summary.to_csv(out / f"{NAME}_summary.csv", index=False, encoding="utf-8-sig")
    for r in summary.to_dict("records"):
        print(f"{r['Run']} {r['Dataset']:<10s} n={r['N_Folds']}: without {r['Without_Text']} vs with {r['With_Text']}; "
              f"lower with stem fusion {r['N_Lower_With_Stem_Fusion']}/{r['N_Folds']}, {r['Relative_Text']}, "
              f"p={r['Wilcoxon_p']:.2g}, FDR={r['FDR_p']:.2g} {r['Significance']}")
    print(f"saved under {out}")


if __name__ == "__main__":
    main()
