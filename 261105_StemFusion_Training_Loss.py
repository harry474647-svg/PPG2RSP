r"""261105_StemFusion_Training_Loss: the final training loss of the models with
and without the 1x1 stem fusion, every dataset, as box plots with the paired
test (nothing is trained or re-run; the values are read from the results of
261028_Stem_ablation_quantative_result).

Values
------
Train_Loss_Eval of Summary\stem_fusion_fold_measures.csv of
261028_Stem_ablation_quantative_result: the training (v16 waveform) loss of
the restored model of a fold (its best validation epoch) on the training
windows of that fold, eval mode, without the jitter of training. One model
per fold (member m0, the same seed for both models):
  CapnoBase   261007 (full vs no_stem_fusion)
  BIDMC       261013 (full model of the weight decay selected in the fold vs
  STEAM2      the model without stem fusion trained by
              261028_Stem_ablation_noise_v1 with that weight decay)
The epochs at which the models were restored (Best_Epoch) are listed beside
the losses.

Statistics
----------
Per dataset, paired by fold: median [IQR], mean +- SD, the number of folds
with a lower loss with stem fusion, median of the relative difference
(with - without) / without, rank-biserial correlation, two-sided Wilcoxon
signed-rank test, Benjamini-Hochberg over the datasets. A significant
difference (FDR < 0.05) is marked on the box plot with a bracket and stars
(* < 0.05, ** < 0.01, *** < 0.001, **** < 1e-4) and the p value; otherwise
"n.s.".

Outputs (default D:\PPG2RSP_RRWaveNet_Inspired\261105_StemFusion_Feature_Interpretation\Training_Loss)
  StemFusion_Training_Loss.png / .pdf   box plots, one panel per dataset
  StemFusion_Training_Loss.xlsx         Summary (statistics), Per_Fold (the
                                        extracted values), Notes
  StemFusion_Training_Loss_per_fold.csv, StemFusion_Training_Loss_summary.csv

Run:  python 261105_StemFusion_Training_Loss.py
      --fold-measures <261028_Stem_ablation_quantative_result\Summary\stem_fusion_fold_measures.csv>
      --output-root <folder>
"""

import argparse
from pathlib import Path
from typing import Optional, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

WORKSPACE = Path(r"D:\PPG2RSP_RRWaveNet_Inspired")
FOLD_MEASURES = WORKSPACE / "261028_Stem_ablation_quantative_result" / "Summary" / "stem_fusion_fold_measures.csv"
OUTPUT_ROOT = WORKSPACE / "261105_StemFusion_Feature_Interpretation" / "Training_Loss"
DATASETS = ("CapnoBase", "BIDMC", "STEAM2")
COLORS = {"without": "#e8968c", "with": "#5b8db8"}
LABELS = {"without": "without\nstem fusion", "with": "with\nstem fusion"}


def stars(p: float) -> str:
    for threshold, mark in ((1e-4, "****"), (1e-3, "***"), (1e-2, "**"), (0.05, "*")):
        if np.isfinite(p) and p < threshold:
            return mark
    return "n.s."


def p_text(p: float) -> str:
    if not np.isfinite(p):
        return "p = n/a"
    if p < 1e-3:
        mantissa, exponent = f"{p:.1e}".split("e")
        return rf"p = {mantissa}$\times$10$^{{{int(exponent)}}}$"
    return f"p = {p:.3f}"


def bh(p: np.ndarray) -> np.ndarray:
    p = np.asarray(p, float)
    out = np.full(len(p), np.nan)
    ok = np.isfinite(p)
    if ok.any():
        q = p[ok]
        order = np.argsort(q)
        ranked = q[order] * len(q) / (np.arange(len(q)) + 1)
        ranked = np.minimum.accumulate(ranked[::-1])[::-1]
        values = np.empty(len(q))
        values[order] = np.clip(ranked, 0, 1)
        out[ok] = values
    return out


def extract(frame: pd.DataFrame) -> pd.DataFrame:
    frame = frame.copy()
    frame["Model_Key"] = np.where(frame["Folder"].astype(str) == "full", "with", "without")
    keep = ["Train_Loss_Eval", "Best_Epoch", "Epochs_Run"]
    wide = frame.pivot_table(index=["Dataset", "Training_Protocol", "Subject"], columns="Model_Key", values=keep)
    wide.columns = [f"{name}_{'With' if key == 'with' else 'Without'}" for name, key in wide.columns]
    wide = wide.reset_index().rename(columns={"Training_Protocol": "Source",
                                              "Train_Loss_Eval_With": "Train_Loss_With",
                                              "Train_Loss_Eval_Without": "Train_Loss_Without"})
    wide = wide.dropna(subset=["Train_Loss_With", "Train_Loss_Without"])
    wide["Difference_With_minus_Without"] = wide.Train_Loss_With - wide.Train_Loss_Without
    wide["Relative_Difference_Pct"] = 100.0 * wide.Difference_With_minus_Without / wide.Train_Loss_Without
    order = {d: i for i, d in enumerate(DATASETS)}
    wide = wide.sort_values(["Dataset", "Subject"], key=lambda s: s.map(order) if s.name == "Dataset" else s)
    cols = ["Dataset", "Source", "Subject", "Train_Loss_Without", "Train_Loss_With", "Difference_With_minus_Without",
            "Relative_Difference_Pct", "Best_Epoch_Without", "Best_Epoch_With", "Epochs_Run_Without", "Epochs_Run_With"]
    return wide[[c for c in cols if c in wide.columns]]


def summarize(per_fold: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for dataset in [d for d in DATASETS if d in set(per_fold.Dataset)]:
        d = per_fold[per_fold.Dataset == dataset]
        o, x = d.Train_Loss_With.to_numpy(), d.Train_Loss_Without.to_numpy()
        diff = o - x
        nz = diff[diff != 0]
        p = float(stats.wilcoxon(o, x).pvalue) if len(nz) else 1.0
        ranks = stats.rankdata(np.abs(nz))
        rb = float((ranks[nz > 0].sum() - ranks[nz < 0].sum()) / ranks.sum()) if len(nz) else 0.0
        rows.append({
            "Dataset": dataset, "Source": "/".join(sorted(set(d.Source.astype(str)))), "N_Folds": len(d),
            "Without_Median": np.median(x), "Without_Q1": np.percentile(x, 25), "Without_Q3": np.percentile(x, 75),
            "With_Median": np.median(o), "With_Q1": np.percentile(o, 25), "With_Q3": np.percentile(o, 75),
            "Without_Mean": x.mean(), "Without_SD": x.std(ddof=1), "With_Mean": o.mean(), "With_SD": o.std(ddof=1),
            "N_Lower_With_Stem_Fusion": int((diff < 0).sum()), "N_Higher_With_Stem_Fusion": int((diff > 0).sum()),
            "Median_Relative_Difference_Pct": float(np.median(d.Relative_Difference_Pct)),
            "Rank_Biserial_With_minus_Without": rb, "Wilcoxon_p": p,
            "Best_Epoch_Without_Median": float(np.median(d.Best_Epoch_Without)),
            "Best_Epoch_With_Median": float(np.median(d.Best_Epoch_With)),
        })
    summary = pd.DataFrame(rows)
    summary["Wilcoxon_FDR_p"] = bh(summary.Wilcoxon_p.to_numpy())
    summary["Significance"] = [stars(q) for q in summary.Wilcoxon_FDR_p]
    return summary


def figure(per_fold: pd.DataFrame, summary: pd.DataFrame, path: Path) -> None:
    datasets = list(summary.Dataset)
    fig, axes = plt.subplots(1, len(datasets), figsize=(4.0 * len(datasets), 4.9), squeeze=False)
    rng = np.random.default_rng(0)
    for ax, dataset in zip(axes[0], datasets):
        d = per_fold[per_fold.Dataset == dataset]
        s = summary[summary.Dataset == dataset].iloc[0]
        values = {"without": d.Train_Loss_Without.to_numpy(), "with": d.Train_Loss_With.to_numpy()}
        for a, b in zip(values["without"], values["with"]):
            ax.plot([1, 2], [a, b], color="#cfcecb", lw=0.6, zorder=1)
        boxes = ax.boxplot([values["without"], values["with"]], widths=0.5, patch_artist=True, showfliers=False,
                           medianprops={"color": "#1f1f1f", "lw": 1.4}, whiskerprops={"color": "#555"},
                           capprops={"color": "#555"})
        for patch, key in zip(boxes["boxes"], ("without", "with")):
            patch.set_facecolor(COLORS[key])
            patch.set_alpha(0.55)
            patch.set_edgecolor("#555")
        for k, key in enumerate(("without", "with"), start=1):
            ax.scatter(k + rng.uniform(-0.06, 0.06, len(values[key])), values[key], s=10, color=COLORS[key],
                       edgecolor="white", linewidth=0.3, zorder=3)
        top = max(values["without"].max(), values["with"].max())
        bottom = min(values["without"].min(), values["with"].min())
        span = top - bottom if top > bottom else abs(top) or 1.0
        y = top + 0.06 * span
        significant = s.Significance != "n.s."
        if significant:
            ax.plot([1, 1, 2, 2], [y, y + 0.03 * span, y + 0.03 * span, y], color="#1f1f1f", lw=1.0)
            label = f"{s.Significance}\n{p_text(s.Wilcoxon_FDR_p)}"
        else:
            label = f"n.s.\n{p_text(s.Wilcoxon_FDR_p)}"
        ax.text(1.5, y + 0.05 * span, label, ha="center", va="bottom", fontsize=9,
                fontweight="bold" if significant else "normal")
        ax.set_ylim(bottom - 0.05 * span, top + 0.32 * span)
        ax.set_xticks([1, 2])
        ax.set_xticklabels([LABELS["without"], LABELS["with"]], fontsize=9)
        ax.set_title(f"{dataset} ({s.Source})\nlower with stem fusion in {s.N_Lower_With_Stem_Fusion}/{s.N_Folds} folds,"
                     f" median {s.Median_Relative_Difference_Pct:+.1f}%", fontsize=9.5)
        ax.grid(axis="y", color="#e4e3df", lw=0.6)
        ax.tick_params(labelsize=8.5)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
    axes[0, 0].set_ylabel("Final training loss\n(restored model, training windows)", fontsize=9.5)
    fig.suptitle("Final training loss with and without the 1x1 stem fusion (single model m0 per fold)", fontsize=11)
    fig.text(0.5, 0.01, "Wilcoxon signed-rank test paired by fold, Benjamini-Hochberg over the datasets; "
             "lines join the two models of a fold", ha="center", fontsize=8, color="#555")
    fig.tight_layout(rect=(0, 0.04, 1, 0.97))
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path.with_suffix(".png"), dpi=300)
    fig.savefig(path.with_suffix(".pdf"))
    plt.close(fig)


def main(argv: Optional[Sequence[str]] = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--fold-measures", type=Path, default=FOLD_MEASURES,
                        help="stem_fusion_fold_measures.csv of 261028_Stem_ablation_quantative_result.")
    parser.add_argument("--output-root", type=Path, default=OUTPUT_ROOT)
    args = parser.parse_args(argv)
    frame = pd.read_csv(args.fold_measures, encoding="utf-8-sig", dtype={"Subject": str})
    per_fold = extract(frame)
    summary = summarize(per_fold)
    out = Path(args.output_root)
    out.mkdir(parents=True, exist_ok=True)
    figure(per_fold, summary, out / "StemFusion_Training_Loss")
    notes = pd.DataFrame({"Item": ["Value", "Models", "Statistics", "Relative difference"], "Meaning": [
        "Train_Loss_Eval of 261028_Stem_ablation_quantative_result: training (v16 waveform) loss of the restored model "
        "(best validation epoch) on the training windows of its fold, eval mode, no jitter",
        "member m0 per fold, same seed; CapnoBase 261007 full vs no_stem_fusion; BIDMC / STEAM2 261013 full (selected "
        "weight decay) vs 261028_Stem_ablation_noise_v1 no_stem_fusion (same weight decay)",
        "Wilcoxon signed-rank test paired by fold, Benjamini-Hochberg over the datasets; stars from the FDR p",
        "(with - without) / without x 100 per fold; negative = lower loss with stem fusion"]})
    with pd.ExcelWriter(out / "StemFusion_Training_Loss.xlsx") as writer:
        rounded = summary.round({c: 6 for c in summary.columns if c not in ("Wilcoxon_p", "Wilcoxon_FDR_p")})
        rounded.to_excel(writer, sheet_name="Summary", index=False)
        per_fold.round(6).to_excel(writer, sheet_name="Per_Fold", index=False)
        notes.to_excel(writer, sheet_name="Notes", index=False)
    per_fold.to_csv(out / "StemFusion_Training_Loss_per_fold.csv", index=False, encoding="utf-8-sig")
    summary.to_csv(out / "StemFusion_Training_Loss_summary.csv", index=False, encoding="utf-8-sig")
    for r in summary.to_dict("records"):
        print(f"{r['Dataset']:<10s} ({r['Source']}): without {r['Without_Median']:.4f} "
              f"[{r['Without_Q1']:.4f}-{r['Without_Q3']:.4f}] vs with {r['With_Median']:.4f} "
              f"[{r['With_Q1']:.4f}-{r['With_Q3']:.4f}]; lower with stem fusion in "
              f"{r['N_Lower_With_Stem_Fusion']}/{r['N_Folds']}, median {r['Median_Relative_Difference_Pct']:+.1f}%, "
              f"p={r['Wilcoxon_p']:.2g}, FDR={r['Wilcoxon_FDR_p']:.2g} {r['Significance']}")
    print(f"saved under {out}")


if __name__ == "__main__":
    main()
