r"""261105_StemFusion_Gradient_Figure: how the 1x1 stem fusion changes the
learning signal (gradient) of the three stem branches, as box plots with the
paired tests, from the A_Gradients results of
261105_StemFusion_Feature_Interpretation (nothing is trained or re-run).

What is drawn (per dataset, Stem Fusion O = with, X = without; member m0 of
the fold, the same seed; paired by held-out subject)
--------------------------------------------------------------------------
(a) Branch gradient balance at the initial weights and after training:
    entropy of the gradient shares of the k32 / k64 / k128 branches / log 3
    (1 = every branch gets the same learning signal).
(b) Change of the balance during training (after - before): negative = the
    learning signal concentrates on fewer branches.
(c) Gradient share of the long-kernel branch (k128, 2 s) after training;
    dashed line 0.5 = the branch gets half of the stem's learning signal.
(d) Stem / decoder+head gradient norm at the initial weights: how much
    learning signal reaches the stem when training starts.
Gradients: training windows of the fold, the training loss, eval mode, no
jitter (see 261105_StemFusion_Feature_Interpretation).

Figures
-------
  main:          CapnoBase (261007), BIDMC (261013), STEAM2 (261013) - the
                 models of the paper
  supplementary: BIDMC (261007), STEAM2 (261007) - the same comparison with
                 the 261007 training (replication)
Statistics: two-sided Wilcoxon signed-rank test paired by subject,
Benjamini-Hochberg over the tests of a figure; brackets with stars from the
FDR p (* < 0.05, ** < 0.01, *** < 0.001, **** < 1e-4) or "n.s.". The
workbook also lists the change from before to after training within each
model (X and O separately).

Inputs (default, written by 261105_StemFusion_Feature_Interpretation)
  D:\PPG2RSP_RRWaveNet_Inspired\261105_StemFusion_Feature_Interpretation\261007\A_Gradients\A_Gradients.csv
  D:\PPG2RSP_RRWaveNet_Inspired\261105_StemFusion_Feature_Interpretation\261013\A_Gradients\A_Gradients.csv
Outputs (default ...\261105_StemFusion_Feature_Interpretation\Gradient_Figure)
  StemFusion_Gradients_main.png / .pdf
  StemFusion_Gradients_supplementary_261007.png / .pdf
  StemFusion_Gradients.xlsx   Tests (O vs X), Within_Model (before vs after),
                              Notes

Run:  python 261105_StemFusion_Gradient_Figure.py
      [--a-261007 <A_Gradients.csv>] [--a-261013 <A_Gradients.csv>] [--output-root <folder>]
"""

import argparse
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

ROOT = Path(r"D:\PPG2RSP_RRWaveNet_Inspired\261105_StemFusion_Feature_Interpretation")
A_261007 = ROOT / "261007" / "A_Gradients" / "A_Gradients.csv"
A_261013 = ROOT / "261013" / "A_Gradients" / "A_Gradients.csv"
OUTPUT_ROOT = ROOT / "Gradient_Figure"
MAIN = (("261007", "CapnoBase"), ("261013", "BIDMC"), ("261013", "STEAM2"))
SUPPLEMENTARY = (("261007", "BIDMC"), ("261007", "STEAM2"))
COLORS = {"X": "#e8968c", "O": "#5b8db8"}
NAMES = {"X": "without", "O": "with"}


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


def wilcoxon(a: np.ndarray, b: np.ndarray) -> Tuple[float, float]:
    """Two-sided Wilcoxon p and the rank-biserial correlation (positive = a higher)."""
    diff = np.asarray(a, float) - np.asarray(b, float)
    nz = diff[diff != 0]
    if not len(nz):
        return 1.0, 0.0
    ranks = stats.rankdata(np.abs(nz))
    rb = float((ranks[nz > 0].sum() - ranks[nz < 0].sum()) / ranks.sum())
    try:
        return float(stats.wilcoxon(a, b).pvalue), rb
    except ValueError:
        return np.nan, rb


def bh(p: Sequence[float]) -> np.ndarray:
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


def paired(frame: pd.DataFrame, source: str, dataset: str) -> pd.DataFrame:
    """One row per subject with the O and X columns side by side."""
    d = frame[(frame.Source.astype(str) == source) & (frame.Dataset == dataset)]
    o = d[d.Model == "O"].set_index("Subject")
    x = d[d.Model == "X"].set_index("Subject")
    common = o.index.intersection(x.index)
    o, x = o.loc[common], x.loc[common]
    out = pd.DataFrame(index=common)
    for col in [c for c in d.columns if c.startswith("Grad_")]:
        out[f"O_{col}"] = o[col].astype(float)
        out[f"X_{col}"] = x[col].astype(float)
    for m in ("O", "X"):
        out[f"{m}_Balance_Change"] = out[f"{m}_Grad_Branch_Balance_Final"] - out[f"{m}_Grad_Branch_Balance_Init"]
    return out


# (panel, label, O column, X column)
TESTS = (
    ("a", "Balance, initial weights", "O_Grad_Branch_Balance_Init", "X_Grad_Branch_Balance_Init"),
    ("a", "Balance, after training", "O_Grad_Branch_Balance_Final", "X_Grad_Branch_Balance_Final"),
    ("b", "Balance change during training", "O_Balance_Change", "X_Balance_Change"),
    ("c", "k128 gradient share after training", "O_Grad_Share_k128_Final", "X_Grad_Share_k128_Final"),
    ("d", "Stem / head gradient, initial weights", "O_Grad_Stem_to_Head_Init", "X_Grad_Stem_to_Head_Init"),
)


def tests_for(groups: Sequence[Tuple[str, str]], data: dict, figure: str) -> pd.DataFrame:
    rows = []
    for source, dataset in groups:
        t = data[(source, dataset)]
        for panel, label, o_col, x_col in TESTS:
            o, x = t[o_col].to_numpy(), t[x_col].to_numpy()
            p, rb = wilcoxon(o, x)
            rows.append({"Figure": figure, "Source": source, "Dataset": dataset, "Panel": panel, "Measure": label,
                         "N": len(o), "With_Median": np.median(o), "With_Q1": np.percentile(o, 25),
                         "With_Q3": np.percentile(o, 75), "Without_Median": np.median(x),
                         "Without_Q1": np.percentile(x, 25), "Without_Q3": np.percentile(x, 75),
                         "N_With_Higher": int((o > x).sum()), "N_Without_Higher": int((o < x).sum()),
                         "Rank_Biserial_With_minus_Without": rb, "Wilcoxon_p": p})
    frame = pd.DataFrame(rows)
    frame["Wilcoxon_FDR_p"] = bh(frame.Wilcoxon_p)
    frame["Significance"] = [stars(q) for q in frame.Wilcoxon_FDR_p]
    return frame


def within_model(groups: Sequence[Tuple[str, str]], data: dict) -> pd.DataFrame:
    rows = []
    for source, dataset in groups:
        t = data[(source, dataset)]
        for model in ("X", "O"):
            for name in ("Grad_Branch_Balance", "Grad_Share_k128", "Grad_Share_k32", "Grad_Stem_to_Head"):
                before, after = t[f"{model}_{name}_Init"].to_numpy(), t[f"{model}_{name}_Final"].to_numpy()
                p, rb = wilcoxon(after, before)
                rows.append({"Source": source, "Dataset": dataset, "Model": NAMES[model] + " stem fusion",
                             "Measure": name, "N": len(before), "Before_Median": np.median(before),
                             "After_Median": np.median(after), "Change_Median": np.median(after - before),
                             "N_Lower_After": int((after < before).sum()), "Rank_Biserial_After_minus_Before": rb,
                             "Wilcoxon_p": p})
    return pd.DataFrame(rows)


def draw_pair(ax, positions, values, test: pd.Series, rng, show_lines: bool = True) -> Tuple[float, float]:
    """Box plots of X and O at two positions with the fold-paired lines; returns the data range."""
    x, o = values
    if show_lines:
        for a, b in zip(x, o):
            ax.plot(positions, [a, b], color="#d3d2cf", lw=0.5, zorder=1)
    boxes = ax.boxplot([x, o], positions=positions, widths=0.55, patch_artist=True, showfliers=False,
                       medianprops={"color": "#1f1f1f", "lw": 1.3}, whiskerprops={"color": "#555"},
                       capprops={"color": "#555"})
    for patch, key in zip(boxes["boxes"], ("X", "O")):
        patch.set_facecolor(COLORS[key])
        patch.set_alpha(0.55)
        patch.set_edgecolor("#555")
    for pos, key, v in zip(positions, ("X", "O"), (x, o)):
        ax.scatter(pos + rng.uniform(-0.07, 0.07, len(v)), v, s=7, color=COLORS[key], edgecolor="white",
                   linewidth=0.25, zorder=3)
    return float(min(x.min(), o.min())), float(max(x.max(), o.max()))


def bracket(ax, positions, y, height, test: pd.Series) -> None:
    significant = test.Significance != "n.s."
    if significant:
        ax.plot([positions[0], positions[0], positions[1], positions[1]], [y, y + height, y + height, y],
                color="#1f1f1f", lw=0.9)
    ax.text(np.mean(positions), y + 1.3 * height, f"{test.Significance}\n{p_text(test.Wilcoxon_FDR_p)}",
            ha="center", va="bottom", fontsize=7.2, fontweight="bold" if significant else "normal")


def figure(groups: Sequence[Tuple[str, str]], data: dict, tests: pd.DataFrame, title: str, path: Path) -> None:
    rng = np.random.default_rng(0)
    fig, axes = plt.subplots(len(groups), 4, figsize=(15.5, 3.6 * len(groups) + 0.9), squeeze=False,
                             gridspec_kw={"width_ratios": [1.7, 1, 1, 1]})
    headers = ("(a) Branch gradient balance\n(1 = equal learning signal to k32 / k64 / k128)",
               "(b) Change of the balance\nduring training (after - before)",
               "(c) Gradient share of the k128 branch\n(2 s kernel) after training",
               "(d) Stem / head gradient norm\nat the initial weights")
    for i, (source, dataset) in enumerate(groups):
        t = data[(source, dataset)]
        tt = tests[(tests.Source == source) & (tests.Dataset == dataset)].set_index("Measure")
        # (a) before / after
        ax = axes[i, 0]
        spans = []
        for positions, (label, o_col, x_col) in zip(((1, 2), (3.4, 4.4)), (
                ("Balance, initial weights", "O_Grad_Branch_Balance_Init", "X_Grad_Branch_Balance_Init"),
                ("Balance, after training", "O_Grad_Branch_Balance_Final", "X_Grad_Branch_Balance_Final"))):
            spans.append(draw_pair(ax, positions, (t[x_col].to_numpy(), t[o_col].to_numpy()), tt.loc[label], rng))
        low, high = min(s[0] for s in spans), max(s[1] for s in spans)
        span = high - low
        for positions, label in (((1, 2), "Balance, initial weights"), ((3.4, 4.4), "Balance, after training")):
            bracket(ax, positions, high + 0.05 * span, 0.025 * span, tt.loc[label])
        ax.set_ylim(low - 0.05 * span, high + 0.36 * span)
        ax.set_yticks([v for v in ax.get_yticks() if low - 0.05 * span <= v <= 1.0 + 1e-9])  # balance <= 1
        ax.set_xticks([1, 2, 3.4, 4.4])
        ax.set_xticklabels(["without", "with", "without", "with"], fontsize=8)
        ax.text(1.5, -0.2, "initial weights", transform=ax.get_xaxis_transform(), ha="center", fontsize=8.5,
                color="#333")
        ax.text(3.9, -0.2, "after training", transform=ax.get_xaxis_transform(), ha="center", fontsize=8.5,
                color="#333")
        # (b)-(d)
        for j, (label, o_col, x_col) in enumerate((
                ("Balance change during training", "O_Balance_Change", "X_Balance_Change"),
                ("k128 gradient share after training", "O_Grad_Share_k128_Final", "X_Grad_Share_k128_Final"),
                ("Stem / head gradient, initial weights", "O_Grad_Stem_to_Head_Init", "X_Grad_Stem_to_Head_Init")),
                start=1):
            ax = axes[i, j]
            x, o = t[x_col].to_numpy(), t[o_col].to_numpy()
            low, high = draw_pair(ax, (1, 2), (x, o), tt.loc[label], rng)
            span = high - low if high > low else 1.0
            if j == 1:
                ax.axhline(0.0, color="#888", lw=0.8, ls="--", zorder=0)
            if j == 2:
                ax.axhline(0.5, color="#888", lw=0.8, ls="--", zorder=0)
                ax.set_xlabel(f"share > 0.5: without {int((x > 0.5).sum())}/{len(x)}, "
                              f"with {int((o > 0.5).sum())}/{len(o)} subjects", fontsize=7.5, color="#333")
            bracket(ax, (1, 2), high + 0.05 * span, 0.03 * span, tt.loc[label])
            ax.set_ylim(low - 0.08 * span, high + 0.4 * span)
            ax.set_xticks([1, 2])
            ax.set_xticklabels(["without\nstem fusion", "with\nstem fusion"], fontsize=8)
        for j in range(4):
            ax = axes[i, j]
            ax.grid(axis="y", color="#e4e3df", lw=0.6)
            ax.tick_params(axis="y", labelsize=7.5)
            for side in ("top", "right"):
                ax.spines[side].set_visible(False)
            if i == 0:
                ax.set_title(headers[j], fontsize=9)
        axes[i, 0].set_ylabel(f"{dataset} ({source})\n(n = {len(t)})", fontsize=9.5)
    fig.suptitle(title, fontsize=11)
    fig.text(0.5, 0.005, "Wilcoxon signed-rank test paired by held-out subject (member m0, same seed), "
             "Benjamini-Hochberg over the tests of the figure; lines join the two models of a subject",
             ha="center", fontsize=7.8, color="#555")
    fig.tight_layout(rect=(0, 0.02, 1, 0.97))
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path.with_suffix(".png"), dpi=300)
    fig.savefig(path.with_suffix(".pdf"))
    plt.close(fig)


def main(argv: Optional[Sequence[str]] = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--a-261007", type=Path, default=A_261007, help="A_Gradients.csv of 261007.")
    parser.add_argument("--a-261013", type=Path, default=A_261013, help="A_Gradients.csv of 261013.")
    parser.add_argument("--output-root", type=Path, default=OUTPUT_ROOT)
    args = parser.parse_args(argv)
    frames = [pd.read_csv(p, encoding="utf-8-sig", dtype={"Subject": str, "Source": str})
              for p in (args.a_261007, args.a_261013)]
    frame = pd.concat(frames, ignore_index=True)
    data = {(s, d): paired(frame, s, d) for s, d in MAIN + SUPPLEMENTARY}
    out = Path(args.output_root)
    out.mkdir(parents=True, exist_ok=True)
    sheets: List[pd.DataFrame] = []
    for figure_name, groups, title, filename in (
            ("main", MAIN, "Learning signal of the three stem branches with and without the 1x1 stem fusion",
             "StemFusion_Gradients_main"),
            ("supplementary", SUPPLEMENTARY,
             "Replication with the 261007 training: BIDMC and STEAM2", "StemFusion_Gradients_supplementary_261007")):
        groups = [g for g in groups if len(data[g])]
        if not groups:
            continue
        tests = tests_for(groups, data, figure_name)
        figure(groups, data, tests, title, out / filename)
        sheets.append(tests)
    tests = pd.concat(sheets, ignore_index=True)
    within = within_model(MAIN + SUPPLEMENTARY, data)
    notes = pd.DataFrame({"Item": ["Balance", "Balance change", "k128 share", "Stem / head", "Pairing", "Statistics"],
                          "Meaning": [
                              "entropy of the gradient shares of k32 / k64 / k128 / log 3 (1 = equal)",
                              "balance after training - balance at the initial weights (negative = concentrated)",
                              "gradient norm of the k128 branch / sum over the three branches, after training",
                              "stem gradient norm / decoder + head gradient norm at the initial weights",
                              "held-out subject (fold), member m0 of both models (same seed); main: CapnoBase 261007, "
                              "BIDMC / STEAM2 261013; supplementary: BIDMC / STEAM2 261007",
                              "Wilcoxon signed-rank, BH-FDR over the tests of a figure; Within_Model: after vs before "
                              "training per model, not corrected"]})
    with pd.ExcelWriter(out / "StemFusion_Gradients.xlsx") as writer:
        keep_p = ("Wilcoxon_p", "Wilcoxon_FDR_p")
        tests.round({c: 6 for c in tests.columns if c not in keep_p}).to_excel(writer, sheet_name="Tests", index=False)
        within.round({c: 6 for c in within.columns if c not in keep_p}).to_excel(writer, sheet_name="Within_Model",
                                                                                index=False)
        notes.to_excel(writer, sheet_name="Notes", index=False)
    for r in tests.to_dict("records"):
        print(f"{r['Figure']:<13s} {r['Dataset']:<9s} ({r['Source']}) {r['Measure']:<38s} with {r['With_Median']:.3f} "
              f"vs without {r['Without_Median']:.3f}  FDR={r['Wilcoxon_FDR_p']:.2g} {r['Significance']}")
    print(f"saved under {out}")


if __name__ == "__main__":
    main()
