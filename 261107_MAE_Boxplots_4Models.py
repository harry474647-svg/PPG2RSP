# -*- coding: utf-8 -*-
r"""261107_MAE_Boxplots_4Models: respiratory-rate MAE of four models per
dataset, the same subjects, with the subjects that had an MAE issue left out:
    RRWaveNet           waveform model, CapnoBase 261007, BIDMC / STEAM2 261013
    RRWaveNet-RR (W16)  the paper-style RRWaveNet RR regression (261106 /
                        261107_RRWaveNet_PaperStyle_RR_W16_BIDMC_CapnoBase),
                        0.1-0.5 Hz input, 16 s windows
    Deep corr-encoder   261107_DeepCorrencoder_PaperVersion_v1 (stored results)
    RespDiff            261020_RespDiff_v2 (ddpm50)
Nothing is trained or measured again: every value is read from a result file.

Values (subject MAE, breaths/min)
---------------------------------
RRWaveNet, RespDiff   Waveform_Summary_all_runs.xlsx written by
    261107_Boxplots_DeepCorrencoderPaper_RespDiff_RRWaveNet (sheet
    Subject_Summary, Size_Sec 60; Run 261007 CapnoBase, 261013 BIDMC / STEAM2;
    RRWaveNet_MAE, RespDiff_MAE): mean over the subject's non-overlapping 60 s
    windows of |FFT rate of the waveform - GT rate from the breaths|.
RRWaveNet-RR (W16)    {paperstyle-root}\{Dataset}\W16\metrics\
    RRWaveNet_PaperStyle_Final_PerSubject.csv, Fold_MAE_Mean (input case
    low_only): mean over the subject's 16 s windows (2 s stride) and the 5
    inner-fold models of |predicted RR - reference RR (Welch peak)|. When a
    dataset has no final table yet, its running table is used and the figure
    says so (not finished).
Deep corr-encoder     {correncoder-root}\Summary\261107_DeepCorrencoder_
    PaperVersion_v1_results.xlsx, sheet Per_subject, Evaluation_Window_Sec
    60.6, MAE (as stored: FFT rate of the fused output vs the reference rate).
Subjects left out of every model (MAE issue; subject set 04 of the RRWaveNet
runs): STEAM2 22_LMS, 26_CKK, 12_YJJ; BIDMC 27; CapnoBase 0115 (--exclude-*);
not written on the figures, listed in the workbook.

Statistics (per dataset)
------------------------
Friedman test of the four models (subjects with all four), Kendall's W =
chi2 / (n (k - 1)); paired Wilcoxon signed-rank tests of the 6 pairs,
Benjamini-Hochberg over the 6 pairs of the dataset; only significant pairs
(FDR < 0.05) are marked (* < 0.05, ** < 0.01, *** < 0.001, **** < 1e-4).
A box holds every subject the model has. No reference lines.

Outputs (default D:\PPG2RSP_RRWaveNet_Inspired\261107_MAE_Boxplots_4Models)
  Figures\MAE_4models_per_dataset.png             (a) STEAM2 (b) BIDMC (c) CapnoBase, every subject a point
  Figures\MAE_4models_per_dataset_no_points.png   the same without the points
  Figures\MAE_4models_per_dataset[_no_points]_{STEAM2, BIDMC, CapnoBase}.png
  261107_MAE_4models_Tests.xlsx   Friedman, Pairwise, Values, Sources, Excluded (and CSV files)

Run:
  python 261107_MAE_Boxplots_4Models.py
Options: --waveform-summary, --correncoder-root, --correncoder-results,
  --paperstyle-root, --paperstyle-window, --output-root, --exclude-steam2,
  --exclude-bidmc, --exclude-capnobase
"""

import argparse
import math
import re
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

# ---------------------------------------------------------------------------
# Files and settings
# ---------------------------------------------------------------------------

WORKSPACE = Path(r"D:\PPG2RSP_RRWaveNet_Inspired")
PROJECT_ROOT = WORKSPACE / "261107_MAE_Boxplots_4Models"
WAVEFORM_SUMMARY = (WORKSPACE / "261107_Boxplots_DeepCorrencoderPaper_RespDiff_RRWaveNet"
                    / "Waveform_Summary_all_runs.xlsx")
CORRENCODER_FOLDER = WORKSPACE / "261107_DeepCorrencoder_PaperVersion_v1"
CORRENCODER_RESULTS_FILE = Path("Summary") / "261107_DeepCorrencoder_PaperVersion_v1_results.xlsx"
CORRENCODER_WINDOW_SEC = 60.6
# results folder of the paper-style runs (the first of these that holds the window folders)
PAPERSTYLE_ROOTS = (
    WORKSPACE / "261106_DeepCorrencoder_PaperStyle_All" / "RRWaveNet_PaperStyle_RR",
    WORKSPACE / "261106_RRWaveNet_PaperStyle_RR" / "261106_RRWaveNet_PaperStyle_RR_Results",
    WORKSPACE / "261106_RRWaveNet_PaperStyle_RR_Results",
)
PAPERSTYLE_WINDOW = 16
PAPERSTYLE_CASE = "low_only"

DATASETS = ("STEAM2", "BIDMC", "CapnoBase")  # panel order (a), (b), (c)
RRWAVENET_RUN = {"CapnoBase": "261007", "BIDMC": "261013", "STEAM2": "261013"}
SIZE_SEC = 60
EXCLUDED = {"STEAM2": ("22_LMS", "26_CKK", "12_YJJ"), "BIDMC": ("27",), "CapnoBase": ("0115",)}
ID_DIGITS = {"CapnoBase": 4, "BIDMC": 2}

MODELS = ("RRWaveNet", "RRWaveNet-RR", "Deep corr-encoder", "RespDiff")
LABEL = {"RRWaveNet": "RRWaveNet", "RRWaveNet-RR": "RRWaveNet-RR\n(W16)", "Deep corr-encoder": "Corr-encoder",
         "RespDiff": "RespDiff"}
SHORT = {"RRWaveNet": "RRWaveNet", "RRWaveNet-RR": "RRWaveNet-RR", "Deep corr-encoder": "Corr-encoder",
         "RespDiff": "RespDiff"}
LEGEND = {"RRWaveNet": "RRWaveNet waveform\n(STEAM2 / BIDMC 261013,\nCapnoBase 261007)",
          "RRWaveNet-RR": "RRWaveNet RR regression\n(paper style, W16,\n0.1-0.5 Hz input)",
          "Deep corr-encoder": "Deep corr-encoder\n(261107 paper version)", "RespDiff": "RespDiff"}
# categorical slots of the reference palette, adjacent pairs checked with the palette validator
COLOR = {"RRWaveNet": "#1baf7a", "RRWaveNet-RR": "#6250d6", "Deep corr-encoder": "#eb6834", "RespDiff": "#2a78d6"}
INK, MUTED, GRID = "#1f1f1f", "#6b6a66", "#e4e3df"
PAIRS = tuple((MODELS[i], MODELS[j]) for i in range(len(MODELS)) for j in range(i + 1, len(MODELS)))
NOTE = ("Only significant pairs are marked:\npaired Wilcoxon signed-rank test\n(same subjects), Benjamini-Hochberg\n"
        "over the 6 pairs of the dataset.\n* < 0.05, ** < 0.01,\n*** < 0.001, **** < 1e-4\n\n"
        "Friedman: the four models\ntogether (subjects with all four).\n\n"
        "MAE (breaths/min), mean over the\nsubject's windows:\n"
        "RRWaveNet, RespDiff: |FFT rate -\nGT rate from the breaths|, 60 s.\n"
        "RRWaveNet-RR: |predicted RR -\nreference RR|, 16 s windows.\n"
        "Corr-encoder: its stored result\n(60.6 s windows, paper evaluation).")


# ---------------------------------------------------------------------------
# Reading the result files
# ---------------------------------------------------------------------------

def norm_id(dataset: str, subject) -> str:
    """Subject IDs as the data files name them (CapnoBase 0009, BIDMC 01), also when a cell lost the zeros."""
    text = re.sub(r"\.0+$", "", str(subject).strip())
    if dataset in ID_DIGITS and text.isdigit():
        text = text.zfill(ID_DIGITS[dataset])
    return text


def rows_of(dataset: str, subjects, values, model: str, source: str) -> List[dict]:
    return [{"Dataset": dataset, "Subject": norm_id(dataset, s), "Model": model, "Value": float(v), "Source": source}
            for s, v in zip(subjects, pd.to_numeric(pd.Series(list(values)), errors="coerce")) if np.isfinite(v)]


def waveform_values(path: Path) -> List[dict]:
    """RRWaveNet and RespDiff: Subject_Summary (60 s) of the waveform comparison."""
    summary = pd.read_excel(path, sheet_name="Subject_Summary", dtype={"Subject": str})
    summary["Run"] = summary["Run"].astype(str).str.replace(r"\.0+$", "", regex=True)
    out = []
    for dataset in DATASETS:
        part = summary[(summary["Dataset"] == dataset) & (summary["Run"] == RRWAVENET_RUN[dataset])
                       & (pd.to_numeric(summary["Size_Sec"], errors="coerce") == SIZE_SEC)]
        for model in ("RRWaveNet", "RespDiff"):
            column = f"{model}_MAE"
            if column in part:
                out += rows_of(dataset, part["Subject"], part[column], model,
                               f"{path}: Subject_Summary {column} (Run {RRWAVENET_RUN[dataset]}, {SIZE_SEC} s)")
    return out


def correncoder_values(path: Path) -> List[dict]:
    """The deep corr-encoder's stored MAE (Per_subject, 60.6 s rows)."""
    frame = pd.read_excel(path, sheet_name="Per_subject", dtype={"Subject": str})
    frame = frame[np.isclose(pd.to_numeric(frame["Evaluation_Window_Sec"], errors="coerce"), CORRENCODER_WINDOW_SEC)]
    out = []
    for dataset in DATASETS:
        part = frame[frame["Dataset"] == dataset]
        out += rows_of(dataset, part["Subject"], part["MAE"], "Deep corr-encoder",
                       f"{path}: Per_subject MAE (Evaluation_Window_Sec {CORRENCODER_WINDOW_SEC:g})")
    return out


def find_paperstyle_root(window: int) -> Optional[Path]:
    for root in PAPERSTYLE_ROOTS:
        if any((root / d / f"W{window}" / "metrics").is_dir() for d in DATASETS):
            return root
    return None


def paperstyle_values(root: Path, window: int) -> Tuple[List[dict], Dict[str, str]]:
    """The paper-style RR regression: Fold_MAE_Mean of its PerSubject table (final, else running)."""
    out, state = [], {}
    for dataset in DATASETS:
        metrics = root / dataset / f"W{window}" / "metrics"
        final = metrics / "RRWaveNet_PaperStyle_Final_PerSubject.csv"
        running = metrics / "RRWaveNet_PaperStyle_Running_PerSubject.csv"
        path = final if final.is_file() else running if running.is_file() else None
        if path is None:
            state[dataset] = "no results"
            print(f"[RRWaveNet-RR] {dataset} W{window}: no results in {metrics}")
            continue
        frame = pd.read_csv(path, dtype={"Test_Subject": str}, encoding="utf-8-sig")
        if "Input_Case" in frame:
            frame = frame[frame["Input_Case"].astype(str) == PAPERSTYLE_CASE]
        state[dataset] = "finished" if path == final else "not finished (running table)"
        if path != final:
            print(f"[RRWaveNet-RR] {dataset} W{window}: not finished, its running table is used "
                  f"({frame['Test_Subject'].nunique()} subjects so far)")
        out += rows_of(dataset, frame["Test_Subject"], frame["Fold_MAE_Mean"], "RRWaveNet-RR",
                       f"{path}: Fold_MAE_Mean ({PAPERSTYLE_CASE}, W{window})"
                       + ("" if path == final else " [not finished]"))
    return out, state


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


def tests(values: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    friedman, pairwise = [], []
    for dataset in DATASETS:
        wide = values[values["Dataset"] == dataset].pivot_table(index="Subject", columns="Model", values="Value",
                                                                 aggfunc="first").reindex(columns=list(MODELS))
        full = wide.dropna()
        chi2 = p = math.nan
        if len(full) >= 3:
            try:
                chi2, p = stats.friedmanchisquare(*[full[m] for m in MODELS])
            except ValueError:
                pass
        friedman.append({"Dataset": dataset, "N_Subjects_All_Four": len(full), "Chi2": chi2,
                         "Kendall_W": chi2 / (len(full) * (len(MODELS) - 1)) if len(full) and np.isfinite(chi2)
                         else math.nan, "P": p,
                         **{f"N_{SHORT[m]}": int(wide[m].notna().sum()) for m in MODELS},
                         **{f"Mean_{SHORT[m]}": wide[m].mean() for m in MODELS},
                         **{f"SD_{SHORT[m]}": wide[m].std() for m in MODELS},
                         **{f"Median_{SHORT[m]}": wide[m].median() for m in MODELS}})
        rows = []
        for a, b in PAIRS:
            pair = wide[[a, b]].dropna()
            diff = pair[a] - pair[b]
            p_pair = math.nan
            if len(pair) >= 2 and (diff != 0).any():
                try:
                    p_pair = float(stats.wilcoxon(pair[a], pair[b]).pvalue)
                except ValueError:
                    pass
            rows.append({"Dataset": dataset, "A": a, "B": b, "N_Pairs": len(pair), "A_Median": pair[a].median(),
                         "B_Median": pair[b].median(), "N_A_Lower": int((diff < 0).sum()),
                         "N_B_Lower": int((diff > 0).sum()), "P": p_pair})
        for row, q in zip(rows, bh([r["P"] for r in rows])):
            row["FDR_within_dataset"] = q
            row["Stars"] = stars(q)
            row["Better"] = (row["A"] if row["A_Median"] < row["B_Median"] else row["B"]) \
                if np.isfinite(q) and q < 0.05 else ""
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


def bracket_levels(pairs: List[Tuple[int, int]]) -> List[int]:
    """Stacking level of every bracket (shortest first): a bracket goes on the lowest level where it neither
    overlaps nor touches another one."""
    levels, placed = [], []
    for x1, x2 in pairs:
        level = 0
        while any(lv == level and not (x2 < a or x1 > b) for (a, b), lv in placed):
            level += 1
        placed.append(((x1, x2), level))
        levels.append(level)
    return levels


def draw(values: pd.DataFrame, fr: pd.DataFrame, pw: pd.DataFrame, points: bool, out: Path,
         state: Dict[str, str]) -> int:
    vals = {(d, m): values[(values["Dataset"] == d) & (values["Model"] == m)]["Value"].to_numpy(float)
            for d in DATASETS for m in MODELS}
    present = [v for v in vals.values() if len(v)]
    if not present:
        print("no values; no figure")
        return 0
    if points:
        lo, top = min(v.min() for v in present), max(v.max() for v in present)
    else:
        ranges = [whisker_range(v) for v in present]
        lo, top = min(r[0] for r in ranges), max(r[1] for r in ranges)
    span = max(top - lo, 1e-6)
    sig = pw[pw["FDR_within_dataset"] < 0.05]
    plans = {}
    for dataset in DATASETS:
        ordered = sorted((r for r in sig[sig["Dataset"] == dataset].itertuples()),
                         key=lambda r: abs(MODELS.index(r.B) - MODELS.index(r.A)))
        spans = [tuple(sorted((MODELS.index(r.A), MODELS.index(r.B)))) for r in ordered]
        plans[dataset] = list(zip(ordered, spans, bracket_levels(spans)))
    n_levels = max([lv + 1 for plan in plans.values() for _, _, lv in plan] + [1])
    ylim = (lo - 0.05 * span, top + 0.05 * span + n_levels * 0.09 * span + 0.03 * span)
    step = nice_step(span)
    drawn: List[tuple] = []

    def panel(ax, dataset: str, ptitle: str, show_ylabel: bool) -> None:
        rng = np.random.default_rng(0)
        for j, model in enumerate(MODELS):
            v = vals[(dataset, model)]
            if not len(v):
                ax.text(j, 0.5, "no results", transform=ax.get_xaxis_transform(), ha="center", va="center",
                        fontsize=8.5, color=MUTED)
                continue
            box = ax.boxplot([v], positions=[j], widths=0.62, patch_artist=True, showfliers=False,
                             medianprops={"color": INK, "lw": 1.8}, whiskerprops={"color": MUTED, "lw": 1.1},
                             capprops={"color": MUTED, "lw": 1.1})
            box["boxes"][0].set(facecolor=COLOR[model], alpha=0.3 if points else 0.45, edgecolor=COLOR[model], lw=1.2)
            if points:
                ax.scatter(j + rng.uniform(-0.18, 0.18, len(v)), v, s=9, color=COLOR[model], edgecolors="white",
                           lw=0.3, zorder=3)
            ax.text(j, -0.13, f"median {np.median(v):.2f}\nn = {len(v)}", transform=ax.get_xaxis_transform(),
                    ha="center", va="top", fontsize=7, color=MUTED)
        local = [(v.max() if points else whisker_range(v)[1]) for m in MODELS for v in [vals[(dataset, m)]] if len(v)]
        local_top = max(local) if local else top
        for r, (x1, x2), level in plans[dataset]:
            y = local_top + 0.05 * span + level * 0.09 * span
            ax.plot([x1 + 0.04, x1 + 0.04, x2 - 0.04, x2 - 0.04], [y, y + 0.02 * span, y + 0.02 * span, y],
                    color=INK, lw=0.9, clip_on=False)
            ax.text((x1 + x2) / 2, y + 0.022 * span, r.Stars, ha="center", va="bottom", fontsize=9.5, color=INK)
            drawn.append((dataset, r.A, r.B))
        f = fr[fr["Dataset"] == dataset].iloc[0]
        head = (f"Friedman p = {pstr(f['P'])}, Kendall's W = {f['Kendall_W']:.2f}, n = {f['N_Subjects_All_Four']}"
                if np.isfinite(f["Kendall_W"]) else f"Friedman: n/a (n = {f['N_Subjects_All_Four']} with all four)")
        flag = "" if state.get(dataset, "finished") == "finished" else f"\nRRWaveNet-RR: {state[dataset]}"
        ax.set_title(f"{ptitle}\n{head}{flag}", loc="left", fontsize=9.5, pad=8)
        ax.set_ylim(*ylim)
        ticks = np.arange(math.floor(ylim[0] / step) * step, top + 1e-9, step)
        ax.set_yticks([t for t in ticks if ylim[0] <= t <= top + 0.02 * span])
        ax.set_xticks(range(len(MODELS)))
        ax.set_xticklabels([LABEL[m] for m in MODELS], fontsize=8.5)
        if show_ylabel:
            ax.set_ylabel("Subject RR MAE (breaths/min)", fontsize=9)
        ax.grid(True, axis="y", color=GRID, lw=0.7)
        ax.set_axisbelow(True)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)

    handles = [plt.Rectangle((0, 0), 1, 1, facecolor=COLOR[m], alpha=0.5, edgecolor=COLOR[m]) for m in MODELS]
    sub = ("box: median and IQR, whiskers 1.5 IQR" + ("; every subject a point" if points else " (points not drawn)")
           + "; the same y range in every panel")
    name = "MAE_4models_per_dataset" + ("" if points else "_no_points")
    fig, axes = plt.subplots(1, 3, figsize=(20, 6.6))
    fig.subplots_adjust(left=0.045, right=0.83, top=0.78, bottom=0.15, wspace=0.16)
    for c, dataset in enumerate(DATASETS):
        panel(axes[c], dataset, f"({'abc'[c]}) {dataset}", c == 0)
    fig.legend(handles, [LEGEND[m] for m in MODELS], loc="upper left", bbox_to_anchor=(0.842, 0.80), frameon=False,
               fontsize=8.5)
    fig.text(0.842, 0.45, NOTE, fontsize=7.8, color=MUTED, va="top")
    fig.text(0.045, 0.94, "Respiratory rate MAE: four models compared within each dataset (lower = better)",
             fontsize=13, color=INK)
    fig.text(0.045, 0.905, sub, fontsize=9, color=MUTED)
    fig.savefig(out / f"{name}.png", dpi=200)
    plt.close(fig)
    significant = {(r.Dataset, r.A, r.B) for r in sig.itertuples()}
    assert significant == set(drawn), (significant, drawn)
    run_of = {"STEAM2": "261013", "BIDMC": "261013", "CapnoBase": "261007"}
    for dataset in DATASETS:
        fig, ax = plt.subplots(figsize=(9.6, 6.6))
        fig.subplots_adjust(left=0.09, right=0.64, top=0.78, bottom=0.15)
        panel(ax, dataset, dataset, True)
        labels = [f"RRWaveNet waveform ({run_of[dataset]})"] + [LEGEND[m] for m in MODELS[1:]]
        fig.legend(handles, labels, loc="upper left", bbox_to_anchor=(0.66, 0.80), frameon=False, fontsize=8.5)
        fig.text(0.66, 0.47, NOTE, fontsize=7.5, color=MUTED, va="top")
        fig.text(0.09, 0.94, f"{dataset}: respiratory rate MAE of four models (lower = better)", fontsize=12,
                 color=INK)
        fig.text(0.09, 0.905, sub.replace("; the same y range in every panel", ""), fontsize=8.5, color=MUTED)
        fig.savefig(out / f"{name}_{dataset}.png", dpi=200)
        plt.close(fig)
    return len(significant)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="261107 MAE box plots of four models")
    parser.add_argument("--waveform-summary", type=Path, default=WAVEFORM_SUMMARY,
                        help="Waveform_Summary_all_runs.xlsx (RRWaveNet and RespDiff)")
    parser.add_argument("--correncoder-root", type=Path, default=CORRENCODER_FOLDER)
    parser.add_argument("--correncoder-results", type=Path, default=None,
                        help=f"default: {{correncoder-root}}\\{CORRENCODER_RESULTS_FILE}")
    parser.add_argument("--paperstyle-root", type=Path, default=None,
                        help="results folder of the paper-style runs (default: the first found of "
                             + ", ".join(str(r) for r in PAPERSTYLE_ROOTS) + ")")
    parser.add_argument("--paperstyle-window", type=int, default=PAPERSTYLE_WINDOW)
    parser.add_argument("--output-root", type=Path, default=PROJECT_ROOT)
    parser.add_argument("--exclude-steam2", nargs="*", default=list(EXCLUDED["STEAM2"]))
    parser.add_argument("--exclude-bidmc", nargs="*", default=list(EXCLUDED["BIDMC"]))
    parser.add_argument("--exclude-capnobase", nargs="*", default=list(EXCLUDED["CapnoBase"]))
    args = parser.parse_args(argv)
    if args.correncoder_results is None:
        args.correncoder_results = Path(args.correncoder_root) / CORRENCODER_RESULTS_FILE
    return args


def main(argv: Optional[Sequence[str]] = None) -> None:
    args = parse_args(argv)
    excluded = {"STEAM2": [norm_id("STEAM2", s) for s in args.exclude_steam2],
                "BIDMC": [norm_id("BIDMC", s) for s in args.exclude_bidmc],
                "CapnoBase": [norm_id("CapnoBase", s) for s in args.exclude_capnobase]}
    for path in (args.waveform_summary, args.correncoder_results):
        if not Path(path).is_file():
            raise SystemExit(f"Not found: {path}")
    ps_root = Path(args.paperstyle_root) if args.paperstyle_root else find_paperstyle_root(args.paperstyle_window)
    if ps_root is None or not ps_root.is_dir():
        raise SystemExit("Results folder of the paper-style runs not found; give --paperstyle-root")
    print(f"RRWaveNet / RespDiff: {args.waveform_summary}\nCorr-encoder: {args.correncoder_results}\n"
          f"RRWaveNet-RR: {ps_root} (W{args.paperstyle_window})")
    ps_rows, state = paperstyle_values(ps_root, args.paperstyle_window)
    values = pd.DataFrame(waveform_values(Path(args.waveform_summary))
                          + correncoder_values(Path(args.correncoder_results)) + ps_rows)
    removed = []
    for dataset, subjects in excluded.items():
        for subject in subjects:
            hit = (values["Dataset"] == dataset) & (values["Subject"] == subject)
            removed.append({"Dataset": dataset, "Subject": subject, "Rows_Removed": int(hit.sum()),
                            "Models": ", ".join(sorted(values.loc[hit, "Model"])) if hit.any() else "",
                            "Status": "left out of every model" if hit.any() else "not found"})
    values = values[~values.apply(lambda r: r["Subject"] in excluded.get(r["Dataset"], []), axis=1)]
    values = values.reset_index(drop=True)

    out = Path(args.output_root)
    figures = out / "Figures"
    figures.mkdir(parents=True, exist_ok=True)
    fr, pw = tests(values)
    counts = [draw(values, fr, pw, points, figures, state) for points in (True, False)]
    wide = values.pivot_table(index=["Dataset", "Subject"], columns="Model", values="Value", aggfunc="first")
    wide = wide.reindex(columns=[m for m in MODELS if m in wide.columns])
    wide.columns = [f"MAE__{SHORT[m]}" for m in wide.columns]
    sources = values.drop_duplicates(["Dataset", "Model"])[["Dataset", "Model", "Source"]]
    sources = pd.concat([sources, pd.DataFrame([{"Dataset": d, "Model": "RRWaveNet-RR", "Source": f"state: {s}"}
                                                for d, s in state.items()])], ignore_index=True)
    tables = {"Friedman": fr, "Pairwise": pw, "Values": wide.reset_index(), "Sources": sources,
              "Excluded": pd.DataFrame(removed)}
    with pd.ExcelWriter(out / "261107_MAE_4models_Tests.xlsx", engine="openpyxl") as writer:
        for name, frame in tables.items():
            frame.to_excel(writer, sheet_name=name, index=False)
    for name, frame in tables.items():
        frame.to_csv(out / f"{name}.csv", index=False, encoding="utf-8-sig")
    pd.set_option("display.width", 240)
    print("\nMAE, Mean ± SD (Median) over subjects (MAE-issue subjects left out):")
    for _, row in fr.iterrows():
        print(f"  {row['Dataset']:<9s} n(all four)={row['N_Subjects_All_Four']:<3d} Friedman p={pstr(row['P'])}  "
              + " | ".join(f"{SHORT[m]} {row[f'Mean_{SHORT[m]}']:.2f} ± {row[f'SD_{SHORT[m]}']:.2f} "
                           f"({row[f'Median_{SHORT[m]}']:.2f})" for m in MODELS))
    sig = pw[pw["FDR_within_dataset"] < 0.05]
    print(sig[["Dataset", "A", "B", "N_Pairs", "A_Median", "B_Median", "FDR_within_dataset", "Stars", "Better"]]
          .round(4).to_string(index=False) if not sig.empty else "no significant pair")
    print(f"significant pairs drawn (with points, without): {counts}")
    print(f"Saved under: {out}")


if __name__ == "__main__":
    main()
