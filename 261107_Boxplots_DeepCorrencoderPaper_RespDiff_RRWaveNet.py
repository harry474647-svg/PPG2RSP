# -*- coding: utf-8 -*-
r"""261107_Boxplots_DeepCorrencoderPaper_RespDiff_RRWaveNet: per dataset, the
three models compared on the same subjects, with the subjects that had an MAE
issue left out:
    RRWaveNet           CapnoBase 261007, BIDMC 261013, STEAM2 261013
    Deep corr-encoder   261107_DeepCorrencoder_PaperVersion_v1 (the paper's model, LOSOCV)
    RespDiff            261020_RespDiff_v2 (ddpm50)
Measures: respiratory-rate MAE, waveform PCC and duty cycle error, each with
the same rule for the three models. Nothing is trained or measured again here:
the values are read from the result workbooks below.

Values (per subject)
--------------------
MAE and PCC: Waveform_Summary_all_runs.xlsx of 261106_RRWaveNet_Performance_
Boxplots_v2, run with the paper-version corr-encoder as its corr-encoder
(--correncoder-root, below); sheet Subject_Summary, Size_Sec = 60, Run 261007
for CapnoBase and 261013 for BIDMC / STEAM2; columns RRWaveNet_MAE /
RRWaveNet_PCC, CorrEncoder_MAE / CorrEncoder_PCC, RespDiff_MAE / RespDiff_PCC.
  Rule (v2, the same for the three models): every model against the GT as
  recorded at 30 Hz, no time shift, a model inverted once per recording when
  its r with the GT is negative, RRWaveNet with its training sign / lag undone
  and low-pass 1 Hz; non-overlapping 60 s windows; AE = |FFT rate of the
  model's waveform - GT rate from the breaths| (breaths/min), PCC with the GT;
  subject MAE / PCC = mean over its windows.
  The file is checked: its Notes must name the paper-version folder as the
  corr-encoder (else the script stops and prints the v2 command; with
  --allow-other-correncoder it goes on).
Duty cycle error (rule of 261107_DutyCycle_RespDiff_DeepCorrencoder, the
official RespDiff code; Duty_MAE = mean |model - GT duty cycle| over the 60 s
windows moved by 20 s):
  RRWaveNet  261107_DutyCycle_Results.xlsx, sheet RRWaveNet_Results (Run 261007
             CapnoBase, 261013 BIDMC / STEAM2)
  RespDiff   261107_DutyCycle_Results.xlsx, sheet RespDiff_Duty
  Corr-enc.  261107_DeepCorrencoder_PaperVersion_v1_results.xlsx (Summary
             folder of the paper version), sheet Per_subject (computed there
             with the same functions)

Subjects left out (MAE issue; subject set 04 of the RRWaveNet runs), of every
model: STEAM2 22_LMS, 26_CKK, 12_YJJ; BIDMC 27; CapnoBase 0115 (--exclude-*).

Statistics (per dataset and measure)
------------------------------------
Friedman test of the three models (subjects with all three), Kendall's W;
paired Wilcoxon signed-rank tests of the three pairs, Benjamini-Hochberg over
the 3 pairs; only significant pairs (FDR < 0.05) are marked (* < 0.05,
** < 0.01, *** < 0.001, **** < 1e-4). A box holds every subject the model has.

Figures (no reference lines)
----------------------------
{MAE, PCC, Duty_error}_per_dataset.png              (a) STEAM2 (b) BIDMC (c) CapnoBase, every subject a point
{MAE, PCC, Duty_error}_per_dataset_no_points.png    the same without the points
... _{STEAM2, BIDMC, CapnoBase}.png                  one dataset per figure
and 261107_Boxplot_Tests.xlsx (Friedman, Pairwise, Values, Sources, Excluded).

Before the first run (once the paper version has finished): the v2 waveform
comparison with the paper-version corr-encoder, into this project's folder:
  python 261106_RRWaveNet_Performance_Boxplots_v2.py --runs 261007 261013 --sizes 60
      --correncoder-root D:\PPG2RSP_RRWaveNet_Inspired\261107_DeepCorrencoder_PaperVersion_v1
      --output-root D:\PPG2RSP_RRWaveNet_Inspired\261107_Boxplots_DeepCorrencoderPaper_RespDiff_RRWaveNet\v2_with_PaperCE
Then:
  python 261107_Boxplots_DeepCorrencoderPaper_RespDiff_RRWaveNet.py
Options: --waveform-summary, --duty-results, --correncoder-results (the three
workbooks), --output-root, --exclude-steam2 / --exclude-bidmc /
--exclude-capnobase, --allow-other-correncoder
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
PROJECT_ROOT = WORKSPACE / "261107_Boxplots_DeepCorrencoderPaper_RespDiff_RRWaveNet"
PAPER_CE_FOLDER = "261107_DeepCorrencoder_PaperVersion_v1"
WAVEFORM_SUMMARY = PROJECT_ROOT / "v2_with_PaperCE" / "Waveform_Summary_all_runs.xlsx"
DUTY_RESULTS = WORKSPACE / "261107_DutyCycle_RespDiff_DeepCorrencoder" / "261107_DutyCycle_Results.xlsx"
CORRENCODER_RESULTS = (WORKSPACE / PAPER_CE_FOLDER / "Summary"
                       / "261107_DeepCorrencoder_PaperVersion_v1_results.xlsx")

DATASETS = ("STEAM2", "BIDMC", "CapnoBase")  # panel order (a), (b), (c)
RRWAVENET_RUN = {"CapnoBase": "261007", "BIDMC": "261013", "STEAM2": "261013"}
SIZE_SEC = 60
CORRENCODER_WINDOW_SEC = 60.6  # the paper version's per-subject rows (duty is the same in both rows)
EXCLUDED = {"STEAM2": ("22_LMS", "26_CKK", "12_YJJ"), "BIDMC": ("27",), "CapnoBase": ("0115",)}
ID_DIGITS = {"CapnoBase": 4, "BIDMC": 2}

MODELS = ("RRWaveNet", "Deep corr-encoder", "RespDiff")
LABEL = {"RRWaveNet": "RRWaveNet", "Deep corr-encoder": "Corr-encoder", "RespDiff": "RespDiff"}
LEGEND = {"RRWaveNet": "RRWaveNet\n(STEAM2 / BIDMC 261013,\nCapnoBase 261007)",
          "Deep corr-encoder": "Deep corr-encoder\n(261107 paper version)", "RespDiff": "RespDiff"}
COLOR = {"RRWaveNet": "#1baf7a", "Deep corr-encoder": "#eb6834", "RespDiff": "#2a78d6"}
INK, MUTED, GRID = "#1f1f1f", "#6b6a66", "#e4e3df"
PAIRS = (("RRWaveNet", "Deep corr-encoder"), ("Deep corr-encoder", "RespDiff"), ("RRWaveNet", "RespDiff"))

# measure -> (better, title, y label, note)
MEASURES = {
    "MAE": ("lower", "Respiratory rate MAE", "Subject RR MAE (breaths/min), 60 s windows",
            "MAE: |FFT rate of the model\nwaveform - GT rate from the\nbreaths|, mean over the subject's\n"
            "60 s windows (v2 rule, the same\nfor the three models)."),
    "PCC": ("higher", "Waveform PCC", "Subject waveform PCC with the recorded GT (r)",
            "PCC: Pearson r of model and\nrecorded GT, mean over the\nsubject's 60 s windows; no time\n"
            "shift, sign matched once per\nrecording (v2 rule)."),
    "Duty_error": ("lower", "Duty cycle error", "Subject duty cycle error |model - GT| (0-1)",
                   "Duty = share of time the [0, 1]\nwaveform is >= 0.5 in 60 s\nwindows (RespDiff definition)."),
}


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


def waveform_values(path: Path, allow_other: bool) -> Tuple[pd.DataFrame, str]:
    """MAE and PCC of the three models (v2 Subject_Summary, 60 s)."""
    notes = pd.read_excel(path, sheet_name="Notes")
    models = " ".join(str(v) for v in notes.loc[notes["Item"].astype(str) == "Models", "Description"])
    if PAPER_CE_FOLDER.lower() not in models.lower():
        message = (f"{path}: its corr-encoder is not the paper version ({models or 'Notes: no Models row'}).\n"
                   "Run the v2 waveform comparison with the paper-version corr-encoder first:\n"
                   "  python 261106_RRWaveNet_Performance_Boxplots_v2.py --runs 261007 261013 --sizes 60 "
                   f"--correncoder-root {WORKSPACE / PAPER_CE_FOLDER} --output-root {WAVEFORM_SUMMARY.parent}")
        if not allow_other:
            raise SystemExit(message)
        print("[Warning] " + message.splitlines()[0] + " (--allow-other-correncoder: used anyway)")
    summary = read_sheet(path, "Subject_Summary")
    summary["Run"] = summary["Run"].astype(str).str.replace(r"\.0+$", "", regex=True)
    rows = []
    for dataset in DATASETS:
        part = summary[(summary["Dataset"] == dataset) & (summary["Run"] == RRWAVENET_RUN[dataset])
                       & (pd.to_numeric(summary["Size_Sec"], errors="coerce") == SIZE_SEC)]
        for measure in ("MAE", "PCC"):
            for model, prefix in (("RRWaveNet", "RRWaveNet"), ("Deep corr-encoder", "CorrEncoder"),
                                  ("RespDiff", "RespDiff")):
                column = f"{prefix}_{measure}"
                if column not in part:
                    continue
                for subject, value in zip(part["Subject"], pd.to_numeric(part[column], errors="coerce")):
                    rows.append({"Dataset": dataset, "Subject": subject, "Model": model, "Measure": measure,
                                 "Value": value, "Source": f"{path.name}: Subject_Summary {column} "
                                                           f"(Run {RRWAVENET_RUN[dataset]}, {SIZE_SEC} s)"})
    return pd.DataFrame(rows), models


def duty_values(duty_path: Path, ce_path: Path) -> pd.DataFrame:
    """Duty_MAE of RRWaveNet and RespDiff (261107 duty workbook) and of the paper-version corr-encoder."""
    rows = []
    rr = read_sheet(duty_path, "RRWaveNet_Results")
    rr["Run"] = rr["Run"].astype(str).str.replace(r"\.0+$", "", regex=True)
    rd = read_sheet(duty_path, "RespDiff_Duty")
    ce = read_sheet(ce_path, "Per_subject")
    ce = ce[np.isclose(pd.to_numeric(ce["Evaluation_Window_Sec"], errors="coerce"), CORRENCODER_WINDOW_SEC)]
    for dataset in DATASETS:
        sources = (
            ("RRWaveNet", rr[(rr["Dataset"] == dataset) & (rr["Run"] == RRWAVENET_RUN[dataset])],
             f"{duty_path.name}: RRWaveNet_Results Duty_MAE (Run {RRWAVENET_RUN[dataset]})"),
            ("Deep corr-encoder", ce[ce["Dataset"] == dataset], f"{ce_path.name}: Per_subject Duty_MAE"),
            ("RespDiff", rd[rd["Dataset"] == dataset], f"{duty_path.name}: RespDiff_Duty Duty_MAE"),
        )
        for model, part, source in sources:
            if "Duty_MAE" not in part:
                continue
            for subject, value in zip(part["Subject"], pd.to_numeric(part["Duty_MAE"], errors="coerce")):
                rows.append({"Dataset": dataset, "Subject": subject, "Model": model, "Measure": "Duty_error",
                             "Value": value, "Source": source})
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
    for dataset in DATASETS:
        part = values[(values["Measure"] == measure) & (values["Dataset"] == dataset)]
        wide = part.pivot_table(index="Subject", columns="Model", values="Value", aggfunc="first")
        wide = wide.reindex(columns=list(MODELS))
        full = wide.dropna()
        chi2 = p = math.nan
        if len(full) >= 3:
            try:
                chi2, p = stats.friedmanchisquare(*[full[m] for m in MODELS])
            except ValueError:
                pass
        friedman.append({"Measure": measure, "Dataset": dataset, "N_Subjects_All_Three": len(full), "Chi2": chi2,
                         "Kendall_W": chi2 / (len(full) * 2) if len(full) and np.isfinite(chi2) else math.nan, "P": p,
                         **{f"N_{LABEL[m]}": int(wide[m].notna().sum()) for m in MODELS},
                         **{f"Median_{LABEL[m]}": wide[m].median() for m in MODELS},
                         **{f"Mean_{LABEL[m]}": wide[m].mean() for m in MODELS},
                         **{f"SD_{LABEL[m]}": wide[m].std() for m in MODELS}})
        rows = []
        for a, b in PAIRS:
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


def draw(values: pd.DataFrame, fr: pd.DataFrame, pw: pd.DataFrame, measure: str, points: bool, out: Path,
         excluded: Dict[str, Sequence[str]]) -> int:
    better, title, ylabel, note_extra = MEASURES[measure]
    data = values[values["Measure"] == measure]
    vals = {(d, m): data[(data["Dataset"] == d) & (data["Model"] == m)]["Value"].dropna().to_numpy(float)
            for d in DATASETS for m in MODELS}
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
        for j, model in enumerate(MODELS):
            v = vals[(dataset, model)]
            if not len(v):
                ax.text(j, 0.5, "no results", transform=ax.get_xaxis_transform(), ha="center", va="center",
                        fontsize=9, color=MUTED)
                continue
            box = ax.boxplot([v], positions=[j], widths=0.62, patch_artist=True, showfliers=False,
                             medianprops={"color": INK, "lw": 1.8}, whiskerprops={"color": MUTED, "lw": 1.1},
                             capprops={"color": MUTED, "lw": 1.1})
            box["boxes"][0].set(facecolor=COLOR[model], alpha=0.3 if points else 0.45, edgecolor=COLOR[model], lw=1.2)
            if points:
                ax.scatter(j + rng.uniform(-0.18, 0.18, len(v)), v, s=10, color=COLOR[model], edgecolors="white",
                           lw=0.3, zorder=3)
            ax.text(j, -0.1, f"median {np.median(v):.3f}\nn = {len(v)}", transform=ax.get_xaxis_transform(),
                    ha="center", va="top", fontsize=7.5, color=MUTED)
        local = [(v.max() if points else whisker_range(v)[1]) for m in MODELS
                 for v in [vals[(dataset, m)]] if len(v)]
        local_top = max(local) if local else top
        sig = pw[(pw["Dataset"] == dataset) & (pw["FDR_within_dataset"] < 0.05)]
        ordered = sorted(sig.itertuples(), key=lambda r: abs(MODELS.index(r.B) - MODELS.index(r.A)))
        for level, r in enumerate(ordered):
            x1, x2 = sorted((MODELS.index(r.A), MODELS.index(r.B)))
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
        ax.set_xticks(range(len(MODELS)))
        ax.set_xticklabels([LABEL[m] for m in MODELS], fontsize=9.5)
        if show_ylabel:
            ax.set_ylabel(ylabel, fontsize=9)
        ax.grid(True, axis="y", color=GRID, lw=0.7)
        ax.set_axisbelow(True)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)

    handles = [plt.Rectangle((0, 0), 1, 1, facecolor=COLOR[m], alpha=0.5, edgecolor=COLOR[m]) for m in MODELS]
    note = ("Only significant pairs are marked:\npaired Wilcoxon signed-rank test\n(same subjects), Benjamini-Hochberg\n"
            "over the 3 pairs of the dataset.\n* < 0.05, ** < 0.01,\n*** < 0.001, **** < 1e-4\n\n"
            "Friedman: the three models\ntogether (subjects with all three).\n\n" + note_extra)
    sub = ("box: median and IQR, whiskers 1.5 IQR" + ("; every subject a point" if points else " (points not drawn)")
           + "; the same y range in every panel")
    left_out = "; ".join(f"{d} {', '.join(excluded[d])}" for d in DATASETS if excluded.get(d))
    head = f"{title}: the three models compared within each dataset ({better} = better)"
    name = f"{measure}_per_dataset" + ("" if points else "_no_points")

    fig, axes = plt.subplots(1, 3, figsize=(17, 6.2))
    fig.subplots_adjust(left=0.05, right=0.83, top=0.78, bottom=0.13, wspace=0.16)
    for c, dataset in enumerate(DATASETS):
        panel(axes[c], dataset, f"({'abc'[c]}) {dataset}", c == 0)
    fig.legend(handles, [LEGEND[m] for m in MODELS], loc="upper left", bbox_to_anchor=(0.845, 0.78), frameon=False,
               fontsize=9)
    fig.text(0.845, 0.46, note, fontsize=8.3, color=MUTED, va="top")
    fig.text(0.05, 0.94, head, fontsize=13, color=INK)
    fig.text(0.05, 0.905, sub, fontsize=9, color=MUTED)
    fig.text(0.05, 0.875, "Subjects with an MAE issue left out of every model: " + (left_out or "none"), fontsize=9,
             color=MUTED)
    fig.savefig(out / f"{name}.png", dpi=200)
    plt.close(fig)
    significant = {(r.Dataset, r.A, r.B) for r in pw.itertuples() if r.FDR_within_dataset < 0.05}
    assert significant == set(drawn), (measure, significant, drawn)

    run_of = {"STEAM2": "261013", "BIDMC": "261013", "CapnoBase": "261007"}
    for dataset in DATASETS:
        fig, ax = plt.subplots(figsize=(8.6, 6.2))
        fig.subplots_adjust(left=0.1, right=0.66, top=0.78, bottom=0.13)
        panel(ax, dataset, dataset, True)
        labels = [f"RRWaveNet ({run_of[dataset]})", LEGEND["Deep corr-encoder"], "RespDiff"]
        fig.legend(handles, labels, loc="upper left", bbox_to_anchor=(0.68, 0.78), frameon=False, fontsize=9)
        fig.text(0.68, 0.48, note, fontsize=8, color=MUTED, va="top")
        fig.text(0.1, 0.94, f"{dataset}: {title[0].lower() + title[1:]} of the three models ({better} = better)", fontsize=12,
                 color=INK)
        fig.text(0.1, 0.905, sub.replace("; the same y range in every panel", ""), fontsize=8.5, color=MUTED)
        fig.text(0.1, 0.875, "Subjects with an MAE issue left out of every model: "
                 + (", ".join(excluded.get(dataset, ())) or "none"), fontsize=8.5, color=MUTED)
        fig.savefig(out / f"{name}_{dataset}.png", dpi=200)
        plt.close(fig)
    return len(significant)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="261107 box plots: paper-version deep corr-encoder, RespDiff, "
                                                 "RRWaveNet (261007 CapnoBase, 261013 BIDMC / STEAM2)")
    parser.add_argument("--waveform-summary", type=Path, default=WAVEFORM_SUMMARY,
                        help="Waveform_Summary_all_runs.xlsx of v2 run with the paper-version corr-encoder")
    parser.add_argument("--duty-results", type=Path, default=DUTY_RESULTS, help="261107_DutyCycle_Results.xlsx")
    parser.add_argument("--correncoder-results", type=Path, default=CORRENCODER_RESULTS,
                        help="261107_DeepCorrencoder_PaperVersion_v1_results.xlsx")
    parser.add_argument("--output-root", type=Path, default=PROJECT_ROOT)
    parser.add_argument("--exclude-steam2", nargs="*", default=list(EXCLUDED["STEAM2"]))
    parser.add_argument("--exclude-bidmc", nargs="*", default=list(EXCLUDED["BIDMC"]))
    parser.add_argument("--exclude-capnobase", nargs="*", default=list(EXCLUDED["CapnoBase"]))
    parser.add_argument("--allow-other-correncoder", action="store_true",
                        help="go on when the waveform summary's corr-encoder is not the paper version")
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> None:
    args = parse_args(argv)
    excluded = {"STEAM2": [norm_id("STEAM2", s) for s in args.exclude_steam2],
                "BIDMC": [norm_id("BIDMC", s) for s in args.exclude_bidmc],
                "CapnoBase": [norm_id("CapnoBase", s) for s in args.exclude_capnobase]}
    for path in (args.waveform_summary, args.duty_results, args.correncoder_results):
        if not Path(path).is_file():
            raise SystemExit(f"Not found: {path}")
    waveform, models_note = waveform_values(Path(args.waveform_summary), args.allow_other_correncoder)
    values = pd.concat([waveform, duty_values(Path(args.duty_results), Path(args.correncoder_results))],
                       ignore_index=True)
    removed = []
    for dataset, subjects in excluded.items():
        for subject in subjects:
            hit = (values["Dataset"] == dataset) & (values["Subject"] == subject)
            removed.append({"Dataset": dataset, "Subject": subject, "Rows_Removed": int(hit.sum()),
                            "Status": "left out of every model" if hit.any() else "not found in the workbooks"})
    keep = ~values.apply(lambda r: r["Subject"] in excluded.get(r["Dataset"], []), axis=1)
    values = values[keep].reset_index(drop=True)

    out = Path(args.output_root)
    figures = out / "Figures"
    figures.mkdir(parents=True, exist_ok=True)
    friedman, pairwise, counts = [], [], {}
    for measure in MEASURES:
        fr, pw = tests(values, measure)
        friedman.append(fr)
        pairwise.append(pw)
        counts[measure] = [draw(values, fr, pw, measure, points, figures, excluded) for points in (True, False)]
    friedman = pd.concat(friedman, ignore_index=True)
    pairwise = pd.concat(pairwise, ignore_index=True)
    wide = values.pivot_table(index=["Dataset", "Subject"], columns=["Measure", "Model"], values="Value",
                              aggfunc="first")
    wide.columns = [f"{m}__{LABEL[model]}" for m, model in wide.columns]
    sources = values.drop_duplicates(["Dataset", "Model", "Measure"])[["Dataset", "Model", "Measure", "Source"]]
    sources = pd.concat([sources, pd.DataFrame([{"Dataset": "", "Model": "Deep corr-encoder", "Measure": "MAE / PCC",
                                                 "Source": f"waveform summary Notes: {models_note}"}])],
                        ignore_index=True)
    tables = {"Friedman": friedman, "Pairwise": pairwise, "Values": wide.reset_index(), "Sources": sources,
              "Excluded": pd.DataFrame(removed)}
    with pd.ExcelWriter(out / "261107_Boxplot_Tests.xlsx", engine="openpyxl") as writer:
        for name, frame in tables.items():
            frame.to_excel(writer, sheet_name=name, index=False)
    for name, frame in tables.items():
        frame.to_csv(out / f"{name}.csv", index=False, encoding="utf-8-sig")
    pd.set_option("display.width", 220)
    print(friedman[["Measure", "Dataset", "N_Subjects_All_Three", "P", "Kendall_W", "Median_RRWaveNet",
                    "Median_Corr-encoder", "Median_RespDiff"]].round(4).to_string(index=False))
    sig = pairwise[pairwise["FDR_within_dataset"] < 0.05]
    print(sig[["Measure", "Dataset", "A", "B", "N_Pairs", "FDR_within_dataset", "Stars", "Better"]]
          .round(5).to_string(index=False) if not sig.empty else "no significant pair")
    print("significant pairs drawn (with points, without):", counts)
    print(pd.DataFrame(removed).to_string(index=False))
    print(f"Saved under: {out}")


if __name__ == "__main__":
    main()
