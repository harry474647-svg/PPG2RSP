# -*- coding: utf-8 -*-
r"""261127_RRWaveNet_B6_Block_ChannelAbsPCC_PercentChange: the block comparison figure of the channel-averaged absolute
PCC for the six-block encoder of 261110_RRWaveNet_DeepEncoder_B6 (Fusion Output, 1st to 6th Block), one panel per
dataset (In-House = STEAM2, BIDMC, CapnoBase) side by side: y axis the mean PCC over the subjects, and at every point
the change from Fusion Output (Stem) in %.

Values: the PCC computed by 261110_RRWaveNet_DeepEncoder_B6 with the stored six-block models (its feature analysis,
Subjects sheet, {Stage}_Channel_AbsR_Mean; nothing is recomputed here and no value is taken from a figure)
  per subject: the 24-channel feature map of every 10 s test window, Pearson r of every channel with the target
  respiratory waveform, |r| averaged over the windows, then over the 24 channels (no PCA), then over the subject's
  models (m0-m2)
  Stem = stem fusion output (encoder input); B1-B6 = the output of the encoder blocks (dilation 1, 2, 4, 8, 16, 32)
Models (as the final results): In-House (STEAM2) and BIDMC from the six-block 261013 run (01_261013_B6, the weight decay
selected in the fold), CapnoBase from the six-block 261007 run (02_261007_B6, full); --source 261007 or 261013 draws
every dataset from one run.
Subjects: the MAE issue subjects of the final results are left out (CapnoBase 0115, BIDMC 27, STEAM2 12_YJJ, 22_LMS,
26_CKK), as in 261126_RRWaveNet_Block_ChannelAbsPCC_PercentChange; --all-subjects keeps them.
Change from Stem (%) = 100 x (mean PCC of the stage - mean PCC of Stem) / mean PCC of Stem, the means over the
subjects of the dataset.

Input: 261110_RRWaveNet_DeepEncoder_B6_Feature_Analysis.xlsx, found under
D:\PPG2RSP_RRWaveNet_Inspired\261110_RRWaveNet_DeepEncoder_B6\Feature_Analysis\ (ACF_2-10s_lag0-12s or
ACF_1-5s_lag0-10s; the channel PCC is the same in both), next to this script or in the working folder; --feature-xlsx
gives it directly.

Outputs (D:\PPG2RSP_RRWaveNet_Inspired\261127_RRWaveNet_B6_Block_ChannelAbsPCC_PercentChange)
  B6_block_channel_abs_pcc.png / .pdf                     the three datasets side by side
  261127_RRWaveNet_B6_Block_ChannelAbsPCC_PercentChange.xlsx
    Summary    per dataset and stage: subjects, mean PCC, SD, change from Stem (%)
    Subjects   the PCC of every subject and stage that went into the figure
    Notes

Run: python 261127_RRWaveNet_B6_Block_ChannelAbsPCC_PercentChange.py     (Spyder: runfile(...), no options needed)
Options: --feature-xlsx, --output-root, --source final|261007|261013, --all-subjects, --decimals
"""

import argparse
import sys
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.ticker import MaxNLocator

# ---------------------------------------------------------------------------
# Settings
# ---------------------------------------------------------------------------

NAME = "261127_RRWaveNet_B6_Block_ChannelAbsPCC_PercentChange"
WORKSPACE = Path(r"D:\PPG2RSP_RRWaveNet_Inspired")
OUTPUT_ROOT = WORKSPACE / NAME
B6_NAME = "261110_RRWaveNet_DeepEncoder_B6"
FEATURE_FILE = f"{B6_NAME}_Feature_Analysis.xlsx"
FEATURE_FOLDERS = ("ACF_2-10s_lag0-12s", "ACF_1-5s_lag0-10s")
FIGURE_NAME = "B6_block_channel_abs_pcc"

ENCODER = "Six blocks"
STAGES = ("Stem", "B1", "B2", "B3", "B4", "B5", "B6")
VALUE = "Channel_AbsR_Mean"
PANELS = ("STEAM2", "BIDMC", "CapnoBase")
TITLES = {"STEAM2": "In-House", "BIDMC": "BIDMC", "CapnoBase": "CapnoBase"}
SOURCES = {
    "final": {"STEAM2": "261013", "BIDMC": "261013", "CapnoBase": "261007"},
    "261007": {"STEAM2": "261007", "BIDMC": "261007", "CapnoBase": "261007"},
    "261013": {"STEAM2": "261013", "BIDMC": "261013", "CapnoBase": "261013"},
}
ISSUE_SUBJECTS = {"STEAM2": ("22_LMS", "26_CKK", "12_YJJ"), "BIDMC": ("27",), "CapnoBase": ("0115",)}

COLOR = "#1f77b4"
GRID = "#e6e6e6"
# Two lines each: seven stages do not fit on one line per panel
STAGE_LABELS = ("Fusion\nOutput", "1$^{\\mathbf{st}}$\nBlock", "2$^{\\mathbf{nd}}$\nBlock", "3$^{\\mathbf{rd}}$\nBlock",
                "4$^{\\mathbf{th}}$\nBlock", "5$^{\\mathbf{th}}$\nBlock", "6$^{\\mathbf{th}}$\nBlock")
Y_LABEL = "Channel-Averaged PCC"
# Arial (Windows); Liberation Sans (the same widths) or DejaVu Sans where it is missing; superscripts in the same font
FONT = {"font.family": "sans-serif", "font.sans-serif": ["Arial", "Liberation Sans", "DejaVu Sans"],
        "mathtext.fontset": "custom", "mathtext.rm": "sans", "mathtext.bf": "sans:bold", "mathtext.it": "sans:italic",
        "mathtext.cal": "sans", "mathtext.sf": "sans"}


# ---------------------------------------------------------------------------
# Values
# ---------------------------------------------------------------------------

def find_feature_xlsx(given: Optional[Path]) -> Path:
    if given is not None:
        path = Path(given)
        if not path.exists():
            raise FileNotFoundError(f"--feature-xlsx not found: {path}")
        return path
    here = Path(__file__).resolve().parent if "__file__" in globals() else Path.cwd()
    analysis = WORKSPACE / B6_NAME / "Feature_Analysis"
    candidates = [analysis / folder / FEATURE_FILE for folder in FEATURE_FOLDERS]
    candidates += [analysis / FEATURE_FILE, here / FEATURE_FILE, Path.cwd() / FEATURE_FILE]
    for path in candidates:
        if path.exists():
            return path
    raise FileNotFoundError("261110 feature analysis not found; looked for:\n  "
                            + "\n  ".join(str(p) for p in candidates) + "\nGive it with --feature-xlsx")


def same_subject(a: object, b: object) -> bool:
    a, b = str(a).strip(), str(b).strip()
    if a == b:
        return True
    return a.isdigit() and b.isdigit() and int(a) == int(b)


def is_issue(dataset: str, subject: object) -> bool:
    return any(same_subject(subject, issue) for issue in ISSUE_SUBJECTS.get(dataset, ()))


def subject_table(path: Path, source: Dict[str, str], all_subjects: bool) -> pd.DataFrame:
    sheet = pd.read_excel(path, sheet_name="Subjects", dtype={"Subject": str})
    columns = [f"{stage}_{VALUE}" for stage in STAGES]
    missing = [c for c in ["Version", "Encoder", "Dataset", "Subject"] + columns if c not in sheet.columns]
    if missing:
        raise KeyError(f"{path.name}, Subjects: missing columns {missing}")
    parts = []
    for dataset in PANELS:
        version = source[dataset]
        part = sheet[(sheet["Version"].astype(str) == version) & (sheet["Encoder"] == ENCODER)
                     & (sheet["Dataset"] == dataset)].copy()
        if part.empty:
            raise ValueError(f"{path.name}: no six-block subjects of {dataset} in version {version}")
        part["MAE_Issue_Subject"] = [is_issue(dataset, s) for s in part["Subject"]]
        if not all_subjects:
            part = part[~part["MAE_Issue_Subject"]]
        keep = ["Version", "Encoder", "Dataset", "Subject", "N_Models", "MAE_Issue_Subject"]
        parts.append(part[[c for c in keep if c in part.columns] + columns])
    return pd.concat(parts, ignore_index=True)


def summary_table(subjects: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for dataset in PANELS:
        frame = subjects[subjects["Dataset"] == dataset]
        stem = float(frame[f"Stem_{VALUE}"].astype(float).mean())
        for stage in STAGES:
            values = frame[f"{stage}_{VALUE}"].astype(float).dropna()
            mean = float(values.mean())
            rows.append({"Dataset": dataset, "Panel": TITLES[dataset], "Version": str(frame["Version"].iloc[0]),
                         "Stage": stage, "N_Subjects": int(values.size), "Mean_PCC": mean,
                         "SD_PCC": float(values.std(ddof=1)) if values.size > 1 else float("nan"),
                         "Change_from_Stem_Pct": 100.0 * (mean - stem) / stem if stem > 0 else float("nan")})
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Figure
# ---------------------------------------------------------------------------

def change_text(percent: float, decimals: int) -> str:
    text = f"{percent:+.{decimals}f}"
    if float(text) == 0.0:
        text = f"{0.0:.{decimals}f}"
    return text + "%"


def label_place(values: np.ndarray, i: int) -> Tuple[Tuple[float, float], str, str]:
    """Offset (points), horizontal and vertical alignment of the value at dot i, on the side away from the line:
    a peak or the first dot above, a dip under, a dot on the way up above-left, on the way down above-right, the
    last dot above (under it if the line comes down to it)."""
    previous = values[i - 1] if i > 0 else np.nan
    following = values[i + 1] if i + 1 < len(values) else np.nan
    value = values[i]
    above, below = ((0, 9), "center", "bottom"), ((0, -11), "center", "top")
    if not np.isfinite(previous):
        return above
    if not np.isfinite(following):
        return above if value >= previous else below
    if value >= previous and value >= following:
        return above
    if value < previous and value < following:
        return below
    return ((-7, 7), "right", "bottom") if previous < value else ((7, 7), "left", "bottom")


def draw_figure(base: Path, summary: pd.DataFrame, decimals: int) -> None:
    x = np.arange(len(STAGES))
    with plt.rc_context(FONT):
        fig, axes = plt.subplots(1, len(PANELS), figsize=(22.0, 5.9))
        for ax, dataset in zip(axes, PANELS):
            part = summary[summary["Dataset"] == dataset].set_index("Stage").reindex(list(STAGES))
            values = part["Mean_PCC"].to_numpy(float)
            change = part["Change_from_Stem_Pct"].to_numpy(float)
            ax.plot(x, values, "o-", color=COLOR, lw=2.2, ms=9, zorder=3)
            for i, pct in enumerate(change):
                offset, ha, va = label_place(values, i)
                ax.annotate(change_text(pct, decimals), (i, values[i]), textcoords="offset points", xytext=offset,
                            ha=ha, va=va, fontsize=15, color=COLOR, fontweight="bold", zorder=5,
                            bbox={"boxstyle": "round,pad=0.1", "facecolor": "white", "edgecolor": "none",
                                  "alpha": 0.9})
            finite = values[np.isfinite(values)]
            low, high = float(finite.min()), float(finite.max())
            span = max(high - low, 0.01)
            ax.set_ylim(low - 0.18 * span, high + 0.17 * span)
            ax.yaxis.set_major_locator(MaxNLocator(nbins=8, steps=[1, 2, 2.5, 5, 10]))
            ax.set_xticks(x)
            ax.set_xticklabels(STAGE_LABELS, fontsize=15, fontweight="bold", color="black")
            ax.set_xlim(-0.45, len(STAGES) - 0.55)
            ax.tick_params(axis="y", labelsize=13, colors="black")
            ax.set_title(TITLES[dataset], fontsize=19, fontweight="bold", color="black", pad=12)
            ax.grid(True, color=GRID, lw=1.0)
            ax.set_axisbelow(True)
            for spine in ax.spines.values():
                spine.set_linewidth(1.2)
        axes[0].set_ylabel(Y_LABEL, fontsize=18, fontweight="bold", color="black", labelpad=10)
        fig.tight_layout(w_pad=2.0)
        for ext in ("png", "pdf"):
            fig.savefig(base.with_name(f"{base.name}.{ext}"), dpi=300)
        plt.close(fig)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Six-block encoder: channel-averaged absolute PCC of Fusion Output and the "
                                            "1st-6th Block with the change from Fusion Output (%), three datasets")
    p.add_argument("--feature-xlsx", type=Path, default=None, help=f"{FEATURE_FILE} (found by default)")
    p.add_argument("--output-root", type=Path, default=OUTPUT_ROOT)
    p.add_argument("--source", choices=tuple(SOURCES), default="final",
                   help="final: STEAM2/BIDMC 261013, CapnoBase 261007 (default); or every dataset from one run")
    p.add_argument("--all-subjects", action="store_true", help="keep the MAE issue subjects")
    p.add_argument("--decimals", type=int, default=0, help="decimals of the % labels (default 0)")
    return p.parse_args(list(sys.argv[1:] if argv is None else argv))


def main(argv: Optional[Sequence[str]] = None) -> None:
    args = parse_args(argv)
    source = SOURCES[args.source]
    path = find_feature_xlsx(args.feature_xlsx)
    out_root = Path(args.output_root)
    out_root.mkdir(parents=True, exist_ok=True)
    subjects = subject_table(path, source, args.all_subjects)
    summary = summary_table(subjects)
    draw_figure(out_root / FIGURE_NAME, summary, args.decimals)

    subject_set = "all subjects" if args.all_subjects else "MAE issue subjects left out"
    notes = pd.DataFrame([
        ("Input", str(path)),
        ("Models", ", ".join(f"{TITLES[d]} ({d}) {source[d]} six blocks" for d in PANELS)),
        ("Subjects", subject_set + ("" if args.all_subjects else ": " + ", ".join(
            f"{d} {' '.join(ISSUE_SUBJECTS[d])}" for d in PANELS))),
        ("PCC", "per subject: |Pearson r| of every channel with the target per 10 s window, mean over the windows, "
                "over the 24 channels (no PCA) and over the subject's models (261110 feature analysis)"),
        ("Stages", "Fusion Output = Stem (stem fusion output); 1st-6th Block = B1-B6 (dilation 1, 2, 4, 8, 16, 32)"),
        ("Change", "100 x (mean PCC of the stage - mean PCC of Stem) / mean PCC of Stem, means over the subjects"),
    ], columns=["Item", "Description"])
    with pd.ExcelWriter(out_root / f"{NAME}.xlsx") as writer:
        summary.round({"Mean_PCC": 4, "SD_PCC": 4, "Change_from_Stem_Pct": 2}).to_excel(writer, sheet_name="Summary", index=False)
        subjects.to_excel(writer, sheet_name="Subjects", index=False)
        notes.to_excel(writer, sheet_name="Notes", index=False)

    print(f"Input: {path}")
    print(f"Subjects: {subject_set}")
    for dataset in PANELS:
        part = summary[summary["Dataset"] == dataset]
        print(f"  {TITLES[dataset]:<10} ({source[dataset]}, n={int(part['N_Subjects'].iloc[0])}) "
              + ", ".join(f"{r.Stage} {r.Mean_PCC:.3f} ({change_text(r.Change_from_Stem_Pct, 0)})"
                          for r in part.itertuples()))
    print(f"Figure and table under {out_root}")


if __name__ == "__main__":
    main()
