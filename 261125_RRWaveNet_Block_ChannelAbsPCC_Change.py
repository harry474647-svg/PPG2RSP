# -*- coding: utf-8 -*-
r"""261125_RRWaveNet_Block_ChannelAbsPCC_Change: the block comparison figure of the channel-averaged absolute PCC
(Stem, B1, B2, B3, B4; STEAM2, BIDMC, CapnoBase) with the change from Stem in % written at every point instead of
the PCC value.

Values: the channel-averaged absolute PCC of the earlier block comparison figures (261011 channel_r_summary: mean over
the subjects of (1/24) sum |r_c|), as printed there with 3 decimals (VALUES below; replace them to draw other values).
Change from Stem (%) = (stage - Stem) / Stem x 100. It is computed from the 3-decimal values, so it is written to the
whole percent by default (the rounding of the values can move it by about 1%; --decimals for more).
The line is drawn at the PCC values (y axis as before); only the labels are the change in %.

Outputs (D:\PPG2RSP_RRWaveNet_Inspired\261125_RRWaveNet_Block_ChannelAbsPCC_Change)
  Block_ChannelAbsPCC_Change_from_Stem.png / .pdf   the three datasets side by side
  Block_ChannelAbsPCC_Change_from_Stem.csv          PCC and change from Stem (%) of every dataset and stage

Run: python 261125_RRWaveNet_Block_ChannelAbsPCC_Change.py     (Spyder: runfile(...), no options needed)
Options: --output-root, --decimals
"""

import argparse
import sys
from pathlib import Path
from typing import Optional, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

NAME = "261125_RRWaveNet_Block_ChannelAbsPCC_Change"
OUTPUT_ROOT = Path(r"D:\PPG2RSP_RRWaveNet_Inspired") / NAME
FIGURE = "Block_ChannelAbsPCC_Change_from_Stem"
STAGES = ("Stem", "B1", "B2", "B3", "B4")
# Channel-averaged absolute PCC (mean over the subjects) of the earlier figures, Stem to B4.
VALUES = {
    "STEAM2": (0.134, 0.134, 0.185, 0.229, 0.290),
    "BIDMC": (0.179, 0.161, 0.205, 0.249, 0.306),
    "CapnoBase": (0.202, 0.210, 0.276, 0.310, 0.313),
}
COLOR = "#1f77b4"
GRID = "#e6e6e6"
MUTED = "#6b6a66"


def change_text(percent: float, decimals: int) -> str:
    text = f"{percent:+.{decimals}f}"
    if float(text) == 0.0:
        text = f"{0.0:.{decimals}f}"
    return text + "%"


def draw(path: Path, table: pd.DataFrame, decimals: int) -> None:
    datasets = list(VALUES)
    fig, axes = plt.subplots(1, len(datasets), figsize=(5.4 * len(datasets), 4.4))
    x = np.arange(len(STAGES))
    for ax, dataset in zip(np.atleast_1d(axes), datasets):
        part = table[table.Dataset == dataset].set_index("Stage").reindex(list(STAGES))
        values = part["Channel_AbsPCC"].to_numpy(float)
        change = part["Change_from_Stem_Pct"].to_numpy(float)
        ax.plot(x, values, "o-", color=COLOR, lw=1.6, ms=7)
        for xi, value, pct in zip(x, values, change):
            ax.annotate(change_text(pct, decimals), (xi, value), textcoords="offset points", xytext=(0, 8),
                        ha="center", va="bottom", fontsize=11, color=COLOR, fontweight="bold", zorder=5,
                        bbox={"boxstyle": "round,pad=0.12", "facecolor": "white", "edgecolor": "none",
                              "alpha": 0.9})
        low, high = float(np.min(values)), float(np.max(values))
        span = max(high - low, 0.02)
        ax.set_ylim(low - 0.3 * span, high + 0.22 * span)
        ax.set_xticks(x)
        ax.set_xticklabels(STAGES, fontsize=12, fontweight="bold")
        ax.set_xlim(-0.35, len(STAGES) - 0.65)
        ax.tick_params(axis="y", labelsize=10.5)
        ax.grid(True, color=GRID)
        ax.set_axisbelow(True)
        ax.set_title(dataset, fontsize=14, fontweight="bold", pad=10)
    np.atleast_1d(axes)[0].set_ylabel("Channel-Averaged PCC", fontsize=13, fontweight="bold")
    fig.text(0.5, 0.01, "Labels: change from Stem (%) = (stage - Stem) / Stem x 100", ha="center", va="bottom",
             fontsize=10, color=MUTED)
    fig.tight_layout(rect=(0, 0.05, 1, 1))
    for ext in ("png", "pdf"):
        fig.savefig(path.with_name(f"{path.name}.{ext}"), dpi=300)
    plt.close(fig)


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Block comparison of the channel-averaged absolute PCC with the change "
                                            "from Stem (%) at every point")
    p.add_argument("--output-root", type=Path, default=OUTPUT_ROOT)
    p.add_argument("--decimals", type=int, default=0, help="decimals of the % labels (default 0)")
    return p.parse_args(list(sys.argv[1:] if argv is None else argv))


def main(argv: Optional[Sequence[str]] = None) -> None:
    args = parse_args(argv)
    out_root = Path(args.output_root)
    out_root.mkdir(parents=True, exist_ok=True)
    rows = []
    for dataset, values in VALUES.items():
        stem = float(values[0])
        for stage, value in zip(STAGES, values):
            rows.append({"Dataset": dataset, "Stage": stage, "Channel_AbsPCC": float(value),
                         "Change_from_Stem_Pct": (float(value) - stem) / stem * 100.0})
    table = pd.DataFrame(rows)
    draw(out_root / FIGURE, table, args.decimals)
    table.round({"Channel_AbsPCC": 3, "Change_from_Stem_Pct": 2}).to_csv(out_root / f"{FIGURE}.csv", index=False, encoding="utf-8-sig")
    print("Change from Stem (%):")
    for dataset in VALUES:
        part = table[table.Dataset == dataset]
        print(f"  {dataset:<10} " + ", ".join(f"{r.Stage} {r.Change_from_Stem_Pct:+.1f}" for r in part.itertuples()))
    print(f"Figure and table under {out_root}")


if __name__ == "__main__":
    main()
