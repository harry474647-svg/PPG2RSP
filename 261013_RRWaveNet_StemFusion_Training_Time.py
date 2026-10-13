r"""261013_RRWaveNet_StemFusion_Training_Time: training time of Stem Fusion O
(full model) vs Stem Fusion X (no_stem_fusion) in the results of
261007_RRWaveNet_Ensemble_NoEvent (depthwise residual encoder in both).

Nothing is trained or loaded except the stored per-model summaries
(Fold_Summaries\<subject>_m<k>_summary.json, written by 261007 when a model
finishes). Per model (LOSOCV fold x ensemble member):
* Elapsed_Min: wall time of the model (training, then scoring of the
  held-out subject), Elapsed_Sec / 60;
* Epochs_Run: epochs trained before early stopping (History_Epochs);
* Sec_per_Epoch: Elapsed_Sec / Epochs_Run, the time of one epoch (does not
  depend on when early stopping ended the run).
O and X are paired by held-out subject and member (same seed, same
training windows). The values of a subject are averaged over its paired
members, then O and X are compared over subjects with the Wilcoxon
signed-rank test (two-sided), for each dataset and for all datasets
together: mean +- SD, median [IQR], O / X ratio (median over subjects),
number of subjects where O took longer, p (**** < 1e-4, *** < 1e-3,
** < 1e-2, * < 0.05, ns).
Wall time also depends on what else ran on the machine at the same time
(other workers on the same GPU, other programs); the run_manifest files of
the results root say how the run was split.

Outputs: <results root>\Summary_<window>\Training_Time_StemFusion\
  training_time_members.csv    every model of both types
  training_time_subjects.csv   per subject: O and X side by side
  training_time_tests.csv      per dataset / all datasets and metric
  training_time.xlsx           all tables
  Figures\<window>_<metric>.png   box plots X (red) vs O (blue) per dataset

Run next to 261007_RRWaveNet_Ensemble_NoEvent.py (Spyder: just run the file):
  python 261013_RRWaveNet_StemFusion_Training_Time.py
Options: --results-root <261007 results> (default: its PROJECT_ROOT),
  --windows w10s_nonoverlap (default) w20s_nonoverlap, --members 3.
"""

import argparse
import importlib.util
import json
import re
import sys
from pathlib import Path
from typing import List, Optional, Sequence

sys.dont_write_bytecode = True

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

SCRIPT_DIR = Path(__file__).resolve().parent
BACKBONE = SCRIPT_DIR / "261007_RRWaveNet_Ensemble_NoEvent.py"
# (name, 261007 variant, colour); box order X (left, red), O (right, blue).
MODEL_TYPES = (
    ("Stem Fusion X", "no_stem_fusion", "#e8968c"),
    ("Stem Fusion O", "full", "#5b8db8"),
)
METRICS = (
    ("Sec_per_Epoch", "Training time per epoch (s)"),
    ("Elapsed_Min", "Time per model (min)"),
    ("Epochs_Run", "Epochs run (early stopping)"),
)
ALL_DATASETS = "All datasets"
SUMMARY_FILE = re.compile(r"^(?P<tag>.+)_m(?P<member>\d+)_summary\.json$")


def load_backbone(path: Path):
    """Import the 261007 script as a module (no training is started)."""
    spec = importlib.util.spec_from_file_location("rrwavenet_261007_time", str(path))
    module = importlib.util.module_from_spec(spec)
    sys.modules["rrwavenet_261007_time"] = module
    spec.loader.exec_module(module)
    return module


def sd(values) -> float:
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    return float(np.std(values, ddof=1)) if values.size > 1 else np.nan


def wilcoxon_p(a: np.ndarray, b: np.ndarray) -> float:
    if len(a) < 2:
        return np.nan
    if np.allclose(a, b):
        return 1.0
    try:
        return float(stats.wilcoxon(a, b).pvalue)
    except ValueError:
        return np.nan


def significance_stars(p_value: float) -> str:
    if not np.isfinite(p_value):
        return "n/a"
    for threshold, stars in ((1e-4, "****"), (1e-3, "***"), (1e-2, "**"), (0.05, "*")):
        if p_value < threshold:
            return stars
    return "ns"


def read_models(root: Path, model_type: str, variant: str, window: str, dataset: str,
                members: int) -> List[dict]:
    """Timing of every finished model stored under one condition root."""
    rows = []
    folder = root / "Fold_Summaries"
    if not folder.exists():
        return rows
    for path in sorted(folder.glob("*_m*_summary.json")):
        match = SUMMARY_FILE.match(path.name)
        if match is None or int(match.group("member")) >= members:
            continue
        try:
            summary = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        elapsed = float(summary.get("Elapsed_Sec", np.nan))
        epochs = float(summary.get("History_Epochs", np.nan))
        rows.append({
            "Model_Type": model_type,
            "Variant": variant,
            "Window_Config": window,
            "Dataset": dataset,
            "Subject": str(summary.get("Subject", match.group("tag"))),
            "Member": int(match.group("member")),
            "Seed": summary.get("Seed"),
            "Model": summary.get("Model"),
            "Max_Epochs": summary.get("Max_Epochs"),
            "Epochs_Run": epochs,
            "Best_Epoch": summary.get("Best_Epoch"),
            "Train_Windows": summary.get("Train_Windows"),
            "N_Params": summary.get("N_Params"),
            "Elapsed_Sec": elapsed,
            "Elapsed_Min": elapsed / 60.0,
            "Sec_per_Epoch": elapsed / epochs if epochs and np.isfinite(epochs) and epochs > 0 else np.nan,
        })
    return rows


def draw_box(axis, values_x: np.ndarray, values_o: np.ndarray, label: str, p_value: float) -> None:
    boxes = axis.boxplot([values_x, values_o], widths=0.55, patch_artist=True, showfliers=False,
                         medianprops={"color": "#ff7f0e", "linewidth": 1.5})
    for patch, (_, _, color) in zip(boxes["boxes"], MODEL_TYPES):
        patch.set_facecolor(color)
        patch.set_edgecolor("#333333")
    axis.set_xticks([1, 2])
    axis.set_xticklabels([name for name, _, _ in MODEL_TYPES], fontweight="bold")
    axis.set_ylabel(label)
    axis.yaxis.grid(True, color="#e6e6e6")
    axis.set_axisbelow(True)
    top = max(float(np.max(line.get_ydata())) for line in boxes["caps"])
    bottom = min(float(np.min(line.get_ydata())) for line in boxes["caps"])
    span = max(top - bottom, 1e-6)
    level, tick = top + 0.08 * span, 0.03 * span
    axis.plot([1, 1, 2, 2], [level - tick, level, level, level - tick], color="#111111", linewidth=1.5)
    axis.text(1.5, level + 0.01 * span, significance_stars(p_value), ha="center", va="bottom",
              fontsize=14, fontweight="bold")
    axis.set_ylim(bottom - 0.08 * span, level + 0.18 * span)


def main(args: argparse.Namespace) -> None:
    backbone = load_backbone(Path(args.backbone).resolve())
    results_root = Path(args.results_root or backbone.DEFAULT_RESULTS_ROOT).resolve()
    window_specs = [spec for spec in backbone.WINDOW_SPECS if spec.name in args.windows]
    dataset_keys = [key for key in backbone.DATASET_ORDER if key in args.datasets]
    print(f"Results: {results_root}")
    for spec in window_specs:
        tag = spec.name.split("_")[0].lstrip("w")  # e.g. "10s"
        out_root = results_root / f"Summary_{tag}" / "Training_Time_StemFusion"
        figure_root = out_root / "Figures"
        figure_root.mkdir(parents=True, exist_ok=True)
        rows = []
        datasets = []
        for key in dataset_keys:
            label = backbone.DATASET_CONFIGS[key]["label"]
            found = 0
            for name, variant, _ in MODEL_TYPES:
                root = backbone.condition_root(results_root, variant, spec, key)
                models = read_models(root, name, variant, spec.name, label, args.members)
                found += len(models)
                rows += models
                print(f"  {label} {spec.name} {name} ({variant}): {len(models)} models")
            if found:
                datasets.append(label)
        members = pd.DataFrame(rows)
        if members.empty:
            print(f"{spec.name}: no stored models")
            continue
        keys = ["Dataset", "Subject", "Member"]
        o = members[members["Model_Type"] == "Stem Fusion O"].set_index(keys)
        x = members[members["Model_Type"] == "Stem Fusion X"].set_index(keys)
        common = o.index.intersection(x.index)
        metric_columns = [metric for metric, _ in METRICS]
        paired = pd.concat(
            {"O": o.loc[common, metric_columns], "X": x.loc[common, metric_columns]}, axis=1
        )
        paired.columns = [f"{metric}_{side}" for side, metric in paired.columns]
        subjects = paired.groupby(level=["Dataset", "Subject"]).mean()
        subjects["N_Paired_Members"] = paired.groupby(level=["Dataset", "Subject"]).size()
        subjects = subjects.reset_index()

        test_rows = []
        groups = [(d, subjects[subjects["Dataset"] == d]) for d in datasets]
        groups = [(d, f) for d, f in groups if not f.empty] + [(ALL_DATASETS, subjects)]
        for metric, axis_label in METRICS:
            overview = []
            for dataset, frame in groups:
                vo = frame[f"{metric}_O"].to_numpy(float)
                vx = frame[f"{metric}_X"].to_numpy(float)
                keep = np.isfinite(vo) & np.isfinite(vx)
                vo, vx = vo[keep], vx[keep]
                p_value = wilcoxon_p(vo, vx)
                with np.errstate(divide="ignore", invalid="ignore"):
                    ratio = vo / vx
                test_rows.append({
                    "Window_Config": spec.name, "Dataset": dataset, "Metric": metric,
                    "N_Subjects": int(len(vo)),
                    "O_Mean": float(np.mean(vo)) if len(vo) else np.nan, "O_SD": sd(vo),
                    "X_Mean": float(np.mean(vx)) if len(vx) else np.nan, "X_SD": sd(vx),
                    "O_Median": float(np.median(vo)) if len(vo) else np.nan,
                    "O_Q1": float(np.percentile(vo, 25)) if len(vo) else np.nan,
                    "O_Q3": float(np.percentile(vo, 75)) if len(vo) else np.nan,
                    "X_Median": float(np.median(vx)) if len(vx) else np.nan,
                    "X_Q1": float(np.percentile(vx, 25)) if len(vx) else np.nan,
                    "X_Q3": float(np.percentile(vx, 75)) if len(vx) else np.nan,
                    "Diff_O_minus_X_Mean": float(np.mean(vo - vx)) if len(vo) else np.nan,
                    "Diff_O_minus_X_SD": sd(vo - vx),
                    "Ratio_O_over_X_Median": float(np.nanmedian(ratio)) if len(vo) else np.nan,
                    "N_Subjects_O_Longer": int(np.sum(vo > vx)),
                    "N_Subjects_O_Shorter": int(np.sum(vo < vx)),
                    "Wilcoxon_p": p_value, "Significance": significance_stars(p_value),
                })
                overview.append((dataset, vx, vo, p_value))
            fig, axes = plt.subplots(1, len(overview), figsize=(4.2 * len(overview), 4.0), squeeze=False)
            for axis, (dataset, vx, vo, p_value) in zip(axes[0], overview):
                if len(vo) >= 2:
                    draw_box(axis, vx, vo, axis_label, p_value)
                axis.set_title(f"{dataset} (n = {len(vo)})", fontsize=10)
            fig.suptitle(f"{spec.label}: {axis_label}, depthwise encoder", fontsize=11)
            fig.tight_layout()
            fig.savefig(figure_root / f"{spec.name}_{metric}.png", dpi=150)
            plt.close(fig)
        tests = pd.DataFrame(test_rows)
        members.to_csv(out_root / "training_time_members.csv", index=False, encoding="utf-8-sig")
        subjects.to_csv(out_root / "training_time_subjects.csv", index=False, encoding="utf-8-sig")
        tests.to_csv(out_root / "training_time_tests.csv", index=False, encoding="utf-8-sig")
        backbone.write_results_excel(out_root / "training_time.xlsx",
                                     {"Tests": tests, "Subjects": subjects, "Members": members})
        print(f"\n{spec.label}: Stem Fusion O vs X (per subject, mean of the paired members; "
              "mean ± SD; Wilcoxon signed-rank):")
        for _, r in tests.iterrows():
            print(f"  {r['Dataset']:<13s} {r['Metric']:<14s} O {r['O_Mean']:.2f} ± {r['O_SD']:.2f}  "
                  f"X {r['X_Mean']:.2f} ± {r['X_SD']:.2f}  O/X {r['Ratio_O_over_X_Median']:.3f}  "
                  f"O longer {r['N_Subjects_O_Longer']}/{r['N_Subjects']}  "
                  f"p={r['Wilcoxon_p']:.3g} {r['Significance']}")
        print(f"Saved under: {out_root}")


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--backbone", type=Path, default=BACKBONE)
    parser.add_argument("--results-root", type=Path, default=None,
                        help="Results root of the 261007 run (default: its PROJECT_ROOT).")
    parser.add_argument("--windows", nargs="+", default=["w10s_nonoverlap"])
    parser.add_argument("--datasets", nargs="+", default=["capnobase", "bidmc", "steam2"],
                        choices=["capnobase", "bidmc", "steam2"])
    parser.add_argument("--members", type=int, default=3, help="Models per fold (default 3).")
    return parser.parse_args(argv)


if __name__ == "__main__":
    main(parse_args())
