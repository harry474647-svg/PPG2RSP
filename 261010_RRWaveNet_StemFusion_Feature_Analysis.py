r"""261010_RRWaveNet_StemFusion_Feature_Analysis: feature-level effect of the
1x1 stem fusion in the results of 261007_RRWaveNet_Ensemble_NoEvent.

Stem Fusion O is the full model of 261007 (three stem branches -> 1x1 conv +
GroupNorm -> encoder). Stem Fusion X is its no_stem_fusion ablation (the 24
concatenated branch channels feed the encoder). Both are trained in the same
LOSOCV folds with the same seeds. For each model, the feature map that
enters the encoder is analysed on the held-out subject:
* O: the 24 channels after the 1x1 conv + GroupNorm;
* X: the 24 concatenated branch channels (k32 / k64 / k128, 8 each).

Metrics (per test window, on its 24 x T feature map)
----------------------------------------------------
Effective rank (Roy & Vetterli, 2007), entropy-based:
  centre every channel over time, take the eigenvalues of the 24 x 24
  channel covariance (the energy of each principal direction), normalize
  them to a distribution p, and compute exp(-sum p log p). It is the
  number of directions that carry the energy evenly (1 to 24); it is also
  reported divided by 24.
Off-diagonal correlation:
  Pearson correlation of every pair of channels over time in the window.
  The mean over the 24 x 23 off-diagonal entries is reported as the mean
  |r| (how strongly channels repeat each other) and as the signed mean.
  A channel that is constant within a window has no correlation; its pairs
  are left out, and the share left out is reported.
The window values are averaged over all test windows of the held-out
subject, then over the 3 ensemble members (models), giving one value per
subject and model type.

Statistics and figures
----------------------
O and X are paired by subject (same held-out subject, same seeds).
* Per dataset (CapnoBase, BIDMC, STEAM2) and over all datasets together
  (every subject of every dataset once): mean +- SD, median [IQR], and the
  number of subjects with a higher value in O.
* Test: Wilcoxon signed-rank test, two-sided (**** p < 1e-4, *** < 1e-3,
  ** < 1e-2, * < 0.05, ns). The O row of the summary table is starred when
  p < 0.05.
* Box plots: Stem Fusion X (left) vs Stem Fusion O (right) with the
  significance bracket, for every dataset and for all datasets, plus one
  overview figure per metric.
Each window condition of 261007 (10 s, 20 s) is analysed separately.

Outputs (default: <results root>\Summary\Stem_Fusion_OX)
--------------------------------------------------------
  stem_fusion_ox_members.csv    one row per model type, window, dataset,
                                subject and member (window-averaged metrics;
                                the Pooled_* columns are the same metrics on
                                all windows pooled, as in the training-time
                                diagnostics of 261007)
  stem_fusion_ox_subjects.csv   one row per subject: O and X side by side
  stem_fusion_ox_tests.csv      statistics per window, dataset / all, metric
  stem_fusion_ox_table_<metric>.csv   mean +- SD table as in the slides
  Figures\   box plots
  stem_fusion_ox.xlsx           all tables

Run (next to 261007_RRWaveNet_Ensemble_NoEvent.py, after its LOSOCV):
  python 261010_RRWaveNet_StemFusion_Feature_Analysis.py
  --results-root <261007 results>   (default: PROJECT_ROOT of the 261007 file)
  --datasets capnobase bidmc        --windows w10s_nonoverlap
  --backbone <path to 261007_RRWaveNet_Ensemble_NoEvent.py>
Data paths and preprocessing are those of the 261007 file (DATASET_CONFIGS),
so the held-out windows are exactly the ones the models were tested on.

Reference: Roy O, Vetterli M. The effective rank: a measure of effective
dimensionality. EUSIPCO 2007.
"""

import argparse
import importlib.util
import sys
from pathlib import Path
from typing import Dict, List, Optional, Sequence

sys.dont_write_bytecode = True

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from scipy import stats

BACKBONE_NAME = "261007_RRWaveNet_Ensemble_NoEvent.py"
# Box order and colours: X (left, red), O (right, blue).
MODEL_TYPES = (("Stem Fusion X", "#e8968c"), ("Stem Fusion O", "#5b8db8"))
METRICS = (
    # (column, axis label, drawn)
    ("EffectiveRank", "Effective rank", True),
    ("EffectiveRank_Normalized", "Effective rank / channels", False),
    ("OffDiag_AbsMean", "Off-diagonal correlation (mean |r|)", True),
    ("OffDiag_SignedMean", "Off-diagonal correlation (signed mean)", True),
)
ALL_DATASETS = "All datasets"
CONSTANT_CHANNEL_SD = 1e-6


def load_backbone(path: Path):
    """Import the 261007 script as a module (no training is started)."""
    spec = importlib.util.spec_from_file_location("rrwavenet_261007", str(path))
    module = importlib.util.module_from_spec(spec)
    sys.modules["rrwavenet_261007"] = module
    spec.loader.exec_module(module)
    return module


def window_metrics(features: np.ndarray) -> Dict[str, np.ndarray]:
    """Effective rank and off-diagonal correlation of every window.

    features: (windows, channels, samples). Returns per-window arrays.
    """
    x = np.asarray(features, dtype=np.float64)
    windows, channels, samples = x.shape
    x = x - x.mean(axis=2, keepdims=True)
    covariance = np.einsum("wct,wdt->wcd", x, x) / max(1, samples - 1)
    covariance = (covariance + np.transpose(covariance, (0, 2, 1))) / 2.0
    eigenvalues = np.clip(np.linalg.eigvalsh(covariance), 0.0, None)
    total = eigenvalues.sum(axis=1, keepdims=True)
    with np.errstate(divide="ignore", invalid="ignore"):
        p = np.where(total > 1e-12, eigenvalues / total, 0.0)
        entropy = -np.sum(np.where(p > 0, p * np.log(np.where(p > 0, p, 1.0)), 0.0), axis=1)
    effective_rank = np.where(total[:, 0] > 1e-12, np.exp(entropy), np.nan)
    sd = np.sqrt(np.clip(np.einsum("wcc->wc", covariance), 0.0, None))
    active = sd > CONSTANT_CHANNEL_SD
    pairs = active[:, :, None] & active[:, None, :] & ~np.eye(channels, dtype=bool)[None]
    with np.errstate(divide="ignore", invalid="ignore"):
        correlation = covariance / (sd[:, :, None] * sd[:, None, :])
    correlation = np.where(pairs, correlation, 0.0)
    n_pairs = pairs.sum(axis=(1, 2))
    with np.errstate(divide="ignore", invalid="ignore"):
        abs_mean = np.where(n_pairs > 0, np.abs(correlation).sum(axis=(1, 2)) / n_pairs, np.nan)
        signed_mean = np.where(n_pairs > 0, correlation.sum(axis=(1, 2)) / n_pairs, np.nan)
    return {
        "EffectiveRank": effective_rank,
        "EffectiveRank_Normalized": effective_rank / channels,
        "OffDiag_AbsMean": abs_mean,
        "OffDiag_SignedMean": signed_mean,
        "Undefined_Pair_Share": 1.0 - n_pairs / float(channels * (channels - 1)),
    }


def encoder_input_features(model, loader, device) -> np.ndarray:
    """The feature map that enters the encoder, for every test window."""
    model.eval()
    chunks = []
    with torch.no_grad():
        for x, _, _, _ in loader:
            chunks.append(model.fused_representation(x.to(device)).cpu().numpy().astype(np.float32))
    return np.concatenate(chunks)


def significance_stars(p_value: float) -> str:
    if not np.isfinite(p_value):
        return "n/a"
    for threshold, stars in ((1e-4, "****"), (1e-3, "***"), (1e-2, "**"), (0.05, "*")):
        if p_value < threshold:
            return stars
    return "ns"


def wilcoxon_p(a: np.ndarray, b: np.ndarray) -> float:
    if len(a) < 2 or np.allclose(a, b):
        return np.nan if len(a) < 2 else 1.0
    try:
        return float(stats.wilcoxon(a, b).pvalue)
    except ValueError:
        return np.nan


def draw_box(axis, values_x: np.ndarray, values_o: np.ndarray, label: str, p_value: float) -> None:
    """Stem Fusion X vs O box plot with the significance bracket."""
    boxes = axis.boxplot(
        [values_x, values_o],
        widths=0.55,
        patch_artist=True,
        showfliers=False,
        medianprops={"color": "#ff7f0e", "linewidth": 1.5},
        whiskerprops={"color": "#222222"},
        capprops={"color": "#222222"},
    )
    for patch, (_, color) in zip(boxes["boxes"], MODEL_TYPES):
        patch.set_facecolor(color)
        patch.set_edgecolor("#333333")
    axis.set_xticks([1, 2])
    axis.set_xticklabels([name for name, _ in MODEL_TYPES], fontweight="bold")
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


def analyse(args: argparse.Namespace) -> None:
    module = load_backbone(Path(args.backbone).resolve())
    configs = module.DATASET_CONFIGS
    if args.capnobase_path:
        configs["capnobase"]["data_path"] = str(args.capnobase_path)
    if args.bidmc_path:
        configs["bidmc"]["data_path"] = str(args.bidmc_path)
    if args.steam2_ppg_dir:
        configs["steam2"]["ppg_dir"] = str(args.steam2_ppg_dir)
    if args.steam2_rsp_dir:
        configs["steam2"]["rsp_dir"] = str(args.steam2_rsp_dir)
    if args.threads:
        torch.set_num_threads(int(args.threads))
    results_root = Path(args.results_root or module.DEFAULT_RESULTS_ROOT).resolve()
    out_root = Path(args.output).resolve() if args.output else results_root / "Summary" / "Stem_Fusion_OX"
    figure_root = out_root / "Figures"
    figure_root.mkdir(parents=True, exist_ok=True)
    device = torch.device(
        f"cuda:{args.gpu}" if torch.cuda.is_available() and not args.force_cpu else "cpu"
    )
    model_types = [
        ("Stem Fusion X", args.without_fusion),
        ("Stem Fusion O", args.with_fusion),
    ]
    for _, variant in model_types:
        if variant not in module.VARIANTS:
            raise ValueError(f"{variant} is not a variant of {BACKBONE_NAME}")
    print(f"Results: {results_root}\nDevice: {device}")

    rows, skipped = [], []
    window_specs = [spec for spec in module.WINDOW_SPECS if spec.name in args.windows]
    dataset_keys = [key for key in module.DATASET_ORDER if key in args.datasets]
    for spec in window_specs:
        for dataset_key in dataset_keys:
            label = configs[dataset_key]["label"]
            roots = {
                variant: module.condition_root(results_root, variant, spec, dataset_key)
                for _, variant in model_types
            }
            if not all((root / "Members").exists() for root in roots.values()):
                print(f"{label} {spec.name}: no results of both model types, skipped")
                continue
            try:
                subject_data = module.load_dataset(dataset_key, spec)
            except Exception as exc:
                print(f"{label}: could not be loaded ({exc})")
                continue
            subjects = sorted(subject_data)
            if args.subjects:
                subjects = [s for s in subjects if s in set(args.subjects)]
            print(f"{label} {spec.name}: {len(subjects)} subjects")
            for name, variant in model_types:
                root = roots[variant]
                model = module.VARIANTS[variant].options.build().to(device)
                for subject in subjects:
                    tag = module.safe_tag(subject)
                    checkpoints = [
                        root / "Members" / f"m{member}" / "Models" / f"{tag}_best.pth"
                        for member in range(args.members)
                    ]
                    complete = all(
                        module.member_paths(root, subject, member)[0].exists() and checkpoints[member].exists()
                        for member in range(args.members)
                    )
                    if not complete:
                        skipped.append({"Model_Type": name, "Window_Config": spec.name, "Dataset": label,
                                        "Subject": subject, "Reason": "not all members stored"})
                        continue
                    loader = module.make_eval_loader(
                        module.SubjectWindowDataset(subject_data, [subject], "test", False),
                        args.batch_size,
                        device,
                    )
                    for member, checkpoint in enumerate(checkpoints):
                        model.load_state_dict(torch.load(checkpoint, map_location=device))
                        features = encoder_input_features(model, loader, device)
                        values = window_metrics(features)
                        row = {
                            "Model_Type": name,
                            "Variant": variant,
                            "Window_Config": spec.name,
                            "Dataset": label,
                            "Subject": subject,
                            "Member": member,
                            "N_Windows": int(features.shape[0]),
                            "Channels": int(features.shape[1]),
                        }
                        for key, array in values.items():
                            row[key] = float(np.nanmean(array)) if np.isfinite(array).any() else np.nan
                        pooled = module.feature_representation_summary(features)
                        row["Pooled_EffectiveRank"] = pooled["Feature_EffectiveRank"]
                        row["Pooled_OffDiag_AbsMean"] = pooled["Feature_OffDiag_AbsMean"]
                        row["Pooled_OffDiag_SignedMean"] = pooled["Feature_OffDiag_SignedMean"]
                        rows.append(row)
                    del loader
                del model
            del subject_data
            module.release_memory(device)

    members = pd.DataFrame(rows)
    skipped_frame = pd.DataFrame(skipped, columns=["Model_Type", "Window_Config", "Dataset", "Subject", "Reason"])
    if members.empty:
        skipped_frame.to_csv(out_root / "stem_fusion_ox_skipped.csv", index=False, encoding="utf-8-sig")
        print("No subject with stored members of both model types.")
        return
    metric_columns = [metric for metric, _, _ in METRICS] + ["Undefined_Pair_Share"]
    keys = ["Window_Config", "Dataset", "Subject"]
    per_type = members.groupby(["Model_Type"] + keys, sort=False)[metric_columns].mean().reset_index()
    wide = per_type.pivot_table(index=keys, columns="Model_Type", values=metric_columns, sort=False)
    wide.columns = [f"{metric}_{name.replace('Stem Fusion ', '')}" for metric, name in wide.columns]
    # Subjects analysed with both model types.
    subjects_table = wide.dropna(subset=["EffectiveRank_O", "EffectiveRank_X"]).reset_index()

    test_rows = []
    datasets_in_order = [configs[key]["label"] for key in dataset_keys]
    for window in [spec.name for spec in window_specs]:
        frame_window = subjects_table[subjects_table["Window_Config"] == window]
        if frame_window.empty:
            continue
        groups = [(d, frame_window[frame_window["Dataset"] == d]) for d in datasets_in_order]
        groups = [(d, f) for d, f in groups if not f.empty]
        groups.append((ALL_DATASETS, frame_window))
        for metric, axis_label, drawn in METRICS:
            overview = []
            for dataset, frame in groups:
                o = frame[f"{metric}_O"].to_numpy(float)
                x = frame[f"{metric}_X"].to_numpy(float)
                keep = np.isfinite(o) & np.isfinite(x)
                o, x = o[keep], x[keep]
                p_value = wilcoxon_p(o, x)
                test_rows.append({
                    "Window_Config": window, "Dataset": dataset, "Metric": metric,
                    "N_Subjects": int(len(o)),
                    "O_Mean": float(np.mean(o)), "O_SD": float(np.std(o, ddof=1)) if len(o) > 1 else np.nan,
                    "X_Mean": float(np.mean(x)), "X_SD": float(np.std(x, ddof=1)) if len(x) > 1 else np.nan,
                    "O_Median": float(np.median(o)), "O_Q1": float(np.percentile(o, 25)), "O_Q3": float(np.percentile(o, 75)),
                    "X_Median": float(np.median(x)), "X_Q1": float(np.percentile(x, 25)), "X_Q3": float(np.percentile(x, 75)),
                    "Diff_O_minus_X_Mean": float(np.mean(o - x)),
                    "N_Subjects_O_Higher": int(np.sum(o > x)), "N_Subjects_O_Lower": int(np.sum(o < x)),
                    "Wilcoxon_p": p_value, "Significance": significance_stars(p_value),
                })
                overview.append((dataset, x, o, p_value))
                if drawn and len(o) >= 2:
                    fig, axis = plt.subplots(figsize=(6.2, 4.0))
                    draw_box(axis, x, o, axis_label, p_value)
                    axis.set_title(f"{dataset}, {window}: n = {len(o)} subjects "
                                   f"(mean of {args.members} models), Wilcoxon p = {p_value:.2g}", fontsize=9)
                    fig.tight_layout()
                    fig.savefig(figure_root / f"{window}_{dataset.replace(' ', '_')}_{metric}.png", dpi=150)
                    plt.close(fig)
            if drawn and overview:
                fig, axes = plt.subplots(1, len(overview), figsize=(4.2 * len(overview), 4.0), squeeze=False)
                for axis, (dataset, x, o, p_value) in zip(axes[0], overview):
                    if len(o) >= 2:
                        draw_box(axis, x, o, axis_label, p_value)
                    axis.set_title(f"{dataset} (n = {len(o)})", fontsize=10)
                fig.suptitle(f"{window}: {axis_label}", fontsize=11)
                fig.tight_layout()
                fig.savefig(figure_root / f"{window}_overview_{metric}.png", dpi=150)
                plt.close(fig)
    tests = pd.DataFrame(test_rows)
    tables = {}
    for metric, _, _ in METRICS:
        table_rows = []
        for (window, dataset), row in tests[tests["Metric"] == metric].set_index(["Window_Config", "Dataset"]).iterrows():
            star = "*" if np.isfinite(row["Wilcoxon_p"]) and row["Wilcoxon_p"] < 0.05 else ""
            table_rows.append({"Window_Config": window, "Model_Type": "Stem Fusion O", "Dataset": dataset,
                               metric: f"{row['O_Mean']:.3f} ± {row['O_SD']:.3f}{star}",
                               "Wilcoxon_p": row["Wilcoxon_p"], "N_Subjects": row["N_Subjects"]})
        for (window, dataset), row in tests[tests["Metric"] == metric].set_index(["Window_Config", "Dataset"]).iterrows():
            table_rows.append({"Window_Config": window, "Model_Type": "Stem Fusion X", "Dataset": dataset,
                               metric: f"{row['X_Mean']:.3f} ± {row['X_SD']:.3f}",
                               "Wilcoxon_p": row["Wilcoxon_p"], "N_Subjects": row["N_Subjects"]})
        tables[metric] = pd.DataFrame(table_rows).sort_values(["Window_Config", "Model_Type"], ascending=[True, True], kind="stable")
        tables[metric].to_csv(out_root / f"stem_fusion_ox_table_{metric}.csv", index=False, encoding="utf-8-sig")
    members.to_csv(out_root / "stem_fusion_ox_members.csv", index=False, encoding="utf-8-sig")
    subjects_table.to_csv(out_root / "stem_fusion_ox_subjects.csv", index=False, encoding="utf-8-sig")
    tests.to_csv(out_root / "stem_fusion_ox_tests.csv", index=False, encoding="utf-8-sig")
    skipped_frame.to_csv(out_root / "stem_fusion_ox_skipped.csv", index=False, encoding="utf-8-sig")
    sheets = {"Tests": tests, "Subjects": subjects_table, "Members": members, "Skipped": skipped_frame}
    sheets.update({f"Table_{metric}"[:31]: frame for metric, frame in tables.items()})
    module.write_results_excel(out_root / "stem_fusion_ox.xlsx", sheets)

    print("\nStem Fusion O vs X (per subject, mean of the models; mean +- SD; Wilcoxon signed-rank):")
    for _, row in tests.iterrows():
        print(f"  {row['Window_Config'][1:4]:<4s} {row['Dataset']:<13s} {row['Metric']:<25s} "
              f"O {row['O_Mean']:.3f} +- {row['O_SD']:.3f}  X {row['X_Mean']:.3f} +- {row['X_SD']:.3f}  "
              f"O higher {row['N_Subjects_O_Higher']}/{row['N_Subjects']}  p={row['Wilcoxon_p']:.2g} {row['Significance']}")
    print(f"Saved under: {out_root}")


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--backbone", type=Path, default=Path(__file__).with_name(BACKBONE_NAME),
                        help=f"Path of {BACKBONE_NAME} (data paths and model are taken from it).")
    parser.add_argument("--results-root", type=Path, default=None,
                        help="Results root of the 261007 run (default: its PROJECT_ROOT).")
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--datasets", nargs="+", default=["capnobase", "bidmc", "steam2"],
                        choices=["capnobase", "bidmc", "steam2"])
    parser.add_argument("--windows", nargs="+", default=["w10s_nonoverlap", "w20s_nonoverlap"])
    parser.add_argument("--with-fusion", default="full", help="261007 variant with the stem fusion (O).")
    parser.add_argument("--without-fusion", default="no_stem_fusion",
                        help="261007 variant without the stem fusion (X).")
    parser.add_argument("--members", type=int, default=3, help="Models per fold (default 3).")
    parser.add_argument("--subjects", nargs="+", default=None, help="Only these subjects (quick tests).")
    parser.add_argument("--batch-size", type=int, default=48)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--force-cpu", action="store_true")
    parser.add_argument("--threads", type=int, default=None)
    parser.add_argument("--capnobase-path", type=Path, default=None)
    parser.add_argument("--bidmc-path", type=Path, default=None)
    parser.add_argument("--steam2-ppg-dir", type=Path, default=None)
    parser.add_argument("--steam2-rsp-dir", type=Path, default=None)
    return parser.parse_args(argv)


if __name__ == "__main__":
    analyse(parse_args())
