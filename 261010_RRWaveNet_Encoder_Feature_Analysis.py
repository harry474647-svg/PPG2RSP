r"""261010_RRWaveNet_Encoder_Feature_Analysis: feature-level analysis of the
four residual encoder blocks in the results of
261007_RRWaveNet_Ensemble_NoEvent.

Every stored model (default: the full model) is reloaded and run on the
test windows of its held-out subject. The 24-channel feature map is taken
at five stages:
  Stem = the stem fusion output (encoder input),
  B1 (d=1), B2 (d=2), B3 (d=4), B4 (d=8) = the output of each depthwise
  residual block (B4 = encoder output, "DeepEnc").
The blocks are depthwise, so channel c is the same channel at every stage.
The target is the respiratory waveform of each window as the model was
trained on it (min-max to [-1, 1]).

1. Channel correlation with the target (frequency analysis, left)
   For every window, channel and stage: Pearson r between the channel and
   the target over time. |r| is averaged over the windows -> |r| per
   channel. Reported per stage: the mean |r| over the 24 channels (PCC),
   its change against Stem (mean delta |r|), and the number of channels
   (of 24) whose |r| is higher than at Stem ("better"). Figures: heatmap of
   the mean delta |r| per channel and block, and the mean |channel r| per
   stage with delta, better and the p value.
2. PC1 power in the respiratory band (frequency analysis, right)
   PCA over the 24 channels: the channels are standardized over all
   samples of the subject, PC1 is the first principal component (the share
   of variance it explains is reported). PSD of PC1 in every window
   (periodogram with a Hann window over the whole window), averaged over
   the windows. Respiratory-band power ratio = power in 0.1-0.5 Hz /
   power above 0 Hz. The same ratio is reported for the target. Figure:
   mean PSD (0-1 Hz, each curve scaled to its maximum) with the ratios.
3. PC1 periodicity (periodicity analysis)
   Consecutive test windows of a recording segment are joined into runs
   (windows do not overlap). Normalized autocorrelation (ACF) of PC1 and of
   the target for lags 0-12 s in every run of at least 24 s, averaged over
   the runs (weighted by length). Similarity to the target = Pearson r
   between the PC1 ACF and the target ACF over lags 2-10 s (breathing at
   0.1-0.5 Hz). Figure: mean ACF of the target, DeepEnc (B4) and Stem Fuse
   with their r.
Every value is computed per model and averaged over the 3 models of a
subject. Statistics are over subjects: each block against Stem with the
paired Wilcoxon signed-rank test (two-sided), per dataset and over all
datasets together, separately for each window condition (10 s, 20 s).

Outputs (default: <results root>\Summary\Encoder_Feature_Analysis)
  encoder_channel_corr_tests.csv   per stage: PCC, mean delta |r|, better, p
  encoder_psd_tests.csv            per stage: resp-band ratio, p vs Stem
  encoder_acf_tests.csv            per stage: ACF similarity r, p vs Stem
  encoder_subjects.csv             per subject and stage: all metrics
  encoder_channels.csv             per subject, stage and channel: |r|
  encoder_members.csv              per model (member)
  encoder_curves.npz               mean PSD and ACF curves
  Figures\                         heatmaps, summaries, PSD and ACF figures
  encoder_feature_analysis.xlsx    all tables

Run (next to 261007_RRWaveNet_Ensemble_NoEvent.py, after its LOSOCV):
  python 261010_RRWaveNet_Encoder_Feature_Analysis.py
  --results-root <261007 results>   (default: PROJECT_ROOT of the 261007 file)
  --variant full                    (any 261007 variant)
  --datasets capnobase bidmc steam2 --windows w10s_nonoverlap w20s_nonoverlap
"""

import argparse
import importlib.util
import sys
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

sys.dont_write_bytecode = True

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
import numpy as np
import pandas as pd
import torch
from scipy import stats
from scipy.signal import periodogram

BACKBONE_NAME = "261007_RRWaveNet_Ensemble_NoEvent.py"
STAGES = ("Stem", "B1", "B2", "B3", "B4")
STAGE_COLORS = {
    "Target": "#111111",
    "Stem": "#4c78a8",
    "B1": "#f2a33a",
    "B2": "#59a14f",
    "B3": "#e15759",
    "B4": "#9467bd",
}
RESP_BAND_HZ = (0.1, 0.5)
ACF_MAX_LAG_SEC = 12.0
ACF_BAND_SEC = (2.0, 10.0)
ACF_MIN_RUN_SEC = 24.0
PSD_PLOT_MAX_HZ = 1.0
ALL_DATASETS = "All datasets"


def load_backbone(path: Path):
    """Import the 261007 script as a module (no training is started)."""
    spec = importlib.util.spec_from_file_location("rrwavenet_261007", str(path))
    module = importlib.util.module_from_spec(spec)
    sys.modules["rrwavenet_261007"] = module
    spec.loader.exec_module(module)
    return module


def stage_labels(module) -> List[str]:
    dilations = module.MODEL_CONFIG["encoder_dilations"]
    return ["Stem"] + [f"B{i + 1} (d={d})" for i, d in enumerate(dilations)]


def stage_features(model, loader, device) -> Tuple[List[np.ndarray], np.ndarray]:
    """Feature maps at Stem and after every encoder block, and the targets."""
    model.eval()
    stages: List[List[np.ndarray]] = [[] for _ in STAGES]
    targets = []
    with torch.no_grad():
        for x, y, _, _ in loader:
            features = model.fused_representation(x.to(device))
            stages[0].append(features.cpu().numpy().astype(np.float32))
            for index, block in enumerate(model.encoder, start=1):
                features = block(features)
                stages[index].append(features.cpu().numpy().astype(np.float32))
            targets.append(y.squeeze(1).numpy().astype(np.float32))
    return [np.concatenate(chunks) for chunks in stages], np.concatenate(targets)


def channel_abs_r(features: np.ndarray, target: np.ndarray) -> np.ndarray:
    """|Pearson r| of every channel with the target, averaged over windows."""
    f = features.astype(np.float64) - features.mean(axis=2, keepdims=True)
    t = target.astype(np.float64) - target.mean(axis=1, keepdims=True)
    numerator = np.einsum("nct,nt->nc", f, t)
    denominator = np.sqrt(np.einsum("nct,nct->nc", f, f) * np.einsum("nt,nt->n", t, t)[:, None])
    with np.errstate(divide="ignore", invalid="ignore"):
        r = np.where(denominator > 1e-12, numerator / denominator, np.nan)
    return np.nanmean(np.abs(r), axis=0)


def first_component(features: np.ndarray) -> Tuple[np.ndarray, float]:
    """PC1 of the standardized channels: (windows, samples), variance share."""
    windows, channels, samples = features.shape
    matrix = np.transpose(features, (1, 0, 2)).reshape(channels, -1).astype(np.float64)
    mean = matrix.mean(axis=1, keepdims=True)
    sd = matrix.std(axis=1, keepdims=True)
    keep = sd[:, 0] > 1e-8
    z = (matrix[keep] - mean[keep]) / sd[keep]
    covariance = z @ z.T / max(1, z.shape[1] - 1)
    eigenvalues, eigenvectors = np.linalg.eigh(covariance)
    weights = eigenvectors[:, -1]
    share = float(eigenvalues[-1] / max(eigenvalues.sum(), 1e-12))
    pc1 = (weights @ z).reshape(windows, samples)
    return pc1, share


def mean_psd(signals: np.ndarray, fs: float) -> Tuple[np.ndarray, np.ndarray]:
    """Mean periodogram (Hann window over each whole window)."""
    frequencies, power = periodogram(signals, fs=fs, window="hann", detrend="constant", axis=-1)
    return frequencies, power.mean(axis=0)


def band_ratio(frequencies: np.ndarray, power: np.ndarray) -> float:
    band = (frequencies >= RESP_BAND_HZ[0]) & (frequencies <= RESP_BAND_HZ[1])
    total = power[frequencies > 0].sum()
    return float(power[band].sum() / total) if total > 0 else np.nan


def contiguous_runs(metadata: Sequence[dict], samples: int) -> List[List[int]]:
    """Indices of consecutive windows (same segment, adjacent starts)."""
    runs: List[List[int]] = []
    for index, item in enumerate(metadata):
        if runs:
            previous = metadata[runs[-1][-1]]
            if item["segment_id"] == previous["segment_id"] and item["start"] == previous["start"] + samples:
                runs[-1].append(index)
                continue
        runs.append([index])
    return runs


def mean_acf(signals: np.ndarray, runs: List[List[int]], max_lag: int, min_length: int) -> np.ndarray:
    """Normalized ACF (lags 0..max_lag) averaged over the runs, weighted by length."""
    total = np.zeros(max_lag + 1)
    weight = 0.0
    for run in runs:
        x = signals[run].reshape(-1).astype(np.float64)
        if x.size < max(min_length, max_lag + 1):
            continue
        x = x - x.mean()
        energy = float(np.dot(x, x))
        if energy <= 1e-12:
            continue
        spectrum = np.fft.rfft(x, n=2 * x.size)
        acf = np.fft.irfft(spectrum * np.conj(spectrum))[: max_lag + 1] / energy
        total += acf * x.size
        weight += x.size
    return total / weight if weight > 0 else np.full(max_lag + 1, np.nan)


def acf_similarity(acf: np.ndarray, target_acf: np.ndarray, fs: float) -> float:
    lags = np.arange(acf.size) / fs
    band = (lags >= ACF_BAND_SEC[0]) & (lags <= ACF_BAND_SEC[1])
    a, b = acf[band], target_acf[band]
    if not (np.isfinite(a).all() and np.isfinite(b).all()) or np.std(a) == 0 or np.std(b) == 0:
        return np.nan
    return float(np.corrcoef(a, b)[0, 1])


def wilcoxon_p(a: np.ndarray, b: np.ndarray) -> float:
    keep = np.isfinite(a) & np.isfinite(b)
    a, b = a[keep], b[keep]
    if len(a) < 2:
        return np.nan
    if np.allclose(a, b):
        return 1.0
    try:
        return float(stats.wilcoxon(a, b).pvalue)
    except ValueError:
        return np.nan


def p_text(p_value: float) -> str:
    if not np.isfinite(p_value):
        return "p=n/a"
    return "p<1e-4" if p_value < 1e-4 else f"p={p_value:.2g}"


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
    if args.variant not in module.VARIANTS:
        raise ValueError(f"{args.variant} is not a variant of {BACKBONE_NAME}")
    results_root = Path(args.results_root or module.DEFAULT_RESULTS_ROOT).resolve()
    out_root = (
        Path(args.output).resolve()
        if args.output
        else results_root / "Summary" / "Encoder_Feature_Analysis"
    )
    figure_root = out_root / "Figures"
    figure_root.mkdir(parents=True, exist_ok=True)
    device = torch.device(
        f"cuda:{args.gpu}" if torch.cuda.is_available() and not args.force_cpu else "cpu"
    )
    fs = float(module.TARGET_FS)
    labels = stage_labels(module)
    max_lag = int(round(ACF_MAX_LAG_SEC * fs))
    print(f"Results: {results_root}\nVariant: {args.variant}; device: {device}")

    member_rows, channel_rows = [], []
    curves: Dict[Tuple[str, str, str], Dict[str, List[np.ndarray]]] = {}
    window_specs = [spec for spec in module.WINDOW_SPECS if spec.name in args.windows]
    dataset_keys = [key for key in module.DATASET_ORDER if key in args.datasets]
    for spec in window_specs:
        for dataset_key in dataset_keys:
            label = configs[dataset_key]["label"]
            root = module.condition_root(results_root, args.variant, spec, dataset_key)
            if not (root / "Members").exists():
                print(f"{label} {spec.name}: no results, skipped")
                continue
            try:
                subject_data = module.load_dataset(dataset_key, spec)
            except Exception as exc:
                print(f"{label}: could not be loaded ({exc})")
                continue
            subjects = sorted(subject_data)
            if args.subjects:
                subjects = [s for s in subjects if s in set(args.subjects)]
            model = module.VARIANTS[args.variant].options.build().to(device)
            analysed = 0
            for subject in subjects:
                tag = module.safe_tag(subject)
                checkpoints = [
                    root / "Members" / f"m{member}" / "Models" / f"{tag}_best.pth"
                    for member in range(args.members)
                ]
                if not all(
                    module.member_paths(root, subject, member)[0].exists() and checkpoints[member].exists()
                    for member in range(args.members)
                ):
                    continue
                loader = module.make_eval_loader(
                    module.SubjectWindowDataset(subject_data, [subject], "test", False),
                    args.batch_size,
                    device,
                )
                metadata = module.test_metadata(subject_data, subject)
                for member, checkpoint in enumerate(checkpoints):
                    model.load_state_dict(torch.load(checkpoint, map_location=device))
                    stages, target = stage_features(model, loader, device)
                    samples = target.shape[1]
                    runs = contiguous_runs(metadata, samples)
                    min_length = int(round(ACF_MIN_RUN_SEC * fs))
                    target_frequencies, target_psd = mean_psd(target, fs)
                    target_acf = mean_acf(target, runs, max_lag, min_length)
                    key = (spec.name, label, subject)
                    store = curves.setdefault(key, {})
                    store.setdefault("Target_PSD", []).append(target_psd)
                    store.setdefault("Target_ACF", []).append(target_acf)
                    row = {
                        "Window_Config": spec.name,
                        "Dataset": label,
                        "Subject": subject,
                        "Member": member,
                        "N_Windows": int(target.shape[0]),
                        "Target_RespBand_Ratio": band_ratio(target_frequencies, target_psd),
                    }
                    for stage, features in zip(STAGES, stages):
                        abs_r = channel_abs_r(features, target)
                        for channel, value in enumerate(abs_r):
                            channel_rows.append({
                                "Window_Config": spec.name, "Dataset": label, "Subject": subject,
                                "Member": member, "Stage": stage, "Channel": channel, "Abs_r": float(value),
                            })
                        pc1, share = first_component(features)
                        frequencies, psd = mean_psd(pc1, fs)
                        acf = mean_acf(pc1, runs, max_lag, min_length)
                        store.setdefault(f"{stage}_PSD", []).append(psd)
                        store.setdefault(f"{stage}_ACF", []).append(acf)
                        row[f"{stage}_Channel_AbsR_Mean"] = float(np.nanmean(abs_r))
                        row[f"{stage}_PC1_Variance_Share"] = share
                        row[f"{stage}_RespBand_Ratio"] = band_ratio(frequencies, psd)
                        row[f"{stage}_ACF_r"] = acf_similarity(acf, target_acf, fs)
                    member_rows.append(row)
                    curves[("frequencies", spec.name, "")] = {"f": [target_frequencies]}
                analysed += 1
                del loader
            print(f"{label} {spec.name}: {analysed} subjects analysed ({len(subjects)} in the dataset)")
            del subject_data, model
            module.release_memory(device)

    members = pd.DataFrame(member_rows)
    if members.empty:
        print("No subject with all members stored.")
        return
    channels = pd.DataFrame(channel_rows)
    keys = ["Window_Config", "Dataset", "Subject"]
    value_columns = [c for c in members.columns if c not in keys + ["Member", "N_Windows"]]
    subjects_table = members.groupby(keys, sort=False)[value_columns].mean().reset_index()
    channel_subject = (
        channels.groupby(keys + ["Stage", "Channel"], sort=False)["Abs_r"].mean().reset_index()
    )

    corr_rows, psd_rows, acf_rows = [], [], []
    datasets_in_order = [configs[key]["label"] for key in dataset_keys]
    curve_arrays = {}
    for spec in window_specs:
        window = spec.name
        frame_window = subjects_table[subjects_table["Window_Config"] == window]
        if frame_window.empty:
            continue
        groups = [(d, frame_window[frame_window["Dataset"] == d]) for d in datasets_in_order]
        groups = [(d, f) for d, f in groups if not f.empty] + [(ALL_DATASETS, frame_window)]
        frequencies = curves[("frequencies", window, "")]["f"][0]
        lags = np.arange(max_lag + 1) / fs
        for dataset, frame in groups:
            tag = f"{window}_{dataset.replace(' ', '_')}"
            n = len(frame)
            # 1. channel correlation with the target
            ch = channel_subject[
                (channel_subject["Window_Config"] == window)
                & channel_subject["Subject"].isin(frame["Subject"])
                & (channel_subject["Dataset"].isin(frame["Dataset"].unique()))
            ]
            pivot = ch.pivot_table(index=["Dataset", "Subject", "Channel"], columns="Stage", values="Abs_r")
            delta = pivot[list(STAGES[1:])].sub(pivot["Stem"], axis=0)
            better = (delta > 0).groupby(level=["Dataset", "Subject"]).sum()
            stem_mean = frame["Stem_Channel_AbsR_Mean"].to_numpy(float)
            for stage, stage_label in zip(STAGES, labels):
                values = frame[f"{stage}_Channel_AbsR_Mean"].to_numpy(float)
                row = {
                    "Window_Config": window, "Dataset": dataset, "Stage": stage_label, "N_Subjects": n,
                    "PCC_Mean_Abs_r": float(np.mean(values)), "PCC_SD": float(np.std(values, ddof=1)) if n > 1 else np.nan,
                }
                if stage != "Stem":
                    row.update({
                        "Mean_Delta_Abs_r": float(np.mean(values - stem_mean)),
                        "Better_Channels_of_24": float(better[stage].mean()),
                        "N_Subjects_Higher": int(np.sum(values > stem_mean)),
                        "Wilcoxon_p_vs_Stem": wilcoxon_p(values, stem_mean),
                    })
                corr_rows.append(row)
            channel_delta = delta.groupby(level="Channel").mean()
            order = channel_delta[STAGES[-1]].sort_values(ascending=False).index
            fig, axis = plt.subplots(figsize=(6.0, 6.5))
            matrix = channel_delta.loc[order, list(STAGES[1:])].to_numpy()
            limit = max(float(np.nanmax(np.abs(matrix))), 1e-6)
            image = axis.imshow(matrix, aspect="auto", cmap="RdYlBu_r", norm=TwoSlopeNorm(0.0, -limit, limit))
            axis.set_xticks(range(len(STAGES) - 1))
            axis.set_xticklabels([l.replace(" ", "\n") for l in labels[1:]])
            axis.set_yticks(range(len(order)))
            axis.set_yticklabels([str(c) for c in order], fontsize=7)
            axis.set_ylabel("Shared channel index")
            for column, stage in enumerate(STAGES[1:]):
                axis.text(column, -0.8, f"{delta[stage].groupby(level=['Dataset', 'Subject']).mean().mean():+.3f}\n"
                          f"{better[stage].mean():.1f}/24", ha="center", va="bottom", fontsize=8,
                          bbox={"facecolor": "white", "edgecolor": "#999999", "pad": 1.5})
            axis.set_ylim(len(order) - 0.5, -2.2)
            fig.colorbar(image, ax=axis, label="delta |r|")
            axis.set_title(f"Mean delta |r| vs Stem: {dataset}, {window} (n = {n})", fontsize=10)
            fig.tight_layout()
            fig.savefig(figure_root / f"{tag}_channel_delta_heatmap.png", dpi=150)
            plt.close(fig)
            means = [float(frame[f"{s}_Channel_AbsR_Mean"].mean()) for s in STAGES]
            fig, axis = plt.subplots(figsize=(6.4, 4.4))
            axis.plot(range(len(STAGES)), means, "o-", color="#1f77b4")
            for i, value in enumerate(means):
                axis.annotate(f"{value:.3f}", (i, value), textcoords="offset points", xytext=(0, 7),
                              ha="center", fontsize=8, color="#1f77b4", fontweight="bold")
            bottom = min(means)
            for i, stage in enumerate(STAGES[1:], start=1):
                r = [c for c in corr_rows if c["Window_Config"] == window and c["Dataset"] == dataset
                     and c["Stage"] == labels[i]][0]
                axis.annotate(f"delta={r['Mean_Delta_Abs_r']:+.3f}\nbetter={r['Better_Channels_of_24']:.1f}/24\n"
                              f"{p_text(r['Wilcoxon_p_vs_Stem'])}", (i, bottom), textcoords="offset points",
                              xytext=(0, -42), ha="center", fontsize=6.5,
                              bbox={"facecolor": "white", "edgecolor": "#999999", "pad": 1.5})
            axis.set_xticks(range(len(STAGES)))
            axis.set_xticklabels([l.replace(" ", "\n") for l in labels])
            axis.set_ylabel("Mean |channel r| with the target")
            span = max(means) - bottom
            axis.set_ylim(bottom - 0.6 * max(span, 0.02), max(means) + 0.15 * max(span, 0.02))
            axis.grid(True, color="#e6e6e6")
            axis.set_title(f"Mean |channel r| summary: {dataset}, {window} (n = {n})", fontsize=10)
            fig.tight_layout()
            fig.savefig(figure_root / f"{tag}_channel_r_summary.png", dpi=150)
            plt.close(fig)

            # 2. PC1 respiratory-band power ratio, 3. PC1 periodicity
            subject_keys = list(zip(frame["Window_Config"], frame["Dataset"], frame["Subject"]))

            def mean_curve(name: str) -> np.ndarray:
                per_subject = [np.mean(curves[k][name], axis=0) for k in subject_keys]
                return np.nanmean(per_subject, axis=0)

            target_ratio = frame["Target_RespBand_Ratio"].to_numpy(float)
            stem_ratio = frame["Stem_RespBand_Ratio"].to_numpy(float)
            stem_acf = frame["Stem_ACF_r"].to_numpy(float)
            for stage, stage_label in zip(STAGES, labels):
                ratio = frame[f"{stage}_RespBand_Ratio"].to_numpy(float)
                acf_r = frame[f"{stage}_ACF_r"].to_numpy(float)
                psd_rows.append({
                    "Window_Config": window, "Dataset": dataset, "Stage": stage_label, "N_Subjects": n,
                    "RespBand_Ratio_Mean": float(np.nanmean(ratio)),
                    "RespBand_Ratio_SD": float(np.nanstd(ratio, ddof=1)) if n > 1 else np.nan,
                    "Target_RespBand_Ratio_Mean": float(np.nanmean(target_ratio)),
                    "PC1_Variance_Share_Mean": float(frame[f"{stage}_PC1_Variance_Share"].mean()),
                    "N_Subjects_Higher_than_Stem": int(np.sum(ratio > stem_ratio)) if stage != "Stem" else np.nan,
                    "Wilcoxon_p_vs_Stem": wilcoxon_p(ratio, stem_ratio) if stage != "Stem" else np.nan,
                })
                acf_rows.append({
                    "Window_Config": window, "Dataset": dataset, "Stage": stage_label, "N_Subjects": n,
                    "ACF_r_Mean": float(np.nanmean(acf_r)),
                    "ACF_r_SD": float(np.nanstd(acf_r, ddof=1)) if n > 1 else np.nan,
                    "N_Subjects_Higher_than_Stem": int(np.sum(acf_r > stem_acf)) if stage != "Stem" else np.nan,
                    "Wilcoxon_p_vs_Stem": wilcoxon_p(acf_r, stem_acf) if stage != "Stem" else np.nan,
                })
            psd_curves = {name: mean_curve(f"{name}_PSD") for name in ("Target",) + STAGES}
            acf_curves = {name: mean_curve(f"{name}_ACF") for name in ("Target",) + STAGES}
            for name, curve in psd_curves.items():
                curve_arrays[f"{tag}_PSD_{name}"] = curve
            for name, curve in acf_curves.items():
                curve_arrays[f"{tag}_ACF_{name}"] = curve
            curve_arrays[f"{window}_frequencies"] = frequencies
            curve_arrays["acf_lags_sec"] = lags

            shown = frequencies <= PSD_PLOT_MAX_HZ
            fig, axis = plt.subplots(figsize=(7.0, 4.6))
            axis.axvspan(*RESP_BAND_HZ, color="#59a14f", alpha=0.10, label="Respiratory band")
            legend_lines = []
            for name, stage_label in zip(("Target",) + STAGES, ["Target RSP"] + labels):
                curve = psd_curves[name][shown]
                scale = np.nanmax(curve) if np.nanmax(curve) > 0 else 1.0
                axis.plot(frequencies[shown], curve / scale, color=STAGE_COLORS[name],
                          linewidth=2.0 if name in ("Target", "B4") else 1.4, label=stage_label)
                ratio_value = (np.nanmean(target_ratio) if name == "Target"
                               else float(frame[f"{name}_RespBand_Ratio"].mean()))
                legend_lines.append(f"{stage_label}: {ratio_value:.3f}")
            axis.set_xlabel("Frequency (Hz)")
            axis.set_ylabel("PSD (scaled to its maximum)")
            axis.set_xlim(0, PSD_PLOT_MAX_HZ)
            axis.grid(True, color="#eeeeee")
            axis.legend(fontsize=7, loc="upper right")
            axis.text(1.02, 1.0, "Mean resp-band power ratio\n" + "\n".join(legend_lines),
                      transform=axis.transAxes, va="top", fontsize=8,
                      bbox={"facecolor": "white", "edgecolor": "#cc0000"})
            axis.set_title(f"Mean representative PSD (PC1): {dataset}, {window} (n = {n})", fontsize=10)
            fig.tight_layout()
            fig.savefig(figure_root / f"{tag}_pc1_psd.png", dpi=150, bbox_inches="tight")
            plt.close(fig)

            fig, axis = plt.subplots(figsize=(7.6, 4.6))
            axis.axvspan(*ACF_BAND_SEC, color="#e15759", alpha=0.06, label="Resp lag band")
            axis.plot(lags, acf_curves["Target"], color="#111111", linewidth=2.0, label="Target RSP")
            axis.plot(lags, acf_curves["B4"], color="#4c78a8", linewidth=1.6, label="DeepEnc (B4)")
            axis.plot(lags, acf_curves["Stem"], color="#e15759", linewidth=1.4, label="Stem Fuse")
            deep = [r for r in acf_rows if r["Window_Config"] == window and r["Dataset"] == dataset]
            p_deep = deep[-1]["Wilcoxon_p_vs_Stem"]
            axis.text(1.02, 1.0, f"Auto-correlation\nDeepEnc: r = {deep[-1]['ACF_r_Mean']:.3f}\n"
                      f"Stem Fuse: r = {deep[0]['ACF_r_Mean']:.3f}\n{p_text(p_deep)} (Wilcoxon)",
                      transform=axis.transAxes, va="top", fontsize=9,
                      bbox={"facecolor": "white", "edgecolor": "#cc0000"})
            axis.set_xlabel("Lag (s)")
            axis.set_ylabel("Normalized autocorrelation")
            axis.set_xlim(0, ACF_MAX_LAG_SEC)
            axis.grid(True, color="#eeeeee")
            axis.legend(fontsize=8, loc="upper right")
            axis.set_title(f"Mean readout autocorrelation (PC1): {dataset}, {window} (n = {n})", fontsize=10)
            fig.tight_layout()
            fig.savefig(figure_root / f"{tag}_pc1_acf.png", dpi=150, bbox_inches="tight")
            plt.close(fig)

    corr = pd.DataFrame(corr_rows)
    psd = pd.DataFrame(psd_rows)
    acf = pd.DataFrame(acf_rows)
    corr.to_csv(out_root / "encoder_channel_corr_tests.csv", index=False, encoding="utf-8-sig")
    psd.to_csv(out_root / "encoder_psd_tests.csv", index=False, encoding="utf-8-sig")
    acf.to_csv(out_root / "encoder_acf_tests.csv", index=False, encoding="utf-8-sig")
    subjects_table.to_csv(out_root / "encoder_subjects.csv", index=False, encoding="utf-8-sig")
    channel_subject.to_csv(out_root / "encoder_channels.csv", index=False, encoding="utf-8-sig")
    members.to_csv(out_root / "encoder_members.csv", index=False, encoding="utf-8-sig")
    np.savez_compressed(out_root / "encoder_curves.npz", **curve_arrays)
    module.write_results_excel(
        out_root / "encoder_feature_analysis.xlsx",
        {"Channel_corr": corr, "PC1_PSD": psd, "PC1_ACF": acf, "Subjects": subjects_table,
         "Members": members},
    )
    print("\nChannel |r| with the target (mean over subjects; vs Stem, Wilcoxon over subjects):")
    for _, r in corr.iterrows():
        extra = "" if r["Stage"] == "Stem" else (
            f"  delta {r['Mean_Delta_Abs_r']:+.3f}, better {r['Better_Channels_of_24']:.1f}/24, "
            f"{p_text(r['Wilcoxon_p_vs_Stem'])}")
        print(f"  {r['Window_Config'][1:4]:<4s} {r['Dataset']:<13s} {r['Stage']:<10s} PCC {r['PCC_Mean_Abs_r']:.3f}{extra}")
    print("\nPC1 respiratory-band power ratio (0.1-0.5 Hz) and ACF similarity to the target (2-10 s):")
    for (_, p), (_, a) in zip(psd.iterrows(), acf.iterrows()):
        stem = p["Stage"] == "Stem"
        print(f"  {p['Window_Config'][1:4]:<4s} {p['Dataset']:<13s} {p['Stage']:<10s} ratio {p['RespBand_Ratio_Mean']:.3f} "
              f"(target {p['Target_RespBand_Ratio_Mean']:.3f}{'' if stem else ', ' + p_text(p['Wilcoxon_p_vs_Stem'])})  "
              f"ACF r {a['ACF_r_Mean']:.3f}{'' if stem else ' (' + p_text(a['Wilcoxon_p_vs_Stem']) + ')'}")
    print(f"Saved under: {out_root}")


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--backbone", type=Path, default=Path(__file__).with_name(BACKBONE_NAME),
                        help=f"Path of {BACKBONE_NAME} (data paths and model are taken from it).")
    parser.add_argument("--results-root", type=Path, default=None,
                        help="Results root of the 261007 run (default: its PROJECT_ROOT).")
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--variant", default="full", help="261007 variant to analyse (default: full).")
    parser.add_argument("--datasets", nargs="+", default=["capnobase", "bidmc", "steam2"],
                        choices=["capnobase", "bidmc", "steam2"])
    parser.add_argument("--windows", nargs="+", default=["w10s_nonoverlap", "w20s_nonoverlap"])
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
