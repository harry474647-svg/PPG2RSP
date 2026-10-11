r"""261011_RRWaveNet_Encoder_Feature_Results_10s: feature-level analysis of the
four residual encoder blocks of the 261007 full model, 10 s windows, for
every dataset (CapnoBase, BIDMC, STEAM2) and for all datasets together, in
one script.

Nothing is trained. Every stored model of the full model of
261007_RRWaveNet_Ensemble_NoEvent (3 members per held-out subject) is
reloaded and run on the test windows of its held-out subject. The
24-channel feature map is taken at five stages:
  Stem = the stem fusion output (encoder input),
  B1 (d=1), B2 (d=2), B3 (d=4), B4 (d=8) = the output of each depthwise
  residual block (B4 = encoder output, "DeepEnc").
The blocks are depthwise, so channel c is the same channel at every stage.
The target is the respiratory waveform of each window as the model was
trained on it (min-max to [-1, 1]).

1. Channel correlation with the target
   For every window, channel and stage: Pearson r between the channel and
   the target over time. |r| is averaged over the windows -> |r| per
   channel. Per stage: the mean |r| over the 24 channels, its change
   against Stem (mean delta |r|) and the number of channels (of 24) whose
   |r| is higher than at Stem ("better"). Figures: heatmap of the mean
   delta |r| per channel and block (channels in their own order, 23 at the
   top to 0), and the mean |channel r| per stage with delta, better and
   the p value.
2. PC1 power in the respiratory band
   PCA over the 24 channels (channels standardized over all samples of the
   subject); PC1 = first principal component (its share of the variance is
   reported). PSD of PC1 in every window (periodogram, Hann window over the
   whole window), averaged over the windows. Respiratory-band power ratio =
   power in 0.1-0.5 Hz / power above 0 Hz; the same ratio for the target.
   Figure: mean PSD (0-1 Hz, each curve scaled to its maximum).
3. PC1 periodicity
   Consecutive test windows of a recording segment are joined into runs
   (windows do not overlap). Normalized autocorrelation (ACF) of PC1 and of
   the target for lags 0-12 s in every run of at least 24 s, averaged over
   the runs (weighted by length). Similarity to the target = Pearson r
   between the PC1 ACF and the target ACF over lags 2-10 s. Figures: mean
   ACF of the target, DeepEnc (B4) and Stem Fuse; the mean ACF of every
   stage; the similarity r per stage with its changes and p values.
Every value is computed per model and averaged over the 3 models of a
subject. Statistics are over subjects: each block against Stem (and, for
the ACF, against the block before it) with the paired Wilcoxon
signed-rank test (two-sided), for each dataset and for all datasets
together (every subject of the three datasets). Subjects whose 3 models
are not all stored yet are left out and listed in encoder_coverage.csv.

Outputs: <261007 results root>\Summary_10s\Encoder_Feature_Analysis\
  encoder_channel_corr_tests.csv   per dataset and stage: mean |r|, delta, better, p
  encoder_psd_tests.csv            per dataset and stage: resp-band ratio, p vs Stem
  encoder_acf_tests.csv            per dataset and stage: ACF similarity r, p vs
                                   Stem and vs the previous block
  encoder_subjects.csv             per subject and stage: all metrics
  encoder_channels.csv             per subject, stage and channel: |r|
  encoder_members.csv              per model (member)
  encoder_coverage.csv             subjects analysed / left out per dataset
  encoder_curves.npz               mean PSD and ACF curves
  encoder_feature_analysis.xlsx    all tables
  Figures\<Dataset>_10s_channel_delta_heatmap.png
         \<Dataset>_10s_channel_r_summary.png
         \<Dataset>_10s_pc1_psd.png
         \<Dataset>_10s_pc1_acf.png          (Target, DeepEnc, Stem Fuse)
         \<Dataset>_10s_pc1_acf_blocks.png   (every block)
         \<Dataset>_10s_pc1_acf_summary.png  (similarity r per block)
  with <Dataset> = CapnoBase, BIDMC, STEAM2 and All_datasets.

It uses the 261007 file as it is (data paths, model and results root are
taken from it; default results root
D:\PPG2RSP_RRWaveNet_Inspired\261007_RRWaveNet_NoEvent_v1). Run next to
261007_RRWaveNet_Ensemble_NoEvent.py (Spyder: just run the file):
  python 261011_RRWaveNet_Encoder_Feature_Results_10s.py
Options: --results-root <261007 results>, --datasets capnobase bidmc steam2,
  --force-cpu, --gpu, and the data path options of 261007.
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

SCRIPT_DIR = Path(__file__).resolve().parent
BACKBONE = SCRIPT_DIR / "261007_RRWaveNet_Ensemble_NoEvent.py"
WINDOW = "w10s_nonoverlap"
WINDOW_TAG = "10s"
VARIANT = "full"
MEMBERS = 3
OUTPUT_FOLDER = "Summary_10s"
ANALYSIS_FOLDER = "Encoder_Feature_Analysis"
ALL_DATASETS = "All datasets"
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


def load_backbone(path: Path):
    """Import the 261007 script as a module (no training is started)."""
    spec = importlib.util.spec_from_file_location("rrwavenet_261007_encoder_10s", str(path))
    module = importlib.util.module_from_spec(spec)
    sys.modules["rrwavenet_261007_encoder_10s"] = module
    spec.loader.exec_module(module)
    return module


def stage_labels(backbone) -> List[str]:
    dilations = backbone.MODEL_CONFIG["encoder_dilations"]
    return ["Stem"] + [f"B{i + 1} (d={d})" for i, d in enumerate(dilations)]


# ---------------------------------------------------------------------------
# Feature extraction and metrics
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# 1. Run every stored model of the full model on its held-out subject
# ---------------------------------------------------------------------------

def collect(backbone, results_root: Path, dataset_keys: Sequence[str], device, args):
    configs = backbone.DATASET_CONFIGS
    spec = next(s for s in backbone.WINDOW_SPECS if s.name == WINDOW)
    fs = float(backbone.TARGET_FS)
    max_lag = int(round(ACF_MAX_LAG_SEC * fs))
    min_length = int(round(ACF_MIN_RUN_SEC * fs))
    member_rows: List[dict] = []
    channel_rows: List[dict] = []
    coverage_rows: List[dict] = []
    curves: Dict[Tuple[str, str], Dict[str, List[np.ndarray]]] = {}
    frequencies = None
    for dataset_key in dataset_keys:
        label = configs[dataset_key]["label"]
        root = backbone.condition_root(results_root, VARIANT, spec, dataset_key)
        if not (root / "Members").exists():
            print(f"{label}: no {VARIANT} results under {root}, skipped")
            continue
        try:
            subject_data = backbone.load_dataset(dataset_key, spec)
        except Exception as exc:
            print(f"{label}: could not be loaded ({exc})")
            continue
        subjects = sorted(subject_data)
        if args.subjects:
            subjects = [s for s in subjects if s in set(args.subjects)]
        model = backbone.VARIANTS[VARIANT].options.build().to(device)
        analysed, left_out = [], []
        for subject in subjects:
            tag = backbone.safe_tag(subject)
            checkpoints = [
                root / "Members" / f"m{member}" / "Models" / f"{tag}_best.pth" for member in range(MEMBERS)
            ]
            if not all(
                backbone.member_paths(root, subject, member)[0].exists() and checkpoints[member].exists()
                for member in range(MEMBERS)
            ):
                left_out.append(subject)
                continue
            loader = backbone.make_eval_loader(
                backbone.SubjectWindowDataset(subject_data, [subject], "test", False), args.batch_size, device
            )
            metadata = backbone.test_metadata(subject_data, subject)
            store = curves.setdefault((label, subject), {})
            for member, checkpoint in enumerate(checkpoints):
                model.load_state_dict(torch.load(checkpoint, map_location=device))
                stages, target = stage_features(model, loader, device)
                runs = contiguous_runs(metadata, target.shape[1])
                frequencies, target_psd = mean_psd(target, fs)
                target_acf = mean_acf(target, runs, max_lag, min_length)
                store.setdefault("Target_PSD", []).append(target_psd)
                store.setdefault("Target_ACF", []).append(target_acf)
                row = {
                    "Window_Config": WINDOW,
                    "Dataset": label,
                    "Subject": subject,
                    "Member": member,
                    "N_Windows": int(target.shape[0]),
                    "Target_RespBand_Ratio": band_ratio(frequencies, target_psd),
                }
                for stage, features in zip(STAGES, stages):
                    abs_r = channel_abs_r(features, target)
                    for channel, value in enumerate(abs_r):
                        channel_rows.append({
                            "Window_Config": WINDOW, "Dataset": label, "Subject": subject, "Member": member,
                            "Stage": stage, "Channel": channel, "Abs_r": float(value),
                        })
                    pc1, share = first_component(features)
                    _, psd = mean_psd(pc1, fs)
                    acf = mean_acf(pc1, runs, max_lag, min_length)
                    store.setdefault(f"{stage}_PSD", []).append(psd)
                    store.setdefault(f"{stage}_ACF", []).append(acf)
                    row[f"{stage}_Channel_AbsR_Mean"] = float(np.nanmean(abs_r))
                    row[f"{stage}_PC1_Variance_Share"] = share
                    row[f"{stage}_RespBand_Ratio"] = band_ratio(frequencies, psd)
                    row[f"{stage}_ACF_r"] = acf_similarity(acf, target_acf, fs)
                member_rows.append(row)
            analysed.append(subject)
            del loader
        coverage_rows.append({
            "Dataset": label,
            "N_Subjects_in_Dataset": len(subjects),
            "N_Subjects_Analysed": len(analysed),
            "N_Subjects_Left_Out": len(left_out),
            "Left_Out_Subjects": " ".join(left_out),
        })
        print(f"{label}: {len(analysed)} of {len(subjects)} subjects analysed"
              + (f" (models not all stored: {' '.join(left_out)})" if left_out else ""))
        del subject_data, model
        backbone.release_memory(device)
    return (pd.DataFrame(member_rows), pd.DataFrame(channel_rows), pd.DataFrame(coverage_rows),
            curves, frequencies)


# ---------------------------------------------------------------------------
# 2. Statistics over the subjects of one group (a dataset or all datasets)
# ---------------------------------------------------------------------------

def group_statistics(frame: pd.DataFrame, channel_subject: pd.DataFrame, dataset: str,
                     labels: List[str]) -> dict:
    n = len(frame)
    channels = channel_subject.merge(frame[["Dataset", "Subject"]], on=["Dataset", "Subject"])
    pivot = channels.pivot_table(index=["Dataset", "Subject", "Channel"], columns="Stage", values="Abs_r")
    delta = pivot[list(STAGES[1:])].sub(pivot["Stem"], axis=0)
    better = (delta > 0).groupby(level=["Dataset", "Subject"]).sum()

    corr_rows, psd_rows, acf_rows = [], [], []
    stem_mean = frame["Stem_Channel_AbsR_Mean"].to_numpy(float)
    target_ratio = frame["Target_RespBand_Ratio"].to_numpy(float)
    stem_ratio = frame["Stem_RespBand_Ratio"].to_numpy(float)
    stem_acf = frame["Stem_ACF_r"].to_numpy(float)
    for position, (stage, stage_label) in enumerate(zip(STAGES, labels)):
        values = frame[f"{stage}_Channel_AbsR_Mean"].to_numpy(float)
        row = {
            "Window_Config": WINDOW, "Dataset": dataset, "Stage": stage_label, "N_Subjects": n,
            "PCC_Mean_Abs_r": float(np.mean(values)),
            "PCC_SD": float(np.std(values, ddof=1)) if n > 1 else np.nan,
        }
        if position:
            row.update({
                "Mean_Delta_Abs_r": float(np.mean(values - stem_mean)),
                "Better_Channels_of_24": float(better[stage].mean()),
                "N_Subjects_Higher": int(np.sum(values > stem_mean)),
                "Wilcoxon_p_vs_Stem": wilcoxon_p(values, stem_mean),
            })
        corr_rows.append(row)

        ratio = frame[f"{stage}_RespBand_Ratio"].to_numpy(float)
        psd_rows.append({
            "Window_Config": WINDOW, "Dataset": dataset, "Stage": stage_label, "N_Subjects": n,
            "RespBand_Ratio_Mean": float(np.nanmean(ratio)),
            "RespBand_Ratio_SD": float(np.nanstd(ratio, ddof=1)) if n > 1 else np.nan,
            "Target_RespBand_Ratio_Mean": float(np.nanmean(target_ratio)),
            "PC1_Variance_Share_Mean": float(frame[f"{stage}_PC1_Variance_Share"].mean()),
            "N_Subjects_Higher_than_Stem": int(np.sum(ratio > stem_ratio)) if position else np.nan,
            "Wilcoxon_p_vs_Stem": wilcoxon_p(ratio, stem_ratio) if position else np.nan,
        })

        acf_r = frame[f"{stage}_ACF_r"].to_numpy(float)
        previous = frame[f"{STAGES[position - 1]}_ACF_r"].to_numpy(float) if position else None
        acf_rows.append({
            "Window_Config": WINDOW, "Dataset": dataset, "Stage": stage_label, "N_Subjects": n,
            "ACF_r_Mean": float(np.nanmean(acf_r)),
            "ACF_r_SD": float(np.nanstd(acf_r, ddof=1)) if n > 1 else np.nan,
            "ACF_r_Median": float(np.nanmedian(acf_r)),
            "Delta_vs_Stem": float(np.nanmean(acf_r - stem_acf)) if position else np.nan,
            "N_Subjects_Higher_than_Stem": int(np.sum(acf_r > stem_acf)) if position else np.nan,
            "Wilcoxon_p_vs_Stem": wilcoxon_p(acf_r, stem_acf) if position else np.nan,
            "Delta_vs_Previous": float(np.nanmean(acf_r - previous)) if position else np.nan,
            "N_Subjects_Higher_than_Previous": int(np.sum(acf_r > previous)) if position else np.nan,
            "Wilcoxon_p_vs_Previous": wilcoxon_p(acf_r, previous) if position else np.nan,
        })
    return {"n": n, "delta": delta, "better": better, "corr": corr_rows, "psd": psd_rows, "acf": acf_rows}


def group_curves(frame: pd.DataFrame, curves: dict) -> Tuple[Dict[str, np.ndarray], Dict[str, np.ndarray]]:
    """Mean PSD and ACF curves: mean over the 3 models, then over the subjects."""
    keys = list(zip(frame["Dataset"], frame["Subject"]))

    def mean_curve(name: str) -> np.ndarray:
        return np.nanmean([np.mean(curves[key][name], axis=0) for key in keys], axis=0)

    names = ("Target",) + STAGES
    return ({name: mean_curve(f"{name}_PSD") for name in names},
            {name: mean_curve(f"{name}_ACF") for name in names})


# ---------------------------------------------------------------------------
# 3. Figures
# ---------------------------------------------------------------------------

def figure_channel_heatmap(path: Path, stats_: dict, labels: List[str], title: str) -> None:
    delta, better = stats_["delta"], stats_["better"]
    channel_delta = delta.groupby(level="Channel").mean()
    # Rows in channel order, 23 (top) to 0 (bottom); no sorting by value.
    order = sorted(channel_delta.index, reverse=True)
    matrix = channel_delta.loc[order, list(STAGES[1:])].to_numpy()
    limit = max(float(np.nanmax(np.abs(matrix))), 1e-6)
    fig, axis = plt.subplots(figsize=(6.0, 6.5))
    image = axis.imshow(matrix, aspect="auto", cmap="RdYlBu_r", norm=TwoSlopeNorm(0.0, -limit, limit))
    axis.set_xticks(range(len(STAGES) - 1))
    axis.set_xticklabels([l.replace(" ", "\n") for l in labels[1:]])
    axis.set_yticks(range(len(order)))
    axis.set_yticklabels([str(c) for c in order], fontsize=7)
    axis.set_ylabel("Shared channel index")
    for column, stage in enumerate(STAGES[1:]):
        subject_delta = delta[stage].groupby(level=["Dataset", "Subject"]).mean().mean()
        axis.text(column, -0.8, f"{subject_delta:+.3f}\n{better[stage].mean():.1f}/24", ha="center",
                  va="bottom", fontsize=8, bbox={"facecolor": "white", "edgecolor": "#999999", "pad": 1.5})
    axis.set_ylim(len(order) - 0.5, -2.2)
    fig.colorbar(image, ax=axis, label="delta |r|")
    axis.set_title(f"Mean delta |r| vs Stem: {title}", fontsize=10)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def figure_channel_summary(path: Path, frame: pd.DataFrame, corr_rows: List[dict], labels: List[str],
                           title: str) -> None:
    means = [float(frame[f"{s}_Channel_AbsR_Mean"].mean()) for s in STAGES]
    fig, axis = plt.subplots(figsize=(6.4, 4.4))
    axis.plot(range(len(STAGES)), means, "o-", color="#1f77b4")
    for i, value in enumerate(means):
        axis.annotate(f"{value:.3f}", (i, value), textcoords="offset points", xytext=(0, 7),
                      ha="center", fontsize=8, color="#1f77b4", fontweight="bold")
    bottom = min(means)
    for i, row in enumerate(corr_rows[1:], start=1):
        axis.annotate(f"delta={row['Mean_Delta_Abs_r']:+.3f}\nbetter={row['Better_Channels_of_24']:.1f}/24\n"
                      f"{p_text(row['Wilcoxon_p_vs_Stem'])}", (i, bottom), textcoords="offset points",
                      xytext=(0, -42), ha="center", fontsize=6.5,
                      bbox={"facecolor": "white", "edgecolor": "#999999", "pad": 1.5})
    axis.set_xticks(range(len(STAGES)))
    axis.set_xticklabels([l.replace(" ", "\n") for l in labels])
    axis.set_ylabel("Mean |channel r| with the target")
    span = max(means) - bottom
    axis.set_ylim(bottom - 0.6 * max(span, 0.02), max(means) + 0.15 * max(span, 0.02))
    axis.grid(True, color="#e6e6e6")
    axis.set_title(f"Mean |channel r| summary: {title}", fontsize=10)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def figure_pc1_psd(path: Path, frequencies: np.ndarray, psd_curves: Dict[str, np.ndarray],
                   frame: pd.DataFrame, labels: List[str], title: str) -> None:
    shown = frequencies <= PSD_PLOT_MAX_HZ
    fig, axis = plt.subplots(figsize=(7.0, 4.6))
    axis.axvspan(*RESP_BAND_HZ, color="#59a14f", alpha=0.10, label="Respiratory band")
    legend_lines = []
    for name, stage_label in zip(("Target",) + STAGES, ["Target RSP"] + labels):
        curve = psd_curves[name][shown]
        scale = np.nanmax(curve) if np.nanmax(curve) > 0 else 1.0
        axis.plot(frequencies[shown], curve / scale, color=STAGE_COLORS[name],
                  linewidth=2.0 if name in ("Target", "B4") else 1.4, label=stage_label)
        ratio = float(frame[f"{name}_RespBand_Ratio"].mean())
        legend_lines.append(f"{stage_label}: {ratio:.3f}")
    axis.set_xlabel("Frequency (Hz)")
    axis.set_ylabel("PSD (scaled to its maximum)")
    axis.set_xlim(0, PSD_PLOT_MAX_HZ)
    axis.grid(True, color="#eeeeee")
    axis.legend(fontsize=7, loc="upper right")
    axis.text(1.02, 1.0, "Mean resp-band power ratio\n" + "\n".join(legend_lines),
              transform=axis.transAxes, va="top", fontsize=8,
              bbox={"facecolor": "white", "edgecolor": "#cc0000"})
    axis.set_title(f"Mean representative PSD (PC1): {title}", fontsize=10)
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def figure_pc1_acf(path: Path, lags: np.ndarray, acf_curves: Dict[str, np.ndarray], acf_rows: List[dict],
                   title: str) -> None:
    fig, axis = plt.subplots(figsize=(7.6, 4.6))
    axis.axvspan(*ACF_BAND_SEC, color="#e15759", alpha=0.06, label="Resp lag band")
    axis.plot(lags, acf_curves["Target"], color="#111111", linewidth=2.0, label="Target RSP")
    axis.plot(lags, acf_curves["B4"], color="#4c78a8", linewidth=1.6, label="DeepEnc (B4)")
    axis.plot(lags, acf_curves["Stem"], color="#e15759", linewidth=1.4, label="Stem Fuse")
    axis.text(1.02, 1.0, f"Auto-correlation\nDeepEnc: r = {acf_rows[-1]['ACF_r_Mean']:.3f}\n"
              f"Stem Fuse: r = {acf_rows[0]['ACF_r_Mean']:.3f}\n"
              f"{p_text(acf_rows[-1]['Wilcoxon_p_vs_Stem'])} (Wilcoxon)",
              transform=axis.transAxes, va="top", fontsize=9,
              bbox={"facecolor": "white", "edgecolor": "#cc0000"})
    axis.set_xlabel("Lag (s)")
    axis.set_ylabel("Normalized autocorrelation")
    axis.set_xlim(0, ACF_MAX_LAG_SEC)
    axis.grid(True, color="#eeeeee")
    axis.legend(fontsize=8, loc="upper right")
    axis.set_title(f"Mean readout autocorrelation (PC1): {title}", fontsize=10)
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def figure_pc1_acf_blocks(path: Path, lags: np.ndarray, acf_curves: Dict[str, np.ndarray],
                          acf_rows: List[dict], labels: List[str], title: str) -> None:
    fig, axis = plt.subplots(figsize=(7.6, 4.6))
    axis.axvspan(*ACF_BAND_SEC, color="#e15759", alpha=0.06, label="Resp lag band")
    for name, stage_label in zip(("Target",) + STAGES, ["Target RSP"] + labels):
        axis.plot(lags, acf_curves[name], color=STAGE_COLORS[name],
                  linewidth=2.2 if name == "Target" else 1.4, label=stage_label)
    lines = []
    for position, row in enumerate(acf_rows):
        extra = f" ({p_text(row['Wilcoxon_p_vs_Stem'])})" if position else ""
        lines.append(f"{row['Stage']}: r = {row['ACF_r_Mean']:.3f}{extra}")
    axis.text(1.02, 1.0, "Auto-correlation vs target\n(lags 2-10 s; p vs Stem)\n" + "\n".join(lines),
              transform=axis.transAxes, va="top", fontsize=8,
              bbox={"facecolor": "white", "edgecolor": "#cc0000"})
    axis.set_xlabel("Lag (s)")
    axis.set_ylabel("Normalized autocorrelation")
    axis.set_xlim(0, ACF_MAX_LAG_SEC)
    axis.grid(True, color="#eeeeee")
    axis.legend(fontsize=7, loc="upper right")
    axis.set_title(f"Mean readout autocorrelation per block (PC1): {title}", fontsize=10)
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def figure_pc1_acf_summary(path: Path, acf_rows: List[dict], labels: List[str], n: int, title: str) -> None:
    values = [row["ACF_r_Mean"] for row in acf_rows]
    fig, axis = plt.subplots(figsize=(6.4, 4.4))
    axis.plot(range(len(STAGES)), values, "o-", color="#1f77b4")
    for i, value in enumerate(values):
        axis.annotate(f"{value:.3f}", (i, value), textcoords="offset points", xytext=(0, 7),
                      ha="center", fontsize=8, color="#1f77b4", fontweight="bold")
    bottom = min(values)
    for i, row in enumerate(acf_rows[1:], start=1):
        axis.annotate(f"vs Stem {row['Delta_vs_Stem']:+.3f}\n"
                      f"higher {row['N_Subjects_Higher_than_Stem']}/{n}, {p_text(row['Wilcoxon_p_vs_Stem'])}\n"
                      f"vs prev. {row['Delta_vs_Previous']:+.3f}, {p_text(row['Wilcoxon_p_vs_Previous'])}",
                      (i, bottom), textcoords="offset points", xytext=(0, -46), ha="center",
                      fontsize=6, bbox={"facecolor": "white", "edgecolor": "#999999", "pad": 1.5})
    axis.set_xticks(range(len(STAGES)))
    axis.set_xticklabels([l.replace(" ", "\n") for l in labels])
    axis.set_ylabel("ACF similarity to the target (r, lags 2-10 s)")
    span = max(max(values) - bottom, 0.05)
    axis.set_ylim(bottom - 0.7 * span, max(values) + 0.15 * span)
    axis.grid(True, color="#e6e6e6")
    axis.set_title(f"PC1 autocorrelation similarity per block: {title}", fontsize=10)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(args: argparse.Namespace) -> None:
    backbone = load_backbone(Path(args.backbone).resolve())
    configs = backbone.DATASET_CONFIGS
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
    results_root = Path(args.results_root or backbone.DEFAULT_RESULTS_ROOT).resolve()
    out_root = results_root / OUTPUT_FOLDER / ANALYSIS_FOLDER
    figure_root = out_root / "Figures"
    figure_root.mkdir(parents=True, exist_ok=True)
    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() and not args.force_cpu else "cpu")
    fs = float(backbone.TARGET_FS)
    labels = stage_labels(backbone)
    dataset_keys = [key for key in backbone.DATASET_ORDER if key in args.datasets]
    print(f"Results: {results_root}\nModel: {VARIANT}, {WINDOW}, {MEMBERS} members; device: {device}")

    members, channels, coverage, curves, frequencies = collect(backbone, results_root, dataset_keys, device, args)
    if members.empty:
        print("No subject with all members stored.")
        return
    keys = ["Window_Config", "Dataset", "Subject"]
    value_columns = [c for c in members.columns if c not in keys + ["Member", "N_Windows"]]
    subjects_table = members.groupby(keys, sort=False)[value_columns].mean().reset_index()
    channel_subject = channels.groupby(keys + ["Stage", "Channel"], sort=False)["Abs_r"].mean().reset_index()

    datasets = [configs[key]["label"] for key in dataset_keys]
    groups = [(d, subjects_table[subjects_table["Dataset"] == d]) for d in datasets]
    groups = [(d, f) for d, f in groups if not f.empty] + [(ALL_DATASETS, subjects_table)]
    lags = np.arange(int(round(ACF_MAX_LAG_SEC * fs)) + 1) / fs
    corr_rows, psd_rows, acf_rows = [], [], []
    curve_arrays = {"frequencies_hz": frequencies, "acf_lags_sec": lags}
    for dataset, frame in groups:
        tag = f"{dataset.replace(' ', '_')}_{WINDOW_TAG}"
        stats_ = group_statistics(frame, channel_subject, dataset, labels)
        n = stats_["n"]
        title = f"{dataset}, {WINDOW_TAG} (n = {n})"
        corr_rows += stats_["corr"]
        psd_rows += stats_["psd"]
        acf_rows += stats_["acf"]
        psd_curves, acf_curves = group_curves(frame, curves)
        for name in psd_curves:
            curve_arrays[f"{tag}_PSD_{name}"] = psd_curves[name]
            curve_arrays[f"{tag}_ACF_{name}"] = acf_curves[name]
        figure_channel_heatmap(figure_root / f"{tag}_channel_delta_heatmap.png", stats_, labels, title)
        figure_channel_summary(figure_root / f"{tag}_channel_r_summary.png", frame, stats_["corr"], labels, title)
        figure_pc1_psd(figure_root / f"{tag}_pc1_psd.png", frequencies, psd_curves, frame, labels, title)
        figure_pc1_acf(figure_root / f"{tag}_pc1_acf.png", lags, acf_curves, stats_["acf"], title)
        figure_pc1_acf_blocks(figure_root / f"{tag}_pc1_acf_blocks.png", lags, acf_curves, stats_["acf"],
                              labels, title)
        figure_pc1_acf_summary(figure_root / f"{tag}_pc1_acf_summary.png", stats_["acf"], labels, n, title)

    corr = pd.DataFrame(corr_rows)
    psd = pd.DataFrame(psd_rows)
    acf = pd.DataFrame(acf_rows)
    corr.to_csv(out_root / "encoder_channel_corr_tests.csv", index=False, encoding="utf-8-sig")
    psd.to_csv(out_root / "encoder_psd_tests.csv", index=False, encoding="utf-8-sig")
    acf.to_csv(out_root / "encoder_acf_tests.csv", index=False, encoding="utf-8-sig")
    subjects_table.to_csv(out_root / "encoder_subjects.csv", index=False, encoding="utf-8-sig")
    channel_subject.to_csv(out_root / "encoder_channels.csv", index=False, encoding="utf-8-sig")
    members.to_csv(out_root / "encoder_members.csv", index=False, encoding="utf-8-sig")
    coverage.to_csv(out_root / "encoder_coverage.csv", index=False, encoding="utf-8-sig")
    np.savez_compressed(out_root / "encoder_curves.npz", **curve_arrays)
    backbone.write_results_excel(
        out_root / "encoder_feature_analysis.xlsx",
        {"Channel_corr": corr, "PC1_PSD": psd, "PC1_ACF": acf, "Subjects": subjects_table,
         "Members": members, "Coverage": coverage},
    )

    print("\nChannel |r| with the target (mean over subjects; vs Stem, Wilcoxon over subjects):")
    for _, r in corr.iterrows():
        extra = "" if r["Stage"] == "Stem" else (
            f"  delta {r['Mean_Delta_Abs_r']:+.3f}, better {r['Better_Channels_of_24']:.1f}/24, "
            f"{p_text(r['Wilcoxon_p_vs_Stem'])}")
        print(f"  {r['Dataset']:<13s} {r['Stage']:<10s} mean |r| {r['PCC_Mean_Abs_r']:.3f}{extra}")
    print("\nPC1 respiratory-band power ratio (0.1-0.5 Hz) and ACF similarity to the target (2-10 s):")
    for (_, p), (_, a) in zip(psd.iterrows(), acf.iterrows()):
        stem = p["Stage"] == "Stem"
        print(f"  {p['Dataset']:<13s} {p['Stage']:<10s} ratio {p['RespBand_Ratio_Mean']:.3f} "
              f"(target {p['Target_RespBand_Ratio_Mean']:.3f}{'' if stem else ', ' + p_text(p['Wilcoxon_p_vs_Stem'])})"
              f"  ACF r {a['ACF_r_Mean']:.3f}"
              + ("" if stem else f" (vs Stem {p_text(a['Wilcoxon_p_vs_Stem'])}, "
                                 f"vs previous {p_text(a['Wilcoxon_p_vs_Previous'])})"))
    print(f"\nSaved under: {out_root}")


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--backbone", type=Path, default=BACKBONE,
                        help="Path of 261007_RRWaveNet_Ensemble_NoEvent.py (data paths and model).")
    parser.add_argument("--results-root", type=Path, default=None,
                        help="Results root of the 261007 run (default: its PROJECT_ROOT).")
    parser.add_argument("--datasets", nargs="+", default=["capnobase", "bidmc", "steam2"],
                        choices=["capnobase", "bidmc", "steam2"])
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
    main(parse_args())
