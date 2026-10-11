r"""261011_RRWaveNet_Feature_Results_10s: feature-level analyses of the
261007_RRWaveNet_Ensemble_NoEvent results, 10 s windows, for every dataset
(CapnoBase, BIDMC, STEAM2) and for all datasets together, in one script:
  1. stem fusion: effective rank and off-diagonal correlation of the
     encoder input, Stem Fusion O (full) vs X (no_stem_fusion);
  2. the four residual encoder blocks of the full model: channel
     correlation with the target, PC1 respiratory-band power, PC1
     autocorrelation.

Nothing is trained. Every stored model (3 members per held-out subject) is
reloaded and run on the test windows of its held-out subject, so the
windows are exactly the ones the models were tested on. Every value is
computed per model and averaged over the 3 models of a subject. Statistics
are over subjects with the paired Wilcoxon signed-rank test (two-sided),
for each dataset and for all datasets together (every subject of the three
datasets once). Subjects whose 3 models are not all stored yet are left out
and listed.

1. Stem fusion O vs X (encoder input)
   Stem Fusion O = the full model (three stem branches -> 1x1 conv +
   GroupNorm -> encoder): the 24 channels after the 1x1 conv + GroupNorm.
   Stem Fusion X = its no_stem_fusion ablation (same folds and seeds): the
   24 concatenated branch channels (k32 / k64 / k128, 8 each).
   Per test window, on its 24 x T feature map:
   * Effective rank (Roy & Vetterli, 2007): centre every channel over time,
     take the eigenvalues of the 24 x 24 channel covariance, normalize them
     to a distribution p and compute exp(-sum p log p) (1 to 24; also
     reported divided by 24).
   * Off-diagonal correlation: Pearson correlation of every pair of
     channels over time; the mean over the 24 x 23 off-diagonal entries as
     mean |r| and as signed mean. Pairs with a channel that is constant in
     the window are left out (their share is reported).
   The window values are averaged over the test windows of the subject,
   then over the 3 models. O and X are paired by subject: mean +- SD,
   median [IQR], number of subjects with a higher value in O, Wilcoxon
   p (**** p < 1e-4, *** < 1e-3, ** < 1e-2, * < 0.05, ns). Box plots: Stem
   Fusion X (left, red) vs Stem Fusion O (right, blue) with the
   significance bracket, per dataset and for all datasets, and one overview
   figure per metric. Skipped when the no_stem_fusion models are not stored.
2. Residual encoder blocks (full model)
   The 24-channel feature map at five stages: Stem = the stem fusion output
   (encoder input), B1 (d=1), B2 (d=2), B3 (d=4), B4 (d=8) = the output of
   each depthwise residual block (B4 = encoder output, "DeepEnc"). The
   blocks are depthwise, so channel c is the same channel at every stage.
   The target is the respiratory waveform of each window as the model was
   trained on it (min-max to [-1, 1]).
   * Channel correlation with the target: Pearson r between every channel
     and the target in every window, |r| averaged over the windows. Per
     stage: the mean |r| over the 24 channels, its change against Stem
     (mean delta |r|) and the number of channels (of 24) whose |r| is
     higher than at Stem. Figures: heatmap of the mean delta |r| per
     channel and block (channels in their own order, 23 at the top to 0)
     and the mean |channel r| per stage with delta, better and p.
   * PC1 respiratory-band power: PCA over the 24 channels (standardized over
     all samples of the subject); PSD of PC1 in every window (periodogram,
     Hann window), averaged over the windows. Ratio = power in 0.1-0.5 Hz /
     power above 0 Hz; the same for the target. Figure: mean PSD (0-1 Hz,
     each curve scaled to its maximum).
   * PC1 periodicity: consecutive test windows of a segment are joined into
     runs; normalized autocorrelation (ACF) of PC1 and of the target for
     lags 0-12 s in every run of at least 24 s, averaged over the runs
     (weighted by length). Similarity = Pearson r between the PC1 ACF and
     the target ACF over lags 2-10 s; each block against Stem and against
     the block before it. Figures: ACF of the target, DeepEnc (B4) and Stem
     Fuse; the ACF of every stage; the similarity r per stage.

Outputs: <261007 results root>\Summary_10s\
  Stem_Fusion_OX\
    stem_fusion_ox_members.csv    per model type, subject and member (the
                                  Pooled_* columns: the same metrics on all
                                  windows pooled, as in 261007 training)
    stem_fusion_ox_subjects.csv   per subject: O and X side by side
    stem_fusion_ox_tests.csv      per dataset / all datasets and metric
    stem_fusion_ox_table_<metric>.csv   mean +- SD table as in the slides
    stem_fusion_ox_skipped.csv    subjects left out
    stem_fusion_ox.xlsx           all tables
    Figures\<Dataset>_10s_<metric>.png, overview_10s_<metric>.png
  Encoder_Feature_Analysis\
    encoder_channel_corr_tests.csv, encoder_psd_tests.csv,
    encoder_acf_tests.csv, encoder_subjects.csv, encoder_channels.csv,
    encoder_members.csv, encoder_coverage.csv, encoder_curves.npz,
    encoder_feature_analysis.xlsx
    Figures\<Dataset>_10s_channel_delta_heatmap.png, _channel_r_summary.png,
            _pc1_psd.png, _pc1_acf.png, _pc1_acf_blocks.png,
            _pc1_acf_summary.png
  with <Dataset> = CapnoBase, BIDMC, STEAM2 and All_datasets.

It uses the 261007 file as it is (data paths, model and results root are
taken from it; default results root
D:\PPG2RSP_RRWaveNet_Inspired\261007_RRWaveNet_NoEvent_v1). Run next to
261007_RRWaveNet_Ensemble_NoEvent.py (Spyder: just run the file):
  python 261011_RRWaveNet_Feature_Results_10s.py
Options: --results-root <261007 results>, --datasets capnobase bidmc steam2,
  --force-cpu, --gpu, and the data path options of 261007.

Reference: Roy O, Vetterli M. The effective rank: a measure of effective
dimensionality. EUSIPCO 2007.
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
MEMBERS = 3
OUTPUT_FOLDER = "Summary_10s"
ALL_DATASETS = "All datasets"

# 1. Stem fusion O vs X
STEM_FOLDER = "Stem_Fusion_OX"
STEM_O_VARIANT = "full"
STEM_X_VARIANT = "no_stem_fusion"
# Box order and colours: X (left, red), O (right, blue).
MODEL_TYPES = (("Stem Fusion X", "#e8968c"), ("Stem Fusion O", "#5b8db8"))
STEM_METRICS = (
    # (column, axis label, drawn)
    ("EffectiveRank", "Effective rank", True),
    ("EffectiveRank_Normalized", "Effective rank / channels", False),
    ("OffDiag_AbsMean", "Off-diagonal correlation (mean |r|)", True),
    ("OffDiag_SignedMean", "Off-diagonal correlation (signed mean)", True),
)
CONSTANT_CHANNEL_SD = 1e-6

# 2. Residual encoder blocks
ENCODER_FOLDER = "Encoder_Feature_Analysis"
ENCODER_VARIANT = "full"
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
    spec = importlib.util.spec_from_file_location("rrwavenet_261007_features_10s", str(path))
    module = importlib.util.module_from_spec(spec)
    sys.modules["rrwavenet_261007_features_10s"] = module
    spec.loader.exec_module(module)
    return module


def stage_labels(backbone) -> List[str]:
    dilations = backbone.MODEL_CONFIG["encoder_dilations"]
    return ["Stem"] + [f"B{i + 1} (d={d})" for i, d in enumerate(dilations)]


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


def significance_stars(p_value: float) -> str:
    if not np.isfinite(p_value):
        return "n/a"
    for threshold, stars in ((1e-4, "****"), (1e-3, "***"), (1e-2, "**"), (0.05, "*")):
        if p_value < threshold:
            return stars
    return "ns"


# ---------------------------------------------------------------------------
# Feature extraction and metrics
# ---------------------------------------------------------------------------

def encoder_input_features(model, loader, device) -> np.ndarray:
    """The feature map that enters the encoder, for every test window."""
    model.eval()
    chunks = []
    with torch.no_grad():
        for x, _, _, _ in loader:
            chunks.append(model.fused_representation(x.to(device)).cpu().numpy().astype(np.float32))
    return np.concatenate(chunks)


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


def stem_row(backbone, features: np.ndarray, model_type: str, variant: str, dataset: str,
             subject: str, member: int) -> dict:
    """Window-averaged stem metrics of one model on its held-out subject."""
    row = {
        "Model_Type": model_type, "Variant": variant, "Window_Config": WINDOW, "Dataset": dataset,
        "Subject": subject, "Member": member, "N_Windows": int(features.shape[0]),
        "Channels": int(features.shape[1]),
    }
    for key, array in window_metrics(features).items():
        row[key] = float(np.nanmean(array)) if np.isfinite(array).any() else np.nan
    pooled = backbone.feature_representation_summary(features)
    row["Pooled_EffectiveRank"] = pooled["Feature_EffectiveRank"]
    row["Pooled_OffDiag_AbsMean"] = pooled["Feature_OffDiag_AbsMean"]
    row["Pooled_OffDiag_SignedMean"] = pooled["Feature_OffDiag_SignedMean"]
    return row


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


# ---------------------------------------------------------------------------
# Run every stored model on its held-out subject
# ---------------------------------------------------------------------------

def checkpoints_of(backbone, root: Path, subject: str) -> Optional[List[Path]]:
    """The 3 member checkpoints of a subject, or None if not all are stored."""
    tag = backbone.safe_tag(subject)
    paths = [root / "Members" / f"m{member}" / "Models" / f"{tag}_best.pth" for member in range(MEMBERS)]
    complete = all(
        backbone.member_paths(root, subject, member)[0].exists() and paths[member].exists()
        for member in range(MEMBERS)
    )
    return paths if complete else None


def collect(backbone, results_root: Path, dataset_keys: Sequence[str], device, args) -> dict:
    configs = backbone.DATASET_CONFIGS
    spec = next(s for s in backbone.WINDOW_SPECS if s.name == WINDOW)
    fs = float(backbone.TARGET_FS)
    max_lag = int(round(ACF_MAX_LAG_SEC * fs))
    min_length = int(round(ACF_MIN_RUN_SEC * fs))
    stem_rows: List[dict] = []
    stem_skipped: List[dict] = []
    member_rows: List[dict] = []
    channel_rows: List[dict] = []
    coverage_rows: List[dict] = []
    curves: Dict[Tuple[str, str], Dict[str, List[np.ndarray]]] = {}
    frequencies = None
    for dataset_key in dataset_keys:
        label = configs[dataset_key]["label"]
        root_o = backbone.condition_root(results_root, ENCODER_VARIANT, spec, dataset_key)
        root_x = backbone.condition_root(results_root, STEM_X_VARIANT, spec, dataset_key)
        has_o = (root_o / "Members").exists()
        has_x = (root_x / "Members").exists()
        if not has_o:
            print(f"{label}: no {ENCODER_VARIANT} results under {root_o}, skipped")
            continue
        if not has_x:
            print(f"{label}: no {STEM_X_VARIANT} results under {root_x}; stem fusion O vs X skipped")
        try:
            subject_data = backbone.load_dataset(dataset_key, spec)
        except Exception as exc:
            print(f"{label}: could not be loaded ({exc})")
            continue
        subjects = sorted(subject_data)
        if args.subjects:
            subjects = [s for s in subjects if s in set(args.subjects)]
        model_o = backbone.VARIANTS[ENCODER_VARIANT].options.build().to(device)
        model_x = backbone.VARIANTS[STEM_X_VARIANT].options.build().to(device) if has_x else None
        analysed, left_out, paired = [], [], 0
        for subject in subjects:
            checkpoints_o = checkpoints_of(backbone, root_o, subject)
            checkpoints_x = checkpoints_of(backbone, root_x, subject) if has_x else None
            if has_x:
                for name, found in (("Stem Fusion O", checkpoints_o), ("Stem Fusion X", checkpoints_x)):
                    if found is None:
                        stem_skipped.append({"Model_Type": name, "Window_Config": WINDOW, "Dataset": label,
                                             "Subject": subject, "Reason": "not all members stored"})
            if checkpoints_o is None:
                left_out.append(subject)
                continue
            loader = backbone.make_eval_loader(
                backbone.SubjectWindowDataset(subject_data, [subject], "test", False), args.batch_size, device
            )
            metadata = backbone.test_metadata(subject_data, subject)
            store = curves.setdefault((label, subject), {})
            for member, checkpoint in enumerate(checkpoints_o):
                model_o.load_state_dict(torch.load(checkpoint, map_location=device))
                stages, target = stage_features(model_o, loader, device)
                # Stem Fusion O: the encoder input of the full model (= stage Stem).
                if checkpoints_x is not None:
                    stem_rows.append(stem_row(backbone, stages[0], "Stem Fusion O", STEM_O_VARIANT,
                                              label, subject, member))
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
            # Stem Fusion X: the 24 concatenated branch channels of no_stem_fusion.
            if checkpoints_x is not None:
                for member, checkpoint in enumerate(checkpoints_x):
                    model_x.load_state_dict(torch.load(checkpoint, map_location=device))
                    features = encoder_input_features(model_x, loader, device)
                    stem_rows.append(stem_row(backbone, features, "Stem Fusion X", STEM_X_VARIANT,
                                              label, subject, member))
                paired += 1
            analysed.append(subject)
            del loader
        coverage_rows.append({
            "Dataset": label,
            "N_Subjects_in_Dataset": len(subjects),
            "N_Subjects_Analysed": len(analysed),
            "N_Subjects_Left_Out": len(left_out),
            "Left_Out_Subjects": " ".join(left_out),
            "N_Subjects_Stem_Fusion_OX": paired,
        })
        print(f"{label}: encoder blocks {len(analysed)} of {len(subjects)} subjects"
              + (f" (models not all stored: {' '.join(left_out)})" if left_out else "")
              + (f"; stem fusion O vs X {paired} subjects" if has_x else ""))
        del subject_data, model_o, model_x
        backbone.release_memory(device)
    return {
        "stem": pd.DataFrame(stem_rows),
        "stem_skipped": pd.DataFrame(stem_skipped,
                                     columns=["Model_Type", "Window_Config", "Dataset", "Subject", "Reason"]),
        "members": pd.DataFrame(member_rows),
        "channels": pd.DataFrame(channel_rows),
        "coverage": pd.DataFrame(coverage_rows),
        "curves": curves,
        "frequencies": frequencies,
    }


# ---------------------------------------------------------------------------
# 1. Stem fusion O vs X: statistics, tables and box plots
# ---------------------------------------------------------------------------

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


def stem_fusion_outputs(backbone, members: pd.DataFrame, skipped: pd.DataFrame, datasets: List[str],
                        out_root: Path) -> None:
    figure_root = out_root / "Figures"
    figure_root.mkdir(parents=True, exist_ok=True)
    skipped.to_csv(out_root / "stem_fusion_ox_skipped.csv", index=False, encoding="utf-8-sig")
    if members.empty:
        print("Stem fusion O vs X: no subject with the models of both types stored.")
        return
    metric_columns = [metric for metric, _, _ in STEM_METRICS] + ["Undefined_Pair_Share"]
    keys = ["Window_Config", "Dataset", "Subject"]
    per_type = members.groupby(["Model_Type"] + keys, sort=False)[metric_columns].mean().reset_index()
    wide = per_type.pivot_table(index=keys, columns="Model_Type", values=metric_columns, sort=False)
    wide.columns = [f"{metric}_{name.replace('Stem Fusion ', '')}" for metric, name in wide.columns]
    # Subjects analysed with both model types.
    subjects_table = wide.dropna(subset=["EffectiveRank_O", "EffectiveRank_X"]).reset_index()

    groups = [(d, subjects_table[subjects_table["Dataset"] == d]) for d in datasets]
    groups = [(d, f) for d, f in groups if not f.empty] + [(ALL_DATASETS, subjects_table)]
    test_rows = []
    for metric, axis_label, drawn in STEM_METRICS:
        overview = []
        for dataset, frame in groups:
            o = frame[f"{metric}_O"].to_numpy(float)
            x = frame[f"{metric}_X"].to_numpy(float)
            keep = np.isfinite(o) & np.isfinite(x)
            o, x = o[keep], x[keep]
            p_value = wilcoxon_p(o, x)
            test_rows.append({
                "Window_Config": WINDOW, "Dataset": dataset, "Metric": metric, "N_Subjects": int(len(o)),
                "O_Mean": float(np.mean(o)), "O_SD": float(np.std(o, ddof=1)) if len(o) > 1 else np.nan,
                "X_Mean": float(np.mean(x)), "X_SD": float(np.std(x, ddof=1)) if len(x) > 1 else np.nan,
                "O_Median": float(np.median(o)), "O_Q1": float(np.percentile(o, 25)),
                "O_Q3": float(np.percentile(o, 75)),
                "X_Median": float(np.median(x)), "X_Q1": float(np.percentile(x, 25)),
                "X_Q3": float(np.percentile(x, 75)),
                "Diff_O_minus_X_Mean": float(np.mean(o - x)),
                "N_Subjects_O_Higher": int(np.sum(o > x)), "N_Subjects_O_Lower": int(np.sum(o < x)),
                "Wilcoxon_p": p_value, "Significance": significance_stars(p_value),
            })
            overview.append((dataset, x, o, p_value))
            if drawn and len(o) >= 2:
                fig, axis = plt.subplots(figsize=(6.2, 4.0))
                draw_box(axis, x, o, axis_label, p_value)
                axis.set_title(f"{dataset}, {WINDOW_TAG}: n = {len(o)} subjects "
                               f"(mean of {MEMBERS} models), Wilcoxon p = {p_value:.2g}", fontsize=9)
                fig.tight_layout()
                fig.savefig(figure_root / f"{dataset.replace(' ', '_')}_{WINDOW_TAG}_{metric}.png", dpi=150)
                plt.close(fig)
        if drawn and overview:
            fig, axes = plt.subplots(1, len(overview), figsize=(4.2 * len(overview), 4.0), squeeze=False)
            for axis, (dataset, x, o, p_value) in zip(axes[0], overview):
                if len(o) >= 2:
                    draw_box(axis, x, o, axis_label, p_value)
                axis.set_title(f"{dataset} (n = {len(o)})", fontsize=10)
            fig.suptitle(f"{WINDOW_TAG}: {axis_label}", fontsize=11)
            fig.tight_layout()
            fig.savefig(figure_root / f"overview_{WINDOW_TAG}_{metric}.png", dpi=150)
            plt.close(fig)
    tests = pd.DataFrame(test_rows)

    # Tables as in the slides: mean +- SD, O starred when p < 0.05.
    tables = {}
    for metric, _, _ in STEM_METRICS:
        rows = tests[tests["Metric"] == metric]
        table_rows = []
        for name, prefix in (("Stem Fusion O", "O"), ("Stem Fusion X", "X")):
            for _, row in rows.iterrows():
                star = "*" if prefix == "O" and np.isfinite(row["Wilcoxon_p"]) and row["Wilcoxon_p"] < 0.05 else ""
                table_rows.append({
                    "Window_Config": WINDOW, "Model_Type": name, "Dataset": row["Dataset"],
                    metric: f"{row[f'{prefix}_Mean']:.3f} ± {row[f'{prefix}_SD']:.3f}{star}",
                    "Wilcoxon_p": row["Wilcoxon_p"], "N_Subjects": row["N_Subjects"],
                })
        tables[metric] = pd.DataFrame(table_rows)
        tables[metric].to_csv(out_root / f"stem_fusion_ox_table_{metric}.csv", index=False, encoding="utf-8-sig")
    members.to_csv(out_root / "stem_fusion_ox_members.csv", index=False, encoding="utf-8-sig")
    subjects_table.to_csv(out_root / "stem_fusion_ox_subjects.csv", index=False, encoding="utf-8-sig")
    tests.to_csv(out_root / "stem_fusion_ox_tests.csv", index=False, encoding="utf-8-sig")
    sheets = {"Tests": tests, "Subjects": subjects_table, "Members": members, "Skipped": skipped}
    sheets.update({f"Table_{metric}"[:31]: frame for metric, frame in tables.items()})
    backbone.write_results_excel(out_root / "stem_fusion_ox.xlsx", sheets)

    print("\n1. Stem Fusion O vs X (per subject, mean of the models; mean +- SD; Wilcoxon signed-rank):")
    for _, row in tests.iterrows():
        print(f"  {row['Dataset']:<13s} {row['Metric']:<25s} "
              f"O {row['O_Mean']:.3f} +- {row['O_SD']:.3f}  X {row['X_Mean']:.3f} +- {row['X_SD']:.3f}  "
              f"O higher {row['N_Subjects_O_Higher']}/{row['N_Subjects']}  p={row['Wilcoxon_p']:.2g} "
              f"{row['Significance']}")
    print(f"Saved under: {out_root}")


# ---------------------------------------------------------------------------
# 2. Residual encoder blocks: statistics over the subjects of one group
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


def encoder_outputs(backbone, collected: dict, datasets: List[str], out_root: Path) -> None:
    members, channels = collected["members"], collected["channels"]
    curves, frequencies = collected["curves"], collected["frequencies"]
    figure_root = out_root / "Figures"
    figure_root.mkdir(parents=True, exist_ok=True)
    fs = float(backbone.TARGET_FS)
    labels = stage_labels(backbone)
    keys = ["Window_Config", "Dataset", "Subject"]
    value_columns = [c for c in members.columns if c not in keys + ["Member", "N_Windows"]]
    subjects_table = members.groupby(keys, sort=False)[value_columns].mean().reset_index()
    channel_subject = channels.groupby(keys + ["Stage", "Channel"], sort=False)["Abs_r"].mean().reset_index()

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
    coverage = collected["coverage"]
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

    print("\n2. Channel |r| with the target (mean over subjects; vs Stem, Wilcoxon over subjects):")
    for _, r in corr.iterrows():
        extra = "" if r["Stage"] == "Stem" else (
            f"  delta {r['Mean_Delta_Abs_r']:+.3f}, better {r['Better_Channels_of_24']:.1f}/24, "
            f"{p_text(r['Wilcoxon_p_vs_Stem'])}")
        print(f"  {r['Dataset']:<13s} {r['Stage']:<10s} mean |r| {r['PCC_Mean_Abs_r']:.3f}{extra}")
    print("\n   PC1 respiratory-band power ratio (0.1-0.5 Hz) and ACF similarity to the target (2-10 s):")
    for (_, p), (_, a) in zip(psd.iterrows(), acf.iterrows()):
        stem = p["Stage"] == "Stem"
        print(f"  {p['Dataset']:<13s} {p['Stage']:<10s} ratio {p['RespBand_Ratio_Mean']:.3f} "
              f"(target {p['Target_RespBand_Ratio_Mean']:.3f}{'' if stem else ', ' + p_text(p['Wilcoxon_p_vs_Stem'])})"
              f"  ACF r {a['ACF_r_Mean']:.3f}"
              + ("" if stem else f" (vs Stem {p_text(a['Wilcoxon_p_vs_Stem'])}, "
                                 f"vs previous {p_text(a['Wilcoxon_p_vs_Previous'])})"))
    print(f"Saved under: {out_root}")


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
    out_root = results_root / OUTPUT_FOLDER
    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() and not args.force_cpu else "cpu")
    dataset_keys = [key for key in backbone.DATASET_ORDER if key in args.datasets]
    datasets = [configs[key]["label"] for key in dataset_keys]
    print(f"Results: {results_root}\n{WINDOW}, {MEMBERS} members per subject; "
          f"stem fusion O = {STEM_O_VARIANT}, X = {STEM_X_VARIANT}; encoder: {ENCODER_VARIANT}; device: {device}")

    collected = collect(backbone, results_root, dataset_keys, device, args)
    if collected["members"].empty:
        print("No subject with all members of the full model stored.")
        return
    stem_fusion_outputs(backbone, collected["stem"], collected["stem_skipped"], datasets, out_root / STEM_FOLDER)
    encoder_outputs(backbone, collected, datasets, out_root / ENCODER_FOLDER)


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
