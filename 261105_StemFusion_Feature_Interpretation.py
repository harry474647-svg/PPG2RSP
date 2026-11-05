r"""261105_StemFusion_Feature_Interpretation: what the 1x1 stem fusion changes
inside the trained networks (Stem Fusion O = full model, X = no_stem_fusion),
from the stored models of 261007 and 261013 (nothing is trained).

Why the stem fusion is the place to look
----------------------------------------
The four residual encoder blocks are depthwise (every channel is filtered on
its own) and their GroupNorm is per channel. Without the stem fusion (X) the
24 encoder channels are the 8 channels of each stem branch (k32, k64, k128)
and stay separate through the whole encoder: the time scales first meet in
the decoder. With the stem fusion (O) the 1x1 convolution is the only layer
before the decoder that mixes the branches, so every encoder channel works on
a mixture of the time scales.

Models (10 s non-overlapping windows; every model on its held-out subject)
-------------------------------------------------------------------------
* 261007 (--results-root-261007; CapnoBase, BIDMC, STEAM2 as 261007 read and
  trained them): O = full\, X = no_stem_fusion\, members m0-m2 of both (the
  same seeds), read with the 261007 code.
* 261013 (--results-root-261013; BIDMC, STEAM2): O = the full model of the
  weight decay 261013 selected in the fold (members m0-m2); X = the model
  without stem fusion trained by 261028_Stem_ablation_noise_v1 with the 261013
  settings of the fold (--results-root-noise; single model m0, checked to be
  trained with the selected weight decay), read with the 261013 code.
O vs X is paired by held-out subject on member m0 (same seed, training data
and weight decay). E uses m0-m2 and is therefore 261007 only (261013 has no
X members m1, m2). Initial weights ("init") are drawn with the member's seed
exactly as the training scripts draw them (set_seed, then the network is
built); no step of training is run.

Analyses
--------
A. Gradient flow (training windows of the fold, the validation windows and
   the held-out subject excluded as in training: 261007 the first 80% of every
   training subject, 261013 the training subjects that are not validation
   subjects; --grad-windows windows spread over them, batches of 48, no
   jitter, eval mode so that dropout does not add noise; init and final
   weights). Loss = the training loss (v16 waveform loss). Per batch the
   gradient of every stem branch, of the stem as a whole, of the encoder and
   of the decoder + head; then
     Grad_Share_k32/k64/k128   mean gradient norm of a branch / sum of the 3
     Grad_Branch_Balance       entropy of the 3 shares / log 3 (1 = every
                               branch gets the same learning signal)
     Grad_Stem_Coherence       mean cosine similarity of the stem gradients of
                               the batches (1 = every batch pushes the stem the
                               same way; a consistent learning signal)
     Grad_Stem_to_Head         stem gradient norm / decoder + head gradient
                               norm (how much learning signal reaches the
                               input layers)
   Tested at init (the learning signal the stem gets when training starts)
   and Grad_Branch_Balance also with the final weights; coherence and the
   stem / head ratio of the final weights are descriptive (near a minimum the
   batch gradients largely cancel).
B. Scale composition of the fused channels (O; X is fixed: every encoder
   channel is one branch). From the 1x1 weights W (24 x 24) and the SD of
   every branch channel on the test windows: contribution |W_cj| x SD_j;
   per fused channel the share of each branch, Mix_Entropy (entropy / log 3,
   0 = one branch, 1 = all equally), Mixed_Channels (channels with >= 2
   branches of >= 20% share), Share_k32/k64/k128 (mean over the channels);
   the same from |W| alone (Weight_*). Final vs init (does training make the
   channels mix more, and which scale do they draw on).
C. Branch reliance (test windows): the output of one branch replaced by its
   time mean in every window (its temporal information removed), the rest of
   the network unchanged; Delta_Loss_k* = loss increase (test loss with the
   target), Output_Change_k* = 1 - Pearson r of the intact and the occluded
   output (mean over windows); Reliance_Share_k* = Delta_Loss_k / sum,
   Reliance_Balance = entropy / log 3, Reliance_Max_Share.
D. Fundamental vs second harmonic (test windows). f0 of every window =
   spectral peak of the target in the rate band (CapnoBase 3-60, BIDMC /
   STEAM2 6-30 breaths/min; windows with f0 >= 0.08 Hz). Periodogram with a
   Hann window, zero padded to 4096 points; every channel scaled to unit
   variance, the channels summed. P1 = power within +-0.04 Hz of f0, P2 =
   within +-0.04 Hz of 2 f0. Per window log10(P2 / P1) of the stem fusion
   output (Stem), the encoder output (B4), the model output and the target
   (the target is the same for O and X, so it is listed but not compared);
   Fundamental_Share_B4 = P1 / power 0.05-2 Hz; Harmonic_Dominant_B4 = share
   of windows with P2 > P1. Median over the windows of a subject. Mean
   spectra on a f / f0 axis per dataset and for the --case-subjects.
E. Consistency across seeds (261007; members m0-m2 of O and of X, same
   seeds): Output_Agreement_r (mean Pearson r of the outputs of every member
   pair, per window), CKA_Stem / CKA_B4 (linear CKA of the features of every
   member pair, Kornblith et al. 2019; time points as samples), from the
   stored per-minute rates of the members (Pred_Wave_FFT_RR):
   Rate_Agreement_1bpm (share of minutes in which the 3 members are within
   1 breath/min) and Harmonic_Disagreement (share of minutes in which one
   member is at 1.75-2.25 x another).
F. Respiratory phase in the encoder output (test windows, member m0). Phase
   of the target = angle of its analytic signal (Hilbert); time points 1 s
   from the window edges, every 4th sample; features standardized per
   subject. Phase_Probe_R2_B4 / _Stem: ridge regression (alpha 1) of
   cos / sin of the phase from the 24 channels, 5-fold cross-validation over
   blocks of consecutive windows, R2 averaged over cos and sin;
   Phase_Probe_Error_B4: mean |phase error| (rad); Phase_kNN_Error_B4: mean
   |phase difference| to the 10 nearest time points of other windows (<= 3000
   points; low = the feature space is ordered by the breathing phase).
   t-SNE (perplexity 30, PCA init) of <= 2000 time points of B4, X and O side
   by side, coloured by the phase, for --tsne-per-dataset subjects per
   dataset (the 25th / 50th / 75th percentile of the test MAE of O m0) and the
   --case-subjects (or --tsne-subjects). A t-SNE map is drawn per subject:
   every fold has its own network, so points of different folds do not share
   a feature space.

Statistics
----------
Per source (261007, 261013), dataset and over the datasets of the source
together: O vs X paired by subject, median [IQR], mean, number of subjects
with a higher value in O, rank-biserial correlation (positive = O higher),
two-sided Wilcoxon signed-rank test, Benjamini-Hochberg over the tested
metrics of a source and dataset. The direction that favours the stem fusion
is fixed per metric before the run (Favourable_If); Result = "favours O" /
"favours X" when FDR < 0.05 in that / the other direction, "n.s." otherwise,
"descriptive" for metrics without a direction. B: final vs init of O (paired
Wilcoxon). Every metric is reported whichever way it comes out.

Folders
-------
The script is placed in and run from
D:\PPG2RSP_RRWaveNet_Inspired\261105_StemFusion_Feature_Interpretation; every
result is written there (--output-root), separately for 261007 and 261013.
261007_RRWaveNet_Ensemble_NoEvent.py and 261013_RRWaveNet_Adult_SubjectVal_WD.py
(data reading and network) are looked for in this folder, the current folder,
the results folder of their run, D:\PPG2RSP_RRWaveNet_Inspired and its
subfolders (first found; or --backbone-261007 / --backbone-261013). The result
folders of 261007, 261013 and 261028_Stem_ablation_noise_v1 are only read.

Outputs (in --output-root, one folder per source, written when the source is done)
  261007\  and  261013\
    {source}_StemFusion_Feature_Interpretation.xlsx   Tests, every table
                               of the source, Coverage, Notes
    Tests.csv, Coverage.csv
    A_Gradients\        A_Gradients.csv, A_Gradients.png
    B_Fusion\           B_Fusion.csv, B_Fusion_Channels.csv,
                        B_Training_Effect.csv, B_Fusion_Shares.png
    C_Branch_Reliance\  C_Branch_Reliance.csv, C_Branch_Reliance.png
    D_Harmonics\        D_Harmonics.csv, D_Harmonics.png, D_Spectra.png,
                        D_Spectra_Cases.png, harmonic_curves.npz
    E_Consistency\      E_Consistency.csv, E_Consistency.png (261007 only)
    F_Phase\            F_Phase.csv, F_Phase.png, tSNE_{dataset}.pdf,
                        tSNE_{dataset}\{subject}.png

Run (in D:\PPG2RSP_RRWaveNet_Inspired\261105_StemFusion_Feature_Interpretation):
  python 261105_StemFusion_Feature_Interpretation.py
Options: --sources 261007 261013, --datasets capnobase bidmc steam2,
  --subjects, --results-root-261007 / --results-root-261013 /
  --results-root-noise, --backbone-261007 / --backbone-261013,
  --output-root, --grad-windows 384,
  --case-subjects CapnoBase:0312 BIDMC:05 BIDMC:27, --tsne-per-dataset 3,
  --tsne-subjects BIDMC:05 ... or all, --no-tsne, --force-cpu, --gpu, and
  the data paths of 261007 / 261013.
"""

import argparse
import importlib.util
import json
import math
import sys
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

sys.dont_write_bytecode = True

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from matplotlib.backends.backend_pdf import PdfPages
from scipy import stats
from scipy.signal import hilbert
from torch.utils.data import DataLoader, Subset

SCRIPT_DIR = Path(__file__).resolve().parent
# Where the script is run from; every result goes here (one folder per source).
PROJECT_ROOT = Path(r"D:\PPG2RSP_RRWaveNet_Inspired\261105_StemFusion_Feature_Interpretation")
WORKSPACE = PROJECT_ROOT.parent
RESULTS_ROOT_261007 = WORKSPACE / "261007_RRWaveNet_NoEvent_v1"
RESULTS_ROOT_261013 = WORKSPACE / "261013_RRWavNet_STEAM_BIDMC_v1"
RESULTS_ROOT_NOISE = WORKSPACE / "261028_Stem_ablation_noise_v1"
BACKBONE_FILES = {"261007": "261007_RRWaveNet_Ensemble_NoEvent.py",
                  "261013": "261013_RRWaveNet_Adult_SubjectVal_WD.py"}
SECTION_FOLDERS = {"A": "A_Gradients", "B": "B_Fusion", "C": "C_Branch_Reliance", "D": "D_Harmonics",
                   "E": "E_Consistency", "F": "F_Phase"}
SOURCE_DATASETS = {"261007": ("capnobase", "bidmc", "steam2"), "261013": ("bidmc", "steam2")}
WINDOW = "w10s_nonoverlap"
MEMBERS = 3
O_VARIANT, X_VARIANT = "full", "no_stem_fusion"
MODEL_NAMES = {"O": "Stem Fusion O", "X": "Stem Fusion X"}
MODEL_COLORS = {"X": "#e8968c", "O": "#5b8db8"}
ALL_DATASETS = "All datasets"
RATE_BAND_BPM = {"capnobase": (3.0, 60.0), "bidmc": (6.0, 30.0), "steam2": (6.0, 30.0)}
GRAD_WINDOWS = 384
BATCH = 48
MIXED_SHARE = 0.2
HARMONIC_HALF_WIDTH_HZ = 0.04
HARMONIC_MIN_F0_HZ = 0.08
NFFT = 4096
F0_GRID = np.linspace(0.0, 3.0, 121)
EDGE_SEC = 1.0
POINT_STEP = 4
PROBE_FOLDS = 5
RIDGE_ALPHA = 1.0
KNN_POINTS, KNN_K = 3000, 10
TSNE_POINTS, TSNE_PERPLEXITY = 2000, 30.0
CASE_SUBJECTS = ("CapnoBase:0312", "BIDMC:05", "BIDMC:27")
TSNE_PER_DATASET = 3

# Metric -> (section, direction that favours the stem fusion: "higher", "lower" or "" = descriptive)
METRICS = {
    "Grad_Branch_Balance_Init": ("A", "higher"), "Grad_Branch_Balance_Final": ("A", "higher"),
    "Grad_Stem_Coherence_Init": ("A", "higher"), "Grad_Stem_Coherence_Final": ("A", ""),
    "Grad_Stem_to_Head_Init": ("A", "higher"), "Grad_Stem_to_Head_Final": ("A", ""),
    "Grad_Share_k32_Final": ("A", ""), "Grad_Share_k64_Final": ("A", ""), "Grad_Share_k128_Final": ("A", ""),
    "Loss_Intact": ("C", ""), "Reliance_Balance": ("C", "higher"), "Reliance_Max_Share": ("C", "lower"),
    "Reliance_Share_k32": ("C", ""), "Reliance_Share_k64": ("C", ""), "Reliance_Share_k128": ("C", ""),
    "Delta_Loss_Total": ("C", ""),
    "Harmonic_LogRatio_Stem": ("D", "lower"), "Harmonic_LogRatio_B4": ("D", "lower"),
    "Harmonic_LogRatio_Output": ("D", "lower"), "Fundamental_Share_B4": ("D", "higher"),
    "Harmonic_Dominant_B4": ("D", "lower"),
    "Output_Agreement_r": ("E", "higher"), "CKA_Stem": ("E", "higher"), "CKA_B4": ("E", "higher"),
    "Rate_Agreement_1bpm": ("E", "higher"), "Harmonic_Disagreement": ("E", "lower"),
    "Phase_Probe_R2_B4": ("F", "higher"), "Phase_Probe_Error_B4": ("F", "lower"),
    "Phase_Probe_R2_Stem": ("F", "higher"), "Phase_kNN_Error_B4": ("F", "lower"),
}
FIGURE_METRICS = {
    "A": ("Grad_Branch_Balance_Init", "Grad_Stem_Coherence_Init", "Grad_Stem_to_Head_Init",
          "Grad_Branch_Balance_Final"),
    "C": ("Reliance_Balance", "Reliance_Max_Share", "Reliance_Share_k128"),
    "D": ("Harmonic_LogRatio_B4", "Harmonic_LogRatio_Output", "Fundamental_Share_B4"),
    "E": ("Output_Agreement_r", "CKA_B4", "Rate_Agreement_1bpm", "Harmonic_Disagreement"),
    "F": ("Phase_Probe_R2_B4", "Phase_Probe_Error_B4", "Phase_kNN_Error_B4"),
}


def find_backbone(source: str, explicit: Optional[Path], results_root: Path) -> Path:
    """The 261007 / 261013 script: --backbone-*, else the first found in this
    folder, the current folder, the results folder of the run, the workspace
    and its subfolders."""
    name = BACKBONE_FILES[source]
    if explicit is not None:
        if Path(explicit).is_file():
            return Path(explicit).resolve()
        raise FileNotFoundError(f"--backbone-{source}: {explicit} not found")
    folders = [SCRIPT_DIR, Path.cwd(), results_root, WORKSPACE, SCRIPT_DIR.parent]
    for base in (WORKSPACE, SCRIPT_DIR.parent):
        if base.is_dir():
            folders += sorted(child for child in base.iterdir() if child.is_dir())
    tried = []
    for folder in folders:
        path = Path(folder) / name
        if path not in tried:
            tried.append(path)
            if path.is_file():
                return path.resolve()
    raise FileNotFoundError(f"{name} not found (looked in {', '.join(str(p.parent) for p in tried[:6])} and the "
                            f"subfolders of {WORKSPACE}); give it with --backbone-{source}")


def load_backbone(path: Path, name: str):
    """Import the 261007 or 261013 script as a module (no training is started)."""
    spec = importlib.util.spec_from_file_location(name, str(path))
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


# ---------------------------------------------------------------------------
# Small helpers
# ---------------------------------------------------------------------------

def norm_entropy(shares: np.ndarray) -> float:
    p = np.clip(np.asarray(shares, float), 0.0, None)
    total = p.sum()
    if total <= 0:
        return math.nan
    p = p / total
    p = p[p > 0]
    return float(-(p * np.log(p)).sum() / math.log(3))


def pearson_rows(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    a = a - a.mean(axis=1, keepdims=True)
    b = b - b.mean(axis=1, keepdims=True)
    den = np.sqrt((a * a).sum(axis=1) * (b * b).sum(axis=1))
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(den > 1e-12, (a * b).sum(axis=1) / den, np.nan)


def linear_cka(a: np.ndarray, b: np.ndarray) -> float:
    """Linear CKA of two (samples x channels) matrices."""
    a = a - a.mean(axis=0)
    b = b - b.mean(axis=0)
    cross = np.linalg.norm(a.T @ b) ** 2
    den = np.linalg.norm(a.T @ a) * np.linalg.norm(b.T @ b)
    return float(cross / den) if den > 1e-12 else math.nan


def as_points(features: np.ndarray) -> np.ndarray:
    """(windows, channels, T) -> (windows x T, channels)."""
    return features.transpose(0, 2, 1).reshape(-1, features.shape[1]).astype(np.float64)


def wrap(angle: np.ndarray) -> np.ndarray:
    return (angle + np.pi) % (2 * np.pi) - np.pi


def channel_points(features: np.ndarray, fs: float) -> Tuple[np.ndarray, np.ndarray]:
    """(windows, channels, T) -> time points (n, channels) and their window index."""
    edge = int(round(EDGE_SEC * fs))
    idx = np.arange(edge, features.shape[2] - edge, POINT_STEP)
    pts = np.transpose(features[:, :, idx], (0, 2, 1)).reshape(-1, features.shape[1])
    windows = np.repeat(np.arange(features.shape[0]), len(idx))
    return pts.astype(np.float64), windows


def target_phase(target: np.ndarray, fs: float) -> np.ndarray:
    """Phase of the analytic signal of every target window at the probe points."""
    edge = int(round(EDGE_SEC * fs))
    idx = np.arange(edge, target.shape[1] - edge, POINT_STEP)
    phase = np.angle(hilbert(target - target.mean(axis=1, keepdims=True), axis=1))
    return phase[:, idx].reshape(-1)


def standardize(points: np.ndarray) -> np.ndarray:
    sd = points.std(axis=0)
    keep = sd > 1e-8
    return (points[:, keep] - points[:, keep].mean(axis=0)) / sd[keep]


# ---------------------------------------------------------------------------
# Running a network
# ---------------------------------------------------------------------------

def forward_parts(model, x: torch.Tensor):
    """Concatenated stem branches, stem fusion output, encoder output, output."""
    stem = torch.cat([branch(x) for branch in model.stem_branches], dim=1)
    fused = model.stem_fuse(stem)
    encoded = model.encoder(fused)
    out = torch.tanh(model.wave_head(model.decoder(encoded)))
    return stem, fused, encoded, out


def from_stem(model, stem: torch.Tensor) -> torch.Tensor:
    return torch.tanh(model.wave_head(model.decoder(model.encoder(model.stem_fuse(stem)))))


def run_test(model, x: torch.Tensor, device) -> Dict[str, np.ndarray]:
    model.eval()
    parts = {"stem": [], "fused": [], "b4": [], "out": []}
    with torch.no_grad():
        for start in range(0, len(x), BATCH):
            stem, fused, encoded, out = forward_parts(model, x[start:start + BATCH].to(device))
            parts["stem"].append(stem.cpu().numpy())
            parts["fused"].append(fused.cpu().numpy())
            parts["b4"].append(encoded.cpu().numpy())
            parts["out"].append(out[:, 0].cpu().numpy())
    return {key: np.concatenate(value).astype(np.float32) for key, value in parts.items()}


def branch_slices(model) -> List[Tuple[str, slice]]:
    width = int(model.stem_base_channels)
    return [(label, slice(k * width, (k + 1) * width)) for k, label in enumerate(model.branch_labels)]


# A ---------------------------------------------------------------------------

def gradient_stats(model, batches, loss_fn, device) -> Dict[str, float]:
    model.eval()
    branch_params = [list(branch.parameters()) for branch in model.stem_branches]
    head_params = list(model.decoder.parameters()) + list(model.wave_head.parameters())
    norms = {k: [] for k in range(len(branch_params))}
    stem_vectors, stem_norms, head_norms = [], [], []
    with torch.enable_grad():
        for x, y in batches:
            model.zero_grad(set_to_none=True)
            loss, _ = loss_fn(model(x.to(device)), y.to(device))
            loss.backward()
            vectors = []
            for k, params in enumerate(branch_params):
                vector = torch.cat([p.grad.reshape(-1) for p in params if p.grad is not None]).detach().cpu()
                norms[k].append(float(vector.norm()))
                vectors.append(vector)
            stem = torch.cat(vectors)
            stem_vectors.append(stem.numpy().astype(np.float64))
            stem_norms.append(float(stem.norm()))
            head = torch.cat([p.grad.reshape(-1) for p in head_params if p.grad is not None]).detach().cpu()
            head_norms.append(float(head.norm()))
    model.zero_grad(set_to_none=True)
    mean_norms = np.array([np.mean(norms[k]) for k in sorted(norms)])
    shares = mean_norms / mean_norms.sum() if mean_norms.sum() > 0 else np.full(len(mean_norms), np.nan)
    g = np.array(stem_vectors)
    unit = g / np.clip(np.linalg.norm(g, axis=1, keepdims=True), 1e-20, None)
    cos = unit @ unit.T
    pairs = cos[np.triu_indices(len(g), 1)]
    row = {f"Grad_Share_{label}": float(s) for label, s in zip(model.branch_labels, shares)}
    row.update({
        "Grad_Branch_Balance": norm_entropy(shares),
        "Grad_Stem_Coherence": float(pairs.mean()) if len(pairs) else math.nan,
        "Grad_Stem_to_Head": float(np.mean(stem_norms) / max(np.mean(head_norms), 1e-20)),
    })
    return row


# B ---------------------------------------------------------------------------

def fusion_composition(model, stem: np.ndarray) -> Tuple[dict, List[dict]]:
    """Branch shares of every fused channel (activation weighted and from |W|)."""
    weight = np.abs(model.stem_fuse[0].weight.detach().cpu().numpy()[:, :, 0])
    sd = stem.transpose(1, 0, 2).reshape(stem.shape[1], -1).std(axis=1)
    slices = branch_slices(model)
    summary, channels = {}, []
    for name, contribution in (("", weight * sd[None, :]), ("Weight_", weight)):
        shares = np.stack([contribution[:, s].sum(axis=1) for _, s in slices], axis=1)
        shares = shares / np.clip(shares.sum(axis=1, keepdims=True), 1e-20, None)
        entropy = np.array([norm_entropy(row) for row in shares])
        summary[f"{name}Mix_Entropy"] = float(np.nanmean(entropy))
        summary[f"{name}Mixed_Channels"] = int(((shares >= MIXED_SHARE).sum(axis=1) >= 2).sum())
        for k, (label, _) in enumerate(slices):
            summary[f"{name}Share_{label}"] = float(shares[:, k].mean())
            summary[f"{name}Dominant_{label}"] = int((shares.argmax(axis=1) == k).sum())
        if not name:
            for c in range(shares.shape[0]):
                channels.append({"Channel": c, **{f"Share_{label}": float(shares[c, k])
                                                   for k, (label, _) in enumerate(slices)},
                                 "Mix_Entropy": float(entropy[c])})
    return summary, channels


# C ---------------------------------------------------------------------------

def branch_reliance(model, stem: np.ndarray, out: np.ndarray, y: torch.Tensor, loss_fn, device) -> dict:
    model.eval()

    def loss_and_output(stem_array: np.ndarray) -> Tuple[float, np.ndarray]:
        total, outputs = 0.0, []
        with torch.no_grad():
            for start in range(0, len(stem_array), BATCH):
                s = torch.from_numpy(stem_array[start:start + BATCH]).to(device)
                o = from_stem(model, s)
                total += float(loss_fn(o, y[start:start + BATCH].to(device))[0]) * len(s)
                outputs.append(o[:, 0].cpu().numpy())
        return total / len(stem_array), np.concatenate(outputs)

    intact, _ = loss_and_output(stem)
    row = {"Loss_Intact": intact}
    deltas = []
    for label, s in branch_slices(model):
        occluded = stem.copy()
        occluded[:, s, :] = occluded[:, s, :].mean(axis=2, keepdims=True)
        loss, output = loss_and_output(occluded)
        row[f"Delta_Loss_{label}"] = loss - intact
        row[f"Output_Change_{label}"] = float(np.nanmean(1.0 - pearson_rows(out, output)))
        deltas.append(max(loss - intact, 0.0))
    deltas = np.array(deltas)
    total = deltas.sum()
    for (label, _), value in zip(branch_slices(model), deltas):
        row[f"Reliance_Share_{label}"] = float(value / total) if total > 0 else math.nan
    row["Reliance_Balance"] = norm_entropy(deltas) if total > 0 else math.nan
    row["Reliance_Max_Share"] = float(deltas.max() / total) if total > 0 else math.nan
    row["Delta_Loss_Total"] = float(total)
    return row


# D ---------------------------------------------------------------------------

def spectra(signals: np.ndarray, fs: float) -> Tuple[np.ndarray, np.ndarray]:
    """(windows, channels, T) -> frequencies and the channel sum of the unit-variance periodograms."""
    x = signals - signals.mean(axis=-1, keepdims=True)
    sd = x.std(axis=-1, keepdims=True)
    x = np.where(sd > 1e-8, x / np.where(sd > 1e-8, sd, 1.0), 0.0)
    x = x * np.hanning(x.shape[-1])
    power = np.abs(np.fft.rfft(x, n=NFFT, axis=-1)) ** 2
    return np.fft.rfftfreq(NFFT, 1.0 / fs), power.sum(axis=1)


def harmonics(target: np.ndarray, parts: Dict[str, np.ndarray], fs: float, band_bpm) -> Tuple[dict, Dict[str, np.ndarray]]:
    freqs, target_power = spectra(target[:, None, :], fs)
    in_band = (freqs >= band_bpm[0] / 60.0) & (freqs <= band_bpm[1] / 60.0)
    f0 = freqs[in_band][np.argmax(target_power[:, in_band], axis=1)]
    keep = f0 >= HARMONIC_MIN_F0_HZ
    row: dict = {"N_Windows_Harmonic": int(keep.sum())}
    curves: Dict[str, np.ndarray] = {}
    if not keep.any():
        return row, curves

    def p_at(power: np.ndarray, centre: np.ndarray) -> np.ndarray:
        mask = np.abs(freqs[None, :] - centre[:, None]) <= HARMONIC_HALF_WIDTH_HZ
        return (power * mask).sum(axis=1)

    f0k = f0[keep]
    ratios = {}
    for name, signal in (("Target", target[keep][:, None, :]), ("Stem", parts["fused"][keep]),
                         ("B4", parts["b4"][keep]), ("Output", parts["out"][keep][:, None, :])):
        _, power = spectra(signal, fs)
        p1, p2 = p_at(power, f0k), p_at(power, 2 * f0k)
        with np.errstate(divide="ignore", invalid="ignore"):
            ratios[name] = np.log10(np.clip(p2, 1e-20, None) / np.clip(p1, 1e-20, None))
        if name == "B4":
            wide = ((freqs >= 0.05) & (freqs <= 2.0))
            row["Fundamental_Share_B4"] = float(np.median(p1 / np.clip(power[:, wide].sum(axis=1), 1e-20, None)))
            row["Harmonic_Dominant_B4"] = float(np.mean(p2 > p1))
        if name in ("B4", "Output", "Target"):
            normed = power / np.clip(power[:, (freqs >= 0.05) & (freqs <= 2.0)].sum(axis=1, keepdims=True), 1e-20, None)
            curves[name] = np.mean([np.interp(F0_GRID * f, freqs, p) for f, p in zip(f0k, normed)], axis=0)
    for name in ("Stem", "B4", "Output"):
        row[f"Harmonic_LogRatio_{name}"] = float(np.median(ratios[name]))
    row["Harmonic_LogRatio_Target"] = float(np.median(ratios["Target"]))
    return row, curves


# F ---------------------------------------------------------------------------

def phase_probe(features: np.ndarray, phase: np.ndarray, windows: np.ndarray) -> Tuple[float, float]:
    """Blocked cross-validated ridge regression of cos / sin of the phase."""
    x = standardize(features)
    y = np.column_stack([np.cos(phase), np.sin(phase)])
    blocks = np.array_split(np.unique(windows), PROBE_FOLDS)
    pred = np.full_like(y, np.nan)
    for block in blocks:
        test = np.isin(windows, block)
        if test.all() or not test.any():
            continue
        xt, yt = x[~test], y[~test]
        mx, my = xt.mean(axis=0), yt.mean(axis=0)
        a = (xt - mx).T @ (xt - mx) + RIDGE_ALPHA * np.eye(x.shape[1])
        w = np.linalg.solve(a, (xt - mx).T @ (yt - my))
        pred[test] = (x[test] - mx) @ w + my
    ok = np.isfinite(pred).all(axis=1)
    if ok.sum() < 10:
        return math.nan, math.nan
    sse = ((pred[ok] - y[ok]) ** 2).sum(axis=0)
    sst = ((y[ok] - y[ok].mean(axis=0)) ** 2).sum(axis=0)
    r2 = float(np.mean(1.0 - sse / np.clip(sst, 1e-12, None)))
    error = float(np.mean(np.abs(wrap(np.arctan2(pred[ok, 1], pred[ok, 0]) - phase[ok]))))
    return r2, error


def knn_phase_error(features: np.ndarray, phase: np.ndarray, windows: np.ndarray) -> float:
    if len(np.unique(windows)) < 2:
        return math.nan
    rng = np.random.default_rng(0)
    pick = rng.choice(len(features), size=min(KNN_POINTS, len(features)), replace=False)
    x = standardize(features[pick])
    ph, win = phase[pick], windows[pick]
    sq = (x * x).sum(axis=1)
    dist = sq[:, None] + sq[None, :] - 2 * x @ x.T
    dist[win[:, None] == win[None, :]] = np.inf
    k = min(KNN_K, max(1, int(np.isfinite(dist).sum(axis=1).min())))
    nearest = np.argpartition(dist, k - 1, axis=1)[:, :k]
    return float(np.mean(np.abs(wrap(ph[nearest] - ph[:, None]))))


def tsne_map(features: np.ndarray) -> Optional[np.ndarray]:
    """2-D t-SNE map, or None (scikit-learn missing or the map failed; the run goes on)."""
    try:
        from sklearn.manifold import TSNE
    except ImportError:
        print("    t-SNE skipped (scikit-learn not installed)")
        return None
    n = len(features)
    perplexity = min(TSNE_PERPLEXITY, max(5.0, (n - 1) / 3.0))
    # learning rate as scikit-learn's "auto" (max(n / early exaggeration 12 / 4, 50)), given as a number so that
    # versions before 1.0 (which do not know "auto") work too
    learning_rate = max(n / 12.0 / 4.0, 50.0)
    try:
        return TSNE(n_components=2, perplexity=perplexity, init="pca", learning_rate=learning_rate,
                    random_state=0).fit_transform(standardize(features))
    except Exception as exc:
        print(f"    t-SNE skipped ({type(exc).__name__}: {exc})")
        return None


# ---------------------------------------------------------------------------
# Folds
# ---------------------------------------------------------------------------

def checkpoint(root: Path, backbone, subject: str, member: int) -> Optional[Path]:
    path = root / "Members" / f"m{member}" / "Models" / f"{backbone.safe_tag(subject)}_best.pth"
    return path if path.exists() and backbone.member_paths(root, subject, member)[0].exists() else None


def x_261013_checkpoint(backbone, root: Path, subject: str, weight_decay: float) -> Tuple[Optional[Path], str]:
    path = checkpoint(root, backbone, subject, 0)
    if path is None:
        return None, "no model without stem fusion (m0) in the noise_v1 results"
    model_key = str(json.loads(backbone.member_paths(root, subject, 0)[0].read_text(encoding="utf-8")).get("Model", ""))
    if not model_key.endswith("_wd{:g}_subjval".format(weight_decay)):
        return None, f"model without stem fusion not trained with the selected weight decay ({weight_decay:g})"
    return path, ""


def member_rates(root: Path, backbone, subject: str, member: int) -> Optional[pd.Series]:
    path = backbone.member_paths(root, subject, member)[1]
    if not path.exists():
        return None
    frame = pd.read_csv(path, encoding="utf-8-sig")
    if "Pred_Wave_FFT_RR" not in frame:
        return None
    return frame.set_index(["Segment_ID", "RR_Window_Index"])["Pred_Wave_FFT_RR"]


def rate_consistency(rates: List[pd.Series]) -> dict:
    table = pd.concat(rates, axis=1).dropna()
    if table.empty:
        return {"Rate_Agreement_1bpm": math.nan, "Harmonic_Disagreement": math.nan, "N_Minutes_Members": 0}
    values = table.to_numpy(float)
    spread = values.max(axis=1) - values.min(axis=1)
    with np.errstate(divide="ignore", invalid="ignore"):
        ratio = values.max(axis=1) / values.min(axis=1)
    return {"Rate_Agreement_1bpm": float(np.mean(spread <= 1.0)),
            "Harmonic_Disagreement": float(np.mean((ratio >= 1.75) & (ratio <= 2.25))),
            "N_Minutes_Members": int(len(values))}


def test_arrays(backbone, subject_data: dict, subject: str) -> Tuple[torch.Tensor, torch.Tensor]:
    dataset = backbone.SubjectWindowDataset(subject_data, [subject], "test", False)
    xs, ys = [], []
    for x, y, _, _ in DataLoader(dataset, batch_size=BATCH, shuffle=False):
        xs.append(x)
        ys.append(y)
    return torch.cat(xs), torch.cat(ys)


def train_batches(backbone, source: str, dataset_key: str, subject_data: dict, subject: str,
                  n_windows: int, val_subjects: Optional[Sequence[str]]) -> list:
    """Training windows of the fold (validation windows excluded as in training;
    261013: the validation subjects stored with the weight decay selection)."""
    others = [s for s in sorted(subject_data) if s != subject]
    if source == "261013":
        held_out = set(str(s) for s in (val_subjects if val_subjects is not None
                                        else backbone.validation_subjects(dataset_key, subject, others)))
        others = [s for s in others if s not in held_out]
    dataset = backbone.SubjectWindowDataset(subject_data, others, "train", False)
    pick = np.unique(np.linspace(0, len(dataset) - 1, min(len(dataset), n_windows)).round().astype(int))
    batches = []
    for x, y, _, _ in DataLoader(Subset(dataset, pick.tolist()), batch_size=BATCH, shuffle=False):
        batches.append((x, y))
    return batches


def init_model(backbone, options, seed: int, device):
    backbone.set_seed(seed)
    return options.build().to(device)


def o_mae(root: Path, backbone, subject: str) -> float:
    summary = backbone.read_json(backbone.member_paths(root, subject, 0)[0]) or {}
    value = summary.get("RR_MAE_Wave_FFT")
    return float(value) if value is not None else math.nan


def tsne_subjects(args, label: str, maes: Dict[str, float]) -> List[str]:
    wanted = set()
    if args.tsne_subjects:
        if [s.lower() for s in args.tsne_subjects] == ["all"]:
            return sorted(maes)
        return [s.split(":", 1)[1] for s in args.tsne_subjects if s.split(":", 1)[0] == label and s.split(":", 1)[1] in maes]
    finite = {s: v for s, v in maes.items() if np.isfinite(v)}
    if finite and args.tsne_per_dataset > 0:
        ordered = sorted(finite, key=finite.get)
        for q in np.linspace(0.25, 0.75, args.tsne_per_dataset) if args.tsne_per_dataset > 1 else [0.5]:
            wanted.add(ordered[int(round(q * (len(ordered) - 1)))])
    wanted |= {s.split(":", 1)[1] for s in args.case_subjects if s.split(":", 1)[0] == label and s.split(":", 1)[1] in maes}
    return sorted(wanted)


def analyse_source(source: str, backbone, x_backbone, results_root: Path, noise_root: Path, args, device,
                   store: dict) -> None:
    spec = next(s for s in backbone.WINDOW_SPECS if s.name == WINDOW)
    fs = float(backbone.TARGET_FS)
    loss_fn = backbone.WaveformLoss()
    x_options = x_backbone.VARIANTS[X_VARIANT].options
    for dataset_key in SOURCE_DATASETS[source]:
        if dataset_key not in args.datasets:
            continue
        label = backbone.DATASET_CONFIGS[dataset_key]["label"]
        if source == "261007":
            root_o = backbone.condition_root(results_root, O_VARIANT, spec, dataset_key)
            root_x = backbone.condition_root(results_root, X_VARIANT, spec, dataset_key)
            available = (root_o / "Members").exists() and (root_x / "Members").exists()
        else:
            root_x = backbone.condition_root(noise_root, X_VARIANT, spec, dataset_key)
            available = (results_root / "WD_Selection" / f"{spec.order:02d}_{spec.name}" / label).exists() \
                and (root_x / "Members").exists()
        if not available:
            print(f"{source} {label}: models not found, skipped")
            store["coverage"].append({"Source": source, "Dataset": label, "N_Subjects": 0, "N_Analysed": 0,
                                      "Note": "models not found"})
            continue
        try:
            subject_data = backbone.load_dataset(dataset_key, spec)
        except Exception as exc:
            print(f"{source} {label}: data not read ({exc})")
            continue
        subjects = sorted(subject_data)
        if args.subjects:
            subjects = [s for s in subjects if s in set(args.subjects)]
        # model folders of every subject
        folds = {}
        for subject in subjects:
            if source == "261007":
                o_root, o_options, decay, val_subjects = root_o, backbone.VARIANTS[O_VARIANT].options, None, None
                x_paths = [checkpoint(root_x, backbone, subject, m) for m in range(MEMBERS)]
                reason = "" if x_paths[0] is not None else "no model without stem fusion (m0)"
            else:
                found = backbone.stored_selection(results_root, list(backbone.VARIANTS.values()), spec,
                                                  dataset_key, subject)
                if found is None:
                    store["skipped"].append({"Source": source, "Dataset": label, "Subject": subject,
                                             "Reason": "no weight decay selection"})
                    continue
                record, variant = found
                o_root = backbone.condition_root(results_root, variant.name, spec, dataset_key)
                o_options, decay = variant.options, float(record["Selected_Weight_Decay"])
                val_subjects = record.get("Validation_Subjects")
                path, reason = x_261013_checkpoint(backbone, root_x, subject, decay)
                x_paths = [path, None, None]
            o_paths = [checkpoint(o_root, backbone, subject, m) for m in range(MEMBERS)]
            if o_paths[0] is None or x_paths[0] is None:
                store["skipped"].append({"Source": source, "Dataset": label, "Subject": subject,
                                         "Reason": reason or "no full model (m0)"})
                continue
            folds[subject] = (o_root, o_options, o_paths, x_paths, decay, val_subjects)
        tsne_for = set() if args.no_tsne else set(tsne_subjects(
            args, label, {s: o_mae(f[0], backbone, s) for s, f in folds.items()}))
        print(f"{source} {label}: {len(folds)} of {len(subjects)} subjects with O and X (m0)"
              + (f"; t-SNE: {' '.join(sorted(tsne_for))}" if tsne_for else ""), flush=True)
        for subject, (o_root, o_options, o_paths, x_paths, decay, val_subjects) in folds.items():
            key = {"Source": source, "Dataset": label, "Subject": subject}
            x_test, y_test = test_arrays(backbone, subject_data, subject)
            batches = train_batches(backbone, source, dataset_key, subject_data, subject, args.grad_windows,
                                    val_subjects)
            seed = backbone.member_seed(spec, dataset_key, subject, 0)
            parts_m0, members = {}, {}
            for model_key, options, paths in (("O", o_options, o_paths), ("X", x_options, x_paths)):
                row = {**key, "Model": model_key, "Weight_Decay": decay}
                # A: gradients at init and with the trained weights
                model = init_model(backbone, options, seed, device)
                for name, value in gradient_stats(model, batches, loss_fn, device).items():
                    row[f"{name}_Init"] = value
                if model_key == "O":
                    b_init, _ = fusion_composition(model, run_test(model, x_test, device)["stem"])
                model.load_state_dict(torch.load(paths[0], map_location=device))
                for name, value in gradient_stats(model, batches, loss_fn, device).items():
                    row[f"{name}_Final"] = value
                store["A"].append(row)
                parts = run_test(model, x_test, device)
                parts_m0[model_key] = parts
                # B: scale composition (O)
                if model_key == "O":
                    b_final, channels = fusion_composition(model, parts["stem"])
                    store["B"].append({**key, "State": "init", **b_init})
                    store["B"].append({**key, "State": "final", **b_final})
                    store["B_channels"] += [{**key, **c} for c in channels]
                # C: branch reliance
                store["C"].append({**key, "Model": model_key,
                                   **branch_reliance(model, parts["stem"], parts["out"], y_test, loss_fn, device)})
                # D: harmonics
                d_row, curves = harmonics(y_test[:, 0].numpy(), parts, fs, RATE_BAND_BPM[dataset_key])
                store["D"].append({**key, "Model": model_key, **d_row})
                for name, curve in curves.items():
                    store["curves"].setdefault((source, label, subject, model_key, name), curve)
                # F: respiratory phase
                points, windows = channel_points(parts["b4"], fs)
                stem_points, _ = channel_points(parts["fused"], fs)
                phase = target_phase(y_test[:, 0].numpy(), fs)
                r2, error = phase_probe(points, phase, windows)
                r2_stem, _ = phase_probe(stem_points, phase, windows)
                store["F"].append({**key, "Model": model_key, "Phase_Probe_R2_B4": r2, "Phase_Probe_Error_B4": error,
                                   "Phase_Probe_R2_Stem": r2_stem,
                                   "Phase_kNN_Error_B4": knn_phase_error(points, phase, windows),
                                   "N_Points": int(len(points))})
                if subject in tsne_for:
                    pick = np.random.default_rng(1).choice(len(points), size=min(TSNE_POINTS, len(points)),
                                                           replace=False)
                    store["tsne"][(source, label, subject, model_key)] = (
                        tsne_map(points[pick]), phase[pick], r2, error)
                # E: members m0-m2 (261007)
                if source == "261007" and all(p is not None for p in paths):
                    members[model_key] = [parts]
                    for path in paths[1:]:
                        model.load_state_dict(torch.load(path, map_location=device))
                        members[model_key].append(run_test(model, x_test, device))
                del model
            if source == "261007" and len(members) == 2:
                for model_key, root in (("O", o_root), ("X", root_x)):
                    runs = members[model_key]
                    pairs = [(a, b) for a in range(len(runs)) for b in range(a + 1, len(runs))]
                    rates = [member_rates(root, backbone, subject, m) for m in range(MEMBERS)]
                    e_row = {**key, "Model": model_key,
                             "Output_Agreement_r": float(np.nanmean([np.nanmean(pearson_rows(runs[a]["out"], runs[b]["out"]))
                                                                     for a, b in pairs])),
                             "CKA_Stem": float(np.mean([linear_cka(*(as_points(r["fused"]) for r in (runs[a], runs[b])))
                                                        for a, b in pairs])),
                             "CKA_B4": float(np.mean([linear_cka(*(as_points(r["b4"]) for r in (runs[a], runs[b])))
                                                      for a, b in pairs]))}
                    e_row.update(rate_consistency([r for r in rates if r is not None]) if all(r is not None for r in rates)
                                 else {"Rate_Agreement_1bpm": math.nan, "Harmonic_Disagreement": math.nan})
                    store["E"].append(e_row)
            print(f"  {subject} done", flush=True)
            del parts_m0, members, batches
        store["coverage"].append({"Source": source, "Dataset": label, "N_Subjects": len(subjects),
                                  "N_Analysed": len(folds), "Note": ""})
        del subject_data
        backbone.release_memory(device)


# ---------------------------------------------------------------------------
# Statistics
# ---------------------------------------------------------------------------

def paired_tests(frame: pd.DataFrame, source: str, dataset: str) -> List[dict]:
    rows = []
    for metric, (section, direction) in METRICS.items():
        if metric not in frame:
            continue
        wide = frame.pivot_table(index="Subject", columns="Model", values=metric)
        if not {"O", "X"} <= set(wide.columns):
            continue
        wide = wide[["O", "X"]].dropna()
        if len(wide) < 2:
            continue
        o, x = wide["O"].to_numpy(), wide["X"].to_numpy()
        diff = o - x
        nz = diff[diff != 0]
        p = math.nan
        if len(nz) >= 1:
            try:
                p = float(stats.wilcoxon(o, x).pvalue)
            except ValueError:
                p = math.nan
        ranks = stats.rankdata(np.abs(nz)) if len(nz) else np.array([])
        rb = float((ranks[nz > 0].sum() - ranks[nz < 0].sum()) / ranks.sum()) if len(nz) else 0.0
        rows.append({"Source": source, "Dataset": dataset, "Section": section, "Metric": metric,
                     "Favourable_If": f"{direction} in O" if direction else "descriptive", "N": len(wide),
                     "O_Median": np.median(o), "O_Q1": np.percentile(o, 25), "O_Q3": np.percentile(o, 75),
                     "X_Median": np.median(x), "X_Q1": np.percentile(x, 25), "X_Q3": np.percentile(x, 75),
                     "O_Mean": o.mean(), "X_Mean": x.mean(), "N_O_Higher": int((diff > 0).sum()),
                     "N_X_Higher": int((diff < 0).sum()), "Rank_Biserial_O_minus_X": rb, "Wilcoxon_p": p,
                     "_direction": direction})
    if rows:
        ps = np.array([r["Wilcoxon_p"] if r["_direction"] else np.nan for r in rows])
        fdr = bh(ps)
        for r, q in zip(rows, fdr):
            r["Wilcoxon_FDR_p"] = q
            if not r["_direction"]:
                r["Result"] = "descriptive"
            elif np.isfinite(q) and q < 0.05:
                higher = r["Rank_Biserial_O_minus_X"] > 0
                r["Result"] = "favours O" if higher == (r["_direction"] == "higher") else "favours X"
            else:
                r["Result"] = "n.s."
            del r["_direction"]
    return rows


def bh(p: np.ndarray) -> np.ndarray:
    out = np.full(len(p), np.nan)
    ok = np.isfinite(p)
    if not ok.any():
        return out
    q = p[ok]
    order = np.argsort(q)
    ranked = q[order] * len(q) / (np.arange(len(q)) + 1)
    ranked = np.minimum.accumulate(ranked[::-1])[::-1]
    values = np.empty(len(q))
    values[order] = np.clip(ranked, 0, 1)
    out[ok] = values
    return out


def training_effect(b: pd.DataFrame) -> pd.DataFrame:
    rows = []
    metrics = [c for c in b.columns if c.startswith(("Mix_Entropy", "Mixed_Channels", "Share_", "Weight_Mix",
                                                     "Weight_Share"))]
    for (source, dataset), d in list(b.groupby(["Source", "Dataset"])) + \
            [((s, ALL_DATASETS), d) for s, d in b.groupby("Source")]:
        for metric in metrics:
            wide = d.pivot_table(index=["Dataset", "Subject"], columns="State", values=metric)
            if not {"init", "final"} <= set(wide.columns):
                continue
            wide = wide.dropna()
            if len(wide) < 2:
                continue
            fin, ini = wide["final"].to_numpy(), wide["init"].to_numpy()
            try:
                p = float(stats.wilcoxon(fin, ini).pvalue) if np.any(fin != ini) else 1.0
            except ValueError:
                p = math.nan
            rows.append({"Source": source, "Dataset": dataset, "Metric": metric, "N": len(wide),
                         "Final_Median": np.median(fin), "Init_Median": np.median(ini),
                         "N_Final_Higher": int((fin > ini).sum()), "Wilcoxon_p": p})
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------

def box_figure(path: Path, frame: pd.DataFrame, tests: pd.DataFrame, metrics: Sequence[str], title: str) -> None:
    metrics = [m for m in metrics if m in frame]
    datasets = list(dict.fromkeys(frame.Dataset))
    if len(datasets) > 1:
        datasets.append(ALL_DATASETS)
    if not metrics or frame.empty:
        return
    fig, axes = plt.subplots(len(datasets), len(metrics), figsize=(3.3 * len(metrics), 2.7 * len(datasets)),
                             squeeze=False)
    for i, dataset in enumerate(datasets):
        d = frame if dataset == ALL_DATASETS else frame[frame.Dataset == dataset]
        for j, metric in enumerate(metrics):
            ax = axes[i, j]
            values = [d[d.Model == m][metric].dropna().to_numpy() for m in ("X", "O")]
            ax.boxplot(values, widths=0.55, patch_artist=True, showfliers=False,
                       boxprops={"facecolor": "white"}, medianprops={"color": "#333333"})
            for k, (m, v) in enumerate(zip(("X", "O"), values)):
                ax.scatter(np.full(len(v), k + 1) + np.random.default_rng(k).uniform(-0.12, 0.12, len(v)), v,
                           s=9, color=MODEL_COLORS[m], alpha=0.8, zorder=3)
            ax.set_xticks([1, 2])
            ax.set_xticklabels(["X", "O"])
            t = tests[(tests.Dataset == dataset) & (tests.Metric == metric)]
            note = ""
            if len(t):
                r = t.iloc[0]
                note = f"p={r.Wilcoxon_p:.2g}, FDR={r.Wilcoxon_FDR_p:.2g}" if np.isfinite(r.Wilcoxon_p) else ""
                note += f"\n{r.Result}"
            ax.set_title(f"{metric}\n{note}" if i == 0 else note, fontsize=7.5)
            if j == 0:
                ax.set_ylabel(dataset, fontsize=9)
            ax.tick_params(labelsize=7.5)
            ax.grid(axis="y", color="#e4e3df", lw=0.6)
    fig.suptitle(title, fontsize=10)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=140)
    plt.close(fig)


def fusion_figure(path: Path, channels: pd.DataFrame, title: str) -> None:
    datasets = list(dict.fromkeys(channels.Dataset))
    if not datasets:
        return
    fig, axes = plt.subplots(1, len(datasets), figsize=(max(7.0, 3.6 * len(datasets)), 5.0), squeeze=False)
    shares = [c for c in channels.columns if c.startswith("Share_")]
    for ax, dataset in zip(axes[0], datasets):
        d = channels[channels.Dataset == dataset].groupby("Channel")[shares].mean()
        d = d.sort_values(shares[-1])
        left = np.zeros(len(d))
        for name, color in zip(shares, ("#f2a33a", "#59a14f", "#4c78a8")):
            ax.barh(np.arange(len(d)), d[name], left=left, color=color, label=name.replace("Share_", ""))
            left += d[name].to_numpy()
        ax.set_title(dataset, fontsize=9)
        ax.set_xlabel("share of the fused channel (mean over subjects)", fontsize=8)
        ax.set_yticks([])
        ax.set_xlim(0, 1)
    axes[0, 0].set_ylabel("fused channels (sorted)")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, fontsize=8, loc="lower center", ncol=3, frameon=False)
    fig.suptitle(title, fontsize=10)
    fig.tight_layout(rect=(0, 0.06, 1, 1))
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=140)
    plt.close(fig)


def spectra_figure(path: Path, curves: dict, source: str, panels: List[Tuple[str, Optional[str]]], title: str) -> None:
    if not panels:
        return
    fig, axes = plt.subplots(len(panels), 2, figsize=(10, 2.6 * len(panels)), squeeze=False)
    for i, (dataset, subject) in enumerate(panels):
        for j, name in enumerate(("B4", "Output")):
            ax = axes[i, j]
            for model in ("X", "O"):
                stack = [c for (s, d, subj, m, n), c in curves.items()
                         if s == source and d == dataset and m == model and n == name and (subject is None or subj == subject)]
                if stack:
                    ax.plot(F0_GRID, np.mean(stack, axis=0), color=MODEL_COLORS[model], lw=1.6,
                            label=f"{MODEL_NAMES[model]} (n={len(stack)})")
            target = [c for (s, d, subj, m, n), c in curves.items()
                      if s == source and d == dataset and m == "O" and n == "Target" and (subject is None or subj == subject)]
            if target:
                ax.plot(F0_GRID, np.mean(target, axis=0), color="#111111", lw=1.0, ls="--", label="target")
            for v in (1, 2):
                ax.axvline(v, color="#bbbbbb", lw=0.8, zorder=0)
            ax.set_title(f"{dataset}{' ' + subject if subject else ''}: {'encoder output (B4)' if name == 'B4' else 'model output'}",
                         fontsize=8.5)
            ax.set_xlabel("frequency / f0 of the target", fontsize=8)
            ax.tick_params(labelsize=7.5)
            ax.legend(fontsize=7)
    fig.suptitle(title, fontsize=10)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=140)
    plt.close(fig)


def tsne_figures(folder: Path, tsne: dict) -> None:
    groups = {}
    for (source, dataset, subject, model), value in tsne.items():
        groups.setdefault((source, dataset), {}).setdefault(subject, {})[model] = value
    for (source, dataset), subjects in groups.items():
        folder.mkdir(parents=True, exist_ok=True)
        with PdfPages(folder / f"tSNE_{dataset}.pdf") as pdf:
            for subject, models in sorted(subjects.items()):
                if any(models.get(m, (None,))[0] is None for m in ("X", "O")):
                    continue
                fig, axes = plt.subplots(1, 2, figsize=(10, 4.6))
                for ax, model in zip(axes, ("X", "O")):
                    xy, phase, r2, error = models[model]
                    sc = ax.scatter(xy[:, 0], xy[:, 1], c=phase, cmap="twilight", vmin=-np.pi, vmax=np.pi, s=5)
                    ax.set_title(f"{MODEL_NAMES[model]}: phase probe R2 {r2:.2f}, error {error:.2f} rad", fontsize=9)
                    ax.set_xticks([])
                    ax.set_yticks([])
                fig.colorbar(sc, ax=axes, label="respiratory phase of the target (rad)", shrink=0.8)
                fig.suptitle(f"{source} {dataset} {subject}: t-SNE of the encoder output (B4) time points "
                             f"(held-out subject, member m0)", fontsize=10)
                png = folder / f"tSNE_{dataset}" / f"{subject}.png"
                png.parent.mkdir(parents=True, exist_ok=True)
                fig.savefig(png, dpi=130)
                pdf.savefig(fig)
                plt.close(fig)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def notes_table(source: str, args) -> pd.DataFrame:
    return pd.DataFrame({"Item": ["Source", "O / X", "Pairing", "Init", "Favourable_If", "A", "B", "C", "D", "E", "F",
                                  "t-SNE"],
                         "Meaning": [
                             f"{source}: " + ("CapnoBase, BIDMC, STEAM2 as 261007 read and trained them"
                                              if source == "261007" else "BIDMC, STEAM2 as 261013 read and trained them"),
                             "Stem Fusion O = full model; X = no_stem_fusion (concatenated branches into the depthwise "
                             "encoder: the time scales stay separate until the decoder)",
                             "member m0 of O and X (E: m0-m2)" if source == "261007" else
                             "O = full model of the weight decay selected in the fold (m0); X = "
                             "261028_Stem_ablation_noise_v1 m0 trained with that weight decay",
                             "initial weights drawn with the member's seed as in training (no training step run)",
                             "direction fixed before the run; Result from the BH-FDR p (< 0.05)",
                             f"gradients on {args.grad_windows} training windows of the fold (eval mode, no jitter): "
                             "branch shares and balance (entropy / log 3), cosine coherence of the batch gradients of "
                             "the stem, stem / decoder+head gradient norm",
                             "O only: shares of k32 / k64 / k128 in every fused channel (|W| x branch SD), mixing "
                             f"entropy, channels with >= 2 branches >= {MIXED_SHARE:.0%}; final vs init (B_Training_Effect)",
                             "one branch replaced by its time mean per window: loss increase and output change; shares, "
                             "balance, largest share",
                             f"log10(P(2 f0) / P(f0)), +-{HARMONIC_HALF_WIDTH_HZ} Hz, f0 = target peak in the rate band "
                             f"(f0 >= {HARMONIC_MIN_F0_HZ} Hz); the target's own ratio is listed",
                             "agreement of members m0-m2 (output r, linear CKA, rates within 1 breath/min, one member at "
                             "~2x another)" if source == "261007" else "not available (one model without stem fusion per fold)",
                             "ridge probe of cos / sin of the target phase (blocked 5-fold CV), mean |phase error|, kNN "
                             "phase difference to time points of other windows",
                             "per subject (every fold has its own network); coloured by the target phase"]})


def save_source(source: str, store: dict, args, folder: Path) -> None:
    """Statistics, tables and figures of one source in its own folder."""
    tables = {k: pd.DataFrame(store[k]) for k in ("A", "B", "B_channels", "C", "D", "E", "F")}
    coverage = pd.concat([pd.DataFrame(store["coverage"]), pd.DataFrame(store["skipped"])], ignore_index=True)
    folder.mkdir(parents=True, exist_ok=True)
    if not coverage.empty:
        coverage.to_csv(folder / "Coverage.csv", index=False, encoding="utf-8-sig")
    if tables["A"].empty:
        print(f"{source}: no subject with both models was found (check the result folders).")
        return
    per_subject = None
    for name in ("A", "C", "D", "E", "F"):
        t = tables[name]
        if t.empty:
            continue
        keep = [c for c in t.columns if c in METRICS or c in ("Source", "Dataset", "Subject", "Model")]
        per_subject = t[keep] if per_subject is None else per_subject.merge(
            t[keep], on=["Source", "Dataset", "Subject", "Model"], how="outer")
    tests = []
    for dataset, e in per_subject.groupby("Dataset", sort=False):
        tests += paired_tests(e, source, dataset)
    if per_subject.Dataset.nunique() > 1:
        tests += paired_tests(per_subject.assign(Subject=per_subject.Dataset + ":" + per_subject.Subject),
                              source, ALL_DATASETS)
    tests = pd.DataFrame(tests)
    effect = training_effect(tables["B"]) if not tables["B"].empty else pd.DataFrame()

    sections = {s: folder / name for s, name in SECTION_FOLDERS.items()}
    tables_by_section = {"A": {"A_Gradients": tables["A"]},
                         "B": {"B_Fusion": tables["B"], "B_Fusion_Channels": tables["B_channels"],
                               "B_Training_Effect": effect},
                         "C": {"C_Branch_Reliance": tables["C"]}, "D": {"D_Harmonics": tables["D"]},
                         "E": {"E_Consistency": tables["E"]}, "F": {"F_Phase": tables["F"]}}
    for section, frames in tables_by_section.items():
        for name, frame in frames.items():
            if not frame.empty:
                sections[section].mkdir(parents=True, exist_ok=True)
                frame.to_csv(sections[section] / f"{name}.csv", index=False, encoding="utf-8-sig")
    for section in ("A", "C", "D", "E", "F"):
        frame = tables[section]
        if not frame.empty:
            box_figure(sections[section] / f"{SECTION_FOLDERS[section]}.png", frame, tests, FIGURE_METRICS[section],
                       f"{source}: {SECTION_FOLDERS[section]} (X left, O right; paired by subject, "
                       + ("members m0-m2)" if section == "E" else "member m0)"))
    if not tables["B_channels"].empty:
        fusion_figure(sections["B"] / "B_Fusion_Shares.png", tables["B_channels"],
                      f"{source}: k32 / k64 / k128 shares of the fused channels\n(O, |final weight| x branch SD)")
    if not tables["D"].empty:
        datasets = list(dict.fromkeys(tables["D"].Dataset))
        spectra_figure(sections["D"] / "D_Spectra.png", store["curves"], source, [(d, None) for d in datasets],
                       f"{source}: mean spectrum on the f / f0 axis (test windows, member m0)")
        cases = [(c.split(":", 1)[0], c.split(":", 1)[1]) for c in args.case_subjects
                 if any(k[1] == c.split(":", 1)[0] and k[2] == c.split(":", 1)[1] for k in store["curves"])]
        spectra_figure(sections["D"] / "D_Spectra_Cases.png", store["curves"], source, cases,
                       f"{source}: case subjects")
        np.savez_compressed(sections["D"] / "harmonic_curves.npz", grid=F0_GRID,
                            **{"|".join(k): v for k, v in store["curves"].items()})
    if store["tsne"]:
        tsne_figures(sections["F"], store["tsne"])

    if not tests.empty:
        tests.to_csv(folder / "Tests.csv", index=False, encoding="utf-8-sig")
    sheets = {"Tests": tests, "B_Training_Effect": effect, "A_Gradients": tables["A"], "B_Fusion": tables["B"],
              "B_Fusion_Channels": tables["B_channels"], "C_Branch_Reliance": tables["C"],
              "D_Harmonics": tables["D"], "E_Consistency": tables["E"], "F_Phase": tables["F"],
              "Coverage": coverage, "Notes": notes_table(source, args)}
    with pd.ExcelWriter(folder / f"{source}_StemFusion_Feature_Interpretation.xlsx") as writer:
        for name, frame in sheets.items():
            if not frame.empty:
                frame.round(5).to_excel(writer, sheet_name=name[:31], index=False)
    print(f"\n{source}: O vs X (Result by BH-FDR, direction fixed beforehand)")
    for r in tests.to_dict("records"):
        print(f"  {r['Dataset']:<12s} {r['Metric']:<28s} O {r['O_Median']:.4g} vs X {r['X_Median']:.4g} "
              f"(O higher {r['N_O_Higher']}/{r['N']}, p={r['Wilcoxon_p']:.2g}) {r['Result']}")
    print(f"{source}: saved under {folder}", flush=True)


def main(args: argparse.Namespace) -> None:
    roots = {"261007": Path(args.results_root_261007), "261013": Path(args.results_root_261013)}
    paths = {"261007": find_backbone("261007", args.backbone_261007, roots["261007"]),
             "261013": find_backbone("261013", args.backbone_261013, roots["261013"])}
    backbones = {source: load_backbone(path, f"rrwavenet_{source}_interpretation") for source, path in paths.items()}
    for module in backbones.values():
        configs = module.DATASET_CONFIGS
        if args.capnobase_path and "capnobase" in configs:
            configs["capnobase"]["data_path"] = str(args.capnobase_path)
        if args.bidmc_path:
            configs["bidmc"]["data_path"] = str(args.bidmc_path)
        if args.steam2_ppg_dir:
            configs["steam2"]["ppg_dir"] = str(args.steam2_ppg_dir)
        if args.steam2_rsp_dir:
            configs["steam2"]["rsp_dir"] = str(args.steam2_rsp_dir)
    if args.threads:
        torch.set_num_threads(int(args.threads))
    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() and not args.force_cpu else "cpu")
    out = Path(args.output_root)
    out.mkdir(parents=True, exist_ok=True)
    print(f"Scripts: 261007 {paths['261007']}\n         261013 {paths['261013']}\n"
          f"Results: 261007 {roots['261007']}\n         261013 {roots['261013']} "
          f"(without stem fusion: {args.results_root_noise})\n"
          f"Output:  {out / '261007'}, {out / '261013'}\ndevice:  {device}", flush=True)
    for source in args.sources:
        store = {k: [] for k in ("A", "B", "B_channels", "C", "D", "E", "F", "coverage", "skipped")}
        store["curves"], store["tsne"] = {}, {}
        analyse_source(source, backbones[source], backbones["261007"], roots[source], Path(args.results_root_noise),
                       args, device, store)
        save_source(source, store, args, out / source)


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--backbone-261007", type=Path, default=None,
                        help="261007_RRWaveNet_Ensemble_NoEvent.py (default: looked for, see the docstring).")
    parser.add_argument("--backbone-261013", type=Path, default=None,
                        help="261013_RRWaveNet_Adult_SubjectVal_WD.py (default: looked for).")
    parser.add_argument("--results-root-261007", type=Path, default=RESULTS_ROOT_261007)
    parser.add_argument("--results-root-261013", type=Path, default=RESULTS_ROOT_261013)
    parser.add_argument("--results-root-noise", type=Path, default=RESULTS_ROOT_NOISE,
                        help="261028_Stem_ablation_noise_v1 results (261013 models without stem fusion).")
    parser.add_argument("--output-root", type=Path, default=PROJECT_ROOT,
                        help="Every result goes here, in 261007\\ and 261013\\.")
    parser.add_argument("--sources", nargs="+", default=["261007", "261013"], choices=["261007", "261013"])
    parser.add_argument("--datasets", nargs="+", default=["capnobase", "bidmc", "steam2"],
                        choices=["capnobase", "bidmc", "steam2"])
    parser.add_argument("--subjects", nargs="+", default=None, help="Only these subjects (quick tests).")
    parser.add_argument("--grad-windows", type=int, default=GRAD_WINDOWS,
                        help="Training windows of the fold used for the gradients (A).")
    parser.add_argument("--case-subjects", nargs="*", default=list(CASE_SUBJECTS),
                        help="Dataset:Subject drawn in D (spectra) and F (t-SNE).")
    parser.add_argument("--tsne-per-dataset", type=int, default=TSNE_PER_DATASET,
                        help="Subjects per dataset with a t-SNE map (quantiles of the test MAE of O m0).")
    parser.add_argument("--tsne-subjects", nargs="*", default=None,
                        help="Dataset:Subject list (or 'all') instead of the default choice.")
    parser.add_argument("--no-tsne", action="store_true")
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
