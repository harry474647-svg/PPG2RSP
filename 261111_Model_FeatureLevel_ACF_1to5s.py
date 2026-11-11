r"""261111_Model_FeatureLevel_ACF_1to5s: 261106_Model_FeatureLevel with the autocorrelation (ACF)
analysis on lags 1-5 s and the ACF figures drawn up to a lag of 10 s.

Changes from 261106_Model_FeatureLevel (everything else is the same: models,
windows, forward pass, PC1, channel |r|, PSD, stem fusion before / after,
statistics):
  * ACF similarity to the target (ACF r): Pearson r over lags 1-5 s
    (261106: 2-10 s). 1 s keeps the first trough of the breathing ACF (half a
    breathing period); 5 s is half of the 10 s window, so at least half of
    the sample pairs lie in the same window, and it stays inside the encoder
    receptive field (6.58 s at B4).
  * ACF computed and drawn up to a lag of 10 s (261106: 12 s).
  * Its own name and results folder.
The ACF itself is computed as in 261106: PC1 (and the target) of contiguous
10 s test windows joined into runs of at least 24 s, mean removed per run,
normalized by the lag-0 value, runs averaged by length.

The 261106 description follows (lags there read 1-5 s and 10 s).

261106_Model_FeatureLevel: feature-level analysis of the trained RRWaveNet
full models of 261013 and 261105 (261007 on request), in one script.

The two analyses of 261011 are run together, on the same models, windows and
forward pass:
  A. Stem fusion before / after, in the same model
     (261011_RRWaveNet_StemFusion_PrePost_10s).
     Before = the concatenated outputs of the model's three stem branches
     (24 channels). After = the same tensor passed through that model's
     stem_fuse (1x1 conv + GroupNorm, 24 channels).
     Per 10 s window:
     - effective rank: exp(entropy of the normalized eigenvalues of the
       time-centred channel covariance), also divided by 24;
     - off-diagonal channel correlation, as mean |r| and as signed mean.
  B. Temporal pattern extractor: the four residual encoder blocks
     (part 2 of 261011_RRWaveNet_Feature_Results_10s). Stages: Stem (= the
     stem fusion output, encoder input), then B1 (d=1), B2 (d=2), B3 (d=4)
     and B4 (d=8):
     - channel |r| with the target (mean over channels, change against
       Stem, channels of 24 higher than at Stem);
     - PC1 respiratory-band power ratio (0.1-0.5 Hz);
     - PC1 autocorrelation and its similarity to the target autocorrelation
       (Pearson r over lags 1-5 s; each block against Stem and against the
       block before it).
The O vs X part of 261011_RRWaveNet_Feature_Results_10s (full vs
no_stem_fusion) is not included: 261013 and 261105 trained no
no_stem_fusion models.

Models: the models of each run's reported (ensemble) result. Nothing is
trained; every model is reloaded and run on the test windows of its
held-out subject.
  261105  final ensemble (as reported): m0 of every weight decay candidate
          and m1, m2 of the candidate selected in the fold (6 models).
          With --models-261105 selected: m0-m2 of the selected candidate.
  261013  m0-m2 of the weight decay candidate selected in the fold
          (WD_Selection).
  261007  full, m0-m2.
Every window value is averaged over the subject's test windows per model,
then over the subject's models. Statistics: two-sided paired Wilcoxon
signed-rank test over subjects, per dataset and for all datasets of a run
together; p values not adjusted. Checks per model:
  - the saved test windows (segment IDs, starts, fs, window, model key,
    targets) match the reloaded data;
  - the After tensor equals the model's own fused_representation;
  - the model output is compared with the saved prediction
    (Max_Abs_Diff_Saved_Prediction in input_models.csv).
Each run's data are read by its own script (imported, nothing is
trained), so the windows are exactly the ones the models were tested on.

Figures
-------
Legends and statistics boxes are placed outside the plot area, to the
right of it.

Outputs (PROJECT_ROOT = D:\PPG2RSP_RRWaveNet_Inspired\261111_Model_FeatureLevel_ACF_1to5s)
  {run}\input_models.csv, coverage.csv, skipped_subjects.csv
  {run}\Stem_Fusion_PrePost\
      prepost_windows.csv, prepost_members.csv, prepost_subjects.csv,
      prepost_tests.csv
      Figures\overview_10s_<metric>.png, <Dataset>_10s_<metric>.png,
      all_datasets_prepost_summary.png
  {run}\Encoder_Feature_Analysis\
      encoder_channel_corr_tests.csv, encoder_psd_tests.csv,
      encoder_acf_tests.csv, encoder_subjects.csv, encoder_channels.csv,
      encoder_members.csv, encoder_curves.npz
      Figures\<Dataset>_10s_channel_delta_heatmap.png, _channel_r_summary.png,
      _pc1_psd.png, _pc1_acf.png, _pc1_acf_blocks.png, _pc1_acf_summary.png
  Summary\261111_Model_FeatureLevel_ACF_1to5s_results.xlsx and feature_level_key_results.csv
      (A and B of every run and dataset side by side), RESULTS.md
  with <Dataset> = CapnoBase, BIDMC, STEAM2 and All_datasets.

Run from D:\PPG2RSP_RRWaveNet_Inspired\261111_Model_FeatureLevel_ACF_1to5s.
Each run's script is looked for in its results folder:
  261105  D:\PPG2RSP_RRWaveNet_Inspired\261105_RRWaveNet_Unified_Optimized\261105_RRWaveNet_Unified_Optimized.py
  261013  D:\PPG2RSP_RRWaveNet_Inspired\261013_RRWavNet_STEAM_BIDMC_v1\261013_RRWaveNet_Adult_SubjectVal_WD.py
  261007  D:\PPG2RSP_RRWaveNet_Inspired\261007_RRWaveNet_NoEvent_v1\261007_RRWaveNet_Ensemble_NoEvent.py
  If it is not there, it is looked for next to this script.

Commands:
  python 261111_Model_FeatureLevel_ACF_1to5s.py                 (261013 and 261105)
  python 261111_Model_FeatureLevel_ACF_1to5s.py --runs 261007 261013 261105
  --check-inputs-only       list the model files only (no data, no inference)
  --root-261105 <folder>    other results folders (also --root-261013, --root-261007)
  --datasets bidmc steam2, --subjects ..., --force-cpu, --gpu,
  the data path options of the runs.

Reference: Roy O, Vetterli M. The effective rank: a measure of effective
dimensionality. EUSIPCO 2007.
"""

import argparse
import importlib.util
import json
import sys
import time
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

SCRIPT_NAME = "261111_Model_FeatureLevel_ACF_1to5s"
SCRIPT_DIR = Path(__file__).resolve().parent
WORKSPACE = Path(r"D:\PPG2RSP_RRWaveNet_Inspired")
PROJECT_ROOT = WORKSPACE / SCRIPT_NAME

RUNS = {
    "261105": {"root": WORKSPACE / "261105_RRWaveNet_Unified_Optimized",
               "script": "261105_RRWaveNet_Unified_Optimized.py", "models": "final"},
    "261013": {"root": WORKSPACE / "261013_RRWavNet_STEAM_BIDMC_v1",
               "script": "261013_RRWaveNet_Adult_SubjectVal_WD.py", "models": "selected"},
    "261007": {"root": WORKSPACE / "261007_RRWaveNet_NoEvent_v1",
               "script": "261007_RRWaveNet_Ensemble_NoEvent.py", "models": "full"},
}
DEFAULT_RUNS = ("261013", "261105")
WINDOW = "w10s_nonoverlap"
WINDOW_TAG = "10s"
MEMBERS = 3
ALL_DATASETS = "All datasets"
CONSTANT_CHANNEL_SD = 1e-6

# A. Stem fusion before / after
PREPOST_FOLDER = "Stem_Fusion_PrePost"
PREPOST_STAGES = ("Before", "After")
PREPOST_COLORS = ("#e8968c", "#5b8db8")
PREPOST_METRICS = {
    "EffectiveRank": "Effective rank",
    "EffectiveRank_Normalized": "Effective rank / channels",
    "OffDiag_AbsMean": "Mean absolute off-diagonal correlation",
    "OffDiag_SignedMean": "Mean signed off-diagonal correlation",
}

# B. Residual encoder blocks
ENCODER_FOLDER = "Encoder_Feature_Analysis"
STAGES = ("Stem", "B1", "B2", "B3", "B4")
STAGE_COLORS = {"Target": "#111111", "Stem": "#4c78a8", "B1": "#f2a33a", "B2": "#59a14f",
                "B3": "#e15759", "B4": "#9467bd"}
RESP_BAND_HZ = (0.1, 0.5)
ACF_MAX_LAG_SEC = 10.0  # 261106: 12 s
ACF_BAND_SEC = (1.0, 5.0)  # lags of the ACF r; 261106: 2-10 s
ACF_MIN_RUN_SEC = 24.0
PSD_PLOT_MAX_HZ = 1.0


# ---------------------------------------------------------------------------
# Runs, scripts and model files
# ---------------------------------------------------------------------------

def run_root(args: argparse.Namespace, run: str) -> Path:
    given = getattr(args, f"root_{run}")
    return Path(given).expanduser().resolve() if given else RUNS[run]["root"]


def load_backbone(run: str, root: Path):
    """Import the run's own script as a module (no training is started)."""
    name = RUNS[run]["script"]
    for folder in (root, SCRIPT_DIR, Path.cwd()):
        path = folder / name
        if path.is_file():
            spec = importlib.util.spec_from_file_location(f"rrwavenet_{run}_featurelevel", str(path))
            module = importlib.util.module_from_spec(spec)
            sys.modules[spec.name] = module
            spec.loader.exec_module(module)
            return module, path
    raise FileNotFoundError(f"{name} not found in {root}, {SCRIPT_DIR} or {Path.cwd()}")


def read_json(path: Path) -> Optional[dict]:
    try:
        return json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


def window_spec(backbone):
    return next(s for s in backbone.WINDOW_SPECS if s.name == WINDOW)


def result_tags(run: str, backbone, root: Path, dataset_key: str) -> List[str]:
    """Subject tags (safe_tag) with a stored fold of the run."""
    spec = window_spec(backbone)
    label = backbone.DATASET_CONFIGS[dataset_key]["label"]
    if RUNS[run]["models"] == "full":
        folder = backbone.condition_root(root, "full", spec, dataset_key) / "Members" / "m0" / "Models"
        return sorted(p.name[:-len("_best.pth")] for p in folder.glob("*_best.pth"))
    folder = root / "WD_Selection" / f"{spec.order:02d}_{spec.name}" / label
    return sorted(p.name[:-len("_selection.json")] for p in folder.glob("*_selection.json"))


def public(item: dict) -> dict:
    """A model entry for the tables (paths as text, no objects)."""
    return {k: (str(v) if isinstance(v, Path) else v) for k, v in item.items() if k != "variant"}


def model_list(run: str, backbone, root: Path, dataset_key: str, subject: str,
               choice_261105: str) -> Tuple[List[dict], str]:
    """The models of the subject's reported result: (variant, member, files).
    Returns ([], reason) when the fold is not complete."""
    spec = window_spec(backbone)
    if RUNS[run]["models"] == "full":
        variant = backbone.VARIANTS["full"]
        entries = [(variant, member) for member in range(MEMBERS)]
    else:
        record = read_json(backbone.selection_path(root, spec, dataset_key, subject))
        if record is None or "Selected_Weight_Decay" not in record:
            return [], "no weight decay selection (WD_Selection) for the fold"
        chosen = backbone.build_variants([float(record["Selected_Weight_Decay"])])
        chosen = next(iter(chosen.values()))
        if run == "261105" and choice_261105 == "final":
            first = [next(iter(backbone.build_variants([float(row["Weight_Decay"])]).values()))
                     for row in record.get("Candidates", [])]
            entries = [(variant, 0) for variant in first] + [(chosen, m) for m in range(1, MEMBERS)]
        else:
            entries = [(chosen, member) for member in range(MEMBERS)]
    models, missing = [], []
    tag = backbone.safe_tag(subject)
    for variant, member in entries:
        condition = backbone.condition_root(root, variant.name, spec, dataset_key)
        checkpoint = condition / "Members" / f"m{member}" / "Models" / f"{tag}_best.pth"
        summary_path, _, prediction = backbone.member_paths(condition, subject, member)
        summary = read_json(summary_path)
        problems = []
        if not checkpoint.is_file():
            problems.append("checkpoint")
        if summary is None:
            problems.append("member summary")
        elif summary.get("Model") != variant.options.key:
            problems.append("summary of another model")
        if not prediction.is_file():
            problems.append("saved test outputs")
        models.append({"Variant": variant.name, "Member": member, "Model": f"{variant.name}/m{member}",
                       "variant": variant, "Checkpoint": checkpoint, "Prediction": prediction,
                       "Ready": not problems, "Missing": "; ".join(problems)})
        if problems:
            missing.append(f"{variant.name}/m{member}: {', '.join(problems)}")
    if missing:
        return models, "incomplete models: " + " | ".join(missing)
    return models, ""


# ---------------------------------------------------------------------------
# Metrics (as in the two 261011 scripts)
# ---------------------------------------------------------------------------

def wilcoxon_p(a: np.ndarray, b: np.ndarray) -> float:
    a, b = np.asarray(a, float), np.asarray(b, float)
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


def sd(values) -> float:
    values = np.asarray(values, float)
    values = values[np.isfinite(values)]
    return float(np.std(values, ddof=1)) if values.size > 1 else np.nan


def window_metrics(features: np.ndarray) -> Dict[str, np.ndarray]:
    """Effective rank and off-diagonal correlation of every window
    (features: windows x channels x samples)."""
    x = np.asarray(features, dtype=np.float64)
    _, channels, samples = x.shape
    x = x - x.mean(axis=2, keepdims=True)
    covariance = np.einsum("wct,wdt->wcd", x, x) / max(1, samples - 1)
    covariance = (covariance + np.transpose(covariance, (0, 2, 1))) / 2.0
    eigenvalues = np.clip(np.linalg.eigvalsh(covariance), 0.0, None)
    total = eigenvalues.sum(axis=1, keepdims=True)
    with np.errstate(divide="ignore", invalid="ignore"):
        p = np.where(total > 1e-12, eigenvalues / total, 0.0)
        entropy = -np.sum(np.where(p > 0, p * np.log(np.where(p > 0, p, 1.0)), 0.0), axis=1)
    effective_rank = np.where(total[:, 0] > 1e-12, np.exp(entropy), np.nan)
    channel_sd = np.sqrt(np.clip(np.einsum("wcc->wc", covariance), 0.0, None))
    active = channel_sd > CONSTANT_CHANNEL_SD
    pairs = active[:, :, None] & active[:, None, :] & ~np.eye(channels, dtype=bool)[None]
    with np.errstate(divide="ignore", invalid="ignore"):
        correlation = covariance / (channel_sd[:, :, None] * channel_sd[:, None, :])
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
    channel_sd = matrix.std(axis=1, keepdims=True)
    keep = channel_sd[:, 0] > 1e-8
    z = (matrix[keep] - mean[keep]) / channel_sd[keep]
    covariance = z @ z.T / max(1, z.shape[1] - 1)
    eigenvalues, eigenvectors = np.linalg.eigh(covariance)
    share = float(eigenvalues[-1] / max(eigenvalues.sum(), 1e-12))
    return (eigenvectors[:, -1] @ z).reshape(windows, samples), share


def mean_psd(signals: np.ndarray, fs: float) -> Tuple[np.ndarray, np.ndarray]:
    frequencies, power = periodogram(signals, fs=fs, window="hann", detrend="constant", axis=-1)
    return frequencies, power.mean(axis=0)


def band_ratio(frequencies: np.ndarray, power: np.ndarray) -> float:
    band = (frequencies >= RESP_BAND_HZ[0]) & (frequencies <= RESP_BAND_HZ[1])
    total = power[frequencies > 0].sum()
    return float(power[band].sum() / total) if total > 0 else np.nan


def contiguous_runs(metadata: Sequence[dict], samples: int) -> List[List[int]]:
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
# One forward pass per model: stem branches -> fusion -> encoder blocks
# ---------------------------------------------------------------------------

def check_saved_windows(stored, metadata: Sequence[dict], fs: int, samples: int, model_key: str) -> str:
    """'' when the saved test outputs belong to these windows and this model."""
    ids = np.array([item["segment_id"] for item in metadata], dtype=np.int64)
    starts = np.array([item["start"] for item in metadata], dtype=np.int64)
    if not (np.array_equal(stored["segment_ids"], ids) and np.array_equal(stored["window_starts"], starts)):
        return "saved test-window IDs / starts differ from the reloaded data"
    if (int(stored["fs"]) != fs or float(stored["input_window_sec"]) != 10.0
            or float(stored["input_stride_sec"]) != 10.0 or str(stored["model_key"].item()) != model_key
            or stored["targets"].shape != (len(metadata), samples)):
        return "saved outputs are not those of this 10 s model"
    return ""


def forward_features(model, loader, device, saved_targets: np.ndarray, saved_predictions: Optional[np.ndarray]):
    """Before / after fusion, the encoder stages, the targets, and the largest
    difference between the model output and the saved prediction."""
    before, targets = [], []
    stages: List[List[np.ndarray]] = [[] for _ in STAGES]
    max_diff = 0.0
    offset = 0
    with torch.no_grad():
        for batch in loader:
            x, y = batch[0].to(device), batch[1]
            pre = torch.cat([branch(x) for branch in model.stem_branches], dim=1)
            post = model.stem_fuse(pre)
            if offset == 0 and not torch.allclose(post, model.fused_representation(x), atol=1e-6, rtol=1e-5):
                raise ValueError("post-fusion features differ from the model's fused_representation")
            size = len(x)
            target = y.squeeze(1).numpy()
            if not np.allclose(target, saved_targets[offset:offset + size], rtol=1e-5, atol=1e-6, equal_nan=True):
                raise ValueError("targets differ from the saved test windows")
            before.append(pre.cpu().numpy().astype(np.float32))
            features = post
            stages[0].append(features.cpu().numpy().astype(np.float32))
            for index, block in enumerate(model.encoder, start=1):
                features = block(features)
                stages[index].append(features.cpu().numpy().astype(np.float32))
            if saved_predictions is not None:
                output = model(x).squeeze(1).cpu().numpy()
                saved = np.asarray(saved_predictions[offset:offset + size]).reshape(output.shape)
                max_diff = max(max_diff, float(np.nanmax(np.abs(output - saved))))
            targets.append(target.astype(np.float32))
            offset += size
    if offset != len(saved_targets):
        raise ValueError("not every saved test window was processed")
    return (np.concatenate(before), [np.concatenate(chunks) for chunks in stages],
            np.concatenate(targets), max_diff if saved_predictions is not None else np.nan)


def collect_run(run: str, backbone, root: Path, args: argparse.Namespace, device) -> dict:
    spec = window_spec(backbone)
    fs = int(backbone.TARGET_FS)
    samples = int(round(spec.window_sec * fs))
    max_lag = int(round(ACF_MAX_LAG_SEC * fs))
    min_length = int(round(ACF_MIN_RUN_SEC * fs))
    window_rows, prepost_rows, member_rows, channel_rows = [], [], [], []
    input_rows, skipped, coverage = [], [], []
    curves: Dict[Tuple[str, str], Dict[str, List[np.ndarray]]] = {}
    frequencies = None
    built: Dict[str, torch.nn.Module] = {}
    for dataset_key in [key for key in backbone.DATASET_ORDER if key in args.datasets]:
        label = backbone.DATASET_CONFIGS[dataset_key]["label"]
        tags = result_tags(run, backbone, root, dataset_key)
        if args.subjects:
            tags = [tag for tag in tags if tag in {backbone.safe_tag(s) for s in args.subjects}]
        if not tags:
            print(f"[{run}] {label}: no stored fold, skipped", flush=True)
            coverage.append({"Run": run, "Dataset": label, "Status": "no stored fold"})
            continue
        started = time.perf_counter()
        print(f"\n[{run}] {label}: reading and preprocessing the raw data ...", flush=True)
        data = backbone.load_dataset(dataset_key, spec)
        by_tag = {backbone.safe_tag(subject): subject for subject in data}
        for tag in tags:
            if tag not in by_tag:
                skipped.append({"Run": run, "Dataset": label, "Subject": tag, "Reason": "raw subject unavailable"})
        subjects = [by_tag[tag] for tag in tags if tag in by_tag]
        print(f"[{run}] {label}: {len(subjects)} subjects with a stored fold "
              f"({time.perf_counter() - started:.1f} s)", flush=True)
        analysed = 0
        for number, subject in enumerate(subjects, start=1):
            subject_started = time.perf_counter()
            models, reason = model_list(run, backbone, root, dataset_key, subject, args.models_261105)
            base = {"Run": run, "Dataset": label, "Subject": subject}
            if reason:
                skipped.append({**base, "Reason": reason})
                input_rows += [{**base, **public(m)} for m in models]
                print(f"[skip] {run} {label} {subject}: {reason}", flush=True)
                continue
            metadata = backbone.test_metadata(data, subject)
            loader = backbone.make_eval_loader(
                backbone.SubjectWindowDataset(data, [subject], "test", False), args.batch_size, device)
            runs = contiguous_runs(metadata, samples)
            store = curves.setdefault((label, subject), {})
            subject_rows = []
            try:
                for item in models:
                    variant = item["variant"]
                    if variant.name not in built:
                        built[variant.name] = variant.options.build().to(device).eval()
                    model = built[variant.name]
                    if getattr(model, "stem_fusion", "conv1x1") != "conv1x1":
                        raise ValueError("the model has no 1x1 + GroupNorm stem fusion")
                    model.load_state_dict(torch.load(item["Checkpoint"], map_location=device))
                    model.eval()
                    with np.load(item["Prediction"], allow_pickle=False) as stored:
                        problem = check_saved_windows(stored, metadata, fs, samples, variant.options.key)
                        if problem:
                            raise ValueError(problem)
                        saved_targets = stored["targets"]
                        saved_predictions = stored["predictions"] if "predictions" in stored.files else None
                    before, stages, target, max_diff = forward_features(
                        model, loader, device, saved_targets, saved_predictions)
                    input_rows.append({**base, **public(item),
                                       "N_Windows": int(len(target)), "Max_Abs_Diff_Saved_Prediction": max_diff})
                    key = {**base, "Model": item["Model"]}
                    # A. stem fusion before / after
                    for stage, features in zip(PREPOST_STAGES, (before, stages[0])):
                        metrics = window_metrics(features)
                        for index in range(len(features)):
                            window_rows.append({**key, "Stage": stage, "Window_Index": index,
                                                "Segment_ID": metadata[index]["segment_id"],
                                                "Start_Sample": metadata[index]["start"],
                                                **{name: float(values[index]) for name, values in metrics.items()}})
                        prepost_rows.append({**key, "Stage": stage, "N_Windows": int(len(features)),
                                             **{name: float(np.nanmean(values)) if np.isfinite(values).any() else np.nan
                                                for name, values in metrics.items()}})
                    # B. encoder blocks
                    frequencies, target_psd = mean_psd(target, fs)
                    target_acf = mean_acf(target, runs, max_lag, min_length)
                    store.setdefault("Target_PSD", []).append(target_psd)
                    store.setdefault("Target_ACF", []).append(target_acf)
                    row = {**key, "N_Windows": int(len(target)),
                           "Target_RespBand_Ratio": band_ratio(frequencies, target_psd)}
                    for stage, features in zip(STAGES, stages):
                        abs_r = channel_abs_r(features, target)
                        channel_rows += [{**key, "Stage": stage, "Channel": channel, "Abs_r": float(value)}
                                         for channel, value in enumerate(abs_r)]
                        pc1, share = first_component(features)
                        _, psd = mean_psd(pc1, fs)
                        acf = mean_acf(pc1, runs, max_lag, min_length)
                        store.setdefault(f"{stage}_PSD", []).append(psd)
                        store.setdefault(f"{stage}_ACF", []).append(acf)
                        row[f"{stage}_Channel_AbsR_Mean"] = float(np.nanmean(abs_r))
                        row[f"{stage}_PC1_Variance_Share"] = share
                        row[f"{stage}_RespBand_Ratio"] = band_ratio(frequencies, psd)
                        row[f"{stage}_ACF_r"] = acf_similarity(acf, target_acf, fs)
                    subject_rows.append(row)
            except Exception as exc:  # one subject must not stop a long run
                skipped.append({**base, "Reason": f"{type(exc).__name__}: {exc}"})
                curves.pop((label, subject), None)
                window_rows[:] = [r for r in window_rows if (r["Dataset"], r["Subject"]) != (label, subject)]
                prepost_rows[:] = [r for r in prepost_rows if (r["Dataset"], r["Subject"]) != (label, subject)]
                channel_rows[:] = [r for r in channel_rows if (r["Dataset"], r["Subject"]) != (label, subject)]
                print(f"[skip] {run} {label} {subject}: {exc}", flush=True)
                continue
            member_rows += subject_rows
            analysed += 1
            print(f"[{number}/{len(subjects)}] {run} {label} {subject}: {len(metadata)} windows x "
                  f"{len(models)} models ({time.perf_counter() - subject_started:.1f} s)", flush=True)
            del loader
        coverage.append({"Run": run, "Dataset": label, "Status": "analysed" if analysed else "no subject analysed",
                         "N_Stored_Folds": len(tags), "N_Subjects_Analysed": analysed})
        del data
        backbone.release_memory(device)
    return {"windows": pd.DataFrame(window_rows), "prepost": pd.DataFrame(prepost_rows),
            "members": pd.DataFrame(member_rows), "channels": pd.DataFrame(channel_rows),
            "inputs": pd.DataFrame(input_rows),
            "skipped": pd.DataFrame(skipped, columns=["Run", "Dataset", "Subject", "Reason"]),
            "coverage": pd.DataFrame(coverage), "curves": curves, "frequencies": frequencies, "fs": fs}


# ---------------------------------------------------------------------------
# Figure helpers: legend and statistics outside the plot, to the right
# ---------------------------------------------------------------------------

def legend_outside(fig, axis, note: Optional[str] = None, fontsize: float = 8.0) -> None:
    legend = axis.legend(loc="upper left", bbox_to_anchor=(1.02, 1.0), borderaxespad=0.0,
                         fontsize=fontsize, frameon=True)
    if note:
        fig.canvas.draw()
        box = legend.get_window_extent().transformed(axis.transAxes.inverted())
        axis.text(1.02, box.y0 - 0.04, note, transform=axis.transAxes, va="top", ha="left", fontsize=fontsize,
                  bbox={"facecolor": "white", "edgecolor": "#cc0000"})


# ---------------------------------------------------------------------------
# A. Stem fusion before / after: statistics and figures
# ---------------------------------------------------------------------------

def prepost_outputs(run: str, collected: dict, datasets: List[str], out_root: Path) -> pd.DataFrame:
    out_root.mkdir(parents=True, exist_ok=True)
    members = collected["prepost"]
    if members.empty:
        return pd.DataFrame()
    metrics = list(PREPOST_METRICS) + ["Undefined_Pair_Share"]
    keys = ["Run", "Dataset", "Subject"]
    subject_long = members.groupby(keys + ["Stage"], sort=False)[metrics].mean().reset_index()
    subjects = subject_long.pivot_table(index=keys, columns="Stage", values=metrics, sort=False)
    subjects.columns = [f"{metric}_{stage}" for metric, stage in subjects.columns]
    subjects = subjects.reset_index()
    for metric in metrics:
        subjects[f"{metric}_Delta"] = subjects[f"{metric}_After"] - subjects[f"{metric}_Before"]
    groups = [(d, subjects[subjects["Dataset"] == d]) for d in datasets]
    groups = [(d, f) for d, f in groups if not f.empty] + [(ALL_DATASETS, subjects)]
    rows = []
    for dataset, frame in groups:
        for metric in PREPOST_METRICS:
            before = frame[f"{metric}_Before"].to_numpy(float)
            after = frame[f"{metric}_After"].to_numpy(float)
            keep = np.isfinite(before) & np.isfinite(after)
            before, after = before[keep], after[keep]
            if not len(before):
                continue
            p_value = wilcoxon_p(after, before)
            row = {"Run": run, "Window_Config": WINDOW, "Dataset": dataset, "Metric": metric,
                   "N_Subjects": int(len(before)), "Delta_After_minus_Before": float(np.mean(after - before)),
                   "N_After_Higher": int(np.sum(after > before)), "N_After_Lower": int(np.sum(after < before)),
                   "Wilcoxon_p": p_value, "Significance": significance_stars(p_value)}
            for name, values in (("Before", before), ("After", after)):
                row.update({f"{name}_Mean": float(np.mean(values)), f"{name}_SD": sd(values),
                            f"{name}_Median": float(np.median(values)),
                            f"{name}_Q1": float(np.percentile(values, 25)),
                            f"{name}_Q3": float(np.percentile(values, 75))})
            rows.append(row)
    tests = pd.DataFrame(rows)
    collected["windows"].to_csv(out_root / "prepost_windows.csv", index=False, encoding="utf-8-sig")
    members.to_csv(out_root / "prepost_members.csv", index=False, encoding="utf-8-sig")
    subjects.to_csv(out_root / "prepost_subjects.csv", index=False, encoding="utf-8-sig")
    tests.to_csv(out_root / "prepost_tests.csv", index=False, encoding="utf-8-sig")

    figure_root = out_root / "Figures"
    figure_root.mkdir(parents=True, exist_ok=True)

    def plot_pair(axis, frame, metric, row):
        before = frame[f"{metric}_Before"].to_numpy(float)
        after = frame[f"{metric}_After"].to_numpy(float)
        keep = np.isfinite(before) & np.isfinite(after)
        before, after = before[keep], after[keep]
        axis.plot([1, 2], np.vstack([before, after]), color="#777777", alpha=0.14, linewidth=0.65, zorder=1)
        boxes = axis.boxplot([before, after], positions=[1, 2], widths=0.48, showfliers=False,
                             patch_artist=True, medianprops={"color": "#111111", "linewidth": 1.8})
        for box, color in zip(boxes["boxes"], PREPOST_COLORS):
            box.set_facecolor(color)
            box.set_alpha(0.9)
        axis.scatter([1, 2], [before.mean(), after.mean()], color="#111111", marker="D", s=20, zorder=4)
        axis.set_xticks([1, 2])
        axis.set_xticklabels(["Before fusion", "After fusion"])
        axis.set_ylabel(PREPOST_METRICS[metric], fontsize=9)
        axis.grid(axis="y", color="#e7e7e7")
        axis.set_axisbelow(True)
        axis.set_title(f"{row['Dataset']} (n = {row['N_Subjects']})\n{before.mean():.3f} -> {after.mean():.3f} | "
                       f"{p_text(row['Wilcoxon_p'])} {row['Significance']}", fontsize=10)
        axis.margins(y=0.12)

    def test_row(dataset, metric):
        found = tests[(tests["Dataset"] == dataset) & (tests["Metric"] == metric)]
        return None if found.empty else found.iloc[0]

    for metric in PREPOST_METRICS:
        drawn = [(d, f, test_row(d, metric)) for d, f in groups if test_row(d, metric) is not None]
        fig, axes = plt.subplots(1, len(drawn), figsize=(4.1 * len(drawn), 4.4), squeeze=False)
        for axis, (dataset, frame, row) in zip(axes[0], drawn):
            plot_pair(axis, frame, metric, row)
        fig.suptitle(f"{run}: same trained model, stem fusion before vs after | {WINDOW_TAG} windows", fontsize=12)
        fig.tight_layout(rect=(0, 0, 1, 0.94))
        fig.savefig(figure_root / f"overview_{WINDOW_TAG}_{metric}.png", dpi=180)
        plt.close(fig)
        for dataset, frame, row in drawn:
            fig, axis = plt.subplots(figsize=(6.0, 4.5))
            plot_pair(axis, frame, metric, row)
            fig.suptitle(f"{run}: same trained model | {WINDOW_TAG} windows", fontsize=11)
            fig.tight_layout()
            fig.savefig(figure_root / f"{dataset.replace(' ', '_')}_{WINDOW_TAG}_{metric}.png", dpi=180)
            plt.close(fig)
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.7))
    for axis, metric in zip(axes, ("EffectiveRank", "OffDiag_AbsMean")):
        plot_pair(axis, subjects, metric, test_row(ALL_DATASETS, metric))
    fig.suptitle(f"{run}: stem fusion before / after in the same model | {WINDOW_TAG}", fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(figure_root / "all_datasets_prepost_summary.png", dpi=180)
    plt.close(fig)
    return tests


# ---------------------------------------------------------------------------
# B. Encoder blocks: statistics and figures
# ---------------------------------------------------------------------------

def stage_labels(backbone) -> List[str]:
    return ["Stem"] + [f"B{i + 1} (d={d})" for i, d in enumerate(backbone.MODEL_CONFIG["encoder_dilations"])]


def group_statistics(run: str, frame: pd.DataFrame, channel_subject: pd.DataFrame, dataset: str,
                     labels: List[str]) -> dict:
    n = len(frame)
    channels = channel_subject.merge(frame[["Dataset", "Subject"]], on=["Dataset", "Subject"])
    pivot = channels.pivot_table(index=["Dataset", "Subject", "Channel"], columns="Stage", values="Abs_r")
    delta = pivot[list(STAGES[1:])].sub(pivot["Stem"], axis=0)
    better = (delta > 0).groupby(level=["Dataset", "Subject"]).sum()
    stem_mean = frame["Stem_Channel_AbsR_Mean"].to_numpy(float)
    target_ratio = frame["Target_RespBand_Ratio"].to_numpy(float)
    stem_ratio = frame["Stem_RespBand_Ratio"].to_numpy(float)
    stem_acf = frame["Stem_ACF_r"].to_numpy(float)
    corr_rows, psd_rows, acf_rows = [], [], []
    for position, (stage, stage_label) in enumerate(zip(STAGES, labels)):
        base = {"Run": run, "Window_Config": WINDOW, "Dataset": dataset, "Stage": stage_label, "N_Subjects": n}
        values = frame[f"{stage}_Channel_AbsR_Mean"].to_numpy(float)
        row = {**base, "PCC_Mean_Abs_r": float(np.mean(values)), "PCC_SD": sd(values)}
        if position:
            row.update({"Mean_Delta_Abs_r": float(np.mean(values - stem_mean)),
                        "Better_Channels_of_24": float(better[stage].mean()),
                        "N_Subjects_Higher": int(np.sum(values > stem_mean)),
                        "Wilcoxon_p_vs_Stem": wilcoxon_p(values, stem_mean)})
        corr_rows.append(row)
        ratio = frame[f"{stage}_RespBand_Ratio"].to_numpy(float)
        psd_rows.append({**base, "RespBand_Ratio_Mean": float(np.nanmean(ratio)), "RespBand_Ratio_SD": sd(ratio),
                         "Target_RespBand_Ratio_Mean": float(np.nanmean(target_ratio)),
                         "PC1_Variance_Share_Mean": float(frame[f"{stage}_PC1_Variance_Share"].mean()),
                         "N_Subjects_Higher_than_Stem": int(np.sum(ratio > stem_ratio)) if position else np.nan,
                         "Wilcoxon_p_vs_Stem": wilcoxon_p(ratio, stem_ratio) if position else np.nan})
        acf_r = frame[f"{stage}_ACF_r"].to_numpy(float)
        previous = frame[f"{STAGES[position - 1]}_ACF_r"].to_numpy(float) if position else None
        acf_rows.append({**base, "ACF_r_Mean": float(np.nanmean(acf_r)), "ACF_r_SD": sd(acf_r),
                         "ACF_r_Median": float(np.nanmedian(acf_r)),
                         "Delta_vs_Stem": float(np.nanmean(acf_r - stem_acf)) if position else np.nan,
                         "N_Subjects_Higher_than_Stem": int(np.sum(acf_r > stem_acf)) if position else np.nan,
                         "Wilcoxon_p_vs_Stem": wilcoxon_p(acf_r, stem_acf) if position else np.nan,
                         "Delta_vs_Previous": float(np.nanmean(acf_r - previous)) if position else np.nan,
                         "N_Subjects_Higher_than_Previous": int(np.sum(acf_r > previous)) if position else np.nan,
                         "Wilcoxon_p_vs_Previous": wilcoxon_p(acf_r, previous) if position else np.nan})
    return {"n": n, "delta": delta, "better": better, "corr": corr_rows, "psd": psd_rows, "acf": acf_rows}


def group_curves(frame: pd.DataFrame, curves: dict) -> Tuple[Dict[str, np.ndarray], Dict[str, np.ndarray]]:
    """Mean PSD and ACF curves: mean over a subject's models, then over subjects."""
    keys = list(zip(frame["Dataset"], frame["Subject"]))

    def mean_curve(name: str) -> np.ndarray:
        return np.nanmean([np.mean(curves[key][name], axis=0) for key in keys], axis=0)

    names = ("Target",) + STAGES
    return ({name: mean_curve(f"{name}_PSD") for name in names},
            {name: mean_curve(f"{name}_ACF") for name in names})


def figure_channel_heatmap(path: Path, stats_: dict, labels: List[str], title: str) -> None:
    delta, better = stats_["delta"], stats_["better"]
    channel_delta = delta.groupby(level="Channel").mean()
    order = sorted(channel_delta.index, reverse=True)
    matrix = channel_delta.loc[order, list(STAGES[1:])].to_numpy()
    limit = max(float(np.nanmax(np.abs(matrix))), 1e-6)
    fig, axis = plt.subplots(figsize=(6.0, 6.5))
    image = axis.imshow(matrix, aspect="auto", cmap="RdYlBu_r", norm=TwoSlopeNorm(0.0, -limit, limit))
    axis.set_xticks(range(len(STAGES) - 1))
    axis.set_xticklabels([label.replace(" ", "\n") for label in labels[1:]])
    axis.set_yticks(range(len(order)))
    axis.set_yticklabels([str(channel) for channel in order], fontsize=7)
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


def figure_line_summary(path: Path, values: List[float], notes: List[str], labels: List[str],
                        ylabel: str, title: str, offset: int) -> None:
    fig, axis = plt.subplots(figsize=(6.4, 4.4))
    axis.plot(range(len(STAGES)), values, "o-", color="#1f77b4")
    for i, value in enumerate(values):
        axis.annotate(f"{value:.3f}", (i, value), textcoords="offset points", xytext=(0, 7),
                      ha="center", fontsize=8, color="#1f77b4", fontweight="bold")
    bottom = min(values)
    for i, note in enumerate(notes, start=1):
        axis.annotate(note, (i, bottom), textcoords="offset points", xytext=(0, offset), ha="center",
                      fontsize=6.3, bbox={"facecolor": "white", "edgecolor": "#999999", "pad": 1.5})
    axis.set_xticks(range(len(STAGES)))
    axis.set_xticklabels([label.replace(" ", "\n") for label in labels])
    axis.set_ylabel(ylabel)
    span = max(max(values) - bottom, 0.05)
    axis.set_ylim(bottom - 0.7 * span, max(values) + 0.15 * span)
    axis.grid(True, color="#e6e6e6")
    axis.set_title(title, fontsize=10)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def figure_pc1_psd(path: Path, frequencies: np.ndarray, psd_curves: Dict[str, np.ndarray],
                   frame: pd.DataFrame, labels: List[str], title: str) -> None:
    shown = frequencies <= PSD_PLOT_MAX_HZ
    fig, axis = plt.subplots(figsize=(9.6, 4.6))
    axis.axvspan(*RESP_BAND_HZ, color="#59a14f", alpha=0.10, label="Respiratory band")
    lines = []
    for name, stage_label in zip(("Target",) + STAGES, ["Target RSP"] + labels):
        curve = psd_curves[name][shown]
        scale = np.nanmax(curve) if np.nanmax(curve) > 0 else 1.0
        axis.plot(frequencies[shown], curve / scale, color=STAGE_COLORS[name],
                  linewidth=2.0 if name in ("Target", "B4") else 1.4, label=stage_label)
        lines.append(f"{stage_label}: {float(frame[f'{name}_RespBand_Ratio'].mean()):.3f}")
    axis.set_xlabel("Frequency (Hz)")
    axis.set_ylabel("PSD (scaled to its maximum)")
    axis.set_xlim(0, PSD_PLOT_MAX_HZ)
    axis.grid(True, color="#eeeeee")
    axis.set_title(f"Mean representative PSD (PC1): {title}", fontsize=10)
    fig.tight_layout()
    legend_outside(fig, axis, "Mean resp-band power ratio\n" + "\n".join(lines))
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def figure_pc1_acf(path: Path, lags: np.ndarray, acf_curves: Dict[str, np.ndarray], acf_rows: List[dict],
                   title: str) -> None:
    fig, axis = plt.subplots(figsize=(9.6, 4.6))
    axis.axvspan(*ACF_BAND_SEC, color="#e15759", alpha=0.06, label="Resp lag band")
    axis.plot(lags, acf_curves["Target"], color="#111111", linewidth=2.0, label="Target RSP")
    axis.plot(lags, acf_curves["B4"], color="#4c78a8", linewidth=1.6, label="DeepEnc (B4)")
    axis.plot(lags, acf_curves["Stem"], color="#e15759", linewidth=1.4, label="Stem Fuse")
    axis.set_xlabel("Lag (s)")
    axis.set_ylabel("Normalized autocorrelation")
    axis.set_xlim(0, ACF_MAX_LAG_SEC)
    axis.grid(True, color="#eeeeee")
    axis.set_title(f"Mean readout autocorrelation (PC1): {title}", fontsize=10)
    fig.tight_layout()
    legend_outside(fig, axis, f"Auto-correlation\nDeepEnc: r = {acf_rows[-1]['ACF_r_Mean']:.3f}\n"
                              f"Stem Fuse: r = {acf_rows[0]['ACF_r_Mean']:.3f}\n"
                              f"{p_text(acf_rows[-1]['Wilcoxon_p_vs_Stem'])} (Wilcoxon)")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def figure_pc1_acf_blocks(path: Path, lags: np.ndarray, acf_curves: Dict[str, np.ndarray],
                          acf_rows: List[dict], labels: List[str], title: str) -> None:
    fig, axis = plt.subplots(figsize=(9.6, 4.6))
    axis.axvspan(*ACF_BAND_SEC, color="#e15759", alpha=0.06, label="Resp lag band")
    for name, stage_label in zip(("Target",) + STAGES, ["Target RSP"] + labels):
        axis.plot(lags, acf_curves[name], color=STAGE_COLORS[name],
                  linewidth=2.2 if name == "Target" else 1.4, label=stage_label)
    lines = [f"{row['Stage']}: r = {row['ACF_r_Mean']:.3f}"
             + (f" ({p_text(row['Wilcoxon_p_vs_Stem'])})" if position else "")
             for position, row in enumerate(acf_rows)]
    axis.set_xlabel("Lag (s)")
    axis.set_ylabel("Normalized autocorrelation")
    axis.set_xlim(0, ACF_MAX_LAG_SEC)
    axis.grid(True, color="#eeeeee")
    axis.set_title(f"Mean readout autocorrelation per block (PC1): {title}", fontsize=10)
    fig.tight_layout()
    legend_outside(fig, axis, f"Auto-correlation vs target\n(lags {ACF_BAND_SEC[0]:g}-{ACF_BAND_SEC[1]:g} s; p vs Stem)\n" + "\n".join(lines))
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def encoder_outputs(run: str, backbone, collected: dict, datasets: List[str], out_root: Path) -> dict:
    out_root.mkdir(parents=True, exist_ok=True)
    members, channels = collected["members"], collected["channels"]
    if members.empty:
        return {}
    curves, frequencies, fs = collected["curves"], collected["frequencies"], float(collected["fs"])
    figure_root = out_root / "Figures"
    figure_root.mkdir(parents=True, exist_ok=True)
    labels = stage_labels(backbone)
    keys = ["Run", "Dataset", "Subject"]
    value_columns = [c for c in members.columns if c not in keys + ["Model", "N_Windows"]]
    subjects = members.groupby(keys, sort=False)[value_columns].mean().reset_index()
    channel_subject = channels.groupby(keys + ["Stage", "Channel"], sort=False)["Abs_r"].mean().reset_index()
    groups = [(d, subjects[subjects["Dataset"] == d]) for d in datasets]
    groups = [(d, f) for d, f in groups if not f.empty] + [(ALL_DATASETS, subjects)]
    lags = np.arange(int(round(ACF_MAX_LAG_SEC * fs)) + 1) / fs
    corr_rows, psd_rows, acf_rows = [], [], []
    curve_arrays = {"frequencies_hz": frequencies, "acf_lags_sec": lags}
    for dataset, frame in groups:
        tag = f"{dataset.replace(' ', '_')}_{WINDOW_TAG}"
        stats_ = group_statistics(run, frame, channel_subject, dataset, labels)
        n = stats_["n"]
        title = f"{run} | {dataset}, {WINDOW_TAG} (n = {n})"
        corr_rows += stats_["corr"]
        psd_rows += stats_["psd"]
        acf_rows += stats_["acf"]
        psd_curves, acf_curves = group_curves(frame, curves)
        for name in psd_curves:
            curve_arrays[f"{tag}_PSD_{name}"] = psd_curves[name]
            curve_arrays[f"{tag}_ACF_{name}"] = acf_curves[name]
        figure_channel_heatmap(figure_root / f"{tag}_channel_delta_heatmap.png", stats_, labels, title)
        figure_line_summary(
            figure_root / f"{tag}_channel_r_summary.png",
            [float(frame[f"{s}_Channel_AbsR_Mean"].mean()) for s in STAGES],
            [f"delta={r['Mean_Delta_Abs_r']:+.3f}\nbetter={r['Better_Channels_of_24']:.1f}/24\n"
             f"{p_text(r['Wilcoxon_p_vs_Stem'])}" for r in stats_["corr"][1:]],
            labels, "Mean |channel r| with the target", f"Mean |channel r| summary: {title}", -42)
        figure_pc1_psd(figure_root / f"{tag}_pc1_psd.png", frequencies, psd_curves, frame, labels, title)
        figure_pc1_acf(figure_root / f"{tag}_pc1_acf.png", lags, acf_curves, stats_["acf"], title)
        figure_pc1_acf_blocks(figure_root / f"{tag}_pc1_acf_blocks.png", lags, acf_curves, stats_["acf"],
                              labels, title)
        figure_line_summary(
            figure_root / f"{tag}_pc1_acf_summary.png",
            [row["ACF_r_Mean"] for row in stats_["acf"]],
            [f"vs Stem {r['Delta_vs_Stem']:+.3f}\nhigher {r['N_Subjects_Higher_than_Stem']}/{n}, "
             f"{p_text(r['Wilcoxon_p_vs_Stem'])}\nvs prev. {r['Delta_vs_Previous']:+.3f}, "
             f"{p_text(r['Wilcoxon_p_vs_Previous'])}" for r in stats_["acf"][1:]],
            labels, f"ACF similarity to the target (r, lags {ACF_BAND_SEC[0]:g}-{ACF_BAND_SEC[1]:g} s)",
            f"PC1 autocorrelation similarity per block: {title}", -46)
    result = {"corr": pd.DataFrame(corr_rows), "psd": pd.DataFrame(psd_rows), "acf": pd.DataFrame(acf_rows)}
    result["corr"].to_csv(out_root / "encoder_channel_corr_tests.csv", index=False, encoding="utf-8-sig")
    result["psd"].to_csv(out_root / "encoder_psd_tests.csv", index=False, encoding="utf-8-sig")
    result["acf"].to_csv(out_root / "encoder_acf_tests.csv", index=False, encoding="utf-8-sig")
    subjects.to_csv(out_root / "encoder_subjects.csv", index=False, encoding="utf-8-sig")
    channel_subject.to_csv(out_root / "encoder_channels.csv", index=False, encoding="utf-8-sig")
    members.to_csv(out_root / "encoder_members.csv", index=False, encoding="utf-8-sig")
    np.savez_compressed(out_root / "encoder_curves.npz", **curve_arrays)
    return result


# ---------------------------------------------------------------------------
# Integrated results: A and B of every run and dataset side by side
# ---------------------------------------------------------------------------

def key_results(prepost: pd.DataFrame, corr: pd.DataFrame, psd: pd.DataFrame, acf: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (run, dataset), part in prepost.groupby(["Run", "Dataset"], sort=False):
        row = {"Run": run, "Dataset": dataset, "N_Subjects": int(part["N_Subjects"].max())}
        for metric, name in (("EffectiveRank", "EffRank"), ("OffDiag_AbsMean", "OffDiag_AbsR")):
            r = part[part["Metric"] == metric].iloc[0]
            row.update({f"A_{name}_Before": r["Before_Mean"], f"A_{name}_After": r["After_Mean"],
                        f"A_{name}_p": r["Wilcoxon_p"], f"A_{name}_Sig": r["Significance"]})
        for table, value, name in ((corr, "PCC_Mean_Abs_r", "ChannelAbsR"), (psd, "RespBand_Ratio_Mean", "RespBandRatio"),
                                   (acf, "ACF_r_Mean", "ACF_r")):
            sub = table[(table["Run"] == run) & (table["Dataset"] == dataset)]
            if sub.empty:
                continue
            stem, last = sub.iloc[0], sub.iloc[-1]
            p_value = last["Wilcoxon_p_vs_Stem"]
            row.update({f"B_{name}_Stem": stem[value], f"B_{name}_B4": last[value],
                        f"B_{name}_B4_minus_Stem": last[value] - stem[value],
                        f"B_{name}_p_B4_vs_Stem": p_value, f"B_{name}_Sig": significance_stars(p_value)})
            if name == "RespBandRatio":
                row["B_Target_RespBandRatio"] = stem["Target_RespBand_Ratio_Mean"]
        rows.append(row)
    return pd.DataFrame(rows)


def write_summary(out_root: Path, tables: Dict[str, pd.DataFrame]) -> None:
    out_root.mkdir(parents=True, exist_ok=True)
    key = tables["Key_Results"]
    key.to_csv(out_root / "feature_level_key_results.csv", index=False, encoding="utf-8-sig")
    with pd.ExcelWriter(out_root / f"{SCRIPT_NAME}_results.xlsx", engine="openpyxl") as writer:
        for name, frame in tables.items():
            frame.to_excel(writer, sheet_name=name[:31], index=False)
    lines = ["# Feature-level 분석 결과 (stem fusion 전후 + temporal pattern extractor)", "",
             "A: 같은 모델의 stem fusion 전(branch 연결)과 후(1×1 conv + GroupNorm). "
             "B: encoder block (Stem → B1 → B4)의 feature. 피험자 단위 양측 paired Wilcoxon, p는 보정 전 값입니다.", "",
             "| Run | 데이터셋 | N | A: Effective rank 전→후 (p) | A: 평균 \\|r\\| 전→후 (p) | "
             "B: channel \\|r\\| Stem→B4 (p) | B: 호흡대역 비율 Stem→B4 (p) | B: ACF r Stem→B4 (p) |",
             "|---|---|---:|---|---|---|---|---|"]

    def cell(row, before, after, p):
        if before not in row or pd.isna(row.get(before)):
            return "-"
        return f"{row[before]:.3f}→{row[after]:.3f} ({p_text(row[p])})"

    for _, row in key.iterrows():
        lines.append(f"| {row['Run']} | {row['Dataset']} | {row['N_Subjects']} | "
                     f"{cell(row, 'A_EffRank_Before', 'A_EffRank_After', 'A_EffRank_p')} | "
                     f"{cell(row, 'A_OffDiag_AbsR_Before', 'A_OffDiag_AbsR_After', 'A_OffDiag_AbsR_p')} | "
                     f"{cell(row, 'B_ChannelAbsR_Stem', 'B_ChannelAbsR_B4', 'B_ChannelAbsR_p_B4_vs_Stem')} | "
                     f"{cell(row, 'B_RespBandRatio_Stem', 'B_RespBandRatio_B4', 'B_RespBandRatio_p_B4_vs_Stem')} | "
                     f"{cell(row, 'B_ACF_r_Stem', 'B_ACF_r_B4', 'B_ACF_r_p_B4_vs_Stem')} |")
    lines += ["", "Effective rank와 off-diagonal 상관은 채널 분산의 영향도 받으므로, 이 지표만으로 예측 성능의 좋고 나쁨을 판단하지 않습니다."]
    (out_root / "RESULTS.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\nKey results (A: stem fusion before -> after; B: encoder Stem -> B4):")
    print(key.to_string(index=False, float_format=lambda v: f"{v:.3g}"))
    print(f"\nSummary: {out_root}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def check_inputs(run: str, backbone, root: Path, args: argparse.Namespace, out_root: Path) -> pd.DataFrame:
    rows = []
    for dataset_key in [key for key in backbone.DATASET_ORDER if key in args.datasets]:
        label = backbone.DATASET_CONFIGS[dataset_key]["label"]
        tags = result_tags(run, backbone, root, dataset_key)
        if args.subjects:
            tags = [tag for tag in tags if tag in {backbone.safe_tag(s) for s in args.subjects}]
        ready = 0
        for tag in tags:
            models, reason = model_list(run, backbone, root, dataset_key, tag, args.models_261105)
            ready += int(not reason)
            rows += [{"Run": run, "Dataset": label, "Subject": tag, **public(m),
                      "Fold_Reason": reason} for m in models] or [{"Run": run, "Dataset": label, "Subject": tag,
                                                                  "Fold_Reason": reason}]
        print(f"[files] {run} {label}: {ready}/{len(tags)} folds with every model stored", flush=True)
    table = pd.DataFrame(rows)
    out_root.mkdir(parents=True, exist_ok=True)
    table.to_csv(out_root / "input_models.csv", index=False, encoding="utf-8-sig")
    return table


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--runs", nargs="+", choices=list(RUNS), default=list(DEFAULT_RUNS))
    for run in RUNS:
        parser.add_argument(f"--root-{run}", dest=f"root_{run}", default=None,
                            help=f"results folder of {run} (default {RUNS[run]['root']})")
    parser.add_argument("--models-261105", choices=["final", "selected"], default="final",
                        help="261105: final ensemble (6 models, as reported) or m0-m2 of the selected candidate")
    parser.add_argument("--output", type=Path, default=PROJECT_ROOT)
    parser.add_argument("--datasets", nargs="+", choices=["capnobase", "bidmc", "steam2"],
                        default=["capnobase", "bidmc", "steam2"])
    parser.add_argument("--subjects", nargs="+", default=None, help="only these subjects (quick tests)")
    parser.add_argument("--check-inputs-only", action="store_true",
                        help="list the model files per fold (input_models.csv) without data or inference")
    parser.add_argument("--batch-size", type=int, default=48)
    parser.add_argument("--threads", type=int, default=None)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--force-cpu", action="store_true")
    for option in ("capnobase-path", "bidmc-path", "steam2-ppg-dir", "steam2-rsp-dir"):
        parser.add_argument(f"--{option}", type=Path, default=None)
    return parser.parse_args(argv)


def main(args: argparse.Namespace) -> None:
    out_root = Path(args.output).expanduser().resolve()
    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() and not args.force_cpu else "cpu")
    if args.threads:
        torch.set_num_threads(int(args.threads))
    runs = [run for run in RUNS if run in args.runs]
    print(f"Runs: {', '.join(runs)}\nOutput: {out_root}\nDevice: {device}", flush=True)
    tables = {name: [] for name in ("PrePost_Tests", "Encoder_ChannelR", "Encoder_PSD", "Encoder_ACF",
                                    "PrePost_Subjects", "Encoder_Subjects", "Models", "Coverage", "Skipped")}
    for run in runs:
        root = run_root(args, run)
        backbone, script = load_backbone(run, root)
        for dataset_key, key, option in (("capnobase", "data_path", "capnobase_path"), ("bidmc", "data_path", "bidmc_path"),
                                         ("steam2", "ppg_dir", "steam2_ppg_dir"), ("steam2", "rsp_dir", "steam2_rsp_dir")):
            if getattr(args, option):
                backbone.DATASET_CONFIGS[dataset_key][key] = str(Path(getattr(args, option)).resolve())
        run_out = out_root / run
        print(f"\n=== {run}: results {root}\n    script {script}", flush=True)
        if args.check_inputs_only:
            check_inputs(run, backbone, root, args, run_out)
            continue
        collected = collect_run(run, backbone, root, args, device)
        run_out.mkdir(parents=True, exist_ok=True)
        collected["inputs"].to_csv(run_out / "input_models.csv", index=False, encoding="utf-8-sig")
        collected["coverage"].to_csv(run_out / "coverage.csv", index=False, encoding="utf-8-sig")
        collected["skipped"].to_csv(run_out / "skipped_subjects.csv", index=False, encoding="utf-8-sig")
        datasets = [backbone.DATASET_CONFIGS[k]["label"] for k in backbone.DATASET_ORDER if k in args.datasets]
        prepost = prepost_outputs(run, collected, datasets, run_out / PREPOST_FOLDER)
        encoder = encoder_outputs(run, backbone, collected, datasets, run_out / ENCODER_FOLDER)
        if not prepost.empty:
            tables["PrePost_Tests"].append(prepost)
            tables["PrePost_Subjects"].append(pd.read_csv(run_out / PREPOST_FOLDER / "prepost_subjects.csv",
                                                          dtype={"Subject": str, "Run": str}))
        if encoder:
            tables["Encoder_ChannelR"].append(encoder["corr"])
            tables["Encoder_PSD"].append(encoder["psd"])
            tables["Encoder_ACF"].append(encoder["acf"])
            tables["Encoder_Subjects"].append(pd.read_csv(run_out / ENCODER_FOLDER / "encoder_subjects.csv",
                                                          dtype={"Subject": str, "Run": str}))
        tables["Models"].append(collected["inputs"])
        tables["Coverage"].append(collected["coverage"])
        tables["Skipped"].append(collected["skipped"])
        del collected
    if args.check_inputs_only:
        print("Input check complete: no data loaded, no inference run.", flush=True)
        return
    merged = {name: pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
              for name, frames in tables.items()}
    if merged["PrePost_Tests"].empty:
        raise RuntimeError(f"No subject could be analysed; see the skipped_subjects.csv files under {out_root}")
    merged = {"Key_Results": key_results(merged["PrePost_Tests"], merged["Encoder_ChannelR"],
                                         merged["Encoder_PSD"], merged["Encoder_ACF"]), **merged}
    write_summary(out_root / "Summary", merged)


if __name__ == "__main__":
    main(parse_args())
