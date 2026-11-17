# -*- coding: utf-8 -*-
r"""261117_RRWaveNet_Subject_Clustering: how strongly the respiratory waveform shape of each subject is kept,
per dataset, by clustering shape features of the scored windows and comparing the clusters with the subjects.

Waveforms (the same windows, the same features, the same 0.05-1 Hz band for all three)
  GT         the respiratory reference as recorded (CapnoBase capnogram, BIDMC impedance, STEAM2 belt), with the
             source script's alignment (lag, sign) so it lies on the model's time axis
  Target     the waveform RRWaveNet was trained and scored on (BIDMC, STEAM2: the reference band-passed
             0.1-0.5 Hz; CapnoBase: the respiratory-phase waveform built from the expert breath labels)
  RRWaveNet  the final ensemble output (mean of the members' stored test outputs): CapnoBase 261007 variant
             "full"; BIDMC, STEAM2 261013 weight decay selected per fold. Nothing is trained.
  Every subject's output comes from the fold that left this subject out, so the output waveforms (not the
  models' internal features, which live in a different space per fold) are compared.

Windows: the minutes scored with the rescored ground truth (CapnoBase capnogram peak detection, BIDMC ann1,
STEAM2 stored GT; the helpers of 261113), each cut into --window-sec windows (default 30 s = the halves).
A window is kept when all three waveforms give the features. Issue subjects left out as in the final results.

Shape features per window (rate-free: the breathing rate itself is not used unless --include-rate)
  Duty_Cycle              share of samples >= 0.5 after min-max scaling of every 5 s (RespDiff duty cycle)
  Rise_Time_Share         share of samples with a rising slope
  Skewness, Kurtosis      of the z-scored window
  Harmonic_2, Harmonic_3  |FFT| at 2 f0 and 3 f0 / |FFT| at f0 (f0 = largest peak in the recording's rate band;
                          a harmonic above 1 Hz is not measured and takes the dataset median)
  Spectral_Concentration  power within f0 +/- 0.05 Hz / power in 0.05-1 Hz
  Interval_CV, Amplitude_CV  of the breath-to-breath intervals and peak-to-trough amplitudes
  Scaled per dataset and waveform with the median and IQR, clipped at +/- 5.

Per dataset and waveform
  t-SNE (and UMAP when umap-learn is installed) of the windows, coloured by subject
  Clusters: Ward clustering into as many clusters as subjects; cluster vs. subject table
  ARI and NMI of clusters vs. subjects with a permutation test (subject labels shuffled, --permutations),
  purity, and Matched_Share = share of windows in the cluster matched one-to-one to their subject
  (Hungarian matching; the diagonal of the table); ARI_vs_GT_Clusters = ARI of this waveform's clusters
  with the GT's clusters (do the windows group the same way?)
  Feature_ICC: ICC(1) of every feature over the subjects (which features carry the subject)

Outputs (D:\PPG2RSP_RRWaveNet_Inspired\261117_RRWaveNet_Subject_Clustering)
  {Dataset}\{Dataset}_tsne_by_subject.png         GT | Target | RRWaveNet, coloured by subject, subject IDs at
                                                  the subject medians, ARI / NMI / matched share in the titles
  {Dataset}\{Dataset}_umap_by_subject.png         the same with UMAP (when installed)
  {Dataset}\{Dataset}_cluster_vs_subject.png      cluster vs. subject tables (share of each subject's windows)
  Summary_subject_grouping.png                    ARI and NMI per dataset and waveform with the permutation null
  261117_RRWaveNet_Subject_Clustering.xlsx        Summary, Windows, Feature_ICC, Contingency, Missing, Notes

Run: python 261117_RRWaveNet_Subject_Clustering.py     (Spyder: runfile(...), no options needed)
Options: --window-sec, --datasets, --subjects CapnoBase:0332 BIDMC:05 ..., --keep-issue-subjects,
--bidmc-gt (ann1 | ann1_nogap), --include-rate, --embeddings (tsne umap), --perplexity, --permutations, --seed,
--members, --output-root, --root-261007, --root-261013, --script-261007, --script-261013, --script-261113,
--capnobase-path, --bidmc-path, --steam2-ppg-dir, --steam2-rsp-dir
"""

import argparse
import importlib.util
import inspect
import json
import math
import re
import sys
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from scipy.optimize import linear_sum_assignment
from scipy.signal import butter, detrend, find_peaks, sosfiltfilt
from scipy.stats import kurtosis, skew
from sklearn.cluster import AgglomerativeClustering
from sklearn.manifold import TSNE
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score

# ---------------------------------------------------------------------------
# Settings
# ---------------------------------------------------------------------------

NAME = "261117_RRWaveNet_Subject_Clustering"
HERE = Path(__file__).resolve().parent
WORKSPACE = Path(r"D:\PPG2RSP_RRWaveNet_Inspired")
OUTPUT_ROOT = WORKSPACE / NAME
ROOT_261007 = WORKSPACE / "261007_RRWaveNet_NoEvent_v1"
ROOTS_261013 = (WORKSPACE / "261013_RRWavNet_STEAM_BIDMC_v1", WORKSPACE / "261013_RRWaveNet_Adult_SubjectVal_WD")
SCRIPTS = {"261007": "261007_RRWaveNet_Ensemble_NoEvent.py", "261013": "261013_RRWaveNet_Adult_SubjectVal_WD.py",
           "261113": "261113_RRWaveNet_Rescore_ReferenceGT.py"}
WINDOW = "01_w10s_nonoverlap"
DATASETS = ("CapnoBase", "BIDMC", "STEAM2")
KEYS = {"CapnoBase": "capnobase", "BIDMC": "bidmc", "STEAM2": "steam2"}
SOURCE = {"CapnoBase": "261007", "BIDMC": "261013", "STEAM2": "261013"}
GT = {"CapnoBase": "peak", "BIDMC": "ann1", "STEAM2": "original"}
EXCLUDED = {"STEAM2": ("22_LMS", "26_CKK", "12_YJJ"), "BIDMC": ("27",), "CapnoBase": ("0115",)}
MODALITY = {"CapnoBase": "capnogram", "BIDMC": "impedance", "STEAM2": "belt"}
TARGET_TEXT = {"CapnoBase": "respiratory-phase waveform", "BIDMC": "reference 0.1-0.5 Hz",
               "STEAM2": "reference 0.1-0.5 Hz"}
WAVES = ("GT", "Target", "RRWaveNet")
WINDOW_SEC = 30.0
MEMBERS = 3
SHAPE_BAND_HZ = (0.05, 1.0)   # the same band for the three waveforms
DUTY_BLOCK_SEC = 5.0          # RespDiff duty cycle: min-max scaling per 5 s
CONCENTRATION_HZ = 0.05       # spectral concentration: f0 +/- 0.05 Hz
PEAK_DISTANCE = 0.6           # breath peaks at least 0.6 / f0 apart
PEAK_PROMINENCE = 0.5         # in SD of the window
FEATURES = ("Duty_Cycle", "Rise_Time_Share", "Skewness", "Kurtosis", "Harmonic_2", "Harmonic_3",
            "Spectral_Concentration", "Interval_CV", "Amplitude_CV")
SCALE_CLIP = 5.0
PERMUTATIONS = 999
PERPLEXITY = 30.0
SEED = 0
METHOD_NAME = {"tsne": "t-SNE", "umap": "UMAP"}
WAVE_COLOR = {"GT": "#000000", "Target": "#7a7a7a", "RRWaveNet": "#f64636"}
INK, MUTED, GRID = "#1f1f1f", "#6b6a66", "#e4e3df"

# ---------------------------------------------------------------------------
# Scripts used as modules
# ---------------------------------------------------------------------------

_MODULES: Dict[str, object] = {}


def find_script(name: str, given: Optional[Path], roots: Sequence[Path]) -> Path:
    """A script: given, written in a run manifest of the results, or found nearby."""
    candidates: List[Path] = [Path(given)] if given else []
    for root in roots:
        for manifest in sorted(Path(root).glob("run_manifest*.json")):
            try:
                script = json.loads(manifest.read_text(encoding="utf-8")).get("script")
            except (OSError, ValueError, AttributeError):
                continue
            if script and Path(script).name == name:
                candidates.append(Path(script))
    folders = [HERE, HERE.parent, WORKSPACE] + [Path(r) for r in roots]
    if WORKSPACE.is_dir():
        folders += sorted(p for p in WORKSPACE.iterdir() if p.is_dir())
    candidates += [f / name for f in folders]
    for path in candidates:
        if path.is_file():
            return path.resolve()
    raise FileNotFoundError(f"{name} not found (run manifests of {', '.join(map(str, roots))}, {HERE}, {WORKSPACE}); "
                            f"give it with --script-{name[:6]}")


def load_script(path: Path):
    key = str(path)
    if key not in _MODULES:
        name = "subject_cluster_src_" + re.sub(r"\W", "_", path.stem)
        spec = importlib.util.spec_from_file_location(name, key)
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module
        spec.loader.exec_module(module)
        _MODULES[key] = module
    return _MODULES[key]


def root_261013(args) -> Path:
    if args.root_261013:
        return Path(args.root_261013)
    for root in ROOTS_261013:
        if any(root.glob("*/*/BIDMC/Fold_Summaries")) or any(root.glob("*/*/STEAM2/Fold_Summaries")):
            return root
    return next((root for root in ROOTS_261013 if root.is_dir()), ROOTS_261013[0])


def keep_raw_reference(mod) -> None:
    """Keep the reference as recorded (resampled, with the target's lag and sign) in every reference dict.

    The source scripts keep only the band-passed reference; this wraps their preprocess_segment without
    changing what it returns otherwise.
    """
    if getattr(mod, "_keeps_raw_reference", False):
        return
    original = mod.preprocess_segment

    def wrapped(ppg_raw, rsp_raw, fs, spec, segment_id, *args, **kwargs):
        out = original(ppg_raw, rsp_raw, fs, spec, segment_id, *args, **kwargs)
        if out is not None:
            ref = out["reference"]
            raw, _ = mod.sanitize_signal(np.asarray(rsp_raw, dtype=np.float64))
            raw = raw[: len(ref["signal"])]
            lag = int(round(float(ref["diagnostics"].get("Target_Lag_Sec", 0.0)) * fs))
            sign = float(ref["diagnostics"].get("Target_Sign", 1.0))
            ref["raw_reference"] = (sign * mod.shift_signal(raw, lag)).astype(np.float32)
        return out

    mod.preprocess_segment = wrapped
    mod._keeps_raw_reference = True


# ---------------------------------------------------------------------------
# Waveforms and shape features
# ---------------------------------------------------------------------------

def shape_band(values: np.ndarray, fs: float) -> np.ndarray:
    """0.05-1 Hz zero-phase band-pass over every finite run (NaN elsewhere)."""
    x = np.asarray(values, dtype=np.float64)
    out = np.full(x.size, np.nan)
    sos = butter(2, SHAPE_BAND_HZ, btype="band", fs=fs, output="sos")
    ok = np.concatenate([[False], np.isfinite(x), [False]])
    for a, b in np.flatnonzero(ok[1:] != ok[:-1]).reshape(-1, 2):
        if b - a > 3 * fs / SHAPE_BAND_HZ[0]:  # padding of filtfilt needs a long enough run
            out[a:b] = sosfiltfilt(sos, x[a:b] - x[a:b].mean())
    return out


def shape_features(values: np.ndarray, fs: float, rate_band_hz: Tuple[float, float]) -> Optional[Dict[str, float]]:
    """Rate-free shape features of one window (None when the window has no usable breathing)."""
    x = np.asarray(values, dtype=np.float64)
    if x.size < 10 * fs or not np.isfinite(x).all():
        return None
    x = detrend(x)
    sd = float(x.std())
    if sd < 1e-12:
        return None
    z = x / sd
    n = z.size
    nfft = int(2 ** math.ceil(math.log2(max(8 * n, 4096))))
    mag = np.abs(np.fft.rfft(z * np.hanning(n), nfft))
    freq = np.fft.rfftfreq(nfft, 1.0 / fs)
    in_band = (freq >= rate_band_hz[0]) & (freq <= rate_band_hz[1])
    if not in_band.any():
        return None
    f0 = float(freq[in_band][np.argmax(mag[in_band])])
    half = fs / n  # one frequency resolution step of the window

    def amplitude(f: float) -> float:
        if f > SHAPE_BAND_HZ[1]:
            return math.nan
        near = np.abs(freq - f) <= half
        return float(mag[near].max()) if near.any() else math.nan

    a0 = amplitude(f0)
    band = (freq >= SHAPE_BAND_HZ[0]) & (freq <= SHAPE_BAND_HZ[1])
    power = mag ** 2
    block = max(2, int(round(DUTY_BLOCK_SEC * fs)))
    above = []
    for a in range(0, n, block):
        part = z[a:a + block]
        if part.size >= 2 and np.ptp(part) > 1e-12:
            above.append((part - part.min()) / np.ptp(part) >= 0.5)
    peaks, _ = find_peaks(z, distance=max(1, int(PEAK_DISTANCE * fs / f0)), prominence=PEAK_PROMINENCE)
    intervals = np.diff(peaks) / fs
    amps = np.asarray([z[p] - z[q:p + 1].min() for q, p in zip(peaks[:-1], peaks[1:])])
    return {
        "Duty_Cycle": float(np.mean(np.concatenate(above))) if above else math.nan,
        "Rise_Time_Share": float(np.mean(np.diff(z) > 0)),
        "Skewness": float(skew(z)),
        "Kurtosis": float(kurtosis(z)),
        "Harmonic_2": amplitude(2 * f0) / a0 if a0 > 0 else math.nan,
        "Harmonic_3": amplitude(3 * f0) / a0 if a0 > 0 else math.nan,
        "Spectral_Concentration": float(power[band & (np.abs(freq - f0) <= CONCENTRATION_HZ)].sum()
                                        / max(power[band].sum(), 1e-12)),
        "Interval_CV": float(intervals.std() / intervals.mean()) if intervals.size >= 2 else math.nan,
        "Amplitude_CV": float(amps.std() / amps.mean()) if amps.size >= 2 and amps.mean() > 0 else math.nan,
        "Rate_BPM": 60.0 * f0,
    }


def window_spans(start: float, end: float, length: float) -> List[Tuple[float, float]]:
    if length >= end - start - 1e-6:
        return [(start, end)]
    spans, k = [], 0
    while start + (k + 1) * length <= end + 1e-6:
        spans.append((start + k * length, start + (k + 1) * length))
        k += 1
    return spans


# ---------------------------------------------------------------------------
# Windows of one dataset
# ---------------------------------------------------------------------------

def same_subject(a: object, b: object) -> bool:
    a, b = str(a).strip(), str(b).strip()
    return a == b or (a.isdigit() and b.isdigit() and int(a) == int(b))


def parse_subjects(items: Optional[Sequence[str]]) -> Dict[str, List[str]]:
    chosen: Dict[str, List[str]] = {}
    for item in items or []:
        if ":" not in item:
            raise ValueError(f"--subjects takes Dataset:Subject, got {item}")
        dataset, subject = item.split(":", 1)
        match = [d for d in DATASETS if d.lower() == dataset.lower()]
        if not match:
            raise ValueError(f"unknown dataset in {item} ({', '.join(DATASETS)})")
        chosen.setdefault(match[0], []).append(subject)
    return chosen


def dataset_windows(args, dataset: str, chosen: Dict[str, List[str]], rescore) -> Tuple[pd.DataFrame, List[dict]]:
    source = SOURCE[dataset]
    root = Path(args.root_261007) if source == "261007" else root_261013(args)
    script = find_script(SCRIPTS[source], args.script_261007 if source == "261007" else args.script_261013, [root])
    mod = load_script(script)
    keep_raw_reference(mod)
    for key, field, value in (("capnobase", "data_path", args.capnobase_path), ("bidmc", "data_path", args.bidmc_path),
                              ("steam2", "ppg_dir", args.steam2_ppg_dir), ("steam2", "rsp_dir", args.steam2_rsp_dir)):
        if value:
            mod.DATASET_CONFIGS[key][field] = str(value)
    dkey = KEYS[dataset]
    gt_name = args.bidmc_gt if dataset == "BIDMC" else GT[dataset]
    spec = next(s for s in mod.WINDOW_SPECS if f"{s.order:02d}_{s.name}" == WINDOW)
    print(f"\n{dataset}: results {root}; source {script.name}; GT {gt_name}", flush=True)
    data = mod.load_dataset(dkey, spec)
    ann1 = rescore.read_bidmc_ann1(mod) if gt_name.startswith("ann1") else None
    fs = float(mod.TARGET_FS)
    excluded = [] if args.keep_issue_subjects else list(EXCLUDED[dataset])
    rows, missing = [], []
    for subject in sorted(data):
        if dataset in chosen and not any(same_subject(subject, s) for s in chosen[dataset]):
            continue
        if any(same_subject(subject, x) for x in excluded):
            continue
        variant = rescore.final_variant(mod, source, root, spec, dkey, subject)
        if variant is None:
            missing.append({"Dataset": dataset, "Subject": subject, "Missing": "final variant / weight decay selection"})
            continue
        outputs, problem = rescore.stored_outputs(mod, root, variant, spec, dkey, subject, args.members)
        if outputs is None:
            missing.append({"Dataset": dataset, "Subject": subject, "Missing": problem})
            continue
        ens = outputs["ensemble"]
        metadata = mod.test_metadata(data, subject)
        if not (np.array_equal(ens["segment_ids"], [m["segment_id"] for m in metadata])
                and np.array_equal(ens["window_starts"], [m["start"] for m in metadata])):
            missing.append({"Dataset": dataset, "Subject": subject,
                            "Missing": "stored windows differ from the windows of the data read now"})
            continue
        references = data[subject]["references"]
        ann1_labels, spans = None, {}
        if ann1 is not None:
            if subject not in ann1 or len(ann1[subject]) < 2:
                missing.append({"Dataset": dataset, "Subject": subject, "Missing": "no ann1 breaths"})
                continue
            ann1_labels = {sid: rescore.ann1_on_timeline(mod, ann1[subject], ref) for sid, ref in references.items()}
            spans = {sid: rescore.gap_spans(ann1_labels[sid], len(references[sid]["signal"]), rescore.ANN1_GAP_FACTOR)
                     for sid in references}
        refs = rescore.reference_variant(references, "ann1" if gt_name.startswith("ann1") else gt_name, ann1_labels)
        frame = rescore.score(mod, ens, metadata, refs)
        if frame.empty:
            missing.append({"Dataset": dataset, "Subject": subject, "Missing": "no scored minute"})
            continue
        if gt_name == "ann1_nogap":
            frame.loc[rescore.flag_gap_minutes(frame, spans, mod.TARGET_FS), "Ref_Valid"] = False
        scored = frame[frame["Ref_Valid"].astype(bool)]
        n_before = len(rows)
        for segment_id, seg_minutes in scored.groupby("Segment_ID"):
            ref = refs[int(segment_id)]
            pred, _, _ = mod.reconstruct_segment(ens["predictions"], ens["targets"], metadata, int(segment_id),
                                                 mod.TARGET_FS)
            waves = {"GT": shape_band(ref["raw_reference"], fs), "Target": shape_band(ref["target_signal"], fs),
                     "RRWaveNet": shape_band(pred, fs)}
            rate_band = tuple(float(v) / 60.0 for v in ref["profile"]["rr_band_bpm"])
            for _, m in seg_minutes.iterrows():
                for t0, t1 in window_spans(float(m["RR_Window_Start_Sec"]), float(m["RR_Window_End_Sec"]),
                                           args.window_sec):
                    a, b = int(round(t0 * fs)), int(round(t1 * fs))
                    feats = {w: shape_features(v[a:b], fs, rate_band) if b <= len(v) else None
                             for w, v in waves.items()}
                    if any(f is None for f in feats.values()):
                        continue
                    for w in WAVES:
                        rows.append({"Dataset": dataset, "Subject": str(subject), "Segment_ID": int(segment_id),
                                     "Window_Start_Sec": t0, "Window_End_Sec": t1,
                                     "Minute_Start_Sec": float(m["RR_Window_Start_Sec"]), "Waveform": w,
                                     "Minute_AE": float(m["AE_Wave_FFT"]), "Minute_PCC": float(m["Wave_PCC"]),
                                     **feats[w]})
        print(f"  {subject}: {len(scored)} scored minutes, {(len(rows) - n_before) // len(WAVES)} windows", flush=True)
    del data
    return pd.DataFrame(rows), missing


# ---------------------------------------------------------------------------
# Clustering against the subjects
# ---------------------------------------------------------------------------

def robust_scale(frame: pd.DataFrame) -> np.ndarray:
    x = frame.to_numpy(dtype=np.float64)
    med = np.nanmedian(x, axis=0)
    x = np.where(np.isfinite(x), x, med)  # a missing value (e.g. a harmonic above 1 Hz) takes the median
    iqr = np.nanpercentile(x, 75, axis=0) - np.nanpercentile(x, 25, axis=0)
    iqr = np.where(iqr > 1e-12, iqr, np.nanstd(x, axis=0))
    iqr = np.where(iqr > 1e-12, iqr, 1.0)
    return np.clip((x - med) / iqr, -SCALE_CLIP, SCALE_CLIP)


def purity(labels: np.ndarray, clusters: np.ndarray) -> float:
    table = pd.crosstab(clusters, labels)
    return float(table.max(axis=1).sum() / table.to_numpy().sum())


def matched_order(table: pd.DataFrame) -> Tuple[List[object], float]:
    """Clusters matched one-to-one to the subjects (Hungarian); order of the clusters and matched share."""
    rows, cols = linear_sum_assignment(-table.to_numpy())
    match = {table.columns[c]: table.index[r] for r, c in zip(rows, cols)}
    order = [match[s] for s in table.columns if s in match]
    order += [c for c in table.index if c not in order]
    share = float(table.to_numpy()[rows, cols].sum() / table.to_numpy().sum())
    return order, share


def icc1(values: np.ndarray, groups: np.ndarray) -> float:
    frame = pd.DataFrame({"v": values, "g": groups}).dropna()
    g = frame.groupby("g")["v"]
    k = g.size()
    k = k[k >= 2]
    frame = frame[frame["g"].isin(k.index)]
    if len(k) < 2:
        return math.nan
    g = frame.groupby("g")["v"]
    n, total = len(k), int(k.sum())
    msb = float((k * (g.mean() - frame["v"].mean()) ** 2).sum() / (n - 1))
    msw = float(((frame["v"] - g.transform("mean")) ** 2).sum() / (total - n))
    k0 = (total - float((k ** 2).sum()) / total) / (n - 1)
    denom = msb + (k0 - 1.0) * msw
    return (msb - msw) / denom if denom > 0 else math.nan


def embed(x: np.ndarray, method: str, args) -> Optional[np.ndarray]:
    n = x.shape[0]
    if n < 5:
        return None
    if method == "tsne":
        params = inspect.signature(TSNE).parameters
        kwargs = {"n_components": 2, "perplexity": float(min(args.perplexity, max(2.0, (n - 1) / 3.0))),
                  "init": "pca", "random_state": args.seed}
        kwargs["learning_rate"] = "auto" if "learning_rate" in params and _sklearn_auto_lr() else 200.0
        return TSNE(**kwargs).fit_transform(x)
    try:
        import umap  # optional: umap-learn
    except ImportError:
        return None
    return umap.UMAP(n_neighbors=int(min(15, n - 1)), min_dist=0.1, random_state=args.seed).fit_transform(x)


def _sklearn_auto_lr() -> bool:
    import sklearn
    major, minor = (int(v) for v in sklearn.__version__.split(".")[:2])
    return (major, minor) >= (1, 0)


def analyse(frame: pd.DataFrame, features: Sequence[str], args) -> Tuple[List[dict], dict, List[dict], List[dict]]:
    """Clusters, metrics, embeddings and tables of one dataset (every waveform)."""
    rng = np.random.default_rng(args.seed)
    summary, extra, icc_rows, tables = [], {}, [], []
    dataset = frame["Dataset"].iloc[0]
    clusters_by_wave = {}
    for wave in WAVES:
        g = frame[frame["Waveform"] == wave].reset_index(drop=True)
        labels = g["Subject"].to_numpy()
        subjects = sorted(set(labels))
        cols = [c for c in features if g[c].notna().any()]
        x = robust_scale(g[cols])
        row = {"Dataset": dataset, "Waveform": wave, "Subjects": len(subjects), "Windows": len(g),
               "Windows_per_Subject": len(g) / max(1, len(subjects)), "Features": ", ".join(cols)}
        for c in features:
            icc_rows.append({"Dataset": dataset, "Waveform": wave, "Feature": c, "ICC": icc1(g[c].to_numpy(), labels),
                             "Median": float(g[c].median()), "IQR": float(g[c].quantile(0.75) - g[c].quantile(0.25))})
        if len(subjects) < 2 or len(g) <= len(subjects):
            row["Note"] = "fewer than 2 subjects or not more windows than subjects: no clustering"
            summary.append(row)
            continue
        clusters = AgglomerativeClustering(n_clusters=len(subjects), linkage="ward").fit_predict(x)
        clusters_by_wave[wave] = clusters
        ari, nmi = adjusted_rand_score(labels, clusters), normalized_mutual_info_score(labels, clusters)
        null_ari, null_nmi = np.empty(args.permutations), np.empty(args.permutations)
        for i in range(args.permutations):
            shuffled = rng.permutation(labels)
            null_ari[i] = adjusted_rand_score(shuffled, clusters)
            null_nmi[i] = normalized_mutual_info_score(shuffled, clusters)
        table = pd.crosstab(pd.Series(clusters, name="Cluster"), pd.Series(labels, name="Subject"))
        table = table.reindex(columns=subjects, fill_value=0)
        order, share = matched_order(table)
        for c in table.index:
            for s in subjects:
                if table.at[c, s]:
                    tables.append({"Dataset": dataset, "Waveform": wave, "Cluster": int(c),
                                   "Cluster_Order": order.index(c), "Subject": s, "Windows": int(table.at[c, s])})
        row.update({"ARI": ari, "ARI_Null_Mean": float(null_ari.mean()),
                    "ARI_Null_95": float(np.percentile(null_ari, 95)),
                    "ARI_p": float((1 + np.sum(null_ari >= ari)) / (1 + args.permutations)),
                    "NMI": nmi, "NMI_Null_Mean": float(null_nmi.mean()),
                    "NMI_Null_95": float(np.percentile(null_nmi, 95)),
                    "NMI_p": float((1 + np.sum(null_nmi >= nmi)) / (1 + args.permutations)),
                    "Purity": purity(labels, clusters), "Matched_Share": share,
                    "Chance_Share": float(pd.Series(labels).value_counts().max() / len(labels))})
        coords = {m: embed(x, m, args) for m in args.embeddings}
        extra[wave] = {"table": table, "order": order, "coords": coords, "labels": labels, "clusters": clusters,
                       "row": row}
        frame.loc[frame["Waveform"] == wave, "Cluster"] = clusters
        for m, xy in coords.items():
            if xy is not None:
                frame.loc[frame["Waveform"] == wave, f"{m.upper()}_1"] = xy[:, 0]
                frame.loc[frame["Waveform"] == wave, f"{m.upper()}_2"] = xy[:, 1]
        summary.append(row)
    for row in summary:
        if "GT" in clusters_by_wave and row["Waveform"] in clusters_by_wave:
            row["ARI_vs_GT_Clusters"] = adjusted_rand_score(clusters_by_wave["GT"], clusters_by_wave[row["Waveform"]])
    return summary, extra, icc_rows, tables


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------

def subject_colors(subjects: Sequence[str]) -> Dict[str, tuple]:
    """One colour per subject (up to 60 distinct before repeating): enough to see whether a subject's windows
    lie together; the cluster table and the metrics are the quantitative read."""
    base = [plt.get_cmap(n)(i) for n in ("tab20", "tab20b", "tab20c") for i in range(20)]
    return {s: base[i % len(base)] for i, s in enumerate(subjects)}


def style(ax) -> None:
    ax.grid(True, color=GRID, lw=0.7)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.tick_params(labelsize=8)


def metric_text(row: dict) -> str:
    return (f"ARI {row['ARI']:.2f} (chance {row['ARI_Null_Mean']:.2f}, p {row['ARI_p']:.3f}) · "
            f"NMI {row['NMI']:.2f} (chance {row['NMI_Null_Mean']:.2f})\nmatched share {row['Matched_Share']:.0%} · "
            f"purity {row['Purity']:.0%}" + (f" · ARI vs GT clusters {row['ARI_vs_GT_Clusters']:.2f}"
                                               if row["Waveform"] != "GT" and "ARI_vs_GT_Clusters" in row else ""))


def wave_label(dataset: str, wave: str) -> str:
    return {"GT": f"GT ({MODALITY[dataset]} as recorded)", "Target": f"Target ({TARGET_TEXT[dataset]})",
            "RRWaveNet": "RRWaveNet (final ensemble output)"}[wave]


def draw_embedding(path: Path, dataset: str, method: str, extra: dict, window_sec: float) -> bool:
    waves = [w for w in WAVES if w in extra and extra[w]["coords"].get(method) is not None]
    if not waves:
        return False
    subjects = sorted(set(extra[waves[0]]["labels"]))
    colors = subject_colors(subjects)
    fig, axes = plt.subplots(1, len(waves), figsize=(6.0 * len(waves), 6.6))
    axes = np.atleast_1d(axes)
    for ax, w in zip(axes, waves):
        xy, labels = extra[w]["coords"][method], extra[w]["labels"]
        for s in subjects:
            sel = labels == s
            ax.scatter(xy[sel, 0], xy[sel, 1], s=16, color=colors[s], alpha=0.85, lw=0)
            mx, my = np.median(xy[sel, 0]), np.median(xy[sel, 1])
            ax.text(mx, my, s, fontsize=6.2, color=INK, ha="center", va="center", alpha=0.85)
        style(ax)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_title(wave_label(dataset, w), loc="left", fontsize=10, color=INK, pad=32)
        ax.text(0.0, 1.012, metric_text(extra[w]["row"]), transform=ax.transAxes, fontsize=7.6, color=MUTED,
                va="bottom", linespacing=1.4)
    fig.suptitle(f"{dataset}: {METHOD_NAME[method]} of the shape features of every {window_sec:g} s window of the scored "
                 f"minutes, one colour per subject ({len(subjects)} subjects; labels at the subject medians)",
                 x=0.01, ha="left", fontsize=11, color=INK)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=160)
    plt.close(fig)
    return True


def draw_tables(path: Path, dataset: str, extra: dict) -> bool:
    waves = [w for w in WAVES if w in extra]
    if not waves:
        return False
    fig, axes = plt.subplots(1, len(waves), figsize=(6.0 * len(waves), 6.4))
    axes = np.atleast_1d(axes)
    fig.subplots_adjust(left=0.04, right=0.93, bottom=0.12, top=0.86, wspace=0.12)
    image = None
    for ax, w in zip(axes, waves):
        table = extra[w]["table"]
        share = table.div(table.sum(axis=0), axis=1).reindex(extra[w]["order"])  # share of each subject's windows
        image = ax.imshow(share.to_numpy(), aspect="auto", cmap="Blues", vmin=0, vmax=1, interpolation="nearest")
        ax.set_xlabel(f"Subject ({table.shape[1]})", fontsize=9)
        if ax is axes[0]:
            ax.set_ylabel("Cluster (matched one-to-one to the subjects, in their order)", fontsize=9)
        if table.shape[1] <= 60:
            ax.set_xticks(range(table.shape[1]))
            ax.set_xticklabels(table.columns, rotation=90, fontsize=5.5)
        ax.set_yticks([])
        ax.set_title(wave_label(dataset, w), loc="left", fontsize=10, color=INK, pad=24)
        row = extra[w]["row"]
        ax.text(0.0, 1.01, f"diagonal = windows in the cluster matched to their subject: matched share "
                           f"{row['Matched_Share']:.0%} · ARI {row['ARI']:.2f} · NMI {row['NMI']:.2f}",
                transform=ax.transAxes, fontsize=7.6, color=MUTED, va="bottom")
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
    cbar = fig.colorbar(image, cax=fig.add_axes([0.945, 0.12, 0.01, 0.74]))
    cbar.set_label("Share of the subject's windows", fontsize=8.5)
    fig.suptitle(f"{dataset}: clusters (Ward, as many as subjects) vs. subjects; a strong diagonal = the windows of "
                 f"each subject form their own cluster", x=0.01, ha="left", fontsize=11, color=INK)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=160)
    plt.close(fig)
    return True


def draw_summary(path: Path, summary: pd.DataFrame) -> bool:
    s = summary.dropna(subset=["ARI"]) if "ARI" in summary else pd.DataFrame()
    if s.empty:
        return False
    datasets = [d for d in DATASETS if d in set(s["Dataset"])]
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 1.6 + 1.2 * len(datasets)), sharey=True)
    for ax, metric in zip(axes, ("ARI", "NMI")):
        for i, d in enumerate(datasets):
            for j, w in enumerate(WAVES):
                r = s[(s["Dataset"] == d) & (s["Waveform"] == w)]
                if r.empty:
                    continue
                r = r.iloc[0]
                y = i + (j - 1) * 0.22
                ax.plot([r[f"{metric}_Null_Mean"], r[f"{metric}_Null_95"]], [y, y], color="#c9c9c9", lw=5,
                        solid_capstyle="butt")
                ax.plot(r[metric], y, "o", color=WAVE_COLOR[w], ms=7)
                ax.annotate(f"{r[metric]:.2f}", (r[metric], y), xytext=(7, 0), textcoords="offset points",
                            fontsize=7.5, color=INK, va="center")
        ax.set_yticks(range(len(datasets)))
        ax.set_yticklabels(datasets)
        ax.set_xlabel(f"{metric} of clusters vs. subjects (grey: chance, mean to 95th percentile of shuffled labels)",
                      fontsize=8.5)
        style(ax)
    axes[0].set_ylim(len(datasets) - 0.5, -0.5)  # first dataset on top (the y axis is shared)
    fig.legend([Line2D([], [], marker="o", ls="", color=WAVE_COLOR[w]) for w in WAVES]
               + [Line2D([], [], color="#c9c9c9", lw=5)], list(WAVES) + ["chance"], loc="upper right",
               ncol=4, frameon=False, fontsize=8.5)
    fig.suptitle("How strongly the windows group by subject: GT, training target and RRWaveNet output",
                 x=0.01, ha="left", fontsize=11, color=INK)
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    fig.savefig(path, dpi=160)
    plt.close(fig)
    return True


# ---------------------------------------------------------------------------
# Run
# ---------------------------------------------------------------------------

def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Clustering of the scored windows' shape features against the subjects")
    p.add_argument("--window-sec", type=float, default=WINDOW_SEC, help="window length in s (default 30; 60 = minute)")
    p.add_argument("--datasets", nargs="+", choices=list(DATASETS), default=list(DATASETS))
    p.add_argument("--subjects", nargs="+", default=None, help="Dataset:Subject (default: every subject)")
    p.add_argument("--keep-issue-subjects", action="store_true")
    p.add_argument("--bidmc-gt", choices=("ann1", "ann1_nogap"), default="ann1")
    p.add_argument("--include-rate", action="store_true", help="also cluster on the breathing rate")
    p.add_argument("--embeddings", nargs="+", choices=("tsne", "umap"), default=["tsne", "umap"])
    p.add_argument("--perplexity", type=float, default=PERPLEXITY)
    p.add_argument("--permutations", type=int, default=PERMUTATIONS)
    p.add_argument("--seed", type=int, default=SEED)
    p.add_argument("--members", type=int, default=MEMBERS)
    p.add_argument("--output-root", type=Path, default=OUTPUT_ROOT)
    p.add_argument("--root-261007", type=Path, default=ROOT_261007)
    p.add_argument("--root-261013", type=Path, default=None)
    p.add_argument("--script-261007", type=Path, default=None)
    p.add_argument("--script-261013", type=Path, default=None)
    p.add_argument("--script-261113", type=Path, default=None)
    p.add_argument("--capnobase-path", type=Path, default=None)
    p.add_argument("--bidmc-path", type=Path, default=None)
    p.add_argument("--steam2-ppg-dir", type=Path, default=None)
    p.add_argument("--steam2-rsp-dir", type=Path, default=None)
    return p.parse_args(list(sys.argv[1:] if argv is None else argv))


def main(argv: Optional[Sequence[str]] = None) -> None:
    args = parse_args(argv)
    chosen = parse_subjects(args.subjects)
    datasets = [d for d in args.datasets if not chosen or d in chosen]
    out_root = Path(args.output_root)
    out_root.mkdir(parents=True, exist_ok=True)
    rescore = load_script(find_script(SCRIPTS["261113"], args.script_261113, []))
    features = list(FEATURES) + (["Rate_BPM"] if args.include_rate else [])
    if "umap" in args.embeddings and importlib.util.find_spec("umap") is None:
        print("UMAP skipped: umap-learn is not installed (pip install umap-learn); t-SNE is drawn", flush=True)
    windows, summary, iccs, tables, missing, figures = [], [], [], [], [], []
    for dataset in datasets:
        frame, miss = dataset_windows(args, dataset, chosen, rescore)
        missing += miss
        if frame.empty:
            print(f"{dataset}: no window", flush=True)
            continue
        rows, extra, icc_rows, table_rows = analyse(frame, features, args)
        summary += rows
        iccs += icc_rows
        tables += table_rows
        windows.append(frame)
        folder = out_root / dataset
        for method in args.embeddings:
            path = folder / f"{dataset}_{method}_by_subject.png"
            if draw_embedding(path, dataset, method, extra, args.window_sec):
                figures.append(path)
        path = folder / f"{dataset}_cluster_vs_subject.png"
        if draw_tables(path, dataset, extra):
            figures.append(path)
        for r in rows:
            if "ARI" in r:
                print(f"  {dataset} {r['Waveform']:9s}: {r['Subjects']} subjects, {r['Windows']} windows | "
                      f"ARI {r['ARI']:.3f} (chance {r['ARI_Null_Mean']:.3f}, p {r['ARI_p']:.3f}) | NMI {r['NMI']:.3f} "
                      f"(chance {r['NMI_Null_Mean']:.3f}) | matched share {r['Matched_Share']:.0%}"
                      + (f" | ARI vs GT clusters {r['ARI_vs_GT_Clusters']:.3f}" if r["Waveform"] != "GT" else ""),
                      flush=True)
            else:
                print(f"  {dataset} {r['Waveform']}: {r.get('Note', '')}", flush=True)
    summary_frame = pd.DataFrame(summary)
    if draw_summary(out_root / "Summary_subject_grouping.png", summary_frame):
        figures.append(out_root / "Summary_subject_grouping.png")
    notes = pd.DataFrame([
        ("Waveforms", "GT = reference as recorded (resampled, the source's lag and sign); Target = the training / scoring "
                      "target (BIDMC, STEAM2: reference 0.1-0.5 Hz; CapnoBase: respiratory-phase waveform from the "
                      "expert breath labels); RRWaveNet = final ensemble output (CapnoBase 261007 full; BIDMC, STEAM2 "
                      "261013 weight decay selected per fold); all three band-passed 0.05-1 Hz (zero phase)"),
        ("Windows", f"{args.window_sec:g} s windows of the minutes scored with the rescored GT (CapnoBase peak, BIDMC "
                    f"{args.bidmc_gt}, STEAM2 stored GT); kept when all three waveforms give the features"),
        ("Features", "Duty_Cycle (RespDiff: share >= 0.5 after min-max per 5 s), Rise_Time_Share, Skewness, Kurtosis "
                     "(excess), Harmonic_2 / Harmonic_3 (|FFT| at 2f0 / 3f0 over f0; above 1 Hz: dataset median), "
                     "Spectral_Concentration (f0 +/- 0.05 Hz over 0.05-1 Hz), Interval_CV, Amplitude_CV (breath peaks "
                     "at least 0.6 / f0 apart, prominence 0.5 SD)" + ("; Rate_BPM" if args.include_rate else
                                                                     "; the breathing rate is not used")),
        ("Scaling", "per dataset and waveform: (value - median) / IQR, clipped at +/- 5"),
        ("Clusters", "Ward agglomerative clustering into as many clusters as subjects"),
        ("ARI, NMI", f"clusters vs. subjects; chance = the same with the subject labels shuffled ({args.permutations} "
                     "times); p = share of shuffles at least as high"),
        ("Purity", "share of windows whose cluster's most frequent subject is their own"),
        ("Matched_Share", "share of windows in the cluster matched one-to-one to their subject (Hungarian matching; "
                          "the diagonal of the cluster vs. subject figure); Chance_Share = largest subject's share"),
        ("ARI_vs_GT_Clusters", "ARI of this waveform's clusters with the GT's clusters on the same windows: whether the "
                               "windows group the same way as in the GT"),
        ("Feature_ICC", "ICC(1) of each feature over the subjects (one-way random effects, unequal windows)"),
        ("Embeddings", f"t-SNE (perplexity {args.perplexity:g}, PCA init, seed {args.seed}) and UMAP (umap-learn, 15 "
                       "neighbours) of the scaled features; only for viewing: distances between groups and group sizes "
                       "in these maps carry no meaning"),
        ("Caution", "windows of one subject are adjacent in time and alike for that reason alone: compare the waveforms "
                    "(GT vs. Target vs. RRWaveNet) on the same windows rather than read one value alone"),
        ("Subjects", "issue subjects left out (CapnoBase 0115, BIDMC 27, STEAM2 12_YJJ, 22_LMS, 26_CKK)"
                     if not args.keep_issue_subjects else "every subject (issue subjects kept)"),
    ], columns=["Item", "Description"])
    path = out_root / f"{NAME}.xlsx"
    window_table = pd.concat(windows, ignore_index=True) if windows else pd.DataFrame()
    with pd.ExcelWriter(path, engine="openpyxl") as writer:
        summary_frame.round(4).to_excel(writer, sheet_name="Summary", index=False)
        window_table.round(5).to_excel(writer, sheet_name="Windows", index=False)
        pd.DataFrame(iccs).round(4).to_excel(writer, sheet_name="Feature_ICC", index=False)
        pd.DataFrame(tables).to_excel(writer, sheet_name="Contingency", index=False)
        pd.DataFrame(missing, columns=["Dataset", "Subject", "Missing"]).to_excel(writer, sheet_name="Missing", index=False)
        notes.to_excel(writer, sheet_name="Notes", index=False)
    print(f"\nfigures: {len(figures)} under {out_root}; table: {path}")


if __name__ == "__main__":
    main()
