# -*- coding: utf-8 -*-
r"""261118_StemFusion_Seed_Variability_v2: the stem fusion compared seed against seed. Every LOSOCV fold gets the
same number of seeds K (default 5: m0-m4) for both models, Fusion O (with the 1x1 stem fusion) and Fusion X (without),
in every dataset; the subject is then judged on all K vs K seeds instead of one model (m0) against one model.

Why v2: in v1 Fusion X had one seed (m0) in BIDMC and STEAM2 while Fusion O had three, and a subject was judged
from m0 against m0. A change of a single seed pair cannot be told apart from the seed: in v1, 0018 and 0332
(CapnoBase) looked like large improvements, but one Fusion X seed of 0018 failed (the other two matched Fusion O)
and one Fusion O seed of 0332 happened to succeed (the mean of the three Fusion O seeds was worse).

Seeds and models (nothing that exists is trained again)
  The seed of member k is the seed formula of 261007 / 261013 / 261028 (stable_seed('261006-ensemble-member',
  'w10s_nonoverlap', dataset, subject, k)), so m0-m2 are the stored models and m3, m4, ... continue the series.
  A seed changes the initial weights, the batch order, the training jitter and dropout; data, validation, weight
  decay and every setting stay the same. Fusion O and Fusion X of a seed k share that seed.
  Stored members are read from (in this order)
    CapnoBase      261007_RRWaveNet_NoEvent_v1\{full | no_stem_fusion}  (m0-m2 of both models)
    BIDMC, STEAM2  Fusion O: 261013_RRWavNet_STEAM_BIDMC_v1\full_wd{weight decay selected in the fold} (m0-m2)
                   Fusion X: 261028_Stem_ablation_noise_v1\no_stem_fusion (m0)
    then the models trained by this script (v2\Trained_Models).
  Missing members are trained with run_member of 261028_Stem_ablation_noise_v1 (loaded as a module), i.e. exactly
  the training of the stored models: CapnoBase with the 261007 settings (temporal validation, weight decay 5e-5),
  BIDMC and STEAM2 with the 261013 settings (subject-wise validation subjects of 261013, the weight decay 261013
  selected in the fold, convolution weights only); up to 240 epochs, at least 20, patience 25, best epoch restored.
  Training runs in stages so that a stopped run is still balanced: first every model up to 3 seeds (3 vs 3), then
  up to K. Finished members are never trained again (rerun to continue; interrupted members continue from their
  last epoch). The MAE issue subjects are not trained (--keep-issue-subjects trains and keeps them).

GT and scored minutes: the rescored GT of 261118_StemFusion_Threshold_Groups_Waveforms (loaded as a module):
CapnoBase peak detection on the capnogram (RRest rule), BIDMC ann1 (expert-label rule), STEAM2 stored GT. All
members of a subject are scored on the same minutes (scored and present in every member table).
Subject values of a member: MAE = mean |stored rate - GT breath count|; |PCC| = mean |Wave_PCC|.

Analysis, for MAE (lower is better) and |PCC| (higher is better)
  Seed noise         |value m_a - value m_b| of every pair of seeds of the same model and subject; quantiles per
                     dataset (50 / 75 / 90 / 95 %, max)
  Fusion vs seed     |value O m_k - value X m_k| of the same seed k, its quantiles next to those of the seed noise
                     (same size = the stem fusion moves a subject no more than a new seed does)
  Per subject (K vs K seeds)
    Mean / median / SD / range of the K seeds of each model; Diff_Mean = mean O - mean X; Diff_m0 (as v1)
    Exact permutation test of the K + K seed values (every split of the 2K values into two groups of K):
      p_O_better = share of splits with a difference at least as favourable to the first group as observed,
      p_O_worse the opposite; with K = 3 the smallest p is 1/20 = 0.05 (all three O seeds better than all three
      X seeds), with K = 5 it is 1/252 = 0.004
    Class: improved if p_O_better <= --alpha (0.05), worsened if p_O_worse <= alpha, else within seed noise;
    q values (Benjamini-Hochberg over the subjects of a dataset) are reported
    Separation_All_K: all O seeds better than all X seeds ("O") or the opposite ("X"); Separation_3v3: the same
    with m0-m2 only; Seeds_O_Better: seeds k where O m_k is better than X m_k
  Per seed (does the conclusion depend on the seed?)
    For every seed k, and for the mean of the seeds: mean over the subjects of O and X, subjects with O better /
    worse, Wilcoxon signed-rank test over the subjects (paired by subject)

Outputs (D:\PPG2RSP_RRWaveNet_Inspired\261118_StemFusion_Seed_Variability\v2)
  Trained_Models\{full | no_stem_fusion}\01_w10s_nonoverlap\{dataset}\   members trained here (261028 layout)
  Training_Log.csv                                                         every member trained here, its time
  261118_StemFusion_Seed_Variability_v2.xlsx
    Summary        per metric and dataset: subjects, seeds per model, classes, separations, mean O / X
    Subjects       every subject: every seed of O and X (MAE, |PCC|), mean, median, SD, range, Diff_Mean, Diff_m0,
                   p values, q values, class, separations
    Seed_Noise     seed-pair quantiles per metric and dataset; minute share with seeds > 1 breath/min apart
    Fusion_vs_Seed quantiles of |O m_k - X m_k| next to the seed-pair quantiles
    Seed_Level     per metric, dataset and seed (and the seed mean): O, X, difference, subjects better / worse,
                   Wilcoxon p
    Seed_Pairs, Fusion_Pairs, Members (where every member was read from), Minutes, Missing, Notes
  Figures (PNG and PDF)
    StemFusion_{MAE|PCC}_Slope_KvK_AllDatasets / _AllSubjects   seed mean Without -> With Fusion per subject, the
        seed range (min-max) as a vertical bar at both ends; red = improved, blue = worsened (K vs K test),
        gray = within seed noise
    StemFusion_{MAE|PCC}_Seed_vs_Fusion_ECDF   distribution of |difference| between seeds of one model and between
        O and X of the same seed, per dataset
    StemFusion_{MAE|PCC}_Per_Seed   mean over the subjects of O and X for every seed, per dataset

Run: python 261118_StemFusion_Seed_Variability_v2.py   (Spyder: runfile(...), no options needed; it trains the
     missing members first - a long run on the GPU, stoppable and resumable - then analyses)
     261118_StemFusion_Threshold_Groups_Waveforms.py and 261028_Stem_ablation_noise_v1.py are found next to this
     script, one folder up, in their own folders of D:\PPG2RSP_RRWaveNet_Inspired or any folder of it.
Options: --members K (5), --no-train (analyse the members that exist), --shard i/n (parallel workers: every n-th
training job from i), --gpu, --allow-cpu, --alpha (0.05), --datasets, --subjects (quick tests),
--keep-issue-subjects, --ann1-nogap, --epoch-cap (quick tests), --root-261007, --root-261013, --root-noise,
--output-root, --script-261118, --script-261028, --script-261007, --script-261013, --capnobase-path, --bidmc-path,
--steam2-ppg-dir, --steam2-rsp-dir
"""

import argparse
import importlib.util
import itertools
import json
import math
import sys
import time
from functools import lru_cache
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from scipy.stats import wilcoxon

# ---------------------------------------------------------------------------
# Settings
# ---------------------------------------------------------------------------

NAME = "261118_StemFusion_Seed_Variability_v2"
HERE = Path(__file__).resolve().parent
BASE_SCRIPT = "261118_StemFusion_Threshold_Groups_Waveforms.py"
TRAIN_SCRIPT = "261028_Stem_ablation_noise_v1.py"
WORKSPACE = Path(r"D:\PPG2RSP_RRWaveNet_Inspired")
OUTPUT_ROOT = WORKSPACE / "261118_StemFusion_Seed_Variability" / "v2"
MEMBERS = 5
STAGE_MEMBERS = 3  # first stage: every model up to 3 seeds (3 vs 3)
ALPHA = 0.05
QUANTILES = (0.5, 0.75, 0.9, 0.95)
SPREAD_BPM = 1.0
DATASETS = ("CapnoBase", "BIDMC", "STEAM2")
MODELS = {"O": "full", "X": "no_stem_fusion"}
LABELS = {"O": "Fusion O (with stem fusion)", "X": "Fusion X (without stem fusion)"}
CLASSES = ("improved", "worsened", "within seed noise")
METRICS = {
    "MAE": {"label": "MAE (bpm)", "word": "MAE", "unit": "bpm", "sign": -1, "target": 3.0,
            "target_label": "3 bpm target", "rule": "MAE < 3", "ylim": None},
    "PCC": {"label": "|PCC|", "word": "|PCC|", "unit": "", "sign": 1, "target": 0.6,
            "target_label": "|PCC| 0.6 target", "rule": "|PCC| > 0.6", "ylim": (0.0, 1.0)},
}
RED, BLUE, GRAY = "#ed665a", "#559ed5", "#b4b2ad"  # class colors: red = improved with stem fusion, blue = worsened
MODEL_COLORS = {"X": "#ed665a", "O": "#559ed5"}  # model colors of the box plots: Without red, With Fusion blue
CLASS_COLOR = {"improved": RED, "worsened": BLUE, "within seed noise": GRAY}
TARGET, INK, MUTED, GRID = "#a8823c", "#1f1f1f", "#6b6a66", "#e6e5e1"


# ---------------------------------------------------------------------------
# Source scripts
# ---------------------------------------------------------------------------

def script_candidates(name: str, given: Optional[Path]) -> List[Path]:
    """Where a script may be: given, next to this script, one folder up, in its own folder of the workspace, in the
    workspace, the working folder, then any folder of the workspace (newest first)."""
    stem = Path(name).stem
    paths = ([Path(given)] if given else []) + [HERE / name, HERE.parent / name, HERE.parent / stem / name,
                                                 HERE.parent.parent / stem / name, WORKSPACE / stem / name,
                                                 WORKSPACE / name, Path.cwd() / name]
    if WORKSPACE.is_dir():
        found = list(WORKSPACE.glob(f"*/{name}")) + list(WORKSPACE.glob(f"*/*/{name}"))
        paths += sorted(found, key=lambda p: p.stat().st_mtime, reverse=True)
    return paths


def load_module(name: str, given: Optional[Path], module_name: str, option: str):
    tried = script_candidates(name, given)
    for path in tried:
        if path.is_file():
            print(f"Using {path}", flush=True)
            spec = importlib.util.spec_from_file_location(module_name, str(path))
            module = importlib.util.module_from_spec(spec)
            sys.modules[spec.name] = module
            spec.loader.exec_module(module)
            return module
    raise FileNotFoundError(f"{name} not found (looked in: " + "; ".join(str(p.parent) for p in tried[:7])
                            + f" and the folders of {WORKSPACE}); give it with {option}")


def source_module(base, args, dataset: str):
    """Preprocessing module of the dataset (261007 for CapnoBase, 261013 for BIDMC / STEAM2) with the data paths."""
    source = base.SOURCE[dataset]
    roots = [args.root_261007] if source == "261007" else [args.root_261013]
    script = base.find_script(base.SCRIPTS[source], args.script_261007 if source == "261007" else args.script_261013,
                              roots)
    mod = base.load_script(script)
    set_paths(mod, args)
    return mod, script


def set_paths(mod, args) -> None:
    for key, field, value in (("capnobase", "data_path", args.capnobase_path), ("bidmc", "data_path", args.bidmc_path),
                              ("steam2", "ppg_dir", args.steam2_ppg_dir), ("steam2", "rsp_dir", args.steam2_rsp_dir)):
        if value:
            mod.DATASET_CONFIGS[key][field] = str(value)


# ---------------------------------------------------------------------------
# Stored members
# ---------------------------------------------------------------------------

def member_roots(base, args, tag: str, dataset: str, model: str) -> List[Path]:
    """Folders that may hold a member of a model and fold: the stored results first, then this script's models."""
    variant = MODELS[model]
    roots = []
    if base.SOURCE[dataset] == "261007":
        roots.append(Path(args.root_261007) / variant / base.WINDOW / dataset)
    elif model == "O":
        record = base.read_json(Path(args.root_261013) / "WD_Selection" / base.WINDOW / dataset
                                / f"{tag}_selection.json") or {}
        if record.get("Selected_Variant"):
            roots.append(Path(args.root_261013) / record["Selected_Variant"] / base.WINDOW / dataset)
    roots.append(Path(args.root_noise) / variant / base.WINDOW / dataset)
    roots.append(Path(args.output_root) / "Trained_Models" / variant / base.WINDOW / dataset)
    return roots


def find_member(base, args, tag: str, dataset: str, model: str, member: int) -> Optional[Path]:
    """Root of a complete member: its minute table, and its summary (written last) or its predictions."""
    for root in member_roots(base, args, tag, dataset, model):
        summaries = root / "Fold_Summaries"
        if (summaries / f"{tag}_m{member}_RR60s_windows.csv").exists() and (
                (summaries / f"{tag}_m{member}_summary.json").exists()
                or (root / "Members" / f"m{member}" / "Predictions" / f"{tag}_test_outputs.npz").exists()):
            return root
    return None


def is_issue(base, dataset: str, subject: str) -> bool:
    return any(base.same_subject(subject, x) for x in base.EXCLUDED[dataset])


# ---------------------------------------------------------------------------
# Training of the missing members (261028 run_member)
# ---------------------------------------------------------------------------

def elapsed_estimate(base, args, dataset: str, tags: Sequence[str]) -> float:
    """Median training time (s) of the stored members of a dataset (their summaries), NaN when unknown."""
    times = []
    for tag in tags[:12]:
        for model in MODELS:
            for member in range(3):
                root = find_member(base, args, tag, dataset, model, member)
                if root is None:
                    continue
                data = base.read_json(root / "Fold_Summaries" / f"{tag}_m{member}_summary.json") or {}
                if np.isfinite(float(data.get("Elapsed_Sec", np.nan) or np.nan)):
                    times.append(float(data["Elapsed_Sec"]))
    return float(np.median(times)) if times else math.nan


def train_missing(base, args, log_rows: List[dict]) -> None:
    """Train every missing member up to --members seeds, in stages (3 seeds first, then K)."""
    shard_index, shard_count = parse_shard(args.shard)
    stages = sorted({min(STAGE_MEMBERS, args.members), args.members})
    mod28 = load_module(TRAIN_SCRIPT, args.script_261028, "seed_v2_train_261028", "--script-261028")
    set_paths(mod28, args)
    import torch  # available through 261028

    use_cuda = torch.cuda.is_available()
    device = torch.device(f"cuda:{args.gpu}" if use_cuda else "cpu")
    argv = ["--results-root", str(Path(args.output_root) / "Trained_Models"), "--reference-261007-root",
            str(args.root_261007), "--reference-261013-root", str(args.root_261013), "--resume"]
    if args.epoch_cap:
        argv += ["--epoch-cap", str(args.epoch_cap)]
    args28 = mod28.parse_args(argv)
    args28.resume = True
    if not hasattr(args28, "save_features"):
        args28.save_features = False
    spec28 = next(s for s in mod28.WINDOW_SPECS if f"{s.order:02d}_{s.name}" == base.WINDOW)
    train_root = Path(args.output_root) / "Trained_Models"
    log_path = Path(args.output_root) / "Training_Log.csv"
    for stage in stages:
        print(f"\n=== Training stage: every model up to {stage} seeds (m0-m{stage - 1}) ===", flush=True)
        for dataset in args.datasets:
            key = base.KEYS[dataset]
            print(f"\n{dataset}: reading the data with {TRAIN_SCRIPT}", flush=True)
            data = mod28.load_dataset(key, spec28)
            all_subjects = sorted(data)
            subjects = [s for s in all_subjects if (not args.subjects or any(base.same_subject(s, x) for x in args.subjects))
                        and (args.keep_issue_subjects or not is_issue(base, dataset, s))]
            trainings, jobs = {}, []
            for subject in subjects:
                others = [s for s in all_subjects if s != subject]
                try:
                    trainings[subject] = mod28.fold_training(args28, spec28, key, subject, others)
                except mod28.SelectionMissing as exc:
                    print(f"  {exc}; not trained", flush=True)
                    continue
                tag = mod28.safe_tag(subject)
                for member in range(stage):
                    for model in MODELS:
                        if find_member(base, args, tag, dataset, model, member) is None:
                            jobs.append((member, subject, model))
            jobs.sort(key=lambda j: (j[0], all_subjects.index(j[1]), j[2]))
            mine = [job for index, job in enumerate(jobs) if index % shard_count == shard_index]
            estimate = elapsed_estimate(base, args, dataset, [mod28.safe_tag(s) for s in subjects])
            eta = f", about {len(mine) * estimate / 3600:.1f} h at {estimate / 60:.1f} min each" if np.isfinite(estimate) else ""
            print(f"{dataset}: {len(jobs)} members to train up to {stage} seeds"
                  + (f", {len(mine)} in this worker ({args.shard})" if shard_count > 1 else "") + eta, flush=True)
            if mine and not use_cuda and not args.allow_cpu:
                print(f"{dataset}: no CUDA GPU - training skipped (use --allow-cpu to train on the CPU)", flush=True)
                mine = []
            for n, (member, subject, model) in enumerate(mine, 1):
                start = time.time()
                status = "trained"
                try:
                    mod28.run_member(args28, train_root, mod28.VARIANTS[MODELS[model]], spec28, key, data, all_subjects,
                                     subject, member, device, trainings[subject])
                except Exception as exc:  # one failed member must not stop the others
                    if mod28.is_fatal_error(exc):
                        print(f"  fatal error ({exc}); run again to continue", flush=True)
                        raise
                    status = f"failed: {exc!r}"
                    print(f"  {dataset} {subject} {model} m{member}: {status}", flush=True)
                elapsed = time.time() - start
                log_rows.append({"Dataset": dataset, "Subject": subject, "Model": LABELS[model], "Member": member,
                                 "Seed": int(mod28.member_seed(spec28, key, subject, member)), "Status": status,
                                 "Elapsed_Min": round(elapsed / 60.0, 2), "Finished": time.strftime("%Y-%m-%d %H:%M:%S")})
                pd.DataFrame(log_rows).to_csv(log_path, index=False, encoding="utf-8-sig")
                print(f"  [{n}/{len(mine)}] {dataset} {subject} {LABELS[model]} m{member}: {status} "
                      f"({elapsed / 60:.1f} min)", flush=True)
                mod28.release_memory(device)
            del data
            mod28.release_memory(device)


def parse_shard(value: Optional[str]) -> Tuple[int, int]:
    if not value:
        return 0, 1
    i, n = (int(v) for v in str(value).split("/"))
    if not (n >= 1 and 0 <= i < n):
        raise ValueError("--shard must be i/n with 0 <= i < n")
    return i, n


# ---------------------------------------------------------------------------
# Minutes and GT
# ---------------------------------------------------------------------------

def read_minutes(path: Path) -> pd.DataFrame:
    t = pd.read_csv(path, dtype={"Subject": str}, encoding="utf-8-sig")
    t["Segment_ID"] = t["Segment_ID"].astype(int)
    t["RR_Window_Start_Sec"] = t["RR_Window_Start_Sec"].astype(float).round(3)
    if "Wave_PCC" not in t.columns:
        t["Wave_PCC"] = np.nan
    return t.set_index(["Segment_ID", "RR_Window_Start_Sec"])


def gt_minutes(base, args, mod, dataset: str, record: dict, table: pd.DataFrame,
               ann1: Optional[np.ndarray]) -> pd.DataFrame:
    """Rescored GT of every minute of the grid (the rules of 261118)."""
    fs = mod.TARGET_FS
    rows = []
    for (segment_id, t0), o in table.iterrows():
        ref = record["references"].get(int(segment_id))
        if ref is None:
            continue
        t1 = float(o["RR_Window_End_Sec"])
        a, b = int(round(float(t0) * fs)), int(round(t1 * fs))
        if dataset == "STEAM2":
            valid, reason, count = bool(o["Ref_Valid"]), str(o.get("Ref_Exclusion_Reason", "") or ""), int(o["GT_Breath_Count"])
        else:
            breaths, label_based = base.gt_breaths(dataset, mod, ref, ann1)
            valid, reason, count = base.minute_gt(mod, ref, a, b, breaths, label_based)
            if dataset == "BIDMC" and args.ann1_nogap and valid and any(
                    lo < b and hi > a for lo, hi in base.gap_spans(breaths, len(ref["signal"]))):
                valid, reason = False, "ref_ann1_gap"
        rows.append({"Segment_ID": int(segment_id), "Start_Sec": float(t0), "End_Sec": t1, "Scored": valid,
                     "Reason": "" if valid else reason, "GT_Count": count})
    return pd.DataFrame(rows).set_index(["Segment_ID", "Start_Sec"]) if rows else pd.DataFrame()


# ---------------------------------------------------------------------------
# Statistics
# ---------------------------------------------------------------------------

@lru_cache(maxsize=None)
def splits(n_first: int, n_total: int) -> np.ndarray:
    """Every way to choose the first group (indices) out of n_total values."""
    return np.array(list(itertools.combinations(range(n_total), n_first)), dtype=np.int16)


def permutation_p(o: np.ndarray, x: np.ndarray, sign: int) -> Tuple[float, float]:
    """Exact permutation test of the difference of means, both one-sided p values: (O better, O worse)."""
    if len(o) < 2 or len(x) < 2:
        return math.nan, math.nan
    values = np.concatenate([o, x])
    idx = splits(len(o), len(values))
    total = values.sum()
    first = values[idx].sum(axis=1)
    diff = sign * (first / len(o) - (total - first) / len(x))  # > 0: the first group is better
    observed = sign * (o.mean() - x.mean())
    eps = 1e-12 * max(1.0, abs(observed))
    return float(np.mean(diff >= observed - eps)), float(np.mean(diff <= observed + eps))


def bh(p: Sequence[float]) -> np.ndarray:
    p = np.asarray(p, float)
    out = np.full(len(p), np.nan)
    ok = np.isfinite(p)
    if not ok.any():
        return out
    v = p[ok]
    order = np.argsort(v)
    ranked = v[order] * len(v) / (np.arange(len(v)) + 1)
    ranked = np.minimum.accumulate(ranked[::-1])[::-1]
    q = np.empty(len(v))
    q[order] = np.minimum(ranked, 1.0)
    out[ok] = q
    return out


def wilcoxon_p(a: np.ndarray, b: np.ndarray) -> float:
    d = np.asarray(a, float) - np.asarray(b, float)
    d = d[np.isfinite(d)]
    if len(d) < 3 or np.all(d == 0):
        return math.nan
    try:
        return float(wilcoxon(d).pvalue)
    except ValueError:
        return math.nan


def better(a: float, b: float, sign: int) -> bool:
    return sign * (a - b) > 0


def quantile_row(values: pd.Series) -> dict:
    row = {f"Q{int(round(q * 100))}": float(values.quantile(q)) if len(values) else math.nan for q in QUANTILES}
    row["Max"] = float(values.max()) if len(values) else math.nan
    row["N"] = int(len(values))
    return row


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------

def style(ax, show_ylabel: bool, label: str) -> None:
    ax.yaxis.grid(True, color=GRID, lw=0.9)
    ax.set_axisbelow(True)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    ax.tick_params(axis="y", labelsize=10, length=0)
    if show_ylabel:
        ax.set_ylabel(label, fontsize=12)


def slope_panel(ax, d: pd.DataFrame, metric: str, title: str, show_ylabel: bool, small: bool, ylim, k: int) -> None:
    spec = METRICS[metric]
    cls = f"Class_{metric}"
    for r in sorted(d.to_dict("records"), key=lambda r: (r[cls] != "within seed noise", abs(r[f"Diff_Mean_{metric}"]))):
        gray = r[cls] == "within seed noise"
        c = CLASS_COLOR[r[cls]]
        ys = [r[f"Mean_X_{metric}"], r[f"Mean_O_{metric}"]]
        for xpos, m in ((0, "X"), (1, "O")):
            lo, hi = r[f"Min_{m}_{metric}"], r[f"Max_{m}_{metric}"]
            if np.isfinite(lo) and np.isfinite(hi) and hi > lo:
                ax.plot([xpos, xpos], [lo, hi], color=c, lw=3.2, alpha=0.18 if gray else 0.35, zorder=0.8,
                        solid_capstyle="round")
        ax.plot([0, 1], ys, color=c, lw=1.6, alpha=0.6 if gray else 0.85, zorder=1 if gray else 2,
                solid_capstyle="round")
        ax.plot([0, 1], ys, ls="none", marker="o", ms=6.5, color=c, mec="white", mew=0.8, zorder=3 if gray else 4)
    ax.axhline(spec["target"], color=TARGET, lw=1.6, ls=(0, (4, 3)), zorder=1.5)
    ax.set_xlim(-0.2, 1.2)
    ax.set_ylim(*ylim)
    ax.set_xticks([0, 1])
    ax.set_xticklabels(["Without", "With Fusion"], fontsize=11.5 if small else 12)
    ax.tick_params(axis="x", length=0, pad=8)
    style(ax, show_ylabel, f"{spec['label']}, mean of {k} seeds")
    counts = d[cls].value_counts()
    sep_o = int((d[f"Separation_All_K_{metric}"] == "O").sum())
    sep_x = int((d[f"Separation_All_K_{metric}"] == "X").sum())
    ax.set_title(title, fontsize=12, loc="left", color=INK, pad=24)
    ax.text(0.0, 1.02, f"improved {int(counts.get('improved', 0))} · worsened {int(counts.get('worsened', 0))} · within "
                       f"seed noise {int(counts.get('within seed noise', 0))} · all seeds O better {sep_o}, X better {sep_x}",
            transform=ax.transAxes, fontsize=8.0 if small else 8.6, color=MUTED, va="bottom")


def draw_slopes(subjects: pd.DataFrame, metric: str, k: int, alpha: float, out_root: Path) -> None:
    spec = METRICS[metric]
    d = subjects.dropna(subset=[f"Mean_X_{metric}", f"Mean_O_{metric}"])
    if d.empty:
        return
    top = max(d[f"Max_X_{metric}"].max(), d[f"Max_O_{metric}"].max())
    ylim = spec["ylim"] or (0.0, max(10.0, float(math.ceil(top))))
    handles = [Line2D([], [], color=RED, lw=1.6, marker="o", ms=6.5, mec="white"),
               Line2D([], [], color=BLUE, lw=1.6, marker="o", ms=6.5, mec="white"),
               Line2D([], [], color=GRAY, lw=1.6, marker="o", ms=6.5, mec="white"),
               Line2D([], [], color=GRAY, lw=3.2, alpha=0.4),
               Line2D([], [], color=TARGET, lw=1.6, ls=(0, (4, 3)))]
    labels = [f"Improved with stem fusion ({k} vs {k} seeds, p ≤ {alpha:g})",
              f"Worsened with stem fusion ({k} vs {k} seeds, p ≤ {alpha:g})", "Within seed noise",
              "Seed range of a model (min-max)", spec["target_label"]]
    foot = (f"Point = mean of the {k} seeds of a model; test = exact permutation of the {k} + {k} seed values "
            f"(one-sided); rescored GT minutes; MAE issue subjects excluded")
    fig, ax = plt.subplots(figsize=(7.8, 6.6))
    fig.patch.set_facecolor("white")
    sizes = ", ".join(f"{n} {int((d.Dataset == n).sum())}" for n in DATASETS if (d.Dataset == n).any())
    slope_panel(ax, d, metric, f"All subjects (n = {len(d)}: {sizes})", True, False, ylim, k)
    fig.legend(handles, labels, loc="upper center", ncol=2, frameon=False, fontsize=8.4, bbox_to_anchor=(0.5, 1.0))
    fig.text(0.5, 0.012, foot, ha="center", fontsize=7.4, color=MUTED)
    fig.subplots_adjust(left=0.11, right=0.97, top=0.79, bottom=0.11)
    save(fig, out_root / f"StemFusion_{metric}_Slope_KvK_AllSubjects")
    names = [n for n in DATASETS if (d.Dataset == n).any()]
    fig, axes = plt.subplots(1, len(names), figsize=(4.6 * len(names), 5.8), sharey=True, squeeze=False)
    fig.patch.set_facecolor("white")
    for i, (ax, name) in enumerate(zip(axes[0], names)):
        s = d[d.Dataset == name]
        slope_panel(ax, s, metric, f"{name} (n = {len(s)})", i == 0, True, ylim, k)
    three = len(names) > 2
    fig.legend(handles, labels, loc="upper center", ncol=3 if three else 2, frameon=False, fontsize=9.0 if three else 8.4,
               bbox_to_anchor=(0.5, 1.0))
    fig.text(0.5, 0.012, foot, ha="center", fontsize=8.0, color=MUTED)
    fig.subplots_adjust(left=0.06 if three else 0.08, right=0.985, top=0.78 if three else 0.74, bottom=0.11,
                        wspace=0.12)
    save(fig, out_root / f"StemFusion_{metric}_Slope_KvK_AllDatasets")


def draw_ecdf(seed_pairs: pd.DataFrame, fusion_pairs: pd.DataFrame, metric: str, out_root: Path) -> None:
    spec = METRICS[metric]
    names = [n for n in DATASETS if (seed_pairs.Dataset == n).any() or (fusion_pairs.Dataset == n).any()]
    if not names:
        return
    fig, axes = plt.subplots(1, len(names), figsize=(4.6 * len(names), 4.4), sharey=True, squeeze=False)
    fig.patch.set_facecolor("white")
    for i, (ax, name) in enumerate(zip(axes[0], names)):
        for frame, color, ls, label in ((seed_pairs, "#8f8d88", (0, (4, 2)), "between seeds of one model"),
                                        (fusion_pairs, INK, "-", "O vs X of the same seed")):
            v = np.sort(frame[(frame.Dataset == name) & (frame.Metric == metric)].Abs_Diff.to_numpy(float))
            if len(v):
                ax.step(v, np.arange(1, len(v) + 1) / len(v), where="post", color=color, lw=1.8, ls=ls,
                        label=f"{label} (n = {len(v)})")
        ax.set_xscale("symlog", linthresh=0.1 if metric == "MAE" else 0.01)
        ax.set_ylim(0, 1.02)
        style(ax, i == 0, "Share of pairs ≤ x")
        ax.set_xlabel(f"|Δ{spec['word']}|" + (f" ({spec['unit']})" if spec["unit"] else ""), fontsize=10.5)
        ax.set_title(name, fontsize=12, loc="left", color=INK)
        ax.legend(frameon=False, fontsize=8.4, loc="upper left")
    fig.text(0.5, 0.01, f"Subject {spec['word']}: difference between two seeds of the same model (seed noise) and "
                        "between Fusion O and Fusion X trained with the same seed", ha="center", fontsize=8.2, color=MUTED)
    fig.subplots_adjust(left=0.07, right=0.985, top=0.9, bottom=0.2, wspace=0.1)
    save(fig, out_root / f"StemFusion_{metric}_Seed_vs_Fusion_ECDF")


def draw_per_seed(level: pd.DataFrame, metric: str, out_root: Path) -> None:
    spec = METRICS[metric]
    lv = level[(level.Metric == metric) & level.Seed.astype(str).str.startswith("m")]
    names = [n for n in DATASETS if (lv.Dataset == n).any()]
    if not names:
        return
    fig, axes = plt.subplots(1, len(names), figsize=(4.6 * len(names), 4.4), squeeze=False)
    fig.patch.set_facecolor("white")
    for i, (ax, name) in enumerate(zip(axes[0], names)):
        s = lv[lv.Dataset == name].copy()
        s["k"] = s.Seed.str[1:].astype(int)
        s = s.sort_values("k")
        for m in ("X", "O"):
            ax.plot(s.k, s[f"Mean_{m}"], color=MODEL_COLORS[m], lw=1.8, marker="o", ms=6, mec="white",
                    label=LABELS[m])
        for r in s.itertuples():
            if np.isfinite(r.Wilcoxon_p) and r.Wilcoxon_p < 0.05:
                ax.annotate("*", (r.k, max(r.Mean_O, r.Mean_X)), textcoords="offset points", xytext=(0, 4),
                            ha="center", fontsize=12, color=INK)
        ax.set_xticks(s.k)
        ax.set_xticklabels([f"m{v}" for v in s.k])
        style(ax, i == 0, f"Mean subject {spec['word']}")
        ax.set_xlabel("Seed", fontsize=10.5)
        ax.set_title(name, fontsize=12, loc="left", color=INK)
    fig.legend([Line2D([], [], color=MODEL_COLORS[m], lw=1.8, marker="o", ms=6, mec="white") for m in ("X", "O")],
               [LABELS[m] for m in ("X", "O")], loc="upper center", ncol=2, frameon=False, fontsize=9.0,
               bbox_to_anchor=(0.5, 1.0))
    fig.text(0.5, 0.01, f"Mean over the subjects of the subject {spec['word']} of every seed; * Wilcoxon signed-rank "
                        "p < 0.05 (O vs X over the subjects, paired by subject)", ha="center", fontsize=8.2, color=MUTED)
    fig.subplots_adjust(left=0.07, right=0.985, top=0.82, bottom=0.2, wspace=0.22)
    save(fig, out_root / f"StemFusion_{metric}_Per_Seed")


def save(fig, stem: Path) -> None:
    fig.savefig(stem.with_suffix(".png"), dpi=300)
    fig.savefig(stem.with_suffix(".pdf"))
    plt.close(fig)


# ---------------------------------------------------------------------------
# Analysis
# ---------------------------------------------------------------------------

def analyse(base, args) -> None:
    out_root = Path(args.output_root)
    k_max = args.members
    subject_rows, seed_pairs, fusion_pairs, member_rows, minute_frames, missing, shares = [], [], [], [], [], [], []
    for dataset in args.datasets:
        mod, script = source_module(base, args, dataset)
        spec = next(s for s in mod.WINDOW_SPECS if f"{s.order:02d}_{s.name}" == base.WINDOW)
        print(f"\n{dataset}: analysis (preprocessing of {script.name})", flush=True)
        data = mod.load_dataset(base.KEYS[dataset], spec)
        ann1_all = base.read_ann1(mod) if dataset == "BIDMC" else {}
        for subject in sorted(data):
            if args.subjects and not any(base.same_subject(subject, x) for x in args.subjects):
                continue
            issue = is_issue(base, dataset, subject)
            if issue and not args.keep_issue_subjects:
                continue
            tag = mod.safe_tag(subject)
            frames = {}
            for model in MODELS:
                for member in range(k_max):
                    root = find_member(base, args, tag, dataset, model, member)
                    member_rows.append({"Dataset": dataset, "Subject": subject, "Model": LABELS[model],
                                        "Member": f"m{member}", "Read_From": str(root) if root else "missing"})
                    if root is not None:
                        frames[(model, member)] = read_minutes(root / "Fold_Summaries" / f"{tag}_m{member}_RR60s_windows.csv")
            have = {m: sorted(k for (mm, k) in frames if mm == m) for m in MODELS}
            if not have["O"] or not have["X"]:
                missing.append({"Dataset": dataset, "Subject": subject,
                                "Missing": "no member of " + " and ".join(LABELS[m] for m in MODELS if not have[m])})
                continue
            grid = frames[("O", have["O"][0])]
            gt = gt_minutes(base, args, mod, dataset, data[subject], grid, ann1_all.get(subject))
            if gt.empty:
                missing.append({"Dataset": dataset, "Subject": subject, "Missing": "no minute with a reference"})
                continue
            common = gt.index
            for frame in frames.values():
                common = common.intersection(frame.index)
            scored = gt.loc[common]
            scored = scored[scored["Scored"]]
            if scored.empty:
                missing.append({"Dataset": dataset, "Subject": subject, "Missing": "no scored minute in every member"})
                continue
            keys = scored.index
            count = scored["GT_Count"].to_numpy(float)
            rates = {key: f["Pred_Wave_FFT_RR"].astype(float) for key, f in frames.items()}
            pccs = {key: f["Wave_PCC"].astype(float) for key, f in frames.items()}
            values = {"MAE": {key: float(np.mean(np.abs(r.loc[keys].to_numpy(float) - count))) for key, r in rates.items()},
                      "PCC": {key: (float(np.nanmean(np.abs(p.loc[keys].to_numpy(float))))
                                    if np.isfinite(p.loc[keys].to_numpy(float)).any() else math.nan)
                              for key, p in pccs.items()}}
            row = {"Dataset": dataset, "Subject": subject, "Issue_Subject": issue, "Scored_Minutes": int(len(keys)),
                   "Seeds_O": len(have["O"]), "Seeds_X": len(have["X"])}
            for m in MODELS:
                members = have[m]
                v = np.column_stack([rates[(m, k)].loc[keys].to_numpy(float) for k in members])
                shares.append({"Dataset": dataset, "Model": m, "N": len(keys),
                               "Share": float(np.mean(v.max(axis=1) - v.min(axis=1) > SPREAD_BPM)) if len(members) > 1
                               else math.nan})
            for metric, val in values.items():
                sign = METRICS[metric]["sign"]
                arr = {}
                for m in MODELS:
                    for k in range(k_max):
                        row[f"{metric}_{m}_m{k}"] = val.get((m, k), math.nan)
                    arr[m] = np.array([val[(m, k)] for k in have[m] if np.isfinite(val[(m, k)])])
                    a = arr[m]
                    row[f"Mean_{m}_{metric}"] = float(a.mean()) if len(a) else math.nan
                    row[f"Median_{m}_{metric}"] = float(np.median(a)) if len(a) else math.nan
                    row[f"SD_{m}_{metric}"] = float(a.std(ddof=1)) if len(a) > 1 else math.nan
                    row[f"Min_{m}_{metric}"] = float(a.min()) if len(a) else math.nan
                    row[f"Max_{m}_{metric}"] = float(a.max()) if len(a) else math.nan
                    for ka, kb in itertools.combinations([k for k in have[m] if np.isfinite(val[(m, k)])], 2):
                        seed_pairs.append({"Metric": metric, "Dataset": dataset, "Subject": subject, "Model": LABELS[m],
                                           "Member_a": f"m{ka}", "Member_b": f"m{kb}", "Value_a": val[(m, ka)],
                                           "Value_b": val[(m, kb)], "Abs_Diff": abs(val[(m, ka)] - val[(m, kb)])})
                paired = [k for k in range(k_max) if np.isfinite(val.get(("O", k), np.nan))
                          and np.isfinite(val.get(("X", k), np.nan))]
                for k in paired:
                    fusion_pairs.append({"Metric": metric, "Dataset": dataset, "Subject": subject, "Member": f"m{k}",
                                         "Value_O": val[("O", k)], "Value_X": val[("X", k)],
                                         "Diff_O_minus_X": val[("O", k)] - val[("X", k)],
                                         "Abs_Diff": abs(val[("O", k)] - val[("X", k)])})
                row[f"Diff_Mean_{metric}"] = row[f"Mean_O_{metric}"] - row[f"Mean_X_{metric}"]
                row[f"Diff_m0_{metric}"] = row[f"{metric}_O_m0"] - row[f"{metric}_X_m0"]
                p_better, p_worse = permutation_p(arr["O"], arr["X"], sign)
                row[f"p_O_better_{metric}"], row[f"p_O_worse_{metric}"] = p_better, p_worse
                row[f"Seeds_O_Better_{metric}"] = int(sum(better(val[("O", k)], val[("X", k)], sign) for k in paired))
                row[f"Seeds_Paired_{metric}"] = len(paired)
                o, x = arr["O"], arr["X"]
                if len(o) and len(x):
                    o_best, o_worst = (o.min(), o.max()) if sign < 0 else (o.max(), o.min())
                    x_best, x_worst = (x.min(), x.max()) if sign < 0 else (x.max(), x.min())
                    row[f"Separation_All_K_{metric}"] = ("O" if better(o_worst, x_best, sign) else
                                                         "X" if better(x_worst, o_best, sign) else "none")
                else:
                    row[f"Separation_All_K_{metric}"] = "n/a"
                three = [k for k in range(3) if k in have["O"] and k in have["X"]]
                if len(three) == 3:
                    o3 = np.array([val[("O", k)] for k in three])
                    x3 = np.array([val[("X", k)] for k in three])
                    o_worst = o3.max() if sign < 0 else o3.min()
                    x_worst = x3.max() if sign < 0 else x3.min()
                    o_best = o3.min() if sign < 0 else o3.max()
                    x_best = x3.min() if sign < 0 else x3.max()
                    row[f"Separation_3v3_{metric}"] = ("O" if better(o_worst, x_best, sign) else
                                                       "X" if better(x_worst, o_best, sign) else "none")
                else:
                    row[f"Separation_3v3_{metric}"] = "n/a"
            subject_rows.append(row)
            mf = scored.reset_index().assign(Dataset=dataset, Subject=subject)
            for (m, k), r in rates.items():
                mf[f"Rate_{m}_m{k}"] = r.loc[keys].to_numpy(float)
                mf[f"PCC_{m}_m{k}"] = pccs[(m, k)].loc[keys].to_numpy(float)
            minute_frames.append(mf)
            print(f"  {subject}: seeds O {len(have['O'])}, X {len(have['X'])}; MAE mean O {row['Mean_O_MAE']:.2f} "
                  f"(range {row['Min_O_MAE']:.2f}-{row['Max_O_MAE']:.2f}), X {row['Mean_X_MAE']:.2f} "
                  f"({row['Min_X_MAE']:.2f}-{row['Max_X_MAE']:.2f}); p(O better) {row['p_O_better_MAE']:.3f}, "
                  f"p(O worse) {row['p_O_worse_MAE']:.3f}", flush=True)
        del data
    subjects = pd.DataFrame(subject_rows)
    seed_pairs = pd.DataFrame(seed_pairs, columns=["Metric", "Dataset", "Subject", "Model", "Member_a", "Member_b",
                                                   "Value_a", "Value_b", "Abs_Diff"])
    fusion_pairs = pd.DataFrame(fusion_pairs, columns=["Metric", "Dataset", "Subject", "Member", "Value_O", "Value_X",
                                                       "Diff_O_minus_X", "Abs_Diff"])
    shares = pd.DataFrame(shares)
    if subjects.empty:
        print("No subject with members of both models.")
        return
    alpha = float(args.alpha)
    for metric in METRICS:
        subjects[f"Class_{metric}"] = np.where(subjects[f"p_O_better_{metric}"] <= alpha, "improved",
                                               np.where(subjects[f"p_O_worse_{metric}"] <= alpha, "worsened",
                                                        "within seed noise"))
        observed = np.where(subjects[f"Diff_Mean_{metric}"] * METRICS[metric]["sign"] > 0,
                            subjects[f"p_O_better_{metric}"], subjects[f"p_O_worse_{metric}"])
        subjects[f"p_Observed_Direction_{metric}"] = observed
        subjects[f"q_{metric}"] = np.nan
        for d in subjects.Dataset.unique():
            sel = subjects.Dataset == d
            subjects.loc[sel, f"q_{metric}"] = bh(subjects.loc[sel, f"p_Observed_Direction_{metric}"])
    # dataset tables
    names = [d for d in args.datasets if (subjects.Dataset == d).any()] + ["All"]
    summary_rows, noise_rows, compare_rows, level_rows = [], [], [], []
    for metric, spec in METRICS.items():
        sign = spec["sign"]
        for name in names:
            s = subjects if name == "All" else subjects[subjects.Dataset == name]
            sp = seed_pairs[(seed_pairs.Metric == metric) & ((seed_pairs.Dataset == name) if name != "All" else True)]
            fp = fusion_pairs[(fusion_pairs.Metric == metric) & ((fusion_pairs.Dataset == name) if name != "All" else True)]
            counts = s[f"Class_{metric}"].value_counts()
            summary_rows.append({
                "Metric": metric, "Dataset": name, "Subjects": len(s),
                "Seeds_O_Median": float(s.Seeds_O.median()), "Seeds_X_Median": float(s.Seeds_X.median()),
                "Subjects_K_vs_K": int(((s.Seeds_O >= args.members) & (s.Seeds_X >= args.members)).sum()),
                "Improved": int(counts.get("improved", 0)), "Worsened": int(counts.get("worsened", 0)),
                "Within_Seed_Noise": int(counts.get("within seed noise", 0)),
                "Improved_q_le_alpha": int(((s[f"q_{metric}"] <= alpha) & (s[f"Class_{metric}"] == "improved")).sum()),
                "Worsened_q_le_alpha": int(((s[f"q_{metric}"] <= alpha) & (s[f"Class_{metric}"] == "worsened")).sum()),
                "All_Seeds_O_Better": int((s[f"Separation_All_K_{metric}"] == "O").sum()),
                "All_Seeds_X_Better": int((s[f"Separation_All_K_{metric}"] == "X").sum()),
                "Sep_3v3_O_Better": int((s[f"Separation_3v3_{metric}"] == "O").sum()),
                "Sep_3v3_X_Better": int((s[f"Separation_3v3_{metric}"] == "X").sum()),
                "Mean_O": float(s[f"Mean_O_{metric}"].mean()), "Mean_X": float(s[f"Mean_X_{metric}"].mean()),
                "Seed_Mean_Wilcoxon_p": wilcoxon_p(s[f"Mean_O_{metric}"], s[f"Mean_X_{metric}"]),
                f"Meets_{spec['rule'].replace(' ', '')}_X": int(((s[f"Mean_X_{metric}"] < spec["target"]) if sign < 0
                                                                   else (s[f"Mean_X_{metric}"] > spec["target"])).sum()),
                f"Meets_{spec['rule'].replace(' ', '')}_O": int(((s[f"Mean_O_{metric}"] < spec["target"]) if sign < 0
                                                                   else (s[f"Mean_O_{metric}"] > spec["target"])).sum()),
            })
            row = {"Metric": metric, "Dataset": name, "Seed_Pairs": len(sp),
                   "Seed_Pairs_O": int((sp.Model == LABELS["O"]).sum()), "Seed_Pairs_X": int((sp.Model == LABELS["X"]).sum())}
            row.update({f"Seed_AbsDiff_{k}": v for k, v in quantile_row(sp.Abs_Diff).items() if k != "N"})
            if metric == "MAE":
                for m in MODELS:
                    sh = shares[(shares.Model == m) & ((shares.Dataset == name) if name != "All" else True)].dropna()
                    row[f"Minutes_Rate_Spread_gt1_{m}"] = (float((sh.Share * sh.N).sum() / sh.N.sum())
                                                          if len(sh) and sh.N.sum() else math.nan)
            noise_rows.append(row)
            comp = {"Metric": metric, "Dataset": name}
            for label, frame in (("Seed", sp), ("Fusion", fp)):
                for k, v in quantile_row(frame.Abs_Diff).items():
                    comp[f"{label}_{k}"] = v
            for q in QUANTILES:
                key = f"Q{int(round(q * 100))}"
                comp[f"Ratio_{key}"] = comp[f"Fusion_{key}"] / comp[f"Seed_{key}"] if comp[f"Seed_{key}"] else math.nan
            compare_rows.append(comp)
            for k in list(range(args.members)) + ["mean"]:
                if k == "mean":
                    o, x = s[f"Mean_O_{metric}"], s[f"Mean_X_{metric}"]
                    label = "seed mean"
                else:
                    o, x = s[f"{metric}_O_m{k}"], s[f"{metric}_X_m{k}"]
                    label = f"m{k}"
                ok = o.notna() & x.notna()
                if not ok.any():
                    continue
                diff = (o - x)[ok]
                level_rows.append({"Metric": metric, "Dataset": name, "Seed": label, "Subjects": int(ok.sum()),
                                   "Mean_O": float(o[ok].mean()), "Mean_X": float(x[ok].mean()),
                                   "Mean_Diff_O_minus_X": float(diff.mean()), "Median_Diff_O_minus_X": float(diff.median()),
                                   "Subjects_O_Better": int((sign * diff > 0).sum()),
                                   "Subjects_O_Worse": int((sign * diff < 0).sum()),
                                   "Wilcoxon_p": wilcoxon_p(o[ok], x[ok])})
    summary = pd.DataFrame(summary_rows)
    noise = pd.DataFrame(noise_rows)
    compare = pd.DataFrame(compare_rows)
    level = pd.DataFrame(level_rows)
    # console
    for metric, spec in METRICS.items():
        print(f"\n=== {spec['word']} ===")
        for r in compare[compare.Metric == metric].to_dict("records"):
            print(f"  {r['Dataset']}: |difference| between seeds of one model (median / 75 / 90 / 95 %) "
                  f"{r['Seed_Q50']:.3f} / {r['Seed_Q75']:.3f} / {r['Seed_Q90']:.3f} / {r['Seed_Q95']:.3f}; "
                  f"O vs X of the same seed {r['Fusion_Q50']:.3f} / {r['Fusion_Q75']:.3f} / {r['Fusion_Q90']:.3f} / "
                  f"{r['Fusion_Q95']:.3f}")
        for r in summary[summary.Metric == metric].to_dict("records"):
            print(f"  {r['Dataset']}: {r['Subjects']} subjects (median seeds O {r['Seeds_O_Median']:g}, X "
                  f"{r['Seeds_X_Median']:g}); improved {r['Improved']}, worsened {r['Worsened']}, within seed noise "
                  f"{r['Within_Seed_Noise']}; all seeds O better {r['All_Seeds_O_Better']}, X better "
                  f"{r['All_Seeds_X_Better']}; mean O {r['Mean_O']:.3f}, X {r['Mean_X']:.3f} (Wilcoxon p "
                  f"{r['Seed_Mean_Wilcoxon_p']:.3g})")
        for c in ("improved", "worsened"):
            s = subjects[subjects[f"Class_{metric}"] == c].sort_values(f"Diff_Mean_{metric}")
            print(f"  {c} ({len(s)}): " + (", ".join(
                f"{r['Dataset']} {r['Subject']} (mean {r[f'Mean_X_{metric}']:.2f} -> {r[f'Mean_O_{metric}']:.2f}, "
                f"p {min(r[f'p_O_better_{metric}'], r[f'p_O_worse_{metric}']):.3f})" for r in s.to_dict("records"))
                or "none"))
        draw_slopes(subjects, metric, int(args.members), alpha, out_root)
        draw_ecdf(seed_pairs, fusion_pairs, metric, out_root)
        draw_per_seed(level, metric, out_root)
    notes = pd.DataFrame([
        ("Seeds", f"K = {args.members} seeds per model and fold (m0-m{args.members - 1}); seed of member k = "
                  "stable_seed('261006-ensemble-member', 'w10s_nonoverlap', dataset, subject, k) as in 261007 / 261013 / "
                  "261028; O and X of a seed share it; the seed sets initial weights, batch order, jitter and dropout"),
        ("Members", "stored members are read (261007 for CapnoBase, 261013 Fusion O and 261028 Fusion X for BIDMC / "
                    "STEAM2); missing ones are trained with run_member of 261028 (the training of the stored models) "
                    "into Trained_Models; sheet Members lists where every member was read from"),
        ("GT", "rescored GT of 261118 (CapnoBase peak detection, BIDMC ann1"
               + (" without ann1 gap minutes" if args.ann1_nogap else "") + ", STEAM2 stored GT); all members of a "
               "subject on the same scored minutes"),
        ("Values", "MAE = mean |stored rate - GT breath count|, |PCC| = mean |Wave_PCC| over the scored minutes"),
        ("Test", f"exact permutation test of the difference of the seed means (every split of the K + K values); "
                 f"one-sided p for O better and for O worse; improved / worsened if p <= {alpha:g}; q = "
                 "Benjamini-Hochberg over the subjects of a dataset (observed direction)"),
        ("Separation", "All_K: every O seed better than every X seed (O) or the opposite (X); 3v3: the same with m0-m2"),
        ("Seed level", "per seed k (and the seed mean): mean over the subjects of O and X, subjects better / worse, "
                       "Wilcoxon signed-rank test over the subjects"),
        ("Subjects", "MAE issue subjects (CapnoBase 0115, BIDMC 27, STEAM2 12_YJJ, 22_LMS, 26_CKK) "
                     + ("kept" if args.keep_issue_subjects else "left out")),
    ], columns=["Item", "Description"])
    path = out_root / f"{NAME}.xlsx"
    with pd.ExcelWriter(path, engine="openpyxl") as writer:
        summary.round(4).to_excel(writer, sheet_name="Summary", index=False)
        subjects.sort_values(["Dataset", "Diff_Mean_MAE"]).round(4).to_excel(writer, sheet_name="Subjects", index=False)
        noise.round(4).to_excel(writer, sheet_name="Seed_Noise", index=False)
        compare.round(4).to_excel(writer, sheet_name="Fusion_vs_Seed", index=False)
        level.round(4).to_excel(writer, sheet_name="Seed_Level", index=False)
        seed_pairs.round(4).to_excel(writer, sheet_name="Seed_Pairs", index=False)
        fusion_pairs.round(4).to_excel(writer, sheet_name="Fusion_Pairs", index=False)
        pd.DataFrame(member_rows).to_excel(writer, sheet_name="Members", index=False)
        (pd.concat(minute_frames, ignore_index=True).round(4) if minute_frames else pd.DataFrame()).to_excel(
            writer, sheet_name="Minutes", index=False)
        pd.DataFrame(missing, columns=["Dataset", "Subject", "Missing"]).to_excel(writer, sheet_name="Missing", index=False)
        notes.to_excel(writer, sheet_name="Notes", index=False)
    print(f"\nTable: {path}\nFigures: {out_root}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Stem fusion judged K seeds against K seeds (trains the missing seeds)")
    p.add_argument("--members", type=int, default=MEMBERS, help="seeds per model and fold (default 5: m0-m4)")
    p.add_argument("--no-train", dest="train", action="store_false", help="analyse the members that exist only")
    p.add_argument("--shard", default=None, help="i/n: this worker trains every n-th job from i (parallel workers)")
    p.add_argument("--gpu", type=int, default=0)
    p.add_argument("--allow-cpu", action="store_true", help="train on the CPU when there is no CUDA GPU")
    p.add_argument("--epoch-cap", type=int, default=None, help="upper limit on the epochs (quick tests only)")
    p.add_argument("--alpha", type=float, default=ALPHA, help="one-sided p value of the K vs K test (default 0.05)")
    p.add_argument("--datasets", nargs="+", choices=list(DATASETS), default=list(DATASETS))
    p.add_argument("--subjects", nargs="+", default=None, help="only these held-out subjects (quick tests)")
    p.add_argument("--keep-issue-subjects", action="store_true", help="train and keep the MAE issue subjects")
    p.add_argument("--ann1-nogap", action="store_true", help="BIDMC: leave out minutes overlapping an ann1 gap")
    p.add_argument("--root-261007", type=Path, default=WORKSPACE / "261007_RRWaveNet_NoEvent_v1")
    p.add_argument("--root-261013", type=Path, default=WORKSPACE / "261013_RRWavNet_STEAM_BIDMC_v1")
    p.add_argument("--root-noise", type=Path, default=WORKSPACE / "261028_Stem_ablation_noise_v1")
    p.add_argument("--output-root", type=Path, default=OUTPUT_ROOT)
    p.add_argument("--script-261118", type=Path, default=None)
    p.add_argument("--script-261028", type=Path, default=None)
    p.add_argument("--script-261007", type=Path, default=None)
    p.add_argument("--script-261013", type=Path, default=None)
    p.add_argument("--capnobase-path", type=Path, default=None)
    p.add_argument("--bidmc-path", type=Path, default=None)
    p.add_argument("--steam2-ppg-dir", type=Path, default=None)
    p.add_argument("--steam2-rsp-dir", type=Path, default=None)
    args = p.parse_args(list(sys.argv[1:] if argv is None else argv))
    if args.members < 2:
        p.error("--members must be at least 2")
    return args


def main(argv: Optional[Sequence[str]] = None) -> None:
    args = parse_args(argv)
    out_root = Path(args.output_root)
    out_root.mkdir(parents=True, exist_ok=True)
    (out_root / "run_settings.json").write_text(json.dumps({k: str(v) for k, v in vars(args).items()}, indent=2),
                                                encoding="utf-8")
    base = load_module(BASE_SCRIPT, args.script_261118, "seed_v2_base_261118", "--script-261118")
    if args.train:
        log_path = out_root / "Training_Log.csv"
        log_rows = pd.read_csv(log_path).to_dict("records") if log_path.exists() else []
        train_missing(base, args, log_rows)
    if args.shard and parse_shard(args.shard)[1] > 1:
        print("\nWorker finished. When every worker is done, run once without --shard (or with --no-train) for the "
              "analysis.")
        return
    analyse(base, args)


if __name__ == "__main__":
    main()
