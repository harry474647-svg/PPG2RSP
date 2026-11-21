# -*- coding: utf-8 -*-
r"""261121_RRWaveNet_Block_Waveforms_10s_v3: the ground truth respiratory waveform and the feature waveforms of the
encoder input (Stem) and of the residual encoder blocks (B1-B4) of the stored RRWaveNet models, every curve min-max
normalized to [-1, 1]; (1) in every 10 s test window of every subject and (2) averaged over the breath cycles of the
ground truth (per subject, per dataset and over all datasets). Every figure is drawn for Stem and B4 only and for every
stage (Stem, B1, B2, B3, B4), each overlapped (one axis) and separated (one row per stage, each with the ground truth).

Changes from 261121_RRWaveNet_Block_Waveforms_10s / _v2
  * normalization: min-max to [-1, 1] in the window (channels, stage curves and ground truth) instead of z-scores;
  * channel sign: fixed per subject and channel - the sign of its r with the ground truth summed over the subject's
    other windows - instead of the sign of its r in the window itself (that sign is chosen with the window's own
    ground truth, so the curve resembled any ground truth by construction);
  * breath-cycle averages; Stem and B4 and every stage in one script, overlapped and separated.

Models (stored, nothing is trained): the final results
  CapnoBase      261007_RRWaveNet_Ensemble_NoEvent    (D:\PPG2RSP_RRWaveNet_Inspired\261007_RRWaveNet_NoEvent_v1),
                 variant full
  BIDMC, STEAM2  261013_RRWaveNet_Adult_SubjectVal_WD (D:\PPG2RSP_RRWaveNet_Inspired\261013_RRWavNet_STEAM_BIDMC_v1),
                 the full model of the weight decay selected in the fold (WD_Selection)
  member m0 of every fold (--member 1 / 2 for another member: the channels of different members are different
  features, so the members are not averaged). The model of a fold is run (eval mode) on the test windows of its
  held-out subject, read by its own script (the windows and the target exactly as in the results).

Stages (24 channels x 640 samples per 10 s window, 64 Hz; the blocks are depthwise, so channel c is the same channel
at every stage)
  Stem  the stem fusion output (1x1 conv + GroupNorm), the encoder input
  B1-B4 the output of the residual encoder blocks (dilation 1, 2, 4, 8; B4 = encoder output)
Ground truth (GT): the respiratory waveform of the window as the model was trained on it (min-max to [-1, 1]).

Values of a stage in a window
  r_c       Pearson r of channel c with the GT (does not depend on the normalization)
  mean |r|  1/24 sum |r_c|: the channel-averaged absolute PCC (the value of the earlier block comparison)
  curve     every channel min-max normalized to [-1, 1], multiplied by its fixed sign (above), averaged over the
            channels, min-max normalized to [-1, 1]
  r         Pearson r of the drawn curve with the GT. It is higher than mean |r|: the average of the 24 channels keeps
            what they share and cancels what each channel has on its own.

Breath-cycle average: the breaths of the GT in every window - troughs of the GT (at least 1 s apart, prominence 0.5
of the [-1, 1] range), a breath = trough to the next trough inside the window, 1-20 s long - each resampled to 0-100%
(101 points); the GT and every stage curve of these breaths averaged per subject (subjects with at least 3 breaths);
per dataset and over all datasets the mean of the subject averages with its 95% confidence interval (1.96 x SE over
the subjects). Beside the curves: mean |r| and r of every stage, mean +- SD over the subjects (their window means).
Every subject is included (the MAE issue subjects are marked on their pages).

Outputs (D:\PPG2RSP_RRWaveNet_Inspired\261121_RRWaveNet_Block_Waveforms_10s_v3); {set} = StemB4 or AllBlocks,
{layout} = overlap or separate
  Windows\{Dataset}\{Dataset}_{subject}_m{member}_{set}_{layout}.pdf   every test window of the subject in time order
                                                                       (--no-window-figures: not drawn)
  Breath_Cycle\{Dataset | All_datasets}_breath_cycle_{set}_{layout}.png / .pdf
  Breath_Cycle\Subjects\{Dataset}_subjects_breath_cycle_{set}_{layout}.pdf    every subject's average
  261121_RRWaveNet_Block_Waveforms_10s_v3.xlsx
    Datasets  per dataset and all datasets, every stage: mean |r| and r (mean, SD over the subjects), breaths
    Subjects  per subject, every stage: mean |r| and r (mean over the windows), windows, breaths
    Windows   every window: mean |r| and r of every stage
    Missing   subjects without a stored model of the member (or without a weight decay selection)
    Notes

Run: python 261121_RRWaveNet_Block_Waveforms_10s_v3.py     (Spyder: runfile(...), no options needed)
Options: --datasets, --subjects CapnoBase:0009 BIDMC:05 ..., --member, --windows-per-page (overlap, default 4),
--no-window-figures, --root-261007, --root-261013, --output-root, --script-261007, --script-261013, --capnobase-path,
--bidmc-path, --steam2-ppg-dir, --steam2-rsp-dir, --batch-size, --cpu
"""

import argparse
import importlib.util
import json
import re
import sys
import warnings
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.lines import Line2D
from scipy.signal import find_peaks

# ---------------------------------------------------------------------------
# Settings
# ---------------------------------------------------------------------------

NAME = "261121_RRWaveNet_Block_Waveforms_10s_v3"
HERE = Path(__file__).resolve().parent
WORKSPACE = Path(r"D:\PPG2RSP_RRWaveNet_Inspired")
OUTPUT_ROOT = WORKSPACE / NAME
ROOT_261007 = WORKSPACE / "261007_RRWaveNet_NoEvent_v1"
ROOT_261013 = WORKSPACE / "261013_RRWavNet_STEAM_BIDMC_v1"
SCRIPTS = {"261007": "261007_RRWaveNet_Ensemble_NoEvent.py", "261013": "261013_RRWaveNet_Adult_SubjectVal_WD.py"}
VARIANT_261007 = "full"
WINDOW = "w10s_nonoverlap"
DATASETS = ("CapnoBase", "BIDMC", "STEAM2")
ALL = "All datasets"
KEYS = {"CapnoBase": "capnobase", "BIDMC": "bidmc", "STEAM2": "steam2"}
SOURCE = {"CapnoBase": "261007", "BIDMC": "261013", "STEAM2": "261013"}
ISSUE_SUBJECTS = {"STEAM2": ("22_LMS", "26_CKK", "12_YJJ"), "BIDMC": ("27",), "CapnoBase": ("0115",)}
STAGES = ("Stem", "B1", "B2", "B3", "B4")
SETS = {"StemB4": ("Stem", "B4"), "AllBlocks": STAGES}
SET_TITLES = {"StemB4": "Stem and B4", "AllBlocks": "Stem, B1, B2, B3 and B4"}
LAYOUTS = ("overlap", "separate")
COLORS = {"GT": "#111111", "Stem": "#2457d6", "B1": "#d9b800", "B2": "#e0559f", "B3": "#e8800e", "B4": "#d42a2a"}
WIDTHS = {"GT": 2.4, "Stem": 1.5, "B1": 1.5, "B2": 1.5, "B3": 1.5, "B4": 1.7}
CONSTANT_RANGE = 1e-8
MIN_BREATH_SEC = 1.0      # GT troughs at least 1 s apart (60 breaths/min)
MAX_BREATH_SEC = 20.0
TROUGH_PROMINENCE = 0.5   # of the [-1, 1] range of the GT
CYCLE_POINTS = 101
MIN_SUBJECT_BREATHS = 3
Y_LIMITS = (-1.18, 1.18)
INK, MUTED, GRID = "#1f1f1f", "#6b6a66", "#e4e3df"
METHOD_TEXT = ("Every curve min-max normalized to [-1, 1] in the window. Stage curve = its 24 channels, each min-max "
               "normalized and turned over when its r with the GT (summed over the subject's other windows) is negative, "
               "averaged.\nmean |r| = 1/24 sum |r_c| of the channels (channel-averaged absolute PCC); r = Pearson r of the "
               "drawn curve with the GT (higher: the average of the channels cancels what each channel has on its own)")

_MODULES: Dict[str, object] = {}


# ---------------------------------------------------------------------------
# Source scripts and stored models
# ---------------------------------------------------------------------------

def find_script(name: str, given: Optional[Path], roots: Sequence[Path]) -> Path:
    """The script of the results: given, written in a run manifest, or found nearby."""
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
        name = "blockwave_v3_src_" + re.sub(r"\W", "_", path.stem)
        spec = importlib.util.spec_from_file_location(name, key)
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module
        spec.loader.exec_module(module)
        _MODULES[key] = module
    return _MODULES[key]


def read_json(path: Path) -> Optional[dict]:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


def model_checkpoint(args, mod, spec, dataset: str, subject: str) -> Tuple[Optional[Path], str, str]:
    """Checkpoint of the member of the subject's fold, its model folder name (variant) and why it is missing."""
    key = KEYS[dataset]
    if SOURCE[dataset] == "261007":
        root, variant = Path(args.root_261007), VARIANT_261007
    else:
        root = Path(args.root_261013)
        found = mod.stored_selection(root, list(mod.VARIANTS.values()), spec, key, subject)
        if found is not None:
            variant = found[1].name
        else:  # selection stored with other candidates than the module's defaults: the variant it names
            variant = (read_json(mod.selection_path(root, spec, key, subject)) or {}).get("Selected_Variant")
        if not variant:
            return None, "", f"no weight decay selection ({mod.selection_path(root, spec, key, subject)})"
    path = (mod.condition_root(root, variant, spec, key) / "Members" / f"m{args.member}" / "Models"
            / f"{mod.safe_tag(subject)}_best.pth")
    if not path.exists():
        return None, variant, f"no stored model ({path})"
    return path, variant, ""


def stage_features(model, loader, device) -> Tuple[List[np.ndarray], np.ndarray]:
    """Feature maps (windows, 24, samples) at Stem and after every encoder block, and the GT (windows, samples)."""
    model.eval()
    stages: List[List[np.ndarray]] = [[] for _ in STAGES]
    targets = []
    with torch.no_grad():
        for batch in loader:
            x, y = batch[0].to(device), batch[1]
            features = model.fused_representation(x)
            stages[0].append(features.cpu().numpy().astype(np.float64))
            for index, block in enumerate(model.encoder, start=1):
                features = block(features)
                stages[index].append(features.cpu().numpy().astype(np.float64))
            targets.append(y.reshape(len(y), -1).numpy().astype(np.float64))
    return [np.concatenate(chunks) for chunks in stages], np.concatenate(targets)


# ---------------------------------------------------------------------------
# Curves and values
# ---------------------------------------------------------------------------

def minmax(x: np.ndarray) -> np.ndarray:
    """Min-max to [-1, 1] along the last axis; a constant series becomes NaN."""
    low = np.min(x, axis=-1, keepdims=True)
    span = np.max(x, axis=-1, keepdims=True) - low
    ok = span > CONSTANT_RANGE
    return np.where(ok, 2.0 * (x - low) / np.where(ok, span, 1.0) - 1.0, np.nan)


def pearson_rows(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Pearson r along the last axis (NaN where a series is constant or not finite)."""
    a = a - np.mean(a, axis=-1, keepdims=True)
    b = b - np.mean(b, axis=-1, keepdims=True)
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.sum(a * b, axis=-1) / np.sqrt(np.sum(a * a, axis=-1) * np.sum(b * b, axis=-1))


def subject_curves(stages: List[np.ndarray], target: np.ndarray) -> Tuple[np.ndarray, Dict[str, dict]]:
    """GT and, per stage, the drawn curve, mean |r| and r of every window."""
    gt = minmax(target)
    out = {}
    for stage, features in zip(STAGES, stages):
        channels = minmax(features)                                  # (windows, 24, samples)
        valid = np.isfinite(channels).all(axis=2)
        r = np.where(valid, pearson_rows(features, target[:, None, :]), np.nan)
        with warnings.catch_warnings():  # a window without any valid channel stays NaN
            warnings.simplefilter("ignore", RuntimeWarning)
            mean_abs_r = np.nanmean(np.abs(r), axis=1)
        r0 = np.nan_to_num(r)
        sign = np.sign(r0.sum(axis=0)[None, :] - r0) * valid        # fixed sign from the subject's other windows
        used = np.abs(sign).sum(axis=1)
        combined = np.einsum("nc,nct->nt", sign, np.nan_to_num(channels)) / np.maximum(used, 1)[:, None]
        curve = np.where((used > 0)[:, None], minmax(combined), np.nan)
        out[stage] = {"curve": curve, "mean_abs_r": mean_abs_r, "r": pearson_rows(curve, gt)}
    return gt, out


def breath_cycles(gt: np.ndarray, curves: Dict[str, dict], fs: float) -> Dict[str, np.ndarray]:
    """Breaths of the GT (trough to trough inside a window) of the GT and every stage curve, resampled to 0-100%."""
    cycles: Dict[str, List[np.ndarray]] = {key: [] for key in ("GT",) + STAGES}
    samples = np.arange(gt.shape[1])
    for i in range(len(gt)):
        if not np.isfinite(gt[i]).all() or not all(np.isfinite(curves[s]["curve"][i]).all() for s in STAGES):
            continue
        troughs, _ = find_peaks(-gt[i], distance=max(1, int(round(MIN_BREATH_SEC * fs))), prominence=TROUGH_PROMINENCE)
        for a, b in zip(troughs[:-1], troughs[1:]):
            if not (MIN_BREATH_SEC * fs <= b - a <= MAX_BREATH_SEC * fs):
                continue
            x = np.linspace(a, b, CYCLE_POINTS)
            cycles["GT"].append(np.interp(x, samples, gt[i]))
            for s in STAGES:
                cycles[s].append(np.interp(x, samples, curves[s]["curve"][i]))
    return {k: np.array(v).reshape(-1, CYCLE_POINTS) for k, v in cycles.items()}


# ---------------------------------------------------------------------------
# Figure parts
# ---------------------------------------------------------------------------

def style(ax) -> None:
    ax.grid(True, color=GRID, lw=0.6)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.tick_params(labelsize=8.5)


def stage_names(mod) -> Dict[str, str]:
    dilations = mod.MODEL_CONFIG["encoder_dilations"]
    return {"Stem": "Stem", **{f"B{i + 1}": f"B{i + 1} (d={d})" for i, d in enumerate(dilations)}}


def fmt(value: float, sd: Optional[float] = None) -> str:
    if not np.isfinite(value):
        return "n/a"
    return f"{value:.2f}" if sd is None or not np.isfinite(sd) else f"{value:.2f}±{sd:.2f}"


def legend_items(stages: Sequence[str], names: Dict[str, str]) -> Tuple[list, list]:
    handles = [Line2D([], [], color=COLORS["GT"], lw=WIDTHS["GT"])] + [
        Line2D([], [], color=COLORS[s], lw=2.0) for s in stages]
    return handles, ["Ground truth"] + [names[s] for s in stages]


def value_table(side, stages: Sequence[str], values: Dict[str, Tuple[str, str]], top: float = 0.93) -> None:
    """Small table beside an axis: swatch, stage, mean |r|, r."""
    side.axis("off")
    side.set_xlim(0, 1)
    side.set_ylim(0, 1)
    side.text(0.36, top, "mean |r|", ha="center", va="center", fontsize=8.2, color=MUTED)
    side.text(0.80, top, "r", ha="center", va="center", fontsize=8.2, color=MUTED)
    step = min(0.16, (top - 0.08) / max(1, len(stages)))
    for row, stage in enumerate(stages):
        y = top - 0.17 - row * step
        side.add_line(Line2D([-0.30, -0.16], [y, y], color=COLORS[stage], lw=2.4, transform=side.transAxes,
                             clip_on=False))
        side.text(-0.11, y, stage, ha="left", va="center", fontsize=8.4, color=INK)
        side.text(0.36, y, values[stage][0], ha="center", va="center", fontsize=8.2, color=INK, family="monospace")
        side.text(0.80, y, values[stage][1], ha="center", va="center", fontsize=8.2, color=INK, family="monospace")


def row_text(side, stage: str, name: str, mean_abs_r: str, r: str) -> None:
    """Text beside one separated row: swatch, stage, mean |r| and r."""
    side.axis("off")
    side.set_xlim(0, 1)
    side.set_ylim(0, 1)
    side.add_line(Line2D([-0.30, -0.16], [0.68, 0.68], color=COLORS[stage], lw=2.4, transform=side.transAxes,
                         clip_on=False))
    side.text(-0.11, 0.68, name, ha="left", va="center", fontsize=8.6, color=INK)
    side.text(-0.11, 0.36, f"mean |r| {mean_abs_r}\nr {r}", ha="left", va="center", fontsize=8.0, color=INK,
              family="monospace", linespacing=1.35)


def page_header(fig, height: float, title: str, subtitle: str, stages: Sequence[str], names: Dict[str, str]) -> None:
    fig.text(0.065, 1.0 - 0.22 / height, title, fontsize=12, color=INK, va="top")
    fig.text(0.065, 1.0 - 0.52 / height, subtitle, fontsize=7.8, color=MUTED, va="top", linespacing=1.4)
    handles, labels = legend_items(stages, names)
    fig.legend(handles, labels, loc="upper left", bbox_to_anchor=(0.06, 1.0 - 1.02 / height), ncol=6,
               frameon=False, fontsize=9.0, handlelength=2.4, columnspacing=1.8)


# ---------------------------------------------------------------------------
# Window figures
# ---------------------------------------------------------------------------

def window_pdf(path: Path, title: str, names: Dict[str, str], windows: List[dict], stages: Sequence[str],
               layout: str, per_page: int) -> None:
    """Every window of a subject: overlapped (one axis per window) or separated (one row per stage)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    rows_per_window = 1 if layout == "overlap" else len(stages)
    row_in = 2.0 if layout == "overlap" else 1.15
    if layout == "separate":
        per_page = 2 if len(stages) <= 2 else 1
    head_in, title_in, label_in, foot_in = 1.6, 0.32, 0.42, 0.25
    block_in = title_in + rows_per_window * row_in + label_in
    height = head_in + per_page * block_in + foot_in
    pages = int(np.ceil(len(windows) / per_page))
    with PdfPages(path) as pdf:
        for page in range(pages):
            fig = plt.figure(figsize=(12.5, height))
            page_header(fig, height, f"{title} · page {page + 1}/{pages}", METHOD_TEXT, stages, names)
            y_top = height - head_in
            for w in windows[page * per_page:(page + 1) * per_page]:
                y = y_top - title_in
                axes = []
                for k in range(rows_per_window):
                    bottom = y - row_in
                    ax = fig.add_axes([0.065, (bottom + 0.06) / height, 0.74, (row_in - 0.12) / height])
                    side = fig.add_axes([0.86, (bottom + 0.06) / height, 0.12, (row_in - 0.12) / height])
                    style(ax)
                    drawn = stages if layout == "overlap" else (stages[k],)
                    for stage in drawn:
                        ax.plot(w["t"], w["curves"][stage], color=COLORS[stage], lw=WIDTHS[stage], zorder=3)
                    ax.plot(w["t"], w["gt"], color=COLORS["GT"], lw=WIDTHS["GT"], zorder=5)
                    ax.set_xlim(w["t"][0], w["t"][-1] + (w["t"][1] - w["t"][0]))
                    ax.set_ylim(*Y_LIMITS)
                    ax.set_yticks([-1, 0, 1])
                    if layout == "overlap":
                        value_table(side, stages, {s: (fmt(w["values"][s][0]), fmt(w["values"][s][1]))
                                                   for s in stages})
                    else:
                        stage = stages[k]
                        row_text(side, stage, names[stage], fmt(w["values"][stage][0]), fmt(w["values"][stage][1]))
                    axes.append(ax)
                    y = bottom
                for ax in axes[:-1]:
                    ax.tick_params(labelbottom=False)
                axes[0].set_title(w["title"], loc="left", fontsize=10.2, color=INK, pad=4)
                axes[-1].set_xlabel("Time in the recording (s)", fontsize=9)
                y_top = y - label_in
            pdf.savefig(fig)
            plt.close(fig)


# ---------------------------------------------------------------------------
# Breath-cycle figures
# ---------------------------------------------------------------------------

def group_average(subjects: List[dict]) -> Tuple[Dict[str, Tuple[np.ndarray, np.ndarray]], Dict[str, tuple], int]:
    """Mean of the subject breath-cycle averages with its 95% CI, the stage values (mean, SD over subjects) and the
    number of breaths."""
    with_cycles = [s for s in subjects if s["average"] is not None]
    average = {}
    for key in ("GT",) + STAGES:
        if not with_cycles:
            break
        stack = np.array([s["average"][key] for s in with_cycles])
        se = stack.std(axis=0, ddof=1) / np.sqrt(len(stack)) if len(stack) > 1 else np.zeros(CYCLE_POINTS)
        average[key] = (stack.mean(axis=0), 1.96 * se)
    values = {}
    for stage in STAGES:
        m = np.array([s["mean_abs_r"][stage] for s in subjects], float)
        r = np.array([s["r"][stage] for s in subjects], float)
        values[stage] = (np.nanmean(m) if np.isfinite(m).any() else np.nan,
                         np.nanstd(m, ddof=1) if np.isfinite(m).sum() > 1 else np.nan,
                         np.nanmean(r) if np.isfinite(r).any() else np.nan,
                         np.nanstd(r, ddof=1) if np.isfinite(r).sum() > 1 else np.nan)
    return average, values, int(sum(s["n_breaths"] for s in with_cycles))


def draw_cycle(ax, average: Dict[str, Tuple[np.ndarray, np.ndarray]], stages: Sequence[str], band: bool = True) -> None:
    phase = np.linspace(0, 100, CYCLE_POINTS)
    style(ax)
    for key in tuple(stages) + ("GT",):
        mean, ci = average[key]
        if band:
            ax.fill_between(phase, mean - ci, mean + ci, color=COLORS[key], alpha=0.14, lw=0, zorder=2)
        ax.plot(phase, mean, color=COLORS[key], lw=WIDTHS[key] + 0.4, zorder=5 if key == "GT" else 3)
    ax.set_xlim(0, 100)
    ax.set_ylim(*Y_LIMITS)
    ax.set_yticks([-1, 0, 1])


def cycle_figure(base: Path, group: str, subjects: List[dict], stages: Sequence[str], names: Dict[str, str],
                 set_name: str, layout: str) -> None:
    """Breath-cycle average of a dataset (or all datasets), overlapped or separated."""
    average, values, breaths = group_average(subjects)
    if not average:
        return
    n = sum(1 for s in subjects if s["average"] is not None)
    rows = 1 if layout == "overlap" else len(stages)
    row_in = 3.6 if layout == "overlap" else 1.45
    head_in, foot_in = 1.6, 0.75
    height = head_in + rows * row_in + foot_in
    fig = plt.figure(figsize=(11.0, height))
    title = (f"{group}: breath-cycle average of the ground truth and {SET_TITLES[set_name]} "
             f"({n} subjects, {breaths} breaths)")
    subtitle = ("Breaths of the GT (trough to trough) in every 10 s window, each 0-100%; curves min-max [-1, 1] in the "
                "window; mean of the subject averages, band = 95% CI over the subjects.\nStage curve = its 24 channels "
                "(min-max, fixed sign per subject), averaged. mean |r| = 1/24 sum |r_c|; r = Pearson r of the stage curve "
                "with the GT; beside the curves: mean ± SD over the subjects")
    page_header(fig, height, title, subtitle, stages, names)
    y = height - head_in
    axes = []
    for k in range(rows):
        bottom = y - row_in
        ax = fig.add_axes([0.075, (bottom + 0.1) / height, 0.68, (row_in - 0.2) / height])
        side = fig.add_axes([0.82, (bottom + 0.1) / height, 0.16, (row_in - 0.2) / height])
        drawn = stages if layout == "overlap" else (stages[k],)
        draw_cycle(ax, average, drawn)
        if layout == "overlap":
            value_table(side, stages, {s: (fmt(values[s][0], values[s][1]), fmt(values[s][2], values[s][3]))
                                       for s in stages}, top=0.9)
        else:
            s = stages[k]
            row_text(side, s, names[s], fmt(values[s][0], values[s][1]), fmt(values[s][2], values[s][3]))
        ax.set_ylabel("[-1, 1]", fontsize=8.5)
        axes.append(ax)
        y = bottom
    for ax in axes[:-1]:
        ax.tick_params(labelbottom=False)
    axes[-1].set_xlabel("Breath cycle (%): trough to trough of the ground truth", fontsize=9)
    base.parent.mkdir(parents=True, exist_ok=True)
    for ext in ("png", "pdf"):
        fig.savefig(base.with_name(f"{base.name}.{ext}"), dpi=200)
    plt.close(fig)


def subject_cycle_pdf(path: Path, dataset: str, subjects: List[dict], stages: Sequence[str], names: Dict[str, str],
                      set_name: str, layout: str) -> None:
    """Every subject's breath-cycle average: overlapped (3 x 3 subjects per page) or separated (3 subjects per page,
    one row per stage)."""
    shown = [s for s in subjects if s["average"] is not None]
    if not shown:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    cols = 3
    rows = 3 if layout == "overlap" else len(stages)
    per_page = cols * 3 if layout == "overlap" else cols
    row_in = (2.4 if len(stages) <= 2 else 2.6) if layout == "overlap" else 1.3
    head_in, foot_in = (1.6, 0.35) if layout == "overlap" else (1.95, 0.55)
    height = head_in + rows * row_in + foot_in
    pages = int(np.ceil(len(shown) / per_page))
    phase_label = "Breath cycle (%)"
    with PdfPages(path) as pdf:
        for page in range(pages):
            chunk = shown[page * per_page:(page + 1) * per_page]
            fig = plt.figure(figsize=(12.5, height))
            page_header(fig, height, f"{dataset}: breath-cycle average of every subject, {SET_TITLES[set_name]} · "
                        f"page {page + 1}/{pages}",
                        "Breaths of the GT (trough to trough) in every 10 s window, each 0-100%; curves min-max [-1, 1] "
                        "in the window; band = 95% CI over the breaths of the subject\nmean |r| and r: mean over the "
                        "subject's windows (mean |r| = 1/24 sum |r_c|; r = Pearson r of the stage curve with the GT)",
                        stages, names)
            width = 0.88 / cols
            for j, subject in enumerate(chunk):
                if layout == "overlap":
                    col, row = j % cols, j // cols
                    cells = [(row, stages)]
                else:
                    col = j
                    cells = [(k, (stage,)) for k, stage in enumerate(stages)]
                label = (f"{subject['subject']}{' (MAE issue)' if subject['issue'] else ''} · "
                         f"{subject['n_breaths']} breaths")
                for k, (row, drawn) in enumerate(cells):
                    bottom = height - head_in - (row + 1) * row_in
                    if layout == "overlap":  # title lines above the axis
                        lines_in = 0.9 if len(stages) <= 2 else 1.08
                        box = [0.07 + col * width, (bottom + 0.32) / height, width - 0.05, (row_in - lines_in) / height]
                    else:
                        box = [0.07 + col * width, (bottom + 0.12) / height, width - 0.05, (row_in - 0.42) / height]
                    ax = fig.add_axes(box)
                    draw_cycle(ax, subject["cycle_ci"], drawn)
                    ax.tick_params(labelsize=7.5)
                    if layout == "overlap" and len(drawn) <= 2:
                        values = "mean |r| / r: " + " · ".join(
                            f"{s} {fmt(subject['mean_abs_r'][s])} / {fmt(subject['r'][s])}" for s in drawn)
                    else:  # every stage: one line per value, stages in order
                        values = (f"mean |r| {drawn[0]}-{drawn[-1]}: "
                                  + " ".join(fmt(subject['mean_abs_r'][s]) for s in drawn)
                                  + f"\nr {drawn[0]}-{drawn[-1]}: " + " ".join(fmt(subject['r'][s]) for s in drawn))
                    if layout == "overlap":
                        ax.set_title(f"{label}\n{values}", loc="left", fontsize=7.6, color=INK, pad=3,
                                     linespacing=1.4)
                    else:
                        if k == 0:  # column header: the subject
                            fig.text(box[0], (bottom + row_in + 0.22) / height, label, fontsize=9.2, color=INK,
                                     va="bottom")
                        ax.set_title(f"mean |r| {fmt(subject['mean_abs_r'][drawn[0]])} · r "
                                     f"{fmt(subject['r'][drawn[0]])}", loc="right", fontsize=7.4, color=INK, pad=2)
                    if layout == "separate" and col == 0:
                        ax.set_ylabel(names[drawn[0]], fontsize=8.5, color=INK)
                    if layout == "separate" and k < len(cells) - 1:
                        ax.tick_params(labelbottom=False)
                    elif layout == "separate":
                        ax.set_xlabel(phase_label, fontsize=8)
            pdf.savefig(fig)
            plt.close(fig)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def same_subject(a: object, b: object) -> bool:
    a, b = str(a).strip(), str(b).strip()
    return a == b or (a.isdigit() and b.isdigit() and int(a) == int(b))


def parse_subjects(items: Optional[Sequence[str]]) -> Dict[str, List[str]]:
    chosen: Dict[str, List[str]] = {}
    for item in items or []:
        dataset, _, subject = str(item).partition(":")
        if dataset not in DATASETS or not subject:
            raise SystemExit(f"--subjects {item}: write Dataset:Subject with Dataset in {', '.join(DATASETS)}")
        chosen.setdefault(dataset, []).append(subject)
    return chosen


def process_dataset(args, dataset: str, chosen: Dict[str, List[str]], out_root: Path, device, store: dict) -> None:
    source = SOURCE[dataset]
    roots = [args.root_261007] if source == "261007" else [args.root_261013]
    script = find_script(SCRIPTS[source], args.script_261007 if source == "261007" else args.script_261013, roots)
    mod = load_script(script)
    for key, field, value in (("capnobase", "data_path", args.capnobase_path), ("bidmc", "data_path", args.bidmc_path),
                              ("steam2", "ppg_dir", args.steam2_ppg_dir), ("steam2", "rsp_dir", args.steam2_rsp_dir)):
        if value:
            mod.DATASET_CONFIGS[key][field] = str(value)
    spec = next(s for s in mod.WINDOW_SPECS if s.name == WINDOW)
    print(f"\n{dataset}: data and models of {script.name}", flush=True)
    data = mod.load_dataset(KEYS[dataset], spec)
    fs = float(mod.TARGET_FS)
    store["names"] = stage_names(mod)
    names = store["names"]
    # every fold has the same network (the weight decay is a training setting only)
    model = (mod.VARIANTS[VARIANT_261007] if source == "261007" else next(iter(mod.VARIANTS.values()))).options.build()
    model = model.to(device)
    subjects = []
    for subject in sorted(data):
        if dataset in chosen and not any(same_subject(subject, s) for s in chosen[dataset]):
            continue
        checkpoint, variant, reason = model_checkpoint(args, mod, spec, dataset, subject)
        if checkpoint is None:
            store["missing"].append({"Dataset": dataset, "Subject": subject, "Member": f"m{args.member}",
                                     "Missing": reason})
            print(f"  {subject}: {reason}", flush=True)
            continue
        model.load_state_dict(torch.load(checkpoint, map_location=device))
        loader = mod.make_eval_loader(mod.SubjectWindowDataset(data, [subject], "test", False), args.batch_size,
                                      device)
        metadata = mod.test_metadata(data, subject)
        stages, target = stage_features(model, loader, device)
        if len(metadata) != len(target):
            raise RuntimeError(f"{dataset} {subject}: {len(metadata)} windows in the metadata, {len(target)} computed")
        gt, curves = subject_curves(stages, target)
        del stages
        issue = any(same_subject(subject, x) for x in ISSUE_SUBJECTS[dataset])
        samples = target.shape[1]
        multi = len({m["segment_id"] for m in metadata}) > 1
        windows = []
        for index, meta in enumerate(metadata):
            t0 = meta["start"] / fs
            values = {s: (float(curves[s]["mean_abs_r"][index]), float(curves[s]["r"][index])) for s in STAGES}
            windows.append({"t": t0 + np.arange(samples) / fs, "gt": gt[index],
                            "curves": {s: curves[s]["curve"][index] for s in STAGES}, "values": values,
                            "title": f"{'recording ' + str(meta['segment_id']) + ' · ' if multi else ''}"
                                     f"{t0:.0f}-{t0 + samples / fs:.0f} s"})
            row = {"Dataset": dataset, "Subject": subject, "Issue_Subject": issue, "Member": f"m{args.member}",
                   "Model": variant, "Segment_ID": int(meta["segment_id"]), "Start_Sec": t0,
                   "End_Sec": t0 + samples / fs}
            for s in STAGES:
                row[f"{s}_Mean_Abs_r"], row[f"{s}_r"] = values[s]
            store["windows"].append(row)
        cycles = breath_cycles(gt, curves, fs)
        n_breaths = int(len(cycles["GT"]))
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            mean_abs_r = {s: float(np.nanmean(curves[s]["mean_abs_r"])) for s in STAGES}
            r_mean = {s: float(np.nanmean(curves[s]["r"])) for s in STAGES}
        average = cycle_ci = None
        if n_breaths >= MIN_SUBJECT_BREATHS:
            average = {k: v.mean(axis=0) for k, v in cycles.items()}
            cycle_ci = {k: (v.mean(axis=0), 1.96 * v.std(axis=0, ddof=1) / np.sqrt(len(v))) for k, v in cycles.items()}
        record = {"dataset": dataset, "subject": subject, "issue": issue, "n_breaths": n_breaths,
                  "average": average, "cycle_ci": cycle_ci, "mean_abs_r": mean_abs_r, "r": r_mean}
        subjects.append(record)
        srow = {"Dataset": dataset, "Subject": subject, "Issue_Subject": issue, "Member": f"m{args.member}",
                "Model": variant, "Windows": len(windows), "Breaths": n_breaths,
                "In_Breath_Cycle_Average": average is not None}
        for s in STAGES:
            srow[f"{s}_Mean_Abs_r"], srow[f"{s}_r"] = mean_abs_r[s], r_mean[s]
        store["subjects"].append(srow)
        if not args.no_window_figures:
            tag = mod.safe_tag(subject)
            for set_name, set_stages in SETS.items():
                for layout in LAYOUTS:
                    title = (f"{dataset} {subject} · RRWaveNet {source} ({variant}), member m{args.member} · ground "
                             f"truth vs {SET_TITLES[set_name]}, {layout}" + (" · MAE issue subject" if issue else ""))
                    window_pdf(out_root / "Windows" / dataset / f"{dataset}_{tag}_m{args.member}_{set_name}_{layout}.pdf",
                               title, names, windows, set_stages, layout, args.windows_per_page)
        print(f"  {subject}: {len(windows)} windows, {n_breaths} breaths ({variant}); mean |r| "
              + ", ".join(f"{s} {mean_abs_r[s]:.2f}" for s in STAGES) + "; r "
              + ", ".join(f"{s} {r_mean[s]:.2f}" for s in STAGES) + (" [MAE issue]" if issue else ""), flush=True)
        del loader, target, gt, curves, windows
    store["groups"][dataset] = subjects
    for set_name, set_stages in SETS.items():
        for layout in LAYOUTS:
            cycle_figure(out_root / "Breath_Cycle" / f"{dataset}_breath_cycle_{set_name}_{layout}", dataset, subjects,
                         set_stages, names, set_name, layout)
            subject_cycle_pdf(out_root / "Breath_Cycle" / "Subjects"
                              / f"{dataset}_subjects_breath_cycle_{set_name}_{layout}.pdf", dataset, subjects,
                              set_stages, names, set_name, layout)
    del data, model
    mod.release_memory(device)


def dataset_rows(groups: Dict[str, List[dict]]) -> pd.DataFrame:
    rows = []
    for group, subjects in list(groups.items()) + [(ALL, [s for v in groups.values() for s in v])]:
        if not subjects:
            continue
        _, values, breaths = group_average(subjects)
        row = {"Dataset": group, "Subjects": len(subjects),
               "Subjects_In_Breath_Cycle_Average": sum(1 for s in subjects if s["average"] is not None),
               "Breaths": breaths}
        for s in STAGES:
            m, msd, r, rsd = values[s]
            row.update({f"{s}_Mean_Abs_r": m, f"{s}_Mean_Abs_r_SD": msd, f"{s}_r": r, f"{s}_r_SD": rsd})
        rows.append(row)
    return pd.DataFrame(rows)


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Ground truth vs the Stem and encoder block feature waveforms of the stored "
                                            "RRWaveNet models (min-max [-1, 1]): every 10 s window and breath-cycle "
                                            "averages, Stem and B4 and every stage, overlapped and separated")
    p.add_argument("--datasets", nargs="+", choices=list(DATASETS), default=list(DATASETS))
    p.add_argument("--subjects", nargs="+", default=None, help="Dataset:Subject (default: every subject)")
    p.add_argument("--member", type=int, default=0, help="ensemble member of the fold (default 0)")
    p.add_argument("--windows-per-page", type=int, default=4, help="windows per page of the overlapped window figures")
    p.add_argument("--no-window-figures", action="store_true", help="breath-cycle figures and tables only")
    p.add_argument("--root-261007", type=Path, default=ROOT_261007)
    p.add_argument("--root-261013", type=Path, default=ROOT_261013)
    p.add_argument("--output-root", type=Path, default=OUTPUT_ROOT)
    p.add_argument("--script-261007", type=Path, default=None)
    p.add_argument("--script-261013", type=Path, default=None)
    p.add_argument("--capnobase-path", type=Path, default=None)
    p.add_argument("--bidmc-path", type=Path, default=None)
    p.add_argument("--steam2-ppg-dir", type=Path, default=None)
    p.add_argument("--steam2-rsp-dir", type=Path, default=None)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--cpu", action="store_true", help="run the models on the CPU")
    return p.parse_args(list(sys.argv[1:] if argv is None else argv))


def main(argv: Optional[Sequence[str]] = None) -> None:
    args = parse_args(argv)
    args.windows_per_page = max(1, int(args.windows_per_page))
    chosen = parse_subjects(args.subjects)
    datasets = [d for d in args.datasets if not chosen or d in chosen]
    out_root = Path(args.output_root)
    out_root.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    print(f"{NAME}: device {device}, member m{args.member}; results under {out_root}")
    store = {"windows": [], "subjects": [], "missing": [], "groups": {}, "names": None}
    for dataset in datasets:
        process_dataset(args, dataset, chosen, out_root, device, store)
    groups = {k: v for k, v in store["groups"].items() if v}
    if len(groups) > 1 and store["names"] is not None:
        everyone = [s for v in groups.values() for s in v]
        for set_name, set_stages in SETS.items():
            for layout in LAYOUTS:
                cycle_figure(out_root / "Breath_Cycle" / f"All_datasets_breath_cycle_{set_name}_{layout}", ALL,
                             everyone, set_stages, store["names"], set_name, layout)
    table = dataset_rows(groups) if groups else pd.DataFrame()
    if not table.empty:
        print("\nmean |r| and r (mean over the subjects)")
        for row in table.itertuples(index=False):
            d = row._asdict()
            print(f"  {d['Dataset']:<13} n {d['Subjects']:3d}: mean |r| "
                  + ", ".join(f"{s} {d[s + '_Mean_Abs_r']:.2f}" for s in STAGES) + " | r "
                  + ", ".join(f"{s} {d[s + '_r']:.2f}" for s in STAGES))
    notes = pd.DataFrame([
        ("Models", "final results: CapnoBase 261007 full; BIDMC / STEAM2 261013, weight decay selected in the fold; "
                   f"member m{args.member} of every fold, run on the test windows of its held-out subject"),
        ("Stages", "Stem = stem fusion output (encoder input); B1-B4 = residual encoder block outputs (d = 1, 2, 4, 8)"),
        ("Normalization", "min-max to [-1, 1] in the window: every channel, every stage curve and the GT"),
        ("mean |r|", "1/24 sum |r_c|, r_c = Pearson r of channel c with the GT in the window (the channel-averaged "
                     "absolute PCC; Pearson r does not depend on the normalization)"),
        ("Stage curve", "its 24 channels (min-max) times a fixed sign per subject and channel (sign of r_c summed over "
                        "the subject's other windows), averaged, min-max; r = Pearson r of the curve with the GT. "
                        "r > mean |r|: the average of the channels keeps what they share and cancels the rest"),
        ("Breath cycles", f"GT troughs at least {MIN_BREATH_SEC:g} s apart, prominence {TROUGH_PROMINENCE:g}; breath = "
                          f"trough to trough inside a window, {MIN_BREATH_SEC:g}-{MAX_BREATH_SEC:g} s, resampled to "
                          f"{CYCLE_POINTS} points; subject average with at least {MIN_SUBJECT_BREATHS} breaths; "
                          "group = mean of the subject averages, 95% CI = 1.96 x SE over subjects"),
        ("Subjects", "every subject (MAE issue subjects included and marked)"),
    ], columns=["Item", "Description"])
    path = out_root / f"{NAME}.xlsx"
    with pd.ExcelWriter(path, engine="openpyxl") as writer:
        table.round(4).to_excel(writer, sheet_name="Datasets", index=False)
        pd.DataFrame(store["subjects"]).round(4).to_excel(writer, sheet_name="Subjects", index=False)
        pd.DataFrame(store["windows"]).round(4).to_excel(writer, sheet_name="Windows", index=False)
        pd.DataFrame(store["missing"], columns=["Dataset", "Subject", "Member", "Missing"]).to_excel(
            writer, sheet_name="Missing", index=False)
        notes.to_excel(writer, sheet_name="Notes", index=False)
    print(f"\nFigures and table under {out_root}")


if __name__ == "__main__":
    main()
