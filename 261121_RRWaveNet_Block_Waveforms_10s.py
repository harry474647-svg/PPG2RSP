# -*- coding: utf-8 -*-
r"""261121_RRWaveNet_Block_Waveforms_10s: in every 10 s test window of every subject, the ground truth respiratory
waveform overlapped with the feature waveform of the encoder input (Stem) and of every residual encoder block (B1-B4)
of the stored RRWaveNet model of the subject's LOSOCV fold - the waveforms behind the channel-averaged absolute PCC
of each block, window by window, instead of one averaged value per block.

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

One curve per stage and window (as the channel-averaged absolute PCC, 1/24 sum |r_c|):
  1. every channel z-scored over the window (a constant channel is left out);
  2. r_c = Pearson r of channel c with the GT in the window;
  3. curve = mean over the channels of sign(r_c) x channel (a channel in antiphase with the GT is turned over, as |r_c|
     does), z-scored over the window.
  Beside every window: mean |r| = 1/24 sum |r_c| of the stage in that window, and r = Pearson r of the stage curve
  with the GT. The GT is z-scored over the window too; the y axis is in z units.
Colors: GT black, Stem blue, B1 yellow, B2 pink, B3 orange, B4 red (slightly darkened).

Outputs (D:\PPG2RSP_RRWaveNet_Inspired\261121_RRWaveNet_Block_Waveforms_10s)
  {Dataset}\{Dataset}_{subject}_m{member}_block_waveforms.pdf   every test window of the subject, in time order
                                                               (4 windows per page, --windows-per-page)
  261121_RRWaveNet_Block_Waveforms_10s.xlsx
    Windows  every window: mean |r| and curve r of every stage
    Missing  subjects without a stored model of the member (or without a weight decay selection)

Run: python 261121_RRWaveNet_Block_Waveforms_10s.py     (Spyder: runfile(...), no options needed)
Options: --datasets, --subjects CapnoBase:0009 BIDMC:05 ..., --member, --windows-per-page, --root-261007,
--root-261013, --output-root, --script-261007, --script-261013, --capnobase-path, --bidmc-path, --steam2-ppg-dir,
--steam2-rsp-dir, --batch-size, --cpu
"""

import argparse
import importlib.util
import json
import re
import sys
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

# ---------------------------------------------------------------------------
# Settings
# ---------------------------------------------------------------------------

NAME = "261121_RRWaveNet_Block_Waveforms_10s"
HERE = Path(__file__).resolve().parent
WORKSPACE = Path(r"D:\PPG2RSP_RRWaveNet_Inspired")
OUTPUT_ROOT = WORKSPACE / NAME
ROOT_261007 = WORKSPACE / "261007_RRWaveNet_NoEvent_v1"
ROOT_261013 = WORKSPACE / "261013_RRWavNet_STEAM_BIDMC_v1"
SCRIPTS = {"261007": "261007_RRWaveNet_Ensemble_NoEvent.py", "261013": "261013_RRWaveNet_Adult_SubjectVal_WD.py"}
VARIANT_261007 = "full"
WINDOW = "w10s_nonoverlap"
DATASETS = ("CapnoBase", "BIDMC", "STEAM2")
KEYS = {"CapnoBase": "capnobase", "BIDMC": "bidmc", "STEAM2": "steam2"}
SOURCE = {"CapnoBase": "261007", "BIDMC": "261013", "STEAM2": "261013"}
ISSUE_SUBJECTS = {"STEAM2": ("22_LMS", "26_CKK", "12_YJJ"), "BIDMC": ("27",), "CapnoBase": ("0115",)}
STAGES = ("Stem", "B1", "B2", "B3", "B4")
COLORS = {"GT": "#111111", "Stem": "#2457d6", "B1": "#d9b800", "B2": "#e0559f", "B3": "#e8800e", "B4": "#d42a2a"}
WIDTHS = {"GT": 2.4, "Stem": 1.5, "B1": 1.5, "B2": 1.5, "B3": 1.5, "B4": 1.7}
CONSTANT_SD = 1e-8
INK, MUTED, GRID = "#1f1f1f", "#6b6a66", "#e4e3df"

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
        name = "blockwave_src_" + re.sub(r"\W", "_", path.stem)
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
# One curve per stage
# ---------------------------------------------------------------------------

def zscore(x: np.ndarray) -> np.ndarray:
    sd = float(np.std(x))
    return (x - np.mean(x)) / sd if sd > CONSTANT_SD else np.full_like(x, np.nan)


def stage_curve(features: np.ndarray, gt_z: np.ndarray) -> Tuple[np.ndarray, float, float]:
    """Curve of one stage in one window (sign-aligned mean of the z-scored channels, z-scored), mean |r_c| over the
    channels and the Pearson r of the curve with the GT."""
    centred = features - features.mean(axis=1, keepdims=True)
    sd = centred.std(axis=1)
    keep = sd > CONSTANT_SD
    if not keep.any() or not np.isfinite(gt_z).all():
        return np.full(features.shape[1], np.nan), np.nan, np.nan
    channels = centred[keep] / sd[keep, None]
    r = channels @ gt_z / len(gt_z)  # both z-scored with the population SD: Pearson r
    curve = zscore(np.mean(np.sign(r)[:, None] * channels, axis=0))
    curve_r = float(np.mean(curve * gt_z)) if np.isfinite(curve).all() else np.nan
    return curve, float(np.mean(np.abs(r))), curve_r


# ---------------------------------------------------------------------------
# Figures
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


def draw_window(ax, side, t: np.ndarray, gt_z: np.ndarray, curves: Dict[str, np.ndarray],
                values: Dict[str, Tuple[float, float]], title: str) -> None:
    style(ax)
    for stage in STAGES:
        ax.plot(t, curves[stage], color=COLORS[stage], lw=WIDTHS[stage], zorder=3)
    ax.plot(t, gt_z, color=COLORS["GT"], lw=WIDTHS["GT"], zorder=5)
    ax.set_xlim(t[0], t[-1] + (t[1] - t[0]))
    ax.set_ylabel("z", fontsize=9)
    ax.set_title(title, loc="left", fontsize=10.2, color=INK, pad=4)
    side.axis("off")
    side.set_xlim(0, 1)
    side.set_ylim(0, 1)
    side.text(0.30, 0.93, "mean |r|", ha="center", va="center", fontsize=8.4, color=MUTED)
    side.text(0.62, 0.93, "r", ha="center", va="center", fontsize=8.4, color=MUTED)
    for row, stage in enumerate(STAGES):
        y = 0.77 - row * 0.16
        side.add_line(Line2D([-0.30, -0.15], [y, y], color=COLORS[stage], lw=2.4, transform=side.transAxes,
                             clip_on=False))
        side.text(-0.08, y, stage, ha="left", va="center", fontsize=8.6, color=INK)
        mean_abs_r, curve_r = values[stage]
        side.text(0.30, y, f"{mean_abs_r:.2f}" if np.isfinite(mean_abs_r) else "n/a", ha="center", va="center",
                  fontsize=8.6, color=INK, family="monospace")
        side.text(0.62, y, f"{curve_r:.2f}" if np.isfinite(curve_r) else "n/a", ha="center", va="center",
                  fontsize=8.6, color=INK, family="monospace")


def subject_pdf(path: Path, header: str, names: Dict[str, str], windows: List[dict], per_page: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pages = int(np.ceil(len(windows) / per_page))
    handles = [Line2D([], [], color=COLORS["GT"], lw=WIDTHS["GT"])] + [
        Line2D([], [], color=COLORS[s], lw=2.0) for s in STAGES]
    labels = ["Ground truth"] + [names[s] for s in STAGES]
    with PdfPages(path) as pdf:
        for page in range(pages):
            chunk = windows[page * per_page:(page + 1) * per_page]
            height = 1.75 + 2.55 * per_page
            fig = plt.figure(figsize=(12.5, height))
            top = 1.0 - 1.55 / height
            row_h = (top - 0.45 / height) / per_page
            for k, w in enumerate(chunk):
                y0 = top - (k + 1) * row_h + 0.42 / height
                ax = fig.add_axes([0.065, y0, 0.74, row_h - 0.78 / height])
                side = fig.add_axes([0.86, y0, 0.12, row_h - 0.78 / height])
                draw_window(ax, side, w["t"], w["gt"], w["curves"], w["values"], w["title"])
                if k == len(chunk) - 1:
                    ax.set_xlabel("Time in the recording (s)", fontsize=9)
            fig.text(0.065, 1.0 - 0.22 / height, f"{header} · page {page + 1}/{pages}", fontsize=12, color=INK,
                     va="top")
            fig.text(0.065, 1.0 - 0.55 / height,
                     "Each stage: mean of its 24 channels, each z-scored in the window and turned over when its r with "
                     "the GT is negative; curves and GT z-scored in the window\nmean |r| = 1/24 sum |r_c| of the stage "
                     "in the window; r = Pearson r of the stage curve with the GT",
                     fontsize=8.2, color=MUTED, va="top", linespacing=1.4)
            fig.legend(handles, labels, loc="upper left", bbox_to_anchor=(0.06, 1.0 - 0.98 / height), ncol=6,
                       frameon=False, fontsize=9.0, handlelength=2.4, columnspacing=1.8)
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


def process_dataset(args, dataset: str, chosen: Dict[str, List[str]], out_root: Path, device, rows: List[dict],
                    missing: List[dict]) -> None:
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
    names = stage_names(mod)
    # every fold has the same network (the weight decay is a training setting only)
    model = (mod.VARIANTS[VARIANT_261007] if source == "261007" else next(iter(mod.VARIANTS.values()))).options.build()
    model = model.to(device)
    for subject in sorted(data):
        if dataset in chosen and not any(same_subject(subject, s) for s in chosen[dataset]):
            continue
        checkpoint, variant, reason = model_checkpoint(args, mod, spec, dataset, subject)
        if checkpoint is None:
            missing.append({"Dataset": dataset, "Subject": subject, "Member": f"m{args.member}", "Missing": reason})
            print(f"  {subject}: {reason}", flush=True)
            continue
        model.load_state_dict(torch.load(checkpoint, map_location=device))
        loader = mod.make_eval_loader(mod.SubjectWindowDataset(data, [subject], "test", False), args.batch_size,
                                      device)
        metadata = mod.test_metadata(data, subject)
        stages, target = stage_features(model, loader, device)
        if len(metadata) != len(target):
            raise RuntimeError(f"{dataset} {subject}: {len(metadata)} windows in the metadata, {len(target)} computed")
        issue = any(same_subject(subject, x) for x in ISSUE_SUBJECTS[dataset])
        samples = target.shape[1]
        multi = len({m["segment_id"] for m in metadata}) > 1
        windows = []
        for index, meta in enumerate(metadata):
            gt_z = zscore(target[index])
            t0 = meta["start"] / fs
            curves, values = {}, {}
            for stage, features in zip(STAGES, stages):
                curves[stage], mean_abs_r, curve_r = stage_curve(features[index], gt_z)
                values[stage] = (mean_abs_r, curve_r)
            title = (f"{'recording ' + str(meta['segment_id']) + ' · ' if multi else ''}"
                     f"{t0:.0f}-{t0 + samples / fs:.0f} s")
            windows.append({"t": t0 + np.arange(samples) / fs, "gt": gt_z, "curves": curves, "values": values,
                            "title": title})
            row = {"Dataset": dataset, "Subject": subject, "Issue_Subject": issue, "Member": f"m{args.member}",
                   "Model": variant, "Segment_ID": int(meta["segment_id"]), "Start_Sec": t0,
                   "End_Sec": t0 + samples / fs}
            for stage in STAGES:
                row[f"{stage}_Mean_Abs_r"], row[f"{stage}_Curve_r"] = values[stage]
            rows.append(row)
        tag = mod.safe_tag(subject)
        header = (f"{dataset} {subject} · RRWaveNet {source} ({variant}), member m{args.member} · ground truth vs Stem "
                  f"and encoder blocks, every 10 s test window" + (" · MAE issue subject" if issue else ""))
        subject_pdf(out_root / dataset / f"{dataset}_{tag}_m{args.member}_block_waveforms.pdf", header, names, windows,
                    args.windows_per_page)
        means = {s: np.nanmean([w["values"][s][0] for w in windows]) for s in STAGES}
        print(f"  {subject}: {len(windows)} windows ({variant}); mean |r| "
              + ", ".join(f"{s} {means[s]:.2f}" for s in STAGES) + (" [MAE issue]" if issue else ""), flush=True)
        del loader, stages, target, windows
    del data, model
    mod.release_memory(device)


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Ground truth vs the Stem and encoder block feature waveforms of the stored "
                                            "RRWaveNet models, every 10 s test window")
    p.add_argument("--datasets", nargs="+", choices=list(DATASETS), default=list(DATASETS))
    p.add_argument("--subjects", nargs="+", default=None, help="Dataset:Subject (default: every subject)")
    p.add_argument("--member", type=int, default=0, help="ensemble member of the fold (default 0)")
    p.add_argument("--windows-per-page", type=int, default=4)
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
    rows: List[dict] = []
    missing: List[dict] = []
    for dataset in datasets:
        process_dataset(args, dataset, chosen, out_root, device, rows, missing)
    path = out_root / f"{NAME}.xlsx"
    with pd.ExcelWriter(path, engine="openpyxl") as writer:
        pd.DataFrame(rows).round(4).to_excel(writer, sheet_name="Windows", index=False)
        pd.DataFrame(missing, columns=["Dataset", "Subject", "Member", "Missing"]).to_excel(
            writer, sheet_name="Missing", index=False)
    print(f"\n{len(rows)} windows drawn; figures and table under {out_root}")


if __name__ == "__main__":
    main()
