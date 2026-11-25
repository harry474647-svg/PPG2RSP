# -*- coding: utf-8 -*-
r"""261125_RRWaveNet_Block_ChannelAbsPCC: the channel-averaged absolute PCC with the respiratory waveform at the encoder
input (Stem) and after every residual encoder block (B1-B4) of the stored RRWaveNet models - the block comparison of
the feature-level analysis (261010 / 261011, {Stage}_Channel_AbsR_Mean) - per dataset and averaged over all datasets,
one figure per group.

Models (stored, nothing is trained): the final results
  CapnoBase      261007_RRWaveNet_Ensemble_NoEvent    (D:\PPG2RSP_RRWaveNet_Inspired\261007_RRWaveNet_NoEvent_v1),
                 variant full
  BIDMC, STEAM2  261013_RRWaveNet_Adult_SubjectVal_WD (D:\PPG2RSP_RRWaveNet_Inspired\261013_RRWavNet_STEAM_BIDMC_v1),
                 the full model of the weight decay selected in the fold (WD_Selection)
  members m0, m1, m2 of every fold (as 261011). The model of a fold is run (eval mode) on the test windows of its
  held-out subject, read by its own script (the windows and the target exactly as in the results).

Value of a subject and stage (as channel_abs_r of 261011)
  1. the 24-channel feature map of every 10 s test window (24 x 640 samples, 64 Hz)
       Stem  the stem fusion output (1x1 conv + GroupNorm), the encoder input
       B1-B4 the output of the residual encoder blocks (dilation 1, 2, 4, 8; B4 = encoder output)
  2. r_c = Pearson r of channel c with the target respiratory waveform of the window (min-max to [-1, 1], as trained)
  3. |r_c| averaged over the windows of the subject, then over the 24 channels: (1/24) sum_c |r_c|
  4. computed per model, then the mean over the members (m0-m2) of the subject
  No PCA: the PCC is computed on each of the 24 channels directly, and only the |PCC| values are averaged.

Groups
  CapnoBase, BIDMC, STEAM2   the subjects of the dataset
  All datasets               every subject of the three datasets once: the mean over all subjects (as the All
                             datasets of 261011). The mean of the three dataset means is in the table as well.

Figure (one per group, the same y axis in every figure): the mean over the subjects at Stem, B1, B2, B3, B4 (line,
open circles, the value above each point): the trend only, no range drawn.
Table per group and stage: n, mean, SD, 95% CI (1.96 x SE over the subjects), median, Q1, Q3, min, max; from B1 on the
change from Stem (mean over
the subjects), the subjects higher than at Stem and the Wilcoxon signed-rank p vs Stem (two-sided, over the subjects).

Outputs (D:\PPG2RSP_RRWaveNet_Inspired\261125_RRWaveNet_Block_ChannelAbsPCC)
  All_datasets_block_channel_abs_pcc.png / .pdf
  CapnoBase_block_channel_abs_pcc.png / .pdf, BIDMC_..., STEAM2_...
  261125_RRWaveNet_Block_ChannelAbsPCC.xlsx
    Summary   per group and stage (above)
    Subjects  every subject: value of every stage (mean over the members), members, windows
    Members   every subject and member: value of every stage
    Missing   folds without a stored model of a member (or without a weight decay selection)
    Notes

Run: python 261125_RRWaveNet_Block_ChannelAbsPCC.py     (Spyder: runfile(...), no options needed)
Options: --datasets, --subjects CapnoBase:0009 BIDMC:05 ..., --members 0 1 2, --ymax (top of the y axis, e.g. 1.0;
default: from the data), --root-261007, --root-261013, --output-root, --script-261007, --script-261013,
--capnobase-path, --bidmc-path, --steam2-ppg-dir, --steam2-rsp-dir, --batch-size, --cpu
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
from matplotlib.ticker import MultipleLocator
from scipy import stats

# ---------------------------------------------------------------------------
# Settings
# ---------------------------------------------------------------------------

NAME = "261125_RRWaveNet_Block_ChannelAbsPCC"
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
DATASET_MEANS = "Mean of the dataset means"
KEYS = {"CapnoBase": "capnobase", "BIDMC": "bidmc", "STEAM2": "steam2"}
SOURCE = {"CapnoBase": "261007", "BIDMC": "261013", "STEAM2": "261013"}
STAGES = ("Stem", "B1", "B2", "B3", "B4")
MEMBERS = (0, 1, 2)
Z95 = 1.96
VALUE = "Channel_AbsPCC"
FIGURE_NAME = "block_channel_abs_pcc"

LINE = "#2457d6"
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
        name = "blockpcc_src_" + re.sub(r"\W", "_", path.stem)
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


def model_root(args, mod, spec, dataset: str, subject: str) -> Tuple[Optional[Path], str, str]:
    """Results folder of the final model of the subject's fold, its variant, and why it is missing."""
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
    return mod.condition_root(root, variant, spec, key), variant, ""


def stage_features(model, loader, device) -> Tuple[List[np.ndarray], np.ndarray]:
    """Feature maps (windows, 24, samples) at Stem and after every encoder block, and the target (windows, samples)."""
    model.eval()
    stages: List[List[np.ndarray]] = [[] for _ in STAGES]
    targets = []
    with torch.no_grad():
        for batch in loader:
            x, y = batch[0].to(device), batch[1]
            features = model.fused_representation(x)
            stages[0].append(features.cpu().numpy().astype(np.float32))
            for index, block in enumerate(model.encoder, start=1):
                features = block(features)
                stages[index].append(features.cpu().numpy().astype(np.float32))
            targets.append(y.reshape(len(y), -1).numpy().astype(np.float32))
    return [np.concatenate(chunks) for chunks in stages], np.concatenate(targets)


# ---------------------------------------------------------------------------
# Channel-averaged absolute PCC
# ---------------------------------------------------------------------------

def channel_abs_r(features: np.ndarray, target: np.ndarray) -> np.ndarray:
    """|Pearson r| of every channel with the target, averaged over the windows (channel_abs_r of 261011)."""
    f = features.astype(np.float64) - features.mean(axis=2, keepdims=True)
    t = target.astype(np.float64) - target.mean(axis=1, keepdims=True)
    numerator = np.einsum("nct,nt->nc", f, t)
    denominator = np.sqrt(np.einsum("nct,nct->nc", f, f) * np.einsum("nt,nt->n", t, t)[:, None])
    with np.errstate(divide="ignore", invalid="ignore"):
        r = np.where(denominator > 1e-12, numerator / denominator, np.nan)
    return np.nanmean(np.abs(r), axis=0)


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


def process_dataset(args, dataset: str, chosen: Dict[str, List[str]], device, store: dict) -> None:
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
    # every fold has the same network (the weight decay is a training setting only)
    model = (mod.VARIANTS[VARIANT_261007] if source == "261007" else next(iter(mod.VARIANTS.values()))).options.build()
    model = model.to(device)
    for subject in sorted(data):
        if dataset in chosen and not any(same_subject(subject, s) for s in chosen[dataset]):
            continue
        root, variant, reason = model_root(args, mod, spec, dataset, subject)
        if root is None:
            store["missing"].append({"Dataset": dataset, "Subject": subject, "Member": "", "Missing": reason})
            print(f"  {subject}: {reason}", flush=True)
            continue
        loader = mod.make_eval_loader(mod.SubjectWindowDataset(data, [subject], "test", False), args.batch_size,
                                      device)
        values, used, windows = [], [], 0
        for member in args.members:
            checkpoint = root / "Members" / f"m{member}" / "Models" / f"{mod.safe_tag(subject)}_best.pth"
            if not checkpoint.exists():
                store["missing"].append({"Dataset": dataset, "Subject": subject, "Member": f"m{member}",
                                         "Missing": f"no stored model ({checkpoint})"})
                continue
            model.load_state_dict(torch.load(checkpoint, map_location=device))
            stages, target = stage_features(model, loader, device)
            windows = int(len(target))
            value = {stage: float(np.nanmean(channel_abs_r(features, target)))
                     for stage, features in zip(STAGES, stages)}
            values.append(value)
            used.append(f"m{member}")
            store["members"].append({"Dataset": dataset, "Subject": subject, "Model": variant,
                                     "Member": f"m{member}", "Windows": windows,
                                     **{f"{s}_{VALUE}": value[s] for s in STAGES}})
            del stages, target
        del loader
        if not used:
            continue
        mean = {s: float(np.mean([v[s] for v in values])) for s in STAGES}
        store["subjects"].append({"Dataset": dataset, "Subject": subject, "Model": variant,
                                  "Members": " ".join(used), "Windows": windows,
                                  **{f"{s}_{VALUE}": mean[s] for s in STAGES}})
        print(f"  {subject} ({variant}, {' '.join(used)}): {windows} windows; channel-averaged |PCC| "
              + ", ".join(f"{s} {mean[s]:.3f}" for s in STAGES), flush=True)
    del data, model
    mod.release_memory(device)


# ---------------------------------------------------------------------------
# Statistics
# ---------------------------------------------------------------------------

def wilcoxon_p(a: np.ndarray, b: np.ndarray) -> float:
    keep = np.isfinite(a) & np.isfinite(b)
    a, b = a[keep], b[keep]
    if len(a) < 2:
        return float("nan")
    if np.allclose(a, b):
        return 1.0
    try:
        return float(stats.wilcoxon(a, b).pvalue)
    except ValueError:
        return float("nan")


def describe(values: np.ndarray) -> dict:
    v = values[np.isfinite(values)]
    n = int(v.size)
    if n == 0:
        return {"N_Subjects": 0}
    mean = float(v.mean())
    sd = float(v.std(ddof=1)) if n > 1 else float("nan")
    half = Z95 * sd / np.sqrt(n) if n > 1 else float("nan")
    return {"N_Subjects": n, "Mean": mean, "SD": sd, "CI95_Low": mean - half, "CI95_High": mean + half,
            "Median": float(np.median(v)), "Q1": float(np.quantile(v, 0.25)), "Q3": float(np.quantile(v, 0.75)),
            "Min": float(v.min()), "Max": float(v.max())}


def groups_of(subjects: pd.DataFrame, datasets: Sequence[str]) -> List[Tuple[str, pd.DataFrame]]:
    groups = [(d, subjects[subjects.Dataset == d]) for d in datasets]
    groups = [(d, f) for d, f in groups if not f.empty]
    if len(groups) > 1:
        groups.append((ALL, subjects[subjects.Dataset.isin([d for d, _ in groups])]))
    return groups


def summary_table(groups: List[Tuple[str, pd.DataFrame]]) -> pd.DataFrame:
    rows = []
    for group, frame in groups:
        stem = frame[f"Stem_{VALUE}"].to_numpy(float)
        for stage in STAGES:
            values = frame[f"{stage}_{VALUE}"].to_numpy(float)
            row = {"Group": group, "Stage": stage, **describe(values)}
            if stage != "Stem":
                paired = np.isfinite(values) & np.isfinite(stem)
                row.update({
                    "Change_vs_Stem": float(np.mean(values[paired] - stem[paired])) if paired.any() else np.nan,
                    "Subjects_Higher_than_Stem": int(np.sum(values[paired] > stem[paired])),
                    "Wilcoxon_p_vs_Stem": wilcoxon_p(values, stem),
                })
            rows.append(row)
    datasets = [(g, f) for g, f in groups if g != ALL]
    if len(datasets) > 1:
        for stage in STAGES:
            means = [float(np.nanmean(f[f"{stage}_{VALUE}"].to_numpy(float))) for _, f in datasets]
            rows.append({"Group": DATASET_MEANS, "Stage": stage, "N_Subjects": int(sum(len(f) for _, f in datasets)),
                         "Mean": float(np.mean(means))})
    summary = pd.DataFrame(rows)
    if "Subjects_Higher_than_Stem" in summary:
        summary["Subjects_Higher_than_Stem"] = summary["Subjects_Higher_than_Stem"].astype("Int64")
    return summary


# ---------------------------------------------------------------------------
# Figure
# ---------------------------------------------------------------------------

def y_top(summary: pd.DataFrame, given: Optional[float]) -> float:
    if given:
        return float(given)
    high = np.nanmax(summary.loc[summary.Group != DATASET_MEANS, "Mean"].to_numpy(float))
    return float(min(1.0, np.ceil((high + 0.05) * 10.0) / 10.0)) if np.isfinite(high) else 1.0


def block_figure(base: Path, group: str, frame: pd.DataFrame, summary: pd.DataFrame, top: float) -> None:
    part = summary[summary.Group == group].set_index("Stage").reindex(list(STAGES))
    mean = part["Mean"].to_numpy(float)
    x = np.arange(len(STAGES))
    fig, ax = plt.subplots(figsize=(5.6, 4.3))
    ax.plot(x, mean, "-", color=LINE, lw=2.2, zorder=3)
    ax.plot(x, mean, "o", ms=9, mfc="white", mec=LINE, mew=2.2, zorder=4)
    for xi, m in zip(x, mean):
        ax.annotate(f"{m:.3f}", (xi, m), textcoords="offset points", xytext=(0, 10), ha="center", va="bottom",
                    fontsize=9.5, color=INK, zorder=5,
                    bbox={"boxstyle": "round,pad=0.15", "facecolor": "white", "edgecolor": "none", "alpha": 0.85})
    ax.set_xticks(x)
    ax.set_xticklabels(STAGES, fontsize=11, fontweight="bold", color=INK)
    ax.set_xlim(-0.4, len(STAGES) - 0.6)
    ax.set_ylim(0.0, top)
    ax.yaxis.set_major_locator(MultipleLocator(0.1 if top > 0.35 else 0.05))
    ax.tick_params(axis="y", labelsize=9.5, colors=INK)
    ax.set_ylabel("Channel-averaged\nabsolute PCC", fontsize=11, color=INK)
    ax.grid(True, color=GRID, lw=0.7)
    ax.set_axisbelow(True)
    n = int(frame.Subject.size)
    subtitle = f"n = {n} subjects"
    if group == ALL:
        counts = frame.groupby("Dataset", sort=False).size()
        subtitle += " (" + ", ".join(f"{d} {int(counts[d])}" for d in DATASETS if d in counts.index) + ")"
    subtitle += "; mean over subjects"
    ax.set_title(f"{group}\n", fontsize=12, fontweight="bold", color=INK, loc="center")
    ax.text(0.5, 1.02, subtitle, transform=ax.transAxes, ha="center", va="bottom", fontsize=8.5, color=MUTED)
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(base.with_name(f"{base.name}.{ext}"), dpi=300)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Channel-averaged absolute PCC with the target at Stem and after every "
                                            "encoder block of the stored RRWaveNet models, per dataset and over all "
                                            "datasets")
    p.add_argument("--datasets", nargs="+", choices=list(DATASETS), default=list(DATASETS))
    p.add_argument("--subjects", nargs="+", default=None, help="Dataset:Subject (default: every subject)")
    p.add_argument("--members", nargs="+", type=int, default=list(MEMBERS), help="members of the fold (default 0 1 2)")
    p.add_argument("--ymax", type=float, default=None, help="top of the y axis (default: from the data)")
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
    chosen = parse_subjects(args.subjects)
    datasets = [d for d in args.datasets if not chosen or d in chosen]
    out_root = Path(args.output_root)
    out_root.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    print(f"{NAME}: device {device}, members {' '.join(f'm{m}' for m in args.members)}; results under {out_root}")
    store = {"subjects": [], "members": [], "missing": []}
    for dataset in datasets:
        process_dataset(args, dataset, chosen, device, store)
    subjects = pd.DataFrame(store["subjects"])
    if subjects.empty:
        raise SystemExit("No stored model found: nothing to draw.")
    groups = groups_of(subjects, datasets)
    summary = summary_table(groups)
    top = y_top(summary, args.ymax)
    for group, frame in groups:
        block_figure(out_root / f"{group.replace(' ', '_')}_{FIGURE_NAME}", group, frame, summary, top)

    print("\nChannel-averaged absolute PCC (mean ± SD over subjects; vs Stem: change, Wilcoxon p)")
    for r in summary.itertuples(index=False):
        if r.Group == DATASET_MEANS:
            print(f"  {r.Group:<26} {r.Stage:<4} {r.Mean:.3f}")
            continue
        extra = "" if r.Stage == "Stem" else (f"   {r.Change_vs_Stem:+.3f}, {int(r.Subjects_Higher_than_Stem)}/"
                                              f"{r.N_Subjects} higher, p {r.Wilcoxon_p_vs_Stem:.2g}")
        print(f"  {r.Group:<26} {r.Stage:<4} n {r.N_Subjects:3d}: {r.Mean:.3f} ± {r.SD:.3f}{extra}")

    notes = pd.DataFrame([
        ("Models", "final results: CapnoBase 261007 full; BIDMC / STEAM2 261013, weight decay selected in the fold; "
                   f"members {' '.join(f'm{m}' for m in args.members)}, each run on the 10 s test windows of its "
                   "held-out subject"),
        ("Stages", "Stem = stem fusion output (encoder input); B1-B4 = residual encoder block outputs (d = 1, 2, 4, 8)"),
        ("Value", "Pearson r of every channel with the target (min-max [-1, 1]) in every window; |r| averaged over the "
                  "windows, then over the 24 channels: (1/24) sum |r_c| (channel_abs_r / Channel_AbsR_Mean of "
                  "261011); per model, then the mean over the members of the subject"),
        ("All datasets", "every subject of the three datasets once (mean over all subjects, as 261011); "
                         f"'{DATASET_MEANS}' = the mean of the dataset means"),
        ("95% CI", "mean ± 1.96 x SD / sqrt(n) over the subjects (table only; the figures show the mean)"),
        ("vs Stem", "change = mean over the subjects of (stage - Stem); subjects higher than at Stem; Wilcoxon "
                    "signed-rank test, two-sided, over the subjects"),
        ("Figures", f"<Group>_{FIGURE_NAME}.png / .pdf, the same y axis (0 to {top:g}) in every figure"),
    ], columns=["Item", "Description"])
    path = out_root / f"{NAME}.xlsx"
    with pd.ExcelWriter(path, engine="openpyxl") as writer:
        summary.round(4).to_excel(writer, sheet_name="Summary", index=False)
        subjects.round(4).to_excel(writer, sheet_name="Subjects", index=False)
        pd.DataFrame(store["members"]).round(4).to_excel(writer, sheet_name="Members", index=False)
        pd.DataFrame(store["missing"], columns=["Dataset", "Subject", "Member", "Missing"]).to_excel(
            writer, sheet_name="Missing", index=False)
        notes.to_excel(writer, sheet_name="Notes", index=False)
    print(f"\nFigures and table under {out_root}")


if __name__ == "__main__":
    main()
