# -*- coding: utf-8 -*-
r"""261126_RRWaveNet_Block_ChannelAbsPCC_PercentChange: the block comparison of the channel-averaged absolute PCC with
the target respiratory waveform (Stem, B1, B2, B3, B4), from the PCC computed here with the stored RRWaveNet models (not
from values of earlier figures): y axis the mean PCC, and at every point the change from Stem in % - one figure for
STEAM2, BIDMC, CapnoBase and all datasets each, with the MAE issue subjects left out.

Models (stored, nothing is trained): the final results
  CapnoBase      261007_RRWaveNet_Ensemble_NoEvent    (D:\PPG2RSP_RRWaveNet_Inspired\261007_RRWaveNet_NoEvent_v1),
                 variant full
  BIDMC, STEAM2  261013_RRWaveNet_Adult_SubjectVal_WD (D:\PPG2RSP_RRWaveNet_Inspired\261013_RRWavNet_STEAM_BIDMC_v1),
                 the full model of the weight decay selected in the fold (WD_Selection)
  members m0, m1, m2 of every fold (as 261011). The model of a fold is run (eval mode) on the test windows of its
  held-out subject, read by its own script (the windows and the target exactly as in the results).

PCC of a subject and stage (as channel_abs_r / {Stage}_Channel_AbsR_Mean of 261011 and 261125_RRWaveNet_Block_
ChannelAbsPCC)
  1. the 24-channel feature map of every 10 s test window (24 x 640 samples, 64 Hz)
       Stem  the stem fusion output (1x1 conv + GroupNorm), the encoder input
       B1-B4 the output of the residual encoder blocks (dilation 1, 2, 4, 8; B4 = encoder output)
  2. r_c = Pearson r of channel c with the target respiratory waveform of the window (min-max to [-1, 1], as trained)
  3. |r_c| averaged over the windows of the subject, then over the 24 channels: (1/24) sum_c |r_c|
  4. computed per model, then the mean over the members (m0-m2) of the subject
  No PCA: the PCC is computed on each of the 24 channels directly, and only the |PCC| values are averaged.

Subject sets
  MAE issue subjects left out   without the MAE issue subjects of the final results: CapnoBase 0115, BIDMC 27,
                                STEAM2 12_YJJ, 22_LMS, 26_CKK (the figures)
  All subjects                  every subject with stored models, as 261011 (whose block figures did not leave out
                                the MAE issue subjects): in the table only

Change from Stem (%) of a group (a dataset; All datasets = every subject of the three datasets once)
  100 x (mean PCC of the stage - mean PCC of Stem) / mean PCC of Stem, the means over the subjects of the group: the
  curve of the mean PCC on a scale relative to Stem (Stem = 0%).

Figures (MAE issue subjects left out; one per group): x axis Stem, B1, B2, B3, B4; y axis the mean PCC over the
subjects, fitted to the values of the figure; the line of the mean PCC with a dot at every stage, and above the dot
(under it where it is lower than its neighbours) the change from Stem in % (Stem 0%).

Outputs (D:\PPG2RSP_RRWaveNet_Inspired\261126_RRWaveNet_Block_ChannelAbsPCC_PercentChange)
  Issue_subjects_left_out\
    STEAM2_block_channel_abs_pcc.png / .pdf, BIDMC_..., CapnoBase_..., All_datasets_...
  261126_RRWaveNet_Block_ChannelAbsPCC_PercentChange.xlsx
    Summary   per subject set, group and stage: subjects, mean PCC, SD, change from Stem (%)
    Subjects  every subject: MAE issue subject or not, PCC of every stage (mean over the members), members, windows
    Members   every subject and member: PCC of every stage
    Missing   folds without a stored model of a member (or without a weight decay selection)
    Notes

Run: python 261126_RRWaveNet_Block_ChannelAbsPCC_PercentChange.py     (Spyder: runfile(...), no options needed)
Options: --datasets, --subjects CapnoBase:0009 BIDMC:05 ..., --members 0 1 2, --decimals (of the % at the points,
default 0), --root-261007, --root-261013, --output-root, --script-261007, --script-261013, --capnobase-path,
--bidmc-path, --steam2-ppg-dir, --steam2-rsp-dir, --batch-size, --cpu
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
from matplotlib.ticker import MaxNLocator

# ---------------------------------------------------------------------------
# Settings
# ---------------------------------------------------------------------------

NAME = "261126_RRWaveNet_Block_ChannelAbsPCC_PercentChange"
HERE = Path(__file__).resolve().parent
WORKSPACE = Path(r"D:\PPG2RSP_RRWaveNet_Inspired")
OUTPUT_ROOT = WORKSPACE / NAME
ROOT_261007 = WORKSPACE / "261007_RRWaveNet_NoEvent_v1"
ROOT_261013 = WORKSPACE / "261013_RRWavNet_STEAM_BIDMC_v1"
SCRIPTS = {"261007": "261007_RRWaveNet_Ensemble_NoEvent.py", "261013": "261013_RRWaveNet_Adult_SubjectVal_WD.py"}
VARIANT_261007 = "full"
WINDOW = "w10s_nonoverlap"
DATASETS = ("CapnoBase", "BIDMC", "STEAM2")
PANEL_ORDER = ("STEAM2", "BIDMC", "CapnoBase")
ALL = "All datasets"
KEYS = {"CapnoBase": "capnobase", "BIDMC": "bidmc", "STEAM2": "steam2"}
SOURCE = {"CapnoBase": "261007", "BIDMC": "261013", "STEAM2": "261013"}
STAGES = ("Stem", "B1", "B2", "B3", "B4")
MEMBERS = (0, 1, 2)
VALUE = "Channel_AbsPCC"
FIGURE_NAME = "block_channel_abs_pcc"
# MAE issue subjects of the final results
ISSUE_SUBJECTS = {"STEAM2": ("22_LMS", "26_CKK", "12_YJJ"), "BIDMC": ("27",), "CapnoBase": ("0115",)}
SUBJECT_SETS = ("All subjects", "MAE issue subjects left out")
FIGURE_SET, FIGURE_FOLDER = "MAE issue subjects left out", "Issue_subjects_left_out"   # the set drawn

COLOR = "#1f77b4"
INK, MUTED, GRID = "#1f1f1f", "#6b6a66", "#e6e6e6"

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
        name = "blockpccchange_src_" + re.sub(r"\W", "_", path.stem)
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
        issue = any(same_subject(subject, x) for x in ISSUE_SUBJECTS[dataset])
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
            store["members"].append({"Dataset": dataset, "Subject": subject, "Issue_Subject": issue,
                                     "Model": variant, "Member": f"m{member}", "Windows": windows,
                                     **{f"{s}_{VALUE}": value[s] for s in STAGES}})
            del stages, target
        del loader
        if not used:
            continue
        mean = {s: float(np.mean([v[s] for v in values])) for s in STAGES}
        store["subjects"].append({"Dataset": dataset, "Subject": subject, "Issue_Subject": issue, "Model": variant,
                                  "Members": " ".join(used), "Windows": windows,
                                  **{f"{s}_{VALUE}": mean[s] for s in STAGES}})
        print(f"  {subject} ({variant}, {' '.join(used)}{', MAE issue subject' if issue else ''}): {windows} windows; "
              "channel-averaged |PCC| " + ", ".join(f"{s} {mean[s]:.3f}" for s in STAGES), flush=True)
    del data, model
    mod.release_memory(device)


# ---------------------------------------------------------------------------
# Change from Stem
# ---------------------------------------------------------------------------

def groups_of(subjects: pd.DataFrame, datasets: Sequence[str]) -> List[Tuple[str, pd.DataFrame]]:
    groups = [(d, subjects[subjects.Dataset == d]) for d in datasets]
    groups = [(d, f) for d, f in groups if not f.empty]
    if len(groups) > 1:
        groups.append((ALL, subjects[subjects.Dataset.isin([d for d, _ in groups])]))
    return groups


def summary_table(subject_set: str, groups: List[Tuple[str, pd.DataFrame]]) -> pd.DataFrame:
    """Mean PCC of every stage over the subjects of a group and its change from the mean PCC of Stem (%)."""
    rows = []
    for group, frame in groups:
        stem = float(frame[f"Stem_{VALUE}"].astype(float).mean())
        for stage in STAGES:
            values = frame[f"{stage}_{VALUE}"].astype(float).dropna()
            mean = float(values.mean())
            rows.append({"Subject_Set": subject_set, "Group": group, "Stage": stage, "N_Subjects": int(values.size),
                         "Mean_PCC": mean,
                         "SD_PCC": float(values.std(ddof=1)) if values.size > 1 else float("nan"),
                         "Change_from_Stem_Pct": 100.0 * (mean - stem) / stem if stem > 0 else float("nan")})
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------

def change_text(percent: float, decimals: int) -> str:
    text = f"{percent:+.{decimals}f}"
    if float(text) == 0.0:
        text = f"{0.0:.{decimals}f}"
    return text + "%"


def label_place(values: np.ndarray, i: int) -> Tuple[Tuple[float, float], str, str]:
    """Offset (points), horizontal and vertical alignment of the value at dot i, on the side away from the line:
    a peak or the first dot above, a dip under, a dot on the way up above-left, on the way down above-right, the
    last dot above (under it if the line comes down to it)."""
    previous = values[i - 1] if i > 0 else np.nan
    following = values[i + 1] if i + 1 < len(values) else np.nan
    value = values[i]
    above, below = ((0, 9), "center", "bottom"), ((0, -11), "center", "top")
    if not np.isfinite(previous):
        return above
    if not np.isfinite(following):
        return above if value >= previous else below
    if value >= previous and value >= following:
        return above
    if value < previous and value < following:
        return below
    return ((-7, 7), "right", "bottom") if previous < value else ((7, 7), "left", "bottom")


def draw_figure(base: Path, summary: pd.DataFrame, group: str, decimals: int) -> None:
    """One group: the mean PCC of every stage (line and dots) and the change from Stem (%) at every dot."""
    part = summary[summary.Group == group].set_index("Stage").reindex(list(STAGES))
    values = part["Mean_PCC"].to_numpy(float)
    change = part["Change_from_Stem_Pct"].to_numpy(float)
    x = np.arange(len(STAGES))
    fig, ax = plt.subplots(figsize=(7.0, 4.9))
    ax.plot(x, values, "o-", color=COLOR, lw=2.6, ms=9.5, zorder=3)
    for i, pct in enumerate(change):
        offset, ha, va = label_place(values, i)
        ax.annotate(change_text(pct, decimals), (i, values[i]), textcoords="offset points", xytext=offset, ha=ha,
                    va=va, fontsize=15.5, color=COLOR, fontweight="bold", zorder=5,
                    bbox={"boxstyle": "round,pad=0.1", "facecolor": "white", "edgecolor": "none", "alpha": 0.9})
    finite = values[np.isfinite(values)]
    low, high = float(finite.min()), float(finite.max())
    span = max(high - low, 0.01)
    ax.set_ylim(low - 0.22 * span, high + 0.2 * span)
    ax.yaxis.set_major_locator(MaxNLocator(nbins=8, steps=[1, 2, 2.5, 5, 10]))
    ax.set_xticks(x)
    ax.set_xticklabels(STAGES)
    ax.set_xlim(-0.35, len(STAGES) - 0.65)
    ax.tick_params(axis="x", labelsize=14.5)
    ax.tick_params(axis="y", labelsize=13)
    ax.grid(True, color=GRID, lw=1.0)
    ax.set_axisbelow(True)
    for spine in ax.spines.values():
        spine.set_linewidth(1.3)
    fig.tight_layout()
    save(fig, base)


def save(fig, base: Path) -> None:
    for ext in ("png", "pdf"):
        fig.savefig(base.with_name(f"{base.name}.{ext}"), dpi=300)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Channel-averaged absolute PCC of Stem and every encoder block of the "
                                            "stored RRWaveNet models as the change from Stem (%)")
    p.add_argument("--datasets", nargs="+", choices=list(DATASETS), default=list(DATASETS))
    p.add_argument("--subjects", nargs="+", default=None, help="Dataset:Subject (default: every subject)")
    p.add_argument("--members", nargs="+", type=int, default=list(MEMBERS), help="members of the fold (default 0 1 2)")
    p.add_argument("--decimals", type=int, default=0, help="decimals of the % written at the points (default 0)")
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
    datasets = [d for d in DATASETS if d in args.datasets and (not chosen or d in chosen)]
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
    issue = subjects.Issue_Subject.astype(bool)
    print("\nMAE issue subjects: "
          + "; ".join(f"{d} {', '.join(ISSUE_SUBJECTS[d])}" for d in PANEL_ORDER if d in datasets)
          + f". In the data with stored models: {', '.join(subjects.Dataset[issue] + ' ' + subjects.Subject[issue].astype(str)) or 'none'}"
          + " - left out in the figures ('MAE issue subjects left out'); 'All subjects' in the table only")
    summaries = []
    for subject_set in SUBJECT_SETS:
        frame = subjects if subject_set == "All subjects" else subjects[~issue]
        if frame.empty:
            continue
        groups = groups_of(frame, datasets)
        summary = summary_table(subject_set, groups)
        summaries.append(summary)
        if subject_set == FIGURE_SET:
            folder = out_root / FIGURE_FOLDER
            folder.mkdir(parents=True, exist_ok=True)
            for group, _ in groups:
                draw_figure(folder / f"{group.replace(' ', '_')}_{FIGURE_NAME}", summary, group, args.decimals)
        print(f"\nChannel-averaged |PCC| (mean over the subjects) and its change from Stem: {subject_set}")
        for group, _ in groups:
            part = summary[summary.Group == group]
            print(f"  {group:<13} n {int(part.N_Subjects.iloc[0]):3d}: "
                  + ", ".join(f"{r.Stage} {r.Mean_PCC:.3f} ({change_text(r.Change_from_Stem_Pct, 1)})"
                              for r in part.itertuples(index=False)))
    summary = pd.concat(summaries, ignore_index=True)

    notes = pd.DataFrame([
        ("Models", "final results: CapnoBase 261007 full; BIDMC / STEAM2 261013, weight decay selected in the fold; "
                   f"members {' '.join(f'm{m}' for m in args.members)}, each run on the 10 s test windows of its "
                   "held-out subject"),
        ("Stages", "Stem = stem fusion output (encoder input); B1-B4 = residual encoder block outputs (d = 1, 2, 4, 8)"),
        ("PCC", "Pearson r of every channel with the target (min-max [-1, 1]) in every window; |r| averaged over the "
                "windows, then over the 24 channels: (1/24) sum |r_c| (channel_abs_r / Channel_AbsR_Mean of 261011); "
                "per model, then the mean over the members of the subject; no PCA"),
        ("Subject sets", "MAE issue subjects left out (CapnoBase 0115, BIDMC 27, STEAM2 12_YJJ, 22_LMS, 26_CKK): "
                         "figures and table; All subjects (as 261011: the MAE issue subjects included): table only; "
                         "Subjects: Issue_Subject"),
        ("Groups", "every dataset; All datasets = every subject of the three datasets once"),
        ("Change from Stem", "100 x (mean PCC of the stage - mean PCC of Stem) / mean PCC of Stem, means over the "
                             "subjects of the group; Stem = 0%"),
        ("Figures", f"{FIGURE_FOLDER}\\<STEAM2 | BIDMC | CapnoBase | All_datasets>_{FIGURE_NAME}.png / .pdf "
                    "(MAE issue subjects left out): y axis the mean PCC; at every point the change from Stem (%)"),
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
