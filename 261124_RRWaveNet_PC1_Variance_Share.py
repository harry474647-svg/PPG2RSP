# -*- coding: utf-8 -*-
r"""261124_RRWaveNet_PC1_Variance_Share: how much of the variance of the 24-channel feature map the first principal
component (PC1) explains, at the encoder input (Stem) and after every residual encoder block (B1-B4) of the stored
RRWaveNet models - for every 10 s window and for every subject - per dataset, with mean, SD, median, quartiles,
minimum and maximum (and where the minimum and the maximum are).

Models (stored, nothing is trained): the final results
  CapnoBase      261007_RRWaveNet_Ensemble_NoEvent    (D:\PPG2RSP_RRWaveNet_Inspired\261007_RRWaveNet_NoEvent_v1),
                 variant full
  BIDMC, STEAM2  261013_RRWaveNet_Adult_SubjectVal_WD (D:\PPG2RSP_RRWaveNet_Inspired\261013_RRWavNet_STEAM_BIDMC_v1),
                 the full model of the weight decay selected in the fold (WD_Selection)
  members m0, m1, m2 of every fold (the ensemble of the results). The model of a fold is run (eval mode) on the test
  windows of its held-out subject, read by its own script (the windows exactly as in the results).

Stages (24 channels x 640 samples per 10 s window, 64 Hz)
  Stem  the stem fusion output (1x1 conv + GroupNorm), the encoder input
  B1-B4 the output of the residual encoder blocks (dilation 1, 2, 4, 8; B4 = encoder output)

PC1 variance share = largest eigenvalue / sum of the eigenvalues of the covariance of the standardized channels
(= the correlation matrix of the channels; a constant channel is left out), in %. 100 / 24 = 4.2% would be 24 equal,
unrelated channels; 100% one shared waveform in every channel.
  Window   PCA of every 10 s window on its own: every channel standardized over the window's 640 samples; PC1 is the
           direction of that window. Per window and model, then the mean over the members (m0-m2) of the window.
  Subject  PCA of all test windows of a subject together, as in the feature-level analysis (PC1 of the PSD and ACF
           analyses, 261010 / 261011 / 261106 / 261111): every channel standardized over all samples of the subject,
           one PC1 direction per subject, model and stage. Per model, then the mean over the members of the subject.
  The window share is not smaller than the subject share as a rule: a direction chosen for one window fits that
  window, and the level differences between windows (part of the subject's variance) are not in it.

Statistics per dataset and over all datasets, per stage and level (Window: over every window of every subject;
Subject: over the subjects): N, mean, SD, median, Q1, Q3, minimum, maximum (%), and the subject (and window) of the
minimum and the maximum. Subject sets: every subject, and the MAE issue subjects of the final results left out
(CapnoBase 0115, BIDMC 27, STEAM2 12_YJJ, 22_LMS, 26_CKK).

Outputs (D:\PPG2RSP_RRWaveNet_Inspired\261124_RRWaveNet_PC1_Variance_Share)
  261124_RRWaveNet_PC1_Variance_Share.xlsx
    Summary   per subject set, level (Window / Subject), dataset and stage: N, mean, SD, median, Q1, Q3, min, max (%),
              where the min and the max are
    Windows   every 10 s window: PC1 share (%) of every stage, mean over the members, and the members used
    Subjects  every subject: subject-level PC1 share (%) of every stage, mean over the members; the mean and median of
              its window shares
    Members   every subject and member: subject-level PC1 share (%) of every stage
    Missing   folds without a stored model of a member (or without a weight decay selection)
    Notes
  Windows_members.csv   every window and member: PC1 share (%) of every stage

Run: python 261124_RRWaveNet_PC1_Variance_Share.py     (Spyder: runfile(...), no options needed)
Options: --datasets, --subjects CapnoBase:0009 BIDMC:05 ..., --members 0 1 2, --root-261007, --root-261013,
--output-root, --script-261007, --script-261013, --capnobase-path, --bidmc-path, --steam2-ppg-dir, --steam2-rsp-dir,
--batch-size, --cpu
"""

import argparse
import importlib.util
import json
import re
import sys
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import torch

# ---------------------------------------------------------------------------
# Settings
# ---------------------------------------------------------------------------

NAME = "261124_RRWaveNet_PC1_Variance_Share"
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
MEMBERS = (0, 1, 2)
CONSTANT_SD = 1e-8
SUBJECT_SETS = ("All subjects", "MAE issue subjects left out")

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
        name = "pc1share_src_" + re.sub(r"\W", "_", path.stem)
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


def stage_features(model, loader, device) -> List[np.ndarray]:
    """Feature maps (windows, 24, samples) at Stem and after every encoder block."""
    model.eval()
    stages: List[List[np.ndarray]] = [[] for _ in STAGES]
    with torch.no_grad():
        for batch in loader:
            features = model.fused_representation(batch[0].to(device))
            stages[0].append(features.cpu().numpy().astype(np.float64))
            for index, block in enumerate(model.encoder, start=1):
                features = block(features)
                stages[index].append(features.cpu().numpy().astype(np.float64))
    return [np.concatenate(chunks) for chunks in stages]


# ---------------------------------------------------------------------------
# PC1 variance share
# ---------------------------------------------------------------------------

def window_pc1_share(features: np.ndarray) -> np.ndarray:
    """PC1 share (%) of every window: PCA of the window's channels standardized over the window (constant channels
    left out)."""
    centred = features - features.mean(axis=2, keepdims=True)
    sd = centred.std(axis=2, keepdims=True)
    keep = sd > CONSTANT_SD
    z = np.where(keep, centred / np.where(keep, sd, 1.0), 0.0)
    correlation = np.einsum("nct,ndt->ncd", z, z) / features.shape[2]   # 1 on the diagonal of a kept channel
    eigenvalues = np.linalg.eigvalsh(correlation)
    total = eigenvalues.sum(axis=1)
    with np.errstate(divide="ignore", invalid="ignore"):
        share = np.where(total > 1e-12, eigenvalues[:, -1] / total, np.nan)
    return 100.0 * share


def subject_pc1_share(features: np.ndarray) -> float:
    """PC1 share (%) of all windows of a subject together, as first_component of the feature-level analysis:
    channels standardized over all samples of the subject (constant channels left out)."""
    windows, channels, samples = features.shape
    matrix = np.transpose(features, (1, 0, 2)).reshape(channels, -1)
    mean = matrix.mean(axis=1, keepdims=True)
    sd = matrix.std(axis=1, keepdims=True)
    keep = sd[:, 0] > CONSTANT_SD
    if not keep.any():
        return float("nan")
    z = (matrix[keep] - mean[keep]) / sd[keep]
    covariance = z @ z.T / max(1, z.shape[1] - 1)
    eigenvalues = np.linalg.eigvalsh(covariance)
    return float(100.0 * eigenvalues[-1] / max(eigenvalues.sum(), 1e-12))


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
    fs = float(mod.TARGET_FS)
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
        metadata = mod.test_metadata(data, subject)
        loader = mod.make_eval_loader(mod.SubjectWindowDataset(data, [subject], "test", False), args.batch_size,
                                      device)
        window_shares, subject_shares, used = [], [], []
        window_sec = float("nan")
        for member in args.members:
            checkpoint = root / "Members" / f"m{member}" / "Models" / f"{mod.safe_tag(subject)}_best.pth"
            if not checkpoint.exists():
                store["missing"].append({"Dataset": dataset, "Subject": subject, "Member": f"m{member}",
                                         "Missing": f"no stored model ({checkpoint})"})
                continue
            model.load_state_dict(torch.load(checkpoint, map_location=device))
            stages = stage_features(model, loader, device)
            if len(stages[0]) != len(metadata):
                raise RuntimeError(f"{dataset} {subject} m{member}: {len(metadata)} windows in the metadata, "
                                   f"{len(stages[0])} computed")
            window_sec = stages[0].shape[2] / fs
            shares = {stage: window_pc1_share(f) for stage, f in zip(STAGES, stages)}
            pooled = {stage: subject_pc1_share(f) for stage, f in zip(STAGES, stages)}
            window_shares.append(shares)
            subject_shares.append(pooled)
            used.append(f"m{member}")
            store["members"].append({"Dataset": dataset, "Subject": subject, "Issue_Subject": issue,
                                     "Model": variant, "Member": f"m{member}",
                                     **{f"{s}_PC1_Share_Pct": pooled[s] for s in STAGES}})
            for index, meta in enumerate(metadata):
                store["window_members"].append({
                    "Dataset": dataset, "Subject": subject, "Issue_Subject": issue, "Model": variant,
                    "Member": f"m{member}", "Segment_ID": int(meta["segment_id"]),
                    "Start_Sec": meta["start"] / fs, "End_Sec": meta["start"] / fs + window_sec,
                    **{f"{s}_PC1_Share_Pct": float(shares[s][index]) for s in STAGES}})
            del stages
        if not used:
            continue
        mean_window = {s: np.nanmean(np.vstack([w[s] for w in window_shares]), axis=0) for s in STAGES}
        mean_subject = {s: float(np.nanmean([p[s] for p in subject_shares])) for s in STAGES}
        for index, meta in enumerate(metadata):
            store["windows"].append({
                "Dataset": dataset, "Subject": subject, "Issue_Subject": issue, "Model": variant,
                "Members": " ".join(used), "Segment_ID": int(meta["segment_id"]), "Start_Sec": meta["start"] / fs,
                "End_Sec": meta["start"] / fs + window_sec,
                **{f"{s}_PC1_Share_Pct": float(mean_window[s][index]) for s in STAGES}})
        srow = {"Dataset": dataset, "Subject": subject, "Issue_Subject": issue, "Model": variant,
                "Members": " ".join(used), "Windows": len(metadata)}
        for s in STAGES:
            srow[f"{s}_PC1_Share_Pct"] = mean_subject[s]
        for s in STAGES:
            srow[f"{s}_Window_Share_Mean_Pct"] = float(np.nanmean(mean_window[s]))
            srow[f"{s}_Window_Share_Median_Pct"] = float(np.nanmedian(mean_window[s]))
        store["subjects"].append(srow)
        print(f"  {subject} ({variant}, {' '.join(used)}{', MAE issue' if issue else ''}): {len(metadata)} windows; "
              "PC1 share % (window mean / subject) "
              + ", ".join(f"{s} {srow[f'{s}_Window_Share_Mean_Pct']:.1f}/{mean_subject[s]:.1f}" for s in STAGES),
              flush=True)
        del loader
    del data, model
    mod.release_memory(device)


def describe(values: pd.Series, labels: pd.Series) -> dict:
    """N, mean, SD, median, quartiles, min, max of the values, and the labels of the min and the max."""
    ok = values.notna()
    v, lab = values[ok].astype(float), labels[ok]
    if v.empty:
        return {"N": 0}
    return {"N": int(v.size), "Mean": float(v.mean()), "SD": float(v.std(ddof=1)) if v.size > 1 else float("nan"),
            "Median": float(v.median()), "Q1": float(v.quantile(0.25)), "Q3": float(v.quantile(0.75)),
            "Min": float(v.min()), "Max": float(v.max()), "Min_At": str(lab.loc[v.idxmin()]),
            "Max_At": str(lab.loc[v.idxmax()])}


def summary_table(windows: pd.DataFrame, subjects: pd.DataFrame, datasets: Sequence[str]) -> pd.DataFrame:
    rows = []
    for subject_set in SUBJECT_SETS:
        w_all = windows if subject_set == SUBJECT_SETS[0] else windows[~windows.Issue_Subject.astype(bool)]
        s_all = subjects if subject_set == SUBJECT_SETS[0] else subjects[~subjects.Issue_Subject.astype(bool)]
        for level, frame in (("Window", w_all), ("Subject", s_all)):
            if frame.empty:
                continue
            if level == "Window":
                labels = (frame.Dataset + " " + frame.Subject.astype(str) + " (rec " + frame.Segment_ID.astype(str)
                          + ", " + frame.Start_Sec.map(lambda x: f"{x:.0f}") + "-"
                          + frame.End_Sec.map(lambda x: f"{x:.0f}") + " s)")
            else:
                labels = frame.Dataset + " " + frame.Subject.astype(str)
            for group in list(datasets) + [ALL]:
                part = frame if group == ALL else frame[frame.Dataset == group]
                if part.empty or (group == ALL and len(datasets) < 2):
                    continue
                for stage in STAGES:
                    row = {"Subject_Set": subject_set, "Level": level, "Dataset": group, "Stage": stage,
                           "Subjects": int(part.Subject.nunique())}
                    row.update(describe(part[f"{stage}_PC1_Share_Pct"], labels.loc[part.index]))
                    rows.append(row)
    return pd.DataFrame(rows)


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="PC1 variance share of the Stem and encoder block feature maps of the "
                                            "stored RRWaveNet models, per 10 s window and per subject")
    p.add_argument("--datasets", nargs="+", choices=list(DATASETS), default=list(DATASETS))
    p.add_argument("--subjects", nargs="+", default=None, help="Dataset:Subject (default: every subject)")
    p.add_argument("--members", nargs="+", type=int, default=list(MEMBERS), help="members of the fold (default 0 1 2)")
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
    store = {"windows": [], "window_members": [], "subjects": [], "members": [], "missing": []}
    for dataset in datasets:
        process_dataset(args, dataset, chosen, device, store)
    windows = pd.DataFrame(store["windows"])
    subjects = pd.DataFrame(store["subjects"])
    done = [d for d in datasets if not subjects.empty and d in set(subjects.Dataset)]
    summary = summary_table(windows, subjects, done) if not subjects.empty else pd.DataFrame()
    if not summary.empty:
        print("\nPC1 variance share (%), every subject: mean ± SD, median [min, max]")
        for level in ("Window", "Subject"):
            part = summary[(summary.Subject_Set == SUBJECT_SETS[0]) & (summary.Level == level)]
            print(f" {level} level")
            for r in part.itertuples(index=False):
                print(f"  {r.Dataset:<13} {r.Stage:<4} n {r.N:5d}: {r.Mean:5.1f} ± {r.SD:4.1f}, median {r.Median:5.1f} "
                      f"[{r.Min:5.1f}, {r.Max:5.1f}]")
    notes = pd.DataFrame([
        ("Models", "final results: CapnoBase 261007 full; BIDMC / STEAM2 261013, weight decay selected in the fold; "
                   f"members {' '.join(f'm{m}' for m in args.members)}, each run on the 10 s test windows of its "
                   "held-out subject"),
        ("Stages", "Stem = stem fusion output (encoder input); B1-B4 = residual encoder block outputs (d = 1, 2, 4, 8)"),
        ("PC1 share", "largest eigenvalue / sum of the eigenvalues of the covariance of the standardized channels (the "
                      "channel correlation matrix), %; constant channels left out; 100/24 = 4.2% = 24 equal unrelated "
                      "channels"),
        ("Window level", "PCA of every 10 s window on its own (channels standardized over its 640 samples); mean over "
                         "the members of the window; statistics over every window of every subject"),
        ("Subject level", "PCA of all test windows of a subject together (channels standardized over all its samples), "
                          "as the PC1 of the feature-level PSD and ACF analyses (261010 / 261011 / 261106 / 261111); "
                          "mean over the members; statistics over the subjects"),
        ("Subject sets", "every subject; and the MAE issue subjects of the final results left out (CapnoBase 0115, "
                         "BIDMC 27, STEAM2 12_YJJ, 22_LMS, 26_CKK)"),
        ("Min_At / Max_At", "dataset, subject (window: recording and seconds) of the minimum and the maximum"),
    ], columns=["Item", "Description"])
    path = out_root / f"{NAME}.xlsx"
    with pd.ExcelWriter(path, engine="openpyxl") as writer:
        summary.round(3).to_excel(writer, sheet_name="Summary", index=False)
        windows.round(3).to_excel(writer, sheet_name="Windows", index=False)
        subjects.round(3).to_excel(writer, sheet_name="Subjects", index=False)
        pd.DataFrame(store["members"]).round(3).to_excel(writer, sheet_name="Members", index=False)
        pd.DataFrame(store["missing"], columns=["Dataset", "Subject", "Member", "Missing"]).to_excel(
            writer, sheet_name="Missing", index=False)
        notes.to_excel(writer, sheet_name="Notes", index=False)
    pd.DataFrame(store["window_members"]).round(3).to_csv(out_root / "Windows_members.csv", index=False,
                                                          encoding="utf-8-sig")
    print(f"\nTables under {out_root}")


if __name__ == "__main__":
    main()
