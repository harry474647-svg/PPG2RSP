# -*- coding: utf-8 -*-
r"""261115_StemFusion_BIDMC_ann1_Rescore: BIDMC results of the 1x1 stem fusion ablation (Fusion O vs
Fusion X) rescored with annotator 1 (ann1) on every minute that ann1 makes scorable.

Why: the stem fusion tables were computed from the minute table of 261028_Stem_ablation_quantative_result
(minutes.csv), which holds only the minutes valid under the stored GT. The minutes that are invalid under
the stored GT but valid under ann1 (BIDMC: 21 minutes of 14 subjects, e.g. 38 180-240 s) were missing.
Here every minute is read from the stored minute files of both models, so all ann1-valid minutes count
(as in 261115_StemFusion_MAEDiff_Waveforms).

Models (stored outputs, nothing is trained): member m0 of every LOSOCV fold
  Fusion O  261013 (D:\PPG2RSP_RRWaveNet_Inspired\261013_RRWavNet_STEAM_BIDMC_v1), the weight decay
            selected in the fold (WD_Selection); read from 261028_Stem_ablation_noise_v1\full first when it
            is there (as 261115_StemFusion_MAEDiff_Waveforms)
  Fusion X  261028_Stem_ablation_noise_v1 (no_stem_fusion)
  Fold_Summaries\{subject}_m0_RR60s_windows.csv   every 60 s minute: stored rate (Pred_Wave_FFT_RR),
                                                   waveform PCC (Wave_PCC), stored GT validity (Ref_Valid)

Ground truth: breaths of annotator 1 (bidmc_data.mat ref.breaths.ann1, 125 Hz -> 64 Hz, the target lag of
261013 removed); a minute is scored with the expert-label rule of 261013 (usable reference, no artifact
label, >= 2 breaths, rate in 6-30 /min); --ann1-nogap also leaves out minutes overlapping an ann1 gap
(> 2.5 x median interval). The reference and its preprocessing come from 261013 loaded as a module.

Per subject (the same scored minutes for both models):
  FFT MAE  mean |stored rate - ann1 breath count|
  |PCC|    mean of |Wave_PCC| (every minute's PCC made positive first)
  *_Prev   the same on the minutes valid under the stored GT too (what the earlier tables used)
Summary (BIDMC without the MAE issue subject 27, and all subjects): mean +- SD and median [IQR] of the
subject values, O - X, subjects with O better / worse / the same, Wilcoxon signed-rank p (paired by
subject), FFT MAE over all scored minutes, subjects with MAE >= 2 and >= 3 breaths/min.

Outputs (D:\PPG2RSP_RRWaveNet_Inspired\261115_StemFusion_BIDMC_ann1_Rescore)
  261115_StemFusion_BIDMC_ann1_Rescore.xlsx
    Summary     the table rows (issue subject excluded / all subjects)
    Subjects    every subject: MAE and |PCC| of X and O, differences, minutes, earlier values
    Large_Diff  the subjects with |MAE O - MAE X| >= 2 breaths/min (--threshold)
    Minutes     every minute: ann1 validity and count, stored validity, rates, errors, PCC of both models
    Missing     subjects without both models
    Notes

Run: python 261115_StemFusion_BIDMC_ann1_Rescore.py     (Spyder: runfile(...), no options needed)
Options: --threshold, --ann1-nogap, --root-261013, --root-noise, --output-root, --script-261013,
--bidmc-path
"""

import argparse
import importlib.util
import json
import math
import re
import sys
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import scipy.io as sio
from scipy.stats import wilcoxon

# ---------------------------------------------------------------------------
# Settings
# ---------------------------------------------------------------------------

NAME = "261115_StemFusion_BIDMC_ann1_Rescore"
HERE = Path(__file__).resolve().parent
WORKSPACE = Path(r"D:\PPG2RSP_RRWaveNet_Inspired")
OUTPUT_ROOT = WORKSPACE / NAME
ROOT_261013 = WORKSPACE / "261013_RRWavNet_STEAM_BIDMC_v1"
ROOT_NOISE = WORKSPACE / "261028_Stem_ablation_noise_v1"
SCRIPT_261013 = "261013_RRWaveNet_Adult_SubjectVal_WD.py"
WINDOW = "01_w10s_nonoverlap"
DATASET, KEY = "BIDMC", "bidmc"
ISSUE_SUBJECTS = ("27",)
THRESHOLD_BPM = 2.0
ANN1_GAP_FACTOR = 2.5
MEMBER = 0
SAME_TOL = 1e-9  # |MAE O - MAE X| below this: the same


# ---------------------------------------------------------------------------
# Source script and stored files
# ---------------------------------------------------------------------------

def find_script(name: str, given: Optional[Path], roots: Sequence[Path]) -> Path:
    """The script that produced the results: given, written in a run manifest, or found nearby."""
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
                            "give it with --script-261013")


def load_script(path: Path):
    name = "bidmc_rescore_src_" + re.sub(r"\W", "_", path.stem)
    spec = importlib.util.spec_from_file_location(name, str(path))
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def read_json(path: Path) -> Optional[dict]:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


def model_files(args: argparse.Namespace, mod, model: str, subject: str) -> Optional[Dict[str, Path]]:
    """Minute table of member m0 of a fold: Fusion O from 261028_Stem_ablation_noise_v1\\full when there,
    otherwise 261013 (selected weight decay); Fusion X from 261028_Stem_ablation_noise_v1\\no_stem_fusion."""
    tag = mod.safe_tag(subject)
    variant = "full" if model == "O" else "no_stem_fusion"
    roots = [Path(args.root_noise) / variant / WINDOW / DATASET]
    if model == "O":
        record = read_json(Path(args.root_261013) / "WD_Selection" / WINDOW / DATASET / f"{tag}_selection.json") or {}
        if record.get("Selected_Variant"):
            roots.append(Path(args.root_261013) / record["Selected_Variant"] / WINDOW / DATASET)
    for root in roots:
        minutes = root / "Fold_Summaries" / f"{tag}_m{MEMBER}_RR60s_windows.csv"
        if minutes.exists() and minutes.stat().st_size > 0:
            return {"minutes": minutes, "root": root}
    return None


# ---------------------------------------------------------------------------
# Ground truth: ann1
# ---------------------------------------------------------------------------

def read_ann1(mod) -> Dict[str, np.ndarray]:
    mat = sio.loadmat(mod.DATASET_CONFIGS["bidmc"]["data_path"])
    return {f"{i + 1:02d}": np.asarray(r["ref"][0, 0]["breaths"][0, 0]["ann1"], float).ravel()
            for i, r in enumerate(mat["data"][0])}


def ann1_timeline(mod, ann1: np.ndarray, reference: dict) -> np.ndarray:
    """ann1 (MATLAB sample numbers at 125 Hz) as sample indices of the reference timeline."""
    fs_in = float(mod.DATASET_CONFIGS["bidmc"]["default_fs"])
    lag = int(round(float(reference["diagnostics"].get("Target_Lag_Sec", 0.0)) * mod.TARGET_FS))
    labels = np.round((np.asarray(ann1, float) - 1.0) * (mod.TARGET_FS / fs_in)).astype(np.int64) - lag
    return np.unique(labels[(labels >= 0) & (labels < len(reference["signal"]))])


def gap_spans(labels: np.ndarray, length: int, factor: float = ANN1_GAP_FACTOR) -> List[Tuple[int, int]]:
    if len(labels) < 3:
        return []
    limit = factor * float(np.median(np.diff(labels)))
    spans = [(0, int(labels[0]))] if labels[0] > limit else []
    spans += [(int(a) + 1, int(b)) for a, b in zip(labels[:-1], labels[1:]) if b - a > limit]
    if length - 1 - labels[-1] > limit:
        spans.append((int(labels[-1]) + 1, int(length)))
    return spans


def minute_gt(mod, ref: dict, start: int, end: int, breaths: np.ndarray) -> Tuple[bool, str, int]:
    """The expert-label rule of 261013 (evaluate_subject_minutes) with the ann1 breaths."""
    fs = mod.TARGET_FS
    rr_band = ref["profile"]["rr_band_bpm"]
    inside = (breaths >= start) & (breaths < end)
    count, labels = int(inside.sum()), breaths[inside]
    label_rr = float(60.0 * fs / np.mean(np.diff(labels))) if len(labels) >= 2 else math.nan
    if float(np.mean(ref["structural_valid"][start:end])) < mod.MIN_USABLE_PROPORTION:
        return False, "ref_structural_invalid", count
    if ref.get("artifact_mask") is not None and float(np.mean(ref["artifact_mask"][start:end])) > 0.0:
        return False, "ref_artifact_label", count
    if not np.isfinite(label_rr):
        return False, "ref_rr_undefined", count
    if not (rr_band[0] <= label_rr <= rr_band[1]):
        return False, "ref_out_of_band", count
    return True, "", count


# ---------------------------------------------------------------------------
# Minutes and subjects
# ---------------------------------------------------------------------------

def as_bool(value) -> bool:
    if isinstance(value, str):
        return value.strip().lower() in ("true", "1", "yes")
    return bool(value) if pd.notna(value) else False


def subject_minutes(args, mod, subject: str, record: dict, files: Dict[str, dict], ann1: np.ndarray) -> pd.DataFrame:
    """Every minute stored for both models, scored with ann1."""
    fs = mod.TARGET_FS
    tables = {}
    for model in ("O", "X"):
        t = pd.read_csv(files[model]["minutes"], dtype={"Subject": str}, encoding="utf-8-sig")
        tables[model] = t.set_index(["Segment_ID", "RR_Window_Start_Sec"])
    keys = tables["O"].index.intersection(tables["X"].index)
    rows = []
    for segment_id, t0 in keys:
        ref = record["references"].get(int(segment_id))
        if ref is None:
            continue
        o, x = tables["O"].loc[(segment_id, t0)], tables["X"].loc[(segment_id, t0)]
        t1 = float(o["RR_Window_End_Sec"])
        a, b = int(round(float(t0) * fs)), int(round(t1 * fs))
        breaths = ann1_timeline(mod, ann1, ref)
        valid, reason, count = minute_gt(mod, ref, a, b, breaths)
        gap = any(lo < b and hi > a for lo, hi in gap_spans(breaths, len(ref["signal"])))
        if gap and args.ann1_nogap and valid:
            valid, reason = False, "ref_ann1_gap"
        stored_valid = as_bool(o["Ref_Valid"]) and as_bool(x["Ref_Valid"])
        row = {"Dataset": DATASET, "Subject": subject, "Segment_ID": int(segment_id), "Start_Sec": float(t0),
               "End_Sec": t1, "Scored": valid, "Reason": "" if valid else reason, "Ann1_Count": count,
               "Ann1_Gap": gap, "Stored_GT_Valid": stored_valid, "Stored_GT_Count": o.get("GT_Breath_Count", np.nan),
               "Ann1_Only": valid and not stored_valid}
        for m, src in (("X", x), ("O", o)):
            rate, pcc = float(src["Pred_Wave_FFT_RR"]), float(src["Wave_PCC"])
            row[f"Rate_{m}"] = rate
            row[f"AE_{m}"] = abs(rate - count) if valid else math.nan
            row[f"PCC_{m}"] = pcc
            row[f"AbsPCC_{m}"] = abs(pcc) if np.isfinite(pcc) else math.nan
        rows.append(row)
    return pd.DataFrame(rows)


def subject_row(subject: str, minutes: pd.DataFrame, files: Dict[str, dict]) -> dict:
    scored = minutes[minutes.Scored]
    prev = scored[scored.Stored_GT_Valid]
    row = {"Dataset": DATASET, "Subject": subject, "Issue_Subject": subject in ISSUE_SUBJECTS,
           "Scored_Minutes": int(len(scored)), "Ann1_Only_Minutes": int(scored.Ann1_Only.sum())}
    for m in ("X", "O"):
        row[f"MAE_{m}"] = float(scored[f"AE_{m}"].mean()) if len(scored) else math.nan
    row["MAE_Diff_O_minus_X"] = row["MAE_O"] - row["MAE_X"]
    for m in ("X", "O"):
        row[f"AbsPCC_{m}"] = float(scored[f"AbsPCC_{m}"].mean()) if len(scored) else math.nan
    row["AbsPCC_Diff_O_minus_X"] = row["AbsPCC_O"] - row["AbsPCC_X"]
    row["Prev_Minutes"] = int(len(prev))
    for m in ("X", "O"):
        row[f"MAE_{m}_Prev"] = float(prev[f"AE_{m}"].mean()) if len(prev) else math.nan
        row[f"AbsPCC_{m}_Prev"] = float(prev[f"AbsPCC_{m}"].mean()) if len(prev) else math.nan
    row["Files_O"], row["Files_X"] = str(files["O"]["root"]), str(files["X"]["root"])
    return row


# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------

def mean_sd(v: np.ndarray, digits: int) -> str:
    return f"{np.mean(v):.{digits}f} ± {np.std(v, ddof=1):.{digits}f}" if len(v) > 1 else "n/a"


def median_iqr(v: np.ndarray, digits: int) -> str:
    q1, q2, q3 = np.percentile(v, [25, 50, 75])
    return f"{q2:.{digits}f} [{q1:.{digits}f}, {q3:.{digits}f}]"


def wilcoxon_p(o: np.ndarray, x: np.ndarray) -> float:
    if len(o) < 2 or np.allclose(o, x):
        return math.nan if len(o) < 2 else 1.0
    try:
        return float(wilcoxon(o, x).pvalue)
    except ValueError:
        return math.nan


def summary_row(label: str, subjects: pd.DataFrame, minutes: pd.DataFrame) -> dict:
    s = subjects.dropna(subset=["MAE_X", "MAE_O"])
    x, o = s.MAE_X.to_numpy(float), s.MAE_O.to_numpy(float)
    px, po = s.AbsPCC_X.to_numpy(float), s.AbsPCC_O.to_numpy(float)
    d = o - x
    m = minutes[minutes.Scored & minutes.Subject.isin(s.Subject)]
    return {
        "Set": label, "N_Subjects": int(len(s)), "Scored_Minutes": int(len(m)),
        "Ann1_Only_Minutes": int(m.Ann1_Only.sum()),
        "FFT_MAE_X (mean ± SD)": mean_sd(x, 2), "FFT_MAE_O (mean ± SD)": mean_sd(o, 2),
        "FFT_MAE_O_minus_X": round(float(np.mean(d)), 3),
        "O_Better/Worse/Same": f"{int((d < -SAME_TOL).sum())}/{int((d > SAME_TOL).sum())}/"
                               f"{int((np.abs(d) <= SAME_TOL).sum())}",
        "FFT_MAE_Wilcoxon_p": wilcoxon_p(o, x),
        "Minute_MAE_X->O (minutes)": f"{m.AE_X.mean():.2f} -> {m.AE_O.mean():.2f} ({len(m)})" if len(m) else "n/a",
        "FFT_MAE_X (median [IQR])": median_iqr(x, 2), "FFT_MAE_O (median [IQR])": median_iqr(o, 2),
        "MAE_SD_X": round(float(np.std(x, ddof=1)), 3), "MAE_SD_O": round(float(np.std(o, ddof=1)), 3),
        "MAE>=2_X/O": f"{int((x >= 2).sum())}/{int((o >= 2).sum())}",
        "MAE>=3_X/O": f"{int((x >= 3).sum())}/{int((o >= 3).sum())}",
        "AbsPCC_X (mean ± SD)": mean_sd(px, 3), "AbsPCC_O (mean ± SD)": mean_sd(po, 3),
        "AbsPCC_O_minus_X": round(float(np.mean(po - px)), 4),
        "AbsPCC_Wilcoxon_p": wilcoxon_p(po, px),
        "AbsPCC_X (median [IQR])": median_iqr(px, 3), "AbsPCC_O (median [IQR])": median_iqr(po, 3),
        "AbsPCC_SD_X": round(float(np.std(px, ddof=1)), 3), "AbsPCC_SD_O": round(float(np.std(po, ddof=1)), 3),
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="BIDMC stem fusion results (Fusion O vs X) rescored with ann1 on every "
                                            "ann1-valid minute")
    p.add_argument("--threshold", type=float, default=THRESHOLD_BPM, help="|MAE O - MAE X| listed (breaths/min)")
    p.add_argument("--ann1-nogap", action="store_true", help="leave out minutes overlapping an ann1 gap")
    p.add_argument("--root-261013", type=Path, default=ROOT_261013)
    p.add_argument("--root-noise", type=Path, default=ROOT_NOISE, help="results of 261028_Stem_ablation_noise_v1")
    p.add_argument("--output-root", type=Path, default=OUTPUT_ROOT)
    p.add_argument("--script-261013", type=Path, default=None)
    p.add_argument("--bidmc-path", type=Path, default=None)
    return p.parse_args(list(sys.argv[1:] if argv is None else argv))


def main(argv: Optional[Sequence[str]] = None) -> None:
    args = parse_args(argv)
    out_root = Path(args.output_root)
    out_root.mkdir(parents=True, exist_ok=True)
    script = find_script(SCRIPT_261013, args.script_261013, [args.root_261013])
    mod = load_script(script)
    if args.bidmc_path:
        mod.DATASET_CONFIGS["bidmc"]["data_path"] = str(args.bidmc_path)
    spec = next(s for s in mod.WINDOW_SPECS if f"{s.order:02d}_{s.name}" == WINDOW)
    print(f"BIDMC: preprocessing of {script.name}", flush=True)
    data = mod.load_dataset(KEY, spec)
    ann1_all = read_ann1(mod)

    subject_rows, minute_frames, missing = [], [], []
    for subject in sorted(data):
        files = {m: model_files(args, mod, m, subject) for m in ("O", "X")}
        if not files["O"] or not files["X"]:
            missing.append({"Subject": subject, "Fusion_O": bool(files["O"]), "Fusion_X": bool(files["X"])})
            continue
        if subject not in ann1_all:
            missing.append({"Subject": subject, "Fusion_O": True, "Fusion_X": True, "Note": "no ann1"})
            continue
        minutes = subject_minutes(args, mod, subject, data[subject], files, ann1_all[subject])
        if minutes.empty or not minutes.Scored.any():
            missing.append({"Subject": subject, "Fusion_O": True, "Fusion_X": True, "Note": "no ann1-valid minute"})
            continue
        minute_frames.append(minutes)
        row = subject_row(subject, minutes, files)
        subject_rows.append(row)
        print(f"  {subject}: MAE X {row['MAE_X']:.2f}, O {row['MAE_O']:.2f} (O - X {row['MAE_Diff_O_minus_X']:+.2f}); "
              f"|PCC| X {row['AbsPCC_X']:.3f}, O {row['AbsPCC_O']:.3f}; {row['Scored_Minutes']} minutes "
              f"({row['Ann1_Only_Minutes']} valid only with ann1){' [issue]' if row['Issue_Subject'] else ''}",
              flush=True)
    if not subject_rows:
        print("No BIDMC subject with both models and an ann1-valid minute.")
        return

    subjects = pd.DataFrame(subject_rows)
    minutes_all = pd.concat(minute_frames, ignore_index=True)
    summary = pd.DataFrame([
        summary_row(f"BIDMC (ann1{', no gap' if args.ann1_nogap else ''}), without the issue subject "
                    f"{', '.join(ISSUE_SUBJECTS)}", subjects[~subjects.Issue_Subject], minutes_all),
        summary_row(f"BIDMC (ann1{', no gap' if args.ann1_nogap else ''}), all subjects", subjects, minutes_all),
    ])
    large = subjects[subjects.MAE_Diff_O_minus_X.abs() >= args.threshold].sort_values("MAE_Diff_O_minus_X")
    notes = pd.DataFrame([
        ("Models", "member m0 of every LOSOCV fold; Fusion O = with the 1x1 stem fusion (261013, selected weight "
                   "decay), Fusion X = without (261028_Stem_ablation_noise_v1)"),
        ("GT", "ann1 breaths; expert-label rule of 261013 (usable reference, no artifact label, >= 2 breaths, "
               "rate in 6-30 /min)" + ("; minutes overlapping an ann1 gap left out" if args.ann1_nogap else "")),
        ("Minutes", "every minute stored for both models and valid with ann1, the same minutes for both models; "
                    "Ann1_Only = valid with ann1 but not with the stored GT (missing from the earlier tables)"),
        ("FFT MAE", "per subject, mean |stored rate - ann1 breath count| over the scored minutes"),
        ("|PCC|", "per subject, mean of |Wave_PCC| over the scored minutes (every PCC made positive first)"),
        ("*_Prev", "the same on the minutes valid with the stored GT too (as the earlier tables)"),
        ("Summary", "mean ± SD and median [IQR] over subjects; O better / worse / same by subject MAE; Wilcoxon "
                    "signed-rank p, paired by subject; MAE >= 2 / >= 3 = subjects of X / O at or above"),
        ("Large_Diff", f"|MAE O - MAE X| >= {args.threshold:g} breaths/min"),
    ], columns=["Item", "Description"])

    path = out_root / f"{NAME}.xlsx"
    with pd.ExcelWriter(path, engine="openpyxl") as writer:
        summary.to_excel(writer, sheet_name="Summary", index=False)
        subjects.sort_values("MAE_Diff_O_minus_X").round(4).to_excel(writer, sheet_name="Subjects", index=False)
        large.round(4).to_excel(writer, sheet_name="Large_Diff", index=False)
        minutes_all.round(4).to_excel(writer, sheet_name="Minutes", index=False)
        pd.DataFrame(missing, columns=["Subject", "Fusion_O", "Fusion_X", "Note"]).to_excel(
            writer, sheet_name="Missing", index=False)
        notes.to_excel(writer, sheet_name="Notes", index=False)

    print("\nSummary")
    for _, r in summary.iterrows():
        print(f"  {r['Set']}: n={r['N_Subjects']}, minutes {r['Scored_Minutes']} ({r['Ann1_Only_Minutes']} ann1 only)\n"
              f"    FFT MAE X {r['FFT_MAE_X (mean ± SD)']}, O {r['FFT_MAE_O (mean ± SD)']} "
              f"(O better/worse/same {r['O_Better/Worse/Same']}, p={r['FFT_MAE_Wilcoxon_p']:.3g}); "
              f"minute MAE {r['Minute_MAE_X->O (minutes)']}; MAE>=3 X/O {r['MAE>=3_X/O']}\n"
              f"    |PCC| X {r['AbsPCC_X (mean ± SD)']}, O {r['AbsPCC_O (mean ± SD)']} (p={r['AbsPCC_Wilcoxon_p']:.3g})")
    print(f"\n|MAE O - MAE X| >= {args.threshold:g}: "
          + (", ".join(f"{r.Subject} ({r.MAE_X:.2f} -> {r.MAE_O:.2f})" for r in large.itertuples()) or "none"))
    if missing:
        print(f"Left out: {', '.join(str(m['Subject']) for m in missing)}")
    print(f"Table: {path}")


if __name__ == "__main__":
    main()
