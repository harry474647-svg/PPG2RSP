# -*- coding: utf-8 -*-
r"""261116_RRWaveNet_Waveform_30s: the respiratory waveform RRWaveNet reconstructed from the PPG against the
ground truth (GT), in every 30 s window of every subject of the final results (GT black, RRWaveNet red).

Results (stored outputs, nothing is trained; window 01_w10s_nonoverlap)
  CapnoBase      261007 (D:\PPG2RSP_RRWaveNet_Inspired\261007_RRWaveNet_NoEvent_v1), full model
  BIDMC, STEAM2  261013 (D:\PPG2RSP_RRWaveNet_Inspired\261013_RRWavNet_STEAM_BIDMC_v1), the full model of the
                 weight decay selected in the fold (WD_Selection)
  Ensemble_Predictions\{subject}_ensemble_outputs.npz   the ensemble output of every 10 s test window (64 Hz)
  Fold_Summaries\{subject}_ensemble_RR60s_windows.csv   the scored minutes (stored GT and RRWaveNet rate)
The subjects with an MAE issue are left out as in the final results (CapnoBase 0115, BIDMC 27, STEAM2
12_YJJ, 22_LMS, 26_CKK; --keep-issue-subjects keeps them).

Signals
  RRWaveNet  the windows put back on the recording's time axis (overlap-add mean), every run of contiguous
             windows band-passed with the recording's respiratory filter of the source script, as the rate
             was scored (--raw-predictions: the raw output)
  GT         the continuous reference signal of the source script (CapnoBase capnogram, BIDMC impedance,
             STEAM2 belt), band-passed and aligned in its preprocessing, the signal the GT breaths are found on
  The reference comes from the preprocessing of the script of the results (261007 for CapnoBase, 261013 for
  BIDMC / STEAM2), loaded as a module (its data paths; the path options override). No signal is shifted in
  time; in every window both waveforms are min-max scaled to [-1, 1].

Windows: 30 s (--window-sec) on a grid from the start of the recording (0-30, 30-60 s, ... = the halves of
the scored minutes); a window is drawn when RRWaveNet covers at least 90% of it. Every window of every
recording of every subject is drawn. Beside the title: the Pearson r of the two waveforms in the window and
the stored values of the minute the window lies in (scored or not, GT breath count, RRWaveNet rate, error).

Outputs (D:\PPG2RSP_RRWaveNet_Inspired\261116_RRWaveNet_Waveform_30s)
  {Dataset}\{subject}\{subject}[_rec{k}]_{t0}-{t1}s.png      one figure per window
  {Dataset}\{subject}\{Dataset}_{subject}_waveforms.pdf      every window of the subject, one page each
  261116_RRWaveNet_Waveform_30s.xlsx
    Windows   every window: coverage, Pearson r, the stored values of its minute, the figure
    Subjects  per subject: windows, mean and median r, windows with r >= 0.7
    Notes

Run: python 261116_RRWaveNet_Waveform_30s.py     (Spyder: runfile(...), no options needed)
Options: --window-sec, --datasets, --subjects CapnoBase:0332 BIDMC:05 ..., --keep-issue-subjects,
--raw-predictions, --root-261007, --root-261013, --output-root, --script-261007, --script-261013,
--capnobase-path, --bidmc-path, --steam2-ppg-dir, --steam2-rsp-dir
"""

import argparse
import importlib.util
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
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.lines import Line2D

# ---------------------------------------------------------------------------
# Settings
# ---------------------------------------------------------------------------

NAME = "261116_RRWaveNet_Waveform_30s"
HERE = Path(__file__).resolve().parent
WORKSPACE = Path(r"D:\PPG2RSP_RRWaveNet_Inspired")
OUTPUT_ROOT = WORKSPACE / NAME
ROOT_261007 = WORKSPACE / "261007_RRWaveNet_NoEvent_v1"
ROOT_261013 = WORKSPACE / "261013_RRWavNet_STEAM_BIDMC_v1"
SCRIPTS = {"261007": "261007_RRWaveNet_Ensemble_NoEvent.py", "261013": "261013_RRWaveNet_Adult_SubjectVal_WD.py"}
WINDOW = "01_w10s_nonoverlap"
DATASETS = ("CapnoBase", "BIDMC", "STEAM2")
KEYS = {"CapnoBase": "capnobase", "BIDMC": "bidmc", "STEAM2": "steam2"}
SOURCE = {"CapnoBase": "261007", "BIDMC": "261013", "STEAM2": "261013"}
MODALITY = {"CapnoBase": "capnogram", "BIDMC": "impedance", "STEAM2": "belt"}
EXCLUDED = {"STEAM2": ("22_LMS", "26_CKK", "12_YJJ"), "BIDMC": ("27",), "CapnoBase": ("0115",)}
WINDOW_SEC = 30.0
MIN_COVER = 0.9
GT_COLOR, EST_COLOR = "#000000", "#f64636"  # as 261108 / 261114: GT black, RRWaveNet red
INK, MUTED, GRID = "#1f1f1f", "#6b6a66", "#e4e3df"


# ---------------------------------------------------------------------------
# Source scripts and stored results
# ---------------------------------------------------------------------------

_MODULES: Dict[str, object] = {}


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
        name = "wave30_src_" + re.sub(r"\W", "_", path.stem)
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


def result_files(args: argparse.Namespace, mod, dataset: str, subject: str) -> Optional[Dict[str, Path]]:
    """Ensemble outputs and minute table of the fold: 261007 full (CapnoBase), 261013 selected weight decay."""
    tag = mod.safe_tag(subject)
    if SOURCE[dataset] == "261007":
        root = Path(args.root_261007) / "full" / WINDOW / dataset
        variant = "full"
    else:
        record = read_json(Path(args.root_261013) / "WD_Selection" / WINDOW / dataset / f"{tag}_selection.json") or {}
        variant = record.get("Selected_Variant")
        if not variant:
            return None
        root = Path(args.root_261013) / variant / WINDOW / dataset
    outputs = root / "Ensemble_Predictions" / f"{tag}_ensemble_outputs.npz"
    minutes = root / "Fold_Summaries" / f"{tag}_ensemble_RR60s_windows.csv"
    if not outputs.exists():
        return None
    return {"outputs": outputs, "minutes": minutes, "variant": variant}


# ---------------------------------------------------------------------------
# Signals
# ---------------------------------------------------------------------------

def series(outputs: Path, segment_id: int, length: int) -> np.ndarray:
    """Ensemble output of one recording on its 64 Hz axis (overlap-add mean; NaN where no window)."""
    with np.load(outputs, allow_pickle=False) as data:
        pred, seg, starts = data["predictions"], data["segment_ids"].astype(int), data["window_starts"].astype(int)
    pick = seg == segment_id
    out_sum, cover = np.zeros(length), np.zeros(length)
    for p, s in zip(pred[pick], starts[pick]):
        e = min(length, s + p.shape[0])
        if e <= s:
            continue
        out_sum[s:e] += np.nan_to_num(np.asarray(p, float)[: e - s])
        cover[s:e] += 1
    out = np.full(length, np.nan)
    out[cover > 0] = out_sum[cover > 0] / cover[cover > 0]
    return out


def as_scored(mod, values: np.ndarray, ref: dict) -> np.ndarray:
    """Every run of contiguous windows band-passed with the recording's respiratory filter (as scored)."""
    out = np.full(len(values), np.nan)
    ok = np.isfinite(values)
    padded = np.concatenate([[False], ok, [False]])
    for a, b in np.flatnonzero(padded[1:] != padded[:-1]).reshape(-1, 2):
        out[a:b] = mod.respiratory_filter(values[a:b], mod.TARGET_FS, ref["band_hz"], ref["profile"])
    return out


def minmax(v: np.ndarray) -> np.ndarray:
    v = np.asarray(v, float)
    ok = np.isfinite(v)
    if ok.sum() < 3 or np.ptp(v[ok]) <= 1e-12:
        return np.full_like(v, np.nan)
    return 2.0 * (v - np.nanmin(v)) / (np.nanmax(v) - np.nanmin(v)) - 1.0


def pearson(a: np.ndarray, b: np.ndarray) -> float:
    ok = np.isfinite(a) & np.isfinite(b)
    if ok.sum() < 3 or np.std(a[ok]) < 1e-12 or np.std(b[ok]) < 1e-12:
        return math.nan
    return float(np.corrcoef(a[ok], b[ok])[0, 1])


def windows(covered: np.ndarray, fs: float, size: float) -> List[Tuple[float, float]]:
    """Windows of `size` s on a grid from the start of the recording, covered at least 90%."""
    n = int(round(size * fs))
    out = []
    for k in range(int(math.ceil(len(covered) / n))):
        a, b = k * n, min(len(covered), (k + 1) * n)
        if covered[a:b].sum() >= MIN_COVER * n:
            out.append((a / fs, (a + n) / fs))
    return out


# ---------------------------------------------------------------------------
# Figure
# ---------------------------------------------------------------------------

def draw(path: Path, pdf, title: str, subtitle: str, t: np.ndarray, gt: np.ndarray, est: np.ndarray,
         gt_label: str, est_label: str, t0: float, t1: float) -> None:
    fig = plt.figure(figsize=(10.0, 4.3))
    ax = fig.add_axes([0.07, 0.13, 0.9, 0.69])
    ax.plot(t, minmax(gt), color=GT_COLOR, lw=2.0, zorder=3)
    ax.plot(t, minmax(est), color=EST_COLOR, lw=2.2, zorder=4)
    ax.set_xlim(t0, t1)
    ax.set_ylim(-1.15, 1.15)
    ax.set_xlabel("Time in the recording (s)", fontsize=9)
    ax.set_ylabel("Min-max scaled", fontsize=9)
    ax.grid(True, color=GRID, lw=0.8)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.tick_params(labelsize=8.5)
    fig.legend([Line2D([], [], color=GT_COLOR, lw=2.0), Line2D([], [], color=EST_COLOR, lw=2.2)], [gt_label, est_label],
               loc="upper right", bbox_to_anchor=(0.97, 0.985), ncol=2, frameon=False, fontsize=8.5)
    fig.text(0.07, 0.955, title, fontsize=10.5, color=INK, va="top")
    fig.text(0.07, 0.885, subtitle, fontsize=7.6, color=MUTED, va="top")
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=160)
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


def read_minutes(path: Path) -> pd.DataFrame:
    try:
        if path.exists() and path.stat().st_size > 0:
            return pd.read_csv(path, dtype={"Subject": str}, encoding="utf-8-sig")
    except (pd.errors.EmptyDataError, ValueError):
        pass
    return pd.DataFrame()


def dataset_rows(args, dataset: str, chosen: Dict[str, List[str]], out_root: Path) -> List[dict]:
    source = SOURCE[dataset]
    roots = [args.root_261007] if source == "261007" else [args.root_261013]
    script = find_script(SCRIPTS[source], args.script_261007 if source == "261007" else args.script_261013, roots)
    mod = load_script(script)
    for key, field, value in (("capnobase", "data_path", args.capnobase_path), ("bidmc", "data_path", args.bidmc_path),
                              ("steam2", "ppg_dir", args.steam2_ppg_dir), ("steam2", "rsp_dir", args.steam2_rsp_dir)):
        if value:
            mod.DATASET_CONFIGS[key][field] = str(value)
    spec = next(s for s in mod.WINDOW_SPECS if f"{s.order:02d}_{s.name}" == WINDOW)
    print(f"\n{dataset}: preprocessing of {script.name}", flush=True)
    data = mod.load_dataset(KEYS[dataset], spec)
    fs = float(mod.TARGET_FS)
    excluded = [] if args.keep_issue_subjects else list(EXCLUDED[dataset])
    rows = []
    for subject in sorted(data):
        if dataset in chosen and not any(same_subject(subject, s) for s in chosen[dataset]):
            continue
        if any(same_subject(subject, x) for x in excluded):
            print(f"  {subject}: left out (MAE issue)")
            continue
        files = result_files(args, mod, dataset, subject)
        if files is None:
            print(f"  {subject}: no stored ensemble output")
            continue
        minutes = read_minutes(files["minutes"])
        tag = mod.safe_tag(subject)
        folder = out_root / dataset / tag
        folder.mkdir(parents=True, exist_ok=True)
        references = data[subject]["references"]
        multi = len(references) > 1
        drawn = 0
        with PdfPages(folder / f"{dataset}_{tag}_waveforms.pdf") as pdf:
            for segment_id, ref in sorted(references.items()):
                gt = np.asarray(ref["signal"], dtype=np.float64)
                est = series(files["outputs"], int(segment_id), len(gt))
                if not args.raw_predictions:
                    est = as_scored(mod, est, ref)
                covered = np.isfinite(est) & np.isfinite(gt)
                seg_minutes = (minutes[minutes["Segment_ID"].astype(int) == int(segment_id)]
                               if not minutes.empty and "Segment_ID" in minutes else pd.DataFrame())
                for t0, t1 in windows(covered, fs, args.window_sec):
                    a, b = int(round(t0 * fs)), min(len(gt), int(round(t1 * fs)))
                    t = np.arange(a, b) / fs
                    r = pearson(gt[a:b], est[a:b])
                    row = {"Dataset": dataset, "Subject": subject, "Segment_ID": int(segment_id),
                           "Run_Variant": files["variant"], "Start_Sec": t0, "End_Sec": t1,
                           "Coverage": float(covered[a:b].mean()), "Pearson_r": r}
                    minute = (seg_minutes[(seg_minutes["RR_Window_Start_Sec"] <= t0 + 1e-6)
                                          & (seg_minutes["RR_Window_End_Sec"] >= t1 - 1e-6)]
                              if not seg_minutes.empty else pd.DataFrame())
                    if not minute.empty:
                        m = minute.iloc[0]
                        scored = str(m.get("Ref_Valid")).strip().lower() in ("true", "1")
                        row.update({"Minute_Start_Sec": float(m["RR_Window_Start_Sec"]),
                                    "Minute_End_Sec": float(m["RR_Window_End_Sec"]), "Minute_Scored": scored,
                                    "Stored_GT_Breaths": float(m["GT_Breath_Count"]),
                                    "Stored_RRWaveNet_Rate": float(m["Pred_Wave_FFT_RR"]),
                                    "Stored_AE": abs(float(m["Pred_Wave_FFT_RR"]) - float(m["GT_Breath_Count"]))})
                        info = (f"minute {m['RR_Window_Start_Sec']:.0f}-{m['RR_Window_End_Sec']:.0f} s "
                                f"({'scored' if scored else 'not scored'}): GT {float(m['GT_Breath_Count']):.0f} /min, "
                                f"RRWaveNet {float(m['Pred_Wave_FFT_RR']):.1f} /min (stored)")
                    else:
                        info = "outside the scored minute grid"
                    rec = f" · recording {segment_id}" if multi else ""
                    title = f"{dataset} {subject}{rec} · {t0:.0f}-{t1:.0f} s · r = {r:.2f}"
                    subtitle = (f"{info}; RRWaveNet {source} ensemble"
                                + (" (raw output)" if args.raw_predictions else ", band-passed as scored")
                                + "; both min-max scaled in the window")
                    stem = f"{tag}{f'_rec{segment_id}' if multi else ''}_{t0:.0f}-{t1:.0f}s"
                    draw(folder / f"{stem}.png", pdf, title, subtitle, t, gt[a:b], est[a:b],
                         f"GT ({MODALITY[dataset]})", "RRWaveNet", t0, t1)
                    row["Figure"] = str((folder / f"{stem}.png").relative_to(out_root))
                    rows.append(row)
                    drawn += 1
        if drawn == 0:
            (folder / f"{dataset}_{tag}_waveforms.pdf").unlink(missing_ok=True)
        rs = [r["Pearson_r"] for r in rows if r["Subject"] == subject and r["Dataset"] == dataset]
        print(f"  {subject}: {drawn} windows of {args.window_sec:g} s, median r "
              f"{np.nanmedian(rs) if rs else math.nan:.2f}", flush=True)
    del data
    return rows


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="RRWaveNet waveform vs GT in every 30 s window of every subject")
    p.add_argument("--window-sec", type=float, default=WINDOW_SEC)
    p.add_argument("--datasets", nargs="+", choices=list(DATASETS), default=list(DATASETS))
    p.add_argument("--subjects", nargs="+", default=None, help="Dataset:Subject (default: every subject)")
    p.add_argument("--keep-issue-subjects", action="store_true")
    p.add_argument("--raw-predictions", action="store_true", help="draw the raw output (default: band-passed as scored)")
    p.add_argument("--root-261007", type=Path, default=ROOT_261007)
    p.add_argument("--root-261013", type=Path, default=ROOT_261013)
    p.add_argument("--output-root", type=Path, default=OUTPUT_ROOT)
    p.add_argument("--script-261007", type=Path, default=None)
    p.add_argument("--script-261013", type=Path, default=None)
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
    rows = []
    for dataset in datasets:
        rows += dataset_rows(args, dataset, chosen, out_root)
    table = pd.DataFrame(rows)
    subjects = []
    if not table.empty:
        for (dataset, subject), g in table.groupby(["Dataset", "Subject"], sort=False):
            r = g["Pearson_r"].dropna()
            subjects.append({"Dataset": dataset, "Subject": subject, "Windows": int(len(g)),
                             "Mean_r": float(r.mean()) if len(r) else math.nan,
                             "Median_r": float(r.median()) if len(r) else math.nan,
                             "Windows_r_ge_0.7": int((r >= 0.7).sum())})
    notes = pd.DataFrame([
        ("Results", "CapnoBase 261007 full ensemble; BIDMC, STEAM2 261013 ensemble of the weight decay selected per fold"),
        ("RRWaveNet", "ensemble output put back on the recording (overlap-add mean)"
                      + ("; raw output" if args.raw_predictions else "; band-passed with the recording's respiratory "
                                                                     "filter of the source script, as scored")),
        ("GT", "continuous reference signal of the source script (CapnoBase capnogram, BIDMC impedance, STEAM2 belt), "
               "band-passed and aligned in its preprocessing"),
        ("Windows", f"{args.window_sec:g} s on a grid from the start of the recording, drawn when RRWaveNet covers at "
                    f"least {MIN_COVER:.0%}; both waveforms min-max scaled in the window; no time shift"),
        ("Pearson_r", "Pearson r of the GT and RRWaveNet waveforms in the window"),
        ("Stored_*", "values of the minute the window lies in, from the stored ensemble minute table (stored GT)"),
    ], columns=["Item", "Description"])
    path = out_root / f"{NAME}.xlsx"
    with pd.ExcelWriter(path, engine="openpyxl") as writer:
        table.round(4).to_excel(writer, sheet_name="Windows", index=False)
        pd.DataFrame(subjects).round(4).to_excel(writer, sheet_name="Subjects", index=False)
        notes.to_excel(writer, sheet_name="Notes", index=False)
    print(f"\n{len(table)} windows drawn; figures under {out_root}; table: {path}")


if __name__ == "__main__":
    main()
