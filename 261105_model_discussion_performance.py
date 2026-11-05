r"""261105_model_discussion_performance: the subjects and the 30 s / 60 s windows
that show most clearly what each model does wrong or right, for the
discussion of RespDiff, the deep corr-encoder and RRWaveNet (stored test
results of every subject; nothing is trained).

Observations (each listed per dataset, the --top subjects where it is clearest)
-----------------------------------------------------------------------------
  ce_double    the corr-encoder at twice the breathing rate while RRWaveNet is
               right (harmonic-rich capnograms with a short inspiration)
  rd_missed    RespDiff misses breathing cycles or is at half the rate while
               RRWaveNet is right (faster breathing)
  rd_double    RespDiff at twice the rate while RRWaveNet is right (slow
               breathing)
  rr_coverage  RRWaveNet detects breathing cycles that the other two miss
In a window
  rate         FFT peak of the model's waveform in the rate band (CapnoBase
               3-60, BIDMC / STEAM2 6-30 breaths/min); ratio = rate / GT rate;
               AE = |rate - GT rate|, GT rate from the breaths in the window
               (CapnoBase expert labels, BIDMC ann1; STEAM2: the GT cycles)
  cycles       breathing cycles of the GT and of every model (261030's
               detector on the aligned waveforms); a GT cycle inside the window
               is matched by a model cycle starting within half its length
               (1:1); Missed = GT cycles not matched, Extra = model cycles inside
               the window not matched, Coverage = matched / GT cycles
  "RRWaveNet right": AE <= 1 breath/min
  ce_double    corr-encoder ratio 1.75-2.25, RRWaveNet right; score = AE gap
  rd_double    RespDiff ratio 1.75-2.25, RRWaveNet right; score = AE gap
  rd_missed    RRWaveNet right and missing <= 1 cycle, RespDiff missing >= 2
               cycles more than RRWaveNet or at ratio 0.4-0.6; score = cycles
               missed by RespDiff minus by RRWaveNet + 0.1 x AE gap (<= 5)
  rr_coverage  >= 3 GT cycles, RRWaveNet coverage >= the best other + 0.15,
               RRWaveNet extra <= 1 (so that cycles found twice do not count);
               score = the coverage margin
Subjects: on the non-overlapping windows (from 0 s) of every size, the share
of windows that show the observation and the mean score of those windows;
ranked once per dataset by the mean of both over the sizes; subjects without
such a window are not listed (a dataset may list fewer than --top). Windows:
per listed subject and every window size, the window (start every --step s)
with the highest score.

Waveforms (as in the overlay figures and 261105_Model_Strength_Analysis,
whose readers this script imports): RespDiff and the corr-encoder as stored,
RRWaveNet (as 261104: CapnoBase from 261007, the ensemble of the full model;
BIDMC / STEAM2 from 261013, the ensemble of the weight decay selected per fold;
--rrwavenet-member m0 / m1 / m2 for one member instead) with its training sign
/ lag undone and low-pass 1 Hz; every model shifted by at most half a breath
and, if needed, inverted once per recording to its best r with the GT (the
same rule for all).

Outputs (default D:\PPG2RSP_RRWaveNet_Inspired\261105_model_discussion_performance)
  discussion_subjects.xlsx   Subjects_{observation} (every subject, ranked,
                             Listed = among the top), Windows (the window of
                             every listed subject and size), Notes
  windows_30s.csv, windows_60s.csv   every window (start every --step s)
  {observation}\{dataset}\{rank}_{subject}_{size}s.png and
  {observation}_{dataset}.pdf   one figure per listed subject and size: one
                             row per model (GT and the model, min-max scaled
                             in the window), GT inspirations as a gray
                             background, the model's inspirations as bars,
                             GT cycles the model missed marked, rate / AE /
                             PCC / cycles beside every row

Run:                 python 261105_model_discussion_performance.py
Result folders:      --respdiff-root, --correncoder-root, --rrwavenet-261007-root,
                     --rrwavenet-261013-root
Data files:          --capnobase-path, --bidmc-path, --steam2-ppg-dir, --steam2-rsp-dir
Some datasets:       --datasets capnobase bidmc
"""

import argparse
import importlib.util
import math
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
from matplotlib.patches import Patch

HERE = Path(__file__).resolve().parent


def load_script(name: str, filename: str):
    spec = importlib.util.spec_from_file_location(name, HERE / filename)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


msa = load_script("model_strength_261105", "261105_Model_Strength_Analysis.py")
ov = msa.ov
FS = msa.FS

PROJECT_ROOT = Path(r"D:\PPG2RSP_RRWaveNet_Inspired\261105_model_discussion_performance")
MODELS = ("RespDiff", "Deep corr-encoder", "RRWaveNet")
OTHERS = ("RespDiff", "Deep corr-encoder")
WINDOW_SIZES = (30, 60)
STEP_SEC = 5.0
TOP = 5
RATE_OK_BPM = 1.0
DOUBLE = (1.75, 2.25)
HALF = (0.4, 0.6)
MISSED_MIN = 2
COVERAGE_MARGIN = 0.15
GT_BAND = "#dddcd8"
MISSED_COLOR = "#b3261e"
OBSERVATIONS = {
    "ce_double": ("Deep corr-encoder at twice the rate, RRWaveNet right", ("Deep corr-encoder", "RRWaveNet")),
    "rd_missed": ("RespDiff misses breathing cycles / at half the rate, RRWaveNet right", ("RespDiff", "RRWaveNet")),
    "rd_double": ("RespDiff at twice the rate, RRWaveNet right", ("RespDiff", "RRWaveNet")),
    "rr_coverage": ("RRWaveNet detects the breathing cycles the others miss", MODELS),
}


# ---------------------------------------------------------------------------
# Windows
# ---------------------------------------------------------------------------

def match_cycles(gt: List[dict], model: List[dict]) -> Tuple[int, List[dict]]:
    """1:1 matching: a GT cycle is matched by the nearest unmatched model cycle
    starting within half the GT cycle's length; returns the number matched and
    the GT cycles left unmatched."""
    used, missed, matched = set(), [], 0
    starts = np.array([c["start"] for c in model]) if model else np.empty(0)
    for c in gt:
        tolerance = 0.5 * (c["end"] - c["start"])
        best, best_gap = None, np.inf
        for k in np.flatnonzero(np.abs(starts - c["start"]) <= tolerance):
            gap = abs(starts[k] - c["start"])
            if k not in used and gap < best_gap:
                best, best_gap = int(k), gap
        if best is None:
            missed.append(c)
        else:
            used.add(best)
            matched += 1
    return matched, missed


def window_row(values: Dict[str, np.ndarray], cycles: Dict[str, List[dict]], breaths: np.ndarray, t0: float,
               t1: float, band: Tuple[float, float]) -> Optional[dict]:
    a, b = int(round(t0 * FS)), int(round(t1 * FS))
    if not all(np.isfinite(v[a:b]).all() for v in values.values()):
        return None
    br = breaths[(breaths >= t0) & (breaths < t1)]
    if len(br) < msa.MIN_BREATHS:
        return None
    gt_cycles = msa.inside(cycles["GT"], t0, t1)
    clean = [c for c in gt_cycles if not c.get("clipped", False)]
    gt_rr = msa.rate_of(br)
    row = {"Start_Sec": t0, "End_Sec": t1, "GT_RR": gt_rr, "GT_Cycles": len(gt_cycles),
           "GT_Clipped_Cycles": len(gt_cycles) - len(clean), "GT_TiTtot": msa.ti_ttot(clean),
           "GT_Harmonic2": msa.harmonic2(values["GT"][a:b], FS, gt_rr)}
    for model in MODELS:
        if model not in values:
            continue
        seg = values[model][a:b]
        rate = msa.fft_rate(seg, FS, band)
        near = [c for c in cycles[model] if c["end"] > t0 and c["start"] < t1]
        matched, _ = match_cycles(gt_cycles, near)
        inside = msa.inside(cycles[model], t0, t1)
        row.update({f"{model}_Rate": rate, f"{model}_Ratio": rate / gt_rr, f"{model}_AE": abs(rate - gt_rr),
                    f"{model}_PCC": ov.pearson(values["GT"][a:b], seg), f"{model}_Cycles": len(inside),
                    f"{model}_Matched": matched, f"{model}_Missed": len(gt_cycles) - matched,
                    f"{model}_Extra": max(0, len(inside) - matched),
                    f"{model}_Coverage": matched / len(gt_cycles) if gt_cycles else math.nan})
    return row


def observe(row: dict, key: str) -> Tuple[bool, float]:
    """(the window shows the observation, its score)."""
    needs = OBSERVATIONS[key][1]
    if any(not np.isfinite(row.get(f"{m}_AE", math.nan)) for m in needs):
        return False, math.nan
    rr_ok = row["RRWaveNet_AE"] <= RATE_OK_BPM
    if key in ("ce_double", "rd_double"):
        model = "Deep corr-encoder" if key == "ce_double" else "RespDiff"
        ok = rr_ok and DOUBLE[0] <= row[f"{model}_Ratio"] <= DOUBLE[1]
        return bool(ok), row[f"{model}_AE"] - row["RRWaveNet_AE"]
    if key == "rd_missed":
        more = row["RespDiff_Missed"] - row["RRWaveNet_Missed"]
        half = HALF[0] <= row["RespDiff_Ratio"] <= HALF[1]
        ok = rr_ok and row["RRWaveNet_Missed"] <= 1 and (more >= MISSED_MIN or half)
        return bool(ok), more + 0.1 * min(max(row["RespDiff_AE"] - row["RRWaveNet_AE"], 0.0), 5.0)
    others = max(row[f"{m}_Coverage"] for m in OTHERS)
    margin = row["RRWaveNet_Coverage"] - others
    ok = row["GT_Cycles"] >= 3 and margin >= COVERAGE_MARGIN and row["RRWaveNet_Extra"] <= 1
    return bool(ok), margin


# ---------------------------------------------------------------------------
# Reading and windowing every recording
# ---------------------------------------------------------------------------

def dataset_windows(dataset_key: str, args, store: dict) -> List[dict]:
    label = ov.DATASET_CONFIGS[dataset_key]["label"]
    modality = ov.MODALITIES[dataset_key]
    band = msa.RATE_BAND_BPM[dataset_key]
    corr = ov.load_correncoder(Path(args.correncoder_root), label)
    respdiff = ov.load_respdiff(Path(args.respdiff_root), label, args.respdiff_sampling)
    rr_files, rr_root = msa.rrwavenet_sources(dataset_key, args)
    rr_alignment = ov.rrwavenet_alignment_table(rr_root, label)
    subjects = sorted(set(rr_files) & (set(corr) | set(respdiff)))
    if args.subjects:
        subjects = [s for s in subjects if s in {str(x) for x in args.subjects}]
    print(f"{label}: results - RespDiff {len(respdiff)}, corr-encoder {len(corr)}, RRWaveNet {len(rr_files)}; "
          f"{len(subjects)} subjects with RRWaveNet and at least one other model", flush=True)
    if not subjects:
        return []
    recordings = msa.read_recordings(dataset_key)
    rows = []
    for subject in subjects:
        if subject not in recordings:
            continue
        segments = ov.load_rrwavenet(rr_files[subject][0])
        corr_by_name = {r["name"]: r for r in corr.get(subject, [])}
        for k, (name, rec) in enumerate(recordings[subject].items()):
            gt = np.asarray(rec["gt"], float)
            times = np.arange(len(gt)) / FS
            index = corr_by_name[name]["index"] if name in corr_by_name else k
            raw = {}
            if name in respdiff.get(subject, {}):
                raw["RespDiff"] = ov.Track(respdiff[subject][name], FS).at(times)
            if name in corr_by_name:
                raw["Deep corr-encoder"] = ov.Track(corr_by_name[name]["output"], FS).at(times)
            if index in segments:
                raw["RRWaveNet"] = msa.rrwavenet_track(segments[index], rr_alignment.get((subject, index)),
                                                       dataset_key, gt, args.rrwavenet_lowpass_hz)
            if "RRWaveNet" not in raw or len(raw) < 2:
                continue
            breaths = rec["breaths"]
            if breaths is None or len(breaths) < msa.MIN_BREATHS:
                breaths = np.array([c["start"] for c in msa.recording_cycles({"GT": gt}, modality)["GT"]])
            period = float(np.median(np.diff(breaths))) if len(breaths) >= 2 else 60.0 / np.mean(band)
            values = {"GT": gt}
            for model, v in raw.items():
                values[model], _ = msa.align(gt, v, period)
            cycles = msa.recording_cycles(values, modality)
            clipped = np.asarray(rec["clipped"], bool)
            for c in cycles["GT"]:
                c["clipped"] = bool(clipped[int(c["start"] * FS):int(math.ceil(c["end"] * FS)) + 1].any())
            store[(label, subject, name)] = {"values": values, "cycles": cycles}
            length = len(gt) / FS
            for size in args.sizes:
                grid = set(np.arange(0.0, length - size + 1e-9, size).round(3))
                starts = sorted(grid | set(np.arange(0.0, length - size + 1e-9, args.step).round(3)))
                for t0 in starts:
                    row = window_row(values, cycles, breaths, float(t0), float(t0) + size, band)
                    if row is None:
                        continue
                    row = {"Dataset": label, "Subject": subject, "Recording": name, "Size_Sec": size,
                           "Grid": t0 in grid, **row}
                    for key in OBSERVATIONS:
                        row[f"{key}_shown"], row[f"{key}_score"] = observe(row, key)
                    rows.append(row)
        print(f"  {subject}: models {', '.join(m for m in MODELS if m in raw)}", flush=True)
    return rows


# ---------------------------------------------------------------------------
# Subjects and windows of every observation
# ---------------------------------------------------------------------------

def size_table(win: pd.DataFrame, key: str, size: int) -> pd.DataFrame:
    needs = OBSERVATIONS[key][1]
    part = win[(win.Size_Sec == size) & win.Grid]
    part = part[part[[f"{m}_AE" for m in needs]].notna().all(axis=1)]
    rows = []
    for (ds, s), d in part.groupby(["Dataset", "Subject"]):
        shown = d[d[f"{key}_shown"]]
        row = {"Dataset": ds, "Subject": s, "Size_Sec": size, "Windows": len(d), "Shown_Windows": len(shown),
               "Shown_Share": len(shown) / len(d), "Mean_Score_Shown": shown[f"{key}_score"].mean(),
               "GT_RR_median": d.GT_RR.median(), "GT_TiTtot_median": d.GT_TiTtot.median(),
               "GT_Harmonic2_median": d.GT_Harmonic2.median(), "GT_Cycles": int(d.GT_Cycles.sum())}
        for m in MODELS:
            if f"{m}_AE" in d and d[f"{m}_AE"].notna().all():
                row.update({f"{m}_AE_mean": d[f"{m}_AE"].mean(), f"{m}_Ratio_median": d[f"{m}_Ratio"].median(),
                            f"{m}_PCC_mean": d[f"{m}_PCC"].mean(),
                            f"{m}_Coverage": d[f"{m}_Matched"].sum() / max(d.GT_Cycles.sum(), 1),
                            f"{m}_Extra": int(d[f"{m}_Extra"].sum())})
        rows.append(row)
    return pd.DataFrame(rows)


def subject_table(win: pd.DataFrame, key: str, sizes: Sequence[int], top: int) -> pd.DataFrame:
    """Every subject per window size, ranked once per dataset by the mean over
    the sizes of the share of windows showing the observation, then of the
    mean score of those windows; Listed = the top subjects with such a window."""
    tables = [t for t in (size_table(win, key, size) for size in sizes) if not t.empty]
    if not tables:
        return pd.DataFrame()
    long = pd.concat(tables, ignore_index=True)
    combined = long.groupby(["Dataset", "Subject"]).agg(
        Combined_Share=("Shown_Share", "mean"), Combined_Score=("Mean_Score_Shown", "mean"),
        Shown_Windows_All_Sizes=("Shown_Windows", "sum")).reset_index()
    combined = combined.sort_values(["Dataset", "Combined_Share", "Combined_Score"], ascending=[True, False, False])
    combined["Rank"] = combined.groupby("Dataset").cumcount() + 1
    combined["Listed"] = (combined.Shown_Windows_All_Sizes > 0) & (combined.Rank <= top)
    return long.merge(combined, on=["Dataset", "Subject"]).sort_values(["Dataset", "Rank", "Size_Sec"])


def best_windows(win: pd.DataFrame, listed: pd.DataFrame, key: str, size: int) -> List[dict]:
    rows = []
    for r in listed.itertuples():
        part = win[(win.Dataset == r.Dataset) & (win.Subject == r.Subject) & (win.Size_Sec == size)
                   & win[f"{key}_shown"]]
        if part.empty:
            continue
        best = part.loc[part[f"{key}_score"].idxmax()]
        rows.append({"Observation": key, "Rank": int(r.Rank), **best.to_dict()})
    return rows


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------

def draw(key: str, row: dict, data: dict, path: Path) -> plt.Figure:
    values, cycles = data["values"], data["cycles"]
    models = [m for m in MODELS if m in values]
    t0, t1 = row["Start_Sec"], row["End_Sec"]
    a, b = int(round(t0 * FS)), int(round(t1 * FS))
    t = np.arange(a, b) / FS
    gt_in = msa.inside(cycles["GT"], t0, t1)
    fig = plt.figure(figsize=(15.5, 1.55 + 1.7 * len(models)))
    h = fig.get_figheight()
    top, bottom, gap = 1 - 0.95 / h, 0.62 / h, 0.25 / h
    ph = (top - bottom - gap * (len(models) - 1)) / len(models)
    for p, model in enumerate(models):
        ax = fig.add_axes([0.05, top - (p + 1) * ph - p * gap, 0.665, ph])
        for c in cycles["GT"]:
            if c["end"] > t0 and c["start"] < t1:
                ax.axvspan(max(c["start"], t0), min(c["transition"], t1), color=GT_BAND, lw=0, zorder=0)
        for c in cycles[model]:
            if c["end"] > t0 and c["start"] < t1:
                ax.fill_between([max(c["start"], t0), min(c["transition"], t1)], -1.42, -1.24,
                                color=ov.MODEL_COLORS[model], alpha=0.8, lw=0)
        _, missed = match_cycles(gt_in, [c for c in cycles[model] if c["end"] > t0 and c["start"] < t1])
        if missed:
            ax.plot([c["start"] for c in missed], [1.28] * len(missed), "v", color=MISSED_COLOR, ms=7, zorder=5)
        ax.plot(t, ov.minmax(values["GT"][a:b]), color=ov.GT_COLOR, lw=2.0, alpha=0.85, zorder=2)
        ax.plot(t, ov.minmax(values[model][a:b]), color=ov.MODEL_COLORS[model], lw=1.6, zorder=3)
        ax.set_xlim(t0, t1)
        ax.set_ylim(-1.5, 1.45)
        ax.set_yticks([])
        ov.style_axis(ax)
        ax.spines["left"].set_visible(False)
        if p < len(models) - 1:
            ax.set_xticklabels([])
        text = (f"{model}\nrate {row[f'{model}_Rate']:5.1f} /min ({row[f'{model}_Ratio']:.2f} x GT)  "
                f"AE {row[f'{model}_AE']:5.2f}\nPCC {row[f'{model}_PCC']:5.2f}   cycles matched "
                f"{int(row[f'{model}_Matched'])}/{int(row['GT_Cycles'])}, extra {int(row[f'{model}_Extra'])}")
        fig.text(0.725, top - p * (ph + gap) - ph / 2, text, family="monospace", fontsize=8.6,
                 color=ov.TEXT_PRIMARY, va="center", ha="left")
    recording = "" if str(row["Recording"]) == str(row["Subject"]) else f" ({row['Recording']})"
    fig.text(0.05, 1 - 0.32 / h, f"{OBSERVATIONS[key][0]}: {row['Dataset']} {row['Subject']}{recording}, "
             f"{t0:.0f}-{t1:.0f} s ({t1 - t0:.0f} s)", fontsize=11.5, color=ov.TEXT_PRIMARY, va="center")
    fig.text(0.05, 1 - 0.62 / h, f"GT {row['GT_RR']:.1f} breaths/min, {int(row['GT_Cycles'])} cycles in the window; x axis: time "
             f"in the recording (s); each waveform min-max scaled to [-1, 1] in the window; every model shifted <= half a breath and, "
             f"if needed, inverted once per recording (same rule for all)", fontsize=8.5, color=ov.TEXT_SECONDARY,
             va="center")
    handles = [Line2D([0], [0], color=ov.GT_COLOR, lw=2.0)] + \
              [Line2D([0], [0], color=ov.MODEL_COLORS[m], lw=1.8) for m in models] + \
              [Patch(color=GT_BAND), Patch(color="#8f8e8a"),
               Line2D([0], [0], marker="v", ls="", color=MISSED_COLOR, ms=7)]
    labels = ["GT"] + list(models) + ["GT inspiration", "model inspiration (bar, model colour)",
                                       "GT cycle the model missed"]
    fig.legend(handles, labels, loc="lower center", ncol=len(labels), frameon=False, fontsize=8.8,
               bbox_to_anchor=(0.45, 0.0))
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=130)
    return fig


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def rrwavenet_note(args) -> str:
    return (f"{args.rrwavenet_member}: CapnoBase from 261007, full model ({args.rrwavenet_261007_root}); "
            f"BIDMC / STEAM2 from 261013, weight decay selected per fold ({args.rrwavenet_261013_root})")


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--datasets", nargs="+", choices=list(ov.DATASET_ORDER), default=list(ov.DATASET_ORDER))
    parser.add_argument("--subjects", nargs="+", default=None, help="Only these subjects.")
    parser.add_argument("--sizes", nargs="+", type=int, default=list(WINDOW_SIZES), help="Window sizes (s).")
    parser.add_argument("--step", type=float, default=STEP_SEC, help="Start of a window every this many s.")
    parser.add_argument("--top", type=int, default=TOP, help="Subjects listed per observation and dataset.")
    parser.add_argument("--respdiff-root", type=Path, default=ov.RESPDIFF_ROOT)
    parser.add_argument("--respdiff-sampling", default=ov.RESPDIFF_SAMPLING)
    parser.add_argument("--correncoder-root", type=Path, default=ov.CORRENCODER_ROOT)
    parser.add_argument("--rrwavenet-261007-root", type=Path, default=ov.RRWAVENET_261007_ROOT,
                        help="261007 results folder (CapnoBase, full model).")
    parser.add_argument("--rrwavenet-261013-root", type=Path, default=ov.RRWAVENET_261013_ROOT,
                        help="261013 results folder (BIDMC / STEAM2, weight decay selected per fold).")
    parser.add_argument("--rrwavenet-member", choices=["ensemble", "m0", "m1", "m2"], default="ensemble",
                        help="RRWaveNet output: the ensemble (default) or one member.")
    parser.add_argument("--rrwavenet-lowpass-hz", type=float, default=ov.RRWAVENET_LOWPASS_HZ)
    parser.add_argument("--capnobase-path", type=Path, default=None)
    parser.add_argument("--bidmc-path", type=Path, default=None)
    parser.add_argument("--steam2-ppg-dir", type=Path, default=None)
    parser.add_argument("--steam2-rsp-dir", type=Path, default=None)
    parser.add_argument("--output-root", type=Path, default=PROJECT_ROOT)
    parser.add_argument("--no-png", action="store_true", help="Only the PDF of every observation and dataset.")
    parser.set_defaults(rrwavenet_unified_root=None)  # read by msa.rrwavenet_sources: 261007 / 261013 only
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> None:
    args = parse_args(argv)
    for key, path in (("capnobase", args.capnobase_path), ("bidmc", args.bidmc_path)):
        if path:
            ov.DATASET_CONFIGS[key]["data_path"] = str(path)
    if args.steam2_ppg_dir:
        ov.DATASET_CONFIGS["steam2"]["ppg_dir"] = str(args.steam2_ppg_dir)
    if args.steam2_rsp_dir:
        ov.DATASET_CONFIGS["steam2"]["rsp_dir"] = str(args.steam2_rsp_dir)
    out = Path(args.output_root)
    out.mkdir(parents=True, exist_ok=True)
    store: dict = {}
    rows = []
    for dataset_key in [k for k in ov.DATASET_ORDER if k in args.datasets]:
        rows += dataset_windows(dataset_key, args, store)
    win = pd.DataFrame(rows)
    if win.empty:
        print("No subject with RRWaveNet and another model was found (check the result folders).")
        return
    for size in args.sizes:
        win[win.Size_Sec == size].to_csv(out / f"windows_{size}s.csv", index=False, encoding="utf-8-sig")
    sheets, chosen = {}, []
    for key in OBSERVATIONS:
        table = subject_table(win, key, args.sizes, args.top)
        if table.empty:
            continue
        sheets[f"Subjects_{key}"] = table
        listed = table[table.Listed].drop_duplicates(["Dataset", "Subject"])
        for size in args.sizes:
            chosen += best_windows(win, listed, key, size)
    windows = pd.DataFrame(chosen)
    print()
    for key, (title, _) in OBSERVATIONS.items():
        part = windows[windows.Observation == key] if not windows.empty else windows
        print(f"{key}: {title}")
        if part.empty:
            print("   no subject shows it")
            continue
        for ds, d in part.groupby("Dataset", sort=False):
            for (rank, s), e in d.sort_values(["Rank", "Size_Sec"]).groupby(["Rank", "Subject"], sort=True):
                spans = ", ".join(f"{r.Size_Sec:.0f} s: {r.Start_Sec:.0f}-{r.End_Sec:.0f} s" for r in e.itertuples())
                print(f"   {ds} #{rank} {s}: {spans}")
    for key in OBSERVATIONS:
        part = windows[windows.Observation == key] if not windows.empty else windows
        for ds, d in (part.groupby("Dataset", sort=False) if not part.empty else []):
            with PdfPages(out / f"{key}_{ds}.pdf") as pdf:
                for r in d.sort_values(["Rank", "Size_Sec"]).to_dict("records"):
                    data = store[(r["Dataset"], r["Subject"], r["Recording"])]
                    path = out / key / ds / f"{r['Rank']:02d}_{ov.safe_name(r['Subject'])}_{int(r['Size_Sec'])}s.png"
                    fig = draw(key, r, data, path)
                    if args.no_png:
                        path.unlink(missing_ok=True)
                    pdf.savefig(fig)
                    plt.close(fig)
    notes = pd.DataFrame({"Item": ["RRWaveNet right", "ce_double", "rd_double", "rd_missed", "rr_coverage",
                                   "Subjects", "Windows", "Cycles", "RRWaveNet output"],
                          "Meaning": [
                              f"AE <= {RATE_OK_BPM} breath/min",
                              f"corr-encoder rate / GT rate {DOUBLE[0]}-{DOUBLE[1]}, RRWaveNet right; score = AE gap",
                              f"RespDiff rate / GT rate {DOUBLE[0]}-{DOUBLE[1]}, RRWaveNet right; score = AE gap",
                              f"RRWaveNet right and missing <= 1 cycle; RespDiff missing >= {MISSED_MIN} cycles more "
                              f"or at {HALF[0]}-{HALF[1]} x the rate; score = cycles missed by RespDiff - by "
                              "RRWaveNet + 0.1 x AE gap (<= 5)",
                              f">= 3 GT cycles; RRWaveNet coverage >= best other + {COVERAGE_MARGIN}; RRWaveNet "
                              "extra <= 1; score = coverage margin",
                              "non-overlapping windows from 0 s, per size: share of windows showing it and the mean "
                              "score of those; ranked once per dataset by the mean over the sizes (Combined_Share, "
                              f"then Combined_Score); the top {args.top} with at least one such window are listed",
                              f"per listed subject and size, the window (start every {args.step:g} s) with the "
                              "highest score",
                              "261030 detector on the aligned waveforms; a GT cycle is matched by a model cycle "
                              "starting within half its length (1:1); clipped GT cycles are counted (GT_Clipped_Cycles)",
                              rrwavenet_note(args)]})
    with pd.ExcelWriter(out / "discussion_subjects.xlsx") as writer:
        for name, frame in sheets.items():
            frame.round(4).to_excel(writer, sheet_name=name[:31], index=False)
        if not windows.empty:
            windows.round(4).to_excel(writer, sheet_name="Windows", index=False)
        notes.to_excel(writer, sheet_name="Notes", index=False)
    print(f"\nsaved under {out}")


if __name__ == "__main__":
    main()
