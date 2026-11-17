# -*- coding: utf-8 -*-
r"""261117_RRWaveNet_Scored_Waveforms: RRWaveNet waveform reconstruction and the RIIV / RIAV / RIFV spectra of
the PPG, drawn only for the minutes scored with the rescored ground truth (minutes that have a model input,
a model output and a valid ground truth), in 60 s and 30 s windows and in two figure styles.

Model: RRWaveNet as in the final results, the ensemble (mean of the members' stored test outputs):
  CapnoBase      261007_RRWaveNet_Ensemble_NoEvent.py, variant "full"
  BIDMC, STEAM2  261013_RRWaveNet_Adult_SubjectVal_WD.py, the weight decay selected per fold (WD_Selection)
Nothing is trained; the stored outputs of every held-out subject are read.

Scored minutes (rescored ground truth, scored with the minute scoring of the source script through the
helpers of 261113_RRWaveNet_Rescore_ReferenceGT, so the minutes are the ones of the rescored report):
  CapnoBase  "peak": breaths detected by count-orig on the capnogram, RRest agreement rule, expert CO2
             artifact spans excluded
  BIDMC      "ann1": annotator 1 breaths, expert-label rule (--bidmc-gt ann1_nogap: minutes overlapping an
             ann1 gap left out as well)
  STEAM2     "original": the stored ground truth of 261013 (belt breaths, RRest agreement rule)
Only minutes with Ref_Valid = True are drawn and listed; their minute values (GT breaths, RRWaveNet rate,
AE, waveform PCC) are the values of the report. The subjects with an MAE issue are left out as in the final
results (CapnoBase 0115, BIDMC 27, STEAM2 12_YJJ, 22_LMS, 26_CKK; --keep-issue-subjects keeps them).

Windows: 60 s = each scored minute; 30 s = the two halves of each scored minute (--window-sec 60 30).

Figures (each kind and style in its own folder)
  waveform\overlap    GT (black) and RRWaveNet (red) on one axis, each min-max scaled to [-1, 1] in the window
  waveform\separate   (a) GT and (b) RRWaveNet in two rows with a shared time axis, each in its own units
      GT = the continuous respiratory reference of the source script (CapnoBase capnogram, BIDMC impedance,
      STEAM2 belt; band-passed and aligned as in the scoring); RRWaveNet = the ensemble output band-passed
      with the recording's respiratory filter over each run of windows (as scored)
      Title: PCC as scored (stitched target vs. ensemble output in the window; for a 60 s window = the
      minute's reported PCC). Subtitle: the minute's rescored GT, RRWaveNet rate and AE (the 60 s scoring
      unit, also for its 30 s halves) and r of the drawn curves.
  respmod\overlap     (a) GT, RIIV, RIAV, RIFV over time (band-passed, z-scored) on one axis; (b) their |FFT|
  respmod\separate    one row per signal (GT, RIIV, RIAV, RIFV): time series left, |FFT| right
      Drawn with the functions of 261116_PPG_RIIV_RIAV_RIFV_FFT: beats of the PPG model input (0.5-5 Hz pulse
      band), RIIV (peak value), RIAV (peak - preceding trough), RIFV (60 / interval), 4 Hz cubic spline,
      band-passed with the recording's respiratory band over the whole recording, then cut to the window;
      spectrum: detrend, Hamming, zero padding; peak searched in the recording's rate band of the scoring
      profile. A window needs >= 90% series cover and >= 10 used beats (otherwise only the waveform is drawn).

Outputs (D:\PPG2RSP_RRWaveNet_Inspired\261117_RRWaveNet_Scored_Waveforms)
  {kind}\{style}\{Dataset}\{subject}\w{W}s\{subject}[_rec{k}]_{t0}-{t1}s.png       one figure per window
  {kind}\{style}\{Dataset}\{subject}\w{W}s\{Dataset}_{subject}_w{W}s_{kind}_{style}.pdf
  261117_RRWaveNet_Scored_Waveforms.xlsx
    Windows   every drawn window: minute, window, rescored GT, RRWaveNet rate, AE, PCC as scored, r of the drawn
              curves, RIIV / RIAV / RIFV / GT peak frequencies, figure paths
    Minutes   every scored minute (the report's minutes)
    Subjects  per subject: scored minutes, MAE, SD, mean PCC, mean |PCC|
    Datasets  per dataset: subjects, minutes, MAE (minute mean, subject mean), PCC
    Missing   subjects without stored results, selection or matching windows
    Notes

Run: python 261117_RRWaveNet_Scored_Waveforms.py     (Spyder: runfile(...), no options needed)
Options: --window-sec (default 60 30), --kinds (waveform respmod), --styles (overlap separate), --datasets,
--subjects CapnoBase:0332 BIDMC:05 ..., --keep-issue-subjects, --bidmc-gt (ann1 | ann1_nogap), --members,
--output-root, --root-261007, --root-261013, --script-261007, --script-261013, --script-261113,
--script-261116, --capnobase-path, --bidmc-path, --steam2-ppg-dir, --steam2-rsp-dir
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

NAME = "261117_RRWaveNet_Scored_Waveforms"
HERE = Path(__file__).resolve().parent
WORKSPACE = Path(r"D:\PPG2RSP_RRWaveNet_Inspired")
OUTPUT_ROOT = WORKSPACE / NAME
ROOT_261007 = WORKSPACE / "261007_RRWaveNet_NoEvent_v1"
ROOTS_261013 = (WORKSPACE / "261013_RRWavNet_STEAM_BIDMC_v1", WORKSPACE / "261013_RRWaveNet_Adult_SubjectVal_WD")
SCRIPTS = {"261007": "261007_RRWaveNet_Ensemble_NoEvent.py", "261013": "261013_RRWaveNet_Adult_SubjectVal_WD.py",
           "261113": "261113_RRWaveNet_Rescore_ReferenceGT.py", "261116": "261116_PPG_RIIV_RIAV_RIFV_FFT.py"}
WINDOW = "01_w10s_nonoverlap"
DATASETS = ("CapnoBase", "BIDMC", "STEAM2")
KEYS = {"CapnoBase": "capnobase", "BIDMC": "bidmc", "STEAM2": "steam2"}
SOURCE = {"CapnoBase": "261007", "BIDMC": "261013", "STEAM2": "261013"}
GT = {"CapnoBase": "peak", "BIDMC": "ann1", "STEAM2": "original"}
GT_TEXT = {"peak": "capnogram peak detection, RRest rule", "ann1": "BIDMC ann1, expert-label rule",
           "ann1_nogap": "BIDMC ann1 without gap minutes", "original": "stored GT"}
MODALITY = {"CapnoBase": "capnogram", "BIDMC": "impedance", "STEAM2": "belt"}
EXCLUDED = {"STEAM2": ("22_LMS", "26_CKK", "12_YJJ"), "BIDMC": ("27",), "CapnoBase": ("0115",)}
WINDOW_SECS = (60.0, 30.0)
KINDS = ("waveform", "respmod")
STYLES = ("overlap", "separate")
MEMBERS = 3
EVAL_SEC = 60.0
GT_COLOR, EST_COLOR = "#000000", "#f64636"  # as 261108 / 261116: GT black, RRWaveNet red
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
        name = "scored_wave_src_" + re.sub(r"\W", "_", path.stem)
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


# ---------------------------------------------------------------------------
# Signals
# ---------------------------------------------------------------------------

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
    a, b = np.asarray(a, float), np.asarray(b, float)
    ok = np.isfinite(a) & np.isfinite(b)
    if ok.sum() < 3 or np.std(a[ok]) < 1e-12 or np.std(b[ok]) < 1e-12:
        return math.nan
    return float(np.corrcoef(a[ok], b[ok])[0, 1])


def window_spans(start: float, end: float, lengths: Sequence[float]) -> List[Tuple[float, float, float]]:
    """(length, t0, t1) of every window inside one scored minute: the minute itself for 60 s, its parts otherwise."""
    spans = []
    for w in lengths:
        if w >= end - start - 1e-6:
            spans.append((w, start, end))
            continue
        k = 0
        while start + (k + 1) * w <= end + 1e-6:
            spans.append((w, start + k * w, start + (k + 1) * w))
            k += 1
    return spans


# ---------------------------------------------------------------------------
# Waveform figures
# ---------------------------------------------------------------------------

def style_axis(ax) -> None:
    ax.grid(True, color=GRID, lw=0.8)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.tick_params(labelsize=8.5)


def draw_wave_overlap(path: Path, pdf, title: str, subtitle: str, t: np.ndarray, gt: np.ndarray, est: np.ndarray,
                      labels: Tuple[str, str], t0: float, t1: float) -> None:
    fig = plt.figure(figsize=(10.0 if t1 - t0 <= 30 else 13.0, 4.3))
    ax = fig.add_axes([0.07, 0.13, 0.9, 0.64])
    ax.plot(t, minmax(gt), color=GT_COLOR, lw=2.0, zorder=3)
    ax.plot(t, minmax(est), color=EST_COLOR, lw=2.2, zorder=4)
    ax.set_xlim(t0, t1)
    ax.set_ylim(-1.15, 1.15)
    ax.set_xlabel("Time in the recording (s)", fontsize=9)
    ax.set_ylabel("Min-max scaled", fontsize=9)
    style_axis(ax)
    fig.legend([Line2D([], [], color=GT_COLOR, lw=2.0), Line2D([], [], color=EST_COLOR, lw=2.2)], list(labels),
               loc="upper right", bbox_to_anchor=(0.97, 0.985), ncol=2, frameon=False, fontsize=8.5)
    fig.text(0.07, 0.955, title, fontsize=10.5, color=INK, va="top")
    fig.text(0.07, 0.885, subtitle, fontsize=7.6, color=MUTED, va="top", linespacing=1.5)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=160)
    pdf.savefig(fig)
    plt.close(fig)


def draw_wave_separate(path: Path, pdf, title: str, subtitle: str, t: np.ndarray, gt: np.ndarray, est: np.ndarray,
                       labels: Tuple[str, str], t0: float, t1: float) -> None:
    fig = plt.figure(figsize=(10.0 if t1 - t0 <= 30 else 13.0, 6.0))
    ax_g = fig.add_axes([0.08, 0.52, 0.89, 0.29])
    ax_e = fig.add_axes([0.08, 0.1, 0.89, 0.31], sharex=ax_g)
    ax_g.plot(t, gt, color=GT_COLOR, lw=1.8)
    ax_e.plot(t, est, color=EST_COLOR, lw=1.9)
    for ax, letter, label in ((ax_g, "a", labels[0]), (ax_e, "b", labels[1])):
        style_axis(ax)
        ax.set_title(f"({letter}) {label}", loc="left", fontsize=9.5, color=INK, pad=3)
        ax.set_ylabel("a.u.", fontsize=9)
    ax_e.set_xlim(t0, t1)
    ax_e.set_xlabel("Time in the recording (s)", fontsize=9)
    ax_g.tick_params(labelbottom=False)
    fig.text(0.08, 0.965, title, fontsize=10.5, color=INK, va="top")
    fig.text(0.08, 0.92, subtitle, fontsize=7.6, color=MUTED, va="top", linespacing=1.5)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=160)
    pdf.savefig(fig)
    plt.close(fig)


WAVE_DRAW = {"overlap": draw_wave_overlap, "separate": draw_wave_separate}


# ---------------------------------------------------------------------------
# Run
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


class Pdfs:
    """One PDF per (kind, style, length) of a subject, opened on first use."""

    def __init__(self):
        self.files: Dict[Tuple[str, str, float], PdfPages] = {}

    def get(self, key: Tuple[str, str, float], path: Path) -> PdfPages:
        if key not in self.files:
            path.parent.mkdir(parents=True, exist_ok=True)
            self.files[key] = PdfPages(path)
        return self.files[key]

    def close(self) -> None:
        for pdf in self.files.values():
            pdf.close()
        self.files = {}


def modulation_series(rm, mod, ref: dict, fs: float):
    """RIIV / RIAV / RIFV of the whole recording, band-passed (functions of 261116)."""
    profile = ref["profile"]
    beats = rm.extract_modulations(mod, ref["model_input"], fs, tuple(profile["ppg_hr_range_bpm"]),
                                   ref.get("ppg_artifact_mask"))
    if "times" not in beats or len(beats["times"]) < rm.MIN_WINDOW_BEATS:
        return None
    band = tuple(float(v) for v in ref["band_hz"])
    full = {}
    for k in rm.MODS:
        grid, values = rm.resample(beats["times"], beats["values"][k])
        full[k] = (grid, rm.respiratory_band(values, rm.RESAMPLE_HZ, band))
    return {"beats": beats, "full": full, "band": band}


def dataset_rows(args, dataset: str, chosen: Dict[str, List[str]], out_root: Path, rescore, rm) -> Tuple[list, list, list]:
    source = SOURCE[dataset]
    root = Path(args.root_261007) if source == "261007" else root_261013(args)
    script = find_script(SCRIPTS[source], args.script_261007 if source == "261007" else args.script_261013, [root])
    mod = load_script(script)
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
    lengths = sorted({float(w) for w in args.window_sec}, reverse=True)
    excluded = [] if args.keep_issue_subjects else list(EXCLUDED[dataset])
    rows, minutes, missing = [], [], []
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
            gap = rescore.flag_gap_minutes(frame, spans, mod.TARGET_FS)
            frame.loc[gap & frame["Ref_Valid"].astype(bool), "Ref_Exclusion_Reason"] = "ref_ann1_gap"
            frame.loc[gap, "Ref_Valid"] = False
        scored = frame[frame["Ref_Valid"].astype(bool)].copy()
        scored.insert(0, "Subject", str(subject))
        scored.insert(0, "Dataset", dataset)
        scored.insert(2, "GT", gt_name)
        scored.insert(3, "Variant", variant.name)
        minutes.append(scored)
        tag = mod.safe_tag(subject)
        multi = len(references) > 1
        pdfs = Pdfs()
        drawn = 0
        try:
            for segment_id, seg_minutes in scored.groupby("Segment_ID"):
                ref = refs[int(segment_id)]
                pred, true, _ = mod.reconstruct_segment(ens["predictions"], ens["targets"], metadata, int(segment_id),
                                                         mod.TARGET_FS)
                est = as_scored(mod, pred, ref)
                gt_sig = np.asarray(ref["signal"], dtype=np.float64)
                breaths = np.asarray(ref["breaths"], dtype=np.float64) / fs
                mods = modulation_series(rm, mod, ref, fs) if "respmod" in args.kinds else None
                rate_band = tuple(np.asarray(ref["profile"]["rr_band_bpm"], float) / 60.0)
                max_freq = max(1.0, rate_band[1])
                rec_tag = f"_rec{segment_id}" if multi else ""
                rec_text = f" · recording {segment_id}" if multi else ""
                for _, m in seg_minutes.iterrows():
                    ma, mb = float(m["RR_Window_Start_Sec"]), float(m["RR_Window_End_Sec"])
                    gt_count, rate = float(m["GT_Breath_Count"]), float(m["Pred_Wave_FFT_RR"])
                    ae, pcc_min = float(m["AE_Wave_FFT"]), float(m["Wave_PCC"])
                    for length, t0, t1 in window_spans(ma, mb, lengths):
                        a, b = int(round(t0 * fs)), int(round(t1 * fs))
                        t = np.arange(a, b) / fs
                        n = min(b, len(gt_sig), len(est))
                        g_w, e_w = gt_sig[a:n], est[a:n]
                        pcc = pearson(true[a:n], pred[a:n])  # as scored: stitched target vs. ensemble output
                        r_drawn = pearson(g_w, e_w)
                        stem = f"{tag}{rec_tag}_{t0:.0f}-{t1:.0f}s"
                        row = {"Dataset": dataset, "Subject": subject, "Segment_ID": int(segment_id), "GT": gt_name,
                               "Variant": variant.name, "Window_Sec": length, "Start_Sec": t0, "End_Sec": t1,
                               "Minute_Start_Sec": ma, "Minute_End_Sec": mb, "GT_Breaths_Minute": gt_count,
                               "RRWaveNet_Rate_Minute": rate, "AE_Minute": ae, "PCC_Minute_Reported": pcc_min,
                               "PCC_As_Scored_Window": pcc, "r_Drawn_Curves": r_drawn}
                        title = f"{dataset} {subject}{rec_text} · {t0:.0f}-{t1:.0f} s · PCC (as scored) = {pcc:.2f}"
                        subtitle = (f"scored minute {ma:.0f}-{mb:.0f} s: GT {gt_count:.0f} /min ({GT_TEXT[gt_name]}), "
                                    f"RRWaveNet {rate:.1f} /min, AE {ae:.2f}\n"
                                    f"r of the drawn curves {r_drawn:.2f} · RRWaveNet ensemble ({source} {variant.name}), "
                                    f"band-passed as scored")
                        labels = (f"GT ({MODALITY[dataset]})", "RRWaveNet")
                        if "waveform" in args.kinds:
                            for st in args.styles:
                                folder = out_root / "waveform" / st / dataset / tag / f"w{length:g}s"
                                pdf = pdfs.get(("waveform", st, length),
                                               folder / f"{dataset}_{tag}_w{length:g}s_waveform_{st}.pdf")
                                WAVE_DRAW[st](folder / f"{stem}.png", pdf, title, subtitle, t[: n - a], g_w, e_w,
                                              labels, t0, t1)
                                row[f"Figure_Waveform_{st.capitalize()}"] = str((folder / f"{stem}.png").relative_to(out_root))
                        if mods is not None:
                            times = mods["beats"]["times"]
                            in_window = (times >= t0) & (times < t1)
                            series = {}
                            for k in rm.MODS:
                                grid, values = mods["full"][k]
                                keep = (grid >= t0) & (grid < t1)
                                series[k] = (grid[keep], values[keep])
                            cover = min(len(series[k][0]) for k in rm.MODS) / (rm.RESAMPLE_HZ * (t1 - t0))
                            if int(in_window.sum()) >= rm.MIN_WINDOW_BEATS and cover >= rm.MIN_WINDOW_COVER:
                                spectra = {}
                                for k in rm.MODS:
                                    freq, mag = rm.spectrum(series[k][1], rm.RESAMPLE_HZ)
                                    peak_hz, peak_mag = rm.band_peak(freq, mag, rate_band)
                                    spectra[k] = {"freq": freq, "mag": mag, "peak_hz": peak_hz, "peak_mag": peak_mag}
                                    row[f"{k}_Peak_Hz"] = peak_hz
                                ok = np.isfinite(g_w)
                                gt_freq, gt_mag = rm.spectrum(g_w[ok], fs)
                                gt_peak_hz, gt_peak_mag = rm.band_peak(gt_freq, gt_mag, rate_band)
                                gt_info = {"label": f"GT ({MODALITY[dataset]})", "series": (t[: n - a][ok], g_w[ok]),
                                           "freq": gt_freq, "mag": gt_mag, "peak_hz": gt_peak_hz, "peak_mag": gt_peak_mag}
                                gt_rate_hz = float(((breaths >= t0) & (breaths < t1)).sum()) / (t1 - t0)
                                row.update({"GT_Ref_Peak_Hz": gt_peak_hz, "GT_Rate_Window_Hz": gt_rate_hz,
                                            "Beats_Used": int(in_window.sum())})
                                pulse = float(np.mean(mods["beats"]["values"]["RIFV"][in_window]))
                                header = (f"{dataset} {subject}{rec_text}: respiratory modulations of the PPG · "
                                          f"{length:g} s window {t0:.0f}-{t1:.0f} s · {int(in_window.sum())} beats, "
                                          f"mean pulse rate {pulse:.0f} /min",
                                          f"scored minute {ma:.0f}-{mb:.0f} s: GT {gt_count:.0f} /min ({GT_TEXT[gt_name]}), "
                                          f"RRWaveNet {rate:.1f} /min; PPG = model input; series at {rm.RESAMPLE_HZ:g} Hz, "
                                          f"band-passed {mods['band'][0]:.2f}-{mods['band'][1]:.2f} Hz; peaks in "
                                          f"{rate_band[0]:.2f}-{rate_band[1]:.2f} Hz")
                                for st in args.styles:
                                    folder = out_root / "respmod" / st / dataset / tag / f"w{length:g}s"
                                    pdf = pdfs.get(("respmod", st, length),
                                                   folder / f"{dataset}_{tag}_w{length:g}s_respmod_{st}.pdf")
                                    rm.DRAW[st](folder / f"{stem}_RIIV_RIAV_RIFV.png", pdf, header, series, spectra,
                                                gt_info, gt_rate_hz, rate_band, max_freq, f"{length:g} s window")
                                    row[f"Figure_Respmod_{st.capitalize()}"] = str(
                                        (folder / f"{stem}_RIIV_RIAV_RIFV.png").relative_to(out_root))
                            else:
                                row["Respmod_Skipped"] = (f"{int(in_window.sum())} usable beats, "
                                                          f"series cover {cover:.0%}")
                        elif "respmod" in args.kinds:
                            row["Respmod_Skipped"] = "too few usable beats in the recording"
                        rows.append(row)
                        drawn += 1
        finally:
            pdfs.close()
        print(f"  {subject}: {len(scored)} scored minutes, {drawn} windows drawn", flush=True)
    del data
    return rows, minutes, missing


def summaries(minutes: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    m = minutes.copy()
    m["AE"] = m["AE_Wave_FFT"].astype(float)
    m["Abs_PCC"] = m["Wave_PCC"].abs()
    subj = m.groupby(["Dataset", "Subject"], sort=False).agg(
        GT=("GT", "first"), Scored_Minutes=("AE", "size"), MAE=("AE", "mean"), AE_SD=("AE", "std"),
        Mean_PCC=("Wave_PCC", "mean"), Mean_Abs_PCC=("Abs_PCC", "mean")).reset_index()
    ds = []
    for dataset, g in m.groupby("Dataset", sort=False):
        s = subj[subj.Dataset == dataset]
        ds.append({"Dataset": dataset, "GT": g["GT"].iloc[0], "Subjects": int(len(s)), "Scored_Minutes": int(len(g)),
                   "MAE_Minutes": float(g.AE.mean()), "SD_Minutes": float(g.AE.std()),
                   "MAE_Subject_Mean": float(s.MAE.mean()), "MAE_Subject_SD": float(s.MAE.std()),
                   "PCC_Minutes": float(g.Wave_PCC.mean()), "Abs_PCC_Minutes": float(g.Abs_PCC.mean())})
    return subj, pd.DataFrame(ds)


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="RRWaveNet waveforms and RIIV / RIAV / RIFV spectra of the scored minutes")
    p.add_argument("--window-sec", nargs="+", type=float, default=list(WINDOW_SECS),
                   help="window lengths in s (default 60 30): 60 = the scored minute, shorter = its parts")
    p.add_argument("--kinds", nargs="+", choices=list(KINDS), default=list(KINDS))
    p.add_argument("--styles", nargs="+", choices=list(STYLES), default=list(STYLES))
    p.add_argument("--datasets", nargs="+", choices=list(DATASETS), default=list(DATASETS))
    p.add_argument("--subjects", nargs="+", default=None, help="Dataset:Subject (default: every subject)")
    p.add_argument("--keep-issue-subjects", action="store_true")
    p.add_argument("--bidmc-gt", choices=("ann1", "ann1_nogap"), default="ann1")
    p.add_argument("--members", type=int, default=MEMBERS)
    p.add_argument("--output-root", type=Path, default=OUTPUT_ROOT)
    p.add_argument("--root-261007", type=Path, default=ROOT_261007)
    p.add_argument("--root-261013", type=Path, default=None)
    p.add_argument("--script-261007", type=Path, default=None)
    p.add_argument("--script-261013", type=Path, default=None)
    p.add_argument("--script-261113", type=Path, default=None)
    p.add_argument("--script-261116", type=Path, default=None)
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
    rm = load_script(find_script(SCRIPTS["261116"], args.script_261116, []))
    rows, minutes, missing = [], [], []
    for dataset in datasets:
        r, m, x = dataset_rows(args, dataset, chosen, out_root, rescore, rm)
        rows += r
        minutes += m
        missing += x
    table = pd.DataFrame(rows)
    minute_table = pd.concat(minutes, ignore_index=True) if minutes else pd.DataFrame()
    subj, ds = summaries(minute_table) if not minute_table.empty else (pd.DataFrame(), pd.DataFrame())
    notes = pd.DataFrame([
        ("Model", "RRWaveNet ensemble (mean of the members' stored test outputs): CapnoBase 261007 variant full; "
                  "BIDMC, STEAM2 261013 weight decay selected per fold"),
        ("Scored minutes", "rescored ground truth, scored with the source script's minute scoring (helpers of 261113): "
                           "CapnoBase peak (capnogram peak detection, RRest rule), BIDMC " + args.bidmc_gt
                           + ", STEAM2 stored GT; only Ref_Valid minutes; every minute has a model input and output"),
        ("Windows", "60 s = the scored minute; 30 s = its halves; minute values (GT, rate, AE, PCC_Minute_Reported) "
                    "are those of the scoring unit (60 s)"),
        ("PCC_As_Scored_Window", "Pearson r of the stitched target and the ensemble output in the window (the scoring "
                                 "definition; equals PCC_Minute_Reported for 60 s windows)"),
        ("r_Drawn_Curves", "Pearson r of the drawn curves: continuous reference vs. output band-passed as scored"),
        ("Respmod", "functions of 261116_PPG_RIIV_RIAV_RIFV_FFT; peaks in the recording's rate band of the scoring "
                    "profile; windows need >= 90% series cover and >= 10 used beats"),
        ("Subjects", "issue subjects left out (CapnoBase 0115, BIDMC 27, STEAM2 12_YJJ, 22_LMS, 26_CKK)"
                     if not args.keep_issue_subjects else "every subject (issue subjects kept)"),
    ], columns=["Item", "Description"])
    path = out_root / f"{NAME}.xlsx"
    with pd.ExcelWriter(path, engine="openpyxl") as writer:
        table.round(5).to_excel(writer, sheet_name="Windows", index=False)
        minute_table.round(5).to_excel(writer, sheet_name="Minutes", index=False)
        subj.round(4).to_excel(writer, sheet_name="Subjects", index=False)
        ds.round(4).to_excel(writer, sheet_name="Datasets", index=False)
        pd.DataFrame(missing, columns=["Dataset", "Subject", "Missing"]).to_excel(writer, sheet_name="Missing", index=False)
        notes.to_excel(writer, sheet_name="Notes", index=False)
    print(f"\n{len(table)} windows drawn from {len(minute_table)} scored minutes; figures under {out_root}; table: {path}")
    if not ds.empty:
        print(ds.round(3).to_string(index=False))


if __name__ == "__main__":
    main()
