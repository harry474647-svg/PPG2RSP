# -*- coding: utf-8 -*-
r"""261118_StemFusionO_Best_FFT_Subjects: the subjects of every dataset whose Fusion O (with the 1x1 stem fusion)
waveform is restored best AND whose |FFT| curve is closest to the GT curve over all their 30 s windows.

Model: the single model m0 of every LOSOCV fold with the stem fusion (Fusion O), the stored outputs read exactly as
261118_StemFusion_Threshold_Groups_Waveforms reads them (that script is loaded as a module and its functions are
used: stored files, rescored GT, predicted waveform band-passed as scored, |FFT| of a window). Nothing is trained.
  CapnoBase      261007 full, member m0
  BIDMC, STEAM2  261013, member m0 of the weight decay selected in the fold (WD_Selection)
  (261028_Stem_ablation_noise_v1 first when it holds the model, as in 261118)

Windows: every 30 s half of every scored 60 s minute (the windows of the 261118 figures inside the scored minutes;
windows outside a scored minute have no GT rate and are not used). Per window:
  Window_r       Pearson r of the GT signal (as scored) and the Fusion O waveform in the 30 s window
  Spectral_r     Pearson r of the two |FFT| curves as drawn in the FFT figures: detrend, Hamming window, zero padding
                 to >= 32768 points, each spectrum scaled to its peak = 1, 0 Hz to the end of the frequency axis
                 (1 Hz, or the top of the rate band if higher); 1 = the same curve
  Peak_Diff_BPM  |peak of Fusion O - peak of GT| (largest |FFT| in the rate band of the scoring), breaths/min
Per subject (scored minutes and their windows):
  MAE_O          mean |stored rate - GT breath count| over the scored minutes (as 261118)
  AbsPCC_O       mean |Wave_PCC| over the scored minutes (as 261118)
  Spectral_r     mean Spectral_r over the windows (Min_Spectral_r: the worst window)

Selection, per dataset, among the subjects whose restoration is good (the clinical criteria of 261118:
MAE_O < 3 breaths/min and AbsPCC_O > 0.6; --clinical-mae, --clinical-pcc) with >= --min-minutes scored minutes
(default 4) and not an MAE issue subject (CapnoBase 0115, BIDMC 27, STEAM2 12_YJJ, 22_LMS, 26_CKK):
  rank of MAE_O (low = 1), rank of AbsPCC_O (high = 1), rank of Spectral_r (high = 1);
  Score = mean of the three ranks; the --top (default 3) lowest scores are selected (ties: higher Spectral_r).

Outputs (D:\PPG2RSP_RRWaveNet_Inspired\261118_StemFusionO_Best_FFT_Subjects)
  {Dataset}\{rank}_{Dataset}_{subject}\{t0}-{t1}s.png        GT vs Fusion O waveform of every scored 30 s window
                                       {Dataset}_{subject}_windows.pdf
  {Dataset}\{rank}_{Dataset}_{subject}\FFT\{t0}-{t1}s_FFT.png |FFT| GT vs Fusion O of the same window
                                       FFT\{Dataset}_{subject}_FFT.pdf
  261118_StemFusionO_Best_FFT_Subjects.xlsx
    Top            the selected subjects of every dataset with their measures and ranks
    Ranking        every subject: measures, ranks, score, selected or why not eligible
    Windows        every scored 30 s window of every subject: GT, rate, error, Window_r, Spectral_r, peaks
    Missing        subjects without stored Fusion O results or without a scored minute
    Notes

Run: python 261118_StemFusionO_Best_FFT_Subjects.py     (Spyder: runfile(...), no options needed;
     261118_StemFusion_Threshold_Groups_Waveforms.py must be in the same folder, or give --script-261118)
Options: --top (3), --min-minutes (4), --clinical-mae (3), --clinical-pcc (0.6), --datasets, --keep-issue-subjects, --ann1-nogap, --window-sec (30),
--raw-predictions, --root-261007, --root-261013, --root-noise, --output-root, --script-261118, --script-261007,
--script-261013, --capnobase-path, --bidmc-path, --steam2-ppg-dir, --steam2-rsp-dir
"""

import argparse
import importlib.util
import math
import sys
from pathlib import Path
from typing import Optional, Sequence, Tuple

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

NAME = "261118_StemFusionO_Best_FFT_Subjects"
HERE = Path(__file__).resolve().parent
BASE_SCRIPT = "261118_StemFusion_Threshold_Groups_Waveforms.py"
WORKSPACE = Path(r"D:\PPG2RSP_RRWaveNet_Inspired")
OUTPUT_ROOT = WORKSPACE / NAME
TOP_N = 3
MIN_SCORED_MINUTES = 4
CLINICAL_MAE = 3.0  # good restoration: MAE_O < 3 breaths/min and mean |PCC| > 0.6 (clinical criteria of 261118)
CLINICAL_PCC = 0.6
MODEL = "O"  # Fusion O: with the 1x1 stem fusion


def load_base(given: Optional[Path]):
    """261118_StemFusion_Threshold_Groups_Waveforms as a module (its loading, GT and FFT functions)."""
    for path in ([Path(given)] if given else []) + [HERE / BASE_SCRIPT, Path.cwd() / BASE_SCRIPT]:
        if path.is_file():
            spec = importlib.util.spec_from_file_location("best_fft_base_261118", str(path))
            module = importlib.util.module_from_spec(spec)
            sys.modules[spec.name] = module
            spec.loader.exec_module(module)
            return module
    raise FileNotFoundError(f"{BASE_SCRIPT} not found next to this script ({HERE}); give it with --script-261118")


# ---------------------------------------------------------------------------
# Minutes and windows of Fusion O
# ---------------------------------------------------------------------------

def o_minutes(base, args, mod, dataset: str, subject: str, record: dict, files: dict,
              ann1: Optional[np.ndarray]) -> pd.DataFrame:
    """Every stored minute of Fusion O with the rescored GT (the GT of 261118)."""
    fs = mod.TARGET_FS
    table = pd.read_csv(files["minutes"], dtype={"Subject": str}, encoding="utf-8-sig")
    rows = []
    for _, o in table.iterrows():
        segment_id, t0, t1 = int(o["Segment_ID"]), float(o["RR_Window_Start_Sec"]), float(o["RR_Window_End_Sec"])
        ref = record["references"].get(segment_id)
        if ref is None:
            continue
        a, b = int(round(t0 * fs)), int(round(t1 * fs))
        if dataset == "STEAM2":  # no new GT: the stored one
            valid, reason, count = bool(o["Ref_Valid"]), str(o.get("Ref_Exclusion_Reason", "") or ""), int(o["GT_Breath_Count"])
        else:
            breaths, label_based = base.gt_breaths(dataset, mod, ref, ann1)
            valid, reason, count = base.minute_gt(mod, ref, a, b, breaths, label_based)
            if dataset == "BIDMC" and args.ann1_nogap and valid and any(
                    lo < b and hi > a for lo, hi in base.gap_spans(breaths, len(ref["signal"]))):
                valid, reason = False, "ref_ann1_gap"
        rate = float(o["Pred_Wave_FFT_RR"])
        rows.append({"Dataset": dataset, "Subject": subject, "Segment_ID": segment_id, "Start_Sec": t0, "End_Sec": t1,
                     "Scored": valid, "Reason": "" if valid else reason, "GT_Count": count, "Rate_O": rate,
                     "AE_O": abs(rate - count) if valid else math.nan, "PCC_O": float(o.get("Wave_PCC", math.nan))})
    return pd.DataFrame(rows)


def curve_r(base, gt: dict, model: dict, max_freq: float) -> float:
    """Pearson r of the two |FFT| curves as drawn (each scaled to its peak = 1, 0 Hz to max_freq)."""
    if not all(np.isfinite(s["peak_mag"]) and s["peak_mag"] > 0 for s in (gt, model)):
        return math.nan
    keep = gt["freq"] <= max_freq
    return base.pearson(gt["mag"][keep] / gt["peak_mag"], model["mag"][keep] / model["peak_mag"])


def subject_windows(base, args, mod, dataset: str, subject: str, record: dict, files: dict,
                    minutes: pd.DataFrame) -> Tuple[pd.DataFrame, dict]:
    """Every 30 s half of every scored minute: measures, and the signals for the figures."""
    fs = mod.TARGET_FS
    rows, signals = [], {}
    scored = minutes[minutes.Scored]
    for segment_id, ref in sorted(record["references"].items()):
        mins = scored[scored.Segment_ID == int(segment_id)].sort_values("Start_Sec")
        if mins.empty:
            continue
        length = len(ref["signal"])
        pred = base.series(files["outputs"], int(segment_id), length)
        if not args.raw_predictions:
            pred = base.as_scored(mod, pred, ref)
        rate_band = tuple(float(v) / 60.0 for v in ref["profile"]["rr_band_bpm"])
        max_freq = max(base.MAX_FREQ_HZ, rate_band[1])
        for m in mins.itertuples(index=False):
            for k in range(int(round((m.End_Sec - m.Start_Sec) / args.window_sec))):
                t0 = m.Start_Sec + k * args.window_sec
                t1 = min(t0 + args.window_sec, m.End_Sec)
                a, b = int(round(t0 * fs)), int(round(t1 * fs))
                gt_w, pred_w = np.asarray(ref["signal"][a:b], float), pred[a:b]
                spectra = {}
                if (b - a) / fs >= base.FFT_MIN_SEC - 1e-6:
                    for key, values in (("GT", gt_w), (MODEL, pred_w)):
                        freq, mag = base.spectrum(values, fs)
                        peak_hz, peak_mag = base.band_peak(freq, mag, rate_band)
                        spectra[key] = {"freq": freq, "mag": mag, "peak_hz": peak_hz, "peak_mag": peak_mag}
                peak_gt = spectra["GT"]["peak_hz"] * 60.0 if spectra else math.nan
                peak_o = spectra[MODEL]["peak_hz"] * 60.0 if spectra else math.nan
                key = (int(segment_id), round(t0, 3))
                rows.append({"Dataset": dataset, "Subject": subject, "Segment_ID": int(segment_id),
                             "Window_Start_Sec": t0, "Window_End_Sec": t1, "Minute_Start_Sec": m.Start_Sec,
                             "Minute_End_Sec": m.End_Sec, "GT_Count": m.GT_Count, "Rate_O": m.Rate_O, "AE_O": m.AE_O,
                             "Minute_PCC_O": m.PCC_O, "Window_r": base.pearson(gt_w, pred_w),
                             "Spectral_r": curve_r(base, spectra["GT"], spectra[MODEL], max_freq) if spectra else math.nan,
                             "Peak_GT_BPM": peak_gt, "Peak_O_BPM": peak_o, "Peak_Diff_BPM": abs(peak_o - peak_gt)})
                signals[key] = {"t": np.arange(a, b) / fs, "gt": gt_w, "pred": pred_w, "spectra": spectra,
                                "rate_band": rate_band, "max_freq": max_freq}
    return pd.DataFrame(rows), signals


# ---------------------------------------------------------------------------
# Figures (GT vs Fusion O only)
# ---------------------------------------------------------------------------

def one_panel(base, path: Path, pdf: PdfPages, header: Tuple[str, str], title: str, side_text: str, xlabel: str,
              draw) -> None:
    width, side, left, panel, top_in, bottom_in = 12.0, 3.4, 0.7, 2.2, 0.95, 0.6
    height = top_in + bottom_in + panel
    fig = plt.figure(figsize=(width, height))
    ax = fig.add_axes([left / width, bottom_in / height, (width - left - side - 0.2) / width, panel / height])
    draw(ax)
    ax.grid(True, color=base.GRID, lw=0.6)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.set_title(title, fontsize=9.5, loc="left", pad=3)
    ax.set_xlabel(xlabel)
    x, y = (width - side + 0.05) / width, (bottom_in + panel) / height
    fig.legend([Line2D([], [], color=base.COLORS["GT"], lw=1.4), Line2D([], [], color=base.COLORS[MODEL], lw=base.MODEL_LW)],
               ["GT", base.LABELS[MODEL]], loc="upper left", bbox_to_anchor=(x, y), frameon=False, fontsize=7.8,
               borderaxespad=0.0)
    fig.text(x + 0.004, y - 0.75 / height, side_text, family="monospace", fontsize=7.6, color=base.INK, va="top")
    fig.text(left / width, 1.0 - 0.18 / height, header[0], fontsize=10.5, color=base.INK, va="top")
    fig.text(left / width, 1.0 - 0.47 / height, header[1], fontsize=7.6, color=base.MUTED, va="top")
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=160)
    pdf.savefig(fig)
    plt.close(fig)


def draw_subject(base, args, mod, dataset: str, subject: str, row: dict, windows: pd.DataFrame, signals: dict,
                 folder: Path, multi: bool) -> int:
    tag = mod.safe_tag(subject)
    head = (f"{int(row['Rank'])}. {dataset} {subject}, {base.LABELS[MODEL]}: MAE {row['MAE_O']:.2f} breaths/min, "
            f"|PCC| {row['AbsPCC_O']:.2f}, |FFT| curve r {row['Spectral_r']:.2f} (mean of {int(row['Windows'])} windows)")
    kind = "raw output" if args.raw_predictions else "band-passed as scored"
    count = 0
    (folder / "FFT").mkdir(parents=True, exist_ok=True)
    with PdfPages(folder / f"{dataset}_{tag}_windows.pdf") as pdf, \
            PdfPages(folder / "FFT" / f"{dataset}_{tag}_FFT.pdf") as fft_pdf:
        for w in windows.sort_values(["Segment_ID", "Window_Start_Sec"]).itertuples(index=False):
            s = signals[(int(w.Segment_ID), round(w.Window_Start_Sec, 3))]
            t0, t1 = w.Window_Start_Sec, w.Window_End_Sec
            rec = f" · recording {w.Segment_ID}" if multi else ""
            name = f"{f'rec{w.Segment_ID}_' if multi else ''}{t0:06.1f}-{t1:06.1f}s.png".replace(".0-", "-").replace(".0s", "s")
            wave_text = (f"minute {w.Minute_Start_Sec:.0f}-{w.Minute_End_Sec:.0f} s, scored\n"
                         f"GT rate      {w.GT_Count:5.1f} /min\n"
                         f"model rate   {w.Rate_O:5.1f} /min\n"
                         f"error (AE)   {w.AE_O:5.2f} /min\n"
                         f"r (window)   {w.Window_r:5.2f}")

            def wave(ax, s=s):
                ax.plot(s["t"], base.minmax(s["gt"]), color=base.COLORS["GT"], lw=1.4, zorder=2)
                ax.plot(s["t"], base.minmax(s["pred"]), color=base.COLORS[MODEL], lw=base.MODEL_LW, zorder=3)
                ax.set_xlim(s["t"][0], s["t"][-1])
                ax.set_ylim(-1.2, 1.25)
                ax.set_ylabel("Min-max scaled", fontsize=8)

            one_panel(base, folder / name, pdf, (f"{head}{rec} · {t0:.0f}-{t1:.0f} s",
                                                 f"GT: {base.MODALITY[dataset]} as scored, GT breaths = "
                                                 f"{base.NEW_GT[dataset]}; model: member m0, {kind}; each waveform "
                                                 f"min-max scaled in the window"),
                      f"GT vs {base.LABELS[MODEL]}", wave_text, "Time in the recording (s)", wave)
            count += 1
            if not s["spectra"]:
                continue
            band = s["rate_band"]
            peak = {k: (f"{v['peak_hz']:.3f} Hz {v['peak_hz'] * 60:5.1f} /min" if np.isfinite(v["peak_hz"]) else "n/a")
                    for k, v in s["spectra"].items()}
            fft_text = (f"peak in {band[0] * 60:g}-{band[1] * 60:g} /min\n"
                        f"GT         {peak['GT']}\n"
                        f"Fusion O   {peak[MODEL]}\n"
                        f"|peak diff| {w.Peak_Diff_BPM:5.2f} /min\n"
                        f"curve r    {w.Spectral_r:5.2f}")

            def spec(ax, s=s):
                top = 1.0
                for key, lw, z in (("GT", 1.4, 2), (MODEL, base.MODEL_LW, 3)):
                    sp = s["spectra"][key]
                    if not (np.isfinite(sp["peak_mag"]) and sp["peak_mag"] > 0):
                        continue
                    keep = sp["freq"] <= s["max_freq"]
                    yv = sp["mag"][keep] / sp["peak_mag"]
                    top = max(top, float(np.nanmax(yv)))
                    ax.plot(sp["freq"][keep], yv, color=base.COLORS[key], lw=lw, zorder=z + 2)
                    ax.axvline(sp["peak_hz"], color=base.COLORS[key], lw=1.0, alpha=0.9, zorder=z)
                    ax.plot([sp["peak_hz"]], [1.0], marker="s", ms=5.5, color=base.COLORS[key], mec="white", mew=0.8,
                            zorder=z + 4)
                ax.set_xlim(0.0, s["max_freq"])
                ax.set_ylim(0.0, top * 1.12)
                ax.set_ylabel("|FFT| (peak = 1)", fontsize=8)

            one_panel(base, folder / "FFT" / name.replace(".png", "_FFT.png"), fft_pdf,
                      (f"{head}{rec} · {t0:.0f}-{t1:.0f} s",
                       f"|FFT| of the {t1 - t0:g} s window (detrend, Hamming, zero padding), each spectrum scaled to its "
                       f"peak = 1 (largest |FFT| in the rate band of the scoring); model: member m0, {kind}"),
                      f"|FFT|: GT vs {base.LABELS[MODEL]}", fft_text, "Frequency (Hz)", spec)
    return count


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Fusion O: per dataset the subjects with the best restoration and the "
                                            "|FFT| curve closest to the GT over their 30 s windows")
    p.add_argument("--top", type=int, default=TOP_N, help="subjects selected per dataset (default 3)")
    p.add_argument("--min-minutes", type=int, default=MIN_SCORED_MINUTES,
                   help="scored minutes a subject needs to be selected (default 4)")
    p.add_argument("--clinical-mae", type=float, default=CLINICAL_MAE, help="selectable: MAE_O below (default 3.0)")
    p.add_argument("--clinical-pcc", type=float, default=CLINICAL_PCC, help="selectable: mean |PCC| above (default 0.6)")
    p.add_argument("--datasets", nargs="+", choices=["CapnoBase", "BIDMC", "STEAM2"],
                   default=["CapnoBase", "BIDMC", "STEAM2"])
    p.add_argument("--keep-issue-subjects", action="store_true", help="let the MAE issue subjects be selected")
    p.add_argument("--ann1-nogap", action="store_true", help="BIDMC: leave out minutes overlapping an ann1 gap")
    p.add_argument("--window-sec", type=float, default=30.0)
    p.add_argument("--raw-predictions", action="store_true",
                   help="use the raw model output (default: band-passed as the rate was scored)")
    p.add_argument("--root-261007", type=Path, default=WORKSPACE / "261007_RRWaveNet_NoEvent_v1")
    p.add_argument("--root-261013", type=Path, default=WORKSPACE / "261013_RRWavNet_STEAM_BIDMC_v1")
    p.add_argument("--root-noise", type=Path, default=WORKSPACE / "261028_Stem_ablation_noise_v1")
    p.add_argument("--output-root", type=Path, default=OUTPUT_ROOT)
    p.add_argument("--script-261118", type=Path, default=None)
    p.add_argument("--script-261007", type=Path, default=None)
    p.add_argument("--script-261013", type=Path, default=None)
    p.add_argument("--capnobase-path", type=Path, default=None)
    p.add_argument("--bidmc-path", type=Path, default=None)
    p.add_argument("--steam2-ppg-dir", type=Path, default=None)
    p.add_argument("--steam2-rsp-dir", type=Path, default=None)
    return p.parse_args(list(sys.argv[1:] if argv is None else argv))


def rank_dataset(frame: pd.DataFrame, top: int) -> pd.DataFrame:
    """Ranks of MAE (low first), |PCC| and |FFT| curve r (high first) among the eligible subjects; Score = mean."""
    frame = frame.copy()
    ok = frame["Eligible"]
    for col, name, ascending in (("MAE_O", "Rank_MAE", True), ("AbsPCC_O", "Rank_AbsPCC", False),
                                 ("Spectral_r", "Rank_Spectral_r", False)):
        frame[name] = math.nan
        frame.loc[ok, name] = frame.loc[ok, col].rank(ascending=ascending, method="average")
    frame["Score"] = frame[["Rank_MAE", "Rank_AbsPCC", "Rank_Spectral_r"]].mean(axis=1)
    frame = frame.sort_values(["Eligible", "Score", "Spectral_r", "MAE_O"], ascending=[False, True, False, True])
    frame["Rank"] = math.nan
    n = int(ok.sum())
    frame.loc[frame["Eligible"], "Rank"] = np.arange(1, n + 1)
    frame["Selected"] = frame["Eligible"] & (frame["Rank"] <= top)
    return frame


def main(argv: Optional[Sequence[str]] = None) -> None:
    args = parse_args(argv)
    base = load_base(args.script_261118)
    out_root = Path(args.output_root)
    out_root.mkdir(parents=True, exist_ok=True)
    window_frames, missing, ranked = [], [], []
    for dataset in args.datasets:
        source = base.SOURCE[dataset]
        roots = [args.root_261007] if source == "261007" else [args.root_261013]
        script = base.find_script(base.SCRIPTS[source], args.script_261007 if source == "261007" else args.script_261013,
                                  roots)
        mod = base.load_script(script)
        for key, field, value in (("capnobase", "data_path", args.capnobase_path), ("bidmc", "data_path", args.bidmc_path),
                                  ("steam2", "ppg_dir", args.steam2_ppg_dir), ("steam2", "rsp_dir", args.steam2_rsp_dir)):
            if value:
                mod.DATASET_CONFIGS[key][field] = str(value)
        spec = next(s for s in mod.WINDOW_SPECS if f"{s.order:02d}_{s.name}" == base.WINDOW)
        print(f"\n{dataset}: preprocessing of {script.name}", flush=True)
        data = mod.load_dataset(base.KEYS[dataset], spec)
        ann1_all = base.read_ann1(mod) if dataset == "BIDMC" else {}
        rows, signals_of = [], {}
        for subject in sorted(data):
            issue = any(base.same_subject(subject, x) for x in base.EXCLUDED[dataset])
            files = base.model_files(args, mod, dataset, MODEL, subject)
            if not files:
                missing.append({"Dataset": dataset, "Subject": subject, "Missing": f"stored results of {base.LABELS[MODEL]}"})
                continue
            minutes = o_minutes(base, args, mod, dataset, subject, data[subject], files, ann1_all.get(subject))
            scored = minutes[minutes.Scored] if not minutes.empty else minutes
            if scored.empty:
                missing.append({"Dataset": dataset, "Subject": subject, "Missing": "no scored minute"})
                continue
            windows, signals = subject_windows(base, args, mod, dataset, subject, data[subject], files, minutes)
            row = {"Dataset": dataset, "Subject": subject, "Issue_Subject": issue, "Scored_Minutes": int(len(scored)),
                   "Windows": int(windows["Spectral_r"].notna().sum()) if not windows.empty else 0,
                   "MAE_O": float(scored.AE_O.mean()), "AbsPCC_O": float(scored.PCC_O.abs().mean()),
                   "Spectral_r": float(windows["Spectral_r"].mean()) if not windows.empty else math.nan,
                   "Min_Spectral_r": float(windows["Spectral_r"].min()) if not windows.empty else math.nan,
                   "Window_r": float(windows["Window_r"].mean()) if not windows.empty else math.nan,
                   "Peak_Diff_BPM": float(windows["Peak_Diff_BPM"].mean()) if not windows.empty else math.nan,
                   "Max_Peak_Diff_BPM": float(windows["Peak_Diff_BPM"].max()) if not windows.empty else math.nan,
                   "Max_AE_O": float(scored.AE_O.max()), "Files_O": str(files["root"])}
            reasons = []
            if issue and not args.keep_issue_subjects:
                reasons.append("MAE issue subject")
            if not row["MAE_O"] < args.clinical_mae:
                reasons.append(f"MAE {row['MAE_O']:.2f} (not < {args.clinical_mae:g})")
            if not row["AbsPCC_O"] > args.clinical_pcc:
                reasons.append(f"|PCC| {row['AbsPCC_O']:.2f} (not > {args.clinical_pcc:g})")
            if row["Scored_Minutes"] < args.min_minutes:
                reasons.append(f"{row['Scored_Minutes']} scored minutes (< {args.min_minutes})")
            if not row["Windows"] or not np.isfinite(row["Spectral_r"]):
                reasons.append("no window with an |FFT|")
            row["Eligible"], row["Not_Eligible_Because"] = not reasons, "; ".join(reasons)
            rows.append(row)
            if not windows.empty:
                window_frames.append(windows)
            signals_of[subject] = (windows, signals, files)
            print(f"  {subject}: MAE {row['MAE_O']:.2f}, |PCC| {row['AbsPCC_O']:.2f}, |FFT| curve r "
                  f"{row['Spectral_r']:.3f} ({row['Windows']} windows){'' if row['Eligible'] else ' - ' + row['Not_Eligible_Because']}",
                  flush=True)
        if rows:
            table = rank_dataset(pd.DataFrame(rows), args.top)
            ranked.append(table)
            print(f"\n  {dataset}: top {args.top} of {int(table['Eligible'].sum())} eligible subjects"
                  f" (MAE < {args.clinical_mae:g}, |PCC| > {args.clinical_pcc:g}, >= {args.min_minutes} scored minutes)")
            for r in table[table["Selected"]].to_dict("records"):
                windows, signals, files = signals_of[r["Subject"]]
                folder = out_root / dataset / f"{int(r['Rank'])}_{dataset}_{mod.safe_tag(r['Subject'])}"
                n = draw_subject(base, args, mod, dataset, r["Subject"], r, windows, signals, folder,
                                 len(data[r["Subject"]]["references"]) > 1)
                print(f"    {int(r['Rank'])}. {r['Subject']}: MAE {r['MAE_O']:.2f}, |PCC| {r['AbsPCC_O']:.2f}, "
                      f"|FFT| curve r {r['Spectral_r']:.3f} (worst window {r['Min_Spectral_r']:.3f}), "
                      f"|peak diff| {r['Peak_Diff_BPM']:.2f} /min; score {r['Score']:.2f}; {n} windows drawn -> {folder}",
                      flush=True)
        del data
    ranking = pd.concat(ranked, ignore_index=True) if ranked else pd.DataFrame()
    top = ranking[ranking["Selected"]] if not ranking.empty else ranking
    cols = ["Dataset", "Rank", "Subject", "Score", "MAE_O", "AbsPCC_O", "Spectral_r", "Min_Spectral_r", "Window_r",
            "Peak_Diff_BPM", "Max_Peak_Diff_BPM", "Max_AE_O", "Scored_Minutes", "Windows", "Rank_MAE", "Rank_AbsPCC",
            "Rank_Spectral_r"]
    notes = pd.DataFrame([
        ("Model", "Fusion O (with the 1x1 stem fusion), member m0 of every LOSOCV fold; stored outputs read as in "
                  "261118_StemFusion_Threshold_Groups_Waveforms"),
        ("GT", "CapnoBase: peak detection on the capnogram (RRest rule); BIDMC: ann1 (expert-label rule"
               + (", minutes with an ann1 gap left out" if args.ann1_nogap else "") + "); STEAM2: stored GT"),
        ("Windows", f"every {args.window_sec:g} s half of every scored 60 s minute; windows outside a scored minute "
                    "are not used"),
        ("MAE_O / AbsPCC_O", "mean |stored rate - GT breath count| and mean |Wave_PCC| over the scored minutes"),
        ("Spectral_r", "Pearson r of the GT and Fusion O |FFT| curves of a window as drawn (detrend, Hamming, zero "
                       "padding to >= 32768 points, each scaled to its peak = 1, 0 Hz to 1 Hz or the top of the rate "
                       "band); subject value = mean over its windows"),
        ("Peak_Diff_BPM", "|peak of Fusion O - peak of GT| in the rate band of the scoring, breaths/min; the true "
                          "resolution of a 30 s window is 1/30 Hz = 2 breaths/min"),
        ("Selection", f"eligible: MAE_O < {args.clinical_mae:g} and AbsPCC_O > {args.clinical_pcc:g} (good restoration), "
                      f">= {args.min_minutes} scored minutes"
                      + ("" if args.keep_issue_subjects else ", not an MAE issue subject (CapnoBase 0115, BIDMC 27, "
                                                             "STEAM2 12_YJJ, 22_LMS, 26_CKK)")
                      + f"; Score = mean of the ranks of MAE_O (low first), AbsPCC_O and Spectral_r (high first) "
                        f"within the dataset; the {args.top} lowest scores are selected (ties: higher Spectral_r)"),
    ], columns=["Item", "Description"])
    path = out_root / f"{NAME}.xlsx"
    with pd.ExcelWriter(path, engine="openpyxl") as writer:
        (top[cols].round(4) if not top.empty else pd.DataFrame({"Result": ["none selected"]})).to_excel(
            writer, sheet_name="Top", index=False)
        (ranking[cols + ["Eligible", "Not_Eligible_Because", "Selected", "Issue_Subject", "Files_O"]].round(4)
         if not ranking.empty else ranking).to_excel(writer, sheet_name="Ranking", index=False)
        (pd.concat(window_frames, ignore_index=True).round(4) if window_frames else pd.DataFrame()).to_excel(
            writer, sheet_name="Windows", index=False)
        pd.DataFrame(missing, columns=["Dataset", "Subject", "Missing"]).to_excel(writer, sheet_name="Missing", index=False)
        notes.to_excel(writer, sheet_name="Notes", index=False)
    print("\nSelected (Fusion O: best restoration and |FFT| curve closest to the GT)")
    for r in (top.to_dict("records") if not top.empty else []):
        print(f"  {r['Dataset']} {int(r['Rank'])}. {r['Subject']}: MAE {r['MAE_O']:.2f}, |PCC| {r['AbsPCC_O']:.2f}, "
              f"|FFT| curve r {r['Spectral_r']:.3f}, |peak diff| {r['Peak_Diff_BPM']:.2f} /min")
    print(f"\nFigures under {out_root}; table: {path}")


if __name__ == "__main__":
    main()
