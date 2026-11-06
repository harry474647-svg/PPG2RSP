r"""261106_RRWaveNet_Performance_Boxplots: box plots of the per-subject
performance of RRWaveNet (full model, ensemble) from stored results of three
runs, and its waveforms drawn over those of RespDiff and the deep
corr-encoder; each run drawn and saved on its own (no comparison between
runs, no statistical test; nothing is trained or re-run).

Runs (each in its own output folder)
------------------------------------
  261105  261105_RRWaveNet_Unified_Optimized: CapnoBase, BIDMC, STEAM2; final
          ensemble (m0 of every weight decay candidate + m1, m2 of the
          candidate selected in the fold), in the folder of the selected
          candidate (WD_Selection\...\{subject}_selection.json)
  261007  261007_RRWaveNet_Ensemble_NoEvent: CapnoBase, BIDMC, STEAM2; full
          model ("full"), ensemble of m0-m2
  261013  261013_RRWaveNet_Adult_SubjectVal_WD: BIDMC, STEAM2; ensemble of the
          weight decay candidate selected in the fold (WD_Selection)
Window: 10 s without overlap (01_w10s_nonoverlap), as in the papers' models.

Per subject (held-out subject of its LOSOCV fold)
-------------------------------------------------
Scored minutes = the minutes the run itself scored: reference-valid 60 s
minutes with an FFT rate ({subject}_ensemble_RR60s_windows.csv), so the
values are those of per_subject_fft.csv of each run.
  RR MAE        mean over the subject's minutes of |FFT rate of the predicted
                waveform - reference rate| (breaths/min) (= FFT_MAE)
  PCC           mean over the subject's minutes of the Pearson r between the
                predicted and the target waveform (= Wave_PCC_Mean)
  Waveform MAE  mean over the same minutes of the mean |predicted - target|
                sample by sample in the minute. Predicted and target
                waveforms of the stored ensemble output
                ({subject}_ensemble_outputs.npz), the 10 s windows put back on
                the recording's time axis as in the runs (overlap-add mean).
                Unit: the target's scale (each 10 s target window min-max
                scaled to [-1, 1]; the model output is tanh, also [-1, 1]).
The PCC of every minute is recomputed from the same arrays and checked
against the stored one (Coverage: PCC_Check_Max_Abs_Diff), so the waveform
MAE is taken on the same samples as the stored PCC.

Subject sets (each run, each in its own folder)
  01_All_subjects               every subject
  02_Excluded_STEAM2_worst1     CapnoBase without 0115, BIDMC without 27,
                                STEAM2 without its subject with the largest
                                subject RR MAE in this run
  03_Excluded_STEAM2_worst2     the same, STEAM2 without its 2 subjects with
                                the largest subject RR MAE in this run
--exclude-capnobase / --exclude-bidmc change the given subjects (an ID
written with or without leading zeros matches, 0115 = 115);
--exclude-steam2 gives the STEAM2 subjects instead of the ranking;
--steam2-worst sets the numbers of STEAM2 subjects (default 1 2). A given
subject not found in a run is reported and nothing is removed for it.

Outputs (default D:\PPG2RSP_RRWaveNet_Inspired\261106_RRWaveNet_Performance_Boxplots)
  {run}\                        261105, 261007, 261013
    {run}_Summary_by_subject_set.xlsx   Summary of every subject set (per
                                dataset: N, mean, SD, median, IQR, min,
                                max), Exclusions (removed subjects, their
                                RR MAE and rank), Coverage (files found, PCC
                                check), Notes; the same as CSV files
    01_All_subjects\, 02_Excluded_STEAM2_worst1\, 03_Excluded_STEAM2_worst2\
      {run}_{set}_RR_MAE_boxplot.png / .pdf
      {run}_{set}_PCC_boxplot.png / .pdf
      {run}_{set}_Waveform_MAE_boxplot.png / .pdf
      {run}_{set}_Performance_boxplots.png / .pdf   the three side by side
      {run}_{set}_Performance.xlsx   Summary, Per_Subject, Per_Minute,
                                     Exclusions, Notes (and CSV files)
      ({set} = all, excl_worst1, excl_worst2)
Box: median and IQR, whiskers 1.5 IQR; every subject is a point; the
second title line names the subjects left out.

Waveform overlays with the earlier models (each run, its own folder)
--------------------------------------------------------------------
RRWaveNet of the run (the same stored ensemble output) drawn over the GT with
RespDiff (261028 v3 / 261101 v4, {respdiff-root}\respdiff\{dataset}\
Predictions\{subject}_ddpm50.npz) and the deep corr-encoder (261028,
{correncoder-root}\{dataset}\Predictions\{subject}_outputs.npz), read,
aligned and measured as in 261104_Waveform_Overlay and
261105_model_discussion_analysis (their functions are used: helper scripts
261104_Waveform_Overlay.py, 261030_Waveform_Morphology_Analysis.py and
261007_RRWaveNet_Ensemble_NoEvent.py are looked for in this folder, the
current folder, D:\PPG2RSP_RRWaveNet_Inspired and its subfolders, or
--scripts-dir):
* GT = the reference as recorded at 30 Hz (CapnoBase capnogram, BIDMC
  impedance, STEAM2 belt); RRWaveNet with its training sign / lag undone and
  low-pass 1 Hz; every model shifted by <= half a breath and inverted if
  needed, once per recording, to its best r with the GT (same rule for all).
* Windows of 30 s and 60 s (--sizes), one starting every 5 s (--step); in a
  window per model: AE = |FFT rate - GT rate| (breaths/min; GT rate from the
  breaths: CapnoBase expert labels, BIDMC ann1, STEAM2 the GT cycles), PCC
  with the GT, cycles missed / extra; score = PCC - 0.1 x min(AE, 5);
  Advantage = RRWaveNet's score - the best other score.
* Subject MAE / PCC per model = mean over the subject's non-overlapping
  windows of the size.
* One figure per subject and size: the window with the largest Advantage;
  GT (gray, thick) and the models, each min-max scaled in the window; beside
  the plot (top right, outside the waveforms) the subject's MAE and PCC per
  model and this window's AE and r.
Outputs per run: {run}\Waveform_Overlay\{30s,60s}\{dataset}\{subject or recording}_{t0}-{t1}s.png,
{run}\Waveform_Overlay\{30s,60s}\{run}_{dataset}_{size}s_overlays.pdf,
{run}\Waveform_Overlay\{run}_Waveform_Overlay_Summary.xlsx (Candidates,
Drawn_Windows, Subject_Summary, Dataset_Summary, Grid_Windows, Notes; every
window in {run}_overlay_windows_all.csv), and for all runs together
Waveform_Overlay_Summary_all_runs.xlsx (Candidates = drawn windows with
RRWaveNet AE <= 1 and Advantage > 0, largest first, with what the others do
there; Subject_Summary with RRWaveNet's rank and gain per subject;
Dataset_Summary; Drawn_Windows; Notes).

Run:  python 261106_RRWaveNet_Performance_Boxplots.py
      [--runs 261105 261007 261013] [--root-261105 <folder>]
      [--root-261007 <folder>] [--root-261013 <folder>] [--output-root <folder>]
      [--exclude-capnobase 0115] [--exclude-bidmc 27] [--steam2-worst 1 2]
      [--exclude-steam2 <subject> <subject>]
      overlays: [--no-overlay] [--sizes 30 60] [--step 5] [--subjects ...]
      [--respdiff-root <folder>] [--correncoder-root <folder>] [--scripts-dir <folder>]
      [--capnobase-path ...] [--bidmc-path ...] [--steam2-ppg-dir ...] [--steam2-rsp-dir ...]
"""

import argparse
import importlib.util
import json
import math
import sys
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.io as sio
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.lines import Line2D

WORKSPACE = Path(r"D:\PPG2RSP_RRWaveNet_Inspired")
PROJECT_ROOT = WORKSPACE / "261106_RRWaveNet_Performance_Boxplots"
FS = 64  # model and analysis sampling rate of the runs (Hz)
WINDOW = "01_w10s_nonoverlap"
DATASETS = ("CapnoBase", "BIDMC", "STEAM2")

# run -> results folder, how the fold's model folder is found, title
RUNS = {
    "261105": {
        "root": WORKSPACE / "261105_RRWaveNet_Unified_Optimized",
        "selected": True,
        "title": "261105 unified (LOSOCV, subject-wise validation, WD per fold, final ensemble of 6)",
    },
    "261007": {
        "root": WORKSPACE / "261007_RRWaveNet_NoEvent_v1",
        "selected": False,
        "title": "261007 (full model, temporal validation, ensemble of 3)",
    },
    "261013": {
        "root": WORKSPACE / "261013_RRWavNet_STEAM_BIDMC_v1",
        "selected": True,
        "title": "261013 (subject-wise validation, WD per fold, ensemble of 3)",
    },
}

METRICS = (
    ("RR_MAE", "RR MAE", "Subject RR MAE (breaths/min)"),
    ("PCC", "PCC", "Subject waveform PCC (r)"),
    ("Waveform_MAE", "Waveform MAE", "Subject waveform MAE (target scale, [-1, 1])"),
)
# Subjects left out of the excluded subject sets (see the docstring).
EXCLUDE_CAPNOBASE = ("0115",)
EXCLUDE_BIDMC = ("27",)
STEAM2_WORST = (1, 2)
COLORS = {"CapnoBase": "#2a78d6", "BIDMC": "#eb6834", "STEAM2": "#1baf7a"}
INK, MUTED, GRID = "#1f1f1f", "#6b6a66", "#e4e3df"


# ---------------------------------------------------------------------------
# Locating the stored ensemble results
# ---------------------------------------------------------------------------

def read_json(path: Path) -> Optional[dict]:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


def fold_files(root: Path, dataset: str, selected: bool) -> List[dict]:
    """Every fold of a dataset: subject tag, the model folder of the fold and
    its stored per-minute table and ensemble outputs."""
    folds = []
    if selected:
        for path in sorted((root / "WD_Selection" / WINDOW / dataset).glob("*_selection.json")):
            tag = path.name[: -len("_selection.json")]
            variant = (read_json(path) or {}).get("Selected_Variant")
            folds.append({"Tag": tag, "Variant": variant or ""})
    else:
        summaries = root / "full" / WINDOW / dataset / "Fold_Summaries"
        for path in sorted(summaries.glob("*_ensemble_RR60s_windows.csv")):
            if path.name.endswith("_selected_ensemble_RR60s_windows.csv"):
                continue
            folds.append({"Tag": path.name[: -len("_ensemble_RR60s_windows.csv")], "Variant": "full"})
    for fold in folds:
        condition = root / fold["Variant"] / WINDOW / dataset
        fold["Minutes_File"] = condition / "Fold_Summaries" / f"{fold['Tag']}_ensemble_RR60s_windows.csv"
        fold["Outputs_File"] = condition / "Ensemble_Predictions" / f"{fold['Tag']}_ensemble_outputs.npz"
    return folds


# ---------------------------------------------------------------------------
# Per-minute and per-subject metrics
# ---------------------------------------------------------------------------

def finite_or_median(x: np.ndarray) -> np.ndarray:
    """The runs' sanitize_signal: non-finite samples -> median of the rest."""
    x = np.asarray(x, dtype=np.float64).reshape(-1)
    ok = np.isfinite(x)
    if ok.all():
        return x
    if not ok.any():
        return np.zeros_like(x)
    x = x.copy()
    x[~ok] = float(np.median(x[ok]))
    return x


def reconstruct(arrays: dict, segment_id: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Predicted and target waveform of one segment on its time axis
    (overlap-add mean of the windows, as reconstruct_segment of the runs)."""
    predictions, targets = arrays["predictions"], arrays["targets"]
    index = np.flatnonzero(np.asarray(arrays["segment_ids"]).astype(int) == int(segment_id))
    if not len(index):
        return np.empty(0), np.empty(0), np.empty(0, bool)
    n = int(predictions.shape[1])
    starts = np.asarray(arrays["window_starts"]).astype(int)
    length = int(starts[index].max()) + n
    p_sum, t_sum, cover = np.zeros(length), np.zeros(length), np.zeros(length, int)
    for i in index:
        a = starts[i]
        p_sum[a:a + n] += finite_or_median(predictions[i])[:n]
        t_sum[a:a + n] += finite_or_median(targets[i])[:n]
        cover[a:a + n] += 1
    valid = cover > 0
    pred, true = np.full(length, np.nan), np.full(length, np.nan)
    pred[valid] = p_sum[valid] / cover[valid]
    true[valid] = t_sum[valid] / cover[valid]
    return pred, true, valid


def pearson(x: np.ndarray, y: np.ndarray) -> float:
    x, y = finite_or_median(x), finite_or_median(y)
    if len(x) < 2 or np.std(x) < 1e-8 or np.std(y) < 1e-8:
        return np.nan
    return float(np.corrcoef(x, y)[0, 1])


def true_flags(values: pd.Series) -> pd.Series:
    return values.astype(str).str.strip().str.lower().isin(("true", "1", "1.0"))


def fold_minutes(fold: dict, dataset: str) -> Tuple[pd.DataFrame, dict]:
    """Scored minutes of one fold with the waveform MAE, and a coverage row."""
    info = {"Dataset": dataset, "Subject_Tag": fold["Tag"], "Variant": fold["Variant"],
            "Minutes_File": str(fold["Minutes_File"]), "Outputs_File": str(fold["Outputs_File"])}
    for key, path in (("Minutes_File", fold["Minutes_File"]), ("Outputs_File", fold["Outputs_File"])):
        if not path.exists():
            info["Status"] = f"missing {key.replace('_', ' ').lower()}"
            return pd.DataFrame(), info
    minutes = pd.read_csv(fold["Minutes_File"], dtype={"Subject": str}, encoding="utf-8-sig")
    if minutes.empty:
        info["Status"] = "no minutes"
        return pd.DataFrame(), info
    minutes["AE_Wave_FFT"] = pd.to_numeric(minutes["AE_Wave_FFT"], errors="coerce")
    scored = minutes[true_flags(minutes["Ref_Valid"]) & np.isfinite(minutes["AE_Wave_FFT"])].copy()
    info.update({"N_Minutes_Stored": int(len(minutes)), "N_Minutes_Scored": int(len(scored))})
    if scored.empty:
        info["Status"] = "no scored minute"
        return pd.DataFrame(), info
    with np.load(fold["Outputs_File"], allow_pickle=False) as data:
        arrays = {key: data[key] for key in ("predictions", "targets", "segment_ids", "window_starts")}
    series: Dict[int, tuple] = {}
    wave_mae, pcc_check = [], []
    for _, row in scored.iterrows():
        segment = int(row["Segment_ID"])
        if segment not in series:
            series[segment] = reconstruct(arrays, segment)
        pred, true, valid = series[segment]
        a = int(round(float(row["RR_Window_Start_Sec"]) * FS))
        b = int(round(float(row["RR_Window_End_Sec"]) * FS))
        keep = valid[a:b]
        if b > len(pred) or not keep.any():
            wave_mae.append(np.nan)
            pcc_check.append(np.nan)
            continue
        p, t = pred[a:b][keep], true[a:b][keep]
        wave_mae.append(float(np.mean(np.abs(p - t))))
        pcc_check.append(pearson(t, p))
    scored["Waveform_MAE"] = wave_mae
    scored["PCC_Recomputed"] = pcc_check
    stored = pd.to_numeric(scored["Wave_PCC"], errors="coerce").to_numpy()
    diff = np.abs(stored - np.asarray(pcc_check, float))
    info["PCC_Check_Max_Abs_Diff"] = float(np.nanmax(diff)) if np.isfinite(diff).any() else np.nan
    info["N_Minutes_No_Waveform"] = int(np.isnan(wave_mae).sum())
    info["Status"] = "ok"
    subject = str(scored["Subject"].iloc[0]) if "Subject" in scored.columns else fold["Tag"]
    info["Subject"] = subject
    out = pd.DataFrame({
        "Dataset": dataset,
        "Subject": subject,
        "Variant": fold["Variant"],
        "Segment_ID": scored["Segment_ID"].astype(int).to_numpy(),
        "Minute_Start_Sec": scored["RR_Window_Start_Sec"].to_numpy(float),
        "GT_RR": pd.to_numeric(scored["GT_Breath_Count"], errors="coerce").to_numpy(float),
        "Pred_RR": pd.to_numeric(scored["Pred_Wave_FFT_RR"], errors="coerce").to_numpy(float),
        "RR_AE": scored["AE_Wave_FFT"].to_numpy(float),
        "PCC": stored,
        "Waveform_MAE": scored["Waveform_MAE"].to_numpy(float),
    })
    return out, info


def per_subject(minutes: pd.DataFrame) -> pd.DataFrame:
    return (
        minutes.groupby(["Dataset", "Subject", "Variant"], sort=False)
        .agg(N_Minutes=("RR_AE", "size"), RR_MAE=("RR_AE", "mean"), RR_MAE_SD=("RR_AE", "std"),
             PCC=("PCC", "mean"), PCC_SD=("PCC", "std"),
             Waveform_MAE=("Waveform_MAE", "mean"), Waveform_MAE_SD=("Waveform_MAE", "std"))
        .reset_index()
    )


def summary_table(subjects: pd.DataFrame, datasets: Sequence[str]) -> pd.DataFrame:
    rows = []
    for dataset in datasets:
        frame = subjects[subjects["Dataset"] == dataset]
        for key, label, _ in METRICS:
            v = frame[key].dropna().to_numpy(float)
            if not len(v):
                continue
            q1, med, q3 = np.percentile(v, [25, 50, 75])
            rows.append({
                "Dataset": dataset, "Metric": label, "N_Subjects": len(v),
                "N_Minutes": int(frame["N_Minutes"].sum()),
                "Mean": float(v.mean()), "SD": float(v.std(ddof=1)) if len(v) > 1 else np.nan,
                "Median": float(med), "Q1": float(q1), "Q3": float(q3), "IQR": float(q3 - q1),
                "Min": float(v.min()), "Max": float(v.max()),
            })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------

def value_text(value: float, key: str) -> str:
    return f"{value:.3f}" if key != "RR_MAE" else f"{value:.2f}"


def draw_box(ax, subjects: pd.DataFrame, datasets: Sequence[str], key: str, ylabel: str) -> None:
    rng = np.random.default_rng(0)
    values = [subjects.loc[subjects["Dataset"] == d, key].dropna().to_numpy(float) for d in datasets]
    positions = np.arange(1, len(datasets) + 1)
    box = ax.boxplot(values, positions=positions, widths=0.5, showfliers=False, patch_artist=True,
                     medianprops={"color": INK, "lw": 1.4}, whiskerprops={"color": MUTED, "lw": 1},
                     capprops={"color": MUTED, "lw": 1}, boxprops={"edgecolor": MUTED, "lw": 1})
    for patch, dataset in zip(box["boxes"], datasets):
        patch.set_facecolor(COLORS.get(dataset, "#8a8984"))
        patch.set_alpha(0.35)
    top = max((v.max() for v in values if len(v)), default=1.0)
    bottom = min((v.min() for v in values if len(v)), default=0.0)
    span = (top - bottom) or 1.0
    for pos, dataset, v in zip(positions, datasets, values):
        if not len(v):
            continue
        ax.scatter(pos + rng.uniform(-0.12, 0.12, len(v)), v, s=12, color=COLORS.get(dataset, "#8a8984"),
                   edgecolor="white", linewidth=0.5, zorder=3)
        ax.text(pos, top + 0.05 * span, f"median {value_text(float(np.median(v)), key)}",
                ha="center", va="bottom", fontsize=8, color=INK)
    ax.set_ylim(bottom - 0.05 * span, top + 0.14 * span)
    ax.set_xticks(positions)
    ax.set_xticklabels([f"{d}\n(n={len(v)})" for d, v in zip(datasets, values)])
    ax.set_ylabel(ylabel)
    ax.grid(axis="y", color=GRID, lw=0.6)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)


def save(fig, path: Path) -> None:
    fig.savefig(path.with_suffix(".png"), dpi=300, bbox_inches="tight")
    fig.savefig(path.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def figures(prefix: str, title: str, subset: str, subjects: pd.DataFrame, datasets: Sequence[str],
            out: Path) -> None:
    """One box plot per metric and the three side by side; `subset` names
    the subjects left out (second title line)."""
    width = 1.6 + 1.3 * len(datasets)
    for key, label, ylabel in METRICS:
        fig, ax = plt.subplots(figsize=(max(width, 4.6), 4.4))
        draw_box(ax, subjects, datasets, key, ylabel)
        ax.set_title(f"{label}: {title}\n{subset}", fontsize=8.5, loc="left")
        save(fig, out / f"{prefix}_{key}_boxplot")
    fig, axes = plt.subplots(1, len(METRICS), figsize=(width * len(METRICS), 4.4))
    for ax, (key, label, ylabel), letter in zip(axes, METRICS, "abc"):
        draw_box(ax, subjects, datasets, key, ylabel)
        ax.set_title(f"({letter}) {label}", fontsize=10, loc="left")
    fig.suptitle(f"{title}\n{subset}", fontsize=10, x=0.01, ha="left")
    fig.tight_layout()
    save(fig, out / f"{prefix}_Performance_boxplots")


# ---------------------------------------------------------------------------
# One run
# ---------------------------------------------------------------------------

def notes(run: str, root: Path, title: str) -> pd.DataFrame:
    lines = [
        ("Run", title),
        ("Results folder", str(root)),
        ("Window", f"{WINDOW} (10 s windows without overlap)"),
        ("Model", "the run's stored ensemble output of each held-out subject "
                  "({subject}_ensemble_outputs.npz and _ensemble_RR60s_windows.csv)"),
        ("Scored minutes", "reference-valid 60 s minutes with an FFT rate, as scored by the run "
                           "(same minutes as per_subject_fft.csv)"),
        ("RR MAE", "per subject: mean |FFT rate of the predicted waveform - reference rate| over its "
                   "minutes (breaths/min) = FFT_MAE of per_subject_fft.csv"),
        ("PCC", "per subject: mean over its minutes of the Pearson r between predicted and target "
                "waveform = Wave_PCC_Mean of per_subject_fft.csv"),
        ("Waveform MAE", "per subject: mean over the same minutes of mean |predicted - target| per "
                         "sample; target scale (each 10 s target window min-max to [-1, 1])"),
        ("PCC check", "PCC recomputed from the stored waveforms vs the stored PCC "
                      "(Coverage: PCC_Check_Max_Abs_Diff); ~0 = the waveform MAE uses the same samples"),
        ("Box", "median and IQR, whiskers 1.5 IQR, every subject a point; no test (runs are not compared)"),
        ("Subject sets", "01_All_subjects: every subject; 02 / 03: CapnoBase and BIDMC without the given "
                         "subjects (default 0115 and 27) and STEAM2 without its 1 / 2 subjects with the "
                         "largest subject RR MAE in this run (or the given ones, --exclude-steam2); the "
                         "removed subjects with their RR MAE and rank are in Exclusions"),
    ]
    return pd.DataFrame(lines, columns=["Item", "Description"])


# ---------------------------------------------------------------------------
# Subject sets (all subjects / outliers removed)
# ---------------------------------------------------------------------------

def same_subject(a: object, b: object) -> bool:
    """'0115' == '115' == '0115'; '27' == '27'; other names must match exactly."""
    a, b = str(a).strip(), str(b).strip()
    return a == b or (a.isdigit() and b.isdigit() and int(a) == int(b))


def subject_sets(args: argparse.Namespace) -> List[Tuple[str, str, Optional[int]]]:
    """(folder, file prefix, number of STEAM2 subjects removed or None)."""
    sets = [("01_All_subjects", "all", None)]
    for i, k in enumerate(args.steam2_worst, start=2):
        sets.append((f"{i:02d}_Excluded_STEAM2_worst{k}", f"excl_worst{k}", k))
    return sets


def removed_subjects(subjects: pd.DataFrame, k: Optional[int], args: argparse.Namespace,
                     folder: str) -> pd.DataFrame:
    """Subjects left out of a set, with their subject RR MAE and its rank
    (1 = largest) within the dataset of this run."""
    columns = ["Subject_Set", "Dataset", "Subject", "Rule", "Found", "N_Minutes", "RR_MAE", "Rank_RR_MAE",
               "N_Subjects_Dataset"]
    if k is None:
        return pd.DataFrame(columns=columns)
    rows = []
    for dataset in DATASETS:
        frame = subjects[subjects["Dataset"] == dataset].sort_values("RR_MAE", ascending=False)
        frame = frame.reset_index(drop=True)
        if frame.empty:
            continue
        if dataset == "STEAM2":
            if args.exclude_steam2:
                wanted, rule = list(args.exclude_steam2[:k]), f"given (--exclude-steam2), first {k}"
            else:
                wanted, rule = frame["Subject"].head(k).tolist(), f"{k} largest subject RR MAE of this run"
        else:
            wanted = list(args.exclude_capnobase if dataset == "CapnoBase" else args.exclude_bidmc)
            rule = "given"
        for name in wanted:
            hit = frame.index[frame["Subject"].map(lambda s: same_subject(s, name))]
            row = {"Subject_Set": folder, "Dataset": dataset, "Subject": str(name), "Rule": rule,
                   "Found": bool(len(hit)), "N_Subjects_Dataset": len(frame)}
            if len(hit):
                found = frame.loc[hit[0]]
                row.update({"Subject": found["Subject"], "N_Minutes": int(found["N_Minutes"]),
                            "RR_MAE": float(found["RR_MAE"]), "Rank_RR_MAE": int(hit[0]) + 1})
            rows.append(row)
    return pd.DataFrame(rows, columns=columns)


def subset_text(removed: pd.DataFrame) -> str:
    found = removed[removed["Found"].astype(bool)]
    if found.empty:
        return "All subjects"
    parts = [f"{d} {', '.join(found.loc[found['Dataset'] == d, 'Subject'].astype(str))}"
             for d in DATASETS if (found["Dataset"] == d).any()]
    return "Excluded: " + "; ".join(parts)


def without(frame: pd.DataFrame, removed: pd.DataFrame) -> pd.DataFrame:
    if removed.empty:
        return frame
    found = removed[removed["Found"].astype(bool)]
    drop = set(zip(found["Dataset"], found["Subject"].astype(str)))
    keep = [(d, str(s)) not in drop for d, s in zip(frame["Dataset"], frame["Subject"])]
    return frame[np.asarray(keep, bool)]


def write_excel(path: Path, sheets: Dict[str, pd.DataFrame]) -> None:
    with pd.ExcelWriter(path, engine="openpyxl") as writer:
        for name, frame in sheets.items():
            frame.to_excel(writer, sheet_name=name, index=False)
            sheet = writer.sheets[name]
            for i, column in enumerate(frame.columns):
                width = max([len(str(column))] + [len(str(v)) for v in frame[column].head(200)])
                sheet.column_dimensions[sheet.cell(1, i + 1).column_letter].width = min(max(width + 2, 8), 80)


def run_one(run: str, root: Path, out_root: Path, args: argparse.Namespace
            ) -> Optional[Tuple[pd.DataFrame, pd.DataFrame]]:
    """Box plots of one run; returns its per-subject table and the subjects
    left out of the excluded sets (None when the run has no result)."""
    config = RUNS[run]
    title = config["title"]
    if not root.exists():
        print(f"[{run}] results folder not found: {root} (skipped)")
        return None
    minute_frames, coverage = [], []
    datasets = []
    for dataset in DATASETS:
        folds = fold_files(root, dataset, config["selected"])
        if not folds:
            continue
        for fold in folds:
            frame, info = fold_minutes(fold, dataset)
            coverage.append(info)
            if not frame.empty:
                minute_frames.append(frame)
        if any(f["Dataset"].iloc[0] == dataset for f in minute_frames):
            datasets.append(dataset)
    coverage = pd.DataFrame(coverage)
    if not minute_frames:
        print(f"[{run}] no stored ensemble result under {root} (skipped)")
        return None
    minutes = pd.concat(minute_frames, ignore_index=True)
    subjects = per_subject(minutes)
    out = out_root / run
    out.mkdir(parents=True, exist_ok=True)
    print(f"\n[{run}] {title}")
    for dataset in datasets:
        n_ok = int(((coverage["Dataset"] == dataset) & (coverage["Status"] == "ok")).sum())
        print(f"  {dataset:<9s} subjects with results {n_ok}/{int((coverage['Dataset'] == dataset).sum())}")
    bad = coverage[coverage["Status"] != "ok"]
    if not bad.empty:
        print(f"  {len(bad)} fold(s) not used (see {run}_coverage.csv)")
    if "PCC_Check_Max_Abs_Diff" in coverage.columns:
        check = pd.to_numeric(coverage["PCC_Check_Max_Abs_Diff"], errors="coerce").to_numpy(float)
        if np.isfinite(check).any():
            print(f"  PCC check: largest |recomputed - stored| = {np.nanmax(check):.2e}")

    summaries, removals = [], []
    for folder, tag, k in subject_sets(args):
        removed = removed_subjects(subjects, k, args, folder)
        subset = subset_text(removed)
        kept = without(subjects, removed)
        kept_minutes = without(minutes, removed)
        summary = summary_table(kept, datasets)
        target = out / folder
        target.mkdir(parents=True, exist_ok=True)
        prefix = f"{run}_{tag}"
        figures(prefix, title, subset, kept, datasets, target)
        for name, frame in {"summary": summary, "per_subject": kept, "per_minute": kept_minutes,
                            "exclusions": removed}.items():
            frame.to_csv(target / f"{prefix}_{name}.csv", index=False, encoding="utf-8-sig")
        write_excel(target / f"{prefix}_Performance.xlsx", {
            "Summary": summary.round(4), "Per_Subject": kept.round(4), "Per_Minute": kept_minutes.round(4),
            "Exclusions": removed.round(4), "Notes": notes(run, root, title)})
        summaries.append(summary.assign(Subject_Set=folder, Subjects_Excluded=subset))
        removals.append(removed)
        print(f"  {folder}: {subset}")
        found = removed["Found"].astype(bool)
        for _, row in removed[~found].iterrows():
            print(f"    ! {row['Dataset']} {row['Subject']} not found in this run (nothing removed)")
        for _, row in removed[found].iterrows():
            print(f"    removed {row['Dataset']} {row['Subject']}: RR MAE {row['RR_MAE']:.2f} "
                  f"(rank {int(row['Rank_RR_MAE'])} of {int(row['N_Subjects_Dataset'])})")
        for dataset in datasets:
            rows = summary[summary["Dataset"] == dataset].set_index("Metric")
            parts = [f"{m} {rows.loc[m, 'Median']:.3f} [{rows.loc[m, 'Q1']:.3f}, {rows.loc[m, 'Q3']:.3f}]"
                     for _, m, _ in METRICS if m in rows.index]
            n = int(rows["N_Subjects"].iloc[0]) if len(rows) else 0
            print(f"    {dataset:<9s} n={n:<3d} median [IQR]: " + "; ".join(parts))

    by_set = pd.concat(summaries, ignore_index=True)
    by_set = by_set[["Subject_Set", "Subjects_Excluded"] + [c for c in by_set.columns
                                                          if c not in ("Subject_Set", "Subjects_Excluded")]]
    removals = pd.concat(removals, ignore_index=True)
    for name, frame in {"summary_by_subject_set": by_set, "exclusions": removals, "coverage": coverage}.items():
        frame.to_csv(out / f"{run}_{name}.csv", index=False, encoding="utf-8-sig")
    write_excel(out / f"{run}_Summary_by_subject_set.xlsx", {
        "Summary": by_set.round(4), "Exclusions": removals.round(4), "Coverage": coverage,
        "Notes": notes(run, root, title)})
    print(f"  saved under {out}")
    return subjects, removals


# ---------------------------------------------------------------------------
# Waveform overlays: GT, RespDiff, deep corr-encoder and RRWaveNet
# (readers of 261104_Waveform_Overlay, window measures of
# 261105_model_discussion_analysis)
# ---------------------------------------------------------------------------

SCRIPT_DIR = Path(__file__).resolve().parent
HELPER_FILES = {
    "ov": "261104_Waveform_Overlay.py",              # data / result readers, Track, filters
    "m30": "261030_Waveform_Morphology_Analysis.py",  # breathing cycle detector
    "m07": "261007_RRWaveNet_Ensemble_NoEvent.py",    # CapnoBase reader (expert breath labels)
}
RESPDIFF_ROOT = WORKSPACE / "261020_RespDiff_v2"
CORRENCODER_ROOT = WORKSPACE / "261028_Deep_correncoder"
RESPDIFF_SAMPLING = "ddpm50"
RRWAVENET_LOWPASS_HZ = 1.0
FS30 = 30  # time axis of the overlays: the 30 Hz of RespDiff, the corr-encoder and the GT
MODELS = ("RespDiff", "Deep corr-encoder", "RRWaveNet")
OTHERS = MODELS[:2]
SHORT = {"RespDiff": "RespDiff", "Deep corr-encoder": "CorrEncoder", "RRWaveNet": "RRWaveNet"}
MODEL_COLORS = {"RespDiff": "#2a78d6", "Deep corr-encoder": "#eb6834", "RRWaveNet": "#1baf7a"}
DATASET_KEYS = {"CapnoBase": "capnobase", "BIDMC": "bidmc", "STEAM2": "steam2"}
GT_NAMES = {"capnobase": "capnogram", "bidmc": "impedance", "steam2": "belt"}
RATE_BAND_BPM = {"capnobase": (3.0, 60.0), "bidmc": (6.0, 30.0), "steam2": (6.0, 30.0)}
WINDOW_SIZES = (30, 60)
STEP_SEC = 5.0
MIN_RUN_SEC = 30.0
MIN_BREATHS = 3
AE_WEIGHT, AE_CAP = 0.1, 5.0
RATE_OK_BPM = 1.0
DOUBLE, HALF = (1.75, 2.25), (0.4, 0.6)

ov = m30 = m07 = None  # helper modules, loaded by load_helpers


def find_script(name: str, scripts_dir: Optional[Path]) -> Path:
    """A helper script: --scripts-dir, else the first found in this script's
    folder, the current folder, the workspace, the parent folder and their
    subfolders."""
    folders = [Path(scripts_dir)] if scripts_dir else []
    folders += [SCRIPT_DIR, Path.cwd(), WORKSPACE, SCRIPT_DIR.parent]
    for base in (WORKSPACE, SCRIPT_DIR.parent):
        if base.is_dir():
            folders += sorted(child for child in base.iterdir() if child.is_dir())
    seen = []
    for folder in folders:
        path = folder / name
        if path not in seen:
            seen.append(path)
            if path.is_file():
                return path.resolve()
    raise FileNotFoundError(f"{name} not found (looked in {SCRIPT_DIR}, {Path.cwd()}, {WORKSPACE} and its "
                            "subfolders); give its folder with --scripts-dir")


def load_module(key: str, scripts_dir: Optional[Path]):
    path = find_script(HELPER_FILES[key], scripts_dir)
    spec = importlib.util.spec_from_file_location(f"helper_{key}", str(path))
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    print(f"  {HELPER_FILES[key]}: {path}")
    return module


def load_helpers(args: argparse.Namespace, capnobase: bool) -> None:
    global ov, m30, m07
    ov = load_module("ov", args.scripts_dir)
    m30 = load_module("m30", args.scripts_dir)
    if capnobase:
        m07 = load_module("m07", args.scripts_dir)
    for key, path in (("capnobase", args.capnobase_path), ("bidmc", args.bidmc_path)):
        if path:
            ov.DATASET_CONFIGS[key]["data_path"] = str(path)
    if args.steam2_ppg_dir:
        ov.DATASET_CONFIGS["steam2"]["ppg_dir"] = str(args.steam2_ppg_dir)
    if args.steam2_rsp_dir:
        ov.DATASET_CONFIGS["steam2"]["rsp_dir"] = str(args.steam2_rsp_dir)


def read_recordings(dataset_key: str) -> Dict[str, Dict[str, dict]]:
    """Subject -> {recording name: GT (30 Hz), clipped GT samples, breath times
    (s; None without labels)}, named and cut as 261104 and the corr-encoder
    (as 261105_model_discussion_analysis)."""
    cfg = ov.DATASET_CONFIGS[dataset_key]
    modality = ov.MODALITIES[dataset_key]
    out: Dict[str, Dict[str, dict]] = {}

    def add(subject, name, ppg, resp, ppg_fs, resp_fs, breaths):
        ppg30, resp30 = ov.to_fs(ppg, ppg_fs), ov.to_fs(resp, resp_fs)
        length = min(len(ppg30), len(resp30))
        clipped = ov.to_fs_mask(ov.saturated_samples(resp, resp_fs, modality, ov.SATURATION_CONFIG), resp_fs,
                                length, ov.SATURATION_CONFIG["saturation_margin_sec"])
        out.setdefault(subject, {})[name] = {"gt": resp30[:length], "clipped": clipped, "breaths": breaths}

    if dataset_key == "capnobase":
        csv_dir = Path(cfg["data_path"])
        for case_id in ov.capnobase_case_ids(csv_dir):
            try:
                case = m07.read_capnobase_case(csv_dir, case_id, cfg["default_fs"])
                breaths = case.get("breaths")
                breaths = None if breaths is None else np.sort(np.asarray(breaths, float)) / float(case["co2_fs"])
                add(case_id, case_id, case["ppg"], case["co2"], case["ppg_fs"], case["co2_fs"], breaths)
            except Exception as exc:
                print(f"[CapnoBase {case_id}] not read: {exc}")
    elif dataset_key == "bidmc":
        fs = float(cfg["default_fs"])
        mat = sio.loadmat(cfg["data_path"])
        for index, record in enumerate(mat["data"][0]):
            subject = f"{index + 1:02d}"
            try:
                ppg = np.asarray(record["ppg"][0, 0]["v"]).squeeze()
                ref = record["ref"][0, 0]
                imp = np.asarray(ref["resp_sig"][0, 0]["imp"][0, 0]["v"]).squeeze()
                ann = np.asarray(ref["breaths"][0, 0]["ann1"], float).ravel()  # MATLAB sample numbers
                add(subject, subject, ppg, imp, fs, fs, np.sort(ann - 1.0) / fs)
            except Exception as exc:
                print(f"[BIDMC {subject}] not read: {exc}")
    else:
        for subject, items in ov.steam2_session_files(Path(cfg["ppg_dir"]), Path(cfg["rsp_dir"])).items():
            try:
                for name, ppg_path, rsp_path in items:
                    ppg = ov.read_eeglab_channel(ppg_path, "PPG", cfg["default_fs"])
                    rsp = ov.read_eeglab_channel(rsp_path, "RSP", cfg["default_fs"])
                    pieces, _ = ov.steam2_session_pieces(ppg, rsp)
                    for k, (p, r, p_fs, r_fs) in enumerate(pieces):
                        add(subject, f"{name}_piece{k + 1}", p, r, p_fs, r_fs, None)
            except Exception as exc:
                print(f"[STEAM2 {subject}] not read: {exc}")
    return out


def rrwavenet_track(segment: dict, stored: Optional[dict], dataset_key: str, gt: np.ndarray,
                    lowpass_hz: float) -> np.ndarray:
    """RRWaveNet on the GT axis with its training sign / lag undone and the
    1 Hz low-pass (as 261104)."""
    lag = stored["lag_sec"] if stored else 0.0
    if stored:
        sign = stored["sign"]
    elif dataset_key != "capnobase":
        r = ov.target_reference_r(ov.Track(segment["target"], ov.RRWAVENET_FS, lag), ov.Track(gt, FS30))
        sign = -1.0 if r < 0 else 1.0
    else:
        sign = 1.0
    smoothed = ov.lowpass_runs(sign * segment["output"], ov.RRWAVENET_FS, lowpass_hz, ov.RRWAVENET_LOWPASS_ORDER)
    return ov.Track(smoothed, ov.RRWAVENET_FS, lag).at(np.arange(len(gt)) / FS30)


def fft_rate(x: np.ndarray, fs: float, band_bpm: Tuple[float, float]) -> float:
    """Frequency (per minute) of the largest FFT peak in the band."""
    x = np.asarray(x, float)
    x = x[np.isfinite(x)]
    if len(x) < 8 or np.std(x) <= 1e-12:
        return math.nan
    n = max(16384, len(x))
    f = np.fft.rfftfreq(n, 1.0 / fs) * 60.0
    p = np.abs(np.fft.rfft(x - np.mean(x), n))
    keep = (f >= band_bpm[0]) & (f <= band_bpm[1])
    return float(f[keep][np.argmax(p[keep])])


def shifted(values: np.ndarray, shift: int) -> np.ndarray:
    out = np.full(len(values), np.nan)
    if shift >= 0:
        out[shift:] = values[: len(values) - shift]
    else:
        out[:shift] = values[-shift:]
    return out


def align(gt: np.ndarray, values: np.ndarray, period_sec: float) -> np.ndarray:
    """The model shifted by at most half a breath and, if needed, inverted, to
    its largest Pearson r with the GT over the recording (every model alike)."""
    best = (-np.inf, 1.0, 0)
    reach = int(round(0.5 * period_sec * FS30))
    for shift in range(-reach, reach + 1):
        r = ov.pearson(gt, shifted(values, shift))
        if np.isfinite(r):
            for sign in (1.0, -1.0):
                if sign * r > best[0]:
                    best = (sign * r, sign, shift)
    if not np.isfinite(best[0]):
        return np.asarray(values, float)
    return best[1] * shifted(values, best[2])


def recording_cycles(values: Dict[str, np.ndarray], modality: str) -> Dict[str, List[dict]]:
    """Breathing cycles of every waveform (261030's detector) on the stretches
    (>= 30 s) where all exist."""
    valid = np.ones(len(values["GT"]), bool)
    for v in values.values():
        valid &= np.isfinite(v)
    out = {key: [] for key in values}
    for s, e in m30.finite_runs(valid, int(MIN_RUN_SEC * FS30)):
        for key, v in values.items():
            y = m30.band_filter(v[s:e], FS30, modality, m30.CONFIG)
            cycles, _ = m30.detect_cycles(y, FS30, modality, m30.CONFIG)
            out[key] += [{k: c[k] + s / FS30 for k in ("start", "transition", "end")} for c in cycles]
    return out


def inside(cycles: List[dict], t0: float, t1: float) -> List[dict]:
    return [c for c in cycles if c["start"] >= t0 and c["end"] <= t1]


def match_cycles(gt: List[dict], model: List[dict]) -> int:
    """GT cycles matched 1:1 by a model cycle starting within half their length."""
    used, matched = set(), 0
    starts = np.array([c["start"] for c in model]) if model else np.empty(0)
    for c in gt:
        tolerance = 0.5 * (c["end"] - c["start"])
        best, best_gap = None, np.inf
        for k in np.flatnonzero(np.abs(starts - c["start"]) <= tolerance):
            gap = abs(starts[k] - c["start"])
            if k not in used and gap < best_gap:
                best, best_gap = int(k), gap
        if best is not None:
            used.add(best)
            matched += 1
    return matched


def window_row(values: Dict[str, np.ndarray], cycles: Dict[str, List[dict]], breaths: np.ndarray, t0: float,
               t1: float, band: Tuple[float, float]) -> Optional[dict]:
    """Rate, AE, PCC, score and cycles of every model in one window (None when
    a waveform has a gap in it or the GT has fewer than 3 breaths)."""
    a, b = int(round(t0 * FS30)), int(round(t1 * FS30))
    if not all(np.isfinite(v[a:b]).all() for v in values.values()):
        return None
    br = breaths[(breaths >= t0) & (breaths < t1)]
    if len(br) < MIN_BREATHS:
        return None
    gt_cycles = inside(cycles["GT"], t0, t1)
    gt_rr = 60.0 / float(np.mean(np.diff(br)))
    row = {"Start_Sec": t0, "End_Sec": t1, "GT_RR": gt_rr, "GT_Cycles": len(gt_cycles),
           "GT_Clipped_Cycles": sum(bool(c.get("clipped")) for c in gt_cycles)}
    for model in MODELS:
        if model not in values:
            continue
        seg = values[model][a:b]
        rate = fft_rate(seg, FS30, band)
        near = [c for c in cycles[model] if c["end"] > t0 and c["start"] < t1]
        matched = match_cycles(gt_cycles, near)
        n_in = len(inside(cycles[model], t0, t1))
        ae = abs(rate - gt_rr)
        pcc = ov.pearson(values["GT"][a:b], seg)
        s = SHORT[model]
        row.update({f"{s}_Rate": rate, f"{s}_Ratio": rate / gt_rr, f"{s}_AE": ae, f"{s}_PCC": pcc,
                    f"{s}_Score": pcc - AE_WEIGHT * min(ae, AE_CAP), f"{s}_Cycles": n_in,
                    f"{s}_Missed": len(gt_cycles) - matched, f"{s}_Extra": max(0, n_in - matched),
                    f"{s}_Coverage": matched / len(gt_cycles) if gt_cycles else math.nan})
    others = [row[f"{SHORT[m]}_Score"] for m in OTHERS if np.isfinite(row.get(f"{SHORT[m]}_Score", math.nan))]
    row["Advantage"] = row["RRWaveNet_Score"] - max(others) if others else math.nan
    return row


def why_text(row: dict) -> str:
    """What the other models do in a window where RRWaveNet is right."""
    parts = []
    for model in OTHERS:
        s = SHORT[model]
        ratio = row.get(f"{s}_Ratio", math.nan)
        if not np.isfinite(ratio):
            continue
        if DOUBLE[0] <= ratio <= DOUBLE[1]:
            parts.append(f"{s} at 2x rate")
        elif HALF[0] <= ratio <= HALF[1]:
            parts.append(f"{s} at half rate")
        elif row.get(f"{s}_AE", 0) > 2:
            parts.append(f"{s} rate off by {row[f'{s}_AE']:.1f}")
        missed = row.get(f"{s}_Missed", 0) - row.get("RRWaveNet_Missed", 0)
        if missed >= 2:
            parts.append(f"{s} misses {int(missed)} more cycles")
        gap = row.get("RRWaveNet_PCC", math.nan) - row.get(f"{s}_PCC", math.nan)
        if np.isfinite(gap) and gap >= 0.3:
            parts.append(f"{s} PCC lower by {gap:.2f}")
    return "; ".join(parts)


def subject_means(rows: List[dict], size: int) -> Tuple[dict, str]:
    """Per model the mean AE (MAE), PCC and score over the subject's
    non-overlapping windows of a size (all its windows when none is complete)."""
    grid = [r for r in rows if r["Size_Sec"] == size and r["Grid"]]
    basis = "non-overlapping windows"
    if not grid:
        grid = [r for r in rows if r["Size_Sec"] == size]
        basis = "overlapping windows (no complete non-overlapping one)"
    out = {"N_Windows": len(grid)}
    for model in MODELS:
        s = SHORT[model]
        ae = [r[f"{s}_AE"] for r in grid if f"{s}_AE" in r and np.isfinite(r[f"{s}_AE"])]
        if ae:
            pcc = [r[f"{s}_PCC"] for r in grid if np.isfinite(r.get(f"{s}_PCC", math.nan))]
            out.update({f"{s}_MAE": float(np.mean(ae)), f"{s}_PCC": float(np.mean(pcc)) if pcc else math.nan,
                        f"{s}_Score": float(np.mean([r[f"{s}_Score"] for r in grid if f"{s}_Score" in r]))})
    wins = [r["Advantage"] > 0 for r in grid if np.isfinite(r.get("Advantage", math.nan))]
    out["RRWaveNet_Best_Window_Share"] = float(np.mean(wins)) if wins else math.nan
    return out, basis


def overlay_figure(path: Path, pdf, run: str, dataset: str, subject: str, recording: Optional[str], size: int,
                   row: dict, values: Dict[str, np.ndarray], means: dict, basis: str, gt_name: str) -> None:
    """GT and the models in one window, each min-max scaled in the window; the
    subject means (MAE, PCC) and this window's AE / r beside the plot (top
    right, outside the waveforms)."""
    t0, t1 = row["Start_Sec"], row["End_Sec"]
    a, b = int(round(t0 * FS30)), int(round(t1 * FS30))
    t = t0 + np.arange(b - a) / FS30
    side_in, left_in = 3.5, 0.65
    width = 7.0 + 4.0 * size / 60.0
    fig = plt.figure(figsize=(width, 3.7))
    ax = fig.add_axes([left_in / width, 0.15, (width - left_in - side_in - 0.15) / width, 0.66])
    handles = [Line2D([], [], color="#0b0b0b", lw=2.6, alpha=0.35)]
    labels = [f"{'GT (' + gt_name + ')':<18s}{row['GT_RR']:5.1f} breaths/min"]
    ax.plot(t, ov.minmax(values["GT"][a:b]), color="#0b0b0b", lw=2.6, alpha=0.35, zorder=1)
    for z, model in enumerate(MODELS):
        s = SHORT[model]
        if model not in values or f"{s}_MAE" not in means:
            continue
        lw = 1.9 if model == "RRWaveNet" else 1.3
        ax.plot(t, ov.minmax(values[model][a:b]), color=MODEL_COLORS[model], lw=lw, zorder=2 + z)
        handles.append(Line2D([], [], color=MODEL_COLORS[model], lw=lw))
        labels.append(f"{model:<18s}MAE {means[f'{s}_MAE']:5.2f}  PCC {means[f'{s}_PCC']:5.2f}")
    ax.set_xlim(t0, t1)
    ax.set_ylim(-1.18, 1.18)
    ax.set_xlabel("Time in the recording (s)")
    ax.set_ylabel("Min-max scaled in the window")
    ax.grid(True, color=GRID, lw=0.6)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    name = f"{subject}" + (f" ({recording})" if recording else "")
    fig.text(left_in / width, 0.95, f"{run} · {dataset} {name} · {size} s window {t0:.0f}-{t1:.0f} s",
             fontsize=10, color=INK, va="top")
    fig.text(left_in / width, 0.885, "models aligned to the GT (shift <= half a breath, sign); each waveform "
             "min-max scaled in the window; RRWaveNet low-pass 1 Hz", fontsize=7.5, color=MUTED, va="top")
    x = 1.0 - (side_in - 0.1) / width
    legend = fig.legend(handles, labels, loc="upper left", bbox_to_anchor=(x, 0.84), frameon=False,
                        prop={"family": "monospace", "size": 7.5},
                        title=f"Subject mean, {means['N_Windows']} x {size} s windows", title_fontsize=7.5)
    try:
        legend._legend_box.align = "left"
    except Exception:
        pass
    lines = [f"{'This window':<18s}{'AE':>6s}{'r':>7s}"]
    for model in MODELS:
        s = SHORT[model]
        if f"{s}_AE" in row and model in values:
            lines.append(f"{model:<18s}{row[f'{s}_AE']:6.2f}{row[f'{s}_PCC']:7.2f}")
    lines += ["", "MAE / AE: |FFT rate - GT rate| (breaths/min)", f"subject mean over {basis}"]
    fig.text(x + 0.004, 0.36, "\n".join(lines), family="monospace", fontsize=7, color=INK, va="top")
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=200)
    pdf.savefig(fig)
    plt.close(fig)


def excluded_sets(exclusions: Optional[pd.DataFrame], dataset: str, subject: str) -> str:
    if exclusions is None or exclusions.empty:
        return ""
    found = exclusions[exclusions["Found"].astype(bool) & (exclusions["Dataset"] == dataset)]
    hits = found[found["Subject"].map(lambda s: same_subject(s, subject))]
    return ", ".join(sorted(set(hits["Subject_Set"].astype(str))))


def overlay_dataset(args: argparse.Namespace, dataset: str, runs: Dict[str, dict], out_root: Path) -> None:
    """Every subject of every run with RRWaveNet and at least one other model:
    windows of every size, the subject means and one figure per size (the
    window where RRWaveNet's score is furthest above the best other model)."""
    key = DATASET_KEYS[dataset]
    band = RATE_BAND_BPM[key]
    modality = ov.MODALITIES[key]
    corr = ov.load_correncoder(Path(args.correncoder_root), dataset)
    respdiff = ov.load_respdiff(Path(args.respdiff_root), dataset, args.respdiff_sampling)
    plan = {}
    for run, info in runs.items():
        files = {f["Tag"]: f["Outputs_File"] for f in fold_files(info["root"], dataset, RUNS[run]["selected"])
                 if f["Outputs_File"].exists()}
        subjects = sorted(set(files) & (set(corr) | set(respdiff)))
        if args.subjects:
            subjects = [s for s in subjects if any(same_subject(s, x) for x in args.subjects)]
        print(f"{dataset} [{run}]: RespDiff {len(respdiff)}, corr-encoder {len(corr)}, RRWaveNet {len(files)}; "
              f"{len(subjects)} subjects with RRWaveNet and another model", flush=True)
        if subjects:
            plan[run] = (files, subjects)
    if not plan:
        return
    recordings = read_recordings(key)
    for run, (files, subjects) in plan.items():
        info = runs[run]
        alignment = ov.rrwavenet_alignment_table(info["root"], dataset)
        pdfs = {}
        for subject in subjects:
            if subject not in recordings:
                print(f"  [{run}] {dataset} {subject}: no data file read (skipped)")
                continue
            segments = ov.load_rrwavenet(files[subject])
            corr_by_name = {r["name"]: r for r in corr.get(subject, [])}
            store, rows = {}, []
            for k, (name, rec) in enumerate(recordings[subject].items()):
                gt = np.asarray(rec["gt"], float)
                times = np.arange(len(gt)) / FS30
                index = corr_by_name[name]["index"] if name in corr_by_name else k
                raw = {}
                if name in respdiff.get(subject, {}):
                    raw["RespDiff"] = ov.Track(respdiff[subject][name], FS30).at(times)
                if name in corr_by_name:
                    raw["Deep corr-encoder"] = ov.Track(corr_by_name[name]["output"], FS30).at(times)
                if index in segments:
                    raw["RRWaveNet"] = rrwavenet_track(segments[index], alignment.get((subject, index)), key, gt,
                                                       args.rrwavenet_lowpass_hz)
                if "RRWaveNet" not in raw or len(raw) < 2:
                    continue
                breaths = rec["breaths"]
                if breaths is None or len(breaths) < MIN_BREATHS:
                    breaths = np.array([c["start"] for c in recording_cycles({"GT": gt}, modality)["GT"]])
                period = float(np.median(np.diff(breaths))) if len(breaths) >= 2 else 60.0 / np.mean(band)
                values = {"GT": gt, **{m: align(gt, v, period) for m, v in raw.items()}}
                cycles = recording_cycles(values, modality)
                clipped = np.asarray(rec["clipped"], bool)
                for c in cycles["GT"]:
                    c["clipped"] = bool(clipped[int(c["start"] * FS30):int(math.ceil(c["end"] * FS30)) + 1].any())
                store[name] = values
                length = len(gt) / FS30
                for size in args.sizes:
                    grid = set(np.arange(0.0, length - size + 1e-9, size).round(3))
                    starts = sorted(grid | set(np.arange(0.0, length - size + 1e-9, args.step).round(3)))
                    for t0 in starts:
                        row = window_row(values, cycles, breaths, float(t0), float(t0) + size, band)
                        if row is not None:
                            rows.append({"Run": run, "Dataset": dataset, "Subject": subject, "Recording": name,
                                         "Size_Sec": size, "Grid": t0 in grid,
                                         "Models": ", ".join(SHORT[m] for m in MODELS if m in values), **row})
            info["windows"] += rows
            flag = excluded_sets(info["exclusions"], dataset, subject)
            official = info["official"].get((dataset, subject), {})
            for size in args.sizes:
                sized = [r for r in rows if r["Size_Sec"] == size and np.isfinite(r["Advantage"])]
                if not sized:
                    continue
                means, basis = subject_means(rows, size)
                best = max(sized, key=lambda r: (r["Advantage"], r["RRWaveNet_PCC"]))
                single = len(recordings[subject]) == 1
                rec_name = None if single else best["Recording"]
                recording = str(best["Recording"])
                stem = (ov.safe_name(subject) if single else ov.safe_name(recording) if
                        recording.startswith(subject) else f"{ov.safe_name(subject)}__{ov.safe_name(recording)}")
                figure = (out_root / run / "Waveform_Overlay" / f"{size}s" / dataset /
                          f"{stem}_{best['Start_Sec']:.0f}-{best['End_Sec']:.0f}s.png")
                if size not in pdfs:
                    pdf_path = out_root / run / "Waveform_Overlay" / f"{size}s" / f"{run}_{dataset}_{size}s_overlays.pdf"
                    pdf_path.parent.mkdir(parents=True, exist_ok=True)
                    pdfs[size] = PdfPages(pdf_path)
                overlay_figure(figure, pdfs[size], run, dataset, subject, rec_name, size, best,
                               store[best["Recording"]], means, basis, GT_NAMES[key])
                summary = {"Run": run, "Dataset": dataset, "Subject": subject, "Size_Sec": size,
                           "Subject_Means_Over": basis, **means,
                           "Official_RR_MAE": official.get("RR_MAE", math.nan),
                           "Official_PCC": official.get("PCC", math.nan), "In_Excluded_Sets": flag}
                info["subjects"].append(summary)
                drawn = {**best, "Why": why_text(best), "In_Excluded_Sets": flag,
                         **{f"Subject_{k}": v for k, v in means.items()},
                         "Figure": str(figure.relative_to(out_root))}
                info["drawn"].append(drawn)
            print(f"  [{run}] {dataset} {subject}: {len(rows)} windows, models "
                  f"{', '.join(sorted({m for r in rows for m in r['Models'].split(', ')}))}", flush=True)
        for pdf in pdfs.values():
            pdf.close()


def ranks(frame: pd.DataFrame, metric: str, lower_better: bool) -> pd.Series:
    """Rank of RRWaveNet among the models present (1 = best)."""
    columns = [f"{SHORT[m]}_{metric}" for m in MODELS if f"{SHORT[m]}_{metric}" in frame.columns]
    values = frame[columns]
    rr = values[f"RRWaveNet_{metric}"]
    better = (values.lt(rr, axis=0) if lower_better else values.gt(rr, axis=0)).sum(axis=1)
    return (better + 1).where(rr.notna())


def subject_table(subjects: pd.DataFrame) -> pd.DataFrame:
    if subjects.empty:
        return subjects
    out = subjects.copy()
    others_mae = out[[c for c in ("RespDiff_MAE", "CorrEncoder_MAE") if c in out]].min(axis=1)
    others_pcc = out[[c for c in ("RespDiff_PCC", "CorrEncoder_PCC") if c in out]].max(axis=1)
    out["RRWaveNet_MAE_Rank"] = ranks(out, "MAE", True)
    out["RRWaveNet_PCC_Rank"] = ranks(out, "PCC", False)
    out["MAE_Gain_vs_Best_Other"] = others_mae - out["RRWaveNet_MAE"]
    out["PCC_Gain_vs_Best_Other"] = out["RRWaveNet_PCC"] - others_pcc
    return out


def dataset_table(subjects: pd.DataFrame) -> pd.DataFrame:
    """Per run, dataset and size: means and medians of the subject values of
    the subjects with every model of the dataset, and how often RRWaveNet is
    best."""
    rows = []
    if subjects.empty:
        return pd.DataFrame()
    for (run, dataset, size), d in subjects.groupby(["Run", "Dataset", "Size_Sec"], sort=False):
        present = [f"{SHORT[m]}_MAE" for m in MODELS if f"{SHORT[m]}_MAE" in d and d[f"{SHORT[m]}_MAE"].notna().any()]
        full = d.dropna(subset=present)
        row = {"Run": run, "Dataset": dataset, "Size_Sec": size, "N_Subjects": len(d),
               "Models_Compared": ", ".join(c[:-4] for c in present), "N_Subjects_All_Models": len(full)}
        for model in MODELS:
            s = SHORT[model]
            for metric in ("MAE", "PCC"):
                if f"{s}_{metric}" in full:
                    row[f"{s}_{metric}_Mean"] = full[f"{s}_{metric}"].mean()
                    row[f"{s}_{metric}_Median"] = full[f"{s}_{metric}"].median()
        if len(full):
            row["RRWaveNet_Best_MAE_Subjects"] = int((full["RRWaveNet_MAE_Rank"] == 1).sum())
            row["RRWaveNet_Best_PCC_Subjects"] = int((full["RRWaveNet_PCC_Rank"] == 1).sum())
            row["RRWaveNet_Best_Window_Share_Mean"] = full["RRWaveNet_Best_Window_Share"].mean()
        rows.append(row)
    return pd.DataFrame(rows)


def candidates(drawn: pd.DataFrame) -> pd.DataFrame:
    """Drawn windows where RRWaveNet is right (AE <= 1 breath/min) and ahead
    of the best other model, the largest lead first."""
    if drawn.empty:
        return drawn
    keep = drawn[(drawn["RRWaveNet_AE"] <= RATE_OK_BPM) & (drawn["Advantage"] > 0)].copy()
    keep = keep.sort_values("Advantage", ascending=False)
    keep.insert(0, "Rank", np.arange(1, len(keep) + 1))
    first = ["Rank", "Run", "Dataset", "Subject", "Recording", "Size_Sec", "Start_Sec", "End_Sec", "GT_RR",
             "Advantage", "Why", "RRWaveNet_AE", "RRWaveNet_PCC", "RespDiff_AE", "RespDiff_PCC", "RespDiff_Ratio",
             "CorrEncoder_AE", "CorrEncoder_PCC", "CorrEncoder_Ratio", "Subject_RRWaveNet_MAE",
             "Subject_RRWaveNet_PCC", "Subject_RespDiff_MAE", "Subject_RespDiff_PCC", "Subject_CorrEncoder_MAE",
             "Subject_CorrEncoder_PCC", "In_Excluded_Sets", "Figure"]
    first = [c for c in first if c in keep.columns]
    return keep[first + [c for c in keep.columns if c not in first]]


def overlay_notes(args: argparse.Namespace) -> pd.DataFrame:
    lines = [
        ("Models", f"RespDiff ({args.respdiff_root}, {args.respdiff_sampling}), deep corr-encoder "
                   f"({args.correncoder_root}), RRWaveNet = the run's stored ensemble output (as the box plots)"),
        ("GT", "the reference as recorded (CapnoBase capnogram, BIDMC impedance, STEAM2 belt) at 30 Hz, read as "
               "261104; GT rate from the breaths in the window (CapnoBase expert labels, BIDMC ann1, STEAM2 the "
               "GT cycles); a window needs >= 3 breaths and every waveform without a gap"),
        ("Alignment", "every model shifted by <= half a breath and, if needed, inverted once per recording to its "
                      "best r with the GT (the same rule for all); RRWaveNet: training sign / lag undone, "
                      f"low-pass {args.rrwavenet_lowpass_hz:g} Hz (as 261104 / 261105_model_discussion_analysis)"),
        ("AE / MAE", "|FFT peak rate of the model's waveform in the rate band - GT rate| (breaths/min); "
                     "CapnoBase band 3-60, BIDMC / STEAM2 6-30; subject MAE = mean AE over the subject's "
                     "non-overlapping windows of a size (all its windows when none is complete)"),
        ("PCC", "Pearson r of the aligned model and the GT in the window; subject PCC = mean over the same windows"),
        ("Score / Advantage", f"score = PCC - {AE_WEIGHT} x min(AE, {AE_CAP}); Advantage = RRWaveNet's score - the "
                              "best score of the other models in the window (> 0: RRWaveNet ahead)"),
        ("Drawn window", "per subject and size the window (start every "
                         f"{args.step:g} s) with the largest Advantage; legend: subject means, beside it this "
                         "window's AE and r"),
        ("Candidates", f"drawn windows with RRWaveNet AE <= {RATE_OK_BPM} and Advantage > 0, largest Advantage "
                       "first; Why: the other models at twice / half the rate, rate off by > 2, >= 2 more GT "
                       "cycles missed than RRWaveNet, PCC lower by >= 0.3"),
        ("Official_RR_MAE / PCC", "the run's own per-subject values (60 s minutes, its target; the box plots), "
                                  "for reference; the overlay values use the recorded GT and the windows above"),
        ("In_Excluded_Sets", "the subject sets of the box plots that leave this subject out"),
        ("Cycles", "261030 detector on the aligned waveforms; Missed = GT cycles without a model cycle starting "
                   "within half their length (1:1); Extra = model cycles not matched; Coverage = matched / GT"),
    ]
    return pd.DataFrame(lines, columns=["Item", "Description"])


def write_overlay_tables(args: argparse.Namespace, runs: Dict[str, dict], out_root: Path) -> None:
    combined = {"Candidates": [], "Subject_Summary": [], "Dataset_Summary": [], "Drawn_Windows": []}
    for run, info in runs.items():
        if not info["windows"]:
            continue
        folder = out_root / run / "Waveform_Overlay"
        folder.mkdir(parents=True, exist_ok=True)
        windows = pd.DataFrame(info["windows"])
        subjects = subject_table(pd.DataFrame(info["subjects"]))
        drawn = pd.DataFrame(info["drawn"])
        cand = candidates(drawn)
        data = dataset_table(subjects)
        windows.to_csv(folder / f"{run}_overlay_windows_all.csv", index=False, encoding="utf-8-sig")
        sheets = {"Candidates": cand, "Drawn_Windows": drawn, "Subject_Summary": subjects,
                  "Dataset_Summary": data, "Grid_Windows": windows[windows["Grid"]]}
        for name, frame in sheets.items():
            frame.to_csv(folder / f"{run}_overlay_{name.lower()}.csv", index=False, encoding="utf-8-sig")
        write_excel(folder / f"{run}_Waveform_Overlay_Summary.xlsx",
                    {**{k: v.round(4) for k, v in sheets.items()}, "Notes": overlay_notes(args)})
        for name in combined:
            combined[name].append(sheets[name])
        print(f"[{run}] overlays: {len(drawn)} figures, {len(cand)} candidate windows; saved under {folder}")
    frames = {k: pd.concat(v, ignore_index=True) for k, v in combined.items() if v}
    if not frames:
        return
    if "Candidates" in frames and not frames["Candidates"].empty:
        cand = frames["Candidates"].drop(columns="Rank").sort_values("Advantage", ascending=False)
        cand.insert(0, "Rank", np.arange(1, len(cand) + 1))
        frames["Candidates"] = cand
    path = out_root / "Waveform_Overlay_Summary_all_runs.xlsx"
    write_excel(path, {**{k: v.round(4) for k, v in frames.items()}, "Notes": overlay_notes(args)})
    print(f"Overlay summary of every run: {path}")


def run_overlays(args: argparse.Namespace, results: Dict[str, tuple], out_root: Path) -> None:
    runs = {}
    for run, (root, result) in results.items():
        official = {}
        exclusions = None
        if result is not None:
            subjects, exclusions = result
            official = {(r["Dataset"], str(r["Subject"])): r for r in subjects.to_dict("records")}
        runs[run] = {"root": root, "exclusions": exclusions, "official": official,
                     "windows": [], "subjects": [], "drawn": []}
    datasets = [d for d in DATASETS if any(fold_files(info["root"], d, RUNS[run]["selected"])
                                           for run, info in runs.items())]
    if not datasets:
        print("\nOverlays: no RRWaveNet result found (skipped)")
        return
    print("\nOverlays: helper scripts")
    try:
        load_helpers(args, "CapnoBase" in datasets)
    except FileNotFoundError as exc:
        print(f"Overlays skipped: {exc}")
        return
    for dataset in datasets:
        overlay_dataset(args, dataset, runs, out_root)
    write_overlay_tables(args, runs, out_root)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--runs", nargs="+", choices=tuple(RUNS), default=list(RUNS))
    for run, config in RUNS.items():
        parser.add_argument(f"--root-{run}", type=Path, default=config["root"],
                            help=f"results folder of {run} (default {config['root']})")
    parser.add_argument("--output-root", type=Path, default=PROJECT_ROOT)
    parser.add_argument("--exclude-capnobase", nargs="*", default=list(EXCLUDE_CAPNOBASE),
                        help="CapnoBase subjects left out of the excluded sets (default 0115)")
    parser.add_argument("--exclude-bidmc", nargs="*", default=list(EXCLUDE_BIDMC),
                        help="BIDMC subjects left out of the excluded sets (default 27)")
    parser.add_argument("--steam2-worst", nargs="+", type=int, default=list(STEAM2_WORST),
                        help="number of STEAM2 subjects left out, one subject set each (default 1 2)")
    parser.add_argument("--exclude-steam2", nargs="*", default=[],
                        help="STEAM2 subjects to leave out, in order (default: the subjects with the largest "
                             "subject RR MAE of each run)")
    overlay = parser.add_argument_group("waveform overlays")
    overlay.add_argument("--no-overlay", action="store_true", help="only the box plots")
    overlay.add_argument("--sizes", nargs="+", type=int, default=list(WINDOW_SIZES), help="window sizes (s)")
    overlay.add_argument("--step", type=float, default=STEP_SEC, help="a window starts every this many s")
    overlay.add_argument("--subjects", nargs="+", default=None, help="only these subjects")
    overlay.add_argument("--respdiff-root", type=Path, default=RESPDIFF_ROOT)
    overlay.add_argument("--respdiff-sampling", default=RESPDIFF_SAMPLING)
    overlay.add_argument("--correncoder-root", type=Path, default=CORRENCODER_ROOT)
    overlay.add_argument("--rrwavenet-lowpass-hz", type=float, default=RRWAVENET_LOWPASS_HZ)
    overlay.add_argument("--scripts-dir", type=Path, default=None,
                         help="folder of 261104_Waveform_Overlay.py, 261030_Waveform_Morphology_Analysis.py and "
                              "261007_RRWaveNet_Ensemble_NoEvent.py (default: looked for)")
    overlay.add_argument("--capnobase-path", type=Path, default=None)
    overlay.add_argument("--bidmc-path", type=Path, default=None)
    overlay.add_argument("--steam2-ppg-dir", type=Path, default=None)
    overlay.add_argument("--steam2-rsp-dir", type=Path, default=None)
    args = parser.parse_args()
    out_root = Path(args.output_root)
    results = {}
    for run in args.runs:
        root = Path(getattr(args, f"root_{run}"))
        results[run] = (root, run_one(run, root, out_root, args))
    if not args.no_overlay:
        run_overlays(args, results, out_root)


if __name__ == "__main__":
    main()
