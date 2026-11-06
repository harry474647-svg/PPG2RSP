r"""261106_RRWaveNet_Performance_Boxplots: box plots of the per-subject
performance of RRWaveNet (full model, ensemble) from stored results of three
runs, each run drawn and saved on its own (no comparison between runs, no
statistical test; nothing is trained or re-run).

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

Run:  python 261106_RRWaveNet_Performance_Boxplots.py
      [--runs 261105 261007 261013] [--root-261105 <folder>]
      [--root-261007 <folder>] [--root-261013 <folder>] [--output-root <folder>]
      [--exclude-capnobase 0115] [--exclude-bidmc 27] [--steam2-worst 1 2]
      [--exclude-steam2 <subject> <subject>]
"""

import argparse
import json
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

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


def run_one(run: str, root: Path, out_root: Path, args: argparse.Namespace) -> None:
    config = RUNS[run]
    title = config["title"]
    if not root.exists():
        print(f"[{run}] results folder not found: {root} (skipped)")
        return
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
        return
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
    args = parser.parse_args()
    for run in args.runs:
        run_one(run, Path(getattr(args, f"root_{run}")), Path(args.output_root), args)


if __name__ == "__main__":
    main()
