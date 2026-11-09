# -*- coding: utf-8 -*-
r"""261109_Ensemble_Comparison: the statistical comparison of the RRWaveNet results
with and without the ensemble (ensemble of 3 models vs its single models), per run,
dataset and window. Nothing is trained; the stored minute tables are read.

Runs (every one found is compared; --groups chooses)
  freq     D:\PPG2RSP_RRWaveNet_Inspired\261109_Waveform_FreqAnalysis
             01_PPG_0.1-0.8Hz_261013, 02_PPG_0.1-0.8Hz_261007, 03_PPG_0.1-8.0Hz_261013,
             04_PPG_0.1-8.0Hz_261007
  target   D:\PPG2RSP_RRWaveNet_Inspired\261109_RRWaveNet_TargetBand
             01_RSP_0.1-0.8Hz_261013, 02_RSP_0.1-0.8Hz_261007
  earlier  D:\PPG2RSP_RRWaveNet_Inspired\261013_RRWavNet_STEAM_BIDMC_v1, 261007_RRWaveNet_NoEvent_v1
           (raw PPG input, RSP target 0.1-0.5 Hz)
  and any run folder given with --roots.
From each run: Summary\minutes_ensemble.csv (the ensemble) and
Summary\minutes_single_models.csv (its members m0-m2, every scored minute); 261007: the
full model only (Variant = full; its ablation variants are other models); 261013: the
final model (selected weight decay, its three members). The minutes are those the base
scripts scored (reference-valid 60 s minutes), so both sides use the same minutes.

Per subject (as the base scripts): FFT MAE = mean |FFT rate - reference rate| over the
subject's minutes; Wave PCC = mean waveform Pearson r over the minutes. Single models:
each member's value, and their mean ("single-model average", the "w/o ensemble" row of
the base summaries).
Tests (subjects paired; subject-level is the unit, minutes are nested in subjects):
  ensemble vs single-model average  (primary)
  ensemble vs m0, vs m1, vs m2      (each single model)
  Wilcoxon signed-rank (two-sided), median and mean difference (single - ensemble for
  FFT MAE, ensemble - single for PCC: positive = the ensemble is better), subjects better /
  worse / tied, matched-pairs rank-biserial r (effect size, -1..1), Benjamini-Hochberg
  adjustment over all tests of one subject set and metric. Supplementary: the same
  Wilcoxon over the paired minutes (members averaged per minute; not independent).
Subject sets: all subjects, and issue subjects left out (STEAM2 22_LMS, 26_CKK, 12_YJJ;
BIDMC 27; CapnoBase 0115).

Outputs (default D:\PPG2RSP_RRWaveNet_Inspired\261109_Ensemble_Comparison)
  261109_Ensemble_Comparison.xlsx
    Summary      per run x dataset x window x subject set x metric: subjects, ensemble and
                 single-model average (mean +- SD, median [IQR]), difference, subjects
                 better, Wilcoxon p, BH p, effect size, significance (* < 0.05 after BH)
    Tests        every test (vs the average and vs every member), with the minute-level p
    Per_Subject  every subject: ensemble, m0, m1, m2, average (FFT MAE, PCC, minutes)
    Runs         runs found, their folders and minutes, runs not found
    Notes
  and Summary.csv, Tests.csv, Per_Subject.csv in the same folder.

Run:
  python 261109_Ensemble_Comparison.py
  python 261109_Ensemble_Comparison.py --groups freq          only the 261109_Waveform_FreqAnalysis runs
  python 261109_Ensemble_Comparison.py --roots D:\...\some_run  also this run folder
Options: --groups, --roots, --workspace, --output-root, --alpha
"""

import argparse
import math
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

NAME = "261109_Ensemble_Comparison"
WORKSPACE = Path(r"D:\PPG2RSP_RRWaveNet_Inspired")
ALPHA = 0.05
DATASET_ORDER = ("STEAM2", "BIDMC", "CapnoBase")
EXCLUDED = {"STEAM2": ("22_LMS", "26_CKK", "12_YJJ"), "BIDMC": ("27",), "CapnoBase": ("0115",)}
METRICS = (("FFT_MAE", "FFT_AE", True), ("Wave_PCC", "Wave_PCC", False))  # name, minute column, lower is better
MINUTE_KEY = ["Subject", "Segment_ID", "RR_Window_Start_Sec"]

# group -> [(folder relative to the workspace, label)]
RUN_GROUPS = {
    "freq": [
        ("261109_Waveform_FreqAnalysis/01_PPG_0.1-0.8Hz_261013", "FreqAnalysis 01: PPG 0.1-0.8 Hz, 261013"),
        ("261109_Waveform_FreqAnalysis/02_PPG_0.1-0.8Hz_261007", "FreqAnalysis 02: PPG 0.1-0.8 Hz, 261007"),
        ("261109_Waveform_FreqAnalysis/03_PPG_0.1-8.0Hz_261013", "FreqAnalysis 03: PPG 0.1-8.0 Hz, 261013"),
        ("261109_Waveform_FreqAnalysis/04_PPG_0.1-8.0Hz_261007", "FreqAnalysis 04: PPG 0.1-8.0 Hz, 261007"),
    ],
    "target": [
        ("261109_RRWaveNet_TargetBand/01_RSP_0.1-0.8Hz_261013", "TargetBand 01: RSP target 0.1-0.8 Hz, 261013"),
        ("261109_RRWaveNet_TargetBand/02_RSP_0.1-0.8Hz_261007", "TargetBand 02: RSP target 0.1-0.8 Hz, 261007"),
    ],
    "earlier": [
        ("261013_RRWavNet_STEAM_BIDMC_v1", "Earlier: raw PPG, RSP target 0.1-0.5 Hz, 261013"),
        ("261007_RRWaveNet_NoEvent_v1", "Earlier: raw PPG, RSP target 0.1-0.5 Hz, 261007"),
    ],
}


# ---------------------------------------------------------------------------
# Reading
# ---------------------------------------------------------------------------

def same_subject(a, b) -> bool:
    a, b = str(a).strip(), str(b).strip()
    return a == b or (a.isdigit() and b.isdigit() and int(a) == int(b))


def is_issue(dataset: str, subject) -> bool:
    return any(same_subject(subject, x) for x in EXCLUDED.get(dataset, ()))


def read_minutes(path: Path) -> pd.DataFrame:
    if not path.exists() or path.stat().st_size == 0:
        return pd.DataFrame()
    try:
        return pd.read_csv(path, dtype={"Subject": str}, encoding="utf-8-sig", low_memory=False)
    except Exception as exc:
        print(f"  {path} not read ({exc})")
        return pd.DataFrame()


def model_minutes(frame: pd.DataFrame) -> Tuple[pd.DataFrame, str]:
    """The minutes of the compared model: 261007 the full model (Variant = full), 261013 the final model."""
    if frame.empty:
        return frame, ""
    if "Variant" in frame.columns and (frame["Variant"].astype(str) == "full").any():
        return frame[frame["Variant"].astype(str) == "full"].copy(), "full model (Variant = full)"
    return frame.copy(), "final model (selected weight decay)"


def find_runs(args: argparse.Namespace) -> Tuple[List[dict], List[dict]]:
    found, missing = [], []
    items = [(Path(args.workspace) / rel, label, group) for group in args.groups for rel, label in RUN_GROUPS[group]]
    items += [(Path(root), f"Given: {Path(root).name}", "given") for root in args.roots or []]
    for root, label, group in items:
        ens_path = root / "Summary" / "minutes_ensemble.csv"
        mem_path = root / "Summary" / "minutes_single_models.csv"
        if ens_path.exists() and mem_path.exists():
            found.append({"Run": label, "Group": group, "Folder": root, "ens_path": ens_path, "mem_path": mem_path})
        else:
            missing.append({"Run": label, "Group": group, "Folder": str(root),
                            "Missing": ", ".join(p.name for p in (ens_path, mem_path) if not p.exists())})
    return found, missing


# ---------------------------------------------------------------------------
# Statistics
# ---------------------------------------------------------------------------

def wilcoxon_p(x: np.ndarray, y: np.ndarray) -> float:
    from scipy.stats import wilcoxon
    d = np.asarray(x, float) - np.asarray(y, float)
    if d.size < 2 or np.allclose(d, 0):
        return math.nan
    try:
        return float(wilcoxon(x, y).pvalue)
    except ValueError:
        return math.nan


def rank_biserial(d: np.ndarray) -> float:
    """Matched-pairs rank-biserial r = (sum of ranks of positive - negative differences) / total rank sum."""
    from scipy.stats import rankdata
    d = np.asarray(d, float)
    d = d[np.isfinite(d) & (np.abs(d) > 1e-12)]
    if d.size == 0:
        return math.nan
    ranks = rankdata(np.abs(d))
    return float((ranks[d > 0].sum() - ranks[d < 0].sum()) / ranks.sum())


def bh(p: Sequence[float]) -> List[float]:
    p = np.asarray(p, float)
    out = np.full(p.size, np.nan)
    ok = np.flatnonzero(np.isfinite(p))
    if ok.size:
        order = ok[np.argsort(p[ok])]
        ranked = p[order] * ok.size / (np.arange(ok.size) + 1)
        out[order] = np.minimum(1.0, np.minimum.accumulate(ranked[::-1])[::-1])
    return out.tolist()


def describe(values: np.ndarray) -> Dict[str, float]:
    v = np.asarray(values, float)
    v = v[np.isfinite(v)]
    if not v.size:
        return {"Mean": math.nan, "SD": math.nan, "Median": math.nan, "Q1": math.nan, "Q3": math.nan}
    return {"Mean": float(v.mean()), "SD": float(v.std(ddof=1)) if v.size > 1 else math.nan,
            "Median": float(np.median(v)), "Q1": float(np.percentile(v, 25)), "Q3": float(np.percentile(v, 75))}


def text(stats: Dict[str, float], digits: int) -> str:
    f = f"{{:.{digits}f}}"
    return (f"{f.format(stats['Mean'])} ± {f.format(stats['SD'])} "
            f"({f.format(stats['Median'])} [{f.format(stats['Q1'])}, {f.format(stats['Q3'])}])")


# ---------------------------------------------------------------------------
# Per subject, per test
# ---------------------------------------------------------------------------

def per_subject(ens: pd.DataFrame, mem: pd.DataFrame) -> pd.DataFrame:
    """Every subject: the ensemble, every member and the member average (FFT MAE, PCC)."""
    rows = []
    members = sorted(int(m) for m in mem["Member"].dropna().unique()) if "Member" in mem else []
    for subject, e in ens.groupby("Subject"):
        m = mem[mem["Subject"] == subject]
        row = {"Subject": subject, "Minutes": int(len(e)), "Members": int(m["Member"].nunique()) if len(m) else 0}
        for name, column, _ in METRICS:
            row[f"{name}_Ensemble"] = float(pd.to_numeric(e[column], errors="coerce").mean())
            values = []
            for k in members:
                part = pd.to_numeric(m.loc[m["Member"] == k, column], errors="coerce")
                value = float(part.mean()) if len(part) else math.nan
                row[f"{name}_m{k}"] = value
                values.append(value)
            finite = [v for v in values if np.isfinite(v)]
            row[f"{name}_Single_Average"] = float(np.mean(finite)) if finite else math.nan
        rows.append(row)
    return pd.DataFrame(rows)


def minute_p(ens: pd.DataFrame, mem: pd.DataFrame, column: str, subjects: Sequence[str]) -> Tuple[float, int]:
    """Supplementary: Wilcoxon over the paired minutes (members averaged per minute)."""
    if not set(MINUTE_KEY) <= set(ens.columns) or not set(MINUTE_KEY) <= set(mem.columns):
        return math.nan, 0
    e = ens[ens["Subject"].isin(subjects)].groupby(MINUTE_KEY)[column].mean()
    m = mem[mem["Subject"].isin(subjects)].groupby(MINUTE_KEY)[column].mean()
    keys = e.index.intersection(m.index)
    x, y = e.loc[keys].to_numpy(float), m.loc[keys].to_numpy(float)
    ok = np.isfinite(x) & np.isfinite(y)
    return wilcoxon_p(x[ok], y[ok]), int(ok.sum())


def test_rows(table: pd.DataFrame, ens: pd.DataFrame, mem: pd.DataFrame, base: dict) -> List[dict]:
    rows = []
    members = sorted(c.split("_m")[-1] for c in table.columns if c.startswith("FFT_MAE_m"))
    for name, column, lower_better in METRICS:
        comparisons = [("Single-model average", f"{name}_Single_Average")] + [(f"m{k}", f"{name}_m{k}") for k in members]
        for label, col in comparisons:
            if col not in table:
                continue
            e, s = table[f"{name}_Ensemble"].to_numpy(float), table[col].to_numpy(float)
            ok = np.isfinite(e) & np.isfinite(s)
            e, s = e[ok], s[ok]
            gain = (s - e) if lower_better else (e - s)  # positive = the ensemble is better
            p_min, n_min = (minute_p(ens, mem, column, table.loc[ok, "Subject"].tolist())
                            if label == "Single-model average" else (math.nan, 0))
            rows.append({**base, "Metric": name, "Compared_With": label, "Subjects": int(ok.sum()),
                         "Ensemble_Mean": float(np.mean(e)) if e.size else math.nan,
                         "Ensemble_Median": float(np.median(e)) if e.size else math.nan,
                         "Single_Mean": float(np.mean(s)) if s.size else math.nan,
                         "Single_Median": float(np.median(s)) if s.size else math.nan,
                         "Gain_Median": float(np.median(gain)) if gain.size else math.nan,
                         "Gain_Mean": float(np.mean(gain)) if gain.size else math.nan,
                         "Ensemble_Better": int(np.sum(gain > 1e-12)), "Ensemble_Worse": int(np.sum(gain < -1e-12)),
                         "Tied": int(np.sum(np.abs(gain) <= 1e-12)),
                         "Wilcoxon_p": wilcoxon_p(e, s), "Rank_Biserial_r": rank_biserial(gain),
                         "Minutes_Paired": n_min, "Minute_Wilcoxon_p": p_min,
                         "Ensemble_Text": text(describe(e), 2 if name == "FFT_MAE" else 3),
                         "Single_Text": text(describe(s), 2 if name == "FFT_MAE" else 3)})
    return rows


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="RRWaveNet: ensemble vs single models, statistics per run")
    parser.add_argument("--groups", nargs="+", choices=list(RUN_GROUPS), default=list(RUN_GROUPS))
    parser.add_argument("--roots", nargs="*", type=Path, default=None, help="further run folders (with Summary\\)")
    parser.add_argument("--workspace", type=Path, default=WORKSPACE)
    parser.add_argument("--output-root", type=Path, default=None)
    parser.add_argument("--alpha", type=float, default=ALPHA)
    args = parser.parse_args(argv)
    if args.output_root is None:
        args.output_root = Path(args.workspace) / NAME
    return args


def main(argv: Optional[Sequence[str]] = None) -> None:
    args = parse_args(argv)
    out_dir = Path(args.output_root)
    out_dir.mkdir(parents=True, exist_ok=True)
    runs, missing = find_runs(args)
    print(f"{NAME}: {len(runs)} run(s) found, {len(missing)} not found\nOutput: {out_dir}")
    tests, subjects, run_rows = [], [], []
    for run in runs:
        ens_all, what = model_minutes(read_minutes(run["ens_path"]))
        mem_all, _ = model_minutes(read_minutes(run["mem_path"]))
        print(f"\n{run['Run']} ({what}): {run['Folder']}")
        run_rows.append({"Run": run["Run"], "Group": run["Group"], "Folder": str(run["Folder"]), "Model": what,
                         "Ensemble_Minutes": int(len(ens_all)), "Single_Model_Minutes": int(len(mem_all))})
        if ens_all.empty or mem_all.empty or "Member" not in mem_all:
            print("  no ensemble or single-model minutes")
            continue
        for column in ("Dataset", "Window_Config"):
            for frame in (ens_all, mem_all):
                if column not in frame:
                    frame[column] = ""
        keys = ens_all[["Dataset", "Window_Config"]].drop_duplicates()
        keys["_d"] = keys["Dataset"].map({d: i for i, d in enumerate(DATASET_ORDER)}).fillna(99)
        for _, key in keys.sort_values(["_d", "Window_Config"]).iterrows():
            dataset, window = key["Dataset"], key["Window_Config"]
            ens = ens_all[(ens_all["Dataset"] == dataset) & (ens_all["Window_Config"] == window)]
            mem = mem_all[(mem_all["Dataset"] == dataset) & (mem_all["Window_Config"] == window)]
            table = per_subject(ens, mem)
            if table.empty:
                continue
            table["Issue_Subject"] = [is_issue(dataset, s) for s in table["Subject"]]
            base = {"Run": run["Run"], "Dataset": dataset, "Window_Config": window}
            subjects.append(table.assign(**base))
            for subject_set, part in (("All subjects", table), ("Issue subjects left out", table[~table["Issue_Subject"]])):
                if len(part) >= 2:
                    tests += test_rows(part.reset_index(drop=True), ens, mem, {**base, "Subject_Set": subject_set})
            primary = [t for t in tests if t["Run"] == run["Run"] and t["Dataset"] == dataset
                       and t["Window_Config"] == window and t["Subject_Set"] == "Issue subjects left out"
                       and t["Compared_With"] == "Single-model average"]
            for t in primary:
                print(f"  {dataset:<9} {window:<16} {t['Metric']:<8} ensemble {t['Ensemble_Median']:.3f} vs single "
                      f"{t['Single_Median']:.3f}; ensemble better {t['Ensemble_Better']}/{t['Subjects']}, "
                      f"p = {t['Wilcoxon_p']:.3g}")
    tests_frame = pd.DataFrame(tests)
    if not tests_frame.empty:
        tests_frame["BH_p"] = np.nan
        for _, idx in tests_frame.groupby(["Subject_Set", "Metric"]).groups.items():
            tests_frame.loc[idx, "BH_p"] = bh(tests_frame.loc[idx, "Wilcoxon_p"].tolist())
        tests_frame["Significant_BH"] = tests_frame["BH_p"] < args.alpha
        tests_frame["Result"] = [
            ("ensemble better" if g > 1e-12 else "single better" if g < -1e-12 else "no difference")
            + (" *" if s else " (n.s.)")
            for g, s in zip(tests_frame["Gain_Median"].fillna(0), tests_frame["Significant_BH"])]
        first = ["Run", "Dataset", "Window_Config", "Subject_Set", "Metric", "Compared_With", "Subjects",
                 "Ensemble_Text", "Single_Text", "Gain_Median", "Ensemble_Better", "Ensemble_Worse", "Tied",
                 "Wilcoxon_p", "BH_p", "Significant_BH", "Result", "Rank_Biserial_r"]
        tests_frame = tests_frame[first + [c for c in tests_frame if c not in first]]
    summary = (tests_frame[tests_frame["Compared_With"] == "Single-model average"].drop(columns="Compared_With")
               .reset_index(drop=True) if not tests_frame.empty else pd.DataFrame())
    per_subject_frame = pd.concat(subjects, ignore_index=True) if subjects else pd.DataFrame()
    if not per_subject_frame.empty:
        lead = ["Run", "Dataset", "Window_Config", "Subject", "Issue_Subject", "Minutes", "Members"]
        per_subject_frame = per_subject_frame[lead + [c for c in per_subject_frame if c not in lead]]
    notes = pd.DataFrame([
        ("Compared", "the ensemble of 3 models vs its single models (members m0-m2) on the same scored minutes; "
                     "261007: the full model (Variant = full); 261013: the final model (selected weight decay)"),
        ("Per subject", "FFT MAE = mean |FFT rate - reference rate| over the subject's minutes; Wave PCC = mean "
                        "waveform r; single-model average = mean of the members' values"),
        ("Gain", "single - ensemble for FFT MAE, ensemble - single for PCC: positive = the ensemble is better"),
        ("Tests", "Wilcoxon signed-rank over the paired subjects (two-sided); BH over all tests of one subject set "
                  f"and metric; * = BH p < {args.alpha:g}; rank-biserial r: effect size (-1..1, positive = ensemble "
                  "better for FFT MAE gain)"),
        ("Minute level", "supplementary: Wilcoxon over the paired minutes (members averaged per minute); the minutes "
                         "of one subject are not independent"),
        ("Issue subjects", "; ".join(f"{d} {', '.join(s)}" for d, s in EXCLUDED.items())),
        ("Not found", "; ".join(f"{m['Run']} ({m['Missing']})" for m in missing) or "-"),
    ], columns=["Item", "Description"])
    sheets = {"Summary": summary, "Tests": tests_frame, "Per_Subject": per_subject_frame,
              "Runs": pd.DataFrame(run_rows + [{**m, "Model": "not found"} for m in missing]), "Notes": notes}
    stamp = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")

    def write(path: Path, writer) -> Path:
        try:
            writer(path)
            return path
        except PermissionError:
            alt = path.with_name(f"{path.stem}_{stamp}{path.suffix}")
            writer(alt)
            print(f"  [Locked] {path.name} is held by another program; saved as {alt.name}")
            return alt

    def xlsx(path: Path) -> None:
        with pd.ExcelWriter(path, engine="openpyxl") as w:
            for name, frame in sheets.items():
                frame.to_excel(w, sheet_name=name, index=False)

    for name in ("Summary", "Tests", "Per_Subject"):
        write(out_dir / f"{name}.csv", lambda p, f=sheets[name]: f.to_csv(p, index=False, encoding="utf-8-sig"))
    path = write(out_dir / f"{NAME}.xlsx", xlsx)
    if missing:
        print("\nNot found: " + "; ".join(m["Run"] for m in missing))
    if not summary.empty:
        pd.set_option("display.width", 240)
        show = summary[summary["Subject_Set"] == "Issue subjects left out"]
        print("\nEnsemble vs single-model average (issue subjects left out; BH within metric):")
        print(show[["Run", "Dataset", "Window_Config", "Metric", "Subjects", "Ensemble_Text", "Single_Text",
                    "Ensemble_Better", "BH_p", "Result"]].to_string(index=False))
    print(f"\nSaved: {path}")


if __name__ == "__main__":
    main()
