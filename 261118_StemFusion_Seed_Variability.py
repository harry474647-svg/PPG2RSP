# -*- coding: utf-8 -*-
r"""261118_StemFusion_Seed_Variability: how much the subject MAE moves when only the random seed changes (members
m0, m1, m2 of the same model), and which stem fusion changes (Fusion O m0 vs Fusion X m0) are larger than that.

Why: the stem fusion comparison uses one model per fold (member m0). Members m0-m2 of a fold have the same
architecture, data and training settings and differ only in the seed (initial weights, batch order, jitter,
dropout), so their differences show how far a subject MAE moves by chance. A change between Fusion O and Fusion X
inside that range cannot be told apart from a seed effect.

Models (stored outputs, nothing is trained; the files of 261118_StemFusion_Threshold_Groups_Waveforms, loaded as a
module, with members m0, m1, m2)
  CapnoBase      Fusion O: 261007 full m0-m2;  Fusion X: 261007 no_stem_fusion m0-m2
  BIDMC, STEAM2  Fusion O: 261013 m0-m2 of the weight decay selected in the fold (WD_Selection);
                 Fusion X: 261028_Stem_ablation_noise_v1 no_stem_fusion, m0 only (m1, m2 are used when present)
GT and scored minutes: the rescored GT of 261118 (CapnoBase peak detection on the capnogram, BIDMC ann1, STEAM2
stored GT). Every member of a subject is scored on the same minutes (scored and present in every member table).

Measures
  Minute level   spread of the stored rate over the members of a model (max - min) in every minute; share of minutes
                 with a spread > 1 breath/min (all minutes, and the scored minutes)
  Subject level  MAE of every member; Seed_Range = max - min of the member MAEs of a model;
                 seed pairs = |MAE m_a - MAE m_b| for every pair of members of the same model (3 pairs per model
                 with 3 members)
  Seed noise     per dataset, the quantiles (50, 75, 90, 95 %, max) of the seed pairs of all subjects (Fusion O pairs,
                 and Fusion X pairs where X has more than one member); T = the --seed-quantile quantile (default 95 %):
                 95 % of the MAE differences caused by the seed alone are <= T
  Stem fusion    Diff = MAE O m0 - MAE X m0 (the subject values of the 261118 figures)
                 within seed noise   |Diff| <= T of the dataset                                 -> gray
                 improved / worsened Diff < -T / Diff > T                                       -> red / blue
                 Beyond_Own_Seed_Range: |Diff| > the subject's own Seed_Range of Fusion O (and of Fusion X when
                 it has members) - a stricter check per subject (3 seeds give a rough range)

Outputs (D:\PPG2RSP_RRWaveNet_Inspired\261118_StemFusion_Seed_Variability)
  261118_StemFusion_Seed_Variability.xlsx
    Seed_Noise     per dataset (and all): seed pair quantiles, T, minute shares with spread > 1 breath/min, number of
                   subjects improved / worsened / within seed noise
    Subjects       every subject: MAE of every member of O and X, Seed_Range, Diff, class, Beyond_Own_Seed_Range
    Seed_Pairs     every seed pair: dataset, subject, model, members, MAE a, MAE b, |difference|
    Minutes        every minute: GT, scored, rate of every member
    Missing, Notes
  StemFusion_MAE_Slope_SeedNoise.png / .pdf   MAE Without -> With Fusion of every subject (m0), red = improved beyond
                                              the seed noise of its dataset, blue = worsened beyond it, gray = within

Run: python 261118_StemFusion_Seed_Variability.py     (Spyder: runfile(...), no options needed;
     261118_StemFusion_Threshold_Groups_Waveforms.py must be in the same folder, or give --script-261118)
Options: --seed-quantile (0.95), --datasets, --keep-issue-subjects, --ann1-nogap, --root-261007, --root-261013,
--root-noise, --output-root, --script-261118, --script-261007, --script-261013, --capnobase-path, --bidmc-path,
--steam2-ppg-dir, --steam2-rsp-dir
"""

import argparse
import importlib.util
import itertools
import math
import sys
from pathlib import Path
from typing import Dict, Optional, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D

# ---------------------------------------------------------------------------
# Settings
# ---------------------------------------------------------------------------

NAME = "261118_StemFusion_Seed_Variability"
HERE = Path(__file__).resolve().parent
BASE_SCRIPT = "261118_StemFusion_Threshold_Groups_Waveforms.py"
WORKSPACE = Path(r"D:\PPG2RSP_RRWaveNet_Inspired")
OUTPUT_ROOT = WORKSPACE / NAME
MEMBERS = (0, 1, 2)
SEED_QUANTILE = 0.95
QUANTILES = (0.5, 0.75, 0.9, 0.95)
SPREAD_BPM = 1.0  # minute level: members more than 1 breath/min apart
DATASETS = ("CapnoBase", "BIDMC", "STEAM2")
RED, BLUE, GRAY = "#ed665a", "#559ed5", "#b4b2ad"  # red = improved with stem fusion, blue = worsened
TARGET, INK, MUTED, GRID = "#a8823c", "#1f1f1f", "#6b6a66", "#e6e5e1"


def load_base(given: Optional[Path]):
    """261118_StemFusion_Threshold_Groups_Waveforms as a module (its stored files, GT and minute rules)."""
    for path in ([Path(given)] if given else []) + [HERE / BASE_SCRIPT, Path.cwd() / BASE_SCRIPT]:
        if path.is_file():
            spec = importlib.util.spec_from_file_location("seed_var_base_261118", str(path))
            module = importlib.util.module_from_spec(spec)
            sys.modules[spec.name] = module
            spec.loader.exec_module(module)
            return module
    raise FileNotFoundError(f"{BASE_SCRIPT} not found next to this script ({HERE}); give it with --script-261118")


# ---------------------------------------------------------------------------
# Stored minute tables of every member
# ---------------------------------------------------------------------------

def member_table(base, args, mod, dataset: str, model: str, subject: str, member: int) -> Optional[Path]:
    """Minute table of one member, searched as 261118 searches m0: 261028_Stem_ablation_noise_v1 first, then 261007
    (CapnoBase) or the weight decay selected in 261013 (BIDMC, STEAM2, Fusion O)."""
    tag = mod.safe_tag(subject)
    variant = "full" if model == "O" else "no_stem_fusion"
    roots = [Path(args.root_noise) / variant / base.WINDOW / dataset]
    if base.SOURCE[dataset] == "261007":
        roots.append(Path(args.root_261007) / variant / base.WINDOW / dataset)
    elif model == "O":
        record = base.read_json(Path(args.root_261013) / "WD_Selection" / base.WINDOW / dataset
                                / f"{tag}_selection.json") or {}
        if record.get("Selected_Variant"):
            roots.append(Path(args.root_261013) / record["Selected_Variant"] / base.WINDOW / dataset)
    for root in roots:
        path = root / "Fold_Summaries" / f"{tag}_m{member}_RR60s_windows.csv"
        if path.exists():
            return path
    return None


def read_rates(path: Path) -> pd.DataFrame:
    t = pd.read_csv(path, dtype={"Subject": str}, encoding="utf-8-sig")
    t["Segment_ID"] = t["Segment_ID"].astype(int)
    t["RR_Window_Start_Sec"] = t["RR_Window_Start_Sec"].astype(float).round(3)
    return t.set_index(["Segment_ID", "RR_Window_Start_Sec"])


def gt_minutes(base, args, mod, dataset: str, record: dict, table: pd.DataFrame,
               ann1: Optional[np.ndarray]) -> pd.DataFrame:
    """Rescored GT of every minute of the grid (the rules of 261118)."""
    fs = mod.TARGET_FS
    rows = []
    for (segment_id, t0), o in table.iterrows():
        ref = record["references"].get(int(segment_id))
        if ref is None:
            continue
        t1 = float(o["RR_Window_End_Sec"])
        a, b = int(round(float(t0) * fs)), int(round(t1 * fs))
        if dataset == "STEAM2":
            valid, reason, count = bool(o["Ref_Valid"]), str(o.get("Ref_Exclusion_Reason", "") or ""), int(o["GT_Breath_Count"])
        else:
            breaths, label_based = base.gt_breaths(dataset, mod, ref, ann1)
            valid, reason, count = base.minute_gt(mod, ref, a, b, breaths, label_based)
            if dataset == "BIDMC" and args.ann1_nogap and valid and any(
                    lo < b and hi > a for lo, hi in base.gap_spans(breaths, len(ref["signal"]))):
                valid, reason = False, "ref_ann1_gap"
        rows.append({"Segment_ID": int(segment_id), "Start_Sec": float(t0), "End_Sec": t1, "Scored": valid,
                     "Reason": "" if valid else reason, "GT_Count": count})
    return pd.DataFrame(rows).set_index(["Segment_ID", "Start_Sec"]) if rows else pd.DataFrame()


def spread_share(rates: Dict[int, pd.Series], keys) -> float:
    if len(rates) < 2 or len(keys) == 0:
        return math.nan
    v = np.column_stack([rates[m].loc[keys].to_numpy(float) for m in sorted(rates)])
    return float(np.mean(v.max(axis=1) - v.min(axis=1) > SPREAD_BPM))


# ---------------------------------------------------------------------------
# Figure
# ---------------------------------------------------------------------------

def draw_slope(subjects: pd.DataFrame, thresholds: Dict[str, float], q: float, path: Path) -> None:
    plt.rcParams.update({"axes.edgecolor": "#7a7975", "axes.linewidth": 1.0})
    d = subjects.dropna(subset=["MAE_X_m0", "MAE_O_m0"]).copy()
    color = {"improved": RED, "worsened": BLUE, "within seed noise": GRAY}
    fig, ax = plt.subplots(figsize=(7.6, 5.8))
    fig.patch.set_facecolor("white")
    for r in sorted(d.itertuples(), key=lambda r: (r.Class != "within seed noise", abs(r.Diff_O_minus_X))):
        gray = r.Class == "within seed noise"
        c = color[r.Class]
        ax.plot([0, 1], [r.MAE_X_m0, r.MAE_O_m0], color=c, lw=1.6, alpha=0.6 if gray else 0.85, zorder=1 if gray else 2,
                solid_capstyle="round")
        ax.plot([0, 1], [r.MAE_X_m0, r.MAE_O_m0], ls="none", marker="o", ms=6.5, color=c, mec="white", mew=0.8,
                zorder=3 if gray else 4)
    ax.axhline(3.0, color=TARGET, lw=1.6, ls=(0, (4, 3)), zorder=1.5)
    top = max(10.0, float(math.ceil(max(d.MAE_X_m0.max(), d.MAE_O_m0.max()))) + 0.5) if len(d) else 10.0
    ax.set_xlim(-0.2, 1.2)
    ax.set_ylim(0, top)
    ax.set_xticks([0, 1])
    ax.set_xticklabels(["Without", "With Fusion"], fontsize=12)
    ax.tick_params(axis="x", length=0, pad=8)
    ax.tick_params(axis="y", labelsize=10.5, length=0)
    ax.yaxis.grid(True, color=GRID, lw=0.9)
    ax.set_axisbelow(True)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    ax.set_ylabel("MAE (bpm)", fontsize=12)
    counts = d["Class"].value_counts()
    sizes = ", ".join(f"{k} {int((d.Dataset == k).sum())}" for k in DATASETS if (d.Dataset == k).any())
    ax.set_title(f"All subjects (n = {len(d)}: {sizes})", fontsize=12, loc="left", color=INK, pad=40)
    seed = " · ".join(f"{k} {thresholds[k]:.2f}" for k in DATASETS if k in thresholds and np.isfinite(thresholds[k]))
    ax.text(0.0, 1.02, f"improved {int(counts.get('improved', 0))} · worsened {int(counts.get('worsened', 0))} · within "
                       f"seed noise {int(counts.get('within seed noise', 0))} · MAE < 3: {int((d.MAE_X_m0 < 3).sum())} → "
                       f"{int((d.MAE_O_m0 < 3).sum())}\nseed noise T (|ΔMAE| between seeds, {q * 100:g}th percentile, "
                       f"bpm): {seed}", transform=ax.transAxes, fontsize=8.4, color=MUTED, va="bottom", linespacing=1.5)
    handles = [Line2D([], [], color=RED, lw=1.6, marker="o", ms=6.5, mec="white"),
               Line2D([], [], color=BLUE, lw=1.6, marker="o", ms=6.5, mec="white"),
               Line2D([], [], color=GRAY, lw=1.6, marker="o", ms=6.5, mec="white"),
               Line2D([], [], color=TARGET, lw=1.6, ls=(0, (4, 3)))]
    labels = ["Improved with stem fusion (beyond seed noise)", "Worsened with stem fusion (beyond seed noise)",
              "Within seed noise (|ΔMAE| ≤ T of its dataset)", "3 bpm target"]
    fig.legend(handles, labels, loc="upper center", ncol=2, frameon=False, fontsize=8.6, bbox_to_anchor=(0.5, 1.0))
    fig.text(0.5, 0.012, "Subject MAE of member m0 per LOSOCV fold (rescored GT; MAE issue subjects excluded)",
             ha="center", fontsize=8.2, color=MUTED)
    fig.subplots_adjust(left=0.1, right=0.97, top=0.75, bottom=0.11)
    fig.savefig(path.with_suffix(".png"), dpi=300)
    fig.savefig(path.with_suffix(".pdf"))
    plt.close(fig)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Subject MAE differences between seeds (members m0-m2) and the stem "
                                            "fusion changes beyond them")
    p.add_argument("--seed-quantile", type=float, default=SEED_QUANTILE,
                   help="quantile of the seed MAE differences used as the noise threshold T (default 0.95)")
    p.add_argument("--datasets", nargs="+", choices=list(DATASETS), default=list(DATASETS))
    p.add_argument("--keep-issue-subjects", action="store_true", help="keep the MAE issue subjects")
    p.add_argument("--ann1-nogap", action="store_true", help="BIDMC: leave out minutes overlapping an ann1 gap")
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


def main(argv: Optional[Sequence[str]] = None) -> None:
    args = parse_args(argv)
    base = load_base(args.script_261118)
    q = float(args.seed_quantile)
    out_root = Path(args.output_root)
    out_root.mkdir(parents=True, exist_ok=True)
    subject_rows, pair_rows, minute_frames, missing, minute_shares = [], [], [], [], []
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
        for subject in sorted(data):
            issue = any(base.same_subject(subject, x) for x in base.EXCLUDED[dataset])
            if issue and not args.keep_issue_subjects:
                continue
            tables = {(m, k): member_table(base, args, mod, dataset, m, subject, k) for m in ("O", "X") for k in MEMBERS}
            if not tables[("O", 0)] or not tables[("X", 0)]:
                missing.append({"Dataset": dataset, "Subject": subject, "Missing": "member m0 of "
                                + " and ".join(f"Fusion {m}" for m in ("O", "X") if not tables[(m, 0)])})
                continue
            frames = {key: read_rates(path) for key, path in tables.items() if path}
            gt = gt_minutes(base, args, mod, dataset, data[subject], frames[("O", 0)], ann1_all.get(subject))
            if gt.empty:
                missing.append({"Dataset": dataset, "Subject": subject, "Missing": "no minute with a reference"})
                continue
            common = gt.index
            for frame in frames.values():
                common = common.intersection(frame.index)
            scored = gt.loc[common]
            scored = scored[scored["Scored"]]
            if scored.empty:
                missing.append({"Dataset": dataset, "Subject": subject, "Missing": "no scored minute in every member"})
                continue
            keys = scored.index
            rates = {key: frame["Pred_Wave_FFT_RR"].astype(float) for key, frame in frames.items()}
            mae = {key: float(np.mean(np.abs(r.loc[keys].to_numpy(float) - scored["GT_Count"].to_numpy(float))))
                   for key, r in rates.items()}
            row = {"Dataset": dataset, "Subject": subject, "Issue_Subject": issue, "Scored_Minutes": int(len(keys))}
            for m in ("O", "X"):
                members = [k for k in MEMBERS if (m, k) in mae]
                for k in MEMBERS:
                    row[f"MAE_{m}_m{k}"] = mae.get((m, k), math.nan)
                row[f"Members_{m}"] = len(members)
                row[f"Seed_Range_{m}"] = (max(mae[(m, k)] for k in members) - min(mae[(m, k)] for k in members)
                                          if len(members) > 1 else math.nan)
                for a, b in itertools.combinations(members, 2):
                    pair_rows.append({"Dataset": dataset, "Subject": subject, "Model": f"Fusion {m}", "Member_a": f"m{a}",
                                      "Member_b": f"m{b}", "MAE_a": mae[(m, a)], "MAE_b": mae[(m, b)],
                                      "Abs_Diff": abs(mae[(m, a)] - mae[(m, b)])})
                share_all = spread_share({k: rates[(m, k)] for k in members}, common)
                share_scored = spread_share({k: rates[(m, k)] for k in members}, keys)
                row[f"Minutes_Spread_gt1_{m}"] = share_scored
                minute_shares.append({"Dataset": dataset, "Model": m, "Subject": subject, "N_All": len(common),
                                      "N_Scored": len(keys), "Share_All": share_all, "Share_Scored": share_scored})
            row["Diff_O_minus_X"] = row["MAE_O_m0"] - row["MAE_X_m0"]
            subject_rows.append(row)
            mf = scored.reset_index().assign(Dataset=dataset, Subject=subject)
            for (m, k), r in rates.items():
                mf[f"Rate_{m}_m{k}"] = r.loc[keys].to_numpy(float)
            minute_frames.append(mf)
            print(f"  {subject}: MAE O m0-m2 " + " ".join(f"{row[f'MAE_O_m{k}']:.2f}" for k in MEMBERS)
                  + " | X " + " ".join(f"{row[f'MAE_X_m{k}']:.2f}" for k in MEMBERS)
                  + f" | O - X (m0) {row['Diff_O_minus_X']:+.2f}", flush=True)
        del data
    subjects = pd.DataFrame(subject_rows)
    pairs = pd.DataFrame(pair_rows, columns=["Dataset", "Subject", "Model", "Member_a", "Member_b", "MAE_a", "MAE_b",
                                             "Abs_Diff"])
    shares = pd.DataFrame(minute_shares)
    thresholds, noise_rows = {}, []
    groups = [(d, pairs[pairs.Dataset == d], subjects[subjects.Dataset == d] if not subjects.empty else subjects)
              for d in args.datasets] + [("All", pairs, subjects)]
    for name, p, s in groups:
        row = {"Dataset": name, "Seed_Pairs": len(p), "Seed_Pairs_O": int((p.Model == "Fusion O").sum()),
               "Seed_Pairs_X": int((p.Model == "Fusion X").sum()), "Subjects": len(s)}
        for qq in QUANTILES:
            row[f"Seed_AbsDiff_Q{int(round(qq * 100))}"] = float(p.Abs_Diff.quantile(qq)) if len(p) else math.nan
        row["Seed_AbsDiff_Max"] = float(p.Abs_Diff.max()) if len(p) else math.nan
        row["T"] = float(p.Abs_Diff.quantile(q)) if len(p) else math.nan
        for m in ("O", "X"):
            sh = shares[(shares.Model == m) & ((shares.Dataset == name) if name != "All" else True)] if not shares.empty else shares
            ok = sh.dropna(subset=["Share_All"]) if not sh.empty else sh
            row[f"Minutes_Spread_gt1_All_{m}"] = (float((ok.Share_All * ok.N_All).sum() / ok.N_All.sum())
                                                  if len(ok) and ok.N_All.sum() else math.nan)
            ok = sh.dropna(subset=["Share_Scored"]) if not sh.empty else sh
            row[f"Minutes_Spread_gt1_Scored_{m}"] = (float((ok.Share_Scored * ok.N_Scored).sum() / ok.N_Scored.sum())
                                                     if len(ok) and ok.N_Scored.sum() else math.nan)
        noise_rows.append(row)
        if name != "All":
            thresholds[name] = row["T"]
    if not subjects.empty:
        t = subjects["Dataset"].map(thresholds).astype(float)
        diff = subjects["Diff_O_minus_X"]
        subjects["Seed_Noise_T"] = t
        subjects["Class"] = np.where(diff < -t, "improved", np.where(diff > t, "worsened", "within seed noise"))
        own = subjects[["Seed_Range_O", "Seed_Range_X"]].max(axis=1)
        subjects["Beyond_Own_Seed_Range"] = np.where(own.notna(), diff.abs() > own, np.nan)
        for row in noise_rows:
            s = subjects if row["Dataset"] == "All" else subjects[subjects.Dataset == row["Dataset"]]
            for c in ("improved", "worsened", "within seed noise"):
                row[f"N_{c.replace(' ', '_')}"] = int((s["Class"] == c).sum())
            row["N_Beyond_Own_Seed_Range"] = int((s["Beyond_Own_Seed_Range"] == True).sum())  # noqa: E712
    noise = pd.DataFrame(noise_rows)
    print(f"\nSeed noise: |subject MAE difference| between members of the same model (T = {q * 100:g}th percentile)")
    for r in noise.to_dict("records"):
        print(f"  {r['Dataset']}: {r['Seed_Pairs']} pairs; median {r['Seed_AbsDiff_Q50']:.2f}, 75% "
              f"{r['Seed_AbsDiff_Q75']:.2f}, 90% {r['Seed_AbsDiff_Q90']:.2f}, 95% {r['Seed_AbsDiff_Q95']:.2f}, max "
              f"{r['Seed_AbsDiff_Max']:.2f} bpm; T = {r['T']:.2f}; minutes with members > 1 bpm apart: Fusion O "
              f"{100 * r['Minutes_Spread_gt1_All_O']:.1f}% (scored {100 * r['Minutes_Spread_gt1_Scored_O']:.1f}%)"
              + (f", Fusion X {100 * r['Minutes_Spread_gt1_All_X']:.1f}%" if np.isfinite(r["Minutes_Spread_gt1_All_X"])
                 else ", Fusion X: one member only")
              + (f"; improved {r.get('N_improved', 0)}, worsened {r.get('N_worsened', 0)}, within seed noise "
                 f"{r.get('N_within_seed_noise', 0)}" if "N_improved" in r else ""))
    if not subjects.empty:
        print("\nBeyond the seed noise of the dataset (O - X of m0)")
        for c in ("improved", "worsened"):
            s = subjects[subjects.Class == c].sort_values("Diff_O_minus_X")
            print(f"  {c} ({len(s)}): " + (", ".join(
                f"{r.Dataset} {r.Subject} ({r.MAE_X_m0:.2f} -> {r.MAE_O_m0:.2f}"
                + ("" if r.Beyond_Own_Seed_Range in (True, 1.0) else ", within its own seed range") + ")"
                for r in s.itertuples()) or "none"))
        draw_slope(subjects, thresholds, q, out_root / "StemFusion_MAE_Slope_SeedNoise")
    notes = pd.DataFrame([
        ("Members", "m0, m1, m2 of a fold: same model, data and settings, different seed (initial weights, batch order, "
                    "jitter, dropout)"),
        ("Models", "CapnoBase: 261007 full / no_stem_fusion m0-m2; BIDMC, STEAM2: Fusion O 261013 m0-m2 of the selected "
                   "weight decay, Fusion X 261028_Stem_ablation_noise_v1 m0 (m1, m2 when present)"),
        ("GT", "rescored GT of 261118 (CapnoBase peak detection, BIDMC ann1"
               + (" without ann1 gap minutes" if args.ann1_nogap else "") + ", STEAM2 stored GT); every member of a "
               "subject on the same scored minutes"),
        ("Seed noise", f"|MAE m_a - MAE m_b| of every pair of members of the same model; T = {q * 100:g}th percentile "
                       "per dataset (pairs of Fusion O, and of Fusion X where it has more than one member)"),
        ("Classes", "Diff = MAE O m0 - MAE X m0; improved Diff < -T, worsened Diff > T, within seed noise otherwise; "
                    "Beyond_Own_Seed_Range = |Diff| larger than the subject's own member range (max - min)"),
        ("Minute spread", f"share of minutes in which the members of a model are more than {SPREAD_BPM:g} breath/min "
                          "apart (stored rates; all minutes present in every member, and the scored minutes)"),
        ("Subjects", "MAE issue subjects (CapnoBase 0115, BIDMC 27, STEAM2 12_YJJ, 22_LMS, 26_CKK) "
                     + ("kept" if args.keep_issue_subjects else "left out")),
    ], columns=["Item", "Description"])
    path = out_root / f"{NAME}.xlsx"
    with pd.ExcelWriter(path, engine="openpyxl") as writer:
        noise.round(4).to_excel(writer, sheet_name="Seed_Noise", index=False)
        (subjects.sort_values(["Dataset", "Diff_O_minus_X"]).round(4) if not subjects.empty else subjects).to_excel(
            writer, sheet_name="Subjects", index=False)
        pairs.round(4).to_excel(writer, sheet_name="Seed_Pairs", index=False)
        (pd.concat(minute_frames, ignore_index=True).round(4) if minute_frames else pd.DataFrame()).to_excel(
            writer, sheet_name="Minutes", index=False)
        pd.DataFrame(missing, columns=["Dataset", "Subject", "Missing"]).to_excel(writer, sheet_name="Missing", index=False)
        notes.to_excel(writer, sheet_name="Notes", index=False)
    print(f"\nTable: {path}\nFigure: {out_root / 'StemFusion_MAE_Slope_SeedNoise.png'}")


if __name__ == "__main__":
    main()
