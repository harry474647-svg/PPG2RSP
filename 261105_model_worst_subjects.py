r"""261105_model_worst_subjects: for every model (RespDiff, the deep corr-encoder,
RRWaveNet) the subjects on which it performs worst, with the 30 s and 60 s
window where it is worst drawn (stored test results of every subject; nothing
is trained).

Score of a model in a window: PCC with the GT - 0.1 x min(rate AE, 5) (rate
AE of 1 breath/min costs 0.1 of PCC; as in 261105_Model_Strength_Analysis).
Subjects: per window size, the mean score of the model over the subject's
non-overlapping windows (from 0 s); ranked by the mean over the sizes, lowest
first; the --top lowest are listed (pooled over the datasets, or per dataset
with --per-dataset). Beside every listed subject: the model's mean rate AE,
PCC, cycle coverage and the same for the other models, and
  Deficit   = the model's score - the best score of the other models
  Pattern   "model-specific" when Deficit <= -0.2 (the others do clearly
            better on the subject), "others worse" when Deficit >= +0.2 (the
            model is low but still clearly the best), otherwise "similar to
            others"
Windows: per listed subject and size, the window (start every --step s) with
the model's lowest score.

Waveforms, alignment, windows and figures are those of
261105_model_discussion_performance (imported): RespDiff and the
corr-encoder as stored, RRWaveNet as 261104 (CapnoBase: 261007, ensemble of
the full model; BIDMC / STEAM2: 261013, ensemble of the weight decay selected
per fold; --rrwavenet-member for one member) with its sign / lag undone and
low-pass 1 Hz, every model shifted <= half a breath and, if needed, inverted
once per recording to its best r with the GT; rate = FFT peak of the window,
GT rate from the breaths in the window; cycles by 261030's detector, matched
1:1.
Subjects need RRWaveNet and at least one other model (as in that script).

Outputs (default D:\PPG2RSP_RRWaveNet_Inspired\261105_model_worst_subjects)
  worst_subjects.xlsx        one sheet per model (ranked subjects), Windows,
                             Notes
  worst_{model}.pdf, worst_{model}\{rank}_{dataset}_{subject}_{size}s.png
                             one row per model in the window, as in the
                             discussion figures

Run:          python 261105_model_worst_subjects.py
Options:      --top 10, --per-dataset, --sizes 30 60, --step 5, result folders
              (--respdiff-root, --correncoder-root, --rrwavenet-261007-root,
              --rrwavenet-261013-root) and data files as in
              261105_model_discussion_performance
"""

import argparse
import importlib.util
import sys
from pathlib import Path
from typing import Optional, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.backends.backend_pdf import PdfPages

HERE = Path(__file__).resolve().parent


def load_script(name: str, filename: str):
    spec = importlib.util.spec_from_file_location(name, HERE / filename)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


mdp = load_script("discussion_261105", "261105_model_discussion_performance.py")
ov = mdp.ov
MODELS = mdp.MODELS
SHORT = {"RespDiff": "RespDiff", "Deep corr-encoder": "CorrEncoder", "RRWaveNet": "RRWaveNet"}
PROJECT_ROOT = Path(r"D:\PPG2RSP_RRWaveNet_Inspired\261105_model_worst_subjects")
AE_WEIGHT, AE_CAP = 0.1, 5.0
SPECIFIC_DEFICIT = -0.2
TOP = 10


def add_scores(win: pd.DataFrame) -> pd.DataFrame:
    win = win.copy()
    for m in MODELS:
        if f"{m}_AE" in win:
            win[f"{m}_Score"] = win[f"{m}_PCC"] - AE_WEIGHT * win[f"{m}_AE"].clip(upper=AE_CAP)
    return win


def subject_table(win: pd.DataFrame, model: str, sizes: Sequence[int]) -> pd.DataFrame:
    grid = win[win.Grid & win[f"{model}_Score"].notna()]
    rows = []
    for (ds, s), d in grid.groupby(["Dataset", "Subject"]):
        row = {"Dataset": ds, "Subject": s, "Windows": len(d)}
        scores = []
        for size in sizes:
            part = d[d.Size_Sec == size]
            sc = part[f"{model}_Score"].mean() if len(part) else np.nan
            row[f"Score_{size}s"] = sc
            scores.append(sc)
        row["Score"] = np.nanmean(scores)
        others = []
        for m in MODELS:
            if f"{m}_AE" not in d or d[f"{m}_AE"].isna().all():
                continue
            ok = d[f"{m}_AE"].notna()
            row[f"{m}_AE"] = d.loc[ok, f"{m}_AE"].mean()
            row[f"{m}_PCC"] = d.loc[ok, f"{m}_PCC"].mean()
            row[f"{m}_Coverage"] = d.loc[ok, f"{m}_Matched"].sum() / max(d.loc[ok, "GT_Cycles"].sum(), 1)
            row[f"{m}_Score"] = np.nanmean([d.loc[ok & (d.Size_Sec == z), f"{m}_Score"].mean() for z in sizes])
            if m != model:
                others.append(row[f"{m}_Score"])
        row["GT_RR_median"] = d.GT_RR.median()
        row["Others_Best_Score"] = max(others) if others else np.nan
        row["Deficit"] = row["Score"] - row["Others_Best_Score"]
        deficit = row["Deficit"]
        row["Pattern"] = ("model-specific" if np.isfinite(deficit) and deficit <= SPECIFIC_DEFICIT
                          else "others worse" if np.isfinite(deficit) and deficit >= -SPECIFIC_DEFICIT
                          else "similar to others")
        rows.append(row)
    return pd.DataFrame(rows)


def main(argv: Optional[Sequence[str]] = None) -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--top", type=int, default=TOP, help="Subjects listed per model (per dataset with --per-dataset).")
    p.add_argument("--per-dataset", action="store_true", help="Rank within every dataset instead of pooled.")
    p.add_argument("--output-root", type=Path, default=PROJECT_ROOT)
    p.add_argument("--no-png", action="store_true")
    own, rest = p.parse_known_args(argv)
    args = mdp.parse_args(rest)
    for key, path in (("capnobase", args.capnobase_path), ("bidmc", args.bidmc_path)):
        if path:
            ov.DATASET_CONFIGS[key]["data_path"] = str(path)
    if args.steam2_ppg_dir:
        ov.DATASET_CONFIGS["steam2"]["ppg_dir"] = str(args.steam2_ppg_dir)
    if args.steam2_rsp_dir:
        ov.DATASET_CONFIGS["steam2"]["rsp_dir"] = str(args.steam2_rsp_dir)
    out = Path(own.output_root)
    out.mkdir(parents=True, exist_ok=True)
    store: dict = {}
    rows = []
    for dataset_key in [k for k in ov.DATASET_ORDER if k in args.datasets]:
        rows += mdp.dataset_windows(dataset_key, args, store)
    win = pd.DataFrame(rows)
    if win.empty:
        print("No subject with RRWaveNet and another model was found (check the result folders).")
        return
    win = add_scores(win)
    sheets, chosen = {}, []
    for model in MODELS:
        if f"{model}_Score" not in win or win[f"{model}_Score"].isna().all():
            continue
        table = subject_table(win, model, args.sizes)
        groups = table.groupby("Dataset") if own.per_dataset else [("All", table)]
        listed = []
        for _, g in groups:
            g = g.sort_values("Score").copy()
            g["Rank"] = np.arange(1, len(g) + 1)
            listed.append(g)
        table = pd.concat(listed, ignore_index=True)
        table["Listed"] = table.Rank <= own.top
        sheets[SHORT[model]] = table
        listed_rows = table[table.Listed].to_dict("records")
        for r in listed_rows:
            for size in args.sizes:
                part = win[(win.Dataset == r["Dataset"]) & (win.Subject == r["Subject"]) & (win.Size_Sec == size)
                           & win[f"{model}_Score"].notna()]
                if part.empty:
                    continue
                worst = part.loc[part[f"{model}_Score"].idxmin()]
                chosen.append({"Model": model, "Rank": int(r["Rank"]), "Pattern": r["Pattern"], **worst.to_dict()})
        print(f"\n{model}: lowest score first ({'per dataset' if own.per_dataset else 'pooled'})")
        for r in listed_rows:
            others = ", ".join(f"{SHORT[m]} {r[f'{m}_Score']:.2f}" for m in MODELS
                               if m != model and np.isfinite(r.get(f"{m}_Score", np.nan)))
            print(f"   #{int(r['Rank']):2d} {r['Dataset']} {r['Subject']}: score {r['Score']:.2f} (AE "
                  f"{r[f'{model}_AE']:.2f}, PCC {r[f'{model}_PCC']:.2f}); others {others}; {r['Pattern']}")
    windows = pd.DataFrame(chosen)
    for model in MODELS:
        part = windows[windows.Model == model] if not windows.empty else windows
        if part.empty:
            continue
        key = f"worst_{SHORT[model]}"
        mdp.OBSERVATIONS[key] = (f"{model}: lowest score of the model", MODELS)
        with PdfPages(out / f"{key}.pdf") as pdf:
            for r in part.sort_values(["Dataset", "Rank", "Size_Sec"] if own.per_dataset
                                      else ["Rank", "Size_Sec"]).to_dict("records"):
                data = store[(r["Dataset"], r["Subject"], r["Recording"])]
                path = out / key / (f"{r['Rank']:02d}_{r['Dataset']}_{ov.safe_name(r['Subject'])}_"
                                    f"{int(r['Size_Sec'])}s.png")
                fig = mdp.draw(key, r, data, path)
                if own.no_png:
                    path.unlink(missing_ok=True)
                pdf.savefig(fig)
                plt.close(fig)
    notes = pd.DataFrame({"Item": ["Score", "Subject score", "Deficit", "Pattern", "Windows", "RRWaveNet output"],
                          "Meaning": [
                              f"PCC - {AE_WEIGHT} x min(rate AE, {AE_CAP}) per window",
                              "mean over the non-overlapping windows per size, then the mean over the sizes; lowest "
                              f"first, the {own.top} lowest listed" + (" per dataset" if own.per_dataset else ""),
                              "the model's subject score - the best subject score of the other models",
                              f"model-specific when Deficit <= {SPECIFIC_DEFICIT}, others worse when Deficit >= "
                              f"{-SPECIFIC_DEFICIT}, otherwise similar to others",
                              f"per listed subject and size, the window (start every {args.step:g} s) with the lowest "
                              "score of the model",
                              mdp.rrwavenet_note(args)]})
    with pd.ExcelWriter(out / "worst_subjects.xlsx") as writer:
        for name, frame in sheets.items():
            frame.round(4).to_excel(writer, sheet_name=name[:31], index=False)
        if not windows.empty:
            windows.round(4).to_excel(writer, sheet_name="Windows", index=False)
        notes.to_excel(writer, sheet_name="Notes", index=False)
    print(f"\nsaved under {out}")


if __name__ == "__main__":
    main()
