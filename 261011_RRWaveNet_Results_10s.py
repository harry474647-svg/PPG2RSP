r"""261011_RRWaveNet_Results_10s: all results of the 10 s windows of
261007_RRWaveNet_Ensemble_NoEvent, while its 20 s part is still training.

Nothing is trained. From the members already stored by 261007 (10 s
windows only) it writes:
1. the 261007 summary of the 10 s windows: ensembles of the finished folds,
   FFT and Davies & Mandic metrics of every dataset and variant, the
   ablation comparison (paired Wilcoxon against the full model), the
   per-subject table, the per-minute tables, figures and the workbook;
   every mean comes with its SD (ddof=1): MAE and waveform PCC (SD over
   minutes), bias (SD over minutes), subject MAE (SD over subjects), and in
   the ablation comparison the subject and minute MAE of both models and
   their difference; fft_davies_overall_mean_sd.csv lists them as
   "mean ± SD" (see add_standard_deviations);
2. waveform figures of the full model: for every held-out subject, 30 s of
   the ground truth (black) and the ensemble prediction (red), z-normalized,
   drawn twice: the unfiltered prediction, and the prediction after the
   respiratory filter that 261007 applies before the FFT rate. The excerpt is
   the first 30 s of the subject's scored minute with the median FFT error;
3. the FFT rate of every scored minute from the unfiltered prediction next
   to the filtered one (the reported rate): FFT and Davies metrics of both,
   the minutes whose rate differs and paired Wilcoxon tests;
4. the stem fusion feature analysis, Stem Fusion O (full) vs X
   (no_stem_fusion): 261010_RRWaveNet_StemFusion_Feature_Analysis;
5. the residual encoder feature analysis of the full model:
   261010_RRWaveNet_Encoder_Feature_Analysis.
Folds whose 3 members are not all stored yet are left out and listed in
incomplete_folds.csv, so the script can be run at any time and again later.

Outputs: <261007 results root>\Summary_10s\
  the 261007 summary files (10 s),
  Waveforms\<dataset>_w10s_nonoverlap_full\<subject>_waveform_unfiltered.png
                                          \<subject>_waveform_filtered.png
  Waveforms\waveform_excerpts.csv, prediction_filter_comparison.csv,
           prediction_filter_minutes.csv
  Stem_Fusion_OX\ and Encoder_Feature_Analysis\
The 261007 run itself writes its own Summary\ (both windows) when it ends;
Summary_10s\ is not touched by it.

It uses the 261007 file as it is (no change to it is needed). Run next to
the 261007 and the two 261010 scripts (Spyder: just run the file):
  python 261011_RRWaveNet_Results_10s.py
Options: --results-root <261007 results>, --datasets capnobase bidmc,
  --skip-feature-analyses (steps 4-5), --force-cpu, and the data path
  options of 261007. Colours, excerpt length and variant of the waveform
  figures: WAVEFORM_* constants below.
The data paths, model and results root default to those in the 261007 file.
"""

import argparse
import importlib.util
import shutil
import sys
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

sys.dont_write_bytecode = True

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

SCRIPT_DIR = Path(__file__).resolve().parent
BACKBONE = SCRIPT_DIR / "261007_RRWaveNet_Ensemble_NoEvent.py"
STEM_ANALYSIS = SCRIPT_DIR / "261010_RRWaveNet_StemFusion_Feature_Analysis.py"
ENCODER_ANALYSIS = SCRIPT_DIR / "261010_RRWaveNet_Encoder_Feature_Analysis.py"
WINDOW = "w10s_nonoverlap"
OUTPUT_FOLDER = "Summary_10s"
# Set to False to write only the performance summary and the waveforms.
RUN_FEATURE_ANALYSES = True
# Waveform figures: model variant, excerpt length and colours.
WAVEFORM_VARIANT = "full"
WAVEFORM_EXCERPT_SEC = 30.0
WAVEFORM_TRUE_COLOR = "#111111"  # ground truth: black
WAVEFORM_PRED_COLOR = "#d62728"  # prediction: red


def sd(values) -> float:
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    return float(np.std(values, ddof=1)) if values.size > 1 else np.nan


def insert_after(row: dict, extra: Dict[str, Tuple[str, float]]) -> dict:
    """Copy of row with extra[key] = (anchor, value) placed right after anchor."""
    out = {}
    for key, value in row.items():
        out[key] = value
        for name, (anchor, extra_value) in extra.items():
            if anchor == key:
                out[name] = extra_value
    for name, (anchor, extra_value) in extra.items():
        out.setdefault(name, extra_value)
    return out


def add_standard_deviations(backbone) -> None:
    """Report the SD (ddof=1) next to every mean in the 261007 summary.

    The 261007 file is not changed; its summary functions are wrapped here:
    * fft_davies_overall: FFT_MAE_SD (SD of the per-minute absolute errors),
      FFT_Bias_SD (SD of the per-minute errors; LoA = Bias +- 1.96 SD) and
      Wave_PCC_SD (SD of the per-minute waveform PCC), next to FFT_MAE,
      FFT_Bias and Wave_PCC_Mean (FFT_Subject_MAE_SD, the SD over subjects,
      is already there). Rows of single models (no_ensemble): each model's
      SD, averaged over the models, as for every other metric of that row.
    * ablation_comparison: SD of the per-subject MAE of both models and of
      their difference, SD of the per-minute absolute errors of both.
    * per_subject_fft: SD over the subject's minutes of the absolute error,
      the error and the waveform PCC.
    RMSE, Pearson r and the within-1/2 bpm percentages are not means of
    per-minute values; the Davies mAE / mMAE are medians (with IQR)."""
    base_metrics = backbone.fft_davies_metrics
    base_comparison = backbone.paired_comparison
    base_subject_table = backbone.per_subject_table

    def fft_davies_metrics(minutes: pd.DataFrame) -> dict:
        result = base_metrics(minutes)
        mae_sd = bias_sd = pcc_sd = np.nan
        if not minutes.empty:
            pcc_sd = sd(pd.to_numeric(minutes["Wave_PCC"], errors="coerce"))
            errors = minutes.loc[np.isfinite(minutes["FFT_AE"]), "FFT_Error"].to_numpy(float)
            mae_sd, bias_sd = sd(np.abs(errors)), sd(errors)
        return insert_after(result, {
            "FFT_MAE_SD": ("FFT_MAE", mae_sd),
            "FFT_Bias_SD": ("FFT_Bias", bias_sd),
            "Wave_PCC_SD": ("Wave_PCC_Mean", pcc_sd),
        })

    def paired_comparison(full: pd.DataFrame, other: pd.DataFrame) -> dict:
        result = base_comparison(full, other)
        a, b = backbone.subject_errors(full), backbone.subject_errors(other)
        subjects = a.index.intersection(b.index)
        sa, sb = a.loc[subjects].to_numpy(float), b.loc[subjects].to_numpy(float)
        ma, mb = backbone.minute_errors(full), backbone.minute_errors(other)
        keys = ma.index.intersection(mb.index)
        return insert_after(result, {
            "Subject_MAE_Full_SD": ("Subject_MAE_Full", sd(sa)),
            "Subject_MAE_Variant_SD": ("Subject_MAE_Variant", sd(sb)),
            "Subject_MAE_Diff_SD": ("Subject_MAE_Diff_Variant_minus_Full", sd(sb - sa)),
            "Minute_MAE_Full_SD": ("Minute_MAE_Full", sd(ma.loc[keys])),
            "Minute_MAE_Variant_SD": ("Minute_MAE_Variant", sd(mb.loc[keys])),
        })

    def per_subject_table(minutes: pd.DataFrame) -> pd.DataFrame:
        table = base_subject_table(minutes)
        if table.empty:
            return table
        have = minutes[np.isfinite(minutes["FFT_AE"])]
        if "Member" in have.columns:
            have = have.groupby(["Subject", *backbone.MINUTE_KEY[1:]], as_index=False).agg(
                FFT_Error=("FFT_Error", "mean"), FFT_AE=("FFT_AE", "mean"), Wave_PCC=("Wave_PCC", "mean"),
            )
        spread = have.groupby("Subject").agg(
            FFT_MAE_SD=("FFT_AE", "std"), FFT_Bias_SD=("FFT_Error", "std"), Wave_PCC_SD=("Wave_PCC", "std"),
        ).reset_index()
        table = table.merge(spread, on="Subject", how="left")
        order = []
        for column in table.columns:
            if column.endswith("_SD") and column in spread.columns:
                continue
            order.append(column)
            partner = {"FFT_MAE": "FFT_MAE_SD", "FFT_Bias": "FFT_Bias_SD", "Wave_PCC_Mean": "Wave_PCC_SD"}.get(column)
            if partner:
                order.append(partner)
        return table[order]

    def with_sd(metrics: Sequence[str]) -> tuple:
        partners = {"FFT_MAE": "FFT_MAE_SD", "FFT_Bias": "FFT_Bias_SD", "Wave_PCC_Mean": "Wave_PCC_SD"}
        out = []
        for metric in metrics:
            out.append(metric)
            if metric in partners and partners[metric] not in metrics:
                out.append(partners[metric])
        return tuple(out)

    backbone.fft_davies_metrics = fft_davies_metrics
    backbone.paired_comparison = paired_comparison
    backbone.per_subject_table = per_subject_table
    # Single-model rows average every metric of SUMMARY_METRICS over the models.
    backbone.SUMMARY_METRICS = with_sd(backbone.SUMMARY_METRICS)


def mean_sd_text(mean: float, spread: float, digits: int = 3) -> str:
    if not np.isfinite(mean):
        return "n/a"
    return f"{mean:.{digits}f} ± {spread:.{digits}f}" if np.isfinite(spread) else f"{mean:.{digits}f}"


def write_mean_sd_tables(out_root: Path) -> None:
    """Readable mean ± SD tables of the final performance (and the console)."""
    overall_path = out_root / "fft_davies_overall.csv"
    if not overall_path.exists():
        return
    overall = pd.read_csv(overall_path, encoding="utf-8-sig")
    rows = []
    for _, r in overall.iterrows():
        rows.append({
            "Dataset": r["Dataset"], "Window_Config": r["Window_Config"], "Row": r["Row"], "Model": r["Model"],
            "N_Subjects": int(r["N_Subjects"]), "N_Minutes": int(r["N_Minutes"]),
            "MAE (mean ± SD over minutes)": mean_sd_text(r["FFT_MAE"], r.get("FFT_MAE_SD", np.nan)),
            "Subject MAE (mean ± SD over subjects)": mean_sd_text(r["FFT_Subject_MAE_Mean"], r["FFT_Subject_MAE_SD"]),
            "Bias (mean ± SD)": mean_sd_text(r["FFT_Bias"], r.get("FFT_Bias_SD", np.nan)),
            "LoA": f"{r['FFT_LoA_Lower']:.2f} to {r['FFT_LoA_Upper']:.2f}",
            "RMSE": f"{r['FFT_RMSE']:.3f}",
            "Pearson r": f"{r['FFT_Pearson_r']:.3f}",
            "Within 1 / 2 bpm (%)": f"{r['FFT_Within_1_BPM_Pct']:.1f} / {r['FFT_Within_2_BPM_Pct']:.1f}",
            "mAE [IQR]": f"{r['Davies_mAE']:.3f} [{r['Davies_mAE_Q1']:.3f}-{r['Davies_mAE_Q3']:.3f}]",
            "mMAE [IQR]": f"{r['Davies_mMAE']:.3f} [{r['Davies_mMAE_Q1']:.3f}-{r['Davies_mMAE_Q3']:.3f}]",
            "Wave PCC (mean ± SD over minutes)": mean_sd_text(r["Wave_PCC_Mean"], r.get("Wave_PCC_SD", np.nan)),
        })
    table = pd.DataFrame(rows)
    table.to_csv(out_root / "fft_davies_overall_mean_sd.csv", index=False, encoding="utf-8-sig")
    print("\nFinal performance, mean ± SD (MAE and PCC: SD over minutes; subject MAE: SD over subjects; "
          "mAE / mMAE: median [IQR]):")
    for _, r in table.iterrows():
        print(f"  {r['Dataset']:<10s} {str(r['Model'])[:42]:<42s} MAE {r['MAE (mean ± SD over minutes)']:<15s} "
              f"subject MAE {r['Subject MAE (mean ± SD over subjects)']:<15s} "
              f"bias {r['Bias (mean ± SD)']:<16s} PCC {r['Wave PCC (mean ± SD over minutes)']:<15s} "
              f"mAE {r['mAE [IQR]']}  mMAE {r['mMAE [IQR]']}")
    comparison_path = out_root / "ablation_comparison.csv"
    if comparison_path.exists():
        comparison = pd.read_csv(comparison_path, encoding="utf-8-sig")
        if not comparison.empty and "Subject_MAE_Full_SD" in comparison.columns:
            print("\nAblation (subject MAE, mean ± SD over subjects; minute MAE, mean ± SD over minutes):")
            for _, r in comparison.iterrows():
                print(f"  {r['Dataset']:<10s} {r['Comparison'][:58]:<58s} subject "
                      f"{mean_sd_text(r['Subject_MAE_Full'], r['Subject_MAE_Full_SD'])} vs "
                      f"{mean_sd_text(r['Subject_MAE_Variant'], r['Subject_MAE_Variant_SD'])} "
                      f"(diff {mean_sd_text(r['Subject_MAE_Diff_Variant_minus_Full'], r['Subject_MAE_Diff_SD'])}, "
                      f"p={r['Subject_Wilcoxon_p']:.3g}); minute "
                      f"{mean_sd_text(r['Minute_MAE_Full'], r['Minute_MAE_Full_SD'])} vs "
                      f"{mean_sd_text(r['Minute_MAE_Variant'], r['Minute_MAE_Variant_SD'])}")


def load_script(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, str(path))
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def zscore(signal: np.ndarray) -> np.ndarray:
    values = np.asarray(signal, dtype=np.float64)
    sd = np.nanstd(values)
    if not np.isfinite(sd) or sd <= 1e-12:
        return np.full_like(values, np.nan)
    return (values - np.nanmean(values)) / sd


def draw_waveform(path: Path, time: np.ndarray, true: np.ndarray, pred: np.ndarray,
                  label: str, title: str) -> None:
    fig, axis = plt.subplots(figsize=(10.0, 5.4))
    axis.plot(time, zscore(true), color=WAVEFORM_TRUE_COLOR, linewidth=2.6, label="Ground truth")
    axis.plot(time, zscore(pred), color=WAVEFORM_PRED_COLOR, linewidth=2.6, label=label)
    axis.set_xlim(0, WAVEFORM_EXCERPT_SEC)
    axis.set_xlabel("Time (s)", fontsize=12)
    axis.set_ylabel("z-normalized respiratory waveform", fontsize=12)
    axis.grid(True, color="#e6e6e6")
    axis.legend(loc="upper right", fontsize=9)
    axis.set_title(title, fontsize=9)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def subject_waveforms(backbone, root: Path, spec, dataset_label: str, subject: str,
                      minutes: pd.DataFrame, references: Dict[int, dict], figure_root: Path):
    """Both waveform figures of one held-out subject and the filtered and
    unfiltered FFT rate of each of its scored minutes."""
    fs = backbone.TARGET_FS
    eval_len = int(round(backbone.EVAL_WINDOW_SEC * fs))
    arrays = np.load(backbone.ensemble_paths(root, subject)[2], allow_pickle=False)
    metadata = [
        {"segment_id": int(segment), "start": int(start)}
        for segment, start in zip(arrays["segment_ids"], arrays["window_starts"])
    ]
    series = {}
    for segment_id in sorted({int(value) for value in minutes["Segment_ID"]}):
        pred, true, coverage = backbone.reconstruct_segment(
            arrays["predictions"], arrays["targets"], metadata, segment_id, fs
        )
        reference = references[segment_id]
        filtered = np.full_like(pred, np.nan)
        # As in the 261007 evaluation: every contiguous run of at least one
        # minute is filtered with the profile's respiratory band.
        for run_start, run_end in backbone.contiguous_runs(coverage):
            if run_end - run_start >= eval_len:
                filtered[run_start:run_end] = backbone.respiratory_filter(
                    pred[run_start:run_end], fs, reference["band_hz"], reference["profile"]
                )
        series[segment_id] = (pred, true, filtered, reference["profile"]["rr_band_bpm"])
    rows = []
    for _, minute in minutes.iterrows():
        pred, true, filtered, rr_band = series[int(minute["Segment_ID"])]
        start = int(round(float(minute["RR_Window_Start_Sec"]) * fs))
        end = start + eval_len
        gt = float(minute["GT_RR"])
        rate_unfiltered = backbone.fft_rr_bpm(pred[start:end], fs, rr_band, False)
        rate_filtered = backbone.fft_rr_bpm(filtered[start:end], fs, rr_band, False)
        rows.append({
            "Dataset": dataset_label, "Window_Config": spec.name, "Variant": WAVEFORM_VARIANT,
            "Subject": subject, "Segment_ID": int(minute["Segment_ID"]),
            "RR_Window_Start_Sec": float(minute["RR_Window_Start_Sec"]), "GT_RR": gt,
            "FFT_RR_Reported": float(minute["FFT_RR"]),
            "FFT_RR_Filtered": rate_filtered, "FFT_RR_Unfiltered": rate_unfiltered,
            "AE_Filtered": abs(rate_filtered - gt) if np.isfinite(rate_filtered) else np.nan,
            "AE_Unfiltered": abs(rate_unfiltered - gt) if np.isfinite(rate_unfiltered) else np.nan,
            "FFT_RR_Differs": bool(not np.isclose(rate_filtered, rate_unfiltered, equal_nan=True)),
            "Wave_PCC_Unfiltered": backbone.safe_pcc(true[start:end], pred[start:end]),
            "Wave_PCC_Filtered": backbone.safe_pcc(true[start:end], filtered[start:end]),
        })
    frame = pd.DataFrame(rows)
    ranked = frame[np.isfinite(frame["AE_Filtered"])].sort_values(
        ["AE_Filtered", "Segment_ID", "RR_Window_Start_Sec"]
    )
    if ranked.empty:
        return rows, None
    chosen = ranked.iloc[(len(ranked) - 1) // 2]
    pred, true, filtered, _ = series[int(chosen["Segment_ID"])]
    start = int(round(float(chosen["RR_Window_Start_Sec"]) * fs))
    excerpt = slice(start, start + int(round(WAVEFORM_EXCERPT_SEC * fs)))
    time = np.arange(pred[excerpt].size) / float(fs)
    members = len(np.atleast_1d(arrays["seeds"]))
    head = (f"{dataset_label} {subject}, {spec.label}: segment {int(chosen['Segment_ID'])}, "
            f"{float(chosen['RR_Window_Start_Sec']):.0f} s; reference {chosen['GT_RR']:.1f} breaths/min")
    figures = {}
    for version, prediction, rate, label in (
        ("unfiltered", pred, chosen["FFT_RR_Unfiltered"], f"Predicted (ensemble of {members})"),
        ("filtered", filtered, chosen["FFT_RR_Filtered"],
         f"Predicted, respiratory filter (ensemble of {members})"),
    ):
        path = figure_root / f"{backbone.safe_tag(subject)}_waveform_{version}.png"
        draw_waveform(path, time, true[excerpt], prediction[excerpt], label,
                      f"{head}, predicted {rate:.1f} ({version} prediction)")
        figures[version] = str(path)
    excerpt_row = {key: chosen[key] for key in (
        "Dataset", "Window_Config", "Variant", "Subject", "Segment_ID", "GT_RR",
        "FFT_RR_Filtered", "FFT_RR_Unfiltered", "AE_Filtered", "AE_Unfiltered",
        "Wave_PCC_Unfiltered", "Wave_PCC_Filtered")}
    excerpt_row.update({
        "Excerpt_Start_Sec": float(chosen["RR_Window_Start_Sec"]),
        "Excerpt_Sec": float(time.size / fs),
        "Figure_Unfiltered": figures["unfiltered"], "Figure_Filtered": figures["filtered"],
    })
    return rows, excerpt_row


def filter_comparison(backbone, minute_rows: List[dict]) -> pd.DataFrame:
    """FFT and Davies metrics with the filtered and the unfiltered rate."""
    frame = pd.DataFrame(minute_rows)
    rows = []
    for (dataset, window, variant), group in frame.groupby(["Dataset", "Window_Config", "Variant"], sort=False):
        per_version = {}
        for version in ("Filtered", "Unfiltered"):
            minutes = pd.DataFrame({
                "Subject": group["Subject"], "GT_RR": group["GT_RR"],
                "FFT_RR": group[f"FFT_RR_{version}"],
                "FFT_Error": group[f"FFT_RR_{version}"] - group["GT_RR"],
                "FFT_AE": group[f"AE_{version}"], "Wave_PCC": group[f"Wave_PCC_{version}"],
            })
            per_version[version] = minutes
            metrics = backbone.fft_davies_metrics(minutes)
            metrics.update({
                "Dataset": dataset, "Window_Config": window, "Variant": variant,
                "Prediction": ("respiratory filter before the FFT (reported rates)"
                               if version == "Filtered" else "no filter"),
                "N_Minutes_FFT_Rate_Differs": int(group["FFT_RR_Differs"].sum()),
            })
            rows.append(metrics)
        a = per_version["Filtered"].groupby("Subject")["FFT_AE"].mean()
        b = per_version["Unfiltered"].groupby("Subject")["FFT_AE"].mean()
        subjects = a.index.intersection(b.index)
        keep = np.isfinite(group["AE_Filtered"]) & np.isfinite(group["AE_Unfiltered"])
        p_subject = backbone.safe_pvalue(stats.wilcoxon, a.loc[subjects].to_numpy(float),
                                         b.loc[subjects].to_numpy(float))
        p_minute = backbone.safe_pvalue(stats.wilcoxon, group.loc[keep, "AE_Filtered"].to_numpy(float),
                                        group.loc[keep, "AE_Unfiltered"].to_numpy(float))
        for row in rows[-2:]:
            row["Subject_MAE_Wilcoxon_p"] = p_subject
            row["Minute_AE_Wilcoxon_p"] = p_minute
    table = pd.DataFrame(rows)
    first = ["Dataset", "Window_Config", "Variant", "Prediction", "N_Minutes_FFT_Rate_Differs"]
    return table[first + [c for c in table.columns if c not in first]] if not table.empty else table


def waveform_results(backbone, results_root: Path, window_specs, dataset_keys, subjects_filter,
                     out_root: Path) -> None:
    """Waveform figures and the filtered / unfiltered rate comparison."""
    waveform_root = out_root / "Waveforms"
    waveform_root.mkdir(parents=True, exist_ok=True)
    excerpt_rows, minute_rows = [], []
    variant = backbone.VARIANTS[WAVEFORM_VARIANT]
    for spec in window_specs:
        for dataset_key in dataset_keys:
            label = backbone.DATASET_CONFIGS[dataset_key]["label"]
            root = backbone.condition_root(results_root, variant.name, spec, dataset_key)
            if not (root / "Ensemble_Predictions").exists():
                print(f"{label} {spec.name}: no ensemble predictions, skipped")
                continue
            subject_data = backbone.load_dataset(dataset_key, spec)
            figure_root = waveform_root / f"{label}_{spec.name}_{variant.name}"
            figure_root.mkdir(parents=True, exist_ok=True)
            done = 0
            for subject in sorted(subject_data):
                if subjects_filter and subject not in subjects_filter:
                    continue
                summary_path, window_path, prediction_path = backbone.ensemble_paths(root, subject)
                if not (summary_path.exists() and window_path.exists() and prediction_path.exists()):
                    continue
                minutes = backbone.scored_minutes(backbone.read_minutes([window_path]))
                if minutes.empty:
                    continue
                rows, excerpt = subject_waveforms(
                    backbone, root, spec, label, subject, minutes,
                    subject_data[subject]["references"], figure_root,
                )
                minute_rows += rows
                if excerpt is not None:
                    excerpt_rows.append(excerpt)
                    done += 1
            print(f"{label} {spec.name}: waveform figures of {done} subjects")
            del subject_data
            backbone.release_memory()
    if not minute_rows:
        print("No ensemble result to draw yet.")
        return
    pd.DataFrame(excerpt_rows).to_csv(waveform_root / "waveform_excerpts.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(minute_rows).to_csv(waveform_root / "prediction_filter_minutes.csv", index=False,
                                     encoding="utf-8-sig")
    table = filter_comparison(backbone, minute_rows)
    table.to_csv(waveform_root / "prediction_filter_comparison.csv", index=False, encoding="utf-8-sig")
    print("\nFFT rate of the filtered (reported) vs the unfiltered prediction (ensembles):")
    for _, row in table.iterrows():
        print(f"  {row['Dataset']:<10s} {row['Prediction'][:28]:<28s} "
              f"MAE {mean_sd_text(row['FFT_MAE'], row['FFT_MAE_SD'])}, "
              f"mAE {row['Davies_mAE']:.3f}, mMAE {row['Davies_mMAE']:.3f}, "
              f"PCC {mean_sd_text(row['Wave_PCC_Mean'], row['Wave_PCC_SD'])}; "
              f"rate differs in {row['N_Minutes_FFT_Rate_Differs']}/{row['N_Minutes']} minutes, "
              f"subject p={row['Subject_MAE_Wilcoxon_p']:.3g}")
    print(f"Waveforms saved under: {waveform_root}")


def data_path_arguments(args: argparse.Namespace):
    extra = []
    for option, value in (
        ("--capnobase-path", args.capnobase_path),
        ("--bidmc-path", args.bidmc_path),
        ("--steam2-ppg-dir", args.steam2_ppg_dir),
        ("--steam2-rsp-dir", args.steam2_rsp_dir),
    ):
        if value:
            extra += [option, str(value)]
    return extra


def main(args: argparse.Namespace) -> None:
    backbone = load_script(Path(args.backbone), "rrwavenet_261007_results")
    add_standard_deviations(backbone)
    paths = data_path_arguments(args)
    summary_args = backbone.parse_args(
        ["--summary-only", "--windows", WINDOW, "--datasets", *args.datasets, *paths]
        + (["--results-root", str(args.results_root)] if args.results_root else [])
    )
    results_root = Path(summary_args.results_root).resolve()
    configs = backbone.DATASET_CONFIGS
    configs["capnobase"]["use_artifact_labels"] = summary_args.capnobase_artifacts
    if summary_args.capnobase_path:
        configs["capnobase"]["data_path"] = str(summary_args.capnobase_path)
    if summary_args.bidmc_path:
        configs["bidmc"]["data_path"] = str(summary_args.bidmc_path)
    if summary_args.steam2_ppg_dir:
        configs["steam2"]["ppg_dir"] = str(summary_args.steam2_ppg_dir)
    if summary_args.steam2_rsp_dir:
        configs["steam2"]["rsp_dir"] = str(summary_args.steam2_rsp_dir)
    out_root = results_root / OUTPUT_FOLDER
    out_root.mkdir(parents=True, exist_ok=True)

    # 1. Performance summary of the 10 s windows (summarize_results of 261007,
    #    written to Summary\ and copied to Summary_10s\; no manifest is written).
    print(f"=== 1. 261007 summary, 10 s windows ({results_root}) ===")
    variants = [backbone.VARIANTS[name] for name in backbone.VARIANTS if name in summary_args.variants]
    window_specs = [spec for spec in backbone.WINDOW_SPECS if spec.name == WINDOW]
    dataset_keys = [key for key in backbone.DATASET_ORDER if key in summary_args.datasets]
    backbone.summarize_results(summary_args, results_root, variants, window_specs, dataset_keys)
    summary_root = results_root / "Summary"
    for item in summary_root.iterdir():
        if item.is_dir() and item.name in ("Stem_Fusion_OX", "Encoder_Feature_Analysis", "Waveforms"):
            continue
        target = out_root / item.name
        if item.is_dir():
            shutil.copytree(item, target, dirs_exist_ok=True)
        else:
            shutil.copy2(item, target)
    write_mean_sd_tables(out_root)

    # 2. Waveform figures (unfiltered and filtered prediction) and the
    #    filtered / unfiltered FFT rate comparison.
    print("\n=== 2. Waveform figures and filtered vs unfiltered rate, 10 s ===")
    waveform_results(backbone, results_root, window_specs, dataset_keys, None, out_root)

    if args.skip_feature_analyses or not RUN_FEATURE_ANALYSES:
        print(f"\nSaved under: {out_root}")
        return
    common = ["--backbone", str(Path(args.backbone).resolve()), "--results-root", str(results_root),
              "--windows", WINDOW, "--datasets", *args.datasets, *paths]
    if args.force_cpu:
        common.append("--force-cpu")

    # 3. Stem Fusion O vs X.
    print("\n=== 3. Stem fusion feature analysis (O = full, X = no_stem_fusion), 10 s ===")
    stem = load_script(STEM_ANALYSIS, "stem_fusion_analysis_10s")
    stem.analyse(stem.parse_args(common + ["--output", str(out_root / "Stem_Fusion_OX")]))

    # 4. Residual encoder blocks of the full model.
    print("\n=== 4. Residual encoder feature analysis (full model), 10 s ===")
    encoder = load_script(ENCODER_ANALYSIS, "encoder_feature_analysis_10s")
    encoder.analyse(encoder.parse_args(common + ["--output", str(out_root / "Encoder_Feature_Analysis")]))
    print(f"\nAll 10 s results saved under: {out_root}")


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--backbone", type=Path, default=BACKBONE)
    parser.add_argument("--results-root", type=Path, default=None,
                        help="Results root of the 261007 run (default: its PROJECT_ROOT).")
    parser.add_argument("--datasets", nargs="+", default=["capnobase", "bidmc", "steam2"],
                        choices=["capnobase", "bidmc", "steam2"])
    parser.add_argument("--skip-feature-analyses", action="store_true")
    parser.add_argument("--force-cpu", action="store_true")
    parser.add_argument("--capnobase-path", type=Path, default=None)
    parser.add_argument("--bidmc-path", type=Path, default=None)
    parser.add_argument("--steam2-ppg-dir", type=Path, default=None)
    parser.add_argument("--steam2-rsp-dir", type=Path, default=None)
    return parser.parse_args(argv)


if __name__ == "__main__":
    main(parse_args())
