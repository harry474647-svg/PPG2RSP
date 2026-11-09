# -*- coding: utf-8 -*-
r"""261109_Waveform_FreqAnalysis: the same RRWaveNet models (261013 and 261007
versions) trained and tested by LOSOCV on STEAM2, BIDMC and CapnoBase with the
input PPG band-passed 0.1-0.8 Hz and 0.1-8.0 Hz, and the comparison of the
results. Every result is saved under one folder.

Runs, in this order (each: STEAM2 -> BIDMC -> CapnoBase)
  01  PPG 0.1-0.8 Hz + 261013 (subject-wise validation, weight-decay selection, ensemble of 3)
  02  PPG 0.1-0.8 Hz + 261007 (temporal validation, full model, ensemble of 3)
  03  PPG 0.1-8.0 Hz + 261013   (the 261109_RRWaveNet_HighFreq input)
  04  PPG 0.1-8.0 Hz + 261007
Each run is the code of the base script with exactly these text changes (each
must match exactly once, otherwise nothing runs):
  1. PPG_INPUT = "raw"  ->  PPG_INPUT = "bandpass", PPG_INPUT_BAND_HZ = (low, high),
                            PPG_INPUT_FILTER_ORDER = 3
  2. the model_input line -> with PPG_INPUT == "bandpass": sos_bandpass(ppg_raw, fs, low,
     high, order=3) (zero-phase Butterworth, second-order sections, over the whole
     segment; the windows are cut from it and min-max scaled per window as before)
  3. PROJECT_ROOT -> the run's folder below
The base runs fed the raw PPG (no filter). Model, loss, training, ensembles,
validation, LOSOCV, quality rules, targets, scoring and outputs are unchanged.
The model is trained per ensemble member with its own seed, so running the
datasets one by one gives the same models as one call.

How a run is executed
  * one call of the run's script per dataset, in the order STEAM2, BIDMC,
    CapnoBase (the scripts themselves walk CapnoBase, BIDMC, STEAM2), with
    --resume (finished models are kept, interrupted ones continue; a new folder
    trains everything); the failed_members.csv / run_manifest.json of each call
    are kept as failed_members_<dataset>.csv / run_manifest_<dataset>.json;
  * then one --summary-only call over the three datasets (the summaries and
    results workbook of the run cover all three);
  * 261007 runs the full model only (--variants full; its ablation variants are
    other models) and its default windows (10 s, 20 s); 261013 its default
    (10 s, weight decays 0.01 / 0.1 / 1); the paired comparison of 261013 with
    261007 reads the 261007 run of the same band (its summary is written again
    after that run is finished).
Then the comparison of all runs (and of the earlier raw-input runs when found:
D:\PPG2RSP_RRWaveNet_Inspired\261007_RRWaveNet_NoEvent_v1 and
261013_RRWavNet_STEAM_BIDMC_v1, read only).

Base scripts (searched in --scripts-dir, the folder of this file, the current
folder, D:\PPG2RSP_RRWaveNet_Inspired, its run folders and the parent folder):
  261013_RRWaveNet_Adult_SubjectVal_WD.py, 261007_RRWaveNet_Ensemble_NoEvent.py
Their text (results-folder line left out) is compared with the versions this code
was checked against; another version is reported and still used.

Outputs (D:\PPG2RSP_RRWaveNet_Inspired\261109_Waveform_FreqAnalysis)
  Code\                                   the four scripts that run, changes.txt
  01_PPG_0.1-0.8Hz_261013\ ... 04_PPG_0.1-8.0Hz_261007\
                                          results of every run, in the layout of the base
                                          run (Summary\..._results.xlsx, per_subject_fft.csv, ...)
  run_log.csv                             every call: run, dataset, start, end, minutes, exit code
  Comparison\261109_Waveform_FreqAnalysis_Comparison.xlsx (+ csv)
    Summary        dataset x version x input band x window: subjects, FFT MAE and
                   waveform PCC (mean +- SD, median [IQR]); all subjects and
                   issue subjects left out (STEAM2 22_LMS, 26_CKK, 12_YJJ; BIDMC 27;
                   CapnoBase 0115)
    Paired_Tests   per dataset, version and window: 0.1-0.8 vs 0.1-8.0 Hz, and each band
                   vs the raw input when found: paired subjects, median difference,
                   subjects better, Wilcoxon p, BH-adjusted p
    Overall        the pooled metrics of every run (fft_davies_overall of its summary)
    Per_Subject    every subject of every run
    Run_Status     subjects per dataset and window, incomplete folds
    Notes

Run (needs CUDA as the base scripts; long: four LOSOCV runs on three datasets):
  python 261109_Waveform_FreqAnalysis.py
  python 261109_Waveform_FreqAnalysis.py --plan            print the calls only
  python 261109_Waveform_FreqAnalysis.py --compare-only    only the comparison of what is finished
  python 261109_Waveform_FreqAnalysis.py --runs 03 04      only these runs
  python 261109_Waveform_FreqAnalysis.py --args-261007 "--windows w10s_nonoverlap"
Other options go to every script call unchanged (e.g. --gpu 1, --no-require-cuda).
Options of this file: --runs, --scripts-dir, --output-root, --args-261007,
  --args-261013, --no-resume, --write-only, --plan, --compare-only
"""

import argparse
import hashlib
import math
import re
import shlex
import shutil
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

NAME = "261109_Waveform_FreqAnalysis"
WORKSPACE = Path(r"D:\PPG2RSP_RRWaveNet_Inspired")
OUTPUT_ROOT = WORKSPACE / NAME
FILTER_ORDER = 3
DATASET_RUN_ORDER = ("steam2", "bidmc", "capnobase")
DATASET_LABELS = {"steam2": "STEAM2", "bidmc": "BIDMC", "capnobase": "CapnoBase"}
EXCLUDED = {"STEAM2": ("22_LMS", "26_CKK", "12_YJJ"), "BIDMC": ("27",), "CapnoBase": ("0115",)}

# version -> base script, earlier (raw-input) run folder, SHA-256 of the checked base text (CRLF -> LF, PROJECT_ROOT
# line left out), options of the version, results workbook name of the base
VERSIONS = {
    "261013": {
        "script": "261013_RRWaveNet_Adult_SubjectVal_WD.py",
        "raw_run": "261013_RRWavNet_STEAM_BIDMC_v1",
        "sha256": "364b109b1e0da1f7ed6ea4a9276a77ed0ad554363b32605cb6484a37642ca3dc",
        "options": [],
    },
    "261007": {
        "script": "261007_RRWaveNet_Ensemble_NoEvent.py",
        "raw_run": "261007_RRWaveNet_NoEvent_v1",
        "sha256": "18ed5840118f660f7c80ef6c7ed34744597675ba9ebd8115703aa265739f1841",
        "options": ["--variants", "full"],
    },
}
# the four runs in the requested order: (id, band label, band, version)
RUNS = (
    ("01", "0.1-0.8Hz", (0.1, 0.8), "261013"),
    ("02", "0.1-0.8Hz", (0.1, 0.8), "261007"),
    ("03", "0.1-8.0Hz", (0.1, 8.0), "261013"),
    ("04", "0.1-8.0Hz", (0.1, 8.0), "261007"),
)

INPUT_LINE = 'PPG_INPUT = "raw"\n'
MODEL_INPUT_LINE = '    model_input = (ppg_raw if PPG_INPUT == "raw" else ppg_low).astype(np.float32)\n'
ROOT_PATTERN = re.compile(r'^PROJECT_ROOT = Path\(r"[^"\n]*"\)$', re.MULTILINE)


# ---------------------------------------------------------------------------
# The scripts of the four runs
# ---------------------------------------------------------------------------

def run_folder(output_root: Path, run_id: str, band_label: str, version: str) -> Path:
    return output_root / f"{run_id}_PPG_{band_label}_{version}"


def base_hash(text: str) -> str:
    return hashlib.sha256(ROOT_PATTERN.sub("PROJECT_ROOT = <root>", text).encode("utf-8")).hexdigest()


def find_script(name: str, folders: Sequence[Optional[Path]]) -> Optional[Path]:
    here = Path(__file__).resolve().parent
    candidates = [Path(f) for f in folders if f]
    candidates += [here, Path.cwd(), WORKSPACE] + [WORKSPACE / v["raw_run"] for v in VERSIONS.values()] + [here.parent]
    for base in (WORKSPACE, here.parent):
        if base.is_dir():
            candidates += sorted(child for child in base.iterdir() if child.is_dir())
    seen = set()
    for folder in candidates:
        path = folder / name
        if path in seen:
            continue
        seen.add(path)
        if path.is_file():
            return path.resolve()
    return None


def patched_text(text: str, base_name: str, band: Tuple[float, float], root: Path) -> str:
    """The base script with exactly the three changes; stops when one does not match once."""
    text = text.replace("\r\n", "\n")
    for old, label in ((INPUT_LINE, "PPG_INPUT line"), (MODEL_INPUT_LINE, "model_input line")):
        count = text.count(old)
        if count != 1:
            raise SystemExit(f"{base_name}: the {label} was found {count} times (expected once); nothing is run")
    if len(ROOT_PATTERN.findall(text)) != 1:
        raise SystemExit(f"{base_name}: PROJECT_ROOT line not found once; nothing is run")
    input_new = (f'PPG_INPUT = "bandpass"  # {NAME}: the PPG band-passed {band[0]:g}-{band[1]:g} Hz is the model input '
                 '(was "raw")\n'
                 f"PPG_INPUT_BAND_HZ = ({band[0]!r}, {band[1]!r})\n"
                 f"PPG_INPUT_FILTER_ORDER = {FILTER_ORDER}\n")
    model_input_new = (
        '    if PPG_INPUT == "bandpass":\n'
        f"        # {NAME}: zero-phase Butterworth band-pass of the whole segment, then the windows are cut\n"
        "        model_input = sos_bandpass(\n"
        "            ppg_raw, fs, PPG_INPUT_BAND_HZ[0], PPG_INPUT_BAND_HZ[1], order=PPG_INPUT_FILTER_ORDER\n"
        "        ).astype(np.float32)\n"
        "    else:\n"
        '        model_input = (ppg_raw if PPG_INPUT == "raw" else ppg_low).astype(np.float32)\n')
    text = text.replace(INPUT_LINE, input_new).replace(MODEL_INPUT_LINE, model_input_new)
    text = ROOT_PATTERN.sub(f'PROJECT_ROOT = Path(r"{root}")', text)
    head = (f"# {NAME}: generated from {base_name}; the only changes: the model input is the PPG band-passed "
            f"{band[0]:g}-{band[1]:g} Hz (PPG_INPUT, model_input) and PROJECT_ROOT = {root}\n")
    if text.startswith("# -*- coding"):
        first, rest = text.split("\n", 1)
        return f"{first}\n{head}{rest}"
    return head + text


def write_scripts(args: argparse.Namespace, runs: Sequence[tuple]) -> Dict[str, Path]:
    code_dir = Path(args.output_root) / "Code"
    code_dir.mkdir(parents=True, exist_ok=True)
    report = [f"{NAME} ({datetime.now():%Y-%m-%d %H:%M:%S})",
              "Changes in each script: 1. PPG_INPUT line  2. model_input line  3. PROJECT_ROOT", ""]
    bases: Dict[str, Tuple[Path, str]] = {}
    scripts = {}
    for run_id, band_label, band, version in runs:
        if version not in bases:
            base = find_script(VERSIONS[version]["script"], [args.scripts_dir])
            if base is None:
                raise SystemExit(f"{VERSIONS[version]['script']} not found (give its folder with --scripts-dir)")
            text = base.read_text(encoding="utf-8").replace("\r\n", "\n")
            digest = base_hash(text)
            same = digest == VERSIONS[version]["sha256"]
            bases[version] = (base, text)
            line = (f"[{version}] base script {base}; SHA-256 {digest} - "
                    + ("same as the checked version" if same else
                       "DIFFERENT from the version this code was checked against (still used)"))
            print(line)
            report.append(line)
        base, text = bases[version]
        root = run_folder(Path(args.output_root), run_id, band_label, version)
        path = code_dir / f"{NAME}_{run_id}_PPG_{band_label}_{version}.py"
        path.write_text(patched_text(text, base.name, band, root), encoding="utf-8", newline="\n")
        scripts[run_id] = path
        report.append(f"[{run_id}] input band {band[0]:g}-{band[1]:g} Hz, {version}: {path.name} -> {root}")
    (code_dir / "changes.txt").write_text("\n".join(report) + "\n", encoding="utf-8")
    return scripts


# ---------------------------------------------------------------------------
# Running
# ---------------------------------------------------------------------------

def log_call(output_root: Path, row: dict) -> None:
    path = output_root / "run_log.csv"
    frame = pd.DataFrame([row])
    try:
        frame.to_csv(path, mode="a", header=not path.exists(), index=False, encoding="utf-8-sig")
    except PermissionError:
        frame.to_csv(path.with_name(f"run_log_{datetime.now():%Y%m%d_%H%M%S}.csv"), index=False, encoding="utf-8-sig")


def keep_call_files(root: Path, dataset: str) -> None:
    """failed_members.csv / run_manifest.json of a one-dataset call, kept per dataset."""
    for name in ("failed_members.csv", "run_manifest.json"):
        source = root / name
        if source.exists():
            shutil.copy2(source, root / f"{source.stem}_{DATASET_LABELS[dataset]}{source.suffix}")


def call(command: List[str], output_root: Path, run_id: str, step: str, plan: bool) -> int:
    print(f"\n{'=' * 78}\n[{run_id} {step}] {' '.join(shlex.quote(c) for c in command)}\n{'=' * 78}", flush=True)
    if plan:
        return 0
    start = datetime.now()
    result = subprocess.run(command)
    end = datetime.now()
    log_call(output_root, {"Run": run_id, "Step": step, "Start": f"{start:%Y-%m-%d %H:%M:%S}",
                           "End": f"{end:%Y-%m-%d %H:%M:%S}", "Minutes": round((end - start).total_seconds() / 60, 2),
                           "Exit_Code": result.returncode, "Command": " ".join(command)})
    print(f"[{run_id} {step}] exit code {result.returncode} after {(end - start).total_seconds() / 60:.1f} min", flush=True)
    return result.returncode


def run_commands(run: tuple, script: Path, args: argparse.Namespace, others: List[str]) -> Tuple[List[str], List[str]]:
    run_id, band_label, band, version = run
    base = [sys.executable, str(script)]
    extra = list(VERSIONS[version]["options"]) + others
    extra += shlex.split(args.args_261007 if version == "261007" else args.args_261013)
    if version == "261013" and "--reference-root" not in extra:
        partner = next(r for r in RUNS if r[1] == band_label and r[3] == "261007")
        extra += ["--reference-root", str(run_folder(Path(args.output_root), partner[0], partner[1], partner[3]))]
    resume = [] if args.no_resume or "--resume" in extra else ["--resume"]
    return base + resume + extra, base + ["--summary-only"] + extra


def execute(args: argparse.Namespace, others: List[str], runs: Sequence[tuple], scripts: Dict[str, Path]) -> int:
    status = 0
    output_root = Path(args.output_root)
    for run in runs:
        run_id, band_label, band, version = run
        train, summary = run_commands(run, scripts[run_id], args, others)
        root = run_folder(output_root, run_id, band_label, version)
        print(f"\n### Run {run_id}: PPG {band_label.replace('Hz', ' Hz')} + {version} -> {root}", flush=True)
        for dataset in DATASET_RUN_ORDER:
            code = call(train + ["--datasets", dataset], output_root, run_id, DATASET_LABELS[dataset], args.plan)
            status = status or code
            if not args.plan:
                keep_call_files(root, dataset)
        status = status or call(summary + ["--datasets", *DATASET_RUN_ORDER], output_root, run_id, "summary", args.plan)
        if version == "261007":
            # the 261013 run of the same band compares with this run: its summary again, now that this run exists
            partner = next((r for r in RUNS if r[1] == band_label and r[3] == "261013"), None)
            if partner and partner[0] in scripts:
                _, partner_summary = run_commands(partner, scripts[partner[0]], args, others)
                call(partner_summary + ["--datasets", *DATASET_RUN_ORDER], output_root, partner[0],
                     "summary vs 261007", args.plan)
    return status


# ---------------------------------------------------------------------------
# Comparison
# ---------------------------------------------------------------------------

def same_subject(a, b) -> bool:
    a, b = str(a).strip(), str(b).strip()
    return a == b or (a.isdigit() and b.isdigit() and int(a) == int(b))


def is_issue(dataset: str, subject) -> bool:
    return any(same_subject(subject, x) for x in EXCLUDED.get(dataset, ()))


def read_csv(path: Path) -> pd.DataFrame:
    try:
        return pd.read_csv(path, dtype={"Subject": str}) if path.exists() else pd.DataFrame()
    except Exception as exc:
        print(f"  {path} not read ({exc})")
        return pd.DataFrame()


def run_tables(output_root: Path) -> List[dict]:
    """Every run of this folder and the earlier raw-input runs: label, band, version, per-subject and overall tables."""
    sources = [(run_folder(output_root, r[0], r[1], r[3]), r[0], r[1].replace("Hz", " Hz"), r[3]) for r in RUNS]
    sources += [(WORKSPACE / v["raw_run"], "raw", "raw (no filter)", version) for version, v in VERSIONS.items()]
    out = []
    for root, run_id, band, version in sources:
        per_subject = read_csv(root / "Summary" / "per_subject_fft.csv")
        if per_subject.empty:
            continue
        if "Row" in per_subject.columns:  # 261007: the full model's ensemble
            per_subject = per_subject[per_subject["Row"].astype(str) == "full"]
        overall = read_csv(root / "Summary" / "fft_davies_overall.csv")
        incomplete = read_csv(root / "Summary" / "incomplete_folds.csv")
        out.append({"root": root, "run": run_id, "band": band, "version": version,
                    "per_subject": per_subject, "overall": overall, "incomplete": incomplete})
    return out


def describe(values: np.ndarray) -> Dict[str, float]:
    values = values[np.isfinite(values)]
    if not values.size:
        return {"Mean": math.nan, "SD": math.nan, "Median": math.nan, "Q1": math.nan, "Q3": math.nan}
    return {"Mean": float(values.mean()), "SD": float(values.std(ddof=1)) if values.size > 1 else math.nan,
            "Median": float(np.median(values)), "Q1": float(np.percentile(values, 25)),
            "Q3": float(np.percentile(values, 75))}


def bh(p: Sequence[float]) -> List[float]:
    p = np.asarray(p, float)
    out = np.full(p.size, np.nan)
    ok = np.flatnonzero(np.isfinite(p))
    if ok.size:
        order = ok[np.argsort(p[ok])]
        ranked = p[order] * ok.size / (np.arange(ok.size) + 1)
        out[order] = np.minimum(1.0, np.minimum.accumulate(ranked[::-1])[::-1])
    return out.tolist()


def wilcoxon_p(a: np.ndarray, b: np.ndarray) -> float:
    from scipy.stats import wilcoxon
    d = a - b
    if d.size < 2 or np.allclose(d, 0):
        return math.nan
    try:
        return float(wilcoxon(a, b).pvalue)
    except ValueError:
        return math.nan


def compare(output_root: Path) -> Optional[Path]:
    runs = run_tables(output_root)
    if not runs:
        print("Comparison: no finished run (no Summary\\per_subject_fft.csv) yet")
        return None
    rows = []
    for r in runs:
        frame = r["per_subject"].copy()
        frame.insert(0, "Input_Band", r["band"])
        frame.insert(0, "Version", r["version"])
        frame.insert(0, "Run", r["run"])
        rows.append(frame)
    per_subject = pd.concat(rows, ignore_index=True)
    per_subject["Issue_Subject"] = [is_issue(d, s) for d, s in zip(per_subject["Dataset"], per_subject["Subject"])]
    keys = ["Dataset", "Version", "Input_Band", "Window_Config"]
    summary_rows = []
    for subject_set, part in (("All subjects", per_subject), ("Issue subjects left out", per_subject[~per_subject["Issue_Subject"]])):
        for key, group in part.groupby(keys, sort=False):
            mae, pcc = describe(group["FFT_MAE"].to_numpy(float)), describe(group["Wave_PCC_Mean"].to_numpy(float))
            summary_rows.append({"Subject_Set": subject_set, **dict(zip(keys, key)),
                                 "Run": group["Run"].iloc[0], "Subjects": int(group["Subject"].nunique()),
                                 **{f"FFT_MAE_{k}": v for k, v in mae.items()},
                                 **{f"PCC_{k}": v for k, v in pcc.items()},
                                 "FFT_MAE_Text": f"{mae['Mean']:.2f} ± {mae['SD']:.2f} ({mae['Median']:.2f} [{mae['Q1']:.2f}, {mae['Q3']:.2f}])",
                                 "PCC_Text": f"{pcc['Mean']:.3f} ± {pcc['SD']:.3f} ({pcc['Median']:.3f})"})
    # order of the request: STEAM2, BIDMC, CapnoBase; 261013 before 261007; 0.1-0.8, 0.1-8.0 Hz, raw
    order = {"Dataset": {DATASET_LABELS[d]: i for i, d in enumerate(DATASET_RUN_ORDER)},
             "Version": {v: i for i, v in enumerate(VERSIONS)},
             "Input_Band": {"0.1-0.8 Hz": 0, "0.1-8.0 Hz": 1, "raw (no filter)": 2}}
    summary = pd.DataFrame(summary_rows)
    summary = summary.sort_values(["Subject_Set", "Dataset", "Version", "Window_Config", "Input_Band"],
                                  key=lambda s: s.map(order[s.name]).fillna(99) if s.name in order else s,
                                  kind="stable").reset_index(drop=True)
    tests = []
    pairs = (("0.1-0.8 Hz", "0.1-8.0 Hz"), ("0.1-0.8 Hz", "raw (no filter)"), ("0.1-8.0 Hz", "raw (no filter)"))
    for subject_set, part in (("All subjects", per_subject), ("Issue subjects left out", per_subject[~per_subject["Issue_Subject"]])):
        for (dataset, version, window), group in part.groupby(["Dataset", "Version", "Window_Config"], sort=False):
            for band_a, band_b in pairs:
                a = group[group["Input_Band"] == band_a].set_index("Subject")
                b = group[group["Input_Band"] == band_b].set_index("Subject")
                common = a.index.intersection(b.index)
                if len(common) < 2:
                    continue
                for metric, better_low in (("FFT_MAE", True), ("Wave_PCC_Mean", False)):
                    x, y = a.loc[common, metric].to_numpy(float), b.loc[common, metric].to_numpy(float)
                    ok = np.isfinite(x) & np.isfinite(y)
                    x, y = x[ok], y[ok]
                    tests.append({"Subject_Set": subject_set, "Dataset": dataset, "Version": version,
                                  "Window_Config": window, "Metric": metric, "A": band_a, "B": band_b,
                                  "Paired_Subjects": int(x.size), "Median_A": float(np.median(x)) if x.size else math.nan,
                                  "Median_B": float(np.median(y)) if y.size else math.nan,
                                  "Median_Difference_A_minus_B": float(np.median(x - y)) if x.size else math.nan,
                                  "A_Better": int(np.sum(x < y) if better_low else np.sum(x > y)),
                                  "B_Better": int(np.sum(y < x) if better_low else np.sum(y > x)),
                                  "Wilcoxon_p": wilcoxon_p(x, y)})
    tests = pd.DataFrame(tests)
    if not tests.empty:
        tests["BH_p"] = np.nan
        for _, idx in tests.groupby(["Subject_Set", "Metric"]).groups.items():
            tests.loc[idx, "BH_p"] = bh(tests.loc[idx, "Wilcoxon_p"].tolist())
        tests = tests.sort_values(["Subject_Set", "Metric", "Dataset", "Version", "Window_Config"],
                                  key=lambda s: s.map(order[s.name]).fillna(99) if s.name in order else s,
                                  kind="stable").reset_index(drop=True)
    overall = []
    for r in runs:
        if not r["overall"].empty:
            frame = r["overall"].copy()
            frame.insert(0, "Input_Band", r["band"])
            frame.insert(0, "Version", r["version"])
            frame.insert(0, "Run", r["run"])
            overall.append(frame)
    status = []
    for r in runs:
        for (dataset, window), group in r["per_subject"].groupby(["Dataset", "Window_Config"]):
            inc = r["incomplete"]
            n_inc = 0
            if not inc.empty and {"Dataset", "Window_Config"} <= set(inc.columns):
                keep = (inc["Dataset"] == dataset) & (inc["Window_Config"] == window)
                if "Variant" in inc.columns:  # 261007: the full model only
                    keep &= inc["Variant"].astype(str) == "full"
                n_inc = int(keep.sum())
            status.append({"Run": r["run"], "Version": r["version"], "Input_Band": r["band"], "Folder": str(r["root"]),
                           "Dataset": dataset, "Window_Config": window, "Subjects": int(group["Subject"].nunique()),
                           "Incomplete_Folds": n_inc})
    notes = pd.DataFrame([
        ("Runs", "01 0.1-0.8 Hz + 261013, 02 0.1-0.8 Hz + 261007, 03 0.1-8.0 Hz + 261013, 04 0.1-8.0 Hz + 261007 "
                 f"(folders {output_root}\\NN_PPG_<band>_<version>); raw = the earlier runs with the raw PPG input "
                 "(261007_RRWaveNet_NoEvent_v1, 261013_RRWavNet_STEAM_BIDMC_v1), when found"),
        ("Input", f"PPG band-passed by sos_bandpass of the base script ({FILTER_ORDER}rd-order Butterworth, zero phase) "
                  "over the whole segment; everything else as the base runs"),
        ("Per subject", "Summary\\per_subject_fft.csv of each run: FFT_MAE (mean |FFT rate - reference rate| over the "
                        "subject's scored minutes) and Wave_PCC_Mean; 261007: the full model's ensemble (Row = full); "
                        "261013: the final ensemble"),
        ("Issue subjects", "; ".join(f"{d} {', '.join(s)}" for d, s in EXCLUDED.items())),
        ("Paired_Tests", "same subjects in both conditions; Wilcoxon signed-rank; BH over all tests of a subject set and "
                         "metric; A_Better = subjects with lower FFT_MAE / higher PCC in A"),
        ("Overall", "fft_davies_overall.csv of each run (pooled over minutes, every row of its summary)"),
    ], columns=["Item", "Description"])
    sheets = {"Summary": summary, "Paired_Tests": tests,
              "Overall": pd.concat(overall, ignore_index=True) if overall else pd.DataFrame(),
              "Per_Subject": per_subject, "Run_Status": pd.DataFrame(status), "Notes": notes}
    out_dir = output_root / "Comparison"
    out_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")

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

    for name in ("Summary", "Paired_Tests", "Per_Subject"):
        write(out_dir / f"{name}.csv", lambda p, f=sheets[name]: f.to_csv(p, index=False, encoding="utf-8-sig"))
    path = write(out_dir / f"{NAME}_Comparison.xlsx", xlsx)
    show = summary[(summary["Subject_Set"] == "Issue subjects left out")]
    if not show.empty:
        pd.set_option("display.width", 220)
        print("\nFFT MAE / PCC per dataset, version and input band (issue subjects left out):")
        print(show[["Dataset", "Version", "Window_Config", "Input_Band", "Subjects", "FFT_MAE_Text",
                    "PCC_Text"]].to_string(index=False))
    print(f"\nComparison: {path}")
    return path


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args(argv: Optional[Sequence[str]] = None) -> Tuple[argparse.Namespace, List[str]]:
    parser = argparse.ArgumentParser(description="RRWaveNet 261013 / 261007 with the PPG input band-passed 0.1-0.8 Hz "
                                                 "and 0.1-8.0 Hz on STEAM2, BIDMC and CapnoBase, and the comparison")
    parser.add_argument("--runs", nargs="+", choices=[r[0] for r in RUNS], default=[r[0] for r in RUNS])
    parser.add_argument("--scripts-dir", type=Path, default=None, help="folder of the two base scripts")
    parser.add_argument("--output-root", type=Path, default=OUTPUT_ROOT)
    parser.add_argument("--args-261007", default="", help='options only for the 261007 runs, e.g. "--windows w10s_nonoverlap"')
    parser.add_argument("--args-261013", default="", help='options only for the 261013 runs, e.g. "--members 3"')
    parser.add_argument("--no-resume", action="store_true", help="do not pass --resume (train finished models again)")
    parser.add_argument("--write-only", action="store_true", help="write the four scripts, do not run")
    parser.add_argument("--plan", action="store_true", help="print every call, do not run")
    parser.add_argument("--compare-only", action="store_true", help="only the comparison of the finished runs")
    # --args-261007 "--windows ...": a value starting with "--" would be read as an option; join it
    tokens, joined = list(sys.argv[1:] if argv is None else argv), []
    while tokens:
        token = tokens.pop(0)
        if token in ("--args-261007", "--args-261013") and tokens:
            token = f"{token}={tokens.pop(0)}"
        joined.append(token)
    return parser.parse_known_args(joined)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args, others = parse_args(argv)
    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    if args.compare_only:
        compare(output_root)
        return 0
    runs = [r for r in RUNS if r[0] in args.runs]
    print(f"{NAME}: {', '.join(f'{r[0]} {r[1]} + {r[3]}' for r in runs)}; datasets "
          f"{' -> '.join(DATASET_LABELS[d] for d in DATASET_RUN_ORDER)}\nResults: {output_root}")
    scripts = write_scripts(args, runs)
    if args.write_only:
        print(f"\nWritten only (--write-only): {output_root / 'Code'}")
        return 0
    status = execute(args, others, runs, scripts)
    if not args.plan:
        compare(output_root)
    return status


if __name__ == "__main__":
    sys.exit(main())
