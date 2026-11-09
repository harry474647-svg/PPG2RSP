# -*- coding: utf-8 -*-
r"""261109_RRWaveNet_TargetBand: the RRWaveNet runs 261013 and 261007 with only the
filter of the waveform target (the answer RSP) changed from 0.1-0.5 Hz to 0.1-0.8 Hz,
LOSOCV on STEAM2 and BIDMC, and the comparison with the earlier runs (target
0.1-0.5 Hz). Every result is saved under one folder.

What changes (only this)
  The waveform the model learns and is tested against (training and validation
  loss, the test targets, the waveform PCC) is the reference RSP band-passed
  0.1-0.8 Hz instead of 0.1-0.5 Hz: the same filter of the base script
  (bandpass_filter: 3rd-order Butterworth, zero phase, over the whole segment),
  only the upper edge differs, with the sign the base script chose (BIDMC and
  261007 STEAM2: from the 0.1-0.5 Hz PPG and reference, as before; 261013 STEAM2:
  as recorded). Per-window min-max scaling to -1..1 as before.
What stays the same (everything else)
  PPG input (raw, as the base runs), model, loss (0.45 SmoothL1 + 0.45 Pearson +
  0.10 derivative), training, seeds, ensembles, validation, weight decay
  selection (261013), LOSOCV, the 0.1-0.5 Hz reference used for the breath onsets
  (ground-truth rate), the quality maps (training windows), the sign choice, the
  scored minutes, the 0.1-0.5 Hz filter of the predicted waveform before its FFT
  rate and the 6-30 bpm rate band. So the FFT MAE of a subject is scored on the
  same minutes against the same ground truth as in the earlier runs, and the
  difference comes from the target alone. The waveform PCC is against the target
  of the run (0.1-0.8 Hz here, 0.1-0.5 Hz in the earlier runs).
CapnoBase is not run by default: its target is the respiratory-phase waveform of
  the expert breath labels and its band is 0.05 Hz to 0.5-1.0 Hz (capnography
  profile); the 0.1-0.5 Hz filter is not used there, so this change does not
  touch it (--datasets steam2 bidmc capnobase runs it anyway, unchanged).

Runs, in this order (each: STEAM2 -> BIDMC)
  01  RSP target 0.1-0.8 Hz + 261013 (subject-wise validation, weight-decay selection, ensemble of 3)
  02  RSP target 0.1-0.8 Hz + 261007 (temporal validation, full model, ensemble of 3)
Each run is the code of the base script with exactly these text changes (each
must match exactly once, otherwise nothing runs):
  1. after RESP_BAND_HZ = (0.1, 0.5): TARGET_BAND_HZ = (0.1, 0.8)
  2. in preprocess_segment, before the PPG quality map: for the adult profiles
     (fixed 0.1-0.5 Hz band, no lag) the waveform target becomes
     sign x bandpass_filter(RSP, 0.1, 0.8)
  3. PROJECT_ROOT -> the run's folder below
How a run is executed: one call of the run's script per dataset (STEAM2, then
BIDMC) with --resume (finished models are kept, interrupted ones continue), then
one --summary-only call over the datasets; 261007 runs the full model only
(--variants full) and its default windows (10 s, 20 s); 261013 its default (10 s,
weight decays 0.01 / 0.1 / 1); the paired comparison of 261013 with 261007 inside
the 261013 summary reads run 02 (written again after run 02).

Base scripts (searched in --scripts-dir, the folder of this file, the current
folder, D:\PPG2RSP_RRWaveNet_Inspired, its run folders and the parent folder):
  261013_RRWaveNet_Adult_SubjectVal_WD.py, 261007_RRWaveNet_Ensemble_NoEvent.py
Their text (results-folder line left out) is compared with the versions this code
was checked against; another version is reported and still used.

Outputs (D:\PPG2RSP_RRWaveNet_Inspired\261109_RRWaveNet_TargetBand)
  Code\                                   the two scripts that run, changes.txt
  01_RSP_0.1-0.8Hz_261013\, 02_RSP_0.1-0.8Hz_261007\
                                          results of every run, in the layout of the base run
  run_log.csv, Logs\<run>_<step>.log      every call (start, end, minutes, exit code) and its output
  Comparison\261109_RRWaveNet_TargetBand_Comparison.xlsx (+ csv)
    Summary        dataset x version x target band x window: subjects, FFT MAE and
                   waveform PCC (mean +- SD, median [IQR]); all subjects and issue
                   subjects left out (STEAM2 22_LMS, 26_CKK, 12_YJJ; BIDMC 27)
    Paired_Tests   target 0.1-0.8 Hz vs the earlier run (0.1-0.5 Hz) per dataset, version,
                   window: paired subjects, median difference, subjects better, Wilcoxon p,
                   BH-adjusted p, and the check that both are scored on the same minutes
                   with the same ground truth (Same_Minutes_And_GT)
    Overall, Per_Subject, Run_Status, Notes
  The earlier runs are read only: D:\PPG2RSP_RRWaveNet_Inspired\261013_RRWavNet_STEAM_BIDMC_v1
  and 261007_RRWaveNet_NoEvent_v1 (Summary\per_subject_fft.csv).

Run (needs CUDA as the base scripts):
  python 261109_RRWaveNet_TargetBand.py
  python 261109_RRWaveNet_TargetBand.py --plan            print the calls only
  python 261109_RRWaveNet_TargetBand.py --compare-only    only the comparison of what is finished
  python 261109_RRWaveNet_TargetBand.py --runs 02         only this run
Spyder: runfile(r'D:\PPG2RSP_RRWaveNet_Inspired\...\261109_RRWaveNet_TargetBand.py', args='--plan')
Other options go to every script call unchanged (e.g. --gpu 1, --no-require-cuda).
Options of this file: --runs, --datasets, --scripts-dir, --output-root,
  --args-261007, --args-261013, --no-resume, --write-only, --plan, --compare-only
"""

import argparse
import hashlib
import math
import os
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

NAME = "261109_RRWaveNet_TargetBand"
WORKSPACE = Path(r"D:\PPG2RSP_RRWaveNet_Inspired")
OUTPUT_ROOT = WORKSPACE / NAME
TARGET_BAND = (0.1, 0.8)
TARGET_LABEL = "0.1-0.8 Hz"
EARLIER_LABEL = "0.1-0.5 Hz (earlier run)"
DATASET_ORDER = ("steam2", "bidmc", "capnobase")  # the order of the calls
DEFAULT_DATASETS = ("steam2", "bidmc")
DATASET_LABELS = {"steam2": "STEAM2", "bidmc": "BIDMC", "capnobase": "CapnoBase"}
EXCLUDED = {"STEAM2": ("22_LMS", "26_CKK", "12_YJJ"), "BIDMC": ("27",), "CapnoBase": ("0115",)}

# version -> base script, earlier run folder (target 0.1-0.5 Hz), SHA-256 of the checked base text (CRLF -> LF,
# PROJECT_ROOT line left out), options of the version
VERSIONS = {
    "261013": {
        "script": "261013_RRWaveNet_Adult_SubjectVal_WD.py",
        "earlier_run": "261013_RRWavNet_STEAM_BIDMC_v1",
        "sha256": "364b109b1e0da1f7ed6ea4a9276a77ed0ad554363b32605cb6484a37642ca3dc",
        "options": [],
    },
    "261007": {
        "script": "261007_RRWaveNet_Ensemble_NoEvent.py",
        "earlier_run": "261007_RRWaveNet_NoEvent_v1",
        "sha256": "18ed5840118f660f7c80ef6c7ed34744597675ba9ebd8115703aa265739f1841",
        "options": ["--variants", "full"],
    },
}
RUNS = (("01", "261013"), ("02", "261007"))  # the requested order

BAND_LINE = "RESP_BAND_HZ = (0.1, 0.5)\n"
TARGET_ANCHOR = "    beat_distance = (\n"
ROOT_PATTERN = re.compile(r'^PROJECT_ROOT = Path\(r"[^"\n]*"\)$', re.MULTILINE)


# ---------------------------------------------------------------------------
# The scripts of the runs
# ---------------------------------------------------------------------------

def run_folder(output_root: Path, run_id: str, version: str) -> Path:
    return output_root / f"{run_id}_RSP_{TARGET_BAND[0]:g}-{TARGET_BAND[1]:g}Hz_{version}"


def base_hash(text: str) -> str:
    return hashlib.sha256(ROOT_PATTERN.sub("PROJECT_ROOT = <root>", text).encode("utf-8")).hexdigest()


def find_script(name: str, folders: Sequence[Optional[Path]]) -> Optional[Path]:
    here = Path(__file__).resolve().parent
    candidates = [Path(f) for f in folders if f]
    candidates += [here, Path.cwd(), WORKSPACE] + [WORKSPACE / v["earlier_run"] for v in VERSIONS.values()]
    candidates += [here.parent]
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


def patched_text(text: str, base_name: str, root: Path) -> str:
    """The base script with exactly the three changes; stops when one does not match once."""
    text = text.replace("\r\n", "\n")
    for old, label in ((BAND_LINE, "RESP_BAND_HZ line"), (TARGET_ANCHOR, "beat_distance line of preprocess_segment")):
        count = text.count(old)
        if count != 1:
            raise SystemExit(f"{base_name}: the {label} was found {count} times (expected once); nothing is run")
    if len(ROOT_PATTERN.findall(text)) != 1:
        raise SystemExit(f"{base_name}: PROJECT_ROOT line not found once; nothing is run")
    band_new = (BAND_LINE +
                f"TARGET_BAND_HZ = ({TARGET_BAND[0]!r}, {TARGET_BAND[1]!r})  # {NAME}: band of the waveform target "
                "(the reference for breaths, quality and evaluation stays RESP_BAND_HZ)\n")
    target_new = (
        '    if not profile["adaptive_upper"] and tuple(band) == tuple(RESP_BAND_HZ) and lag == 0:\n'
        f"        # {NAME}: the waveform target (training / validation loss, test targets, PCC) is the\n"
        "        # reference band-passed TARGET_BAND_HZ with the same filter and the sign chosen above; the\n"
        "        # 0.1-0.5 Hz reference (rsp_low) still gives the breaths, quality maps and scored minutes\n"
        "        target_wide = sign * bandpass_filter(rsp_raw_arr, fs, TARGET_BAND_HZ[0], TARGET_BAND_HZ[1])[0]\n"
        "        target_signal = np.asarray(target_wide, dtype=np.asarray(target_signal).dtype)\n"
        + TARGET_ANCHOR)
    text = text.replace(BAND_LINE, band_new).replace(TARGET_ANCHOR, target_new)
    root_line = f'PROJECT_ROOT = Path(r"{root}")'
    # a function as replacement: the backslashes of a Windows path are kept as they are (no regex escapes)
    text = ROOT_PATTERN.sub(lambda _: root_line, text)
    head = (f"# {NAME}: generated from {base_name}; the only changes: the waveform target is the RSP band-passed "
            f"{TARGET_BAND[0]:g}-{TARGET_BAND[1]:g} Hz (TARGET_BAND_HZ, preprocess_segment) and PROJECT_ROOT = {root}\n")
    if text.startswith("# -*- coding"):
        first, rest = text.split("\n", 1)
        return f"{first}\n{head}{rest}"
    return head + text


def write_scripts(args: argparse.Namespace, runs: Sequence[tuple]) -> Dict[str, Path]:
    code_dir = Path(args.output_root) / "Code"
    code_dir.mkdir(parents=True, exist_ok=True)
    report = [f"{NAME} ({datetime.now():%Y-%m-%d %H:%M:%S})",
              "Changes in each script: 1. TARGET_BAND_HZ after RESP_BAND_HZ  2. waveform target in preprocess_segment"
              "  3. PROJECT_ROOT", ""]
    scripts = {}
    for run_id, version in runs:
        base = find_script(VERSIONS[version]["script"], [args.scripts_dir])
        if base is None:
            raise SystemExit(f"{VERSIONS[version]['script']} not found (give its folder with --scripts-dir)")
        text = base.read_text(encoding="utf-8").replace("\r\n", "\n")
        digest = base_hash(text)
        same = digest == VERSIONS[version]["sha256"]
        line = (f"[{version}] base script {base}; SHA-256 {digest} - "
                + ("same as the checked version" if same else
                   "DIFFERENT from the version this code was checked against (still used)"))
        print(line)
        report.append(line)
        root = run_folder(Path(args.output_root), run_id, version)
        path = code_dir / f"{NAME}_{run_id}_RSP_{TARGET_BAND[0]:g}-{TARGET_BAND[1]:g}Hz_{version}.py"
        # open(..., newline="\n"): Path.write_text has no newline argument before Python 3.10
        with open(path, "w", encoding="utf-8", newline="\n") as handle:
            handle.write(patched_text(text, base.name, root))
        scripts[run_id] = path
        report.append(f"[{run_id}] target {TARGET_LABEL}, {version}: {path.name} -> {root}")
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


def keep_call_files(root: Path, dataset: str, since: float) -> None:
    """failed_members.csv / run_manifest.json written by a one-dataset call, kept per dataset."""
    for name in ("failed_members.csv", "run_manifest.json"):
        source = root / name
        if source.exists() and source.stat().st_mtime >= since - 1:
            shutil.copy2(source, root / f"{source.stem}_{DATASET_LABELS[dataset]}{source.suffix}")


def command_text(command: Sequence[str]) -> str:
    return subprocess.list2cmdline(list(command)) if os.name == "nt" else " ".join(shlex.quote(c) for c in command)


def split_options(text: str) -> List[str]:
    """--args-261007 / --args-261013 into tokens; on Windows the backslashes of paths are kept."""
    if os.name != "nt":
        return shlex.split(text)
    tokens = shlex.split(text, posix=False)
    return [tok[1:-1] if len(tok) >= 2 and tok[0] == tok[-1] and tok[0] in "\"'" else tok for tok in tokens]


def call(command: List[str], output_root: Path, run_id: str, step: str, plan: bool) -> int:
    """Run one script call; its output goes to this console (also in Spyder) and to Logs\\<run>_<step>.log."""
    print(f"\n{'=' * 78}\n[{run_id} {step}] {command_text(command)}\n{'=' * 78}", flush=True)
    if plan:
        return 0
    log_dir = output_root / "Logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / f"{run_id}_{step.replace(' ', '_')}.log"
    env = dict(os.environ, PYTHONUNBUFFERED="1", PYTHONIOENCODING="utf-8")
    start = datetime.now()
    with open(log_path, "a", encoding="utf-8") as log, subprocess.Popen(
            command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, env=env,
            text=True, encoding="utf-8", errors="replace", bufsize=1) as process:
        log.write(f"\n# {start:%Y-%m-%d %H:%M:%S} {command_text(command)}\n")
        try:
            for line in process.stdout:
                print(line, end="", flush=True)
                log.write(line)
                log.flush()
            returncode = process.wait()
        except BaseException:  # interrupted (Ctrl+C / Spyder stop): the script call stops too
            process.kill()
            raise
    end = datetime.now()
    log_call(output_root, {"Run": run_id, "Step": step, "Start": f"{start:%Y-%m-%d %H:%M:%S}",
                           "End": f"{end:%Y-%m-%d %H:%M:%S}", "Minutes": round((end - start).total_seconds() / 60, 2),
                           "Exit_Code": returncode, "Log": str(log_path), "Command": command_text(command)})
    print(f"[{run_id} {step}] exit code {returncode} after {(end - start).total_seconds() / 60:.1f} min", flush=True)
    return returncode


def run_commands(run: tuple, script: Path, args: argparse.Namespace, others: List[str]) -> Tuple[List[str], List[str]]:
    run_id, version = run
    base = [sys.executable, str(script)]
    extra = list(VERSIONS[version]["options"]) + others
    extra += split_options(args.args_261007 if version == "261007" else args.args_261013)
    if version == "261013" and "--reference-root" not in extra:
        partner = next(r for r in RUNS if r[1] == "261007")
        extra += ["--reference-root", str(run_folder(Path(args.output_root), *partner))]
    resume = [] if args.no_resume or "--resume" in extra else ["--resume"]
    return base + resume + extra, base + ["--summary-only"] + extra


def execute(args: argparse.Namespace, others: List[str], runs: Sequence[tuple], scripts: Dict[str, Path]) -> int:
    status = 0
    output_root = Path(args.output_root)
    datasets = [d for d in DATASET_ORDER if d in args.datasets]
    for run in runs:
        run_id, version = run
        train, summary = run_commands(run, scripts[run_id], args, others)
        root = run_folder(output_root, run_id, version)
        print(f"\n### Run {run_id}: RSP target {TARGET_LABEL} + {version} -> {root}", flush=True)
        for dataset in datasets:
            since = datetime.now().timestamp()
            code = call(train + ["--datasets", dataset], output_root, run_id, DATASET_LABELS[dataset], args.plan)
            status = status or code
            if not args.plan:
                keep_call_files(root, dataset, since)
        status = status or call(summary + ["--datasets", *datasets], output_root, run_id, "summary", args.plan)
        if version == "261007":
            # the 261013 run compares with this run in its summary: its summary again, now that this run exists
            partner = next((r for r in RUNS if r[1] == "261013"), None)
            if partner and partner[0] in scripts:
                _, partner_summary = run_commands(partner, scripts[partner[0]], args, others)
                call(partner_summary + ["--datasets", *datasets], output_root, partner[0], "summary vs 261007",
                     args.plan)
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
        return pd.read_csv(path, dtype={"Subject": str}, encoding="utf-8-sig") if path.exists() else pd.DataFrame()
    except Exception as exc:
        print(f"  {path} not read ({exc})")
        return pd.DataFrame()


def run_tables(output_root: Path) -> List[dict]:
    """The runs of this folder (target 0.1-0.8 Hz) and the earlier runs (target 0.1-0.5 Hz)."""
    sources = [(run_folder(output_root, run_id, version), run_id, TARGET_LABEL, version) for run_id, version in RUNS]
    sources += [(WORKSPACE / v["earlier_run"], "earlier", EARLIER_LABEL, version) for version, v in VERSIONS.items()]
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
    if not any(r["band"] == TARGET_LABEL for r in runs):
        print("Comparison: no finished run of this folder (no Summary\\per_subject_fft.csv) yet")
        return None
    rows = []
    for r in runs:
        frame = r["per_subject"].copy()
        frame.insert(0, "Target_Band", r["band"])
        frame.insert(0, "Version", r["version"])
        frame.insert(0, "Run", r["run"])
        rows.append(frame)
    per_subject = pd.concat(rows, ignore_index=True)
    per_subject["Issue_Subject"] = [is_issue(d, s) for d, s in zip(per_subject["Dataset"], per_subject["Subject"])]
    # only the datasets and windows run here (the earlier runs may hold more)
    ours = per_subject[per_subject["Target_Band"] == TARGET_LABEL][["Dataset", "Window_Config"]].drop_duplicates()
    per_subject = per_subject.merge(ours, on=["Dataset", "Window_Config"], how="inner")
    subject_sets = (("All subjects", per_subject), ("Issue subjects left out", per_subject[~per_subject["Issue_Subject"]]))
    keys = ["Dataset", "Version", "Target_Band", "Window_Config"]
    order = {"Dataset": {DATASET_LABELS[d]: i for i, d in enumerate(DATASET_ORDER)},
             "Version": {v: i for i, v in enumerate(VERSIONS)},
             "Target_Band": {TARGET_LABEL: 0, EARLIER_LABEL: 1}}

    def ordered(frame: pd.DataFrame, columns: List[str]) -> pd.DataFrame:
        return frame.sort_values(columns, key=lambda s: s.map(order[s.name]).fillna(99) if s.name in order else s,
                                 kind="mergesort").reset_index(drop=True)

    summary_rows = []
    for subject_set, part in subject_sets:
        for key, group in part.groupby(keys, sort=False):
            mae, pcc = describe(group["FFT_MAE"].to_numpy(float)), describe(group["Wave_PCC_Mean"].to_numpy(float))
            summary_rows.append({"Subject_Set": subject_set, **dict(zip(keys, key)),
                                 "Run": group["Run"].iloc[0], "Subjects": int(group["Subject"].nunique()),
                                 **{f"FFT_MAE_{k}": v for k, v in mae.items()},
                                 **{f"PCC_{k}": v for k, v in pcc.items()},
                                 "FFT_MAE_Text": f"{mae['Mean']:.2f} ± {mae['SD']:.2f} ({mae['Median']:.2f} [{mae['Q1']:.2f}, {mae['Q3']:.2f}])",
                                 "PCC_Text": f"{pcc['Mean']:.3f} ± {pcc['SD']:.3f} ({pcc['Median']:.3f})"})
    summary = ordered(pd.DataFrame(summary_rows), ["Subject_Set", "Dataset", "Version", "Window_Config", "Target_Band"])
    tests = []
    for subject_set, part in subject_sets:
        for (dataset, version, window), group in part.groupby(["Dataset", "Version", "Window_Config"], sort=False):
            a = group[group["Target_Band"] == TARGET_LABEL].set_index("Subject")
            b = group[group["Target_Band"] == EARLIER_LABEL].set_index("Subject")
            common = a.index.intersection(b.index)
            if len(common) < 2:
                continue
            # the same minutes and the same ground truth in both runs (only the target differs)
            same = int(sum(
                int(a.at[s, "N_Minutes"]) == int(b.at[s, "N_Minutes"])
                and np.isclose(float(a.at[s, "GT_RR_Median"]), float(b.at[s, "GT_RR_Median"]), atol=1e-6, equal_nan=True)
                for s in common))
            for metric, better_low in (("FFT_MAE", True), ("Wave_PCC_Mean", False)):
                x, y = a.loc[common, metric].to_numpy(float), b.loc[common, metric].to_numpy(float)
                ok = np.isfinite(x) & np.isfinite(y)
                x, y = x[ok], y[ok]
                tests.append({"Subject_Set": subject_set, "Dataset": dataset, "Version": version,
                              "Window_Config": window, "Metric": metric, "A": TARGET_LABEL, "B": EARLIER_LABEL,
                              "Paired_Subjects": int(x.size),
                              "Same_Minutes_And_GT": f"{same}/{len(common)}",
                              "Median_A": float(np.median(x)) if x.size else math.nan,
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
        tests = ordered(tests, ["Subject_Set", "Metric", "Dataset", "Version", "Window_Config"])
    overall = []
    for r in runs:
        if not r["overall"].empty:
            frame = r["overall"].copy()
            frame.insert(0, "Target_Band", r["band"])
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
            status.append({"Run": r["run"], "Version": r["version"], "Target_Band": r["band"], "Folder": str(r["root"]),
                           "Dataset": dataset, "Window_Config": window, "Subjects": int(group["Subject"].nunique()),
                           "Incomplete_Folds": n_inc})
    notes = pd.DataFrame([
        ("Runs", f"01 RSP target {TARGET_LABEL} + 261013, 02 RSP target {TARGET_LABEL} + 261007 "
                 f"(folders {output_root}\\NN_RSP_0.1-0.8Hz_<version>); earlier = the runs with the 0.1-0.5 Hz target "
                 "(261013_RRWavNet_STEAM_BIDMC_v1, 261007_RRWaveNet_NoEvent_v1), read only"),
        ("Change", "only the waveform target: the reference RSP band-passed 0.1-0.8 Hz (bandpass_filter of the base "
                   "script, 3rd-order Butterworth, zero phase) with the same sign; input (raw PPG), model, loss, "
                   "training, ground-truth breaths, scored minutes and the evaluation as the earlier runs"),
        ("FFT_MAE", "scored on the same minutes against the same ground truth in both runs "
                    "(Paired_Tests: Same_Minutes_And_GT)"),
        ("Wave_PCC", "against the target of each run: 0.1-0.8 Hz here, 0.1-0.5 Hz in the earlier runs"),
        ("CapnoBase", "not run by default: its target (expert-label phase waveform, capnography band) does not use "
                      "the 0.1-0.5 Hz filter, so this change does not touch it"),
        ("Per subject", "Summary\\per_subject_fft.csv of each run; 261007: the full model's ensemble (Row = full); "
                        "261013: the final ensemble"),
        ("Issue subjects", "; ".join(f"{d} {', '.join(s)}" for d, s in EXCLUDED.items())),
        ("Paired_Tests", "same subjects in both runs; Wilcoxon signed-rank; BH over all tests of a subject set and "
                         "metric; A_Better = subjects with lower FFT_MAE / higher PCC with the 0.1-0.8 Hz target"),
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
    show = summary[summary["Subject_Set"] == "Issue subjects left out"]
    if not show.empty:
        pd.set_option("display.width", 220)
        print("\nFFT MAE / PCC per dataset, version and target band (issue subjects left out; PCC against each run's target):")
        print(show[["Dataset", "Version", "Window_Config", "Target_Band", "Subjects", "FFT_MAE_Text",
                    "PCC_Text"]].to_string(index=False))
    print(f"\nComparison: {path}")
    return path


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args(argv: Optional[Sequence[str]] = None) -> Tuple[argparse.Namespace, List[str]]:
    parser = argparse.ArgumentParser(description="RRWaveNet 261013 / 261007 with only the RSP target band-passed "
                                                 "0.1-0.8 Hz (instead of 0.1-0.5 Hz), and the comparison")
    parser.add_argument("--runs", nargs="+", choices=[r[0] for r in RUNS], default=[r[0] for r in RUNS])
    parser.add_argument("--datasets", nargs="+", choices=DATASET_ORDER, default=list(DEFAULT_DATASETS),
                        help="run in the order STEAM2, BIDMC, CapnoBase (default: STEAM2 BIDMC)")
    parser.add_argument("--scripts-dir", type=Path, default=None, help="folder of the two base scripts")
    parser.add_argument("--output-root", type=Path, default=OUTPUT_ROOT)
    parser.add_argument("--args-261007", default="", help='options only for the 261007 run, e.g. "--windows w10s_nonoverlap"')
    parser.add_argument("--args-261013", default="", help='options only for the 261013 run, e.g. "--members 3"')
    parser.add_argument("--no-resume", action="store_true", help="do not pass --resume (train finished models again)")
    parser.add_argument("--write-only", action="store_true", help="write the two scripts, do not run")
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
    datasets = [d for d in DATASET_ORDER if d in args.datasets]
    print(f"{NAME}: RSP target {TARGET_LABEL} (was 0.1-0.5 Hz); runs {', '.join(f'{r[0]} {r[1]}' for r in runs)}; "
          f"datasets {' -> '.join(DATASET_LABELS[d] for d in datasets)}\nResults: {output_root}")
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
