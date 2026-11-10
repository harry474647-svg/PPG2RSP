# -*- coding: utf-8 -*-
r"""261110_RRWaveNet_DeepEncoder_B6: the temporal pattern extractor (residual encoder)
of RRWaveNet with six blocks instead of four - B5 (d = 16) and B6 (d = 32) added after
B1-B4 (d = 1, 2, 4, 8) - trained by LOSOCV, then the block-by-block feature analysis
(channel |r| with the target, PC1 respiratory-band power, PC1 autocorrelation) from Stem
to B6 and the comparison with the four-block models. Every result is saved under one folder.

Why new training: the trained models have four blocks; a B5 / B6 does not exist in them,
and blocks added to a trained model would be untrained (random weights). So the models
are trained again with the six-block encoder, everything else as the version.

Runs, in this order (each: STEAM2 -> BIDMC -> CapnoBase)
  01  261013 with six blocks (subject-wise validation, weight decay selection, ensemble of 3)
  02  261007 with six blocks (temporal validation, full model, ensemble of 3; 10 s windows -
      the analysis uses 10 s; --args-261007 "--windows w10s_nonoverlap w20s_nonoverlap" adds 20 s)
  03  261105 with six blocks (unified protocol, final ensemble of 6)
Each run is the code of the version with exactly these text changes (each must match once,
otherwise nothing runs):
  1. "encoder_dilations": (1, 2, 4, 8)  ->  (1, 2, 4, 8, 16, 32)
  2. PROJECT_ROOT -> the run's folder
Model, stem, decoder, loss, training, validation, ensembles, LOSOCV, data, preprocessing and
evaluation are those of the version. Receptive field of the encoder (two kernel-15 convolutions
per block, 64 Hz): B1-B4 421 samples = 6.58 s; with B5 869 = 13.58 s; with B6 1765 = 27.58 s.
A 10 s window has 640 samples, so B5 and B6 reach beyond the window: their outer taps see the
zero padding at the window edges (part of what the analysis shows).

Feature analysis: 261106_Model_FeatureLevel (the analysis of the earlier B1-B4 results) run on
the six-block models, from a generated copy with exactly these changes: the stages Stem, B1-B6
(two more colours), the last block (B6) in the highlighted curves and in the Stem -> last-block
key results, a wider per-block summary figure, its own output name, and the run scripts = the
generated six-block scripts. Its
models, windows, forward pass, measures and statistics are unchanged (inference only).
Comparison: six blocks vs the earlier four-block runs of the same version
(261013_RRWavNet_STEAM_BIDMC_v1, 261007_RRWaveNet_NoEvent_v1, 261105_RRWaveNet_Unified_Optimized;
read only): per subject FFT MAE and waveform PCC, Wilcoxon signed-rank, BH, and the check that
both are scored on the same minutes against the same ground truth.

GPU: the training needs it (the base scripts require CUDA; --no-require-cuda trains on the CPU,
far slower). The feature analysis and the comparison need no GPU (forward passes / tables only;
the analysis uses the GPU when there is one, --force-cpu runs it on the CPU).

Outputs (D:\PPG2RSP_RRWaveNet_Inspired\261110_RRWaveNet_DeepEncoder_B6)
  Code\                         the generated training scripts and analysis copy, changes.txt
  01_261013_B6\, 02_261007_B6\, 03_261105_B6\   LOSOCV results in the layout of each version
  Feature_Analysis\             <run>\Encoder_Feature_Analysis\ (Stem-B6 tables and figures),
                                <run>\Stem_Fusion_PrePost\, Summary\..._results.xlsx, RESULTS.md
  Comparison\261110_RRWaveNet_DeepEncoder_B6_Comparison.xlsx (+ csv)   six vs four blocks
  run_log.csv, Logs\<run>_<step>.log

Run:
  python 261110_RRWaveNet_DeepEncoder_B6.py                   training, analysis, comparison
  python 261110_RRWaveNet_DeepEncoder_B6.py --plan            print the calls only
  python 261110_RRWaveNet_DeepEncoder_B6.py --analysis-only   analysis and comparison of what is trained
  python 261110_RRWaveNet_DeepEncoder_B6.py --compare-only    only the comparison
  python 261110_RRWaveNet_DeepEncoder_B6.py --versions 261013 261007
Spyder: runfile(r'D:\PPG2RSP_RRWaveNet_Inspired\...\261110_RRWaveNet_DeepEncoder_B6.py', args='--plan')
Other options go to every training call unchanged (e.g. --gpu 1, --no-require-cuda); --gpu,
--force-cpu, --threads, --subjects and the data paths also go to the analysis.
Options of this file: --versions, --datasets, --scripts-dir, --output-root, --args-261013,
  --args-261007, --args-261105, --no-resume, --write-only, --plan, --analysis-only, --compare-only
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

NAME = "261110_RRWaveNet_DeepEncoder_B6"
WORKSPACE = Path(r"D:\PPG2RSP_RRWaveNet_Inspired")
OUTPUT_ROOT = WORKSPACE / NAME
DILATIONS = (1, 2, 4, 8, 16, 32)
DATASET_LABELS = {"steam2": "STEAM2", "bidmc": "BIDMC", "capnobase": "CapnoBase"}
DATASET_RUN_ORDER = ("steam2", "bidmc", "capnobase")
EXCLUDED = {"STEAM2": ("22_LMS", "26_CKK", "12_YJJ"), "BIDMC": ("27",), "CapnoBase": ("0115",)}
ANALYSIS_SCRIPT = "261106_Model_FeatureLevel.py"

VERSIONS = {
    "261013": {"script": "261013_RRWaveNet_Adult_SubjectVal_WD.py", "earlier_run": "261013_RRWavNet_STEAM_BIDMC_v1",
               "sha256": "364b109b1e0da1f7ed6ea4a9276a77ed0ad554363b32605cb6484a37642ca3dc", "options": []},
    "261007": {"script": "261007_RRWaveNet_Ensemble_NoEvent.py", "earlier_run": "261007_RRWaveNet_NoEvent_v1",
               "sha256": "18ed5840118f660f7c80ef6c7ed34744597675ba9ebd8115703aa265739f1841",
               "options": ["--variants", "full", "--windows", "w10s_nonoverlap"]},
    "261105": {"script": "261105_RRWaveNet_Unified_Optimized.py", "earlier_run": "261105_RRWaveNet_Unified_Optimized",
               "sha256": "b424bd24d7d15817cfb569d82e862572825a4ee23bcacad0240d5a7ebd2f523f", "options": []},
}
RUNS = (("01", "261013"), ("02", "261007"), ("03", "261105"))  # the order of the runs

ROOT_PATTERN = re.compile(r'^PROJECT_ROOT = Path\(r"[^"\n]*"\)$', re.MULTILINE)
DILATION_LINE = '    "encoder_dilations": (1, 2, 4, 8),\n'
# options of the training calls that the analysis also takes
ANALYSIS_OPTIONS = {"--gpu": 1, "--force-cpu": 0, "--threads": 1, "--subjects": "+", "--capnobase-path": 1,
                    "--bidmc-path": 1, "--steam2-ppg-dir": 1, "--steam2-rsp-dir": 1}


# ---------------------------------------------------------------------------
# Generated scripts
# ---------------------------------------------------------------------------

def run_folder(output_root: Path, run_id: str, version: str) -> Path:
    return output_root / f"{run_id}_{version}_B6"


def script_name(run_id: str, version: str) -> str:
    return f"{NAME}_{run_id}_{version}.py"


def base_hash(text: str) -> str:
    return hashlib.sha256(ROOT_PATTERN.sub("PROJECT_ROOT = <root>", text).encode("utf-8")).hexdigest()


def find_script(name: str, folders: Sequence[Optional[Path]]) -> Optional[Path]:
    here = Path(__file__).resolve().parent
    candidates = [Path(f) for f in folders if f]
    candidates += [here, Path.cwd(), WORKSPACE] + [WORKSPACE / v["earlier_run"] for v in VERSIONS.values()]
    candidates += [WORKSPACE / "261106_Model_FeatureLevel", here.parent]
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


def with_head(text: str, head: str) -> str:
    if text.startswith("# -*- coding"):
        first, rest = text.split("\n", 1)
        return f"{first}\n{head}{rest}"
    return head + text


def patched_training(text: str, base_name: str, root: Path) -> str:
    """The version's code with exactly the two changes; stops when one does not match once."""
    text = text.replace("\r\n", "\n")
    if text.count(DILATION_LINE) != 1:
        raise SystemExit(f"{base_name}: the encoder_dilations line was found {text.count(DILATION_LINE)} times "
                         "(expected once); nothing is run")
    if len(ROOT_PATTERN.findall(text)) != 1:
        raise SystemExit(f"{base_name}: PROJECT_ROOT line not found once; nothing is run")
    text = text.replace(DILATION_LINE, f'    "encoder_dilations": {DILATIONS},  # {NAME}: B5 (d=16) and B6 (d=32) '
                                       "added; encoder receptive field 27.58 s\n")
    root_line = f'PROJECT_ROOT = Path(r"{root}")'
    text = ROOT_PATTERN.sub(lambda _: root_line, text)  # a function: Windows backslashes stay as they are
    return with_head(text, f"# {NAME}: generated from {base_name}; the only changes: encoder_dilations "
                           f"{DILATIONS} (six residual encoder blocks) and PROJECT_ROOT = {root}\n")


def patched_analysis(text: str, base_name: str, versions: Sequence[str], run_ids: Dict[str, str]) -> str:
    """261106_Model_FeatureLevel for the six-block models: stages Stem-B6, the last block B6, its own name and
    the generated scripts; stops when a change does not match."""
    text = text.replace("\r\n", "\n")
    changes = [
        ('STAGES = ("Stem", "B1", "B2", "B3", "B4")\n',
         'STAGES = ("Stem", "B1", "B2", "B3", "B4", "B5", "B6")\n'),
        ('                "B3": "#e15759", "B4": "#9467bd"}\n',
         '                "B3": "#e15759", "B4": "#9467bd", "B5": "#17becf", "B6": "#8c564b"}\n'),
        ('linewidth=2.0 if name in ("Target", "B4") else 1.4',
         'linewidth=2.0 if name in ("Target", "B6") else 1.4'),
        ('axis.plot(lags, acf_curves["B4"], color="#4c78a8", linewidth=1.6, label="DeepEnc (B4)")',
         'axis.plot(lags, acf_curves["B6"], color="#4c78a8", linewidth=1.6, label="DeepEnc (B6)")'),
        ('SCRIPT_NAME = "261106_Model_FeatureLevel"\n', f'SCRIPT_NAME = "{NAME}_FeatureLevel"\n'),
        # the per-block summary figure: wider for seven stages, so that the statistics boxes do not overlap
        ('    fig, axis = plt.subplots(figsize=(6.4, 4.4))\n    axis.plot(range(len(STAGES)), values, "o-", '
         'color="#1f77b4")\n',
         '    fig, axis = plt.subplots(figsize=(max(6.4, 1.4 * len(STAGES)), 4.4))\n    axis.plot(range(len(STAGES)), '
         'values, "o-", color="#1f77b4")\n'),
    ]
    for old, new in changes:
        if text.count(old) != 1:
            raise SystemExit(f"{base_name}: '{old.strip()[:60]}' found {text.count(old)} times (expected once)")
        text = text.replace(old, new)
    for version in versions:  # the run scripts: the generated six-block scripts (next to this copy in Code\)
        old = f'"script": "{VERSIONS[version]["script"]}"'
        if text.count(old) != 1:
            raise SystemExit(f"{base_name}: the script of {version} found {text.count(old)} times (expected once)")
        text = text.replace(old, f'"script": "{script_name(run_ids[version], version)}"')
    marker = "def key_results("
    if text.count(marker) != 1:
        raise SystemExit(f"{base_name}: key_results not found once")
    head, tail = text.split(marker, 1)
    # key results and RESULTS.md: Stem -> the last block (B6 here, B4 in the four-block analysis)
    text = head + marker + tail.replace("B4", "B6")
    return with_head(text, f"# {NAME}: generated from {base_name} for the six-block encoder; the only changes: "
                           "stages Stem-B6 (+2 colours), B6 as the last block in the highlighted curves and the key "
                           "results, SCRIPT_NAME, and the run scripts = the generated six-block scripts\n")


def write_scripts(args: argparse.Namespace, runs: Sequence[tuple]) -> Tuple[Dict[str, Path], Path]:
    code_dir = Path(args.output_root) / "Code"
    code_dir.mkdir(parents=True, exist_ok=True)
    report = [f"{NAME} ({datetime.now():%Y-%m-%d %H:%M:%S})",
              f"Training scripts: 1. encoder_dilations (1, 2, 4, 8) -> {DILATIONS}  2. PROJECT_ROOT", ""]
    scripts = {}
    for run_id, version in runs:
        base = find_script(VERSIONS[version]["script"], [args.scripts_dir])
        if base is None:
            raise SystemExit(f"{VERSIONS[version]['script']} not found (give its folder with --scripts-dir)")
        text = base.read_text(encoding="utf-8").replace("\r\n", "\n")
        digest = base_hash(text)
        line = (f"[{version}] base script {base}; SHA-256 {digest} - "
                + ("same as the checked version" if digest == VERSIONS[version]["sha256"] else
                   "DIFFERENT from the version this code was checked against (still used)"))
        print(line)
        report.append(line)
        root = run_folder(Path(args.output_root), run_id, version)
        path = code_dir / script_name(run_id, version)
        with open(path, "w", encoding="utf-8", newline="\n") as handle:  # newline: Python < 3.10 too
            handle.write(patched_training(text, base.name, root))
        scripts[run_id] = path
        report.append(f"[{run_id}] {version}: {path.name} -> {root}")
    analysis_base = find_script(ANALYSIS_SCRIPT, [args.scripts_dir])
    if analysis_base is None:
        raise SystemExit(f"{ANALYSIS_SCRIPT} not found (give its folder with --scripts-dir)")
    analysis = code_dir / f"{NAME}_FeatureLevel.py"
    with open(analysis, "w", encoding="utf-8", newline="\n") as handle:
        handle.write(patched_analysis(analysis_base.read_text(encoding="utf-8"), analysis_base.name,
                                      [r[1] for r in runs], {r[1]: r[0] for r in runs}))
    report += ["", f"Analysis: {analysis.name} from {analysis_base}: stages Stem-B6, B6 as the last block, "
                   "SCRIPT_NAME, run scripts = the generated six-block scripts"]
    (code_dir / "changes.txt").write_text("\n".join(report) + "\n", encoding="utf-8")
    return scripts, analysis


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
    """--args-<version> into tokens; on Windows the backslashes of paths are kept."""
    if os.name != "nt":
        return shlex.split(text)
    tokens = shlex.split(text, posix=False)
    return [tok[1:-1] if len(tok) >= 2 and tok[0] == tok[-1] and tok[0] in "\"'" else tok for tok in tokens]


def call(command: List[str], output_root: Path, run_id: str, step: str, plan: bool) -> int:
    """Run one call; its output goes to this console (also in Spyder) and to Logs\\<run>_<step>.log."""
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
        except BaseException:  # interrupted (Ctrl+C / Spyder stop): the call stops too
            process.kill()
            raise
    end = datetime.now()
    log_call(output_root, {"Run": run_id, "Step": step, "Start": f"{start:%Y-%m-%d %H:%M:%S}",
                           "End": f"{end:%Y-%m-%d %H:%M:%S}", "Minutes": round((end - start).total_seconds() / 60, 2),
                           "Exit_Code": returncode, "Log": str(log_path), "Command": command_text(command)})
    print(f"[{run_id} {step}] exit code {returncode} after {(end - start).total_seconds() / 60:.1f} min", flush=True)
    return returncode


def version_options(version: str, args: argparse.Namespace, others: List[str]) -> List[str]:
    return list(VERSIONS[version]["options"]) + others + split_options(getattr(args, f"args_{version}"))


def run_commands(run: tuple, script: Path, args: argparse.Namespace, others: List[str]) -> Tuple[List[str], List[str]]:
    run_id, version = run
    base = [sys.executable, str(script)]
    extra = version_options(version, args, others)
    if version == "261013" and "--reference-root" not in extra:  # its paired comparison with the 261007 run here
        extra += ["--reference-root", str(run_folder(Path(args.output_root), "02", "261007"))]
    resume = [] if args.no_resume or "--resume" in extra else ["--resume"]
    return base + resume + extra, base + ["--summary-only"] + extra


def train(args: argparse.Namespace, others: List[str], runs: Sequence[tuple], scripts: Dict[str, Path]) -> int:
    status = 0
    output_root = Path(args.output_root)
    for run in runs:
        run_id, version = run
        root = run_folder(output_root, run_id, version)
        train_cmd, summary_cmd = run_commands(run, scripts[run_id], args, others)
        print(f"\n### Run {run_id}: {version} with six encoder blocks (d = {', '.join(map(str, DILATIONS))}) -> {root}",
              flush=True)
        for dataset in args.datasets:
            since = datetime.now().timestamp()
            code = call(train_cmd + ["--datasets", dataset], output_root, run_id, f"training {DATASET_LABELS[dataset]}",
                        args.plan)
            status = status or code
            if not args.plan:
                keep_call_files(root, dataset, since)
        status = status or call(summary_cmd + ["--datasets", *args.datasets], output_root, run_id, "summary", args.plan)
        if version == "261007" and any(r[1] == "261013" for r in runs):
            # the 261013 summary compares with this run: written again now that it exists
            partner = next(r for r in runs if r[1] == "261013")
            _, partner_summary = run_commands(partner, scripts[partner[0]], args, others)
            call(partner_summary + ["--datasets", *args.datasets], output_root, partner[0], "summary vs 261007",
                 args.plan)
    return status


def analysis_options(others: List[str]) -> List[str]:
    """The options of the training calls that the analysis also takes (--gpu, --force-cpu, ... data paths)."""
    out, tokens = [], list(others)
    while tokens:
        token = tokens.pop(0)
        name = token.split("=", 1)[0]
        if name not in ANALYSIS_OPTIONS:
            continue
        out.append(token)
        if "=" in token or ANALYSIS_OPTIONS[name] == 0:
            continue
        if ANALYSIS_OPTIONS[name] == 1 and tokens:
            out.append(tokens.pop(0))
        else:
            while tokens and not tokens[0].startswith("--"):
                out.append(tokens.pop(0))
    return out


def analyse(args: argparse.Namespace, others: List[str], runs: Sequence[tuple], analysis: Path) -> int:
    output_root = Path(args.output_root)
    command = [sys.executable, str(analysis), "--runs", *[r[1] for r in runs], "--output",
               str(output_root / "Feature_Analysis"), "--datasets", *args.datasets]
    for run_id, version in runs:
        command += [f"--root-{version}", str(run_folder(output_root, run_id, version))]
    return call(command + analysis_options(others), output_root, "all", "feature analysis", args.plan)


# ---------------------------------------------------------------------------
# Comparison: six vs four blocks
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


def wilcoxon_p(a: np.ndarray, b: np.ndarray) -> float:
    from scipy.stats import wilcoxon
    d = a - b
    if d.size < 2 or np.allclose(d, 0):
        return math.nan
    try:
        return float(wilcoxon(a, b).pvalue)
    except ValueError:
        return math.nan


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
    values = values[np.isfinite(values)]
    if not values.size:
        return {"Mean": math.nan, "SD": math.nan, "Median": math.nan, "Q1": math.nan, "Q3": math.nan}
    return {"Mean": float(values.mean()), "SD": float(values.std(ddof=1)) if values.size > 1 else math.nan,
            "Median": float(np.median(values)), "Q1": float(np.percentile(values, 25)),
            "Q3": float(np.percentile(values, 75))}


def compare(output_root: Path, runs: Sequence[tuple]) -> Optional[Path]:
    six, four = "Six blocks (B1-B6, d up to 32)", "Four blocks (earlier run)"
    frames, status = [], []
    for run_id, version in runs:
        for root, label, run in ((run_folder(output_root, run_id, version), six, run_id),
                                 (WORKSPACE / VERSIONS[version]["earlier_run"], four, "earlier")):
            per_subject = read_csv(root / "Summary" / "per_subject_fft.csv")
            if per_subject.empty:
                continue
            if "Row" in per_subject.columns:  # 261007: the full model's ensemble
                per_subject = per_subject[per_subject["Row"].astype(str) == "full"]
            frame = per_subject.copy()
            frame.insert(0, "Encoder", label)
            frame.insert(0, "Version", version)
            frame.insert(0, "Run", run)
            frames.append(frame)
            for (dataset, window), group in per_subject.groupby(["Dataset", "Window_Config"]):
                status.append({"Run": run, "Version": version, "Encoder": label, "Folder": str(root),
                               "Dataset": dataset, "Window_Config": window, "Subjects": int(group["Subject"].nunique())})
    if not any(f["Encoder"].iloc[0] == six for f in frames):
        print("Comparison: no six-block run finished yet (no Summary\\per_subject_fft.csv)")
        return None
    per_subject = pd.concat(frames, ignore_index=True)
    ours = per_subject[per_subject["Encoder"] == six][["Dataset", "Version", "Window_Config"]].drop_duplicates()
    per_subject = per_subject.merge(ours, on=["Dataset", "Version", "Window_Config"], how="inner")
    per_subject["Issue_Subject"] = [is_issue(d, s) for d, s in zip(per_subject["Dataset"], per_subject["Subject"])]
    sets = (("All subjects", per_subject), ("Issue subjects left out", per_subject[~per_subject["Issue_Subject"]]))
    order = {"Dataset": {DATASET_LABELS[d]: i for i, d in enumerate(DATASET_RUN_ORDER)},
             "Version": {v: i for i, (_, v) in enumerate(RUNS)}, "Encoder": {six: 0, four: 1}}

    def ordered(frame: pd.DataFrame, columns: List[str]) -> pd.DataFrame:
        return frame.sort_values(columns, key=lambda s: s.map(order[s.name]).fillna(99) if s.name in order else s,
                                 kind="mergesort").reset_index(drop=True)

    keys = ["Dataset", "Version", "Encoder", "Window_Config"]
    summary_rows, tests = [], []
    for subject_set, part in sets:
        for key, group in part.groupby(keys, sort=False):
            mae, pcc = describe(group["FFT_MAE"].to_numpy(float)), describe(group["Wave_PCC_Mean"].to_numpy(float))
            summary_rows.append({"Subject_Set": subject_set, **dict(zip(keys, key)), "Subjects": int(group["Subject"].nunique()),
                                 **{f"FFT_MAE_{k}": v for k, v in mae.items()}, **{f"PCC_{k}": v for k, v in pcc.items()},
                                 "FFT_MAE_Text": f"{mae['Mean']:.2f} ± {mae['SD']:.2f} ({mae['Median']:.2f} [{mae['Q1']:.2f}, {mae['Q3']:.2f}])",
                                 "PCC_Text": f"{pcc['Mean']:.3f} ± {pcc['SD']:.3f} ({pcc['Median']:.3f})"})
        for (dataset, version, window), group in part.groupby(["Dataset", "Version", "Window_Config"], sort=False):
            a = group[group["Encoder"] == six].set_index("Subject")
            b = group[group["Encoder"] == four].set_index("Subject")
            common = a.index.intersection(b.index)
            if len(common) < 2:
                continue
            same = int(sum(int(a.at[s, "N_Minutes"]) == int(b.at[s, "N_Minutes"]) and
                           np.isclose(float(a.at[s, "GT_RR_Median"]), float(b.at[s, "GT_RR_Median"]), atol=1e-6,
                                      equal_nan=True) for s in common))
            for metric, better_low in (("FFT_MAE", True), ("Wave_PCC_Mean", False)):
                x, y = a.loc[common, metric].to_numpy(float), b.loc[common, metric].to_numpy(float)
                ok = np.isfinite(x) & np.isfinite(y)
                x, y = x[ok], y[ok]
                tests.append({"Subject_Set": subject_set, "Dataset": dataset, "Version": version, "Window_Config": window,
                              "Metric": metric, "A": six, "B": four, "Paired_Subjects": int(x.size),
                              "Same_Minutes_And_GT": f"{same}/{len(common)}",
                              "Median_A": float(np.median(x)) if x.size else math.nan,
                              "Median_B": float(np.median(y)) if y.size else math.nan,
                              "Median_Difference_A_minus_B": float(np.median(x - y)) if x.size else math.nan,
                              "A_Better": int(np.sum(x < y) if better_low else np.sum(x > y)),
                              "B_Better": int(np.sum(y < x) if better_low else np.sum(y > x)),
                              "Wilcoxon_p": wilcoxon_p(x, y)})
    summary = ordered(pd.DataFrame(summary_rows), ["Subject_Set", "Dataset", "Version", "Window_Config", "Encoder"])
    tests = pd.DataFrame(tests)
    if not tests.empty:
        tests["BH_p"] = np.nan
        for _, idx in tests.groupby(["Subject_Set", "Metric"]).groups.items():
            tests.loc[idx, "BH_p"] = bh(tests.loc[idx, "Wilcoxon_p"].tolist())
        tests = ordered(tests, ["Subject_Set", "Metric", "Dataset", "Version", "Window_Config"])
    notes = pd.DataFrame([
        ("Runs", "six-block encoder (d = 1, 2, 4, 8, 16, 32) trained with each version's own LOSOCV; four blocks = "
                 "the earlier runs (261013_RRWavNet_STEAM_BIDMC_v1, 261007_RRWaveNet_NoEvent_v1, "
                 "261105_RRWaveNet_Unified_Optimized), read only; 261013 earlier: BIDMC and STEAM2 only"),
        ("Per subject", "Summary\\per_subject_fft.csv: FFT_MAE, Wave_PCC_Mean; 261007 the full model's ensemble, "
                        "261013 / 261105 the final ensemble"),
        ("Paired_Tests", "same subjects; Wilcoxon signed-rank; BH over all tests of a subject set and metric; "
                         "A_Better = subjects with lower FFT_MAE / higher PCC with six blocks; Same_Minutes_And_GT: "
                         "subjects scored on the same minutes against the same ground truth"),
        ("Feature analysis", "Feature_Analysis\\ (Stem-B6 per block: channel |r|, PC1 respiratory-band power, PC1 ACF)"),
        ("Issue subjects", "; ".join(f"{d} {', '.join(s)}" for d, s in EXCLUDED.items())),
    ], columns=["Item", "Description"])
    sheets = {"Summary": summary, "Paired_Tests": tests, "Per_Subject": per_subject,
              "Run_Status": pd.DataFrame(status), "Notes": notes}
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
        print("\nFFT MAE / PCC, six vs four encoder blocks (issue subjects left out):")
        print(show[["Dataset", "Version", "Window_Config", "Encoder", "Subjects", "FFT_MAE_Text",
                    "PCC_Text"]].to_string(index=False))
    print(f"\nComparison: {path}")
    return path


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args(argv: Optional[Sequence[str]] = None) -> Tuple[argparse.Namespace, List[str]]:
    parser = argparse.ArgumentParser(description="RRWaveNet with a six-block encoder (B5 d=16, B6 d=32): training, "
                                                 "block-by-block feature analysis and comparison with four blocks")
    parser.add_argument("--versions", nargs="+", choices=list(VERSIONS), default=[v for _, v in RUNS])
    parser.add_argument("--datasets", nargs="+", choices=list(DATASET_LABELS), default=list(DATASET_RUN_ORDER))
    parser.add_argument("--scripts-dir", type=Path, default=None,
                        help="folder of the version scripts and of 261106_Model_FeatureLevel.py")
    parser.add_argument("--output-root", type=Path, default=OUTPUT_ROOT)
    for version in VERSIONS:
        parser.add_argument(f"--args-{version}", dest=f"args_{version}", default="",
                            help=f"options only for the {version} training")
    parser.add_argument("--no-resume", action="store_true", help="do not pass --resume to the training calls")
    parser.add_argument("--write-only", action="store_true", help="write the scripts, do not run")
    parser.add_argument("--plan", action="store_true", help="print every call, do not run")
    parser.add_argument("--analysis-only", action="store_true", help="feature analysis and comparison, no training")
    parser.add_argument("--compare-only", action="store_true", help="only the comparison of the finished runs")
    # --args-<version> "--windows ...": a value starting with "--" would be read as an option; join it
    tokens, joined = list(sys.argv[1:] if argv is None else argv), []
    while tokens:
        token = tokens.pop(0)
        if token in tuple(f"--args-{v}" for v in VERSIONS) and tokens:
            token = f"{token}={tokens.pop(0)}"
        joined.append(token)
    args, others = parser.parse_known_args(joined)
    args.datasets = [d for d in DATASET_RUN_ORDER if d in args.datasets]  # STEAM2 -> BIDMC -> CapnoBase
    return args, others


def main(argv: Optional[Sequence[str]] = None) -> int:
    args, others = parse_args(argv)
    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    runs = [r for r in RUNS if r[1] in args.versions]
    if args.compare_only:
        compare(output_root, runs)
        return 0
    print(f"{NAME}: encoder d = {DILATIONS}; runs {', '.join(f'{r[0]} {r[1]}' for r in runs)}; datasets "
          f"{' -> '.join(DATASET_LABELS[d] for d in args.datasets)}\nResults: {output_root}")
    scripts, analysis = write_scripts(args, runs)
    if args.write_only:
        print(f"\nWritten only (--write-only): {output_root / 'Code'}")
        return 0
    status = 0 if args.analysis_only else train(args, others, runs, scripts)
    status = analyse(args, others, runs, analysis) or status
    if not args.plan:
        compare(output_root, runs)
    return status


if __name__ == "__main__":
    sys.exit(main())
