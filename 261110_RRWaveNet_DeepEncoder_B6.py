# -*- coding: utf-8 -*-
r"""261110_RRWaveNet_DeepEncoder_B6: the temporal pattern extractor (residual encoder)
of RRWaveNet with six blocks instead of four - B5 (d = 16) and B6 (d = 32) added after
B1-B4 (d = 1, 2, 4, 8) - trained by LOSOCV with the 261013 and 261007 code, then the
block-by-block feature analysis (channel |r| with the target, PC1 respiratory-band power,
PC1 autocorrelation) from Stem to B6, next to the same analysis of the earlier four-block
models (Stem to B4) of 261013 and 261007, and the comparison of the results. Every result
is saved under one folder. This file needs only the 261013 and 261007 scripts.

Why new training: the trained models have four blocks; a B5 / B6 does not exist in them,
and blocks added to a trained model would be untrained (random weights). So the models
are trained again with the six-block encoder, everything else as the version. The earlier
four-block models are used as they are (read only) for the Stem-B4 reference.

Runs, in this order (each: STEAM2 -> BIDMC -> CapnoBase)
  01  261013 with six blocks (subject-wise validation, weight decay selection, ensemble of 3)
  02  261007 with six blocks (temporal validation, full model, ensemble of 3; 10 s windows -
      the analysis uses 10 s; --args-261007 "--windows w10s_nonoverlap w20s_nonoverlap" adds 20 s)
Each run is the code of the version with exactly these text changes (each must match once,
otherwise nothing runs):
  1. "encoder_dilations": (1, 2, 4, 8)  ->  (1, 2, 4, 8, 16, 32)
  2. PROJECT_ROOT -> the run's folder
Model, stem, decoder, loss, training, validation, ensembles, LOSOCV, data, preprocessing and
evaluation are those of the version. Receptive field of the encoder (two kernel-15 convolutions
per block, 64 Hz): B1-B4 421 samples = 6.58 s; with B5 869 = 13.58 s; with B6 1765 = 27.58 s.
A 10 s window has 640 samples, so B5 and B6 reach beyond the window: their outer taps see the
zero padding at the window edges (part of what the analysis shows).

Feature analysis (in this file; inference only, nothing is trained)
  Models   six blocks: the runs 01 / 02 above; four blocks: the earlier runs
           261013_RRWavNet_STEAM_BIDMC_v1 and 261007_RRWaveNet_NoEvent_v1 (read only).
           261013: m0-m2 of the weight decay selected in the fold (WD_Selection);
           261007: full, m0-m2. A subject is analysed when all three models are stored.
  Data     the 10 s test windows of the held-out subject, read and preprocessed by the
           version's own code (imported; the six- and four-block scripts preprocess alike).
  Forward  Stem = stem_fuse(concatenated stem branches) (the encoder input), then every
           encoder block: B1 ... B6 (six) or B1 ... B4 (four). Checks per model: the saved
           test windows (segment IDs, starts, fs, window, model key, targets) match, the Stem
           tensor equals the model's fused_representation, and the output is compared with
           the saved prediction (Max_Abs_Diff_Saved_Prediction).
  Measures (as in the earlier B1-B4 analysis, 261106_Model_FeatureLevel part B)
    - channel |r|: |Pearson r| of every channel with the target per window, mean over
      windows, mean over the 24 channels; Better_Channels_of_24 = channels with higher |r|
      than at Stem;
    - PC1 respiratory-band ratio: PC1 of the standardized channels (all test windows of the
      subject), periodogram (Hann), power 0.1-0.5 Hz / power above 0 Hz;
    - PC1 ACF r: autocorrelation of PC1 and of the target over contiguous windows (runs
      >= 24 s, mean removed per run, lag 0 = 1), Pearson r of the two over a band of lags.
      Two settings in one run (the ACF is computed once, up to 12 s), each written as a
      complete set to its own subfolder of Feature_Analysis\:
        ACF_1-5s_lag0-10s    r over lags 1-5 s, ACF drawn 0-10 s
        ACF_2-10s_lag0-12s   r over lags 2-10 s, ACF drawn 0-12 s (as in 261106)
      Channel |r|, PSD and PC1 variance share are the same in both;
    - PC1 variance share.
  Values are averaged over the subject's models. Statistics: two-sided paired Wilcoxon
  signed-rank over subjects, per dataset and for all datasets: each block vs Stem and vs the
  block before it; six blocks B6 vs B4 (what B5 and B6 add); six vs four blocks (Stem vs Stem,
  B4 vs B4, last block B6 vs B4) on the subjects analysed in both. p values not adjusted.
Comparison: six blocks vs the earlier four-block runs of the same version: per subject FFT MAE
and waveform PCC, Wilcoxon signed-rank, BH, and the check that both are scored on the same
minutes against the same ground truth.

GPU: the training needs it (the base scripts require CUDA; --no-require-cuda trains on the CPU,
far slower). The feature analysis and the comparison need no GPU (forward passes / tables only;
the analysis uses the GPU when there is one, --force-cpu runs it on the CPU).

Outputs (D:\PPG2RSP_RRWaveNet_Inspired\261110_RRWaveNet_DeepEncoder_B6)
  Code\                         the generated training scripts, changes.txt
  01_261013_B6\, 02_261007_B6\  LOSOCV results in the layout of each version
  Feature_Analysis\ACF_1-5s_lag0-10s\, Feature_Analysis\ACF_2-10s_lag0-12s\   one complete set each:
                                261110_RRWaveNet_DeepEncoder_B6_Feature_Analysis.xlsx, RESULTS.md,
                                key_results.csv, block_tests.csv, six_vs_four_tests.csv,
                                subjects.csv, channels.csv, models.csv, coverage.csv,
                                skipped_subjects.csv, curves.npz,
                                Figures\<version>_<Dataset>_block_summary.png (six and four blocks),
                                Figures\<version>_<Dataset>_<six|four>_blocks_pc1_curves.png
  Comparison\261110_RRWaveNet_DeepEncoder_B6_Comparison.xlsx (+ csv)   six vs four blocks
  run_log.csv, Logs\<run>_<step>.log

Run:
  python 261110_RRWaveNet_DeepEncoder_B6.py                   training, analysis, comparison
  python 261110_RRWaveNet_DeepEncoder_B6.py --plan            print the calls only
  python 261110_RRWaveNet_DeepEncoder_B6.py --analysis-only   analysis and comparison of what is trained
                                                              (the earlier four-block models: right away)
  python 261110_RRWaveNet_DeepEncoder_B6.py --compare-only    only the comparison
  python 261110_RRWaveNet_DeepEncoder_B6.py --versions 261013
Spyder: runfile(r'D:\PPG2RSP_RRWaveNet_Inspired\...\261110_RRWaveNet_DeepEncoder_B6.py', args='--plan')
Other options go to every training call unchanged (e.g. --gpu 1, --no-require-cuda); --gpu,
--force-cpu, --threads, --subjects and the data paths also go to the analysis.
Options of this file: --versions, --datasets, --scripts-dir, --output-root, --args-261013,
  --args-261007, --earlier-root-261013, --earlier-root-261007, --no-resume, --write-only, --plan,
  --analysis-only, --compare-only
"""

import argparse
import hashlib
import importlib.util
import json
import math
import os
import re
import shlex
import shutil
import subprocess
import sys
import time
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

VERSIONS = {
    "261013": {"script": "261013_RRWaveNet_Adult_SubjectVal_WD.py", "earlier_run": "261013_RRWavNet_STEAM_BIDMC_v1",
               "sha256": "364b109b1e0da1f7ed6ea4a9276a77ed0ad554363b32605cb6484a37642ca3dc", "options": [],
               "models": "selected"},
    "261007": {"script": "261007_RRWaveNet_Ensemble_NoEvent.py", "earlier_run": "261007_RRWaveNet_NoEvent_v1",
               "sha256": "18ed5840118f660f7c80ef6c7ed34744597675ba9ebd8115703aa265739f1841",
               "options": ["--variants", "full", "--windows", "w10s_nonoverlap"], "models": "full"},
}
RUNS = (("01", "261013"), ("02", "261007"))  # the order of the runs

ROOT_PATTERN = re.compile(r'^PROJECT_ROOT = Path\(r"[^"\n]*"\)$', re.MULTILINE)
DILATION_LINE = '    "encoder_dilations": (1, 2, 4, 8),\n'
# options of the training calls that the analysis also takes
ANALYSIS_OPTIONS = {"--gpu": 1, "--force-cpu": 0, "--threads": 1, "--subjects": "+", "--capnobase-path": 1,
                    "--bidmc-path": 1, "--steam2-ppg-dir": 1, "--steam2-rsp-dir": 1}

# Feature analysis (the definitions of the earlier B1-B4 analysis)
ANALYSIS_FOLDER = "Feature_Analysis"
WINDOW = "w10s_nonoverlap"
WINDOW_SEC = 10.0
MEMBERS = 3
RESP_BAND_HZ = (0.1, 0.5)
# ACF analyses, each written to its own subfolder of Feature_Analysis:
# (subfolder, lags of the ACF r in s, largest lag drawn in s)
ACF_SETTINGS = (("ACF_1-5s_lag0-10s", (1.0, 5.0), 10.0),
                ("ACF_2-10s_lag0-12s", (2.0, 10.0), 12.0))
ACF_MAX_LAG_SEC = max(setting[2] for setting in ACF_SETTINGS)  # the ACF is computed once, up to this lag
ACF_MIN_RUN_SEC = 24.0
PSD_PLOT_MAX_HZ = 1.0
ALL_DATASETS = "All datasets"
SIX, FOUR = "Six blocks", "Four blocks"
ENCODERS = {SIX: DILATIONS, FOUR: DILATIONS[:4]}
ENCODER_TAGS = {SIX: "six", FOUR: "four"}
MEASURES = {
    "Channel_AbsR_Mean": "Mean |channel r| with the target",
    "RespBand_Ratio": "PC1 resp-band power ratio (0.1-0.5 Hz)",
    "ACF_r": "PC1 ACF similarity to the target (r, lags {band})",  # {band}: the lags of the ACF setting
    "PC1_Variance_Share": "PC1 variance share",
}
FIGURE_MEASURES = ("Channel_AbsR_Mean", "RespBand_Ratio", "ACF_r")
ENCODER_COLORS = {SIX: "#2a78d6", FOUR: "#eb6834"}  # categorical slots 1 and 2 (validated pair)
STAGE_RAMP = ("#86b6ef", "#5598e7", "#3987e5", "#2a78d6", "#256abf", "#184f95", "#0d366b")  # Stem -> B6
TARGET_COLOR = "#0b0b0b"
TEXT_SECONDARY = "#52514e"
GRID_COLOR = "#e6e6e3"


# ---------------------------------------------------------------------------
# Generated scripts
# ---------------------------------------------------------------------------

def run_folder(output_root: Path, run_id: str, version: str) -> Path:
    return output_root / f"{run_id}_{version}_B6"


def script_name(run_id: str, version: str) -> str:
    return f"{NAME}_{run_id}_{version}.py"


def earlier_root(args: argparse.Namespace, version: str) -> Path:
    given = getattr(args, f"earlier_root_{version}", None)
    return Path(given) if given else WORKSPACE / VERSIONS[version]["earlier_run"]


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


def base_script(args: argparse.Namespace, version: str) -> Path:
    path = find_script(VERSIONS[version]["script"], [args.scripts_dir])
    if path is None:
        raise SystemExit(f"{VERSIONS[version]['script']} not found (give its folder with --scripts-dir)")
    return path


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


def write_scripts(args: argparse.Namespace, runs: Sequence[tuple]) -> Dict[str, Path]:
    code_dir = Path(args.output_root) / "Code"
    code_dir.mkdir(parents=True, exist_ok=True)
    report = [f"{NAME} ({datetime.now():%Y-%m-%d %H:%M:%S})",
              f"Training scripts: 1. encoder_dilations (1, 2, 4, 8) -> {DILATIONS}  2. PROJECT_ROOT", ""]
    scripts = {}
    for run_id, version in runs:
        base = base_script(args, version)
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
        report.append(f"[{run_id}] {version}: four-block reference (analysis) {earlier_root(args, version)}, "
                      f"models built with {base.name}")
    report += ["", "Feature analysis: in this file (no other analysis script needed)"]
    with open(code_dir / "changes.txt", "w", encoding="utf-8", newline="\n") as handle:
        handle.write("\n".join(report) + "\n")
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


def analyse(args: argparse.Namespace, others: List[str], runs: Sequence[tuple]) -> int:
    """The feature analysis in its own process (this file with --analysis-worker): its output is logged like a
    training call, and the memory of the models and data is released when it ends."""
    output_root = Path(args.output_root)
    command = [sys.executable, str(Path(__file__).resolve()), "--analysis-worker", "--output-root", str(output_root),
               "--versions", *[r[1] for r in runs], "--datasets", *args.datasets]
    if args.scripts_dir:
        command += ["--scripts-dir", str(args.scripts_dir)]
    for version in VERSIONS:
        if getattr(args, f"earlier_root_{version}", None):
            command += [f"--earlier-root-{version}", str(getattr(args, f"earlier_root_{version}"))]
    return call(command + analysis_options(others), output_root, "all", "feature analysis", args.plan)


# ---------------------------------------------------------------------------
# Feature analysis: models, measures, forward pass
# ---------------------------------------------------------------------------

def stage_names(blocks: int) -> List[str]:
    return ["Stem"] + [f"B{i + 1}" for i in range(blocks)]


def stage_label(stage: str) -> str:
    """Stem, or B5 (d=16, RF 13.6 s): the block's dilation and the encoder receptive field up to it."""
    if stage == "Stem":
        return "Stem"
    index = int(stage[1:])
    field = 1 + 2 * 14 * sum(DILATIONS[:index])  # two kernel-15 convolutions per block
    return f"{stage}\nd={DILATIONS[index - 1]}\n{field / 64.0:.1f} s"


def read_json(path: Path) -> Optional[dict]:
    try:
        return json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


def load_module(path: Path, name: str):
    """Import a version script as a module (nothing is trained)."""
    spec = importlib.util.spec_from_file_location(name, str(path))
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def window_spec(module):
    return next(s for s in module.WINDOW_SPECS if s.name == WINDOW)


def result_tags(version: str, module, root: Path, dataset_key: str) -> List[str]:
    """Subject tags (safe_tag) with a stored fold."""
    spec = window_spec(module)
    if VERSIONS[version]["models"] == "full":
        folder = module.condition_root(root, "full", spec, dataset_key) / "Members" / "m0" / "Models"
        return sorted(p.name[:-len("_best.pth")] for p in folder.glob("*_best.pth"))
    folder = root / "WD_Selection" / f"{spec.order:02d}_{spec.name}" / module.DATASET_CONFIGS[dataset_key]["label"]
    return sorted(p.name[:-len("_selection.json")] for p in folder.glob("*_selection.json"))


def model_list(version: str, module, root: Path, dataset_key: str, subject: str) -> Tuple[List[dict], str]:
    """The models of the subject's reported (ensemble) result; ([], reason) when the fold is not complete."""
    spec = window_spec(module)
    if VERSIONS[version]["models"] == "full":
        variant = module.VARIANTS["full"]
    else:
        record = read_json(module.selection_path(root, spec, dataset_key, subject))
        if record is None or "Selected_Weight_Decay" not in record:
            return [], "no weight decay selection (WD_Selection) for the fold"
        variant = next(iter(module.build_variants([float(record["Selected_Weight_Decay"])]).values()))
    models, missing = [], []
    tag = module.safe_tag(subject)
    for member in range(MEMBERS):
        condition = module.condition_root(root, variant.name, spec, dataset_key)
        checkpoint = condition / "Members" / f"m{member}" / "Models" / f"{tag}_best.pth"
        summary_path, _, prediction = module.member_paths(condition, subject, member)
        summary = read_json(summary_path)
        problems = []
        if not checkpoint.is_file():
            problems.append("checkpoint")
        if summary is None:
            problems.append("member summary")
        elif summary.get("Model") != variant.options.key:
            problems.append("summary of another model")
        if not prediction.is_file():
            problems.append("saved test outputs")
        models.append({"Variant": variant.name, "Member": member, "Model": f"{variant.name}/m{member}",
                       "variant": variant, "Checkpoint": checkpoint, "Prediction": prediction,
                       "Ready": not problems, "Missing": "; ".join(problems)})
        if problems:
            missing.append(f"{variant.name}/m{member}: {', '.join(problems)}")
    if missing:
        return models, "incomplete models: " + " | ".join(missing)
    return models, ""


def public(item: dict) -> dict:
    return {k: (str(v) if isinstance(v, Path) else v) for k, v in item.items() if k != "variant"}


def wilcoxon_paired(a, b) -> float:
    from scipy import stats
    a, b = np.asarray(a, float), np.asarray(b, float)
    keep = np.isfinite(a) & np.isfinite(b)
    a, b = a[keep], b[keep]
    if len(a) < 2:
        return np.nan
    if np.allclose(a, b):
        return 1.0
    try:
        return float(stats.wilcoxon(a, b).pvalue)
    except ValueError:
        return np.nan


def p_text(p_value: float) -> str:
    if not np.isfinite(p_value):
        return "p=n/a"
    return "p<1e-4" if p_value < 1e-4 else f"p={p_value:.2g}"


def stars(p_value: float) -> str:
    if not np.isfinite(p_value):
        return "n/a"
    for threshold, mark in ((1e-4, "****"), (1e-3, "***"), (1e-2, "**"), (0.05, "*")):
        if p_value < threshold:
            return mark
    return "ns"


def sd(values) -> float:
    values = np.asarray(values, float)
    values = values[np.isfinite(values)]
    return float(np.std(values, ddof=1)) if values.size > 1 else np.nan


def channel_abs_r(features: np.ndarray, target: np.ndarray) -> np.ndarray:
    """|Pearson r| of every channel with the target, averaged over windows."""
    f = features.astype(np.float64) - features.mean(axis=2, keepdims=True)
    t = target.astype(np.float64) - target.mean(axis=1, keepdims=True)
    numerator = np.einsum("nct,nt->nc", f, t)
    denominator = np.sqrt(np.einsum("nct,nct->nc", f, f) * np.einsum("nt,nt->n", t, t)[:, None])
    with np.errstate(divide="ignore", invalid="ignore"):
        r = np.where(denominator > 1e-12, numerator / denominator, np.nan)
    return np.nanmean(np.abs(r), axis=0)


def first_component(features: np.ndarray) -> Tuple[np.ndarray, float]:
    """PC1 of the standardized channels: (windows, samples), variance share."""
    windows, channels, samples = features.shape
    matrix = np.transpose(features, (1, 0, 2)).reshape(channels, -1).astype(np.float64)
    mean = matrix.mean(axis=1, keepdims=True)
    channel_sd = matrix.std(axis=1, keepdims=True)
    keep = channel_sd[:, 0] > 1e-8
    z = (matrix[keep] - mean[keep]) / channel_sd[keep]
    covariance = z @ z.T / max(1, z.shape[1] - 1)
    eigenvalues, eigenvectors = np.linalg.eigh(covariance)
    share = float(eigenvalues[-1] / max(eigenvalues.sum(), 1e-12))
    return (eigenvectors[:, -1] @ z).reshape(windows, samples), share


def mean_psd(signals: np.ndarray, fs: float) -> Tuple[np.ndarray, np.ndarray]:
    from scipy.signal import periodogram
    frequencies, power = periodogram(signals, fs=fs, window="hann", detrend="constant", axis=-1)
    return frequencies, power.mean(axis=0)


def band_ratio(frequencies: np.ndarray, power: np.ndarray) -> float:
    band = (frequencies >= RESP_BAND_HZ[0]) & (frequencies <= RESP_BAND_HZ[1])
    total = power[frequencies > 0].sum()
    return float(power[band].sum() / total) if total > 0 else np.nan


def contiguous_windows(metadata: Sequence[dict], samples: int) -> List[List[int]]:
    """Runs of consecutive test windows (same segment, next start = previous start + window)."""
    runs: List[List[int]] = []
    for index, item in enumerate(metadata):
        if runs:
            previous = metadata[runs[-1][-1]]
            if item["segment_id"] == previous["segment_id"] and item["start"] == previous["start"] + samples:
                runs[-1].append(index)
                continue
        runs.append([index])
    return runs


def mean_acf(signals: np.ndarray, runs: List[List[int]], max_lag: int, min_length: int) -> np.ndarray:
    total = np.zeros(max_lag + 1)
    weight = 0.0
    for run in runs:
        x = signals[run].reshape(-1).astype(np.float64)
        if x.size < max(min_length, max_lag + 1):
            continue
        x = x - x.mean()
        energy = float(np.dot(x, x))
        if energy <= 1e-12:
            continue
        spectrum = np.fft.rfft(x, n=2 * x.size)
        acf = np.fft.irfft(spectrum * np.conj(spectrum))[: max_lag + 1] / energy
        total += acf * x.size
        weight += x.size
    return total / weight if weight > 0 else np.full(max_lag + 1, np.nan)


def acf_similarity(acf: np.ndarray, target_acf: np.ndarray, fs: float, lag_band: Tuple[float, float]) -> float:
    """Pearson r of the two autocorrelations over the lags of lag_band (s)."""
    lags = np.arange(acf.size) / fs
    band = (lags >= lag_band[0]) & (lags <= lag_band[1])
    a, b = acf[band], target_acf[band]
    if not (np.isfinite(a).all() and np.isfinite(b).all()) or np.std(a) == 0 or np.std(b) == 0:
        return np.nan
    return float(np.corrcoef(a, b)[0, 1])


def check_saved_windows(stored, metadata: Sequence[dict], fs: int, samples: int, model_key: str) -> str:
    """'' when the saved test outputs belong to these windows and this model."""
    ids = np.array([item["segment_id"] for item in metadata], dtype=np.int64)
    starts = np.array([item["start"] for item in metadata], dtype=np.int64)
    if not (np.array_equal(stored["segment_ids"], ids) and np.array_equal(stored["window_starts"], starts)):
        return "saved test-window IDs / starts differ from the reloaded data"
    if (int(stored["fs"]) != fs or float(stored["input_window_sec"]) != WINDOW_SEC
            or float(stored["input_stride_sec"]) != WINDOW_SEC or str(stored["model_key"].item()) != model_key
            or stored["targets"].shape != (len(metadata), samples)):
        return "saved outputs are not those of this 10 s model"
    return ""


def forward_features(model, loader, device, blocks: int, saved_targets: np.ndarray,
                     saved_predictions: Optional[np.ndarray]):
    """Stem (stem fusion output) and every encoder block, the targets, and the largest difference between the
    model output and the saved prediction."""
    import torch
    stages: List[List[np.ndarray]] = [[] for _ in range(blocks + 1)]
    targets = []
    max_diff = 0.0
    offset = 0
    with torch.no_grad():
        for batch in loader:
            x, y = batch[0].to(device), batch[1]
            features = model.stem_fuse(torch.cat([branch(x) for branch in model.stem_branches], dim=1))
            if offset == 0 and not torch.allclose(features, model.fused_representation(x), atol=1e-6, rtol=1e-5):
                raise ValueError("stem features differ from the model's fused_representation")
            size = len(x)
            target = y.squeeze(1).numpy()
            if not np.allclose(target, saved_targets[offset:offset + size], rtol=1e-5, atol=1e-6, equal_nan=True):
                raise ValueError("targets differ from the saved test windows")
            stages[0].append(features.cpu().numpy().astype(np.float32))
            for index, block in enumerate(model.encoder, start=1):
                features = block(features)
                stages[index].append(features.cpu().numpy().astype(np.float32))
            if saved_predictions is not None:
                output = model(x).squeeze(1).cpu().numpy()
                saved = np.asarray(saved_predictions[offset:offset + size]).reshape(output.shape)
                max_diff = max(max_diff, float(np.nanmax(np.abs(output - saved))))
            targets.append(target.astype(np.float32))
            offset += size
    if offset != len(saved_targets):
        raise ValueError("not every saved test window was processed")
    return ([np.concatenate(chunks) for chunks in stages], np.concatenate(targets),
            max_diff if saved_predictions is not None else np.nan)


def analyse_encoder(version: str, encoder: str, module, root: Path, dataset_key: str, data: dict, tags: List[str],
                    options: argparse.Namespace, device, store: dict) -> int:
    """All subjects of one dataset for one model set (six or four blocks); returns the subjects analysed."""
    import torch
    label = DATASET_LABELS[dataset_key]
    blocks = len(ENCODERS[encoder])
    stages = stage_names(blocks)
    spec = window_spec(module)
    fs = int(module.TARGET_FS)
    samples = int(round(spec.window_sec * fs))
    max_lag = int(round(ACF_MAX_LAG_SEC * fs))
    min_length = int(round(ACF_MIN_RUN_SEC * fs))
    by_tag = {module.safe_tag(subject): subject for subject in data}
    base = {"Version": version, "Encoder": encoder, "Dataset": label}
    for tag in tags:
        if tag not in by_tag:
            store["skipped"].append({**base, "Subject": tag, "Reason": "raw subject unavailable"})
    subjects = [by_tag[tag] for tag in tags if tag in by_tag]
    analysed = 0
    for number, subject in enumerate(subjects, start=1):
        started = time.perf_counter()
        key = {**base, "Subject": subject}
        models, reason = model_list(version, module, root, dataset_key, subject)
        if reason:
            store["skipped"].append({**key, "Reason": reason})
            store["models"] += [{**key, **public(m)} for m in models]
            print(f"[skip] {version} {encoder} {label} {subject}: {reason}", flush=True)
            continue
        metadata = module.test_metadata(data, subject)
        loader = module.make_eval_loader(module.SubjectWindowDataset(data, [subject], "test", False),
                                         options.batch_size, device)
        runs = contiguous_windows(metadata, samples)
        rows, channel_rows, model_rows, curves = [], [], [], {}
        try:
            for item in models:
                variant = item["variant"]
                model = variant.options.build().to(device)
                if len(model.encoder) != blocks:
                    raise ValueError(f"the model has {len(model.encoder)} encoder blocks, expected {blocks}")
                if getattr(model, "stem_fusion", "conv1x1") != "conv1x1":
                    raise ValueError("the model has no 1x1 + GroupNorm stem fusion")
                model.load_state_dict(torch.load(item["Checkpoint"], map_location=device))  # strict: same blocks
                model.eval()
                with np.load(item["Prediction"], allow_pickle=False) as stored:
                    problem = check_saved_windows(stored, metadata, fs, samples, variant.options.key)
                    if problem:
                        raise ValueError(problem)
                    saved_targets = stored["targets"]
                    saved_predictions = stored["predictions"] if "predictions" in stored.files else None
                features, target, max_diff = forward_features(model, loader, device, blocks, saved_targets,
                                                              saved_predictions)
                del model
                model_rows.append({**key, **public(item), "N_Windows": int(len(target)),
                                   "Max_Abs_Diff_Saved_Prediction": max_diff})
                frequencies, target_psd = mean_psd(target, fs)
                target_acf = mean_acf(target, runs, max_lag, min_length)
                curves.setdefault("Target_PSD", []).append(target_psd)
                curves.setdefault("Target_ACF", []).append(target_acf)
                row = {**key, "Model": item["Model"], "N_Windows": int(len(target)),
                       "Target_RespBand_Ratio": band_ratio(frequencies, target_psd)}
                for stage, stage_features in zip(stages, features):
                    abs_r = channel_abs_r(stage_features, target)
                    channel_rows += [{**key, "Model": item["Model"], "Stage": stage, "Channel": channel,
                                      "Abs_r": float(value)} for channel, value in enumerate(abs_r)]
                    pc1, share = first_component(stage_features)
                    _, psd = mean_psd(pc1, fs)
                    acf = mean_acf(pc1, runs, max_lag, min_length)
                    curves.setdefault(f"{stage}_PSD", []).append(psd)
                    curves.setdefault(f"{stage}_ACF", []).append(acf)
                    row[f"{stage}_Channel_AbsR_Mean"] = float(np.nanmean(abs_r))
                    row[f"{stage}_PC1_Variance_Share"] = share
                    row[f"{stage}_RespBand_Ratio"] = band_ratio(frequencies, psd)
                    for setting, lag_band, _ in ACF_SETTINGS:  # the ACF r of every setting
                        row[acf_column(stage, setting)] = acf_similarity(acf, target_acf, fs, lag_band)
                rows.append(row)
                store["frequencies"] = frequencies
                store["fs"] = fs
        except Exception as exc:  # one subject must not stop a long run
            store["skipped"].append({**key, "Reason": f"{type(exc).__name__}: {exc}"})
            print(f"[skip] {version} {encoder} {label} {subject}: {exc}", flush=True)
            continue
        finally:
            del loader
        store["rows"] += rows
        store["channels"] += channel_rows
        store["models"] += model_rows
        store["curves"][(version, encoder, label, subject)] = {name: np.mean(values, axis=0)
                                                              for name, values in curves.items()}
        analysed += 1
        print(f"[{number}/{len(subjects)}] {version} {encoder} {label} {subject}: {len(metadata)} windows x "
              f"{len(models)} models ({time.perf_counter() - started:.1f} s)", flush=True)
    return analysed


def worker_options(tokens: List[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(prog=f"{NAME} --analysis-worker")
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--force-cpu", action="store_true")
    parser.add_argument("--threads", type=int, default=None)
    parser.add_argument("--subjects", nargs="+", default=None)
    parser.add_argument("--batch-size", type=int, default=48)
    for option in ("capnobase-path", "bidmc-path", "steam2-ppg-dir", "steam2-rsp-dir"):
        parser.add_argument(f"--{option}", default=None)
    options, _ = parser.parse_known_args(tokens)
    return options


def set_data_paths(module, options: argparse.Namespace) -> None:
    for dataset_key, key, option in (("capnobase", "data_path", "capnobase_path"), ("bidmc", "data_path", "bidmc_path"),
                                     ("steam2", "ppg_dir", "steam2_ppg_dir"), ("steam2", "rsp_dir", "steam2_rsp_dir")):
        if getattr(options, option):
            module.DATASET_CONFIGS[dataset_key][key] = str(Path(getattr(options, option)).resolve())


def analysis_worker(args: argparse.Namespace, others: List[str]) -> int:
    """Block-by-block feature analysis of the six-block models (Stem-B6) and the earlier four-block models (Stem-B4)."""
    sys.dont_write_bytecode = True  # no __pycache__ next to the imported scripts
    import torch
    options = worker_options(others)
    device = torch.device(f"cuda:{options.gpu}" if torch.cuda.is_available() and not options.force_cpu else "cpu")
    if options.threads:
        torch.set_num_threads(int(options.threads))
    output_root = Path(args.output_root)
    out_root = output_root / ANALYSIS_FOLDER
    runs = [r for r in RUNS if r[1] in args.versions]
    print(f"Feature analysis ({WINDOW}): {', '.join(v for _, v in runs)}; six blocks (Stem-B6) and four blocks "
          f"(Stem-B4)\nDevice: {device} (no GPU needed)\nOutput: {out_root}", flush=True)
    store = {"rows": [], "channels": [], "models": [], "skipped": [], "coverage": [], "curves": {},
             "frequencies": None, "fs": 64}
    wanted = None
    for run_id, version in runs:
        four_root = earlier_root(args, version)
        four_script = four_root / VERSIONS[version]["script"]  # the script stored with the earlier run, else the base
        sets = [(SIX, run_folder(output_root, run_id, version), output_root / "Code" / script_name(run_id, version)),
                (FOUR, four_root, four_script if four_script.is_file() else base_script(args, version))]
        modules = {}
        for encoder, root, script in sets:
            if not script.is_file():
                print(f"[{version}] {encoder}: {script} not found (written by this file before the analysis)")
                continue
            module = load_module(script, f"deepencoder_b6_{version}_{ENCODER_TAGS[encoder]}")
            set_data_paths(module, options)
            modules[encoder] = (module, root)
            print(f"[{version}] {encoder}: models {root}\n    built with {script}", flush=True)
        if not modules:
            continue
        reader = next(iter(modules.values()))[0]  # the six- and four-block scripts preprocess alike
        if options.subjects:
            wanted = {reader.safe_tag(s) for s in options.subjects}
        spec = window_spec(reader)
        for dataset_key in args.datasets:
            label = DATASET_LABELS[dataset_key]
            tags = {}
            for encoder, (module, root) in modules.items():
                found = result_tags(version, module, root, dataset_key)
                tags[encoder] = [t for t in found if wanted is None or t in wanted]
                if not tags[encoder]:
                    print(f"[{version}] {encoder} {label}: no stored fold in {root}", flush=True)
                    store["coverage"].append({"Version": version, "Encoder": encoder, "Dataset": label,
                                              "Status": "no stored fold", "Folder": str(root)})
            if not any(tags.values()):
                continue
            started = time.perf_counter()
            print(f"\n[{version}] {label}: reading and preprocessing the raw data ...", flush=True)
            data = reader.load_dataset(dataset_key, spec)
            print(f"[{version}] {label}: {len(data)} subjects ({time.perf_counter() - started:.1f} s)", flush=True)
            for encoder, (module, root) in modules.items():
                if not tags[encoder]:
                    continue
                analysed = analyse_encoder(version, encoder, module, root, dataset_key, data, tags[encoder], options,
                                           device, store)
                store["coverage"].append({"Version": version, "Encoder": encoder, "Dataset": label,
                                          "Status": "analysed" if analysed else "no subject analysed",
                                          "Folder": str(root), "N_Stored_Folds": len(tags[encoder]),
                                          "N_Subjects_Analysed": analysed})
            del data
            reader.release_memory(device)
    write_analysis(out_root, store, [DATASET_LABELS[d] for d in args.datasets])
    return 0


# ---------------------------------------------------------------------------
# Feature analysis: statistics, tables and figures
# ---------------------------------------------------------------------------

def dataset_groups(frame: pd.DataFrame, datasets: List[str]) -> List[Tuple[str, pd.DataFrame]]:
    groups = [(d, frame[frame["Dataset"] == d]) for d in datasets]
    groups = [(d, f) for d, f in groups if not f.empty]
    return groups + [(ALL_DATASETS, frame)] if len(groups) > 1 else groups


def acf_column(stage: str, setting: str) -> str:
    return f"{stage}_ACF_r|{setting}"


def setting_columns(frame: pd.DataFrame, setting: str) -> pd.DataFrame:
    """The ACF r of one setting as <stage>_ACF_r; the columns of the other settings dropped."""
    suffix = f"|{setting}"
    keep = {c: c[:-len(suffix)] for c in frame.columns if c.endswith(suffix)}
    other = [c for c in frame.columns if "|" in c and c not in keep]
    return frame.drop(columns=other).rename(columns=keep)


def measure_labels(lag_band: Tuple[float, float]) -> Dict[str, str]:
    lag_text = f"{lag_band[0]:g}-{lag_band[1]:g} s"
    return {measure: label.replace("{band}", lag_text) for measure, label in MEASURES.items()}


def block_tests(subjects: pd.DataFrame, channels: pd.DataFrame, datasets: List[str],
                labels: Dict[str, str]) -> pd.DataFrame:
    """Per version, encoder, dataset, measure and stage: mean, vs Stem and vs the block before it."""
    rows = []
    for (version, encoder), part in subjects.groupby(["Version", "Encoder"], sort=False):
        stages = stage_names(len(ENCODERS[encoder]))
        part_channels = channels[(channels["Version"] == version) & (channels["Encoder"] == encoder)]
        pivot = part_channels.pivot_table(index=["Dataset", "Subject", "Channel"], columns="Stage", values="Abs_r")
        better = (pivot[stages[1:]].sub(pivot["Stem"], axis=0) > 0).groupby(level=["Dataset", "Subject"]).sum()
        for dataset, frame in dataset_groups(part, datasets):
            n = len(frame)
            index = pd.MultiIndex.from_arrays([frame["Dataset"], frame["Subject"]])
            for measure, description in labels.items():
                stem = frame[f"Stem_{measure}"].to_numpy(float)
                for position, stage in enumerate(stages):
                    values = frame[f"{stage}_{measure}"].to_numpy(float)
                    row = {"Version": version, "Encoder": encoder, "Window_Config": WINDOW, "Dataset": dataset,
                           "Measure": measure, "Description": description, "Stage": stage,
                           "Stage_Label": stage_label(stage).replace("\n", " "), "N_Subjects": n,
                           "Mean": float(np.nanmean(values)), "SD": sd(values),
                           "Median": float(np.nanmedian(values)), "Q1": float(np.nanpercentile(values, 25)),
                           "Q3": float(np.nanpercentile(values, 75))}
                    if position:
                        previous = frame[f"{stages[position - 1]}_{measure}"].to_numpy(float)
                        row.update({"Delta_vs_Stem": float(np.nanmean(values - stem)),
                                    "N_Higher_than_Stem": int(np.sum(values > stem)),
                                    "Wilcoxon_p_vs_Stem": wilcoxon_paired(values, stem),
                                    "Delta_vs_Previous": float(np.nanmean(values - previous)),
                                    "N_Higher_than_Previous": int(np.sum(values > previous)),
                                    "Wilcoxon_p_vs_Previous": wilcoxon_paired(values, previous)})
                        if measure == "Channel_AbsR_Mean":
                            row["Better_Channels_vs_Stem"] = float(better.reindex(index)[stage].mean())
                    if measure == "RespBand_Ratio":
                        row["Target_RespBand_Ratio_Mean"] = float(frame["Target_RespBand_Ratio"].mean())
                    rows.append(row)
    return pd.DataFrame(rows)


def six_vs_four(subjects: pd.DataFrame, datasets: List[str]) -> pd.DataFrame:
    """The same subjects with six and with four blocks: Stem vs Stem, B4 vs B4, last block (B6) vs last block (B4)."""
    rows = []
    pairs = (("Stem", "Stem", "Stem (six) vs Stem (four)"), ("B4", "B4", "B4 (six) vs B4 (four)"),
             ("B6", "B4", "Last block: B6 (six) vs B4 (four)"))
    keys = ["Version", "Dataset", "Subject"]
    for version, part in subjects.groupby("Version", sort=False):
        six = part[part["Encoder"] == SIX].set_index(keys)
        four = part[part["Encoder"] == FOUR].set_index(keys)
        common = six.index.intersection(four.index)
        if not len(common):
            continue
        both = part[part.set_index(keys).index.isin(common) & (part["Encoder"] == SIX)]
        for dataset, frame in dataset_groups(both, datasets):
            index = frame.set_index(keys).index
            for measure in FIGURE_MEASURES + ("PC1_Variance_Share",):
                for stage_six, stage_four, comparison in pairs:
                    a = six.loc[index, f"{stage_six}_{measure}"].to_numpy(float)
                    b = four.loc[index, f"{stage_four}_{measure}"].to_numpy(float)
                    ok = np.isfinite(a) & np.isfinite(b)
                    rows.append({"Version": version, "Window_Config": WINDOW, "Dataset": dataset, "Measure": measure,
                                 "Comparison": comparison, "Paired_Subjects": int(ok.sum()),
                                 "Mean_Six": float(np.mean(a[ok])) if ok.any() else np.nan,
                                 "Mean_Four": float(np.mean(b[ok])) if ok.any() else np.nan,
                                 "Median_Difference_Six_minus_Four": float(np.median(a[ok] - b[ok])) if ok.any()
                                 else np.nan,
                                 "N_Six_Higher": int(np.sum(a[ok] > b[ok])), "N_Four_Higher": int(np.sum(b[ok] > a[ok])),
                                 "Wilcoxon_p": wilcoxon_paired(a, b)})
    return pd.DataFrame(rows)


def key_results(subjects: pd.DataFrame, tests: pd.DataFrame, versus: pd.DataFrame) -> pd.DataFrame:
    """Per version and dataset: six blocks Stem -> B4 -> B6 (B6 vs Stem, B6 vs B4), four blocks Stem -> B4."""
    rows = []
    names = {"Channel_AbsR_Mean": "ChannelAbsR", "RespBand_Ratio": "RespBandRatio", "ACF_r": "ACF_r"}
    for (version, dataset), part in tests.groupby(["Version", "Dataset"], sort=False):
        row = {"Version": version, "Dataset": dataset}
        for encoder in (SIX, FOUR):
            sub = part[part["Encoder"] == encoder]
            row[f"N_{ENCODER_TAGS[encoder].title()}"] = int(sub["N_Subjects"].max()) if not sub.empty else 0
        for measure, name in names.items():
            six = part[(part["Encoder"] == SIX) & (part["Measure"] == measure)].set_index("Stage")
            four = part[(part["Encoder"] == FOUR) & (part["Measure"] == measure)].set_index("Stage")
            if not six.empty:
                frame = subjects[(subjects["Version"] == version) & (subjects["Encoder"] == SIX)]
                if dataset != ALL_DATASETS:
                    frame = frame[frame["Dataset"] == dataset]
                p_b6_b4 = wilcoxon_paired(frame[f"B6_{measure}"], frame[f"B4_{measure}"])
                row.update({f"Six_{name}_Stem": six.at["Stem", "Mean"], f"Six_{name}_B4": six.at["B4", "Mean"],
                            f"Six_{name}_B6": six.at["B6", "Mean"],
                            f"Six_{name}_p_B6_vs_Stem": six.at["B6", "Wilcoxon_p_vs_Stem"],
                            f"Six_{name}_B6_minus_B4": six.at["B6", "Mean"] - six.at["B4", "Mean"],
                            f"Six_{name}_p_B6_vs_B4": p_b6_b4})
            if not four.empty:
                row.update({f"Four_{name}_Stem": four.at["Stem", "Mean"], f"Four_{name}_B4": four.at["B4", "Mean"],
                            f"Four_{name}_p_B4_vs_Stem": four.at["B4", "Wilcoxon_p_vs_Stem"]})
            if not versus.empty:
                last = versus[(versus["Version"] == version) & (versus["Dataset"] == dataset)
                              & (versus["Measure"] == measure) & versus["Comparison"].str.startswith("Last block")]
                if not last.empty:
                    row.update({f"SixB6_vs_FourB4_{name}_n": int(last.iloc[0]["Paired_Subjects"]),
                                f"SixB6_vs_FourB4_{name}_p": last.iloc[0]["Wilcoxon_p"]})
        rows.append(row)
    return pd.DataFrame(rows)


def figure_block_summary(path: Path, version: str, dataset: str, frames: Dict[str, pd.DataFrame],
                         tests: pd.DataFrame, labels: Dict[str, str]) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    six_stages = stage_names(len(DILATIONS))
    fig, axes = plt.subplots(1, len(FIGURE_MEASURES), figsize=(15.5, 4.9))
    for axis, measure in zip(axes, FIGURE_MEASURES):
        axis.axvspan(4.5, len(six_stages) - 0.5, color="#f0efec", zorder=0)
        for encoder, frame in frames.items():
            if frame.empty:
                continue
            stages = stage_names(len(ENCODERS[encoder]))
            values = np.array([frame[f"{s}_{measure}"].to_numpy(float) for s in stages])
            means = np.nanmean(values, axis=1)
            counts = np.sum(np.isfinite(values), axis=1)
            errors = np.array([sd(v) for v in values]) / np.sqrt(np.maximum(counts, 1))
            x = np.arange(len(stages)) + (0.06 if encoder == FOUR else -0.06)
            axis.errorbar(x, means, yerr=errors, color=ENCODER_COLORS[encoder], linewidth=2.0, marker="o",
                          markersize=8, capsize=3, elinewidth=1.0, linestyle="-" if encoder == SIX else "--",
                          markeredgecolor="#fcfcfb", markeredgewidth=1.5,
                          label=f"{encoder} (n = {len(frame)})", zorder=3)
            if encoder == SIX:
                sub = tests[(tests["Version"] == version) & (tests["Encoder"] == encoder)
                            & (tests["Dataset"] == dataset) & (tests["Measure"] == measure)].set_index("Stage")
                for position, stage in enumerate(stages[1:], start=1):
                    mark = stars(sub.at[stage, "Wilcoxon_p_vs_Previous"]) if stage in sub.index else "n/a"
                    if mark not in ("ns", "n/a") and np.isfinite(means[position]):
                        axis.annotate(mark, (x[position], means[position] + errors[position]),
                                      textcoords="offset points", xytext=(0, 4), ha="center", fontsize=9,
                                      color=TEXT_SECONDARY)
        axis.set_xticks(range(len(six_stages)))
        axis.set_xticklabels([stage_label(s) for s in six_stages], fontsize=8)
        axis.set_xlim(-0.5, len(six_stages) - 0.5)
        axis.set_title(labels[measure], fontsize=10)
        axis.grid(axis="y", color=GRID_COLOR)
        axis.set_axisbelow(True)
        for side in ("top", "right"):
            axis.spines[side].set_visible(False)
        axis.margins(y=0.15)
        axis.text(5.5, 0.02, "encoder RF > 10 s window", transform=axis.get_xaxis_transform(), ha="center",
                  va="bottom", fontsize=7.5, color=TEXT_SECONDARY)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper right", ncol=2, frameon=False, fontsize=9)
    fig.suptitle(f"{version} | {dataset} | 10 s windows | mean ± SE over subjects", x=0.01, ha="left", fontsize=11)
    fig.text(0.01, 0.01, "x: block, dilation, encoder receptive field up to the block (RF).  "
                         "* p<0.05, ** p<0.01, *** p<0.001: six blocks, block vs the block before it "
                         "(paired Wilcoxon). Six and four blocks are different trained models.",
             fontsize=8, color=TEXT_SECONDARY)
    fig.tight_layout(rect=(0, 0.05, 1, 0.92))
    fig.savefig(path, dpi=160)
    plt.close(fig)


def figure_pc1_curves(path: Path, title: str, frequencies: np.ndarray, lags: np.ndarray,
                      psd: Dict[str, np.ndarray], acf: Dict[str, np.ndarray], stages: List[str],
                      frame: pd.DataFrame, lag_band: Tuple[float, float], max_lag_sec: float) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    shown = frequencies <= PSD_PLOT_MAX_HZ
    fig, axes = plt.subplots(1, 2, figsize=(14.0, 4.8))
    axes[0].axvspan(*RESP_BAND_HZ, color="#f0efec", zorder=0)
    axes[1].axvspan(*lag_band, color="#f0efec", zorder=0)
    names = ["Target"] + stages
    for name in names:
        color = TARGET_COLOR if name == "Target" else STAGE_RAMP[stages.index(name)]
        width = 2.4 if name == "Target" else 1.6
        style = "--" if name == "Target" else "-"
        curve = psd[name][shown]
        scale = np.nanmax(curve) if np.nanmax(curve) > 0 else 1.0
        if name == "Target":
            label = f"Target RSP (ratio {frame['Target_RespBand_Ratio'].mean():.3f})"
            acf_label = "Target RSP"
        else:
            label = f"{name} (ratio {frame[f'{name}_RespBand_Ratio'].mean():.3f})"
            acf_label = f"{name} (r = {frame[f'{name}_ACF_r'].mean():.3f})"
        axes[0].plot(frequencies[shown], curve / scale, color=color, linewidth=width, linestyle=style, label=label)
        axes[1].plot(lags, acf[name], color=color, linewidth=width, linestyle=style, label=acf_label)
    axes[0].set_xlabel("Frequency (Hz)")
    axes[0].set_ylabel("PSD of PC1 (scaled to its maximum)")
    axes[0].set_xlim(0, PSD_PLOT_MAX_HZ)
    axes[0].set_title("PC1 power spectrum (shaded: 0.1-0.5 Hz)", fontsize=10)
    axes[1].set_xlabel("Lag (s)")
    axes[1].set_ylabel("Normalized autocorrelation")
    axes[1].set_xlim(0, max_lag_sec)
    axes[1].set_title(f"PC1 autocorrelation (shaded: lags {lag_band[0]:g}-{lag_band[1]:g} s used for r)",
                      fontsize=10)
    for axis in axes:
        axis.grid(True, color=GRID_COLOR)
        axis.set_axisbelow(True)
        for side in ("top", "right"):
            axis.spines[side].set_visible(False)
        axis.legend(loc="upper left", bbox_to_anchor=(1.01, 1.0), fontsize=8, frameon=False)
    fig.suptitle(title, x=0.01, ha="left", fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    fig.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(fig)


def mean_curves(frame: pd.DataFrame, curves: dict, names: List[str]) -> Dict[str, np.ndarray]:
    keys = list(zip(frame["Version"], frame["Encoder"], frame["Dataset"], frame["Subject"]))
    return {name: np.nanmean([curves[key][name] for key in keys], axis=0) for name in names}


def save_table(frame: pd.DataFrame, path: Path) -> None:
    try:
        frame.to_csv(path, index=False, encoding="utf-8-sig")
    except PermissionError:
        alt = path.with_name(f"{path.stem}_{datetime.now():%Y%m%d_%H%M%S}{path.suffix}")
        frame.to_csv(alt, index=False, encoding="utf-8-sig")
        print(f"  [Locked] {path.name} is held by another program; saved as {alt.name}")


def write_analysis(out_root: Path, store: dict, datasets: List[str]) -> None:
    """One complete set of tables and figures per ACF setting, each in its own subfolder."""
    for setting, lag_band, max_lag_sec in ACF_SETTINGS:
        print(f"\n=== {setting}: ACF r over lags {lag_band[0]:g}-{lag_band[1]:g} s, ACF drawn 0-{max_lag_sec:g} s "
              f"-> {out_root / setting}", flush=True)
        write_setting(out_root / setting, store, datasets, setting, lag_band, max_lag_sec)


def write_setting(out_root: Path, store: dict, datasets: List[str], setting: str, lag_band: Tuple[float, float],
                  max_lag_sec: float) -> None:
    out_root.mkdir(parents=True, exist_ok=True)
    labels = measure_labels(lag_band)
    figure_root = out_root / "Figures"
    figure_root.mkdir(parents=True, exist_ok=True)
    members = setting_columns(pd.DataFrame(store["rows"]), setting)
    channels = pd.DataFrame(store["channels"])
    models = pd.DataFrame(store["models"])
    coverage = pd.DataFrame(store["coverage"])
    for column in ("N_Stored_Folds", "N_Subjects_Analysed"):
        if column in coverage.columns:
            coverage[column] = coverage[column].astype("Int64")  # counts stay whole numbers next to empty cells
    skipped =pd.DataFrame(store["skipped"], columns=["Version", "Encoder", "Dataset", "Subject", "Reason"])
    for frame, name in ((models, "models.csv"), (coverage, "coverage.csv"), (skipped, "skipped_subjects.csv")):
        save_table(frame, out_root / name)
    if members.empty:
        print("\nFeature analysis: no subject with all three models stored yet (see coverage.csv, "
              "skipped_subjects.csv)")
        return
    keys = ["Version", "Encoder", "Dataset", "Subject"]
    values = [c for c in members.columns if c not in keys + ["Model", "N_Windows"]]
    subjects = members.groupby(keys, sort=False)[values].mean().reset_index()
    subjects.insert(4, "N_Models", members.groupby(keys, sort=False).size().to_numpy())
    channel_subject = channels.groupby(keys + ["Stage", "Channel"], sort=False)["Abs_r"].mean().reset_index()
    tests = block_tests(subjects, channel_subject, datasets, labels)
    versus = six_vs_four(subjects, datasets)
    key = key_results(subjects, tests, versus)
    fs = float(store["fs"])
    lags = np.arange(int(round(max_lag_sec * fs)) + 1) / fs
    # the ACF curves (computed up to ACF_MAX_LAG_SEC) cut at the largest lag of this setting
    curves = {key: {name: (values[:len(lags)] if name.endswith("_ACF") else values) for name, values in item.items()}
              for key, item in store["curves"].items()}
    frequencies = store["frequencies"]
    curve_arrays = {"frequencies_hz": frequencies, "acf_lags_sec": lags}
    for version, part in subjects.groupby("Version", sort=False):
        frames = {encoder: part[part["Encoder"] == encoder] for encoder in (SIX, FOUR)}
        for dataset, _ in dataset_groups(part, datasets):
            chosen = {e: (f if dataset == ALL_DATASETS else f[f["Dataset"] == dataset]) for e, f in frames.items()}
            tag = f"{version}_{dataset.replace(' ', '_')}"
            figure_block_summary(figure_root / f"{tag}_block_summary.png", version, dataset, chosen, tests, labels)
            for encoder, frame in chosen.items():
                if frame.empty:
                    continue
                stages = stage_names(len(ENCODERS[encoder]))
                names = ["Target"] + stages
                psd = mean_curves(frame, curves, [f"{n}_PSD" for n in names])
                acf = mean_curves(frame, curves, [f"{n}_ACF" for n in names])
                psd = {n: psd[f"{n}_PSD"] for n in names}
                acf = {n: acf[f"{n}_ACF"] for n in names}
                for n in names:
                    curve_arrays[f"{tag}_{ENCODER_TAGS[encoder]}_PSD_{n}"] = psd[n]
                    curve_arrays[f"{tag}_{ENCODER_TAGS[encoder]}_ACF_{n}"] = acf[n]
                figure_pc1_curves(figure_root / f"{tag}_{ENCODER_TAGS[encoder]}_blocks_pc1_curves.png",
                                  f"{version} | {dataset} | {encoder} (n = {len(frame)}) | mean over subjects",
                                  frequencies, lags, psd, acf, stages, frame, lag_band, max_lag_sec)
    np.savez_compressed(out_root / "curves.npz", **curve_arrays)
    tables = {"Key_Results": key, "Block_Tests": tests, "Six_vs_Four": versus, "Subjects": subjects,
              "Models": models, "Coverage": coverage, "Skipped": skipped,
              "Notes": pd.DataFrame([
                  ("Models", "Six blocks: the runs 01_261013_B6 / 02_261007_B6 of this folder; four blocks: the earlier "
                             "runs (261013_RRWavNet_STEAM_BIDMC_v1, 261007_RRWaveNet_NoEvent_v1), read only. 261013: "
                             "m0-m2 of the weight decay selected in the fold; 261007: full m0-m2"),
                  ("Windows", "10 s test windows of the held-out subject (LOSOCV), read by the version's own code"),
                  ("Stages", "Stem = stem fusion output (encoder input); B1-B6 d = 1, 2, 4, 8, 16, 32; encoder "
                             "receptive field B4 6.58 s, B5 13.58 s, B6 27.58 s (10 s window = 640 samples)"),
                  ("Measures", "; ".join(f"{k}: {v}" for k, v in labels.items())),
                  ("ACF", f"{setting}: r over lags {lag_band[0]:g}-{lag_band[1]:g} s, ACF drawn up to {max_lag_sec:g} s; "
                          "PC1 and target of contiguous 10 s test windows joined into runs >= 24 s, mean removed per "
                          "run, lag 0 = 1, runs averaged by length"),
                  ("Statistics", "values averaged over windows, then over the subject's 3 models; paired two-sided "
                                 "Wilcoxon over subjects; p not adjusted"),
              ], columns=["Item", "Description"])}
    for name, frame in (("key_results.csv", key), ("block_tests.csv", tests), ("six_vs_four_tests.csv", versus),
                        ("subjects.csv", subjects), ("channels.csv", channel_subject), ("members.csv", members)):
        save_table(frame, out_root / name)
    path = out_root / f"{NAME}_Feature_Analysis.xlsx"
    try:
        writer_path = path
        with pd.ExcelWriter(writer_path, engine="openpyxl") as writer:
            for name, frame in tables.items():
                frame.to_excel(writer, sheet_name=name[:31], index=False)
    except PermissionError:
        writer_path = path.with_name(f"{path.stem}_{datetime.now():%Y%m%d_%H%M%S}{path.suffix}")
        with pd.ExcelWriter(writer_path, engine="openpyxl") as writer:
            for name, frame in tables.items():
                frame.to_excel(writer, sheet_name=name[:31], index=False)
        print(f"  [Locked] {path.name} is held by another program; saved as {writer_path.name}")
    write_results_md(out_root, key, lag_band)
    print(f"\nFeature analysis: {writer_path}")


def write_results_md(out_root: Path, key: pd.DataFrame, lag_band: Tuple[float, float]) -> None:
    def cell(row, columns, p_column) -> str:
        if any(c not in row or pd.isna(row[c]) for c in columns):
            return "-"
        return "→".join(f"{row[c]:.3f}" for c in columns) + f" ({p_text(row[p_column])})"

    lag_text = f"{lag_band[0]:g}-{lag_band[1]:g} s"
    lines = [f"# {NAME}: encoder block 별 feature 분석 (10 s window, ACF r: lag {lag_text})", "",
             "6-block: Stem → B4 → B6 (괄호: B6 vs Stem의 p). B5/B6 추가 효과: 같은 6-block 모델의 B6 vs B4. "
             "4-block (이전 학습 모델): Stem → B4 (괄호: B4 vs Stem의 p). 피험자 단위 양측 paired Wilcoxon, p는 보정 전 값.", ""]
    for name, title in (("ChannelAbsR", "channel |r|"), ("RespBandRatio", "PC1 호흡대역 비율"), ("ACF_r", f"PC1 ACF r (lag {lag_text})")):
        lines += [f"## {title}", "",
                  "| Version | 데이터셋 | N (6/4) | 6-block Stem→B4→B6 (p B6 vs Stem) | 6-block B6−B4 (p) | "
                  "4-block Stem→B4 (p) | 6-block B6 vs 4-block B4 (p) |",
                  "|---|---|---|---|---|---|---|"]
        for _, row in key.iterrows():
            delta = (f"{row[f'Six_{name}_B6_minus_B4']:+.3f} ({p_text(row[f'Six_{name}_p_B6_vs_B4'])})"
                     if f"Six_{name}_B6_minus_B4" in row and pd.notna(row[f"Six_{name}_B6_minus_B4"]) else "-")
            versus = (p_text(row[f"SixB6_vs_FourB4_{name}_p"])
                      if f"SixB6_vs_FourB4_{name}_p" in row and pd.notna(row[f"SixB6_vs_FourB4_{name}_p"]) else "-")
            lines.append(f"| {row['Version']} | {row['Dataset']} | {row.get('N_Six', 0)}/{row.get('N_Four', 0)} | "
                         f"{cell(row, [f'Six_{name}_Stem', f'Six_{name}_B4', f'Six_{name}_B6'], f'Six_{name}_p_B6_vs_Stem')}"
                         f" | {delta} | "
                         f"{cell(row, [f'Four_{name}_Stem', f'Four_{name}_B4'], f'Four_{name}_p_B4_vs_Stem')} | {versus} |")
        lines.append("")
    lines += ["B5 (13.58 s)와 B6 (27.58 s)의 receptive field는 10 s window(640 samples)보다 넓어, 바깥쪽 tap은 "
              "window 가장자리의 zero padding을 봅니다. 6-block과 4-block은 서로 다르게 학습된 모델입니다."]
    with open(out_root / "RESULTS.md", "w", encoding="utf-8", newline="\n") as handle:
        handle.write("\n".join(lines) + "\n")
    shown = [c for c in key.columns if c in ("Version", "Dataset", "N_Six", "N_Four") or c.startswith("Six_ACF_r")
             or c.startswith("Four_ACF_r")]
    pd.set_option("display.width", 220)
    print(f"\nKey results (PC1 ACF r, lags {lag_text}; all measures in key_results.csv / RESULTS.md):")
    print(key[shown].to_string(index=False, float_format=lambda v: f"{v:.3g}"))


# ---------------------------------------------------------------------------
# Comparison: six vs four blocks (prediction results)
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


def compare(args: argparse.Namespace, runs: Sequence[tuple]) -> Optional[Path]:
    output_root = Path(args.output_root)
    six, four = "Six blocks (B1-B6, d up to 32)", "Four blocks (earlier run)"
    frames, status = [], []
    for run_id, version in runs:
        for root, label, run in ((run_folder(output_root, run_id, version), six, run_id),
                                 (earlier_root(args, version), four, "earlier")):
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
                 "the earlier runs (261013_RRWavNet_STEAM_BIDMC_v1, 261007_RRWaveNet_NoEvent_v1), read only; "
                 "261013 earlier: BIDMC and STEAM2 only"),
        ("Per subject", "Summary\\per_subject_fft.csv: FFT_MAE, Wave_PCC_Mean; 261007 the full model's ensemble, "
                        "261013 the final ensemble"),
        ("Paired_Tests", "same subjects; Wilcoxon signed-rank; BH over all tests of a subject set and metric; "
                         "A_Better = subjects with lower FFT_MAE / higher PCC with six blocks; Same_Minutes_And_GT: "
                         "subjects scored on the same minutes against the same ground truth"),
        ("Feature analysis", f"{ANALYSIS_FOLDER}\\<ACF setting>\\ (six blocks Stem-B6 and four blocks Stem-B4: "
                             "channel |r|, PC1 respiratory-band power, PC1 ACF; ACF r over lags 1-5 s and 2-10 s)"),
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
    parser = argparse.ArgumentParser(description="RRWaveNet with a six-block encoder (B5 d=16, B6 d=32) from the "
                                                 "261013 / 261007 code: training, block-by-block feature analysis "
                                                 "(with the earlier four-block models) and comparison")
    parser.add_argument("--versions", nargs="+", choices=list(VERSIONS), default=[v for _, v in RUNS])
    parser.add_argument("--datasets", nargs="+", choices=list(DATASET_LABELS), default=list(DATASET_RUN_ORDER))
    parser.add_argument("--scripts-dir", type=Path, default=None, help="folder of the 261013 / 261007 scripts")
    parser.add_argument("--output-root", type=Path, default=OUTPUT_ROOT)
    for version in VERSIONS:
        parser.add_argument(f"--args-{version}", dest=f"args_{version}", default="",
                            help=f"options only for the {version} training")
        parser.add_argument(f"--earlier-root-{version}", dest=f"earlier_root_{version}", type=Path, default=None,
                            help=f"the earlier four-block results of {version} "
                                 f"(default {WORKSPACE / VERSIONS[version]['earlier_run']})")
    parser.add_argument("--no-resume", action="store_true", help="do not pass --resume to the training calls")
    parser.add_argument("--write-only", action="store_true", help="write the scripts, do not run")
    parser.add_argument("--plan", action="store_true", help="print every call, do not run")
    parser.add_argument("--analysis-only", action="store_true", help="feature analysis and comparison, no training")
    parser.add_argument("--compare-only", action="store_true", help="only the comparison of the finished runs")
    parser.add_argument("--analysis-worker", action="store_true", help=argparse.SUPPRESS)
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
    if args.analysis_worker:
        return analysis_worker(args, others)
    if args.compare_only:
        compare(args, runs)
        return 0
    print(f"{NAME}: encoder d = {DILATIONS}; runs {', '.join(f'{r[0]} {r[1]}' for r in runs)}; datasets "
          f"{' -> '.join(DATASET_LABELS[d] for d in args.datasets)}\nResults: {output_root}")
    scripts = write_scripts(args, runs)
    if args.write_only:
        print(f"\nWritten only (--write-only): {output_root / 'Code'}")
        return 0
    status = 0 if args.analysis_only else train(args, others, runs, scripts)
    status = analyse(args, others, runs) or status
    if not args.plan:
        compare(args, runs)
    return status


if __name__ == "__main__":
    sys.exit(main())
