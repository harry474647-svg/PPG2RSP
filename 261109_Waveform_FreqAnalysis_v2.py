# -*- coding: utf-8 -*-
r"""261109_Waveform_FreqAnalysis_v2: 261109_Waveform_FreqAnalysis with the earlier
training reused - every member already trained for the same run (same input band, same
version, same code) is taken over and only what is not trained yet is trained. Results
in a new folder; the earlier folders are only read.

Runs (as v1), in this order (each: STEAM2 -> BIDMC -> CapnoBase)
  01  PPG 0.1-0.8 Hz + 261013     02  PPG 0.1-0.8 Hz + 261007
  03  PPG 0.1-8.0 Hz + 261013     04  PPG 0.1-8.0 Hz + 261007
Each run is the code of the base script with exactly the v1 text changes (PPG_INPUT and
model_input: zero-phase 3rd-order Butterworth band-pass of the PPG; PROJECT_ROOT); model,
loss, training, ensembles, validation, LOSOCV, targets, scoring and outputs unchanged.

Reuse of the earlier training (before the calls of every run)
  sources, in this order (read only):
    v1        D:\PPG2RSP_RRWaveNet_Inspired\261109_Waveform_FreqAnalysis\<NN_PPG_band_version>
              (and ...\261109_Waveform_FrequencyAnalysis\..., the folder the v1 code was kept in,
              when v1 was run with --output-root there; more with --previous-roots)
    HighFreq  D:\PPG2RSP_RRWaveNet_Inspired\261109_RRWaveNet_HighFreq\261007_RRWaveNet_NoEvent_HighFreq
              and \261013_RRWaveNet_STEAM_BIDMC_HighFreq (PPG 0.1-8.0 Hz: runs 03 and 04 only)
  a source is used only when its generated script (its Code\ folder) is the same code as the
  script of the run here (comments, blank lines and the results folder left out): the same
  input band, filter and base script give the same training. Otherwise it is reported and
  not used (--reuse-unverified uses it anyway).
  per member (fold subject, m0/m1/m2, window, dataset; 261007: the full model only; 261013:
  every weight decay candidate): a finished member (summary, scored minutes, predictions) is
  copied with its model, learning curve and audit; a member still in training (its training
  state) is copied so that it continues from its last epoch; a member the run here already
  has is never overwritten. Ensembles, weight decay choices (261013) and summaries are
  computed again from the members (no training). Then the calls of v1 run with --resume:
  the base scripts skip every finished member and train only the rest.
  Every copy is listed in Reuse_Log.csv.

How a run is executed, comparison and outputs: as v1 (261007 --variants full and its default
windows 10 s / 20 s; 261013 its default; one call per dataset, then --summary-only; the 261013
summary is written again after the 261007 run of the same band).

Outputs (D:\PPG2RSP_RRWaveNet_Inspired\261109_Waveform_FreqAnalysis_v2)
  Code\, 01_PPG_0.1-0.8Hz_261013\ ... 04_PPG_0.1-8.0Hz_261007\, run_log.csv, Logs\,
  Reuse_Log.csv (what was taken over from where), Comparison\..._Comparison.xlsx (as v1)

Run (needs CUDA as the base scripts):
  python 261109_Waveform_FreqAnalysis_v2.py
  python 261109_Waveform_FreqAnalysis_v2.py --plan          the calls and what would be reused
  python 261109_Waveform_FreqAnalysis_v2.py --compare-only  only the comparison
  python 261109_Waveform_FreqAnalysis_v2.py --runs 03 04    only these runs
Spyder: runfile(r'D:\PPG2RSP_RRWaveNet_Inspired\...\261109_Waveform_FreqAnalysis_v2.py', args='--plan')
Other options go to every script call unchanged (e.g. --gpu 1, --no-require-cuda).
Options of this file: --runs, --scripts-dir, --output-root, --previous-roots, --highfreq-root, --no-reuse,
  --reuse-unverified, --args-261007, --args-261013, --no-resume, --write-only, --plan, --compare-only
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

NAME = "261109_Waveform_FreqAnalysis_v2"
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

# Earlier results of the same runs (read only): v1 of this code and, for 0.1-8.0 Hz, 261109_RRWaveNet_HighFreq
V1_NAME = "261109_Waveform_FreqAnalysis"
PREVIOUS_V1_ROOTS = (WORKSPACE / V1_NAME, WORKSPACE / "261109_Waveform_FrequencyAnalysis")
PREVIOUS_HIGHFREQ = WORKSPACE / "261109_RRWaveNet_HighFreq"
HIGHFREQ_RESULTS = {"261007": "261007_RRWaveNet_NoEvent_HighFreq", "261013": "261013_RRWaveNet_STEAM_BIDMC_HighFreq"}
# the files of one member in a condition folder (<variant>\<NN_window>\<dataset>); the first three mark it finished
MEMBER_FILES = (
    "Fold_Summaries/{tag}_m{k}_summary.json",
    "Fold_Summaries/{tag}_m{k}_RR60s_windows.csv",
    "Members/m{k}/Predictions/{tag}_test_outputs.npz",
    "Members/m{k}/Models/{tag}_best.pth",
    "Members/m{k}/Models/{tag}_training_state.pt",
    "Members/m{k}/Learning_Curves/{tag}_Learning_Curve.csv",
    "Members/m{k}/Training_Audits/{tag}_Training_Audit.csv",
)
SUMMARY_PATTERN = re.compile(r"^(?P<tag>.+)_m(?P<k>\d+)_summary\.json$")


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
    root_line = f'PROJECT_ROOT = Path(r"{root}")'
    # a function as replacement: the backslashes of a Windows path are kept as they are (no regex escapes)
    text = ROOT_PATTERN.sub(lambda _: root_line, text)
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
        # open(..., newline="\n"): Path.write_text has no newline argument before Python 3.10
        with open(path, "w", encoding="utf-8", newline="\n") as handle:
            handle.write(patched_text(text, base.name, band, root))
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
    run_id, band_label, band, version = run
    base = [sys.executable, str(script)]
    extra = list(VERSIONS[version]["options"]) + others
    extra += split_options(args.args_261007 if version == "261007" else args.args_261013)
    if version == "261013" and "--reference-root" not in extra:
        partner = next(r for r in RUNS if r[1] == band_label and r[3] == "261007")
        extra += ["--reference-root", str(run_folder(Path(args.output_root), partner[0], partner[1], partner[3]))]
    resume = [] if args.no_resume or "--resume" in extra else ["--resume"]
    return base + resume + extra, base + ["--summary-only"] + extra


# ---------------------------------------------------------------------------
# Reuse of the earlier training
# ---------------------------------------------------------------------------

def code_signature(text: str) -> str:
    """The code without comments, blank lines and the results folder: the same signature is the same training."""
    lines = []
    for line in ROOT_PATTERN.sub("PROJECT_ROOT = <root>", text.replace("\r\n", "\n")).split("\n"):
        if not line.strip() or line.strip().startswith("#"):
            continue
        lines.append(re.sub(r'^(PPG_INPUT = "bandpass")\s+#.*$', r"\1", line).rstrip())
    return hashlib.sha256("\n".join(lines).encode("utf-8")).hexdigest()


def previous_sources(run: tuple, args: argparse.Namespace) -> List[Tuple[str, Path, Path]]:
    """(name, earlier run folder, its generated script) of the earlier runs with the same band and version."""
    run_id, band_label, band, version = run
    roots = [Path(p) for p in PREVIOUS_V1_ROOTS] + [Path(p) for p in (args.previous_roots or [])]
    out, seen = [], set()
    for root in roots:
        folder = root / f"{run_id}_PPG_{band_label}_{version}"
        if folder.resolve() in seen or folder.resolve() == run_folder(Path(args.output_root), *run[:2], version).resolve():
            continue
        seen.add(folder.resolve())
        out.append((f"v1 ({root.name})", folder, root / "Code" / f"{V1_NAME}_{run_id}_PPG_{band_label}_{version}.py"))
    if tuple(band) == (0.1, 8.0):
        highfreq = Path(args.highfreq_root)
        out.append(("HighFreq", highfreq / HIGHFREQ_RESULTS[version],
                    highfreq / "Code" / f"261109_RRWaveNet_HighFreq_from_{version}.py"))
    return out


def wanted_variant(version: str, name: str) -> bool:
    """261007: the full model only (its ablation variants are other models); 261013: every weight decay candidate."""
    return name == "full" if version == "261007" else name.startswith("full_wd")


def members_in(condition: Path) -> List[Tuple[str, int]]:
    found = set()
    for path in (condition / "Fold_Summaries").glob("*_m*_summary.json"):
        match = SUMMARY_PATTERN.match(path.name)
        if match:
            found.add((match.group("tag"), int(match.group("k"))))
    for path in condition.glob("Members/m*/Models/*_training_state.pt"):
        member = path.parent.parent.name
        if member[1:].isdigit():
            found.add((path.name[: -len("_training_state.pt")], int(member[1:])))
    return sorted(found)


def member_finished(condition: Path, tag: str, k: int) -> bool:
    return all((condition / f.format(tag=tag, k=k)).exists() for f in MEMBER_FILES[:3])


def reuse_previous(run: tuple, script: Path, root: Path, args: argparse.Namespace) -> List[dict]:
    """Copy the earlier members of this run into its folder (see the module text); returns the log rows."""
    run_id, band_label, band, version = run
    signature = code_signature(script.read_text(encoding="utf-8"))
    rows = []
    for name, source, code in previous_sources(run, args):
        base = {"Run": run_id, "Source": name, "Source_Folder": str(source)}
        if not source.is_dir():
            rows.append({**base, "Status": "not found"})
            continue
        same = code.exists() and code_signature(code.read_text(encoding="utf-8")) == signature
        if not same and not args.reuse_unverified:
            rows.append({**base, "Status": ("generated script not found" if not code.exists() else
                                            "different code") + " - not reused (--reuse-unverified to use it)"})
            continue
        used = 0
        for variant_dir in sorted(p for p in source.iterdir() if p.is_dir() and wanted_variant(version, p.name)):
            for condition in sorted(p for p in variant_dir.glob("*/*") if p.is_dir()):
                target = root / condition.relative_to(source)
                finished = in_training = 0
                for tag, k in members_in(condition):
                    if member_finished(target, tag, k):
                        continue
                    if member_finished(condition, tag, k):
                        files, finished = MEMBER_FILES, finished + 1
                        stale = target / MEMBER_FILES[4].format(tag=tag, k=k)
                        if not args.plan and stale.exists():
                            stale.unlink()  # a finished member needs no training state
                    elif not any((target / f.format(tag=tag, k=k)).exists() for f in MEMBER_FILES):
                        files, in_training = MEMBER_FILES[3:], in_training + 1
                    else:
                        continue
                    if args.plan:
                        continue
                    for f in files:
                        src = condition / f.format(tag=tag, k=k)
                        if src.exists():
                            dst = target / f.format(tag=tag, k=k)
                            dst.parent.mkdir(parents=True, exist_ok=True)
                            shutil.copy2(src, dst)
                if finished or in_training:
                    used += 1
                    rows.append({**base, "Status": "would be reused (--plan)" if args.plan else "reused",
                                 "Condition": condition.relative_to(source).as_posix(),
                                 "Finished_Members": finished, "Members_In_Training": in_training})
        if not used:
            rows.append({**base, "Status": "same code; nothing new to reuse" if same else "nothing new to reuse"})
    return rows


def log_reuse(output_root: Path, rows: List[dict]) -> None:
    if not rows:
        return
    stamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    columns = ["Time", "Run", "Source", "Status", "Condition", "Finished_Members", "Members_In_Training",
               "Source_Folder"]  # fixed columns: rows of every run go into one table
    frame = pd.DataFrame([{"Time": stamp, **row} for row in rows]).reindex(columns=columns)
    path = output_root / "Reuse_Log.csv"
    try:
        frame.to_csv(path, mode="a", header=not path.exists(), index=False, encoding="utf-8-sig")
    except PermissionError:
        frame.to_csv(path.with_name(f"Reuse_Log_{datetime.now():%Y%m%d_%H%M%S}.csv"), index=False,
                     encoding="utf-8-sig")
    for row in rows:
        detail = (f"{row['Condition']}: {row.get('Finished_Members', 0)} finished, "
                  f"{row.get('Members_In_Training', 0)} in training" if row.get("Condition") else "")
        print(f"  [reuse {row['Run']}] {row['Source']}: {row['Status']}{' - ' + detail if detail else ''}", flush=True)


def execute(args: argparse.Namespace, others: List[str], runs: Sequence[tuple], scripts: Dict[str, Path]) -> int:
    status = 0
    output_root = Path(args.output_root)
    for run in runs:
        run_id, band_label, band, version = run
        train, summary = run_commands(run, scripts[run_id], args, others)
        root = run_folder(output_root, run_id, band_label, version)
        print(f"\n### Run {run_id}: PPG {band_label.replace('Hz', ' Hz')} + {version} -> {root}", flush=True)
        if not args.no_reuse:
            rows = reuse_previous(run, scripts[run_id], root, args)
            if args.plan:
                for row in rows:
                    print(f"  [reuse {row['Run']}] {row['Source']}: {row['Status']}"
                          + (f" - {row['Condition']}: {row['Finished_Members']} finished, "
                             f"{row['Members_In_Training']} in training" if row.get("Condition") else ""), flush=True)
            else:
                log_reuse(output_root, rows)
        for dataset in DATASET_RUN_ORDER:
            since = datetime.now().timestamp()
            code = call(train + ["--datasets", dataset], output_root, run_id, DATASET_LABELS[dataset], args.plan)
            status = status or code
            if not args.plan:
                keep_call_files(root, dataset, since)
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
        return pd.read_csv(path, dtype={"Subject": str}, encoding="utf-8-sig") if path.exists() else pd.DataFrame()
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
                                  kind="mergesort").reset_index(drop=True)
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
                                  kind="mergesort").reset_index(drop=True)
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
    parser.add_argument("--previous-roots", nargs="*", type=Path, default=None,
                        help="further output folders of v1 (with Code\\ and NN_PPG_<band>_<version>\\)")
    parser.add_argument("--highfreq-root", type=Path, default=PREVIOUS_HIGHFREQ,
                        help="output folder of 261109_RRWaveNet_HighFreq")
    parser.add_argument("--no-reuse", action="store_true", help="do not take over earlier training")
    parser.add_argument("--reuse-unverified", action="store_true",
                        help="take over earlier members even when their generated script is missing or differs")
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
