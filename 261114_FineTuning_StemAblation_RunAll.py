# -*- coding: utf-8 -*-
r"""261114_FineTuning_StemAblation_RunAll: one run of the whole fine-tuning study.
Started without options (python 261114_FineTuning_StemAblation_RunAll.py, or runfile in
Spyder) it runs, one after the other:

  1. 261109_RRWaveNet_FineTune       fine-tuning WITH the 1x1 stem fusion (full model):
                                      RRWaveNet 261013 and 261007 pretrained on CapnoBase,
                                      fine-tuned by LOSOCV on STEAM2 -> BIDMC, compared with
                                      training from scratch
  2. 261113_FineTuning_StemAblation  the same fine-tuning WITHOUT the 1x1 stem fusion (stem
                                      fusion ablation) and the comparison: fine-tuned w/o vs
                                      with stem fusion (the results of step 1) and vs training
                                      from scratch

Nothing of the two scripts is changed; each is run as it is, in its own process, with its
output shown here and kept in a log. Both continue their own results folder: pretrained
members and fine-tuning datasets that are finished are skipped and an interrupted member
continues from its last epoch, so a rerun of this file continues where it stopped. Step 2
runs also when step 1 ended with an error (its training does not need step 1; its comparison
then lists what step 1 has not finished, and a later rerun completes it).

Results
  D:\PPG2RSP_RRWaveNet_Inspired\261109_RRWaveNet_FineTune\        (step 1)
  D:\PPG2RSP_RRWaveNet_Inspired\261113_FineTuning_StemAblation\   (step 2; its
      Comparison\261113_FineTuning_StemAblation_Comparison.xlsx holds the ablation)
  D:\PPG2RSP_RRWaveNet_Inspired\261114_FineTuning_StemAblation_RunAll\
      Logs\<time>_<step>.log, run_log.csv   (every step: start, end, minutes, exit code)

The two scripts are looked for in --scripts-dir, the folder of this file and its parent,
the working folder, D:\PPG2RSP_RRWaveNet_Inspired, its folders named like the scripts, and
every folder directly under it (the first match is used and printed).

Options of this file: --steps (finetune, ablation; default both), --scripts-dir, --log-root.
Every other option is given to both scripts unchanged (--plan: both only print their calls),
e.g. --gpu 1, --runs 02, --targets bidmc, or for a quick test
--no-require-cuda --epoch-cap 2 --subjects ... .
Spyder: runfile(r'D:\PPG2RSP_RRWaveNet_Inspired\...\261114_FineTuning_StemAblation_RunAll.py')
"""

import argparse
import csv
import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

NAME = "261114_FineTuning_StemAblation_RunAll"
WORKSPACE = Path(r"D:\PPG2RSP_RRWaveNet_Inspired")
OUTPUT_ROOT = WORKSPACE / NAME
HERE = Path(__file__).resolve().parent
# (step, script, description), in the order they run
STEPS: Tuple[Tuple[str, str, str], ...] = (
    ("finetune", "261109_RRWaveNet_FineTune.py", "fine-tuning with the 1x1 stem fusion"),
    ("ablation", "261113_FineTuning_StemAblation.py",
     "fine-tuning without the 1x1 stem fusion (ablation) and the comparison"),
)


def find_script(name: str, scripts_dir: Optional[Path]) -> Optional[Path]:
    folders: List[Path] = []
    if scripts_dir:
        folders.append(Path(scripts_dir))
    folders += [HERE, HERE.parent, Path.cwd(), WORKSPACE, WORKSPACE / Path(name).stem]
    for base in (WORKSPACE, HERE.parent):
        if base.is_dir():
            folders += sorted(child for child in base.iterdir() if child.is_dir())
    seen = set()
    for folder in folders:
        path = folder / name
        if path in seen:
            continue
        seen.add(path)
        if path.is_file():
            return path.resolve()
    return None


def command_text(command: Sequence[str]) -> str:
    return subprocess.list2cmdline(list(command))


def append_log_row(output_root: Path, row: dict) -> None:
    path = output_root / "run_log.csv"
    columns = ["Step", "Script", "Start", "End", "Minutes", "Exit_Code", "Log", "Command"]
    new = not path.exists()
    try:
        with open(path, "a", encoding="utf-8-sig" if new else "utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=columns)
            if new:
                writer.writeheader()
            writer.writerow({c: row.get(c, "") for c in columns})
    except PermissionError:  # run_log.csv open in Excel: the row is only printed
        print(f"  [run_log.csv is open in another program; not written] {row}")


def run_step(step: str, script: Path, options: List[str], output_root: Path) -> int:
    """One script in its own process; its output goes to this console (also Spyder) and to a log."""
    command = [sys.executable, "-u", str(script)] + options
    print(f"\n{'#' * 78}\n# {step}: {script.name}\n# {command_text(command)}\n{'#' * 78}", flush=True)
    start = datetime.now()
    log_dir = output_root / "Logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / f"{start:%Y%m%d_%H%M%S}_{step}.log"
    env = dict(os.environ, PYTHONUNBUFFERED="1", PYTHONIOENCODING="utf-8")
    with open(log_path, "w", encoding="utf-8") as log, subprocess.Popen(
            command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, env=env,
            text=True, encoding="utf-8", errors="replace", bufsize=1) as process:
        log.write(f"# {start:%Y-%m-%d %H:%M:%S} {command_text(command)}\n")
        try:
            for line in process.stdout:
                print(line, end="", flush=True)
                log.write(line)
                log.flush()
            code = process.wait()
        except BaseException:  # stopped (Ctrl+C / Spyder stop): the script stops too
            process.kill()
            raise
    end = datetime.now()
    minutes = (end - start).total_seconds() / 60.0
    append_log_row(output_root, {"Step": step, "Script": str(script), "Start": f"{start:%Y-%m-%d %H:%M:%S}",
                                 "End": f"{end:%Y-%m-%d %H:%M:%S}", "Minutes": round(minutes, 2),
                                 "Exit_Code": code, "Log": str(log_path), "Command": command_text(command)})
    print(f"\n[{step}] {script.name}: exit code {code} after {minutes:.1f} min (log: {log_path})", flush=True)
    return code


def parse_args(argv: Optional[Sequence[str]] = None) -> Tuple[argparse.Namespace, List[str]]:
    parser = argparse.ArgumentParser(
        description="Fine-tuning with the 1x1 stem fusion (261109) and without it (261113), one after the other")
    parser.add_argument("--steps", nargs="+", choices=[s[0] for s in STEPS], default=[s[0] for s in STEPS])
    parser.add_argument("--scripts-dir", type=Path, default=None, help="folder of the two scripts")
    parser.add_argument("--log-root", type=Path, default=OUTPUT_ROOT, help="folder of the logs of this file")
    return parser.parse_known_args(list(sys.argv[1:] if argv is None else argv))


def main(argv: Optional[Sequence[str]] = None) -> int:
    args, options = parse_args(argv)
    output_root = Path(args.log_root)
    output_root.mkdir(parents=True, exist_ok=True)
    steps = [s for s in STEPS if s[0] in args.steps]
    scripts = {}
    for step, name, description in steps:
        path = find_script(name, args.scripts_dir)
        if path is None:
            print(f"{name} not found (looked in the folder of this file, its parent, {WORKSPACE} and the folders "
                  f"under it); give its folder with --scripts-dir. Nothing is run.")
            return 2
        scripts[step] = path
    print(f"{NAME} ({datetime.now():%Y-%m-%d %H:%M:%S})")
    for index, (step, _, description) in enumerate(steps, start=1):
        print(f"  {index}. {step}: {description}\n     {scripts[step]}")
    if options:
        print(f"  options given to both scripts: {command_text(options)}")
    results = []
    for step, _, _ in steps:
        code = run_step(step, scripts[step], options, output_root)
        results.append((step, code))
        if code != 0:
            print(f"[{step}] ended with exit code {code}; the next step is still run "
                  "(a rerun of this file continues what is not finished)", flush=True)
    print(f"\n{'=' * 78}\n{NAME}: done")
    for step, code in results:
        print(f"  {step:<9} exit code {code}" + ("" if code == 0 else "  <- see its log; rerun this file to continue"))
    print(f"  logs: {output_root / 'Logs'}")
    return next((code for _, code in results if code != 0), 0)


if __name__ == "__main__":
    exit_code = main()
    if exit_code:
        sys.exit(exit_code)
