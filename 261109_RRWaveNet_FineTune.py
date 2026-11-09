# -*- coding: utf-8 -*-
r"""261109_RRWaveNet_FineTune: the RRWaveNet runs 261013 and 261007 with transfer
learning - pretrained on one dataset (default CapnoBase), then fine-tuned by LOSOCV on
the others (default STEAM2, BIDMC) - and the comparison with the same runs trained
from scratch. Every result is saved under one folder.

Runs, in this order
  01  261013: pretraining on CapnoBase -> fine-tuning on STEAM2 -> BIDMC
  02  261007: pretraining on CapnoBase -> fine-tuning on STEAM2 -> BIDMC
  (--source and --targets choose other datasets, e.g. --source steam2 --targets bidmc capnobase)

1. Pretraining (every subject of the source dataset; no subject is held out, because the
   targets are other datasets), with the training of the version and nothing else changed:
     261007  validation = the last 20% of every subject's windows (temporal, as 261007);
             full model, weight decay 5e-5; windows 10 s and 20 s (its default)
     261013  validation = 20% of the subjects held out whole (subject-wise, drawn once with
             a seed of the dataset); member m0 trained with every weight decay candidate
             (0.01 / 0.1 / 1), the one with the lowest validation loss chosen, m1 and m2
             trained with it; window 10 s
   same epochs, minimum epochs, patience, batch size, learning rate and schedule, loss,
   augmentation, early stopping (best validation checkpoint) as the version; three members
   m0-m2, each with its own seed. Saved in <run>\Pretrained_<source>\<NN_window>\m<k>_best.pth
   (+ history, summary; 261013: WD_Candidates\ and selection.json).
2. Fine-tuning: the code of the version (generated copy, Code\) with exactly these changes
   (each must match once, otherwise nothing runs):
     a. PROJECT_ROOT -> the run's folder
     b. PRETRAINED_ROOT (the folder above) and PRETRAINED_SOURCE after DEFAULT_RESULTS_ROOT
     c. pretrained_state(): the weights of a member, before train_one
     d. in train_one, after the model is built: the weights of the same member (m0 <- m0,
        m1 <- m1, m2 <- m2; 261013: every weight decay candidate of m0 starts from m0)
   Everything else is the version's LOSOCV unchanged: data, preprocessing, folds,
   validation, weight decay selection (261013), learning rate and schedule (from the
   start, every layer trained), epochs, early stopping, ensemble, evaluation, summary.
   A run is one call per target dataset (STEAM2, then BIDMC) with --resume, then one
   --summary-only call over the targets; 261007 runs the full model only (--variants full).
3. Comparison: fine-tuned vs from scratch = the earlier runs with the same version
   (D:\PPG2RSP_RRWaveNet_Inspired\261013_RRWavNet_STEAM_BIDMC_v1, 261007_RRWaveNet_NoEvent_v1;
   read only): per subject FFT MAE and waveform PCC, Wilcoxon signed-rank, BH, and the check
   that both are scored on the same minutes against the same ground truth.

Outputs (D:\PPG2RSP_RRWaveNet_Inspired\261109_RRWaveNet_FineTune)
  Code\                     the generated scripts, changes.txt
  01_261013_CapnoBase-to-STEAM2-BIDMC\   Pretrained_CapnoBase\ and the LOSOCV results in the
  02_261007_CapnoBase-to-STEAM2-BIDMC\   layout of the base run (Summary\..._results.xlsx, ...)
  run_log.csv, Logs\<run>_<step>.log     every call and its console output
  Comparison\261109_RRWaveNet_FineTune_Comparison.xlsx (+ csv)
    Summary, Paired_Tests, Pretraining (per member: best epoch, validation loss, weight
    decay), Per_Subject, Run_Status, Notes

Run (needs CUDA as the base scripts):
  python 261109_RRWaveNet_FineTune.py
  python 261109_RRWaveNet_FineTune.py --plan                 print the calls only
  python 261109_RRWaveNet_FineTune.py --compare-only         only the comparison
  python 261109_RRWaveNet_FineTune.py --runs 02              only 261007
  python 261109_RRWaveNet_FineTune.py --source capnobase --targets steam2 bidmc
Spyder: runfile(r'D:\PPG2RSP_RRWaveNet_Inspired\...\261109_RRWaveNet_FineTune.py', args='--plan')
Other options go to every call unchanged (e.g. --gpu 1, --no-require-cuda, --epoch-cap 5 for a
quick test). Options of this file: --runs, --source, --targets, --scripts-dir, --output-root,
--args-261007, --args-261013, --no-resume, --write-only, --plan, --compare-only
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

NAME = "261109_RRWaveNet_FineTune"
WORKSPACE = Path(r"D:\PPG2RSP_RRWaveNet_Inspired")
OUTPUT_ROOT = WORKSPACE / NAME
DATASET_LABELS = {"steam2": "STEAM2", "bidmc": "BIDMC", "capnobase": "CapnoBase"}
DATASET_RUN_ORDER = ("steam2", "bidmc", "capnobase")
DEFAULT_SOURCE = "capnobase"
DEFAULT_TARGETS = ("steam2", "bidmc")
EXCLUDED = {"STEAM2": ("22_LMS", "26_CKK", "12_YJJ"), "BIDMC": ("27",), "CapnoBase": ("0115",)}
PRETRAIN_TAG = "pretraining"  # the "subject" in the seeds of the pretrained members

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
RUNS = (("01", "261013"), ("02", "261007"))  # the order of the runs

ROOT_PATTERN = re.compile(r'^PROJECT_ROOT = Path\(r"[^"\n]*"\)$', re.MULTILINE)
RESULTS_LINE = "DEFAULT_RESULTS_ROOT = PROJECT_ROOT\n"
TRAIN_ONE_LINE = "def train_one(\n"
BUILD_LINE = "    model = model_options.build().to(device)\n"


# ---------------------------------------------------------------------------
# Folders and the generated scripts
# ---------------------------------------------------------------------------

def run_folder(output_root: Path, run_id: str, version: str, source: str, targets: Sequence[str]) -> Path:
    return output_root / (f"{run_id}_{version}_{DATASET_LABELS[source]}-to-"
                          f"{'-'.join(DATASET_LABELS[t] for t in targets)}")


def pretrained_folder(run_root: Path, source: str) -> Path:
    return run_root / f"Pretrained_{DATASET_LABELS[source]}"


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


def patched_text(text: str, base_name: str, root: Path, pretrained: Path, source: str) -> str:
    """The base script with exactly the four changes; stops when one does not match once."""
    text = text.replace("\r\n", "\n")
    for old, label in ((RESULTS_LINE, "DEFAULT_RESULTS_ROOT line"), (TRAIN_ONE_LINE, "train_one definition"),
                       (BUILD_LINE, "model build line of train_one")):
        count = text.count(old)
        if count != 1:
            raise SystemExit(f"{base_name}: the {label} was found {count} times (expected once); nothing is run")
    if len(ROOT_PATTERN.findall(text)) != 1:
        raise SystemExit(f"{base_name}: PROJECT_ROOT line not found once; nothing is run")
    results_new = (RESULTS_LINE +
                   f"# {NAME}: every member of a LOSOCV fold starts from the same member pretrained on the source\n"
                   f'PRETRAINED_ROOT = Path(r"{pretrained}")\n'
                   f'PRETRAINED_SOURCE = "{DATASET_LABELS[source]}"\n')
    helper = (
        "def pretrained_state(spec: WindowSpec, member_root: Path, device: torch.device) -> dict:\n"
        f'    """{NAME}: the weights of the same member (m0, m1, ...) pretrained on PRETRAINED_SOURCE\n'
        '    (written by the pretraining of 261109_RRWaveNet_FineTune)."""\n'
        '    path = PRETRAINED_ROOT / f"{spec.order:02d}_{spec.name}" / f"{member_root.name}_best.pth"\n'
        "    if not path.exists():\n"
        '        raise RuntimeError(f"Pretrained model not found: {path} (run the pretraining first)")\n'
        "    return torch.load(path, map_location=device)\n"
        "\n\n" + TRAIN_ONE_LINE)
    build_new = (BUILD_LINE +
                 f"    # {NAME}: fine-tuning - the model starts from the pretrained weights of this member\n"
                 "    model.load_state_dict(pretrained_state(spec, member_root, device))\n")
    text = (text.replace(RESULTS_LINE, results_new).replace(TRAIN_ONE_LINE, helper)
            .replace(BUILD_LINE, build_new))
    root_line = f'PROJECT_ROOT = Path(r"{root}")'
    # a function as replacement: the backslashes of a Windows path are kept as they are (no regex escapes)
    text = ROOT_PATTERN.sub(lambda _: root_line, text)
    head = (f"# {NAME}: generated from {base_name}; the only changes: every member starts from the member "
            f"pretrained on {DATASET_LABELS[source]} (PRETRAINED_ROOT, pretrained_state, train_one) and "
            f"PROJECT_ROOT = {root}\n")
    if text.startswith("# -*- coding"):
        first, rest = text.split("\n", 1)
        return f"{first}\n{head}{rest}"
    return head + text


def write_scripts(args: argparse.Namespace, runs: Sequence[tuple]) -> Dict[str, Path]:
    code_dir = Path(args.output_root) / "Code"
    code_dir.mkdir(parents=True, exist_ok=True)
    report = [f"{NAME} ({datetime.now():%Y-%m-%d %H:%M:%S})",
              "Changes in each script: 1. PROJECT_ROOT  2. PRETRAINED_ROOT / PRETRAINED_SOURCE  3. pretrained_state()"
              "  4. train_one loads the pretrained member after building the model", ""]
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
        root = run_folder(Path(args.output_root), run_id, version, args.source, args.targets)
        pretrained = pretrained_folder(root, args.source)
        path = code_dir / f"{NAME}_{run_id}_{version}.py"
        # open(..., newline="\n"): Path.write_text has no newline argument before Python 3.10
        with open(path, "w", encoding="utf-8", newline="\n") as handle:
            handle.write(patched_text(text, base.name, root, pretrained, args.source))
        scripts[run_id] = path
        report.append(f"[{run_id}] {version}: {path.name} -> {root} (pretrained: {pretrained})")
    (code_dir / "changes.txt").write_text("\n".join(report) + "\n", encoding="utf-8")
    return scripts


# ---------------------------------------------------------------------------
# Pretraining (run in its own process: python 261109_RRWaveNet_FineTune.py --pretrain-worker ...)
# ---------------------------------------------------------------------------

def load_module(script: Path):
    spec = importlib.util.spec_from_file_location(f"ft_{abs(hash(str(script)))}", str(script))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def pretrain_options(argv: Sequence[str]) -> argparse.Namespace:
    """The options of the version that the pretraining follows (taken from the options of the calls)."""
    p = argparse.ArgumentParser(add_help=False)
    p.add_argument("--windows", nargs="+", default=None)
    p.add_argument("--members", type=int, default=None)
    p.add_argument("--min-epochs", type=int, default=None)
    p.add_argument("--patience", type=int, default=None)
    p.add_argument("--batch-size", type=int, default=None)
    p.add_argument("--epoch-cap", type=int, default=None)
    p.add_argument("--weight-decays", nargs="+", type=float, default=None)
    p.add_argument("--gpu", type=int, default=0)
    p.add_argument("--force-cpu", action="store_true")
    p.add_argument("--no-require-cuda", dest="require_cuda", action="store_false")
    p.add_argument("--threads", type=int, default=None)
    return p.parse_known_args(list(argv))[0]


def pretrain_member(mod, version: str, spec, source: str, subject_data: dict, subjects: Sequence[str],
                    val_subjects: Sequence[str], variant, member: int, out_dir: Path, device, opt) -> dict:
    """One pretrained member (resumable); returns its summary."""
    out_dir.mkdir(parents=True, exist_ok=True)
    checkpoint = out_dir / f"m{member}_best.pth"
    summary_path = out_dir / f"m{member}_summary.json"
    if summary_path.exists() and checkpoint.exists():
        return json.loads(summary_path.read_text(encoding="utf-8"))
    state_path = out_dir / f"m{member}_training_state.pt"
    seed = int(mod.member_seed(spec, source, PRETRAIN_TAG, member))
    mod.set_seed(seed)
    if version == "261007":  # temporal validation: the last 20% of every subject's windows
        fit_subjects = list(subjects)
        train_ds = mod.SubjectWindowDataset(subject_data, fit_subjects, "train", True)
        val_ds = mod.SubjectWindowDataset(subject_data, fit_subjects, "val", False)
    else:  # subject-wise validation
        held = {str(s) for s in val_subjects}
        fit_subjects = [s for s in subjects if str(s) not in held]
        train_ds = mod.SubjectWindowDataset(subject_data, fit_subjects, "train", True)
        val_ds = mod.SubjectWindowDataset(subject_data, list(val_subjects), "val", False)
    batch = opt.batch_size or mod.BATCH_SIZE
    train_loader = mod.make_train_loader(train_ds, batch, device)
    val_loader = mod.make_eval_loader(val_ds, batch, device)
    epochs = variant.epochs if opt.epoch_cap is None else min(variant.epochs, opt.epoch_cap)
    min_epochs = opt.min_epochs if opt.min_epochs is not None else mod.MIN_EPOCHS
    patience = opt.patience if opt.patience is not None else mod.PATIENCE
    model = variant.options.build().to(device)
    start = time.time()
    if version == "261007":
        history = mod.fit_model(model, train_loader, val_loader, device, epochs, min_epochs, patience,
                                checkpoint, state_path)
    else:
        history = mod.fit_model(model, train_loader, val_loader, device, epochs, min_epochs, patience,
                                checkpoint, state_path, variant.options.weight_decay)
    pd.DataFrame(history).to_csv(out_dir / f"m{member}_Learning_Curve.csv", index=False, encoding="utf-8-sig")
    audit = mod.build_training_audit(history, checkpoint)
    if audit["Audit_Status"] != "PASS_BEST_VALIDATION_CHECKPOINT":
        raise RuntimeError(f"No valid pretrained checkpoint: {checkpoint}")
    summary = {"Source": DATASET_LABELS[source], "Window_Config": spec.name, "Member": member, "Seed": seed,
               "Variant": variant.name, "Weight_Decay": float(getattr(variant.options, "weight_decay",
                                                                      getattr(mod, "WEIGHT_DECAY", math.nan))),
               "Validation": ("temporal (last 20% of every subject)" if version == "261007" else
                              "subject-wise: " + " ".join(map(str, val_subjects))),
               "Subjects": len(subjects), "Fit_Subjects": len(fit_subjects),
               "Train_Windows": len(train_ds), "Val_Windows": len(val_ds), "Max_Epochs": int(epochs),
               "Best_Epoch": audit.get("Best_Epoch"), "Best_Val_Loss": audit.get("Best_Val_Loss"),
               "History_Epochs": audit.get("History_Epochs"), "Elapsed_Sec": time.time() - start}
    summary_path.write_text(json.dumps(summary, indent=2, default=str), encoding="utf-8")
    if state_path.exists():
        state_path.unlink()
    del model, train_loader, val_loader
    mod.release_memory(device)
    return summary


def pretrain(mod, version: str, source: str, out_root: Path, opt) -> None:
    """Every window condition of the version: the pretrained members m0.. on every subject of the source."""
    import torch
    if opt.threads:
        torch.set_num_threads(int(opt.threads))
    if opt.require_cuda and not torch.cuda.is_available():
        raise RuntimeError("CUDA is required by default. Use --no-require-cuda to run on the CPU.")
    device = torch.device(f"cuda:{opt.gpu}" if torch.cuda.is_available() and not opt.force_cpu else "cpu")
    members = opt.members or mod.ENSEMBLE_MEMBERS
    windows = opt.windows or list(mod.DEFAULT_WINDOWS)
    specs = [spec for spec in mod.WINDOW_SPECS if spec.name in windows]
    print(f"Pretraining {version} on {DATASET_LABELS[source]}: windows {[s.name for s in specs]}, {members} members, "
          f"device {device}", flush=True)
    for spec in specs:
        out_dir = out_root / f"{spec.order:02d}_{spec.name}"
        if all((out_dir / f"m{m}_best.pth").exists() and (out_dir / f"m{m}_summary.json").exists()
               for m in range(members)):
            print(f"  {spec.name}: pretrained members found, kept")
            continue
        subject_data = mod.load_dataset(source, spec)
        subjects = sorted(subject_data)
        if len(subjects) < 3:
            raise RuntimeError(f"{DATASET_LABELS[source]}: fewer than three valid subjects")
        print(f"  {spec.name}: {len(subjects)} subjects of {DATASET_LABELS[source]}", flush=True)
        if version == "261007":
            variant = mod.VARIANTS["full"]
            for member in range(members):
                s = pretrain_member(mod, version, spec, source, subject_data, subjects, [], variant, member,
                                    out_dir, device, opt)
                print(f"    m{member}: best epoch {s['Best_Epoch']}/{s['History_Epochs']}, "
                      f"val loss {s['Best_Val_Loss']:.4f}", flush=True)
        else:
            # validation subjects drawn once (seed of the source dataset), as one 261013 fold without a test subject
            val_subjects = mod.validation_subjects(source, PRETRAIN_TAG, subjects)
            candidates = mod.build_variants(opt.weight_decays or mod.WEIGHT_DECAY_CANDIDATES)
            sel = mod.SELECTION_MEMBER
            rows = []
            for variant in candidates.values():
                wd_dir = out_dir / "WD_Candidates" / variant.name
                s = pretrain_member(mod, version, spec, source, subject_data, subjects, val_subjects, variant, sel,
                                    wd_dir, device, opt)
                rows.append({"Weight_Decay": float(variant.options.weight_decay), "Variant": variant.name,
                             "Best_Val_Loss": float(s["Best_Val_Loss"]), "Best_Epoch": s["Best_Epoch"]})
                print(f"    m{sel} wd {variant.options.weight_decay:g}: val loss {s['Best_Val_Loss']:.4f}", flush=True)
            best = min(rows, key=lambda row: (row["Best_Val_Loss"], row["Weight_Decay"]))
            chosen = candidates[best["Variant"]]
            (out_dir / "selection.json").write_text(json.dumps(
                {"Source": DATASET_LABELS[source], "Window_Config": spec.name, "Validation_Subjects": val_subjects,
                 "Candidates": rows, "Selected_Weight_Decay": best["Weight_Decay"],
                 "Selected_Variant": best["Variant"]}, indent=2, default=str), encoding="utf-8")
            for suffix in ("_best.pth", "_summary.json", "_Learning_Curve.csv"):
                shutil.copy2(out_dir / "WD_Candidates" / chosen.name / f"m{sel}{suffix}", out_dir / f"m{sel}{suffix}")
            print(f"    weight decay {chosen.options.weight_decay:g} chosen", flush=True)
            for member in range(members):
                if member == sel:
                    continue
                s = pretrain_member(mod, version, spec, source, subject_data, subjects, val_subjects, chosen,
                                    member, out_dir, device, opt)
                print(f"    m{member}: best epoch {s['Best_Epoch']}/{s['History_Epochs']}, "
                      f"val loss {s['Best_Val_Loss']:.4f}", flush=True)
        del subject_data
        mod.release_memory(device)


def pretrain_worker(argv: Sequence[str]) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--pretrain-worker", action="store_true")
    p.add_argument("--version", required=True, choices=list(VERSIONS))
    p.add_argument("--script", type=Path, required=True)
    p.add_argument("--source", required=True, choices=list(DATASET_LABELS))
    p.add_argument("--out", type=Path, required=True)
    args, rest = p.parse_known_args(list(argv))
    pretrain(load_module(args.script), args.version, args.source, args.out, pretrain_options(rest))
    return 0


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
    return list(VERSIONS[version]["options"]) + others + split_options(
        args.args_261007 if version == "261007" else args.args_261013)


def run_commands(run: tuple, script: Path, args: argparse.Namespace, others: List[str]) -> Tuple[List[str], List[str]]:
    run_id, version = run
    base = [sys.executable, str(script)]
    extra = version_options(version, args, others)
    if version == "261013" and "--reference-root" not in extra:
        partner = next(r for r in RUNS if r[1] == "261007")
        extra += ["--reference-root", str(run_folder(Path(args.output_root), *partner, args.source, args.targets))]
    resume = [] if args.no_resume or "--resume" in extra else ["--resume"]
    return base + resume + extra, base + ["--summary-only"] + extra


def execute(args: argparse.Namespace, others: List[str], runs: Sequence[tuple], scripts: Dict[str, Path]) -> int:
    status = 0
    output_root = Path(args.output_root)
    for run in runs:
        run_id, version = run
        root = run_folder(output_root, run_id, version, args.source, args.targets)
        print(f"\n### Run {run_id}: {version}, pretraining on {DATASET_LABELS[args.source]} -> fine-tuning on "
              f"{' -> '.join(DATASET_LABELS[t] for t in args.targets)}\n    {root}", flush=True)
        worker = [sys.executable, str(Path(__file__).resolve()), "--pretrain-worker", "--version", version,
                  "--script", str(scripts[run_id]), "--source", args.source,
                  "--out", str(pretrained_folder(root, args.source))] + version_options(version, args, others)
        code = call(worker, output_root, run_id, f"pretraining {DATASET_LABELS[args.source]}", args.plan)
        if code != 0:
            print(f"[{run_id}] pretraining failed (exit code {code}); its fine-tuning is skipped", flush=True)
            status = status or code
            continue
        train, summary = run_commands(run, scripts[run_id], args, others)
        for dataset in args.targets:
            since = datetime.now().timestamp()
            code = call(train + ["--datasets", dataset], output_root, run_id, f"fine-tuning {DATASET_LABELS[dataset]}",
                        args.plan)
            status = status or code
            if not args.plan:
                keep_call_files(root, dataset, since)
        status = status or call(summary + ["--datasets", *args.targets], output_root, run_id, "summary", args.plan)
        if version == "261007":
            partner = next((r for r in RUNS if r[1] == "261013"), None)
            if partner and partner[0] in scripts:
                _, partner_summary = run_commands(partner, scripts[partner[0]], args, others)
                call(partner_summary + ["--datasets", *args.targets], output_root, partner[0], "summary vs 261007",
                     args.plan)
    return status


# ---------------------------------------------------------------------------
# Comparison: fine-tuned vs from scratch
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


def pretraining_table(output_root: Path, source: str, targets: Sequence[str]) -> pd.DataFrame:
    rows = []
    for run_id, version in RUNS:
        root = pretrained_folder(run_folder(output_root, run_id, version, source, targets), source)
        for path in sorted(root.glob("*/m*_summary.json")):
            try:
                rows.append({"Run": run_id, "Version": version, **json.loads(path.read_text(encoding="utf-8"))})
            except Exception:
                continue
    return pd.DataFrame(rows)


def compare(output_root: Path, source: str, targets: Sequence[str]) -> Optional[Path]:
    fine_label = f"Fine-tuned ({DATASET_LABELS[source]} pretraining)"
    scratch_label = "From scratch (earlier run)"
    target_labels = {DATASET_LABELS[t] for t in targets}
    runs = []
    for run_id, version in RUNS:
        runs.append((run_folder(output_root, run_id, version, source, targets), run_id, fine_label, version))
        runs.append((WORKSPACE / VERSIONS[version]["earlier_run"], "earlier", scratch_label, version))
    frames, status = [], []
    for root, run_id, label, version in runs:
        per_subject = read_csv(root / "Summary" / "per_subject_fft.csv")
        if per_subject.empty:
            continue
        if "Row" in per_subject.columns:  # 261007: the full model's ensemble
            per_subject = per_subject[per_subject["Row"].astype(str) == "full"]
        per_subject = per_subject[per_subject["Dataset"].isin(target_labels)]
        if per_subject.empty:
            continue
        frame = per_subject.copy()
        frame.insert(0, "Training", label)
        frame.insert(0, "Version", version)
        frame.insert(0, "Run", run_id)
        frames.append(frame)
        for (dataset, window), group in per_subject.groupby(["Dataset", "Window_Config"]):
            status.append({"Run": run_id, "Version": version, "Training": label, "Folder": str(root),
                           "Dataset": dataset, "Window_Config": window, "Subjects": int(group["Subject"].nunique())})
    if not any(f["Training"].iloc[0] == fine_label for f in frames):
        print("Comparison: no fine-tuned run finished yet (no Summary\\per_subject_fft.csv)")
        return None
    per_subject = pd.concat(frames, ignore_index=True)
    ours = per_subject[per_subject["Training"] == fine_label][["Dataset", "Version", "Window_Config"]].drop_duplicates()
    per_subject = per_subject.merge(ours, on=["Dataset", "Version", "Window_Config"], how="inner")
    per_subject["Issue_Subject"] = [is_issue(d, s) for d, s in zip(per_subject["Dataset"], per_subject["Subject"])]
    sets = (("All subjects", per_subject), ("Issue subjects left out", per_subject[~per_subject["Issue_Subject"]]))
    order = {"Dataset": {DATASET_LABELS[d]: i for i, d in enumerate(DATASET_RUN_ORDER)},
             "Version": {v: i for i, v in enumerate(VERSIONS)}, "Training": {fine_label: 0, scratch_label: 1}}

    def ordered(frame: pd.DataFrame, columns: List[str]) -> pd.DataFrame:
        return frame.sort_values(columns, key=lambda s: s.map(order[s.name]).fillna(99) if s.name in order else s,
                                 kind="mergesort").reset_index(drop=True)

    keys = ["Dataset", "Version", "Training", "Window_Config"]
    summary_rows, tests = [], []
    for subject_set, part in sets:
        for key, group in part.groupby(keys, sort=False):
            mae, pcc = describe(group["FFT_MAE"].to_numpy(float)), describe(group["Wave_PCC_Mean"].to_numpy(float))
            summary_rows.append({"Subject_Set": subject_set, **dict(zip(keys, key)), "Subjects": int(group["Subject"].nunique()),
                                 **{f"FFT_MAE_{k}": v for k, v in mae.items()}, **{f"PCC_{k}": v for k, v in pcc.items()},
                                 "FFT_MAE_Text": f"{mae['Mean']:.2f} ± {mae['SD']:.2f} ({mae['Median']:.2f} [{mae['Q1']:.2f}, {mae['Q3']:.2f}])",
                                 "PCC_Text": f"{pcc['Mean']:.3f} ± {pcc['SD']:.3f} ({pcc['Median']:.3f})"})
        for (dataset, version, window), group in part.groupby(["Dataset", "Version", "Window_Config"], sort=False):
            a = group[group["Training"] == fine_label].set_index("Subject")
            b = group[group["Training"] == scratch_label].set_index("Subject")
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
                              "Metric": metric, "A": fine_label, "B": scratch_label, "Paired_Subjects": int(x.size),
                              "Same_Minutes_And_GT": f"{same}/{len(common)}",
                              "Median_A": float(np.median(x)) if x.size else math.nan,
                              "Median_B": float(np.median(y)) if y.size else math.nan,
                              "Median_Difference_A_minus_B": float(np.median(x - y)) if x.size else math.nan,
                              "A_Better": int(np.sum(x < y) if better_low else np.sum(x > y)),
                              "B_Better": int(np.sum(y < x) if better_low else np.sum(y > x)),
                              "Wilcoxon_p": wilcoxon_p(x, y)})
    summary = ordered(pd.DataFrame(summary_rows), ["Subject_Set", "Dataset", "Version", "Window_Config", "Training"])
    tests = pd.DataFrame(tests)
    if not tests.empty:
        tests["BH_p"] = np.nan
        for _, idx in tests.groupby(["Subject_Set", "Metric"]).groups.items():
            tests.loc[idx, "BH_p"] = bh(tests.loc[idx, "Wilcoxon_p"].tolist())
        tests = ordered(tests, ["Subject_Set", "Metric", "Dataset", "Version", "Window_Config"])
    notes = pd.DataFrame([
        ("Runs", f"01 261013, 02 261007: pretraining on {DATASET_LABELS[source]} (every subject), fine-tuning by "
                 f"LOSOCV on {', '.join(DATASET_LABELS[t] for t in targets)}; from scratch = the earlier runs "
                 "(261013_RRWavNet_STEAM_BIDMC_v1, 261007_RRWaveNet_NoEvent_v1), read only"),
        ("Change", "only the start of every member's training: the weights of the same member pretrained on the "
                   "source; data, preprocessing, folds, validation, weight decay selection, learning rate and "
                   "schedule, epochs, early stopping, ensemble and evaluation as the version"),
        ("Per subject", "Summary\\per_subject_fft.csv: FFT_MAE and Wave_PCC_Mean; 261007 the full model's ensemble, "
                        "261013 the final ensemble"),
        ("Paired_Tests", "same subjects; Wilcoxon signed-rank; BH over all tests of a subject set and metric; "
                         "A_Better = subjects with lower FFT_MAE / higher PCC when fine-tuned; Same_Minutes_And_GT: "
                         "subjects scored on the same minutes against the same ground truth in both"),
        ("Issue subjects", "; ".join(f"{d} {', '.join(s)}" for d, s in EXCLUDED.items())),
    ], columns=["Item", "Description"])
    sheets = {"Summary": summary, "Paired_Tests": tests, "Pretraining": pretraining_table(output_root, source, targets),
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
        print("\nFFT MAE / PCC, fine-tuned vs from scratch (issue subjects left out):")
        print(show[["Dataset", "Version", "Window_Config", "Training", "Subjects", "FFT_MAE_Text",
                    "PCC_Text"]].to_string(index=False))
    print(f"\nComparison: {path}")
    return path


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args(argv: Optional[Sequence[str]] = None) -> Tuple[argparse.Namespace, List[str]]:
    parser = argparse.ArgumentParser(description="RRWaveNet 261013 / 261007: pretraining on one dataset, fine-tuning "
                                                 "by LOSOCV on the others, and the comparison with training from scratch")
    parser.add_argument("--runs", nargs="+", choices=[r[0] for r in RUNS], default=[r[0] for r in RUNS])
    parser.add_argument("--source", choices=list(DATASET_LABELS), default=DEFAULT_SOURCE)
    parser.add_argument("--targets", nargs="+", choices=list(DATASET_LABELS), default=list(DEFAULT_TARGETS))
    parser.add_argument("--scripts-dir", type=Path, default=None, help="folder of the two base scripts")
    parser.add_argument("--output-root", type=Path, default=OUTPUT_ROOT)
    parser.add_argument("--args-261007", default="", help='options only for 261007, e.g. "--windows w10s_nonoverlap"')
    parser.add_argument("--args-261013", default="", help='options only for 261013, e.g. "--members 3"')
    parser.add_argument("--no-resume", action="store_true", help="do not pass --resume to the fine-tuning calls")
    parser.add_argument("--write-only", action="store_true", help="write the scripts, do not run")
    parser.add_argument("--plan", action="store_true", help="print every call, do not run")
    parser.add_argument("--compare-only", action="store_true", help="only the comparison of the finished runs")
    # --args-261007 "--windows ...": a value starting with "--" would be read as an option; join it
    tokens, joined = list(sys.argv[1:] if argv is None else argv), []
    while tokens:
        token = tokens.pop(0)
        if token in ("--args-261007", "--args-261013") and tokens:
            token = f"{token}={tokens.pop(0)}"
        joined.append(token)
    args, others = parser.parse_known_args(joined)
    if args.source in args.targets:
        parser.error("the source dataset cannot be a target")
    args.targets = [d for d in DATASET_RUN_ORDER if d in args.targets]  # STEAM2 -> BIDMC -> CapnoBase
    return args, others


def main(argv: Optional[Sequence[str]] = None) -> int:
    tokens = list(sys.argv[1:] if argv is None else argv)
    if "--pretrain-worker" in tokens:
        return pretrain_worker(tokens)
    args, others = parse_args(tokens)
    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    if args.compare_only:
        compare(output_root, args.source, args.targets)
        return 0
    runs = [r for r in RUNS if r[0] in args.runs]
    print(f"{NAME}: {', '.join(f'{r[0]} {r[1]}' for r in runs)}; pretraining on {DATASET_LABELS[args.source]}, "
          f"fine-tuning on {' -> '.join(DATASET_LABELS[t] for t in args.targets)}\nResults: {output_root}")
    scripts = write_scripts(args, runs)
    if args.write_only:
        print(f"\nWritten only (--write-only): {output_root / 'Code'}")
        return 0
    status = execute(args, others, runs, scripts)
    if not args.plan:
        compare(output_root, args.source, args.targets)
    return status


if __name__ == "__main__":
    sys.exit(main())
