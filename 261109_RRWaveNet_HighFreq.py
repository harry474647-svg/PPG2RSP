# -*- coding: utf-8 -*-
r"""261109_RRWaveNet_HighFreq: RRWaveNet of 261007 (CapnoBase) and 261013
(BIDMC, STEAM2) with the input PPG band-passed 0.1-8.0 Hz. Everything else is
the code of those two runs, unchanged.

What changes (nothing else)
---------------------------
The two runs fed the model the raw PPG (PPG_INPUT = "raw": the resampled
64 Hz PPG, min-max [0, 1] per window, no filter). Here the model input is the
same PPG band-passed 0.1-8.0 Hz (3rd-order zero-phase Butterworth in
second-order sections: sos_bandpass of the scripts), filtered over the whole
segment before the windows are cut; the min-max per window follows as before.
The filtered signal is what the model sees in training, validation and the
held-out test inference (the windows "low" of every segment are cut from it).
Text changes made to each script (each must be found exactly once, otherwise
nothing runs):
  1. PPG_INPUT = "raw"  ->  PPG_INPUT = "bandpass", PPG_INPUT_BAND_HZ = (0.1, 8.0),
                            PPG_INPUT_FILTER_ORDER = 3
  2. model_input = (ppg_raw if PPG_INPUT == "raw" else ppg_low)...
        -> with PPG_INPUT == "bandpass": sos_bandpass(ppg_raw, fs, 0.1, 8.0, order=3)
  3. PROJECT_ROOT -> the new results folder (below), so no earlier result is touched.
Unchanged: model, loss, training, ensembles, validation (261007: last 20% of
every training subject; 261013: validation subjects and weight-decay
selection), LOSOCV, quality rules, targets, scoring, outputs and every
command-line option of the two scripts.

Base scripts (searched in --scripts-dir, the folder of this file, the current
folder, D:\PPG2RSP_RRWaveNet_Inspired, its run folders and their parents):
  261007_RRWaveNet_Ensemble_NoEvent.py     -> CapnoBase (as before)
  261013_RRWaveNet_Adult_SubjectVal_WD.py  -> BIDMC, STEAM2 (as before)
Their text is compared (results-folder line left out) with the versions this
code was checked against; another version is reported and still used.

Outputs (default D:\PPG2RSP_RRWaveNet_Inspired\261109_RRWaveNet_HighFreq)
  Code\261109_RRWaveNet_HighFreq_from_261007.py   the full script that runs (changes above)
  Code\261109_RRWaveNet_HighFreq_from_261013.py
  Code\changes.txt                                 base scripts, hashes, the three changes
  261007_RRWaveNet_NoEvent_HighFreq\               results of the 261007 part (same layout
                                                   as 261007_RRWaveNet_NoEvent_v1)
  261013_RRWaveNet_STEAM_BIDMC_HighFreq\           results of the 261013 part (same layout
                                                   as 261013_RRWavNet_STEAM_BIDMC_v1); its paired
                                                   comparison reads the 261007 part above
Run:
  python 261109_RRWaveNet_HighFreq.py
    both parts with the scripts' own defaults (261007: CapnoBase, 10 s and 20 s
    windows, every ablation variant; 261013: BIDMC and STEAM2, 10 s windows,
    weight-decay selection)
  python 261109_RRWaveNet_HighFreq.py --parts 261007
  python 261109_RRWaveNet_HighFreq.py --args-261007 "--variants full --windows w10s_nonoverlap"
  python 261109_RRWaveNet_HighFreq.py --resume        (any other option goes to both scripts)
Options of this file: --parts, --scripts-dir, --output-root, --datasets-261007,
  --datasets-261013, --args-261007, --args-261013, --write-only (write the two
  scripts, do not run)
"""

import argparse
import hashlib
import re
import shlex
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

NAME = "261109_RRWaveNet_HighFreq"
WORKSPACE = Path(r"D:\PPG2RSP_RRWaveNet_Inspired")
OUTPUT_ROOT = WORKSPACE / NAME
PPG_INPUT_BAND_HZ = (0.1, 8.0)
PPG_INPUT_FILTER_ORDER = 3

# part -> base script, earlier run folder (where the script may sit), new results folder, datasets of the earlier use,
# SHA-256 of the base text (CRLF -> LF, PROJECT_ROOT line left out) this code was checked against
PARTS = {
    "261007": {
        "script": "261007_RRWaveNet_Ensemble_NoEvent.py",
        "run_folder": "261007_RRWaveNet_NoEvent_v1",
        "results": "261007_RRWaveNet_NoEvent_HighFreq",
        "datasets": ("capnobase",),
        "sha256": "18ed5840118f660f7c80ef6c7ed34744597675ba9ebd8115703aa265739f1841",
    },
    "261013": {
        "script": "261013_RRWaveNet_Adult_SubjectVal_WD.py",
        "run_folder": "261013_RRWavNet_STEAM_BIDMC_v1",
        "results": "261013_RRWaveNet_STEAM_BIDMC_HighFreq",
        "datasets": ("bidmc", "steam2"),
        "sha256": "364b109b1e0da1f7ed6ea4a9276a77ed0ad554363b32605cb6484a37642ca3dc",
    },
}

INPUT_LINE = 'PPG_INPUT = "raw"\n'
INPUT_NEW = (
    f'PPG_INPUT = "bandpass"  # {NAME}: the PPG band-passed {PPG_INPUT_BAND_HZ[0]:g}-{PPG_INPUT_BAND_HZ[1]:g} Hz '
    'is the model input (was "raw")\n'
    f"PPG_INPUT_BAND_HZ = ({PPG_INPUT_BAND_HZ[0]!r}, {PPG_INPUT_BAND_HZ[1]!r})\n"
    f"PPG_INPUT_FILTER_ORDER = {PPG_INPUT_FILTER_ORDER}\n"
)
MODEL_INPUT_LINE = '    model_input = (ppg_raw if PPG_INPUT == "raw" else ppg_low).astype(np.float32)\n'
MODEL_INPUT_NEW = (
    '    if PPG_INPUT == "bandpass":\n'
    f"        # {NAME}: zero-phase Butterworth band-pass of the whole segment, then the windows are cut\n"
    "        model_input = sos_bandpass(\n"
    "            ppg_raw, fs, PPG_INPUT_BAND_HZ[0], PPG_INPUT_BAND_HZ[1], order=PPG_INPUT_FILTER_ORDER\n"
    "        ).astype(np.float32)\n"
    "    else:\n"
    '        model_input = (ppg_raw if PPG_INPUT == "raw" else ppg_low).astype(np.float32)\n'
)
ROOT_PATTERN = re.compile(r'^PROJECT_ROOT = Path\(r"[^"\n]*"\)$', re.MULTILINE)


def base_hash(text: str) -> str:
    return hashlib.sha256(ROOT_PATTERN.sub("PROJECT_ROOT = <root>", text).encode("utf-8")).hexdigest()


def find_script(name: str, folders: Sequence[Optional[Path]]) -> Optional[Path]:
    """The base script: the first found in the given folders, this file's folder, the current folder, the
    workspace, the earlier run folders and the parents of this file's folder."""
    here = Path(__file__).resolve().parent
    candidates = [Path(f) for f in folders if f]
    candidates += [here, Path.cwd(), WORKSPACE] + [WORKSPACE / p["run_folder"] for p in PARTS.values()] + [here.parent]
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


def patched_text(text: str, base_name: str, results_root: Path) -> str:
    """The base script with exactly the three changes; stops when a change does not match once."""
    text = text.replace("\r\n", "\n")
    for old, label in ((INPUT_LINE, "PPG_INPUT line"), (MODEL_INPUT_LINE, "model_input line")):
        count = text.count(old)
        if count != 1:
            raise SystemExit(f"{base_name}: the {label} was found {count} times (expected once); nothing is run")
    if len(ROOT_PATTERN.findall(text)) != 1:
        raise SystemExit(f"{base_name}: PROJECT_ROOT line not found once; nothing is run")
    text = text.replace(INPUT_LINE, INPUT_NEW).replace(MODEL_INPUT_LINE, MODEL_INPUT_NEW)
    text = ROOT_PATTERN.sub(f'PROJECT_ROOT = Path(r"{results_root}")', text)
    head = (f"# {NAME}: generated from {base_name}; the only changes: the model input is the PPG band-passed "
            f"{PPG_INPUT_BAND_HZ[0]:g}-{PPG_INPUT_BAND_HZ[1]:g} Hz (PPG_INPUT, model_input) and PROJECT_ROOT = "
            f"{results_root}\n")
    if text.startswith("# -*- coding"):
        first, rest = text.split("\n", 1)
        return f"{first}\n{head}{rest}"
    return head + text


def write_part(part: str, args: argparse.Namespace, code_dir: Path) -> Tuple[Path, Path, List[str]]:
    info = PARTS[part]
    base = find_script(info["script"], [args.scripts_dir])
    if base is None:
        raise SystemExit(f"{info['script']} not found (give its folder with --scripts-dir)")
    text = base.read_text(encoding="utf-8").replace("\r\n", "\n")
    digest = base_hash(text)
    note = "same as the checked version" if digest == info["sha256"] else (
        "DIFFERENT from the version this code was checked against (still used; the three changes matched)")
    results_root = Path(args.output_root) / info["results"]
    out = code_dir / f"{NAME}_from_{part}.py"
    out.write_text(patched_text(text, info["script"], results_root), encoding="utf-8", newline="\n")
    lines = [f"[{part}] base script: {base}", f"[{part}] SHA-256 (results-folder line left out): {digest} - {note}",
             f"[{part}] written: {out}", f"[{part}] results: {results_root}"]
    print("\n".join(lines))
    return out, results_root, lines


def part_command(part: str, script: Path, args: argparse.Namespace, others: List[str],
                 roots: Dict[str, Path]) -> List[str]:
    datasets = args.datasets_261007 if part == "261007" else args.datasets_261013
    command = [sys.executable, str(script), "--datasets", *datasets]
    if part == "261013" and "--reference-root" not in others and "--reference-root" not in (args.args_261013 or ""):
        # 261013's paired comparison: against the 261007 part of this run (skipped by 261013 when not found)
        command += ["--reference-root", str(roots.get("261007", Path(args.output_root) / PARTS["261007"]["results"]))]
    command += others + shlex.split(args.args_261007 if part == "261007" else args.args_261013 or "", posix=True)
    return command


def parse_args(argv: Optional[Sequence[str]] = None) -> Tuple[argparse.Namespace, List[str]]:
    parser = argparse.ArgumentParser(description="RRWaveNet 261007 / 261013 with the PPG input band-passed 0.1-8.0 Hz; "
                                                 "other options go to both scripts unchanged")
    parser.add_argument("--parts", nargs="+", choices=list(PARTS), default=list(PARTS))
    parser.add_argument("--scripts-dir", type=Path, default=None, help="folder of the two base scripts")
    parser.add_argument("--output-root", type=Path, default=OUTPUT_ROOT)
    parser.add_argument("--datasets-261007", nargs="+", default=list(PARTS["261007"]["datasets"]),
                        choices=["capnobase", "bidmc", "steam2"])
    parser.add_argument("--datasets-261013", nargs="+", default=list(PARTS["261013"]["datasets"]),
                        choices=["capnobase", "bidmc", "steam2"])
    parser.add_argument("--args-261007", default="", help='options only for the 261007 part, e.g. "--variants full"')
    parser.add_argument("--args-261013", default="", help='options only for the 261013 part, e.g. "--weight-decays 0.1"')
    parser.add_argument("--write-only", action="store_true", help="write the two scripts, do not run them")
    # --args-261007 "--variants full": argparse would read a value starting with "--" as an option; join it
    tokens, joined = list(sys.argv[1:] if argv is None else argv), []
    while tokens:
        token = tokens.pop(0)
        if token in ("--args-261007", "--args-261013") and tokens:
            token = f"{token}={tokens.pop(0)}"
        joined.append(token)
    return parser.parse_known_args(joined)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args, others = parse_args(argv)
    code_dir = Path(args.output_root) / "Code"
    code_dir.mkdir(parents=True, exist_ok=True)
    report = [f"{NAME} ({datetime.now():%Y-%m-%d %H:%M:%S})",
              f"Model input: PPG band-passed {PPG_INPUT_BAND_HZ[0]:g}-{PPG_INPUT_BAND_HZ[1]:g} Hz "
              f"(sos_bandpass, order {PPG_INPUT_FILTER_ORDER}, zero phase) instead of the raw PPG",
              "Changes in each script: 1. PPG_INPUT line  2. model_input line  3. PROJECT_ROOT", ""]
    scripts, roots = {}, {}
    for part in args.parts:
        scripts[part], roots[part], lines = write_part(part, args, code_dir)
        report += lines + [""]
    (code_dir / "changes.txt").write_text("\n".join(report), encoding="utf-8")
    if args.write_only:
        print(f"\nWritten only (--write-only): {code_dir}")
        return 0
    status = 0
    for part in args.parts:
        command = part_command(part, scripts[part], args, others, roots)
        print(f"\n{'=' * 78}\n[{part}] {' '.join(shlex.quote(c) for c in command)}\n{'=' * 78}", flush=True)
        result = subprocess.run(command)
        print(f"[{part}] finished with exit code {result.returncode}", flush=True)
        status = status or result.returncode
    return status


if __name__ == "__main__":
    sys.exit(main())
