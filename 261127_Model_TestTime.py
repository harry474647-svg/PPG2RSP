# -*- coding: utf-8 -*-
r"""261127_Model_TestTime: the test (inference) time of the three models of the comparison, measured on one device
with the stored models of every LOSOCV fold - nothing is trained.

Models (stored, as in the comparison of 261108_Wavefom_Comparison)
  RRWaveNet          CapnoBase 261007 full (D:\PPG2RSP_RRWaveNet_Inspired\261007_RRWaveNet_NoEvent_v1), BIDMC and STEAM2
                     261013 with the weight decay selected in the fold (261013_RRWavNet_STEAM_BIDMC_v1);
                     Members\m{k}\Models\{subject}_best.pth; timed as one model (m0) and as the ensemble of 3 (m0-m2,
                     outputs averaged), as the final results
  Deep Corr-Encoder  261107_DeepCorrencoder_PaperVersion_v1 (the paper's model): {Dataset}\Models\model_sub{subject}.pth
  RespDiff           260519_RespDiff_AllDatasets_v3, Baseline (low-band PPG input): {Dataset}\{run tag}\Models\
                     {subject}_baseline_best.pt; CapnoBase first from 261108_RespDiff_CapnoBase_Baseline_v3 (as the
                     waveform comparison); run tag preset_balanced_e90_ss80_ve2_tr2_va2_te1_mtr240_mva160 (80 sampling
                     steps)

Test windows of the held-out subject, as each model is tested in its own run
  RRWaveNet          10 s windows without overlap (640 samples, 64 Hz), read by the version's own script (its
                     preprocessing; SubjectWindowDataset "test": every kept window, min-max normalized); batch 48
  Deep Corr-Encoder  9.6 s windows shifted by 1 s (288 samples, 30 Hz), every window min-max scaled to [-1, 1], of the
                     30 Hz PPG stored with the run (Predictions\{subject}_outputs.npz, recording{k}_ppg); batch 512
  RespDiff           32 s windows shifted by 16 s (fs 125 Hz CapnoBase / BIDMC, 200 Hz STEAM2) that v3 kept, from v3's
                     preprocessing cache ({Dataset}\Processed_Subject_Cache), z-scored per window as in v3; DDPM
                     sampling with the run's 80 steps; batch 32 (v3's batch size); the count is checked against v3's
                     stored predictions (Windows_Match_Stored)
  The denoiser, sampler and model settings of RespDiff v3 are copied unchanged below (Baseline: base_dim 32, rnn_hidden
  128, 1000 diffusion steps).

What is timed (Test_Seconds of a fold): the model's inference over every test window of the held-out subject - the
inputs moved to the device, the forward pass (RespDiff: the whole sampling loop), the outputs moved back to the CPU -
in the batches of the model's run. Not timed: reading and preprocessing the data, loading the model, scoring.
  The device is synchronised before and after; every fold is run --repeats times (default 3) and the median is used;
  before the first fold of every model and dataset one untimed warm-up run (--warmup).
  Ms_per_Window = Test_Seconds / windows x 1000; Seconds_per_Recording_Minute = the time to cover one minute of a
  recording with the model's test windows = Ms_per_Window / 1000 x 60 / window shift (RRWaveNet 6 windows per minute,
  Deep Corr-Encoder 60, RespDiff 3.75).

Subjects: every fold with a stored model; the summary leaves out the MAE issue subjects of the final results (CapnoBase
0115, BIDMC 27, STEAM2 12_YJJ, 22_LMS, 26_CKK; --all-subjects keeps them).

Outputs (D:\PPG2RSP_RRWaveNet_Inspired\261127_Model_TestTime)
  261127_Model_TestTime.xlsx
    Summary   per dataset (and all datasets) and model: folds, test time per fold, per window and per recording minute
              (mean +- SD, median), parameters, device
    Folds     every fold and model: windows, repeats, test time (median, mean, min of the repeats), per window, per
              recording minute
    Skipped   folds without a stored model, test windows or cache
    Settings  device, PyTorch, repeats, batches, folders
  Folds.csv, Summary.csv

Run: python 261127_Model_TestTime.py     (Spyder: runfile(...), no options needed; on the GPU of the runs)
Options: --datasets, --models rrwavenet correncoder respdiff, --subjects Dataset:Subject ..., --repeats, --warmup,
  --all-subjects, --cpu, --gpu, --root-261007, --root-261013, --script-261007, --script-261013, --correncoder-root,
  --correncoder-script, --respdiff-root, --respdiff-capnobase-root, --respdiff-run-tag, --respdiff-sample-steps,
  --respdiff-batch, --rrwavenet-batch, --correncoder-batch, --capnobase-path, --bidmc-path, --steam2-ppg-dir,
  --steam2-rsp-dir, --output-root
"""

import argparse
import importlib.util
import json
import platform
import re
import sys
import time
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

# ---------------------------------------------------------------------------
# Settings
# ---------------------------------------------------------------------------

NAME = "261127_Model_TestTime"
HERE = Path(__file__).resolve().parent if "__file__" in globals() else Path.cwd()
WORKSPACE = Path(r"D:\PPG2RSP_RRWaveNet_Inspired")
OUTPUT_ROOT = WORKSPACE / NAME
DATASETS = ("STEAM2", "BIDMC", "CapnoBase")
ALL = "All datasets"
KEYS = {"CapnoBase": "capnobase", "BIDMC": "bidmc", "STEAM2": "steam2"}
ISSUE_SUBJECTS = {"STEAM2": ("22_LMS", "26_CKK", "12_YJJ"), "BIDMC": ("27",), "CapnoBase": ("0115",)}
MODELS = ("rrwavenet", "correncoder", "respdiff")
LABELS = {"rrwavenet_single": "RRWaveNet (single model, m0)", "rrwavenet_ensemble": "RRWaveNet (ensemble of 3)",
          "correncoder": "Deep Corr-Encoder", "respdiff": "RespDiff"}
LABEL_ORDER = tuple(LABELS.values())

# RRWaveNet (final results)
ROOT_261007 = WORKSPACE / "261007_RRWaveNet_NoEvent_v1"
ROOT_261013 = WORKSPACE / "261013_RRWavNet_STEAM_BIDMC_v1"
SCRIPTS = {"261007": "261007_RRWaveNet_Ensemble_NoEvent.py", "261013": "261013_RRWaveNet_Adult_SubjectVal_WD.py"}
SOURCE = {"CapnoBase": "261007", "BIDMC": "261013", "STEAM2": "261013"}
VARIANT_261007 = "full"
WINDOW = "w10s_nonoverlap"
MEMBERS = (0, 1, 2)
RRWAVENET_BATCH = 48  # BATCH_SIZE of 261007 / 261013
RRWAVENET_WINDOW_SEC, RRWAVENET_SHIFT_SEC = 10.0, 10.0

# Deep Corr-Encoder (261107 paper version)
CORRENCODER_ROOT = WORKSPACE / "261107_DeepCorrencoder_PaperVersion_v1"
CORRENCODER_SCRIPT = "261107_DeepCorrencoder_PaperVersion_v1.py"

# RespDiff (260519_RespDiff_AllDatasets_v3, Baseline)
RESPDIFF_ROOT = Path(r"D:\STEAM\Multimodal_prediction\PPG-RSP\Results\Waveform\Hilbert_OpenDataset"
                     r"\RespDiff_AllDatasets_v3")
RESPDIFF_CAPNOBASE_ROOT = Path(r"D:\STEAM\Multimodal_prediction\PPG-RSP\Results\Waveform\Hilbert_OpenDataset"
                               r"\RespDiff_CapnoBase_Baseline_261108")
RESPDIFF_RUN_TAG = "preset_balanced_e90_ss80_ve2_tr2_va2_te1_mtr240_mva160"
RESPDIFF_FOLDERS = {"CapnoBase": "CapnoBase", "BIDMC": "BIDMC", "STEAM2": "STEAM"}
RESPDIFF_KEYS = {"CapnoBase": "capnobase", "BIDMC": "bidmc", "STEAM2": "steam"}
RESPDIFF_MODE = "baseline"
RESPDIFF_WINDOW_SEC, RESPDIFF_STRIDE_SEC = 32.0, 16.0  # v3's WINDOW_SEC / STRIDE_SEC
RESPDIFF_SAMPLE_STEPS = 80  # the run tag's ss80 (speed preset balanced)
RESPDIFF_BATCH = 32  # v3's BATCH_SIZE (test loader)
DIFF_TIMESTEPS = 1000

_MODULES: Dict[str, object] = {}


# ===========================================================================
# RespDiff v3 (260519_RespDiff_AllDatasets_v3): Baseline denoiser and sampler, copied unchanged
# ===========================================================================

RESPDIFF_BASELINE = {"input_features": ["low"], "base_dim": 32, "rnn_hidden": 128, "dropout": 0.05,
                     "feature_dropout": 0.0}  # MODE_CONFIGS["baseline"] of v3 (model part)


class CNNEncoder(nn.Module):
    def __init__(self, in_channels, base_dim=32, dropout=0.0):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(in_channels, base_dim, kernel_size=15, padding=7),
            nn.GroupNorm(1, base_dim),
            nn.Mish(),
            nn.Dropout(dropout),
            nn.Conv1d(base_dim, base_dim, kernel_size=15, padding=7),
            nn.GroupNorm(1, base_dim),
            nn.Mish(),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)


class DenoiserV6(nn.Module):
    def __init__(self, ppg_channels=1, base_dim=32, rnn_hidden=128, dropout=0.0, feature_dropout=0.0):
        super().__init__()
        self.feature_dropout = feature_dropout
        self.ppg_enc = CNNEncoder(ppg_channels, base_dim, dropout=dropout)
        self.rsp_enc = CNNEncoder(1, base_dim, dropout=dropout)
        self.time_mlp = nn.Sequential(
            nn.Linear(1, 64),
            nn.Mish(),
            nn.Dropout(dropout),
            nn.Linear(64, 64),
        )
        self.lstm = nn.LSTM(
            base_dim * 2 + 64,
            rnn_hidden,
            num_layers=2,
            batch_first=True,
            bidirectional=True,
            dropout=dropout,
        )
        self.out_head = nn.Sequential(
            nn.Linear(rnn_hidden * 2, rnn_hidden),
            nn.Mish(),
            nn.Dropout(dropout),
            nn.Linear(rnn_hidden, 1),
        )

    def _drop_aux_features(self, x):
        if not self.training or self.feature_dropout <= 0.0 or x.shape[1] <= 1:
            return x
        keep_mask = (
            torch.rand((x.shape[0], x.shape[1] - 1, 1), device=x.device)
            > self.feature_dropout
        ).float()
        x = x.clone()
        x[:, 1:, :] = x[:, 1:, :] * keep_mask
        return x

    def forward(self, y_t, t, x_ppg):
        x_ppg = self._drop_aux_features(x_ppg)
        f_ppg = self.ppg_enc(x_ppg)
        f_rsp = self.rsp_enc(y_t)
        t_emb = self.time_mlp(t.float().view(-1, 1) / 1000.0)
        t_emb = t_emb.unsqueeze(-1).repeat(1, 1, y_t.shape[2])
        feat = torch.cat([f_ppg, f_rsp, t_emb], dim=1).permute(0, 2, 1)
        rnn_out, _ = self.lstm(feat)
        return self.out_head(rnn_out).permute(0, 2, 1)


class RespDiffHilbert(nn.Module):
    """v3's RespDiffHilbert with its buffers and sampler (the training losses are not needed here)."""

    def __init__(self, denoiser, timesteps=1000):
        super().__init__()
        self.denoiser = denoiser
        self.timesteps = timesteps

        beta = torch.linspace(1e-4, 0.02, timesteps)
        alpha = 1.0 - beta
        alpha_bar = torch.cumprod(alpha, dim=0)

        self.register_buffer("beta", beta)
        self.register_buffer("alpha", alpha)
        self.register_buffer("alpha_bar", alpha_bar)
        self.register_buffer("sqrt_alpha_bar", torch.sqrt(alpha_bar))
        self.register_buffer("sqrt_one_minus_alpha_bar", torch.sqrt(1.0 - alpha_bar))

    @torch.no_grad()
    def sample(self, x_ppg, sample_steps=200):
        y = torch.randn((x_ppg.shape[0], 1, x_ppg.shape[2]), device=x_ppg.device)
        step_ids = torch.linspace(
            self.timesteps - 1,
            0,
            steps=min(sample_steps, self.timesteps),
            device=x_ppg.device,
        ).long()
        step_ids = torch.unique_consecutive(step_ids)

        for idx, step in enumerate(step_ids):
            i = int(step.item())
            t = torch.full((x_ppg.shape[0],), i, device=x_ppg.device, dtype=torch.long)
            pred_noise = self.denoiser(y, t, x_ppg)

            ab_t = self.alpha_bar[i].view(1, 1, 1)
            x0_pred = (y - torch.sqrt(1.0 - ab_t) * pred_noise) / (torch.sqrt(ab_t) + 1e-8)
            x0_pred = torch.clamp(x0_pred, -5.0, 5.0)

            if idx == len(step_ids) - 1:
                y = x0_pred
            else:
                prev_i = int(step_ids[idx + 1].item())
                ab_prev = self.alpha_bar[prev_i].view(1, 1, 1)
                y = torch.sqrt(ab_prev) * x0_pred + torch.sqrt(1.0 - ab_prev) * pred_noise

        return y


def build_respdiff_baseline(device: torch.device) -> RespDiffHilbert:
    cfg = RESPDIFF_BASELINE
    denoiser = DenoiserV6(ppg_channels=len(cfg["input_features"]), base_dim=cfg["base_dim"],
                          rnn_hidden=cfg["rnn_hidden"], dropout=cfg["dropout"],
                          feature_dropout=cfg["feature_dropout"]).to(device)
    return RespDiffHilbert(denoiser, timesteps=DIFF_TIMESTEPS).to(device)


def instance_normalize(data):
    """v3's instance_normalize (z-score per window)."""
    data = np.asarray(data, dtype=np.float32)
    mean_val = np.mean(data, axis=-1, keepdims=True)
    std_val = np.std(data, axis=-1, keepdims=True)
    std_val[std_val < 1e-6] = 1.0
    return ((data - mean_val) / std_val).astype(np.float32)


def safe_tag(text) -> str:
    """v3's safe_tag (file names)."""
    return "".join(c if c.isalnum() or c in "-_." else "_" for c in str(text))


# ===========================================================================
# Helpers
# ===========================================================================

def same_subject(a: object, b: object) -> bool:
    a, b = str(a).strip(), str(b).strip()
    return a == b or (a.isdigit() and b.isdigit() and int(a) == int(b))


def is_issue(dataset: str, subject: object) -> bool:
    return any(same_subject(subject, s) for s in ISSUE_SUBJECTS[dataset])


def parse_subjects(items: Optional[Sequence[str]]) -> Dict[str, List[str]]:
    chosen: Dict[str, List[str]] = {}
    for item in items or []:
        dataset, _, subject = str(item).partition(":")
        if dataset not in DATASETS or not subject:
            raise SystemExit(f"--subjects {item}: write Dataset:Subject with Dataset in {', '.join(DATASETS)}")
        chosen.setdefault(dataset, []).append(subject)
    return chosen


def wanted(chosen: Dict[str, List[str]], dataset: str, subject: str) -> bool:
    return dataset not in chosen or any(same_subject(subject, s) for s in chosen[dataset])


def read_json(path: Path) -> Optional[dict]:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


def load_script(path: Path):
    key = str(path)
    if key not in _MODULES:
        name = "testtime_src_" + re.sub(r"\W", "_", path.stem)
        spec = importlib.util.spec_from_file_location(name, key)
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module
        spec.loader.exec_module(module)
        _MODULES[key] = module
    return _MODULES[key]


def find_script(name: str, given: Optional[Path], roots: Sequence[Path]) -> Path:
    """The script of a run: given, written in a run manifest, or found nearby."""
    candidates: List[Path] = [Path(given)] if given else []
    for root in roots:
        for manifest in sorted(Path(root).glob("run_manifest*.json")):
            script = (read_json(manifest) or {}).get("script") if manifest.is_file() else None
            if script and Path(script).name == name:
                candidates.append(Path(script))
    folders = [HERE, HERE.parent, Path.cwd(), WORKSPACE] + [Path(r) for r in roots]
    if WORKSPACE.is_dir():
        folders += sorted(p for p in WORKSPACE.iterdir() if p.is_dir())
    candidates += [f / name for f in folders]
    for path in candidates:
        if path.is_file():
            return path.resolve()
    raise FileNotFoundError(f"{name} not found (run manifests of {', '.join(map(str, roots))}, {HERE}, "
                            f"{WORKSPACE}); give it with the --script option")


def count_parameters(model: nn.Module) -> int:
    return int(sum(p.numel() for p in model.parameters()))


def sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def timed(run: Callable[[], None], device: torch.device, repeats: int) -> List[float]:
    """Seconds of every repeat of run(), the device synchronised before and after."""
    seconds = []
    for _ in range(max(1, int(repeats))):
        sync(device)
        started = time.perf_counter()
        run()
        sync(device)
        seconds.append(time.perf_counter() - started)
    return seconds


def batched(inputs: torch.Tensor, batch: int):
    for start in range(0, len(inputs), int(batch)):
        yield inputs[start:start + int(batch)]


class Recorder:
    """Folds and skipped folds; the warm-up done once per model and dataset."""

    def __init__(self, args, device: torch.device):
        self.args, self.device = args, device
        self.folds: List[dict] = []
        self.skipped: List[dict] = []
        self.warmed: set = set()

    def skip(self, dataset: str, model: str, subject: str, reason: str) -> None:
        self.skipped.append({"Dataset": dataset, "Model": model, "Subject": subject, "Reason": reason})
        print(f"  {model} {dataset} {subject}: skipped ({reason})", flush=True)

    def measure(self, dataset: str, label: str, subject: str, run: Callable[[], None], windows: int, window_sec: float,
                shift_sec: float, parameters: int, detail: str, extra: Optional[dict] = None) -> None:
        if (label, dataset) not in self.warmed:
            for _ in range(max(0, int(self.args.warmup))):
                run()
            sync(self.device)
            self.warmed.add((label, dataset))
        seconds = timed(run, self.device, self.args.repeats)
        median = float(np.median(seconds))
        per_window = median / windows * 1000.0 if windows else float("nan")
        row = {"Dataset": dataset, "Subject": subject, "Issue_Subject": is_issue(dataset, subject), "Model": label,
               "Model_Detail": detail, "Parameters": parameters, "Windows": int(windows), "Window_Sec": window_sec,
               "Window_Shift_Sec": shift_sec, "Input_Minutes": windows * window_sec / 60.0,
               "Repeats": len(seconds), "Test_Seconds": median, "Test_Seconds_Mean": float(np.mean(seconds)),
               "Test_Seconds_Min": float(np.min(seconds)), "Ms_per_Window": per_window,
               "Seconds_per_Recording_Minute": per_window / 1000.0 * 60.0 / shift_sec,
               "Device": device_name(self.device), **(extra or {})}
        self.folds.append(row)
        print(f"  {label:<30} {dataset} {subject}: {windows} windows, {median:.3f} s "
              f"({per_window:.2f} ms per window, {row['Seconds_per_Recording_Minute']:.3f} s per recording minute)",
              flush=True)


def device_name(device: torch.device) -> str:
    if device.type == "cuda":
        return f"cuda:{device.index if device.index is not None else 0} {torch.cuda.get_device_name(device)}"
    return f"cpu ({platform.processor() or platform.machine()}, {torch.get_num_threads()} threads)"


# ===========================================================================
# RRWaveNet
# ===========================================================================

def rrwavenet_root(args, mod, spec, dataset: str, subject: str) -> Tuple[Optional[Path], str, str]:
    """Results folder of the final model of the subject's fold, its variant, and why it is missing."""
    key = KEYS[dataset]
    if SOURCE[dataset] == "261007":
        root, variant = Path(args.root_261007), VARIANT_261007
    else:
        root = Path(args.root_261013)
        found = mod.stored_selection(root, list(mod.VARIANTS.values()), spec, key, subject)
        if found is not None:
            variant = found[1].name
        else:
            variant = (read_json(mod.selection_path(root, spec, key, subject)) or {}).get("Selected_Variant")
        if not variant:
            return None, "", f"no weight decay selection ({mod.selection_path(root, spec, key, subject)})"
    return mod.condition_root(root, variant, spec, key), variant, ""


def time_rrwavenet(args, dataset: str, chosen: Dict[str, List[str]], rec: Recorder) -> None:
    source = SOURCE[dataset]
    roots = [args.root_261007] if source == "261007" else [args.root_261013]
    try:
        script = find_script(SCRIPTS[source], args.script_261007 if source == "261007" else args.script_261013, roots)
    except FileNotFoundError as exc:
        rec.skip(dataset, "RRWaveNet", "-", str(exc))
        return
    mod = load_script(script)
    for key, field, value in (("capnobase", "data_path", args.capnobase_path), ("bidmc", "data_path", args.bidmc_path),
                              ("steam2", "ppg_dir", args.steam2_ppg_dir), ("steam2", "rsp_dir", args.steam2_rsp_dir)):
        if value:
            mod.DATASET_CONFIGS[key][field] = str(value)
    spec = next(s for s in mod.WINDOW_SPECS if s.name == WINDOW)
    print(f"\nRRWaveNet {dataset}: test windows and models of {script.name} ({source})", flush=True)
    data = mod.load_dataset(KEYS[dataset], spec)
    build = (mod.VARIANTS[VARIANT_261007] if source == "261007" else next(iter(mod.VARIANTS.values()))).options.build
    device = rec.device
    models = [build().to(device).eval() for _ in args.members]
    parameters = count_parameters(models[0])
    for subject in sorted(data):
        if not wanted(chosen, dataset, subject):
            continue
        root, variant, reason = rrwavenet_root(args, mod, spec, dataset, subject)
        if root is None:
            rec.skip(dataset, "RRWaveNet", subject, reason)
            continue
        checkpoints = [root / "Members" / f"m{m}" / "Models" / f"{mod.safe_tag(subject)}_best.pth" for m in args.members]
        missing = [str(p) for p in checkpoints if not p.exists()]
        if missing:
            rec.skip(dataset, "RRWaveNet", subject, f"no stored model ({'; '.join(missing)})")
            continue
        for model, checkpoint in zip(models, checkpoints):
            model.load_state_dict(torch.load(checkpoint, map_location=device))
            model.eval()
        dataset_test = mod.SubjectWindowDataset(data, [subject], "test", False)
        inputs = torch.stack([dataset_test[i][0] for i in range(len(dataset_test))]).float()  # (windows, 1, 640)

        def run_members(members: Sequence[nn.Module]) -> Callable[[], None]:
            def run() -> None:
                with torch.no_grad():
                    for x in batched(inputs, args.rrwavenet_batch):
                        x = x.to(device)
                        out = torch.stack([m(x) for m in members]).mean(dim=0)
                        out.cpu()
            return run

        detail = f"{source} {variant}"
        rec.measure(dataset, LABELS["rrwavenet_single"], subject, run_members(models[:1]), len(inputs),
                    RRWAVENET_WINDOW_SEC, RRWAVENET_SHIFT_SEC, parameters, f"{detail}, m{args.members[0]}")
        rec.measure(dataset, LABELS["rrwavenet_ensemble"], subject, run_members(models), len(inputs),
                    RRWAVENET_WINDOW_SEC, RRWAVENET_SHIFT_SEC, parameters * len(models),
                    f"{detail}, {' '.join(f'm{m}' for m in args.members)} (outputs averaged)")
        del inputs, dataset_test
    del data, models
    mod.release_memory(device)


# ===========================================================================
# Deep Corr-Encoder (261107 paper version)
# ===========================================================================

def time_correncoder(args, dataset: str, chosen: Dict[str, List[str]], rec: Recorder) -> None:
    root = Path(args.correncoder_root)
    try:
        script = find_script(CORRENCODER_SCRIPT, args.correncoder_script, [root])
    except FileNotFoundError as exc:
        rec.skip(dataset, "Deep Corr-Encoder", "-", str(exc))
        return
    ce = load_script(script)
    label = ce.DATASET_CONFIGS[KEYS[dataset]]["label"]
    files = sorted((root / label / "Predictions").glob("*_outputs.npz"))
    print(f"\nDeep Corr-Encoder {dataset}: {len(files)} stored folds under {root / label} ({script.name})", flush=True)
    if not files:
        rec.skip(dataset, "Deep Corr-Encoder", "-", f"no stored predictions in {root / label / 'Predictions'}")
        return
    device = rec.device
    model = ce.Correncoder_model().to(device).eval()
    parameters = count_parameters(model)
    hop_sec = ce.TEST_HOP / float(ce.CORR_FS)
    window_sec = ce.SEGMENT_LEN / float(ce.CORR_FS)
    batch = int(args.correncoder_batch or ce.PREDICT_BATCH)
    for path in files:
        subject = path.name[: -len("_outputs.npz")]
        if not wanted(chosen, dataset, subject):
            continue
        checkpoint = root / label / "Models" / f"model_sub{subject}.pth"
        if not checkpoint.exists():
            rec.skip(dataset, "Deep Corr-Encoder", subject, f"no stored model ({checkpoint})")
            continue
        recordings = []
        with np.load(path, allow_pickle=False) as stored:
            for name in sorted(k for k in stored.files if re.fullmatch(r"recording\d+_ppg", k)):
                ppg = stored[name].astype(np.float64)
                starts = ce.test_starts(len(ppg))
                if starts:
                    windows = np.stack([ppg[s:s + ce.SEGMENT_LEN] for s in starts])
                    recordings.append(torch.from_numpy(ce.scale(windows, ce.PPG_RANGE)[0][:, None, :]
                                                       .astype("float32")))
        if not recordings:
            rec.skip(dataset, "Deep Corr-Encoder", subject, f"no test window in {path.name}")
            continue
        model.load_state_dict(torch.load(checkpoint, map_location=device))
        model.eval()

        def run() -> None:
            with torch.no_grad():  # as predict() of 261107: every recording in batches of 512
                for windows in recordings:
                    for x in batched(windows, batch):
                        model(x.to(device))[:, 0, :].cpu()

        rec.measure(dataset, LABELS["correncoder"], subject, run, int(sum(len(r) for r in recordings)), window_sec,
                    hop_sec, parameters, f"261107 paper version, epoch 80 ({len(recordings)} recordings)")
    del model
    release(device)


# ===========================================================================
# RespDiff (260519_RespDiff_AllDatasets_v3, Baseline)
# ===========================================================================

def respdiff_run_dir(base: Path, run_tag: str) -> Optional[Path]:
    """{base}\\{run tag} when it has stored models, else the run folder with the most Baseline models."""
    if (base / run_tag / "Models").is_dir():
        return base / run_tag
    runs = sorted(p for p in base.glob("*") if (p / "Models").is_dir()) if base.is_dir() else []
    if not runs:
        return None
    best = max(runs, key=lambda p: len(list((p / "Models").glob(f"*_{RESPDIFF_MODE}_best.pt"))))
    print(f"  RespDiff: {base / run_tag} not found; using {best}")
    return best


def respdiff_models(args, dataset: str) -> Dict[str, Tuple[Path, Path]]:
    """Subject -> (Baseline model, run folder): RespDiff_AllDatasets_v3, then (CapnoBase) the 261108 run, which wins
    for a subject in both (as the waveform comparison)."""
    roots = [Path(args.respdiff_root)]
    if dataset == "CapnoBase":
        roots.append(Path(args.respdiff_capnobase_root))
    models: Dict[str, Tuple[Path, Path]] = {}
    for root in roots:
        run_dir = respdiff_run_dir(root / RESPDIFF_FOLDERS[dataset], args.respdiff_run_tag)
        if run_dir is None:
            continue
        for path in sorted((run_dir / "Models").glob(f"*_{RESPDIFF_MODE}_best.pt")):
            models[path.name[: -len(f"_{RESPDIFF_MODE}_best.pt")]] = (path, run_dir)
    return models


def respdiff_cache_files(args, dataset: str, subject: str) -> List[Path]:
    """v3's preprocessing cache of a subject (STEAM2: one file per recording, joined as v3 joins them)."""
    tag = f"w{RESPDIFF_WINDOW_SEC:g}_s{RESPDIFF_STRIDE_SEC:g}"
    key = RESPDIFF_KEYS[dataset]
    caches = [Path(r) / RESPDIFF_FOLDERS[dataset] / "Processed_Subject_Cache"
              for r in ([args.respdiff_capnobase_root] if dataset == "CapnoBase" else []) + [args.respdiff_root]]
    for cache in caches:
        if dataset == "STEAM2":
            files = sorted(cache.glob(f"{key}_{safe_tag(subject)}_*_{safe_tag(tag)}_processed.npz"))
        else:
            single = cache / f"{key}_{safe_tag(subject)}_{safe_tag(tag)}_processed.npz"
            files = [single] if single.exists() else []
        if files:
            return files
    return []


def time_respdiff(args, dataset: str, chosen: Dict[str, List[str]], rec: Recorder) -> None:
    stored = respdiff_models(args, dataset)
    print(f"\nRespDiff v3 {dataset}: {len(stored)} stored Baseline models ({args.respdiff_sample_steps} sampling "
          f"steps, batch {args.respdiff_batch})", flush=True)
    if not stored:
        rec.skip(dataset, "RespDiff", "-", f"no stored Baseline model under {args.respdiff_root}")
        return
    device = rec.device
    model = build_respdiff_baseline(device).eval()
    parameters = count_parameters(model)
    for subject, (checkpoint, run_dir) in sorted(stored.items()):
        if not wanted(chosen, dataset, subject):
            continue
        files = respdiff_cache_files(args, dataset, subject)
        if not files:
            rec.skip(dataset, "RespDiff", subject, "no v3 preprocessing cache (Processed_Subject_Cache)")
            continue
        packs = []
        fs = None
        for path in files:
            with np.load(path, allow_pickle=False) as cached:
                packs.append(cached["low"].astype(np.float32))
                fs = int(cached["fs"])
        low = np.concatenate(packs, axis=0)
        inputs = torch.from_numpy(instance_normalize(low[:, None, :]))  # build_xy of v3: (windows, 1, samples)
        stored_windows = None
        prediction = run_dir / "Predictions" / f"{safe_tag(subject)}_predictions.npz"
        if prediction.exists():
            with np.load(prediction, allow_pickle=False) as p:
                stored_windows = int(len(p["true"])) if "true" in p.files else None
        try:
            state = torch.load(checkpoint, map_location=device, weights_only=True)
        except TypeError:
            state = torch.load(checkpoint, map_location=device)
        model.load_state_dict(state)
        model.eval()
        torch.manual_seed(0)

        def run() -> None:
            with torch.no_grad():  # sample_loader of v3
                for x in batched(inputs, args.respdiff_batch):
                    pred = model.sample(x.to(device), sample_steps=args.respdiff_sample_steps)
                    pred.squeeze(1).float().cpu()

        rec.measure(dataset, LABELS["respdiff"], subject, run, len(inputs), RESPDIFF_WINDOW_SEC, RESPDIFF_STRIDE_SEC,
                    parameters, f"v3 Baseline, {run_dir.parent.parent.name}\\{run_dir.name}, "
                                f"{args.respdiff_sample_steps} steps, fs {fs} Hz",
                    {"Windows_Stored_Predictions": "" if stored_windows is None else stored_windows,
                     "Windows_Match_Stored": ("no stored predictions" if stored_windows is None else
                                              "yes" if stored_windows == len(inputs) else "no")})
        del inputs, low, packs
    del model
    release(device)


def release(device: torch.device) -> None:
    import gc
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()


# ===========================================================================
# Summary
# ===========================================================================

def mean_sd(values: pd.Series, digits: int) -> str:
    values = values.astype(float).dropna()
    if values.empty:
        return ""
    if len(values) == 1:
        return f"{values.iloc[0]:.{digits}f}"
    return f"{values.mean():.{digits}f} ± {values.std(ddof=1):.{digits}f} ({values.median():.{digits}f})"


def summary_table(folds: pd.DataFrame, datasets: Sequence[str]) -> pd.DataFrame:
    rows = []
    groups = [(d, folds[folds.Dataset == d]) for d in datasets] + [(ALL, folds)]
    for group, frame in groups:
        for label in LABEL_ORDER:
            part = frame[frame.Model == label]
            if part.empty:
                continue
            rows.append({
                "Dataset": group, "Model": label, "Folds": int(len(part)), "Windows": int(part.Windows.sum()),
                "Parameters": int(part.Parameters.iloc[0]),
                "Test_Seconds_per_Fold_Mean": float(part.Test_Seconds.mean()),
                "Test_Seconds_per_Fold_SD": float(part.Test_Seconds.std(ddof=1)) if len(part) > 1 else float("nan"),
                "Test_Seconds_per_Fold_Median": float(part.Test_Seconds.median()),
                "Test_Seconds_Total": float(part.Test_Seconds.sum()),
                "Ms_per_Window_Mean": float(part.Ms_per_Window.mean()),
                "Ms_per_Window_SD": float(part.Ms_per_Window.std(ddof=1)) if len(part) > 1 else float("nan"),
                "Seconds_per_Recording_Minute_Mean": float(part.Seconds_per_Recording_Minute.mean()),
                "Seconds_per_Recording_Minute_SD": (float(part.Seconds_per_Recording_Minute.std(ddof=1))
                                                    if len(part) > 1 else float("nan")),
                "Test_Seconds_per_Fold": mean_sd(part.Test_Seconds, 3),
                "Ms_per_Window": mean_sd(part.Ms_per_Window, 2),
                "Seconds_per_Recording_Minute": mean_sd(part.Seconds_per_Recording_Minute, 4),
                "Window_Sec": float(part.Window_Sec.iloc[0]), "Window_Shift_Sec": float(part.Window_Shift_Sec.iloc[0]),
                "Device": str(part.Device.iloc[0]),
            })
    return pd.DataFrame(rows)


# ===========================================================================
# Main
# ===========================================================================

def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Test (inference) time of RRWaveNet, the Deep Corr-Encoder (paper version) "
                                            "and RespDiff v3 with the stored models of every LOSOCV fold")
    p.add_argument("--datasets", nargs="+", choices=list(DATASETS), default=list(DATASETS))
    p.add_argument("--models", nargs="+", choices=list(MODELS), default=list(MODELS))
    p.add_argument("--subjects", nargs="+", default=None, help="Dataset:Subject (default: every fold)")
    p.add_argument("--repeats", type=int, default=3, help="timed runs of every fold; the median is used (default 3)")
    p.add_argument("--warmup", type=int, default=1, help="untimed runs before the first fold of every model and "
                                                         "dataset (default 1)")
    p.add_argument("--all-subjects", action="store_true", help="keep the MAE issue subjects in the summary")
    p.add_argument("--cpu", action="store_true", help="time on the CPU")
    p.add_argument("--gpu", type=int, default=0, help="GPU index (default 0, as the runs)")
    p.add_argument("--members", nargs="+", type=int, default=list(MEMBERS), help="RRWaveNet members (default 0 1 2)")
    p.add_argument("--root-261007", type=Path, default=ROOT_261007)
    p.add_argument("--root-261013", type=Path, default=ROOT_261013)
    p.add_argument("--script-261007", type=Path, default=None)
    p.add_argument("--script-261013", type=Path, default=None)
    p.add_argument("--correncoder-root", type=Path, default=CORRENCODER_ROOT)
    p.add_argument("--correncoder-script", type=Path, default=None)
    p.add_argument("--respdiff-root", type=Path, default=RESPDIFF_ROOT)
    p.add_argument("--respdiff-capnobase-root", type=Path, default=RESPDIFF_CAPNOBASE_ROOT)
    p.add_argument("--respdiff-run-tag", default=RESPDIFF_RUN_TAG)
    p.add_argument("--respdiff-sample-steps", type=int, default=RESPDIFF_SAMPLE_STEPS)
    p.add_argument("--respdiff-batch", type=int, default=RESPDIFF_BATCH)
    p.add_argument("--rrwavenet-batch", type=int, default=RRWAVENET_BATCH)
    p.add_argument("--correncoder-batch", type=int, default=None, help="default: PREDICT_BATCH of 261107 (512)")
    p.add_argument("--capnobase-path", type=Path, default=None)
    p.add_argument("--bidmc-path", type=Path, default=None)
    p.add_argument("--steam2-ppg-dir", type=Path, default=None)
    p.add_argument("--steam2-rsp-dir", type=Path, default=None)
    p.add_argument("--output-root", type=Path, default=OUTPUT_ROOT)
    return p.parse_args(list(sys.argv[1:] if argv is None else argv))


def write_excel(path: Path, sheets: Dict[str, pd.DataFrame]) -> Path:
    try:
        with pd.ExcelWriter(path) as writer:
            for name, frame in sheets.items():
                frame.to_excel(writer, sheet_name=name, index=False)
        return path
    except PermissionError:  # open in Excel: written next to it
        other = path.with_name(f"{path.stem}_{time.strftime('%Y%m%d_%H%M%S')}{path.suffix}")
        return write_excel(other, sheets)


def main(argv: Optional[Sequence[str]] = None) -> None:
    args = parse_args(argv)
    chosen = parse_subjects(args.subjects)
    datasets = [d for d in DATASETS if d in args.datasets and (not chosen or d in chosen)]
    out_root = Path(args.output_root)
    out_root.mkdir(parents=True, exist_ok=True)
    use_cuda = torch.cuda.is_available() and not args.cpu
    device = torch.device(f"cuda:{args.gpu}" if use_cuda else "cpu")
    if use_cuda:
        torch.cuda.set_device(device)
    print(f"{NAME}: device {device_name(device)}, {args.repeats} timed runs per fold (median), warm-up "
          f"{args.warmup}; results under {out_root}")
    rec = Recorder(args, device)
    steps = {"rrwavenet": time_rrwavenet, "correncoder": time_correncoder, "respdiff": time_respdiff}
    for dataset in datasets:
        for model in MODELS:
            if model in args.models:
                steps[model](args, dataset, chosen, rec)
    folds = pd.DataFrame(rec.folds)
    skipped = pd.DataFrame(rec.skipped, columns=["Dataset", "Model", "Subject", "Reason"])
    if folds.empty:
        write_excel(out_root / f"{NAME}.xlsx", {"Skipped": skipped})
        raise SystemExit("No fold was timed (see Skipped).")
    used = folds if args.all_subjects else folds[~folds.Issue_Subject.astype(bool)]
    summary = summary_table(used, [d for d in datasets if d in set(used.Dataset)])
    subject_set = "all subjects" if args.all_subjects else "MAE issue subjects left out"
    settings = pd.DataFrame([
        ("Device", device_name(device)), ("PyTorch", torch.__version__), ("Python", platform.python_version()),
        ("Timed runs per fold", f"{args.repeats} (median used)"), ("Warm-up", f"{args.warmup} untimed run(s) before "
                                                                              "the first fold of every model and dataset"),
        ("Summary subjects", subject_set),
        ("What is timed", "inference over every test window of the held-out subject: inputs to the device, forward "
                          "pass (RespDiff: the whole sampling loop), outputs to the CPU; not reading / preprocessing / "
                          "model loading / scoring"),
        ("RRWaveNet", f"CapnoBase 261007 {VARIANT_261007} ({args.root_261007}), BIDMC / STEAM2 261013 with the weight "
                      f"decay selected in the fold ({args.root_261013}); 10 s windows without overlap; batch "
                      f"{args.rrwavenet_batch}; single model m{args.members[0]} and ensemble of "
                      f"{' '.join(f'm{m}' for m in args.members)} (outputs averaged)"),
        ("Deep Corr-Encoder", f"261107 paper version ({args.correncoder_root}); 9.6 s windows shifted by 1 s (30 Hz) "
                              "of the stored 30 Hz PPG, each min-max scaled to [-1, 1]; batch "
                              f"{args.correncoder_batch or 512} per recording"),
        ("RespDiff", f"260519_RespDiff_AllDatasets_v3 Baseline ({args.respdiff_root}; CapnoBase first "
                     f"{args.respdiff_capnobase_root}), run {args.respdiff_run_tag}; 32 s windows shifted by 16 s from "
                     f"v3's preprocessing cache, z-scored; {args.respdiff_sample_steps} sampling steps; batch "
                     f"{args.respdiff_batch}"),
        ("Per recording minute", "Ms_per_Window / 1000 x 60 / window shift: RRWaveNet 6, Deep Corr-Encoder 60, "
                                 "RespDiff 3.75 windows per minute of recording"),
        ("MAE issue subjects", "; ".join(f"{d} {', '.join(ISSUE_SUBJECTS[d])}" for d in DATASETS)),
    ], columns=["Item", "Value"])
    path = write_excel(out_root / f"{NAME}.xlsx", {"Summary": summary, "Folds": folds, "Skipped": skipped,
                                                   "Settings": settings})
    summary.to_csv(out_root / "Summary.csv", index=False, encoding="utf-8-sig")
    folds.to_csv(out_root / "Folds.csv", index=False, encoding="utf-8-sig")
    print(f"\nTest time ({subject_set}; mean ± SD (median)):")
    for row in summary.itertuples(index=False):
        print(f"  {row.Dataset:<13} {row.Model:<30} {row.Folds:3d} folds: {row.Test_Seconds_per_Fold} s per fold, "
              f"{row.Ms_per_Window} ms per window, {row.Seconds_per_Recording_Minute} s per recording minute")
    print(f"\n{len(folds)} timed folds, {len(skipped)} skipped; table: {path}")


if __name__ == "__main__":
    main()
