r"""261028_Deep_correncoder: the deep corr-encoder of Davies & Mandic
("Rapid extraction of respiratory waveforms from photoplethysmography: A deep
corr-encoder approach", Biomed. Signal Process. Control 85:104992, 2023,
doi:10.1016/j.bspc.2023.104992; arXiv:2212.12578) on CapnoBase, BIDMC and
STEAM2, trained with the authors' published training code
(correncoder_capnobase_training.py, Harry J Davies, 2023), LOSOCV with one
model per fold, and scored with the metrics of this project (60 s
breathing-rate MAE and waveform PCC) on the same minutes as the models of
261007 (CapnoBase) and 261013 (BIDMC, STEAM2).

Model and training (the authors' code, unchanged)
-------------------------------------------------
Correncoder_model: Conv1d(1 -> 8, kernel 150, padding 20) + ReLU + dropout
0.5 -> Conv1d(8 -> 8, kernel 75, padding 20) + ReLU + dropout 0.5 ->
Conv1d(8 -> 8, kernel 50, padding 10) + sigmoid + dropout 0.5 ->
ConvTranspose1d(8 -> 8, kernel 50, padding 10) + sigmoid ->
ConvTranspose1d(8 -> 8, kernel 75, padding 20) + ReLU ->
ConvTranspose1d(8 -> 1, kernel 150, padding 20); input and output 288
samples. MSE loss, Adam with learning rate 1e-3, 80 epochs, batch 30; the
training segments are shuffled once (as sklearn.utils.shuffle) and the
batches taken in that order every epoch, the last incomplete batch left out;
no validation set, the model after the last epoch is used. Seed 55 (torch,
random, numpy), set at the start of every fold here (the authors' script
sets it once before all folds) so that every fold is reproducible on its own.
Leave one subject out: the model of a fold is trained on all other subjects
of the dataset and tested on the held-out subject. The test outputs are made
in evaluation mode (dropout off); the authors' script prints a test loss
every epoch in training mode, which is only logged here (eval mode) and never
used to choose a model.

Data (paper; the authors' preprocessing script is not published)
---------------------------------------------------------------
The same for the three datasets:
* PPG and respiratory reference of every recording resampled to 30 Hz
  (polyphase, anti-aliasing FIR with a Kaiser window, as MATLAB resample; the
  paper downsamples CapnoBase to 30 Hz and resamples BIDMC to 30 Hz), no
  filtering, missing samples interpolated linearly (counted);
* reference: CapnoBase the CO2 capnogram, BIDMC the impedance pneumogram,
  STEAM2 the respiratory belt (as recorded);
* training segments: 288 samples (9.6 s), non-overlapping, from the start of
  every recording piece (CapnoBase / BIDMC: 480 s = 50 segments per subject,
  as in the paper); all data are used, artifacts are not removed;
* test segments: a sliding window of 288 samples moved by 1 s (30 samples;
  CapnoBase 471 windows per subject, as in the paper, plus one window ending
  at the last sample so that every sample has an output); the outputs are
  averaged where windows overlap into one waveform per recording piece;
* every segment: PPG standardized (zero mean, unit SD), reference scaled to
  0-1 (min-max), as the authors' data (segments_all_standardised; the outputs
  are trained to an amplitude between 0 and 1).
The recordings are read with the readers of 261007 (CapnoBase) and 261013
(BIDMC, STEAM2): the PPG and reference of every recording piece as read
there (STEAM2: EEGLAB sessions split at boundary events, all sessions of a
person in one subject), before any processing of those scripts.
Differences from the paper: every dataset is trained and tested in its own
LOSOCV (the paper also tested BIDMC with a model trained on CapnoBase and then
retrained on BIDMC); STEAM2 is not in the paper and is processed as the
other two.

Evaluation (this project's metrics, the 261007 / 261013 code)
-------------------------------------------------------------
The 30 Hz output waveform of every recording piece is resampled to 64 Hz and
cut at the 10 s test windows of 261007 (CapnoBase) / 261013 (BIDMC, STEAM2),
then scored with their evaluation, unchanged: every complete 60 s minute
with a valid reference (the same minutes as those models); breathing rate =
largest FFT peak of the predicted waveform in the dataset's rate band (3-60
breaths/min CapnoBase, 6-30 BIDMC and STEAM2); ground truth = the number of
reference breaths in the minute (CapnoBase: the expert labels). MAE = mean
absolute rate error (breaths/min); waveform PCC = Pearson r per minute:
* Wave_PCC: against the target waveform of 261007 / 261013 (the same
  comparison as for their models: CapnoBase the respiratory-phase waveform
  of the expert labels aligned to the PPG, BIDMC / STEAM2 the respiratory
  band of the reference, BIDMC with the 261007 sign rule);
* Wave_PCC_Own_Reference: against the reference the corr-encoder is trained
  to reproduce (the 30 Hz 0-1 reference of its test windows, averaged as the
  outputs). The corr-encoder learns the capnogram as recorded (delayed
  against the PPG) and the raw impedance / belt (no sign rule), so this is
  its waveform agreement in its own terms.
Per dataset: FFT MAE +- SD, RMSE, bias, LoA, Pearson r, within 1 / 2
breaths/min, subject MAE +- SD, Davies & Mandic mAE and mMAE with IQR, PCC
mean +- SD (both).

Outputs (PROJECT_ROOT = D:\PPG2RSP_RRWaveNet_Inspired\261028_Deep_correncoder)
  {dataset}\Models\{subject}_model.pth             model after the last epoch
  {dataset}\Training_Logs\{subject}_loss.csv       loss of every epoch
  {dataset}\Predictions\{subject}_outputs.npz      30 Hz output, reference and
                                                   PPG of every recording piece
  {dataset}\Fold_Results\{subject}_summary.json / _minutes.csv
  Summary\   correncoder_overall.csv, correncoder_overall_mean_sd.csv,
             per_subject.csv, minutes.csv, data_check.csv, run_settings.json,
             261028_Deep_correncoder_results.xlsx

Run next to 261007_RRWaveNet_Ensemble_NoEvent.py and
261013_RRWaveNet_Adult_SubjectVal_WD.py (data paths, readers and evaluation
are taken from them; Spyder: just run the file):
  python 261028_Deep_correncoder.py
Resume after an interruption:    --resume (finished folds are kept)
Some datasets:                   --datasets capnobase bidmc
Summary of stored folds only:    --summary-only
Data paths:                      --capnobase-path, --bidmc-path,
                                 --steam2-ppg-dir, --steam2-rsp-dir
"""

import argparse
import importlib.util
import json
import math
import random
import sys
import time
from fractions import Fraction
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

sys.dont_write_bytecode = True

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from scipy.signal import resample_poly


# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

PROJECT_ROOT = Path(r"D:\PPG2RSP_RRWaveNet_Inspired\261028_Deep_correncoder")
DEFAULT_RESULTS_ROOT = PROJECT_ROOT
SCRIPT_DIR = Path(__file__).resolve().parent
BACKBONE_261007 = SCRIPT_DIR / "261007_RRWaveNet_Ensemble_NoEvent.py"
BACKBONE_261013 = SCRIPT_DIR / "261013_RRWaveNet_Adult_SubjectVal_WD.py"
# Readers and evaluation of every dataset: CapnoBase 261007, BIDMC and STEAM2
# 261013 (as the models compared with).
DATASET_SOURCES = {"capnobase": "261007", "bidmc": "261013", "steam2": "261013"}
DATASET_ORDER = ("capnobase", "bidmc", "steam2")
WINDOW = "w10s_nonoverlap"
CODE_NAME = "261028_Deep_correncoder"


# ---------------------------------------------------------------------------
# The authors' settings (correncoder_capnobase_training.py), unchanged
# ---------------------------------------------------------------------------

# define number of epochs and batch size
num_epochs = 80
batch_size = 30

# define number of kernels per layer
n_in, n_out = 1, 8
n_out2 = 8
n_out3 = 8
n_outputs = 1

# define kernel lengths, padding, dilation, stride, and dropout
kernel_size = 150
kernel_size2 = 75
kernel_size3 = 50
padding = 20
dilation = 1
stride = 1
dropout_val = 0.5
padding2 = 20
padding3 = 10
dilation2 = 1
dilation3 = 1
stride2 = 1
stride3 = 1

# seed and learning rate for Adam optimisation
seed_val = 55
learning_rate = 0.001


# define the model
class Correncoder_model(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer1 = nn.Sequential(
            nn.Conv1d(n_in, n_out, kernel_size=kernel_size, padding=padding),
            nn.ReLU(),
            nn.Dropout(dropout_val)
        )
        self.layer2 = nn.Sequential(
            nn.Conv1d(n_out, n_out2, kernel_size=kernel_size2, padding=padding2),
            nn.ReLU(),
            nn.Dropout(dropout_val)
        )
        self.layer3 = nn.Sequential(
            nn.Conv1d(n_out2, n_out3, kernel_size=kernel_size3, padding=padding3),
            nn.Sigmoid(),
            nn.Dropout(dropout_val)
        )
        self.layer4 = nn.Sequential(
            nn.ConvTranspose1d(n_out3, n_out2, kernel_size=kernel_size3, padding=padding3),
            nn.Sigmoid()
        )
        self.layer5 = nn.Sequential(
            nn.ConvTranspose1d(n_out2, n_out, kernel_size=kernel_size2, padding=padding2),
            nn.ReLU()
        )
        self.layer6 = nn.Sequential(
            nn.ConvTranspose1d(n_out, n_in, kernel_size=kernel_size, padding=padding)
        )

    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = self.layer5(out)
        out = self.layer6(out)
        return out


# ---------------------------------------------------------------------------
# Data of the paper
# ---------------------------------------------------------------------------

CORR_FS = 30
SEGMENT_LEN = 288  # 9.6 s at 30 Hz
TEST_HOP = 30  # 1 s
PREDICT_BATCH = 512


def load_backbone(path: Path, name: str):
    """Import the 261007 or 261013 script as a module (nothing is trained)."""
    spec = importlib.util.spec_from_file_location(name, str(path))
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def read_dataset(backbone, dataset_key: str, spec) -> Tuple[Dict[str, dict], Dict[str, list]]:
    """The data of 261007 / 261013 (test windows, references and minutes for
    the evaluation) and, per subject, the PPG and reference of every recording
    piece as read there (PPG, reference, PPG fs, reference fs), in the order
    of their segment ids."""
    captured: Dict[int, list] = {}
    original = backbone.preprocess_subject_from_pairs

    def recording(pairs, *args, **kwargs):
        pairs = list(pairs)
        result = original(pairs, *args, **kwargs)
        if result is not None:
            captured[id(result)] = [
                (np.array(ppg, dtype=np.float64), np.array(rsp, dtype=np.float64), float(ppg_fs), float(rsp_fs))
                for ppg, rsp, ppg_fs, rsp_fs in pairs
            ]
        return result

    backbone.preprocess_subject_from_pairs = recording
    try:
        data = backbone.load_dataset(dataset_key, spec)
    finally:
        backbone.preprocess_subject_from_pairs = original
    return data, {subject: captured[id(record)] for subject, record in data.items()}


def fill_missing(signal: np.ndarray) -> Tuple[np.ndarray, int]:
    """Missing samples interpolated linearly (all missing: zeros)."""
    x = np.asarray(signal, dtype=np.float64).copy()
    bad = ~np.isfinite(x)
    if not bad.any():
        return x, 0
    if bad.all():
        return np.zeros_like(x), int(bad.sum())
    index = np.arange(x.size)
    x[bad] = np.interp(index[bad], index[~bad], x[~bad])
    return x, int(bad.sum())


def to_corr_fs(signal: np.ndarray, fs: float) -> Tuple[np.ndarray, int]:
    """Resampled to 30 Hz (polyphase, as MATLAB resample)."""
    x, missing = fill_missing(signal)
    ratio = Fraction(CORR_FS / float(fs)).limit_denominator(1000)
    if ratio != 1:
        x = resample_poly(x, ratio.numerator, ratio.denominator)
    return x, missing


def corr_pieces(raw_pairs: Sequence[tuple]) -> Tuple[List[Tuple[np.ndarray, np.ndarray]], dict]:
    """PPG and reference of every recording piece at 30 Hz."""
    pieces, missing_ppg, missing_ref = [], 0, 0
    for ppg, rsp, ppg_fs, rsp_fs in raw_pairs:
        ppg30, m_ppg = to_corr_fs(ppg, ppg_fs)
        ref30, m_ref = to_corr_fs(rsp, rsp_fs)
        length = min(len(ppg30), len(ref30))
        pieces.append((ppg30[:length], ref30[:length]))
        missing_ppg += m_ppg
        missing_ref += m_ref
    return pieces, {"Missing_PPG_Samples": missing_ppg, "Missing_Reference_Samples": missing_ref}


def standardise(segments: np.ndarray) -> Tuple[np.ndarray, int]:
    """Zero mean and unit SD per segment (a constant segment: zeros)."""
    mean = segments.mean(axis=1, keepdims=True)
    sd = segments.std(axis=1, keepdims=True)
    constant = sd[:, 0] <= 1e-12
    out = np.where(constant[:, None], 0.0, (segments - mean) / np.where(constant[:, None], 1.0, sd))
    return out, int(constant.sum())


def scale_01(segments: np.ndarray) -> Tuple[np.ndarray, int]:
    """Min-max to 0-1 per segment (a constant segment: zeros)."""
    low = segments.min(axis=1, keepdims=True)
    span = segments.max(axis=1, keepdims=True) - low
    constant = span[:, 0] <= 1e-12
    out = np.where(constant[:, None], 0.0, (segments - low) / np.where(constant[:, None], 1.0, span))
    return out, int(constant.sum())


def training_segments(pieces: Sequence[Tuple[np.ndarray, np.ndarray]]) -> Tuple[np.ndarray, np.ndarray]:
    """Non-overlapping 288-sample segments from the start of every piece."""
    xs, ys = [], []
    for ppg30, ref30 in pieces:
        count = len(ppg30) // SEGMENT_LEN
        if count:
            xs.append(ppg30[:count * SEGMENT_LEN].reshape(count, SEGMENT_LEN))
            ys.append(ref30[:count * SEGMENT_LEN].reshape(count, SEGMENT_LEN))
    if not xs:
        return np.empty((0, SEGMENT_LEN)), np.empty((0, SEGMENT_LEN))
    return np.concatenate(xs), np.concatenate(ys)


def test_starts(length: int) -> List[int]:
    """Sliding window of 288 samples moved by 1 s, plus one window ending at
    the last sample."""
    if length < SEGMENT_LEN:
        return []
    starts = list(range(0, length - SEGMENT_LEN + 1, TEST_HOP))
    if starts[-1] != length - SEGMENT_LEN:
        starts.append(length - SEGMENT_LEN)
    return starts


def test_segments(ppg30: np.ndarray, ref30: np.ndarray) -> Tuple[List[int], np.ndarray, np.ndarray]:
    starts = test_starts(len(ppg30))
    if not starts:
        return starts, np.empty((0, SEGMENT_LEN)), np.empty((0, SEGMENT_LEN))
    x = np.stack([ppg30[s:s + SEGMENT_LEN] for s in starts])
    y = np.stack([ref30[s:s + SEGMENT_LEN] for s in starts])
    return starts, standardise(x)[0], scale_01(y)[0]


def overlap_average(windows: np.ndarray, starts: Sequence[int], length: int) -> np.ndarray:
    total = np.zeros(length)
    count = np.zeros(length)
    for window, start in zip(windows, starts):
        total[start:start + SEGMENT_LEN] += window
        count[start:start + SEGMENT_LEN] += 1
    out = np.full(length, np.nan)
    out[count > 0] = total[count > 0] / count[count > 0]
    return out


# ---------------------------------------------------------------------------
# Training and testing of one fold
# ---------------------------------------------------------------------------

def set_seed() -> None:
    torch.manual_seed(seed_val)
    random.seed(seed_val)
    np.random.seed(seed_val)


def predict(model: nn.Module, segments: np.ndarray, device: torch.device) -> np.ndarray:
    model.eval()
    outputs = []
    with torch.no_grad():
        for start in range(0, len(segments), PREDICT_BATCH):
            batch = torch.from_numpy(segments[start:start + PREDICT_BATCH, None, :].astype("float32")).to(device)
            outputs.append(model(batch)[:, 0, :].cpu().numpy().astype(np.float64))
    return np.concatenate(outputs) if outputs else np.empty((0, SEGMENT_LEN))


def train_fold(train_x: np.ndarray, train_y: np.ndarray, test_x: np.ndarray, test_y: np.ndarray,
               epochs: int, device: torch.device) -> Tuple[nn.Module, pd.DataFrame]:
    """The authors' training loop; returns the model after the last epoch and
    the loss of every epoch (test loss in eval mode, logged only)."""
    set_seed()
    # shuffle training data (as sklearn.utils.shuffle with the global numpy state)
    order = np.arange(len(train_x))
    np.random.shuffle(order)
    trainX, trainy = train_x[order], train_y[order]
    model = Correncoder_model().to(device)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    total_step = trainX.shape[0]
    trainXT = torch.from_numpy(trainX[:, None, :].astype("float32")).to(device)
    trainyT = torch.from_numpy(trainy.astype("float32")).to(device)
    testXT = torch.from_numpy(test_x[:, None, :].astype("float32")).to(device)
    testyT = torch.from_numpy(test_y.astype("float32")).to(device)
    history = []
    for epoch in range(epochs):
        started = time.time()
        model.train()
        losses = []
        for i in range(total_step // batch_size):  # split data into batches
            trainXT_seg = trainXT[i * batch_size:(i + 1) * batch_size, :, :]
            trainyT_seg = trainyT[i * batch_size:(i + 1) * batch_size, None]
            outputs = model(trainXT_seg)
            loss = criterion(outputs, trainyT_seg)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            losses.append(loss.item())
        model.eval()
        with torch.no_grad():
            test_loss = float(criterion(model(testXT), testyT[:, None]).item()) if len(testXT) else math.nan
        history.append({
            "Epoch": epoch + 1,
            "Train_Loss_Mean": float(np.mean(losses)) if losses else math.nan,
            "Train_Loss_Last_Batch": losses[-1] if losses else math.nan,
            "Test_Loss_Logged_Only": test_loss,
            "Seconds": time.time() - started,
        })
    return model, pd.DataFrame(history)


def to_model_fs(series30: np.ndarray, fs: int, length: int) -> np.ndarray:
    """A 30 Hz waveform at the evaluation rate (64 Hz), cut / padded to the
    length of the 261007 / 261013 recording piece."""
    ratio = Fraction(fs, CORR_FS)
    x = resample_poly(np.nan_to_num(series30, nan=0.0), ratio.numerator, ratio.denominator)
    if len(x) >= length:
        return x[:length]
    return np.concatenate([x, np.full(length - len(x), x[-1] if len(x) else 0.0)])


def evaluate_fold(backbone, subject_data: Dict[str, dict], subject: str, outputs30: Dict[int, np.ndarray],
                  references30: Dict[int, np.ndarray]) -> Tuple[dict, pd.DataFrame]:
    """The 261007 / 261013 evaluation on their test windows of the subject."""
    record = subject_data[subject]
    fs = int(backbone.TARGET_FS)
    metadata = backbone.test_metadata(subject_data, subject)
    window_len = int(np.asarray(record["low"]).shape[1])
    dataset = backbone.SubjectWindowDataset(subject_data, [subject], "test", False)
    targets = np.stack([dataset[index][1][0].numpy() for index in range(len(dataset))]).astype(np.float64)
    pred64, ref64 = {}, {}
    for segment_id in sorted({int(item["segment_id"]) for item in metadata}):
        length = len(record["references"][segment_id]["model_input"])
        pred64[segment_id] = to_model_fs(outputs30[segment_id], fs, length)
        ref64[segment_id] = to_model_fs(references30[segment_id], fs, length)

    def windows(series: Dict[int, np.ndarray]) -> np.ndarray:
        return np.stack([
            series[int(item["segment_id"])][int(item["start"]):int(item["start"]) + window_len] for item in metadata
        ])

    predictions = windows(pred64)
    summary, minutes = backbone.evaluate_subject_minutes(
        {"predictions": predictions, "targets": targets}, metadata, record["references"], fs
    )
    _, own = backbone.evaluate_subject_minutes(
        {"predictions": predictions, "targets": windows(ref64)}, metadata, record["references"], fs
    )
    if not minutes.empty:
        minutes["Wave_PCC_Own_Reference"] = own["Wave_PCC"].to_numpy()
        valid = minutes[minutes["Ref_Valid"].astype(bool)]
        summary["Wave_PCC_Own_Reference_Mean"] = (
            float(valid["Wave_PCC_Own_Reference"].mean()) if len(valid) else math.nan
        )
    return summary, minutes


def fold_paths(results_root: Path, label: str, subject: str) -> Dict[str, Path]:
    root = results_root / label
    return {
        "model": root / "Models" / f"{subject}_model.pth",
        "log": root / "Training_Logs" / f"{subject}_loss.csv",
        "predictions": root / "Predictions" / f"{subject}_outputs.npz",
        "summary": root / "Fold_Results" / f"{subject}_summary.json",
        "minutes": root / "Fold_Results" / f"{subject}_minutes.csv",
    }


def read_json(path: Path) -> Optional[dict]:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


def json_default(value):
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f"Not JSON serializable: {type(value)!r}")


def run_settings(args: argparse.Namespace) -> dict:
    return {
        "Code": CODE_NAME,
        "Epochs": int(args.epochs),
        "Batch_Size": batch_size,
        "Learning_Rate": learning_rate,
        "Seed": seed_val,
        "Kernels": [kernel_size, kernel_size2, kernel_size3],
        "Padding": [padding, padding2, padding3],
        "Dropout": dropout_val,
        "Sampling_Rate_Hz": CORR_FS,
        "Segment_Samples": SEGMENT_LEN,
        "Test_Hop_Samples": TEST_HOP,
    }


def run_dataset(args: argparse.Namespace, backbone, dataset_key: str, results_root: Path,
                device: torch.device) -> List[dict]:
    label = backbone.DATASET_CONFIGS[dataset_key]["label"]
    spec = next(s for s in backbone.WINDOW_SPECS if s.name == WINDOW)
    print(f"\n[{time.strftime('%H:%M:%S')}] {label}: reading the data ({DATASET_SOURCES[dataset_key]} reader) ...",
          flush=True)
    subject_data, raw = read_dataset(backbone, dataset_key, spec)
    subjects_all = sorted(subject_data)
    if len(subjects_all) < 2:
        print(f"{label}: fewer than two subjects, skipped")
        return []
    pieces, checks = {}, []
    for subject in subjects_all:
        pieces[subject], info = corr_pieces(raw[subject])
        x, y = training_segments(pieces[subject])
        _, constant_ppg = standardise(x)
        _, constant_ref = scale_01(y)
        checks.append({
            "Dataset": label, "Subject": subject, "Pieces": len(pieces[subject]),
            "Seconds_30Hz": float(sum(len(p) for p, _ in pieces[subject]) / CORR_FS),
            "Training_Segments": int(len(x)), "Constant_PPG_Segments": constant_ppg,
            "Constant_Reference_Segments": constant_ref, **info,
        })
    del raw
    segments = {subject: training_segments(pieces[subject]) for subject in subjects_all}
    print(f"{label}: {len(subjects_all)} subjects, "
          f"{sum(len(x) for x, _ in segments.values())} training segments of {SEGMENT_LEN} samples at {CORR_FS} Hz",
          flush=True)
    subjects = [s for s in subjects_all if not args.subjects or s in {str(x) for x in args.subjects}]
    settings = run_settings(args)
    for subject in subjects:
        paths = fold_paths(results_root, label, subject)
        stored = read_json(paths["summary"]) if args.resume else None
        if stored is not None and stored.get("Settings") == settings and paths["minutes"].exists():
            print(f"[{time.strftime('%H:%M:%S')}] {label} {subject}: stored, kept", flush=True)
            continue
        train_subjects = [s for s in subjects_all if s != subject]
        train_x = np.concatenate([segments[s][0] for s in train_subjects])
        train_y = np.concatenate([segments[s][1] for s in train_subjects])
        train_x, train_y = standardise(train_x)[0], scale_01(train_y)[0]
        tests = [test_segments(ppg30, ref30) for ppg30, ref30 in pieces[subject]]
        test_x = np.concatenate([x for _, x, _ in tests])
        test_y = np.concatenate([y for _, _, y in tests])
        print(f"[{time.strftime('%H:%M:%S')}] {label} {subject}: training on {len(train_x)} segments of "
              f"{len(train_subjects)} subjects, {args.epochs} epochs", flush=True)
        started = time.time()
        model, history = train_fold(train_x, train_y, test_x, test_y, args.epochs, device)
        training_seconds = time.time() - started
        outputs30, references30, arrays = {}, {}, {}
        for segment_id, ((ppg30, ref30), (starts, x, y)) in enumerate(zip(pieces[subject], tests)):
            length = len(ppg30)
            outputs30[segment_id] = overlap_average(predict(model, x, device), starts, length)
            references30[segment_id] = overlap_average(y, starts, length)
            arrays[f"piece{segment_id}_output_30hz"] = outputs30[segment_id].astype(np.float32)
            arrays[f"piece{segment_id}_reference_30hz"] = references30[segment_id].astype(np.float32)
            arrays[f"piece{segment_id}_ppg_30hz"] = ppg30.astype(np.float32)
        summary, minutes = evaluate_fold(backbone, subject_data, subject, outputs30, references30)
        for path in paths.values():
            path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(model.state_dict(), paths["model"])
        history.to_csv(paths["log"], index=False, encoding="utf-8-sig")
        np.savez_compressed(paths["predictions"], **arrays)
        if not minutes.empty:
            minutes.insert(0, "Subject", str(subject))
            minutes.insert(0, "Dataset", label)
        minutes.to_csv(paths["minutes"], index=False, encoding="utf-8-sig")
        summary.update({
            "Dataset": label, "Subject": str(subject), "Settings": settings,
            "Train_Subjects": len(train_subjects), "Train_Segments": int(len(train_x)),
            "Test_Segments": int(len(test_x)), "Training_Seconds": training_seconds,
            "Final_Train_Loss_Mean": float(history["Train_Loss_Mean"].iloc[-1]),
        })
        # Written last: its presence marks the fold as complete.
        paths["summary"].write_text(json.dumps(summary, indent=2, default=json_default), encoding="utf-8")
        print(f"[{time.strftime('%H:%M:%S')}] {label} {subject}: FFT MAE {summary['RR_MAE_Wave_FFT']:.3f} "
              f"breaths/min ({summary['N60_RefValid']} valid minutes), PCC {summary['Wave_PCC_Mean']:.3f}, "
              f"PCC own reference {summary.get('Wave_PCC_Own_Reference_Mean', math.nan):.3f}, "
              f"training {training_seconds / 60:.1f} min", flush=True)
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()
    return checks


# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------

def summarize(results_root: Path, labels: Sequence[str], evaluator, checks: pd.DataFrame) -> None:
    summary_root = results_root / "Summary"
    summary_root.mkdir(parents=True, exist_ok=True)
    frames = []
    for label in labels:
        for path in sorted((results_root / label / "Fold_Results").glob("*_minutes.csv")):
            if path.stat().st_size > 0:
                frame = pd.read_csv(path, dtype={"Subject": str}, encoding="utf-8-sig")
                if not frame.empty:
                    frames.append(frame)
    if not frames:
        print("No fold results to summarize yet.")
        return
    minutes = pd.concat(frames, ignore_index=True)
    scored = evaluator.scored_minutes(minutes)
    overall_rows, subject_tables = [], []
    for label in labels:
        part = scored[scored["Dataset"] == label]
        if part.empty:
            continue
        own = pd.to_numeric(part["Wave_PCC_Own_Reference"], errors="coerce").dropna()
        overall_rows.append({
            "Dataset": label, "Window_Config": WINDOW, "Row": "Deep corr-encoder", "Model": "Deep corr-encoder",
            **evaluator.fft_davies_metrics(part),
            "Wave_PCC_Own_Reference_Mean": float(own.mean()) if len(own) else math.nan,
            "Wave_PCC_Own_Reference_SD": float(own.std(ddof=1)) if len(own) > 1 else math.nan,
            "Wave_PCC_Own_Reference_Median": float(own.median()) if len(own) else math.nan,
        })
        table = evaluator.per_subject_table(part)
        if not table.empty:
            own_subject = part.groupby("Subject")["Wave_PCC_Own_Reference"].mean()
            table["Wave_PCC_Own_Reference_Mean"] = table["Subject"].map(own_subject)
            table.insert(0, "Dataset", label)
            subject_tables.append(table)
    overall = pd.DataFrame(overall_rows)
    mean_sd = evaluator.mean_sd_table(overall)
    mean_sd["Wave PCC own reference (mean ± SD over minutes)"] = [
        f"{m:.3f} ± {s:.3f}" for m, s in zip(overall["Wave_PCC_Own_Reference_Mean"], overall["Wave_PCC_Own_Reference_SD"])
    ]
    per_subject = pd.concat(subject_tables, ignore_index=True) if subject_tables else pd.DataFrame()
    tables = {
        "correncoder_overall": overall,
        "correncoder_overall_mean_sd": mean_sd,
        "per_subject": per_subject,
        "minutes": minutes,
        "data_check": checks,
    }
    for name, frame in tables.items():
        frame.to_csv(summary_root / f"{name}.csv", index=False, encoding="utf-8-sig")
    evaluator.write_results_excel(
        summary_root / "261028_Deep_correncoder_results.xlsx",
        {"Overall_mean_SD": mean_sd, "Overall": overall, "Per_subject": per_subject, "Data_check": checks,
         "Minutes": minutes},
    )
    print("\nDeep corr-encoder, 60 s minutes (breaths/min; mean ± SD, medians with IQR):")
    for _, row in mean_sd.iterrows():
        print(f"  {row['Dataset']:<10s} subj {row['N_Subjects']:>3d} min {row['N_Minutes']:>4d}  "
              f"MAE {row['MAE (mean ± SD over minutes)']:<15s} subject MAE "
              f"{row['Subject MAE (mean ± SD over subjects)']:<15s} mAE {row['mAE [IQR]']}  "
              f"mMAE {row['mMAE [IQR]']}  PCC {row['Wave PCC (mean ± SD over minutes)']}  "
              f"PCC own reference {row['Wave PCC own reference (mean ± SD over minutes)']}")
    print(f"\nSaved the summary under: {summary_root}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--results-root", type=Path, default=DEFAULT_RESULTS_ROOT)
    parser.add_argument("--datasets", nargs="+", choices=list(DATASET_ORDER), default=list(DATASET_ORDER))
    parser.add_argument("--subjects", nargs="+", default=None,
                        help="Only these held-out subjects (quick tests); training still uses all others.")
    parser.add_argument("--epochs", type=int, default=num_epochs, help="Epochs (authors: %(default)s).")
    parser.add_argument("--backbone-261007", type=Path, default=BACKBONE_261007,
                        help="261007_RRWaveNet_Ensemble_NoEvent.py (CapnoBase reader and evaluation).")
    parser.add_argument("--backbone-261013", type=Path, default=BACKBONE_261013,
                        help="261013_RRWaveNet_Adult_SubjectVal_WD.py (BIDMC, STEAM2 reader and evaluation).")
    parser.add_argument("--resume", action="store_true", help="Keep the folds already finished.")
    parser.add_argument("--summary-only", action="store_true")
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--force-cpu", action="store_true")
    parser.add_argument("--threads", type=int, default=None)
    parser.add_argument("--capnobase-path", type=Path, default=None)
    parser.add_argument("--bidmc-path", type=Path, default=None)
    parser.add_argument("--steam2-ppg-dir", type=Path, default=None)
    parser.add_argument("--steam2-rsp-dir", type=Path, default=None)
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> None:
    args = parse_args(argv)
    backbones = {
        "261007": load_backbone(Path(args.backbone_261007).resolve(), "rrwavenet_261007_correncoder"),
        "261013": load_backbone(Path(args.backbone_261013).resolve(), "rrwavenet_261013_correncoder"),
    }
    for module in backbones.values():
        configs = module.DATASET_CONFIGS
        if args.capnobase_path and "capnobase" in configs:
            configs["capnobase"]["data_path"] = str(args.capnobase_path)
        if args.bidmc_path:
            configs["bidmc"]["data_path"] = str(args.bidmc_path)
        if args.steam2_ppg_dir:
            configs["steam2"]["ppg_dir"] = str(args.steam2_ppg_dir)
        if args.steam2_rsp_dir:
            configs["steam2"]["rsp_dir"] = str(args.steam2_rsp_dir)
    if args.threads:
        torch.set_num_threads(int(args.threads))
    results_root = Path(args.results_root).resolve()
    results_root.mkdir(parents=True, exist_ok=True)
    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() and not args.force_cpu else "cpu")
    dataset_keys = [key for key in DATASET_ORDER if key in args.datasets]
    labels = [backbones[DATASET_SOURCES[key]].DATASET_CONFIGS[key]["label"] for key in dataset_keys]
    (results_root / "Summary").mkdir(parents=True, exist_ok=True)
    (results_root / "Summary" / "run_settings.json").write_text(
        json.dumps({**run_settings(args), "Datasets": labels, "Device": str(device),
                    "Readers_and_evaluation": {label: DATASET_SOURCES[key] for key, label in zip(dataset_keys, labels)}},
                   indent=2),
        encoding="utf-8",
    )
    n_parameters = sum(p.numel() for p in Correncoder_model().parameters())
    print(f"Deep corr-encoder ({n_parameters} parameters), {args.epochs} epochs, batch {batch_size}, "
          f"learning rate {learning_rate:g}, seed {seed_val}; {SEGMENT_LEN} samples at {CORR_FS} Hz; device={device}")
    checks = []
    if not args.summary_only:
        for key in dataset_keys:
            checks += run_dataset(args, backbones[DATASET_SOURCES[key]], key, results_root, device)
        pd.DataFrame(checks).to_csv(results_root / "Summary" / "data_check.csv", index=False, encoding="utf-8-sig")
    else:
        path = results_root / "Summary" / "data_check.csv"
        checks = pd.read_csv(path, dtype={"Subject": str}).to_dict("records") if path.exists() else []
    summarize(results_root, labels, backbones["261013"], pd.DataFrame(checks))


if __name__ == "__main__":
    main()
