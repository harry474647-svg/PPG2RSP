r"""261030_StemFusion_Training_Dynamics: training dynamics of Stem Fusion O
(full model) and Stem Fusion X (no_stem_fusion) from their stored learning
curves and models, 10 s windows: convergence, stability of the training and
the gap between training and validation. CapnoBase from
261007_RRWaveNet_Ensemble_NoEvent, BIDMC and STEAM2 from
261013_RRWaveNet_Adult_SubjectVal_WD, all in one run. The stored models are
used; only a model without stem fusion that is stored nowhere is trained
(see "One run for everything").

Models (LOSOCV fold = held-out subject; 10 s non-overlapping windows)
---------------------------------------------------------------------
* CapnoBase (261007 results, --results-root-261007): O = full\, X =
  no_stem_fusion\, members m0-m2 of both (the same three seeds);
* BIDMC, STEAM2 (261013 results, --results-root-261013): O = the full model
  of the weight decay 261013 selected in the fold
  (WD_Selection\01_w10s_nonoverlap\<Dataset>\<subject>_selection.json ->
  full_wd<wd>\), members m0-m2. 261013 trained no model without stem fusion:
  X is member m0 of no_stem_fusion\ in the results of
  261028_Stem_ablation_noise_v1 (--results-root-noise), trained with the
  261013 settings of the fold (its validation subjects, the selected weight
  decay, the seed of member m0); a model trained with another weight decay
  is not used.

One run for everything
----------------------
One run covers CapnoBase (261007) and BIDMC and STEAM2 (261013). The X models
are looked for in the 261007 results (CapnoBase) or the
261028_Stem_ablation_noise_v1 results (BIDMC, STEAM2), then in
<output root>\Trained_no_stem_fusion\. An X model found in neither is
trained here first, as its script trains it (run_member of the 261007 /
261013 file: the fold's training and validation split, the seed of the
member, the selected weight decay for 261013, the epochs of the fold's O
model, early stopping as there), stored in <output root>\
Trained_no_stem_fusion\ and then analysed with the others; the next run
reuses it (an interrupted training continues). The result folders of 261007,
261013 and 261028_Stem_ablation_noise_v1 are only read.
--no-train-missing leaves such folds without X. The number of X models to
train is printed per dataset before anything is trained. Training runs on
the CPU unless --use-gpu (much faster when the GPU is free).

O vs X is paired by fold on member m0 (same seed, training and validation
data and weight decay) for every dataset; for CapnoBase also on the mean of
members m0-m2 of each (pair metrics per member pair m_k / m_k). Every model
needs its fold summary and learning curve
(Members\m<k>\Learning_Curves\<subject>_Learning_Curve.csv: epoch,
training loss and validation loss of every epoch, as stored by 261007 /
261013); the restored-model gap also reloads its checkpoint
(Members\m<k>\Models\<subject>_best.pth, the best validation epoch).

Training settings, as stored (the same for O and X of a fold)
--------------------------------------------------------------
CapnoBase (261007): validation = the last 20% of every training subject's
windows; BIDMC and STEAM2 (261013): validation = 20% of the training
subjects held out whole. AdamW, learning rate 2e-4 with cosine decay, up to
240 epochs, at least 20, patience 25, best validation epoch restored. The
training loss of the learning curve is the mean over the epoch's batches
with the jitter of training windows and dropout on; the validation loss is
in eval mode.

Measures per model (learning curve; late phase = the last 20% of the epochs
run, at least 5)
---------------------------------------------------------------------------
1. Convergence
   * Best_Val_Loss: lowest validation loss (the restored model);
   * Epochs_To_Common_Target: first epoch at which the validation loss
     reaches the common target of the pair, the higher of the two models'
     lowest validation losses (a level both reach);
   * Epochs_To_95pct: first epoch with 95% of the model's improvement from
     the first to the lowest validation loss (as 261007 / 261013);
   * Loss_Reduction_Ratio: (first - lowest validation loss) / first;
   * Val_AUC_Common: mean validation loss over the epochs both models of the
     pair trained (1 to the shorter run), i.e. the area under the
     validation curve per epoch on a common epoch axis.
2. Stability of the training
   * Val_Oscillation_Pct: SD of the epoch-to-epoch change of the validation
     loss in the late phase, in % of the lowest validation loss;
   * Monotonic_Decrease_Ratio: share of epochs whose validation loss is
     below that of the epoch before;
   * Spike_Count: epochs whose validation loss is more than 5% above that
     of the epoch before;
   * Rebound_Final_Pct / Rebound_Late_Pct / Rebound_Max_Pct: last epoch,
     mean of the late phase and highest loss after the best epoch, in % above
     the lowest validation loss (0 when the best epoch is the last);
   * seed variability (models with members m0-m2: CapnoBase O and X,
     BIDMC / STEAM2 O): per fold the SD over the three seeds of the lowest
     validation loss (also as % of their mean), of the best epoch, of
     Epochs_To_95pct and of the restored-model gap;
   * fold variability: SD and IQR over folds of the best epoch, the lowest
     validation loss, Epochs_To_95pct and the restored-model gap, O vs X
     with the Pitman-Morgan test for the variances of paired samples.
3. Training vs validation
   * Gap_Loss / Gap_Loss_Pct / Gap_PCC: the restored model (best epoch) on
     the training and the validation windows of its fold, both in eval mode
     without the jitter: validation - training waveform loss, the same in %
     of the training loss, and training - validation waveform PCC (mean
     Pearson r of output and target per window);
   * Gap_History_Abs: |validation - training loss| of the learning curve at
     the best epoch (signed value: Gap_History);
   * Gap_Slope_per100: slope of (validation - training loss) over the epochs
     of the learning curve (least squares), per 100 epochs: how fast the gap
     widens;
   * Late_Gap_Abs_Mean / Late_Gap_SD: mean |validation - training loss| and
     SD of (validation - training loss) in the late phase;
   * Overfit_Signature_Epochs: epochs after the best epoch with a higher
     validation and a lower training loss than at the best epoch;
   * training length (descriptive): Best_Epoch, Epochs_Run, Epochs_After_Best
     and the share of folds stopped early.
The gap of the learning curve mixes training mode (jitter, dropout) with eval
mode, so it can be negative; the restored-model gap compares both splits in
eval mode and is the main gap measure. The validation sets differ between
datasets (CapnoBase: later windows of the training subjects; BIDMC, STEAM2:
other subjects), so the gaps are compared within a dataset.

Statistics
----------
Per dataset and pairing (m0; CapnoBase also mean of m0-m2), O vs X paired by
fold: mean +- SD and median [IQR] of each, the median of the paired
differences (O - X), the folds in which each model is better (by the
direction of the measure), the matched-pairs rank-biserial correlation and a
two-sided Wilcoxon signed-rank test, Benjamini-Hochberg over the measures of
a dataset and pairing. Seed variability: Wilcoxon on the per-fold SDs
(CapnoBase); fold variability: Pitman-Morgan test.

Outputs (PROJECT_ROOT = D:\PPG2RSP_RRWaveNet_Inspired\261030_StemFusion_Training_Dynamics)
  verdict.csv            per dataset and measure group (convergence, stability,
                         gap): the measures where O or X is significantly
                         better, and the verdict (O better / X better / mixed /
                         no difference)
  models.csv             every model (dataset, fold, O / X, member): all measures
  pairs.csv              O and X side by side per fold and pairing
  tests.csv              O vs X: summary statistics and tests of every measure
  seed_variability.csv   per fold and model: SD over the three seeds;
                         seed_variability_tests.csv
  fold_variability.csv   SD and IQR over folds, Pitman-Morgan test
  early_stopping.csv     folds stopped early per dataset and model
  coverage.csv           folds and models found (and where) / left out (with
                         the reason)
  Trained_no_stem_fusion\  X models trained here (only those stored nowhere)
  261030_StemFusion_Training_Dynamics.xlsx   all tables
  Figures\<Dataset>_learning_curves.png      median validation loss and gap
                                             over epochs (O vs X, m0)
  Figures\<Dataset>_<group>.png              box plots per measure group

Running next to a training on the GPU (e.g. RespDiff)
-----------------------------------------------------
The defaults keep out of the way of another training on the same computer:
* CPU only: the GPU is not touched (no second process on it, no CUDA
  libraries loaded for it); the reloaded networks have 8,729-9,377
  parameters, so the CPU is enough. --use-gpu uses the GPU;
* lower process priority (below normal) and 2 CPU threads (--threads), so
  the training keeps the computer first; --normal-priority to keep normal;
* free memory check: before the raw data of a dataset are read (only for
  the restored-model gap or to train a missing X model, a few GB of RAM),
  the free RAM is checked; below 4 GB (--min-free-ram-gb) the data are not
  read and that dataset gets the learning-curve measures only (listed at the
  end; --min-free-ram-gb 0 reads anyway). The datasets are read one at a
  time and released after use;
* lightest run: --no-restored-gap reads only the learning curves and fold
  summaries (no model reloading), everything except the restored-model gap
  (Gap_Loss, Gap_Loss_Pct, Gap_PCC); the raw data are then read only when an
  X model has to be trained (with --no-train-missing never);
* training missing X models (see "One run for everything") uses the CPU by
  default, below normal priority, so it is slow but leaves the GPU to the
  other training; with the GPU free, --use-gpu is much faster;
* the result folders of 261007, 261013 and 261028_Stem_ablation_noise_v1 are
  only read; everything is written under --output-root, so nothing is shared
  with a training writing its own results (RespDiff: 261020_RespDiff_v2).
Start it in its own console (a new Spyder console or Anaconda Prompt), not in
the console of the training.

Run
---
Put this file in the folder of 261007_RRWaveNet_Ensemble_NoEvent.py and
261013_RRWaveNet_Adult_SubjectVal_WD.py (their data paths, reading, split and
model are used), or give their paths with --backbone-261007 /
--backbone-261013. Spyder: run the file in a new console.
  python 261030_StemFusion_Training_Dynamics.py                    (all measures)
  python 261030_StemFusion_Training_Dynamics.py --no-restored-gap  (lightest)
Options: --results-root-261007 / --results-root-261013 / --results-root-noise,
  --output-root, --datasets capnobase bidmc steam2, --use-gpu, --gpu,
  --threads, --min-free-ram-gb, --normal-priority, --no-train-missing,
  --train-min-epochs, --train-patience, and the data path options of
  261007 / 261013.
"""

import argparse
import importlib.util
import json
import math
import os
import sys
import warnings
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

sys.dont_write_bytecode = True

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from scipy import stats

SCRIPT_DIR = Path(__file__).resolve().parent
BACKBONE_261007 = SCRIPT_DIR / "261007_RRWaveNet_Ensemble_NoEvent.py"
BACKBONE_261013 = SCRIPT_DIR / "261013_RRWaveNet_Adult_SubjectVal_WD.py"
PROJECT_ROOT = Path(r"D:\PPG2RSP_RRWaveNet_Inspired\261030_StemFusion_Training_Dynamics")
RESULTS_ROOT_261007 = Path(r"D:\PPG2RSP_RRWaveNet_Inspired\261007_RRWaveNet_NoEvent_v1")
RESULTS_ROOT_261013 = Path(r"D:\PPG2RSP_RRWaveNet_Inspired\261013_RRWavNet_STEAM_BIDMC_v1")
# BIDMC / STEAM2 models without stem fusion: single models m0 trained by
# 261028_Stem_ablation_noise_v1 with the 261013 settings of the fold.
RESULTS_ROOT_NOISE = Path(r"D:\PPG2RSP_RRWaveNet_Inspired\261028_Stem_ablation_noise_v1")
DATASET_SOURCES = {"capnobase": "261007", "bidmc": "261013", "steam2": "261013"}
DATASET_ORDER = ("capnobase", "bidmc", "steam2")
WINDOW = "w10s_nonoverlap"
MEMBERS = 3
PAIR_MEMBER = 0
O_VARIANT = "full"
X_VARIANT = "no_stem_fusion"
MODEL_O, MODEL_X = "Stem Fusion O", "Stem Fusion X"
# Colours of the earlier stem fusion figures (X red, O blue).
MODEL_COLORS = {MODEL_X: "#e8968c", MODEL_O: "#5b8db8"}
TEXT_PRIMARY = "#0b0b0b"
TEXT_SECONDARY = "#52514e"
GRID_COLOR = "#e4e3df"

# Sharing the computer with a training on the GPU: CPU threads of this
# analysis and the free RAM needed before the raw data of a dataset are read.
DEFAULT_THREADS = 2
MIN_FREE_RAM_GB = 4.0
# Models without stem fusion stored nowhere are trained into this folder of
# the output root (and reused from there by the next run).
TRAINED_FOLDER = "Trained_no_stem_fusion"

LATE_PHASE_FRACTION = 0.20
LATE_PHASE_MIN_EPOCHS = 5
SPIKE_RISE = 0.05
# Figure: epochs at which at least this share of the folds still trains.
CURVE_MIN_SHARE = 0.5

# (group, column, label, unit, better: "lower" / "higher" / None = descriptive)
MEASURES = (
    ("Convergence", "Best_Val_Loss", "Lowest validation loss", "", "lower"),
    ("Convergence", "Epochs_To_Common_Target", "Epochs to the common target", "epochs", "lower"),
    ("Convergence", "Epochs_To_95pct", "Epochs to 95% of the improvement", "epochs", "lower"),
    ("Convergence", "Loss_Reduction_Ratio", "(first - lowest) / first validation loss", "", "higher"),
    ("Convergence", "Val_AUC_Common", "Mean validation loss, common epochs (AUC)", "", "lower"),
    ("Stability", "Val_Oscillation_Pct", "Validation loss oscillation, late phase", "% of lowest", "lower"),
    ("Stability", "Monotonic_Decrease_Ratio", "Share of epochs with a lower validation loss", "", "higher"),
    ("Stability", "Spike_Count", "Epochs with a validation loss rise > 5%", "epochs", "lower"),
    ("Stability", "Rebound_Final_Pct", "Final vs lowest validation loss", "%", "lower"),
    ("Stability", "Rebound_Late_Pct", "Late phase vs lowest validation loss", "%", "lower"),
    ("Stability", "Rebound_Max_Pct", "Highest after best vs lowest validation loss", "%", "lower"),
    ("Gap", "Gap_Loss", "Val - train loss, restored model", "", "lower"),
    ("Gap", "Gap_Loss_Pct", "(Val - train) / train loss, restored model", "%", "lower"),
    ("Gap", "Gap_PCC", "Train - val waveform PCC, restored model", "", "lower"),
    ("Gap", "Gap_History_Abs", "|Val - train loss| at the best epoch (curve)", "", "lower"),
    ("Gap", "Gap_Slope_per100", "Slope of (val - train loss) per 100 epochs", "", "lower"),
    ("Gap", "Late_Gap_Abs_Mean", "|Val - train loss|, late phase mean", "", "lower"),
    ("Gap", "Late_Gap_SD", "SD of (val - train loss), late phase", "", "lower"),
    ("Gap", "Overfit_Signature_Epochs", "Epochs with val up and train down after best", "epochs", "lower"),
    ("Training length", "Best_Epoch", "Epoch of the lowest validation loss", "epochs", None),
    ("Training length", "Epochs_Run", "Epochs trained", "epochs", None),
    ("Training length", "Epochs_After_Best", "Epochs after the best epoch", "epochs", None),
)
MEASURE_INFO = {column: (group, label, unit, better) for group, column, label, unit, better in MEASURES}
SEED_MEASURES = ("Best_Val_Loss", "Best_Epoch", "Epochs_To_95pct", "Gap_Loss")
FOLD_MEASURES = ("Best_Epoch", "Best_Val_Loss", "Epochs_To_95pct", "Gap_Loss")


def load_backbone(path: Path, name: str):
    """Import the 261007 or 261013 script as a module (no training is started)."""
    spec = importlib.util.spec_from_file_location(name, str(path))
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def read_json(path: Path) -> Optional[dict]:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


def free_ram_gb() -> Optional[float]:
    """Free (available) RAM in GB, None if it cannot be read."""
    try:
        import psutil

        return psutil.virtual_memory().available / 2 ** 30
    except Exception:
        return None


def lower_priority() -> str:
    """Run this process below normal priority, so that a training running at
    the same time keeps the CPU first."""
    try:
        import psutil

        process = psutil.Process()
        process.nice(psutil.BELOW_NORMAL_PRIORITY_CLASS if os.name == "nt" else 10)
        return "below normal"
    except Exception:
        try:
            os.nice(10)
            return "below normal"
        except Exception:
            return "normal (could not be lowered)"


# ---------------------------------------------------------------------------
# Learning curve measures
# ---------------------------------------------------------------------------

def read_curve(path: Path) -> Optional[pd.DataFrame]:
    """Epoch, training and validation loss of a stored learning curve."""
    try:
        frame = pd.read_csv(path, encoding="utf-8-sig")
    except (OSError, ValueError):
        return None
    if not {"Epoch", "Train_Loss", "Val_Loss"} <= set(frame.columns):
        return None
    frame = frame[["Epoch", "Train_Loss", "Val_Loss"]].apply(pd.to_numeric, errors="coerce").dropna()
    return frame.reset_index(drop=True) if len(frame) else None


def epochs_to_95pct(epochs: np.ndarray, val: np.ndarray) -> int:
    """First epoch reaching 95% of the improvement from the first to the
    lowest validation loss (convergence_epoch_95pct of 261007 / 261013)."""
    best = int(np.argmin(val))
    improvement = float(val[0] - val[best])
    if improvement <= 1e-12:
        return int(epochs[best])
    reached = np.flatnonzero(val <= val[0] - 0.95 * improvement)
    return int(epochs[reached[0]]) if len(reached) else int(epochs[best])


def curve_measures(curve: pd.DataFrame, max_epochs: int) -> dict:
    """Every measure of one learning curve that needs no partner model."""
    epochs = curve["Epoch"].to_numpy(float)
    val = curve["Val_Loss"].to_numpy(float)
    train = curve["Train_Loss"].to_numpy(float)
    count = len(val)
    best = int(np.argmin(val))
    best_val = float(val[best])
    late = min(count, max(LATE_PHASE_MIN_EPOCHS, int(math.ceil(LATE_PHASE_FRACTION * count))))
    change = np.diff(val)
    late_change = np.diff(val[-late:])
    gap = val - train
    after = val[best + 1:]
    signature = (after > best_val) & (train[best + 1:] < train[best])
    return {
        "Epochs_Run": int(count),
        "Max_Epochs": int(max_epochs),
        "Early_Stopped": int(count < max_epochs),
        "Best_Epoch": int(epochs[best]),
        "Epochs_After_Best": int(count - 1 - best),
        "First_Val_Loss": float(val[0]),
        "Best_Val_Loss": best_val,
        "Final_Val_Loss": float(val[-1]),
        "Epochs_To_95pct": epochs_to_95pct(epochs, val),
        "Loss_Reduction_Ratio": float((val[0] - best_val) / val[0]) if val[0] > 0 else math.nan,
        "Val_Oscillation_Pct": float(100.0 * np.std(late_change, ddof=1) / best_val)
        if len(late_change) >= 2 and best_val > 0 else math.nan,
        "Monotonic_Decrease_Ratio": float(np.mean(change < 0)) if len(change) else math.nan,
        "Spike_Count": int(np.sum(val[1:] > val[:-1] * (1.0 + SPIKE_RISE))),
        "Rebound_Final_Pct": float(100.0 * (val[-1] - best_val) / best_val),
        "Rebound_Late_Pct": float(100.0 * (np.mean(val[-late:]) - best_val) / best_val),
        "Rebound_Max_Pct": float(100.0 * (np.max(after) - best_val) / best_val) if len(after) else 0.0,
        "Gap_History": float(gap[best]),
        "Gap_History_Abs": float(abs(gap[best])),
        "Gap_Slope_per100": float(100.0 * np.polyfit(epochs, gap, 1)[0]) if count >= 3 else math.nan,
        "Late_Gap_Mean": float(np.mean(gap[-late:])),
        "Late_Gap_Abs_Mean": float(np.mean(np.abs(gap[-late:]))),
        "Late_Gap_SD": float(np.std(gap[-late:], ddof=1)) if late >= 2 else math.nan,
        "Overfit_Signature_Epochs": int(signature.sum()),
    }


def pair_measures(curve_o: pd.DataFrame, curve_x: pd.DataFrame) -> Tuple[dict, dict]:
    """Measures that need both models of a pair: epochs to the common
    target and the mean validation loss over the common epochs."""
    val_o, val_x = curve_o["Val_Loss"].to_numpy(float), curve_x["Val_Loss"].to_numpy(float)
    target = max(float(val_o.min()), float(val_x.min()))
    common = min(len(val_o), len(val_x))
    out = []
    for curve, val in ((curve_o, val_o), (curve_x, val_x)):
        reached = np.flatnonzero(val <= target + 1e-12)
        out.append({
            "Common_Target": target,
            "Epochs_To_Common_Target": float(curve["Epoch"].iloc[reached[0]]) if len(reached) else math.nan,
            "Common_Epochs": int(common),
            "Val_AUC_Common": float(np.mean(val[:common])),
        })
    return out[0], out[1]


# ---------------------------------------------------------------------------
# Restored-model gap (the 261007 / 261013 data, split and loss)
# ---------------------------------------------------------------------------

def split_loaders(backbone, subject_data: Dict[str, dict], subject: str, val_subjects: Optional[List[str]],
                  batch_size: int, device: torch.device):
    """Training and validation windows of a fold, without jitter: 261007
    (val_subjects None) the temporal split of every training subject,
    261013 the training subjects minus the validation subjects and the
    validation subjects."""
    train_subjects = [s for s in sorted(subject_data) if s != subject]
    if val_subjects is None:
        train_set = backbone.SubjectWindowDataset(subject_data, train_subjects, "train", False)
        val_set = backbone.SubjectWindowDataset(subject_data, train_subjects, "val", False)
    else:
        held_out = {str(s) for s in val_subjects}
        fit = [s for s in train_subjects if str(s) not in held_out]
        train_set = backbone.SubjectWindowDataset(subject_data, fit, "train", False)
        val_set = backbone.SubjectWindowDataset(subject_data, [s for s in val_subjects if s in subject_data],
                                                "val", False)
    return (backbone.make_eval_loader(train_set, batch_size, device),
            backbone.make_eval_loader(val_set, batch_size, device))


def loss_and_pcc(backbone, model: torch.nn.Module, loader, device: torch.device) -> Tuple[float, float, int]:
    """Waveform loss (mean over batches, as the validation loss of training)
    and mean per-window Pearson r of output and target, eval mode."""
    criterion = backbone.WaveformLoss()
    model.eval()
    total, batches, pccs = 0.0, 0, []
    with torch.no_grad():
        for x, y, _, _ in loader:
            wave = model(x.to(device))
            loss, _ = criterion(wave, y.to(device))
            total += float(loss.item())
            batches += 1
            outputs, targets = wave.squeeze(1).cpu().numpy(), y.squeeze(1).numpy()
            pccs.extend(backbone.safe_pcc(a, b) for a, b in zip(outputs, targets))
    return total / max(1, batches), float(np.nanmean(pccs)) if pccs else math.nan, len(pccs)


def restored_gap(backbone, model: torch.nn.Module, loaders, device: torch.device) -> dict:
    train_loss, train_pcc, n_train = loss_and_pcc(backbone, model, loaders[0], device)
    val_loss, val_pcc, n_val = loss_and_pcc(backbone, model, loaders[1], device)
    return {
        "Train_Loss_Eval": train_loss, "Val_Loss_Eval": val_loss, "Train_PCC": train_pcc, "Val_PCC": val_pcc,
        "Gap_Loss": val_loss - train_loss,
        "Gap_Loss_Pct": 100.0 * (val_loss - train_loss) / train_loss if train_loss > 0 else math.nan,
        "Gap_PCC": train_pcc - val_pcc, "Train_Windows_Eval": n_train, "Val_Windows_Eval": n_val,
    }


# ---------------------------------------------------------------------------
# Stored models of every fold
# ---------------------------------------------------------------------------

def member_files(backbone, root: Path, subject: str, member: int) -> Dict[str, Path]:
    tag = backbone.safe_tag(subject)
    folder = root / "Members" / f"m{member}"
    return {
        "summary": backbone.member_paths(root, subject, member)[0],
        "curve": folder / "Learning_Curves" / f"{tag}_Learning_Curve.csv",
        "checkpoint": folder / "Models" / f"{tag}_best.pth",
    }


def fold_models(backbone, source: str, results_root: Path, x_roots: Sequence[Tuple[str, Path]], spec,
                dataset_key: str, subject: str) -> Tuple[List[dict], List[dict]]:
    """The stored models of a fold (model type, member, files, summary,
    weight decay, where found) and the models left out with the reason. X is
    looked for in x_roots in that order; an X model found nowhere is
    trainable (Trainable=True, with the weight decay of the fold)."""
    found, missing = [], []
    weight_decay = None
    if source == "261007":
        o_root = backbone.condition_root(results_root, O_VARIANT, spec, dataset_key)
        x_members = list(range(MEMBERS))
    else:
        selection = backbone.stored_selection(results_root, list(backbone.VARIANTS.values()), spec, dataset_key,
                                              subject)
        if selection is None:
            return [], [{"Model_Type": MODEL_O, "Member": "", "Reason": "no 261013 weight decay selection",
                         "Trainable": False}]
        record, variant = selection
        weight_decay = float(record["Selected_Weight_Decay"])
        o_root = backbone.condition_root(results_root, variant.name, spec, dataset_key)
        x_members = [PAIR_MEMBER]
    for member in range(MEMBERS):
        files = member_files(backbone, o_root, subject, member)
        summary = read_json(files["summary"])
        if summary is None or not files["curve"].exists():
            missing.append({"Model_Type": MODEL_O, "Member": member, "Trainable": False,
                            "Reason": "fold summary or learning curve not stored"})
            continue
        found.append({"Model_Type": MODEL_O, "Member": member, "Files": files, "Summary": summary,
                      "Weight_Decay": weight_decay, "Source": source})
    for member in x_members:
        reasons = []
        for name, root in x_roots:
            files = member_files(backbone, backbone.condition_root(root, X_VARIANT, spec, dataset_key), subject, member)
            summary = read_json(files["summary"])
            if summary is None or not files["curve"].exists():
                reasons.append(f"not in {name}")
                continue
            if weight_decay is not None and not str(summary.get("Model", "")).endswith(
                    "_wd{:g}_subjval".format(weight_decay)):
                reasons.append(f"{name}: trained with another weight decay than the selected {weight_decay:g}")
                continue
            found.append({"Model_Type": MODEL_X, "Member": member, "Files": files, "Summary": summary,
                          "Weight_Decay": weight_decay, "Source": name})
            break
        else:
            missing.append({"Model_Type": MODEL_X, "Member": member, "Trainable": True,
                            "Weight_Decay": weight_decay, "Reason": "; ".join(reasons)})
    return found, missing


def x_variant(backbone, source: str, weight_decay: Optional[float]):
    """The network without stem fusion as its script trains it: 261007 its
    no_stem_fusion ablation; 261013 the full model's settings with the
    weight decay selected in the fold (as 261028_Stem_ablation_noise_v1)."""
    if source == "261007":
        return backbone.VARIANTS[X_VARIANT]
    return backbone.Variant(X_VARIANT, "Without stem fusion, weight decay {:g}".format(weight_decay),
                            backbone.ModelOptions(stem_fusion="none", weight_decay=float(weight_decay)),
                            backbone.DEFAULT_EPOCHS)


def train_missing(backbone, source: str, spec, dataset_key: str, subject_data: Dict[str, dict],
                  plan: Dict[str, Tuple[List[dict], List[dict]]], trained_root: Path, args: argparse.Namespace,
                  device: torch.device) -> List[str]:
    """Train the X models of the plan found nowhere, with the training of
    their script (run_member of 261007 / 261013: the fold's training and
    validation split, the seed of the member, the epochs of the fold's O
    model) into trained_root. Returns the folds trained."""
    label = backbone.DATASET_CONFIGS[dataset_key]["label"]
    jobs = [(subject, item) for subject, (_, missing) in plan.items() for item in missing if item["Trainable"]]
    if not jobs:
        return []
    print(f"[{label}] {len(jobs)} model(s) without stem fusion are stored nowhere: training them with the "
          f"{source} settings on the {device.type.upper()} into {trained_root} (reused by the next run)",
          flush=True)
    all_subjects = sorted(subject_data)
    done = []
    for subject, item in jobs:
        if subject not in subject_data:
            print(f"[{label}] {subject}: not in the data read, not trained")
            continue
        o_summary = next((m["Summary"] for m in plan[subject][0]
                          if m["Model_Type"] == MODEL_O and m["Member"] == PAIR_MEMBER), {})
        epochs = int(o_summary.get("Max_Epochs", backbone.DEFAULT_EPOCHS))
        train_args = argparse.Namespace(
            epoch_cap=epochs, resume=True, save_features=False, batch_size=backbone.BATCH_SIZE,
            min_epochs=backbone.MIN_EPOCHS if args.train_min_epochs is None else args.train_min_epochs,
            patience=backbone.PATIENCE if args.train_patience is None else args.train_patience,
        )
        print(f"[{label}] {subject}: training {MODEL_X} m{item['Member']}"
              + (f" (weight decay {item['Weight_Decay']:g})" if item["Weight_Decay"] is not None else "")
              + f", up to {epochs} epochs ...", flush=True)
        try:
            backbone.run_member(train_args, trained_root, x_variant(backbone, source, item["Weight_Decay"]), spec,
                                dataset_key, subject_data, all_subjects, subject, item["Member"], device)
            done.append(subject)
        except Exception as exc:
            print(f"[{label}] {subject}: training failed ({exc})")
        finally:
            backbone.release_memory(device)
    return sorted(set(done))


def collect(backbones: Dict[str, object], roots: Dict[str, Path], dataset_keys: Sequence[str],
            args: argparse.Namespace, device: torch.device):
    """Measures of every model (the missing X models trained first), the
    curves (for the figures), the coverage and the restored-gap note of every
    dataset."""
    rows, coverage, curves = [], [], []
    gap_notes: Dict[str, str] = {}
    x_network = backbones["261007"].VARIANTS[X_VARIANT]
    trained_root = Path(args.output_root) / TRAINED_FOLDER
    for dataset_key in dataset_keys:
        source = DATASET_SOURCES[dataset_key]
        backbone = backbones[source]
        results_root = roots[source]
        spec = next(s for s in backbone.WINDOW_SPECS if s.name == WINDOW)
        label = backbone.DATASET_CONFIGS[dataset_key]["label"]
        if source == "261007":
            listing = backbone.condition_root(results_root, O_VARIANT, spec, dataset_key) / "Fold_Summaries"
            subjects = sorted({p.name[: -len("_m0_summary.json")] for p in listing.glob("*_m0_summary.json")})
            o_network = backbone.VARIANTS[O_VARIANT]
            x_roots = [("261007", results_root), ("trained here", trained_root)]
        else:
            listing = results_root / "WD_Selection" / f"{spec.order:02d}_{spec.name}" / label
            subjects = sorted(p.name[: -len("_selection.json")] for p in listing.glob("*_selection.json"))
            o_network = next(iter(backbone.VARIANTS.values()))
            x_roots = [("261028_Stem_ablation_noise_v1", roots["noise"]), ("trained here", trained_root)]
        if args.subjects:
            subjects = [s for s in subjects if s in set(args.subjects)]
        if not subjects:
            print(f"{label}: no stored folds under {listing}")
            continue
        plan = {s: fold_models(backbone, source, results_root, x_roots, spec, dataset_key, s) for s in subjects}
        to_train = sum(item["Trainable"] for _, missing in plan.values() for item in missing)
        train = bool(to_train) and not args.no_train_missing
        print(f"[{label}] {len(subjects)} folds; models without stem fusion stored nowhere: {to_train}"
              + (" (--no-train-missing: left out)" if to_train and not train else ""), flush=True)

        # Raw data: for the restored-model gap and for training missing models.
        subject_data, gap_note = None, "off (--no-restored-gap)" if args.no_restored_gap else ""
        if not args.no_restored_gap or train:
            free = free_ram_gb()
            if free is not None and free < args.min_free_ram_gb:
                gap_note = (f"not computed: {free:.1f} GB RAM free, below {args.min_free_ram_gb:g} GB "
                            "(--min-free-ram-gb)")
                print(f"[{label}] only {free:.1f} GB RAM free: the raw data are not read (no restored-model gap, "
                      f"no training of missing models; --min-free-ram-gb 0 reads anyway)", flush=True)
            else:
                print(f"[{label}] reading the data with the {source} code"
                      + (f" ({free:.1f} GB RAM free)" if free is not None else "") + " ...", flush=True)
                try:
                    subject_data = backbone.load_dataset(dataset_key, spec)
                except Exception as exc:
                    gap_note = f"not computed: the data could not be read ({exc})"
                    print(f"[{label}] the data could not be read ({exc}); learning-curve measures only")
        if train and subject_data is not None:
            for subject in train_missing(backbone, source, spec, dataset_key, subject_data, plan, trained_root,
                                         args, device):
                plan[subject] = fold_models(backbone, source, results_root, x_roots, spec, dataset_key, subject)
        if args.no_restored_gap:
            subject_data = None
        gap_notes[label] = gap_note or "computed"
        model_o = o_network.options.build().to(device)
        model_x = x_network.options.build().to(device)
        for subject in subjects:
            models, missing = plan[subject]
            for item in missing:
                coverage.append({"Dataset": label, "Subject": subject, "Found": False, "Source": "",
                                 **{k: v for k, v in item.items() if k != "Weight_Decay"}})
            loaders = None
            for item in models:
                curve = read_curve(item["Files"]["curve"])
                summary = item["Summary"]
                coverage.append({"Dataset": label, "Subject": subject, "Found": curve is not None,
                                 "Model_Type": item["Model_Type"], "Member": item["Member"],
                                 "Source": item["Source"], "Trainable": False,
                                 "Reason": "" if curve is not None else "learning curve unreadable"})
                if curve is None:
                    continue
                max_epochs = int(summary.get("Max_Epochs", backbone.DEFAULT_EPOCHS))
                row = {"Dataset": label, "Subject": subject, "Model_Type": item["Model_Type"],
                       "Member": item["Member"], "Source": item["Source"], "Weight_Decay": item["Weight_Decay"],
                       "Model_Key": summary.get("Model", ""), **curve_measures(curve, max_epochs)}
                if subject_data is None:
                    row["Gap_Note"] = gap_notes[label]
                else:
                    if not item["Files"]["checkpoint"].exists():
                        row["Gap_Note"] = "checkpoint not stored"
                    elif subject not in subject_data:
                        row["Gap_Note"] = "subject not in the data read"
                    else:
                        if loaders is None:
                            val_subjects = None
                            if source != "261007":
                                # Stored as "02 09 18 ..." in the fold summary.
                                stored = summary.get("Validation_Subjects")
                                val_subjects = (stored.split() if isinstance(stored, str) else
                                                [str(s) for s in stored] if stored else
                                                backbone.validation_subjects(
                                                    dataset_key, subject,
                                                    [s for s in sorted(subject_data) if s != subject]))
                            loaders = split_loaders(backbone, subject_data, subject, val_subjects,
                                                    args.batch_size, device)
                        model = model_o if item["Model_Type"] == MODEL_O else model_x
                        model.load_state_dict(torch.load(item["Files"]["checkpoint"], map_location=device))
                        row.update(restored_gap(backbone, model, loaders, device))
                rows.append(row)
                curves.append({"Dataset": label, "Subject": subject, "Model_Type": item["Model_Type"],
                               "Member": item["Member"], "Curve": curve})
            n_o = sum(1 for m in models if m["Model_Type"] == MODEL_O)
            n_x = sum(1 for m in models if m["Model_Type"] == MODEL_X)
            print(f"[{label}] {subject}: O {n_o} member(s), X {n_x} member(s)", flush=True)
            del loaders
        del subject_data, model_o, model_x
        backbone.release_memory(device)
    return pd.DataFrame(rows), pd.DataFrame(coverage), curves, gap_notes


def add_pair_measures(models: pd.DataFrame, curves: List[dict]) -> pd.DataFrame:
    """Epochs to the common target and AUC over the common epochs, for every
    O / X pair of the same fold and member."""
    lookup = {(c["Dataset"], c["Subject"], c["Model_Type"], c["Member"]): c["Curve"] for c in curves}
    models = models.copy()
    for column in ("Common_Target", "Epochs_To_Common_Target", "Common_Epochs", "Val_AUC_Common"):
        models[column] = np.nan
    for (dataset, subject, member), part in models.groupby(["Dataset", "Subject", "Member"]):
        if set(part["Model_Type"]) != {MODEL_O, MODEL_X}:
            continue
        values_o, values_x = pair_measures(lookup[(dataset, subject, MODEL_O, member)],
                                           lookup[(dataset, subject, MODEL_X, member)])
        for model_type, values in ((MODEL_O, values_o), (MODEL_X, values_x)):
            index = part.index[part["Model_Type"] == model_type]
            for key, value in values.items():
                models.loc[index, key] = value
    return models


# ---------------------------------------------------------------------------
# Statistics
# ---------------------------------------------------------------------------

def benjamini_hochberg(p_values: Sequence[float]) -> np.ndarray:
    p = np.asarray(p_values, dtype=float)
    out = np.full(p.shape, np.nan)
    finite = np.flatnonzero(np.isfinite(p))
    if not len(finite):
        return out
    order = finite[np.argsort(p[finite])]
    adjusted = p[order] * len(order) / np.arange(1, len(order) + 1)
    out[order] = np.clip(np.minimum.accumulate(adjusted[::-1])[::-1], 0.0, 1.0)
    return out


def wilcoxon(a: np.ndarray, b: np.ndarray) -> Tuple[float, float]:
    """Two-sided Wilcoxon signed-rank test and the matched-pairs rank-biserial
    correlation of a - b (positive: a larger)."""
    difference = np.asarray(a, dtype=float) - np.asarray(b, dtype=float)
    nonzero = difference[difference != 0]
    if not len(nonzero):
        return 1.0, 0.0
    ranks = stats.rankdata(np.abs(nonzero))
    rank_biserial = float((ranks[nonzero > 0].sum() - ranks[nonzero < 0].sum()) / ranks.sum())
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        p = float(stats.wilcoxon(a, b).pvalue)
    return p, rank_biserial


def pitman_morgan(a: np.ndarray, b: np.ndarray) -> Tuple[float, float]:
    """Pitman-Morgan test of equal variances of two paired samples: variance
    ratio a / b and the two-sided p value."""
    a, b = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    n = len(a)
    if n < 4:
        return math.nan, math.nan
    var_a, var_b = a.var(ddof=1), b.var(ddof=1)
    if var_a <= 0 or var_b <= 0:
        return math.nan, math.nan
    ratio = var_a / var_b
    r = float(np.corrcoef(a, b)[0, 1])
    if not np.isfinite(r) or abs(r) >= 1:
        return float(ratio), math.nan
    t = (ratio - 1.0) * math.sqrt(n - 2) / (2.0 * math.sqrt(ratio * (1.0 - r * r)))
    return float(ratio), float(2.0 * stats.t.sf(abs(t), n - 2))


def mean_sd(values) -> str:
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if not len(values):
        return ""
    sd = values.std(ddof=1) if len(values) > 1 else math.nan
    return f"{values.mean():.4g} ± {sd:.3g}"


def median_iqr(values) -> str:
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if not len(values):
        return ""
    q1, q2, q3 = np.percentile(values, [25, 50, 75])
    return f"{q2:.4g} [{q1:.4g}, {q3:.4g}]"


def pair_tables(models: pd.DataFrame) -> pd.DataFrame:
    """O and X side by side per fold: member m0 (every dataset) and the mean
    of members m0-m2 (CapnoBase, where X has three members)."""
    columns = [m[1] for m in MEASURES]
    frames = []
    m0 = models[models["Member"] == PAIR_MEMBER]
    pivot = m0.pivot_table(index=["Dataset", "Subject"], columns="Model_Type", values=columns, aggfunc="first")
    pivot.columns = [f"{measure}__{model}" for measure, model in pivot.columns]
    frames.append(pivot.reset_index().assign(Pairing="m0"))
    complete = models.groupby(["Dataset", "Subject", "Model_Type"])["Member"].nunique().unstack()
    full_folds = complete[(complete.get(MODEL_O) == MEMBERS) & (complete.get(MODEL_X) == MEMBERS)].index \
        if MODEL_X in complete.columns and MODEL_O in complete.columns else []
    if len(full_folds):
        keep = models.set_index(["Dataset", "Subject"]).index.isin(full_folds)
        mean = models[keep].groupby(["Dataset", "Subject", "Model_Type"])[columns].mean().unstack("Model_Type")
        mean.columns = [f"{measure}__{model}" for measure, model in mean.columns]
        frames.append(mean.reset_index().assign(Pairing=f"mean of m0-m{MEMBERS - 1}"))
    pairs = pd.concat(frames, ignore_index=True)
    # Datasets in the order CapnoBase, BIDMC, STEAM2.
    order = {label: i for i, label in enumerate(("CapnoBase", "BIDMC", "STEAM2"))}
    pairs["_order"] = pairs["Dataset"].map(order).fillna(len(order))
    return pairs.sort_values(["_order", "Subject"], kind="stable").drop(columns="_order").reset_index(drop=True)


def test_table(pairs: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (dataset, pairing), part in pairs.groupby(["Dataset", "Pairing"], sort=False):
        family = []
        for group, column, label, unit, better in MEASURES:
            o_col, x_col = f"{column}__{MODEL_O}", f"{column}__{MODEL_X}"
            if o_col not in part or x_col not in part:
                continue
            both = part[[o_col, x_col]].dropna()
            o, x = both[o_col].to_numpy(float), both[x_col].to_numpy(float)
            row = {"Dataset": dataset, "Pairing": pairing, "Group": group, "Measure": column, "Label": label,
                   "Unit": unit, "Better": better or "descriptive", "N_Folds": int(len(both)),
                   "O (mean ± SD)": mean_sd(o), "X (mean ± SD)": mean_sd(x),
                   "O (median [IQR])": median_iqr(o), "X (median [IQR])": median_iqr(x),
                   "O_Median": float(np.median(o)) if len(o) else math.nan,
                   "X_Median": float(np.median(x)) if len(x) else math.nan,
                   "Median_Difference_O_minus_X": float(np.median(o - x)) if len(o) else math.nan}
            if better:
                o_better = (o < x) if better == "lower" else (o > x)
                x_better = (x < o) if better == "lower" else (x > o)
                row.update({"N_O_Better": int(o_better.sum()), "N_X_Better": int(x_better.sum()),
                            "N_Tied": int(len(o) - o_better.sum() - x_better.sum())})
            p, rank_biserial = wilcoxon(o, x) if len(o) >= 3 else (math.nan, math.nan)
            row.update({"Wilcoxon_p": p, "Rank_Biserial_O_minus_X": rank_biserial})
            family.append(row)
        adjusted = benjamini_hochberg([row["Wilcoxon_p"] for row in family])
        for row, q in zip(family, adjusted):
            row["p_BH"] = q
            if row["Better"] != "descriptive" and np.isfinite(q) and q < 0.05:
                o_lower = row["Median_Difference_O_minus_X"] < 0 or (
                    row["Median_Difference_O_minus_X"] == 0 and row["Rank_Biserial_O_minus_X"] < 0)
                row["Favours"] = MODEL_O if o_lower == (row["Better"] == "lower") else MODEL_X
            else:
                row["Favours"] = "" if row["Better"] == "descriptive" else "ns"
            rows.append(row)
    return pd.DataFrame(rows)


def seed_tables(models: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    rows = []
    for (dataset, subject, model_type), part in models.groupby(["Dataset", "Subject", "Model_Type"]):
        if part["Member"].nunique() < MEMBERS:
            continue
        row = {"Dataset": dataset, "Subject": subject, "Model_Type": model_type, "Members": int(len(part))}
        for column in SEED_MEASURES:
            values = part[column].to_numpy(float) if column in part else np.array([])
            values = values[np.isfinite(values)]
            row[f"{column}_Seed_SD"] = float(values.std(ddof=1)) if len(values) >= 2 else math.nan
        best = part["Best_Val_Loss"].to_numpy(float)
        row["Best_Val_Loss_Seed_CV_Pct"] = float(100.0 * best.std(ddof=1) / best.mean()) if len(best) >= 2 else math.nan
        rows.append(row)
    seeds = pd.DataFrame(rows)
    tests = []
    if seeds.empty:
        return seeds, pd.DataFrame()
    columns = [c for c in seeds.columns if c.endswith("_Seed_SD") or c.endswith("_CV_Pct")]
    for dataset, part in seeds.groupby("Dataset", sort=False):
        wide = part.pivot(index="Subject", columns="Model_Type", values=columns)
        family = []
        for column in columns:
            row = {"Dataset": dataset, "Measure": column}
            for model_type in (MODEL_O, MODEL_X):
                values = wide[column][model_type].dropna() if (column, model_type) in wide.columns \
                    else pd.Series(dtype=float)
                row[f"N_{model_type}"] = int(len(values))
                row[f"{model_type} (median [IQR])"] = median_iqr(values)
            if (column, MODEL_O) in wide.columns and (column, MODEL_X) in wide.columns:
                both = wide[column][[MODEL_O, MODEL_X]].dropna()
                if len(both) >= 3:
                    p, rank_biserial = wilcoxon(both[MODEL_O].to_numpy(float), both[MODEL_X].to_numpy(float))
                    row.update({"N_Paired": int(len(both)), "Wilcoxon_p": p, "Rank_Biserial_O_minus_X": rank_biserial,
                                "N_O_Lower": int((both[MODEL_O] < both[MODEL_X]).sum())})
            family.append(row)
        adjusted = benjamini_hochberg([row.get("Wilcoxon_p", math.nan) for row in family])
        for row, q in zip(family, adjusted):
            row["p_BH"] = q
            tests.append(row)
    return seeds, pd.DataFrame(tests)


def fold_variability(pairs: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (dataset, pairing), part in pairs.groupby(["Dataset", "Pairing"], sort=False):
        for column in FOLD_MEASURES:
            o_col, x_col = f"{column}__{MODEL_O}", f"{column}__{MODEL_X}"
            if o_col not in part or x_col not in part:
                continue
            both = part[[o_col, x_col]].dropna()
            row = {"Dataset": dataset, "Pairing": pairing, "Measure": column, "N_Folds": int(len(both))}
            for model_type, col in ((MODEL_O, o_col), (MODEL_X, x_col)):
                values = both[col].to_numpy(float)
                row[f"{model_type}_SD"] = float(values.std(ddof=1)) if len(values) > 1 else math.nan
                row[f"{model_type}_IQR"] = float(np.subtract(*np.percentile(values, [75, 25]))) if len(values) else math.nan
            ratio, p = pitman_morgan(both[o_col].to_numpy(float), both[x_col].to_numpy(float))
            row.update({"Variance_Ratio_O_over_X": ratio, "Pitman_Morgan_p": p})
            rows.append(row)
    return pd.DataFrame(rows)


def verdict_table(tests: pd.DataFrame) -> pd.DataFrame:
    """Per dataset, pairing and measure group: the measures in which O is
    significantly better, in which X is, and those without a difference
    (p after Benjamini-Hochberg < 0.05)."""
    rows = []
    rated = tests[tests["Better"] != "descriptive"]
    for (dataset, pairing, group), part in rated.groupby(["Dataset", "Pairing", "Group"], sort=False):
        o_better = part.loc[part["Favours"] == MODEL_O, "Measure"].tolist()
        x_better = part.loc[part["Favours"] == MODEL_X, "Measure"].tolist()
        rows.append({
            "Dataset": dataset, "Pairing": pairing, "Group": group, "N_Measures": int(len(part)),
            "N_Folds": int(part["N_Folds"].max()),
            "O_Better": ", ".join(o_better), "X_Better": ", ".join(x_better),
            "No_Difference": ", ".join(part.loc[part["Favours"] == "ns", "Measure"]),
            "Verdict": ("O better" if o_better and not x_better else "X better" if x_better and not o_better
                        else "mixed" if o_better and x_better else "no difference"),
        })
    return pd.DataFrame(rows)


def early_stopping_table(models: pd.DataFrame) -> pd.DataFrame:
    m0 = models[models["Member"] == PAIR_MEMBER]
    return (m0.groupby(["Dataset", "Model_Type"], sort=False)
            .agg(N_Folds=("Early_Stopped", "size"), N_Early_Stopped=("Early_Stopped", "sum"),
                 Epochs_Run_Median=("Epochs_Run", "median"), Best_Epoch_Median=("Best_Epoch", "median"))
            .reset_index()
            .assign(Share_Early_Stopped=lambda f: f["N_Early_Stopped"] / f["N_Folds"]))


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------

def style_axis(axis) -> None:
    axis.grid(True, color=GRID_COLOR, linewidth=0.8)
    axis.set_axisbelow(True)
    for side in ("top", "right"):
        axis.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        axis.spines[side].set_color(TEXT_SECONDARY)
    axis.tick_params(colors=TEXT_SECONDARY, labelsize=8)


def curve_figure(path: Path, dataset: str, curves: List[dict]) -> None:
    """Median (IQR) over folds of the validation loss and of the gap
    (validation - training loss), member m0; both models of a fold are
    divided by the mean of their first validation losses, so folds can be
    combined without changing O vs X."""
    pairs = {}
    for item in curves:
        if item["Dataset"] == dataset and item["Member"] == PAIR_MEMBER:
            pairs.setdefault(item["Subject"], {})[item["Model_Type"]] = item["Curve"]
    pairs = {s: p for s, p in pairs.items() if len(p) == 2}
    if not pairs:
        return
    figure, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    for axis, kind, title in ((axes[0], "val", "Validation loss (relative to the first epoch)"),
                              (axes[1], "gap", "Validation - training loss (relative)")):
        style_axis(axis)
        for model_type in (MODEL_X, MODEL_O):
            series = []
            for pair in pairs.values():
                scale = 0.5 * (pair[MODEL_O]["Val_Loss"].iloc[0] + pair[MODEL_X]["Val_Loss"].iloc[0])
                curve = pair[model_type]
                values = curve["Val_Loss"] if kind == "val" else curve["Val_Loss"] - curve["Train_Loss"]
                series.append((values / scale).to_numpy(float))
            length = max(len(s) for s in series)
            matrix = np.full((len(series), length), np.nan)
            for i, s in enumerate(series):
                matrix[i, :len(s)] = s
            running = np.isfinite(matrix).mean(axis=0)
            keep = running >= CURVE_MIN_SHARE
            epochs = np.arange(1, length + 1)[keep]
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                q1, q2, q3 = np.nanpercentile(matrix[:, keep], [25, 50, 75], axis=0)
            axis.fill_between(epochs, q1, q3, color=MODEL_COLORS[model_type], alpha=0.2, linewidth=0)
            axis.plot(epochs, q2, color=MODEL_COLORS[model_type], linewidth=2, label=model_type)
        axis.set_xlabel("Epoch", color=TEXT_PRIMARY, fontsize=9)
        axis.set_title(title, color=TEXT_PRIMARY, fontsize=10)
        if kind == "gap":
            axis.axhline(0.0, color=TEXT_SECONDARY, linewidth=0.8, linestyle="--")
    axes[0].legend(frameon=False, fontsize=9)
    figure.suptitle(f"{dataset}: median and IQR over {len(pairs)} folds (member m0; epochs where at least "
                    f"{CURVE_MIN_SHARE:.0%} of the folds still train)", color=TEXT_PRIMARY, fontsize=10)
    figure.tight_layout()
    figure.savefig(path, dpi=150)
    plt.close(figure)


def box_figure(path: Path, dataset: str, group: str, pairs: pd.DataFrame, tests: pd.DataFrame) -> None:
    measures = [(column, label, unit) for g, column, label, unit, _ in MEASURES
                if g == group and f"{column}__{MODEL_O}" in pairs and f"{column}__{MODEL_X}" in pairs]
    part = pairs[(pairs["Dataset"] == dataset) & (pairs["Pairing"] == "m0")]
    measures = [m for m in measures if part[[f"{m[0]}__{MODEL_O}", f"{m[0]}__{MODEL_X}"]].dropna().shape[0]]
    if not measures:
        return
    columns = min(4, len(measures))
    rows = int(math.ceil(len(measures) / columns))
    figure, axes = plt.subplots(rows, columns, figsize=(3.4 * columns, 3.3 * rows), squeeze=False)
    for axis in axes.ravel()[len(measures):]:
        axis.set_visible(False)
    for axis, (column, label, unit) in zip(axes.ravel(), measures):
        style_axis(axis)
        both = part[[f"{column}__{MODEL_X}", f"{column}__{MODEL_O}"]].dropna()
        x, o = both.iloc[:, 0].to_numpy(float), both.iloc[:, 1].to_numpy(float)
        for a, b in zip(x, o):
            axis.plot([1, 2], [a, b], color=GRID_COLOR, linewidth=0.8, zorder=1)
        box = axis.boxplot([x, o], positions=[1, 2], widths=0.5, patch_artist=True, showfliers=False,
                           medianprops={"color": TEXT_PRIMARY, "linewidth": 1.4})
        for patch, model_type in zip(box["boxes"], (MODEL_X, MODEL_O)):
            patch.set_facecolor(MODEL_COLORS[model_type])
            patch.set_alpha(0.55)
        rng = np.random.default_rng(0)
        for position, values, model_type in ((1, x, MODEL_X), (2, o, MODEL_O)):
            axis.scatter(position + rng.uniform(-0.07, 0.07, len(values)), values, s=9,
                         color=MODEL_COLORS[model_type], edgecolors="white", linewidths=0.4, zorder=3)
        axis.set_xticks([1, 2])
        axis.set_xticklabels(["X", "O"], color=TEXT_PRIMARY, fontsize=9)
        test = tests[(tests["Dataset"] == dataset) & (tests["Pairing"] == "m0") & (tests["Measure"] == column)]
        p_text = ""
        if len(test) and np.isfinite(test["p_BH"].iloc[0]):
            p_text = f"\np(BH) = {test['p_BH'].iloc[0]:.2g}"
        axis.set_title(f"{label}{p_text}", color=TEXT_PRIMARY, fontsize=8)
        if unit:
            axis.set_ylabel(unit, color=TEXT_SECONDARY, fontsize=8)
    figure.suptitle(f"{dataset}: {group} (Stem Fusion X vs O, member m0, n = {len(part)} folds)",
                    color=TEXT_PRIMARY, fontsize=10)
    figure.tight_layout()
    figure.savefig(path, dpi=150)
    plt.close(figure)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def write_excel(path: Path, sheets: Dict[str, pd.DataFrame]) -> None:
    try:
        with pd.ExcelWriter(path, engine="openpyxl") as writer:
            for name, frame in sheets.items():
                frame.to_excel(writer, index=False, sheet_name=name[:31])
    except Exception as exc:
        print(f"Excel export failed ({exc}); the CSV files are complete.")


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--backbone-261007", type=Path, default=BACKBONE_261007,
                        help="Path of 261007_RRWaveNet_Ensemble_NoEvent.py (CapnoBase: data and model).")
    parser.add_argument("--backbone-261013", type=Path, default=BACKBONE_261013,
                        help="Path of 261013_RRWaveNet_Adult_SubjectVal_WD.py (BIDMC, STEAM2: data and model).")
    parser.add_argument("--results-root-261007", type=Path, default=RESULTS_ROOT_261007)
    parser.add_argument("--results-root-261013", type=Path, default=RESULTS_ROOT_261013)
    parser.add_argument("--results-root-noise", type=Path, default=RESULTS_ROOT_NOISE,
                        help="Results of 261028_Stem_ablation_noise_v1 (BIDMC, STEAM2 without stem fusion).")
    parser.add_argument("--output-root", type=Path, default=PROJECT_ROOT)
    parser.add_argument("--datasets", nargs="+", choices=list(DATASET_ORDER), default=list(DATASET_ORDER))
    parser.add_argument("--subjects", nargs="+", default=None, help="Only these folds (quick tests).")
    parser.add_argument("--no-restored-gap", action="store_true",
                        help="Learning curves only (no model reloading; the raw data are read only to train "
                             "missing models).")
    parser.add_argument("--no-train-missing", action="store_true",
                        help="Do not train the models without stem fusion that are stored nowhere (their folds "
                             "are compared without them).")
    parser.add_argument("--train-min-epochs", type=int, default=None,
                        help="Minimum epochs when training a missing model (default: that of 261007 / 261013).")
    parser.add_argument("--train-patience", type=int, default=None,
                        help="Early-stopping patience when training a missing model (default: that of "
                             "261007 / 261013).")
    parser.add_argument("--batch-size", type=int, default=48)
    parser.add_argument("--use-gpu", action="store_true",
                        help="Use the GPU for the restored-model gap (default: CPU, so that a training on the "
                             "GPU, e.g. RespDiff, is not disturbed).")
    parser.add_argument("--gpu", type=int, default=0, help="GPU index with --use-gpu.")
    parser.add_argument("--force-cpu", action="store_true", help="CPU only (the default; kept for older commands).")
    parser.add_argument("--threads", type=int, default=DEFAULT_THREADS,
                        help="CPU threads of this analysis (default %(default)s).")
    parser.add_argument("--min-free-ram-gb", type=float, default=MIN_FREE_RAM_GB,
                        help="Free RAM needed before the raw data of a dataset are read for the restored-model "
                             "gap; with less, that dataset gets the learning-curve measures only (0 = no check).")
    parser.add_argument("--normal-priority", action="store_true",
                        help="Keep normal process priority (default: below normal).")
    parser.add_argument("--capnobase-path", type=Path, default=None)
    parser.add_argument("--bidmc-path", type=Path, default=None)
    parser.add_argument("--steam2-ppg-dir", type=Path, default=None)
    parser.add_argument("--steam2-rsp-dir", type=Path, default=None)
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> None:
    args = parse_args(argv)
    backbones = {
        "261007": load_backbone(Path(args.backbone_261007).resolve(), "rrwavenet_261007_dynamics"),
        "261013": load_backbone(Path(args.backbone_261013).resolve(), "rrwavenet_261013_dynamics"),
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
    priority = "normal" if args.normal_priority else lower_priority()
    roots = {"261007": Path(args.results_root_261007), "261013": Path(args.results_root_261013),
             "noise": Path(args.results_root_noise)}
    # The GPU is asked for only with --use-gpu (CPU by default, next to a
    # training on the GPU).
    device = torch.device("cpu")
    if args.use_gpu and not args.force_cpu:
        if torch.cuda.is_available():
            device = torch.device(f"cuda:{args.gpu}")
        else:
            print("--use-gpu: no CUDA GPU found, the CPU is used")
    dataset_keys = [key for key in DATASET_ORDER if key in args.datasets]
    print(f"Results: CapnoBase {roots['261007']} (261007, O full / X no_stem_fusion, m0-m2)\n"
          f"         BIDMC, STEAM2 {roots['261013']} (261013, O = selected weight decay, m0-m2)\n"
          f"         BIDMC, STEAM2 X {roots['noise']} (261028_Stem_ablation_noise_v1, m0)\n"
          f"Output:  {Path(args.output_root)} (the result folders above are only read)\n"
          f"Device {device}, {args.threads} CPU threads, priority {priority}; restored-model gap: "
          + ("off (--no-restored-gap)" if args.no_restored_gap
             else f"on (raw data read when at least {args.min_free_ram_gb:g} GB RAM is free)"), flush=True)

    models, coverage, curves, gap_notes = collect(backbones, roots, dataset_keys, args, device)
    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    coverage.to_csv(output_root / "coverage.csv", index=False, encoding="utf-8-sig")
    if models.empty:
        print("No stored model found; see coverage.csv.")
        return
    for column in ("Gap_Loss", "Gap_Loss_Pct", "Gap_PCC"):
        if column not in models:
            models[column] = np.nan
    models = add_pair_measures(models, curves)
    pairs = pair_tables(models)
    tests = test_table(pairs)
    seeds, seed_tests = seed_tables(models)
    folds = fold_variability(pairs)
    early = early_stopping_table(models)
    verdict = verdict_table(tests)
    tables = {"verdict": verdict, "tests": tests, "models": models, "pairs": pairs, "seed_variability": seeds,
              "seed_variability_tests": seed_tests, "fold_variability": folds, "early_stopping": early,
              "coverage": coverage}
    for name, frame in tables.items():
        frame.to_csv(output_root / f"{name}.csv", index=False, encoding="utf-8-sig")
    write_excel(output_root / "261030_StemFusion_Training_Dynamics.xlsx", tables)
    figure_root = output_root / "Figures"
    figure_root.mkdir(parents=True, exist_ok=True)
    for dataset in models["Dataset"].unique():
        curve_figure(figure_root / f"{dataset}_learning_curves.png", dataset, curves)
        for group in ("Convergence", "Stability", "Gap"):
            box_figure(figure_root / f"{dataset}_{group.lower()}.png", dataset, group, pairs, tests)

    print("\nStem Fusion O vs X (member m0; median O | median X; p after Benjamini-Hochberg):")
    for dataset, part in tests[tests["Pairing"] == "m0"].groupby("Dataset", sort=False):
        print(f"\n  {dataset} ({part['N_Folds'].max()} folds)")
        for _, row in part.iterrows():
            p = f"{row['p_BH']:.3g}" if np.isfinite(row["p_BH"]) else "-"
            print(f"    {row['Group']:<15s} {row['Measure']:<26s} O {row['O_Median']:<10.4g} X "
                  f"{row['X_Median']:<10.4g} p(BH) {p:<8s} {row['Favours']}")
    print("\nPer group (member m0; better = significant after Benjamini-Hochberg):")
    for _, row in verdict[verdict["Pairing"] == "m0"].iterrows():
        print(f"  {row['Dataset']:<10s} {row['Group']:<12s} {row['Verdict']:<14s}"
              + (f" O better: {row['O_Better']}" if row["O_Better"] else "")
              + (f" X better: {row['X_Better']}" if row["X_Better"] else ""))
    print("\nRestored-model gap per dataset: "
          + "; ".join(f"{dataset} {note}" for dataset, note in gap_notes.items()))
    print(f"\nSaved under: {output_root}")


if __name__ == "__main__":
    main()
