r"""261106_StemFusion_Generalization_FeatureLevel: does the 1x1 stem fusion
make the features generalize better to new subjects? Three feature-level
analyses and the convergence of the training and validation loss, for the
models with (O) and without (X) stem fusion of 261007 and of 261013, each
run analysed on its own so that the two can be compared.

Nothing is trained. The single model m0 of every LOSOCV fold is reloaded
and run (eval mode, no jitter) on the windows of its own fold:
  fit windows   the windows the model was trained on;
  val windows   its validation windows;
  test windows  every kept window of the held-out subject.

Runs and models
---------------
  261007  CapnoBase, BIDMC, STEAM2. O = full, X = no_stem_fusion (both
          stored by 261007, the same seed and training). Validation: the
          last 20% of every training subject's windows (temporal), so the
          validation windows come from the training subjects.
  261013  BIDMC, STEAM2. O = full model of the weight decay 261013 selected
          in the fold (full_wd{wd}); X = the model without stem fusion
          trained by 261028_Stem_ablation_noise_v1 with that weight decay
          and the same validation subjects. Validation: 20% of the training
          subjects held out whole (subject-wise), so the validation windows
          come from new subjects.
Data are read by each run's own script (imported), so the windows are
exactly those of the run. A fold is left out when a model file is missing,
the stored test windows differ from the reloaded data, or (261013) the
validation subjects of the selection differ; skipped_folds.csv lists them.

Stages
------
The 24-channel feature map at Stem (the encoder input: the 1x1 fusion
output for O, the concatenated branch channels for X) and after each
residual encoder block, B1 (d=1), B2 (d=2), B3 (d=4) and B4 (d=8).

1. Feature-level train-test gap (how much better the features describe the
   target on the fit windows than on a new subject)
   * Channel |r|: Pearson r of every channel with the target in every
     window, |r| averaged over channels and windows, on the fit, val and
     test windows;
   * linear readout r: a ridge regression from the 24 channels (and a
     constant) to the target sample by sample, fitted on the fit windows of
     the fold (lambda = 1e-3 x mean diagonal of X'X), applied to the fit,
     val and test windows; Pearson r of readout and target per window,
     averaged;
   * gaps: fit - test (Gap_*_FitTest) and fit - val (Gap_*_FitVal).
   Smaller gap = less overfitting at the feature level.
2. Subject information in the features (how much the features identify
   the person). Per window and stage, the summary vector = time mean and
   time SD of each channel (48 values), standardized over the fit windows:
   * Subject_F_Ratio: between-subject / within-subject variance (trace
     ratio) over the fit subjects;
   * subject probe: a linear classifier (LDA with Ledoit-Wolf shrinkage)
     that predicts the subject from the summary vector, trained on the
     first half of every fit subject's windows and tested on the second
     half; balanced accuracy and the chance-normalized value
     (acc - 1/K) / (1 - 1/K).
   Lower = the features carry less subject-specific information.
3. Train-test feature distribution shift (how far a new subject lies from
   the training features), on the same standardized summary vectors:
   * MMD2_Test: unbiased squared maximum mean discrepancy (RBF kernel,
     median-distance bandwidth) between the test windows and up to 600
     fit windows (fixed random subset);
   * Mahal_Test_per_dim: mean squared Mahalanobis distance of the test
     windows to the fit windows (Ledoit-Wolf covariance), divided by the
     number of dimensions; the same for the val windows (reference).
   Lower = new-subject features closer to the training features.
Loss convergence
   * learning curves (Train_Loss, with dropout and jitter, and Val_Loss of
     every epoch, from the stored learning curve of each model): median and
     IQR over folds per epoch (epochs that at least 25% of the folds
     reached), median best epoch marked;
   * final values of the restored model (eval mode, waveform loss): fit,
     val and test loss and their gaps.

Statistics
----------
Per run, dataset, stage and measure: O and X paired by fold, median and
mean +- SD, the number of folds in which O is better (lower, or higher for
the r values of the test and val windows), two-sided Wilcoxon signed-rank
test, Benjamini-Hochberg over all measures and stages of a run and dataset.

Figures (legends outside the plots)
-----------------------------------
  Fig1_feature_train_test_gap.png     rows: run x dataset; channel |r| gap
                                      and readout r gap (fit - test) per stage
  Fig2_subject_information.png        subject F ratio and probe accuracy
  Fig3_train_test_shift.png           MMD2 and Mahalanobis distance of test
  Fig4_loss_convergence.png           learning curves and final losses
Every stage with FDR < 0.05 is starred; O blue, X red, median and IQR.

Outputs (PROJECT_ROOT = D:\PPG2RSP_RRWaveNet_Inspired\261106_StemFusion_Generalization_FeatureLevel)
  {run}\{dataset}\Folds\{subject}.json    per fold (reused when the run is
                                          repeated: delete to recompute)
  Summary\fold_measures.csv, stage_measures.csv, tests.csv, curves.csv,
          key_results.csv, skipped_folds.csv, input_models.csv,
          261106_StemFusion_Generalization_FeatureLevel_results.xlsx, figures

Run from D:\PPG2RSP_RRWaveNet_Inspired\261106_StemFusion_Generalization_FeatureLevel:
  python 261106_StemFusion_Generalization_FeatureLevel.py
Each run's script is looked for in its results folder, then next to this
script:
  261007  D:\PPG2RSP_RRWaveNet_Inspired\261007_RRWaveNet_NoEvent_v1\261007_RRWaveNet_Ensemble_NoEvent.py
  261013  D:\PPG2RSP_RRWaveNet_Inspired\261013_RRWavNet_STEAM_BIDMC_v1\261013_RRWaveNet_Adult_SubjectVal_WD.py
Models of X for 261013: D:\PPG2RSP_RRWaveNet_Inspired\261028_Stem_ablation_noise_v1
Options: --runs 261007 261013, --datasets, --subjects, --root-261007,
  --root-261013, --root-noise, --check-inputs-only (model files only),
  --summary-only (tables and figures from the stored folds), --force-cpu,
  --gpu, and the data path options.
"""

import argparse
import importlib.util
import json
import sys
import time
import zlib
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

sys.dont_write_bytecode = True

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd
import torch
from scipy import stats
from sklearn.covariance import LedoitWolf
from scipy.spatial.distance import cdist
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.metrics import balanced_accuracy_score

SCRIPT_NAME = "261106_StemFusion_Generalization_FeatureLevel"
SCRIPT_DIR = Path(__file__).resolve().parent
WORKSPACE = Path(r"D:\PPG2RSP_RRWaveNet_Inspired")
PROJECT_ROOT = WORKSPACE / SCRIPT_NAME
RUNS = {
    "261007": {"root": WORKSPACE / "261007_RRWaveNet_NoEvent_v1", "script": "261007_RRWaveNet_Ensemble_NoEvent.py",
               "datasets": ("capnobase", "bidmc", "steam2"), "validation": "temporal (last 20% of each training subject)"},
    "261013": {"root": WORKSPACE / "261013_RRWavNet_STEAM_BIDMC_v1", "script": "261013_RRWaveNet_Adult_SubjectVal_WD.py",
               "datasets": ("bidmc", "steam2"), "validation": "subject-wise (20% of the training subjects)"},
}
NOISE_ROOT = WORKSPACE / "261028_Stem_ablation_noise_v1"
DATASET_ORDER = ("capnobase", "bidmc", "steam2")
LABELS = {"capnobase": "CapnoBase", "bidmc": "BIDMC", "steam2": "STEAM2"}
WINDOW = "w10s_nonoverlap"
MEMBER = 0
MODELS = ("O", "X")
MODEL_LABELS = {"O": "with stem fusion (O)", "X": "without stem fusion (X)"}
COLORS = {"O": "#3a6ea5", "X": "#d1495b"}
STAGES = ("Stem", "B1", "B2", "B3", "B4")
STAGE_LABELS = ("Stem", "B1\n(d=1)", "B2\n(d=2)", "B3\n(d=4)", "B4\n(d=8)")
RIDGE_LAMBDA = 1e-3
MMD_MAX_FIT_WINDOWS = 600
CURVE_MIN_FOLD_SHARE = 0.25

# Stage measures: (column, label, better)
STAGE_MEASURES = (
    ("ChannelAbsR_Fit", "Channel |r|, fit windows", "higher"),
    ("ChannelAbsR_Val", "Channel |r|, val windows", "higher"),
    ("ChannelAbsR_Test", "Channel |r|, test windows", "higher"),
    ("Gap_ChannelAbsR_FitTest", "Channel |r| gap (fit - test)", "lower"),
    ("Gap_ChannelAbsR_FitVal", "Channel |r| gap (fit - val)", "lower"),
    ("Readout_r_Fit", "Linear readout r, fit windows", "higher"),
    ("Readout_r_Val", "Linear readout r, val windows", "higher"),
    ("Readout_r_Test", "Linear readout r, test windows", "higher"),
    ("Gap_Readout_FitTest", "Readout r gap (fit - test)", "lower"),
    ("Gap_Readout_FitVal", "Readout r gap (fit - val)", "lower"),
    ("Subject_F_Ratio", "Subject F ratio (between / within)", "lower"),
    ("Subject_Probe_BalAcc", "Subject probe balanced accuracy", "lower"),
    ("Subject_Probe_Norm", "Subject probe accuracy, chance-normalized", "lower"),
    ("MMD2_Test", "MMD2 test vs fit features", "lower"),
    ("MMD2_Val", "MMD2 val vs fit features", "lower"),
    ("Mahal_Test_per_dim", "Mahalanobis2 / dim of test windows", "lower"),
    ("Mahal_Val_per_dim", "Mahalanobis2 / dim of val windows", "lower"),
)
# Model measures (no stage): (column, label, better)
MODEL_MEASURES = (
    ("Loss_Fit", "Waveform loss, fit windows (eval)", "lower"),
    ("Loss_Val", "Waveform loss, val windows (eval)", "lower"),
    ("Loss_Test", "Waveform loss, test windows (eval)", "lower"),
    ("Gap_Loss_ValFit", "Loss gap (val - fit)", "lower"),
    ("Gap_Loss_TestFit", "Loss gap (test - fit)", "lower"),
    ("PCC_Fit", "Waveform PCC, fit windows", "higher"),
    ("PCC_Test", "Waveform PCC, test windows", "higher"),
    ("Best_Epoch", "Best (restored) epoch", "lower"),
    ("Curve_Best_Val_Loss", "Lowest validation loss (curve)", "lower"),
    ("Curve_Final_Train_Loss", "Last-epoch training loss (curve)", "lower"),
    ("Curve_Final_Val_Loss", "Last-epoch validation loss (curve)", "lower"),
)


# ---------------------------------------------------------------------------
# Runs, scripts and model files
# ---------------------------------------------------------------------------

def load_backbone(run: str, root: Path):
    """Import the run's own script as a module (no training is started)."""
    name = RUNS[run]["script"]
    for folder in (root, SCRIPT_DIR, Path.cwd()):
        path = folder / name
        if path.is_file():
            spec = importlib.util.spec_from_file_location(f"stemgen_{run}", str(path))
            module = importlib.util.module_from_spec(spec)
            sys.modules[spec.name] = module
            spec.loader.exec_module(module)
            return module, path
    raise FileNotFoundError(f"{name} not found in {root}, {SCRIPT_DIR} or {Path.cwd()}")


def read_json(path: Path) -> Optional[dict]:
    try:
        return json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


def stable_seed(*parts: object) -> int:
    return zlib.crc32("|".join(str(p) for p in parts).encode("utf-8")) % (2**31 - 1)


def window_spec(backbone):
    return next(s for s in backbone.WINDOW_SPECS if s.name == WINDOW)


def member_files(backbone, condition: Path, subject: str) -> dict:
    tag = backbone.safe_tag(subject)
    summary, _, prediction = backbone.member_paths(condition, subject, MEMBER)
    member = condition / "Members" / f"m{MEMBER}"
    return {"condition": condition, "summary": summary, "prediction": prediction,
            "checkpoint": member / "Models" / f"{tag}_best.pth",
            "curve": member / "Learning_Curves" / f"{tag}_Learning_Curve.csv"}


def fold_subjects(run: str, backbone, root: Path, dataset_key: str) -> List[str]:
    """Subject tags with a stored fold of O."""
    spec = window_spec(backbone)
    if run == "261007":
        folder = backbone.condition_root(root, "full", spec, dataset_key) / "Members" / f"m{MEMBER}" / "Models"
        return sorted(p.name[:-len("_best.pth")] for p in folder.glob("*_best.pth"))
    folder = root / "WD_Selection" / f"{spec.order:02d}_{spec.name}" / LABELS[dataset_key]
    return sorted(p.name[:-len("_selection.json")] for p in folder.glob("*_selection.json"))


def fold_models(run: str, backbone, root: Path, noise_root: Path, dataset_key: str,
                subject: str) -> Tuple[dict, Optional[List[str]], Optional[float], str]:
    """O and X of a fold (files and model options), the validation subjects
    (261013) and the weight decay; reason is '' when every file is there."""
    spec = window_spec(backbone)
    val_subjects, weight_decay = None, None
    if run == "261007":
        models = {
            "O": {**member_files(backbone, backbone.condition_root(root, "full", spec, dataset_key), subject),
                  "options": backbone.VARIANTS["full"].options},
            "X": {**member_files(backbone, backbone.condition_root(root, "no_stem_fusion", spec, dataset_key), subject),
                  "options": backbone.VARIANTS["no_stem_fusion"].options},
        }
    else:
        record = read_json(backbone.selection_path(root, spec, dataset_key, subject))
        if record is None or "Selected_Weight_Decay" not in record:
            return {}, None, None, "no 261013 weight decay selection (WD_Selection)"
        weight_decay = float(record["Selected_Weight_Decay"])
        val_subjects = [str(s) for s in record.get("Validation_Subjects", [])]
        full = next(iter(backbone.build_variants([weight_decay]).values()))
        models = {
            "O": {**member_files(backbone, backbone.condition_root(root, full.name, spec, dataset_key), subject),
                  "options": full.options},
            "X": {**member_files(backbone, backbone.condition_root(noise_root, "no_stem_fusion", spec, dataset_key),
                                 subject),
                  "options": backbone.ModelOptions(stem_fusion="none", weight_decay=weight_decay)},
        }
    problems = []
    for key, item in models.items():
        missing = [name for name in ("checkpoint", "curve", "summary", "prediction") if not item[name].is_file()]
        if missing:
            problems.append(f"{key}: missing {', '.join(missing)} ({item['condition']})")
            continue
        model_key = str((read_json(item["summary"]) or {}).get("Model", ""))
        item["model_key"] = model_key
        wanted = "_stemfusion_" if key == "O" else "_nostemfusion_"
        if wanted not in f"_{model_key}_":
            problems.append(f"{key}: the stored summary is of model {model_key}")
        if run == "261013" and not model_key.endswith("_wd{:g}_subjval".format(weight_decay)):
            problems.append(f"{key}: model {model_key} not trained with the selected weight decay {weight_decay:g}")
    return models, val_subjects, weight_decay, "; ".join(problems)


# ---------------------------------------------------------------------------
# Features, measures
# ---------------------------------------------------------------------------

def pearson_rows(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Pearson r along the last axis (a and b broadcast)."""
    a = a - a.mean(axis=-1, keepdims=True)
    b = b - b.mean(axis=-1, keepdims=True)
    den = np.sqrt((a * a).sum(-1) * (b * b).sum(-1))
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(den > 1e-12, (a * b).sum(-1) / den, np.nan)


@torch.no_grad()
def feature_pass(model, loader, device, loss_fn, readout: Optional[List[np.ndarray]] = None,
                 accumulate: bool = False) -> dict:
    """One pass over a window set: per window and stage the channel |r|, the
    summary vector (time mean and SD of every channel) and the readout r;
    the waveform loss (mean over batches) and PCC; the ridge sums."""
    model.eval()
    abs_r, summary, readout_r, pcc = [], [], [], []
    xtx = [np.zeros((25, 25)) for _ in STAGES] if accumulate else None
    xty = [np.zeros(25) for _ in STAGES] if accumulate else None
    loss_total, batches = 0.0, 0
    for batch in loader:
        x, y = batch[0].to(device), batch[1]
        target = y.squeeze(1).numpy().astype(np.float64)
        features = model.fused_representation(x)
        stage_maps = [features]
        for block in model.encoder:
            features = block(features)
            stage_maps.append(features)
        wave = model(x)
        loss, _ = loss_fn(wave, y.to(device))
        loss_total += float(loss.item())
        batches += 1
        pcc.append(pearson_rows(wave.squeeze(1).cpu().numpy().astype(np.float64), target))
        batch_abs, batch_summary, batch_readout = [], [], []
        for index, feature in enumerate(stage_maps):
            f = feature.cpu().numpy().astype(np.float64)
            batch_abs.append(np.nanmean(np.abs(pearson_rows(f, target[:, None, :])), axis=1))
            batch_summary.append(np.concatenate([f.mean(-1), f.std(-1)], axis=1))
            if accumulate:
                z = np.concatenate([np.transpose(f, (0, 2, 1)).reshape(-1, f.shape[1]),
                                    np.ones((f.shape[0] * f.shape[2], 1))], axis=1)
                xtx[index] += z.T @ z
                xty[index] += z.T @ target.reshape(-1)
            if readout is not None:
                weights = readout[index]
                prediction = np.einsum("bct,c->bt", f, weights[:-1]) + weights[-1]
                batch_readout.append(pearson_rows(prediction, target))
        abs_r.append(np.stack(batch_abs, axis=1))
        summary.append(np.stack(batch_summary, axis=1))
        if readout is not None:
            readout_r.append(np.stack(batch_readout, axis=1))
    out = {"abs_r": np.concatenate(abs_r), "summary": np.concatenate(summary),
           "loss": loss_total / max(1, batches), "pcc": float(np.nanmean(np.concatenate(pcc)))}
    if readout is not None:
        out["readout_r"] = np.concatenate(readout_r)
    if accumulate:
        out["xtx"], out["xty"] = xtx, xty
    return out


def ridge_weights(xtx: np.ndarray, xty: np.ndarray) -> np.ndarray:
    lam = RIDGE_LAMBDA * float(np.mean(np.diag(xtx)[:-1]))
    penalty = np.eye(xtx.shape[0]) * lam
    penalty[-1, -1] = 0.0  # the constant is not penalized
    return np.linalg.solve(xtx + penalty, xty)


def standardizer(reference: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    mean = reference.mean(axis=0)
    sd = reference.std(axis=0)
    keep = sd > 1e-8
    return mean, np.where(keep, sd, 1.0), keep


def f_ratio(z: np.ndarray, subjects: np.ndarray) -> float:
    grand = z.mean(axis=0)
    between, within = 0.0, 0.0
    for subject in np.unique(subjects):
        part = z[subjects == subject]
        between += len(part) * float(((part.mean(axis=0) - grand) ** 2).sum())
        within += float(((part - part.mean(axis=0)) ** 2).sum())
    groups = len(np.unique(subjects))
    between /= max(1, groups - 1)
    within /= max(1, len(z) - groups)
    return between / within if within > 0 else np.nan


def subject_probe(z: np.ndarray, subjects: np.ndarray, order: np.ndarray) -> Tuple[float, float]:
    """Balanced accuracy of a subject classifier trained on the first half of
    every subject's windows, tested on the second half; and the chance-
    normalized value."""
    train_mask = np.zeros(len(z), dtype=bool)
    for subject in np.unique(subjects):
        idx = np.flatnonzero(subjects == subject)
        idx = idx[np.argsort(order[idx])]
        if len(idx) >= 2:
            train_mask[idx[: (len(idx) + 1) // 2]] = True
    test_mask = ~train_mask & np.isin(subjects, np.unique(subjects[train_mask]))
    classes = np.unique(subjects[train_mask])
    if len(classes) < 2 or not test_mask.any():
        return np.nan, np.nan
    probe = LinearDiscriminantAnalysis(solver="lsqr", shrinkage="auto")
    probe.fit(z[train_mask], subjects[train_mask])
    accuracy = balanced_accuracy_score(subjects[test_mask], probe.predict(z[test_mask]))
    chance = 1.0 / len(classes)
    return float(accuracy), float((accuracy - chance) / (1.0 - chance))


def mmd2_unbiased(a: np.ndarray, b: np.ndarray) -> float:
    if len(a) < 2 or len(b) < 2:
        return np.nan
    pooled = np.concatenate([a, b])
    d2 = cdist(pooled, pooled, "sqeuclidean")
    positive = d2[np.triu_indices(len(pooled), 1)]
    positive = positive[positive > 0]
    if not len(positive):
        return 0.0
    kernel = np.exp(-d2 / (2.0 * float(np.median(positive))))
    n = len(a)
    kaa, kbb, kab = kernel[:n, :n], kernel[n:, n:], kernel[:n, n:]
    m = len(b)
    return float((kaa.sum() - np.trace(kaa)) / (n * (n - 1)) + (kbb.sum() - np.trace(kbb)) / (m * (m - 1))
                 - 2.0 * kab.mean())


def model_measures(model, sets: dict, subject_ids: dict, device, loss_fn, seed: int) -> Tuple[dict, List[dict]]:
    """Every measure of one model on the fit / val / test windows of its fold."""
    loaders = sets
    fit = feature_pass(model, loaders["fit"], device, loss_fn, accumulate=True)
    weights = [ridge_weights(fit["xtx"][i], fit["xty"][i]) for i in range(len(STAGES))]
    fit_readout = feature_pass(model, loaders["fit"], device, loss_fn, readout=weights)
    val = feature_pass(model, loaders["val"], device, loss_fn, readout=weights)
    test = feature_pass(model, loaders["test"], device, loss_fn, readout=weights)
    model_row = {
        "Loss_Fit": fit["loss"], "Loss_Val": val["loss"], "Loss_Test": test["loss"],
        "Gap_Loss_ValFit": val["loss"] - fit["loss"], "Gap_Loss_TestFit": test["loss"] - fit["loss"],
        "PCC_Fit": fit["pcc"], "PCC_Val": val["pcc"], "PCC_Test": test["pcc"],
        "N_Fit_Windows": int(len(fit["abs_r"])), "N_Val_Windows": int(len(val["abs_r"])),
        "N_Test_Windows": int(len(test["abs_r"])),
    }
    fit_subjects, fit_order = subject_ids["fit"]
    rng = np.random.default_rng(seed)
    stage_rows = []
    for index, stage in enumerate(STAGES):
        row = {"Stage": stage}
        for name, part in (("Fit", fit), ("Val", val), ("Test", test)):
            row[f"ChannelAbsR_{name}"] = float(np.nanmean(part["abs_r"][:, index]))
        for name, part in (("Fit", fit_readout), ("Val", val), ("Test", test)):
            row[f"Readout_r_{name}"] = float(np.nanmean(part["readout_r"][:, index]))
        row["Gap_ChannelAbsR_FitTest"] = row["ChannelAbsR_Fit"] - row["ChannelAbsR_Test"]
        row["Gap_ChannelAbsR_FitVal"] = row["ChannelAbsR_Fit"] - row["ChannelAbsR_Val"]
        row["Gap_Readout_FitTest"] = row["Readout_r_Fit"] - row["Readout_r_Test"]
        row["Gap_Readout_FitVal"] = row["Readout_r_Fit"] - row["Readout_r_Val"]
        mean, sd, keep = standardizer(fit["summary"][:, index])
        z_fit = ((fit["summary"][:, index] - mean) / sd)[:, keep]
        z_val = ((val["summary"][:, index] - mean) / sd)[:, keep]
        z_test = ((test["summary"][:, index] - mean) / sd)[:, keep]
        row["Subject_F_Ratio"] = f_ratio(z_fit, fit_subjects)
        row["Subject_Probe_BalAcc"], row["Subject_Probe_Norm"] = subject_probe(z_fit, fit_subjects, fit_order)
        row["N_Fit_Subjects"] = int(len(np.unique(fit_subjects)))
        reference = z_fit if len(z_fit) <= MMD_MAX_FIT_WINDOWS else z_fit[
            np.sort(rng.choice(len(z_fit), MMD_MAX_FIT_WINDOWS, replace=False))]
        row["MMD2_Test"] = mmd2_unbiased(z_test, reference)
        row["MMD2_Val"] = mmd2_unbiased(z_val, reference)
        covariance = LedoitWolf().fit(z_fit)
        row["Mahal_Test_per_dim"] = float(np.mean(covariance.mahalanobis(z_test)) / z_fit.shape[1])
        row["Mahal_Val_per_dim"] = float(np.mean(covariance.mahalanobis(z_val)) / z_fit.shape[1])
        row["Feature_Dims"] = int(z_fit.shape[1])
        stage_rows.append(row)
    return model_row, stage_rows


def curve_summary(curve: pd.DataFrame) -> dict:
    epochs = pd.to_numeric(curve["Epoch"], errors="coerce").to_numpy(float)
    train = pd.to_numeric(curve["Train_Loss"], errors="coerce").to_numpy(float)
    val = pd.to_numeric(curve["Val_Loss"], errors="coerce").to_numpy(float)
    best = int(np.nanargmin(val))
    return {"Best_Epoch": int(epochs[best]), "Epochs_Run": int(len(val)), "Curve_Best_Val_Loss": float(val[best]),
            "Curve_Train_Loss_At_Best": float(train[best]), "Curve_Final_Train_Loss": float(train[-1]),
            "Curve_Final_Val_Loss": float(val[-1]),
            "Curve": {"Epoch": epochs.tolist(), "Train_Loss": train.tolist(), "Val_Loss": val.tolist()}}


def run_fold(run: str, backbone, models: dict, data: dict, all_subjects: Sequence[str], subject: str,
             val_subjects: Optional[List[str]], device, batch_size: int) -> dict:
    train_subjects = [s for s in all_subjects if s != subject]
    if run == "261007":
        fit_set = backbone.SubjectWindowDataset(data, train_subjects, "train", False)
        val_set = backbone.SubjectWindowDataset(data, train_subjects, "val", False)
    else:
        held_out = set(val_subjects)
        fit_set = backbone.SubjectWindowDataset(data, [s for s in train_subjects if s not in held_out], "train", False)
        val_set = backbone.SubjectWindowDataset(data, list(val_subjects), "val", False)
    test_set = backbone.SubjectWindowDataset(data, [subject], "test", False)
    loaders = {name: backbone.make_eval_loader(dataset, batch_size, device)
               for name, dataset in (("fit", fit_set), ("val", val_set), ("test", test_set))}
    fit_subjects = np.asarray([str(s) for s, _ in fit_set.samples])
    fit_order = np.asarray([int(i) for _, i in fit_set.samples])
    loss_fn = backbone.WaveformLoss()
    result = {"Models": {}}
    for key in MODELS:
        item = models[key]
        model = item["options"].build().to(device)
        model.load_state_dict(torch.load(item["checkpoint"], map_location=device))
        model_row, stage_rows = model_measures(model, loaders, {"fit": (fit_subjects, fit_order)}, device, loss_fn,
                                               stable_seed(run, subject, key))
        curve = pd.read_csv(item["curve"], encoding="utf-8-sig")
        result["Models"][key] = {"Model_Key": item.get("model_key", ""), "Checkpoint": str(item["checkpoint"]),
                                 **model_row, **curve_summary(curve), "Stages": stage_rows}
        del model
    return result


# ---------------------------------------------------------------------------
# Collection
# ---------------------------------------------------------------------------

def collect(args: argparse.Namespace, out_root: Path, device) -> Tuple[List[dict], List[dict]]:
    skipped, inputs = [], []
    for run in [r for r in RUNS if r in args.runs]:
        root = Path(getattr(args, f"root_{run}") or RUNS[run]["root"]).resolve()
        noise_root = Path(args.root_noise or NOISE_ROOT).resolve()
        backbone, script = load_backbone(run, root)
        for dataset_key, key, option in (("capnobase", "data_path", "capnobase_path"), ("bidmc", "data_path", "bidmc_path"),
                                         ("steam2", "ppg_dir", "steam2_ppg_dir"), ("steam2", "rsp_dir", "steam2_rsp_dir")):
            if getattr(args, option):
                backbone.DATASET_CONFIGS[dataset_key][key] = str(Path(getattr(args, option)).resolve())
        spec = window_spec(backbone)
        print(f"\n=== {run}: {root}\n    script {script}", flush=True)
        for dataset_key in [k for k in RUNS[run]["datasets"] if k in args.datasets]:
            label = LABELS[dataset_key]
            tags = fold_subjects(run, backbone, root, dataset_key)
            if args.subjects:
                tags = [t for t in tags if t in {backbone.safe_tag(s) for s in args.subjects}]
            fold_dir = out_root / run / label / "Folds"
            plans = []
            for tag in tags:
                models, val_subjects, weight_decay, reason = fold_models(run, backbone, root, noise_root, dataset_key, tag)
                for key, item in models.items():
                    inputs.append({"Run": run, "Dataset": label, "Subject": tag, "Model": key,
                                   **{k: str(item[k]) for k in ("checkpoint", "curve", "summary", "prediction")},
                                   "Model_Key": item.get("model_key", ""), "Weight_Decay": weight_decay})
                if reason:
                    skipped.append({"Run": run, "Dataset": label, "Subject": tag, "Reason": reason})
                    continue
                plans.append((tag, models, val_subjects, weight_decay))
            print(f"[{run}] {label}: {len(plans)} of {len(tags)} folds with O and X stored", flush=True)
            if args.check_inputs_only or not plans:
                continue
            todo = [p for p in plans if not (fold_dir / f"{p[0]}.json").is_file()]
            if not todo:
                print(f"[{run}] {label}: every fold already analysed", flush=True)
                continue
            started = time.perf_counter()
            data = backbone.load_dataset(dataset_key, spec)
            by_tag = {backbone.safe_tag(s): s for s in data}
            all_subjects = sorted(data)
            print(f"[{run}] {label}: data read ({time.perf_counter() - started:.0f} s)", flush=True)
            fold_dir.mkdir(parents=True, exist_ok=True)
            for number, (tag, models, val_subjects, weight_decay) in enumerate(todo, start=1):
                subject = by_tag.get(tag)
                if subject is None:
                    skipped.append({"Run": run, "Dataset": label, "Subject": tag, "Reason": "raw subject unavailable"})
                    continue
                if run == "261013":
                    train = [s for s in all_subjects if s != subject]
                    expected = [str(s) for s in backbone.validation_subjects(dataset_key, subject, train)]
                    if sorted(expected) != sorted(val_subjects):
                        skipped.append({"Run": run, "Dataset": label, "Subject": tag,
                                        "Reason": "validation subjects of the selection differ from the data"})
                        continue
                metadata = backbone.test_metadata(data, subject)
                problem = ""
                for key in MODELS:
                    with np.load(models[key]["prediction"], allow_pickle=False) as stored:
                        if not (np.array_equal(stored["segment_ids"], [m["segment_id"] for m in metadata])
                                and np.array_equal(stored["window_starts"], [m["start"] for m in metadata])):
                            problem = f"{key}: stored test windows differ from the reloaded data"
                if problem:
                    skipped.append({"Run": run, "Dataset": label, "Subject": tag, "Reason": problem})
                    continue
                fold_started = time.perf_counter()
                try:
                    result = run_fold(run, backbone, models, data, all_subjects, subject, val_subjects, device,
                                      args.batch_size)
                except Exception as exc:
                    skipped.append({"Run": run, "Dataset": label, "Subject": tag,
                                    "Reason": f"{type(exc).__name__}: {exc}"})
                    print(f"[skip] {run} {label} {tag}: {exc}", flush=True)
                    continue
                result.update({"Run": run, "Dataset": label, "Subject": str(subject), "Weight_Decay": weight_decay,
                               "Validation": RUNS[run]["validation"],
                               "Val_Subjects": val_subjects})
                (fold_dir / f"{tag}.json").write_text(json.dumps(result), encoding="utf-8")
                print(f"[{number}/{len(todo)}] {run} {label} {tag}: O and X analysed "
                      f"({time.perf_counter() - fold_started:.1f} s)", flush=True)
            del data
            backbone.release_memory(device)
    return skipped, inputs


# ---------------------------------------------------------------------------
# Tables, statistics, figures
# ---------------------------------------------------------------------------

def read_folds(out_root: Path) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    model_rows, stage_rows, curve_rows = [], [], []
    for path in sorted(out_root.glob("*/*/Folds/*.json")):
        record = read_json(path)
        if not record:
            continue
        base = {"Run": str(record["Run"]), "Dataset": record["Dataset"], "Subject": str(record["Subject"])}
        for key, item in record["Models"].items():
            model_rows.append({**base, "Model": key, **{k: v for k, v in item.items() if k not in ("Stages", "Curve")}})
            for row in item["Stages"]:
                stage_rows.append({**base, "Model": key, **row})
            curve = item["Curve"]
            for epoch, train, val in zip(curve["Epoch"], curve["Train_Loss"], curve["Val_Loss"]):
                curve_rows.append({**base, "Model": key, "Epoch": int(epoch), "Train_Loss": train, "Val_Loss": val})
    return pd.DataFrame(model_rows), pd.DataFrame(stage_rows), pd.DataFrame(curve_rows)


def bh(p: Sequence[float]) -> np.ndarray:
    p = np.asarray(p, float)
    out = np.full(len(p), np.nan)
    ok = np.isfinite(p)
    if ok.any():
        q = p[ok]
        order = np.argsort(q)
        ranked = q[order] * len(q) / (np.arange(len(q)) + 1)
        ranked = np.minimum.accumulate(ranked[::-1])[::-1]
        values = np.empty(len(q))
        values[order] = np.clip(ranked, 0, 1)
        out[ok] = values
    return out


def stars(p: float) -> str:
    for threshold, mark in ((1e-4, "****"), (1e-3, "***"), (1e-2, "**"), (0.05, "*")):
        if np.isfinite(p) and p < threshold:
            return mark
    return "n.s."


def paired_test(o: np.ndarray, x: np.ndarray, better: str) -> dict:
    keep = np.isfinite(o) & np.isfinite(x)
    o, x = o[keep], x[keep]
    if len(o) < 2:
        return {"N_Folds": int(len(o)), "P": np.nan}
    diff = o - x
    p = 1.0 if np.allclose(diff, 0) else float(stats.wilcoxon(o, x).pvalue)
    o_better = int(np.sum(diff < 0)) if better == "lower" else int(np.sum(diff > 0))
    return {"N_Folds": int(len(o)), "O_Median": float(np.median(o)), "X_Median": float(np.median(x)),
            "O_Mean": float(np.mean(o)), "O_SD": float(np.std(o, ddof=1)), "X_Mean": float(np.mean(x)),
            "X_SD": float(np.std(x, ddof=1)), "Median_Diff_O_minus_X": float(np.median(diff)),
            "N_O_Better": o_better, "N_X_Better": int(len(o) - o_better - int(np.sum(diff == 0))), "P": p}


def run_tests(models: pd.DataFrame, stages: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (run, dataset), part in stages.groupby(["Run", "Dataset"], sort=False):
        group_rows = []
        for stage in STAGES:
            sub = part[part.Stage == stage].pivot_table(index="Subject", columns="Model", aggfunc="first")
            for column, label, better in STAGE_MEASURES:
                if (column, "O") not in sub.columns or (column, "X") not in sub.columns:
                    continue
                result = paired_test(sub[(column, "O")].to_numpy(float), sub[(column, "X")].to_numpy(float), better)
                group_rows.append({"Run": run, "Dataset": dataset, "Stage": stage, "Measure": column,
                                   "Label": label, "Better": better, **result})
        mpart = models[(models.Run == run) & (models.Dataset == dataset)].pivot_table(
            index="Subject", columns="Model", aggfunc="first")
        for column, label, better in MODEL_MEASURES:
            if (column, "O") not in mpart.columns:
                continue
            result = paired_test(mpart[(column, "O")].to_numpy(float), mpart[(column, "X")].to_numpy(float), better)
            group_rows.append({"Run": run, "Dataset": dataset, "Stage": "model", "Measure": column, "Label": label,
                               "Better": better, **result})
        fdr = bh([row["P"] for row in group_rows])
        for row, q in zip(group_rows, fdr):
            row["FDR"] = q
            row["Stars"] = stars(q)
            if np.isfinite(q) and q < 0.05:
                o_wins = row["N_O_Better"] > row["N_X_Better"]
                row["Favors"] = "O (stem fusion)" if o_wins else "X (no stem fusion)"
            else:
                row["Favors"] = "n.s."
        rows += group_rows
    return pd.DataFrame(rows)


def panel_keys(stages: pd.DataFrame) -> List[Tuple[str, str]]:
    keys = []
    for run in RUNS:
        for dataset_key in RUNS[run]["datasets"]:
            label = LABELS[dataset_key]
            if ((stages.Run == run) & (stages.Dataset == label)).any():
                keys.append((run, label))
    return keys


def figure_stage_measures(path: Path, stages: pd.DataFrame, tests: pd.DataFrame, columns: Sequence[Tuple[str, str]],
                          title: str) -> None:
    keys = panel_keys(stages)
    if not keys:
        return
    fig, axes = plt.subplots(len(keys), len(columns), figsize=(5.4 * len(columns), 2.9 * len(keys) + 0.9),
                             squeeze=False)
    x = np.arange(len(STAGES))
    for r, (run, dataset) in enumerate(keys):
        part = stages[(stages.Run == run) & (stages.Dataset == dataset)]
        for c, (column, label) in enumerate(columns):
            ax = axes[r][c]
            for key, offset in (("O", -0.06), ("X", 0.06)):
                values = [part[(part.Model == key) & (part.Stage == s)][column].to_numpy(float) for s in STAGES]
                med = np.array([np.nanmedian(v) if len(v) else np.nan for v in values])
                q1 = np.array([np.nanpercentile(v, 25) if len(v) else np.nan for v in values])
                q3 = np.array([np.nanpercentile(v, 75) if len(v) else np.nan for v in values])
                ax.fill_between(x + offset, q1, q3, color=COLORS[key], alpha=0.15, linewidth=0)
                ax.plot(x + offset, med, "o-", color=COLORS[key], linewidth=1.6, markersize=4)
            top = ax.get_ylim()[1]
            for i, stage in enumerate(STAGES):
                hit = tests[(tests.Run == run) & (tests.Dataset == dataset) & (tests.Stage == stage)
                            & (tests.Measure == column)]
                if len(hit) and hit.iloc[0]["Stars"] != "n.s.":
                    favors = "O" if hit.iloc[0]["Favors"].startswith("O") else "X"
                    ax.text(i, top, hit.iloc[0]["Stars"], ha="center", va="bottom", fontsize=10, color=COLORS[favors])
            ax.set_xticks(x)
            ax.set_xticklabels(STAGE_LABELS, fontsize=8)
            ax.grid(axis="y", color="#e6e6e6")
            ax.set_axisbelow(True)
            if r == 0:
                ax.set_title(label, fontsize=10, pad=14)
            if c == 0:
                n = part[part.Model == "O"].Subject.nunique()
                ax.set_ylabel(f"{run} {dataset}\n(n = {n})", fontsize=9)
    handles = [Line2D([0], [0], color=COLORS[k], marker="o", linewidth=1.6, label=MODEL_LABELS[k]) for k in MODELS]
    fig.legend(handles=handles + [Line2D([0], [0], linestyle="none", label="median and IQR over folds; stars: FDR "
                                         "< 0.05, colored by the better model")],
               loc="upper center", bbox_to_anchor=(0.5, 1.0), ncol=3, frameon=False, fontsize=9)
    fig.suptitle(title, y=1.0 + 0.35 / (2.9 * len(keys) + 0.9), fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 1 - 0.55 / (2.9 * len(keys) + 0.9)))
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def figure_loss(path: Path, models: pd.DataFrame, curves: pd.DataFrame, tests: pd.DataFrame) -> None:
    keys = panel_keys(models) if not models.empty else []
    if not keys:
        return
    fig, axes = plt.subplots(len(keys), 2, figsize=(13.0, 3.1 * len(keys) + 0.9), squeeze=False,
                             gridspec_kw={"width_ratios": [1.6, 1.0]})
    for r, (run, dataset) in enumerate(keys):
        ax = axes[r][0]
        part = curves[(curves.Run == run) & (curves.Dataset == dataset)]
        for key in MODELS:
            sub = part[part.Model == key]
            folds = sub.Subject.nunique()
            if not folds:
                continue
            grouped = sub.groupby("Epoch")
            count = grouped.size()
            epochs = count.index[count >= max(1, int(np.ceil(CURVE_MIN_FOLD_SHARE * folds)))].to_numpy()
            for column, style in (("Train_Loss", "--"), ("Val_Loss", "-")):
                values = grouped[column]
                med = values.median().loc[epochs]
                ax.fill_between(epochs, values.quantile(0.25).loc[epochs], values.quantile(0.75).loc[epochs],
                                color=COLORS[key], alpha=0.10, linewidth=0)
                ax.plot(epochs, med, style, color=COLORS[key], linewidth=1.5)
            best = models[(models.Run == run) & (models.Dataset == dataset) & (models.Model == key)].Best_Epoch
            if len(best):
                ax.axvline(float(np.median(best)), color=COLORS[key], linestyle=":", linewidth=1.1)
        ax.set_xlabel("Epoch", fontsize=9)
        ax.set_ylabel(f"{run} {dataset}\nwaveform loss", fontsize=9)
        ax.grid(color="#eeeeee")
        if r == 0:
            ax.set_title("Learning curves: median and IQR over folds\n(train: dashed, validation: solid; dotted: "
                         "median best epoch)", fontsize=10)
        ax = axes[r][1]
        mpart = models[(models.Run == run) & (models.Dataset == dataset)]
        groups = (("Loss_Fit", "fit"), ("Loss_Val", "val"), ("Loss_Test", "test"))
        tops, bottoms, marks = [], [], []
        for g, (column, name) in enumerate(groups):
            data = [mpart[mpart.Model == key][column].to_numpy(float) for key in MODELS]
            positions = [g * 3 + 0.65, g * 3 + 1.35]
            box = ax.boxplot(data, positions=positions, widths=0.55, patch_artist=True, showfliers=False,
                             medianprops={"color": "#111111"})
            for patch, key in zip(box["boxes"], MODELS):
                patch.set_facecolor(COLORS[key])
                patch.set_alpha(0.55)
            whiskers = [line.get_ydata() for line in box["whiskers"]]
            tops.append(max(float(np.max(w)) for w in whiskers))
            bottoms.append(min(float(np.min(w)) for w in whiskers))
            hit = tests[(tests.Run == run) & (tests.Dataset == dataset) & (tests.Measure == column)]
            marks.append(hit.iloc[0]["Stars"] if len(hit) else "")
        span = max(max(tops) - min(bottoms), 1e-6)
        for g, (top, mark) in enumerate(zip(tops, marks)):
            ax.text(g * 3 + 1.0, top + 0.03 * span, mark, ha="center", va="bottom", fontsize=9)
        ax.set_ylim(min(bottoms) - 0.06 * span, max(tops) + 0.18 * span)
        ax.set_xticks([g * 3 + 1.0 for g in range(len(groups))])
        ax.set_xticklabels([name for _, name in groups])
        ax.grid(axis="y", color="#eeeeee")
        if r == 0:
            ax.set_title("Restored model, eval mode (O vs X; FDR stars)", fontsize=10)
    handles = [Line2D([0], [0], color=COLORS[k], linewidth=2.0, label=MODEL_LABELS[k]) for k in MODELS]
    handles += [Line2D([0], [0], color="#555555", linestyle="--", label="training loss"),
                Line2D([0], [0], color="#555555", linestyle="-", label="validation loss")]
    fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, 1.0), ncol=4, frameon=False, fontsize=9)
    fig.tight_layout(rect=(0, 0, 1, 1 - 0.5 / (3.1 * len(keys) + 0.9)))
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def key_results(tests: pd.DataFrame) -> pd.DataFrame:
    wanted = ["Gap_ChannelAbsR_FitTest", "Gap_Readout_FitTest", "Readout_r_Test", "Subject_F_Ratio",
              "Subject_Probe_Norm", "MMD2_Test", "Mahal_Test_per_dim"]
    part = tests[tests.Measure.isin(wanted) & tests.Stage.isin(["Stem", "B4"])]
    model_part = tests[tests.Stage.eq("model") & tests.Measure.isin(["Loss_Fit", "Loss_Val", "Loss_Test",
                                                                      "Gap_Loss_ValFit", "Gap_Loss_TestFit"])]
    columns = ["Run", "Dataset", "Stage", "Measure", "Better", "N_Folds", "O_Median", "X_Median", "N_O_Better",
               "N_X_Better", "P", "FDR", "Stars", "Favors"]
    return pd.concat([part, model_part])[columns]


def summarize(out_root: Path, skipped: Sequence[dict], inputs: Sequence[dict]) -> None:
    summary = out_root / "Summary"
    summary.mkdir(parents=True, exist_ok=True)
    if skipped:
        pd.DataFrame(skipped).to_csv(summary / "skipped_folds.csv", index=False, encoding="utf-8-sig")
    if inputs:
        pd.DataFrame(inputs).to_csv(summary / "input_models.csv", index=False, encoding="utf-8-sig")
    models, stages, curves = read_folds(out_root)
    if models.empty:
        print("No analysed fold found; nothing to summarize.")
        return
    tests = run_tests(models, stages)
    key = key_results(tests)
    models.to_csv(summary / "fold_measures.csv", index=False, encoding="utf-8-sig")
    stages.to_csv(summary / "stage_measures.csv", index=False, encoding="utf-8-sig")
    curves.to_csv(summary / "curves.csv", index=False, encoding="utf-8-sig")
    tests.to_csv(summary / "tests.csv", index=False, encoding="utf-8-sig")
    key.to_csv(summary / "key_results.csv", index=False, encoding="utf-8-sig")
    notes = pd.DataFrame({"Note": [
        "O = with stem fusion, X = without; single model m0 of every fold, restored best epoch, eval mode.",
        "261007: validation = last 20% of each training subject's windows; 261013: subject-wise validation.",
        "Fit windows = the windows the model was trained on; test = every kept window of the held-out subject.",
        "Better: the direction that means less overfitting / better generalization for the measure.",
        "FDR: Benjamini-Hochberg over all stages and measures of a run and dataset.",
        "Learning-curve training loss is computed in training mode (dropout, jitter); Loss_Fit is in eval mode.",
    ]})
    try:
        with pd.ExcelWriter(summary / f"{SCRIPT_NAME}_results.xlsx", engine="openpyxl") as writer:
            key.to_excel(writer, sheet_name="Key_Results", index=False)
            tests.to_excel(writer, sheet_name="Tests", index=False)
            models.to_excel(writer, sheet_name="Fold_Measures", index=False)
            stages.to_excel(writer, sheet_name="Stage_Measures", index=False)
            notes.to_excel(writer, sheet_name="Notes", index=False)
            if skipped:
                pd.DataFrame(skipped).to_excel(writer, sheet_name="Skipped", index=False)
    except Exception as exc:
        print(f"Workbook not written ({exc}); the CSV files hold the same tables")
    figure_stage_measures(summary / "Fig1_feature_train_test_gap.png", stages, tests,
                          [("Gap_ChannelAbsR_FitTest", "Channel |r| gap (fit - test), lower = less overfit"),
                           ("Gap_Readout_FitTest", "Linear readout r gap (fit - test), lower = less overfit"),
                           ("Readout_r_Test", "Linear readout r on the test subject, higher = better")],
                          "1. Feature-level train-test gap")
    figure_stage_measures(summary / "Fig2_subject_information.png", stages, tests,
                          [("Subject_F_Ratio", "Subject F ratio (between / within), lower = less subject info"),
                           ("Subject_Probe_Norm", "Subject probe accuracy (chance-normalized), lower = less")],
                          "2. Subject information in the features")
    figure_stage_measures(summary / "Fig3_train_test_shift.png", stages, tests,
                          [("MMD2_Test", "MMD2, test vs fit features, lower = closer"),
                           ("Mahal_Test_per_dim", "Mahalanobis2 / dim of test windows, lower = closer")],
                          "3. Train-test feature distribution shift")
    figure_loss(summary / "Fig4_loss_convergence.png", models, curves, tests)
    print("\nKey results (O = with stem fusion, X = without; FDR within run and dataset):")
    shown = key.copy()
    for column in ("O_Median", "X_Median", "P", "FDR"):
        shown[column] = shown[column].map(lambda v: f"{v:.3g}")
    print(shown.to_string(index=False))
    print(f"\nSummary: {summary}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=SCRIPT_NAME)
    parser.add_argument("--runs", nargs="+", choices=list(RUNS), default=list(RUNS))
    parser.add_argument("--datasets", nargs="+", choices=list(DATASET_ORDER), default=list(DATASET_ORDER))
    parser.add_argument("--subjects", nargs="+", default=None)
    parser.add_argument("--root-261007", dest="root_261007", default=None)
    parser.add_argument("--root-261013", dest="root_261013", default=None)
    parser.add_argument("--root-noise", dest="root_noise", default=None,
                        help="results of 261028_Stem_ablation_noise_v1 (X models of 261013)")
    parser.add_argument("--output", type=Path, default=PROJECT_ROOT)
    parser.add_argument("--check-inputs-only", action="store_true")
    parser.add_argument("--summary-only", action="store_true")
    parser.add_argument("--batch-size", type=int, default=48)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--force-cpu", action="store_true")
    parser.add_argument("--threads", type=int, default=None)
    for option in ("capnobase-path", "bidmc-path", "steam2-ppg-dir", "steam2-rsp-dir"):
        parser.add_argument(f"--{option}", type=Path, default=None)
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> None:
    args = parse_args(argv)
    out_root = Path(args.output).expanduser().resolve()
    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() and not args.force_cpu else "cpu")
    if args.threads:
        torch.set_num_threads(int(args.threads))
    print(f"Output: {out_root}\nDevice: {device}", flush=True)
    skipped, inputs = ([], []) if args.summary_only else collect(args, out_root, device)
    if args.check_inputs_only:
        summary = out_root / "Summary"
        summary.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(inputs).to_csv(summary / "input_models.csv", index=False, encoding="utf-8-sig")
        if skipped:
            pd.DataFrame(skipped).to_csv(summary / "skipped_folds.csv", index=False, encoding="utf-8-sig")
        print(f"Input check complete (no data read): {summary / 'input_models.csv'}")
        return
    summarize(out_root, skipped, inputs)


if __name__ == "__main__":
    main()
