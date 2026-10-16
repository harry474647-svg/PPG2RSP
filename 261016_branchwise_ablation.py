r"""261016_branchwise_ablation: branch-wise ablation of the multi-scale stem
(the first module) of the full model; 10 s non-overlapping input windows, a
single model per fold (no ensemble), LOSOCV on every dataset. CapnoBase is
trained as in 261007_RRWaveNet_Ensemble_NoEvent, BIDMC and STEAM2 with the
settings tuned for them in 261013_RRWaveNet_Adult_SubjectVal_WD (see
"Training settings per dataset").

The stem has three parallel branches, Conv1d with kernels of 32, 64 and 128
samples (0.5, 1 and 2 s at 64 Hz), 8 filters each, GroupNorm and GELU; their
outputs are concatenated and fused by the 1x1 stem fusion (+ GroupNorm) into
the 24 encoder channels. Every combination of the branches except all three
(the full model) and none is trained:

  model (name in the results)   removed          stem channels   folder
  k=64 + k=128                  k=32             16              no_k32
  k=32 + k=128                  k=64             16              no_k64
  k=32 + k=64                   k=128            16              no_k128
  only_k=32                     k=64, k=128       8              only_k32
  only_k=64                     k=32, k=128       8              only_k64
  only_k=128                    k=32, k=64        8              only_k128
Every result table, figure and log names a model by its kept branches as
above (the full model: k=32 + k=64 + k=128); the folder name is also the
value of --variants.

A removed branch is left out of the network; the remaining branches are
unchanged and the 1x1 stem fusion maps their 8 or 16 channels to the same 24
encoder channels. Everything after the stem fusion (four depthwise residual
encoder blocks, kernel 15, dilations 1/2/4/8; decoder Conv(24 -> 16, k7,
d12) -> Conv(16 -> 8, k5, d24); waveform head 1x1 conv + tanh; dropout
0.10), the input normalization (min-max per window), the target, the loss,
the training schedule and the 60 s evaluation are those of the full model.
Summary/model_variants.csv and model_layers.csv list the branches,
receptive field, parameters and MACs of every model.

The full model (k=32 + k=64 + k=128) is not trained here: its 10 s / 0%
single model m0 is read from the results of 261007 (CapnoBase) and 261013
(BIDMC, STEAM2; the weight decay 261013 selected in the fold), which is the
same model, seed and training (--reference-261007-root,
--reference-261013-root). Training it here instead: --variants full.

Differences from 261007 (CapnoBase) / 261013 (BIDMC, STEAM2):
* input windows: 10 s, non-overlapping;
* one model per LOSOCV fold, scored alone: no ensemble and no averaging
  of models. Its seed is the seed of member m0 in 261007 / 261013
  (stable_seed('261006-ensemble-member', 'w10s_nonoverlap', dataset,
  subject, 0)), the same for every model;
* the stem branches of the models above;
* every mean in the summary comes with its SD (ddof=1).

Training settings per dataset (the same for every model of a fold)
--------------------------------------------------------------------
* CapnoBase (261007): validation = the last 20% of every training
  subject's windows; AdamW weight decay 5e-5 on every parameter; CapnoBase
  read and processed as in 261007.
* BIDMC and STEAM2 (261013, tuned for these datasets): subject-wise
  validation, i.e. 20% of the training subjects of the fold held out whole
  (the validation subjects 261013 drew for the fold), all windows of the
  other training subjects for training; AdamW weight decay on the
  convolution weights only, with the value 261013 selected in that fold
  by the validation loss of these validation subjects (candidates 1e-2,
  1e-1, 1). The value is read from the 261013 results
  (--reference-261013-root, WD_Selection\01_w10s_nonoverlap\{dataset}\
  {subject}_selection.json); a fold without a selection, or whose 261013
  validation subjects differ (other data), is skipped and listed in
  failed_members*.csv and incomplete_folds.csv. STEAM2 is read with the
  261013 reader (EEGLAB sessions at 200 Hz, channels by label, PPG and RSP
  paired by identical file names, sessions split at internal boundary
  events, all sessions of a person in one subject) and its belt reference
  keeps the polarity it was recorded with (profile
  adult_belt_as_recorded); BIDMC is read as in 261007 and 261013.
The weight decay of every fold is in Summary/training_by_fold.csv.
Otherwise for all datasets: raw PPG input without a band-pass, target =
respiratory-phase waveform from the expert CO2 breath labels (CapnoBase) or
the respiratory-band reference (BIDMC, STEAM2), the v16 waveform loss
(SmoothL1 + Pearson + derivative), learning rate 2e-4 with cosine decay, up
to 240 epochs, at least 20, patience 25, best validation epoch restored,
v16 jitter of training windows (random shift up to 0.094 s, gain
0.95-1.05, small noise); no speed augmentation, no rate-balanced sampling.

Evaluation
----------
On the held-out subject the outputs of the 10 s windows are joined into
one waveform per contiguous run, and every complete 60 s minute with a
valid reference is scored (Davies & Mandic 2022 style; no subject is
excluded):
* breathing rate = frequency of the largest FFT peak of the predicted
  waveform (3-60 breaths/min for CapnoBase, 6-30 for BIDMC and STEAM2);
  ground truth = the number of reference breaths in the minute;
* FFT metrics over all minutes: MAE +- SD, RMSE, bias +- SD and 95% limits
  of agreement, Pearson r, share of minutes within 1 and 2 breaths/min,
  and the mean +- SD of the per-subject MAE;
* Davies & Mandic (2022): mAE (median minute AE) and mMAE (median subject
  MAE), each with its interquartile range;
* waveform agreement: Pearson r (PCC) between the predicted and the target
  waveform in every minute (mean +- SD and median);
* every subject with its median true and predicted rate, MAE and PCC.
Results are per subject (all sessions of a STEAM2 subject form one fold).
model_comparison.csv compares every pair of models per dataset (the full
model included) on the subjects scored by both: metrics side by side,
per-subject MAE difference mean +- SD, paired Wilcoxon tests of the
per-subject FFT MAE, of the per-subject PCC and of the per-minute absolute
error, Benjamini-Hochberg over the pairs of one dataset.

Memory and progress
-------------------
Every dataset and model is trained in its own Python process (42-53 runs;
started by the main process, which then writes the summary), so its host
and GPU memory is returned before the next one. Out of host or GPU memory
or a CUDA / cuDNN failure stops the worker instead of failing every
following fold (a memory error while reading the data stops it instead of
skipping subjects); the worker is started again with --resume, which keeps
every stored run and continues an interrupted one from its last epoch. The output of every worker is printed by the main
process (also in IDE consoles such as Spyder): the start of every run, the
training and validation loss, learning rate, best validation loss and
patience after every epoch, early stopping and the result of every run.
None of this changes the models or the training: data, preprocessing,
network (apart from the stem branches), loss, optimizer, schedule, seeds,
cuDNN settings and evaluation are those of 261007 (CapnoBase) and 261013
(BIDMC, STEAM2). --single-process trains everything in one process.

Data and outputs
----------------
Datasets: CapnoBase, BIDMC and STEAM2 (paths in DATASET_CONFIGS, or
--capnobase-path / --bidmc-path / --steam2-ppg-dir / --steam2-rsp-dir).
All results go to PROJECT_ROOT
(D:\PPG2RSP_RRWaveNet_Inspired\261016_branchwise_ablation); {model} is
the folder of a model (no_k32, no_k64, no_k128, only_k32, only_k64,
only_k128; see the table above):
  {model}\01_w10s_nonoverlap\{dataset}\Members\m0\   model, learning
      curve, training audit and test predictions of each fold
  {model}\01_w10s_nonoverlap\{dataset}\Fold_Summaries\   per-subject
      JSON and per-minute CSV
  Data_Quality\   window and subject quality of every dataset
  Summary\   fft_davies_overall.csv, fft_davies_overall_mean_sd.csv,
      model_comparison.csv, model_variants.csv, model_layers.csv,
      training_by_fold.csv, per_subject_fft.csv, minutes.csv,
      incomplete_folds.csv, figures and
      261016_branchwise_ablation_results.xlsx
Order: CapnoBase, BIDMC, STEAM2; within a dataset model by model in the
order of the table above (every fold of one model, then the next).

Run (GPU):                       python 261016_branchwise_ablation.py
Resume after an interruption:    python 261016_branchwise_ablation.py --resume
                                 (finished folds are reused, an interrupted
                                 fold continues from its last epoch)
Some datasets / models:          --datasets capnobase --variants only_k32 only_k64
                                 (folder names; only_k32 = only_k=32,
                                 no_k128 = k=32 + k=64, ...)
Results of 261007 / 261013 (full model; weight decay per fold of BIDMC /
STEAM2):                         --reference-261007-root <261007 results>
                                 --reference-261013-root <261013 results>
Parallel workers (e.g. 2):       --resume --shard 0/2, --resume --shard 1/2,
                                 then --resume --summary-only
Summary of stored results only:  --summary-only
Check the data without training: --dry-run --no-require-cuda
Check how STEAM2 is read:        --inspect-steam2 --no-require-cuda
"""

import argparse
import csv
import gc
import hashlib
import json
import math
import os
import random
import re
import subprocess
import sys
import time
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.io as sio
import torch
import torch.nn as nn
from scipy import stats
from scipy.signal import (
    butter,
    correlate,
    detrend,
    filtfilt,
    find_peaks,
    resample,
    resample_poly,
    sosfiltfilt,
    welch,
)
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader, Dataset


# ---------------------------------------------------------------------------
# Paths, datasets, v16 constants, and requested experiment order
# ---------------------------------------------------------------------------

PROJECT_ROOT = Path(r"D:\PPG2RSP_RRWaveNet_Inspired\261016_branchwise_ablation")
# Every result of this script is written under PROJECT_ROOT.
DEFAULT_RESULTS_ROOT = PROJECT_ROOT
# 261007 results: the 10 s single model m0 of the full model (CapnoBase) is
# read from here for the comparison (it is not trained again).
REFERENCE_261007_ROOT = Path(r"D:\PPG2RSP_RRWaveNet_Inspired\261007_RRWaveNet_NoEvent_v1")
# 261013_RRWaveNet_Adult_SubjectVal_WD results: the weight decay it selected in
# every BIDMC / STEAM2 fold (WD_Selection) is read from here, and its member m0
# of that weight decay is the full model of the comparison.
REFERENCE_261013_ROOT = Path(r"D:\PPG2RSP_RRWaveNet_Inspired\261013_RRWavNet_STEAM_BIDMC_v1")

DATASET_CONFIGS = {
    "bidmc": {
        "label": "BIDMC",
        "data_path": (
            r"D:\STEAM\Multimodal_prediction\PPG-RSP\Preprocessed"
            r"\bidmc-ppg-and-respiration-dataset-1.0.0"
            r"\bidmc-ppg-and-respiration-dataset-1.0.0\bidmc_data.mat"
        ),
        "target_fs": 128,
        "default_fs": 125,
        "profile": "adult_fixed_band",
    },
    "capnobase": {
        "label": "CapnoBase",
        "data_path": (
            r"D:\STEAM\Multimodal_prediction\PPG-RSP\Opendata"
            r"\Capnobase\data\csv"
        ),
        "target_fs": 128,
        "default_fs": 300,
        "profile": "capnography_wide_rr",
        # Expert CO2 breath annotations ({case}_8min_labels.csv) are the
        # ground truth when present; detection is the fallback.
        "expert_breath_labels": True,
        # Expert artifact spans: PPG artifacts are unusable input samples,
        # CO2 artifacts invalidate the reference (training windows and
        # scored minutes that overlap them are excluded).
        "use_artifact_labels": True,
    },
    "steam2": {
        "label": "STEAM2",
        "ppg_dir": (
            r"D:\STEAM\Multimodal_prediction\PPG-RSP\Preprocessed"
            r"\PPG\2back_raw_v2"
        ),
        "rsp_dir": (
            r"D:\STEAM\Multimodal_prediction\PPG-RSP\Preprocessed"
            r"\RSP\2back_raw_v2"
        ),
        "target_fs": 128,
        # EEGLAB sessions (EEG.data, EEG.srate = 200 Hz, one channel labelled
        # PPG or RSP); default_fs is used only when a file has no srate
        # (261013).
        "default_fs": 200,
        "profile": "adult_belt_as_recorded",
    },
}

DATASET_ORDER = ("capnobase", "bidmc", "steam2")
# Datasets run when none are given on the command line.
DEFAULT_DATASETS = ("capnobase", "bidmc", "steam2")


@dataclass(frozen=True)
class WindowSpec:
    order: int
    window_sec: float
    stride_sec: float
    name: str
    label: str


# 10 s non-overlapping windows only (the 261007 10 s condition).
WINDOW_SPECS = (WindowSpec(1, 10.0, 10.0, "w10s_nonoverlap", "10 s / 0% overlap"),)

# Model input: resampled PPG time series, min-max [0, 1] per window (no filter).
PPG_INPUT = "raw"

# v16 input/output and training constants.
TARGET_FS = 64  # model and analysis sampling rate (Hz)
EVAL_WINDOW_SEC = 60.0
EVAL_STRIDE_SEC = 60.0
BATCH_SIZE = 48
# Maximum epochs of the final model (model A: 120).
DEFAULT_EPOCHS = 240
MIN_EPOCHS = 20
PATIENCE = 25
LEARNING_RATE = 2e-4
WEIGHT_DECAY = 5e-5
LOSS_WEIGHTS = (0.45, 0.45, 0.10)
VAL_WINDOW_RATIO = 0.20
# Training settings per dataset. "261007" (CapnoBase): temporal validation
# (last VAL_WINDOW_RATIO of every training subject's windows) and AdamW weight
# decay WEIGHT_DECAY on every parameter. "261013" (BIDMC, STEAM2): the settings
# tuned in 261013_RRWaveNet_Adult_SubjectVal_WD, i.e. subject-wise validation
# (VAL_SUBJECT_RATIO of the training subjects held out whole, drawn as in
# 261013) and, in every LOSOCV fold, the weight decay that 261013 selected for
# that fold (read from its WD_Selection), on the convolution weights only.
TRAINING_PROTOCOLS = {"capnobase": "261007", "bidmc": "261013", "steam2": "261013"}
VAL_SUBJECT_RATIO = 0.20
# Test windows of the held-out subject: 10 s without overlap, as in 261007
# and 261013 (recorded in every summary).
INFERENCE_PROTOCOL = "stored_windows"
# Analysis profile of summaries written before it was recorded (261007).
LEGACY_PROFILES = {
    "CapnoBase": "capnography_wide_rr",
    "BIDMC": "adult_fixed_band",
    "STEAM2": "adult_fixed_band",
}


# Breath detection and reference quality (RRest / Charlton et al. 2016).
RESP_BAND_HZ = (0.1, 0.5)
RR_BAND_BPM = (6.0, 30.0)  # RR representable in the 0.1-0.5 Hz analysis band
CTO_PEAK_FRACTION = 0.2  # count-orig: peaks above 0.2 x Q3 of all peak values
CTO_PEAK_QUANTILE = 0.75
FFT_MIN_POINTS = 16384
REF_QC_WINDOW_SEC = 32.0  # RRest window length
REF_QC_HOP_SEC = 8.0
REF_AGREEMENT_BPM = 2.0  # RRest: count-orig RR and FFT RR within 2 bpm
# Share of the breathing time that must lie in clean cycles (exactly one
# sub-zero trough between consecutive breaths). A cycle with more troughs
# skipped a breath below the count-orig threshold, so the count and the RR
# disagree and the reference is ambiguous.
REF_MIN_VALID_CYCLE_FRACTION = 0.75
BREATH_BLOCK_SEC = 60.0  # continuous breath detection in blocks with margins
BREATH_BLOCK_MARGIN_SEC = 10.0
EVENT_SIGMA_SEC = 0.3  # event width when a breath has no neighbour
# Event width proportional to the local breath period (0.12 x period).
EVENT_SIGMA_PERIOD_FRACTION = 0.12
EVENT_SIGMA_LIMITS_SEC = (0.15, 0.5)
# Count loss: heatmap counted above this background level.
EVENT_COUNT_THRESHOLD = 0.1
# v16 jitter of training windows: random shift up to 0.094 s (12 samples
# at 128 Hz), gain 0.95-1.05 and small noise.
SHIFT_AUG_SEC = 0.094
# conv1x1: 1x1 stem fusion + GroupNorm; none: ablation, the concatenated
# stem branches feed the encoder.
STEM_FUSION_MODES = ("conv1x1", "none")

# PPG-only quality rules. These are applied to each candidate input window.
PPG_SQI_THRESHOLD = 0.86
PPG_HR_MIN_BPM = 40.0
PPG_HR_MAX_BPM = 180.0
PPG_MAX_GAP_SEC = 3.0
PPG_MAX_INTERVAL_RATIO = 2.2
PPG_MIN_PULSES = 5
PPG_PULSE_BAND = (0.5, 5.0)
PPG_MIN_PEAK_DISTANCE_SEC = 60.0 / PPG_HR_MAX_BPM
TECHNICAL_MIN_FINITE_RATIO = 0.50
TECHNICAL_MAX_FLAT_FRACTION = 0.995
TECHNICAL_MAX_SATURATION_FRACTION = 0.80

# Dataset analysis profiles. BIDMC and STEAM2 (adults, impedance or belt
# reference) keep the v16 processing. CapnoBase differs in three documented
# ways: RR spans 5-48 bpm (median 11.7; 29 children), the reference is a
# sidestream capnogram (trapezoid waveform, 1-3 s transport delay), and
# infant heart rates can exceed 180 bpm.
ANALYSIS_PROFILES = {
    "adult_fixed_band": {
        "description": "v16 processing: fixed 0.1-0.5 Hz, sign alignment",
        "band_hz": RESP_BAND_HZ,
        "adaptive_upper": False,
        "rr_band_bpm": RR_BAND_BPM,
        "min_breath_interval_sec": 1.5,
        "ppg_hr_range_bpm": (PPG_HR_MIN_BPM, PPG_HR_MAX_BPM),
        "ppg_spectral_beat_distance": False,
        "max_target_lag_sec": 0.0,  # no lag search: v16 zero-lag sign alignment
        "fixed_target_polarity": False,
        "target_polarity": "zero_lag_sign",
        "fft_subharmonic_check": False,
        "resampler": "fft",
    },
    # STEAM2 belt: as adult_fixed_band, but the reference keeps the polarity it
    # was recorded with (one belt type for every subject). The v16 zero-lag
    # sign rule picks the sign from the correlation of the PPG and the belt in
    # the respiratory band at zero lag, which is near zero for the belt (the
    # PPG respiratory modulation is a variable fraction of a cycle off the
    # chest movement), so it flips the target at random between sessions.
    "adult_belt_as_recorded": {
        "description": "v16 processing: fixed 0.1-0.5 Hz, reference polarity as recorded",
        "band_hz": RESP_BAND_HZ,
        "adaptive_upper": False,
        "rr_band_bpm": RR_BAND_BPM,
        "min_breath_interval_sec": 1.5,
        "ppg_hr_range_bpm": (PPG_HR_MIN_BPM, PPG_HR_MAX_BPM),
        "ppg_spectral_beat_distance": False,
        "max_target_lag_sec": 0.0,  # no lag search
        "fixed_target_polarity": True,
        "target_polarity": "as_recorded",
        "fft_subharmonic_check": False,
        "resampler": "fft",
    },
    "capnography_wide_rr": {
        "description": (
            "0.05 Hz to clip(0.5 x PPG heart rate, 0.5, 1.0) Hz, capnogram "
            "delay alignment with fixed polarity, 40-220 bpm PPG beats"
        ),
        # Lower edge 0.05 Hz keeps 5 bpm breathing; the upper edge follows
        # the PPG heart rate (an octave below it, so the cardiac component
        # stays strongly attenuated) up to 1.0 Hz = 60 bpm.
        "band_hz": (0.05, 1.0),
        "adaptive_upper": True,
        "rr_band_bpm": (3.0, 60.0),
        "min_breath_interval_sec": 0.8,  # up to 75 bpm
        "ppg_hr_range_bpm": (40.0, 220.0),
        "ppg_spectral_beat_distance": True,
        # Lag search window: at least 3 s (sidestream transport delay) and at
        # least half a breath period, because the PPG follows breathing
        # mechanics while CO2 follows gas exchange (about a quarter cycle
        # later), and positive-pressure ventilation inverts the PPG
        # modulation. Polarity is fixed: high CO2 is expiration.
        "max_target_lag_sec": 3.0,
        "fixed_target_polarity": True,
        # Waveform target when expert breath labels exist: "phase" is a
        # respiratory-phase waveform built from the labels (see
        # label_phase_waveform); "co2" is the band-passed capnogram.
        "capnography_target": "phase",
        "fft_subharmonic_check": True,  # trapezoid capnogram harmonics
        "resampler": "poly",
    },
}
MAX_TARGET_LAG_SEC = 15.0  # upper bound of the lag window (half of 4 bpm)
ADAPTIVE_UPPER_HR_FRACTION = 0.5
ADAPTIVE_UPPER_LIMITS_HZ = (0.5, 1.0)
# FFT RR: a spectral peak at f/2 or f/3 with at least this share of the main
# peak's power is taken as the fundamental (trapezoid capnograms have strong
# harmonics, so the largest peak can be the 2nd harmonic at slow rates).
FFT_SUBHARMONIC_POWER_RATIO = 0.3

# Network of the full model (261007 = 260928 layout). Only the sampling rate
# (TARGET_FS), the decoder dilations and the dropout differ from 260928.
MODEL_CONFIG = {
    # Stem kernels 32 / 64 / 128 samples = 0.5 / 1 / 2 s at 64 Hz, 8 filters each.
    "stem_kernel_sizes": (32, 64, 128),
    "stem_base_channels": 8,
    "hidden_channels": 24,
    "encoder_kernel_size": 15,
    # Encoder receptive field 6.58 s.
    "encoder_dilations": (1, 2, 4, 8),
    "dropout": 0.10,
    "decoder_mid_channels": (16, 8),
    "decoder_kernel_sizes": (7, 5),
    # Decoder receptive field 2.64 s.
    "decoder_dilations": (12, 24),
    "decoder_heads": ("waveform_tanh",),
}
# One model per LOSOCV fold, scored alone (no ensemble). Its index is 0, so
# its seed is the seed of ensemble member m0 in 261007.
SINGLE_MEMBER = 0
# Post-fusion representation diagnostics written per member.
FEATURE_METRICS = (
    "Feature_OffDiag_SignedMean",
    "Feature_OffDiag_AbsMean",
    "Feature_EffectiveRank",
    "Feature_EffectiveRank_Normalized",
    "Feature_ParticipationRatio",
)


FULL_STEM_KERNELS = tuple(MODEL_CONFIG["stem_kernel_sizes"])


@dataclass(frozen=True)
class ModelOptions:
    """The stem branches (kernel sizes in samples at 64 Hz) of the model;
    everything else is the 261007 full model."""

    stem_fusion: str = "conv1x1"
    dropout: float = MODEL_CONFIG["dropout"]
    decoder_dilations: Tuple[int, ...] = tuple(MODEL_CONFIG["decoder_dilations"])
    stem_kernel_sizes: Tuple[int, ...] = FULL_STEM_KERNELS

    @property
    def key(self) -> str:
        key = "rrwavenet_fs{}_{}_dec{}_p{:.2f}".format(
            TARGET_FS,
            "stemfusion" if self.stem_fusion == "conv1x1" else "nostemfusion",
            "-".join(str(value) for value in self.decoder_dilations),
            self.dropout,
        )
        if tuple(self.stem_kernel_sizes) == FULL_STEM_KERNELS:
            # The 261007 key, so the full model here is the 261007 full model.
            return key
        return key + "_stem" + "-".join(str(value) for value in self.stem_kernel_sizes)

    @property
    def stem_description(self) -> str:
        return " + ".join(f"k={value}" for value in self.stem_kernel_sizes)

    @property
    def stem_channels(self) -> int:
        return int(MODEL_CONFIG["stem_base_channels"]) * len(self.stem_kernel_sizes)

    @property
    def label(self) -> str:
        fusion = (
            "1x1 stem fusion + GroupNorm"
            if self.stem_fusion == "conv1x1"
            else "no stem fusion (concatenated stem into the encoder)"
        )
        dilations = "/".join(str(value) for value in self.decoder_dilations)
        return (
            f"stem branches {self.stem_description} x 8 filters ({self.stem_channels} channels), "
            f"{fusion}, residual encoder, decoder dilations {dilations}, dropout {self.dropout:.2f}"
        )

    def receptive_fields(self) -> Dict[str, float]:
        """Receptive fields in samples and seconds: the longest stem kernel,
        the encoder blocks (two convolutions each) and the decoder layers
        (unchanged in every model), and the whole network."""
        stem = max(int(value) for value in self.stem_kernel_sizes) - 1
        encoder = sum(
            2 * (int(MODEL_CONFIG["encoder_kernel_size"]) - 1) * int(d)
            for d in MODEL_CONFIG["encoder_dilations"]
        )
        decoder = sum(
            (int(k) - 1) * int(d)
            for k, d in zip(MODEL_CONFIG["decoder_kernel_sizes"], self.decoder_dilations)
        )
        samples = {
            "Stem_RF": stem + 1,
            "Encoder_RF": encoder + 1,
            "Decoder_RF": decoder + 1,
            "Model_RF": stem + encoder + decoder + 1,
        }
        result: Dict[str, float] = {}
        for name, value in samples.items():
            result[f"{name}_Samples"] = int(value)
            result[f"{name}_Sec"] = float(value) / TARGET_FS
        return result

    def build(self) -> "DeepOnlyV16Decoder":
        return DeepOnlyV16Decoder(
            stem_fusion=self.stem_fusion,
            dropout=self.dropout,
            decoder_dilations=self.decoder_dilations,
            stem_kernel_sizes=self.stem_kernel_sizes,
        )


@dataclass(frozen=True)
class Variant:
    """One LOSOCV run: the model, its options and its epochs."""

    name: str
    label: str
    options: ModelOptions
    epochs: int


def branch_variant(name: str, label: str, kernels: Sequence[int]) -> Variant:
    return Variant(name, label, ModelOptions(stem_kernel_sizes=tuple(kernels)), DEFAULT_EPOCHS)


# The full model (all three stem branches; read from 261007 / 261013) and
# every other combination of the branches except none. The key is the
# results folder and the --variants value; the label (the kept branches) is
# the name of the model in every result table, figure and log.
VARIANTS = {
    "full": branch_variant("full", "k=32 + k=64 + k=128", (32, 64, 128)),
    "no_k32": branch_variant("no_k32", "k=64 + k=128", (64, 128)),
    "no_k64": branch_variant("no_k64", "k=32 + k=128", (32, 128)),
    "no_k128": branch_variant("no_k128", "k=32 + k=64", (32, 64)),
    "only_k32": branch_variant("only_k32", "only_k=32", (32,)),
    "only_k64": branch_variant("only_k64", "only_k=64", (64,)),
    "only_k128": branch_variant("only_k128", "only_k=128", (128,)),
}
# Trained by default: every model except the full one.
ABLATION_VARIANTS = tuple(name for name in VARIANTS if name != "full")


# ---------------------------------------------------------------------------
# Small utilities and signal preprocessing
# ---------------------------------------------------------------------------

def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = False
        torch.backends.cudnn.benchmark = True


def stable_seed(*parts: object) -> int:
    text = "|".join(str(part) for part in parts)
    digest = hashlib.sha256(text.encode("utf-8")).hexdigest()
    return int(digest[:8], 16) % (2**31 - 1)


def safe_tag(value: object) -> str:
    return "".join(
        char if char.isalnum() or char in "-_ ." else "_"
        for char in str(value)
    ).replace(" ", "_")


def sanitize_signal(signal: np.ndarray) -> Tuple[np.ndarray, int]:
    arr = np.asarray(signal, dtype=np.float64).reshape(-1)
    if arr.size == 0:
        return arr, 0
    finite = np.isfinite(arr)
    replaced = int((~finite).sum())
    if not finite.any():
        return np.zeros_like(arr), replaced
    if replaced:
        arr = arr.copy()
        arr[~finite] = float(np.median(arr[finite]))
    return arr, replaced


def bandpass_filter(
    signal: np.ndarray,
    fs: int,
    low: float,
    high: float,
    order: int = 3,
) -> Tuple[np.ndarray, int]:
    arr, replaced = sanitize_signal(signal)
    nyquist = 0.5 * float(fs)
    low = max(float(low), 1e-5)
    high = min(float(high), nyquist * 0.99)
    if arr.size < max(32, order * 6 + 1) or high <= low:
        return arr.astype(np.float64), replaced
    b, a = butter(order, [low / nyquist, high / nyquist], btype="band")
    try:
        filtered = filtfilt(b, a, arr)
    except ValueError:
        filtered = arr
    return np.asarray(filtered, dtype=np.float64), replaced


def maybe_resample(
    signal: np.ndarray,
    source_fs: float,
    target_fs: float,
) -> np.ndarray:
    arr, _ = sanitize_signal(signal)
    if int(round(source_fs)) == int(round(target_fs)):
        return arr.astype(np.float32)
    target_len = max(1, int(round(len(arr) * target_fs / float(source_fs))))
    return resample(arr, target_len).astype(np.float32)


def maybe_resample_with_mask(
    signal: np.ndarray,
    source_fs: float,
    target_fs: float,
    method: str = "fft",
) -> Tuple[np.ndarray, np.ndarray]:
    """Resample a signal and retain a finite-sample validity mask.

    "fft" is the v16 resampler. "poly" (anti-aliased polyphase) is used for
    CapnoBase: FFT resampling treats the 8 min record as periodic and rings
    at the sharp edges of the capnogram.
    """
    raw = np.asarray(signal, dtype=np.float64).reshape(-1)
    finite = np.isfinite(raw).astype(np.float64)
    clean, _ = sanitize_signal(raw)
    if int(round(source_fs)) == int(round(target_fs)):
        return clean.astype(np.float32), finite >= 0.5
    if method == "poly":
        source, target = int(round(source_fs)), int(round(target_fs))
        divisor = math.gcd(source, target)
        up, down = target // divisor, source // divisor
        signal_out = resample_poly(clean, up, down).astype(np.float32)
        mask_out = resample_poly(finite, up, down) >= 0.5
        return signal_out, mask_out
    target_len = max(1, int(round(len(clean) * target_fs / float(source_fs))))
    signal_out = resample(clean, target_len).astype(np.float32)
    mask_out = resample(finite, target_len) >= 0.5
    return signal_out, mask_out


def resample_signal_poly(
    signal: np.ndarray,
    source_fs: float,
    target_fs: float,
) -> np.ndarray:
    arr, _ = sanitize_signal(signal)
    source_fs = int(round(float(source_fs)))
    target_fs = int(round(float(target_fs)))
    if source_fs == target_fs:
        return arr.astype(np.float32)
    divisor = math.gcd(source_fs, target_fs)
    return resample_poly(
        arr,
        target_fs // divisor,
        source_fs // divisor,
    ).astype(np.float32)


def align_target_orientation(
    ppg_low: np.ndarray,
    rsp_low: np.ndarray,
) -> np.ndarray:
    """Preserve v16 target orientation without using it for quality selection."""
    x, _ = sanitize_signal(ppg_low)
    y, _ = sanitize_signal(rsp_low)
    length = min(len(x), len(y))
    x = x[:length]
    y = y[:length]
    if length >= 2:
        x_centered = x - float(np.mean(x))
        y_centered = y - float(np.mean(y))
        if float(np.mean(x_centered * y_centered)) < 0.0:
            y = -y
    return y.astype(np.float32)


def sos_bandpass(
    signal: np.ndarray,
    fs: int,
    low: float,
    high: float,
    order: int = 3,
) -> np.ndarray:
    """Zero-phase Butterworth band-pass in second-order sections.

    Used for the wide capnography band: at 0.05 Hz and 128 Hz the transfer
    function form of the v16 filter is numerically fragile.
    """
    arr, _ = sanitize_signal(signal)
    nyquist = 0.5 * float(fs)
    high = min(float(high), nyquist * 0.99)
    if arr.size < max(32, order * 6 + 1) or high <= low:
        return arr.astype(np.float64)
    sos = butter(order, [low / nyquist, high / nyquist], btype="band", output="sos")
    try:
        return np.asarray(sosfiltfilt(sos, arr), dtype=np.float64)
    except ValueError:
        return arr.astype(np.float64)


def estimate_ppg_heart_rate_hz(
    ppg_signal: np.ndarray,
    fs: int,
    hr_range_bpm: Tuple[float, float],
) -> float:
    """Spectral heart rate of the PPG pulse band (Welch, 16 s segments)."""
    pulse, _ = bandpass_filter(ppg_signal, fs, PPG_PULSE_BAND[0], PPG_PULSE_BAND[1])
    if len(pulse) < 4 * fs or float(np.std(pulse)) < 1e-8:
        return np.nan
    freqs, power = welch(pulse, fs=fs, nperseg=min(len(pulse), int(16 * fs)))
    band = (freqs >= hr_range_bpm[0] / 60.0) & (freqs <= hr_range_bpm[1] / 60.0)
    if not band.any():
        return np.nan
    return float(freqs[band][np.argmax(power[band])])


def profile_name(profile: dict) -> str:
    for name, candidate in ANALYSIS_PROFILES.items():
        if candidate is profile:
            return name
    return "custom"


def resolve_respiratory_band(profile: dict, hr_hz: float) -> Tuple[float, float]:
    """Analysis band of one segment, from the profile and the PPG heart rate."""
    low, high = profile["band_hz"]
    if profile["adaptive_upper"]:
        if np.isfinite(hr_hz):
            high = float(
                np.clip(ADAPTIVE_UPPER_HR_FRACTION * hr_hz, *ADAPTIVE_UPPER_LIMITS_HZ)
            )
        else:
            high = ADAPTIVE_UPPER_LIMITS_HZ[0]
    return float(low), float(high)


def respiratory_filter(
    signal: np.ndarray,
    fs: int,
    band: Tuple[float, float],
    profile: dict,
) -> np.ndarray:
    """Respiratory band-pass: the v16 filter for the adult profile, SOS otherwise."""
    if not profile["adaptive_upper"] and tuple(band) == tuple(RESP_BAND_HZ):
        return bandpass_filter(signal, fs, band[0], band[1])[0]
    return sos_bandpass(signal, fs, band[0], band[1])


def label_phase_waveform(labels: np.ndarray, length: int) -> np.ndarray:
    """Respiratory-phase waveform from expert breath onsets.

    The phase rises linearly from 0 to 1 between consecutive labels (and is
    extrapolated with the first and last interval), and the waveform is
    sin(2 pi phase), so every labelled breath is one cycle whose upward zero
    crossing is exactly the label. RR needs only breath timing; the rest of
    a capnogram (trapezoid shape, inspiratory/expiratory split, EtCO2 level)
    varies between subjects and cannot be inferred from the PPG, so this
    target keeps the timing and drops what the model cannot predict.
    """
    labels = np.unique(np.asarray(labels, dtype=np.float64))
    samples = np.arange(length, dtype=np.float64)
    intervals = np.diff(labels)
    index = np.clip(np.searchsorted(labels, samples, side="right") - 1, 0, len(labels) - 2)
    phase = (samples - labels[index]) / intervals[index]
    return np.sin(2.0 * np.pi * phase).astype(np.float32)


def shift_signal(signal: np.ndarray, lag: int, fill=None) -> np.ndarray:
    """Return z with z[t] = signal[t + lag]; edges padded with `fill` or edge values."""
    arr = np.asarray(signal)
    if lag == 0 or len(arr) == 0:
        return arr.copy()
    if lag > 0:
        pad = np.full(lag, arr[-1] if fill is None else fill, dtype=arr.dtype)
        return np.concatenate([arr[lag:], pad])
    pad = np.full(-lag, arr[0] if fill is None else fill, dtype=arr.dtype)
    return np.concatenate([pad, arr[:lag]])


def align_target_lag_and_sign(
    ppg_low: np.ndarray,
    rsp_low: np.ndarray,
    fs: int,
    max_lag_sec: float,
    allow_sign_flip: bool = True,
) -> Tuple[np.ndarray, int, float, float]:
    """Delay-align the reference to the PPG respiratory component.

    The PPG modulation follows breathing mechanics; a capnogram follows gas
    exchange, about a quarter cycle later, plus the sidestream transport
    delay (1-3 s), and positive-pressure ventilation inverts the PPG
    modulation. The zero-lag correlation that v16 used to pick the sign
    therefore changes sign with each subject's RR, delay and ventilation
    mode. The lag within +-max_lag_sec with the best correlation is removed.

    With allow_sign_flip=False (capnography) only positive correlation is
    accepted: a capnogram has a physical polarity (high CO2 = expiration),
    and flipping it would also move the breath-onset targets by about half
    a cycle relative to the PPG for some subjects but not others. A window
    of at least half a breath period lets the lag alone reach every phase.
    Breath counts do not depend on this shift.

    Returns (aligned reference, lag in samples, sign, correlation); a positive
    lag means the reference lagged the PPG.
    """
    x, _ = sanitize_signal(ppg_low)
    y, _ = sanitize_signal(rsp_low)
    length = min(len(x), len(y))
    x = x[:length] - float(np.mean(x[:length]))
    y = y[:length] - float(np.mean(y[:length]))
    scale = float(np.std(x) * np.std(y))
    if length < 2 or scale < 1e-12:
        return y.astype(np.float32), 0, 1.0, np.nan
    full = correlate(y, x, mode="full", method="fft")
    lags = np.arange(-(length - 1), length)
    window = np.abs(lags) <= int(round(max_lag_sec * fs))
    values = full[window] / ((length - np.abs(lags[window])) * scale)
    best = int(np.argmax(np.abs(values) if allow_sign_flip else values))
    lag = int(lags[window][best])
    corr = float(values[best])
    sign = 1.0 if corr >= 0 or not allow_sign_flip else -1.0
    return (sign * shift_signal(y, lag)).astype(np.float32), lag, sign, corr


def instance_minmax_normalize(
    data: np.ndarray,
    feature_range: Tuple[float, float],
) -> np.ndarray:
    arr = np.asarray(data, dtype=np.float32)
    min_value = np.min(arr, axis=-1, keepdims=True)
    max_value = np.max(arr, axis=-1, keepdims=True)
    denominator = max_value - min_value
    denominator[denominator < 1e-8] = 1.0
    scaled = (arr - min_value) / denominator
    lower, upper = feature_range
    return (scaled * (upper - lower) + lower).astype(np.float32)


def extract_window_starts(
    signal_length: int,
    fs: int,
    window_sec: float,
    stride_sec: float,
) -> np.ndarray:
    window_len = int(round(window_sec * fs))
    stride_len = int(round(stride_sec * fs))
    if signal_length < window_len:
        return np.empty(0, dtype=np.int64)
    return np.arange(
        0,
        signal_length - window_len + 1,
        stride_len,
        dtype=np.int64,
    )


# ---------------------------------------------------------------------------
# Window-independent PPG and RSP quality maps
# ---------------------------------------------------------------------------

PPG_QC_UNIT_SEC = 10.0
PPG_USABLE_SQI_THRESHOLD = 0.50
MIN_USABLE_PROPORTION = 0.50


def finite_ratio(signal: np.ndarray) -> float:
    arr = np.asarray(signal).reshape(-1)
    return float(np.isfinite(arr).mean()) if arr.size else 0.0


def flat_fraction(signal: np.ndarray) -> float:
    arr, _ = sanitize_signal(signal)
    if len(arr) < 2:
        return 1.0
    robust_range = float(np.percentile(arr, 95) - np.percentile(arr, 5))
    tolerance = max(1e-12, robust_range * 1e-4)
    return float(np.mean(np.abs(np.diff(arr)) <= tolerance))


def saturation_fraction(signal: np.ndarray) -> float:
    arr, _ = sanitize_signal(signal)
    if len(arr) < 2:
        return 1.0
    low = float(np.min(arr))
    high = float(np.max(arr))
    span = max(1e-12, high - low)
    tolerance = span * 1e-6
    saturated = (np.abs(arr - low) <= tolerance) | (
        np.abs(arr - high) <= tolerance
    )
    return float(np.mean(saturated))


def pulse_template_diagnostics(
    ppg_signal: np.ndarray,
    structural_mask: np.ndarray,
    fs: int,
    hr_range_bpm: Tuple[float, float] = (PPG_HR_MIN_BPM, PPG_HR_MAX_BPM),
    min_peak_distance_sec: float = PPG_MIN_PEAK_DISTANCE_SEC,
) -> Tuple[np.ndarray, dict]:
    """Build a continuous beat-quality map from the full PPG segment.

    The defaults are the v16 adult rules. The capnography profile passes a
    40-220 bpm range and a beat spacing derived from the PPG's own spectral
    heart rate, because infants can exceed 180 bpm.
    """
    clean, _ = sanitize_signal(ppg_signal)
    pulse_band, _ = bandpass_filter(
        clean,
        fs,
        PPG_PULSE_BAND[0],
        PPG_PULSE_BAND[1],
    )
    spread = float(np.std(pulse_band))
    quality_map = np.zeros(len(clean), dtype=np.int8)
    if spread < 1e-8:
        return quality_map, {
            "PPG_SQI": np.nan,
            "PPG_Pulse_Count": 0,
            "PPG_Usable_Pulse_Count": 0,
            "PPG_High_Pulse_Count": 0,
            "PPG_Max_Pulse_Gap_Sec": np.nan,
            "PPG_Interval_Ratio": np.nan,
        }
    distance = max(1, int(round(min_peak_distance_sec * fs)))
    peaks, _ = find_peaks(
        pulse_band,
        distance=distance,
        prominence=max(1e-8, 0.10 * spread),
    )
    peaks = np.asarray(peaks, dtype=int)
    if len(peaks) < 2:
        return quality_map, {
            "PPG_SQI": np.nan,
            "PPG_Pulse_Count": int(len(peaks)),
            "PPG_Usable_Pulse_Count": 0,
            "PPG_High_Pulse_Count": 0,
            "PPG_Max_Pulse_Gap_Sec": np.nan,
            "PPG_Interval_Ratio": np.nan,
        }
    intervals = np.diff(peaks).astype(float) / float(fs)
    median_interval = float(np.median(intervals))
    heart_rate = 60.0 / max(1e-8, median_interval)
    max_gap = float(np.max(intervals))
    interval_ratio = float(np.max(intervals) / max(1e-8, np.min(intervals)))

    pulses = []
    for left, right in zip(peaks[:-1], peaks[1:]):
        if right - left < 2:
            continue
        source_x = np.linspace(0.0, 1.0, right - left, endpoint=False)
        target_x = np.linspace(0.0, 1.0, 64, endpoint=False)
        pulse = np.interp(target_x, source_x, pulse_band[left:right])
        pulse = pulse - float(np.mean(pulse))
        pulse_std = float(np.std(pulse))
        if pulse_std >= 1e-8:
            pulses.append(pulse / pulse_std)

    correlations = []
    if len(pulses) >= max(2, PPG_MIN_PULSES - 1):
        matrix = np.stack(pulses, axis=0)
        template = np.mean(matrix, axis=0)
        template_norm = np.linalg.norm(template)
        if template_norm >= 1e-8:
            for pulse in matrix:
                denominator = template_norm * np.linalg.norm(pulse)
                if denominator >= 1e-8:
                    correlations.append(float(np.dot(template, pulse) / denominator))

    correlations = np.asarray(correlations, dtype=float)
    sqi = float(np.mean(correlations)) if len(correlations) else np.nan
    interval_ok = (
        (60.0 / np.maximum(intervals, 1e-8) >= hr_range_bpm[0])
        & (60.0 / np.maximum(intervals, 1e-8) <= hr_range_bpm[1])
        & (intervals <= PPG_MAX_GAP_SEC)
        & (intervals / max(1e-8, median_interval) <= PPG_MAX_INTERVAL_RATIO)
        & (median_interval / np.maximum(intervals, 1e-8) <= PPG_MAX_INTERVAL_RATIO)
    )
    pulse_quality = np.zeros(len(intervals), dtype=np.int8)
    for index in range(len(intervals)):
        corr = correlations[index] if index < len(correlations) else np.nan
        if not interval_ok[index] or not np.isfinite(corr):
            pulse_quality[index] = 0
        elif corr >= PPG_SQI_THRESHOLD:
            pulse_quality[index] = 2
        elif corr >= PPG_USABLE_SQI_THRESHOLD:
            pulse_quality[index] = 1

    for index, (left, right) in enumerate(zip(peaks[:-1], peaks[1:])):
        if pulse_quality[index] > 0:
            quality_map[left:right] = pulse_quality[index]
    quality_map[~np.asarray(structural_mask, dtype=bool)] = 0
    return quality_map, {
        "PPG_SQI": sqi,
        "PPG_Pulse_Count": int(len(peaks)),
        "PPG_Usable_Pulse_Count": int((pulse_quality > 0).sum()),
        "PPG_High_Pulse_Count": int((pulse_quality == 2).sum()),
        "PPG_Max_Pulse_Gap_Sec": max_gap,
        "PPG_Interval_Ratio": interval_ratio,
        "PPG_Median_HR_BPM": heart_rate,
    }


def build_structural_map(
    signal: np.ndarray,
    finite_mask: np.ndarray,
    fs: int,
    unit_sec: float,
) -> np.ndarray:
    clean, _ = sanitize_signal(signal)
    valid = np.asarray(finite_mask, dtype=bool).copy()
    unit_len = max(1, int(round(unit_sec * fs)))
    for start in range(0, len(clean), unit_len):
        end = min(len(clean), start + unit_len)
        block = clean[start:end]
        block_valid = valid[start:end]
        robust_range = (
            float(np.percentile(block, 95) - np.percentile(block, 5))
            if len(block)
            else 0.0
        )
        invalid = bool(
            block_valid.mean() < TECHNICAL_MIN_FINITE_RATIO
            or len(block) < 2
            or float(np.std(block)) < 1e-8
            or robust_range < 1e-8
            or flat_fraction(block) > TECHNICAL_MAX_FLAT_FRACTION
            or saturation_fraction(block) > TECHNICAL_MAX_SATURATION_FRACTION
        )
        if invalid:
            valid[start:end] = False
    return valid


def build_ppg_quality_map(
    ppg_signal: np.ndarray,
    ppg_finite_mask: np.ndarray,
    fs: int,
    hr_range_bpm: Tuple[float, float] = (PPG_HR_MIN_BPM, PPG_HR_MAX_BPM),
    min_peak_distance_sec: float = PPG_MIN_PEAK_DISTANCE_SEC,
) -> Tuple[np.ndarray, dict]:
    structural = build_structural_map(
        ppg_signal,
        ppg_finite_mask,
        fs,
        PPG_QC_UNIT_SEC,
    )
    quality_map, diagnostics = pulse_template_diagnostics(
        ppg_signal,
        structural,
        fs,
        hr_range_bpm,
        min_peak_distance_sec,
    )
    diagnostics.update(
        {
            "PPG_Structural_Valid_Fraction": float(structural.mean())
            if len(structural)
            else 0.0,
            "PPG_Usable_Sample_Fraction": float(
                (quality_map > 0).mean()
            )
            if len(quality_map)
            else 0.0,
            "PPG_High_Sample_Fraction": float(
                (quality_map == 2).mean()
            )
            if len(quality_map)
            else 0.0,
        }
    )
    return quality_map, diagnostics


def build_rsp_quality_map(
    rsp_signal: np.ndarray,
    rsp_finite_mask: np.ndarray,
    fs: int,
) -> Tuple[np.ndarray, dict]:
    quality_map = build_structural_map(
        rsp_signal,
        rsp_finite_mask,
        fs,
        PPG_QC_UNIT_SEC,
    )
    return quality_map.astype(np.int8), {
        "RSP_Structural_Valid_Fraction": float(quality_map.mean())
        if len(quality_map)
        else 0.0,
    }


# ---------------------------------------------------------------------------
# Breath detection, reference quality and breath-event targets
# ---------------------------------------------------------------------------

def _local_extrema(x: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Strict local maxima and minima, as in RRest ref_cto."""
    slope = np.diff(x)
    peaks = np.flatnonzero((slope[:-1] > 0) & (slope[1:] < 0)) + 1
    troughs = np.flatnonzero((slope[:-1] < 0) & (slope[1:] > 0)) + 1
    return peaks, troughs


def _count_between(events: np.ndarray, marks: np.ndarray) -> np.ndarray:
    """Number of events strictly between consecutive marks."""
    return np.searchsorted(events, marks[1:]) - np.searchsorted(
        events, marks[:-1], side="right"
    )


def count_orig_breaths(
    signal: np.ndarray,
    fs: int,
) -> Tuple[np.ndarray, np.ndarray, float, float]:
    """Count-orig breath detection on one analysis window.

    Follows RRest ref_cto (Schaefer and Kratky 2008): after z-scoring, peaks
    above 0.2 x the 75th percentile of all peak values are candidate breath
    apexes, and consecutive apexes without a dip below zero between them are
    one breath (the higher apex is kept), so a hump is never a breath.

    Two changes make it valid for capnograms as well as impedance signals:
    * a cycle is clean when the signal crosses zero upwards exactly once
      between consecutive apexes (RRest asks for exactly one sub-zero
      trough, which rejects every cycle of a flat-bottomed capnogram whose
      band-passed trough is W-shaped);
    * each breath is timed at its upward zero crossing (onset) rather than
      its apex. On long capnogram plateaus the band-passed apex jumps between
      the start and the end of the plateau, which moves breaths across minute
      edges; the onset is the point RRest uses for CapnoBase (co2_zex) and
      corresponds to the expert "start of expiration" labels.

    Returns (apex indices, onset indices with -1 where the onset lies before
    the window, cycle-based RR from clean cycles, share of the breathing time
    in clean cycles).
    """
    x, _ = sanitize_signal(signal)
    empty = np.empty(0, dtype=np.int64)
    spread = float(np.std(x)) if len(x) else 0.0
    if len(x) < 3 or spread < 1e-8:
        return empty, empty, np.nan, np.nan
    z = (x - float(np.mean(x))) / spread
    peaks, troughs = _local_extrema(z)
    if len(peaks) == 0:
        return empty, empty, np.nan, np.nan
    threshold = CTO_PEAK_FRACTION * float(np.quantile(z[peaks], CTO_PEAK_QUANTILE))
    rel_peaks = peaks[z[peaks] > threshold]
    rel_troughs = troughs[z[troughs] < 0.0]
    if len(rel_peaks) == 0:
        return empty, empty, np.nan, np.nan
    apexes = [int(rel_peaks[0])]
    for peak, n_troughs in zip(rel_peaks[1:], _count_between(rel_troughs, rel_peaks)):
        if n_troughs >= 1:
            apexes.append(int(peak))
        elif z[peak] > z[apexes[-1]]:
            apexes[-1] = int(peak)
    apexes = np.asarray(apexes, dtype=np.int64)
    upward = np.flatnonzero((z[:-1] < 0.0) & (z[1:] >= 0.0)) + 1
    previous = np.concatenate([[-1], apexes[:-1]])
    onsets = np.full(len(apexes), -1, dtype=np.int64)
    for index, (apex, before) in enumerate(zip(apexes, previous)):
        candidates = upward[(upward > before) & (upward <= apex)]
        if len(candidates):
            onsets[index] = int(candidates[-1])
    if len(apexes) < 2:
        return apexes, onsets, np.nan, np.nan
    clean = _count_between(upward, apexes) == 1
    cycles = np.diff(apexes) / float(fs)
    rr = float(60.0 / np.mean(cycles[clean])) if clean.any() else np.nan
    return apexes, onsets, rr, float(cycles[clean].sum() / cycles.sum())


def fft_rr_bpm(
    signal: np.ndarray,
    fs: int,
    rr_band_bpm: Tuple[float, float] = RR_BAND_BPM,
    subharmonic_check: bool = False,
) -> float:
    """Spectral-peak RR (RRest ref_fft: detrend + Hamming) within the RR band.

    With subharmonic_check, a peak at f/2 or f/3 holding at least 30% of the
    main peak's power is taken as the fundamental: a trapezoid capnogram at
    slow rates puts more power in its 2nd harmonic than in its fundamental.
    """
    x, _ = sanitize_signal(signal)
    if len(x) < 8 or float(np.std(x)) < 1e-8:
        return np.nan
    windowed = detrend(x) * np.hamming(len(x))
    n_fft = max(FFT_MIN_POINTS, 1 << int(math.ceil(math.log2(len(x)))))
    power = np.abs(np.fft.rfft(windowed, n=n_fft)) ** 2
    freqs_bpm = np.fft.rfftfreq(n_fft, d=1.0 / float(fs)) * 60.0
    in_band = (freqs_bpm >= rr_band_bpm[0]) & (freqs_bpm <= rr_band_bpm[1])
    if not in_band.any():
        return np.nan
    best = int(np.flatnonzero(in_band)[np.argmax(power[in_band])])
    rr = float(freqs_bpm[best])
    if subharmonic_check:
        resolution = float(freqs_bpm[1] - freqs_bpm[0])
        for divisor in (3, 2):
            target = rr / divisor
            near = in_band & (np.abs(freqs_bpm - target) <= max(2 * resolution, 0.05 * target))
            if near.any() and power[near].max() >= FFT_SUBHARMONIC_POWER_RATIO * power[best]:
                return float(freqs_bpm[near][np.argmax(power[near])])
    return rr


def reference_agreement(
    signal: np.ndarray,
    fs: int,
    rr_band_bpm: Tuple[float, float] = RR_BAND_BPM,
    subharmonic_check: bool = False,
    label_times: Optional[np.ndarray] = None,
) -> Tuple[bool, str, float, float]:
    """Respiratory quality of one reference window (reference signal only).

    Without expert labels this is the RRest agreement rule: count-orig RR and
    FFT RR within 2 bpm. With expert labels (sample indices relative to the
    window) the waveform's count-orig RR must instead agree with the label
    RR, i.e. the band-passed target reproduces the annotated breaths. In both
    cases at least 75% of the breathing time must be in clean cycles and the
    RR must lie in the profile's RR band.

    Returns (valid, exclusion_reason, count-orig RR, comparison RR).
    """
    _, _, rr_cto, clean_time = count_orig_breaths(signal, fs)
    if label_times is not None:
        labels = np.sort(np.asarray(label_times, dtype=np.float64))
        rr_other = (
            float(60.0 * fs / np.mean(np.diff(labels))) if len(labels) >= 2 else np.nan
        )
    else:
        rr_other = fft_rr_bpm(signal, fs, rr_band_bpm, subharmonic_check)
    if not (np.isfinite(rr_cto) and np.isfinite(rr_other)):
        return False, "ref_rr_undefined", rr_cto, rr_other
    if clean_time < REF_MIN_VALID_CYCLE_FRACTION:
        return False, "ref_irregular_cycles", rr_cto, rr_other
    if abs(rr_cto - rr_other) >= REF_AGREEMENT_BPM:
        return False, "ref_disagreement", rr_cto, rr_other
    rr_mean = 0.5 * (rr_cto + rr_other)
    if not (rr_band_bpm[0] <= rr_mean <= rr_band_bpm[1]):
        return False, "ref_out_of_band", rr_cto, rr_other
    return True, "", rr_cto, rr_other


def build_reference_resp_quality_map(
    ref_filtered: np.ndarray,
    fs: int,
    rr_band_bpm: Tuple[float, float] = RR_BAND_BPM,
    subharmonic_check: bool = False,
    label_times: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, dict]:
    """Sample-level respiratory quality of the reference RSP.

    The quality rule is evaluated on 32 s windows every 8 s; a sample is
    usable when at least half of the windows covering it pass. Only the
    reference (and its expert labels, if any) is used, never the PPG or a
    model output.
    """
    length = len(ref_filtered)
    window = int(round(REF_QC_WINDOW_SEC * fs))
    hop = int(round(REF_QC_HOP_SEC * fs))
    votes = np.zeros(length, dtype=np.float64)
    cover = np.zeros(length, dtype=np.float64)
    reasons = {
        "ref_rr_undefined": 0,
        "ref_irregular_cycles": 0,
        "ref_disagreement": 0,
        "ref_out_of_band": 0,
    }
    passed = 0
    starts = list(range(0, length - window + 1, hop)) if length >= window else []
    if starts and starts[-1] != length - window:
        starts.append(length - window)
    for start in starts:
        window_labels = None
        if label_times is not None:
            inside = label_times[(label_times >= start) & (label_times < start + window)]
            window_labels = inside - start
        valid, reason, _, _ = reference_agreement(
            ref_filtered[start : start + window],
            fs,
            rr_band_bpm,
            subharmonic_check,
            window_labels,
        )
        cover[start : start + window] += 1.0
        if valid:
            votes[start : start + window] += 1.0
            passed += 1
        else:
            reasons[reason] += 1
    quality = (cover > 0) & (votes >= 0.5 * cover)
    diagnostics = {
        "RSP_Resp_QC_Windows": int(len(starts)),
        "RSP_Resp_QC_Pass_Windows": int(passed),
        "RSP_Resp_Quality_Fraction": float(quality.mean()) if length else 0.0,
        **{f"RSP_Resp_QC_{key}": int(value) for key, value in reasons.items()},
    }
    return quality, diagnostics


def detect_breaths_continuous(filtered: np.ndarray, fs: int) -> np.ndarray:
    """Breath onsets of a continuous filtered signal.

    Count-orig runs on 60 s blocks with 10 s of context on each side, so the
    thresholds stay local when breathing depth drifts; a breath belongs to
    the block that contains its onset (upward zero crossing). Detections
    closer than 0.5 s are merged as block-edge duplicates.
    """
    length = len(filtered)
    block = int(round(BREATH_BLOCK_SEC * fs))
    margin = int(round(BREATH_BLOCK_MARGIN_SEC * fs))
    found = []
    for block_start in range(0, length, block):
        block_end = min(length, block_start + block)
        low = max(0, block_start - margin)
        high = min(length, block_end + margin)
        _, onsets, _, _ = count_orig_breaths(filtered[low:high], fs)
        onsets = onsets[onsets >= 0] + low
        found.append(onsets[(onsets >= block_start) & (onsets < block_end)])
    if not found:
        return np.empty(0, dtype=np.int64)
    breaths = np.unique(np.concatenate(found)).astype(np.int64)
    if len(breaths) > 1:
        keep = np.concatenate([[True], np.diff(breaths) >= int(round(0.5 * fs))])
        breaths = breaths[keep]
    return breaths


def _thresholded_bump_area(threshold: float) -> float:
    """Area of max(exp(-x^2/2) - threshold, 0) for unit sigma."""
    x = np.linspace(-8.0, 8.0, 160001)
    y = np.clip(np.exp(-0.5 * x * x) - threshold, 0.0, None)
    return float(np.sum(y) * (x[1] - x[0]))


COUNT_BUMP_AREA_FACTOR = _thresholded_bump_area(EVENT_COUNT_THRESHOLD)


def event_sigmas_samples(peaks: np.ndarray, fs: float) -> np.ndarray:
    """Gaussian width of every breath: a share of its local breath period.

    The period is the mean of the intervals to the neighbouring breaths;
    sigma = EVENT_SIGMA_PERIOD_FRACTION x period, limited to
    EVENT_SIGMA_LIMITS_SEC (0.15-0.5 s), so a slow breath is one broad bump
    and the bumps of fast breaths stay apart.
    """
    peaks = np.asarray(peaks, dtype=np.float64)
    if len(peaks) < 2:
        return np.full(len(peaks), EVENT_SIGMA_SEC * float(fs))
    gaps = np.diff(peaks)
    left = np.concatenate([[gaps[0]], gaps])
    right = np.concatenate([gaps, [gaps[-1]]])
    period_sec = 0.5 * (left + right) / float(fs)
    low, high = EVENT_SIGMA_LIMITS_SEC
    return np.clip(EVENT_SIGMA_PERIOD_FRACTION * period_sec, low, high) * float(fs)


def breath_event_target_and_norm(
    peaks: np.ndarray,
    start: float,
    length: int,
    fs: float,
) -> Tuple[np.ndarray, float]:
    """Breath heatmap of one window and its count normalizer.

    Peaks may be fractional sample positions (speed-augmented windows). The
    normalizer is the thresholded area of one bump with the window's mean
    width, so sum(max(heatmap - threshold, 0)) / normalizer counts breaths.
    """
    peaks = np.sort(np.asarray(peaks, dtype=np.float64))
    default_norm = COUNT_BUMP_AREA_FACTOR * EVENT_SIGMA_SEC * float(fs)
    if len(peaks) == 0:
        return np.zeros(length, dtype=np.float32), default_norm
    sigmas = event_sigmas_samples(peaks, fs)
    reach = 4.0 * sigmas
    near = (peaks + reach >= start) & (peaks - reach < start + length)
    if not np.any(near):
        return np.zeros(length, dtype=np.float32), default_norm
    samples = np.arange(length, dtype=np.float64) + float(start)
    bumps = np.exp(
        -0.5 * ((samples[None, :] - peaks[near, None]) / sigmas[near, None]) ** 2
    )
    norm = COUNT_BUMP_AREA_FACTOR * float(np.mean(sigmas[near]))
    return bumps.max(axis=0).astype(np.float32), norm


def breath_event_target(
    peaks: np.ndarray,
    start: int,
    length: int,
    fs: int,
) -> np.ndarray:
    """Breath heatmap (height 1 at every reference breath onset)."""
    return breath_event_target_and_norm(peaks, start, length, fs)[0]


def preprocess_segment(
    ppg_raw: np.ndarray,
    rsp_raw: np.ndarray,
    fs: int,
    spec: WindowSpec,
    segment_id: int,
    ppg_valid_mask: Optional[np.ndarray] = None,
    rsp_valid_mask: Optional[np.ndarray] = None,
    profile: Optional[dict] = None,
    label_samples: Optional[np.ndarray] = None,
    ppg_artifact_mask: Optional[np.ndarray] = None,
    rsp_artifact_mask: Optional[np.ndarray] = None,
) -> Optional[dict]:
    """Quality maps, fixed ground truth and kept windows of one segment.

    `profile` is the dataset analysis profile; `label_samples` are expert
    breath onsets (sample indices at `fs`) when the dataset provides them;
    the artifact masks mark expert-labelled artifacts (True = artifact).
    """
    profile = ANALYSIS_PROFILES["adult_fixed_band"] if profile is None else profile
    ppg_source = np.asarray(ppg_raw, dtype=np.float64).reshape(-1)
    ppg_raw, _ = sanitize_signal(ppg_source)
    rsp_raw_arr = np.asarray(rsp_raw, dtype=np.float64).reshape(-1)
    length = min(len(ppg_raw), len(rsp_raw_arr))
    ppg_raw = ppg_raw[:length]
    rsp_raw_arr = rsp_raw_arr[:length]
    if ppg_valid_mask is None:
        ppg_valid_mask = np.ones(length, dtype=bool)
    else:
        ppg_valid_mask = np.asarray(ppg_valid_mask, dtype=bool)[:length]
    if rsp_valid_mask is None:
        rsp_valid_mask = np.ones(length, dtype=bool)
    else:
        rsp_valid_mask = np.asarray(rsp_valid_mask, dtype=bool)[:length]
    window_len = int(round(spec.window_sec * fs))
    if length < window_len:
        return None

    def as_artifact_mask(mask: Optional[np.ndarray]) -> np.ndarray:
        out = np.zeros(length, dtype=bool)
        if mask is not None:
            mask = np.asarray(mask, dtype=bool)[:length]
            out[: len(mask)] = mask
        return out

    # Expert-labelled artifacts are unusable samples of their signal.
    ppg_artifact = as_artifact_mask(ppg_artifact_mask)
    rsp_artifact = as_artifact_mask(rsp_artifact_mask)
    ppg_valid_mask = ppg_valid_mask & ~ppg_artifact
    rsp_valid_mask = rsp_valid_mask & ~rsp_artifact

    # Analysis band: fixed for the adult profile; for the capnography profile
    # the upper edge follows the PPG heart rate (PPG only, so it is also
    # available at inference).
    ppg_hr_hz = estimate_ppg_heart_rate_hz(ppg_raw, fs, profile["ppg_hr_range_bpm"])
    band = resolve_respiratory_band(profile, ppg_hr_hz)
    ppg_low = respiratory_filter(ppg_raw, fs, band, profile)
    rsp_low = respiratory_filter(rsp_raw_arr, fs, band, profile)
    use_phase_target = bool(
        profile.get("capnography_target") == "phase"
        and label_samples is not None
        and len(label_samples) >= 3
    )
    if profile["max_target_lag_sec"] > 0:
        # Half a breath period from the expert labels (or breaths detected on
        # the unshifted reference), never below the profile minimum.
        if label_samples is not None and len(label_samples) >= 3:
            period_sec = float(np.median(np.diff(np.sort(label_samples)))) / fs
        else:
            onsets = detect_breaths_continuous(rsp_low, fs)
            period_sec = float(np.median(np.diff(onsets))) / fs if len(onsets) >= 3 else 0.0
        max_lag_sec = float(
            np.clip(0.5 * period_sec, profile["max_target_lag_sec"], MAX_TARGET_LAG_SEC)
        )
        # The waveform target (label phase waveform or capnogram) is aligned
        # to the PPG; the capnogram reference moves by the same lag.
        unaligned_target = (
            label_phase_waveform(label_samples, length) if use_phase_target else rsp_low
        )
        target_signal, lag, sign, alignment_corr = align_target_lag_and_sign(
            ppg_low,
            unaligned_target,
            fs,
            max_lag_sec,
            allow_sign_flip=not profile["fixed_target_polarity"],
        )
        rsp_low = target_signal if not use_phase_target else (
            sign * shift_signal(rsp_low, lag)
        ).astype(np.float32)
    elif profile.get("target_polarity") == "as_recorded":
        # Polarity as recorded, no lag: the zero-lag correlation is kept only
        # as a diagnostic.
        sign = 1.0
        target_signal = rsp_low
        lag = 0
        alignment_corr = safe_pcc(ppg_low, rsp_low)
    else:
        centered = (ppg_low - np.mean(ppg_low)) * (rsp_low - np.mean(rsp_low))
        sign = -1.0 if float(np.mean(centered)) < 0.0 else 1.0
        rsp_low = align_target_orientation(ppg_low, rsp_low)
        target_signal = rsp_low
        lag = 0
        alignment_corr = safe_pcc(ppg_low, rsp_low)
    beat_distance = (
        0.6 / ppg_hr_hz
        if profile["ppg_spectral_beat_distance"] and np.isfinite(ppg_hr_hz)
        else PPG_MIN_PEAK_DISTANCE_SEC
    )
    ppg_quality_map, ppg_diagnostics = build_ppg_quality_map(
        ppg_raw,
        ppg_valid_mask,
        fs,
        profile["ppg_hr_range_bpm"],
        beat_distance,
    )
    rsp_quality_map, rsp_diagnostics = build_rsp_quality_map(
        rsp_raw_arr,
        rsp_valid_mask,
        fs,
    )
    # The reference was shifted by `lag`; shift its validity maps with it.
    rsp_quality_map = shift_signal(rsp_quality_map, lag, fill=0)
    rsp_valid_mask = shift_signal(rsp_valid_mask, lag, fill=False)
    rsp_artifact = shift_signal(rsp_artifact, lag, fill=False)

    # Fixed ground truth: expert breath labels when available (shifted with
    # the reference), otherwise breaths detected once on the continuous
    # reference. Its respiratory quality (reference and labels only) decides
    # which windows may be used for training and validation.
    detected_breaths = detect_breaths_continuous(rsp_low, fs)
    labels = None
    if label_samples is not None and len(label_samples) >= 2:
        labels = np.round(np.asarray(label_samples, dtype=np.float64)).astype(np.int64) - lag
        labels = np.unique(labels[(labels >= 0) & (labels < length)])
    label_based = labels is not None and len(labels) >= 2
    reference_breaths = labels if label_based else detected_breaths
    resp_quality_map, resp_diagnostics = build_reference_resp_quality_map(
        rsp_low,
        fs,
        profile["rr_band_bpm"],
        profile["fft_subharmonic_check"],
        reference_breaths if label_based else None,
    )
    reference_structural = rsp_quality_map > 0
    segment_diagnostics = {
        "Segment_ID": int(segment_id),
        "Profile": profile_name(profile),
        "Resp_Band_Low_Hz": band[0],
        "Resp_Band_High_Hz": band[1],
        "PPG_HR_BPM_Spectral": float(60.0 * ppg_hr_hz) if np.isfinite(ppg_hr_hz) else np.nan,
        "Target_Lag_Sec": float(lag / float(fs)),
        "Target_Sign": float(sign),
        "Target_Alignment_Corr": float(alignment_corr),
        "Expert_Labels_Used": bool(label_based),
        "Waveform_Target": "label_phase" if use_phase_target else "reference_signal",
        "GT_Breaths": int(len(reference_breaths)),
        "Detected_Breaths": int(len(detected_breaths)),
        "PPG_Artifact_Label_Sec": float(ppg_artifact.sum() / float(fs)),
        "RSP_Artifact_Label_Sec": float(rsp_artifact.sum() / float(fs)),
    }
    train_quality_map = resp_quality_map & reference_structural
    starts = extract_window_starts(
        length,
        fs,
        spec.window_sec,
        spec.stride_sec,
    )

    model_input = (ppg_raw if PPG_INPUT == "raw" else ppg_low).astype(np.float32)
    low_kept = []
    rsp_kept = []
    event_kept = []
    event_norm_kept = []
    train_ok_kept = []
    kept_starts = []
    quality_rows = []
    for index, start in enumerate(starts):
        end = int(start) + window_len
        ppg_quality = ppg_quality_map[int(start):end]
        rsp_quality = rsp_quality_map[int(start):end]
        ppg_usable_ratio = float((ppg_quality > 0).mean())
        ppg_high_ratio = float((ppg_quality == 2).mean())
        rsp_usable_ratio = float((rsp_quality > 0).mean())
        passed = bool(
            ppg_usable_ratio >= MIN_USABLE_PROPORTION
            and rsp_usable_ratio >= MIN_USABLE_PROPORTION
        )
        resp_usable_ratio = float(train_quality_map[int(start):end].mean())
        ref_artifact_ratio = float(rsp_artifact[int(start):end].mean())
        # Labels may be missing inside a labelled reference artifact, so such
        # windows would teach the event head "no breath" there.
        train_ok = bool(
            passed
            and resp_usable_ratio >= MIN_USABLE_PROPORTION
            and ref_artifact_ratio == 0.0
        )
        diagnostics = {
            **ppg_diagnostics,
            **rsp_diagnostics,
            **resp_diagnostics,
            **segment_diagnostics,
            "PPG_Usable_Proportion": ppg_usable_ratio,
            "PPG_High_Quality_Proportion": ppg_high_ratio,
            "RSP_Usable_Proportion": rsp_usable_ratio,
            "Structural_Invalid_Proportion": float(
                1.0 - min(ppg_usable_ratio, rsp_usable_ratio)
            ),
        }
        diagnostics.update(
            {
                "Segment_ID": int(segment_id),
                "Candidate_Window_Index": int(index),
                "Window_Start_Sec": float(start / float(fs)),
                "Window_End_Sec": float(end / float(fs)),
                "Target_Finite_Ratio": float(
                    rsp_valid_mask[int(start):end].mean()
                ),
                "Target_Technical_Valid": int(
                    rsp_usable_ratio >= MIN_USABLE_PROPORTION
                ),
                "Kept": int(passed),
                "RSP_Resp_Usable_Proportion": resp_usable_ratio,
                "PPG_Artifact_Label_Proportion": float(
                    ppg_artifact[int(start):end].mean()
                ),
                "RSP_Artifact_Label_Proportion": ref_artifact_ratio,
                "Train_OK": int(train_ok),
            }
        )
        quality_rows.append(diagnostics)
        if passed:
            # Model input window: the PPG itself (min-max normalized per
            # window in the dataset) or, with --ppg-input resp_band, the
            # respiratory-band PPG of 261001_RRWaveNet_fv.
            low_kept.append(model_input[int(start):end])
            rsp_kept.append(target_signal[int(start):end])
            event_window, event_norm = breath_event_target_and_norm(
                reference_breaths, int(start), window_len, fs
            )
            event_kept.append(event_window)
            event_norm_kept.append(event_norm)
            train_ok_kept.append(train_ok)
            kept_starts.append(int(start))

    reference = {
        "signal": rsp_low.astype(np.float32),
        "breaths": reference_breaths,
        "detected_breaths": detected_breaths,
        "label_based": bool(label_based),
        "structural_valid": reference_structural,
        "artifact_mask": rsp_artifact if rsp_artifact.any() else None,
        "ppg_artifact_mask": ppg_artifact if ppg_artifact.any() else None,
        "band_hz": band,
        "profile": profile,
        "diagnostics": segment_diagnostics,
        # Continuous arrays for speed augmentation: model input, waveform
        # target and the samples usable for training (reference quality,
        # no labelled reference artifact, usable PPG).
        "model_input": model_input,
        "target_signal": np.asarray(target_signal, dtype=np.float32),
        "train_map": train_quality_map & ~rsp_artifact & (ppg_quality_map > 0),
    }
    if not low_kept:
        return {
            "low": np.empty((0, window_len), dtype=np.float32),
            "rsp": np.empty((0, window_len), dtype=np.float32),
            "event": np.empty((0, window_len), dtype=np.float32),
            "event_norm": np.empty(0, dtype=np.float32),
            "train_ok": np.empty(0, dtype=bool),
            "starts": np.empty(0, dtype=np.int64),
            "segment_ids": np.empty(0, dtype=np.int64),
            "quality": quality_rows,
            "reference": reference,
            "segment_id": int(segment_id),
            "original_length": int(length),
            "fs": int(fs),
        }
    return {
        "low": np.stack(low_kept).astype(np.float32),
        "rsp": np.stack(rsp_kept).astype(np.float32),
        "event": np.stack(event_kept).astype(np.float32),
        "event_norm": np.asarray(event_norm_kept, dtype=np.float32),
        "train_ok": np.asarray(train_ok_kept, dtype=bool),
        "starts": np.asarray(kept_starts, dtype=np.int64),
        "segment_ids": np.full(len(kept_starts), segment_id, dtype=np.int64),
        "quality": quality_rows,
        "reference": reference,
        "segment_id": int(segment_id),
        "original_length": int(length),
        "fs": int(fs),
    }


def subject_reference_diagnostics(references: Dict[int, dict], fs: int) -> dict:
    """Breathing-rate profile of one subject's ground truth.

    Reports where the subject's breathing lies relative to the v16 band
    (6-30 bpm), so subjects whose RR the v16 processing could not represent
    can be identified, and how well breath detection on the reference agrees
    with the expert labels (per-minute count error) when labels exist.
    """
    rates = []
    detection_errors = []
    label_based = False
    for reference in references.values():
        breaths = np.asarray(reference["breaths"], dtype=np.float64)
        if len(breaths) > 1:
            rates.append(60.0 * fs / np.diff(breaths))
        if reference["label_based"]:
            label_based = True
            detected = np.asarray(reference["detected_breaths"])
            minute = int(round(60.0 * fs))
            for start in range(0, len(reference["signal"]) - minute + 1, minute):
                in_labels = int(((breaths >= start) & (breaths < start + minute)).sum())
                in_detected = int(((detected >= start) & (detected < start + minute)).sum())
                detection_errors.append(abs(in_labels - in_detected))
    rates = np.concatenate(rates) if rates else np.empty(0)
    diagnostics = {
        "GT_Source": "expert_labels" if label_based else "detected_breaths",
        "GT_RR_Median_BPM": float(np.median(rates)) if len(rates) else np.nan,
        "GT_RR_P05_BPM": float(np.percentile(rates, 5)) if len(rates) else np.nan,
        "GT_RR_P95_BPM": float(np.percentile(rates, 95)) if len(rates) else np.nan,
        "GT_RR_Below_6_Fraction": float(np.mean(rates < 6.0)) if len(rates) else np.nan,
        "GT_RR_Above_30_Fraction": float(np.mean(rates > 30.0)) if len(rates) else np.nan,
        "Detection_vs_Label_Count_MAE_per_min": (
            float(np.mean(detection_errors)) if detection_errors else np.nan
        ),
    }
    segment_rows = [reference["diagnostics"] for reference in references.values()]
    for key in (
        "Resp_Band_Low_Hz",
        "Resp_Band_High_Hz",
        "PPG_HR_BPM_Spectral",
        "Target_Lag_Sec",
        "Target_Sign",
        "Target_Alignment_Corr",
    ):
        values = [row[key] for row in segment_rows if np.isfinite(row[key])]
        diagnostics[key] = float(np.mean(values)) if values else np.nan
    return diagnostics


def combine_segments(
    segments: Sequence[dict],
    spec: WindowSpec,
    fs: int,
) -> Optional[dict]:
    valid = [segment for segment in segments if len(segment["low"])]
    if not valid:
        return None
    low = np.concatenate([segment["low"] for segment in valid], axis=0)
    rsp = np.concatenate([segment["rsp"] for segment in valid], axis=0)
    event = np.concatenate([segment["event"] for segment in valid], axis=0)
    event_norm = np.concatenate([segment["event_norm"] for segment in valid], axis=0)
    train_ok = np.concatenate([segment["train_ok"] for segment in valid], axis=0)
    starts = np.concatenate([segment["starts"] for segment in valid], axis=0)
    segment_ids = np.concatenate(
        [segment["segment_ids"] for segment in valid], axis=0
    )
    quality = []
    for segment in segments:
        quality.extend(segment["quality"])
    references = {
        int(segment["segment_id"]): segment["reference"] for segment in valid
    }
    return {
        "low": low.astype(np.float32),
        "rsp": rsp.astype(np.float32),
        "event": event.astype(np.float32),
        "event_norm": event_norm.astype(np.float32),
        "train_ok": train_ok.astype(bool),
        "starts": starts.astype(np.int64),
        "segment_ids": segment_ids.astype(np.int64),
        "references": references,
        "reference_diagnostics": subject_reference_diagnostics(references, fs),
        "quality": quality,
        "fs": int(fs),
        "window_sec": float(spec.window_sec),
        "stride_sec": float(spec.stride_sec),
    }


# ---------------------------------------------------------------------------
# Dataset readers
# ---------------------------------------------------------------------------

def _unwrap_mat(value):
    arr = np.asarray(value)
    while arr.dtype == object and arr.size == 1:
        arr = np.asarray(arr.item())
    return arr


def _first_mat_payload(mat: dict):
    keys = [key for key in mat if not key.startswith("_")]
    if not keys:
        raise KeyError("No non-private MATLAB variable found")
    payload = _unwrap_mat(mat[keys[0]])
    if payload.ndim >= 2 and payload.shape[0] == 1 and payload.shape[1] == 1:
        payload = payload[0, 0]
    return payload


def _extract_signal_and_fs(
    payload,
    default_fs: int,
) -> Tuple[np.ndarray, float]:
    signal = np.asarray(
        _unwrap_mat(payload["data"]),
        dtype=np.float64,
    ).reshape(-1)
    try:
        fs = float(np.asarray(_unwrap_mat(payload["srate"])).squeeze())
    except Exception:
        fs = float(default_fs)
    return signal.astype(np.float32), fs


def spans_to_mask(spans_sec: np.ndarray, length: int, fs: float) -> np.ndarray:
    """Boolean mask (True inside any (start, end) span given in seconds)."""
    mask = np.zeros(length, dtype=bool)
    for start, end in np.asarray(spans_sec, dtype=np.float64).reshape(-1, 2):
        low = max(0, int(np.floor(start * fs)))
        high = min(length, int(np.ceil(end * fs)) + 1)
        if high > low:
            mask[low:high] = True
    return mask


def preprocess_subject_from_pairs(
    pairs: Sequence[Tuple[np.ndarray, np.ndarray, float, float]],
    spec: WindowSpec,
    profile: Optional[dict] = None,
    pair_labels: Optional[Sequence[Optional[np.ndarray]]] = None,
    pair_artifacts: Optional[Sequence[Optional[Tuple[np.ndarray, np.ndarray]]]] = None,
) -> Optional[dict]:
    """Preprocess (PPG, RSP, PPG fs, RSP fs) pairs of one subject.

    `pair_labels` holds, per pair, expert breath onsets as sample indices at
    the RSP sampling rate (or None); `pair_artifacts` holds, per pair,
    expert artifact spans in seconds as (PPG spans, RSP spans), each of
    shape (k, 2) (or None).
    """
    profile = ANALYSIS_PROFILES["adult_fixed_band"] if profile is None else profile
    segments = []
    quality_segment_id = 0
    for pair_index, (ppg, rsp, ppg_fs, rsp_fs) in enumerate(pairs):
        labels = None
        if pair_labels is not None and pair_labels[pair_index] is not None:
            labels = np.asarray(pair_labels[pair_index], dtype=np.float64) * (
                TARGET_FS / float(rsp_fs)
            )
        if int(round(ppg_fs)) != int(round(rsp_fs)):
            rsp = resample_signal_poly(rsp, rsp_fs, ppg_fs)
        ppg_target, ppg_mask = maybe_resample_with_mask(
            ppg,
            ppg_fs,
            TARGET_FS,
            profile["resampler"],
        )
        rsp_target, rsp_mask = maybe_resample_with_mask(
            rsp,
            ppg_fs,
            TARGET_FS,
            profile["resampler"],
        )
        ppg_artifact = rsp_artifact = None
        if pair_artifacts is not None and pair_artifacts[pair_index] is not None:
            ppg_spans, rsp_spans = pair_artifacts[pair_index]
            ppg_artifact = spans_to_mask(ppg_spans, len(ppg_target), TARGET_FS)
            rsp_artifact = spans_to_mask(rsp_spans, len(rsp_target), TARGET_FS)
        segment = preprocess_segment(
            ppg_target,
            rsp_target,
            TARGET_FS,
            spec,
            quality_segment_id,
            ppg_valid_mask=ppg_mask,
            rsp_valid_mask=rsp_mask,
            profile=profile,
            label_samples=labels,
            ppg_artifact_mask=ppg_artifact,
            rsp_artifact_mask=rsp_artifact,
        )
        if segment is not None:
            segments.append(segment)
        quality_segment_id += 1
    return combine_segments(segments, spec, TARGET_FS)


def load_bidmc_subjects(spec: WindowSpec) -> Dict[str, dict]:
    cfg = DATASET_CONFIGS["bidmc"]
    data = {}
    mat = sio.loadmat(cfg["data_path"])
    all_data = mat["data"][0]
    for index, subject_record in enumerate(all_data):
        subject = f"{index + 1:02d}"
        try:
            ppg = np.asarray(subject_record["ppg"][0, 0]["v"]).squeeze()
            rsp = np.asarray(
                subject_record["ref"][0, 0]["resp_sig"][0, 0]
                ["imp"][0, 0]["v"]
            ).squeeze()
            result = preprocess_subject_from_pairs(
                [(ppg, rsp, cfg["default_fs"], cfg["default_fs"])],
                spec,
                ANALYSIS_PROFILES[cfg["profile"]],
            )
            if result is not None:
                data[subject] = result
        except Exception as exc:
            if is_fatal_error(exc):
                raise
            print(f"[BIDMC {subject}] skipped: {exc}")
    return data


# CapnoBase IEEE TBME RR benchmark (Borealis doi:10.5683/SP2/NLB8IT), one
# group of files per case (CSV; Borealis also serves them as .tab):
#   {case}_8min_signal.csv     co2_y, pleth_y (and ecg_y), one row per sample
#   {case}_8min_param.csv      samplingrate_co2, samplingrate_pleth, ...
#   {case}_8min_labels.csv     expert annotations: co2_startexp_x,
#                              co2_startinsp_x, pleth_peak_x, *_artif_x, units_x
#   {case}_8min_meta.csv       demographics and treatment (ventilation)
#   {case}_8min_reference.csv  trends derived from the labels (RR, HR)
# MATLAB structure fields are flattened into names (labels.co2.startexp.x ->
# co2_startexp_x). An annotation field keeps all of its events in one cell
# as space-separated sample numbers with MATLAB (one-based) indexing.
CAPNOBASE_LABEL_INDEX_BASE = 1
CAPNOBASE_RECORD_SEC = 480.0
CAPNOBASE_FILE_SUFFIXES = (".csv", ".tab")
CAPNOBASE_RR_CHECK_BPM = 2.0
_NUMBER_PATTERN = re.compile(r"[-+]?(?:\d+\.?\d*|\.\d+)(?:[eE][-+]?\d+)?")


def capnobase_file(csv_dir: Path, case_id: str, kind: str) -> Optional[Path]:
    for suffix in CAPNOBASE_FILE_SUFFIXES:
        path = csv_dir / f"{case_id}_8min_{kind}{suffix}"
        if path.exists():
            return path
    return None


def _sniff_delimiter(path: Path) -> str:
    with open(path, newline="", encoding="utf-8-sig") as handle:
        first = handle.readline()
    return "\t" if first.count("\t") > first.count(",") else ","


def _normalize_field_name(name: object) -> str:
    return re.sub(r"[^0-9a-z]+", "_", str(name).lower()).strip("_")


def read_capnobase_fields(path: Path) -> Dict[str, List[str]]:
    """Fields of a CapnoBase param, meta, labels or reference file.

    Fields are read one per column (header row, then value rows) or one per
    row (name in the first cell, values after it), comma or tab separated.
    Every cell is split on whitespace, so a field whose events are stored in
    one cell (" 974 1990 ...") yields one token per event. Names are lower
    case with every non-alphanumeric run replaced by "_".
    """
    with open(path, newline="", encoding="utf-8-sig") as handle:
        rows = [
            [cell.strip() for cell in row]
            for row in csv.reader(handle, delimiter=_sniff_delimiter(path))
            if any(cell.strip() for cell in row)
        ]
    fields: Dict[str, List[str]] = {}
    if not rows:
        return fields
    header = rows[0]
    column_layout = all(len(row) <= len(header) for row in rows[1:]) and not any(
        _NUMBER_PATTERN.fullmatch(cell) for cell in header if cell
    )
    if column_layout and len(header) == 2 and len(rows) > 2:
        # A two-column "name, value" table is a row layout under a header.
        names = [row[0] for row in rows[1:] if row]
        if names and not any(_NUMBER_PATTERN.fullmatch(name) for name in names if name):
            column_layout = False
            rows = rows[1:]
    if column_layout:
        for index, name in enumerate(header):
            if name:
                fields[_normalize_field_name(name)] = [
                    token
                    for row in rows[1:]
                    if index < len(row)
                    for token in row[index].split()
                ]
    else:
        for row in rows:
            if row and row[0]:
                fields[_normalize_field_name(row[0])] = [
                    token for cell in row[1:] for token in cell.split()
                ]
    return fields


def _field_numbers(tokens: Sequence[str]) -> np.ndarray:
    return np.asarray(
        [float(match) for token in tokens for match in _NUMBER_PATTERN.findall(token)],
        dtype=np.float64,
    )


def _find_field(fields: Dict[str, List[str]], *parts: str, suffix: str = "") -> Optional[str]:
    """Shortest field name containing every part; names ending in `suffix` win."""
    names = [name for name in fields if all(part in name for part in parts)]
    if suffix:
        names = [name for name in names if name.endswith(suffix)] or names
    return min(names, key=len) if names else None


def _sampling_rate(
    fields: Dict[str, List[str]],
    signal_name: str,
    default: float,
) -> Tuple[float, str]:
    name = _find_field(fields, "samplingrate", signal_name) or _find_field(
        fields, "rate", signal_name
    )
    values = _field_numbers(fields[name]) if name else np.empty(0)
    values = values[np.isfinite(values) & (values > 0)]
    if len(values):
        return float(values[0]), name
    return float(default), "default"


def read_capnobase_signals(path: Path) -> Tuple[np.ndarray, np.ndarray, str]:
    """PPG (pleth) and capnogram (co2) columns of a signal file."""
    frame = pd.read_csv(path, sep=_sniff_delimiter(path))
    frame.columns = [_normalize_field_name(column) for column in frame.columns]

    def column(signal_name: str) -> str:
        names = [name for name in frame.columns if signal_name in name]
        names = [name for name in names if name.endswith("_y")] or names
        if not names:
            raise KeyError(f"{path.name}: no {signal_name} column in {list(frame.columns)}")
        return min(names, key=len)

    ppg_column, co2_column = column("pleth"), column("co2")
    ppg = pd.to_numeric(frame[ppg_column], errors="coerce").to_numpy(np.float32)
    co2 = pd.to_numeric(frame[co2_column], errors="coerce").to_numpy(np.float32)
    return ppg, co2, f"{ppg_column},{co2_column}"


def _artifact_spans(values: np.ndarray) -> Tuple[np.ndarray, str]:
    """(start, end) rows of an artifact field.

    The bounds are stored either interleaved (start1 end1 start2 end2 ...)
    or as all starts followed by all ends (a column-major N x 2 matrix); the
    interleaved order is the one whose values never decrease.
    """
    values = values[np.isfinite(values)]
    note = ""
    if len(values) % 2:
        values = values[:-1]
        note = "odd number of artifact bounds, last ignored"
    if len(values) < 2:
        return np.empty((0, 2)), note
    half = len(values) // 2
    if np.all(np.diff(values) >= 0):
        return values.reshape(-1, 2), note
    starts, ends = values[:half], values[half:]
    if np.all(ends >= starts) and np.all(np.diff(starts) >= 0):
        return np.column_stack([starts, ends]), note
    note = "; ".join(filter(None, [note, "artifact bounds out of order"]))
    return np.sort(values.reshape(-1, 2), axis=1), note


def read_capnobase_annotations(
    path: Path,
    ppg_fs: float,
    co2_fs: float,
    ppg_length: int,
    co2_length: int,
) -> dict:
    """Expert breaths, pulse peaks and artifact spans of one labels file.

    Event positions become zero-based sample indices of their own signal.
    The unit comes from units_x ("samples" in CapnoBase); without it, a
    breath field ending within the record duration is read as seconds. The
    one-based MATLAB index is removed unless a zero position shows that the
    file is zero-based. Breaths are the starts of expiration (the CO2
    upstroke, where the capnogram crosses zero upward after band-pass
    filtering); starts of inspiration are the fallback.
    """
    fields = read_capnobase_fields(path)
    units_name = _find_field(fields, "unit", suffix="_x")
    units_text = " ".join(fields[units_name]).lower() if units_name else ""
    events = {
        name: _field_numbers(tokens)
        for name, tokens in fields.items()
        if name.endswith("_x") and "unit" not in name
    }
    event_fields = {name: fields[name] for name in events}
    breath_name = None
    for key in ("startexp", "startinsp"):
        name = _find_field(event_fields, "co2", key)
        if name is not None and len(events[name]) >= 2:
            breath_name = name
            break
    if "sample" in units_text:
        units = "samples"
    elif "sec" in units_text or units_text.strip() == "s":
        units = "seconds"
    elif breath_name is not None and events[breath_name].max() <= CAPNOBASE_RECORD_SEC + 1.0:
        units = "seconds (inferred)"
    else:
        units = "samples (inferred)"
    zero_based = any(np.any(values == 0) for values in events.values() if len(values))
    index_base = 0 if (zero_based or units.startswith("seconds")) else CAPNOBASE_LABEL_INDEX_BASE

    warnings_list = []

    def to_samples(values: np.ndarray, fs: float) -> np.ndarray:
        values = values[np.isfinite(values)]
        return values * fs if units.startswith("seconds") else values - index_base

    def events_of(name: Optional[str], fs: float, length: int) -> np.ndarray:
        if name is None:
            return np.empty(0, dtype=np.int64)
        samples = np.round(to_samples(events[name], fs)).astype(np.int64)
        inside = (samples >= 0) & (samples < length)
        if not np.all(inside):
            warnings_list.append(
                f"{name}: {int((~inside).sum())} of {len(samples)} positions outside the record"
            )
        return np.unique(samples[inside])

    breaths = events_of(breath_name, co2_fs, co2_length)
    startinsp_name = _find_field(event_fields, "co2", "startinsp")
    pleth_peak_name = _find_field(event_fields, "pleth", "peak")
    artifacts = {"pleth": [], "co2": []}
    for name, values in events.items():
        if "artif" not in name:
            continue
        signal_name = "pleth" if "pleth" in name else "co2" if "co2" in name else None
        if signal_name is None:
            continue
        fs = ppg_fs if signal_name == "pleth" else co2_fs
        spans, note = _artifact_spans(to_samples(values, fs))
        if note:
            warnings_list.append(f"{name}: {note}")
        if len(spans):
            artifacts[signal_name].append(spans / fs)
    artifacts = {
        key: np.concatenate(spans, axis=0) if spans else np.empty((0, 2))
        for key, spans in artifacts.items()
    }
    if breath_name is None:
        warnings_list.append(f"no co2 startexp/startinsp field (fields: {list(events)[:12]})")
    elif len(breaths) < 2:
        warnings_list.append(f"{breath_name}: fewer than two breaths inside the record")
    return {
        "breaths": breaths if len(breaths) >= 2 else None,
        "breath_field": breath_name or "",
        "startinsp": events_of(startinsp_name, co2_fs, co2_length)
        if startinsp_name != breath_name
        else np.empty(0, dtype=np.int64),
        "pleth_peaks": events_of(pleth_peak_name, ppg_fs, ppg_length),
        "ppg_artifacts_sec": artifacts["pleth"],
        "co2_artifacts_sec": artifacts["co2"],
        "units": units,
        "index_base": index_base,
        "fields": sorted(events),
        "warnings": warnings_list,
    }


def read_capnobase_reference_rr(path: Path) -> Tuple[float, str]:
    """Median of the label-derived CO2 respiratory-rate trend (reference file)."""
    fields = read_capnobase_fields(path)
    name = _find_field(fields, "rr", "co2", suffix="_y")
    values = _field_numbers(fields[name]) if name else np.empty(0)
    values = values[np.isfinite(values) & (values > 0) & (values < 150)]
    return (float(np.median(values)) if len(values) else np.nan), name or ""


def _text_field(fields: Dict[str, List[str]], *parts: str) -> str:
    name = _find_field(fields, *parts)
    return " ".join(fields[name]) if name else ""


def read_capnobase_case(
    csv_dir: Path,
    case_id: str,
    default_fs: float,
    read_labels: bool = True,
) -> dict:
    """Signals, sampling rates, expert annotations and metadata of one case.

    Breath labels are zero-based sample indices at the CO2 rate, artifact
    spans are (start, end) seconds. Every assumption of the conversion
    (field names, units, index base, sampling rate source) is recorded in
    `diagnostics`, inconsistencies in `warnings`.
    """
    signal_path = capnobase_file(csv_dir, case_id, "signal")
    if signal_path is None:
        raise FileNotFoundError(f"{case_id}_8min_signal.csv not found in {csv_dir}")
    ppg, co2, signal_columns = read_capnobase_signals(signal_path)
    warnings_list = []
    param_path = capnobase_file(csv_dir, case_id, "param")
    param = read_capnobase_fields(param_path) if param_path else {}
    if param_path is None:
        warnings_list.append(f"param file missing, {default_fs:g} Hz assumed")
    ppg_fs, ppg_fs_source = _sampling_rate(param, "pleth", default_fs)
    co2_fs, co2_fs_source = _sampling_rate(param, "co2", ppg_fs)
    record_sec = len(ppg) / ppg_fs
    if abs(record_sec - CAPNOBASE_RECORD_SEC) > 2.0:
        warnings_list.append(
            f"record is {record_sec:.1f} s at {ppg_fs:g} Hz "
            f"(CapnoBase cases are {CAPNOBASE_RECORD_SEC:.0f} s): check the sampling rate"
        )
    if int(round(ppg_fs)) != int(round(co2_fs)):
        warnings_list.append(
            f"pleth {ppg_fs:g} Hz and co2 {co2_fs:g} Hz share one table of {len(ppg)} rows"
        )
    meta_path = capnobase_file(csv_dir, case_id, "meta")
    meta = read_capnobase_fields(meta_path) if meta_path else {}

    annotations = None
    labels_path = capnobase_file(csv_dir, case_id, "labels") if read_labels else None
    if read_labels and labels_path is None:
        warnings_list.append("labels file missing, breaths are detected instead")
    if labels_path is not None:
        annotations = read_capnobase_annotations(
            labels_path, ppg_fs, co2_fs, len(ppg), len(co2)
        )
        warnings_list.extend(annotations["warnings"])
    breaths = annotations["breaths"] if annotations else None
    label_rr = (
        float(60.0 * co2_fs / np.median(np.diff(breaths)))
        if breaths is not None
        else np.nan
    )
    reference_path = capnobase_file(csv_dir, case_id, "reference")
    reference_rr, reference_field = (
        read_capnobase_reference_rr(reference_path) if reference_path else (np.nan, "")
    )
    if np.isfinite(label_rr) and np.isfinite(reference_rr):
        if abs(label_rr - reference_rr) > CAPNOBASE_RR_CHECK_BPM:
            warnings_list.append(
                f"label RR {label_rr:.1f} vs reference-file RR {reference_rr:.1f} bpm: "
                "check label units and sampling rate"
            )
    empty_spans = np.empty((0, 2))
    ppg_artifacts = annotations["ppg_artifacts_sec"] if annotations else empty_spans
    co2_artifacts = annotations["co2_artifacts_sec"] if annotations else empty_spans
    age_values = [
        _field_numbers(tokens)
        for name, tokens in meta.items()
        if "age" in name.split("_")
    ]
    age_values = [values[0] for values in age_values if len(values)]
    diagnostics = {
        "CapnoBase_Signal_Columns": signal_columns,
        "CapnoBase_FS_Pleth": ppg_fs,
        "CapnoBase_FS_CO2": co2_fs,
        "CapnoBase_FS_Source": f"{ppg_fs_source},{co2_fs_source}",
        "CapnoBase_Record_Sec": float(record_sec),
        "CapnoBase_Label_Field": annotations["breath_field"] if annotations else "",
        "CapnoBase_Label_Units": annotations["units"] if annotations else "",
        "CapnoBase_Label_Index_Base": annotations["index_base"] if annotations else np.nan,
        "CapnoBase_Breath_Labels": int(len(breaths)) if breaths is not None else 0,
        "CapnoBase_Label_RR_Median_BPM": label_rr,
        "CapnoBase_Reference_RR_Median_BPM": reference_rr,
        "CapnoBase_Reference_RR_Field": reference_field,
        "CapnoBase_PPG_Artifacts": int(len(ppg_artifacts)),
        "CapnoBase_PPG_Artifact_Sec": float(np.sum(ppg_artifacts[:, 1] - ppg_artifacts[:, 0])),
        "CapnoBase_CO2_Artifacts": int(len(co2_artifacts)),
        "CapnoBase_CO2_Artifact_Sec": float(np.sum(co2_artifacts[:, 1] - co2_artifacts[:, 0])),
        "CapnoBase_Ventilation": _text_field(param, "vent") or _text_field(meta, "vent"),
        "CapnoBase_Age": float(age_values[0]) if age_values else np.nan,
        "CapnoBase_Loading_Warnings": " | ".join(warnings_list),
    }
    return {
        "ppg": ppg,
        "co2": co2,
        "ppg_fs": ppg_fs,
        "co2_fs": co2_fs,
        "breaths": breaths,
        "startinsp": annotations["startinsp"] if annotations else np.empty(0, dtype=np.int64),
        "pleth_peaks": annotations["pleth_peaks"] if annotations else np.empty(0, dtype=np.int64),
        "ppg_artifacts_sec": ppg_artifacts,
        "co2_artifacts_sec": co2_artifacts,
        "label_fields": annotations["fields"] if annotations else [],
        "diagnostics": diagnostics,
        "warnings": warnings_list,
    }


def capnobase_case_ids(csv_dir: Path) -> List[str]:
    return sorted(
        {
            path.name.split("_")[0]
            for suffix in CAPNOBASE_FILE_SUFFIXES
            for path in csv_dir.glob(f"*_8min_signal{suffix}")
        }
    )


def load_capnobase_subjects(spec: WindowSpec) -> Dict[str, dict]:
    cfg = DATASET_CONFIGS["capnobase"]
    data = {}
    csv_dir = Path(cfg["data_path"])
    for case_id in capnobase_case_ids(csv_dir):
        try:
            case = read_capnobase_case(
                csv_dir,
                case_id,
                cfg["default_fs"],
                read_labels=bool(cfg.get("expert_breath_labels")),
            )
            for message in case["warnings"]:
                print(f"[CapnoBase {case_id}] {message}")
            artifacts = (
                [(case["ppg_artifacts_sec"], case["co2_artifacts_sec"])]
                if cfg.get("use_artifact_labels")
                else None
            )
            result = preprocess_subject_from_pairs(
                [(case["ppg"], case["co2"], case["ppg_fs"], case["co2_fs"])],
                spec,
                ANALYSIS_PROFILES[cfg["profile"]],
                [case["breaths"]],
                artifacts,
            )
            if result is not None:
                result["reference_diagnostics"].update(case["diagnostics"])
                data[case_id] = result
        except Exception as exc:
            if is_fatal_error(exc):
                raise
            print(f"[CapnoBase {case_id}] skipped: {exc}")
    return data


def capnobase_label_checks(case: dict) -> dict:
    """Checks that the labels sit where the signals say they should.

    A start of expiration is where CO2 rises (mean of the 0.5 s after the
    label above the 0.5 s before it), a start of inspiration is where it
    falls, the two alternate, and an expert pulse peak is the PPG maximum
    within +-0.1 s (offset 0 means index base and sampling rate agree).
    """
    co2 = np.asarray(case["co2"], dtype=np.float64)
    ppg = np.asarray(case["ppg"], dtype=np.float64)
    half = int(round(0.5 * case["co2_fs"]))

    def co2_step(labels: np.ndarray) -> np.ndarray:
        labels = labels[(labels >= half) & (labels < len(co2) - half)]
        return np.asarray(
            [
                np.nanmean(co2[index : index + half]) - np.nanmean(co2[index - half : index])
                for index in labels
            ]
        )

    breaths = case["breaths"] if case["breaths"] is not None else np.empty(0, dtype=np.int64)
    exp_step = co2_step(breaths)
    insp_step = co2_step(case["startinsp"])
    alternation = np.nan
    if len(breaths) > 1 and len(case["startinsp"]):
        between = np.searchsorted(case["startinsp"], breaths[1:]) - np.searchsorted(
            case["startinsp"], breaths[:-1]
        )
        alternation = float(np.mean(between == 1))
    reach = int(round(0.1 * case["ppg_fs"]))
    offsets = [
        int(np.nanargmax(ppg[p - reach : p + reach + 1])) - reach
        for p in case["pleth_peaks"]
        if reach <= p < len(ppg) - reach and np.any(np.isfinite(ppg[p - reach : p + reach + 1]))
    ]
    offsets = np.asarray(offsets, dtype=np.float64)
    return {
        "Check_CO2_Rises_At_Breath_Label": float(np.mean(exp_step > 0)) if len(exp_step) else np.nan,
        "Check_CO2_Falls_At_StartInsp": float(np.mean(insp_step < 0)) if len(insp_step) else np.nan,
        "Check_One_StartInsp_Per_Breath": alternation,
        "Check_Pleth_Peak_Labels": int(len(case["pleth_peaks"])),
        "Check_Pleth_Peak_Offset_Median_Samples": float(np.median(offsets)) if len(offsets) else np.nan,
        "Check_Pleth_Peak_Within_10ms": float(
            np.mean(np.abs(offsets) <= 0.01 * case["ppg_fs"])
        )
        if len(offsets)
        else np.nan,
        "Check_Label_HR_Median_BPM": float(
            60.0 * case["ppg_fs"] / np.median(np.diff(case["pleth_peaks"]))
        )
        if len(case["pleth_peaks"]) > 1
        else np.nan,
    }


def inspect_capnobase(results_root: Path) -> pd.DataFrame:
    """Print and save how every CapnoBase case is read (no training).

    Writes capnobase_loading_check.csv. A correct load shows 300 Hz, a
    480 s record, label RR equal to the reference-file RR, CO2 rising at
    nearly every breath label, and a median pulse-peak offset of 0 samples.
    """
    cfg = DATASET_CONFIGS["capnobase"]
    csv_dir = Path(cfg["data_path"])
    case_ids = capnobase_case_ids(csv_dir)
    print(f"CapnoBase directory: {csv_dir} ({len(case_ids)} cases)")
    rows = []
    for case_id in case_ids:
        try:
            case = read_capnobase_case(csv_dir, case_id, cfg["default_fs"])
            row = {
                "Case": case_id,
                **case["diagnostics"],
                **capnobase_label_checks(case),
                "CapnoBase_Label_Fields": " ".join(case["label_fields"]),
            }
        except Exception as exc:
            row = {"Case": case_id, "CapnoBase_Loading_Warnings": f"failed: {exc}"}
        rows.append(row)
    frame = pd.DataFrame(rows)
    if frame.empty:
        print("No {case}_8min_signal.csv files found.")
        return frame
    results_root.mkdir(parents=True, exist_ok=True)
    path = results_root / "capnobase_loading_check.csv"
    frame.to_csv(path, index=False, encoding="utf-8-sig")
    shown = {
        "Case": "case",
        "CapnoBase_FS_Pleth": "fs",
        "CapnoBase_Record_Sec": "sec",
        "CapnoBase_Breath_Labels": "breaths",
        "CapnoBase_Label_RR_Median_BPM": "RR_lab",
        "CapnoBase_Reference_RR_Median_BPM": "RR_ref",
        "Check_CO2_Rises_At_Breath_Label": "CO2up",
        "Check_Pleth_Peak_Offset_Median_Samples": "pk_off",
        "CapnoBase_PPG_Artifact_Sec": "ppg_art_s",
        "CapnoBase_CO2_Artifact_Sec": "co2_art_s",
        "CapnoBase_Ventilation": "vent",
    }
    table = frame[[column for column in shown if column in frame.columns]].rename(columns=shown)
    with pd.option_context("display.max_rows", None, "display.width", 160):
        print(table.to_string(index=False, float_format=lambda value: f"{value:.2f}"))
    warned = frame[frame["CapnoBase_Loading_Warnings"].fillna("").astype(str) != ""]
    for _, row in warned.iterrows():
        print(f"[CapnoBase {row['Case']}] {row['CapnoBase_Loading_Warnings']}")
    print(f"Saved: {path}")
    return frame


def _mat_stem(filename: str) -> str:
    return os.path.splitext(filename)[0]


def _match_mat_file_pairs(
    ppg_dir: Path,
    rsp_dir: Path,
) -> List[Tuple[str, str]]:
    ppg_files = sorted(path.name for path in ppg_dir.glob("*.mat"))
    rsp_files = sorted(path.name for path in rsp_dir.glob("*.mat"))
    ppg_map = {_mat_stem(name): name for name in ppg_files}
    rsp_map = {_mat_stem(name): name for name in rsp_files}
    common = sorted(set(ppg_map).intersection(rsp_map))
    if common:
        return [(ppg_map[key], rsp_map[key]) for key in common]
    if len(ppg_files) != len(rsp_files):
        raise RuntimeError(
            "PPG/RSP MAT file counts differ and no common stems exist"
        )
    return list(zip(ppg_files, rsp_files))


# STEAM2 sessions are EEGLAB structures saved as .mat files (EEG.data,
# EEG.srate, EEG.chanlocs.labels, EEG.event), one channel per file: the
# PPG file and the RSP (belt) file of a session have the same name in the
# PPG and RSP folders and come from the same recording (same length).
STEAM2_CHANNELS = {"PPG": ("PPG", "PLETH", "BVP"), "RSP": ("RSP", "RESP", "RESPIRATION", "BELT")}
STEAM2_SESSION_PATTERN = re.compile(r"^(?P<subject>.+?)[_\- ]*session[_\- ]*(?P<session>\d+)$", re.IGNORECASE)


def steam2_session_key(stem: str) -> Tuple[str, int]:
    """(subject, session number) from a file stem such as 13_LSB_Session2."""
    match = STEAM2_SESSION_PATTERN.match(stem)
    if match is None:
        return stem, 0
    return match.group("subject"), int(match.group("session"))


def _as_list(value) -> list:
    if value is None:
        return []
    if isinstance(value, np.ndarray):
        return list(value.ravel()) if value.size else []
    return [value]


def read_eeglab_channel(path: Path, kind: str, default_fs: float) -> dict:
    """One channel of an EEGLAB .mat file.

    The channel is chosen by its label (STEAM2_CHANNELS[kind]); a file with
    one channel is accepted when it has no label, and rejected when its label
    names the other signal (swapped files). EEGLAB "boundary" events inside
    the recording mark discontinuities (removed data); their positions are
    returned so that the session is split there. Files without an EEG
    structure fall back to the generic reader (data and srate fields).
    """
    mat = sio.loadmat(path, squeeze_me=True, struct_as_record=False)
    names = [key for key in mat if not key.startswith("_")]
    payload = mat["EEG"] if "EEG" in mat else (mat[names[0]] if names else None)
    if payload is None or not hasattr(payload, "data"):
        signal, fs = _extract_signal_and_fs(_first_mat_payload(sio.loadmat(path)), default_fs)
        return {"signal": signal, "fs": fs, "boundaries": [], "label": "", "channels": 1}
    data = np.asarray(payload.data, dtype=np.float64)
    data = data.reshape(1, -1) if data.ndim == 1 else data
    labels = [str(getattr(item, "labels", "")).strip() for item in _as_list(getattr(payload, "chanlocs", None))]
    wanted = {name.upper() for name in STEAM2_CHANNELS[kind]}
    other = {name.upper() for key, names in STEAM2_CHANNELS.items() if key != kind for name in names}
    index = next((i for i, label in enumerate(labels) if label.upper() in wanted), None)
    if index is None:
        if data.shape[0] != 1:
            raise RuntimeError(f"{path.name}: no {kind} channel among {labels}")
        if labels and labels[0].upper() in other:
            raise RuntimeError(f"{path.name}: channel '{labels[0]}' found where {kind} was expected")
        index = 0
    signal = data[index].astype(np.float32)
    try:
        fs = float(np.asarray(payload.srate).squeeze())
    except Exception:
        fs = float(default_fs)
    boundaries = []
    for event in _as_list(getattr(payload, "event", None)):
        if str(getattr(event, "type", "")).strip().lower() != "boundary":
            continue
        try:
            latency = float(getattr(event, "latency"))
        except (TypeError, ValueError):
            continue
        # EEGLAB latencies are 1-based and a boundary at k.5 lies between
        # samples k and k + 1: the next piece starts at 0-based index k. The
        # markers at the start (0.5, "New Segment" at 1) and at the end are
        # not discontinuities inside the data.
        position = int(np.ceil(latency - 1.0))
        if 0 < position < signal.size:
            boundaries.append(position)
    label = labels[index] if index < len(labels) else ""
    return {"signal": signal, "fs": fs, "boundaries": sorted(set(boundaries)), "label": label,
            "channels": int(data.shape[0])}


def steam2_session_files(ppg_root: Path, rsp_root: Path) -> Dict[str, List[Tuple[str, Path, Path]]]:
    """Subject -> [(session name, PPG file, RSP file)], sessions in order.

    Subject folders (PPG\\<subject>\\<session>.mat and RSP\\<subject>\\...) are
    used when present; otherwise the files of one flat folder are grouped by
    the subject part of their name (<subject>_Session<k>.mat). Every session
    of a person stays in one subject, so a person is never in the training
    and the test data of the same fold.
    """
    sessions: Dict[str, List[Tuple[str, Path, Path]]] = {}
    subject_dirs = sorted(path for path in ppg_root.iterdir() if path.is_dir())
    if subject_dirs:
        for subject_dir in subject_dirs:
            rsp_dir = rsp_root / subject_dir.name
            if not rsp_dir.is_dir():
                print(f"[STEAM2 {subject_dir.name}] no RSP folder, skipped")
                continue
            # PPG and RSP files of a session have the same name; files without
            # a partner are reported and left out (never paired by position).
            rsp_files = {path.stem: path for path in rsp_dir.glob("*.mat")}
            for ppg_path in sorted(subject_dir.glob("*.mat")):
                if ppg_path.stem not in rsp_files:
                    print(f"[STEAM2 {subject_dir.name}] {ppg_path.name}: no RSP file of the same name, left out")
                    continue
                sessions.setdefault(subject_dir.name, []).append(
                    (ppg_path.stem, ppg_path, rsp_files[ppg_path.stem])
                )
            for stem in sorted(set(rsp_files) - {path.stem for path in subject_dir.glob("*.mat")}):
                print(f"[STEAM2 {subject_dir.name}] {stem}.mat: no PPG file of the same name, left out")
    else:
        rsp_files = {path.stem: path for path in rsp_root.glob("*.mat")}
        for ppg_path in sorted(ppg_root.glob("*.mat")):
            if ppg_path.stem not in rsp_files:
                print(f"[STEAM2 {ppg_path.name}] no RSP file of the same name, skipped")
                continue
            subject, _ = steam2_session_key(ppg_path.stem)
            sessions.setdefault(subject, []).append((ppg_path.stem, ppg_path, rsp_files[ppg_path.stem]))
    for subject in sessions:
        sessions[subject].sort(key=lambda item: (steam2_session_key(item[0])[1], item[0]))
    return sessions


def steam2_session_pieces(ppg: dict, rsp: dict) -> Tuple[List[Tuple[np.ndarray, np.ndarray, float, float]], dict]:
    """The (PPG, RSP, fs, fs) pieces of one session, split at the boundary
    events of either file, and a description of the session."""
    ppg_sec = ppg["signal"].size / ppg["fs"]
    rsp_sec = rsp["signal"].size / rsp["fs"]
    cuts = sorted({b / ppg["fs"] for b in ppg["boundaries"]} | {b / rsp["fs"] for b in rsp["boundaries"]})
    edges = [0.0] + [cut for cut in cuts if 0.0 < cut < min(ppg_sec, rsp_sec)] + [min(ppg_sec, rsp_sec)]
    pieces = []
    for start, end in zip(edges[:-1], edges[1:]):
        p = ppg["signal"][int(round(start * ppg["fs"])):int(round(end * ppg["fs"]))]
        r = rsp["signal"][int(round(start * rsp["fs"])):int(round(end * rsp["fs"]))]
        if p.size and r.size:
            pieces.append((p, r, ppg["fs"], rsp["fs"]))
    info = {
        "PPG_Label": ppg["label"], "RSP_Label": rsp["label"],
        "PPG_FS": ppg["fs"], "RSP_FS": rsp["fs"],
        "PPG_Sec": ppg_sec, "RSP_Sec": rsp_sec, "Length_Difference_Sec": ppg_sec - rsp_sec,
        "Internal_Boundaries": len(edges) - 2, "Pieces": len(pieces),
        "Piece_Sec": " ".join(f"{end - start:.1f}" for start, end in zip(edges[:-1], edges[1:])),
    }
    return pieces, info


def load_steam2_subjects(spec: WindowSpec) -> Dict[str, dict]:
    cfg = DATASET_CONFIGS["steam2"]
    profile = ANALYSIS_PROFILES[cfg["profile"]]
    data = {}
    sessions = steam2_session_files(Path(cfg["ppg_dir"]), Path(cfg["rsp_dir"]))
    for subject, items in sessions.items():
        try:
            pairs = []
            for name, ppg_path, rsp_path in items:
                ppg = read_eeglab_channel(ppg_path, "PPG", cfg["default_fs"])
                rsp = read_eeglab_channel(rsp_path, "RSP", cfg["default_fs"])
                pieces, info = steam2_session_pieces(ppg, rsp)
                if abs(info["Length_Difference_Sec"]) > 1.0:
                    print(f"[STEAM2 {subject} {name}] PPG {info['PPG_Sec']:.1f} s vs RSP "
                          f"{info['RSP_Sec']:.1f} s; the common part is used")
                pairs += pieces
            result = preprocess_subject_from_pairs(pairs, spec, profile)
            if result is not None:
                data[subject] = result
        except Exception as exc:
            if is_fatal_error(exc):
                raise
            print(f"[STEAM2 {subject}] skipped: {exc}")
    return data


def inspect_steam2(results_root: Path) -> pd.DataFrame:
    """Read every STEAM2 session, print how it was loaded and save
    steam2_loading_check.csv (no training)."""
    cfg = DATASET_CONFIGS["steam2"]
    rows = []
    sessions = steam2_session_files(Path(cfg["ppg_dir"]), Path(cfg["rsp_dir"]))
    for subject, items in sessions.items():
        for name, ppg_path, rsp_path in items:
            row = {"Subject": subject, "Session": name, "PPG_File": str(ppg_path), "RSP_File": str(rsp_path)}
            try:
                ppg = read_eeglab_channel(ppg_path, "PPG", cfg["default_fs"])
                rsp = read_eeglab_channel(rsp_path, "RSP", cfg["default_fs"])
                pieces, info = steam2_session_pieces(ppg, rsp)
                row.update(info)
                # Zero-lag correlation of PPG and belt in 0.1-0.5 Hz: the sign
                # the v16 rule would have given the target of this session.
                p, r, ppg_fs, rsp_fs = max(pieces, key=lambda piece: piece[0].size)
                if int(round(ppg_fs)) != int(round(rsp_fs)):
                    r = resample_signal_poly(r, rsp_fs, ppg_fs)
                length = min(p.size, r.size)
                band = ANALYSIS_PROFILES[cfg["profile"]]["band_hz"]
                corr = safe_pcc(bandpass_filter(p[:length], ppg_fs, *band)[0],
                                bandpass_filter(r[:length], ppg_fs, *band)[0])
                row.update({"ZeroLag_r_PPG_RSP": corr, "v16_Sign": -1.0 if corr < 0 else 1.0, "Error": ""})
            except Exception as exc:
                row["Error"] = repr(exc)
            rows.append(row)
    frame = pd.DataFrame(rows)
    path = results_root / "steam2_loading_check.csv"
    frame.to_csv(path, index=False, encoding="utf-8-sig")
    if frame.empty:
        print("No STEAM2 session found.")
        return frame
    ok = frame[frame["Error"] == ""]
    print(f"STEAM2: {frame['Subject'].nunique()} subjects, {len(frame)} sessions ({len(frame) - len(ok)} unreadable)")
    if not ok.empty:
        print(f"  sampling rate PPG {sorted(ok['PPG_FS'].unique())}, RSP {sorted(ok['RSP_FS'].unique())} Hz; "
              f"session length {ok['PPG_Sec'].median():.1f} s (median); sessions split at internal boundaries: "
              f"{int((ok['Internal_Boundaries'] > 0).sum())}")
        absolute = ok["ZeroLag_r_PPG_RSP"].abs()
        print(f"  zero-lag |r| PPG vs belt: median {absolute.median():.3f}, below 0.2 in "
              f"{int((absolute < 0.2).sum())}/{len(ok)} sessions; the v16 rule would flip "
              f"{int((ok['v16_Sign'] < 0).sum())}/{len(ok)} sessions")
    print(f"Saved: {path}")
    return frame


def load_dataset(dataset_key: str, spec: WindowSpec) -> Dict[str, dict]:
    if dataset_key == "bidmc":
        return load_bidmc_subjects(spec)
    if dataset_key == "capnobase":
        return load_capnobase_subjects(spec)
    if dataset_key == "steam2":
        return load_steam2_subjects(spec)
    raise KeyError(dataset_key)


# ---------------------------------------------------------------------------
# LOSOCV window datasets
# ---------------------------------------------------------------------------

def select_subject_indices(
    n_windows: int,
    split: str,
) -> np.ndarray:
    if split == "test":
        return np.arange(n_windows, dtype=int)
    val_count = (
        max(1, int(round(n_windows * VAL_WINDOW_RATIO)))
        if n_windows > 1
        else 0
    )
    val_count = min(val_count, max(0, n_windows - 1))
    if split == "train":
        return np.arange(0, n_windows - val_count, dtype=int)
    if split == "val":
        return (
            np.arange(n_windows - val_count, n_windows, dtype=int)
            if val_count
            else np.empty(0, dtype=int)
        )
    raise ValueError(split)


def validation_subjects(dataset_key: str, test_subject: str, train_subjects: Sequence[str]) -> List[str]:
    """Training subjects held out whole as validation subjects in one fold.

    VAL_SUBJECT_RATIO of the training subjects (rounded, at least one, at
    most all but one), drawn once per fold with a seed of the dataset and the
    held-out subject: every member and weight decay candidate of the fold
    uses the same validation subjects, and the test subject is never one.
    """
    candidates = sorted(str(subject) for subject in train_subjects if str(subject) != str(test_subject))
    count = int(round(VAL_SUBJECT_RATIO * len(candidates)))
    count = min(max(1, count), max(1, len(candidates) - 1))
    rng = np.random.default_rng(stable_seed("261013-validation-subjects", dataset_key, test_subject))
    return sorted(rng.choice(candidates, size=count, replace=False).tolist())


def test_metadata(
    subject_data: Dict[str, dict],
    test_subject: str,
) -> List[dict]:
    record = subject_data[test_subject]
    return [
        {
            "segment_id": int(segment_id),
            "start": int(start),
        }
        for segment_id, start in zip(
            record["segment_ids"],
            record["starts"],
        )
    ]


FATAL_ERROR_WORDS = ("CUDA", "cuDNN", "CUDNN", "CUBLAS", "cublas", "out of memory", "Memory allocation")


def is_fatal_error(exc: BaseException) -> bool:
    """Out of host or GPU memory, or a CUDA / cuDNN failure: the process (and
    its CUDA context) cannot be trusted any more, so it stops instead of
    failing every following fold; it is restarted with --resume."""
    if isinstance(exc, MemoryError):
        return True
    text = f"{type(exc).__name__}: {exc}"
    return any(word in text for word in FATAL_ERROR_WORDS)


def release_memory(device: Optional[torch.device] = None) -> None:
    """Release objects from completed folds before the next LOSOCV fold."""
    gc.collect()
    if device is not None and device.type == "cuda":
        torch.cuda.empty_cache()


class SubjectWindowDataset(Dataset):
    """Lazy window view that avoids train/validation/test array duplication.

    Splits: "train"/"val" are the temporal split of every subject (last 20%
    of its kept windows for validation, Train_OK only; 261007), or with
    subject_wise every Train_OK window of the given subjects (training or
    validation subjects of the fold; 261013); "test" is every kept window
    (inference never depends on the reference). Training windows get the v16
    jitter (random shift up to SHIFT_AUG_SEC, gain 0.95-1.05, small noise).
    """

    def __init__(
        self,
        subject_data: Dict[str, dict],
        subjects: Sequence[str],
        split: str,
        augment: bool,
        subject_wise: bool = False,
    ):
        self.subject_data = subject_data
        self.samples = []
        for subject in subjects:
            record = subject_data[subject]
            indices = (
                np.arange(len(record["low"]), dtype=int)
                if subject_wise
                else select_subject_indices(len(record["low"]), split)
            )
            if split in ("train", "val"):
                indices = indices[record["train_ok"][indices]]
            self.samples.extend((subject, int(index)) for index in indices)
        if not self.samples:
            raise RuntimeError(f"No windows available for split={split}")
        self.augment = augment

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int):
        subject, window_index = self.samples[index]
        record = self.subject_data[subject]
        x = record["low"][window_index]
        y = record["rsp"][window_index]
        e = record["event"][window_index]
        norm = float(record["event_norm"][window_index])
        x = instance_minmax_normalize(np.asarray(x, dtype=np.float32)[None, :], (0.0, 1.0))
        y = instance_minmax_normalize(np.asarray(y, dtype=np.float32)[None, :], (-1.0, 1.0))[0]
        x_tensor = torch.from_numpy(np.asarray(x, dtype=np.float32).copy())
        y_tensor = torch.from_numpy(np.asarray(y, dtype=np.float32).copy())
        e_tensor = torch.from_numpy(np.asarray(e, dtype=np.float32).copy())
        if self.augment:
            reach = int(round(SHIFT_AUG_SEC * float(record["fs"])))
            shift = random.randint(-reach, reach)
            scale = random.uniform(0.95, 1.05)
            noise_std = random.uniform(0.002, 0.01)
            if shift:
                x_tensor = torch.roll(x_tensor, shifts=shift, dims=-1)
                y_tensor = torch.roll(y_tensor, shifts=shift, dims=-1)
                e_tensor = torch.roll(e_tensor, shifts=shift, dims=-1)
            x_tensor = x_tensor * scale + torch.randn_like(x_tensor) * noise_std
        return (
            x_tensor,
            y_tensor.unsqueeze(0),
            e_tensor.unsqueeze(0),
            torch.tensor(norm, dtype=torch.float32),
        )


def make_train_loader(
    dataset: SubjectWindowDataset,
    batch_size: int,
    device: torch.device,
) -> DataLoader:
    return DataLoader(
        dataset,
        batch_size=int(batch_size),
        shuffle=True,
        drop_last=False,
        num_workers=0,
        pin_memory=device.type == "cuda",
    )


def make_eval_loader(dataset: SubjectWindowDataset, batch_size: int, device: torch.device) -> DataLoader:
    return DataLoader(
        dataset,
        batch_size=int(batch_size),
        shuffle=False,
        drop_last=False,
        num_workers=0,
        pin_memory=device.type == "cuda",
    )


# ---------------------------------------------------------------------------
# v16 Deep-only building blocks (unchanged)
# ---------------------------------------------------------------------------

def choose_group_count(channels: int) -> int:
    for groups in (8, 4, 2):
        if channels % groups == 0:
            return groups
    return 1


class ResidualConvBlock(nn.Module):
    """v16 depthwise dilated encoder block: GELU(block(x) + x)."""

    def __init__(
        self,
        channels: int,
        kernel_size: int,
        dilation: int,
        dropout: float,
    ):
        super().__init__()
        padding = int(dilation) * (int(kernel_size) // 2)
        groups = int(channels)
        self.conv1 = nn.Conv1d(
            channels,
            channels,
            kernel_size,
            padding=padding,
            dilation=dilation,
            groups=groups,
        )
        self.norm1 = nn.GroupNorm(channels, channels)
        self.conv2 = nn.Conv1d(
            channels,
            channels,
            kernel_size,
            padding=padding,
            dilation=dilation,
            groups=groups,
        )
        self.norm2 = nn.GroupNorm(channels, channels)
        self.act = nn.GELU()
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.drop(self.act(self.norm1(self.conv1(x))))
        out = self.drop(self.norm2(self.conv2(out)))
        return self.act(out + x)


class Conv1dExplicitSame(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
    ):
        super().__init__()
        total_pad = int(kernel_size) - 1
        self.pad = nn.ConstantPad1d(
            (total_pad // 2, total_pad - total_pad // 2),
            0.0,
        )
        self.conv = nn.Conv1d(
            in_channels,
            out_channels,
            kernel_size,
            padding=0,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(self.pad(x))


def make_encoder(channels: int, dropout: float) -> nn.Sequential:
    return nn.Sequential(
        *[
            ResidualConvBlock(
                channels,
                MODEL_CONFIG["encoder_kernel_size"],
                dilation,
                dropout,
            )
            for dilation in MODEL_CONFIG["encoder_dilations"]
        ]
    )


# ---------------------------------------------------------------------------
# Multi-scale stem
# ---------------------------------------------------------------------------

class StemBranch(nn.Module):
    """v16 stem branch: explicit-same Conv1d -> GroupNorm -> GELU."""

    def __init__(self, stem_channels: int, kernel_size: int):
        super().__init__()
        self.conv = Conv1dExplicitSame(1, stem_channels, kernel_size)
        self.norm = nn.GroupNorm(
            choose_group_count(stem_channels),
            stem_channels,
        )
        self.act = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(self.norm(self.conv(x)))


# ---------------------------------------------------------------------------
# Decoder and network
# ---------------------------------------------------------------------------

def make_breath_decoder_trunk(
    channels: int,
    dilations: Sequence[int],
    dropout: float,
) -> nn.Sequential:
    """v16 decoder layers (24 -> 16 -> 8 channels) with dilated convolutions."""
    width = int(channels)
    mids = tuple(MODEL_CONFIG["decoder_mid_channels"])
    kernels = tuple(MODEL_CONFIG["decoder_kernel_sizes"])
    dilation_list = tuple(dilations)
    if not len(kernels) == len(dilation_list) == len(mids):
        raise ValueError("decoder channels, kernels and dilations differ in length")
    layers = []
    # Conv -> GroupNorm -> GELU -> Dropout per layer, as in the v16 decoder.
    for out_channels, kernel, dilation in zip(mids, kernels, dilation_list):
        layers += [
            nn.Conv1d(
                width,
                out_channels,
                kernel,
                padding=dilation * (kernel // 2),
                dilation=dilation,
            ),
            nn.GroupNorm(choose_group_count(out_channels), out_channels),
            nn.GELU(),
            nn.Dropout(dropout),
        ]
        width = out_channels
    return nn.Sequential(*layers)


class DeepOnlyV16Decoder(nn.Module):
    """260928 early stem fusion network with the dilated decoder (the stem
    branches are an option; the default is the 261007 full model).

    three stem branches -> concatenation (24 channels) -> 1x1 fusion +
    GroupNorm (no GELU) -> depthwise residual encoder -> dilated decoder
    trunk -> waveform head (tanh), the only output. With
    stem_fusion="none" (ablation) the concatenated stem channels feed the
    encoder directly.
    """

    def __init__(
        self,
        stem_fusion: str = "conv1x1",
        dropout: Optional[float] = None,
        decoder_dilations: Optional[Sequence[int]] = None,
        stem_kernel_sizes: Optional[Sequence[int]] = None,
    ):
        super().__init__()
        if stem_fusion not in STEM_FUSION_MODES:
            raise ValueError(f"stem_fusion must be one of {STEM_FUSION_MODES}")
        self.stem_fusion = stem_fusion
        stem_channels = int(MODEL_CONFIG["stem_base_channels"])
        hidden = MODEL_CONFIG["hidden_channels"]
        kernel_sizes = tuple(
            MODEL_CONFIG["stem_kernel_sizes"] if stem_kernel_sizes is None else stem_kernel_sizes
        )
        if not kernel_sizes or not set(kernel_sizes) <= set(MODEL_CONFIG["stem_kernel_sizes"]):
            raise ValueError(f"stem branches must be a non-empty subset of {MODEL_CONFIG['stem_kernel_sizes']}")
        dropout = float(MODEL_CONFIG["dropout"] if dropout is None else dropout)
        dilations = tuple(
            MODEL_CONFIG["decoder_dilations"]
            if decoder_dilations is None
            else decoder_dilations
        )
        self.stem_base_channels = stem_channels
        self.branch_labels = tuple(f"k{size}" for size in kernel_sizes)
        self.stem_branches = nn.ModuleList(
            [StemBranch(stem_channels, size) for size in kernel_sizes]
        )
        stem_width = stem_channels * len(kernel_sizes)
        if stem_fusion == "conv1x1":
            self.stem_fuse = nn.Sequential(
                nn.Conv1d(stem_width, hidden, 1),
                nn.GroupNorm(choose_group_count(hidden), hidden),
            )
            width = hidden
        else:
            # Ablation: no 1x1 fusion and no GroupNorm after it. Every stem
            # branch already ends in GroupNorm + GELU, so the concatenation
            # is the encoder input.
            self.stem_fuse = nn.Identity()
            width = stem_width
        self.feature_channels = width
        self.encoder = make_encoder(width, dropout)
        self.decoder = make_breath_decoder_trunk(width, dilations, dropout)
        decoder_channels = MODEL_CONFIG["decoder_mid_channels"][-1]
        self.wave_head = nn.Conv1d(decoder_channels, 1, 1)
        # 261006_RRWaveNet_Ensemble had a 1x1 breath-event head here. Its
        # random draws (construction, Kaiming and N(0, 0.01) initialization)
        # are made and discarded so that a seed gives the same initial
        # weights, batch order and dropout as in 261006; the head itself is
        # not part of this model.
        removed_head = nn.Conv1d(decoder_channels, 1, 1)
        self._init_weights()
        nn.init.kaiming_normal_(removed_head.weight, mode="fan_out", nonlinearity="relu")
        nn.init.normal_(removed_head.weight, mean=0.0, std=0.01)
        del removed_head

    def _init_weights(self) -> None:
        for module in self.modules():
            if isinstance(module, (nn.Conv1d, nn.Linear)):
                nn.init.kaiming_normal_(
                    module.weight,
                    mode="fan_out",
                    nonlinearity="relu",
                )
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0.0)

    def fused_representation(self, x: torch.Tensor) -> torch.Tensor:
        stem = torch.cat([branch(x) for branch in self.stem_branches], dim=1)
        return self.stem_fuse(stem)

    def forward_with_diagnostics(
        self,
        x: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Return (waveform, post-fusion features)."""
        fused = self.fused_representation(x)
        features = self.decoder(self.encoder(fused))
        return torch.tanh(self.wave_head(features)), fused

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        waveform, _ = self.forward_with_diagnostics(x)
        return waveform


def count_parameters(model: nn.Module) -> int:
    return int(sum(p.numel() for p in model.parameters() if p.requires_grad))


# ---------------------------------------------------------------------------
# Exact v16 loss, training, checkpointing, and audit
# ---------------------------------------------------------------------------

class PearsonLoss(nn.Module):
    def forward(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
    ) -> torch.Tensor:
        pred = pred.squeeze(1)
        target = target.squeeze(1)
        pred_centered = pred - pred.mean(dim=1, keepdim=True)
        target_centered = target - target.mean(dim=1, keepdim=True)
        covariance = (pred_centered * target_centered).sum(dim=1)
        denominator = torch.sqrt(
            (pred_centered**2).sum(dim=1)
            * (target_centered**2).sum(dim=1)
        ) + 1e-6
        return 1.0 - (covariance / denominator).mean()


class DerivativeLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.base = nn.SmoothL1Loss()

    def forward(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
    ) -> torch.Tensor:
        return self.base(
            pred[:, :, 1:] - pred[:, :, :-1],
            target[:, :, 1:] - target[:, :, :-1],
        )


class HybridLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.time_loss = nn.SmoothL1Loss()
        self.corr_loss = PearsonLoss()
        self.shape_loss = DerivativeLoss()

    def forward(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
    ) -> torch.Tensor:
        w_time, w_corr, w_shape = LOSS_WEIGHTS
        return (
            w_time * self.time_loss(pred, target)
            + w_corr * self.corr_loss(pred, target)
            + w_shape * self.shape_loss(pred, target)
        )


class WaveformLoss(nn.Module):
    """The v16 hybrid waveform loss (the 260928 loss), with its parts."""

    def __init__(self):
        super().__init__()
        self.wave_loss = HybridLoss()

    def forward(
        self,
        wave_pred: torch.Tensor,
        wave_target: torch.Tensor,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        wave = self.wave_loss(wave_pred, wave_target)
        return wave, {"wave": wave.detach()}


def evaluate_loader(
    model: nn.Module,
    loader: DataLoader,
    criterion: WaveformLoss,
    device: torch.device,
) -> Tuple[float, Dict[str, float]]:
    """Mean validation loss over the loader."""
    model.eval()
    sums = {"total": 0.0, "wave": 0.0}
    batches = 0
    with torch.no_grad():
        for x, y, _, _ in loader:
            wave = model(x.to(device))
            loss, parts = criterion(wave, y.to(device))
            sums["total"] += float(loss.item())
            for key, value in parts.items():
                sums[key] += float(value.item())
            batches += 1
    if batches == 0:
        return np.nan, {key: np.nan for key in sums}
    means = {key: value / batches for key, value in sums.items()}
    return means["total"], means


def atomic_torch_save(state: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    torch.save(state, temporary)
    if path.exists():
        path.unlink()
    os.replace(temporary, path)


def build_training_audit(history: Sequence[dict], checkpoint: Path) -> dict:
    frame = pd.DataFrame(history)
    required = {"Epoch", "Train_Loss", "Val_Loss"}
    if frame.empty or not required.issubset(frame.columns):
        raise RuntimeError(f"Invalid training history for {checkpoint}")
    numeric = frame[["Epoch", "Train_Loss", "Val_Loss"]].apply(
        pd.to_numeric,
        errors="coerce",
    )
    if not np.isfinite(numeric.to_numpy(dtype=float)).all():
        raise RuntimeError(f"Non-finite loss detected: {checkpoint}")
    best_index = int(numeric["Val_Loss"].idxmin())
    best_row = numeric.loc[best_index]
    final_row = numeric.iloc[-1]
    best_position = int(numeric.index.get_loc(best_index))
    after_best = numeric.iloc[best_position + 1 :]
    val_increase = after_best["Val_Loss"] > best_row["Val_Loss"]
    train_decrease = after_best["Train_Loss"] < best_row["Train_Loss"]
    signature = val_increase & train_decrease
    return {
        "History_Epochs": int(len(numeric)),
        "Best_Epoch": int(best_row["Epoch"]),
        "Final_Epoch": int(final_row["Epoch"]),
        "Best_Train_Loss": float(best_row["Train_Loss"]),
        "Best_Val_Loss": float(best_row["Val_Loss"]),
        "Final_Train_Loss": float(final_row["Train_Loss"]),
        "Final_Val_Loss": float(final_row["Val_Loss"]),
        "Late_Val_Degradation": float(
            final_row["Val_Loss"] - best_row["Val_Loss"]
        ),
        "Generalization_Gap_At_Best": float(
            best_row["Val_Loss"] - best_row["Train_Loss"]
        ),
        "Post_Best_Overfit_Signature_Epochs": int(signature.sum()),
        "Post_Best_Overfit_Signature_Observed": int(signature.any()),
        "Losses_Finite": 1,
        "Best_Checkpoint_Exists": int(checkpoint.exists()),
        "Test_Evaluated_From_Best_Validation_Checkpoint": 1,
        "Model_Selection": "lowest_validation_loss_then_restore_checkpoint",
        "Overfit_Control": "early_stopping_plus_best_validation_checkpoint_restore",
        "Audit_Status": (
            "PASS_BEST_VALIDATION_CHECKPOINT"
            if checkpoint.exists()
            else "FAIL_MISSING_CHECKPOINT"
        ),
    }


def convergence_epoch_95pct(history: Sequence[dict]) -> int:
    """Return the first epoch reaching 95% of the initial-to-best val improvement."""
    frame = pd.DataFrame(history)
    if frame.empty:
        return 0
    values = pd.to_numeric(frame["Val_Loss"], errors="coerce").to_numpy(float)
    epochs = pd.to_numeric(frame["Epoch"], errors="coerce").to_numpy(int)
    finite = np.isfinite(values)
    if not finite.any():
        return 0
    values = values[finite]
    epochs = epochs[finite]
    best_index = int(np.argmin(values))
    initial = float(values[0])
    best = float(values[best_index])
    improvement = initial - best
    if improvement <= 1e-12:
        return int(epochs[best_index])
    threshold = initial - 0.95 * improvement
    reached = np.flatnonzero(values <= threshold)
    if len(reached) == 0:
        return int(epochs[best_index])
    return int(epochs[int(reached[0])])


# ---------------------------------------------------------------------------
# Test-time outputs and diagnostics of the fused representation
# ---------------------------------------------------------------------------

def predict_with_diagnostics(
    model: DeepOnlyV16Decoder,
    loader: DataLoader,
    device: torch.device,
) -> Dict[str, np.ndarray]:
    """Predicted waveforms, targets and post-fusion features."""
    model.eval()
    chunks = {
        "predictions": [],
        "targets": [],
        "features": [],
    }
    with torch.no_grad():
        for x, y, _, _ in loader:
            wave, fused = model.forward_with_diagnostics(x.to(device))
            chunks["predictions"].append(wave.squeeze(1).cpu().numpy())
            chunks["targets"].append(y.squeeze(1).numpy())
            chunks["features"].append(fused.cpu().numpy().astype(np.float32))
    if not chunks["predictions"]:
        empty = np.empty((0, 0), dtype=np.float32)
        outputs = {key: empty for key in chunks}
        outputs["features"] = np.empty((0, 0, 0), dtype=np.float32)
        return outputs
    return {
        key: np.concatenate(values).astype(np.float32)
        for key, values in chunks.items()
    }


def feature_representation_summary(features: np.ndarray) -> dict:
    """Summarize the 24-channel stem representation for one held-out subject."""
    result = {metric: np.nan for metric in FEATURE_METRICS}
    result["Feature_N_Windows"] = 0
    result["Feature_N_Samples"] = 0
    values = np.asarray(features, dtype=np.float64)
    if values.ndim != 3 or values.shape[0] == 0:
        return result
    windows, channels, samples = values.shape
    matrix = np.transpose(values, (1, 0, 2)).reshape(channels, -1)
    matrix = np.nan_to_num(matrix, nan=0.0, posinf=0.0, neginf=0.0)
    if channels < 2 or matrix.shape[1] < 3:
        return result
    matrix = matrix - np.mean(matrix, axis=1, keepdims=True)
    covariance = (matrix @ matrix.T) / max(1, matrix.shape[1] - 1)
    covariance = (covariance + covariance.T) / 2.0
    eigenvalues = np.clip(np.linalg.eigvalsh(covariance), 0.0, None)
    total = float(eigenvalues.sum())
    if total > 1e-12:
        probabilities = eigenvalues / total
        positive = probabilities > 0
        effective_rank = float(
            np.exp(-np.sum(probabilities[positive] * np.log(probabilities[positive])))
        )
        result["Feature_EffectiveRank"] = effective_rank
        result["Feature_EffectiveRank_Normalized"] = effective_rank / channels
        denom = float(np.sum(eigenvalues**2))
        if denom > 1e-12:
            result["Feature_ParticipationRatio"] = float(total**2 / denom)
    std = np.sqrt(np.maximum(np.diag(covariance), 0.0))
    denominator = np.outer(std, std)
    with np.errstate(divide="ignore", invalid="ignore"):
        correlation = covariance / denominator
    correlation = np.where(np.isfinite(correlation), correlation, 0.0)
    off_diagonal = correlation[~np.eye(channels, dtype=bool)]
    if off_diagonal.size:
        result["Feature_OffDiag_SignedMean"] = float(np.mean(off_diagonal))
        result["Feature_OffDiag_AbsMean"] = float(np.mean(np.abs(off_diagonal)))
    result["Feature_N_Windows"] = int(windows)
    result["Feature_N_Samples"] = int(matrix.shape[1])
    return result


# ---------------------------------------------------------------------------
# One LOSOCV fold
# ---------------------------------------------------------------------------

def make_optimizer(model: DeepOnlyV16Decoder, epochs: int, weight_decay: Optional[float] = None):
    """AdamW. Without weight_decay (261007): WEIGHT_DECAY on every parameter.
    With weight_decay (261013): the decay on the convolution weights only;
    biases and GroupNorm scale / shift are not decayed."""
    if weight_decay is None:
        optimizer = torch.optim.AdamW(
            [{"params": list(model.parameters())}],
            lr=LEARNING_RATE,
            weight_decay=WEIGHT_DECAY,
        )
    else:
        decay = [p for p in model.parameters() if p.ndim >= 2]
        no_decay = [p for p in model.parameters() if p.ndim < 2]
        optimizer = torch.optim.AdamW(
            [
                {"params": decay, "weight_decay": float(weight_decay)},
                {"params": no_decay, "weight_decay": 0.0},
            ],
            lr=LEARNING_RATE,
        )
    return optimizer, CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)


def fit_model(
    model: DeepOnlyV16Decoder,
    train_loader: DataLoader,
    val_loader: DataLoader,
    device: torch.device,
    epochs: int,
    min_epochs: int,
    patience: int,
    checkpoint: Path,
    state_path: Optional[Path] = None,
    weight_decay: Optional[float] = None,
) -> List[dict]:
    """Train with early stopping on the validation loss.

    The best validation state is saved to `checkpoint`. The cosine schedule
    spans `epochs`. With `state_path`, the full training state (weights,
    optimizer, schedule, early-stopping counters, history and the random
    number generators) is saved after every epoch, and an existing state is
    continued from, so an interrupted run resumes exactly where it stopped.
    """
    criterion = WaveformLoss()
    optimizer, scheduler = make_optimizer(model, epochs, weight_decay)
    best_val = np.inf
    patience_counter = 0
    history = []
    start_epoch = 0
    if state_path is not None and state_path.exists():
        state = torch.load(state_path, map_location=device, weights_only=False)
        model.load_state_dict(state["model"])
        optimizer.load_state_dict(state["optimizer"])
        scheduler.load_state_dict(state["scheduler"])
        best_val = float(state["best_val"])
        patience_counter = int(state["patience_counter"])
        history = list(state["history"])
        start_epoch = int(state["epoch"])
        random.setstate(state["python_rng"])
        np.random.set_state(state["numpy_rng"])
        torch.set_rng_state(state["torch_rng"])
        if torch.cuda.is_available() and state.get("cuda_rng") is not None:
            torch.cuda.set_rng_state_all(state["cuda_rng"])
        if state.get("stopped", False):
            return history
        print(f"    resumed after epoch {start_epoch} (best val {best_val:.4f})", flush=True)
    for epoch in range(start_epoch, epochs):
        model.train()
        train_loss = 0.0
        for x, y, _, _ in train_loader:
            optimizer.zero_grad(set_to_none=True)
            wave = model(x.to(device))
            loss, _ = criterion(wave, y.to(device))
            if not torch.isfinite(loss):
                raise RuntimeError(f"Non-finite training loss at epoch {epoch + 1}")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            train_loss += float(loss.item())
        train_loss /= max(1, len(train_loader))
        val_loss, val_parts = evaluate_loader(model, val_loader, criterion, device)
        if not np.isfinite(val_loss):
            raise RuntimeError(f"Non-finite validation loss at epoch {epoch + 1}")
        row = {
            "Epoch": epoch + 1,
            "Train_Loss": train_loss,
            "Val_Loss": val_loss,
            "Val_Wave_Loss": val_parts["wave"],
        }
        epoch_lr = float(optimizer.param_groups[0]["lr"])
        scheduler.step()
        history.append(row)
        improved = row["Val_Loss"] < best_val - 1e-4
        if improved:
            best_val = row["Val_Loss"]
            atomic_torch_save(model.state_dict(), checkpoint)
            patience_counter = 0
        elif epoch + 1 >= min_epochs:
            patience_counter += 1
        stop = epoch + 1 >= min_epochs and patience_counter >= patience
        # Progress output only.
        print(
            f"    epoch {epoch + 1:3d}/{epochs}  train loss {train_loss:.4f}  val loss {val_loss:.4f}  "
            f"lr {epoch_lr:.2e}  best val {best_val:.4f}{' *' if improved else '  '}  "
            f"patience {patience_counter}/{patience}",
            flush=True,
        )
        if stop:
            print(
                f"    early stop after epoch {epoch + 1}: no validation improvement for {patience} epochs",
                flush=True,
            )
        if state_path is not None:
            atomic_torch_save(
                {
                    "epoch": epoch + 1,
                    "stopped": bool(stop),
                    "model": model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "scheduler": scheduler.state_dict(),
                    "best_val": best_val,
                    "patience_counter": patience_counter,
                    "history": history,
                    "python_rng": random.getstate(),
                    "numpy_rng": np.random.get_state(),
                    "torch_rng": torch.get_rng_state(),
                    "cuda_rng": torch.cuda.get_rng_state_all()
                    if torch.cuda.is_available()
                    else None,
                },
                state_path,
            )
        if stop:
            break
    return history


def train_one(
    dataset_key: str,
    spec: WindowSpec,
    test_subject: str,
    subject_data: Dict[str, dict],
    train_subjects: Sequence[str],
    member_root: Path,
    device: torch.device,
    epochs: int,
    min_epochs: int,
    patience: int,
    batch_size: int,
    seed: int,
    model_options: ModelOptions,
    save_features: bool = False,
    resume: bool = False,
    val_subjects: Optional[Sequence[str]] = None,
    weight_decay: Optional[float] = None,
    model_key: Optional[str] = None,
) -> Tuple[Dict[str, np.ndarray], List[dict], dict, Path, dict]:
    """Train the model of one LOSOCV fold and predict the held-out subject.

    Validation: without val_subjects the last 20% of each training subject's
    windows (v16, 261007); with val_subjects every window of those subjects,
    trained on every window of the other training subjects (subject-wise,
    261013). weight_decay: None = 261007 (WEIGHT_DECAY on every parameter),
    else that decay on the convolution weights (261013). The best validation
    state is restored before testing. The training state is
    saved after every epoch; with `resume` an interrupted member continues
    from it, otherwise it starts over.
    """
    set_seed(seed)
    model_root = member_root / "Models"
    curve_root = member_root / "Learning_Curves"
    audit_root = member_root / "Training_Audits"
    prediction_root = member_root / "Predictions"
    for path in (model_root, curve_root, audit_root, prediction_root):
        path.mkdir(parents=True, exist_ok=True)

    tag = safe_tag(test_subject)
    checkpoint = model_root / f"{tag}_best.pth"
    history_path = curve_root / f"{tag}_Learning_Curve.csv"
    audit_path = audit_root / f"{tag}_Training_Audit.csv"
    prediction_path = prediction_root / f"{tag}_test_outputs.npz"
    state_path = model_root / f"{tag}_training_state.pt"
    if not resume and state_path.exists():
        state_path.unlink()

    if val_subjects is None:
        fit_subjects, held_out = list(train_subjects), set()
        train_dataset = SubjectWindowDataset(subject_data, train_subjects, "train", True)
        val_dataset = SubjectWindowDataset(subject_data, train_subjects, "val", False)
        validation_mode = "temporal (last 20% of each training subject)"
    else:
        held_out = {str(subject) for subject in val_subjects}
        fit_subjects = [subject for subject in train_subjects if str(subject) not in held_out]
        if str(test_subject) in held_out or not fit_subjects:
            raise RuntimeError(f"{test_subject}: invalid validation subjects {sorted(held_out)}")
        train_dataset = SubjectWindowDataset(subject_data, fit_subjects, "train", True, subject_wise=True)
        val_dataset = SubjectWindowDataset(subject_data, list(val_subjects), "val", False, subject_wise=True)
        validation_mode = (
            f"subject-wise ({len(held_out)} validation subjects, {len(fit_subjects)} training subjects)"
        )
    test_dataset = SubjectWindowDataset(subject_data, [test_subject], "test", False)
    train_loader = make_train_loader(train_dataset, batch_size, device)
    val_loader = make_eval_loader(val_dataset, batch_size, device)
    test_loader = make_eval_loader(test_dataset, batch_size, device)
    metadata = test_metadata(subject_data, test_subject)

    model = model_options.build().to(device)
    history = fit_model(
        model,
        train_loader,
        val_loader,
        device,
        epochs,
        min_epochs,
        patience,
        checkpoint,
        state_path,
        weight_decay,
    )
    history_frame = pd.DataFrame(history)
    history_frame.to_csv(history_path, index=False, encoding="utf-8-sig")
    audit = build_training_audit(history, checkpoint)
    audit["Convergence_Epoch_95pct"] = convergence_epoch_95pct(history)
    if audit["Audit_Status"] != "PASS_BEST_VALIDATION_CHECKPOINT":
        raise RuntimeError(f"No valid checkpoint: {checkpoint}")
    audit["Validation_Mode"] = validation_mode
    pd.DataFrame([audit]).to_csv(audit_path, index=False, encoding="utf-8-sig")

    model.load_state_dict(torch.load(checkpoint, map_location=device))
    outputs = predict_with_diagnostics(model, test_loader, device)
    features = outputs.pop("features")
    best_row = history_frame.loc[history_frame["Epoch"] == audit["Best_Epoch"]].iloc[0]
    diagnostics = {
        "N_Params": count_parameters(model),
        "Weight_Decay": float(WEIGHT_DECAY if weight_decay is None else weight_decay),
        "Weight_Decay_Scope": "all parameters" if weight_decay is None else "convolution weights",
        "Validation_Mode": validation_mode,
        "N_Train_Subjects": len(fit_subjects),
        "N_Val_Subjects": len(held_out),
        "Train_Windows": len(train_loader.dataset),
        "Val_Windows": len(val_loader.dataset),
        "Val_Wave_Loss_At_Best": float(best_row["Val_Wave_Loss"]),
    }
    if held_out:
        diagnostics["Validation_Subjects"] = " ".join(sorted(held_out))
    diagnostics.update(feature_representation_summary(features))
    arrays = {
        "predictions": outputs["predictions"],
        "targets": outputs["targets"],
        "segment_ids": np.asarray(
            [item["segment_id"] for item in metadata],
            dtype=np.int64,
        ),
        "window_starts": np.asarray(
            [item["start"] for item in metadata],
            dtype=np.int64,
        ),
        "fs": np.asarray(TARGET_FS, dtype=np.int32),
        "input_window_sec": np.asarray(spec.window_sec, dtype=np.float32),
        "input_stride_sec": np.asarray(spec.stride_sec, dtype=np.float32),
        "model_key": np.asarray(model_key or model_options.key),
        "seed": np.asarray(seed, dtype=np.int64),
    }
    if save_features:
        arrays["feature_representation"] = features.astype(np.float32)
        arrays["feature_representation_location"] = np.asarray(
            "post_stem_fusion"
        )
    np.savez_compressed(prediction_path, **arrays)
    if state_path.exists():
        state_path.unlink()
    del features, arrays, train_loader, val_loader, test_loader
    del model
    release_memory(device)
    return outputs, metadata, audit, prediction_path, diagnostics


# ---------------------------------------------------------------------------
# Timeline reconstruction and direct 60-second RR evaluation
# ---------------------------------------------------------------------------

def safe_pcc(x: np.ndarray, y: np.ndarray) -> float:
    x_arr, _ = sanitize_signal(x)
    y_arr, _ = sanitize_signal(y)
    length = min(len(x_arr), len(y_arr))
    x_arr = x_arr[:length]
    y_arr = y_arr[:length]
    if length < 2:
        return np.nan
    if np.std(x_arr) < 1e-8 or np.std(y_arr) < 1e-8:
        return np.nan
    return float(stats.pearsonr(x_arr, y_arr)[0])


def reconstruct_segment(
    predictions: np.ndarray,
    targets: np.ndarray,
    metadata: Sequence[dict],
    segment_id: int,
    fs: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    selected = [
        index
        for index, item in enumerate(metadata)
        if int(item["segment_id"]) == int(segment_id)
    ]
    if not selected:
        return (
            np.empty(0, dtype=np.float64),
            np.empty(0, dtype=np.float64),
            np.empty(0, dtype=bool),
        )
    window_len = int(predictions.shape[1])
    max_end = max(int(metadata[index]["start"]) for index in selected)
    max_end += window_len
    pred_sum = np.zeros(max_end, dtype=np.float64)
    true_sum = np.zeros(max_end, dtype=np.float64)
    coverage = np.zeros(max_end, dtype=np.int32)
    for index in selected:
        start = int(metadata[index]["start"])
        end = start + window_len
        pred = np.asarray(predictions[index], dtype=np.float64)
        true = np.asarray(targets[index], dtype=np.float64)
        pred, _ = sanitize_signal(pred)
        true, _ = sanitize_signal(true)
        pred_sum[start:end] += pred[:window_len]
        true_sum[start:end] += true[:window_len]
        coverage[start:end] += 1
    valid = coverage > 0
    pred_out = np.full(max_end, np.nan, dtype=np.float64)
    true_out = np.full(max_end, np.nan, dtype=np.float64)
    pred_out[valid] = pred_sum[valid] / coverage[valid]
    true_out[valid] = true_sum[valid] / coverage[valid]
    return pred_out, true_out, valid


def contiguous_runs(mask: np.ndarray) -> List[Tuple[int, int]]:
    mask = np.asarray(mask, dtype=bool)
    if not mask.any():
        return []
    padded = np.concatenate([[False], mask, [False]])
    changes = np.flatnonzero(padded[1:] != padded[:-1])
    return [(int(left), int(right)) for left, right in changes.reshape(-1, 2)]


REFERENCE_EXCLUSION_REASONS = (
    "ref_structural_invalid",
    "ref_artifact_label",
    "ref_rr_undefined",
    "ref_irregular_cycles",
    "ref_disagreement",
    "ref_out_of_band",
)


def _mean_or_nan(values: Sequence[float]) -> float:
    values = np.asarray(values, dtype=np.float64)
    values = values[np.isfinite(values)]
    return float(np.mean(values)) if len(values) else np.nan


def summarize_subject_minutes(frame: pd.DataFrame) -> dict:
    """Subject-level summary of the per-minute evaluation table."""
    summary = {
        "N60": 0,
        "N60_RefValid": 0,
        "Ref_Valid_Ratio": np.nan,
        **{f"N60_Excluded_{reason}": 0 for reason in REFERENCE_EXCLUSION_REASONS},
        "RR_MAE_Wave_FFT": np.nan,
        "RR_Bias_Wave_FFT": np.nan,
        "Wave_PCC_Mean": np.nan,
    }
    if frame.empty:
        return summary
    summary["N60"] = int(len(frame))
    for reason in REFERENCE_EXCLUSION_REASONS:
        summary[f"N60_Excluded_{reason}"] = int(
            (frame["Ref_Exclusion_Reason"] == reason).sum()
        )
    valid = frame[frame["Ref_Valid"].astype(bool)]
    summary["N60_RefValid"] = int(len(valid))
    summary["Ref_Valid_Ratio"] = float(len(valid) / len(frame))
    if valid.empty:
        return summary
    error = valid["Pred_Wave_FFT_RR"] - valid["GT_Breath_Count"] * (60.0 / EVAL_WINDOW_SEC)
    summary.update(
        {
            "RR_MAE_Wave_FFT": _mean_or_nan(valid["AE_Wave_FFT"]),
            "RR_Bias_Wave_FFT": _mean_or_nan(error),
            "Wave_PCC_Mean": _mean_or_nan(valid["Wave_PCC"]),
        }
    )
    return summary


def evaluate_subject_minutes(
    outputs: Dict[str, np.ndarray],
    metadata: Sequence[dict],
    references: Dict[int, dict],
    fs: int,
) -> Tuple[dict, pd.DataFrame]:
    """Score every complete 60 s minute of one held-out subject (waveform only).

    Minutes are non-overlapping 60 s slices of each contiguous run of
    reconstructed windows (overlap-add mean of the window outputs). Per minute:

    * reference: with expert labels (CapnoBase) the labels are the ground
      truth and a minute needs a valid reference, two labelled breaths and a
      rate in band; otherwise the minute of the continuous reference must
      pass the RRest agreement rule and the ground truth is the number of
      breath onsets detected once on the continuous reference. Excluded
      minutes keep a reason. Band and rate range follow the segment's profile;
    * breathing rate: largest FFT peak of the predicted waveform within the
      profile's rate band, after the profile's respiratory filter is applied
      to the whole run;
    * waveform agreement: Pearson r of the predicted and target waveforms.
    """
    predictions = outputs["predictions"]
    targets = outputs["targets"]
    if len(predictions) == 0:
        return summarize_subject_minutes(pd.DataFrame()), pd.DataFrame()

    eval_len = int(round(EVAL_WINDOW_SEC * fs))
    per_minute = 60.0 / EVAL_WINDOW_SEC
    rows = []
    for segment_id in sorted({int(item["segment_id"]) for item in metadata}):
        pred_series, true_series, coverage = reconstruct_segment(
            predictions, targets, metadata, segment_id, fs
        )
        reference = references[segment_id]
        ref_signal = reference["signal"]
        ref_breaths = reference["breaths"]
        ref_structural = reference["structural_valid"]
        profile = reference["profile"]
        rr_band = profile["rr_band_bpm"]
        subharmonic = profile["fft_subharmonic_check"]
        for run_index, (run_start, run_end) in enumerate(
            contiguous_runs(coverage),
            start=1,
        ):
            if run_end - run_start < eval_len:
                continue
            minute_starts = list(range(run_start, run_end - eval_len + 1, eval_len))
            wave_run = respiratory_filter(
                pred_series[run_start:run_end], fs, reference["band_hz"], profile
            )
            for local_index, start in enumerate(minute_starts, start=1):
                end = start + eval_len
                # Reference validity and fixed ground truth. With expert labels
                # the labels are the ground truth: the minute needs a valid
                # reference signal, at least two labelled breaths and an RR in
                # the profile's band. Without labels the agreement rule decides.
                in_minute = (ref_breaths >= start) & (ref_breaths < end)
                gt_count = int(in_minute.sum())
                gt_rate = gt_count * per_minute
                minute_labels = ref_breaths[in_minute]
                label_rr = (
                    float(60.0 * fs / np.mean(np.diff(minute_labels)))
                    if reference["label_based"] and len(minute_labels) >= 2
                    else np.nan
                )
                structural = float(np.mean(ref_structural[start:end]))
                ref_artifact = (
                    float(np.mean(reference["artifact_mask"][start:end]))
                    if reference.get("artifact_mask") is not None
                    else 0.0
                )
                ppg_artifact = (
                    float(np.mean(reference["ppg_artifact_mask"][start:end]))
                    if reference.get("ppg_artifact_mask") is not None
                    else 0.0
                )
                _, _, ref_rr_cto, _ = count_orig_breaths(ref_signal[start:end], fs)
                ref_rr_fft = fft_rr_bpm(ref_signal[start:end], fs, rr_band, subharmonic)
                if structural < MIN_USABLE_PROPORTION:
                    ref_valid, reason = False, "ref_structural_invalid"
                elif ref_artifact > 0.0:
                    # Expert-labelled reference artifact: breaths inside it
                    # may be unlabelled, so the minute has no ground truth.
                    ref_valid, reason = False, "ref_artifact_label"
                elif reference["label_based"]:
                    if not np.isfinite(label_rr):
                        ref_valid, reason = False, "ref_rr_undefined"
                    elif not (rr_band[0] <= label_rr <= rr_band[1]):
                        ref_valid, reason = False, "ref_out_of_band"
                    else:
                        ref_valid, reason = True, ""
                else:
                    ref_valid, reason, _, _ = reference_agreement(
                        ref_signal[start:end], fs, rr_band, subharmonic
                    )

                # Breathing rate of the predicted waveform. The f/2, f/3 check
                # is for capnogram references; on predicted waveforms it
                # halved fast rates, so it is not used here.
                wave_fft = fft_rr_bpm(
                    wave_run[start - run_start : end - run_start],
                    fs,
                    rr_band,
                    False,
                )
                rows.append(
                    {
                        "Segment_ID": segment_id,
                        "Coverage_Run_Index": run_index,
                        "RR_Window_Index": local_index,
                        "RR_Window_Start_Sec": start / float(fs),
                        "RR_Window_End_Sec": end / float(fs),
                        "Ref_Valid": bool(ref_valid),
                        "Ref_Exclusion_Reason": reason,
                        "Ref_Structural_Fraction": structural,
                        "Ref_Artifact_Label_Fraction": ref_artifact,
                        "PPG_Artifact_Label_Fraction": ppg_artifact,
                        "Ref_RR_CtO": ref_rr_cto,
                        "Ref_RR_FFT": ref_rr_fft,
                        "Ref_RR_Label": label_rr,
                        "GT_Source": "expert_labels"
                        if reference["label_based"]
                        else "detected_breaths",
                        "GT_Breath_Count": gt_count,
                        "Pred_Wave_FFT_RR": wave_fft,
                        "AE_Wave_FFT": abs(wave_fft - gt_rate)
                        if np.isfinite(wave_fft)
                        else np.nan,
                        "Wave_PCC": safe_pcc(true_series[start:end], pred_series[start:end]),
                    }
                )
    frame = pd.DataFrame(rows)
    return summarize_subject_minutes(frame), frame


# ---------------------------------------------------------------------------
# Statistics, plots, and output tables
# ---------------------------------------------------------------------------

def safe_pvalue(function, x: np.ndarray, y: np.ndarray) -> float:
    try:
        if len(x) < 2 or len(y) < 2:
            return np.nan
        if np.allclose(np.asarray(x, dtype=float), np.asarray(y, dtype=float)):
            return 1.0
        return float(function(x, y).pvalue)
    except Exception:
        return np.nan


def bh_fdr(values: Sequence[float]) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    output = np.full(values.shape, np.nan, dtype=float)
    finite = np.isfinite(values)
    if not finite.any():
        return output
    raw = values[finite]
    order = np.argsort(raw)
    ranked = raw[order]
    adjusted = ranked * len(ranked) / np.arange(1, len(ranked) + 1)
    adjusted = np.minimum.accumulate(adjusted[::-1])[::-1]
    adjusted = np.clip(adjusted, 0.0, 1.0)
    restored = np.empty_like(adjusted)
    restored[order] = adjusted
    output[finite] = restored
    return output


MINUTE_KEY = ["Subject", "Segment_ID", "RR_Window_Start_Sec"]


def scored_minutes(frame: pd.DataFrame) -> pd.DataFrame:
    """Reference-valid minutes with the FFT rate error (breaths/min)."""
    if frame.empty:
        return frame.copy()
    valid = frame[frame["Ref_Valid"].astype(bool)].copy()
    valid["Subject"] = valid["Subject"].astype(str)
    valid["GT_RR"] = pd.to_numeric(valid["GT_Breath_Count"], errors="coerce") * (
        60.0 / EVAL_WINDOW_SEC
    )
    valid["FFT_RR"] = pd.to_numeric(valid["Pred_Wave_FFT_RR"], errors="coerce")
    valid["FFT_Error"] = valid["FFT_RR"] - valid["GT_RR"]
    valid["FFT_AE"] = valid["FFT_Error"].abs()
    return valid


def sd(values) -> float:
    """Sample SD (ddof=1) of the finite values."""
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    return float(np.std(values, ddof=1)) if values.size > 1 else np.nan


def fft_davies_metrics(minutes: pd.DataFrame) -> dict:
    """FFT metrics over all minutes and the Davies & Mandic (2022) medians.

    mAE: median absolute error over all minutes; mMAE: median over subjects
    of each subject's mean absolute error. Every mean comes with its SD
    (ddof=1): FFT_MAE_SD and FFT_Bias_SD over the minutes (LoA = bias +- 1.96
    SD), FFT_Subject_MAE_SD over the subjects, Wave_PCC_SD over the minutes.
    Minutes without an FFT peak are counted in N_Minutes_No_FFT and left out
    of the error statistics.
    """
    result = {"N_Subjects": 0, "N_Minutes": 0, "N_Minutes_No_FFT": 0}
    result.update({metric: np.nan for metric in SUMMARY_METRICS})
    if minutes.empty:
        return result
    pcc = pd.to_numeric(minutes["Wave_PCC"], errors="coerce").dropna()
    if len(pcc):
        result["Wave_PCC_Mean"] = float(pcc.mean())
        result["Wave_PCC_SD"] = sd(pcc)
        result["Wave_PCC_Median"] = float(pcc.median())
    have = minutes[np.isfinite(minutes["FFT_AE"])]
    result["N_Minutes"] = int(len(minutes))
    result["N_Minutes_No_FFT"] = int(len(minutes) - len(have))
    if have.empty:
        return result
    errors = have["FFT_Error"].to_numpy(dtype=float)
    absolute = np.abs(errors)
    subject_mae = have.groupby("Subject")["FFT_AE"].mean()
    error_sd = sd(errors)
    result.update(
        {
            "N_Subjects": int(subject_mae.size),
            "FFT_MAE": float(np.mean(absolute)),
            "FFT_MAE_SD": sd(absolute),
            "FFT_RMSE": float(np.sqrt(np.mean(errors**2))),
            "FFT_Bias": float(np.mean(errors)),
            "FFT_Bias_SD": error_sd,
            "FFT_LoA_Lower": float(np.mean(errors) - 1.96 * error_sd),
            "FFT_LoA_Upper": float(np.mean(errors) + 1.96 * error_sd),
            "FFT_Pearson_r": safe_pcc(
                have["FFT_RR"].to_numpy(dtype=float),
                have["GT_RR"].to_numpy(dtype=float),
            ),
            "FFT_Within_1_BPM_Pct": float(np.mean(absolute <= 1.0) * 100.0),
            "FFT_Within_2_BPM_Pct": float(np.mean(absolute <= 2.0) * 100.0),
            "FFT_Subject_MAE_Mean": float(subject_mae.mean()),
            "FFT_Subject_MAE_SD": sd(subject_mae),
            "Davies_mAE": float(np.median(absolute)),
            "Davies_mAE_Q1": float(np.percentile(absolute, 25)),
            "Davies_mAE_Q3": float(np.percentile(absolute, 75)),
            "Davies_mMAE": float(subject_mae.median()),
            "Davies_mMAE_Q1": float(subject_mae.quantile(0.25)),
            "Davies_mMAE_Q3": float(subject_mae.quantile(0.75)),
        }
    )
    return result


SUMMARY_METRICS = (
    "FFT_MAE",
    "FFT_MAE_SD",
    "FFT_RMSE",
    "FFT_Bias",
    "FFT_Bias_SD",
    "FFT_LoA_Lower",
    "FFT_LoA_Upper",
    "FFT_Pearson_r",
    "FFT_Within_1_BPM_Pct",
    "FFT_Within_2_BPM_Pct",
    "FFT_Subject_MAE_Mean",
    "FFT_Subject_MAE_SD",
    "Davies_mAE",
    "Davies_mAE_Q1",
    "Davies_mAE_Q3",
    "Davies_mMAE",
    "Davies_mMAE_Q1",
    "Davies_mMAE_Q3",
    "Wave_PCC_Mean",
    "Wave_PCC_SD",
    "Wave_PCC_Median",
)


def subject_errors(minutes: pd.DataFrame) -> pd.Series:
    """Per-subject FFT MAE."""
    have = minutes[np.isfinite(minutes["FFT_AE"])]
    return have.groupby("Subject")["FFT_AE"].mean()


def minute_errors(minutes: pd.DataFrame) -> pd.Series:
    """Per-minute FFT absolute error, indexed by subject, segment and minute."""
    have = minutes[np.isfinite(minutes["FFT_AE"])]
    return have.set_index(MINUTE_KEY)["FFT_AE"]


MODEL_COMPARISON_METRICS = (
    "N_Minutes",
    "FFT_MAE",
    "FFT_MAE_SD",
    "FFT_RMSE",
    "FFT_Bias",
    "FFT_Bias_SD",
    "FFT_Pearson_r",
    "FFT_Within_2_BPM_Pct",
    "FFT_Subject_MAE_Mean",
    "FFT_Subject_MAE_SD",
    "Davies_mAE",
    "Davies_mMAE",
    "Wave_PCC_Mean",
    "Wave_PCC_SD",
)


def compare_models(minutes: pd.DataFrame) -> pd.DataFrame:
    """Every pair of models, per dataset, on the subjects scored by both: the
    metrics of both side by side, the per-subject MAE difference (mean +-
    SD), paired Wilcoxon tests of the per-subject FFT MAE, of the
    per-subject waveform PCC (mean of its minutes) and of the per-minute
    absolute error (Benjamini-Hochberg over the pairs of one dataset)."""
    rows = []
    if minutes.empty or "Variant" not in minutes.columns:
        return pd.DataFrame()
    order = [variant.label for variant in VARIANTS.values()]
    for dataset, frame in minutes.groupby("Dataset", sort=False):
        present = set(frame["Variant"])
        models = [name for name in order if name in present]
        block = []
        for position, first in enumerate(models):
            for second in models[position + 1 :]:
                reference = frame[frame["Variant"] == first]
                compared = frame[frame["Variant"] == second]
                a, b = subject_errors(reference), subject_errors(compared)
                subjects = a.index.intersection(b.index)
                metrics_a = fft_davies_metrics(reference[reference["Subject"].isin(subjects)])
                metrics_b = fft_davies_metrics(compared[compared["Subject"].isin(subjects)])
                sa = a.loc[subjects].to_numpy(float)
                sb = b.loc[subjects].to_numpy(float)
                difference = sb - sa
                pa = reference.groupby("Subject")["Wave_PCC"].mean()
                pb = compared.groupby("Subject")["Wave_PCC"].mean()
                pcc_subjects = pa.index.intersection(pb.index)
                pcc_a = pa.loc[pcc_subjects].to_numpy(float)
                pcc_b = pb.loc[pcc_subjects].to_numpy(float)
                pcc_difference = pcc_b - pcc_a
                ma, mb = minute_errors(reference), minute_errors(compared)
                keys = ma.index.intersection(mb.index)
                minute_a = ma.loc[keys].to_numpy(float)
                minute_b = mb.loc[keys].to_numpy(float)
                row = {
                    "Dataset": dataset,
                    "Reference_Model": first,
                    "Compared_Model": second,
                    "N_Subjects_Paired": int(len(subjects)),
                }
                for key in MODEL_COMPARISON_METRICS:
                    row[f"Reference_{key}"] = metrics_a[key]
                    row[f"Compared_{key}"] = metrics_b[key]
                row.update(
                    {
                        "Subject_MAE_Diff_Compared_minus_Reference": float(np.mean(difference))
                        if len(difference)
                        else np.nan,
                        "Subject_MAE_Diff_SD": sd(difference),
                        "N_Subjects_Compared_Better": int(np.sum(difference < -1e-9)),
                        "N_Subjects_Reference_Better": int(np.sum(difference > 1e-9)),
                        "N_Subjects_Tied": int(np.sum(np.abs(difference) <= 1e-9)),
                        "Subject_Wilcoxon_p": safe_pvalue(stats.wilcoxon, sa, sb),
                        "Subject_PCC_Reference": float(np.mean(pcc_a)) if len(pcc_a) else np.nan,
                        "Subject_PCC_Reference_SD": sd(pcc_a),
                        "Subject_PCC_Compared": float(np.mean(pcc_b)) if len(pcc_b) else np.nan,
                        "Subject_PCC_Compared_SD": sd(pcc_b),
                        "Subject_PCC_Diff_Compared_minus_Reference": float(np.mean(pcc_difference))
                        if len(pcc_difference)
                        else np.nan,
                        "Subject_PCC_Diff_SD": sd(pcc_difference),
                        "N_Subjects_Compared_Higher_PCC": int(np.sum(pcc_difference > 0)),
                        "Subject_PCC_Wilcoxon_p": safe_pvalue(stats.wilcoxon, pcc_a, pcc_b),
                        "N_Minutes_Paired": int(len(keys)),
                        "Minute_MAE_Reference": float(np.mean(minute_a)) if len(keys) else np.nan,
                        "Minute_MAE_Reference_SD": sd(minute_a),
                        "Minute_MAE_Compared": float(np.mean(minute_b)) if len(keys) else np.nan,
                        "Minute_MAE_Compared_SD": sd(minute_b),
                        "Minute_Wilcoxon_p": safe_pvalue(stats.wilcoxon, minute_a, minute_b),
                    }
                )
                block.append(row)
        for key in ("Subject_Wilcoxon_p", "Subject_PCC_Wilcoxon_p", "Minute_Wilcoxon_p"):
            for row, value in zip(block, bh_fdr([row[key] for row in block])):
                row[key.replace("_p", "_FDR_p")] = value
        rows += block
    return pd.DataFrame(rows)


def model_layer_table(variant: Variant) -> pd.DataFrame:
    """Every convolution of one model: channels, kernel, dilation, the span
    (k - 1) d + 1 it covers, its parameters and its multiply-accumulates for
    one input window (bias and GroupNorm parameters are in the model total)."""
    model = variant.options.build().eval()
    names = {module: name for name, module in model.named_modules()}
    rows: List[dict] = []

    def record(module: nn.Conv1d, inputs, output) -> None:
        kernel, dilation = int(module.kernel_size[0]), int(module.dilation[0])
        span = (kernel - 1) * dilation + 1
        rows.append(
            {
                "Model": variant.label,
                "Layer": names[module],
                "In_Channels": int(module.in_channels),
                "Out_Channels": int(module.out_channels),
                "Groups": int(module.groups),
                "Kernel": kernel,
                "Dilation": dilation,
                "Span_Samples": span,
                "Span_Sec": span / TARGET_FS,
                "Parameters": int(sum(p.numel() for p in module.parameters())),
                "MACs_Per_Window": int(
                    output.shape[-1]
                    * module.out_channels
                    * (module.in_channels // module.groups)
                    * kernel
                ),
            }
        )

    hooks = [
        module.register_forward_hook(record)
        for module in model.modules()
        if isinstance(module, nn.Conv1d)
    ]
    window = int(round(WINDOW_SPECS[0].window_sec * TARGET_FS))
    with torch.no_grad():
        model(torch.zeros(1, 1, window))
    for hook in hooks:
        hook.remove()
    return pd.DataFrame(rows)


def model_variant_table(variants: Sequence[Variant]) -> pd.DataFrame:
    """Stem branches, receptive fields, parameters and MACs of every model."""
    rows = []
    for variant in variants:
        model = variant.options.build()
        layers = model_layer_table(variant)
        row = {
            "Model": variant.label,
            "Folder": variant.name,
            "Model_Key": variant.options.key,
            "Stem_Branches": variant.options.stem_description,
            "Stem_Channels": variant.options.stem_channels,
        }
        row.update(variant.options.receptive_fields())
        row.update(
            {
                "Parameters": count_parameters(model),
                "Stem_Parameters": count_parameters(model.stem_branches),
                "Stem_Fusion_Parameters": count_parameters(model.stem_fuse),
                "MACs_Per_10s_Window": int(layers["MACs_Per_Window"].sum()),
            }
        )
        rows.append(row)
    return pd.DataFrame(rows)


def per_subject_table(minutes: pd.DataFrame) -> pd.DataFrame:
    """Every subject: minutes, median true and predicted rate, MAE, bias and
    waveform PCC, each mean with its SD over the subject's minutes."""
    have = minutes[np.isfinite(minutes["FFT_AE"])]
    if have.empty:
        return pd.DataFrame()
    return (
        have.groupby("Subject")
        .agg(
            N_Minutes=("FFT_AE", "size"),
            GT_RR_Median=("GT_RR", "median"),
            FFT_RR_Median=("FFT_RR", "median"),
            FFT_MAE=("FFT_AE", "mean"),
            FFT_MAE_SD=("FFT_AE", "std"),
            FFT_Bias=("FFT_Error", "mean"),
            FFT_Bias_SD=("FFT_Error", "std"),
            Wave_PCC_Mean=("Wave_PCC", "mean"),
            Wave_PCC_SD=("Wave_PCC", "std"),
        )
        .reset_index()
    )


def save_summary_figures(
    summary_root: Path,
    dataset_label: str,
    spec: WindowSpec,
    rows: Sequence[Tuple[str, pd.DataFrame]],
) -> None:
    """Per-subject FFT MAE of every model, and the full model's per-subject
    median rates against the reference (Davies & Mandic, Fig. 3b)."""
    figure_root = summary_root / "Figures"
    figure_root.mkdir(parents=True, exist_ok=True)
    stem = f"{dataset_label}_{spec.name}"
    labels, values = [], []
    for label, minutes in rows:
        errors = subject_errors(minutes)
        if errors.size:
            labels.append(label)
            values.append(errors.to_numpy(float))
    if values:
        fig, axis = plt.subplots(figsize=(max(6.0, 1.6 * len(values)), 4.5))
        axis.boxplot(values, showfliers=False)
        for position, data in enumerate(values, start=1):
            jitter = np.random.default_rng(position).uniform(-0.12, 0.12, len(data))
            axis.scatter(position + jitter, data, s=10, alpha=0.6, color="#4c78a8")
        axis.set_xticks(range(1, len(labels) + 1))
        axis.set_xticklabels(labels, rotation=20, ha="right")
        axis.set_ylabel("Per-subject FFT MAE (breaths/min)")
        axis.set_title(f"{dataset_label}, {spec.label}")
        fig.tight_layout()
        fig.savefig(figure_root / f"{stem}_subject_fft_mae.png", dpi=150)
        plt.close(fig)
    if rows:
        table = per_subject_table(rows[0][1])
        if not table.empty:
            fig, axis = plt.subplots(figsize=(4.8, 4.8))
            low = float(min(table["GT_RR_Median"].min(), table["FFT_RR_Median"].min()))
            high = float(max(table["GT_RR_Median"].max(), table["FFT_RR_Median"].max()))
            axis.plot([low, high], [low, high], color="#888888", linewidth=1)
            axis.scatter(table["GT_RR_Median"], table["FFT_RR_Median"], s=18, color="#4c78a8")
            axis.set_xlabel("Reference median rate (breaths/min)")
            axis.set_ylabel("Predicted median rate (breaths/min)")
            axis.set_title(f"{dataset_label}: {rows[0][0]}")
            fig.tight_layout()
            fig.savefig(figure_root / f"{stem}_subject_median_rate.png", dpi=150)
            plt.close(fig)


def write_results_excel(path: Path, sheets: Dict[str, pd.DataFrame]) -> Path:
    try:
        with pd.ExcelWriter(path, engine="openpyxl") as writer:
            for name, frame in sheets.items():
                frame.to_excel(writer, index=False, sheet_name=name[:31])
    except Exception as exc:
        warnings.warn(f"Excel export failed: {exc}")
    return path


# ---------------------------------------------------------------------------
# Main experiment runner
# ---------------------------------------------------------------------------

def write_quality_report(
    quality_root: Path,
    dataset_label: str,
    subject_data: Dict[str, dict],
) -> None:
    rows = []
    quality_root.mkdir(parents=True, exist_ok=True)
    all_window_rows = []
    for subject, record in subject_data.items():
        quality = pd.DataFrame(record["quality"])
        candidate = len(quality)
        kept = int(quality["Kept"].sum()) if not quality.empty else 0
        train_ok = int(quality["Train_OK"].sum()) if not quality.empty else 0
        row = {
            "Dataset": dataset_label,
            "Subject": subject,
            "Candidate_Windows": candidate,
            "Kept_Windows": kept,
            "Kept_Ratio": float(kept / candidate) if candidate else np.nan,
            "Train_OK_Windows": train_ok,
            "Train_OK_Ratio_Of_Kept": float(train_ok / kept) if kept else np.nan,
            "Reference_Breaths": int(
                sum(len(ref["breaths"]) for ref in record["references"].values())
            ),
            **record["reference_diagnostics"],
            "Window_Sec": record["window_sec"],
            "Stride_Sec": record["stride_sec"],
        }
        rows.append(row)
        if not quality.empty:
            quality.insert(0, "Subject", subject)
            all_window_rows.append(quality)
    pd.DataFrame(rows).to_csv(
        quality_root / f"{dataset_label}_subject_quality.csv",
        index=False,
        encoding="utf-8-sig",
    )
    if all_window_rows:
        pd.concat(all_window_rows, ignore_index=True).to_csv(
            quality_root / f"{dataset_label}_window_quality.csv",
            index=False,
            encoding="utf-8-sig",
        )


def _json_default(value):
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f"Not JSON serializable: {type(value)!r}")


def build_manifest(
    args: argparse.Namespace,
    results_root: Path,
    device: torch.device,
    variants: Sequence[Variant],
    window_specs: Sequence[WindowSpec],
    dataset_keys: Sequence[str],
) -> dict:
    return {
        "script": str(Path(__file__).resolve()),
        "results_root": str(results_root),
        "device": str(device),
        "model": {
            "path": (
                "raw PPG (min-max per window) -> 3 stem branches (kernels "
                f"{list(MODEL_CONFIG['stem_kernel_sizes'])} samples at {TARGET_FS} Hz, "
                f"{MODEL_CONFIG['stem_base_channels']} filters, GroupNorm, GELU) -> "
                "1x1 fusion + GroupNorm -> depthwise residual encoder (4 blocks, "
                "kernels and dilations per model) -> decoder (channels "
                f"{list(MODEL_CONFIG['decoder_mid_channels'])}, kernels and dilations "
                "per model) -> waveform head (tanh), no breath-event head"
            ),
            "identity_channel": False,
            "gating": False,
            "model_config": MODEL_CONFIG,
            "model_fs": TARGET_FS,
        },
        "variants": {
            variant.name: {
                "label": variant.label,
                "model_key": variant.options.key,
                "model_label": variant.options.label,
                "stem_kernel_sizes": list(variant.options.stem_kernel_sizes),
                "stem_channels": variant.options.stem_channels,
                **variant.options.receptive_fields(),
                "max_epochs": variant.epochs,
                "parameters": count_parameters(variant.options.build()),
                "macs_per_10s_window": int(model_layer_table(variant)["MACs_Per_Window"].sum()),
            }
            for variant in variants
        },
        "full_model_from": (
            "trained here"
            if "full" in args.variants
            else {
                "CapnoBase": str(args.reference_261007_root),
                "BIDMC / STEAM2": str(args.reference_261013_root),
            }
        ),
        "training_protocols": {
            DATASET_CONFIGS[key]["label"]: (
                "261007: temporal validation (last 20% of each training subject), "
                f"weight decay {WEIGHT_DECAY:g} on every parameter"
                if TRAINING_PROTOCOLS[key] == "261007"
                else "261013: subject-wise validation "
                f"({VAL_SUBJECT_RATIO:.0%} of the training subjects), weight decay "
                "selected by 261013 in this fold (convolution weights), read from "
                f"{args.reference_261013_root}"
            )
            for key in dataset_keys
        },
        "models_per_fold": {
            "models": 1,
            "ensemble": False,
            "seed": (
                "stable_seed('261006-ensemble-member', 'w10s_nonoverlap', dataset, subject, 0): "
                "the seed of ensemble member m0 in 261007 (same for every model)"
            ),
        },
        "training": {
            "min_epochs": args.min_epochs,
            "patience": args.patience,
            "epoch_cap": args.epoch_cap,
            "batch_size": args.batch_size,
            "learning_rate": LEARNING_RATE,
            "weight_decay": "per dataset (training_protocols)",
            "loss": "v16 hybrid waveform loss (SmoothL1, Pearson, derivative)",
            "loss_weights": list(LOSS_WEIGHTS),
            "validation": "per dataset (training_protocols), best checkpoint",
            "window_jitter": f"shift up to {SHIFT_AUG_SEC} s, gain 0.95-1.05, noise 0.002-0.01",
            "speed_augmentation": False,
            "rate_balanced_sampling": False,
        },
        "evaluation": {
            "unit": f"complete {EVAL_WINDOW_SEC:g} s minutes with a valid reference",
            "rate_estimate": "largest FFT peak of the predicted waveform",
            "waveform_agreement": "Pearson r of predicted and target waveform per minute",
            "ground_truth": "reference breaths in the minute (CapnoBase expert labels)",
            "fft_metrics": list(SUMMARY_METRICS),
            "davies_mandic_2022": "mAE = median minute AE, mMAE = median subject MAE",
            "subjects_excluded": "none",
            "smart_fusion": "not used",
        },
        "analysis_profiles": {
            DATASET_CONFIGS[key]["label"]: {
                "profile": DATASET_CONFIGS[key]["profile"],
                **ANALYSIS_PROFILES[DATASET_CONFIGS[key]["profile"]],
            }
            for key in dataset_keys
        },
        "data_paths": {
            DATASET_CONFIGS[key]["label"]: {
                field: DATASET_CONFIGS[key][field]
                for field in ("data_path", "ppg_dir", "rsp_dir")
                if field in DATASET_CONFIGS[key]
            }
            for key in dataset_keys
        },
        "window_conditions": [
            {"name": spec.name, "window_sec": spec.window_sec, "stride_sec": spec.stride_sec}
            for spec in window_specs
        ],
        "datasets": [DATASET_CONFIGS[key]["label"] for key in dataset_keys],
        "cross_validation": "Leave-One-Subject-Out; each subject is held out once",
    }


def condition_root(results_root: Path, variant: str, spec: WindowSpec, dataset_key: str) -> Path:
    return (
        results_root
        / variant
        / f"{spec.order:02d}_{spec.name}"
        / DATASET_CONFIGS[dataset_key]["label"]
    )


def member_seed(spec: WindowSpec, dataset_key: str, subject: str, member: int) -> int:
    return stable_seed("261006-ensemble-member", spec.name, dataset_key, subject, member)


def member_paths(root: Path, subject: str, member: int) -> Tuple[Path, Path, Path]:
    tag = safe_tag(subject)
    summaries = root / "Fold_Summaries"
    return (
        summaries / f"{tag}_m{member}_summary.json",
        summaries / f"{tag}_m{member}_RR60s_windows.csv",
        root / "Members" / f"m{member}" / "Predictions" / f"{tag}_test_outputs.npz",
    )


def read_json(path: Path) -> Optional[dict]:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


def expected_profile(dataset_label: str) -> Optional[str]:
    for config in DATASET_CONFIGS.values():
        if config["label"] == dataset_label:
            return config["profile"]
    return None


def epoch_limit(args: argparse.Namespace, variant: Variant) -> int:
    return variant.epochs if args.epoch_cap is None else min(variant.epochs, args.epoch_cap)


class SelectionMissing(RuntimeError):
    """No usable 261013 weight decay selection for a BIDMC / STEAM2 fold."""


def selection_path_261013(
    args: argparse.Namespace, spec: WindowSpec, dataset_key: str, subject: str
) -> Path:
    return (
        Path(args.reference_261013_root)
        / "WD_Selection"
        / f"{spec.order:02d}_{spec.name}"
        / DATASET_CONFIGS[dataset_key]["label"]
        / f"{safe_tag(subject)}_selection.json"
    )


def fold_training(
    args: argparse.Namespace,
    spec: WindowSpec,
    dataset_key: str,
    test_subject: str,
    train_subjects: Sequence[str],
) -> dict:
    """Validation and weight decay of one LOSOCV fold (the same for all three
    models). CapnoBase: 261007. BIDMC / STEAM2: the validation subjects of
    261013 and the weight decay 261013 selected for this fold; raises
    SelectionMissing when 261013 has no selection for the fold or its
    validation subjects differ (other data)."""
    if TRAINING_PROTOCOLS[dataset_key] == "261007":
        return {
            "Training_Protocol": "261007",
            "Weight_Decay": float(WEIGHT_DECAY),
            "Weight_Decay_Arg": None,
            "Weight_Decay_Scope": "all parameters",
            "Weight_Decay_Source": "261007 (fixed)",
            "Validation": "temporal (last 20% of each training subject)",
            "Val_Subjects": None,
            "Model_Suffix": "",
        }
    val_subjects = validation_subjects(dataset_key, test_subject, train_subjects)
    path = selection_path_261013(args, spec, dataset_key, test_subject)
    record = read_json(path)
    label = DATASET_CONFIGS[dataset_key]["label"]
    if record is None or "Selected_Weight_Decay" not in record:
        raise SelectionMissing(f"{label} {test_subject}: no 261013 weight decay selection ({path})")
    if record.get("Window_Config") != spec.name or sorted(
        str(subject) for subject in record.get("Validation_Subjects", [])
    ) != sorted(val_subjects):
        raise SelectionMissing(
            f"{label} {test_subject}: the 261013 selection ({path}) was made with other "
            "validation subjects or windows"
        )
    weight_decay = float(record["Selected_Weight_Decay"])
    return {
        "Training_Protocol": "261013",
        "Weight_Decay": weight_decay,
        "Weight_Decay_Arg": weight_decay,
        "Weight_Decay_Scope": "convolution weights",
        "Weight_Decay_Source": str(path),
        "Validation": f"subject-wise ({len(val_subjects)} validation subjects)",
        "Val_Subjects": val_subjects,
        # The 261013 key suffix, so "full" here has the key of the 261013
        # member of the selected weight decay.
        "Model_Suffix": "_wd{:g}_subjval".format(weight_decay),
    }


def fold_model_key(variant: Variant, training: dict) -> str:
    return variant.options.key + training["Model_Suffix"]


def member_done(
    root: Path,
    variant: Variant,
    subject: str,
    member: int,
    epochs: int,
    model_key: str,
    dataset_label: str,
) -> bool:
    """A member is reused only when it was trained with the same model and
    training settings (key), epochs, analysis profile (target handling) and
    test windows."""
    summary_path, window_path, prediction_path = member_paths(root, subject, member)
    summary = read_json(summary_path)
    return bool(
        summary is not None
        and summary.get("Model") == model_key
        and int(summary.get("Max_Epochs", -1)) == int(epochs)
        and summary.get("Analysis_Profile", LEGACY_PROFILES.get(dataset_label))
        == expected_profile(dataset_label)
        and summary.get("Inference", INFERENCE_PROTOCOL) == INFERENCE_PROTOCOL
        and window_path.exists()
        and prediction_path.exists()
    )


def reference_full_root(
    args: argparse.Namespace, spec: WindowSpec, dataset_key: str, training: dict
) -> Path:
    """Where the reference script stored "full" for this fold: 261007 (full)
    for CapnoBase, 261013 (full_wd{selected weight decay}) for BIDMC / STEAM2."""
    if training["Training_Protocol"] == "261007":
        return condition_root(Path(args.reference_261007_root), "full", spec, dataset_key)
    return condition_root(
        Path(args.reference_261013_root), "full_wd{:g}".format(training["Weight_Decay"]), spec, dataset_key
    )


def full_minute_path(
    args: argparse.Namespace,
    results_root: Path,
    spec: WindowSpec,
    dataset_key: str,
    subject: str,
    training: dict,
) -> Optional[Path]:
    """Per-minute table of the full model of one fold: trained here (with
    --variants full) or, otherwise, the single model m0 of 261007 (CapnoBase)
    or 261013 (BIDMC / STEAM2, the weight decay selected in the fold), the
    same model, seed and training. None when neither is complete."""
    variant = VARIANTS["full"]
    key = fold_model_key(variant, training)
    label = DATASET_CONFIGS[dataset_key]["label"]
    epochs = epoch_limit(args, variant)
    for root in (
        condition_root(results_root, variant.name, spec, dataset_key),
        reference_full_root(args, spec, dataset_key, training),
    ):
        if member_done(root, variant, subject, SINGLE_MEMBER, epochs, key, label):
            return member_paths(root, subject, SINGLE_MEMBER)[1]
    return None


def run_member(
    args: argparse.Namespace,
    results_root: Path,
    variant: Variant,
    spec: WindowSpec,
    dataset_key: str,
    subject_data: Dict[str, dict],
    subjects: Sequence[str],
    test_subject: str,
    member: int,
    device: torch.device,
    training: dict,
) -> Optional[dict]:
    """Train and score one member of one fold; returns its summary."""
    root = condition_root(results_root, variant.name, spec, dataset_key)
    summary_path, window_path, _ = member_paths(root, test_subject, member)
    epochs = epoch_limit(args, variant)
    dataset_label = DATASET_CONFIGS[dataset_key]["label"]
    model_key = fold_model_key(variant, training)
    if args.resume and member_done(root, variant, test_subject, member, epochs, model_key, dataset_label):
        return None
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    seed = member_seed(spec, dataset_key, test_subject, member)
    train_subjects = [subject for subject in subjects if subject != test_subject]
    print(
        f"[{time.strftime('%H:%M:%S')}] {dataset_label} {variant.label} {test_subject} "
        f"(wd {training['Weight_Decay']:g}): "
        f"training, up to {epochs} epochs",
        flush=True,
    )
    start_time = time.time()
    outputs, metadata, audit, prediction_path, diagnostics = train_one(
        dataset_key,
        spec,
        test_subject,
        subject_data,
        train_subjects,
        root / "Members" / f"m{member}",
        device,
        epochs,
        args.min_epochs,
        args.patience,
        args.batch_size,
        seed,
        variant.options,
        args.save_features,
        args.resume,
        val_subjects=training["Val_Subjects"],
        weight_decay=training["Weight_Decay_Arg"],
        model_key=model_key,
    )
    summary, window_frame = evaluate_subject_minutes(
        outputs,
        metadata,
        subject_data[test_subject]["references"],
        TARGET_FS,
    )
    del outputs
    summary.update(
        {
            "Dataset": dataset_label,
            "Subject": str(test_subject),
            "Variant": variant.name,
            "Model_Name": variant.label,
            "Member": int(member),
            "Seed": int(seed),
            "LOSO_Fold_Index": subjects.index(test_subject) + 1,
            "Window_Config": spec.name,
            "Model": model_key,
            "Model_Label": variant.options.label,
            "Analysis_Profile": DATASET_CONFIGS[dataset_key]["profile"],
            "Inference": INFERENCE_PROTOCOL,
            "Training_Protocol": training["Training_Protocol"],
            "Weight_Decay_Source": training["Weight_Decay_Source"],
            "Max_Epochs": int(epochs),
            "Best_Val_Loss": audit["Best_Val_Loss"],
            "Best_Epoch": audit["Best_Epoch"],
            "Convergence_Epoch_95pct": audit["Convergence_Epoch_95pct"],
            "History_Epochs": audit["History_Epochs"],
            "Generalization_Gap_At_Best": audit["Generalization_Gap_At_Best"],
            "Elapsed_Sec": time.time() - start_time,
            "Prediction_File": str(prediction_path),
        }
    )
    summary.update(diagnostics)
    summary.update(subject_data[test_subject]["reference_diagnostics"])
    if not window_frame.empty:
        window_frame.insert(0, "Member", int(member))
        window_frame.insert(0, "Variant", variant.name)
        window_frame.insert(0, "Subject", str(test_subject))
        window_frame.insert(0, "Dataset", dataset_label)
    window_frame.to_csv(window_path, index=False, encoding="utf-8-sig")
    # Written last: its presence marks the member as complete.
    summary_path.write_text(
        json.dumps(summary, indent=2, default=_json_default),
        encoding="utf-8",
    )
    return summary


def read_minutes(paths: Sequence[Path]) -> pd.DataFrame:
    frames = []
    for path in paths:
        if path.exists() and path.stat().st_size > 0:
            frame = pd.read_csv(path, dtype={"Subject": str}, encoding="utf-8-sig")
            if not frame.empty:
                frames.append(frame)
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def mean_sd_text(mean: float, spread: float, digits: int = 3) -> str:
    if not np.isfinite(mean):
        return "n/a"
    return f"{mean:.{digits}f} ± {spread:.{digits}f}" if np.isfinite(spread) else f"{mean:.{digits}f}"


def mean_sd_table(overall: pd.DataFrame) -> pd.DataFrame:
    """The final performance as "mean ± SD" (medians with IQR for mAE / mMAE)."""
    rows = []
    for _, r in overall.iterrows():
        rows.append(
            {
                "Dataset": r["Dataset"],
                "Window_Config": r["Window_Config"],
                "Model": r["Model"],
                "Training": r["Training_Protocol"],
                "Parameters": int(r["Parameters"]),
                "Model RF (s)": f"{r['Model_RF_Sec']:.2f}",
                "N_Subjects": int(r["N_Subjects"]),
                "N_Minutes": int(r["N_Minutes"]),
                "MAE (mean ± SD over minutes)": mean_sd_text(r["FFT_MAE"], r["FFT_MAE_SD"]),
                "Subject MAE (mean ± SD over subjects)": mean_sd_text(
                    r["FFT_Subject_MAE_Mean"], r["FFT_Subject_MAE_SD"]
                ),
                "Bias (mean ± SD over minutes)": mean_sd_text(r["FFT_Bias"], r["FFT_Bias_SD"]),
                "LoA": f"{r['FFT_LoA_Lower']:.2f} to {r['FFT_LoA_Upper']:.2f}",
                "RMSE": f"{r['FFT_RMSE']:.3f}",
                "Pearson r": f"{r['FFT_Pearson_r']:.3f}",
                "Within 1 / 2 bpm (%)": f"{r['FFT_Within_1_BPM_Pct']:.1f} / {r['FFT_Within_2_BPM_Pct']:.1f}",
                "mAE [IQR]": f"{r['Davies_mAE']:.3f} [{r['Davies_mAE_Q1']:.3f}-{r['Davies_mAE_Q3']:.3f}]",
                "mMAE [IQR]": f"{r['Davies_mMAE']:.3f} [{r['Davies_mMAE_Q1']:.3f}-{r['Davies_mMAE_Q3']:.3f}]",
                "Wave PCC (mean ± SD over minutes)": mean_sd_text(r["Wave_PCC_Mean"], r["Wave_PCC_SD"]),
            }
        )
    return pd.DataFrame(rows)


def summarize_results(
    args: argparse.Namespace,
    results_root: Path,
    variants: Sequence[Variant],
    window_specs: Sequence[WindowSpec],
    dataset_keys: Sequence[str],
) -> None:
    """Score the single model of every fold and write the FFT / Davies & Mandic
    summary and the comparison of the models."""
    summary_root = results_root / "Summary"
    summary_root.mkdir(parents=True, exist_ok=True)
    structure = model_variant_table(variants)
    layers = pd.concat([model_layer_table(variant) for variant in variants], ignore_index=True)
    structure_by_variant = structure.set_index("Folder")
    overall_rows, subject_tables, minute_frames, missing_rows = [], [], [], []
    training_frames = []
    for spec in window_specs:
        for dataset_key in dataset_keys:
            dataset_label = DATASET_CONFIGS[dataset_key]["label"]
            try:
                subject_data = load_dataset(dataset_key, spec)
            except Exception as exc:
                if is_fatal_error(exc):
                    raise
                print(f"{dataset_label}: not summarized ({exc})")
                continue
            all_subjects = sorted(subject_data)
            subjects = selected_subjects(args, subject_data)
            del subject_data
            release_memory()
            trainings, no_selection = fold_trainings(args, spec, dataset_key, all_subjects, subjects)
            training_frames.append(training_table(dataset_label, spec, trainings))
            figure_rows: List[Tuple[str, pd.DataFrame]] = []
            for variant in variants:
                root = condition_root(results_root, variant.name, spec, dataset_key)
                epochs = epoch_limit(args, variant)
                done_paths = []
                for subject in subjects:
                    training = trainings.get(subject)
                    path = None
                    if training is not None and variant.name == "full":
                        path = full_minute_path(args, results_root, spec, dataset_key, subject, training)
                    elif training is not None and member_done(
                        root,
                        variant,
                        subject,
                        SINGLE_MEMBER,
                        epochs,
                        fold_model_key(variant, training),
                        dataset_label,
                    ):
                        path = member_paths(root, subject, SINGLE_MEMBER)[1]
                    if path is not None:
                        done_paths.append(path)
                    else:
                        missing_rows.append(
                            {
                                "Dataset": dataset_label,
                                "Window_Config": spec.name,
                                "Model": variant.label,
                                "Subject": subject,
                                "Reason": no_selection.get(
                                    subject,
                                    "not found in 261007 / 261013" if variant.name == "full" else "not trained yet",
                                ),
                            }
                        )
                minutes = scored_minutes(read_minutes(done_paths))
                if minutes.empty:
                    continue
                # Named by the kept branches (the full model read from 261013
                # carries its run name, full_wd...).
                minutes["Variant"] = variant.label
                minutes["Window_Config"] = spec.name
                minute_frames.append(minutes)
                label = variant.label
                info = structure_by_variant.loc[variant.name]
                row = {
                    "Dataset": dataset_label,
                    "Window_Config": spec.name,
                    "Model": label,
                    "Folder": variant.name,
                    "Training_Protocol": TRAINING_PROTOCOLS[dataset_key],
                    "Stem_Branches": info["Stem_Branches"],
                    "Stem_Channels": int(info["Stem_Channels"]),
                    "Model_RF_Sec": float(info["Model_RF_Sec"]),
                    "Parameters": int(info["Parameters"]),
                    "MACs_Per_10s_Window": int(info["MACs_Per_10s_Window"]),
                    "Subjects_Complete": int(len(done_paths)),
                }
                row.update(fft_davies_metrics(minutes))
                overall_rows.append(row)
                table = per_subject_table(minutes)
                if not table.empty:
                    table.insert(0, "Model", label)
                    table.insert(0, "Window_Config", spec.name)
                    table.insert(0, "Dataset", dataset_label)
                    subject_tables.append(table)
                figure_rows.append((label, minutes))
            if figure_rows:
                save_summary_figures(summary_root, dataset_label, spec, figure_rows)

    overall = pd.DataFrame(overall_rows)
    mean_sd = mean_sd_table(overall) if not overall.empty else pd.DataFrame()
    per_subject = pd.concat(subject_tables, ignore_index=True) if subject_tables else pd.DataFrame()
    minutes_all = pd.concat(minute_frames, ignore_index=True) if minute_frames else pd.DataFrame()
    missing = pd.DataFrame(missing_rows)
    comparison = compare_models(minutes_all)
    training_by_fold = (
        pd.concat(training_frames, ignore_index=True) if training_frames else pd.DataFrame()
    )
    tables = {
        "fft_davies_overall": overall,
        "fft_davies_overall_mean_sd": mean_sd,
        "model_comparison": comparison,
        "model_variants": structure,
        "model_layers": layers,
        "training_by_fold": training_by_fold,
        "per_subject_fft": per_subject,
        "minutes": minutes_all,
        "incomplete_folds": missing,
    }
    for name, frame in tables.items():
        frame.to_csv(summary_root / f"{name}.csv", index=False, encoding="utf-8-sig")
    excel_path = write_results_excel(
        summary_root / "261016_branchwise_ablation_results.xlsx",
        {
            "Overall_mean_SD": mean_sd,
            "Overall_FFT_Davies": overall,
            "Model_comparison": comparison,
            "Model_variants": structure,
            "Model_layers": layers,
            "Training_by_fold": training_by_fold,
            "Per_subject": per_subject,
            "Minutes": minutes_all,
            "Incomplete_folds": missing,
        },
    )
    if not training_by_fold.empty:
        print("\nTraining settings (validation; weight decay per fold):")
        for dataset, frame in training_by_fold.groupby("Dataset", sort=False):
            counts = frame["Weight_Decay"].value_counts().sort_index()
            print(
                f"  {dataset:<10s} {frame['Training_Protocol'].iloc[0]}: {frame['Validation'].iloc[0]}; "
                f"weight decay ({frame['Weight_Decay_Scope'].iloc[0]}) "
                + ", ".join(f"{value:g} in {count} folds" for value, count in counts.items())
            )
    if overall.empty:
        print("\nNo complete fold to summarize yet.")
        return
    print("\nModels (receptive field in s; MACs per 10 s window):")
    for _, row in structure.iterrows():
        print(
            f"  {row['Model']:<20s} ({row['Stem_Channels']:>2d} stem channels)  "
            f"RF {row['Model_RF_Sec']:.2f} s  params {row['Parameters']:,}  "
            f"MACs {row['MACs_Per_10s_Window'] / 1e6:.2f} M"
        )
    print(
        "\nFFT and Davies & Mandic metrics, single model per fold, 10 s windows "
        "(breaths/min; all reference-valid minutes; mean ± SD, medians with IQR; "
        "PCC = waveform Pearson r):"
    )
    for _, row in mean_sd.iterrows():
        print(
            f"  {row['Dataset']:<10s} {row['Model']:<20s} subj {row['N_Subjects']:>3d} "
            f"min {row['N_Minutes']:>4d}  MAE {row['MAE (mean ± SD over minutes)']:<15s} "
            f"subject MAE {row['Subject MAE (mean ± SD over subjects)']:<15s} "
            f"r {row['Pearson r']}  mAE {row['mAE [IQR]']}  mMAE {row['mMAE [IQR]']}  "
            f"PCC {row['Wave PCC (mean ± SD over minutes)']}"
        )
    if not comparison.empty:
        print("\nModel pairs (subjects scored by both; per-subject MAE, Wilcoxon, BH-FDR):")
        for _, row in comparison.iterrows():
            print(
                f"  {row['Dataset']:<10s} {row['Reference_Model']} vs {row['Compared_Model']}: "
                f"subject MAE {mean_sd_text(row['Reference_FFT_Subject_MAE_Mean'], row['Reference_FFT_Subject_MAE_SD'])}"
                f" vs {mean_sd_text(row['Compared_FFT_Subject_MAE_Mean'], row['Compared_FFT_Subject_MAE_SD'])}, "
                f"diff {mean_sd_text(row['Subject_MAE_Diff_Compared_minus_Reference'], row['Subject_MAE_Diff_SD'])}; "
                f"compared better in {row['N_Subjects_Compared_Better']}/{row['N_Subjects_Paired']}, "
                f"p={row['Subject_Wilcoxon_p']:.3g} (FDR {row['Subject_Wilcoxon_FDR_p']:.3g}); "
                f"subject PCC {mean_sd_text(row['Subject_PCC_Reference'], row['Subject_PCC_Reference_SD'])} vs "
                f"{mean_sd_text(row['Subject_PCC_Compared'], row['Subject_PCC_Compared_SD'])}, "
                f"p={row['Subject_PCC_Wilcoxon_p']:.3g} (FDR {row['Subject_PCC_Wilcoxon_FDR_p']:.3g})"
            )
    if not missing.empty:
        print(f"\n{len(missing)} fold(s) incomplete; see {summary_root / 'incomplete_folds.csv'}")
    print(f"\nSaved the summary under: {summary_root}")
    print(f"Excel workbook: {excel_path}")


def summary_variants(variants: Sequence[Variant]) -> List[Variant]:
    """The full model first (read from 261007 / 261013 when not trained here),
    then the trained models."""
    return [VARIANTS["full"]] + [variant for variant in variants if variant.name != "full"]


def fold_trainings(
    args: argparse.Namespace,
    spec: WindowSpec,
    dataset_key: str,
    all_subjects: Sequence[str],
    subjects: Sequence[str],
) -> Tuple[Dict[str, dict], Dict[str, str]]:
    """Training settings of every fold to run, and the folds without a usable
    261013 selection (subject -> reason)."""
    trainings, missing = {}, {}
    for subject in subjects:
        train_subjects = [other for other in all_subjects if other != subject]
        try:
            trainings[subject] = fold_training(args, spec, dataset_key, subject, train_subjects)
        except SelectionMissing as exc:
            missing[subject] = str(exc)
    return trainings, missing


def training_table(dataset_label: str, spec: WindowSpec, trainings: Dict[str, dict]) -> pd.DataFrame:
    """Validation and weight decay of every fold."""
    return pd.DataFrame(
        [
            {
                "Dataset": dataset_label,
                "Window_Config": spec.name,
                "Subject": subject,
                "Training_Protocol": training["Training_Protocol"],
                "Validation": training["Validation"],
                "Validation_Subjects": " ".join(training["Val_Subjects"] or []),
                "Weight_Decay": training["Weight_Decay"],
                "Weight_Decay_Scope": training["Weight_Decay_Scope"],
                "Weight_Decay_Source": training["Weight_Decay_Source"],
            }
            for subject, training in trainings.items()
        ]
    )


def task_progress(results_root: Path, spec: WindowSpec, dataset_key: str, variant: Variant) -> int:
    """Number of trained runs stored for one dataset and model."""
    folder = condition_root(results_root, variant.name, spec, dataset_key) / "Fold_Summaries"
    return len(list(folder.glob("*_summary.json")))


def run_worker(command: Sequence[str]) -> int:
    """Run a worker process and print its output here line by line, so it is
    seen in any console (IDE consoles such as Spyder do not show the output a
    child process writes itself). Stopping the main process stops the worker."""
    env = dict(os.environ, PYTHONUNBUFFERED="1", PYTHONIOENCODING="utf-8")
    process = subprocess.Popen(
        list(command),
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        encoding="utf-8",
        errors="replace",
        bufsize=1,
        env=env,
    )
    try:
        for line in process.stdout:
            print(line, end="", flush=True)
        return process.wait()
    except BaseException:
        process.kill()
        process.wait()
        raise


def run_tasks_in_subprocesses(
    args: argparse.Namespace,
    results_root: Path,
    window_specs: Sequence[WindowSpec],
    dataset_keys: Sequence[str],
    variants: Sequence[Variant],
) -> List[str]:
    """Train every dataset and model in its own Python process (at most one
    model per subject of one dataset, i.e. 42-53 runs).

    The process ends when its dataset and model are done, so all its host and
    GPU memory is returned before the next one. A process that stops on a
    fatal error (out of memory, CUDA / cuDNN failure) is started again with
    --resume, which keeps the stored runs and continues an interrupted run
    from its last epoch; after two attempts without a new stored run the task
    is left and reported. Returns the tasks left incomplete."""
    script = str(Path(__file__).resolve())
    incomplete = []
    for spec in window_specs:
        for dataset_key in dataset_keys:
            label = DATASET_CONFIGS[dataset_key]["label"]
            for variant in variants:
                attempt, stalled = 0, 0
                while True:
                    before = task_progress(results_root, spec, dataset_key, variant)
                    command = [sys.executable, "-u", script, *sys.argv[1:], "--datasets", dataset_key,
                               "--variants", variant.name, "--child-task"]
                    if attempt > 0 and "--resume" not in command:
                        command.append("--resume")
                    print(f"\n=== {label}, {variant.label}: worker process (attempt {attempt + 1}) ===",
                          flush=True)
                    code = run_worker(command)
                    if code == 0:
                        break
                    attempt += 1
                    progress = task_progress(results_root, spec, dataset_key, variant)
                    stalled = stalled + 1 if progress <= before else 0
                    if stalled >= 2:
                        print(f"{label}, {variant.label}: worker stopped (exit code {code}) twice without a "
                              "new run; left incomplete (run again with --resume)", flush=True)
                        incomplete.append(f"{label} / {variant.label}")
                        break
                    print(f"{label}, {variant.label}: worker stopped (exit code {code}); restarting with "
                          "--resume", flush=True)
    return incomplete


def selected_subjects(args: argparse.Namespace, subject_data: Dict[str, dict]) -> List[str]:
    subjects = sorted(subject_data)
    if args.subjects:
        wanted = {str(subject) for subject in args.subjects}
        subjects = [subject for subject in subjects if subject in wanted]
    return subjects


def parse_shard(value: Optional[str]) -> Tuple[int, int]:
    if not value:
        return 0, 1
    index, count = (int(part) for part in value.split("/"))
    if count < 1 or not 0 <= index < count:
        raise ValueError("--shard must be K/N with 0 <= K < N")
    return index, count


def run_experiment(args: argparse.Namespace) -> None:
    results_root = Path(args.results_root).resolve()
    results_root.mkdir(parents=True, exist_ok=True)
    DATASET_CONFIGS["capnobase"]["use_artifact_labels"] = args.capnobase_artifacts
    if args.capnobase_path:
        DATASET_CONFIGS["capnobase"]["data_path"] = str(args.capnobase_path)
    if args.bidmc_path:
        DATASET_CONFIGS["bidmc"]["data_path"] = str(args.bidmc_path)
    if args.steam2_ppg_dir:
        DATASET_CONFIGS["steam2"]["ppg_dir"] = str(args.steam2_ppg_dir)
    if args.steam2_rsp_dir:
        DATASET_CONFIGS["steam2"]["rsp_dir"] = str(args.steam2_rsp_dir)
    if args.inspect_capnobase:
        inspect_capnobase(results_root)
        return
    if args.inspect_steam2:
        inspect_steam2(results_root)
        return
    if args.threads:
        torch.set_num_threads(int(args.threads))
    if args.require_cuda and not torch.cuda.is_available() and not args.summary_only:
        raise RuntimeError(
            "CUDA is required by default. Use --no-require-cuda to run on the CPU."
        )
    device = torch.device(
        f"cuda:{args.gpu}" if torch.cuda.is_available() and not args.force_cpu else "cpu"
    )
    variants = [VARIANTS[name] for name in VARIANTS if name in args.variants]
    window_specs = list(WINDOW_SPECS)
    dataset_keys = [key for key in DATASET_ORDER if key in args.datasets]
    shard_index, shard_count = parse_shard(args.shard)
    manifest = build_manifest(args, results_root, device, variants, window_specs, dataset_keys)
    manifest_name = (
        "run_manifest.json" if shard_count == 1 else f"run_manifest_shard{shard_index}.json"
    )
    if args.child_task:
        manifest_name = manifest_name.replace(".json", f"_{dataset_keys[0]}_{variants[0].name}.json")
    (results_root / manifest_name).write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False, default=_json_default),
        encoding="utf-8",
    )
    for variant in variants:
        fields = variant.options.receptive_fields()
        print(
            f"{variant.label}: {variant.options.label}; receptive field "
            f"{fields['Model_RF_Sec']:.2f} s; up to {variant.epochs} epochs; "
            f"{count_parameters(variant.options.build())} parameters"
        )
    print(
        "Single model per LOSOCV fold (no ensemble); windows: "
        + ", ".join(spec.label for spec in window_specs)
        + f"; device={device}"
    )

    if args.dry_run:
        for spec in window_specs:
            for dataset_key in dataset_keys:
                label = DATASET_CONFIGS[dataset_key]["label"]
                try:
                    data = load_dataset(dataset_key, spec)
                except Exception as exc:
                    print(f"{label}: could not be loaded ({exc})")
                    continue
                print(
                    f"{label} ({spec.label}): "
                    f"{len(data)} valid subjects = {len(data)} LOSOCV folds"
                )
                del data
                release_memory()
        return

    if not args.summary_only and not args.child_task and not args.single_process:
        # Every dataset and model in its own process (memory).
        incomplete = run_tasks_in_subprocesses(args, results_root, window_specs, dataset_keys, variants)
        if incomplete:
            print("\nIncomplete tasks: " + "; ".join(incomplete))
        if shard_count > 1:
            print(
                "\nWorker finished. When every worker is done, run with "
                "--resume --summary-only to build the summary."
            )
            return
        summarize_results(args, results_root, summary_variants(variants), window_specs, dataset_keys)
        return

    if not args.summary_only:
        task = (
            f"_{DATASET_CONFIGS[dataset_keys[0]]['label']}_{variants[0].name}"
            if args.child_task and len(dataset_keys) == 1 and len(variants) == 1
            else ""
        )
        failed_path = results_root / (
            f"failed_members{task}.csv" if shard_count == 1 else f"failed_members{task}_shard{shard_index}.csv"
        )
        failed_rows = []
        for spec in window_specs:
            for dataset_key in dataset_keys:
                dataset_label = DATASET_CONFIGS[dataset_key]["label"]
                print(f"[{time.strftime('%H:%M:%S')}] {dataset_label} ({spec.label}): reading the data ...", flush=True)
                try:
                    subject_data = load_dataset(dataset_key, spec)
                except Exception as exc:
                    if is_fatal_error(exc):
                        failed_rows.append({"Dataset": dataset_label, "Subject": "", "Error": repr(exc)})
                        pd.DataFrame(failed_rows).to_csv(failed_path, index=False, encoding="utf-8-sig")
                        print(f"{dataset_label}: loading stopped by a fatal error ({exc}); "
                              "the worker stops (run again with --resume)", flush=True)
                        raise SystemExit(3)
                    print(f"{dataset_label}: could not be loaded, skipped ({exc})")
                    failed_rows.append(
                        {"Dataset": dataset_label, "Subject": "", "Error": repr(exc)}
                    )
                    continue
                all_subjects = sorted(subject_data)
                if len(all_subjects) < 3:
                    raise RuntimeError(f"{dataset_label}: fewer than three valid subjects")
                if shard_index == 0:
                    write_quality_report(
                        results_root / "Data_Quality" / f"{spec.order:02d}_{spec.name}",
                        dataset_label,
                        subject_data,
                    )
                subjects = selected_subjects(args, subject_data)
                trainings, no_selection = fold_trainings(
                    args, spec, dataset_key, all_subjects, subjects
                )
                for subject, reason in no_selection.items():
                    print(f"{reason}; fold skipped")
                    failed_rows.append(
                        {
                            "Dataset": dataset_label,
                            "Window_Config": spec.name,
                            "Model": "",
                            "Subject": subject,
                            "Member": SINGLE_MEMBER,
                            "Error": reason,
                        }
                    )
                if trainings and TRAINING_PROTOCOLS[dataset_key] == "261013":
                    counts = pd.Series([t["Weight_Decay"] for t in trainings.values()]).value_counts()
                    print(
                        f"{dataset_label}: 261013 settings, subject-wise validation, weight decay "
                        "selected by 261013 per fold ("
                        + ", ".join(f"{value:g}: {count}" for value, count in counts.sort_index().items())
                        + ")"
                    )
                jobs = [
                    (variant, subject, SINGLE_MEMBER)
                    for variant in variants
                    for subject in subjects
                    if subject in trainings
                ]
                if args.resume:
                    # Only the folds still to train are shared out, so the
                    # workers stay balanced.
                    jobs = [
                        (variant, subject, member)
                        for variant, subject, member in jobs
                        if not member_done(
                            condition_root(results_root, variant.name, spec, dataset_key),
                            variant,
                            subject,
                            member,
                            epoch_limit(args, variant),
                            fold_model_key(variant, trainings[subject]),
                            dataset_label,
                        )
                    ]
                mine = [job for index, job in enumerate(jobs) if index % shard_count == shard_index]
                print(
                    f"\n{dataset_label} ({spec.label}): {len(all_subjects)} LOSOCV folds, "
                    f"{len(jobs)} folds to train ({len(mine)} in this worker)"
                )
                for variant, subject, member in mine:
                    try:
                        summary = run_member(
                            args,
                            results_root,
                            variant,
                            spec,
                            dataset_key,
                            subject_data,
                            all_subjects,
                            subject,
                            member,
                            device,
                            trainings[subject],
                        )
                        if summary is not None:
                            print(
                                f"[{time.strftime('%H:%M:%S')}] {dataset_label} {variant.label} "
                                f"{subject} (wd {summary['Weight_Decay']:g}): "
                                f"FFT MAE={summary['RR_MAE_Wave_FFT']:.3f} "
                                f"(valid min {summary['N60_RefValid']}), best epoch "
                                f"{summary['Best_Epoch']}/{summary['History_Epochs']}, "
                                f"train/val windows {summary['Train_Windows']}/"
                                f"{summary['Val_Windows']}, "
                                f"{summary['Elapsed_Sec'] / 60.0:.1f} min"
                            )
                    except Exception as exc:
                        failed_rows.append(
                            {
                                "Dataset": dataset_label,
                                "Window_Config": spec.name,
                                "Model": variant.label,
                                "Subject": subject,
                                "Member": member,
                                "Error": repr(exc),
                            }
                        )
                        print(f"{dataset_label} {variant.label} {subject}: FAILED {exc}")
                        if is_fatal_error(exc):
                            # The CUDA context and the host memory are not
                            # reliable any more: stop instead of failing every
                            # following run; the run is continued with --resume.
                            pd.DataFrame(failed_rows).to_csv(failed_path, index=False, encoding="utf-8-sig")
                            print("Fatal error (memory / CUDA): the worker stops; it continues with --resume.",
                                  flush=True)
                            raise SystemExit(3)
                    finally:
                        release_memory(device)
                del subject_data
                release_memory(device)
        pd.DataFrame(failed_rows).to_csv(failed_path, index=False, encoding="utf-8-sig")
        if args.child_task:
            return
        if shard_count > 1:
            print(
                "\nWorker finished. When every worker is done, run with "
                "--resume --summary-only to build the summary."
            )
            return
    summarize_results(args, results_root, summary_variants(variants), window_specs, dataset_keys)


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--results-root", type=Path, default=DEFAULT_RESULTS_ROOT)
    parser.add_argument(
        "--datasets",
        nargs="+",
        choices=DATASET_ORDER,
        default=list(DEFAULT_DATASETS),
    )
    parser.add_argument(
        "--variants",
        nargs="+",
        choices=list(VARIANTS),
        default=list(ABLATION_VARIANTS),
        help=(
            "Models to train (default: the six branch combinations; the full "
            "model is read from 261007 / 261013 unless 'full' is given)."
        ),
    )
    parser.add_argument(
        "--reference-261007-root",
        type=Path,
        default=REFERENCE_261007_ROOT,
        help="Results root of 261007 (CapnoBase full model, 10 s member m0).",
    )
    parser.add_argument(
        "--reference-261013-root",
        type=Path,
        default=REFERENCE_261013_ROOT,
        help=(
            "Results root of 261013: the weight decay selected in every BIDMC / "
            "STEAM2 fold (WD_Selection) and its full model (member m0 of that weight decay)."
        ),
    )
    parser.add_argument("--min-epochs", type=int, default=MIN_EPOCHS)
    parser.add_argument("--patience", type=int, default=PATIENCE)
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    parser.add_argument(
        "--epoch-cap",
        type=int,
        default=None,
        help="Upper limit on the epochs (quick tests only).",
    )
    parser.add_argument(
        "--subjects",
        nargs="+",
        default=None,
        help="Only these held-out subjects (quick tests); training still uses all others.",
    )
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--force-cpu", action="store_true")
    parser.add_argument(
        "--threads",
        type=int,
        default=None,
        help="torch CPU threads per process (e.g. 1 with several --shard workers).",
    )
    parser.add_argument(
        "--no-require-cuda",
        dest="require_cuda",
        action="store_false",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help=(
            "Reuse folds whose summary, minutes and predictions already exist "
            "and continue interrupted folds from their last epoch."
        ),
    )
    parser.add_argument(
        "--shard",
        default=None,
        help="K/N: run every N-th fold starting at K (parallel workers).",
    )
    parser.add_argument(
        "--summary-only",
        action="store_true",
        help="Do not train; score the stored folds and write the summary.",
    )
    parser.add_argument(
        "--single-process",
        action="store_true",
        help=(
            "Train every dataset and model in this process (no worker processes; "
            "a fatal error then stops the run, continue with --resume)."
        ),
    )
    parser.add_argument("--child-task", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--capnobase-path", type=Path, default=None)
    parser.add_argument("--bidmc-path", type=Path, default=None)
    parser.add_argument("--steam2-ppg-dir", type=Path, default=None)
    parser.add_argument("--steam2-rsp-dir", type=Path, default=None)
    parser.add_argument(
        "--inspect-capnobase",
        action="store_true",
        help=(
            "Only read every CapnoBase case, print how it was loaded and save "
            "capnobase_loading_check.csv (no training, no GPU needed)."
        ),
    )
    parser.add_argument(
        "--inspect-steam2",
        action="store_true",
        help=(
            "Only read every STEAM2 session (261013 reader), print how it was "
            "loaded and save steam2_loading_check.csv (no training)."
        ),
    )
    parser.add_argument(
        "--ignore-capnobase-artifacts",
        dest="capnobase_artifacts",
        action="store_false",
        help="Do not use the expert artifact labels of CapnoBase.",
    )
    parser.add_argument(
        "--save-features",
        action="store_true",
        help="Also store the post-fusion features in each prediction file.",
    )
    parser.set_defaults(
        capnobase_artifacts=DATASET_CONFIGS["capnobase"]["use_artifact_labels"],
        require_cuda=True,
    )
    return parser.parse_args(argv)


if __name__ == "__main__":
    run_experiment(parse_args())
