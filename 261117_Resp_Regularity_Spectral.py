# -*- coding: utf-8 -*-
r"""261117_Resp_Regularity_Spectral: regularity of the respiration reference waveform of every recording,
classified from the concentration of its spectrum with the criteria of published studies, in 2 and 3 classes.

Reference waveform (raw recording as stored, read with the readers of the script of the final results,
loaded as a module; its data paths are used, the path options override):
  CapnoBase   capnogram (co2)               261007_RRWaveNet_Ensemble_NoEvent.py
  BIDMC       impedance (ref.resp_sig.imp)  261013_RRWaveNet_Adult_SubjectVal_WD.py
  STEAM2      belt (RSP channel), sessions split at the EEGLAB boundary events like the source script
The subjects with an MAE issue are left out as in the final results (CapnoBase 0115, BIDMC 27, STEAM2
12_YJJ, 22_LMS, 26_CKK; --keep-issue-subjects keeps them). CapnoBase windows that overlap an expert CO2
artifact label are left out (--keep-artifacts keeps them). A window needs >= 90% valid samples.

Spectral concentration ("peakedness") = the share of spectral power in a narrow interval around the
spectral peak. Three published criteria, each with its own window, spectrum and threshold:

A  Charlton et al. 2016 (RRest, spectral peak-conditioned averaging "SPA", modified from Lazaro Plaza 2015;
   parameters of the RRest v3.0 code: eta = 0.65, delta = 0.1 Hz)                         -> 2 and 3 classes
     preprocessing  (RRest reference) whole recording detrended, Tukey taper (2 s), low-pass -3 dB 1 Hz
                    (zero-phase Butterworth here), resampled to 5 Hz
     windows        32 s, consecutive (RRest winLeng 32, winStep 0); each window detrended
     spectrum       Welch periodogram, Hamming segments of 2^nextpow2(12 x 5 Hz) = 64 samples, 50% overlap,
                    NFFT 256 (as pwelch in RRest WCH)
     peak f_p       frequency of the largest power of the spectrum
     P(d)           power in [f_p - 0.4 d, f_p + 0.4 d] (limited to the RR range 4-60 /min) / power in
                    [f_p - d, f_p + d]
     2 classes      Regular: P(0.1 Hz) > 0.65;  Irregular: otherwise
     3 classes      Regular: P(0.1 Hz) > 0.65;  Intermediate: P(0.1 Hz) <= 0.65 and P(0.2 Hz) > 0.65 (RRest
                    repeats the test with twice delta when no spectrum passes);  Irregular: neither
B  Iozzia et al. 2019 (method of Lazaro et al., Laguna and Gil group)                       -> 2 classes
     windows        30 s every 10 s, signal at 4 Hz (same preprocessing as A, resampled to 4 Hz)
     spectrum       Welch, Hamming 12 s segments, 50% overlap, NFFT 256 (segments are not given in the paper)
     peak f_p       largest local maximum in 0.15-0.5 Hz; if there is none, in 0.08-0.5 Hz (as in the paper)
     P              100 x power in [f_p - 0.05, f_p + 0.05] Hz / power in [0.15, 0.5] Hz
     2 classes      Peaked: P >= 40 (xi = 40%);  Not peaked: otherwise
C  Sandberg et al. 2019                                                                     -> 2 classes
     preprocessing  low-pass 6th-order Butterworth 2 Hz, resampled to 4 Hz
     windows        40 s every 5 s; spectrum = average of the power spectra of 50% overlapping 12 s
                    sub-intervals (Welch, Hamming), power normalized in 0-1 Hz
     peak f_p       largest local maximum in 0.15-0.5 Hz; it must exceed 85% of the largest peak of the
                    spectrum (the paper searches around a reference rate; no reference rate is used here)
     2 classes      Sufficient: energy in [f_p - 0.048, f_p + 0.048] Hz > 30% of the energy in 0-1 Hz;
                    Insufficient: otherwise
Criteria B and C search 0.15-0.5 Hz (9-30 /min): faster or slower breathing (CapnoBase children, slow
ventilation) is judged at the wrong frequency, so A (4-60 /min) is the main classification.

Spectral entropy (reported, not used for the classes: no published fixed cut-off was found): normalized
Shannon entropy of the criterion A spectrum within the RR range 4-60 /min, H = -sum(q log q) / log(n);
0 = all power in one bin, 1 = flat spectrum.

Subject level: share of windows in each class; subject class = the class with the most windows (ties go
to the less regular class); 2 classes: Regular when at least half of the windows are Regular.

Outputs (D:\PPG2RSP_RRWaveNet_Inspired\261117_Resp_Regularity_Spectral)
  261117_Resp_Regularity_Spectral.xlsx
    Windows_A   every 32 s window: peak, P(0.1), P(0.2), class (2 and 3), RR peak (/min), spectral entropy
    Windows_B   every 30 s window: peak, P (%), class
    Windows_C   every 40 s window: peak, energy share (%), class
    Subjects    per subject: class shares and subject classes of A, B, C; median P, entropy, RR peak;
                CapnoBase age and ventilation mode as written in the meta file (treatment_ventilation) and in
                the param file (case_ventilation); the two differ for 0016, 0018, 0127, 0142 and 0322
    Datasets    per dataset: windows and subjects per class of each criterion, median entropy per class
    Criteria    the criteria, parameters and references

Run: python 261117_Resp_Regularity_Spectral.py     (Spyder: runfile(...), no options needed)
Options: --datasets, --subjects CapnoBase:0104 BIDMC:09 ..., --keep-issue-subjects, --keep-artifacts,
--output-root, --script-261007, --script-261013, --root-261007, --root-261013, --capnobase-path,
--bidmc-path, --steam2-ppg-dir, --steam2-rsp-dir

References
  Charlton PH, Bonnici T, Tarassenko L, Clifton DA, Beale R, Watkinson PJ. An assessment of algorithms to
    estimate respiratory rate from the electrocardiogram and photoplethysmogram. Physiol Meas
    2016;37(4):610-26 (RRest toolbox, github.com/peterhcharlton/RRest, Algorithms/fuse_rr/SPA.m)
  Lazaro Plaza J. Non-invasive techniques for respiratory information extraction based on pulse
    photoplethysmogram and electrocardiogram. PhD thesis, Universidad de Zaragoza, 2015
  Iozzia L, Lazaro J, Cerina L, Silvestri D, Mainardi L, Laguna P, Gil E. Monitoring breathing rate by
    fusing the physiological impact of respiration on video-photoplethysmogram with head movements.
    Physiol Meas 2019;40(9):094002
  Sandberg F, Holmer M, Olde B. Monitoring respiration using the pressure sensors in a dialysis machine.
    Physiol Meas 2019;40:014001 (doi 10.1088/1361-6579/aaf978)
  Bailon R, Sornmo L, Laguna P. A robust method for ECG-based estimation of the respiratory frequency
    during stress testing. IEEE Trans Biomed Eng 2006;53(7):1273-85 (origin of peak-conditioned averaging)
"""

import argparse
import importlib.util
import json
import math
import re
import sys
from pathlib import Path
from typing import Dict, Iterator, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import scipy.io as sio
from scipy.signal import butter, detrend, resample_poly, sosfiltfilt, welch
from scipy.signal.windows import tukey

# ---------------------------------------------------------------------------
# Settings
# ---------------------------------------------------------------------------

NAME = "261117_Resp_Regularity_Spectral"
HERE = Path(__file__).resolve().parent
WORKSPACE = Path(r"D:\PPG2RSP_RRWaveNet_Inspired")
OUTPUT_ROOT = WORKSPACE / NAME
ROOT_261007 = WORKSPACE / "261007_RRWaveNet_NoEvent_v1"
ROOT_261013 = WORKSPACE / "261013_RRWavNet_STEAM_BIDMC_v1"
SCRIPTS = {"261007": "261007_RRWaveNet_Ensemble_NoEvent.py", "261013": "261013_RRWaveNet_Adult_SubjectVal_WD.py"}
DATASETS = ("CapnoBase", "BIDMC", "STEAM2")
SOURCE = {"CapnoBase": "261007", "BIDMC": "261013", "STEAM2": "261013"}
MODALITY = {"CapnoBase": "capnogram", "BIDMC": "impedance", "STEAM2": "belt"}
EXCLUDED = {"STEAM2": ("22_LMS", "26_CKK", "12_YJJ"), "BIDMC": ("27",), "CapnoBase": ("0115",)}
MIN_VALID_FRACTION = 0.9

# A: Charlton et al. 2016, RRest v3.0 (SPA.m, setup_universal_params.m, estimate_ref_rr.m)
A = {"fs": 5.0, "win_sec": 32.0, "step_sec": 32.0, "tukey_sec": 2.0, "lowpass_hz": 1.0, "lowpass_order": 4,
     "seg_samples": 64, "nfft": 256, "rr_range_bpm": (4.0, 60.0), "eta": 0.65, "delta_hz": 0.1}
# B: Iozzia et al. 2019 (Lazaro et al. method)
B = {"fs": 4.0, "win_sec": 30.0, "step_sec": 10.0, "seg_sec": 12.0, "nfft": 256, "band_hz": (0.15, 0.5),
     "fallback_band_hz": (0.08, 0.5), "half_width_hz": 0.05, "xi_percent": 40.0}
# C: Sandberg et al. 2019
C = {"fs": 4.0, "win_sec": 40.0, "step_sec": 5.0, "seg_sec": 12.0, "nfft": 256, "lowpass_hz": 2.0,
     "lowpass_order": 6, "norm_band_hz": (0.0, 1.0), "search_band_hz": (0.15, 0.5), "peak_ratio": 0.85,
     "half_width_hz": 0.048, "energy_percent": 30.0}

CLASS2 = ("Regular", "Irregular")
CLASS3 = ("Regular", "Intermediate", "Irregular")

# ---------------------------------------------------------------------------
# Source scripts
# ---------------------------------------------------------------------------

_MODULES: Dict[str, object] = {}


def find_script(name: str, given: Optional[Path], roots: Sequence[Path]) -> Path:
    """The script of the results: given, written in a run manifest, or found nearby."""
    candidates: List[Path] = [Path(given)] if given else []
    for root in roots:
        for manifest in sorted(Path(root).glob("run_manifest*.json")):
            try:
                script = json.loads(manifest.read_text(encoding="utf-8")).get("script")
            except (OSError, ValueError, AttributeError):
                continue
            if script and Path(script).name == name:
                candidates.append(Path(script))
    folders = [HERE, HERE.parent, WORKSPACE] + [Path(r) for r in roots]
    if WORKSPACE.is_dir():
        folders += sorted(p for p in WORKSPACE.iterdir() if p.is_dir())
    candidates += [f / name for f in folders]
    for path in candidates:
        if path.is_file():
            return path.resolve()
    raise FileNotFoundError(f"{name} not found (run manifests of {', '.join(map(str, roots))}, {HERE}, {WORKSPACE}); "
                            f"give it with --script-{name[:6]}")


def load_script(path: Path):
    key = str(path)
    if key not in _MODULES:
        name = "regularity_src_" + re.sub(r"\W", "_", path.stem)
        spec = importlib.util.spec_from_file_location(name, key)
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module
        spec.loader.exec_module(module)
        _MODULES[key] = module
    return _MODULES[key]


# ---------------------------------------------------------------------------
# Reference recordings: (subject, [recording, ...]); a recording is a dict with
# name, rsp, fs, artifacts (seconds, k x 2) and ventilation
# ---------------------------------------------------------------------------

def same_subject(a: object, b: object) -> bool:
    a, b = str(a), str(b)
    return a == b or (a.isdigit() and b.isdigit() and int(a) == int(b))


def wanted(subject: str, dataset: str, chosen: Dict[str, List[str]], excluded: Sequence[str]) -> bool:
    if dataset in chosen and not any(same_subject(subject, s) for s in chosen[dataset]):
        return False
    return not any(same_subject(subject, x) for x in excluded)


def capnobase_text(mod, csv_dir: Path, case_id: str, kind: str, part: str) -> str:
    """A field of the CapnoBase param or meta file (e.g. the ventilation mode); empty when missing."""
    path = mod.capnobase_file(csv_dir, case_id, kind)
    if path is None:
        return ""
    try:
        return mod._text_field(mod.read_capnobase_fields(path), part).strip('"').strip()
    except Exception:
        return ""


def capnobase_recordings(mod, chosen, excluded) -> Iterator[Tuple[str, List[dict]]]:
    cfg = mod.DATASET_CONFIGS["capnobase"]
    csv_dir = Path(cfg["data_path"])
    for case_id in mod.capnobase_case_ids(csv_dir):
        if not wanted(case_id, "CapnoBase", chosen, excluded):
            continue
        try:
            case = mod.read_capnobase_case(csv_dir, case_id, cfg["default_fs"], read_labels=True)
        except Exception as exc:  # report and go on with the next case
            print(f"  [CapnoBase {case_id}] skipped: {exc}")
            continue
        yield case_id, [{"name": "", "rsp": case["co2"], "fs": float(case["co2_fs"]),
                         "artifacts": np.asarray(case["co2_artifacts_sec"], float).reshape(-1, 2),
                         "ventilation_param": capnobase_text(mod, csv_dir, case_id, "param", "vent"),
                         "ventilation_meta": capnobase_text(mod, csv_dir, case_id, "meta", "vent"),
                         "age": capnobase_text(mod, csv_dir, case_id, "meta", "age")}]


def bidmc_recordings(mod, chosen, excluded) -> Iterator[Tuple[str, List[dict]]]:
    cfg = mod.DATASET_CONFIGS["bidmc"]
    records = sio.loadmat(cfg["data_path"])["data"][0]
    fs = float(cfg["default_fs"])
    for index, record in enumerate(records):
        subject = f"{index + 1:02d}"
        if not wanted(subject, "BIDMC", chosen, excluded):
            continue
        try:
            rsp = np.asarray(record["ref"][0, 0]["resp_sig"][0, 0]["imp"][0, 0]["v"]).squeeze()
        except Exception as exc:
            print(f"  [BIDMC {subject}] skipped: {exc}")
            continue
        yield subject, [{"name": "", "rsp": rsp, "fs": fs, "artifacts": np.empty((0, 2))}]


def steam2_recordings(mod, chosen, excluded) -> Iterator[Tuple[str, List[dict]]]:
    cfg = mod.DATASET_CONFIGS["steam2"]
    sessions = mod.steam2_session_files(Path(cfg["ppg_dir"]), Path(cfg["rsp_dir"]))
    for subject, items in sessions.items():
        if not wanted(subject, "STEAM2", chosen, excluded):
            continue
        recordings = []
        try:
            for session, ppg_path, rsp_path in items:
                ppg = mod.read_eeglab_channel(ppg_path, "PPG", cfg["default_fs"])
                rsp = mod.read_eeglab_channel(rsp_path, "RSP", cfg["default_fs"])
                pieces, _ = mod.steam2_session_pieces(ppg, rsp)  # same pieces as the source script
                for k, (_, r, _, r_fs) in enumerate(pieces):
                    name = session if len(pieces) == 1 else f"{session} piece {k + 1}"
                    recordings.append({"name": name, "rsp": r, "fs": float(r_fs), "artifacts": np.empty((0, 2))})
        except Exception as exc:
            print(f"  [STEAM2 {subject}] skipped: {exc}")
            continue
        yield subject, recordings


READERS = {"CapnoBase": capnobase_recordings, "BIDMC": bidmc_recordings, "STEAM2": steam2_recordings}

# ---------------------------------------------------------------------------
# Signal preparation
# ---------------------------------------------------------------------------


def bridge(x: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Signal with missing samples bridged by linear interpolation, and the valid mask."""
    x = np.asarray(x, dtype=np.float64).ravel()
    valid = np.isfinite(x)
    if valid.sum() >= 2 and not valid.all():
        idx = np.arange(x.size)
        x = np.interp(idx, idx[valid], x[valid])
    return x, valid


def to_rate(x: np.ndarray, valid: np.ndarray, fs_in: float, fs_out: float) -> Tuple[np.ndarray, np.ndarray]:
    """Resample to fs_out (polyphase, anti-aliased) and the valid mask on the new grid."""
    frac = fs_out / fs_in
    up, down = (int(round(fs_out * 1000)), int(round(fs_in * 1000)))
    g = math.gcd(up, down)
    y = resample_poly(x, up // g, down // g)
    n = y.size
    pos = np.minimum((np.arange(n) / frac).astype(int), valid.size - 1)
    return y, valid[pos]


def prep_a(rsp: np.ndarray, fs: float, fs_out: float) -> Tuple[np.ndarray, np.ndarray]:
    """RRest reference preprocessing: detrend and Tukey taper of the whole recording, low-pass, resample."""
    x, valid = bridge(rsp)
    if valid.sum() < 2:
        return np.empty(0), np.empty(0, bool)
    alpha = min(1.0, 2.0 * A["tukey_sec"] / max(1e-9, x.size / fs))
    x = detrend(x) * tukey(x.size, alpha)
    sos = butter(A["lowpass_order"], A["lowpass_hz"], btype="lowpass", fs=fs, output="sos")
    x = sosfiltfilt(sos, x)
    return to_rate(x, valid, fs, fs_out)


def prep_c(rsp: np.ndarray, fs: float) -> Tuple[np.ndarray, np.ndarray]:
    """Sandberg et al.: low-pass 6th-order Butterworth 2 Hz, resample to 4 Hz."""
    x, valid = bridge(rsp)
    if valid.sum() < 2:
        return np.empty(0), np.empty(0, bool)
    x = x - np.mean(x)
    sos = butter(C["lowpass_order"], min(C["lowpass_hz"], 0.45 * fs), btype="lowpass", fs=fs, output="sos")
    x = sosfiltfilt(sos, x)
    return to_rate(x, valid, fs, C["fs"])


def windows(n: int, fs: float, win_sec: float, step_sec: float) -> List[Tuple[int, int]]:
    w, s = int(round(win_sec * fs)), int(round(step_sec * fs))
    return [(a, a + w) for a in range(0, n - w + 1, s)]


def overlaps_artifact(a_sec: float, b_sec: float, spans: np.ndarray) -> bool:
    return bool(len(spans) and np.any((spans[:, 0] < b_sec) & (spans[:, 1] > a_sec)))


def welch_psd(x: np.ndarray, fs: float, seg: int, nfft: int) -> Tuple[np.ndarray, np.ndarray]:
    seg = min(seg, x.size)
    return welch(x, fs=fs, window="hamming", nperseg=seg, noverlap=seg // 2, nfft=max(nfft, seg),
                 detrend=False, scaling="density")


def local_peaks(p: np.ndarray, f: np.ndarray, band: Tuple[float, float]) -> np.ndarray:
    k = np.arange(1, p.size - 1)
    is_peak = (p[k] > p[k - 1]) & (p[k] > p[k + 1]) & (f[k] > band[0]) & (f[k] < band[1])
    return k[is_peak]


# ---------------------------------------------------------------------------
# Criteria
# ---------------------------------------------------------------------------

def criterion_a(seg: np.ndarray) -> dict:
    x = detrend(seg)
    f, p = welch_psd(x, A["fs"], A["seg_samples"], A["nfft"])
    rr = (A["rr_range_bpm"][0] / 60.0, A["rr_range_bpm"][1] / 60.0)
    peak = int(np.argmax(p))
    fp = float(f[peak])

    def share(delta: float) -> float:
        lo, hi = max(rr[0], fp - 0.4 * delta), min(rr[1], fp + 0.4 * delta)
        den = p[(f >= fp - delta) & (f <= fp + delta)].sum()
        return float(p[(f >= lo) & (f <= hi)].sum() / den) if den > 0 else math.nan

    p1, p2 = share(A["delta_hz"]), share(2 * A["delta_hz"])
    cand = local_peaks(p, f, rr)
    rr_peak = float(60 * f[cand[np.argmax(p[cand])]]) if cand.size else math.nan
    band = (f >= rr[0]) & (f <= rr[1])
    q = p[band] / p[band].sum() if p[band].sum() > 0 else np.full(int(band.sum()), np.nan)
    q = q[q > 0]
    entropy = float(-(q * np.log(q)).sum() / np.log(band.sum())) if q.size and band.sum() > 1 else math.nan
    class2 = "Regular" if p1 > A["eta"] else "Irregular"
    class3 = "Regular" if p1 > A["eta"] else ("Intermediate" if p2 > A["eta"] else "Irregular")
    return {"Peak_Hz": fp, "P_delta": p1, "P_2delta": p2, "RR_Peak_per_min": rr_peak, "Spectral_Entropy": entropy,
            "Class_2": class2, "Class_3": class3}


def criterion_b(seg: np.ndarray) -> dict:
    x = detrend(seg)
    f, p = welch_psd(x, B["fs"], int(round(B["seg_sec"] * B["fs"])), B["nfft"])
    used = B["band_hz"]
    cand = local_peaks(p, f, used)
    if not cand.size:
        used = B["fallback_band_hz"]
        cand = local_peaks(p, f, used)
    if not cand.size:
        return {"Peak_Hz": math.nan, "Search_Band": f"{used[0]:g}-{used[1]:g}", "P_percent": math.nan,
                "Class_2": "Irregular"}
    fp = float(f[cand[np.argmax(p[cand])]])
    den = p[(f >= B["band_hz"][0]) & (f <= B["band_hz"][1])].sum()
    num = p[(f >= fp - B["half_width_hz"]) & (f <= fp + B["half_width_hz"])].sum()
    share = float(100.0 * num / den) if den > 0 else math.nan
    return {"Peak_Hz": fp, "Search_Band": f"{used[0]:g}-{used[1]:g}", "P_percent": share,
            "Class_2": "Regular" if share >= B["xi_percent"] else "Irregular"}


def criterion_c(seg: np.ndarray) -> dict:
    x = detrend(seg)
    f, p = welch_psd(x, C["fs"], int(round(C["seg_sec"] * C["fs"])), C["nfft"])
    norm = (f >= C["norm_band_hz"][0]) & (f <= C["norm_band_hz"][1])
    total = p[norm].sum()
    if total <= 0:
        return {"Peak_Hz": math.nan, "Energy_percent": math.nan, "Peak_Ratio": math.nan, "Class_2": "Irregular"}
    p = p / total
    cand = local_peaks(p, f, C["search_band_hz"])
    all_peaks = local_peaks(p, f, C["norm_band_hz"])
    if not cand.size or not all_peaks.size:
        return {"Peak_Hz": math.nan, "Energy_percent": math.nan, "Peak_Ratio": math.nan, "Class_2": "Irregular"}
    k = cand[np.argmax(p[cand])]
    fp, ratio = float(f[k]), float(p[k] / p[all_peaks].max())
    share = float(100.0 * p[norm & (f >= fp - C["half_width_hz"]) & (f <= fp + C["half_width_hz"])].sum())
    ok = ratio > C["peak_ratio"] and share > C["energy_percent"]
    return {"Peak_Hz": fp, "Energy_percent": share, "Peak_Ratio": ratio, "Class_2": "Regular" if ok else "Irregular"}


# ---------------------------------------------------------------------------
# Run
# ---------------------------------------------------------------------------

def parse_subjects(items: Optional[Sequence[str]]) -> Dict[str, List[str]]:
    chosen: Dict[str, List[str]] = {}
    for item in items or []:
        if ":" not in item:
            raise ValueError(f"--subjects takes Dataset:Subject, got {item}")
        dataset, subject = item.split(":", 1)
        match = [d for d in DATASETS if d.lower() == dataset.lower()]
        if not match:
            raise ValueError(f"unknown dataset in {item} ({', '.join(DATASETS)})")
        chosen.setdefault(match[0], []).append(subject)
    return chosen


def dataset_rows(args, dataset: str, chosen: Dict[str, List[str]]) -> Dict[str, List[dict]]:
    source = SOURCE[dataset]
    roots = [args.root_261007] if source == "261007" else [args.root_261013]
    script = find_script(SCRIPTS[source], args.script_261007 if source == "261007" else args.script_261013, roots)
    mod = load_script(script)
    for key, field, value in (("capnobase", "data_path", args.capnobase_path), ("bidmc", "data_path", args.bidmc_path),
                              ("steam2", "ppg_dir", args.steam2_ppg_dir), ("steam2", "rsp_dir", args.steam2_rsp_dir)):
        if value:
            mod.DATASET_CONFIGS[key][field] = str(value)
    print(f"\n{dataset}: reference ({MODALITY[dataset]}) read with {script.name}", flush=True)
    excluded = [] if args.keep_issue_subjects else list(EXCLUDED[dataset])
    out: Dict[str, List[dict]] = {"A": [], "B": [], "C": [], "info": []}
    for subject, recordings in READERS[dataset](mod, chosen, excluded):
        counts = {"A": 0, "B": 0, "C": 0}
        for rec_index, rec in enumerate(recordings):
            spans = rec["artifacts"] if not args.keep_artifacts else np.empty((0, 2))
            base = {"Dataset": dataset, "Subject": subject, "Recording_Index": rec_index, "Recording": rec["name"]}
            out["info"].append({**base, "Duration_Sec": np.asarray(rec["rsp"]).size / rec["fs"],
                                "Age": rec.get("age", ""), "Ventilation_Meta": rec.get("ventilation_meta", ""),
                                "Ventilation_Param": rec.get("ventilation_param", "")})
            xa, va = prep_a(rec["rsp"], rec["fs"], A["fs"])
            xb, vb = prep_a(rec["rsp"], rec["fs"], B["fs"])
            xc, vc = prep_c(rec["rsp"], rec["fs"])
            for key, (x, v, cfg, fun) in {"A": (xa, va, A, criterion_a), "B": (xb, vb, B, criterion_b),
                                          "C": (xc, vc, C, criterion_c)}.items():
                for a, b in windows(x.size, cfg["fs"], cfg["win_sec"], cfg["step_sec"]):
                    t0, t1 = a / cfg["fs"], b / cfg["fs"]
                    if v[a:b].mean() < MIN_VALID_FRACTION or overlaps_artifact(t0, t1, spans):
                        continue
                    out[key].append({**base, "Start_Sec": t0, "End_Sec": t1, **fun(x[a:b])})
                    counts[key] += 1
        print(f"  {subject}: {len(recordings)} recording(s); windows A {counts['A']}, B {counts['B']}, C {counts['C']}",
              flush=True)
    return out


def subject_class(shares: pd.Series, order: Sequence[str]) -> str:
    """The class with the most windows; ties go to the less regular class."""
    best = shares.max()
    return [c for c in order if shares.get(c, 0.0) == best][-1]


def summarize(win: Dict[str, pd.DataFrame], info: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    keys = ["Dataset", "Subject"]
    meta = info.groupby(keys, sort=False).agg(Recordings=("Recording_Index", "nunique"),
                                              Duration_Min=("Duration_Sec", lambda s: s.sum() / 60.0),
                                              Age=("Age", "first"), Ventilation_Meta=("Ventilation_Meta", "first"),
                                              Ventilation_Param=("Ventilation_Param", "first")).reset_index()
    rows = []
    for (dataset, subject), m in meta.set_index(keys).iterrows():
        row = {"Dataset": dataset, "Subject": subject, **m.to_dict()}
        a = win["A"][(win["A"].Dataset == dataset) & (win["A"].Subject == subject)]
        row["A_Windows"] = int(len(a))
        if len(a):
            s3 = a.Class_3.value_counts(normalize=True)
            for c in CLASS3:
                row[f"A_{c}_%"] = 100.0 * float(s3.get(c, 0.0))
            row["A_Class_2"] = "Regular" if row["A_Regular_%"] >= 50.0 else "Irregular"
            row["A_Class_3"] = subject_class(s3, CLASS3)
            row["A_P_delta_Median"] = float(a.P_delta.median())
            row["Spectral_Entropy_Median"] = float(a.Spectral_Entropy.median())
            row["RR_Peak_Median_per_min"] = float(a.RR_Peak_per_min.median())
        for key, col in (("B", "P_percent"), ("C", "Energy_percent")):
            w = win[key][(win[key].Dataset == dataset) & (win[key].Subject == subject)]
            row[f"{key}_Windows"] = int(len(w))
            if len(w):
                row[f"{key}_Regular_%"] = 100.0 * float((w.Class_2 == "Regular").mean())
                row[f"{key}_Class_2"] = "Regular" if row[f"{key}_Regular_%"] >= 50.0 else "Irregular"
                row[f"{key}_{col}_Median"] = float(w[col].median())
        rows.append(row)
    subjects = pd.DataFrame(rows)
    drows = []
    for dataset, g in subjects.groupby("Dataset", sort=False):
        aw = win["A"][win["A"].Dataset == dataset]
        for key, col in (("A2", "Class_2"), ("A3", "Class_3"), ("B", "Class_2"), ("C", "Class_2")):
            w = aw if key.startswith("A") else win[key][win[key].Dataset == dataset]
            order = CLASS3 if key == "A3" else CLASS2
            sub_col = {"A2": "A_Class_2", "A3": "A_Class_3", "B": "B_Class_2", "C": "C_Class_2"}[key]
            for c in order:
                drow = {"Dataset": dataset, "Criterion": {"A2": "A (2 classes)", "A3": "A (3 classes)",
                                                          "B": "B (2 classes)", "C": "C (2 classes)"}[key],
                        "Class": c, "Windows": int((w[col] == c).sum()) if len(w) else 0,
                        "Windows_%": 100.0 * float((w[col] == c).mean()) if len(w) else math.nan,
                        "Subjects": int((g[sub_col] == c).sum()) if sub_col in g else 0,
                        "Subjects_%": 100.0 * float((g[sub_col] == c).mean()) if sub_col in g else math.nan}
                if key.startswith("A") and len(w):
                    drow["Spectral_Entropy_Median"] = float(w.loc[w[col] == c, "Spectral_Entropy"].median()) \
                        if (w[col] == c).any() else math.nan
                drows.append(drow)
    return subjects, pd.DataFrame(drows)


def criteria_table() -> pd.DataFrame:
    return pd.DataFrame([
        ("A", "Charlton et al. 2016, RRest v3.0 SPA (modified from Lazaro Plaza 2015)",
         "RRest reference preprocessing (detrend, Tukey 2 s, low-pass 1 Hz, 5 Hz); 32 s consecutive windows; Welch "
         "Hamming 64 samples, 50% overlap, NFFT 256; f_p = largest power; P(d) = power in f_p +- 0.4 d (within 4-60 /min) "
         "/ power in f_p +- d",
         "2 classes: Regular P(0.1 Hz) > 0.65, else Irregular. 3 classes: Regular P(0.1) > 0.65; Intermediate "
         "P(0.2) > 0.65; Irregular otherwise (RRest retries with twice delta)"),
        ("B", "Iozzia et al. 2019 (method of Lazaro et al.)",
         "same preprocessing at 4 Hz; 30 s windows every 10 s; Welch Hamming 12 s, 50% overlap (segments not given in "
         "the paper); f_p = largest local peak in 0.15-0.5 Hz (0.08-0.5 Hz if none); P = 100 x power in f_p +- 0.05 Hz / "
         "power in 0.15-0.5 Hz", "2 classes: Regular (peaked) P >= 40%, else Irregular"),
        ("C", "Sandberg et al. 2019",
         "low-pass Butterworth 6th order 2 Hz, 4 Hz; 40 s windows every 5 s; Welch of 50% overlapping 12 s "
         "sub-intervals, power normalized in 0-1 Hz; f_p = largest local peak in 0.15-0.5 Hz, > 85% of the largest "
         "peak (no reference-rate tracking here)",
         "2 classes: Regular (sufficient) energy in f_p +- 0.048 Hz > 30% of the 0-1 Hz energy, else Irregular"),
        ("Spectral entropy", "normalized Shannon entropy of the criterion A spectrum within 4-60 /min",
         "H = -sum(q log q) / log(n), q = power / total power in the band", "reported only (no published cut-off)"),
        ("Subject level", "this script", "share of windows per class",
         "subject class = class with the most windows (ties to the less regular class); 2 classes: Regular when at "
         "least half of the windows are Regular"),
        ("Windows used", "this script", f"at least {MIN_VALID_FRACTION:.0%} valid samples; CapnoBase windows "
         "overlapping an expert CO2 artifact label left out", ""),
    ], columns=["Criterion", "Source", "Spectrum and window", "Classes"])


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Regularity of the respiration reference from its spectral concentration")
    p.add_argument("--datasets", nargs="+", choices=list(DATASETS), default=list(DATASETS))
    p.add_argument("--subjects", nargs="+", default=None, help="Dataset:Subject (default: every subject)")
    p.add_argument("--keep-issue-subjects", action="store_true")
    p.add_argument("--keep-artifacts", action="store_true", help="keep CapnoBase windows with expert CO2 artifacts")
    p.add_argument("--root-261007", type=Path, default=ROOT_261007)
    p.add_argument("--root-261013", type=Path, default=ROOT_261013)
    p.add_argument("--output-root", type=Path, default=OUTPUT_ROOT)
    p.add_argument("--script-261007", type=Path, default=None)
    p.add_argument("--script-261013", type=Path, default=None)
    p.add_argument("--capnobase-path", type=Path, default=None)
    p.add_argument("--bidmc-path", type=Path, default=None)
    p.add_argument("--steam2-ppg-dir", type=Path, default=None)
    p.add_argument("--steam2-rsp-dir", type=Path, default=None)
    return p.parse_args(list(sys.argv[1:] if argv is None else argv))


def main(argv: Optional[Sequence[str]] = None) -> None:
    args = parse_args(argv)
    chosen = parse_subjects(args.subjects)
    datasets = [d for d in args.datasets if not chosen or d in chosen]
    out_root = Path(args.output_root)
    out_root.mkdir(parents=True, exist_ok=True)
    collected: Dict[str, List[dict]] = {"A": [], "B": [], "C": [], "info": []}
    for dataset in datasets:
        for key, rows in dataset_rows(args, dataset, chosen).items():
            collected[key] += rows
    win = {k: pd.DataFrame(collected[k]) for k in ("A", "B", "C")}
    info = pd.DataFrame(collected["info"])
    if info.empty or win["A"].empty:
        print("no windows: nothing written")
        return
    subjects, datasets_table = summarize(win, info)
    path = out_root / f"{NAME}.xlsx"
    with pd.ExcelWriter(path, engine="openpyxl") as writer:
        win["A"].round(5).to_excel(writer, sheet_name="Windows_A", index=False)
        win["B"].round(5).to_excel(writer, sheet_name="Windows_B", index=False)
        win["C"].round(5).to_excel(writer, sheet_name="Windows_C", index=False)
        subjects.round(4).to_excel(writer, sheet_name="Subjects", index=False)
        datasets_table.round(3).to_excel(writer, sheet_name="Datasets", index=False)
        criteria_table().to_excel(writer, sheet_name="Criteria", index=False)
    print(f"\n{len(subjects)} subjects; windows A {len(win['A'])}, B {len(win['B'])}, C {len(win['C'])}; table: {path}")
    print(datasets_table[datasets_table.Criterion.str.startswith("A")].round(1).to_string(index=False))


if __name__ == "__main__":
    main()
