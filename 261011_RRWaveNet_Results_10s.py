r"""261011_RRWaveNet_Results_10s: all results of the 10 s windows of
261007_RRWaveNet_Ensemble_NoEvent, while its 20 s part is still training.

Nothing is trained. From the members already stored by 261007 (10 s
windows only) it writes:
1. the 261007 summary of the 10 s windows: ensembles of the finished folds,
   FFT and Davies & Mandic metrics of every dataset and variant, the
   ablation comparison (paired Wilcoxon against the full model), the
   per-subject table, the per-minute tables, figures and the workbook;
2. the stem fusion feature analysis, Stem Fusion O (full) vs X
   (no_stem_fusion): 261010_RRWaveNet_StemFusion_Feature_Analysis;
3. the residual encoder feature analysis of the full model:
   261010_RRWaveNet_Encoder_Feature_Analysis.
Folds whose 3 members are not all stored yet are left out and listed in
incomplete_folds.csv, so the script can be run at any time and again later.

Outputs: <261007 results root>\Summary_10s\
  (the 261007 summary files, Stem_Fusion_OX\ and Encoder_Feature_Analysis\)
The 261007 run itself writes its own Summary\ (both windows) when it ends;
Summary_10s\ is not touched by it.

Run next to the three scripts (Spyder: just run the file):
  python 261011_RRWaveNet_Results_10s.py
Options: --results-root <261007 results>, --datasets capnobase bidmc,
  --skip-feature-analyses, --force-cpu, and the data path options of 261007.
The data paths, model and results root default to those in the 261007 file.
"""

import argparse
import importlib.util
import shutil
import sys
from pathlib import Path
from typing import Optional, Sequence

sys.dont_write_bytecode = True

SCRIPT_DIR = Path(__file__).resolve().parent
BACKBONE = SCRIPT_DIR / "261007_RRWaveNet_Ensemble_NoEvent.py"
STEM_ANALYSIS = SCRIPT_DIR / "261010_RRWaveNet_StemFusion_Feature_Analysis.py"
ENCODER_ANALYSIS = SCRIPT_DIR / "261010_RRWaveNet_Encoder_Feature_Analysis.py"
WINDOW = "w10s_nonoverlap"
OUTPUT_FOLDER = "Summary_10s"
# Set to False to write only the performance summary (faster).
RUN_FEATURE_ANALYSES = True


def load_script(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, str(path))
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def data_path_arguments(args: argparse.Namespace):
    extra = []
    for option, value in (
        ("--capnobase-path", args.capnobase_path),
        ("--bidmc-path", args.bidmc_path),
        ("--steam2-ppg-dir", args.steam2_ppg_dir),
        ("--steam2-rsp-dir", args.steam2_rsp_dir),
    ):
        if value:
            extra += [option, str(value)]
    return extra


def main(args: argparse.Namespace) -> None:
    backbone = load_script(Path(args.backbone), "rrwavenet_261007_results")
    paths = data_path_arguments(args)
    summary_args = backbone.parse_args(
        ["--summary-only", "--windows", WINDOW, "--datasets", *args.datasets, *paths]
        + (["--results-root", str(args.results_root)] if args.results_root else [])
    )
    results_root = Path(summary_args.results_root).resolve()
    configs = backbone.DATASET_CONFIGS
    configs["capnobase"]["use_artifact_labels"] = summary_args.capnobase_artifacts
    if summary_args.capnobase_path:
        configs["capnobase"]["data_path"] = str(summary_args.capnobase_path)
    if summary_args.bidmc_path:
        configs["bidmc"]["data_path"] = str(summary_args.bidmc_path)
    if summary_args.steam2_ppg_dir:
        configs["steam2"]["ppg_dir"] = str(summary_args.steam2_ppg_dir)
    if summary_args.steam2_rsp_dir:
        configs["steam2"]["rsp_dir"] = str(summary_args.steam2_rsp_dir)
    out_root = results_root / OUTPUT_FOLDER
    out_root.mkdir(parents=True, exist_ok=True)

    # 1. Performance summary of the 10 s windows (summarize_results of 261007,
    #    written to Summary\ and copied to Summary_10s\; no manifest is written).
    print(f"=== 1. 261007 summary, 10 s windows ({results_root}) ===")
    variants = [backbone.VARIANTS[name] for name in backbone.VARIANTS if name in summary_args.variants]
    window_specs = [spec for spec in backbone.WINDOW_SPECS if spec.name == WINDOW]
    dataset_keys = [key for key in backbone.DATASET_ORDER if key in summary_args.datasets]
    backbone.summarize_results(summary_args, results_root, variants, window_specs, dataset_keys)
    summary_root = results_root / "Summary"
    for item in summary_root.iterdir():
        if item.is_dir() and item.name in ("Stem_Fusion_OX", "Encoder_Feature_Analysis"):
            continue
        target = out_root / item.name
        if item.is_dir():
            shutil.copytree(item, target, dirs_exist_ok=True)
        else:
            shutil.copy2(item, target)

    if args.skip_feature_analyses or not RUN_FEATURE_ANALYSES:
        print(f"\nSaved under: {out_root}")
        return
    common = ["--backbone", str(Path(args.backbone).resolve()), "--results-root", str(results_root),
              "--windows", WINDOW, "--datasets", *args.datasets, *paths]
    if args.force_cpu:
        common.append("--force-cpu")

    # 2. Stem Fusion O vs X.
    print("\n=== 2. Stem fusion feature analysis (O = full, X = no_stem_fusion), 10 s ===")
    stem = load_script(STEM_ANALYSIS, "stem_fusion_analysis_10s")
    stem.analyse(stem.parse_args(common + ["--output", str(out_root / "Stem_Fusion_OX")]))

    # 3. Residual encoder blocks of the full model.
    print("\n=== 3. Residual encoder feature analysis (full model), 10 s ===")
    encoder = load_script(ENCODER_ANALYSIS, "encoder_feature_analysis_10s")
    encoder.analyse(encoder.parse_args(common + ["--output", str(out_root / "Encoder_Feature_Analysis")]))
    print(f"\nAll 10 s results saved under: {out_root}")


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--backbone", type=Path, default=BACKBONE)
    parser.add_argument("--results-root", type=Path, default=None,
                        help="Results root of the 261007 run (default: its PROJECT_ROOT).")
    parser.add_argument("--datasets", nargs="+", default=["capnobase", "bidmc", "steam2"],
                        choices=["capnobase", "bidmc", "steam2"])
    parser.add_argument("--skip-feature-analyses", action="store_true")
    parser.add_argument("--force-cpu", action="store_true")
    parser.add_argument("--capnobase-path", type=Path, default=None)
    parser.add_argument("--bidmc-path", type=Path, default=None)
    parser.add_argument("--steam2-ppg-dir", type=Path, default=None)
    parser.add_argument("--steam2-rsp-dir", type=Path, default=None)
    return parser.parse_args(argv)


if __name__ == "__main__":
    main(parse_args())
