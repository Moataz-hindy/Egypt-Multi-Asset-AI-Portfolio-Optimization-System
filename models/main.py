# =============================================================================
# main.py — Orchestrator for the AI Investment Recommendation System
#
# Pipeline order per dataset:
#   1. Classical Classification  (LogReg, DT, XGBoost)
#   2. Classical Regression      (LinReg, RF, SVR)
#   3. LSTM Classification
#   4. LSTM Regression
#
# Usage:
#   python main.py                   # run everything
#   python main.py --skip-lstm       # skip LSTM (faster, no GPU needed)
#   python main.py --dataset EGX30   # run only one dataset
# =============================================================================

import os
import sys
import argparse
import warnings
import traceback
import pandas as pd

warnings.filterwarnings("ignore")

from config import DATASETS
from utils import create_output_dirs, load_dataset
from classification_models import run_all_classifiers
from regression_models import run_all_regressors
from lstm_classification import run_all_lstm_classifiers
from lstm_regression import run_all_lstm_regressors


# =============================================================================
# CLI arguments
# =============================================================================

def parse_args():
    parser = argparse.ArgumentParser(
        description="AI Investment Recommendation System — Full ML Pipeline"
    )
    parser.add_argument(
        "--skip-lstm", action="store_true",
        help="Skip LSTM models (useful when TensorFlow is not available / no GPU)"
    )
    parser.add_argument(
        "--dataset", type=str, default=None,
        help="Run only a specific dataset by name (e.g. EGX30, Gold, TBills)"
    )
    parser.add_argument(
        "--skip-classical", action="store_true",
        help="Skip classical ML models (run only LSTM)"
    )
    return parser.parse_args()


# =============================================================================
# Per-dataset pipeline
# =============================================================================

def run_pipeline_for_dataset(
    dataset_name: str,
    dataset_path: str,
    skip_lstm: bool = False,
    skip_classical: bool = False,
):
    print(f"\n{'#'*70}")
    print(f"#  DATASET: {dataset_name}")
    print(f"{'#'*70}")

    # Load data
    if not os.path.exists(dataset_path):
        print(f"  [WARNING] File not found: {dataset_path}. Skipping.")
        return

    df = load_dataset(dataset_path, dataset_name)

    if len(df) < 60:
        print(f"  [WARNING] Only {len(df)} rows — dataset too small, skipping.")
        return

    # ------------------------------------------------------------------
    # 1. Classical Classification
    # ------------------------------------------------------------------
    if not skip_classical:
        print(f"\n{'─'*60}")
        print(f"  CLASSICAL CLASSIFICATION — {dataset_name}")
        print(f"{'─'*60}")
        try:
            run_all_classifiers(df, dataset_name)
        except Exception as exc:
            print(f"  [ERROR] Classical classification failed: {exc}")
            traceback.print_exc()

    # ------------------------------------------------------------------
    # 2. Classical Regression
    # ------------------------------------------------------------------
    if not skip_classical:
        print(f"\n{'─'*60}")
        print(f"  CLASSICAL REGRESSION — {dataset_name}")
        print(f"{'─'*60}")
        try:
            run_all_regressors(df, dataset_name)
        except Exception as exc:
            print(f"  [ERROR] Classical regression failed: {exc}")
            traceback.print_exc()

    # ------------------------------------------------------------------
    # 3. LSTM Classification
    # ------------------------------------------------------------------
    if not skip_lstm:
        print(f"\n{'─'*60}")
        print(f"  LSTM CLASSIFICATION — {dataset_name}")
        print(f"{'─'*60}")
        try:
            run_all_lstm_classifiers(df, dataset_name)
        except Exception as exc:
            print(f"  [ERROR] LSTM classification failed: {exc}")
            traceback.print_exc()

    # ------------------------------------------------------------------
    # 4. LSTM Regression
    # ------------------------------------------------------------------
    if not skip_lstm:
        print(f"\n{'─'*60}")
        print(f"  LSTM REGRESSION — {dataset_name}")
        print(f"{'─'*60}")
        try:
            run_all_lstm_regressors(df, dataset_name)
        except Exception as exc:
            print(f"  [ERROR] LSTM regression failed: {exc}")
            traceback.print_exc()


# =============================================================================
# Global summary: merge all per-dataset metric CSVs into one master table
# =============================================================================

def build_master_summary():
    """Read all individual metric CSVs and combine into master_summary.csv."""
    from config import METRICS_DIR
    all_rows = []
    for fname in os.listdir(METRICS_DIR):
        if fname.endswith("_metrics.csv"):
            path = os.path.join(METRICS_DIR, fname)
            try:
                chunk = pd.read_csv(path)
                all_rows.append(chunk)
            except Exception:
                pass

    if not all_rows:
        print("\n[INFO] No metric files found to summarise.")
        return

    master = pd.concat(all_rows, ignore_index=True)
    out_path = os.path.join(METRICS_DIR, "master_summary.csv")
    master.to_csv(out_path, index=False)
    print(f"\n[SAVED] Master summary → {out_path}")
    print(master.to_string(index=False))


# =============================================================================
# Entry point
# =============================================================================

def main():
    args = parse_args()

    print("=" * 70)
    print("  AI-BASED INVESTMENT RECOMMENDATION SYSTEM")
    print("  Egypt Multi-Asset ML Pipeline")
    print("=" * 70)

    # Prepare output folders
    create_output_dirs()

    # Determine which datasets to process
    datasets_to_run = {}
    if args.dataset:
        if args.dataset not in DATASETS:
            print(f"[ERROR] Unknown dataset '{args.dataset}'. "
                  f"Available: {list(DATASETS.keys())}")
            sys.exit(1)
        datasets_to_run[args.dataset] = DATASETS[args.dataset]
    else:
        datasets_to_run = DATASETS

    # Run pipeline for each dataset
    for name, path in datasets_to_run.items():
        run_pipeline_for_dataset(
            dataset_name=name,
            dataset_path=path,
            skip_lstm=args.skip_lstm,
            skip_classical=args.skip_classical,
        )

    # Aggregate all metrics into one master file
    build_master_summary()

    print("\n" + "=" * 70)
    print("  PIPELINE COMPLETE — All outputs saved in outputs/")
    print("=" * 70)


if __name__ == "__main__":
    main()
