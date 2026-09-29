"""Walk-forward (rolling-origin) evaluation of the AAPL model over the
processed training table, instead of relying on a single train/validation/
test split.

Usage:
    python -m src.evaluate_walk_forward
    python -m src.evaluate_walk_forward --train-sessions 8 --test-sessions 4

Addresses limitations reports/aapl_v2_experiment.md calls out explicitly:
a single four-session test partition, and overlapping 30-minute targets
making adjacent rows non-independent. Multiple folds, each an independent
out-of-sample window, give a distribution of accuracy estimates instead of
one point estimate.
"""
import argparse
from pathlib import Path

import pandas as pd

from src.evaluation import run_walk_forward, summarize_folds
from src.logging_config import configure_logging

logger = configure_logging("evaluate_walk_forward")

project_root = Path(__file__).resolve().parent.parent

FEATURE_COLS = [
    "range_pct",
    "body_pct",
    "return_5m_pct",
    "return_15m_pct",
    "return_30m_pct",
    "volatility_30m",
    "volume_relative",
    "rsi_14",
    "macd_diff",
    "bb_width_pct",
    "atr_pct",
]
TARGET_COL = "target_up_30m"


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Walk-forward evaluation of the AAPL model over trading sessions."
    )
    parser.add_argument("--train-sessions", type=int, default=10)
    parser.add_argument("--test-sessions", type=int, default=2)
    parser.add_argument("--step-sessions", type=int, default=None)
    args = parser.parse_args()

    file_path = project_root / "data" / "processed" / "aapl_training_1mo.parquet"
    df = pd.read_parquet(file_path).sort_index()

    n_sessions = df.index.normalize().nunique()
    logger.info(
        "Loaded %d rows across %d trading sessions from %s",
        len(df), n_sessions, file_path,
    )

    fold_results = run_walk_forward(
        df,
        feature_cols=FEATURE_COLS,
        target_col=TARGET_COL,
        train_sessions=args.train_sessions,
        test_sessions=args.test_sessions,
        step_sessions=args.step_sessions,
    )

    if fold_results.empty:
        logger.warning(
            "No usable folds: %d sessions is not enough for "
            "train_sessions=%d + test_sessions=%d.",
            n_sessions, args.train_sessions, args.test_sessions,
        )
        raise SystemExit(0)

    pd.set_option("display.width", 120)
    print(f"\n{len(fold_results)} walk-forward folds:\n")
    print(fold_results.round(4).to_string(index=False))

    summary = summarize_folds(fold_results)
    print("\nAcross-fold summary (mean / std):\n")
    print(summary.round(4).to_string())

    mean_model = fold_results["model_accuracy"].mean()
    mean_baseline = fold_results["baseline_accuracy"].mean()
    folds_model_wins = (
        fold_results["model_accuracy"] > fold_results["baseline_accuracy"]
    ).sum()

    print(
        f"\nModel beat the baseline in {folds_model_wins}/{len(fold_results)} folds. "
        f"Mean accuracy: model {mean_model:.2%} vs baseline {mean_baseline:.2%}."
    )


if __name__ == "__main__":
    main()
