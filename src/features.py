"""Build the AAPL training table (features + same-session target).

Usage:
    python -m src.features
    python -m src.features --input <raw.parquet> --output <processed.parquet>

Reads config.RAW_PRICES_PATH and writes config.PROCESSED_PATH by default; a
table already at the output path is archived first, not overwritten.
"""

import argparse
from pathlib import Path

import pandas as pd

from src.artifacts import archive_existing
from src.config import (
    FEATURE_COLS,
    HORIZON_BARS,
    HORIZON_MINUTES,
    PROCESSED_PATH,
    RAW_PRICES_PATH,
    TARGET_COL,
    TICKER,
)
from src.feature_utils import build_features
from src.logging_config import configure_logging
from src.target_utils import build_target

logger = configure_logging("features")


def build_training_table(raw: pd.DataFrame, ticker: str = TICKER) -> pd.DataFrame:
    """Features and target for one ticker, keeping only fully usable rows.

    A row is kept only if every feature, the target, and the target's
    timestamp are present — the timestamp is what train.py and evaluation.py
    use to keep a label that resolves in a later partition out of training.
    """
    prices = raw.xs(ticker, axis=1, level="Ticker").copy().sort_index()
    prices = build_features(prices)
    prices = build_target(prices, horizon_minutes=HORIZON_MINUTES, periods=HORIZON_BARS)

    columns = FEATURE_COLS + [TARGET_COL, "target_time"]
    training_data = prices[columns].dropna(subset=columns).copy()
    training_data[TARGET_COL] = training_data[TARGET_COL].astype(int)
    return training_data


def main() -> None:
    parser = argparse.ArgumentParser(description="Build the AAPL training table.")
    parser.add_argument("--input", type=Path, default=RAW_PRICES_PATH)
    parser.add_argument("--output", type=Path, default=PROCESSED_PATH)
    args = parser.parse_args()

    raw = pd.read_parquet(args.input)
    training_data = build_training_table(raw)

    logger.info("Raw rows for %s: %d", TICKER, len(raw))
    logger.info("Rows kept for training: %d", len(training_data))
    logger.info("First rows:\n%s", training_data.head())

    args.output.parent.mkdir(parents=True, exist_ok=True)
    archived = archive_existing(args.output)
    if archived:
        logger.info("Previous table archived to: %s", archived)

    training_data.to_parquet(args.output, index=True)
    logger.info("Saved: %s", args.output)


if __name__ == "__main__":
    main()
