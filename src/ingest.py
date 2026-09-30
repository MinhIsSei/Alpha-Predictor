"""Download five-minute OHLCV data for the configured tickers.

Usage:
    python -m src.ingest
    python -m src.ingest --output experiments/raw.parquet

Writes to config.RAW_PRICES_PATH by default. Any snapshot already there is
moved to an archive/ folder first rather than overwritten, since a rolling
one-month download cannot be re-fetched once Yahoo drops the older days.
"""

import argparse
from pathlib import Path

import yfinance as yf

from src.artifacts import archive_existing
from src.config import INGEST_TICKERS, INTERVAL, INTERVAL_MINUTES, RAW_PRICES_PATH
from src.data_quality import validate_price_quality
from src.logging_config import configure_logging
from src.net_utils import with_retries

logger = configure_logging("ingest")


def main() -> None:
    parser = argparse.ArgumentParser(description="Download five-minute OHLCV data.")
    parser.add_argument("--output", type=Path, default=RAW_PRICES_PATH)
    args = parser.parse_args()

    logger.info(
        "Downloading %d tickers (interval=%s, period=1mo): %s",
        len(INGEST_TICKERS),
        INTERVAL,
        INGEST_TICKERS,
    )

    df = with_retries(
        lambda: yf.download(
            tickers=INGEST_TICKERS,
            period="1mo",
            interval=INTERVAL,
            auto_adjust=True,
        ),
        description="yfinance download",
    )

    if df.empty:
        logger.error("Yahoo Finance returned no data.")
        raise ValueError("Yahoo Finance returned no data.")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    archived = archive_existing(args.output)
    if archived:
        logger.info("Previous snapshot archived to: %s", archived)

    df.to_parquet(args.output, engine="pyarrow", index=True)

    logger.info("Saved: %s", args.output)
    logger.info("Rows: %d", len(df))
    logger.info("Trading dates: %d", df.index.normalize().nunique())
    logger.info("From: %s", df.index.min())
    logger.info("To: %s", df.index.max())
    logger.info("Missing values by column:\n%s", df.isna().sum())

    # Quality-check each ticker's own OHLCV slice (not the whole multi-index
    # frame): a per-ticker duplicate timestamp or interval gap on one symbol
    # shouldn't be masked by another symbol's clean data in the same download.
    for ticker in INGEST_TICKERS:
        ticker_prices = df.xs(ticker, axis=1, level="Ticker").dropna(how="all")
        issues = validate_price_quality(ticker_prices, expected_interval_minutes=INTERVAL_MINUTES)
        if issues:
            logger.warning("%s: data quality issues found: %s", ticker, issues)
        else:
            logger.info("%s: no data quality issues found.", ticker)


if __name__ == "__main__":
    main()
