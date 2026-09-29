from pathlib import Path
import yfinance as yf

from src.data_quality import validate_price_quality
from src.logging_config import configure_logging
from src.net_utils import with_retries

logger = configure_logging("ingest")

project_root = Path(__file__).resolve().parent.parent
raw_dir = project_root / "data" / "raw"
raw_dir.mkdir(parents=True, exist_ok=True)

tickers = ["AAPL", "MSFT", "NVDA", "GOOGL", "AMZN"]

logger.info("Downloading %d tickers (interval=5m, period=1mo): %s", len(tickers), tickers)

df = with_retries(
    lambda: yf.download(
        tickers=tickers,
        period="1mo",
        interval="5m",
        auto_adjust=True,
    ),
    description="yfinance download",
)

if df.empty:
    logger.error("Yahoo Finance returned no data.")
    raise ValueError("Yahoo Finance returned no data.")

file_path = raw_dir / "stock-trend_1mo.parquet"
df.to_parquet(file_path, engine="pyarrow", index=True)

logger.info("Saved: %s", file_path)
logger.info("Rows: %d", len(df))
logger.info("Trading dates: %d", df.index.normalize().nunique())
logger.info("From: %s", df.index.min())
logger.info("To: %s", df.index.max())
logger.info("Missing values by column:\n%s", df.isna().sum())

# Quality-check each ticker's own OHLCV slice (not the whole multi-index
# frame): a per-ticker duplicate timestamp or interval gap on one symbol
# shouldn't be masked by another symbol's clean data in the same download.
for ticker in tickers:
    ticker_prices = df.xs(ticker, axis=1, level="Ticker").dropna(how="all")
    issues = validate_price_quality(ticker_prices, expected_interval_minutes=5)
    if issues:
        logger.warning("%s: data quality issues found: %s", ticker, issues)
    else:
        logger.info("%s: no data quality issues found.", ticker)
