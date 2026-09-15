from pathlib import Path
import pandas as pd
import joblib
import pandas_market_calendars as mcal
import argparse
import yfinance as yf
import sqlite3

from feature_utils import build_features
from logging_config import configure_logging
from net_utils import with_retries

logger = configure_logging("predict")

parser = argparse.ArgumentParser(
    description="Run predictions in live or historical replay mode."
)

parser.add_argument(
    "--mode",
    choices=["replay", "live"],
    default="replay",
)

parser.add_argument(
    "--as-of",
    help="Replay time in New York timezone, e.g. '2026-09-11 14:05:00'.",
)

args = parser.parse_args()

if args.mode == "replay" and args.as_of is None:
    parser.error("--as-of is required in replay mode.")

if args.mode == "live" and args.as_of is not None:
    parser.error("--as-of is only supported in replay mode.")

project_root = Path(__file__).resolve().parent.parent


def validate_price_quality(prices: pd.DataFrame) -> list[str]:
    """Return a list of data-quality problems found in raw OHLCV rows.

    Mirrors the checks notebooks/exploration.ipynb runs manually on the raw
    snapshot, so predict.py never feeds obviously broken candles (e.g. a
    High below Low from a bad tick) into the model.
    """
    issues = []

    price_cols = ["Open", "High", "Low", "Close"]
    if (prices[price_cols] <= 0).any().any():
        issues.append("non-positive Open/High/Low/Close value")

    if (prices["High"] < prices["Low"]).any():
        issues.append("High below Low on at least one candle")

    if (prices["Close"] > prices["High"]).any() or (prices["Close"] < prices["Low"]).any():
        issues.append("Close outside [Low, High] on at least one candle")

    if (prices["Open"] > prices["High"]).any() or (prices["Open"] < prices["Low"]).any():
        issues.append("Open outside [Low, High] on at least one candle")

    if (prices["Volume"] < 0).any():
        issues.append("negative Volume")

    return issues


# 1. Load the model and feature list
artifact = joblib.load(
    project_root / "models" / "aapl_logistic_v2.joblib"
)

model = artifact["pipeline"]
feature_cols = artifact["feature_cols"]

# 2. Load historical data for replay or fetch recent data for live mode.
ticker = artifact["ticker"]

if args.mode == "replay":
    now = pd.Timestamp(args.as_of)

    if now.tzinfo is None:
        now = now.tz_localize("America/New_York")
    else:
        now = now.tz_convert("America/New_York")

    df = pd.read_parquet(
        project_root / "data" / "raw" / "stock-trend_1mo.parquet"
    )

    prices = df.xs(
        ticker, axis=1, level="Ticker"
    ).copy().sort_index()

else:
    # Fetch recent candles without overwriting the training snapshot.
    prices = with_retries(
        lambda: yf.Ticker(ticker).history(
            period="5d",
            interval=artifact["interval"],
            auto_adjust=True,
            prepost=False,
        ).sort_index(),
        description=f"yfinance history fetch for {ticker}",
    )

    # Capture the evaluation time after the download completes.
    now = pd.Timestamp.now(tz="America/New_York")

if prices.empty:
    logger.info("Skipped: no price data is available.")
    raise SystemExit(0)

if prices.index.tz is None:
    raise ValueError("Price timestamps must include a timezone.")

logger.info("Mode: %s", args.mode)
logger.info("Evaluation time: %s", now)

quality_issues = validate_price_quality(prices)
if quality_issues:
    logger.warning("Skipped: raw price data failed quality checks: %s", quality_issues)
    raise SystemExit(0)

prices = prices[
    prices.index + pd.Timedelta(minutes=5) <= now
]

# 3. Generate the exact features used during training.
prices = build_features(prices)

# 4. Check and predict for the last candle
if prices.empty:
    raise ValueError("No completed candle is available.")

# The timestamp in the data represents the START time of the candle.
candle_start = prices.index[-1].tz_convert("America/New_York")
candle_end = candle_start + pd.Timedelta(minutes=5)

prediction_end = candle_end + pd.Timedelta(
    minutes=artifact["horizon_minutes"]
)

# The schedule lookup is based on the simulation date, not the system date.
evaluation_time = now.tz_convert("America/New_York")
session_date = evaluation_time.date()

calendar = mcal.get_calendar("NASDAQ")
schedule = calendar.schedule(
    start_date=session_date,
    end_date=session_date,
    tz="America/New_York",
)

if schedule.empty:
    logger.info("Skipped: no trading session on this date.")
    raise SystemExit(0)

session_open = schedule.iloc[0]["market_open"]
session_close = schedule.iloc[0]["market_close"]

# Check trading hours before checking the latest candle.
if not (session_open <= evaluation_time < session_close):
    logger.info("Skipped: evaluation time is outside trading hours.")
    raise SystemExit(0)

# Ensure the latest candle belongs to the current session (not a leftover
# candle from the previous session or outside trading hours).
if candle_start < session_open or candle_end > session_close:
    logger.info("Skipped: latest candle is outside this session.")
    raise SystemExit(0)

if prediction_end > session_close:
    logger.info(
        "Skipped: insufficient time remaining in session (prediction_end=%s, session_close=%s).",
        prediction_end, session_close,
    )
    raise SystemExit(0)

logger.info("Session open: %s", session_open)
logger.info("Session close: %s", session_close)

# Time elapsed from the close of the last candle to the evaluation time
data_age = evaluation_time - candle_end
max_data_age = pd.Timedelta(minutes=5)

if data_age > max_data_age:
    logger.info(
        "Skipped: latest completed candle is stale (candle_end=%s, evaluation_time=%s, age=%s).",
        candle_end, evaluation_time, data_age,
    )
    raise SystemExit(0)

# Predict only if the check is passed
X_latest = prices[feature_cols].tail(1)

if X_latest.isna().any().any():
    # Most commonly this is volume_relative or one of the rolling/indicator
    # features still warming up early in the session (e.g. volume_relative
    # needs 20 same-day bars, ~100 minutes after the open) -- not an error,
    # just too early in the day for this candle to have every feature yet.
    missing_cols = X_latest.columns[X_latest.isna().any()].tolist()
    logger.info(
        "Skipped: latest candle is missing required features: %s.",
        missing_cols,
    )
    raise SystemExit(0)

prediction = int(model.predict(X_latest)[0])

up_index = list(model.classes_).index(1)
probability_up = model.predict_proba(X_latest)[0, up_index]

logger.info("Candle start: %s", X_latest.index[0])
logger.info("Prediction: %s", "Up" if prediction == 1 else "Not up")
logger.info("Model probability of Up: %.2f%%", probability_up * 100)

# Store successful predictions in a local SQLite database.
prediction_dir = project_root / "data" / "predictions"
prediction_dir.mkdir(parents=True, exist_ok=True)

db_path = prediction_dir / "predictions.sqlite"

# This identifier must change whenever a new model is released.
model_version = "aapl_logistic_v2"

# Store timestamps consistently in UTC.
candle_start_utc = candle_start.tz_convert("UTC").isoformat()
candle_end_utc = candle_end.tz_convert("UTC").isoformat()
evaluation_time_utc = evaluation_time.tz_convert("UTC").isoformat()
prediction_end_utc = prediction_end.tz_convert("UTC").isoformat()

# Raw OHLCV of the candle the prediction was based on, kept alongside the
# prediction so it can be audited later without re-downloading from Yahoo
# Finance (which may no longer have the same intraday history available).
reference_row = prices.loc[X_latest.index[0]]
reference_open = float(reference_row["Open"])
reference_high = float(reference_row["High"])
reference_low = float(reference_row["Low"])
reference_close = float(reference_row["Close"])
reference_volume = float(reference_row["Volume"])

with sqlite3.connect(db_path) as connection:
    connection.execute("""
        CREATE TABLE IF NOT EXISTS predictions (
            ticker TEXT NOT NULL,
            interval TEXT NOT NULL,
            candle_start TEXT NOT NULL,
            candle_end TEXT NOT NULL,
            evaluation_time TEXT NOT NULL,
            prediction_end TEXT NOT NULL,
            mode TEXT NOT NULL,
            model_version TEXT NOT NULL,
            horizon_minutes INTEGER NOT NULL,
            predicted_class INTEGER NOT NULL,
            probability_up REAL NOT NULL,
            reference_open REAL NOT NULL,
            reference_high REAL NOT NULL,
            reference_low REAL NOT NULL,
            reference_close REAL NOT NULL,
            reference_volume REAL NOT NULL,
            UNIQUE (
                ticker,
                interval,
                candle_start,
                mode,
                model_version,
                horizon_minutes
            )
        )
    """)

    cursor = connection.execute("""
        INSERT INTO predictions (
            ticker,
            interval,
            candle_start,
            candle_end,
            evaluation_time,
            prediction_end,
            mode,
            model_version,
            horizon_minutes,
            predicted_class,
            probability_up,
            reference_open,
            reference_high,
            reference_low,
            reference_close,
            reference_volume
        )
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ON CONFLICT (
            ticker,
            interval,
            candle_start,
            mode,
            model_version,
            horizon_minutes
        ) DO NOTHING
    """, (
        artifact["ticker"],
        artifact["interval"],
        candle_start_utc,
        candle_end_utc,
        evaluation_time_utc,
        prediction_end_utc,
        args.mode,
        model_version,
        int(artifact["horizon_minutes"]),
        prediction,
        float(probability_up),
        reference_open,
        reference_high,
        reference_low,
        reference_close,
        reference_volume,
    ))

    if cursor.rowcount == 1:
        logger.info("Prediction saved: %s", db_path)
    else:
        logger.info("Prediction already exists; no duplicate was saved.")

    total = connection.execute(
        "SELECT COUNT(*) FROM predictions"
    ).fetchone()[0]

    logger.info("Total stored predictions: %d", total)
