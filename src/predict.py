from contextlib import closing
from pathlib import Path
import pandas as pd
import joblib
import pandas_market_calendars as mcal
import argparse
import yfinance as yf
import sqlite3

from src.data_quality import validate_price_quality
from src.feature_utils import build_features
from src.logging_config import configure_logging
from src.net_utils import with_retries
from src.prediction_rules import target_fits_session

logger = configure_logging("predict")

project_root = Path(__file__).resolve().parent.parent

INTERVAL_MINUTES = 5
MAX_DATA_AGE = pd.Timedelta(minutes=INTERVAL_MINUTES)


def filter_completed_candles(
    prices: pd.DataFrame,
    now: pd.Timestamp,
    interval_minutes: int = INTERVAL_MINUTES,
) -> pd.DataFrame:
    """Keep only candles whose bar has fully closed by `now`."""
    return prices[prices.index + pd.Timedelta(minutes=interval_minutes) <= now]


def check_session_timing(
    candle_start: pd.Timestamp,
    candle_end: pd.Timestamp,
    prediction_end: pd.Timestamp,
    session_open: pd.Timestamp,
    session_close: pd.Timestamp,
    evaluation_time: pd.Timestamp,
    max_data_age: pd.Timedelta = MAX_DATA_AGE,
) -> str | None:
    """Return a skip reason if the latest candle isn't safe to predict on.

    Checks, in order: the candle belongs to the current session, its
    30-minute target ends within the session, and the candle isn't stale
    relative to the evaluation time. Returns None when every check passes.
    """
    if candle_start < session_open or candle_end > session_close:
        return "latest candle is outside this session."

    if not target_fits_session(prediction_end, session_close):
        return (
            "insufficient time remaining in session "
            f"(prediction_end={prediction_end}, session_close={session_close})."
        )

    data_age = evaluation_time - candle_end
    if data_age > max_data_age:
        return (
            "latest completed candle is stale "
            f"(candle_end={candle_end}, evaluation_time={evaluation_time}, age={data_age})."
        )

    return None


def insert_prediction(db_path: Path, record: dict) -> bool:
    """Insert a prediction row, skipping silently on a duplicate key.

    Returns True if a new row was inserted, False if it already existed.
    """
    with closing(sqlite3.connect(db_path)) as connection, connection:
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
                ticker, interval, candle_start, candle_end, evaluation_time,
                prediction_end, mode, model_version, horizon_minutes,
                predicted_class, probability_up,
                reference_open, reference_high, reference_low,
                reference_close, reference_volume
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT (
                ticker, interval, candle_start, mode, model_version, horizon_minutes
            ) DO NOTHING
        """, (
            record["ticker"],
            record["interval"],
            record["candle_start"],
            record["candle_end"],
            record["evaluation_time"],
            record["prediction_end"],
            record["mode"],
            record["model_version"],
            record["horizon_minutes"],
            record["predicted_class"],
            record["probability_up"],
            record["reference_open"],
            record["reference_high"],
            record["reference_low"],
            record["reference_close"],
            record["reference_volume"],
        ))

        inserted = cursor.rowcount == 1

        total = connection.execute(
            "SELECT COUNT(*) FROM predictions"
        ).fetchone()[0]

    if inserted:
        logger.info("Prediction saved: %s", db_path)
    else:
        logger.info("Prediction already exists; no duplicate was saved.")

    logger.info("Total stored predictions: %d", total)

    return inserted


def load_prices(mode: str, as_of: str | None, artifact: dict) -> tuple[pd.DataFrame, pd.Timestamp]:
    """Load candles for replay or live mode and the evaluation timestamp."""
    ticker = artifact["ticker"]

    if mode == "replay":
        now = pd.Timestamp(as_of)
        now = (
            now.tz_localize("America/New_York")
            if now.tzinfo is None
            else now.tz_convert("America/New_York")
        )

        df = pd.read_parquet(
            project_root / "data" / "raw" / "stock-trend_1mo.parquet"
        )
        prices = df.xs(ticker, axis=1, level="Ticker").copy().sort_index()

        return prices, now

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

    return prices, now


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run predictions in live or historical replay mode."
    )
    parser.add_argument("--mode", choices=["replay", "live"], default="replay")
    parser.add_argument(
        "--as-of",
        help="Replay time in New York timezone, e.g. '2026-09-11 14:05:00'.",
    )
    args = parser.parse_args()

    if args.mode == "replay" and args.as_of is None:
        parser.error("--as-of is required in replay mode.")
    if args.mode == "live" and args.as_of is not None:
        parser.error("--as-of is only supported in replay mode.")

    # 1. Load the model and feature list
    artifact = joblib.load(project_root / "models" / "aapl_logistic_v2.joblib")
    model = artifact["pipeline"]
    feature_cols = artifact["feature_cols"]

    # 2. Load historical data for replay or fetch recent data for live mode.
    prices, now = load_prices(args.mode, args.as_of, artifact)

    if prices.empty:
        logger.info("Skipped: no price data is available.")
        raise SystemExit(0)

    if prices.index.tz is None:
        raise ValueError("Price timestamps must include a timezone.")

    logger.info("Mode: %s", args.mode)
    logger.info("Evaluation time: %s", now)

    quality_issues = validate_price_quality(
        prices, expected_interval_minutes=INTERVAL_MINUTES
    )
    if quality_issues:
        logger.warning("Skipped: raw price data failed quality checks: %s", quality_issues)
        raise SystemExit(0)

    prices = filter_completed_candles(prices, now)

    # 3. Generate the exact features used during training.
    prices = build_features(prices)

    # 4. Check and predict for the last candle
    if prices.empty:
        raise ValueError("No completed candle is available.")

    # The timestamp in the data represents the START time of the candle.
    candle_start = prices.index[-1].tz_convert("America/New_York")
    candle_end = candle_start + pd.Timedelta(minutes=INTERVAL_MINUTES)
    prediction_end = candle_end + pd.Timedelta(minutes=artifact["horizon_minutes"])

    # The schedule lookup is based on the simulation date, not the system date.
    evaluation_time = now.tz_convert("America/New_York")
    session_date = evaluation_time.date()

    calendar = mcal.get_calendar("NASDAQ")
    schedule = calendar.schedule(
        start_date=session_date, end_date=session_date, tz="America/New_York",
    )

    if schedule.empty:
        logger.info("Skipped: no trading session on this date.")
        raise SystemExit(0)

    session_open = schedule.iloc[0]["market_open"]
    session_close = schedule.iloc[0]["market_close"]

    if not (session_open <= evaluation_time < session_close):
        logger.info("Skipped: evaluation time is outside trading hours.")
        raise SystemExit(0)

    logger.info("Session open: %s", session_open)
    logger.info("Session close: %s", session_close)

    skip_reason = check_session_timing(
        candle_start=candle_start,
        candle_end=candle_end,
        prediction_end=prediction_end,
        session_open=session_open,
        session_close=session_close,
        evaluation_time=evaluation_time,
    )
    if skip_reason is not None:
        logger.info("Skipped: %s", skip_reason)
        raise SystemExit(0)

    X_latest = prices[feature_cols].tail(1)

    if X_latest.isna().any().any():
        # Most commonly this is volume_relative or one of the rolling/indicator
        # features still warming up early in the session (e.g. volume_relative
        # needs 20 same-day bars, ~100 minutes after the open) -- not an error,
        # just too early in the day for this candle to have every feature yet.
        missing_cols = X_latest.columns[X_latest.isna().any()].tolist()
        logger.info("Skipped: latest candle is missing required features: %s.", missing_cols)
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

    # Raw OHLCV of the candle the prediction was based on, kept alongside the
    # prediction so it can be audited later without re-downloading from Yahoo
    # Finance (which may no longer have the same intraday history available).
    reference_row = prices.loc[X_latest.index[0]]

    record = {
        "ticker": artifact["ticker"],
        "interval": artifact["interval"],
        "candle_start": candle_start.tz_convert("UTC").isoformat(),
        "candle_end": candle_end.tz_convert("UTC").isoformat(),
        "evaluation_time": evaluation_time.tz_convert("UTC").isoformat(),
        "prediction_end": prediction_end.tz_convert("UTC").isoformat(),
        "mode": args.mode,
        # This identifier must change whenever a new model is released.
        "model_version": "aapl_logistic_v2",
        "horizon_minutes": int(artifact["horizon_minutes"]),
        "predicted_class": prediction,
        "probability_up": float(probability_up),
        "reference_open": float(reference_row["Open"]),
        "reference_high": float(reference_row["High"]),
        "reference_low": float(reference_row["Low"]),
        "reference_close": float(reference_row["Close"]),
        "reference_volume": float(reference_row["Volume"]),
    }

    insert_prediction(db_path, record)


if __name__ == "__main__":
    main()
