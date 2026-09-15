from pathlib import Path
import pandas as pd
import joblib
import pandas_market_calendars as mcal
import argparse
import yfinance as yf
import sqlite3

from feature_utils import build_features

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

# 1. Load the model and feature list
artifact = joblib.load(
    project_root / "models" / "aapl_logistic_v1.joblib"
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
    prices = yf.Ticker(ticker).history(
        period="5d",
        interval=artifact["interval"],
        auto_adjust=True,
        prepost=False,
    ).sort_index()

    # Capture the evaluation time after the download completes.
    now = pd.Timestamp.now(tz="America/New_York")

if prices.empty:
    print("Skipped: no price data is available.")
    raise SystemExit(0)

if prices.index.tz is None:
    raise ValueError("Price timestamps must include a timezone.")

print("Mode:", args.mode)
print("Evaluation time:", now)

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
    print("Skipped: no trading session on this date.")
    raise SystemExit(0)

session_open = schedule.iloc[0]["market_open"]
session_close = schedule.iloc[0]["market_close"]

# Check trading hours before checking the latest candle.
if not (session_open <= evaluation_time < session_close):
    print("Skipped: evaluation time is outside trading hours.")
    raise SystemExit(0)

# Ensure the latest candle belongs to the current session (not a leftover
# candle from the previous session or outside trading hours).
if candle_start < session_open or candle_end > session_close:
    print("Skipped: latest candle is outside this session.")
    raise SystemExit(0)

if prediction_end > session_close:
    print("Skipped: insufficient time remaining in session.")
    print("Prediction end:", prediction_end)
    print("Session close:", session_close)
    raise SystemExit(0)

print("Session open:", session_open)
print("Session close:", session_close)

# Time elapsed from the close of the last candle to the evaluation time
data_age = evaluation_time - candle_end
max_data_age = pd.Timedelta(minutes=5)

if data_age > max_data_age:
    print("Skipped: latest completed candle is stale.")
    print("Latest candle end:", candle_end)
    print("Evaluation time:", evaluation_time)
    print("Data age:", data_age)
    raise SystemExit(0)

# Predict only if the check is passed
X_latest = prices[feature_cols].tail(1)

if X_latest.isna().any().any():
    raise ValueError("The latest candle has missing features.")

prediction = int(model.predict(X_latest)[0])

up_index = list(model.classes_).index(1)
probability_up = model.predict_proba(X_latest)[0, up_index]

print("Candle start:", X_latest.index[0])
print("Prediction:", "Up" if prediction == 1 else "Not up")
print(f"Model probability of Up: {probability_up:.2%}")

# Store successful predictions in a local SQLite database.
prediction_dir = project_root / "data" / "predictions"
prediction_dir.mkdir(parents=True, exist_ok=True)

db_path = prediction_dir / "predictions.sqlite"

# This identifier must change whenever a new model is released.
model_version = "aapl_logistic_v1"

# Store timestamps consistently in UTC.
candle_start_utc = candle_start.tz_convert("UTC").isoformat()
candle_end_utc = candle_end.tz_convert("UTC").isoformat()
evaluation_time_utc = evaluation_time.tz_convert("UTC").isoformat()
prediction_end_utc = prediction_end.tz_convert("UTC").isoformat()

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
            probability_up
        )
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
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
    ))

    if cursor.rowcount == 1:
        print("Prediction saved:", db_path)
    else:
        print("Prediction already exists; no duplicate was saved.")

    total = connection.execute(
        "SELECT COUNT(*) FROM predictions"
    ).fetchone()[0]

    print("Total stored predictions:", total)