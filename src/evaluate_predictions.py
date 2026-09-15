from pathlib import Path
import sqlite3
import pandas as pd
import argparse
import yfinance as yf

def save_outcome(db_path, row, reference_close, target_close,
                 actual_class, is_correct):
    """Save an outcome once without changing the original prediction."""
    with sqlite3.connect(db_path) as connection:
        connection.execute("""
            CREATE TABLE IF NOT EXISTS prediction_outcomes (
                ticker TEXT NOT NULL,
                interval TEXT NOT NULL,
                candle_start TEXT NOT NULL,
                mode TEXT NOT NULL,
                model_version TEXT NOT NULL,
                horizon_minutes INTEGER NOT NULL,
                reference_close REAL NOT NULL,
                target_close REAL NOT NULL,
                actual_class INTEGER NOT NULL,
                is_correct INTEGER NOT NULL,
                evaluated_at TEXT NOT NULL,
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
            INSERT INTO prediction_outcomes (
                ticker,
                interval,
                candle_start,
                mode,
                model_version,
                horizon_minutes,
                reference_close,
                target_close,
                actual_class,
                is_correct,
                evaluated_at
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
            row["ticker"],
            row["interval"],
            row["candle_start"],
            row["mode"],
            row["model_version"],
            int(row["horizon_minutes"]),
            float(reference_close),
            float(target_close),
            int(actual_class),
            int(is_correct),
            pd.Timestamp.now(tz="UTC").isoformat(),
        ))

        if cursor.rowcount == 1:
            print("Outcome saved.")
        else:
            print("Outcome already exists; no duplicate was saved.")

        total = connection.execute(
            "SELECT COUNT(*) FROM prediction_outcomes"
        ).fetchone()[0]

        print("Total stored outcomes:", total)

project_root = Path(__file__).resolve().parent.parent
db_path = project_root / "data" / "predictions" / "predictions.sqlite"
price_path = project_root / "data" / "raw" / "stock-trend_1mo.parquet"

if not db_path.is_file():
    raise FileNotFoundError("Prediction database does not exist.")

parser = argparse.ArgumentParser(
    description="Evaluate pending replay or live predictions."
)
parser.add_argument(
    "--mode",
    choices=["replay", "live"],
    default="replay",
)
args = parser.parse_args()

# Select predictions that do not already have an outcome.
with sqlite3.connect(db_path) as connection:
    outcomes_exists = connection.execute("""
        SELECT 1 FROM sqlite_master
        WHERE type = 'table' AND name = 'prediction_outcomes'
    """).fetchone()

    if outcomes_exists:
        query = """
            SELECT p.*
            FROM predictions AS p
            WHERE p.mode = ?
              AND NOT EXISTS (
                  SELECT 1
                  FROM prediction_outcomes AS o
                  WHERE o.ticker = p.ticker
                    AND o.interval = p.interval
                    AND o.candle_start = p.candle_start
                    AND o.mode = p.mode
                    AND o.model_version = p.model_version
                    AND o.horizon_minutes = p.horizon_minutes
              )
        """
    else:
        query = "SELECT * FROM predictions WHERE mode = ?"

    predictions = pd.read_sql_query(
        query, connection, params=(args.mode,)
    )

print("Mode:", args.mode)
print("Predictions without outcomes:", len(predictions))

if predictions.empty:
    print("Nothing to evaluate.")
    raise SystemExit(0)

# Use a fixed cutoff for this evaluation run.
evaluation_time = pd.Timestamp.now(tz="UTC")

if args.mode == "live":
    target_times = pd.to_datetime(
        predictions["prediction_end"], utc=True
    )
    matured = target_times <= evaluation_time

    print("Waiting for target close:", int((~matured).sum()))
    predictions = predictions.loc[matured].copy()

    if predictions.empty:
        print("No predictions are ready for evaluation.")
        raise SystemExit(0)

historical_prices = (
    pd.read_parquet(price_path)
    if args.mode == "replay"
    else None
)

# Download once per ticker and interval, not once per prediction.
for (ticker, interval), group in predictions.groupby(
    ["ticker", "interval"]
):
    if interval != "5m":
        print(f"Skipped {ticker}: unsupported candle interval.")
        continue

    if args.mode == "replay":
        close = historical_prices[("Close", ticker)].copy()

    else:
        # Request the dates needed by pending predictions.
        first_time = pd.to_datetime(
            group["candle_start"], utc=True
        ).min()
        last_time = pd.to_datetime(
            group["prediction_end"], utc=True
        ).max()

        start_date = first_time.tz_convert(
            "America/New_York"
        ).date()

        end_date = (
            last_time.tz_convert("America/New_York").normalize()
            + pd.Timedelta(days=1)
        ).date()

        try:
            downloaded = yf.Ticker(ticker).history(
                start=start_date.isoformat(),
                end=end_date.isoformat(),
                interval=interval,
                auto_adjust=True,
                prepost=False,
            )
        except Exception as exc:
            print(f"Pending {ticker}: download failed: {exc}")
            continue

        if downloaded.empty:
            print(f"Pending {ticker}: no price data returned.")
            continue

        close = downloaded["Close"].copy()

    if close.index.tz is None:
        print(f"Skipped {ticker}: timestamps have no timezone.")
        continue

    close.index = close.index.tz_convert("UTC")
    close = close.sort_index()

    if close.index.has_duplicates:
        print(f"Skipped {ticker}: duplicate price timestamps.")
        continue

    if args.mode == "live":
        # Exclude candles that had not closed at the evaluation cutoff.
        close = close[
            close.index + pd.Timedelta(minutes=5) <= evaluation_time
        ]

    for _, row in group.iterrows():
        candle_start = pd.to_datetime(
            row["candle_start"], utc=True
        )
        prediction_end = pd.to_datetime(
            row["prediction_end"], utc=True
        )

        # The target price belongs to the candle ending at prediction_end.
        target_candle_start = (
            prediction_end - pd.Timedelta(minutes=5)
        )

        required_times = [candle_start, target_candle_start]

        if any(timestamp not in close.index for timestamp in required_times):
            print(f"Pending {ticker} {candle_start}: required candle unavailable.")
            continue

        reference_close = close.loc[candle_start]
        target_close = close.loc[target_candle_start]

        if (
            pd.isna(reference_close)
            or pd.isna(target_close)
            or reference_close <= 0
            or target_close <= 0
        ):
            print(f"Pending {ticker} {candle_start}: invalid or missing price.")
            continue

        actual_class = int(target_close > reference_close)
        predicted_class = int(row["predicted_class"])
        is_correct = predicted_class == actual_class

        save_outcome(
            db_path=db_path,
            row=row,
            reference_close=reference_close,
            target_close=target_close,
            actual_class=actual_class,
            is_correct=is_correct,
        )

        print("\nTicker:", ticker)
        print("Candle start:", candle_start)
        print(f"Reference close: {reference_close:.4f}")
        print(f"Target close: {target_close:.4f}")
        print("Predicted:", "Up" if predicted_class == 1 else "Not up")
        print("Actual:", "Up" if actual_class == 1 else "Not up")
        print("Correct:", is_correct)