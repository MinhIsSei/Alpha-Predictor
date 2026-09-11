"""
Feature and target engineering for AAPL 5-minute intraday bars.

Reads the shared, quality-checked snapshot at data/raw/stock-trend.parquet
(see notebooks/exploration.ipynb for the data quality checks) and builds a
per-bar feature/target table for model training (see src/train.py).

Design rules carried over from the exploratory notebooks:
- Chronological order is preserved throughout; nothing is shuffled.
- Any return or target that would span a session gap (overnight/weekend) is
  masked to NaN rather than silently mixing non-adjacent bars.
- Future prices never enter a feature; only the target columns look ahead.
"""

from pathlib import Path

import pandas as pd
import ta  # technical analysis indicators: RSI, MACD, Bollinger Bands, ATR

PROJECT_ROOT = Path(__file__).resolve().parent.parent
RAW_SNAPSHOT_PATH = PROJECT_ROOT / "data" / "raw" / "stock-trend.parquet"
PROCESSED_TRAINING_PATH = PROJECT_ROOT / "data" / "processed" / "aapl_training.parquet"

BAR_MINUTES = 5
TARGET_HORIZON_BARS = 6  # 6 bars * 5 minutes = 30 minutes ahead

FEATURE_COLUMNS = [
    "range_pct", "body_pct",
    "return_5m_pct", "return_15m_pct", "return_30m_pct",
    "volatility_30m", "volume_relative",
    "rsi_14", "macd_diff", "bb_width_pct", "atr_pct",
]
TARGET_COLUMNS = ["target_return_30m_pct", "target_up_30m", "target_time"]


def load_aapl_intraday(raw_snapshot_path: Path = RAW_SNAPSHOT_PATH) -> pd.DataFrame:
    """
    Load the shared 5-minute intraday snapshot and return AAPL's OHLCV columns
    only, with the "Ticker" column level dropped.
    """
    df_saved = pd.read_parquet(raw_snapshot_path)
    return df_saved.xs("AAPL", axis=1, level="Ticker").copy()


def add_candle_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add candle-shape features:
    - range_pct: full high-low range of the candle, relative to the close price.
      Larger values indicate higher intra-bar volatility.
    - body_pct: open-to-close move, relative to the open price.
      Sign shows candle direction (bullish/bearish); magnitude shows conviction.
    """
    df["range_pct"] = (df["High"] - df["Low"]) / df["Close"] * 100
    df["body_pct"] = (df["Close"] - df["Open"]) / df["Open"] * 100
    return df


def masked_pct_return(close: pd.Series, bars: int) -> pd.Series:
    """
    Percentage return over `bars` consecutive 5-minute candles.

    Returns NaN when the lookback window crosses a session gap (e.g. the last
    bars of one day into the first bars of the next), the same way
    exploration.ipynb masks returns to a strict 5-minute spacing -- this stops
    an overnight jump from being reported as a 5/15/30-minute move.
    """
    pct = close.pct_change(periods=bars, fill_method=None) * 100
    expected_start_time = close.index - pd.Timedelta(minutes=BAR_MINUTES * bars)
    actual_start_time = close.index.to_series().shift(bars)
    is_contiguous = actual_start_time == expected_start_time
    return pct.where(is_contiguous)


def add_return_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add gap-masked lagged returns over 5, 15 and 30 minutes."""
    df["return_5m_pct"] = masked_pct_return(df["Close"], bars=1)
    df["return_15m_pct"] = masked_pct_return(df["Close"], bars=3)
    df["return_30m_pct"] = masked_pct_return(df["Close"], bars=6)
    return df


def add_volatility_volume_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add rolling volatility and relative volume features.

    Bars are grouped by calendar day first, so a rolling window never blends
    yesterday's last few candles with today's first few candles.
    """
    trading_day = df.index.normalize()

    # Rolling std-dev of 5-minute returns over the trailing 6 bars (~30 minutes):
    # a short-horizon realized-volatility measure.
    df["volatility_30m"] = (
        df.groupby(trading_day)["return_5m_pct"]
        .rolling(window=6, min_periods=6)
        .std()
        .reset_index(level=0, drop=True)
    )

    # Volume relative to its trailing 20-bar (~100 minute) average for that day.
    # Values above 1 flag unusually heavy trading for that time of day.
    rolling_avg_volume = (
        df.groupby(trading_day)["Volume"]
        .rolling(window=20, min_periods=20)
        .mean()
        .reset_index(level=0, drop=True)
    )
    df["volume_relative"] = df["Volume"] / rolling_avg_volume
    return df


def add_technical_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add RSI, MACD, Bollinger Band width and ATR.

    Computed on the raw, gap-inclusive Close/High/Low series -- this matches
    how these indicators are used in practice (they are running averages of
    price history, not bar-to-bar returns), but it does mean the first
    indicator value after an overnight gap still incorporates the prior
    session's data.
    """
    rsi = ta.momentum.RSIIndicator(close=df["Close"], window=14)
    df["rsi_14"] = rsi.rsi()  # 0-100 momentum oscillator; >70 overbought, <30 oversold

    macd = ta.trend.MACD(close=df["Close"])
    df["macd_diff"] = macd.macd_diff()  # MACD line minus its signal line (momentum shift)

    bb = ta.volatility.BollingerBands(close=df["Close"], window=20, window_dev=2)
    df["bb_width_pct"] = (bb.bollinger_hband() - bb.bollinger_lband()) / df["Close"] * 100

    atr = ta.volatility.AverageTrueRange(
        high=df["High"], low=df["Low"], close=df["Close"], window=14
    )
    df["atr_pct"] = atr.average_true_range() / df["Close"] * 100  # ATR normalized by price
    return df


def add_target(df: pd.DataFrame, horizon_bars: int = TARGET_HORIZON_BARS) -> pd.DataFrame:
    """
    Add the forward-looking target: will price be higher `horizon_bars` * 5
    minutes from now?

    Same gap-safety idea as the feature returns above: the label for a bar is
    only kept when the timestamp ahead is an actual observed bar in the data,
    not a slot that only exists because we skipped over an overnight/weekend gap.
    """
    future_close = df["Close"].shift(-horizon_bars)
    future_time = df.index.to_series().shift(-horizon_bars)
    expected_future_time = df.index + pd.Timedelta(minutes=BAR_MINUTES * horizon_bars)
    horizon_is_valid = future_time == expected_future_time

    # Continuous target: forward return, useful for a regression model.
    df["target_return_30m_pct"] = (
        (future_close - df["Close"]) / df["Close"] * 100
    ).where(horizon_is_valid)

    # Binary target: direction only, useful for a classification model.
    # `.where(horizon_is_valid)` is applied explicitly (not inferred from the NaN
    # in target_return_30m_pct) because `np.nan > 0` evaluates to False, not NaN,
    # and would otherwise mislabel invalid rows as "down" instead of "unknown".
    df["target_up_30m"] = (
        (df["target_return_30m_pct"] > 0).astype("Int64").where(horizon_is_valid)
    )

    # Kept for traceability: the exact timestamp each target refers to.
    df["target_time"] = future_time.where(horizon_is_valid)
    return df


def build_feature_table(df: pd.DataFrame) -> pd.DataFrame:
    """
    Run the full feature/target pipeline on raw AAPL OHLCV bars and return the
    final training table: FEATURE_COLUMNS + TARGET_COLUMNS, with warm-up and
    tail NaN rows dropped.
    """
    df = add_candle_features(df)
    df = add_return_features(df)
    df = add_volatility_volume_features(df)
    df = add_technical_indicators(df)
    df = add_target(df)

    df_features = df[FEATURE_COLUMNS + TARGET_COLUMNS].copy()

    # Drop rows with missing values. NaNs come from two sources only:
    #   1. Indicator/rolling warm-up at the start of the series (not enough
    #      history yet).
    #   2. The last `horizon_bars` rows, which have no future bar to compute a
    #      target from.
    # The DatetimeIndex keeps everything in chronological order throughout, so
    # this never shuffles data and features are never computed from future
    # information.
    return df_features.dropna()


def build_and_save_aapl_features(
    raw_snapshot_path: Path = RAW_SNAPSHOT_PATH,
    output_path: Path = PROCESSED_TRAINING_PATH,
) -> pd.DataFrame:
    """Load AAPL intraday bars, build the feature table, and persist it as Parquet."""
    df = load_aapl_intraday(raw_snapshot_path)
    df_features = build_feature_table(df)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    df_features.to_parquet(output_path)
    return df_features


if __name__ == "__main__":
    features = build_and_save_aapl_features()
    print(f"Saved {features.shape[0]} rows x {features.shape[1]} columns to {PROCESSED_TRAINING_PATH}")
