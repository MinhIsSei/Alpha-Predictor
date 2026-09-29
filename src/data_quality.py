import pandas as pd


def validate_price_quality(
    prices: pd.DataFrame,
    expected_interval_minutes: int | None = None,
) -> list[str]:
    """Return a list of data-quality problems found in raw OHLCV rows.

    Mirrors the checks notebooks/exploration.ipynb runs manually on the raw
    snapshot, so no pipeline stage feeds obviously broken candles (e.g. a
    High below Low from a bad tick, or a duplicated/misaligned timestamp)
    into feature generation or the model.

    `expected_interval_minutes` is optional: when given, candles are also
    checked for gaps shorter than that interval within the same trading day
    (a longer gap is normal at a session boundary, so only short gaps are
    flagged).
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

    if prices.index.has_duplicates:
        issues.append("duplicate timestamps")

    if expected_interval_minutes is not None and len(prices) > 1:
        trading_day = prices.index.normalize()
        gaps = prices.index.to_series().groupby(trading_day).diff().dropna()
        expected = pd.Timedelta(minutes=expected_interval_minutes)

        if (gaps < expected).any():
            issues.append(
                f"candle interval shorter than {expected_interval_minutes} "
                "minutes within a trading day"
            )

    return issues
