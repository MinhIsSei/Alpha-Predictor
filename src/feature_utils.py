import pandas as pd

def build_features(prices: pd.DataFrame) -> pd.DataFrame:
    """Create features from single-ticker OHLCV data."""
    result = prices.copy().sort_index()
    timetamps = result.index.to_series()

    result["range_pct"] = (
        (result["High"] - result["Low"])
        / result["Open"]
        * 100
    )

    result["body_pct"] = (
        (result["Close"] - result["Open"])
        / result["Open"]
        * 100
    )

    for periods, minutes in [(1, 5), (3, 15)]:
        valid_gap = timetamps.diff(periods=periods).eq(pd.Timedelta(minutes=minutes))

        result[f"return_{minutes}m_pct"] = (
            result["Close"]
            .pct_change(periods=periods, fill_method=None)
            .mul(100)
            .where(valid_gap)
        )
    return result