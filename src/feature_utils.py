import pandas as pd
import ta  # technical analysis indicators: RSI, MACD, Bollinger Bands, ATR


def build_features(prices: pd.DataFrame) -> pd.DataFrame:
    """Create features from single-ticker OHLCV data."""
    result = prices.copy().sort_index()
    timestamps = result.index.to_series()
    trading_day = result.index.normalize()

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

    for periods, minutes in [(1, 5), (3, 15), (6, 30)]:
        valid_gap = timestamps.diff(periods=periods).eq(pd.Timedelta(minutes=minutes))

        result[f"return_{minutes}m_pct"] = (
            result["Close"]
            .pct_change(periods=periods, fill_method=None)
            .mul(100)
            .where(valid_gap)
        )

    # Rolling std-dev of 5-minute returns over the trailing 6 bars (~30 minutes).
    # Grouped by trading day so a window never blends yesterday's last candles
    # with today's first candles.
    result["volatility_30m"] = (
        result.groupby(trading_day)["return_5m_pct"]
        .rolling(window=6, min_periods=6)
        .std()
        .reset_index(level=0, drop=True)
    )

    # Volume relative to its trailing 20-bar (~100 minute) average for that day.
    rolling_avg_volume = (
        result.groupby(trading_day)["Volume"]
        .rolling(window=20, min_periods=20)
        .mean()
        .reset_index(level=0, drop=True)
    )
    result["volume_relative"] = result["Volume"] / rolling_avg_volume

    rsi = ta.momentum.RSIIndicator(close=result["Close"], window=14)
    result["rsi_14"] = rsi.rsi()  # 0-100 momentum oscillator; >70 overbought, <30 oversold

    macd = ta.trend.MACD(close=result["Close"])
    result["macd_diff"] = macd.macd_diff()  # MACD line minus its signal line

    bb = ta.volatility.BollingerBands(close=result["Close"], window=20, window_dev=2)
    result["bb_width_pct"] = (
        (bb.bollinger_hband() - bb.bollinger_lband()) / result["Close"] * 100
    )

    # Leave ATR unavailable until enough historical candles exist.
    atr_window = 14
    result["atr_pct"] = float("nan")

    if len(result) >= atr_window:
        atr = ta.volatility.AverageTrueRange(
            high=result["High"],
            low=result["Low"],
            close=result["Close"],
            window=atr_window,
        )
        atr_values = atr.average_true_range()

        # Mask the library's zero placeholders during the warm-up period.
        atr_values.iloc[: atr_window - 1] = float("nan")
        result["atr_pct"] = atr_values / result["Close"] * 100

    return result
