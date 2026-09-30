import pandas as pd


def build_target(
    prices: pd.DataFrame,
    horizon_minutes: int,
    periods: int,
) -> pd.DataFrame:
    """Add a same-session future-return target to an OHLCV/feature frame.

    Adds three columns:
    - `future_close_{horizon}m`: the Close price `periods` bars ahead.
    - `target_up_{horizon}m`: 1 if that future Close is greater than the
      current Close, else 0.
    - `target_time`: the timestamp of the future bar used for the target.

    All three are <NA> when the future bar does not exist, is not exactly
    `horizon_minutes` later (a missing/irregular candle), or falls on a
    different trading day — mirroring how `feature_utils.build_features`
    masks `return_*_pct` so a target never bridges a session gap.

    `periods` (bars) and `horizon_minutes` are both required because this
    function does not infer the candle interval from the data: the caller
    already knows it (e.g. 6 bars = 30 minutes at a 5-minute interval).
    """
    result = prices.copy()
    timestamps = result.index.to_series()

    future_close = result["Close"].shift(-periods)
    future_time = timestamps.shift(-periods)

    valid_target = (future_time - timestamps).eq(
        pd.Timedelta(minutes=horizon_minutes)
    ) & future_time.dt.normalize().eq(timestamps.dt.normalize())

    future_col = f"future_close_{horizon_minutes}m"
    target_col = f"target_up_{horizon_minutes}m"

    result[future_col] = future_close.where(valid_target)
    result[target_col] = (future_close > result["Close"]).astype("Int64").where(valid_target)
    result["target_time"] = future_time.where(valid_target)

    return result
