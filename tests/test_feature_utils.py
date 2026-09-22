import unittest

import pandas as pd

from src.feature_utils import build_features


def make_prices(timestamps, closes):
    """Create a small OHLCV dataset with known values."""
    index = pd.DatetimeIndex(timestamps, tz="America/New_York")

    return pd.DataFrame(
        {
            "Open": closes,
            "High": [value + 2 for value in closes],
            "Low": [value - 2 for value in closes],
            "Close": closes,
            "Volume": [1000] * len(closes)
        },
        index=index,
    )

class TestBuildFeatures(unittest.TestCase):
    def test_known_feature_values(self):
        """Check feature values against manually calculated answers."""
        prices = make_prices(
            [
                "2026-09-11 10:00",
                "2026-09-11 10:05",
                "2026-09-11 10:10",
                "2026-09-11 10:15",
            ],
            [100, 102, 104, 106],
        )
        prices.iloc[0,
                    prices.columns.get_loc("Open")] = 99

        result = build_features(prices)

    def test_atr_is_missing_with_13_candles(self):
        """ATR must remain unavailable with fewer than 14 candles."""
        timestamps = pd.date_range(
            "2026-09-11 10:00", periods=13, freq="5min"
        )
        prices = make_prices(timestamps, [100.0] * 13)

        result = build_features(prices)

        self.assertEqual(len(result), 13)
        self.assertTrue(result["atr_pct"].isna().all())

    def test_atr_becomes_available_at_14_candles(self):
        """ATR must first become available on the 14th candle."""
        timestamps = pd.date_range(
            "2026-09-11 10:00", periods=14, freq="5min"
        )
        prices = make_prices(timestamps, [100.0] * 14)

        result = build_features(prices)

        self.assertTrue(result["atr_pct"].iloc[:13].isna().all())

        # Each candle has a true range of 4 and a close of 100.
        # Expected ATR percentage: 4 / 100 * 100 = 4.
        self.assertAlmostEqual(result["atr_pct"].iloc[13], 4.0)