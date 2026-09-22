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

    def test_returns_do_not_cross_sessions(self):
        """Returns must not bridge the overnight gap."""
        prices = make_prices(
            [
                "2026-09-10 15:30",
                "2026-09-10 15:35",
                "2026-09-10 15:40",
                "2026-09-10 15:45",
                "2026-09-10 15:50",
                "2026-09-10 15:55",
                "2026-09-11 09:30",
                "2026-09-11 09:35", 
            ],
            [100.0] * 6 + [110.0, 112.0],
        )

        result = build_features(prices)

        # The first candle of the new session has no intraday return.
        for minutes in (5, 15, 30):
            with self.subTest(minutes=minutes):
                self.assertTrue(
                    pd.isna(result[f"return_{minutes}m_pct"].iloc[6])
                )

        # The next candle has a valid five-minute return.
        self.assertAlmostEqual(
            result["return_5m_pct"].iloc[7],
            (112.0 / 110.0 - 1) * 100,
        )

    def test_returns_respect_missing_candles(self):
        """Returns must match their stated time intervals."""
        prices = make_prices(
            [
                "2026-09-11 10:00",
                "2026-09-11 10:05",
                "2026-09-11 10:15",
                "2026-09-11 10:20",
            ],
            [100.0, 102.0, 106.0, 108.0],
        )

        result = build_features(prices)

        # The previous row is ten minutes earlier, not five
        self.assertTrue(pd.isna(result["return_5m_pct"].iloc[2]))

        # Three rows back spans every twenty minutes, not fifteen
        self.assertTrue(pd.isna(result["return_15m_pct"].iloc[3]))

        # A valid five-minute interval resumes after the gap
        self.assertAlmostEqual(
            result["return_5m_pct"].iloc[3], 
            (108.0 / 106.0 - 1) * 100,
        )

    def test_future_candles_do_not_change_past_features(self):
        """Appending future candles must not change past features."""
        timestamps = pd.date_range(
            "2026-09-11 09:30", periods=60, freq="5min"
        )
        closes = [
            100.0 + i * 0.1 + (i % 7 - 3) * 0.4
            for i in range(60)
        ]

        # Add a large future price change after the comparision boundary
        closes[40:] = [value + 20.0 for value in closes[40:]]
        prices = make_prices(timestamps, closes)
        prices["Volume"] = [1000 + (i % 9) * 100 for i in range(60)]

        feature_columns = [
            "range_pct",
            "body_pct",
            "return_5m_pct",
            "return_15m_pct",
            "return_30m_pct",
            "volatility_30m",
            "volume_relative",
            "rsi_14",
            "macd_diff",
            "bb_width_pct",
            "atr_pct"
        ]

        past_features = build_features(prices.iloc[:40].copy())
        full_features = build_features(prices)

        # Ensure every feature has values beyond its warm-up period
        self.assertTrue(
            past_features[feature_columns].iloc[-1].notna().all()
        )

        pd.testing.assert_frame_equal(
            past_features[feature_columns],
            full_features[feature_columns].iloc[:40],
            check_exact=True,
        )