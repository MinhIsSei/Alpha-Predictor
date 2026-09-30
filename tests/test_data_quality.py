import unittest

import pandas as pd

from src.data_quality import validate_price_quality


def make_prices(timestamps, opens, highs, lows, closes, volumes):
    index = pd.DatetimeIndex(timestamps, tz="America/New_York")
    return pd.DataFrame(
        {
            "Open": opens,
            "High": highs,
            "Low": lows,
            "Close": closes,
            "Volume": volumes,
        },
        index=index,
    )


class TestValidatePriceQuality(unittest.TestCase):
    def test_clean_data_has_no_issues(self):
        prices = make_prices(
            pd.date_range("2026-09-11 10:00", periods=3, freq="5min"),
            [100, 101, 102],
            [102, 103, 104],
            [99, 100, 101],
            [101, 102, 103],
            [1000, 1100, 1200],
        )

        self.assertEqual(validate_price_quality(prices), [])

    def test_non_positive_price_is_flagged(self):
        prices = make_prices(
            ["2026-09-11 10:00"],
            [0],
            [1],
            [-1],
            [0.5],
            [1000],
        )

        issues = validate_price_quality(prices)
        self.assertIn("non-positive Open/High/Low/Close value", issues)

    def test_high_below_low_is_flagged(self):
        prices = make_prices(
            ["2026-09-11 10:00"],
            [100],
            [98],
            [99],
            [98.5],
            [1000],
        )

        issues = validate_price_quality(prices)
        self.assertIn("High below Low on at least one candle", issues)

    def test_close_outside_range_is_flagged(self):
        prices = make_prices(
            ["2026-09-11 10:00"],
            [100],
            [102],
            [99],
            [105],
            [1000],
        )

        issues = validate_price_quality(prices)
        self.assertIn("Close outside [Low, High] on at least one candle", issues)

    def test_open_outside_range_is_flagged(self):
        prices = make_prices(
            ["2026-09-11 10:00"],
            [110],
            [102],
            [99],
            [100],
            [1000],
        )

        issues = validate_price_quality(prices)
        self.assertIn("Open outside [Low, High] on at least one candle", issues)

    def test_negative_volume_is_flagged(self):
        prices = make_prices(
            ["2026-09-11 10:00"],
            [100],
            [102],
            [99],
            [100],
            [-5],
        )

        issues = validate_price_quality(prices)
        self.assertIn("negative Volume", issues)

    def test_duplicate_timestamps_are_flagged(self):
        prices = make_prices(
            ["2026-09-11 10:00", "2026-09-11 10:00"],
            [100, 100],
            [102, 102],
            [99, 99],
            [101, 101],
            [1000, 1000],
        )

        issues = validate_price_quality(prices)
        self.assertIn("duplicate timestamps", issues)

    def test_short_interval_within_day_is_flagged(self):
        prices = make_prices(
            ["2026-09-11 10:00", "2026-09-11 10:02"],
            [100, 100],
            [102, 102],
            [99, 99],
            [101, 101],
            [1000, 1000],
        )

        issues = validate_price_quality(prices, expected_interval_minutes=5)
        self.assertTrue(any("shorter than 5 minutes" in issue for issue in issues))

    def test_session_boundary_gap_is_not_flagged(self):
        """An overnight gap between sessions is normal, not a data problem."""
        prices = make_prices(
            ["2026-09-10 15:55", "2026-09-11 09:30"],
            [100, 105],
            [102, 107],
            [99, 104],
            [101, 106],
            [1000, 1100],
        )

        issues = validate_price_quality(prices, expected_interval_minutes=5)
        self.assertEqual(issues, [])

    def test_interval_check_is_skipped_when_not_requested(self):
        prices = make_prices(
            ["2026-09-11 10:00", "2026-09-11 10:02"],
            [100, 100],
            [102, 102],
            [99, 99],
            [101, 101],
            [1000, 1000],
        )

        self.assertEqual(validate_price_quality(prices), [])


if __name__ == "__main__":
    unittest.main()
