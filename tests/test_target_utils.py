import unittest

import pandas as pd

from src.target_utils import build_target


def make_closes(timestamps, closes):
    index = pd.DatetimeIndex(timestamps, tz="America/New_York")
    return pd.DataFrame({"Close": closes}, index=index)


class TestBuildTarget(unittest.TestCase):
    def test_target_up_when_future_close_is_higher(self):
        prices = make_closes(
            [
                "2026-09-11 10:00",
                "2026-09-11 10:05",
                "2026-09-11 10:10",
                "2026-09-11 10:15",
                "2026-09-11 10:20",
                "2026-09-11 10:25",
                "2026-09-11 10:30",
            ],
            [100.0, 101, 102, 103, 104, 105, 106.0],
        )

        result = build_target(prices, horizon_minutes=30, periods=6)

        self.assertEqual(result["target_up_30m"].iloc[0], 1)
        self.assertAlmostEqual(result["future_close_30m"].iloc[0], 106.0)
        self.assertEqual(
            result["target_time"].iloc[0],
            pd.Timestamp("2026-09-11 10:30", tz="America/New_York"),
        )

    def test_target_not_up_when_future_close_is_lower_or_equal(self):
        prices = make_closes(
            [
                "2026-09-11 10:00",
                "2026-09-11 10:05",
                "2026-09-11 10:10",
                "2026-09-11 10:15",
                "2026-09-11 10:20",
                "2026-09-11 10:25",
                "2026-09-11 10:30",
            ],
            [100.0, 100, 100, 100, 100, 100, 100.0],
        )

        result = build_target(prices, horizon_minutes=30, periods=6)

        # Equal future close must count as "Not up" (class 0), not Up.
        self.assertEqual(result["target_up_30m"].iloc[0], 0)

    def test_missing_future_bar_is_na(self):
        """The last rows, with no bar far enough ahead, must be <NA>."""
        prices = make_closes(
            pd.date_range("2026-09-11 10:00", periods=6, freq="5min"),
            [100.0, 101, 102, 103, 104, 105.0],
        )

        result = build_target(prices, horizon_minutes=30, periods=6)

        self.assertTrue(result["target_up_30m"].isna().all())
        self.assertTrue(result["future_close_30m"].isna().all())
        self.assertTrue(result["target_time"].isna().all())

    def test_target_does_not_cross_session_gap(self):
        prices = make_closes(
            [
                "2026-09-10 15:55",
                "2026-09-11 09:30",
                "2026-09-11 09:35",
                "2026-09-11 09:40",
                "2026-09-11 09:45",
                "2026-09-11 09:50",
                "2026-09-11 09:55",
            ],
            [100.0, 200, 201, 202, 203, 204, 205.0],
        )

        result = build_target(prices, horizon_minutes=30, periods=6)

        # The pre-gap candle's "future" bar is 6 rows ahead in position, but
        # it lands on the next trading day - not a valid same-session target.
        self.assertTrue(pd.isna(result["target_up_30m"].iloc[0]))
        self.assertTrue(pd.isna(result["target_time"].iloc[0]))

    def test_target_respects_missing_candle_gap(self):
        """A skipped candle must invalidate the target's exact-minutes check."""
        prices = make_closes(
            [
                "2026-09-11 10:00",
                "2026-09-11 10:05",
                # 10:10 candle missing
                "2026-09-11 10:15",
                "2026-09-11 10:20",
            ],
            [100.0, 101, 108, 109.0],
        )

        # One bar ahead (periods=1) is meant to mean exactly 5 minutes.
        # Row 1 (10:05) -> row 2 (10:15) is actually a 10-minute jump because
        # 10:10 is missing, so this must not count as a valid 5-minute target.
        result = build_target(prices, horizon_minutes=5, periods=1)
        self.assertTrue(pd.isna(result["target_up_5m"].iloc[1]))

        # Row 2 (10:15) -> row 3 (10:20) is a genuine 5-minute step, so it
        # resumes producing a valid target once the gap is behind it.
        self.assertFalse(pd.isna(result["target_up_5m"].iloc[2]))


if __name__ == "__main__":
    unittest.main()
