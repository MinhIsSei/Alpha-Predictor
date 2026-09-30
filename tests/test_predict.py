import shutil
import sqlite3
import tempfile
import unittest
from contextlib import closing, contextmanager
from pathlib import Path

import pandas as pd

from src.predict import check_session_timing, filter_completed_candles, insert_prediction


@contextmanager
def temp_dir():
    """Like tempfile.TemporaryDirectory, but tolerates a Windows virus
    scanner briefly holding a lock on a just-closed sqlite file instead of
    raising during cleanup."""
    path = tempfile.mkdtemp()
    try:
        yield path
    finally:
        shutil.rmtree(path, ignore_errors=True)


class TestFilterCompletedCandles(unittest.TestCase):
    def test_keeps_only_fully_closed_candles(self):
        index = pd.DatetimeIndex(
            ["2026-09-11 10:00", "2026-09-11 10:05", "2026-09-11 10:10"],
            tz="America/New_York",
        )
        prices = pd.DataFrame({"Close": [100, 101, 102]}, index=index)

        # 10:10 candle closes at 10:15; "now" is only 10:12, so it's still open.
        now = pd.Timestamp("2026-09-11 10:12", tz="America/New_York")
        result = filter_completed_candles(prices, now)

        self.assertEqual(len(result), 2)
        self.assertNotIn(pd.Timestamp("2026-09-11 10:10", tz="America/New_York"), result.index)


class TestCheckSessionTiming(unittest.TestCase):
    def setUp(self):
        self.session_open = pd.Timestamp("2026-09-11 09:30", tz="America/New_York")
        self.session_close = pd.Timestamp("2026-09-11 16:00", tz="America/New_York")

    def test_passes_all_checks(self):
        candle_start = pd.Timestamp("2026-09-11 14:00", tz="America/New_York")
        candle_end = candle_start + pd.Timedelta(minutes=5)
        prediction_end = candle_end + pd.Timedelta(minutes=30)
        evaluation_time = candle_end

        reason = check_session_timing(
            candle_start,
            candle_end,
            prediction_end,
            self.session_open,
            self.session_close,
            evaluation_time,
        )
        self.assertIsNone(reason)

    def test_rejects_candle_before_session_open(self):
        candle_start = pd.Timestamp("2026-09-11 09:00", tz="America/New_York")
        candle_end = candle_start + pd.Timedelta(minutes=5)
        prediction_end = candle_end + pd.Timedelta(minutes=30)

        reason = check_session_timing(
            candle_start,
            candle_end,
            prediction_end,
            self.session_open,
            self.session_close,
            candle_end,
        )
        self.assertEqual(reason, "latest candle is outside this session.")

    def test_rejects_candle_ending_after_session_close(self):
        candle_start = pd.Timestamp("2026-09-11 15:57", tz="America/New_York")
        candle_end = candle_start + pd.Timedelta(minutes=5)
        prediction_end = candle_end + pd.Timedelta(minutes=30)

        reason = check_session_timing(
            candle_start,
            candle_end,
            prediction_end,
            self.session_open,
            self.session_close,
            candle_end,
        )
        self.assertEqual(reason, "latest candle is outside this session.")

    def test_rejects_target_beyond_session_close(self):
        # Candle itself fits, but its 30-minute target would land after close.
        candle_start = pd.Timestamp("2026-09-11 15:45", tz="America/New_York")
        candle_end = candle_start + pd.Timedelta(minutes=5)
        prediction_end = candle_end + pd.Timedelta(minutes=30)

        reason = check_session_timing(
            candle_start,
            candle_end,
            prediction_end,
            self.session_open,
            self.session_close,
            candle_end,
        )
        self.assertIn("insufficient time remaining in session", reason)

    def test_rejects_stale_candle(self):
        candle_start = pd.Timestamp("2026-09-11 14:00", tz="America/New_York")
        candle_end = candle_start + pd.Timedelta(minutes=5)
        prediction_end = candle_end + pd.Timedelta(minutes=30)
        # Evaluating 10 minutes after the candle closed, beyond the 5-minute tolerance.
        evaluation_time = candle_end + pd.Timedelta(minutes=10)

        reason = check_session_timing(
            candle_start,
            candle_end,
            prediction_end,
            self.session_open,
            self.session_close,
            evaluation_time,
        )
        self.assertIn("latest completed candle is stale", reason)


class TestInsertPrediction(unittest.TestCase):
    def _make_record(self, candle_start="2026-09-11T18:00:00+00:00"):
        return {
            "ticker": "AAPL",
            "interval": "5m",
            "candle_start": candle_start,
            "candle_end": "2026-09-11T18:05:00+00:00",
            "evaluation_time": "2026-09-11T18:05:30+00:00",
            "prediction_end": "2026-09-11T18:35:00+00:00",
            "mode": "live",
            "model_version": "aapl_logistic_v2",
            "horizon_minutes": 30,
            "predicted_class": 1,
            "probability_up": 0.55,
            "reference_open": 100.0,
            "reference_high": 101.0,
            "reference_low": 99.0,
            "reference_close": 100.5,
            "reference_volume": 1000.0,
        }

    def _row_count(self, db_path):
        with closing(sqlite3.connect(db_path)) as connection:
            return connection.execute("SELECT COUNT(*) FROM predictions").fetchone()[0]

    def test_inserts_new_row(self):
        with temp_dir() as tmp_dir:
            db_path = Path(tmp_dir) / "predictions.sqlite"
            inserted = insert_prediction(db_path, self._make_record())
            self.assertTrue(inserted)
            self.assertEqual(self._row_count(db_path), 1)

    def test_duplicate_key_is_not_inserted_twice(self):
        with temp_dir() as tmp_dir:
            db_path = Path(tmp_dir) / "predictions.sqlite"
            record = self._make_record()

            first = insert_prediction(db_path, record)
            second = insert_prediction(db_path, record)

            self.assertTrue(first)
            self.assertFalse(second)
            self.assertEqual(self._row_count(db_path), 1)

    def test_different_candle_start_is_a_separate_row(self):
        with temp_dir() as tmp_dir:
            db_path = Path(tmp_dir) / "predictions.sqlite"
            insert_prediction(db_path, self._make_record("2026-09-11T18:00:00+00:00"))
            insert_prediction(db_path, self._make_record("2026-09-11T18:05:00+00:00"))
            self.assertEqual(self._row_count(db_path), 2)


if __name__ == "__main__":
    unittest.main()
