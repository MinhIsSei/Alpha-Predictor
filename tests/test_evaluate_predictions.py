import shutil
import sqlite3
import tempfile
import unittest
from contextlib import closing, contextmanager
from pathlib import Path

from src.evaluate_predictions import (
    compute_outcome,
    prices_are_usable,
    save_outcome,
    select_pending_predictions,
)


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


class TestComputeOutcome(unittest.TestCase):
    def test_correct_up_prediction(self):
        actual_class, is_correct = compute_outcome(
            reference_close=100.0, target_close=101.0, predicted_class=1
        )
        self.assertEqual(actual_class, 1)
        self.assertTrue(is_correct)

    def test_incorrect_up_prediction(self):
        actual_class, is_correct = compute_outcome(
            reference_close=100.0, target_close=99.0, predicted_class=1
        )
        self.assertEqual(actual_class, 0)
        self.assertFalse(is_correct)

    def test_equal_price_counts_as_not_up(self):
        actual_class, is_correct = compute_outcome(
            reference_close=100.0, target_close=100.0, predicted_class=0
        )
        self.assertEqual(actual_class, 0)
        self.assertTrue(is_correct)


class TestPricesAreUsable(unittest.TestCase):
    def test_both_positive_is_usable(self):
        self.assertTrue(prices_are_usable(100.0, 101.0))

    def test_nan_reference_is_not_usable(self):
        self.assertFalse(prices_are_usable(float("nan"), 101.0))

    def test_nan_target_is_not_usable(self):
        self.assertFalse(prices_are_usable(100.0, float("nan")))

    def test_zero_price_is_not_usable(self):
        self.assertFalse(prices_are_usable(0.0, 101.0))

    def test_negative_price_is_not_usable(self):
        self.assertFalse(prices_are_usable(100.0, -5.0))


class TestSaveOutcomeAndSelectPending(unittest.TestCase):
    def _make_prediction_row(self, candle_start, predicted_class=1):
        return {
            "ticker": "AAPL",
            "interval": "5m",
            "candle_start": candle_start,
            "candle_end": "2026-09-11T18:05:00+00:00",
            "evaluation_time": "2026-09-11T18:05:30+00:00",
            "prediction_end": "2026-09-11T18:35:00+00:00",
            "mode": "replay",
            "model_version": "aapl_logistic_v2",
            "horizon_minutes": 30,
            "predicted_class": predicted_class,
            "probability_up": 0.6,
            "reference_open": 100.0,
            "reference_high": 101.0,
            "reference_low": 99.0,
            "reference_close": 100.5,
            "reference_volume": 1000.0,
        }

    def _seed_predictions_table(self, db_path, rows):
        with closing(sqlite3.connect(db_path)) as connection, connection:
            connection.execute("""
                CREATE TABLE predictions (
                    ticker TEXT NOT NULL,
                    interval TEXT NOT NULL,
                    candle_start TEXT NOT NULL,
                    candle_end TEXT NOT NULL,
                    evaluation_time TEXT NOT NULL,
                    prediction_end TEXT NOT NULL,
                    mode TEXT NOT NULL,
                    model_version TEXT NOT NULL,
                    horizon_minutes INTEGER NOT NULL,
                    predicted_class INTEGER NOT NULL,
                    probability_up REAL NOT NULL,
                    reference_open REAL NOT NULL,
                    reference_high REAL NOT NULL,
                    reference_low REAL NOT NULL,
                    reference_close REAL NOT NULL,
                    reference_volume REAL NOT NULL
                )
            """)
            for row in rows:
                connection.execute(
                    """
                    INSERT INTO predictions VALUES (
                        :ticker, :interval, :candle_start, :candle_end,
                        :evaluation_time, :prediction_end, :mode,
                        :model_version, :horizon_minutes, :predicted_class,
                        :probability_up, :reference_open, :reference_high,
                        :reference_low, :reference_close, :reference_volume
                    )
                    """,
                    row,
                )

    def test_select_pending_returns_predictions_without_outcomes_table(self):
        with temp_dir() as tmp_dir:
            db_path = Path(tmp_dir) / "predictions.sqlite"
            self._seed_predictions_table(
                db_path, [self._make_prediction_row("2026-09-11T18:00:00+00:00")]
            )

            with closing(sqlite3.connect(db_path)) as connection:
                pending = select_pending_predictions(connection, "replay")

            self.assertEqual(len(pending), 1)

    def test_select_pending_excludes_already_evaluated(self):
        with temp_dir() as tmp_dir:
            db_path = Path(tmp_dir) / "predictions.sqlite"
            row = self._make_prediction_row("2026-09-11T18:00:00+00:00")
            self._seed_predictions_table(db_path, [row])

            save_outcome(
                db_path=db_path,
                row=row,
                reference_close=100.0,
                target_close=101.0,
                actual_class=1,
                is_correct=True,
            )

            with closing(sqlite3.connect(db_path)) as connection:
                pending = select_pending_predictions(connection, "replay")

            self.assertEqual(len(pending), 0)

    def test_select_pending_filters_by_mode(self):
        with temp_dir() as tmp_dir:
            db_path = Path(tmp_dir) / "predictions.sqlite"
            live_row = self._make_prediction_row("2026-09-11T18:00:00+00:00")
            live_row["mode"] = "live"
            self._seed_predictions_table(db_path, [live_row])

            with closing(sqlite3.connect(db_path)) as connection:
                pending = select_pending_predictions(connection, "replay")

            self.assertEqual(len(pending), 0)

    def test_save_outcome_does_not_duplicate(self):
        with temp_dir() as tmp_dir:
            db_path = Path(tmp_dir) / "predictions.sqlite"
            row = self._make_prediction_row("2026-09-11T18:00:00+00:00")

            save_outcome(db_path, row, 100.0, 101.0, 1, True)
            save_outcome(db_path, row, 100.0, 101.0, 1, True)

            with closing(sqlite3.connect(db_path)) as connection:
                count = connection.execute("SELECT COUNT(*) FROM prediction_outcomes").fetchone()[0]
            self.assertEqual(count, 1)


if __name__ == "__main__":
    unittest.main()
