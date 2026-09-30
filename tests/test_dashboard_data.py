import shutil
import sqlite3
import tempfile
import unittest
from contextlib import closing, contextmanager
from pathlib import Path

import pandas as pd

from src.dashboard_data import (
    STATUS_AWAITING,
    STATUS_CORRECT,
    STATUS_INCORRECT,
    STATUS_UNRESOLVED,
    connect_read_only,
    cumulative_accuracy,
    load_predictions,
    model_comparison_long,
    summarize,
)

DEMO_DB = Path(__file__).resolve().parent.parent / "data" / "demo" / "predictions_demo.sqlite"


@contextmanager
def temp_dir():
    """Tolerates a Windows virus scanner briefly locking a just-closed sqlite file."""
    path = tempfile.mkdtemp()
    try:
        yield Path(path)
    finally:
        shutil.rmtree(path, ignore_errors=True)


PREDICTIONS_SCHEMA = """
    CREATE TABLE predictions (
        ticker TEXT, interval TEXT, candle_start TEXT, candle_end TEXT,
        evaluation_time TEXT, prediction_end TEXT, mode TEXT, model_version TEXT,
        horizon_minutes INTEGER, predicted_class INTEGER, probability_up REAL,
        reference_open REAL, reference_high REAL, reference_low REAL,
        reference_close REAL, reference_volume REAL
    )
"""
OUTCOMES_SCHEMA = """
    CREATE TABLE prediction_outcomes (
        ticker TEXT, interval TEXT, candle_start TEXT, mode TEXT, model_version TEXT,
        horizon_minutes INTEGER, reference_close REAL, target_close REAL,
        actual_class INTEGER, is_correct INTEGER, evaluated_at TEXT
    )
"""


def prediction_row(candle_start, prediction_end, evaluation_time=None, predicted_class=1):
    return (
        "AAPL", "5m", candle_start, candle_start, evaluation_time or candle_start,
        prediction_end, "live", "aapl_logistic_v2", 30, predicted_class, 0.6,
        100.0, 101.0, 99.0, 100.5, 1000.0,
    )


def outcome_row(candle_start, is_correct):
    return (
        "AAPL", "5m", candle_start, "live", "aapl_logistic_v2", 30,
        100.5, 101.0, 1, is_correct, "2026-09-15T16:00:00+00:00",
    )


def build_db(path, predictions, outcomes=None):
    with closing(sqlite3.connect(path)) as connection, connection:
        connection.execute(PREDICTIONS_SCHEMA)
        connection.executemany(f"INSERT INTO predictions VALUES ({','.join('?' * 16)})", predictions)
        if outcomes is not None:
            connection.execute(OUTCOMES_SCHEMA)
            connection.executemany(
                f"INSERT INTO prediction_outcomes VALUES ({','.join('?' * 11)})", outcomes
            )


class TestConnectReadOnly(unittest.TestCase):
    def test_writes_are_rejected(self):
        with temp_dir() as tmp:
            db_path = tmp / "p.sqlite"
            build_db(db_path, [prediction_row("2026-09-15T15:00:00+00:00", "2026-09-15T15:35:00+00:00")])

            with closing(connect_read_only(db_path)) as connection:
                with self.assertRaises(sqlite3.OperationalError):
                    connection.execute("DELETE FROM predictions")

    def test_missing_file_is_not_created(self):
        with temp_dir() as tmp:
            db_path = tmp / "does_not_exist.sqlite"
            with self.assertRaises(sqlite3.OperationalError):
                connect_read_only(db_path)
            self.assertFalse(db_path.exists())


class TestLoadPredictions(unittest.TestCase):
    NOW = pd.Timestamp("2026-09-15T16:00:00", tz="UTC")

    def test_left_join_keeps_pending_predictions_and_labels_status(self):
        with temp_dir() as tmp:
            db_path = tmp / "p.sqlite"
            build_db(
                db_path,
                predictions=[
                    prediction_row("2026-09-15T15:00:00+00:00", "2026-09-15T15:35:00+00:00"),
                    prediction_row("2026-09-15T15:05:00+00:00", "2026-09-15T15:40:00+00:00"),
                    prediction_row("2026-09-15T15:10:00+00:00", "2026-09-15T15:45:00+00:00"),
                    prediction_row("2026-09-15T15:50:00+00:00", "2026-09-15T16:25:00+00:00"),
                ],
                outcomes=[
                    outcome_row("2026-09-15T15:00:00+00:00", 1),
                    outcome_row("2026-09-15T15:05:00+00:00", 0),
                ],
            )

            df = load_predictions(db_path, now=self.NOW)

        self.assertEqual(
            df["status"].tolist(),
            [STATUS_CORRECT, STATUS_INCORRECT, STATUS_UNRESOLVED, STATUS_AWAITING],
        )

    def test_works_before_outcomes_table_exists(self):
        with temp_dir() as tmp:
            db_path = tmp / "p.sqlite"
            build_db(db_path, [prediction_row("2026-09-15T15:00:00+00:00", "2026-09-15T15:35:00+00:00")])

            df = load_predictions(db_path, now=self.NOW)

        self.assertEqual(len(df), 1)
        self.assertEqual(df["status"].iloc[0], STATUS_UNRESOLVED)

    def test_converts_utc_to_new_york_and_accepts_mixed_iso_formats(self):
        with temp_dir() as tmp:
            db_path = tmp / "p.sqlite"
            build_db(db_path, [
                prediction_row("2026-09-15T15:00:00+00:00", "2026-09-15T15:35:00+00:00",
                               evaluation_time="2026-09-15T15:05:30.123456+00:00"),
                prediction_row("2026-09-15T15:05:00+00:00", "2026-09-15T15:40:00+00:00",
                               evaluation_time="2026-09-15T15:10:00+00:00"),
            ])

            df = load_predictions(db_path, now=self.NOW)

        self.assertEqual(str(df["candle_start"].dt.tz), "America/New_York")
        self.assertEqual(df["candle_start"].iloc[0].hour, 11)  # 15:00 UTC = 11:00 EDT

    def test_demo_database_loads(self):
        df = load_predictions(DEMO_DB)
        self.assertGreater(len(df), 0)
        self.assertTrue(set(df["status"]).issubset({STATUS_CORRECT, STATUS_INCORRECT}))


class TestSummarize(unittest.TestCase):
    def _df(self, statuses, predicted):
        return pd.DataFrame({"status": statuses, "predicted_class": predicted})

    def test_accuracy_uses_only_evaluated_rows(self):
        stats = summarize(self._df(
            [STATUS_CORRECT, STATUS_INCORRECT, STATUS_CORRECT, STATUS_UNRESOLVED, STATUS_AWAITING],
            [1, 1, 0, 1, 1],
        ))
        self.assertEqual(stats["n_evaluated"], 3)
        self.assertAlmostEqual(stats["accuracy"], 2 / 3)
        self.assertEqual(stats["n_unresolved"], 1)
        self.assertEqual(stats["n_awaiting"], 1)
        self.assertAlmostEqual(stats["predicted_up_rate"], 0.8)

    def test_accuracy_is_none_not_zero_when_nothing_evaluated(self):
        stats = summarize(self._df([STATUS_AWAITING], [1]))
        self.assertIsNone(stats["accuracy"])


class TestCumulativeAccuracy(unittest.TestCase):
    def test_running_accuracy_skips_pending_rows(self):
        df = pd.DataFrame({
            "candle_start": pd.date_range("2026-09-15 11:00", periods=4, freq="5min"),
            "status": [STATUS_CORRECT, STATUS_UNRESOLVED, STATUS_INCORRECT, STATUS_CORRECT],
        })
        running = cumulative_accuracy(df)
        self.assertEqual(running["n"].tolist(), [1, 2, 3])
        self.assertEqual(running["cumulative_accuracy"].round(4).tolist(), [1.0, 0.5, 0.6667])


class TestModelComparisonLong(unittest.TestCase):
    def test_baseline_appears_once_per_fold(self):
        fold_results = pd.DataFrame({
            "fold": [0, 0],
            "test_start": [pd.Timestamp("2026-09-01")] * 2,
            "test_end": [pd.Timestamp("2026-09-02")] * 2,
            "model_name": ["logistic_regression", "gradient_boosting"],
            "model_accuracy": [0.55, 0.50],
            "baseline_accuracy": [0.48, 0.48],
        })
        long_df = model_comparison_long(fold_results)

        self.assertEqual(sorted(long_df["series"]), ["baseline", "gradient_boosting", "logistic_regression"])
        self.assertAlmostEqual(long_df.set_index("series").loc["baseline", "accuracy"], 0.48)


if __name__ == "__main__":
    unittest.main()
