import unittest

import numpy as np
import pandas as pd

from src.evaluation import evaluate_fold, run_walk_forward, session_walk_forward_splits


class TestSessionWalkForwardSplits(unittest.TestCase):
    def setUp(self):
        self.dates = pd.date_range("2026-08-03", periods=10, freq="B")

    def test_non_overlapping_folds_by_default(self):
        folds = session_walk_forward_splits(self.dates, train_sessions=4, test_sessions=2)
        # Windows: [0:4]->[4:6], [2:6]->[6:8] step defaults to test_sessions=2,
        # so start advances 0, 2, 4, 6 -> folds while start+4+2<=10.
        self.assertEqual(len(folds), 3)

        first_train, first_test = folds[0]
        self.assertEqual(list(first_train), list(self.dates[0:4]))
        self.assertEqual(list(first_test), list(self.dates[4:6]))

    def test_folds_roll_forward_by_step(self):
        folds = session_walk_forward_splits(
            self.dates, train_sessions=4, test_sessions=2, step_sessions=3
        )
        starts = [train[0] for train, _ in folds]
        self.assertEqual(starts, [self.dates[0], self.dates[3]])

    def test_too_few_sessions_yields_no_folds(self):
        folds = session_walk_forward_splits(self.dates, train_sessions=8, test_sessions=5)
        self.assertEqual(folds, [])

    def test_duplicate_and_unordered_input_is_normalized(self):
        shuffled = list(self.dates[:5]) + [self.dates[0]] + list(self.dates[5:])
        folds = session_walk_forward_splits(shuffled, train_sessions=4, test_sessions=2)
        first_train, _ = folds[0]
        self.assertEqual(list(first_train), list(self.dates[0:4]))


def _make_synthetic_dataset(n_days: int, bars_per_day: int = 10, seed: int = 0):
    """A small, fast dataset with a real (if weak) learnable signal.

    Each day gets its own session date; `feature_1` is noise plus a boost
    exactly matching the target, so the model has something to learn and
    both classes appear, avoiding degenerate folds.
    """
    rng = np.random.default_rng(seed)
    rows = []

    for day in range(n_days):
        date = pd.Timestamp("2026-08-03") + pd.Timedelta(days=day)
        for bar in range(bars_per_day):
            timestamp = date + pd.Timedelta(minutes=5 * bar)
            target = int(rng.random() < 0.5)
            rows.append(
                {
                    "timestamp": timestamp,
                    "feature_1": target + rng.normal(scale=0.1),
                    "feature_2": rng.normal(),
                    "target_up_5m": target,
                    "target_time": timestamp,
                }
            )

    df = pd.DataFrame(rows).set_index("timestamp")
    return df


class TestEvaluateFold(unittest.TestCase):
    def test_returns_none_for_single_class_training_split(self):
        df = _make_synthetic_dataset(n_days=4)
        df["target_up_5m"] = 0  # force a degenerate, single-class fold

        train_dates = df.index.normalize().unique()[:3]
        test_dates = df.index.normalize().unique()[3:]

        result = evaluate_fold(
            df, ["feature_1", "feature_2"], "target_up_5m", train_dates, test_dates
        )
        self.assertIsNone(result)

    def test_produces_metrics_in_valid_ranges(self):
        df = _make_synthetic_dataset(n_days=6)
        dates = df.index.normalize().unique()
        train_dates, test_dates = dates[:4], dates[4:]

        result = evaluate_fold(
            df, ["feature_1", "feature_2"], "target_up_5m", train_dates, test_dates
        )

        self.assertIsNotNone(result)
        for key in (
            "baseline_accuracy",
            "model_accuracy",
            "model_precision",
            "model_recall",
            "model_f1",
        ):
            self.assertGreaterEqual(result[key], 0.0)
            self.assertLessEqual(result[key], 1.0)

        # feature_1 is a near-perfect proxy for the target, so the model
        # should clearly beat a most-frequent-class baseline here.
        self.assertGreater(result["model_accuracy"], result["baseline_accuracy"])

    def test_training_excludes_targets_resolving_in_test_window(self):
        """A label whose target_time falls in the test window must not train."""
        df = _make_synthetic_dataset(n_days=4)
        dates = df.index.normalize().unique()

        # Push the last training day's target_time into the test window.
        last_train_day_rows = df.index.normalize() == dates[2]
        df.loc[last_train_day_rows, "target_time"] = dates[3]

        result = evaluate_fold(
            df,
            ["feature_1", "feature_2"],
            "target_up_5m",
            train_dates=dates[:3],
            test_dates=dates[3:],
        )

        # Only day 0 and day 1 rows remain valid for training (day 2's
        # targets now point into the test day and must be excluded).
        expected_train_rows = (df.index.normalize().isin(dates[:2])).sum()
        self.assertEqual(result["n_train"], expected_train_rows)


class TestRunWalkForward(unittest.TestCase):
    def test_collects_one_row_per_fold(self):
        df = _make_synthetic_dataset(n_days=10)
        results = run_walk_forward(
            df,
            ["feature_1", "feature_2"],
            "target_up_5m",
            train_sessions=4,
            test_sessions=2,
        )
        self.assertEqual(len(results), 3)
        self.assertIn("model_accuracy", results.columns)

    def test_empty_when_not_enough_sessions(self):
        df = _make_synthetic_dataset(n_days=3)
        results = run_walk_forward(
            df,
            ["feature_1", "feature_2"],
            "target_up_5m",
            train_sessions=4,
            test_sessions=2,
        )
        self.assertTrue(results.empty)


if __name__ == "__main__":
    unittest.main()
