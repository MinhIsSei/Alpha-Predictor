import unittest

import pandas as pd

from src.error_analysis import accuracy_by_hour, calibration_table, feature_importance
from src.models import build_logistic_regression


class TestAccuracyByHour(unittest.TestCase):
    def test_splits_by_hour_and_computes_accuracy(self):
        index = pd.DatetimeIndex([
            "2026-09-11 10:00", "2026-09-11 10:05",
            "2026-09-11 11:00", "2026-09-11 11:05", "2026-09-11 11:10",
        ])
        y_true = pd.Series([1, 0, 1, 1, 0], index=index)
        y_pred = [1, 1, 1, 0, 0]  # hour 10: 1/2 correct; hour 11: 2/3 correct

        result = accuracy_by_hour(index, y_true, y_pred)

        self.assertEqual(result.loc[10, "n"], 2)
        self.assertAlmostEqual(result.loc[10, "accuracy"], 0.5)
        self.assertEqual(result.loc[11, "n"], 3)
        self.assertAlmostEqual(result.loc[11, "accuracy"], 2 / 3)

    def test_up_rate_reflects_actual_class_balance(self):
        index = pd.DatetimeIndex(["2026-09-11 10:00", "2026-09-11 10:05"])
        y_true = pd.Series([1, 1], index=index)
        y_pred = [0, 0]

        result = accuracy_by_hour(index, y_true, y_pred)

        self.assertAlmostEqual(result.loc[10, "up_rate"], 1.0)
        self.assertAlmostEqual(result.loc[10, "accuracy"], 0.0)


class TestCalibrationTable(unittest.TestCase):
    def test_perfectly_calibrated_bucket(self):
        y_true = pd.Series([1, 1, 0, 0])
        y_proba = [0.65, 0.65, 0.65, 0.65]  # all land in the (0.6, 1.0] bucket

        result = calibration_table(y_true, y_proba)

        bucket = result.iloc[0]
        self.assertEqual(bucket["n"], 4)
        self.assertAlmostEqual(bucket["mean_predicted_prob"], 0.65)
        self.assertAlmostEqual(bucket["actual_up_rate"], 0.5)

    def test_separates_low_and_high_confidence_buckets(self):
        y_true = pd.Series([0, 0, 1, 1])
        y_proba = [0.1, 0.2, 0.8, 0.9]

        result = calibration_table(y_true, y_proba)
        non_empty = result[result["n"] > 0]

        # Low-probability bucket should show a low actual Up-rate here, and
        # the high-probability bucket a high one -- a sane, non-inverted table.
        low_bucket_up_rate = non_empty.iloc[0]["actual_up_rate"]
        high_bucket_up_rate = non_empty.iloc[-1]["actual_up_rate"]
        self.assertLess(low_bucket_up_rate, high_bucket_up_rate)


class TestFeatureImportance(unittest.TestCase):
    def test_ranks_informative_feature_above_pure_noise(self):
        import numpy as np
        rng = np.random.default_rng(0)
        n = 200
        informative = rng.normal(size=n)
        noise = rng.normal(size=n)
        y = (informative > 0).astype(int)

        X = pd.DataFrame({"informative": informative, "noise": noise})
        model = build_logistic_regression().fit(X, y)

        result = feature_importance(model, X, pd.Series(y), ["informative", "noise"])

        self.assertEqual(result.index[0], "informative")
        self.assertGreater(result["informative"], result["noise"])


if __name__ == "__main__":
    unittest.main()
