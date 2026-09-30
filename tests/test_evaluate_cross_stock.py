import unittest

import pandas as pd
from sklearn.dummy import DummyClassifier

from src.evaluate_cross_stock import evaluate_ticker


class TestEvaluateTicker(unittest.TestCase):
    def test_reports_accuracy_for_a_perfect_model(self):
        data = pd.DataFrame(
            {
                "feature_1": [0, 0, 1, 1, 1, 0],
                "target_up_30m": [0, 0, 1, 1, 1, 0],
            }
        )
        # A model that always predicts the (majority) class 0 is "perfect"
        # against this data's baseline, since the baseline predicts the same.
        model = DummyClassifier(strategy="most_frequent").fit(
            data[["feature_1"]], data["target_up_30m"]
        )

        result = evaluate_ticker(model, data, target_col="target_up_30m")

        self.assertEqual(result["n_rows"], 6)
        self.assertAlmostEqual(result["up_rate"], 3 / 6)
        self.assertAlmostEqual(result["baseline_accuracy"], result["model_accuracy"])

    def test_up_rate_reflects_class_balance(self):
        data = pd.DataFrame(
            {
                "feature_1": [0, 1, 1, 1],
                "target_up_30m": [0, 1, 1, 1],
            }
        )
        model = DummyClassifier(strategy="most_frequent").fit(
            data[["feature_1"]], data["target_up_30m"]
        )

        result = evaluate_ticker(model, data, target_col="target_up_30m")

        self.assertAlmostEqual(result["up_rate"], 0.75)


if __name__ == "__main__":
    unittest.main()
