import unittest

import numpy as np
import pandas as pd

from src.compare_models import compare_models
from src.models import build_logistic_regression


def _make_synthetic_dataset(n_days: int, bars_per_day: int = 10, seed: int = 0):
    rng = np.random.default_rng(seed)
    rows = []

    for day in range(n_days):
        date = pd.Timestamp("2026-08-03") + pd.Timedelta(days=day)
        for bar in range(bars_per_day):
            timestamp = date + pd.Timedelta(minutes=5 * bar)
            target = int(rng.random() < 0.5)
            rows.append({
                "timestamp": timestamp,
                "feature_1": target + rng.normal(scale=0.1),
                "feature_2": rng.normal(),
                "target_up_5m": target,
                "target_time": timestamp,
            })

    return pd.DataFrame(rows).set_index("timestamp")


class TestCompareModels(unittest.TestCase):
    def test_one_row_per_fold_per_model(self):
        df = _make_synthetic_dataset(n_days=10)

        results = compare_models(
            df, ["feature_1", "feature_2"], "target_up_5m",
            train_sessions=4, test_sessions=2,
            model_factories={"logistic_regression": build_logistic_regression},
        )

        self.assertEqual(results["model_name"].unique().tolist(), ["logistic_regression"])
        self.assertEqual(len(results), results["fold"].nunique())

    def test_multiple_models_share_the_same_fold_boundaries(self):
        df = _make_synthetic_dataset(n_days=10)

        results = compare_models(
            df, ["feature_1", "feature_2"], "target_up_5m",
            train_sessions=4, test_sessions=2,
            model_factories={
                "a": build_logistic_regression,
                "b": build_logistic_regression,
            },
        )

        # Every fold must appear once per model, with identical test windows.
        pivot = results.pivot(index="fold", columns="model_name", values="test_start")
        self.assertTrue((pivot["a"] == pivot["b"]).all())


if __name__ == "__main__":
    unittest.main()
