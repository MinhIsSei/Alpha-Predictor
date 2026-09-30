import unittest

import numpy as np
import pandas as pd

from src.models import MODEL_FACTORIES, build_gradient_boosting, build_logistic_regression


def _toy_data():
    rng = np.random.default_rng(0)
    X = pd.DataFrame({
        "feature_1": rng.normal(size=40),
        "feature_2": rng.normal(size=40),
    })
    y = (X["feature_1"] > 0).astype(int)
    return X, y


class TestModelFactories(unittest.TestCase):
    def test_logistic_regression_fits_and_predicts(self):
        X, y = _toy_data()
        model = build_logistic_regression().fit(X, y)

        predictions = model.predict(X)
        self.assertEqual(len(predictions), len(y))
        self.assertTrue(set(predictions).issubset({0, 1}))

    def test_gradient_boosting_fits_and_predicts(self):
        X, y = _toy_data()
        model = build_gradient_boosting().fit(X, y)

        predictions = model.predict(X)
        self.assertEqual(len(predictions), len(y))
        self.assertTrue(set(predictions).issubset({0, 1}))

    def test_each_call_returns_a_fresh_unfitted_estimator(self):
        # evaluate_fold calls the factory once per fold; each call must be
        # independent, not a shared fitted instance leaking across folds.
        first = build_logistic_regression()
        second = build_logistic_regression()
        self.assertIsNot(first, second)

    def test_predict_proba_is_available_for_both_models(self):
        # evaluation.py relies on predict_proba to compute ROC-AUC/log-loss.
        X, y = _toy_data()
        for name, factory in MODEL_FACTORIES.items():
            with self.subTest(model=name):
                model = factory().fit(X, y)
                proba = model.predict_proba(X)
                self.assertEqual(proba.shape, (len(y), 2))


if __name__ == "__main__":
    unittest.main()
