"""Model factories for evaluation.py's walk-forward harness.

Each factory takes no arguments and returns an unfitted, scikit-learn
-compatible estimator, so `evaluation.evaluate_fold`/`run_walk_forward` can
call it once per fold without knowing anything about the model inside.
"""
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


def build_logistic_regression() -> Pipeline:
    """The model used since train.py / Phase 2: scaled Logistic Regression."""
    return Pipeline([
        ("scaler", StandardScaler()),
        ("classifier", LogisticRegression(max_iter=1000)),
    ])


def build_gradient_boosting() -> Pipeline:
    """A gradient-boosted-tree alternative to the linear model.

    No StandardScaler: tree splits only compare a feature to a threshold,
    so they're invariant to monotonic rescaling -- unlike LogisticRegression,
    which needs comparable feature scales for its coefficients and solver
    to behave well.

    `max_depth=3` and `learning_rate=0.05` are deliberately conservative:
    with roughly 1,000 training rows and 11 features, deeper trees or a
    higher learning rate fit training noise (like the 5-minute return
    features' day-to-day randomness) rather than a real pattern. This is a
    reasonable starting configuration, not a tuned one -- see
    reports/ for whether it's worth tuning further.
    """
    return Pipeline([
        ("classifier", HistGradientBoostingClassifier(
            max_depth=3,
            learning_rate=0.05,
            max_iter=200,
            random_state=0,
        )),
    ])


MODEL_FACTORIES = {
    "logistic_regression": build_logistic_regression,
    "gradient_boosting": build_gradient_boosting,
}
