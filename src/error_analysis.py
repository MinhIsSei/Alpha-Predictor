"""Feature importance and error analysis for each model type.

Usage:
    python -m src.error_analysis
    python -m src.error_analysis --test-sessions 4

Fits each model in models.MODEL_FACTORIES once on a single held-out window
(the last `--test-sessions` trading days; everything before that is
training) and inspects the fitted model and its test-set errors in more
detail than an accuracy number alone shows: a confusion matrix, accuracy
broken down by hour of day, whether predicted probabilities are calibrated
(the README repeatedly notes probability is not accuracy -- this is the
check for that), and which features the model actually relies on.

This reuses evaluation.py's session_walk_forward_splits/
session_train_test_masks for the split (with train_sessions set to use
everything before the test window) rather than a third split
implementation, so this script's train/test boundary is defined by the
same leakage-safe rule as compare_models.py and evaluate_walk_forward.py.
A single fit (not walk-forward across many folds) is deliberate here: fold
-to-fold *accuracy* robustness is compare_models.py's job, while this
script's row-level error breakdown needs one concrete set of predictions to
look at.
"""

import argparse

import pandas as pd
from sklearn.inspection import permutation_importance
from sklearn.metrics import classification_report, confusion_matrix

from src.config import FEATURE_COLS, PROCESSED_PATH, TARGET_COL
from src.evaluation import session_train_test_masks, session_walk_forward_splits
from src.logging_config import configure_logging
from src.models import MODEL_FACTORIES

logger = configure_logging("error_analysis")


def accuracy_by_hour(index: pd.DatetimeIndex, y_true: pd.Series, y_pred) -> pd.DataFrame:
    """Break down accuracy and class balance by hour of day (exchange time)."""
    frame = pd.DataFrame(
        {
            "hour": index.hour,
            "correct": (y_true.to_numpy() == y_pred),
            "actual_up": y_true.to_numpy(),
        }
    )
    grouped = frame.groupby("hour").agg(
        n=("correct", "size"),
        accuracy=("correct", "mean"),
        up_rate=("actual_up", "mean"),
    )
    return grouped


def calibration_table(
    y_true: pd.Series,
    y_proba,
    bins=(0.0, 0.4, 0.5, 0.6, 1.0),
) -> pd.DataFrame:
    """Compare mean predicted probability to actual Up-rate, per probability bucket.

    A well-calibrated model's `mean_predicted_prob` and `actual_up_rate`
    should be close within each bucket. A gap is exactly the situation the
    README warns about: "the probability reported by the model is not its
    accuracy" -- this makes that gap visible instead of just asserting it.
    """
    buckets = pd.cut(y_proba, bins=bins, include_lowest=True)
    frame = pd.DataFrame(
        {
            "bucket": buckets,
            "y_proba": y_proba,
            "y_true": y_true.to_numpy(),
        }
    )
    grouped = frame.groupby("bucket", observed=True).agg(
        n=("y_true", "size"),
        mean_predicted_prob=("y_proba", "mean"),
        actual_up_rate=("y_true", "mean"),
    )
    return grouped


def feature_importance(
    model,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    feature_cols: list[str],
    n_repeats: int = 20,
    random_state: int = 0,
) -> pd.Series:
    """Permutation importance on held-out data, most important first.

    For each feature, shuffle its values in `X_test` (breaking its
    relationship with the target while leaving every other feature and the
    row count untouched) and measure how much test accuracy drops,
    averaged over `n_repeats` shuffles. Unlike Logistic Regression
    coefficients vs. a tree ensemble's internal gain -- two numbers on
    different scales that were the initial approach here -- this uses the
    *same* out-of-sample metric for every model, including
    HistGradientBoostingClassifier, which does not expose
    `feature_importances_` at all. That makes rankings directly comparable
    across model types, and reflects each feature's actual out-of-sample
    contribution rather than an in-sample training statistic.
    """
    result = permutation_importance(
        model,
        X_test,
        y_test,
        n_repeats=n_repeats,
        random_state=random_state,
        scoring="accuracy",
    )
    return pd.Series(result.importances_mean, index=feature_cols).sort_values(ascending=False)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Feature importance and error analysis for each model."
    )
    parser.add_argument("--test-sessions", type=int, default=4)
    args = parser.parse_args()

    file_path = PROCESSED_PATH
    df = pd.read_parquet(file_path).sort_index()

    session_dates = df.index.normalize().unique()
    train_sessions = len(session_dates) - args.test_sessions

    folds = session_walk_forward_splits(
        session_dates, train_sessions=train_sessions, test_sessions=args.test_sessions
    )
    if not folds:
        logger.warning(
            "Not enough sessions (%d) for a %d-session held-out test window.",
            len(session_dates),
            args.test_sessions,
        )
        raise SystemExit(0)

    train_dates, test_dates = folds[0]
    train_mask, test_mask = session_train_test_masks(df, train_dates, test_dates)
    train, test = df.loc[train_mask], df.loc[test_mask]

    logger.info(
        "Train: %d rows (%s to %s). Test: %d rows (%s to %s).",
        len(train),
        train_dates[0].date(),
        train_dates[-1].date(),
        len(test),
        test_dates[0].date(),
        test_dates[-1].date(),
    )

    X_train, y_train = train[FEATURE_COLS], train[TARGET_COL].astype(int)
    X_test, y_test = test[FEATURE_COLS], test[TARGET_COL].astype(int)

    for model_name, factory in MODEL_FACTORIES.items():
        print(f"\n{'=' * 60}\n{model_name}\n{'=' * 60}")

        model = factory().fit(X_train, y_train)
        y_pred = model.predict(X_test)
        up_index = list(model.classes_).index(1)
        y_proba = model.predict_proba(X_test)[:, up_index]

        print("\nConfusion matrix [rows: actual, columns: predicted] [0, 1]:")
        print(confusion_matrix(y_test, y_pred, labels=[0, 1]))

        print(
            classification_report(
                y_test,
                y_pred,
                labels=[0, 1],
                target_names=["Not up", "Up"],
                digits=3,
                zero_division=0,
            )
        )

        print("Accuracy by hour (exchange time):")
        print(accuracy_by_hour(test.index, y_test, y_pred).round(4).to_string())

        print("\nCalibration (predicted probability vs. actual Up-rate):")
        print(calibration_table(y_test, y_proba).round(4).to_string())

        print("\nPermutation importance on held-out data (most influential first):")
        print(feature_importance(model, X_test, y_test, FEATURE_COLS).round(4).to_string())


if __name__ == "__main__":
    main()
