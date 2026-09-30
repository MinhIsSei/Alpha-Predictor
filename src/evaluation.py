import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    log_loss,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


def session_walk_forward_splits(
    session_dates,
    train_sessions: int,
    test_sessions: int,
    step_sessions: int | None = None,
):
    """Yield rolling-origin (train_dates, test_dates) folds over trading days.

    Unlike a single fixed train/validation/test split, this walks a fixed-size
    training window forward through the whole date range, evaluating on the
    `test_sessions` days immediately after each window. That produces several
    independent out-of-sample estimates instead of one, which is what
    `reports/aapl_v2_experiment.md` flags as missing: its test partition was
    only four sessions, and adjacent 30-minute-horizon rows within a fold are
    not independent of each other, so a single fold's accuracy is a noisy
    estimate on its own.

    `step_sessions` controls how far the window advances between folds
    (defaults to `test_sessions`, i.e. non-overlapping test windows). Returns
    a list of (train_dates, test_dates) tuples, each a list of `Timestamp`.
    """
    session_dates = sorted(pd.DatetimeIndex(session_dates).unique())
    step = step_sessions or test_sessions

    folds = []
    start = 0
    while start + train_sessions + test_sessions <= len(session_dates):
        train_dates = session_dates[start : start + train_sessions]
        test_dates = session_dates[start + train_sessions : start + train_sessions + test_sessions]
        folds.append((train_dates, test_dates))
        start += step

    return folds


def _build_model() -> Pipeline:
    return Pipeline(
        [
            ("scaler", StandardScaler()),
            ("classifier", LogisticRegression(max_iter=1000)),
        ]
    )


def session_train_test_masks(df: pd.DataFrame, train_dates, test_dates):
    """Return (train_mask, test_mask) boolean Series for one date split.

    A row counts toward training only if both its own session and its
    target's session (`target_time`, from `target_utils.build_target`) fall
    inside `train_dates` — otherwise its label resolves into the test window
    and would leak the future into training. Shared by `evaluate_fold` and
    any other caller that needs the identical split (e.g. error analysis on
    a single held-out window), so the leakage rule is defined in one place.
    """
    train_dates_set = set(train_dates)
    test_dates_set = set(test_dates)
    session = df.index.normalize()

    train_mask = session.isin(train_dates_set) & df["target_time"].dt.normalize().isin(
        train_dates_set
    )
    test_mask = session.isin(test_dates_set)

    return train_mask, test_mask


def evaluate_fold(
    df: pd.DataFrame,
    feature_cols: list[str],
    target_col: str,
    train_dates,
    test_dates,
    model_factory=_build_model,
) -> dict | None:
    """Fit a baseline and a model on one walk-forward fold and score both.

    `df` must already carry a `target_time` column (see
    `target_utils.build_target`); see `session_train_test_masks` for how
    that's used to keep a future label out of training.

    `model_factory` returns an unfitted, scikit-learn-compatible estimator
    (default: the scaled Logistic Regression pipeline used since Phase 2) —
    pass a different factory (see `models.py`) to compare model types on the
    exact same folds without duplicating the walk-forward machinery.

    Returns None for a degenerate fold (an empty split, or a training split
    with only one class), so callers can skip it rather than crash a
    multi-fold run.
    """
    train_mask, test_mask = session_train_test_masks(df, train_dates, test_dates)

    train = df.loc[train_mask]
    test = df.loc[test_mask]

    if train.empty or test.empty or train[target_col].nunique() < 2:
        return None

    X_train, y_train = train[feature_cols], train[target_col].astype(int)
    X_test, y_test = test[feature_cols], test[target_col].astype(int)

    baseline = DummyClassifier(strategy="most_frequent").fit(X_train, y_train)
    model = model_factory().fit(X_train, y_train)

    baseline_pred = baseline.predict(X_test)
    model_pred = model.predict(X_test)

    up_index = list(model.classes_).index(1) if 1 in model.classes_ else None
    model_proba = model.predict_proba(X_test)[:, up_index] if up_index is not None else None

    result = {
        "train_start": train_dates[0],
        "train_end": train_dates[-1],
        "test_start": test_dates[0],
        "test_end": test_dates[-1],
        "n_train": len(train),
        "n_test": len(test),
        "up_rate_test": y_test.mean(),
        "baseline_accuracy": accuracy_score(y_test, baseline_pred),
        "model_accuracy": accuracy_score(y_test, model_pred),
        "model_precision": precision_score(y_test, model_pred, zero_division=0),
        "model_recall": recall_score(y_test, model_pred, zero_division=0),
        "model_f1": f1_score(y_test, model_pred, zero_division=0),
    }

    # ROC-AUC and log-loss need both classes present in the test fold.
    if model_proba is not None and y_test.nunique() == 2:
        result["model_roc_auc"] = roc_auc_score(y_test, model_proba)
        result["model_log_loss"] = log_loss(y_test, model_proba, labels=[0, 1])
    else:
        result["model_roc_auc"] = float("nan")
        result["model_log_loss"] = float("nan")

    return result


def run_walk_forward(
    df: pd.DataFrame,
    feature_cols: list[str],
    target_col: str,
    train_sessions: int,
    test_sessions: int,
    step_sessions: int | None = None,
    model_factory=_build_model,
) -> pd.DataFrame:
    """Run every walk-forward fold over `df`'s trading days and collect results.

    `model_factory` is forwarded to `evaluate_fold` — see there for what it
    must return.

    Returns one row per non-degenerate fold. An empty result means there
    weren't enough trading days, or every fold that fit was degenerate
    (see `evaluate_fold`).
    """
    session_dates = df.index.normalize().unique()
    folds = session_walk_forward_splits(session_dates, train_sessions, test_sessions, step_sessions)

    results = [
        evaluate_fold(df, feature_cols, target_col, train_dates, test_dates, model_factory)
        for train_dates, test_dates in folds
    ]
    results = [result for result in results if result is not None]

    return pd.DataFrame(results)


def summarize_folds(fold_results: pd.DataFrame) -> pd.DataFrame:
    """Aggregate mean/std across folds for every numeric metric column."""
    metric_cols = [
        col
        for col in fold_results.columns
        if col not in ("train_start", "train_end", "test_start", "test_end")
    ]
    return fold_results[metric_cols].agg(["mean", "std"])
