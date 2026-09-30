"""Train, evaluate, and save the AAPL model.

Usage:
    python -m src.train
    python -m src.train --model-path experiments/candidate.joblib

Two evaluations, for two different questions:

1. Walk-forward (evaluation.py) -- "how well does this *method* generalize?"
   Rolling 10-session training windows, each tested on the next 2 sessions.
   This is the performance estimate to trust: it averages several
   out-of-sample windows instead of relying on one short test period.
2. A single chronological train / validation / test split -- produces the
   saved model and a final holdout check on the most recent sessions, with
   per-day and per-hour diagnostics on the validation period.

The saved model is fit on the train partition of (2), unchanged from
earlier versions, so retraining on the same snapshot reproduces the
deployed model. Refitting the final model on all sessions would use more
recent data, but it changes the model (and so its version and every live
comparison) — a modeling decision, not a pipeline one.
"""

import argparse
from datetime import UTC, datetime
from pathlib import Path

import joblib
import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

from src.artifacts import archive_existing, file_sha256, git_commit
from src.config import (
    FEATURE_COLS,
    HORIZON_MINUTES,
    INTERVAL,
    MODEL_PATH,
    MODEL_VERSION,
    PROCESSED_PATH,
    PROJECT_ROOT,
    TARGET_COL,
    TICKER,
    WALK_FORWARD_TEST_SESSIONS,
    WALK_FORWARD_TRAIN_SESSIONS,
)
from src.evaluation import run_walk_forward
from src.logging_config import configure_logging
from src.models import build_logistic_regression

logger = configure_logging("train")

VALIDATION_SESSIONS = 4
TEST_SESSIONS = 4


def chronological_split(df: pd.DataFrame):
    """Split into train / validation / test by trading session.

    The last TEST_SESSIONS sessions are test, the VALIDATION_SESSIONS before
    them validation, everything earlier training. A row only stays in a
    partition if its target also resolves before the next partition starts.
    """
    sessions = df.index.normalize().unique().sort_values()
    if len(sessions) < VALIDATION_SESSIONS + TEST_SESSIONS + 2:
        raise ValueError(
            f"At least {VALIDATION_SESSIONS + TEST_SESSIONS + 2} sessions are required; "
            f"got {len(sessions)}."
        )

    validation_start = sessions[-(VALIDATION_SESSIONS + TEST_SESSIONS)]
    test_start = sessions[-TEST_SESSIONS]

    train = df[(df.index < validation_start) & (df["target_time"] < validation_start)]
    validation = df[
        (df.index >= validation_start) & (df.index < test_start) & (df["target_time"] < test_start)
    ]
    test = df[df.index >= test_start]
    return train.copy(), validation.copy(), test.copy()


def summarize_walk_forward(fold_results: pd.DataFrame) -> dict | None:
    if fold_results.empty:
        return None
    model = fold_results["model_accuracy"]
    baseline = fold_results["baseline_accuracy"]
    return {
        "train_sessions": WALK_FORWARD_TRAIN_SESSIONS,
        "test_sessions": WALK_FORWARD_TEST_SESSIONS,
        "n_folds": len(fold_results),
        "mean_accuracy": float(model.mean()),
        "std_accuracy": float(model.std()),
        "mean_baseline_accuracy": float(baseline.mean()),
        "folds_beating_baseline": int((model > baseline).sum()),
    }


def log_classification(title: str, y_true, y_pred) -> None:
    logger.info(
        "%s -- accuracy %.2f%%\n"
        "Confusion matrix [rows: actual, columns: predicted] [0, 1]:\n%s\n%s",
        title,
        accuracy_score(y_true, y_pred) * 100,
        confusion_matrix(y_true, y_pred, labels=[0, 1]),
        classification_report(
            y_true,
            y_pred,
            labels=[0, 1],
            target_names=["Not up", "Up"],
            digits=3,
            zero_division=0,
        ),
    )


def log_validation_diagnostics(y_validation, baseline_pred, model_pred) -> None:
    results = pd.DataFrame({"actual": y_validation, "baseline": baseline_pred, "model": model_pred})
    results["baseline_correct"] = results["baseline"] == results["actual"]
    results["model_correct"] = results["model"] == results["actual"]

    daily = results.groupby(results.index.normalize()).agg(
        samples=("actual", "size"),
        up_rate=("actual", "mean"),
        baseline_accuracy=("baseline_correct", "mean"),
        model_accuracy=("model_correct", "mean"),
    )
    logger.info("Validation results by trading day:\n%s", daily.round(3))

    # Drill into whichever validation day the model did worst on.
    worst_day = daily["model_accuracy"].idxmin()
    day = results.loc[results.index.normalize() == worst_day].copy()
    day["hour"] = day.index.hour
    hourly = day.groupby("hour").agg(
        samples=("actual", "size"),
        actual_up_rate=("actual", "mean"),
        predicted_up_rate=("model", "mean"),
        accuracy=("model_correct", "mean"),
    )
    logger.info(
        "Worst validation day %s -- actual up rate %.2f%%, predicted up rate %.2f%%\n"
        "Confusion matrix:\n%s\nBy hour (New York time):\n%s",
        worst_day.date(),
        day["actual"].mean() * 100,
        day["model"].mean() * 100,
        confusion_matrix(day["actual"], day["model"], labels=[0, 1]),
        hourly.round(3),
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Train, evaluate, and save the AAPL model.")
    parser.add_argument("--input", type=Path, default=PROCESSED_PATH)
    parser.add_argument("--model-path", type=Path, default=MODEL_PATH)
    args = parser.parse_args()

    df = pd.read_parquet(args.input).sort_index()

    # 1. Walk-forward: the performance estimate.
    fold_results = run_walk_forward(
        df,
        FEATURE_COLS,
        TARGET_COL,
        train_sessions=WALK_FORWARD_TRAIN_SESSIONS,
        test_sessions=WALK_FORWARD_TEST_SESSIONS,
        model_factory=build_logistic_regression,
    )
    walk_forward = summarize_walk_forward(fold_results)
    if walk_forward is None:
        logger.warning("Not enough sessions for walk-forward evaluation; skipped.")
    else:
        logger.info(
            "Walk-forward folds:\n%s",
            fold_results[
                [
                    "test_start",
                    "test_end",
                    "n_train",
                    "n_test",
                    "baseline_accuracy",
                    "model_accuracy",
                ]
            ]
            .round(4)
            .to_string(index=False),
        )
        logger.info(
            "Walk-forward: mean accuracy %.2f%% (+/- %.2f%%) vs baseline %.2f%%; "
            "model beat baseline in %d/%d folds.",
            walk_forward["mean_accuracy"] * 100,
            walk_forward["std_accuracy"] * 100,
            walk_forward["mean_baseline_accuracy"] * 100,
            walk_forward["folds_beating_baseline"],
            walk_forward["n_folds"],
        )

    # 2. Chronological split: the saved model and a final holdout check.
    train, validation, test = chronological_split(df)
    for name, subset in [("Train", train), ("Validation", validation), ("Test", test)]:
        logger.info(
            "%s: %d rows, %s to %s, labels %s",
            name,
            len(subset),
            subset.index.min(),
            subset.index.max(),
            subset[TARGET_COL].value_counts().to_dict(),
        )

    X_train, y_train = train[FEATURE_COLS], train[TARGET_COL]
    X_validation, y_validation = validation[FEATURE_COLS], validation[TARGET_COL]
    X_test, y_test = test[FEATURE_COLS], test[TARGET_COL]

    baseline = DummyClassifier(strategy="most_frequent").fit(X_train, y_train)
    model = build_logistic_regression().fit(X_train, y_train)

    baseline_validation = baseline.predict(X_validation)
    model_validation = model.predict(X_validation)
    log_classification("Baseline -- validation", y_validation, baseline_validation)
    log_classification("Logistic Regression -- validation", y_validation, model_validation)
    log_validation_diagnostics(y_validation, baseline_validation, model_validation)

    baseline_test = baseline.predict(X_test)
    model_test = model.predict(X_test)
    log_classification("Baseline -- test", y_test, baseline_test)
    log_classification("Logistic Regression -- test", y_test, model_test)

    artifact = {
        "pipeline": model,
        "model_version": MODEL_VERSION,
        "feature_cols": FEATURE_COLS,
        "ticker": TICKER,
        "interval": INTERVAL,
        "horizon_minutes": HORIZON_MINUTES,
        "train_start": str(train.index.min()),
        "train_end": str(train.index.max()),
        "validation_accuracy": accuracy_score(y_validation, model_validation),
        "test_accuracy": accuracy_score(y_test, model_test),
        "baseline_test_accuracy": accuracy_score(y_test, baseline_test),
        "walk_forward": walk_forward,
        # Provenance: enough to tell later exactly which data and code
        # produced this file.
        "trained_at": datetime.now(tz=UTC).isoformat(),
        "training_data_path": str(args.input),
        "training_data_sha256": file_sha256(args.input),
        "git_commit": git_commit(PROJECT_ROOT),
    }

    args.model_path.parent.mkdir(parents=True, exist_ok=True)
    archived = archive_existing(args.model_path)
    if archived:
        logger.info("Previous model archived to: %s", archived)

    joblib.dump(artifact, args.model_path)
    logger.info("Saved model: %s", args.model_path)


if __name__ == "__main__":
    main()
