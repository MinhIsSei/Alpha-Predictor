"""Compare model types on the exact same walk-forward folds.

Usage:
    python -m src.compare_models
    python -m src.compare_models --train-sessions 8 --test-sessions 4

Reuses evaluate_walk_forward.py's rolling-origin methodology (Phase 2)
rather than a single train/test split, for the same reason: one split's
accuracy is a noisy estimate, especially with only ~20 trading sessions of
data. Each model is evaluated on identical fold boundaries (same train/test
rows), so differences in the results are attributable to the model, not to
different data.
"""
import argparse
from pathlib import Path

import pandas as pd

from src.evaluate_walk_forward import FEATURE_COLS, TARGET_COL
from src.evaluation import evaluate_fold, session_walk_forward_splits
from src.logging_config import configure_logging
from src.models import MODEL_FACTORIES

logger = configure_logging("compare_models")

project_root = Path(__file__).resolve().parent.parent


def compare_models(
    df: pd.DataFrame,
    feature_cols: list[str],
    target_col: str,
    train_sessions: int,
    test_sessions: int,
    step_sessions: int | None = None,
    model_factories: dict = MODEL_FACTORIES,
) -> pd.DataFrame:
    """Evaluate every model in `model_factories` on the same fold boundaries.

    Returns one row per (fold, model) pair with a `model_name` column, so
    the caller can compare models within a fold or aggregate per model.
    """
    session_dates = df.index.normalize().unique()
    folds = session_walk_forward_splits(
        session_dates, train_sessions, test_sessions, step_sessions
    )

    rows = []
    for fold_index, (train_dates, test_dates) in enumerate(folds):
        for model_name, factory in model_factories.items():
            result = evaluate_fold(
                df, feature_cols, target_col, train_dates, test_dates, factory
            )
            if result is None:
                continue
            result["fold"] = fold_index
            result["model_name"] = model_name
            rows.append(result)

    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compare model types on identical walk-forward folds."
    )
    parser.add_argument("--train-sessions", type=int, default=10)
    parser.add_argument("--test-sessions", type=int, default=2)
    parser.add_argument("--step-sessions", type=int, default=None)
    args = parser.parse_args()

    file_path = project_root / "data" / "processed" / "aapl_training_1mo.parquet"
    df = pd.read_parquet(file_path).sort_index()

    logger.info(
        "Loaded %d rows across %d trading sessions from %s",
        len(df), df.index.normalize().nunique(), file_path,
    )

    results = compare_models(
        df, FEATURE_COLS, TARGET_COL,
        args.train_sessions, args.test_sessions, args.step_sessions,
    )

    if results.empty:
        logger.warning("No usable folds; try smaller --train-sessions/--test-sessions.")
        raise SystemExit(0)

    pd.set_option("display.width", 120)

    print(f"\n{results['fold'].nunique()} folds x {results['model_name'].nunique()} models:\n")
    print(
        results[["fold", "model_name", "n_train", "n_test", "baseline_accuracy", "model_accuracy", "model_f1"]]
        .round(4).to_string(index=False)
    )

    per_model = results.groupby("model_name")[
        ["baseline_accuracy", "model_accuracy", "model_precision", "model_recall", "model_f1", "model_roc_auc"]
    ].agg(["mean", "std"])

    print("\nPer-model summary across folds (mean / std):\n")
    print(per_model.round(4).to_string())

    # Head-to-head: how often does each model have the higher accuracy
    # *within the same fold*, rather than each model's average in isolation.
    pivot = results.pivot(index="fold", columns="model_name", values="model_accuracy")
    model_names = list(model_factories_used(results))
    if len(model_names) == 2:
        a, b = model_names
        a_wins = (pivot[a] > pivot[b]).sum()
        b_wins = (pivot[b] > pivot[a]).sum()
        ties = (pivot[a] == pivot[b]).sum()
        print(
            f"\nHead-to-head on {len(pivot)} shared folds: "
            f"{a} wins {a_wins}, {b} wins {b_wins}, ties {ties}."
        )


def model_factories_used(results: pd.DataFrame):
    return results["model_name"].unique()


if __name__ == "__main__":
    main()
