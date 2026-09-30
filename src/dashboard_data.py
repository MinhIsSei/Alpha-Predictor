"""Read-only data layer for the Streamlit dashboard (src/dashboard.py).

Kept free of any Streamlit import so every calculation the dashboard shows
can be unit-tested directly, and so the rules in docs/schema.md — read-only
access, a left join that keeps pending predictions visible, accuracy computed
only over evaluated rows — live in one testable place rather than inside UI
code.
"""
from contextlib import closing
from pathlib import Path
import sqlite3

import pandas as pd

DISPLAY_TZ = "America/New_York"

KEY_COLUMNS = [
    "ticker", "interval", "candle_start", "mode", "model_version", "horizon_minutes",
]

STATUS_CORRECT = "Correct"
STATUS_INCORRECT = "Incorrect"
STATUS_AWAITING = "Awaiting target"
STATUS_UNRESOLVED = "Not evaluated"


def connect_read_only(db_path: Path) -> sqlite3.Connection:
    """Open the SQLite file in read-only mode.

    docs/schema.md: "The dashboard must only read from them." `mode=ro`
    makes that a guarantee enforced by SQLite itself (any write raises),
    rather than a convention the UI code has to remember to follow. It also
    refuses to create an empty database file if the path doesn't exist,
    unlike a plain `sqlite3.connect`.
    """
    return sqlite3.connect(f"{Path(db_path).resolve().as_uri()}?mode=ro", uri=True)


def _table_exists(connection: sqlite3.Connection, name: str) -> bool:
    return connection.execute(
        "SELECT 1 FROM sqlite_master WHERE type = 'table' AND name = ?", (name,)
    ).fetchone() is not None


def load_predictions(db_path: Path, now: pd.Timestamp | None = None) -> pd.DataFrame:
    """Load every prediction left-joined to its outcome, if any.

    Adds a `status` column: Correct / Incorrect for evaluated rows, and for
    rows without an outcome, "Awaiting target" if the 30-minute target
    hasn't closed yet at `now`, else "Not evaluated" — the target has
    passed but no outcome is stored, either because evaluate_predictions.py
    hasn't run since, or (schema.md) because Yahoo no longer serves the old
    intraday candle. That is a different situation from one that simply
    isn't due yet, so the two are kept apart. Timestamps are converted from
    stored UTC to New York time for display.
    """
    now = now if now is not None else pd.Timestamp.now(tz="UTC")

    with closing(connect_read_only(db_path)) as connection:
        if not _table_exists(connection, "predictions"):
            return pd.DataFrame()

        if _table_exists(connection, "prediction_outcomes"):
            join_on = " AND ".join(f"o.{col} = p.{col}" for col in KEY_COLUMNS)
            query = f"""
                SELECT p.*, o.target_close, o.actual_class, o.is_correct, o.evaluated_at
                FROM predictions AS p
                LEFT JOIN prediction_outcomes AS o ON {join_on}
                ORDER BY p.candle_start
            """
        else:
            # prediction_outcomes is created lazily by evaluate_predictions.py,
            # so a brand-new database may not have it yet.
            query = """
                SELECT p.*, NULL AS target_close, NULL AS actual_class,
                       NULL AS is_correct, NULL AS evaluated_at
                FROM predictions AS p
                ORDER BY p.candle_start
            """
        df = pd.read_sql_query(query, connection)

    if df.empty:
        return df

    # format="ISO8601": live-mode rows store wall-clock times with
    # microseconds while replay rows don't, so one column mixes both shapes.
    for col in ["candle_start", "candle_end", "evaluation_time", "prediction_end", "evaluated_at"]:
        df[col] = pd.to_datetime(df[col], utc=True, format="ISO8601").dt.tz_convert(DISPLAY_TZ)

    df["status"] = df.apply(lambda row: _status(row, now), axis=1)
    df["predicted_label"] = df["predicted_class"].map({1: "Up", 0: "Not up"})
    df["actual_label"] = df["actual_class"].map({1: "Up", 0: "Not up"})

    return df


def _status(row, now: pd.Timestamp) -> str:
    if pd.notna(row["is_correct"]):
        return STATUS_CORRECT if int(row["is_correct"]) == 1 else STATUS_INCORRECT
    if row["prediction_end"] > now:
        return STATUS_AWAITING
    return STATUS_UNRESOLVED


def summarize(df: pd.DataFrame) -> dict:
    """Headline counts and accuracy, always paired with its sample size.

    Accuracy is None (not 0) when nothing has been evaluated yet, so the UI
    can say "no evaluated predictions" instead of showing a misleading 0%.
    """
    evaluated = df["status"].isin([STATUS_CORRECT, STATUS_INCORRECT])
    n_evaluated = int(evaluated.sum())
    n_correct = int((df["status"] == STATUS_CORRECT).sum())

    return {
        "n_predictions": len(df),
        "n_evaluated": n_evaluated,
        "n_correct": n_correct,
        "n_awaiting": int((df["status"] == STATUS_AWAITING).sum()),
        "n_unresolved": int((df["status"] == STATUS_UNRESOLVED).sum()),
        "accuracy": n_correct / n_evaluated if n_evaluated else None,
        "predicted_up_rate": float(df["predicted_class"].mean()) if len(df) else None,
    }


def cumulative_accuracy(df: pd.DataFrame) -> pd.DataFrame:
    """Running accuracy over evaluated predictions, in candle order.

    With few evaluated predictions this line swings hard on each new
    outcome — that volatility is the honest picture of a small sample, which
    is why the chart shows it next to the sample count rather than smoothing
    it away.
    """
    evaluated = df[df["status"].isin([STATUS_CORRECT, STATUS_INCORRECT])].copy()
    evaluated = evaluated.sort_values("candle_start").reset_index(drop=True)

    evaluated["n"] = evaluated.index + 1
    evaluated["cumulative_accuracy"] = (
        (evaluated["status"] == STATUS_CORRECT).cumsum() / evaluated["n"]
    )
    return evaluated[["n", "candle_start", "status", "cumulative_accuracy"]]


def model_comparison_long(fold_results: pd.DataFrame) -> pd.DataFrame:
    """Reshape compare_models output into one row per (fold, series).

    The baseline is identical for every model within a fold (it only depends
    on the fold's training labels), so it's taken once per fold rather than
    repeated per model — otherwise the chart would draw the same baseline
    bar twice.
    """
    models = fold_results[["fold", "test_start", "test_end", "model_name", "model_accuracy"]].rename(
        columns={"model_name": "series", "model_accuracy": "accuracy"}
    )
    baseline = (
        fold_results.groupby("fold", as_index=False)
        .first()[["fold", "test_start", "test_end", "baseline_accuracy"]]
        .rename(columns={"baseline_accuracy": "accuracy"})
        .assign(series="baseline")
    )
    combined = pd.concat([baseline, models], ignore_index=True)
    combined["fold_label"] = combined.apply(
        lambda row: f"Fold {row['fold'] + 1}: {row['test_start']:%b %d}–{row['test_end']:%b %d}",
        axis=1,
    )
    return combined.sort_values(["fold", "series"]).reset_index(drop=True)
