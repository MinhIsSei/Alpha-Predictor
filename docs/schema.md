# Prediction Database Schema

SQLite file: `data/predictions/predictions.sqlite` (created automatically by
`src/predict.py` on first run). A small sample with real replay data is
checked into `data/demo/predictions_demo.sqlite` for building/testing the
dashboard without running the pipeline first.

Both tables are written only by the pipeline (`predict.py`,
`evaluate_predictions.py`). The dashboard must only read from them.

## `predictions`

One row per prediction attempt that passed all quality/timing checks.

| Column | Type | Notes |
|---|---|---|
| `ticker` | TEXT | e.g. `AAPL` |
| `interval` | TEXT | candle interval, e.g. `5m` |
| `candle_start` | TEXT | UTC ISO 8601 — start of the candle the prediction was based on |
| `candle_end` | TEXT | UTC ISO 8601 — `candle_start` + interval |
| `evaluation_time` | TEXT | UTC ISO 8601 — wall-clock time the prediction was made |
| `prediction_end` | TEXT | UTC ISO 8601 — `candle_end` + horizon; when the target candle closes |
| `mode` | TEXT | `replay` or `live` |
| `model_version` | TEXT | e.g. `aapl_logistic_v2` — bump whenever the feature set or model changes |
| `horizon_minutes` | INTEGER | minutes ahead being predicted (currently 30) |
| `predicted_class` | INTEGER | `1` = Up, `0` = Not up |
| `probability_up` | REAL | model's probability of `1`, range 0–1. **Not the same as accuracy.** |
| `reference_open` | REAL | OHLCV of `candle_start`, kept for audit without re-downloading from Yahoo Finance |
| `reference_high` | REAL | " |
| `reference_low` | REAL | " |
| `reference_close` | REAL | " |
| `reference_volume` | REAL | " |

Uniqueness: `(ticker, interval, candle_start, mode, model_version, horizon_minutes)`.
Re-running `predict.py` for the same candle/mode/model never inserts a duplicate
(`ON CONFLICT ... DO NOTHING`).

## `prediction_outcomes`

One row per prediction whose outcome has been determined. Created lazily by
`evaluate_predictions.py` the first time it has something to save — it may
not exist yet in a brand-new database.

| Column | Type | Notes |
|---|---|---|
| `ticker` | TEXT | |
| `interval` | TEXT | |
| `candle_start` | TEXT | matches the `predictions` row it evaluates |
| `mode` | TEXT | |
| `model_version` | TEXT | |
| `horizon_minutes` | INTEGER | |
| `reference_close` | REAL | Close at `candle_start`, re-fetched (or replayed) at evaluation time |
| `target_close` | REAL | Close at `prediction_end - interval` |
| `actual_class` | INTEGER | `1` if `target_close > reference_close`, else `0` |
| `is_correct` | INTEGER | `1`/`0` — whether `predicted_class == actual_class` |
| `evaluated_at` | TEXT | UTC ISO 8601 — when this outcome row was written |

Uniqueness: same key as `predictions`. A prediction with no matching row here
is still pending (either the target candle hasn't closed yet, or the
required price data wasn't available — old intraday history disappears from
Yahoo Finance after some time).

## Building a dashboard on top of this

```sql
-- Left join so pending predictions (no outcome yet) stay visible.
SELECT p.*, o.actual_class, o.is_correct
FROM predictions AS p
LEFT JOIN prediction_outcomes AS o
  ON  o.ticker           = p.ticker
  AND o.interval         = p.interval
  AND o.candle_start     = p.candle_start
  AND o.mode             = p.mode
  AND o.model_version    = p.model_version
  AND o.horizon_minutes  = p.horizon_minutes
ORDER BY p.candle_start DESC;
```

- Compute accuracy only over rows where `is_correct IS NOT NULL`, and always
  show the sample count next to it.
- All timestamps are UTC; convert to `America/New_York` (or the viewer's
  timezone) for display and say so in the UI.
- `probability_up` is a model confidence score, not a guarantee — never
  present it as accuracy.

## Schema changes

`predictions` gained the five `reference_*` columns when the audit-trail
requirement was added. `CREATE TABLE IF NOT EXISTS` does **not** migrate an
existing table, so a `predictions.sqlite` created before this change will
fail on insert (missing columns). Delete the old file (or add the columns
manually with `ALTER TABLE`) before running `predict.py` again.
