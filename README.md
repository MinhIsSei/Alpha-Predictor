# Alpha-Predictor — Pipeline v1

A Python project for predicting whether AAPL's closing price will increase over the next 30 minutes, using five-minute market data from Yahoo Finance.

This version includes data ingestion, feature engineering, chronological model evaluation, historical replay, live-data prediction, and SQLite storage for predictions and outcomes.

**Current status:** historical replay and a complete live prediction-to-outcome cycle have both been verified. The model has not outperformed the baseline on the evaluated test set.

## Prediction Task

- **Prediction ticker:** AAPL
- **Data interval:** 5 minutes
- **Prediction horizon:** 30 minutes within the same trading session
- **Class 1 — Up:** future Close is greater than the reference Close
- **Class 0 — Not up:** future Close is less than or equal to the reference Close

Predictions use completed candles only.

For example, a candle timestamped `14:00` closes at `14:05`. Its prediction compares that Close with the Close of the `14:30` candle, which closes at `14:35`.

The probability reported by the model is not its accuracy.

## Project Structure

```text
src/
├── ingest.py                # Download and save historical OHLCV data
├── feature_utils.py         # Shared feature calculations
├── target_utils.py          # Shared same-session future-return target calculation
├── data_quality.py          # Shared raw-OHLCV quality checks
├── features.py              # Build the training dataset and targets
├── train.py                 # Train, evaluate, and save the model
├── evaluation.py             # Walk-forward (rolling-origin) evaluation harness
├── evaluate_walk_forward.py  # CLI: walk-forward evaluation of the AAPL model
├── evaluate_cross_stock.py   # CLI: evaluate the AAPL model on other tickers
├── models.py                 # Model factories used by the harness above (Logistic Regression, gradient boosting)
├── compare_models.py         # CLI: compare model types on identical walk-forward folds
├── error_analysis.py         # CLI: confusion matrix, calibration, permutation importance per model
├── dashboard_data.py         # Read-only data layer for the dashboard (no Streamlit import; unit-tested)
├── dashboard.py              # Streamlit dashboard UI
├── predict.py                # Run replay/live predictions and store results
├── evaluate_predictions.py  # Match predictions with observed outcomes
├── run_live_loop.py         # Repeat predict.py/evaluate_predictions.py on a schedule
├── logging_config.py        # Shared logging setup used by all scripts above
└── net_utils.py              # Retry helper for Yahoo Finance calls

notebooks/
├── exploration.ipynb        # Exploratory data analysis
└── AI_Brain.ipynb           # Step-by-step walkthrough of the feature pipeline

docs/
└── schema.md                # SQLite schema reference for the predictions database

data/
├── raw/
├── processed/
├── predictions/
│   └── predictions.sqlite
└── demo/
    └── predictions_demo.sqlite  # Sample data, checked into Git, for dashboard work

models/
└── aapl_logistic_v2.joblib

requirements.txt
.gitignore
README.md
```

Data, model artifacts, and SQLite databases are generated locally and are not included in Git,
**except** `data/demo/`, which holds a small checked-in sample so the dashboard can be built
and tested without running the full pipeline first (see [`docs/schema.md`](docs/schema.md)).

## Setup

Clone this branch:

```bash
git clone https://github.com/MinhIsSei/Alpha-Predictor.git
cd Alpha-Predictor
```

Create and activate a virtual environment.

**macOS / Linux:**

```bash
python -m venv .venv
source .venv/bin/activate
```

**Windows PowerShell:**

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
```

Install dependencies:

```bash
python -m pip install -r requirements.txt
python -m pip check
```

Run all following commands from the repository root, using `python -m src.<module>`
rather than `python src/<module>.py`. The `-m` form puts the repository root
on the import path, which every script under `src/` needs since they import
each other with absolute imports (e.g. `from src.feature_utils import
build_features`) — the same style the test suite uses.

Dependencies were installed and replay was verified in a fresh virtual environment on the developer's Mac. Other platforms have not yet been verified.

## Data and Features

The ingestion script downloads one month of five-minute OHLCV data for:

```text
AAPL, MSFT, NVDA, GOOGL, AMZN
```

Training and prediction currently use AAPL only.

The saved model uses eleven features, shared between training and prediction
through `feature_utils.build_features`:

| Feature | Definition |
|---|---|
| `range_pct` | `(High - Low) / Open × 100` |
| `body_pct` | `(Close - Open) / Open × 100` |
| `return_5m_pct` | Percentage change in Close over exactly five minutes |
| `return_15m_pct` | Percentage change in Close over exactly fifteen minutes |
| `return_30m_pct` | Percentage change in Close over exactly thirty minutes |
| `volatility_30m` | Rolling std-dev of `return_5m_pct` over the trailing 6 bars, per trading day |
| `volume_relative` | Volume relative to its trailing 20-bar average, per trading day |
| `rsi_14` | 14-period Relative Strength Index |
| `macd_diff` | MACD line minus its signal line |
| `bb_width_pct` | Bollinger Band width, normalized by Close |
| `atr_pct` | Average True Range, normalized by Close |

Each return feature is masked to `NaN` when its lookback window crosses a
session gap (overnight/weekend), so it never mixes non-adjacent bars.
Volatility and volume features are computed per trading day for the same
reason. Rows missing any feature — including indicator warm-up rows at the
start of the series — are dropped before training.

Targets and future prices are never included in model inputs.

## Workflow A — Create a New Experiment

Download data:

```bash
python -m src.ingest
```

Output:

```text
data/raw/stock-trend_1mo.parquet
```

Build features and targets:

```bash
python -m src.features
```

Output:

```text
data/processed/aapl_training_1mo.parquet
```

Train and evaluate:

```bash
python -m src.train
```

Model output:

```text
models/aapl_logistic_v2.joblib
```

The artifact filename and `model_version` string were bumped from `v1` to
`v2` when the feature set was expanded from 3–4 candle-shape features to the
full 11-feature set (technical indicators, volatility, relative volume) —
per the coordination rule that a feature-set change requires a version bump.

The training script uses chronological session splits:

- Earlier sessions: training
- Four sessions before test: validation
- Final four sessions: test

Training labels must end before the next partition begins.

The model is a scikit-learn Pipeline containing `StandardScaler` and `LogisticRegression`. It is compared with a `DummyClassifier` that predicts the most frequent training class.

### Current experiment limitations

These scripts use fixed output filenames. Running ingestion, feature generation, or training again can overwrite existing artifacts. Preserve the original snapshot and model before starting a new experiment.

A rolling one-month download changes over time. It will not necessarily reproduce the dates, row counts, or scores reported below.

The training script drills into whichever validation day has the lowest
model accuracy, so its day-level diagnostics adapt to a new date range
automatically. Update the model version in both the artifact filename and
prediction-storage metadata when releasing a new model.

Do not repeatedly tune against the test set.

## Workflow B — Historical Replay

To reproduce the demonstrated replay, obtain the matching original artifacts from the project maintainers and place them at:

```text
data/raw/stock-trend_1mo.parquet
models/aapl_logistic_v2.joblib
```

Only load trusted model artifacts.

**Note:** the demonstrated output below is from the `v2` artifact (11
features). A rolling one-month download will eventually age this replay
date out of range, and any retrain will shift the model's exact
probabilities — treat these figures as an example of the expected output
shape, not a fixed target to reproduce.

Run:

```bash
python -m src.predict --mode replay --as-of "2026-09-11 14:05:00"
```

A timezone-naive `--as-of` value is interpreted as New York time. Observations whose candles close after that time are excluded.

With the `v2` artifact, the demonstrated result is:

```text
Candle start: 2026-09-11 14:00:00-04:00
Prediction: Up
Model probability of Up: 51.63%
```

Evaluate stored replay predictions:

```bash
python -m src.evaluate_predictions --mode replay
```

The demonstrated outcome was:

```text
Reference close: 333.3450
Target close: 332.8150
Actual: Not up
Correct: False
```

This single example verifies the workflow, not predictive performance. A newly downloaded snapshot may not contain this replay date.

## Live Mode

Run one prediction attempt:

```bash
python -m src.predict --mode live
```

Live mode fetches recent Yahoo Finance data and checks:

- The raw OHLCV candles pass basic sanity checks (no non-positive prices, no
  `High < Low`, `Open`/`Close` inside the `[Low, High]` range, no negative volume).
- A trading session exists for the evaluation date.
- The evaluation time is within regular trading hours.
- The latest selected candle has closed and belongs to the session.
- Its 30-minute target ends within the session.
- The completed candle is not older than the configured five-minute tolerance.
- Required features are available.

The NASDAQ calendar provides session opening and closing times, including scheduled early closes.

A skipped prediction can be expected behavior, for example:

```text
Skipped: raw price data failed quality checks: [...]
Skipped: evaluation time is outside trading hours.
Skipped: latest completed candle is stale.
Skipped: insufficient time remaining in session.
Skipped: latest candle is missing required features: [...]
```

The last one is expected for roughly the first 100 minutes of every session:
`volume_relative` needs 20 same-day 5-minute bars (`min_periods=20`) before
it has a value, so live predictions only start succeeding after about
11:10 ET. This is a feature-design limitation, not a bug — shortening it
would mean changing how `volume_relative` is computed (e.g. falling back to
a prior-day average early in the session).

`predict.py` also stores the OHLCV of the candle each prediction was based
on (`reference_open/high/low/close/volume` in the `predictions` table — see
[`docs/schema.md`](docs/schema.md)), so a prediction can be audited later
without re-downloading from Yahoo Finance, which does not keep old intraday
history indefinitely.

Each invocation of `predict.py --mode live` performs one attempt. To repeat it automatically:

```bash
python -m src.run_live_loop
```

This runs `predict.py --mode live` then `evaluate_predictions.py --mode
live` every 5 minutes (matching the candle interval), skipping cycles when
the market is closed. Every run, from every script, is logged to stdout
with a timestamp and level (`logging_config.py`); redirect to a file if you
want persistent logs, e.g. `python -m src.run_live_loop >> logs/live.log 2>&1`.

Stop it safely with Ctrl+C (or `kill <pid>` if run in the background): the
current cycle finishes — so a prediction or outcome write is never left
half-done — before the process exits.

Network calls to Yahoo Finance (in `ingest.py`, `predict.py --mode live`,
and `evaluate_predictions.py --mode live`) automatically retry up to 3 times
with a delay before giving up, since these requests occasionally fail
transiently.

“Live” does not guarantee exchange-level real-time data.

## Evaluate Live Outcomes

Run:

```bash
python -m src.evaluate_predictions --mode live
```

The evaluator:

1. Selects predictions without stored outcomes.
2. Waits until their target close time has passed.
3. Downloads the required historical candles.
4. Compares reference and target Close.
5. Saves the observed class and whether the prediction was correct.

If required candles are unavailable, the prediction remains pending. Old intraday data may no longer be available from the source.

## SQLite Storage

Full column-by-column schema, join example, and demo data pointer:
[`docs/schema.md`](docs/schema.md).

Database location:

```text
data/predictions/predictions.sqlite
```

A small checked-in sample with real replay data lives at
`data/demo/predictions_demo.sqlite`, for dashboard work that shouldn't
depend on running the pipeline first.

### `predictions`

Stores the ticker, interval, candle timestamps, evaluation time, target end time, mode, model version, horizon, predicted class, probability of Up, and the OHLCV of the candle the prediction was based on (for audit).

### `prediction_outcomes`

Stores the corresponding reference and target prices, observed class, correctness, and evaluation timestamp.

Both tables identify records using:

```text
ticker
interval
candle_start
mode
model_version
horizon_minutes
```

Repeated runs do not insert duplicates. Existing records are retained rather than overwritten.

- Stored timestamps use UTC.
- Probabilities use the range `0–1`.
- Correctness uses `1` for correct and `0` for incorrect.
- Replay and live records are distinguished by `mode`.
- Outcomes are stored only when required prices are available.

For a dashboard, use a left join from predictions to outcomes so pending records remain visible. Compute accuracy only from evaluated records and display the sample count.

## Original Experiment Results

**Historical note:** the numbers below are from an earlier run of this
pipeline using only 3–4 candle-shape features. The feature set was since
expanded to the eleven features listed above (technical indicators,
volatility, and relative volume), so a fresh run of `train.py` will produce
different row counts and accuracy figures. This section is kept as a record
of that experiment, not as a claim about the current model.

The original snapshot covered August 12–September 11, 2026:

- 22 observed trading dates
- 1,716 raw timestamps
- 1,518 retained AAPL training-table rows
- Train: 966 rows
- Validation: 276 rows
- Test: 276 rows

| Model | Validation accuracy | Test accuracy |
|---|---:|---:|
| Most-frequent-class baseline | 54.35% | 54.35% |
| Logistic Regression — 3 features | 49.64% | 51.09% |

Adding the 15-minute return produced 48.19% validation accuracy on the same retained observations and was not selected.

**The selected model did not outperform the baseline on accuracy.** These results do not establish a useful trading signal.

The test period had previously appeared in EDA. Further evaluation on unseen future sessions is needed. Overlapping 30-minute targets also mean adjacent examples are not independent.

## Time-Series and Cross-Stock Evaluation

A single train/validation/test split (Workflow A) gives one point-in-time
accuracy estimate, evaluated on as few as four sessions. Two additional
scripts give a more complete picture of how stable that estimate is.

### Walk-forward evaluation

```bash
python -m src.evaluate_walk_forward
```

Rolls a fixed-size training window (10 sessions by default) forward through
the whole processed AAPL table, testing on the following 2 sessions at each
step (`--train-sessions` / `--test-sessions` to change this). Each fold fits
its own baseline and model — a training row is dropped if its own target
resolves into the test window, so no fold leaks a future label. This reuses
`feature_utils`/`target_utils`, not `train.py`'s saved artifact: it evaluates
the *method*, not the deployed model.

On the current snapshot (20 sessions, 5 folds of 10-train/2-test), the model
beat the baseline in only 3 of 5 folds — mean accuracy 51.51% (±4.65 across
folds) vs. a 48.49% baseline. **This is the more trustworthy read of this
model's performance than the single-split figures above**: the improvement
over baseline is smaller than the fold-to-fold variation, so it should not
be treated as an established edge.

### Cross-stock evaluation

```bash
python -m src.evaluate_cross_stock
```

`ingest.py` downloads five tickers, but training only ever used AAPL. This
script builds the same features and target for the other four
(MSFT, NVDA, GOOGL, AMZN) and scores the AAPL-trained pipeline against a
same-ticker baseline. None of these tickers were in training, so no
train/test split is needed per ticker — the whole built dataset is
out-of-sample.

On the current snapshot, the AAPL model beat the same-ticker baseline on
only 2 of 4 other tickers — evidence it has not learned a pattern that
generalizes across symbols, consistent with the walk-forward result above.

## Modeling: Comparing Model Types

`models.py` defines model factories the walk-forward harness above can
swap in and out, so a second model type is evaluated on the exact same
folds as Logistic Regression rather than a separately-run, harder-to-compare
experiment.

**Gradient boosting, not XGBoost:** the second model is scikit-learn's
`HistGradientBoostingClassifier`, not XGBoost. XGBoost is a separate PyPI
package this environment could not install (no network access to PyPI at
the time), so `HistGradientBoostingClassifier` — already available via the
pinned `scikit-learn` dependency, and a comparably capable gradient-boosted
-tree model — was used instead to keep every result below actually run and
verified rather than speculative. Swapping in real XGBoost later only means
adding it to `requirements.txt` and changing the one factory function in
`models.py` that builds it; nothing else here depends on which library.

### Model comparison

```bash
python -m src.compare_models
```

Evaluates every model in `models.MODEL_FACTORIES` on identical walk-forward
folds (same train/test rows per fold, so any difference is attributable to
the model). On the current snapshot (5 folds): Logistic Regression won 4 of
5 folds head-to-head, with a higher mean accuracy (51.51% vs. 49.25%) and
better calibration (see below). With roughly 500 training rows per fold and
11 features, gradient boosting's extra capacity to fit interactions more
often fits noise than signal here — a linear model generalizes better on
data this size, which is a legitimate result, not a shortcoming of the
comparison.

### Feature importance and error analysis

```bash
python -m src.error_analysis
```

Fits each model once on a single held-out window (the last 4 sessions by
default) and reports, beyond a single accuracy number:

- A confusion matrix and full classification report.
- **Accuracy by hour of day** — on the current snapshot, both models are
  noticeably weaker in the last trading hour (15:00–16:00 ET) than mid-day.
- **Calibration** — predicted probability vs. actual Up-rate, bucketed.
  Logistic Regression is reasonably well calibrated (e.g. its lowest bucket
  predicts 33.6% and actually resolves Up 34.7% of the time). Gradient
  boosting is not (its lowest bucket predicts 27.1% but actually resolves
  Up 53.3% of the time) — a concrete instance of what this README already
  warns generally: **the probability the model reports is not its
  accuracy**, and that gap can be worse for some models than others.
- **Permutation importance**, computed on the held-out test set by
  shuffling one feature at a time and measuring the accuracy drop — chosen
  over Logistic Regression's own coefficients or a tree model's internal
  gain because those two are on different, not-comparable scales, and
  `HistGradientBoostingClassifier` does not expose a gain-based importance
  at all. On the current snapshot, `rsi_14` ranks first for both models —
  the one feature both model types agree matters most.

## Dashboard

```bash
python -m streamlit run src/dashboard.py
```

(`python -m streamlit`, not the bare `streamlit` command, so the repository
root is on the import path — same reason as `python -m src.<module>` above.)

A read-only view of the predictions database. It opens SQLite with
`mode=ro`, so the "dashboard must only read" rule in
[`docs/schema.md`](docs/schema.md) is enforced by SQLite rather than by
convention. Without a local live database it falls back to the checked-in
demo database, so it runs on a fresh clone.

- **Filters** (one row, top): database (live / demo), mode, model version.
- **Headline tiles**: prediction count; accuracy over *evaluated*
  predictions only, always shown with its sample size ("4 of 8 correct");
  pending predictions split into *awaiting target* (not due yet) vs. *not
  evaluated* (target passed, no outcome stored — run
  `evaluate_predictions.py`, or Yahoo no longer has the candle); and the
  share of predictions that said Up. The dashboard warns when every
  prediction in the selection is the same class — on the current live data
  the model predicted Up 14 of 14 times, so its 50% accuracy there says more
  about how often AAPL rose than about the model.
- **Predictions over time**: model probability of Up per candle, outcome
  shown by both color and marker shape, against the 0.5 decision threshold.
  One slot per candle rather than a continuous time axis, since overnight
  and weekend gaps would otherwise squash each session into one clump.
- **Running accuracy** over evaluated predictions, against 50%.
- **Model comparison**: the same walk-forward comparison as
  `python -m src.compare_models`, cached until the processed table changes.
  Skipped with an explanation if `data/processed/` hasn't been built.

Every chart has a table view underneath. Times are New York time.

## Verification Status

Verified through manual checks:

- Dependency installation and imports in a fresh virtual environment.
- Historical replay prediction and outcome matching.
- Duplicate prediction and outcome prevention.
- Missing-target handling.
- Stale-data rejection.
- Outside-hours and insufficient-session-time rejection.
- Replay in the cloned repository using the original artifacts.

Added since the initial pipeline-v1 build:

- Raw-candle sanity checks (no non-positive prices, `High < Low`, out-of-range Open/Close, negative volume) before a prediction is attempted.
- Retry with backoff on Yahoo Finance calls in `ingest.py`, `predict.py`, and `evaluate_predictions.py`.
- Structured, timestamped logging across all pipeline scripts (`logging_config.py`).
- Repeating live schedule with a graceful stop (`run_live_loop.py`).
- Prediction audit trail: the OHLCV of the candle each prediction used is stored alongside it.
- Model version bump (`v1` → `v2`) when the feature set changed.
- **A complete live prediction-to-outcome cycle**, run with `run_live_loop.py` during NASDAQ market hours on 2026-09-15: multiple live predictions were saved, `evaluate_predictions.py --mode live` correctly waited for each 30-minute target to mature ("Waiting for target close: N") before saving an outcome, and both a correct and an incorrect prediction were recorded (e.g. candle `15:30:00+00:00`: predicted Up, actual Up, correct; candle `15:25:00+00:00`: predicted Up, actual Not up, incorrect). `Ctrl+C` was also verified to stop the loop gracefully, finishing the in-flight cycle first.
- **An automated test suite** (`tests/`, run via `python -m unittest discover -s tests` and in CI on every push/PR): covers `feature_utils`, `target_utils`, `data_quality`, `prediction_rules`, the pure logic extracted from `predict.py` and `evaluate_predictions.py` (session-timing checks, SQLite insert/dedup, outcome matching), and `evaluation`'s walk-forward split generation.
- Shared raw-OHLCV quality checks (`data_quality.py`, including duplicate-timestamp and short-interval detection) now run in both `ingest.py` and `predict.py --mode live`, not live mode only.
- Walk-forward and cross-stock evaluation (`evaluate_walk_forward.py`, `evaluate_cross_stock.py` — see above) for a more complete accuracy picture than a single split.

Not yet fully verified:

- Fresh ingestion-to-training reproduction on another machine.
- Runtime behavior on early-close dates.
- Recovery during sustained source failures.

## Remaining Work

- Improve artifact versioning and remove date-specific experiment assumptions.
- Add portable environment packaging and a demo.
- Swap `models.build_gradient_boosting` for real XGBoost once this environment can reach PyPI, and re-run `compare_models.py`/`error_analysis.py` to see whether the comparison in "Modeling: Comparing Model Types" holds.
- `train.py` still trains and saves only Logistic Regression on a single split. Decide whether to fold the walk-forward/model-comparison methodology into it (or replace it) now that `compare_models.py` suggests Logistic Regression remains the better choice — currently the two live side by side rather than one replacing the other.

## Data Source

Market data is accessed through the independent `yfinance` library. Availability and latency depend on Yahoo Finance and the underlying exchange.

Review the applicable data-use terms before redistributing downloaded data or publishing a public data service:

- [yfinance project](https://github.com/ranaroussi/yfinance)
- [Yahoo terms](https://legal.yahoo.com/us/en/yahoo/terms/otos/index.html)

This project is an educational portfolio experiment, not an automated trading system.