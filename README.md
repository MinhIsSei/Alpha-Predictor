# Alpha-Predictor — Pipeline v1

A Python project for predicting whether AAPL's closing price will increase over the next 30 minutes, using five-minute market data from Yahoo Finance.

This version includes data ingestion, feature engineering, chronological model evaluation, historical replay, live-data prediction, and SQLite storage for predictions and outcomes.

**Current status:** historical replay has been verified. A complete live prediction-to-outcome cycle during market hours remains unverified. The model has not outperformed the baseline on the evaluated test set.

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
├── features.py              # Build the training dataset and targets
├── train.py                 # Train, evaluate, and save the model
├── predict.py               # Run replay/live predictions and store results
└── evaluate_predictions.py  # Match predictions with observed outcomes

notebooks/
└── exploration.ipynb        # Exploratory data analysis

data/
├── raw/
├── processed/
└── predictions/
    └── predictions.sqlite

models/
└── aapl_logistic_v1.joblib

requirements.txt
.gitignore
README.md
```

Data, model artifacts, and SQLite databases are generated locally and are not included in Git.

## Setup

Clone this branch:

```bash
git clone --branch pipeline-v1 https://github.com/MinhIsSei/Alpha-Predictor.git
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

Run all following commands from the repository root.

Dependencies were installed and replay was verified in a fresh virtual environment on the developer's Mac. Other platforms have not yet been verified.

## Data and Features

The ingestion script downloads one month of five-minute OHLCV data for:

```text
AAPL, MSFT, NVDA, GOOGL, AMZN
```

Training and prediction currently use AAPL only.

The saved model uses three features:

| Feature | Definition |
|---|---|
| `range_pct` | `(High - Low) / Open × 100` |
| `body_pct` | `(Close - Open) / Open × 100` |
| `return_5m_pct` | Percentage change in Close over exactly five minutes |

The feature builder also calculates `return_15m_pct`. This feature is retained in the processed dataset but excluded from the selected model.

The current training table drops rows missing any of the four generated features, so the three-feature model uses the same retained observations as the four-feature experiment.

Targets and future prices are never included in model inputs.

## Workflow A — Create a New Experiment

Download data:

```bash
python src/ingest.py
```

Output:

```text
data/raw/stock-trend_1mo.parquet
```

Build features and targets:

```bash
python src/features.py
```

Output:

```text
data/processed/aapl_training_1mo.parquet
```

Train and evaluate:

```bash
python src/train.py
```

Model output:

```text
models/aapl_logistic_v1.joblib
```

The training script uses chronological session splits:

- Earlier sessions: training
- Four sessions before test: validation
- Final four sessions: test

Training labels must end before the next partition begins.

The model is a scikit-learn Pipeline containing `StandardScaler` and `LogisticRegression`. It is compared with a `DummyClassifier` that predicts the most frequent training class.

### Current experiment limitations

These scripts use fixed output filenames. Running ingestion, feature generation, or training again can overwrite existing artifacts. Preserve the original snapshot and model before starting a new experiment.

A rolling one-month download changes over time. It will not necessarily reproduce the dates, row counts, or scores reported below.

The current training script also contains date-specific diagnostics and experiment metadata from the original run. Review these before using a new date range. Update the model version in both the artifact filename and prediction-storage metadata when releasing a new model.

Do not repeatedly tune against the test set.

## Workflow B — Historical Replay

To reproduce the demonstrated replay, obtain the matching original artifacts from the project maintainers and place them at:

```text
data/raw/stock-trend_1mo.parquet
models/aapl_logistic_v1.joblib
```

Only load trusted model artifacts.

Run:

```bash
python src/predict.py --mode replay --as-of "2026-09-11 14:05:00"
```

A timezone-naive `--as-of` value is interpreted as New York time. Observations whose candles close after that time are excluded.

With the original artifacts, the demonstrated result is:

```text
Candle start: 2026-09-11 14:00:00-04:00
Prediction: Not up
Model probability of Up: 49.98%
```

Evaluate stored replay predictions:

```bash
python src/evaluate_predictions.py --mode replay
```

The demonstrated outcome was:

```text
Reference close: 333.3450
Target close: 332.8150
Actual: Not up
Correct: True
```

This single example verifies the workflow, not predictive performance. A newly downloaded snapshot may not contain this replay date.

## Live Mode

Run one prediction attempt:

```bash
python src/predict.py --mode live
```

Live mode fetches recent Yahoo Finance data and checks:

- A trading session exists for the evaluation date.
- The evaluation time is within regular trading hours.
- The latest selected candle has closed and belongs to the session.
- Its 30-minute target ends within the session.
- The completed candle is not older than the configured five-minute tolerance.
- Required features are available.

The NASDAQ calendar provides session opening and closing times, including scheduled early closes.

A skipped prediction can be expected behavior, for example:

```text
Skipped: evaluation time is outside trading hours.
Skipped: latest completed candle is stale.
Skipped: insufficient time remaining in session.
```

Each invocation performs one attempt. Automatic scheduling is not yet implemented. “Live” does not guarantee exchange-level real-time data.

## Evaluate Live Outcomes

Run:

```bash
python src/evaluate_predictions.py --mode live
```

The evaluator:

1. Selects predictions without stored outcomes.
2. Waits until their target close time has passed.
3. Downloads the required historical candles.
4. Compares reference and target Close.
5. Saves the observed class and whether the prediction was correct.

If required candles are unavailable, the prediction remains pending. Old intraday data may no longer be available from the source.

## SQLite Storage

Database location:

```text
data/predictions/predictions.sqlite
```

### `predictions`

Stores the ticker, interval, candle timestamps, evaluation time, target end time, mode, model version, horizon, predicted class, and probability of Up.

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

## Verification Status

Verified through manual checks:

- Dependency installation and imports in a fresh virtual environment.
- Historical replay prediction and outcome matching.
- Duplicate prediction and outcome prevention.
- Missing-target handling.
- Stale-data rejection.
- Outside-hours and insufficient-session-time rejection.
- Replay in the cloned repository using the original artifacts.

Not yet fully verified:

- A complete live prediction-to-outcome cycle during market hours.
- Fresh ingestion-to-training reproduction on another machine.
- Runtime behavior on early-close dates.
- Recovery during sustained source failures.

## Remaining Work

- Complete in-session live verification.
- Strengthen automated data-quality checks and regression tests.
- Add retry handling, structured logs, and scheduling.
- Preserve source data needed to audit predictions.
- Improve artifact versioning and remove date-specific experiment assumptions.
- Build a read-only dashboard.
- Add portable environment packaging and a demo.

## Data Source

Market data is accessed through the independent `yfinance` library. Availability and latency depend on Yahoo Finance and the underlying exchange.

Review the applicable data-use terms before redistributing downloaded data or publishing a public data service:

- [yfinance project](https://github.com/ranaroussi/yfinance)
- [Yahoo terms](https://legal.yahoo.com/us/en/yahoo/terms/otos/index.html)

This project is an educational portfolio experiment, not an automated trading system.