# Alpha-Predictor

A Python pipeline that predicts whether AAPL's closing price will rise over the
next 30 minutes, from five-minute Yahoo Finance candles. It covers data
ingestion and validation, feature engineering, leakage-safe time-series
evaluation, live and historical-replay prediction with an SQLite audit trail,
outcome scoring, and a read-only Streamlit dashboard.

**Status:** the pipeline is complete and has run a full live
prediction-to-outcome cycle. **The model has not shown a reliable edge over a
naive baseline** — see the results below. This is an educational portfolio
project, not a trading system.

## Results at a glance

On the current local snapshot (August 17 – September 14, 2026, 20 trading
sessions). Full write-up: [`reports/aapl_v2_evaluation_update.md`](reports/aapl_v2_evaluation_update.md).

| Evaluation | Result |
|---|---|
| Walk-forward, Logistic Regression (5 folds: train 10 sessions, test the next 2) | 51.5% mean accuracy (±4.7 across folds) vs. 48.5% for a most-frequent-class baseline; beat the baseline in 3 of 5 folds |
| Same folds, gradient boosting | 49.3% mean accuracy; Logistic Regression won 4 of 5 folds head-to-head |
| AAPL model applied to MSFT, NVDA, GOOGL, AMZN | Beat each ticker's own baseline on 2 of 4 |
| Live predictions (September 15, 2026) | 9 of 12 correct (75%) — but the model predicted Up all 12 times and AAPL rose in exactly 9 of them, so this matches an always-Up rule |

The gain over the baseline is smaller than the variation between folds, and the
live sample shows no discrimination between up and down moves. Logistic
Regression remains the deployed model because it outperformed the
gradient-boosting alternative, not because it has been shown to work.

## Quick start

```bash
git clone https://github.com/MinhIsSei/Alpha-Predictor.git
cd Alpha-Predictor
python -m venv .venv
```

Activate the environment — macOS/Linux: `source .venv/bin/activate`;
Windows PowerShell: `.venv\Scripts\Activate.ps1`. Then install the project
with its pinned dependencies and developer tools:

```bash
python -m pip install -e ".[dev]"
```

(`python -m pip install -e .` installs the runtime set only, without the
notebook libraries and linter.)

Open the dashboard — it works on a fresh clone from the checked-in demo data:

```bash
streamlit run streamlit_app.py
```

Run the tests:

```bash
python -m unittest discover -s tests
```

Pipeline scripts are run as modules from the repository root, e.g.
`python -m src.train`. Developed on Windows 11 with Python 3.11 (replay was
also verified on macOS); CI runs the tests on Ubuntu with Python 3.11 and 3.13.

## Prediction task

- **Ticker:** AAPL, five-minute candles.
- **Horizon:** 30 minutes, within the same trading session.
- **Class 1 — Up:** the Close 30 minutes later is greater than the reference Close.
- **Class 0 — Not up:** it is less than or equal to it.

Predictions use completed candles only. A candle timestamped `14:00` closes at
`14:05`; its prediction compares that Close with the Close of the `14:30`
candle, which closes at `14:35`.

The probability the model reports is its score, not its accuracy.

## Project structure

```text
src/
├── config.py                # Paths, model version, feature list — the one place to change them
├── ingest.py                # Download OHLCV data (archives the previous snapshot)
├── data_quality.py          # Raw-OHLCV sanity checks
├── feature_utils.py         # The 11 features, shared by training and prediction
├── target_utils.py          # Same-session 30-minute target
├── features.py              # Build the training table
├── models.py                # Model definitions (Logistic Regression, gradient boosting)
├── evaluation.py            # Leakage-safe walk-forward evaluation harness
├── train.py                 # Walk-forward estimate + train and save the model
├── artifacts.py             # Archiving and provenance (hashes, git commit)
├── predict.py               # Replay/live prediction into SQLite
├── prediction_rules.py      # Session cut-off rule
├── evaluate_predictions.py  # Score predictions against observed prices
├── run_live_loop.py         # predict + evaluate every 5 minutes during market hours
├── evaluate_walk_forward.py # CLI: walk-forward report
├── evaluate_cross_stock.py  # CLI: AAPL model on other tickers
├── compare_models.py        # CLI: model types on identical folds
├── error_analysis.py        # CLI: confusion matrix, calibration, permutation importance
├── dashboard_data.py        # Dashboard data layer (read-only, no Streamlit import)
├── dashboard.py             # Dashboard UI
├── logging_config.py        # Shared logging format
└── net_utils.py             # Retry helper for Yahoo Finance calls

streamlit_app.py             # Dashboard entry point (also what Streamlit Cloud runs)
tests/                       # unittest suite, run in CI
notebooks/                   # exploration.ipynb (EDA), AI_Brain.ipynb (feature walkthrough)
reports/                     # Experiment write-ups
docs/schema.md               # SQLite schema for the predictions database
data/demo/                   # Checked-in demo data for the dashboard
pyproject.toml               # Package metadata and ruff configuration
requirements.txt             # Pinned runtime dependencies
requirements-dev.txt         # Pinned notebook and developer tools
```

Generated data (`data/raw`, `data/processed`, `data/predictions`) and model
files (`models/`) are local and not in Git. The exception is `data/demo/`: a
copy of the predictions database and a precomputed model comparison, so the
dashboard has something to show on a fresh clone or a cloud deploy.

## Configuration and versioning

`src/config.py` holds every path, the feature list, the target, the
walk-forward window, and `MODEL_VERSION`. The model file name
(`models/<MODEL_VERSION>.joblib`) and the `model_version` stored with every
prediction both come from it, so **bumping the version is a one-line change**.
Bump it whenever the model or the feature set changes. `predict.py` refuses to
run if the saved model records a different version than `config.py` expects.

`ingest.py`, `features.py` and `train.py` write to fixed paths that the rest of
the pipeline reads. Before overwriting, each moves the previous file into an
`archive/` folder next to it, named by when that file was written (for example
`models/archive/aapl_logistic_v2_20260915T121600Z.joblib`), so rerunning an
experiment never destroys the last one. Each also accepts `--input`/`--output`
(`--model-path` for `train.py`) to write somewhere else entirely.

Saved models carry provenance: training time, the SHA-256 of the training
table, the git commit (marked `-dirty` if there were uncommitted changes), and
the walk-forward summary.

## Data and features

`ingest.py` downloads one month of five-minute candles for AAPL, MSFT, NVDA,
GOOGL and AMZN and runs quality checks on each ticker (non-positive prices,
`High < Low`, Open/Close outside `[Low, High]`, negative volume, duplicate
timestamps, sub-interval gaps). Training and prediction use AAPL; the other
four are used for cross-stock evaluation.

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

Return features are masked when their window crosses a session gap
(overnight or weekend), and volatility and volume are computed per trading
day, so no feature mixes non-adjacent bars. Targets and future prices never
enter the model's inputs. Rows missing any feature — including indicator
warm-up rows — are dropped before training.

## Workflow A — create a new experiment

```bash
python -m src.ingest     # data/raw/stock-trend_1mo.parquet
python -m src.features   # data/processed/aapl_training_1mo.parquet
python -m src.train      # models/aapl_logistic_v2.joblib
```

`train.py` reports two evaluations, for two different questions:

1. **Walk-forward** — how well the *method* generalizes. A 10-session training
   window rolls forward through the data, each time tested on the next 2
   sessions; every fold refits its own model and baseline. A training row is
   dropped if its target resolves inside the test window, so no fold sees a
   future label. This is the estimate to trust.
2. **A single chronological split** (earlier sessions train, four validation,
   the final four test) — produces the saved model, with a final holdout check
   and per-day and per-hour diagnostics on the validation period.

The saved model is fit on the train partition of (2), so retraining on the
same snapshot reproduces the deployed model. A one-month download is rolling:
a new download will not reproduce the dates, row counts or scores above. Do
not tune repeatedly against the test set.

The model is a scikit-learn pipeline of `StandardScaler` and
`LogisticRegression`, compared against a `DummyClassifier` that predicts the
most frequent training class.

## Workflow B — historical replay

Replay needs the original snapshot and model at `data/raw/stock-trend_1mo.parquet`
and `models/aapl_logistic_v2.joblib` (ask the maintainers; only load trusted
model files).

```bash
python -m src.predict --mode replay --as-of "2026-09-11 14:05:00"
python -m src.evaluate_predictions --mode replay
```

A timezone-naive `--as-of` is read as New York time; candles closing after it
are excluded. With the `v2` model this predicts **Up** (model probability
51.63%) for the 14:00 candle; the outcome was **Not up** (reference close
333.3450, target close 332.8150). One example verifies the workflow, not the
model.

## Live mode

```bash
python -m src.predict --mode live        # one attempt
python -m src.run_live_loop              # predict + evaluate every 5 minutes
python -m src.evaluate_predictions --mode live
```

Before predicting, live mode checks that the candles pass the quality checks,
a NASDAQ session exists (including scheduled early closes), the time is within
regular hours, the latest candle has closed, belongs to the session and is at
most five minutes old, its 30-minute target ends within the session, and every
feature is available. A skipped attempt logs why, for example:

```text
Skipped: evaluation time is outside trading hours.
Skipped: latest candle is missing required features: ['volume_relative'].
```

The second is expected for about the first 100 minutes of each session:
`volume_relative` needs 20 same-day bars, so live predictions start around
11:10 ET.

Each prediction stores the OHLCV of the candle it used, so it can be audited
without re-downloading. The evaluator scores predictions whose target has
passed; **run it within about 60 days**, since Yahoo Finance stops serving
five-minute candles after that and those predictions can then never be scored.

`run_live_loop` skips cycles while the market is closed and stops cleanly on
Ctrl+C, finishing the current cycle first. Yahoo Finance calls retry up to three
times. "Live" does not mean exchange-level real-time data.

## Evaluation and modeling tools

```bash
python -m src.evaluate_walk_forward   # walk-forward report for Logistic Regression
python -m src.evaluate_cross_stock    # the AAPL model on MSFT, NVDA, GOOGL, AMZN
python -m src.compare_models          # model types on identical folds
python -m src.error_analysis          # confusion matrix, calibration, importance
```

- **Model comparison.** Every model in `models.MODEL_FACTORIES` is evaluated on
  identical walk-forward folds, so differences come from the model, not the
  data. The alternative is scikit-learn's `HistGradientBoostingClassifier`;
  XGBoost could not be installed when this was built and was not pursued once
  Logistic Regression came out ahead.
  `python -m src.compare_models --output data/demo/model_comparison_demo.csv`
  refreshes the dashboard's demo snapshot.
- **Error analysis** fits each model on all but the last four sessions and
  inspects the held-out errors: accuracy by hour (both models are weakest in the
  last trading hour), calibration (Logistic Regression's predicted
  probabilities are close to observed Up-rates; gradient boosting's are not),
  and permutation importance, measured as the accuracy drop on held-out data
  when a feature is shuffled. `rsi_14` ranks first for both models.

## Dashboard

```bash
streamlit run streamlit_app.py
```

A read-only view of the predictions database. SQLite is opened with
`mode=ro`, so the dashboard cannot write even by mistake. It shows headline
figures (accuracy is computed over evaluated predictions only, always with its
sample size), a warning when every prediction in the selection is the same
class, each prediction's probability and outcome, running accuracy, and the
walk-forward model comparison. Every chart has a table view; times are New
York time.

Without a local live database or processed training table it uses the
checked-in demo data, which is what a cloud deployment sees.

**Deploying to Streamlit Community Cloud:** at
[share.streamlit.io](https://share.streamlit.io), create an app from this
repository's `main` branch with main file `streamlit_app.py`, and choose Python
3.11 under *Advanced settings*. It installs `requirements.txt`. The
demo-only scenario is covered by a test; the hosted deploy itself has not yet
been run.

## SQLite storage

Predictions go to `data/predictions/predictions.sqlite`: table `predictions`
(one row per successful attempt) and `prediction_outcomes` (one row per scored
prediction), both keyed on ticker, interval, candle start, mode, model version
and horizon. Reruns never insert duplicates or overwrite existing rows.
Timestamps are UTC. Column-by-column schema and a join example:
[`docs/schema.md`](docs/schema.md).

## Development

```bash
python -m unittest discover -s tests   # 89 tests
python -m ruff check .                 # lint
python -m ruff format .                # format (line length 100)
```

CI (`.github/workflows/tests.yml`) runs ruff on every push and pull request,
and the test suite on Python 3.11 and 3.13, including an editable install of
the package.

## Verification status

Verified:

- Historical replay and outcome matching; duplicate prevention; missing-target,
  stale-data, outside-hours and end-of-session rejection.
- A complete live cycle during NASDAQ hours on 2026-09-15, including waiting for
  targets to mature and a graceful Ctrl+C stop; the remaining outcomes were
  scored on 2026-09-30.
- Every pipeline entry point after the move to `config.py` and archiving: the
  rebuilt training table is identical to the previous one, and replay
  reproduces the result above.
- Dashboard rendering in light and dark mode, and with demo data only.

Not yet verified:

- Fresh ingestion-to-training reproduction on another machine.
- Behavior on early-close dates, and recovery from sustained source failures.
- The Streamlit Community Cloud deployment.

## Earlier results

Before the feature set grew from 3–4 candle-shape features to the current
eleven (the `v1` → `v2` bump), a single split of an August 12 – September 11
snapshot gave Logistic Regression 49.64% validation / 51.09% test accuracy
against a 54.35% baseline: it did not beat the baseline. The `v2` single-split
experiment is documented in [`reports/aapl_v2_experiment.md`](reports/aapl_v2_experiment.md).

## Remaining work

- **Model quality**, the main open problem. Options, roughly in order of
  expected value: more history (Yahoo keeps only about 60 days of five-minute
  data, so this needs another data source); market-context features such as
  SPY/QQQ returns or VIX; time-of-day features; a target with a "flat" band
  that drops near-zero moves; hyperparameter tuning on the walk-forward
  harness; probability calibration.
- Decide whether the saved model should be refit on all sessions rather than
  the train partition. That uses more recent data but changes the model, so it
  needs a version bump and a fresh live comparison.
- Optional: PostgreSQL instead of SQLite, scheduled ingestion, more tickers.

## Data source

Market data comes through the independent `yfinance` library; availability
and latency depend on Yahoo Finance and the exchange. Review the data-use terms
before redistributing downloaded data or publishing a data service:

- [yfinance project](https://github.com/ranaroussi/yfinance)
- [Yahoo terms](https://legal.yahoo.com/us/en/yahoo/terms/otos/index.html)

This project is an educational portfolio experiment, not an automated trading
system.
