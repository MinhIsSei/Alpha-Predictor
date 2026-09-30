# AAPL Logistic Regression v2 — Evaluation Update

This update re-evaluates the `aapl_logistic_v2` model and its training method
with methods the [original v2 report](aapl_v2_experiment.md) lacked:
walk-forward evaluation, a comparison with a second model type, evaluation on
other tickers, error analysis, and the first scored live predictions.

**Conclusion:** the method does not show a reliable edge over a
most-frequent-class baseline. Its apparent advantage comes from folds where the
baseline guessed the wrong majority class, and every live prediction was Up.
Logistic Regression remains the deployed model because gradient boosting did
worse on the same folds.

## Task

Predict whether AAPL's close rises over the next 30 minutes within the same
session, from completed five-minute candles. Class 1 is Up; class 0 is Not up,
including unchanged prices. The eleven features and the target are unchanged
from the v2 report.

## Data

The snapshot covers August 17 – September 14, 2026: 20 trading sessions, 1,560
five-minute timestamps per ticker. After feature warm-up and dropping rows
without a same-session target, 1,046 AAPL rows remain. All five tickers pass the
data quality checks (non-positive prices, invalid OHLC relationships, negative
volume, duplicate timestamps, sub-interval gaps).

This is a different window from the v2 report's (August 12 – September 11), so
figures are not directly comparable. The deployed model was trained on the
earlier window; the walk-forward and comparison results below refit models on
this one.

## Method

**Walk-forward evaluation.** A 10-session training window rolls forward through
the 20 sessions in steps of 2, each time tested on the following 2 sessions:
five folds, each with about 530 training rows and 106 test rows. Every fold
refits its own model and baseline. A training row is dropped if its target
resolves inside the test window, so no label from the test period reaches
training.

**Baseline.** A `DummyClassifier` that predicts the most frequent class in each
fold's training data.

**Models.** The deployed pipeline (`StandardScaler` + `LogisticRegression`) and
scikit-learn's `HistGradientBoostingClassifier` (maximum depth 3, learning rate
0.05, 200 iterations, no tuning), evaluated on identical folds.

## Results

### Walk-forward

| Fold | Test sessions | Up rate | Baseline | Logistic Regression | Gradient boosting |
|---|---|---:|---:|---:|---:|
| 1 | Aug 31 – Sep 1 | 54.7% | 45.3% | 46.2% | 45.3% |
| 2 | Sep 2 – Sep 3 | 47.2% | 52.8% | 50.9% | 48.1% |
| 3 | Sep 4 – Sep 8 | 60.4% | 39.6% | 57.5% | 50.9% |
| 4 | Sep 9 – Sep 10 | 62.3% | 62.3% | 48.1% | 54.7% |
| 5 | Sep 11 – Sep 14 | 42.5% | 42.5% | 54.7% | 47.2% |
| **Mean (std)** | | | **48.5%** (9.1) | **51.5%** (4.7) | **49.3%** (3.7) |

Logistic Regression beat the baseline in 3 of 5 folds and beat gradient boosting
in 4 of 5. Gradient boosting beat the baseline in 2 folds and tied it in one.
Mean ROC-AUC was 0.54 for Logistic Regression and 0.49 for gradient boosting.

The baseline's accuracy is the test Up rate when the training majority was Up,
and one minus it when the majority was Not up. In folds 1, 3 and 5 the training
majority was the opposite of the test majority, so the baseline scored below
50% — and those are exactly the three folds Logistic Regression won. Where the
baseline guessed the majority correctly (folds 2 and 4), Logistic Regression
lost to it. The mean gain of 3.0 points is also smaller than the fold-to-fold
standard deviation of either model.

### Other tickers

The AAPL model, unchanged, applied to the same window for tickers it never saw
in training, against each ticker's own most-frequent-class baseline:

| Ticker | Rows | Up rate | Baseline | AAPL model |
|---|---:|---:|---:|---:|
| MSFT | 1,046 | 49.8% | 50.2% | 51.6% |
| NVDA | 1,046 | 45.4% | 54.6% | 46.6% |
| GOOGL | 1,046 | 52.6% | 52.6% | 53.5% |
| AMZN | 1,046 | 52.3% | 52.3% | 52.1% |

It beat the baseline on 2 of 4 tickers, by at most 1.4 points, and lost by 8.0
points on NVDA. The baseline here is fit on each ticker's own labels, which
favors it slightly.

### Error analysis

Both models fit on the first 16 sessions and scored on the last four
(September 9–14, 212 rows): Logistic Regression 56.6%, gradient boosting
51.9%. On this single window Logistic Regression looks better than its
walk-forward average, which is why walk-forward is the figure to rely on.

**By hour.** Both models were most accurate at 11:00 (75% of 44 rows) and least
accurate at 15:00 (29% and 33% of 24 rows), the last hour in which a 30-minute
target still fits in the session.

**Calibration.** Predicted probability against the observed Up rate:

| Predicted probability | LR rows | LR predicted | LR observed | GB rows | GB predicted | GB observed |
|---|---:|---:|---:|---:|---:|---:|
| 0 – 40% | 49 | 33.6% | 34.7% | 75 | 27.1% | 53.3% |
| 40 – 50% | 58 | 45.2% | 55.2% | 34 | 45.5% | 44.1% |
| 50 – 60% | 78 | 54.4% | 59.0% | 31 | 55.0% | 38.7% |
| 60 – 100% | 27 | 68.0% | 59.3% | 72 | 72.7% | 61.1% |

For Logistic Regression the observed Up rate rises across its buckets, though
it understates the middle buckets and overstates the top one (68% predicted,
59% observed). Gradient boosting's buckets are not ordered: rows it scored
below 40% went up 53% of the time, more than those it scored 50–60%.

**Feature importance.** Measured as the drop in held-out accuracy when one
feature is shuffled (mean of 20 shuffles). `rsi_14` ranked first for both
models (7.1 points for Logistic Regression, 4.3 for gradient boosting).
`volatility_30m` (4.3) and `atr_pct` (2.6) followed for Logistic Regression.
`macd_diff` had a negative score for both, meaning shuffling it slightly
improved accuracy.

### Live predictions

The deployed model made 12 live predictions on September 15, 2026, between
11:05 and 12:00 New York time, scored on September 15 and 30.

| | Predictions | Predicted Up | Actually Up | Correct |
|---|---:|---:|---:|---:|
| Live | 12 | 12 | 9 | 9 (75%) |
| Replay | 2 | 2 | 1 | 1 (50%) |

Every prediction was Up, so live accuracy equals the share of candles that
rose. The model's probabilities ranged from 54% to 72%. Twelve consecutive
five-minute candles in one hour have overlapping 30-minute targets and are not
independent observations.

## Interpretation and limitations

- The evidence does not support a predictive edge. The walk-forward gain is
  smaller than its own variability and comes from folds where the baseline
  mispredicted the majority; the live predictions do not distinguish up from
  down moves.
- Twenty sessions is little data. Each walk-forward test window is two
  sessions, and adjacent rows share overlapping targets.
- This window overlaps data examined during earlier development; it is not an
  untouched evaluation period.
- Gradient boosting was not tuned. Its result shows that added model capacity
  alone does not help on data this size, not that tree models cannot work here.
- Yahoo Finance serves five-minute data for about 60 days, which caps the
  history available to this pipeline.

## Next steps

Longer history from another data source, market-context features (index
returns, volatility indices), time-of-day features, and a target that excludes
near-zero moves are more likely to help than a different model on the same
data. Any change should be judged on the walk-forward harness against the
baseline, reporting per-fold results rather than the mean alone.

## Reproduction

With the snapshot at `data/raw/stock-trend_1mo.parquet`:

```bash
python -m src.features
python -m src.evaluate_walk_forward
python -m src.compare_models
python -m src.evaluate_cross_stock
python -m src.error_analysis
```

Live results are in `data/predictions/predictions.sqlite`; a copy is checked in
as `data/demo/predictions_demo.sqlite`.
