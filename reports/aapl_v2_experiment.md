# AAPL Logistic Regression v2 — Historical Experiment

## Objective

Predict whether AAPL's closing price will increase over the next 30 minutes within the same trading session, using completed five-minute candles.

Class 1 means Up; class 0 means Not up, including unchanged prices.

## Data and validation

The raw snapshot covers August 12–September 11, 2026, with 1,716 timestamps across 22 observed dates.

AAPL passed checks for missing and non-finite OHLCV values, non-positive prices, invalid OHLC relationships, negative volume, duplicate timestamps, and irregular within-day intervals. Each observed date contains 78 candles.

These checks do not independently verify source accuracy or completeness against an exchange calendar.

Feature and target generation retained 1,152 rows with 11 features. Targets are exactly 30 minutes ahead and remain within the same session date.

## Method

The model is a StandardScaler and LogisticRegression pipeline fitted on the training partition. The baseline is a DummyClassifier that always predicts the most frequent training class: Not up.

Partitions follow chronological session order. Training and validation targets end before the next partition begins.

| Partition | Dates in 2026 | Rows | Up labels |
|---|---|---:|---:|
| Train | August 12–31 | 728 | 358 |
| Validation | September 1–4 | 212 | 113 |
| Test | September 8–11 | 212 | 112 |

## Results

| Model | Validation accuracy | Test accuracy |
|---|---:|---:|
| Most-frequent-training-class baseline | 46.70% | 47.17% |
| Logistic Regression v2 | 56.60% | 52.83% |

The model exceeds the selected baseline by 9.90 percentage points on validation and 5.66 percentage points on test.

Test confusion matrix, with actual classes as rows and predicted classes as columns, ordered [Not up, Up]:

| Actual / Predicted | Not up | Up |
|---|---:|---:|
| Not up | 52 | 48 |
| Up | 52 | 60 |

The model correctly classifies 112 of 212 test rows. Always predicting Up would also achieve 52.83% on this test partition. This is a post-hoc comparison, not a replacement for the baseline selected from training data.

## Interpretation and limitations

This run demonstrates training and evaluation on a preserved historical snapshot. It does not establish reliable predictive performance or trading profitability.

- The test partition contains only four sessions.
- Overlapping 30-minute targets make adjacent observations dependent.
- This historical period was previously examined; it is not a new, untouched evaluation period.
- Accuracy does not measure probability calibration.
- The retained samples exclude feature warm-up periods and rows without valid targets.
- Differences from earlier experiments cannot be attributed solely to added features because the retained evaluation rows differ.

Further evaluation should use a fixed model on later, previously unexamined sessions.

## Reproducibility artifacts

A local experiment archive contains the raw snapshot, processed training table, saved model, training log, environment package list, and a manifest with artifact hashes and the Git state recorded at archive time.

The archive is local and is not bundled with this report. The recorded environment and Git state describe the archive-time state; they are not a substitute for an independent rerun.