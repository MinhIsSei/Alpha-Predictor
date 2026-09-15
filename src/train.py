from pathlib import Path
import pandas as pd

from sklearn.dummy import DummyClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report

project_root = Path(__file__).resolve().parent.parent
file_path = project_root / "data" / "processed" / "aapl_training_1mo.parquet"

df = pd.read_parquet(file_path).sort_index()

feature_cols = [
    "range_pct",
    "body_pct",
    "return_5m_pct",
    "return_15m_pct",
    "return_30m_pct",
    "volatility_30m",
    "volume_relative",
    "rsi_14",
    "macd_diff",
    "bb_width_pct",
    "atr_pct",
]
target_col = "target_up_30m"

#Identify the trading days present in the data
session_dates = df.index.normalize()
sessions = session_dates.unique().sort_values()

if len(sessions) < 10:
    raise ValueError("At least 10 sessions are required for this distribution method.")

validation_start = sessions[-8]
test_start = sessions[-4]

#Retain the end label only until the next stage
train = df[
    (df.index < validation_start)
    & (df["target_time"] < validation_start)
].copy()

validation = df[
    (df.index >= validation_start)
    & (df.index < test_start)
    & (df["target_time"] < test_start)
].copy()

test = df[df.index >= test_start].copy()

for name, subset in [
    ("Train", train),
    ("Validation", validation),
    ("Test", test),
]:
    print(f"\n{name}: {len(subset)} rows")
    print("From:", subset.index.min())
    print("To:", subset.index.max())
    print("Number of labels:")
    print(subset[target_col].value_counts())

# X: Input Features
# y: Predicted Label
X_train = train[feature_cols]
y_train = train[target_col]

X_validation = validation[feature_cols]
y_validation = validation[target_col]

# Learn the most frequent label from 'train'
baseline = DummyClassifier(strategy="most_frequent")
baseline.fit(X_train, y_train)

# Predict on 'validation'
baseline_predictions = baseline.predict(X_validation)

print("\nBaseline - Validation")
print("Predicted class:", baseline_predictions[0])
print(
    "Accuracy:",
    f"{accuracy_score(y_validation, baseline_predictions):.2%}"
)

print("Confusion matrix - rows: actual, columns: predicted [0, 1]")
print(
    confusion_matrix(
        y_validation,
        baseline_predictions,
        labels=[0, 1],
    )
)

model = Pipeline([
    ("scaler", StandardScaler()),
    ("classifier", LogisticRegression(max_iter=1000)),
])

# Only learn from 'train'
model.fit(X_train, y_train)

# Evaluate on the same validation set as the baseline
model_predictions = model.predict(X_validation)

print("\nLogisctic Regression - Validation")
print(
    "Accuracy:",
    f"{accuracy_score(y_validation, model_predictions):.2%}",
)

print("Confusion matrix - rows: actual, columns: predicted [0, 1]")
print(
    confusion_matrix(
        y_validation,
        model_predictions,
        labels=[0, 1],
    )
)

print(
    classification_report(
        y_validation,
        model_predictions,
        labels=[0, 1],
        target_names=["Not up", "Up"],
        digits=3,
        zero_division=0,
    )
)

validation_results = pd.DataFrame({
    "actual": y_validation,
    "baseline": baseline_predictions,
    "model": model_predictions,
})

validation_results["baseline_correct"] = (
    validation_results["baseline"] == validation_results["actual"]
)
validation_results["model_correct"] = (
    validation_results["model"] == validation_results["actual"]
)

daily_results = validation_results.groupby(
    validation_results.index.normalize()
).agg(
    samples=("actual", "size"),
    up_rate=("actual", "mean"),
    baseline_accuracy=("baseline_correct", "mean"),
    model_accuracy=("model_correct", "mean"),
)

print("\nValidation results by trading day:")
print(daily_results.round(3))

# Drill into the worst-performing validation day, whichever date that is.
worst_day = daily_results["model_accuracy"].idxmin()
day_results = validation_results.loc[
    validation_results.index.normalize() == worst_day
].copy()

print(f"\nWorst validation day ({worst_day.date()}) - Confusion matrix:")
print(
    confusion_matrix(
        day_results["actual"],
        day_results["model"],
        labels=[0, 1],
    )
)

# Compare the projected growth rate with the actual growth rate
print("Actual up rate:", f"{day_results['actual'].mean():.2%}")
print("Predicted up rate:", f"{day_results['model'].mean():.2%}")

# Check at what time the errors are concentrated
day_results["hour"] = day_results.index.hour

hourly_results = day_results.groupby("hour").agg(
    samples=("actual", "size"),
    actual_up_rate=("actual", "mean"),
    predicted_up_rate=("model", "mean"),
    accuracy=("model_correct", "mean"),
)

print("\nResults by hour - New York time:")
print(hourly_results.round(3))

X_test = test[feature_cols]
y_test = test[target_col]

baseline_test_predictions = baseline.predict(X_test)
model_test_predictions = model.predict(X_test)

print("\nFinal evaluation - Test")

print(
    "Baseline accuracy:",
    f"{accuracy_score(y_test, baseline_test_predictions):.2%}"
)

print(
    "Logistic accuracy:",
    f"{accuracy_score(y_test, model_test_predictions):.2%}"
)

print("\nConfusion matrix - rows: actual, columns: predicted [0, 1]")
print(
    confusion_matrix(
        y_test,
        model_test_predictions,
        labels=[0, 1]
    )
)

print(
    classification_report(
        y_test,
        model_test_predictions,
        labels=[0, 1],
        target_names=["Not up", "Up"],
        digits=3,
        zero_division=0,
    )
)

import joblib

model_dir = project_root / "models"
model_dir.mkdir(parents=True, exist_ok=True)

artifact = {
    "pipeline": model,
    "feature_cols": feature_cols,
    "ticker": "AAPL",
    "interval": "5m",
    "horizon_minutes": 30,
    "train_start": str(train.index.min()),
    "train_end": str(train.index.max()),
    "validation_accuracy": accuracy_score(y_validation, model_predictions),
    "test_accuracy": accuracy_score(y_test, model_test_predictions),
    "baseline_test_accuracy": accuracy_score(
        y_test, baseline_test_predictions
    ),
}

model_path = model_dir / "aapl_logistic_v1.joblib"
joblib.dump(artifact, model_path)

print("Saved model:", model_path)