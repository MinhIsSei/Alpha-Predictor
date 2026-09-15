import pandas as pd
from pathlib import Path

from feature_utils import build_features

project_root = Path(__file__).resolve().parent.parent
file_path = project_root / "data" / "raw" / "stock-trend_1mo.parquet"

df = pd.read_parquet(file_path)

df_model = (
    df.xs("AAPL", axis=1, level="Ticker")
    .copy()
    .sort_index()
)
df_model = build_features(df_model)

timestamps = df_model.index.to_series()

future_close = df_model["Close"].shift(-6)
future_time = timestamps.shift(-6)

#Labels are accepted only 
#if they are exactly 30 minutes old and from the same trading day
valid_target = (
    (future_time - timestamps).eq(pd.Timedelta(minutes=30))
    & future_time.dt.normalize().eq(timestamps.dt.normalize())
)

df_model["future_close_30m"] = future_close.where(valid_target)

df_model["target_up_30m"] = (
    (future_close > df_model["Close"])
    .astype("Int64")
    .where(valid_target)
)

feature_cols = [
    "range_pct",
    "body_pct",
    "return_5m_pct",
    "return_15m_pct",
]

target_col = "target_up_30m"

#The timing of futures prices for checking the train/test boundary
df_model["target_time"] = future_time.where(valid_target)

#Only keep items that have sufficient characteristics and labels
training_data = df_model[
    feature_cols + [target_col, "target_time"]
].dropna(subset=feature_cols + [target_col, "target_time"]).copy()

#Convert to a pure integer format so the ML library works
training_data[target_col] = training_data[target_col].astype(int)

print("Number of rows before filter:", len(df_model))
print("Number of rows for training:", len(training_data))
print(training_data.head())

#Save table
processed_dir = project_root / "data" / "processed"
processed_dir.mkdir(parents=True, exist_ok=True)

output_path = processed_dir / "aapl_training_1mo.parquet"
training_data.to_parquet(output_path, index=True)

print(f"Saved: {output_path}")