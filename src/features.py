import pandas as pd
from pathlib import Path

from src.feature_utils import build_features
from src.target_utils import build_target

project_root = Path(__file__).resolve().parent.parent
file_path = project_root / "data" / "raw" / "stock-trend_1mo.parquet"

df = pd.read_parquet(file_path)

df_model = (
    df.xs("AAPL", axis=1, level="Ticker")
    .copy()
    .sort_index()
)
df_model = build_features(df_model)

# Horizon is 30 minutes = 6 bars at the 5-minute candle interval.
df_model = build_target(df_model, horizon_minutes=30, periods=6)

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