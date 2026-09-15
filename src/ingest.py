from pathlib import Path
import yfinance as yf

project_root = Path(__file__).resolve().parent.parent
raw_dir = project_root / "data" / "raw"
raw_dir.mkdir(parents=True, exist_ok=True)

tickers = ["AAPL", "MSFT", "NVDA", "GOOGL", "AMZN"]

df = yf.download(
    tickers=tickers,
    period="1mo",
    interval="5m",
    auto_adjust=True,
)

if df.empty:
    raise ValueError("Yahoo Finance returned no data.")

file_path = raw_dir / "stock-trend_1mo.parquet"
df.to_parquet(file_path, engine="pyarrow", index=True)

print("Saved:", file_path)
print("Rows:", len(df))
print("Trading dates:", df.index.normalize().nunique())
print("From:", df.index.min())
print("To:", df.index.max())

print("\nMissing values by column:")
print(df.isna().sum())