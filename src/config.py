"""Single source of truth for paths, the model version, and the feature set.

These used to be repeated across ingest.py, features.py, train.py,
predict.py and the evaluation scripts, so bumping the model version (which
docs/schema.md requires whenever the model or feature set changes) meant
editing the artifact filename in train.py, the load path in predict.py, and
the stored version string in predict.py separately — easy to get out of sync.
"""

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"

RAW_PRICES_PATH = DATA_DIR / "raw" / "stock-trend_1mo.parquet"
PROCESSED_PATH = DATA_DIR / "processed" / "aapl_training_1mo.parquet"
PREDICTIONS_DB_PATH = DATA_DIR / "predictions" / "predictions.sqlite"
DEMO_DIR = DATA_DIR / "demo"
DEMO_DB_PATH = DEMO_DIR / "predictions_demo.sqlite"
DEMO_MODEL_COMPARISON_PATH = DEMO_DIR / "model_comparison_demo.csv"

INGEST_TICKERS = ["AAPL", "MSFT", "NVDA", "GOOGL", "AMZN"]
TICKER = "AAPL"
INTERVAL = "5m"
INTERVAL_MINUTES = 5
HORIZON_MINUTES = 30
HORIZON_BARS = HORIZON_MINUTES // INTERVAL_MINUTES

# Bump when the model or the feature set changes. The artifact filename and
# the `model_version` stored with every prediction both derive from this.
MODEL_VERSION = "aapl_logistic_v2"
MODEL_PATH = PROJECT_ROOT / "models" / f"{MODEL_VERSION}.joblib"

FEATURE_COLS = [
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
TARGET_COL = f"target_up_{HORIZON_MINUTES}m"

# Walk-forward window used by train.py, evaluate_walk_forward.py,
# compare_models.py and the dashboard, so they all report the same folds.
WALK_FORWARD_TRAIN_SESSIONS = 10
WALK_FORWARD_TEST_SESSIONS = 2
