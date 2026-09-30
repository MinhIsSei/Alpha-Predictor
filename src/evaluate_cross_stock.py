"""Evaluate the AAPL-trained model on other tickers from the same download.

Usage:
    python -m src.evaluate_cross_stock

`ingest.py` already downloads five tickers (AAPL, MSFT, NVDA, GOOGL, AMZN)
but training only ever used AAPL, so the other four sat unused. This script
builds the same 11 features and the same same-session 30-minute target for
each of them and scores the AAPL model's pipeline (its fitted StandardScaler
and LogisticRegression coefficients, unchanged) against a most-frequent-class
baseline fit on that ticker's own data.

None of these tickers were seen during training, so no train/test split is
needed per ticker: the entire built dataset is out-of-sample for this model.
This is a generalization diagnostic, not a claim that the model would be
deployed on these tickers -- see reports/aapl_v2_experiment.md for the
model's own (AAPL) evaluation and its limitations.
"""

import joblib
import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.metrics import accuracy_score

from src.config import INTERVAL_MINUTES, MODEL_PATH, RAW_PRICES_PATH
from src.feature_utils import build_features
from src.logging_config import configure_logging
from src.target_utils import build_target

logger = configure_logging("evaluate_cross_stock")

OTHER_TICKERS = ["MSFT", "NVDA", "GOOGL", "AMZN"]


def build_ticker_dataset(
    raw_df: pd.DataFrame,
    ticker: str,
    feature_cols: list[str],
    horizon_minutes: int,
    periods: int,
) -> tuple[pd.DataFrame, str]:
    """Build features/target for one ticker's slice of the raw download."""
    prices = raw_df.xs(ticker, axis=1, level="Ticker").copy().sort_index()
    prices = build_features(prices)

    target_col = f"target_up_{horizon_minutes}m"
    prices = build_target(prices, horizon_minutes=horizon_minutes, periods=periods)

    data = prices[feature_cols + [target_col]].dropna()
    return data, target_col


def evaluate_ticker(model, data: pd.DataFrame, target_col: str) -> dict:
    """Score the (already-fitted) AAPL model against a same-ticker baseline."""
    X = data.drop(columns=[target_col])
    y = data[target_col].astype(int)

    # The baseline is fit on this ticker's own labels purely to characterize
    # its class balance -- it is not a claim of proper train/test separation,
    # since this whole script is a diagnostic, not a trained deployment.
    baseline = DummyClassifier(strategy="most_frequent").fit(X, y)

    return {
        "n_rows": len(data),
        "up_rate": y.mean(),
        "baseline_accuracy": accuracy_score(y, baseline.predict(X)),
        "model_accuracy": accuracy_score(y, model.predict(X)),
    }


def main() -> None:
    artifact = joblib.load(MODEL_PATH)
    model = artifact["pipeline"]
    feature_cols = artifact["feature_cols"]
    horizon_minutes = artifact["horizon_minutes"]
    periods = horizon_minutes // INTERVAL_MINUTES

    raw_path = RAW_PRICES_PATH
    raw_df = pd.read_parquet(raw_path)

    logger.info("Evaluating AAPL model (%s) on: %s", artifact.get("ticker"), OTHER_TICKERS)

    results = {}
    for ticker in OTHER_TICKERS:
        data, target_col = build_ticker_dataset(
            raw_df, ticker, feature_cols, horizon_minutes, periods
        )
        if data.empty:
            logger.warning("%s: no usable rows after dropping NaNs, skipped.", ticker)
            continue
        results[ticker] = evaluate_ticker(model, data, target_col)

    if not results:
        logger.warning("No ticker produced usable rows.")
        raise SystemExit(0)

    summary = pd.DataFrame(results).T
    summary.index.name = "ticker"

    print("\nAAPL model evaluated on other tickers (never seen in training):\n")
    print(summary.round(4).to_string())

    beats_baseline = (summary["model_accuracy"] > summary["baseline_accuracy"]).sum()
    print(f"\nModel beat the same-ticker baseline on {beats_baseline}/{len(summary)} tickers.")


if __name__ == "__main__":
    main()
