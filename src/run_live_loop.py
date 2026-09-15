"""Run predict.py and evaluate_predictions.py in live mode on a repeating schedule.

Usage:
    python src/run_live_loop.py

Stop safely with Ctrl+C (SIGINT) or `kill <pid>` (SIGTERM): the current
cycle finishes, then the loop exits. There is no forced mid-cycle kill, so a
prediction or outcome write is never left half-done.
"""
import signal
import subprocess
import sys
import time
from pathlib import Path

import pandas as pd
import pandas_market_calendars as mcal

from logging_config import configure_logging

logger = configure_logging("run_live_loop")

PROJECT_ROOT = Path(__file__).resolve().parent.parent
SRC_DIR = PROJECT_ROOT / "src"

# Matches the 5-minute candle interval predict.py/evaluate_predictions.py work with.
POLL_INTERVAL_SECONDS = 5 * 60

_stop_requested = False


def _handle_stop_signal(signum, _frame):
    global _stop_requested
    logger.info("Stop requested (signal %s); finishing current cycle then exiting.", signum)
    _stop_requested = True


signal.signal(signal.SIGINT, _handle_stop_signal)
signal.signal(signal.SIGTERM, _handle_stop_signal)


def _market_is_open_now() -> bool:
    calendar = mcal.get_calendar("NASDAQ")
    now = pd.Timestamp.now(tz="America/New_York")
    schedule = calendar.schedule(
        start_date=now.date(), end_date=now.date(), tz="America/New_York"
    )
    if schedule.empty:
        return False
    return schedule.iloc[0]["market_open"] <= now < schedule.iloc[0]["market_close"]


def _run_step(script_name: str, *args: str) -> None:
    command = [sys.executable, str(SRC_DIR / script_name), *args]
    logger.info("Running: %s", " ".join(command))
    result = subprocess.run(command, cwd=SRC_DIR)
    if result.returncode != 0:
        logger.warning("%s exited with code %d", script_name, result.returncode)


def _sleep_interruptibly(seconds: int) -> None:
    for _ in range(seconds):
        if _stop_requested:
            return
        time.sleep(1)


def main() -> None:
    logger.info(
        "Starting live loop (poll every %ds). Press Ctrl+C to stop.",
        POLL_INTERVAL_SECONDS,
    )
    while not _stop_requested:
        if _market_is_open_now():
            _run_step("predict.py", "--mode", "live")
            _run_step("evaluate_predictions.py", "--mode", "live")
        else:
            logger.info("Market closed; skipping this cycle.")

        _sleep_interruptibly(POLL_INTERVAL_SECONDS)

    logger.info("Stopped.")


if __name__ == "__main__":
    main()
