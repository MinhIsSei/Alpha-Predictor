import logging
import sys

_CONFIGURED = False


def configure_logging(name: str) -> logging.Logger:
    """Configure root logging once per process and return a named logger.

    Using one shared setup keeps the format consistent across ingest.py,
    predict.py, evaluate_predictions.py, and run_live_loop.py, so log lines
    from any of them can be grepped/aggregated the same way.
    """
    global _CONFIGURED
    if not _CONFIGURED:
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
            stream=sys.stdout,
        )
        _CONFIGURED = True
    return logging.getLogger(name)
