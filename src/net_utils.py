import logging
import time
from typing import Callable, TypeVar

T = TypeVar("T")

logger = logging.getLogger(__name__)


def with_retries(
    fn: Callable[[], T],
    *,
    attempts: int = 3,
    delay_seconds: float = 5.0,
    description: str = "network call",
) -> T:
    """Call fn() with up to `attempts` tries, waiting `delay_seconds` between failures.

    Yahoo Finance requests occasionally fail transiently (rate limiting,
    timeouts). Retrying a few times avoids treating those as a hard pipeline
    failure while still raising once genuinely exhausted.
    """
    last_error: Exception | None = None
    for attempt in range(1, attempts + 1):
        try:
            return fn()
        except Exception as exc:
            last_error = exc
            logger.warning(
                "%s failed (attempt %d/%d): %s", description, attempt, attempts, exc
            )
            if attempt < attempts:
                time.sleep(delay_seconds)
    raise last_error
