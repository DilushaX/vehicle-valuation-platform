import logging
import time
from typing import Any, Callable, TypeVar

logger = logging.getLogger(__name__)

T = TypeVar("T")


def execute_with_retry(
    func: Callable[[], T],
    max_retries: int = 3,
    initial_delay: float = 1.0,
    backoff_factor: float = 2.0,
    retryable_exceptions: tuple = (Exception,),
    on_retry: Callable[[Exception, int, float], None] | None = None,
) -> T:
    """
    Executes a callable with exponential backoff on specified transient exceptions.
    Limits retries to max_retries.
    """
    delay = initial_delay
    for attempt in range(1, max_retries + 1):
        try:
            return func()
        except retryable_exceptions as exc:
            if attempt == max_retries:
                logger.error(
                    f"Operation failed after {attempt} attempts: {exc}"
                )
                raise
            if on_retry:
                on_retry(exc, attempt, delay)
            else:
                logger.warning(
                    f"Attempt {attempt}/{max_retries} failed ({exc.__class__.__name__}: {exc}). "
                    f"Retrying in {delay:.2f}s..."
                )
            time.sleep(delay)
            delay *= backoff_factor
