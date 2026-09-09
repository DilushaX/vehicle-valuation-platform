import logging
import time
from typing import Any, Callable, TypeVar

from config import settings

logger = logging.getLogger(__name__)

T = TypeVar("T")


def get_retry_after(exc: Exception) -> float | None:
    """Extracts Retry-After header in seconds if present on an HTTP exception."""
    resp = getattr(exc, "response", None)
    if resp is not None:
        headers = getattr(resp, "headers", {})
        val = headers.get("retry-after") or headers.get("Retry-After")
        if val is not None:
            try:
                return float(val)
            except (ValueError, TypeError):
                pass
    return None


def is_transient_error(exc: Exception) -> bool:
    """
    Determines whether an exception is a transient/retryable network or HTTP error.
    Permanent client errors (e.g. 400, 401, 403, 404, 405, 410, 422) are non-transient.
    Transient server errors (500, 502, 503, 504), rate limits (429), and connection/timeout
    errors are retryable.
    """
    try:
        import httpx
        if isinstance(exc, httpx.HTTPStatusError):
            code = exc.response.status_code
            if code in (429, 500, 502, 503, 504):
                return True
            if 400 <= code < 500:
                return False
        if isinstance(exc, (httpx.TimeoutException, httpx.NetworkError, TimeoutError, ConnectionError, OSError)):
            return True
    except ImportError:
        pass

    if isinstance(exc, (TimeoutError, ConnectionError, OSError)):
        return True

    return True


def execute_with_retry(
    func: Callable[[], T],
    max_retries: int = settings.MAX_RETRIES,
    initial_delay: float = settings.RETRY_DELAY,
    backoff_factor: float = settings.RETRY_BACKOFF,
    max_retry_after: float = settings.MAX_RETRY_AFTER,
    retryable_exceptions: tuple = (Exception,),
    on_retry: Callable[[Exception, int, float], None] | None = None,
) -> T:
    """
    Executes a callable with exponential backoff on specified transient exceptions.
    Limits retries to max_retries. Respects Retry-After header when present.
    Non-transient errors (e.g. 404, 400) and excessive rate limits (> max_retry_after)
    abort immediately without retrying.
    """
    delay = initial_delay
    for attempt in range(1, max_retries + 1):
        try:
            return func()
        except retryable_exceptions as exc:
            if not is_transient_error(exc):
                logger.warning(
                    f"Non-transient error encountered ({exc.__class__.__name__}: {exc}). "
                    "Aborting retries immediately."
                )
                raise

            if attempt == max_retries:
                logger.error(
                    f"Operation failed after {attempt} attempts: {exc}"
                )
                raise

            retry_after = get_retry_after(exc)
            if retry_after is not None:
                if retry_after > max_retry_after:
                    logger.warning(
                        f"Server rate limit: Retry-After is {retry_after}s "
                        f"(exceeds safe policy maximum of {max_retry_after}s). "
                        "Aborting retries to protect collection pipeline."
                    )
                    raise
                delay = retry_after

            if on_retry:
                on_retry(exc, attempt, delay)
            else:
                logger.warning(
                    f"Attempt {attempt}/{max_retries} failed ({exc.__class__.__name__}: {exc}). "
                    f"Retrying in {delay:.2f}s..."
                )
            time.sleep(delay)
            delay *= backoff_factor
