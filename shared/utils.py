"""General utility functions."""

import asyncio
from functools import wraps
from typing import Any, Callable, Optional, TypeVar
from urllib.parse import urlparse

from shared.logging_config import get_logger

logger = get_logger(__name__)

T = TypeVar("T")


def validate_url(url: str) -> bool:
    """
    Validate URL format.

    Args:
        url: URL string to validate

    Returns:
        True if valid, False otherwise
    """
    try:
        result = urlparse(url)
        return all([result.scheme, result.netloc])
    except Exception:
        return False


def retry_async(
    max_attempts: int = 3, delay: float = 1.0, backoff: float = 2.0, exceptions: tuple = (Exception,)
) -> Callable:
    """
    Decorator for retrying async functions with exponential backoff.

    Args:
        max_attempts: Maximum number of retry attempts
        delay: Initial delay between retries in seconds
        backoff: Backoff multiplier for delay
        exceptions: Tuple of exceptions to catch and retry

    Returns:
        Decorated async function
    """

    def decorator(func: Callable[..., Any]) -> Callable[..., Any]:
        @wraps(func)
        async def wrapper(*args: Any, **kwargs: Any) -> Any:
            current_delay = delay
            last_exception: Optional[Exception] = None

            for attempt in range(max_attempts):
                try:
                    return await func(*args, **kwargs)
                except exceptions as e:
                    last_exception = e
                    if attempt < max_attempts - 1:
                        logger.warning(
                            "retry_attempt",
                            function=func.__name__,
                            attempt=attempt + 1,
                            max_attempts=max_attempts,
                            error=str(e),
                            delay=current_delay,
                        )
                        await asyncio.sleep(current_delay)
                        current_delay *= backoff
                    else:
                        logger.error(
                            "retry_exhausted",
                            function=func.__name__,
                            attempts=max_attempts,
                            error=str(e),
                        )

            if last_exception:
                raise last_exception
            raise RuntimeError(f"Retry failed for {func.__name__}")

        return wrapper

    return decorator


def format_context_size(tokens: int) -> str:
    """
    Format token count in human-readable format.

    Args:
        tokens: Number of tokens

    Returns:
        Formatted string (e.g., "2.5K tokens")
    """
    if tokens < 1000:
        return f"{tokens} tokens"
    elif tokens < 1_000_000:
        return f"{tokens / 1000:.1f}K tokens"
    else:
        return f"{tokens / 1_000_000:.1f}M tokens"