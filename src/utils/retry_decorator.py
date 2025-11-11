"""Retry decorator for handling transient errors."""

import time
import functools
import logging
import torch

logger = logging.getLogger(__name__)


def retry_on_oom(max_retries=3, wait_seconds=5):
    """Decorator to retry function on CUDA OOM error.

    Args:
        max_retries: Maximum number of retries
        wait_seconds: Seconds to wait between retries

    Usage:
        @retry_on_oom(max_retries=3)
        def train_model(...):
            ...
    """
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            last_exception = None

            for attempt in range(max_retries + 1):
                try:
                    return func(*args, **kwargs)

                except RuntimeError as e:
                    last_exception = e

                    # Check if it's CUDA OOM error
                    if "out of memory" in str(e).lower() or "cuda" in str(e).lower():
                        if attempt < max_retries:
                            logger.warning(
                                f"CUDA OOM error on attempt {attempt + 1}/{max_retries + 1}. "
                                f"Retrying in {wait_seconds}s..."
                            )

                            # Clear cache and wait
                            torch.cuda.empty_cache()
                            time.sleep(wait_seconds)
                        else:
                            logger.error(f"Max retries reached. Last error: {e}")
                            raise
                    else:
                        # Not an OOM error, re-raise immediately
                        raise

                except Exception as e:
                    # For other exceptions, re-raise immediately
                    raise

            # Should not reach here, but just in case
            raise last_exception

        return wrapper
    return decorator


def adaptive_batch_training(train_fn, initial_batch_size, min_batch_size=4):
    """Adaptive training that reduces batch size on OOM.

    Args:
        train_fn: Training function that takes batch_size as argument
        initial_batch_size: Starting batch size
        min_batch_size: Minimum allowed batch size

    Returns:
        Result from successful training

    Usage:
        def my_train_fn(batch_size):
            # Training code using batch_size
            return results

        results = adaptive_batch_training(my_train_fn, initial_batch_size=128)
    """
    current_batch_size = initial_batch_size

    while current_batch_size >= min_batch_size:
        try:
            logger.info(f"Attempting training with batch_size={current_batch_size}")
            result = train_fn(current_batch_size)
            logger.info(f"Training successful with batch_size={current_batch_size}")
            return result

        except RuntimeError as e:
            if "out of memory" in str(e).lower():
                torch.cuda.empty_cache()
                current_batch_size = current_batch_size // 2

                if current_batch_size < min_batch_size:
                    logger.error(
                        f"Batch size {current_batch_size} below minimum {min_batch_size}. "
                        "Cannot continue."
                    )
                    raise

                logger.warning(
                    f"OOM error. Reducing batch size to {current_batch_size} and retrying..."
                )
            else:
                # Not an OOM error
                raise

    raise RuntimeError(f"Training failed even with minimum batch size {min_batch_size}")
