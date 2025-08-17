import signal
import inspect
import asyncio
import threading
import warnings
from functools import wraps
from concurrent import futures


class TimeoutError(Exception):
    """Custom exception raised when a function times out."""
    pass


def _parse_timeout_value(value):
    """Return int timeout in seconds or None if not provided/invalid."""
    if value is None:
        return None
    try:
        val = float(value)
    except (ValueError, TypeError):
        return None
    return val if val > 0 else None


def timeout(_func=None, *, timeout=None):
    """
    A decorator to add a timeout to both synchronous and asynchronous functions.

    The timeout can be set at the decorator level or on a per-call basis. The
    decorator-level timeout takes precedence.

    Usage:
        @timeout                 # no default; use per-call `timeout=` or no timeout
        def f(..., timeout=...): ...

        @timeout(timeout=5)      # default timeout of 5s; decorator timeout takes priority
        def f(...): ...

        @timeout
        async def af(...): ...

        @timeout(timeout=3)
        async def af(...): ...

    Priority: decorator-level `timeout=` (if provided and valid) > per-call `timeout=` kwarg > no timeout.

    Limitations for synchronous functions:
    1.  Using signal.SIGALRM: Only works in the main thread on Unix-like systems.
        Sub-second precision is not supported; timeouts < 1s are rounded up to 1s.
    2.  Using ThreadPoolExecutor (fallback): Does NOT terminate the function. It
        only stops waiting for the result. The function will continue running in
        the background.
    """
    decorator_timeout = _parse_timeout_value(timeout)

    def _decorate(func):
        if inspect.iscoroutinefunction(func):

            @wraps(func)
            async def async_wrapper(*args, **kwargs):
                call_timeout = _parse_timeout_value(kwargs.pop("timeout", None))
                t = decorator_timeout if decorator_timeout is not None else call_timeout

                if t is None:
                    return await func(*args, **kwargs)

                try:
                    return await asyncio.wait_for(func(*args, **kwargs), timeout=t)
                except asyncio.TimeoutError:
                    print(
                        f"Error: Async function '{func.__name__}' timed out after {t} seconds."
                    )
                    return None

            return async_wrapper

        else:

            @wraps(func)
            def sync_wrapper(*args, **kwargs):
                call_timeout = _parse_timeout_value(kwargs.pop("timeout", None))
                t = decorator_timeout if decorator_timeout is not None else call_timeout

                if t is None:
                    return func(*args, **kwargs)

                # Prefer signal-based timeout for its ability to interrupt execution
                can_use_sigalrm = (
                    hasattr(signal, "SIGALRM")
                    and threading.current_thread() is threading.main_thread()
                )

                if t and can_use_sigalrm:

                    def timeout_handler(signum, frame):
                        """Signal handler that raises our custom exception."""
                        raise TimeoutError(f"Function call timed out after {t} seconds")

                    original_handler = signal.signal(signal.SIGALRM, timeout_handler)

                    # signal.alarm only accepts integers. Round up for sub-second timeouts.
                    alarm_secs = int(t)
                    if 0 <= t < 1:
                        warnings.warn(
                            f"Timeout of {t}s is not supported by signal.alarm. Rounding up to 1s.",
                            UserWarning,
                        )
                        alarm_secs = 1

                    signal.alarm(alarm_secs)
                    try:
                        return func(*args, **kwargs)
                    except TimeoutError:
                        print(
                            f"Error: Function '{func.__name__}' timed out after {t} seconds (using signal)."
                        )
                        return None
                    finally:
                        signal.alarm(0)
                        signal.signal(signal.SIGALRM, original_handler)
                elif t is not None:
                    with futures.ThreadPoolExecutor(max_workers=1) as executor:
                        future = executor.submit(func, *args, **kwargs)
                        try:
                            return future.result(timeout=t)
                        except futures.TimeoutError:
                            print(
                                f"Error: Function '{func.__name__}' timed out after {t} seconds (using thread)."
                            )
                            print(
                                "Note: The function may still be running in the background."
                            )
                            return None
                else:
                    # No timeout specified; call directly
                    return func(*args, **kwargs)

            return sync_wrapper

    # handle decorator used with or without parentheses:
    # - @timeout  -> _func is the function
    # - @timeout(timeout=3) -> _func is None, returns decorator
    if _func is None:
        return _decorate
    else:
        return _decorate(_func)
