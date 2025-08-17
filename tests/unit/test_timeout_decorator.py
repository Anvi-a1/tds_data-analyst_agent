import time
import asyncio
import pytest

from app.wrappers.timeout import timeout


@pytest.mark.asyncio
async def test_async_finishes_before_timeout():
    @timeout(timeout=1.0)
    async def short_wait():
        await asyncio.sleep(0.05)
        return "ok"

    start = time.monotonic()
    res = await short_wait()
    dt = time.monotonic() - start

    assert res == "ok"
    assert dt < 0.5


@pytest.mark.asyncio
async def test_async_times_out_and_returns_none():
    @timeout(timeout=0.1)
    async def long_wait():
        await asyncio.sleep(1.0)
        return "done"

    start = time.monotonic()
    res = await long_wait()
    dt = time.monotonic() - start

    assert res is None
    assert 0.05 < dt < 0.5


@pytest.mark.asyncio
async def test_async_blocking_sleep_blocks_event_loop_and_ignores_timeout():
    import time as _time

    @timeout(timeout=0.1)
    async def blocking():
        # blocks the whole event loop
        _time.sleep(0.25)
        return "done"

    start = time.monotonic()
    res = await blocking()  # this will block ~0.25s
    dt = time.monotonic() - start

    assert res == "done"
    assert dt >= 0.25


@pytest.mark.asyncio
async def test_async_swallow_cancellederror_defeats_timeout():
    @timeout(timeout=0.05)
    async def swallow_cancel():
        try:
            await asyncio.sleep(1.0)
        except asyncio.CancelledError:
            await asyncio.sleep(0.1)
            return "survived"
        return "never"

    start = time.monotonic()
    res = await swallow_cancel()
    dt = time.monotonic() - start

    # Because the coroutine swallowed CancelledError, it finishes normally
    assert res == "survived"
    assert dt >= 0.1


def test_sync_finishes_before_timeout():
    @timeout(timeout=1.0)
    def quick():
        import time as _t

        _t.sleep(0.02)
        return 42

    start = time.monotonic()
    res = quick()
    dt = time.monotonic() - start

    assert res == 42
    assert dt < 0.5


def test_sync_times_out_and_returns_none():
    @timeout(timeout=1)
    def slow():
        import time as _t

        _t.sleep(2)
        return "done"

    start = time.monotonic()
    res = slow()
    dt = time.monotonic() - start

    assert res is None
    assert 0.9 < dt < 1.5
