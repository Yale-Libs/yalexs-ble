import asyncio
import logging
import struct
import time
from collections.abc import Callable
from typing import Any
from unittest.mock import AsyncMock, MagicMock, call, patch

import pytest
from bleak.backends.device import BLEDevice
from bleak.backends.scanner import AdvertisementData
from bleak.exc import BleakDBusError, BleakError

import yalexs_ble
from yalexs_ble.const import (
    AuthState,
    AutoLockMode,
    AutoLockState,
    BatteryState,
    DoorStatus,
    LockInfo,
    LockState,
    LockStatus,
)
from yalexs_ble.lock import Lock
from yalexs_ble.push import (
    _AUTH_FAILURE_HISTORY,
    APPLE_MFR_ID,
    AUTH_FAILURE_TO_START_REAUTH,
    AUTO_LOCK_READ_FAILURE_BACKOFF,
    AUTO_LOCK_READ_FAILURE_THRESHOLD,
    AUTO_LOCK_READ_REFRESH_INTERVAL,
    AUTO_LOCK_READ_RESPONSE_TIMEOUT,
    AUTO_LOCK_WRITE_ATTEMPTS,
    BATTERY_REFRESH_INTERVAL,
    BATTERY_TIMEOUT_COOLDOWN,
    DEADLINE_WAKEUP_RETRY_DELAY,
    DEFAULT_ATTEMPTS,
    HAP_FIRST_BYTE,
    KEEP_ALIVE_TIME,
    LOCK_STALE_STATE_DEBOUNCE_DELAY,
    NEVER_TIME,
    NO_BATTERY_SUPPORT_MODELS,
    SLOW_LATENCY,
    SLOW_MAX_INTERVAL,
    SLOW_MIN_INTERVAL,
    SLOW_TIMEOUT,
    YALE_MFR_ID,
    PushLock,
    operation_lock,
    retry_bluetooth_connection_error,
)
from yalexs_ble.session import (
    DisconnectedError,
    OperationIncompleteError,
    ResponseError,
)

# Shared battery-supporting lock used across tests. model is NOT in
# NO_BATTERY_SUPPORT_MODELS, so the battery-workaround path is not taken.
TEST_LOCK_INFO = LockInfo(
    manufacturer="August",
    model="ASL-03",
    serial="12345",
    firmware="2.0.0",
)


def publishing_read(push_lock: PushLock, *states: Any) -> AsyncMock:
    """Mock a Lock read that answers the way a real one does.

    Session._notify hands every frame to _state_callback before it resolves
    the waiter the read is blocked on, so a real read's answer is applied. A
    mock that only sets a return value reproduces the call but not the answer,
    so reads are stubbed with this.
    """

    async def _read(*args: Any, **kwargs: Any) -> None:
        push_lock._state_callback(list(states))

    return AsyncMock(side_effect=_read)


@pytest.mark.asyncio
async def test_operation_lock():
    """Test the operation_lock function."""

    counter = 0

    class MockPushLock:
        def __init__(self):
            self._operation_lock = asyncio.Lock()

        @property
        def name(self):
            return "lock"

        @operation_lock
        async def do_something(self):
            nonlocal counter
            counter += 1
            await asyncio.sleep(1)
            counter -= 1

    lock = MockPushLock()
    tasks = []
    for _ in range(10):
        tasks.append(asyncio.create_task(lock.do_something()))

    await asyncio.sleep(0)

    for _ in range(10):
        await asyncio.sleep(0)
        assert counter == 1

    for task in tasks:
        task.cancel()
    await asyncio.sleep(0)


async def _run_serialized_retry_calls(
    decorate: Callable[[Callable[..., Any]], Callable[..., Any]],
) -> list[list[int]]:
    """Drive concurrent always-failing calls through a decorated method.

    Asserts the behavior shared by both decorator orders, every call runs its
    full attempt count under the lock one at a time and surfaces the final
    error, then returns the call order chunked into DEFAULT_ATTEMPTS sized
    blocks so each test can assert its own attempt grouping.
    """
    CALLS = 10
    real_sleep = asyncio.sleep
    active = 0
    max_active = 0
    calls: list[int] = []

    class MockPushLock:
        def __init__(self):
            self._operation_lock = asyncio.Lock()

        @property
        def name(self):
            return "lock"

        async def _async_handle_disconnected(self, exc: Exception) -> None:
            """The retry wrapper awaits this hook on every retryable failure."""

        @decorate
        async def do_something(self, idx: int) -> None:
            nonlocal active, max_active
            active += 1
            max_active = max(max_active, active)
            calls.append(idx)
            try:
                # Yield while inside the lock so an exclusion failure would
                # let a second call enter and be recorded in max_active. The
                # binding taken before the patch keeps this a real yield.
                await real_sleep(0)
                raise TimeoutError
            finally:
                active -= 1

    lock = MockPushLock()
    # Patch out the retry backoff sleep so the test stays event driven even
    # if the backoff policy widens to cover TimeoutError.
    with patch("yalexs_ble.push.asyncio.sleep", new=AsyncMock()):
        tasks = [asyncio.create_task(lock.do_something(idx)) for idx in range(CALLS)]
        results = await asyncio.gather(*tasks, return_exceptions=True)

    # Every call ran its full attempt count and surfaced the final error.
    assert [type(result) for result in results] == [TimeoutError] * CALLS
    assert len(calls) == CALLS * DEFAULT_ATTEMPTS
    assert all(calls.count(idx) == DEFAULT_ATTEMPTS for idx in range(CALLS))
    # The lock serialized the attempts: no two calls were ever inside at once.
    assert max_active == 1
    assert active == 0
    return [
        calls[i : i + DEFAULT_ATTEMPTS] for i in range(0, len(calls), DEFAULT_ATTEMPTS)
    ]


@pytest.mark.asyncio
async def test_operation_lock_with_retry_bluetooth_connection_error():
    """Retry outside the operation lock: every attempt of every call runs
    under the lock, exactly one at a time, the lock is released between
    attempts, and the final error reaches the caller once the attempts are
    exhausted."""
    blocks = await _run_serialized_retry_calls(
        lambda func: retry_bluetooth_connection_error(operation_lock(func))
    )
    # Retrying outside the lock releases it between attempts, so a call's
    # attempts are not contiguous: another call gets in before the retry.
    assert not any(len(set(block)) == 1 for block in blocks)


@pytest.mark.asyncio
async def test_retry_bluetooth_connection_error_with_operation_lock():
    """The operation lock outside the retry wrapper: a call holds the lock
    across its whole retry loop, so its attempts run back to back before the
    next call starts, and the final error reaches the caller."""
    blocks = await _run_serialized_retry_calls(
        lambda func: operation_lock(retry_bluetooth_connection_error(func))
    )
    # Holding the lock across the retry loop keeps a call's attempts
    # contiguous: each consecutive block of DEFAULT_ATTEMPTS entries in the
    # call order belongs to a single call.
    assert all(len(set(block)) == 1 for block in blocks)


def test_needs_battery_workaround():
    assert "SL-103" in NO_BATTERY_SUPPORT_MODELS
    assert "CERES" in NO_BATTERY_SUPPORT_MODELS
    assert "Yale Linus L2" in NO_BATTERY_SUPPORT_MODELS
    assert "ASL-03" not in NO_BATTERY_SUPPORT_MODELS
    assert "MD-04I" not in NO_BATTERY_SUPPORT_MODELS


@pytest.mark.asyncio
async def test_background_task_logs_exception(caplog):
    """Background task failures should be logged with the lock name."""
    push_lock = PushLock(
        address="aa:bb:cc:dd:ee:ff",
        key="0800200c9a66",
        key_index=1,
        always_connected=False,
    )
    push_lock._name = "Test Lock"

    async def boom():
        raise BleakError("simulated background failure")

    with caplog.at_level("ERROR", logger="yalexs_ble.push"):
        push_lock.background_task(boom())
        (task,) = push_lock._background_tasks
        await asyncio.wait([task])
        await asyncio.sleep(0)  # let the done-callback run

    assert not push_lock._background_tasks
    assert any(
        "Background task failed" in record.message
        and "Test Lock" in record.message
        and "simulated background failure" in record.message
        for record in caplog.records
    )


@pytest.mark.asyncio
async def test_background_task_cancellation_not_logged(caplog):
    """Cancelled background tasks should not emit an error log."""
    push_lock = PushLock(
        address="aa:bb:cc:dd:ee:ff",
        key="0800200c9a66",
        key_index=1,
        always_connected=False,
    )
    push_lock._name = "Test Lock"

    async def long_running():
        await asyncio.sleep(60)

    with caplog.at_level("ERROR", logger="yalexs_ble.push"):
        push_lock.background_task(long_running())
        (task,) = push_lock._background_tasks
        task.cancel()
        await asyncio.wait([task])
        await asyncio.sleep(0)  # let the done-callback run

    assert not push_lock._background_tasks
    assert not any(
        "Background task failed" in record.message for record in caplog.records
    )


@pytest.mark.asyncio
async def test_background_task_success_not_logged(caplog):
    """Successful background tasks should not emit an error log."""
    push_lock = PushLock(
        address="aa:bb:cc:dd:ee:ff",
        key="0800200c9a66",
        key_index=1,
        always_connected=False,
    )
    push_lock._name = "Test Lock"

    async def ok():
        return None

    with caplog.at_level("ERROR", logger="yalexs_ble.push"):
        push_lock.background_task(ok())
        (task,) = push_lock._background_tasks
        await asyncio.wait([task])
        await asyncio.sleep(0)  # let the done-callback run

    assert not push_lock._background_tasks
    assert not any(
        "Background task failed" in record.message for record in caplog.records
    )


@pytest.mark.asyncio
async def test_background_task_done_callback_is_idempotent():
    """_on_background_task_done should tolerate being called for an absent task."""
    push_lock = PushLock(
        address="aa:bb:cc:dd:ee:ff",
        key="0800200c9a66",
        key_index=1,
        always_connected=False,
    )
    push_lock._name = "Test Lock"

    async def ok():
        return None

    push_lock.background_task(ok())
    (task,) = push_lock._background_tasks
    await asyncio.wait([task])
    assert task.done()  # exception() below requires a finished task

    push_lock._on_background_task_done(task)
    push_lock._on_background_task_done(task)
    assert task not in push_lock._background_tasks


@pytest.mark.asyncio
async def test_update_continues_after_battery_timeout():
    """
    Test that _update() continues and completes successfully
    even when battery() times out.

    Requirements:
    - battery() timeout does not fail entire update
    - lock_status/door_status/auto_lock_status still get called
    - the display has valid lock/door values (not UNKNOWN)
    - no forced disconnect due to battery timeout
    """

    push_lock = PushLock(
        address="aa:bb:cc:dd:ee:ff",
        key="0800200c9a66",
        key_index=1,
        always_connected=False,
    )
    push_lock._name = "Test Lock"

    # Mock lock that times out on battery()
    mock_lock = MagicMock()
    mock_lock.lock_info = AsyncMock(return_value=TEST_LOCK_INFO)

    # Battery times out
    mock_lock.battery = AsyncMock(side_effect=TimeoutError("Battery timeout"))

    # But other calls succeed
    mock_lock.door_status = publishing_read(push_lock, DoorStatus.CLOSED)
    mock_lock.auto_lock_status = AsyncMock()
    mock_lock.lock_status = publishing_read(push_lock, LockStatus.LOCKED)

    push_lock._lock_info = TEST_LOCK_INFO
    push_lock._running = True

    # Mock advertisement_data for connection_info
    push_lock._advertisement_data = AdvertisementData(
        local_name="Test Lock",
        service_data={},
        service_uuids=[],
        rssi=-50,
        manufacturer_data={},
        platform_data=(),
        tx_power=0,
    )

    with patch.object(push_lock, "_ensure_connected", return_value=mock_lock):
        # Should NOT raise exception
        await push_lock._update()

        # Battery call was attempted
        mock_lock.battery.assert_called_once()

        # Other status calls still happened
        mock_lock.door_status.assert_called_once()
        mock_lock.auto_lock_status.assert_called_once()
        mock_lock.lock_status.assert_called_once()

        # The display has valid lock/door (from the successful calls)
        assert push_lock.lock_status == LockStatus.LOCKED
        assert push_lock.door_status == DoorStatus.CLOSED

        # Battery should be None since it timed out
        assert push_lock.battery is None


@pytest.mark.asyncio
async def test_poll_battery_cooldown_skip(caplog: pytest.LogCaptureFixture) -> None:
    """Test that _poll_battery skips when on cooldown."""
    push_lock = PushLock(
        address="aa:bb:cc:dd:ee:ff",
        key="0800200c9a66",
        key_index=1,
        always_connected=False,
    )
    push_lock._name = "Test Lock"
    push_lock._lock_info = TEST_LOCK_INFO

    # Set cooldown to 5 seconds in the future
    push_lock._earliest_battery_attempt_time = time.monotonic() + 5.0

    mock_lock = MagicMock()
    mock_lock.battery = AsyncMock()

    # Call _poll_battery
    with caplog.at_level(logging.DEBUG, logger="yalexs_ble.push"):
        made_request = await push_lock._poll_battery(mock_lock)

    # Should skip the request
    assert made_request is False
    mock_lock.battery.assert_not_called()
    # The message names only the cooldown; the gate cannot know what armed it.
    # Asserted up to the fixed prefix, since the remaining seconds vary.
    assert "Skipping battery request; not asking again for" in caplog.text
    # Nothing was published
    assert push_lock.battery is None


@pytest.mark.asyncio
async def test_poll_battery_success():
    """Test that _poll_battery fetches battery and arms no cooldown."""
    push_lock = PushLock(
        address="aa:bb:cc:dd:ee:ff",
        key="0800200c9a66",
        key_index=1,
        always_connected=False,
    )
    push_lock._name = "Test Lock"
    push_lock._lock_info = TEST_LOCK_INFO

    # A cooldown from an earlier failure, already lapsed, so the gate lets
    # this poll through.
    lapsed_cooldown = time.monotonic() - 1.0
    push_lock._earliest_battery_attempt_time = lapsed_cooldown

    mock_lock = MagicMock()
    battery_state = BatteryState(voltage=6.0, percentage=80)
    mock_lock.battery = publishing_read(push_lock, battery_state)

    made_request = await push_lock._poll_battery(mock_lock)

    # Should make the request
    assert made_request is True
    mock_lock.battery.assert_called_once()

    # The lock's answering frame put the reading on display.
    assert push_lock.battery == battery_state
    assert push_lock.auth is not None
    assert push_lock.auth.successful is True

    # A reading the lock answered with leaves the lapsed deadline untouched,
    # so nothing here can wipe a cooldown armed while the read was in flight.
    assert push_lock._earliest_battery_attempt_time == lapsed_cooldown


@pytest.mark.asyncio
async def test_poll_battery_bleak_error():
    """Test that _poll_battery handles BleakError gracefully."""
    push_lock = PushLock(
        address="aa:bb:cc:dd:ee:ff",
        key="0800200c9a66",
        key_index=1,
        always_connected=False,
    )
    push_lock._name = "Test Lock"
    push_lock._lock_info = TEST_LOCK_INFO

    mock_lock = MagicMock()
    mock_lock.battery = AsyncMock(side_effect=BleakError("Connection failed"))

    # Call _poll_battery
    made_request = await push_lock._poll_battery(mock_lock)

    # Should make the request
    assert made_request is True
    mock_lock.battery.assert_called_once()

    # Nothing was published (error was logged but not raised)
    assert push_lock.battery is None

    # Cooldown should NOT be set (only TimeoutError sets cooldown)
    assert push_lock._earliest_battery_attempt_time == NEVER_TIME


@pytest.mark.asyncio
async def test_poll_battery_bleak_dbus_error():
    """Test that _poll_battery handles BleakDBusError gracefully."""
    push_lock = PushLock(
        address="aa:bb:cc:dd:ee:ff",
        key="0800200c9a66",
        key_index=1,
        always_connected=False,
    )
    push_lock._name = "Test Lock"
    push_lock._lock_info = TEST_LOCK_INFO

    mock_lock = MagicMock()
    mock_lock.battery = AsyncMock(
        side_effect=BleakDBusError("DBus error", "error body")
    )

    # Call _poll_battery
    made_request = await push_lock._poll_battery(mock_lock)

    # Should make the request
    assert made_request is True
    mock_lock.battery.assert_called_once()

    # Nothing was published (error was logged but not raised)
    assert push_lock.battery is None

    # Cooldown should NOT be set (only TimeoutError sets cooldown)
    assert push_lock._earliest_battery_attempt_time == NEVER_TIME


@pytest.mark.asyncio
async def test_update_keeps_a_value_delivered_mid_cycle_over_unknown() -> None:
    """A notify arriving mid-cycle survives when the cycle started at UNKNOWN.

    The cycle starts at UNKNOWN and reads no lock status of its own, since
    one is already seen this session. A notify delivers LOCKED and CLOSED
    while the cycle is awaiting a read, and the delivered values are still
    on display when the cycle ends.
    """
    push_lock = PushLock(
        address="aa:bb:cc:dd:ee:ff",
        key="0800200c9a66",
        key_index=1,
        always_connected=False,
    )
    push_lock._name = "Test Lock"

    # Start with UNKNOWN state; update will normally leave it UNKNOWN
    push_lock._lock_state = LockState(
        lock=LockStatus.UNKNOWN,
        door=DoorStatus.UNKNOWN,
        battery=None,
        auth=None,
        auto_lock=None,
        auto_lock_prev=None,
    )

    # Mock lock that doesn't return lock/door (simulating skipped polling)
    mock_lock = MagicMock()
    mock_lock.lock_info = AsyncMock(
        return_value=MagicMock(model="ASL-03", door_sense=True)
    )

    push_lock._lock_info = MagicMock(model="ASL-03", door_sense=True)
    push_lock._running = True

    # Mark lock/door/battery as already seen to simulate skipped polling
    push_lock._seen_this_session.add(LockStatus)
    push_lock._seen_this_session.add(DoorStatus)
    push_lock._seen_this_session.add(BatteryState)

    # Mock advertisement_data for connection_info
    push_lock._advertisement_data = AdvertisementData(
        local_name="Test Lock",
        service_data={},
        service_uuids=[],
        rssi=-50,
        manufacturer_data={},
        platform_data=(),
        tx_power=0,
    )

    # Gate auto_lock_status so we can inject notify updates mid-_update
    auto_lock_in_progress = asyncio.Event()
    allow_auto_lock_to_continue = asyncio.Event()

    async def auto_lock_status():
        auto_lock_in_progress.set()
        await allow_auto_lock_to_continue.wait()
        return AutoLockState(mode=AutoLockMode.OFF, duration=0)

    mock_lock.auto_lock_status = AsyncMock(side_effect=auto_lock_status)

    with patch.object(
        push_lock, "_ensure_connected", AsyncMock(return_value=mock_lock)
    ):
        update_task = asyncio.create_task(push_lock._update())

        # Wait until _update is awaiting auto_lock_status, then simulate notify callback
        await auto_lock_in_progress.wait()
        push_lock._update_any_state([LockStatus.LOCKED, DoorStatus.CLOSED])
        allow_auto_lock_to_continue.set()

        await update_task

        # The values the lock sent mid-cycle are still on display.
        assert push_lock.lock_status == LockStatus.LOCKED, (
            f"Lock status should still be LOCKED, got {push_lock.lock_status}"
        )
        assert push_lock.door_status == DoorStatus.CLOSED, (
            f"Door status should still be CLOSED, got {push_lock.door_status}"
        )


@pytest.mark.asyncio
async def test_update_does_not_revert_a_mid_cycle_change() -> None:
    """A notify arriving mid-cycle survives when the cycle started at a real value.

    The cycle begins with the lock LOCKED and the door CLOSED, so it holds a
    reading for both. While it is awaiting one of its reads, a key turned by
    hand delivers UNLOCKED and OPENED, and a battery frame delivers a voltage.
    Nothing in the cycle read a lock status, a door status, or a battery, so it
    has nothing newer to say about any of them, and the delivered values must
    be what is on display when it ends.
    """
    push_lock = _named_push_lock("aa:bb:cc:dd:ee:ff", always_connected=False)

    push_lock._lock_state = LockState(
        lock=LockStatus.LOCKED,
        door=DoorStatus.CLOSED,
        battery=None,
        auth=None,
        auto_lock=None,
        auto_lock_prev=None,
    )

    mock_lock = MagicMock()
    push_lock._lock_info = MagicMock(model="ASL-03", door_sense=True)
    push_lock._running = True

    # Everything but the auto lock setting is already seen, so the cycle reads
    # no lock or door status of its own.
    push_lock._seen_this_session.add(LockStatus)
    push_lock._seen_this_session.add(DoorStatus)
    push_lock._seen_this_session.add(BatteryState)

    push_lock._advertisement_data = _advertisement({})

    mid_cycle_battery = BatteryState(voltage=6.0, percentage=80)

    # Gate the auto lock read so the change can be delivered mid-cycle.
    auto_lock_in_progress = asyncio.Event()
    allow_auto_lock_to_continue = asyncio.Event()

    async def auto_lock_status() -> None:
        auto_lock_in_progress.set()
        await allow_auto_lock_to_continue.wait()

    mock_lock.auto_lock_status = AsyncMock(side_effect=auto_lock_status)

    with patch.object(
        push_lock, "_ensure_connected", AsyncMock(return_value=mock_lock)
    ):
        update_task = asyncio.create_task(push_lock._update())
        # As _deferred_update would, so the resync the change arms defers to
        # this cycle instead of starting a second one mid-test.
        push_lock._update_task = update_task

        await auto_lock_in_progress.wait()
        push_lock._update_any_state([LockStatus.UNLOCKED, DoorStatus.OPENED])
        push_lock._update_any_state([mid_cycle_battery])
        allow_auto_lock_to_continue.set()

        await update_task
        push_lock._cancel_future_update()

    assert push_lock.lock_status == LockStatus.UNLOCKED
    assert push_lock.door_status == DoorStatus.OPENED
    assert push_lock.battery == mid_cycle_battery


@pytest.mark.asyncio
async def test_update_auto_lock_from_notify_path_survives_poll_result() -> None:
    """A mid-cycle auto-lock publish is what _update ends holding.

    The auto-lock read's return value is the READSETTING acknowledgment
    constant (OFF) and is discarded; the stored setting arrives as the 0xBB
    settings response on the notify path during the cycle. The value on
    display when the cycle ends must be the published one, not the cycle's
    starting value or the poll constant.
    """
    push_lock = PushLock(
        address="aa:bb:cc:dd:ee:ff",
        key="0800200c9a66",
        key_index=1,
        always_connected=False,
    )
    push_lock._name = "Test Lock"

    push_lock._lock_state = LockState(
        lock=LockStatus.LOCKED,
        door=DoorStatus.CLOSED,
        battery=None,
        auth=None,
        auto_lock=None,
        auto_lock_prev=None,
    )

    mock_lock = MagicMock()
    push_lock._lock_info = MagicMock(model="ASL-03", door_sense=True)
    push_lock._running = True

    # Mark everything but AutoLockState seen so only the auto-lock read runs.
    push_lock._seen_this_session.add(LockStatus)
    push_lock._seen_this_session.add(DoorStatus)
    push_lock._seen_this_session.add(BatteryState)

    push_lock._advertisement_data = AdvertisementData(
        local_name="Test Lock",
        service_data={},
        service_uuids=[],
        rssi=-50,
        manufacturer_data={},
        platform_data=(),
        tx_power=0,
    )

    # Gate the read so the settings-response notify publish lands mid-cycle.
    auto_lock_in_progress = asyncio.Event()
    allow_auto_lock_to_continue = asyncio.Event()

    async def auto_lock_status():
        auto_lock_in_progress.set()
        await allow_auto_lock_to_continue.wait()
        # The acknowledgment constant -- must not reach the final state.
        return AutoLockState(mode=AutoLockMode.OFF, duration=0)

    mock_lock.auto_lock_status = AsyncMock(side_effect=auto_lock_status)

    with patch.object(
        push_lock, "_ensure_connected", AsyncMock(return_value=mock_lock)
    ):
        update_task = asyncio.create_task(push_lock._update())

        await auto_lock_in_progress.wait()
        # The 0xBB settings response publishing through the notify path.
        push_lock._update_any_state([AutoLockState(AutoLockMode.TIMER, 1800)])
        allow_auto_lock_to_continue.set()

        await update_task

        assert push_lock.auto_lock == AutoLockState(AutoLockMode.TIMER, 1800), (
            f"Auto-lock should be the notify-published value, got {push_lock.auto_lock}"
        )


@pytest.mark.asyncio
async def test_update_continues_when_lock_info_probe_fails() -> None:
    """Test that _update() proceeds with defaults when lock_info() raises."""
    push_lock = PushLock(
        address="aa:bb:cc:dd:ee:ff",
        key="0800200c9a66",
        key_index=1,
        always_connected=False,
    )
    push_lock._name = "Test Lock"
    push_lock._running = True

    mock_lock = MagicMock()
    mock_lock.lock_info = AsyncMock(side_effect=TimeoutError("probe timed out"))
    mock_lock.battery = publishing_read(
        push_lock, BatteryState(voltage=6.0, percentage=80)
    )
    mock_lock.door_status = publishing_read(push_lock, DoorStatus.CLOSED)
    mock_lock.auto_lock_status = AsyncMock()
    mock_lock.lock_status = publishing_read(push_lock, LockStatus.LOCKED)

    push_lock._advertisement_data = AdvertisementData(
        local_name="Test Lock",
        service_data={},
        service_uuids=[],
        rssi=-50,
        manufacturer_data={},
        platform_data=(),
        tx_power=0,
    )

    with patch.object(push_lock, "_ensure_connected", return_value=mock_lock):
        await push_lock._update()

    # lock_info was attempted
    mock_lock.lock_info.assert_called_once()

    # Update still completed with real data
    assert push_lock.lock_status == LockStatus.LOCKED

    # door_status not called because model="" makes door_sense=False
    mock_lock.door_status.assert_not_called()
    assert push_lock.door_status == DoorStatus.UNKNOWN

    # Defaults were used for lock_info, serial falls back to MAC address
    assert push_lock._lock_info is not None
    assert push_lock._lock_info.model == ""
    assert push_lock._lock_info.serial == "aa:bb:cc:dd:ee:ff"
    assert push_lock._lock_info.door_sense is False


@pytest.mark.asyncio
async def test_update_continues_when_lock_info_probe_bleak_error() -> None:
    """Test that _update() proceeds with defaults when lock_info() raises BleakError."""
    push_lock = PushLock(
        address="aa:bb:cc:dd:ee:ff",
        key="0800200c9a66",
        key_index=1,
        always_connected=False,
    )
    push_lock._name = "Test Lock"
    push_lock._running = True

    mock_lock = MagicMock()
    mock_lock.lock_info = AsyncMock(
        side_effect=BleakError("connection dropped during probe")
    )
    mock_lock.battery = publishing_read(
        push_lock, BatteryState(voltage=6.0, percentage=80)
    )
    mock_lock.door_status = publishing_read(push_lock, DoorStatus.CLOSED)
    mock_lock.auto_lock_status = AsyncMock()
    mock_lock.lock_status = publishing_read(push_lock, LockStatus.LOCKED)

    push_lock._advertisement_data = AdvertisementData(
        local_name="Test Lock",
        service_data={},
        service_uuids=[],
        rssi=-50,
        manufacturer_data={},
        platform_data=(),
        tx_power=0,
    )

    with patch.object(push_lock, "_ensure_connected", return_value=mock_lock):
        await push_lock._update()

    assert push_lock.lock_status == LockStatus.LOCKED
    assert push_lock._lock_info is not None
    assert push_lock._lock_info.manufacturer == "Unknown"
    assert push_lock._lock_info.serial == "aa:bb:cc:dd:ee:ff"
    assert push_lock._lock_info.door_sense is False


@pytest.mark.asyncio
async def test_update_sets_slow_connection_params_when_always_connected():
    """Test _update() sets slow BLE connection params when always connected."""
    push_lock = PushLock(
        address="aa:bb:cc:dd:ee:ff",
        key="0800200c9a66",
        key_index=1,
        always_connected=True,
    )
    push_lock._name = "Test Lock"
    push_lock._running = True

    mock_client = MagicMock()
    mock_client.set_connection_params = AsyncMock()

    mock_lock = MagicMock()
    mock_lock.client = mock_client
    mock_lock.battery = publishing_read(
        push_lock, BatteryState(voltage=5.5, percentage=95)
    )
    mock_lock.door_status = publishing_read(push_lock, DoorStatus.CLOSED)
    mock_lock.lock_status = publishing_read(push_lock, LockStatus.LOCKED)
    mock_lock.auto_lock_status = AsyncMock()

    push_lock._lock_info = TEST_LOCK_INFO
    push_lock._advertisement_data = AdvertisementData(
        local_name="Test Lock",
        service_data={},
        service_uuids=[],
        rssi=-50,
        manufacturer_data={},
        platform_data=(),
        tx_power=0,
    )

    with patch.object(push_lock, "_ensure_connected", return_value=mock_lock):
        await push_lock._update()

    mock_client.set_connection_params.assert_called_once_with(
        SLOW_MIN_INTERVAL, SLOW_MAX_INTERVAL, SLOW_LATENCY, SLOW_TIMEOUT
    )


def test_slow_connection_params_are_latency_based():
    """Slow mode must idle via peripheral latency, not a long interval.

    A long min == max interval throttles notification delivery (one frame per
    two connection events), so an operation's reply cannot drain before the
    next command is written. The idle duty cycle has to come from latency
    while the base interval stays short.
    """
    # Short base interval so notifications drain quickly once latency drops.
    assert SLOW_MAX_INTERVAL * 1.25 <= 50  # ms
    assert SLOW_MIN_INTERVAL == SLOW_MAX_INTERVAL
    # Idle wake-up is ~500ms, which is where the battery saving comes from.
    assert 400 <= (1 + SLOW_LATENCY) * SLOW_MAX_INTERVAL * 1.25 <= 600  # ms
    # Core spec: supervision timeout > (1 + latency) * max_interval * 2.
    assert SLOW_TIMEOUT * 10 > (1 + SLOW_LATENCY) * SLOW_MAX_INTERVAL * 1.25 * 2


@pytest.mark.asyncio
async def test_update_does_not_set_connection_params_when_not_always_connected():
    """Test _update() skips connection params when not always connected."""
    push_lock = PushLock(
        address="aa:bb:cc:dd:ee:ff",
        key="0800200c9a66",
        key_index=1,
        always_connected=False,
    )
    push_lock._name = "Test Lock"
    push_lock._running = True

    mock_client = MagicMock()
    mock_client.set_connection_params = AsyncMock()

    mock_lock = MagicMock()
    mock_lock.client = mock_client
    mock_lock.battery = publishing_read(
        push_lock, BatteryState(voltage=5.5, percentage=95)
    )
    mock_lock.door_status = publishing_read(push_lock, DoorStatus.CLOSED)
    mock_lock.lock_status = publishing_read(push_lock, LockStatus.LOCKED)
    mock_lock.auto_lock_status = AsyncMock()

    push_lock._lock_info = TEST_LOCK_INFO
    push_lock._advertisement_data = AdvertisementData(
        local_name="Test Lock",
        service_data={},
        service_uuids=[],
        rssi=-50,
        manufacturer_data={},
        platform_data=(),
        tx_power=0,
    )

    with patch.object(push_lock, "_ensure_connected", return_value=mock_lock):
        await push_lock._update()

    mock_client.set_connection_params.assert_not_called()


@pytest.mark.asyncio
async def test_update_handles_connection_params_failure():
    """Test that _update() continues even if set_connection_params fails."""
    push_lock = PushLock(
        address="aa:bb:cc:dd:ee:ff",
        key="0800200c9a66",
        key_index=1,
        always_connected=True,
    )
    push_lock._name = "Test Lock"
    push_lock._running = True

    mock_client = MagicMock()
    mock_client.set_connection_params = AsyncMock(
        side_effect=BleakError("Failed to set params")
    )

    mock_lock = MagicMock()
    mock_lock.client = mock_client
    mock_lock.battery = publishing_read(
        push_lock, BatteryState(voltage=5.5, percentage=95)
    )
    mock_lock.door_status = publishing_read(push_lock, DoorStatus.CLOSED)
    mock_lock.lock_status = publishing_read(push_lock, LockStatus.LOCKED)
    mock_lock.auto_lock_status = AsyncMock()

    push_lock._lock_info = TEST_LOCK_INFO
    push_lock._advertisement_data = AdvertisementData(
        local_name="Test Lock",
        service_data={},
        service_uuids=[],
        rssi=-50,
        manufacturer_data={},
        platform_data=(),
        tx_power=0,
    )

    with patch.object(push_lock, "_ensure_connected", return_value=mock_lock):
        # Should NOT raise even though set_connection_params failed
        await push_lock._update()

    assert push_lock.lock_status == LockStatus.LOCKED
    mock_client.set_connection_params.assert_called_once()


@pytest.mark.asyncio
async def test_battery_refresh_clears_seen_and_repoll_when_due():
    """In always_connected mode, _update() should evict BatteryState from
    _seen_this_session and re-poll battery once BATTERY_REFRESH_INTERVAL
    has elapsed since the last refresh."""
    push_lock = PushLock(
        address="aa:bb:cc:dd:ee:ff",
        key="0800200c9a66",
        key_index=1,
        always_connected=True,
    )
    push_lock._name = "Test Lock"

    battery_state = BatteryState(voltage=4.0, percentage=90)
    mock_lock = MagicMock()
    mock_lock.battery = publishing_read(push_lock, battery_state)
    mock_lock.lock_status = publishing_read(push_lock, LockStatus.LOCKED)
    mock_lock.door_status = publishing_read(push_lock, DoorStatus.CLOSED)
    mock_lock.auto_lock_status = AsyncMock()
    mock_lock.client = MagicMock()
    mock_lock.client.set_connection_params = AsyncMock()

    push_lock._lock_info = TEST_LOCK_INFO
    push_lock._advertisement_data = AdvertisementData(
        local_name="Test Lock",
        service_data={},
        service_uuids=[],
        rssi=-50,
        manufacturer_data={},
        platform_data=(),
        tx_power=0,
    )
    push_lock._running = True

    # Simulate battery already polled this session
    push_lock._seen_this_session.add(BatteryState)

    # Set the refresh deadline in the past so a refresh is due
    push_lock._next_battery_refresh_time = time.monotonic() - 1.0
    before_update = time.monotonic()

    with patch.object(push_lock, "_ensure_connected", return_value=mock_lock):
        await push_lock._update()

    # Battery should have been re-polled
    mock_lock.battery.assert_called_once()
    assert push_lock.battery == battery_state
    # Deadline should have been pushed out a full interval from the poll
    assert (
        push_lock._next_battery_refresh_time >= before_update + BATTERY_REFRESH_INTERVAL
    )


@pytest.mark.asyncio
async def test_battery_refresh_not_due_skips_repoll():
    """In always_connected mode, _update() should NOT re-poll battery when
    BATTERY_REFRESH_INTERVAL has not yet elapsed."""
    push_lock = PushLock(
        address="aa:bb:cc:dd:ee:ff",
        key="0800200c9a66",
        key_index=1,
        always_connected=True,
    )
    push_lock._name = "Test Lock"

    mock_lock = MagicMock()
    mock_lock.battery = AsyncMock()
    mock_lock.lock_status = publishing_read(push_lock, LockStatus.LOCKED)
    mock_lock.door_status = publishing_read(push_lock, DoorStatus.CLOSED)
    mock_lock.auto_lock_status = AsyncMock()
    mock_lock.client = MagicMock()
    mock_lock.client.set_connection_params = AsyncMock()

    push_lock._lock_info = TEST_LOCK_INFO
    push_lock._advertisement_data = AdvertisementData(
        local_name="Test Lock",
        service_data={},
        service_uuids=[],
        rssi=-50,
        manufacturer_data={},
        platform_data=(),
        tx_power=0,
    )
    push_lock._running = True

    # Simulate battery already polled this session
    push_lock._seen_this_session.add(BatteryState)

    # Set the refresh deadline in the future — not yet due
    push_lock._next_battery_refresh_time = time.monotonic() + BATTERY_REFRESH_INTERVAL

    with patch.object(push_lock, "_ensure_connected", return_value=mock_lock):
        await push_lock._update()

    # Battery should NOT have been re-polled
    mock_lock.battery.assert_not_called()


@pytest.mark.asyncio
async def test_battery_refresh_does_not_fire_when_not_always_connected():
    """The periodic battery refresh must not affect non-always-connected locks.
    In normal mode _seen_this_session clears on each new connection, so battery
    is polled naturally and the interval guard must stay dormant."""
    push_lock = PushLock(
        address="aa:bb:cc:dd:ee:ff",
        key="0800200c9a66",
        key_index=1,
        always_connected=False,
    )
    push_lock._name = "Test Lock"

    mock_lock = MagicMock()
    mock_lock.battery = AsyncMock()
    mock_lock.lock_status = publishing_read(push_lock, LockStatus.LOCKED)
    mock_lock.door_status = publishing_read(push_lock, DoorStatus.CLOSED)
    mock_lock.auto_lock_status = AsyncMock()
    mock_lock.client = MagicMock()
    mock_lock.client.set_connection_params = AsyncMock()

    push_lock._lock_info = TEST_LOCK_INFO
    push_lock._advertisement_data = AdvertisementData(
        local_name="Test Lock",
        service_data={},
        service_uuids=[],
        rssi=-50,
        manufacturer_data={},
        platform_data=(),
        tx_power=0,
    )
    push_lock._running = True

    # Simulate battery already seen and a refresh deadline in the past
    push_lock._seen_this_session.add(BatteryState)
    refresh_deadline = time.monotonic() - 1.0
    push_lock._next_battery_refresh_time = refresh_deadline

    with patch.object(push_lock, "_ensure_connected", return_value=mock_lock):
        await push_lock._update()

    # Refresh block should not have fired — battery skipped because it is
    # in _seen_this_session and always_connected is False
    mock_lock.battery.assert_not_called()
    # Deadline must not have been touched
    assert push_lock._next_battery_refresh_time == refresh_deadline


@pytest.mark.asyncio
async def test_battery_refresh_due_but_on_cooldown_does_not_evict():
    """A refresh that comes due while the battery cooldown is active must not
    evict BatteryState or poll early. The cooldown gate precedes eviction, so
    BatteryState stays in _seen_this_session and the deadline is untouched until
    a later cycle (after cooldown) can actually re-poll — never an early poll."""
    push_lock = PushLock(
        address="aa:bb:cc:dd:ee:ff",
        key="0800200c9a66",
        key_index=1,
        always_connected=True,
    )
    push_lock._name = "Test Lock"
    push_lock._lock_info = TEST_LOCK_INFO

    mock_lock = MagicMock()
    mock_lock.battery = AsyncMock()

    # Battery already polled this session and the refresh is due...
    push_lock._seen_this_session.add(BatteryState)
    refresh_deadline = time.monotonic() - 1.0
    push_lock._next_battery_refresh_time = refresh_deadline
    # ...but a prior timeout left the battery cooldown active.
    push_lock._earliest_battery_attempt_time = time.monotonic() + 100.0

    made_request = await push_lock._poll_battery(mock_lock)

    # Cooldown gate wins: no poll, no eviction, deadline untouched.
    assert made_request is False
    mock_lock.battery.assert_not_called()
    assert BatteryState in push_lock._seen_this_session
    assert push_lock._next_battery_refresh_time == refresh_deadline
    assert push_lock.battery is None


@pytest.mark.asyncio
async def test_disconnected_callback_schedules_reconnect_when_always_connected() -> (
    None
):
    """Disconnect callback schedules keep-alive when always_connected and auth ok."""
    push_lock = PushLock(
        address="aa:bb:cc:dd:ee:01",
        key="0800200c9a66",
        key_index=1,
        always_connected=True,
    )
    push_lock._name = "Test Lock"
    _AUTH_FAILURE_HISTORY.auth_success(push_lock.address)

    with patch.object(push_lock, "_keep_alive") as mock_keep_alive:
        push_lock._disconnected_callback()

    mock_keep_alive.assert_called_once()


@pytest.mark.asyncio
async def test_disconnected_callback_skips_reconnect_after_auth_failures() -> None:
    """Disconnect callback skips keep-alive when auth has failed enough times."""
    push_lock = PushLock(
        address="aa:bb:cc:dd:ee:02",
        key="0800200c9a66",
        key_index=1,
        always_connected=True,
    )
    push_lock._name = "Test Lock"
    for _ in range(AUTH_FAILURE_TO_START_REAUTH):
        _AUTH_FAILURE_HISTORY.auth_failed(push_lock.address)

    try:
        with patch.object(push_lock, "_keep_alive") as mock_keep_alive:
            push_lock._disconnected_callback()
        mock_keep_alive.assert_not_called()
    finally:
        _AUTH_FAILURE_HISTORY.auth_success(push_lock.address)


@pytest.mark.asyncio
async def test_disconnected_callback_noop_when_not_always_connected() -> None:
    """Disconnect callback does nothing in non-always-connected mode."""
    push_lock = PushLock(
        address="aa:bb:cc:dd:ee:03",
        key="0800200c9a66",
        key_index=1,
        always_connected=False,
    )
    push_lock._name = "Test Lock"

    with patch.object(push_lock, "_keep_alive") as mock_keep_alive:
        push_lock._disconnected_callback()

    mock_keep_alive.assert_not_called()


@pytest.mark.asyncio
async def test_keep_alive_noop_when_not_always_connected() -> None:
    """Keep-alive returns immediately when not always_connected."""
    push_lock = PushLock(
        address="aa:bb:cc:dd:ee:04",
        key="0800200c9a66",
        key_index=1,
        always_connected=False,
    )
    push_lock._name = "Test Lock"

    with (
        patch.object(push_lock, "_schedule_future_update") as mock_schedule_update,
        patch.object(push_lock, "_schedule_next_keep_alive") as mock_next_keep_alive,
    ):
        push_lock._keep_alive()

    mock_schedule_update.assert_not_called()
    mock_next_keep_alive.assert_not_called()


@pytest.mark.asyncio
async def test_keep_alive_schedules_update_and_next_when_always_connected() -> None:
    """Keep-alive schedules update and next keep-alive when always_connected."""
    push_lock = PushLock(
        address="aa:bb:cc:dd:ee:05",
        key="0800200c9a66",
        key_index=1,
        always_connected=True,
    )
    push_lock._name = "Test Lock"

    with (
        patch.object(push_lock, "_schedule_future_update") as mock_schedule_update,
        patch.object(push_lock, "_schedule_next_keep_alive") as mock_next_keep_alive,
    ):
        push_lock._keep_alive()

    mock_schedule_update.assert_called_once_with(0)
    mock_next_keep_alive.assert_called_once()


@pytest.mark.asyncio
async def test_disconnect_with_timer_skips_when_operation_lock_held() -> None:
    """Disconnect timer reschedules itself when an operation is in progress."""
    push_lock = PushLock(
        address="aa:bb:cc:dd:ee:06",
        key="0800200c9a66",
        key_index=1,
        always_connected=False,
    )
    push_lock._name = "Test Lock"

    async with push_lock._operation_lock:
        with (
            patch.object(push_lock, "_reset_disconnect_timer") as mock_reset,
            patch.object(push_lock, "background_task") as mock_bg,
        ):
            push_lock._disconnect_with_timer(5.0)

    mock_reset.assert_called_once()
    mock_bg.assert_not_called()


@pytest.mark.asyncio
async def test_disconnect_with_timer_runs_deferred_update_when_pending() -> None:
    """Disconnect timer cancels future update and runs it when one is pending."""
    push_lock = PushLock(
        address="aa:bb:cc:dd:ee:07",
        key="0800200c9a66",
        key_index=1,
        always_connected=False,
    )
    push_lock._name = "Test Lock"
    # Simulate a pending deferred update without actually scheduling on the loop
    push_lock._cancel_deferred_update = MagicMock()

    with (
        patch.object(push_lock, "_reset_disconnect_timer") as mock_reset,
        patch.object(push_lock, "_cancel_future_update") as mock_cancel_future,
        patch.object(push_lock, "_deferred_update") as mock_deferred,
        patch.object(push_lock, "background_task") as mock_bg,
    ):
        push_lock._disconnect_with_timer(5.0)

    mock_reset.assert_called_once()
    mock_cancel_future.assert_called_once()
    mock_deferred.assert_called_once()
    mock_bg.assert_not_called()


@pytest.mark.asyncio
async def test_disconnect_with_timer_triggers_disconnect_when_idle() -> None:
    """Disconnect timer schedules a forced disconnect when idle."""
    push_lock = PushLock(
        address="aa:bb:cc:dd:ee:08",
        key="0800200c9a66",
        key_index=1,
        always_connected=False,
    )
    push_lock._name = "Test Lock"

    with (
        patch.object(push_lock, "_cancel_disconnect_timer") as mock_cancel,
        patch.object(push_lock, "background_task") as mock_bg,
    ):
        push_lock._disconnect_with_timer(5.0)
        # Close the coroutine that would have been scheduled, to avoid
        # an unawaited-coroutine warning at GC time.
        (coro,), _ = mock_bg.call_args
        coro.close()

    mock_cancel.assert_called_once()
    mock_bg.assert_called_once()


@pytest.mark.asyncio
async def test_async_handle_disconnected_skips_when_connect_in_progress() -> None:
    """Handle-disconnected returns early when a connect is in progress."""
    push_lock = PushLock(
        address="aa:bb:cc:dd:ee:09",
        key="0800200c9a66",
        key_index=1,
        always_connected=False,
    )
    push_lock._name = "Test Lock"

    async with push_lock._connect_lock:
        with (
            patch.object(push_lock, "_cancel_disconnect_timer") as mock_cancel,
            patch.object(
                push_lock, "_execute_disconnect", new_callable=AsyncMock
            ) as mock_disconnect,
        ):
            await push_lock._async_handle_disconnected(RuntimeError("boom"))

    mock_cancel.assert_not_called()
    mock_disconnect.assert_not_called()


@pytest.mark.asyncio
async def test_async_handle_disconnected_executes_disconnect_when_idle() -> None:
    """Handle-disconnected runs full cleanup when no connect is in progress."""
    push_lock = PushLock(
        address="aa:bb:cc:dd:ee:0a",
        key="0800200c9a66",
        key_index=1,
        always_connected=False,
    )
    push_lock._name = "Test Lock"

    with (
        patch.object(push_lock, "_cancel_disconnect_timer") as mock_cancel,
        patch.object(
            push_lock, "_execute_disconnect", new_callable=AsyncMock
        ) as mock_disconnect,
    ):
        await push_lock._async_handle_disconnected(RuntimeError("boom"))

    mock_cancel.assert_called_once()
    mock_disconnect.assert_called_once()


class _MockRetryLock:
    """Minimal PushLock surface needed by retry_bluetooth_connection_error."""

    def __init__(self) -> None:
        self.address = "aa:bb:cc:dd:ee:ff"
        self._async_handle_disconnected = AsyncMock()

    @property
    def name(self) -> str:
        return "lock"


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "exc",
    [
        BleakDBusError("org.bluez.Error.Failed", []),
        DisconnectedError("disconnected"),
        BleakError("bleak error"),
        TimeoutError(),
        ResponseError("response"),
    ],
)
async def test_retry_eventually_succeeds_for_all_retryable_exceptions(
    exc: Exception,
) -> None:
    """All retryable exceptions get retried, then succeed on a later attempt."""
    lock = _MockRetryLock()
    calls = 0

    @retry_bluetooth_connection_error
    async def op(self):
        nonlocal calls
        calls += 1
        if calls == 1:
            raise exc
        return "ok"

    with patch("yalexs_ble.push.asyncio.sleep", new=AsyncMock()):
        result = await op(lock)

    assert result == "ok"
    assert calls == 2
    lock._async_handle_disconnected.assert_awaited_once_with(exc)


@pytest.mark.asyncio
async def test_retry_disconnected_error_reraised_unchanged_at_max_attempts() -> None:
    """DisconnectedError is not a BleakError, so it re-raises unchanged."""
    lock = _MockRetryLock()
    err = DisconnectedError("gone")

    @retry_bluetooth_connection_error
    async def op(self):
        raise err

    with (
        patch("yalexs_ble.push.asyncio.sleep", new=AsyncMock()),
        pytest.raises(DisconnectedError) as exc_info,
    ):
        await op(lock)

    assert exc_info.value is err
    # Called once per attempt.
    assert lock._async_handle_disconnected.await_count == DEFAULT_ATTEMPTS


@pytest.mark.asyncio
async def test_retry_disconnect_bleak_error_converted_to_disconnected_error() -> None:
    """A BleakError that reads as a disconnect converts to DisconnectedError."""
    lock = _MockRetryLock()
    err = BleakError("device disconnected")

    @retry_bluetooth_connection_error
    async def op(self):
        raise err

    with (
        patch("yalexs_ble.push.asyncio.sleep", new=AsyncMock()),
        pytest.raises(DisconnectedError) as exc_info,
    ):
        await op(lock)

    assert exc_info.value.__cause__ is err
    assert lock._async_handle_disconnected.await_count == DEFAULT_ATTEMPTS


@pytest.mark.asyncio
async def test_retry_bleak_error_raises_at_max_attempts() -> None:
    """Non-disconnect retryable exceptions propagate their original type."""
    lock = _MockRetryLock()
    err = BleakError("nope")

    @retry_bluetooth_connection_error
    async def op(self):
        raise err

    with (
        patch("yalexs_ble.push.asyncio.sleep", new=AsyncMock()),
        pytest.raises(BleakError),
    ):
        await op(lock)

    assert lock._async_handle_disconnected.await_count == DEFAULT_ATTEMPTS


@pytest.mark.asyncio
async def test_retry_backoff_exceptions_sleep_between_attempts() -> None:
    """RETRY_BACKOFF_EXCEPTIONS pause 0.25s between retries; others do not."""
    lock = _MockRetryLock()

    @retry_bluetooth_connection_error
    async def op_backoff(self):
        raise BleakDBusError("org.bluez.Error.Failed", [])

    @retry_bluetooth_connection_error
    async def op_nobackoff(self):
        raise TimeoutError

    with patch("yalexs_ble.push.asyncio.sleep", new=AsyncMock()) as sleep_mock:
        with pytest.raises(BleakError):
            await op_backoff(lock)
        backoff_calls = list(sleep_mock.await_args_list)

    # Sleeps happen only between non-final attempts.
    assert backoff_calls == [call(0.25)] * (DEFAULT_ATTEMPTS - 1)

    lock2 = _MockRetryLock()
    with patch("yalexs_ble.push.asyncio.sleep", new=AsyncMock()) as sleep_mock:
        with pytest.raises(TimeoutError):
            await op_nobackoff(lock2)
        assert sleep_mock.await_args_list == []


@pytest.mark.asyncio
@pytest.mark.parametrize("setter", ["set_auto_lock_duration", "set_auto_lock_mode"])
async def test_set_auto_lock_timeout_warns_and_names_the_failure(
    caplog: pytest.LogCaptureFixture, setter: str
) -> None:
    """An unconfirmed auto lock write warns once and re-raises with a message."""
    push_lock = PushLock(
        address="aa:bb:cc:dd:ee:0b",
        key="0800200c9a66",
        key_index=1,
        always_connected=False,
    )
    push_lock._name = "Test Lock"

    with patch.object(push_lock, "_set_auto_lock", new_callable=AsyncMock) as mock_set:
        mock_set.side_effect = TimeoutError()
        with pytest.raises(TimeoutError) as exc_info:
            if setter == "set_auto_lock_duration":
                await push_lock.set_auto_lock_duration(30)
            else:
                await push_lock.set_auto_lock_mode(AutoLockMode.TIMER)

    assert "Lock did not confirm the auto lock setting write" in str(exc_info.value)
    assert isinstance(exc_info.value.__cause__, TimeoutError)
    warnings = [record for record in caplog.records if record.levelname == "WARNING"]
    assert len(warnings) == 1
    assert "the lock may not support auto lock" in warnings[0].getMessage()
    assert f"after {AUTO_LOCK_WRITE_ATTEMPTS} attempts" in warnings[0].getMessage()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("setter", "arg", "auto_lock", "auto_lock_prev", "expected"),
    [
        # Turning auto lock off writes OFF with a zero duration.
        ("set_auto_lock_mode", AutoLockMode.OFF, None, None, (AutoLockMode.OFF, 0)),
        ("set_auto_lock_duration", 0, None, None, (AutoLockMode.OFF, 0)),
        # Already off returns without writing.
        (
            "set_auto_lock_mode",
            AutoLockMode.OFF,
            AutoLockState(mode=AutoLockMode.OFF, duration=0),
            None,
            None,
        ),
        (
            "set_auto_lock_duration",
            0,
            AutoLockState(mode=AutoLockMode.OFF, duration=0),
            None,
            None,
        ),
        # A mode change keeps the current duration; a duration change keeps
        # the current mode.
        (
            "set_auto_lock_mode",
            AutoLockMode.INSTANT,
            AutoLockState(mode=AutoLockMode.TIMER, duration=120),
            None,
            (AutoLockMode.INSTANT, 120),
        ),
        (
            "set_auto_lock_duration",
            60,
            AutoLockState(mode=AutoLockMode.INSTANT, duration=5),
            None,
            (AutoLockMode.INSTANT, 60),
        ),
        # When auto lock is currently off, fall back to the previous state.
        (
            "set_auto_lock_mode",
            AutoLockMode.TIMER,
            AutoLockState(mode=AutoLockMode.OFF, duration=0),
            AutoLockState(mode=AutoLockMode.TIMER, duration=300),
            (AutoLockMode.TIMER, 300),
        ),
        (
            "set_auto_lock_duration",
            60,
            AutoLockState(mode=AutoLockMode.OFF, duration=0),
            AutoLockState(mode=AutoLockMode.INSTANT, duration=10),
            (AutoLockMode.INSTANT, 60),
        ),
    ],
)
async def test_set_auto_lock_wrappers_choose_the_written_pair(
    setter: str,
    arg: AutoLockMode | int,
    auto_lock: AutoLockState | None,
    auto_lock_prev: AutoLockState | None,
    expected: tuple[AutoLockMode, int] | None,
) -> None:
    """The public setters pick mode and duration from current, then previous state."""
    push_lock = PushLock(
        address="aa:bb:cc:dd:ee:0c",
        key="0800200c9a66",
        key_index=1,
        always_connected=False,
    )
    push_lock._name = "Test Lock"
    if auto_lock or auto_lock_prev:
        push_lock._lock_state = LockState(
            lock=LockStatus.LOCKED,
            door=DoorStatus.CLOSED,
            battery=None,
            auth=None,
            auto_lock=auto_lock,
            auto_lock_prev=auto_lock_prev,
        )

    with patch.object(push_lock, "_set_auto_lock", new_callable=AsyncMock) as mock_set:
        await getattr(push_lock, setter)(arg)

    if expected is None:
        mock_set.assert_not_awaited()
    else:
        mock_set.assert_awaited_once_with(*expected)


@pytest.mark.parametrize("always_connected", [False, True])
@pytest.mark.asyncio
async def test_auto_lock_read_backoff_arms_after_threshold_timeouts(
    always_connected: bool, caplog: pytest.LogCaptureFixture
) -> None:
    """The read backs off only after THRESHOLD consecutive timeouts, not before.

    The arm path is mode-independent, so it is exercised with and without
    always_connected for symmetry with the response-timeout arming test.
    """
    caplog.set_level(logging.INFO)
    push_lock = PushLock(
        address="aa:bb:cc:dd:ee:0d",
        key="0800200c9a66",
        key_index=1,
        always_connected=always_connected,
    )
    push_lock._name = "Test Lock"

    mock_lock = MagicMock()
    mock_lock.auto_lock_status = AsyncMock(side_effect=TimeoutError)

    # Below the threshold: each timeout is counted, but the backoff is not armed.
    for expected in range(1, AUTO_LOCK_READ_FAILURE_THRESHOLD):
        assert await push_lock._read_auto_lock_setting(mock_lock) is False
        assert push_lock._auto_lock_read_ack_failures == expected
        assert push_lock._earliest_auto_lock_read_time == NEVER_TIME
    assert not [
        r for r in caplog.records if "may not support auto lock" in r.getMessage()
    ]

    # The threshold-th consecutive timeout arms the backoff and logs once.
    # Arming restarts the count, so the field reads zero afterwards.
    before = time.monotonic()
    assert await push_lock._read_auto_lock_setting(mock_lock) is False
    assert push_lock._auto_lock_read_ack_failures == 0
    assert (
        push_lock._earliest_auto_lock_read_time
        >= before + AUTO_LOCK_READ_FAILURE_BACKOFF
    )
    latch = [
        r
        for r in caplog.records
        if r.levelno == logging.INFO and "may not support auto lock" in r.getMessage()
    ]
    assert len(latch) == 1
    assert mock_lock.auto_lock_status.await_count == AUTO_LOCK_READ_FAILURE_THRESHOLD

    # Now backed off: the read is skipped without ever touching the lock.
    assert await push_lock._read_auto_lock_setting(mock_lock) is False
    assert mock_lock.auto_lock_status.await_count == AUTO_LOCK_READ_FAILURE_THRESHOLD


@pytest.mark.asyncio
async def test_auto_lock_read_backoff_reearned_after_window() -> None:
    """When the backoff window expires the count restarts and is re-earned."""
    push_lock = PushLock(
        address="aa:bb:cc:dd:ee:0d",
        key="0800200c9a66",
        key_index=1,
        always_connected=False,
    )
    push_lock._name = "Test Lock"
    # Arriving as if a prior window has just armed and reset: no failures held,
    # and the window is already past so reads resume.
    push_lock._auto_lock_read_ack_failures = 0
    push_lock._earliest_auto_lock_read_time = NEVER_TIME

    mock_lock = MagicMock()
    mock_lock.auto_lock_status = AsyncMock(side_effect=TimeoutError)

    # A fresh run of consecutive timeouts is needed to arm the backoff again.
    for expected in range(1, AUTO_LOCK_READ_FAILURE_THRESHOLD):
        assert await push_lock._read_auto_lock_setting(mock_lock) is False
        assert push_lock._auto_lock_read_ack_failures == expected
        assert push_lock._earliest_auto_lock_read_time == NEVER_TIME

    before = time.monotonic()
    assert await push_lock._read_auto_lock_setting(mock_lock) is False
    assert push_lock._auto_lock_read_ack_failures == 0
    assert (
        push_lock._earliest_auto_lock_read_time
        >= before + AUTO_LOCK_READ_FAILURE_BACKOFF
    )
    assert mock_lock.auto_lock_status.await_count == AUTO_LOCK_READ_FAILURE_THRESHOLD


@pytest.mark.asyncio
async def test_auto_lock_read_success_resets_failure_count() -> None:
    """A settings response arriving clears the failures and arms the refresh."""
    push_lock = PushLock(
        address="aa:bb:cc:dd:ee:0d",
        key="0800200c9a66",
        key_index=1,
        always_connected=False,
    )
    push_lock._name = "Test Lock"
    push_lock._auto_lock_read_ack_failures = AUTO_LOCK_READ_FAILURE_THRESHOLD - 1
    push_lock._auto_lock_read_response_failures = 1
    push_lock._awaiting_auto_lock_response = True
    push_lock._auto_lock_response_deadline = time.monotonic() + 10.0
    push_lock._earliest_auto_lock_read_time = time.monotonic() + 100.0

    before = time.monotonic()
    push_lock._update_any_state([AutoLockState(mode=AutoLockMode.TIMER, duration=30)])

    # The value landing -- not the read call returning -- is the success signal:
    # it clears both failure counts and disarms the pending-response deadline.
    assert push_lock._auto_lock_read_ack_failures == 0
    assert push_lock._auto_lock_read_response_failures == 0
    assert push_lock._awaiting_auto_lock_response is False
    assert push_lock._earliest_auto_lock_read_time == NEVER_TIME
    assert (
        push_lock._next_auto_lock_read_time >= before + AUTO_LOCK_READ_REFRESH_INTERVAL
    )
    assert AutoLockState in push_lock._seen_this_session
    assert push_lock.auto_lock == AutoLockState(mode=AutoLockMode.TIMER, duration=30)


@pytest.mark.asyncio
async def test_auto_lock_read_backoff_survives_reconnect() -> None:
    """The failure backoff outlives a reconnect -- the W2 regression guard."""
    push_lock = PushLock(
        address="aa:bb:cc:dd:ee:0e",
        key="0800200c9a66",
        key_index=1,
        always_connected=True,
    )
    push_lock._name = "Test Lock"
    deadline = time.monotonic() + AUTO_LOCK_READ_FAILURE_BACKOFF
    push_lock._earliest_auto_lock_read_time = deadline
    push_lock._auto_lock_read_ack_failures = AUTO_LOCK_READ_FAILURE_THRESHOLD
    push_lock._seen_this_session.add(AutoLockState)

    mock_lock = MagicMock()
    mock_lock.connect = AsyncMock()
    mock_lock.auto_lock_status = AsyncMock(side_effect=TimeoutError)

    with patch.object(push_lock, "_get_lock_instance", return_value=mock_lock):
        client = await push_lock._ensure_connected()

    # The reconnect cleared _seen_this_session but must NOT clear the backoff.
    assert AutoLockState not in push_lock._seen_this_session
    assert push_lock._earliest_auto_lock_read_time == deadline
    assert push_lock._auto_lock_read_ack_failures == AUTO_LOCK_READ_FAILURE_THRESHOLD

    # Still backed off after the reconnect: the read stays skipped, so the
    # case-2 storm cannot restart.
    assert await push_lock._read_auto_lock_setting(client) is False
    mock_lock.auto_lock_status.assert_not_called()
    push_lock._cancel_disconnect_timer()


@pytest.mark.asyncio
async def test_auto_lock_read_refresh_evicts_only_when_due() -> None:
    """The refresh re-reads only after the interval, mirroring the battery refresh."""
    push_lock = PushLock(
        address="aa:bb:cc:dd:ee:0f",
        key="0800200c9a66",
        key_index=1,
        always_connected=True,
    )
    push_lock._name = "Test Lock"
    mock_lock = MagicMock()
    mock_lock.auto_lock_status = AsyncMock(return_value=None)
    push_lock._seen_this_session.add(AutoLockState)

    # Not yet due: the seen gate holds, no re-read.
    push_lock._next_auto_lock_read_time = (
        time.monotonic() + AUTO_LOCK_READ_REFRESH_INTERVAL
    )
    assert await push_lock._read_auto_lock_setting(mock_lock) is False
    assert AutoLockState in push_lock._seen_this_session
    mock_lock.auto_lock_status.assert_not_called()

    # Deadline passed: evict AutoLockState and issue a fresh read.
    push_lock._next_auto_lock_read_time = time.monotonic() - 1.0
    assert await push_lock._read_auto_lock_setting(mock_lock) is True
    mock_lock.auto_lock_status.assert_awaited_once()


@pytest.mark.asyncio
async def test_auto_lock_read_no_evict_outside_always_connected() -> None:
    """Outside always_connected mode the refresh never evicts a seen value."""
    push_lock = PushLock(
        address="aa:bb:cc:dd:ee:10",
        key="0800200c9a66",
        key_index=1,
        always_connected=False,
    )
    push_lock._name = "Test Lock"
    mock_lock = MagicMock()
    mock_lock.auto_lock_status = AsyncMock(return_value=None)
    push_lock._seen_this_session.add(AutoLockState)
    # Deadline long past, but the eviction is gated on always_connected.
    push_lock._next_auto_lock_read_time = time.monotonic() - 1.0

    assert await push_lock._read_auto_lock_setting(mock_lock) is False
    assert AutoLockState in push_lock._seen_this_session
    mock_lock.auto_lock_status.assert_not_called()


@pytest.mark.asyncio
async def test_auto_lock_read_transport_error_does_not_arm_backoff() -> None:
    """A transport fault mirrors _poll_battery: skip without arming the backoff."""
    push_lock = PushLock(
        address="aa:bb:cc:dd:ee:14",
        key="0800200c9a66",
        key_index=1,
        always_connected=False,
    )
    push_lock._name = "Test Lock"
    mock_lock = MagicMock()
    mock_lock.auto_lock_status = AsyncMock(side_effect=BleakError("boom"))

    # A non-timeout fault is not the "alive but silent" signature: the read is
    # skipped, but the failure count and backoff are left untouched.
    assert await push_lock._read_auto_lock_setting(mock_lock) is False
    assert push_lock._auto_lock_read_ack_failures == 0
    assert push_lock._earliest_auto_lock_read_time == NEVER_TIME


@pytest.mark.asyncio
async def test_auto_lock_read_timeout_does_not_propagate_out_of_update() -> None:
    """A read timeout is caught in the helper, so _update completes normally."""
    push_lock = PushLock(
        address="aa:bb:cc:dd:ee:11",
        key="0800200c9a66",
        key_index=1,
        always_connected=False,
    )
    push_lock._name = "Test Lock"

    mock_lock = MagicMock()
    mock_lock.lock_info = AsyncMock(return_value=TEST_LOCK_INFO)
    mock_lock.battery = publishing_read(
        push_lock, BatteryState(voltage=6.0, percentage=80)
    )
    mock_lock.door_status = publishing_read(push_lock, DoorStatus.CLOSED)
    mock_lock.auto_lock_status = AsyncMock(side_effect=TimeoutError)
    mock_lock.lock_status = publishing_read(push_lock, LockStatus.LOCKED)

    push_lock._lock_info = TEST_LOCK_INFO
    push_lock._running = True
    push_lock._advertisement_data = AdvertisementData(
        local_name="Test Lock",
        service_data={},
        service_uuids=[],
        rssi=-50,
        manufacturer_data={},
        platform_data=(),
        tx_power=0,
    )

    with patch.object(push_lock, "_ensure_connected", return_value=mock_lock):
        await push_lock._update()

    # The update returned instead of raising: no forced disconnect, and the
    # timeout was counted as a failure rather than propagated.
    assert push_lock.lock_status == LockStatus.LOCKED
    assert push_lock._auto_lock_read_ack_failures == 1
    mock_lock.auto_lock_status.assert_awaited_once()
    push_lock._cancel_disconnect_timer()


@pytest.mark.asyncio
async def test_set_auto_lock_write_resets_read_backoff() -> None:
    """A confirmed write clears the backoff and evicts the seen value."""
    push_lock = PushLock(
        address="aa:bb:cc:dd:ee:12",
        key="0800200c9a66",
        key_index=1,
        always_connected=False,
    )
    push_lock._name = "Test Lock"
    push_lock._running = True
    push_lock._auto_lock_read_ack_failures = AUTO_LOCK_READ_FAILURE_THRESHOLD
    push_lock._auto_lock_read_response_failures = 2
    push_lock._awaiting_auto_lock_response = True
    push_lock._auto_lock_response_deadline = time.monotonic() + 10.0
    push_lock._earliest_auto_lock_read_time = (
        time.monotonic() + AUTO_LOCK_READ_FAILURE_BACKOFF
    )
    push_lock._next_auto_lock_read_time = (
        time.monotonic() + AUTO_LOCK_READ_REFRESH_INTERVAL
    )
    push_lock._seen_this_session.add(AutoLockState)

    mock_lock = MagicMock()
    mock_lock.set_auto_lock = AsyncMock()

    with (
        patch.object(push_lock, "_ensure_connected", return_value=mock_lock),
        patch.object(push_lock, "_complete_operation"),
    ):
        await push_lock._set_auto_lock(AutoLockMode.TIMER, 30)

    mock_lock.set_auto_lock.assert_awaited_once_with(AutoLockMode.TIMER, 30)
    assert push_lock._auto_lock_read_ack_failures == 0
    assert push_lock._auto_lock_read_response_failures == 0
    assert push_lock._awaiting_auto_lock_response is False
    assert push_lock._earliest_auto_lock_read_time == NEVER_TIME
    assert push_lock._next_auto_lock_read_time == NEVER_TIME
    assert AutoLockState not in push_lock._seen_this_session

    # With the value evicted and no backoff, the next read is issued again.
    mock_lock.auto_lock_status = AsyncMock(return_value=None)
    assert await push_lock._read_auto_lock_setting(mock_lock) is True


# ---------------------------------------------------------------------------
# Auto lock read: the four settings-command outcomes (see
# notes/yale/autolock_settings_command_outcome_taxonomy.md), each with and
# without always_connected, plus dropout and advert-driven connect-on-demand.
#
#   Case 1 -- dead lock, answers nothing. Caught by the earlier unguarded reads
#            in _update, so the auto lock read is never reached.
#   Case 2 -- alive but silent to the read: the ack times out.
#   Case 3 -- acks the read but withholds the 0xBB value: the response window
#            lapses with the value unseen.
#   Case 4 -- full working lock: ack, then the 0xBB value on the notify path.
# ---------------------------------------------------------------------------


def _named_push_lock(address: str, *, always_connected: bool) -> PushLock:
    """A named PushLock with the canonical test key and slot."""
    push_lock = PushLock(
        address=address,
        key="0800200c9a66",
        key_index=1,
        always_connected=always_connected,
    )
    push_lock._name = "Test Lock"
    return push_lock


def _auto_lock_update_lock(
    push_lock: PushLock, auto_lock_status: AsyncMock
) -> MagicMock:
    """A mock Lock answering every read so _update reaches the auto lock read.

    The auto lock read itself is wired per the outcome under test.
    """
    lock = MagicMock()
    lock.connect = AsyncMock()
    lock.is_connected = True
    lock.battery = publishing_read(push_lock, BatteryState(voltage=6.0, percentage=80))
    lock.door_status = publishing_read(push_lock, DoorStatus.CLOSED)
    lock.lock_status = publishing_read(push_lock, LockStatus.LOCKED)
    lock.auto_lock_status = auto_lock_status
    return lock


@pytest.mark.parametrize("always_connected", [False, True])
@pytest.mark.asyncio
async def test_auto_lock_read_response_timeout_arms_backoff(
    always_connected: bool, caplog: pytest.LogCaptureFixture
) -> None:
    """Case 3: the lock acks the read but withholds the 0xBB value.

    The read completes on the ack, so no timeout fires; the pending-response
    deadline lapses on the next cycle with the value still unseen. After
    THRESHOLD such response timeouts in a row the read backs off, and the INFO
    log reports the ack and response counts separately.
    """
    caplog.set_level(logging.INFO)
    push_lock = _named_push_lock("aa:bb:cc:dd:ee:20", always_connected=always_connected)
    mock_lock = MagicMock()
    mock_lock.auto_lock_status = AsyncMock(return_value=None)  # ack ok, no 0xBB

    # First cycle issues the read and arms the pending-response deadline.
    assert await push_lock._read_auto_lock_setting(mock_lock) is True
    assert push_lock._awaiting_auto_lock_response is True
    assert push_lock._auto_lock_read_response_failures == 0

    # Each later cycle finds the window lapsed with the value still unseen: a
    # response timeout, which re-reads until the threshold is reached.
    for expected in range(1, AUTO_LOCK_READ_FAILURE_THRESHOLD):
        push_lock._auto_lock_response_deadline = time.monotonic() - 1.0
        assert await push_lock._read_auto_lock_setting(mock_lock) is True
        assert push_lock._auto_lock_read_response_failures == expected
        assert push_lock._auto_lock_read_ack_failures == 0
        assert push_lock._earliest_auto_lock_read_time == NEVER_TIME

    # The threshold-th response timeout arms the backoff and logs the breakdown.
    push_lock._auto_lock_response_deadline = time.monotonic() - 1.0
    before = time.monotonic()
    assert await push_lock._read_auto_lock_setting(mock_lock) is False
    assert push_lock._auto_lock_read_response_failures == 0
    assert (
        push_lock._earliest_auto_lock_read_time
        >= before + AUTO_LOCK_READ_FAILURE_BACKOFF
    )
    latch = [
        r
        for r in caplog.records
        if r.levelno == logging.INFO
        and f"0 ack timeouts, {AUTO_LOCK_READ_FAILURE_THRESHOLD} response timeouts"
        in r.getMessage()
    ]
    assert len(latch) == 1


@pytest.mark.asyncio
async def test_auto_lock_read_value_in_flight_holds_without_strike() -> None:
    """Case 3 timing: within the response window the read waits, not strikes.

    Before the deadline the 0xBB may still be in flight, so the read neither
    books a response timeout nor issues another read.
    """
    push_lock = _named_push_lock("aa:bb:cc:dd:ee:21", always_connected=True)
    mock_lock = MagicMock()
    mock_lock.auto_lock_status = AsyncMock(return_value=None)

    assert await push_lock._read_auto_lock_setting(mock_lock) is True  # arms pending
    mock_lock.auto_lock_status.reset_mock()

    # Still inside the window (deadline is ~now + AUTO_LOCK_READ_RESPONSE_TIMEOUT).
    assert push_lock._auto_lock_response_deadline > time.monotonic()
    assert await push_lock._read_auto_lock_setting(mock_lock) is False
    mock_lock.auto_lock_status.assert_not_awaited()
    assert push_lock._awaiting_auto_lock_response is True
    assert push_lock._auto_lock_read_response_failures == 0


@pytest.mark.asyncio
async def test_auto_lock_read_value_landing_during_ack_does_not_arm_pending() -> None:
    """A 0xBB that lands in the same loop turn as the ack is not waited on.

    If the notify path delivers the value before the read coroutine resumes,
    AutoLockState is already seen, so the read must not arm a pending-response
    deadline for a value already in hand -- otherwise the next cycle would book
    one spurious response timeout.
    """
    push_lock = _named_push_lock("aa:bb:cc:dd:ee:2a", always_connected=True)

    def _ack_and_value(*_args: object) -> None:
        # The 0xBB is dispatched on the notify path before the await resumes.
        push_lock._update_any_state(
            [AutoLockState(mode=AutoLockMode.TIMER, duration=30)]
        )

    mock_lock = MagicMock()
    mock_lock.auto_lock_status = AsyncMock(side_effect=_ack_and_value)

    assert await push_lock._read_auto_lock_setting(mock_lock) is True
    assert push_lock._awaiting_auto_lock_response is False
    assert AutoLockState in push_lock._seen_this_session

    # A later cycle finds the value already seen, not a phantom pending read.
    assert await push_lock._read_auto_lock_setting(mock_lock) is False
    assert push_lock._auto_lock_read_response_failures == 0


@pytest.mark.asyncio
async def test_auto_lock_read_pending_survives_reconnect() -> None:
    """Dropout: a reconnect mid-read keeps the pending-response state.

    The hold must outlive the connection, so the pending flag, its deadline, and
    the response count all persist across the reconnect that clears the seen set.
    """
    push_lock = _named_push_lock("aa:bb:cc:dd:ee:23", always_connected=True)
    deadline = time.monotonic() + AUTO_LOCK_READ_RESPONSE_TIMEOUT
    push_lock._awaiting_auto_lock_response = True
    push_lock._auto_lock_response_deadline = deadline
    push_lock._auto_lock_read_response_failures = 1
    push_lock._seen_this_session.add(AutoLockState)

    mock_lock = MagicMock()
    mock_lock.connect = AsyncMock()
    mock_lock.is_connected = True

    with patch.object(push_lock, "_get_lock_instance", return_value=mock_lock):
        await push_lock._ensure_connected()

    # The reconnect cleared _seen_this_session but preserved the pending state.
    assert AutoLockState not in push_lock._seen_this_session
    assert push_lock._awaiting_auto_lock_response is True
    assert push_lock._auto_lock_response_deadline == deadline
    assert push_lock._auto_lock_read_response_failures == 1
    push_lock._cancel_disconnect_timer()


@pytest.mark.parametrize("always_connected", [False, True])
@pytest.mark.asyncio
async def test_auto_lock_read_success_ack_then_value(always_connected: bool) -> None:
    """Case 4: a full working lock. The ack arms the pending-response deadline;
    the 0xBB value landing afterwards clears it and arms the refresh timer."""
    push_lock = _named_push_lock("aa:bb:cc:dd:ee:24", always_connected=always_connected)
    push_lock._lock_info = TEST_LOCK_INFO
    mock_lock = _auto_lock_update_lock(push_lock, AsyncMock(return_value=None))

    before = time.monotonic()
    with patch.object(push_lock, "_ensure_connected", return_value=mock_lock):
        await push_lock._update()

    # The ack completed the read and armed both the pending flag and its
    # deadline; the value has not arrived yet. The flag is read into a typed
    # local so asserting it True does not narrow the later is-False check away.
    mock_lock.auto_lock_status.assert_awaited_once()
    armed: bool = push_lock._awaiting_auto_lock_response
    assert armed is True
    assert push_lock._auto_lock_response_deadline > before
    assert AutoLockState not in push_lock._seen_this_session

    # The 0xBB then lands on the notify path and clears the pending state.
    push_lock._update_any_state([AutoLockState(mode=AutoLockMode.TIMER, duration=30)])
    cleared: bool = push_lock._awaiting_auto_lock_response
    assert cleared is False
    assert AutoLockState in push_lock._seen_this_session
    assert (
        push_lock._next_auto_lock_read_time >= before + AUTO_LOCK_READ_REFRESH_INTERVAL
    )
    assert push_lock.auto_lock == AutoLockState(mode=AutoLockMode.TIMER, duration=30)
    push_lock._cancel_disconnect_timer()
    push_lock._cancel_keepalive_timer()


@pytest.mark.asyncio
async def test_update_any_state_auth_change_is_applied() -> None:
    """An AuthState change through _update_any_state updates the auth field.

    Covers the auth branch of _update_any_state, which sits directly beside the
    auto lock success block; the two share the "if lock_state.x != state" shape,
    so an inserted auto lock block anchors against the auth branch in the diff.
    """
    push_lock = _named_push_lock("aa:bb:cc:dd:ee:29", always_connected=False)
    assert push_lock._get_current_state().auth is None

    push_lock._update_any_state([AuthState(successful=True)])

    assert push_lock.auth == AuthState(successful=True)


@pytest.mark.asyncio
async def test_auto_lock_read_response_backoff_survives_connect_on_demand(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Case 3 under connect-on-demand: the hold outlives each connection.

    A not-always-connected lock that acks the read but withholds the value
    idle-disconnects between adverts. Each advert reconnects (clearing the seen
    set), yet the pending state and response count carry across, so the response
    backoff still latches instead of the read repeating on every connection.
    """
    caplog.set_level(logging.INFO)
    push_lock = _named_push_lock("aa:bb:cc:dd:ee:25", always_connected=False)

    async def _advert_connect() -> Lock:
        # A fresh advert-driven connection: was disconnected, now reconnects,
        # which clears _seen_this_session exactly as _update's connect does.
        push_lock._client = None
        lock = MagicMock()
        lock.connect = AsyncMock()
        lock.is_connected = True
        lock.auto_lock_status = AsyncMock(return_value=None)  # ack ok, no 0xBB
        with patch.object(push_lock, "_get_lock_instance", return_value=lock):
            return await push_lock._ensure_connected()

    # First connection: the read is issued and the pending deadline armed.
    client = await _advert_connect()
    assert await push_lock._read_auto_lock_setting(client) is True

    # Each later connection: the inter-advert gap lapsed the window, so the
    # withheld value books a response timeout that survived the reconnect.
    for expected in range(1, AUTO_LOCK_READ_FAILURE_THRESHOLD):
        push_lock._auto_lock_response_deadline = time.monotonic() - 1.0
        client = await _advert_connect()
        assert push_lock._awaiting_auto_lock_response is True  # survived reconnect
        assert await push_lock._read_auto_lock_setting(client) is True
        assert push_lock._auto_lock_read_response_failures == expected
        assert push_lock._earliest_auto_lock_read_time == NEVER_TIME

    # The threshold connection finally arms the backoff.
    push_lock._auto_lock_response_deadline = time.monotonic() - 1.0
    client = await _advert_connect()
    before = time.monotonic()
    assert await push_lock._read_auto_lock_setting(client) is False
    assert (
        push_lock._earliest_auto_lock_read_time
        >= before + AUTO_LOCK_READ_FAILURE_BACKOFF
    )
    push_lock._cancel_disconnect_timer()


@pytest.mark.asyncio
async def test_auto_lock_read_ack_backoff_survives_connect_on_demand() -> None:
    """Case 2 under connect-on-demand: a lock silent to the read accumulates ack
    timeouts across reconnects and backs off, rather than being re-asked on
    every connection."""
    push_lock = _named_push_lock("aa:bb:cc:dd:ee:26", always_connected=False)

    async def _advert_connect() -> Lock:
        push_lock._client = None
        lock = MagicMock()
        lock.connect = AsyncMock()
        lock.is_connected = True
        lock.auto_lock_status = AsyncMock(side_effect=TimeoutError)
        with patch.object(push_lock, "_get_lock_instance", return_value=lock):
            return await push_lock._ensure_connected()

    for expected in range(1, AUTO_LOCK_READ_FAILURE_THRESHOLD):
        client = await _advert_connect()
        assert await push_lock._read_auto_lock_setting(client) is False
        assert push_lock._auto_lock_read_ack_failures == expected
        assert push_lock._earliest_auto_lock_read_time == NEVER_TIME

    client = await _advert_connect()
    before = time.monotonic()
    assert await push_lock._read_auto_lock_setting(client) is False
    assert push_lock._auto_lock_read_ack_failures == 0
    assert (
        push_lock._earliest_auto_lock_read_time
        >= before + AUTO_LOCK_READ_FAILURE_BACKOFF
    )
    push_lock._cancel_disconnect_timer()


@pytest.mark.asyncio
async def test_auto_lock_read_connects_after_advertisement() -> None:
    """Connect-on-demand wiring: an advertisement drives the connect, then the
    read runs on that connection.

    The lock is disconnected; an advertisement arrives and schedules the update;
    the deferred update connects on demand and issues the auto lock read.
    """
    push_lock = _named_push_lock("aa:bb:cc:dd:ee:27", always_connected=False)
    push_lock._lock_info = TEST_LOCK_INFO
    push_lock._running = True
    mock_lock = _auto_lock_update_lock(push_lock, AsyncMock(return_value=None))
    ble_device = BLEDevice(push_lock.address, "Test Lock", None)
    ad = AdvertisementData(
        local_name="Test Lock",
        service_data={},
        service_uuids=[],
        rssi=-50,
        manufacturer_data={YALE_MFR_ID: b"\x01"},
        platform_data=(),
        tx_power=0,
    )

    with patch.object(push_lock, "_get_lock_instance", return_value=mock_lock):
        # The advertisement schedules a connect-on-demand update.
        push_lock.update_advertisement(ble_device, ad)
        assert push_lock._cancel_deferred_update is not None
        # Drive the scheduled update to completion.
        push_lock._deferred_update()
        assert push_lock._update_task is not None
        await push_lock._update_task

    # The connect happened after the advertisement, and the read ran on it.
    mock_lock.connect.assert_awaited()
    mock_lock.auto_lock_status.assert_awaited_once()
    assert push_lock._awaiting_auto_lock_response is True
    push_lock._running = False
    push_lock._cancel_disconnect_timer()
    push_lock._cancel_keepalive_timer()


@pytest.mark.parametrize("always_connected", [False, True])
@pytest.mark.asyncio
async def test_dead_lock_read_not_reached_earlier_read_propagates(
    always_connected: bool,
) -> None:
    """Case 1: a dead lock answers nothing. The unguarded door read runs before
    the auto lock read in _update and its timeout propagates (the connection
    layer handles the dead lock), so the auto lock read is never reached and its
    counters stay clean.
    """
    push_lock = _named_push_lock("aa:bb:cc:dd:ee:28", always_connected=always_connected)
    push_lock._lock_info = TEST_LOCK_INFO
    mock_lock = _auto_lock_update_lock(push_lock, AsyncMock(return_value=None))
    mock_lock.door_status = AsyncMock(side_effect=TimeoutError)

    with (
        patch.object(push_lock, "_ensure_connected", return_value=mock_lock),
        pytest.raises(TimeoutError),
    ):
        await push_lock._update()

    mock_lock.auto_lock_status.assert_not_awaited()
    assert push_lock._auto_lock_read_ack_failures == 0
    assert push_lock._auto_lock_read_response_failures == 0
    assert push_lock._awaiting_auto_lock_response is False
    push_lock._cancel_disconnect_timer()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "error",
    [ResponseError("no confirmation"), TimeoutError("no confirmation")],
)
async def test_set_auto_lock_write_retries_twice_then_gives_up(
    error: Exception,
) -> None:
    """The write retries AUTO_LOCK_WRITE_ATTEMPTS times, not the default four.

    A stalled write surfaces as either a ResponseError or, when the settings
    response never lands, a TimeoutError; both are retryable, so the count holds
    for the actual field failure as well as the synthetic one.
    """
    push_lock = PushLock(
        address="aa:bb:cc:dd:ee:13",
        key="0800200c9a66",
        key_index=1,
        always_connected=False,
    )
    push_lock._name = "Test Lock"
    push_lock._running = True

    mock_lock = MagicMock()
    mock_lock.set_auto_lock = AsyncMock(side_effect=error)

    with (
        patch.object(push_lock, "_ensure_connected", return_value=mock_lock),
        patch.object(push_lock, "_async_handle_disconnected", new_callable=AsyncMock),
        patch("yalexs_ble.push.asyncio.sleep", new_callable=AsyncMock),
        pytest.raises(type(error)),
    ):
        await push_lock._set_auto_lock(AutoLockMode.TIMER, 30)

    assert mock_lock.set_auto_lock.await_count == AUTO_LOCK_WRITE_ATTEMPTS


@pytest.mark.asyncio
async def test_poll_battery_skips_models_without_battery_support() -> None:
    """A model on the no-battery-support list is never asked for a reading."""
    push_lock = _named_push_lock("aa:bb:cc:dd:ee:19", always_connected=False)
    push_lock._lock_info = LockInfo(
        manufacturer="Yale",
        model="SL-103",
        serial="12345",
        firmware="2.0.0",
    )

    mock_lock = MagicMock()
    mock_lock.battery = AsyncMock()

    assert await push_lock._poll_battery(mock_lock) is False
    mock_lock.battery.assert_not_called()


@pytest.mark.asyncio
@pytest.mark.parametrize("voltage", [2.5, 3.0])
async def test_impossible_battery_voltage_is_refused_and_surfaced(
    caplog: pytest.LogCaptureFixture, voltage: float
) -> None:
    """A reading at or below 3.0 V is refused, reported, and starts the cooldown.

    The 3.0 V threshold is upstream's; the warning and the cooldown armed
    at the refusal are what this change adds.
    """
    push_lock = _named_push_lock("aa:bb:cc:dd:ee:15", always_connected=False)

    earliest = time.monotonic() + BATTERY_TIMEOUT_COOLDOWN
    with caplog.at_level(logging.WARNING, logger="yalexs_ble.push"):
        push_lock._update_any_state([BatteryState(voltage=voltage, percentage=0)])

    assert push_lock.battery is None
    assert "Battery voltage is impossible" in caplog.text
    # The warning says how long the lock will not be asked again.
    assert f"not asking again for {BATTERY_TIMEOUT_COOLDOWN} seconds" in caplog.text
    # Warning, not error: the lock behaved unexpectedly, the host did not fail.
    assert caplog.records[0].levelno == logging.WARNING
    # A refused reading must not suppress the next poll.
    assert BatteryState not in push_lock._seen_this_session
    # The cooldown is armed here, where the reading is thrown away.
    assert push_lock._earliest_battery_attempt_time >= earliest


@pytest.mark.asyncio
async def test_a_refused_battery_reading_starts_the_cooldown() -> None:
    """A poll whose reading is refused comes back with the cooldown running.

    The companion to the test above, taken through _poll_battery: the lock's
    answering frame is refused while the read is still awaiting it, so the
    cooldown is already armed when the poll returns.
    """
    push_lock = _named_push_lock("aa:bb:cc:dd:ee:34", always_connected=False)
    push_lock._lock_info = TEST_LOCK_INFO

    mock_lock = MagicMock()
    mock_lock.battery = publishing_read(
        push_lock, BatteryState(voltage=2.5, percentage=0)
    )

    earliest = time.monotonic() + BATTERY_TIMEOUT_COOLDOWN
    assert await push_lock._poll_battery(mock_lock) is True

    assert push_lock.battery is None
    assert push_lock._earliest_battery_attempt_time >= earliest
    # The cooldown, not BatteryState in _seen_this_session, stops the next ask.
    assert await push_lock._poll_battery(mock_lock) is False
    mock_lock.battery.assert_called_once()


def _real_decoder_pair(address: str) -> tuple[PushLock, Lock]:
    """A PushLock fed by a real Lock decoder rather than a stub."""
    push_lock = _named_push_lock(address, always_connected=False)
    lock = Lock(
        lambda: BLEDevice(address, "lock"),
        "0800200c9a66",
        1,
        "Test Lock",
        push_lock._state_callback,
    )
    return push_lock, lock


@pytest.mark.asyncio
async def test_a_battery_frame_reaches_the_display_through_the_real_decoder() -> None:
    """Pin the only route from a battery frame to the display.

    With the fetch return discarded, Lock._parse_state's BATTERY branch is the
    one route between a battery frame from the lock and the published state, so
    this drives a real captured frame through the real decoder into
    _update_any_state rather than stubbing either side.
    """
    push_lock, lock = _real_decoder_pair("aa:bb:cc:dd:ee:17")

    lock._internal_state_callback(bytes.fromhex("bb0200a50f00000079140000000000000200"))

    battery = push_lock.battery
    assert battery is not None
    assert battery.voltage == 5.241
    assert battery.percentage == 28


@pytest.mark.asyncio
async def test_a_door_frame_reaches_the_display_through_the_real_decoder() -> None:
    """Pin the only route from a door status frame to the display.

    door_status() asks for DOOR_ONLY, and with the fetch return discarded that
    branch of Lock._parse_state is the one route between the lock's answering
    frame and the published state, so this drives a real captured frame
    through the real decoder into _update_any_state rather than stubbing
    either side.
    """
    push_lock, lock = _real_decoder_pair("aa:bb:cc:dd:ee:18")

    lock._internal_state_callback(bytes.fromhex("bb0200122e00000003000000000000000000"))

    assert push_lock.door_status is DoorStatus.OPENED


@pytest.mark.asyncio
@pytest.mark.parametrize("reported", [LockStatus.UNKNOWN_01, LockStatus.UNKNOWN_06])
async def test_update_does_not_reconnect_on_a_setup_condition(
    caplog: pytest.LogCaptureFixture,
    reported: LockStatus,
) -> None:
    """A polled 0x01 or 0x06 reaches the display, warns, and forces no reconnect.

    The reconnect that used to answer these two is gone, so the warning is the
    only thing that records the condition. Without it a lock left in
    calibration or polarity discovery is diagnosable from nothing.
    """
    push_lock = _named_push_lock("aa:bb:cc:dd:ee:16", always_connected=False)
    push_lock._lock_info = TEST_LOCK_INFO
    push_lock._running = True

    mock_lock = _auto_lock_update_lock(push_lock, AsyncMock(return_value=None))
    mock_lock.lock_status = publishing_read(push_lock, reported)

    with (
        caplog.at_level(logging.WARNING, logger="yalexs_ble.push"),
        patch.object(push_lock, "_ensure_connected", AsyncMock(return_value=mock_lock)),
        patch.object(
            push_lock, "_execute_forced_disconnect", new_callable=AsyncMock
        ) as forced_disconnect,
    ):
        await push_lock._update()
        push_lock._cancel_future_update()

    forced_disconnect.assert_not_awaited()
    assert push_lock.lock_status is reported
    assert "a setup condition that ends at the lock by hand" in caplog.text
    assert str(reported) in caplog.text


@pytest.mark.asyncio
async def test_a_repeated_setup_condition_is_recorded_once(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """A repeated 0x01 is recorded once, not once per frame that carries it.

    The warning sits inside the lock_state.lock != state guard, so a frame
    that repeats the held status records nothing, however many arrive.
    """
    push_lock = _named_push_lock("aa:bb:cc:dd:ee:35", always_connected=False)

    with caplog.at_level(logging.WARNING, logger="yalexs_ble.push"):
        push_lock._update_any_state([LockStatus.UNKNOWN_01])
        push_lock._update_any_state([LockStatus.UNKNOWN_01])

    assert push_lock.lock_status is LockStatus.UNKNOWN_01
    assert len(caplog.records) == 1


def _advertisement(manufacturer_data: dict[int, bytes]) -> AdvertisementData:
    """An advertisement carrying the given manufacturer payloads."""
    return AdvertisementData(
        local_name="Test Lock",
        service_data={},
        service_uuids=[],
        rssi=-50,
        manufacturer_data=manufacturer_data,
        platform_data=(),
        tx_power=0,
    )


@pytest.mark.parametrize(
    "manufacturer_data",
    [
        # A HomeKit payload that ends before the state number it advertises.
        {APPLE_MFR_ID: bytes([HAP_FIRST_BYTE]) + b"\x00" * 8},
        # One byte short of the state record, which is the boundary the guard
        # turns on: at 14 bytes the unpack still runs off the end.
        {APPLE_MFR_ID: bytes([HAP_FIRST_BYTE]) + b"\x00" * 13},
        # An empty payload under either identifier.
        {APPLE_MFR_ID: b""},
        {YALE_MFR_ID: b""},
    ],
)
@pytest.mark.asyncio
async def test_a_short_advertisement_payload_is_skipped_not_parsed(
    manufacturer_data: dict[int, bytes],
) -> None:
    """A payload too short for the fields read from it schedules nothing.

    An advertisement is radio input and its length is not ours to assume. The
    parse used to index and unpack it unconditionally, so a truncated payload
    raised out of the callback the consumer dispatches from.
    """
    push_lock = _named_push_lock("aa:bb:cc:dd:ee:28", always_connected=False)
    ble_device = BLEDevice(push_lock.address, "Test Lock", None)

    push_lock.update_advertisement(ble_device, _advertisement(manufacturer_data))

    assert push_lock._cancel_deferred_update is None
    assert push_lock._last_hk_state == -1


def _hap_payload(state_num: int) -> bytes:
    """A full HomeKit advertisement payload carrying the given state number."""
    # <HHBB at byte 9: acid, then the global state number.
    return (
        bytes([HAP_FIRST_BYTE]) + b"\x00" * 8 + struct.pack("<HHBB", 1, state_num, 0, 0)
    )


@pytest.mark.asyncio
async def test_a_full_homekit_advertisement_still_reads_its_state_number() -> None:
    """The guard admits a payload long enough for the fields it reads."""
    push_lock = _named_push_lock("aa:bb:cc:dd:ee:29", always_connected=False)
    ble_device = BLEDevice(push_lock.address, "Test Lock", None)

    push_lock.update_advertisement(
        ble_device, _advertisement({APPLE_MFR_ID: _hap_payload(0x1234)})
    )

    assert push_lock._last_hk_state == 0x1234
    assert push_lock._cancel_deferred_update is not None
    push_lock._cancel_future_update()

    # A changed state number schedules another update; a repeat does not.
    push_lock.update_advertisement(
        ble_device, _advertisement({APPLE_MFR_ID: _hap_payload(0x1235)})
    )
    assert push_lock._last_hk_state == 0x1235
    assert push_lock._cancel_deferred_update is not None
    push_lock._cancel_future_update()

    push_lock.update_advertisement(
        ble_device, _advertisement({APPLE_MFR_ID: _hap_payload(0x1235)})
    )
    assert push_lock._last_hk_state == 0x1235
    assert push_lock._cancel_deferred_update is None


@pytest.mark.asyncio
async def test_a_cycle_that_changed_nothing_still_reports() -> None:
    """A cycle that read the same values as the last one still publishes once.

    A consumer may mark the lock unavailable from its own advertisement
    tracking and mark it available again only from this callback, so the
    callback has to report that the lock is still answering and not only
    that it changed.
    """
    push_lock = _named_push_lock("aa:bb:cc:dd:ee:30", always_connected=True)
    push_lock._lock_info = TEST_LOCK_INFO
    push_lock._running = True
    push_lock._advertisement_data = _advertisement({})

    # Everything the cycle could read is already held at the value the lock
    # will answer with, so no read in the cycle changes any field.
    push_lock._lock_state = LockState(
        lock=LockStatus.LOCKED,
        door=DoorStatus.CLOSED,
        battery=None,
        auth=AuthState(successful=True),
        auto_lock=None,
        auto_lock_prev=None,
        secure=LockStatus.UNLOCKED,
    )
    push_lock._seen_this_session.add(DoorStatus)
    push_lock._seen_this_session.add(BatteryState)
    push_lock._seen_this_session.add(AutoLockState)

    published: list[LockState] = []
    push_lock.register_callback(lambda state, info, conn: published.append(state))

    mock_lock = MagicMock()
    mock_lock.lock_status = publishing_read(push_lock, LockStatus.LOCKED)
    mock_lock.door_status = publishing_read(push_lock, DoorStatus.CLOSED)
    mock_lock.auto_lock_status = AsyncMock(return_value=None)
    mock_lock.battery = AsyncMock()

    with patch.object(
        push_lock, "_ensure_connected", AsyncMock(return_value=mock_lock)
    ):
        await push_lock._update()
        push_lock._cancel_future_update()

    push_lock._running = False
    push_lock._cancel_disconnect_timer()

    # The lock status was re-read and matched, so no field changed.
    mock_lock.lock_status.assert_awaited_once()
    assert push_lock.lock_status is LockStatus.LOCKED
    # The cycle still reported exactly once.
    assert len(published) == 1


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("address", "unseen", "read_name", "answer"),
    [
        ("aa:bb:cc:dd:ee:31", DoorStatus, "door_status", DoorStatus.OPENED),
        (
            "aa:bb:cc:dd:ee:32",
            AutoLockState,
            "auto_lock_status",
            AutoLockState(mode=AutoLockMode.TIMER, duration=30),
        ),
        ("aa:bb:cc:dd:ee:33", LockStatus, "lock_status", LockStatus.LOCKED),
    ],
)
async def test_every_read_a_cycle_issues_records_the_round_trip_as_a_success(
    address: str,
    unseen: type,
    read_name: str,
    answer: Any,
) -> None:
    """A read that answers is a successful round trip, at each site that issues one.

    _seen_this_session is filled except for one type, so the cycle issues
    exactly the read that fetches it and no other. An answer applies
    AuthState(successful=True) and clears the consecutive-failure count that
    arms the reauth latch, whichever read it was.
    """
    push_lock = _named_push_lock(address, always_connected=False)
    push_lock._lock_info = TEST_LOCK_INFO
    push_lock._running = True
    push_lock._advertisement_data = _advertisement({})

    # At the reauth latch, so a reset is observable.
    for _ in range(AUTH_FAILURE_TO_START_REAUTH):
        _AUTH_FAILURE_HISTORY.auth_failed(address)
    assert _AUTH_FAILURE_HISTORY.should_raise(address) is True

    reads = {"door_status", "auto_lock_status", "lock_status"}
    for seen in {LockStatus, DoorStatus, BatteryState, AutoLockState} - {unseen}:
        push_lock._seen_this_session.add(seen)

    mock_lock = MagicMock()
    mock_lock.battery = AsyncMock()
    for name in reads:
        setattr(mock_lock, name, AsyncMock())
    setattr(mock_lock, read_name, publishing_read(push_lock, answer))

    with patch.object(
        push_lock, "_ensure_connected", AsyncMock(return_value=mock_lock)
    ):
        await push_lock._update()
        push_lock._cancel_future_update()

    push_lock._running = False
    push_lock._cancel_disconnect_timer()

    # Exactly the one read, so the assertions below name one site.
    getattr(mock_lock, read_name).assert_awaited_once()
    for name in reads - {read_name}:
        getattr(mock_lock, name).assert_not_awaited()
    mock_lock.battery.assert_not_awaited()

    assert push_lock.auth == AuthState(successful=True)
    assert _AUTH_FAILURE_HISTORY.should_raise(address) is False


# ---------------------------------------------------------------------------
# Lock operations completed by their own op-response
# ---------------------------------------------------------------------------


def _operational_push_lock(address: str = "aa:bb:cc:dd:ee:50") -> PushLock:
    """A running lock with lock_info and advertisement data, ready to operate."""
    push_lock = _named_push_lock(address, always_connected=False)
    push_lock._lock_info = TEST_LOCK_INFO
    push_lock._running = True
    push_lock._advertisement_data = _advertisement({})
    return push_lock


def _known_state(
    lock: LockStatus,
    door: DoorStatus = DoorStatus.CLOSED,
    secure: LockStatus = LockStatus.UNLOCKED,
) -> LockState:
    """A settled cycle state, so an operation starts from a known position.

    The secure lock defaults to not secured.
    """
    return LockState(
        lock=lock,
        door=door,
        battery=None,
        auth=None,
        auto_lock=None,
        auto_lock_prev=None,
        secure=secure,
    )


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("method", "op_attr", "complete_state"),
    [
        ("lock", "force_lock", LockStatus.LOCKED),
        ("unlock", "force_unlock", LockStatus.UNLOCKED),
        ("securemode", "force_securemode", LockStatus.SECUREMODE),
    ],
)
async def test_execute_lock_operation_success_stamps_complete_state(
    method: str, op_attr: str, complete_state: LockStatus
) -> None:
    """A completed force_* advances the state to the completed status.

    Drives each public operation to completion: the transitional is stamped,
    the operation returns, and the completed status is applied.
    """
    push_lock = _operational_push_lock()
    mock_lock = MagicMock()
    setattr(mock_lock, op_attr, AsyncMock())

    with patch.object(
        push_lock, "_ensure_connected", AsyncMock(return_value=mock_lock)
    ):
        await getattr(push_lock, method)()

    getattr(mock_lock, op_attr).assert_awaited_once()
    assert push_lock.lock_status == complete_state


@pytest.mark.asyncio
async def test_securemode_forges_securing_and_neither_lock_flaps() -> None:
    """Securing an already-locked lock moves the secure lock and not the main.

    The stamped SECURING never publishes; the projection turns it into the
    pair. The main lock holds at LOCKED and then settles at SECUREMODE, so
    it cannot run the phantom locked -> locking -> locked cycle a LOCKING
    stamp caused, while the secure lock runs its own securing transitional.
    """
    push_lock = _operational_push_lock("aa:bb:cc:dd:ee:60")
    emissions: list[tuple[LockStatus, LockStatus]] = []
    push_lock.register_callback(
        lambda ls, li, ci: emissions.append((ls.lock, ls.secure))
    )

    # The lock is already locked when securemode is commanded.
    push_lock._update_any_state([LockStatus.LOCKED])
    assert emissions == [(LockStatus.LOCKED, LockStatus.UNLOCKED)]

    async def force_securemode(write_success_callback: Callable[[], None]) -> None:
        write_success_callback()
        # The previous operation's settled push can still land inside this
        # window; it is refused outright, so neither channel moves off the
        # stamped pair.
        push_lock._state_callback([LockStatus.LOCKED])

    mock_lock = MagicMock()
    mock_lock.force_securemode = force_securemode

    with patch.object(
        push_lock, "_ensure_connected", AsyncMock(return_value=mock_lock)
    ):
        await push_lock.securemode()

    # The main lock never shows LOCKING, so it cannot flap, and the secure
    # lock runs unlocked -> locking -> locked.
    assert emissions == [
        (LockStatus.LOCKED, LockStatus.UNLOCKED),
        (LockStatus.LOCKED, LockStatus.LOCKING),
        (LockStatus.SECUREMODE, LockStatus.LOCKED),
    ]
    push_lock._cancel_disconnect_timer()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("start", "operation", "op_attr", "expected"),
    [
        (
            (LockStatus.UNLOCKED, LockStatus.UNLOCKED),
            "lock",
            "force_lock",
            [
                (LockStatus.LOCKING, LockStatus.UNLOCKED),
                (LockStatus.LOCKED, LockStatus.UNLOCKED),
            ],
        ),
        (
            (LockStatus.LOCKED, LockStatus.UNLOCKED),
            "securemode",
            "force_securemode",
            [
                (LockStatus.LOCKED, LockStatus.LOCKING),
                (LockStatus.SECUREMODE, LockStatus.LOCKED),
            ],
        ),
        (
            (LockStatus.UNLOCKED, LockStatus.UNLOCKED),
            "securemode",
            "force_securemode",
            [
                (LockStatus.LOCKING, LockStatus.LOCKING),
                (LockStatus.SECUREMODE, LockStatus.LOCKED),
            ],
        ),
        (
            (LockStatus.SECUREMODE, LockStatus.LOCKED),
            "securemode",
            "force_securemode",
            [
                (LockStatus.SECUREMODE, LockStatus.LOCKING),
                (LockStatus.SECUREMODE, LockStatus.LOCKED),
            ],
        ),
        (
            (LockStatus.LOCKED, LockStatus.UNLOCKED),
            "unlock",
            "force_unlock",
            [
                (LockStatus.UNLOCKING, LockStatus.UNLOCKED),
                (LockStatus.UNLOCKED, LockStatus.UNLOCKED),
            ],
        ),
        (
            (LockStatus.SECUREMODE, LockStatus.LOCKED),
            "unlock",
            "force_unlock",
            [
                (LockStatus.UNLOCKING, LockStatus.UNLOCKING),
                (LockStatus.UNLOCKED, LockStatus.UNLOCKED),
            ],
        ),
    ],
    ids=[
        "lock_unlocked_to_locked",
        "securemode_locked_to_secured",
        "securemode_unlocked_to_secured",
        "securemode_already_secured_moves_only_secure",
        "unlock_locked_to_unlocked",
        "unlock_secured_to_unlocked",
    ],
)
async def test_secure_projection_invariant_table(
    start: tuple[LockStatus, LockStatus],
    operation: str,
    op_attr: str,
    expected: list[tuple[LockStatus, LockStatus]],
) -> None:
    """Every operation publishes its own pair for the two logical locks.

    One reported status feeds both, so this table is the invariant. The main
    lock keeps its present vocabulary, SECUREMODE as the settled secured
    position and a transitional only where the position itself has to change, and the
    secure lock reads secured or not secured with transitionals of its own.
    Each row drives the real operation and records every published pair.
    """
    push_lock = _operational_push_lock("aa:bb:cc:dd:ee:61")
    push_lock._lock_state = _known_state(start[0], secure=start[1])
    emissions: list[tuple[LockStatus, LockStatus]] = []
    push_lock.register_callback(
        lambda ls, li, ci: emissions.append((ls.lock, ls.secure))
    )

    async def force_operation(write_success_callback: Callable[[], None]) -> None:
        write_success_callback()

    mock_lock = MagicMock()
    setattr(mock_lock, op_attr, force_operation)

    with (
        patch.object(push_lock, "_ensure_connected", AsyncMock(return_value=mock_lock)),
        patch.object(push_lock, "_schedule_future_update"),
    ):
        await getattr(push_lock, operation)()

    assert emissions == expected
    # The synthetic SECURING is consumed by the projection: no published
    # value on either channel ever carries it.
    assert all(LockStatus.SECURING not in pair for pair in emissions)
    push_lock._cancel_disconnect_timer()


@pytest.mark.asyncio
async def test_settled_securemode_push_is_not_a_securing_transitional() -> None:
    """A SECUREMODE push outside an operation is a settled secured position.

    Only securemode() produces SECURING, so a reported SECUREMODE needs no
    other discriminator: the pair settles at secured at once.
    """
    push_lock = _operational_push_lock("aa:bb:cc:dd:ee:63")
    push_lock._lock_state = _known_state(LockStatus.LOCKED)

    push_lock._state_callback([LockStatus.SECUREMODE])

    assert push_lock.lock_status is LockStatus.SECUREMODE
    assert push_lock.secure_status is LockStatus.LOCKED
    push_lock._cancel_disconnect_timer()


@pytest.mark.asyncio
async def test_a_stamped_plain_locking_reads_not_secured() -> None:
    """A plain lock() out of Secured reads the secure lock as not secured.

    Only a reported SECUREMODE proves the lock is secured, so a plain
    LOCKING drops the secure lock to unlocked whatever it held; the settled
    SECUREMODE that follows a securing motion re-secures it.
    """
    push_lock = _operational_push_lock("aa:bb:cc:dd:ee:69")
    push_lock._lock_state = _known_state(
        LockStatus.SECUREMODE, secure=LockStatus.LOCKED
    )

    push_lock._pending_op_state = LockStatus.LOCKING
    push_lock._operation_write_success()

    assert push_lock.lock_status is LockStatus.LOCKING
    assert push_lock.secure_status is LockStatus.UNLOCKED


@pytest.mark.asyncio
async def test_the_forged_securing_is_not_a_position_the_lock_holds() -> None:
    """A stamped SECURING must not suppress the follow-up status poll.

    SECURING is the library's own transitional and the motor is running
    while it stands, so the reading that replaces it can only come from the
    poll the operation schedules at its exit. Recording LockStatus as seen
    this session would suppress that poll's read.
    """
    push_lock = _operational_push_lock("aa:bb:cc:dd:ee:6a")
    push_lock._lock_state = _known_state(LockStatus.LOCKED)

    push_lock._update_any_state([LockStatus.SECURING], arm_resync=False)

    assert push_lock.secure_status is LockStatus.LOCKING
    assert LockStatus not in push_lock._seen_this_session


@pytest.mark.asyncio
async def test_window_filter_refusal_leaves_the_secure_lock_alone() -> None:
    """A status refused by the window filter never reaches the projection.

    Part way through a securemode out of locked the published pair is
    (LOCKED, LOCKING). The filter returns None for a refused value, so
    nothing is projected from it. Re-projecting the standing display instead
    would read the settled main lock as "the lock is not secured" and end
    the secure transitional while the motor is still running.
    """
    push_lock = _operational_push_lock("aa:bb:cc:dd:ee:64")
    push_lock._lock_state = _known_state(LockStatus.LOCKED)
    push_lock._pending_op_state = LockStatus.SECURING
    push_lock._operation_write_success()
    assert (push_lock.lock_status, push_lock.secure_status) == (
        LockStatus.LOCKED,
        LockStatus.LOCKING,
    )

    # The previous operation's settled push arriving inside the window is
    # refused, so the pair stands.
    push_lock._update_any_state([LockStatus.LOCKED])
    assert (push_lock.lock_status, push_lock.secure_status) == (
        LockStatus.LOCKED,
        LockStatus.LOCKING,
    )


@pytest.mark.asyncio
async def test_retried_unlock_keeps_the_secure_transitional() -> None:
    """A second write-success projects the same UNLOCKING again.

    Every attempt stamps its own transitional at write-success, so one
    unlock can project UNLOCKING more than once. Taking LOCKED alone as
    evidence that the lock was secured would make the second stamp answer
    "not secured" and end the transitional with the motor still running.
    """
    push_lock = _operational_push_lock("aa:bb:cc:dd:ee:65")
    push_lock._lock_state = _known_state(
        LockStatus.SECUREMODE, secure=LockStatus.LOCKED
    )

    # Attempt 1 writes, then fails retryably: the window closes and the
    # transitional stays on display with nothing stamped over it.
    push_lock._pending_op_state = LockStatus.UNLOCKING
    push_lock._operation_write_success()
    push_lock._close_operation_window()
    assert push_lock.secure_status is LockStatus.UNLOCKING

    # Attempt 2 writes and stamps the same pending state.
    push_lock._pending_op_state = LockStatus.UNLOCKING
    push_lock._operation_write_success()
    assert (push_lock.lock_status, push_lock.secure_status) == (
        LockStatus.UNLOCKING,
        LockStatus.UNLOCKING,
    )


@pytest.mark.asyncio
async def test_nonretryable_securemode_after_write_stamps_unknown() -> None:
    """A securemode that dies after its write settles the pair at UNKNOWN.

    The write reached the lock and no op-response followed, so nothing said
    where the mechanism stopped. The transitional the write stamped is not a
    position, and the operation replaces it with the unknown one, which the
    projection carries to both channels.
    """
    exc = OperationIncompleteError("no op-response")
    push_lock = _operational_push_lock("aa:bb:cc:dd:ee:66")
    push_lock._lock_state = _known_state(LockStatus.LOCKED)

    mock_lock = MagicMock()

    async def force_securemode(write_success_callback: Callable[[], None]) -> None:
        write_success_callback()  # opens window, publishes (LOCKED, LOCKING)
        raise exc

    mock_lock.force_securemode = force_securemode

    with (
        patch.object(push_lock, "_ensure_connected", AsyncMock(return_value=mock_lock)),
        pytest.raises(type(exc)),
    ):
        await push_lock.securemode()

    assert push_lock._operation_window_open is False
    assert push_lock.lock_status is LockStatus.UNKNOWN
    assert push_lock.secure_status is LockStatus.UNKNOWN


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("secure", "expected_delay"),
    [
        (LockStatus.LOCKING, LOCK_STALE_STATE_DEBOUNCE_DELAY),
        (LockStatus.UNLOCKED, KEEP_ALIVE_TIME),
    ],
    ids=["secure_transitional_polls_at_the_debounce", "settled_pair_keeps_alive"],
)
async def test_operation_exit_delay_follows_the_displayed_pair(
    secure: LockStatus, expected_delay: float
) -> None:
    """The exit poll delay reads the pair, not the main lock alone.

    A cancelled securemode out of locked leaves (LOCKED, LOCKING) on
    display: the main lock holds a position, but the secure transitional is
    unresolved and only a read can settle it, so the poll runs at the settle
    debounce rather than waiting a keep-alive interval.
    """
    push_lock = _operational_push_lock("aa:bb:cc:dd:ee:67")
    push_lock._lock_state = _known_state(LockStatus.LOCKED, secure=secure)

    with patch.object(push_lock, "_schedule_future_update_with_debounce") as schedule:
        push_lock._finalize_operation()

    schedule.assert_called_once_with(expected_delay)


@pytest.mark.asyncio
async def test_deferred_update_defers_inside_the_stale_state_window() -> None:
    """An update landing right after an operation completes is rescheduled.

    The lock's reported state is still settling for the debounce delay after
    the op-response, so a lock polled now would report a stale state. The
    cycle is re-armed for the time the floor has left and starts no update
    task.
    """
    push_lock = _operational_push_lock("aa:bb:cc:dd:ee:51")
    push_lock._earliest_update_time = time.monotonic() + LOCK_STALE_STATE_DEBOUNCE_DELAY

    with patch.object(push_lock, "_schedule_future_update") as mock_reschedule:
        push_lock._deferred_update()

    mock_reschedule.assert_called_once()
    assert 0 < mock_reschedule.call_args.args[0] <= LOCK_STALE_STATE_DEBOUNCE_DELAY
    assert push_lock._update_task is None


@pytest.mark.asyncio
async def test_failed_operation_anchors_the_stale_state_debounce() -> None:
    """A failed force_* holds the next cycle off as a completed one does.

    The command reached the lock, so the motor may have run and the position
    is unknown rather than unchanged. The operation's exit moves the floor
    whether it completed or failed, and a cycle falling due before the floor
    re-arms for the remainder rather than polling.
    """
    push_lock = _operational_push_lock("aa:bb:cc:dd:ee:39")
    mock_lock = MagicMock()
    mock_lock.force_lock = AsyncMock(
        side_effect=OperationIncompleteError("no op-response")
    )

    with (
        patch.object(push_lock, "_ensure_connected", AsyncMock(return_value=mock_lock)),
        pytest.raises(OperationIncompleteError),
    ):
        await push_lock.lock()

    assert push_lock.lock_status == LockStatus.UNKNOWN

    with patch.object(push_lock, "_schedule_future_update") as mock_reschedule:
        push_lock._deferred_update()

    assert push_lock._update_task is None
    (delay,) = mock_reschedule.call_args.args
    assert 0 < delay <= LOCK_STALE_STATE_DEBOUNCE_DELAY


@pytest.mark.asyncio
async def test_deferred_update_backs_off_while_an_operation_holds_the_lock() -> None:
    """A cycle falling due mid-operation backs off rather than queueing.

    The floor can lapse while the motion is still running, and a cycle
    created there would wait on the operation lock and poll the lock the
    moment the operation released it, inside the post-operation debounce delay
    the operation is about to stamp.
    """
    push_lock = _operational_push_lock("aa:bb:cc:dd:ee:38")
    # The floor has lapsed, so only the operation lock is left to hold the
    # cycle off.
    push_lock._earliest_update_time = time.monotonic() - 1.0

    await push_lock._operation_lock.acquire()
    try:
        with patch.object(
            push_lock, "_schedule_future_update_with_debounce"
        ) as mock_reschedule:
            push_lock._deferred_update()
    finally:
        push_lock._operation_lock.release()

    assert push_lock._update_task is None
    mock_reschedule.assert_called_once_with(DEADLINE_WAKEUP_RETRY_DELAY)


@pytest.mark.asyncio
async def test_lock_stamps_transitional_only_at_write_success():
    """The LOCKING transitional is stamped only when the command write reaches
    the lock (write-success), never at issue time, and exactly once."""
    push_lock = _operational_push_lock()
    order: list[tuple[str, LockStatus | None]] = []

    def cb(lock_state, lock_info, connection_info):
        order.append(("state", lock_state.lock))

    push_lock.register_callback(cb)

    mock_lock = MagicMock()

    async def force_lock(write_success_callback):
        order.append(("write_success", None))
        write_success_callback()

    mock_lock.force_lock = force_lock

    with patch.object(
        push_lock, "_ensure_connected", AsyncMock(return_value=mock_lock)
    ):
        await push_lock.lock()

    # No state was emitted before the write reached the lock.
    first_state_index = next(i for i, ev in enumerate(order) if ev[0] == "state")
    assert first_state_index > order.index(("write_success", None))
    # Exactly one LOCKING stamp, then LOCKED on completion.
    assert [ev for ev in order if ev == ("state", LockStatus.LOCKING)] == [
        ("state", LockStatus.LOCKING)
    ]
    assert order[-1] == ("state", LockStatus.LOCKED)
    assert push_lock.lock_status == LockStatus.LOCKED


@pytest.mark.asyncio
async def test_write_success_without_pending_state_stamps_nothing():
    """A write-success with no pending transitional stamps no state.

    Every gated operation arms a pending transitional before its write, so the
    stamp's guard never sees None on any in-repo path; this pins the guard's
    other arc. The window itself still opens: opening is unconditional at the
    write-success site.
    """
    push_lock = _operational_push_lock()
    events: list[LockState] = []
    push_lock.register_callback(
        lambda lock_state, lock_info, connection_info: events.append(lock_state)
    )
    assert push_lock._pending_op_state is None

    push_lock._operation_write_success()

    assert events == []
    assert push_lock._operation_window_open is True


@pytest.mark.asyncio
async def test_a_raising_stamp_still_opens_the_window():
    """The window opens even when stamping the transitional raises.

    The session contains an exception from the write-success hook and runs
    the staged wait to its end, so a raise that left the window closed would
    run the whole operation with every mid-motion status admitted and no
    intervention status recorded.
    """
    push_lock = _operational_push_lock()
    push_lock._pending_op_state = LockStatus.LOCKING

    with (
        patch.object(
            push_lock, "_update_any_state", side_effect=RuntimeError("stamp failed")
        ),
        pytest.raises(RuntimeError),
    ):
        push_lock._operation_write_success()

    assert push_lock._operation_window_open is True


@pytest.mark.asyncio
async def test_window_filters_foreign_settle_until_close():
    """While the operation window is open a foreign lock-status settle is
    dropped; once the window closes the same value is admitted."""
    push_lock = _operational_push_lock()
    push_lock._lock_state = _known_state(LockStatus.UNLOCKED)

    push_lock._operation_window_open = True
    push_lock._update_any_state([LockStatus.LOCKED])
    assert push_lock.lock_status == LockStatus.UNLOCKED  # dropped mid-window

    push_lock._close_operation_window()
    push_lock._update_any_state([LockStatus.LOCKED])
    assert push_lock.lock_status == LockStatus.LOCKED  # admitted after close


@pytest.mark.asyncio
async def test_window_admits_the_door_member():
    """Only the lock member of a call is filtered mid-window.

    The filter runs on the lock status alone, so a door member handled in the
    same call still reaches the display, which is what a door event arriving
    during an operation needs.
    """
    push_lock = _operational_push_lock()
    push_lock._lock_state = _known_state(LockStatus.UNLOCKED)
    push_lock._operation_window_open = True

    push_lock._update_any_state([DoorStatus.OPENED, LockStatus.LOCKED])

    assert push_lock.door_status == DoorStatus.OPENED  # door applied
    assert push_lock.lock_status == LockStatus.UNLOCKED  # lock filtered


@pytest.mark.asyncio
async def test_a_recorded_outcome_stands_when_the_operation_fails() -> None:
    """The arm stamps the unknown position only over a blank record.

    An operation that recorded what the lock reported has an answer already,
    and it is a better one than the unknown position, so a failure on the way
    out leaves it alone. The record is seeded by hand here to pin the arm's
    other branch.
    """
    push_lock = _operational_push_lock("aa:bb:cc:dd:ee:6b")
    push_lock._lock_state = _known_state(LockStatus.UNLOCKED)

    async def force_lock(write_success_callback: Callable[[], None]) -> None:
        write_success_callback()
        push_lock._operation_outcome = LockStatus.LOCKED
        raise OperationIncompleteError("no op-response")

    mock_lock = MagicMock()
    mock_lock.force_lock = force_lock

    with (
        patch.object(push_lock, "_ensure_connected", AsyncMock(return_value=mock_lock)),
        pytest.raises(OperationIncompleteError),
    ):
        await push_lock.lock()

    assert push_lock.lock_status is LockStatus.LOCKED


@pytest.mark.asyncio
async def test_early_error_before_write_leaves_no_window_and_stamps_unknown():
    """A retryable failure before the command write opens no window and stamps
    UNKNOWN.

    The write-success hook fires when the lock's ATT server takes the command
    bytes, so this failure covers a command that never left the radio and a
    command that was delivered with its write response lost. The position the
    display held before the operation is not what either case leaves behind.
    """
    push_lock = _operational_push_lock()
    push_lock._lock_state = _known_state(LockStatus.LOCKED)

    mock_lock = MagicMock()
    mock_lock.force_lock = AsyncMock(
        side_effect=DisconnectedError("dropped before write")
    )

    with (
        patch.object(push_lock, "_ensure_connected", AsyncMock(return_value=mock_lock)),
        patch("yalexs_ble.push.asyncio.sleep", AsyncMock()),
        pytest.raises(DisconnectedError),
    ):
        await push_lock.lock()

    assert push_lock._operation_window_open is False
    assert push_lock._pending_op_state is None
    assert push_lock.lock_status == LockStatus.UNKNOWN


@pytest.mark.asyncio
async def test_nonretryable_after_write_stamps_unknown():
    """A non-retryable failure raised after write-success (a transitional is on
    display with no result coming) closes the window and stamps UNKNOWN."""
    exc = OperationIncompleteError("no op-response")
    push_lock = _operational_push_lock()
    push_lock._lock_state = _known_state(LockStatus.UNLOCKED)

    mock_lock = MagicMock()

    async def force_lock(write_success_callback):
        write_success_callback()  # opens window, stamps LOCKING
        raise exc

    mock_lock.force_lock = force_lock

    with (
        patch.object(push_lock, "_ensure_connected", AsyncMock(return_value=mock_lock)),
        pytest.raises(type(exc)),
    ):
        await push_lock.lock()

    assert push_lock._operation_window_open is False
    assert push_lock._pending_op_state is None
    assert push_lock.lock_status == LockStatus.UNKNOWN


@pytest.mark.asyncio
async def test_queued_second_operation_waits_for_the_first_to_settle():
    """A second operation queued mid-flight runs only after the first settles.

    The operation lock covers _run_lock_operation whole, so the queued caller
    waits: the first operation's terminal failure after write-success stamps
    UNKNOWN, and the second operation then runs from that display.
    """
    push_lock = _operational_push_lock("aa:bb:cc:dd:ee:5c")
    push_lock._lock_state = _known_state(LockStatus.UNLOCKED)
    states: list[LockStatus] = []

    def cb(lock_state, lock_info, connection_info):
        states.append(lock_state.lock)

    push_lock.register_callback(cb)

    proceed = asyncio.Event()
    first_attempt = True

    async def force_lock(write_success_callback):
        nonlocal first_attempt
        if first_attempt:
            first_attempt = False
            write_success_callback()  # opens window, stamps LOCKING
            await proceed.wait()
            raise OperationIncompleteError("no op-response")
        write_success_callback()  # the second operation succeeds

    mock_lock = MagicMock()
    mock_lock.force_lock = force_lock

    with patch.object(
        push_lock, "_ensure_connected", AsyncMock(return_value=mock_lock)
    ):
        first = asyncio.create_task(push_lock.lock())
        await asyncio.sleep(0)  # the first operation is past write-success
        second = asyncio.create_task(push_lock.lock())
        await asyncio.sleep(0)  # the second caller queues on the operation lock
        proceed.set()
        with pytest.raises(OperationIncompleteError):
            await first
        await second

    assert states == [
        LockStatus.LOCKING,
        LockStatus.UNKNOWN,
        LockStatus.LOCKING,
        LockStatus.LOCKED,
    ]


@pytest.mark.asyncio
async def test_cancelled_queued_operation_leaves_the_window_alone():
    """A queued operation cancelled while waiting changes nothing.

    The cancel lands in the lock acquisition, ahead of the method body, so
    the cancelled caller runs neither the record reset nor _finalize_operation:
    the first operation keeps its window open and completes normally.
    """
    push_lock = _operational_push_lock("aa:bb:cc:dd:ee:2c")
    push_lock._lock_state = _known_state(LockStatus.UNLOCKED)

    proceed = asyncio.Event()

    async def force_lock(write_success_callback):
        write_success_callback()  # opens window, stamps LOCKING
        await proceed.wait()

    mock_lock = MagicMock()
    mock_lock.force_lock = force_lock

    with patch.object(
        push_lock, "_ensure_connected", AsyncMock(return_value=mock_lock)
    ):
        first = asyncio.create_task(push_lock.lock())
        await asyncio.sleep(0)  # the first operation is past write-success
        second = asyncio.create_task(push_lock.lock())
        await asyncio.sleep(0)  # the second caller queues on the operation lock
        second.cancel()
        with pytest.raises(asyncio.CancelledError):
            await second
        assert push_lock._operation_window_open is True
        assert push_lock._pending_op_state == LockStatus.LOCKING
        proceed.set()
        await first

    assert push_lock.lock_status == LockStatus.LOCKED


@pytest.mark.asyncio
async def test_retry_restamps_at_write_success_without_unknown():
    """A retryable failure after write-success keeps the last transitional on
    display (never UNKNOWN); the next attempt re-sets the pending transitional
    and re-stamps at its own write-success."""
    push_lock = _operational_push_lock()
    events: list[LockStatus] = []

    def cb(lock_state, lock_info, connection_info):
        events.append(lock_state.lock)

    push_lock.register_callback(cb)

    pending_at_entry: list[LockStatus | None] = []
    attempts = 0

    async def force_lock(write_success_callback):
        nonlocal attempts
        attempts += 1
        pending_at_entry.append(push_lock._pending_op_state)
        write_success_callback()  # opens window, stamps LOCKING
        if attempts == 1:
            raise DisconnectedError("dropped before ack")

    mock_lock = MagicMock()
    mock_lock.force_lock = force_lock

    with (
        patch.object(push_lock, "_ensure_connected", AsyncMock(return_value=mock_lock)),
        patch("yalexs_ble.push.asyncio.sleep", AsyncMock()),
    ):
        await push_lock.lock()

    assert attempts == 2  # first attempt was retried
    # Each attempt re-set the pending transitional before its write.
    assert pending_at_entry == [LockStatus.LOCKING, LockStatus.LOCKING]
    # Never UNKNOWN; LOCKING once (the second stamp is a no-op since the
    # display already reads LOCKING) then LOCKED.
    assert LockStatus.UNKNOWN not in events
    assert events == [LockStatus.LOCKING, LockStatus.LOCKED]
    assert push_lock.lock_status == LockStatus.LOCKED


@pytest.mark.asyncio
async def test_a_settled_status_between_attempts_reaches_the_display():
    """A settled reading delivered between attempts reaches the display.

    A retryable failure closes the operation window on its way to the retry,
    so between attempts the display is open to the lock's own reporting; the
    next attempt opens a window of its own at its write-success. Without the
    close on the retryable exit the first attempt's window would span the
    gap and filter this reading.
    """
    push_lock = _operational_push_lock()
    events: list[LockStatus] = []

    def cb(lock_state, lock_info, connection_info):
        events.append(lock_state.lock)

    push_lock.register_callback(cb)

    attempts = 0

    async def force_lock(write_success_callback):
        nonlocal attempts
        attempts += 1
        if attempts == 1:
            write_success_callback()  # opens the window, stamps LOCKING
            raise DisconnectedError("dropped before ack")
        # This runs between the failure above and this attempt's write, where
        # the window is closed and a settled reading is admitted.
        push_lock._update_any_state([LockStatus.LOCKED])
        write_success_callback()

    mock_lock = MagicMock()
    mock_lock.force_lock = force_lock

    with (
        patch.object(push_lock, "_ensure_connected", AsyncMock(return_value=mock_lock)),
        patch("yalexs_ble.push.asyncio.sleep", AsyncMock()),
    ):
        await push_lock.lock()

    assert attempts == 2
    assert events == [
        LockStatus.LOCKING,
        LockStatus.LOCKED,
        LockStatus.LOCKING,
        LockStatus.LOCKED,
    ]
    assert push_lock.lock_status == LockStatus.LOCKED


@pytest.mark.asyncio
async def test_a_failure_before_any_write_replaces_a_reported_transitional():
    """What the display holds does not decide the unknown position.

    An operation started at the lock has it pushing UNLOCKING and the display
    holding it, and our unlock() then fails before its command write. The
    reading is real and the lock's own result is still coming, and the stamp
    lands on it anyway: a failed operation of ours reports that we cannot say
    where the mechanism is, whatever the display was showing.
    """
    push_lock = _operational_push_lock()
    push_lock._lock_state = _known_state(LockStatus.UNLOCKING)

    mock_lock = MagicMock()
    mock_lock.force_unlock = AsyncMock(
        side_effect=OperationIncompleteError("before write")
    )

    with (
        patch.object(push_lock, "_ensure_connected", AsyncMock(return_value=mock_lock)),
        pytest.raises(OperationIncompleteError),
    ):
        await push_lock.unlock()

    assert push_lock.lock_status == LockStatus.UNKNOWN
    push_lock._cancel_future_update()
    push_lock._cancel_disconnect_timer()


@pytest.mark.asyncio
async def test_an_operation_settling_after_the_stop_arms_nothing():
    """An operation ending after the watcher stopped leaves no timer behind.

    _cancel does not reach an operation in flight, so _finalize_operation runs
    once the operation fails on its own. The window is closed either way, but
    scheduling the status poll would arm a cycle holding the lock past the stop
    with nothing left to run it.
    """
    push_lock = _operational_push_lock()
    push_lock._lock_state = _known_state(LockStatus.UNLOCKED)

    async def _stop_then_fail(write_success_callback):
        write_success_callback()
        push_lock._running = False
        raise OperationIncompleteError("no op-response, and we were stopped")

    mock_lock = MagicMock()
    mock_lock.force_lock = AsyncMock(side_effect=_stop_then_fail)

    with (
        patch.object(push_lock, "_ensure_connected", AsyncMock(return_value=mock_lock)),
        pytest.raises(OperationIncompleteError),
    ):
        await push_lock.lock()

    assert push_lock._operation_window_open is False
    assert push_lock._cancel_deferred_update is None
    # The unknown position _finalize_operation would otherwise have applied.
    assert push_lock.lock_status == LockStatus.LOCKING
    push_lock._cancel_disconnect_timer()


@pytest.mark.asyncio
async def test_a_stop_mid_operation_keeps_the_owed_poll_and_the_floor():
    """The stopped exit still stamps the owed poll and the stale-state floor.

    Both are facts about the mechanism, so a watcher started again on this
    instance inherits them: its first cycle waits out the motor and asks the
    lock for the status the stopped operation never displayed.
    """
    push_lock = _operational_push_lock()
    push_lock._lock_state = _known_state(LockStatus.UNLOCKED)

    async def _stop_then_fail(write_success_callback):
        write_success_callback()
        push_lock._running = False
        raise OperationIncompleteError("no op-response, and we were stopped")

    mock_lock = MagicMock()
    mock_lock.force_lock = AsyncMock(side_effect=_stop_then_fail)

    with (
        patch.object(push_lock, "_ensure_connected", AsyncMock(return_value=mock_lock)),
        pytest.raises(OperationIncompleteError),
    ):
        await push_lock.lock()

    assert push_lock._force_lock_status_poll is True
    assert push_lock._earliest_update_time > time.monotonic()
    push_lock._cancel_disconnect_timer()


@pytest.mark.asyncio
async def test_a_jam_recorded_before_the_stop_reaches_no_later_operation():
    """A stopped operation discharges the jam record instead of leaving it set.

    _finalize_operation returns before it can apply the outcome, so the window
    closing is what clears the record. Were it left set, the next operation on this
    instance would find a jam from the stopped session and end its attempt
    ladder as OperationIncompleteError, displaying JAMMED for a mechanism the
    new command never reached.
    """
    push_lock = _operational_push_lock("aa:bb:cc:dd:ee:2d")
    push_lock._lock_state = _known_state(LockStatus.UNLOCKED)

    async def _jam_then_stop_then_fail(write_success_callback):
        write_success_callback()
        # Filtered by the open window, so the operation carries the record.
        push_lock._update_any_state([LockStatus.JAMMED])
        push_lock._running = False
        raise OperationIncompleteError("no op-response, and we were stopped")

    mock_lock = MagicMock()
    mock_lock.force_lock = AsyncMock(side_effect=_jam_then_stop_then_fail)

    with (
        patch.object(push_lock, "_ensure_connected", AsyncMock(return_value=mock_lock)),
        pytest.raises(OperationIncompleteError),
    ):
        await push_lock.lock()

    assert push_lock._seen_intervention_status is None

    # Started again, an operation that fails before its write-success keeps its
    # own retryable failure and stamps the unknown position; a leaked record
    # would raise OperationIncompleteError and display JAMMED instead.
    push_lock._running = True
    with (
        patch.object(
            push_lock,
            "_ensure_connected",
            AsyncMock(side_effect=DisconnectedError("no link")),
        ),
        patch("yalexs_ble.push.asyncio.sleep", AsyncMock()),
        pytest.raises(DisconnectedError),
    ):
        await push_lock.unlock()

    assert push_lock.lock_status == LockStatus.UNKNOWN
    push_lock._cancel_future_update()
    push_lock._cancel_disconnect_timer()


@pytest.mark.asyncio
async def test_jam_inside_the_window_outranks_a_successful_result() -> None:
    """A jam received while the window is open reaches the display.

    The filter drops the jam like every other status, but it is recorded, and
    the operation applies it at its exit in place of the complete_state its
    own command implies: the target state is inferred, the jam is a reading of
    where the mechanism stopped.
    """
    push_lock = _operational_push_lock()
    push_lock._lock_state = _known_state(LockStatus.UNLOCKED)

    mock_lock = MagicMock()

    async def force_lock(write_success_callback):
        write_success_callback()  # opens window, stamps LOCKING
        push_lock._update_any_state([LockStatus.JAMMED])
        assert push_lock.lock_status == LockStatus.LOCKING  # dropped mid-window

    mock_lock.force_lock = force_lock

    with patch.object(
        push_lock, "_ensure_connected", AsyncMock(return_value=mock_lock)
    ):
        await push_lock.lock()

    assert push_lock.lock_status == LockStatus.JAMMED
    assert push_lock._seen_intervention_status is None  # cleared when the window closed


def test_manual_intervention_statuses_is_public_and_complete() -> None:
    """The set is importable from the package and names all three statuses.

    Home Assistant maps calibration, polarity discovery and a jam onto one
    jammed attribute, and enumerates the three members itself today. This is
    the name it can import instead, so the membership is pinned here rather
    than left to whoever next edits the status table.
    """
    assert {
        LockStatus.UNKNOWN_01,
        LockStatus.UNKNOWN_06,
        LockStatus.JAMMED,
    } == yalexs_ble.MANUAL_INTERVENTION_STATUSES
    assert "MANUAL_INTERVENTION_STATUSES" in yalexs_ble.__all__


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "setup_condition",
    [LockStatus.UNKNOWN_01, LockStatus.UNKNOWN_06],
)
async def test_a_setup_condition_inside_the_window_reaches_the_display(
    setup_condition: LockStatus,
) -> None:
    """Calibration and polarity discovery survive the window as a jam does.

    The filter exists to keep mid-motion readings off the display, and the
    op-response is a true picture of where the mechanism stopped. Neither
    argument covers a lock that reports it needs setting up: that is not a
    transient reading, the op-response says nothing about it, and discarding
    it would leave the condition invisible until some later frame carried it
    again. So it is recorded like a jam and applied at the operation's exit.
    """
    push_lock = _operational_push_lock()
    push_lock._lock_state = _known_state(LockStatus.UNLOCKED)

    mock_lock = MagicMock()

    async def force_lock(write_success_callback):
        write_success_callback()  # opens window, stamps LOCKING
        push_lock._update_any_state([setup_condition])
        assert push_lock.lock_status == LockStatus.LOCKING  # dropped mid-window

    mock_lock.force_lock = force_lock

    with patch.object(
        push_lock, "_ensure_connected", AsyncMock(return_value=mock_lock)
    ):
        await push_lock.lock()

    # The reported condition itself, not a stand-in: a consumer reads
    # MANUAL_INTERVENTION_STATUSES to decide how to present it.
    assert push_lock.lock_status is setup_condition
    assert push_lock._seen_intervention_status is None  # cleared when the window closed


@pytest.mark.asyncio
async def test_jam_inside_the_window_replaces_the_unknown_of_a_lost_result() -> None:
    """A result that never arrives settles on UNKNOWN, unless a jam was
    reported inside the window: a position beats an unknown position."""
    push_lock = _operational_push_lock()
    push_lock._lock_state = _known_state(LockStatus.UNLOCKED)

    mock_lock = MagicMock()

    async def force_lock(write_success_callback):
        write_success_callback()
        push_lock._update_any_state([LockStatus.JAMMED])
        raise OperationIncompleteError("no op-response")

    mock_lock.force_lock = force_lock

    with (
        patch.object(push_lock, "_ensure_connected", AsyncMock(return_value=mock_lock)),
        pytest.raises(OperationIncompleteError),
    ):
        await push_lock.lock()

    assert push_lock.lock_status == LockStatus.JAMMED
    assert push_lock._operation_window_open is False
    assert push_lock._seen_intervention_status is None


@pytest.mark.asyncio
async def test_jam_inside_the_window_ends_the_attempt_ladder() -> None:
    """A retryable failure after a jam was reported does not re-send.

    The retryable types mean the command may never have been delivered, so the
    next attempt would write it again and drive the motor into a mechanism the
    lock has just reported jammed. The attempt ladder ends instead, with the
    jam on display and OperationIncompleteError to the caller.
    """
    push_lock = _operational_push_lock()
    push_lock._lock_state = _known_state(LockStatus.UNLOCKED)
    attempts = 0

    async def force_lock(write_success_callback):
        nonlocal attempts
        attempts += 1
        write_success_callback()
        push_lock._update_any_state([LockStatus.JAMMED])
        raise DisconnectedError("dropped after the jam was reported")

    mock_lock = MagicMock()
    mock_lock.force_lock = force_lock

    with (
        patch.object(push_lock, "_ensure_connected", AsyncMock(return_value=mock_lock)),
        patch("yalexs_ble.push.asyncio.sleep", AsyncMock()),
        pytest.raises(OperationIncompleteError),
    ):
        await push_lock.lock()

    assert attempts == 1  # the command was written once, and not again
    assert push_lock.lock_status == LockStatus.JAMMED
    assert push_lock._operation_window_open is False
    assert push_lock._seen_intervention_status is None


@pytest.mark.asyncio
async def test_a_new_commands_write_success_clears_the_jam_record() -> None:
    """A command that reaches the lock supersedes a jam still on record.

    No exit leaves _seen_intervention_status set for the next command to
    find: _close_operation_window clears it above the stop check. The record is
    seeded by hand here to pin the backstop: were one ever to reach a
    write-success, the command the caller issued after the jam is the
    intervention that status calls for, and its own result is what the display
    carries.
    """
    push_lock = _operational_push_lock()
    push_lock._lock_state = _known_state(LockStatus.JAMMED)
    push_lock._seen_intervention_status = LockStatus.JAMMED
    seen_at_write_success: list[LockStatus | None] = []

    async def force_unlock(write_success_callback):
        write_success_callback()
        seen_at_write_success.append(push_lock._seen_intervention_status)

    mock_lock = MagicMock()
    mock_lock.force_unlock = force_unlock

    with patch.object(
        push_lock, "_ensure_connected", AsyncMock(return_value=mock_lock)
    ):
        await push_lock.unlock()

    assert seen_at_write_success == [None]
    assert push_lock.lock_status == LockStatus.UNLOCKED


@pytest.mark.asyncio
async def test_queued_operation_emits_no_transitional_until_dequeued() -> None:
    """A second operation queued on the operation lock stamps nothing.

    With the first operation in flight, force_lock blocked on an Event, a
    second lock() is issued. While it waits on the operation lock it emits no
    transitional; its LOCKING appears only after the first operation's
    completed state.
    """
    push_lock = _operational_push_lock("aa:bb:cc:dd:ee:3a")
    emissions: list[LockStatus] = []
    push_lock.register_callback(lambda ls, li, ci: emissions.append(ls.lock))

    gate = asyncio.Event()
    calls = 0

    async def gated_force_lock(write_success_callback: Callable[[], None]) -> None:
        nonlocal calls
        calls += 1
        write_success_callback()
        if calls == 1:
            await gate.wait()  # hold op1 open while op2 queues behind it

    mock_lock = MagicMock()
    mock_lock.force_lock = gated_force_lock

    with (
        patch.object(push_lock, "_ensure_connected", AsyncMock(return_value=mock_lock)),
        patch.object(push_lock, "_schedule_future_update"),
    ):
        op1 = asyncio.create_task(push_lock.lock())
        for _ in range(50):
            await asyncio.sleep(0)
            if emissions:
                break
        # op1 has stamped its single LOCKING at write-success.
        assert emissions == [LockStatus.LOCKING]

        op2 = asyncio.create_task(push_lock.lock())
        for _ in range(10):
            await asyncio.sleep(0)
        # op2 is queued on the operation lock: no transitional while queued.
        assert emissions == [LockStatus.LOCKING]

        gate.set()
        await op1
        await op2

    # op2's LOCKING appears only AFTER op1's completion state (LOCKED).
    assert emissions == [
        LockStatus.LOCKING,
        LockStatus.LOCKED,
        LockStatus.LOCKING,
        LockStatus.LOCKED,
    ]
    push_lock._cancel_disconnect_timer()


@pytest.mark.asyncio
async def test_window_filter_drops_lock_status_admits_the_door_member() -> None:
    """Literal filter opened via write-success: even mid-window JAMMED is dropped.

    Distinct from the existing window tests (which set the flag directly and do
    not feed JAMMED): the window is opened through the real
    _operation_write_success path, and mid-window JAMMED is dropped with NO
    special-casing, while the door member of the same frame still passes.
    """
    push_lock = _operational_push_lock("aa:bb:cc:dd:ee:3b")

    # Open the window as write-success would: stamp the pending transitional.
    push_lock._pending_op_state = LockStatus.LOCKING
    push_lock._operation_write_success()
    assert push_lock.lock_status is LockStatus.LOCKING
    assert push_lock._operation_window_open is True

    # A foreign settle mid-window is dropped (literal filter).
    push_lock._update_any_state([LockStatus.LOCKED])
    assert push_lock.lock_status is LockStatus.LOCKING

    # Foreign jam evidence mid-window is dropped too, with no special-casing.
    push_lock._update_any_state([LockStatus.JAMMED])
    assert push_lock.lock_status is LockStatus.LOCKING

    # A door event is the one status the lock sends during an operation, and it
    # still reaches the display: only the lock member goes through
    # _admit_lock_status.
    push_lock._update_any_state([DoorStatus.OPENED])
    assert push_lock.door_status is DoorStatus.OPENED


@pytest.mark.asyncio
async def test_early_disconnect_retries_to_exhaustion_then_stamps_unknown() -> None:
    """A retryable disconnect before any write-success retries to exhaustion.

    Distinct from the existing early-error test (which checks only the final
    display): this pins the retry COUNT, force_lock re-sent exactly
    DEFAULT_ATTEMPTS times because the write never reached the lock, and the
    single emission the whole operation produces, the UNKNOWN of the exhausted
    attempts with no transitional ahead of it.
    """
    push_lock = _operational_push_lock("aa:bb:cc:dd:ee:3c")
    # Seed a settled position: the fixture starts at UNKNOWN, where the stamp
    # would change nothing and emit nothing.
    push_lock._lock_state = _known_state(LockStatus.LOCKED)
    emissions: list[LockStatus] = []
    push_lock.register_callback(lambda ls, li, ci: emissions.append(ls.lock))

    mock_lock = MagicMock()
    mock_lock.force_lock = AsyncMock(side_effect=DisconnectedError("boom"))

    with (
        patch.object(push_lock, "_ensure_connected", AsyncMock(return_value=mock_lock)),
        patch.object(push_lock, "_schedule_future_update"),
        patch.object(push_lock, "_async_handle_disconnected", AsyncMock()),
        patch("yalexs_ble.push.asyncio.sleep", AsyncMock()),
        pytest.raises(DisconnectedError),
    ):
        await push_lock.lock()

    # Retried to exhaustion (each attempt re-sends because the write never
    # reached the lock)...
    assert mock_lock.force_lock.await_count == DEFAULT_ATTEMPTS
    # ...and the operation emitted once, at the end: no write, so no
    # transitional, and the exhausted attempts leave the position unknown.
    assert emissions == [LockStatus.UNKNOWN]
    assert push_lock.lock_status is LockStatus.UNKNOWN
    assert push_lock._operation_window_open is False
    push_lock._cancel_disconnect_timer()


@pytest.mark.asyncio
async def test_operation_incomplete_after_write_success_stamps_unknown_once() -> None:
    """OperationIncompleteError after write-success: exact emission order, no retry.

    Distinct from the existing non-retryable test (which checks only the final
    display): this pins the full emission SEQUENCE [LOCKING, UNKNOWN] and that
    the non-retryable error ran force_lock exactly once.
    """
    push_lock = _operational_push_lock("aa:bb:cc:dd:ee:3d")
    emissions: list[LockStatus] = []
    push_lock.register_callback(lambda ls, li, ci: emissions.append(ls.lock))

    def force_lock_side_effect(write_success_callback: Callable[[], None]) -> None:
        # The write reached the lock (window opens, LOCKING on display) but the
        # result never arrived.
        write_success_callback()
        raise OperationIncompleteError("no op-response")

    mock_lock = MagicMock()
    mock_lock.force_lock = AsyncMock(side_effect=force_lock_side_effect)

    with (
        patch.object(push_lock, "_ensure_connected", AsyncMock(return_value=mock_lock)),
        patch.object(push_lock, "_schedule_future_update"),
        pytest.raises(OperationIncompleteError),
    ):
        await push_lock.lock()

    # Non-retryable: force_lock ran exactly once.
    assert mock_lock.force_lock.await_count == 1
    # LOCKING at write-success, then UNKNOWN when the result never came.
    assert emissions == [LockStatus.LOCKING, LockStatus.UNKNOWN]
    assert push_lock.lock_status is LockStatus.UNKNOWN
    assert push_lock._operation_window_open is False
    push_lock._cancel_disconnect_timer()


@pytest.mark.asyncio
async def test_a_retryable_failure_disconnects_before_the_retry() -> None:
    """A retryable failure tears the session down before the backoff sleep.

    The retry arm awaits _async_handle_disconnected before it sleeps the
    backoff and re-enters the operation, so the gap between attempts has no
    session to deliver a frame.
    """
    push_lock = _operational_push_lock("aa:bb:cc:dd:ee:68")
    push_lock._lock_state = _known_state(LockStatus.UNLOCKED)

    attempts = 0
    order: list[str] = []

    async def force_lock(write_success_callback: Callable[[], None]) -> None:
        nonlocal attempts
        attempts += 1
        order.append(f"attempt {attempts}")
        if attempts == 1:
            write_success_callback()
            raise DisconnectedError("dropped after the write")
        raise OperationIncompleteError("no op-response")

    async def record_teardown(_exc: Exception) -> None:
        order.append("teardown")

    async def record_backoff(_delay: float) -> None:
        order.append("backoff")

    mock_lock = MagicMock()
    mock_lock.force_lock = force_lock

    with (
        patch.object(push_lock, "_ensure_connected", AsyncMock(return_value=mock_lock)),
        patch.object(push_lock, "_schedule_future_update"),
        patch.object(
            push_lock,
            "_async_handle_disconnected",
            AsyncMock(side_effect=record_teardown),
        ),
        patch("yalexs_ble.push.asyncio.sleep", AsyncMock(side_effect=record_backoff)),
        pytest.raises(OperationIncompleteError),
    ):
        await push_lock.lock()

    assert order == ["attempt 1", "teardown", "backoff", "attempt 2"]
    push_lock._cancel_disconnect_timer()


@pytest.mark.asyncio
async def test_cancelled_mid_operation_closes_window_without_unknown() -> None:
    """A bare CancelledError mid-operation closes the window and re-raises.

    CancelledError is a BaseException, so the generic ``except Exception`` never
    sees it; without the dedicated arm the operation window would leak open and
    the transitional freeze on display, with no poll able to replace it. A cancel is not
    evidence the lock did or did not move, so, unlike the non-retryable
    OperationIncompleteError path, NO UNKNOWN is stamped; the transitional
    stays until the next poll settles it.
    """
    push_lock = _operational_push_lock()
    push_lock._lock_state = _known_state(LockStatus.UNLOCKED)
    emissions: list[LockStatus] = []
    push_lock.register_callback(lambda ls, li, ci: emissions.append(ls.lock))

    async def force_lock(write_success_callback: Callable[[], None]) -> None:
        write_success_callback()  # opens window, stamps LOCKING
        raise asyncio.CancelledError

    mock_lock = MagicMock()
    mock_lock.force_lock = force_lock

    with (
        patch.object(push_lock, "_ensure_connected", AsyncMock(return_value=mock_lock)),
        pytest.raises(asyncio.CancelledError),
    ):
        await push_lock.lock()

    assert push_lock._operation_window_open is False
    # No UNKNOWN stamp; the transitional persists until a poll's reading replaces it.
    assert LockStatus.UNKNOWN not in emissions
    assert push_lock.lock_status == LockStatus.LOCKING


@pytest.mark.asyncio
async def test_cancelled_mid_operation_displays_a_jam_it_received() -> None:
    """A cancel applies a jam the window filtered out, as every other exit does.

    A cancel is not evidence the lock did or did not move, which is why the test
    above pins that no status of the operation's own is stamped. A jam is
    evidence, and this exit is the only chance to apply it: the post-jam
    register fabricates a plain position, so the status poll this exit schedules puts
    that on display instead, and no later signal marks the jam.
    """
    push_lock = _operational_push_lock("aa:bb:cc:dd:ee:4f")
    push_lock._lock_state = _known_state(LockStatus.UNLOCKED)

    async def force_lock(write_success_callback: Callable[[], None]) -> None:
        write_success_callback()  # opens the window, stamps LOCKING
        push_lock._update_any_state([LockStatus.JAMMED])
        assert push_lock.lock_status == LockStatus.LOCKING  # dropped mid-window
        raise asyncio.CancelledError

    mock_lock = MagicMock()
    mock_lock.force_lock = force_lock

    with (
        patch.object(push_lock, "_ensure_connected", AsyncMock(return_value=mock_lock)),
        pytest.raises(asyncio.CancelledError),
    ):
        await push_lock.lock()

    assert push_lock.lock_status == LockStatus.JAMMED
    assert push_lock._operation_window_open is False
    assert push_lock._seen_intervention_status is None  # cleared when the window closed
    push_lock._cancel_future_update()
    push_lock._cancel_disconnect_timer()


@pytest.mark.asyncio
async def test_operation_outside_the_gate_cannot_open_the_window():
    """A force_* issued straight at the Lock leaves the operation window shut.

    Only _execute_lock_operation hands over the write-success hook, and only
    that method closes the window again. A caller reaching the Lock directly
    therefore gets no hook and cannot leave a window open with no operation in
    flight, which would freeze status admission for the object's life.
    """
    push_lock = _operational_push_lock("aa:bb:cc:dd:ee:52")
    push_lock._ble_device = MagicMock()
    lock = push_lock._get_lock_instance()
    lock.session = MagicMock()
    lock.secure_session = MagicMock()
    lock.client = MagicMock(is_connected=True)

    handed: list[object] = []

    async def _capture(
        command: bytearray,
        command_name: str,
        response_timeout: float = 0.0,
        write_success_callback: Callable[[], None] | None = None,
    ) -> None:
        handed.append(write_success_callback)
        if write_success_callback is not None:
            write_success_callback()

    lock._execute_operation_command = _capture  # type: ignore[method-assign]

    await lock.force_lock()

    assert handed == [None]
    assert push_lock._operation_window_open is False


@pytest.mark.asyncio
async def test_execute_lock_operation_hands_the_hook_to_the_operation():
    """The gated path passes its own write-success hook to the operation."""
    push_lock = _operational_push_lock("aa:bb:cc:dd:ee:53")
    received: list[object] = []

    async def force_lock(write_success_callback):
        received.append(write_success_callback)

    mock_lock = MagicMock()
    mock_lock.force_lock = force_lock

    with patch.object(
        push_lock, "_ensure_connected", AsyncMock(return_value=mock_lock)
    ):
        await push_lock.lock()

    assert received == [push_lock._operation_write_success]


def _answering_lock(push_lock: PushLock) -> MagicMock:
    """A mock Lock that answers every member an on-demand update cycle asks for.

    Each read answers the way the real one does: the session hands the frame to
    the state path before it resolves the waiter the read is blocked on, so the
    reading is applied through _update_any_state and the returned value carries
    nothing the state does not already hold.
    """
    mock_lock = MagicMock()

    async def lock_status() -> LockStatus:
        push_lock._update_any_state([LockStatus.LOCKED])
        return LockStatus.LOCKED

    async def door_status() -> DoorStatus:
        push_lock._update_any_state([DoorStatus.CLOSED])
        return DoorStatus.CLOSED

    async def battery() -> BatteryState:
        reading = BatteryState(voltage=6.0, percentage=80)
        push_lock._update_any_state([reading])
        return reading

    mock_lock.lock_status = AsyncMock(side_effect=lock_status)
    mock_lock.door_status = AsyncMock(side_effect=door_status)
    mock_lock.battery = AsyncMock(side_effect=battery)
    return mock_lock


@pytest.mark.asyncio
async def test_failed_operation_leaves_the_status_poll_armed() -> None:
    """A failed operation must not count as having polled the lock status.

    The join the suite was missing: the UNKNOWN _execute_lock_operation stamps
    on a non-retryable failure, and the poll gate in _update that reads
    _seen_this_session, exercised in one test with nothing seeded by hand. In
    on-demand mode nothing reconnects after this error, so a suppressed poll
    would leave UNKNOWN on display until the idle disconnect.
    """
    push_lock = _operational_push_lock("aa:bb:cc:dd:ee:40")
    push_lock._lock_state = _known_state(LockStatus.UNLOCKED)
    mock_lock = _answering_lock(push_lock)

    async def force_lock(write_success_callback: Callable[[], None]) -> None:
        write_success_callback()  # opens the window, stamps LOCKING
        raise OperationIncompleteError("no op-response")

    mock_lock.force_lock = force_lock

    with (
        patch.object(push_lock, "_ensure_connected", AsyncMock(return_value=mock_lock)),
        patch.object(
            push_lock, "_read_auto_lock_setting", AsyncMock(return_value=False)
        ),
        patch.object(push_lock, "_schedule_future_update"),
        pytest.raises(OperationIncompleteError),
    ):
        await push_lock.lock()

    assert push_lock.lock_status == LockStatus.UNKNOWN
    assert LockStatus not in push_lock._seen_this_session

    # The following cycle asks the lock for its status, and its reading replaces
    # the UNKNOWN.
    with (
        patch.object(push_lock, "_ensure_connected", AsyncMock(return_value=mock_lock)),
        patch.object(
            push_lock, "_read_auto_lock_setting", AsyncMock(return_value=False)
        ),
    ):
        await push_lock._update()

    mock_lock.lock_status.assert_awaited_once()
    # Read through _lock_state rather than the lock_status property: an
    # earlier assert in this test narrows that property to the status it
    # held before the cycle, which would make everything below unreachable.
    final_state = push_lock._lock_state
    assert final_state is not None
    assert final_state.lock == LockStatus.LOCKED
    push_lock._cancel_disconnect_timer()


@pytest.mark.asyncio
async def test_cancelled_operation_leaves_the_status_poll_armed() -> None:
    """Cancellation mid-operation leaves the transitional on display, and the
    following cycle polls it away.

    The CancelledError handler closes the window so acceptance resumes; this
    pins the other half of that claim, that a later poll's reading replaces it.
    """
    push_lock = _operational_push_lock("aa:bb:cc:dd:ee:41")
    push_lock._lock_state = _known_state(LockStatus.UNLOCKED)
    mock_lock = _answering_lock(push_lock)
    gate = asyncio.Event()

    async def force_lock(write_success_callback: Callable[[], None]) -> None:
        write_success_callback()  # opens the window, stamps LOCKING
        await gate.wait()

    mock_lock.force_lock = force_lock

    with (
        patch.object(push_lock, "_ensure_connected", AsyncMock(return_value=mock_lock)),
        patch.object(
            push_lock, "_read_auto_lock_setting", AsyncMock(return_value=False)
        ),
        patch.object(push_lock, "_schedule_future_update"),
    ):
        op = asyncio.create_task(push_lock.lock())
        for _ in range(50):
            await asyncio.sleep(0)
            if push_lock.lock_status == LockStatus.LOCKING:
                break
        op.cancel()
        with pytest.raises(asyncio.CancelledError):
            await op

        assert push_lock.lock_status == LockStatus.LOCKING
        assert push_lock._operation_window_open is False
        assert LockStatus not in push_lock._seen_this_session

        await push_lock._update()

    mock_lock.lock_status.assert_awaited_once()
    # Read through _lock_state rather than the lock_status property: an
    # earlier assert in this test narrows that property to the status it
    # held before the cycle, which would make everything below unreachable.
    final_state = push_lock._lock_state
    assert final_state is not None
    assert final_state.lock == LockStatus.LOCKED
    push_lock._cancel_disconnect_timer()


@pytest.mark.asyncio
async def test_a_refused_reading_leaves_the_status_poll_owed() -> None:
    """A reading the window refuses does not stand in for a published status.

    _seen_this_session suppresses the follow-up status poll for a value the
    cycle already carries. Nothing published the refused reading, so keeping
    the mark would answer the poll obligation with a status no consumer saw.
    """
    push_lock = _operational_push_lock("aa:bb:cc:dd:ee:6c")
    push_lock._lock_state = _known_state(LockStatus.UNLOCKED)
    push_lock._pending_op_state = LockStatus.UNLOCKING
    push_lock._operation_write_success()  # opens the window

    push_lock._update_any_state([LockStatus.UNLOCKED])

    assert push_lock.lock_status is LockStatus.UNLOCKING  # the reading was refused
    assert LockStatus not in push_lock._seen_this_session
    push_lock._cancel_disconnect_timer()


@pytest.mark.asyncio
async def test_exhausted_retries_after_write_success_stamp_unknown() -> None:
    """Every attempt's write reaches the lock, and every attempt then fails.

    Through the retries the transitional stays on display because the next
    attempt re-stamps it. Once the attempts run out there is no next attempt,
    so the operation applies UNKNOWN to the display rather than leaving a
    transitional standing with no result coming.
    """
    push_lock = _operational_push_lock("aa:bb:cc:dd:ee:42")
    push_lock._lock_state = _known_state(LockStatus.UNLOCKED)
    emissions: list[LockStatus] = []
    push_lock.register_callback(lambda ls, li, ci: emissions.append(ls.lock))
    calls = 0

    async def force_lock(write_success_callback: Callable[[], None]) -> None:
        nonlocal calls
        calls += 1
        write_success_callback()  # opens the window, stamps LOCKING
        raise DisconnectedError("dropped after the write, before the ack")

    mock_lock = MagicMock()
    mock_lock.force_lock = force_lock

    with (
        patch.object(push_lock, "_ensure_connected", AsyncMock(return_value=mock_lock)),
        patch.object(push_lock, "_async_handle_disconnected", AsyncMock()),
        patch.object(push_lock, "_schedule_future_update"),
        patch("yalexs_ble.push.asyncio.sleep", AsyncMock()),
        pytest.raises(DisconnectedError),
    ):
        await push_lock.lock()

    assert calls == DEFAULT_ATTEMPTS
    # One transitional across all the attempts, then UNKNOWN when they run out.
    assert emissions == [LockStatus.LOCKING, LockStatus.UNKNOWN]
    assert push_lock._operation_window_open is False
    # UNKNOWN is not a reading, so the follow-up poll stays armed.
    assert LockStatus not in push_lock._seen_this_session
    # The attempts ran out with no result, so this exit schedules the poll itself.
    assert push_lock._force_lock_status_poll is True
    push_lock._cancel_disconnect_timer()


@pytest.mark.asyncio
async def test_no_update_cycle_is_armed_inside_an_operation() -> None:
    """Nothing the operation applies arms a cycle while the operation runs.

    A cycle armed here waits on the operation lock and runs the instant the
    operation ends, inside the post-operation debounce delay the stale-state
    guard exists to protect. The transitional and the completed status are
    applied by the operation itself, so they arm nothing; the status poll is
    scheduled once the operation is over.
    """
    push_lock = _operational_push_lock("aa:bb:cc:dd:ee:43")
    # A known prior status, so a change to the transitional would arm a resync.
    push_lock._lock_state = _known_state(LockStatus.UNLOCKED)
    armed_during: list[float] = []

    async def force_lock(write_success_callback: Callable[[], None]) -> None:
        write_success_callback()  # opens the window, stamps LOCKING
        if push_lock._cancel_deferred_update is not None:
            armed_during.append(
                push_lock._cancel_deferred_update.when() - push_lock.loop.time()
            )

    mock_lock = MagicMock()
    mock_lock.force_lock = force_lock

    with patch.object(
        push_lock, "_ensure_connected", AsyncMock(return_value=mock_lock)
    ):
        await push_lock.lock()

    assert armed_during == []
    assert push_lock._update_task is None
    assert push_lock.lock_status == LockStatus.LOCKED
    push_lock._cancel_future_update()
    push_lock._cancel_disconnect_timer()


@pytest.mark.asyncio
async def test_the_operation_cancels_the_pending_update_on_the_way_in() -> None:
    """A deferred update armed before the operation is gone once it starts.

    _run_lock_operation cancels the pending update on the way in, before
    anything awaits, so a cycle scheduled before the command cannot fire while
    the operation is connecting and poll the lock mid-command. The probe runs
    inside _ensure_connected, the operation's first await, so only the entry
    cancel can have cleared the pending cycle by then.
    """
    push_lock = _operational_push_lock("aa:bb:cc:dd:ee:63")
    push_lock._lock_state = _known_state(LockStatus.UNLOCKED)
    push_lock._schedule_future_update(30.0)
    assert push_lock._cancel_deferred_update is not None

    mock_lock = MagicMock()
    mock_lock.force_lock = AsyncMock(return_value=True)
    pending_at_connect: list[bool] = []

    async def connected() -> MagicMock:
        pending_at_connect.append(push_lock._cancel_deferred_update is not None)
        return mock_lock

    with (
        patch.object(push_lock, "_ensure_connected", AsyncMock(side_effect=connected)),
        patch.object(push_lock, "_schedule_future_update"),
    ):
        await push_lock.lock()

    # The pending cycle was already gone when the operation connected.
    assert pending_at_connect == [False]
    push_lock._cancel_disconnect_timer()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("error", "jam", "delay"),
    [
        (None, False, KEEP_ALIVE_TIME),
        (
            OperationIncompleteError("no op-response"),
            False,
            LOCK_STALE_STATE_DEBOUNCE_DELAY,
        ),
        (
            DisconnectedError("dropped after the jam was reported"),
            True,
            KEEP_ALIVE_TIME,
        ),
    ],
    ids=["success", "no-result", "jam-ends-the-ladder"],
)
async def test_every_operation_exit_schedules_the_status_poll(
    error: Exception | None, jam: bool, delay: float
) -> None:
    """Every way an operation can end schedules the follow-up status poll.

    Success, a result that never came, and a jam that ends the attempt ladder
    all leave through the same settle, so all three schedule a poll. The delay
    follows the status each one leaves on display: a position is polled at the
    keep-alive interval, the cadence an always-connected lock polls at anyway,
    and the UNKNOWN of a result that never came is polled at the settle
    debounce, because only a read can replace it.
    """
    push_lock = _operational_push_lock("aa:bb:cc:dd:ee:44")
    push_lock._lock_state = _known_state(LockStatus.UNLOCKED)

    async def force_lock(write_success_callback: Callable[[], None]) -> None:
        write_success_callback()  # opens the window, stamps LOCKING
        if jam:
            push_lock._update_any_state([LockStatus.JAMMED])
        if error is not None:
            raise error

    mock_lock = MagicMock()
    mock_lock.force_lock = force_lock

    with (
        patch.object(push_lock, "_ensure_connected", AsyncMock(return_value=mock_lock)),
        patch.object(push_lock, "_schedule_future_update") as scheduled,
    ):
        if error is None:
            await push_lock.lock()
        else:
            # The jam ends the ladder with OperationIncompleteError, whatever
            # retryable type reached it.
            with pytest.raises(OperationIncompleteError if jam else type(error)):
                await push_lock.lock()

    assert push_lock._force_lock_status_poll is True
    scheduled.assert_called_once_with(delay)
    push_lock._cancel_disconnect_timer()


@pytest.mark.asyncio
async def test_the_status_poll_keeps_a_sooner_pending_update() -> None:
    """The status poll inherits the debounce, so a sooner update is kept.

    _finalize_operation schedules the status poll through the deferred-update
    debounce. An update already due sooner than the keep-alive interval keeps
    its slot; an undebounced request would displace it a full interval out.
    Keeping it costs nothing, because the floor _finalize_operation stamps
    holds the poll itself back: the sooner cycle falls due, finds the floor, and
    re-arms for the remainder instead of reading the lock before it has passed.
    """
    push_lock = _operational_push_lock("aa:bb:cc:dd:ee:64")
    push_lock._schedule_future_update(1.0)
    handle = push_lock._cancel_deferred_update
    assert handle is not None

    push_lock._finalize_operation()

    assert push_lock._force_lock_status_poll is True
    # The sooner cycle is untouched: same handle, still due within a second.
    assert push_lock._cancel_deferred_update is handle
    remaining = handle.when() - push_lock.loop.time()
    assert 0 < remaining <= 1.0

    # It fires early and reads nothing: the floor re-arms it for its remainder.
    push_lock._deferred_update()

    assert push_lock._update_task is None
    rearmed = push_lock._cancel_deferred_update
    assert rearmed is not None and rearmed is not handle
    assert (
        LOCK_STALE_STATE_DEBOUNCE_DELAY - 1.0
        < rearmed.when() - push_lock.loop.time()
        <= LOCK_STALE_STATE_DEBOUNCE_DELAY
    )
    push_lock._cancel_future_update()


@pytest.mark.asyncio
async def test_a_collapsed_schedule_still_waits_for_the_floor() -> None:
    """A cycle the debounce shortens cannot poll the lock before the floor.

    An update already due inside the coalescing interval makes the debounce
    rewrite any request to that interval, so the poll _finalize_operation asks
    for is not the delay the cycle gets. The floor is what holds the poll: the
    collapsed request is armed for the remainder of that delay instead
    of firing 25 ms after the operation, when the lock still reports the
    pre-operation position.
    """
    push_lock = _operational_push_lock("aa:bb:cc:dd:ee:65")
    push_lock._lock_state = _known_state(LockStatus.UNLOCKED)
    push_lock._schedule_future_update(0.01)

    push_lock._finalize_operation()

    handle = push_lock._cancel_deferred_update
    assert handle is not None
    assert (
        LOCK_STALE_STATE_DEBOUNCE_DELAY - 0.5
        < handle.when() - push_lock.loop.time()
        <= LOCK_STALE_STATE_DEBOUNCE_DELAY
    )
    push_lock._cancel_future_update()


@pytest.mark.asyncio
async def test_cancellation_polls_sooner_than_the_keep_alive() -> None:
    """A cancelled operation schedules its own status poll, and sooner.

    A cancel leaves a transitional on display with no result coming while the
    link is still up, so this poll is deliberately not the keep-alive one. It
    is armed at the post-operation debounce delay rather than the resync delay,
    because the written command is still driving the motor and an immediate read
    would return the pre-operation position. _finalize_operation stamps the floor
    at the same
    moment, so a cancel is held out for the same window a completion is,
    however the request itself is debounced.
    """
    push_lock = _operational_push_lock("aa:bb:cc:dd:ee:45")
    push_lock._lock_state = _known_state(LockStatus.UNLOCKED)

    async def force_lock(write_success_callback: Callable[[], None]) -> None:
        write_success_callback()  # opens the window, stamps LOCKING
        raise asyncio.CancelledError

    mock_lock = MagicMock()
    mock_lock.force_lock = force_lock

    with (
        patch.object(push_lock, "_ensure_connected", AsyncMock(return_value=mock_lock)),
        patch.object(push_lock, "_schedule_future_update") as scheduled,
        pytest.raises(asyncio.CancelledError),
    ):
        await push_lock.lock()

    assert push_lock._force_lock_status_poll is True
    scheduled.assert_called_once_with(LOCK_STALE_STATE_DEBOUNCE_DELAY)
    push_lock._cancel_disconnect_timer()


@pytest.mark.asyncio
async def test_force_lock_status_poll_reads_past_seen_this_session() -> None:
    """The scheduled status poll asks the lock with LockStatus in _seen_this_session.

    A status the operation applied, or one the display is holding, adds
    LockStatus to _seen_this_session in the ordinary way; a cycle that honored
    that would ask the lock nothing and the display would keep whatever it was
    holding.
    """
    push_lock = _operational_push_lock("aa:bb:cc:dd:ee:46")
    push_lock._lock_state = _known_state(LockStatus.UNLOCKED)
    push_lock._seen_this_session.add(LockStatus)
    push_lock._force_lock_status_poll = True
    mock_lock = _answering_lock(push_lock)

    with (
        patch.object(push_lock, "_ensure_connected", AsyncMock(return_value=mock_lock)),
        patch.object(
            push_lock, "_read_auto_lock_setting", AsyncMock(return_value=False)
        ),
    ):
        await push_lock._update()

    mock_lock.lock_status.assert_awaited_once()
    assert push_lock.lock_status == LockStatus.LOCKED
    # One-shot: the next cycle honours _seen_this_session again.
    assert push_lock._force_lock_status_poll is False
    push_lock._cancel_disconnect_timer()


@pytest.mark.asyncio
async def test_a_failed_status_read_leaves_the_poll_obligation_armed() -> None:
    """A lock_status read that raises leaves the forced poll flag set.

    The flag is one-shot, discharged only by a read that answered. Every
    attempt of this cycle fails at the read itself, so every retry must ask
    again, and once the retries run out the obligation must still stand for
    the next cycle.
    """
    push_lock = _operational_push_lock("aa:bb:cc:dd:ee:66")
    push_lock._lock_state = _known_state(LockStatus.UNLOCKED)
    push_lock._seen_this_session.add(LockStatus)
    push_lock._force_lock_status_poll = True
    mock_lock = _answering_lock(push_lock)
    mock_lock.lock_status = AsyncMock(side_effect=DisconnectedError("dropped mid-read"))

    with (
        patch.object(push_lock, "_ensure_connected", AsyncMock(return_value=mock_lock)),
        patch.object(
            push_lock, "_read_auto_lock_setting", AsyncMock(return_value=False)
        ),
        patch.object(push_lock, "_async_handle_disconnected", AsyncMock()),
        patch("yalexs_ble.push.asyncio.sleep", AsyncMock()),
        pytest.raises(DisconnectedError),
    ):
        await push_lock._update()

    # Only the flag can be asking here (the type is in _seen_this_session), so each
    # attempt proves the obligation was still set when it started.
    assert mock_lock.lock_status.await_count == DEFAULT_ATTEMPTS
    # The read never answered, so the obligation stands for the next cycle.
    assert push_lock._force_lock_status_poll is True
    push_lock._cancel_disconnect_timer()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "status",
    [
        LockStatus.LOCKING,
        LockStatus.UNLOCKING,
        LockStatus.UNLATCHING,
        LockStatus.UNLATCHED,
        LockStatus.UNKNOWN,
    ],
)
async def test_a_value_the_lock_is_not_holding_is_not_a_status_reading(
    status: LockStatus,
) -> None:
    """None of these is a position the lock stays in, so none suppresses the
    poll.

    Recording one in _seen_this_session would suppress the follow-up lock_status()
    poll in _update, and that poll's reading is what replaces it once the
    mechanism stops. UNLATCHED is here with the transitionals: the latch is
    open for its dwell and the lock leaves that state on its own.
    """
    push_lock = _operational_push_lock("aa:bb:cc:dd:ee:47")
    push_lock._lock_state = _known_state(LockStatus.LOCKED)

    push_lock._update_any_state([status], arm_resync=False)

    assert push_lock.lock_status == status
    assert LockStatus not in push_lock._seen_this_session


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "position",
    [
        LockStatus.LOCKED,
        LockStatus.UNLOCKED,
        LockStatus.SECUREMODE,
        LockStatus.JAMMED,
        LockStatus.UNKNOWN_01,
        LockStatus.UNKNOWN_06,
    ],
)
async def test_a_position_the_lock_holds_counts_as_a_status_reading(
    position: LockStatus,
) -> None:
    """The other side of the same set: each of these does suppress the poll.

    The lock stays in each of these until an operation or a person moves it,
    calibration and polarity discovery included, so the reading stands and
    asking again this session would tell us nothing new.
    """
    push_lock = _operational_push_lock("aa:bb:cc:dd:ee:67")
    push_lock._lock_state = _known_state(LockStatus.UNKNOWN)

    push_lock._update_any_state([position], arm_resync=False)

    assert push_lock.lock_status == position
    assert LockStatus in push_lock._seen_this_session
