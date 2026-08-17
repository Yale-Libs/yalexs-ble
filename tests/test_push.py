import asyncio
import itertools
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
    DEADLINE_WAKEUP_RETRY_DELAY,
    DEFAULT_ATTEMPTS,
    DISCONNECT_DELAY,
    HAP_FIRST_BYTE,
    JAMMED_HOLD_TIME,
    KEEP_ALIVE_TIME,
    LOCK_STALE_STATE_DEBOUNCE_DELAY,
    NEVER_TIME,
    NO_BATTERY_SUPPORT_MODELS,
    POST_OP_RESPONSE_DEBOUNCE_DELAY,
    RESYNC_DELAY,
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
    AuthError,
    DisconnectedError,
    OperationFailedError,
    OperationIncompleteError,
    OperationProgress,
    ResponseError,
    UnlatchError,
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

    Session._notify hands every frame to the state path before it resolves the
    waiter the read is blocked on, so a real read has already published its
    answer by the time the await returns. A mock that only sets a return value
    reproduces the call but not the answer, so reads are stubbed with this.
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
async def test_update_continues_after_battery_timeout():
    """
    Test that _update() continues and completes successfully
    even when battery() times out.

    Requirements:
    - battery() timeout does not fail entire update
    - lock_status/door_status/auto_lock_status still get called
    - final state has valid lock/door values (not UNKNOWN)
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
        final_state = await push_lock._update()

        # Battery call was attempted
        mock_lock.battery.assert_called_once()

        # Other status calls still happened
        mock_lock.door_status.assert_called_once()
        mock_lock.auto_lock_status.assert_called_once()
        mock_lock.lock_status.assert_called_once()

        # Final state has valid lock/door (from the successful calls)
        assert final_state.lock == LockStatus.LOCKED
        assert final_state.door == DoorStatus.CLOSED

        # Battery should be None since it timed out
        assert final_state.battery is None


@pytest.mark.asyncio
async def test_poll_battery_cooldown_skip():
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
    made_request = await push_lock._poll_battery(mock_lock)

    # Should skip the request
    assert made_request is False
    mock_lock.battery.assert_not_called()
    # Nothing was published
    assert push_lock.battery is None


@pytest.mark.asyncio
async def test_poll_battery_success():
    """Test that _poll_battery successfully fetches battery and resets cooldown."""
    push_lock = PushLock(
        address="aa:bb:cc:dd:ee:ff",
        key="0800200c9a66",
        key_index=1,
        always_connected=False,
    )
    push_lock._name = "Test Lock"
    push_lock._lock_info = TEST_LOCK_INFO

    # Set cooldown to simulate previous timeout
    push_lock._earliest_battery_attempt_time = time.monotonic() + 100.0

    mock_lock = MagicMock()
    battery_state = BatteryState(voltage=6.0, percentage=80)
    mock_lock.battery = publishing_read(push_lock, battery_state)

    # Call _poll_battery (cooldown should be ignored since it's in the future)
    # Wait a moment to ensure cooldown expires
    push_lock._earliest_battery_attempt_time = NEVER_TIME

    made_request = await push_lock._poll_battery(mock_lock)

    # Should make the request
    assert made_request is True
    mock_lock.battery.assert_called_once()

    # The reading reached the state through the receive path
    assert push_lock.battery == battery_state
    assert push_lock.auth is not None
    assert push_lock.auth.successful is True

    # Cooldown should be reset to NEVER_TIME
    assert push_lock._earliest_battery_attempt_time == NEVER_TIME


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

    1. Update starts with UNKNOWN state
    2. Notify delivers LOCKED/CLOSED while the cycle is awaiting a read
    3. Update skips polling lock_status (already seen this session)
    4. The delivered values are still on display when the cycle ends
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

        final_state = await update_task

        # The critical assertion: the delivered values are still on display
        assert final_state.lock == LockStatus.LOCKED, (
            f"Lock status should still be LOCKED, got {final_state.lock}"
        )
        assert final_state.door == DoorStatus.CLOSED, (
            f"Door status should still be CLOSED, got {final_state.door}"
        )


@pytest.mark.asyncio
async def test_update_does_not_revert_a_mid_cycle_change() -> None:
    """A notify arriving mid-cycle survives when the cycle started at a real value.

    The cycle begins with the lock LOCKED and the door CLOSED, so it holds a
    reading for both. A key turned by hand while the cycle is awaiting one of
    its reads delivers UNLOCKED and OPENED. Nothing in the cycle read a lock or
    door status, so it has nothing newer to say about either, and the delivered
    values must be what is on display when it ends.
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

    # Everything but the auto lock setting is already seen, so the cycle reads
    # no lock or door status of its own.
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
        allow_auto_lock_to_continue.set()

        final_state = await update_task
        push_lock._cancel_future_update()

    assert final_state.lock == LockStatus.UNLOCKED
    assert final_state.door == DoorStatus.OPENED
    assert push_lock.lock_status == LockStatus.UNLOCKED
    assert push_lock.door_status == DoorStatus.OPENED


@pytest.mark.asyncio
async def test_update_auto_lock_from_notify_path_survives_poll_result() -> None:
    """_update() carries the notify-published auto-lock into its final state.

    The auto-lock read's return value is the READSETTING acknowledgment
    constant (OFF) and is discarded; the stored setting arrives as the 0xBB
    settings response on the notify path during the cycle. The end-of-update
    restore must apply the notify-published value, not revert to the cycle's
    starting snapshot or the poll constant.
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

        final_state = await update_task

        assert final_state.auto_lock == AutoLockState(AutoLockMode.TIMER, 1800), (
            f"Auto-lock should be the notify-published value, "
            f"got {final_state.auto_lock}"
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
        final_state = await push_lock._update()

    # lock_info was attempted
    mock_lock.lock_info.assert_called_once()

    # Update still completed with real data
    assert final_state.lock == LockStatus.LOCKED

    # door_status not called because model="" makes door_sense=False
    mock_lock.door_status.assert_not_called()
    assert final_state.door == DoorStatus.UNKNOWN

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
        final_state = await push_lock._update()

    assert final_state.lock == LockStatus.LOCKED
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
        final_state = await push_lock._update()

    assert final_state.lock == LockStatus.LOCKED
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
        final_state = await push_lock._update()

    # Battery should have been re-polled
    mock_lock.battery.assert_called_once()
    assert final_state.battery == battery_state
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
        state = await push_lock._update()

    # The update returned instead of raising: no forced disconnect, and the
    # timeout was counted as a failure rather than propagated.
    assert state.lock == LockStatus.LOCKED
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


def _auto_lock_push_lock(address: str, *, always_connected: bool) -> PushLock:
    """A named PushLock for the auto lock read outcome tests."""
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
    push_lock = _auto_lock_push_lock(
        "aa:bb:cc:dd:ee:20", always_connected=always_connected
    )
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
    push_lock = _auto_lock_push_lock("aa:bb:cc:dd:ee:21", always_connected=True)
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
    push_lock = _auto_lock_push_lock("aa:bb:cc:dd:ee:2a", always_connected=True)

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
    push_lock = _auto_lock_push_lock("aa:bb:cc:dd:ee:23", always_connected=True)
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
    push_lock = _auto_lock_push_lock(
        "aa:bb:cc:dd:ee:24", always_connected=always_connected
    )
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
    push_lock = _auto_lock_push_lock("aa:bb:cc:dd:ee:29", always_connected=False)
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
    push_lock = _auto_lock_push_lock("aa:bb:cc:dd:ee:25", always_connected=False)

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
    push_lock = _auto_lock_push_lock("aa:bb:cc:dd:ee:26", always_connected=False)

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
    push_lock = _auto_lock_push_lock("aa:bb:cc:dd:ee:27", always_connected=False)
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
    push_lock = _auto_lock_push_lock(
        "aa:bb:cc:dd:ee:28", always_connected=always_connected
    )
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
    push_lock = PushLock(
        address="aa:bb:cc:dd:ee:14",
        key="0800200c9a66",
        key_index=1,
        always_connected=False,
    )
    push_lock._name = "Test Lock"
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
async def test_impossible_battery_voltage_is_refused_and_surfaced(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """The state path refuses a reading at or below 3.0 V and says so.

    3.0 V across four cells is 0.75 V each, against a table that already
    treats 1.24 V per cell as empty, so the frame is unexpected lock behaviour
    rather than a flat battery.
    """
    push_lock = PushLock(
        address="aa:bb:cc:dd:ee:15",
        key="0800200c9a66",
        key_index=1,
        always_connected=False,
    )
    push_lock._name = "Test Lock"

    with caplog.at_level(logging.WARNING, logger="yalexs_ble.push"):
        push_lock._update_any_state([BatteryState(voltage=2.5, percentage=0)])

    assert push_lock.battery is None
    assert "Battery voltage is impossible" in caplog.text
    # A refused reading must not suppress the next poll.
    assert BatteryState not in push_lock._seen_this_session


@pytest.mark.asyncio
async def test_a_battery_frame_reaches_the_display_through_the_real_decoder() -> None:
    """Pin the only route from a battery frame to the display.

    With the fetch return discarded, Lock._parse_state's BATTERY branch is the
    one join between a received battery frame and the published state, so this
    drives a real captured frame through the real decoder into the push
    layer's state path rather than stubbing either side.
    """
    push_lock = PushLock(
        address="aa:bb:cc:dd:ee:17",
        key="0800200c9a66",
        key_index=1,
        always_connected=False,
    )
    push_lock._name = "Test Lock"
    lock = Lock(
        lambda: BLEDevice("aa:bb:cc:dd:ee:17", "lock"),
        "0800200c9a66",
        1,
        "Test Lock",
        push_lock._state_callback,
    )

    lock._internal_state_callback(bytes.fromhex("bb0200a50f00000079140000000000000200"))

    battery = push_lock.battery
    assert battery is not None
    assert battery.voltage == 5.241
    assert battery.percentage == 28


@pytest.mark.asyncio
@pytest.mark.parametrize("reported", [LockStatus.UNKNOWN_01, LockStatus.UNKNOWN_06])
async def test_update_does_not_reconnect_on_an_unknown_position(
    reported: LockStatus,
) -> None:
    """A polled 0x01 or 0x06 reaches the display and forces no reconnect."""
    push_lock = PushLock(
        address="aa:bb:cc:dd:ee:16",
        key="0800200c9a66",
        key_index=1,
        always_connected=False,
    )
    push_lock._name = "Test Lock"
    push_lock._lock_info = TEST_LOCK_INFO
    push_lock._running = True

    mock_lock = MagicMock()
    mock_lock.lock_status = publishing_read(push_lock, reported)
    mock_lock.door_status = publishing_read(push_lock, DoorStatus.CLOSED)
    mock_lock.auto_lock_status = AsyncMock(return_value=None)
    mock_lock.battery = publishing_read(
        push_lock, BatteryState(voltage=6.0, percentage=80)
    )

    with (
        patch.object(push_lock, "_ensure_connected", AsyncMock(return_value=mock_lock)),
        patch.object(
            push_lock, "_execute_forced_disconnect", new_callable=AsyncMock
        ) as forced_disconnect,
    ):
        await push_lock._update()
        push_lock._cancel_future_update()

    forced_disconnect.assert_not_awaited()
    assert push_lock.lock_status is reported


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
    push_lock = PushLock(
        address="aa:bb:cc:dd:ee:28",
        key="0800200c9a66",
        key_index=1,
        always_connected=False,
    )
    push_lock._name = "Test Lock"
    ble_device = BLEDevice(push_lock.address, "Test Lock", None)

    push_lock.update_advertisement(ble_device, _advertisement(manufacturer_data))

    assert push_lock._cancel_deferred_update is None
    assert push_lock._last_hk_state == -1


@pytest.mark.asyncio
async def test_a_full_homekit_advertisement_still_reads_its_state_number() -> None:
    """The guard admits a payload long enough for the fields it reads."""
    push_lock = PushLock(
        address="aa:bb:cc:dd:ee:29",
        key="0800200c9a66",
        key_index=1,
        always_connected=False,
    )
    push_lock._name = "Test Lock"
    ble_device = BLEDevice(push_lock.address, "Test Lock", None)
    # <HHBB at byte 9: acid, then the global state number 0x1234.
    payload = (
        bytes([HAP_FIRST_BYTE]) + b"\x00" * 8 + struct.pack("<HHBB", 1, 0x1234, 0, 0)
    )

    push_lock.update_advertisement(ble_device, _advertisement({APPLE_MFR_ID: payload}))

    assert push_lock._last_hk_state == 0x1234
    assert push_lock._cancel_deferred_update is not None
    push_lock._cancel_future_update()


@pytest.mark.asyncio
async def test_a_cycle_that_changed_nothing_still_reports() -> None:
    """A quiet cycle still publishes once, because the callback is a liveness
    signal and not only a change notification.

    A consumer may mark the lock unavailable from its own advertisement
    tracking, with this callback as its only path back, so a cycle that read
    the same values as the last one must still report.
    """
    push_lock = PushLock(
        address="aa:bb:cc:dd:ee:30",
        key="0800200c9a66",
        key_index=1,
        always_connected=True,
    )
    push_lock._name = "Test Lock"
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

    # Everything the cycle could read is already held at the value it will
    # read back, so nothing in the cycle changes any member.
    push_lock._lock_state = LockState(
        lock=LockStatus.LOCKED,
        door=DoorStatus.CLOSED,
        battery=None,
        auth=AuthState(successful=True),
        auto_lock=None,
        auto_lock_prev=None,
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

    # The lock status was re-read and matched, so no member changed.
    mock_lock.lock_status.assert_awaited_once()
    assert push_lock.lock_status is LockStatus.LOCKED
    # The cycle still reported exactly once.
    assert len(published) == 1
    push_lock._running = False
    push_lock._cancel_disconnect_timer()


# ---------------------------------------------------------------------------
# Lock operations completed by their own op-response
# ---------------------------------------------------------------------------


def _operational_push_lock(address: str = "aa:bb:cc:dd:ee:20") -> PushLock:
    """A running lock with lock_info and advertisement data, ready to operate."""
    push_lock = PushLock(
        address=address,
        key="0800200c9a66",
        key_index=1,
        always_connected=False,
    )
    push_lock._name = "Test Lock"
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
    return push_lock


def _known_state(lock: LockStatus, door: DoorStatus = DoorStatus.CLOSED) -> LockState:
    """A settled cycle state, so an operation starts from a known position."""
    return LockState(
        lock=lock,
        door=door,
        battery=None,
        auth=None,
        auto_lock=None,
        auto_lock_prev=None,
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
    """A force_* that returns advances the state to the completed status.

    Drives each public operation to completion: the transitional is stamped,
    the op-response reports success, and the completed status is applied.
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
async def test_failure_op_response_applies_jammed_after_window():
    """The parser emits JAMMED for our own failure op-response mid-window (the
    state callback runs before session resolves the future), so it is dropped;
    the OperationFailedError path re-applies JAMMED after closing the window
    and propagates the failure.

    The op-response result driving it is 0x1F MECH_POSITION, which the Lock
    parser turns into LockStatus.JAMMED.
    """
    push_lock = _operational_push_lock()
    events: list[LockStatus] = []

    def cb(lock_state, lock_info, connection_info):
        events.append(lock_state.lock)

    push_lock.register_callback(cb)

    mock_lock = MagicMock()

    async def force_lock(write_success_callback):
        write_success_callback()  # opens window, stamps LOCKING
        # Parser emission of the failure op-response lands inside our window.
        push_lock._state_callback([LockStatus.JAMMED])
        # op-response byte[15] != 0
        raise OperationFailedError("force_lock reported failure", 0x1F)

    mock_lock.force_lock = force_lock

    with (
        patch.object(push_lock, "_ensure_connected", AsyncMock(return_value=mock_lock)),
        pytest.raises(OperationFailedError),
    ):
        await push_lock.lock()

    # LOCKING at write-success; the mid-window JAMMED produced no event; JAMMED
    # applied once, only after the window closed.
    assert events == [LockStatus.LOCKING, LockStatus.JAMMED]
    assert push_lock.lock_status == LockStatus.JAMMED
    # The exchange completed, so the failure path runs _complete_operation and
    # the settle stamps the floor: the disconnect timer and the earliest next
    # read both move despite the jam.
    assert push_lock._last_operation_complete_time != NEVER_TIME
    assert push_lock._earliest_update_time != NEVER_TIME
    push_lock._cancel_future_update()
    push_lock._cancel_disconnect_timer()


@pytest.mark.asyncio
async def test_reported_failure_stamps_the_stale_state_floor() -> None:
    """A reported failure stamps the floor exactly as a success does.

    The op-response ends the motor's movement whether byte[15] reported
    success or a fault, and the lock's reported state is settling from that
    moment either way, so the failure path holds a read back for the same
    debounce delay.
    """
    push_lock = _operational_push_lock("aa:bb:cc:dd:ee:22")
    mock_lock = MagicMock()
    mock_lock.force_lock = AsyncMock(
        side_effect=OperationFailedError("reported failure", 0x1F)
    )
    assert push_lock._earliest_update_time == NEVER_TIME

    before = time.monotonic()
    with (
        patch.object(push_lock, "_ensure_connected", AsyncMock(return_value=mock_lock)),
        pytest.raises(OperationFailedError),
    ):
        await push_lock.lock()
    after = time.monotonic()

    assert (
        before + LOCK_STALE_STATE_DEBOUNCE_DELAY
        <= push_lock._earliest_update_time
        <= after + LOCK_STALE_STATE_DEBOUNCE_DELAY
    )


@pytest.mark.asyncio
async def test_deferred_update_defers_inside_the_stale_state_window() -> None:
    """An update landing right after an operation completes is rescheduled.

    The lock's reported state is still settling for the debounce delay after
    the op-response, so a read issued now would return a stale state. A zero
    delay would let the read through immediately, so the guard must hold for
    a real window.
    """
    push_lock = _operational_push_lock("aa:bb:cc:dd:ee:23")
    push_lock._earliest_update_time = time.monotonic() + LOCK_STALE_STATE_DEBOUNCE_DELAY

    with patch.object(push_lock, "_schedule_future_update") as mock_reschedule:
        push_lock._deferred_update()

    mock_reschedule.assert_called_once()
    assert 0 < mock_reschedule.call_args.args[0] <= LOCK_STALE_STATE_DEBOUNCE_DELAY
    assert push_lock._update_task is None


@pytest.mark.asyncio
async def test_deferred_update_backs_off_while_an_operation_holds_the_lock() -> None:
    """A cycle falling due mid-operation backs off rather than queueing.

    The floor can lapse while the motion is still running, and a cycle
    created there would wait on the operation lock and read the lock the
    moment the operation released it, inside the settle window the operation
    is about to open.
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
    the lock (write-success) -- never at issue time -- and exactly once."""
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
async def test_window_admits_door_and_battery_members():
    """Door and battery members of a frame received mid-window still apply;
    only the lock status is filtered out."""
    push_lock = _operational_push_lock()
    push_lock._lock_state = _known_state(LockStatus.UNLOCKED)
    push_lock._operation_window_open = True

    battery = BatteryState(voltage=6.0, percentage=80)
    push_lock._update_any_state([DoorStatus.OPENED, battery, LockStatus.LOCKED])

    assert push_lock.door_status == DoorStatus.OPENED  # door applied
    assert push_lock.battery == battery  # battery applied
    assert push_lock.lock_status == LockStatus.UNLOCKED  # lock filtered


@pytest.mark.asyncio
async def test_early_error_before_write_leaves_no_window_no_unknown():
    """A retryable failure before the command write opens no window and never
    stamps UNKNOWN; the display keeps its prior value."""
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
    assert push_lock.lock_status == LockStatus.LOCKED  # unchanged; no UNKNOWN


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "exc",
    [OperationIncompleteError("no op-response"), UnlatchError("after write")],
)
async def test_nonretryable_after_write_stamps_unknown(exc):
    """The two non-retryable types raised after write-success (a transitional
    is on display with no result coming) close the window and stamp UNKNOWN."""
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
async def test_nonretryable_before_write_success_stamps_no_unknown():
    """A non-retryable failure before write-success leaves the state alone.

    UNKNOWN answers a transitional left on display with no result coming. If
    the write never reached the lock, no transitional was stamped and the
    displayed state is still the truth, so the handler's window-open guard
    takes its other arc: no close, no UNKNOWN.
    """
    push_lock = _operational_push_lock()
    push_lock._lock_state = _known_state(LockStatus.UNLOCKED)

    mock_lock = MagicMock()
    mock_lock.force_lock = AsyncMock(
        side_effect=OperationIncompleteError("before write")
    )

    with (
        patch.object(push_lock, "_ensure_connected", AsyncMock(return_value=mock_lock)),
        pytest.raises(OperationIncompleteError),
    ):
        await push_lock.lock()

    assert push_lock._operation_window_open is False
    assert push_lock._pending_op_state is None
    assert push_lock.lock_status == LockStatus.UNLOCKED


@pytest.mark.asyncio
async def test_unlatch_error_without_a_window_still_stamps_unknown():
    """An UnlatchError with no window open leaves the position unknown.

    force_unlatch converts every failure from the write attempt onward, so a
    write call that raised arrives here with the write-success hook never run
    and the window never opened. The request PDU may still have left the
    radio, in which case the latch fired and the door is open, so the position
    on display is no longer evidence.
    """
    push_lock = _operational_push_lock()
    push_lock._lock_state = _known_state(LockStatus.LOCKED)

    mock_lock = MagicMock()
    mock_lock.force_unlatch = AsyncMock(side_effect=UnlatchError("errored write"))

    with (
        patch.object(push_lock, "_ensure_connected", AsyncMock(return_value=mock_lock)),
        pytest.raises(UnlatchError),
    ):
        await push_lock.unlatch()

    assert push_lock._operation_window_open is False
    assert push_lock._pending_op_state is None
    assert push_lock.lock_status == LockStatus.UNKNOWN


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
    # Never UNKNOWN; LOCKING once (the second stamp is a no-op -- display
    # already reads LOCKING) then LOCKED.
    assert LockStatus.UNKNOWN not in events
    assert events == [LockStatus.LOCKING, LockStatus.LOCKED]
    assert push_lock.lock_status == LockStatus.LOCKED


@pytest.mark.asyncio
async def test_a_failure_before_any_write_leaves_a_reported_transitional_alone():
    """A transitional the lock reported is not claimed by an operation of ours.

    An operation started at the lock has it pushing UNLOCKING and the display
    holding it. Our unlock() then fails before its command write, so no
    transitional of ours was stamped and the displayed one is a real reading
    with the lock's own result still coming. It reads exactly like the one
    this operation would have stamped, which is why the displayed status
    cannot be the test on its own.
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

    assert push_lock._operation_command_written is False
    assert push_lock.lock_status == LockStatus.UNLOCKING
    push_lock._cancel_future_update()
    push_lock._cancel_disconnect_timer()


@pytest.mark.asyncio
async def test_an_operation_settling_after_the_stop_arms_nothing():
    """An operation ending after the watcher stopped leaves no timer behind.

    _cancel does not reach an operation in flight, so its settle still runs
    once the operation fails on its own. The window is closed either way, but
    booking the status poll would arm a cycle holding the lock past the stop
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
    # The unknown position the settle would otherwise have applied.
    assert push_lock.lock_status == LockStatus.LOCKING
    push_lock._cancel_disconnect_timer()


@pytest.mark.asyncio
async def test_a_jam_recorded_before_the_stop_reaches_no_later_operation():
    """The stopped settle discharges the jam record instead of leaving it set.

    That settle returns before it can apply the outcome, so the window closing
    is what clears the record. Were it left set, the next operation on this
    instance would find a jam from the stopped session and end its attempt
    ladder as OperationIncompleteError, displaying JAMMED for a mechanism the
    new command never reached.
    """
    push_lock = _operational_push_lock("aa:bb:cc:dd:ee:4e")
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

    assert push_lock._seen_jam is False

    # Started again, an operation that fails before its write-success keeps its
    # own retryable failure and stamps no position.
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

    assert push_lock.lock_status == LockStatus.LOCKING
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
        return True

    mock_lock.force_lock = force_lock

    with patch.object(
        push_lock, "_ensure_connected", AsyncMock(return_value=mock_lock)
    ):
        await push_lock.lock()

    assert push_lock.lock_status == LockStatus.JAMMED
    assert push_lock._seen_jam is False  # discharged by reaching the display


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
    assert push_lock._seen_jam is False


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
    assert push_lock._seen_jam is False


@pytest.mark.asyncio
async def test_a_new_commands_write_success_clears_the_jam_record() -> None:
    """A command that reaches the lock supersedes a jam still on record.

    Every operation exit applies a jam it recorded, so no exit leaves _seen_jam
    set for the next command to find. The record is seeded by hand here to pin
    the backstop: were one ever to reach a write-success, the command the caller
    issued after the jam is the manual intervention a jam calls for, and its own
    result is what the display carries.
    """
    push_lock = _operational_push_lock()
    push_lock._lock_state = _known_state(LockStatus.JAMMED)
    push_lock._seen_jam = True
    seen_at_write_success: list[bool] = []

    async def force_unlock(write_success_callback):
        write_success_callback()
        seen_at_write_success.append(push_lock._seen_jam)
        return True

    mock_lock = MagicMock()
    mock_lock.force_unlock = force_unlock

    with patch.object(
        push_lock, "_ensure_connected", AsyncMock(return_value=mock_lock)
    ):
        await push_lock.unlock()

    assert seen_at_write_success == [False]
    assert push_lock.lock_status == LockStatus.UNLOCKED


@pytest.mark.asyncio
async def test_queued_operation_emits_no_transitional_until_dequeued() -> None:
    """A second operation queued on the operation lock stamps nothing.

    The flapping-bug regression: with op1 in flight (force_lock blocked on an
    Event), a second lock() is issued. While queued behind op1 on the operation
    lock it must emit NO transitional; op2's LOCKING appears only after op1
    completes. No existing test drives two serialized operations at once.
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
async def test_window_filter_drops_lock_status_admits_door_and_battery() -> None:
    """Literal filter opened via write-success: even mid-window JAMMED is dropped.

    Distinct from the existing window tests (which set the flag directly and do
    not feed JAMMED): the window is opened through the real
    _operation_write_success path, and mid-window JAMMED is dropped with NO
    special-casing (the window check precedes the jam-hold logic), while door
    and battery members of the same frame still pass.
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

    # Foreign jam evidence mid-window is dropped too (no special-casing), and it
    # arms no hold -- the window check comes before the jam-hold logic.
    push_lock._update_any_state([LockStatus.JAMMED])
    assert push_lock.lock_status is LockStatus.LOCKING
    assert push_lock._jammed_hold_deadline == NEVER_TIME

    # Door and battery members of a mid-window frame still pass (filtered
    # per-member by the callers, not by _admit_lock_status).
    push_lock._update_any_state([DoorStatus.OPENED])
    assert push_lock.door_status is DoorStatus.OPENED

    battery = BatteryState(voltage=6.0, percentage=80)
    push_lock._update_any_state([battery])
    assert push_lock.battery == battery


@pytest.mark.asyncio
async def test_early_disconnect_leaves_no_window_and_no_unknown() -> None:
    """A retryable disconnect before any write-success retries to exhaustion.

    Distinct from the existing early-error test (which checks only the final
    display): this pins the retry COUNT -- force_lock is re-sent exactly
    DEFAULT_ATTEMPTS times because the write never reached the lock -- and that
    the display never changed (no transitional, no UNKNOWN stamp).
    """
    push_lock = _operational_push_lock("aa:bb:cc:dd:ee:3c")
    # Seed a settled position: without it the final assertion would match the
    # fixture's own starting value and could not tell "left alone" from a
    # stamped UNKNOWN.
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
    # ...and the display never changed: no transitional, no UNKNOWN stamp.
    assert emissions == []
    assert push_lock.lock_status is LockStatus.LOCKED
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
async def test_cancelled_mid_operation_closes_window_without_unknown() -> None:
    """A bare CancelledError mid-operation closes the window and re-raises.

    CancelledError is a BaseException, so the generic ``except Exception`` never
    sees it; without the dedicated arm the operation window would leak open and
    the transitional freeze on display, with no poll able to replace it. A cancel is not
    evidence the lock did or did not move, so -- unlike the non-retryable
    OperationIncompleteError path -- NO UNKNOWN is stamped; the transitional
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
    register fabricates a plain position, so the status poll this exit books puts
    that on display instead, and no later signal marks the jam.
    """
    push_lock = _operational_push_lock("aa:bb:cc:dd:ee:3a")
    push_lock._lock_state = _known_state(LockStatus.UNLOCKED)

    async def force_lock(write_success_callback: Callable[[], None]) -> None:
        write_success_callback()  # opens the window, stamps LOCKING
        push_lock._update_any_state([LockStatus.JAMMED])
        assert push_lock.lock_status == LockStatus.LOCKING  # dropped mid-window
        raise asyncio.CancelledError

    async def lock_status() -> LockStatus:
        push_lock._update_any_state([LockStatus.LOCKED])
        return LockStatus.LOCKED

    mock_lock = MagicMock()
    mock_lock.force_lock = force_lock
    mock_lock.lock_status = AsyncMock(side_effect=lock_status)
    mock_lock.door_status = AsyncMock(return_value=DoorStatus.CLOSED)

    with (
        patch.object(push_lock, "_ensure_connected", AsyncMock(return_value=mock_lock)),
        pytest.raises(asyncio.CancelledError),
    ):
        await push_lock.lock()

    assert push_lock.lock_status == LockStatus.JAMMED
    assert push_lock._operation_window_open is False
    assert push_lock._seen_jam is False  # discharged by reaching the display
    assert push_lock._jammed_hold_deadline != NEVER_TIME  # the hold is armed

    # The status poll this exit books reads the fabricated position back, and the
    # hold masks it.
    with (
        patch.object(push_lock, "_ensure_connected", AsyncMock(return_value=mock_lock)),
        patch.object(
            push_lock, "_read_auto_lock_setting", AsyncMock(return_value=False)
        ),
        patch.object(push_lock, "_poll_battery", AsyncMock(return_value=False)),
    ):
        await push_lock._update()

    mock_lock.lock_status.assert_awaited_once()
    assert push_lock.lock_status == LockStatus.JAMMED
    push_lock._cancel_future_update()
    push_lock._cancel_disconnect_timer()
    push_lock._jam_hold_wakeup.cancel()


@pytest.mark.asyncio
async def test_operation_outside_the_gate_cannot_open_the_window():
    """A force_* issued straight at the Lock leaves the operation window shut.

    Only _execute_lock_operation hands over the write-success hook, and only
    that method closes the window again. A caller reaching the Lock directly
    therefore gets no hook and cannot leave a window open with no operation in
    flight, which would freeze status admission for the object's life.
    """
    push_lock = _operational_push_lock("aa:bb:cc:dd:ee:34")
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
        progress: OperationProgress | None = None,
        write_success_callback: Callable[[], None] | None = None,
        wait_for_ack: bool = True,
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
    push_lock = _operational_push_lock("aa:bb:cc:dd:ee:35")
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
    """A failed operation must not count as having read the lock status.

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
        final_state = await push_lock._update()

    mock_lock.lock_status.assert_awaited_once()
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

        final_state = await push_lock._update()

    mock_lock.lock_status.assert_awaited_once()
    assert final_state.lock == LockStatus.LOCKED
    push_lock._cancel_disconnect_timer()


@pytest.mark.asyncio
async def test_exhausted_retries_after_write_success_stamp_unknown() -> None:
    """Every attempt's write reaches the lock, and every attempt then fails.

    Through the retries the transitional stays on display because the next
    attempt re-stamps it. Once the attempts run out there is no next attempt,
    so the operation settles the display on UNKNOWN rather than leaving a
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
    # The attempts ran out with no result, so this exit books the poll itself.
    assert push_lock._force_lock_status_poll is True
    push_lock._cancel_disconnect_timer()


@pytest.mark.asyncio
async def test_unlatch_stamps_unlatching_then_unlocked():
    """The new public unlatch() maps to force_unlatch, stamping UNLATCHING at
    write-success and UNLOCKED on success: the op-response arrives when the
    latch has returned from its open dwell, so UNLATCHED is never the
    completed state."""
    push_lock = _operational_push_lock()
    order: list[str | LockStatus] = []

    def cb(lock_state, lock_info, connection_info):
        order.append(lock_state.lock)

    push_lock.register_callback(cb)

    mock_lock = MagicMock()

    async def force_unlatch(write_success_callback):
        order.append("write_success")
        write_success_callback()

    mock_lock.force_unlatch = force_unlatch

    with patch.object(
        push_lock, "_ensure_connected", AsyncMock(return_value=mock_lock)
    ):
        await push_lock.unlatch()

    assert order == ["write_success", LockStatus.UNLATCHING, LockStatus.UNLOCKED]
    assert push_lock.lock_status == LockStatus.UNLOCKED
    push_lock._cancel_future_update()
    push_lock._cancel_disconnect_timer()


@pytest.mark.asyncio
async def test_no_update_cycle_is_armed_inside_an_operation() -> None:
    """Nothing the operation applies arms a cycle while the operation runs.

    A cycle armed here waits on the operation lock and runs the instant the
    operation ends, inside the settle window the stale-state guard exists to
    protect. The transitional and the completed status are applied by the
    operation itself, so they arm nothing; the status poll is scheduled once the
    operation is over.
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
    anything awaits, so a cycle booked before the command cannot fire while
    the operation is connecting and read the lock mid-command. The probe runs
    inside _ensure_connected, the operation's first await, so only the entry
    cancel can have cleared the booking by then.
    """
    push_lock = _operational_push_lock("aa:bb:cc:dd:ee:48")
    push_lock._lock_state = _known_state(LockStatus.UNLOCKED)
    push_lock._schedule_future_update(30.0)
    assert push_lock._cancel_deferred_update is not None

    mock_lock = MagicMock()
    mock_lock.force_lock = AsyncMock()
    pending_at_connect: list[bool] = []

    async def connected() -> MagicMock:
        pending_at_connect.append(push_lock._cancel_deferred_update is not None)
        return mock_lock

    with (
        patch.object(push_lock, "_ensure_connected", AsyncMock(side_effect=connected)),
        patch.object(push_lock, "_schedule_future_update"),
    ):
        await push_lock.lock()

    # The booking was already gone when the operation connected.
    assert pending_at_connect == [False]
    push_lock._cancel_disconnect_timer()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("error", "jam", "delay"),
    [
        (None, False, KEEP_ALIVE_TIME),
        (OperationFailedError("reported failure", 0x1F), False, KEEP_ALIVE_TIME),
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
    ids=["success", "reported-failure", "no-result", "jam-ends-the-ladder"],
)
async def test_every_operation_exit_books_the_status_poll(
    error: Exception | None, jam: bool, delay: float
) -> None:
    """Every way an operation can end schedules the follow-up status poll.

    Success, a reported failure, a result that never came, and a jam that ends
    the attempt ladder all leave through the same settle, so all four book the
    poll. The delay follows the status each one leaves on display: a position
    is polled at the keep-alive interval, the cadence an always-connected lock
    polls at anyway, and the UNKNOWN of a result that never came is polled at
    the settle debounce, because only a read can replace it.
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
async def test_an_operation_that_never_authenticated_books_the_status_poll() -> None:
    """An operation whose attempts all fail authentication books the status poll.

    The retry decorator returns rather than raising when every attempt raises
    AuthError below the re-authentication threshold, so this operation ends
    without an exception and without an outcome. No command reached the lock,
    so the display is untouched, and the status poll is booked all the same: the
    settle is the one exit, and it does not ask how the operation ended.
    """
    push_lock = _operational_push_lock("aa:bb:cc:dd:ee:4a")
    push_lock._lock_state = _known_state(LockStatus.UNLOCKED)
    calls = 0

    async def force_lock(write_success_callback: Callable[[], None]) -> None:
        nonlocal calls
        calls += 1
        raise AuthError("the key is wrong")

    mock_lock = MagicMock()
    mock_lock.force_lock = force_lock

    with (
        patch.object(push_lock, "_ensure_connected", AsyncMock(return_value=mock_lock)),
        patch.object(push_lock, "_schedule_future_update") as scheduled,
        patch("yalexs_ble.push.asyncio.sleep", AsyncMock()),
    ):
        await push_lock.lock()

    assert calls == DEFAULT_ATTEMPTS
    assert push_lock.lock_status == LockStatus.UNLOCKED
    assert push_lock._force_lock_status_poll is True
    scheduled.assert_called_once_with(KEEP_ALIVE_TIME)
    _AUTH_FAILURE_HISTORY.auth_success(push_lock.address)
    push_lock._cancel_disconnect_timer()


@pytest.mark.asyncio
async def test_the_status_poll_keeps_a_sooner_pending_update() -> None:
    """The status poll inherits the debounce, so a sooner update is kept.

    _settle_operation books the status poll through the deferred-update
    debounce. An update already due sooner than the keep-alive interval keeps
    its slot; an undebounced booking would displace it a full interval out.
    Keeping it costs nothing, because the floor the settle stamps holds the
    read itself back: the sooner cycle falls due, finds the floor, and re-arms
    for the remainder instead of reading the lock mid-settle.
    """
    push_lock = _operational_push_lock("aa:bb:cc:dd:ee:49")
    push_lock._schedule_future_update(1.0)
    handle = push_lock._cancel_deferred_update
    assert handle is not None

    push_lock._settle_operation()

    assert push_lock._force_lock_status_poll is True
    # The sooner booking is untouched: same handle, still due within a second.
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
async def test_a_collapsed_booking_still_waits_for_the_floor() -> None:
    """A booking the debounce shortens cannot read the lock mid-settle.

    An update already due inside the coalescing interval makes the debounce
    rewrite any request to that interval, so the poll the settle asks for
    is not the delay the cycle gets. The floor is what holds the read: the
    collapsed booking is armed for the remainder of the settle window instead
    of firing 25 ms after the operation, when the lock still reports the
    pre-operation position.
    """
    push_lock = _operational_push_lock("aa:bb:cc:dd:ee:4b")
    push_lock._lock_state = _known_state(LockStatus.UNLOCKED)
    push_lock._schedule_future_update(0.01)

    push_lock._settle_operation()

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
    is armed at the settle debounce rather than the resync delay, because the
    written command is still driving the motor and an immediate read would
    return the pre-operation position. The settle stamps the floor at the same
    moment, so a cancel is held out for the same window a completion is,
    however the booking itself is debounced.
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
async def test_the_status_poll_reads_past_the_seen_mark() -> None:
    """The booked status poll asks the lock even with the status marked seen.

    A status the operation applied, or one the display is holding, keeps its
    mark in the ordinary way; a cycle that honoured it would ask the lock
    nothing and the display would keep whatever it was holding.
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
        state = await push_lock._update()

    mock_lock.lock_status.assert_awaited_once()
    assert state.lock == LockStatus.LOCKED
    # One-shot: the next cycle honours the seen mark again.
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
    push_lock = _operational_push_lock("aa:bb:cc:dd:ee:4a")
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

    # Only the flag can be asking here (the status is marked seen), so each
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

    Recording one in the seen set would suppress the follow-up lock_status()
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
    push_lock = _operational_push_lock("aa:bb:cc:dd:ee:4b")
    push_lock._lock_state = _known_state(LockStatus.UNKNOWN)

    push_lock._update_any_state([position], arm_resync=False)

    assert push_lock.lock_status == position
    assert LockStatus in push_lock._seen_this_session


# ---------------------------------------------------------------------------
# Hold JAMMED on display
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_repeated_jammed_extends_hold_deadline():
    """Every admitted JAMMED event -- even one identical to the displayed
    value -- pushes the hold deadline out again (the helper runs before the
    equality check)."""
    push_lock = _operational_push_lock("aa:bb:cc:dd:ee:28")
    push_lock._lock_state = _known_state(LockStatus.LOCKED)

    with patch("yalexs_ble.push.time.monotonic", return_value=1000.0):
        push_lock._update_any_state([LockStatus.JAMMED])
        assert push_lock.lock_status == LockStatus.JAMMED
        assert push_lock._jammed_hold_deadline == 1000.0 + JAMMED_HOLD_TIME

    # A second JAMMED while already displaying JAMMED still re-arms the hold.
    with patch("yalexs_ble.push.time.monotonic", return_value=1010.0):
        push_lock._update_any_state([LockStatus.JAMMED])
        assert push_lock._jammed_hold_deadline == 1010.0 + JAMMED_HOLD_TIME


@pytest.mark.asyncio
async def test_an_admitted_jam_arms_the_wakeup_and_the_holds_end_polls_the_lock():
    """An admitted jam arms the hold's ending wakeup; the fire at the hold's
    end discards the seen mark and schedules the near-term follow-up poll,
    and that poll reads the lock status on a lock that is not always
    connected.

    A jam is a real reading, so it keeps its seen mark; without the wakeup
    nothing would ask the lock again and the display would hold JAMMED until
    the next operation or a restart.
    """
    push_lock = _operational_push_lock("aa:bb:cc:dd:ee:48")
    push_lock._lock_state = _known_state(LockStatus.LOCKED)

    with (
        patch("yalexs_ble.push.time.monotonic", return_value=1000.0),
        patch.object(push_lock, "_schedule_future_update") as scheduled,
    ):
        push_lock._update_any_state([LockStatus.JAMMED])

    assert push_lock._jammed_hold_deadline == 1000.0 + JAMMED_HOLD_TIME
    assert push_lock._jam_hold_wakeup._timer is not None
    # The jam keeps its seen mark until the hold ends.
    assert LockStatus in push_lock._seen_this_session
    # A jam arriving from the lock is an ordinary status change, so it keeps
    # its resync too; only the states an operation applies itself do not.
    scheduled.assert_any_call(RESYNC_DELAY)

    # The wakeup fires past the deadline: the seen mark is discarded and the
    # follow-up poll is booked near-term.
    with (
        patch("yalexs_ble.push.time.monotonic", return_value=1031.0),
        patch.object(push_lock, "_schedule_future_update_with_debounce") as debounced,
    ):
        push_lock._jam_hold_wakeup._on_timer()

    assert LockStatus not in push_lock._seen_this_session
    debounced.assert_called_once_with(0)

    # The poll the fire booked reads the lock status, because the seen mark
    # is gone, and the answer is admitted, because the hold has lapsed.
    push_lock._seen_this_session.update({BatteryState, DoorStatus, AutoLockState})
    mock_lock = MagicMock()

    async def lock_status() -> LockStatus:
        # The read answers through the state path, as the real one does.
        push_lock._update_any_state([LockStatus.LOCKED])
        return LockStatus.LOCKED

    mock_lock.lock_status = AsyncMock(side_effect=lock_status)

    with (
        patch.object(push_lock, "_ensure_connected", AsyncMock(return_value=mock_lock)),
        patch("yalexs_ble.push.time.monotonic", return_value=1031.1),
    ):
        final_state = await push_lock._update()

    mock_lock.lock_status.assert_awaited_once()
    assert final_state.lock is LockStatus.LOCKED


@pytest.mark.asyncio
async def test_a_second_jam_extends_the_hold_and_the_wakeup_polls_at_the_new_end():
    """A wakeup firing at a deadline that has moved out re-arms for the
    remainder instead of polling; the poll happens at the extended end.

    The deadline field is the source of truth and the timer is only a wakeup,
    so the fire re-checks the field before acting.
    """
    push_lock = _operational_push_lock("aa:bb:cc:dd:ee:49")
    push_lock._lock_state = _known_state(LockStatus.LOCKED)

    with patch("yalexs_ble.push.time.monotonic", return_value=1000.0):
        push_lock._update_any_state([LockStatus.JAMMED])
    # A second jam pushes the deadline out to 1040.
    with patch("yalexs_ble.push.time.monotonic", return_value=1010.0):
        push_lock._update_any_state([LockStatus.JAMMED])
    assert push_lock._jammed_hold_deadline == 1010.0 + JAMMED_HOLD_TIME

    # A fire at the FIRST deadline finds the extended one still ahead: it
    # re-arms for the remainder and does not poll.
    with (
        patch("yalexs_ble.push.time.monotonic", return_value=1030.0),
        patch.object(push_lock, "_schedule_future_update_with_debounce") as debounced,
    ):
        push_lock._jam_hold_wakeup._on_timer()

    debounced.assert_not_called()
    assert LockStatus in push_lock._seen_this_session
    assert push_lock._jam_hold_wakeup._timer is not None

    # Past the extended end the fire polls.
    with (
        patch("yalexs_ble.push.time.monotonic", return_value=1041.0),
        patch.object(push_lock, "_schedule_future_update_with_debounce") as debounced,
    ):
        push_lock._jam_hold_wakeup._on_timer()

    debounced.assert_called_once_with(0)
    assert LockStatus not in push_lock._seen_this_session


@pytest.mark.asyncio
async def test_a_wakeup_fire_during_an_operation_re_arms_and_creates_no_update():
    """A wakeup firing while an operation runs books no cycle and re-arms.

    A cycle armed here would wait on the operation lock and run the instant
    the operation ends, inside the settle window, so the fire backs off
    briefly instead; the retry finds the hold released or still ended.
    """
    push_lock = _operational_push_lock("aa:bb:cc:dd:ee:4a")
    push_lock._lock_state = _known_state(LockStatus.LOCKED)

    with patch("yalexs_ble.push.time.monotonic", return_value=1000.0):
        push_lock._update_any_state([LockStatus.JAMMED])
    push_lock._cancel_future_update()

    await push_lock._operation_lock.acquire()
    try:
        with (
            patch("yalexs_ble.push.time.monotonic", return_value=1031.0),
            patch.object(
                push_lock, "_schedule_future_update_with_debounce"
            ) as debounced,
        ):
            push_lock._jam_hold_wakeup._on_timer()
            # Re-armed for the brief retry, not for a new hold. Patching
            # time.monotonic moves loop.time() with it, so when() is read
            # against the same patched base.
            timer = push_lock._jam_hold_wakeup._timer
            assert timer is not None
            assert timer.when() - push_lock.loop.time() == pytest.approx(
                DEADLINE_WAKEUP_RETRY_DELAY, abs=0.1
            )
    finally:
        push_lock._operation_lock.release()

    debounced.assert_not_called()
    assert push_lock._update_task is None
    # The seen mark stays: the follow-up poll is still owed.
    assert LockStatus in push_lock._seen_this_session
    push_lock._jam_hold_wakeup.cancel()


@pytest.mark.asyncio
async def test_write_success_releases_the_hold_and_cancels_the_wakeup():
    """A new operation's write-success releases the hold and cancels its
    wakeup, so no follow-up poll fires after the release."""
    push_lock = _operational_push_lock("aa:bb:cc:dd:ee:4b")
    push_lock._lock_state = _known_state(LockStatus.LOCKED)

    with patch("yalexs_ble.push.time.monotonic", return_value=1000.0):
        push_lock._update_any_state([LockStatus.JAMMED])
    armed_timer = push_lock._jam_hold_wakeup._timer
    assert armed_timer is not None

    push_lock._pending_op_state = LockStatus.LOCKING
    push_lock._operation_write_success()

    assert push_lock._jammed_hold_deadline == NEVER_TIME
    assert push_lock._jam_hold_wakeup._timer is None
    assert armed_timer.cancelled()


@pytest.mark.asyncio
async def test_stop_releases_the_jam_hold():
    """Stopping the lock releases the hold and cancels its ending wakeup.

    The wakeup cannot fire after the stop and book a cycle on a lock nothing is
    watching, and no deadline is left behind to mask the lock's status on a
    watcher started again, where nothing would be armed to end the hold or to
    ask the lock what the mechanism is doing now.
    """
    push_lock = _operational_push_lock("aa:bb:cc:dd:ee:4c")
    push_lock._lock_state = _known_state(LockStatus.LOCKED)

    push_lock._update_any_state([LockStatus.JAMMED])
    armed_timer = push_lock._jam_hold_wakeup._timer
    assert armed_timer is not None

    push_lock._cancel()

    assert push_lock._jammed_hold_deadline == NEVER_TIME
    assert push_lock._jam_hold_wakeup._timer is None
    assert armed_timer.cancelled()
    await asyncio.gather(*push_lock._background_tasks)


_UNSET = object()

# Re-set at the start of every operation by _run_lock_operation, so a stale value
# between operations is read by nobody. Every other field the initialisers install
# carries across operations, and one left behind is read by the next one.
_RESET_AT_OPERATION_START = frozenset(
    {"_operation_outcome", "_operation_command_written"}
)


def _carrier_defaults() -> dict[str, Any]:
    """The fields the two initialisers install, and the values they install.

    Discovered by poisoning a throwaway lock and re-running both initialisers
    rather than listed here, so a carrier added to either one joins the test
    below on its own. That is the point of it: the field that went unnoticed
    was one nobody decided the teardown rule for when it was added.
    """
    probe = _operational_push_lock("aa:bb:cc:dd:ee:5f")
    for name in list(vars(probe)):
        setattr(probe, name, _UNSET)
    probe._init_operation_state()
    probe._init_jam_state()
    return {n: v for n, v in vars(probe).items() if v is not _UNSET}


@pytest.mark.asyncio
async def test_a_teardown_returns_every_cross_operation_carrier_to_its_initial_value():
    """Nothing an operation or the watcher recorded survives into the next operation.

    Both halves below were real defects. A jam recorded while the watcher was
    stopped survived the settle's early return and ended the next operation's
    attempt ladder as OperationIncompleteError; a hold deadline the stop left
    behind masked the lock's status on a watcher started again, with no wakeup
    left to end the hold or to ask the lock. Each was a single exit that did
    not discharge what it should, which is why this asks the object as a whole
    rather than walking the exits.
    """
    defaults = _carrier_defaults()
    carriers = {
        name: value
        for name, value in defaults.items()
        # The wakeup is an object rather than a value, so a fresh one never
        # compares equal; it is asserted through its armed timer instead.
        if name not in _RESET_AT_OPERATION_START and name != "_jam_hold_wakeup"
    }
    assert {"_seen_jam", "_jammed_hold_deadline"} <= set(carriers)

    # An operation that records a jam and then ends after the watcher stopped.
    push_lock = _operational_push_lock("aa:bb:cc:dd:ee:5e")
    push_lock._lock_state = _known_state(LockStatus.UNLOCKED)

    async def _jam_then_stop_then_fail(write_success_callback):
        write_success_callback()
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

    for name, value in carriers.items():
        assert getattr(push_lock, name) == value, f"{name} survived the settle"
    push_lock._cancel_disconnect_timer()

    # A stop taken with a jam hold armed.
    stopped = _operational_push_lock("aa:bb:cc:dd:ee:5d")
    stopped._lock_state = _known_state(LockStatus.LOCKED)
    stopped._update_any_state([LockStatus.JAMMED])
    armed_timer = stopped._jam_hold_wakeup._timer
    assert armed_timer is not None

    stopped._cancel()

    for name, value in carriers.items():
        assert getattr(stopped, name) == value, f"{name} survived the stop"
    assert stopped._jam_hold_wakeup._timer is None
    assert armed_timer.cancelled()
    await asyncio.gather(*stopped._background_tasks)


@pytest.mark.asyncio
async def test_the_hold_does_not_keep_the_link_up_on_a_demand_lock():
    """During the hold nothing sits in the deferred-update slot, so the idle
    disconnect proceeds instead of being converted into a poll.

    This pins the absence of the measured defect shape: with a status poll
    booked 31 s out at admission, _disconnect_with_timer found a pending
    update at every idle expiry and converted it into an immediate poll, so
    the link stayed up and polled for the hold's whole duration.
    """
    push_lock = _operational_push_lock("aa:bb:cc:dd:ee:4d")
    push_lock._lock_state = _known_state(LockStatus.LOCKED)

    with patch("yalexs_ble.push.time.monotonic", return_value=1000.0):
        push_lock._update_any_state([LockStatus.JAMMED])
    # Drop the near-term resync the status change armed; it runs well before
    # the idle expiry and is not the subject here.
    push_lock._cancel_future_update()

    # A reading masked mid-hold books nothing into the deferred-update slot.
    with patch("yalexs_ble.push.time.monotonic", return_value=1010.0):
        push_lock._update_any_state([LockStatus.LOCKED])
    assert push_lock.lock_status is LockStatus.JAMMED
    assert push_lock._cancel_deferred_update is None

    # With the slot empty the idle disconnect disconnects. The clock stays
    # patched mid-hold so the hold's own wakeup, which is not due yet, does
    # not fire during the awaits.
    with (
        patch("yalexs_ble.push.time.monotonic", return_value=1010.0),
        patch.object(
            push_lock, "_execute_timed_disconnect", AsyncMock()
        ) as timed_disconnect,
        patch.object(push_lock, "_deferred_update") as deferred,
    ):
        push_lock._disconnect_with_timer(DISCONNECT_DELAY)
        await asyncio.gather(*push_lock._background_tasks)

    deferred.assert_not_called()
    timed_disconnect.assert_awaited_once_with(DISCONNECT_DELAY)
    push_lock._jam_hold_wakeup.cancel()


@pytest.mark.asyncio
async def test_a_polled_unknown_mid_hold_does_not_extend_the_jam_hold():
    """A poll answering UNKNOWN mid-hold keeps JAMMED on display without
    moving the hold deadline or its wakeup.

    Only a status actually read from the lock this cycle is jam evidence; the
    cached JAMMED substituted for a polled UNKNOWN is the display's own value
    coming back around, and it must not extend its own hold.
    """
    push_lock = _operational_push_lock("aa:bb:cc:dd:ee:4e")
    push_lock._lock_state = _known_state(LockStatus.LOCKED)

    with patch("yalexs_ble.push.time.monotonic", return_value=1000.0):
        push_lock._update_any_state([LockStatus.JAMMED])
    push_lock._cancel_future_update()
    timer_before = push_lock._jam_hold_wakeup._timer

    # A reconnect cleared the session, so the cycle polls the status again.
    push_lock._seen_this_session.clear()
    push_lock._seen_this_session.update({BatteryState, DoorStatus, AutoLockState})
    mock_lock = MagicMock()

    async def lock_status() -> LockStatus:
        push_lock._update_any_state([LockStatus.UNKNOWN])
        return LockStatus.UNKNOWN

    mock_lock.lock_status = AsyncMock(side_effect=lock_status)

    with (
        patch.object(push_lock, "_ensure_connected", AsyncMock(return_value=mock_lock)),
        patch("yalexs_ble.push.time.monotonic", return_value=1015.0),
    ):
        final_state = await push_lock._update()

    mock_lock.lock_status.assert_awaited_once()
    assert final_state.lock is LockStatus.JAMMED
    # The deadline is still the admission's, not 1015.0 + JAMMED_HOLD_TIME.
    assert push_lock._jammed_hold_deadline == 1000.0 + JAMMED_HOLD_TIME
    # The wakeup was not re-armed either: the handle is the admission's.
    assert push_lock._jam_hold_wakeup._timer is timer_before


@pytest.mark.asyncio
async def test_extended_hold_pins_a_settle_past_the_first_deadline() -> None:
    """A repeated jam extends the hold, pinning a LOCKED settle past the FIRST end.

    Distinct from the existing extend test (which only checks the deadline
    value moved): this proves the extension actually PINS -- a LOCKED settle
    that falls past the original deadline but inside the extended one stays
    JAMMED, and only clears past the extended deadline.
    """
    push_lock = _operational_push_lock("aa:bb:cc:dd:ee:3e")

    with patch("yalexs_ble.push.time.monotonic", return_value=1000.0):
        push_lock._update_any_state([LockStatus.JAMMED])
        assert push_lock._jammed_hold_deadline == 1000.0 + JAMMED_HOLD_TIME

    # Identical value -> no display change, but the deadline extends to 1050.
    with patch("yalexs_ble.push.time.monotonic", return_value=1020.0):
        push_lock._update_any_state([LockStatus.JAMMED])
        assert push_lock.lock_status is LockStatus.JAMMED
        assert push_lock._jammed_hold_deadline == 1020.0 + JAMMED_HOLD_TIME

    # Past the FIRST deadline (1030), inside the extended one.
    with patch("yalexs_ble.push.time.monotonic", return_value=1040.0):
        push_lock._update_any_state([LockStatus.LOCKED])
        assert push_lock.lock_status is LockStatus.JAMMED  # pinned by the extension

    with patch("yalexs_ble.push.time.monotonic", return_value=1051.0):
        push_lock._update_any_state([LockStatus.LOCKED])
        assert push_lock.lock_status is LockStatus.LOCKED


@pytest.mark.asyncio
async def test_admit_lock_status_poll_path_pins_and_extends() -> None:
    """The poll path (_admit_lock_status directly) pins JAMMED and extends on jam.

    Distinct entry path from the existing poll-site test (which drives the full
    _update merge): this unit-tests the shared _admit_lock_status helper, both
    the pin (a LOCKED poll answer inside the hold reads JAMMED) and the extend
    (a polled JAMMED arms a fresh deadline).
    """
    push_lock = _operational_push_lock("aa:bb:cc:dd:ee:3f")
    # A hold is armed (e.g. by a settled 0x07 push or an op-response jam).
    push_lock._jammed_hold_deadline = 1000.0 + JAMMED_HOLD_TIME

    with patch("yalexs_ble.push.time.monotonic", return_value=1000.0):
        # A LOCKED poll answer inside the hold is pinned to JAMMED.
        assert (
            push_lock._admit_lock_status(LockStatus.LOCKED, LockStatus.JAMMED)
            is LockStatus.JAMMED
        )

    d0 = push_lock._jammed_hold_deadline
    with patch("yalexs_ble.push.time.monotonic", return_value=1005.0):
        # An incoming JAMMED from a poll answer arms/extends the deadline.
        assert (
            push_lock._admit_lock_status(LockStatus.JAMMED, LockStatus.LOCKED)
            is LockStatus.JAMMED
        )
    assert push_lock._jammed_hold_deadline > d0
    assert push_lock._jammed_hold_deadline == 1005.0 + JAMMED_HOLD_TIME


@pytest.mark.asyncio
async def test_hold_pins_locked_poll_then_admits_after_expiry():
    """Within the hold a LOCKED value is pinned to JAMMED; past the deadline it
    is admitted."""
    push_lock = _operational_push_lock("aa:bb:cc:dd:ee:29")
    push_lock._lock_state = _known_state(LockStatus.JAMMED)
    push_lock._jammed_hold_deadline = 1030.0

    with patch("yalexs_ble.push.time.monotonic", return_value=1010.0):
        push_lock._update_any_state([LockStatus.LOCKED])
        assert push_lock.lock_status == LockStatus.JAMMED  # pinned within hold

    with patch("yalexs_ble.push.time.monotonic", return_value=1040.0):
        push_lock._update_any_state([LockStatus.LOCKED])
        assert push_lock.lock_status == LockStatus.LOCKED  # admitted after expiry


@pytest.mark.asyncio
async def test_write_success_releases_hold_and_shows_transitional():
    """A new operation's write-success releases the JAMMED hold FIRST, so its
    transitional passes the filter and is displayed (order: release hold ->
    stamp pending -> open window)."""
    push_lock = _operational_push_lock("aa:bb:cc:dd:ee:2a")
    push_lock._lock_state = _known_state(LockStatus.JAMMED)
    push_lock._jammed_hold_deadline = time.monotonic() + JAMMED_HOLD_TIME
    push_lock._pending_op_state = LockStatus.LOCKING

    push_lock._operation_write_success()

    assert push_lock._jammed_hold_deadline == NEVER_TIME  # hold released
    assert push_lock.lock_status == LockStatus.LOCKING  # transitional shown
    assert push_lock._operation_window_open is True


@pytest.mark.asyncio
async def test_operation_failure_arms_hold():
    """The operation-failure JAMMED also arms the
    display hold deadline before the failure propagates."""
    push_lock = _operational_push_lock("aa:bb:cc:dd:ee:2e")

    mock_lock = MagicMock()

    async def force_lock(write_success_callback):
        write_success_callback()
        # op-response byte[15] != 0
        raise OperationFailedError("force_lock reported failure", 0x1F)

    mock_lock.force_lock = force_lock

    before = time.monotonic()
    with (
        patch.object(push_lock, "_ensure_connected", AsyncMock(return_value=mock_lock)),
        pytest.raises(OperationFailedError),
    ):
        await push_lock.lock()

    assert push_lock.lock_status == LockStatus.JAMMED
    assert push_lock._jammed_hold_deadline >= before + JAMMED_HOLD_TIME


@pytest.mark.asyncio
async def test_settled_jammed_frame_arms_hold():
    """A settled 0x07 push (parsed to JAMMED) with no operation in flight arms
    the display hold and shows JAMMED.

    The frame this stands for is a settled GETSTATUS push carrying byte[8] =
    0x07 STATICPOSITION, which the parser reads as JAMMED; the status is
    handed straight to the callback here, so no frame is parsed.
    """
    push_lock = _operational_push_lock("aa:bb:cc:dd:ee:2f")
    push_lock._lock_state = _known_state(LockStatus.LOCKED)

    before = time.monotonic()
    push_lock._state_callback([LockStatus.JAMMED])

    assert push_lock.lock_status == LockStatus.JAMMED
    assert push_lock._jammed_hold_deadline >= before + JAMMED_HOLD_TIME


@pytest.mark.asyncio
async def test_battery_only_cycle_does_not_move_hold_deadline():
    """CONFORMANCE FIX: a cycle that polls battery but not lock status must not
    touch the JAMMED hold deadline -- a carried-forward cached JAMMED is not
    fresh evidence and must not re-arm the hold."""
    push_lock = _operational_push_lock("aa:bb:cc:dd:ee:30")
    push_lock._lock_state = _known_state(LockStatus.JAMMED)
    deadline = time.monotonic() + JAMMED_HOLD_TIME
    push_lock._jammed_hold_deadline = deadline
    # Lock status already seen (skipped; not always_connected); battery due.
    push_lock._seen_this_session.update({LockStatus, DoorStatus, AutoLockState})

    mock_lock = MagicMock()
    mock_lock.battery = AsyncMock(return_value=BatteryState(voltage=6.0, percentage=80))

    with patch.object(
        push_lock, "_ensure_connected", AsyncMock(return_value=mock_lock)
    ):
        final_state = await push_lock._update()

    mock_lock.battery.assert_awaited_once()
    mock_lock.lock_status.assert_not_called()
    # Deadline untouched; display still JAMMED (carried forward, not re-armed).
    assert push_lock._jammed_hold_deadline == deadline
    assert final_state.lock == LockStatus.JAMMED


# ---------------------------------------------------------------------------
# The stale-state floor and the moments that set it
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_ack_and_op_response_callbacks_hold_the_next_read():
    """Each observed moment holds the next read off the lock for its own delay.

    The acknowledgement holds for the full settle debounce and the op-response
    for the shorter one, and the floor only ever moves forward: an op-response
    arriving inside the window the acknowledgement claimed leaves that window
    standing.
    """
    push_lock = _operational_push_lock("aa:bb:cc:dd:ee:32")
    assert push_lock._earliest_update_time == NEVER_TIME
    # A fake clock ahead of NEVER_TIME, which is a day behind the real one.
    now = NEVER_TIME + 90000.0

    with patch("yalexs_ble.push.time.monotonic", return_value=now):
        push_lock._ack_callback()
    assert push_lock._earliest_update_time == now + LOCK_STALE_STATE_DEBOUNCE_DELAY

    with patch("yalexs_ble.push.time.monotonic", return_value=now + 1.0):
        push_lock._op_response_callback()
    assert push_lock._earliest_update_time == now + LOCK_STALE_STATE_DEBOUNCE_DELAY

    with patch("yalexs_ble.push.time.monotonic", return_value=now + 10.0):
        push_lock._op_response_callback()
    assert (
        push_lock._earliest_update_time == now + 10.0 + POST_OP_RESPONSE_DEBOUNCE_DELAY
    )


@pytest.mark.asyncio
async def test_get_lock_instance_wires_only_the_stream_observers():
    """_get_lock_instance stores the two stream observers and nothing else.

    The acknowledgement and op-response hooks fire on any matching frame,
    including operations we did not issue, so they belong to the connection.
    The write-success hook belongs to a single operation and is handed over at
    the call site instead, so a Lock built here carries no way to open the
    operation window on its own.
    """
    push_lock = _operational_push_lock("aa:bb:cc:dd:ee:33")
    push_lock._ble_device = MagicMock()

    lock = push_lock._get_lock_instance()

    assert lock._ack_callback == push_lock._ack_callback
    assert lock._op_response_callback == push_lock._op_response_callback
    # Lock holds no write-success hook of its own: it is a parameter of the
    # operation call, so there is no attribute here to assert against.


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("stamps", "expected_delay"),
    [
        # The acknowledgement's window outlasts an op-response inside it.
        ((("ack", -4.0), ("op_response", -3.0)), 2.1),
        # An op-response more than 2 s later carries the floor past it.
        ((("ack", -5.0), ("op_response", -1.0)), 3.1),
        # A command issued after a stale acknowledgement holds from itself.
        ((("ack", -5.0), ("command", -2.0)), 4.1),
    ],
    ids=["acknowledgement-holds", "late-op-response-holds", "command-holds"],
)
async def test_deferred_update_reschedules_to_the_standing_floor(
    stamps, expected_delay
):
    """Within the hold the update reschedules by exactly the floor minus now."""
    push_lock = _operational_push_lock("aa:bb:cc:dd:ee:34")
    # A fake clock ahead of NEVER_TIME, which is a day behind the real one.
    now = NEVER_TIME + 90000.0
    delays = {
        "ack": LOCK_STALE_STATE_DEBOUNCE_DELAY,
        "op_response": POST_OP_RESPONSE_DEBOUNCE_DELAY,
        "command": LOCK_STALE_STATE_DEBOUNCE_DELAY,
    }
    for moment, offset in stamps:
        with patch("yalexs_ble.push.time.monotonic", return_value=now + offset):
            push_lock._hold_update(delays[moment])

    with (
        patch("yalexs_ble.push.time.monotonic", return_value=now),
        patch.object(push_lock, "_schedule_future_update") as mock_reschedule,
    ):
        push_lock._deferred_update()

    mock_reschedule.assert_called_once()
    assert mock_reschedule.call_args.args[0] == pytest.approx(expected_delay)


@pytest.mark.asyncio
async def test_external_op_response_alone_defers_poll():
    """An op-response we issued no command for holds the next poll on its own."""
    push_lock = _operational_push_lock("aa:bb:cc:dd:ee:35")
    # A fake "now" comfortably after NEVER_TIME, so an unstamped floor stays
    # firmly in the past.
    now = NEVER_TIME + 90000.0

    with patch("yalexs_ble.push.time.monotonic", return_value=now):
        push_lock._op_response_callback()
        with patch.object(push_lock, "_schedule_future_update") as mock_reschedule:
            push_lock._deferred_update()

    mock_reschedule.assert_called_once()
    assert mock_reschedule.call_args.args[0] == pytest.approx(
        POST_OP_RESPONSE_DEBOUNCE_DELAY
    )


@pytest.mark.asyncio
async def test_no_hold_proceeds_immediately():
    """With the floor still at NEVER_TIME, about a day in the past, the update
    starts instead of deferring."""
    push_lock = _operational_push_lock("aa:bb:cc:dd:ee:36")
    assert push_lock._earliest_update_time == NEVER_TIME

    push_lock._execute_deferred_update = AsyncMock()  # type: ignore[method-assign]

    with patch.object(push_lock, "_schedule_future_update") as mock_reschedule:
        push_lock._deferred_update()
        assert push_lock._update_task is not None
        await push_lock._update_task

    mock_reschedule.assert_not_called()


@pytest.mark.asyncio
async def test_a_cycle_due_mid_operation_keeps_the_operations_outcome() -> None:
    """A motion that outlasts the stale-state deadline keeps its own outcome.

    A cycle booked while an operation runs waits for the floor, 6.1 s past the
    acknowledgement. An unlatch answers about 10 s after the command, so that
    moment falls due with the latch still out. Queueing the cycle there
    would poll the instant the op-response landed, reading the position the
    lock still reports from before the operation and putting it on display in
    place of the outcome the operation just applied.
    """
    push_lock = _operational_push_lock("aa:bb:cc:dd:ee:39")
    push_lock._lock_state = _known_state(LockStatus.LOCKED)
    gate = asyncio.Event()

    async def force_unlatch(write_success_callback: Callable[[], None]) -> None:
        write_success_callback()  # opens the window, stamps UNLATCHING
        # The command's own hold has already run out, mid-motion.
        push_lock._earliest_update_time = time.monotonic() - 1.0
        await gate.wait()  # the motor is still running
        push_lock._op_response_callback()

    mock_lock = _answering_lock(push_lock)
    mock_lock.force_unlatch = force_unlatch

    with patch.object(
        push_lock, "_ensure_connected", AsyncMock(return_value=mock_lock)
    ):
        op = asyncio.create_task(push_lock.unlatch())
        for _ in range(50):
            await asyncio.sleep(0)
            if push_lock._operation_window_open:
                break
        assert push_lock._operation_lock.locked()

        # The keep-alive falls due mid-operation: _keep_alive schedules the
        # update, and _deferred_update is where it lands.
        push_lock._deferred_update()
        assert push_lock._update_task is None
        handle = push_lock._cancel_deferred_update
        assert handle is not None
        assert handle.when() - push_lock.loop.time() == pytest.approx(
            DEADLINE_WAKEUP_RETRY_DELAY, abs=0.1
        )

        gate.set()
        await op

    # The operation's own outcome is what the display carries.
    assert push_lock.lock_status == LockStatus.UNLOCKED
    mock_lock.lock_status.assert_not_awaited()

    # The retry finds the floor pushed out again, by the settle this time: our
    # own operation ends there, later than the op-response that preceded it.
    with patch.object(push_lock, "_schedule_future_update") as mock_reschedule:
        push_lock._deferred_update()

    assert push_lock._update_task is None
    mock_reschedule.assert_called_once()
    assert mock_reschedule.call_args.args[0] == pytest.approx(
        LOCK_STALE_STATE_DEBOUNCE_DELAY, abs=0.1
    )
    push_lock._cancel_future_update()
    push_lock._cancel_disconnect_timer()


@pytest.mark.asyncio
async def test_execute_operation_holds_the_next_read_per_attempt():
    """The floor is stamped immediately before each operation attempt, so a
    retry moves it forward and it tracks the latest attempt."""
    push_lock = _operational_push_lock("aa:bb:cc:dd:ee:37")
    start_stamps: list[float] = []
    attempts = 0

    async def force_lock(write_success_callback):
        nonlocal attempts
        attempts += 1
        start_stamps.append(push_lock._earliest_update_time)
        if attempts == 1:
            raise DisconnectedError("dropped before write")

    mock_lock = MagicMock()
    mock_lock.force_lock = force_lock
    # A fake clock ahead of NEVER_TIME, which is a day behind the real one.
    monotonic = itertools.count(NEVER_TIME + 90000.0, 1.0)

    with (
        patch.object(push_lock, "_ensure_connected", AsyncMock(return_value=mock_lock)),
        patch("yalexs_ble.push.asyncio.sleep", AsyncMock()),
        patch("yalexs_ble.push.time.monotonic", side_effect=monotonic),
    ):
        await push_lock.lock()

    assert attempts == 2
    # Each attempt moved the floor, the retry strictly further out than the
    # first attempt left it.
    assert len(start_stamps) == 2
    assert start_stamps[1] > start_stamps[0]
