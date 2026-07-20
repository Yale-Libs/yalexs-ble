import asyncio
import time
from unittest.mock import AsyncMock, MagicMock, call, patch

import pytest
from bleak.backends.scanner import AdvertisementData
from bleak.exc import BleakDBusError, BleakError

from yalexs_ble.const import (
    AutoLockMode,
    AutoLockState,
    BatteryState,
    DoorStatus,
    LockInfo,
    LockState,
    LockStatus,
)
from yalexs_ble.push import (
    _AUTH_FAILURE_HISTORY,
    AUTH_FAILURE_TO_START_REAUTH,
    BATTERY_REFRESH_INTERVAL,
    DEFAULT_ATTEMPTS,
    NEVER_TIME,
    NO_BATTERY_SUPPORT_MODELS,
    SLOW_LATENCY,
    SLOW_MAX_INTERVAL,
    SLOW_MIN_INTERVAL,
    SLOW_TIMEOUT,
    PushLock,
    operation_lock,
    retry_bluetooth_connection_error,
)
from yalexs_ble.session import DisconnectedError, ResponseError

# Shared battery-supporting lock used across tests. model is NOT in
# NO_BATTERY_SUPPORT_MODELS, so the battery-workaround path is not taken.
TEST_LOCK_INFO = LockInfo(
    manufacturer="August",
    model="ASL-03",
    serial="12345",
    firmware="2.0.0",
)


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


@pytest.mark.asyncio
async def test_operation_lock_with_retry_bluetooth_connection_error():
    """Test the operation_lock and retry_bluetooth_connection_error function."""

    counter = 0

    class MockPushLock:
        def __init__(self):
            self._operation_lock = asyncio.Lock()

        @property
        def name(self):
            return "lock"

        @retry_bluetooth_connection_error
        @operation_lock
        async def do_something(self):
            nonlocal counter
            counter += 1
            try:
                await asyncio.sleep(0.001)
                raise TimeoutError
            finally:
                counter -= 1

    lock = MockPushLock()
    tasks = []
    for _ in range(10):
        tasks.append(asyncio.create_task(lock.do_something()))

    for _ in range(10):
        await asyncio.sleep(0)
        assert counter == 1

    await asyncio.sleep(0.1)
    for _ in range(10):
        await asyncio.sleep(0)
        assert counter == 0

    for task in tasks:
        task.cancel()
    await asyncio.sleep(0)


@pytest.mark.asyncio
async def test_retry_bluetooth_connection_error_with_operation_lock():
    """Test the operation_lock and retry_bluetooth_connection_error function."""

    counter = 0

    class MockPushLock:
        def __init__(self):
            self._operation_lock = asyncio.Lock()

        @property
        def name(self):
            return "lock"

        @operation_lock
        @retry_bluetooth_connection_error
        async def do_something(self):
            nonlocal counter
            counter += 1
            try:
                await asyncio.sleep(0.001)
                raise TimeoutError
            finally:
                counter -= 1

    lock = MockPushLock()
    tasks = []
    for _ in range(10):
        tasks.append(asyncio.create_task(lock.do_something()))

    for _ in range(10):
        await asyncio.sleep(0)
        assert counter == 1

    await asyncio.sleep(0.1)
    for _ in range(10):
        await asyncio.sleep(0)
        assert counter == 0

    for task in tasks:
        task.cancel()
    await asyncio.sleep(0)


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
    mock_lock.door_status = AsyncMock(return_value=DoorStatus.CLOSED)
    mock_lock.auto_lock_status = AsyncMock(
        return_value=AutoLockState(mode=AutoLockMode.OFF, duration=0)
    )
    mock_lock.lock_status = AsyncMock(return_value=LockStatus.LOCKED)

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

    initial_state = LockState(
        lock=LockStatus.LOCKED,
        door=DoorStatus.CLOSED,
        battery=None,
        auth=None,
        auto_lock=None,
        auto_lock_prev=None,
    )

    # Call _poll_battery
    result_state, made_request = await push_lock._poll_battery(mock_lock, initial_state)

    # Should skip the request
    assert made_request is False
    mock_lock.battery.assert_not_called()
    # State should be unchanged
    assert result_state == initial_state


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
    mock_lock.battery = AsyncMock(return_value=battery_state)

    initial_state = LockState(
        lock=LockStatus.LOCKED,
        door=DoorStatus.CLOSED,
        battery=None,
        auth=None,
        auto_lock=None,
        auto_lock_prev=None,
    )

    # Call _poll_battery (cooldown should be ignored since it's in the future)
    # Wait a moment to ensure cooldown expires
    push_lock._earliest_battery_attempt_time = NEVER_TIME

    result_state, made_request = await push_lock._poll_battery(mock_lock, initial_state)

    # Should make the request
    assert made_request is True
    mock_lock.battery.assert_called_once()

    # State should have battery data
    assert result_state.battery == battery_state
    assert result_state.auth is not None
    assert result_state.auth.successful is True

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

    initial_state = LockState(
        lock=LockStatus.LOCKED,
        door=DoorStatus.CLOSED,
        battery=None,
        auth=None,
        auto_lock=None,
        auto_lock_prev=None,
    )

    # Call _poll_battery
    result_state, made_request = await push_lock._poll_battery(mock_lock, initial_state)

    # Should make the request
    assert made_request is True
    mock_lock.battery.assert_called_once()

    # State should be unchanged (error was logged but not raised)
    assert result_state == initial_state

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

    initial_state = LockState(
        lock=LockStatus.LOCKED,
        door=DoorStatus.CLOSED,
        battery=None,
        auth=None,
        auto_lock=None,
        auto_lock_prev=None,
    )

    # Call _poll_battery
    result_state, made_request = await push_lock._poll_battery(mock_lock, initial_state)

    # Should make the request
    assert made_request is True
    mock_lock.battery.assert_called_once()

    # State should be unchanged (error was logged but not raised)
    assert result_state == initial_state

    # Cooldown should NOT be set (only TimeoutError sets cooldown)
    assert push_lock._earliest_battery_attempt_time == NEVER_TIME


@pytest.mark.asyncio
async def test_update_preserves_notify_state_from_cache() -> None:
    """
    Test that _update() does not overwrite lock/door state with UNKNOWN
    when notify callbacks have updated the cached state.

    Regression test for race condition where:
    1. Update starts with UNKNOWN state
    2. Notify callback updates cached state to LOCKED/CLOSED during update
    3. Update skips polling lock_status (already seen this session)
    4. Final state should preserve LOCKED/CLOSED from cache, not revert to UNKNOWN
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

        # The critical assertion: lock/door must be preserved from cache
        assert final_state.lock == LockStatus.LOCKED, (
            f"Lock status should be LOCKED from cache, got {final_state.lock}"
        )
        assert final_state.door == DoorStatus.CLOSED, (
            f"Door status should be CLOSED from cache, got {final_state.door}"
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
    mock_lock.battery = AsyncMock(return_value=BatteryState(voltage=6.0, percentage=80))
    mock_lock.door_status = AsyncMock(return_value=DoorStatus.CLOSED)
    mock_lock.auto_lock_status = AsyncMock(
        return_value=AutoLockState(mode=AutoLockMode.OFF, duration=0)
    )
    mock_lock.lock_status = AsyncMock(return_value=LockStatus.LOCKED)

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
    mock_lock.battery = AsyncMock(return_value=BatteryState(voltage=6.0, percentage=80))
    mock_lock.door_status = AsyncMock(return_value=DoorStatus.CLOSED)
    mock_lock.auto_lock_status = AsyncMock(
        return_value=AutoLockState(mode=AutoLockMode.OFF, duration=0)
    )
    mock_lock.lock_status = AsyncMock(return_value=LockStatus.LOCKED)

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
    mock_lock.battery = AsyncMock(return_value=BatteryState(voltage=5.5, percentage=95))
    mock_lock.door_status = AsyncMock(return_value=DoorStatus.CLOSED)
    mock_lock.lock_status = AsyncMock(return_value=LockStatus.LOCKED)
    mock_lock.auto_lock_status = AsyncMock(
        return_value=AutoLockState(mode=AutoLockMode.OFF, duration=0)
    )

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
    mock_lock.battery = AsyncMock(return_value=BatteryState(voltage=5.5, percentage=95))
    mock_lock.door_status = AsyncMock(return_value=DoorStatus.CLOSED)
    mock_lock.lock_status = AsyncMock(return_value=LockStatus.LOCKED)
    mock_lock.auto_lock_status = AsyncMock(
        return_value=AutoLockState(mode=AutoLockMode.OFF, duration=0)
    )

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
    mock_lock.battery = AsyncMock(return_value=BatteryState(voltage=5.5, percentage=95))
    mock_lock.door_status = AsyncMock(return_value=DoorStatus.CLOSED)
    mock_lock.lock_status = AsyncMock(return_value=LockStatus.LOCKED)
    mock_lock.auto_lock_status = AsyncMock(
        return_value=AutoLockState(mode=AutoLockMode.OFF, duration=0)
    )

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
    mock_lock.battery = AsyncMock(return_value=battery_state)
    mock_lock.lock_status = AsyncMock(return_value=LockStatus.LOCKED)
    mock_lock.door_status = AsyncMock(return_value=DoorStatus.CLOSED)
    mock_lock.auto_lock_status = AsyncMock(
        return_value=AutoLockState(mode=AutoLockMode.OFF, duration=0)
    )
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
    mock_lock.lock_status = AsyncMock(return_value=LockStatus.LOCKED)
    mock_lock.door_status = AsyncMock(return_value=DoorStatus.CLOSED)
    mock_lock.auto_lock_status = AsyncMock(
        return_value=AutoLockState(mode=AutoLockMode.OFF, duration=0)
    )
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
    mock_lock.lock_status = AsyncMock(return_value=LockStatus.LOCKED)
    mock_lock.door_status = AsyncMock(return_value=DoorStatus.CLOSED)
    mock_lock.auto_lock_status = AsyncMock(
        return_value=AutoLockState(mode=AutoLockMode.OFF, duration=0)
    )
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

    initial_state = LockState(
        lock=LockStatus.LOCKED,
        door=DoorStatus.CLOSED,
        battery=None,
        auth=None,
        auto_lock=None,
        auto_lock_prev=None,
    )

    # Battery already polled this session and the refresh is due...
    push_lock._seen_this_session.add(BatteryState)
    refresh_deadline = time.monotonic() - 1.0
    push_lock._next_battery_refresh_time = refresh_deadline
    # ...but a prior timeout left the battery cooldown active.
    push_lock._earliest_battery_attempt_time = time.monotonic() + 100.0

    result_state, made_request = await push_lock._poll_battery(mock_lock, initial_state)

    # Cooldown gate wins: no poll, no eviction, deadline untouched.
    assert made_request is False
    mock_lock.battery.assert_not_called()
    assert BatteryState in push_lock._seen_this_session
    assert push_lock._next_battery_refresh_time == refresh_deadline
    assert result_state == initial_state


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
