import asyncio
import logging
import time
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
    AUTH_FAILURE_TO_START_REAUTH,
    AUTO_LOCK_READ_FAILURE_BACKOFF,
    AUTO_LOCK_READ_FAILURE_THRESHOLD,
    AUTO_LOCK_READ_REFRESH_INTERVAL,
    AUTO_LOCK_READ_RESPONSE_TIMEOUT,
    AUTO_LOCK_WRITE_ATTEMPTS,
    BATTERY_REFRESH_INTERVAL,
    DEFAULT_ATTEMPTS,
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
    mock_lock.auto_lock_status = AsyncMock()
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
    mock_lock.battery = AsyncMock(return_value=BatteryState(voltage=6.0, percentage=80))
    mock_lock.door_status = AsyncMock(return_value=DoorStatus.CLOSED)
    mock_lock.auto_lock_status = AsyncMock()
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
    mock_lock.auto_lock_status = AsyncMock()
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
    mock_lock.battery = AsyncMock(return_value=BatteryState(voltage=5.5, percentage=95))
    mock_lock.door_status = AsyncMock(return_value=DoorStatus.CLOSED)
    mock_lock.lock_status = AsyncMock(return_value=LockStatus.LOCKED)
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
    mock_lock.battery = AsyncMock(return_value=battery_state)
    mock_lock.lock_status = AsyncMock(return_value=LockStatus.LOCKED)
    mock_lock.door_status = AsyncMock(return_value=DoorStatus.CLOSED)
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
    mock_lock.lock_status = AsyncMock(return_value=LockStatus.LOCKED)
    mock_lock.door_status = AsyncMock(return_value=DoorStatus.CLOSED)
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
    mock_lock.lock_status = AsyncMock(return_value=LockStatus.LOCKED)
    mock_lock.door_status = AsyncMock(return_value=DoorStatus.CLOSED)
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
    mock_lock.battery = AsyncMock(return_value=BatteryState(voltage=6.0, percentage=80))
    mock_lock.door_status = AsyncMock(return_value=DoorStatus.CLOSED)
    mock_lock.auto_lock_status = AsyncMock(side_effect=TimeoutError)
    mock_lock.lock_status = AsyncMock(return_value=LockStatus.LOCKED)

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


def _auto_lock_update_lock(auto_lock_status: AsyncMock) -> MagicMock:
    """A mock Lock answering every read so _update reaches the auto lock read.

    The auto lock read itself is wired per the outcome under test.
    """
    lock = MagicMock()
    lock.connect = AsyncMock()
    lock.is_connected = True
    lock.battery = AsyncMock(return_value=BatteryState(voltage=6.0, percentage=80))
    lock.door_status = AsyncMock(return_value=DoorStatus.CLOSED)
    lock.lock_status = AsyncMock(return_value=LockStatus.LOCKED)
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
    mock_lock = _auto_lock_update_lock(AsyncMock(return_value=None))

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
    mock_lock = _auto_lock_update_lock(AsyncMock(return_value=None))
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
    mock_lock = _auto_lock_update_lock(AsyncMock(return_value=None))
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
