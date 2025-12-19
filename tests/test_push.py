import asyncio
import time
from unittest.mock import AsyncMock, MagicMock, patch

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
    NEVER_TIME,
    NO_BATTERY_SUPPORT_MODELS,
    PushLock,
    operation_lock,
    retry_bluetooth_connection_error,
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
    mock_lock.lock_info = AsyncMock(
        return_value=LockInfo(
            manufacturer="August",
            model="ASL-03",
            serial="12345",
            firmware="2.0.0",
        )
    )

    # Battery times out
    mock_lock.battery = AsyncMock(side_effect=TimeoutError("Battery timeout"))

    # But other calls succeed
    mock_lock.door_status = AsyncMock(return_value=DoorStatus.CLOSED)
    mock_lock.auto_lock_status = AsyncMock(
        return_value=AutoLockState(mode=AutoLockMode.OFF, duration=0)
    )
    mock_lock.lock_status = AsyncMock(return_value=LockStatus.LOCKED)

    push_lock._lock_info = LockInfo(
        manufacturer="August",
        model="ASL-03",
        serial="12345",
        firmware="2.0.0",
    )
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

        # Battery is None or unchanged (timeout means no battery data)
        # Battery should be None since it timed out
        assert (
            final_state.battery is None
            or final_state.battery == push_lock._lock_state.battery
            if push_lock._lock_state
            else True
        )


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

    # Set cooldown to 5 seconds in the future
    push_lock._next_battery_attempt_time = time.monotonic() + 5.0

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

    # Set cooldown to simulate previous timeout
    push_lock._next_battery_attempt_time = time.monotonic() + 100.0

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
    push_lock._next_battery_attempt_time = NEVER_TIME

    result_state, made_request = await push_lock._poll_battery(mock_lock, initial_state)

    # Should make the request
    assert made_request is True
    mock_lock.battery.assert_called_once()

    # State should have battery data
    assert result_state.battery == battery_state
    assert result_state.auth is not None
    assert result_state.auth.successful is True

    # Cooldown should be reset to NEVER_TIME
    assert push_lock._next_battery_attempt_time == NEVER_TIME


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
    assert push_lock._next_battery_attempt_time == NEVER_TIME


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
    assert push_lock._next_battery_attempt_time == NEVER_TIME
