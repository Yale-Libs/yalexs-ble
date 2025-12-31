import asyncio
import contextlib
from datetime import datetime
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from bleak_retry_connector import BLEDevice

from yalexs_ble.const import (
    AutoLockMode,
    AutoLockState,
    BatteryState,
    Commands,
    DoorStatus,
    LockActivity,
    LockOperationRemoteType,
    LockOperationSource,
    LockStatus,
    SettingType,
    StatusType,
)
from yalexs_ble.lock import Lock


@pytest.fixture
def lock() -> Lock:
    """Create a Lock instance for testing."""
    return Lock(
        lambda: BLEDevice("aa:bb:cc:dd:ee:ff", "lock"),
        "0800200c9a66",
        1,
        "mylock",
        lambda _: None,
    )


def test_create_lock(lock: Lock) -> None:
    # Simply verify the lock fixture creates a valid Lock instance
    assert lock is not None
    assert lock.name == "mylock"
    assert lock.key_index == 1


@pytest.mark.asyncio
async def test_connection_canceled_on_disconnect(lock: Lock) -> None:
    disconnect_mock = AsyncMock()
    mock_client = MagicMock(connected=True, disconnect=disconnect_mock)
    # Update the ble_device_callback if needed for delegate
    lock.ble_device_callback = lambda: BLEDevice(
        "aa:bb:cc:dd:ee:ff", "lock", delegate=""
    )
    lock.client = mock_client

    async def connect_and_wait():
        await lock.connect()
        await asyncio.sleep(2)

    with patch("yalexs_ble.lock.Lock.connect"):
        task = asyncio.create_task(connect_and_wait())
        await asyncio.sleep(0)
        task.cancel()

    with contextlib.suppress(asyncio.CancelledError):
        await task

    assert task.cancelled() is True


def test_parse_operation_source(lock: Lock) -> None:
    """Test parsing operation source and remote type."""

    # Test remote source with BLE type
    source, remote_type = lock._parse_operation_source(0x00, 0x03)
    assert source is LockOperationSource.REMOTE
    assert remote_type is LockOperationRemoteType.BLE

    # Test manual source (remote_type should be None)
    source, remote_type = lock._parse_operation_source(0x01, 0x03)
    assert source is LockOperationSource.MANUAL
    assert remote_type is None

    # Test auto lock source (remote_type should be None)
    source, remote_type = lock._parse_operation_source(0x05, 0x00)
    assert source is LockOperationSource.AUTO_LOCK
    assert remote_type is None

    # Test PIN source (remote_type should be None)
    source, remote_type = lock._parse_operation_source(0x0B, 0x03)
    assert source is LockOperationSource.PIN
    assert remote_type is None

    # Test unknown source
    source, remote_type = lock._parse_operation_source(0x99, 0x03)
    assert source is LockOperationSource.UNKNOWN
    assert remote_type is None

    # Test remote source with unknown remote type
    source, remote_type = lock._parse_operation_source(0x00, 0x99)
    assert source is LockOperationSource.REMOTE
    assert remote_type is LockOperationRemoteType.UNKNOWN

    # Test remote source with UNKNOWN (0x00) remote type
    source, remote_type = lock._parse_operation_source(0x00, 0x00)
    assert source is LockOperationSource.REMOTE
    assert remote_type is LockOperationRemoteType.UNKNOWN


def test_parse_bb_response_lock_activity(lock: Lock) -> None:
    """Test parsing 0xBB responses with lock activity."""

    # Mock _parse_lock_activity to return a mock activity
    with patch.object(lock, "_parse_lock_activity") as mock_parse:
        mock_activity = LockActivity(
            timestamp=datetime(2024, 1, 1, 12, 0),
            status=LockStatus.LOCKED,
            source=LockOperationSource.MANUAL,
        )
        mock_parse.return_value = mock_activity

        # Create a response with lock activity command
        response = bytearray(20)
        response[0] = 0xBB
        response[1] = Commands.LOCK_ACTIVITY.value

        state, activity = lock._parse_bb_response(response)
        assert state is None
        assert activity == [mock_activity]
        mock_parse.assert_called_once_with(response)

    # Test when _parse_lock_activity returns None
    with patch.object(lock, "_parse_lock_activity", return_value=None):
        state, activity = lock._parse_bb_response(response)
        assert state is None
        assert activity is None


def test_parse_bb_response_status_commands(lock: Lock) -> None:
    """Test parsing 0xBB responses with GETSTATUS command."""

    # Test GETSTATUS command
    response = bytearray(20)
    response[0] = 0xBB
    response[1] = Commands.GETSTATUS.value

    with patch.object(lock, "_parse_status_response") as mock_parse_status:
        mock_parse_status.return_value = [LockStatus.LOCKED]

        state, activity = lock._parse_bb_response(response)
        assert state == [LockStatus.LOCKED]
        assert activity is None
        mock_parse_status.assert_called_once_with(response)


def test_parse_bb_response_settings_commands(lock: Lock) -> None:
    """Test parsing 0xBB responses with settings commands."""

    # Test WRITESETTING command with autolock
    response = bytearray(20)
    response[0] = 0xBB
    response[1] = Commands.WRITESETTING.value
    response[4] = SettingType.AUTOLOCK.value

    with patch.object(lock, "_parse_auto_lock_state") as mock_parse_auto:
        mock_auto_state = AutoLockState(mode=AutoLockMode.TIMER, duration=30)
        mock_parse_auto.return_value = mock_auto_state

        state, activity = lock._parse_bb_response(response)
        assert state == [mock_auto_state]
        assert activity is None
        mock_parse_auto.assert_called_once_with(response)

    # Test READSETTING command with autolock
    response[1] = Commands.READSETTING.value
    with patch.object(lock, "_parse_auto_lock_state") as mock_parse_auto:
        mock_auto_state = AutoLockState(mode=AutoLockMode.OFF, duration=0)
        mock_parse_auto.return_value = mock_auto_state

        state, activity = lock._parse_bb_response(response)
        assert state == [mock_auto_state]
        assert activity is None


def test_parse_aa_response(lock: Lock) -> None:
    """Test parsing 0xAA responses (direct lock/unlock commands)."""

    # Test UNLOCK command
    response = bytearray(20)
    response[0] = 0xAA
    response[1] = Commands.UNLOCK.value

    state, activity = lock._parse_aa_response(response)
    assert state == [LockStatus.UNLOCKED]
    assert activity is None

    # Test LOCK command
    response[1] = Commands.LOCK.value
    state, activity = lock._parse_aa_response(response)
    assert state == [LockStatus.LOCKED]
    assert activity is None

    # Test unknown command
    response[1] = 0xFF
    state, activity = lock._parse_aa_response(response)
    assert state is None
    assert activity is None


def test_parse_status_response(lock: Lock) -> None:
    """Test parsing different status types from GETSTATUS responses."""

    # Test LOCK_ONLY status
    response = bytearray(20)
    response[4] = StatusType.LOCK_ONLY.value
    response[0x08] = LockStatus.LOCKED.value

    state = lock._parse_status_response(response)
    assert state == [LockStatus.LOCKED]

    # Test DOOR_ONLY status
    response[4] = StatusType.DOOR_ONLY.value
    response[0x08] = DoorStatus.CLOSED.value

    state = lock._parse_status_response(response)
    assert state == [DoorStatus.CLOSED]

    # Test DOOR_AND_LOCK status
    response[4] = StatusType.DOOR_AND_LOCK.value
    with patch.object(lock, "_parse_lock_and_door_state") as mock_parse:
        mock_parse.return_value = [LockStatus.LOCKED, DoorStatus.CLOSED]

        state = lock._parse_status_response(response)
        assert state == [LockStatus.LOCKED, DoorStatus.CLOSED]
        mock_parse.assert_called_once_with(response)

    # Test BATTERY status
    response[4] = StatusType.BATTERY.value
    with patch.object(lock, "_parse_battery_state") as mock_parse:
        mock_battery = BatteryState(voltage=6.0, percentage=85)
        mock_parse.return_value = mock_battery

        state = lock._parse_status_response(response)
        assert state == [mock_battery]
        mock_parse.assert_called_once_with(response)

    # Test unknown status type
    response[4] = 0xFF
    state = lock._parse_status_response(response)
    assert state is None


def test_parse_state(lock: Lock) -> None:
    """Test the main _parse_state method."""

    # Test 0xBB response
    response = bytearray(20)
    response[0] = 0xBB
    with patch.object(lock, "_parse_bb_response") as mock_parse:
        mock_parse.return_value = ([LockStatus.LOCKED], None)

        state, activity = lock._parse_state(response)
        assert state == [LockStatus.LOCKED]
        assert activity is None
        mock_parse.assert_called_once_with(response)

    # Test 0xAA response
    response[0] = 0xAA
    with patch.object(lock, "_parse_aa_response") as mock_parse:
        mock_parse.return_value = ([LockStatus.UNLOCKED], None)

        state, activity = lock._parse_state(response)
        assert state == [LockStatus.UNLOCKED]
        assert activity is None
        mock_parse.assert_called_once_with(response)

    # Test unknown response prefix
    response[0] = 0xFF
    state, activity = lock._parse_state(response)
    assert state is None
    assert activity is None


def test_parse_lock_command_response_jammed():
    """Test parsing LOCK command response with JAMMED status."""
    lock = Lock(
        lambda: BLEDevice("aa:bb:cc:dd:ee:ff", "lock"),
        "0800200c9a66",
        1,
        "mylock",
        lambda _: None,
    )

    # Frame: bb0b001b00000000000000000000001f0000
    # 0xBB = Status response, 0x0B = LOCK command, byte[3] = 0x1B = JAMMED
    frame = bytes.fromhex("bb0b001b00000000000000000000001f0000")
    result, activity = lock._parse_state(frame)

    assert result is not None
    result_list = list(result)
    assert len(result_list) == 1
    assert result_list[0] is LockStatus.JAMMED
    assert activity is None


def test_parse_lock_command_response_unlocked():
    """Test parsing LOCK command response with UNLOCKED (jam as unlocked)."""
    lock = Lock(
        lambda: BLEDevice("aa:bb:cc:dd:ee:ff", "lock"),
        "0800200c9a66",
        1,
        "mylock",
        lambda _: None,
    )

    # Frame: bb0b00030000000000000000000000370000
    # 0xBB = Status response, 0x0B = LOCK command, byte[3] = 0x03 = UNLOCKED
    frame = bytes.fromhex("bb0b00030000000000000000000000370000")
    result, activity = lock._parse_state(frame)

    assert result, activity is not None
    result_list = list(result)
    assert len(result_list) == 1
    assert result_list[0] is LockStatus.UNLOCKED
    assert activity is None


def test_parse_unlock_command_response():
    """Test parsing UNLOCK command response."""
    lock = Lock(
        lambda: BLEDevice("aa:bb:cc:dd:ee:ff", "lock"),
        "0800200c9a66",
        1,
        "mylock",
        lambda _: None,
    )

    # Frame: bb0a00030000000000000000000000000000
    # 0xBB = Status response, 0x0A = UNLOCK command, byte[3] = 0x03 = UNLOCKED
    frame = bytes.fromhex("bb0a00030000000000000000000000000000")
    result, activity = lock._parse_state(frame)

    assert result is not None
    result_list = list(result)
    assert len(result_list) == 1
    assert result_list[0] is LockStatus.UNLOCKED
    assert activity is None


def test_parse_lock_command_response_locked_success():
    """Test parsing LOCK command response with successful LOCKED status."""
    lock = Lock(
        lambda: BLEDevice("aa:bb:cc:dd:ee:ff", "lock"),
        "0800200c9a66",
        1,
        "mylock",
        lambda _: None,
    )

    # Frame: bb0b00050000000000000000000000000000
    # 0xBB = Status response, 0x0B = LOCK command, byte[3] = 0x05 = LOCKED
    frame = bytes.fromhex("bb0b00050000000000000000000000000000")
    result, activity = lock._parse_state(frame)

    assert result is not None
    result_list = list(result)
    assert len(result_list) == 1
    assert result_list[0] is LockStatus.LOCKED
    assert activity is None
