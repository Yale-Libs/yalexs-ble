import asyncio
import contextlib
from datetime import datetime
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from bleak.exc import BleakError
from bleak_retry_connector import BLEDevice

from yalexs_ble.const import (
    FIRMWARE_REVISION_CHARACTERISTIC,
    MODEL_NUMBER_CHARACTERISTIC,
    SERIAL_NUMBER_CHARACTERISTIC,
    AutoLockMode,
    AutoLockState,
    BatteryState,
    Commands,
    DoorStatus,
    LockActivity,
    LockInfo,
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

    async def connect_and_wait() -> None:
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


def test_parse_lock_command_response_jammed() -> None:
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


def test_parse_lock_command_response_unlocked() -> None:
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

    assert result is not None
    result_list = list(result)
    assert len(result_list) == 1
    assert result_list[0] is LockStatus.UNLOCKED
    assert activity is None


def test_parse_unlock_command_response() -> None:
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


def test_parse_lock_command_response_locked_success() -> None:
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


_CHAR_DATA: dict[str, bytes] = {
    MODEL_NUMBER_CHARACTERISTIC: b"ASL-03",
    SERIAL_NUMBER_CHARACTERISTIC: b"12345",
    FIRMWARE_REVISION_CHARACTERISTIC: b"2.0.0",
}

# Model is read first, then serial, firmware.
_CHAR_ORDER: tuple[str, ...] = (
    MODEL_NUMBER_CHARACTERISTIC,
    SERIAL_NUMBER_CHARACTERISTIC,
    FIRMWARE_REVISION_CHARACTERISTIC,
)


def _make_lock_with_mock_client(
    side_effects: dict[str, Exception] | None = None,
) -> tuple[Lock, MagicMock]:
    """Create a Lock with a mock BLE client for lock_info tests."""
    lock = Lock(
        lambda: BLEDevice("aa:bb:cc:dd:ee:ff", "lock", details=None),
        "0800200c9a66",
        1,
        "mylock",
        lambda _: None,
    )
    mock_client = MagicMock()
    mock_client.is_connected = True
    lock.client = mock_client
    lock.session = MagicMock()
    lock.secure_session = MagicMock()

    effects = side_effects or {}

    # Map each characteristic UUID to a unique mock object so
    # read_gatt_char can identify which UUID is being read.
    char_mocks: dict[str, MagicMock] = {}
    mock_to_uuid: dict[int, str] = {}
    for uuid in _CHAR_ORDER:
        m = MagicMock()
        char_mocks[uuid] = m
        mock_to_uuid[id(m)] = uuid

    mock_client.services.get_characteristic = char_mocks.get

    async def read_gatt_char(char: MagicMock) -> bytes:
        uuid = mock_to_uuid[id(char)]
        if uuid in effects:
            raise effects[uuid]
        return _CHAR_DATA[uuid]

    mock_client.read_gatt_char = read_gatt_char
    mock_client._mock_to_uuid = mock_to_uuid
    return lock, mock_client


@pytest.mark.asyncio
async def test_lock_info_success() -> None:
    """Test lock_info reads all characteristics successfully."""
    lock, _ = _make_lock_with_mock_client()

    info = await lock.lock_info()

    assert info == LockInfo(
        manufacturer="Yale/August",
        model="ASL-03",
        serial="12345",
        firmware="2.0.0",
    )


@pytest.mark.asyncio
async def test_lock_info_partial_failure() -> None:
    """Test lock_info continues when individual reads fail."""
    lock, _ = _make_lock_with_mock_client(
        side_effects={SERIAL_NUMBER_CHARACTERISTIC: BleakError("Connection dropped")}
    )

    info = await lock.lock_info()

    assert info.manufacturer == "Yale/August"
    assert info.model == "ASL-03"
    assert info.serial == "aa:bb:cc:dd:ee:ff"
    assert info.firmware == "2.0.0"


@pytest.mark.asyncio
async def test_lock_info_all_reads_fail() -> None:
    """Test lock_info returns all Unknown when every read fails."""
    lock, _ = _make_lock_with_mock_client(
        side_effects={uuid: BleakError("Failed") for uuid in _CHAR_ORDER}
    )

    info = await lock.lock_info()

    assert info == LockInfo(
        manufacturer="Yale/August",
        model="",
        serial="aa:bb:cc:dd:ee:ff",
        firmware="Unknown",
    )


@pytest.mark.asyncio
async def test_lock_info_timeout() -> None:
    """Test lock_info returns partial results when reads hang."""
    lock, mock_client = _make_lock_with_mock_client()

    async def hang_forever(char: MagicMock) -> bytes:
        await asyncio.sleep(999)
        return b""  # unreachable

    mock_client.read_gatt_char = hang_forever

    with patch("yalexs_ble.lock.LOCK_INFO_TIMEOUT", 0):
        info = await lock.lock_info()

    # All reads hung so no results, but we get defaults instead of an exception
    assert info.manufacturer == "Yale/August"
    assert info.model == ""
    assert info.serial == "aa:bb:cc:dd:ee:ff"
    assert info.firmware == "Unknown"


@pytest.mark.asyncio
async def test_lock_info_missing_characteristic() -> None:
    """Test lock_info skips missing characteristics instead of aborting."""
    lock, mock_client = _make_lock_with_mock_client()

    original_get = mock_client.services.get_characteristic

    def get_char_skip_serial(uuid: str) -> MagicMock | None:
        if uuid == SERIAL_NUMBER_CHARACTERISTIC:
            return None
        return original_get(uuid)

    mock_client.services.get_characteristic = get_char_skip_serial

    info = await lock.lock_info()

    assert info.manufacturer == "Yale/August"
    assert info.model == "ASL-03"
    assert info.serial == "aa:bb:cc:dd:ee:ff"
    assert info.firmware == "2.0.0"


@pytest.mark.asyncio
async def test_lock_info_reads_model_first() -> None:
    """Test that model is read first so it's available as early as possible."""
    lock, mock_client = _make_lock_with_mock_client()
    call_order: list[str] = []
    original_read = mock_client.read_gatt_char
    mock_to_uuid = mock_client._mock_to_uuid

    async def tracking_read(char: MagicMock) -> bytes:
        call_order.append(mock_to_uuid[id(char)])
        return await original_read(char)

    mock_client.read_gatt_char = tracking_read

    await lock.lock_info()

    assert call_order[0] == MODEL_NUMBER_CHARACTERISTIC


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
