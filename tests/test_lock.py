import asyncio
import contextlib
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from bleak_retry_connector import BLEDevice

from yalexs_ble.const import LockOperationRemoteType, LockOperationSource, LockStatus
from yalexs_ble.lock import Lock


def test_create_lock():
    Lock(
        lambda: BLEDevice("aa:bb:cc:dd:ee:ff", "lock"),
        "0800200c9a66",
        1,
        "mylock",
        lambda _: None,
    )


@pytest.mark.asyncio
async def test_connection_canceled_on_disconnect():
    disconnect_mock = AsyncMock()
    mock_client = MagicMock(connected=True, disconnect=disconnect_mock)
    lock = Lock(
        lambda: BLEDevice("aa:bb:cc:dd:ee:ff", "lock", delegate=""),
        "0800200c9a66",
        1,
        "mylock",
        lambda _: None,
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


def test_parse_operation_source():
    """Test parsing operation source and remote type."""
    lock = Lock(
        lambda: BLEDevice("aa:bb:cc:dd:ee:ff", "lock"),
        "0800200c9a66",
        1,
        "mylock",
        lambda _: None,
    )

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
    result = lock._parse_state(frame)

    assert result is not None
    result_list = list(result)
    assert len(result_list) == 1
    assert result_list[0] is LockStatus.JAMMED


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
    result = lock._parse_state(frame)

    assert result is not None
    result_list = list(result)
    assert len(result_list) == 1
    assert result_list[0] is LockStatus.UNLOCKED


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
    result = lock._parse_state(frame)

    assert result is not None
    result_list = list(result)
    assert len(result_list) == 1
    assert result_list[0] is LockStatus.UNLOCKED


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
    result = lock._parse_state(frame)

    assert result is not None
    result_list = list(result)
    assert len(result_list) == 1
    assert result_list[0] is LockStatus.LOCKED
