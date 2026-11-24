import asyncio
import contextlib
import logging
from collections.abc import Callable, Iterable
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from bleak.exc import BleakError
from bleak_retry_connector import BLEDevice
from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes

import yalexs_ble
from yalexs_ble import util
from yalexs_ble.const import (
    FIRMWARE_REVISION_CHARACTERISTIC,
    MECHANICAL_OPERATION_ERRORS,
    MODEL_NUMBER_CHARACTERISTIC,
    RESPONSE_FRAME_LEN,
    SERIAL_NUMBER_CHARACTERISTIC,
    VALUE_TO_LOCK_STATUS,
    BatteryState,
    Commands,
    DoorStatus,
    LockInfo,
    LockStateValue,
    LockStatus,
    OpCode,
    OperationError,
    StatusType,
)
from yalexs_ble.lock import (
    AA_BATTERY_VOLTAGE_TO_PERCENTAGE,
    MAX_ACTIVITY_RECORDS,
    UNLATCH_OPERATION_BYTE,
    Lock,
    _ack_matcher,
    _keycode_response_matcher,
    _operation_response_matcher,
    _parameter_response_matcher,
    _poll_response_matcher,
    convert_voltage_to_percentage,
)
from yalexs_ble.session import (
    DisconnectedError,
    KeycodeError,
    OperationFailedError,
    OperationIncompleteError,
    OperationProgress,
    ResponseError,
    Session,
    UnityParameterError,
)
from yalexs_ble.util import _simple_checksum


def test_aa_battery_voltage_to_percentage_is_monotonic() -> None:
    """Percentage must be non-increasing as voltage decreases.

    Guards against copy/paste regressions in the lookup table — a non-monotonic
    table makes ``convert_voltage_to_percentage`` return higher percentages for
    lower voltages, which erodes user trust in the battery indicator.
    """
    sorted_pairs = sorted(AA_BATTERY_VOLTAGE_TO_PERCENTAGE)
    percents = [pct for _, pct in sorted_pairs]
    assert percents == sorted(percents), (
        f"voltage→pct table is non-monotonic: {sorted_pairs}"
    )


def test_convert_voltage_to_percentage_is_monotonic_across_table() -> None:
    """``convert_voltage_to_percentage`` must be non-decreasing in voltage."""
    voltages = sorted(v for v, _ in AA_BATTERY_VOLTAGE_TO_PERCENTAGE)
    results = [convert_voltage_to_percentage(v) for v in voltages]
    assert results == sorted(results), (
        f"convert_voltage_to_percentage is non-monotonic across table voltages: "
        f"{list(zip(voltages, results, strict=True))}"
    )


def test_create_lock() -> None:
    Lock(
        lambda: BLEDevice("aa:bb:cc:dd:ee:ff", "lock"),
        "0800200c9a66",
        1,
        "mylock",
        lambda _: None,
    )


@pytest.mark.asyncio
async def test_connection_canceled_on_disconnect() -> None:
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


def test_parse_lock_command_response_jammed() -> None:
    """LOCK op-response with a MECH_* result (byte[15]) parses as JAMMED."""
    lock = _make_lock()

    # Real lock-jam capture: byte[15] = 0x1F MECH_POSITION. byte[3] (0x1B
    # here) is only the frame checksum, not a status.
    frame = bytes.fromhex("bb0b001b00000000000000000000001f0000")
    result = lock._parse_state(frame)

    assert result is not None
    assert list(result) == [LockStatus.JAMMED]


def test_parse_unlock_command_response_jammed() -> None:
    """UNLOCK op-response with a MECH_* result (byte[15]) parses as JAMMED.

    The old byte[3] path missed this: an unlock jam's checksum is 0x1C, not
    the 0x1B it looked for. The result is in byte[15] (0x1F MECH_POSITION)
    regardless of direction.
    """
    lock = _make_lock()

    frame = bytes.fromhex("bb0a001c00000000000000000000001f0000")
    result = lock._parse_state(frame)

    assert result is not None
    assert list(result) == [LockStatus.JAMMED]


def test_parse_lock_command_response_success_is_no_update() -> None:
    """A successful LOCK op-response (byte[15]=0x00) carries no state update.

    The op-response reports the result of the issued command; which state
    resulted is known to the command issuer, not the parser (lock and
    securemode op-responses are byte-identical), so the parser emits nothing.
    """
    lock = _make_lock()

    frame = bytes.fromhex("bb0b003a0000000000000000000000000000")
    result = lock._parse_state(frame)

    assert result is not None
    assert list(result) == []


def test_parse_unlock_command_response_success_is_no_update() -> None:
    """A successful UNLOCK op-response (byte[15]=0x00) carries no state update."""
    lock = _make_lock()

    frame = bytes.fromhex("bb0a003b0000000000000000000000000000")
    result = lock._parse_state(frame)

    assert result is not None
    assert list(result) == []


def test_parse_getstatus_staticposition() -> None:
    """A settled GETSTATUS lock state of 0x07 (STATICPOSITION) parses as JAMMED."""
    lock = _make_lock()

    # bb02 GETSTATUS, byte[4]=0x02 LOCK_ONLY, byte[8]=0x07 (settled jam state).
    frame = bytes.fromhex("bb02003a0200000007000000000000000000")
    result = lock._parse_state(frame)

    assert result is not None
    assert list(result) == [LockStatus.JAMMED]


@pytest.mark.parametrize(
    ("frame_hex", "expected"),
    [
        # Synthetic frames in the GETSTATUS layout, not captured from a device.
        ("bb0200380200000009000000000000000000", LockStatus.UNLATCHING),
        ("bb020037020000000a000000000000000000", LockStatus.UNLATCHED),
    ],
    ids=["unlatching", "unlatched"],
)
def test_parse_getstatus_unlatch_states(frame_hex: str, expected: LockStatus) -> None:
    """A GETSTATUS lock state of 0x09 or 0x0A decodes to the unlatch states."""
    lock = _make_lock()

    result = lock._parse_state(bytes.fromhex(frame_hex))

    assert result is not None
    assert list(result) == [expected]


@pytest.mark.parametrize(
    ("value", "expected", "diagnostic_logged"),
    [
        (0x09, LockStatus.UNLATCHING, False),
        (0x0A, LockStatus.UNLATCHED, False),
        (0x08, LockStatus.UNKNOWN, True),
    ],
    ids=["unlatching", "unlatched", "still_unmapped"],
)
def test_parse_lock_status_decodes_the_unlatch_states(
    value: int,
    expected: LockStatus,
    diagnostic_logged: bool,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """_parse_lock_status decodes 0x09 and 0x0A without logging; 0x08 still logs."""
    lock = _make_lock()

    with caplog.at_level("INFO", logger="yalexs_ble.lock"):
        assert lock._parse_lock_status(value) is expected

    assert ("Unrecognized lock_status_str" in caplog.text) is diagnostic_logged


def test_parse_success_op_response_with_0200_trailer_is_no_update(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """The issue #317 fixture: byte[3] shifts with the plaintext trailer.

    A successful unlock op-response with the ``0200`` CommandType trailer
    moves byte[3] to 0x39. Keying off byte[3] would miss it; keying off
    byte[15]=0x00 recognizes it as a successful op-response with no state
    update -- and it must not log "Unknown state".
    """
    lock = _make_lock()

    frame = bytes.fromhex("bb0a00390000000000000000000000000200")
    with caplog.at_level("INFO", logger="yalexs_ble.lock"):
        result = lock._parse_state(frame)
        lock._internal_state_callback(frame)

    assert result is not None
    assert list(result) == []
    assert "Unknown state" not in caplog.text


def test_a_get_log_reply_is_no_state_update(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """A GETLOG (0xBB 0x2D) frame is recognized with no state update."""
    lock = _make_lock()

    frame = bytes.fromhex("bb2d008000000000000000000000000000")
    with caplog.at_level("INFO", logger="yalexs_ble.lock"):
        result = lock._parse_state(frame)
        lock._internal_state_callback(frame)

    assert result is not None
    assert list(result) == []
    assert "Unknown state" not in caplog.text


def _failure_log_levels(caplog: pytest.LogCaptureFixture) -> list[str]:
    """The levels of the operation-failure log lines, in order."""
    return [
        record.levelname
        for record in caplog.records
        if "Operation failed with result" in record.message
    ]


def _arm_op_response_wait(lock: Lock, opcode: int) -> None:
    """Arm the session's staged wait as a running operation on opcode would."""
    session = lock.session
    assert session is not None
    session._operation_progress = OperationProgress()
    session._notify_matcher = _operation_response_matcher(opcode)
    session._notify_future = asyncio.get_running_loop().create_future()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("awaited_opcode", "result_byte", "expected_level"),
    [
        *(
            (Commands.LOCK.value, error.value, "DEBUG")
            for error in sorted(MECHANICAL_OPERATION_ERRORS)
        ),
        (Commands.UNLOCK.value, 0x1F, "WARNING"),
        (None, 0x1F, "WARNING"),
    ],
)
async def test_parse_op_response_failure_log_level(
    awaited_opcode: int | None,
    result_byte: int,
    expected_level: str,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Only a mechanical failure of our own awaited operation logs at DEBUG."""
    lock = _make_connected_lock_with_session()
    if awaited_opcode is not None:
        _arm_op_response_wait(lock, awaited_opcode)

    frame = _op_response_frame(Commands.LOCK, result_byte)
    with caplog.at_level("DEBUG", logger="yalexs_ble.lock"):
        result = lock._parse_state(frame)

    assert result is not None
    assert list(result) == [LockStatus.JAMMED]
    assert _failure_log_levels(caplog) == [expected_level]


@pytest.mark.asyncio
async def test_parse_non_mech_error_is_jammed_and_logs_decoded_name(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """A non-MECH failure result still parses as JAMMED and logs its name."""
    lock = _make_connected_lock_with_session()
    _arm_op_response_wait(lock, Commands.LOCK.value)

    # byte[15] = 0x32 VBAT_LOW (synthetic; no real capture for a non-MECH error).
    # Armed, so the WARNING pins a non-mechanical result of our own operation.
    frame = bytes.fromhex("bb0b00000000000000000000000000320000")
    with caplog.at_level("WARNING", logger="yalexs_ble.lock"):
        result = lock._parse_state(frame)

    assert result is not None
    assert list(result) == [LockStatus.JAMMED]
    assert [record.levelname for record in caplog.records] == ["WARNING"]
    assert "0x32" in caplog.text
    assert "VBAT_LOW" in caplog.text


def test_parse_unknown_error_code_is_jammed_and_logs_unknown(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """An unmapped non-zero result is JAMMED and logs the raw value as unknown."""
    lock = _make_lock()

    frame = bytes.fromhex("bb0b00000000000000000000000000770000")
    with caplog.at_level("WARNING", logger="yalexs_ble.lock"):
        result = lock._parse_state(frame)

    assert result is not None
    assert list(result) == [LockStatus.JAMMED]
    assert "0x77" in caplog.text
    assert "unknown" in caplog.text


@pytest.mark.parametrize(
    "frame",
    [
        # Constructed frames.
        bytes.fromhex("cc00000000000000000000000000000000"),
        bytes.fromhex("bb0500000000000000000000000000000000"),
        bytes.fromhex("aa2e00282800000000000000000000000200"),
    ],
    ids=["unhandled-0xcc-flag", "unhandled-0xbb-opcode", "unhandled-0xaa-opcode"],
)
def test_parse_bogus_frame_is_none_and_logs_unknown(
    frame: bytes, caplog: pytest.LogCaptureFixture
) -> None:
    """_parse_state returns None for a frame it does not handle, logged as unknown."""
    lock = _make_lock()

    with caplog.at_level("INFO", logger="yalexs_ble.lock"):
        assert lock._parse_state(frame) is None
        lock._internal_state_callback(frame)

    assert "Unknown state" in caplog.text


def test_internal_state_callback_emits_recognized_state() -> None:
    """A recognized frame with state content reaches the state callback."""
    received: list[list[LockStateValue]] = []
    lock = _make_lock(lambda states: received.append(list(states)))

    # Settled status push after a jam: GETSTATUS/LOCK_ONLY with state 0x07
    # (production capture).
    lock._internal_state_callback(bytes.fromhex("bb02003a0200000007000000000000000000"))

    assert received == [[LockStatus.JAMMED]]


def test_jammed_maps_to_the_settled_static_position_value() -> None:
    """JAMMED is the settled post-jam status value 0x07 (STATICPOSITION)."""
    assert LockStatus(0x07) is LockStatus.JAMMED
    assert VALUE_TO_LOCK_STATUS[0x07] is LockStatus.JAMMED


def _make_lock(
    state_callback: Callable[[Iterable[LockStateValue]], None] = lambda _: None,
    *,
    op_response_callback: Callable[[], None] | None = None,
) -> Lock:
    return Lock(
        lambda: BLEDevice("aa:bb:cc:dd:ee:ff", "lock"),
        "0800200c9a66",
        1,
        "mylock",
        state_callback,
        op_response_callback=op_response_callback,
    )


# Captured settings frames, each with setting id 0x28 at byte 4.
SETTINGS_READ_ACK = bytes.fromhex("aa0400282800000000000000000000000200")
SETTINGS_READ_RESPONSE = bytes.fromhex("bb0400fb2800000008070807000000000000")
SETTINGS_WRITE_ACK = bytes.fromhex("aa030075280000005a005a00000000000200")
SETTINGS_WRITE_RESPONSE = bytes.fromhex("bb030066280000005a005a00000000000000")

# Captured Unity acknowledgments, each with id 0x01 at byte 4.
UNITY_READ_ACK = bytes.fromhex("aa35001e0100000000000000000000000200")
UNITY_WRITE_ACK = bytes.fromhex("aa34001d0102000000000000000000000200")


def test_parameter_response_matcher_takes_value_frame_not_ack() -> None:
    """The matcher takes the 0xBB frame with its opcode and setting id and no other."""
    write_matcher = _parameter_response_matcher(
        Commands.WRITESETTING.value, SETTINGS_WRITE_RESPONSE[0x04]
    )
    other_id_matcher = _parameter_response_matcher(Commands.WRITESETTING.value, 0x9E)
    battery_answer = bytes.fromhex("bb0200a50f00000079140000000000000200")

    assert write_matcher(SETTINGS_WRITE_RESPONSE)
    assert not write_matcher(SETTINGS_WRITE_ACK)
    assert not write_matcher(SETTINGS_READ_RESPONSE)  # wrong opcode for the write
    assert not write_matcher(battery_answer)
    assert not write_matcher(SETTINGS_WRITE_RESPONSE[:4])  # truncated below byte 4
    assert not other_id_matcher(SETTINGS_WRITE_RESPONSE)


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
    data_overrides: dict[str, bytes] | None = None,
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
    overrides = data_overrides or {}

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
        return overrides.get(uuid, _CHAR_DATA[uuid])

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
async def test_lock_info_non_utf8_read_degrades_to_fallback() -> None:
    """A corrupt read that is not UTF-8 degrades like a failed one.

    The characteristic read is radio input too — the BLE controller bug
    noted in lock_info corrupts packets — and decode() raises
    UnicodeDecodeError, which is not a BleakError, so it used to escape the
    handler and abort lock_info with the partial results discarded.
    """
    lock, _ = _make_lock_with_mock_client(
        data_overrides={SERIAL_NUMBER_CHARACTERISTIC: b"\xff\xfe\xff"}
    )

    info = await lock.lock_info()

    assert info.model == "ASL-03"
    # The BLE address stands in for the unreadable serial.
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
async def test_lock_info_timeout_retries_once_then_succeeds() -> None:
    """A transient timeout on the first pass is retried and recovers."""
    lock, mock_client = _make_lock_with_mock_client()
    original_read = mock_client.read_gatt_char
    calls = 0

    async def hang_first_call(char: MagicMock) -> bytes:
        nonlocal calls
        calls += 1
        if calls == 1:
            await asyncio.sleep(999)
        return await original_read(char)

    mock_client.read_gatt_char = hang_first_call

    with patch("yalexs_ble.lock.LOCK_INFO_TIMEOUT", 0.05):
        info = await lock.lock_info()

    assert info == LockInfo(
        manufacturer="Yale/August",
        model="ASL-03",
        serial="12345",
        firmware="2.0.0",
    )


@pytest.mark.asyncio
async def test_lock_info_model_read_error_is_retried() -> None:
    """A model read that fails outright, not only one that hangs, gets the retry."""
    lock, mock_client = _make_lock_with_mock_client()
    original_read = mock_client.read_gatt_char
    calls = 0

    async def fail_first_call(char: MagicMock) -> bytes:
        nonlocal calls
        calls += 1
        if calls == 1:
            raise BleakError("Operation already in progress")
        return await original_read(char)

    mock_client.read_gatt_char = fail_first_call
    info = await lock.lock_info()
    assert info.model == "ASL-03"
    assert calls == 4


@pytest.mark.asyncio
async def test_lock_info_retry_keeps_partial_results() -> None:
    """The retry pass only reads the characteristics still missing."""
    lock, mock_client = _make_lock_with_mock_client()
    original_read = mock_client.read_gatt_char
    mock_to_uuid = mock_client._mock_to_uuid
    call_counts: dict[str, int] = {}

    async def hang_first_serial_read(char: MagicMock) -> bytes:
        uuid = mock_to_uuid[id(char)]
        call_counts[uuid] = call_counts.get(uuid, 0) + 1
        if uuid == SERIAL_NUMBER_CHARACTERISTIC and call_counts[uuid] == 1:
            await asyncio.sleep(999)
        return await original_read(char)

    mock_client.read_gatt_char = hang_first_serial_read

    with patch("yalexs_ble.lock.LOCK_INFO_TIMEOUT", 0.05):
        info = await lock.lock_info()

    assert info == LockInfo(
        manufacturer="Yale/August",
        model="ASL-03",
        serial="12345",
        firmware="2.0.0",
    )
    # The model landed on the first pass and is not read again.
    assert call_counts[MODEL_NUMBER_CHARACTERISTIC] == 1
    assert call_counts[SERIAL_NUMBER_CHARACTERISTIC] == 2


@pytest.mark.asyncio
async def test_lock_info_timeout_on_both_attempts_falls_back(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Both passes timing out degrades to the fallback with one warning."""
    lock, mock_client = _make_lock_with_mock_client()
    mock_to_uuid = mock_client._mock_to_uuid
    read_uuids: list[str] = []

    async def hang_forever(char: MagicMock) -> bytes:
        read_uuids.append(mock_to_uuid[id(char)])
        await asyncio.sleep(999)
        return b""  # unreachable

    mock_client.read_gatt_char = hang_forever

    with patch("yalexs_ble.lock.LOCK_INFO_TIMEOUT", 0.05):
        info = await lock.lock_info()

    assert info == LockInfo(
        manufacturer="Yale/August",
        model="",
        serial="aa:bb:cc:dd:ee:ff",
        firmware="Unknown",
    )
    # One read per pass proves the retry ran; the warning fires only once,
    # on the final pass.
    assert read_uuids == [MODEL_NUMBER_CHARACTERISTIC, MODEL_NUMBER_CHARACTERISTIC]
    warnings = [
        record
        for record in caplog.records
        if record.levelname == "WARNING" and "Lock info incomplete" in record.message
    ]
    assert len(warnings) == 1


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


# --------------------------------------------------------------------------- #
# Typed poll waits
# --------------------------------------------------------------------------- #
# byte[1] carries the polled opcode and byte[4] the status type; byte[3] is the
# checksum and each frame sums to zero.
BATTERY_FRAME = bytes.fromhex("bb0200a50f00000079140000000000000200")
LOCK_FRAME = bytes.fromhex("bb02003c0200000003000000000000000200")
DOOR_FRAME = bytes.fromhex("bb0200122e00000001000000000000000200")
# Field-captured 0xAA acknowledgments; byte[4] echoes the operation byte.
LOCK_ACK = bytes.fromhex("aa0b00490000000000000000000000000200")
UNLOCK_ACK = bytes.fromhex("aa0a004a0000000000000000000000000200")
SECUREMODE_ACK = bytes.fromhex("aa0b00450400000000000000000000000200")


def _with_checksum(hex_str: str) -> bytes:
    """Build an 18-byte frame with a valid simple checksum in byte[3].

    byte[3] is the checksum field and ``_validate_response`` requires the
    frame to sum to zero.
    """
    frame = bytearray.fromhex(hex_str)
    frame[0x03] = 0
    frame[0x03] = _simple_checksum(frame)
    return bytes(frame)


# A LOCK_EVENTS_UNREAD reply captured on 2026-07-26 from a Yale lock: 242 records.
DEPTH_FRAME = bytes.fromhex("bb02004609000000f2000000000000000200")
# Built to the same layout with 0xDEADBEEF in the count at byte[8:12].
NEGATIVE_DEPTH_FRAME = _with_checksum("bb02000009000000efbeadde000000000200")


# Built GETLOG replies, each holding one activity record: byte[2] is the
# record index and byte[4] the record type. 0x80 ends the log; 0x20 is a door
# state change.
ACTIVITY_FRAME = _with_checksum("bb2d00008000000000000000000000000000")
DOOR_ACTIVITY_FRAME = _with_checksum("bb2d00002000000000010000000000000000")


def _connected_lock(
    state_callback: Callable[[Iterable[LockStateValue]], None] = lambda _: None,
) -> tuple[Lock, Session]:
    """A Lock wired to a real Session with pass-through crypto.

    The matcher is applied inside Session.execute, so pinning the wiring
    between a poll and its matcher needs a real session rather than a mock.
    """
    lock = _make_lock(state_callback)
    client = MagicMock(is_connected=True)
    session = Session(
        client, "mylock", asyncio.Lock(), set(), lock._internal_state_callback
    )
    session.decrypt = bytes  # type: ignore[method-assign, assignment]
    session.cipher_encrypt = MagicMock(update=bytes)
    lock.client = client
    lock.session = session
    lock.secure_session = MagicMock()
    return lock, session


def test_poll_response_matcher_takes_only_the_requested_subtype() -> None:
    """0xBB plus the polled opcode plus the byte[4] status type."""
    matches = _poll_response_matcher(Commands.GETSTATUS.value, StatusType.BATTERY.value)
    assert matches(BATTERY_FRAME)
    # Right opcode, wrong subtype.
    assert not matches(DOOR_FRAME)
    assert not matches(LOCK_FRAME)
    # Right opcode and subtype, but an acknowledgment rather than a response.
    assert not matches(_with_checksum("aa02000f0f00000000000000000000000000"))
    # Wrong opcode.
    assert not matches(ACTIVITY_FRAME)


def test_poll_response_matcher_without_a_subtype_ignores_byte_four() -> None:
    """get_log's answer has a record type at byte[4], not a status type.

    So its matcher keys on the opcode alone and must accept any byte[4].
    """
    matches = _poll_response_matcher(Commands.GETLOG.value)
    assert matches(ACTIVITY_FRAME)
    assert matches(_with_checksum("bb2d00002000000000000000000000000000"))
    # Still opcode-gated.
    assert not matches(BATTERY_FRAME)
    # An acknowledgment carries the request's own byte[4], which is zero for
    # this command and so reads as a lock operation record. Only the 0xBB check
    # keeps it from being returned as a record.
    assert not matches(_with_checksum("aa2d00000000000000000000000000000000"))


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("poll", "stray", "stray_state", "answer", "expected"),
    [
        (
            "battery",
            DOOR_FRAME,
            DoorStatus.CLOSED,
            BATTERY_FRAME,
            BatteryState(5.241, 28),
        ),
        (
            "lock_status",
            BATTERY_FRAME,
            BatteryState(5.241, 28),
            LOCK_FRAME,
            LockStatus.UNLOCKED,
        ),
        # The door answer is CLOSED, not OPENED: DoorStatus.OPENED and
        # LockStatus.UNLOCKED are both 0x03, so the lock stray mis-parses to
        # exactly OPENED and an OPENED expectation would pass either way.
        ("door_status", LOCK_FRAME, LockStatus.UNLOCKED, DOOR_FRAME, DoorStatus.CLOSED),
        (
            "unread_event_count",
            LOCK_FRAME,
            LockStatus.UNLOCKED,
            DEPTH_FRAME,
            242,
        ),
    ],
)
async def test_a_poll_is_not_answered_by_a_frame_of_another_subtype(
    poll: str,
    stray: bytes,
    stray_state: LockStateValue,
    answer: bytes,
    expected: object,
) -> None:
    """A stray push arriving mid-poll does not resolve the wait.

    Every one of these strays is a valid frame that the untyped wait accepted
    as the poll's answer, so the poll's own parser read the wrong bytes: the
    door frame answering a battery poll is the recurring near-zero voltage
    seen in the field. The stray still reaches the state callback, which is
    where it belonged all along. Each answer decodes to a value its own stray
    cannot produce, so the returned value discriminates on its own.
    """
    emitted: list[list[LockStateValue]] = []
    lock, session = _connected_lock(lambda states: emitted.append(list(states)))

    async def deliver(*_args: object, **_kwargs: object) -> None:
        session._notify(0, bytearray(stray))
        # The stray must not have answered the poll.
        assert session._notify_future is not None
        session._notify(0, bytearray(answer))

    session.client.write_gatt_char = AsyncMock(side_effect=deliver)

    assert await getattr(lock, poll)() == expected
    # The stray was decoded as what it is, on the path it belonged on.
    assert [stray_state] in emitted


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("frame", "unread"),
    [(DEPTH_FRAME, 242), (NEGATIVE_DEPTH_FRAME, -559038737)],
    ids=["captured", "top-bit-set"],
)
async def test_unread_event_count_returns_the_signed_four_byte_count(
    frame: bytes, unread: int
) -> None:
    """The count read sends GETSTATUS LOCK_EVENTS_UNREAD and reads [8:12] signed.

    A negative count is not a depth and raises instead of returning.
    """
    lock, session = _connected_lock()

    async def deliver(*_args: object, **_kwargs: object) -> None:
        session._notify(0, bytearray(frame))

    write = AsyncMock(side_effect=deliver)
    session.client.write_gatt_char = write

    if unread < 0:
        with pytest.raises(ResponseError, match=f"{unread} is not a depth"):
            await lock.unread_event_count()
    else:
        assert await lock.unread_event_count() == unread
    (write_call,) = write.await_args_list
    written = write_call.args[1]
    assert (written[0x01], written[0x04]) == (
        Commands.GETSTATUS.value,
        StatusType.LOCK_EVENTS_UNREAD.value,
    )


@pytest.mark.asyncio
async def test_an_unsolicited_count_reply_answers_no_other_wait(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """A count reply during a battery poll ends no wait and publishes nothing."""
    emitted: list[list[LockStateValue]] = []
    lock, session = _connected_lock(lambda states: emitted.append(list(states)))

    async def deliver(*_args: object, **_kwargs: object) -> None:
        session._notify(0, bytearray(DEPTH_FRAME))
        assert session._notify_future is not None
        session._notify(0, bytearray(BATTERY_FRAME))

    session.client.write_gatt_char = AsyncMock(side_effect=deliver)

    with caplog.at_level(logging.INFO, logger="yalexs_ble.lock"):
        assert await lock.battery() == BatteryState(5.241, 28)

    assert emitted == [[BatteryState(5.241, 28)]]
    assert "Unknown state" not in caplog.text


def test_another_getstatus_subtype_is_still_unknown(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """A GETSTATUS reply of a subtype the library does not decode stays unknown."""
    lock = _make_lock()
    # Built: a GETSTATUS reply with subtype 0x0A at byte[4].
    frame = _with_checksum("bb0200000a00000000000000000000000000")

    with caplog.at_level(logging.INFO, logger="yalexs_ble.lock"):
        assert lock._parse_state(frame) is None
        lock._internal_state_callback(frame)

    assert "Unknown state" in caplog.text


@pytest.mark.asyncio
async def test_unread_event_count_requires_a_connection() -> None:
    """Reading the count from a disconnected lock raises."""
    with pytest.raises(DisconnectedError):
        await _make_lock().unread_event_count()


@pytest.mark.asyncio
async def test_a_get_log_is_not_answered_by_a_status_frame() -> None:
    """get_log matches on the opcode alone, so a status push does not answer it."""
    emitted: list[list[LockStateValue]] = []
    lock, session = _connected_lock(lambda states: emitted.append(list(states)))

    async def deliver(*_args: object, **_kwargs: object) -> None:
        session._notify(0, bytearray(LOCK_FRAME))
        assert session._notify_future is not None
        session._notify(0, bytearray(DOOR_ACTIVITY_FRAME))

    session.client.write_gatt_char = AsyncMock(side_effect=deliver)

    assert await lock.get_log() == DOOR_ACTIVITY_FRAME
    assert emitted == [[LockStatus.UNLOCKED]]


GET_200_PIN = bytes.fromhex("bb39004ec800135790ffffffff0000000000")
GET_200_EMPTY = bytes.fromhex("bb39004bc800ffffffffffffff0000000000")
GET_1_NOSPACE = bytes.fromhex("bb3900000100000000000000000000070000")
ACK_CLEAR = bytes.fromhex("aa2800000000000000000000000000000200")


def _result(opcode: int, error: int = 0, slot: int = 0) -> bytes:
    """A synthetic 0xBB result frame (checksum not needed by the stub session)."""
    frame = bytearray(18)
    frame[0x00] = 0xBB
    frame[0x01] = opcode
    frame[0x04:0x06] = slot.to_bytes(2, "little")
    frame[0x0F] = error
    return bytes(frame)


class _KeycodeSession:
    """Session stand-in that records commands and replays canned frames.

    Each replayed frame is asserted to pass the command's matcher.
    """

    def __init__(self, responses: list[bytes]) -> None:
        self.sent: list[bytearray] = []
        self.responses = responses

    build_command = Session.build_command

    async def execute(
        self,
        command: bytearray,
        command_name: str,
        response_matcher: Callable[[bytes], bool],
    ) -> bytes:
        self.sent.append(command)
        response = self.responses.pop(0)
        assert response_matcher(response)
        return response


def _keycode_lock(responses: list[bytes]) -> tuple[Lock, _KeycodeSession]:
    session = _KeycodeSession(responses)
    lock = _make_lock()
    lock.session = session  # type: ignore[assignment]
    lock.secure_session = MagicMock()
    lock.client = MagicMock(is_connected=True)
    return lock, session


SET_KEYCODE_STEPS = (0x28, 0x27, 0x2B, 0x2C)


@pytest.mark.asyncio
async def test_get_keycode_decodes_the_pin() -> None:
    lock, session = _keycode_lock([GET_200_PIN])
    assert await lock.get_keycode(200) == "135790"
    cmd = session.sent[0]
    assert cmd[0x01] == Commands.KEYCODE_GET.value
    assert cmd[0x04:0x06] == bytes.fromhex("c800")


@pytest.mark.asyncio
async def test_get_keycode_empty_slot_is_none() -> None:
    lock, _ = _keycode_lock([GET_200_EMPTY])
    assert await lock.get_keycode(200) is None


@pytest.mark.asyncio
async def test_get_keycode_reports_the_lock_error() -> None:
    lock, _ = _keycode_lock([GET_1_NOSPACE])
    with pytest.raises(KeycodeError) as exc_info:
        await lock.get_keycode(1)
    assert exc_info.value.error == OperationError.KEYCODE_NOSPACE
    assert exc_info.value.command == "get_keycode"


@pytest.mark.asyncio
async def test_unknown_error_code_is_carried_as_an_int() -> None:
    lock, _ = _keycode_lock([_result(Commands.KEYCODE_CLEAR.value, 0x7F)])
    with pytest.raises(KeycodeError) as exc_info:
        await lock.clear_keycode(5)
    assert exc_info.value.error == 0x7F
    assert not isinstance(exc_info.value.error, OperationError)


@pytest.mark.asyncio
async def test_clear_keycode_command_layout() -> None:
    lock, session = _keycode_lock([_result(Commands.KEYCODE_CLEAR.value)])
    await lock.clear_keycode(0x01C8)
    cmd = session.sent[0]
    assert cmd[0x01] == Commands.KEYCODE_CLEAR.value
    assert cmd[0x04:0x0B] == b"\xff" * 7
    assert (cmd[0x0B], cmd[0x0C], cmd[0x0D]) == (0xC8, 0x00, 0x01)
    assert cmd[0x10] == 0x02


@pytest.mark.asyncio
async def test_set_keycode_runs_clear_set_access_commit_in_order() -> None:
    lock, session = _keycode_lock([_result(op) for op in SET_KEYCODE_STEPS])
    await lock.set_keycode(200, "135790")
    clear, set_, access, commit = session.sent
    assert [c[0x01] for c in session.sent] == list(SET_KEYCODE_STEPS)
    assert clear[0x04:0x0B] == b"\xff" * 7
    assert clear[0x0B] == 200
    assert set_[0x04:0x0B] == bytes.fromhex("135790ffffffff")
    assert set_[0x0C] == 0x00
    assert access[0x04:0x0C] == bytes(8)
    assert access[0x0C] == 0x80
    assert access[0x0D] == 0x00
    assert commit[0x04:0x0B] == bytes.fromhex("135790ffffffff")
    assert (commit[0x0B], commit[0x0C], commit[0x0D]) == (200, 0x00, 0x00)


@pytest.mark.asyncio
async def test_set_keycode_odd_length_pin_and_high_slot() -> None:
    lock, session = _keycode_lock([_result(op) for op in SET_KEYCODE_STEPS])
    await lock.set_keycode(0x0102, "12345")
    assert session.sent[1][0x04:0x0B] == bytes.fromhex("12345fffffffff")
    assert (session.sent[3][0x0B], session.sent[3][0x0D]) == (0x02, 0x01)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("error", "failing_step"),
    [(0x06, 0x2C), (0x39, 0x2C), (0x07, 0x28)],
)
async def test_set_keycode_stops_at_the_failing_step(
    error: int, failing_step: int
) -> None:
    steps = SET_KEYCODE_STEPS
    responses = [
        _result(op, error if op == failing_step else 0)
        for op in steps[: steps.index(failing_step) + 1]
    ]
    lock, session = _keycode_lock(responses)
    with pytest.raises(KeycodeError) as exc_info:
        await lock.set_keycode(251, "1234")
    assert exc_info.value.error == error
    assert session.sent[-1][0x01] == failing_step
    assert len(session.sent) == steps.index(failing_step) + 1


@pytest.mark.asyncio
@pytest.mark.parametrize("slot", [0, -1, 0x10000])
async def test_keycode_slot_validation(slot: int) -> None:
    lock, session = _keycode_lock([])
    for call in (
        lock.get_keycode(slot),
        lock.clear_keycode(slot),
        lock.set_keycode(slot, "1234"),
    ):
        with pytest.raises(ValueError, match="slot out of range"):
            await call
    assert session.sent == []


@pytest.mark.asyncio
@pytest.mark.parametrize("pin", ["", "12a4", "123456789012345", "12 4", "١٢٣٤"])
async def test_set_keycode_pin_validation(pin: str) -> None:
    lock, session = _keycode_lock([])
    with pytest.raises(ValueError, match="PIN must be"):
        await lock.set_keycode(1, pin)
    assert session.sent == []


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("method", "args"),
    [
        ("get_keycode", (1,)),
        ("get_parameter", (0xB2,)),
        ("set_parameter", (0xB2, 30)),
        ("get_unity_parameter", (0x01,)),
        ("set_unity_parameter", (0x0D, 0xFF)),
    ],
)
async def test_lock_commands_require_a_connection(
    method: str, args: tuple[int, ...]
) -> None:
    lock = _make_lock()
    with pytest.raises(DisconnectedError):
        await getattr(lock, method)(*args)


def test_encode_decode_keycode_pin() -> None:
    assert util.encode_keycode_pin("135790") == bytes.fromhex("135790ffffffff")
    assert util.encode_keycode_pin("12345") == bytes.fromhex("12345fffffffff")
    assert util.encode_keycode_pin("0" * 14) == bytes(7)
    assert util.decode_keycode_pin(bytes.fromhex("135790ffffffff")) == "135790"
    assert util.decode_keycode_pin(bytes.fromhex("12345fffffffff")) == "12345"
    assert util.decode_keycode_pin(b"\xff" * 7) is None
    assert util.decode_keycode_pin(bytes(7)) == "0" * 14
    for malformed in ("1a23ffffffffff", "12f4ffffffffff", "aaaaaaaaaaaaaa"):
        with pytest.raises(ValueError, match="Malformed PIN"):
            util.decode_keycode_pin(bytes.fromhex(malformed))


def test_keycode_matcher_takes_only_the_result_frame() -> None:
    matches = _keycode_response_matcher(Commands.KEYCODE_CLEAR.value)
    assert matches(_result(0x28))
    assert not matches(ACK_CLEAR)
    assert not matches(_result(0x27))
    assert not matches(_result(0x28)[:15])


def test_keycode_get_matcher_requires_the_echoed_slot() -> None:
    matches = _keycode_response_matcher(Commands.KEYCODE_GET.value, 200)
    assert matches(GET_200_PIN)
    assert matches(GET_200_EMPTY)
    assert not matches(GET_1_NOSPACE)
    assert not matches(bytes.fromhex("aa39004ec800000000000000000000000200"))


@pytest.mark.parametrize(
    "frame",
    [
        # Constructed keycode and activity log frames on either flag, and the captured
        # settings frames and Unity acknowledgments.
        *(
            bytes([flag]) + _result(opcode, 0x06, 200)[1:]
            for opcode in (0x27, 0x28, 0x2B, 0x2C, 0x39, 0x2D)
            for flag in (0xBB, 0xAA)
        ),
        SETTINGS_READ_ACK,
        SETTINGS_READ_RESPONSE,
        SETTINGS_WRITE_ACK,
        SETTINGS_WRITE_RESPONSE,
        UNITY_READ_ACK,
        UNITY_WRITE_ACK,
    ],
    ids=bytes.hex,
)
def test_no_state_frames_produce_no_state(frame: bytes) -> None:
    received: list[object] = []
    lock = _make_lock(received.append)
    assert lock._parse_state(frame) == ()
    lock._internal_state_callback(frame)
    assert received == []


# KEYPAD_UNLOCK_SLOT_* are keypad PIN unlock records (type 0x07) captured from
# a Yale YRD256; KEYPAD_LOCK_BUTTON (type 0x00) and END_OF_LOG (0x80) are
# captured from a Yale lock. UNKNOWN_ACTIVITY is built with record type 0x55.
KEYPAD_UNLOCK_SLOT_205 = bytes.fromhex("bb2d0072079aa176a333004bea14cd000200")
KEYPAD_UNLOCK_SLOT_200 = bytes.fromhex("bb2d00a4077a9e76a333004cdf14c8000200")
KEYPAD_LOCK_BUTTON = bytes.fromhex("bb2d0056000b0500a0a176a3df14194a0200")
END_OF_LOG = bytes.fromhex("bb2d00de800200009aa176a3a8a176a30200")
UNKNOWN_ACTIVITY = _with_checksum("bb2d00005500000000000000000000000000")


def test_op_code_refuses_a_value_that_is_not_an_int() -> None:
    """Only an int becomes an UNKNOWN member; anything else is a ValueError."""
    with pytest.raises(ValueError, match="not a valid OpCode"):
        OpCode("0x0D")


def test_op_code_holds_the_vendor_table() -> None:
    """OpCode holds the vendor's opCode table, with no member for 0x0D or 0x16."""
    assert {member.name: member.value for member in OpCode} == {
        "DOOR_STATE_CHANGED": 0x00,
        "DOOR_MANUAL_STATE_CHANGE": 0x01,
        "KNOCK_RECEIVED": 0x02,
        "BLUETOOTH_RECEIVED": 0x03,
        "ANOMALY": 0x04,
        "AUTO_RELOCK": 0x05,
        "AUTO_RETRY": 0x06,
        "OVER_TEMP": 0x07,
        "BURN_IN": 0x08,
        "HOME_KIT": 0x09,
        "Z_WAVE": 0x0A,
        "ONE_TOUCH": 0x0B,
        "INSIDE_BUTTON": 0x0C,
        "DOOR_SENSE": 0x0E,
        "LOW_BAT_AUTO_UNLOCK": 0x0F,
        "ESCAPE_UNLOCK": 0x10,
        "UWB": 0x11,
        "EXTERNAL_MODULE_OPERATION": 0x12,
        "SHABBAT_MODE": 0x13,
        "SCHEDULED_OPERATION": 0x14,
        "MATTER": 0x15,
        "DOUBLE_PRESS": 0x17,
        "ALIRO": 0x18,
        "UNDEFINED": 0xFF,
    }
    # A plain Enum, so DOOR_STATE_CHANGED (0) is truthy like every other member.
    assert all(OpCode)
    assert "OpCode" in yalexs_ble.__all__


def _activity_lock(frames: list[bytes | Exception]) -> tuple[Lock, AsyncMock]:
    """A connected Lock whose activity reads answer with frames in order."""
    lock, session = _connected_lock()
    answers = iter(frames)

    async def deliver(*_args: object, **_kwargs: object) -> None:
        answer = next(answers)
        if isinstance(answer, Exception):
            raise answer
        session._notify(0, bytearray(answer))

    write = AsyncMock(side_effect=deliver)
    session.client.write_gatt_char = write
    return lock, write


async def _drain(lock: Lock) -> list[bytes]:
    return [reply async for reply in lock.drain_lock_activity()]


@pytest.mark.asyncio
async def test_drain_lock_activity_yields_every_reply_through_the_get_log_error() -> (
    None
):
    """Every reply is yielded unchanged, whatever its type, the GetLogError last."""
    lock, write = _activity_lock(
        [
            UNKNOWN_ACTIVITY,
            KEYPAD_UNLOCK_SLOT_205,
            KEYPAD_LOCK_BUTTON,
            END_OF_LOG,
            KEYPAD_UNLOCK_SLOT_200,
        ]
    )
    assert await _drain(lock) == [
        UNKNOWN_ACTIVITY,
        KEYPAD_UNLOCK_SLOT_205,
        KEYPAD_LOCK_BUTTON,
        END_OF_LOG,
    ]
    assert write.await_count == 4


@pytest.mark.asyncio
async def test_drain_lock_activity_sends_the_next_get_log_when_asked() -> None:
    """A reply is yielded before the next GetLog is sent."""
    lock, write = _activity_lock([KEYPAD_UNLOCK_SLOT_205, END_OF_LOG])
    replies = lock.drain_lock_activity()
    assert await anext(replies) == KEYPAD_UNLOCK_SLOT_205
    assert write.await_count == 1
    assert await anext(replies) == END_OF_LOG
    assert write.await_count == 2
    await replies.aclose()


@pytest.mark.asyncio
async def test_drain_lock_activity_yields_each_record_as_it_is_read() -> None:
    """Records read before a failure have already been yielded."""
    lock, _ = _activity_lock([KEYPAD_UNLOCK_SLOT_205, BleakError("gone")])
    seen: list[bytes] = []
    with pytest.raises(BleakError):
        async for reply in lock.drain_lock_activity():
            seen.append(reply)
    assert seen == [KEYPAD_UNLOCK_SLOT_205]


@pytest.mark.asyncio
async def test_drain_lock_activity_raises_past_the_cap() -> None:
    """Without an end marker the drain raises after max_records reads."""
    lock, write = _activity_lock([KEYPAD_UNLOCK_SLOT_205] * (MAX_ACTIVITY_RECORDS + 5))
    seen: list[bytes] = []
    with pytest.raises(ResponseError, match=f"within {MAX_ACTIVITY_RECORDS} reads"):
        async for reply in lock.drain_lock_activity():
            seen.append(reply)
    assert len(seen) == MAX_ACTIVITY_RECORDS
    assert write.await_count == MAX_ACTIVITY_RECORDS


@pytest.mark.asyncio
async def test_drain_lock_activity_requires_a_connection() -> None:
    """Draining a disconnected lock raises."""
    lock = _make_lock()
    with pytest.raises(DisconnectedError):
        await _drain(lock)


@pytest.mark.asyncio
async def test_a_get_log_error_record_ends_the_drain() -> None:
    """A LOG_TYPE_GET_LOG_ERROR reply is the last one a drain yields."""
    assert yalexs_ble.LOG_TYPE_GET_LOG_ERROR == END_OF_LOG[0x04] == 0x80
    assert "LOG_TYPE_GET_LOG_ERROR" in yalexs_ble.__all__

    lock, write = _activity_lock([END_OF_LOG, KEYPAD_UNLOCK_SLOT_205])
    assert await _drain(lock) == [END_OF_LOG]
    assert write.await_count == 1


@pytest.mark.asyncio
async def test_get_log_sends_getlog_and_returns_the_matched_reply() -> None:
    """One GETLOG is sent and the reply its matcher takes is returned unchanged."""
    lock, session = _keycode_lock([KEYPAD_LOCK_BUTTON])
    assert await lock.get_log() is KEYPAD_LOCK_BUTTON
    assert [command[0x01] for command in session.sent] == [Commands.GETLOG.value]


@pytest.mark.asyncio
async def test_get_log_requires_a_connection() -> None:
    """Reading a record from a disconnected lock raises."""
    with pytest.raises(DisconnectedError):
        await _make_lock().get_log()


def test_ack_matcher_matches_only_the_written_operation() -> None:
    """The ack matcher keys on 0xAA + the written opcode + operation byte."""
    matches = _ack_matcher(0x0B, 0x04)

    assert matches(SECUREMODE_ACK)
    assert not matches(LOCK_ACK)  # same opcode, plain-lock operation byte
    assert not matches(UNLOCK_ACK)
    # An op-response (0xBB), not an acknowledgment.
    assert not matches(bytes.fromhex("bb0b00450400000000000000000000000200"))


def test_operation_response_matcher_matches_only_its_opcode() -> None:
    """The op-response matcher keys on 0xBB + the sent opcode, full length."""
    matches = _operation_response_matcher(0x0A)

    # An 18-byte 0xBB 0x0A op-response.
    assert matches(bytes.fromhex("bb0a00000000000000000000000000000200"))
    # Wrong opcode (0x0B).
    assert not matches(bytes.fromhex("bb0b00000000000000000000000000000200"))
    # An acknowledgment (0xAA), not an op-response.
    assert not matches(bytes.fromhex("aa0a00000000000000000000000000000200"))
    # Truncated: byte[15] (the result) is not present.
    assert not matches(bytes.fromhex("bb0a0000000000000000"))


async def _spin_until(predicate: Callable[[], bool]) -> None:
    """Yield to the event loop until predicate() holds (bounded)."""
    for _ in range(1000):
        if predicate():
            return
        await asyncio.sleep(0)
    raise AssertionError("condition was never reached")


def _make_connected_lock_with_session(
    state_callback: Callable[[Iterable[LockStateValue]], None] = lambda _: None,
) -> Lock:
    """Connected Lock over a real Session: pass-through decrypt, real encryptor."""
    lock = _make_lock(state_callback)
    client = MagicMock()
    client.is_connected = True
    client.write_gatt_char = AsyncMock()
    lock.client = client
    lock.secure_session = MagicMock()
    session = Session(
        client, "mylock", asyncio.Lock(), set(), lock._internal_state_callback
    )
    session.cipher_encrypt = Cipher(
        algorithms.AES(bytes(16)),
        modes.CBC(bytes(16)),
    ).encryptor()
    lock.session = session
    return lock


# --------------------------------------------------------------------------- #
# Mechanical operations through the staged session wait
# --------------------------------------------------------------------------- #


def _ack_frame(opcode: int, operation_byte: int) -> bytes:
    """A 0xAA acknowledgment carrying an operation's opcode and operation byte."""
    frame = bytearray(0x12)
    frame[0x00] = 0xAA
    frame[0x01] = opcode
    frame[0x04] = operation_byte
    frame[0x10] = 0x02
    return _with_checksum(frame.hex())


UNLATCH_ACK = _ack_frame(Commands.UNLOCK, UNLATCH_OPERATION_BYTE)


def _op_response_frame(opcode: int, result: int = OperationError.COMM_SUCCESS) -> bytes:
    """A 0xBB op-response carrying the operation result in byte[15]."""
    frame = bytearray(0x12)
    frame[0x00] = 0xBB
    frame[0x01] = opcode
    frame[0x0F] = result
    return _with_checksum(frame.hex())


async def _drive_operation(
    lock: Lock,
    op_attr: str,
    opcode: int,
    ack: bytes | None,
    before_ack: bytes | None = None,
    before_response: bytes | None = None,
    result: int = OperationError.COMM_SUCCESS,
    raises: type[Exception] | None = None,
) -> list[str]:
    """Run a force_* method, feeding its ack then op-response through notify.

    Returns the order of the write-success and result callbacks and the fed
    frames. Optional before_* frames must leave the stage they precede armed;
    result is the op-response's result byte. When raises is set the call must
    raise it; the events are returned either way.
    """
    session = lock.session
    assert session is not None
    events: list[str] = []

    async def feed() -> None:
        await _spin_until(lambda: session._ack_future is not None)
        if before_ack is not None:
            session._notify(0, bytearray(before_ack))
            assert session._ack_future is not None, "taken for the acknowledgment"
            await asyncio.sleep(0)
        if ack is not None:
            session._notify(0, bytearray(ack))
            assert session._ack_future is None, "the acknowledgment was not matched"
            events.append("ack")
            await asyncio.sleep(0)
        if before_response is not None:
            session._notify(0, bytearray(before_response))
            assert session._notify_future is not None, "taken for the op-response"
        session._notify(0, bytearray(_op_response_frame(opcode, result)))
        events.append("op_response")

    feeder = asyncio.create_task(feed())
    call = getattr(lock, op_attr)(
        write_success_callback=lambda: events.append("write_success"),
        result_callback=lambda succeeded: events.append(f"result {succeeded}"),
    )
    with pytest.raises(raises) if raises is not None else contextlib.nullcontext():
        await call
    await feeder
    return events


def test_parse_operation_ack_reports_no_state(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Operation acks (0xAA LOCK/UNLOCK) are recognized but carry no state."""
    states: list[list[LockStateValue]] = []
    lock = _make_lock(lambda s: states.append(list(s)))

    with caplog.at_level("INFO", logger="yalexs_ble.lock"):
        for frame in (LOCK_ACK, UNLOCK_ACK):
            result = lock._parse_state(frame)
            assert result is not None
            assert list(result) == []
            lock._internal_state_callback(frame)

    assert states == []  # the state callback was never invoked
    assert "Unknown state" not in caplog.text


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("op_attr", "opcode", "ack"),
    [
        ("force_lock", Commands.LOCK, LOCK_ACK),
        ("force_unlock", Commands.UNLOCK, UNLOCK_ACK),
        ("force_securemode", Commands.LOCK, SECUREMODE_ACK),
        ("force_unlatch", Commands.UNLOCK, UNLATCH_ACK),
    ],
    ids=["lock", "unlock", "securemode", "unlatch"],
)
async def test_force_operations_complete_on_ack_then_op_response(
    op_attr: str, opcode: int, ack: bytes
) -> None:
    """Each force_* sees write-success, matches its ack, and completes on the result."""
    lock = _make_connected_lock_with_session()
    events = await _drive_operation(lock, op_attr, opcode, ack)
    assert events == ["write_success", "ack", "result True", "op_response"]


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("result", "succeeded"),
    [(OperationError.COMM_SUCCESS, True), (OperationError.MECH_POSITION, False)],
    ids=["success", "failure"],
)
async def test_the_result_callback_is_told_whether_the_op_response_reported_success(
    result: int, succeeded: bool
) -> None:
    """The result byte of the op-response decides what the result callback is told."""
    lock = _make_connected_lock_with_session()
    events = await _drive_operation(
        lock,
        "force_lock",
        Commands.LOCK,
        LOCK_ACK,
        result=result,
        raises=None if succeeded else OperationFailedError,
    )
    assert f"result {succeeded}" in events


@pytest.mark.asyncio
async def test_an_op_response_for_another_opcode_does_not_complete_the_wait() -> None:
    """Only the op-response carrying the sent opcode completes the wait."""
    lock = _make_connected_lock_with_session()
    foreign = _op_response_frame(Commands.UNLOCK, OperationError.MECH_POSITION)
    await _drive_operation(
        lock, "force_lock", Commands.LOCK, LOCK_ACK, before_response=foreign
    )


@pytest.mark.asyncio
async def test_a_door_push_does_not_answer_the_acknowledgment_stage() -> None:
    """A door push mid-operation leaves the acknowledgment stage armed."""
    states: list[list[LockStateValue]] = []
    lock = _make_connected_lock_with_session(lambda s: states.append(list(s)))
    await _drive_operation(
        lock, "force_lock", Commands.LOCK, LOCK_ACK, before_ack=DOOR_FRAME
    )
    assert states == [[DoorStatus.CLOSED]]


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("wrapper", "force_attr", "target_status"),
    [
        ("securemode", "force_securemode", LockStatus.SECUREMODE),
        ("lock", "force_lock", LockStatus.LOCKED),
        ("unlock", "force_unlock", LockStatus.UNLOCKED),
    ],
    ids=["securemode", "lock", "unlock"],
)
@pytest.mark.parametrize("in_target_state", [False, True])
async def test_convenience_wrappers_run_the_operation_outside_the_target_state(
    wrapper: str, force_attr: str, target_status: LockStatus, in_target_state: bool
) -> None:
    """A wrapper runs its force_* unless the lock is already in the target state."""
    lock = _make_lock()
    current = target_status if in_target_state else LockStatus.UNKNOWN
    with (
        patch.object(lock, "lock_status", AsyncMock(return_value=current)),
        patch.object(lock, force_attr, AsyncMock()) as mock_force,
    ):
        await getattr(lock, wrapper)()
    assert mock_force.await_count == (0 if in_target_state else 1)


@pytest.mark.asyncio
async def test_public_unlatch_wrapper_runs_force_unlatch() -> None:
    """Lock.unlatch() always runs force_unlatch, without reading the status."""
    lock = _make_lock()
    with (
        patch.object(lock, "lock_status", AsyncMock()) as mock_status,
        patch.object(lock, "force_unlatch", AsyncMock()) as mock_force,
    ):
        await lock.unlatch()
    mock_force.assert_awaited_once()
    mock_status.assert_not_awaited()


@pytest.mark.asyncio
async def test_an_unlatch_completes_without_an_acknowledgment(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """An unlatch with no acknowledgment completes on its op-response and logs it."""
    lock = _make_connected_lock_with_session()
    with caplog.at_level("INFO", logger="yalexs_ble.session"):
        events = await _drive_operation(lock, "force_unlatch", Commands.UNLOCK, None)
    assert events == ["write_success", "result True", "op_response"]
    assert "completed on its op-response; no acknowledgment was received" in caplog.text


@pytest.mark.asyncio
async def test_force_unlatch_does_not_retry_a_raising_write() -> None:
    """A raising GATT write ends an unlatch as OperationIncompleteError."""
    lock = _make_connected_lock_with_session()
    assert lock.client is not None
    lock.client.write_gatt_char = AsyncMock(side_effect=BleakError("write failed"))

    with pytest.raises(OperationIncompleteError) as excinfo:
        await lock.force_unlatch()
    assert isinstance(excinfo.value.__cause__, BleakError)


@pytest.mark.parametrize("error", [OperationFailedError, UnityParameterError])
def test_an_error_is_reachable_from_the_package_root(error: type[Exception]) -> None:
    """The error is exported from the package root."""
    assert getattr(yalexs_ble, error.__name__) is error
    assert error.__name__ in yalexs_ble.__all__


@pytest.mark.asyncio
async def test_force_lock_failure_op_response_raises_operation_failed(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """A mechanical failure after write-success raises OperationFailedError.

    The frame logs at DEBUG and the error names the reported result.
    """
    lock = _make_connected_lock_with_session()
    session = lock.session
    assert session is not None

    async def feed() -> None:
        await _spin_until(lambda: session._ack_future is not None)
        session._notify(0, bytearray(LOCK_ACK))
        await asyncio.sleep(0)
        session._notify(
            0,
            bytearray(_op_response_frame(Commands.LOCK, OperationError.MECH_POSITION)),
        )

    feeder = asyncio.create_task(feed())
    with (
        caplog.at_level("DEBUG", logger="yalexs_ble.lock"),
        pytest.raises(OperationFailedError, match="MECH_POSITION") as excinfo,
    ):
        await lock.force_lock()
    await feeder
    assert excinfo.value.error is OperationError.MECH_POSITION
    assert _failure_log_levels(caplog) == ["DEBUG"]


@pytest.mark.asyncio
async def test_a_failure_after_our_op_response_logs_at_warning(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """A same-opcode failure handled right after our op-response logs at warning."""
    states: list[list[LockStateValue]] = []
    lock = _make_connected_lock_with_session(lambda s: states.append(list(s)))
    session = lock.session
    assert session is not None

    async def feed() -> None:
        await _spin_until(lambda: session._ack_future is not None)
        session._notify(0, bytearray(LOCK_ACK))
        await asyncio.sleep(0)
        session._notify(0, bytearray(_op_response_frame(Commands.LOCK)))
        session._notify(
            0,
            bytearray(_op_response_frame(Commands.LOCK, OperationError.MECH_POSITION)),
        )

    feeder = asyncio.create_task(feed())
    with caplog.at_level("DEBUG", logger="yalexs_ble.lock"):
        await lock.force_lock()
    await feeder
    assert _failure_log_levels(caplog) == ["WARNING"]
    assert states == [[LockStatus.JAMMED]]


@pytest.mark.asyncio
async def test_a_failure_after_an_op_response_during_the_write_logs_at_warning(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Once the op-response landed during the write, a later failure is unsolicited."""
    lock = _make_connected_lock_with_session()
    session = lock.session
    assert session is not None

    async def _write(*args: object, **kwargs: object) -> None:
        session._notify(0, bytearray(_op_response_frame(Commands.LOCK)))

    def _on_write_success() -> None:
        session._notify(
            0,
            bytearray(_op_response_frame(Commands.LOCK, OperationError.MECH_POSITION)),
        )

    assert lock.client is not None
    lock.client.write_gatt_char = AsyncMock(side_effect=_write)
    with caplog.at_level("DEBUG", logger="yalexs_ble.lock"):
        await lock.force_lock(write_success_callback=_on_write_success)
    assert _failure_log_levels(caplog) == ["WARNING"]


def test_op_response_callback_fires_on_op_responses_not_acks() -> None:
    """op_response_callback fires for success and failure op-responses, not acks."""
    op_calls: list[int] = []
    lock = _make_lock(op_response_callback=lambda: op_calls.append(1))

    lock._parse_state(LOCK_ACK)
    lock._parse_state(UNLOCK_ACK)

    assert op_calls == []

    lock._parse_state(_op_response_frame(Commands.UNLOCK))
    lock._parse_state(_op_response_frame(Commands.UNLOCK, OperationError.MECH_POSITION))

    assert op_calls == [1, 1]


def _settings_frame(opcode: int, parameter: int, value: int) -> bytes:
    """Build the 0xBB settings response with the opcode, the id, and the value."""
    frame = bytearray(RESPONSE_FRAME_LEN)
    frame[0x00] = 0xBB
    frame[0x01] = opcode
    frame[0x04] = parameter
    frame[0x08:0x0C] = value.to_bytes(4, "little")
    frame[0x03] = _simple_checksum(frame)
    return bytes(frame)


def _unity_frame(opcode: int, parameter: int, value: int) -> bytes:
    """Build the 0xBB Unity reply with entry 0 set and entries 1 to 3 zero."""
    frame = bytearray(RESPONSE_FRAME_LEN)
    frame[0x00] = 0xBB
    frame[0x01] = opcode
    frame[0x04:0x07] = bytes((parameter, value, 0))
    frame[0x03] = _simple_checksum(frame)
    return bytes(frame)


def _answered_by(session: Session, *frames: bytes) -> AsyncMock:
    """Deliver frames on the notify path when the command is written."""

    async def deliver(*_args: object, **_kwargs: object) -> None:
        for frame in frames:
            session._notify(0, bytearray(frame))

    return AsyncMock(side_effect=deliver)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("method", "args", "frames", "value", "states"),
    [
        pytest.param(
            "get_parameter",
            (0x28,),
            [SETTINGS_READ_ACK, SETTINGS_READ_RESPONSE],
            0x07080708,
            [],
            id="read-ack-then-answer",
        ),
        pytest.param(
            "set_parameter",
            (0x28, 0x005A005A),
            [
                SETTINGS_WRITE_ACK,
                _settings_frame(Commands.WRITESETTING.value, 0x28, value=5),
            ],
            5,
            [],
            id="write-ack-then-answer-reporting-another-value",
        ),
        pytest.param(
            "get_parameter",
            (0xB2,),
            [_settings_frame(Commands.READSETTING.value, 0xB2, value=30)],
            30,
            [],
            id="read-lone-answer",
        ),
        pytest.param(
            "get_parameter",
            (0xB2,),
            [
                _settings_frame(Commands.READSETTING.value, 0x84, value=4000),
                _settings_frame(Commands.READSETTING.value, 0xB2, value=30),
            ],
            30,
            [],
            id="read-answer-for-another-id-first",
        ),
        pytest.param(
            "get_parameter",
            (0xB2,),
            [
                # Status push reporting UNLOCKED.
                LOCK_FRAME,
                _settings_frame(Commands.READSETTING.value, 0xB2, value=30),
            ],
            30,
            [[LockStatus.UNLOCKED]],
            id="read-status-push-first",
        ),
        pytest.param(
            "get_parameter",
            (0xB2,),
            [
                # Failure op-response with no operation in flight.
                _op_response_frame(Commands.LOCK, OperationError.MECH_POSITION),
                _settings_frame(Commands.READSETTING.value, 0xB2, value=30),
            ],
            30,
            [[LockStatus.JAMMED]],
            id="read-op-response-first",
        ),
        pytest.param(
            "get_parameter",
            (0xB2,),
            # Captured answer to a LATCH_PULL_TIME read on a lock without unlatch.
            [bytes.fromhex("bb04008db20200005aa55aa5000000000200")],
            0xA55AA55A,
            [],
            id="read-latch-pull-time-without-unlatch",
        ),
        pytest.param(
            "get_unity_parameter",
            (0x01,),
            [_unity_frame(Commands.GETUNITY.value, 0x01, value=2)],
            2,
            [],
            id="unity-read-lone-answer",
        ),
        pytest.param(
            "get_unity_parameter",
            (0x01,),
            [UNITY_READ_ACK, _unity_frame(Commands.GETUNITY.value, 0x01, value=2)],
            2,
            [],
            id="unity-read-ack-then-answer",
        ),
        pytest.param(
            "set_unity_parameter",
            (0x01, 0x02),
            [
                UNITY_WRITE_ACK,
                _unity_frame(Commands.SETUNITY.value, 0x01, value=5),
            ],
            5,
            [],
            id="unity-write-ack-then-answer-reporting-another-value",
        ),
        pytest.param(
            "get_unity_parameter",
            (0x01,),
            [
                _unity_frame(Commands.GETUNITY.value, 0x08, value=1),
                _unity_frame(Commands.GETUNITY.value, 0x01, value=2),
            ],
            2,
            [],
            id="unity-read-answer-for-another-id-first",
        ),
        pytest.param(
            "get_unity_parameter",
            (0x01,),
            [
                _unity_frame(Commands.SETUNITY.value, 0x01, value=5),
                _unity_frame(Commands.GETUNITY.value, 0x01, value=2),
            ],
            2,
            [],
            id="unity-read-setunity-reply-first",
        ),
        pytest.param(
            "set_unity_parameter",
            (0x0D, 0xFF),
            [_unity_frame(Commands.SETUNITY.value, 0x0D, value=0xFF)],
            0xFF,
            [],
            id="unity-write-lone-answer",
        ),
    ],
)
async def test_a_parameter_command_completes_on_its_own_bb_frame(
    method: str,
    args: tuple[int, ...],
    frames: list[bytes],
    value: int,
    states: list[list[LockStateValue]],
) -> None:
    """The command returns the value field of the 0xBB frame with its opcode and id."""
    received: list[list[LockStateValue]] = []
    lock, session = _connected_lock(lambda s: received.append(list(s)))
    session.client.write_gatt_char = _answered_by(session, *frames)

    assert await getattr(lock, method)(*args) == value
    assert received == states


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("parameter", "value", "value_bytes"),
    [
        pytest.param(0xB2, 0x80000001, "01000080", id="little-endian"),
        pytest.param(0x00, 0, "00000000", id="lowest-id-and-value"),
        pytest.param(0xFF, 0xFFFFFFFF, "ffffffff", id="highest-id-and-value"),
    ],
)
async def test_set_parameter_writes_the_value_little_endian_and_returns_the_answer(
    parameter: int, value: int, value_bytes: str
) -> None:
    """The frame holds the id at byte 4 and the value little-endian at bytes 8 to 11."""
    lock, session = _connected_lock()
    session.client.write_gatt_char = _answered_by(
        session,
        _settings_frame(Commands.WRITESETTING.value, parameter, value=value),
    )

    assert await lock.set_parameter(parameter, value) == value

    (written,) = (
        call.args[1] for call in session.client.write_gatt_char.await_args_list
    )
    assert written[0x00] == 0xEE
    assert written[0x01] == Commands.WRITESETTING.value
    assert written[0x04] == parameter
    assert written[0x08:0x0C] == bytes.fromhex(value_bytes)
    assert written[0x10] == 0x02
    assert len(written) == RESPONSE_FRAME_LEN


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("method", "args", "match"),
    [
        ("get_parameter", (-1,), "not an unsigned one-byte value"),
        ("get_parameter", (0x100,), "not an unsigned one-byte value"),
        ("set_parameter", (0x100, 30), "not an unsigned one-byte value"),
        ("set_parameter", (0xB2, -1), "not an unsigned four-byte value"),
        ("set_parameter", (0xB2, 0x100000000), "not an unsigned four-byte value"),
        ("get_unity_parameter", (-1,), "Parameter id is not an unsigned one-byte"),
        ("get_unity_parameter", (0x100,), "Parameter id is not an unsigned one-byte"),
        ("set_unity_parameter", (-1, 2), "Parameter id is not an unsigned one-byte"),
        ("set_unity_parameter", (0x100, 2), "Parameter id is not an unsigned one-byte"),
        ("set_unity_parameter", (0x01, -1), "value is not an unsigned one-byte"),
        ("set_unity_parameter", (0x01, 0x100), "value is not an unsigned one-byte"),
    ],
)
async def test_parameter_commands_reject_an_argument_the_frame_cannot_hold(
    method: str, args: tuple[int, ...], match: str
) -> None:
    """An argument outside its field raises ValueError and nothing is written."""
    lock, session = _connected_lock()
    session.client.write_gatt_char = AsyncMock()

    with pytest.raises(ValueError, match=match):
        await getattr(lock, method)(*args)

    session.client.write_gatt_char.assert_not_awaited()
